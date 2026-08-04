"""The verdict: which category one unit's readings put it in, and why.

Separate from :mod:`jobscope.report` because judging and rendering are different
questions, and only one of them is a policy. Everything here reads ``Thresholds``
and returns a name; nothing here knows about columns, colour or terminal width. It
is also the module a site's ``[eff]`` config lands in, so keeping it free of
display concerns is what makes the config surface reviewable.

Two mechanisms, and the asymmetry between them is the design (see :func:`classify`):
votes may only raise a verdict, floors may only lower one.
"""

from typing import Dict, List, Optional

from . import metrics
from .config import EDGE_KEYS, TIERS, Thresholds

# --eff bands, worst first: (name, the tier whose colour paints it). Derived
# from config.TIERS rather than restated, because the order is load-bearing in three
# places that have to agree -- which bucket a cell counts in, which verdict
# classify() calls "best", and the order the categories are listed in. The colours
# come from [colors] via tint(); the numeric edges from Thresholds, per metric (see
# tier_criteria()/tier_range()).
CATEGORIES = tuple((name, name) for name, _key in TIERS)

# Not a tier: the label for a unit nothing could be measured for. It is deliberately
# outside TIERS, so it has no band, no colour role and no cutoff -- there is nothing
# to grade. It exists so that "we did not measure this" stops being reported as
# "this job wasted its allocation", which is the reading that gets someone an email.
NO_DATA = "no-data"


def classify_metrics(columns, thresholds: Optional[Thresholds] = None) -> List[str]:
    """The columns a verdict is taken over, in CSV order.

    Without a configured ``[eff] vote`` this derives the ballot: every graded
    percentage that is not a capacity reading. Memory is excluded by its catalog role
    rather than by name, so a cgroup or GPU memory metric added later cannot quietly
    start voting -- see jobscope.metrics.MEMORY.

    With one, the list narrows to it. That is what lets ``--all-metrics`` widen the
    *columns* from four metrics to fifteen without widening the ballot: a job busy on
    ENC% alone would otherwise read `good`. A configured name overrides the memory
    role too -- a site naming GMEM% is making a claim, and explicit beats inferred.
    """
    allowed = getattr(thresholds, "vote", None)
    if allowed:
        return [c for c in columns if c in allowed]
    return metrics.votable(columns)


_VERDICT_ORDER = {name: i for i, (name, _role) in enumerate(CATEGORIES)}


def classify(values: Dict[str, float], thresholds: Thresholds,
             floor_readings: Optional[Dict[str, Optional[float]]] = None,
             columns=None) -> Optional[str]:
    """The category for one unit: the band of its *best* voting metric, then lowered
    by any floor metric reading below its floor. ``None`` when nothing was judged.

    Two roles, and the asymmetry between them is the whole design:

    * ``values`` are the **votes** -- ``{header: mean}``. Best-of-N, so a vote can
      only ever *raise* a verdict. A metric may also carry a ceiling limiting how
      high it may vote (CPU%'s, see :func:`_capped_vote`).
    * ``floor_readings`` are the **floors** -- ``{header: mean}`` for metrics that
      can only *lower* one. POWER_W is the built-in case.

    A floor cannot be expressed as a vote, which is why the two exist. Under
    best-of-N a low reading is simply outvoted: a job at GPU% 48 drawing 80 W would
    read `good` on the duty cycle alone, and watts are the one signal a duty cycle
    cannot fake. Measured on 94 real GPU jobs, 8 depended on this.

    The vote is charitable on purpose -- one busy measure is enough to call a unit
    not-idle. The floors are what keep that from being naive. They are graded per GPU
    model, since idle draw runs from 27 W on a V100 to 165 W on an RTX PRO 6000, and
    a floor only ever pushes a verdict down: a unit already "wasteful" stays
    "wasteful". ``floor_readings`` of ``None`` skips them entirely.

    "Best" is the best *tier*, not the largest number: the edges are per metric, so
    a GPU% of 4 (above its 2% wasteful edge) and a CPU% of 4 (below its 5% one) are
    the same reading in different bands and magnitude no longer orders them. Where
    every metric shares its edges this is the same answer as taking the max, since
    tier() is then monotonic in the value.
    """
    # An empty ballot yields no verdict at all, rather than `wasteful`. It used to
    # default to `wasteful` on the reading "no measured metric is no evidence of
    # work" -- true of a CPU-only job in a combined sweep, whose GPU columns are
    # *not applicable*, and false of a job whose exporter was down or that predates
    # Prometheus retention, whose columns are merely *unknown*. Both arrive here as
    # an empty dict, so this cannot tell them apart and must not guess: the caller
    # knows which case it is holding, and NO_DATA is what an unknown one becomes.
    if not values:
        return None
    # A ceiling only applies while something else can carry the verdict. CPU%'s
    # exists because a busy host does not justify a *GPU* allocation -- but a
    # CPU-only job has no GPU allocation to justify, and there CPU% 50 is simply
    # good. If every voter is ceilinged, the ceilings are what would make a healthy
    # verdict unreachable, so they lift.
    ceilinged = _ceiling_applies(thresholds, values if columns is None else columns)
    verdict = max((_capped_vote(thresholds, header, value, ceilinged)
                   for header, value in values.items()),
                  key=_VERDICT_ORDER.__getitem__)
    if _below_a_floor(thresholds, floor_readings):
        return _lowered_to(verdict, "inefficient")
    return verdict


def unceilinged(thresholds: Thresholds, headers) -> List[str]:
    """The metrics among ``headers`` that may vote a unit healthy.

    The distinction a caller needs twice. Ceilings must be judged against the
    *columns a series carries*, not against one unit's readings: those are the same
    "not applicable" versus "not measured" split the Measure states draw. A ``--cpu``
    series carries no GPU column at all, so CPU% is all there is and votes freely. A
    combined series that carries GPU% but has no value for *this* unit has a gap, and
    a busy host must not fill it -- that unit is no-data.
    """
    return [h for h in headers if thresholds.vote_ceiling(h) is None]


def _ceiling_applies(thresholds: Thresholds, headers) -> bool:
    """Whether vote ceilings bind, given the columns available."""
    return bool(unceilinged(thresholds, headers))


def _capped_vote(thresholds: Thresholds, header: str, value: float,
                 ceilinged: bool = True) -> str:
    """The tier ``header`` votes for, no better than its ceiling.

    A metric may band higher than it is allowed to *vote* -- CPU% at 50 is genuinely
    half-used and paints green, but on a GPU job a busy host is not evidence the
    cards were needed, so it votes no higher than `inefficient`. See
    config.DEFAULT_VOTE_CEILING.
    """
    voted = thresholds.tier(header, value)
    ceiling = thresholds.vote_ceiling(header) if ceilinged else None
    return _lowered_to(voted, ceiling) if ceiling else voted


def _lowered_to(verdict: str, limit: str) -> str:
    """``verdict``, or ``limit`` if that is worse. Never raises a verdict."""
    if verdict not in _VERDICT_ORDER or limit not in _VERDICT_ORDER:
        return verdict
    return CATEGORIES[min(_VERDICT_ORDER[verdict], _VERDICT_ORDER[limit])][0]


def _below_a_floor(thresholds: Thresholds, readings) -> bool:
    """Whether any floor metric read below its floor.

    ``readings`` is ``{header: measured value}``; the floors come from ``thresholds``,
    already resolved for this unit's hardware by ``for_model``. Any one of them being
    low is enough -- a floor asserts "below this, the resource is idle", and two such
    assertions do not cancel out.
    """
    for header, value in (readings or {}).items():
        if value is None:
            continue
        floor = thresholds.floor_of(header)
        if floor is not None and value < floor:
            return True
    return False


def classify_description(voted: List[str], floors: List[str],
                         thresholds: Thresholds) -> str:
    """What a verdict was judged from: which metrics voted, and which could lower it.

    Shared by ``timeseries_eff()`` and the single-job summary's Efficiency line, so
    the two describe the same rule in the same words. The ceiling is named
    where it applies, because "CPU% voted" and "CPU% voted but could not call this
    healthy" are different claims and the second is the one that is true.

    Ceilings come from ``thresholds``, not from config.DEFAULT_VOTE_CEILING, so a
    site that set ``[eff.ceiling]`` is told the rule its own verdicts were
    reached by. Reading the built-in table here would have described a rule the run
    did not use, which is the one thing this line exists to prevent.
    """
    if not voted:
        return "on nothing -- no metric was measured"
    # Only name a ceiling that actually binds; see classify()'s _ceiling_applies.
    unlimited = unceilinged(thresholds, voted)
    limited = [h for h in voted if h not in unlimited] if unlimited else []
    text = "by best of %s" % ", ".join(voted)
    extra = []
    for header in limited:
        extra.append("%s can vote no higher than %s"
                     % (header, thresholds.vote_ceiling(header)))
    if floors:
        extra.append("%s lowers it below the floor" % ", ".join(sorted(floors)))
    if extra:
        text += " (%s)" % "; ".join(extra)
    return text


_WORST_TIERS = ("wasteful",)


def _metric_range(name: str, thresholds: Thresholds, header: str) -> str:
    """One metric's own span in tier ``name``, e.g. ``"20-40%"`` or ``">40%"``."""
    if name in _WORST_TIERS:
        return "<%g%%" % thresholds.edge("wasteful", header)
    below = dict(TIERS)[name]
    above = EDGE_KEYS[EDGE_KEYS.index(below) - 1] if below else None
    lo = thresholds.edge(above, header) if above else None
    return ("%g-%g%%" % (lo, thresholds.edge(below, header)) if below
            else ">%g%%" % thresholds.edge(EDGE_KEYS[-1], header))


def tier_range(name: str, thresholds: Thresholds, metrics: List[str]) -> str:
    """The numeric range for tier ``name`` over ``metrics``.

    ``"20-40%"`` while they agree on *this* tier -- which for a site that has tuned
    nothing is always, and for one that tuned only the wasteful edge is still every
    tier above it -- and ``"GPU% 20-40%, CPU% 40-60%"` once they do not, because
    then there is no one range and quoting a single pair would be a lie. Judged per
    tier rather than on the whole edge vector so a metric that differs lower down
    does not make every heading above it longer for nothing.
    """
    metrics = list(metrics) or [""]
    spans = [_metric_range(name, thresholds, m) for m in metrics]
    if len(set(spans)) == 1:
        return spans[0]
    return ", ".join("%s %s" % (m, span) for m, span in zip(metrics, spans))


def _tier_agrees(name: str, thresholds: Thresholds, metrics: List[str]) -> bool:
    """Whether every one of ``metrics`` has the same span in tier ``name``."""
    return len({_metric_range(name, thresholds, m) for m in metrics or [""]}) == 1


def tier_criteria(name: str, metrics: List[str], thresholds: Thresholds) -> str:
    """The rule that put a unit in category ``name``, spelled out with its own
    metrics -- e.g. ``"best of GPU%, SM_ACT%: 2-10%"`` -- so a --eff heading
    is self-explanatory without a separate legend lookup. Numbers come from
    ``thresholds``, the same table the run graded against, so the heading can never
    quote a cutoff the verdict did not use.

    The worst tier is the one band where *every* metric, not just the best one,
    must clear the cutoff (the max is under it only if all of them are), so they
    are listed one by one with their own cutoff instead of a shared range.
    """
    if name in _WORST_TIERS:
        return ", ".join("%s <%g%%" % (m, thresholds.edge("wasteful", m))
                         for m in metrics)
    if _tier_agrees(name, thresholds, metrics):
        # One range for all of them, so name the metrics once and the range once.
        return "best of %s: %s" % (", ".join(metrics),
                                   tier_range(name, thresholds, metrics))
    # Each carries its own range, which already names it -- saying the metrics
    # twice would be the only thing longer than saying them once.
    return "best of %s" % tier_range(name, thresholds, metrics)