"""The verdict: which category one unit's readings put it in, and why.

Separate from :mod:`jobscope.report` because judging and rendering are different
questions, and only one of them is a policy. Everything here reads ``Thresholds``
and returns a name; nothing here knows about columns, colour or terminal width. It
is also the module a site's ``[eff]`` config lands in, so keeping it free of
display concerns is what makes the config surface reviewable.

Two mechanisms, and the asymmetry between them is the design (see :func:`classify`):
votes may only raise a verdict, floors may only lower one.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from . import metrics
from .config import EDGE_KEYS, TIER_NAMES, TIERS, Thresholds

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

# --- the pre-action check: what a series' shape says --------------------------
#
# A single accurate number is still the wrong output before a scancel. Measured on one
# 4-GPU job: the newest scrape read GPU% 0 and SM_ACT% 0.0 -- below the wasteful edge on
# both, an obvious kill -- while the job was computing in bursts and had saturated all
# four cards within the previous half hour. What separates those two readings is not a
# better mean; it is `max`, and how the mean moves as the window narrows.

# Not a tier and not a shape: no measured samples at all. Reuses job_eff.NO_DATA rather
# than inventing a sixth word for the same idea.
FLAT_IDLE = "flat-idle"
BURSTY = "bursty"
DECLINING = "declining"
STEADY = "steady"


def cutoff(thresholds: Thresholds, header: str, model: str = "") -> Optional[float]:
    """The value below which ``header`` counts as idle, or None if it has no such line.

    A percentage answers to its own ``wasteful`` edge. ``POWER_W`` answers to a *floor*
    instead, per GPU model, because watts have no tier and idle draw runs from 27 W on a
    V100 to 165 W on an RTX PRO 6000 -- one number would call the second card idle at full
    load or the first busy at rest.

    A capacity reading has no idle line at all, and that is the first question asked.
    ``MEM%`` and ``GMEM%`` are percentages, so they used to answer to the wasteful edge
    like any other -- which made "this job's memory was idle for 3h" a sentence the
    ladder could print. Memory that is full while nothing computes is exactly the case
    the ``memory`` role exists to name, and it is already excluded from every verdict for
    the same reason: see :data:`jobscope.metrics.MEMORY`.
    """
    if metrics.has_role(header, metrics.MEMORY):
        return None
    floor = thresholds.floor_of(header, model)
    if floor is not None:
        return floor
    if not header.endswith("%"):
        return None
    return thresholds.edge("wasteful", header)


def below_share(samples, limit: Optional[float]) -> Optional[float]:
    """The fraction of ``samples`` under ``limit``, or None when nothing was measured.

    A fraction of *measured samples*, deliberately, not of elapsed time. An exporter that
    stopped answering leaves no samples, and dividing by the wall clock instead would
    report its silence as idleness -- the same collapse between "not measured" and "not
    working" that :mod:`jobscope.models` exists to prevent.
    """
    if limit is None:
        return None
    values = [v for v in samples if v is not None]
    if not values:
        return None
    return sum(1 for v in values if v < limit) / len(values)


def longest_idle(stamped, limit: Optional[float], step: int) -> Optional[int]:
    """The longest unbroken run under ``limit``, in seconds, from ``(stamp, value)`` pairs.

    The most decision-relevant single figure here: four minutes is a job between batches,
    three hours is a job that has stopped, and the same mean covers both.

    **A gap breaks the run rather than extending it.** Two idle samples either side of a
    missing scrape are two short runs, not one long one, because the time between them was
    not measured and claiming otherwise would turn an exporter outage into a wedged job.
    ``step`` is the sampling interval, so a stretch of *n* consecutive samples is n*step
    seconds; adjacency is judged on the stamps themselves.
    """
    if limit is None:
        return None
    ordered = sorted((s, v) for s, v in stamped if v is not None)
    if not ordered:
        return None
    best = run = 0
    previous = None
    for stamp, value in ordered:
        contiguous = previous is not None and stamp - previous <= step
        run = (run + 1) if (value < limit and contiguous) else (1 if value < limit else 0)
        best = max(best, run)
        previous = stamp
    return best * step


def _shape_rank(thresholds: Thresholds, header: str, value: Optional[float],
                limit: Optional[float]) -> Optional[int]:
    """Where ``value`` sits on whatever scale ``header`` is graded by, worst 0.

    A percentage has tiers. A metric graded by a *floor* has two states and no tiers at
    all -- ``Thresholds.tier`` returns ``""`` for a non-percentage -- and reading that
    through ``_VERDICT_ORDER.get(tier, 0)`` ranked every POWER_W reading as the worst
    tier. Every rank was then equal, which made ``flat-idle``, ``declining`` and
    ``bursty`` all unreachable: a card that never rose off its idle floor, and one that
    fell through it, both reported ``steady``.

    None where there is no scale to place the value on: an ungraded column has no shape,
    and saying ``steady`` about one is a claim rather than an absence.
    """
    if value is None:
        return None
    tier = thresholds.tier(header, value)
    if tier:
        return _VERDICT_ORDER[tier]
    if limit is None:
        return None
    return 0 if value < limit else len(TIER_NAMES) - 1


def series_shape(thresholds: Thresholds, header: str, peak: Optional[float],
                 rung_means: List[Optional[float]],
                 limit: Optional[float] = None) -> str:
    """Which of the shapes a metric's series has, over the rungs widest-first.

    Stated in tiers rather than in raw numbers so a site's own ``[thresholds]`` govern it,
    and so the answer means the same thing as the verdict elsewhere in the report.

    ``limit`` is the metric's :func:`cutoff`, which is what places a floor-graded metric
    on a scale at all -- see :func:`_shape_rank`. Passed in rather than recomputed
    because the caller has already resolved it against this unit's GPU model, and a
    second resolution here could disagree with the one the ``BELOW`` column used.

    ``declining`` outranks ``bursty`` where both hold: a job that was fine an hour ago and
    is not now is the actionable one. ``flat-idle`` is the only shape that says nothing
    ever ran, and it is the one an instant reading produces by accident.
    """
    ranks = [rank for rank in
             (_shape_rank(thresholds, header, m, limit) for m in rung_means)
             if rank is not None]
    peak_rank = _shape_rank(thresholds, header, peak, limit)
    if peak_rank is None or not ranks:
        return NO_DATA
    if peak_rank == 0:
        return FLAT_IDLE
    # Widest first, so "no better than the one before" is non-increasing, and the
    # narrowest -- now -- has to be strictly worse than the whole run. With one rung the
    # second test is x < x, so a single-rung ladder cannot be declining and needs no guard.
    if all(b <= a for a, b in zip(ranks, ranks[1:])) and ranks[-1] < ranks[0]:
        return DECLINING
    if peak_rank == len(TIER_NAMES) - 1 and ranks[0] <= _VERDICT_ORDER[TIER_NAMES[1]]:
        return BURSTY
    return STEADY


def median_swing(stamped, step: int) -> Optional[float]:
    """How far a metric moves between one scrape and the next, as a median.

    The figure that says whether a mean is reproducible. Measured on one bursty job, the
    duty cycle moved a median of 67 points between consecutive 60s scrapes and swung more
    than 40 points at 64% of them -- so the two GPU exporters, sampling the same card at
    slightly different moments, reported whole-run means of 51 and 58. Neither was wrong:
    a 60-second scrape cannot represent a signal that completes a cycle inside a minute,
    and the mean of an aliased signal is an artefact of when the sampler happened to look.

    A median rather than a mean, so one transition does not stand in for the whole run.

    **Only across adjacent samples.** A gap is not a step change: the value either side of
    a missing scrape may differ for a minute's worth of reasons, and counting that as
    movement between scrapes would report an exporter outage as a volatile workload -- the
    same rule :func:`longest_idle` follows for the same reason.
    """
    ordered = sorted((s, v) for s, v in stamped if v is not None)
    jumps = [abs(b - a) for (sa, a), (sb, b) in zip(ordered, ordered[1:])
             if sb - sa <= step]
    if not jumps:
        return None
    jumps.sort()
    middle = len(jumps) // 2
    if len(jumps) % 2:
        return jumps[middle]
    return (jumps[middle - 1] + jumps[middle]) / 2


# --- how much of it, and when -------------------------------------------------
#
# :func:`below_share` answers "what fraction", which is the honest denominator but not
# the figure someone about to act on a job asks for. These turn the same samples into
# durations and instants, under the one rule that makes them honest: a duration here is
# *measured* time, samples times the interval, and never the wall clock. What was not
# measured is carried as its own third figure rather than folded into either side --
# an exporter that stopped answering leaves no samples, and counting its silence as
# idleness is the collapse between "not measured" and "not working" that this whole
# module is arranged to prevent.


def bucket_edges(start: int, end: int, n: int) -> List[Tuple[int, int]]:
    """``n`` half-open ``[lo, hi)`` spans tiling ``[start, end]``, the last one closed.

    Returned rather than left for the caller to recompute, because a display draws the
    cells and then labels them: two hand-rolled copies of ``start + i * span // n``
    drift the moment one of them rounds the other way, and the labels then sit under
    the wrong cells while looking perfectly plausible.
    """
    n = max(1, n)
    span = max(0, end - start)
    return [(start + span * i // n, start + span * (i + 1) // n) for i in range(n)]


def bucket(pairs, start: int, end: int, n: int,
           reduce=None) -> List[Optional[float]]:
    """One value per bucket from ``(stamp, value)`` pairs, or None where none landed.

    ``None`` is a bucket nothing was measured in, and it must stay distinguishable from
    a bucket measured at zero -- the same rule :func:`longest_idle` follows, for the same
    reason. A display that renders both the same way says an exporter outage and an idle
    card are the same event.

    ``reduce`` defaults to the arithmetic mean; ``max`` gives the peak per bucket, which
    is what an aliased signal needs (see :func:`aliased`).
    """
    n = max(1, n)
    span = max(0, end - start)
    held: List[List[float]] = [[] for _ in range(n)]
    for stamp, value in pairs:
        if value is None or stamp < start or stamp > end:
            continue
        index = n - 1 if span == 0 else min(n - 1, (stamp - start) * n // span)
        held[index].append(value)
    if reduce is None:
        def reduce(values):
            return sum(values) / len(values)
    return [reduce(values) if values else None for values in held]


def full_scale(thresholds: Thresholds, header: str,
               model: str = "") -> Optional[float]:
    """The top of a *fixed* display axis for ``header``, or None if it has no scale.

    A percentage is drawn against 100, not against its own min and max. That is the
    whole difference between this and a chart: a chart prints its axis, so scaling to
    the data is informative, while a bare strip has none -- and a flat-idle series
    scaled to its own range draws as a full-height sawtooth, which is the one reading
    this view exists to make impossible.

    A metric graded by a floor has no percentage to be a fraction of, so twice the floor:
    idle draw then sits mid-axis, and the axis stays a property of the metric rather than
    of the data. None where there is neither, which is a metric that should not be drawn.
    """
    if metrics.has_role(header, metrics.MEMORY):
        return None
    if header.endswith("%"):
        return 100.0
    floor = thresholds.floor_of(header, model)
    return 2.0 * floor if floor else None


class Measured(NamedTuple):
    """Seconds of measured time split by a cutoff, and what was not measured at all."""

    idle: int
    active: int
    missing: int


def measured_time(stamped, limit: Optional[float], step: int,
                  expected: Optional[int] = None) -> Optional[Measured]:
    """``idle``/``active``/``missing`` seconds from ``(stamp, value)`` pairs.

    Samples times the interval, the arithmetic :func:`longest_idle` already ends on.
    :func:`below_share` gives the share; this gives the amount, and ``missing`` is what
    keeps the other two honest -- ``idle + active`` is the time that was *measured*, not
    the window, and the difference is a fact about the collection that belongs on screen
    rather than inside one of the first two figures.

    ``expected`` is how many scrapes the window should have held. Without it there is
    nothing to compare against and ``missing`` is zero -- which is a statement about what
    is known, not a claim that nothing is missing.
    """
    if limit is None:
        return None
    values = [v for _s, v in stamped if v is not None]
    if not values:
        return None
    idle = sum(1 for v in values if v < limit)
    missing = max(0, (expected or 0) - len(values))
    return Measured(idle * step, (len(values) - idle) * step, missing * step)


def idle_stamps(stamped, limit: Optional[float]) -> Optional[set]:
    """The instants at which this unit was measured under ``limit``."""
    if limit is None:
        return None
    return {s for s, v in stamped if v is not None and v < limit}


def concurrent_idle(stamp_sets: Sequence[Optional[set]], step: int) -> int:
    """Seconds at which *every* unit was measured idle at the same scrape.

    An intersection rather than a span, and that is the point. "How long was the job
    idle" asked as ``last_all_idle - first_all_idle`` counts an exporter outage in the
    middle as job-wide idleness; asked as the instants they were all idle *together*, a
    unit with no sample at some instant removes it rather than being assumed idle
    through it. Still measured time, and still the number the question wanted.
    """
    sets = [s for s in stamp_sets if s is not None]
    if not sets:
        return 0
    shared = set(sets[0])
    for other in sets[1:]:
        shared &= other
    return len(shared) * step


def sustain_samples(step: int) -> int:
    """How many consecutive scrapes make a state change rather than a dip.

    Five minutes, in samples, so it scales with the scrape interval. The scale is
    :func:`longest_idle`'s own -- "four minutes is a job between batches, three hours is
    a job that has stopped" -- so the shortest stretch that may be called a change has to
    clear the between-batches figure the view already prints. At two samples the rule
    fires on aliasing alone: on the measured bursty job 64% of adjacent pairs moved more
    than 40 points, so a two-sample idle run is the norm there and naming a time a
    working job "went idle" is precisely the failure this view exists to prevent.

    Never below three, so a coarsely downsampled series still needs a run rather than a
    single sample: one sample cannot establish a state.
    """
    return max(3, -(-300 // max(1, step)))


class IdleSince(NamedTuple):
    """When a unit last stopped, and how much measured idle time since."""

    at: int
    seconds: int


def went_idle(stamped, limit: Optional[float], step: int,
              sustained: Optional[int] = None) -> Optional[IdleSince]:
    """The last sustained active-to-idle transition, or None if there is not one.

    Walks back from the newest sample while the values stay under ``limit`` and the
    stamps stay adjacent -- a gap ends the run rather than extending it, as everywhere
    else here.

    None in three cases that are all "there is no transition to report", and they are
    worth keeping apart from each other by the caller: the newest samples are above the
    cutoff, so it is working now; the idle run is shorter than ``sustained``, so it is a
    dip; or nothing before it ever sustained work, which is :data:`FLAT_IDLE` -- a job
    that never ran, and one "went idle at" would describe as a job that once did.
    """
    if limit is None:
        return None
    ordered = sorted((s, v) for s, v in stamped if v is not None)
    if not ordered:
        return None
    sustained = sustain_samples(step) if sustained is None else sustained
    tail: List[int] = []
    previous = None
    for stamp, value in reversed(ordered):
        if value >= limit or (previous is not None and previous - stamp > step):
            break
        tail.append(stamp)
        previous = stamp
    if len(tail) < sustained:
        return None
    # A transition needs something to have transitioned *from*. Without this, a series
    # that was idle from its first scrape reports the first scrape as the moment it
    # stopped, which reads as a job that ran and died in a window it never worked in.
    #
    # Counted rather than consecutive, and that distinction is load-bearing: a bursty
    # job alternating 100 and 0 every scrape never has two adjacent samples above the
    # cutoff, so a run-length test finds no work to have stopped and reports a card that
    # flatlined two hours ago as still working. Cumulative also keeps a lone spike from
    # qualifying, which is the case the run-length test was reaching for.
    worked = sum(1 for stamp, value in ordered if stamp < tail[-1] and value >= limit)
    if worked < sustained:
        return None
    return IdleSince(at=tail[-1], seconds=len(tail) * step)


def distribution(values, thresholds: Thresholds, header: str,
                 model: str = "") -> List[Tuple[str, int]]:
    """``[(bin label, sample count)]`` worst first, every bin present even at zero.

    The bins are the tiers -- the same :meth:`Thresholds.tier` call the verdict is taken
    through -- so a histogram cannot disagree with the conclusion printed under it. A
    metric graded by a floor has two states and no tiers, so it gets two bins.

    Counts rather than shares or durations: the caller divides by the total for one and
    multiplies by the step for the other, and both of those conversions are established
    elsewhere. Every bin present at zero, so a block's height does not change with the
    data and two jobs can be read side by side. Empty for a metric with no cutoff, which
    has nothing for a bin to mean.
    """
    measured = [v for v in values if v is not None]
    limit = cutoff(thresholds, header, model)
    if limit is None or not measured:
        return []
    if not header.endswith("%"):
        below = sum(1 for v in measured if v < limit)
        return [("below", below), ("at or above", len(measured) - below)]
    counts = {name: 0 for name in TIER_NAMES}
    for value in measured:
        tier = thresholds.tier(header, value)
        if tier in counts:
            counts[tier] += 1
    return [(name, counts[name]) for name in TIER_NAMES]


# --- what stops the conclusion overclaiming -----------------------------------
#
# Tokens, not sentences: the numbers are policy and belong beside the [eff] table a site
# tunes, while the wording is the renderer's. The same split CATEGORIES and tint already
# use. Ordered worst-first, and that order is the precedence -- "not enough" beats "not
# measured" beats "not now" beats "not reproducible" beats what the figures say.

THIN = "thin"           # the window is too short to assert anything over -- suppresses
SPARSE = "sparse"       # too little of it was measured to describe it -- suppresses
STALE = "stale"         # measured once, but not lately -- suppresses the present tense
GAPPY = "gappy"         # measured well enough to describe, with a stated bound
ALIASED = "aliased"     # moves faster than it is sampled; no mean of it is reproducible
PEAKED = "peaked"       # a low mean, but it did run: wasteful is not the same as idle

# The three that withhold a verdict rather than qualifying one. Named as a set because
# the distinction is the whole contract of this section, and a caller that has to
# rebuild it from a tuple of string comparisons will eventually rebuild it differently.
SUPPRESSING = frozenset({THIN, SPARSE})

# A verdict is not taken over fewer than this many scrapes. 30 is the narrowest rung the
# ladder is willing to print a mean for (`30m` at a 60s scrape), so asserting on less
# evidence than the view will *display* would be incoherent. It is also where a binomial
# proportion's standard error at p=0.5 falls to 0.091 -- a +-18 point interval, just tight
# enough to separate "idle a third of the time" from "idle two thirds", which is the
# finest distinction any verdict here rests on. At N=10 that interval is +-31 points and
# the two readings are indistinguishable.
#
# Deliberately not scaled to the bucket or rung count: buckets are a display width and
# rungs are a config list, and neither may decide whether a verdict is asserted. Scaled
# to buckets it would refuse a 90-minute job, which is the most common thing anyone
# verifies before a scancel.
MIN_SAMPLES = 30

# Below this, the coverage line states the bound the gaps put on every share; below
# SPARSE_COVERAGE there is no verdict at all. 0.90 because under it the interval the
# missing scrapes allow is wider than 10 points -- wider than the whole `inefficient`
# band -- so the number cannot be printed bare. 0.50 because a verdict must describe the
# majority of the window it claims to describe; no arithmetic makes that 0.4 or 0.6.
THIN_COVERAGE = 0.90
SPARSE_COVERAGE = 0.50

# The typical step between scrapes, as a share of the whole observed range, at which no
# mean is reproducible. It has to fire on the measured case -- swing 67 over a 0-100
# range, 0.67 -- and must not fire on ordinary training with a periodic sync, which
# oscillates 20-100 in 20-point steps for 0.25. A guard that fires on healthy jobs
# teaches readers to skip guards. 0.50 is the round number between them, and it reads
# plainly: the typical step is at least half the entire range.
ALIASED_SHARE = 0.50
# Gated on a range that can span more than one band -- the `inefficient` edge. Inside a
# single band a mean is reproducible to within that band whatever the swing, so the ratio
# means nothing there (and is 0/0 on a constant series).
ALIASED_RANGE = 10.0
# median_swing already drops gap-crossing pairs; a median over fewer than ten jumps is
# not a median.
ALIASED_PAIRS = 10


def coverage(measured: int, expected: Optional[int]) -> Optional[float]:
    """The share of the window's scrapes that arrived, or None when it is not known."""
    if not expected or expected <= 0:
        return None
    return min(1.0, measured / expected)


def aliased(low: Optional[float], peak: Optional[float], swing: Optional[float],
            pairs: int) -> bool:
    """Whether ``swing`` says no mean of this series is reproducible.

    The existing legend already asserts that a swing approaching MAX-MIN means the metric
    moves faster than it is sampled. This is the number under "approaching".
    """
    if swing is None or low is None or peak is None or pairs < ALIASED_PAIRS:
        return False
    spread = peak - low
    return spread >= ALIASED_RANGE and swing / spread >= ALIASED_SHARE


def qualifiers(n: int, expected: Optional[int], low: Optional[float],
               peak: Optional[float], swing: Optional[float], pairs: int,
               limit: Optional[float], newest_age: Optional[int] = None,
               step: int = 60) -> List[str]:
    """Which robustness conditions hold for one series, worst first.

    Worst first *is* the precedence: a caller that wants the one sentence to lead with
    takes the first token, and a caller that wants every caveat takes them all.
    """
    found = []
    if expected is not None and expected < MIN_SAMPLES:
        found.append(THIN)
    share = coverage(n, expected)
    if share is not None:
        if share < SPARSE_COVERAGE:
            found.append(SPARSE)
        elif share < THIN_COVERAGE:
            found.append(GAPPY)
    # Independent of coverage, and the one the coverage test alone cannot catch: half a
    # window measured, the last samples busy, and nothing heard since. Coverage says 50%
    # and lets it through; had those samples been idle, that would have licensed "idle
    # for the last two hours" about a job nobody has heard from. Three missed scrapes
    # rather than two, since two consecutive is a routine exporter reload, with a
    # five-minute floor so a 30s series does not trip on ninety seconds of nothing.
    if newest_age is not None and newest_age > max(3 * step, 300):
        found.append(STALE)
    if aliased(low, peak, swing, pairs):
        found.append(ALIASED)
    if limit is not None and peak is not None and peak >= limit:
        found.append(PEAKED)
    return found
