"""Configuration loading for jobscope.

Site-specific settings (the Prometheus endpoint above all) come from a TOML
file, an environment variable, or, for backward compatibility, an existing
jobstats ``config`` module. Nothing here ever prints the resolved Prometheus
URL, which commonly embeds a credential.
"""

import importlib
import importlib.resources
import importlib.util
import os
import re
import shutil
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from .errors import JobscopeError

try:
    import tomllib as _toml
except ModuleNotFoundError:
    import tomli as _toml

CONFIG_ENV = "JOBSCOPE_CONFIG"
PROM_URL_ENV = "JOBSCOPE_PROM_URL"

DEFAULT_SAMPLING_PERIOD = 60
DEFAULT_WORKERS = 8
DEFAULT_TIMEOUT = 60.0
# Runtime floor for the running view: jobs younger than this are hidden, since a
# job still ramping up reads as idle. A duration string, as --min-elapsed takes.
DEFAULT_MIN_ELAPSED = "10m"

# Pacing for the query fan-out, rather than a cap on it. The running view averages each
# job over its own runtime, and PromQL cannot vary a window per series, so that is one
# query per (job, metric) where the newest scrape is one per metric for the whole
# selection -- measured on one partition, 12 queries and 0.6s of server time became 628
# and 33.9s at 110 jobs. A cap denied the better number to exactly the selections that
# most need it; this spreads the same queries out instead, and the rows stream as they
# arrive so the wait is visible rather than blank.
#
# The burst is what keeps this invisible for ordinary use: an explicit job ID is three
# queries, --verify a handful, a small partition under a hundred -- none of them reach the
# steady rate. 0 for the rate disables pacing entirely.
DEFAULT_MAX_QUERIES_PER_SECOND = 50.0
DEFAULT_QUERY_BURST = 200

# A guard against a typo'd sweep, not a cost policy: an unfiltered cluster-wide running
# selection is thousands of jobs, and the pacing above would work through it for minutes.
DEFAULT_MAX_RUNNING_JOBS = 2000

# --verify's rungs below its collected window, widest first. Two, because the ladder has
# to show a direction and three columns of numbers is already a lot to read; and these two
# because an hour is the span a stalled job becomes worth acting on and half an hour is the
# shortest window with enough samples at a 60s scrape to mean anything. Both sit inside
# DEFAULT_VERDICT_WINDOW, which is what makes them rungs *below* it rather than a
# second, wider question.
DEFAULT_VERIFY_WINDOWS: Tuple[Tuple[str, int], ...] = (("2h", 7200), ("30m", 1800))
# How far back --eff and --verify look when neither was given a window of its own.
#
# Those two ask a question about *now* -- is this job wasting its allocation, and is it
# still doing so -- and a mean over a job's whole runtime answers a different one: a job
# that ran well for two days and stalled an hour ago still averages well. Three hours is
# long enough to be about the job rather than about a checkpoint pause, and short enough
# that a verdict is about the present.
#
# Deliberately not applied to a bare --ts or --plot-ts. Those dump or chart a series
# rather than judging one, and truncating them to three hours would silently change what
# `jobscope plot` is given.
DEFAULT_VERDICT_WINDOW = "180m"
# The `finished` window when no -D/-N/-S/-E is given.
DEFAULT_DAYS = 1
# Which job endings `finished` reports without -t. See slurm.STATE_GROUPS.
DEFAULT_STATE = "completed"
# How many jobs each Problem-jobs "Wasteful" row lists. Three because the row is a
# lead, not a census -- the counts beside its heading say how many there were.
DEFAULT_WORST_JOBS = 3
# The Wasteful-row highlight threshold, as a duration string.
DEFAULT_LONG_RUNNING = "3h"
# The efficiency tiers, worst first: (name, the config key naming its upper edge).
# Four edges make five tiers -- "good" has none, being everything above `average`.
# One ordered definition rather than several, because three things depend on this
# order agreeing: which bucket a cell falls in, which verdict classify() calls
# "best", and the order --eff lists its categories in. It lives here because
# report imports config, never the reverse.
TIERS = (
    ("wasteful", "wasteful"),
    ("inefficient", "inefficient"),
    ("needs improvement", "improvement"),
    ("average", "average"),
    ("good", None),
)
TIER_NAMES = tuple(name for name, _key in TIERS)
# The four edge keys, in tier order -- the names a [thresholds.<view>.<key>] table
# takes, and the order they must be non-decreasing in. Also the names [colors]
# takes, plus "good", so the two sections line up.
EDGE_KEYS = tuple(key for _name, key in TIERS if key)

# The five tiers collapse to three buckets for the summary table's RED/YELLOW/GREEN
# columns: wasteful and inefficient both count as red. These three names are
# *identifiers*, not colours -- they are the keys of EfficiencyTally.bands and the
# `red=`/`yellow=`/`green=` fields of the --csv output, so they are fixed whatever
# palette a site chooses. See [colors] and Palette for the colours themselves.
BUCKET_OF = {"wasteful": "red", "inefficient": "red", "needs improvement": "yellow",
             "average": "green", "good": "green"}
BUCKETS = ("red", "yellow", "green")
# Which tier's colour stands in for a bucket, where a bucket has to be painted (the
# three band columns): the least-bad tier it covers, so "red" reads as inefficient
# rather than as the more alarming wasteful.
BUCKET_TIER = {"red": "inefficient", "yellow": "needs improvement", "green": "good"}

# Applied to any %-metric a config does not name explicitly, which the open-ended
# DCGM catalog (18 %-columns under --dcgm) needs. A site tunes the metrics it cares
# about and lets the rest fall here.
DEFAULT_BANDS = {"wasteful": 2.0, "inefficient": 10.0, "improvement": 20.0,
                 "average": 40.0}

# The best tier a metric is allowed to *vote* for, where that differs from the tier
# its value bands into.
#
# CPU% is the case, and the distinction matters. Its **banding** is the ordinary
# utilization ladder: a job at CPU% 50 is using half its cores and the summary table
# should paint that green, because it is. Its **vote** is capped at `inefficient`,
# because a busy host is not evidence that a GPU allocation was justified -- a job
# saturating its cores while holding four idle cards is still wasting the cards.
#
# Two different questions about one number, so two different answers. Conflating them
# by giving CPU% upper edges of 100 would turn every ordinary CPU reading red in the
# tables and the charts.
#
# Built in rather than left to config: omitting it silently restores the bug this
# replaces. Measured on 94 real GPU jobs, 4 read `good` on a busy host while using
# 0% of their GPUs.
DEFAULT_VOTE_CEILING = {"CPU%": "inefficient"}

# Per-metric edge defaults, where the catalog-wide ladder is calibrated for the wrong
# quantity. Only the edges that genuinely differ -- these *do* affect banding and
# colour, unlike the vote ceiling above.
#
# CPU%'s wasteful edge is 5 rather than 2: a GPU job legitimately holds cores it never
# uses, so the bar for calling its host idle is higher than for a GPU metric. Its
# upper edges stay on the ladder, because half a job's cores in use is half in use.
DEFAULT_BY_METRIC = {"CPU%": {"wasteful": 5.0}}
# POWER_W is watts, not percent, so it is not tiered at all -- it is a floor, and a
# GPU below it is idle. That is the one signal a duty cycle cannot fake: a job
# spinning on a trivial kernel reads busy on GPU% and draws idle watts. 100 sits in
# the measured gap -- on kempner_eng the idle jobs drew 73-74 W with GPU% 0 and
# SM_ACT% 0.0, the next values were 99-101 W, and the median was 289 W against a
# 573 W maximum.
DEFAULT_POWER_W = 100.0
# The one metric that ships with a floor. Named because several places have to agree
# on it and it is not a percentage, so it never appears in an edge table.
POWER_HEADER = "POWER_W"

# The two view-scoped band tables. Each is defined in full or not at all: a config
# naming only one leaves the other on DEFAULT_BANDS rather than copying across, so
# neither view can silently inherit numbers that were calibrated for the other.
BAND_VIEWS = ("summary", "timeslice")

# Superseded by the [thresholds.<view>.<edge>] tables. Named so a config that still
# sets them gets told where they went, rather than being silently ignored. The
# metric names among them (gpu/gmem/mem/cpu) now mean something again one level
# down -- as keys *inside* an edge table -- which the note below says.
LEGACY_THRESHOLD_KEYS = ("gpu", "gmem", "mem", "default", "red", "cpu",
                         "wasteful", "inefficient", "improvement", "average")

# Renamed keys in [prometheus], as {old: new}. The value is *not* honoured -- this only
# exists so the rename cannot happen in silence. [prometheus] takes its keys with plain
# .get(), so an old name is not rejected the way [eff] or [site] would reject it; it is
# read by nobody and the section still resolves, which is the worst of both. Worse here
# than elsewhere because the fallback hides it: a site that dropped this key still gets
# an endpoint from `which("jobstats")`, so the only symptom is a *quietly different*
# server, at whichever site pointed the key somewhere non-default.
RENAMED_PROMETHEUS_KEYS = {"site_jobstats_config_path": "site_prom_config_path"}

# Every top-level name load_config consumes. Its purpose is the inverse of the tables
# above: those name what *was* legal, this names what is, so that anything else can be
# reported instead of ignored. Every inner table already rejects a name it does not know
# ([eff] and [report] and [plot] all raise); the top level was the one surface where a
# typo'd header meant the whole section silently did nothing.
KNOWN_SECTIONS = frozenset({
    "prometheus", "thresholds", "defaults", "metrics", "gpu", "host", "eff",
    "colors", "site", "report", "plot",
})

# The colour each tier is painted, and the one role that is not a tier: an entry on
# a Wasteful row whose job ran longer than [defaults] long_running. Two tiers
# sharing a colour is the default, not a requirement -- a site wanting five distinct
# colours (for a colourblind-safe palette, say) can set five.
#
# long_running is a *brighter* red rather than plain red because the Problem-jobs
# entries around it are now painted `wasteful` too. It used to be the only colour in
# that section, which made a section full of wasteful jobs read as ungraded unless one
# of them happened to have run for over three hours.
DEFAULT_COLORS = {"wasteful": "red", "inefficient": "red", "improvement": "yellow",
                  "average": "green", "good": "green", "long_running": "bright_red"}
COLOR_ROLES = tuple(DEFAULT_COLORS)
# The eight ANSI colours and their bright variants, by SGR foreground code. `rich`
# accepts these names too, which is what lets one config value drive both the
# report's raw escapes and the chart's styles. `color(N)` for the 256-colour cube is
# accepted as well -- see Palette.sgr.
_SGR_NAMED = {"black": 30, "red": 31, "green": 32, "yellow": 33, "blue": 34,
              "magenta": 35, "cyan": 36, "white": 37,
              "bright_black": 90, "bright_red": 91, "bright_green": 92,
              "bright_yellow": 93, "bright_blue": 94, "bright_magenta": 95,
              "bright_cyan": 96, "bright_white": 97}
_COLOR_N_RE = re.compile(r"^color\((\d{1,3})\)$")


def _sgr(colour: str) -> str:
    """The ANSI escape that sets ``colour``, or ``""`` if it names none.

    Empty rather than raising: an unknown colour should leave text unpainted, not
    kill a report. load_config validates the config's own values up front, so the
    only way to reach this with a bad name is a library caller.
    """
    code = _SGR_NAMED.get(colour)
    if code is not None:
        return "\033[%dm" % code
    cube = _COLOR_N_RE.match(colour or "")
    return "\033[38;5;%dm" % int(cube.group(1)) if cube else ""


def valid_colour(colour: str) -> bool:
    """Whether ``colour`` is a name both the report and the charts understand."""
    return colour in _SGR_NAMED or bool(_COLOR_N_RE.match(colour or ""))


@dataclass(frozen=True)
class Palette:
    """What colour each classified tier is painted.

    Roles are the four edge keys plus ``good`` -- the same names ``[thresholds]``
    uses, so the two sections read alike -- plus ``long_running``. Kept apart from
    :class:`Thresholds` because the two answer different questions: thresholds decide
    *which* tier a job is in, this decides only what that looks like, and a site
    should be able to change one without touching the other.
    """

    colors: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_COLORS))

    def __post_init__(self) -> None:
        if set(self.colors) != set(DEFAULT_COLORS):
            object.__setattr__(self, "colors", {**DEFAULT_COLORS, **self.colors})

    def for_tier(self, tier: str) -> str:
        """``tier``'s colour name. Takes a tier name or an edge key, since
        "needs improvement" is spelled `improvement` in the config."""
        return self.colors.get(dict(TIERS).get(tier, tier) or tier, "")

    def for_bucket(self, bucket: str) -> str:
        """A band column's colour: the least-bad tier that bucket covers."""
        return self.for_tier(BUCKET_TIER.get(bucket, bucket))

    def sgr(self) -> Mapping[str, str]:
        """``{role: escape}`` for every name a renderer may ask to paint by.

        Tier names, edge keys, bucket identifiers and ``long_running`` all resolve,
        because the call sites hold different ones: cell tinting has a bucket, a
        --eff heading has a tier, the Wasteful rows have neither.
        """
        table = {role: _sgr(colour) for role, colour in self.colors.items()}
        for tier, key in TIERS:
            table[tier] = _sgr(self.for_tier(tier))
            if key:
                table[key] = table[tier]
        for bucket in BUCKETS:
            table[bucket] = _sgr(self.for_bucket(bucket))
        return table


def parse_duration(text: str) -> int:
    """Seconds from a compact duration such as ``30s``, ``5m``, ``2h``, ``7d``.

    Here rather than in running.py, which is where it started: config needs it too,
    for the duration-valued ``[defaults]`` keys, and config cannot import it.
    """
    match = re.match(r"^(\d+)([smhd])$", str(text).strip())
    if not match:
        raise JobscopeError(
            "invalid duration %r: use a count and a unit, e.g. '30s', '5m', '2h', '7d'" % text)
    value, unit = int(match.group(1)), match.group(2)
    return value * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def metric_header(key: str) -> str:
    """Config spelling of a metric -> its column header.

    ``gpu``, ``GPU`` and ``"GPU%"`` all mean ``GPU%``; ``sm_act`` means ``SM_ACT%``;
    ``power`` means ``POWER_W``. Sites write the short lowercase form, which is how
    the metrics get talked about, and the tables are keyed on the header the rest of
    the tool uses.

    Resolved through the catalogs, not guessed. Appending ``%`` to an upper-cased key
    is right for a percentage and wrong for everything else -- ``power`` became
    ``POWER%``, a column nothing answers to, which is exactly the silent-drop the
    stray-key note was invented to catch. The guess survives only as the fallback for
    a name no catalog knows, so that note still fires on a genuine typo.
    """
    text = str(key).strip()
    # A family-qualified name resolves to the same header as the bare one, so a site
    # may write either -- `dcgm-sm_act` where it wants to be explicit, `sm_act` where
    # the short name is unambiguous.
    family, _, rest = text.partition("-")
    if rest and family.lower() in ("dcgm", "nvml", "cgroup"):
        text = rest
    for spec in _lookup(text):
        return spec.header
    header = text.upper()
    return header if header.endswith("%") else header + "%"


def _lookup(name: str):
    """The catalog spec ``name`` refers to, as a 0-or-1 iterable.

    **Host catalog first, and that ordering is load-bearing.** ``mem`` is a key in
    both: the cgroup one is host RSS (``MEM%``), the GPU one is a card's memory
    (``GMEM_GB``). A site writing ``mem = 3`` under ``[thresholds]`` means the MEM%
    column it can see in the summary table, so the host wins; GPU memory is ``gmem``.
    Resolving GPU-first silently re-pointed such a config at a different quantity.

    Deferred imports for the usual reason -- at module scope either would close the
    cycle config -> dcgm/cpu -> prometheus -> config.
    """
    from .cpu import spec_named as cgroup_named
    from .dcgm import spec_named as gpu_named
    found = cgroup_named(name) or gpu_named(name)
    return (found,) if found is not None else ()


@dataclass(frozen=True)
class Thresholds:
    """One view's grading edges: a five-way percent split, per metric.

    Per metric because the metrics do not mean the same thing: a GPU job
    legitimately holds cores it does not use, so CPU% at 4% is ordinary where GPU%
    at 4% is idle, and SM_ACT% sits structurally below GPU% on the same work. One
    uniform cutoff had to be wrong for some row of the table.

    Two of these exist per run -- one for the whole-job summary, one for the ``--ts``
    time slice (see :data:`BAND_VIEWS`) -- because a two-hour window that catches a
    checkpoint pause should not have to answer to a nineteen-hour job's bar.
    """

    # Per-edge fallback for any metric ``by_metric`` does not name.
    defaults: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_BANDS))
    # ``{header: {edge key: value}}`` for the metrics a site named explicitly. A
    # metric may name only some of its four edges; the rest fall to ``defaults``.
    by_metric: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    # ``{header: {"": global floor, "<gpu model>": that card's floor}}``.
    #
    # A *floor* metric is one that can only pull a verdict **down**, never up, which
    # is a different thing from a tiered percentage and cannot be expressed as one:
    # under best-of-N voting a metric can only ever raise a verdict. POWER_W is the
    # built-in case -- a job spinning on a trivial kernel reads busy on GPU% and
    # draws idle watts, and watts are the one signal a duty cycle cannot fake.
    #
    # Per-model because idle draw is hardware, not policy: measured on one cluster it
    # ran from 27 W on a V100 to 165 W on an RTX PRO 6000, so a single number is
    # wrong at one end or the other. Keyed on the exporter's exact model string,
    # which is what is available where the grading happens; normalising model names
    # would be a second thing to get wrong.
    floors: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {POWER_HEADER: {"": DEFAULT_POWER_W}})
    # Which metrics may vote at all. ``None`` means derive it -- every graded
    # percentage that is not a capacity reading -- which is what keeps a site that
    # has configured nothing following the catalog as it grows. A list narrows it,
    # and is how ``--dcgm`` can widen the *columns* without widening the ballot from
    # four metrics to fifteen.
    vote: Optional[Tuple[str, ...]] = None
    # ``{header: best tier it may vote for}``. See :data:`DEFAULT_VOTE_CEILING`.
    ceilings: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_VOTE_CEILING))

    def __post_init__(self) -> None:
        # Fill any edge the caller left out, so a partial ``defaults`` -- a TOML that
        # names only `wasteful`, or a caller passing one edge -- cannot KeyError deep
        # inside edge(). object.__setattr__ because the dataclass is frozen.
        if set(self.defaults) != set(DEFAULT_BANDS):
            object.__setattr__(self, "defaults", {**DEFAULT_BANDS, **self.defaults})
        object.__setattr__(self, "by_metric", self._with_calibrations())

    def _with_calibrations(self) -> Mapping[str, Mapping[str, float]]:
        """``by_metric`` plus jobscope's own per-metric edges, where they still apply.

        Applied here rather than in the loader so a directly-constructed Thresholds --
        a test, a library caller -- gets them too; leaving them to ``load_config``
        meant CPU%'s cutoff quietly reverted to the generic ladder for everyone who
        did not read a config file.

        Skipped for any edge the caller **retuned**: a site writing ``default = 3``
        means "3 for everything I did not name", and our opinion silently overriding
        an explicit instruction is the kind of thing nobody can debug. Detected by
        comparing against DEFAULT_BANDS, which is the only signal available here.
        """
        merged = {header: dict(edges) for header, edges in self.by_metric.items()}
        for header, edges in DEFAULT_BY_METRIC.items():
            for key, value in edges.items():
                if self.defaults.get(key) != DEFAULT_BANDS.get(key):
                    continue          # the site moved this edge; respect it
                if key in merged.get(header, {}):
                    continue          # the site named this metric's edge
                merged.setdefault(header, {})[key] = value
        return merged

    def floor_of(self, header: str, model: Optional[str] = None) -> Optional[float]:
        """``header``'s floor for ``model``, or None if it has no floor at all.

        None rather than zero: "this metric does not cap anything" and "this metric
        caps at 0 W" are different claims, and a caller that treats a missing floor
        as 0 would silently stop capping.
        """
        table = self.floors.get(header)
        if not table:
            return None
        if model and model in table:
            return float(table[model])
        return float(table[""]) if "" in table else None

    def floor_for(self, model: Optional[str] = None) -> float:
        """POWER_W's floor for ``model``. The common case, kept as its own name."""
        found = self.floor_of(POWER_HEADER, model)
        return DEFAULT_POWER_W if found is None else found

    @property
    def power_w(self) -> float:
        """POWER_W's global floor -- the pre-``floors`` spelling, still read widely."""
        return self.floor_for(None)

    @property
    def power_w_by_model(self) -> Mapping[str, float]:
        """POWER_W's per-model floors, without the global entry."""
        return {model: value
                for model, value in (self.floors.get(POWER_HEADER) or {}).items()
                if model}

    def for_model(self, model: Optional[str]) -> "Thresholds":
        """These thresholds with every floor resolved for one card.

        Binding the model once beats handing it to every grading call. The floor was
        previously an optional argument on four separate methods, and the sites that
        forgot it -- the printed cell, and the waste ledger behind the Worst rows --
        graded the same reading against the global floor while the tally used the
        card's. One 165 W sample came out red in the table and green in the cell.
        Resolved here, a caller cannot forget what it never passes.
        """
        resolved = {}
        changed = False
        for header in self.floors:
            here = self.floor_of(header, model)
            if here is None:
                continue
            resolved[header] = {"": here}
            changed = changed or here != self.floor_of(header, None)
        return replace(self, floors=resolved) if changed else self

    def edge(self, key: str, header: str = "") -> float:
        """``header``'s value for edge ``key``: its own if set, else the default.

        ``by_metric`` already carries jobscope's per-metric calibrations where a
        site left room for them -- see the merge in :func:`_band_table`, which is
        where "did the site say something that covers this" is known.
        """
        return self.by_metric.get(header, {}).get(key, self.defaults[key])

    def vote_ceiling(self, header: str) -> Optional[str]:
        """The best tier ``header`` may vote for, or None for no limit.

        Separate from :meth:`edge` because it answers a different question -- see
        :data:`DEFAULT_VOTE_CEILING`. CPU% bands and colours on the ordinary ladder
        while being unable to vote a job healthy.
        """
        return self.ceilings.get(header)

    def edges(self, header: str = "") -> Tuple[float, ...]:
        """``header``'s four edges in tier order -- what validation checks."""
        return tuple(self.edge(key, header) for key in EDGE_KEYS)

    def uniform(self, headers) -> bool:
        """Whether every one of ``headers`` is graded by the same four edges.

        The output stays in its short form while this holds, which for a site that
        has tuned nothing is always -- so naming each metric's own cutoffs is a cost
        paid only by the configs that made them differ.
        """
        seen = {self.edges(h) for h in headers}
        return len(seen) <= 1

    def tier(self, header: str, value: Optional[float]) -> str:
        """The five-way band ``value`` falls in for ``header``, or ``""`` if ungraded.

        Edges are inclusive on their upper end except ``wasteful``, which is a strict
        less-than: at the default edges 1.9 is wasteful, 2.0 and 10.0 are inefficient,
        10.1 needs improvement. Ungraded covers a missing reading and any column that
        is not a percentage -- POWER_W is watts against a floor, see :meth:`grade`.
        """
        if value is None or not str(header).endswith("%"):
            return ""
        if value < self.edge("wasteful", header):
            return "wasteful"
        for name, key in TIERS[1:]:
            if key is None or value <= self.edge(key, header):
                return name
        return "good"

    def grade(self, header: str, value: Optional[float]) -> str:
        """The bucket ``value`` counts in: ``red``/``yellow``/``green``, or ``""``
        when ``header`` is not graded.

        The single entry point for banding, shared by the tables and the charts, so a
        job bucketed red in `jobscope plot` is red in the report too and a site that
        retunes ``[thresholds]`` moves both at once. POWER_W is graded against an
        absolute floor rather than the five-way percent split, so it has its own,
        two-band rule.

        Returns a bucket *identifier*, not a colour -- see :data:`BUCKET_OF`. What
        the bucket looks like is :class:`Palette`'s business, and the identifiers
        also name columns in the ``--csv`` output, so they do not move.
        """
        if value is None:
            return ""
        if header == "POWER_W":
            return floor_band(value, self.power_w)
        return BUCKET_OF.get(self.tier(header, value), "")


def _eff(table: Mapping, power_w: float,
         by_model: Mapping[str, float]) -> Tuple:
    """``[eff]`` -> ``(vote, floors, ceilings)`` for the band tables.

    Three keys, and the difference between the first two is the whole model:

    * ``vote`` -- a list. Best-of-N, so a metric here can only ever *raise* a
      verdict. Omitted means "derive it" -- every graded percentage that is not a
      capacity reading -- which is what lets the catalog grow without editing config.
    * ``floor`` -- a table per metric, ``[eff.floor.<metric>]``. A floor can
      only *lower* a verdict, and that is why it is not a vote: under best-of-N a low
      reading is simply outvoted. Omitted keeps the built-in POWER_W floor; written
      as a bare ``[eff.floor]`` with nothing under it means no floors at all.
    * ``ceiling`` -- how high a metric may vote, without stopping it voting.

    ``floor`` is a table rather than a list-plus-values because TOML forbids a key
    being both, and one concept beats two near-identical names.

    ``[thresholds] power_w`` still feeds POWER_W's floor when ``[eff.floor]``
    says nothing, so an existing config keeps working.
    """
    known = ("vote", "floor", "ceiling")
    unknown = sorted(set(table) - set(known))
    if unknown:
        raise JobscopeError("[eff] has no %s; it takes %s"
                            % (", ".join(repr(k) for k in unknown), ", ".join(known)))

    vote = _metric_list("[eff] vote", table.get("vote"))
    if vote is not None and not vote:
        raise JobscopeError(
            "[eff] vote is empty, which leaves nothing to judge a job by. Omit "
            "it to use every graded percentage, or name at least one metric.")
    resolved = tuple(metric_header(n) for n in vote) if vote else None

    floor_table = table.get("floor")
    if floor_table is None:
        floors = power_floors(power_w, by_model)
    elif not isinstance(floor_table, Mapping):
        raise JobscopeError(
            "[eff] floor must be a table per metric, e.g.\n"
            "  [eff.floor.power]\n  default = 100\n"
            "-- not %r. Write a bare [eff.floor] for no floors at all."
            % (floor_table,))
    else:
        # An explicit table replaces the built-in, so an empty one really means "no
        # metric caps a verdict" -- which re-opens what the power floor closes.
        floors = {metric_header(name): _floor_table(
            "[eff.floor.%s]" % name, body, power_w)
            for name, body in floor_table.items()}

    ceiling_table = table.get("ceiling") or {}
    ceilings = dict(DEFAULT_VOTE_CEILING)
    for name, tier in ceiling_table.items():
        if str(tier) not in TIER_NAMES:
            raise JobscopeError(
                "[eff.ceiling] %s = %r is not a tier; the tiers are %s"
                % (name, tier, ", ".join(TIER_NAMES)))
        ceilings[metric_header(name)] = str(tier)

    both = sorted(set(resolved or ()) & set(floors))
    if both:
        raise JobscopeError(
            "[eff] names %s as both a vote and a floor. A vote can only raise a "
            "verdict and a floor can only lower it, so a metric cannot be both -- "
            "pick one." % ", ".join(both))

    _check_eff_names("[eff] vote", resolved)
    _check_eff_names("[eff] floor", tuple(floors) if floor_table else None)
    _check_eff_names("[eff.ceiling]",
                     tuple(metric_header(n) for n in ceiling_table))
    return resolved, floors, ceilings


def _check_sections(data: Mapping) -> None:
    """Report a top-level name jobscope does not read.

    A note rather than an error, unlike every inner table's unknown-key check. The
    asymmetry is deliberate: a section jobscope does not read is inert, and the rest of
    the file still resolves, so raising would take a site's whole CLI down over one dead
    paragraph. But silence is worse than either -- a typo'd ``[promtheus]`` costs a site
    every setting under it with nothing on screen to say so, which is the same failure
    the ``[thresholds]`` note exists to prevent one level down.

    "section or key" because a scalar written above the first header lands here too, and
    telling someone to check a section they cannot find is a worse hint than none.
    """
    for name in sorted(set(data) - KNOWN_SECTIONS):
        print("note: [%s] is not a section or key jobscope reads, so it is being "
              "ignored. 'jobscope config --example' lists the sections."
              % name, file=sys.stderr)


def _check_eff_names(where: str, headers: Optional[Tuple[str, ...]]) -> None:
    """Reject a name no metric answers to.

    Named rather than ignored, and this one matters more than most: a typo in
    ``vote`` silently empties the ballot, and every job then reports ``no-data``
    while looking exactly like a Prometheus outage.
    """
    if not headers:
        return
    known = _known_percent_headers() | _all_headers()
    strays = sorted(h for h in headers if h not in known)
    if strays:
        raise JobscopeError(
            "%s names no metric %s. Use the short names 'jobscope probe --metrics' "
            "lists, e.g. gpu, sm_act, cpu, power."
            % (where, ", ".join(repr(h) for h in strays)))


def _all_headers() -> frozenset:
    """Every column header in the catalogs, percentage or not.

    Wider than :func:`_known_percent_headers` because a floor metric is typically
    *not* a percentage -- POWER_W being the case that ships.

    Asked of :mod:`jobscope.metrics` rather than reassembled here: "resolved GPU
    candidates, plus the derived columns, plus the host family" is exactly what that
    module composes, and a second spelling of it is one of the hand-kept header sets
    it was written to end. A third family, or a change to what counts as derived,
    then lands in one place -- and missing the second copy fails silently, as a
    legitimate ``[eff] vote`` name rejected as a typo.

    Imported inside the function for the reason the catalogs are; see
    :func:`_known_percent_headers`.
    """
    from . import metrics
    return frozenset(metrics.catalog().by_header)


def _metric_list(where: str, raw) -> Optional[list]:
    """An ``[eff]`` metric list, or None when the key is absent."""
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        # `[eff.floor.power]` tables make `floor` a table as well as a list; the
        # list form is what names which of them are active.
        return sorted(raw)
    if not isinstance(raw, (list, tuple)):
        raise JobscopeError("%s must be a list of metric names, not %r" % (where, raw))
    if not all(isinstance(n, str) and n.strip() for n in raw):
        raise JobscopeError("%s must contain metric names" % where)
    return [n.strip() for n in raw]


def _floor_table(where: str, body, fallback: float) -> dict:
    """One ``[eff.floor.<metric>]`` table -> ``{"": default, model: value}``."""
    if body is None:
        return {"": float(fallback)}
    if not isinstance(body, Mapping):
        raise JobscopeError("%s must be a table, e.g.\n  %s\n  default = 100"
                            % (where, where))
    table = {}
    for key, value in body.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise JobscopeError("%s: %s = %r is not a number" % (where, key, value))
        table["" if key == "default" else str(key)] = number
    if "" not in table:
        raise JobscopeError(
            "%s needs a `default`, the floor for hardware it does not name -- "
            "otherwise a card with no entry has no floor and never caps" % where)
    return table


def power_floors(default: float = DEFAULT_POWER_W,
                 by_model: Optional[Mapping[str, float]] = None) -> dict:
    """A ``floors`` table holding only POWER_W -- the common case, spelled once."""
    table = {"": float(default)}
    table.update({str(model): float(value) for model, value in (by_model or {}).items()})
    return {POWER_HEADER: table}


def floor_band(value: float, floor: float) -> str:
    """Band for a metric graded against an absolute floor: red below, green at or above.

    No yellow, deliberately. Yellow means "close to the cutoff", which for a percentage
    is everything up to twice it -- reasonable when red is 10%. An absolute floor has
    no such headroom: at a 330 W floor, twice is 660 W and an RTX PRO 6000 tops out
    around 480, so every one of those cards would read red or yellow forever, idle or
    flat out. A floor asserts one thing -- below this is idle -- so it answers one.
    """
    return "green" if value >= floor else "red"


@dataclass(frozen=True)
class Defaults:
    """Default values for CLI options that a site may want to override."""

    workers: int
    timeout: float
    # Defaulted so existing constructions keep working.
    min_elapsed: str = DEFAULT_MIN_ELAPSED
    # The `finished` window when none of -D/-N/-S/-E is given.
    days: int = DEFAULT_DAYS
    # Which job endings `finished` reports without -t. The default hides
    # FAILED/TIMEOUT/CANCELLED, which is worth being able to set once per site
    # rather than typing every run. Validated against slurm.STATE_GROUPS.
    state: str = DEFAULT_STATE
    # How many jobs each Problem-jobs "Wasteful" row lists.
    worst_jobs: int = DEFAULT_WORST_JOBS
    # A Wasteful-row entry whose job ran at least this long is highlighted: hours of
    # idle hardware do not come back, where a short bad job costs little.
    long_running: str = DEFAULT_LONG_RUNNING
    # How many running jobs one selection will sweep before refusing. See
    # DEFAULT_MAX_RUNNING_JOBS -- a backstop, deliberately far above any real partition.
    max_running_jobs: int = DEFAULT_MAX_RUNNING_JOBS
    # What --eff and --verify look back over when given no window. See
    # DEFAULT_VERDICT_WINDOW; a bare --ts is deliberately not covered.
    verdict_window: str = DEFAULT_VERDICT_WINDOW


# The summary block's sections, in the order they print by default. Each answers a
# different question -- how was each metric used, how do they compare, and which jobs
# are the problem -- so a site that only wants one of the three should not have to
# scroll past the others. Keys are short names; the headings stay in report.py, since
# a heading is wording and this is structure.
REPORT_SECTIONS = ("metrics", "efficiency", "problems")

# What a chart defaults to plotting, in the order the tables use. DUTY% is listed only
# so a CSV written before that column was renamed to GPU% still charts; the two are the
# same quantity and ts_defaults() shows at most one.
#
# GMEM% sits next to GPU% because the two are *resources* -- how full the card is and
# how busy it is -- where OCC%/TENSOR%/DRAM% describe how the SMs were used, which only
# means something once the GPU is known to be busy. Memory also catches a failure none
# of them do: GPU% 96 with GMEM% 3 is under-batched, and no profiling column says so.
# GMEM_GB stays out as the same quantity without a denominator.
DEFAULT_PLOT_METRICS = ("GPU%", "DUTY%", "GMEM%", "SM_ACT%", "OCC%", "TENSOR%", "DRAM%")

# Distinct 256-colour codes for the time-series chart, one per metric, so the plotext
# line and the rich-tinted per-metric stats render the exact same colour. Separate from
# [colors], which grades a *value* by its band -- these only tell series apart.
DEFAULT_PLOT_PALETTE = (196, 46, 33, 208, 201, 51, 226, 129, 244, 39)

DEFAULT_HEAT_MAX_ROWS = 40
DEFAULT_PLOT_PANELS = 12


@dataclass(frozen=True)
class Report:
    """``[report]``: which parts of the summary block print, and in what order.

    Only the block below the job table. The table itself is the report, and its
    columns are chosen by ``[metrics]`` and the view flags.
    """

    sections: Tuple[str, ...] = REPORT_SECTIONS
    # --verify's window ladder, narrowest last. The *collected* window is always the
    # first rung and is not listed here -- that is DEFAULT_VERDICT_WINDOW (180m) unless
    # --verify was given a span of its own, and it was the job's whole runtime before
    # that default existed. Two rungs by default: enough to show a trend without making
    # the table wide, and 30m is 31 samples at a 60s scrape -- thin, which is why max and
    # the idle-stretch figure carry the decision rather than the mean alone.
    verify_windows: Tuple[Tuple[str, int], ...] = DEFAULT_VERIFY_WINDOWS


@dataclass(frozen=True)
class Plot:
    """``[plot]``: chart defaults, for both ``jobscope plot`` and ``--plot_ts``.

    Held here rather than in plot.py so a site sets them once instead of typing
    ``--max-rows`` every run, and so ``--plot_ts`` and a piped ``jobscope plot`` cannot
    disagree about what a chart looks like.
    """

    metrics: Tuple[str, ...] = DEFAULT_PLOT_METRICS
    palette: Tuple[int, ...] = DEFAULT_PLOT_PALETTE
    # Heatmap row cap. --max-rows overrides it per run.
    max_rows: int = DEFAULT_HEAT_MAX_ROWS
    # Side-by-side panels for --by metric --gpu a,b,c.
    panels: int = DEFAULT_PLOT_PANELS


@dataclass(frozen=True)
class Metrics:
    """Which metrics each view collects and shows.

    Split by side rather than mixed, because the two are consumed by different
    collectors: the GPU lists become Prometheus GPU queries keyed by card UUID, the
    host lists become cgroup queries keyed by job id. One list holding both would have
    to be re-split at every call site.

    A ``[metrics]`` view list may name either kind -- ``summary = ["cpu", "sm_act"]``
    -- and :func:`_metrics` sorts them into these fields by family, so a site states
    the columns it wants without having to know which collector answers.

    Held as resolved spec lists rather than names, so every consumer reads one place
    and a name is validated once, at load.
    """

    summary: Tuple = ()      # the summary table's profiling block
    timeseries: Tuple = ()   # --ts / --plot_ts / --eff
    extended: Tuple = ()     # --all-metrics
    host_summary: Tuple = ()      # CPU%/MEM% and any other cgroup column
    host_timeseries: Tuple = ()
    host_extended: Tuple = ()

    def __post_init__(self) -> None:
        # Deferred so config stays importable without dcgm (which reaches
        # prometheus, and so back to config) -- see _known_percent_headers.
        from . import cpu as cpu_module
        from .dcgm import catalog, default_view, specs_named
        gpu_catalog = catalog()
        # Per leading source, not one list for both: see dcgm.default_view. The
        # nvidia exporter publishes no profiling metrics, so leading with it must not
        # leave a summary asking for four columns it will render as "-".
        for name in ("summary", "timeseries", "extended"):
            if not getattr(self, name):
                object.__setattr__(self, name, tuple(default_view(name)))
            host = "host_" + name
            if not getattr(self, host):
                object.__setattr__(self, host, tuple(cpu_module.default_view(name)))
        # The jobstats-backed metrics feed GPU% and GMEM%, which are *fixed* columns of
        # the summary and detail tables rather than part of the configurable
        # profiling block. A config that leaves them out is not asking for narrower
        # output, it is asking for two of its own columns to read "-" -- and only in
        # the running view, where they come from Prometheus rather than the jobstats summary. So
        # they are added back rather than obeyed. Not to `timeseries`: its CSV has no
        # fixed columns, so there a narrower list means exactly what it says.
        # Compared by column rather than by key: a list that named the other
        # provider of GPU% already has that column, and adding this one too would
        # print it twice from two exporters.
        required = [spec for spec in gpu_catalog.all_specs
                    if spec.key in gpu_catalog.jobstats_backed_keys]
        for name in ("summary", "extended"):
            listed = getattr(self, name)
            missing = [s for s in required if s.column not in {x.column for x in listed}]
            if missing:
                object.__setattr__(self, name, tuple(specs_named(
                    [s.key for s in listed] + [s.key for s in missing])))


@dataclass(frozen=True)
class Site:
    """The label conventions jobscope's queries assume, so a port needs no patch.

    These are not preferences -- they are facts about how a cluster's exporters
    label their series, and getting one wrong fails *silently*. A stock Prometheus
    calls the scrape target ``instance`` where jobstats' exporter calls it ``host``;
    read the wrong one and every node reads ``?``, the cgroup divisor lookup misses,
    and CPU%/MEM% come back blank with no error at all. ``jobscope probe`` checks
    each of these against the live server for exactly that reason.

    ``cgroup_selector`` is the matcher appended to every ``cgroup_*`` query. It pins
    the job-level cgroup rather than a per-step one; ``=''`` also matches the label
    being absent, which is the case on exporters that do not emit it. A site whose
    exporter labels steps differently overrides the whole matcher rather than
    guessing at its parts.
    """

    host_label: str = "host"            # the node a series came from
    jobid_label: str = "jobid"          # cgroup_*'s job label
    gpu_job_join: str = "nvidia_gpu_jobId"   # series whose *value* is the job id
    cgroup_selector: str = "step='',task=''"
    # A *second* job-to-card mapping, structurally unlike the one above: the job id is a
    # label on the GPU metric series rather than the value of a dedicated series.
    # dcgm-exporter publishes one as `hpc_job` when its HPC job mapping is configured.
    #
    # Worth naming even though most sites have only the join, because the two fail
    # independently. Measured on this cluster while `nvidia_gpu_jobId` was frozen: 76% of
    # cards named a job that had already finished and every job started after the freeze
    # reported blank GPU columns, with no second mapping to fall back on. A site running
    # both would have kept reporting.
    #
    # Empty disables it, which is the default: a label that is not published matches
    # nothing, and silently querying for it would cost a round trip per report.
    gpu_job_label: str = ""
    # Which series to read that label from. SM_ACTIVE rather than GPU_UTIL because a
    # partitioned card publishes no whole-device duty cycle -- measured here, 1966 series
    # against 1743 -- so this is the one that also covers MIG hosts.
    gpu_job_label_series: str = "DCGM_FI_PROF_SM_ACTIVE"


def gpu_join() -> str:
    """The series whose *value* is the job id holding each card.

    A function rather than a module constant because ``[site]`` can override it, and
    a constant read at import time would freeze the default before any config was
    loaded.
    """
    return get_config().site.gpu_job_join


def jobid_label() -> str:
    """The label ``cgroup_*`` series carry their job id in."""
    return get_config().site.jobid_label


def gpu_job_label() -> str:
    """The label a GPU metric series carries its job id in, or "" if none does.

    The second mapping described on :class:`Site`. Empty at most sites, and the callers
    treat that as "no fallback exists" rather than querying for a label nothing publishes.
    """
    return get_config().site.gpu_job_label


def gpu_job_label_series() -> str:
    """The series to read :func:`gpu_job_label` from."""
    return get_config().site.gpu_job_label_series


def host_of(labels: Mapping[str, str], site: Optional["Site"] = None) -> str:
    """The node name from a series' labels, port stripped.

    One definition for every collector: the four that each had their own copy of
    ``labels.get("host", "?").split(":")[0]`` could not be ported without finding
    all four, and three of them would have failed quietly.
    """
    site = get_config().site if site is None else site
    return str(labels.get(site.host_label, "?")).split(":")[0]


def short_host(name) -> str:
    """A node name without its domain, so ``n1.cluster.edu`` and ``n1`` compare equal.

    Beside :func:`host_of` because it is the other half of the same question and was, for
    one commit, a private copy in a collector -- which left ``probe``'s cross-check
    comparing unnormalised names and reporting a clean mapping at a site whose exporter
    labels are fully qualified, while the report path stripped the domain and disagreed.
    ``host_of`` strips a port; Slurm's node lists carry neither.
    """
    return str(name).split(".")[0]


@dataclass(frozen=True)
class Config:
    """Resolved jobscope settings."""

    prometheus_url: Optional[str]
    sampling_period: int
    sampling_period_explicit: bool
    site_prom_config_path: Optional[str]
    # The whole-elapsed-time job summary's bands. Named plainly because it is the
    # default view; the time-slice table below is the one that needs qualifying.
    thresholds: Thresholds
    defaults: Defaults
    source_path: Optional[Path] = None
    # --ts's bands. Last and defaulted because `defaults` above is not, so nothing
    # defaulted can be slotted before it. cli.handle_report picks between the two.
    timeslice_thresholds: Thresholds = field(default_factory=Thresholds)
    metrics: "Metrics" = field(default_factory=lambda: Metrics())
    palette: Palette = field(default_factory=Palette)
    site: "Site" = field(default_factory=lambda: Site())
    # Where prometheus_url came from, for `jobscope config` to name. "" when nothing
    # set it directly, in which case resolve_prometheus may still find one through
    # jobstats -- endpoint_source() reports that case, since only resolution knows.
    prometheus_from: str = ""
    # Query pacing, read by prometheus.client_from_config. On Config rather than Defaults
    # because they describe the endpoint's tolerance, not this run's preferences. Defaulted
    # so existing constructions keep working, as Defaults.min_elapsed is.
    max_queries_per_second: float = DEFAULT_MAX_QUERIES_PER_SECOND
    query_burst: int = DEFAULT_QUERY_BURST
    report: "Report" = field(default_factory=lambda: Report())
    plot: "Plot" = field(default_factory=lambda: Plot())
    gpu: "Gpu" = field(default_factory=lambda: Gpu())
    host: "Host" = field(default_factory=lambda: Host())


@dataclass(frozen=True)
class Gpu:
    """Where each GPU column's numbers come from.

    ``source`` is a preference order over ``jobstats``, ``dcgm`` and ``nvml``, not a
    choice of one: every column resolves to its own best available provider, so
    naming dcgm cannot take away a column only the nvidia exporter publishes. See
    :mod:`jobscope.source` for the resolution, and why the default puts the free
    source first.
    """

    source: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source:
            from .source import DEFAULT_PREFERENCE
            object.__setattr__(self, "source", DEFAULT_PREFERENCE)


@dataclass(frozen=True)
class Host:
    """Where CPU%/MEM% come from -- the host counterpart of :class:`Gpu`.

    A separate section rather than a second key in ``[gpu]`` because the axes are
    independent: the candidates differ (``cgroup`` against ``dcgm``/``nvml``), and a
    cluster commonly has one exporter and not the other.
    """

    source: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source:
            from .source import DEFAULT_HOST_PREFERENCE
            object.__setattr__(self, "source", DEFAULT_HOST_PREFERENCE)


def default_config_path(env: Optional[Mapping[str, str]] = None) -> Path:
    """Per-user path, the last tier of :func:`config_search_paths` and its fallback."""
    env = os.environ if env is None else env
    base = env.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "jobscope" / "config.toml"


def _repo_root() -> Optional[Path]:
    """The checkout jobscope is running from, or None for a non-editable install.

    Resolved from this module's own location rather than the working directory: a
    site's config should not change depending on where someone happened to `cd`, and
    the runs that matter here are cron lines and sbatch scripts, whose cwd is nobody's
    choice. An editable install leaves ``__file__`` inside the clone, so the walk finds
    it; a wheel in site-packages has no marker above it and this returns None, which is
    what keeps `pip install jobscope` behaving exactly as it did before repo tiers.

    Anchored on this file's own position rather than walked upward, and that is a
    security property, not a tidiness one. A walk looking for ``.git`` or
    ``pyproject.toml`` finds the *nearest* one, which on a shared login node is not
    necessarily ours: an empty ``/tmp/.git`` owned by another user is enough to make
    ``/tmp`` the root, and jobscope would then read ``/tmp/config.toml`` -- a file that
    user can write, naming the Prometheus endpoint we authenticate to. (There is such a
    directory on this cluster already, which is how this was found.) So the layout must
    match exactly: only ``<root>/src/jobscope/config.py`` yields a root, which a wheel in
    ``site-packages/jobscope/`` never does.

    The marker is still required on top of the layout, so that a source tree someone
    merely copied around is not mistaken for a checkout.
    """
    here = Path(__file__).resolve()
    if here.parent.name != "jobscope" or here.parent.parent.name != "src":
        return None
    root = here.parent.parent.parent
    if (root / "pyproject.toml").is_file() or (root / ".git").exists():
        return root
    return None


def config_search_paths(env: Optional[Mapping[str, str]] = None) -> Tuple[Path, ...]:
    """Where jobscope looks, in order, once -c and $JOBSCOPE_CONFIG have had their say.

    The two repo tiers exist only inside a checkout, and they are ordered
    override-before-tracked for the same reason ``-c`` beats everything: the narrower
    statement of intent wins. ``jobscope.toml`` is the site's policy, reviewed like code
    and shared by every admin; ``config.toml`` beside it is git-ignored and belongs to
    whoever is trying something out, so it has to be able to win locally without a commit.
    """
    paths = []
    root = _repo_root()
    if root is not None:
        paths.append(root / "config.toml")
        paths.append(root / "jobscope.toml")
    paths.append(default_config_path(env))
    return tuple(paths)


def resolve_config_path(path: Optional[str] = None,
                        env: Optional[Mapping[str, str]] = None) -> Path:
    """The file jobscope reads, whether or not it exists.

    The single answer to "which config is in play", so that ``jobscope config``,
    ``probe``'s config line and :func:`load_config` cannot drift apart -- they used to
    each spell the precedence out again, which was survivable while it was two tiers.
    Falls back to the per-user path when nothing exists: that is the one a reader can
    be told to create, since a checkout is not something everyone has.
    """
    env = os.environ if env is None else env
    explicit = path if path is not None else env.get(CONFIG_ENV)
    if explicit:
        return Path(explicit)
    searched = config_search_paths(env)
    return next((p for p in searched if p.exists()), searched[-1])


def init_target_path(env: Optional[Mapping[str, str]] = None) -> Path:
    """Where ``probe --init`` writes when nothing named a path.

    Deliberately *not* :func:`resolve_config_path`'s answer inside a checkout. ``--init``
    emits a generated file from what probe just measured, and the repo's tracked
    ``jobscope.toml`` is shared policy that several admins read through pull requests --
    machine output should not land there by default. It goes to the git-ignored
    ``config.toml`` beside it instead, and promoting anything worth keeping into the
    tracked file stays a deliberate act.
    """
    env = os.environ if env is None else env
    explicit = env.get(CONFIG_ENV)
    if explicit:
        return Path(explicit)
    root = _repo_root()
    return (root / "config.toml") if root is not None else default_config_path(env)


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as fh:
        return _toml.load(fh)


def load_config(path: Optional[str] = None,
                env: Optional[Mapping[str, str]] = None,
                gpu_source: Optional[str] = None) -> Config:
    """Build a Config from a TOML file plus environment overrides.

    Resolution order for the file: ``path`` argument, then ``$JOBSCOPE_CONFIG``,
    then the first existing entry of :func:`config_search_paths` -- the repo's
    ``config.toml``, the repo's tracked ``jobscope.toml``, the per-user path. A file
    named explicitly (argument or env var) must exist; the searched paths may all be
    absent, in which case built-in defaults apply. The Prometheus URL prefers
    ``$JOBSCOPE_PROM_URL`` over the file.

    ``gpu_source`` overrides ``[gpu] source``, and is taken here rather than applied
    afterwards because the order decides which candidate wins each column *and* which
    per-source ``[metrics.<family>]`` view list applies -- both of which are resolved
    during this call. Applied after the fact, a report would collect one source's
    metrics while resolving names against another's.
    """
    env = os.environ if env is None else env
    explicit = path if path is not None else env.get(CONFIG_ENV)
    chosen = resolve_config_path(path, env)
    if explicit and not chosen.exists():
        raise JobscopeError("jobscope config not found: %s" % chosen)
    data = _read_toml(chosen) if chosen.exists() else {}

    # First, before any section's *contents* are validated: everything below can raise
    # (a bad source name, a typo'd metric), and a file carrying a retired section
    # usually carries a retired value too -- so a note deferred until after those
    # checks is a note the one config that needs it never sees.
    _check_sections(data)

    prom = data.get("prometheus") or {}
    thr = data.get("thresholds") or {}
    dfl = data.get("defaults") or {}

    for old, new in sorted(RENAMED_PROMETHEUS_KEYS.items()):
        if old in prom:
            print("note: [prometheus] %s has been renamed to %s and its value is no"
                  " longer read. Rename the key in %s; the value is unchanged."
                  % (old, new, chosen), file=sys.stderr)

    # Before anything resolves a metric name. [metrics.<family>.<name>] tables add to
    # the catalogs and the preference decides which candidate wins each column, so the
    # view selections below and [thresholds]' typo check both need this to have
    # happened -- and to have happened in the right order. build_catalogs owns that
    # order; see its docstring for what each half-applied state looks like.
    gpu_section = _gpu(data.get("gpu") or {})
    host_section = _host(data.get("host") or {})
    if gpu_source:
        from .source import parse_preference
        gpu_section = Gpu(source=parse_preference(gpu_source, "--gpu-source"))
    build_catalogs(data.get("metrics") or {}, gpu_section.source, host_section.source)

    stale = [key for key in LEGACY_THRESHOLD_KEYS if key in thr]
    if stale:
        # Not silently: a site that set gpu = 25 would otherwise stop taking effect
        # with no sign of it. The metric names among these have a new home one level
        # down, so the note points there rather than just declaring them dead.
        print("note: [thresholds] %s no longer apply at the top level; the band edges"
              " now live per view and per metric, as [thresholds.summary.<edge>] and"
              " [thresholds.timeslice.<edge>] (see jobscope config --example)"
              % ", ".join(sorted(stale)), file=sys.stderr)

    power_w = float(thr.get("power_w", DEFAULT_POWER_W))
    by_model = {str(k): float(v) for k, v in (thr.get("power_w_by_model") or {}).items()}
    edges = _edges(thr.get("edges"))
    named = [view for view in BAND_VIEWS if view in thr]
    if len(named) == 1:
        # Nothing is inherited between the two, by design -- so a config that tunes
        # one and forgets the other grades the same job differently depending on
        # whether --ts was passed. Say so once; it stops as soon as both are set.
        #
        # What the other view falls back to depends on whether `edges` was given, and
        # naming the wrong one sends someone looking for a number that is not there.
        other = [view for view in BAND_VIEWS if view != named[0]][0]
        print("note: [thresholds.%s] is set but [thresholds.%s] is not, so the two"
              " views grade differently -- %s keeps %s (nothing is inherited between"
              " them). 'jobscope config' prints both."
              % (named[0], other, other,
                 "the [thresholds] edges" if edges else "the built-in edges"),
              file=sys.stderr)
    vote, floors, ceilings = _eff(data.get("eff") or {}, power_w, by_model)
    bands = {view: _band_table(thr.get(view) or {}, view, floors, vote, ceilings,
                               base=edges)
             for view in BAND_VIEWS}

    defaults = Defaults(
        workers=int(dfl.get("workers", DEFAULT_WORKERS)),
        timeout=float(dfl.get("timeout", DEFAULT_TIMEOUT)),
        min_elapsed=str(dfl.get("min_elapsed", DEFAULT_MIN_ELAPSED)),
        days=int(dfl.get("days", DEFAULT_DAYS)),
        state=_state_name(dfl.get("state", DEFAULT_STATE)),
        worst_jobs=_positive(dfl.get("worst_jobs", DEFAULT_WORST_JOBS),
                             "[defaults] worst_jobs"),
        long_running=str(dfl.get("long_running", DEFAULT_LONG_RUNNING)),
        max_running_jobs=_positive(dfl.get("max_running_jobs", DEFAULT_MAX_RUNNING_JOBS),
                                   "[defaults] max_running_jobs"),
        verdict_window=_duration(dfl.get("verdict_window", DEFAULT_VERDICT_WINDOW),
                                 "[defaults] verdict_window"),
    )
    url, url_from = _prometheus_url(prom, env, chosen)
    return Config(
        prometheus_url=url,
        prometheus_from=url_from,
        sampling_period=int(prom.get("sampling_period", DEFAULT_SAMPLING_PERIOD)),
        sampling_period_explicit="sampling_period" in prom,
        # Zero is a real setting for the rate and means "do not pace".
        max_queries_per_second=_non_negative_float(
            prom.get("max_queries_per_second", DEFAULT_MAX_QUERIES_PER_SECOND),
            "[prometheus] max_queries_per_second"),
        query_burst=_positive(prom.get("query_burst", DEFAULT_QUERY_BURST),
                              "[prometheus] query_burst"),
        site_prom_config_path=prom.get("site_prom_config_path"),
        thresholds=bands["summary"],
        defaults=defaults,
        source_path=chosen if chosen.exists() else None,
        timeslice_thresholds=bands["timeslice"],
        metrics=_metrics(data.get("metrics") or {}),
        palette=_palette(data.get("colors") or {}),
        site=_site(data.get("site") or {}),
        report=_report(data.get("report") or {}),
        plot=_plot(data.get("plot") or {}),
        gpu=gpu_section,
        host=host_section,
    )


def _non_negative_float(raw, where: str) -> float:
    """``raw`` as a non-negative float. Zero is a real setting for a rate: no pacing."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise JobscopeError("%s = %r is not a number" % (where, raw))
    if value < 0:
        raise JobscopeError("%s cannot be negative, got %g" % (where, value))
    return value


def _non_negative(raw, where: str) -> int:
    """``raw`` as a non-negative int. Zero is a real setting for a cap, meaning "never"."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise JobscopeError("%s = %r is not a whole number" % (where, raw))
    if value < 0:
        raise JobscopeError("%s cannot be negative, got %d" % (where, value))
    return value


def _duration(raw, where: str) -> str:
    """A duration-valued key, checked here and kept as written.

    Validated at load rather than at use, because this one is read by every ``--verify``
    and ``--eff``: a typo that surfaced at render time would look like a broken report
    rather than a bad line of config. Returned as the string it was written as, so
    ``jobscope config`` echoes back what the file says.
    """
    text = str(raw).strip()
    try:
        parse_duration(text)
    except JobscopeError as exc:
        raise JobscopeError("%s: %s" % (where, exc))
    return text


def _positive(raw, where: str) -> int:
    """``raw`` as a positive int, or a JobscopeError naming ``where``."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise JobscopeError("%s = %r is not a whole number" % (where, raw))
    if value < 1:
        raise JobscopeError("%s must be at least 1, not %d" % (where, value))
    return value


def _state_name(raw) -> str:
    """A ``[defaults] state`` value, checked by the same rule ``-t`` uses.

    Delegated rather than re-listed, so the config and the flag cannot come to
    disagree about the vocabulary -- including ``all``, comma-separated groups, and
    the refusal of live states.
    """
    from .slurm import states_for
    name = str(raw).strip().lower()
    try:
        states_for(name)
    except JobscopeError as exc:
        raise JobscopeError("[defaults] state = %r: %s" % (raw, exc))
    return name


def _site(table: Mapping) -> Site:
    """``[site]`` -> a :class:`Site`, rejecting keys it does not understand.

    Unknown keys are an error rather than a shrug: a misspelled ``host_lable`` that
    parsed silently would leave the default in place and produce a report full of
    ``?`` nodes, which looks like a broken cluster rather than a typo.
    """
    defaults = {f.name: f.default for f in fields(Site)}
    known = set(defaults)
    unknown = sorted(set(table) - known)
    if unknown:
        raise JobscopeError(
            "[site] does not know %s. Valid keys: %s."
            % (", ".join(repr(k) for k in unknown), ", ".join(sorted(known))))
    values = {}
    for name in known:
        if name in table:
            raw = table[name]
            if not isinstance(raw, str):
                raise JobscopeError("[site] %s must be a string" % name)
            value = raw.strip()
            # Empty is allowed only where the default is empty, which is how a key that
            # names an *optional* mapping is switched off. Elsewhere it stays an error:
            # blanking host_label would leave every node reading "?" and look like a
            # broken cluster rather than a blanked setting.
            if not value and defaults[name] != "":
                raise JobscopeError("[site] %s must be a non-empty string" % name)
            values[name] = value
    return Site(**values)


def _verify_windows(table: Mapping) -> Tuple[str, ...]:
    """``[report] verify_windows`` -> durations, widest first, validated.

    Sorted here rather than trusted in file order, because the shape rules read the ladder
    widest-to-narrowest: a list written the other way round would invert "declining" into
    "improving" and report it as the former.
    """
    if "verify_windows" not in table:
        return DEFAULT_VERIFY_WINDOWS
    raw = table["verify_windows"]
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise JobscopeError(
            "[report] verify_windows must be a list of durations, e.g. %r"
            % (list(DEFAULT_VERIFY_WINDOWS),))
    # parse_duration raises with the offending value named, which is the whole
    # diagnostic -- "30" without a unit is the mistake people make. Parsed here and
    # handed on as (label, seconds) so no consumer parses it again: verify_rungs used to,
    # behind an `except` that could not fire, which is a second validation policy for one
    # setting and would silently drop what this reports.
    labels = [str(item).strip() for item in raw]
    return tuple(sorted(((text, parse_duration(text)) for text in labels),
                        key=lambda pair: pair[1], reverse=True))


def _report(table: Mapping) -> Report:
    """``[report]`` -> a :class:`Report`."""
    unknown = sorted(set(table) - {"sections", "verify_windows"})
    if unknown:
        raise JobscopeError(
            "[report] does not know %s. Valid keys: sections, verify_windows."
            % ", ".join(repr(k) for k in unknown))
    windows = _verify_windows(table)
    if "sections" not in table:
        return Report(verify_windows=windows)
    raw = table["sections"]
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise JobscopeError(
            "[report] sections must be a list, e.g. sections = %r" % (list(REPORT_SECTIONS),))
    chosen = [str(x).strip() for x in raw]
    bad = [x for x in chosen if x not in REPORT_SECTIONS]
    if bad:
        raise JobscopeError("[report] sections has no %s; it takes %s"
                            % (", ".join(repr(x) for x in bad), ", ".join(REPORT_SECTIONS)))
    # A repeat would print the section twice and renumber everything after it, which
    # reads as a rendering bug rather than as the config it is.
    seen = [x for i, x in enumerate(chosen) if x in chosen[:i]]
    if seen:
        raise JobscopeError("[report] sections repeats %s" % ", ".join(sorted(set(seen))))
    return Report(sections=tuple(chosen), verify_windows=windows)


def _plot(table: Mapping) -> Plot:
    """``[plot]`` -> a :class:`Plot`, with each value range-checked.

    The counts are checked because a zero or negative one does not fail -- it renders
    an empty chart, which looks like "no data" and sends someone looking at the
    cluster instead of at their config.
    """
    known = {f.name for f in fields(Plot)}
    unknown = sorted(set(table) - known)
    if unknown:
        raise JobscopeError("[plot] does not know %s. Valid keys: %s."
                            % (", ".join(repr(k) for k in unknown), ", ".join(sorted(known))))
    values = {}
    for name in ("metrics", "palette"):
        if name not in table:
            continue
        raw = table[name]
        if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
            raise JobscopeError("[plot] %s must be a list" % name)
        if not raw:
            raise JobscopeError("[plot] %s is empty; omit the key to keep the default"
                                % name)
        values[name] = tuple(raw)
    if "metrics" in values:
        # Resolved to column headers, because a chart reads a CSV that was already
        # written and can only plot what its header row says. Through metric_header so
        # the short names [metrics] and [thresholds] take work here too -- `sm_act` and
        # `SM_ACT%` are the same request, and a site should not have to remember which
        # spelling each section wants.
        #
        # Not validated against the catalog: the default list names DUTY%, which has no
        # catalog entry and exists only so a CSV saved before the GPU% rename still
        # charts. A name absent from a given CSV simply does not plot -- that is what
        # makes one list serve files with different columns.
        values["metrics"] = tuple(metric_header(str(m).strip())
                                  for m in values["metrics"])
    if "palette" in values:
        for colour in values["palette"]:
            if not isinstance(colour, int) or not 0 <= colour <= 255:
                raise JobscopeError(
                    "[plot] palette takes 256-colour codes (0-255), not %r" % (colour,))
    for name in ("max_rows", "panels"):
        if name in table:
            raw = table[name]
            if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
                raise JobscopeError("[plot] %s must be a positive integer, not %r"
                                    % (name, raw))
            values[name] = raw
    return Plot(**values)


VIEWS = ("summary", "timeseries", "extended")

# The families a [metrics.<family>.<name>] table can define a metric in. `nvml` and
# `dcgm` share one catalog and differ only by which label carries the GPU UUID, so
# naming the family is how a site says which -- getting it wrong yields a response
# whose rows cannot be attributed to a card.
FAMILIES = ("dcgm", "nvml", "cgroup")

# What a definition may set, per family, beyond the required query/header. Kept
# explicit so a key that belongs to the other family is rejected rather than
# silently dropped -- `scale` on a cgroup metric, or `denom` on a DCGM one, means
# the author has the wrong mental model and the numbers would be wrong.
_GPU_KEYS = {"query", "header", "reducer", "agg", "scale", "decimals", "unit"}
_CGROUP_KEYS = {"query", "header", "kind", "denom", "decimals"}


def _definitions(table: Mapping) -> Mapping:
    """The ``[metrics.<family>]`` sub-tables, separated from the view selections.

    Distinguished by type rather than by name: a view is a list or ``"all"``, a
    family is a table of tables. That keeps ``[metrics] summary = [...]`` and
    ``[metrics.dcgm.xid]`` in one section without either needing a marker.
    """
    return {key: value for key, value in table.items() if isinstance(value, Mapping)}


def _builtin_named(family: str, name: str):
    """The built-in spec ``name`` would override in ``family``, or None.

    Looked up so an override inherits every field it does not mention. Overriding
    ``cpu`` to point at another exporter's series must not also rename the column
    from ``CPU%`` to ``CPU`` -- the header is the identity ``[thresholds]``,
    ``--csv`` consumers and job_eff's own literals all key on.

    Matched on family as well as key for the GPU families, because one column can
    have a candidate in each: without it, ``[metrics.nvml.power]`` would inherit
    the *DCGM* power spec, and so its uppercase ``UUID`` label, and return rows
    that cannot be attributed to a card.
    """
    if family == "cgroup":
        from .cpu import CGROUP_METRICS
        return next((spec for spec in CGROUP_METRICS if spec.key == name), None)
    from .dcgm import METRICS
    return next((spec for spec in METRICS
                 if spec.key == name and spec.family == family), None)


def _one_of(where: str, body: Mapping, key: str, allowed, default: str) -> str:
    raw = body.get(key, default)
    value = str(raw).strip().lower()
    if value not in allowed:
        raise JobscopeError("%s %s must be %s, not %r"
                            % (where, key, " or ".join(allowed), raw))
    return value


def _gpu_spec(family: str, name: str, body: Mapping):
    """One ``[metrics.dcgm|nvml.<name>]`` table -> a MetricSpec."""
    from .dcgm import MetricSpec
    where = "[metrics.%s.%s]" % (family, name)
    _check_keys(where, body, _GPU_KEYS)
    base = _builtin_named(family, name)
    return MetricSpec(
        key=name,
        header=_header(where, body, name, base),
        metric=_query(where, body),
        scale=_number(where, body, "scale", base.scale if base else 1.0),
        decimals=_int(where, body, "decimals", base.decimals if base else 1),
        # `all` for a new metric: it appears in --dcgm/extended and nowhere else
        # unless [metrics] names it, so adding one cannot silently widen the default
        # report -- or the queries every sweep pays for. An override keeps the
        # built-in's group; see dcgm._inherit, which applies that after this.
        group="all",
        reducer=_one_of(where, body, "reducer", ("avg", "max", "delta"),
                        base.reducer if base else "avg"),
        agg=_one_of(where, body, "agg", ("mean", "sum", "max"),
                    base.agg if base else "mean"),
        # nvml is the lowercase-uuid family, dcgm the uppercase one. MetricSpec
        # checks the pair, so passing both is a guard rather than a repetition.
        family=family,
        uuid_label="uuid" if family == "nvml" else "UUID")


def _cgroup_spec(name: str, body: Mapping):
    """One ``[metrics.cgroup.<name>]`` table -> a CgroupSpec."""
    from .cpu import CgroupSpec
    where = "[metrics.cgroup.%s]" % name
    _check_keys(where, body, _CGROUP_KEYS)
    base = _builtin_named("cgroup", name)
    denom = _one_of(where, body, "denom", ("cpus", "total_memory"),
                    base.denom if base else "total_memory")
    return CgroupSpec(
        key=name, header=_header(where, body, name, base), metric=_query(where, body),
        kind=_one_of(where, body, "kind", ("gauge", "rate"), base.kind if base else "gauge"),
        denom=denom, decimals=_int(where, body, "decimals", base.decimals if base else 1),
        group="all")


def _check_keys(where: str, body: Mapping, allowed) -> None:
    stray = sorted(set(body) - set(allowed))
    if stray:
        raise JobscopeError("%s does not take %s; it takes %s"
                            % (where, ", ".join(repr(k) for k in stray),
                               ", ".join(sorted(allowed))))


def _query(where: str, body: Mapping) -> str:
    raw = body.get("query")
    if not isinstance(raw, str) or not raw.strip():
        raise JobscopeError("%s needs query = \"<prometheus series>\" -- the series "
                            "name exactly as 'jobscope probe --metrics' lists it" % where)
    return raw.strip()


def _header(where: str, body: Mapping, name: str, base=None) -> str:
    """The column heading: the config's, else the overridden built-in's, else the name."""
    raw = body.get("header")
    if raw is None:
        return base.header if base is not None else name.upper()
    if not isinstance(raw, str) or not raw.strip():
        raise JobscopeError("%s header must be a non-empty string" % where)
    return raw.strip()


def _number(where: str, body: Mapping, key: str, default: float) -> float:
    raw = body.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise JobscopeError("%s %s must be a number, not %r" % (where, key, raw))
    return float(raw)


def _int(where: str, body: Mapping, key: str, default: int) -> int:
    raw = body.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise JobscopeError("%s %s must be an integer, not %r" % (where, key, raw))
    if raw < 0:
        raise JobscopeError("%s %s must not be negative" % (where, key))
    return raw


def _gpu(table: Mapping) -> "Gpu":
    """``[gpu]`` -> a resolved source preference."""
    from .source import parse_preference
    unknown = sorted(set(table) - {"source"})
    if unknown:
        raise JobscopeError("[gpu] has no %s; it takes source"
                            % ", ".join(repr(k) for k in unknown))
    if "source" not in table:
        return Gpu()
    return Gpu(source=parse_preference(table["source"]))


def _host(table: Mapping) -> "Host":
    """``[host]`` -> a resolved source preference for CPU%/MEM%."""
    from .source import HOST_SOURCES, parse_preference
    unknown = sorted(set(table) - {"source"})
    if unknown:
        raise JobscopeError("[host] has no %s; it takes source"
                            % ", ".join(repr(k) for k in unknown))
    if "source" not in table:
        return Host()
    return Host(source=parse_preference(table["source"], "[host] source", HOST_SOURCES))


def _site_specs(table: Mapping) -> Tuple[list, list]:
    """``[metrics.<family>.<name>]`` tables -> (gpu specs, cgroup specs)."""
    definitions = _definitions(table)
    stray = sorted(set(definitions) - set(FAMILIES))
    if stray:
        raise JobscopeError(
            "[metrics] has no family %s; the families are %s"
            % (", ".join(repr(k) for k in stray), ", ".join(FAMILIES)))
    gpu, cgroup = [], []
    for family in FAMILIES:
        for name, body in (definitions.get(family) or {}).items():
            if name in VIEWS:
                # A per-source view selection, not a metric: [metrics.dcgm] summary =
                # [...] sits in the same table as [metrics.dcgm.<name>]. Consumed by
                # _metrics; skipped rather than rejected so the two can coexist.
                continue
            if not isinstance(body, Mapping):
                raise JobscopeError(
                    "[metrics.%s.%s] must be a table, e.g.\n"
                    '  [metrics.%s.%s]\n  query = "..."' % (family, name, family, name))
            if family == "cgroup":
                cgroup.append(_cgroup_spec(name, body))
            else:
                gpu.append(_gpu_spec(family, name, body))
    return gpu, cgroup


def build_catalogs(table: Mapping, gpu_pref: Tuple[str, ...],
                   host_pref: Tuple[str, ...]) -> None:
    """Install the site metrics and both source orders, in the one order that works.

    Two steps that have to happen together, and used to be separate calls a caller
    could interleave or forget:

    1. ``[metrics.<family>.<name>]`` definitions join the catalogs, so the view
       selections and ``[thresholds]``' typo check can name what a site just defined.
    2. Each family re-resolves which candidate serves each of its columns under the
       new preference. A name looked up before this resolves against the previous
       order.

    The cross-family role view follows automatically -- :func:`jobscope.metrics.catalog`
    derives it from whatever the two families now hold, so it cannot be left keyed to
    the previous order's winners.

    Both steps are unconditional: a config that *removes* a definition has to
    un-register it, and dropping ``[gpu]`` has to restore the default order.
    """
    from . import cpu, dcgm
    gpu_specs, cgroup_specs = _site_specs(table)
    dcgm.register(gpu_specs)
    cpu.register(cgroup_specs)
    dcgm.set_preference(gpu_pref)
    cpu.set_preference(host_pref)


def _metrics(table: Mapping) -> Metrics:
    """``[metrics]`` -> resolved spec lists.

    Each view key is a list of metric names, or the string ``"all"`` for the whole
    catalog. Absent, a view keeps its built-in list (see :meth:`Metrics.__post_init__`).
    Family sub-tables are metric *definitions* and were consumed by
    :func:`build_catalogs` before this runs.

    A view may also be stated *per source* -- ``[metrics.dcgm] summary = [...]`` --
    which is how a site says that leading with one exporter should collect a
    different set than leading with the other. The one that applies is the leading
    exporter's; a top-level ``[metrics] summary`` outranks both, since it names the
    columns unconditionally.
    """
    from . import cpu as cpu_module
    from . import dcgm as dcgm_module
    from .dcgm import spec_named, specs_named
    gpu_catalog = dcgm_module.catalog()
    families = _definitions(table)
    leading = gpu_catalog.resolved.leading_exporter()
    per_source = {view: names for view, names in (families.get(leading) or {}).items()
                  if view in VIEWS}
    table = {k: v for k, v in table.items() if k not in families}
    # The leading source's lists fill in only where the top level said nothing.
    table = dict(per_source, **table)
    unknown = [key for key in table if key not in VIEWS]
    if unknown:
        raise JobscopeError(
            "[metrics] has no %s; it takes %s, or a family table such as "
            "[metrics.dcgm.<name>] to define a metric"
            % (", ".join(sorted(unknown)), ", ".join(VIEWS)))
    resolved = {}
    for view, names in table.items():
        if isinstance(names, str):
            if names.strip().lower() != "all":
                raise JobscopeError('[metrics] %s = %r must be a list of metric names'
                                    ' or the string "all"' % (view, names))
            resolved[view] = tuple(gpu_catalog.all_specs)
            resolved["host_" + view] = tuple(cpu_module.default_view("extended"))
            continue
        if not isinstance(names, (list, tuple)):
            raise JobscopeError('[metrics] %s must be a list of metric names or "all",'
                               " not %r" % (view, names))
        # Sorted by family, so one list can name CPU%/MEM% beside the GPU columns: the
        # two are collected by different queries and have to reach different callers,
        # but a site writing the list should not have to know that.
        #
        # One name resolves in both catalogs -- `mem` is the DCGM key for GMEM_GB and
        # the cgroup key for MEM% -- and picking a side would be a coin toss that reads
        # as working. So it is an error that names the two unambiguous spellings.
        both = [n for n in names
                if spec_named(n) is not None and cpu_module.spec_named(n) is not None]
        if both:
            raise JobscopeError(
                "[metrics] %s: %s names both %s and %s. Write %s for the GPU column or"
                " %s for the host one."
                % (view, ", ".join(repr(n) for n in both),
                   spec_named(both[0]).header, cpu_module.spec_named(both[0]).header,
                   repr(spec_named(both[0]).header.lower()),
                   repr(cpu_module.spec_named(both[0]).header.lower())))
        gpu_names = [n for n in names if spec_named(n) is not None]
        host_names = [n for n in names if cpu_module.spec_named(n) is not None]
        strays = [n for n in names if n not in gpu_names and n not in host_names]
        if strays:
            # Named, not ignored: a typo would otherwise read as a metric the
            # exporter simply did not have, which is indistinguishable from working.
            raise JobscopeError(
                "[metrics] %s names no metric %s; the catalog is %s"
                % (view, ", ".join(repr(n) for n in strays),
                   ", ".join(tuple(gpu_catalog.names)
                                + tuple(cpu_module.catalog().names))))
        if not names:
            raise JobscopeError("[metrics] %s is empty; omit it to keep the built-in"
                                " list, or name at least one metric" % view)
        # An absent side keeps its built-in list rather than emptying: a list that
        # names only GPU metrics is narrowing the profiling block, not asking for a
        # table with no CPU% in it.
        if gpu_names:
            resolved[view] = tuple(specs_named(gpu_names))
        if host_names:
            resolved["host_" + view] = tuple(cpu_module.specs_named(host_names))
    return Metrics(**resolved)


def _palette(table: Mapping) -> Palette:
    """``[colors]`` -> a :class:`Palette`, with every value checked."""
    unknown = [key for key in table if key not in COLOR_ROLES]
    if unknown:
        raise JobscopeError("[colors] has no %s; it takes %s"
                            % (", ".join(sorted(unknown)), ", ".join(COLOR_ROLES)))
    colors = {}
    for role, raw in table.items():
        colour = str(raw).strip().lower()
        if not valid_colour(colour):
            raise JobscopeError(
                "[colors] %s = %r is not a colour; it takes %s, or color(N) for the"
                " 256-colour palette" % (role, raw, ", ".join(sorted(_SGR_NAMED))))
        colors[role] = colour
    return Palette(colors=colors)


def _known_percent_headers() -> frozenset:
    """Every %-column a metric key may name, for typo-checking the config.

    Both catalogs are imported inside the function on purpose: at module scope
    either would close the cycle config -> dcgm/cpu -> prometheus -> config.

    The host columns come from the cgroup catalog rather than a hand-kept pair, so
    a metric added there is threshold-tunable the moment it exists. It used to be
    literally ``("CPU%", "MEM%")``, which meant any new host column was silently
    dropped from a config with a note saying it was unknown.

    Both read the *resolved* catalog, not the built-in declaration: a site metric
    from ``[metrics.cgroup.<name>]`` lands on the catalog, and reading the
    declaration would put that name back in the "unknown" bucket this exists to
    keep it out of.
    """
    from .cpu import catalog as host_catalog
    from .dcgm import catalog, columns_for
    return frozenset([header for _key, header, _dec
                      in columns_for(catalog().all_specs)
                      if header.endswith("%")]
                     + [header for header in host_catalog().headers
                        if header.endswith("%")])


def _edges(raw) -> Dict[str, float]:
    """``[thresholds] edges = [2, 10, 20, 40]`` -> the four defaults, or ``{}``.

    The one-line form of the ladder, for a site that wants the same bands everywhere.
    It seeds *both* views, which the per-view tables then override -- so the whole
    of ``[thresholds.summary.*]`` and ``[thresholds.timeslice.*]`` collapses to one
    line when a site is not treating a window differently from a whole job.

    A list rather than four named keys because the order *is* the meaning: the edges
    have to be non-decreasing, and reading them left to right is how you check that.
    ``_check_order`` still verifies it per view once the tables are merged in.
    """
    if raw is None:
        return {}
    example = "edges = [%s]" % ", ".join("%g" % DEFAULT_BANDS[k] for k in EDGE_KEYS)
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise JobscopeError("[thresholds] edges must be a list of %d numbers -- %s"
                            % (len(EDGE_KEYS), example))
    if len(raw) != len(EDGE_KEYS):
        raise JobscopeError(
            "[thresholds] edges takes %d numbers, one per band edge (%s), not %d"
            " -- %s" % (len(EDGE_KEYS), ", ".join(EDGE_KEYS), len(raw), example))
    out = {}
    for key, value in zip(EDGE_KEYS, raw):
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            raise JobscopeError("[thresholds] edges: %r is not a number -- %s"
                                % (value, example))
    return out


def _band_table(table: Mapping, view: str, floors: Mapping,
                vote: Optional[Tuple[str, ...]], ceilings: Mapping,
                base: Optional[Mapping[str, float]] = None) -> Thresholds:
    """One view's ``[thresholds.<view>]`` block -> a :class:`Thresholds`.

    Each ``[thresholds.<view>.<edge>]`` sub-table gives that edge a ``default`` plus
    any per-metric overrides. Absent entirely, the view keeps ``base`` -- and absent
    that, :data:`DEFAULT_BANDS`. It never falls back to the other view.

    ``base`` is ``[thresholds] edges``, shared by both views. Four layers, narrowest
    last: DEFAULT_BANDS, then ``edges``, then this view's ``default``, then this
    view's per-metric entries.
    """
    where = "[thresholds.%s]" % view
    if not isinstance(table, Mapping):
        raise JobscopeError("%s must be a table of edge names (%s), not a value"
                            % (where, ", ".join(EDGE_KEYS)))
    unknown = [key for key in table if key not in EDGE_KEYS]
    if unknown:
        raise JobscopeError(
            "%s has no edge %s; it takes %s, each its own table -- e.g.\n"
            "  [thresholds.%s.wasteful]\n  default = 2\n  gpu = 2\n  cpu = 5"
            % (where, ", ".join(sorted(unknown)), ", ".join(EDGE_KEYS), view))

    defaults, by_metric = dict(base or {}), {}
    for key in EDGE_KEYS:
        if key not in table:
            continue
        edges, at = table[key], "[thresholds.%s.%s]" % (view, key)
        if not isinstance(edges, Mapping):
            # The likely migration slip: scoping the old flat block by one level
            # instead of two, leaving `wasteful = 2` where a table belongs.
            raise JobscopeError(
                "%s must be a table of per-metric cutoffs, not a bare value --"
                " write\n  %s\n  default = %s" % (at, at, edges))
        for name, raw in edges.items():
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise JobscopeError("%s: %s = %r is not a number" % (at, name, raw))
            if name == "default":
                defaults[key] = value
            else:
                by_metric.setdefault(metric_header(name), {})[key] = value

    known = _known_percent_headers()
    strays = sorted(h for h in by_metric if h not in known)
    if strays:
        # The %-normalisation would otherwise turn a typo into a silently ignored
        # column: `gpuu = 2` becomes GPUU%, which nothing ever asks about.
        print("note: %s names no metric %s; ignoring (the names are the column"
              " headers, lowercase and without the %%, e.g. gpu, cpu, sm_act)"
              % (where, ", ".join(strays)), file=sys.stderr)
        for stray in strays:
            del by_metric[stray]

    bands = Thresholds(defaults=defaults, by_metric=by_metric, floors=floors,
                       vote=vote, ceilings=ceilings)
    _check_order(bands, where)
    return bands


def _check_order(bands: Thresholds, where: str) -> None:
    """Reject band edges that do not increase, per metric.

    Checked on the *resolved* edges rather than the lines as written, because a
    partial override composes with the defaults: `[.inefficient] cpu = 1` against
    the default `wasteful = 2` leaves CPU% with (2, 1, 20, 40), where nothing can
    ever be inefficient and the printed range reads "2-1%". Equal neighbours are
    allowed -- they collapse a band, which is a coherent thing to ask for.
    """
    for header in [""] + sorted(bands.by_metric):
        edges = bands.edges(header)
        if list(edges) != sorted(edges):
            raise JobscopeError(
                "%s: %s edges must not decrease, but %s are %s -- a band between two"
                " of them can never be reached"
                % (where, "the default" if not header else header,
                   "/".join(EDGE_KEYS), "/".join("%g" % e for e in edges)))


def _discover_site_jobstats_dir() -> Optional[str]:
    """Directory of the ``jobstats`` binary on ``PATH``, expected to hold the
    site ``config.py`` (with ``PROM_SERVER``); None when jobstats isn't on PATH.

    Keying on the jobstats binary keeps jobscope site-agnostic (nothing
    site-specific ships in the package) and is a strong signal: we only import
    a ``config.py`` that sits beside a real jobstats install the user already has,
    never a stray file at a fixed system path.
    """
    path = shutil.which("jobstats")
    return os.path.dirname(path) if path else None


def resolve_prometheus(cfg: Config) -> Tuple[str, int]:
    """Return ``(url, sampling_period)`` for Prometheus.

    When no URL is configured directly, the site jobstats config supplies it: an
    explicit ``site_prom_config_path`` (which must import cleanly), otherwise
    the directory of the ``jobstats`` binary auto-discovered on ``PATH`` (skipped
    silently when it holds no usable config). Raises :class:`JobscopeError` with
    actionable guidance when none is available. The URL can embed a credential,
    so callers must never log or print it -- :func:`redact_url` exists for the one
    caller that has to show the user which endpoint it is talking to.
    """
    url, sampling_period, _source = _resolve_endpoint(cfg)
    return url, sampling_period


def endpoint_source(cfg: Config) -> str:
    """Where the endpoint came from, for ``jobscope config`` to name.

    A separate view onto :func:`_resolve_endpoint` rather than a third return value,
    because ``resolve_prometheus``'s 2-tuple is what every caller and eight tests
    read. Which of the three sources answered matters as much as the value when an
    endpoint is wrong: editing the config file cannot fix a stale $JOBSCOPE_PROM_URL,
    and neither touches the jobstats config.py the URL may really be coming from.
    """
    try:
        return _resolve_endpoint(cfg)[2]
    except JobscopeError:
        return ""


def _resolve_endpoint(cfg: Config) -> Tuple[str, int, str]:
    """``(url, sampling_period, source)`` -- the one implementation of the search."""
    url = cfg.prometheus_url
    sampling_period = cfg.sampling_period
    source = cfg.prometheus_from
    if not url:
        explicit = cfg.site_prom_config_path
        site_path = explicit or _discover_site_jobstats_dir()
        if site_path:
            site_url, site_sp = _import_site_prometheus(site_path, required=bool(explicit))
            if site_url:
                url = site_url
                source = "%s/config.py (jobstats%s)" % (
                    site_path.rstrip("/"), "" if explicit else ", auto-discovered")
                if site_sp and not cfg.sampling_period_explicit:
                    sampling_period = int(site_sp)
    if not url:
        raise JobscopeError(_no_endpoint_message(cfg))
    return url, sampling_period, source


def _import_site_prometheus(config_path: str,
                            required: bool = True) -> Tuple[Optional[str], Optional[int]]:
    """Read ``(PROM_SERVER, SAMPLING_PERIOD)`` from a jobstats ``config.py``.

    The file at ``<config_path>/config.py`` is loaded directly by path, so it
    never depends on ``sys.path`` order and never shadows (or is shadowed by)
    another module named ``config``. With ``required=False`` a missing or
    unimportable file yields ``(None, None)`` instead of raising, used for the
    automatic default path, which must not break jobscope where it is absent.
    """
    cfg_file = os.path.join(config_path, "config.py")
    if not os.path.isfile(cfg_file):
        if required:
            raise JobscopeError("no jobstats config.py found in %r" % config_path)
        return None, None
    try:
        spec = importlib.util.spec_from_file_location("_jobscope_site_config", cfg_file)
        site = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(site)
    except Exception as exc:
        if required:
            raise JobscopeError(
                "could not import the site jobstats config from %r: %s" % (config_path, exc))
        return None, None
    return getattr(site, "PROM_SERVER", None), getattr(site, "SAMPLING_PERIOD", None)


def _read_credentials(spec: str, config_path: Path) -> str:
    """``user:token`` from the file named by ``[prometheus] credentials_file``.

    A relative path resolves against the config file's own directory, not the working
    directory: the point of the key is that a repo-local config can say
    ``secrets/prom_creds`` and stay true when cron runs it from ``$HOME``.
    """
    path = Path(os.path.expanduser(spec))
    if not path.is_absolute():
        path = config_path.parent / path
    if not path.exists():
        raise JobscopeError("[prometheus] credentials_file not found: %s" % path)

    # A credential readable by the rest of the cluster is not a credential. An error
    # rather than the note this module uses elsewhere, because carrying on would mean
    # authenticating with a secret we have just established that everyone can read.
    mode = path.stat().st_mode
    if mode & 0o077:
        raise JobscopeError(
            "[prometheus] credentials_file %s is readable by group or other (mode %s)."
            " It holds a Prometheus credential: chmod 600 %s" % (path, oct(mode & 0o777), path))

    text = path.read_text().strip()
    if not text:
        raise JobscopeError("[prometheus] credentials_file is empty: %s" % path)
    if "://" in text:
        # The whole URL in the file is the obvious misreading, and it would otherwise be
        # spliced into a netloc and produce an unresolvable host rather than an error.
        raise JobscopeError(
            "[prometheus] credentials_file %s looks like a URL. It should hold only the"
            " credential, as USER:TOKEN on one line; the endpoint goes in [prometheus] url."
            % path)
    if "@" in text:
        raise JobscopeError(
            "[prometheus] credentials_file %s must not contain '@' -- write USER:TOKEN"
            " only, without the host." % path)
    return text


def _prometheus_url(prom: Mapping, env: Mapping[str, str],
                    config_path: Path) -> Tuple[Optional[str], str]:
    """``(url, source)`` from ``[prometheus]``, splicing in a credentials_file.

    Precedence is unchanged at the top: ``$JOBSCOPE_PROM_URL`` carries a whole URL and
    wins outright, since it is the one form a single cron line or sbatch script can pass.
    ``credentials_file`` composes with ``url`` below it, which is what lets the endpoint
    be committed while the secret is not.
    """
    if env.get(PROM_URL_ENV):
        if prom.get("credentials_file"):
            # Not silently: the file is the thing an admin just went to the trouble of
            # creating, and an inherited export in a login shell is exactly how it ends
            # up ignored without anyone noticing.
            print("note: $%s is set, so [prometheus] credentials_file is not being read."
                  " Unset it to use the file." % PROM_URL_ENV, file=sys.stderr)
        return env[PROM_URL_ENV], "$" + PROM_URL_ENV

    url = prom.get("url")
    creds = prom.get("credentials_file")
    if not creds:
        return (url or None), ("[prometheus] url" if url else "")
    if not url:
        raise JobscopeError(
            "[prometheus] credentials_file is set but url is not. The file holds only the"
            " credential; the endpoint it belongs to goes in [prometheus] url.")

    scheme, sep, rest = str(url).partition("://")
    if not sep:
        # Only reached with credentials_file set. Without a scheme there is no netloc to
        # splice into, and guessing one would build a URL the admin never wrote.
        raise JobscopeError(
            "[prometheus] url must include a scheme (https://...) to use credentials_file;"
            " got %r" % str(url))
    if "@" in rest.partition("/")[0]:
        raise JobscopeError(
            "[prometheus] url already embeds a credential and credentials_file is also"
            " set. Keep the secret in one place: strip the 'USER:TOKEN@' from url.")

    spliced = "%s://%s@%s" % (scheme, _read_credentials(str(creds), config_path), rest)
    return spliced, "[prometheus] url + credentials_file"


def redact_url(url: str) -> str:
    """``url`` with any embedded credential replaced, safe to print.

    Grafana Cloud and Mimir endpoints commonly carry ``user:token@`` in the
    netloc, so the configured URL is a secret even though it looks like an
    address. Everything else is kept, because the host and path are exactly what
    someone diagnosing a wrong endpoint needs to see.

    Only the userinfo is removed -- a query string is left alone, since none of
    the supported forms put a credential there and blanking it would hide a real
    misconfiguration.
    """
    scheme, sep, rest = str(url).partition("://")
    if not sep:
        return str(url)
    netloc, slash, path = rest.partition("/")
    if "@" not in netloc:
        return str(url)
    _userinfo, _, host = netloc.rpartition("@")
    return "%s://%s%s%s" % (scheme, "***@" + host, slash, path)


def _no_endpoint_message(cfg: Config) -> str:
    target = cfg.source_path or default_config_path()
    return (
        "no Prometheus endpoint configured; the GPU and DCGM views require one.\n"
        "(No 'jobstats' binary with a usable config.py was found on your PATH.)\n"
        "Fix any one of:\n"
        "  - set the JOBSCOPE_PROM_URL environment variable, or\n"
        '  - add [prometheus] url = "https://.../api/prom" to %s, or\n'
        "  - set [prometheus] site_prom_config_path to a dir holding a jobstats config.py.\n"
        "The offline --cpu / --cgpu views need no Prometheus." % target)


_active: Optional[Config] = None


def get_config() -> Config:
    """Return the process-wide Config, loading it from disk on first use."""
    global _active
    if _active is None:
        _active = load_config()
    return _active


def set_config(cfg: Config) -> None:
    """Install a Config (used by the CLI after --config, and by tests)."""
    global _active
    _active = cfg


def reset_config() -> None:
    """Forget the cached Config so the next :func:`get_config` reloads it."""
    global _active
    _active = None


def example_config_text() -> str:
    """The bundled config.example.toml as text."""
    return (importlib.resources.files("jobscope") / "config.example.toml").read_text()
