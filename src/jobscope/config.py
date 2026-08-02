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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Optional, Tuple

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
# The `finished` window when no -D/-N/-S/-E is given.
DEFAULT_DAYS = 1
# Which job endings `finished` reports without -t. See sacct.STATE_GROUPS.
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
# "best", and the order --classify lists its categories in. It lives here because
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
# POWER_W is watts, not percent, so it is not tiered at all -- it is a floor, and a
# GPU below it is idle. That is the one signal a duty cycle cannot fake: a job
# spinning on a trivial kernel reads busy on GPU% and draws idle watts. 100 sits in
# the measured gap -- on kempner_eng the idle jobs drew 73-74 W with GPU% 0 and
# SM_ACT% 0.0, the next values were 99-101 W, and the median was 289 W against a
# 573 W maximum.
DEFAULT_POWER_W = 100.0

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

# %-metrics that do not come from DCGM, so a validator built from that catalog
# would not know them. Both are read from the sacct blob (see report._BLOB_HEADERS)
# and `cpu` is the single likeliest key a site sets.
_BLOB_PERCENT_HEADERS = ("CPU%", "MEM%")

# The colour each tier is painted, and the one role that is not a tier: an entry on
# a Wasteful row whose job ran longer than [defaults] long_running. Two tiers
# sharing a colour is the default, not a requirement -- a site wanting five distinct
# colours (for a colourblind-safe palette, say) can set five.
DEFAULT_COLORS = {"wasteful": "red", "inefficient": "red", "improvement": "yellow",
                  "average": "green", "good": "green", "long_running": "red"}
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
        --classify heading has a tier, the Wasteful rows have neither.
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

    Here rather than in live.py, which is where it started: config needs it too, for
    the duration-valued ``[defaults]`` keys, and config cannot import live.
    """
    match = re.match(r"^(\d+)([smhd])$", str(text).strip())
    if not match:
        raise JobscopeError(
            "invalid duration %r: use a count and a unit, e.g. '30s', '5m', '2h', '7d'" % text)
    value, unit = int(match.group(1)), match.group(2)
    return value * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def metric_header(key: str) -> str:
    """Config spelling of a metric -> its column header.

    ``gpu``, ``GPU`` and ``"GPU%"`` all mean ``GPU%``; ``sm_act`` means ``SM_ACT%``.
    Sites write the short lowercase form, which is how the metrics get talked about,
    and the tables are keyed on the header the rest of the tool uses.
    """
    header = str(key).strip().upper()
    return header if header.endswith("%") else header + "%"


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
    power_w: float = DEFAULT_POWER_W
    # Per-GPU-model watt floors, keyed by the exporter's own model string. Idle draw
    # is hardware, not policy: measured on one cluster it ran from 27 W on a V100 to
    # 165 W on an RTX PRO 6000, so a single number is wrong at one end or the other.
    # Empty by default -- see config.example.toml for how to derive a site's values.
    power_w_by_model: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fill any edge the caller left out, so a partial ``defaults`` -- a TOML that
        # names only `wasteful`, or a caller passing one edge -- cannot KeyError deep
        # inside edge(). object.__setattr__ because the dataclass is frozen.
        if set(self.defaults) != set(DEFAULT_BANDS):
            object.__setattr__(self, "defaults", {**DEFAULT_BANDS, **self.defaults})

    def floor_for(self, model: Optional[str]) -> float:
        """The watt floor for ``model``, or the global one.

        Matched on the exporter's exact string, which is what is available where the
        grading happens; normalising model names would be a second thing to get wrong.
        """
        if model:
            return float(self.power_w_by_model.get(model, self.power_w))
        return self.power_w

    def for_model(self, model: Optional[str]) -> "Thresholds":
        """These thresholds with ``POWER_W``'s floor resolved for one card.

        Binding the model once beats handing it to every grading call. The floor was
        previously an optional argument on four separate methods, and the sites that
        forgot it -- the printed cell, and the waste ledger behind the Worst rows --
        graded the same reading against the global floor while the tally used the
        card's. One 165 W sample came out red in the table and green in the cell.
        Resolved here, a caller cannot forget what it never passes.
        """
        floor = self.floor_for(model)
        return self if floor == self.power_w else replace(self, power_w=floor)

    def edge(self, key: str, header: str = "") -> float:
        """``header``'s value for edge ``key``: its own if set, else the default."""
        return self.by_metric.get(header, {}).get(key, self.defaults[key])

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
    # rather than typing every run. Validated against sacct.STATE_GROUPS.
    state: str = DEFAULT_STATE
    # How many jobs each Problem-jobs "Wasteful" row lists.
    worst_jobs: int = DEFAULT_WORST_JOBS
    # A Wasteful-row entry whose job ran at least this long is highlighted: hours of
    # idle hardware do not come back, where a short bad job costs little.
    long_running: str = DEFAULT_LONG_RUNNING


@dataclass(frozen=True)
class Metrics:
    """Which GPU/DCGM metrics each view collects and shows.

    GPU-side only. The CPU side is ``CPU%``/``MEM%`` and fixed, because ``cpu.py``
    has two queries and no spec catalog -- see its module docstring for why.

    Held as resolved :class:`~jobscope.dcgm.MetricSpec` lists rather than names, so
    every consumer reads one place and a name is validated once, at load.
    """

    summary: Tuple = ()      # the summary table's profiling block
    timeseries: Tuple = ()   # --ts / --plot_ts / --classify
    extended: Tuple = ()     # --dcgm / --ext

    def __post_init__(self) -> None:
        # Deferred so config stays importable without dcgm (which reaches
        # prometheus, and so back to config) -- see _known_percent_headers.
        from .dcgm import (ALL_SPECS, BLOB_BACKED_KEYS, DEFAULT_SPECS, KEY_SPECS,
                           METRICS, specs_named)
        for name, fallback in (("summary", DEFAULT_SPECS),
                               ("timeseries", KEY_SPECS),
                               ("extended", ALL_SPECS)):
            if not getattr(self, name):
                object.__setattr__(self, name, tuple(fallback))
        # The blob-backed metrics feed GPU% and GMEM%, which are *fixed* columns of
        # the summary and detail tables rather than part of the configurable
        # profiling block. A config that leaves them out is not asking for narrower
        # output, it is asking for two of its own columns to read "-" -- and only in
        # the running view, where they come from Prometheus rather than the blob. So
        # they are added back rather than obeyed. Not to `timeseries`: its CSV has no
        # fixed columns, so there a narrower list means exactly what it says.
        required = [spec for spec in METRICS if spec.key in BLOB_BACKED_KEYS]
        for name in ("summary", "extended"):
            listed = getattr(self, name)
            missing = [s for s in required if s.key not in {x.key for x in listed}]
            if missing:
                object.__setattr__(self, name, tuple(specs_named(
                    [s.key for s in listed] + [s.key for s in missing])))

    def live(self, which: str) -> Tuple:
        """``which``'s specs for the running view, which cannot use a counter delta.

        ENERGY_kWh is a difference over a finished window; the running view
        synthesizes a jobstats-shaped blob from a window that has not finished, so
        the number would be meaningless rather than merely partial.
        """
        return tuple(spec for spec in getattr(self, which) if spec.reducer != "delta")


@dataclass(frozen=True)
class Config:
    """Resolved jobscope settings."""

    prometheus_url: Optional[str]
    sampling_period: int
    sampling_period_explicit: bool
    site_jobstats_config_path: Optional[str]
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


def default_config_path(env: Optional[Mapping[str, str]] = None) -> Path:
    """Path jobscope reads when neither an explicit path nor $JOBSCOPE_CONFIG is set."""
    env = os.environ if env is None else env
    base = env.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "jobscope" / "config.toml"


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as fh:
        return _toml.load(fh)


def load_config(path: Optional[str] = None,
                env: Optional[Mapping[str, str]] = None) -> Config:
    """Build a Config from a TOML file plus environment overrides.

    Resolution order for the file: ``path`` argument, then ``$JOBSCOPE_CONFIG``,
    then :func:`default_config_path`. A file named explicitly (argument or env
    var) must exist; the default path may be absent, in which case built-in
    defaults apply. The Prometheus URL prefers ``$JOBSCOPE_PROM_URL`` over the file.
    """
    env = os.environ if env is None else env
    explicit = path if path is not None else env.get(CONFIG_ENV)
    if explicit:
        chosen = Path(explicit)
        if not chosen.exists():
            raise JobscopeError("jobscope config not found: %s" % chosen)
    else:
        chosen = default_config_path(env)
    data = _read_toml(chosen) if chosen.exists() else {}

    prom = data.get("prometheus") or {}
    thr = data.get("thresholds") or {}
    dfl = data.get("defaults") or {}

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
    named = [view for view in BAND_VIEWS if view in thr]
    if len(named) == 1:
        # Nothing is inherited between the two, by design -- so a config that tunes
        # one and forgets the other grades the same job differently depending on
        # whether --ts was passed. Say so once; it stops as soon as both are set.
        other = [view for view in BAND_VIEWS if view != named[0]][0]
        print("note: [thresholds.%s] is set but [thresholds.%s] is not, so the two"
              " views grade differently -- %s keeps the built-in edges (nothing is"
              " inherited between them). 'jobscope config' prints both."
              % (named[0], other, other), file=sys.stderr)
    bands = {view: _band_table(thr.get(view) or {}, view, power_w, by_model)
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
    )
    return Config(
        prometheus_url=(env.get(PROM_URL_ENV) or prom.get("url")) or None,
        sampling_period=int(prom.get("sampling_period", DEFAULT_SAMPLING_PERIOD)),
        sampling_period_explicit="sampling_period" in prom,
        site_jobstats_config_path=prom.get("site_jobstats_config_path"),
        thresholds=bands["summary"],
        defaults=defaults,
        source_path=chosen if chosen.exists() else None,
        timeslice_thresholds=bands["timeslice"],
        metrics=_metrics(data.get("metrics") or {}),
        palette=_palette(data.get("colors") or {}),
    )


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
    from .sacct import states_for
    name = str(raw).strip().lower()
    try:
        states_for(name)
    except JobscopeError as exc:
        raise JobscopeError("[defaults] state = %r: %s" % (raw, exc))
    return name


def _metrics(table: Mapping) -> Metrics:
    """``[metrics]`` -> resolved spec lists.

    Each key is a list of metric names, or the string ``"all"`` for the whole
    catalog. Absent, a view keeps its built-in list (see :meth:`Metrics.__post_init__`).
    """
    from .dcgm import ALL_SPECS, METRIC_NAMES, spec_named, specs_named
    unknown = [key for key in table if key not in ("summary", "timeseries", "extended")]
    if unknown:
        raise JobscopeError(
            "[metrics] has no %s; it takes summary, timeseries, extended"
            % ", ".join(sorted(unknown)))
    resolved = {}
    for view, names in table.items():
        if isinstance(names, str):
            if names.strip().lower() != "all":
                raise JobscopeError('[metrics] %s = %r must be a list of metric names'
                                    ' or the string "all"' % (view, names))
            resolved[view] = tuple(ALL_SPECS)
            continue
        if not isinstance(names, (list, tuple)):
            raise JobscopeError('[metrics] %s must be a list of metric names or "all",'
                               " not %r" % (view, names))
        strays = [n for n in names if spec_named(n) is None]
        if strays:
            # Named, not ignored: a typo would otherwise read as a metric the
            # exporter simply did not have, which is indistinguishable from working.
            raise JobscopeError(
                "[metrics] %s names no metric %s; the catalog is %s"
                % (view, ", ".join(repr(n) for n in strays), ", ".join(METRIC_NAMES)))
        if not names:
            raise JobscopeError("[metrics] %s is empty; omit it to keep the built-in"
                                " list, or name at least one metric" % view)
        resolved[view] = tuple(specs_named(names))
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

    The DCGM catalog is imported inside the function on purpose: at module scope it
    would close the cycle config -> dcgm -> prometheus -> config. The two blob-backed
    columns are not in that catalog and have to be added by hand.
    """
    from .dcgm import ALL_SPECS, columns_for
    return frozenset([header for _key, header, _dec in columns_for(ALL_SPECS)
                      if header.endswith("%")] + list(_BLOB_PERCENT_HEADERS))


def _band_table(table: Mapping, view: str, power_w: float,
                by_model: Mapping[str, float]) -> Thresholds:
    """One view's ``[thresholds.<view>]`` block -> a :class:`Thresholds`.

    Each ``[thresholds.<view>.<edge>]`` sub-table gives that edge a ``default`` plus
    any per-metric overrides. Absent entirely, the view keeps :data:`DEFAULT_BANDS`
    -- it never falls back to the other view.
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

    defaults, by_metric = {}, {}
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

    bands = Thresholds(defaults=defaults, by_metric=by_metric, power_w=power_w,
                       power_w_by_model=by_model)
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
    explicit ``site_jobstats_config_path`` (which must import cleanly), otherwise
    the directory of the ``jobstats`` binary auto-discovered on ``PATH`` (skipped
    silently when it holds no usable config). Raises :class:`JobscopeError` with
    actionable guidance when none is available. The URL can embed a credential,
    so callers must never log or print it.
    """
    url = cfg.prometheus_url
    sampling_period = cfg.sampling_period
    if not url:
        explicit = cfg.site_jobstats_config_path
        site_path = explicit or _discover_site_jobstats_dir()
        if site_path:
            site_url, site_sp = _import_site_prometheus(site_path, required=bool(explicit))
            if site_url:
                url = site_url
                if site_sp and not cfg.sampling_period_explicit:
                    sampling_period = int(site_sp)
    if not url:
        raise JobscopeError(_no_endpoint_message(cfg))
    return url, sampling_period


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


def _no_endpoint_message(cfg: Config) -> str:
    target = cfg.source_path or default_config_path()
    return (
        "no Prometheus endpoint configured; the GPU and DCGM views require one.\n"
        "(No 'jobstats' binary with a usable config.py was found on your PATH.)\n"
        "Fix any one of:\n"
        "  - set the JOBSCOPE_PROM_URL environment variable, or\n"
        '  - add [prometheus] url = "https://.../api/prom" to %s, or\n'
        "  - set [prometheus] site_jobstats_config_path to a dir holding a jobstats config.py.\n"
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
