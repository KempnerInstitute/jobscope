"""DCGM profiling metric catalog and the Prometheus join that populates it.

These metrics are not in the jobstats summary in sacct; they come from the same Prometheus that
jobstats uses. Each value is the time-average (or max/delta) over the job's
``[start, end]`` window. GPUs are joined to the job by UUID via the
``nvidia_gpu_jobId`` companion series, because the DCGM ``gpu`` index and Slurm
``minor_number`` disagree and only the UUID is stable across the two exporters.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from . import config, source
from .prometheus import PrometheusClient
from .slurm import JobRecord

# The join between Slurm's world and the GPU exporters' is a series whose *value*
# is the job id, because neither nvidia_gpu_* nor DCGM_FI_* carries a jobid label.
# Which series that is, is a site convention -- see config.Site.gpu_job_join, read
# through config.gpu_join() so a [site] override takes effect.


@dataclass(frozen=True)
class MetricSpec:
    """One DCGM metric and how to query, reduce, and display it.

    ``scale`` is the display multiplier (fraction to percent, MiB to GiB, mJ to
    kWh); ``decimals`` of 0 renders an integer; ``group`` is ``default`` (always
    shown) or ``all`` (only with the extended catalog). ``reducer`` collapses the
    window (avg | max | delta); ``agg`` reduces across a job's GPUs for the overall
    row (mean | sum | max); ``uuid_label`` is the Prometheus label holding the GPU
    UUID. Those three default to the common case and are set only where a metric
    differs.

    ``roles`` is what the metric is *for* -- see :mod:`jobscope.metrics`, which
    reads it instead of the hand-kept header lists that used to say the same thing
    in five places. ``slug`` and ``tag`` are its short forms for a row label and a
    combined-share suffix; both derive from the header and are set only where that
    derivation reads badly.

    ``family`` and ``provides`` are what let one column have more than one source.
    ``family`` names the exporter (see :data:`jobscope.config.FAMILIES`) and is no
    longer inferred from ``uuid_label`` -- the two are checked against each other
    instead, since a mismatch yields a response whose rows cannot be attributed to
    a card. ``provides`` is the header this spec is a *candidate* for, so ``duty``
    (NVML) and ``duty_dcgm`` (DCGM) can both offer ``GPU%`` under different keys;
    :mod:`jobscope.source` picks one per column and only the winner goes live,
    which is what keeps headers unique. ``group`` is read per family: ``default``
    means default *for this source*, not across the catalog.
    """

    key: str
    header: str
    metric: str
    scale: float
    decimals: int
    group: str
    reducer: str = "avg"
    agg: str = "mean"
    uuid_label: str = "UUID"
    show: bool = True   # False = queried only to feed a derived column
    roles: FrozenSet[str] = frozenset()
    slug: str = ""      # row-label form; defaults to the header without its "%"
    tag: str = ""       # share-tag form; defaults to the lowercased slug
    family: str = "dcgm"
    provides: str = ""  # column this is a candidate for; defaults to the header

    def __post_init__(self) -> None:
        # The label a family's series carry the UUID in is a property of the
        # exporter, not a free choice: nvml publishes lowercase, DCGM uppercase.
        # Checked rather than derived so a spec cannot claim one and query the
        # other -- that combination returns rows silently keyed to nothing.
        expected = "uuid" if self.family == "nvml" else "UUID"
        if self.uuid_label != expected:
            raise ValueError(
                "%s: family %r uses the %r label, not %r"
                % (self.key, self.family, expected, self.uuid_label))

    @property
    def column(self) -> str:
        """The header this spec is a candidate to serve."""
        return self.provides or self.header

    @property
    def label(self) -> str:
        """Short row-label form, e.g. ``SM_ACT%`` -> ``SM``."""
        return self.slug or self.header.rstrip("%")

    @property
    def share_tag(self) -> str:
        """Suffix in a combined share, e.g. ``35%gpu+24%cpu``."""
        return self.tag or self.label.lower()


METRICS: List[MetricSpec] = [
    MetricSpec("duty", "GPU%", "nvidia_gpu_duty_cycle", 1, 0, "default", uuid_label="uuid",
               family="nvml", roles=frozenset({"worst", "resource"})),
    # The other candidate for GPU%, from dcgm-exporter. Same quantity, same 0-100
    # scale, and measured to agree with the nvidia exporter to a mean of 3.6 points
    # per card at one instant -- which is scrape offset, not disagreement about what
    # is being counted. Immediately after `duty` so whichever wins lands in the same
    # column position; jobscope.source picks one, never both.
    MetricSpec("duty_dcgm", "GPU%", "DCGM_FI_DEV_GPU_UTIL", 1, 0, "default",
               provides="GPU%", roles=frozenset({"worst", "resource"})),
    MetricSpec("smact", "SM_ACT%", "DCGM_FI_PROF_SM_ACTIVE", 100, 1, "default",
               roles=frozenset({"worst"}), slug="SM"),
    MetricSpec("tensor", "TENSOR%", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", 100, 1, "default"),
    MetricSpec("dram", "DRAM%", "DCGM_FI_PROF_DRAM_ACTIVE", 100, 1, "default"),
    # POWER_W is a watt reading, not a percentage, so it never votes on its own --
    # it only ever pulls a verdict down. See classify()'s cap.
    MetricSpec("power", "POWER_W", "DCGM_FI_DEV_POWER_USAGE", 1, 0, "default",
               roles=frozenset({"worst", "cap"}), slug="POWER", tag="pw"),
    # The nvidia exporter's board watts, in milliwatts. Here so choosing nvml does
    # not cost the POWER_W floor, which is what pulls a verdict below its band --
    # without a candidate, an nvml-only site would silently lose that check.
    MetricSpec("power_nvml", "POWER_W", "nvidia_gpu_power_usage_milliwatts", 1e-3, 0,
               "default", uuid_label="uuid", family="nvml", provides="POWER_W",
               roles=frozenset({"worst", "cap"}), slug="POWER", tag="pw"),
    # OCC% sits here, right after the default group, so the extended catalog's
    # column order keeps the default specs a contiguous prefix -- it is the first
    # "all"-only metric rather than interspersed among the default ones.
    MetricSpec("occ", "OCC%", "DCGM_FI_PROF_SM_OCCUPANCY", 100, 1, "all"),
    MetricSpec("engine", "ENGINE%", "DCGM_FI_PROF_GR_ENGINE_ACTIVE", 100, 1, "all"),
    MetricSpec("hmma", "HMMA%", "DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE", 100, 1, "all"),
    MetricSpec("imma", "IMMA%", "DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE", 100, 1, "all"),
    MetricSpec("dfma", "DFMA%", "DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE", 100, 1, "all"),
    MetricSpec("fp16", "FP16%", "DCGM_FI_PROF_PIPE_FP16_ACTIVE", 100, 1, "all"),
    MetricSpec("fp32", "FP32%", "DCGM_FI_PROF_PIPE_FP32_ACTIVE", 100, 1, "all"),
    MetricSpec("fp64", "FP64%", "DCGM_FI_PROF_PIPE_FP64_ACTIVE", 100, 1, "all"),
    MetricSpec("memcp", "MEMCP%", "DCGM_FI_DEV_MEM_COPY_UTIL", 1, 0, "all"),
    MetricSpec("pwrmax", "PWRmax_W", "DCGM_FI_DEV_POWER_USAGE", 1, 0, "all", reducer="max", agg="max"),
    MetricSpec("energy", "ENERGY_kWh", "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", 1 / 3.6e9, 3, "all",
               reducer="delta", agg="sum"),
    MetricSpec("fbused", "FB_USED_GB", "DCGM_FI_DEV_FB_USED", 1 / 1024, 1, "all"),
    MetricSpec("fbfree", "FB_FREE_GB", "DCGM_FI_DEV_FB_FREE", 1 / 1024, 1, "all"),
    MetricSpec("fbrsvd", "FB_RSVD_GB", "DCGM_FI_DEV_FB_RESERVED", 1 / 1024, 1, "all"),
    MetricSpec("pcietx", "PCIE_TX_MBs", "DCGM_FI_PROF_PCIE_TX_BYTES", 1e-6, 1, "all"),
    MetricSpec("pcierx", "PCIE_RX_MBs", "DCGM_FI_PROF_PCIE_RX_BYTES", 1e-6, 1, "all"),
    MetricSpec("nvlink", "NVLINK_MBs", "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL", 1 / 1024, 1, "all"),
    MetricSpec("smclk", "SMCLK_MHz", "DCGM_FI_DEV_SM_CLOCK", 1, 0, "all"),
    MetricSpec("memclk", "MEMCLK_MHz", "DCGM_FI_DEV_MEM_CLOCK", 1, 0, "all"),
    MetricSpec("temp", "TEMP_C", "DCGM_FI_DEV_GPU_TEMP", 1, 0, "all"),
    MetricSpec("temp_nvml", "TEMP_C", "nvidia_gpu_temperature_celsius", 1, 0, "all",
               uuid_label="uuid", family="nvml", provides="TEMP_C"),
    MetricSpec("memtemp", "MEMTEMP_C", "DCGM_FI_DEV_MEMORY_TEMP", 1, 0, "all"),
    MetricSpec("enc", "ENC%", "DCGM_FI_DEV_ENC_UTIL", 1, 0, "all"),
    MetricSpec("dec", "DEC%", "DCGM_FI_DEV_DEC_UTIL", 1, 0, "all"),
    # NVML GPU memory, the pair jobstats reports as "GPU memory usage per node -
    # maximum used/total". Peaked, not averaged, so it is comparable to jobstats.
    # Named GMEM_* to match the jobstats-derived GMEM% of the summary and detail views:
    # a bare MEM% means HOST memory there, and reusing it for GPU memory both reads
    # as the wrong quantity and grades against the host threshold in plots.
    MetricSpec("mem", "GMEM_GB", "nvidia_gpu_memory_used_bytes", 1 / 1024 ** 3, 1, "default",
               reducer="max", agg="max", uuid_label="uuid", family="nvml",
               roles=frozenset({"memory"})),
    MetricSpec("memtot", "GMEM_TOTAL_GB", "nvidia_gpu_memory_total_bytes", 1 / 1024 ** 3, 1,
               "default", reducer="max", agg="max", uuid_label="uuid", family="nvml",
               show=False),
]


class JobGpuData(NamedTuple):
    """One job's GPU figures at all three levels.

    A NamedTuple rather than a bare tuple so the fields have names: every reader of this
    used to index ``[0]`` or ``[1]`` and the reader had to know which was which. Still a
    tuple, so those existing reads keep working unchanged.

    ``per_node`` exists because a node figure cannot be recovered downstream: it has to
    be pooled from the per-UUID values, and ``per_gpu`` has already collapsed MIG siblings
    onto a shared ``(node, minor)`` key. See :func:`pool_uuids`.
    """

    # Shared empty dicts, deliberately: an "absent" JobGpuData is only ever read, and
    # a NamedTuple cannot take a default_factory. Every populated instance is built with
    # its own dicts at the two construction sites, so nothing mutates these. A future
    # caller that wants to mutate one must construct it explicitly.
    overall: dict = {}
    per_gpu: dict = {}
    per_node: dict = {}


class Derived(NamedTuple):
    """A column computed from queried metrics rather than fetched from Prometheus."""

    key: str
    header: str
    decimals: int
    deps: Tuple[str, ...]   # metric keys it needs; absent -> the column is skipped
    fn: Callable[[Dict[str, Optional[float]]], Optional[float]]
    source: str             # what it is computed from, for --describe
    roles: FrozenSet[str] = frozenset()   # as MetricSpec.roles; see jobscope.metrics


def _gmem_percent(values: Dict[str, Optional[float]]) -> Optional[float]:
    """GPU memory used as a percentage of that GPU's own total."""
    used, total = values.get("mem"), values.get("memtot")
    if used is None or not total:
        return None
    return used / total * 100


# Columns computed from other metrics. Shared by the dcgm and running views so both
# render the same set; each declares the metric keys it needs, so it appears only
# where those were actually collected. ``fn`` receives a dict keyed by metric key,
# which callers storing values by header must build first -- see values_by_key.
DERIVED_COLUMNS: List[Derived] = [
    # `memory`: a capacity reading, not a utilization one. A job that fills the
    # card's memory and then computes nothing is idle, so this must never vote.
    Derived("gmempct", "GMEM%", 1, ("mem", "memtot"), _gmem_percent,
            "GMEM_GB / nvidia_gpu_memory_total", roles=frozenset({"memory"})),
]


def applicable_derived(specs: List[MetricSpec]) -> List[Derived]:
    """The derived columns whose input metrics are all present in ``specs``."""
    fetched = {spec.key for spec in specs}
    return [d for d in DERIVED_COLUMNS if fetched.issuperset(d.deps)]


def columns_for(specs: List[MetricSpec]) -> List[Tuple[str, str, int]]:
    """``(key, header, decimals)`` per displayed column, derived ones included.

    Hidden specs (``show=False``) are dropped, and each derived column is inserted
    next to the metric it is computed from rather than trailing at the far end.
    """
    cols = [(s.key, s.header, s.decimals) for s in specs if s.show]
    for derived in applicable_derived(specs):
        from_deps = [i for i, (key, _h, _d) in enumerate(cols) if key in derived.deps]
        cols.insert(max(from_deps) + 1 if from_deps else len(cols),
                    (derived.key, derived.header, derived.decimals))
    return cols


def values_by_key(specs: List[MetricSpec], by_header: Dict[str, float]
                  ) -> Dict[str, Optional[float]]:
    """Re-key one GPU's values from header to metric key, for a derived ``fn``."""
    return {spec.key: by_header.get(spec.header) for spec in specs}


# The --ts/--plot_ts/--eff default when --all-metrics is not given: a smaller,
# curated set than the default specs (which also carry the GPU memory pair) --
# deliberately narrower, for the time-series family specifically. Named by *column*
# rather than by key, because which key serves GPU% depends on the source.
_KEY_SPEC_COLUMNS = ("GPU%", "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W")


@dataclass(frozen=True)
class GpuCatalog:
    """The resolved GPU view: one winner per column, and everything derived from it.

    ``metrics`` is the *candidates* -- several may offer the same column from
    different exporters, so its headers are deliberately not unique. Every other
    field is the resolved view under ``preference``, which is what has unique headers
    and what every consumer reads. See :mod:`jobscope.source`.

    One frozen object rather than the dozen module globals this replaces. Each of
    those was a view of the catalog that ``set_preference()`` reassigned, so
    ``from .dcgm import DEFAULT_SPECS`` took a binding once at import and never saw
    the rebuild. That is not theoretical: report.py held ``DCGM_JOBSTATS_HEADERS``
    that way, so ``--gpu-source dcgm`` resolved GPU% to DCGM, said so on the Source
    line, and rendered the nvml value anyway. It cost an AST test to police.

    There is nowhere for that bug to live now. No derived name is importable, and a
    caller that does hold a catalog holds one whole consistent snapshot rather than
    one stale field of a set that moved on without it.
    """

    metrics: Tuple[MetricSpec, ...]
    preference: Tuple[str, ...]
    resolved: source.Resolution
    spec_by_header: Mapping[str, MetricSpec]
    all_specs: Tuple[MetricSpec, ...]
    default_specs: Tuple[MetricSpec, ...]
    key_specs: Tuple[MetricSpec, ...]
    # Quantities the jobstats summary in sacct already supplies, which the summary and
    # detail views render from it directly (GPU%, GMEM%, and GPU-MEM). Excluded from
    # those views' DCGM columns so a job does not get two columns for one number --
    # and for a finished job they would be the very same number, see _prefer_stored.
    #
    # Derived from the resolution rather than written out, so that naming an exporter
    # ahead of the jobstats summary genuinely moves these: under the default
    # preference this is exactly the ("duty", "mem", "memtot") it was once spelled as.
    jobstats_backed_keys: Tuple[str, ...]
    gpu_summary_specs: Tuple[MetricSpec, ...]
    headers: Tuple[str, ...]
    aliases: Mapping[str, MetricSpec]
    names: Tuple[str, ...]
    # Position in ``metrics``, so a resolved selection can be put back in catalog order.
    order: Mapping[str, int]


def _alias_table(metrics: Sequence[MetricSpec],
                 resolved: source.Resolution) -> Dict[str, MetricSpec]:
    """Every name a config may call a metric by -> its spec.

    Derived from the catalog rather than spelled out, so a metric added to
    ``metrics`` is nameable immediately. Three forms per spec: its ``key``
    (``duty``, ``smact``, ``power``), its lowercased ``header`` (``gpu%``,
    ``power_w``), and the header without a trailing ``%`` (``gpu``, ``sm_act``).
    That makes the short lowercase names ``[thresholds]`` already takes -- ``gpu``,
    ``sm_act``, ``dram`` -- work here too, which is what a reader expects.

    Keys name a *candidate*, so every candidate has one and ``duty`` and
    ``duty_dcgm`` stay separately nameable -- a per-source view list has to be able
    to say which provider it means. The header forms name a *column*, so they
    resolve to whichever candidate currently serves it: ``gpu`` is the active GPU%,
    whatever source that is. Without that split the two GPU% candidates would both
    claim ``gpu``.

    A collision within either kind would silently shadow one spec with another, so
    it is an error when the catalog is built rather than a mystery at render: the
    catalog is ours to keep unambiguous.
    """
    table: Dict[str, MetricSpec] = {}
    for spec in metrics:
        if table.setdefault(spec.key, spec) is not spec:
            raise AssertionError("metric key %r is claimed by both %s and %s"
                                 % (spec.key, table[spec.key].header, spec.header))
    for spec in resolved.specs:
        lower = spec.header.lower()
        for alias in (lower, lower.rstrip("%")):
            claimed = table.get(alias)
            if claimed is not None and claimed is not spec and claimed.column != spec.column:
                raise AssertionError(
                    "metric alias %r is claimed by both %s and %s"
                    % (alias, claimed.header, spec.header))
            table[alias] = spec
    return table


def _build(metrics: Sequence[MetricSpec],
           preference: Sequence[str]) -> GpuCatalog:
    """Resolve ``metrics`` under ``preference``. Pure -- nothing here reads state.

    Called whenever a site metric joins the catalog and whenever the source
    preference changes, since both alter which candidate wins a column. A site metric
    always joins ``group="all"``, so it cannot quietly widen what the default view
    collects; a preference can change *where* a default column comes from, which is
    the point of naming one.
    """
    metrics = tuple(metrics)
    preference = tuple(preference)
    resolved = source.resolve(metrics, preference, source.JOBSTATS_COLUMNS)
    default_specs = tuple(s for s in resolved.specs if s.group == "default")
    backed_keys = tuple(s.key for s in resolved.specs
                        if s.column in resolved.from_jobstats)
    summary_specs = tuple(s for s in default_specs
                          if s.column not in resolved.from_jobstats)
    return GpuCatalog(
        metrics=metrics,
        preference=preference,
        resolved=resolved,
        spec_by_header={spec.header: spec for spec in resolved.specs},
        all_specs=tuple(resolved.specs),
        default_specs=default_specs,
        key_specs=tuple(s for s in default_specs if s.column in _KEY_SPEC_COLUMNS),
        jobstats_backed_keys=backed_keys,
        gpu_summary_specs=summary_specs,
        headers=tuple(spec.header for spec in summary_specs),
        aliases=_alias_table(metrics, resolved),
        # One canonical name per metric, to offer when a config gets one wrong. The
        # header without its "%" where there is one (``gpu``, ``sm_act``) and the key
        # otherwise (``power``, not ``power_w``) -- the shorter and more readable of
        # the two forms in each case. Every alias still resolves.
        names=tuple(spec.header.lower()[:-1] if spec.header.endswith("%") else spec.key
                    for spec in resolved.specs),
        order={spec.key: i for i, spec in enumerate(metrics)},
    )


# ``METRICS`` above is the built-in declaration and is never mutated, so a
# re-registration always starts from a pristine set: site metrics are additive, not
# cumulative, and two loads of the same config must produce one copy of each metric.
_ACTIVE: GpuCatalog = _build(METRICS, source.DEFAULT_PREFERENCE)


def catalog() -> GpuCatalog:
    """The GPU catalog in force. Read it at call time, never hold the fields.

    Holding the whole object is fine -- it is frozen, and a snapshot of a consistent
    resolution. Holding one of its fields across a :func:`set_preference` is what the
    dozen module globals this replaced made easy and this makes pointless.
    """
    return _ACTIVE


def default_view(view: str, family: Optional[str] = None) -> List[MetricSpec]:
    """The built-in metric list for ``view`` under the leading source.

    This is what makes each source's defaults its own rather than one list with
    holes in it. dcgm publishes the whole profiling catalog, so leading with it gives
    the activity columns; the nvidia exporter publishes duty cycle, memory, power and
    temperature and no profiling metrics at all, so leading with it gives *those*
    instead of quietly keeping a set of dcgm columns the choice was meant to leave.

    Falls back to the whole pool for a family with nothing of its own, so a source
    that turns out to publish none of a view's metrics still yields a report rather
    than an empty table.
    """
    active = catalog()
    pool = {"summary": active.default_specs, "timeseries": active.key_specs,
            "extended": active.all_specs}[view]
    family = family or active.resolved.leading_exporter()
    own = [spec for spec in pool if spec.family == family]
    return own or list(pool)


def set_preference(preference: Tuple[str, ...]) -> None:
    """Choose which source serves each column, then recompute the resolved view.

    Separate from :func:`register` because the two are independent: a site names its
    series in ``[metrics.<family>]``, and names its order in ``[gpu] source``. Both
    replace the catalog wholesale, which is what lets
    :func:`jobscope.metrics.catalog` notice by identity that its view is stale.
    """
    global _ACTIVE
    _ACTIVE = _build(_ACTIVE.metrics, preference)


def register(extra: List[MetricSpec]) -> None:
    """Replace the site-defined additions to the GPU catalog with ``extra``.

    Called once per config load, and always from the built-ins, so loading a config
    twice -- which tests and ``jobscope config`` both do -- does not accumulate
    duplicates, and dropping a metric from the file actually drops it.

    A spec whose ``key`` matches a built-in **replaces** it, which is how a site
    whose exporter publishes a different series name for the same quantity ports
    without a patch -- keeping the built-in's ``group`` and ``roles``, see
    :func:`_inherit`. Anything else is appended after the built-ins, so catalog order
    stays stable and site metrics sort last in every view that shows them.
    """
    global _ACTIVE
    _ACTIVE = _build(source.merged_metrics(METRICS, extra, _inherit),
                     _ACTIVE.preference)


def _inherit(builtin: MetricSpec, override: Optional[MetricSpec]) -> MetricSpec:
    """``override`` with the built-in's *purpose* kept, or the built-in unchanged.

    A site overriding ``duty`` means "my exporter calls that series something else",
    not "take GPU% out of the default view and out of the efficiency ballot". So
    ``group``, ``roles``, ``show``, which column it serves and the short label forms
    come from the built-in; only how to *fetch and scale* the value comes from the
    config.
    """
    if override is None:
        return builtin
    return replace(override, group=builtin.group, roles=builtin.roles,
                   show=builtin.show, slug=builtin.slug, tag=builtin.tag,
                   provides=builtin.provides)


def spec_named(name: str) -> Optional[MetricSpec]:
    """The spec ``name`` refers to, or None when the catalog has no such metric."""
    return catalog().aliases.get(str(name).strip().lower())


def specs_named(names, running: bool = False) -> List[MetricSpec]:
    """Resolve metric names to specs, in catalog order, dropping duplicates.

    Catalog order rather than the order given, because column order is a property
    of the report and not of how a site happened to list them -- and because it is
    what keeps a narrower selection a prefix of a wider one, which several callers
    rely on to tell "default" from "extended".

    ``running`` drops metrics whose reducer is ``delta`` (ENERGY_kWh): the running view
    synthesizes a jobstats-shaped summary and a counter difference has no meaning over
    a window that has not finished. Unknown names are the caller's to validate --
    :func:`spec_named` returns None and this skips them.

    Deduplicated per *column*, not per key: naming both providers of one column --
    ``["duty", "duty_dcgm"]``, or ``["gpu", "duty"]`` where ``gpu`` is the resolved
    spelling of whichever is live -- asks for one column twice, from two exporters.
    The first named wins, so a list can still say which provider it means.
    """
    order = catalog().order
    found = {}
    for name in names:
        spec = spec_named(name)
        if spec is None or (running and spec.reducer == "delta"):
            continue
        found.setdefault(spec.column, spec)
    return [spec for spec in sorted(found.values(), key=lambda s: order[s.key])]

DESCRIPTIONS: Dict[str, str] = {
    "GPU%": "NVML's duty cycle: the fraction of the run during which at least one kernel was "
            "executing on the GPU, the same number jobstats reports. Says the GPU was occupied "
            "in time, NOT how intensely: a 1-thread kernel and a full-GPU kernel both read ~100%. "
            "For a finished job this comes from the stored summary, so every view agrees. NVML stops "
            "reporting it once MIG is enabled, so it is \"-\" on a MIG node.",
    "SM_ACT%": "Fraction of time at least one warp was resident on an SM, averaged across all "
               "SMs. Distinguishes 'one SM busy' from 'all SMs busy'; low while GPU% is high "
               "means the GPU was barely loaded (parked / underfed). Measured cluster-wide it runs "
               "~20 points BELOW GPU% on 93% of active GPUs, so do not read it as "
               "\"GPU utilization\"; the gap between the two is the diagnostic signal.",
    "OCC%": "SM occupancy: the fraction of warp slots that were filled, averaged over SMs and "
            "time (active warps / the hardware max per SM). Low occupancy means kernels under-fill "
            "the GPU: small launches, or register / shared-memory limits.",
    "TENSOR%": "Fraction of time the tensor-core pipe was active. High only for mixed-precision "
               "matmul-heavy work (fp16/bf16/tf32 training or inference); ~0 means the tensor cores "
               "sat idle.",
    "DRAM%": "Fraction of time the device-memory (HBM) interface was busy moving data, a "
             "memory-bandwidth duty cycle. High while SM_ACT% is low suggests the job is "
             "memory-bound, not compute-bound.",
    "POWER_W": "Mean board power draw over the run, in watts. Compare to the GPU's TDP (~700 W for "
               "H100/H200, ~400 W for A100); near-idle watts mean the GPU was not really working.",
    "ENGINE%": "Fraction of time the graphics/compute engine had work in flight, DCGM's "
               "finer-grained analogue of GPU%. Tracks it closely over a job-length window "
               "(median |delta| 1.2, r=0.99 at 1h), so it is a usable stand-in; the two disagree "
               "far more on a single scrape, as they are scraped independently.",
    "HMMA%": "Tensor-core activity for half-precision matrix ops (fp16/bf16). A precision breakdown "
             "of TENSOR%.",
    "IMMA%": "Tensor-core activity for integer matrix ops (int8). Precision breakdown of TENSOR%: "
             "nonzero for quantized / int8 inference.",
    "DFMA%": "Tensor-core activity for double-precision matrix ops (fp64). Precision breakdown of "
             "TENSOR%: relevant to fp64 HPC on tensor cores.",
    "FP16%": "Fraction of time the (non-tensor) fp16 floating-point pipe was active.",
    "FP32%": "Fraction of time the (non-tensor) fp32 floating-point pipe was active, the default "
             "precision for much numerical code.",
    "FP64%": "Fraction of time the fp64 (double-precision) pipe was active. High for "
             "double-precision HPC (CFD, MD, dense linear algebra).",
    "MEMCP%": "Percent of time the memory-copy engine was moving data (nvidia-smi's 'Memory' "
              "utilization). Not the same as DRAM bandwidth.",
    "PWRmax_W": "Peak board power seen during the run, in watts (vs POWER_W, the mean).",
    "ENERGY_kWh": "Total energy the GPU consumed over the run, in kWh (from the monotonic energy "
                  "counter, end minus start). Useful for cost / efficiency accounting.",
    "FB_USED_GB": "Mean GPU (framebuffer) memory in use over the run, in GiB. Complements jobstats' "
                  "GMEM%, which is the PEAK: a job that loads a model then idles shows high peak but a "
                  "lower mean.",
    "FB_FREE_GB": "Mean free GPU memory over the run, in GiB.",
    "FB_RSVD_GB": "Mean GPU memory reserved by the driver/system over the run, in GiB (not available to "
                  "your job).",
    "PCIE_TX_MBs": "Mean PCIe transmit throughput (GPU -> host), in MB/s. A bottleneck if the GPU waits "
                   "on host transfers. (DCGM rate; treat the absolute value as approximate.)",
    "PCIE_RX_MBs": "Mean PCIe receive throughput (host -> GPU), in MB/s, e.g. input batches streamed to "
                   "the GPU. (DCGM rate; absolute value approximate.)",
    "NVLINK_MBs": "Mean NVLink throughput, in MiB/s. ~0 for single-GPU jobs; nonzero indicates "
                  "multi-GPU communication (e.g. NCCL all-reduce). (DCGM rate; absolute approximate.)",
    "SMCLK_MHz": "Mean SM (core) clock frequency, in MHz. A low mean alongside high temperature / power "
                 "can indicate throttling.",
    "MEMCLK_MHz": "Mean memory clock frequency, in MHz.",
    "TEMP_C": "Mean GPU core temperature, in degrees Celsius.",
    "MEMTEMP_C": "Mean GPU memory (HBM) temperature, in degrees Celsius.",
    "ENC%": "Mean hardware video-encoder (NVENC) utilization. Usually 0 unless encoding video.",
    "DEC%": "Mean hardware video-decoder (NVDEC) utilization. Usually 0 unless decoding video.",
    "GMEM_GB": "GPU framebuffer memory in use, in GiB, from the NVML exporter. Peaked rather than "
               "averaged, so it is jobstats' \"GPU memory usage per node - maximum used/total\". "
               "For a finished job this comes from the stored summary, so every view agrees.",
    "GMEM_TOTAL_GB": "Total framebuffer memory on the GPU, in GiB. Fetched only to derive GMEM%; on "
                     "a MIG instance this is the slice's share, not the physical card's.",
    "GMEM%": "GMEM_GB as a percent of that GPU's total memory, the per-GPU form of the summary "
             "view's GMEM%. Named GMEM% rather than MEM% because a bare MEM% means HOST memory "
             "elsewhere. On a MIG row the total is the slice's, so the percentage is per slice.",
}

def format_number(value: Optional[float], decimals: int, missing: str = "-") -> str:
    """Format one metric cell to ``decimals`` places; ``missing`` when value is None."""
    if value is None:
        return missing
    if decimals:
        return ("{:.%df}" % decimals).format(value)
    return str(int(round(value)))


def format_value(spec: MetricSpec, value: Optional[float]) -> str:
    """Format one metric cell ('-' when missing), using the metric's decimals."""
    return format_number(value, spec.decimals)


def format_by_header(header: str, value: Optional[float]) -> str:
    """Format a metric cell by header (summary/detail callers); see format_value."""
    return format_value(catalog().spec_by_header[header], value)


def gpu_minor_key(minor):
    """Numeric sort key for a GPU minor number; falls back to string."""
    return int(minor) if str(minor).isdigit() else minor


def job_model(per_gpu: dict) -> str:
    """The GPU model a job ran on, or ``""`` when its cards disagree.

    Slurm allocates from one partition, so in practice a job's GPUs are one model and
    this is exact. When they are not there is no honest single floor -- the highest
    over-flags, the lowest under-flags -- so it says nothing and the global value is
    used instead of an invented rule.

    Here rather than in the renderer that reads it, because ``MODEL_KEY`` is this
    module's and because :mod:`jobscope.rows` has to answer the same question one
    layer down.
    """
    models = {v.get(MODEL_KEY, "") for v in per_gpu.values() if isinstance(v, dict)}
    models.discard("")
    return models.pop() if len(models) == 1 else ""


# Where a card's model rides in a per-GPU metric dict. Not a metric, so it is keyed
# out of the header namespace: POWER_W's floor is per architecture and the renderers
# need to know which card produced a reading.
MODEL_KEY = "__model__"


def _reduce(selector: str, reducer: str, duration: int, instant: bool = False) -> str:
    """Wrap a selector in its window reducer, or leave it bare when ``instant``.

    ``instant`` is for a job that has not finished: the newest scrape rather than a
    mean over a window that is still growing. It leaves the selector unwrapped, which
    is the same query :func:`jobscope.running.collect_instant` builds -- so a running
    job reads the same whether it was selected by ``-j`` or by a partition sweep.

    ``delta`` is exempt and keeps its window. It is a counter difference (ENERGY_kWh),
    so "the newest scrape" of it is not a smaller version of the answer, it is no
    answer at all -- a single sample has no difference to report.
    """
    if instant and reducer != "delta":
        return selector
    if reducer == "avg":
        return "avg_over_time((%s)[%ds:])" % (selector, duration)
    if reducer == "max":
        return "max_over_time((%s)[%ds:])" % (selector, duration)
    return "(max_over_time((%s)[%ds:]) - min_over_time((%s)[%ds:]))" % (
        selector, duration, selector, duration)


def ownership_clip(uuid_label: str, raw_jobid) -> Optional[str]:
    """The join series to intersect a selector with, so only this job's samples count.

    An nvml GPU metric carries the owning job in a ``jobid`` *label*, so one card has one
    series **per job that has held it**. Selecting by UUID alone therefore returns every
    past owner's series over any window long enough to reach them -- measured on one A100
    over two hours: ten series for one card, means from 0.00 to 100.00 -- and a collector
    keyed by UUID keeps whichever lands last. That is how ``--gpu-source nvml`` reported
    GPU% 0 for a job running at 73%.

    ``None`` for the DCGM family, whose series carry no job label at all: its labels
    (``UUID``/``Hostname``/``gpu``) can never match the join's, so PromQL's ``and`` would
    match nothing and blank the column instead of narrowing it. There the window alone
    bounds the samples, which is why a DCGM mean can still include a previous owner's --
    a real limitation, and not one this can fix.
    """
    return ("%s == %s" % (config.gpu_join(), raw_jobid)
            if uuid_label == "uuid" and raw_jobid else None)


def window_query(spec: MetricSpec, uuids: List[str], duration: int,
                 clip: Optional[str] = None, instant: bool = False) -> str:
    """PromQL that reduces ``spec`` over a ``duration``-second window for ``uuids``.

    ``clip`` is an optional series to intersect the selector with, which restricts
    the window to the samples where that series also existed. Only usable when the
    two come from the same exporter, since PromQL's ``and`` requires identical
    label sets -- see :func:`ownership_clip`, which decides when there is one.
    """
    regex = "^(" + "|".join(uuids) + ")$"  # UUIDs are hex+hyphen, RE2-safe as-is
    selector = '%s{%s=~"%s"}' % (spec.metric, spec.uuid_label, regex)
    if clip:
        selector = "%s and %s" % (selector, clip)
    return _reduce(selector, spec.reducer, duration, instant)


# The label a grouped query's metric name is copied into. Needed because
# `avg_over_time` and every other function that transforms a value **drops
# `__name__`** -- so a response covering several metrics is indistinguishable
# without it, and a demultiplexer keyed on `__name__` silently matches nothing.
NAME_LABEL = "jsname"


def group_key(spec: MetricSpec) -> Tuple[str, str]:
    """The batch a spec can share a query with.

    Both halves matter. ``reducer`` picks the function, so metrics reduced
    differently cannot share one call. ``uuid_label`` differs by *family* -- NVML
    publishes a lowercase ``uuid`` where DCGM uses uppercase ``UUID`` -- and mixing
    them yields a response whose rows cannot be attributed to a card.
    """
    return (spec.reducer, spec.uuid_label)


def grouped_window_query(reducer: str, uuid_label: str, metrics: List[str],
                         uuids: List[str], duration: int,
                         instant: bool = False, clip: Optional[str] = None) -> str:
    """One query covering several metrics that share a reducer and a UUID label.

    Replaces N per-metric round trips with one: 7 becomes 3 for the default column
    set and 30 becomes 5 under ``--dcgm``, measured at 2.1x and 5.6x with values
    identical to the per-metric path.

    ``label_replace`` copies the series name into :data:`NAME_LABEL` *before* the
    reduction, which is what makes the response demultiplexable -- see that
    constant. For the ``delta`` reducer it has to appear inside both
    ``max_over_time`` and ``min_over_time``, since each is its own selector.
    """
    names = "^(" + "|".join(sorted(set(metrics))) + ")$"
    regex = "^(" + "|".join(uuids) + ")$"
    inner = '{__name__=~"%s",%s=~"%s"}' % (names, uuid_label, regex)
    if clip:
        # Inside the label_replace, not after it. ``and`` keeps only left-hand elements
        # whose label set matches one on the right, and label_replace has by then added
        # NAME_LABEL to the left alone -- so the outer form matches nothing and empties
        # every column. Verified against the live server: outer 0 series, inner 2.
        # ``__name__`` survives the ``and`` (set operators carry the left side over whole
        # and ignore the metric name when matching), so it is still there to copy.
        inner = "(%s and %s)" % (inner, clip)
    selector = ('label_replace(%s,"%s","$1","__name__","(.*)")'
                % (inner, NAME_LABEL))
    # The label_replace sits inside the selector, so an instant (unreduced) query still
    # carries NAME_LABEL and stays demultiplexable -- the reduction is what would have
    # dropped __name__, and its absence cannot reintroduce that problem.
    return _reduce(selector, reducer, duration, instant)


def _store_value(per_uuid: Dict[str, dict], spec: MetricSpec, series: dict) -> bool:
    """Record one series' value under its card and header; True if it landed.

    The return value is what tells a grouped query whether its response was
    attributable at all -- an empty or unexpectedly-labelled one stores nothing, and
    that is the signal to retry per metric.
    """
    labels = series["metric"]
    uuid = (labels.get(spec.uuid_label) or labels.get("uuid") or labels.get("UUID"))
    if uuid not in per_uuid:
        return False
    try:
        per_uuid[uuid][spec.header] = float(series["value"][1]) * spec.scale
    except (KeyError, IndexError, TypeError, ValueError):
        # KeyError/IndexError for a response with no value at all: this reports whether
        # anything was attributable, and a malformed row is exactly the case the caller
        # answers by retrying per metric. Raising here made one unexpected row abort the
        # whole report instead.
        return False
    return True


def _collect_per_spec(per_uuid, specs, uuids, duration, at, client, timeout,
                      instant: bool = False, raw_jobid=None) -> None:
    """One query per metric -- the original path, and the fallback."""
    for spec in specs:
        clip = ownership_clip(spec.uuid_label, raw_jobid)
        try:
            found = client.query(window_query(spec, uuids, duration, clip=clip,
                                              instant=instant),
                                 at, timeout)
        except Exception:
            continue        # a failed metric leaves its column empty, as before
        for series in found:
            _store_value(per_uuid, spec, series)


def collect_window(per_uuid: Dict[str, dict], specs: List[MetricSpec], uuids: List[str],
                   duration: int, at, client: PrometheusClient,
                   timeout: Optional[float], instant: bool = False,
                   raw_jobid=None) -> None:
    """Fill ``per_uuid`` with every spec's windowed value for these cards.

    ``instant`` asks for the newest scrape instead of a reduction over ``duration``;
    see :func:`_reduce`. ``duration`` is still required, because the ``delta`` reducer
    ignores ``instant`` and because the caller's discovery query needs it.

    Metrics that share a reducer and a UUID label go in one query
    (:func:`grouped_window_query`); a group that fails or comes back unusable falls
    back to per-metric queries for *that group only*. The fallback matters: a server
    without ``label_replace``, or one labelling results unexpectedly, must produce
    the same report a little slower rather than a report with blank columns.

    ``raw_jobid`` restricts nvml-family selectors to the samples this job owned the card
    for -- see :func:`ownership_clip`. Optional so a caller asking about cards rather than
    a job (probe, a bare UUID query) still gets every series; the report path always
    passes it, and without it an nvml column can report a previous owner's number.
    """
    groups: Dict[Tuple[str, str], List[MetricSpec]] = {}
    for spec in specs:
        groups.setdefault(group_key(spec), []).append(spec)

    for (reducer, uuid_label), members in groups.items():
        if len(members) == 1:
            _collect_per_spec(per_uuid, members, uuids, duration, at, client, timeout,
                              instant, raw_jobid)
            continue
        # One series can back two specs -- DCGM_FI_DEV_POWER_USAGE feeds POWER_W and
        # PWRmax_W -- so a name maps to a *list*. Those two differ by reducer and so
        # land in different groups, but nothing guarantees that for a future pair.
        by_metric: Dict[str, List[MetricSpec]] = {}
        for spec in members:
            by_metric.setdefault(spec.metric, []).append(spec)
        query = grouped_window_query(reducer, uuid_label, list(by_metric), uuids,
                                     duration, instant,
                                     clip=ownership_clip(uuid_label, raw_jobid))
        try:
            found = client.query(query, at, timeout)
        except Exception:
            found = []
        stored = 0
        for series in found:
            for spec in by_metric.get(series["metric"].get(NAME_LABEL), ()):
                stored += _store_value(per_uuid, spec, series)
        if not stored:
            # Nothing attributable came back: either the group query failed or the
            # response carried no NAME_LABEL. Ask per metric rather than report gaps.
            _collect_per_spec(per_uuid, members, uuids, duration, at, client, timeout,
                              instant, raw_jobid)


def _jobid_query(record: JobRecord) -> str:
    cluster = "slurm_cluster='%s'" % record.cluster if record.cluster else ""
    return ("max_over_time((%s{%s} == %s)[%ds:])"
            % (config.gpu_join(), cluster, record.jobid_raw, record.duration))


def gpu_from_series(metric: dict) -> Optional[dict]:
    """One GPU descriptor from a join or GPU metric series' labels, either family.

    The two spell card identity differently in three of four places: DCGM writes
    ``UUID``/``gpu``/``modelName`` where nvml writes ``uuid``/``minor_number``/``name``.
    So this reads both spellings rather than existing twice -- on an nvml series it returns
    exactly what the nvml-only reader it replaced returned, which is what made that reader
    redundant. Only the host label is shared, which is why ``config.host_of`` serves both.

    The UUID itself *is* shared, byte for byte -- measured, 1734 of 1744 nvml uuids appear
    verbatim as a DCGM ``UUID`` -- and that is the whole reason a card discovered through
    one family can be queried through the other.

    ``GPU_I_ID`` is deliberately ignored even though it is a real MIG instance index and
    would identify a slice better than nvml can. Using it here would label the same card
    differently depending on which mapping found it, and two names for one GPU is worse
    than one imperfect name; ``running.gpu_labels`` enumerates siblings by sorted UUID for
    both.
    """
    uuid = metric.get("UUID") or metric.get("uuid")
    if not uuid:
        return None
    return {"uuid": uuid,
            "node": config.host_of(metric),
            "minor": str(metric.get("gpu", metric.get("minor_number", "?"))),
            "model": metric.get("modelName", metric.get("name", ""))}


def _sorted_gpus(gpus: List[dict]) -> List[dict]:
    return sorted(gpus, key=lambda g: (g["node"], gpu_minor_key(g["minor"])))


def discover_gpus(record: JobRecord, client: PrometheusClient,
                  timeout: Optional[float]) -> List[dict]:
    """The GPUs that ran a job, as ``{uuid, node, minor, model}``, by (node, minor).

    Empty for CPU-only jobs or when no GPU samples exist. Joins via
    the site's GPU-join series, the same mapping jobstats uses.

    One query per job, and measurement says to keep it that way: answering this for
    a whole selection in one range query is correct but *slower* here, because the
    per-job calls already run 8-wide against ~90ms queries while the batched form
    returns one or two million samples serially. See the note in ``compute_dcgm``.
    """
    if not (record.gpus and record.jobid_raw and record.duration):
        return []
    try:
        found = client.query(_jobid_query(record), record.end, timeout)
    except Exception:
        found = []
    gpus = _sorted_gpus([g for g in (gpu_from_series(s["metric"]) for s in found) if g])
    if gpus:
        return gpus
    # Nothing claimed this job. Ask the label-based mapping, if the site has one -- see
    # jobscope.config.Site on why a second one is worth naming. Only when the first found
    # nothing: a mapping that answered should not be second-guessed by another.
    return _sorted_gpus(_labelled_gpus(record, client, timeout))


def _labelled_gpus(record: JobRecord, client: PrometheusClient,
                   timeout: Optional[float]) -> List[dict]:
    """This job's cards from the label-based mapping, or [] if the site has none.

    Windowed and anchored at ``record.end`` like :func:`_jobid_query`, and for the same
    reason: it asks which cards carried the job *during its own run*, so a label still
    naming the job after it finished cannot pull in a card that has moved on.
    """
    label = config.gpu_job_label()
    if not label:
        return []
    query = 'max_over_time((%s{%s="%s"})[%ds:])' % (
        config.gpu_job_label_series(), label, record.jobid_raw, record.duration)
    try:
        rows = client.query(query, record.end, timeout)
    except Exception:
        return []
    return [g for g in (gpu_from_series(r["metric"]) for r in rows) if g]


def dcgm_for_job(record: JobRecord, specs: List[MetricSpec], client: PrometheusClient,
                 timeout: Optional[float],
                 gpus_found: Optional[List[dict]] = None,
                 nodename: Optional[str] = None, gpu_ids=(),
                 average: bool = False) -> JobGpuData:
    """``(overall, per_gpu)`` metric dicts for one job over ``specs``.

    ``({}, {})`` when the job has no GPUs or no samples. ``overall`` is keyed by
    header; ``per_gpu`` is keyed by ``(node, minor)``. Series are joined to the
    job on UUID via the site's GPU-join series.

    ``gpus_found`` supplies the job's cards when a caller has already discovered
    them, which skips the discovery round trip. Omit it and this discovers them
    itself, which is what every caller does today -- the parameter exists so a
    caller that already knows (a future batched discovery, or a test) need not
    re-ask.

    ``nodename``/``gpu_ids`` narrow the cards, for ``--nodename``/``--gpuid`` on the
    summary. Applied after discovery but before the metric queries, so a filtered
    card costs one entry in a regex rather than a window of samples fetched and
    discarded -- the same ordering the time-series path uses.

    The **window follows the job, not the caller.** A finished job is folded over its
    whole runtime, which is what makes the numbers comparable to jobstats. A job still
    running gets the newest scrape unless ``average``, matching the squeue path and
    ``--runtime-avg``'s documented default. Decided per record, so a selection spanning both
    states reports each on its own rule -- this used to key on the *selection mode*
    instead, which made ``jobscope -j <running job>`` silently report a runtime mean
    while every other view of the same job showed the latest scrape.
    """
    if not (record.gpus and record.jobid_raw and record.duration):
        return JobGpuData()
    if gpus_found is None:
        # discover_gpus, not a second copy of its body. This carried its own inline query
        # until the label-based fallback was added to that function alone -- which left the
        # whole report path (summary, --per-gpu, --per-node) with no second mapping while
        # --ts and --verify had one, so the same job showed cards in one view and
        # dashes in another. Calling it means a future mapping reaches every path by
        # construction rather than by remembering there were two discovery bodies.
        gpus_found = discover_gpus(record, client, timeout)
    if nodename:
        gpus_found = [g for g in gpus_found if g["node"] == nodename]
    if gpu_ids:
        wanted = {str(x) for x in gpu_ids}
        gpus_found = [g for g in gpus_found if str(g["minor"]) in wanted]
    gpus = [(g["node"], g["minor"], g["uuid"]) for g in gpus_found]
    uuids = [g["uuid"] for g in gpus_found]
    models = {g["uuid"]: g["model"] for g in gpus_found}
    if not uuids:
        return JobGpuData()

    per_uuid: Dict[str, dict] = {uuid: {} for uuid in uuids}
    collect_window(per_uuid, specs, uuids, record.duration, record.end, client, timeout,
                   instant=record.unfinished and not average,
                   raw_jobid=record.jobid_raw)

    per_gpu = {}
    for node, minor, uuid in gpus:
        values = per_uuid.get(uuid, {})
        # The card's model rides along with its metrics: POWER_W's floor is per
        # architecture, and this is the only place that knows which card it was.
        values[MODEL_KEY] = models.get(uuid, "")
        per_gpu[(node, minor)] = values
    overall = pool_uuids(per_uuid, specs, uuids)
    by_node: Dict[str, List[str]] = {}
    for node, _minor, uuid in gpus:
        by_node.setdefault(node, []).append(uuid)
    per_node = {node: pool_and_derive(per_uuid, specs, uuids_here)
                for node, uuids_here in by_node.items()}
    _prefer_stored(record, specs, per_gpu, overall)
    _add_derived(specs, per_gpu, overall)
    return JobGpuData(overall, per_gpu, per_node)


def pool_and_derive(per_uuid: Dict[str, dict], specs: List[MetricSpec],
                    uuids: List[str]) -> Dict[str, float]:
    """:func:`pool_uuids` over ``uuids``, then the derived columns from the result.

    The two steps belong together and in this order. ``GMEM%`` is used/total, so a pooled
    figure has to be the ratio of the pooled halves -- deriving first and pooling after
    would average per-card ratios, which is a different number (see
    ``running.host_series``). One function so the running and finished paths cannot get
    the order different from each other.
    """
    pooled = pool_uuids(per_uuid, specs, uuids)
    _add_derived(specs, {}, pooled)
    return pooled


def pool_uuids(per_uuid: Dict[str, dict], specs: List[MetricSpec],
               uuids: List[str]) -> Dict[str, float]:
    """One figure per column over ``uuids``, reduced by each spec's own ``agg``.

    ``mean`` for a utilization, ``max`` for peak memory and PWRmax_W, ``sum`` for
    ENERGY_kWh -- a metric declares how it combines across cards, so this needs no table
    of its own.

    **Keyed by UUID, deliberately, never by ``(node, minor)``.** MIG siblings share a
    minor number, so ``per_gpu`` has already collapsed them and the last one written wins;
    pooling that dict would silently reduce a partitioned card to one of its instances.
    That was already true of the job-level figure and is what makes this shared rather
    than reimplemented per level.
    """
    out: Dict[str, float] = {}
    for spec in specs:
        values = [per_uuid[u][spec.header] for u in uuids
                  if u in per_uuid and spec.header in per_uuid[u]]
        if values:
            out[spec.header] = (sum(values) if spec.agg == "sum"
                                else max(values) if spec.agg == "max"
                                else sum(values) / len(values))
    return out


# Summary field -> the *column* it supersedes, and how a job-level figure is formed
# from the per-GPU values. GMEM_GB sums because the summary's GMEM% is the ratio of
# summed used to summed total across the job's GPUs.
#
# By column rather than by metric key, because which key serves a column depends on
# the source preference: keyed on "duty" this silently stopped applying the moment
# dcgm became the preferred exporter for GPU%, since the resolved spec is then
# `duty_dcgm`. The column is the stable identity -- and it is what
# source.JOBSTATS_COLUMNS states the jobstats summary can serve.
# The summary field of each comes from source.JOBSTATS_COLUMNS, which is the one statement
# of what jobstats stored; only the aggregation is this view's business.
_STORED_AGG: Dict[str, str] = {"GPU%": "mean", "GMEM_GB": "sum", "GMEM_TOTAL_GB": "sum"}
_STORED_FIELDS: Tuple[Tuple[str, str, str], ...] = tuple(
    (field, column, _STORED_AGG[column])
    for column, field in source.JOBSTATS_COLUMNS.items())


def stored_per_gpu(record: JobRecord, field: str) -> Dict[Tuple[str, str], float]:
    """One per-GPU map from the job's stored summary, keyed ``(node, minor)``.

    Empty when the job has no jobstats summary, which is every running job -- Slurm writes it
    at job end.
    """
    found: Dict[Tuple[str, str], float] = {}
    for node, info in (record.stats or {}).get("nodes", {}).items():
        for minor, value in (info.get(field) or {}).items():
            try:
                found[(node, str(minor))] = float(value)
            except (TypeError, ValueError):
                continue
    return found


def stored_utilization(record: JobRecord) -> Dict[Tuple[str, str], float]:
    """Per-GPU utilization from the job's stored summary, keyed ``(node, minor)``."""
    return stored_per_gpu(record, "gpu_utilization")


def _prefer_stored(record: JobRecord, specs: List[MetricSpec],
                   per_gpu: Dict[Tuple[str, str], dict],
                   overall: Dict[str, float]) -> None:
    """Overwrite GPU% and GPU memory with the summary's values where it has them.

    These come from the same ``nvidia_gpu_*`` series under the same reducers, so
    they measure the same thing -- but the jobstats summary is what jobstats computed at job
    end, while recomputing here has to reconstruct the window from sacct's Start
    and End. On a short job that boundary is worth several points (a 570s job at a
    60s scrape interval has ~10 samples, so one sample in or out moves the mean by
    ~8), which showed up as this view disagreeing with the summary view's GPU% for
    the same job.

    Rather than tune the window to imitate an instant we cannot recover, defer to
    the stored value: a finished job then reports exactly what Slurm recorded, in
    every view. Running jobs have no jobstats summary, so they keep the Prometheus value --
    reconstructed by :mod:`jobscope.job_ave_stats` for the jobstats columns, so those agree
    with each other by construction.

    Only for the columns the jobstats summary actually *wins*. Naming an exporter ahead of it --
    ``--gpu-source dcgm`` -- takes those columns out of ``resolved.from_jobstats``, and
    then the queried value is the answer and must not be overwritten by a stored one
    measured somewhere else. That is what makes the flag do what it says on a
    finished job rather than being quietly ignored.
    """
    from_jobstats = catalog().resolved.from_jobstats
    by_column = {spec.column: spec for spec in specs}
    for field, column, agg in _STORED_FIELDS:
        if column not in from_jobstats:
            continue
        spec = by_column.get(column)
        if spec is None:
            continue
        stored = stored_per_gpu(record, field)
        if not stored:
            continue
        for node_minor, value in stored.items():
            if node_minor in per_gpu:
                per_gpu[node_minor][spec.header] = value * spec.scale
        total = sum(stored.values()) * spec.scale
        overall[spec.header] = total / len(stored) if agg == "mean" else total


def _add_derived(specs: List[MetricSpec], per_gpu: Dict[Tuple[str, str], dict],
                 overall: Dict[str, float]) -> None:
    """Fill in the derived columns, per GPU and for the job as a whole."""
    derived = applicable_derived(specs)
    if not derived:
        return
    for values in list(per_gpu.values()) + [overall]:
        keyed = values_by_key(specs, values)
        for column in derived:
            computed = column.fn(keyed)
            if computed is not None:
                values[column.header] = computed


def compute_dcgm(records: Dict[str, JobRecord], jobids: List[str],
                 specs: List[MetricSpec], client: PrometheusClient,
                 timeout: Optional[float], workers: int,
                 nodename: Optional[str] = None,
                 gpu_ids=(), average: bool = False) -> Dict[str, JobGpuData]:
    """Run :func:`dcgm_for_job` over every GPU job, concurrently.

    The per-job queries are network I/O-bound, so a thread pool overlaps them and
    speeds up wide selections even on one CPU core. The per-metric queries within a
    job stay sequential; the parallelism is across jobs.

    **Three batching strategies were measured here and all three lost.** Recorded
    because the reasoning that recommends them is sound and someone will try again:

    * *One range query on the GPU-join series for the whole selection's discovery.*
      Correct -- verified identical UUID sets on 40/40 jobs -- but slower: 2.1s for
      25 jobs and 3.4s for 120 against 0.4s and 1.6s for the per-job pool, because
      it returns one to two million samples in a single serial call.
    * *More workers.* 8 -> 16 buys 1.25x and then plateaus; 32 and 64 are no better.
      The ceiling is the server's per-query work, not the thread count.
    * *One query per (reducer, uuid_label) group instead of per metric*, via a
      ``__name__`` regex. 3.5x on wall clock and returned no usable values -- the
      grouped selector does not resolve the way the per-metric one does.

    The premise that fewer round trips must be faster assumed ~90ms per query. That
    holds serially; across the pool the effective cost is nearer 10ms, and payload
    size then dominates. Anything tried next should be measured against the *pooled*
    path, not a serial baseline.
    """
    gpu_jobs = [jid for jid in jobids if jid in records and records[jid].gpus]
    if not gpu_jobs:
        return {}

    workers = max(1, min(workers, len(gpu_jobs)))
    # `average` rides with the narrowing kwargs: it is per call, not per record, and
    # dcgm_for_job combines it with each record's own state.
    narrow = {"nodename": nodename, "gpu_ids": gpu_ids, "average": average}
    if workers == 1:
        return {jid: dcgm_for_job(records[jid], specs, client, timeout, **narrow)
                for jid in gpu_jobs}
    results: Dict[str, JobGpuData] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(dcgm_for_job, records[jid], specs, client, timeout,
                                   **narrow): jid
                   for jid in gpu_jobs}
        for future, jid in futures.items():
            try:
                results[jid] = future.result()
            except Exception:
                results[jid] = JobGpuData()
    return results
