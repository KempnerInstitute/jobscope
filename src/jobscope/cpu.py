"""cgroup host metric time series -- the host analogue of jobscope.dcgm's GPU one.

jobstats' own Prometheus exporter scrapes per-job ``cgroup_*`` series labeled
directly by ``jobid`` -- no GPU-UUID-style join needed, unlike DCGM/nvidia-exporter
metrics. :mod:`jobscope.live_blob` only ever reduces four of them to one aggregate
figure per job (client-side, via an instant query); this module range-queries the
ones that vary over a job's run, to build genuine per-sample series.

There is a catalog here, but a separate one from :class:`jobscope.dcgm.MetricSpec`
rather than a reuse of it, because these metrics are shaped differently in two ways
that matter. A DCGM value is multiplied by a constant ``scale``; a cgroup value is
divided by a *dynamic* denominator (cores allocated, bytes allocated) that varies
per host and per job. And the CPU series are counters needing ``rate()`` where the
memory series are gauges read directly. ``kind`` and ``denom`` carry exactly those
two differences.

The denominators are resolved by the caller rather than here: a finished job already
has them in its stored blob, a running job gets them from one batched
``live_blob.host_stats_many`` call, and neither varies enough within a job's
lifetime to be worth re-querying per sample. They arrive as the per-node blob dict
itself, so ``denom`` names a blob field.
"""

from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, List, Optional, Tuple

from . import config
from .prometheus import PrometheusClient

# rate()/increase() need several raw samples to be reliable: a range vector sized to
# exactly the display step can span 0-1 scrapes depending on alignment and silently
# gap out most points. This is independent of the query's own step/cadence.
RATE_LOOKBACK_SCRAPES = 4


@dataclass(frozen=True)
class CgroupSpec:
    """One per-job cgroup metric and how to query, normalise and display it."""

    key: str        # the name config selects it by: cpu, cpu_user, mem, cache, ...
    header: str     # the column header: CPU%, CPU_USER%, MEM%, CACHE%, ...
    metric: str     # the Prometheus series
    kind: str       # "rate" for a counter, "gauge" for a level read directly
    denom: str      # the blob field that divides it: "cpus" or "total_memory"
    decimals: int   # display precision
    group: str      # "default" (always available) or "all" (opt-in via [metrics])
    # What the metric is *for*; read by jobscope.metrics rather than restated in
    # the header lists report.py used to keep. Same vocabulary as MetricSpec.roles.
    roles: FrozenSet[str] = frozenset()

    @property
    def label(self) -> str:
        """Short row-label form, e.g. ``CPU%`` -> ``CPU``."""
        return self.header.rstrip("%")

    @property
    def share_tag(self) -> str:
        """Suffix in a combined share, e.g. ``35%gpu+24%cpu``."""
        return self.label.lower()

    def query(self, raw_jobid: str, step: int, sampling_period: int) -> str:
        """This metric's PromQL over a job's own cgroup.

        ``step``/``task`` are pinned empty to select the job-level cgroup rather
        than a per-step one; an ``=''`` matcher also matches the label being absent,
        which is the case on exporters that do not emit it at all.
        """
        site = config.get_config().site
        selector = "%s{%s='%s',%s}" % (self.metric, site.jobid_label, raw_jobid,
                                       site.cgroup_selector)
        if self.kind != "rate":
            return selector
        window = max(step, RATE_LOOKBACK_SCRAPES * sampling_period)
        return "rate(%s[%ds])" % (selector, window)


# CPU% and MEM% are `default` -- the two the summary and detail views have always
# shown, and the only two a stored sacct blob can reconstruct. The rest are `all`:
# they exist only in a --ts series (see the module docstring on why the summary
# cannot have them), and being outside the default group also keeps them out of the
# default --classify ballot, which they have no business deciding.
CGROUP_METRICS: List[CgroupSpec] = [
    # `split` divides the worst band into wasteful-cpu-gpu and wasteful-gpu: a job
    # idle on the GPU but busy on the host is a different finding from one idle on
    # both. `resource` pairs it with GPU% as the two distinct things a job holds.
    CgroupSpec("cpu", "CPU%", "cgroup_cpu_total_seconds",
               "rate", "cpus", 0, "default",
               roles=frozenset({"worst", "split", "resource"})),
    # `memory`: held bytes are not work. See GMEM%'s note in dcgm.py.
    CgroupSpec("mem", "MEM%", "cgroup_memory_rss_bytes",
               "gauge", "total_memory", 0, "default",
               roles=frozenset({"memory"})),
    # The user/system split. Sums to roughly CPU%, which is the point: 40% CPU that
    # is 30% system time is thrashing in the kernel, not working.
    CgroupSpec("cpu_user", "CPU_USER%", "cgroup_cpu_user_seconds",
               "rate", "cpus", 1, "all"),
    CgroupSpec("cpu_sys", "CPU_SYS%", "cgroup_cpu_system_seconds",
               "rate", "cpus", 1, "all"),
    # Page cache, which MEM% (RSS only) cannot see: a job can hold a lot of memory
    # and still read light.
    CgroupSpec("cache", "CACHE%", "cgroup_memory_cache_bytes",
               "gauge", "total_memory", 1, "all", roles=frozenset({"memory"})),
    # usage_in_bytes, i.e. roughly RSS + cache. Read the caveat in
    # config.example.toml before acting on it: because it counts *reclaimable*
    # cache, a job streaming a dataset drives this to ~100% with nothing at risk.
    # It is the figure the OOM limit is enforced on, not a utilization measure.
    CgroupSpec("mem_used", "MEM_USED%", "cgroup_memory_used_bytes",
               "gauge", "total_memory", 1, "all", roles=frozenset({"memory"})),
]

SPEC_BY_KEY: Dict[str, CgroupSpec] = {spec.key: spec for spec in CGROUP_METRICS}
DEFAULT_CGROUP_SPECS: List[CgroupSpec] = [s for s in CGROUP_METRICS
                                          if s.group == "default"]
# Every header this module can produce, for the config's typo check.
CGROUP_HEADERS: Tuple[str, ...] = tuple(spec.header for spec in CGROUP_METRICS)
CGROUP_NAMES: Tuple[str, ...] = tuple(spec.key for spec in CGROUP_METRICS)
_ORDER: Dict[str, int] = {spec.key: i for i, spec in enumerate(CGROUP_METRICS)}

# See jobscope.dcgm.register for why the built-ins are kept separately.
_BUILTIN: Tuple[CgroupSpec, ...] = tuple(CGROUP_METRICS)


def _rebuild() -> None:
    """Recompute the tables derived from ``CGROUP_METRICS``.

    ``DEFAULT_CGROUP_SPECS`` is rebuilt too, unlike the GPU side: a site can
    legitimately *override* ``cpu`` or ``mem`` -- the two default cgroup metrics --
    with a differently-named series from its own exporter, and the default view has
    to pick that up or the override does nothing where it matters most.
    """
    global SPEC_BY_KEY, DEFAULT_CGROUP_SPECS, CGROUP_HEADERS, CGROUP_NAMES, _ORDER
    SPEC_BY_KEY = {spec.key: spec for spec in CGROUP_METRICS}
    DEFAULT_CGROUP_SPECS = [s for s in CGROUP_METRICS if s.group == "default"]
    CGROUP_HEADERS = tuple(spec.header for spec in CGROUP_METRICS)
    CGROUP_NAMES = tuple(spec.key for spec in CGROUP_METRICS)
    _ORDER = {spec.key: i for i, spec in enumerate(CGROUP_METRICS)}


def register(extra: List[CgroupSpec]) -> None:
    """Replace the site-defined additions to the cgroup catalog with ``extra``."""
    by_key = {spec.key: spec for spec in extra}
    merged = [_inherit(builtin, by_key.pop(builtin.key, None)) for builtin in _BUILTIN]
    CGROUP_METRICS[:] = merged + [spec for spec in extra if spec.key in by_key]
    _rebuild()


def _inherit(builtin: CgroupSpec, override: Optional[CgroupSpec]) -> CgroupSpec:
    """``override`` with the built-in's *purpose* kept, or the built-in unchanged.

    A site overriding ``cpu`` is saying "my exporter calls that series something
    else", not "demote CPU% out of the summary and out of the classifier". So
    ``group`` and ``roles`` come from the built-in: forcing the override's own
    ``group="all"`` would drop CPU% from the default view entirely, and dropping its
    ``split`` role would change how every job is classified -- both silently.
    """
    if override is None:
        return builtin
    return replace(override, group=builtin.group, roles=builtin.roles)


def spec_named(name: str) -> Optional[CgroupSpec]:
    """The cgroup spec ``name`` refers to, by key or by header."""
    text = str(name).strip().lower()
    found = SPEC_BY_KEY.get(text)
    if found is not None:
        return found
    header = text.upper() if text.endswith("%") else text.upper() + "%"
    return next((s for s in CGROUP_METRICS if s.header == header), None)


def specs_named(names) -> List[CgroupSpec]:
    """Resolve cgroup metric names to specs, in catalog order, dropping duplicates.

    Catalog order rather than the order given, for the same reason
    :func:`jobscope.dcgm.specs_named` does it: column order is a property of the
    report, not of how a site listed them. Unknown names are the caller's to
    validate -- they are skipped here.
    """
    found = {}
    for name in names:
        spec = spec_named(name)
        if spec is not None:
            found[spec.key] = spec
    return [found[key] for key in sorted(found, key=_ORDER.__getitem__)]


def _host_of(series: dict) -> str:
    return config.host_of(series["metric"])


def host_series(raw_jobid: str, divisors: Dict[str, Dict[str, float]],
                start: int, end: int, step: int, sampling_period: int,
                client: PrometheusClient, timeout: Optional[float],
                specs: Optional[List[CgroupSpec]] = None
                ) -> Dict[str, Dict[int, Dict[str, float]]]:
    """``{host: {epoch: {header: percent}}}`` over ``[start, end]`` at ``step``.

    ``divisors`` is the per-node blob dict -- ``{host: {"cpus": n, "total_memory":
    b, ...}}`` -- which every caller already holds; each spec names the field that
    divides it. A host with no value for a given spec's ``denom`` is skipped for
    that spec rather than divided by zero, so a node reporting cores but not memory
    still gets its CPU columns.

    One range query per spec, so a wide selection costs proportionally more; the
    caller decides how many specs are worth that.
    """
    series: Dict[str, Dict[int, Dict[str, float]]] = {}
    for spec in (DEFAULT_CGROUP_SPECS if specs is None else specs):
        try:
            found = client.query_range(spec.query(raw_jobid, step, sampling_period),
                                       start, end, step, timeout)
        except Exception:
            # A failed query leaves that column empty rather than killing the
            # series: a partial answer is worth more here than none.
            continue
        for result in found:
            host = _host_of(result)
            divisor = (divisors.get(host) or {}).get(spec.denom)
            if not divisor:
                continue
            for stamp, value in result.get("values", []):
                try:
                    pct = 100 * float(value) / divisor
                except (TypeError, ValueError):
                    continue
                series.setdefault(host, {}).setdefault(
                    int(float(stamp)), {})[spec.header] = pct
    return series
