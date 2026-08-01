"""cgroup CPU%/MEM% time series -- the host analogue of jobscope.dcgm's GPU one.

jobstats' own Prometheus exporter already scrapes four per-job ``cgroup_*`` series
(see :mod:`jobscope.live_blob`), labeled directly by ``jobid`` -- no GPU-UUID-style
join needed, unlike DCGM/nvidia-exporter metrics. :mod:`jobscope.live_blob` only
ever reduces them to one aggregate figure per job (client-side, via an instant
query); this module range-queries the two that vary over a job's run --
``cgroup_cpu_total_seconds`` (a counter) and ``cgroup_memory_rss_bytes`` (a gauge)
-- to build a genuine CPU%/MEM% series.

Only two fixed metrics with two different formulas exist here, so unlike dcgm.py
there is no ``MetricSpec`` catalog -- that would be over-engineering for two
metrics. The (mostly constant per job) divisors -- cores allocated, bytes
allocated -- are resolved by the caller rather than here: a finished job already
has them in its stored blob, a running job gets them from one batched
``live_blob.host_stats_many`` call, and neither varies enough within a job's
lifetime to be worth re-querying per sample.
"""

from typing import Dict, Optional

from .prometheus import PrometheusClient

# rate()/increase() need several raw samples to be reliable: a range vector sized to
# exactly the display step can span 0-1 scrapes depending on alignment and silently
# gap out most points. This is independent of the query's own step/cadence.
RATE_LOOKBACK_SCRAPES = 4


def _rate_window(step: int, sampling_period: int) -> int:
    """The ``rate()`` lookback, at least a few scrapes regardless of ``step``."""
    return max(step, RATE_LOOKBACK_SCRAPES * sampling_period)


def cpu_query(raw_jobid: str, step: int, sampling_period: int) -> str:
    """PromQL for instantaneous CPU utilization (cores in use) over time."""
    window = _rate_window(step, sampling_period)
    return "rate(cgroup_cpu_total_seconds{jobid='%s',step='',task=''}[%ds])" % (
        raw_jobid, window)


def mem_query(raw_jobid: str) -> str:
    """PromQL for RSS bytes over time -- a gauge, so no ``rate()`` needed."""
    return "cgroup_memory_rss_bytes{jobid='%s',step='',task=''}" % raw_jobid


def _host_of(series: dict) -> str:
    return str(series["metric"].get("host", "?")).split(":")[0]


def host_series(raw_jobid: str, cpus_by_host: Dict[str, float],
                mem_total_by_host: Dict[str, float], start: int, end: int, step: int,
                sampling_period: int, client: PrometheusClient,
                timeout: Optional[float]) -> Dict[str, Dict[int, Dict[str, float]]]:
    """``{host: {epoch: {"CPU%": v, "MEM%": v}}}`` over ``[start, end]`` at ``step``.

    Each sample is divided by that host's own divisor -- cores allocated for CPU%,
    bytes allocated for MEM% -- so a host missing from ``cpus_by_host``/
    ``mem_total_by_host`` (no divisor resolved for it) is silently skipped rather
    than dividing by zero.
    """
    series: Dict[str, Dict[int, Dict[str, float]]] = {}
    try:
        cpu_result = client.query_range(cpu_query(raw_jobid, step, sampling_period),
                                        start, end, step, timeout)
    except Exception:
        cpu_result = []
    for result in cpu_result:
        host = _host_of(result)
        cpus = cpus_by_host.get(host)
        if not cpus:
            continue
        for stamp, value in result.get("values", []):
            try:
                pct = 100 * float(value) / cpus
            except (TypeError, ValueError):
                continue
            series.setdefault(host, {}).setdefault(int(float(stamp)), {})["CPU%"] = pct

    try:
        mem_result = client.query_range(mem_query(raw_jobid), start, end, step, timeout)
    except Exception:
        mem_result = []
    for result in mem_result:
        host = _host_of(result)
        total = mem_total_by_host.get(host)
        if not total:
            continue
        for stamp, value in result.get("values", []):
            try:
                pct = 100 * float(value) / total
            except (TypeError, ValueError):
                continue
            series.setdefault(host, {}).setdefault(int(float(stamp)), {})["MEM%"] = pct
    return series
