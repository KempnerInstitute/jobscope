"""NVML GPU metrics -- the ``nvidia_gpu_*`` series, and the job-to-GPU join.

Two GPU exporters are in play and they are not interchangeable. ``dcgm-exporter``
publishes the profiling catalog (:mod:`jobscope.dcgm`) keyed by an uppercase ``UUID``
label; the nvidia exporter publishes duty cycle and memory keyed by a lowercase
``uuid``, and it is the one that also publishes the series whose *value* is the job
id holding each card. That series is the only join between Slurm's world and either
exporter's, since neither carries a job label.

This module is the reduce-over-a-window half of that: the per-GPU figures a job's
utilization is built from. Discovery lives in :mod:`jobscope.dcgm` beside the query
builder that consumes it.
"""

from typing import Dict, Optional, Tuple

from . import config
from .blob import store_as
from .prometheus import PrometheusClient, query_value

# Per-GPU fields, keyed by minor_number as a string. Utilization is a mean and memory
# a high-water mark, matching jobstats' "maximum used/total" -- reversing the two
# silently changes the numbers rather than failing.
FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("gpu_utilization", "nvidia_gpu_duty_cycle", "avg"),
    ("gpu_used_memory", "nvidia_gpu_memory_used_bytes", "max"),
    ("gpu_total_memory", "nvidia_gpu_memory_total_bytes", "max"),
)


def window_query(metric: str, reducer: str, raw_jobid: str, duration: int) -> str:
    """``metric`` reduced over a job's window, restricted to the cards it held.

    GPUs have no jobid label -- the job id is a *value* -- so the selector is
    intersected with the join series. That both selects this job's cards and clips
    the window to the samples it owned them for. Both series come from the same
    exporter, so their label sets match, which PromQL's ``and`` requires.
    """
    return "%s_over_time((%s and %s == %s)[%ds:])" % (
        reducer, metric, config.gpu_join(), raw_jobid, duration)


def per_gpu_stats(raw_jobid: str, duration: int, at, client: PrometheusClient,
                  timeout: Optional[float] = None) -> Dict[str, dict]:
    """``{node: {field: {minor: value}}}`` reduced over the job's runtime.

    Keyed by minor number because that is what the stored blob uses. MIG instances
    share one, so a partitioned card collapses to a single entry -- the same
    limitation jobstats' blob has. The running view keys by UUID and does not.
    """
    nodes: Dict[str, dict] = {}
    for field, metric, reducer in FIELDS:
        query = window_query(metric, reducer, raw_jobid, duration)
        for labels, value in query_value(client, query, at, timeout):
            if value is None:
                continue
            minor = str(labels.get("minor_number", "?"))
            nodes.setdefault(config.host_of(labels), {}).setdefault(
                field, {})[minor] = store_as(field, value)
    return nodes
