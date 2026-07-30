"""Reconstruct the utilization blob for a job that is still running.

jobstats writes its ``JS1:`` blob into sacct's AdminComment when a job *ends*, so
a running job has none and the blob-derived columns (CPU%/MEM%/GPU%/GMEM%) have
nothing to read. Every input is in Prometheus though, from the same exporters
jobstats itself falls back to while a job is live, so this module queries them and
assembles a dict in exactly the shape :mod:`jobscope.blob` decodes.

Producing the blob's own shape -- rather than a parallel set of numbers -- is the
point: :func:`jobscope.blob.blob_metrics` and :func:`jobscope.blob.blob_detail`
then work unchanged, and so do the summary and detail views, ``--csv``, ``plot``
and ``--diagnose``.

Two details are easy to get wrong:

* **Key on the raw job ID.** ``cgroup_*`` carries a real ``jobid`` label, but it
  holds the raw per-element ID: array element ``36410890_2`` appears as
  ``jobid="36410916"``. :attr:`jobscope.sacct.JobRecord.jobid_raw` has it.
* **Aggregate each metric the way jobstats does** -- utilization averaged over the
  runtime, memory peaked. Reversing them silently changes the numbers.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

from .prometheus import PrometheusClient
from .sacct import JobRecord

# Per-node host resources. jobstats reads these with the same reducers; ``step``
# and ``task`` are pinned empty to select the job-level cgroup rather than a
# per-step one (an ='' matcher also matches the label being absent, which is the
# case on exporters that do not emit it at all).
_HOST_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("cpus", "cgroup_cpus", "max"),
    ("total_time", "cgroup_cpu_total_seconds", "max"),
    ("used_memory", "cgroup_memory_rss_bytes", "max"),
    ("total_memory", "cgroup_memory_total_bytes", "max"),
)

# Per-GPU fields, keyed in the blob by minor_number as a string. Utilization is a
# mean and memory a high-water mark, matching jobstats' "maximum used/total".
_GPU_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("gpu_utilization", "nvidia_gpu_duty_cycle", "avg"),
    ("gpu_used_memory", "nvidia_gpu_memory_used_bytes", "max"),
    ("gpu_total_memory", "nvidia_gpu_memory_total_bytes", "max"),
)


def _host_query(metric: str, reducer: str, raw_jobid: str, duration: int) -> str:
    return "%s_over_time(%s{jobid='%s',step='',task=''}[%ds])" % (
        reducer, metric, raw_jobid, duration)


def _gpu_query(metric: str, reducer: str, raw_jobid: str, duration: int) -> str:
    """As above, but GPUs have no jobid label -- the job ID is a *value*.

    Intersecting with ``nvidia_gpu_jobId == <raw>`` both selects this job's GPUs
    and clips the window to the samples it owned them for. Both series come from
    the same exporter, so their label sets match, which PromQL's ``and`` requires.
    """
    return "%s_over_time((%s and nvidia_gpu_jobId == %s)[%ds:])" % (
        reducer, metric, raw_jobid, duration)


# jobstats stores byte counts as integers and utilization to one decimal, and
# blob_detail renders utilization with %g on that assumption -- so round to the
# same precision here, or a synthesized row prints as "93.1386%".
_ROUNDING = {"cpus": 0, "total_time": 1, "used_memory": 0, "total_memory": 0,
             "gpu_utilization": 1, "gpu_used_memory": 0, "gpu_total_memory": 0}


def _store_as(field: str, value: float):
    """Round ``value`` to the precision the stored blob uses for ``field``."""
    decimals = _ROUNDING.get(field, 1)
    return int(round(value)) if decimals == 0 else round(value, decimals)


def _host_of(series: dict) -> str:
    return str(series["metric"].get("host", "?")).split(":")[0]


def _value(series: dict) -> Optional[float]:
    try:
        return float(series["value"][1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def host_stats(raw_jobid: str, duration: int, at, client: PrometheusClient,
               timeout: Optional[float] = None) -> Dict[str, dict]:
    """Per-node CPU and host-memory fields, keyed by node name.

    Split out from :func:`synthesize_stats` because the live view needs exactly
    this: its GPU numbers come from its own collectors (which honour the
    instant-vs-``--avg`` choice), but CPU% and MEM% are cumulative either way --
    CPU-seconds over elapsed x cores, and peak RSS -- so there is nothing to vary.
    """
    nodes: Dict[str, dict] = {}
    for field, metric, reducer in _HOST_FIELDS:
        for series in _query(client, _host_query(metric, reducer, raw_jobid, duration),
                             at, timeout):
            value = _value(series)
            if value is not None:
                nodes.setdefault(_host_of(series), {})[field] = _store_as(field, value)
    return nodes


def gpu_stats(raw_jobid: str, duration: int, at, client: PrometheusClient,
              timeout: Optional[float] = None) -> Dict[str, dict]:
    """Per-node, per-GPU utilization and memory maps, reduced over the runtime."""
    nodes: Dict[str, dict] = {}
    for field, metric, reducer in _GPU_FIELDS:
        for series in _query(client, _gpu_query(metric, reducer, raw_jobid, duration),
                             at, timeout):
            value = _value(series)
            if value is None:
                continue
            minor = str(series["metric"].get("minor_number", "?"))
            nodes.setdefault(_host_of(series), {}).setdefault(
                field, {})[minor] = _store_as(field, value)
    return nodes


def stats_dict(duration: int, *node_maps: Dict[str, dict]) -> dict:
    """Merge per-node maps into the blob's own shape, or ``{}`` if all are empty.

    Top-level ``total_time`` is elapsed wall time, against which the per-node
    CPU-seconds are measured: blob_metrics divides by (total_time x cpus).
    """
    nodes: Dict[str, dict] = {}
    for node_map in node_maps:
        for node, fields in node_map.items():
            nodes.setdefault(node, {}).update(fields)
    return {"total_time": duration, "nodes": nodes} if nodes else {}


def synthesize_stats(record: JobRecord, client: PrometheusClient,
                     timeout: Optional[float] = None) -> dict:
    """Build a jobstats-shaped stats dict for ``record`` from Prometheus.

    Returns ``{}`` when nothing could be read, which leaves the blob columns
    showing ``-`` exactly as they did before. Never raises: a running job that
    Prometheus cannot answer for should degrade, not abort the report.

    MIG note: the blob keys GPUs by minor number, which MIG instances share, so a
    partitioned card collapses to one entry here -- the same limitation jobstats'
    own blob has. ``jobscope live`` keys by UUID and does not.
    """
    if not (record.jobid_raw and record.duration and record.duration > 0):
        return {}
    maps = [host_stats(record.jobid_raw, record.duration, record.end, client, timeout)]
    if record.gpus:
        maps.append(gpu_stats(record.jobid_raw, record.duration, record.end, client, timeout))
    return stats_dict(record.duration, *maps)


def _query(client: PrometheusClient, query: str, at, timeout: Optional[float]):
    try:
        return client.query(query, at, timeout)
    except Exception:
        return []


def fill_running(records: Dict[str, JobRecord], jobids, client: PrometheusClient,
                 timeout: Optional[float] = None, workers: int = 1) -> int:
    """Synthesize stats for every RUNNING record in ``jobids`` that has no blob.

    Mutates the records in place and returns how many were filled. Records that
    already carry a blob are left alone, so a finished job always reports the
    numbers Slurm stored rather than a recomputation.
    """
    pending = [records[jid] for jid in jobids
               if jid in records and records[jid].state == "RUNNING" and not records[jid].stats]
    if not pending:
        return 0

    filled = 0
    if workers > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            results = pool.map(lambda r: (r, synthesize_stats(r, client, timeout)), pending)
            for record, stats in results:
                if stats:
                    record.stats = stats
                    filled += 1
        return filled

    for record in pending:
        stats = synthesize_stats(record, client, timeout)
        if stats:
            record.stats = stats
            filled += 1
    return filled


def note_offline_gap(records: Dict[str, JobRecord], jobids) -> None:
    """Warn when running jobs cannot be filled because no endpoint is configured.

    Slurm writes the blob at job end, so a running job's utilization columns can
    only come from Prometheus. Say that plainly instead of printing a bare row of
    dashes that looks like the job used nothing.
    """
    running = [jid for jid in jobids
               if jid in records and records[jid].state == "RUNNING" and not records[jid].stats]
    if running:
        print("note: %d running job(s) have no stored utilization blob yet, and no Prometheus\n"
              "      endpoint is configured to reconstruct it -- their utilization columns are\n"
              "      blank. See 'jobscope config'." % len(running), file=sys.stderr)
