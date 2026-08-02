"""Reduce a job's raw series into the per-node, per-GPU figures a report shows.

The collectors each answer for one family: :mod:`jobscope.cpu` for ``cgroup_*``,
:mod:`jobscope.nvml` for ``nvidia_gpu_*``, :mod:`jobscope.dcgm` for the profiling
catalog. This assembles their answers into one job's stats, and is the only place
that knows a job's utilization comes from more than one source.

It reconstructs the shape :mod:`jobscope.blob` decodes, which is deliberate rather
than incidental: a running job has no stored blob, so the alternative is a second
set of renderers for jobs Slurm has not finished writing about yet.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from .cpu import host_stats
from .nvml import per_gpu_stats
from .prometheus import PrometheusClient
from .slurm import JobRecord


def stats_dict(duration: int, *node_maps: Dict[str, dict]) -> dict:
    """Merge per-node maps into the blob's own shape, or ``{}`` if all are empty.

    Top-level ``total_time`` is elapsed wall time, against which the per-node
    CPU-seconds are measured: the CPU% arithmetic divides by (total_time x cpus).
    """
    nodes: Dict[str, dict] = {}
    for node_map in node_maps:
        for node, fields in node_map.items():
            nodes.setdefault(node, {}).update(fields)
    return {"total_time": duration, "nodes": nodes} if nodes else {}


def synthesize_stats(record: JobRecord, client: PrometheusClient,
                     timeout: Optional[float] = None) -> dict:
    """Build a stats dict for ``record`` from Prometheus.

    Returns ``{}`` when nothing could be read, which leaves the utilization columns
    showing ``-`` rather than inventing zeros. Never raises.

    Works for a finished job as readily as a running one -- it reduces over
    ``record.duration`` ending at ``record.end``, and neither cares which -- which is
    what ``--no-blob`` uses to check the two sources against each other.

    MIG note: the blob keys GPUs by minor number, which MIG instances share, so a
    partitioned card collapses to one entry here -- the same limitation jobstats' own
    blob has. The running view keys by UUID and does not.
    """
    if not (record.jobid_raw and record.duration and record.duration > 0):
        return {}
    maps = [host_stats(record.jobid_raw, record.duration, record.end, client, timeout)]
    if record.gpus:
        maps.append(per_gpu_stats(record.jobid_raw, record.duration, record.end,
                                  client, timeout))
    return stats_dict(record.duration, *maps)


def needs_fill(record: JobRecord, force: bool = False) -> bool:
    """Whether this record's stats have to be built from Prometheus.

    Normally only a running job: it has no stored blob yet, so its utilization
    columns would be empty. With ``force`` -- ``--no-blob`` -- every record is
    rebuilt, including finished ones that already carry a blob, so the two sources
    can be compared on the same jobs.
    """
    return bool(force or (record.state == "RUNNING" and not record.stats))


def fill_running(records: Dict[str, JobRecord], jobids, client: PrometheusClient,
                 timeout: Optional[float] = None, workers: int = 1,
                 force: bool = False) -> int:
    """Synthesize stats for every record in ``jobids`` that needs it.

    Mutates the records in place and returns how many were filled. By default a
    record already carrying a blob is left alone, so a finished job reports the
    numbers Slurm stored rather than a recomputation; ``force`` overwrites them from
    Prometheus instead.
    """
    pending = [records[jid] for jid in jobids
               if jid in records and needs_fill(records[jid], force)]
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
