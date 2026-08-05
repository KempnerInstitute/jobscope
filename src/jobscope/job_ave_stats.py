"""Reduce a job's raw series into the per-node, per-GPU figures a report shows.

The collectors each answer for one family: :mod:`jobscope.cpu` for ``cgroup_*``,
:mod:`jobscope.nvml` for ``nvidia_gpu_*``, :mod:`jobscope.dcgm` for the profiling
catalog. This assembles their answers into one job's stats, and is the only place
that knows a job's utilization comes from more than one source.

It reconstructs the shape :mod:`jobscope.jobstats` decodes, which is deliberate rather
than incidental: a running job has no stored summary, so the alternative is a second
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
    """Merge per-node maps into the summary's own shape, or ``{}`` if all are empty.

    Top-level ``total_time`` is elapsed wall time, against which the per-node
    CPU-seconds are measured: the CPU% arithmetic divides by (total_time x cpus).
    """
    nodes: Dict[str, dict] = {}
    for node_map in node_maps:
        for node, fields in node_map.items():
            nodes.setdefault(node, {}).update(fields)
    return {"total_time": duration, "nodes": nodes} if nodes else {}


def synthesize_stats(record: JobRecord, client: PrometheusClient,
                     timeout: Optional[float] = None,
                     average: bool = False) -> dict:
    """Build a stats dict for ``record`` from Prometheus.

    Returns ``{}`` when nothing could be read, which leaves the utilization columns
    showing ``-`` rather than inventing zeros. Never raises.

    Works for a finished job as readily as a running one -- it reduces over
    ``record.duration`` ending at ``record.end``, and neither cares which -- which is
    what ``--no-jobstats`` uses to check the two sources against each other.

    MIG note: the jobstats summary keys GPUs by minor number, which MIG instances share, so a
    partitioned card collapses to one entry here -- the same limitation jobstats' own
    summary has. The running view keys by UUID and does not.
    """
    if not (record.jobid_raw and record.duration and record.duration > 0):
        return {}
    # Host fields are cumulative whatever the window (see cpu.host_stats), so only
    # the GPU half takes the instant-versus-fold choice -- and it takes it from the
    # record's own state, so this summary agrees with the squeue view of the same job.
    maps = [host_stats(record.jobid_raw, record.duration, record.end, client, timeout)]
    if record.gpus:
        maps.append(per_gpu_stats(record.jobid_raw, record.duration, record.end,
                                  client, timeout,
                                  instant=record.unfinished and not average))
    return stats_dict(record.duration, *maps)


def needs_fill(record: JobRecord, force: bool = False) -> bool:
    """Whether this record's stats have to be built from Prometheus.

    Normally only a running job: it has no stored summary yet, so its utilization
    columns would be empty. With ``force`` -- ``--no-jobstats`` -- every record is
    rebuilt, including finished ones that already carry a jobstats summary, so the two sources
    can be compared on the same jobs.
    """
    return bool(force or (record.state == "RUNNING" and not record.stats))


def fill_running(records: Dict[str, JobRecord], jobids, client: PrometheusClient,
                 timeout: Optional[float] = None, workers: int = 1,
                 force: bool = False, average: bool = False) -> int:
    """Synthesize stats for every record in ``jobids`` that needs it.

    Mutates the records in place and returns how many were filled. By default a
    record already carrying a jobstats summary is left alone, so a finished job reports the
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
            results = pool.map(
                lambda r: (r, synthesize_stats(r, client, timeout, average)), pending)
            for record, stats in results:
                if stats:
                    record.stats = stats
                    filled += 1
        return filled

    for record in pending:
        stats = synthesize_stats(record, client, timeout, average)
        if stats:
            record.stats = stats
            filled += 1
    return filled


def slurm_host_map(metrics, duration: int) -> Dict[str, dict]:
    """Slurm's CPU and memory accounting, in the summary's per-node shape.

    One entry, not one per node, because sacct accounts a job's CPU-seconds and memory
    as *job* totals -- and CPU%/MEM% are sums over nodes, so a single entry carrying
    the totals yields the right job-level figure. The consequence is that ``--nodename``
    has nothing to narrow on a slurm-sourced job, which is a real limit of the source
    rather than something to paper over with a hostname it cannot support.

    ``total_time`` here is CPU-seconds, matching the summary's per-node field of that name
    -- the top-level ``total_time`` is wall clock. That overload is the summary's, not
    ours; getting the two the wrong way round produces a CPU% off by the core count.
    """
    fields = {}
    if metrics.total_cpu_s is not None and metrics.cpu_time_s:
        # cpu_time_s is elapsed x cores, so dividing it back out recovers the core
        # count without a second sacct field to disagree with.
        cores = metrics.cpu_time_s / duration if duration else 0
        if cores:
            fields["total_time"] = metrics.total_cpu_s
            fields["cpus"] = int(round(cores))
    if metrics.used_mem_bytes is not None and metrics.req_mem_bytes:
        fields["used_memory"] = metrics.used_mem_bytes
        fields["total_memory"] = metrics.req_mem_bytes
    return {SLURM_NODE: fields} if fields else {}


# Not a hostname, and deliberately not one: the figures are job totals, so labelling
# them with a node would claim a per-node measurement Slurm did not make.
SLURM_NODE = "(slurm)"


def accounted(metrics, record: JobRecord) -> bool:
    """Whether Slurm's CPU accounting for ``record`` can be believed.

    ``TotalCPU=0`` is read as *not gathered*, not as an idle job, whatever the job's
    state. A process that ran at all burns some CPU -- even a sleeping shell -- so a
    literal zero over any real elapsed time is a site where ``jobacct_gather`` is not
    recording it. Measured here: ``TotalCPU=00:00:00`` on a finished 128-core job that
    ran 1h29, whose summary reports CPU% 6, and ``sstat`` returning no rows at all.

    That is :func:`jobscope.extra_metric._ratio`'s own rule -- "None rather than 0: a
    job whose denominator Slurm never recorded has *unknown* utilization, and reporting
    0 would make it look idle" -- applied to the numerator. The cost is a genuinely
    idle job reading "-" instead of 0, which is the safe direction to be wrong in.
    """
    return bool(metrics.total_cpu_s)


def apply_slurm_host(records: Dict[str, JobRecord], jobids,
                     timeout: Optional[float] = None, override: bool = False) -> int:
    """Serve host stats from Slurm's accounting, for the records it applies to.

    For the site whose CPU%/MEM% would otherwise be blank: no jobstats summary, and no
    cgroup exporter for Prometheus to read.

    ``override`` when the preference puts slurm ahead of the jobstats summary, in which case it
    *replaces* the host fields rather than only filling gaps -- naming a source has to
    mean the numbers come from it, exactly as ``--gpu-source dcgm`` means GPU% is
    measured rather than read back. Where Slurm then has nothing, the column reads "-"
    rather than quietly reverting to the jobstats summary, because a header line claiming slurm over
    jobstats-derived numbers would be worse than an honest gap.
    """
    from . import extra_metric
    pending = [jid for jid in jobids if jid in records
               and (override or not _has_host_fields(records[jid]))]
    if not pending:
        return 0
    try:
        found = extra_metric.collect(pending, timeout)
    except Exception:
        found = {}
    filled = 0
    for jid in pending:
        record, metrics = records[jid], found.get(jid)
        nodes = ({} if metrics is None or not accounted(metrics, record)
                 else slurm_host_map(metrics, record.duration or 0))
        if not nodes and not override:
            continue
        stats = dict(record.stats or {})
        stats.setdefault("total_time", record.duration or 0)
        merged = {node: dict(fields) for node, fields
                  in (stats.get("nodes") or {}).items()}
        if override:
            # Drop what another source measured, so the row cannot mix a slurm-attributed
            # CPU% with a jobstats-derived one. The GPU maps stay: they are a different axis.
            for fields in merged.values():
                for key in ("total_time", "cpus", "used_memory", "total_memory"):
                    fields.pop(key, None)
            merged = {node: fields for node, fields in merged.items() if fields}
        merged.update(nodes)
        stats["nodes"] = merged
        record.stats = stats
        filled += 1 if nodes else 0
    return filled


def _has_host_fields(record: JobRecord) -> bool:
    """Whether a record already carries CPU/memory figures from any source."""
    for node in ((record.stats or {}).get("nodes") or {}).values():
        if any(key in node for key in ("total_time", "cpus", "used_memory")):
            return True
    return False


def note_missing_host_series(records: Dict[str, JobRecord], jobids) -> None:
    """Say why running CPU%/MEM% are blank when an endpoint *is* configured.

    :func:`note_offline_gap` covers the no-endpoint case. This is the other one, and it
    is the one that reads as a jobscope bug: the GPU columns are full, the CPU ones are
    dashes, and nothing connects that to an exporter. It happens wherever the
    ``cgroup_*`` series do not cover the job -- either the exporter is not deployed on
    the node, or it is running but stuck on cgroups from jobs that have already
    finished, which is what this cluster does on the two nodes that report at all.

    Once per report, not once per job: a hundred running jobs share one cause.
    """
    blind = [jid for jid in jobids if jid in records
             and records[jid].state == "RUNNING"
             and not _has_host_fields(records[jid])]
    if not blind:
        return
    print("note: %d running job(s) have no CPU%%/MEM%% -- no cgroup_* series covers them.\n"
          "      A running job has no stored summary (Slurm writes it at job end), so those\n"
          "      columns can only come from the cgroup exporter. Run 'jobscope probe' to\n"
          "      see how many hosts it reports on." % len(blind), file=sys.stderr)


def note_offline_gap(records: Dict[str, JobRecord], jobids) -> None:
    """Warn when running jobs cannot be filled because no endpoint is configured.

    Slurm writes the jobstats summary at job end, so a running job's utilization columns can
    only come from Prometheus. Say that plainly instead of printing a bare row of
    dashes that looks like the job used nothing.
    """
    running = [jid for jid in jobids
               if jid in records and records[jid].state == "RUNNING" and not records[jid].stats]
    if running:
        print("note: %d running job(s) have no stored utilization summary yet, and no Prometheus\n"
              "      endpoint is configured to reconstruct it -- their utilization columns are\n"
              "      blank. See 'jobscope config'." % len(running), file=sys.stderr)
