"""One selection layer over two very different job sources.

Jobs reach jobscope from either ``sacct`` (finished jobs, or specific IDs) or
``squeue`` (what is running now). Those paths share almost nothing: one streams
batches and reduces each metric over a closed ``[start, end]`` window, the other
reads a single instant and has to reconstruct the utilization blob Slurm has not
written yet.

Everything downstream is indifferent to that. This module is where the difference
stops: :func:`resolve` hands back the same ``(jobids, records, dcgm_data)`` chunks
either way, so a renderer -- and therefore every column, granularity and filter
flag -- works against both without knowing which it got. Keeping the two sources
behind one interface is what lets the CLI treat "which jobs" and "how to show
them" as independent choices.
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

from . import config
from .dcgm import BLOB_BACKED_KEYS, DEFAULT_SPECS, MetricSpec, compute_dcgm
from .errors import JobscopeError
from .running import (
    RunningSelection,
    aggregate_by_job,
    collect_averaged,
    collect_instant,
    collect_timeseries,
    discover_gpus,
    fetch_jobs,
    job_sort_key,
    running_records,
    per_gpu_by_node_minor,
)
from .job_ave_stats import fill_running, needs_fill, note_offline_gap
from . import timeseries
from .prometheus import PrometheusClient, client_from_config
from .report import (
    RenderOptions,
    combined_timeseries,
    context_pairs,
    cpu_timeseries,
    dcgm_timeseries,
    running_combined_timeseries,
    running_cpu_timeseries,
    running_timeseries,
    no_such_node,
)
from .slurm import (
    JobRecord,
    Selection,
    days_to_window,
    end_of_day,
    fetch,
    fetch_chunks,
    select_jobs,
)

# {jobid: (job-level metrics by header, per-GPU metrics by (node, minor))}
DcgmData = Dict[str, Tuple[dict, dict]]
Chunk = Tuple[List[str], Dict[str, JobRecord], DcgmData]

# The metrics the reconstructed blob is built from; enough for CPU%/MEM%/GPU%/GMEM%
# without the DCGM profiling block.
BLOB_SPECS: List[MetricSpec] = [s for s in DEFAULT_SPECS if s.key in BLOB_BACKED_KEYS]

RUNNING = "running"
FINISHED = "finished"
JOBIDS = "jobids"


@dataclass
class Request:
    """What to report on: the mode, its scope, and the filters."""

    mode: str = RUNNING
    jobids: List[str] = field(default_factory=list)
    # finished only
    days: Optional[int] = None
    lastn: Optional[int] = None
    starttime: Optional[str] = None
    endtime: Optional[str] = None
    state: str = "all"
    # filters, every mode
    user: Optional[str] = None          # None with all_users = every user
    all_users: bool = False
    account: Optional[str] = None
    partition: Optional[str] = None
    # running only. The same floor config.DEFAULT_MIN_ELAPSED names, in seconds:
    # this default is only reachable by a library caller (the CLI always passes
    # cli._min_elapsed()), and it used to disagree with it by an hour.
    min_elapsed: int = config.parse_duration(config.DEFAULT_MIN_ELAPSED)
    average: bool = False
    # Ignore the stored jobstats blob and read every metric from Prometheus, for a
    # finished job as well as a running one. The blob is a fast path -- one free
    # sacct field against several range queries -- so it stays preferred by
    # default; this exists to compare the two, and to be what a site without
    # jobstats runs on. See jobscope.doctor's --validate.
    no_blob: bool = False

    @property
    def running(self) -> bool:
        return self.mode == RUNNING


class Resolved(NamedTuple):
    """A resolved selection: the header context, plus chunks to render."""

    context: List[Tuple[str, str]]
    chunks: Iterator[Chunk]


def resolve(request: Request, cfg: config.Config, timeout: Optional[float],
            workers: int, specs: Optional[List[MetricSpec]]) -> Optional[Resolved]:
    """Select jobs and their metrics, or ``None`` when nothing matched.

    ``specs`` of ``None`` means the caller wants no DCGM columns (the ``--cpu``
    view), so those queries are skipped entirely -- a finished CPU-only report then
    needs no Prometheus at all.

    Both halves of ``dcgm_data`` are otherwise populated: the job-level metrics the
    default granularity renders, and the per-GPU ones ``--per-gpu`` needs. Neither
    branch pays extra for the second -- ``compute_dcgm`` returns it anyway, and the
    running equivalent is pure dict work over values already collected.
    """
    if request.running:
        return _resolve_running(request, cfg, timeout, workers, specs)
    return _resolve_historical(request, cfg, timeout, workers, specs)


# --- squeue -----------------------------------------------------------------

def _running_selection(request: Request) -> RunningSelection:
    return RunningSelection(jobids=list(request.jobids), partition=request.partition,
                         user=request.user, min_elapsed=request.min_elapsed)


def _running_context(selection: RunningSelection, jobs: dict, gpus: dict
                  ) -> List[Tuple[str, str]]:
    """Header context for a squeue selection.

    With explicit JOBIDs the -u/-p filters are bypassed, so name the jobs' actual
    owners rather than a filter that was not applied -- as :func:`context_pairs`
    does for the historical modes.
    """
    if selection.jobids:
        owners = sorted({job["user"] for job in jobs.values() if job.get("user")})
        user = ", ".join(owners) if owners else "(explicit job IDs)"
    else:
        user = selection.user or "(all users)"
    pairs = [("User", user)]
    if selection.partition:
        pairs.append(("Partition", selection.partition))
    pairs.append(("Select", selection.describe()))
    pairs.append(("GPUs", "%d across %d job(s)" % (len(gpus), len(jobs))))
    return pairs


def _report_no_running(selection) -> None:
    """Explain an empty running selection, naming every filter and how to widen it.

    The filters are worth spelling out here because the context block that would
    normally show them is not printed when there is nothing to report: the bare
    message reads as "this partition is idle" when it usually means the default
    user filter excluded whoever is on it.
    """
    print("No running jobs match (%s)." % selection.describe_filters(), file=sys.stderr)
    hints = selection.widening_hints()
    if hints:
        print("Widen it: %s." % "; or ".join(hints), file=sys.stderr)


def _resolve_running(request: Request, cfg: config.Config, timeout: Optional[float],
                     workers: int, specs: Optional[List[MetricSpec]]) -> Optional[Resolved]:
    """One chunk from squeue plus Prometheus, shaped like a sacct chunk.

    ``running_records`` synthesizes the blob Slurm has not written yet, so the records
    are indistinguishable from finished ones to everything downstream.
    """
    selection = _running_selection(request)
    jobs = fetch_jobs(selection, timeout)
    if not jobs:
        _report_no_running(selection)
        return None

    client = client_from_config(cfg, timeout)
    gpus = discover_gpus(client, jobs, timeout)
    if not gpus:
        # Not an error: a CPU-only selection legitimately has no GPUs.
        print("No GPU data in Prometheus for these jobs (CPU-only, or not yet scraped).",
              file=sys.stderr)

    # A --cpu report still needs the blob-backed GPU metrics, because running_records
    # assembles the blob from them -- but only those, so it does not pay for the
    # DCGM profiling queries whose columns it will not print.
    specs = specs or BLOB_SPECS
    metrics = (collect_averaged(client, jobs, gpus, specs, timeout, workers)
               if request.average else collect_instant(client, gpus, specs, timeout))
    records = running_records(jobs, gpus, metrics, specs, client, timeout)
    per_job = aggregate_by_job(metrics, specs)
    per_gpu = per_gpu_by_node_minor(metrics, gpus, specs)
    dcgm_data: DcgmData = {
        job["jobid"]: (per_job.get(raw, {}), per_gpu.get(raw, {}))
        for raw, job in jobs.items()}
    jobids = sorted((job["jobid"] for job in jobs.values()),
                    key=lambda jid: job_sort_key({"jobid": jid}))
    return Resolved(_running_context(selection, jobs, gpus),
                    iter([(jobids, records, dcgm_data)]))


# --- sacct ------------------------------------------------------------------

def sacct_selection(request: Request) -> Selection:
    """The sacct-side selection for a finished or explicit-ID request.

    This is where a scope becomes an actual window, and it must happen: with no
    ``starttime``, :func:`jobscope.slurm.select_jobs` falls back to ``now-30days``,
    so a request carrying only ``days`` would scan a month while the header
    truthfully claimed "last 1 day". ``days`` is kept alongside for that header.
    """
    start, end = request.starttime, request.endtime
    if request.days is not None:
        start, end = days_to_window(request.days)
    elif start and not end:
        end = end_of_day(start)     # -S alone selects that calendar day
    # A bare -N deliberately leaves the window unset: select_jobs widens it a rung at
    # a time until it holds enough jobs, and writes back the span it settled on. That
    # is much cheaper than scanning the whole default lookback to find a job or two.
    return Selection(user=request.user, jobids=list(request.jobids),
                     account=request.account, partition=request.partition,
                     state=request.state, lastn=request.lastn, days=request.days,
                     starttime=start, endtime=end,
                     all_users=request.all_users)


def _resolve_historical(request: Request, cfg: config.Config, timeout: Optional[float],
                        workers: int, specs: Optional[List[MetricSpec]]
                        ) -> Optional[Resolved]:
    """Streaming sacct chunks, each enriched with DCGM metrics as it arrives.

    Window selections stream so a wide selection shows rows as they land; a
    mid-stream failure can therefore surface after a partial table. Explicit
    JOBIDs render in one pass, because their context header names every owner.
    """
    selection = sacct_selection(request)
    jobids, desc = select_jobs(selection, timeout)
    if not jobids:
        print("No matching jobs for %s (%s)."
              % ("all users" if request.all_users else "user '%s'" % selection.user, desc),
              file=sys.stderr)
        return None

    if selection.jobids:
        records = fetch(jobids, timeout)
        context = context_pairs(selection, desc, records)
        chunks: Iterator[Tuple[List[str], Dict[str, JobRecord]]] = iter([(jobids, records)])
    else:
        context = context_pairs(selection, desc, {})  # the window branch reads no records
        chunks = fetch_chunks(jobids, timeout)

    return Resolved(context, _enrich(chunks, cfg, timeout, workers, specs,
                                     no_blob=request.no_blob))


def _enrich(chunks, cfg: config.Config, timeout: Optional[float], workers: int,
            specs: Optional[List[MetricSpec]],
            no_blob: bool = False) -> Iterator[Chunk]:
    """Attach DCGM metrics and fill running jobs' blobs, chunk by chunk.

    The client is built lazily and at most once: a selection with no GPU jobs, or a
    --cpu report over finished ones, never contacts Prometheus.
    """
    client: Optional[PrometheusClient] = None
    for chunk_ids, records in chunks:
        dcgm_data: DcgmData = {}
        if specs and any(j in records and records[j].gpus for j in chunk_ids):
            if client is None:
                client = client_from_config(cfg, timeout)
            dcgm_data = compute_dcgm(records, chunk_ids, specs, client, timeout, workers)
        client = _fill_running(records, chunk_ids, cfg, timeout, workers, client,
                               force=no_blob)
        yield chunk_ids, records, dcgm_data


def _fill_running(records, jobids, cfg, timeout, workers, client, force=False):
    """Rebuild the utilization blob for the jobs in this chunk that need one.

    A running job has no stored blob, so CPU%/MEM%/GPU%/GMEM% would all be empty.
    Every input is in Prometheus, so fill them from there. With ``force``
    (``--no-blob``) finished jobs are rebuilt too, ignoring what Slurm stored.

    An install with no endpoint configured stays fully offline: the fill is skipped
    with a note rather than an error -- except under ``force``, where there is no
    stored blob to fall back on and going quiet would print a table of dashes with
    no explanation.
    """
    if not any(j in records and needs_fill(records[j], force) for j in jobids):
        return client
    if client is None:
        try:
            client = client_from_config(cfg, timeout)
        except JobscopeError:
            if force:
                raise JobscopeError(
                    "--no-blob reads every metric from Prometheus, and no endpoint is\n"
                    "configured. Drop --no-blob to use the stored jobstats blob, or see\n"
                    "'jobscope doctor' for how to configure one.")
            note_offline_gap(records, jobids)
            return None
    fill_running(records, jobids, client, timeout, workers, force)
    return client


# --- the time-series granularity -------------------------------------------

def emit_timeseries(request: Request, cfg: config.Config, timeout: Optional[float],
                    workers: int, specs: List[MetricSpec], step: Optional[int],
                    options: RenderOptions, out=None) -> None:
    """Write the per-scrape CSV for ``request``, from whichever source applies.

    Unlike the table granularities this dispatches rather than returning chunks:
    the two emitters need genuinely different inputs (range queries over a closed
    window versus over ``[start, now]``), and forcing them into the chunk shape
    would buy nothing. Both write the identical schema, which is what
    ``jobscope plot`` depends on.

    ``out`` sends the CSV somewhere other than stdout, which is how ``--plot_ts``
    captures it and charts it in the same command.
    """
    if request.running:
        selection = _running_selection(request)
        jobs = fetch_jobs(selection, timeout)
        if not jobs:
            _report_no_running(selection)
            return
        client = client_from_config(cfg, timeout)
        if not options.combined and options.view == "cpu":
            match = timeseries.UnitFilter(options.nodename)
            collected = timeseries.running_host(
                jobs, client, timeout, workers, window=options.window, step=step,
                host_specs=options.cgroup_specs, match=match)
            running_cpu_timeseries(
                collected, timeseries.running_host_order(jobs, collected), options, out=out)
            match.check()
            return
        gpus = discover_gpus(client, jobs, timeout)
        if options.nodename:
            # Filter before collect_timeseries, so the other nodes' GPUs are never
            # queried rather than queried and discarded.
            kept = {u: g for u, g in gpus.items() if g.host == options.nodename}
            if not kept:
                raise no_such_node(options.nodename, {g.host for g in gpus.values()})
            gpus = kept
        if options.gpu_ids:
            # Same reason as --nodename above: narrow before collect_timeseries so the
            # other cards are never queried rather than queried and discarded.
            kept = {u: g for u, g in gpus.items() if str(g.csv_id) in options.gpu_ids}
            if not kept:
                raise JobscopeError(
                    "no rows for GPU %s in this selection; it used: %s"
                    % (", ".join(repr(g) for g in options.gpu_ids),
                       ", ".join(sorted({str(g.csv_id) for g in gpus.values()})) or "(none)"))
            gpus = kept
        samples = collect_timeseries(client, jobs, gpus, specs, timeout, workers, step,
                                     window=options.window)
        if options.combined:
            # warn=False: the GPU rows are the report here, so a job with no cgroup
            # data loses two columns rather than its row.
            collected = timeseries.running_host(
                jobs, client, timeout, workers, window=options.window, step=step,
                host_specs=options.cgroup_specs, warn=False)
            running_combined_timeseries(jobs, samples, gpus, specs, collected, options,
                                        out=out)
            return
        running_timeseries(jobs, samples, gpus, specs, options, out=out)
        return

    selection = sacct_selection(request)
    jobids, desc = select_jobs(selection, timeout)
    if not jobids:
        print("No matching jobs (%s)." % desc, file=sys.stderr)
        return
    records = fetch(jobids, timeout)
    client = client_from_config(cfg, timeout)
    # Collect and render are separate calls, but the collectors are generators, so
    # this still walks one job at a time -- the renderer pulls the next job's samples
    # only once it has written the last one's rows.
    match = timeseries.UnitFilter(options.nodename, options.gpu_ids)
    if not options.combined and options.view == "cpu":
        cpu_timeseries(
            timeseries.finished_host(jobids, records, client, timeout,
                                     window=options.window, step=step,
                                     host_specs=options.cgroup_specs, match=match),
            options, out=out)
    elif options.combined:
        combined_timeseries(
            timeseries.finished_gpu(jobids, records, specs, client, timeout,
                                    window=options.window, step=step, with_host=True,
                                    host_specs=options.cgroup_specs, match=match),
            specs, options, out=out)
    else:
        dcgm_timeseries(
            timeseries.finished_gpu(jobids, records, specs, client, timeout,
                                    window=options.window, step=step, match=match),
            specs, options, out=out)
    match.check()
