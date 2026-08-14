"""One selection layer over two very different job sources.

Jobs reach jobscope from either ``sacct`` (finished jobs, or specific IDs) or
``squeue`` (what is running now). Those paths share almost nothing: one streams
batches and reduces each metric over a closed ``[start, end]`` window, the other
reads a single instant and has to reconstruct the utilization summary Slurm has not
written yet.

Everything downstream is indifferent to that. This module is where the difference
stops: :func:`resolve` hands back the same ``(jobids, records, dcgm_data)`` chunks
either way, so a renderer -- and therefore every column, granularity and filter
flag -- works against both without knowing which it got. Keeping the two sources
behind one interface is what lets the CLI treat "which jobs" and "how to show
them" as independent choices.
"""

import itertools
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

from . import config, cpu, dcgm, jobstats, rows, timeseries
from .dcgm import JobGpuData, MetricSpec, compute_dcgm
from .errors import JobscopeError
from .job_ave_stats import (
    apply_slurm_host,
    fill_running,
    needs_fill,
    note_missing_host_series,
    note_offline_gap,
)
from .prometheus import PrometheusClient, client_from_config
from .report import (
    RenderOptions,
    combined_timeseries,
    context_pairs,
    cpu_timeseries,
    dcgm_timeseries,
    known_pairs,
    narrowing_pairs,
    no_such_node,
    running_combined_timeseries,
    running_cpu_timeseries,
    running_timeseries,
    sampled_pair,
    source_pair,
)
from .running import (
    Gpu,
    RunningJob,
    RunningSelection,
    aggregate_by_job,
    collect_averaged,
    collect_instant,
    collect_timeseries,
    discover_gpus,
    fetch_jobs,
    format_duration,
    host_stats_for,
    job_sort_key,
    note_missing_gpu_join,
    per_gpu_by_node_minor,
    per_node_pooled,
    running_records,
)
from .slurm import (
    SLICE_SECONDS,
    JobRecord,
    Selection,
    days_to_window,
    describe_window,
    end_of_day,
    fetch,
    fetch_chunks,
    fetch_window,
    select_jobs,
    window_slices,
)

# {jobid: JobGpuData(job-level by header, per-GPU by (node, minor), per-node by node)}
DcgmData = Dict[str, JobGpuData]
Chunk = Tuple[List[str], Dict[str, JobRecord], DcgmData]

def jobstats_specs() -> List[MetricSpec]:
    """The metrics the reconstructed summary is built from -- enough for
    CPU%/MEM%/GPU%/GMEM% without the DCGM profiling block.

    A function, not a module constant: dcgm.set_preference() replaces the catalog it
    reads, so a value computed at import would freeze the default preference and
    --gpu-source would pick a source the summary then ignored.
    """
    active = dcgm.catalog()
    return [s for s in active.default_specs if s.key in active.jobstats_backed_keys]

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
    # Ignore the summary jobstats stored and read every metric from Prometheus, for a
    # finished job as well as a running one. The summary is a fast path -- one free
    # sacct field against several range queries -- so it stays preferred by
    # default; this exists to compare the two, and to be what a site without
    # jobstats runs on.
    no_jobstats: bool = False

    @property
    def running(self) -> bool:
        return self.mode == RUNNING


class Resolved(NamedTuple):
    """A resolved selection: the header context, plus chunks to render."""

    context: List[Tuple[str, str]]
    chunks: Iterator[Chunk]
    # Whether every value spans its job's whole runtime, which decides both the summary's
    # weighting and what the notes say (see report.averaging_note). Reported rather than
    # left for the caller to re-derive: only here are the records in hand *and* the cap
    # applied, and the renderer cannot ask either way -- it picks the tallies' unit before
    # the first chunk arrives. A finished record is folded by construction; a running one
    # only when asked and within the cap.
    folded: bool = True
    # How many jobs the selection holds, known here before the first chunk is yielded.
    # The detail views need it *up front*: they print a job's block before knowing whether
    # another follows, and what they draw under it depends on whether this is one job or
    # many (see report.DetailRenderer).
    total: int = 0


def resolve(request: Request, cfg: config.Config, timeout: Optional[float],
            workers: int, specs: Optional[List[MetricSpec]],
            nodename: Optional[str] = None, gpu_ids=(),
            host_specs=None) -> Optional[Resolved]:
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
        return _resolve_running(request, cfg, timeout, workers, specs, nodename,
                                gpu_ids, host_specs)
    return _resolve_historical(request, cfg, timeout, workers, specs, nodename,
                               gpu_ids, host_specs)


# --- squeue -----------------------------------------------------------------

def _running_selection(request: Request) -> RunningSelection:
    # Every filter the Request carries, or squeue is asked a wider question than the
    # header claims: -A used to be dropped here, so `jobscope -A other_lab` reported
    # every account of yours and said nothing about it.
    return RunningSelection(jobids=list(request.jobids), partition=request.partition,
                         account=request.account, user=request.user,
                         min_elapsed=request.min_elapsed)


def _running_context(selection: RunningSelection, jobs: dict, gpus: dict,
                     specs=None, host_specs=None,
                     average: bool = False) -> List[Tuple[str, str]]:
    """Header context for a squeue selection.

    The squeue counterpart of :func:`context_pairs`, and deliberately the same shape:
    with explicit JOBIDs the -u/-A/-p filters are bypassed, so name the jobs' actual
    owner, account and partition; otherwise restate the filters, unfiltered ones
    included. squeue already reports ``%a`` and ``%P`` for every job, so the JOBID
    branch costs nothing extra.

    Not folded into ``context_pairs`` itself, though the two want to be one function:
    this branch also carries a GPUs line between Select and Source, and asks
    ``source_pair`` for ``have_jobstats=False``, neither of which a ReportContext can
    say today. The shared half is :func:`known_pairs`. Whatever changes here changes
    there.

    The provenance line matters more here than in the historical modes, not less: a
    running job has no stored summary, so its GPU% is measured rather than read back, and
    which exporter measured it is the one thing the numbers cannot say themselves.
    """
    if selection.jobids:
        owners = sorted({job["user"] for job in jobs.values() if job.get("user")})
        user = ", ".join(owners) if owners else "(explicit job IDs)"
        pairs = [("User", user)]
        pairs += known_pairs([
            ("Account", tuple(sorted({job["account"] for job in jobs.values()
                                      if job.get("account")}))),
            ("Partition", tuple(sorted({job["partition"] for job in jobs.values()
                                        if job.get("partition")}))),
        ])
    else:
        pairs = [("User", selection.user or "(all users)"),
                 ("Account", selection.account or "(all accounts)"),
                 ("Partition", selection.partition or "(all partitions)")]
    pairs.append(("Select", selection.describe()))
    pairs.append(("GPUs", "%d across %d job(s)" % (len(gpus), len(jobs))))
    # Every job here came from squeue, so all of them are unfinished by definition.
    return (pairs + source_pair(specs, have_jobstats=False, host_specs=host_specs)
            + sampled_pair(specs, True, average, host_specs))


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


# Jobs per streamed batch on the running path. Small, because the point is the first row
# rather than the last: ~8 jobs is ~46 queries, so a row is on screen inside a second and
# several batches fit inside the query burst. slurm.JOBS_PER_CHUNK is 200 for the sacct
# path, where a batch is one cheap sacct call rather than a query per job per metric.
RUNNING_JOBS_PER_CHUNK = 8

# Below this many jobs the fan-out finishes before anyone wonders, so the upfront estimate
# would be noise. Above it the note is the difference between pacing and hanging.
_ESTIMATE_FROM_JOBS = 40


def _note_running_cost(jobs: int, cards: int, specs: int, rate: float) -> None:
    """Say what a wide running sweep is about to cost, before the first row.

    The average is one query per (job, metric) and cannot be made fewer -- see
    prometheus.RateLimiter. So rather than refuse a wide selection or silently downgrade
    it, say the size, say roughly how long, and say what would make it smaller. Estimated
    from the jobs that actually resolved cards, since jobs without any are absent from
    collect_averaged's fan-out -- which is why 110 jobs measured 628 queries and not 770.
    """
    if jobs < _ESTIMATE_FROM_JOBS or not cards:
        return
    queries = cards * specs
    lines = ["note: %d running jobs -- about %d queries" % (jobs, queries)]
    if rate > 0 and queries / rate >= 5:
        secs = queries / rate
        lines[0] += ", ~%s at %g/s" % (format_duration(int(secs)), rate)
    print(lines[0] + ".", file=sys.stderr)
    print("      Rows print as they arrive; Ctrl-C to stop early.", file=sys.stderr)
    print("      Narrow with -p, -u or --min-elapsed to cut the query count.",
          file=sys.stderr)


def _running_chunks(jobs: Dict[int, RunningJob], gpus: Dict[str, Gpu],
                    jobids: List[str], specs: List[MetricSpec], requested,
                    client: PrometheusClient, timeout: Optional[float], workers: int,
                    folded: bool, host_specs, rate: float) -> Iterator[Chunk]:
    """Query the selection in batches, yielding each as it lands.

    Mirrors :func:`jobscope.slurm.fetch_chunks`' contract, which the renderer already
    relies on: ``ready_ids`` is the next run of ids in display order, ``records`` and
    ``dcgm_data`` are the cumulative dicts shared across yields, and concatenating every
    ``ready_ids`` reproduces ``jobids`` exactly. SummaryRenderer.add prints and flushes
    each row, so batching here is what turns a wide sweep from a blank wait into a table
    filling in, and finish() still computes the summary and the bars from everything.

    Only the averaged path batches. ``collect_instant`` is a single grouped query covering
    every card, so splitting it would *add* one query per batch to a path that already
    returns in about a second -- nothing to wait for, so nothing to stream.

    The notes that count jobs run after the last batch, not inside the loop: they report a
    total ("N running job(s) have no CPU%/MEM%"), and per batch they would print once per
    batch. After the final yield puts them between the rows and the summary block, which is
    where a note about the rows belongs.
    """
    batches = ([jobids] if not folded else
               [jobids[i:i + RUNNING_JOBS_PER_CHUNK]
                for i in range(0, len(jobids), RUNNING_JOBS_PER_CHUNK)])
    if folded:
        _note_running_cost(len(jobs), len({g.jobid for g in gpus.values()}),
                           len(specs), rate)
    by_display = {job["jobid"]: raw for raw, job in jobs.items()}
    # Once for the whole selection, not once per batch: the cgroup queries are grouped
    # over every job handed to them, so a per-batch call multiplied them 14-fold on one
    # 110-job partition. See running.host_stats_for.
    hosts = host_stats_for(jobs, client, timeout)
    records: Dict[str, JobRecord] = {}
    dcgm_data: DcgmData = {}

    for batch in batches:
        raws = {by_display[jid] for jid in batch if jid in by_display}
        here = {raw: jobs[raw] for raw in raws}
        # Gpu.jobid is the raw id, the same key collect_averaged groups by -- so filtering
        # the card map is what confines the fan-out to this batch.
        cards = {uuid: gpu for uuid, gpu in gpus.items() if gpu.jobid in raws}
        metrics = (collect_averaged(client, here, cards, specs, timeout, workers)
                   if folded else collect_instant(client, gpus, specs, timeout))
        if not folded:
            here, cards = jobs, gpus
        records.update(running_records(here, cards, metrics, specs, client, timeout,
                                       hosts=hosts))
        per_job = aggregate_by_job(metrics, specs)
        per_gpu = per_gpu_by_node_minor(metrics, cards, specs)
        # Pooled from the same UUID-keyed metrics rather than from per_gpu, which has
        # already collapsed MIG siblings onto a shared (node, minor) key -- see
        # running.per_node_pooled.
        per_node = per_node_pooled(metrics, cards, specs)
        dcgm_data.update({
            job["jobid"]: JobGpuData(per_job.get(raw, {}), per_gpu.get(raw, {}),
                                     per_node.get(raw, {}))
            for raw, job in here.items()})
        yield batch, records, dcgm_data

    # The same two steps the sacct path takes in _enrich, and they belong here more than
    # there: a running job is the case with no stored summary to fall back on, so it is
    # where a named slurm source has something to add and where a missing cgroup
    # exporter is the difference between a CPU% and a dash.
    host = cpu.catalog()
    if "slurm" in host.preference:
        apply_slurm_host(records, jobids, timeout,
                         override=host.resolved.source_of("CPU%") == "slurm")
    _note_host_gap(records, jobids, host_specs)


def _resolve_running(request: Request, cfg: config.Config, timeout: Optional[float],
                     workers: int, specs: Optional[List[MetricSpec]],
                     nodename: Optional[str] = None,
                     gpu_ids=(), host_specs=None) -> Optional[Resolved]:
    """Set up a squeue selection, then hand back batches to render as they arrive.

    Everything here is one call and has to happen before the first row: squeue, the card
    discovery the whole selection joins through, and the context block, which reports the
    job and GPU counts. The per-job metric queries are the expensive part and the only part
    that streams -- see :func:`_running_chunks`.

    ``running_records`` synthesizes the jobstats summary Slurm has not written yet, so the
    records are indistinguishable from finished ones to everything downstream.
    """
    selection = _running_selection(request)
    jobs = fetch_jobs(selection, timeout)
    if not jobs:
        _report_no_running(selection)
        return None
    cap = cfg.defaults.max_running_jobs
    if len(jobs) > cap:
        # A backstop against a typo'd sweep, not a cost policy: the pacing below would
        # work through thousands of jobs, for minutes, having said so first.
        raise JobscopeError(
            "%d running jobs is more than one sweep will cover (limit %d).\n"
            "Narrow with -p, -u or --min-elapsed, or raise [defaults] max_running_jobs."
            % (len(jobs), cap))

    client = client_from_config(cfg, timeout)
    gpus = discover_gpus(client, jobs, timeout)
    if not gpus:
        # Not an error: a CPU-only selection legitimately has no GPUs.
        print("No GPU data in Prometheus for these jobs (CPU-only, or not yet scraped).",
              file=sys.stderr)

    # A --cpu report still needs the jobstats-backed GPU metrics, because running_records
    # assembles the jobstats summary from them -- but only those, so it does not pay for
    # the DCGM profiling queries whose columns it will not print.
    # What the *report* asked for, before the summary-shaped fallback below. --cpu passes
    # None and still needs JOBSTATS_SPECS queried to reconstruct the jobstats summary, but
    # it prints no GPU column -- so naming a source for one would describe a column that
    # is not there.
    requested = specs
    specs = specs or jobstats_specs()
    # The GPU counterpart of the host note, and only where GPU columns are actually
    # printed: under --cpu there is no blank column to explain, and saying so would be
    # noise about a view the caller did not ask for. Once, before any batch -- it counts
    # the whole selection.
    if requested:
        note_missing_gpu_join(jobs, gpus)
    jobids = sorted((job["jobid"] for job in jobs.values()),
                    key=lambda jid: job_sort_key({"jobid": jid}))
    folded = request.average
    return Resolved(_running_context(selection, jobs, gpus, requested, host_specs,
                                     average=folded),
                    _running_chunks(jobs, gpus, jobids, specs, requested, client, timeout,
                                    workers, folded, host_specs,
                                    cfg.max_queries_per_second),
                    folded=folded, total=len(jobids))


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
                        workers: int, specs: Optional[List[MetricSpec]],
                        nodename: Optional[str] = None, gpu_ids=(),
                        host_specs=None) -> Optional[Resolved]:
    """Streaming sacct chunks, each enriched with DCGM metrics as it arrives.

    Window selections stream so a wide selection shows rows as they land; a
    mid-stream failure can therefore surface after a partial table. Explicit
    JOBIDs render in one pass, because their context header names every owner.
    """
    narrowing = narrowing_pairs(nodename, gpu_ids)
    selection = sacct_selection(request)

    # A plain window selection never lists its ids: it streams slice by slice, and the
    # first slice is drawn here so an empty selection can still be reported as empty
    # and so the detail views get their "one job or many" answer before rendering.
    # -N and explicit JOBIDs keep the listing pass; both need every id up front.
    if not selection.jobids and not _walks_back(selection):
        return _resolve_streamed(request, selection, cfg, timeout, workers, specs,
                                 narrowing, nodename, gpu_ids, host_specs)

    jobids, desc = select_jobs(selection, timeout)
    if not jobids:
        print("No matching jobs for %s (%s)."
              % ("all users" if request.all_users else "user '%s'" % selection.user, desc),
              file=sys.stderr)
        return None

    if selection.jobids:
        records = fetch(jobids, timeout)
        context = context_pairs(rows.build_context(selection, desc, records),
                                specs, host_specs,
                                average=request.average) + narrowing
        # The only mode that can name a job which has not ended, and the only one holding
        # every record before the first chunk is yielded. A window selection takes the
        # branch below and is finished by construction: states_for() returns only finished
        # states and _query_ids filters again on UNFINISHED_STATES.
        #
        # No cap here: this path queries per record either way (dcgm_for_job, once per
        # job), so `average` picks a windowed reduction over a bare selector and does not
        # change how many queries there are. Only the squeue path fans out.
        folded = not rows.any_unfinished(records) or request.average
        chunks: Iterator[Tuple[List[str], Dict[str, JobRecord]]] = iter([(jobids, records)])
    else:
        context = context_pairs(rows.build_context(selection, desc, {}),
                                specs, host_specs,
                                average=request.average) + narrowing
        folded = True       # a window selection holds only finished jobs
        chunks = fetch_chunks(jobids, timeout)

    return Resolved(context, _enrich(chunks, cfg, timeout, workers, specs,
                                     no_jobstats=request.no_jobstats,
                                     nodename=nodename, gpu_ids=gpu_ids,
                                     host_specs=host_specs, average=request.average),
                    folded=folded, total=len(jobids))


def _walks_back(selection) -> bool:
    """Whether ``-N`` will day-walk this selection instead of taking the window.

    The same condition ``select_jobs`` applies, named once so the two cannot disagree
    about which path a selection is on -- they would disagree silently, by one of them
    listing ids the other had already streamed.
    """
    return (selection.lastn is not None
            and not selection.starttime and not selection.endtime)


# Below this many projected Prometheus queries, the fan-out finishes before anyone
# would have read a warning about it. Chosen against the pacing default: 3000 queries
# at 50/s is a minute, which is about where a command stops feeling like it is working
# and starts feeling like it is stuck.
NOTEWORTHY_QUERIES = 3000


def _projected_cost(ids: List[str], records: Dict[str, JobRecord], slices: int,
                    specs: Optional[List[MetricSpec]], cfg: config.Config):
    """``(jobs, gpu_jobs, queries, seconds)`` this selection is heading for.

    **Projected from the first slice, not counted.** Counting means asking sacct for
    every row before rendering any, which is the expensive thing the streaming path
    exists to avoid -- measured, listing a cluster-wide day cost 1.66 GB inside sacct
    whatever output format it was asked for, and no field list or id batching changes
    that. A one-hour probe over-projected a full day by ~31%, which is the direction a
    warning should be wrong in.

    ``None`` when there is nothing to warn about: no GPU jobs, no specs (``--cpu`` and
    ``--no-dcgm`` never contact Prometheus), or no pacing configured to project against.
    """
    if not specs or not ids or cfg.max_queries_per_second <= 0:
        return None
    with_gpus = sum(1 for jid in ids if jid in records and records[jid].gpus)
    if not with_gpus:
        return None
    jobs = len(ids) * slices
    gpu_jobs = with_gpus * slices
    # One per (reducer, uuid_label) group rather than one per metric -- seven metrics
    # come to two groups, and projecting per metric would overstate it by 3.5x.
    per_job = len({dcgm.group_key(spec) for spec in specs})
    # Discovery is no longer one of them: it is batched by bucket, so it costs roughly
    # the span in buckets however many jobs there are (dcgm.discover_gpus_batch), plus
    # a per-job fallback for the few a step grid cannot see -- measured at a few
    # percent, and inside the "about" this figure is already hedged with.
    queries = gpu_jobs * per_job + _buckets(selection_span(slices))
    return jobs, gpu_jobs, queries, queries / cfg.max_queries_per_second


def selection_span(slices: int) -> int:
    """Seconds the selection covers, from its slice count.

    A slice is a day by construction (:data:`jobscope.slurm.SLICE_SECONDS`), which is
    close enough for a projection and avoids re-parsing the window here.
    """
    return slices * SLICE_SECONDS


def _buckets(span: int) -> int:
    return max(1, -(-span // dcgm.DISCOVERY_BUCKET_SECONDS))


def _note_cost(ids, records, selection, specs, cfg: config.Config) -> None:
    """Say what a wide selection is about to cost Prometheus, before it spends it.

    Replaces the note the id listing used to print, which could open with an exact
    count because it had already paid for one. This cannot, and says the two things
    that count instead: roughly how many queries, and roughly how long at the rate the
    server will actually see them. Both are levers the reader can pull -- which the
    bare "this will be slow" was not.
    """
    slices = len(window_slices(*selection.window()))
    projected = _projected_cost(ids, records, slices, specs, cfg)
    if projected is None:
        return
    jobs, gpu_jobs, queries, seconds = projected
    if queries < NOTEWORTHY_QUERIES:
        return
    about = "~%d" % jobs if slices > 1 else "%d" % jobs
    print("note: %s jobs, %s with GPUs -- about %d Prometheus queries, ~%s at the"
          " configured %g queries/s.\n"
          "      --no-dcgm skips Prometheus entirely (jobstats columns only);"
          " [prometheus] max_queries_per_second\n"
          "      trades wall clock for load on the server. Or narrow the selection."
          % (about, "~%d" % gpu_jobs if slices > 1 else "%d" % gpu_jobs,
             queries, _duration(seconds), cfg.max_queries_per_second),
          file=sys.stderr)


def _duration(seconds: float) -> str:
    """``45s`` / ``8.6 min`` / ``1.4 h`` -- one figure, in the unit a reader thinks in."""
    if seconds < 90:
        return "%ds" % round(seconds)
    if seconds < 5400:
        return "%.1f min" % (seconds / 60)
    return "%.1f h" % (seconds / 3600)


def _resolve_streamed(request: Request, selection, cfg: config.Config,
                      timeout: Optional[float], workers: int,
                      specs: Optional[List[MetricSpec]], narrowing,
                      nodename: Optional[str], gpu_ids, host_specs) -> Optional[Resolved]:
    """A window selection, streamed a time slice at a time and never listed first.

    One sacct call per slice replaces a listing call plus one per 200 ids -- see
    :func:`jobscope.slurm.fetch_window` for why the listing was not buying anything.

    The cost is that nothing here knows the job count in advance, and two things used
    to be taken from it. The header description is now computed from the selection
    (:func:`describe_window`), which never needed the ids. ``total`` cannot be, so the
    first slice is drawn eagerly and answers the only question anything asks of it --
    whether this is one job or many.
    """
    desc = describe_window(selection)
    chunks = fetch_window(selection, timeout)
    first = next(chunks, None)
    if first is None:
        print("No matching jobs for %s (%s)."
              % ("all users" if request.all_users else "user '%s'" % selection.user, desc),
              file=sys.stderr)
        return None

    # Before the metrics are collected, which is what makes it worth printing: the
    # first slice is in hand and the rest of the run has not been paid for yet.
    _note_cost(first[0], first[1], selection, specs, cfg)

    # Only ever read as `total > 1` (see report.DetailRenderer), so the first slice
    # settles it whenever it holds two jobs -- which any selection wide enough for this
    # to matter does. A single-job first slice is reported as one job; that is wrong
    # only if a later slice holds another, and only for a detail view over a window
    # thin enough to split one job per day.
    context = context_pairs(rows.build_context(selection, desc, {}),
                            specs, host_specs, average=request.average) + narrowing
    return Resolved(context,
                    _enrich(itertools.chain([first], chunks), cfg, timeout, workers, specs,
                            no_jobstats=request.no_jobstats,
                            nodename=nodename, gpu_ids=gpu_ids,
                            host_specs=host_specs, average=request.average),
                    folded=True,        # a window selection holds only finished jobs
                    total=len(first[0]))


def _enrich(chunks, cfg: config.Config, timeout: Optional[float], workers: int,
            specs: Optional[List[MetricSpec]], no_jobstats: bool = False,
            nodename: Optional[str] = None, gpu_ids=(),
            host_specs=None, average: bool = False) -> Iterator[Chunk]:
    """Attach DCGM metrics and fill running jobs' summaries, chunk by chunk.

    The client is built lazily and at most once: a selection with no GPU jobs, or a
    --cpu report over finished ones, never contacts Prometheus.

    ``nodename``/``gpu_ids`` narrow both halves to the same cards -- the jobstats summary the
    summary's CPU%/MEM%/GPU%/GMEM% come from, and the DCGM queries. Both, or the row
    would mix one node's GPU% with every node's SM_ACT%.
    """
    client: Optional[PrometheusClient] = None
    for chunk_ids, records in chunks:
        # Before the jobstats summary is read and before the queries: narrowing the record is what
        # makes every downstream figure -- row, per-metric table, bars, verdict --
        # describe the subset without any of them knowing a filter was applied.
        if nodename or gpu_ids:
            _narrow_records(records, chunk_ids, nodename, gpu_ids)
        dcgm_data: DcgmData = {}
        if specs and any(j in records and records[j].gpus for j in chunk_ids):
            if client is None:
                client = client_from_config(cfg, timeout)
            dcgm_data = compute_dcgm(records, chunk_ids, specs, client, timeout, workers,
                                     nodename=nodename, gpu_ids=gpu_ids,
                                     average=average)
        client = _fill_running(records, chunk_ids, cfg, timeout, workers, client,
                               force=no_jobstats, average=average)
        # Last, and only for what is still missing: Slurm's own accounting, where the
        # site has named it as a host source. After jobstats and Prometheus because it
        # is the coarsest of the three -- job totals rather than per-node series -- so
        # it should never displace a measurement that arrived.
        host = cpu.catalog()
        if "slurm" in host.preference:
            apply_slurm_host(records, chunk_ids, timeout,
                             override=host.resolved.source_of("CPU%") == "slurm")
        _note_host_gap(records, chunk_ids, host_specs)
        yield chunk_ids, records, dcgm_data


def _narrow_records(records, jobids, nodename: Optional[str], gpu_ids) -> None:
    """Restrict each record's stats to ``nodename``/``gpu_ids``, in place.

    Raises when nothing in the selection matched, naming what was there -- the same
    rule the time-series filters follow, and for the same reason: an empty or
    silently-whole-job summary is indistinguishable from a correct one.

    A record whose stats are empty afterwards keeps them empty; it renders as "-"
    rather than as zeros, which is what a job that never touched the named node is.
    """
    # Two separate questions, because narrow_stats keeps a node's CPU/memory entry
    # even when --gpuid removed all of its cards -- --gpuid says nothing about cores.
    # So a surviving `stats` does not mean a surviving *card*, and checking only the
    # former let `--gpuid 9` report the whole job's CPU% under a heading claiming
    # otherwise.
    seen_nodes, seen_gpus, hit_gpus = set(), set(), set()
    matched_node = False
    for jid in jobids:
        record = records.get(jid)
        if record is None:
            continue
        seen_nodes |= jobstats.nodes_in(record.stats)
        seen_gpus |= jobstats.gpu_ids_in(record.stats)
        stats = jobstats.narrow_stats(record.stats, nodename, gpu_ids)
        if stats:
            matched_node = True
        hit_gpus |= jobstats.gpu_ids_in(stats)
        records[jid] = replace(
            record, stats=stats,
            nodes=str(len(jobstats.nodes_in(stats))) if stats else "0",
            # #GPU drives the GPU% denominator's "was this job allocated cards" and
            # the GPU-hours weighting, so it has to shrink with the cards -- counted
            # per card rather than per distinct minor, since nodes number from 0.
            gpus=jobstats.gpu_count(stats) if record.gpus else record.gpus)
    if nodename and not matched_node:
        raise JobscopeError("no data for node %r in this selection; it ran on: %s"
                            % (nodename, ", ".join(sorted(seen_nodes)) or "(none)"))
    # Every id that matched nothing, not just the case where none did: in --gpuid 0,9
    # it is the 9 you need told about, and a shorter summary looks like a correct one.
    # Skipped entirely when the selection had no cards -- a CPU-only job narrowed by
    # --gpuid has nothing to match and nothing to complain about either.
    missing = [g for g in gpu_ids if str(g) not in hit_gpus]
    if gpu_ids and seen_gpus and missing:
        raise JobscopeError("no data for GPU %s in this selection; it used: %s"
                            % (", ".join(repr(g) for g in missing),
                               ", ".join(sorted(seen_gpus))))


def _fill_running(records, jobids, cfg, timeout, workers, client, force=False,
                  average=False):
    """Rebuild the utilization summary for the jobs in this chunk that need one.

    A running job has no stored summary, so CPU%/MEM%/GPU%/GMEM% would all be empty.
    Every input is in Prometheus, so fill them from there. With ``force``
    (``--no-jobstats``) finished jobs are rebuilt too, ignoring what Slurm stored.

    An install with no endpoint configured stays fully offline: the fill is skipped
    with a note rather than an error -- except under ``force``, where there is no
    stored summary to fall back on and going quiet would print a table of dashes with
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
                    "--no-jobstats reads every metric from Prometheus, and no endpoint is\n"
                    "configured. Drop --no-jobstats to use the summary jobstats stored, or see\n"
                    "'jobscope probe' for how to configure one.")
            note_offline_gap(records, jobids)
            return None
    fill_running(records, jobids, client, timeout, workers, force, average)
    return client


def _note_host_gap(records, jobids, host_specs) -> None:
    """Explain blank CPU%/MEM% once the fills have had their turn.

    After both of them, because either may supply the columns: the cgroup fill above,
    or Slurm's accounting below it. Only worth saying when the view actually prints
    those columns -- --gpu has no CPU% to be missing."""
    if host_specs and any(s.column in ("CPU%", "MEM%") for s in host_specs):
        note_missing_host_series(records, jobids)


def _note_gpu_series_gap(gpus, samples, specs) -> None:
    """Say when the chosen GPU source has no series for some of the selection's cards.

    Those jobs are absent from the series entirely, and therefore from --stats and
    --eff's counts -- so without this the same selection reports a different number of
    jobs under --gpu-source nvml than under dcgm, with nothing to say why. Measured
    here: dcgm-exporter down on one host of a partition dropped 2 of 3 jobs silently.

    The counterpart of :func:`note_missing_host_series` for the GPU axis, and it names
    the other source because that is the actionable part: the cards are discovered
    through the nvml join either way, so a card missing from dcgm is usually present in
    nvml rather than genuinely idle.
    """
    blind = sorted({g.host for uuid, g in gpus.items() if not samples.get(uuid)})
    if not blind:
        return
    jobs_hit = {g.jobid for uuid, g in gpus.items() if not samples.get(uuid)}
    leading = dcgm.catalog().resolved.leading_exporter()
    other = "dcgm" if leading == "nvml" else "nvml"
    print("note: %d of %d job(s) have no GPU metrics -- no %s series covers their cards\n"
          "      on %s. They are absent from the series, and so from --stats and --eff\n"
          "      counts. Try --gpu-source %s, which reads a different exporter; 'jobscope\n"
          "      probe' reports which hosts each one covers."
          % (len(jobs_hit), len({g.jobid for g in gpus.values()}), leading,
             ", ".join(blind), other), file=sys.stderr)


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
        _note_gpu_series_gap(gpus, samples, specs)
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
