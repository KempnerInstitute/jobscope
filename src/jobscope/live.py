"""The live view: GPU metrics for the jobs running right now.

The historical views start from ``sacct`` and reduce each metric over a job's
finished ``[start, end]`` window. This one starts from ``squeue`` and, by default,
reports the newest scrape -- answering "what is happening on the GPUs at this
moment" rather than "how did this job do overall". ``--avg`` folds over each job's
runtime instead, which reproduces jobstats' numbers.

Three things make this path genuinely different from :mod:`jobscope.dcgm`, rather
than a variation on it:

* **The raw job ID.** ``squeue``'s ``%i`` is the display form (``12345_6``) but
  Prometheus keys on ``%A``, the raw per-element ID, which differs for array
  elements. Both are read; ``%A`` joins, ``%i`` displays.
* **GPU identity is the UUID.** ``minor_number`` is not unique: on a MIG node every
  instance inherits its parent card's minor number, and two nodes both have a
  minor 0. Rows are therefore keyed by UUID and labelled ``GPU 0`` / ``MIG 0.1``.
* **Ownership is read at the current instant.** A single GPU can host a dozen jobs
  in a day, so the job->GPU mapping comes from one unwindowed
  ``nvidia_gpu_jobId`` query filtered client-side, not from a windowed per-job
  query -- a window would hand the same GPU to every job that touched it.
"""

import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from .dcgm import (
    ALL_SPECS,
    DEFAULT_SPECS,
    MODEL_KEY,
    MetricSpec,
    applicable_derived,
    columns_for,
    window_query,
)
from . import config
from .errors import JobscopeError
from .live_blob import host_stats_many, stats_dict
from .prometheus import PrometheusClient
from .sacct import JobRecord, run_capture

# One job's squeue fields, plus the derived start_epoch / elapsed_seconds.
LiveJob = Dict[str, Any]
# {raw_jobid: {gpu_uuid: {metric key: value}}}. Keyed by UUID rather than
# minor_number, which is not unique per schedulable GPU -- see gpu_labels.
LiveMetrics = Dict[int, Dict[str, Dict[str, Optional[float]]]]

# %A first, then %i: %A is the raw per-element job ID that nvidia_gpu_jobId
# reports, %i the display form (they differ for array elements). Pipe-delimited
# because the start time contains colons.
#
# Note %b (tres-per-node, e.g. "gres/gpu:1") for the GPU request -- NOT %G, which
# is the numeric group ID. GPU rows come from Prometheus regardless; this field is
# only the job's allocation as Slurm records it.
SQUEUE_FORMAT = "%A|%i|%u|%N|%g|%j|%b|%C|%S"

class Gpu(NamedTuple):
    """Identity of one schedulable GPU: a whole card, or a single MIG instance."""

    uuid: str
    jobid: int
    host: str
    minor: int
    label: str          # display form, e.g. "GPU 2" or "MIG 2.0"
    # The exporter's model string, e.g. "NVIDIA H100 80GB HBM3". Carried because the
    # POWER_W floor is per architecture: idle draw runs 27 W to 165 W across a fleet.
    # Defaulted, so a series without the label degrades to the global floor.
    model: str = ""

    @property
    def csv_id(self) -> str:
        """GPU value for CSV output: "2", or "2.0" for a MIG slice.

        Keeps the bare-minor-number convention for whole cards while staying
        unique per slice, since ``jobscope plot`` groups series by (NODE, GPU).
        """
        return self.label.split(" ", 1)[1]


@dataclass
class LiveSelection:
    """Which running jobs to report on."""

    jobids: List[str] = field(default_factory=list)
    partition: Optional[str] = None
    user: Optional[str] = None      # None = every user
    min_elapsed: int = 600

    def describe(self) -> str:
        """Human-readable summary for the context block.

        Omits the user and partition, which the context block prints on their own
        lines. :meth:`describe_filters` is the form to use where those lines are
        not shown.
        """
        if self.jobids:
            return "%d job ID(s)" % len(self.jobids)
        parts = ["running"]
        if self.min_elapsed > 0:
            parts.append("longer than %s" % format_duration(self.min_elapsed))
        return ", ".join(parts)

    def describe_filters(self) -> str:
        """:meth:`describe` plus the user and partition, for the empty-result message.

        An empty result is exactly when every filter has to be named: "no running
        jobs (running, longer than 10m)" reads as an idle partition, when usually
        it means the default user filter excluded everyone who is actually on it.
        """
        if self.jobids:
            return self.describe()
        parts = ["user %s" % self.user if self.user else "all users"]
        if self.partition:
            parts.append("partition %s" % self.partition)
        return ", ".join(parts + [self.describe()])

    def widening_hints(self) -> List[str]:
        """Flags that would loosen this selection, most likely first."""
        hints = []
        if self.user:
            hints.append("add -a to include every user")
        if self.min_elapsed > 0:
            hints.append("set --min-elapsed 0s to include jobs that just started")
        if self.partition:
            hints.append("drop -p to search every partition")
        return hints


# The live catalogs. Deliberately the same specs the dcgm view uses, so a running
# job and a finished one are described by identical columns:
#
#   GPU%  SM_ACT%  OCC%  TENSOR%  DRAM%  POWER_W  GMEM_GB  GMEM%
#
# The summary and detail views omit GPU% and the GMEM columns from their DCGM set
# because they render those from the blob instead; a running job has no blob, so
# here Prometheus is the only source and nothing is dropped.
DEFAULT_LIVE_SPECS: List[MetricSpec] = DEFAULT_SPECS
# --all appends the extended catalog, minus delta-reduced counters (ENERGY_kWh):
# a delta needs two points, so it is meaningless in an instant snapshot.
EXTENDED_LIVE_SPECS: List[MetricSpec] = DEFAULT_LIVE_SPECS + [
    s for s in ALL_SPECS if s.group == "all" and s.reducer != "delta"]


def specs_for(view: Optional[str]) -> List[MetricSpec]:
    """The metric catalog for a live view: the default set, or ``all``."""
    return EXTENDED_LIVE_SPECS if view == "all" else DEFAULT_LIVE_SPECS


# Moved to config, which needs it for the duration-valued [defaults] keys and
# cannot import this module. Re-exported because this is where callers look for it,
# next to format_duration.
parse_duration = config.parse_duration


def format_duration(seconds: int) -> str:
    """Inverse of :func:`parse_duration`, for echoing the selection back."""
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size and seconds % size == 0:
            return "%d%s" % (seconds // size, unit)
    return "%ds" % seconds


def parse_start_time(start_time: str) -> Tuple[Optional[int], Optional[int]]:
    """``(start epoch, elapsed seconds)`` from a squeue ``%S`` value.

    ``(None, None)`` when it is not a timestamp -- squeue prints ``N/A`` or
    ``Unknown`` while a job has no start time yet. ``%S`` is local time.
    """
    try:
        start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError):
        return None, None
    epoch = int(time.mktime(start_dt.timetuple()))
    return epoch, int(time.time()) - epoch


def parse_squeue(stdout: str) -> Dict[int, LiveJob]:
    """Parse :data:`SQUEUE_FORMAT` output into ``{raw_jobid: job}``.

    Keyed by the raw ID so the Prometheus join works for array elements too; the
    displayed ID keeps squeue's own notation.
    """
    jobs: Dict[int, LiveJob] = {}
    for line in stdout.strip().split("\n"):
        if not line or line.startswith("JOBID"):
            continue
        parts = line.split("|")
        if len(parts) < 9:
            continue
        raw_id, disp_id, user, nodelist, group, name, gres, cpus, start_time = parts[:9]
        try:
            raw_jobid = int(raw_id)
        except ValueError:
            continue
        start_epoch, elapsed = parse_start_time(start_time)
        jobs[raw_jobid] = {
            "jobid": disp_id,       # display form; array notation preserved
            "user": user,
            "node": nodelist,       # kept whole: compressed ranges must not be split
            "group": group,
            "name": name,
            "gres": gres,       # as Slurm records it, e.g. "gres/gpu:1"
            "cpus": cpus,
            "start_time": start_time,
            # Runtime drives the --min-elapsed filter, the --avg window and the
            # --ts range, so derive it once here.
            "start_epoch": start_epoch,
            "elapsed_seconds": elapsed,
        }
    return jobs


def job_sort_key(job: LiveJob) -> Tuple[int, int]:
    """Sort key from the displayed job ID: ``(base id, array index)``.

    Keeps elements of one array together and in index order. Raw IDs are assigned
    in submission order and would interleave unrelated jobs.
    """
    base, _, index = str(job["jobid"]).partition("_")
    try:
        base_n = int(base)
    except ValueError:
        return (sys.maxsize, 0)
    try:
        # No index -> a plain job, which sorts ahead of elements of the same base.
        index_n = int(index) if index else -1
    except ValueError:
        index_n = -1
    return (base_n, index_n)


def fetch_jobs(selection: LiveSelection, timeout: Optional[float]) -> Dict[int, LiveJob]:
    """Running jobs matching ``selection``, keyed by raw job ID.

    Explicit job IDs bypass the partition/user filters and the runtime floor, as
    they do in the historical views, and a miss is an error rather than an empty
    table -- with a pointer at the historical view, since the usual cause is that
    the job has already finished.
    """
    cmd = ["squeue", "-h", "-o", SQUEUE_FORMAT]
    if selection.jobids:
        cmd += ["-j", ",".join(selection.jobids)]
        # squeue exits non-zero on an unknown id, so ask it not to raise.
        out = run_capture(cmd, timeout, "squeue query", soft=True)
        jobs = parse_squeue(out) if out else {}
        if not jobs:
            raise JobscopeError(
                "no running job matches %s.\nIf it has already finished, the historical "
                "views cover it: jobscope -j %s" % (", ".join(selection.jobids),
                                                    selection.jobids[0]))
        return jobs

    cmd += ["-t", "RUNNING"]
    if selection.partition:
        cmd += ["-p", selection.partition]
    if selection.user:
        cmd += ["-u", selection.user]
    jobs = parse_squeue(run_capture(cmd, timeout, "squeue query") or "")
    return filter_by_elapsed(jobs, selection.min_elapsed)


def filter_by_elapsed(jobs: Dict[int, LiveJob], min_seconds: int) -> Dict[int, LiveJob]:
    """Drop jobs that have not been running longer than ``min_seconds``."""
    kept = {}
    for raw_jobid, job in jobs.items():
        elapsed = job["elapsed_seconds"]
        if elapsed is None:
            print("note: no usable start time for job %s" % job["jobid"], file=sys.stderr)
            continue
        if elapsed > min_seconds:
            kept[raw_jobid] = job
    return kept


def gpu_labels(found: List[Tuple[str, int, str, int]]) -> Dict[str, str]:
    """Display label per GPU UUID, from ``(uuid, jobid, host, minor)`` tuples.

    ``minor_number`` is not unique per schedulable GPU: on a MIG node every
    instance inherits its parent card's minor number and ordinal, so a 3g.20gb
    pair both report minor 0. Only the UUID is unique -- MIG instances carry a
    ``MIG-`` UUID where whole cards carry ``GPU-``. NVML exposes no instance
    index, so siblings sharing a ``(job, host, minor)`` are enumerated by sorted
    UUID, which is stable for as long as the partitioning is.
    """
    siblings = defaultdict(list)
    for uuid, jobid, host, minor in found:
        if uuid.startswith("MIG-"):
            siblings[(jobid, host, minor)].append(uuid)

    labels = {}
    for uuid, jobid, host, minor in found:
        if not uuid.startswith("MIG-"):
            labels[uuid] = "GPU %d" % minor
            continue
        peers = sorted(siblings[(jobid, host, minor)])
        # Say MIG explicitly: a slice's memory total is the slice, not the card.
        labels[uuid] = ("MIG %d.%d" % (minor, peers.index(uuid))
                        if len(peers) > 1 else "MIG %d" % minor)
    return labels


def discover_gpus(client: PrometheusClient, jobs: Dict[int, LiveJob],
                  timeout: Optional[float]) -> Dict[str, Gpu]:
    """Map GPU UUID -> :class:`Gpu` for the given jobs.

    The job ID is ``nvidia_gpu_jobId``'s *value*, not a label, so there is nothing
    to filter on server-side: one instant query returns every GPU's current job
    and we keep the ones asked about. Deliberately unwindowed -- see the module
    docstring.
    """
    try:
        series = client.query("nvidia_gpu_jobId", int(time.time()), timeout)
    except Exception as exc:
        raise JobscopeError("could not read GPU ownership from Prometheus: %s" % exc)

    found = []
    for entry in series:
        try:
            # Exposed in scientific notation (3.4853925e+07), hence float first.
            jobid = int(float(entry["value"][1]))
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        if jobid not in jobs:
            continue
        labels = entry["metric"]
        uuid = labels.get("uuid")   # the nvidia_* exporter uses lowercase "uuid"
        minor = labels.get("minor_number")
        if not uuid or minor is None:
            continue
        try:
            found.append((uuid, jobid, labels.get("host", "").split(":")[0], int(minor),
                          labels.get("name", "")))
        except ValueError:
            continue

    display = gpu_labels([f[:4] for f in found])
    return {uuid: Gpu(uuid, jobid, host, minor, display[uuid], model)
            for uuid, jobid, host, minor, model in found}


def _uuid_regex(uuids) -> str:
    """An anchored alternation over UUIDs (hex and hyphens, so RE2-safe as-is)."""
    return "^(" + "|".join(uuids) + ")$"


def _store(results: LiveMetrics, gpus: Dict[str, Gpu], spec: MetricSpec,
           series: List[dict], jobid: Optional[int] = None) -> None:
    """File one query's series into ``results`` under (job, uuid, metric key)."""
    for entry in series:
        uuid = entry["metric"].get(spec.uuid_label)
        gpu = gpus.get(uuid)
        if gpu is None:
            continue
        # A GPU reassigned mid-window can surface under another job's query.
        if jobid is not None and gpu.jobid != jobid:
            continue
        try:
            results[gpu.jobid][uuid][spec.key] = float(entry["value"][1]) * spec.scale
        except (KeyError, IndexError, TypeError, ValueError):
            results[gpu.jobid][uuid][spec.key] = None


def collect_instant(client: PrometheusClient, gpus: Dict[str, Gpu],
                    specs: List[MetricSpec], timeout: Optional[float]) -> LiveMetrics:
    """The newest scrape of every metric, one query per metric across all GPUs."""
    results: LiveMetrics = defaultdict(lambda: defaultdict(dict))
    regex = _uuid_regex(gpus)
    for spec in specs:
        query = '%s{%s=~"%s"}' % (spec.metric, spec.uuid_label, regex)
        try:
            series = client.query(query, int(time.time()), timeout)
        except Exception:
            continue
        _store(results, gpus, spec, series)
    add_derived(results, specs)
    return results


def collect_averaged(client: PrometheusClient, jobs: Dict[int, LiveJob],
                     gpus: Dict[str, Gpu], specs: List[MetricSpec],
                     timeout: Optional[float], workers: int) -> LiveMetrics:
    """Each metric folded over each job's own runtime, as jobstats does.

    Utilization is averaged and memory peaked, per each spec's ``reducer``. The
    window differs per job and PromQL cannot vary a window per series, so this is
    one query per (job, metric) -- run concurrently, since the count grows with
    the selection.
    """
    results: LiveMetrics = defaultdict(lambda: defaultdict(dict))
    by_job = defaultdict(list)
    for uuid, gpu in gpus.items():
        by_job[gpu.jobid].append(uuid)

    tasks = []
    for jobid, uuids in by_job.items():
        window = jobs[jobid].get("elapsed_seconds")
        if not window or window <= 0:
            print("note: skipping average for job %s: unknown runtime" % jobs[jobid]["jobid"],
                  file=sys.stderr)
            continue
        for spec in specs:
            tasks.append((jobid, uuids, window, spec))
    if not tasks:
        return results

    def run(task):
        jobid, uuids, window, spec = task
        query = window_query(spec, uuids, window, clip=clip_to_job(spec, jobid))
        try:
            return task, client.query(query, int(time.time()), timeout)
        except Exception:
            return task, []

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        for task, series in pool.map(run, tasks):
            jobid, _uuids, _window, spec = task
            _store(results, gpus, spec, series, jobid=jobid)

    add_derived(results, specs)
    return results


def clip_to_job(spec: MetricSpec, raw_jobid: int) -> Optional[str]:
    """The ``nvidia_gpu_jobId`` series to intersect ``spec``'s window with, if any.

    For ``nvidia_*`` metrics this restricts the window to the samples where the job
    actually owned the GPU, making the window length a harmless upper bound --
    exactly as jobstats does. ``DCGM_*`` metrics come from the other exporter, whose
    label sets differ (``UUID``/``Hostname``/``gpu``), so PromQL's ``and`` can never
    match there and the window alone has to bound them.
    """
    return "nvidia_gpu_jobId == %d" % raw_jobid if spec.uuid_label == "uuid" else None


def add_derived(results: LiveMetrics, specs: List[MetricSpec]) -> None:
    """Fill in derived columns whose input metrics were all queried.

    Values here are keyed by metric key throughout, so the shared ``fn`` can read
    them directly -- unlike the dcgm view, which stores by header.
    """
    derived = applicable_derived(specs)
    if not derived:
        return
    for per_gpu in results.values():
        for values in per_gpu.values():
            for column in derived:
                values[column.key] = column.fn(values)


def timeseries_step(elapsed: int, sampling_period: int,
                    requested: Optional[int] = None) -> int:
    """Range-query step in seconds.

    Never finer than the scrape interval -- there is no more data -- and coarse
    enough to stay under Prometheus' points-per-series cap on long jobs, the same
    bound :func:`jobscope.report.dcgm_timeseries` uses.
    """
    if requested:
        return max(1, requested)
    return max(sampling_period, elapsed // 10000 + 1)


def range_window(start: int, end: int, window: Optional[int], sampling_period: int,
                 requested: Optional[int] = None) -> Tuple[int, int]:
    """``(start, step)`` for a range query over the last ``window`` seconds.

    Narrowing the *query* rather than filtering rows afterwards is the point: an hour
    of a day-long job is a twenty-fourth of the samples to fetch, and the step is
    measured over the span actually queried, so the window keeps its resolution.

    The start is then snapped back onto the run's own sample grid, which matters more
    than it looks. Prometheus anchors a range query's points at ``start``, so moving
    the start moves *every* timestamp: an unaligned window returns the same scrapes
    under different labels, and because a running job's ``end`` is "now", its grid
    drifts with the clock between invocations. Aligned, ``--ts 1h`` returns exactly
    the rows a full ``--ts`` would have, so fetching a window and slicing a whole
    series agree.
    """
    begin = max(start, end - window) if window else start
    step = timeseries_step(end - begin, sampling_period, requested)
    if window:
        # Floor, so the window is never shorter than the one asked for.
        begin = start + ((begin - start) // step) * step
    return begin, step


def collect_timeseries(client: PrometheusClient, jobs: Dict[int, LiveJob],
                       gpus: Dict[str, Gpu], specs: List[MetricSpec],
                       timeout: Optional[float], workers: int,
                       step: Optional[int] = None,
                       window: Optional[int] = None) -> Dict[str, Dict[int, dict]]:
    """Every sample of every metric over each job's runtime.

    Returns ``{uuid: {epoch: {metric key: value}}}``. One range query per
    (job, metric), on the same pool as :func:`collect_averaged`.

    ``window`` shortens that to the most recent N seconds of the run.
    """
    by_job = defaultdict(list)
    for gpu in gpus.values():
        by_job[gpu.jobid].append(gpu)

    tasks = []
    for jobid, job_gpus in by_job.items():
        job = jobs[jobid]
        start, elapsed = job.get("start_epoch"), job.get("elapsed_seconds")
        if not start or not elapsed or elapsed <= 0:
            print("note: skipping timeseries for job %s: unknown runtime" % job["jobid"],
                  file=sys.stderr)
            continue
        regex = _uuid_regex(g.uuid for g in job_gpus)
        end = start + elapsed
        start, span = range_window(start, end, window, client.sampling_period, step)
        for spec in specs:
            tasks.append((jobid, regex, start, end, span, spec))

    samples: Dict[str, Dict[int, dict]] = defaultdict(lambda: defaultdict(dict))
    if not tasks:
        return samples

    def run(task):
        _jobid, regex, start, end, span, spec = task
        query = '%s{%s=~"%s"}' % (spec.metric, spec.uuid_label, regex)
        try:
            return task, client.query_range(query, start, end, span, timeout)
        except Exception:
            return task, []

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        for task, results in pool.map(run, tasks):
            jobid, _regex, _start, _end, _span, spec = task
            for series in results:
                uuid = series["metric"].get(spec.uuid_label)
                gpu = gpus.get(uuid)
                if gpu is None or gpu.jobid != jobid:
                    continue
                for stamp, raw in series.get("values", []):
                    try:
                        epoch = int(float(stamp))
                        samples[uuid][epoch][spec.key] = float(raw) * spec.scale
                    except (TypeError, ValueError):
                        continue
    return samples


# The live and dcgm views lay out the same columns, so they share one builder.
build_columns = columns_for


def aggregate_by_job(metrics: LiveMetrics, specs: List[MetricSpec],
                     ) -> Dict[int, Dict[str, Optional[float]]]:
    """Reduce per-GPU values to one row per job, keyed by column header.

    Uses each spec's own cross-GPU aggregation -- mean for utilization, max for
    peaks -- so the job-level figure means the same thing it does in the historical
    views. Derived columns are recomputed from the aggregated inputs rather than
    averaged, since a ratio of means is not the mean of ratios.
    """
    per_job: Dict[int, Dict[str, Optional[float]]] = {}
    derived = applicable_derived(specs)
    for raw_jobid, by_uuid in metrics.items():
        keyed: Dict[str, Optional[float]] = {}
        for spec in specs:
            values = [v[spec.key] for v in by_uuid.values()
                      if v.get(spec.key) is not None]
            if not values:
                continue
            keyed[spec.key] = (sum(values) if spec.agg == "sum"
                               else max(values) if spec.agg == "max"
                               else sum(values) / len(values))
        for column in derived:
            computed = column.fn(keyed)
            if computed is not None:
                keyed[column.key] = computed
        by_header = {spec.header: keyed[spec.key] for spec in specs if spec.key in keyed}
        by_header.update({d.header: keyed[d.key] for d in derived if d.key in keyed})
        per_job[raw_jobid] = by_header
    return per_job


def per_gpu_by_node_minor(metrics: LiveMetrics, gpus: Dict[str, Gpu],
                          specs: List[MetricSpec],
                          ) -> Dict[int, Dict[Tuple[str, str], Dict[str, Optional[float]]]]:
    """Re-key per-GPU values to ``(node, minor)`` and headers, for ``--per-gpu``.

    The live collectors key by UUID, which is the only unique GPU identity; the
    detail renderer keys by ``(node, minor)``, which is what the blob uses. MIG
    siblings share a minor, so they collapse here exactly as they do in the blob --
    the honest per-instance view is ``--ts``, which keys by UUID throughout.
    """
    derived = applicable_derived(specs)
    out: Dict[int, Dict[Tuple[str, str], Dict[str, Optional[float]]]] = defaultdict(dict)
    for uuid, gpu in gpus.items():
        keyed = metrics.get(gpu.jobid, {}).get(uuid)
        if not keyed:
            continue
        by_header = {spec.header: keyed[spec.key] for spec in specs if spec.key in keyed}
        by_header.update({d.header: keyed[d.key] for d in derived if d.key in keyed})
        by_header[MODEL_KEY] = gpu.model     # for the per-architecture POWER_W floor
        out[gpu.jobid][(gpu.host, str(gpu.minor))] = by_header
    return out


def live_records(jobs: Dict[int, LiveJob], gpus: Dict[str, Gpu],
                 metrics: LiveMetrics, specs: List[MetricSpec],
                 client: PrometheusClient,
                 timeout: Optional[float] = None) -> Dict[str, JobRecord]:
    """Turn squeue jobs into :class:`~jobscope.sacct.JobRecord`s, keyed by display ID.

    This is what lets the live view render through the same code as the historical
    ones: once a running job looks like a record with a utilization blob, the summary
    renderer cannot tell the difference, and the two views cannot drift apart.

    The blob's CPU and host-memory fields come from ``cgroup_*``; its GPU maps are
    built from the values already collected per GPU, so they honour the
    instant-versus-``--avg`` choice instead of silently re-querying a window.
    """
    gpus_per_job: Dict[int, List[Gpu]] = defaultdict(list)
    for gpu in gpus.values():
        gpus_per_job[gpu.jobid].append(gpu)

    # One batched set of cgroup queries for every job, rather than four per job:
    # a partition-wide selection is hundreds of jobs, and per-job round trips made
    # that unusable. The GPU half needs no queries at all -- it is assembled from
    # values already collected, so it honours the instant-versus---avg choice.
    at = int(time.time())
    elapsed_by_job = {raw: job["elapsed_seconds"] for raw, job in jobs.items()
                      if (job.get("elapsed_seconds") or 0) > 0}
    hosts = host_stats_many(elapsed_by_job, at, client, timeout)

    records: Dict[str, JobRecord] = {}
    for raw_jobid, job in jobs.items():
        elapsed = job.get("elapsed_seconds")
        start = job.get("start_epoch")
        stats = stats_dict(elapsed or 0, hosts.get(raw_jobid, {}),
                           _gpu_node_map(gpus_per_job[raw_jobid],
                                         metrics.get(raw_jobid, {})))
        records[job["jobid"]] = JobRecord(
            jobid=job["jobid"],
            state="RUNNING",
            name=job.get("name", "?"),
            runtime=format_elapsed(elapsed),
            # NODE is a count, as in the historical views; squeue gives a nodelist,
            # so count the hosts the job's GPUs are on, falling back to 1.
            nodes=str(len({g.host for g in gpus_per_job[raw_jobid]}) or 1),
            gpus=len(gpus_per_job[raw_jobid]),
            stats=stats,
            start=start,
            end=(start + elapsed) if start and elapsed else int(time.time()),
            duration=elapsed,
            jobid_raw=str(raw_jobid),
            cluster="",
            user=job.get("user", "?"),
        )
    return records


def _gpu_node_map(job_gpus: List[Gpu], by_uuid: Dict[str, dict]) -> Dict[str, dict]:
    """The blob's per-node GPU maps, from values already collected per GPU."""
    nodes: Dict[str, dict] = {}
    for gpu in job_gpus:
        values = by_uuid.get(gpu.uuid, {})
        for blob_field, key in (("gpu_utilization", "duty"),
                                ("gpu_used_memory", "mem"),
                                ("gpu_total_memory", "memtot")):
            value = values.get(key)
            if value is None:
                continue
            # Scaled back to bytes: the blob stores raw byte counts, and blob_metrics
            # divides used by total, so the two must share a unit.
            if key in ("mem", "memtot"):
                value = value * 1024 ** 3
            nodes.setdefault(gpu.host, {}).setdefault(blob_field, {})[str(gpu.minor)] = value
    return nodes


def format_elapsed(seconds: Optional[int]) -> str:
    """``D-HH:MM:SS`` / ``HH:MM:SS``, matching sacct's Elapsed formatting."""
    if not seconds or seconds < 0:
        return "-"
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    stamp = "%02d:%02d:%02d" % (hours, minutes, secs)
    return "%d-%s" % (days, stamp) if days else stamp
