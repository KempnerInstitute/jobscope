#!/usr/bin/env python3
"""jobscope_live - Real-time GPU metrics for running jobs in a partition.

Quick view of GPU utilization and DCGM metrics for jobs currently running on a
given partition, using squeue (instant; no history) + Prometheus (at the current
moment). No historical analysis or job reconstruction -- just the live snapshot.

By default every number is an instantaneous reading, which will NOT agree with
jobstats on a bursty job: jobstats folds over the whole runtime. Pass --avg to do
the same -- averaging utilization and peaking memory, as jobstats does -- which
reproduces its numbers. MEM_GB/MEM% correspond to jobstats' "GPU memory usage per
node - maximum used/total".

Usage:
  ./jobscope_live.py                # all running jobs >1h (default)
  ./jobscope_live.py -j 34622920            # specific job by ID
  ./jobscope_live.py -j 34843528_6          # specific array job element
  ./jobscope_live.py -p kempner            # jobs in partition
  ./jobscope_live.py -p kempner -u alice
  ./jobscope_live.py -p kempner --gpu      # GPU columns only
  ./jobscope_live.py -p kempner --all      # all DCGM metrics
  ./jobscope_live.py -p kempner --avg      # averaged over runtime (= jobstats)
  ./jobscope_live.py -p kempner --min-runtime 5m  # jobs running >5 minutes
  ./jobscope_live.py -p kempner --min-runtime 0s  # all running jobs (no filter)

  # timeseries CSV over each job's runtime, in the schema jobscope plot reads:
  ./jobscope_live.py -j 34843528_6 --ts > ts.csv && jobscope plot ts.csv
  ./jobscope_live.py -p kempner --ts --step 300   # coarser sampling

Author: Bala Desinghu, Senior AI/HPC Research Computing Engineer, Kempner Institute, Harvard
"""

import sys
import csv
import json
import argparse
import subprocess
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

# One job's squeue fields (all str) plus elapsed_seconds (Optional[int]).
Job = Dict[str, Any]
# {raw_jobid: {gpu_uuid: {metric_key: value}}}. Keyed by UUID rather than
# minor_number because minor_number is not unique -- see _gpu_labels.
GpuMetrics = Dict[int, Dict[str, Dict[str, Optional[float]]]]

# Concurrency for the per-job averaged queries; the averaged path needs one
# query per job per metric, which is only tolerable in parallel.
_MAX_QUERY_WORKERS = 8

# Python 3.6 compatibility
if sys.version_info >= (3, 7):
    _PIPE_KWARGS = {"capture_output": True, "text": True}
else:
    _PIPE_KWARGS = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "universal_newlines": True}

sys.path.insert(0, "/usr/local/bin")
sys.path.insert(0, "/usr/bin")

try:
    from config import PROM_SERVER
except ImportError:
    PROM_SERVER = "http://localhost:9090"

try:
    from config import SAMPLING_PERIOD
except ImportError:
    SAMPLING_PERIOD = 60  # exporter scrape interval; the finest useful --ts step

import requests


class Gpu(NamedTuple):
    """Identity of one schedulable GPU: a whole card, or a single MIG instance."""
    uuid: str
    jobid: int
    host: str
    minor: int
    label: str          # display form, e.g. "GPU 2" or "MIG 2.0"

    @property
    def csv_id(self) -> str:
        """GPU value for CSV output: "2", or "2.0" for a MIG slice.

        Keeps the existing convention (bare minor number for a whole card) while
        staying unique per slice, since plot groups series by (NODE, GPU).
        """
        return self.label.split(" ", 1)[1]


class Metric(NamedTuple):
    """One queryable GPU metric and how to render it."""
    key: str            # key under which the value is stored
    header: str         # column header
    promql: str         # bare Prometheus metric name
    scale: float        # multiply the raw value by this
    decimals: int       # 0 renders as an integer
    label: str          # GPU UUID label: nvidia_* uses "uuid", DCGM_* uses "UUID"
    agg: str = "avg"    # how --avg folds it over the runtime: "avg" or "max"
    show: bool = True   # False = fetched only to feed a derived column


# Utilization metrics are time-averaged; memory is a peak (max), matching how
# jobstats treats each -- see _averaged_query.
DCGM_METRICS = [
    Metric("duty", "DUTY%", "nvidia_gpu_duty_cycle", 1, 0, "uuid"),
    Metric("smact", "SM_ACT%", "DCGM_FI_PROF_SM_ACTIVE", 100, 1, "UUID"),
    Metric("occ", "OCC%", "DCGM_FI_PROF_SM_OCCUPANCY", 100, 1, "UUID"),
    Metric("tensor", "TENSOR%", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", 100, 1, "UUID"),
    Metric("dram", "DRAM%", "DCGM_FI_PROF_DRAM_ACTIVE", 100, 1, "UUID"),
    Metric("power", "POWER_W", "DCGM_FI_DEV_POWER_USAGE", 1, 0, "UUID"),
    # Same source and aggregation jobstats uses for "GPU memory usage per node
    # - maximum used/total", so MEM_GB/MEM% are directly comparable to it.
    Metric("mem", "MEM_GB", "nvidia_gpu_memory_used_bytes",
           1 / 1024 ** 3, 1, "uuid", agg="max"),
    Metric("memtot", "", "nvidia_gpu_memory_total_bytes",
           1 / 1024 ** 3, 1, "uuid", agg="max", show=False),
]

GPU_SUMMARY_METRICS = DCGM_METRICS[1:]  # all except duty
ALL_METRICS = [
    Metric("engine", "ENGINE%", "DCGM_FI_PROF_GR_ENGINE_ACTIVE", 100, 1, "UUID"),
    Metric("hmma", "HMMA%", "DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE", 100, 1, "UUID"),
    Metric("imma", "IMMA%", "DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE", 100, 1, "UUID"),
    Metric("dfma", "DFMA%", "DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE", 100, 1, "UUID"),
    Metric("fp16", "FP16%", "DCGM_FI_PROF_PIPE_FP16_ACTIVE", 100, 1, "UUID"),
    Metric("fp32", "FP32%", "DCGM_FI_PROF_PIPE_FP32_ACTIVE", 100, 1, "UUID"),
    Metric("fp64", "FP64%", "DCGM_FI_PROF_PIPE_FP64_ACTIVE", 100, 1, "UUID"),
    Metric("memcp", "MEMCP%", "DCGM_FI_DEV_MEM_COPY_UTIL", 1, 0, "UUID"),
    Metric("temp", "TEMP_C", "DCGM_FI_DEV_GPU_TEMP", 1, 0, "UUID"),
    # DCGM's own framebuffer reading; MEM_GB above is the jobstats-comparable one.
    Metric("fbused", "FB_USED_GB", "DCGM_FI_DEV_FB_USED", 1 / 1024, 1, "UUID", agg="max"),
]


def _mem_percent(values: Dict[str, Optional[float]]) -> Optional[float]:
    """Peak GPU memory used as a percentage of the card's total."""
    used, total = values.get("mem"), values.get("memtot")
    if used is None or not total:
        return None
    return round(used / total * 100, 1)


# Columns computed from fetched metrics rather than queried. Each declares the
# keys it needs so it only appears when those metrics were actually collected.
DERIVED_COLUMNS = [
    ("mempct", "MEM%", ("mem", "memtot"), _mem_percent),
]


def mask_url(text: str) -> str:
    """Redact credentials embedded in a URL's userinfo.

    PROM_SERVER carries a Grafana Cloud API token as basic auth, and this string
    otherwise reaches --help output and requests' exception messages -- which land
    in shell history, logs and pasted terminal output.
    """
    return re.sub(r"//[^/@\s]+@", "//<redacted>@", text)


def parse_time_cutoff(cutoff_str: str) -> int:
    """Parse time cutoff string (e.g., '5m', '1h', '2d') to seconds."""
    match = re.match(r"^(\d+)([smhd])$", cutoff_str.strip())
    if not match:
        raise ValueError(
            f"Invalid time cutoff format: {cutoff_str}. Use format like '5m', '1h', '2d'"
        )
    value, unit = int(match.group(1)), match.group(2)
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return value * multipliers[unit]


def parse_start_time(start_time: str) -> Tuple[Optional[int], Optional[int]]:
    """(start epoch, elapsed seconds) from a squeue %S value.

    (None, None) when it is not a timestamp -- squeue prints "N/A"/"Unknown"
    while a job has no start time yet. %S is local time, hence mktime.
    """
    try:
        # Python 3.6 compatibility: strptime instead of fromisoformat
        start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError):
        return None, None
    epoch = int(time.mktime(start_dt.timetuple()))
    return epoch, int(time.time()) - epoch


def filter_jobs_by_runtime(
    jobs: Dict[int, Job], min_runtime_seconds: int
) -> Dict[int, Job]:
    """Filter jobs to keep only those running for > min_runtime_seconds."""
    filtered = {}

    for jobid, job in jobs.items():
        elapsed = job["elapsed_seconds"]
        if elapsed is None:
            print(f"# No usable start time for job {job['jobid']}", file=sys.stderr)
            continue
        if elapsed > min_runtime_seconds:
            filtered[jobid] = job

    return filtered


class PrometheusQuerier:
    """Minimal Prometheus client for live metrics."""

    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url
        self.timeout = timeout

    def instant_query(self, query: str, now: Optional[int] = None) -> List[dict]:
        """Run an instant query at the current time (or specified epoch)."""
        try:
            resp = requests.get(
                self.url + "/api/v1/query",
                params={"query": query, "time": now} if now else {"query": query},
                timeout=self.timeout,
            )
            payload = resp.json()
            return payload["data"]["result"] if payload.get("status") == "success" else []
        except Exception as e:
            # requests puts the request URL in its exception text, so mask it.
            print(f"# Prometheus query failed: {mask_url(str(e))}", file=sys.stderr)
            return []

    def query_range(self, query: str, start: int, end: int, step: int) -> List[dict]:
        """Run a range query. Each result carries a "values" list of [epoch, value]."""
        try:
            resp = requests.get(
                self.url + "/api/v1/query_range",
                params={"query": query, "start": start, "end": end, "step": step},
                timeout=self.timeout,
            )
            payload = resp.json()
            if payload.get("status") != "success":
                print("# Prometheus range query failed: %s"
                      % mask_url(str(payload.get("error", "unknown"))), file=sys.stderr)
                return []
            return payload["data"]["result"]
        except Exception as e:
            print(f"# Prometheus range query failed: {mask_url(str(e))}", file=sys.stderr)
            return []


# %A first, then %i. %A is the *raw* per-element job ID, which is what
# Prometheus' nvidia_gpu_jobId reports; %i is the display form and differs for
# array elements (%i=34843528_6 -> %A=34843629). Pipe-delimited because the
# start time contains colons.
_SQUEUE_FMT = "%A|%i|%u|%N|%g|%j|%G|%C|%S"


def _parse_squeue(stdout: str, user: Optional[str] = None) -> Dict[int, Job]:
    """Parse squeue output in _SQUEUE_FMT into {raw_jobid: {field: value}}.

    Keyed by the raw job ID so the Prometheus join works for array elements too;
    the displayed ID keeps squeue's own notation.
    """
    jobs = {}
    for line in stdout.strip().split("\n"):
        if not line or line.startswith("JOBID"):
            continue
        parts = line.split("|")
        if len(parts) < 9:
            continue
        raw_id, disp_id, user_name, nodelist, group, name, gpus, cpus, start_time = parts[:9]

        # Filter by user if specified
        if user and user_name != user:
            continue

        try:
            raw_jid = int(raw_id)
        except ValueError:
            continue

        start_epoch, elapsed = parse_start_time(start_time)
        jobs[raw_jid] = {
            "jobid": disp_id,  # display form; array notation preserved
            "user": user_name,
            "node": nodelist,
            "group": group,
            "name": name,
            "gpus": gpus,
            "cpus": cpus,
            "start_time": start_time,  # format: YYYY-MM-DDTHH:MM:SS
            # Runtime drives the --min-runtime filter, the --avg window and the
            # --ts range, so derive it once here.
            "start_epoch": start_epoch,
            "elapsed_seconds": elapsed,
        }

    return jobs


def squeue_job_by_id(jobid: str) -> Dict[int, Job]:
    """Fetch one job by ID from squeue. Return {raw_jobid: {field: value}}.

    Accepts a plain job ID (34843528) or an array element (34843528_6).
    """
    cmd = ["squeue", "-h", "-j", str(jobid), "-o", _SQUEUE_FMT]

    try:
        result = subprocess.run(cmd, check=True, **_PIPE_KWARGS)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if hasattr(e, "stderr") else "unknown error"
        print(f"# squeue failed for job {jobid}: {stderr}", file=sys.stderr)
        return {}

    return _parse_squeue(result.stdout)


def squeue_running_jobs(partition: Optional[str] = None,
                        user: Optional[str] = None) -> Dict[int, Job]:
    """Fetch running jobs from squeue. Return {raw_jobid: {field: value}}."""
    cmd = ["squeue", "-h", "-t", "RUNNING", "-o", _SQUEUE_FMT]

    if partition:
        cmd.extend(["-p", partition])

    try:
        result = subprocess.run(cmd, check=True, **_PIPE_KWARGS)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if hasattr(e, "stderr") else "unknown error"
        print(f"# squeue failed: {stderr}", file=sys.stderr)
        return {}

    return _parse_squeue(result.stdout, user=user)


def _gpu_labels(found: List[Tuple[str, int, str, int]]) -> Dict[str, str]:
    """Display label per GPU UUID, from (uuid, jobid, host, minor) tuples.

    minor_number is NOT unique per schedulable GPU: on a MIG node every instance
    inherits its parent card's minor_number and ordinal, so a 3g.20gb pair both
    report minor 0. Only the UUID is unique -- MIG instances carry a "MIG-" UUID
    where whole cards carry "GPU-". NVML exposes no instance index, so enumerate
    the siblings sharing a (job, host, minor) by sorted UUID, which is stable for
    as long as the partitioning is.
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


def discover_job_gpus(prom: PrometheusQuerier,
                      jobs: Dict[int, Job]) -> Dict[str, Gpu]:
    """Map GPU UUID -> Gpu for the given jobs.

    The job ID is nvidia_gpu_jobId's *value*, not a label, so there is nothing to
    filter on server-side: one instant query returns every GPU's current job and we
    keep the ones we asked about. Deliberately instant rather than windowed -- a
    single GPU can host a dozen jobs in a day, so a windowed lookup would hand the
    same GPU to several of them.
    """
    found = []
    for s in prom.instant_query("nvidia_gpu_jobId"):
        try:
            jobid = int(float(s["value"][1]))
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        if jobid not in jobs:
            continue
        labels = s["metric"]
        uuid = labels.get("uuid")  # nvidia_* exporter uses lowercase "uuid"
        minor = labels.get("minor_number")
        if not uuid or minor is None:
            continue
        try:
            found.append((uuid, jobid, labels.get("host", ""), int(minor)))
        except ValueError:
            continue

    display = _gpu_labels(found)
    return {uuid: Gpu(uuid, jobid, host, minor, display[uuid])
            for uuid, jobid, host, minor in found}


def _scale_value(raw: str, scale: float, decimals: int) -> Optional[float]:
    """Apply a metric's scale factor and rounding. None if unparseable."""
    try:
        value = float(raw) * scale
    except (TypeError, ValueError):
        return None
    return round(value, decimals) if decimals else int(round(value))


def _averaged_query(metric: Metric, uuid_regex: str,
                    window: int, raw_jobid: int) -> str:
    """Build the over-the-runtime query matching how jobstats folds this metric.

    jobstats reports utilization as an average over the job's whole runtime but
    memory as the peak, so an instant read will not agree with it on a bursty job.
    Mirror both: a [<runtime>s:] subquery under the metric's own aggregation.

    For nvidia_* metrics we can additionally intersect with nvidia_gpu_jobId ==
    <raw_jobid>, exactly as jobstats does. Both series come from the same exporter
    and so carry identical label sets, which `and` requires; that clips the window
    to the samples where this job actually owned the GPU, making the window length
    a harmless upper bound. DCGM_* metrics come from a different exporter with
    different labels (UUID/Hostname/gpu), so `and` cannot match and the window
    alone bounds them -- the same compromise jobstats_extended makes.
    """
    selector = f"{metric.promql}{{{metric.label}=~\"{uuid_regex}\"}}"
    if metric.label == "uuid":
        selector = f"({selector} and nvidia_gpu_jobId == {raw_jobid})"
    return f"{metric.agg}_over_time({selector}[{window}s:])"


def query_gpu_metrics(prom: PrometheusQuerier,
                      jobs: Dict[int, Job],
                      metrics: List[Metric],
                      average: bool = False) -> Tuple[GpuMetrics, Dict[str, Gpu]]:
    """Collect GPU metrics. Returns (values keyed by job/UUID, GPU identities).

    average=False reads the current instant (one query per metric, all GPUs at
    once). average=True folds each metric over each job's runtime -- averaging
    utilization, peaking memory -- so the numbers line up with jobstats, which
    costs one query per job per metric.
    """
    if not jobs:
        return {}, {}

    gpus = discover_job_gpus(prom, jobs)
    if not gpus:
        print("# No GPU data available in Prometheus", file=sys.stderr)
        return {}, {}

    results = defaultdict(lambda: defaultdict(dict))

    if average:
        _collect_averaged(prom, jobs, metrics, gpus, results)
        _add_derived(results, metrics)
        return results, gpus

    # Instant: one query per metric covering every GPU we care about.
    uuid_regex = "^(" + "|".join(gpus) + ")$"
    for metric in metrics:
        q = f"{metric.promql}{{{metric.label}=~\"{uuid_regex}\"}}"
        for s in prom.instant_query(q):
            uuid = s["metric"].get(metric.label)
            gpu = gpus.get(uuid)
            if gpu is None:
                continue
            try:
                results[gpu.jobid][uuid][metric.key] = _scale_value(
                    s["value"][1], metric.scale, metric.decimals)
            except (KeyError, IndexError):
                results[gpu.jobid][uuid][metric.key] = None

    _add_derived(results, metrics)
    return results, gpus


def _add_derived(results: GpuMetrics, metrics: List[Metric]) -> None:
    """Fill in derived columns whose input metrics were all collected."""
    fetched = {m.key for m in metrics}
    applicable = [(key, fn) for key, _hdr, deps, fn in DERIVED_COLUMNS
                  if fetched.issuperset(deps)]
    if not applicable:
        return
    for per_gpu in results.values():
        for values in per_gpu.values():
            for key, fn in applicable:
                values[key] = fn(values)


def _collect_averaged(prom: PrometheusQuerier,
                      jobs: Dict[int, Job],
                      metrics: List[Metric],
                      gpus: Dict[str, Gpu],
                      results: GpuMetrics) -> None:
    """Fill `results` with each metric folded over each job's own runtime.

    The window differs per job and PromQL cannot vary a window per series, so
    this is one query per (job, metric) -- run concurrently, since that count
    grows quickly with partition size.
    """
    # Group this job's GPUs so one query covers all of them at its window.
    job_uuids = defaultdict(list)
    for uuid, gpu in gpus.items():
        job_uuids[gpu.jobid].append(uuid)

    tasks = []
    for jobid, uuids in job_uuids.items():
        window = jobs[jobid].get("elapsed_seconds")
        if not window or window <= 0:
            print(f"# Skipping average for job {jobs[jobid]['jobid']}: unknown runtime",
                  file=sys.stderr)
            continue
        uuid_regex = "^(" + "|".join(uuids) + ")$"
        for metric in metrics:
            tasks.append((jobid, window, uuid_regex, metric))

    if not tasks:
        return

    def run(task):
        jobid, window, uuid_regex, metric = task
        return task, prom.instant_query(
            _averaged_query(metric, uuid_regex, window, jobid))

    workers = min(_MAX_QUERY_WORKERS, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task, series in pool.map(run, tasks):
            jobid, _window, _uuid_regex, metric = task
            for s in series:
                uuid = s["metric"].get(metric.label)
                gpu = gpus.get(uuid)
                # A GPU reassigned mid-window can surface under another job here.
                if gpu is None or gpu.jobid != jobid:
                    continue
                try:
                    results[jobid][uuid][metric.key] = _scale_value(
                        s["value"][1], metric.scale, metric.decimals)
                except (KeyError, IndexError):
                    results[jobid][uuid][metric.key] = None


def timeseries_step(elapsed: int, requested: Optional[int] = None) -> int:
    """Range-query step in seconds.

    Never finer than the scrape interval -- there is no more data -- and coarse
    enough to stay under Prometheus' ~11k points-per-series cap on long jobs,
    which is the same bound src/jobscope/report.py uses.
    """
    if requested:
        return max(1, requested)
    return max(SAMPLING_PERIOD, elapsed // 10000 + 1)


def emit_timeseries_csv(prom: PrometheusQuerier,
                        jobs: Dict[int, Job],
                        metrics: List[Metric],
                        gpus: Dict[str, Gpu],
                        step: Optional[int] = None,
                        out=None) -> None:
    """Write one CSV row per GPU per sample over each job's runtime.

    Schema is deliberately the one `jobscope dcgm --ts --csv` emits
    (JOBID,EPOCH,TIME,NODE,GPU,<metrics>) so this feeds `jobscope plot` directly.
    GPU is the minor number, or minor.instance for a MIG slice, because plot
    groups series by (NODE, GPU) and slices would otherwise merge.
    """
    out = out or sys.stdout
    columns = build_columns(metrics)

    by_job = defaultdict(list)
    for gpu in gpus.values():
        by_job[gpu.jobid].append(gpu)

    # (job, metric) range queries are independent; fetch them concurrently and
    # sort afterwards so row order stays deterministic.
    tasks = []
    for jobid, job_gpus in by_job.items():
        job = jobs[jobid]
        start, elapsed = job.get("start_epoch"), job.get("elapsed_seconds")
        if not start or not elapsed or elapsed <= 0:
            print("# Skipping timeseries for job %s: unknown runtime" % job["jobid"],
                  file=sys.stderr)
            continue
        regex = "^(" + "|".join(g.uuid for g in job_gpus) + ")$"
        span = timeseries_step(elapsed, step)
        for metric in metrics:
            tasks.append((jobid, regex, start, start + elapsed, span, metric))

    if not tasks:
        return

    def run(task):
        _jobid, regex, start, end, span, metric = task
        query = f"{metric.promql}{{{metric.label}=~\"{regex}\"}}"
        return task, prom.query_range(query, start, end, span)

    # {uuid: {epoch: {metric_key: value}}}
    samples = defaultdict(lambda: defaultdict(dict))
    workers = min(_MAX_QUERY_WORKERS, len(tasks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for task, results in pool.map(run, tasks):
            jobid, _regex, _start, _end, _span, metric = task
            for series in results:
                uuid = series["metric"].get(metric.label)
                gpu = gpus.get(uuid)
                if gpu is None or gpu.jobid != jobid:
                    continue
                for stamp, raw in series.get("values", []):
                    try:
                        epoch = int(float(stamp))
                    except (TypeError, ValueError):
                        continue
                    samples[uuid][epoch][metric.key] = _scale_value(
                        raw, metric.scale, metric.decimals)

    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["JOBID", "EPOCH", "TIME", "NODE", "GPU"]
                    + [header for _key, header, _w in columns])

    derived = [(key, fn) for key, _h, deps, fn in DERIVED_COLUMNS
               if {m.key for m in metrics}.issuperset(deps)]

    for jobid in sorted(jobs, key=lambda j: job_sort_key(jobs[j])):
        label = jobs[jobid]["jobid"]
        for gpu in sorted(by_job.get(jobid, []), key=lambda g: (g.host, g.minor, g.uuid)):
            for epoch in sorted(samples.get(gpu.uuid, {})):
                values = samples[gpu.uuid][epoch]
                for key, fn in derived:
                    values[key] = fn(values)
                writer.writerow(
                    [label, epoch,
                     time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch)),
                     gpu.host, gpu.csv_id]
                    + ["" if values.get(k) is None else values[k]
                       for k, _h, _w in columns])


def build_columns(metrics: List[Metric]) -> List[Tuple[str, str, int]]:
    """(key, header, width) per displayed column, including derived ones."""
    cols = [(m.key, m.header) for m in metrics if m.show]
    fetched = {m.key for m in metrics}
    for key, header, deps, _fn in DERIVED_COLUMNS:
        if not fetched.issuperset(deps):
            continue
        # Sit beside the column it is derived from rather than at the far end.
        from_deps = [i for i, (k, _h) in enumerate(cols) if k in deps]
        cols.insert(max(from_deps) + 1 if from_deps else len(cols), (key, header))
    # Widen to the header, so long ones like FB_USED_GB do not skew the table.
    return [(key, header, max(6, len(header))) for key, header in cols]


def format_job_row(job: Job, gpu_data: Dict[str, Dict[str, Optional[float]]],
                   gpus: List[Gpu], columns: List[Tuple[str, str, int]],
                   gpu_width: int) -> str:
    """Format one row per GPU of this job (one per MIG instance, where used)."""
    # Truncate to the column widths: array IDs and multi-node nodelists are long
    # enough to push the table out of alignment otherwise.
    jobid = job["jobid"][:12]
    user = job["user"][:12]
    name = job["name"][:15]

    if not gpus:
        # No GPU data; fall back to squeue's nodelist for the node column
        return (f"{jobid:<12} {user:<12} {job['node'][:15]:<15} {name:<15}"
                " [no GPU data]")

    rows = []
    # One row per GPU, so the node column can name that GPU's own host rather
    # than a nodelist -- which matters for a job spread over several nodes.
    for gpu in sorted(gpus, key=lambda g: (g.host, g.minor, g.uuid)):
        values = gpu_data.get(gpu.uuid, {})
        cells = []
        for key, _header, width in columns:
            val = values.get(key)
            cells.append(f"{'-' if val is None else val:>{width}}")
        host = (gpu.host or job["node"])[:15]
        rows.append(f"{jobid:<12} {user:<12} {host:<15} {name:<15} "
                    f"{gpu.label:>{gpu_width}}  " + "  ".join(cells))

    return "\n".join(rows)


def job_sort_key(job: Job) -> Tuple[int, int]:
    """Sort key from the displayed job ID: (base id, array index).

    Keeps elements of one array together and in numeric index order; raw IDs are
    assigned in submission order and would interleave unrelated jobs.
    """
    base, _, index = job["jobid"].partition("_")
    try:
        base_n = int(base)
    except ValueError:
        return (sys.maxsize, 0)
    try:
        # No index -> plain job, sorts ahead of any element of the same base
        index_n = int(index) if index else -1
    except ValueError:
        index_n = -1
    return (base_n, index_n)


def print_results(jobs: Dict[int, Job],
                  gpu_metrics: GpuMetrics,
                  gpus: Dict[str, Gpu],
                  columns: List[Tuple[str, str, int]],
                  average: bool = False) -> None:
    """Pretty-print the job metrics."""
    if not jobs:
        print("# No running jobs in this partition", file=sys.stderr)
        return

    # Say which reading this is; the two are not comparable on a bursty job.
    print("# " + ("folded over each job's runtime -- utilization averaged, "
                  "memory peak (comparable to jobstats)"
                  if average else "instantaneous snapshot"))

    by_job = defaultdict(list)
    for gpu in gpus.values():
        by_job[gpu.jobid].append(gpu)

    # Widen the GPU column only if a longer label ("MIG 0.1") is actually present.
    gpu_width = max([5] + [len(g.label) for g in gpus.values()])

    # Header
    headers = [f"{header:>{width}}" for _key, header, width in columns]
    header_line = (
        f"{'JOBID':<12} {'USER':<12} {'NODE':<15} {'NAME':<15} "
        f"{'GPU':>{gpu_width}}  " + "  ".join(headers)
    )
    print(header_line)
    print("-" * len(header_line))

    # Rows
    for jobid in sorted(jobs, key=lambda j: job_sort_key(jobs[j])):
        job = jobs[jobid]
        print(format_job_row(job, gpu_metrics.get(jobid, {}),
                             by_job.get(jobid, []), columns, gpu_width))


def main():
    parser = argparse.ArgumentParser(
        description="Live GPU metrics for running jobs in a partition"
    )
    parser.add_argument(
        "-j",
        "--jobid",
        default=None,
        help="specific Slurm job ID or array job element (e.g., 34843528 or 34843528_6)",
    )
    parser.add_argument(
        "-p",
        "--partition",
        default=None,
        help="Slurm partition (default: all running jobs)",
    )
    parser.add_argument(
        "-u",
        "--user",
        default=None,
        help="filter by user (default: all users)",
    )
    parser.add_argument(
        "--min-runtime",
        default="1h",
        help="only show jobs running for more than this duration (default: 1h; e.g., '5m', '1h', '2d')",
    )
    parser.add_argument(
        "--prom",
        default=PROM_SERVER,
        # Masked: the site default embeds an API token.
        help=f"Prometheus URL (default: {mask_url(PROM_SERVER)})",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="GPU columns only (no full DCGM)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="all DCGM metrics (extended catalog)",
    )
    parser.add_argument(
        "--avg",
        action="store_true",
        help="average each metric over each job's runtime, so values are "
             "comparable to jobstats (default: instantaneous snapshot)",
    )
    parser.add_argument(
        "--ts",
        action="store_true",
        help="emit a timeseries CSV over each job's runtime instead of a table; "
             "same schema as 'jobscope dcgm --ts --csv', so it pipes to "
             "'jobscope plot'. Ignores --avg",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=f"--ts sample interval in seconds (default: the {SAMPLING_PERIOD}s "
             "scrape interval, widened on long jobs to stay under Prometheus' "
             "point cap)",
    )

    args = parser.parse_args()

    # Select metrics
    if args.all:
        metrics = DCGM_METRICS + ALL_METRICS
    elif args.gpu:
        metrics = GPU_SUMMARY_METRICS
    else:
        metrics = DCGM_METRICS

    # Fetch and query
    if args.jobid:
        # An explicit job is looked up as-is: no partition/user narrowing and no
        # runtime floor, and a miss is an error rather than an empty table.
        jobs = squeue_job_by_id(args.jobid)
        if not jobs:
            print(f"# Job {args.jobid} is not running", file=sys.stderr)
            sys.exit(1)
    else:
        jobs = squeue_running_jobs(partition=args.partition, user=args.user)

        try:
            min_runtime_seconds = parse_time_cutoff(args.min_runtime)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        jobs = filter_jobs_by_runtime(jobs, min_runtime_seconds)

    if not jobs:
        print_results(jobs, {}, {}, build_columns(metrics))
        return

    prom = PrometheusQuerier(args.prom)

    if args.ts:
        if args.avg:
            print("# --avg ignored with --ts (the CSV carries every sample)",
                  file=sys.stderr)
        gpus = discover_job_gpus(prom, jobs)
        if not gpus:
            print("# No GPU data available in Prometheus", file=sys.stderr)
            sys.exit(1)
        emit_timeseries_csv(prom, jobs, metrics, gpus, step=args.step)
        return

    gpu_metrics, gpus = query_gpu_metrics(prom, jobs, metrics, average=args.avg)
    print_results(jobs, gpu_metrics, gpus, build_columns(metrics), average=args.avg)


if __name__ == "__main__":
    main()
