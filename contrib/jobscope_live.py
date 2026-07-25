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

Author: Bala Desinghu, Senior AI/HPC Research Computing Engineer, Kempner Institute, Harvard
"""

import sys
import json
import argparse
import subprocess
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

# One job's squeue fields (all str) plus elapsed_seconds (Optional[int]).
Job = Dict[str, Any]
# {raw_jobid: {gpu_minor: {metric_key: value}}}
GpuMetrics = Dict[int, Dict[int, Dict[str, Optional[float]]]]

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

import requests


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


def elapsed_seconds(start_time: str) -> Optional[int]:
    """Seconds since a squeue %S start time, or None if it is not a timestamp.

    squeue prints "N/A"/"Unknown" when a job has no start time yet.
    """
    try:
        # Python 3.6 compatibility: strptime instead of fromisoformat
        start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
    except (ValueError, TypeError):
        return None
    return int((datetime.now() - start_dt).total_seconds())


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

        jobs[raw_jid] = {
            "jobid": disp_id,  # display form; array notation preserved
            "user": user_name,
            "node": nodelist,
            "group": group,
            "name": name,
            "gpus": gpus,
            "cpus": cpus,
            "start_time": start_time,  # format: YYYY-MM-DDTHH:MM:SS
            # Runtime is both the --min-runtime filter input and the averaging
            # window for --avg, so derive it once here.
            "elapsed_seconds": elapsed_seconds(start_time),
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


def discover_job_gpus(prom: PrometheusQuerier,
                      jobs: Dict[int, Job]) -> Dict[str, Tuple[int, int]]:
    """Map GPU UUID -> (raw_jobid, gpu_minor) for the given jobs.

    The job ID is nvidia_gpu_jobId's *value*, not a label, so there is nothing to
    filter on server-side: one instant query returns every GPU's current job and we
    keep the ones we asked about. Deliberately instant rather than windowed -- a
    single GPU can host a dozen jobs in a day, so a windowed lookup would hand the
    same GPU to several of them.
    """
    uuid_to_job = {}
    for s in prom.instant_query("nvidia_gpu_jobId"):
        try:
            jobid = int(float(s["value"][1]))
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        if jobid not in jobs:
            continue
        uuid = s["metric"].get("uuid")  # nvidia_* exporter uses lowercase "uuid"
        minor = s["metric"].get("minor_number")
        if not uuid or minor is None:
            continue
        try:
            uuid_to_job[uuid] = (jobid, int(minor))
        except ValueError:
            continue

    return uuid_to_job


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
                      metrics: List[Tuple[str, str, str, float, int, str]],
                      average: bool = False) -> GpuMetrics:
    """Collect GPU metrics for the given jobs.

    average=False reads the current instant (one query per metric, all GPUs at
    once). average=True averages each metric over each job's own runtime so the
    numbers line up with jobstats, which costs one query per job per metric.
    """
    if not jobs:
        return {}

    uuid_to_job = discover_job_gpus(prom, jobs)
    if not uuid_to_job:
        print("# No GPU data available in Prometheus", file=sys.stderr)
        return {}

    results = defaultdict(lambda: defaultdict(dict))

    if average:
        _collect_averaged(prom, jobs, metrics, uuid_to_job, results)
        _add_derived(results, metrics)
        return results

    # Instant: one query per metric covering every GPU we care about.
    uuid_regex = "^(" + "|".join(uuid_to_job.keys()) + ")$"
    for metric in metrics:
        q = f"{metric.promql}{{{metric.label}=~\"{uuid_regex}\"}}"
        for s in prom.instant_query(q):
            uuid = s["metric"].get(metric.label)
            if uuid not in uuid_to_job:
                continue
            jobid, minor = uuid_to_job[uuid]
            try:
                results[jobid][minor][metric.key] = _scale_value(
                    s["value"][1], metric.scale, metric.decimals)
            except (KeyError, IndexError):
                results[jobid][minor][metric.key] = None

    _add_derived(results, metrics)
    return results


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
                      uuid_to_job: Dict[str, Tuple[int, int]],
                      results: GpuMetrics) -> None:
    """Fill `results` with each metric folded over each job's own runtime.

    The window differs per job and PromQL cannot vary a window per series, so
    this is one query per (job, metric) -- run concurrently, since that count
    grows quickly with partition size.
    """
    # Group this job's GPUs so one query covers all of them at its window.
    job_uuids = defaultdict(list)
    for uuid, (jobid, _minor) in uuid_to_job.items():
        job_uuids[jobid].append(uuid)

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
                mapped = uuid_to_job.get(uuid)
                # A GPU reassigned mid-window can surface under another job here.
                if not mapped or mapped[0] != jobid:
                    continue
                minor = mapped[1]
                try:
                    results[jobid][minor][metric.key] = _scale_value(
                        s["value"][1], metric.scale, metric.decimals)
                except (KeyError, IndexError):
                    results[jobid][minor][metric.key] = None


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


def format_job_row(job: Job, gpu_data: Dict[int, Dict[str, Optional[float]]],
                   columns: List[Tuple[str, str, int]]) -> str:
    """Format one job row with GPU metrics per GPU."""
    # Truncate to the column widths: array IDs and multi-node nodelists are long
    # enough to push the table out of alignment otherwise.
    jobid = job["jobid"][:12]
    user = job["user"][:12]
    node = job["node"][:15]
    name = job["name"][:15]
    lead = f"{jobid:<12} {user:<12} {node:<15} {name:<15}"

    # Collect metrics across GPUs in this job
    all_gpu_minors = sorted(gpu_data.keys())
    if not all_gpu_minors:
        # No GPU data; just show the job
        return f"{lead} [no GPU data]"

    rows = []
    for gpu_minor in all_gpu_minors:
        values = gpu_data[gpu_minor]
        cells = []
        for key, _header, width in columns:
            val = values.get(key)
            cells.append(f"{'-' if val is None else val:>{width}}")
        rows.append(f"{lead} GPU{gpu_minor:>2}  " + "  ".join(cells))

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

    # Header
    headers = [f"{header:>{width}}" for _key, header, width in columns]
    header_line = (
        f"{'JOBID':<12} {'USER':<12} {'NODE':<15} {'NAME':<15} {'GPU':>5}  "
        + "  ".join(headers)
    )
    print(header_line)
    print("-" * len(header_line))

    # Rows
    for jobid in sorted(jobs, key=lambda j: job_sort_key(jobs[j])):
        job = jobs[jobid]
        gpu_data = gpu_metrics.get(jobid, {})
        print(format_job_row(job, gpu_data, columns))


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

    if jobs:
        prom = PrometheusQuerier(args.prom)
        gpu_metrics = query_gpu_metrics(prom, jobs, metrics, average=args.avg)
    else:
        gpu_metrics = {}

    # Print
    print_results(jobs, gpu_metrics, build_columns(metrics), average=args.avg)


if __name__ == "__main__":
    main()
