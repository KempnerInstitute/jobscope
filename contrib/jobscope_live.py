#!/usr/bin/env python3
"""jobscope_live - Real-time GPU metrics for running jobs in a partition.

Quick view of GPU utilization and DCGM metrics for jobs currently running on a
given partition, using squeue (instant; no history) + Prometheus (at the current
moment). No historical analysis or job reconstruction -- just the live snapshot.

Usage:
  ./jobscope_live.py                # all running jobs >1h (default)
  ./jobscope_live.py -j 34622920            # specific job by ID
  ./jobscope_live.py -p kempner            # jobs in partition
  ./jobscope_live.py -p kempner -u alice
  ./jobscope_live.py -p kempner --gpu      # GPU columns only
  ./jobscope_live.py -p kempner --all      # all DCGM metrics
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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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


# DCGM metrics (same catalog as the main jobscope package)
# Note: nvidia_gpu_duty_cycle uses lowercase "uuid", DCGM_* metrics use uppercase "UUID"
DCGM_METRICS = [
    ("duty", "DUTY%", "nvidia_gpu_duty_cycle", 1, 0, "uuid"),
    ("smact", "SM_ACT%", "DCGM_FI_PROF_SM_ACTIVE", 100, 1, "UUID"),
    ("occ", "OCC%", "DCGM_FI_PROF_SM_OCCUPANCY", 100, 1, "UUID"),
    ("tensor", "TENSOR%", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", 100, 1, "UUID"),
    ("dram", "DRAM%", "DCGM_FI_PROF_DRAM_ACTIVE", 100, 1, "UUID"),
    ("power", "POWER_W", "DCGM_FI_DEV_POWER_USAGE", 1, 0, "UUID"),
]

GPU_SUMMARY_METRICS = DCGM_METRICS[1:]  # all except duty
ALL_METRICS = [
    ("engine", "ENGINE%", "DCGM_FI_PROF_GR_ENGINE_ACTIVE", 100, 1, "UUID"),
    ("hmma", "HMMA%", "DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE", 100, 1, "UUID"),
    ("imma", "IMMA%", "DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE", 100, 1, "UUID"),
    ("dfma", "DFMA%", "DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE", 100, 1, "UUID"),
    ("fp16", "FP16%", "DCGM_FI_PROF_PIPE_FP16_ACTIVE", 100, 1, "UUID"),
    ("fp32", "FP32%", "DCGM_FI_PROF_PIPE_FP32_ACTIVE", 100, 1, "UUID"),
    ("fp64", "FP64%", "DCGM_FI_PROF_PIPE_FP64_ACTIVE", 100, 1, "UUID"),
    ("memcp", "MEMCP%", "DCGM_FI_DEV_MEM_COPY_UTIL", 1, 0, "UUID"),
    ("temp", "TEMP_C", "DCGM_FI_DEV_GPU_TEMP", 1, 0, "UUID"),
    ("fbused", "FB_USED_GB", "DCGM_FI_DEV_FB_USED", 1 / 1024, 1, "UUID"),
]


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


def filter_jobs_by_runtime(
    jobs: Dict[int, Dict[str, str]], min_runtime_seconds: int
) -> Dict[int, Dict[str, str]]:
    """Filter jobs to keep only those running for > min_runtime_seconds."""
    now = datetime.now()
    filtered = {}

    for jobid, job in jobs.items():
        try:
            # Parse start time: format is "YYYY-MM-DDTHH:MM:SS"
            start_time_str = job["start_time"]
            # Python 3.6 compatibility: use strptime instead of fromisoformat
            start_dt = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S")
            elapsed = (now - start_dt).total_seconds()

            if elapsed > min_runtime_seconds:
                job["elapsed_seconds"] = int(elapsed)
                filtered[jobid] = job
        except (ValueError, KeyError) as e:
            print(f"# Failed to parse start time for job {jobid}: {e}", file=sys.stderr)

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
            print(f"# Prometheus query failed: {e}", file=sys.stderr)
            return []


def squeue_job_by_id(jobid: str) -> Optional[Dict[int, Dict[str, str]]]:
    """Fetch a specific job by ID from squeue. Return {jobid: {field: value}} or None.

    Supports both regular job IDs (e.g., 34843528) and array job elements (e.g., 34843528_6).
    """
    cmd = ["squeue", "-j", str(jobid), "-o", "%i|%u|%N|%g|%j|%G|%C|%S"]

    try:
        result = subprocess.run(cmd, check=True, **_PIPE_KWARGS)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if hasattr(e, 'stderr') else "unknown error"
        print(f"# squeue failed for job {jobid}: {stderr}", file=sys.stderr)
        return None

    jobs = {}
    for line in result.stdout.strip().split("\n"):
        if not line or line.startswith("JOBID"):
            continue
        parts = line.split("|")
        if len(parts) < 8:
            continue
        jobid_str, user_name, nodelist, group, name, gpus, cpus, start_time = parts[:8]

        try:
            # Extract base job ID (before underscore for array jobs)
            base_jid = int(jobid_str.split("_")[0])
            jobs[base_jid] = {
                "jobid": jobid_str,
                "user": user_name,
                "node": nodelist.split(",")[0],  # primary node
                "group": group,
                "name": name,
                "gpus": gpus,
                "cpus": cpus,
                "start_time": start_time,
            }
        except ValueError:
            continue

    return jobs if jobs else None


def squeue_running_jobs(partition: Optional[str] = None,
                        user: Optional[str] = None) -> Dict[int, Dict[str, str]]:
    """Fetch running jobs from squeue. Return {jobid: {field: value}}."""
    # Use pipe delimiter since start time contains colons
    cmd = ["squeue", "-t", "RUNNING", "-o", "%i|%u|%N|%g|%j|%G|%C|%S"]

    if partition:
        cmd.extend(["-p", partition])

    try:
        result = subprocess.run(cmd, check=True, **_PIPE_KWARGS)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if hasattr(e, 'stderr') else "unknown error"
        print(f"# squeue failed: {stderr}", file=sys.stderr)
        return {}

    jobs = {}
    for line in result.stdout.strip().split("\n"):
        if not line or line.startswith("JOBID"):
            continue
        parts = line.split("|")
        if len(parts) < 8:
            continue
        jobid, user_name, nodelist, group, name, gpus, cpus, start_time = parts[:8]

        # Filter by user if specified
        if user and user_name != user:
            continue

        try:
            jobs[int(jobid)] = {
                "jobid": jobid,
                "user": user_name,
                "node": nodelist.split(",")[0],  # primary node
                "group": group,
                "name": name,
                "gpus": gpus,
                "cpus": cpus,
                "start_time": start_time,  # format: YYYY-MM-DDTHH:MM:SS
            }
        except ValueError:
            continue

    return jobs


def query_gpu_metrics(prom: PrometheusQuerier,
                      jobs: Dict[int, Dict[str, str]],
                      metrics: List[Tuple[str, str, str, float, int, str]]) -> Dict[int, Dict[int, Dict[str, Optional[float]]]]:
    """Query Prometheus for GPU metrics. Return {jobid: {gpu_minor: {metric_key: value}}}."""
    if not jobs:
        return {}

    # Build job ID list for Prometheus query using OR clauses
    # E.g., (nvidia_gpu_jobId == id1) or (nvidia_gpu_jobId == id2)
    job_filters = " or ".join(f"(nvidia_gpu_jobId == {jid})" for jid in jobs.keys())

    # Map UUID -> (jobid, gpu_minor) from nvidia_gpu_jobId using a range query.
    # Use max_over_time with a 24h window like jobstats does, since instant queries
    # don't reliably return the data.
    uuid_to_job = {}
    disco_q = f"max_over_time(({job_filters})[86400s:])"
    disco_res = prom.instant_query(disco_q)
    for s in disco_res:
        try:
            uuid = s["metric"].get("uuid")  # Note: lowercase "uuid"
            jobid_str = s["value"][1]  # The value is the jobid
            jobid = int(float(jobid_str))
            minor = s["metric"].get("minor_number")
            if uuid and jobid and minor is not None:
                uuid_to_job[uuid] = (jobid, minor)
        except (KeyError, ValueError, TypeError, IndexError):
            continue

    if not uuid_to_job:
        print("# No GPU data available in Prometheus", file=sys.stderr)
        return {}

    # Query each metric by UUID, store under minor_number
    results = defaultdict(lambda: defaultdict(dict))
    uuid_regex = "^(" + "|".join(uuid_to_job.keys()) + ")$"

    for key, header, metric, scale, decimals, uuid_label in metrics:
        try:
            q = f"{metric}{{{uuid_label}=~\"{uuid_regex}\"}}"
            res = prom.instant_query(q)

            for s in res:
                uuid = s["metric"].get(uuid_label)
                if uuid not in uuid_to_job:
                    continue

                jobid, minor = uuid_to_job[uuid]
                try:
                    value = float(s["value"][1]) * scale
                    if decimals:
                        value = round(value, decimals)
                    else:
                        value = int(round(value))
                    results[jobid][minor][key] = value
                except (TypeError, ValueError, IndexError):
                    results[jobid][minor][key] = None

        except Exception as e:
            print(f"# Failed to query {metric}: {e}", file=sys.stderr)

    return results


def format_job_row(job: Dict[str, str], gpu_data: Dict[int, Dict[str, Optional[float]]],
                   metrics: List[Tuple[str, str, str, float, int, str]]) -> str:
    """Format one job row with GPU metrics per GPU."""
    jobid = job["jobid"]
    user = job["user"]
    node = job["node"]
    name = job["name"][:15]  # truncate name

    # Collect metrics across GPUs in this job
    all_gpu_minors = sorted(gpu_data.keys())
    if not all_gpu_minors:
        # No GPU data; just show the job
        return f"{jobid:<10} {user:<12} {node:<15} {name:<15} [no GPU data]"

    rows = []
    for gpu_minor in all_gpu_minors:
        gpu_metrics = gpu_data[gpu_minor]
        metric_vals = []
        for key, header, *_ in metrics:
            val = gpu_metrics.get(key)
            if val is not None:
                metric_vals.append(f"{val:>6}")
            else:
                metric_vals.append("    -")
        row = f"{jobid:<10} {user:<12} {node:<15} {name:<15} GPU{gpu_minor:>2}  " + "  ".join(
            metric_vals
        )
        rows.append(row)

    return "\n".join(rows)


def print_results(jobs: Dict[int, Dict[str, str]],
                  gpu_metrics: Dict[int, Dict[int, Dict[str, Optional[float]]]],
                  metrics: List[Tuple[str, str, str, float, int, str]]) -> None:
    """Pretty-print the live job metrics."""
    if not jobs:
        print("# No running jobs in this partition", file=sys.stderr)
        return

    # Header
    headers = [f"{h:>6}" for _, h, *_ in metrics]
    print(
        f"{'JOBID':<10} {'USER':<12} {'NODE':<15} {'NAME':<15} {'GPU':>5}  "
        + "  ".join(headers)
    )
    print(
        "-" * (10 + 12 + 15 + 15 + 5 + 2 + len(headers) * 8 + (len(headers) - 1) * 2)
    )

    # Rows
    for jobid in sorted(jobs.keys()):
        job = jobs[jobid]
        gpu_data = gpu_metrics.get(jobid, {})
        row = format_job_row(job, gpu_data, metrics)
        print(row)


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
        help=f"Prometheus URL (default: {PROM_SERVER})",
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
        jobs = squeue_job_by_id(args.jobid)
    else:
        jobs = squeue_running_jobs(partition=args.partition, user=args.user)

    # Filter by minimum runtime (default 1h, can be overridden with --min-runtime)
    # Skip runtime filter if querying specific job
    if not args.jobid and jobs:
        try:
            min_runtime_seconds = parse_time_cutoff(args.min_runtime)
            jobs = filter_jobs_by_runtime(jobs, min_runtime_seconds)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    if jobs:
        prom = PrometheusQuerier(args.prom)
        gpu_metrics = query_gpu_metrics(prom, jobs, metrics)
    else:
        gpu_metrics = {}

    # Print
    print_results(jobs, gpu_metrics, metrics)


if __name__ == "__main__":
    main()
