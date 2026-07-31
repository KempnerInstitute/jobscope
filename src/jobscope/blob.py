"""Decode and summarize the utilization blob Slurm stores in sacct AdminComment.

jobstats serializes per-job resource usage into the AdminComment field as
``JS1:<base64 gzip JSON>``. jobscope decodes exactly that blob, so for completed
jobs its CPU/MEM/GPU/GMEM numbers match jobstats. The JSON carries, per node,
``total_time`` (cpu-seconds), ``cpus``, ``used_memory``, ``total_memory`` and the
per-GPU ``gpu_utilization`` / ``gpu_used_memory`` / ``gpu_total_memory`` maps,
plus a top-level ``total_time`` (elapsed wall time).
"""

import base64
import gzip
import json
import re
from typing import List, Optional, Tuple

GIB = 1024 ** 3

BlobMetrics = Tuple[int, int, Optional[int], Optional[int]]
DetailRow = Tuple[str, str, str, str, str, str, str]


def decode_admin_comment(admin_comment) -> dict:
    """Return the decoded jobstats JSON, or {} when the blob is absent/short/bad."""
    text = str(admin_comment)
    if not admin_comment or text in ("JS1:Short", "JS1:None") or not text.startswith("JS1:"):
        return {}
    try:
        return json.loads(gzip.decompress(base64.b64decode(text[4:])))
    except Exception:
        return {}


def gpus_from_tres(alloc_tres) -> int:
    """Number of GPUs parsed from an AllocTRES string (gres/gpu=N)."""
    match = re.search(r"gres/gpu=(\d+)", str(alloc_tres))
    return int(match.group(1)) if match else 0


def bytes_to_gb(num_bytes: float) -> str:
    """Bytes to a GiB value labeled GB (as jobstats does), trailing zeros trimmed."""
    return "{:.1f}".format(num_bytes / GIB).rstrip("0").rstrip(".") + "GB"


def blob_capacity(stats: dict) -> Tuple[int, int]:
    """Allocated ``(cores, memory_bytes)`` summed over a job's nodes.

    The *size* of the allocation, independent of how much of it was used. Paired
    with elapsed time this gives the core-hours and GB-hours a job was charged,
    which is what weights a utilization average by how much hardware it held and
    for how long -- see :meth:`jobscope.report.SummaryRenderer.finish`. Returns
    zeros for an empty blob, so a caller that multiplies by them contributes
    nothing rather than crashing.
    """
    if not stats or "nodes" not in stats:
        return 0, 0
    nodes = list(stats["nodes"].values())
    return (sum(n.get("cpus", 0) or 0 for n in nodes),
            sum(n.get("total_memory", 0) or 0 for n in nodes))


def blob_metrics(stats: dict) -> Optional[BlobMetrics]:
    """Overall ``(cpu%, mem%, gpu%, gmem%)`` for a job, or None if the blob is empty.

    ``gpu%`` / ``gmem%`` are None for CPU-only jobs. Matches jobstats' overall bars.
    """
    if not stats or "nodes" not in stats:
        return None
    nodes = list(stats["nodes"].values())
    runtime = stats.get("total_time", 0) or 0
    cpu_time = sum(n.get("total_time", 0) for n in nodes)
    cpus = sum(n.get("cpus", 0) for n in nodes)
    cpu = 100 * cpu_time / (runtime * cpus) if runtime and cpus else 0
    used_mem = sum(n.get("used_memory", 0) for n in nodes)
    total_mem = sum(n.get("total_memory", 0) for n in nodes)
    mem = 100 * used_mem / total_mem if total_mem else 0
    gpu_utils = [v for n in nodes for v in n.get("gpu_utilization", {}).values()]
    gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else None
    gpu_used = sum(v for n in nodes for v in n.get("gpu_used_memory", {}).values())
    gpu_total = sum(v for n in nodes for v in n.get("gpu_total_memory", {}).values())
    gmem = 100 * gpu_used / gpu_total if gpu_total else None
    return (round(cpu), round(mem),
            round(gpu) if gpu is not None else None,
            round(gmem) if gmem is not None else None)


def blob_detail(stats: dict) -> List[DetailRow]:
    """Per-node / per-GPU rows matching jobstats' Detailed Utilization layout.

    Each row is ``(node, gpu, cpu%, cpu-mem, gpu%, gpu-mem, gmem%)``.
    """
    rows: List[DetailRow] = []
    if not stats or "nodes" not in stats:
        return rows
    runtime = stats.get("total_time", 0) or 0
    for node, info in stats["nodes"].items():
        cpus = info.get("cpus", 0)
        eff = 100 * info.get("total_time", 0) / (runtime * cpus) if runtime and cpus else 0
        cpu_mem = "%s/%s" % (bytes_to_gb(info.get("used_memory", 0)),
                             bytes_to_gb(info.get("total_memory", 0)))
        gpu_util = info.get("gpu_utilization", {})
        gpu_used = info.get("gpu_used_memory", {})
        gpu_total = info.get("gpu_total_memory", {})
        if gpu_util or gpu_total:
            for gpu in sorted(gpu_util or gpu_total, key=str):
                used, total = gpu_used.get(gpu, 0), gpu_total.get(gpu, 0)
                gmem = 100 * used / total if total else 0
                rows.append((node, str(gpu), "%.1f%%" % eff, cpu_mem,
                             "%g%%" % gpu_util.get(gpu, 0),
                             "%s/%s" % (bytes_to_gb(used), bytes_to_gb(total)),
                             "%.1f%%" % gmem))
        else:
            rows.append((node, "-", "%.1f%%" % eff, cpu_mem, "-", "-", "-"))
    return rows
