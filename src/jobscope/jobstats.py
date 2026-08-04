"""Decode and summarize the utilization summary Slurm stores in sacct AdminComment.

jobstats serializes per-job resource usage into the AdminComment field as
``JS1:<base64 gzip JSON>``. jobscope decodes exactly that summary, so for completed
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

from .models import JobMetrics, Measure

GIB = 1024 ** 3

# What a JobMetrics from here is labelled with, so a mixed report can say which
# column came from where -- jobstats and Prometheus do not always agree.
SOURCE = "jobstats"

DetailRow = Tuple[str, str, str, str, str, str, str]

# The precision each field is stored at. jobstats writes byte counts as integers and
# utilization to one decimal, and :func:`jobstats_detail` renders utilization with %g on
# that assumption -- so a figure reconstructed from Prometheus is rounded here to the
# same precision, or a synthesized row prints "93.1386%" beside stored rows printing
# "93.1". Reconstruction lives in :mod:`jobscope.job_ave_stats`; the precision lives
# here, with the shape that defines it.
_ROUNDING = {"cpus": 0, "total_time": 1, "used_memory": 0, "total_memory": 0,
             "gpu_utilization": 1, "gpu_used_memory": 0, "gpu_total_memory": 0}


def store_as(field: str, value: float):
    """Round ``value`` to the precision the stored summary uses for ``field``."""
    decimals = _ROUNDING.get(field, 1)
    return int(round(value)) if decimals == 0 else round(value, decimals)


def decode_admin_comment(admin_comment) -> dict:
    """Return the decoded jobstats JSON, or {} when the jobstats summary is absent/short/bad."""
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


def jobstats_capacity(stats: dict) -> Tuple[int, int]:
    """Allocated ``(cores, memory_bytes)`` summed over a job's nodes.

    The *size* of the allocation, independent of how much of it was used. Paired
    with elapsed time this gives the core-hours and GB-hours a job was charged,
    which is what weights a utilization average by how much hardware it held and
    for how long -- see :meth:`jobscope.report.SummaryRenderer.finish`. Returns
    zeros for an empty summary, so a caller that multiplies by them contributes
    nothing rather than crashing.
    """
    if not stats or "nodes" not in stats:
        return 0, 0
    nodes = list(stats["nodes"].values())
    return (sum(n.get("cpus", 0) or 0 for n in nodes),
            sum(n.get("total_memory", 0) or 0 for n in nodes))


# The three per-GPU maps a node entry carries, each keyed by minor number as a string.
_GPU_MAPS = ("gpu_utilization", "gpu_used_memory", "gpu_total_memory")


def narrow_stats(stats: dict, nodename: Optional[str] = None, gpu_ids=()) -> dict:
    """``stats`` restricted to one node and/or a set of GPU minor numbers.

    What makes ``--nodename`` and ``--gpuid`` work on the *summary*, where there is
    no row per unit to filter: narrow the numbers the summary is computed from, and
    every figure downstream -- the row, the per-metric table, the bars, the verdict --
    follows without knowing a filter happened.

    The arithmetic stays correct under narrowing because every figure in the jobstats summary is
    per node or per GPU already: CPU% is one node's CPU-seconds over its own cores x
    elapsed, and GPU% is the mean over whichever cards remain. Nothing here is a
    whole-job total that a subset would misrepresent.

    Returns ``{}`` when the filter matches nothing, which reads downstream as "no
    data" rather than as zeros -- the caller checks and says which name missed.
    """
    if not stats or "nodes" not in stats or not (nodename or gpu_ids):
        return stats
    wanted = {str(g) for g in gpu_ids}
    nodes = {}
    for host, node in stats["nodes"].items():
        if nodename and host != nodename:
            continue
        if wanted:
            node = dict(node)
            for name in _GPU_MAPS:
                if name in node:
                    node[name] = {m: v for m, v in node[name].items() if str(m) in wanted}
            # A node whose cards were all filtered out keeps its CPU/memory entry --
            # dropping it would silently change CPU% too, and --gpuid says nothing
            # about cores.
        nodes[host] = node
    return dict(stats, nodes=nodes) if nodes else {}


def gpu_ids_in(stats: dict) -> set:
    """Every GPU minor number the stats carry, as strings -- for "it used: ..."."""
    return {str(m) for node in (stats or {}).get("nodes", {}).values()
            for name in _GPU_MAPS for m in node.get(name, {})}


def gpu_count(stats: dict) -> int:
    """How many *cards* the stats carry, across nodes.

    Not the number of distinct minor numbers: every node numbers its cards from 0,
    so a two-node job holding 0-3 on each has eight cards and four distinct minors.
    Counting minors made ``--gpuid 0,1`` on such a job report two GPUs and weight its
    GPU-hours by two, understating the pooled row by half.
    """
    return sum(len(node.get("gpu_utilization", {}))
               for node in (stats or {}).get("nodes", {}).values())


def nodes_in(stats: dict) -> set:
    """Every node name the stats carry."""
    return set((stats or {}).get("nodes", {}))


def jobstats_metrics(stats: dict, gpus: Optional[int] = None) -> JobMetrics:
    """A job's overall CPU%/MEM%/GPU%/GMEM% from its stats dict.

    Matches jobstats' overall bars. Every absence is labelled rather than blanked,
    and ``gpus`` -- the count Slurm allocated -- is what makes the GPU columns
    answerable: without it a missing GPU reading could mean either "CPU-only job"
    or "the exporter was down", and those must not be the same value.

    Two of these used to be ``else 0``. A summary carrying nodes but no core count
    reported **CPU% 0**, indistinguishable from a genuinely idle job -- and worse
    than a bare absence, because a fabricated zero passes every "is it measured"
    guard downstream and lands in the summary averages as a real reading.
    """
    if not stats or "nodes" not in stats:
        return JobMetrics({}, source=SOURCE)

    nodes = list(stats["nodes"].values())
    runtime = stats.get("total_time", 0) or 0
    found = {}

    cpu_time = sum(n.get("total_time", 0) for n in nodes)
    cpus = sum(n.get("cpus", 0) for n in nodes)
    if runtime and cpus:
        found["CPU%"] = Measure.reading(round(100 * cpu_time / (runtime * cpus)))
    else:
        found["CPU%"] = Measure.unmeasured(
            "the stored summary has no %s" % ("elapsed time" if not runtime else "core count"))

    used_mem = sum(n.get("used_memory", 0) for n in nodes)
    total_mem = sum(n.get("total_memory", 0) for n in nodes)
    if total_mem:
        found["MEM%"] = Measure.reading(round(100 * used_mem / total_mem))
    else:
        found["MEM%"] = Measure.unmeasured("the stored summary has no memory allocation")

    utils = [v for n in nodes for v in n.get("gpu_utilization", {}).values()]
    gpu_used = sum(v for n in nodes for v in n.get("gpu_used_memory", {}).values())
    gpu_total = sum(v for n in nodes for v in n.get("gpu_total_memory", {}).values())

    # A job with no GPUs allocated is not missing a GPU reading -- it has none to
    # miss. A job that *was* allocated GPUs and still has no samples is a gap.
    if gpus == 0:
        absent = Measure.not_applicable("CPU-only job")
        found["GPU%"], found["GMEM%"] = absent, absent
        return JobMetrics(found, source=SOURCE)

    no_gpu_data = Measure.unmeasured("no GPU samples in the stored summary")
    found["GPU%"] = (Measure.reading(round(sum(utils) / len(utils)))
                     if utils else no_gpu_data)
    found["GMEM%"] = (Measure.reading(round(100 * gpu_used / gpu_total))
                      if gpu_total else no_gpu_data)
    return JobMetrics(found, source=SOURCE)


def jobstats_detail(stats: dict) -> List[DetailRow]:
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
