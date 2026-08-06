"""What Slurm's own accounting knows about a job -- an independent cross-check.

jobscope reads utilization from Prometheus, and optionally from the jobstats summary.
Slurm is a third source that needs neither: ``jobacct_gather`` records CPU time and
peak RSS for every job, and where ``AccountingStorageTRES`` includes
``gres/gpuutil`` it records GPU utilization too. Nothing here is required for a
report -- it exists so a site can ask "do my exporters agree with my scheduler?"
and get an answer rather than a shrug.

Deliberately *not* a fallback. If Prometheus is thin the report says so; silently
substituting Slurm's numbers would make the columns unattributable, and the two do
not always agree -- see the module tests.

There was a ``probe --validate`` that printed one job's utilization as each of the three
sources measured it. It is gone: Slurm's ``gres/gpuutil`` turned out to be a single point
reading rather than a mean -- ``TRESUsageInTot``, ``InAve``, ``InMax`` and ``InMin`` all
return the same number -- so the GPU row was comparing a snapshot against a whole-run
average, and its disagreement tracked how fast the metric was moving rather than anything
about accuracy. ``--gpu-source nvml`` against ``--gpu-source dcgm`` is the comparison that
answers the question it was reaching for, and ``--verify``'s SWING column says when even
that is aliased. What this module still supplies is ``[host] source = slurm``, where
``TotalCPU``/``CPUTime`` *is* a genuine whole-job ratio.

Four things the sacct data model makes easy to get wrong, each of which produced a
wrong number before it was pinned down:

* **Read the allocation row, not ``.batch``.** On a multi-node job ``.batch`` is
  just the batch script on the head node: for one 16-node job here it reports
  ``TotalCPU=18:35`` against the job's real ``4580-00:04:33``, which as a CPU%
  is 0.00 instead of 99.2. The suffix-less allocation row is already rolled up
  across steps, so it is both correct and simpler.
* **Except ``MaxRSS``, which is only on steps.** It never appears on the allocation
  row, so it has to be maxed across them -- and ``ReqMem``, its denominator, only
  appears on the allocation row. That pair is why this needs a query without
  ``-X``, which is also why it is opt-in: dropping ``-X`` roughly triples the rows.
* **``gres/gpuutil`` is a sum across the job's GPUs, not a mean.** A 4-GPU job here
  reports 382, which is ~95.5% each. Comparing that to jobscope's mean GPU% without
  dividing would be off by the GPU count -- invisible on a 1-GPU job, which is
  exactly how such a bug survives.
* **Durations carry days, up to four digits** (``4580-00:04:33``), and short ones
  drop the hours entirely (``40:30.266`` is 40 minutes, not 40 hours).
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .slurm import run_capture

# The fields one sacct call needs to answer everything here. Order matters: it is
# positional on the way back out.
FIELDS = ("JobID", "TotalCPU", "UserCPU", "SystemCPU", "CPUTime", "MaxRSS", "ReqMem",
          "NNodes", "NCPUS", "AllocTRES", "TRESUsageInTot", "ConsumedEnergyRaw")

# Steps that are bookkeeping rather than work. `.extern` is the container Slurm
# wraps around the allocation; it reports zero CPU and would drag a max down.
SKIP_STEPS = (".extern",)

_SIZE_UNITS = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4, "P": 1024 ** 5}


def parse_duration(text: str) -> Optional[float]:
    """Seconds from a Slurm duration, or None when it is blank or unparseable.

    Handles every shape sacct emits: ``D-HH:MM:SS``, ``HH:MM:SS``, ``MM:SS.mmm``.
    The last is the trap -- ``40:30.266`` is forty *minutes*, so a parser that
    assumes the leftmost field is hours reads it as 40 hours.
    """
    text = str(text or "").strip()
    if not text:
        return None
    days, _, rest = text.partition("-")
    if not rest:
        rest, days = days, "0"
    parts = rest.split(":")
    if not 1 <= len(parts) <= 3:
        return None
    try:
        # Right-aligned: the rightmost field is always seconds, so a two-field
        # value is MM:SS and a one-field value is bare seconds.
        values = [float(p) for p in parts]
        while len(values) < 3:
            values.insert(0, 0.0)
        return int(days) * 86400 + values[0] * 3600 + values[1] * 60 + values[2]
    except ValueError:
        return None


def parse_size(text: str) -> Optional[float]:
    """Bytes from a Slurm size like ``1746956K`` or ``64G``; None when blank.

    A bare number is taken as bytes. Slurm is not consistent about the suffix --
    the same quantity appears as ``168857588`` in one reducer's TRES string and
    ``168857588K`` in another's -- which is why the sized fields here come from
    dedicated columns (``MaxRSS``, ``ReqMem``) rather than from TRES.
    """
    text = str(text or "").strip()
    if not text:
        return None
    match = re.match(r"^([0-9.]+)\s*([KMGTP]?)", text, re.IGNORECASE)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value * _SIZE_UNITS.get(match.group(2).upper(), 1)


def parse_tres(text: str) -> Dict[str, str]:
    """``cpu=00:40:30,gres/gpuutil=382,...`` to a dict. Empty for a blank field."""
    found = {}
    for item in str(text or "").split(","):
        key, sep, value = item.partition("=")
        if sep:
            found[key.strip()] = value.strip()
    return found


def gpus_from_tres(alloc_tres: str) -> int:
    """GPU count from an AllocTRES string.

    Reads the untyped ``gres/gpu=N``, which Slurm emits alongside the typed
    ``gres/gpu:nvidia_h200=N`` form. Falls back to the typed one so a site that
    emits only that still gets a count.
    """
    match = re.search(r"gres/gpu=(\d+)", str(alloc_tres))
    if match:
        return int(match.group(1))
    typed = re.findall(r"gres/gpu:[^=,]+=(\d+)", str(alloc_tres))
    return sum(int(n) for n in typed) if typed else 0


@dataclass(frozen=True)
class SlurmMetrics:
    """One job's utilization as Slurm itself accounts it.

    Percentages are already normalised the way jobscope reports them -- per GPU for
    the GPU figures, against the allocation for CPU and memory -- so they can be
    compared with a Prometheus-derived number directly. ``None`` means Slurm did
    not record it, which is a different thing from zero and is kept distinct all
    the way to the display.
    """

    jobid: str
    gpus: int
    cpu_pct: Optional[float] = None
    cpu_user_pct: Optional[float] = None
    cpu_sys_pct: Optional[float] = None
    mem_pct: Optional[float] = None
    gpu_pct: Optional[float] = None            # gres/gpuutil, divided by GPU count
    gpu_mem_bytes: Optional[float] = None      # gres/gpumem, divided by GPU count
    disk_bytes: Optional[float] = None
    energy_j: Optional[float] = None
    used_mem_bytes: Optional[float] = None     # RSS summed across tasks -- MEM%'s numerator
    max_rss_bytes: Optional[float] = None      # peak of one task; for display only
    req_mem_bytes: Optional[float] = None
    total_cpu_s: Optional[float] = None
    cpu_time_s: Optional[float] = None


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """``100 * a / b``, or None when either side is missing or the divisor is zero.

    None rather than 0: a job whose denominator Slurm never recorded has *unknown*
    utilization, and reporting 0 would make it look idle. That distinction is the
    whole point of the cross-check.
    """
    if not numerator and numerator != 0:
        return None
    if not denominator:
        return None
    return 100.0 * numerator / denominator


def _from_rows(jobid: str, alloc: List[str], steps: List[List[str]]) -> SlurmMetrics:
    """Build one job's metrics from its allocation row and its step rows."""
    idx = {name: i for i, name in enumerate(FIELDS)}

    def field(row, name):
        return row[idx[name]] if row and idx[name] < len(row) else ""

    gpus = gpus_from_tres(field(alloc, "AllocTRES"))
    cpu_time = parse_duration(field(alloc, "CPUTime"))
    total_cpu = parse_duration(field(alloc, "TotalCPU"))
    req_mem = parse_size(field(alloc, "ReqMem"))

    # MaxRSS is the peak of a *single task*, kept only for the detail line. It is
    # the wrong numerator for MEM%: on a 16-node job here it reads 0.4G against a
    # 500G allocation (0.1%) where the job really held 161G (32%), because 15 of
    # the 16 nodes are simply not in it.
    rss_values = [parse_size(field(row, "MaxRSS")) for row in steps]
    max_rss = max([v for v in rss_values if v is not None], default=None)

    # TRES usage lives on steps, and which step holds the real work varies -- the
    # batch script on a single-node job, a numbered srun step otherwise -- so take
    # the largest rather than guessing at the step name. `InTot` rather than
    # `InAve`: `mem` is then the RSS summed across tasks, which is what the cgroup
    # exporter measures and so what these numbers are being compared against.
    # `InAve` would also be ambiguous -- Slurm prints the same quantity as
    # `mem=168857588` there and `mem=168857588K` here, a 1024x trap.
    used_mem = gpu_util = gpu_mem = disk = None
    for row in steps:
        tres = parse_tres(field(row, "TRESUsageInTot"))
        for key, current in (("mem", used_mem), ("gres/gpuutil", gpu_util),
                             ("gres/gpumem", gpu_mem), ("fs/disk", disk)):
            raw = tres.get(key)
            if raw is None:
                continue
            value = parse_size(raw)
            if value is None:
                continue
            best = value if current is None else max(current, value)
            if key == "mem":
                used_mem = best
            elif key == "gres/gpuutil":
                gpu_util = best
            elif key == "gres/gpumem":
                gpu_mem = best
            else:
                disk = best

    energy = None
    for row in [alloc] + steps:
        raw = field(row, "ConsumedEnergyRaw").strip()
        if raw and raw not in ("0",):
            try:
                energy = max(energy or 0.0, float(raw))
            except ValueError:
                pass

    return SlurmMetrics(
        jobid=jobid,
        gpus=gpus,
        cpu_pct=_ratio(total_cpu, cpu_time),
        cpu_user_pct=_ratio(parse_duration(field(alloc, "UserCPU")), cpu_time),
        cpu_sys_pct=_ratio(parse_duration(field(alloc, "SystemCPU")), cpu_time),
        mem_pct=_ratio(used_mem, req_mem),
        # Per GPU: Slurm sums gpuutil across the job's cards, so a 4-GPU job at
        # ~95% each reports 382. jobscope's GPU% is a mean, so divide to compare.
        gpu_pct=(gpu_util / gpus) if (gpu_util is not None and gpus) else None,
        gpu_mem_bytes=(gpu_mem / gpus) if (gpu_mem is not None and gpus) else None,
        disk_bytes=disk,
        energy_j=energy,
        used_mem_bytes=used_mem,
        max_rss_bytes=max_rss,
        req_mem_bytes=req_mem,
        total_cpu_s=total_cpu,
        cpu_time_s=cpu_time,
    )


def collect(jobids: List[str], timeout: Optional[float]) -> Dict[str, SlurmMetrics]:
    """``{jobid: SlurmMetrics}`` from one sacct call.

    Without ``-X``, so the step rows come back too -- that is the cost, and the
    reason callers opt in. Rows are grouped by the part of ``JobID`` before the
    first dot, which keeps array elements (``123_4``) as separate jobs while
    folding their steps (``123_4.batch``) into them.
    """
    if not jobids:
        return {}
    out = run_capture(["sacct", "-j", ",".join(str(j) for j in jobids),
                       "--noheader", "-P", "-o", ",".join(FIELDS)],
                      timeout, "sacct accounting query", soft=True)
    if not out:
        return {}

    alloc_rows: Dict[str, List[str]] = {}
    step_rows: Dict[str, List[List[str]]] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        row = line.split("|")
        base, dot, step = row[0].partition(".")
        if dot and ("." + step) in SKIP_STEPS:
            continue
        if dot:
            step_rows.setdefault(base, []).append(row)
        else:
            alloc_rows[base] = row
    return {jobid: _from_rows(jobid, row, step_rows.get(jobid, []))
            for jobid, row in alloc_rows.items()}


def for_job(jobid: str, timeout: Optional[float]) -> Optional[SlurmMetrics]:
    """One job's Slurm accounting, or None when sacct knows nothing about it."""
    return collect([jobid], timeout).get(jobid)
