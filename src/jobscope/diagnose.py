"""Advisory GPU diagnosis tags derived from the DCGM utilization metrics.

Heuristic and window-averaged, so a multi-phase job blurs; descriptive, not a
verdict. Several legitimately low-utilization workloads (inference, sampling,
sparse HPC, data prep) read as "underfed" without being wasteful. Thresholds are
in percent (SM_ACT/OCC/TENSOR/DRAM range 0-100).
"""

from typing import Optional

IDLE = 5
UNDERFED = 15
BUSY = 40
LOW_OCCUPANCY = 20
MEMORY_BOUND = 40
TENSOR = 1

LEGEND = """\
DIAG: advisory tag(s) from the utilization metrics (heuristic, not a verdict;
window-averaged, so multi-phase jobs blur). Several legitimately-low-util
workloads (inference, sampling, sparse HPC, data prep) read as "underfed".
  idle       SM_ACT < %d%%                    GPU essentially never ran a kernel
  underfed   SM_ACT < %d%%                   resident but cores mostly idle (host/input bound)
  low-occ    SM_ACT >= %d%%, OCC < %d%%        SMs busy but warp slots underfilled
  mem-bound  DRAM >= %d%% and >= SM_ACT       HBM bandwidth is the limiter
  no-tensor  SM_ACT >= %d%%, TENSOR < %d%%       tensor cores idle (e.g. fp32 ML)
  ok         SM_ACT >= %d%%, none of above    healthy compute use
  short      ran under --min-runtime         averages are sampling noise""" % (
    IDLE, UNDERFED, BUSY, LOW_OCCUPANCY, MEMORY_BOUND, BUSY, TENSOR, BUSY)


def diagnose(smact: Optional[float], occ: Optional[float], tensor: Optional[float],
             dram: Optional[float], duration: Optional[int],
             min_runtime: Optional[int]) -> str:
    """One or more advisory tags from SM_ACT%/OCC%/TENSOR%/DRAM% (each 0-100 or None).

    Comma-joined; ``short`` if the job ran under ``min_runtime``; ``-`` when there
    is no clear signal or no data. Tags: idle / underfed / low-occ / mem-bound /
    no-tensor / ok.
    """
    if duration is not None and min_runtime and duration < min_runtime:
        return "short"
    if smact is None:
        return "-"
    if smact < IDLE:
        return "idle"
    if smact < UNDERFED:
        return "underfed"
    tags = []
    if occ is not None and smact >= BUSY and occ < LOW_OCCUPANCY:
        tags.append("low-occ")
    if dram is not None and dram >= MEMORY_BOUND and dram >= smact:
        tags.append("mem-bound")
    if tensor is not None and smact >= BUSY and tensor < TENSOR:
        tags.append("no-tensor")
    if not tags and smact >= BUSY:
        tags.append("ok")
    return ",".join(tags) if tags else "-"


def diagnose_dcgm(values: dict, duration: Optional[int],
                  min_runtime: Optional[int]) -> str:
    """Diagnose one GPU (or the overall dict) from a ``{header: value}`` DCGM dict."""
    return diagnose(values.get("SM_ACT%"), values.get("OCC%"), values.get("TENSOR%"),
                    values.get("DRAM%"), duration, min_runtime)
