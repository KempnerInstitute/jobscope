"""Rendering of the summary, detail, per-GPU DCGM, and time-series views.

Output layout (spacing, context lines, and above all the CSV shape) is kept
stable: the CSV emitted here is what ``jobscope plot`` parses.
"""

import csv
import re
import shutil
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

from .blob import GIB, blob_capacity, blob_detail, blob_metrics
from .config import DEFAULT_THRESHOLDS, Thresholds
from .dcgm import (
    ALL_SPECS,
    DCGM_BLOB_HEADERS,
    DCGM_HEADERS,
    DEFAULT_SPECS,
    DERIVED_COLUMNS,
    DESCRIPTIONS,
    MetricSpec,
    applicable_derived,
    columns_for,
    discover_gpus,
    format_by_header,
    format_number,
    gpu_minor_key,
    values_by_key,
)
from .diagnose import LEGEND, diagnose_dcgm
from .errors import JobscopeError
from .live import Gpu, LiveJob, build_columns, job_sort_key, range_window
from .prometheus import PrometheusClient
from .sacct import JobRecord, Selection, format_window


@dataclass(frozen=True)
class Column:
    """A rendered column: header, str.format spec, group, and (detail) row index."""

    header: str
    fmt: str
    group: str
    index: Optional[int] = None


# One row per job, and the same set for every per-job view -- summary, dcgm and
# live -- so a job reads identically whether it has finished or is still running:
#
#   JOBID USER STATE NODE CPU% MEM% #GPU GPU% GMEM% SM_ACT% OCC% TENSOR% DRAM% POWER_W RUNTIME
#
# NODE is the node count (a name would truncate on a multi-node job and the row is
# already per-job, not per-node); #GPU is the allocated GPU count. The blob group
# is shown by every view, since CPU% next to SM_ACT% is the comparison that tells
# you whether a GPU job is actually CPU-bound -- previously no single view had both.
SUMMARY_COLUMNS: List[Column] = [
    Column("JOBID", "{:<12}", "id"),
    Column("USER", "{:<12}", "id"),
    Column("STATE", "{:<9}", "id"),
    Column("NODE", "{:<5}", "blob"),
    Column("CPU%", "{:<6}", "cpu"),
    Column("MEM%", "{:<6}", "cpu"),
    Column("#GPU", "{:<5}", "gpu"),
    Column("GPU%", "{:<6}", "gpu"),
    Column("GMEM%", "{:<7}", "gpu"),
    Column("SM_ACT%", "{:<8}", "dcgm"),
    Column("OCC%", "{:<7}", "dcgm"),
    Column("TENSOR%", "{:<8}", "dcgm"),
    Column("DRAM%", "{:<7}", "dcgm"),
    Column("POWER_W", "{:<8}", "dcgm"),
    Column("RUNTIME", "{:<12}", "id"),
    Column("DIAG", "{:<22}", "diag"),
]


def summary_columns(specs: Optional[List[MetricSpec]] = None) -> List[Column]:
    """:data:`SUMMARY_COLUMNS` with its DCGM block taken from ``specs``.

    The identity and blob columns are fixed; only the profiling block varies, which
    is what lets `dcgm --ext` widen the table without becoming a different view.
    Blob-backed metrics are dropped from the block -- GPU% and the GMEM columns are
    already rendered from the blob, and one number deserves one column.
    """
    if specs is None:
        return list(SUMMARY_COLUMNS)
    block = [Column(header, "{:<%d}" % max(7, len(header) + 1), "dcgm")
             for _key, header, _dec in columns_for(specs)
             if header not in DCGM_BLOB_HEADERS]
    out = []
    for col in SUMMARY_COLUMNS:
        if col.group == "dcgm":
            out.extend(block)
            block = []          # splice the whole block in at the first dcgm slot
        else:
            out.append(col)
    return out


DETAIL_COLUMNS: List[Column] = [
    Column("NODE", "{:<16}", "id", 0),
    Column("GPU", "{:<4}", "gpu", 1),
    Column("CPU%", "{:<7}", "cpu", 2),
    Column("CPU-MEM", "{:<16}", "cpu", 3),
    Column("GPU%", "{:<7}", "gpu", 4),
    Column("GPU-MEM", "{:<16}", "gpu", 5),
    Column("GMEM%", "{:<7}", "gpu", 6),
    Column("SM_ACT%", "{:<8}", "dcgm", 7),
    Column("OCC%", "{:<7}", "dcgm", 8),
    Column("TENSOR%", "{:<8}", "dcgm", 9),
    Column("DRAM%", "{:<7}", "dcgm", 10),
    Column("POWER_W", "{:<8}", "dcgm", 11),
    Column("DIAG", "{:<22}", "diag", 12),
]

# Positions in a detail row, named rather than repeated as literals.
_NODE_INDEX = 0
_GPU_INDEX = 1

DETAIL_HEADER: Tuple[str, ...] = (
    "NODE", "GPU", "CPU%", "CPU-MEM", "GPU%", "GPU-MEM", "GMEM%",
    "SM_ACT%", "OCC%", "TENSOR%", "DRAM%", "POWER_W", "DIAG")

SUMMARY_DESCRIPTIONS: List[Tuple[str, str, str]] = [
    ("CPU%", "blob (cgroup CPU-seconds)",
     "Average CPU-core utilization: 100 x CPU-seconds used / (elapsed x allocated cores). "
     "100% means every allocated core was busy for the whole job."),
    ("MEM%", "blob (cgroup RSS)",
     "Peak host (CPU) memory used / memory allocated, as a percent."),
    ("GPU%", "blob (nvidia_gpu_duty_cycle)",
     "GPU duty cycle averaged over the job's GPUs: fraction of time at least one kernel ran. "
     "Coarse: says the GPU was occupied in time, not how hard. Use SM_ACT%/OCC% (gpu view) for that."),
    ("GMEM%", "blob (nvidia_gpu_memory_used)",
     "Peak GPU memory used / total, summed over the job's GPUs. A high-water mark, not a time-average."),
    ("SM_ACT%", "DCGM_FI_PROF_SM_ACTIVE (gpu view)",
     "Fraction of time at least one warp was resident on an SM, averaged across all SMs. Low while "
     "GPU% is high means the GPU was barely loaded (parked / underfed)."),
    ("OCC%", "DCGM_FI_PROF_SM_OCCUPANCY (gpu view)",
     "SM occupancy: fraction of warp slots filled, averaged over SMs and time. Low occupancy means "
     "kernels under-fill the GPU: small launches, or register / shared-memory limits."),
    ("TENSOR%", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE (gpu view)",
     "Fraction of time the tensor-core pipe was active. High only for mixed-precision matmul work "
     "(fp16/bf16/tf32); ~0 means the tensor cores sat idle."),
    ("DRAM%", "DCGM_FI_PROF_DRAM_ACTIVE (gpu view)",
     "Fraction of time the device-memory (HBM) interface was busy. High while SM_ACT% is low suggests "
     "the job is memory-bound."),
    ("POWER_W", "DCGM_FI_DEV_POWER_USAGE (gpu view)",
     "Mean board power draw over the run, in watts. Near-idle watts mean the GPU was not really working."),
]


# SGR codes for the utilization grades. Raw escapes rather than rich, because the
# report path is the common one and should not import a rendering library to print
# a table; plot pays for rich because it needs it.
_SGR = {"red": "\033[31m", "yellow": "\033[33m", "green": "\033[32m"}
_RESET = "\033[0m"
# Strips the above, so a rule can be measured against the characters a reader sees
# rather than the escape bytes carrying the colour.
_ESC_RE = re.compile(r"\033\[[0-9;]*m")


@dataclass
class RenderOptions:
    """Flags shared by the rendering functions."""

    view: str = "all"
    show_dcgm: bool = False
    diagnose: bool = False
    csv: bool = False
    header: bool = True
    min_runtime: int = 180
    # Weight the mean by allocated resource-time (GPU-hours, core-hours) instead
    # of by GPU count. Valid only where each job's value already covers its whole
    # runtime -- a finished job's blob, or running --avg. On an instantaneous
    # running snapshot every value is the same moment, so scaling one by two days
    # of elapsed time would claim that instant represents those two days.
    time_weighted: bool = False
    # --per-gpu only: report just this node's GPUs. The per-job table's NODE column
    # is a count, so there is no name there to match against.
    nodename: Optional[str] = None
    # Show the efficiency bars section. On by default: it is the fastest read in
    # the block, and behind a flag it was rarely seen. --no-plot switches it off.
    plot_avgeff: bool = True
    # --ts only: emit just the last N seconds of each job's series, narrowing the
    # range queries rather than filtering rows afterwards.
    window: Optional[int] = None
    # Tint %-metric cells by their threshold band. Off unless the caller has
    # established that the destination is a terminal that wants colour.
    color: bool = False
    thresholds: Optional["Thresholds"] = None


def no_such_node(nodename: str, seen) -> JobscopeError:
    """The error for a ``--nodename`` that matched nothing.

    Naming the nodes the selection *did* touch is the whole value of it: an empty
    report reads as an idle node rather than a typo. Shared by every view that takes
    the filter -- the per-GPU table and both ``--ts`` emitters -- so a mistyped name
    gets the same answer whichever one you were running.
    """
    return JobscopeError("no rows for node %r in this selection; it ran on: %s"
                         % (nodename, ", ".join(sorted(seen)) or "(none)"))


def cell_value(cell) -> Optional[float]:
    """The number in a rendered cell, or None when there is not one.

    The trailing ``%`` matters: the detail view writes "11.4%" where the summary writes
    "11", and a bare ``float()`` rejects the former. One parser for grading and for
    charting, so a cell that gets a colour is a cell that gets a bar.
    """
    try:
        return float(str(cell).rstrip("%"))
    except (TypeError, ValueError):      # "-", "", a hostname, "76.1GB/1400GB"
        return None


def cell_band(options: "RenderOptions", header: str, cell) -> str:
    """The grade for a rendered cell, or ``""`` when it is not a graded metric.

    Shared by the per-job table and ``--per-gpu`` so the two cannot disagree about a
    colour, the same reason ``plot`` calls ``Thresholds.grade`` rather than keeping its
    own copy.

    The trailing ``%`` matters: the detail view writes its cells as "11.4%" where the
    summary writes "11", and a bare ``float()`` rejects the former -- which is why
    those columns were silently the only untinted ones.
    """
    if not options.color or options.thresholds is None:
        return ""
    value = cell_value(cell)
    return "" if value is None else options.thresholds.grade(header, value)


BAR_WIDTH = 34


def bar_lines(items, indent: str = "  ") -> List[str]:
    """Horizontal bars from ``(label, percent, band)`` triples.

    One primitive for the per-job summary and the per-GPU detail charts, so they look
    identical rather than merely similar. A nonzero value always draws at least one
    block and reads "<1%" rather than "0%", because an empty bar beside a "0%" and a
    filled one beside it are each self-contradictory.
    """
    if not items:
        return []
    label_width = max(len(label) for label, _v, _b in items)
    out = []
    for label, value, band in items:
        filled = max(0, min(BAR_WIDTH, int(round(value / 100.0 * BAR_WIDTH))))
        if value > 0 and filled == 0:
            filled = 1
        run = "\u2588" * filled
        out.append("%s%*s  %s%s  %4s" % (
            indent, label_width, label, tint(run, band) if run else run,
            "\u2591" * (BAR_WIDTH - filled),
            "<1%" if 0 < value < 0.5 else "%d%%" % round(value)))
    return out


# A chart block is about 55 characters, so four columns need ~229 -- nearly twice the
# job table's width. How many actually fit is a property of the terminal, not of the
# data, so it is measured rather than chosen.
MAX_CHART_COLUMNS = 4
# Used when the destination is not a terminal, so redirected output does not change
# shape with whatever $COLUMNS happened to be. Two 55-wide cells and a gap.
PIPED_CHART_WIDTH = 116


def terminal_width(out, default: int = PIPED_CHART_WIDTH) -> int:
    """The width to lay out for: the terminal's, or a fixed default off a terminal."""
    if not getattr(out, "isatty", lambda: False)():
        return default
    return shutil.get_terminal_size((default, 24)).columns


def in_columns(blocks: List[List[str]], columns: Optional[int] = None,
               gap: int = 3, available: int = PIPED_CHART_WIDTH) -> List[str]:
    """Pack equal-shaped blocks of lines side by side, in reading order.

    ``columns`` defaults to as many as fit in ``available``, capped at
    :data:`MAX_CHART_COLUMNS`: a wide terminal gets four, the width the rest of the
    report targets gets two, and an 80-column one gets a single column rather than
    wrapped nonsense.

    Widths are measured with the escapes stripped, or a coloured block would be padded
    by the length of its SGR bytes and push its neighbour out of line. Blocks of unequal
    height are padded with blanks, since a metric absent from one group leaves it a line
    short of the others.
    """
    blocks = [b for b in blocks if b]
    if not blocks:
        return []
    width = max(len(_ESC_RE.sub("", line)) for block in blocks for line in block)
    if columns is None:
        columns = max(1, min(MAX_CHART_COLUMNS, (available + gap) // (width + gap)))
    if columns < 2 or len(blocks) < 2:
        return [line for block in blocks for line in block]
    out = []
    for start in range(0, len(blocks), columns):
        row = blocks[start:start + columns]
        for index in range(max(len(b) for b in row)):
            cells = []
            for block in row:
                line = block[index] if index < len(block) else ""
                cells.append(line + " " * (width - len(_ESC_RE.sub("", line))))
            out.append((" " * gap).join(cells).rstrip())
    return out


def tint(text: str, band: str) -> str:
    """``text`` wrapped in ``band``'s colour, or unchanged when there is none.

    Callers pass the *padded* cell: inserting the escapes first would make str.format
    count them toward the column width and shift every later column.
    """
    return _SGR[band] + text + _RESET if band else text


def fmt_context(label: str, value: str) -> str:
    """A '  Label:     value' context line."""
    return "  %-11s%s" % (label + ":", value)


def cols_for(columns: List[Column], view: str, dcgm: bool = False,
             diagnose: bool = False) -> List[Column]:
    """The columns to show for the chosen view.

    ``all`` (the default) shows everything; ``--cpu`` and ``--gpu`` narrow it to one
    resource. ``id``/``blob`` columns identify the row and appear in every view. The
    DCGM block needs Prometheus, so it belongs to the views that carry GPU columns,
    and DIAG rides along with it under --diagnose.
    """
    gpu_views = ("all", "gpu")
    out = []
    for col in columns:
        group = col.group
        if (group in ("id", "blob")
                or (group == "cpu" and view in ("all", "cpu"))
                or (group == "gpu" and view in gpu_views)
                or (group == "dcgm" and dcgm and view in gpu_views)
                or (group == "diag" and diagnose and dcgm and view in gpu_views)):
            out.append(col)
    return out


def context_pairs(selection: Selection, desc: str,
                  records: Dict[str, JobRecord]) -> List[Tuple[str, str]]:
    """Context lines for the header block.

    With explicit JOBIDs the -u/-A/-p filters are bypassed, so show the jobs'
    actual owner(s) rather than the (misleading) default user, and drop the filter
    lines.
    """
    if selection.jobids:
        owners = sorted({r.user for r in records.values() if r.user})
        user_val = ", ".join(owners) if owners else "(explicit job IDs)"
        return [("User", user_val), ("Select", desc)]
    # -a/--all-users leaves `user` unset, so say so rather than printing None.
    pairs = [("User", selection.user or "(all users)")]
    if selection.account:
        pairs.append(("Account", selection.account))
    if selection.partition:
        pairs.append(("Partition", selection.partition))
    pairs.append(("Select", desc))
    if selection.days is not None or selection.lastn is not None:
        # The dates behind "last 1 day" or "last 20 jobs", which the Select line does
        # not show: a -D window is computed from the clock, and a bare -N reaches back
        # the default lookback, so a reader could not otherwise tell what was scanned.
        # Not for an explicit -S/-E, where the Select line already is the window.
        pairs.append(("Window", format_window(*selection.window())))
    return pairs


def extend_detail_row(row, per_gpu, duration=None, min_runtime=None, diagnose_on=False):
    """Append the DCGM cells and (optionally) the DIAG cell to a blob_detail row."""
    values = per_gpu.get((row[0], str(row[1])), {})
    out = tuple(row) + tuple(format_by_header(h, values.get(h)) for h in DCGM_HEADERS)
    if diagnose_on:
        out = out + (diagnose_dcgm(values, duration, min_runtime),)
    return out


# What each graded column is a percentage *of*, as
# (label, unit, scale, pooled-row label, CSV label, weight key). CPU% is a share of
# allocated cores and MEM% of allocated host memory; everything else -- GPU%, GMEM%
# and every DCGM profiling column -- is a share of time on the allocated GPUs.
_RESOURCES = {
    "CPU%": ("Core-hours", "Cores", "cpu", "cpu"),
    "MEM%": ("GB-hours", "GB", "mem", "mem"),
    None: ("GPU-hours", "GPUs", "GPU", "gpu"),
}


_BLOB_HEADERS = ("CPU%", "MEM%", "GPU%", "GMEM%")

# The measures the Worst rows rank by, in print order. Four rather than every graded
# column: these say distinct things -- duty cycle, SM residency, board watts, and
# the host -- while the DCGM catalog would add a dozen near-duplicates.
WORST_METRICS = ("GPU%", "SM_ACT%", "POWER_W", "CPU%")
# The two combined rankings: the distinct *resources*, then every measure.
COMBINED_2 = ("GPU%", "CPU%")
COMBINED_4 = WORST_METRICS
# Short tags for a combined row's component shares, e.g. "35%gpu+24%cpu".
_SHARE_TAG = {"GPU%": "gpu", "SM_ACT%": "sm", "POWER_W": "pw", "CPU%": "cpu"}


# Row-label form of each metric name. Explicit rather than derived, because the
# label plus "Worst " and ":" has to fit the 12-character label column: "Worst
# SM_ACT:" is 13 and shifts the whole row one place right.
_WORST_SLUG = {"GPU%": "GPU", "SM_ACT%": "SM", "POWER_W": "POWER", "CPU%": "CPU"}


# A job running longer than this, and still on a Worst row, is the expensive kind of
# waste: a short bad job costs little, whereas hours of idle hardware do not come
# back. Its entry is printed red.
LONG_RUNNING = 3 * 3600


def _worst_groups(entries):
    """Group ``(user, text, long_running)`` entries by user, keeping rank order.

    Rank order matters: the first user to appear is the one with the worst job, so
    the groups stay sorted by severity rather than alphabetically.
    """
    groups, order = {}, []
    for user, text, long_running in entries:
        if user not in groups:
            groups[user] = []
            order.append(user)
        groups[user].append((text, long_running))
    return [(user, groups[user]) for user in order]


def _combined_cell(jid: str, values, tallies) -> str:
    """One job on a combined row: its value in each of the row's metrics.

    The values, not the waste shares that decide the order. A share written "12%gpu"
    reads exactly like a utilization of 12%, which is the opposite of what puts a job
    on the row -- every value here is *under* its cutoff. Showing the values says why
    the job qualified; the order still says how much it wasted.
    """
    parts = []
    for header, value in values:
        if value is None:
            continue
        unit = "W" if tallies[header].value_unit == "W" else ""
        parts.append("%s%d%s" % (_SHARE_TAG[header], round(value), unit))
    return "%s %s" % (jid, " ".join(parts))


def _worst_slug(header: str) -> str:
    """Row-label form of a metric name, e.g. ``SM_ACT%`` -> ``SM``."""
    return _WORST_SLUG.get(header, header.rstrip("%"))


def _blob_value(metrics, header: str) -> Optional[float]:
    """``header``'s value from a :func:`blob_metrics` tuple, or None.

    None both for a header the blob does not carry (a DCGM column) and for one it
    carries without a measurement, which the caller treats the same way: look to
    Prometheus, then give up rather than invent a zero.
    """
    if metrics is None or header not in _BLOB_HEADERS:
        return None
    return metrics[_BLOB_HEADERS.index(header)]


def _resource_of(header: str, hours: bool):
    """``EfficiencyTally`` arguments for ``header``: the resource it measures.

    ``hours`` selects the resource-time form (a finished job, or ``--avg``) over
    the bare-count form used by the instantaneous running view.
    """
    hourly, counted, name, key = _RESOURCES.get(header, _RESOURCES[None])
    if key == "mem":
        # Weights are byte-seconds; divide to GB-hours (or GB) and let _amount
        # promote to TB when the figure gets long.
        scale = GIB * 3600.0 if hours else float(GIB)
        unit = "GBh" if hours else "GB"
    else:
        scale = 3600.0 if hours else 1.0
        unit = "h" if hours else ""
    return (hourly if hours else counted, unit, scale,
            "Used/%s-hr:" % name if hours else "Used/%s:" % name,
            "UsedPer%sHour" % name.upper() if hours else "UsedPer%s" % name.upper(),
            key)


class WorstJob(NamedTuple):
    """One entry on a Worst row. ``wasted`` is what the ranking sorts by."""

    wasted: float
    jobid: str
    user: str
    weight: float
    value: float
    runtime: str                    # display form, as the job's own row shows it
    duration: Optional[int]         # seconds, for the long-running test


class EfficiencyTally:
    """Where a selection's resource-time went, banded by the utilization thresholds.

    A mean is a poor summary of this data: utilization is bimodal (jobs cluster at
    either end), so the average lands in a range where few jobs live. Measured over
    one day on one partition, GPU% averaged 82 per job while 5% of the jobs held
    64% of the GPU-hours at under 25% -- two of them sat on 285 GPU-hours at 0%.
    Counting jobs *and* the resource-time they held, per band, states that directly.

    Fed one job at a time so the renderer stays streaming; ``worst`` is truncated
    on every insert, so nothing here grows with the selection.
    """

    WORST = 3

    def __init__(self, header: str, thresholds: "Thresholds", label: str,
                 unit: str, scale: float = 1.0, row: str = "Used/GPU:",
                 csv_row: str = "UsedPerGPU", weight_key: str = "gpu",
                 absolute: bool = False, value_unit: str = "%") -> None:
        self.header = header            # the column being banded, "GPU%", "SM_ACT%", ...
        self.thresholds = thresholds
        self.label = label              # "GPU-hours", "GPUs", "Core-hours", ...
        self.unit = unit                # "h" for resource-hours, "" for a count
        self.scale = scale              # seconds -> hours, or 1 for a bare count
        # Which of _weights()'s entries this metric is a percentage *of*: cores for
        # CPU%, host bytes for MEM%, GPUs for GPU% and every DCGM column. Taking it
        # from the same source as the pooled row is what stops the two disagreeing.
        self.weight_key = weight_key
        # What the pooled row is called. Kept to 12 characters, the width of the
        # JOBID column: a longer label shifts every cell in the row one right.
        self.row = row
        self.csv_row = csv_row
        # An absolute metric is not a percentage of its resource -- POWER_W is watts
        # -- so "used" has no meaning for it and it earns no stats-table row. It is
        # still banded, and still ranks jobs by waste; see waste_of.
        self.absolute = absolute
        self.value_unit = value_unit    # "%" or "W", for the @73W in a worst row
        self.bands = {band: [0, 0.0] for band in ("red", "yellow", "green")}
        self.used = 0.0                 # resource-time actually utilized
        self.total = 0.0                # resource-time allocated
        self.waste_total = 0.0          # resource-time wasted, the combined rows' denominator
        self.worst: List["WorstJob"] = []

    def add(self, jobid: str, user: str, value: Optional[float], weight: float,
            runtime: str = "-", duration: Optional[int] = None) -> None:
        band = self.thresholds.grade(self.header, value)
        if not band or weight <= 0:
            # Ungraded (no measurement) or unweighable: counting it would either
            # invent a utilization or give it no resource to account for.
            return
        self.bands[band][0] += 1
        self.bands[band][1] += weight
        wasted = self.waste_of(value, weight)
        # Used is whatever was not wasted, for both kinds of metric. For a percentage
        # that is (value/100) * weight, exactly as before. For POWER_W it is the
        # resource-time at or above the floor, which is the only reading of "used"
        # watts admit. One definition, so IDLE cannot mean two things.
        self.used += weight - wasted
        self.total += weight
        self.waste_total += wasted
        if band == "red":
            # Ranked by resource-time *wasted*, not held: a 100-hour job at 24% is
            # a bigger finding than a 10-hour job at 0%.
            self.worst.append(WorstJob(wasted, jobid, user, weight, value,
                                       runtime, duration))
            self.worst.sort(key=lambda item: -item[0])
            del self.worst[self.WORST:]

    def waste_of(self, value: float, weight: float) -> float:
        """Resource-time this job wasted, in the units the weight is in.

        For a percentage, the unused fraction of what it held. For an absolute
        metric there is no fraction to take, so a job below the cutoff wastes all of
        it and one above wastes none: for POWER_W that reads "GPU-hours spent below
        the idle floor", which is what a floor actually asserts. Grading it as a
        proportion of the cutoff instead would imply 50 W wastes twice what 100 W
        does, and watts are not utilization.
        """
        if self.absolute:
            return weight if value < self.cutoff() else 0.0
        return (1 - value / 100.0) * weight

    def cutoff(self) -> float:
        """The red threshold for this column, from the site config."""
        return self.thresholds.cutoff(self.header)

    def graded(self) -> int:
        """Jobs this metric measured -- the denominator behind its Worst row.

        Differs per metric because coverage does: a finished job with no stored blob
        has no GPU% but still has DCGM data, so SM_ACT% can cover more jobs than
        GPU% over the same selection.
        """
        return sum(count for count, _weight in self.bands.values())

    def band_of(self, value: Optional[float]) -> str:
        """This column's band for ``value``, or "" when it is not graded."""
        return self.thresholds.grade(self.header, value)

    def idle(self) -> float:
        """Allocated resource-time that went unused."""
        return self.total - self.used

    def pooled(self) -> Optional[float]:
        """Utilization over the whole selection, as a percent, or None if unmeasured.

        The same number the pooled row prints for this column: used resource-time
        over allocated. Used to grade the IDLE cell, so a metric that is mostly
        waste reads red.
        """
        return 100.0 * self.used / self.total if self.total else None

    def _amount(self, weight: float) -> str:
        """A resource amount with its unit, one decimal, trailing ``.0`` trimmed.

        Fractional even in the count form: ``used`` is GPU-equivalents busy, not
        whole GPUs. Rounding it to an integer made the numbers stop adding up -- 20
        allocated and 17.4 used printed as "17 used, 3 idle (13%)" while 3/20 is 15%.

        Byte amounts are scaled again here, to GB or TB, because a raw byte-second
        figure is nine digits wide and unreadable in a table cell. That choice is
        made once from the tally's own total, so every amount in a row shares one
        unit -- deciding per value printed "126.5TBh allocated, 5414.7GBh used".
        """
        scale, unit = self._unit()
        value = weight / scale
        if 0 < value < 0.05:
            # Would print as "0", which reads as nothing beside its own "(22%)".
            return "<0.1" + unit
        return ("%.1f" % value).removesuffix(".0") + unit

    def _unit(self) -> Tuple[float, str]:
        """``(divisor, suffix)`` for this tally's amounts, promoting GB to TB."""
        if self.unit.startswith("GB") and self.total / self.scale >= 10000:
            return self.scale * 1024.0, "TB" + self.unit[2:]
        return self.scale, self.unit

    def stat_row(self) -> List[Tuple[str, str]]:
        """This metric's table row as ``(cell, band)`` pairs; band "" means no tint.

        Two columns and three counts. ALLOC and USED came out because IDLE already
        carries the same information in the form anyone acts on -- how much went
        unused, and what share of the allocation that was. The band cells are bare
        job counts; their resource shares stay in the CSV for scripting.

        IDLE carries the metric's own pooled grade, so the eye lands on the metrics
        that wasted their allocation. Each band cell is tinted its own colour, since
        that is the colour it names.
        """
        idle = self.idle()
        idle_pct = round(100 * idle / self.total) if self.total else 0
        row = [(self.header, ""),
               ("%s (%d%%)" % (self._amount(idle), idle_pct),
                self.band_of(self.pooled()))]
        for band in ("red", "yellow", "green"):
            row.append(("%d" % self.bands[band][0], band))
        return row

    def csv_cells(self) -> List[str]:
        cells = ["metric=%s" % self.header,
                 "allocated=%s" % self._amount(self.total),
                 "used=%s" % self._amount(self.used),
                 "idle=%s" % self._amount(self.idle())]
        for band in ("red", "yellow", "green"):
            jobs, weight = self.bands[band]
            cells.append("%s=%d" % (band, jobs))
            cells.append("%s-amount=%s" % (band, self._amount(weight)))
        return cells


class SummaryRenderer:
    """Streaming form of summarize(): add() chunks as they arrive, then finish().

    Output is byte-identical to one summarize() call over the concatenated
    chunks: the context/header block prints once (on the first add or finish),
    rows print per add, and the Mean footer (when more than one row rendered)
    or the empty-selection message prints on finish.
    """

    def __init__(self, context: List[Tuple[str, str]], options: RenderOptions, out=None,
                 specs: Optional[List[MetricSpec]] = None) -> None:
        self.out = out or sys.stdout
        self.options = options
        self.context = context
        self.columns = cols_for(summary_columns(specs), options.view,
                                options.show_dcgm, options.diagnose)
        self.headers = [c.header for c in self.columns]
        self.dcgm_headers = [c.header for c in self.columns if c.group == "dcgm"]
        self.writer = csv.writer(self.out, lineterminator="\n") if options.csv else None
        self.count = 0
        self.sums = {key: [0, 0] for key in ("cpu", "mem", "gpu", "gmem")}  # [total, count]
        self.sums_dcgm = {header: [0.0, 0] for header in self.dcgm_headers}
        # The same figures weighted by how much hardware each job held, and (when
        # the values cover whole runtimes) for how long: [sum(v * w), sum(w)].
        # A job's value is one number for all its GPUs, so multiplying by the
        # weight and dividing by the total weight recovers the per-resource mean
        # across the selection. CPU% and MEM% join this row only under time
        # weighting, since GPU count says nothing about them.
        self.weighted = {key: [0.0, 0.0] for key in ("cpu", "mem", "gpu", "gmem")}
        # Only metrics whose cross-GPU aggregation is itself a mean can be
        # GPU-weighted. ENERGY_kWh sums over a job's GPUs and PWRmax_W takes the
        # max, so scaling either by GPU count would produce a number that means
        # nothing; those cells stay blank.
        self.weightable = {spec.header for spec in (specs or [])
                           if spec.agg == "mean" and spec.header in self.dcgm_headers}
        if specs is None:
            self.weightable = set(self.dcgm_headers)
        self.weighted_dcgm = {header: [0.0, 0.0] for header in self.weightable}
        self.gpu_counts = set()     # distinct GPU counts, to know if weighting matters
        self.durations = set()      # distinct runtimes, likewise
        self.gpu_total = 0          # GPUs across the selection, for the Jobs footer
        self.unweighted = 0         # jobs left out of the weighting for want of a runtime
        self.no_blob = 0            # jobs with no stored blob, excluded from every tally
        # {jobid: ({header: wasted}, user, {headers it is red in})} for jobs red in
        # at least one graded
        # metric, which is what the combined rankings need: a share cannot be taken
        # until the selection's totals are known, so the candidates must be kept.
        # Bounded by the red jobs, not the selection -- on a healthy partition, few.
        self.waste: Dict[str, tuple] = {}
        # One tally per graded column, so every metric on screen gets a summary and
        # the set follows the view for free: 8 by default, CPU%/MEM% under --cpu,
        # 6 under --gpu, the full catalog under --dcgm. Under time weighting the
        # weights are resource-seconds and render as hours; otherwise bare counts.
        thresholds = options.thresholds or Thresholds(**DEFAULT_THRESHOLDS)
        hours = options.time_weighted
        self.tallies = {header: EfficiencyTally(header, thresholds, *_resource_of(header, hours))
                        for header in self.headers if header.endswith("%")}
        # POWER_W joins them even though it is not a percentage: watts are the one
        # idle signal a duty cycle cannot fake. Weighted by GPU-time like the rest of
        # the GPU family, and banded against a watt floor rather than a percentage,
        # so its IDLE reads "GPU-hours spent under the floor" -- all-or-nothing per
        # sample, where a percentage's IDLE is a fraction of each.
        if "POWER_W" in self.headers:
            self.tallies["POWER_W"] = EfficiencyTally(
                "POWER_W", thresholds, *_resource_of("POWER_W", hours),
                absolute=True, value_unit="W")
        self._started = False

    def _line(self, row: dict, color: bool = True) -> str:
        """One rendered row, tinted by grade when the options ask for it.

        The escape codes wrap the *padded* cell, never the value: inserting them
        first would make str.format count them toward the column width and skew
        every column to the right of the first coloured one.
        """
        cells = []
        for col in self.columns:
            text = col.fmt.format(str(row.get(col.header, "")))
            band = self._band(col.header, row.get(col.header)) if color else ""
            cells.append(tint(text, band))
        return " ".join(cells)

    def _band(self, header: str, cell) -> str:
        """The grade for a rendered cell; see :func:`cell_band`."""
        return cell_band(self.options, header, cell)

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        if not self.options.header:
            return
        if self.options.csv:
            for label, value in self.context:
                self.writer.writerow([label, value])
            self.writer.writerow(self.headers)
        else:
            for label, value in self.context:
                print(fmt_context(label, value), file=self.out)
            header_line = self._line({c.header: c.header for c in self.columns},
                                     color=False)
            print(header_line, file=self.out)
            print("-" * len(header_line), file=self.out)

    def _weights(self, record: Optional[JobRecord], gpus: int) -> Dict[str, float]:
        """How much this job counts toward the weighted mean, per column family.

        Each weight is the amount of the resource the column measures: cores for
        CPU%, bytes for MEM%, GPUs for GPU% and GMEM%. Weighting each column by its
        own resource is what makes the row the pooled utilization rather than an
        average of averages.

        Under time weighting the weight is that resource multiplied by the elapsed
        seconds -- the *resource-time* the job was charged -- which is what stops
        100 five-minute jobs from outvoting one two-day job. For CPU% the identity
        is exact: per job it is ``100 x cpu_seconds / (elapsed x cores)``, so
        summing numerator and denominator over the selection is the same as
        averaging the per-job values weighted by ``elapsed x cores``.

        A job with an unknown runtime cannot be placed on that scale, so it is left
        out of the weighted row (and counted in ``unweighted``) rather than silently
        given a weight of zero or one.
        """
        if record is None:
            return {"cpu": 0.0, "mem": 0.0, "gpu": 0.0, "gmem": 0.0}
        cores, memory = blob_capacity(record.stats)
        if not self.options.time_weighted:
            return {"cpu": float(cores), "mem": float(memory),
                    "gpu": float(gpus), "gmem": float(gpus)}
        seconds = record.duration
        if not seconds or seconds <= 0:
            self.unweighted += 1
            return {"cpu": 0.0, "mem": 0.0, "gpu": 0.0, "gmem": 0.0}
        self.durations.add(seconds)
        return {"cpu": cores * seconds, "mem": memory * seconds,
                "gpu": gpus * seconds, "gmem": gpus * seconds}

    def _note_waste(self, jid: str, user: str, values: Dict[str, float],
                    weights: Dict[str, float], runtime: str = "-",
                    duration: Optional[int] = None) -> None:
        """Record what a job wasted per metric, if it is red in any of them.

        Every metric's waste is kept, not just the ones the job is red in, because
        the resource it wasted is real either way; the red test only decides whether
        the job is a candidate at all. That test is what keeps the lists actionable:
        a 95%-efficient job can idle 50 GPU-hours simply by being enormous.
        """
        wasted, red = {}, set()
        for header, tally in self.tallies.items():
            value, weight = values.get(header), weights[tally.weight_key]
            if value is None or weight <= 0:
                continue
            wasted[header] = tally.waste_of(value, weight)
            if tally.band_of(value) == "red":
                red.add(header)
        if red:
            self.waste[jid] = (wasted, user, frozenset(red), dict(values),
                               runtime, duration)

    def _combined_worst(self, headers: Tuple[str, ...]
                        ) -> List[Tuple[float, str, str, List[Tuple[str, float]]]]:
        """Worst jobs red in *every* one of ``headers``, by summed waste share.

        The conjunction is the point: a job on this row is idle by all of the
        measures it names, so there is nothing to argue about. It also means the row
        is often absent, which is itself the answer -- nothing was bad by every
        measure at once.

        The metrics are in different units -- GPU-hours, core-hours, GPU-hours below
        a watt floor -- and cannot be added. Any exchange rate between them would be
        invented, and on a GPU cluster a wrong one decides the ranking by itself.
        Normalising each job by the selection's own total waste in that metric avoids
        the question: a job that caused a third of the idle GPU-time and a fifth of
        the idle core-time scores 0.33 + 0.20. Each share is reported, so the reader
        sees which measure drove the ranking.
        """
        tallies = [(h, self.tallies[h]) for h in headers if h in self.tallies]
        if len(tallies) < len(headers) or any(t.waste_total <= 0 for _h, t in tallies):
            # A narrowed view is missing one, or nothing was wasted in it; a share of
            # zero total is undefined and the metric's own row already says so.
            return []
        scored = []
        for jid, (wasted, user, red, values, runtime, duration) in self.waste.items():
            if not set(headers) <= red:
                # Red in *every* metric of the row, not any of them. An OR let a job
                # that merely wasted some GPU-time onto the four-metric list while
                # drawing full power; requiring all of them means the row answers
                # "idle by every measure we have", which is the unambiguous case.
                continue
            shares = [(h, wasted.get(h, 0.0) / t.waste_total) for h, t in tallies]
            scored.append((sum(v for _h, v in shares), jid, user,
                           [(h, values.get(h)) for h, _t in tallies],
                           runtime, duration))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[:EfficiencyTally.WORST]

    def _combined_candidates(self, headers: Tuple[str, ...]) -> int:
        """How many jobs are red in *all* of ``headers``."""
        return sum(1 for entry in self.waste.values() if set(headers) <= entry[2])

    def add(self, jobids: List[str], records: Dict[str, JobRecord],
            dcgm_data: Dict[str, Tuple[dict, dict]]) -> None:
        self._start()
        options = self.options
        do_dcgm = options.show_dcgm
        if options.view == "gpu":
            jobids = [j for j in jobids if j in records and records[j].gpus]
        self.count += len(jobids)
        for jid in jobids:
            record = records.get(jid)
            row = {
                "JOBID": jid,
                "USER": record.user if record else "?",
                "STATE": record.state if record else "?",
                "NODE": record.nodes if record else "-",
                "#GPU": str(record.gpus) if record and record.gpus else "-",
                "RUNTIME": record.runtime if record else "-",
            }
            gpus = record.gpus if record else 0
            weights = self._weights(record, gpus)
            metrics = blob_metrics(record.stats if record else None)
            if metrics is None:
                for col in ("CPU%", "MEM%", "GPU%", "GMEM%"):
                    row[col] = "-"
            else:
                for key, col, value in zip(("cpu", "mem", "gpu", "gmem"),
                                           ("CPU%", "MEM%", "GPU%", "GMEM%"), metrics):
                    row[col] = "-" if value is None else str(value)
                    if value is not None:
                        self.sums[key][0] += value
                        self.sums[key][1] += 1
                        if weights[key]:
                            self.weighted[key][0] += value * weights[key]
                            self.weighted[key][1] += weights[key]
                        if key == "gpu" and gpus:
                            # Counted here rather than per allocation, so the footer
                            # total matches the GPUs actually behind the GPU figures.
                            self.gpu_total += gpus
                            self.gpu_counts.add(gpus)
            if do_dcgm:
                overall = dcgm_data.get(jid, ({}, {}))[0]
                for header in self.dcgm_headers:
                    value = overall.get(header)
                    row[header] = format_by_header(header, value)
                    if value is not None:
                        self.sums_dcgm[header][0] += value
                        self.sums_dcgm[header][1] += 1
                        if weights["gpu"] and header in self.weightable:
                            self.weighted_dcgm[header][0] += value * weights["gpu"]
                            self.weighted_dcgm[header][1] += weights["gpu"]
                if options.diagnose:
                    row["DIAG"] = diagnose_dcgm(overall, record.duration if record else None,
                                                options.min_runtime)
            # Every graded column at once, now that both the blob and the DCGM
            # values are in hand. Each tally knows which resource weights it.
            # One value map for every graded metric, built once both the blob and
            # the DCGM values are in hand, and shared by the tallies and the waste
            # bookkeeping so the two cannot disagree about what a job scored.
            if metrics is None:
                # No stored blob, so the job is only half measured: it has DCGM
                # numbers but no CPU%/MEM%/GPU%/GMEM%. Feeding it to the DCGM tallies
                # alone made their denominators disagree with the blob ones -- 117
                # against 88 on one partition -- and a job cannot be compared with
                # the rest on a metric it has no value for. It stays in the listing,
                # since it is a real job; it just does not vote.
                self.no_blob += 1
            else:
                values = {}
                for header in self.tallies:
                    value = _blob_value(metrics, header)
                    if value is None and header in self.dcgm_headers and do_dcgm:
                        value = dcgm_data.get(jid, ({}, {}))[0].get(header)
                    if value is not None:
                        values[header] = value
                for header, tally in self.tallies.items():
                    tally.add(jid, row["USER"], values.get(header),
                              weights[tally.weight_key], row["RUNTIME"],
                              record.duration if record else None)
                self._note_waste(jid, row["USER"], values, weights,
                                 row["RUNTIME"], record.duration if record else None)
            if options.csv:
                self.writer.writerow([row[h] for h in self.headers])
            else:
                print(self._line(row), file=self.out)
        self.out.flush()

    def finish(self) -> None:
        self._start()
        options = self.options
        if self.count == 0:
            if options.header and not options.csv:
                print("  (no GPU jobs in this selection)", file=self.out)
            return
        # A single job still gets the metric table: it is how you see which band each
        # of its numbers falls in, which the row itself cannot say. What it does not
        # get is the rest of the block -- for one job the pooled row is that job's own
        # row repeated, a Worst row names it again, and the job counts are all 1.
        alone = self.count == 1

        # The single aggregate row: each column pooled over the resource it
        # measures, so it reads "of all the GPU-hours (or GPUs, or core-hours) this
        # selection held, this fraction was used". There is deliberately no per-job
        # mean. Utilization is bimodal -- jobs cluster near 0% or near 100% -- so
        # its average describes a job that does not exist, and it hides exactly the
        # case worth finding: a handful of large idle jobs among many small busy
        # ones. Unlike a mean this row stays true under that distribution, because
        # it is a ratio of totals rather than a centre.
        used_row = {c.header: "" for c in self.columns}
        for key, header in (("cpu", "CPU%"), ("mem", "MEM%"),
                            ("gpu", "GPU%"), ("gmem", "GMEM%")):
            total, n = self.weighted[key]
            used_row[header] = str(round(total / n)) if n else "-"
        if options.show_dcgm:
            for header in self.dcgm_headers:
                if header in self.weightable:
                    total, n = self.weighted_dcgm[header]
                else:
                    # A per-job total (ENERGY_kWh) or a peak (PWRmax_W): there is no
                    # resource to divide it by, so report the plain figure instead
                    # of a weighting that would mean nothing.
                    total, n = self.sums_dcgm[header]
                used_row[header] = format_by_header(header, total / n) if n else "-"

        # Every graded metric's own summary, in column order, skipping any that no
        # job reported. The pooled row above shows the utilization as a percentage;
        # these rows add the resource-time behind it, and how that time fell across
        # the bands. POWER_W is here too: its IDLE is resource-time spent under the
        # watt floor, which is a real quantity even though "used watts" is not.
        stats = [self.tallies[h] for h in self.headers
                 if h in self.tallies and self.tallies[h].total]
        # One worst row per named measure, in a fixed order so the block is diffable
        # across runs, skipping any with no red job. Not every graded metric: the
        # DCGM catalog would swamp the footer, and these four are the ones that say
        # something distinct -- duty cycle, SM residency, watts, and the host.
        worst = [self.tallies[h] for h in WORST_METRICS
                 if h in self.tallies and self.tallies[h].worst]
        # The pooled row's label names the resource it leads with: GPUs where the
        # view has them, else cores, else whatever metric did report.
        lead = next((t for t in (self.tallies.get("GPU%"), self.tallies.get("CPU%"))
                     if t is not None and t.total), None)
        if lead is None and stats:
            lead = stats[0]
        # Two rankings: the two distinct *resources*, and all four measures. The
        # first answers "which job drained the most hardware", the second "which job
        # looks worst by any measure" -- three of its four terms describe the GPU, so
        # a GPU-idle job outscores an equally wasteful CPU-idle one.
        combined = [("both", COMBINED_2, self._combined_worst(COMBINED_2)),
                    ("all", COMBINED_4, self._combined_worst(COMBINED_4))]
        combined = [(name, hs, rows) for name, hs, rows in combined if rows]

        # The job counts differ whenever the selection mixes CPU-only and GPU work:
        # a CPU-only job has no GPU% to average, so it is absent from the GPU
        # figures rather than counted as zero.
        counts = ["cpu-jobs=%d" % self.sums["cpu"][1]]
        if options.view != "cpu":
            # A --cpu run hid the GPU columns; repeating GPU totals here is noise.
            counts.append("gpu-jobs=%d" % self.sums["gpu"][1])
            if self.gpu_total:
                counts.append("gpus=%d" % self.gpu_total)
        if self.unweighted:
            counts.append("no-runtime=%d" % self.unweighted)
        if self.no_blob:
            counts.append("no-blob=%d" % self.no_blob)

        def padded(label, cells):
            """A footer row padded to the header width, so the CSV stays rectangular.
            parse_csv drops it by its first cell either way."""
            return ([label] + cells + [""] * len(self.headers))[:len(self.headers)]

        if options.csv:
            if not alone:
                used_row["JOBID"] = lead.csv_row if lead else "Used"
                self.writer.writerow([used_row[h] for h in self.headers])
            for one in stats:
                # "Stat<METRIC>": parse_csv skips the prefix, since the metric set is
                # open-ended (18 columns under --dcgm) and cannot be enumerated.
                self.writer.writerow(padded("Stat" + one.header, one.csv_cells()))
            if alone:
                return
            for one in worst:
                # Same shape as the text rows: value, wasted resource-time, elapsed.
                self.writer.writerow(padded("Worst" + _worst_slug(one.header), [
                    "%s=%d%s:%s(%s)" % (job.jobid, round(job.value), one.value_unit,
                                        one._amount(job.weight), job.runtime)
                    for job in one.worst]))
            for name, _hs, rows in combined:
                self.writer.writerow(padded("Worst" + name.capitalize(), [
                    "%s(%s)" % (_combined_cell(jid, values, self.tallies)
                                .replace(" ", "=", 1).replace(" ", "/"), runtime)
                    for _score, jid, _user, values, runtime, _dur in rows]))
            self.writer.writerow(padded("Jobs", counts))
        else:
            # Three sections, because the block answers three questions: how was each
            # metric used, how do they compare, and which jobs are the problem.
            used_row["JOBID"] = lead.row if lead else "Used:"
            summary = ([] if alone else [self._line(used_row)]) + self._stat_table(stats)

            problems = []
            if not alone:
                # (label, [(user, entry text, is long-running)]) per row. Labels carry
                # counts, so widths vary; everything is padded to one width so the job
                # lists line up under each other across rows.
                rows_out = []
                for one in worst:
                    rows_out.append((
                        "Worst %s (%d/%d):" % (_worst_slug(one.header),
                                               one.bands["red"][0], one.graded()),
                        [(job.user,
                          "%s:%d%s:%s(%s)" % (job.jobid, round(job.value),
                                              one.value_unit, one._amount(job.weight),
                                              job.runtime),
                          (job.duration or 0) > LONG_RUNNING)
                         for job in one.worst]))
                for name, headers, ranked in combined:
                    # No denominator: a row spanning metrics with different coverage
                    # has no single honest total, so only the candidate count is shown.
                    rows_out.append((
                        "Worst %s (%d):" % (name, self._combined_candidates(headers)),
                        [(user,
                          "%s(%s)" % (_combined_cell(jid, values, self.tallies)
                                      .replace(" ", ":", 1).replace(" ", "/"), runtime),
                          (duration or 0) > LONG_RUNNING)
                         for _score, jid, user, values, runtime, duration in ranked]))
                # Both widths are over possibly-empty sequences: a selection with
                # nothing red has no Worst rows at all, leaving only Jobs:.
                label_width = max([len(label) for label, _ in rows_out] + [len("Jobs:")])
                user_width = max([len(user) + 1 for _label, entries in rows_out
                                  for user, _t, _l in entries] + [0])
                for label, entries in rows_out:
                    problems.extend(self._worst_rows(label, entries, label_width,
                                                     user_width))
                problems.append("%-*s %s" % (label_width, "Jobs:", "  ".join(counts)))

            self._print_sections([("Summary by metric", summary),
                                  ("Average efficiency  (filled = used, grey = idle)",
                                   self._bar_lines(stats)),
                                  ("Problem jobs", problems)])


    STAT_HEADERS = ("METRIC", "IDLE", "RED", "YELLOW", "GREEN")

    # Two lines of legend. The first states the cutoffs, which no longer need a
    # column now that they are uniform. The second is there because "green" means
    # only "not pathological": at a cutoff of 10 a job at 21% is green while wasting
    # four fifths of its cores, so a selection can be half idle with almost every
    # job green. IDLE is the efficiency number; the bands say whether the waste is
    # concentrated in a few jobs or spread across all of them, which is the
    # difference between someone to talk to and a habit.
    STAT_LEGEND = (
        "red below %(red)g%%, yellow below %(yellow)g%%, green above;"
        " POWER_W red below %(power)g W. Counts are jobs.",
        "IDLE is resource-time that went unused -- for POWER_W, the time spent under"
        " that floor.",
        "bands catch pathological jobs, IDLE measures efficiency:"
        " no red with a high IDLE means every job wastes a little",
    )

    def _legend(self) -> List[str]:
        """:data:`STAT_LEGEND` with this run's actual cutoffs filled in."""
        thresholds = self.options.thresholds or Thresholds(**DEFAULT_THRESHOLDS)
        values = {"red": thresholds.red, "yellow": 2 * thresholds.red,
                  "power": thresholds.power_w}
        return [line % values for line in self.STAT_LEGEND]

    WORST_WIDTH = 132

    def _worst_rows(self, label: str, entries, label_width: int,
                    user_width: int) -> List[str]:
        """One line per user: ``label  user| job:val:wasted(elapsed), job:...``.

        Grouped because a single user usually owns several of the worst jobs, and
        repeating their name three times says less than showing they own the row.
        Entries wrap onto continuation lines rather than running past the table.
        """
        out = []
        for user, jobs in _worst_groups(entries):
            prefix = "%-*s %-*s" % (label_width, label, user_width, user + "|")
            label = ""                          # only the first line is labelled
            line, count = prefix, 0
            for text, long_running in jobs:
                cell = _SGR["red"] + text + _RESET if (
                    long_running and self.options.color) else text
                candidate = line + (" " if count == 0 else ", ") + cell
                if count and len(_ESC_RE.sub("", candidate)) > self.WORST_WIDTH:
                    out.append(line + ",")
                    line, count = " " * len(prefix) + " " + cell, 1
                    continue
                line, count = candidate, count + 1
            out.append(line)
        return out

    def _print_sections(self, sections: List[Tuple[str, List[str]]]) -> None:
        """Print ``(title, lines)`` sections, numbered and ruled.

        Numbering runs over the sections that actually have content, so --no-plot
        leaves "1." and "2." rather than a gap where 2 was -- a missing number reads
        as something having failed. The rule spans the section's own widest line, so
        each hugs its content instead of inheriting the job table's width.

        Titles and rules are furniture: --noheader drops them and keeps the data.
        """
        number = 0
        for title, lines in sections:
            if not lines:
                continue
            number += 1
            print(file=self.out)
            if self.options.header:
                heading = "%d. %s" % (number, title)
                print(heading, file=self.out)
                width = max([len(_ESC_RE.sub("", ln)) for ln in lines] + [len(heading)])
                print("-" * width, file=self.out)
            for line in lines:
                print(line, file=self.out)

    def _bar_lines(self, stats: List["EfficiencyTally"]) -> List[str]:
        """The efficiency bars, or nothing when they are switched off or in CSV.

        Never in CSV: a bar chart has no place in a machine format, and parse_csv
        would have to be taught to skip it.
        """
        if not self.options.plot_avgeff or self.options.csv:
            return []
        return self._eff_bars(stats)

    def _eff_bars(self, stats: List["EfficiencyTally"]) -> List[str]:
        """Horizontal utilization bars, one per graded metric.

        The same data as the table's IDLE column, in the form that answers "which
        resource was wasted" without arithmetic: bar length is the pooled utilization,
        so bar percent and IDLE percent always sum to 100.

        Drawn through :func:`bar_lines`, which the per-GPU charts also use, and with
        this module's own SGR codes rather than rich -- the report path is the common
        one and should not import a rendering library to print a table.
        """
        items = []
        for one in stats:
            used = one.pooled()
            # POWER_W has a table row but no bar. Its "used" is time above the watt
            # floor, which is a detector reading rather than a fraction of a resource:
            # on a partition of GPUs idling at 119 W it fills to 100% beside SM_ACT%
            # at 2%, reading as the healthiest metric when it is describing the same
            # idle GPUs and merely failing to flag them.
            if used is not None and not one.absolute:
                items.append((one.header, used,
                              one.band_of(used) if self.options.color else ""))
        return bar_lines(items)


    def _stat_table(self, stats: List["EfficiencyTally"]) -> List[str]:
        """The per-metric table: one row per graded metric, tinted by band.

        Widths are measured from the content so the columns line up whatever the
        metric names and magnitudes are. Escape codes wrap the *padded* cell, never
        the value -- the same rule :meth:`_line` follows, because padding a string
        that already contains them counts the escape bytes toward the width and
        shifts every later column.
        """
        if not stats:
            return []
        rows = [one.stat_row() for one in stats]
        widths = [max(len(self.STAT_HEADERS[i]), max(len(r[i][0]) for r in rows))
                  for i in range(len(self.STAT_HEADERS))]
        out = []
        last = len(widths) - 1
        if self.options.header:
            for line in self._legend():
                out.append("  " + line)
            out.append("  ".join(h if i == last else h.ljust(widths[i])
                                 for i, h in enumerate(self.STAT_HEADERS)))
        for row in rows:
            cells = []
            for i, ((text, band), width) in enumerate(zip(row, widths)):
                # The final column is left unpadded rather than padded and stripped
                # afterwards: with colour on, its trailing spaces would sit *inside*
                # the escape wrapper where rstrip cannot reach them, and the tinted
                # output would then differ from the plain output by more than the
                # escapes.
                cell = text if i == last else text.ljust(width)
                tint = band if (band and self.options.color) else ""
                cells.append(_SGR[tint] + cell + _RESET if tint else cell)
            out.append("  ".join(cells))
        return out


class DetailRenderer:
    """Streaming form of detail(): independent per-job blocks per add().

    detail has no footer; finish() only emits the text-mode empty-selection
    message when nothing rendered.
    """

    def __init__(self, context: List[Tuple[str, str]], options: RenderOptions, out=None) -> None:
        self.out = out or sys.stdout
        self.options = options
        self.context = context
        self.columns = cols_for(DETAIL_COLUMNS, options.view, options.show_dcgm, options.diagnose)
        self.writer = csv.writer(self.out, lineterminator="\n") if options.csv else None
        self.count = 0
        # Every node seen *before* filtering, so an unmatched --nodename can say what
        # was actually there, and whether anything matched at all.
        self.nodes_seen = set()
        self.matched = 0
        self._started = False

    def _line(self, cells) -> str:
        """One per-GPU row, graded like the per-job table above it."""
        return " ".join(
            tint(c.fmt.format(str(cells[c.index])),
                 cell_band(self.options, c.header, cells[c.index]))
            for c in self.columns)

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        if not self.options.header:
            return
        if self.options.csv:
            for label, value in self.context:
                self.writer.writerow([label, value])
            self.writer.writerow(["JOBID"] + [c.header for c in self.columns])
        else:
            for label, value in self.context:
                print(fmt_context(label, value), file=self.out)
            print(file=self.out)

    def _rows_for(self, jid: str, record: Optional[JobRecord],
                  dcgm_data: Dict[str, Tuple[dict, dict]]):
        rows = blob_detail(record.stats if record else None)
        if self.options.show_dcgm:
            per_gpu = dcgm_data.get(jid, ({}, {}))[1]
            rows = [extend_detail_row(r, per_gpu, record.duration if record else None,
                                      self.options.min_runtime, self.options.diagnose)
                    for r in rows]
        self.nodes_seen.update(row[_NODE_INDEX] for row in rows)
        if self.options.nodename:
            rows = [row for row in rows if row[_NODE_INDEX] == self.options.nodename]
        self.matched += len(rows)
        return rows

    def add(self, jobids: List[str], records: Dict[str, JobRecord],
            dcgm_data: Dict[str, Tuple[dict, dict]]) -> None:
        self._start()
        options = self.options
        if options.view == "gpu":
            jobids = [j for j in jobids if j in records and records[j].gpus]
        self.count += len(jobids)
        if options.csv:
            for jid in jobids:
                record = records.get(jid)
                for row in self._rows_for(jid, record, dcgm_data):
                    self.writer.writerow([jid] + [row[c.index] for c in self.columns])
        else:
            for jid in jobids:
                record = records.get(jid)
                print("Job %s  [%s]  %s" % (jid, record.state if record else "?",
                                            record.name if record else "?"), file=self.out)
                rows = self._rows_for(jid, record, dcgm_data)
                if not rows:
                    print("  (no jobstats data)\n", file=self.out)
                    continue
                header_line = self._line(DETAIL_HEADER)
                print("  " + header_line, file=self.out)
                print("  " + "-" * len(header_line), file=self.out)
                for row in rows:
                    print("  " + self._line(row), file=self.out)
                for line in self._unit_charts(rows):
                    print(line, file=self.out)
                print(file=self.out)
        self.out.flush()

    def _unit_charts(self, rows) -> List[str]:
        """Efficiency bars grouped by node, or by GPU when there is only one node.

        The rows already carry every number; what they do not give is a comparison.
        Grouping by whatever distinguishes them -- the node normally, the card once a
        single node is in play -- is what turns sixteen rows of twelve columns into
        "this node is the slow one".
        """
        if not self.options.plot_avgeff or self.options.csv or not rows:
            return []
        metrics = [c for c in self.columns
                   if c.header.endswith("%") and c.header not in ("GPU",)]
        if not metrics:
            return []
        nodes = list(dict.fromkeys(row[_NODE_INDEX] for row in rows))
        by_gpu = len(nodes) == 1
        if by_gpu:
            groups = [("GPU %s" % row[_GPU_INDEX], [row]) for row in rows]
            title = "Efficiency by GPU on %s" % nodes[0]
        else:
            groups = [(node, [r for r in rows if r[_NODE_INDEX] == node])
                      for node in nodes]
            title = "Efficiency by node"
        blocks = []
        for label, members in groups:
            items = []
            for col in metrics:
                # The mean over the group's rows. Within a node the GPUs are equal, so
                # that is the pooled figure; CPU% is already per-node, repeated on
                # every row, so averaging identical values returns them unchanged.
                values = [v for v in (cell_value(r[col.index]) for r in members)
                          if v is not None]
                if not values:
                    continue
                mean = sum(values) / len(values)
                items.append((col.header, mean,
                              cell_band(self.options, col.header, mean)))
            if items:
                blocks.append(["    " + label] + bar_lines(items, indent="      "))
        if not blocks:
            return []
        # Side by side: a four-node job is eight bars tall rather than thirty-two, and
        # two nodes can be compared without scrolling between them.
        return ["", "  %s  (filled = used, grey = idle)" % title] + in_columns(
            blocks, available=terminal_width(self.out))

    def finish(self) -> None:
        self._start()
        if self.options.nodename and not self.matched:
            raise no_such_node(self.options.nodename, self.nodes_seen)
        if self.count == 0 and not self.options.csv:
            print("(no GPU jobs in this selection)", file=self.out)


def summarize(jobids: List[str], records: Dict[str, JobRecord],
              dcgm_data: Dict[str, Tuple[dict, dict]], context: List[Tuple[str, str]],
              options: RenderOptions, out=None) -> None:
    """One row per job: blob metrics, optional DCGM columns, optional DIAG."""
    renderer = SummaryRenderer(context, options, out)
    renderer.add(jobids, records, dcgm_data)
    renderer.finish()


def detail(jobids: List[str], records: Dict[str, JobRecord],
           dcgm_data: Dict[str, Tuple[dict, dict]], context: List[Tuple[str, str]],
           options: RenderOptions, out=None) -> None:
    """Per-node / per-GPU breakdown for each job."""
    renderer = DetailRenderer(context, options, out)
    renderer.add(jobids, records, dcgm_data)
    renderer.finish()


def dcgm_report(jobids: List[str], records: Dict[str, JobRecord],
                dcgm_data: Dict[str, Tuple[dict, dict]], specs: List[MetricSpec],
                context: List[Tuple[str, str]], options: RenderOptions,
                out=None) -> None:
    """One row per job, with the profiling block taken from ``specs``.

    The same renderer the summary view uses, so the two print identical columns;
    ``--ext`` only widens the profiling block. Per-GPU numbers live in
    ``jobscope detail`` and in the ``--ts`` time series.
    """
    renderer = SummaryRenderer(context, options, out, specs=specs)
    renderer.add(jobids, records, dcgm_data)
    renderer.finish()


def dcgm_timeseries(jobids: List[str], records: Dict[str, JobRecord],
                    specs: List[MetricSpec], client: PrometheusClient,
                    timeout: Optional[float], options: RenderOptions,
                    step: Optional[int] = None, out=None) -> None:
    """Emit the raw per-scrape DCGM time series over the job's window as CSV.

    One row per GPU/timestamp; metrics are de-duplicated by Prometheus name. Each
    cell is the raw sampled value, scaled for display.
    """
    out = out or sys.stdout
    seen, ts_specs = set(), []
    for spec in specs:
        if spec.metric not in seen:
            seen.add(spec.metric)
            ts_specs.append(spec)
    # Query every spec (including hidden ones, which feed a derived column) but
    # emit the displayed columns, so this header matches `jobscope live --ts`.
    columns = columns_for(specs)
    derived = applicable_derived(specs)
    sampling_period = client.sampling_period
    writer = csv.writer(out, lineterminator="\n")
    nodes_seen, matched, wrote_header = set(), False, False

    def write(row) -> None:
        """Emit *row*, writing the header first if it has not been written yet.

        Lazily, because a --nodename that matches nothing raises below: a header with
        no rows under it is a CSV that reads as "this node was idle" and confuses
        `jobscope plot` into "no numeric values". Nothing written is the honest answer,
        and it is what the live path already does.
        """
        nonlocal wrote_header
        if options.header and not wrote_header:
            writer.writerow(["JOBID", "EPOCH", "TIME", "NODE", "GPU"]
                            + [header for _key, header, _dec in columns])
            wrote_header = True
        writer.writerow(row)

    for jid in jobids:
        record = records.get(jid)
        gpus = discover_gpus(record, client, timeout) if record else []
        if not gpus:
            print("warn: job %s has no GPU samples" % jid, file=sys.stderr)
            continue
        uuid_to = {g["uuid"]: (g["node"], g["minor"]) for g in gpus}
        if options.nodename:
            # Before the queries, not after: dropping the other nodes' UUIDs here
            # shrinks the regex, so a 4-node job costs a quarter of the range queries
            # instead of fetching three nodes' samples to throw them away.
            nodes_seen.update(node for node, _ in uuid_to.values())
            uuid_to = {u: nm for u, nm in uuid_to.items() if nm[0] == options.nodename}
            if not uuid_to:
                continue
            matched = True
        regex = "^(" + "|".join(uuid_to) + ")$"
        # --step wins; otherwise never finer than the scrape interval, and coarse
        # enough to stay under Prometheus' points-per-series cap on a long job.
        start, span = range_window(record.start, record.end, options.window,
                                   sampling_period, step)
        series: Dict[str, dict] = {uuid: {} for uuid in uuid_to}
        for spec in ts_specs:
            for result in client.query_range(
                    '%s{%s=~"%s"}' % (spec.metric, spec.uuid_label, regex),
                    start, record.end, span, timeout):
                metric = result["metric"]
                uuid = metric.get(spec.uuid_label) or metric.get("uuid") or metric.get("UUID")
                if uuid not in series:
                    continue
                for stamp, value in result["values"]:
                    try:
                        series[uuid].setdefault(int(stamp), {})[spec.header] = float(value) * spec.scale
                    except (TypeError, ValueError):
                        pass
        rows = []  # (node, minor_sort, ts, csv_row)
        for uuid, (node, minor) in uuid_to.items():
            for stamp in sorted(series[uuid]):
                cells = series[uuid][stamp]
                # Recomputed per timestamp, so a ratio like GMEM% tracks growth.
                keyed = values_by_key(specs, cells)
                for column in derived:
                    cells[column.header] = column.fn(keyed)
                rows.append((node, gpu_minor_key(minor), stamp,
                             [jid, stamp, time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp)),
                              node, minor]
                             + [format_number(cells.get(h), d, missing="")
                                for _k, h, d in columns]))
        for _, _, _, row in sorted(rows, key=lambda x: (x[0], x[1], x[2])):
            write(row)

    if options.nodename and not matched:
        raise no_such_node(options.nodename, nodes_seen)


# How each spec's window reducer reads in the --describe output.
_REDUCER_NAME = {"avg": "mean", "max": "peak", "delta": "delta"}

def _live_rows(jobs: Dict[int, LiveJob], gpus: Dict[str, Gpu]):
    """Yield ``(job, gpu_or_None)`` in display order: by job, then by GPU.

    A job with no GPU samples yields once with ``None``, so it still gets a row
    saying so rather than vanishing from the table.
    """
    by_job: Dict[int, List[Gpu]] = defaultdict(list)
    for gpu in gpus.values():
        by_job[gpu.jobid].append(gpu)
    for raw_jobid in sorted(jobs, key=lambda j: job_sort_key(jobs[j])):
        job = jobs[raw_jobid]
        found = sorted(by_job.get(raw_jobid, []), key=lambda g: (g.host, g.minor, g.uuid))
        if not found:
            yield job, None
            continue
        for gpu in found:
            yield job, gpu


# What a --stats row aggregates over. "gpu" is one row per card, "node" pools a
# job's cards on one host, "job" pools every card it held.
STAT_LEVELS = ("gpu", "node", "job")
TS_STAT_TAIL = ("METRIC", "N", "MIN", "MEAN", "MAX", "LAST")


def _stat_lead(level: str, multi_job: bool) -> Tuple[str, ...]:
    """The identifying columns for a level, before METRIC.

    Each level names what it pooled: a node row says how many GPUs went into it, a job
    row how many nodes and GPUs. Without that the reader cannot tell a one-GPU mean
    from a sixteen-GPU one. JOBID leads only when the series covers more than one job,
    which for the usual single-job selection keeps the table narrow.
    """
    if level == "job":
        return ("JOBID", "NODES", "GPUS")
    job = ("JOBID",) if multi_job else ()
    return job + (("NODE", "GPUS") if level == "node" else ("NODE:GPU",))


def timeseries_stats(rows: List[dict], metrics: List[str], options: "RenderOptions",
                     out=None, level: str = "gpu") -> None:
    """``min / mean / max / last`` per metric, pooled at ``level``.

    Computed from the samples ``--ts`` already fetched rather than from fresh queries,
    so the numbers cannot disagree with the series they summarize -- and so a window
    costs nothing extra to summarize.

    A plain mean of samples, which is not every metric's own reducer: the tables peak
    memory where this averages it. That is the honest reading of "the average over
    this window", and it is what ``jobscope plot`` prints under its charts.

    Pooling the samples themselves rather than averaging per-GPU means is what makes
    the node and job levels right when coverage is uneven: a card the exporter missed
    for half the window then carries half the weight, instead of counting as a full
    peer.
    """
    out = out or sys.stdout
    groups: Dict[tuple, dict] = {}
    for row in rows:
        jobid, node = row.get("JOBID", "?"), row.get("NODE", "?")
        gpu = row.get("GPU", "?")
        key = {"gpu": (jobid, node, gpu), "node": (jobid, node)}.get(level, (jobid,))
        found = groups.setdefault(key, {"values": {}, "nodes": set(), "gpus": set()})
        found["nodes"].add(node)
        found["gpus"].add((node, gpu))
        for metric in metrics:
            value = cell_value(row.get(metric))
            if value is not None:
                found["values"].setdefault(metric, []).append(value)

    multi_job = len({r.get("JOBID", "?") for r in rows}) > 1
    lead = _stat_lead(level, multi_job)
    headers = lead + TS_STAT_TAIL

    def labels(key, found) -> Tuple[str, ...]:
        jobid = key[0]
        if level == "job":
            return (jobid, str(len(found["nodes"])), str(len(found["gpus"])))
        job = (jobid,) if multi_job else ()
        if level == "node":
            return job + (key[1], str(len(found["gpus"])))
        return job + ("%s:%s" % (key[1], key[2]),)

    def order(item):
        key, _found = item
        return (key[0], key[1] if len(key) > 1 else "",
                gpu_minor_key(key[2]) if len(key) > 2 else 0)

    table = []
    for key, found in sorted(groups.items(), key=order):
        for metric in metrics:
            values = found["values"].get(metric)
            if not values:
                continue
            mean = sum(values) / len(values)
            table.append((labels(key, found), metric, len(values),
                          min(values), mean, max(values), values[-1]))
    if not table:
        print("No samples to summarize.", file=sys.stderr)
        return

    cells = [tuple(label) + (metric, str(n), "%.1f" % low, "%.1f" % mean,
                             "%.1f" % high, "%.1f" % last)
             for label, metric, n, low, mean, high, last in table]
    if options.csv:
        writer = csv.writer(out, lineterminator="\n")
        if options.header:
            writer.writerow(headers)
        for row in cells:
            writer.writerow(list(row))
        out.flush()
        return

    # The label columns are text and left-aligned; the numbers right-align under
    # their headers. METRIC is the boundary.
    text_cols = len(lead) + 1
    widths = [max(len(headers[i]), max(len(r[i]) for r in cells))
              for i in range(len(headers))]
    mean_at = len(lead) + 3
    if options.header:
        print("  " + "  ".join(h.ljust(widths[i]) if i < text_cols else h.rjust(widths[i])
                               for i, h in enumerate(headers)), file=out)
    for row, entry in zip(cells, table):
        # Tint the mean by its band, as the summary table tints IDLE: the column
        # anyone reads first should say whether the number is a problem.
        painted = [cell.ljust(widths[i]) if i < text_cols else cell.rjust(widths[i])
                   for i, cell in enumerate(row)]
        painted[mean_at] = tint(painted[mean_at], cell_band(options, entry[1], entry[4]))
        print("  " + "  ".join(painted).rstrip(), file=out)
    out.flush()


def live_timeseries(jobs: Dict[int, LiveJob], samples: Dict[str, Dict[int, dict]],
                    gpus: Dict[str, Gpu], specs: List[MetricSpec],
                    options: RenderOptions, out=None) -> None:
    """Emit one CSV row per GPU per sample over each job's runtime.

    The schema is deliberately the one :func:`dcgm_timeseries` writes
    (``JOBID,EPOCH,TIME,NODE,GPU,<metrics>``), because ``jobscope plot`` keys line
    charts on ``EPOCH``/``TIME`` and groups series by ``(NODE, GPU)`` -- so this
    pipes straight into it. ``GPU`` is the bare minor number for a whole card and
    ``minor.instance`` for a MIG slice, or slices sharing a minor would merge.
    """
    out = out or sys.stdout
    columns = build_columns(specs)
    derived = [d for d in DERIVED_COLUMNS if {s.key for s in specs}.issuperset(d.deps)]
    writer = csv.writer(out, lineterminator="\n")
    if options.header:
        writer.writerow(["JOBID", "EPOCH", "TIME", "NODE", "GPU"]
                        + [header for _k, header, _d in columns])

    for job, gpu in _live_rows(jobs, gpus):
        if gpu is None:
            continue
        for epoch in sorted(samples.get(gpu.uuid, {})):
            values = samples[gpu.uuid][epoch]
            # Recompute per timestamp, so MEM% tracks memory growth over the run.
            for column in derived:
                values[column.key] = column.fn(values)
            writer.writerow(
                [job["jobid"], epoch,
                 time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch)),
                 gpu.host, gpu.csv_id]
                + [format_number(values.get(key), dec, missing="")
                   for key, _h, dec in columns])


def describe(diagnose_on: bool = False, out=None) -> None:
    """Print a plain-English description of each summary column."""
    out = out or sys.stdout
    print("jobscope columns. CPU/MEM/GPU/GMEM come from the sacct blob (no network);", file=out)
    print("the DCGM columns (gpu view) and DIAG (gpu view + --diagnose) come from", file=out)
    print("Prometheus. For the full per-GPU DCGM catalog, run", file=out)
    print("'jobscope describe --dcgm' (add --ext for all %d metrics).\n" % len(ALL_SPECS),
          file=out)
    for header, source, text in SUMMARY_DESCRIPTIONS:
        print("  %-9s %s" % (header, source), file=out)
        for wrapped in textwrap.wrap(text, width=74):
            print("      " + wrapped, file=out)
        print(file=out)
    if diagnose_on:
        print(LEGEND, file=out)


def describe_dcgm(specs: List[MetricSpec], out=None) -> None:
    """Plain-English reference for the DCGM metric catalog."""
    out = out or sys.stdout
    reducer_name = _REDUCER_NAME
    n_default = len(DEFAULT_SPECS)
    # Hidden specs exist only to feed a derived column, so describe the column
    # instead -- what a reader sees in the table.
    shown = [s for s in specs if s.show]
    derived = applicable_derived(specs)
    print("DCGM GPU metrics. Each value is time-averaged over the job's [start,end]", file=out)
    print("window. Showing %d of %d metrics (%s). [reduce] = how the window is collapsed.\n"
          % (len(shown) + len(derived), len(ALL_SPECS),
             "all" if len(specs) > n_default else "default; --ext for the rest"), file=out)
    for spec in shown:
        print("  %-12s %-38s [reduce: %s]" % (spec.header, spec.metric,
              reducer_name.get(spec.reducer, spec.reducer)), file=out)
        for wrapped in textwrap.wrap(DESCRIPTIONS.get(spec.header, "(no description)"), width=74):
            print("      " + wrapped, file=out)
        print(file=out)
    for column in derived:
        print("  %-12s %-38s [reduce: -]" % (column.header, column.source), file=out)
        for wrapped in textwrap.wrap(DESCRIPTIONS.get(column.header, "(no description)"), width=74):
            print("      " + wrapped, file=out)
        print(file=out)
