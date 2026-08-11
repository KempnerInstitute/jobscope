"""Rendering of the summary, detail, per-GPU DCGM, and time-series views.

Output layout (spacing, context lines, and above all the CSV shape) is kept
stable: the CSV emitted here is what ``jobscope plot`` parses.
"""

import csv
import itertools
import re
import shutil
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

from . import cpu, dcgm, metrics
from .config import (
    BUCKET_OF,
    DEFAULT_LONG_RUNNING,
    DEFAULT_WORST_JOBS,
    REPORT_SECTIONS,
    Palette,
    Thresholds,
    parse_duration,
)
from .cpu import CgroupSpec, chosen_specs

# Functions and constants only: everything the source preference can move now lives on
# the frozen dcgm.catalog(), read at call time. This import used to be able to take a
# resolved view by value, which is how --gpu-source once resolved one source, said so on
# the Source line, and rendered another's numbers.
from .dcgm import (
    DESCRIPTIONS,
    MODEL_KEY,
    MetricSpec,
    applicable_derived,
    columns_for,
    format_by_header,
    format_number,
    gpu_minor_key,
    job_model,
)
from .errors import JobscopeError
from .job_eff import (
    ALIASED,
    BURSTY,
    CATEGORIES,
    DECLINING,
    FLAT_IDLE,
    MIN_SAMPLES,
    NO_DATA,
    SPARSE,
    STALE,
    STEADY,
    THIN,
    IdleSince,
    Measured,
    below_share,
    bucket,
    bucket_edges,
    classify,
    classify_description,
    classify_metrics,
    concurrent_idle,
    cutoff,
    distribution,
    full_scale,
    idle_stamps,
    longest_idle,
    measured_time,
    median_swing,
    qualifiers,
    series_shape,
    tier_criteria,
    tier_range,
    unceilinged,
    went_idle,
)

# Two formatters, a constant and the per-unit column list -- not the storage helpers:
# turning a stored summary into rows is jobscope.rows' job now, and
# tests/test_layering.py holds it there.
from .jobstats import GIB, UNIT_HEADERS, bytes_to_gb
from .models import GPU_LEVEL, NODE_LEVEL, JobRow, ReportContext
from .running import Gpu, RunningJob, build_columns, format_duration, job_sort_key

# A job running longer than this, and still on a Wasteful row, is the expensive
# kind of waste: a short bad job costs little, whereas hours of idle hardware do
# not come back. Its entry is highlighted. [defaults] long_running overrides it.
LONG_RUNNING = parse_duration(DEFAULT_LONG_RUNNING)


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
#   JOBID USER STATE NODE CPU% MEM% #GPU GPU% GMEM% SM_ACT% TENSOR% DRAM% POWER_W RUNTIME
#
# NODE is the node count (a name would truncate on a multi-node job and the row is
# already per-job, not per-node); #GPU is the allocated GPU count. The jobstats group
# is shown by every view, since CPU% next to SM_ACT% is the comparison that tells
# you whether a GPU job is actually CPU-bound -- previously no single view had both.
# OCC% moved to the "all" group (dcgm.py's METRICS) -- --all-metrics only, not the
# default -- so it is absent here; this list mirrors DEFAULT_SPECS/GPU_SUMMARY_SPECS.
SUMMARY_COLUMNS: List[Column] = [
    Column("JOBID", "{:<12}", "id"),
    Column("USER", "{:<12}", "id"),
    Column("STATE", "{:<9}", "id"),
    Column("NODE", "{:<5}", "jobstats"),
    Column("CPU%", "{:<6}", "cpu"),
    Column("MEM%", "{:<6}", "cpu"),
    Column("#GPU", "{:<5}", "gpu"),
    Column("GPU%", "{:<6}", "gpu"),
    Column("GMEM%", "{:<7}", "gpu"),
    Column("SM_ACT%", "{:<8}", "dcgm"),
    Column("TENSOR%", "{:<8}", "dcgm"),
    Column("DRAM%", "{:<7}", "dcgm"),
    Column("POWER_W", "{:<8}", "dcgm"),
    Column("RUNTIME", "{:<12}", "id"),
]

# Columns that already have a fixed slot above, so the profiling block must never add a
# second one for them. Unconditional on purpose: their *position* is a property of this
# table, not of which exporter happens to win them.
FIXED_POSITION_HEADERS: Tuple[str, ...] = ("GPU%", "GMEM_GB", "GMEM_TOTAL_GB", "GMEM%")


# `--show` -> the column it adds and the JobRow field that fills it. Opt-in rather than
# always on because these are wide: measured over 31,029 real jobs, account runs to 23
# characters (mean 16.7) and partition to 22 (mean 11.0), which is ~36 characters on a
# row already near 90 and far wider under --all-metrics.
#
# The widths sit near those means rather than at the maxima, so most rows line up and
# the long tail overflows. Overflow rather than truncation because this table *streams*
# -- a width is fixed before the first row is seen, so there is nothing to size to --
# and an overflowing cell still gets its separator from `_line`'s " ".join, so it
# misaligns the row without ever running into the next value.
#
# All in the "id" group, which is what puts them in every view: cols_for passes id
# columns through unconditionally, so --cpu and --gpu need no further thought.
EXTRA_ID_COLUMNS: Dict[str, Tuple[Column, str]] = {
    "account": (Column("ACCOUNT", "{:<20}", "id"), "account"),
    "partition": (Column("PARTITION", "{:<16}", "id"), "partition"),
    "name": (Column("NAME", "{:<16}", "id"), "name"),
    "cluster": (Column("CLUSTER", "{:<10}", "id"), "cluster"),
}

# The vocabulary `--show` accepts, plus the word for all of it. Named here rather than
# in cli because the columns are a rendering fact; cli validates against this so the two
# cannot disagree about what is spellable.
SHOW_KEYWORDS: Tuple[str, ...] = tuple(EXTRA_ID_COLUMNS)
SHOW_ALL = "all"

# What `all` selects, which is deliberately not every keyword: `name` is excluded.
# A job name is free text -- often templated, frequently longer than the account and
# partition put together, and carrying no information the reader is scanning a *table*
# for. `all` is meant to be the useful wide view, not the widest possible one, so the
# name stays available as `--show name` for whoever actually wants it.
SHOW_ALL_KEYWORDS: Tuple[str, ...] = tuple(k for k in SHOW_KEYWORDS if k != "name")


def resolve_show(chosen) -> Tuple[str, ...]:
    """Selected keywords in :data:`SHOW_KEYWORDS` order, whatever order they were typed.

    Fixed rather than as-given so two people running the same report with the flags
    written differently get the same columns in the same places -- a table whose column
    order depended on typing order could not be diffed against itself.

    ``all`` means :data:`SHOW_ALL_KEYWORDS`; naming `name` alongside it still adds it.
    """
    wanted = set(chosen or ())
    if SHOW_ALL in wanted:
        wanted.update(SHOW_ALL_KEYWORDS)
    return tuple(key for key in SHOW_KEYWORDS if key in wanted)


def extra_id_columns(chosen) -> List[Column]:
    return [EXTRA_ID_COLUMNS[key][0] for key in resolve_show(chosen)]


def extra_id_cells(job, chosen) -> Dict[str, str]:
    """``{header: value}`` for the selected extras, read off a :class:`JobRow`."""
    cells = {}
    for key in resolve_show(chosen):
        column, attribute = EXTRA_ID_COLUMNS[key]
        cells[column.header] = str(getattr(job, attribute, "") or "-")
    return cells


def _block_column(header: str, index: Optional[int] = None) -> Column:
    """One profiling column, at the width both tables use.

    Shared rather than written twice: the summary table and the detail table show the same
    block, so a width rule copied into each is two places to change to keep one view
    consistent with itself.
    """
    return Column(header, "{:<%d}" % max(7, len(header) + 1), "dcgm", index)


def summary_columns(specs: Optional[List[MetricSpec]] = None, show=()) -> List[Column]:
    """:data:`SUMMARY_COLUMNS` with its DCGM block taken from ``specs``.

    ``show`` adds the :data:`EXTRA_ID_COLUMNS` selected by ``--show``, spliced in after
    USER so the identity reads left to right: who ran it, under what, then where. They
    land here rather than in a view of their own, which is what makes ``--all-metrics``
    need no separate handling -- that flag varies only the profiling block below, and
    the identity columns in front of it are the same either way.

    The identity and jobstats columns are fixed; only the profiling block varies, which
    is what lets `--all-metrics` widen the table without becoming a different view.
    The fixed columns are dropped from the block whatever source serves them -- one
    number deserves one column, and GPU%/GMEM% already have theirs. Deliberately not
    the jobstats-backed set, which shrinks under ``--gpu-source dcgm`` (GPU% stops being
    jobstats-backed) and would give GPU% a second column beside its fixed one. Which
    source *fills* the fixed cell is settled in :meth:`SummaryRenderer.add`.
    """
    extra = extra_id_columns(show)
    block = ([] if specs is None else
             [_block_column(header) for _key, header, _dec in columns_for(specs)
              if header not in FIXED_POSITION_HEADERS])
    out: List[Column] = []
    for col in SUMMARY_COLUMNS:
        if col.group == "dcgm":
            if specs is None:
                out.append(col)     # the fixed block, unchanged
                continue
            out.extend(block)
            block = []          # splice the whole block in at the first dcgm slot
        else:
            out.append(col)
        if col.header == "USER":
            out.extend(extra)
    return out


# GPU and NODE are the two levels a detail row can be about, and they differ in one cell:
# cell 1 is a card's minor number at gpu level and the count of cards pooled at node level,
# because a pooled row has to say what it pooled. Everything else is identical, which is
# why the two share a renderer. The names come from models -- jobscope.rows reads the same
# three to decide which per-unit tuples to build.
_UNIT_COLUMN = {GPU_LEVEL: Column("GPU", "{:<4}", "gpu", 1),
                NODE_LEVEL: Column("#GPU", "{:<5}", "gpu", 1)}

# How each prefix cell is displayed. The *which* and the *order* come from
# jobstats.UNIT_HEADERS -- the columns a stored summary can answer per unit -- so the row
# builder and this column list cannot drift; only the width and the colour group are
# stated here, because those are presentation and the row builder has no opinion on them.
#
# This set is fixed by the jobstats blob, not by jobscope's config: a site cannot add a
# prefix column the way it adds a profiling one, because there is no per-node field in
# the stored summary for it to come from. A site metric measured by an exporter reaches
# the detail view through the profiling block instead -- see detail_gpu_headers.
_PREFIX_DISPLAY = {
    "CPU%":     ("{:<7}", "cpu"),
    "CPU-MEM":  ("{:<16}", "cpu"),
    "GPU%":     ("{:<7}", "gpu"),
    "GPU-MEM":  ("{:<16}", "gpu"),
    "GMEM%":    ("{:<7}", "gpu"),
}

# Positions in a detail row, named rather than repeated as literals. The two identity
# cells come before the measured ones at both levels.
_NODE_INDEX = 0
_GPU_INDEX = 1
_FIRST_CELL = 2

# Elapsed sits at the cell after the prefix -- before the profiling block -- while
# printing *last*. Column.index decouples the two, and using it here is what keeps this
# cell's position from depending on how many profiling columns there are. Appending it
# after the block instead would make its index vary with --gpu-source, which is precisely
# the bug detail_columns() exists to prevent. Group "id", so it shows in every view.
_RUNTIME_INDEX = _FIRST_CELL + len(UNIT_HEADERS)
_RUNTIME_COLUMN = Column("RUNTIME", "{:<12}", "id", _RUNTIME_INDEX)


def detail_prefix(level: str = GPU_LEVEL) -> Tuple[Column, ...]:
    """The identity and jobstats cells for ``level``, without elapsed.

    Elapsed is not here because it is not a jobstats cell: :func:`detail_columns` appends
    it after the profiling block, which is where it prints.
    """
    return ((Column("NODE", "{:<16}", "id", _NODE_INDEX),
             _UNIT_COLUMN.get(level, _UNIT_COLUMN[GPU_LEVEL]))
            + tuple(Column(header, *_PREFIX_DISPLAY[header], _FIRST_CELL + i)
                    for i, header in enumerate(UNIT_HEADERS)))


def detail_gpu_headers() -> List[str]:
    """The profiling columns ``--per-gpu`` appends, in the order they are written.

    One source of truth for two callers -- :func:`detail_columns`, which says where each
    cell goes, and :func:`extend_detail_row`, which puts it there. They used to be a
    hand-written list of four and a live read of the catalog's headers, and the two
    disagreed the moment anything moved a column off the jobstats summary.

    ``catalog().headers`` is not "the dcgm columns": it is the default-group columns
    that still need a query, i.e. whatever the summary did *not* win (see dcgm.py). So
    it **grows** as an exporter is promoted -- four under the default order, five under
    ``--gpu-source dcgm`` once GPU% stops being summary-backed, seven under ``nvml``
    which also takes the GPU-memory pair. A fixed four columns fed by a variable-length
    block printed every value one slot to the left of its own header and dropped the
    last one, and since cell_band() grades by header, a DRAM% reading landed under
    POWER_W and was judged against its watt floor -- a working card reported as idle.

    The fixed-position headers drop out for the reason :func:`summary_columns` drops
    them: the prefix above already carries a column for each, and one number deserves
    one column. Which source *fills* those prefix cells is a separate question this does
    not answer -- they come from the stored summary either way (see
    :func:`extend_detail_row`).
    """
    return [h for h in dcgm.catalog().headers if h not in FIXED_POSITION_HEADERS]


def detail_columns(level: str = GPU_LEVEL) -> List[Column]:
    """:func:`detail_prefix` plus one column per :func:`detail_gpu_headers` entry.

    The indices are assigned by enumeration rather than written down, which is what
    makes them agree with the row by construction instead of by hand. Widths use
    :func:`summary_columns`' expression, which reproduces the four hand-picked ones
    exactly (SM_ACT%/TENSOR%/POWER_W 8, DRAM% 7), so default output is unchanged.

    The profiling block is identical at both levels -- the same metrics, pooled or not --
    so ``level`` reaches only the identity prefix. That is the whole reason one renderer
    serves both.

    Ordered with elapsed *last* even though it is row cell 7, so the block sits between
    the jobstats cells and it: the reading order is identity, then measurements, then how
    long they were measured over.

    Takes no spec list, unlike :func:`summary_columns`: ``--all-metrics`` deliberately
    does not widen this view. What is being fixed here is a block whose width varied
    without anyone asking; widening it on request is a different decision.
    """
    block = [_block_column(header, _RUNTIME_INDEX + 1 + offset)
             for offset, header in enumerate(detail_gpu_headers())]
    return list(detail_prefix(level)) + block + [_RUNTIME_COLUMN]

SUMMARY_DESCRIPTIONS: List[Tuple[str, str, str]] = [
    ("CPU%", "summary (cgroup CPU-seconds)",
     "Average CPU-core utilization: 100 x CPU-seconds used / (elapsed x allocated cores). "
     "100% means every allocated core was busy for the whole job."),
    ("MEM%", "summary (cgroup RSS)",
     "Peak host (CPU) memory used / memory allocated, as a percent."),
    ("GPU%", "summary (nvidia_gpu_duty_cycle)",
     "GPU duty cycle averaged over the job's GPUs: fraction of time at least one kernel ran. "
     "Coarse: says the GPU was occupied in time, not how hard. Use SM_ACT%/OCC% (gpu view) for that."),
    ("GMEM%", "summary (nvidia_gpu_memory_used)",
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
_SGR = dict(Palette().sgr())
_RESET = "\033[0m"


def set_palette(palette: "Palette") -> None:
    """Install the colours :func:`tint` paints with.

    Module state because tint() is called from a dozen places that hold no config
    between them, and the palette is one per process -- cli._apply_config calls this
    once, before anything renders. A caller that never does keeps the defaults.
    """
    _SGR.clear()
    _SGR.update(palette.sgr())
# Strips the above, so a rule can be measured against the characters a reader sees
# rather than the escape bytes carrying the colour.
_ESC_RE = re.compile(r"\033\[[0-9;]*m")


@dataclass
class RenderOptions:
    """Flags shared by the rendering functions."""

    view: str = "all"
    show_dcgm: bool = False
    csv: bool = False
    header: bool = True
    # Extra identity columns from --show, already resolved to a fixed order by
    # report.resolve_show. See EXTRA_ID_COLUMNS for why they are opt-in.
    show_ids: Tuple[str, ...] = ()
    # Weight the mean by allocated resource-time (GPU-hours, core-hours) instead
    # of by GPU count. Valid only where each job's value already covers its whole
    # runtime -- a finished job's summary, or running --runtime-avg. On an instantaneous
    # running snapshot every value is the same moment, so scaling one by two days
    # of elapsed time would claim that instant represents those two days.
    # Whether every value already spans its job's whole runtime, which is one condition
    # with two consequences: the weights become resource-seconds (rendering as hours),
    # and averaging_note can say the figures are runtime means. They were separate
    # expressions once -- this one derived from the selection mode, the note's from the
    # records -- and they disagreed for `jobscope -j ID` on a running job: mode said
    # weight by resource-time, the records said the value was a single scrape. Only one
    # of them can be right about one table, so there is only one of them. cli reads it
    # straight off select.Resolved.folded, the same value sampled_pair() is given.
    time_weighted: bool = False
    # --per-gpu only: report just this node's GPUs. The per-job table's NODE column
    # is a count, so there is no name there to match against.
    nodename: Optional[str] = None
    # --ts only: report just these GPU ids (--gpuid 0,1). Strings, because a MIG
    # instance is "0.1" rather than a number.
    gpu_ids: Tuple[str, ...] = ()
    # Show the efficiency bars section. On by default: it is the fastest read in
    # the block, and behind a flag it was rarely seen. --no-plot switches it off.
    plot_avgeff: bool = True
    # --ts only: emit just the last N seconds of each job's series, narrowing the
    # range queries rather than filtering rows afterwards.
    window: Optional[int] = None
    # --verify only: add the per-rung ladder table under the verdict block. Off by
    # default because the block answers the question and the ladder is the evidence
    # behind it -- which is worth a flag, not worth twenty rows of a four-GPU job's
    # screen every time.
    verify_full: bool = False
    # Tint %-metric cells by their threshold band. Off unless the caller has
    # established that the destination is a terminal that wants colour.
    color: bool = False
    thresholds: Optional["Thresholds"] = None
    # How many jobs each Problem-jobs "Wasteful" row lists ([defaults]
    # worst_jobs), and the elapsed time at which an entry is highlighted
    # ([defaults] long_running, in seconds here).
    worst_jobs: int = DEFAULT_WORST_JOBS
    long_running: int = LONG_RUNNING
    # Which sections of the summary block print, and in what order ([report]
    # sections). --no-plot still drops the bars regardless: a flag beats a file.
    sections: Tuple[str, ...] = REPORT_SECTIONS
    # --ts only: emit CPU%/MEM% alongside the GPU/DCGM columns in one series --
    # the default --ts view. cli.py resolves this from --cpu/--dcgm; view=="cpu"
    # takes precedence over this when combined is False (cpu-only), and combined
    # takes precedence over view=="cpu" when both are set (see select.emit_timeseries).
    combined: bool = False
    # --ts only: which cgroup metrics the host series carries ([metrics.cgroup]).
    # None keeps the default two, CPU%/MEM%, which is what every view showed before
    # the catalog existed.
    cgroup_specs: Optional[List["CgroupSpec"]] = None


def no_such_node(nodename: str, seen) -> JobscopeError:
    """The error for a ``--nodename`` that matched nothing.

    Naming the nodes the selection *did* touch is the whole value of it: an empty
    report reads as an idle node rather than a typo. Shared by every view that takes
    the filter -- the per-GPU table and both ``--ts`` emitters -- so a mistyped name
    gets the same answer whichever one you were running.
    """
    return JobscopeError("no rows for node %r in this selection; it ran on: %s"
                         % (nodename, ", ".join(sorted(seen)) or "(none)"))


# The identity block every --ts row carries, before the metric columns. USER is here
# so a partition-wide report can name whose job is idle; plot.ID_COLS already lists it,
# so charts ignore it as an identity column.
TS_ID_COLUMNS = ["JOBID", "USER", "EPOCH", "TIME", "NODE", "GPU", "MODEL"]


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


# The built-in edges, for the callers that need *some* table to read numbers out of
# when the options carry none. Distinct from options.thresholds being None, which
# several call sites use to mean "do not grade at all" and "there is no POWER_W
# floor" -- see cell_band below and classify()'s power cap.
_DEFAULT_BANDS = Thresholds()

# Stand-in header for grading a bare share -- a percentage belonging to no column
# of its own, so it takes the default edges rather than any metric's. POWER_W's
# USED cell needs one: the share is a percent even though the metric is watts.
_SHARE_HEADER = "%"


def _bands(options: "RenderOptions") -> Thresholds:
    """This run's band table, or the built-in one when the caller supplied none."""
    return options.thresholds or _DEFAULT_BANDS


def cell_band(options: "RenderOptions", header: str, cell, model: str = "") -> str:
    """The grade for a rendered cell, or ``""`` when it is not a graded metric.

    Shared by the per-job table and ``--per-gpu`` so the two cannot disagree about a
    colour, the same reason ``plot`` calls ``Thresholds.grade`` rather than keeping its
    own copy.

    The trailing ``%`` matters: the detail view writes its cells as "11.4%" where the
    summary writes "11", and a bare ``float()`` rejects the former -- which is why
    those columns were silently the only untinted ones.

    ``model`` names the GPU whose cell this is, for ``POWER_W`` alone: its floor is
    hardware, so an idle RTX PRO 6000 at 165 W and a working V100 at 60 W cannot be
    judged by one number. Empty means "unknown", which falls back to the global.
    """
    if not options.color or options.thresholds is None:
        return ""
    value = cell_value(cell)
    if value is None:
        return ""
    return options.thresholds.for_model(model).grade(header, value)


BAR_WIDTH = 34


def bar_lines(items, indent: str = "  ") -> List[str]:
    """Horizontal bars from ``(label, percent, band)`` triples, or 4-tuples with a
    trailing cell -- a duration beside a share, where the share alone does not size it.

    One primitive for the per-job summary and the per-GPU detail charts, so they look
    identical rather than merely similar. A nonzero value always draws at least one
    block and reads "<1%" rather than "0%", because an empty bar beside a "0%" and a
    filled one beside it are each self-contradictory.
    """
    if not items:
        return []
    items = [one if len(one) == 4 else tuple(one) + ("",) for one in items]
    label_width = max(len(label) for label, _v, _b, _t in items)
    trailing = max(len(text) for _l, _v, _b, text in items)
    out = []
    for label, value, band, text in items:
        filled = max(0, min(BAR_WIDTH, int(round(value / 100.0 * BAR_WIDTH))))
        if value > 0 and filled == 0:
            filled = 1
        run = "\u2588" * filled
        value_text = "<1%" if 0 < value < 0.5 else "%d%%" % round(value)
        out.append(("%s%*s  %4s  %s%s%s" % (
            indent, label_width, label, value_text,
            tint(run, band) if run else run,
            "\u2591" * (BAR_WIDTH - filled),
            ("  %*s" % (trailing, text)) if trailing else "")).rstrip())
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


def tint(text: str, role: str) -> str:
    """``text`` wrapped in ``role``'s colour, or unchanged when it has none.

    ``role`` is whatever the call site holds -- a bucket identifier
    (``red``/``yellow``/``green``) for a graded cell, a tier name for a --eff
    heading, ``long_running`` for a Wasteful-row entry. :meth:`Palette.sgr` resolves
    all three, so no caller has to translate.

    Callers pass the *padded* cell: inserting the escapes first would make str.format
    count them toward the column width and shift every later column.

    An unknown role leaves the text alone rather than raising -- a palette is
    decoration, and losing a colour is not worth losing the report over.
    """
    escape = _SGR.get(role, "") if role else ""
    return escape + text + _RESET if escape else text


# --- the timeline strip -------------------------------------------------------
#
# One character per time bucket, drawn against a *fixed* axis (job_eff.full_scale) so
# two jobs read the same way and a flat series reads flat. jobscope.plot.braille_spark
# is the wrong primitive here despite being the closest one: it scales to the values'
# own min..max, which is right for a chart that prints its axis and wrong for a strip
# that has none -- a job that never rose above 0.6% would draw as a full-range sawtooth.
#
# The ramp is one space, one full stop and seven blocks, and the three kinds of cell it
# distinguishes are the whole point: nothing measured, nothing that cleared the cutoff,
# and how hard it worked.
#
# A full stop rather than U+2581 for the idle rung. Eight block heights sit about two
# pixels apart, which is fine for reading *shape* -- where it is high, where it collapses
# -- and useless for reading one cell. But one distinction has to survive monochrome, and
# it is idle versus working: it is the most decision-relevant bit in the view, and
# U+2581 and U+2582 differ by a pixel. A full stop sits on the baseline and is
# unambiguously not a block at any font. In colour it carries its band as well, so the
# bit is stated twice -- the redundancy the GOOD/OK/BAD headers argue for below.
STRIP_GAP = " "
STRIP_IDLE = "."
STRIP_BLOCKS = "▂▃▄▅▆▇█"
STRIP_RAMP = STRIP_GAP + STRIP_IDLE + STRIP_BLOCKS
# Below this a strip says nothing about shape, so it is not worth narrowing further to
# fit a label; the caller wraps or drops instead.
MIN_STRIP_CELLS = 20

# Tick spacings, coarsest wins: the smallest that leaves at least ten cells between
# ticks, so a five-character HH:MM never touches its neighbour.
_TICK_STEPS = (60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400)


def strip_level(mean: Optional[float], peak: Optional[float],
                limit: Optional[float], scale: Optional[float]) -> int:
    """Which rung of :data:`STRIP_RAMP` a cell draws at: 0 gap, 1 idle, 2..8 working.

    A cell may only draw as idle when *nothing in it* cleared the cutoff, which is why
    the peak is asked for alongside the mean. Without that rule a three-minute cell
    holding 100, 0, 0 draws identically to one holding 0, 0, 0 -- and telling those two
    apart is the entire reason this view exists. The cost is that the lowest block spans
    a wider range of means than the rest, which is the right trade: the ramp is read for
    where it collapses, not for its exact height.

    A metric with no cutoff has no idle rung to fall to, and scales normally.
    """
    if mean is None or not scale:
        return 0
    if limit is not None and (peak is None or peak < limit):
        return 1
    return max(2, min(8, 1 + int(round(mean / scale * 7))))


def level_strip(means, peaks, limit: Optional[float], scale: Optional[float],
                options: Optional["RenderOptions"] = None, header: str = "",
                model: str = "") -> str:
    """One tinted character per bucket, from parallel mean and peak lists.

    Tinted per *run* of equal band rather than per cell: a hundred cells at ten escape
    bytes each is a kilobyte a line, and ``_ESC_RE`` strips escapes before anything
    measures a width, so the grouping is invisible to every caller.
    """
    cells = []
    for mean, peak in zip(means, peaks):
        level = strip_level(mean, peak, limit, scale)
        band = "" if (options is None or level == 0) else cell_band(
            options, header, mean, model)
        cells.append((STRIP_RAMP[level], band))
    out = []
    for band, group in itertools.groupby(cells, key=lambda cell: cell[1]):
        text = "".join(char for char, _band in group)
        out.append(tint(text, band) if band else text)
    return "".join(out)


def strip_cells(width: int, label_width: int, span: int, step: int,
                indent: int = 2) -> int:
    """How many cells to draw a ``span``-second series in, never finer than ``step``.

    The second clamp is the one that is easy to miss: at 116 columns a forty-minute job
    would be given a hundred cells for forty samples, and sixty of them would be gaps
    that are not gaps -- an exporter outage drawn where the data is complete.
    """
    room = max(MIN_STRIP_CELLS, width - label_width - indent)
    return max(1, min(room, span // max(1, step) + 1))


def strip_axis(edges, indent: int = 0) -> List[str]:
    """``[tick row, label row]`` for the cells ``edges`` describes.

    Built from the same edge list the strip was drawn from rather than from a measured
    width, so the labels sit under their own cells structurally -- which is what keeps
    the axis correct once the strip above it is full of colour escapes.
    """
    if not edges:
        return []
    start, end = edges[0][0], edges[-1][1]
    cell = max(1, (end - start) // len(edges))
    every = next((s for s in _TICK_STEPS if s >= 10 * cell), _TICK_STEPS[-1])
    rule = ["─"] * len(edges)
    labels = [" "] * len(edges)
    # Ticks land on aligned wall-clock instants -- 14:00, 14:30 -- rather than on
    # multiples of the fetch's own start, which would put them at 14:07 and 14:37 and
    # make two runs of the same job impossible to lay beside each other.
    stamp = start - start % every
    while stamp <= end:
        if stamp >= start:
            at = next((i for i, (lo, hi) in enumerate(edges) if lo <= stamp < hi),
                      len(edges) - 1)
            rule[at] = "┬"
            text = time.strftime("%H:%M", time.localtime(stamp))
            # All of it or none of it: half a timestamp at the right edge reads as a
            # different time rather than as a truncated one.
            if at + len(text) <= len(labels):
                labels[at:at + len(text)] = list(text)
        stamp += every
    pad = " " * indent
    return [pad + "".join(rule), (pad + "".join(labels)).rstrip()]


def fmt_context(label: str, value: str) -> str:
    """A '  Label:     value' context line."""
    return "  %-11s%s" % (label + ":", value)


def cols_for(columns: List[Column], view: str, dcgm: bool = False) -> List[Column]:
    """The columns to show for the chosen view.

    ``all`` (the default) shows everything; ``--cpu`` and ``--gpu`` narrow it to one
    resource. ``id``/``jobstats`` columns identify the row and appear in every view. The
    DCGM block needs Prometheus, so it belongs to the views that carry GPU columns.
    """
    gpu_views = ("all", "gpu")
    out = []
    for col in columns:
        group = col.group
        if (group in ("id", "jobstats")
                or (group == "cpu" and view in ("all", "cpu"))
                or (group == "gpu" and view in gpu_views)
                or (group == "dcgm" and dcgm and view in gpu_views)):
            out.append(col)
    return out


def narrowing_pairs(nodename: Optional[str], gpu_ids) -> List[Tuple[str, str]]:
    """Header lines naming a ``--nodename`` / ``--gpuid`` narrowing, if any.

    Load-bearing rather than decorative. On ``--per-gpu`` and ``--ts`` a filter is
    self-evident -- the rows that remain carry the node and the card. The summary has
    no such row: narrowed, it prints one line of numbers that looks exactly like the
    whole job's, and a reader who scrolled past the command would have no way to tell
    that GPU% 43 is one node of two. So the narrowing says so.
    """
    pairs = []
    if nodename:
        pairs.append(("Node", "%s only" % nodename))
    if gpu_ids:
        pairs.append(("GPUs", "%s only (per node)" % ", ".join(str(g) for g in gpu_ids)))
    return pairs


def context_pairs(context: ReportContext,
                  specs: Optional[List] = None,
                  host_specs: Optional[List] = None,
                  average: bool = False) -> List[Tuple[str, str]]:
    """Context lines for the header block.

    With explicit JOBIDs the -u/-A/-p filters are bypassed, so show the jobs'
    actual owner(s) rather than the (misleading) default user, and drop the filter
    lines. :func:`jobscope.rows.build_context` is what decides which case this is.
    """
    if context.explicit_jobids:
        user_val = ", ".join(context.owners) or "(explicit job IDs)"
        pairs = [("User", user_val), ("Select", context.desc)]
        return (pairs + source_pair(specs, host_specs=host_specs)
                + sampled_pair(specs, context.unfinished, average, host_specs))
    pairs = [("User", context.user)]
    if context.account:
        pairs.append(("Account", context.account))
    if context.partition:
        pairs.append(("Partition", context.partition))
    pairs.append(("Select", context.desc))
    if context.window:
        pairs.append(("Window", context.window))
    # A window selection holds only finished jobs, so the fold is unconditional there.
    return (pairs + source_pair(specs, host_specs=host_specs)
            + sampled_pair(specs, context.unfinished, average, host_specs))


def gpu_source_line(specs: Optional[List] = None, have_jobstats: bool = True,
                    host_specs: Optional[List] = None) -> str:
    """Which source served which column, for the header block.

    The header already restates the window it actually scanned, so a report cannot
    claim a range it did not read; this is the same promise about *provenance*. It
    matters because the default GPU block is genuinely mixed -- GPU% and the memory
    pair out of the jobstats summary, the activity columns out of dcgm-exporter -- and
    a reader comparing two clusters, or two runs either side of a `--gpu-source`,
    has no other way to tell which numbers moved because the source did.

    Named per column rather than as one word for the same reason
    :mod:`jobscope.extra_metric` refuses to substitute silently: a mixed set
    described as "dcgm" would be wrong about half its own columns.

    ``have_jobstats=False`` for the running view, where the claim would otherwise be
    false in the other direction: Slurm writes the jobstats summary at job *end*, so a running
    job's GPU% is measured by an exporter no matter what the preference says, and
    naming the jobstats summary there would credit a source that had nothing to give.

    Covers the host columns as well, since CPU%/MEM% have the same choice between the
    stored summary and an exporter and the reader has no more way to tell for those than
    for GPU%. Host columns first, matching the order they print in.

    States which source each column *resolved to*, not which one returned data -- a
    column whose source had nothing still prints "-", and the two lines read together:
    asked cgroup, cgroup was empty. That is deliberately more useful than omitting the
    name, because "we queried an exporter this cluster does not run" is exactly the
    diagnostic a port needs, and the alternative hides it.

    An **empty** ``specs`` is not the same as ``None`` here, and the difference is
    ``--no-dcgm`` against ``--cpu``. ``None`` means the view prints no GPU column at
    all, so there is nothing to attribute. ``[]`` means it prints them but collected
    none of them: GPU% and the memory pair come out of the stored jobstats summary,
    which arrived with sacct and cost no query, and they are on screen whether or not
    an exporter was asked anything. Saying nothing about them would leave two
    populated columns unattributed, which is the one thing this line exists to prevent.
    """
    from . import cpu, dcgm
    from . import source as source_module
    if specs is None and not host_specs:
        # --cpu with no host list either: nothing collected, so nothing to attribute.
        return ""
    host, gpu = cpu.catalog(), dcgm.catalog()
    gpu_resolution = (gpu.resolved if have_jobstats
                      else source_module.resolve(gpu.metrics, gpu.preference))
    # The columns still on screen with nothing collected for them -- see above. Only
    # with a stored summary to serve them: a running job has none, so under
    # have_jobstats=False an empty list really does mean nothing was collected.
    if specs == [] and have_jobstats:
        specs = [spec for spec in gpu.metrics
                 if spec.column in set(dict(gpu_resolution.by_source()).get("jobstats", ()))]
    per_source: Dict[str, List[str]] = {}
    for resolution, wanted in (
            (host.resolved if have_jobstats else source_module.resolve(
                host.candidates, host.preference), host_specs),
            (gpu_resolution, specs)):
        if not wanted:
            continue
        shown = {spec.column for spec in wanted}
        for name, columns in resolution.by_source():
            kept = [c for c in columns if c in shown]
            if kept:
                per_source.setdefault(name, []).extend(kept)
    return ";  ".join("%s <- %s" % (" ".join(columns), name)
                      for name, columns in per_source.items())


def source_pair(specs, have_jobstats: bool = True,
                host_specs: Optional[List] = None) -> List[Tuple[str, str]]:
    """The Source context line, or nothing when the view collected nothing."""
    line = gpu_source_line(specs, have_jobstats, host_specs)
    return [("Source", line)] if line else []


# Named here rather than imported from cli: this module is below it in the layering
# (see tests/test_layering.py), and three separate lines of output have to spell the
# flag the same way as each other.
RUNTIME_AVG_FLAG = "--runtime-avg"


def sampled_pair(specs, unfinished: bool, average: bool,
                 host_specs: Optional[List] = None) -> List[Tuple[str, str]]:
    """The Sampled context line: over what span the GPU numbers were taken.

    ``Source`` says *where* a column came from; this says *when*, which is the other
    half of what a number means and the half a reader could not previously recover. A
    running job has two legitimate answers -- the newest scrape, or a mean over a
    runtime that is still growing -- and with nothing on screen to distinguish them,
    two views of one job reporting different figures looks like a bug in the tool
    rather than a difference in the question.

    ``unfinished`` when the selection holds any job that has not ended; ``average``
    when ``--runtime-avg`` folds those anyway. A window selection is finished by construction
    (see slurm.UNFINISHED_STATES and the ``-s`` filter), so it passes False and gets
    the one-clause form.

    Host columns are named only when they are present *and* the GPU answer differs
    from theirs, because that is the only combination where the line could otherwise
    be read as a claim about CPU%/MEM% -- those are cumulative whatever the window
    (see :func:`jobscope.cpu.host_stats`) and never vary with this.
    """
    if not specs:
        # --cpu, or nothing collected: no GPU column, so no GPU window to describe.
        return []
    folded = "averaged over each job's whole runtime"
    if not unfinished or average:
        return [("Sampled", "GPU metrics " + folded)]
    # Names GPU explicitly: the host columns do not vary with this, and a bare "newest
    # scrape" would read as a claim about every number on the row.
    text = ("GPU metrics at their most recent scrape, not averaged -- add %s for the"
            " runtime mean" % RUNTIME_AVG_FLAG)
    if host_specs:
        text += "; CPU%/MEM% cumulative either way"
    return [("Sampled", text)]


def averaging_note(folded: bool, pooled: bool = True) -> str:
    """How the summary block's numbers were reduced -- the section's opening line.

    ``folded`` is :attr:`RenderOptions.time_weighted`: whether every value already spans
    its job's whole runtime. One condition, two things a reader has to know --

    * what each job contributed: a mean over its runtime, or just its most recent scrape.
      Same rule as the ``Sampled`` line (:func:`sampled_pair`); one live partition reads
      GPU% 77% / SM_ACT% 45% instantly and 71% / 57% folded, so a reader comparing two
      reports needs to know which they hold.
    * how those were pooled: weighted by resource-time, or by the resources each job
      holds now. This was the only one ever on screen, as the ``Used/GPU`` versus
      ``Used/GPU-hr`` label, and only to a reader who already knew what the suffix meant.

    First line of the section rather than buried in the legend: it describes the numbers
    themselves, where the rest of the legend describes how they are graded, and a reader
    who stops after one line should have the one that changes what the figures mean.

    ``pooled`` is False for a single job, where there is no weighting to describe and a
    clause about long jobs outweighing short ones would describe a mean of one.
    """
    subject = "each job's" if pooled else "this job's"
    if folded:
        # A finished record, or running under the flag. Nothing left to offer, so no
        # pointer: the flag is already in effect or already implied.
        across = (" Pooled across jobs by resource-time, so a 10-hour job weighs ten"
                  " times a 1-hour one." if pooled else " USED is resource-time.")
        return ("Averaged over time: every value below is %s mean over its whole"
                " runtime.%s" % (subject, across))
    across = (" Pooled across jobs by the cores/GB/GPUs each holds now, not by time."
              if pooled else "")
    return ("Not averaged over time: every value below is %s most recent scrape, as of"
            " now.%s Add %s for runtime means."
            % (subject, across, RUNTIME_AVG_FLAG))


def bars_note(folded: bool, pooled: bool = True) -> str:
    """The same caveat above the efficiency bars, because "Average" does not say average
    over what.

    Above rather than below: it qualifies the bars, and a reader scanning the fastest
    section in the block reaches the picture first and stops. The bars are section 1's
    USED column drawn sideways -- the same number, so the same reduction -- which makes
    this the section most likely to be quoted without the table it came from.
    """
    # The span only. How the bars were *pooled* is section 1's business and is stated
    # there; repeating it here cost a third wrapped line above a 34-column chart, and the
    # question this answers is "averaged over what", which the section title raises and
    # does not answer. The flag is named even so: this is the section a reader is most
    # likely to have arrived at first, and it is the one that has to be actionable.
    if folded:
        return ("bars are the USED share above -- averaged over %s whole runtime"
                % ("each job's" if pooled else "the job's"))
    return ("bars are the USED share above -- most recent scrape; add %s to average"
            " over time" % RUNTIME_AVG_FLAG)


def verdict_note() -> str:
    """The caveat under a single job's Efficiency verdict, when it grades one scrape.

    The most quotable line in the report and the one least able to defend itself: a
    categorical word. A job cycling 0-100% between scrapes is "good" or "wasteful"
    depending only on which second the report ran, where the same job's runtime mean is
    stable. Every other section says what it averaged; this one asserted a judgement.
    """
    return ("Graded on the most recent scrape, not the whole run -- add %s to grade the"
            " runtime mean." % RUNTIME_AVG_FLAG)


def exporter_prefix_cells(values) -> Dict[str, str]:
    """The prefix cells an exporter has won, spelled the way the summary spells them.

    ``GPU%``, ``GPU-MEM`` and ``GMEM%`` have a fixed slot fed by ``jobstats_detail``,
    which reads the stored summary and nothing else. Naming an exporter takes those
    columns out of ``RESOLVED.from_jobstats``, so ``dcgm._prefer_stored`` deliberately
    stops overwriting the queried value -- "the queried value is the answer and must not
    be overwritten by a stored one measured somewhere else". This is the other half of
    that sentence: the answer has to reach the cell. Measured on one A100 job,
    ``--gpu-source dcgm`` reported the summary's ``GPU%`` of 51 where DCGM had measured
    57.8, under a header the Source line attributed to dcgm.

    The counterpart of :meth:`SummaryRenderer.add`'s rule for the same four headers --
    see :func:`summary_columns` on why the fixed slots exist at all. Cells keep the
    summary's own formatting rather than the profiling block's, because a column that
    changed shape depending on ``--gpu-source`` would read as a different metric.

    The memory pair moves together or not at all: ``GMEM%`` is used/total, so halves from
    different sources would be a ratio belonging to neither (see :mod:`jobscope.source`).
    So both cells key off ``GMEM_GB`` -- ``GMEM%`` is derived and never appears in
    ``from_jobstats`` itself -- and the ratio is taken from the very two numbers written
    beside it rather than read from ``values``. ``dcgm._add_derived`` does put a ``GMEM%``
    there, and it is the same figure by the same formula; recomputing means the pair and
    its percentage cannot disagree even if a caller assembled ``values`` without it.

    A missing value leaves the stored cell alone rather than blanking it: if the query
    for a column an exporter won came back empty, the summary's number is still the best
    available and a dash would be a worse answer than a slightly differently-measured
    one. ``CPU%``/``CPU-MEM`` are untouched -- those are host columns, answering to
    ``[host] source`` rather than to this preference, and no per-GPU map carries them.
    """
    out: Dict[str, str] = {}
    from_jobstats = dcgm.catalog().resolved.from_jobstats
    duty = values.get("GPU%")
    if "GPU%" not in from_jobstats and duty is not None:
        # The column's own precision, plus the sign the prefix cells carry. jobstats_detail
        # spells this "%g%%" because the summary hands it a whole percent; a queried float
        # through the same "%g" is "57.8889%", which is eight characters in a seven-wide
        # column and pushes every column after it one to the right. Taking the decimals
        # from the catalog also means a site that reconfigured them gets its own.
        out["GPU%"] = format_by_header("GPU%", duty) + "%"
    if "GMEM_GB" not in from_jobstats:
        used, total = values.get("GMEM_GB"), values.get("GMEM_TOTAL_GB")
        if used is not None and total is not None:
            # Back to bytes so the one formatter jobstats_detail uses spells it, rather
            # than a second "%.1fGB" here that could drift from it.
            out["GPU-MEM"] = "%s/%s" % (bytes_to_gb(used * GIB), bytes_to_gb(total * GIB))
            if total:
                out["GMEM%"] = "%.1f%%" % (100.0 * used / total)
    return out


def detail_row_cells(unit, values, runtime: str = "-") -> tuple:
    """One rendered row: the prefix cells, elapsed at 7, then the profiling block.

    Elapsed goes in at a *fixed* cell, which is what lets its column index be a constant
    rather than a function of how wide the block is -- see :func:`detail_columns`. The
    block therefore starts at 8, not 7.

    The prefix follows ``UNIT_HEADERS`` -- the same list :func:`detail_prefix` builds its
    Columns from -- so the cells and the headers cannot drift apart.
    """
    by_header = {**unit.cells, **exporter_prefix_cells(values)}
    return ((unit.node, unit.unit)
            + tuple(by_header.get(h, "-") for h in UNIT_HEADERS)
            + (runtime,)
            + tuple(format_by_header(h, values.get(h))
                    for h in detail_gpu_headers()))


# What each graded column is a percentage *of*, as
# (label, unit, scale, pooled-row label, CSV label, weight key). CPU% is a share of
# allocated cores and MEM% of allocated host memory; everything else -- GPU%, GMEM%
# and every DCGM profiling column -- is a share of time on the allocated GPUs.
_RESOURCES = {
    "CPU%": ("Core-hours", "Cores", "cpu", "cpu"),
    "MEM%": ("GB-hours", "GB", "mem", "mem"),
    None: ("GPU-hours", "GPUs", "GPU", "gpu"),
}



# The measures the Worst rows rank by, in print order, and the two combined
# rankings: the distinct *resources*, then every measure. Read off the catalog's
# roles rather than restated here -- see jobscope.metrics for why the four, and
# for the short row-label and share-tag forms that used to be two more tables.
WORST_METRICS = metrics.headers_with_role(metrics.WORST)
COMBINED_2 = metrics.headers_with_role(metrics.RESOURCE)
COMBINED_4 = WORST_METRICS


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

    The values, not the waste shares that decide the order -- every value here is
    *under* its cutoff. Showing the values says why the job qualified; the order
    still says how much it wasted. Each is unit-suffixed (``gpu0%``, ``pw73W``) so
    a bare number never reads as a band index or rank -- putting the ``%`` before
    the tag instead ("12%gpu") would read exactly like a utilization of 12%, which
    is the opposite of what puts a job on this row, so it trails the number instead.
    """
    parts = []
    for header, value in values:
        if value is None:
            continue
        unit = "W" if tallies[header].value_unit == "W" else "%"
        parts.append("%s%d%s" % (metrics.share_tag(header), round(value), unit))
    return "%s %s" % (jid, " ".join(parts))


def _worst_slug(header: str) -> str:
    """Row-label form of a metric name, e.g. ``SM_ACT%`` -> ``SM``."""
    return metrics.label(header)


# The four columns the stored summary carries, paired with the internal keys the
# running totals are accumulated under. One pairing in one place, replacing three
# zips over parallel literal tuples that had to stay in the same order.
JOBSTATS_KEYS: Tuple[Tuple[str, str], ...] = (
    ("cpu", "CPU%"), ("mem", "MEM%"), ("gpu", "GPU%"), ("gmem", "GMEM%"))
JOBSTATS_HEADERS: Tuple[str, ...] = tuple(header for _key, header in JOBSTATS_KEYS)


def _jobstats_value(job_metrics, header: str) -> Optional[float]:
    """``header``'s value from a :class:`~jobscope.models.JobMetrics`, or None.

    None both for a header the jobstats summary does not carry (a DCGM column) and for one it
    carries without a measurement, which the caller treats the same way: look to
    Prometheus, then give up rather than invent a zero.
    """
    return job_metrics.value(header) if job_metrics is not None else None


def _resource_of(header: str, hours: bool):
    """``EfficiencyTally`` arguments for ``header``: the resource it measures.

    ``hours`` selects the resource-time form (a finished job, or ``--runtime-avg``) over
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

    WORST = DEFAULT_WORST_JOBS

    def __init__(self, header: str, thresholds: "Thresholds", label: str,
                 unit: str, scale: float = 1.0, row: str = "Used/GPU:",
                 csv_row: str = "UsedPerGPU", weight_key: str = "gpu",
                 absolute: bool = False, value_unit: str = "%",
                 worst: Optional[int] = None) -> None:
        # How many of the worst jobs this tally keeps. Per instance rather than
        # per class so one run's setting cannot leak into another's.
        self.WORST = self.WORST if worst is None else worst
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
            runtime: str = "-", duration: Optional[int] = None,
            model: str = "") -> None:
        # The model is bound per call, not per tally: one selection spans hardware,
        # and POWER_W's floor is a property of the card rather than of the column.
        band = self.band_of(value, model)
        if not band or weight <= 0:
            # Ungraded (no measurement) or unweighable: counting it would either
            # invent a utilization or give it no resource to account for.
            return
        self.bands[band][0] += 1
        self.bands[band][1] += weight
        wasted = self.waste_of(value, weight, model)
        # Used is whatever was not wasted, for both kinds of metric. For a percentage
        # that is (value/100) * weight, exactly as before. For POWER_W it is the
        # resource-time at or above the floor, which is the only reading of "used"
        # watts admit. One definition, so USED cannot mean two things.
        self.used += weight - wasted
        self.total += weight
        self.waste_total += wasted
        if self.is_wasteful(value, model):
            # Ranked by resource-time *wasted*, not held: a 100-hour job at 24% is
            # a bigger finding than a 10-hour job at 0%.
            self.worst.append(WorstJob(wasted, jobid, user, weight, value,
                                       runtime, duration))
            self.worst.sort(key=lambda item: -item[0])
            del self.worst[self.WORST:]

    def is_wasteful(self, value: Optional[float], model: str = "") -> bool:
        """Whether ``value`` clears the strict Wasteful-row cutoff.

        Stricter than the red/yellow/green band: red spans both the wasteful and
        inefficient tiers (matching --ts --eff's own colour grouping), but a
        row meant to flag the jobs actually worth a look uses the tighter
        ``wasteful`` edge alone -- the same one --ts --eff's "wasteful" tier
        means, and this column's own, since the edges are per metric. POWER_W has
        no such split (its floor is already two-band), so its own cutoff is
        unchanged. Shared by :meth:`add` (this tally's own Worst row) and
        :meth:`SummaryRenderer._note_waste` (the combined rows), so the two can
        never disagree about which jobs qualify.
        """
        if value is None:
            return False
        return (value < self.cutoff(model) if self.absolute else
               value < self.thresholds.edge("wasteful", self.header))

    def waste_of(self, value: float, weight: float, model: str = "") -> float:
        """Resource-time this job wasted, in the units the weight is in.

        For a percentage, the unused fraction of what it held. For an absolute
        metric there is no fraction to take, so a job below the cutoff wastes all of
        it and one above wastes none: for POWER_W that reads "GPU-hours spent below
        the idle floor", which is what a floor actually asserts. Grading it as a
        proportion of the cutoff instead would imply 50 W wastes twice what 100 W
        does, and watts are not utilization.
        """
        if self.absolute:
            return weight if value < self.cutoff(model) else 0.0
        return (1 - value / 100.0) * weight

    def cutoff(self, model: str = "") -> Optional[float]:
        """The watt floor for this column -- POWER_W is the only absolute metric."""
        return self.thresholds.for_model(model).power_w

    def graded(self) -> int:
        """Jobs this metric measured -- the denominator behind its Worst row.

        Differs per metric because coverage does: a finished job with no stored summary
        has no GPU% but still has DCGM data, so SM_ACT% can cover more jobs than
        GPU% over the same selection.
        """
        return sum(count for count, _weight in self.bands.values())

    def band_of(self, value: Optional[float], model: str = "") -> str:
        """This column's band for ``value``, or "" when it is not graded."""
        return self.thresholds.for_model(model).grade(self.header, value)

    def pooled_band(self) -> str:
        """The band for the USED cell -- always a percentage grade.

        Even for POWER_W. :meth:`pooled` is the share of resource-time that was
        used, which for POWER_W reads "share of GPU-hours at or above the floor" --
        a percent, whatever the metric's own unit. Grading it through POWER_W's watt
        floor compared a percentage against watts; it only ever looked right because
        the default floor is 100, and at the 130 W floor the shipped example
        recommends for an H100 that cell read red at 100% above-floor.
        """
        header = self.header if self.header.endswith("%") else _SHARE_HEADER
        return self.thresholds.grade(header, self.pooled())

    def idle(self) -> float:
        """Allocated resource-time that went unused."""
        return self.total - self.used

    def pooled(self) -> Optional[float]:
        """Utilization over the whole selection, as a percent, or None if unmeasured.

        The same number the pooled row prints for this column: used resource-time
        over allocated. Printed in the USED cell and used to grade it, so a metric
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

        Two columns and three counts. This cell used to report IDLE -- unused
        resource-time and the unused share -- while carrying the grade of the *used*
        share and sitting above a bar chart of the used share. One number, three
        readings: the cell said 73%, the pooled row above said 27, the bar below drew
        27, and the colour came from the 27. So it reports USED now, which is what
        the rest of the block already reports and what the section is titled.
        Allocated and idle amounts stay in --csv, where nothing was ever ambiguous.

        The grade is this metric's own pooled utilization, so the eye lands on what
        was wasted even though the number counts what was used. Each count keeps its
        colour tint under the BAD/NOT BAD/GOOD headers -- the word says what the column
        means and the colour finds it, rather than the colour having to carry both.
        """
        used_pct = round(self.pooled()) if self.pooled() is not None else 0
        share = "%d%%" % used_pct
        floor = self.cutoff() if self.absolute else None
        if floor is not None:
            # POWER_W's share is time spent above a watt floor, not a fraction of a
            # resource -- a different quantity from every other row, and from the pooled
            # row above, which reports watts. Naming the floor in the cell is what stops
            # 59% reading as a utilization beside seven that are. The floor is the same
            # one the legend states, so the two cannot disagree.
            share = "%s >%gW" % (share, floor)
        row = [(self.header, ""),
               ("%s (%s)" % (self._amount(self.used), share), self.pooled_band()),
               ("%d" % self.graded(), "")]
        for band in ("green", "yellow", "red"):
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
                 specs: Optional[List[MetricSpec]] = None,
                 footer_only: bool = False) -> None:
        self.out = out or sys.stdout
        self.options = options
        self.context = context
        # Accumulate as usual, print only finish()'s sections: no context block, no header,
        # no rows. What DetailRenderer borrows so a per-node or per-GPU listing can end in
        # one aggregate summary instead of a chart per job -- see its `total`. Composition
        # rather than a second copy of the tallies, which is also what keeps the aggregate
        # identical to the per-job view's for the same selection.
        self.footer_only = footer_only
        self.columns = cols_for(summary_columns(specs, options.show_ids), options.view,
                                options.show_dcgm)
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
        self.no_jobstats = 0            # jobs with no stored summary, excluded from every tally
        # {jobid: ({header: wasted}, user, {headers it is red in})} for jobs red in
        # at least one graded
        # metric, which is what the combined rankings need: a share cannot be taken
        # until the selection's totals are known, so the candidates must be kept.
        # Bounded by the red jobs, not the selection -- on a healthy partition, few.
        self.waste: Dict[str, tuple] = {}
        # jobid -> GPU model, so the printed POWER_W cell is graded against the
        # same floor as the tally row beneath it.
        self.models: Dict[str, str] = {}
        # One tally per graded column, so every metric on screen gets a summary and
        # the set follows the view for free: 8 by default, CPU%/MEM% under --cpu,
        # 6 under --gpu, the full catalog under --dcgm. Under time weighting the
        # weights are resource-seconds and render as hours; otherwise bare counts.
        thresholds = _bands(options)
        hours = options.time_weighted
        self.tallies = {header: EfficiencyTally(header, thresholds,
                                               *_resource_of(header, hours),
                                               worst=options.worst_jobs)
                        for header in self.headers if header.endswith("%")}
        # POWER_W joins them even though it is not a percentage: watts are the one
        # idle signal a duty cycle cannot fake. Weighted by GPU-time like the rest of
        # the GPU family, and banded against a watt floor rather than a percentage,
        # so its USED reads "GPU-hours spent above the floor" -- all-or-nothing per
        # sample, where a percentage's USED takes a fraction of each.
        if "POWER_W" in self.headers:
            self.tallies["POWER_W"] = EfficiencyTally(
                "POWER_W", thresholds, *_resource_of("POWER_W", hours),
                absolute=True, value_unit="W", worst=options.worst_jobs)
        self._started = False
        # The most recently added job's plain {header: float} values and model --
        # for a single-job selection this is that job's own, used by finish() to
        # print an Efficiency line from the same classify()
        # already computed for --ts --eff.
        self._last_values: Dict[str, float] = {}
        self._last_model: str = ""

    def _line(self, row: dict, color: bool = True) -> str:
        """One rendered row, tinted by grade when the options ask for it.

        The escape codes wrap the *padded* cell, never the value: inserting them
        first would make str.format count them toward the column width and skew
        every column to the right of the first coloured one.
        """
        cells = []
        for col in self.columns:
            text = col.fmt.format(str(row.get(col.header, "")))
            band = (self._band(col.header, row.get(col.header),
                               self.models.get(row.get("JOBID", ""), ""))
                    if color else "")
            cells.append(tint(text, band))
        return " ".join(cells)

    def _band(self, header: str, cell, model: str = "") -> str:
        """The grade for a rendered cell; see :func:`cell_band`."""
        return cell_band(self.options, header, cell, model)

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        if self.footer_only or not self.options.header:
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

    def _weights(self, job: JobRow) -> Dict[str, float]:
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
        if not job.found:
            return {"cpu": 0.0, "mem": 0.0, "gpu": 0.0, "gmem": 0.0}
        cores, memory, gpus = job.cores, job.memory, job.gpus
        if not self.options.time_weighted:
            return {"cpu": float(cores), "mem": float(memory),
                    "gpu": float(gpus), "gmem": float(gpus)}
        seconds = job.duration
        if not seconds or seconds <= 0:
            self.unweighted += 1
            return {"cpu": 0.0, "mem": 0.0, "gpu": 0.0, "gmem": 0.0}
        self.durations.add(seconds)
        return {"cpu": cores * seconds, "mem": memory * seconds,
                "gpu": gpus * seconds, "gmem": gpus * seconds}

    def _note_waste(self, jid: str, user: str, values: Dict[str, float],
                    weights: Dict[str, float], runtime: str = "-",
                    duration: Optional[int] = None, model: str = "") -> None:
        """Record what a job wasted per metric, if it is Wasteful in any of them.

        Every metric's waste is kept, not just the ones the job is wasteful in,
        because the resource it wasted is real either way; the wasteful test only
        decides whether the job is a candidate at all. That test is what keeps the
        lists actionable: a 95%-efficient job can idle 50 GPU-hours simply by being
        enormous.
        """
        wasted, red = {}, set()
        for header, tally in self.tallies.items():
            value, weight = values.get(header), weights[tally.weight_key]
            if value is None or weight <= 0:
                continue
            wasted[header] = tally.waste_of(value, weight, model)
            if tally.is_wasteful(value, model):
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
        return scored[:self.options.worst_jobs]

    def _combined_candidates(self, headers: Tuple[str, ...]) -> int:
        """How many jobs are red in *all* of ``headers``."""
        return sum(1 for entry in self.waste.values() if set(headers) <= entry[2])

    def _criteria(self, headers: Tuple[str, ...]) -> str:
        """``"GPU < 2%, CPU < 2%"`` -- the rule a Wasteful row's jobs qualify by.

        Reads the run's actual configured edges (each metric's own ``wasteful``, and
        ``power_w``) rather than restating a fixed number, so the heading stays true
        after a site tunes ``config.toml`` -- a hardcoded "< 2%" would silently lie
        the day someone sets ``cpu = 5``. Per header, which is also why the row can
        legitimately show different cutoffs for its own two metrics.
        """
        thresholds = _bands(self.options)
        parts = []
        for header in headers:
            unit = self.tallies[header].value_unit if header in self.tallies else (
                "W" if header == "POWER_W" else "%")
            cutoff = (thresholds.power_w if header == "POWER_W"
                     else thresholds.edge("wasteful", header))
            parts.append("%s < %g%s" % (_worst_slug(header), cutoff, unit))
        return ", ".join(parts)

    def add(self, rows: List[JobRow]) -> None:
        self._start()
        options = self.options
        do_dcgm = options.show_dcgm
        if options.view == "gpu":
            rows = [r for r in rows if r.gpus]
        self.count += len(rows)
        for job in rows:
            row = {
                "JOBID": job.jobid,
                "USER": job.user,
                "STATE": job.state,
                "NODE": job.nodes,
                "#GPU": str(job.gpus) if job.gpus else "-",
                "RUNTIME": job.runtime,
            }
            # Only the headers --show selected; a cell with no column is never read,
            # and a column with no cell renders blank, so the two come from one place.
            row.update(extra_id_cells(job, self.options.show_ids))
            weights = self._weights(job)
            # Read before the summary block, not inside the DCGM one below: a column the
            # summary does not own has to be overridden *before* it is tallied, or the row
            # would show the measured value and the footer average the summary's. Which
            # columns those are is rows._overrides' to decide, not a renderer's, and it
            # is settled once when the row is built rather than asked again here.
            measured = job.overrides if do_dcgm else {}
            if not job.has_summary:
                for col in JOBSTATS_HEADERS:
                    row[col] = "-"
            else:
                for key, col in JOBSTATS_KEYS:
                    value = job.metrics.value(col)
                    if col in measured:
                        value = measured[col]
                        # The block's own formatter, so an overridden cell reads exactly as
                        # it would in the profiling block (GPU% has no decimals).
                        row[col] = format_by_header(col, value)
                    else:
                        row[col] = "-" if value is None else str(value)
                    if value is not None:
                        self.sums[key][0] += value
                        self.sums[key][1] += 1
                        if weights[key]:
                            self.weighted[key][0] += value * weights[key]
                            self.weighted[key][1] += weights[key]
                        if key == "gpu" and job.gpus:
                            # Counted here rather than per allocation, so the footer
                            # total matches the GPUs actually behind the GPU figures.
                            self.gpu_total += job.gpus
                            self.gpu_counts.add(job.gpus)
            if do_dcgm:
                for header in self.dcgm_headers:
                    value = job.measured.get(header)
                    row[header] = format_by_header(header, value)
                    if value is not None:
                        self.sums_dcgm[header][0] += value
                        self.sums_dcgm[header][1] += 1
                        if weights["gpu"] and header in self.weightable:
                            self.weighted_dcgm[header][0] += value * weights["gpu"]
                            self.weighted_dcgm[header][1] += weights["gpu"]
            # Every graded column at once, now that both the summary and the DCGM
            # values are in hand. Each tally knows which resource weights it.
            # One value map for every graded metric, built once both the jobstats summary and
            # the DCGM values are in hand, and shared by the tallies and the waste
            # bookkeeping so the two cannot disagree about what a job scored.
            if not job.has_summary:
                # No stored summary, so the job is only half measured: it has DCGM
                # numbers but no CPU%/MEM%/GPU%/GMEM%. Feeding it to the DCGM tallies
                # alone made their denominators disagree with the jobstats summary ones -- 117
                # against 88 on one partition -- and a job cannot be compared with
                # the rest on a metric it has no value for. It stays in the listing,
                # since it is a real job; it just does not vote.
                self.no_jobstats += 1
            else:
                values = {}
                for header in self.tallies:
                    value = _jobstats_value(job.metrics, header)
                    if value is None and header in self.dcgm_headers and do_dcgm:
                        value = job.measured.get(header)
                    if value is not None:
                        values[header] = value
                self.models[job.jobid] = job.model
                self._last_values, self._last_model = values, job.model
                for header, tally in self.tallies.items():
                    tally.add(job.jobid, job.user, values.get(header),
                              weights[tally.weight_key], job.runtime,
                              job.duration, model=job.model)
                self._note_waste(job.jobid, job.user, values, weights,
                                 job.runtime, job.duration, model=job.model)
            if self.footer_only:
                continue            # the caller printed its own, at its own granularity
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
        # the bands. POWER_W is here too: its USED is resource-time spent above the
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
        combined = [("gpu-cpu", COMBINED_2, self._combined_worst(COMBINED_2)),
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
        if self.no_jobstats:
            counts.append("no-jobstats=%d" % self.no_jobstats)

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
            summary = ([] if alone else [self._line(used_row)]) \
                + self._stat_table(stats, pooled=not alone)

            problems = []
            if not alone:
                # (heading, [(user, entry text, is long-running)]) per row. The
                # heading now carries the row's own criteria, so it gets its own
                # line -- packing the first user onto it the way a bare count
                # once allowed left the longer combined headings ragged.
                rows_out = []
                for one in worst:
                    rows_out.append((
                        "Wasteful %s (%d/%d): %s" % (_worst_slug(one.header),
                                                     one.bands["red"][0], one.graded(),
                                                     self._criteria((one.header,))),
                        [(job.user,
                          "%s:%d%s:%s(%s)" % (job.jobid, round(job.value),
                                              one.value_unit, one._amount(job.weight),
                                              job.runtime),
                          (job.duration or 0) > options.long_running)
                         for job in one.worst]))
                for name, headers, ranked in combined:
                    # No denominator: a row spanning metrics with different coverage
                    # has no single honest total, so only the candidate count is shown.
                    rows_out.append((
                        "Wasteful %s (%d): %s" % (name, self._combined_candidates(headers),
                                                  self._criteria(headers)),
                        [(user,
                          "%s(%s)" % (_combined_cell(jid, values, self.tallies)
                                      .replace(" ", ":", 1).replace(" ", "/"), runtime),
                          (duration or 0) > options.long_running)
                         for _score, jid, user, values, runtime, duration in ranked]))
                user_width = max([len(user) + 1 for _heading, entries in rows_out
                                  for user, _t, _l in entries] + [0])
                for heading, entries in rows_out:
                    problems.extend(self._worst_rows(heading, entries, user_width))
                problems.append("Jobs: %s" % "  ".join(counts))

            # Keyed by [report] sections' short names, which is also what orders
            # them: a site that reads Problem jobs first should not have to scroll.
            # A section with no content still drops out, in _print_sections.
            built = {"metrics": ("Summary by metric", summary),
                     "efficiency": ("Average efficiency  (filled = used, grey = idle)",
                                    self._bar_lines(stats, pooled=not alone)),
                     "problems": ("Problem jobs", problems)}
            self._print_sections([built[name] for name in options.sections])
            if alone and self._last_values:
                self._print_efficiency()

    def _print_efficiency(self) -> None:
        """A single job's classify() verdict, unnumbered.

        Reuses the exact same functions --ts --eff already computes from:
        combined when both GPU and CPU% are present (the default/all view), plain
        GPU-only classify() for a --gpu view, CPU%-only classify() for a --cpu
        view. Printed after the numbered sections, not as one of them, so a
        single-job report still has exactly sections [1, 2].
        """
        options = self.options
        thresholds = _bands(options)
        # One ballot: every graded percentage that is not memory, CPU% included.
        # POWER_W drops out on the `%` test -- it is watts, and a raw wattage in the
        # ballot would win the comparison outright whatever the GPU was doing. It
        # lowers the verdict instead, as a floor.
        voting = classify_metrics(list(self._last_values), thresholds)
        ballot = {h: self._last_values[h] for h in voting if h in self._last_values}
        if not ballot:
            return
        thresholds = thresholds.for_model(self._last_model)
        floor_readings = {h: self._last_values.get(h) for h in thresholds.floors}
        verdict = classify(ballot, thresholds, floor_readings, columns=voting)
        categories = CATEGORIES
        judged = metrics.in_catalog_order(ballot)
        role = next((r for n, r in categories if n == verdict), "")
        # The range is quoted over the metrics that actually voted, so it cannot
        # name a cutoff this verdict was not reached by -- with per-metric edges a
        # bare range would be some other column's.
        label = tier_range(verdict, thresholds, judged)
        floors_applied = [h for h, v in floor_readings.items() if v is not None]
        desc = classify_description(judged, floors_applied, thresholds)
        text = ("Efficiency: %s (%s)" % (verdict, label) if label
               else "Efficiency: %s" % verdict)
        print(file=self.out)
        print("Graded %s." % desc, file=self.out)
        # Ungated by --noheader, matching the "Graded ..." line it qualifies: that one is
        # not furniture either, and a verdict whose basis is stated only when headers are
        # on would be worse than one that never states it.
        if not options.time_weighted:
            print(verdict_note(), file=self.out)
        print(tint(text, role) if options.color and role else text, file=self.out)


    # GOOD / OK / BAD rather than RED / YELLOW / GREEN. The cells are still tinted those
    # colours, so the words are not a substitute for the colour but the other half of it: a
    # header naming a verdict says what the count *means* without asking the reader to hold
    # a colour-to-quality mapping in their head, and it survives --no-color, a pipe, and
    # colour-blindness, where three colour names carry nothing.
    #
    # Best first, so the columns descend the way the words do. JOBS precedes them because it
    # is their total: without it the three counts have no denominator on screen, and one that
    # legitimately differs per row -- a job whose GPU memory resolved but whose duty cycle
    # did not is graded by GMEM% and not by GPU% -- reads as an inconsistency instead of as
    # coverage. See EfficiencyTally.graded.
    STAT_HEADERS = ("METRIC", "USED", "JOBS", "GOOD", "OK", "BAD")

    # Three lines of legend. The first states the cutoffs -- one sentence while
    # every metric shares them, and a metric-by-metric list once they do not, since
    # then no single pair of numbers is true of the table. The last is there because
    # "green" means only "not pathological": at a cutoff of 10 a job at 21% is green
    # while wasting four fifths of its cores, so a selection can be half idle with
    # almost every job green. USED is the efficiency number; the bands say whether
    # the waste is concentrated in a few jobs or spread across all of them, which is
    # the difference between someone to talk to and a habit.
    STAT_CUTOFFS = ("GOOD above %(yellow)g%%, OK above %(red)g%%, BAD otherwise;")
    STAT_PER_METRIC = "each metric's own OK/GOOD edges -- %s;"
    STAT_LEGEND = (
        "JOBS is how many jobs reported the metric and GOOD/OK/BAD sum to it, so the"
        " total differs per row wherever coverage does. Banded by %(cutoffs)s POWER_W is"
        " GOOD above %(power)g W and BAD below, never OK.",
        "USED is the resource-time that did work, over what was allocated --"
        " for POWER_W, the time spent above that watt floor.",
        "the counts locate the waste and USED measures it: no BAD jobs but a low USED"
        " means every job wastes a little, rather than a few jobs wasting a lot",
    )
    LEGEND_WIDTH = 128

    def _cutoff_phrase(self, thresholds: "Thresholds", headers: List[str]) -> str:
        """The cutoff clause: one pair of numbers, or one pair per metric.

        Only the metrics actually in the table, and only the two edges its three columns
        divide on: ``inefficient`` separates BAD from NOT BAD and ``improvement`` separates
        NOT BAD from GOOD. The other two edges are deliberately absent -- ``wasteful``
        splits BAD from BAD and ``average`` splits GOOD from GOOD, so either would be a
        number pointing at no column. (``wasteful`` is the Problem-jobs cutoff, and that
        section states its own inline.)

        Which also decides the form. Keying the short form off Thresholds.uniform meant
        comparing all four edges, and the shipped defaults differ in ``wasteful`` alone --
        CPU% 5 against 2 -- so every default report took the metric-by-metric branch and
        spent two wrapped lines listing one identical pair of cutoffs eight times.
        """
        graded = [h for h in headers if h.endswith("%")]
        pairs = {h: (thresholds.edge("inefficient", h), thresholds.edge("improvement", h))
                 for h in graded}
        if len(set(pairs.values())) <= 1:
            bad, not_bad = (pairs[graded[0]] if graded
                            else (thresholds.edge("inefficient", ""),
                                  thresholds.edge("improvement", "")))
            return self.STAT_CUTOFFS % {"red": bad, "yellow": not_bad}
        each = ", ".join("%s %g/%g" % (h, bad, not_bad)
                         for h, (bad, not_bad) in pairs.items())
        return self.STAT_PER_METRIC % each

    def _legend(self, headers: Optional[List[str]] = None,
                pooled: bool = True) -> List[str]:
        """:data:`STAT_LEGEND` with this run's actual cutoffs filled in, wrapped.

        Wrapped because the per-metric form grows with the table: at ``--dcgm`` it
        names eighteen columns, which on one line would run four times the width of
        everything above it.

        The averaging line comes first, before any of the grading: it says what the
        numbers *are*, where the rest says how they are judged, and a reader who stops
        after one line should have that one. The last entry stays last, directly above
        the METRIC header it explains. Computed rather than a constant; see
        :func:`averaging_note`.
        """
        thresholds = _bands(self.options)
        values = {"cutoffs": self._cutoff_phrase(thresholds, headers or []),
                  "power": thresholds.power_w}
        # Wrapped but not interpolated: it is already complete, and running a sentence
        # that may grow a "GPU%" through `% values` would raise on the bare percent.
        legend = [averaging_note(self.options.time_weighted, pooled)] \
            + [line % values for line in self.STAT_LEGEND]
        lines = []
        for line in legend:
            # break_on_hyphens=False: the terms of art here are hyphenated -- resource-time,
            # GPU-hours, red/yellow/green -- and splitting one across a line break reads as
            # two words, one of them unfamiliar.
            lines.extend(textwrap.wrap(line, self.LEGEND_WIDTH, subsequent_indent="  ",
                                       break_on_hyphens=False) or [""])
        return lines

    WORST_WIDTH = 132

    def _worst_rows(self, heading: str, entries, user_width: int) -> List[str]:
        """``heading`` on its own line, then one line per user: ``  user|
        job:val:wasted(elapsed), job:...``.

        The heading now states the row's own criteria as well as its count, so it
        no longer shares a line with the first user -- grouped by user below it
        because a single user usually owns several of the worst jobs, and
        repeating their name three times says less than showing they own the row.
        Entries wrap onto continuation lines rather than running past the table.
        """
        out = [heading]
        for user, jobs in _worst_groups(entries):
            prefix = "  %-*s" % (user_width, user + "|")
            line, count = prefix, 0
            for text, long_running in jobs:
                # Every entry here is red-band by construction, so it is painted like
                # one: red means wasteful in every other part of the report, and a
                # section titled Wasteful printing plain text read as "not graded".
                # long_running keeps its own role for the expensive ones -- wasteful for
                # hours, not minutes -- which is now a brighter red rather than the only
                # colour in the section.
                cell = tint(text, "long_running" if long_running else "wasteful") if (
                    self.options.color) else text
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

    def _bar_lines(self, stats: List["EfficiencyTally"],
                   pooled: bool = True) -> List[str]:
        """The efficiency bars, or nothing when they are switched off or in CSV.

        Never in CSV: a bar chart has no place in a machine format, and parse_csv
        would have to be taught to skip it.
        """
        if not self.options.plot_avgeff or self.options.csv:
            return []
        bars = self._eff_bars(stats)
        if bars and self.options.header:
            # Furniture, so --noheader drops it with the titles and rules; it says how
            # to read the bars rather than adding a bar.
            #
            # Wrapped to the bars' own width rather than the legend's: _print_sections
            # rules each section to its widest line, and a one-line note runs half again
            # the length of a bar, leaving a rule extending well past the chart.
            width = max(len(_ESC_RE.sub("", line)) for line in bars)
            note = ["  " + line for line in
                    textwrap.wrap(bars_note(self.options.time_weighted, pooled),
                                  max(40, width - 2), subsequent_indent="  ",
                                  break_on_hyphens=False)]
            bars = note + bars
        return bars

    def _eff_bars(self, stats: List["EfficiencyTally"]) -> List[str]:
        """Horizontal utilization bars, one per graded metric.

        The same number as the table's USED column, in the form that answers "which
        resource was wasted" at a glance: bar length is the pooled utilization, so the
        bar and the USED percentage are the same figure drawn two ways.

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


    def _stat_table(self, stats: List["EfficiencyTally"],
                    pooled: bool = True) -> List[str]:
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
            for line in self._legend([one.header for one in stats], pooled):
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
                cells.append(tint(cell, band if self.options.color else ""))
            out.append("  ".join(cells))
        return out


class DetailRenderer:
    """Streaming form of detail(): independent per-job blocks per add().

    What follows the blocks depends on how many jobs there are, and the two answers are
    different questions. For **one** job the per-unit charts answer "which of my nodes is
    the slow one", which the rows alone do not: sixteen rows of twelve columns is not a
    comparison. For **many** they answer it once per job, so a partition sweep ends in a
    dozen charts of one bar each and no reading of the selection anywhere -- there the
    question is "how is this selection doing", and the answer is the aggregate summary the
    per-job view already computes.

    So ``total`` (the selection size, from select.Resolved) picks between them. It has to
    be known up front rather than counted: a block prints before the renderer can know
    whether another follows.

    The aggregate is over **jobs**, not over the printed unit rows -- the same arithmetic
    the per-job view uses, so `--per-node` and the default view report identical summaries
    for one selection and differ only in row granularity. Aggregating the rows instead
    would weight by unit and quietly disagree, and would have to reason about MIG siblings
    sharing a ``(node, minor)`` key.

    Text only: the footer's CSV form is the ``Stat``/``Worst``/``Jobs`` rows, and adding
    those to ``--per-gpu --csv`` would change a machine format for a reader that did not
    ask. The bars already suppress themselves there for the same reason.
    """

    def __init__(self, context: List[Tuple[str, str]], options: RenderOptions, out=None,
                 level: str = GPU_LEVEL, specs: Optional[List[MetricSpec]] = None,
                 total: int = 1) -> None:
        self.out = out or sys.stdout
        self.options = options
        self.context = context
        self.level = level
        self.total = total
        # One job keeps its per-unit charts; a selection gets one aggregate block instead.
        self.aggregate = (SummaryRenderer(context, options, self.out, specs=specs,
                                          footer_only=True)
                          if total > 1 and not options.csv else None)
        # Resolved once per renderer, not read per row: the source preference is
        # settled at config load, and a column list that changed mid-report would
        # put a job's cells under another job's headers.
        self.columns = cols_for(detail_columns(level), options.view, options.show_dcgm)
        self.writer = csv.writer(self.out, lineterminator="\n") if options.csv else None
        self.count = 0
        # Every node seen *before* filtering, so an unmatched --nodename can say what
        # was actually there, and whether anything matched at all.
        self.nodes_seen = set()
        self.matched = 0
        self._started = False

    def _line(self, cells, model: str = "") -> str:
        """One per-GPU row, graded like the per-job table above it."""
        return " ".join(
            tint(c.fmt.format(str(cells[c.index])),
                 cell_band(self.options, c.header, cells[c.index], model))
            for c in self.columns)

    def _header_line(self) -> str:
        """The column names at the widths of the rows below them.

        Off ``self.columns`` rather than a written-down tuple: that tuple was a third
        copy of the header list and could disagree with the two that matter. No tint,
        because a header is not a reading -- which is also why this is byte-identical to
        the old ``_line(DETAIL_HEADER)``, whose cell_band() found nothing numeric to grade.
        """
        return " ".join(c.fmt.format(c.header) for c in self.columns)

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        if not self.options.header:
            return
        if self.options.csv:
            for label, value in self.context:
                self.writer.writerow([label, value])
            self.writer.writerow(
                ["JOBID"] + [c.header for c in extra_id_columns(self.options.show_ids)]
                + [c.header for c in self.columns])
        else:
            for label, value in self.context:
                print(fmt_context(label, value), file=self.out)
            print(file=self.out)

    def _rows_for(self, job: JobRow):
        node_level = self.level == NODE_LEVEL
        all_units = job.node_rows if node_level else job.gpu_rows
        # Elapsed is per job, so it repeats down the block -- accepted for the reason
        # CPU% already repeats: a column is the only way a CSV reader gets it, and the
        # detail CSV carries JOBID and nothing else about the job.
        # Keyed by node at node level and by (node, minor) at gpu level, so the lookup key
        # is the row's own identity either way.
        if not self.options.show_dcgm:
            def values_for(_unit):
                return {}
        elif node_level:
            def values_for(unit):
                return job.per_node.get(unit.node, {})
        else:
            def values_for(unit):
                return job.per_gpu.get((unit.node, str(unit.unit)), {})
        self.nodes_seen.update(u.node for u in all_units)
        units = ([u for u in all_units if u.node == self.options.nodename]
                 if self.options.nodename else all_units)
        rows = [detail_row_cells(u, values_for(u), job.runtime) for u in units]
        self.matched += len(rows)
        return rows

    def add(self, rows: List[JobRow]) -> None:
        self._start()
        options = self.options
        if options.view == "gpu":
            rows = [r for r in rows if r.gpus]
        self.count += len(rows)
        if options.csv:
            for job in rows:
                extra = extra_id_cells(job, options.show_ids)
                lead = [job.jobid] + [extra[c.header]
                                      for c in extra_id_columns(options.show_ids)]
                for row in self._rows_for(job):
                    self.writer.writerow(lead + [row[c.index] for c in self.columns])
        else:
            for job in rows:
                # On the job's own line, not as a column: a detail row is about one card
                # or one host, and the account a job ran under is the same on every one
                # of them. A column would repeat it down the block to say nothing new.
                extra = extra_id_cells(job, options.show_ids)
                print("Job %s  [%s]  %s%s"
                      % (job.jobid, job.state, job.name,
                         "".join("  %s" % extra[c.header]
                                 for c in extra_id_columns(options.show_ids))),
                      file=self.out)
                unit_rows = self._rows_for(job)
                if not unit_rows:
                    print("  (no jobstats data)\n", file=self.out)
                    continue
                header_line = self._header_line()
                print("  " + header_line, file=self.out)
                print("  " + "-" * len(header_line), file=self.out)
                for row in unit_rows:
                    print("  " + self._line(row, job.model), file=self.out)
                if self.aggregate is None:
                    for line in self._unit_charts(unit_rows):
                        print(line, file=self.out)
                print(file=self.out)
        if self.aggregate is not None:
            self.aggregate.add(rows)
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
        # At node level each row *is* a node, so there is nothing to break down by card
        # -- and cell 1 is a count, so labelling a group "GPU 4" would be a lie.
        by_gpu = len(nodes) == 1 and self.level != NODE_LEVEL
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
            return
        if self.aggregate is not None:
            self.aggregate.finish()


def summarize(rows: List[JobRow], context: List[Tuple[str, str]],
              options: RenderOptions, out=None) -> None:
    """One row per job: the stored metrics and, under --all-metrics, the profiling columns."""
    renderer = SummaryRenderer(context, options, out)
    renderer.add(rows)
    renderer.finish()


def detail(rows: List[JobRow], context: List[Tuple[str, str]],
           options: RenderOptions, out=None) -> None:
    """Per-node / per-GPU breakdown for each job."""
    renderer = DetailRenderer(context, options, out, total=len(rows))
    renderer.add(rows)
    renderer.finish()


def dcgm_report(rows: List[JobRow], specs: List[MetricSpec],
                context: List[Tuple[str, str]], options: RenderOptions,
                out=None) -> None:
    """One row per job, with the profiling block taken from ``specs``.

    The same renderer the summary view uses, so the two print identical columns;
    ``--all-metrics`` only widens the profiling block. Per-GPU numbers live in
    ``jobscope detail`` and in the ``--ts`` time series.
    """
    renderer = SummaryRenderer(context, options, out, specs=specs)
    renderer.add(rows)
    renderer.finish()


def dcgm_timeseries(collected, specs: List[MetricSpec], options: RenderOptions,
                    out=None) -> None:
    """Emit the raw per-scrape DCGM time series over the job's window as CSV.

    One row per GPU/timestamp; each cell is the raw sampled value, scaled for display.
    ``collected`` is an iterable of :class:`jobscope.timeseries.JobSeries` -- usually
    the generator, so a partition-wide sweep still renders one job at a time.
    """
    out = out or sys.stdout
    # Every spec was queried, including the hidden ones feeding a derived column, but
    # only the displayed ones are emitted -- so this header matches `running --ts`.
    columns = columns_for(specs)
    writer = csv.writer(out, lineterminator="\n")
    wrote_header = False

    def write(row) -> None:
        """Emit *row*, writing the header first if it has not been written yet.

        Lazily, because a --nodename that matches nothing raises: a header with no
        rows under it is a CSV that reads as "this node was idle" and confuses
        `jobscope plot` into "no numeric values". Nothing written is the honest
        answer, and it is what the running path already does.
        """
        nonlocal wrote_header
        if options.header and not wrote_header:
            writer.writerow(TS_ID_COLUMNS
                            + [header for _key, header, _dec in columns])
            wrote_header = True
        writer.writerow(row)

    for job in collected:
        rows = []  # (node, minor_sort, ts, csv_row)
        for uuid, (node, minor, model) in job.gpus.items():
            for stamp in sorted(job.gpu[uuid]):
                cells = job.gpu[uuid][stamp]
                rows.append((node, gpu_minor_key(minor), stamp,
                             [job.jobid, job.user, stamp,
                              time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp)),
                              node, minor, model]
                             + [format_number(cells.get(h), d, missing="")
                                for _k, h, d in columns]))
        for _, _, _, row in sorted(rows, key=lambda x: (x[0], x[1], x[2])):
            write(row)


def _cgroup_cells(cells: dict, specs: List[CgroupSpec]) -> List[str]:
    """One CSV cell per cgroup spec, at that spec's own precision.

    Per spec rather than a fixed 0 decimals: CPU%/MEM% read as integers as they
    always have, while the finer columns keep a decimal that rounding would erase
    (CACHE% of 0.4 is not the same story as 0).
    """
    return [format_number(cells.get(spec.header), spec.decimals, missing="")
            for spec in specs]


def cpu_timeseries(collected, options: RenderOptions, out=None) -> None:
    """Emit the raw per-scrape cgroup series over the job's window as CSV.

    One row per node/timestamp -- there is no GPU dimension, so GPU/MODEL are left
    blank, keeping the schema :func:`dcgm_timeseries` writes so `jobscope plot`,
    ``--eff`` and ``--stats job`` need no changes to read it.
    """
    out = out or sys.stdout
    writer = csv.writer(out, lineterminator="\n")
    cgroup = chosen_specs(options.cgroup_specs)
    wrote_header = False

    def write(row) -> None:
        nonlocal wrote_header
        if options.header and not wrote_header:
            writer.writerow(TS_ID_COLUMNS + [spec.header for spec in cgroup])
            wrote_header = True
        writer.writerow(row)

    for job in collected:
        rows = []  # (host, ts, csv_row)
        for host in job.hosts:
            for stamp in sorted(job.host.get(host, {})):
                rows.append((host, stamp,
                             [job.jobid, job.user, stamp,
                              time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp)),
                              host, "", ""]
                             + _cgroup_cells(job.host[host][stamp], cgroup)))
        for _, _, row in sorted(rows, key=lambda x: (x[0], x[1])):
            write(row)


def combined_timeseries(collected, specs: List[MetricSpec], options: RenderOptions,
                        out=None) -> None:
    """``dcgm_timeseries`` plus each row's node's cgroup columns appended -- the
    default ``--ts`` view: GPU/DCGM metrics and CPU%/MEM% together in one series.

    GPU and host samples shared one window per job when they were collected, so they
    are already on the same timestamp grid and a row is a plain dict lookup.
    """
    out = out or sys.stdout
    columns = columns_for(specs)
    cgroup = chosen_specs(options.cgroup_specs)
    writer = csv.writer(out, lineterminator="\n")
    wrote_header = False

    def write(row) -> None:
        nonlocal wrote_header
        if options.header and not wrote_header:
            writer.writerow(TS_ID_COLUMNS + [header for _key, header, _dec in columns]
                            + [spec.header for spec in cgroup])
            wrote_header = True
        writer.writerow(row)

    for job in collected:
        rows = []  # (node, minor_sort, ts, csv_row)
        for uuid, (node, minor, model) in job.gpus.items():
            node_cpu = job.host.get(node, {})
            for stamp in sorted(job.gpu[uuid]):
                cells = job.gpu[uuid][stamp]
                rows.append((node, gpu_minor_key(minor), stamp,
                             [job.jobid, job.user, stamp,
                              time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp)),
                              node, minor, model]
                             + [format_number(cells.get(h), d, missing="")
                                for _k, h, d in columns]
                             + _cgroup_cells(node_cpu.get(stamp, {}), cgroup)))
        for _, _, _, row in sorted(rows, key=lambda x: (x[0], x[1], x[2])):
            write(row)


_REDUCER_NAME = {"avg": "mean", "max": "peak", "delta": "delta"}


def _running_rows(jobs: Dict[int, RunningJob], gpus: Dict[str, Gpu]):
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




TS_STAT_TAIL = ("METRIC", "N", "MIN", "MEAN", "MAX", "LAST")



def unit_headers(level: str, multi_job: bool) -> Tuple[str, ...]:
    """The identifying columns for a level, before the metrics.

    Each level names what it pooled: a node row says how many GPUs went into it, a job
    row how many nodes and GPUs. Without that the reader cannot tell a one-GPU mean
    from a sixteen-GPU one. JOBID leads only when the series covers more than one job,
    which for the usual single-job selection keeps the table narrow.
    """
    if level == "job":
        return ("JOBID", "NODES", "GPUS")
    job = ("JOBID",) if multi_job else ()
    return job + (("NODE", "GPUS") if level == "node" else ("NODE:GPU",))


def unit_order(key: tuple):
    """Sort key for a unit, so the views sharing a row identity share its order too.

    Beside :func:`unit_headers` and :func:`unit_values` for the same reason they sit
    together: three hand-written copies of "(jobid, node, gpu minor)" had already drifted
    into two spellings, one of which collapsed a falsy node name to "".
    """
    return (key[0],
            key[1] if len(key) > 1 else "",
            gpu_minor_key(key[2]) if len(key) > 2 else 0)


def unit_values(level: str, multi_job: bool, key: tuple, found: dict) -> Tuple[str, ...]:
    """The values for :func:`unit_headers`, in the same order.

    Paired with it deliberately: these were two mirrored branch chains that had to
    agree positionally with nothing to enforce it, and the widths index straight into
    that agreement. Adding a level now means editing one thing twice, next to itself.
    """
    jobid = key[0]
    if level == "job":
        return (jobid, str(len(found["nodes"])), str(len(found["gpus"])))
    job = (jobid,) if multi_job else ()
    if level == "node":
        return job + (key[1], str(len(found["gpus"])))
    return job + ("%s:%s" % (key[1], key[2]),)


def timeseries_eff(rows: List[dict], columns: List[str], options: "RenderOptions",
                        out=None, level: str = "job", show_all: bool = False) -> None:
    """Group the units into efficiency categories, worst first.

    The verdict for each is :func:`classify`; what this adds is the reading order.
    A partition sweep exists to be acted on
    from the top, and on a healthy one most jobs are fine -- so ``good`` collapses
    to a count unless ``show_all``, which is the difference between a page and a
    hundred of them.

    CPU% votes but never ranks: a unit's position among its peers comes from its
    best GPU reading, because a busy host is not what someone scanning this list is
    looking for.
    """
    out = out or sys.stdout
    voting = classify_metrics(columns, _bands(options))
    if not voting:
        raise JobscopeError("no %-metrics in this series to classify")
    unit = {"gpu": "GPUs", "node": "nodes"}.get(level, "jobs")
    thresholds = _bands(options)

    # One ballot whatever the series carries. CPU% votes like any other metric --
    # its ceiling is what stops a busy host calling a GPU-idle unit healthy -- so a
    # plain `--cpu --ts` series and a combined one take the same path. That replaces
    # `is_combined`, which was `"CPU%" in voting and len(voting) > 1` and would have
    # mislabelled a cpu-only series the moment the cgroup detail columns landed.
    categories = CATEGORIES
    # A unit whose voting columns carried no samples lands here rather than in a
    # band. Last, because it is an absence rather than a severity: putting it at
    # the top would push the findings someone opened the report for off the page.
    categories = categories + ((NO_DATA, NO_DATA),)

    reported = [c for c in columns if c not in TS_ID_COLUMNS]
    groups = pool_samples(rows, reported, level)
    verdicts = []
    for key, found in groups.items():
        # Every metric is averaged; the voting ones decide the label. Keeping the
        # two apart is what lets the CSV report GMEM% and POWER_W without letting
        # them pick the verdict -- POWER_W lowers it as a floor instead.
        means = {m: sum(v) / len(v) for m, v in found["values"].items()}
        judged = {m: means[m] for m in metrics.in_catalog_order(voting) if m in means}
        model = job_model({k: {MODEL_KEY: m} for k, m in found["models"].items()})
        bands = thresholds.for_model(model)
        floor_readings = {h: means.get(h) for h in bands.floors}
        carriers = unceilinged(bands, voting)
        if carriers and not any(m in judged for m in carriers):
            # The series carries columns that could have voted this unit healthy and
            # none of them has a value for it. A busy host must not fill that gap.
            judged = {}
        verdict = classify(judged, bands, floor_readings, columns=voting)
        # None means nothing voted: the series carries these columns but this unit
        # had no samples in any of them -- a dead exporter, a node that restarted,
        # a window past retention. Saying `wasteful` there would report a
        # collection gap as waste, and it is the reading someone acts on.
        if verdict is None:
            verdict = NO_DATA
        gpu_only = [v for m, v in judged.items() if m != "CPU%"]
        best_gpu = max(gpu_only) if gpu_only else 0.0
        verdicts.append((verdict, key, found, means, best_gpu))

    # Ranking: the GPU metric for a combined series (per the design -- CPU never
    # ranks), the group key otherwise, exactly as before this feature existed.
    # Rank by the best GPU reading where there is one, else by the group key, as
    # before -- CPU% never ranks, only votes.
    rank_key = ((lambda v: v[4]) if any(m != "CPU%" for m in voting)
                else (lambda v: v[1]))

    if options.csv:
        # jobid, user, every metric the series carried, then the label. Wider than
        # the metrics the verdict was taken over -- POWER_W and GMEM% do not vote, but
        # a row you are going to sort or join on should carry what was measured.
        writer = csv.writer(out, lineterminator="\n")
        if options.header:
            writer.writerow(["JOBID", "USER"] + reported + ["LABEL"])
        order = {name: i for i, (name, _c) in enumerate(categories)}
        for name, key, found, means, _best_gpu in sorted(
                verdicts, key=lambda v: (order[v[0]], rank_key(v))):
            writer.writerow([key[0], found["user"]]
                            + ["%.1f" % means[m] if m in means else "" for m in reported]
                            + [name])
        out.flush()
        return

    if options.header:
        desc = classify_description(
            voting, [h for h in thresholds.floors if h in columns], thresholds)
        print("  %d %s, %s" % (len(verdicts), unit, desc), file=out)
        print(file=out)

    # Columns rather than inline "NAME value" pairs: this was the one table in the
    # tool that named its metric on every row and aligned nothing, so a jobid, a
    # username and a reading ran together. The identity block also has to say which
    # unit was judged -- at node level every row read as the same job id before.
    multi_job = len({r.get("JOBID", "?") for r in rows}) > 1
    extra_headers = tuple(h for h in ("POWER_W",) if h not in voting)
    headers = unit_headers(level, multi_job) + ("USER",) + tuple(voting) + extra_headers

    def cells_for(key, found, means) -> Tuple[str, ...]:
        return (unit_values(level, multi_job, key, found) + (found["user"],)
                + tuple("%.1f" % means[m] if m in means else "-" for m in voting)
                + tuple("%.0f" % means[h] if h in means else "-" for h in extra_headers))

    table = {id(v): cells_for(v[1], v[2], v[3]) for v in verdicts}
    # Measured across every category, so the columns line up between them and two
    # jobs in different bands stay comparable at a glance.
    widths = [max(len(headers[i]), max((len(c[i]) for c in table.values()), default=0))
              for i in range(len(headers))]
    text_cols = len(unit_headers(level, multi_job)) + 1        # identity, then numbers

    def row_text(cells) -> str:
        return "  ".join(c.ljust(widths[i]) if i < text_cols else c.rjust(widths[i])
                         for i, c in enumerate(cells)).rstrip()

    by_name: Dict[str, list] = {}
    for verdict in verdicts:
        by_name.setdefault(verdict[0], []).append(verdict)
    for name, role in categories:
        found = by_name.get(name, [])
        if not found:
            continue
        heading = "%s (%s)  %d %s" % (name, tier_criteria(name, voting, thresholds),
                                      len(found), unit)
        print("  " + (tint(heading, role) if options.color else heading), file=out)
        if name == "good" and not show_all:
            # Most of a healthy partition, and none of what the report is for.
            print("    (--eff all to list them)", file=out)
            continue
        if options.header:
            print("    " + row_text(headers), file=out)
        for verdict in sorted(found, key=rank_key):
            print("    " + row_text(table[id(verdict)]), file=out)
        print(file=out)
    out.flush()


def pool_samples(rows: List[dict], metrics: List[str], level: str) -> Dict[tuple, dict]:
    """``{key: {values, nodes, gpus, user, models}}``, pooling the series at ``level``.

    Shared by the stats table and the verdict so the two cannot disagree about
    what a job's mean is: one grouping, read two ways. ``models`` is keyed by
    ``(node, gpu)``, the same shape :func:`job_model` already reads elsewhere, so a
    classify verdict can resolve the group's POWER_W floor the same way the table
    views do.
    """
    groups: Dict[tuple, dict] = {}
    for row in rows:
        jobid, node = row.get("JOBID", "?"), row.get("NODE", "?")
        gpu = row.get("GPU", "?")
        key = {"gpu": (jobid, node, gpu), "node": (jobid, node)}.get(level, (jobid,))
        found = groups.setdefault(key, {"values": {}, "nodes": set(), "gpus": set(),
                                        "user": row.get("USER", "?"), "models": {}})
        found["nodes"].add(node)
        found["gpus"].add((node, gpu))
        model = row.get("MODEL")
        if model:
            found["models"][(node, gpu)] = model
        for metric in metrics:
            value = cell_value(row.get(metric))
            if value is not None:
                found["values"].setdefault(metric, []).append(value)
    return groups


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
    groups = pool_samples(rows, metrics, level)

    multi_job = len({r.get("JOBID", "?") for r in rows}) > 1
    lead = unit_headers(level, multi_job)
    headers = lead + TS_STAT_TAIL

    table = []
    for key, found in sorted(groups.items(), key=lambda item: unit_order(item[0])):
        # Carried per group so POWER_W's mean is tinted against the card it was
        # measured on. The same resolution timeseries_eff does for the same
        # samples; dropping it here graded a 165 W RTX -- idle -- against the global
        # 100 W floor and called it green.
        model = job_model({k: {MODEL_KEY: m} for k, m in found["models"].items()})
        for metric in metrics:
            values = found["values"].get(metric)
            if not values:
                continue
            mean = sum(values) / len(values)
            table.append((unit_values(level, multi_job, key, found), metric, len(values),
                          min(values), mean, max(values), values[-1], model))
    if not table:
        print("No samples to summarize.", file=sys.stderr)
        return

    cells = [tuple(label) + (metric, str(n), "%.1f" % low, "%.1f" % mean,
                             "%.1f" % high, "%.1f" % last)
             for label, metric, n, low, mean, high, last, _model in table]
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
        # Tint the mean by its band, as the summary table tints USED: the column
        # anyone reads first should say whether the number is a problem.
        painted = [cell.ljust(widths[i]) if i < text_cols else cell.rjust(widths[i])
                   for i, cell in enumerate(row)]
        painted[mean_at] = tint(painted[mean_at],
                                cell_band(options, entry[1], entry[4], entry[7]))
        print("  " + "  ".join(painted).rstrip(), file=out)
    out.flush()


def running_timeseries(jobs: Dict[int, RunningJob], samples: Dict[str, Dict[int, dict]],
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
    derived = applicable_derived(specs)
    writer = csv.writer(out, lineterminator="\n")
    if options.header:
        writer.writerow(TS_ID_COLUMNS
                        + [header for _k, header, _d in columns])

    for job, gpu in _running_rows(jobs, gpus):
        if gpu is None:
            continue
        for epoch in sorted(samples.get(gpu.uuid, {})):
            values = samples[gpu.uuid][epoch]
            # Recompute per timestamp, so MEM% tracks memory growth over the run.
            for column in derived:
                values[column.key] = column.fn(values)
            writer.writerow(
                [job["jobid"], job.get("user", "?"), epoch,
                 time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch)),
                 gpu.host, gpu.csv_id, gpu.model]
                + [format_number(values.get(key), dec, missing="")
                   for key, _h, dec in columns])


def running_cpu_timeseries(collected, order, options: RenderOptions, out=None) -> None:
    """Emit the raw per-scrape cgroup series for running jobs as CSV.

    Mirrors :func:`cpu_timeseries`; ``order`` is the display order the collector
    resolved, since a dict of running jobs has no meaningful one of its own.
    """
    out = out or sys.stdout
    writer = csv.writer(out, lineterminator="\n")
    cgroup = chosen_specs(options.cgroup_specs)
    wrote_header = False

    for raw_jobid in order:
        job = collected[raw_jobid]
        rows = []  # (host, ts, csv_row)
        for host in job.hosts:
            for stamp in sorted(job.host.get(host, {})):
                rows.append((host, stamp,
                             [job.jobid, job.user, stamp,
                              time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp)),
                              host, "", ""]
                             + _cgroup_cells(job.host[host][stamp], cgroup)))
        for _, _, row in sorted(rows, key=lambda x: (x[0], x[1])):
            if options.header and not wrote_header:
                writer.writerow(TS_ID_COLUMNS + [spec.header for spec in cgroup])
                wrote_header = True
            writer.writerow(row)


def running_combined_timeseries(jobs: Dict[int, RunningJob],
                                samples: Dict[str, Dict[int, dict]],
                                gpus: Dict[str, Gpu], specs: List[MetricSpec],
                                collected, options: RenderOptions, out=None) -> None:
    """``running_timeseries`` plus each row's node's cgroup columns appended -- the
    default running-job ``--ts`` view.

    The GPU side is exactly ``running_timeseries()``'s pre-fetched ``samples``/``gpus``
    (any ``--nodename`` filtering already happened before this is called, on ``gpus``);
    the host side is ``collected``, keyed by raw job ID.
    """
    out = out or sys.stdout
    columns = build_columns(specs)
    derived = applicable_derived(specs)
    writer = csv.writer(out, lineterminator="\n")
    cgroup = chosen_specs(options.cgroup_specs)

    if options.header:
        writer.writerow(TS_ID_COLUMNS + [header for _k, header, _d in columns]
                        + [spec.header for spec in cgroup])

    for job, gpu in _running_rows(jobs, gpus):
        if gpu is None:
            continue
        found = collected.get(gpu.jobid)
        node_cpu = found.host.get(gpu.host, {}) if found else {}
        for epoch in sorted(samples.get(gpu.uuid, {})):
            values = samples[gpu.uuid][epoch]
            for column in derived:
                values[column.key] = column.fn(values)
            writer.writerow(
                [job["jobid"], job.get("user", "?"), epoch,
                 time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch)),
                 gpu.host, gpu.csv_id, gpu.model]
                + [format_number(values.get(key), dec, missing="")
                   for key, _h, dec in columns]
                + _cgroup_cells(node_cpu.get(epoch, {}), cgroup))


def describe(out=None) -> None:
    """Print a plain-English description of each summary column."""
    out = out or sys.stdout
    print("jobscope columns. CPU/MEM/GPU/GMEM come from the jobstats summary"
          " in sacct (no network);", file=out)
    print("the DCGM columns (gpu view) come from Prometheus. For the full per-GPU", file=out)
    print("GPU catalog, run 'jobscope describe --metrics' (or --all-metrics for all %d).\n"
          % len(dcgm.catalog().all_specs), file=out)
    for header, source, text in SUMMARY_DESCRIPTIONS:
        print("  %-9s %s" % (header, source), file=out)
        for wrapped in textwrap.wrap(text, width=74):
            print("      " + wrapped, file=out)
        print(file=out)


def describe_dcgm(specs: List[MetricSpec], out=None, extended=None) -> None:
    """Plain-English reference for the DCGM metric catalog.

    ``extended`` is the widest list this run could show (``[metrics] extended``),
    used only to label whether ``specs`` is already all of it. Compared as a set,
    not by length: the two lists are configurable independently, so a site could
    give them the same size without their being the same metrics.
    """
    out = out or sys.stdout
    reducer_name = _REDUCER_NAME
    widest = list(dcgm.catalog().all_specs if extended is None else extended)
    is_widest = {s.key for s in specs} >= {s.key for s in widest}
    # Hidden specs exist only to feed a derived column, so describe the column
    # instead -- what a reader sees in the table.
    shown = [s for s in specs if s.show]
    derived = applicable_derived(specs)
    print("DCGM GPU metrics. Each value is time-averaged over the job's [start,end]", file=out)
    print("window. Showing %d of %d metrics (%s). [reduce] = how the window is collapsed.\n"
          % (len(shown) + len(derived), len(widest),
             "all" if is_widest else "default; --all-metrics for the rest"), file=out)
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


# --- --verify: the pre-action check ------------------------------------------
#
# The sweep's default reading is one scrape, and one scrape sits more than 20 points from
# a job's own mean about a quarter of the time. Before a scancel that is not good enough:
# measured on a 4-GPU job, the instant reading was GPU% 0 and SM_ACT% 0.0 -- below the
# wasteful edge on both -- while the job was saturating all four cards in bursts.
#
# So this reports a *ladder* rather than a number. Reading the rungs downward is what
# separates "always been mediocre" from "was fine, stopped an hour ago", and `max` is what
# separates dead from bursty. No single mean says either, at any accuracy.

VERIFY_TAIL = ("BELOW", "IDLEMAX", "SHAPE")


def stamped_samples(rows: List[dict], metrics: List[str],
                    level: str = GPU_LEVEL) -> Dict[tuple, dict]:
    """``{key: {stamps: {metric: [(epoch, value)]}, models, ...}}`` at ``level``.

    :func:`pool_samples` keeps only the values, which is all the stats table needs. The
    ladder needs *when* each sample was taken -- to slice a rung, and because a gap has to
    break an idle stretch rather than extend it.
    """
    groups: Dict[tuple, dict] = {}
    for row in rows:
        jobid, node, gpu = row.get("JOBID", "?"), row.get("NODE", "?"), row.get("GPU", "?")
        key = {"gpu": (jobid, node, gpu), "node": (jobid, node)}.get(level, (jobid,))
        found = groups.setdefault(key, {"stamps": {}, "user": row.get("USER", "?"),
                                        "models": {}})
        if row.get("MODEL"):
            found["models"][(node, gpu)] = row["MODEL"]
        try:
            epoch = int(float(row.get("EPOCH")))
        except (TypeError, ValueError):
            continue
        for metric in metrics:
            value = cell_value(row.get(metric))
            if value is not None:
                found["stamps"].setdefault(metric, []).append((epoch, value))
    return groups


def sampling_step(stamps, default: int = 60) -> int:
    """The scrape interval, inferred from the stamps themselves.

    Inferred rather than plumbed through: the renderer already has the timestamps, and a
    step passed down from the query would be the *requested* one, which is not the interval
    the samples actually arrived at when a server downsamples.

    The most common gap between consecutive samples, so one missing scrape does not stretch
    the estimate the way a mean would.

    ``default`` is what to answer when there is no gap to measure -- a single sample, or
    none. A caller asking per unit passes the report-wide step, so a unit with one sample
    inherits the interval its peers were scraped at rather than a hardcoded minute.
    """
    ordered = sorted(set(stamps))
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
    if not gaps:
        return default
    return max(set(gaps), key=gaps.count)


def verify_rungs(span: int, windows,
                 fetched: Optional[int] = None) -> List[Tuple[str, Optional[int]]]:
    """``[(label, seconds or None)]`` widest first, the whole fetch always first.

    A rung at least as long as the fetch is dropped rather than printed as a duplicate of
    it: without that, every job shorter than the widest configured window gets three
    identical columns and the ladder says nothing. A 20-minute job therefore shows one
    rung and a 40-hour job shows all of them.

    ``fetched`` is ``--verify WINDOW`` in seconds, and it is the same rule with a second
    source of "as long as". :func:`jobscope.running.range_window` narrows the *query*
    rather than filtering rows afterwards, so under a window the fetch **is** the window:
    ``--verify 2h`` used to print a ``run`` column of 121 samples beside a ``2h`` column
    of 120, the two differing by one sample and the first one named after a run it did
    not cover.

    ``span < fetched`` is the case that would make the label a lie the other way: a
    40-minute job asked for ``--verify 2h`` got its whole run, so ``run`` is what that
    column is.
    """
    # (label, seconds) as the config resolved them -- parsed and ordered once, at load.
    # Sorted again here rather than trusted, because series_shape reads the ladder
    # widest-first and reversing it turns "declining" into its opposite with no error.
    windowed = fetched is not None and span >= fetched
    rungs: List[Tuple[str, Optional[int]]] = [
        (format_duration(fetched) if windowed else "run", None)]
    cap = min(span, fetched) if windowed else span
    rungs += sorted(((label, seconds) for label, seconds in windows if seconds < cap),
                    key=lambda pair: pair[1], reverse=True)
    return rungs


def _one_per_stamp(pairs):
    """The first value at each stamp, oldest first.

    A combined series repeats a host metric on every GPU row of its node, so a
    node-level grouping sees one node's CPU% once per card -- four copies of one
    reading, which is four times the weight in anything that pools them.
    """
    seen: Dict[int, float] = {}
    for stamp, value in pairs:
        seen.setdefault(stamp, value)
    return sorted(seen.items())


def _unit_cells(level: str, multi_job: bool, key: tuple,
                found: dict) -> Tuple[str, ...]:
    """The identity cells for one row, under the ladder's single ``NODE:GPU`` lead.

    :func:`unit_values` is not usable at node level here: it reports how many GPUs the
    row pooled, off a ``gpus`` set :func:`stamped_samples` does not keep. A host metric
    pooled nothing anyway -- it is measured once per node -- so its row names the node
    and stops, which is also what says it is not a per-card reading.
    """
    if level == GPU_LEVEL:
        return unit_values(GPU_LEVEL, multi_job, key, found)
    return ((key[0],) if multi_job else ()) + (key[1],)


@dataclass(frozen=True)
class SeriesFigures:
    """Every figure --verify computes for one unit's one metric, computed once.

    A record rather than thirteen locals because several blocks read these and each
    would otherwise re-derive them from ``pairs`` -- which is how one quantity ends up
    computed two ways and the table disagrees with the conclusion under it.

    Not an :class:`EfficiencyTally`, despite the resemblance: that one accumulates
    across jobs and has an ``add``. This is one pass's output, frozen, and nothing
    should teach it to grow.
    """

    key: tuple
    level: str
    unit: Tuple[str, ...]
    metric: str
    model: str
    bands: Thresholds
    step: int
    pairs: List[Tuple[int, float]]
    limit: Optional[float]
    values: List[float]
    peak: float
    low: float
    swing: Optional[float]
    means: List[Optional[float]]
    share: Optional[float]
    idlemax: Optional[int]
    shape: str
    missing: int
    expected: int
    measured: Optional["Measured"]
    stopped: Optional["IdleSince"]
    bins: List[Tuple[str, int]]
    flags: List[str]

    @property
    def mean(self) -> Optional[float]:
        """The whole-fetch mean -- the widest rung, which is always first."""
        return self.means[0] if self.means else None

    @property
    def scale(self) -> Optional[float]:
        """The fixed axis this metric draws against, or None if it has no scale."""
        return full_scale(self.bands, self.metric, self.model)


@dataclass(frozen=True)
class VerifyFetch:
    """One fetch, reduced: the frame every block shares plus the per-series figures."""

    level: str
    multi_job: bool
    metrics: List[str]
    thresholds: Thresholds
    step: int
    oldest: int
    newest: int
    span: int
    rungs: List[Tuple[str, Optional[int]]]
    figures: List[SeriesFigures]
    user: str = "?"
    window_end: Optional[int] = None

    @property
    def gaps(self) -> int:
        """Scrapes missing across every series, which is what the note counts."""
        return sum(f.missing for f in self.figures)

    @property
    def shown(self) -> List[str]:
        """The metrics that actually carried samples, in the order asked for."""
        seen = {f.metric for f in self.figures}
        return [m for m in self.metrics if m in seen]

    @property
    def lead(self) -> Optional[str]:
        """The metric the timeline is drawn for and the sentences are written about.

        The first *voting* metric in catalog order that has a cutoff and some samples:
        GPU% on the usual view, CPU% under ``--cpu``. One rather than all of them
        because the default set is five GPU metrics plus two host ones, and a strip per
        metric per card is twenty-eight of them for a four-GPU job.
        """
        voting = classify_metrics(self.shown, self.thresholds)
        for header in metrics.in_catalog_order(voting):
            if any(f.metric == header and f.limit is not None for f in self.figures):
                return header
        return None

    @property
    def judged(self) -> List[str]:
        """The metrics that decide the verdict: the ballot plus the floors, in catalog
        order.

        What "the metrics a wasteful job is filtered on" means, and so what this view
        owes the reader figures for -- the ``Graded by best of ...`` line names them,
        and naming a metric as having voted while showing none of its numbers is the
        gap this closes. A column that neither votes nor floors (MEM%, GMEM%) is
        carried by the ladder and not here: it has no cutoff and no say.
        """
        wanted = set(classify_metrics(self.shown, self.thresholds))
        wanted |= set(self.thresholds.floors)
        return [m for m in metrics.in_catalog_order(self.shown)
                if m in wanted and any(f.metric == m and f.limit is not None
                                       for f in self.figures)]

    def on(self, metric: Optional[str]) -> List[SeriesFigures]:
        """Every unit's figures for one metric, in row order."""
        return [f for f in self.figures if f.metric == metric]


def verify_figures(rows: List[dict], metrics: List[str], options: "RenderOptions",
                   windows=(), window_end: Optional[int] = None
                   ) -> Optional[VerifyFetch]:
    """Reduce one fetch to its figures. Pure: no terminal, no colour, no printing.

    ``window_end`` is when the fetch was asked to stop, which the samples cannot say:
    an exporter that died leaves a series that simply ends, and its last sample looks
    like the present. Given it, a series with nothing recent is flagged ``STALE`` and
    no present-tense claim is made about it. Absent, that check is skipped rather than
    guessed at.

    Every rung comes from the *same fetch*, sliced. ``running.range_window`` guarantees
    that is sound -- "aligned, ``--ts 1h`` returns exactly the rows a full ``--ts`` would
    have" -- so the ladder costs one query rather than one per rung.

    Per GPU, with no level parameter: pooling to a node or a job would need the ``nodes``
    and ``gpus`` sets that :func:`unit_values` reads and :func:`stamped_samples` does not
    keep. Adding the level back means teaching the grouping to carry them, not passing a
    string through.
    """
    thresholds = _bands(options)
    # A host metric is not per GPU. A combined series carries CPU%/MEM% on every row,
    # so grouping the whole lot by card gave a 4-GPU job four identical CPU% rows and
    # counted one node's host samples four times in anything pooled. Split by which
    # family serves the column, and group each at the level it is actually measured at.
    host = [m for m in metrics if m in set(cpu.catalog().headers)]
    per_gpu = [m for m in metrics if m not in set(host)]
    grouped = [(GPU_LEVEL, per_gpu, stamped_samples(rows, per_gpu, GPU_LEVEL)),
               (NODE_LEVEL, host, stamped_samples(rows, host, NODE_LEVEL))]
    stamps = {s for _lvl, _ms, groups in grouped for found in groups.values()
              for pairs in found["stamps"].values() for s, _v in pairs}
    if not stamps:
        return None
    step = sampling_step(stamps)
    newest, oldest = max(stamps), min(stamps)
    span = newest - oldest + step
    rungs = verify_rungs(span, windows or (), options.window)
    multi_job = len({key[0] for _lvl, _ms, groups in grouped for key in groups}) > 1

    figures = []
    for level, wanted, groups in grouped:
        for key in sorted(groups, key=unit_order):
            found = groups[key]
            model = job_model({k: {MODEL_KEY: m} for k, m in found["models"].items()})
            # Resolved once per unit and used for every figure below: the cutoff and the
            # shape have to be read off the same bands, or a POWER_W column can report a
            # share taken against this card's floor and a shape taken against no floor.
            bands = thresholds.for_model(model)
            # This unit's own interval, defaulting to the report's. One modal gap over
            # every stamp in the report is the majority node's, and on a job whose nodes
            # scrape at different rates every duration on the other node was wrong.
            here = sampling_step({s for pairs in found["stamps"].values()
                                  for s, _v in pairs}, default=step)
            # How many scrapes the *fetch* should have held, off the grid the whole
            # report shares rather than off each series' own extent. Read per series, an
            # exporter that died halfway reported zero missing scrapes -- the series
            # simply ended, and its last sample looked like the present.
            expected = (newest - oldest) // here + 1
            for metric in wanted:
                pairs = found["stamps"].get(metric)
                if not pairs:
                    continue
                if level == NODE_LEVEL:
                    pairs = _one_per_stamp(pairs)
                limit = cutoff(bands, metric, model)
                values = [v for _s, v in pairs]
                peak = max(values)
                means = [_rung_mean(pairs, newest, seconds) for _label, seconds in rungs]
                swing = median_swing(pairs, here)
                adjacent = sum(1 for (sa, _a), (sb, _b) in zip(pairs, pairs[1:])
                               if sb - sa <= here)
                figures.append(SeriesFigures(
                    key=key, level=level,
                    unit=_unit_cells(level, multi_job, key, found),
                    metric=metric, model=model, bands=bands, step=here, pairs=pairs,
                    limit=limit, values=values, peak=peak, low=min(values),
                    swing=swing, means=means,
                    share=below_share(values, limit),
                    idlemax=longest_idle(pairs, limit, here),
                    shape=series_shape(bands, metric, peak, means, limit),
                    missing=max(0, expected - len(pairs)), expected=expected,
                    measured=measured_time(pairs, limit, here, expected),
                    stopped=went_idle(pairs, limit, here),
                    bins=distribution(values, bands, metric, model),
                    flags=qualifiers(len(pairs), expected, min(values), peak, swing,
                                     adjacent, limit,
                                     newest_age=(None if window_end is None
                                                 else window_end - max(s for s, _v in pairs)),
                                     step=here)))
    if not figures:
        return None
    return VerifyFetch(
        level=GPU_LEVEL, multi_job=multi_job, metrics=list(metrics),
        thresholds=thresholds, step=step, oldest=oldest, newest=newest, span=span,
        rungs=rungs, figures=figures, window_end=window_end,
        user=next((found["user"] for _lvl, _ms, groups in grouped
                   for found in groups.values() if found.get("user")), "?"))


class JobVerdict(NamedTuple):
    """One job's category over the fetched window, and what it was reached by."""

    name: str
    label: str
    judged: Dict[str, float]
    bands: Thresholds
    floors: List[str]
    withheld: str


def job_verdict(data: VerifyFetch) -> JobVerdict:
    """The whole job's verdict, by the same route the rest of the report takes.

    :func:`jobscope.job_eff.classify` over pooled means, with the guards
    ``timeseries_eff`` already carries: a series that carries columns which could have
    voted this job healthy, and has a value for none of them, is ``no-data`` rather
    than whatever the ceilinged metrics say.

    Pooled over *samples* rather than over per-unit means, so a card the exporter
    answered for twice does not weigh the same as one it answered for two hundred
    times -- the argument ``timeseries_stats`` already makes for its own pooling.

    ``withheld`` names the guard that stopped a verdict being asserted, or is empty.
    NO_DATA is not a tier, and reporting a collection gap as waste is the reading that
    gets someone an email.
    """
    means = {}
    for metric in data.shown:
        values = [v for f in data.on(metric) for v in f.values]
        if values:
            means[metric] = sum(values) / len(values)
    voting = classify_metrics(data.shown, data.thresholds)
    model = job_model({f.key: {MODEL_KEY: f.model} for f in data.figures if f.model})
    bands = data.thresholds.for_model(model)
    judged = {m: means[m] for m in metrics.in_catalog_order(voting) if m in means}
    carriers = unceilinged(bands, voting)
    if carriers and not any(m in judged for m in carriers):
        judged = {}
    floor_readings = {h: means.get(h) for h in bands.floors}
    name = classify(judged, bands, floor_readings, columns=voting) or NO_DATA
    # The lead metric's guards decide whether the figures may be spoken for at all.
    # Taken off the lead rather than off every column, because a thin CPU% series is
    # not a reason to withhold a verdict a full GPU% series supports.
    flags = {flag for f in data.on(data.lead) for flag in f.flags}
    withheld = next((f for f in (THIN, SPARSE) if f in flags), "")
    if withheld:
        name = NO_DATA
    # No range for NO_DATA: it is deliberately outside TIERS, so it has no band and no
    # edges to quote -- asking tier_range for them raises, which is the shape of the
    # claim being refused.
    label = ("" if name == NO_DATA
             else tier_range(name, bands, metrics.in_catalog_order(judged)))
    return JobVerdict(
        name=name, label=label, judged=judged, bands=bands,
        floors=[h for h, v in floor_readings.items() if v is not None],
        withheld=withheld)


def _ladder_lines(data: VerifyFetch) -> List[str]:
    """The per-rung table: one row per unit and metric, widest rung first."""
    # SWING sits with MIN/MAX because all three describe the raw signal, before any
    # window means it into a single number -- and it is what says whether that number is
    # reproducible.
    headers = unit_headers(data.level, data.multi_job) + (
        "METRIC", "N", "MIN", "MAX", "SWING") + tuple(
        label for label, _s in data.rungs) + VERIFY_TAIL
    table = []
    for one in data.figures:
        # The identity cells come off the figure, settled when it was built and paired
        # with unit_headers there: a prefix whose length disagrees with the header is
        # silently truncated by the zip in `line`, which is how SHAPE went missing the
        # first time this ran.
        cells = list(one.unit) + [
            one.metric, len(one.values), "%.1f" % one.low, "%.1f" % one.peak,
            "-" if one.swing is None else "%.1f" % one.swing]
        cells += ["-" if m is None else "%.1f" % m for m in one.means]
        cells += ["-" if one.share is None else "%d%%" % round(100 * one.share),
                  "-" if one.idlemax is None else _span(one.idlemax), one.shape]
        table.append([str(c) for c in cells])
    # Sized to the content, header included, so a long hostname widens its column instead
    # of pushing every cell after it out from under its own heading.
    widths = [max(len(row[i]) for row in [list(headers)] + table) + 2
              for i in range(len(headers))]
    lines = []
    for row in [list(headers), None] + table:
        if row is None:
            lines.append("-" * (sum(widths) - 2))
            continue
        lines.append("".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
    return lines


def _ladder_legend(data: VerifyFetch) -> List[str]:
    """What the ladder's columns mean, and what the figures were taken against."""
    lines = []
    if data.gaps:
        # Said rather than absorbed: a thin series is a fact about the collection, and the
        # figures above divide by the samples that exist, not by the wall clock.
        lines.append("  %d scrape(s) missing from these series -- the shares above are "
                     "of measured samples," % data.gaps)
        lines.append("    and a gap breaks an idle stretch rather than extending it")
    lines.append("  SWING = median change between consecutive scrapes. Approaching "
                 "MAX-MIN, the metric moves")
    lines.append("    faster than it is sampled and no mean of it is reproducible -- "
                 "compare MAX and the shape,")
    lines.append("    not the rung means, and expect a different --gpu-source to "
                 "disagree by several points.")
    lines.append("  BELOW = share of samples under the metric's own cutoff; IDLEMAX = "
                 "the longest unbroken run")
    lines.append("    under it, with a gap breaking the run rather than extending it. "
                 "flat-idle is the only")
    lines.append("    shape that says nothing ever ran. Cutoffs: %s"
                 % _cutoff_summary(data.thresholds, data.shown))
    return lines


def _clock(stamp: int) -> str:
    """A wall-clock instant to the minute. Seconds are dropped for the reason
    :func:`jobscope.slurm.format_window` drops them: nobody acts to the second."""
    return time.strftime("%H:%M", time.localtime(stamp))


def _fetch_lines(data: VerifyFetch) -> List[str]:
    """What was fetched and how much of it arrived, before any figure taken over it."""
    shown = data.on(data.lead)
    jobs = ", ".join(sorted({f.key[0] for f in data.figures}))
    model = job_model({f.key: {MODEL_KEY: f.model} for f in data.figures if f.model})
    lines = ["  %-10s%s  %s  %d unit(s)%s"
             % ("Job:", jobs, data.user, len({f.key for f in shown}),
                "  " + model if model else "")]
    lines.append("  %-10s%s .. %s   %s" % ("Window:", _clock(data.oldest),
                                           _clock(data.newest), _span(data.span)))
    measured = sum(len(f.values) for f in shown)
    expected = sum(f.expected for f in shown)
    # Counted on the lead metric alone, because that is what every figure below is
    # taken over. A whole-report gap count belongs to the ladder, whose rows each carry
    # their own -- said here it would contradict this line, which is the shape of the
    # confusion: "210 measured (100%)" beside "3 scrape(s) missing".
    #
    # Said rather than absorbed either way: every share below divides by the samples
    # that exist, not by the wall clock.
    lines.append("  %-10s%d expected at %ds; %d measured (%d%%)%s"
                 % ("Samples:", expected, data.step, measured,
                    round(100.0 * measured / expected) if expected else 100,
                    "" if measured >= expected else
                    " -- %d missing, and every figure below is of what arrived"
                    % (expected - measured)))
    return lines


def _strip_lines(data: VerifyFetch, options: "RenderOptions", width: int,
                 metric: Optional[str] = None) -> List[str]:
    """The timeline: one row per unit, one cell per bucket, on a fixed axis.

    Drawn for the lead metric alone. All the rows share one ``(oldest, newest, cells)``
    frame -- taken from the fetch, not from each unit -- or they are not comparable,
    which is the only thing a stack of strips is for.
    """
    lead = metric or data.lead
    shown = data.on(lead) if lead else []
    if not shown or shown[0].scale is None:
        return []
    label_width = max(len(" ".join(f.unit)) for f in shown)
    # The *measured* extent, not the span: span counts the closing scrape's own
    # interval, and a cell per that is one more cell than there are samples -- which
    # draws a gap where nothing was missing.
    cells = strip_cells(width, max(label_width, 8), data.newest - data.oldest,
                        data.step)
    edges = bucket_edges(data.oldest, data.newest, cells)
    limit = shown[0].limit
    lines = ["%s  (idle below %g, drawn against 0-%g)"
             % (lead, limit, shown[0].scale) if limit is not None else lead]
    for one in shown:
        means = bucket(one.pairs, data.oldest, data.newest, cells)
        peaks = bucket(one.pairs, data.oldest, data.newest, cells, reduce=max)
        lines.append("  %-*s  %s" % (
            label_width, " ".join(one.unit),
            level_strip(means, peaks, one.limit, one.scale, options, lead, one.model)))
    axis = strip_axis(edges, indent=label_width + 4)
    if axis:
        # The cell width labels the rule rather than sitting above it, so the block
        # costs two lines rather than three and the reader learns the scale where the
        # scale is drawn.
        axis[0] = "  %-*s%s" % (label_width, "cell " + _span(
            max(1, (data.newest - data.oldest) // cells)), axis[0][label_width + 2:])
        lines += axis
    return lines


def _measured_lines(data: VerifyFetch, metric: Optional[str] = None) -> List[str]:
    """How long each unit was measured, and how much of that it spent idle.

    Measured time, never the wall clock. The three figures are what was measured, what
    of it cleared the cutoff, and what did not -- and the job line is the instants every
    unit was idle *together*, not the span between the first and the last, which would
    count an exporter outage in the middle as job-wide idleness.
    """
    lead = metric or data.lead
    shown = [f for f in data.on(lead) if f.measured is not None]
    if not shown:
        return []
    label_width = max(max(len(" ".join(f.unit)) for f in shown), 8)
    lines = ["  %-*s  %9s  %9s       %9s       %s"
             % (label_width, "", "MEASURED", "ACTIVE", "IDLE", "LONGEST IDLE")]
    for one in shown:
        held = one.measured
        total = held.idle + held.active
        lines.append("  %-*s  %9s  %9s %3d%%  %9s %3d%%  %s" % (
            label_width, " ".join(one.unit), _span(total), _span(held.active),
            round(100.0 * held.active / total) if total else 0,
            _span(held.idle), round(100.0 * held.idle / total) if total else 0,
            "none" if not one.idlemax else _span(one.idlemax)))
    if len(shown) > 1:
        together = concurrent_idle(
            [idle_stamps(f.pairs, f.limit) for f in shown], data.step)
        wasted = sum(f.measured.idle for f in shown) / 3600.0
        allocated = sum(f.measured.idle + f.measured.active for f in shown) / 3600.0
        lines.append("  every unit idle at the same scrape for %s of %s; %.1f of %.1f "
                     "unit-hours idle" % (_span(together), _span(data.span), wasted,
                                          allocated))
    return lines


def _metrics_table(data: VerifyFetch) -> List[str]:
    """One line per metric the verdict was taken over, pooled across units.

    The whole of the default output's evidence, and the reason it is a table rather than
    a stack of blocks: the ``Graded by best of ...`` line names every metric that voted,
    and one line each is what lets them be read down. It is the difference between
    "GPU% 98.9, working" and the same card at SM_ACT% 10.3 with TENSOR% flat -- busy by
    duty cycle and barely computing, which is the waste a duty cycle alone cannot show.

    Pooled over samples rather than over per-unit means, for the reason
    :func:`job_verdict` pools that way. ``UNITS`` says what was pooled, because a host
    metric is measured once per node and a GPU one once per card, and without it a
    reader cannot tell why the sample counts differ. Which *unit* was idle is the
    verdict's per-unit sentences, and the timeline behind ``--full``.
    """
    if not data.judged:
        return []
    rows = []
    for metric in data.judged:
        shown = data.on(metric)
        values = [v for f in shown for v in f.values]
        if not values:
            continue
        one = shown[0]
        longest = max((f.idlemax or 0) for f in shown)
        held = [f.measured for f in shown if f.measured is not None]
        active = sum(m.active for m in held)
        idle = sum(m.idle for m in held)
        rows.append([
            metric, "%d %s" % (len(shown), "host" if one.level == NODE_LEVEL else "GPU"),
            str(len(values)), "%.1f" % min(values), "%.1f" % max(values),
            "%.1f" % (sum(values) / len(values)),
            _span(active), _span(idle),
            "none" if not longest else _span(longest), worst_shape(shown)])
    if not rows:
        return []
    # ACTIVE and IDLE are measured time summed over the units, so on a 4-GPU job they
    # total four card-hours per hour of window -- unit-time, like the GPU-hours the
    # summary charges. IDLEMAX is the longest *unbroken* stretch on any one unit, which
    # a total cannot give: four minutes between batches and three hours of a stopped job
    # sum the same and mean the opposite.
    headers = ["METRIC", "UNITS", "N", "MIN", "MAX", "MEAN", "ACTIVE", "IDLE",
               "IDLEMAX", "SHAPE"]
    widths = [max(len(r[i]) for r in [headers] + rows) + 2 for i in range(len(headers))]
    lines = ["  The metrics the verdict was taken over  (ACTIVE/IDLE are measured time, "
             "summed over units)"]
    for row in [headers] + rows:
        lines.append("    " + "".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
    return lines


def worst_shape(figures: List[SeriesFigures]) -> str:
    """The most actionable shape among a metric's units, worst first.

    A pooled row cannot carry four shapes, and the one worth surfacing is the one that
    would change what someone does -- so a metric flat-idle on one card of four says
    flat-idle here rather than averaging into the majority's `steady`.
    """
    order = (NO_DATA, FLAT_IDLE, DECLINING, BURSTY)
    found = {f.shape for f in figures}
    return next((name for name in order if name in found), STEADY)


def _distribution_lines(data: VerifyFetch, options: "RenderOptions") -> List[str]:
    """Where the lead metric's time went, binned on the edges the verdict is taken on.

    A different question from the timeline: it says whether a job is genuinely on/off
    or steadily mediocre, which changes what to do about it. The bins are
    ``Thresholds.tier`` calls, so this cannot disagree with the conclusion under it.
    """
    lines = []
    for metric in data.judged:
        block = _one_distribution(data, options, metric)
        if block:
            lines += block
    if not lines:
        return []
    # Stated once above the stack rather than on each block: it is the same sentence
    # about every one of them, and repeated six times it stops being read.
    return ["  Time by band, per metric  (the cutoffs the verdict is taken on)"] + lines


def _one_distribution(data: VerifyFetch, options: "RenderOptions",
                      metric: str) -> List[str]:
    """One metric's time split across its own bands, pooled over its units."""
    shown = data.on(metric)
    if not shown or not shown[0].bins:
        return []
    totals: Dict[str, int] = {}
    for one in shown:
        for name, count in one.bins:
            totals[name] = totals.get(name, 0) + count
    measured = sum(totals.values())
    if not measured:
        return []
    step = shown[0].step
    # One band holding nearly all of it says the same thing in one line as in five, and
    # the remainder cannot be a sustained anything. It matters more with a block per
    # metric than it did with one: six metrics at five bands each is a page.
    top, count = max(totals.items(), key=lambda pair: pair[1])
    if count / measured >= 0.95:
        return ["    %-8s %d%% %s, %s -- nothing measured in any other band"
                % (metric, round(100.0 * count / measured), top, _span(count * step))]
    items = []
    for name, held in totals.items():
        band = BUCKET_OF.get(name, "") if options.color else ""
        items.append((name, 100.0 * held / measured, band, _span(held * step)))
    return ["    %s" % metric] + bar_lines(items, indent="      ")


def _verdict_lines(data: VerifyFetch, options: "RenderOptions") -> List[str]:
    """The conclusion, and one sentence per unit saying what it rests on."""
    lead = data.lead
    found = job_verdict(data)
    lines = []
    if found.judged:
        lines.append("Graded %s." % classify_description(
            metrics.in_catalog_order(found.judged), found.floors, found.bands))
    if found.withheld:
        lines.append(_withheld_sentence(data, found))
    else:
        role = next((r for n, r in CATEGORIES if n == found.name), "")
        text = ("Verdict: %s (%s)" % (found.name, found.label) if found.label
                else "Verdict: %s" % found.name)
        # The count, because pooling hides it: a job using one card of four and one
        # using four badly reach the same verdict, and only the first is fixed by
        # asking for fewer GPUs. Named for the lead metric's own cutoff.
        shown = data.on(lead)
        dead = [f for f in shown if f.shape == FLAT_IDLE]
        if dead and len(shown) > 1:
            text += " -- %d of %d units never cleared %g" % (
                len(dead), len(shown), shown[0].limit)
        lines.append(tint(text, role) if options.color and role else text)
        # A verdict taken over a mean, on a metric whose mean is an artefact of when
        # the sampler looked, has to say so where the verdict is -- not only in the
        # sentence below it, which a reader who has their answer will not reach.
        if any(ALIASED in f.flags for f in shown):
            lines.append("  on a mean that is not reproducible; read MIN/MAX and the "
                         "idle split above, or --full for the time by band")
        if any(STALE in f.flags for f in shown):
            lines.append("  and only up to the newest scrape that arrived -- see below")
    units = data.on(lead)
    width = max([len(" ".join(f.unit)) for f in units] + [8]) + 2
    for one in units:
        lines.append("  %-*s%s" % (width, " ".join(one.unit),
                                   _unit_sentence(data, one)))
    return lines


def _withheld_sentence(data: VerifyFetch, found: JobVerdict) -> str:
    """Why there is no verdict, naming what would fix it."""
    lead = data.lead
    one = next(iter(data.on(lead)), None)
    if found.withheld == THIN:
        return ("No verdict: %d scrape(s) over %s. Nothing is graded on fewer than %d, "
                "about %s at this %ds scrape, because below that the measured idle "
                "share turns on one sample."
                % (one.expected, _span(data.span), MIN_SAMPLES,
                   _span(MIN_SAMPLES * data.step), data.step))
    return ("No verdict: %d of %d expected scrapes arrived (%d%%). A verdict over less "
            "than half a window describes the minority of it -- check the exporter "
            "before reading the figures above as the job's behaviour."
            % (len(one.values), one.expected,
               round(100.0 * len(one.values) / one.expected)))


def _unit_sentence(data: VerifyFetch, one: SeriesFigures) -> str:
    """What this unit did, in the order that decides which fact leads.

    never ran > stopped > declining > bursty > working. ``never ran`` is first because
    it is the only shape that licenses a kill and must not be reachable any other way;
    ``stopped`` outranks ``declining`` because "stopped at 15:52" is the more actionable
    of the same finding, and because ``declining`` reads rung means that a swing warning
    may have just discredited.
    """
    metric, limit = one.metric, one.limit
    aliased = ALIASED in one.flags
    if STALE in one.flags:
        return ("last measured at %s, %s before the end of the window -- whether it is "
                "idle now is not measured"
                % (_clock(max(s for s, _v in one.pairs)),
                   _span(data.window_end - max(s for s, _v in one.pairs))))
    if one.shape == FLAT_IDLE:
        return ("never ran: no %s sample reached %g anywhere in this window (peak %.1f)"
                % (metric, limit, one.peak))
    if one.stopped is not None:
        return ("ran, then stopped: last sustained work at %s, idle at every scrape "
                "since -- %s, through the newest at %s"
                % (_clock(one.stopped.at), _span(one.stopped.seconds),
                   _clock(max(s for s, _v in one.pairs))))
    if aliased:
        return ("flapping: %s moves a median of %.1f between %ds scrapes over a "
                "%.1f-point range, so no mean of it is reproducible -- read the idle "
                "split, not the mean"
                % (metric, one.swing, one.step, one.peak - one.low))
    if one.shape == DECLINING:
        stated = ", ".join("%s %.1f" % (label, mean)
                           for (label, _s), mean in zip(data.rungs, one.means)
                           if mean is not None)
        return "declining: %s" % stated
    if one.shape == BURSTY:
        return ("bursty: %s peaked at %.1f and was below %g at %d%% of scrapes; "
                "longest idle %s"
                % (metric, one.peak, limit, round(100 * (one.share or 0)),
                   _span(one.idlemax or 0)))
    return ("working: %s %.1f, never below %g, still working at the newest scrape (%s)"
            % (metric, one.mean or 0.0, limit,
               _clock(max(s for s, _v in one.pairs))))


def _source_lines(data: VerifyFetch) -> List[str]:
    """Which exporter served each column. --verify is the pre-action check, and where
    the number came from is part of the answer."""
    lines = []
    for source, columns in dcgm.catalog().resolved.by_source():
        from_here = [c for c in columns if c in data.metrics]
        if from_here:
            lines.append("  %s <- %s" % (" ".join(from_here), source))
    return lines


def _verify_csv(data: VerifyFetch, options: "RenderOptions", out) -> None:
    """One row per unit and metric, carrying every figure the block computed.

    Always the full set, whatever ``--full`` says: that flag is about how much
    of a terminal to spend, and a consumer that has asked for CSV wants the columns.

    Durations in seconds, not in :func:`_span`'s two-unit form -- that one exists for
    reading and its own docstring says it is not for round-tripping. The verdict is the
    job's, repeated on every row, because a row is what gets sorted or joined on and a
    conclusion that only exists in a footer cannot travel with one.
    """
    writer = csv.writer(out, lineterminator="\n")
    found = job_verdict(data)
    rungs = [label for label, _s in data.rungs]
    if options.header:
        writer.writerow(["JOBID", "USER", "NODE", "GPU", "METRIC", "MODEL", "N",
                         "EXPECTED", "MIN", "MAX", "SWING"] + rungs
                        + ["BELOW", "IDLEMAX_S", "IDLE_S", "ACTIVE_S", "MISSING_S",
                           "WENT_IDLE", "SHAPE", "FLAGS", "VERDICT"])
    for one in data.figures:
        held = one.measured
        writer.writerow(
            [one.key[0], data.user, one.key[1] if len(one.key) > 1 else "",
             one.key[2] if len(one.key) > 2 else "", one.metric, one.model,
             len(one.values), one.expected, "%.1f" % one.low, "%.1f" % one.peak,
             "" if one.swing is None else "%.1f" % one.swing]
            + ["" if m is None else "%.1f" % m for m in one.means]
            + ["" if one.share is None else "%.4f" % one.share,
               "" if one.idlemax is None else one.idlemax,
               "" if held is None else held.idle,
               "" if held is None else held.active,
               "" if held is None else held.missing,
               "" if one.stopped is None else one.stopped.at,
               one.shape, " ".join(one.flags), found.name])


def verify_report(rows: List[dict], metrics: List[str], options: "RenderOptions",
                  out=None, windows=(), window_end: Optional[int] = None) -> None:
    """The pre-action check on one job: what its series is shaped like, and what rests
    on that.

    Extends the ``--ts --stats`` row identity rather than inventing a layout, so a reader
    who knows that table knows this one: same ``NODE:GPU``/``METRIC`` lead, a mean per rung
    where it had one mean, then the three figures a decision needs.

    A dispatcher over block builders that each return lines and none of which print --
    the idiom :func:`bar_lines` and :func:`in_columns` already follow here, and what
    lets every block be checked without a terminal.
    """
    out = out or sys.stdout
    data = verify_figures(rows, metrics, options, windows, window_end)
    if data is None:
        print("no samples to verify", file=out)
        return
    if options.csv:
        return _verify_csv(data, options, out)
    width = terminal_width(out)
    blocks = [_fetch_lines(data)]
    if options.verify_full:
        # Every metric that decides the verdict gets the full treatment, which is what
        # the flag buys: five GPU metrics over four cards is twenty strips, too many to
        # lead with and exactly what someone who asked for all of them wants. The band
        # split is here rather than in the default for the same reason -- it is six
        # blocks of up to five bars, which is a page to read past on the way to a
        # verdict, and the table below carries the same metrics in one line each.
        for metric in data.judged:
            blocks += [_strip_lines(data, options, width, metric),
                       _measured_lines(data, metric)]
        blocks += [_distribution_lines(data, options),
                   _ladder_lines(data), _ladder_legend(data)]
    else:
        # One table and the verdict. The timeline, the per-unit idle split and the band
        # breakdown are all evidence for what the table already states in a line per
        # metric, and a reader who wants them is asking --full for them. When a job went
        # idle is still here: the verdict's per-unit sentences name the clock time.
        blocks.append(_metrics_table(data))
    blocks += [_source_lines(data), _verdict_lines(data, options)]
    for lines in blocks:
        if not lines:
            continue
        for line in lines:
            print(line, file=out)
        print(file=out)


def _span(seconds: int) -> str:
    """A duration for reading, not for round-tripping.

    ``running.format_duration`` is the inverse of ``parse_duration`` and so only uses a
    unit that divides exactly -- 2823 minutes comes out as "2823m", which is the number
    this column exists to make legible. Two units, largest first.
    """
    if seconds < 60:
        return "%ds" % seconds
    minutes, hours = seconds // 60, seconds // 3600
    if hours < 1:
        return "%dm" % minutes
    if hours < 24:
        return "%dh%02dm" % (hours, minutes % 60)
    return "%dd%02dh" % (hours // 24, hours % 24)


def _rung_mean(pairs, newest: int, seconds: Optional[int]) -> Optional[float]:
    """The mean over the last ``seconds`` of ``pairs``, or all of it when None."""
    chosen = [v for s, v in pairs if seconds is None or s > newest - seconds]
    return sum(chosen) / len(chosen) if chosen else None


def _cutoff_summary(thresholds, metrics) -> str:
    """The cutoffs in play, so the table needs no separate legend lookup."""
    parts = []
    for metric in metrics:
        limit = cutoff(thresholds, metric)
        if limit is not None:
            parts.append("%s <%g" % (metric, limit))
    return ", ".join(parts) or "none configured"
