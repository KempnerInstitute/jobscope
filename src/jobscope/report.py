"""Rendering of the summary, detail, per-GPU DCGM, and time-series views.

Output layout (spacing, context lines, and above all the CSV shape) is kept
stable: the CSV emitted here is what ``jobscope plot`` parses.
"""

import csv
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .blob import blob_capacity, blob_detail, blob_metrics
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
from .live import Gpu, LiveJob, build_columns, job_sort_key, timeseries_step
from .prometheus import PrometheusClient
from .sacct import JobRecord, Selection


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
    # Tint %-metric cells by their threshold band. Off unless the caller has
    # established that the destination is a terminal that wants colour.
    color: bool = False
    thresholds: Optional["Thresholds"] = None


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
    return pairs


def extend_detail_row(row, per_gpu, duration=None, min_runtime=None, diagnose_on=False):
    """Append the DCGM cells and (optionally) the DIAG cell to a blob_detail row."""
    values = per_gpu.get((row[0], str(row[1])), {})
    out = tuple(row) + tuple(format_by_header(h, values.get(h)) for h in DCGM_HEADERS)
    if diagnose_on:
        out = out + (diagnose_dcgm(values, duration, min_runtime),)
    return out


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
                 csv_row: str = "UsedPerGPU") -> None:
        self.header = header            # the column being banded, "GPU%" or "CPU%"
        self.thresholds = thresholds
        self.label = label              # "GPU-hours", "GPUs", "Core-hours", ...
        self.unit = unit                # "h" for resource-hours, "" for a count
        self.scale = scale              # seconds -> hours, or 1 for a bare count
        # What the pooled row is called. Kept to 12 characters, the width of the
        # JOBID column: a longer label shifts every cell in the row one right.
        self.row = row
        self.csv_row = csv_row
        self.bands = {band: [0, 0.0] for band in ("red", "yellow", "green")}
        self.used = 0.0                 # resource-time actually utilized
        self.total = 0.0                # resource-time allocated
        self.worst: List[Tuple[float, str, str, float, float]] = []

    def add(self, jobid: str, user: str, value: Optional[float], weight: float) -> None:
        band = self.thresholds.grade(self.header, value)
        if not band or weight <= 0:
            # Ungraded (no measurement) or unweighable: counting it would either
            # invent a utilization or give it no resource to account for.
            return
        self.bands[band][0] += 1
        self.bands[band][1] += weight
        self.used += (value / 100.0) * weight
        self.total += weight
        if band == "red":
            # Ranked by resource-time *wasted*, not held: a 100-hour job at 24% is
            # a bigger finding than a 10-hour job at 0%.
            self.worst.append(((1 - value / 100.0) * weight, jobid, user, weight, value))
            self.worst.sort(key=lambda item: -item[0])
            del self.worst[self.WORST:]

    def cutoff(self) -> float:
        """The red threshold for this column, from the site config."""
        return self.thresholds.red_map().get(self.header, self.thresholds.default)

    def _amount(self, weight: float) -> str:
        return "%.1f%s" % (weight / self.scale, self.unit) if self.unit \
            else "%d" % round(weight / self.scale)

    def totals_line(self) -> str:
        """Allocated / used / idle. The unit is left to the row label."""
        idle = self.total - self.used
        bare = self._amount if not self.unit else (
            lambda w: "%.1f" % (w / self.scale))
        return "%s alloc  %s used  %s idle (%d%%)" % (
            bare(self.total), bare(self.used), bare(idle),
            round(100 * idle / self.total) if self.total else 0)

    def bands_line(self) -> str:
        """Each band's share of the jobs *and* of the resource-time.

        Printing both is the point: "5% of jobs holding 64% of the GPU-hours" is
        the finding, and either number alone conceals it.
        """
        red = self.cutoff()
        jobs_total = sum(count for count, _ in self.bands.values())
        parts = []
        for band, edge in (("red", "<%g" % red), ("yellow", "<%g" % (2 * red)), ("green", "")):
            jobs, weight = self.bands[band]
            parts.append("%s%s %d job%s (%d%%)/%s (%d%%)" % (
                band, edge, jobs, "" if jobs == 1 else "s",
                round(100 * jobs / jobs_total) if jobs_total else 0,
                self._amount(weight),
                round(100 * weight / self.total) if self.total else 0))
        return "%s  %s" % (self.header, "  ".join(parts))

    def worst_line(self) -> str:
        return "  ".join("%s %s@%d%% %s" % (jid, self._amount(weight), round(value), user)
                         for _idle, jid, user, weight, value in self.worst)

    def csv_cells(self) -> List[str]:
        cells = ["column=%s" % self.header,
                 "allocated=%s" % self._amount(self.total),
                 "used=%s" % self._amount(self.used)]
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
        # Both resources are tallied; finish() prints whichever the view is about.
        # Under time weighting the weights are resource-seconds, so they render as
        # hours; otherwise they are bare counts of GPUs or cores.
        thresholds = options.thresholds or Thresholds(**DEFAULT_THRESHOLDS)
        hours = options.time_weighted
        self.tallies = {
            "gpu": EfficiencyTally("GPU%", thresholds, "GPU-hours" if hours else "GPUs",
                                   "h" if hours else "", 3600.0 if hours else 1.0,
                                   row="Used/GPU-hr:" if hours else "Used/GPU:",
                                   csv_row="UsedPerGPUHour" if hours else "UsedPerGPU"),
            "cpu": EfficiencyTally("CPU%", thresholds, "Core-hours" if hours else "Cores",
                                   "h" if hours else "", 3600.0 if hours else 1.0,
                                   row="Used/cpu-hr:" if hours else "Used/cpu:",
                                   csv_row="UsedPerCPUHour" if hours else "UsedPerCPU"),
        }
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
            cells.append(_SGR[band] + text + _RESET if band else text)
        return " ".join(cells)

    def _band(self, header: str, cell) -> str:
        """The grade for a rendered cell, or "" when it is not a graded metric."""
        options = self.options
        if not options.color or options.thresholds is None:
            return ""
        try:
            value = float(cell)
        except (TypeError, ValueError):      # "-", "", a job name, a runtime
            return ""
        return options.thresholds.grade(header, value)

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
                    if key in self.tallies:
                        self.tallies[key].add(jid, row["USER"], value, weights[key])
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
        if self.count == 1:
            return

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

        # Where that resource-time actually went. The GPU tally leads when the
        # selection has GPU work and the view shows it; otherwise the CPU one, so a
        # --cpu run is not silently blank.
        tally = self.tallies["gpu"]
        if options.view == "cpu" or not tally.total:
            tally = self.tallies["cpu"]

        # The job counts differ whenever the selection mixes CPU-only and GPU work:
        # a CPU-only job has no GPU% to average, so it is absent from the GPU
        # figures rather than counted as zero.
        counts = ["cpu-jobs=%d" % self.sums["cpu"][1]]
        if options.view != "cpu":
            # A --cpu run hid the GPU columns; repeating GPU totals here is noise.
            counts.append("gpu-jobs=%d" % self.sums["gpu"][1])
            if self.gpu_total:
                counts.append("gpus=%d" % self.gpu_total)
        # gpu-hours is not repeated here: the resource line above already states the
        # weighted row's denominator, and states it split into used and idle.
        if self.unweighted:
            counts.append("no-runtime=%d" % self.unweighted)

        def padded(label, cells):
            """A footer row padded to the header width, so the CSV stays rectangular.
            parse_csv drops it by its first cell either way."""
            return ([label] + cells + [""] * len(self.headers))[:len(self.headers)]

        if options.csv:
            used_row["JOBID"] = tally.csv_row
            self.writer.writerow([used_row[h] for h in self.headers])
            if tally.total:
                self.writer.writerow(padded(tally.label.replace("-", ""),
                                            tally.csv_cells()))
                if tally.worst:
                    self.writer.writerow(padded("Worst", [
                        "%s=%s@%d" % (jid, tally._amount(weight), round(value))
                        for _idle, jid, _user, weight, value in tally.worst]))
            self.writer.writerow(padded("Jobs", counts))
        else:
            used_row["JOBID"] = tally.row
            if options.header:
                print("-" * len(self._line({c.header: c.header for c in self.columns},
                                           color=False)), file=self.out)
            print(self._line(used_row), file=self.out)
            if tally.total:
                print("%-12s %s" % (tally.label + ":", tally.totals_line()), file=self.out)
                print("%-12s %s" % ("Bands:", tally.bands_line()), file=self.out)
                if tally.worst:
                    print("%-12s %s" % ("Worst:", tally.worst_line()), file=self.out)
            print("%-12s %s" % ("Jobs:", "  ".join(counts)), file=self.out)


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
        self._started = False

    def _line(self, cells) -> str:
        return " ".join(c.fmt.format(str(cells[c.index])) for c in self.columns)

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
                print(file=self.out)
        self.out.flush()

    def finish(self) -> None:
        self._start()
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
    if options.header:
        writer.writerow(["JOBID", "EPOCH", "TIME", "NODE", "GPU"]
                        + [header for _key, header, _dec in columns])

    for jid in jobids:
        record = records.get(jid)
        gpus = discover_gpus(record, client, timeout) if record else []
        if not gpus:
            print("warn: job %s has no GPU samples" % jid, file=sys.stderr)
            continue
        uuid_to = {g["uuid"]: (g["node"], g["minor"]) for g in gpus}
        regex = "^(" + "|".join(uuid_to) + ")$"
        # --step wins; otherwise never finer than the scrape interval, and coarse
        # enough to stay under Prometheus' points-per-series cap on a long job.
        span = timeseries_step(record.duration, sampling_period, step)
        series: Dict[str, dict] = {uuid: {} for uuid in uuid_to}
        for spec in ts_specs:
            for result in client.query_range(
                    '%s{%s=~"%s"}' % (spec.metric, spec.uuid_label, regex),
                    record.start, record.end, span, timeout):
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
            writer.writerow(row)


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
