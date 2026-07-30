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

from .blob import blob_detail, blob_metrics
from .dcgm import (
    ALL_SPECS,
    DCGM_HEADERS,
    DEFAULT_SPECS,
    DERIVED_COLUMNS,
    DESCRIPTIONS,
    LIVE_DESCRIPTIONS,
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
from .live import Gpu, LiveJob, LiveMetrics, build_columns, job_sort_key
from .prometheus import PrometheusClient
from .sacct import JobRecord, Selection


@dataclass(frozen=True)
class Column:
    """A rendered column: header, str.format spec, group, and (detail) row index."""

    header: str
    fmt: str
    group: str
    index: Optional[int] = None


SUMMARY_COLUMNS: List[Column] = [
    Column("JOBID", "{:<12}", "id"),
    Column("STATE", "{:<9}", "id"),
    Column("NODES", "{:<5}", "cpu"),
    Column("GPUS", "{:<4}", "gpu"),
    Column("CPU%", "{:<6}", "cpu"),
    Column("MEM%", "{:<6}", "cpu"),
    Column("GPU%", "{:<6}", "gpu"),
    Column("GMEM%", "{:<7}", "gpu"),
    Column("SM_ACT%", "{:<8}", "dcgm"),
    Column("OCC%", "{:<7}", "dcgm"),
    Column("TENSOR%", "{:<8}", "dcgm"),
    Column("DRAM%", "{:<7}", "dcgm"),
    Column("POWER_W", "{:<8}", "dcgm"),
    Column("DIAG", "{:<22}", "diag"),
    Column("RUNTIME", "{:<12}", "id"),
    Column("NAME", "{}", "id"),
]

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
     "Coarse -- says the GPU was occupied in time, not how hard. Use SM_ACT%/OCC% (gpu view) for that."),
    ("GMEM%", "blob (nvidia_gpu_memory_used)",
     "Peak GPU memory used / total, summed over the job's GPUs. A high-water mark, not a time-average."),
    ("SM_ACT%", "DCGM_FI_PROF_SM_ACTIVE (gpu view)",
     "Fraction of time at least one warp was resident on an SM, averaged across all SMs. Low while "
     "GPU% is high means the GPU was barely loaded (parked / underfed)."),
    ("OCC%", "DCGM_FI_PROF_SM_OCCUPANCY (gpu view)",
     "SM occupancy: fraction of warp slots filled, averaged over SMs and time. Low occupancy means "
     "kernels under-fill the GPU -- small launches, or register / shared-memory limits."),
    ("TENSOR%", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE (gpu view)",
     "Fraction of time the tensor-core pipe was active. High only for mixed-precision matmul work "
     "(fp16/bf16/tf32); ~0 means the tensor cores sat idle."),
    ("DRAM%", "DCGM_FI_PROF_DRAM_ACTIVE (gpu view)",
     "Fraction of time the device-memory (HBM) interface was busy. High while SM_ACT% is low suggests "
     "the job is memory-bound."),
    ("POWER_W", "DCGM_FI_DEV_POWER_USAGE (gpu view)",
     "Mean board power draw over the run, in watts. Near-idle watts mean the GPU was not really working."),
]


@dataclass
class RenderOptions:
    """Flags shared by the rendering functions."""

    view: str = "gpu"
    show_dcgm: bool = False
    diagnose: bool = False
    csv: bool = False
    header: bool = True
    min_runtime: int = 180


def fmt_context(label: str, value: str) -> str:
    """A '  Label:     value' context line."""
    return "  %-11s%s" % (label + ":", value)


def cols_for(columns: List[Column], view: str, dcgm: bool = False,
             diagnose: bool = False) -> List[Column]:
    """The columns to show for the chosen view.

    DCGM columns belong to the gpu view only (the only one that pulls DCGM); DIAG
    is added by --diagnose there. cpu/cgpu are blob-only and offline.
    """
    out = []
    for col in columns:
        group = col.group
        if (group == "id"
                or (group == "cpu" and view in ("cpu", "cgpu"))
                or (group == "gpu" and view in ("gpu", "cgpu"))
                or (group == "dcgm" and dcgm and view == "gpu")
                or (group == "diag" and diagnose and dcgm and view == "gpu")):
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
    pairs = [("User", selection.user)]
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


class SummaryRenderer:
    """Streaming form of summarize(): add() chunks as they arrive, then finish().

    Output is byte-identical to one summarize() call over the concatenated
    chunks: the context/header block prints once (on the first add or finish),
    rows print per add, and the Mean footer (when more than one row rendered)
    or the empty-selection message prints on finish.
    """

    def __init__(self, context: List[Tuple[str, str]], options: RenderOptions, out=None) -> None:
        self.out = out or sys.stdout
        self.options = options
        self.context = context
        self.columns = cols_for(SUMMARY_COLUMNS, options.view, options.show_dcgm, options.diagnose)
        self.headers = [c.header for c in self.columns]
        self.writer = csv.writer(self.out, lineterminator="\n") if options.csv else None
        self.count = 0
        self.sums = {key: [0, 0] for key in ("cpu", "mem", "gpu", "gmem")}  # [total, count]
        self.sums_dcgm = {header: [0.0, 0] for header in DCGM_HEADERS}
        self._started = False

    def _line(self, row: dict) -> str:
        return " ".join(c.fmt.format(str(row.get(c.header, ""))) for c in self.columns)

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
            header_line = self._line({c.header: c.header for c in self.columns})
            print(header_line, file=self.out)
            print("-" * len(header_line), file=self.out)

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
                "STATE": record.state if record else "?",
                "NODES": record.nodes if record else "-",
                "GPUS": str(record.gpus) if record and record.gpus else "-",
                "RUNTIME": record.runtime if record else "-",
                "NAME": record.name if record else "(job not found)",
            }
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
            if do_dcgm:
                overall = dcgm_data.get(jid, ({}, {}))[0]
                for header in DCGM_HEADERS:
                    value = overall.get(header)
                    row[header] = format_by_header(header, value)
                    if value is not None:
                        self.sums_dcgm[header][0] += value
                        self.sums_dcgm[header][1] += 1
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

        def mean(key: str) -> str:
            total, count = self.sums[key]
            return str(round(total / count)) if count else "-"

        mean_row = {c.header: "" for c in SUMMARY_COLUMNS}
        mean_row["CPU%"], mean_row["MEM%"] = mean("cpu"), mean("mem")
        mean_row["GPU%"], mean_row["GMEM%"] = mean("gpu"), mean("gmem")
        if options.show_dcgm:
            for header in DCGM_HEADERS:
                total, count = self.sums_dcgm[header]
                mean_row[header] = format_by_header(header, total / count) if count else "-"
        if options.csv:
            mean_row["JOBID"] = "Mean"
            self.writer.writerow([mean_row[h] for h in self.headers])
        else:
            mean_row["JOBID"] = "Mean:"
            if options.header:
                print("-" * len(self._line({c.header: c.header for c in self.columns})),
                      file=self.out)
            print(self._line(mean_row), file=self.out)


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
    """Per-GPU DCGM profiling table, one row per GPU (no averaged row, no DIAG)."""
    out = out or sys.stdout
    jobids = [j for j in jobids if j in records and records[j].gpus]
    # columns_for drops hidden specs and inserts the derived columns, so this view
    # shows the same set as `jobscope live`.
    columns = columns_for(specs)
    headers = [header for _key, header, _dec in columns]

    def cells(values: dict) -> List[str]:
        return [format_number(values.get(header), dec) for _k, header, dec in columns]

    def gpu_key(node_minor):
        return (node_minor[0], gpu_minor_key(node_minor[1]))

    if options.csv:
        writer = csv.writer(out, lineterminator="\n")
        if options.header:
            for label, value in context:
                writer.writerow([label, value])
            writer.writerow(["JOBID", "STATE", "NAME", "NODE", "GPU", "DUR_S"] + headers)
        for jid in jobids:
            record = records.get(jid)
            per_gpu = dcgm_data.get(jid, ({}, {}))[1]
            for node_minor in sorted(per_gpu, key=gpu_key):
                values = per_gpu[node_minor]
                writer.writerow(
                    [jid, record.state if record else "?", record.name if record else "?",
                     node_minor[0], node_minor[1], record.duration if record else ""]
                    + cells(values))
        return

    if options.header:
        for label, value in context:
            print(fmt_context(label, value), file=out)
        print(file=out)
    if not jobids:
        print("(no GPU jobs in this selection)", file=out)
        return

    header_cells = ["NODE", "GPU"] + headers
    widths = [16, 4] + [max(len(h), 8) + 1 for h in headers]

    def line(cells) -> str:
        return " ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    for jid in jobids:
        record = records.get(jid)
        window = ""
        if record and record.start:
            window = "   (%s .. %s, %ds)" % (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(record.start)),
                time.strftime("%H:%M", time.localtime(record.end)),
                record.duration or 0)
        print("Job %s  [%s]  %s%s" % (jid, record.state if record else "?",
                                      record.name if record else "?", window), file=out)
        per_gpu = dcgm_data.get(jid, ({}, {}))[1]
        if not per_gpu:
            print("  (no GPU samples for this job -- too short, no DCGM data, or beyond retention)\n",
                  file=out)
            continue
        if options.header:
            header_line = line(header_cells)
            print("  " + header_line, file=out)
            print("  " + "-" * len(header_line), file=out)
        for node_minor in sorted(per_gpu, key=gpu_key):
            values = per_gpu[node_minor]
            print("  " + line([node_minor[0], node_minor[1]] + cells(values)), file=out)
        print(file=out)


def dcgm_timeseries(jobids: List[str], records: Dict[str, JobRecord],
                    specs: List[MetricSpec], client: PrometheusClient,
                    timeout: Optional[float], options: RenderOptions,
                    out=None) -> None:
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
        step = max(sampling_period, record.duration // 10000 + 1)  # keep under Prometheus' point cap
        series: Dict[str, dict] = {uuid: {} for uuid in uuid_to}
        for spec in ts_specs:
            for result in client.query_range(
                    '%s{%s=~"%s"}' % (spec.metric, spec.uuid_label, regex),
                    record.start, record.end, step, timeout):
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

# Identity columns of the live table, with their widths. One row is one GPU, so
# NODE names that GPU's own host rather than the job's whole nodelist.
LIVE_ID_COLUMNS: List[Tuple[str, int]] = [
    ("JOBID", 12), ("USER", 12), ("NODE", 15), ("NAME", 15)]

LIVE_READINGS = {
    False: "instantaneous snapshot (one scrape; not comparable to jobstats)",
    True: "folded over each job's runtime -- utilization averaged, memory peak "
          "(comparable to jobstats)",
}


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


def live_report(jobs: Dict[int, LiveJob], metrics: LiveMetrics, gpus: Dict[str, Gpu],
                specs: List[MetricSpec], context: List[Tuple[str, str]],
                options: RenderOptions, average: bool = False, out=None) -> None:
    """The live table: one row per GPU, across all selected jobs.

    Deliberately flat rather than the per-job blocks :func:`dcgm_report` uses --
    the live view is usually scanned across many jobs at once looking for the idle
    one, which a flat table sorted by job ID supports and stacked blocks do not.
    """
    out = out or sys.stdout
    columns = build_columns(specs)
    # Widen the GPU column only when a longer label ("MIG 0.1") is actually present.
    gpu_width = max([5] + [len(g.label) for g in gpus.values()])
    widths = [max(6, len(header)) for _key, header, _dec in columns]

    def cells(values: Dict[str, Optional[float]]) -> List[str]:
        return [format_number(values.get(key), dec) for key, _h, dec in columns]

    if options.csv:
        writer = csv.writer(out, lineterminator="\n")
        if options.header:
            for label, value in context:
                writer.writerow([label, value])
            writer.writerow([h for h, _w in LIVE_ID_COLUMNS] + ["GPU"]
                            + [header for _k, header, _d in columns])
        for job, gpu in _live_rows(jobs, gpus):
            values = metrics.get(gpu.jobid, {}).get(gpu.uuid, {}) if gpu else {}
            writer.writerow(
                [job["jobid"], job["user"], (gpu.host if gpu else job["node"]), job["name"],
                 (gpu.csv_id if gpu else "")]
                + (cells(values) if gpu else ["" for _c in columns]))
        return

    if options.header:
        for label, value in context:
            print(fmt_context(label, value), file=out)

    if not jobs:
        print("  (no running jobs in this selection)", file=out)
        return

    header_line = ("%s %s  %s" % (
        " ".join("%-*s" % (w, h) for h, w in LIVE_ID_COLUMNS),
        "%*s" % (gpu_width, "GPU"),
        "  ".join("%*s" % (w, h) for (_k, h, _d), w in zip(columns, widths))))
    if options.header:
        print(header_line, file=out)
        print("-" * len(header_line), file=out)

    for job, gpu in _live_rows(jobs, gpus):
        identity = " ".join("%-*s" % (w, str(v)[:w]) for v, w in zip(
            (job["jobid"], job["user"], gpu.host if gpu else job["node"], job["name"]),
            (w for _h, w in LIVE_ID_COLUMNS)))
        if gpu is None:
            print("%s %s  %s" % (identity, "%*s" % (gpu_width, "-"), "[no GPU data]"), file=out)
            continue
        values = metrics.get(gpu.jobid, {}).get(gpu.uuid, {})
        print("%s %s  %s" % (identity, "%*s" % (gpu_width, gpu.label),
                             "  ".join("%*s" % (w, c) for c, w in zip(cells(values), widths))),
              file=out)


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


# The non-metric columns of the live table. Described separately from the metrics
# because they identify the row rather than measure it -- and because the GPU
# column's MIG notation is easy to misread as a decimal.
LIVE_ROW_DESCRIPTIONS: List[Tuple[str, str]] = [
    ("JOBID", "squeue's display ID (%i), so array elements keep their 12345_6 notation. "
              "Prometheus keys GPU data on the raw per-element ID (%A) instead, which the view "
              "resolves for you."),
    ("USER", "the job's owner."),
    ("NODE", "the host of THIS row's GPU, since there is one row per GPU. It is the job's whole "
             "nodelist only when no GPU data was found."),
    ("NAME", "the job name."),
    ("GPU", "\"GPU n\" is Slurm's GPU number (NVML's minor_number). \"MIG n\" / \"MIG n.i\" is a "
            "MIG instance on card n: sibling slices share their parent's minor number, so they "
            "are enumerated by UUID. DCGM columns are always \"-\" on a MIG row (see below)."),
]

LIVE_DESCRIBE_FOOTER = """\
A "-" cell means Prometheus held no sample for that GPU and metric. Every
DCGM_FI_* column reads "-" on a MIG row: NVML identifies an instance by a
MIG-... UUID where DCGM reports the physical GPU-... UUID, and nothing in the
metrics maps one to the other, so attributing the whole card's DCGM values to
a single slice would be wrong.

An instantaneous reading will not agree with jobstats on a bursty job: one that
alternates compute with gaps is genuinely bimodal, and a single scrape can read
GPU% 0 on a GPU averaging ~88%. Use --avg for a jobstats-comparable number, or
--ts to see the phases themselves."""


def describe_live(specs: List[MetricSpec], average: bool = False, n_all: int = 0,
                  out=None) -> None:
    """Plain-English reference for the columns the live view would show.

    Mirrors :func:`describe_dcgm` -- header, source metric, how the window is
    collapsed, then wrapped prose -- but reflects the ``--gpu``/``--all`` selection
    actually in force, so the reference always matches what was printed.
    """
    out = out or sys.stdout
    columns = build_columns(specs)
    source = {spec.header: (spec.metric, _REDUCER_NAME.get(spec.reducer, spec.reducer))
              for spec in specs if spec.show}
    for derived in DERIVED_COLUMNS:
        # Recomputed from already-reduced inputs, so it has no reducer of its own.
        source[derived.header] = (derived.source, "-")

    print("jobscope live columns. One row per GPU. GPU%%/MEM_GB/MEM%% come from the NVML\n"
          "exporter (nvidia_gpu_*), every other metric from dcgm-exporter (DCGM_FI_*).\n"
          "Showing %d of %d columns (%s)."
          % (len(columns), n_all or len(columns),
             "all" if not n_all or len(columns) >= n_all else "--all describes the rest"),
          file=out)
    print(file=out)
    if average:
        print("Values are folded over each job's own runtime by [fold] below --\n"
              "utilization averaged, memory peaked, exactly as jobstats does.", file=out)
    else:
        print("Values are the newest single scrape. [fold] is how --avg would instead\n"
              "collapse each metric over the job's own runtime -- utilization averaged,\n"
              "memory peaked, exactly as jobstats does.", file=out)
    print(file=out)

    for header, text in LIVE_ROW_DESCRIPTIONS:
        body = textwrap.wrap(text, width=74, initial_indent=" " * 15,
                             subsequent_indent=" " * 15)
        body[0] = "  %-12s %s" % (header, body[0].lstrip())
        print("\n".join(body), file=out)
    print(file=out)

    for _key, header, _dec in columns:
        metric, fold = source.get(header, ("(unknown)", "-"))
        print("  %-12s %-38s [fold: %s]" % (header, metric, fold), file=out)
        text = LIVE_DESCRIPTIONS.get(header) or DESCRIPTIONS.get(header, "(no description)")
        for wrapped in textwrap.wrap(text, width=74):
            print("      " + wrapped, file=out)
        print(file=out)

    print(LIVE_DESCRIBE_FOOTER, file=out)


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
