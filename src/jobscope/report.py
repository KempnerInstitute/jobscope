"""Rendering of the summary, detail, per-GPU DCGM, and time-series views.

Output layout (spacing, context lines, and above all the CSV shape) is kept
stable: the CSV emitted here is what ``jobscope plot`` parses.
"""

import csv
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .blob import blob_detail, blob_metrics
from .dcgm import (
    ALL_SPECS,
    DCGM_HEADERS,
    DEFAULT_SPECS,
    DESCRIPTIONS,
    MetricSpec,
    discover_gpus,
    format_by_header,
    format_value,
    gpu_minor_key,
)
from .diagnose import LEGEND, diagnose_dcgm
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


def summarize(jobids: List[str], records: Dict[str, JobRecord],
              dcgm_data: Dict[str, Tuple[dict, dict]], context: List[Tuple[str, str]],
              options: RenderOptions, out=None) -> None:
    """One row per job: blob metrics, optional DCGM columns, optional DIAG."""
    out = out or sys.stdout
    do_dcgm = options.show_dcgm
    if options.view == "gpu":
        jobids = [j for j in jobids if j in records and records[j].gpus]
    columns = cols_for(SUMMARY_COLUMNS, options.view, options.show_dcgm, options.diagnose)
    headers = [c.header for c in columns]
    writer = csv.writer(out, lineterminator="\n") if options.csv else None

    def line(row: dict) -> str:
        return " ".join(c.fmt.format(str(row.get(c.header, ""))) for c in columns)

    if options.header:
        if options.csv:
            for label, value in context:
                writer.writerow([label, value])
            writer.writerow(headers)
        else:
            for label, value in context:
                print(fmt_context(label, value), file=out)
            header_line = line({c.header: c.header for c in columns})
            print(header_line, file=out)
            print("-" * len(header_line), file=out)

    if not jobids:
        if options.header and not options.csv:
            print("  (no GPU jobs in this selection)", file=out)
        return

    sums = {key: [0, 0] for key in ("cpu", "mem", "gpu", "gmem")}  # [total, count]
    sums_dcgm = {header: [0.0, 0] for header in DCGM_HEADERS}
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
                    sums[key][0] += value
                    sums[key][1] += 1
        if do_dcgm:
            overall = dcgm_data.get(jid, ({}, {}))[0]
            for header in DCGM_HEADERS:
                value = overall.get(header)
                row[header] = format_by_header(header, value)
                if value is not None:
                    sums_dcgm[header][0] += value
                    sums_dcgm[header][1] += 1
            if options.diagnose:
                row["DIAG"] = diagnose_dcgm(overall, record.duration if record else None,
                                            options.min_runtime)
        if options.csv:
            writer.writerow([row[h] for h in headers])
        else:
            print(line(row), file=out)

    if len(jobids) > 1:
        def mean(key: str) -> str:
            total, count = sums[key]
            return str(round(total / count)) if count else "-"

        mean_row = {c.header: "" for c in SUMMARY_COLUMNS}
        mean_row["CPU%"], mean_row["MEM%"] = mean("cpu"), mean("mem")
        mean_row["GPU%"], mean_row["GMEM%"] = mean("gpu"), mean("gmem")
        if do_dcgm:
            for header in DCGM_HEADERS:
                total, count = sums_dcgm[header]
                mean_row[header] = format_by_header(header, total / count) if count else "-"
        if options.csv:
            mean_row["JOBID"] = "Mean"
            writer.writerow([mean_row[h] for h in headers])
        else:
            mean_row["JOBID"] = "Mean:"
            if options.header:
                print("-" * len(line({c.header: c.header for c in columns})), file=out)
            print(line(mean_row), file=out)


def detail(jobids: List[str], records: Dict[str, JobRecord],
           dcgm_data: Dict[str, Tuple[dict, dict]], context: List[Tuple[str, str]],
           options: RenderOptions, out=None) -> None:
    """Per-node / per-GPU breakdown for each job."""
    out = out or sys.stdout
    do_dcgm = options.show_dcgm
    if options.view == "gpu":
        jobids = [j for j in jobids if j in records and records[j].gpus]
    columns = cols_for(DETAIL_COLUMNS, options.view, options.show_dcgm, options.diagnose)

    def line(cells) -> str:
        return " ".join(c.fmt.format(str(cells[c.index])) for c in columns)

    def rows_for(jid: str, record: Optional[JobRecord]):
        rows = blob_detail(record.stats if record else None)
        if do_dcgm:
            per_gpu = dcgm_data.get(jid, ({}, {}))[1]
            rows = [extend_detail_row(r, per_gpu, record.duration if record else None,
                                      options.min_runtime, options.diagnose) for r in rows]
        return rows

    if options.csv:
        writer = csv.writer(out, lineterminator="\n")
        if options.header:
            for label, value in context:
                writer.writerow([label, value])
            writer.writerow(["JOBID"] + [c.header for c in columns])
        for jid in jobids:
            record = records.get(jid)
            for row in rows_for(jid, record):
                writer.writerow([jid] + [row[c.index] for c in columns])
        return

    if options.header:
        for label, value in context:
            print(fmt_context(label, value), file=out)
        print(file=out)
    if not jobids:
        print("(no GPU jobs in this selection)", file=out)
        return
    for jid in jobids:
        record = records.get(jid)
        print("Job %s  [%s]  %s" % (jid, record.state if record else "?",
                                    record.name if record else "?"), file=out)
        rows = rows_for(jid, record)
        if not rows:
            print("  (no jobstats data)\n", file=out)
            continue
        header_line = line(DETAIL_HEADER)
        print("  " + header_line, file=out)
        print("  " + "-" * len(header_line), file=out)
        for row in rows:
            print("  " + line(row), file=out)
        print(file=out)


def dcgm_report(jobids: List[str], records: Dict[str, JobRecord],
                dcgm_data: Dict[str, Tuple[dict, dict]], specs: List[MetricSpec],
                context: List[Tuple[str, str]], options: RenderOptions,
                out=None) -> None:
    """Per-GPU DCGM profiling table, one row per GPU (no averaged row, no DIAG)."""
    out = out or sys.stdout
    jobids = [j for j in jobids if j in records and records[j].gpus]
    headers = [spec.header for spec in specs]

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
                    + [format_value(spec, values.get(spec.header)) for spec in specs])
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
            print("  (no GPU samples for this job: too short, no DCGM data, or beyond retention)\n",
                  file=out)
            continue
        if options.header:
            header_line = line(header_cells)
            print("  " + header_line, file=out)
            print("  " + "-" * len(header_line), file=out)
        for node_minor in sorted(per_gpu, key=gpu_key):
            values = per_gpu[node_minor]
            print("  " + line([node_minor[0], node_minor[1]]
                              + [format_value(spec, values.get(spec.header)) for spec in specs]),
                  file=out)
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
    sampling_period = client.sampling_period
    writer = csv.writer(out, lineterminator="\n")
    if options.header:
        writer.writerow(["JOBID", "EPOCH", "TIME", "NODE", "GPU"] + [s.header for s in ts_specs])

    def cell(value: Optional[float], decimals: int) -> str:
        if value is None:
            return ""
        return ("{:.%df}" % decimals).format(value) if decimals else str(int(round(value)))

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
                rows.append((node, gpu_minor_key(minor), stamp,
                             [jid, stamp, time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp)),
                              node, minor] + [cell(cells.get(s.header), s.decimals) for s in ts_specs]))
        for _, _, _, row in sorted(rows, key=lambda x: (x[0], x[1], x[2])):
            writer.writerow(row)


def describe(diagnose_on: bool = False, out=None) -> None:
    """Print a plain-English description of each summary column."""
    out = out or sys.stdout
    print("jobscope columns. CPU/MEM/GPU/GMEM come from the sacct blob (no network);", file=out)
    print("the DCGM columns (gpu view) and DIAG (gpu view + --diagnose) come from", file=out)
    print("Prometheus. For the full per-GPU DCGM catalog, run", file=out)
    print("'jobscope describe --dcgm' (add --ext for all 28 metrics).\n", file=out)
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
    reducer_name = {"avg": "mean", "max": "peak", "delta": "delta"}
    n_default = len(DEFAULT_SPECS)
    print("DCGM GPU metrics. Each value is time-averaged over the job's [start,end]", file=out)
    print("window. Showing %d of %d metrics (%s). [reduce] = how the window is collapsed.\n"
          % (len(specs), len(ALL_SPECS),
             "all" if len(specs) > n_default else "default; --ext for the rest"), file=out)
    for spec in specs:
        print("  %-12s %-38s [reduce: %s]" % (spec.header, spec.metric,
              reducer_name.get(spec.reducer, spec.reducer)), file=out)
        for wrapped in textwrap.wrap(DESCRIPTIONS.get(spec.header, "(no description)"), width=74):
            print("      " + wrapped, file=out)
        print(file=out)
