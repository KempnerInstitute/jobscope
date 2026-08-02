"""Render jobscope --csv output as terminal graphs.

Reads the CSV that the summary / detail / dcgm views emit and draws it as
utilization bar gauges, a distribution histogram, a colored heatmap table, or a
time-series line chart. The view is auto-detected from the CSV columns; override
with --kind. plotext and rich are imported lazily so the other subcommands never
pay for them.
"""

import argparse
import csv
import os
import shutil
import sys

from . import config
from .errors import JobscopeError
from .report import BAR_WIDTH, cell_value, in_columns, terminal_width

ID_COLS = {"JOBID", "USER", "STATE", "NAME", "NODES", "GPUS", "NODE", "GPU",
           "#GPU", "DUR_S", "RUNTIME", "EPOCH", "TIME", "MODEL"}

# Summary footers, keyed on the JOBID cell. Not job rows, so parse_csv skips them.
# Every footer label the summary view has ever emitted: the retired Mean rows stay
# listed so a previously-saved CSV still parses.
FOOTER_ROWS = {"Mean", "MeanPerGPU", "MeanPerGPUHour", "Jobs",
               "UsedPerGPU", "UsedPerGPUHour", "UsedPerCPU", "UsedPerCPUHour",
               "GPUhours", "GPUs", "Corehours", "Cores",
               "Worst", "WorstGPU", "WorstCPU", "WorstSM", "WorstPOWER",
               "WorstBoth", "WorstGpu-cpu", "WorstAll"}

HEAT_MAX_ROWS = 40
PANEL_CAP = 12
# Side-by-side panels for --by metric --gpu a,b,c. A panel narrower than this has no
# room for an axis and its labels, so the column count drops before the width does.
MIN_PANEL = 28
GRID_GAP = 2

# Distinct 256-color codes for the time-series chart, one per metric, so the
# plotext line and the rich-tinted per-metric stats render the exact same color.
PALETTE = [196, 46, 33, 208, 201, 51, 226, 129, 244, 39]

# Default time-series columns, in the order the tables use. DUTY% is listed only to
# keep charting utilization for CSVs written before that column was renamed to
# GPU%; the two are the same quantity, so ts_defaults() shows at most one.
#
# GMEM% sits next to GPU% because the two are *resources* -- how full the card is and
# how busy it is -- where OCC%/TENSOR%/DRAM% describe how the SMs were used, which only
# means something once the GPU is known to be busy. Memory also catches a failure none
# of them do: GPU% 96 with GMEM% 3 is under-batched, and no profiling column says so.
# GMEM_GB stays out as the same quantity without a denominator.
TS_DEFAULT = ["GPU%", "DUTY%", "GMEM%", "SM_ACT%", "OCC%", "TENSOR%", "DRAM%"]
TS_ALIASES = [("GPU%", "DUTY%")]


def gpu_list(spec) -> list:
    """``--gpu`` as a list of GPU ids, in the order given.

    Comma-separated like ``--metric`` already is, so ``--gpu 0,1,2,3`` names four and a
    bare ``--gpu 0`` still names one. Ids stay strings: a MIG slice is "0.1", not a
    number, and the CSV's own values are compared as text.
    """
    return [part.strip() for part in str(spec).split(",") if part.strip()]


def ts_defaults(columns) -> list:
    """The default series to chart for ``columns``, one per distinct quantity."""
    chosen = [m for m in TS_DEFAULT if m in columns]
    for preferred, superseded in TS_ALIASES:
        if preferred in chosen and superseded in chosen:
            chosen.remove(superseded)
    return chosen

# Braille dot bits for the 1-row --compact sparkline (cell = 2 cols x 4 rows).
_BR_DOT = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
           (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80}


def braille_spark(values, cells):
    """A one-row braille sparkline of ``values`` over ``cells`` chars.

    Scaled to the values' own min..max (top = high). Returns
    ``(sparkline, lo, hi)``; ``('', None, None)`` if empty.
    """
    values = [v for v in values if v is not None]
    if not values:
        return "", None, None
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    cells = max(1, cells)
    cols, n = cells * 2, len(values)
    out = []
    for col in range(cells):
        char = 0
        for sub in (0, 1):
            value = values[min(n - 1, (col * 2 + sub) * n // cols)]
            row = 3 - int(round((value - lo) / span * 3))  # 0 = top (high) .. 3 = bottom (low)
            char |= _BR_DOT[(sub, row)]
        out.append(chr(0x2800 + char))
    return "".join(out), lo, hi


def load_libs():
    """Import plotext + rich on demand, with an actionable message if missing."""
    try:
        import plotext as plt
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
        return plt, Console, Table, Text
    except ImportError as exc:
        raise JobscopeError(
            "plotting needs plotext and rich (missing: %s).\n"
            "Install them with:  pip install plotext rich\n"
            "(or reinstall jobscope, which depends on both)."
            % getattr(exc, "name", exc))


def parse_csv(fobj):
    """Return ``(columns, rows)`` from a jobscope CSV.

    Skips the leading context rows (User, Select, ...) up to the header row (first
    cell 'JOBID') and drops the trailing footer rows. Rows are dicts keyed by
    column.

    The footers must be dropped by name, not by position: 'Mean' holds averages and
    'Jobs' the per-column contributing counts, and charting either as if it were a
    job would invent a data point.
    """
    columns, rows = None, []
    for record in csv.reader(fobj):
        if not record:
            continue
        if columns is None:
            if record[0] == "JOBID":
                columns = record
            continue
        if record[0] in FOOTER_ROWS or record[0].startswith("Stat"):
            # "Stat<METRIC>" rows carry the per-metric efficiency summary. Matched by
            # prefix because the metric set is open-ended: --dcgm emits 18 of them.
            continue
        rows.append({columns[i]: (record[i] if i < len(record) else "")
                     for i in range(len(columns))})
    if columns is None:
        raise JobscopeError(
            "no 'JOBID' header row found; is this 'jobscope <view> --csv' output? "
            "(do not pass -n, so the header is included)")
    return columns, rows


def to_float(value):
    """'-'/'' -> None; otherwise float, or None if unparseable.

    Delegates to report.cell_value, which tolerates the trailing "%" that --per-gpu
    writes into its cells. Without that, charting a detail CSV failed with "no numeric
    values for GPU%" -- every cell in it looks like "94.2%".
    """
    return cell_value(value)


def metric_cols(columns):
    """Plottable metric columns, in CSV order (everything that is not an id column)."""
    return [c for c in columns if c not in ID_COLS]


def is_pct(header):
    return header.endswith("%")


def grade(header, value, thresholds, palette=None):
    """The rich style name for a graded value.

    Two steps, deliberately separate: ``Thresholds.grade`` says which *bucket* the
    value counts in -- the same call the report tables make, so a job is graded
    identically whether it is charted or printed -- and the palette says what that
    bucket looks like. plot keeps no copy of either rule.

    "white" rather than "" for an ungraded column, because rich wants an explicit
    style where report.tint() is happy to leave text alone.
    """
    bucket = thresholds.grade(header, value)
    if not bucket:
        return "white"
    return (palette or config.Palette()).for_bucket(bucket) or "white"


def detect_kind(columns):
    if "EPOCH" in columns and "TIME" in columns:
        return "line"
    if {"NODE", "GPU", "DUR_S"} <= set(columns):
        return "heat"
    return "summary"


def render_bars(columns, rows, args, Console, Text, thresholds, palette=None):
    """Horizontal utilization gauges: one bar per %-metric (mean over rows if >1)."""
    console = Console(no_color=args.no_color)
    mcols = metric_cols(columns)
    title = args.title or ("utilization" + (" (mean of %d jobs)" % len(rows) if len(rows) > 1 else
                           ("  %s" % rows[0].get("JOBID", "")) if rows else ""))
    console.print("[bold]%s[/bold]" % title) if not args.no_color else print(title)
    width = BAR_WIDTH
    for col in mcols:
        values = [to_float(r.get(col)) for r in rows]
        values = [v for v in values if v is not None]
        if not values:
            continue
        value = sum(values) / len(values)
        if is_pct(col):
            filled = max(0, min(width, int(round(value / 100.0 * width))))
            text = Text()
            text.append("%9s " % col)
            text.append("█" * filled, style=grade(col, value, thresholds, palette))
            text.append("░" * (width - filled), style="grey37")
            text.append(" %5.1f%%" % value)
            console.print(text)
        else:
            console.print("%9s  %g" % (col, value))


def render_hist(columns, rows, args, plt):
    """Distribution of one metric across the selection (plotext histogram)."""
    mcols = [c for c in metric_cols(columns) if is_pct(c)]
    metric = args.metric or next((m for m in ("GPU%", "CPU%") if m in mcols),
                                 (mcols[0] if mcols else None))
    if not metric:
        raise JobscopeError("no %%-metric column to histogram (try --metric)")
    if metric not in columns:
        raise JobscopeError("metric %r not in CSV columns: %s"
                            % (metric, ", ".join(metric_cols(columns))))
    values = [to_float(r.get(metric)) for r in rows]
    values = [v for v in values if v is not None]
    if not values:
        raise JobscopeError("no numeric values for %s" % metric)
    plt.clear_figure()
    plt.theme("clear")
    if args.width and args.height:
        plt.plotsize(args.width, args.height)
    plt.hist(values, bins=min(15, max(5, len(set(values)))))
    plt.xlim(0, 100)
    plt.title(args.title or ("%s distribution (%d jobs)" % (metric, len(values))))
    plt.xlabel(metric)
    plt.ylabel("jobs")
    plt.show()


def render_heat(columns, rows, args, Console, Table, Text, thresholds, palette=None):
    """Jobs (or GPUs) x metrics, cell background colored by value."""
    console = Console(no_color=args.no_color)
    mcols = metric_cols(columns)
    per_gpu = {"NODE", "GPU"} <= set(columns)
    label_hdr = "NODE:GPU" if per_gpu else "JOBID"
    capped = rows[:args.max_rows]
    base = "per-GPU metrics" if per_gpu else "per-job metrics"
    table = Table(title=args.title or (base + (" (first %d of %d)" % (args.max_rows, len(rows))
                                       if len(rows) > args.max_rows else "")),
                  header_style="bold")
    table.add_column(label_hdr, no_wrap=True)
    for col in mcols:
        table.add_column(col, justify="right")
    for row in capped:
        label = ("%s:%s" % (row.get("NODE", "?"), row.get("GPU", "?"))) if per_gpu \
            else row.get("JOBID", "?")
        cells = [label]
        for col in mcols:
            raw = row.get(col, "-")
            value = to_float(raw)
            if value is not None and is_pct(col) and not args.no_color:
                cells.append(Text(raw, style="black on %s"
                                  % grade(col, value, thresholds, palette)))
            else:
                cells.append(raw)
        table.add_row(*cells)
    console.print(table)
    if len(rows) > args.max_rows:
        print("note: showing %d of %d rows; narrow the selection or raise --max-rows"
              % (args.max_rows, len(rows)), file=sys.stderr)


def render_line(columns, rows, args, plt, Console):
    """Time-series line chart over the job's window plus a per-metric stats summary."""
    console = Console(no_color=args.no_color)
    mcols = [c for c in metric_cols(columns) if is_pct(c) or c in ("POWER_W",)]
    if args.all:
        # CPU%/MEM% lead so a combined series keeps them even when the panel cap
        # trims a wide extended-GPU-catalog selection -- they are the whole point
        # of a combined --ts and must not be the ones silently dropped.
        cpu_first = [c for c in ("CPU%", "MEM%") if c in mcols]
        metrics = cpu_first + [c for c in mcols if c not in cpu_first]
    elif args.metric:
        metrics = [m for m in args.metric.split(",") if m in columns]
        if not metrics:
            raise JobscopeError("none of --metric in CSV")
    else:
        metrics = ts_defaults(columns) or mcols[:4]

    t0 = min((to_float(r.get("EPOCH")) for r in rows if to_float(r.get("EPOCH")) is not None),
             default=None)
    gpus = {}
    for row in rows:
        epoch = to_float(row.get("EPOCH"))
        if epoch is None:
            continue
        gpus.setdefault((row.get("NODE", "?"), row.get("GPU", "?")), []).append((epoch, row))
    if not gpus or t0 is None:
        raise JobscopeError("no timestamped samples (need 'jobscope dcgm --ts --csv' output)")
    for samples in gpus.values():
        samples.sort(key=lambda x: x[0])
    keys = sorted(gpus)
    nodes = sorted({n for n, _ in keys})
    jid = rows[0].get("JOBID", "?")
    base = args.title or ("job %s  DCGM over time" % jid)
    mcolor = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(metrics)}
    gpu_ids = sorted({g for _, g in keys}, key=lambda g: int(g) if str(g).isdigit() else g)
    gcolor = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(gpu_ids)}

    def pkw(color):
        kw = {"marker": args.marker}
        if not args.no_color:
            kw["color"] = color
        return kw

    def series(samples, metric):
        xs, ys = [], []
        for epoch, row in samples:
            value = to_float(row.get(metric))
            if value is not None:
                xs.append((epoch - t0) / 60.0)
                ys.append(value)
        return (xs, ys) if ys else (None, None)

    def cap(items, what):
        if len(items) > PANEL_CAP:
            print("note: showing %d of %d %s; narrow with --metric/--node/--gpu"
                  % (PANEL_CAP, len(items), what), file=sys.stderr)
        return items[:PANEL_CAP]

    width = args.width or shutil.get_terminal_size((100, 30)).columns

    def build(specs, title, ylabel, height, size=None):
        """One chart as a list of lines, at ``size`` columns wide (default full width).

        plt.build() returns what plt.show() would print, which is what lets several
        charts be packed side by side. The size asked for is the size delivered:
        measured with the escapes stripped, plotsize(w) is exactly w visible columns.
        """
        plt.clear_figure()
        plt.theme("clear")
        plt.plotsize(size or width, height)
        for label, color, samples, metric in specs:
            xs, ys = series(samples, metric)
            if ys:
                # A single-series panel gets no legend: it would name what the title
                # already says, and plotext draws it over the top of the trace.
                kw = pkw(color) if label is None else dict(pkw(color), label=label)
                plt.plot(xs, ys, **kw)
        plt.title(title)
        plt.xlabel("minutes since start")
        plt.ylabel(ylabel)
        return plt.build().splitlines()

    def figure(specs, title, ylabel, height):
        """Render one chart full width, as its own plotext figure."""
        print("\n".join(build(specs, title, ylabel, height)))
        print()

    def grid(metric, gpu_keys, height):
        """One row of panels for `metric`, one panel per GPU, packed side by side.

        Through report.in_columns, the same packer the --per-gpu charts use: it measures
        with the escapes stripped and wraps to further rows when the terminal cannot fit
        the requested count.
        """
        available = terminal_width(sys.stdout, default=width) if args.width is None else width
        want = len(gpu_keys)
        # Never draw a panel too narrow to carry an axis; fewer columns beats unreadable
        # ones, and in_columns wraps the remainder onto the next row.
        fit = max(1, (available + GRID_GAP) // (MIN_PANEL + GRID_GAP))
        ncols = min(want, fit)
        panel = (available - GRID_GAP * (ncols - 1)) // ncols
        # No y-label either: the row heading above already names the metric, and the
        # panel is narrow enough that every column of it counts. A GPU-less series
        # (e.g. the CPU/MEM cgroup series) has no real value here -- label the node
        # instead of a bare "gpu".
        blocks = [build([(None, gcolor[g], gpus[(n, g)], metric)],
                        n if not g or g == "?" else "gpu%s" % g, "", height, size=panel)
                  for n, g in gpu_keys]
        print(metric)
        for line in in_columns(blocks, columns=ncols, gap=GRID_GAP, available=available):
            print(line)
        print()
        return ncols

    if args.compact:
        multi = len(keys) > 1

        def lab(node, gpu, metric):
            return ("%s gpu%s %-8s" % (node, gpu, metric)) if multi else ("%-8s" % metric)

        label_width = max((len(lab(n, g, m)) for n, g in keys for m in metrics), default=8)
        spark_cells = max(12, width - label_width - 30)
        print(base)
        for node, gpu in keys:
            for metric in metrics:
                _, ys = series(gpus[(node, gpu)], metric)
                if not ys:
                    continue
                spark, lo, hi = braille_spark(ys, spark_cells)
                text = "  %-*s ┤%s  (%g-%g, mean %.1f)" % (
                    label_width, lab(node, gpu, metric), spark, lo, hi, sum(ys) / len(ys))
                console.print(text, style="color(%d)" % mcolor[metric], markup=False, highlight=False)
        return

    multinode = len(nodes) > 1
    n_keys, n_metrics = len(keys), len(metrics)
    lines_are_gpus = False
    stat_metrics = metrics
    # Naming several GPUs means "show me each of these", so they become columns instead
    # of lines overlaid in one panel. Without --gpu the overlay stays: it is the view
    # that answers "did one card diverge", and 7 metrics x 4 GPUs is 28 panels, which
    # should be asked for rather than arrived at.
    as_columns = (not multinode and n_keys > 1
                  and (args.columns or len(gpu_list(args.gpu or "")) > 1))

    if multinode:
        metric = metrics[0]
        stat_metrics = [metric]
        lines_are_gpus = True
        if n_metrics > 1:
            print("note: multi-node view charts one metric (%s); use --node NODE to see all metrics "
                  "for a node" % metric, file=sys.stderr)
        for node in cap(nodes, "nodes"):
            figure([("gpu%s" % g, gcolor[g], gpus[(node, g)], metric)
                    for g in gpu_ids if (node, g) in gpus],
                   "%s  %s" % (node, metric), metric, args.height or 9)
    elif n_metrics == 1:
        metric = metrics[0]
        stat_metrics = [metric]
        lines_are_gpus = n_keys > 1
        if as_columns:
            grid(metric, cap(keys, "GPUs"), args.height or 12)
        elif n_keys > 1:
            figure([("gpu%s" % g, gcolor[g], gpus[(n, g)], metric) for n, g in keys],
                   "%s - %s" % (metric, base), metric, args.height or 15)
        else:
            figure([(metric, mcolor[metric], gpus[keys[0]], metric)], base, metric, args.height or 15)
    elif args.by == "gpu":
        for key in cap(keys, "GPUs"):
            figure([(m, mcolor[m], gpus[key], m) for m in metrics],
                   "%s gpu%s" % key, "value", args.height or 10)
    elif as_columns:
        lines_are_gpus = True
        panels = cap(keys, "GPUs")
        for metric in cap(metrics, "metrics"):
            grid(metric, panels, args.height or 10)
    else:
        lines_are_gpus = n_keys > 1
        for metric in cap(metrics, "metrics"):
            if n_keys > 1:
                figure([("gpu%s" % g, gcolor[g], gpus[(n, g)], metric) for n, g in keys],
                       metric, metric, args.height or 10)
            else:
                figure([(metric, mcolor[metric], gpus[keys[0]], metric)], metric, metric,
                       args.height or 10)

    console.rule("min / mean / max / last", style="dim", characters="-")
    for key in keys:
        tag = ("%s gpu%s  " % key) if len(keys) > 1 else ""
        for metric in stat_metrics:
            _, ys = series(gpus[key], metric)
            if not ys:
                continue
            line = "  %s%-9s min %6.1f   mean %6.1f   max %6.1f   last %6.1f" % (
                tag, metric, min(ys), sum(ys) / len(ys), max(ys), ys[-1])
            color = gcolor[key[1]] if lines_are_gpus else mcolor[metric]
            console.print(line, style="color(%d)" % color, markup=False, highlight=False)


def add_arguments(parser):
    """Register the ``plot`` subcommand's options on ``parser``."""
    parser.add_argument("file", nargs="?", help="CSV file (default: stdin)")
    parser.add_argument("-f", "--file", dest="file_opt",
                        help="CSV file (alternative to the positional arg)")
    parser.add_argument("--kind", choices=["auto", "bars", "heat", "hist", "line"], default="auto",
                        help="chart type (default: auto-detect from the CSV columns)")
    parser.add_argument("--metric", help="metric column for hist (one) or line (comma-separated)")
    parser.add_argument("--node", help="plot only this node (where the CSV has a NODE column)")
    parser.add_argument("--gpu", metavar="GPU",
                        help="plot only these GPU indices, comma-separated (where the CSV "
                             "has a GPU column). With --by metric, each one named here "
                             "becomes a column: --gpu 0,1,2,3 gives four side by side")
    parser.add_argument("--all", action="store_true",
                        help="line: draw every metric, not just the default set")
    parser.add_argument("--marker", choices=["braille", "dot", "hd", "fhd"], default="braille",
                        help="line marker (default braille = thin lines; hd = thick blocks; dot = sparse)")
    parser.add_argument("--by", choices=["metric", "gpu"], default="gpu",
                        help="line: facet by 'gpu' (default; one panel per GPU, all metrics on a shared "
                             "axis) or 'metric' (one panel per metric, own y-axis)")
    parser.add_argument("--columns", action="store_true",
                        help="line + --by metric: one panel per GPU side by side, instead of "
                             "overlaying them (what naming several in --gpu also does)")
    parser.add_argument("--compact", action="store_true",
                        help="line: one braille sparkline row per metric, instead of full-height panels")
    parser.add_argument("--title", help="override the chart title")
    parser.add_argument("--width", type=int, help="plot width in chars (plotext charts)")
    parser.add_argument("--height", type=int, help="plot height in chars (plotext charts)")
    parser.add_argument("--max-rows", dest="max_rows", type=int, default=HEAT_MAX_ROWS,
                        help="heatmap row cap (default %d)" % HEAT_MAX_ROWS)
    parser.add_argument("--no-color", action="store_true",
                        help="disable color (also respects $NO_COLOR)")


def default_args(**overrides):
    """A plot argument namespace with every default filled in.

    Derived from :func:`add_arguments` rather than written out, so a caller that
    renders without going through the subcommand -- ``--plot_ts`` -- cannot go stale
    the first time ``plot`` grows an option.
    """
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    args = parser.parse_args([])
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def run(args, fobj=None) -> None:
    """Execute the ``plot`` subcommand from parsed arguments.

    ``fobj`` renders an already-open CSV instead of a file or stdin, which is how
    ``--plot_ts`` charts the series it just emitted: everything below is then shared,
    so the one-command chart cannot drift from the piped one.
    """
    if os.environ.get("NO_COLOR"):
        args.no_color = True

    path = args.file_opt or args.file
    if fobj is not None:
        pass
    elif path:
        try:
            fobj = open(path, newline="")
        except OSError as exc:
            raise JobscopeError("cannot open %s: %s" % (path, exc))
    else:
        if sys.stdin.isatty():
            raise JobscopeError("no input: pipe 'jobscope <view> --csv' in, or pass a CSV file.")
        fobj = sys.stdin

    columns, rows = parse_csv(fobj)
    if not rows:
        raise JobscopeError("no data rows in the CSV (empty selection?).")

    # Which of the two band tables this CSV should be graded by, decided from the
    # columns rather than from `kind` below: --kind is overridable, so keying on it
    # would grade an explicitly-charted time series against the summary's edges.
    cfg = config.get_config()
    thresholds = (cfg.timeslice_thresholds if detect_kind(columns) == "line"
                  else cfg.thresholds)
    palette = cfg.palette

    if args.node is not None:
        if "NODE" not in columns:
            print("note: --node ignored (no NODE column in this CSV)", file=sys.stderr)
        else:
            available = sorted({r.get("NODE") for r in rows if r.get("NODE")})
            rows = [r for r in rows if r.get("NODE") == args.node]
            if not rows:
                raise JobscopeError("no rows for node %r. Available: %s"
                                    % (args.node, ", ".join(available)))
    if args.gpu is not None:
        if "GPU" not in columns:
            print("note: --gpu ignored (no GPU column in this CSV)", file=sys.stderr)
        else:
            available = sorted({str(r.get("GPU")) for r in rows if r.get("GPU") not in (None, "")})
            wanted = gpu_list(args.gpu)
            missing = [g for g in wanted if g not in available]
            if missing:
                # Name every one that is absent, not just the first: with a list it is
                # the typo in the middle that is hard to spot.
                raise JobscopeError("no rows for GPU %s. Available: %s"
                                    % (", ".join(repr(g) for g in missing), ", ".join(available)))
            rows = [r for r in rows if str(r.get("GPU")) in set(wanted)]

    kind = args.kind
    if kind == "auto":
        detected = detect_kind(columns)
        kind = ("bars" if len(rows) == 1 else "hist") if detected == "summary" else detected

    if kind == "line" and not ("EPOCH" in columns and "TIME" in columns):
        raise JobscopeError("--kind line needs a time series: 'jobscope dcgm --ts --csv'")
    if kind == "hist" and len(rows) == 1:
        print("note: histogram needs multiple jobs; showing bars instead", file=sys.stderr)
        kind = "bars"

    plt, Console, Table, Text = load_libs()
    if kind == "bars":
        render_bars(columns, rows, args, Console, Text, thresholds, palette)
    elif kind == "hist":
        render_hist(columns, rows, args, plt)
    elif kind == "heat":
        render_heat(columns, rows, args, Console, Table, Text, thresholds, palette)
    elif kind == "line":
        render_line(columns, rows, args, plt, Console)
