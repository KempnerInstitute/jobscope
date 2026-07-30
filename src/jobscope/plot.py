"""Render jobscope --csv output as terminal graphs.

Reads the CSV that the summary / detail / dcgm views emit and draws it as
utilization bar gauges, a distribution histogram, a colored heatmap table, or a
time-series line chart. The view is auto-detected from the CSV columns; override
with --kind. plotext and rich are imported lazily so the other subcommands never
pay for them.
"""

import csv
import os
import shutil
import sys

from . import config
from .errors import JobscopeError

ID_COLS = {"JOBID", "STATE", "NAME", "NODES", "GPUS", "NODE", "GPU",
           "DUR_S", "RUNTIME", "EPOCH", "TIME"}

HEAT_MAX_ROWS = 40
PANEL_CAP = 12

# Distinct 256-color codes for the time-series chart, one per metric, so the
# plotext line and the rich-tinted per-metric stats render the exact same color.
PALETTE = [196, 46, 33, 208, 201, 51, 226, 129, 244, 39]

# Default time-series columns, in the order the tables use. DUTY% is listed only to
# keep charting utilization for CSVs written before that column was renamed to
# GPU%; the two are the same quantity, so ts_defaults() shows at most one.
TS_DEFAULT = ["GPU%", "DUTY%", "SM_ACT%", "OCC%", "TENSOR%", "DRAM%"]
TS_ALIASES = [("GPU%", "DUTY%")]


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

    Skips the leading context rows (User, Select, ...) up to the header row
    (first cell 'JOBID') and drops the trailing 'Mean' summary row. Rows are dicts
    keyed by column.
    """
    columns, rows = None, []
    for record in csv.reader(fobj):
        if not record:
            continue
        if columns is None:
            if record[0] == "JOBID":
                columns = record
            continue
        if record[0] == "Mean":
            continue
        rows.append({columns[i]: (record[i] if i < len(record) else "")
                     for i in range(len(columns))})
    if columns is None:
        raise JobscopeError(
            "no 'JOBID' header row found -- is this 'jobscope <view> --csv' output? "
            "(do not pass -n, so the header is included)")
    return columns, rows


def to_float(value):
    """'-'/'' -> None; otherwise float, or None if unparseable."""
    if value is None or value in ("-", ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def metric_cols(columns):
    """Plottable metric columns, in CSV order (everything that is not an id column)."""
    return [c for c in columns if c not in ID_COLS]


def is_pct(header):
    return header.endswith("%")


def grade(header, value, red_map, default_red):
    """Color name for a %-metric value (rich/plotext share these names)."""
    if value is None or not is_pct(header):
        return "white"
    red = red_map.get(header, default_red)
    if value < red:
        return "red"
    if value < 2 * red:
        return "yellow"
    return "green"


def detect_kind(columns):
    if "EPOCH" in columns and "TIME" in columns:
        return "line"
    if {"NODE", "GPU", "DUR_S"} <= set(columns):
        return "heat"
    return "summary"


def render_bars(columns, rows, args, Console, Text, red_map, default_red):
    """Horizontal utilization gauges: one bar per %-metric (mean over rows if >1)."""
    console = Console(no_color=args.no_color)
    mcols = metric_cols(columns)
    title = args.title or ("utilization" + (" (mean of %d jobs)" % len(rows) if len(rows) > 1 else
                           ("  %s" % rows[0].get("JOBID", "")) if rows else ""))
    console.print("[bold]%s[/bold]" % title) if not args.no_color else print(title)
    width = 34
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
            text.append("█" * filled, style=grade(col, value, red_map, default_red))
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


def render_heat(columns, rows, args, Console, Table, Text, red_map, default_red):
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
                cells.append(Text(raw, style="black on %s" % grade(col, value, red_map, default_red)))
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
        metrics = mcols
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

    def figure(specs, title, ylabel, height):
        """Render one chart as its own plotext figure (independent height)."""
        plt.clear_figure()
        plt.theme("clear")
        plt.plotsize(width, height)
        for label, color, samples, metric in specs:
            xs, ys = series(samples, metric)
            if ys:
                plt.plot(xs, ys, label=label, **pkw(color))
        plt.title(title)
        plt.xlabel("minutes since start")
        plt.ylabel(ylabel)
        plt.show()
        print()

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
        if n_keys > 1:
            figure([("gpu%s" % g, gcolor[g], gpus[(n, g)], metric) for n, g in keys],
                   "%s  --  %s" % (metric, base), metric, args.height or 15)
        else:
            figure([(metric, mcolor[metric], gpus[keys[0]], metric)], base, metric, args.height or 15)
    elif args.by == "gpu":
        for key in cap(keys, "GPUs"):
            figure([(m, mcolor[m], gpus[key], m) for m in metrics],
                   "%s gpu%s" % key, "value", args.height or 10)
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
    parser.add_argument("--gpu", help="plot only this GPU index (where the CSV has a GPU column)")
    parser.add_argument("--all", action="store_true",
                        help="line: draw every metric, not just the default set")
    parser.add_argument("--marker", choices=["braille", "dot", "hd", "fhd"], default="braille",
                        help="line marker (default braille = thin lines; hd = thick blocks; dot = sparse)")
    parser.add_argument("--by", choices=["metric", "gpu"], default="gpu",
                        help="line: facet by 'gpu' (default; one panel per GPU, all metrics on a shared "
                             "axis) or 'metric' (one panel per metric, own y-axis)")
    parser.add_argument("--compact", action="store_true",
                        help="line: one braille sparkline row per metric, instead of full-height panels")
    parser.add_argument("--title", help="override the chart title")
    parser.add_argument("--width", type=int, help="plot width in chars (plotext charts)")
    parser.add_argument("--height", type=int, help="plot height in chars (plotext charts)")
    parser.add_argument("--max-rows", dest="max_rows", type=int, default=HEAT_MAX_ROWS,
                        help="heatmap row cap (default %d)" % HEAT_MAX_ROWS)
    parser.add_argument("--no-color", action="store_true",
                        help="disable color (also respects $NO_COLOR)")


def run(args) -> None:
    """Execute the ``plot`` subcommand from parsed arguments."""
    if os.environ.get("NO_COLOR"):
        args.no_color = True

    path = args.file_opt or args.file
    if path:
        try:
            fobj = open(path, newline="")
        except OSError as exc:
            raise JobscopeError("cannot open %s: %s" % (path, exc))
    else:
        if sys.stdin.isatty():
            raise JobscopeError("no input -- pipe 'jobscope <view> --csv' in, or pass a CSV file.")
        fobj = sys.stdin

    thresholds = config.get_config().thresholds
    red_map = thresholds.red_map()
    default_red = thresholds.default

    columns, rows = parse_csv(fobj)
    if not rows:
        raise JobscopeError("no data rows in the CSV (empty selection?).")

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
            rows = [r for r in rows if str(r.get("GPU")) == str(args.gpu)]
            if not rows:
                raise JobscopeError("no rows for GPU %r. Available: %s"
                                    % (args.gpu, ", ".join(available)))

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
        render_bars(columns, rows, args, Console, Text, red_map, default_red)
    elif kind == "hist":
        render_hist(columns, rows, args, plt)
    elif kind == "heat":
        render_heat(columns, rows, args, Console, Table, Text, red_map, default_red)
    elif kind == "line":
        render_line(columns, rows, args, plt, Console)
