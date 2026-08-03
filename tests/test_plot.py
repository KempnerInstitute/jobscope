"""Tests for the plotting helpers and a render smoke test per chart kind."""

import io

import pytest

from jobscope import plot
from jobscope.cli import build_parser
from jobscope.config import Thresholds
from jobscope.errors import JobscopeError
from jobscope.report import _ESC_RE

THRESHOLDS = Thresholds()

SUMMARY_CSV = """\
User,alice
Select,x
JOBID,STATE,NODES,GPUS,CPU%,MEM%,GPU%,GMEM%,RUNTIME,NAME
100,COMPLETED,1,2,75,50,70,50,01:00:00,train
"""

HEAT_CSV = """\
User,alice
Select,x
JOBID,STATE,NAME,NODE,GPU,DUR_S,SM_ACT%,OCC%,TENSOR%,DRAM%,POWER_W
100,COMPLETED,train,node01,0,100,80.0,20.0,5.0,10.0,300
100,COMPLETED,train,node01,1,100,40.0,10.0,1.0,5.0,500
"""

LINE_CSV = """\
JOBID,EPOCH,TIME,NODE,GPU,SM_ACT%,OCC%
100,1000,2020-01-01T00:00:00,node01,0,80.0,20.0
100,1060,2020-01-01T00:01:00,node01,0,60.0,15.0
"""


def test_parse_csv_skips_context_and_mean():
    text = SUMMARY_CSV + "Mean,,,,60,40,50,30,,\n"
    columns, rows = plot.parse_csv(io.StringIO(text))
    assert columns[0] == "JOBID"
    assert len(rows) == 1
    assert rows[0]["JOBID"] == "100"


def test_parse_csv_without_header_raises():
    with pytest.raises(JobscopeError):
        plot.parse_csv(io.StringIO("no,header,here\n1,2,3\n"))


def test_to_float():
    assert plot.to_float("-") is None
    assert plot.to_float("") is None
    assert plot.to_float("x") is None
    assert plot.to_float("3.5") == 3.5


def test_metric_cols_excludes_ids():
    columns = ["JOBID", "STATE", "CPU%", "GPU%", "POWER_W"]
    assert plot.metric_cols(columns) == ["CPU%", "GPU%", "POWER_W"]


def test_is_pct():
    assert plot.is_pct("GPU%")
    assert not plot.is_pct("POWER_W")


def test_grade_thresholds():
    assert plot.grade("GPU%", 9, THRESHOLDS) == "red"        # < 10
    assert plot.grade("GPU%", 15, THRESHOLDS) == "yellow"    # < 20
    assert plot.grade("GPU%", 60, THRESHOLDS) == "green"     # >= 20
    assert plot.grade("SM_ACT%", 9, THRESHOLDS) == "red"     # one cutoff for every %
    # POWER_W is graded, in watts: below the floor a GPU counts as idle.
    assert plot.grade("POWER_W", 73, THRESHOLDS) == "red"
    assert plot.grade("POWER_W", 300, THRESHOLDS) == "green"
    assert plot.grade("RUNTIME", 5, THRESHOLDS) == "white"   # not a graded column


def test_detect_kind():
    assert plot.detect_kind(["JOBID", "EPOCH", "TIME", "SM_ACT%"]) == "line"
    assert plot.detect_kind(["JOBID", "NODE", "GPU", "DUR_S", "SM_ACT%"]) == "heat"
    assert plot.detect_kind(["JOBID", "CPU%"]) == "summary"


def test_braille_spark():
    assert plot.braille_spark([], 5) == ("", None, None)
    spark, lo, hi = plot.braille_spark([1, 2, 3, 4], 4)
    assert (lo, hi) == (1, 4)
    assert len(spark) == 4
    assert all(0x2800 <= ord(ch) <= 0x28FF for ch in spark)


def _run_plot(tmp_path, name, csv_text, capsys, extra=None):
    path = tmp_path / name
    path.write_text(csv_text)
    argv = ["plot", "-f", str(path), "--no-color"] + (extra or [])
    args = build_parser()[0].parse_args(argv)
    args.func(args)
    return capsys.readouterr().out


def test_render_bars_smoke(tmp_path, capsys):
    out = _run_plot(tmp_path, "s.csv", SUMMARY_CSV, capsys)
    assert "CPU%" in out and "GPU%" in out


@pytest.mark.parametrize("csv_text,name,expected", [
    (LINE_CSV, "l.csv", 8.0),        # a time series -> the time-slice table
    (SUMMARY_CSV, "s.csv", 5.0),     # a summary CSV -> the summary table
])
def test_the_band_table_follows_the_csv_not_the_kind_flag(tmp_path, monkeypatch,
                                                          csv_text, name, expected):
    """`--kind` is overridable and skips detect_kind entirely, so keying the table
    on it would grade an explicitly charted time series against the summary's
    edges. The columns say what the CSV is; the flag only says how to draw it.

    Both cases pass --kind bars, so the renderer is the same and the only thing
    that can differ is which of the two tables was resolved.
    """
    import dataclasses

    from jobscope import config as config_module
    config_module.set_config(dataclasses.replace(
        config_module.get_config(),
        thresholds=Thresholds(by_metric={"CPU%": {"wasteful": 5.0}}),
        timeslice_thresholds=Thresholds(by_metric={"CPU%": {"wasteful": 8.0}})))
    seen = {}
    monkeypatch.setattr(plot, "render_bars",
                        lambda cols, rows, args, C, T, thr, pal=None:
                        seen.setdefault("t", thr))
    path = tmp_path / name
    path.write_text(csv_text)
    args = build_parser()[0].parse_args(
        ["plot", "-f", str(path), "--no-color", "--kind", "bars"])
    args.func(args)
    assert seen["t"].edge("wasteful", "CPU%") == expected


def test_render_heat_smoke(tmp_path, capsys):
    out = _run_plot(tmp_path, "h.csv", HEAT_CSV, capsys)
    assert "SM_ACT%" in out and "node01" in out


def test_render_line_smoke(tmp_path, capsys):
    out = _run_plot(tmp_path, "l.csv", LINE_CSV, capsys)
    assert "mean" in out


def test_render_hist_smoke(tmp_path, capsys):
    many = SUMMARY_CSV + "".join(
        "%d,COMPLETED,1,1,%d,40,%d,30,00:10:00,j\n" % (200 + i, 50 + i, 60 + i) for i in range(5))
    out = _run_plot(tmp_path, "m.csv", many, capsys, extra=["--kind", "hist", "--metric", "GPU%"])
    assert "GPU%" in out


def test_render_line_compact(tmp_path, capsys):
    out = _run_plot(tmp_path, "lc.csv", LINE_CSV, capsys, extra=["--compact"])
    assert "job 100" in out and "mean" in out


def test_render_line_by_metric(tmp_path, capsys):
    out = _run_plot(tmp_path, "lm.csv", LINE_CSV, capsys, extra=["--by", "metric"])
    assert "mean" in out


# GPU utilization is the headline column, so it is charted by default rather than
# needing --metric. TS_DEFAULT lists both spellings; only one can be in a CSV.
_GPU_LINE_CSV = """\
JOBID,EPOCH,TIME,NODE,GPU,GPU%,SM_ACT%,OCC%
100,1000,2020-01-01T00:00:00,node01,0,95,80.0,20.0
100,1060,2020-01-01T00:01:00,node01,0,90,60.0,15.0
"""


def test_identity_columns_are_not_plotted_as_data():
    """USER/STATE/NAME identify a row; charting them as metrics is nonsense."""
    from jobscope.plot import metric_cols
    header = ["JOBID", "USER", "STATE", "NODE", "NAME", "GPU", "DUR_S", "GPU%", "SM_ACT%"]
    assert metric_cols(header) == ["GPU%", "SM_ACT%"]


def test_render_line_charts_gpu_utilization_by_default(tmp_path, capsys):
    out = _run_plot(tmp_path, "lg.csv", _GPU_LINE_CSV, capsys)
    assert "GPU%" in out and "SM_ACT%" in out


def test_render_line_charts_pre_rename_csvs(tmp_path, capsys):
    """A CSV written before DUTY% became GPU% must still chart its utilization."""
    out = _run_plot(tmp_path, "ld.csv", _GPU_LINE_CSV.replace("GPU%", "DUTY%"), capsys)
    assert "DUTY%" in out


def test_ts_default_never_charts_both_spellings(tmp_path, capsys):
    # Both columns present at once is not a real CSV, but if it happened the chart
    # should not show one quantity twice under two names.
    both = _GPU_LINE_CSV.replace("GPU%,SM_ACT%", "GPU%,DUTY%,SM_ACT%").replace(
        ",0,95,80.0", ",0,95,95,80.0").replace(",0,90,60.0", ",0,90,90,60.0")
    out = _run_plot(tmp_path, "lb.csv", both, capsys)
    stats = [line for line in out.splitlines() if "mean" in line]
    assert sum(1 for line in stats if "GPU%" in line or "DUTY%" in line) <= 1


def test_render_heat_max_rows_note(tmp_path, capsys):
    path = tmp_path / "h.csv"
    path.write_text(HEAT_CSV)
    args = build_parser()[0].parse_args(["plot", "-f", str(path), "--no-color", "--max-rows", "1"])
    args.func(args)
    assert "showing 1 of 2" in capsys.readouterr().err


def test_plot_no_input_errors(tmp_path):
    args = build_parser()[0].parse_args(["plot", "-f", str(tmp_path / "missing.csv")])
    with pytest.raises(JobscopeError):
        args.func(args)


# --- GMEM% in the default set, and GPUs as columns --------------------------

def _visible(text):
    """Line lengths with the SGR escapes stripped, the way in_columns measures.

    plotext emits resets even under --no-color, so a raw len() reads 4 wide per line
    and every width assertion below would be measuring the escapes.
    """
    return [len(_ESC_RE.sub("", ln)) for ln in text.splitlines()] or [0]


# Four GPUs on one node, so --gpu can pick columns between them.
_GRID_CSV = "JOBID,EPOCH,TIME,NODE,GPU,GPU%,GMEM%,SM_ACT%\n" + "".join(
    "100,%d,2020-01-01T00:%02d:00,node01,%s,%d,%d,%d\n" % (1000 + 60 * t, t, g, 90 + t, 50 + t, 70 + t)
    for g in "0123" for t in range(4))


def test_gpu_memory_is_charted_by_default():
    """GMEM% is a resource, next to GPU%; it used to need --all."""
    assert plot.ts_defaults(["GPU%", "GMEM%", "SM_ACT%", "OCC%"]) == \
        ["GPU%", "GMEM%", "SM_ACT%", "OCC%"]


def test_ts_defaults_still_drops_what_is_absent_and_resolves_the_alias():
    assert plot.ts_defaults(["SM_ACT%"]) == ["SM_ACT%"]
    assert plot.ts_defaults(["GPU%", "DUTY%", "GMEM%"]) == ["GPU%", "GMEM%"]


@pytest.mark.parametrize("spec,expected", [
    ("0", ["0"]),
    ("0,1,2,3", ["0", "1", "2", "3"]),
    (" 0 , 2 ", ["0", "2"]),
    ("0,,1", ["0", "1"]),
    ("0.1,0.2", ["0.1", "0.2"]),      # MIG slices are not integers
])
def test_gpu_list_parses_a_comma_list(spec, expected):
    assert plot.gpu_list(spec) == expected


def test_a_listed_gpu_becomes_a_column(tmp_path, capsys):
    """Each named GPU gets its own panel on the metric's row, side by side."""
    out = _run_plot(tmp_path, "g.csv", _GRID_CSV, capsys,
                    extra=["--by", "metric", "--gpu", "0,1,2,3", "--metric", "GMEM%",
                           "--width", "200"])
    grid = [ln for ln in out.splitlines() if "gpu0" in ln]
    assert grid, out
    # All four titles on one line means they are abreast, not stacked.
    assert all(("gpu%d" % g) in grid[0] for g in range(4))


def test_the_grid_rows_stay_aligned(tmp_path, capsys):
    """The invariant in_columns exists to hold: every line one rectangle wide."""
    out = _run_plot(tmp_path, "ga.csv", _GRID_CSV, capsys,
                    extra=["--by", "metric", "--gpu", "0,1,2,3", "--metric", "GMEM%",
                           "--width", "200"])
    body = "\n".join(ln for ln in out.splitlines() if "┤" in ln or "│" in ln)
    assert body
    assert max(_visible(body)) <= 200


def test_a_narrow_terminal_drops_columns_rather_than_overflowing(tmp_path, capsys):
    """Two readable panels beat four unreadable ones; in_columns wraps the rest."""
    out = _run_plot(tmp_path, "gn.csv", _GRID_CSV, capsys,
                    extra=["--by", "metric", "--gpu", "0,1,2,3", "--metric", "GMEM%",
                           "--width", "70"])
    # The chart area only: the stats footer below it is drawn by rich, whose Console
    # has its own width and overflows independently of anything measured here.
    charts = out.split("min / mean / max")[0]
    assert max(_visible(charts)) <= 70
    titles = [ln for ln in charts.splitlines() if "gpu0" in ln]
    assert "gpu3" not in titles[0]      # four did not fit on one row


def test_without_gpu_the_metric_view_still_overlays(tmp_path, capsys):
    """The overlay answers "did one card diverge"; naming GPUs is what asks for columns."""
    out = _run_plot(tmp_path, "go.csv", _GRID_CSV, capsys,
                    extra=["--by", "metric", "--metric", "GMEM%", "--width", "120"])
    titles = [ln for ln in out.splitlines() if "gpu0" in ln and "gpu1" in ln]
    assert not titles          # no row of per-GPU panel titles
    assert "GMEM%" in out


def test_one_listed_gpu_is_not_a_grid(tmp_path, capsys):
    out = _run_plot(tmp_path, "g1.csv", _GRID_CSV, capsys,
                    extra=["--by", "metric", "--gpu", "0", "--metric", "GMEM%",
                           "--width", "120"])
    assert "gpu1" not in out


def test_a_missing_gpu_in_the_list_names_every_one_that_is_absent(tmp_path, capsys):
    """With a list it is the typo in the middle that is hard to spot."""
    with pytest.raises(JobscopeError) as exc:
        _run_plot(tmp_path, "gm.csv", _GRID_CSV, capsys,
                  extra=["--gpu", "0,7,9"])
    assert "'7'" in str(exc.value) and "'9'" in str(exc.value)
    assert "0, 1, 2, 3" in str(exc.value)


def test_default_args_carries_every_option_the_subcommand_defines():
    """--plot_ts renders without going through the subcommand, so its namespace has to
    stay in step. Derived from add_arguments rather than written out, which is what
    this pins: adding a plot option must not leave the one-command path missing it."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    plot.add_arguments(parser)
    defined = {a.dest for a in parser._actions if a.dest != "help"}
    assert defined <= set(vars(plot.default_args()))


def test_default_args_applies_overrides():
    args = plot.default_args(kind="line", by="metric", columns=True)
    assert (args.kind, args.by, args.columns) == ("line", "metric", True)
    assert args.gpu is None and args.compact is False      # the rest stay default


def test_columns_grids_without_naming_the_gpus(tmp_path, capsys):
    """The gap --plot_ts needed filled: a grid without listing every card by hand."""
    out = _run_plot(tmp_path, "gc.csv", _GRID_CSV, capsys,
                    extra=["--by", "metric", "--columns", "--metric", "GMEM%",
                           "--width", "200"])
    titles = [ln for ln in out.splitlines() if "gpu0" in ln]
    assert titles and all(("gpu%d" % g) in titles[0] for g in range(4))


# Two nodes, two GPUs each, and a watts column -- the case --plot_ts refuses and the
# overlay exists for.
_OVERLAY_CSV = "JOBID,EPOCH,TIME,NODE,GPU,GPU%,SM_ACT%,POWER_W\n" + "".join(
    "100,%d,2020-01-01T00:%02d:00,%s,%s,%d,%d,%d\n" % (1000 + 60 * t, t, n, g, 90 + t, 50 + t, 400)
    for n in ("node01", "node02") for g in "01" for t in range(4))


def _overlay(tmp_path, capsys, width="200"):
    return _run_plot(tmp_path, "ov.csv", _OVERLAY_CSV, capsys,
                     extra=["--by", "gpu", "--columns", "--all", "--width", width])


def test_the_overlay_gives_each_node_a_row_and_each_gpu_a_panel(tmp_path, capsys):
    """The layout --plot_ts_overlay exists for. --by gpu overlays the metrics on one
    axis; --columns packs those panels into a row per node."""
    out = _overlay(tmp_path, capsys)
    lines = out.splitlines()
    rows = [i for i, ln in enumerate(lines) if ln.strip() in ("node01", "node02")]
    assert len(rows) == 2, out
    # Both GPUs titled side by side on the line after each node heading.
    for i in rows:
        assert "gpu0" in lines[i + 1] and "gpu1" in lines[i + 1]


def test_the_overlay_beats_the_multinode_branch(tmp_path, capsys):
    """Without this the multi-node branch claims the CSV and charts a single metric,
    which is the collapse the overlay exists to avoid."""
    out = _overlay(tmp_path, capsys)
    assert "multi-node view charts one metric" not in out
    assert "GPU%" in out and "SM_ACT%" in out


def test_the_overlay_drops_watts_and_says_so(tmp_path, capsys):
    """Watts cannot share an axis with percentages: a 400 W line pins the scale and
    flattens every percentage onto the floor."""
    out = _overlay(tmp_path, capsys)
    panels = out.split("node01", 1)[1].split("min / mean")[0]
    assert "GPU%" in panels and "SM_ACT%" in panels     # legended in the panel
    assert "POWER_W" not in panels                       # and omitted from it


def test_the_overlay_legends_each_panel(tmp_path, capsys):
    """The names belong where the traces are. plotext draws the legend inside the axes
    and drops it silently when it will not fit, so this also guards the height: a fixed
    one loses the legend as the metric set grows."""
    out = _overlay(tmp_path, capsys)
    body = out.split("node01", 1)[1].split("min / mean")[0]
    legended = [ln for ln in body.splitlines() if "┤" in ln and "GPU%" in ln]
    assert legended, body


def test_the_overlay_height_grows_with_the_metric_set(tmp_path, capsys):
    """Six metrics need eleven rows; the legend vanishes entirely at ten."""
    csv = _OVERLAY_CSV.replace("GPU%,SM_ACT%,POWER_W", "GPU%,SM_ACT%,TENSOR%,DRAM%,OCC%,POWER_W")
    csv = "\n".join(ln if i == 0 else ln.replace(",400", ",60,70,80,400", 1)
                     for i, ln in enumerate(csv.splitlines()) if ln) + "\n"
    out = _run_plot(tmp_path, "tall.csv", csv, capsys,
                    extra=["--by", "gpu", "--columns", "--all", "--width", "200"])
    body = out.split("node01", 1)[1].split("min / mean")[0]
    assert any("DRAM%" in ln for ln in body.splitlines() if "┤" in ln or "│" in ln), body


def test_a_narrow_overlay_wraps_rather_than_drawing_unreadable_panels(tmp_path, capsys):
    """in_columns + MIN_PANEL already decide this; the node grouping must survive it."""
    out = _overlay(tmp_path, capsys, width="40")
    assert out.count("gpu0") >= 2 and out.count("gpu1") >= 2   # still both, wrapped
    assert [ln.strip() for ln in out.splitlines()].count("node01") == 1


def test_the_chart_grades_with_the_configured_palette(tmp_path, monkeypatch):
    """One [colors] value drives both the report's escapes and the chart's styles,
    so a job is the same colour in a table and in a chart."""
    import dataclasses

    from jobscope import config as config_module
    config_module.set_config(dataclasses.replace(
        config_module.get_config(),
        palette=config_module.Palette(colors={"inefficient": "magenta",
                                              "good": "color(33)"})))
    seen = {}
    monkeypatch.setattr(plot, "render_bars",
                        lambda cols, rows, args, C, T, thr, pal=None:
                        seen.setdefault("pal", pal))
    path = tmp_path / "s.csv"
    path.write_text(SUMMARY_CSV)
    args = build_parser()[0].parse_args(
        ["plot", "-f", str(path), "--no-color", "--kind", "bars"])
    args.func(args)
    palette = seen["pal"]
    # 5% is inefficient -> the red bucket, which takes inefficient's colour.
    assert plot.grade("GPU%", 5.0, THRESHOLDS, palette) == "magenta"
    # 90% is good -> the green bucket, which takes good's.
    assert plot.grade("GPU%", 90.0, THRESHOLDS, palette) == "color(33)"
    # An ungraded column still gets an explicit style, which rich requires.
    assert plot.grade("RUNTIME", 5, THRESHOLDS, palette) == "white"


def test_a_configured_colour_is_valid_as_a_rich_background():
    """render_heat builds "black on %s", so the name has to work in that position --
    which is why the accepted set is the one rich shares with ANSI."""
    from jobscope.config import Palette, valid_colour
    for colour in Palette().colors.values():
        assert valid_colour(colour)
    assert valid_colour("color(200)") and not valid_colour("puce")


def test_ts_defaults_follow_the_plot_config(hermetic_config, tmp_path):
    """[plot] metrics chooses the series; the CSV's columns still gate them."""
    from jobscope import config as config_module
    from jobscope.config import load_config

    columns = ["JOBID", "GPU%", "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W"]
    assert plot.ts_defaults(columns) == ["GPU%", "SM_ACT%", "TENSOR%", "DRAM%"]

    path = tmp_path / "c.toml"
    path.write_text('[plot]\nmetrics = ["sm_act", "power"]\n')
    config_module.set_config(load_config(str(path)))
    assert plot.ts_defaults(columns) == ["SM_ACT%", "POWER_W"]
    # A configured metric the CSV does not carry is skipped, not charted empty.
    assert plot.ts_defaults(["JOBID", "SM_ACT%"]) == ["SM_ACT%"]


def test_plot_palette_and_caps_come_from_config(hermetic_config, tmp_path):
    from jobscope import config as config_module
    from jobscope.config import load_config

    path = tmp_path / "c.toml"
    path.write_text("[plot]\npalette = [1, 2]\nmax_rows = 7\npanels = 3\n")
    config_module.set_config(load_config(str(path)))
    assert plot.palette() == [1, 2]
    assert plot.heat_max_rows() == 7
    assert plot.panel_cap() == 3


def test_max_rows_flag_beats_the_config(hermetic_config, tmp_path, capsys):
    """--max-rows is per-run; [plot] max_rows is the site default it overrides."""
    from jobscope import config as config_module
    from jobscope.config import load_config

    path = tmp_path / "c.toml"
    path.write_text("[plot]\nmax_rows = 2\n")
    config_module.set_config(load_config(str(path)))

    csv = tmp_path / "heat.csv"
    csv.write_text("JOBID,NODE,GPU,DUR_S,GPU%\n"
                   + "".join("1,n%d,0,60,10\n" % i for i in range(5)))
    plot.run(plot.default_args(file=str(csv)))
    assert "showing 2 of 5" in capsys.readouterr().err

    plot.run(plot.default_args(file=str(csv), max_rows=4))
    assert "showing 4 of 5" in capsys.readouterr().err


def test_watts_stay_off_a_shared_axis(tmp_path, capsys):
    """`--by gpu` puts every metric on one axis, so POWER_W cannot join them.

    A 500 W line pins the scale and flattens every percentage onto the floor: the
    chart then shows nothing at all. `--by metric` gives each its own axis, which is
    what makes the wide set (and --plot_ts) legible, so watts belong there.
    """
    csv = tmp_path / "ts.csv"
    csv.write_text(
        "JOBID,USER,EPOCH,TIME,NODE,GPU,GPU%,POWER_W\n"
        + "".join("1,alice,%d,2020-01-01T00:%02d:00,n1,0,90,500\n" % (1000 + 60 * t, t)
                  for t in range(4)))

    plot.run(plot.default_args(file=str(csv), kind="line", by="gpu", all=True,
                               no_color=True, width=80, height=12))
    shared = capsys.readouterr().out
    assert "GPU%" in shared
    assert "POWER_W" not in shared            # would have owned the scale

    plot.run(plot.default_args(file=str(csv), kind="line", by="metric", all=True,
                               no_color=True, width=80, height=12))
    per_metric = capsys.readouterr().out
    assert "POWER_W" in per_metric            # its own axis, so it is welcome
