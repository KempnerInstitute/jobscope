"""Tests for the plotting helpers and a render smoke test per chart kind."""

import io

import pytest

from jobscope import plot
from jobscope.cli import build_parser
from jobscope.config import Thresholds
from jobscope.errors import JobscopeError
from jobscope.report import _ESC_RE

THRESHOLDS = Thresholds(red=10, power_w=100)

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
