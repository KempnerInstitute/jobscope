"""Tests for the plotting helpers and a render smoke test per chart kind."""

import io

import pytest

from jobscope import plot
from jobscope.cli import build_parser
from jobscope.config import Thresholds
from jobscope.errors import JobscopeError

RED_MAP = Thresholds(25, 20, 25, 25, 15).red_map()

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
    assert plot.grade("GPU%", 10, RED_MAP, 15) == "red"      # < 25
    assert plot.grade("GPU%", 30, RED_MAP, 15) == "yellow"   # < 50
    assert plot.grade("GPU%", 60, RED_MAP, 15) == "green"    # >= 50
    assert plot.grade("SM_ACT%", 10, RED_MAP, 15) == "red"   # falls back to default 15
    assert plot.grade("POWER_W", 100, RED_MAP, 15) == "white"  # non-percent


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
