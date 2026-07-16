"""Tests for the summary / detail / dcgm renderers and the CSV contract."""

import io

from jobscope import plot
from jobscope.dcgm import DEFAULT_SPECS
from jobscope.report import (
    SUMMARY_COLUMNS,
    RenderOptions,
    cols_for,
    context_pairs,
    dcgm_report,
    dcgm_timeseries,
    detail,
    extend_detail_row,
    fmt_context,
    summarize,
)
from jobscope.sacct import Selection

CTX = [("User", "alice"), ("Select", "x")]


def _render(func, *args):
    out = io.StringIO()
    func(*args, out=out)
    return out.getvalue()


def test_fmt_context():
    assert fmt_context("User", "alice") == "  " + "User:".ljust(11) + "alice"


def test_cols_for_views():
    cpu = [c.header for c in cols_for(SUMMARY_COLUMNS, "cpu")]
    assert "CPU%" in cpu and "GPU%" not in cpu and "SM_ACT%" not in cpu
    gpu = [c.header for c in cols_for(SUMMARY_COLUMNS, "gpu", dcgm=True, diagnose=True)]
    assert "GPU%" in gpu and "SM_ACT%" in gpu and "DIAG" in gpu and "NODES" not in gpu
    cgpu = [c.header for c in cols_for(SUMMARY_COLUMNS, "cgpu")]
    assert "CPU%" in cgpu and "GPU%" in cgpu and "SM_ACT%" not in cgpu


def test_cols_for_diag_requires_dcgm():
    without = [c.header for c in cols_for(SUMMARY_COLUMNS, "gpu", dcgm=False, diagnose=True)]
    assert "DIAG" not in without
    with_dcgm = [c.header for c in cols_for(SUMMARY_COLUMNS, "gpu", dcgm=True, diagnose=True)]
    assert "DIAG" in with_dcgm


def test_summarize_diagnose_without_dcgm_does_not_crash(gpu_record):
    options = RenderOptions(view="gpu", show_dcgm=False, diagnose=True, csv=True, header=True)
    text = _render(summarize, ["100"], {"100": gpu_record}, {}, CTX, options)
    assert "DIAG" not in text


def test_context_pairs_explicit_ids(gpu_record):
    pairs = context_pairs(Selection(user="alice", jobids=["100"]), "1 job ID(s)", {"100": gpu_record})
    assert pairs == [("User", "alice"), ("Select", "1 job ID(s)")]


def test_context_pairs_selection():
    pairs = context_pairs(Selection(user="bob", account="kempner", partition="gpu"), "last 1 day", {})
    assert [p[0] for p in pairs] == ["User", "Account", "Partition", "Select"]


def test_extend_detail_row_short_tag():
    base = ("node01", "0", "75.0%", "8GB/16GB", "90%", "48GB/80GB", "60.0%")
    per_gpu = {("node01", "0"): {"SM_ACT%": 80.0, "POWER_W": 300.0}}
    row = extend_detail_row(base, per_gpu, duration=100, min_runtime=180, diagnose_on=True)
    assert len(row) == 7 + 5 + 1
    assert row[7] == "80.0"   # SM_ACT%
    assert row[8] == "-"      # OCC% missing
    assert row[12] == "short"  # duration < min_runtime


def test_summarize_gpu_text(gpu_record):
    overall = {"SM_ACT%": 60.0, "OCC%": 20.0, "TENSOR%": 5.0, "DRAM%": 10.0, "POWER_W": 400.0}
    options = RenderOptions(view="gpu", show_dcgm=True, csv=False, header=True)
    text = _render(summarize, ["100"], {"100": gpu_record}, {"100": (overall, {})}, CTX, options)
    assert "User:" in text
    assert "SM_ACT%" in text and "60.0" in text and "400" in text
    assert "train" in text


def test_summary_csv_roundtrips(gpu_record):
    options = RenderOptions(view="cgpu", show_dcgm=False, csv=True, header=True)
    text = _render(summarize, ["100"], {"100": gpu_record}, {}, CTX, options)
    columns, rows = plot.parse_csv(io.StringIO(text))
    assert columns[0] == "JOBID"
    assert len(rows) == 1
    assert rows[0]["CPU%"] == "75" and rows[0]["GPU%"] == "70"


def test_summary_mean_row_and_csv_drop(gpu_record, cpu_record):
    records = {"100": gpu_record, "200": cpu_record}
    text = _render(summarize, ["100", "200"], records, {},
                   CTX, RenderOptions(view="cgpu", show_dcgm=False, csv=False, header=True))
    assert "Mean:" in text
    csv_text = _render(summarize, ["100", "200"], records, {},
                       CTX, RenderOptions(view="cgpu", show_dcgm=False, csv=True, header=True))
    _, rows = plot.parse_csv(io.StringIO(csv_text))
    assert len(rows) == 2  # the Mean row is dropped by the parser


def test_summarize_no_jobs():
    text = _render(summarize, [], {}, {}, CTX,
                   RenderOptions(view="gpu", show_dcgm=True, csv=False, header=True))
    assert "(no GPU jobs" in text


def test_detail_text_and_csv(gpu_record):
    options = RenderOptions(view="cgpu", show_dcgm=False, csv=False, header=True)
    text = _render(detail, ["100"], {"100": gpu_record}, {}, CTX, options)
    assert "Job 100" in text and "node01" in text and "NODE" in text
    csv_options = RenderOptions(view="cgpu", show_dcgm=False, csv=True, header=True)
    csv_text = _render(detail, ["100"], {"100": gpu_record}, {}, CTX, csv_options)
    columns, rows = plot.parse_csv(io.StringIO(csv_text))
    assert columns[0] == "JOBID"
    assert len(rows) == 2 and rows[0]["NODE"] == "node01"


def test_dcgm_report_text_and_csv(gpu_record):
    per_gpu = {("node01", "0"): {"SM_ACT%": 80.0, "POWER_W": 300.0},
               ("node01", "1"): {"SM_ACT%": 40.0, "POWER_W": 500.0}}
    dcgm_data = {"100": ({}, per_gpu)}
    options = RenderOptions(view="gpu", show_dcgm=True, csv=False, header=True)
    text = _render(dcgm_report, ["100"], {"100": gpu_record}, dcgm_data, DEFAULT_SPECS, CTX, options)
    assert "Job 100" in text and "SM_ACT%" in text and "80.0" in text
    csv_options = RenderOptions(view="gpu", show_dcgm=True, csv=True, header=True)
    csv_text = _render(dcgm_report, ["100"], {"100": gpu_record}, dcgm_data, DEFAULT_SPECS, CTX,
                       csv_options)
    columns, rows = plot.parse_csv(io.StringIO(csv_text))
    assert columns[:6] == ["JOBID", "STATE", "NAME", "NODE", "GPU", "DUR_S"]
    assert rows[0]["NODE"] == "node01" and rows[0]["SM_ACT%"] == "80.0"


class _TimeseriesClient:
    sampling_period = 60

    def query(self, query, at, timeout=None):
        return [{"metric": {"uuid": "U0", "host": "node01:9400", "minor_number": "0"}}]

    def query_range(self, query, start, end, step, timeout=None):
        if "DCGM_FI_PROF_SM_ACTIVE" in query:
            return [{"metric": {"UUID": "U0"}, "values": [[1000, "0.8"], [1060, "0.6"]]}]
        return []


def test_dcgm_timeseries_csv(gpu_record):
    options = RenderOptions(view="gpu", show_dcgm=True, csv=True, header=True)
    text = _render(dcgm_timeseries, ["100"], {"100": gpu_record}, DEFAULT_SPECS,
                   _TimeseriesClient(), None, options)
    columns, rows = plot.parse_csv(io.StringIO(text))
    assert columns[:5] == ["JOBID", "EPOCH", "TIME", "NODE", "GPU"]
    assert [r["SM_ACT%"] for r in rows] == ["80.0", "60.0"]
    assert rows[0]["EPOCH"] == "1000"
