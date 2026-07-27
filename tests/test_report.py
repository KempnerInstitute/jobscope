"""Tests for the summary / detail / dcgm renderers and the CSV contract."""

import dataclasses
import io

from jobscope import plot, report
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


def _render_stream(renderer_cls, context, options, chunks):
    out = io.StringIO()
    renderer = renderer_cls(context, options, out)
    for jobids, records, dcgm_data in chunks:
        renderer.add(jobids, records, dcgm_data)
    renderer.finish()
    return out.getvalue()


def _two_gpu_chunks(gpu_record):
    """Two single-job chunks plus the equivalent all-at-once arguments."""
    rec2 = dataclasses.replace(gpu_record, jobid="101", name="eval")
    records = {"100": gpu_record, "101": rec2}
    dcgm = {"100": ({"SM_ACT%": 60.0, "OCC%": 20.0, "TENSOR%": 5.0, "DRAM%": 10.0,
                     "POWER_W": 400.0}, {}),
            "101": ({"SM_ACT%": 30.0, "OCC%": 10.0, "TENSOR%": 2.0, "DRAM%": 5.0,
                     "POWER_W": 200.0}, {})}
    chunks = [(["100"], records, {"100": dcgm["100"]}),
              (["101"], records, {"101": dcgm["101"]})]
    return ["100", "101"], records, dcgm, chunks


def test_summary_renderer_two_adds_equals_summarize_text(gpu_record):
    jobids, records, dcgm, chunks = _two_gpu_chunks(gpu_record)
    options = RenderOptions(view="gpu", show_dcgm=True, diagnose=True, csv=False, header=True)
    single = _render(summarize, jobids, records, dcgm, CTX, options)
    streamed = _render_stream(report.SummaryRenderer, CTX, options, chunks)
    assert streamed == single
    assert streamed.count("Mean:") == 1


def test_summary_renderer_two_adds_equals_summarize_csv(gpu_record):
    jobids, records, dcgm, chunks = _two_gpu_chunks(gpu_record)
    options = RenderOptions(view="gpu", show_dcgm=True, csv=True, header=True)
    single = _render(summarize, jobids, records, dcgm, CTX, options)
    streamed = _render_stream(report.SummaryRenderer, CTX, options, chunks)
    assert streamed == single
    _, rows = plot.parse_csv(io.StringIO(streamed))
    assert len(rows) == 2  # Mean row dropped by the parser


def test_summary_renderer_single_row_no_mean(gpu_record, cpu_record):
    # The second chunk is filtered out by the gpu view, so only one row renders
    # and the Mean footer must stay suppressed.
    options = RenderOptions(view="gpu", show_dcgm=False, csv=False, header=True)
    streamed = _render_stream(report.SummaryRenderer, CTX, options,
                              [(["100"], {"100": gpu_record}, {}),
                               (["200"], {"200": cpu_record}, {})])
    assert "100" in streamed
    assert "Mean:" not in streamed


def test_summary_renderer_all_filtered_empty_message(cpu_record):
    options = RenderOptions(view="gpu", show_dcgm=False, csv=False, header=True)
    streamed = _render_stream(report.SummaryRenderer, CTX, options,
                              [(["200"], {"200": cpu_record}, {})])
    assert "(no GPU jobs" in streamed
    csv_options = RenderOptions(view="gpu", show_dcgm=False, csv=True, header=True)
    streamed_csv = _render_stream(report.SummaryRenderer, CTX, csv_options,
                                  [(["200"], {"200": cpu_record}, {})])
    assert "(no GPU jobs" not in streamed_csv


def test_detail_renderer_two_adds_equals_detail(gpu_record):
    jobids, records, dcgm, chunks = _two_gpu_chunks(gpu_record)
    for csv_mode in (False, True):
        options = RenderOptions(view="cgpu", show_dcgm=False, csv=csv_mode, header=True)
        single = _render(detail, jobids, records, {}, CTX, options)
        streamed = _render_stream(report.DetailRenderer, CTX, options,
                                  [(ids, recs, {}) for ids, recs, _ in chunks])
        assert streamed == single


def test_detail_renderer_empty_message_ignores_header_flag(cpu_record):
    # detail's text-mode empty message prints even with --noheader.
    options = RenderOptions(view="gpu", show_dcgm=False, csv=False, header=False)
    streamed = _render_stream(report.DetailRenderer, CTX, options,
                              [(["200"], {"200": cpu_record}, {})])
    assert "(no GPU jobs in this selection)" in streamed
