"""Tests for the summary / detail / dcgm renderers and the CSV contract."""

import dataclasses
import io

from jobscope import plot, report
from jobscope.blob import GIB
from jobscope.dcgm import DEFAULT_SPECS, GPU_SUMMARY_SPECS
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
from jobscope.sacct import JobRecord, Selection

CTX = [("User", "alice"), ("Select", "x")]


def _render(func, *args):
    out = io.StringIO()
    func(*args, out=out)
    return out.getvalue()


def test_fmt_context():
    assert fmt_context("User", "alice") == "  " + "User:".ljust(11) + "alice"


def test_cols_for_views():
    """`all` is the default; --cpu and --gpu narrow it. There is no cgpu any more."""
    everything = [c.header for c in cols_for(SUMMARY_COLUMNS, "all", dcgm=True, diagnose=True)]
    assert "CPU%" in everything and "GPU%" in everything and "SM_ACT%" in everything
    assert "DIAG" in everything
    cpu = [c.header for c in cols_for(SUMMARY_COLUMNS, "cpu")]
    assert "CPU%" in cpu and "GPU%" not in cpu and "SM_ACT%" not in cpu
    gpu = [c.header for c in cols_for(SUMMARY_COLUMNS, "gpu", dcgm=True)]
    assert "GPU%" in gpu and "SM_ACT%" in gpu and "CPU%" not in gpu
    # NODE and the other identity columns appear in every view.
    assert all("NODE" in cols and "JOBID" in cols for cols in (everything, cpu, gpu))


def test_diag_is_the_last_column():
    cols = [c.header for c in cols_for(SUMMARY_COLUMNS, "all", dcgm=True, diagnose=True)]
    assert cols[-1] == "DIAG"


def test_cols_for_diag_requires_dcgm():
    without = [c.header for c in cols_for(SUMMARY_COLUMNS, "all", dcgm=False, diagnose=True)]
    assert "DIAG" not in without
    with_dcgm = [c.header for c in cols_for(SUMMARY_COLUMNS, "all", dcgm=True, diagnose=True)]
    assert "DIAG" in with_dcgm


def test_summarize_diagnose_without_dcgm_does_not_crash(gpu_record):
    options = RenderOptions(view="all", show_dcgm=False, diagnose=True, csv=True, header=True)
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
    options = RenderOptions(view="all", show_dcgm=True, csv=False, header=True)
    text = _render(summarize, ["100"], {"100": gpu_record}, {"100": (overall, {})}, CTX, options)
    assert "User:" in text
    assert "SM_ACT%" in text and "60.0" in text and "400" in text
    # NAME is no longer a column; a row is identified by JOBID/USER/STATE.
    assert "100" in text and "alice" in text and "COMPLETED" in text
    # CPU% sits beside SM_ACT% now, so one table answers "is this job CPU-bound".
    assert "CPU%" in text and "#GPU" in text


def test_summary_csv_roundtrips(gpu_record):
    options = RenderOptions(view="all", show_dcgm=False, csv=True, header=True)
    text = _render(summarize, ["100"], {"100": gpu_record}, {}, CTX, options)
    columns, rows = plot.parse_csv(io.StringIO(text))
    assert columns[0] == "JOBID"
    assert len(rows) == 1
    assert rows[0]["CPU%"] == "75" and rows[0]["GPU%"] == "70"


def test_summary_mean_row_and_csv_drop(gpu_record, cpu_record):
    records = {"100": gpu_record, "200": cpu_record}
    text = _render(summarize, ["100", "200"], records, {},
                   CTX, RenderOptions(view="all", show_dcgm=False, csv=False, header=True))
    assert "Mean:" in text
    csv_text = _render(summarize, ["100", "200"], records, {},
                       CTX, RenderOptions(view="all", show_dcgm=False, csv=True, header=True))
    _, rows = plot.parse_csv(io.StringIO(csv_text))
    assert len(rows) == 2  # the Mean row is dropped by the parser


def test_summarize_no_jobs():
    text = _render(summarize, [], {}, {}, CTX,
                   RenderOptions(view="all", show_dcgm=True, csv=False, header=True))
    assert "(no GPU jobs" in text


def test_detail_text_and_csv(gpu_record):
    options = RenderOptions(view="all", show_dcgm=False, csv=False, header=True)
    text = _render(detail, ["100"], {"100": gpu_record}, {}, CTX, options)
    assert "Job 100" in text and "node01" in text and "NODE" in text
    csv_options = RenderOptions(view="all", show_dcgm=False, csv=True, header=True)
    csv_text = _render(detail, ["100"], {"100": gpu_record}, {}, CTX, csv_options)
    columns, rows = plot.parse_csv(io.StringIO(csv_text))
    assert columns[0] == "JOBID"
    assert len(rows) == 2 and rows[0]["NODE"] == "node01"


def test_dcgm_report_is_one_row_per_job(gpu_record):
    overall = {"SM_ACT%": 60.0, "POWER_W": 400.0}
    dcgm_data = {"100": (overall, {})}
    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True)
    csv_text = _render(dcgm_report, ["100"], {"100": gpu_record}, dcgm_data, DEFAULT_SPECS, CTX,
                       options)
    columns, rows = plot.parse_csv(io.StringIO(csv_text))
    assert columns == ["JOBID", "USER", "STATE", "NODE", "CPU%", "MEM%", "#GPU", "GPU%", "GMEM%",
                       "SM_ACT%", "OCC%", "TENSOR%", "DRAM%", "POWER_W", "RUNTIME"]
    assert len(rows) == 1                       # one row per job, not per GPU
    assert rows[0]["SM_ACT%"] == "60.0"
    # Blob columns still come from the blob: cpu 75, mem 50, gpu 70, gmem 50.
    assert (rows[0]["CPU%"], rows[0]["GPU%"], rows[0]["GMEM%"]) == ("75", "70", "50")


def test_dcgm_ext_only_widens_the_profiling_block(gpu_record):
    """--ext must not become a different view: identity and blob columns are fixed."""
    from jobscope.dcgm import ALL_SPECS
    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True)

    def headers(specs):
        text = _render(dcgm_report, ["100"], {"100": gpu_record}, {"100": ({}, {})}, specs,
                       CTX, options)
        return plot.parse_csv(io.StringIO(text))[0]

    default, extended = headers(DEFAULT_SPECS), headers(ALL_SPECS)
    assert default[:9] == extended[:9]              # identity + blob unchanged
    assert extended[-1] == default[-1] == "RUNTIME"
    assert len(extended) > len(default)
    # Blob-backed metrics never appear twice, even in the extended catalog.
    assert extended.count("GPU%") == 1 and extended.count("GMEM%") == 1
    assert "GMEM_TOTAL_GB" not in extended


def test_every_per_job_view_shares_one_column_set(gpu_record):
    """summary, dcgm and live must print identical columns, by construction."""
    from jobscope.report import summary_columns

    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True)

    def header(render, *args):
        text = _render(render, *args, CTX, options)
        return [r for r in text.splitlines() if r.startswith("JOBID,")][0]

    records, data = {"100": gpu_record}, {"100": ({}, {})}
    assert header(summarize, ["100"], records, data) == \
        header(dcgm_report, ["100"], records, data, DEFAULT_SPECS)
    # live renders through the same SummaryRenderer, so it cannot diverge: the
    # columns are a pure function of the spec list both are handed.
    assert [c.header for c in summary_columns(DEFAULT_SPECS)] == \
        [c.header for c in summary_columns(GPU_SUMMARY_SPECS)]


class _TimeseriesClient:
    sampling_period = 60

    def query(self, query, at, timeout=None):
        return [{"metric": {"uuid": "U0", "host": "node01:9400", "minor_number": "0"}}]

    def query_range(self, query, start, end, step, timeout=None):
        if "DCGM_FI_PROF_SM_ACTIVE" in query:
            return [{"metric": {"UUID": "U0"}, "values": [[1000, "0.8"], [1060, "0.6"]]}]
        return []


def test_dcgm_timeseries_csv(gpu_record):
    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True)
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
    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True)
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


def test_mean_row_is_followed_by_per_column_job_counts(gpu_record, cpu_record):
    """How many jobs each mean came from -- per column, because they differ.

    A CPU-only job contributes to CPU% but has no GPU% to average, so a single
    count in the label would overstate the GPU columns.
    """
    records = {"100": gpu_record, "200": cpu_record}
    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True)
    text = _render(summarize, ["100", "200"], records,
                   {"100": ({"SM_ACT%": 60.0}, {})}, CTX, options)
    rows = {r.split(",")[0]: r.split(",") for r in text.splitlines()}
    header = rows["JOBID"]
    jobs = dict(zip(header, rows["Jobs"]))
    assert dict(zip(header, rows["Mean"]))["CPU%"]           # a mean was printed
    assert jobs["CPU%"] == "2"        # both jobs have CPU data
    assert jobs["GPU%"] == "1"        # only the GPU job has GPU data
    assert jobs["SM_ACT%"] == "1"     # ... and only it had DCGM metrics
    assert jobs["#GPU"] == "2"        # the row total: jobs actually rendered


def test_footer_rows_are_not_parsed_as_jobs(gpu_record, cpu_record):
    """Otherwise `jobscope plot` would chart the footers as two extra jobs."""
    records = {"100": gpu_record, "200": cpu_record}
    text = _render(summarize, ["100", "200"], records, {}, CTX,
                   RenderOptions(view="all", show_dcgm=False, csv=True, header=True))
    _, rows = plot.parse_csv(io.StringIO(text))
    assert len(rows) == 2
    assert {r["JOBID"] for r in rows} == {"100", "200"}


def test_a_single_row_gets_no_footers(gpu_record):
    text = _render(summarize, ["100"], {"100": gpu_record}, {}, CTX,
                   RenderOptions(view="all", show_dcgm=False, csv=False, header=True))
    assert "Mean:" not in text and "Jobs:" not in text


# --- the GPU mean, with and without GPU use ---------------------------------

def _gpu_job(jid, utils, allocated=None):
    """A record whose blob reports `utils` (minor -> duty%) on one node."""
    node = {"total_time": 100, "cpus": 2, "used_memory": 8 * GIB, "total_memory": 16 * GIB}
    if utils is not None:
        node["gpu_utilization"] = utils
        node["gpu_used_memory"] = {k: 40 * GIB for k in utils}
        node["gpu_total_memory"] = {k: 80 * GIB for k in utils}
    return JobRecord(jobid=jid, state="COMPLETED", name="j", runtime="00:10:00", nodes="1",
                     gpus=allocated if allocated is not None else len(utils or {}),
                     stats={"total_time": 100, "nodes": {"n1": node}},
                     start=1000, end=1100, duration=100, jobid_raw=jid,
                     cluster="c", user="alice")


def _mean_row(records):
    text = _render(summarize, list(records), records, {}, CTX,
                   RenderOptions(view="all", show_dcgm=False, csv=True, header=True))
    rows = {r.split(",")[0]: r.split(",") for r in text.splitlines()}
    header = rows["JOBID"]
    return dict(zip(header, rows["Mean"])), dict(zip(header, rows["Jobs"]))


def test_a_job_with_no_gpu_is_left_out_of_the_gpu_mean():
    """A CPU-only job has no GPU% to average, so it must not dilute the mean."""
    records = {"1": _gpu_job("1", None), "2": _gpu_job("2", {"0": 80.0})}
    mean, jobs = _mean_row(records)
    assert mean["GPU%"] == "80"        # not 40 -- the CPU-only job is excluded
    assert jobs["GPU%"] == "1"         # ... and the footer says so
    assert jobs["CPU%"] == "2"         # while both contributed CPU%


def test_an_idle_gpu_job_is_counted_as_zero():
    """The distinction that matters: allocated-but-unused is 0%, not absent.

    Dropping these would flatter the average by hiding exactly the jobs worth
    finding.
    """
    records = {"1": _gpu_job("1", {"0": 0.0}), "2": _gpu_job("2", {"0": 100.0})}
    mean, jobs = _mean_row(records)
    assert mean["GPU%"] == "50"        # mean(0, 100), not 100
    assert jobs["GPU%"] == "2"


def test_a_jobs_own_gpu_mean_spans_its_gpus():
    records = {"1": _gpu_job("1", {"0": 100.0, "1": 100.0, "2": 100.0, "3": 0.0})}
    text = _render(summarize, ["1"], records, {}, CTX,
                   RenderOptions(view="all", show_dcgm=False, csv=True, header=True))
    _, rows = plot.parse_csv(io.StringIO(text))
    assert rows[0]["GPU%"] == "75"     # mean(100, 100, 100, 0) over the job's own GPUs


def test_allocated_gpus_that_reported_nothing_are_not_assumed_idle():
    """Absence of samples is not evidence of 0% use.

    A job can allocate 4 GPUs and have only 2 in the blob -- MIG reports no duty
    cycle at all, and a very short job may be missed by the scrape. Averaging the
    two that reported matches what jobstats stored; inventing zeros for the others
    would read as waste that was never measured. The Jobs footer is what makes the
    shortfall visible.
    """
    records = {"1": _gpu_job("1", {"0": 100.0, "1": 100.0}, allocated=4)}
    text = _render(summarize, ["1"], records, {}, CTX,
                   RenderOptions(view="all", show_dcgm=False, csv=True, header=True))
    _, rows = plot.parse_csv(io.StringIO(text))
    assert rows[0]["GPU%"] == "100" and rows[0]["#GPU"] == "4"


def test_a_gpu_job_with_no_samples_shows_a_dash_and_is_excluded():
    records = {"1": _gpu_job("1", None, allocated=2), "2": _gpu_job("2", {"0": 60.0})}
    mean, jobs = _mean_row(records)
    assert mean["GPU%"] == "60"
    assert jobs["GPU%"] == "1"         # the sample-less job cannot contribute
    assert jobs["#GPU"] == "2"         # but it is still one of the rows rendered


def test_the_gpu_mean_is_per_job_not_per_gpu():
    """Documented, not accidental: each job counts once, whatever its GPU count.

    A 4-GPU job at 100% and a 1-GPU job at 0% average to 50, not to the
    GPU-weighted 80. The table is one row per job, so the footer matches it.
    """
    records = {"1": _gpu_job("1", {str(i): 100.0 for i in range(4)}),
               "2": _gpu_job("2", {"0": 0.0})}
    mean, jobs = _mean_row(records)
    assert mean["GPU%"] == "50"
    assert jobs["GPU%"] == "2"
