"""Tests for the summary / detail / dcgm renderers and the CSV contract."""

import dataclasses
import io
import itertools
import re

import pytest

from jobscope.rows import build_context, build_rows

from jobscope import models, plot, report
from jobscope.dcgm import catalog as gpu_catalog
from jobscope.errors import JobscopeError
from jobscope.jobstats import GIB, jobstats_metrics
from jobscope.report import (
    SUMMARY_COLUMNS,
    RenderOptions,
    cols_for,
    context_pairs,
    dcgm_report,
    dcgm_timeseries,
    detail,
    fmt_context,
    summarize,
)
from jobscope.slurm import JobRecord, Selection

CTX = [("User", "alice"), ("Select", "x")]


def _render(func, jobids, records, dcgm_data, *rest):
    """Build the rows the way select.py does, then render them.

    summarize/detail/dcgm_report take a list of models.JobRow now. The tests still
    say what they mean in scheduler terms -- these records, this dcgm_data -- so the
    one build_rows call lives here rather than in every case.
    """
    out = io.StringIO()
    func(build_rows(jobids, records, dcgm_data), *rest, out=out)
    return out.getvalue()


def _render_ts(kind, jobids, records, client, options, specs=None, step=None):
    """Collect then render a finished-job --ts view, the way select.py wires it.

    The two halves are separate calls now: `timeseries` issues the queries and the
    report formats what it yields. Kept as one helper here because every test below
    is about the CSV that comes out, not about the seam.
    """
    from jobscope import timeseries as ts

    out = io.StringIO()
    match = ts.UnitFilter(options.nodename)
    common = dict(window=options.window, step=step,
                  host_specs=options.cgroup_specs, match=match)
    if kind == "cpu":
        report.cpu_timeseries(
            ts.finished_host(jobids, records, client, None, **common), options, out=out)
    elif kind == "combined":
        report.combined_timeseries(
            ts.finished_gpu(jobids, records, specs, client, None,
                            with_host=True, **common), specs, options, out=out)
    else:
        report.dcgm_timeseries(
            ts.finished_gpu(jobids, records, specs, client, None, **common),
            specs, options, out=out)
    match.check()
    return out.getvalue()


def test_fmt_context():
    assert fmt_context("User", "alice") == "  " + "User:".ljust(11) + "alice"


def test_cols_for_views():
    """`all` is the default; --cpu and --gpu narrow it. There is no cgpu any more."""
    everything = [c.header for c in cols_for(SUMMARY_COLUMNS, "all", dcgm=True)]
    assert "CPU%" in everything and "GPU%" in everything and "SM_ACT%" in everything
    cpu = [c.header for c in cols_for(SUMMARY_COLUMNS, "cpu")]
    assert "CPU%" in cpu and "GPU%" not in cpu and "SM_ACT%" not in cpu
    gpu = [c.header for c in cols_for(SUMMARY_COLUMNS, "gpu", dcgm=True)]
    assert "GPU%" in gpu and "SM_ACT%" in gpu and "CPU%" not in gpu
    # NODE and the other identity columns appear in every view.
    assert all("NODE" in cols and "JOBID" in cols for cols in (everything, cpu, gpu))


def test_context_pairs_explicit_ids(gpu_record):
    pairs = context_pairs(build_context(Selection(user="alice", jobids=["100"]),
                                        "1 job ID(s)", {"100": gpu_record}))
    # No GPU specs collected, so no provenance line -- see _source_pair.
    assert pairs == [("User", "alice"), ("Select", "1 job ID(s)")]


def test_the_window_line_names_the_dates_behind_a_day_count():
    """"last 1 day" does not say which day, and a -D window moves with the clock."""
    from jobscope.select import FINISHED, Request, sacct_selection
    selection = sacct_selection(Request(mode=FINISHED, days=1, user="alice"))
    pairs = dict(context_pairs(build_context(selection, "last 1 day, completed", {})))
    assert pairs["Select"] == "last 1 day, completed"
    window = pairs["Window"]
    assert " .. " in window and "now" not in window
    # Real dates, to the minute, in the order queried.
    start, end = window.split(" .. ")
    assert start < end
    assert len(start) == len("2026-07-30 11:25")


def test_a_bare_lastn_also_reports_its_window():
    from jobscope.select import FINISHED, Request, sacct_selection
    selection = sacct_selection(Request(mode=FINISHED, lastn=3, user="alice"))
    pairs = dict(context_pairs(build_context(selection, "last 3 jobs, completed", {})))
    assert " .. " in pairs["Window"]


def test_an_explicit_window_is_not_repeated():
    """The Select line already is the window there; a second copy says nothing."""
    from jobscope.select import FINISHED, Request, sacct_selection
    selection = sacct_selection(Request(mode=FINISHED, starttime="2026-07-15",
                                        endtime="2026-07-20", user="alice"))
    pairs = dict(context_pairs(build_context(selection, "2026-07-15 00:00 .. 2026-07-20 00:00", {})))
    assert "Window" not in pairs


def test_explicit_job_ids_have_no_window():
    from jobscope.slurm import Selection
    pairs = dict(context_pairs(build_context(Selection(user="alice", jobids=["1"]), "1 job ID(s)", {})))
    assert "Window" not in pairs


def test_context_pairs_selection():
    pairs = context_pairs(build_context(
        Selection(user="bob", account="kempner", partition="gpu"), "last 1 day", {}))
    assert [p[0] for p in pairs] == ["User", "Account", "Partition", "Select"]


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


def test_summary_footer_row_and_csv_drop(gpu_record, cpu_record):
    records = {"100": gpu_record, "200": cpu_record}
    text = _render(summarize, ["100", "200"], records, {},
                   CTX, RenderOptions(view="all", show_dcgm=False, csv=False, header=True))
    assert "Used/GPU:" in text
    csv_text = _render(summarize, ["100", "200"], records, {},
                       CTX, RenderOptions(view="all", show_dcgm=False, csv=True, header=True))
    _, rows = plot.parse_csv(io.StringIO(csv_text))
    assert len(rows) == 2  # every footer row is dropped by the parser


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
    csv_text = _render(dcgm_report, ["100"], {"100": gpu_record}, dcgm_data, gpu_catalog().default_specs, CTX,
                       options)
    columns, rows = plot.parse_csv(io.StringIO(csv_text))
    assert columns == ["JOBID", "USER", "STATE", "NODE", "CPU%", "MEM%", "#GPU", "GPU%", "GMEM%",
                       "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W", "RUNTIME"]
    assert len(rows) == 1                       # one row per job, not per GPU
    assert rows[0]["SM_ACT%"] == "60.0"
    # Those columns still come from the jobstats summary: cpu 75, mem 50, gpu 70, gmem 50.
    assert (rows[0]["CPU%"], rows[0]["GPU%"], rows[0]["GMEM%"]) == ("75", "70", "50")


def test_dcgm_ext_only_widens_the_profiling_block(gpu_record):
    """--ext must not become a different view: identity and jobstats columns are fixed."""
    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True)

    def headers(specs):
        text = _render(dcgm_report, ["100"], {"100": gpu_record}, {"100": ({}, {})}, specs,
                       CTX, options)
        return plot.parse_csv(io.StringIO(text))[0]

    default, extended = headers(gpu_catalog().default_specs), headers(gpu_catalog().all_specs)
    assert default[:9] == extended[:9]              # identity + jobstats unchanged
    assert extended[-1] == default[-1] == "RUNTIME"
    assert len(extended) > len(default)
    # jobstats-backed metrics never appear twice, even in the extended catalog.
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
        header(dcgm_report, ["100"], records, data, gpu_catalog().default_specs)
    # live renders through the same SummaryRenderer, so it cannot diverge: the
    # columns are a pure function of the spec list both are handed.
    assert [c.header for c in summary_columns(gpu_catalog().default_specs)] == \
        [c.header for c in summary_columns(gpu_catalog().gpu_summary_specs)]


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
    text = _render_ts("dcgm", ["100"], {"100": gpu_record}, _TimeseriesClient(),
                      options, specs=gpu_catalog().default_specs)
    columns, rows = plot.parse_csv(io.StringIO(text))
    assert columns[:6] == ["JOBID", "USER", "EPOCH", "TIME", "NODE", "GPU"]
    assert [r["SM_ACT%"] for r in rows] == ["80.0", "60.0"]
    assert rows[0]["EPOCH"] == "1000"


class _CpuTimeseriesClient:
    sampling_period = 60

    def query_range(self, query, start, end, step, timeout=None):
        if "cgroup_cpu_total_seconds" in query:
            return [{"metric": {"host": "node02:9100"}, "values": [[1000, "0.5"], [1060, "1.0"]]}]
        if "cgroup_memory_rss_bytes" in query:
            return [{"metric": {"host": "node02:9100"},
                     "values": [[1000, str(2 * GIB)], [1060, str(4 * GIB)]]}]
        return []


def test_cpu_timeseries_csv(cpu_record):
    """CPU%/MEM% over time, using the job's own stored summary for the divisors.

    cpu_record (conftest.py) is a CPU-only job on node02, cpus=1, total_memory=8GiB:
    0.5/1.0 cores -> 50%/100% CPU, 2/4 GiB RSS out of 8 GiB -> 25%/50% MEM.
    """
    options = RenderOptions(view="cpu", header=True, csv=True)
    text = _render_ts("cpu", ["200"], {"200": cpu_record}, _CpuTimeseriesClient(),
                      options)
    columns, rows = plot.parse_csv(io.StringIO(text))
    assert columns == ["JOBID", "USER", "EPOCH", "TIME", "NODE", "GPU", "MODEL", "CPU%", "MEM%"]
    assert rows[0]["NODE"] == "node02" and rows[0]["GPU"] == "" and rows[0]["MODEL"] == ""
    assert [r["CPU%"] for r in rows] == ["50", "100"]
    assert [r["MEM%"] for r in rows] == ["25", "50"]


def test_cpu_timeseries_falls_back_to_prometheus_for_a_still_running_job():
    """An explicit -j ID can select a job whose summary is empty because it has not
    finished yet -- the divisors then come from Prometheus, like the live view."""
    class _Client(_CpuTimeseriesClient):
        def query(self, query, at, timeout=None):
            return [{"metric": {"host": "node02:9100"}, "value": [at, "1"]}] if "cpus" in query \
                else [{"metric": {"host": "node02:9100"}, "value": [at, str(8 * GIB)]}]

    record = JobRecord(
        jobid="300", state="RUNNING", name="live", runtime="00:30:00", nodes="1", gpus=0,
        stats={}, start=2000, end=2100, duration=100, jobid_raw="300", cluster="odyssey",
        user="carol")
    options = RenderOptions(view="cpu", header=True, csv=True)
    text = _render_ts("cpu", ["300"], {"300": record}, _Client(), options)
    columns, rows = plot.parse_csv(io.StringIO(text))
    assert rows and [r["CPU%"] for r in rows] == ["50", "100"]


def test_cpu_timeseries_warns_and_skips_a_job_with_no_cpu_records():
    import sys as _sys
    record = JobRecord(jobid="400", state="COMPLETED", name="old", runtime="00:01:00",
                       nodes="1", gpus=0, stats={}, start=1, end=2, duration=1,
                       jobid_raw="400", cluster="odyssey", user="dave")
    options = RenderOptions(view="cpu", header=True, csv=True)
    err = io.StringIO()
    old_stderr, _sys.stderr = _sys.stderr, err
    try:
        text = _render_ts("cpu", ["400"], {"400": record}, _CpuTimeseriesClient(),
                          options)
    finally:
        _sys.stderr = old_stderr
    assert text == ""
    assert "no CPU/memory records" in err.getvalue()


class _CombinedTimeseriesClient(_TimeseriesClient):
    """DCGM queries served like _TimeseriesClient; cgroup queries added on top."""

    def query_range(self, query, start, end, step, timeout=None):
        if "cgroup_cpu_total_seconds" in query:
            return [{"metric": {"host": "node01:9100"}, "values": [[1000, "1.0"], [1060, "1.5"]]}]
        if "cgroup_memory_rss_bytes" in query:
            return [{"metric": {"host": "node01:9100"},
                     "values": [[1000, str(4 * GIB)], [1060, str(8 * GIB)]]}]
        return super().query_range(query, start, end, step, timeout)


def test_combined_timeseries_csv(gpu_record):
    """DCGM columns plus each row's node's CPU%/MEM% merged into one series --
    the default --ts view.

    gpu_record (conftest.py) is on node01, cpus=2, total_memory=16 GiB: 1.0/1.5
    cores -> 50%/75% CPU, 4/8 GiB RSS out of 16 GiB -> 25%/50% MEM.
    """
    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True, combined=True)
    text = _render_ts("combined", ["100"], {"100": gpu_record},
                      _CombinedTimeseriesClient(), options, specs=gpu_catalog().default_specs)
    columns, rows = plot.parse_csv(io.StringIO(text))
    assert columns[:6] == ["JOBID", "USER", "EPOCH", "TIME", "NODE", "GPU"]
    assert columns[-2:] == ["CPU%", "MEM%"]
    assert [r["SM_ACT%"] for r in rows] == ["80.0", "60.0"]
    assert [r["CPU%"] for r in rows] == ["50", "75"]
    assert [r["MEM%"] for r in rows] == ["25", "50"]


def _render_stream(renderer_cls, context, options, chunks, **kw):
    out = io.StringIO()
    renderer = renderer_cls(context, options, out, **kw)
    for jobids, records, dcgm_data in chunks:
        renderer.add(build_rows(jobids, records, dcgm_data))
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
    options = RenderOptions(view="gpu", show_dcgm=True, csv=False, header=True)
    single = _render(summarize, jobids, records, dcgm, CTX, options)
    streamed = _render_stream(report.SummaryRenderer, CTX, options, chunks)
    assert streamed == single
    assert streamed.count("Used/GPU:") == 1


def test_summary_renderer_two_adds_equals_summarize_csv(gpu_record):
    jobids, records, dcgm, chunks = _two_gpu_chunks(gpu_record)
    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True)
    single = _render(summarize, jobids, records, dcgm, CTX, options)
    streamed = _render_stream(report.SummaryRenderer, CTX, options, chunks)
    assert streamed == single
    _, rows = plot.parse_csv(io.StringIO(streamed))
    assert len(rows) == 2  # footer rows dropped by the parser


def test_a_single_job_gets_the_metric_table_but_not_the_rest(gpu_record, cpu_record):
    """The table is how you see which band one job's numbers fall in.

    The rest of the block says nothing for a lone job: the pooled row is that job's
    own row repeated, a Wasteful row names it again, and every job count is 1.
    """
    # The second chunk is filtered out by the gpu view, so only one row renders.
    options = RenderOptions(view="gpu", show_dcgm=False, csv=False, header=True)
    streamed = _render_stream(report.SummaryRenderer, CTX, options,
                              [(["100"], {"100": gpu_record}, {}),
                               (["200"], {"200": cpu_record}, {})])
    assert "100" in streamed
    assert "METRIC" in streamed and "GPU%" in streamed        # the table prints
    assert "GOOD above 20%" in streamed                       # and its legend
    for label in ("Used/", "Wasteful", "Jobs:"):
        assert label not in streamed


def test_a_single_job_lands_in_exactly_one_band_per_metric():
    """Which is the point: one job, so each metric has a single 1 and two 0s."""
    records = {"1": _gpu_job("1", {"0": 90.0})}               # GPU% 90, CPU% 50
    rows = _stat_rows(records)
    assert _counts(rows["GPU%"]) == (1, 0, 0)                 # GOOD
    assert _counts(rows["MEM%"]) == (1, 0, 0)                 # 50% -> GOOD
    assert rows["GPU%"]["USED"].endswith("(90%)")             # and 90% of it used


def test_a_single_job_still_emits_its_stat_rows_in_csv():
    records = {"1": _gpu_job("1", {"0": 90.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="gpu", csv=True, header=True), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    labels = [ln.split(",")[0] for ln in out.getvalue().splitlines()]
    assert "StatGPU%" in labels
    assert not any(la.startswith(("UsedPer", "Worst", "Jobs")) for la in labels)
    # And plot still sees exactly the one job row.
    _, parsed = plot.parse_csv(io.StringIO(out.getvalue()))
    assert [r["JOBID"] for r in parsed] == ["1"]


def test_sub_tenth_amounts_are_not_printed_as_zero():
    """"0h (78%)" reads as nothing used; the two halves of the cell must agree."""
    records = {"1": _timed_job("1", 60, gpu_util=78.0)}       # 0.047 GPU-h used
    assert _stat_rows(records, view="gpu", time_weighted=True)["GPU%"]["USED"] == "<0.1h (78%)"


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
        # Same `total` on both sides: it is the selection size, not something a chunk
        # knows, and it decides whether the blocks end in per-unit charts or one aggregate.
        streamed = _render_stream(report.DetailRenderer, CTX, options,
                                  [(ids, recs, {}) for ids, recs, _ in chunks],
                                  total=len(jobids))
        assert streamed == single


def test_detail_renderer_empty_message_ignores_header_flag(cpu_record):
    # detail's text-mode empty message prints even with --noheader.
    options = RenderOptions(view="gpu", show_dcgm=False, csv=False, header=False)
    streamed = _render_stream(report.DetailRenderer, CTX, options,
                              [(["200"], {"200": cpu_record}, {})])
    assert "(no GPU jobs in this selection)" in streamed


def test_the_footer_reports_the_two_job_counts(gpu_record, cpu_record):
    """How many jobs each figure came from, as two labelled totals.

    They differ whenever the selection mixes CPU-only and GPU work, which is the
    reason to print them at all: a CPU-only job has no GPU% to average.
    """
    pooled, counts = _pooled_row({"100": gpu_record, "200": cpu_record}, show_dcgm=True,
                                 dcgm_data={"100": ({"SM_ACT%": 60.0}, {})})
    assert pooled["CPU%"]                          # a pooled figure was printed
    assert counts == {"cpu-jobs": 2, "gpu-jobs": 1, "gpus": 2}


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


# --- which jobs reach the pooled figure -------------------------------------

def _gpu_job(jid, utils, allocated=None):
    """A record whose summary reports `utils` (minor -> duty%) on one node."""
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


def _pooled_row(records, show_dcgm=False, dcgm_data=None):
    """``(pooled figure by column, {"cpu-jobs": n, ...})`` from a rendered table.

    Finds the pooled row by prefix, since its label names the resource it is per
    (UsedPerGPU / UsedPerGPUHour / UsedPerCPUHour ...).
    """
    text = _render(summarize, list(records), records, dcgm_data or {}, CTX,
                   RenderOptions(view="all", show_dcgm=show_dcgm, csv=True, header=True))
    rows = {r.split(",")[0]: r.split(",") for r in text.splitlines()}
    pooled = next(cells for label, cells in rows.items() if label.startswith("UsedPer"))
    counts = dict(cell.split("=") for cell in rows["Jobs"][1:] if "=" in cell)
    return dict(zip(rows["JOBID"], pooled)), {k: int(v) for k, v in counts.items()}


def test_a_job_with_no_gpu_is_left_out_of_the_gpu_figure():
    """A CPU-only job has no GPU% to pool, so it must not dilute the figure."""
    records = {"1": _gpu_job("1", None), "2": _gpu_job("2", {"0": 80.0})}
    pooled, counts = _pooled_row(records)
    assert pooled["GPU%"] == "80"      # not 40 -- the CPU-only job is excluded
    # ... and the footer says so, while both jobs contributed CPU%.
    assert counts == {"cpu-jobs": 2, "gpu-jobs": 1, "gpus": 1}


def test_an_idle_gpu_job_is_counted_as_zero():
    """The distinction that matters: allocated-but-unused is 0%, not absent.

    Dropping these would flatter the average by hiding exactly the jobs worth
    finding.
    """
    records = {"1": _gpu_job("1", {"0": 0.0}), "2": _gpu_job("2", {"0": 100.0})}
    pooled, counts = _pooled_row(records)
    assert pooled["GPU%"] == "50"      # (0 + 100) over 2 GPUs, not 100
    assert counts["gpu-jobs"] == 2


def test_a_jobs_own_gpu_mean_spans_its_gpus():
    records = {"1": _gpu_job("1", {"0": 100.0, "1": 100.0, "2": 100.0, "3": 0.0})}
    text = _render(summarize, ["1"], records, {}, CTX,
                   RenderOptions(view="all", show_dcgm=False, csv=True, header=True))
    _, rows = plot.parse_csv(io.StringIO(text))
    assert rows[0]["GPU%"] == "75"     # mean(100, 100, 100, 0) over the job's own GPUs


def test_allocated_gpus_that_reported_nothing_are_not_assumed_idle():
    """Absence of samples is not evidence of 0% use.

    A job can allocate 4 GPUs and have only 2 in the jobstats summary -- MIG reports no duty
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
    pooled, counts = _pooled_row(records)
    assert pooled["GPU%"] == "60"
    # The sample-less job cannot contribute, though it is still a rendered row.
    # gpus counts only the GPUs behind a measured value, so its 2 are absent too.
    assert counts == {"cpu-jobs": 2, "gpu-jobs": 1, "gpus": 1}


def test_the_pooled_figure_is_per_gpu_not_per_job():
    """The deliberate inversion: the footer describes the hardware, not the job.

    A 4-GPU job at 100% and a 1-GPU job at 0% average to 50 per job but 80 per
    GPU. There is no per-job mean any more, because on real data it is the
    misleading one: it weights a whole idle node the same as one busy card.
    """
    records = {"1": _gpu_job("1", {str(i): 100.0 for i in range(4)}),
               "2": _gpu_job("2", {"0": 0.0})}
    pooled, counts = _pooled_row(records)
    assert pooled["GPU%"] == "80"      # (100*4 + 0*1)/5, not the per-job 50
    assert counts["gpu-jobs"] == 2


# --- the pooled row ---------------------------------------------------------

def _footers(records, show_dcgm=False, dcgm_data=None, specs=None, **kw):
    """``{label: {column: cell}}`` for every footer row of a rendered table."""
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view=kw.get("view", "all"), show_dcgm=show_dcgm,
                           csv=True, header=True,
                           time_weighted=kw.get("time_weighted", False)),
        out, specs=specs)
    renderer.add(build_rows(list(records), records, dcgm_data or {}))
    renderer.finish()
    rows = {r.split(",")[0]: r.split(",") for r in out.getvalue().splitlines()}
    header = rows["JOBID"]
    return {label: dict(zip(header, cells)) for label, cells in rows.items()}


def test_the_pooled_row_is_always_printed():
    """It is the only aggregate now, so it cannot be omitted as a no-op.

    The old per-GPU row was suppressed when every job held the same number of
    GPUs, because it then repeated the per-job mean. With that mean gone,
    suppressing it would leave a multi-job table with no summary at all.
    """
    records = {"1": _gpu_job("1", {"0": 20.0}), "2": _gpu_job("2", {"0": 80.0})}
    footers = _footers(records)
    assert "Mean" not in footers                            # the per-job mean is gone
    assert footers["UsedPerGPU"]["GPU%"] == "50"            # uniform weights, still shown


def test_the_pooled_row_covers_the_host_columns_too():
    """CPU% is pooled over allocated cores, not left blank.

    Under the old GPU-count weighting these cells had to stay empty, since a GPU
    count says nothing about CPU. Weighting each column by its own resource fixes
    that, and the row is the only summary left, so it has to carry them.
    """
    records = {"1": _gpu_job("1", {"0": 0.0}),
               "2": _gpu_job("2", {"0": 100.0, "1": 100.0})}
    pooled = _footers(records)["UsedPerGPU"]
    assert pooled["GPU%"] == "67"                           # (0*1 + 100*2)/3
    assert pooled["CPU%"] == "50"                           # both jobs hold 2 cores at 50%


def test_a_job_with_no_gpu_does_not_reach_the_pooled_gpu_figure():
    records = {"1": _gpu_job("1", None),
               "2": _gpu_job("2", {"0": 50.0}),
               "3": _gpu_job("3", {"0": 100.0, "1": 100.0})}
    footers = _footers(records)
    assert footers["UsedPerGPU"]["GPU%"] == "83"            # (50*1 + 100*2)/3
    assert footers["Jobs"]["STATE"] == "gpu-jobs=2"


def test_sum_and_max_metrics_fall_back_to_the_plain_mean():
    """ENERGY_kWh sums over a job's GPUs and PWRmax_W takes the max.

    Neither divides by a GPU count, so there is no pooled form of them. They are
    reported as the plain per-job figure rather than left blank, because the row
    they sit in is now the table's only footer.
    """
    records = {"1": _gpu_job("1", {"0": 0.0}),
               "2": _gpu_job("2", {"0": 100.0, "1": 100.0})}
    dcgm = {"1": ({"SM_ACT%": 10.0, "ENERGY_kWh": 1.0, "PWRmax_W": 300.0}, {}),
            "2": ({"SM_ACT%": 90.0, "ENERGY_kWh": 8.0, "PWRmax_W": 500.0}, {})}
    pooled = _footers(records, show_dcgm=True, dcgm_data=dcgm, specs=gpu_catalog().all_specs)["UsedPerGPU"]
    assert pooled["SM_ACT%"] == "63.3"          # (10*1 + 90*2)/3, pooled over GPUs
    assert pooled["ENERGY_kWh"] == "4.500"      # (1 + 8)/2, the per-job mean
    assert pooled["PWRmax_W"] == "400"          # (300 + 500)/2, likewise


# --- the efficiency block ---------------------------------------------------

def test_the_bands_are_plain_job_counts():
    """Just how many jobs fell in each band; the shares live in the CSV.

    One idle 16-GPU job against four busy 1-GPU jobs. The resource concentration
    (that one job holds 80% of the GPUs) is what USED and the CSV carry now.
    """
    records = {"idle": _gpu_job("idle", {str(i): 0.0 for i in range(16)})}
    records.update({str(i): _gpu_job(str(i), {"0": 90.0}) for i in range(4)})
    row = _stat_rows(records)["GPU%"]
    assert _counts(row) == (4, 0, 1)     # GOOD / OK / BAD, jobs only
    # 90% of the 4 busy GPUs and nothing from the idle 16 = 3.6 of 20.
    assert row["USED"] == "3.6 (18%)"


def _stat_rows(records, **kw):
    """``{metric: [cells]}`` from the per-metric table, in table order.

    Keyed on the METRIC cell; ``list(_stat_rows(...))`` therefore gives the row
    order, which must follow the column order of the table above.
    """
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view=kw.get("view", "all"), csv=False, header=True,
                           show_dcgm=kw.get("show_dcgm", False),
                           color=kw.get("color", False),
                           time_weighted=kw.get("time_weighted", False)),
        out, specs=kw.get("specs"))
    renderer.add(build_rows(list(records), records, kw.get("dcgm_data") or {}))
    renderer.finish()
    lines = out.getvalue().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("METRIC"))
    rows = {}
    for line in lines[start + 1:]:
        if not line or line.startswith(("Wasteful", "Jobs")):
            break
        # Columns are separated by two or more spaces; a band cell contains a
        # single one ("1 (20%)/80%"), so splitting on any whitespace would break it.
        cells = re.split(r"\s{2,}", line.strip())
        # Keyed by header, not by position: the table has gained and reordered columns
        # more than once, and every positional reader broke each time even though what
        # it meant to ask for had not moved.
        rows[cells[0]] = dict(zip(report.SummaryRenderer.STAT_HEADERS[1:], cells[1:]))
    return rows


def _counts(row):
    """``(GOOD, OK, BAD)`` as ints from a name-keyed stat row."""
    return tuple(int(row[h]) for h in ("GOOD", "OK", "BAD"))


def _eff_bars(records, **kw):
    """``{metric: (filled_blocks, percent)}`` from the --plot_avgeff bars."""
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view=kw.get("view", "all"), header=kw.get("header", True),
                           csv=kw.get("csv", False),
                           plot_avgeff=kw.get("plot_avgeff", True),
                           color=kw.get("color", False),
                           thresholds=kw.get("thresholds"),
                           show_dcgm=kw.get("show_dcgm", False),
                           time_weighted=kw.get("time_weighted", False)),
        out, specs=kw.get("specs"))
    renderer.add(build_rows(list(records), records, kw.get("dcgm_data") or {}))
    renderer.finish()
    bars = {}
    for line in out.getvalue().splitlines():
        if "\u2588" in line or "\u2591" in line:
            head, _, tail = line.strip().partition("  ")
            value_text, _, bar_part = tail.strip().partition("  ")
            bars[head] = (bar_part.count("\u2588"), int(value_text.rstrip("%")))
    return bars


def _sections(records, **kw):
    """``[(number, title, [line, ...])]`` parsed back out of a rendered report."""
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view=kw.get("view", "all"), header=kw.get("header", True),
                           csv=kw.get("csv", False),
                           plot_avgeff=kw.get("plot_avgeff", True),
                           time_weighted=kw.get("time_weighted", False)),
        out, specs=kw.get("specs"))
    renderer.add(build_rows(list(records), records, kw.get("dcgm_data") or {}))
    renderer.finish()
    found, current = [], None
    for line in out.getvalue().splitlines():
        head = re.match(r"^(\d+)\. (.*)$", line)
        if head:
            current = (int(head.group(1)), head.group(2), [])
            found.append(current)
        elif current is not None and not set(line) <= {"-", ""}:
            current[2].append(line)
    return found


def test_rows_reach_the_screen_before_the_summary(capsys):
    """The property the streaming running path exists to deliver: a row is readable as soon
    as its data lands, and the summary waits for the rest. add() prints and flushes, so a
    caller that hands over batches gets a table filling in rather than a blank wait --
    guarding against anyone re-collecting eagerly and rendering once.
    """
    out = io.StringIO()
    renderer = report.SummaryRenderer(CTX, RenderOptions(view="all", header=True), out)
    renderer.add(build_rows(["1"], {"1": _gpu_job("1", {"0": 90.0})}, {}))
    after_first = out.getvalue()
    assert any(ln.startswith("1 ") or ln.startswith("1\t") or ln.split()[:1] == ["1"]
               for ln in after_first.splitlines() if ln.strip()), after_first
    assert "Summary by metric" not in after_first
    assert "\u2588" not in after_first          # nor the efficiency bars

    renderer.add(build_rows(["2"], {"2": _gpu_job("2", {"0": 10.0})}, {}))
    renderer.finish()
    whole = out.getvalue()
    assert whole.startswith(after_first), "an earlier row must not be rewritten"
    assert "Summary by metric" in whole and "\u2588" in whole


def test_each_add_flushes_so_a_pipe_sees_the_rows(monkeypatch):
    """Rows only stream if they leave the buffer: stdout to a pipe is block-buffered, so
    without the flush a reader watching `jobscope | ...` still waits for the whole run."""
    flushes = []

    class Watched(io.StringIO):
        def flush(self):
            flushes.append(len(self.getvalue()))

    out = Watched()
    renderer = report.SummaryRenderer(CTX, RenderOptions(view="all", header=True), out)
    renderer.add(build_rows(["1"], {"1": _gpu_job("1", {"0": 90.0})}, {}))
    renderer.add(build_rows(["2"], {"2": _gpu_job("2", {"0": 10.0})}, {}))
    assert len(flushes) >= 2 and flushes == sorted(flushes)


def test_the_report_is_three_numbered_sections():
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 5.0})}
    got = [(n, t) for n, t, _lines in _sections(records)]
    assert got == [(1, "Summary by metric"),
                   (2, "Average efficiency  (filled = used, grey = idle)"),
                   (3, "Problem jobs")]


def test_the_sections_hold_what_their_titles_say():
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 1.0})}
    one, two, three = _sections(records)
    assert any(ln.startswith("METRIC") for ln in one[2])
    assert any(ln.startswith("Used/") for ln in one[2])
    # The bars, then the note saying what they were averaged over -- wrapped to the
    # bars' own width, so it is one line or two depending on the reduction.
    bars, note = [], []
    for line in two[2]:
        (bars if ("\u2588" in line or "\u2591" in line) else note).append(line)
    assert bars
    assert " ".join(line.strip() for line in note).startswith(
        "bars are the USED share above --")
    assert any(ln.startswith("Wasteful GPU") for ln in three[2])
    assert any(ln.startswith("Jobs:") for ln in three[2])


def test_numbering_is_sequential_over_the_sections_actually_printed():
    """A gap where a section was would read as something having failed."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 5.0})}
    without = [(n, t) for n, t, _l in _sections(records, plot_avgeff=False)]
    assert without == [(1, "Summary by metric"), (2, "Problem jobs")]


def test_a_single_job_gets_the_first_two_sections_only():
    """No Worst rows for one job, so there is no third section to number."""
    got = [(n, t) for n, t, _l in _sections({"1": _gpu_job("1", {"0": 90.0})})]
    assert [n for n, _t in got] == [1, 2]
    assert "Problem jobs" not in [t for _n, t in got]


def test_each_rule_spans_its_own_sections_widest_visible_line():
    """Measured on visible characters: colour escapes would inflate len()."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 5.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="gpu", header=True, color=True,
                           thresholds=_thresholds()), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    lines = out.getvalue().splitlines()
    for i, line in enumerate(lines):
        if not re.match(r"^\d+\. ", line):
            continue
        rule = lines[i + 1]
        assert set(rule) == {"-"}
        body = []
        for ln in lines[i + 2:]:
            if not ln or re.match(r"^\d+\. ", ln):
                break
            body.append(len(report._ESC_RE.sub("", ln)))
        assert len(rule) == max(body + [len(line)])


def test_noheader_drops_the_section_furniture_but_keeps_the_data():
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 1.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=False), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    text = out.getvalue()
    assert "1. Summary by metric" not in text and "---" not in text
    assert "\u2588" in text and "Wasteful GPU" in text   # the data survives


def test_csv_gets_no_section_furniture():
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 5.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True, csv=True), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    text = out.getvalue()
    for furniture in ("1. ", "2. ", "3. ", "\u2588", "\u2591", "---"):
        assert furniture not in text
    _, parsed = plot.parse_csv(io.StringIO(text))
    assert [r["JOBID"] for r in parsed] == ["1", "2"]


def test_the_bars_complement_the_idle_column():
    """Chart and table are the same figure, so they cannot disagree.

    Bar percent plus IDLE percent is 100 for every metric, because both come from
    the tally's pooled utilization.
    """
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    bars = _eff_bars(records)
    rows = _stat_rows(records)
    assert set(bars) == set(rows)
    for metric, (_blocks, pct) in bars.items():
        idle = int(rows[metric]["USED"].rsplit("(", 1)[1].rstrip("%)"))
        assert pct + idle == 100, (metric, pct, idle)


def test_the_bars_follow_the_tables_metric_set():
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    assert list(_eff_bars(records)) == ["CPU%", "MEM%", "GPU%", "GMEM%"]
    assert list(_eff_bars(records, view="cpu")) == ["CPU%", "MEM%"]
    assert list(_eff_bars(records, view="gpu")) == ["GPU%", "GMEM%"]


def test_power_gets_a_table_row_but_no_bar():
    """The one metric in the table with no bar, and deliberately.

    Its "used" is time above the watt floor -- a detector reading, not a fraction of
    a resource. On GPUs idling at 119 W it fills to 100% beside SM_ACT% at 2%, which
    reads as the healthiest metric while describing the same idle GPUs.
    """
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    dcgm = {j: ({"POWER_W": 300.0, "SM_ACT%": 40.0}, {}) for j in records}
    bars = _eff_bars(records, show_dcgm=True, dcgm_data=dcgm, specs=gpu_catalog().default_specs)
    assert "SM_ACT%" in bars and "POWER_W" not in bars
    assert "POWER_W" in _power_stats(300.0)


def test_a_nonzero_bar_is_never_drawn_empty():
    """An empty bar beside a "1%" contradicts itself."""
    records = {"1": _gpu_job("1", {"0": 1.0}), "2": _gpu_job("2", {"0": 1.0})}
    blocks, pct = _eff_bars(records)["GPU%"]
    assert pct == 1 and blocks == 1


def test_the_bars_are_tinted_by_band_and_only_when_colour_is_on():
    records = {"1": _gpu_job("1", {"0": 95.0}), "2": _gpu_job("2", {"0": 95.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="gpu", header=True, plot_avgeff=True, color=True,
                           thresholds=_thresholds()), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    bar = [ln for ln in out.getvalue().splitlines() if "\u2588" in ln][0]
    assert report._SGR["green"] in bar          # 95% is green at a cutoff of 10
    # Colour off: the block characters carry the shape on their own.
    assert "\033" not in "".join(_eff_bars(records, view="gpu"))


def test_no_bars_in_csv_or_under_no_plot():
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    assert _eff_bars(records, csv=True) == {}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True, plot_avgeff=False), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    assert "\u2588" not in out.getvalue()

    # And they are there without asking, now that they are the default.
    shown = io.StringIO()
    default = report.SummaryRenderer(CTX, RenderOptions(view="all", header=True), shown)
    default.add(build_rows(list(records), records, {}))
    default.finish()
    assert "\u2588" in shown.getvalue()


def test_a_single_job_gets_bars_too():
    records = {"1": _gpu_job("1", {"0": 90.0})}
    assert _eff_bars(records)["GPU%"] == (31, 90)   # 90% of 34 blocks


def _finish(records, view="all", **kw):
    out = io.StringIO()
    dcgm_data = kw.pop("dcgm_data", None)
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view=view, header=kw.pop("header", True), **kw), out)
    renderer.add(build_rows(list(records), records, dcgm_data or {}))
    renderer.finish()
    return out.getvalue()


def test_single_job_efficiency_is_combined_when_cpu_and_gpu_both_present():
    """--all (the default) sees both CPU% and GPU%, so both vote."""
    records = {"1": _gpu_job("1", {"0": 90.0})}   # GPU% 90 (good), CPU%/MEM% 50 (jobstats)
    text = _finish(records, view="all")
    assert "Efficiency: good (>40%)" in text


def test_single_job_efficiency_falls_back_to_plain_gpu_for_a_gpu_view():
    """--gpu excludes CPU%/MEM% from the columns, so there is nothing to combine
    with -- classify() alone decides, unsplit ("wasteful", not "wasteful-*")."""
    records = {"1": _gpu_job("1", {"0": 0.5})}    # GPU% 0.5 -> wasteful
    text = _finish(records, view="gpu")
    assert "Efficiency: wasteful (<2%)" in text


def test_single_job_efficiency_uses_cpu_alone_for_a_cpu_view():
    records = {"1": _gpu_job("1", None)}          # no GPU data at all: CPU-only job
    text = _finish(records, view="cpu")
    assert "Efficiency: good (>40%)" in text   # CPU% 50 from the jobstats summary


def test_single_job_efficiency_is_not_skewed_by_power_w_wattage():
    """POWER_W's raw watts (e.g. 219) must never win classify()'s max() outright --
    it caps the verdict, it does not vote in it (regression: it used to leak into
    the voting dict and its wattage would dominate any percentage)."""
    records = {"1": _gpu_job("1", {"0": 31.0})}   # GPU% 31 -> "average" (20-40%)
    dcgm_data = {"1": ({"SM_ACT%": 20.0, "TENSOR%": 3.0, "DRAM%": 19.0, "POWER_W": 219.0}, {})}
    text = _finish(records, view="all", show_dcgm=True, dcgm_data=dcgm_data)
    assert "Efficiency: average (20-40%)" in text


def test_single_job_efficiency_describes_the_voting_metrics():
    records = {"1": _gpu_job("1", {"0": 90.0})}
    dcgm_data = {"1": ({"SM_ACT%": 20.0, "TENSOR%": 3.0, "DRAM%": 19.0, "POWER_W": 219.0}, {})}
    # Mentioning the POWER_W cap requires a resolvable floor -- thresholds configured.
    text = _finish(records, view="all", show_dcgm=True, dcgm_data=dcgm_data,
                  thresholds=_thresholds())
    assert "Graded by best of GPU%" in text
    assert "POWER_W lowers it below the floor" in text
    assert "CPU% can vote no higher than inefficient" in text


def test_efficiency_description_names_a_site_configured_ceiling(
        hermetic_config, tmp_path):
    """The line must describe the rule the run used, not the built-in default.

    It read config.DEFAULT_VOTE_CEILING directly, so a ceiling added through
    [eff.ceiling] capped verdicts while going unmentioned -- the one thing a
    "judged by" line exists to prevent.
    """
    from jobscope.config import load_config
    from jobscope.job_eff import classify_description

    path = tmp_path / "c.toml"
    path.write_text('[eff]\nvote = ["gpu", "sm_act", "cpu"]\n'
                    '[eff.ceiling]\nsm_act = "average"\n')
    thresholds = load_config(str(path)).thresholds
    text = classify_description(["GPU%", "SM_ACT%", "CPU%"], [], thresholds)
    assert "SM_ACT% can vote no higher than average" in text
    assert "CPU% can vote no higher than inefficient" in text
    # GPU% carries no ceiling, so it must not be named as if it did.
    assert "GPU% can vote" not in text


def test_single_job_efficiency_description_omits_cpu_for_a_gpu_view():
    records = {"1": _gpu_job("1", {"0": 90.0})}
    text = _finish(records, view="gpu")
    assert "Graded by best of GPU%" in text
    assert "CPU% splits the worst band" not in text


def test_single_job_efficiency_description_for_a_cpu_view():
    records = {"1": _gpu_job("1", None)}
    text = _finish(records, view="cpu")
    assert "Graded by best of CPU%" in text


def test_single_job_with_no_jobstats_gets_no_efficiency_line():
    record = JobRecord(jobid="1", state="COMPLETED", name="j", runtime="00:10:00",
                       nodes="1", gpus=0, stats={}, start=1000, end=1100, duration=100,
                       jobid_raw="1", cluster="c", user="alice")
    text = _finish({"1": record})
    assert "Efficiency:" not in text


def test_a_multi_job_selection_gets_no_efficiency_line():
    """The line is single-job-only -- a partition sweep keeps its existing Worst
    rows instead, so there is no ambiguity about which job it would describe."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 5.0})}
    text = _finish(records)
    assert "Efficiency:" not in text


def test_the_bar_title_obeys_noheader():
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="gpu", header=False, plot_avgeff=True), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    text = out.getvalue()
    assert "Avg efficiency" not in text and "\u2588" in text


def test_the_stat_table_carries_a_legend():
    """Two things in the table read wrongly without it.

    RED< is a threshold, not a count, and the yellow edge is implicit at twice it.
    And "green" means only "not pathological": with a cutoff of 10 a job at 21% is
    green while wasting four fifths of its cores, so a selection can be half idle
    with nearly every job green -- which looks like a contradiction until the legend
    says USED is the efficiency number and the bands only locate the waste.
    """
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(CTX, RenderOptions(view="all", header=True), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    text = out.getvalue()
    # "at or below": the boundary is inclusive -- exactly 10.0 bands inefficient,
    # which is red. The per-metric form is what prints by default now, since CPU%
    # ships with its own cutoff of 5; either way the edges are stated, not implied.
    assert "GOOD above 20%, OK above 10%, BAD otherwise" in text
    assert "POWER_W is GOOD above 100 W and BAD below" in text
    # What the RED/YELLOW/GREEN columns hold: the header alone reads as a value.
    assert "GOOD/OK/BAD sum to it" in text
    assert "USED measures it" in text and "no BAD jobs" in text
    # Directly above the header it explains, and inside the table width.
    lines = text.splitlines()
    # Whatever the legend's last line wraps to, it is the one directly above the header
    # it explains. Compared against _legend's own output rather than a literal, so a
    # reworded or re-wrapped entry does not need this test edited to keep the property.
    legend = renderer._legend(["GPU%", "CPU%"])
    header = lines.index(next(ln for ln in lines if ln.startswith("METRIC")))
    assert lines[header - 1].strip() == legend[-1].strip()
    # Measured on what is printed, not on the template: the per-metric form is
    # built at render time, so a template check would pass while output overflowed.
    assert all(len(ln) <= 132 for ln in renderer._legend(["GPU%", "CPU%"]))


_AVERAGING_OPENERS = ("Averaged over time:", "Not averaged over time:")


def _averaging_line(text):
    """The averaging sentence, unwrapped, from a rendered summary block.

    It is the section's first line now, so this reads to the next legend entry rather
    than from a label buried among them.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    start = next(i for i, ln in enumerate(lines) if ln.startswith(_AVERAGING_OPENERS))
    end = next(i for i, ln in enumerate(lines[start:], start)
               if ln.startswith("JOBS is how many"))
    return " ".join(lines[start:end])


@pytest.mark.parametrize("folded,expected,absent", [
    (False, ("Not averaged over time", "most recent scrape",
             "cores/GB/GPUs each holds now", "--runtime-avg"),
     ("whole runtime", "resource-time")),
    (True, ("Averaged over time", "mean over its whole runtime",
            "Pooled across jobs by resource-time", "10-hour job weighs ten times"),
     ("most recent scrape", "--runtime-avg")),
])
def test_the_summary_says_how_it_averaged(folded, expected, absent):
    """The block's own figures move by a third between the two reductions -- GPU% 77%
    instant against 71% folded on one live partition -- and the only thing that ever
    distinguished them was the "Used/GPU" versus "Used/GPU-hr" suffix. Two reports of
    the same partition then disagree with nothing on screen to say why.

    The instant form names the flag because there is another answer to reach for; the
    folded form does not, because by then there is not.
    """
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True, time_weighted=folded), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    line = _averaging_line(out.getvalue())
    for phrase in expected:
        assert phrase in line, phrase
    for phrase in absent:
        assert phrase not in line, phrase


def test_the_note_and_the_used_label_cannot_disagree():
    """Both are read off time_weighted, so "Used/GPU-hr" and "newest scrape" can no longer
    appear together -- the contradiction `jobscope -j ID` printed for a running job."""
    for folded, unit in ((True, "Used/GPU-hr:"), (False, "Used/GPU:")):
        out = io.StringIO()
        renderer = report.SummaryRenderer(
            CTX, RenderOptions(view="all", header=True, time_weighted=folded), out)
        renderer.add(build_rows(["1", "2"], {"1": _gpu_job("1", {"0": 90.0}),
                                  "2": _gpu_job("2", {"0": 10.0})}, {}))
        renderer.finish()
        text = out.getvalue()
        assert unit in text
        assert ("most recent scrape" in text) is not folded, text


def test_a_single_job_is_not_described_as_pooled():
    """"Pooled across jobs" and "a 10-hour job weighs ten times a 1-hour one" would be
    describing a mean of one. The reduction still gets named -- that half is true of a
    single job too."""
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True, time_weighted=True), out)
    renderer.add(build_rows(["1"], {"1": _gpu_job("1", {"0": 90.0})}, {}))
    renderer.finish()
    text = out.getvalue()
    line = _averaging_line(text)
    assert "this job's mean over its whole runtime" in line
    assert "Pooled" not in line and "weighs ten times" not in line
    # And the bars agree. Joined because the note wraps to the bars' own width.
    flat = " ".join(ln.strip() for ln in text.splitlines())
    assert "bars are the USED share above -- averaged over the job's whole runtime" in flat
    assert "pooled by" not in flat


@pytest.mark.parametrize("folded,names_flag", [(False, True), (True, False)])
def test_the_bars_note_names_the_flag_only_when_there_is_another_answer(folded, names_flag):
    """Section 2 is the fastest read in the block, so it is the one a reader is most
    likely to arrive at first -- and it has to be actionable there, not only in section 1.
    Folded output offers nothing, the flag being in effect or implied already."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    _one, two, _three = _sections(records, time_weighted=folded)
    note = " ".join(ln.strip() for ln in two[2] if "\u2588" not in ln and "\u2591" not in ln)
    assert note.startswith("bars are the USED share above --"), note
    assert ("--runtime-avg" in note) is names_flag, note


@pytest.mark.parametrize("folded,caveated", [(False, True), (True, False)])
def test_a_single_job_verdict_says_what_it_graded(folded, caveated):
    """"Efficiency: good" is the most quotable line in the report and the one least able
    to defend itself -- a categorical word. A job cycling 0-100% between scrapes earns it
    or loses it on which second the report ran, where its runtime mean is stable. Every
    other section says what it averaged; this one asserted a judgement.
    """
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True, time_weighted=folded), out)
    renderer.add(build_rows(["1"], {"1": _gpu_job("1", {"0": 90.0})}, {}))
    renderer.finish()
    text = out.getvalue()
    assert "Efficiency:" in text and "Graded by" in text      # the premise
    assert ("Graded on the most recent scrape" in text) is caveated, text
    assert ("--runtime-avg to grade the runtime mean" in text) is caveated, text


def test_the_bars_note_never_widens_its_section():
    """_print_sections rules each section to its widest line, so a one-line note runs
    half again the length of a bar and leaves the rule extending past the chart."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    _one, two, _three = _sections(records)
    widest = max(len(ln) for ln in two[2] if "█" in ln or "░" in ln)
    assert all(len(ln) <= widest for ln in two[2])


def test_noheader_drops_both_averaging_notes():
    """Furniture: it says how to read the numbers rather than being one of them."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=False), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    text = out.getvalue()
    assert not any(o in text for o in _AVERAGING_OPENERS)
    assert "bars are the USED share above" not in text
    assert "█" in text          # the bars themselves stay


def test_the_legend_names_each_metrics_cutoffs_once_they_differ():
    """One sentence cannot be true of the table when the rows are graded differently, so
    it becomes a list -- of the metrics in the table, and of the two edges its three
    columns divide on. A `wasteful` difference is deliberately *not* enough: that edge
    splits BAD from BAD, so naming it would point at no column."""
    from jobscope.config import Thresholds
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True,
                          thresholds=Thresholds(by_metric={"CPU%": {"inefficient": 15.0}})),
        out)
    lines = renderer._legend(["CPU%", "GPU%"])
    text = " ".join(ln.strip() for ln in lines)
    assert "each metric's own OK/GOOD edges" in text
    assert "CPU% 15/20" in text and "GPU% 10/20" in text
    assert "POWER_W is GOOD above 100 W" in text      # still stated, still a floor
    assert all(len(ln) <= 132 for ln in lines)        # and wrapped to the table width


def test_the_short_legend_holds_while_the_columns_divide_at_the_same_edges():
    """The short form asserts one pair of numbers is true of the whole table, and out of
    the box it is: every shipped metric divides BAD from NOT BAD at 10 and NOT BAD from
    GOOD at 20. Only `wasteful` differs -- CPU% 5 against 2, since a GPU job holds cores
    it never uses -- and that edge splits BAD from BAD.

    Keying the form off Thresholds.uniform compared all four edges, so that one unused
    difference pushed every default report onto the per-metric branch, spending two
    wrapped lines to list the same pair of cutoffs eight times.
    """
    from jobscope.config import Thresholds
    out = io.StringIO()

    renderer = report.SummaryRenderer(CTX, RenderOptions(view="all", header=True), out)
    text = " ".join(renderer._legend(["CPU%", "GPU%", "SM_ACT%"]))
    assert "GOOD above 20%, OK above 10%, BAD otherwise" in text
    assert "own OK/GOOD edges" not in text

    # Move a *column* edge and the list is the honest form again.
    differs = RenderOptions(view="all", header=True,
                            thresholds=Thresholds(by_metric={"CPU%": {"improvement": 30.0}}))
    text = " ".join(report.SummaryRenderer(CTX, differs, out)._legend(
        ["CPU%", "GPU%", "SM_ACT%"]))
    assert "own OK/GOOD edges" in text and "CPU% 10/30" in text


def test_a_long_per_metric_legend_wraps_rather_than_running_off():
    """Under --dcgm it names eighteen columns, which on one line would run four
    times the width of everything above it."""
    from jobscope.config import Thresholds
    headers = ["GPU%", "CPU%", "MEM%", "GMEM%", "SM_ACT%", "TENSOR%", "DRAM%",
               "OCC%", "ENGINE%", "FP16%", "FP32%", "FP64%", "MEMCP%"]
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True,
                          thresholds=Thresholds(by_metric={"CPU%": {"improvement": 30.0}})),
        out)
    lines = renderer._legend(headers)
    assert len(lines) > len(report.SummaryRenderer.STAT_LEGEND)   # it did wrap
    assert all(len(ln) <= 132 for ln in lines)
    assert "MEMCP% 10/20" in " ".join(ln.strip() for ln in lines)  # nothing dropped


def test_the_legend_is_suppressed_with_noheader_and_in_csv():
    """It is prose: it has no place in a CSV, and --noheader asked for no furniture."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    for options in (RenderOptions(view="all", header=False),
                    RenderOptions(view="all", header=True, csv=True)):
        out = io.StringIO()
        renderer = report.SummaryRenderer(CTX, options, out)
        renderer.add(build_rows(list(records), records, {}))
        renderer.finish()
        assert "RED<" not in out.getvalue()


def test_a_selection_can_be_half_idle_with_no_red_job():
    """The case that looks like a contradiction, pinned as intended behaviour.

    Every job uses about half its cores: none is below the cutoff of 10, so the red
    band is empty, yet half the allocation is unused. Concentrated waste shows up in
    the red band's resource share; systemic waste shows up only in IDLE.
    """
    records = {str(i): _timed_job(str(i), 3600, cores=4, cpu_seconds=0.5 * 4 * 3600)
               for i in range(6)}
    row = _stat_rows(records, view="cpu", time_weighted=True)["CPU%"]
    assert row["BAD"] == "0"                     # nothing BAD
    assert row["USED"] == "12h (50%)"            # yet half of 24 core-hours is idle
    assert row["GOOD"] == "6"                    # every job GOOD


def test_the_default_view_reports_both_resources():
    """A GPU job holding cores it never uses blocks other work from the node.

    The GPU lines cannot show that, so the default view prints both. `--gpu` and
    `--cpu` narrow it to the one they are about.
    """
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}

    def lines(view):
        out = io.StringIO()
        renderer = report.SummaryRenderer(
            CTX, RenderOptions(view=view, header=True), out)
        renderer.add(build_rows(list(records), records, {}))
        renderer.finish()
        return out.getvalue()

    assert list(_stat_rows(records)) == ["CPU%", "MEM%", "GPU%", "GMEM%"]
    # The leading resource names the pooled row, and GPUs lead in the wide view.
    assert "Used/GPU:" in lines("all")
    assert list(_stat_rows(records, view="gpu")) == ["GPU%", "GMEM%"]
    assert list(_stat_rows(records, view="cpu")) == ["CPU%", "MEM%"]


def test_the_totals_line_reconciles_with_its_own_percentage():
    """used is fractional even in the count form -- it is GPU-equivalents busy.

    Rounding it to an integer made the three numbers contradict the percentage:
    20 allocated and 17.4 used printed as "17 used  3 idle (13%)", and 17.4/20 is 87%.
    """
    records = {"a": _gpu_job("a", {"0": 90.0}), "b": _gpu_job("b", {"0": 80.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(CTX, RenderOptions(view="gpu", header=True), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    # ALLOC and IDLE are gone; USED still reconciles with its own percentage.
    assert _stat_rows(records, view="gpu")["GPU%"]["USED"] == "1.7 (85%)"   # 1.7/2 = 85%


def test_the_used_cell_is_the_share_that_did_work():
    records = {"idle": _gpu_job("idle", {str(i): 0.0 for i in range(3)}),
               "busy": _gpu_job("busy", {"0": 100.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(CTX, RenderOptions(view="all", header=True), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    assert _stat_rows(records)["GPU%"]["USED"] == "1 (25%)"    # 1 of 4 GPUs busy


def test_every_graded_column_gets_a_row_in_column_order():
    """The set follows the view, so the block cannot drift from the table above it."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    dcgm = {j: ({"SM_ACT%": 50.0, "OCC%": 20.0, "TENSOR%": 1.0, "DRAM%": 8.0,
                 "ENGINE%": 30.0}, {})
            for j in records}
    rows = _stat_rows(records, show_dcgm=True, dcgm_data=dcgm, specs=gpu_catalog().default_specs)
    assert list(rows) == ["CPU%", "MEM%", "GPU%", "GMEM%",
                          "SM_ACT%", "TENSOR%", "DRAM%"]
    # --dcgm widens the table, so it widens the block too. ENGINE% is only in the
    # extended catalog, so it can only appear there.
    wide = _stat_rows(records, show_dcgm=True, dcgm_data=dcgm, specs=gpu_catalog().all_specs)
    assert list(wide)[:7] == list(rows)
    assert "ENGINE%" in wide and "ENGINE%" not in rows
    assert "OCC%" in wide and "OCC%" not in rows


def test_a_job_with_no_stored_summary_votes_in_no_tally():
    """Half-measured jobs made the denominators disagree.

    A finished job with no AdminComment has DCGM numbers but no CPU%/MEM%/GPU%/GMEM%.
    Feeding it to the DCGM tallies alone put SM_ACT% over 117 jobs while GPU% had 88
    on one partition, so the two Worst rows could not be compared. It is excluded
    from every tally and counted as no-summary, staying in the listing as a real job.
    """
    good = _gpu_job("good", {"0": 90.0})
    blank = dataclasses.replace(_gpu_job("blank", {"0": 5.0}), stats={})
    records = {"good": good, "blank": blank}
    dcgm = {j: ({"SM_ACT%": 5.0}, {}) for j in records}
    rows = _stat_rows(records, show_dcgm=True, dcgm_data=dcgm, specs=gpu_catalog().default_specs)
    # One denominator everywhere: only the jobstats summary-having job voted.
    for metric in ("GPU%", "SM_ACT%"):
        good, ok, bad = _counts(rows[metric])
        assert good + ok + bad == 1, (metric, rows[metric])
        assert rows[metric]["JOBS"] == "1", rows[metric]
    # And it is still rendered, with the omission stated.
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True, show_dcgm=True), out,
        specs=gpu_catalog().default_specs)
    renderer.add(build_rows(list(records), records, dcgm))
    renderer.finish()
    text = out.getvalue()
    assert "blank" in text and "no-jobstats=1" in text


def test_a_jobstats_summary_without_gpu_data_still_votes():
    """The exclusion is for a *missing* summary, not a CPU-only one.

    A CPU-only job legitimately has no GPU% and must still count toward CPU%,
    otherwise the whole CPU side of a mixed partition would vanish.
    """
    records = {"cpu_only": _gpu_job("cpu_only", None),
               "gpu": _gpu_job("gpu", {"0": 90.0})}
    rows = _stat_rows(records)
    cpu_red, cpu_yellow, cpu_green = _counts(rows["CPU%"])[::-1]
    assert cpu_red + cpu_yellow + cpu_green == 2      # both jobs voted on CPU%
    gpu_red, gpu_yellow, gpu_green = _counts(rows["GPU%"])[::-1]
    assert gpu_red + gpu_yellow + gpu_green == 1      # only one has a GPU%


def test_a_metric_no_job_reported_is_omitted():
    """A row of zeros would read as "nothing used it", not "nothing measured it"."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    dcgm = {j: ({"SM_ACT%": 50.0}, {}) for j in records}    # OCC%/TENSOR%/DRAM% absent
    rows = _stat_rows(records, show_dcgm=True, dcgm_data=dcgm, specs=gpu_catalog().default_specs)
    assert "SM_ACT%" in rows
    assert "OCC%" not in rows and "TENSOR%" not in rows and "DRAM%" not in rows


def test_each_metric_is_weighted_by_the_resource_it_measures():
    """CPU% by core-hours, MEM% by GB-hours, the GPU family by GPU-hours.

    Two one-hour jobs, each holding 4 cores, 2 GPUs and 16GB of host memory. The
    three denominators must match the resource each metric measures, not each other.
    """
    records = {j: _timed_job(j, 3600, gpu_util=50.0, gpus=2, cores=4,
                             cpu_seconds=3600 * 4 * 0.5, total_gb=16)
               for j in ("a", "b")}
    # Read off IDLE, the only amount left: each is half of that metric's own
    # allocation, so the three denominators are visibly different resources.
    rows = _stat_rows(records, time_weighted=True)
    assert rows["CPU%"]["USED"] == "4h (50%)"     # half of 2 jobs x 4 cores x 1h
    assert rows["MEM%"]["USED"] == "16GBh (50%)"  # half of 2 jobs x 16GB x 1h
    assert rows["GPU%"]["USED"] == "2h (50%)"     # half of 2 jobs x 2 GPUs x 1h
    assert rows["GMEM%"]["USED"] == "2h (50%)"    # GPU-hours as well


def test_byte_amounts_promote_to_terabytes_consistently_within_a_row():
    """A row must not read "126.5TBh allocated, 5414.7GBh used"."""
    records = {j: _timed_job(j, 3600, gpu_util=50.0, total_gb=20000, used_gb=1000)
               for j in ("a", "b")}
    row = _stat_rows(records, time_weighted=True)["MEM%"]
    assert "TBh" in row["USED"], row         # promoted, not a 9-digit byte-second


def test_the_band_cells_are_tinted_and_idle_takes_the_pooled_grade():
    records = {"1": _gpu_job("1", {"0": 2.0}), "2": _gpu_job("2", {"0": 4.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="gpu", header=True, color=True,
                           thresholds=_thresholds()), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    line = [ln for ln in out.getvalue().splitlines() if ln.startswith("GPU%")][0]
    # The red band cell, plus IDLE -- pooled utilization is 3%, well under the cutoff.
    assert line.count(report._SGR["red"]) == 2
    assert report._SGR["yellow"] in line and report._SGR["green"] in line
    # METRIC / RED< / ALLOC / USED carry no colour, so the row starts plain.
    assert not line.startswith("\033")


def test_the_table_is_plain_in_csv_mode_and_when_color_is_off():
    records = {"1": _gpu_job("1", {"0": 2.0}), "2": _gpu_job("2", {"0": 90.0})}
    for options in (RenderOptions(view="gpu", header=True, color=False),
                    RenderOptions(view="gpu", header=True, color=True, csv=True)):
        out = io.StringIO()
        renderer = report.SummaryRenderer(CTX, options, out)
        renderer.add(build_rows(list(records), records, {}))
        renderer.finish()
        assert "\033" not in out.getvalue()


def test_worst_ranks_by_wasted_resource_not_by_size():
    """A large job at a mediocre rate wastes more than a small one at zero."""
    # Both wasteful under the <2% cutoff, so both are candidates.
    records = {"big": _timed_job("big", 100 * 3600, gpu_util=1.0),     # 99 GPU-h idle
               "small": _timed_job("small", 10 * 3600, gpu_util=0.0)}  # 10 GPU-h idle
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True, time_weighted=True), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    lines = out.getvalue().splitlines()
    i = next(idx for idx, ln in enumerate(lines) if ln.startswith("Wasteful GPU"))
    worst = lines[i + 1]
    assert worst.index("big") < worst.index("small")


def _wasteful_blocks(text):
    """``{label: text}`` for the Wasteful rows of a text-rendered table.

    Keyed on the label without its "(n/total)" count, so a test can ask for
    "Wasteful GPU" without knowing the counts. `text` joins the row's job lines
    (the heading itself is excluded, so its criteria text -- "GPU < 10%" and the
    like -- cannot be mistaken for a job payload) -- a test can search it with
    `in`/`.index()` without caring whether an entry landed on the first job line
    or wrapped onto another.
    """
    blocks = {}
    label = None
    for ln in text.splitlines():
        if ln.startswith("Wasteful"):
            label = ln.split("(")[0].strip()
            blocks[label] = []
        elif ln.startswith("Jobs:"):
            label = None
        elif label is not None:
            blocks[label].append(ln)
    return {label: "\n".join(rows) for label, rows in blocks.items()}


def _worst_lines(records, view="all", time_weighted=True):
    """``_wasteful_blocks`` of a plain (non-power) table."""
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view=view, header=True, time_weighted=time_weighted), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    return _wasteful_blocks(out.getvalue())


def test_the_combined_worst_requires_red_in_every_metric():
    """AND, not OR: a job idle by one measure does not reach a combined row.

    gpuhog wastes nearly all the GPU-time but keeps 90% of its cores; cpuhog is the
    mirror. Neither belongs on a row that claims "idle by both measures", and under
    an OR both appeared -- alongside jobs showing a 0% component, which is what gave
    the OR away.
    """
    records = {
        "gpuhog": _timed_job("gpuhog", 3600, gpu_util=0.0, gpus=10, cores=2,
                             cpu_seconds=6480),          # CPU% 90, green
        "cpuhog": _timed_job("cpuhog", 3600, gpu_util=95.0, gpus=1, cores=100,
                             cpu_seconds=0),             # GPU% 95, green
        "both": _timed_job("both", 3600, gpu_util=0.0, gpus=4, cores=40,
                           cpu_seconds=0),               # red in both
    }
    lines = _worst_lines(records)
    # The single-metric rows still list whoever is red in that one metric.
    assert "gpuhog" in lines["Wasteful GPU"] and "cpuhog" in lines["Wasteful CPU"]
    combined = lines["Wasteful gpu-cpu"]
    assert "both" in combined
    assert "gpuhog" not in combined and "cpuhog" not in combined


def test_the_combined_worst_orders_by_normalized_waste_and_shows_values():
    """Ranked by summed waste share, but the cells print the qualifying values.

    A share written "12%gpu" reads exactly like a utilization of 12%, which inverts
    the meaning: every value on the row is *below* its cutoff. So the order carries
    the ranking and the cells say why each job is there.
    """
    records = {"big": _timed_job("big", 3600, gpu_util=0.0, gpus=8, cores=8,
                                 cpu_seconds=0),
               "small": _timed_job("small", 3600, gpu_util=0.0, gpus=4, cores=4,
                                   cpu_seconds=0)}
    combined = _worst_lines(records)["Wasteful gpu-cpu"]
    # big wastes twice the resource, so it leads.
    assert combined.index("big") < combined.index("small")
    # And the cells are the values, not shares: both ran at 0%.
    assert "big:gpu0%/cpu0%" in combined and "small:gpu0%/cpu0%" in combined
    assert "%gpu" not in combined


def test_a_job_green_in_one_metric_is_absent_from_the_combined_rows():
    """The inverse of the conjunction, stated directly.

    Its waste in the other metric is still real -- and still counted in that
    metric's own total and its own Wasteful row -- it just does not qualify here.
    """
    records = {
        "red_gpu": _timed_job("red_gpu", 3600, gpu_util=0.0, gpus=1,
                              cores=10, cpu_seconds=18000),   # CPU% 50, green
        "red_both": _timed_job("red_both", 3600, gpu_util=1.0, gpus=1,
                               cores=10, cpu_seconds=0),      # wasteful in both
    }
    lines = _worst_lines(records)
    assert "red_gpu" in lines["Wasteful GPU"]          # its own row still names it
    assert "red_gpu" not in lines["Wasteful gpu-cpu"]     # but not the conjunction
    assert "red_both" in lines["Wasteful gpu-cpu"]


def test_no_combined_line_when_only_one_resource_wasted_anything():
    """With nothing idle on one side, the combined view repeats the other."""
    records = {"1": _timed_job("1", 3600, gpu_util=0.0, gpus=1, cores=4,
                               cpu_seconds=4 * 3600),          # CPU% 100
               "2": _timed_job("2", 3600, gpu_util=10.0, gpus=1, cores=4,
                               cpu_seconds=4 * 3600)}
    lines = _worst_lines(records)
    assert "Wasteful GPU" in lines and "Wasteful gpu-cpu" not in lines


def test_narrow_views_show_only_their_own_worst_row():
    records = {"1": _timed_job("1", 3600, gpu_util=0.0, gpus=2, cores=8, cpu_seconds=0),
               "2": _timed_job("2", 3600, gpu_util=5.0, gpus=1, cores=4, cpu_seconds=0)}
    assert set(_worst_lines(records, view="gpu")) == {"Wasteful GPU"}
    assert set(_worst_lines(records, view="cpu")) == {"Wasteful CPU"}
    assert set(_worst_lines(records)) == {"Wasteful GPU", "Wasteful CPU", "Wasteful gpu-cpu"}


def test_wasteful_headings_state_their_own_criteria():
    """Every heading -- single-metric and combined -- names the cutoff it used,
    not a bare number the reader has to look up, so it stays true after a site
    tunes config.toml. The cutoff is `wasteful` -- the same edge --ts --eff's
    own "wasteful" tier is graded against -- and CPU%'s own default is 5, not the
    catalog-wide 2, because a GPU job legitimately holds cores it never uses."""
    records = {
        "gpuhog": _timed_job("gpuhog", 3600, gpu_util=0.0, gpus=10, cores=2,
                             cpu_seconds=6480),
        "cpuhog": _timed_job("cpuhog", 3600, gpu_util=95.0, gpus=1, cores=100,
                             cpu_seconds=0),
        "both": _timed_job("both", 3600, gpu_util=0.0, gpus=4, cores=40,
                           cpu_seconds=0),
    }
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True, time_weighted=True), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    text = out.getvalue()
    assert "Wasteful GPU (" in text and "GPU < 2%" in text
    assert "Wasteful CPU (" in text and "CPU < 5%" in text
    assert "Wasteful gpu-cpu (" in text and "GPU < 2%, CPU < 5%" in text


def test_a_custom_wasteful_cutoff_changes_membership_and_the_heading():
    """A site-tuned `wasteful` moves both the Wasteful row's membership and its
    printed criteria -- the one definition --ts --eff also reads."""
    from jobscope.config import Thresholds
    # GPU% 0 (always wasteful); CPU% 6 -- wasteful at a cutoff of 8, not at 2.
    records = {"1": _timed_job("1", 3600, gpu_util=0.0, gpus=1, cores=10,
                               cpu_seconds=int(0.06 * 3600 * 10)),
               "2": _timed_job("2", 3600, gpu_util=90.0, gpus=1, cores=1,
                               cpu_seconds=int(0.9 * 3600))}

    def worst(thresholds):
        out = io.StringIO()
        renderer = report.SummaryRenderer(
            CTX, RenderOptions(view="all", header=True, time_weighted=True,
                              thresholds=thresholds), out)
        renderer.add(build_rows(list(records), records, {}))
        renderer.finish()
        return out.getvalue()

    default = worst(Thresholds())
    assert "Wasteful gpu-cpu" not in default          # CPU% 6 is not wasteful at 2%
    assert "Wasteful CPU" not in default

    widened = worst(Thresholds(defaults={"wasteful": 8}))
    assert "Wasteful gpu-cpu" in widened and "GPU < 8%, CPU < 8%" in widened
    assert "Wasteful CPU" in widened and "CPU < 8%" in widened

    # And tuning CPU% alone moves only CPU%'s side of the row, which is the whole
    # point of the edges being per metric.
    cpu_only = worst(Thresholds(by_metric={"CPU%": {"wasteful": 8.0}}))
    assert "Wasteful CPU" in cpu_only and "CPU < 8%" in cpu_only
    assert "Wasteful gpu-cpu" in cpu_only and "GPU < 2%, CPU < 8%" in cpu_only


def test_the_power_heading_states_watts_not_percent():
    records = {"a": _power_job("a", 3600, None, gpu_util=90.0, cpu_seconds=0),
               "b": _power_job("b", 7200, None, gpu_util=90.0, cpu_seconds=0)}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True, show_dcgm=True, time_weighted=True),
        out, specs=gpu_catalog().default_specs)
    renderer.add(build_rows(list(records), records,
                 {j: ({"POWER_W": 50.0, "SM_ACT%": 90.0}, {}) for j in records}))
    renderer.finish()
    assert "POWER < 100W" in out.getvalue()


def _power_job(jid, seconds, watts, gpu_util=50.0, gpus=1, cores=2, cpu_seconds=None):
    """A GPU job; `watts` is supplied separately via the DCGM dict."""
    return _timed_job(jid, seconds, gpu_util=gpu_util, gpus=gpus, cores=cores,
                      cpu_seconds=cpu_seconds)


def _worst_with_power(records, watts, **kw):
    """``_wasteful_blocks`` of a table with per-job POWER_W taken from `watts`."""
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view=kw.get("view", "all"), header=True, show_dcgm=True,
                           time_weighted=kw.get("time_weighted", True)),
        out, specs=gpu_catalog().default_specs)
    dcgm = {j: ({"POWER_W": watts[j], "SM_ACT%": kw.get("sm", {}).get(j, 50.0)}, {})
            for j in records}
    renderer.add(build_rows(list(records), records, dcgm))
    renderer.finish()
    return _wasteful_blocks(out.getvalue())


def test_power_waste_is_gpu_hours_below_the_floor():
    """A floor asserts idle-or-not, so a job below it wastes all of its GPU-time.

    Scaling by how far below would imply 50 W wastes twice what 100 W does, and
    watts are not utilization. Ranking is therefore by GPU-hours held while idle.
    """
    records = {"long_idle": _power_job("long_idle", 10 * 3600, None),
               "short_idle": _power_job("short_idle", 3600, None),
               "busy": _power_job("busy", 50 * 3600, None)}
    rows = _worst_with_power(records, {"long_idle": 73.0, "short_idle": 73.0,
                                      "busy": 300.0})
    power = rows["Wasteful POWER"]
    assert power.index("long_idle") < power.index("short_idle")
    # The 300 W job is green, so it is absent however large it is.
    assert "busy" not in power
    # Watts, not percent, and the entry carries the elapsed time.
    assert "long_idle:73W:10h(10:00:00)" in power
    assert ":73%" not in power


def _owned(jid, user, seconds, gpu_util, gpus=1):
    """A red GPU job belonging to `user`, having run for `seconds`."""
    return dataclasses.replace(
        _timed_job(jid, seconds, gpu_util=gpu_util, gpus=gpus, cpu_seconds=0),
        user=user)


def _worst_block(records, **kw):
    """The Problem-jobs section's lines."""
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view=kw.get("view", "gpu"), header=True,
                           time_weighted=True, color=kw.get("color", False),
                           thresholds=kw.get("thresholds"),
                           worst_jobs=kw.get("worst_jobs", report.DEFAULT_WORST_JOBS),
                           long_running=kw.get("long_running",
                                               report.LONG_RUNNING)), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    lines, keep = [], False
    for line in out.getvalue().splitlines():
        if line.startswith("3. "):
            keep = True
            continue
        if keep and not set(line) <= {"-", ""}:
            lines.append(line)
    return lines


def test_the_worst_rows_group_jobs_under_their_owner():
    """One user usually owns several of the worst jobs; naming them once says more."""
    records = {"a": _owned("a", "avenkat", 4 * 3600, 0.0),
               "b": _owned("b", "avenkat", 3 * 3600, 1.0),
               "c": _owned("c", "binxu", 3600, 1.4)}
    block = _worst_block(records)
    i = next(i for i, ln in enumerate(block) if ln.startswith("Wasteful GPU"))
    first = block[i + 1]
    assert first.lstrip().startswith("avenkat|")
    assert first.count("avenkat") == 1           # named once, not per job
    assert "a:" in first and "b:" in first
    # The second user continues on its own line, under the same column.
    following = block[i + 2]
    assert following.lstrip().startswith("binxu|")
    assert following.index("binxu") == first.index("avenkat")


def test_each_entry_carries_value_wasted_and_elapsed():
    records = {"a": _owned("a", "u1", 2 * 3600, 0.0, gpus=3),
               "b": _owned("b", "u2", 3600, 5.0)}
    block = _worst_block(records)
    i = next(i for i, ln in enumerate(block) if ln.startswith("Wasteful GPU"))
    assert "a:0%:6h(2:00:00)" in block[i + 1]     # 3 GPUs x 2h all idle


def test_every_wasteful_entry_is_red_and_the_long_ones_brighter():
    """Hours of idle hardware do not come back; a brief bad job costs little -- but both
    are wasteful, and a section titled Wasteful printing plain text read as ungraded.
    So both are painted, and long_running is the brighter red that separates them."""
    records = {"long": _owned("long", "u1", 3 * 3600 + 1, 0.0),
               "brief": _owned("brief", "u1", 3 * 3600 - 1, 0.0)}
    block = "".join(_worst_block(records, color=True, thresholds=_thresholds()))
    assert report._SGR["long_running"] + "long:" in block
    assert report._SGR["wasteful"] + "brief:" in block
    # Distinct, or the duration signal is lost.
    assert report._SGR["long_running"] != report._SGR["wasteful"]
    assert report._SGR["long_running"] + "brief:" not in block
    # Exactly at the boundary counts as brief: the test is strictly greater.
    assert report.LONG_RUNNING == 3 * 3600
    # And nothing is tinted when colour is off.
    assert "\033" not in "".join(_worst_block(records))


def test_a_long_entry_list_wraps_under_the_user_column():
    """Rather than running past the table width."""
    records = {"job%02d" % i: _owned("job%02d" % i, "sameuser", 3600, 0.0)
               for i in range(3)}
    lines = [ln for ln in _worst_block(records) if "job" in ln]
    assert all(len(report._ESC_RE.sub("", ln)) <= report.SummaryRenderer.WORST_WIDTH
               for ln in lines)


def test_a_worst_row_per_named_metric_in_a_fixed_order():
    # Wasteful in all four, so every row has something to print.
    records = {"a": _power_job("a", 3600, None, gpu_util=1.0, cpu_seconds=0),
               "b": _power_job("b", 7200, None, gpu_util=1.0, cpu_seconds=0)}
    rows = _worst_with_power(records, {"a": 73.0, "b": 74.0}, sm={"a": 1.0, "b": 1.0})
    assert list(rows) == ["Wasteful GPU", "Wasteful SM", "Wasteful POWER", "Wasteful CPU",
                          "Wasteful gpu-cpu", "Wasteful all"]


def test_the_four_metric_row_prints_every_component_share():
    # Red in all four, which the conjunction requires.
    records = {"a": _power_job("a", 3600, None, gpu_util=0.0, cpu_seconds=0),
               "b": _power_job("b", 7200, None, gpu_util=0.0, cpu_seconds=0)}
    rows = _worst_with_power(records, {"a": 73.0, "b": 73.0}, sm={"a": 0.0, "b": 0.0})
    # Four tagged values, each under its cutoff -- which is why the job qualified.
    # Power carries its unit (watts); the percentage metrics carry theirs too, so
    # none of these ever reads as a band index or rank.
    assert "gpu0%/sm0%/pw73W/cpu0%" in rows["Wasteful all"]
    # The two-metric row names only its two.
    assert "gpu0%/cpu0%" in rows["Wasteful gpu-cpu"]
    assert "sm" not in rows["Wasteful gpu-cpu"] and "pw" not in rows["Wasteful gpu-cpu"]


def _power_stats(watts):
    """The stats table for two GPU jobs drawing `watts`."""
    records = {"a": _power_job("a", 3600, None), "b": _power_job("b", 7200, None)}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True, show_dcgm=True, time_weighted=True),
        out, specs=gpu_catalog().default_specs)
    renderer.add(build_rows(list(records), records,
                 {j: ({"POWER_W": watts, "SM_ACT%": 40.0}, {}) for j in records}))
    renderer.finish()
    lines = out.getvalue().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("METRIC"))
    # Stop at the blank line: section 2 repeats the same metric names as bars.
    table = itertools.takewhile(bool, lines[start + 1:])
    return {ln.split()[0]: ln for ln in table
            if not ln.startswith(("Wasteful", "Jobs"))}


def _power_rows(watts):
    """The same table, name-keyed per cell -- see _stat_rows on why not by position."""
    header = report.SummaryRenderer.STAT_HEADERS[1:]
    return {metric: dict(zip(header, re.split(r"\s{2,}", line.strip())[1:]))
            for metric, line in _power_stats(watts).items()}


def test_power_gets_a_stats_table_row():
    """Its USED is resource-time spent ABOVE the watt floor, not a used fraction.

    "Used watts" means nothing, but "GPU-hours that drew more than the idle floor" is
    a real quantity -- and the complement of the one a floor actually asserts.
    """
    under = _power_stats(73.0)
    assert "POWER_W" in under and "SM_ACT%" in under
    # And marked, so 0% does not read as a utilization beside seven that are.
    assert "(0% >100W)" in under["POWER_W"]      # no GPU-hour cleared the floor
    assert under["POWER_W"].split()[1] == "0h"

    over = _power_stats(300.0)
    assert "(100% >100W)" in over["POWER_W"]     # all 3h of it, and marked as time
    assert over["POWER_W"].split()[1] == "3h"    # 1h + 2h of GPU-time


def _power_used(watts):
    """The USED cell of the POWER_W row, as ``"3h (100% >100W)"``.

    Read by name rather than by slicing words: the cell carries the watt floor now, so it
    contains a space of its own and a fixed word count no longer finds its end.
    """
    return _power_rows(watts)["POWER_W"]["USED"]


def test_powers_used_is_all_or_nothing_not_proportional():
    """50 W does not waste twice what 100 W does; watts are not utilization.

    The bands still separate them -- 101 W is yellow where 500 W is green -- but the
    time above the floor is the same nothing either way.
    """
    assert _power_used(50.0) == _power_used(99.0) == "0h (0% >100W)"
    assert _power_used(101.0) == _power_used(500.0) == "3h (100% >100W)"


def test_a_metric_with_no_red_job_prints_no_worst_row():
    records = {"a": _power_job("a", 3600, None, gpu_util=90.0),
               "b": _power_job("b", 7200, None, gpu_util=80.0)}
    rows = _worst_with_power(records, {"a": 400.0, "b": 500.0}, sm={"a": 60.0, "b": 70.0})
    assert "Wasteful GPU" not in rows and "Wasteful POWER" not in rows
    # CPU% is 50 against a cutoff of 10, so that row is absent too, and with no
    # metric wasting anything the combined rows cannot be computed either.
    assert rows == {}


def test_no_worst_line_when_nothing_is_red():
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 80.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(CTX, RenderOptions(view="all", header=True), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    # No label at all, not merely the old "Worst:" spelling.
    assert "Wasteful" not in out.getvalue()


def test_the_block_follows_the_cpu_view_to_cores():
    """A --cpu run bands CPU% over cores, not GPU% over GPUs."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(CTX, RenderOptions(view="cpu", header=True), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    text = out.getvalue()
    assert list(_stat_rows(records, view="cpu")) == ["CPU%", "MEM%"]
    assert "Used/cpu:" in text
    # GPU totals are noise in a view that hid the GPU columns.
    jobs = [ln for ln in text.splitlines() if ln.startswith("Jobs:")][0]
    assert "gpu-jobs" not in jobs and "gpus=" not in jobs


# --- the time-weighted (per resource-hour) mean -----------------------------

def _timed_job(jid, seconds, gpu_util=None, gpus=1, cores=2, cpu_seconds=None,
               used_gb=8, total_gb=16):
    """A job that ran for `seconds`, holding `cores` cores and `gpus` GPUs.

    Separate from :func:`_gpu_job`, which fixes the runtime at 100s: the point
    here is that runtimes differ. ``cpu_seconds`` defaults to half the available
    core-seconds, i.e. CPU% = 50.
    """
    if cpu_seconds is None:
        cpu_seconds = 0.5 * seconds * cores
    node = {"total_time": cpu_seconds, "cpus": cores,
            "used_memory": used_gb * GIB, "total_memory": total_gb * GIB}
    if gpu_util is not None:
        node["gpu_utilization"] = {str(i): gpu_util for i in range(gpus)}
        node["gpu_used_memory"] = {str(i): 40 * GIB for i in range(gpus)}
        node["gpu_total_memory"] = {str(i): 80 * GIB for i in range(gpus)}
    return JobRecord(jobid=jid, state="COMPLETED", name="j",
                     runtime="%d:%02d:%02d" % (seconds // 3600,
                                               seconds // 60 % 60, seconds % 60),
                     nodes="1",
                     gpus=gpus if gpu_util is not None else 0,
                     stats={"total_time": seconds, "nodes": {"n1": node}},
                     start=1000, end=1000 + seconds, duration=seconds,
                     jobid_raw=jid, cluster="c", user="alice")


def _tw_footers(records, **kw):
    """:func:`_footers` with resource-time weighting on."""
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", csv=True, header=True, time_weighted=True,
                           show_dcgm=kw.get("show_dcgm", False)),
        out, specs=kw.get("specs"))
    renderer.add(build_rows(list(records), records, kw.get("dcgm_data") or {}))
    renderer.finish()
    rows = {r.split(",")[0]: r.split(",") for r in out.getvalue().splitlines()}
    return {label: dict(zip(rows["JOBID"], cells)) for label, cells in rows.items()}


def test_short_jobs_do_not_outvote_one_long_job():
    """The skew that motivates this row: 100 five-minute jobs against one 2-day job.

    Per job the long job is 1/101 of the answer, so a busy two-day run reads as an
    idle cluster. Weighted by GPU-hours it is 85% of the resource-time, which is
    what actually happened to the hardware.
    """
    records = {"long": _timed_job("long", 2 * 86400, gpu_util=100.0)}
    records.update({str(i): _timed_job(str(i), 300, gpu_util=0.0) for i in range(100)})
    footers = _tw_footers(records)
    # A per-job mean would read 1 -- (100 + 0*100)/101 -- and call this idle.
    # 172800 GPU-seconds busy of 202800 allocated is 85.
    assert footers["UsedPerGPUHour"]["GPU%"] == "85"
    assert "allocated=56.3h" in footers["StatGPU%"].values()   # 202800s / 3600
    # 100 of 101 jobs are red, but they hold only 15% of the resource-time.
    assert "red=100" in footers["StatGPU%"].values()


def test_time_weighting_changes_the_answer_when_only_runtimes_differ():
    """Equal GPU counts, unequal runtimes: only the time weighting sees it."""
    records = {"a": _timed_job("a", 36000, gpu_util=90.0),
               "b": _timed_job("b", 360, gpu_util=0.0)}
    assert _footers(records)["UsedPerGPU"]["GPU%"] == "45"        # (90 + 0)/2 GPUs
    assert _tw_footers(records)["UsedPerGPUHour"]["GPU%"] == "89"  # by GPU-hours


def test_the_time_weighted_cpu_mean_is_the_pooled_utilization():
    """Weighting CPU% by core-hours reproduces 100 x sum(cpu_s) / sum(core_s).

    That identity is the reason to weight by allocation size as well as time: the
    row is not a nicer average, it is the real utilization of the pool.
    """
    records = {"a": _timed_job("a", 7200, cores=4, cpu_seconds=28800),   # CPU% 100
               "b": _timed_job("b", 300, cores=8, cpu_seconds=240)}      # CPU% 10
    footers = _tw_footers(records)
    pooled = 100 * (28800 + 240) / (7200 * 4 + 300 * 8)
    assert footers["UsedPerCPUHour"]["CPU%"] == str(round(pooled))     # 93
    # Per core rather than per core-hour it is 40 -- (100*4 + 10*8)/12 -- and a
    # per-job mean would say 55. Only the core-hour form is the real utilization.
    assert _footers(records)["UsedPerCPU"]["CPU%"] == "40"


def test_a_job_with_no_runtime_is_excluded_and_reported():
    """Without an elapsed time a job cannot be placed on the resource-hour scale."""
    good = _timed_job("a", 3600, gpu_util=100.0)
    bad = dataclasses.replace(_timed_job("b", 3600, gpu_util=0.0), duration=None)
    footers = _tw_footers({"a": good, "b": bad})
    assert footers["UsedPerGPUHour"]["GPU%"] == "100"    # only the timed job counts
    assert "no-runtime=1" in footers["Jobs"].values()    # and the omission is stated


def test_instantaneous_running_values_are_not_time_weighted():
    """A snapshot is one moment for every job, so elapsed time is not its weight.

    Weighting a single scrape by two days of runtime would claim that instant
    represents those two days. Only --runtime-avg and finished jobs carry runtime-long
    values, so only they get the GPU-hour row.
    """
    records = {"a": _timed_job("a", 2 * 86400, gpu_util=100.0),
               "b": _timed_job("b", 300, gpu_util=0.0, gpus=3)}
    assert "UsedPerGPUHour" not in _footers(records)     # time_weighted off
    assert _footers(records)["UsedPerGPU"]["GPU%"] == "25"   # (100*1 + 0*3)/4


def test_plot_skips_the_time_weighted_footer():
    records = {"a": _timed_job("a", 7200, gpu_util=100.0),
               "b": _timed_job("b", 300, gpu_util=0.0)}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", csv=True, header=True, time_weighted=True), out)
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    _, rows = plot.parse_csv(io.StringIO(out.getvalue()))
    assert {r["JOBID"] for r in rows} == {"a", "b"}


def test_plot_skips_the_weighted_footer_too():
    records = {"1": _gpu_job("1", {"0": 0.0}),
               "2": _gpu_job("2", {"0": 100.0, "1": 100.0})}
    text = _render(summarize, list(records), records, {}, CTX,
                   RenderOptions(view="all", show_dcgm=False, csv=True, header=True))
    _, rows = plot.parse_csv(io.StringIO(text))
    assert {r["JOBID"] for r in rows} == {"1", "2"}


# --- highlighting efficient and inefficient jobs ----------------------------

_ESC = re.compile(r"\033\[[0-9;]*m")


def _thresholds():
    from jobscope.config import Thresholds
    return Thresholds()


def _render_colored(records, color=True, csv=False, dcgm_data=None, show_dcgm=False):
    options = RenderOptions(view="all", show_dcgm=show_dcgm, csv=csv, header=True,
                            color=color, thresholds=_thresholds())
    return _render(summarize, list(records), records, dcgm_data or {}, CTX, options)


def test_no_color_by_default(gpu_record):
    """RenderOptions must not tint unless a caller has decided the sink wants it."""
    text = _render(summarize, ["100"], {"100": gpu_record}, {}, CTX,
                   RenderOptions(view="all", show_dcgm=False, header=True))
    assert "\033[" not in text


def test_low_utilization_is_red_and_high_is_green():
    records = {"1": _gpu_job("1", {"0": 5.0}), "2": _gpu_job("2", {"0": 95.0})}
    lines = {r.split()[0]: r for r in _render_colored(records).splitlines()
             if r and r.split()[0] in ("1", "2")}
    assert "\033[31m5" in lines["1"]        # GPU% 5 is below the red cutoff of 25
    assert "\033[32m95" in lines["2"]       # 95 is at least twice it, so green


def test_the_middle_band_is_yellow():
    records = {"1": _gpu_job("1", {"0": 15.0}), "2": _gpu_job("2", {"0": 95.0})}
    row = [r for r in _render_colored(records).splitlines() if r.startswith("1 ")][0]
    assert "\033[33m15" in row             # >= 10 but < 20, the uniform cutoff


def test_color_does_not_shift_the_columns():
    """The escapes wrap the padded cell, so stripping them restores the plain row.

    Tinting the value instead would make str.format count the escape bytes toward
    the column width and skew everything to its right.
    """
    records = {"1": _gpu_job("1", {"0": 5.0}), "2": _gpu_job("2", {"0": 95.0})}
    assert _ESC.sub("", _render_colored(records)) == _render_colored(records, color=False)


def _render_detail(records, color=True, dcgm_data=None):
    options = RenderOptions(view="all", show_dcgm=bool(dcgm_data), csv=False,
                            header=True, color=color, thresholds=_thresholds())
    return _render(detail, list(records), records, dcgm_data or {}, CTX, options)


def _multinode_job(jid="1"):
    """A record whose summary spans two nodes, two GPUs each."""
    def node(cpu_seconds, utils):
        return {"total_time": cpu_seconds, "cpus": 4,
                "used_memory": 8 * GIB, "total_memory": 16 * GIB,
                "gpu_utilization": utils,
                "gpu_used_memory": {k: 40 * GIB for k in utils},
                "gpu_total_memory": {k: 80 * GIB for k in utils}}
    return JobRecord(
        jobid=jid, state="COMPLETED", name="j", runtime="01:00:00", nodes="2", gpus=4,
        stats={"total_time": 100,
               "nodes": {"nodeA": node(200, {"0": 90.0, "1": 80.0}),
                         "nodeB": node(100, {"0": 10.0, "1": 20.0})}},
        start=0, end=100, duration=100, jobid_raw=jid, cluster="c", user="alice")


_RESET_T = "\033[0m"
_BAR_RE = re.compile(r"(\S+)\s+(<?\d+)%\s+[\u2588\u2591]+")


def _charts(records, **kw):
    """``{unit: {metric: percent}}`` parsed back out of the --per-gpu charts.

    The charts are laid out in columns, so a heading line names several units and a
    bar line carries one bar per unit in the same order. Parsed positionally rather
    than by character offset, which would break the moment a label width changed.
    """
    options = RenderOptions(view=kw.get("view", "all"), show_dcgm=False,
                            csv=kw.get("csv", False), header=True,
                            color=kw.get("color", False), thresholds=_thresholds(),
                            nodename=kw.get("nodename"),
                            plot_avgeff=kw.get("plot_avgeff", True))
    text = _render(detail, list(records), records, {}, CTX, options)
    charts, row_units = {}, []
    for line in text.splitlines():
        plain = _ESC.sub("", line)
        bars = _BAR_RE.findall(plain)
        if bars:
            for (metric, pct), unit in zip(bars, row_units):
                charts[unit][metric] = int(pct.lstrip("<"))
        elif plain.startswith("    ") and plain.strip() and "Efficiency" not in plain:
            row_units = re.split(r"\s{3,}", plain.strip())
            for unit in row_units:
                charts.setdefault(unit, {})
    return charts, text


def test_in_columns_packs_blocks_side_by_side_in_reading_order():
    from jobscope.report import in_columns
    blocks = [["a1", "a2"], ["b1", "b2"], ["c1", "c2"]]
    out = in_columns(blocks, columns=2, gap=2)
    # Two per row, then the odd block alone -- both of its lines.
    assert out == ["a1  b1", "a2  b2", "c1", "c2"]


def test_the_column_count_follows_the_available_width():
    """Four blocks fit a wide terminal and would wrap a narrow one.

    A chart block is about 55 characters, so four columns need ~229 -- nearly twice
    the width the rest of the report targets. How many fit is a property of the
    terminal, not of the data.
    """
    from jobscope.report import MAX_CHART_COLUMNS, in_columns
    blocks = [["x" * 55] for _ in range(4)]
    assert len(in_columns(blocks, available=80)) == 4        # one per line
    assert len(in_columns(blocks, available=132)) == 2       # two rows of two
    assert len(in_columns(blocks, available=240)) == 1       # all four abreast
    # And never more than the cap, however wide the terminal claims to be.
    many = [["x" * 10] for _ in range(12)]
    widest = in_columns(many, available=10_000)
    assert len(widest) == len(many) / MAX_CHART_COLUMNS


def test_a_non_terminal_gets_a_fixed_width():
    """Redirected output must not change shape with whatever $COLUMNS happened to be."""
    from jobscope.report import PIPED_CHART_WIDTH, terminal_width
    assert terminal_width(io.StringIO()) == PIPED_CHART_WIDTH


# --- the timeline strip -----------------------------------------------------------

def _strip(values, cells=20, header="GPU%", step=60, options=None):
    """A level_strip of ``values``, bucketed the way the verify block will bucket them."""
    from jobscope import job_eff
    start = 1_000_000
    stamped = [(start + i * step, v) for i, v in enumerate(values) if v is not None]
    end = start + (len(values) - 1) * step
    bands = _thresholds()
    return report.level_strip(
        job_eff.bucket(stamped, start, end, cells),
        job_eff.bucket(stamped, start, end, cells, reduce=max),
        job_eff.cutoff(bands, header), job_eff.full_scale(bands, header),
        options=options, header=header)


def test_a_card_that_never_ran_draws_flat_however_much_it_jittered():
    """The reason this is not plot.braille_spark, which scales to the values' own
    min..max: a card idling between 0.4 and 0.6 has a *range*, and normalising to it
    draws a full-height sawtooth for a card that never did anything."""
    from jobscope.plot import braille_spark
    jitter = [0.4, 0.6] * 60
    assert set(_strip(jitter)) == {"."}
    assert len(set(braille_spark(jitter, 20)[0])) >= 1
    assert braille_spark(jitter, 20)[0] != _strip(jitter)


def test_a_fixed_axis_keeps_two_flat_series_apart():
    """Flat is not one character: the height still says how hard it worked."""
    assert set(_strip([0.5] * 120)) == {"."}
    assert set(_strip([50.0] * 120)) == {"▅"}
    assert set(_strip([95.0] * 120)) == {"█"}


def test_a_cell_holding_one_busy_sample_is_not_drawn_idle():
    """A three-minute cell of 100, 0, 0 and one of 0, 0, 0 have nearly the same mean.
    Telling them apart is the entire reason for the view, so the idle rung is reserved
    for cells where nothing cleared the cutoff."""
    assert "." not in _strip([100, 0, 0] * 40, cells=40)
    assert set(_strip([0, 0, 0] * 40, cells=40)) == {"."}


def test_a_bucket_nothing_was_measured_in_draws_as_a_gap():
    """Not as a zero: an exporter outage and an idle card are different events."""
    from jobscope import job_eff
    strip = report.level_strip([1.0, None, 1.0], [1.0, None, 1.0], 2.0, 100.0)
    assert strip == "%s %s" % (report.STRIP_IDLE, report.STRIP_IDLE)
    assert job_eff.bucket([], 0, 60, 3) == [None, None, None]


def test_the_strip_is_one_character_per_bucket():
    for cells in (1, 7, 20, 103):
        assert len(_strip([50.0] * 120, cells=cells)) == cells


def test_a_strip_is_never_drawn_finer_than_the_scrape_interval():
    """At 116 columns a 40-minute job would get 100 cells for 40 samples, and 60 of them
    would be gaps that are not gaps -- an outage drawn where the data is complete."""
    assert report.strip_cells(116, 10, span=2400, step=60) == 41
    # A long job is bounded by the room instead.
    assert report.strip_cells(116, 10, span=86400, step=60) == 104
    # And never narrower than a strip that could say anything.
    assert report.strip_cells(30, 10, span=86400, step=60) == report.MIN_STRIP_CELLS


def test_the_axis_labels_sit_under_their_own_cells():
    from jobscope import job_eff
    edges = job_eff.bucket_edges(1_000_000, 1_000_000 + 240 * 60, 80)
    rule, labels = report.strip_axis(edges, indent=2)
    assert len(rule) == 80 + 2
    # A tick every ten cells at least, so a five-character HH:MM cannot collide.
    ticks = [i for i, ch in enumerate(rule) if ch == "┬"]
    assert ticks and all(b - a >= 10 for a, b in zip(ticks, ticks[1:]))
    for i in ticks:
        assert labels[i:i + 5].strip(), "no label under the tick at %d" % i


def test_the_strip_carries_no_escapes_without_colour():
    plain = _strip([50.0] * 120, options=RenderOptions(thresholds=_thresholds()))
    assert "\033" not in plain


def test_tinting_the_strip_does_not_change_what_it_measures():
    """Every width in this file is taken after _ESC_RE strips the escapes, so a run-length
    tinted strip has to measure the same as an untinted one."""
    options = RenderOptions(thresholds=_thresholds(), color=True)
    painted = _strip(([90.0] * 60) + ([0.5] * 60), options=options)
    plain = _strip(([90.0] * 60) + ([0.5] * 60))
    assert "\033" in painted
    assert report._ESC_RE.sub("", painted) == plain


def test_in_columns_pads_a_short_block():
    """A metric absent from one group leaves its block a line short of its neighbour."""
    from jobscope.report import in_columns
    out = in_columns([["a1", "a2", "a3"], ["b1"]], columns=2, gap=1)
    assert out == ["a1 b1", "a2", "a3"]


def test_in_columns_measures_width_without_the_escapes():
    """Padding by raw length would push the right column out by the SGR bytes."""
    from jobscope.report import in_columns
    colored = report._SGR["red"] + "ab" + _RESET_T
    out = in_columns([[colored], ["cd"]], columns=2, gap=1)
    # "ab" is two visible characters, so the neighbour starts three columns in.
    assert _ESC.sub("", out[0]) == "ab cd"


def test_the_per_gpu_charts_are_two_columns():
    """Four nodes read as two rows of two, not thirty-two stacked lines."""
    charts, text = _charts({"1": _multinode_job()})
    # Only the chart region: the table above it also mentions the node names.
    chart = text[text.index("Efficiency by"):].splitlines()
    heading = [ln for ln in chart if "nodeA" in ln][0]
    assert "nodeB" in heading                      # both headings share a line
    bar = [ln for ln in chart if ln.count("GPU%") == 2][0]
    assert bar.count("%") >= 3                     # two bars, two percentages


def test_a_single_group_is_not_columnised():
    records = {"1": _gpu_job("1", {"0": 90.0})}
    _c, text = _charts(records)
    chart = text[text.index("Efficiency by"):]
    for line in chart.splitlines():
        assert line.count("GPU%") <= 1             # nothing to pair it with


def test_per_gpu_charts_one_group_per_node():
    charts, text = _charts({"1": _multinode_job()})
    assert "Efficiency by node" in text
    assert list(charts) == ["nodeA", "nodeB"]       # in row order


def test_a_nodes_value_is_the_mean_of_its_gpus():
    """Within a node the GPUs are equal, so the mean is the pooled figure."""
    charts, _text = _charts({"1": _multinode_job()})
    assert charts["nodeA"]["GPU%"] == 85            # mean(90, 80)
    assert charts["nodeB"]["GPU%"] == 15            # mean(10, 20)


def test_per_node_cpu_survives_the_averaging_unchanged():
    """CPU% is a node figure repeated on each GPU row, so averaging must not move it."""
    charts, _text = _charts({"1": _multinode_job()})
    # nodeA used 200 cpu-seconds of 100s x 4 cores; nodeB half that.
    assert charts["nodeA"]["CPU%"] == 50 and charts["nodeB"]["CPU%"] == 25


def test_one_node_charts_each_gpu_instead():
    """With a single node the card is the only thing left that distinguishes rows."""
    records = {"1": _gpu_job("1", {"0": 90.0, "1": 10.0})}
    charts, text = _charts(records)
    assert "Efficiency by GPU on n1" in text
    assert list(charts) == ["GPU 0", "GPU 1"]
    assert charts["GPU 0"]["GPU%"] == 90 and charts["GPU 1"]["GPU%"] == 10


def test_nodename_filters_the_rows_and_switches_to_per_gpu():
    charts, text = _charts({"1": _multinode_job()}, nodename="nodeB")
    assert "nodeA" not in text                      # rows dropped entirely
    assert "Efficiency by GPU on nodeB" in text
    assert list(charts) == ["GPU 0", "GPU 1"]


def test_an_unmatched_nodename_names_the_nodes_that_were_there():
    """A silent empty report would read as an idle node rather than a typo."""
    with pytest.raises(JobscopeError, match="nodeA, nodeB"):
        _charts({"1": _multinode_job()}, nodename="typo")


def test_the_chart_metrics_follow_the_view():
    charts, _text = _charts({"1": _multinode_job()}, view="cpu")
    assert list(charts["nodeA"]) == ["CPU%"]        # --cpu narrows it too


def test_power_gets_no_bar_in_the_per_gpu_chart():
    records = {"1": _gpu_job("1", {"0": 90.0})}
    options = RenderOptions(view="all", show_dcgm=True, header=True,
                            thresholds=_thresholds())
    text = _render(detail, ["1"], records, {"1": ({}, {("n1", "0"): {
        "SM_ACT%": 40.0, "POWER_W": 300.0}})}, CTX, options)
    chart = text[text.index("Efficiency by"):]
    assert "SM_ACT%" in chart and "POWER_W" not in chart


def test_no_per_gpu_charts_in_csv_or_under_no_plot():
    records = {"1": _multinode_job()}
    for kw in ({"csv": True}, {"plot_avgeff": False}):
        _charts_out, text = _charts(records, **kw)
        assert "\u2588" not in text and "Efficiency" not in text


def test_the_per_gpu_charts_are_tinted_and_alignment_holds():
    # One idle card and one busy one, so both ends of the scale appear.
    records = {"1": _gpu_job("1", {"0": 5.0, "1": 95.0})}
    _c, colored = _charts(records, color=True)
    assert report._SGR["red"] in colored and report._SGR["green"] in colored
    _c2, plain = _charts(records, color=False)
    assert _ESC.sub("", colored) == plain


def test_per_gpu_grades_its_cells_like_the_per_job_table():
    """One band helper serves both, so a GPU cannot be green in one and red in the
    other. The per-GPU rows were the only untinted table left."""
    records = {"1": _gpu_job("1", {"0": 5.0, "1": 95.0})}
    text = _render_detail(records)
    assert report._SGR["red"] in text and report._SGR["green"] in text


def test_per_gpu_grades_percent_suffixed_cells():
    """Its cells read "5%" where the summary writes "5".

    A bare float() rejects the trailing sign, which is exactly why these columns were
    silently the only ones left plain.
    """
    from jobscope.report import cell_band
    options = RenderOptions(color=True, thresholds=_thresholds())
    assert cell_band(options, "CPU%", "5%") == "red"
    assert cell_band(options, "CPU%", "50%") == "green"
    assert cell_band(options, "GPU%", 95.0) == "green"       # bare numbers too
    # And nothing that is not a measurement.
    for cell in ("-", "", "holygpu8a10302", "76.1GB/1400GB", "short"):
        assert cell_band(options, "CPU-MEM", cell) == ""


def test_per_gpu_leaves_the_identity_and_paired_columns_plain():
    records = {"1": _gpu_job("1", {"0": 5.0})}
    for line in _render_detail(records).splitlines():
        if not line.startswith("  node"):
            continue
        # The node name, the GPU index and the GB/GB pairs carry no grade.
        assert not line.startswith("  " + report._SGR["red"])
        for plain in ("76.1GB", "node01"):
            assert report._SGR["red"] + plain not in line


def test_color_does_not_shift_the_per_gpu_columns():
    """Same rule as the per-job table: wrap the padded cell, never the value."""
    records = {"1": _gpu_job("1", {"0": 5.0, "1": 95.0})}
    assert _ESC.sub("", _render_detail(records)) == _render_detail(records, color=False)


def test_per_gpu_csv_is_never_tinted():
    records = {"1": _gpu_job("1", {"0": 5.0})}
    options = RenderOptions(view="all", csv=True, header=True, color=True,
                            thresholds=_thresholds())
    assert "\033" not in _render(detail, ["1"], records, {}, CTX, options)


def test_the_header_and_rules_stay_plain():
    records = {"1": _gpu_job("1", {"0": 5.0}), "2": _gpu_job("2", {"0": 95.0})}
    for line in _render_colored(records).splitlines():
        if line.startswith("JOBID") or set(line.strip()) == {"-"}:
            assert "\033[" not in line


def test_the_footers_are_tinted_too():
    """A red pooled row is the fastest read on whether a selection is wasteful."""
    records = {"1": _gpu_job("1", {"0": 2.0}), "2": _gpu_job("2", {"0": 4.0})}
    pooled = [r for r in _render_colored(records).splitlines()
              if r.startswith("Used/")][0]
    assert "\033[31m" in pooled


def test_power_is_tinted_but_runtime_and_ids_are_not(gpu_record):
    """Wattage does have a good and a bad, unlike a runtime or a job id.

    Power used to fall through to the %-metric default of 15 -- 15 *watts* -- so
    every job graded green. With a watt floor it grades like any other metric, and
    a near-idle GPU shows red.
    """
    records = {"1": _gpu_job("1", {"0": 5.0}), "2": _gpu_job("2", {"0": 95.0})}
    dcgm = {"1": ({"POWER_W": 90.0}, {}), "2": ({"POWER_W": 600.0}, {})}
    text = _render_colored(records, dcgm_data=dcgm, show_dcgm=True)
    assert "\033[31m90" in text and "\033[32m600" in text
    # The identity columns stay plain: there is no good or bad job id or runtime.
    for plain in ("00:10:00", "alice", "COMPLETED"):
        assert "\033[31m" + plain not in text and "\033[32m" + plain not in text


def test_csv_is_never_tinted():
    records = {"1": _gpu_job("1", {"0": 5.0}), "2": _gpu_job("2", {"0": 95.0})}
    assert "\033[" not in _render_colored(records, csv=True)


def test_missing_values_are_not_tinted():
    records = {"1": _gpu_job("1", None), "2": _gpu_job("2", {"0": 95.0})}
    row = [r for r in _render_colored(records).splitlines() if r.startswith("1 ")][0]
    assert "\033[" not in row.split("COMPLETED")[1].split("50")[0] or "-" in row


def test_the_table_and_the_charts_grade_alike():
    """One rule, so a job red in a chart is red in the table."""
    from jobscope import plot
    thresholds = _thresholds()

    for header in ("GPU%", "GMEM%", "CPU%", "SM_ACT%"):
        for value in (0, 10, 24, 25, 40, 49, 50, 99):
            assert (thresholds.grade(header, value) or "white") == \
                plot.grade(header, value, thresholds)


class _TwoNodeTimeseriesClient(_TimeseriesClient):
    """Two GPUs on two nodes, so --nodename has something to choose between."""

    def query(self, query, at, timeout=None):
        return [{"metric": {"uuid": "U0", "host": "node01:9400", "minor_number": "0"}},
                {"metric": {"uuid": "U1", "host": "node02:9400", "minor_number": "1"}}]

    def query_range(self, query, start, end, step, timeout=None):
        if "DCGM_FI_PROF_SM_ACTIVE" not in query:
            return []
        return [{"metric": {"UUID": "U0"}, "values": [[1000, "0.8"]]},
                {"metric": {"UUID": "U1"}, "values": [[1000, "0.4"]]}]


def _ts_nodes(nodename, gpu_record):
    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True,
                            nodename=nodename)
    text = _render_ts("dcgm", ["100"], {"100": gpu_record}, _TwoNodeTimeseriesClient(),
                      options, specs=gpu_catalog().default_specs)
    _cols, rows = plot.parse_csv(io.StringIO(text))
    return [r["NODE"] for r in rows]


def test_the_timeseries_takes_the_nodename_filter(gpu_record):
    assert _ts_nodes(None, gpu_record) == ["node01", "node02"]
    assert _ts_nodes("node01", gpu_record) == ["node01"]


def test_an_unmatched_nodename_fails_the_timeseries(gpu_record):
    with pytest.raises(JobscopeError) as exc:
        _ts_nodes("node99", gpu_record)
    assert "node01" in str(exc.value) and "node02" in str(exc.value)


def test_a_failed_timeseries_filter_writes_no_header(gpu_record):
    """A header with no rows under it reads as an idle node, and breaks the plot."""
    out = io.StringIO()
    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True,
                            nodename="node99")
    from jobscope import timeseries as ts
    match = ts.UnitFilter(options.nodename)
    # Rendering itself must not raise -- it simply gets nothing to write. The error
    # comes afterwards, from the filter that matched no node, which is what leaves
    # the header unwritten rather than written-then-regretted.
    report.dcgm_timeseries(
        ts.finished_gpu(["100"], {"100": gpu_record}, gpu_catalog().default_specs,
                        _TwoNodeTimeseriesClient(), None, match=match),
        gpu_catalog().default_specs, options, out=out)
    assert out.getvalue() == ""
    with pytest.raises(JobscopeError):
        match.check()


def test_the_timeseries_header_is_unchanged_by_the_filter(gpu_record):
    """`jobscope plot` keys on this schema, so filtering rows must not touch it."""
    def header(nodename):
        options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True,
                                nodename=nodename)
        text = _render_ts("dcgm", ["100"], {"100": gpu_record},
                          _TwoNodeTimeseriesClient(), options, specs=gpu_catalog().default_specs)
        return text.splitlines()[0]

    assert header("node01") == header(None)


# --- --ts --stats: summarizing the window -----------------------------------

def _ts_stats(rows, metrics=("GPU%", "POWER_W"), level="gpu", **kw):
    out = io.StringIO()
    report.timeseries_stats(rows, list(metrics),
                            RenderOptions(view="all", header=True, **kw),
                            out=out, level=level)
    return out.getvalue()


def _sample(node, gpu, **values):
    row = {"NODE": node, "GPU": gpu}
    row.update({k: str(v) for k, v in values.items()})
    return row


def test_the_window_summary_is_min_mean_max_last():
    rows = [_sample("n1", "0", **{"GPU%": v, "POWER_W": 100}) for v in (10, 20, 60)]
    line = [ln for ln in _ts_stats(rows).splitlines() if "GPU%" in ln][0].split()
    # NODE:GPU METRIC N MIN MEAN MAX LAST
    assert line[2:] == ["3", "10.0", "30.0", "60.0", "60.0"]


def test_the_summary_matches_the_series_it_summarizes():
    """Computed from the samples --ts already fetched, so it cannot disagree."""
    values = [1.0, 2.5, 99.0, 4.0]
    rows = [_sample("n1", "0", **{"GPU%": v}) for v in values]
    line = [ln for ln in _ts_stats(rows, metrics=["GPU%"]).splitlines() if "GPU%" in ln][0]
    assert "%.1f" % (sum(values) / len(values)) in line


def test_each_gpu_is_summarized_separately_in_gpu_order():
    rows = ([_sample("n1", "2", **{"GPU%": 10})] + [_sample("n1", "0", **{"GPU%": 90})])
    labels = [ln.split()[0] for ln in _ts_stats(rows, metrics=["GPU%"]).splitlines()[1:]]
    assert labels == ["n1:0", "n1:2"]      # by GPU number, not first-seen


def test_blank_samples_do_not_count_toward_the_mean():
    """A metric the exporter did not report is absent, not zero."""
    rows = [_sample("n1", "0", **{"GPU%": 10}), _sample("n1", "0", **{"GPU%": ""}),
            _sample("n1", "0", **{"GPU%": 30})]
    line = [ln for ln in _ts_stats(rows, metrics=["GPU%"]).splitlines() if "GPU%" in ln][0]
    assert line.split()[2:5] == ["2", "10.0", "20.0"]


def test_a_metric_with_no_samples_at_all_is_omitted():
    rows = [_sample("n1", "0", **{"GPU%": 10, "POWER_W": ""})]
    assert "POWER_W" not in _ts_stats(rows)


def test_the_summary_has_a_csv_form():
    rows = [_sample("n1", "0", **{"GPU%": 10}), _sample("n1", "0", **{"GPU%": 30})]
    text = _ts_stats(rows, metrics=["GPU%"], csv=True)
    columns, parsed = [ln.split(",") for ln in text.strip().splitlines()]
    assert columns == ["NODE:GPU"] + list(report.TS_STAT_TAIL)
    assert parsed == ["n1:0", "GPU%", "2", "10.0", "20.0", "30.0", "30.0"]


def test_the_mean_is_tinted_by_its_band():
    """The column read first should say whether the number is a problem."""
    rows = [_sample("n1", "0", **{"GPU%": 2})]
    text = _ts_stats(rows, metrics=["GPU%"], color=True, thresholds=_thresholds())
    assert "\033[31m" in text          # 2% is red


def _two_node_rows():
    """Two nodes, two GPUs each, two samples -- so pooling has something to pool."""
    return [_sample(node, gpu, **{"GPU%": value, "JOBID": "100"})
            for node, gpu, value in (("n1", "0", 10), ("n1", "0", 20),
                                     ("n1", "1", 30), ("n1", "1", 40),
                                     ("n2", "0", 50), ("n2", "0", 60),
                                     ("n2", "1", 70), ("n2", "1", 80))]


def test_per_node_pools_a_jobs_gpus_on_one_host():
    text = _ts_stats(_two_node_rows(), metrics=["GPU%"], level="node")
    rows = {ln.split()[0]: ln.split() for ln in text.splitlines()[1:]}
    # n1 pools 10,20,30,40 -> mean 25 over 4 samples from 2 GPUs.
    assert rows["n1"][1:] == ["2", "GPU%", "4", "10.0", "25.0", "40.0", "40.0"]
    assert rows["n2"][1:] == ["2", "GPU%", "4", "50.0", "65.0", "80.0", "80.0"]


def test_per_job_pools_every_node_and_gpu():
    text = _ts_stats(_two_node_rows(), metrics=["GPU%"], level="job")
    row = text.splitlines()[1].split()
    # JOBID NODES GPUS METRIC N MIN MEAN MAX LAST -- mean of 10..80 is 45.
    assert row == ["100", "2", "4", "GPU%", "8", "10.0", "45.0", "80.0", "80.0"]


def test_each_level_says_what_it_pooled():
    """A node mean of one GPU and of sixteen must not look the same."""
    rows = _two_node_rows()
    assert _ts_stats(rows, metrics=["GPU%"]).splitlines()[0].split()[0] == "NODE:GPU"
    assert _ts_stats(rows, metrics=["GPU%"], level="node").splitlines()[0].split()[:2] \
        == ["NODE", "GPUS"]
    assert _ts_stats(rows, metrics=["GPU%"], level="job").splitlines()[0].split()[:3] \
        == ["JOBID", "NODES", "GPUS"]


def test_pooling_weights_by_samples_not_by_gpu():
    """A card the exporter missed for most of the window is not a full peer.

    Averaging per-GPU means would give the sparse card equal say; pooling the samples
    gives it the say its coverage earned.
    """
    rows = ([_sample("n1", "0", **{"GPU%": 0, "JOBID": "100"})] * 9
            + [_sample("n1", "1", **{"GPU%": 100, "JOBID": "100"})])
    row = _ts_stats(rows, metrics=["GPU%"], level="node").splitlines()[1].split()
    # NODE GPUS METRIC N MIN MEAN MAX LAST
    assert row[3] == "10"        # N: nine samples plus one
    assert row[5] == "10.0"      # MEAN, not the 50.0 a mean-of-means would give


def test_the_jobid_leads_only_when_several_jobs_are_present():
    one = _ts_stats(_two_node_rows(), metrics=["GPU%"], level="node")
    assert one.splitlines()[0].split()[0] == "NODE"
    two = _ts_stats(_two_node_rows()
                    + [_sample("n3", "0", **{"GPU%": 5, "JOBID": "101"})],
                    metrics=["GPU%"], level="node")
    assert two.splitlines()[0].split()[:2] == ["JOBID", "NODE"]


# --- --eff: efficiency categories --------------------------------------

@pytest.mark.parametrize("best,category", [
    (0.0, "wasteful"), (1.9, "wasteful"),
    (2.0, "inefficient"), (10.0, "inefficient"),
    (10.1, "needs improvement"), (20.0, "needs improvement"),
    (20.1, "average"), (40.0, "average"),
    (40.1, "good"), (100.0, "good"),
])
def test_the_category_edges(best, category):
    """The edges are the specification, so they are pinned rather than inferred.

    Not uniform: "below 2%" excludes 2, where every band above it includes its top.
    """
    assert _thresholds().tier("GPU%", best) == category


def test_the_best_metric_is_the_best_band_not_the_biggest_number():
    """With per-metric edges the same reading means different things: 4% is above
    GPU%'s 2% wasteful edge and below CPU%'s 5% one, so magnitude cannot order them
    and classify() has to compare the bands themselves."""
    from jobscope.config import Thresholds
    t = Thresholds(by_metric={"CPU%": {"wasteful": 5.0}})
    # CPU% is the larger number but the worse band; GPU% decides.
    assert report.classify({"GPU%": 4.0, "CPU%": 4.5}, t) == "inefficient"
    # And with nothing to beat it, CPU%'s own band is the verdict.
    assert report.classify({"CPU%": 4.5}, t) == "wasteful"


def test_an_empty_ballot_yields_no_verdict_rather_than_wasteful():
    """This used to return "wasteful", on the reading that no measurement is no
    evidence of work. That is true of a CPU-only job in a GPU sweep, whose GPU
    columns are *not applicable*, and false of a job whose exporter was down or
    that predates retention, whose columns are merely *unknown*. Both reach
    classify() as an empty dict, so it cannot tell them apart and must not guess --
    the caller knows which it is holding."""
    assert report.classify({}, _thresholds()) is None
def test_a_cpu_only_job_in_a_combined_sweep_is_still_judged_on_cpu():
    """The case the old default was written for has to keep working: the caller
    routes an empty GPU ballot to a CPU%-only classify() rather than relying on
    classify() to invent a verdict."""
    assert report.classify({"CPU%": 0.5}, _thresholds()) == "wasteful"
    assert report.classify({"CPU%": 55.0}, _thresholds()) == "good"


def test_the_category_is_the_best_metric_not_the_worst():
    """A job doing real work on one measure is not idle because others are low.

    The same rule as "every metric is below X", read from the other end.
    """
    t = _thresholds()
    assert report.classify({"GPU%": 0.0, "SM_ACT%": 0.0, "DRAM%": 45.0}, t) == "good"
    assert report.classify({"GPU%": 0.0, "SM_ACT%": 0.0, "DRAM%": 0.0}, t) == "wasteful"


def _with_floor(thresholds, watts):
    """``thresholds`` with POWER_W's floor set, as for_model() leaves it."""
    from jobscope.config import power_floors
    return dataclasses.replace(thresholds, floors=power_floors(watts))


def test_no_floor_reading_means_no_cap():
    """A floor needs both halves: a configured floor and a measured value."""
    t = _thresholds()
    assert report.classify({"GPU%": 45.0}, t) == "good"           # no readings at all
    assert report.classify({"GPU%": 45.0}, t, {"POWER_W": None}) == "good"   # not measured
    assert report.classify({"GPU%": 45.0}, dataclasses.replace(t, floors={}),
                           {"POWER_W": 50.0}) == "good"           # no floor configured


@pytest.mark.parametrize("verdict,best", [
    ("good", 45.0), ("average", 25.0), ("needs improvement", 15.0),
])
def test_a_floor_caps_a_healthy_verdict_at_inefficient(verdict, best):
    """A duty-cycle-style metric can read busy while the card draws idle watts, and
    watts are the one signal a duty cycle cannot fake."""
    t = _thresholds()
    assert report.classify({"GPU%": best}, t) == verdict          # uncapped, for contrast
    assert report.classify({"GPU%": best}, _with_floor(t, 100.0),
                           {"POWER_W": 67.0}) == "inefficient"


def test_a_floor_never_upgrades_an_already_worse_verdict():
    """A floor only pushes down; it cannot promote wasteful to inefficient."""
    assert report.classify({"GPU%": 1.0}, _with_floor(_thresholds(), 100.0),
                           {"POWER_W": 67.0}) == "wasteful"


def test_a_reading_at_or_above_the_floor_does_not_cap():
    t = _with_floor(_thresholds(), 100.0)
    assert report.classify({"GPU%": 45.0}, t, {"POWER_W": 100.0}) == "good"
    assert report.classify({"GPU%": 45.0}, t, {"POWER_W": 219.5}) == "good"


def test_any_one_low_floor_is_enough_to_cap():
    """Two floors do not cancel out -- each asserts "below this is idle"."""
    from jobscope.config import Thresholds
    t = dataclasses.replace(_thresholds(),
                            floors={"POWER_W": {"": 100.0}, "SMCLK_MHz": {"": 500.0}})
    assert isinstance(t, Thresholds)
    assert report.classify({"GPU%": 45.0}, t,
                           {"POWER_W": 300.0, "SMCLK_MHz": 200.0}) == "inefficient"
    assert report.classify({"GPU%": 45.0}, t,
                           {"POWER_W": 300.0, "SMCLK_MHz": 900.0}) == "good"


# --- CPU% votes, but cannot vote a job healthy -----------------------------

def test_cpu_votes_but_is_capped_at_inefficient_on_a_gpu_job():
    """This replaces the wasteful-cpu-gpu / wasteful-gpu split. A busy host is not
    evidence the cards were needed, so CPU% lifts a GPU-idle job off `wasteful` but
    no further."""
    t = _thresholds()
    assert report.classify({"GPU%": 0.5, "CPU%": 1.0}, t) == "wasteful"      # idle both
    assert report.classify({"GPU%": 0.5, "CPU%": 36.0}, t) == "inefficient"  # host busy
    assert report.classify({"GPU%": 0.5, "CPU%": 95.0}, t) == "inefficient"  # very busy


def test_cpu_cannot_drag_a_healthy_gpu_job_down_either():
    """The ceiling is a ceiling, not a floor: best-of-N means a low CPU% is simply
    outvoted by a working GPU."""
    t = _thresholds()
    for cpu in (0.0, 1.0, 50.0, 95.0):
        assert report.classify({"GPU%": 45.0, "CPU%": cpu}, t) == "good"
        assert report.classify({"GPU%": 25.0, "CPU%": cpu}, t) == "average"


def test_a_cpu_only_ballot_votes_freely():
    """A CPU-only job has no GPU allocation to justify, so the ceiling lifts -- it is
    the only thing that could carry a verdict at all."""
    t = _thresholds()
    assert report.classify({"CPU%": 50.0}, t) == "good"
    assert report.classify({"CPU%": 1.0}, t) == "wasteful"


def test_the_ceiling_binds_on_the_columns_not_the_readings():
    """A combined series that carries GPU% but has no value for this unit has a gap,
    and a busy host must not fill it -- that is what `columns` distinguishes."""
    t = _thresholds()
    assert report.classify({"CPU%": 50.0}, t, columns=["CPU%"]) == "good"
    assert report.classify({"CPU%": 50.0}, t,
                           columns=["GPU%", "CPU%"]) == "inefficient"


def test_a_floor_still_applies_with_cpu_in_the_ballot():
    assert report.classify({"GPU%": 45.0, "CPU%": 95.0},
                           _with_floor(_thresholds(), 100.0),
                           {"POWER_W": 50.0}) == "inefficient"


def test_gmem_takes_no_part_in_the_verdict():
    """Reserving 80GB and computing nothing is still computing nothing."""
    assert "GMEM%" not in report.classify_metrics(
        ["JOBID", "GPU%", "GMEM%", "SM_ACT%", "POWER_W", "GMEM_GB"])
    assert report.classify_metrics(["GPU%", "GMEM%", "SM_ACT%"]) == ["GPU%", "SM_ACT%"]


def test_host_mem_takes_no_part_in_the_verdict_either():
    """Reserving host RAM and not using it is the same non-argument as GMEM%."""
    assert "MEM%" not in report.classify_metrics(["JOBID", "CPU%", "MEM%"])
    assert report.classify_metrics(["CPU%", "MEM%"]) == ["CPU%"]


def _ts_eff_rows(jobs):
    """`jobs` is {jobid: {user, metrics...}} -> two samples each."""
    rows = []
    for jobid, spec in jobs.items():
        for _ in range(2):
            row = {"JOBID": jobid, "USER": spec.get("USER", "u"), "NODE": "n1", "GPU": "0"}
            row.update({k: str(v) for k, v in spec.items() if k != "USER"})
            rows.append(row)
    return rows


def _ts_eff(jobs, columns=("GPU%", "SM_ACT%", "GMEM%", "POWER_W"), **kw):
    out = io.StringIO()
    show_all = kw.pop("show_all", False)
    report.timeseries_eff(
        _ts_eff_rows(jobs), list(columns),
        RenderOptions(view="all", header=True, thresholds=_thresholds(), **kw),
        out=out, level="job", show_all=show_all)
    return out.getvalue()


def test_the_report_names_the_metrics_it_judged_on():
    """--dcgm widens the set, so 'best of' means something different per run."""
    text = _ts_eff({"1": {"GPU%": 50, "SM_ACT%": 40, "GMEM%": 90, "POWER_W": 300}})
    assert "by best of GPU%, SM_ACT%" in text and "GMEM%" not in text.splitlines()[0]


def test_good_collapses_unless_asked_for():
    """On a healthy partition it is most of the output and none of the point."""
    jobs = {"1": {"GPU%": 90, "SM_ACT%": 80, "GMEM%": 50, "POWER_W": 400}}
    assert "--eff all" in _ts_eff(jobs)
    assert "1 " in _ts_eff(jobs, show_all=True).split("good")[1]


def test_the_categories_are_listed_worst_first():
    jobs = {"1": {"GPU%": 90, "SM_ACT%": 90, "GMEM%": 1, "POWER_W": 400},
            "2": {"GPU%": 0.5, "SM_ACT%": 0.1, "GMEM%": 1, "POWER_W": 70},
            "3": {"GPU%": 15, "SM_ACT%": 12, "GMEM%": 1, "POWER_W": 300}}
    text = _ts_eff(jobs)
    order = [ln.strip().split(" (")[0] for ln in text.splitlines()
             if ln.startswith("  ") and not ln.startswith("    ") and ln.endswith("jobs")]
    assert order == ["wasteful", "needs improvement", "good"]


def test_every_tier_heading_enumerates_its_own_metrics_and_range():
    """Not just the worst tier -- "needs improvement"/"good" are as
    self-explanatory as "wasteful", each stating the metrics judged and range."""
    jobs = {"1": {"GPU%": 90, "SM_ACT%": 80, "GMEM%": 1, "POWER_W": 400},
            "2": {"GPU%": 0.5, "SM_ACT%": 0.1, "GMEM%": 1, "POWER_W": 70},
            "3": {"GPU%": 15, "SM_ACT%": 12, "GMEM%": 1, "POWER_W": 300}}
    text = _ts_eff(jobs, show_all=True)
    assert "wasteful (GPU% <2%, SM_ACT% <2%)" in text
    assert "needs improvement (best of GPU%, SM_ACT%: 10-20%)" in text
    assert "good (best of GPU%, SM_ACT%: >40%)" in text


def test_cpu_in_a_combined_series_lifts_a_gpu_idle_unit_off_wasteful():
    """This is what replaced the wasteful-cpu-gpu / wasteful-gpu split. CPU% votes,
    so a GPU-idle unit with a busy host separates from one idle on both -- as a
    different *band* rather than a different flavour of the same one."""
    jobs = {"1": {"GPU%": 90, "CPU%": 80},           # good
            "2": {"GPU%": 0.5, "CPU%": 1.0},         # wasteful: idle on both
            "3": {"GPU%": 0.5, "CPU%": 90.0},        # inefficient: host busy, cards idle
            "4": {"GPU%": 15, "CPU%": 1.0}}          # needs improvement
    text = _ts_eff(jobs, columns=("GPU%", "CPU%"))
    order = [ln.strip().split(" (")[0] for ln in text.splitlines()
             if ln.startswith("  ") and not ln.startswith("    ") and ln.endswith("jobs")]
    assert order == ["wasteful", "inefficient", "needs improvement", "good"]
    # CPU% now votes, and the ceiling that keeps it from voting `good` is named.
    header = text.splitlines()[0]
    assert "by best of GPU%, CPU%" in header
    assert "CPU% can vote no higher than inefficient" in header
def test_a_combined_series_ranks_by_gpu_percent_not_cpu():
    """Within one category, jobs sort by the GPU metric, ascending -- not by CPU%
    and not by jobid (a job with a "later" id but lower GPU% still lists first).
    Both jobs share the same (busy) CPU%, so both land in "wasteful-gpu" -- the
    same category -- which is what lets this isolate the ranking key from the
    category split tested above."""
    jobs = {"9": {"GPU%": 1.5, "CPU%": 90.0}, "1": {"GPU%": 0.2, "CPU%": 90.0}}
    text = _ts_eff(jobs, columns=("GPU%", "CPU%"), show_all=True)
    rows = [ln for ln in text.splitlines() if ln.strip().startswith(("9", "1"))]
    assert [r.split()[0] for r in rows] == ["1", "9"]     # 0.2 before 1.5


def test_a_combined_series_shows_cpu_percent_but_it_does_not_vote():
    jobs = {"1": {"GPU%": 50, "CPU%": 3}}
    text = _ts_eff(jobs, columns=("GPU%", "CPU%"), show_all=True)
    assert "good" in text and "CPU%" in text     # displayed
    assert "by best of GPU%" in text             # but not part of the vote


def test_a_cpu_only_series_still_uses_the_plain_categories():
    """CPU% with no other GPU metric present stays on the ordinary path -- there
    is nothing for it to be "combined" with."""
    jobs = {"1": {"CPU%": 0.5}}
    text = _ts_eff(jobs, columns=("CPU%",))
    assert "wasteful-cpu-gpu" not in text and "wasteful-gpu" not in text
    order = [ln.strip().split(" (")[0] for ln in text.splitlines()
             if ln.startswith("  ") and not ln.startswith("    ") and ln.endswith("jobs")]
    assert order == ["wasteful"]


def test_the_csv_is_jobid_user_metrics_label():
    """The shape asked for: plain values, the label last."""
    jobs = {"1": {"GPU%": 0.4, "SM_ACT%": 0.0, "GMEM%": 90, "POWER_W": 73,
                  "USER": "alice"}}
    text = _ts_eff(jobs, csv=True)
    header, row = [ln.split(",") for ln in text.strip().splitlines()]
    assert header == ["JOBID", "USER", "GPU%", "SM_ACT%", "GMEM%", "POWER_W", "LABEL"]
    assert row == ["1", "alice", "0.4", "0.0", "90.0", "73.0", "wasteful"]


def test_the_csv_reports_metrics_the_verdict_did_not_use():
    """GMEM% and POWER_W do not vote, but a row you will sort or join on should
    still carry what was measured."""
    jobs = {"1": {"GPU%": 50, "SM_ACT%": 40, "GMEM%": 88, "POWER_W": 300}}
    header = _ts_eff(jobs, csv=True).splitlines()[0].split(",")
    assert "GMEM%" in header and "POWER_W" in header
    assert "GMEM%" not in report.classify_metrics(list(header))


def test_a_series_with_no_percentages_cannot_be_classified():
    with pytest.raises(JobscopeError, match="no %-metrics"):
        _ts_eff({"1": {"POWER_W": 300}}, columns=("POWER_W",))


# --- the POWER_W floor follows the card -------------------------------------

_FLOORS = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": 330,
           "NVIDIA H100 80GB HBM3": 130,
           "Tesla V100-PCIE-32GB": 45}


def _per_model_thresholds():
    from jobscope.config import Thresholds, power_floors
    return Thresholds(floors=power_floors(100, _FLOORS))


def _per_model_options(**kw):
    return RenderOptions(view="all", color=True, thresholds=_per_model_thresholds(), **kw)


RTX = "NVIDIA RTX PRO 6000 Blackwell Server Edition"


@pytest.mark.parametrize("model,watts,band", [
    # An idle RTX (165 W) draws more than a working V100 (60 W): the same reading
    # means opposite things, which is the whole reason the floor is per card.
    (RTX, 165, "red"),                          # under its 330 W floor
    (RTX, 450, "green"),                        # over it -- and no yellow to fall in
    (RTX, 700, "green"),
    ("NVIDIA H100 80GB HBM3", 118, "red"),      # p90 idle for this card, floor 130
    ("NVIDIA H100 80GB HBM3", 300, "green"),
    ("Tesla V100-PCIE-32GB", 30, "red"),        # under its 45 W floor
    ("Tesla V100-PCIE-32GB", 60, "green"),      # busy, and under every other floor
])
def test_power_is_graded_against_its_own_card(model, watts, band):
    assert report.cell_band(_per_model_options(), "POWER_W", watts, model) == band


def test_eff_caps_using_the_per_model_floor():
    """--eff's cap follows the card, the same as the table views' POWER_W cell."""
    rows = [
        {"JOBID": "1", "USER": "alice", "NODE": "n1", "GPU": "0", "MODEL": RTX,
         "GPU%": "45", "POWER_W": "165"},                              # idle for an RTX
        {"JOBID": "2", "USER": "bob", "NODE": "n2", "GPU": "0",
         "MODEL": "Tesla V100-PCIE-32GB", "GPU%": "45", "POWER_W": "165"},  # busy for a V100
    ]
    out = io.StringIO()
    report.timeseries_eff(
        rows, ["JOBID", "USER", "NODE", "GPU", "MODEL", "GPU%", "POWER_W"],
        _per_model_options(csv=True), out=out, level="job", show_all=True)
    verdicts = {row.split(",")[0]: row.split(",")[-1]
                for row in out.getvalue().strip().splitlines()[1:]}
    assert verdicts["1"] == "inefficient"    # 165 W < 330 W RTX floor
    assert verdicts["2"] == "good"           # 165 W >= 45 W V100 floor -- uncapped


def test_only_a_sustained_low_mean_caps_not_a_momentary_dip():
    """Job 36643729's own numbers: mean 219.5 W (well above the 100 W floor), min 67 W.

    Gating on the mean (not the min) is what keeps a single low sample from capping an
    otherwise-healthy job; only power that stays low across the window should.
    """
    brief_dip = [{"JOBID": "1", "USER": "u", "NODE": "n1", "GPU": "0",
                 "GPU%": "33", "POWER_W": p} for p in ("67", "220", "220", "220", "220")]
    sustained_idle = [{"JOBID": "2", "USER": "u", "NODE": "n1", "GPU": "0",
                       "GPU%": "33", "POWER_W": p} for p in ("67", "70", "68", "72", "69")]
    out = io.StringIO()
    report.timeseries_eff(
        brief_dip + sustained_idle, ["JOBID", "USER", "NODE", "GPU", "GPU%", "POWER_W"],
        RenderOptions(view="all", csv=True, thresholds=_thresholds()),
        out=out, level="job", show_all=True)
    verdicts = {row.split(",")[0]: row.split(",")[-1]
                for row in out.getvalue().strip().splitlines()[1:]}
    assert verdicts["1"] == "average"        # mean 189.4 W, above the 100 W floor
    assert verdicts["2"] == "inefficient"    # mean 69.2 W, sustained below the floor


def test_the_same_reading_bands_differently_per_card():
    """165 W: idle on an RTX, working on a V100. The reason for the whole change."""
    options = _per_model_options()
    assert report.cell_band(options, "POWER_W", 165, RTX) == "red"
    assert report.cell_band(options, "POWER_W", 165, "Tesla V100-PCIE-32GB") == "green"


def test_an_unknown_model_uses_the_global_floor():
    options = _per_model_options()
    assert report.cell_band(options, "POWER_W", 118, "") == "green"      # 100 W global
    assert report.cell_band(options, "POWER_W", 118, "NVIDIA H100 80GB HBM3") == "red"


def test_only_power_is_judged_per_model():
    """Percentages mean the same thing on every card."""
    options = _per_model_options()
    for model in ("Tesla V100-PCIE-32GB", "NVIDIA H100 80GB HBM3", ""):
        assert report.cell_band(options, "GPU%", 50, model) == "green"
        assert report.cell_band(options, "GPU%", 5, model) == "red"


def test_a_jobs_model_is_its_cards_when_they_agree():
    same = {("n1", "0"): {report.MODEL_KEY: "NVIDIA A40"},
            ("n1", "1"): {report.MODEL_KEY: "NVIDIA A40"}}
    assert report.job_model(same) == "NVIDIA A40"


def test_a_job_spanning_models_falls_back_to_the_global_floor():
    """No honest single answer -- the highest over-flags, the lowest under-flags."""
    mixed = {("n1", "0"): {report.MODEL_KEY: "NVIDIA A40"},
             ("n2", "0"): {report.MODEL_KEY: "Tesla V100-PCIE-32GB"}}
    assert report.job_model(mixed) == ""
    assert report.job_model({("n1", "0"): {}}) == ""            # nothing reported


def test_the_power_tally_judges_each_job_on_its_own_hardware():
    """One selection spans hardware, so the floor cannot live on the tally."""
    tally = report.EfficiencyTally("POWER_W", _per_model_thresholds(),
                                   "GPU-hours", "h", 3600.0, absolute=True,
                                   value_unit="W")
    # 165 W: idle for an RTX, busy for a V100. Same reading, opposite verdicts.
    tally.add("1", "alice", 165.0, 3600.0, model=RTX)
    tally.add("2", "bob", 165.0, 3600.0, model="Tesla V100-PCIE-32GB")
    assert tally.bands["red"][0] == 1 and tally.bands["green"][0] == 1
    assert tally.waste_of(165.0, 3600.0, "Tesla V100-PCIE-32GB") == 0.0
    assert tally.waste_of(165.0, 3600.0, RTX) == 3600.0


def test_every_power_grading_path_uses_the_same_floor():
    """The cell, the tally band and the waste ledger must agree on one reading.

    They did not: the model was an optional argument on four methods and three call
    sites omitted it, so 165 W on an RTX read red in the tally row and green in the
    cell above it. Binding the model into the Thresholds is what makes omission
    impossible; this pins that they cannot drift apart again.
    """
    t = _per_model_thresholds()
    options = RenderOptions(view="all", color=True, thresholds=t)
    tally = report.EfficiencyTally("POWER_W", t, "GPU-hours", "h", 3600.0,
                                   absolute=True, value_unit="W")
    for model, expected in ((RTX, "red"), ("Tesla V100-PCIE-32GB", "green"), ("", "green")):
        assert report.cell_band(options, "POWER_W", 165, model) == expected, model
        assert tally.band_of(165.0, model) == expected, model
        wasted = tally.waste_of(165.0, 1.0, model)
        assert (wasted == 1.0) is (expected == "red"), model
        assert tally.cutoff(model) == t.floor_for(model), model


def test_for_model_returns_itself_when_nothing_is_configured():
    """The common path allocates nothing."""
    t = _per_model_thresholds()
    assert t.for_model("") is t and t.for_model("NVIDIA A40") is t
    assert t.for_model(RTX) is not t and t.for_model(RTX).power_w == 330


def test_the_ts_eff_rows_are_aligned_columns_under_a_header():
    """Not inline "NAME value" pairs: a jobid, a username and a reading ran together."""
    jobs = {"1": {"GPU%": 0.4, "SM_ACT%": 0.0, "GMEM%": 9, "POWER_W": 73, "USER": "al"},
            "2": {"GPU%": 0.1, "SM_ACT%": 0.0, "GMEM%": 9, "POWER_W": 70,
                  "USER": "a-very-long-username"}}
    lines = [ln for ln in _ts_eff(jobs).splitlines() if ln.startswith("    ")]
    header, *body = lines
    assert header.split() == ["JOBID", "NODES", "GPUS", "USER", "GPU%", "SM_ACT%",
                              "POWER_W"]
    assert "GPU% 0.4" not in "".join(body)          # the metric name is not repeated
    # Every row is the same shape, and the long username does not shift the columns.
    assert len({len(ln.rstrip()) for ln in body}) <= 2
    assert all(len(ln.split()) == 7 for ln in body)


def test_eff_names_the_unit_it_judged():
    """At node level every row used to print the job id, so the nodes were identical."""
    rows = [{"JOBID": "1", "USER": "u", "NODE": n, "GPU": "0", "GPU%": "50"}
            for n in ("nodeA", "nodeB") for _ in range(2)]
    out = io.StringIO()
    report.timeseries_eff(rows, ["GPU%"],
                               RenderOptions(view="all", header=True,
                                             thresholds=_thresholds()),
                               out=out, level="node", show_all=True)
    text = out.getvalue()
    assert "NODE" in text and "nodeA" in text and "nodeB" in text


def test_the_identity_columns_and_their_values_stay_in_step():
    """They were two mirrored branch chains that had to agree positionally."""
    found = {"nodes": {"n1", "n2"}, "gpus": {("n1", "0"), ("n2", "0")}, "user": "u"}
    for level, key in (("job", ("1",)), ("node", ("1", "n1")), ("gpu", ("1", "n1", "0"))):
        for multi in (False, True):
            assert len(report.unit_headers(level, multi)) == \
                len(report.unit_values(level, multi, key, found)), (level, multi)


# --- the palette: colour is decoration, buckets are data ---------------------

def test_a_configured_palette_changes_the_tint():
    from jobscope.config import Palette
    try:
        report.set_palette(Palette(colors={"inefficient": "magenta"}))
        assert report.tint("x", "red") == "\033[35mx\033[0m"      # the bucket follows
        assert report.tint("x", "inefficient") == "\033[35mx\033[0m"   # so does the tier
    finally:
        report.set_palette(Palette())


def test_the_band_keys_and_the_csv_schema_do_not_follow_the_palette():
    """The three bucket names are identifiers, not colours: they key
    EfficiencyTally.bands and name columns in the --csv output, so a script reading
    that output must not break because someone recoloured the display."""
    from jobscope.config import Palette
    records = {"1": _gpu_job("1", {"0": 5.0}), "2": _gpu_job("2", {"0": 90.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", csv=True, header=True,
                          thresholds=_thresholds()), out)
    try:
        report.set_palette(Palette(colors={"wasteful": "cyan", "inefficient": "cyan",
                                          "average": "magenta", "good": "magenta"}))
        renderer.add(build_rows(list(records), records, {}))
        renderer.finish()
    finally:
        report.set_palette(Palette())
    text = out.getvalue()
    assert "red=" in text and "yellow=" in text and "green=" in text
    assert "cyan=" not in text and "magenta=" not in text
    # The printed headers are verdicts and the CSV keys are identifiers; neither follows
    # the palette, and the two are deliberately allowed to differ -- a script reading the
    # CSV must not break because the table started saying BAD instead of RED.
    assert report.SummaryRenderer.STAT_HEADERS[2:] == ("JOBS", "GOOD", "OK", "BAD")


def test_an_unknown_role_leaves_the_text_alone():
    """A palette is decoration; losing a colour is not worth losing a report over."""
    assert report.tint("x", "nonsense") == "x"
    assert report.tint("x", "") == "x"


def test_the_long_running_highlight_is_its_own_role():
    """It is not a tier -- it marks an entry whose job ran past long_running -- so a
    site can colour it apart from the bands."""
    from jobscope.config import Palette
    records = {"long": _owned("long", "u1", 4 * 3600, 0.0),
               "brief": _owned("brief", "u1", 60, 0.0)}
    try:
        report.set_palette(Palette(colors={"long_running": "magenta"}))
        block = "".join(_worst_block(records, color=True, thresholds=_thresholds()))
    finally:
        report.set_palette(Palette())
    assert "\033[35mlong:" in block
    assert "\033[35mbrief:" not in block


# --- [defaults] worst_jobs / long_running ------------------------------------

def test_worst_jobs_caps_how_many_each_row_lists():
    records = {n: _owned(n, "u1", i * 3600, 0.0)
               for i, n in enumerate(["a", "b", "c", "d"], start=1)}
    for cap, expected in ((2, 2), (4, 4)):
        block = _worst_block(records, worst_jobs=cap)
        entries = sum(ln.count(":0%:") for ln in block)
        assert entries == expected, (cap, block)


def test_long_running_decides_which_entries_are_highlighted():
    # Two jobs: a single-job selection has no Problem-jobs section to look at.
    records = {"long": _owned("long", "u1", 3600, 0.0),
               "other": _owned("other", "u2", 3600, 1.0)}
    plain = "".join(_worst_block(records, color=True, thresholds=_thresholds(),
                                 long_running=2 * 3600))
    lit = "".join(_worst_block(records, color=True, thresholds=_thresholds(),
                              long_running=600))
    assert report._SGR["long_running"] + "long:" not in plain   # an hour is not long
    assert report._SGR["long_running"] + "long:" in lit         # ten minutes is


# --- [metrics] reaches the summary table ------------------------------------

def test_the_summary_profiling_block_follows_the_configured_metrics():
    from jobscope.dcgm import specs_named
    records = {"1": _gpu_job("1", {"0": 50.0})}
    dcgm = {"1": ({"SM_ACT%": 40.0, "TENSOR%": 5.0, "DRAM%": 9.0, "POWER_W": 200.0}, {})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", show_dcgm=True, header=True), out,
        specs=specs_named(["gpu", "sm_act", "power", "mem", "memtot"]))
    renderer.add(build_rows(list(records), records, dcgm))
    renderer.finish()
    header = next(ln for ln in out.getvalue().splitlines() if ln.startswith("JOBID"))
    assert "SM_ACT%" in header and "POWER_W" in header
    assert "TENSOR%" not in header and "DRAM%" not in header
    # GPU% and GMEM% keep their own fixed columns, from the jobstats summary.
    assert "GPU%" in header and "GMEM%" in header


def test_the_detail_view_keeps_its_fixed_columns_whatever_metrics_say():
    """--all-metrics does not widen --per-gpu: detail_columns() takes no spec list.

    The columns follow the *resolved* headers instead, which is a different question
    from which metrics a view was asked to collect -- see the source-preference tests
    below for the half that does vary.
    """
    records = {"1": _gpu_job("1", {"0": 50.0})}
    out = io.StringIO()
    renderer = report.DetailRenderer(
        CTX, RenderOptions(view="all", show_dcgm=True, header=True))
    renderer.out = out
    renderer.add(build_rows(list(records), records, {}))
    renderer.finish()
    text = out.getvalue()
    for header in ("SM_ACT%", "TENSOR%", "DRAM%", "POWER_W"):
        assert header in text, header


# --- the detail view under a non-default source preference -------------------
#
# DCGM_HEADERS is the default-group columns that still need a query, so promoting an
# exporter *grows* it: 4 under the default order, 5 under dcgm (GPU% stops being
# summary-backed), 7 under nvml (which also takes the GPU-memory pair). The detail
# columns used to be a hand-written list of 11 with row indices 7-10 written down, so
# every profiling value printed one slot to the left of its own header and the last was
# dropped. Reproduced on a real job before this was fixed: POWER_W read 21 where the
# card drew 245 W, and since its floor is 100 W the cell painted red.

_PROFILING_VALUES = {"GPU%": 11.0, "SM_ACT%": 22.0, "TENSOR%": 33.0, "DRAM%": 44.0,
                     "POWER_W": 555.0, "GMEM_GB": 66.0, "GMEM_TOTAL_GB": 77.0}


@pytest.fixture
def preference():
    """Set the GPU source order. Putting it back is conftest.hermetic_config's job."""
    from jobscope import dcgm, source
    return lambda name: dcgm.set_preference(source.parse_preference(name))


def _detail_cells(name, preference, csv=False, color=False):
    """``(columns, [row cells])`` for one preference, as the renderer resolves them."""
    preference(name)
    records = {"1": _gpu_job("1", {"0": 50.0})}
    options = RenderOptions(view="all", show_dcgm=True, header=True, csv=csv,
                            color=color, thresholds=_thresholds())
    out = io.StringIO()
    renderer = report.DetailRenderer(CTX, options, out)
    renderer.add(build_rows(["1"], records, {"1": ({}, {("n1", "0"): dict(_PROFILING_VALUES)})}))
    renderer.finish()
    return renderer.columns, out.getvalue()


@pytest.mark.parametrize("name", ["jobstats", "dcgm", "nvml"])
def test_every_detail_cell_lands_under_its_own_header(name, preference):
    """The regression. Each value is distinct, so a shift of even one column shows.

    Under `dcgm` this used to print GPU%'s 11 in the SM_ACT% column; under `nvml` the
    whole block moved. Read off the resolved columns rather than by splitting the text,
    so the assertion is about the column contract and not about padding.
    """
    columns, text = _detail_cells(name, preference)
    # By header name, not by row index: RUNTIME sits at row cell 7 and prints last, so
    # display position and Column.index are deliberately not the same thing any more.
    cells = _by_header(text)
    block = [c for c in columns if c.group == "dcgm"]
    assert block, "%s: no profiling columns at all" % name
    for col in block:
        expected = report.format_by_header(col.header, _PROFILING_VALUES[col.header])
        assert cells[col.header] == expected, (
            "%s: %s column shows %r, want %r" % (name, col.header, cells[col.header],
                                                 expected))


@pytest.mark.parametrize("name", ["jobstats", "dcgm", "nvml"])
def test_the_profiling_block_is_the_same_four_under_every_preference(name, preference):
    """A consequence of dropping the fixed-position headers, worth stating outright.

    What grows in DCGM_HEADERS as an exporter is promoted is exactly GPU% and the
    GPU-memory pair -- all of which already have a prefix column -- so the generated
    block settles at the same four. That is why the fix leaves default output untouched
    while making the width follow the preference rather than a written-down number.
    """
    columns, _text = _detail_cells(name, preference)
    assert [c.header for c in columns if c.group == "dcgm"] == [
        "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W"]


def _by_header(text):
    """The single data row as {header: cell}, zipped against the printed header.

    Read by name rather than by Column.index: elapsed occupies row cell 7 and prints
    last, so the two orders differ on purpose -- see report.detail_columns.
    """
    lines = text.splitlines()
    header = next(ln for ln in lines if "SM_ACT%" in ln or "GMEM%" in ln)
    row = next(ln for ln in lines if ln.strip().startswith("n1"))
    return dict(zip(header.split(), row.split()))


def _cell(columns, text, header):
    """One rendered cell by column name, from the single data row."""
    return _by_header(text)[header]


def test_the_prefix_keeps_the_stored_value_while_the_summary_wins_it(preference):
    """The default order: the summary serves GPU% and the GPU-memory pair, so the
    prefix is its own numbers -- 50% here, not the queried 11."""
    columns, text = _detail_cells("jobstats", preference)
    assert _cell(columns, text, "GPU%") == "50%"


@pytest.mark.parametrize("name", ["dcgm", "nvml"])
def test_naming_an_exporter_moves_the_fixed_gpu_cell_to_it(name, preference):
    """Defect 2. Naming an exporter takes GPU% out of RESOLVED.from_jobstats, so
    _prefer_stored stops overwriting the queried value -- and the cell has to show it.

    Measured on a real A100 job before this: --gpu-source dcgm reported the summary's
    51 where DCGM had measured 57.8, under a header the Source line said was dcgm's.
    """
    columns, text = _detail_cells(name, preference)
    assert _cell(columns, text, "GPU%") == "11%"


def test_the_memory_pair_and_its_ratio_move_together(preference):
    """GMEM% is used/total, so halves from two sources would be a ratio belonging to
    neither. Under nvml the exporter wins the pair, so all three cells move as one."""
    columns, text = _detail_cells("nvml", preference)
    assert _cell(columns, text, "GPU-MEM") == "66GB/77GB"
    assert _cell(columns, text, "GMEM%") == "85.7%"


def test_the_memory_pair_stays_with_the_summary_under_dcgm(preference):
    """dcgm publishes no total-memory series, so the pair keeps its summary values even
    when dcgm leads -- which is why the two halves are keyed off GMEM_GB together."""
    columns, text = _detail_cells("dcgm", preference)
    assert _cell(columns, text, "GPU-MEM") == "40GB/80GB"
    assert _cell(columns, text, "GMEM%") == "50.0%"


def test_an_empty_query_leaves_the_stored_cell_rather_than_blanking_it(preference):
    """A dash is a worse answer than a number measured slightly differently: if the
    query for a column the exporter won came back empty, keep the summary's."""
    preference("dcgm")
    records = {"1": _gpu_job("1", {"0": 50.0})}
    out = io.StringIO()
    renderer = report.DetailRenderer(
        CTX, RenderOptions(view="all", show_dcgm=True, header=True), out)
    renderer.add(build_rows(["1"], records, {"1": ({}, {("n1", "0"): {"SM_ACT%": 22.0}})}))
    renderer.finish()
    assert _cell(renderer.columns, out.getvalue(), "GPU%") == "50%"


def test_the_overwritten_cell_keeps_the_summary_spelling(preference):
    """A cell that changed shape with --gpu-source would read as a different metric.
    GPU% keeps its trailing sign, and the pair keeps one formatter -- jobstats'."""
    for name in ("jobstats", "dcgm", "nvml"):
        columns, text = _detail_cells(name, preference)
        assert _cell(columns, text, "GPU%").endswith("%")
        assert re.fullmatch(r"[\d.]+GB/[\d.]+GB", _cell(columns, text, "GPU-MEM"))
        assert _cell(columns, text, "GMEM%").endswith("%")


@pytest.mark.parametrize("name", ["dcgm", "nvml"])
def test_an_exporter_value_does_not_overflow_its_column(name, preference):
    """Alignment, which the spelling test alone does not catch.

    A queried duty cycle is a full float where the summary's is a whole percent, so
    "%g" yields "57.8889%" -- eight characters in a seven-wide column, shifting every
    column after it. Every cell is space-padded to its own width, so equal line lengths
    across preferences is exactly the claim "nothing overflowed".
    """
    def lines(pref):
        columns, text = _detail_cells(pref, preference)
        row = next(ln for ln in text.splitlines() if ln.strip().startswith("n1"))
        header = next(ln for ln in text.splitlines() if "SM_ACT%" in ln)
        return len(row), len(header)

    assert lines(name) == lines("jobstats"), (
        "%s: row/header width changed, so a cell overflowed its column" % name)


def test_a_long_float_duty_cycle_is_rounded_to_the_column(preference):
    """The specific value measured on a real job: DCGM's 57.888... must not print its
    full expansion. GPU% carries 0 decimals in the catalog."""
    preference("dcgm")
    records = {"1": _gpu_job("1", {"0": 50.0})}
    out = io.StringIO()
    renderer = report.DetailRenderer(
        CTX, RenderOptions(view="all", show_dcgm=True, header=True), out)
    renderer.add(build_rows(["1"], records,
                 {"1": ({}, {("n1", "0"): {"GPU%": 57.888888888}})}))
    renderer.finish()
    assert _cell(renderer.columns, out.getvalue(), "GPU%") == "58%"


@pytest.mark.parametrize("name", ["jobstats", "dcgm", "nvml"])
def test_the_last_profiling_column_is_not_dropped(name, preference):
    """POWER_W is last in the block, so a short column list silently loses it -- and
    watts are the one idle signal a duty cycle cannot fake."""
    _columns, text = _detail_cells(name, preference)
    assert "POWER_W" in text
    assert "555" in text, "%s: POWER_W's value never rendered" % name


@pytest.mark.parametrize("name", ["jobstats", "dcgm", "nvml"])
def test_no_column_is_offered_twice_or_addressed_past_the_row(name, preference):
    """Two guards on the generated block: the prefix already carries GPU%/GMEM%, and
    an index beyond the row would be an IndexError on a real report."""
    columns, _text = _detail_cells(name, preference)
    headers = [c.header for c in columns]
    assert len(headers) == len(set(headers)), headers
    row = report.detail_row_cells(models.UnitRow("n1", "0"),
                                   {("n1", "0"): dict(_PROFILING_VALUES)})
    assert max(c.index for c in columns) < len(row)


@pytest.mark.parametrize("name", ["jobstats", "dcgm", "nvml"])
def test_the_detail_csv_carries_each_value_under_its_own_name(name, preference):
    """The worse blast radius: a dashboard reads this by column name and cannot see
    that it is ingesting one metric's numbers under another's.

    Asserted by name rather than by width -- the widths always agreed, because header
    and cells were drawn from the same short column list. It was the *values* that came
    from a longer row.
    """
    import csv as csv_module

    _columns, text = _detail_cells(name, preference, csv=True)
    rows = [r for r in csv_module.reader(io.StringIO(text)) if len(r) > 2]
    header = next(r for r in rows if r[0] == "JOBID")
    body = [r for r in rows if r is not header]
    assert body, "%s: no data rows" % name
    for row in body:
        assert len(row) == len(header), "%s: %r vs header %r" % (name, row, header)
        cells = dict(zip(header, row))
        for column, value in _PROFILING_VALUES.items():
            if column in cells and column not in report.FIXED_POSITION_HEADERS:
                assert cells[column] == report.format_by_header(column, value), (
                    "%s: CSV %s = %r" % (name, column, cells[column]))


def test_a_dram_reading_is_never_graded_against_the_power_floor(preference):
    """The consequence that made the shift more than cosmetic.

    With the block one short, DRAM%'s value landed in the POWER_W column and was judged
    against its watt floor -- 44 is below the 100 W default, so a card drawing 555 W
    painted red. The value under POWER_W must be POWER_W's, and it must grade green.
    """
    from jobscope.report import cell_band

    columns, _text = _detail_cells("dcgm", preference, color=True)
    power = next(c for c in columns if c.header == "POWER_W")
    options = RenderOptions(color=True, thresholds=_thresholds())
    cell = report.format_by_header("POWER_W", _PROFILING_VALUES["POWER_W"])
    assert cell_band(options, power.header, cell) == "green"
    assert cell_band(options, "POWER_W",
                     report.format_by_header("DRAM%", 44.0)) == "red"


def test_the_default_preference_renders_exactly_what_it_did_before(preference):
    """The four hand-picked widths, reproduced by the generated ones. Pinned as text
    because "unchanged for everyone who did not pass --gpu-source" is the claim."""
    _columns, text = _detail_cells("jobstats", preference)
    header = next(ln for ln in text.splitlines() if "SM_ACT%" in ln)
    # RUNTIME is new and deliberate (the per-unit granularity spec): the detail CSV
    # carries JOBID and nothing else per job, so a column is the only way a reader gets
    # elapsed. Everything before it is unchanged.
    assert header.strip().endswith(
        "GPU%    GPU-MEM          GMEM%   SM_ACT%  TENSOR%  DRAM%   POWER_W  RUNTIME")


# --- units nothing could be measured for ------------------------------------

def test_a_unit_with_no_gpu_samples_reads_no_data_not_wasteful():
    """A combined series where this job's GPU columns are all blank: the exporter
    died, the node restarted, the window predates retention. Reporting that as
    waste turns a collection gap into an accusation, and it is the reading someone
    acts on."""
    text = _ts_eff({"1": {"CPU%": 50.0}}, columns=("GPU%", "SM_ACT%", "CPU%"),
                     csv=True)
    _, rows = plot.parse_csv(io.StringIO(text))
    assert rows[0]["LABEL"] == report.NO_DATA


def test_no_data_does_not_fall_through_to_the_cpu_ballot():
    """The failure this replaces: with GPU columns blank, a busy CPU would carry
    the job to `good` -- reporting a GPU job with dead telemetry as healthy."""
    text = _ts_eff({"1": {"CPU%": 95.0}}, columns=("GPU%", "SM_ACT%", "CPU%"),
                     csv=True)
    _, rows = plot.parse_csv(io.StringIO(text))
    assert rows[0]["LABEL"] == report.NO_DATA


def test_a_measured_unit_is_unaffected_by_the_no_data_path():
    text = _ts_eff({"1": {"GPU%": 90.0, "SM_ACT%": 80.0, "CPU%": 50.0}},
                     columns=("GPU%", "SM_ACT%", "CPU%"), csv=True)
    _, rows = plot.parse_csv(io.StringIO(text))
    assert rows[0]["LABEL"] == "good"


def test_no_data_units_sort_after_the_real_findings():
    """It is an absence, not a severity. Ranking it first would push the rows
    someone opened the report for off the page."""
    text = _ts_eff({"idle": {"GPU%": 0.5, "SM_ACT%": 0.1, "CPU%": 1.0},
                      "blank": {"CPU%": 50.0}},
                     columns=("GPU%", "SM_ACT%", "CPU%"), csv=True)
    _, rows = plot.parse_csv(io.StringIO(text))
    assert [r["JOBID"] for r in rows] == ["idle", "blank"]
    assert rows[1]["LABEL"] == report.NO_DATA


def test_a_cpu_only_series_is_judged_on_cpu_not_called_no_data():
    """--cpu --ts --eff has CPU% as its only voting column, and must stay on
    the ordinary path: CPU% votes for itself there."""
    text = _ts_eff({"1": {"CPU%": 0.5}}, columns=("CPU%",), csv=True)
    _, rows = plot.parse_csv(io.StringIO(text))
    assert rows[0]["LABEL"] == "wasteful"


def test_report_sections_reorder_and_renumber():
    """[report] sections orders the block, and the numbering follows what printed.

    Numbering is derived, not fixed per section, so a dropped section leaves no gap --
    a missing number reads as something having failed.
    """
    records = {str(i): _gpu_job(str(i), {"0": 1.0}) for i in range(1, 4)}

    default = _finish(records)
    assert re.findall(r"^\d\. .*", default, re.M) == [
        "1. Summary by metric",
        "2. Average efficiency  (filled = used, grey = idle)",
        "3. Problem jobs"]

    flipped = _finish(records, sections=("problems", "metrics"))
    assert re.findall(r"^\d\. .*", flipped, re.M) == [
        "1. Problem jobs",
        "2. Summary by metric"]


def test_no_plot_still_drops_the_bars_when_they_are_configured_on():
    """A flag beats a file: --no-plot wins over [report] sections listing efficiency."""
    records = {str(i): _gpu_job(str(i), {"0": 1.0}) for i in range(1, 4)}
    text = _finish(records, sections=("efficiency", "metrics"), plot_avgeff=False)
    assert "Average efficiency" not in text
    assert "1. Summary by metric" in text


# --- --gpuid: charting or tabulating a subset of a job's cards ---------------

def _four_gpu_record():
    return dataclasses.replace(_gpu_job("100", {str(i): 50.0 for i in range(4)}),
                               start=900, end=1100, duration=200, jobid_raw="100")


class _FourGpuClient(_TimeseriesClient):
    """Four GPUs across two nodes -- 0,1 on node01 and 2,3 on node02."""

    def query(self, query, at, timeout=None):
        return [{"metric": {"uuid": "U%d" % i, "minor_number": str(i),
                            "host": "node0%d:9400" % (1 if i < 2 else 2)}}
                for i in range(4)]

    def query_range(self, query, start, end, step, timeout=None):
        """Answer only for the UUIDs the query's regex actually names.

        Modelling the server this way is the point: it means the assertions below
        check that --gpuid narrowed the *query*, not merely that the demultiplexer
        threw the extra cards away afterwards.
        """
        if "DCGM_FI_PROF_SM_ACTIVE" not in query:
            return []
        asked = re.search(r'UUID=~"\^\(([^)]*)\)\$"', query)
        wanted = set(asked.group(1).split("|")) if asked else set()
        return [{"metric": {"UUID": u}, "values": [[1000, "0.5"]]}
                for u in sorted(wanted)]


def _gpuid_rows(gpu_ids, nodename=None):
    """--ts rows for a two-node, four-GPU job, narrowed by --gpuid."""
    from jobscope import timeseries as ts

    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True,
                            nodename=nodename, gpu_ids=gpu_ids)
    out = io.StringIO()
    match = ts.UnitFilter(nodename, gpu_ids)
    report.dcgm_timeseries(
        ts.finished_gpu(["100"], {"100": _four_gpu_record()}, gpu_catalog().default_specs,
                        _FourGpuClient(), None, match=match),
        gpu_catalog().default_specs, options, out=out)
    match.check()
    _cols, rows = plot.parse_csv(io.StringIO(out.getvalue()))
    return sorted({r["GPU"] for r in rows})


def test_gpuid_narrows_the_series_to_the_named_cards():
    assert _gpuid_rows(()) == ["0", "1", "2", "3"]
    assert _gpuid_rows(("0", "1")) == ["0", "1"]
    assert _gpuid_rows(("2",)) == ["2"]


def test_gpuid_names_every_id_that_matched_nothing():
    """The typo in the middle of a list is the one that is hard to spot, so a
    partially-valid list is an error naming each miss -- not a quietly shorter chart.
    This is the rule plot.run already applies to its own --gpu."""
    with pytest.raises(JobscopeError) as excinfo:
        _gpuid_rows(("0", "9"))
    assert "'9'" in str(excinfo.value) and "0, 1, 2, 3" in str(excinfo.value)

    with pytest.raises(JobscopeError) as excinfo:
        _gpuid_rows(("7", "9"))
    assert "'7', '9'" in str(excinfo.value)


def test_gpuid_composes_with_nodename():
    """Both narrow, and the "it used" list names only the kept node's cards -- GPUs
    from a node the user excluded are not an answer to "which GPUs are there"."""
    assert _gpuid_rows(("0",), nodename="node01") == ["0"]
    with pytest.raises(JobscopeError) as excinfo:
        _gpuid_rows(("2",), nodename="node01")       # 2 and 3 are on node02
    assert "'2'" in str(excinfo.value) and "0, 1" in str(excinfo.value)


# --- --per-node: one row per node, its GPU figures pooled -----------------------
#
# A 16-GPU job over 4 nodes gives 16 rows at gpu level and 4 here, which is where "this
# node is the slow one" lives -- the question _unit_charts' docstring already named while
# the table could not answer it. Verified against that job on the cluster: node
# holygpu8a10302's four cards read 95.7/95.6/95.7/94.8 and the pooled row read 95.5.

def _node_stats(nodes):
    """A summary spanning `nodes`, each {"cpu_seconds", "utils", "used", "total"}."""
    out = {}
    for name, n in nodes.items():
        node = {"total_time": n["cpu_seconds"], "cpus": 4,
                "used_memory": 8 * GIB, "total_memory": 16 * GIB}
        if n.get("utils"):
            node["gpu_utilization"] = n["utils"]
            node["gpu_used_memory"] = {k: v * GIB for k, v in n["used"].items()}
            node["gpu_total_memory"] = {k: v * GIB for k, v in n["total"].items()}
        out[name] = node
    return {"total_time": 100, "nodes": out}


def _per_node_rows(stats, dcgm_data=None, csv=False, level="node", runtime="01:00:00"):
    """Rendered rows at `level`, as [{header: cell}] read by name."""
    record = JobRecord(jobid="1", state="COMPLETED", name="j", runtime=runtime,
                       nodes="2", gpus=4, stats=stats, start=1000, end=1100,
                       duration=100, jobid_raw="1", cluster="c", user="alice")
    options = RenderOptions(view="all", show_dcgm=bool(dcgm_data), csv=csv, header=True,
                            thresholds=_thresholds())
    out = io.StringIO()
    renderer = report.DetailRenderer(CTX, options, out, level=level)
    renderer.add(build_rows(["1"], {"1": record}, dcgm_data or {}))
    renderer.finish()
    text = out.getvalue()
    if csv:
        import csv as csv_mod
        rows = [r for r in csv_mod.reader(io.StringIO(text)) if len(r) > 2]
        header = next(r for r in rows if r[0] == "JOBID")
        return [dict(zip(header, r)) for r in rows if r is not header]
    lines = text.splitlines()
    head = next(ln for ln in lines if "CPU-MEM" in ln)
    return [dict(zip(head.split(), ln.split()))
            for ln in lines if ln.startswith("  ") and "GB/" in ln and "CPU-MEM" not in ln]


def test_one_row_per_node_with_the_cards_pooled():
    """Two nodes that differ, so a swap or a job-level fallback fails."""
    rows = _per_node_rows(_node_stats({
        "n1": {"cpu_seconds": 200, "utils": {"0": 90, "1": 70},
               "used": {"0": 40, "1": 20}, "total": {"0": 80, "1": 80}},
        "n2": {"cpu_seconds": 100, "utils": {"0": 10, "1": 30},
               "used": {"0": 8, "1": 4}, "total": {"0": 80, "1": 80}}}))
    assert len(rows) == 2
    by_node = {r["NODE"]: r for r in rows}
    assert by_node["n1"]["#GPU"] == "2" and by_node["n2"]["#GPU"] == "2"
    assert by_node["n1"]["GPU%"] == "80%"      # mean of 90 and 70
    assert by_node["n2"]["GPU%"] == "20%"      # mean of 10 and 30
    assert by_node["n1"]["GPU-MEM"] == "60GB/160GB"     # summed, not one card's
    assert by_node["n2"]["GPU-MEM"] == "12GB/160GB"


def test_gmem_is_the_ratio_of_sums_not_the_mean_of_ratios():
    """The two formulas agree whenever the cards have equal capacity, which is why a
    fixture with unequal ones is the only one that can catch getting it backwards:
    (10+70)/(100+80) = 44.4%, where the mean of 10% and 87.5% is 48.8%.
    """
    rows = _per_node_rows(_node_stats({
        "n1": {"cpu_seconds": 100, "utils": {"0": 50, "1": 50},
               "used": {"0": 10, "1": 70}, "total": {"0": 100, "1": 80}}}))
    assert rows[0]["GMEM%"] == "44.4%"
    assert rows[0]["GPU-MEM"] == "80GB/180GB"


def test_a_single_node_job_pools_to_its_own_per_job_figures():
    """The two reductions have to agree when the subset is everything."""
    stats = _node_stats({"n1": {"cpu_seconds": 100, "utils": {"0": 90, "1": 50},
                                "used": {"0": 48, "1": 32},
                                "total": {"0": 80, "1": 80}}})
    node = _per_node_rows(stats)[0]
    assert node["GPU%"] == "70%"                      # conftest's GPU_STATS figures
    assert node["GMEM%"] == "50.0%"
    assert jobstats_metrics(stats).value("GPU%") == 70
    assert jobstats_metrics(stats).value("GMEM%") == 50


def test_a_cpu_only_node_reports_zero_cards_and_dashes():
    rows = _per_node_rows(_node_stats({"n1": {"cpu_seconds": 100}}))
    assert rows[0]["#GPU"] == "0"
    for column in ("GPU%", "GPU-MEM", "GMEM%"):
        assert rows[0][column] == "-", column


def test_elapsed_is_present_at_both_detail_levels():
    stats = _node_stats({"n1": {"cpu_seconds": 100, "utils": {"0": 90},
                                "used": {"0": 40}, "total": {"0": 80}}})
    for level in ("node", "gpu"):
        assert _per_node_rows(stats, level=level)[0]["RUNTIME"] == "01:00:00", level


def test_elapsed_reaches_the_csv_which_carries_nothing_else_per_job():
    stats = _node_stats({"n1": {"cpu_seconds": 100, "utils": {"0": 90},
                                "used": {"0": 40}, "total": {"0": 80}}})
    for level in ("node", "gpu"):
        row = _per_node_rows(stats, csv=True, level=level)[0]
        assert row["RUNTIME"] == "01:00:00", level


def test_elapsed_survives_a_view_with_no_profiling_block():
    """--cpu drops the dcgm block, and RUNTIME's row cell must not move with it -- the
    whole reason it sits at a fixed index instead of after the block."""
    stats = _node_stats({"n1": {"cpu_seconds": 100, "utils": {"0": 90},
                                "used": {"0": 40}, "total": {"0": 80}}})
    record = JobRecord(jobid="1", state="COMPLETED", name="j", runtime="01:00:00",
                       nodes="1", gpus=1, stats=stats, start=1000, end=1100,
                       duration=100, jobid_raw="1", cluster="c", user="alice")
    out = io.StringIO()
    renderer = report.DetailRenderer(
        CTX, RenderOptions(view="cpu", show_dcgm=False, header=True), out, level="node")
    renderer.add(build_rows(["1"], {"1": record}, {}))
    renderer.finish()
    text = out.getvalue()
    assert "RUNTIME" in text and "01:00:00" in text
    assert "SM_ACT%" not in text


def test_the_node_row_uses_the_node_s_pooled_dcgm_figures_not_the_job_s():
    """The gap a mutation found: substituting the job-level figure for the per-node one
    broke nothing, because the pooling tests above feed no DCGM data at all.

    Two nodes with different SM_ACT%, and a job figure equal to neither.
    """
    stats = _node_stats({
        "n1": {"cpu_seconds": 100, "utils": {"0": 90}, "used": {"0": 40},
               "total": {"0": 80}},
        "n2": {"cpu_seconds": 100, "utils": {"0": 10}, "used": {"0": 8},
               "total": {"0": 80}}})
    dcgm_data = {"1": ({"SM_ACT%": 50.0},                       # overall -- neither node
                       {},                                       # per_gpu, unused here
                       {"n1": {"SM_ACT%": 90.0}, "n2": {"SM_ACT%": 10.0}})}
    rows = {r["NODE"]: r for r in _per_node_rows(stats, dcgm_data=dcgm_data)}
    assert rows["n1"]["SM_ACT%"] == "90.0"
    assert rows["n2"]["SM_ACT%"] == "10.0"
    assert "50.0" not in (rows["n1"]["SM_ACT%"], rows["n2"]["SM_ACT%"])


def test_the_gpu_row_still_reads_the_per_gpu_map():
    """The other half of the same lookup: gpu level keys by (node, minor)."""
    stats = _node_stats({"n1": {"cpu_seconds": 100, "utils": {"0": 90, "1": 50},
                                "used": {"0": 40, "1": 20},
                                "total": {"0": 80, "1": 80}}})
    dcgm_data = {"1": ({"SM_ACT%": 99.0},
                       {("n1", "0"): {"SM_ACT%": 11.0}, ("n1", "1"): {"SM_ACT%": 22.0}},
                       {"n1": {"SM_ACT%": 99.0}})}
    rows = _per_node_rows(stats, dcgm_data=dcgm_data, level="gpu")
    assert [r["SM_ACT%"] for r in rows] == ["11.0", "22.0"]


def test_a_single_node_at_node_level_charts_by_node_not_by_card():
    """At node level a row *is* a node, so there is nothing to break down by card -- and
    cell 1 is a count, so the by-GPU branch would label the group "GPU 2"."""
    stats = _node_stats({"n1": {"cpu_seconds": 100, "utils": {"0": 90, "1": 50},
                                "used": {"0": 40, "1": 20},
                                "total": {"0": 80, "1": 80}}})
    record = JobRecord(jobid="1", state="COMPLETED", name="j", runtime="01:00:00",
                       nodes="1", gpus=2, stats=stats, start=1000, end=1100,
                       duration=100, jobid_raw="1", cluster="c", user="alice")

    def charts(level):
        out = io.StringIO()
        r = report.DetailRenderer(
            CTX, RenderOptions(view="all", header=True, plot_avgeff=True,
                               thresholds=_thresholds()), out, level=level)
        r.add(build_rows(["1"], {"1": record}, {}))
        r.finish()
        return out.getvalue()

    assert "GPU 2" not in charts("node")
    assert "by node" in charts("node")
    # The gpu level keeps the by-card breakdown, which is the whole point of it there.
    assert "GPU 0" in charts("gpu")


# --- --verify: the window ladder ------------------------------------------------
#
# Before a scancel, a single accurate number is still the wrong output. Measured on one
# 4-GPU job: the newest scrape read GPU% 0 and SM_ACT% 0.0 -- below the wasteful edge on
# both, an obvious kill -- while the job was computing in bursts and had saturated all
# four cards within the previous half hour. `max` and the trend across rungs are what
# separate that from a job that never ran.

def _series(values, step=60, start=1_000_000, metric="GPU%", node="n1", gpu="0"):
    """`--ts`-shaped rows: one per scrape, oldest first. None leaves a gap."""
    return [{"JOBID": "1", "USER": "alice", "EPOCH": str(start + i * step),
             "NODE": node, "GPU": gpu, "MODEL": "NVIDIA H200", metric: "" if v is None
             else str(v)}
            for i, v in enumerate(values)]


def _verify(rows, metrics=("GPU%",), windows=(("2h", 7200), ("30m", 1800)), window=None,
            detail=True):
    """The ladder is behind --full now; these tests are about the ladder."""
    out = io.StringIO()
    report.verify_report(rows, list(metrics),
                         RenderOptions(thresholds=_thresholds(), window=window,
                                       verify_full=detail),
                         out, windows=windows)
    return out.getvalue()


def _ladder_header(text):
    """The ladder table's header row, wherever in the block it sits."""
    return next(ln for ln in text.splitlines() if ln.startswith("NODE"))


def _row(text, metric="GPU%"):
    """One ladder row as a dict, taken from the ladder body alone.

    Bounded by the blank line that ends the block rather than searched for across the
    whole output: the verdict block names its metrics in prose, and some of those lines
    have the same field count as a row by coincidence.
    """
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("NODE"))
    fields = lines[start].split()
    for line in lines[start + 1:]:
        if not line.strip():
            break
        if not line.startswith("-") and metric in line.split():
            return dict(zip(fields, line.split()))
    raise AssertionError("no %s row in the ladder:\n%s" % (metric, text))


def test_a_bursty_job_is_not_reported_as_idle():
    """The regression this whole view exists for. A series whose newest samples are zero
    but whose peak is 100 must not read flat-idle -- that is the reading that gets a
    working job killed."""
    # 3 hours of alternating work, then half an hour of nothing.
    values = ([100, 0] * 90) + ([0] * 30)
    text = _verify(_series(values))
    got = _row(text)
    assert got["MAX"] == "100.0"
    assert got["SHAPE"] != "flat-idle"
    assert got["SHAPE"] in ("declining", "bursty")


def test_a_series_that_never_ran_is_flat_idle():
    """The only shape that says nothing happened, and the only kill-candidate one."""
    assert _row(_verify(_series([0.5] * 200)))["SHAPE"] == "flat-idle"


def test_a_job_that_stopped_an_hour_ago_reads_declining():
    """What the whole-run mean alone cannot say."""
    values = ([90] * 150) + ([1] * 60)      # good for 2.5h, dead for the last hour
    got = _row(_verify(_series(values)))
    assert got["SHAPE"] == "declining"
    assert float(got["run"]) > float(got["30m"])


def test_a_steadily_busy_job_reads_steady():
    got = _row(_verify(_series([90] * 200)))
    assert got["SHAPE"] == "steady"
    assert got["BELOW"] == "0%"


def test_a_rung_at_least_as_long_as_the_run_is_dropped():
    """Otherwise every job shorter than the widest window gets identical columns."""
    short = _ladder_header(_verify(_series([50] * 20))).split()      # 20 minutes
    assert "2h" not in short and "30m" not in short and "run" in short
    long = _ladder_header(_verify(_series([50] * 300))).split()      # 5 hours
    assert "2h" in long and "30m" in long


def test_slicing_a_rung_equals_a_narrowed_fetch():
    """range_window guarantees a windowed fetch returns the rows a full one would, so the
    ladder can slice one fetch. Asserted rather than assumed."""
    values = [10] * 150 + [90] * 30
    whole = _row(_verify(_series(values)))
    # The last 30 samples fetched on their own, same stamps.
    narrowed = _row(_verify(_series(values)[-30:], windows=()))
    assert whole["30m"] == narrowed["run"]


def test_a_gap_does_not_lengthen_the_idle_stretch():
    """An exporter outage must not read as a wedged job -- the same collapse between
    'not measured' and 'not working' that models.Measure exists to prevent."""
    unbroken = _row(_verify(_series([0] * 10 + [90] * 100)))
    broken = _row(_verify(_series([0] * 4 + [None] * 2 + [0] * 4 + [90] * 100)))
    assert unbroken["IDLEMAX"] == "10m"
    assert broken["IDLEMAX"] == "4m", "a gap has to break the run, not bridge it"


def test_the_gap_count_is_reported_rather_than_absorbed():
    text = _verify(_series([0] * 4 + [None] * 3 + [90] * 100))
    assert "scrape(s) missing" in text


def test_below_is_a_share_of_measured_samples_not_of_elapsed_time():
    text = _verify(_series([0] * 50 + [None] * 50 + [90] * 50))
    # 50 idle of 100 measured, not of 150 elapsed.
    assert _row(text)["BELOW"] == "50%"


def test_power_answers_to_its_per_model_floor_not_a_percentage_edge():
    """Watts have no tier, and idle draw runs 27 W to 165 W across a fleet."""
    rows = _series([120] * 100, metric="POWER_W")
    got = _row(_verify(rows, metrics=("POWER_W",)), metric="POWER_W")
    # 120 W is above the default 100 W floor, so nothing is below it.
    assert got["BELOW"] == "0%"
    low = _row(_verify(_series([80] * 100, metric="POWER_W"), metrics=("POWER_W",)),
               metric="POWER_W")
    assert low["BELOW"] == "100%"


def test_power_has_a_shape_rather_than_always_reading_steady():
    """Thresholds.tier returns "" for a non-percentage, and reading that through
    _VERDICT_ORDER.get(tier, 0) ranked every POWER_W sample the same. Every rank being
    equal made flat-idle, declining and bursty all unreachable, so a card that never rose
    off its 100 W idle floor reported `steady` -- the one word that says it is fine."""
    never_ran = _series([74] * 200, metric="POWER_W")
    assert _row(_verify(never_ran, metrics=("POWER_W",)), "POWER_W")["SHAPE"] == "flat-idle"
    # Drew 500 W for 2.5h, then fell through the floor for the last hour.
    stopped = _series(([500] * 150) + ([80] * 60), metric="POWER_W")
    assert _row(_verify(stopped, metrics=("POWER_W",)), "POWER_W")["SHAPE"] == "declining"
    # Still above the floor throughout: `steady` is the right answer here, and stays it.
    dipped = _series(([500] * 150) + ([120] * 60), metric="POWER_W")
    assert _row(_verify(dipped, metrics=("POWER_W",)), "POWER_W")["SHAPE"] == "steady"


def test_memory_has_no_idle_line_to_be_below():
    """MEM% and GMEM% are percentages, so they used to answer to the wasteful edge like
    any other column -- which made "this job's memory was idle for 3h" printable. A
    capacity reading full while nothing computes is what the `memory` role names."""
    text = _verify(_series([1.0] * 200, metric="MEM%"), metrics=("MEM%",))
    got = _row(text, "MEM%")
    assert (got["BELOW"], got["IDLEMAX"]) == ("-", "-")
    # The graded columns are unaffected -- this is about the role, not about being low.
    assert _row(_verify(_series([1.0] * 200)))["BELOW"] == "100%"


def test_a_dead_exporter_is_counted_as_missing_rather_than_as_a_short_series():
    """The gap count came off each series' own extent, so a series that simply *stopped*
    was complete by its own measure: an exporter that died halfway reported zero missing
    scrapes and its last sample looked like the present."""
    rows = _series([50] * 60, gpu="0") + _series([50] * 120, gpu="1")
    assert "60 scrape(s) missing" in _verify(rows)


def test_a_windowed_fetch_names_the_window_rather_than_the_run():
    """range_window narrows the query, so under --verify 2h the fetch *is* the window.
    The ladder printed a `run` column of 121 samples beside a `2h` column of 120 -- two
    readings of the same data, the first named after a run it did not cover."""
    windowed = _ladder_header(_verify(_series([50] * 121), window=7200)).split()
    assert "2h" in windowed and "run" not in windowed
    assert windowed.count("2h") == 1
    # Without a window the whole run is the first rung, exactly as before.
    assert "run" in _ladder_header(_verify(_series([50] * 121))).split()
    # A window wider than the run got the whole run, so `run` is what that column is.
    assert "run" in _ladder_header(_verify(_series([50] * 20), window=14400)).split()


def test_a_host_metric_is_reported_once_per_node_not_once_per_card():
    """A combined series carries CPU%/MEM% on every GPU row, so grouping the whole lot by
    card gave a 4-GPU job four identical CPU% rows -- and counted one node's host samples
    four times in anything that pools them."""
    rows = []
    for i in range(60):
        for gpu in ("0", "1", "2", "3"):
            rows.append({"JOBID": "1", "USER": "alice", "EPOCH": str(1_000_000 + i * 60),
                         "NODE": "n1", "GPU": gpu, "MODEL": "NVIDIA H200",
                         "GPU%": "50", "CPU%": "7.5"})
    data = report.verify_figures(rows, ["GPU%", "CPU%"],
                                 RenderOptions(thresholds=_thresholds()))
    host = [f for f in data.figures if f.metric == "CPU%"]
    assert len(host) == 1
    assert len(host[0].values) == 60          # the node's scrapes, not 4x them
    assert host[0].unit == ("n1",)            # named by node: it pooled no cards
    assert len([f for f in data.figures if f.metric == "GPU%"]) == 4


def test_each_node_is_measured_at_its_own_scrape_interval():
    """One modal gap over every stamp in the report is the majority node's, and on a job
    whose nodes scrape at different rates every duration on the other node was wrong."""
    minute = [{"JOBID": "1", "USER": "a", "EPOCH": str(1_000_000 + i * 60), "NODE": "n1",
               "GPU": "0", "MODEL": "NVIDIA H200", "GPU%": "1"} for i in range(100)]
    five = [{"JOBID": "1", "USER": "a", "EPOCH": str(1_000_000 + i * 300), "NODE": "n2",
             "GPU": "0", "MODEL": "NVIDIA H200", "GPU%": "1"} for i in range(20)]
    data = report.verify_figures(minute + five, ["GPU%"],
                                 RenderOptions(thresholds=_thresholds()))
    by_node = {f.unit[0]: f for f in data.figures}
    assert by_node["n1:0"].step == 60 and by_node["n2:0"].step == 300
    # 20 samples five minutes apart is an hour and forty, not twenty minutes.
    assert by_node["n2:0"].idlemax == 20 * 300


def test_rows_that_carry_no_value_for_the_metric_say_so_rather_than_raising():
    """stamped_samples keys a group off the row's identity, so rows whose metric column
    is blank leave a group with no stamps at all -- and `max(set())` ran on it. Reachable
    whenever a column is in the series header and empty in every row of it."""
    blank = [dict(row, **{"GPU%": ""}) for row in _series([50] * 50)]
    assert _verify(blank).strip() == "no samples to verify"
    assert _verify(_series([50] * 50), metrics=("NOSUCH%",)).strip() == "no samples to verify"


def test_the_table_names_which_source_served_each_column():
    """--verify is the pre-action check; where the number came from is part of the answer."""
    text = _verify(_series([50] * 100))
    assert "GPU% <-" in text


# --- --verify: the verdict block --------------------------------------------------

def _block(rows, metrics=("GPU%",), window=None, color=False, windows=(("2h", 7200),)):
    """The default --verify output: the verdict block, no ladder."""
    return _verify(rows, metrics, windows=windows, window=window, detail=False)


def test_the_default_output_concludes_rather_than_tabulating():
    text = _block(_series([0.5] * 200))
    assert "Verdict:" in text
    assert "NODE:GPU" not in text          # the ladder is behind --full
    assert "SWING" not in text


def test_verify_full_adds_the_ladder_under_the_verdict():
    text = _verify(_series([0.5] * 200))   # detail=True
    assert "Verdict:" in text and "NODE:GPU" in text and "SWING" in text
    assert text.index("NODE:GPU") < text.index("Verdict:")


def test_the_verdict_is_the_one_the_rest_of_the_report_reaches():
    """job_eff.classify, not a second opinion. Cross-checked against --ts --eff, which
    is the other caller that turns a series into a category."""
    for values, expected in (([85] * 120, "good"), ([0.5] * 120, "wasteful"),
                             ([15] * 120, "needs improvement")):
        rows = _series(values)
        block = _block(rows)
        out = io.StringIO()
        report.timeseries_eff(rows, ["JOBID", "USER", "EPOCH", "NODE", "GPU", "MODEL",
                                     "GPU%"],
                              RenderOptions(thresholds=_thresholds()), out, level="job")
        assert "Verdict: %s" % expected in block
        assert expected in out.getvalue()


def test_a_job_that_stopped_says_when_it_stopped():
    """The question the whole view is for: not just that it is idle, but since when."""
    text = _block(_series(([90] * 150) + ([0.5] * 60)))
    assert "ran, then stopped" in text
    assert "idle at every scrape since -- 1h00m" in text


def test_a_card_that_never_ran_is_named_as_such_and_counted():
    """Pooling hides it: a job using one card of four and one using four badly reach the
    same verdict, and only the first is fixed by asking for fewer GPUs."""
    rows = _series([90] * 120, gpu="0")
    for gpu in ("1", "2", "3"):
        rows += _series([0.2] * 120, gpu=gpu)
    text = _block(rows)
    assert text.count("never ran") == 3
    assert "3 of 4 units never cleared 2" in text


def test_units_idle_at_different_times_were_never_idle_together():
    """The job-level figure is the instants they shared, not the span between the first
    and the last -- which would count an exporter outage as job-wide idleness."""
    rows = (_series(([0.5] * 60) + ([90] * 60), gpu="0")
            + _series(([90] * 60) + ([0.5] * 60), gpu="1"))
    assert "every unit idle at the same scrape for 0s" in _verify(rows, detail=True)


def test_a_window_too_short_to_grade_withholds_the_verdict():
    text = _block(_series([50, 0, 97, 80, 60, 40]))
    assert "No verdict:" in text and "fewer than 30" in text
    assert "Verdict:" not in text.replace("No verdict:", "")
    # What was measured still prints: refusing to state a fact is not a virtue.
    assert "The metrics the verdict was taken over" in text


def test_half_a_window_measured_withholds_the_verdict():
    rows = _series([90] * 100, gpu="0") + _series([90] * 240, gpu="1")
    text = _block(rows)
    assert "No verdict:" in text and "less than half a window" in text


def test_an_unreproducible_mean_qualifies_the_verdict_without_withholding_it():
    """A bursty job with a low mean can still be genuinely wasteful, so this warns
    rather than refuses -- and it warns where the verdict is, not only below it."""
    text = _block(_series([100 if i % 2 else 0 for i in range(151)]))
    assert "Verdict:" in text
    assert "on a mean that is not reproducible" in text
    assert "flapping" in text


def test_the_timeline_shows_where_the_job_collapsed():
    text = _verify(_series(([90] * 100) + ([0.5] * 100)), detail=True)
    strip = next(ln for ln in text.splitlines() if ln.strip().startswith("n1:0")
                 and report.STRIP_IDLE in ln)
    # Working then idle, in that order, on one line.
    assert strip.rstrip().endswith(report.STRIP_IDLE)
    worked = min(strip.index(ch) for ch in report.STRIP_BLOCKS if ch in strip)
    assert worked < strip.index(report.STRIP_IDLE)


def test_the_bands_block_agrees_with_the_verdict_it_sits_under():
    """The bins are Thresholds.tier calls, so the histogram cannot contradict the
    conclusion -- and it collapses to a line when one band holds nearly all of it."""
    busy = _verify(_series([85] * 200), detail=True)
    assert "100% good" in busy and "nothing measured in any other band" in busy
    mixed = _verify(_series(([85] * 100) + ([0.5] * 100)), detail=True)
    assert "Time by band, per metric" in mixed
    # Half the window either side of the cutoff, so both bands carry real time.
    wasteful = next(ln for ln in mixed.splitlines() if "wasteful" in ln)
    good = next(ln for ln in mixed.splitlines() if ln.strip().startswith("good"))
    assert "50%" in wasteful and "1h40m" in wasteful
    assert "50%" in good and "1h40m" in good


def _multi(gpu_values, sm_values, mem_values=None, gpu="0"):
    """Rows carrying several metrics per scrape, as a combined series does."""
    rows = []
    for i, value in enumerate(gpu_values):
        row = {"JOBID": "1", "USER": "alice", "EPOCH": str(1_000_000 + i * 60),
               "NODE": "n1", "GPU": gpu, "MODEL": "NVIDIA H200",
               "GPU%": str(value), "SM_ACT%": str(sm_values[i])}
        if mem_values is not None:
            row["MEM%"] = str(mem_values[i])
        rows.append(row)
    return rows


def test_the_default_block_reports_every_metric_the_verdict_used():
    """The 'Graded by best of ...' line names them, and naming a metric as having voted
    while showing none of its numbers is the gap. Measured on a real job: GPU% 98.9 and
    TENSOR% flat is a card busy by duty cycle and barely computing."""
    rows = _multi([98] * 120, [1.0] * 120)
    text = _verify(rows, metrics=("GPU%", "SM_ACT%"), detail=False)
    assert "The metrics the verdict was taken over" in text
    # The leading metric is in the table too, or it has no representation at all.
    assert next(ln for ln in text.splitlines() if ln.strip().startswith("GPU%"))
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("SM_ACT%")
                and "flat-idle" in ln)
    assert "1.0" in line                       # its min/max/mean, which GPU% alone hides
    # One table and the verdict: every picture is evidence behind it, and waits for --full.
    assert "drawn against" not in text and "Time by band, per metric" not in text
    full = _verify(rows, metrics=("GPU%", "SM_ACT%"), detail=True)
    assert "Time by band, per metric" in full and "SM_ACT%  100% wasteful" in full
    assert full.count("drawn against") == 2


def test_a_metric_that_neither_votes_nor_floors_is_left_to_the_ladder():
    """MEM% has no cutoff and no say, so it has nothing to contribute to a verdict
    block -- it is carried by --full's ladder instead."""
    rows = _multi([98] * 120, [50] * 120, mem_values=[40] * 120)
    data = report.verify_figures(rows, ["GPU%", "SM_ACT%", "MEM%"],
                                 RenderOptions(thresholds=_thresholds()))
    assert "MEM%" not in data.judged
    assert data.judged[0] == "GPU%" and "SM_ACT%" in data.judged


def test_full_draws_a_timeline_for_every_voting_metric():
    rows = _multi([98] * 120, [1.0] * 120)
    full = _verify(rows, metrics=("GPU%", "SM_ACT%"), detail=True)
    assert full.count("drawn against") == 2
    assert "GPU%  (idle below" in full and "SM_ACT%  (idle below" in full
    # And the ladder underneath it, which is the rest of what the flag buys.
    assert "NODE:GPU" in full and "SWING" in full


def test_a_pooled_row_reports_its_most_actionable_shape():
    """A metric flat-idle on one card of four says so rather than averaging into the
    majority's `steady` -- the row cannot carry four shapes and that is the one that
    would change what someone does."""
    from jobscope.report import worst_shape

    class Fake:
        def __init__(self, shape):
            self.shape = shape
    assert worst_shape([Fake("steady"), Fake("flat-idle")]) == "flat-idle"
    assert worst_shape([Fake("steady"), Fake("bursty")]) == "bursty"
    assert worst_shape([Fake("declining"), Fake("bursty")]) == "declining"
    assert worst_shape([Fake("steady"), Fake("steady")]) == "steady"


def test_csv_carries_every_figure_the_block_computed():
    """--verify --csv wrote the human table into a pipeline expecting CSV. It now emits
    one row per unit and metric, with the full set whatever --full says: that
    flag is about how much terminal to spend."""
    import csv as csv_module
    rows = _series(([90] * 100) + ([0.5] * 60), gpu="0") + _series([0.2] * 160, gpu="1")
    out = io.StringIO()
    report.verify_report(rows, ["GPU%"],
                         RenderOptions(thresholds=_thresholds(), csv=True,
                                       verify_full=False),
                         out, windows=(("30m", 1800),))
    got = list(csv_module.reader(io.StringIO(out.getvalue())))
    header, body = got[0], got[1:]
    assert header[:6] == ["JOBID", "USER", "NODE", "GPU", "METRIC", "MODEL"]
    assert len(body) == 2                     # one row per unit
    by_gpu = {row[3]: dict(zip(header, row)) for row in body}
    # Durations in seconds, not _span's reading form -- that one is not round-trippable.
    assert by_gpu["1"]["IDLE_S"] == str(160 * 60)
    assert by_gpu["1"]["SHAPE"] == "flat-idle"
    # The verdict is the job's, on every row: a row is what gets sorted or joined on.
    assert len({row["VERDICT"] for row in by_gpu.values()}) == 1


def test_the_block_does_not_reshape_with_the_terminal_when_redirected():
    """PIPED_CHART_WIDTH exists for this, and nothing asserted it on this path."""
    import os
    import shutil
    real = shutil.get_terminal_size
    try:
        shutil.get_terminal_size = lambda fallback=(80, 24): os.terminal_size((400, 24))
        wide = _block(_series([50] * 200))
        shutil.get_terminal_size = lambda fallback=(80, 24): os.terminal_size((40, 24))
        narrow = _block(_series([50] * 200))
    finally:
        shutil.get_terminal_size = real
    assert wide == narrow


def test_every_row_has_a_cell_under_every_header():
    """The bug this had on its first run: unit_values and unit_headers disagreed in
    length and the zip silently dropped SHAPE off the end of every row."""
    text = _verify(_series([50] * 200))
    header = next(ln for ln in text.splitlines() if ln.startswith("NODE"))
    body = [ln for ln in text.splitlines()
            if ln.startswith("n1") or ln.startswith("holygpu")]
    assert body
    for row in body:
        assert len(row.split()) == len(header.split()), row


# --- SWING: whether a mean is reproducible at all -------------------------------
#
# Two GPU exporters reported whole-run means of 51 and 58 for the same finished job over
# the same window. Neither was wrong: the duty cycle moved a median of 67 points between
# consecutive 60s scrapes, so each sampler caught a different instant of an oscillation
# faster than the scrape rate. This column is what makes that visible instead of leaving
# the two numbers looking like a bug in one of them.

def test_swing_separates_an_aliased_signal_from_a_steady_one():
    """The whole point: the same MIN/MAX can hide either, and the mean is trustworthy in
    only one of them. Measured on the real pair -- 60.5 against a 92 range, versus 1.0
    against a 100 range."""
    flapping = _row(_verify(_series([92 if i % 2 else 0 for i in range(200)])))
    steady = _row(_verify(_series([90] * 200)))
    assert float(flapping["SWING"]) > 80      # nearly the full range
    assert float(steady["SWING"]) == 0.0
    # ...while MIN/MAX alone cannot tell them apart on the busy end.
    assert flapping["MAX"] == "92.0" and steady["MAX"] == "90.0"


def test_swing_is_a_median_so_one_transition_does_not_speak_for_the_run():
    values = [90] * 100 + [0] + [90] * 99          # one spike
    assert float(_row(_verify(_series(values)))["SWING"]) == 0.0


def test_swing_does_not_count_a_jump_across_a_gap():
    """A gap is not a step change -- counting it would report an exporter outage as a
    volatile workload, the same rule IDLEMAX follows.

    Shaped so the gap-crossing jumps are the *majority*, which is what it takes to move a
    median: three isolated samples swinging 50->100->0->100 across long gaps, against one
    adjacent run that does not move at all. Counting the crossings gives 50; refusing to
    gives 0. A single gap in an otherwise dense series proves nothing here, because the
    median absorbs it -- which is the median doing its job, not the rule doing its.
    """
    values = ([50, 50, 50] + [None] * 4 + [100] + [None] * 6 + [0]
              + [None] * 8 + [100])
    assert float(_row(_verify(_series(values)))["SWING"]) == 0.0


def test_a_single_sample_has_no_swing_to_report():
    assert _row(_verify(_series([50])))["SWING"] == "-"


def test_the_legend_says_what_a_large_swing_means_for_the_means():
    text = _verify(_series([92 if i % 2 else 0 for i in range(120)]))
    assert "SWING = median change between consecutive scrapes" in text
    assert "no mean of it is reproducible" in text


def test_swing_sits_with_min_and_max_not_with_the_quality_figures():
    """All three describe the raw signal, before any window means it into one number."""
    header = next(ln for ln in _verify(_series([50] * 120)).splitlines()
                  if ln.startswith("NODE")).split()
    assert header[header.index("MAX") + 1] == "SWING"
    assert header.index("SWING") < header.index("run")


# --- the detail views' aggregate footer -------------------------------------------

def _detail_out(records, total, level=report.NODE_LEVEL, **kw):
    out = io.StringIO()
    renderer = report.DetailRenderer(
        CTX, RenderOptions(view="all", header=True, show_dcgm=kw.get("show_dcgm", False),
                           csv=kw.get("csv", False)),
        out, level=level, specs=kw.get("specs"), total=total)
    renderer.add(build_rows(list(records), records, kw.get("dcgm_data") or {}))
    renderer.finish()
    return out.getvalue()


def test_one_job_keeps_its_per_unit_charts():
    """They answer "which of my nodes is the slow one", which sixteen rows of twelve
    columns do not. For one job that is the question."""
    records = {"1": _gpu_job("1", {"0": 90.0, "1": 10.0})}
    text = _detail_out(records, total=1)
    assert "Efficiency by node" in text
    assert "Summary by metric" not in text


def test_a_multi_job_listing_ends_in_one_aggregate_not_a_chart_per_job():
    """A partition sweep otherwise ends in a dozen charts of one bar each and no reading of
    the selection anywhere. The question there is "how is this selection doing"."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    text = _detail_out(records, total=2)
    assert "Efficiency by node" not in text
    assert "Summary by metric" in text and "Average efficiency" in text
    assert "█" in text                      # the aggregate bars are drawn
    # And the rows are still per node -- the granularity the flag asked for.
    assert "NODE" in text and "Job 1" in text and "Job 2" in text


def test_the_aggregate_equals_the_per_job_views_summary():
    """The promise that makes this a presentation change rather than a second arithmetic:
    `--per-node` and the default view must report the same summary for one selection and
    differ only in row granularity. Aggregating the printed unit rows instead would weight
    by node and quietly disagree.
    """
    records = {"1": _gpu_job("1", {"0": 90.0, "1": 20.0}),
               "2": _gpu_job("2", {"0": 10.0})}
    detail = _detail_out(records, total=2)

    out = io.StringIO()
    per_job = report.SummaryRenderer(CTX, RenderOptions(view="all", header=True), out)
    per_job.add(build_rows(list(records), records, {}))
    per_job.finish()

    def sections(text):
        start = text.index("1. Summary by metric")
        return text[start:]

    assert sections(detail) == sections(out.getvalue())


def test_the_aggregate_stays_out_of_the_csv():
    """Its CSV form is the Stat/Worst/Jobs rows, and adding those to --per-gpu --csv would
    change a machine format for a reader that did not ask. The bars already self-suppress
    there for the same reason."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    text = _detail_out(records, total=2, csv=True)
    for prefix in ("Stat", "Worst", "Jobs"):
        assert not any(ln.startswith(prefix) for ln in text.splitlines()), (prefix, text)


# --- the CSV column contract, pinned ------------------------------------------------
#
# jobscope plot parses this output (plot.parse_csv), so the header row and its order are
# a public interface, not an implementation detail. The tests above check that each cell
# lands under its own header; these check that the set of headers is what it was. Read
# through parse_csv rather than by splitting, so they assert against the actual consumer.

def test_the_summary_csv_header_row_is_fixed():
    """--all-metrics widens the profiling block and nothing else moves."""
    records = {"1": _gpu_job("1", {"0": 90.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", show_dcgm=True, csv=True, header=True), out)
    renderer.add(build_rows(list(records), records, {"1": ({"SM_ACT%": 40.0}, {})}))
    renderer.finish()
    columns, _rows = plot.parse_csv(io.StringIO(out.getvalue()))
    assert columns == ["JOBID", "USER", "STATE", "NODE", "CPU%", "MEM%", "#GPU", "GPU%",
                       "GMEM%", "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W", "RUNTIME"]


@pytest.mark.parametrize("level, unit", [("gpu", "GPU"), ("node", "#GPU")])
def test_the_detail_csv_header_row_is_fixed(level, unit):
    """Both levels share a renderer and differ in exactly one cell -- see report.py's
    note on why cell 1 is a card's minor number at gpu level and a count at node level.

    RUNTIME prints last while occupying row cell 7, so this also pins that the display
    order and Column.index stay decoupled.
    """
    record = _gpu_job("1", {"0": 90.0, "1": 50.0})
    out = io.StringIO()
    renderer = report.DetailRenderer(
        CTX, RenderOptions(view="all", show_dcgm=True, csv=True, header=True),
        out, level=level)
    renderer.add(build_rows(["1"], {"1": record},
                 {"1": ({}, {("n1", "0"): dict(_PROFILING_VALUES),
                             ("n1", "1"): dict(_PROFILING_VALUES)})}))
    renderer.finish()
    columns, _rows = plot.parse_csv(io.StringIO(out.getvalue()))
    assert columns == ["JOBID", "NODE", unit, "CPU%", "CPU-MEM", "GPU%", "GPU-MEM",
                       "GMEM%", "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W", "RUNTIME"]


@pytest.mark.parametrize("name", ["jobstats", "dcgm", "nvml"])
def test_the_detail_csv_header_row_survives_every_source_order(name, preference):
    """Promoting an exporter moves which source *fills* the prefix cells; it must not
    move the cells. This is the CSV half of the by-header assertions above."""
    _columns, text = _detail_cells(name, preference, csv=True)
    columns, rows = plot.parse_csv(io.StringIO(text))
    assert columns == ["JOBID", "NODE", "GPU", "CPU%", "CPU-MEM", "GPU%", "GPU-MEM",
                       "GMEM%", "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W", "RUNTIME"]
    assert len(rows) == 1


def test_the_detail_prefix_follows_the_per_unit_header_list():
    """One list drives both the cells a row carries and the columns that print them.

    They used to be two hand-kept lists -- jobstats built a 7-tuple in one order and
    detail_prefix wrote out seven Columns in another -- with nothing tying them
    together but the fact that both were correct at the time.
    """
    from jobscope.jobstats import UNIT_HEADERS
    prefix = report.detail_prefix("gpu")
    assert [c.header for c in prefix] == ["NODE", "GPU"] + list(UNIT_HEADERS)
    # Contiguous from 0, so the runtime cell that follows can be a constant index.
    assert [c.index for c in prefix] == list(range(len(prefix)))
    assert report._RUNTIME_INDEX == len(prefix)


def test_the_two_detail_levels_differ_in_exactly_one_column():
    """Which is what lets them share a renderer -- see report.py's note on cell 1."""
    gpu = report.detail_prefix("gpu")
    node = report.detail_prefix("node")
    differ = [(a.header, b.header) for a, b in zip(gpu, node) if a != b]
    assert differ == [("GPU", "#GPU")]
