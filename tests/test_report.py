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
    assert streamed.count("Used/GPU:") == 1


def test_summary_renderer_two_adds_equals_summarize_csv(gpu_record):
    jobids, records, dcgm, chunks = _two_gpu_chunks(gpu_record)
    options = RenderOptions(view="all", show_dcgm=True, csv=True, header=True)
    single = _render(summarize, jobids, records, dcgm, CTX, options)
    streamed = _render_stream(report.SummaryRenderer, CTX, options, chunks)
    assert streamed == single
    _, rows = plot.parse_csv(io.StringIO(streamed))
    assert len(rows) == 2  # footer rows dropped by the parser


def test_summary_renderer_single_row_no_footer(gpu_record, cpu_record):
    # The second chunk is filtered out by the gpu view, so only one row renders
    # and the whole footer block must stay suppressed.
    options = RenderOptions(view="gpu", show_dcgm=False, csv=False, header=True)
    streamed = _render_stream(report.SummaryRenderer, CTX, options,
                              [(["100"], {"100": gpu_record}, {}),
                               (["200"], {"200": cpu_record}, {})])
    assert "100" in streamed
    for label in ("Used/", "GPUs:", "Bands:", "Worst:", "Jobs:"):
        assert label not in streamed


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
    renderer.add(list(records), records, dcgm_data or {})
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
    from jobscope.dcgm import ALL_SPECS
    records = {"1": _gpu_job("1", {"0": 0.0}),
               "2": _gpu_job("2", {"0": 100.0, "1": 100.0})}
    dcgm = {"1": ({"SM_ACT%": 10.0, "ENERGY_kWh": 1.0, "PWRmax_W": 300.0}, {}),
            "2": ({"SM_ACT%": 90.0, "ENERGY_kWh": 8.0, "PWRmax_W": 500.0}, {})}
    pooled = _footers(records, show_dcgm=True, dcgm_data=dcgm, specs=ALL_SPECS)["UsedPerGPU"]
    assert pooled["SM_ACT%"] == "63.3"          # (10*1 + 90*2)/3, pooled over GPUs
    assert pooled["ENERGY_kWh"] == "4.500"      # (1 + 8)/2, the per-job mean
    assert pooled["PWRmax_W"] == "400"          # (300 + 500)/2, likewise


# --- the efficiency block ---------------------------------------------------

def test_the_bands_report_job_share_and_resource_share():
    """The finding is the gap between the two shares.

    One idle 16-GPU job against four busy 1-GPU jobs: 20% of the jobs, but 80% of
    the GPUs. Either number alone hides it.
    """
    records = {"idle": _gpu_job("idle", {str(i): 0.0 for i in range(16)})}
    records.update({str(i): _gpu_job(str(i), {"0": 90.0}) for i in range(4)})
    line = _band_line(records)
    assert "red<25 1 job (20%)/16 (80%)" in line
    assert "green 4 jobs (80%)/4 (20%)" in line


def _band_line(records, column="GPU%", **kw):
    """The `Bands <column>:` line from a text-rendered table."""
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view=kw.get("view", "all"), csv=False, header=True,
                           time_weighted=kw.get("time_weighted", False)), out)
    renderer.add(list(records), records, {})
    renderer.finish()
    return [ln for ln in out.getvalue().splitlines()
            if ln.startswith("Bands " + column)][0]


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
        renderer.add(list(records), records, {})
        renderer.finish()
        return out.getvalue()

    both = lines("all")
    assert "GPUs:" in both and "Cores:" in both
    assert "Bands GPU%:" in both and "Bands CPU%:" in both
    # The leading resource names the pooled row, and GPUs lead in the wide view.
    assert "Used/GPU:" in both
    assert both.index("GPUs:") < both.index("Cores:")

    assert "Cores:" not in lines("gpu")
    assert "GPUs:" not in lines("cpu")


def test_the_totals_line_reconciles_with_its_own_percentage():
    """used is fractional even in the count form -- it is GPU-equivalents busy.

    Rounding it to an integer made the three numbers contradict the percentage:
    20 allocated and 17.4 used printed as "17 used  3 idle (13%)", and 3/20 is 15%.
    """
    records = {"a": _gpu_job("a", {"0": 90.0}), "b": _gpu_job("b", {"0": 80.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(CTX, RenderOptions(view="gpu", header=True), out)
    renderer.add(list(records), records, {})
    renderer.finish()
    line = [ln for ln in out.getvalue().splitlines() if ln.startswith("GPUs:")][0]
    assert "2 alloc  1.7 used  0.3 idle (15%)" in line     # 0.3/2 = 15%


def test_the_totals_line_splits_allocation_into_used_and_idle():
    records = {"idle": _gpu_job("idle", {str(i): 0.0 for i in range(3)}),
               "busy": _gpu_job("busy", {"0": 100.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(CTX, RenderOptions(view="all", header=True), out)
    renderer.add(list(records), records, {})
    renderer.finish()
    text = out.getvalue()
    assert "GPUs:        4 alloc  1 used  3 idle (75%)" in text


def test_worst_ranks_by_wasted_resource_not_by_size():
    """A large job at a mediocre rate wastes more than a small one at zero."""
    records = {"big": _timed_job("big", 100 * 3600, gpu_util=24.0),    # 76 GPU-h idle
               "small": _timed_job("small", 10 * 3600, gpu_util=0.0)}  # 10 GPU-h idle
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view="all", header=True, time_weighted=True), out)
    renderer.add(list(records), records, {})
    renderer.finish()
    worst = [ln for ln in out.getvalue().splitlines() if ln.startswith("Worst GPU:")][0]
    assert worst.index("big") < worst.index("small")


def _worst_lines(records, view="all", time_weighted=True):
    """``{label: line}`` for the Worst rows of a text-rendered table."""
    out = io.StringIO()
    renderer = report.SummaryRenderer(
        CTX, RenderOptions(view=view, header=True, time_weighted=time_weighted), out)
    renderer.add(list(records), records, {})
    renderer.finish()
    # Values are the payload only: a job id like "A" would otherwise match the "A"
    # in "alice", and "C" the "C" in "Worst CPU:".
    return {ln.split(":")[0]: ln.split(":", 1)[1] for ln in out.getvalue().splitlines()
            if ln.startswith("Worst")}


def test_the_combined_worst_normalizes_each_resource():
    """The combined order can differ from both single-resource orders.

    GPU-hours and core-hours are not addable, so each job's waste is expressed as a
    share of the selection's total waste in that resource and the two shares summed.
    Here A wastes almost all the GPU-time, B almost all the core-time, and C a
    little under a third of each -- so B edges out A, and C trails both, an order
    neither single list produces.
    """
    records = {
        # 10 GPU-hours idle, but 90% of its 2 cores used.
        "gpuhog": _timed_job("gpuhog", 3600, gpu_util=0.0, gpus=10, cores=2,
                             cpu_seconds=6480),
        # GPU busy, but 100 core-hours idle.
        "cpuhog": _timed_job("cpuhog", 3600, gpu_util=95.0, gpus=1, cores=100,
                             cpu_seconds=0),
        # Red in both, moderately.
        "middle": _timed_job("middle", 3600, gpu_util=0.0, gpus=4, cores=40,
                             cpu_seconds=0),
    }
    lines = _worst_lines(records)
    # Each single-resource list only holds jobs red in *that* resource.
    assert lines["Worst GPU"].index("gpuhog") < lines["Worst GPU"].index("middle")
    assert "cpuhog" not in lines["Worst GPU"]
    assert lines["Worst CPU"].index("cpuhog") < lines["Worst CPU"].index("middle")
    assert "gpuhog" not in lines["Worst CPU"]
    both = lines["Worst both"]
    assert both.index("cpuhog") < both.index("gpuhog") < both.index("middle")
    # The shares say which resource put each job there.
    assert "gpuhog 71%gpu+0%cpu" in both and "cpuhog 0%gpu+71%cpu" in both


def test_the_combined_worst_counts_waste_from_a_resource_a_job_is_green_in():
    """A job red in one resource still has its other-resource waste counted.

    It is real waste; only the red filter decides who is a candidate at all.
    """
    records = {
        "red_gpu": _timed_job("red_gpu", 3600, gpu_util=0.0, gpus=1,
                              cores=10, cpu_seconds=18000),   # CPU% 50, green
        "red_cpu": _timed_job("red_cpu", 3600, gpu_util=90.0, gpus=1,
                              cores=10, cpu_seconds=0),       # CPU% 0, red
    }
    both = _worst_lines(records)["Worst both"]
    # red_gpu is green on CPU yet still shows a nonzero CPU share (5 core-hours
    # idle of 15 total).
    assert "red_gpu 91%gpu+33%cpu" in both


def test_no_combined_line_when_only_one_resource_wasted_anything():
    """With nothing idle on one side, the combined view repeats the other."""
    records = {"1": _timed_job("1", 3600, gpu_util=0.0, gpus=1, cores=4,
                               cpu_seconds=4 * 3600),          # CPU% 100
               "2": _timed_job("2", 3600, gpu_util=10.0, gpus=1, cores=4,
                               cpu_seconds=4 * 3600)}
    lines = _worst_lines(records)
    assert "Worst GPU" in lines and "Worst both" not in lines


def test_narrow_views_show_only_their_own_worst_row():
    records = {"1": _timed_job("1", 3600, gpu_util=0.0, gpus=2, cores=8, cpu_seconds=0),
               "2": _timed_job("2", 3600, gpu_util=5.0, gpus=1, cores=4, cpu_seconds=0)}
    assert set(_worst_lines(records, view="gpu")) == {"Worst GPU"}
    assert set(_worst_lines(records, view="cpu")) == {"Worst CPU"}
    assert set(_worst_lines(records)) == {"Worst GPU", "Worst CPU", "Worst both"}


def test_no_worst_line_when_nothing_is_red():
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 80.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(CTX, RenderOptions(view="all", header=True), out)
    renderer.add(list(records), records, {})
    renderer.finish()
    # No label at all, not merely the old "Worst:" spelling.
    assert "Worst" not in out.getvalue()


def test_the_block_follows_the_cpu_view_to_cores():
    """A --cpu run bands CPU% over cores, not GPU% over GPUs."""
    records = {"1": _gpu_job("1", {"0": 90.0}), "2": _gpu_job("2", {"0": 10.0})}
    out = io.StringIO()
    renderer = report.SummaryRenderer(CTX, RenderOptions(view="cpu", header=True), out)
    renderer.add(list(records), records, {})
    renderer.finish()
    text = out.getvalue()
    assert "Cores:" in text and "GPUs:" not in text
    assert "Bands CPU%:" in text and "Bands GPU%:" not in text
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
    return JobRecord(jobid=jid, state="COMPLETED", name="j", runtime="-", nodes="1",
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
    renderer.add(list(records), records, kw.get("dcgm_data") or {})
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
    assert "allocated=56.3h" in footers["GPUhours"].values()   # 202800s / 3600
    # 100 of 101 jobs are red, but they hold only 15% of the resource-time.
    assert "red=100" in footers["GPUhours"].values()


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
    represents those two days. Only --avg and finished jobs carry runtime-long
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
    renderer.add(list(records), records, {})
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

import re as _re  # noqa: E402

_ESC = _re.compile(r"\033\[[0-9;]*m")


def _thresholds():
    from jobscope.config import Thresholds
    return Thresholds(gpu=25, gmem=20, cpu=25, mem=25, default=15)


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
    records = {"1": _gpu_job("1", {"0": 30.0}), "2": _gpu_job("2", {"0": 95.0})}
    row = [r for r in _render_colored(records).splitlines() if r.startswith("1 ")][0]
    assert "\033[33m30" in row             # >= 25 but < 50


def test_color_does_not_shift_the_columns():
    """The escapes wrap the padded cell, so stripping them restores the plain row.

    Tinting the value instead would make str.format count the escape bytes toward
    the column width and skew everything to its right.
    """
    records = {"1": _gpu_job("1", {"0": 5.0}), "2": _gpu_job("2", {"0": 95.0})}
    assert _ESC.sub("", _render_colored(records)) == _render_colored(records, color=False)


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


def test_non_percent_columns_are_never_tinted(gpu_record):
    """There is no good or bad wattage, runtime or job id."""
    records = {"1": _gpu_job("1", {"0": 5.0}), "2": _gpu_job("2", {"0": 95.0})}
    dcgm = {"1": ({"POWER_W": 90.0}, {}), "2": ({"POWER_W": 600.0}, {})}
    text = _render_colored(records, dcgm_data=dcgm, show_dcgm=True)
    assert "\033[31m90" not in text and "\033[32m600" not in text


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
    red_map, default = thresholds.red_map(), thresholds.default
    for header in ("GPU%", "GMEM%", "CPU%", "SM_ACT%"):
        for value in (0, 10, 24, 25, 40, 49, 50, 99):
            assert thresholds.grade(header, value) == plot.grade(header, value, red_map, default)
