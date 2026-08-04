"""Tests for the live view: squeue parsing, GPU identity, and the summary synthesis."""

import io

import pytest

from jobscope import timeseries as ts
from jobscope.cpu import host_stats_many
from jobscope.dcgm import (
    ALL_SPECS,
    DEFAULT_SPECS,
    SPEC_BY_HEADER,
    spec_named,
    window_query,
)
from jobscope.errors import JobscopeError
from jobscope.job_ave_stats import synthesize_stats
from jobscope.jobstats import GIB, jobstats_metrics
from jobscope.report import (
    RenderOptions,
    running_combined_timeseries,
    running_cpu_timeseries,
    running_timeseries,
)
from jobscope.running import (
    DEFAULT_RUNNING_SPECS,
    EXTENDED_RUNNING_SPECS,
    SQUEUE_FORMAT,
    Gpu,
    RunningSelection,
    build_columns,
    clip_to_job,
    collect_instant,
    filter_by_elapsed,
    format_duration,
    gpu_labels,
    job_sort_key,
    parse_duration,
    parse_squeue,
    parse_start_time,
    range_window,
    specs_for,
    timeseries_step,
)
from jobscope.slurm import JobRecord


def _render_running_cpu(jobs, client, options, out, workers=1):
    """Collect then render the running --cpu --ts view, as select.py wires it."""
    match = ts.UnitFilter(options.nodename)
    collected = ts.running_host(jobs, client, None, workers=workers,
                                window=options.window,
                                host_specs=options.cgroup_specs, match=match)
    running_cpu_timeseries(collected, ts.running_host_order(jobs, collected),
                           options, out=out)
    match.check()

# squeue -o "%A|%i|%u|%N|%g|%j|%b|%C|%S": raw id, display id, user, nodelist,
# group, name, gres, cpus, start. For array element 12345_6 the raw id differs --
# and Prometheus keys on the raw one, which is the whole reason %A is read.
ARRAY_LINE = "34843629|34843528_6|alice|holygpu01|kempner|train|gpu:1|16|2026-07-24T09:00:00"
PLAIN_LINE = "34622920|34622920|bob|holygpu02|kempner|infer|gpu:2|8|2026-07-24T10:00:00"


def test_parse_squeue_keys_on_the_raw_id_for_array_elements():
    jobs = parse_squeue(ARRAY_LINE)
    # Keyed by %A (what Prometheus reports), displayed as %i.
    assert list(jobs) == [34843629]
    assert jobs[34843629]["jobid"] == "34843528_6"
    assert jobs[34843629]["user"] == "alice"


def test_parse_squeue_raw_and_display_agree_for_plain_jobs():
    jobs = parse_squeue(PLAIN_LINE)
    assert list(jobs) == [34622920]
    assert jobs[34622920]["jobid"] == "34622920"


def test_parse_squeue_skips_headers_short_and_unparseable_lines():
    text = "\n".join(["JOBID|...", "", "too|few|fields", "notanid|x|u|n|g|j|G|C|S", PLAIN_LINE])
    assert list(parse_squeue(text)) == [34622920]


def test_squeue_format_asks_for_gres_not_the_group_id():
    """%b is tres-per-node; %G is the numeric group ID, which looks like data.

    Reading %G here yielded values such as 5137 in a field named "gpus" -- a
    plausible-looking number that is not a GPU count at all.
    """
    assert "%b" in SQUEUE_FORMAT and "%G" not in SQUEUE_FORMAT


def test_parse_squeue_records_the_gres_request():
    assert parse_squeue(ARRAY_LINE)[34843629]["gres"] == "gpu:1"


def test_parse_squeue_keeps_the_whole_nodelist():
    # Splitting on "," would mangle a compressed range like holygpu8a[10402,10404].
    line = "1|1|alice|holygpu8a[10402,10404]|g|n|gpu:2|8|2026-07-24T09:00:00"
    assert parse_squeue(line)[1]["node"] == "holygpu8a[10402,10404]"


def test_parse_squeue_tolerates_a_missing_start_time():
    line = "1|1|alice|node01|g|n|gpu:1|8|N/A"
    job = parse_squeue(line)[1]
    assert job["start_epoch"] is None and job["elapsed_seconds"] is None


def test_parse_start_time_rejects_non_timestamps():
    assert parse_start_time("Unknown") == (None, None)
    epoch, elapsed = parse_start_time("2026-07-24T09:00:00")
    assert epoch is not None and elapsed is not None


def test_filter_by_elapsed_drops_short_and_startless_jobs(capsys):
    jobs = {1: {"jobid": "1", "elapsed_seconds": 7200},
            2: {"jobid": "2", "elapsed_seconds": 60},
            3: {"jobid": "3", "elapsed_seconds": None}}
    assert list(filter_by_elapsed(jobs, 3600)) == [1]
    assert "no usable start time for job 3" in capsys.readouterr().err


# --- job ordering -----------------------------------------------------------

def test_job_sort_key_groups_array_elements_numerically():
    ids = ["34843528_10", "34622920", "34843528_2", "34843527"]
    jobs = [{"jobid": j} for j in ids]
    ordered = [j["jobid"] for j in sorted(jobs, key=job_sort_key)]
    # Raw IDs would interleave these; the display ID keeps the array together and
    # sorts element 2 before element 10.
    assert ordered == ["34622920", "34843527", "34843528_2", "34843528_10"]


def test_job_sort_key_puts_a_plain_job_before_its_own_array_elements():
    assert job_sort_key({"jobid": "100"}) < job_sort_key({"jobid": "100_0"})


def test_job_sort_key_survives_a_non_numeric_id():
    assert job_sort_key({"jobid": "weird"})[0] > 0


# --- GPU identity and MIG ---------------------------------------------------

def test_gpu_labels_names_whole_cards_by_minor():
    found = [("GPU-aaa", 1, "node01", 0), ("GPU-bbb", 1, "node01", 3)]
    assert gpu_labels(found) == {"GPU-aaa": "GPU 0", "GPU-bbb": "GPU 3"}


def test_gpu_labels_enumerates_mig_siblings_sharing_a_minor():
    # Every MIG instance inherits its parent card's minor number, so these would
    # collapse onto one row if minor were the key.
    found = [("MIG-zzz", 1, "node01", 0), ("MIG-aaa", 1, "node01", 0)]
    labels = gpu_labels(found)
    # Enumerated by sorted UUID, so the order is stable run to run.
    assert labels == {"MIG-aaa": "MIG 0.0", "MIG-zzz": "MIG 0.1"}


def test_gpu_labels_leaves_a_lone_slice_unnumbered():
    assert gpu_labels([("MIG-aaa", 1, "node01", 0)]) == {"MIG-aaa": "MIG 0"}


def test_gpu_labels_keeps_the_same_minor_on_two_hosts_distinct():
    found = [("GPU-aaa", 1, "node01", 0), ("GPU-bbb", 1, "node02", 0)]
    labels = gpu_labels(found)
    assert len(labels) == 2 and set(labels.values()) == {"GPU 0"}


def test_csv_id_is_the_bare_minor_for_a_card_and_dotted_for_a_slice():
    assert Gpu("u", 1, "n", 2, "GPU 2").csv_id == "2"
    assert Gpu("u", 1, "n", 2, "MIG 2.1").csv_id == "2.1"


# --- durations --------------------------------------------------------------

@pytest.mark.parametrize("text,seconds", [
    ("30s", 30), ("5m", 300), ("2h", 7200), ("7d", 604800), ("0s", 0)])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("bad", ["5", "m", "1w", "-5m", "5 m", ""])
def test_parse_duration_rejects_junk(bad):
    with pytest.raises(JobscopeError):
        parse_duration(bad)


def test_format_duration_round_trips_whole_units():
    for text in ("30s", "5m", "2h", "7d"):
        assert format_duration(parse_duration(text)) == text


def test_timeseries_step_never_finer_than_the_scrape_interval():
    assert timeseries_step(600, 60) == 60
    assert timeseries_step(600, 60, requested=5) == 5     # explicit wins
    # Long jobs widen to stay under Prometheus' points-per-series cap.
    assert timeseries_step(10_000_000, 60) > 60


def test_a_window_is_the_end_of_the_run_not_the_start():
    """--ts 1h means the last hour, and narrows the query rather than the rows."""
    start, end = 1000, 1000 + 86400
    assert range_window(start, end, 3600, 60)[0] == end - 3600
    assert range_window(start, end, None, 60)[0] == start     # no window: the whole run
    assert range_window(start, end, 0, 60)[0] == start
    # A window longer than the run is simply the run, never a start before it.
    assert range_window(start, end, 999_999, 60)[0] == start


def test_a_windowed_query_keeps_its_resolution():
    """The step is measured over the span queried, so an hour of a week-long job is
    not coarsened by the length of the week."""
    week = 7 * 86400
    assert timeseries_step(week, 60) > 60               # the whole run is coarsened
    assert timeseries_step(3600, 60) == 60              # one hour of it is not
    assert range_window(0, week, 3600, 60)[1] == 60     # and the window is not either


def test_a_window_lands_on_the_runs_own_sample_grid():
    """Prometheus anchors a range query's points at `start`, so an unaligned window
    relabels every sample -- and on a running job, whose end is "now", the grid then
    drifts with the clock between invocations. Aligned, --ts 1h returns exactly the
    rows a full --ts would.
    """
    start, step = 1785521696, 60          # a real job start: 1785521696 % 60 == 56
    for end in range(start + 86400, start + 86400 + step):   # any "now" in one step
        begin, span = range_window(start, end, 3600, step)
        assert span == step
        assert (begin - start) % step == 0                   # on the run's grid
        assert 3600 <= end - begin < 3600 + step             # never short of the ask


# --- catalogs ---------------------------------------------------------------

def test_live_catalog_column_order_and_membership():
    assert [h for _k, h, _d in build_columns(DEFAULT_RUNNING_SPECS)] == [
        "GPU%", "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W", "GMEM_GB", "GMEM%"]


def test_gpu_utilization_is_always_present_in_the_live_view():
    # A running job has no jobstats summary, so this is the only place GPU% comes from; there
    # is deliberately no narrower catalog that could drop it.
    assert "GPU%" in [h for _k, h, _d in build_columns(DEFAULT_RUNNING_SPECS)]
    assert "GPU%" in [h for _k, h, _d in build_columns(EXTENDED_RUNNING_SPECS)]


def test_total_memory_is_queried_but_not_shown():
    # It exists only to derive MEM%.
    assert any(s.header == "GMEM_TOTAL_GB" for s in DEFAULT_RUNNING_SPECS)
    assert "GMEM_TOTAL_GB" not in [h for _k, h, _d in build_columns(DEFAULT_RUNNING_SPECS)]


def test_extended_catalog_excludes_delta_reduced_counters():
    # A delta needs two points, so it is meaningless in an instant snapshot.
    assert all(s.reducer != "delta" for s in EXTENDED_RUNNING_SPECS)
    assert any(s.reducer == "delta" for s in ALL_SPECS), "fixture assumes one exists"


def test_specs_for_maps_the_view_names():
    assert specs_for("all") is EXTENDED_RUNNING_SPECS
    assert specs_for(None) is DEFAULT_RUNNING_SPECS


def test_live_and_dcgm_render_identical_columns():
    """The whole point of sharing the catalog: one job, one column set.

    A finished job and a running one must be described by the same columns, so the
    same eye (and the same script) reads both.
    """
    from jobscope.dcgm import columns_for
    assert build_columns(DEFAULT_RUNNING_SPECS) == columns_for(DEFAULT_SPECS)


def test_gmem_percent_is_per_gpu_and_handles_a_missing_total():
    from jobscope.dcgm import DERIVED_COLUMNS
    mem_pct = DERIVED_COLUMNS[0].fn
    assert mem_pct({"mem": 20.0, "memtot": 80.0}) == 25.0
    assert mem_pct({"mem": 20.0, "memtot": 0}) is None
    assert mem_pct({"mem": None, "memtot": 80.0}) is None


# --- the runtime-window clip ------------------------------------------------

def test_clip_applies_to_nvidia_metrics_only():
    # By provider, not by column: which source serves GPU% depends on the preference,
    # and the whole point of this test is that the two providers clip differently.
    duty, smact = spec_named("duty"), SPEC_BY_HEADER["SM_ACT%"]
    # Same exporter as nvidia_gpu_jobId, so `and` can match on identical labels.
    assert clip_to_job(duty, 42) == "nvidia_gpu_jobId == 42"
    # DCGM carries different labels, so `and` never matches; the window alone bounds it.
    assert clip_to_job(smact, 42) is None
    # Including the DCGM candidate for GPU% itself -- preferring it trades ownership
    # clipping for a window bound, exactly as SM_ACT% is already bounded.
    assert clip_to_job(spec_named("duty_dcgm"), 42) is None


def test_clipped_window_query_is_well_formed():
    duty = spec_named("duty")
    query = window_query(duty, ["GPU-a"], 900, clip=clip_to_job(duty, 42))
    assert query == ('avg_over_time((nvidia_gpu_duty_cycle{uuid=~"^(GPU-a)$"} '
                     'and nvidia_gpu_jobId == 42)[900s:])')


def test_unclipped_window_query_is_unchanged():
    smact = SPEC_BY_HEADER["SM_ACT%"]
    assert window_query(smact, ["GPU-a"], 900) == window_query(
        smact, ["GPU-a"], 900, clip=None)


# --- selection --------------------------------------------------------------

def test_selection_describe_mentions_the_runtime_floor():
    assert "1h" in RunningSelection(min_elapsed=3600).describe()
    assert RunningSelection(jobids=["1", "2"]).describe() == "2 job ID(s)"


def test_describe_omits_user_and_partition_but_describe_filters_names_them():
    """The context block prints those on their own lines; the error message cannot.

    `jobscope -p kempner` reporting only "running, longer than 10m" reads as an
    idle partition, when in practice the default user filter excluded the 20
    people who were on it.
    """
    sel = RunningSelection(partition="kempner", user="bdesinghu", min_elapsed=600)
    assert sel.describe() == "running, longer than 10m"
    assert sel.describe_filters() == (
        "user bdesinghu, partition kempner, running, longer than 10m")


def test_describe_filters_says_all_users_when_unfiltered():
    sel = RunningSelection(partition="kempner", user=None, min_elapsed=600)
    assert sel.describe_filters().startswith("all users, partition kempner")


def test_widening_hints_cover_only_the_active_filters():
    """Suggesting -a when every user is already included would be noise."""
    assert RunningSelection(user="alice", partition="p", min_elapsed=600).widening_hints() == [
        "add -a to include every user",
        "set --min-elapsed 0s to include jobs that just started",
        "drop -p to search every partition"]
    assert RunningSelection(user=None, partition=None, min_elapsed=0).widening_hints() == []


def test_explicit_job_ids_keep_the_plain_description():
    """Job IDs bypass the filters, so naming them would be misleading."""
    sel = RunningSelection(jobids=["1"], user="alice", partition="p")
    assert sel.describe_filters() == "1 job ID(s)"


# --- summary synthesis for running jobs ---------------------------------------

class JobstatsClient:
    """Prometheus stand-in returning canned cgroup_* and nvidia_gpu_* series."""

    def __init__(self, host="node01", minor="0", gpu_series=True):
        self.host, self.minor, self.gpu_series = host, minor, gpu_series
        self.queries = []
        self.sampling_period = 60

    def query(self, query, at, timeout=None):
        self.queries.append(query)
        host_values = {
            "cgroup_cpus": 2,
            "cgroup_cpu_total_seconds": 150,
            "cgroup_memory_rss_bytes": 8 * GIB,
            "cgroup_memory_total_bytes": 16 * GIB,
        }
        for name, value in host_values.items():
            if name in query:
                return [{"metric": {"host": self.host + ":9306"}, "value": [at, str(value)]}]
        if not self.gpu_series:
            return []
        gpu_values = {
            "nvidia_gpu_duty_cycle": 70,
            "nvidia_gpu_memory_used_bytes": 40 * GIB,
            "nvidia_gpu_memory_total_bytes": 80 * GIB,
        }
        for name, value in gpu_values.items():
            if name in query:
                return [{"metric": {"host": self.host, "minor_number": self.minor},
                         "value": [at, str(value)]}]
        return []


def _running(**kw):
    base = dict(jobid="100_6", state="RUNNING", name="train", runtime="00:01:40", nodes="1",
                gpus=1, stats={}, start=1000, end=1100, duration=100, jobid_raw="12345",
                cluster="odyssey", user="alice")
    base.update(kw)
    return JobRecord(**base)


def test_synthesize_stats_feeds_jobstats_metrics():
    stats = synthesize_stats(_running(), JobstatsClient())
    # cpu = 100*150/(100*2) = 75, mem = 100*8/16 = 50, gpu = 70, gmem = 100*40/80 = 50.
    assert jobstats_metrics(stats, gpus=1).known() == {"CPU%": 75, "MEM%": 50,
                                                "GPU%": 70, "GMEM%": 50}


def test_synthesize_stats_queries_the_raw_job_id():
    # cgroup_* carries a real jobid label, but it holds the RAW per-element id:
    # array element 100_6 appears as jobid="12345". Using .jobid would find nothing.
    client = JobstatsClient()
    synthesize_stats(_running(), client)
    assert all("12345" in q for q in client.queries)
    assert not any("100_6" in q for q in client.queries)


def test_synthesize_stats_shapes_gpu_maps_by_minor_string():
    stats = synthesize_stats(_running(), JobstatsClient(minor="3"))
    node = stats["nodes"]["node01"]
    assert node["gpu_utilization"] == {"3": 70.0}
    assert node["gpu_used_memory"] == {"3": 40 * GIB}


def test_synthesize_stats_rounds_like_the_stored_summary():
    # jobstats_detail renders utilization with %g, which assumes stored precision.
    stats = synthesize_stats(_running(), JobstatsClient())
    node = stats["nodes"]["node01"]
    assert isinstance(node["cpus"], int)
    assert isinstance(node["used_memory"], int)
    assert node["gpu_utilization"]["0"] == round(node["gpu_utilization"]["0"], 1)


def test_synthesize_stats_skips_gpu_queries_for_a_cpu_only_job():
    client = JobstatsClient()
    stats = synthesize_stats(_running(gpus=0), client)
    assert not any("nvidia_gpu" in q for q in client.queries)
    got = jobstats_metrics(stats, gpus=0)               # no gpu%, no gmem%
    assert got.value("GPU%") is None and got.value("GMEM%") is None


def test_synthesize_stats_returns_empty_without_a_usable_window():
    assert synthesize_stats(_running(duration=None), JobstatsClient()) == {}
    assert synthesize_stats(_running(jobid_raw=""), JobstatsClient()) == {}


def test_synthesize_stats_degrades_instead_of_raising():
    class Broken:
        sampling_period = 60

        def query(self, *a, **kw):
            raise RuntimeError("prometheus is down")

    assert synthesize_stats(_running(), Broken()) == {}


def test_fill_running_leaves_a_finished_job_alone():
    from jobscope.job_ave_stats import fill_running
    stored = {"total_time": 1, "nodes": {}}
    records = {"1": _running(state="COMPLETED", stats=stored)}
    assert fill_running(records, ["1"], JobstatsClient()) == 0
    assert records["1"].stats is stored


def test_fill_running_fills_only_unsummarised_running_jobs():
    from jobscope.job_ave_stats import fill_running
    records = {"1": _running(), "2": _running(state="COMPLETED")}
    assert fill_running(records, ["1", "2"], JobstatsClient()) == 1
    assert records["1"].stats and not records["2"].stats


# --- rendering --------------------------------------------------------------

def test_live_timeseries_uses_the_schema_plot_reads():
    # jobscope plot keys on EPOCH/TIME and groups series by (NODE, GPU), so this
    # header must stay in lockstep with dcgm_timeseries'.
    jobs = {1: {"jobid": "100_6", "user": "alice", "node": "node01", "name": "train"}}
    gpus = {"GPU-a": Gpu("GPU-a", 1, "node01", 3, "GPU 3")}
    samples = {"GPU-a": {1000: {"duty": 90.0, "mem": 10.0, "memtot": 80.0}}}
    out = io.StringIO()
    running_timeseries(jobs, samples, gpus, DEFAULT_RUNNING_SPECS, RenderOptions(), out=out)
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("JOBID,USER,EPOCH,TIME,NODE,GPU,")
    assert lines[1].startswith("100_6,alice,1000,")   # USER names whose job it is
    # MEM% is recomputed per timestamp, so it tracks memory growth: 10/80 = 12.5.
    assert lines[1].endswith("12.5")


def test_live_cpu_timeseries_uses_the_schema_plot_reads():
    """The divisors come from one host_stats_many call, the samples from range
    queries -- the schema still has to match dcgm_timeseries'/running_timeseries'."""
    class Client:
        sampling_period = 60

        def query(self, query, at, timeout=None):
            # host_stats_many resolving the (mostly-constant) cpus/total_memory.
            if "cgroup_cpus" in query:
                return [{"metric": {"jobid": "100", "host": "node01:9100"}, "value": [at, "4"]}]
            if "cgroup_memory_total_bytes" in query:
                return [{"metric": {"jobid": "100", "host": "node01:9100"},
                         "value": [at, str(8 * GIB)]}]
            return []

        def query_range(self, query, start, end, step, timeout=None):
            if "cgroup_cpu_total_seconds" in query:
                return [{"metric": {"host": "node01:9100"}, "values": [[1000, "2"]]}]
            if "cgroup_memory_rss_bytes" in query:
                return [{"metric": {"host": "node01:9100"}, "values": [[1000, str(4 * GIB)]]}]
            return []

    jobs = {100: {"jobid": "100_6", "user": "alice", "start_epoch": 940, "elapsed_seconds": 60}}
    out = io.StringIO()
    _render_running_cpu(jobs, Client(), RenderOptions(), out)
    lines = out.getvalue().splitlines()
    assert lines[0] == "JOBID,USER,EPOCH,TIME,NODE,GPU,MODEL,CPU%,MEM%"
    assert lines[1].startswith("100_6,alice,1000,")
    # CPU% = 100*2/4 = 50; MEM% = 100*(4 GiB)/(8 GiB) = 50.
    assert lines[1].split(",")[-2:] == ["50", "50"]


def test_live_cpu_timeseries_warns_and_skips_a_job_with_no_divisor():
    class Client:
        sampling_period = 60

        def query(self, query, at, timeout=None):
            return []

        def query_range(self, query, start, end, step, timeout=None):
            return []

    jobs = {100: {"jobid": "100", "user": "alice", "start_epoch": 940, "elapsed_seconds": 60}}
    out, err = io.StringIO(), io.StringIO()
    import sys as _sys
    old_stderr, _sys.stderr = _sys.stderr, err
    try:
        _render_running_cpu(jobs, Client(), RenderOptions(), out)
    finally:
        _sys.stderr = old_stderr
    assert out.getvalue() == ""
    assert "no CPU/memory records" in err.getvalue()


def test_live_combined_timeseries_uses_the_schema_plot_reads():
    """running_timeseries's GPU rows, plus each one's node's CPU%/MEM% appended --
    the default running-job --ts view."""
    class Client:
        sampling_period = 60

        def query(self, query, at, timeout=None):
            if "cgroup_cpus" in query:
                return [{"metric": {"jobid": "1", "host": "node01:9100"}, "value": [at, "4"]}]
            if "cgroup_memory_total_bytes" in query:
                return [{"metric": {"jobid": "1", "host": "node01:9100"},
                         "value": [at, str(8 * GIB)]}]
            return []

        def query_range(self, query, start, end, step, timeout=None):
            if "cgroup_cpu_total_seconds" in query:
                return [{"metric": {"host": "node01:9100"}, "values": [[1000, "2"]]}]
            if "cgroup_memory_rss_bytes" in query:
                return [{"metric": {"host": "node01:9100"}, "values": [[1000, str(4 * GIB)]]}]
            return []

    jobs = {1: {"jobid": "100_6", "user": "alice", "node": "node01", "name": "train",
               "start_epoch": 940, "elapsed_seconds": 60}}
    gpus = {"GPU-a": Gpu("GPU-a", 1, "node01", 3, "GPU 3")}
    samples = {"GPU-a": {1000: {"duty": 90.0, "mem": 10.0, "memtot": 80.0}}}
    out = io.StringIO()
    options = RenderOptions()
    collected = ts.running_host(jobs, Client(), None, workers=1,
                                host_specs=options.cgroup_specs, warn=False)
    running_combined_timeseries(jobs, samples, gpus, DEFAULT_RUNNING_SPECS, collected,
                                options, out=out)
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("JOBID,USER,EPOCH,TIME,NODE,GPU,")
    assert lines[0].endswith("CPU%,MEM%")
    assert lines[1].startswith("100_6,alice,1000,")
    # GPU side ends ...,12.5 (GMEM% = 10/80); CPU side appends CPU%=100*2/4=50,
    # MEM%=100*(4 GiB)/(8 GiB)=50.
    assert lines[1].split(",")[-2:] == ["50", "50"]


def test_live_timeseries_keeps_mig_slices_distinct():
    jobs = {1: {"jobid": "100", "user": "a", "node": "node01", "name": "n"}}
    gpus = {"MIG-a": Gpu("MIG-a", 1, "node01", 0, "MIG 0.0"),
            "MIG-b": Gpu("MIG-b", 1, "node01", 0, "MIG 0.1")}
    samples = {"MIG-a": {1000: {"mem": 1.0}}, "MIG-b": {1000: {"mem": 2.0}}}
    out = io.StringIO()
    running_timeseries(jobs, samples, gpus, DEFAULT_RUNNING_SPECS, RenderOptions(), out=out)
    gpu_column = [row.split(",")[5] for row in out.getvalue().splitlines()[1:]]
    # Both share minor 0; without the .instance suffix plot would merge them.
    assert gpu_column == ["0.0", "0.1"]


def test_collect_instant_stores_by_uuid_and_derives_mem_percent():
    class Client:
        sampling_period = 60

        def query(self, query, at, timeout=None):
            if "nvidia_gpu_memory_used_bytes" in query:
                return [{"metric": {"uuid": "GPU-a"}, "value": [at, str(40 * GIB)]}]
            if "nvidia_gpu_memory_total_bytes" in query:
                return [{"metric": {"uuid": "GPU-a"}, "value": [at, str(80 * GIB)]}]
            return []

    gpus = {"GPU-a": Gpu("GPU-a", 7, "node01", 0, "GPU 0")}
    metrics = collect_instant(Client(), gpus, DEFAULT_RUNNING_SPECS, None)
    values = metrics[7]["GPU-a"]
    assert values["mem"] == 40.0 and values["memtot"] == 80.0
    assert values["gmempct"] == 50.0


# --- batched cgroup queries -------------------------------------------------

def test_host_stats_are_batched_across_jobs():
    """Four queries in total, not four per job.

    Per-job round trips made a partition-wide live view unusable: 8000 running
    jobs meant 32000 queries. These series are per-job, so one shared window
    cannot pull another job's samples in.
    """
    class Counting:
        sampling_period = 60

        def __init__(self):
            self.queries = []

        def query(self, query, at, timeout=None):
            self.queries.append(query)
            return [{"metric": {"host": "node%02d:9306" % j, "jobid": str(j)},
                     "value": [at, "1"]} for j in (1, 2, 3)]

    client = Counting()
    out = host_stats_many({1: 100, 2: 200, 3: 300}, 1000, client)
    assert len(client.queries) == 4          # one per host field, not 4 x 3 jobs
    assert set(out) == {1, 2, 3}             # demultiplexed on the jobid label
    # The shared window is the longest job's.
    assert all("[300s]" in q for q in client.queries)


def test_host_stats_many_ignores_unrequested_jobs():
    class Noisy:
        sampling_period = 60

        def query(self, query, at, timeout=None):
            return [{"metric": {"host": "node01:9306", "jobid": "999"},
                     "value": [at, "1"]}]

    assert host_stats_many({1: 100}, 1000, Noisy()) == {}


def test_host_stats_many_is_empty_without_jobs():
    class Boom:
        def query(self, *a, **kw):
            raise AssertionError("no jobs means no queries")

    assert host_stats_many({}, 1000, Boom()) == {}
