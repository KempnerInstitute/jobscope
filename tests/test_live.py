"""Tests for the live view: squeue parsing, GPU identity, and the blob synthesis."""

import io

import pytest

from jobscope.blob import GIB, blob_metrics
from jobscope.dcgm import ALL_SPECS, DEFAULT_SPECS, SPEC_BY_HEADER, window_query
from jobscope.errors import JobscopeError
from jobscope.live import (
    DEFAULT_LIVE_SPECS,
    EXTENDED_LIVE_SPECS,
    SQUEUE_FORMAT,
    Gpu,
    LiveSelection,
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
    specs_for,
    timeseries_step,
)
from jobscope.live_blob import synthesize_stats
from jobscope.report import RenderOptions, live_report, live_timeseries
from jobscope.sacct import JobRecord

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


# --- catalogs ---------------------------------------------------------------

def test_live_catalog_column_order_and_membership():
    assert [h for _k, h, _d in build_columns(DEFAULT_LIVE_SPECS)] == [
        "GPU%", "SM_ACT%", "OCC%", "TENSOR%", "DRAM%", "POWER_W", "GMEM_GB", "GMEM%"]


def test_gpu_utilization_is_always_present_in_the_live_view():
    # A running job has no blob, so this is the only place GPU% comes from; there
    # is deliberately no narrower catalog that could drop it.
    assert "GPU%" in [h for _k, h, _d in build_columns(DEFAULT_LIVE_SPECS)]
    assert "GPU%" in [h for _k, h, _d in build_columns(EXTENDED_LIVE_SPECS)]


def test_total_memory_is_queried_but_not_shown():
    # It exists only to derive MEM%.
    assert any(s.header == "GMEM_TOTAL_GB" for s in DEFAULT_LIVE_SPECS)
    assert "GMEM_TOTAL_GB" not in [h for _k, h, _d in build_columns(DEFAULT_LIVE_SPECS)]


def test_extended_catalog_excludes_delta_reduced_counters():
    # A delta needs two points, so it is meaningless in an instant snapshot.
    assert all(s.reducer != "delta" for s in EXTENDED_LIVE_SPECS)
    assert any(s.reducer == "delta" for s in ALL_SPECS), "fixture assumes one exists"


def test_specs_for_maps_the_view_names():
    assert specs_for("all") is EXTENDED_LIVE_SPECS
    assert specs_for(None) is DEFAULT_LIVE_SPECS


def test_live_and_dcgm_render_identical_columns():
    """The whole point of sharing the catalog: one job, one column set.

    A finished job and a running one must be described by the same columns, so the
    same eye (and the same script) reads both.
    """
    from jobscope.dcgm import columns_for
    assert build_columns(DEFAULT_LIVE_SPECS) == columns_for(DEFAULT_SPECS)


def test_gmem_percent_is_per_gpu_and_handles_a_missing_total():
    from jobscope.dcgm import DERIVED_COLUMNS
    mem_pct = DERIVED_COLUMNS[0].fn
    assert mem_pct({"mem": 20.0, "memtot": 80.0}) == 25.0
    assert mem_pct({"mem": 20.0, "memtot": 0}) is None
    assert mem_pct({"mem": None, "memtot": 80.0}) is None


# --- the runtime-window clip ------------------------------------------------

def test_clip_applies_to_nvidia_metrics_only():
    duty, smact = SPEC_BY_HEADER["GPU%"], SPEC_BY_HEADER["SM_ACT%"]
    # Same exporter as nvidia_gpu_jobId, so `and` can match on identical labels.
    assert clip_to_job(duty, 42) == "nvidia_gpu_jobId == 42"
    # DCGM carries different labels, so `and` never matches; the window alone bounds it.
    assert clip_to_job(smact, 42) is None


def test_clipped_window_query_is_well_formed():
    duty = SPEC_BY_HEADER["GPU%"]
    query = window_query(duty, ["GPU-a"], 900, clip=clip_to_job(duty, 42))
    assert query == ('avg_over_time((nvidia_gpu_duty_cycle{uuid=~"^(GPU-a)$"} '
                     'and nvidia_gpu_jobId == 42)[900s:])')


def test_unclipped_window_query_is_unchanged():
    smact = SPEC_BY_HEADER["SM_ACT%"]
    assert window_query(smact, ["GPU-a"], 900) == window_query(
        smact, ["GPU-a"], 900, clip=None)


# --- selection --------------------------------------------------------------

def test_selection_describe_mentions_the_runtime_floor():
    assert "1h" in LiveSelection(min_elapsed=3600).describe()
    assert LiveSelection(jobids=["1", "2"]).describe() == "2 job ID(s)"


# --- blob synthesis for running jobs ---------------------------------------

class BlobClient:
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


def test_synthesize_stats_feeds_blob_metrics():
    stats = synthesize_stats(_running(), BlobClient())
    # cpu = 100*150/(100*2) = 75, mem = 100*8/16 = 50, gpu = 70, gmem = 100*40/80 = 50.
    assert blob_metrics(stats) == (75, 50, 70, 50)


def test_synthesize_stats_queries_the_raw_job_id():
    # cgroup_* carries a real jobid label, but it holds the RAW per-element id:
    # array element 100_6 appears as jobid="12345". Using .jobid would find nothing.
    client = BlobClient()
    synthesize_stats(_running(), client)
    assert all("12345" in q for q in client.queries)
    assert not any("100_6" in q for q in client.queries)


def test_synthesize_stats_shapes_gpu_maps_by_minor_string():
    stats = synthesize_stats(_running(), BlobClient(minor="3"))
    node = stats["nodes"]["node01"]
    assert node["gpu_utilization"] == {"3": 70.0}
    assert node["gpu_used_memory"] == {"3": 40 * GIB}


def test_synthesize_stats_rounds_like_the_stored_blob():
    # blob_detail renders utilization with %g, which assumes stored precision.
    stats = synthesize_stats(_running(), BlobClient())
    node = stats["nodes"]["node01"]
    assert isinstance(node["cpus"], int)
    assert isinstance(node["used_memory"], int)
    assert node["gpu_utilization"]["0"] == round(node["gpu_utilization"]["0"], 1)


def test_synthesize_stats_skips_gpu_queries_for_a_cpu_only_job():
    client = BlobClient()
    stats = synthesize_stats(_running(gpus=0), client)
    assert not any("nvidia_gpu" in q for q in client.queries)
    assert blob_metrics(stats)[2:] == (None, None)   # no gpu%, no gmem%


def test_synthesize_stats_returns_empty_without_a_usable_window():
    assert synthesize_stats(_running(duration=None), BlobClient()) == {}
    assert synthesize_stats(_running(jobid_raw=""), BlobClient()) == {}


def test_synthesize_stats_degrades_instead_of_raising():
    class Broken:
        sampling_period = 60

        def query(self, *a, **kw):
            raise RuntimeError("prometheus is down")

    assert synthesize_stats(_running(), Broken()) == {}


def test_fill_running_leaves_a_finished_job_alone():
    from jobscope.live_blob import fill_running
    stored = {"total_time": 1, "nodes": {}}
    records = {"1": _running(state="COMPLETED", stats=stored)}
    assert fill_running(records, ["1"], BlobClient()) == 0
    assert records["1"].stats is stored


def test_fill_running_fills_only_unblobbed_running_jobs():
    from jobscope.live_blob import fill_running
    records = {"1": _running(), "2": _running(state="COMPLETED")}
    assert fill_running(records, ["1", "2"], BlobClient()) == 1
    assert records["1"].stats and not records["2"].stats


# --- rendering --------------------------------------------------------------

def _one_gpu_render(specs=None, average=False, csv=False):
    specs = specs or DEFAULT_LIVE_SPECS
    jobs = {1: {"jobid": "100_6", "user": "alice", "node": "node01", "name": "train",
                "start_epoch": 1000, "elapsed_seconds": 100}}
    gpus = {"GPU-a": Gpu("GPU-a", 1, "node01", 3, "GPU 3")}
    metrics = {1: {"GPU-a": {"duty": 93.0, "smact": 77.6, "mem": 18.4, "gmempct": 13.1}}}
    out = io.StringIO()
    live_report(jobs, metrics, gpus, specs, [("User", "alice")],
                RenderOptions(csv=csv), average=average, out=out)
    return out.getvalue()


def test_live_report_renders_a_row_per_gpu_with_missing_cells_dashed():
    text = _one_gpu_render()
    assert "100_6" in text and "GPU 3" in text
    assert "93" in text and "77.6" in text
    assert "-" in text          # OCC%/TENSOR%/DRAM%/POWER_W were not collected


def test_live_report_names_the_gpus_own_host():
    # Not the job's nodelist: one row is one GPU, possibly on a different node.
    assert "node01" in _one_gpu_render()


def test_live_report_row_for_a_job_with_no_gpu_samples():
    jobs = {1: {"jobid": "100", "user": "alice", "node": "node01,node02", "name": "cpujob"}}
    out = io.StringIO()
    live_report(jobs, {}, {}, DEFAULT_LIVE_SPECS, [], RenderOptions(), out=out)
    text = out.getvalue()
    assert "[no GPU data]" in text
    # Falls back to the nodelist when there is no GPU to name a host from.
    assert "node01,node02" in text


def test_live_report_csv_header_matches_the_table_columns():
    rows = [r for r in _one_gpu_render(csv=True).splitlines() if r]
    assert rows[1].startswith("JOBID,USER,NODE,NAME,GPU,GPU%")


def test_live_timeseries_uses_the_schema_plot_reads():
    # jobscope plot keys on EPOCH/TIME and groups series by (NODE, GPU), so this
    # header must stay in lockstep with dcgm_timeseries'.
    jobs = {1: {"jobid": "100_6", "user": "alice", "node": "node01", "name": "train"}}
    gpus = {"GPU-a": Gpu("GPU-a", 1, "node01", 3, "GPU 3")}
    samples = {"GPU-a": {1000: {"duty": 90.0, "mem": 10.0, "memtot": 80.0}}}
    out = io.StringIO()
    live_timeseries(jobs, samples, gpus, DEFAULT_LIVE_SPECS, RenderOptions(), out=out)
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("JOBID,EPOCH,TIME,NODE,GPU,")
    assert lines[1].startswith("100_6,1000,")
    # MEM% is recomputed per timestamp, so it tracks memory growth: 10/80 = 12.5.
    assert lines[1].endswith("12.5")


def test_live_timeseries_keeps_mig_slices_distinct():
    jobs = {1: {"jobid": "100", "user": "a", "node": "node01", "name": "n"}}
    gpus = {"MIG-a": Gpu("MIG-a", 1, "node01", 0, "MIG 0.0"),
            "MIG-b": Gpu("MIG-b", 1, "node01", 0, "MIG 0.1")}
    samples = {"MIG-a": {1000: {"mem": 1.0}}, "MIG-b": {1000: {"mem": 2.0}}}
    out = io.StringIO()
    live_timeseries(jobs, samples, gpus, DEFAULT_LIVE_SPECS, RenderOptions(), out=out)
    gpu_column = [row.split(",")[4] for row in out.getvalue().splitlines()[1:]]
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
    metrics = collect_instant(Client(), gpus, DEFAULT_LIVE_SPECS, None)
    values = metrics[7]["GPU-a"]
    assert values["mem"] == 40.0 and values["memtot"] == 80.0
    assert values["gmempct"] == 50.0
