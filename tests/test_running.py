"""Tests for the live view: squeue parsing, GPU identity, and the summary synthesis."""

import io

import pytest

from jobscope import dcgm, running
from jobscope import timeseries as ts
from jobscope.cpu import host_stats_many
from jobscope.dcgm import spec_named, window_query
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
    SQUEUE_FIELD_COUNT,
    SQUEUE_FORMAT,
    Gpu,
    RunningSelection,
    build_columns,
    clip_to_job,
    collect_instant,
    default_running_specs,
    discover_gpus,
    extended_running_specs,
    fetch_jobs,
    filter_by_elapsed,
    format_duration,
    gpu_labels,
    job_sort_key,
    note_missing_gpu_join,
    parse_duration,
    parse_squeue,
    parse_start_time,
    per_node_pooled,
    range_window,
    requested_gpus,
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

# squeue -o "%A|%i|%u|%N|%g|%j|%b|%C|%S|%a|%P": raw id, display id, user, nodelist,
# group, name, gres, cpus, start, account, partition. For array element 12345_6 the
# raw id differs -- and Prometheus keys on the raw one, which is why %A is read.
def squeue_line(raw="1", disp="1", user="alice", node="node01", group="g", name="n",
                gres="gpu:1", cpus="8", start="2026-07-24T09:00:00",
                account="kempner_lab", partition="gpu") -> str:
    """One squeue line, in :data:`SQUEUE_FORMAT` order.

    Built rather than spelled out, and checked against the format string: parse_squeue
    reads these positionally and *skips* a line with too few fields, so a literal that
    fell behind the format would silently drop every running job instead of failing.
    """
    fields = [raw, disp, user, node, group, name, gres, cpus, start, account, partition]
    assert len(fields) == SQUEUE_FIELD_COUNT, (
        "this helper builds %d fields; SQUEUE_FORMAT asks for %d"
        % (len(fields), SQUEUE_FIELD_COUNT))
    return "|".join(fields)


ARRAY_LINE = squeue_line("34843629", "34843528_6", "alice", "holygpu01", "kempner",
                         "train", "gpu:1", "16", "2026-07-24T09:00:00")
PLAIN_LINE = squeue_line("34622920", "34622920", "bob", "holygpu02", "kempner",
                         "infer", "gpu:2", "8", "2026-07-24T10:00:00")


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
    text = "\n".join(["JOBID|...", "", "too|few|fields",
                     squeue_line(raw="notanid"), PLAIN_LINE])
    assert list(parse_squeue(text)) == [34622920]


def test_squeue_format_asks_for_gres_not_the_group_id():
    """%b is tres-per-node; %G is the numeric group ID, which looks like data.

    Reading %G here yielded values such as 5137 in a field named "gpus" -- a
    plausible-looking number that is not a GPU count at all.
    """
    assert "%b" in SQUEUE_FORMAT and "%G" not in SQUEUE_FORMAT


def test_parse_squeue_reads_the_account_and_partition():
    """A running job has to carry the same identity a finished one does, or --show
    would answer for half a mixed selection."""
    job = parse_squeue(squeue_line(account="kempner_wharper_lab",
                                   partition="kempner_h100"))[1]
    assert job["account"] == "kempner_wharper_lab"
    assert job["partition"] == "kempner_h100"


def test_the_squeue_format_and_the_parser_agree_on_how_many_fields():
    """parse_squeue *skips* a short line, so a count that fell behind the format string
    would drop every running job rather than raise."""
    assert SQUEUE_FIELD_COUNT == len(SQUEUE_FORMAT.split("|"))
    assert SQUEUE_FORMAT.endswith("|%a|%P")     # appended, never inserted
    assert list(parse_squeue(squeue_line(raw="7", disp="7"))) == [7]


def test_parse_squeue_records_the_gres_request():
    assert parse_squeue(ARRAY_LINE)[34843629]["gres"] == "gpu:1"


def test_parse_squeue_keeps_the_whole_nodelist():
    # Splitting on "," would mangle a compressed range like holygpu8a[10402,10404].
    line = squeue_line(node="holygpu8a[10402,10404]", gres="gpu:2")
    assert parse_squeue(line)[1]["node"] == "holygpu8a[10402,10404]"


def test_parse_squeue_tolerates_a_missing_start_time():
    line = squeue_line(start="N/A")
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
    assert [h for _k, h, _d in build_columns(default_running_specs())] == [
        "GPU%", "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W", "GMEM_GB", "GMEM%"]


def test_gpu_utilization_is_always_present_in_the_live_view():
    # A running job has no jobstats summary, so this is the only place GPU% comes from; there
    # is deliberately no narrower catalog that could drop it.
    assert "GPU%" in [h for _k, h, _d in build_columns(default_running_specs())]
    assert "GPU%" in [h for _k, h, _d in build_columns(extended_running_specs())]


def test_total_memory_is_queried_but_not_shown():
    # It exists only to derive MEM%.
    assert any(s.header == "GMEM_TOTAL_GB" for s in default_running_specs())
    assert "GMEM_TOTAL_GB" not in [h for _k, h, _d in build_columns(default_running_specs())]


def test_extended_catalog_excludes_delta_reduced_counters():
    # A delta needs two points, so it is meaningless in an instant snapshot.
    assert all(s.reducer != "delta" for s in extended_running_specs())
    assert any(s.reducer == "delta" for s in dcgm.catalog().all_specs), "fixture assumes one exists"


def test_specs_for_maps_the_view_names():
    assert specs_for("all") == extended_running_specs()
    assert specs_for(None) == default_running_specs()


def test_live_and_dcgm_render_identical_columns():
    """The whole point of sharing the catalog: one job, one column set.

    A finished job and a running one must be described by the same columns, so the
    same eye (and the same script) reads both.
    """
    from jobscope.dcgm import columns_for
    assert build_columns(default_running_specs()) == columns_for(dcgm.catalog().default_specs)


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
    duty, smact = spec_named("duty"), dcgm.catalog().spec_by_header["SM_ACT%"]
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
    smact = dcgm.catalog().spec_by_header["SM_ACT%"]
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


def _squeue_cmd(monkeypatch, selection, stdout=""):
    """The command ``fetch_jobs`` builds for ``selection``."""
    seen = []

    def capture(cmd, *_a, **_kw):
        seen.append(cmd)
        return stdout

    monkeypatch.setattr(running, "run_capture", capture)
    fetch_jobs(selection, None)
    return seen[-1]


def test_every_filter_reaches_squeue(monkeypatch):
    """-A used to be parsed, packed into the Request, and then silently dropped.

    So `jobscope -A other_lab` reported every account of yours and said nothing
    about it -- and once the header restates the account, an unpushed filter would
    make the header a lie rather than merely incomplete.
    """
    cmd = _squeue_cmd(monkeypatch, RunningSelection(
        user="alice", account="kempner_dev", partition="kempner_h100"))
    assert cmd[cmd.index("-A") + 1] == "kempner_dev"
    assert cmd[cmd.index("-p") + 1] == "kempner_h100"
    assert cmd[cmd.index("-u") + 1] == "alice"


def test_an_unset_filter_reaches_squeue_as_nothing(monkeypatch):
    cmd = _squeue_cmd(monkeypatch, RunningSelection(user="alice"))
    assert "-A" not in cmd and "-p" not in cmd


def test_explicit_job_ids_bypass_the_account_filter(monkeypatch):
    """As they bypass every other one -- the IDs are the selection."""
    cmd = _squeue_cmd(monkeypatch,
                      RunningSelection(jobids=["1"], account="kempner_dev"),
                      stdout=squeue_line(raw="1", disp="1"))
    assert "-A" not in cmd and "-j" in cmd


def test_describe_filters_names_the_account_too():
    """It is a pushed filter like the others, so an empty result must name it.

    "no running jobs (user alice, running, longer than 10m)" sent someone looking at
    the runtime floor for an emptiness that -A caused.
    """
    sel = RunningSelection(account="kempner_dev", partition="kempner",
                           user="alice", min_elapsed=600)
    assert sel.describe_filters() == (
        "user alice, account kempner_dev, partition kempner, running, longer than 10m")


def test_widening_hints_cover_only_the_active_filters():
    assert RunningSelection(user="alice", partition="p", min_elapsed=600).widening_hints() == [
        "set --min-elapsed 0s to include jobs that just started",
        "drop -p to search every partition"]
    assert RunningSelection(user="alice", account="a", partition="p",
                            min_elapsed=600).widening_hints() == [
        "set --min-elapsed 0s to include jobs that just started",
        "drop -A to search every account",
        "drop -p to search every partition"]
    assert RunningSelection(user=None, partition=None, min_elapsed=0).widening_hints() == []


def test_the_user_filter_is_never_widened_by_a_hint():
    """-a lifts it, and -a is an administrator's flag taught in docs/admin.md.

    So the user filter contributes no hint even though it is the filter most often
    responsible for an empty selection, and a selection narrowed by nothing else
    offers no advice at all rather than advice most readers should not take.
    """
    for selection in (RunningSelection(user="alice", partition="p", min_elapsed=600),
                      RunningSelection(user="alice", partition=None, min_elapsed=0)):
        assert not [hint for hint in selection.widening_hints() if "-a" in hint]
    assert RunningSelection(user="alice", partition=None, min_elapsed=0).widening_hints() == []


def test_explicit_job_ids_keep_the_plain_description():
    """Job IDs bypass the filters, so naming them would be misleading."""
    sel = RunningSelection(jobids=["1"], user="alice", account="a", partition="p")
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
    running_timeseries(jobs, samples, gpus, default_running_specs(), RenderOptions(), out=out)
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
    running_combined_timeseries(jobs, samples, gpus, default_running_specs(), collected,
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
    running_timeseries(jobs, samples, gpus, default_running_specs(), RenderOptions(), out=out)
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
    metrics = collect_instant(Client(), gpus, default_running_specs(), None)
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


# --- a job holding GPUs that no card reports ---------------------------------
#
# discover_gpus matches a card to a job through the join series alone, so a stale
# exporter mapping leaves a job holding four H200s with every GPU cell blank. Measured
# when it happened here: 14 of 43 running jobs on one partition, all holding GPUs, none
# named by any card -- and the report said nothing, so it read as a fleet of idle jobs.

def _squeue_job(raw, gres):
    return {"jobid": str(raw), "user": "alice", "node": "n1", "group": "g",
            "name": "j", "gres": gres, "cpus": "4", "start_time": "now",
            "start_epoch": 1000, "elapsed_seconds": 600}


def test_a_gpu_job_no_card_reports_is_explained_not_left_blank(capsys):
    jobs = {1: _squeue_job(1, "gres/gpu:4"), 2: _squeue_job(2, "gres/gpu:1")}
    note_missing_gpu_join(jobs, {})
    err = capsys.readouterr().err
    assert "2 running job(s) hold GPUs that no card reports" in err
    # Names the series, so the reader knows what to go and look at.
    assert "nvidia_gpu_jobId" in err
    # And says which way to read a dash, which is the whole point.
    assert "not an idle job" in err


def test_a_cpu_only_job_gets_no_gpu_note(capsys):
    """It has no GPU to report on. Judged from the allocation (%b), never from what
    Prometheus returned -- asking Prometheus why Prometheus found nothing cannot tell a
    CPU-only job from a lost one."""
    note_missing_gpu_join({1: _squeue_job(1, "N/A")}, {})
    assert capsys.readouterr().err == ""


def test_no_note_when_every_gpu_job_is_served(capsys):
    jobs = {1: _squeue_job(1, "gres/gpu:1")}
    note_missing_gpu_join(jobs, {"GPU-a": Gpu("GPU-a", 1, "n1", 0, "GPU 0")})
    assert capsys.readouterr().err == ""


def test_only_the_unserved_jobs_are_counted(capsys):
    """One cause, one line -- but the count has to be the jobs actually affected."""
    jobs = {1: _squeue_job(1, "gres/gpu:1"), 2: _squeue_job(2, "gres/gpu:1"),
            3: _squeue_job(3, "N/A")}
    note_missing_gpu_join(jobs, {"GPU-a": Gpu("GPU-a", 1, "n1", 0, "GPU 0")})
    assert "1 running job(s) hold GPUs" in capsys.readouterr().err


def test_requested_gpus_reads_the_allocation_not_prometheus():
    assert requested_gpus({"gres": "gres/gpu:4"})
    assert requested_gpus({"gres": "gpu:nvidia_h200:2"})
    assert not requested_gpus({"gres": "N/A"})
    assert not requested_gpus({"gres": ""})
    assert not requested_gpus({})


# --- a card claiming a job Slurm places elsewhere ------------------------------
#
# discover_gpus matches on the join value alone, so a scrambled mapping hands a job a
# card on a node it does not hold -- and then reports another job's work under its name.
# Measured on one partition while the exporter was stale: 34 jobs were being shown a
# foreign card's numbers, with 0 false positives once this check was in place. A blank is
# a missing number; this was a wrong one.

def _join_client(rows):
    """A client answering the join query with (host, jobid, uuid, minor) rows."""
    class Client:
        def query(self, query, at, timeout=None):
            return [{"metric": {"host": h, "uuid": u, "minor_number": str(m),
                                "name": "NVIDIA H200"},
                     "value": [at, str(j)]} for h, j, u, m in rows]
    return Client()


def test_a_card_on_a_node_the_job_does_not_hold_is_dropped(capsys):
    jobs = {1: dict(_squeue_job(1, "gres/gpu:1"), node="n1")}
    got = discover_gpus(_join_client([("n2", 1, "GPU-a", 0)]), jobs, None)
    assert got == {}
    err = capsys.readouterr().err
    assert "ignored 1 GPU(s)" in err
    assert "another job's work" in err


def test_a_card_on_the_job_s_own_node_is_kept(capsys):
    jobs = {1: dict(_squeue_job(1, "gres/gpu:1"), node="n1")}
    got = discover_gpus(_join_client([("n1", 1, "GPU-a", 0)]), jobs, None)
    assert list(got) == ["GPU-a"]
    assert capsys.readouterr().err == ""


def test_a_multi_node_job_keeps_a_card_on_either_node(monkeypatch, capsys):
    monkeypatch.setattr("jobscope.running.expand_nodelist",
                        lambda text, timeout=None: ("n1", "n2"))
    jobs = {1: dict(_squeue_job(1, "gres/gpu:2"), node="n[1-2]")}
    got = discover_gpus(_join_client([("n1", 1, "GPU-a", 0), ("n2", 1, "GPU-b", 0)]),
                        jobs, None)
    assert sorted(got) == ["GPU-a", "GPU-b"]
    assert capsys.readouterr().err == ""


def test_an_unknown_node_list_keeps_the_card(capsys):
    """The fail-safe. A cross-check that cannot run has to weaken to no opinion, or a
    site that reports no node list loses every GPU column it had."""
    jobs = {1: dict(_squeue_job(1, "gres/gpu:1"), node="")}
    got = discover_gpus(_join_client([("n2", 1, "GPU-a", 0)]), jobs, None)
    assert list(got) == ["GPU-a"]
    assert capsys.readouterr().err == ""


def test_an_unexpandable_node_list_keeps_the_card(monkeypatch, capsys):
    """expand_nodelist returns () when scontrol cannot help; that is 'cannot tell'."""
    monkeypatch.setattr("jobscope.running.expand_nodelist",
                        lambda text, timeout=None: ())
    jobs = {1: dict(_squeue_job(1, "gres/gpu:1"), node="weird[")}
    assert list(discover_gpus(_join_client([("n2", 1, "GPU-a", 0)]), jobs, None)) == ["GPU-a"]
    assert capsys.readouterr().err == ""


def test_a_fully_qualified_host_label_still_matches(capsys):
    """host_of strips a port but not a domain, and Slurm's node list carries neither.
    Comparing them unnormalised would drop every card at such a site."""
    jobs = {1: dict(_squeue_job(1, "gres/gpu:1"), node="n1")}
    got = discover_gpus(_join_client([("n1.cluster.example.edu", 1, "GPU-a", 0)]),
                        jobs, None)
    assert list(got) == ["GPU-a"]
    assert capsys.readouterr().err == ""


def test_the_dropped_count_is_the_cards_not_the_jobs(capsys):
    jobs = {1: dict(_squeue_job(1, "gres/gpu:2"), node="n1")}
    discover_gpus(_join_client([("n2", 1, "GPU-a", 0), ("n3", 1, "GPU-b", 1)]),
                  jobs, None)
    assert "ignored 2 GPU(s)" in capsys.readouterr().err


# --- the second job-to-card mapping -------------------------------------------
#
# dcgm-exporter publishes the job id as an `hpc_job` label on its own series when its HPC
# job mapping is configured -- a mapping written from Slurm's side rather than inferred
# from the processes on a card, so it fails independently of nvidia_gpu_jobId. When that
# join froze here, every job started afterwards resolved no card at all and there was
# nothing to fall back on. These are synthetic: the label is not published on this
# cluster, so the happy path has never run against real data.

def _labelled_client(join_rows, labelled_rows, label="hpc_job", boom=False):
    """A client answering the value-based join and the label-based series separately."""
    class Client:
        def query(self, query, at, timeout=None):
            if label in query:
                if boom:
                    raise RuntimeError("fallback query failed")
                return [{"metric": {"UUID": u, "gpu": str(g), "host": h,
                                    "modelName": "NVIDIA H200", label: str(j)},
                         "value": [at, "1"]} for h, j, u, g in labelled_rows]
            return [{"metric": {"host": h, "uuid": u, "minor_number": str(m),
                                "name": "NVIDIA H200"}, "value": [at, str(j)]}
                    for h, j, u, m in join_rows]
    return Client()


@pytest.fixture
def hpc_job(monkeypatch):
    """Enable the label-based mapping, as [site] gpu_job_label would."""
    monkeypatch.setattr("jobscope.config.gpu_job_label", lambda: "hpc_job")
    monkeypatch.setattr("jobscope.config.gpu_job_label_series",
                        lambda: "DCGM_FI_PROF_SM_ACTIVE")


def test_without_the_label_configured_nothing_extra_is_queried(capsys):
    """Every site today, including the one this was written on. A label nothing
    publishes matches nothing, so paying a round trip to learn that is not worth it."""
    asked = []

    class Client:
        def query(self, query, at, timeout=None):
            asked.append(query)
            return []

    jobs = {1: dict(_squeue_job(1, "gres/gpu:1"), node="n1")}
    discover_gpus(Client(), jobs, None)
    assert asked == ["nvidia_gpu_jobId"]
    assert capsys.readouterr().err == ""


def test_a_job_the_join_missed_is_recovered_through_the_label(hpc_job, capsys):
    """The case that would have saved this cluster's report."""
    jobs = {1: dict(_squeue_job(1, "gres/gpu:1"), node="n1")}
    got = discover_gpus(_labelled_client([], [("n1", 1, "GPU-a", 0)]), jobs, None)
    assert list(got) == ["GPU-a"]
    assert got["GPU-a"].host == "n1" and got["GPU-a"].model == "NVIDIA H200"
    err = capsys.readouterr().err
    assert "1 job(s) resolved through the hpc_job label" in err


def test_the_fallback_is_not_consulted_for_jobs_the_join_answered(hpc_job, capsys):
    """A mapping that is merely behind still answers correctly for older jobs, so its
    answers are not replaced -- only the gaps are asked again."""
    jobs = {1: dict(_squeue_job(1, "gres/gpu:1"), node="n1")}
    got = discover_gpus(_labelled_client([("n1", 1, "GPU-join", 0)],
                                         [("n1", 1, "GPU-label", 0)]), jobs, None)
    assert list(got) == ["GPU-join"]
    assert capsys.readouterr().err == ""


def test_the_fallback_is_per_job_not_wholesale(hpc_job, capsys):
    jobs = {1: dict(_squeue_job(1, "gres/gpu:1"), node="n1"),
            2: dict(_squeue_job(2, "gres/gpu:1"), node="n2")}
    got = discover_gpus(
        _labelled_client([("n1", 1, "GPU-a", 0)],
                         [("n1", 1, "GPU-dupe", 0), ("n2", 2, "GPU-b", 0)]), jobs, None)
    assert sorted(got) == ["GPU-a", "GPU-b"]      # job 1 keeps the join's card
    assert "1 job(s) resolved" in capsys.readouterr().err


def test_a_fallback_card_on_the_wrong_node_is_still_dropped(hpc_job, capsys):
    """A second mapping is not a more trusted one; the cross-check applies to both."""
    jobs = {1: dict(_squeue_job(1, "gres/gpu:1"), node="n1")}
    got = discover_gpus(_labelled_client([], [("n2", 1, "GPU-a", 0)]), jobs, None)
    assert got == {}
    assert "resolved through" not in capsys.readouterr().err


def test_a_failing_fallback_leaves_the_primary_answer_standing(hpc_job, capsys):
    jobs = {1: dict(_squeue_job(1, "gres/gpu:1"), node="n1")}
    got = discover_gpus(_labelled_client([("n1", 1, "GPU-a", 0)], [], boom=True),
                        jobs, None)
    assert list(got) == ["GPU-a"]


def test_dcgm_label_names_are_read_not_nvml_s(hpc_job):
    """DCGM writes UUID/gpu/modelName where nvml writes uuid/minor_number/name -- three
    of the four differ, so the reader cannot be shared."""
    from jobscope.dcgm import gpu_from_series
    got = gpu_from_series({"UUID": "GPU-a", "gpu": "3", "host": "n1",
                                    "modelName": "NVIDIA H200"})
    assert got == {"uuid": "GPU-a", "node": "n1", "minor": "3", "model": "NVIDIA H200"}
    assert gpu_from_series({"gpu": "0"}) is None      # no UUID, no card


# --- pooling to a node on the running path --------------------------------------

def test_per_node_pooled_reduces_across_uuids_on_one_host():
    """Pooled from the UUID-keyed metrics, not from per_gpu_by_node_minor's output --
    that keys by (node, minor), which MIG siblings share."""
    smact = next(s for s in dcgm.catalog().default_specs if s.header == "SM_ACT%")
    gpus = {"GPU-a": Gpu("GPU-a", 1, "n1", 0, "GPU 0", "H200"),
            "GPU-b": Gpu("GPU-b", 1, "n1", 1, "GPU 1", "H200"),
            "GPU-c": Gpu("GPU-c", 1, "n2", 0, "GPU 0", "H200")}
    metrics = {1: {"GPU-a": {smact.key: 10.0}, "GPU-b": {smact.key: 30.0},
                   "GPU-c": {smact.key: 90.0}}}
    got = per_node_pooled(metrics, gpus, [smact])
    assert got[1]["n1"]["SM_ACT%"] == 20.0     # mean of the two cards on n1
    assert got[1]["n2"]["SM_ACT%"] == 90.0


def test_per_node_pooled_keeps_mig_siblings_apart():
    """Two instances of one card share (node, minor); reducing per_gpu would see one."""
    smact = next(s for s in dcgm.catalog().default_specs if s.header == "SM_ACT%")
    gpus = {"MIG-a": Gpu("MIG-a", 1, "n1", 0, "MIG 0.0", "A100"),
            "MIG-b": Gpu("MIG-b", 1, "n1", 0, "MIG 0.1", "A100")}
    metrics = {1: {"MIG-a": {smact.key: 10.0}, "MIG-b": {smact.key: 30.0}}}
    assert per_node_pooled(metrics, gpus, [smact])[1]["n1"]["SM_ACT%"] == 20.0


def test_a_mixed_model_node_reports_no_model_so_the_power_floor_is_global():
    """POWER_W's floor is per architecture and a mixed node has no single answer, so
    empty -- which falls back to the global floor -- is the honest reading."""
    from jobscope.dcgm import MODEL_KEY
    smact = next(s for s in dcgm.catalog().default_specs if s.header == "SM_ACT%")
    same = {"a": Gpu("a", 1, "n1", 0, "GPU 0", "H200"),
            "b": Gpu("b", 1, "n1", 1, "GPU 1", "H200")}
    mixed = {"a": Gpu("a", 1, "n1", 0, "GPU 0", "H200"),
             "b": Gpu("b", 1, "n1", 1, "GPU 1", "V100")}
    metrics = {1: {"a": {smact.key: 1.0}, "b": {smact.key: 1.0}}}
    assert per_node_pooled(metrics, same, [smact])[1]["n1"][MODEL_KEY] == "H200"
    assert per_node_pooled(metrics, mixed, [smact])[1]["n1"][MODEL_KEY] == ""
