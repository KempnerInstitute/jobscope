"""Tests for the selection layer that hides sacct and squeue behind one interface."""

import dataclasses
import io
import re
import time

import pytest

from jobscope import dcgm
from jobscope import select as select_mod
from jobscope.errors import JobscopeError
from jobscope.job_ave_stats import needs_fill
from jobscope.report import RenderOptions
from jobscope.rows import build_rows
from jobscope.select import (
    FINISHED,
    JOBIDS,
    RUNNING,
    Request,
    emit_timeseries,
    jobstats_specs,
    resolve,
    sacct_selection,
)
from jobscope.slurm import TIMESTAMP_FORMAT, JobRecord

GIB = 1024 ** 3


def _cfg():
    from jobscope import config
    return config.get_config()


# --- the sacct branch -------------------------------------------------------

def test_days_becomes_a_real_window():
    """-D N must reach the query, not just the header.

    select_jobs falls back to `now-30days` when starttime is unset, so a Selection
    carrying only `days` scanned a month while the header said "last 1 day" -- slow,
    and silently wrong. Assert the window, never the intermediate attribute.
    """
    selection = sacct_selection(Request(mode=FINISHED, days=1, user="alice"))
    assert selection.starttime and selection.endtime
    start = time.mktime(time.strptime(selection.starttime, TIMESTAMP_FORMAT))
    end = time.mktime(time.strptime(selection.endtime, TIMESTAMP_FORMAT))
    assert 0.9 * 86400 <= end - start <= 1.1 * 86400
    # days is kept as well, because the header renders it as "last 1 day".
    assert selection.days == 1


def test_days_window_scales():
    selection = sacct_selection(Request(mode=FINISHED, days=7, user="alice"))
    start = time.mktime(time.strptime(selection.starttime, TIMESTAMP_FORMAT))
    end = time.mktime(time.strptime(selection.endtime, TIMESTAMP_FORMAT))
    assert 6.9 * 86400 <= end - start <= 7.1 * 86400


def test_starttime_alone_closes_at_the_next_midnight():
    """-S DATE selects that calendar day, not "from then until now"."""
    selection = sacct_selection(Request(mode=FINISHED, starttime="2026-07-15", user="alice"))
    assert selection.starttime == "2026-07-15"
    assert selection.endtime == "2026-07-16T00:00:00"


def test_explicit_window_is_left_alone():
    selection = sacct_selection(Request(mode=FINISHED, starttime="2026-07-15",
                                        endtime="2026-07-20", user="alice"))
    assert (selection.starttime, selection.endtime) == ("2026-07-15", "2026-07-20")


def test_lastn_leaves_its_window_to_the_ladder():
    """-N is resolved at query time, not here.

    sacct has no "last N", so a window must be scanned and trimmed. Fixing one here
    would mean scanning the whole default lookback to find a job or two; select_jobs
    instead widens a rung at a time and writes back what it settled on.
    """
    selection = sacct_selection(Request(mode=FINISHED, lastn=5, user="alice"))
    assert selection.lastn == 5
    assert selection.starttime is None and selection.endtime is None


def test_sacct_selection_carries_every_filter():
    request = Request(mode=FINISHED, days=3, lastn=None, user="alice",
                      account="kempner_lab", partition="kempner", state="failed")
    selection = sacct_selection(request)
    assert selection.days == 3 and selection.user == "alice"
    assert selection.account == "kempner_lab" and selection.partition == "kempner"
    assert selection.state == "failed"


def test_sacct_selection_passes_all_users_through():
    selection = sacct_selection(Request(mode=FINISHED, all_users=True, user=None))
    assert selection.all_users and selection.user is None


def test_historical_yields_one_chunk_per_batch(monkeypatch, gpu_record, stream_window):
    rec2 = dataclasses.replace(gpu_record, jobid="101")
    records = {"100": gpu_record, "101": rec2}
    stream_window([(["100"], records), (["101"], records)])
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, t: object())
    monkeypatch.setattr(select_mod, "compute_dcgm",
                        lambda r, ids, *a, **k: {j: ({"SM_ACT%": 1.0}, {}) for j in ids})

    selected = resolve(Request(mode=FINISHED, user="alice"), _cfg(), None, 1, dcgm.catalog().default_specs)
    chunks = list(selected.chunks)
    assert [ids for ids, _r, _d in chunks] == [["100"], ["101"]]
    # Each chunk arrives with its own metrics attached, so rendering can stream.
    assert all(dcgm for _i, _r, dcgm in chunks)


def test_historical_skips_prometheus_without_specs(monkeypatch, gpu_record, stream_window):
    """`--cpu` over finished jobs must stay offline."""
    stream_window([(["100"], {"100": gpu_record})])
    monkeypatch.setattr(select_mod, "fetch", lambda i, t: {"100": gpu_record})

    def boom(*a, **k):
        raise AssertionError("no Prometheus client for a --cpu report")

    monkeypatch.setattr(select_mod, "client_from_config", boom)
    monkeypatch.setattr(select_mod, "compute_dcgm", boom)
    selected = resolve(Request(mode=FINISHED, user="alice"), _cfg(), None, 1, None)
    assert [d for _i, _r, d in selected.chunks] == [{}]


def test_explicit_jobids_do_not_stream(monkeypatch, cpu_record):
    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: (["111"], "1 job ID(s)"))
    monkeypatch.setattr(select_mod, "fetch", lambda i, t: {"111": cpu_record})

    def no_stream(*a, **k):
        raise AssertionError("explicit IDs render in one pass")

    monkeypatch.setattr(select_mod, "fetch_chunks", no_stream)
    selected = resolve(Request(mode=JOBIDS, jobids=["111"], user="alice"),
                       _cfg(), None, 1, None)
    assert [ids for ids, _r, _d in selected.chunks] == [["111"]]


def test_emit_timeseries_dispatches_to_cpu_for_a_finished_request(monkeypatch, gpu_record):
    """--cpu --ts on a historical selection must skip the GPU/DCGM emitter."""
    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: (["100"], "x"))
    monkeypatch.setattr(select_mod, "fetch", lambda i, t: {"100": gpu_record})
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, t: object())
    calls = []
    monkeypatch.setattr(select_mod, "cpu_timeseries",
                        lambda *a, **k: calls.append(("cpu", a, k)))

    def boom(*a, **k):
        raise AssertionError("the GPU/DCGM emitter must not run for a --cpu --ts request")
    monkeypatch.setattr(select_mod, "dcgm_timeseries", boom)

    emit_timeseries(Request(mode=FINISHED, user="alice"), _cfg(), None, 1, dcgm.catalog().default_specs,
                    None, RenderOptions(view="cpu"))
    assert len(calls) == 1


def test_emit_timeseries_still_dispatches_gpu_when_not_cpu_view(monkeypatch, gpu_record):
    """The existing GPU/DCGM path must stay untouched for every other view."""
    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: (["100"], "x"))
    monkeypatch.setattr(select_mod, "fetch", lambda i, t: {"100": gpu_record})
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, t: object())
    calls = []
    monkeypatch.setattr(select_mod, "dcgm_timeseries",
                        lambda *a, **k: calls.append(("dcgm", a, k)))

    def boom(*a, **k):
        raise AssertionError("the CPU emitter must not run for a plain --ts request")
    monkeypatch.setattr(select_mod, "cpu_timeseries", boom)

    emit_timeseries(Request(mode=FINISHED, user="alice"), _cfg(), None, 1, dcgm.catalog().default_specs,
                    None, RenderOptions(view="all"))
    assert len(calls) == 1


def test_emit_timeseries_dispatches_to_cpu_for_a_running_request(monkeypatch):
    """--cpu --ts on the live selection must skip GPU discovery entirely."""
    monkeypatch.setattr(select_mod, "fetch_jobs", lambda sel, t: dict(JOBS))
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, t: object())
    calls = []
    monkeypatch.setattr(select_mod, "running_cpu_timeseries",
                        lambda *a, **k: calls.append(("running_cpu", a, k)))
    # The collector is stubbed too: unlike the finished-job ones it is not a
    # generator, so it would reach for the dummy client's sampling_period.
    monkeypatch.setattr(select_mod.timeseries, "running_host", lambda *a, **k: {})

    def boom(*a, **k):
        raise AssertionError("GPU discovery must not run for a --cpu --ts request")
    monkeypatch.setattr(select_mod, "discover_gpus", boom)
    monkeypatch.setattr(select_mod, "collect_timeseries", boom)

    emit_timeseries(Request(mode=RUNNING, user="alice"), _cfg(), None, 1, dcgm.catalog().default_specs,
                    None, RenderOptions(view="cpu"))
    assert len(calls) == 1


def test_emit_timeseries_dispatches_to_combined_for_a_finished_request(monkeypatch, gpu_record):
    """The new default (combined) view must skip both the GPU-only and CPU-only
    emitters for a historical selection."""
    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: (["100"], "x"))
    monkeypatch.setattr(select_mod, "fetch", lambda i, t: {"100": gpu_record})
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, t: object())
    calls = []
    monkeypatch.setattr(select_mod, "combined_timeseries",
                        lambda *a, **k: calls.append(("combined", a, k)))

    def boom(*a, **k):
        raise AssertionError("neither GPU-only nor CPU-only should run when combined")
    monkeypatch.setattr(select_mod, "dcgm_timeseries", boom)
    monkeypatch.setattr(select_mod, "cpu_timeseries", boom)

    emit_timeseries(Request(mode=FINISHED, user="alice"), _cfg(), None, 1, dcgm.catalog().default_specs,
                    None, RenderOptions(view="all", combined=True))
    assert len(calls) == 1


def test_emit_timeseries_combined_wins_over_cpu_only_when_both_are_set(monkeypatch, gpu_record):
    """--cpu --dcgm resolves to view=="cpu" AND combined=True (cli.py's truth
    table) -- combined must win, not the cpu-only branch."""
    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: (["100"], "x"))
    monkeypatch.setattr(select_mod, "fetch", lambda i, t: {"100": gpu_record})
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, t: object())
    calls = []
    monkeypatch.setattr(select_mod, "combined_timeseries",
                        lambda *a, **k: calls.append(("combined", a, k)))

    def boom(*a, **k):
        raise AssertionError("cpu-only must not run when combined is also set")
    monkeypatch.setattr(select_mod, "cpu_timeseries", boom)

    emit_timeseries(Request(mode=FINISHED, user="alice"), _cfg(), None, 1, dcgm.catalog().default_specs,
                    None, RenderOptions(view="cpu", combined=True))
    assert len(calls) == 1


def test_emit_timeseries_dispatches_to_combined_for_a_running_request(monkeypatch):
    """The new default (combined) view must skip running_cpu_timeseries for a live
    selection, and still do GPU discovery (unlike cpu-only)."""
    _patch_running(monkeypatch)
    calls = []
    monkeypatch.setattr(select_mod, "running_combined_timeseries",
                        lambda *a, **k: calls.append(("running_combined", a, k)))

    def boom(*a, **k):
        raise AssertionError("running_cpu_timeseries must not run when combined")
    monkeypatch.setattr(select_mod, "running_cpu_timeseries", boom)

    emit_timeseries(Request(mode=RUNNING, user="alice"), _cfg(), None, 1, dcgm.catalog().default_specs,
                    None, RenderOptions(view="all", combined=True))
    assert len(calls) == 1


def test_no_matching_jobs_returns_none(capsys, stream_window):
    stream_window([])
    assert resolve(Request(mode=FINISHED, user="nobody"), _cfg(), None, 1, None) is None
    assert "No matching jobs" in capsys.readouterr().err


def test_no_matching_jobs_names_all_users(capsys, stream_window):
    stream_window([])
    resolve(Request(mode=FINISHED, all_users=True, user=None), _cfg(), None, 1, None)
    assert "all users" in capsys.readouterr().err


# --- the squeue branch ------------------------------------------------------

class FakeRunningClient:
    """Serves the series the live branch needs, and counts the queries."""

    sampling_period = 60

    def __init__(self):
        self.queries = []

    def query(self, query, at, timeout=None):
        self.queries.append(query)
        if "nvidia_gpu_jobId" in query:
            return [{"metric": {"uuid": "U0", "host": "node01:9445", "minor_number": "2"},
                     "value": [at, "7.0e+00"]}]
        for name, value in (("nvidia_gpu_duty_cycle", 90),
                            ("nvidia_gpu_memory_used_bytes", 40 * GIB),
                            ("nvidia_gpu_memory_total_bytes", 80 * GIB)):
            if name in query:
                return [{"metric": {"uuid": "U0"}, "value": [at, str(value)]}]
        # The other candidate for GPU%, serving the same 90 so these tests stay about
        # the reconstruction rather than about which exporter the preference picked.
        if "DCGM_FI_DEV_GPU_UTIL" in query:
            return [{"metric": {"UUID": "U0"}, "value": [at, "90"]}]
        for name, value in (("cgroup_cpus", 2), ("cgroup_cpu_total_seconds", 150),
                            ("cgroup_memory_rss_bytes", 8 * GIB),
                            ("cgroup_memory_total_bytes", 16 * GIB)):
            if name in query:
                return [{"metric": {"host": "node01:9306", "jobid": "7"},
                         "value": [at, str(value)]}]
        if "DCGM_FI_PROF_SM_ACTIVE" in query:
            return [{"metric": {"UUID": "U0"}, "value": [at, "0.80"]}]
        return []


JOBS = {7: {"jobid": "7", "user": "alice", "node": "node01", "name": "train",
            "start_epoch": 1000, "elapsed_seconds": 100}}


def _patch_running(monkeypatch, client=None):
    client = client or FakeRunningClient()
    monkeypatch.setattr(select_mod, "fetch_jobs", lambda sel, t: dict(JOBS))
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, t: client)
    return client


def test_running_yields_records_that_look_finished(monkeypatch):
    """The squeue branch must be indistinguishable from sacct downstream."""
    _patch_running(monkeypatch)
    selected = resolve(Request(mode=RUNNING, user="alice"), _cfg(), None, 1, dcgm.catalog().default_specs)
    (jobids, records, dcgm_data), = list(selected.chunks)
    assert jobids == ["7"]
    record = records["7"]
    assert record.state == "RUNNING" and record.jobid_raw == "7"
    # A reconstructed summary, so jobstats_metrics works exactly as for a stored one:
    # cpu = 100*150/(100*2) = 75, mem = 100*8/16 = 50, gpu = 90, gmem = 40/80 = 50.
    from jobscope.jobstats import jobstats_metrics
    assert jobstats_metrics(record.stats, record.gpus).known() == {
        "CPU%": 75, "MEM%": 50, "GPU%": 90, "GMEM%": 50}
    assert dcgm_data["7"][0]["SM_ACT%"] == 80.0


def test_running_provides_the_per_gpu_metrics(monkeypatch):
    _patch_running(monkeypatch)
    selected = resolve(Request(mode=RUNNING, user="alice"), _cfg(), None, 1, dcgm.catalog().default_specs)
    (_ids, _records, dcgm_data), = list(selected.chunks)
    per_gpu = dcgm_data["7"][1]
    # Keyed (node, minor) as the detail renderer and the jobstats summary both expect.
    assert per_gpu[("node01", "2")]["SM_ACT%"] == 80.0


def test_running_without_specs_still_builds_the_summary(monkeypatch):
    """--cpu needs CPU%, which comes from the reconstructed summary."""
    client = _patch_running(monkeypatch)
    selected = resolve(Request(mode=RUNNING, user="alice"), _cfg(), None, 1, None)
    (_ids, records, _d), = list(selected.chunks)
    from jobscope.jobstats import jobstats_metrics
    summary = jobstats_metrics(records["7"].stats, records["7"].gpus)
    assert (summary.value("CPU%"), summary.value("MEM%")) == (75, 50)
    # ... but it does not pay for the DCGM profiling queries it will not print. Tested
    # against the PROF catalog specifically, not "DCGM_FI" at large: GPU% may itself be
    # served by DCGM_FI_DEV_GPU_UTIL, which is one of the three jobstats columns rather
    # than a profiling metric this view declined to print.
    assert not any("DCGM_FI_PROF" in q for q in client.queries)
    specs = jobstats_specs()
    assert specs and all(
        s.column in ("GPU%", "GMEM_GB", "GMEM_TOTAL_GB") for s in specs)


def test_running_context_names_the_owner_for_explicit_ids(monkeypatch):
    _patch_running(monkeypatch)
    selected = resolve(Request(mode=RUNNING, jobids=["7"], user=None),
                       _cfg(), None, 1, dcgm.catalog().default_specs)
    assert ("User", "alice") in selected.context


def test_running_context_says_all_users(monkeypatch):
    _patch_running(monkeypatch)
    selected = resolve(Request(mode=RUNNING, all_users=True, user=None),
                       _cfg(), None, 1, dcgm.catalog().default_specs)
    assert ("User", "(all users)") in selected.context


def test_running_no_jobs_returns_none(monkeypatch, capsys):
    monkeypatch.setattr(select_mod, "fetch_jobs", lambda sel, t: {})
    assert resolve(Request(mode=RUNNING, user="alice"), _cfg(), None, 1, None) is None
    assert "No running jobs match" in capsys.readouterr().err


def test_the_empty_running_message_names_the_filters_and_how_to_widen(monkeypatch, capsys):
    """An empty result is when the filters matter most, and the context block is gone.

    Reported from the field: `jobscope -p kempner` printed only "running, longer
    than 10m", which reads as an idle partition. Twenty jobs were running there;
    none were the caller's.
    """
    monkeypatch.setattr(select_mod, "fetch_jobs", lambda sel, t: {})
    resolve(Request(mode=RUNNING, user="alice", partition="kempner"),
            _cfg(), None, 1, None)
    err = capsys.readouterr().err
    assert "user alice" in err and "partition kempner" in err
    assert "drop -p to search every partition" in err
    # The filter that actually excluded them is the user filter, and the message still
    # names it -- but the flag that lifts it is -a, which belongs to docs/admin.md.
    assert "-a" not in err and "--all-users" not in err


def test_both_branches_yield_the_same_chunk_shape(monkeypatch, gpu_record):
    """The contract that lets one renderer serve both sources."""
    _patch_running(monkeypatch)
    live = list(resolve(Request(mode=RUNNING, user="alice"),
                        _cfg(), None, 1, dcgm.catalog().default_specs).chunks)

    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: (["100"], "x"))
    monkeypatch.setattr(select_mod, "fetch", lambda i, t: {"100": gpu_record})
    monkeypatch.setattr(select_mod, "compute_dcgm",
                        lambda r, ids, *a, **k: {"100": select_mod.JobGpuData()})
    past = list(resolve(Request(mode=JOBIDS, jobids=["100"], user="alice"),
                        _cfg(), None, 1, dcgm.catalog().default_specs).chunks)

    for chunks in (live, past):
        (jobids, records, dcgm_data), = chunks
        assert isinstance(jobids, list) and isinstance(records, dict)
        assert isinstance(dcgm_data, dict)
        jid = jobids[0]
        assert records[jid].jobid_raw          # both carry the raw ID for the joins
        # Three levels, the same three from either source -- per_node cannot be
        # recovered downstream, so both branches have to supply it.
        overall, per_gpu, per_node = dcgm_data[jid]
        assert isinstance(overall, dict) and isinstance(per_gpu, dict)
        assert isinstance(per_node, dict)


def _record(state, stats, jobid="1"):
    """A JobRecord in one line, for the needs_fill()/--no-jobstats checks."""
    return JobRecord(jobid=jobid, state=state, name="j", runtime="00:10:00",
                     nodes="1", gpus=1, stats=stats, start=0, end=600,
                     duration=600, jobid_raw=jobid, cluster="", user="u")


def test_no_jobstats_forces_the_prometheus_path_for_finished_jobs():
    """A finished job carries a stored summary, so needs_fill() normally leaves it
    alone. --no-summary is what makes the two sources comparable on the same job."""
    finished = _record("COMPLETED", {"total_time": 600, "nodes": {}})
    assert needs_fill(finished) is False
    assert needs_fill(finished, force=True) is True


def test_a_running_job_with_no_jobstats_is_filled_either_way():
    running = _record("RUNNING", {})
    assert needs_fill(running) is True and needs_fill(running, force=True) is True


def test_no_jobstats_without_an_endpoint_is_an_error_not_a_silent_table_of_dashes():
    """Without --no-jobstats a missing endpoint degrades to the stored summary with a note.
    With it there is nothing to fall back on, so going quiet would print a table of
    dashes and no explanation."""
    records = {"1": _record("COMPLETED", {"total_time": 1, "nodes": {}})}
    broken = dataclasses.replace(_cfg(), prometheus_url=None,
                                 site_jobstats_config_path="/nonexistent")
    with pytest.raises(JobscopeError) as exc:
        select_mod._fill_running(records, ["1"], broken, None, 1, None, force=True)
    assert "--no-jobstats" in str(exc.value)


# --- narrowing the summary: --nodename / --gpuid on a view with no row per unit ---

_TWO_NODE_STATS = {
    "total_time": 100,
    "nodes": {
        "n1": {"cpus": 2, "total_time": 100, "used_memory": 4, "total_memory": 8,
               "gpu_utilization": {"0": 90.0, "1": 90.0},
               "gpu_used_memory": {"0": 8, "1": 8},
               "gpu_total_memory": {"0": 10, "1": 10}},
        "n2": {"cpus": 2, "total_time": 0, "used_memory": 1, "total_memory": 8,
               "gpu_utilization": {"0": 10.0, "1": 10.0},
               "gpu_used_memory": {"0": 1, "1": 1},
               "gpu_total_memory": {"0": 10, "1": 10}},
    },
}


def _narrowed(nodename=None, gpu_ids=()):
    from jobscope.jobstats import jobstats_metrics
    from jobscope.select import _narrow_records

    record = JobRecord(jobid="1", state="COMPLETED", name="j", runtime="00:01:40",
                       nodes="2", gpus=4, stats=_TWO_NODE_STATS, start=0, end=100,
                       duration=100, jobid_raw="1", cluster="c", user="alice")
    records = {"1": record}
    _narrow_records(records, ["1"], nodename, gpu_ids)
    got = records["1"]
    return got, jobstats_metrics(got.stats, got.gpus)


def test_narrowing_the_summary_to_one_node_changes_its_numbers():
    """The point of the feature: n1 is busy and n2 idle, so the whole-job GPU% of 50
    hides both. Narrowed, each node reports its own."""
    _whole, metrics = _narrowed()
    assert metrics.value("GPU%") == 50          # (90 + 90 + 10 + 10) / 4

    record, metrics = _narrowed(nodename="n1")
    assert metrics.value("GPU%") == 90
    assert metrics.value("CPU%") == 50          # 100 CPU-s / (100s x 2 cores)
    assert (record.nodes, record.gpus) == ("1", 2)

    _record, metrics = _narrowed(nodename="n2")
    assert metrics.value("GPU%") == 10
    assert metrics.value("CPU%") == 0           # n2 burned no CPU at all


def test_gpu_count_is_cards_not_distinct_minors():
    """Both nodes number their cards from 0, so --gpuid 0 keeps two cards, not one.

    Counting distinct minors made #GPU read 1 here and weighted the pooled row's
    GPU-hours by one card instead of two, halving it.
    """
    record, _metrics = _narrowed(gpu_ids=("0",))
    assert record.gpus == 2
    assert record.nodes == "2"                  # both nodes still hold a card


def test_gpuid_leaves_the_cpu_side_alone():
    """--gpuid says nothing about cores, so narrowing cards must not move CPU%."""
    _whole, before = _narrowed()
    _record, after = _narrowed(gpu_ids=("0",))
    assert after.value("CPU%") == before.value("CPU%")
    assert after.value("GPU%") == 50            # (90 + 10) / 2, card 0 of each node


def test_narrowing_names_what_missed():
    from jobscope.select import _narrow_records

    def narrow(nodename=None, gpu_ids=()):
        record = JobRecord(jobid="1", state="COMPLETED", name="j", runtime="x",
                           nodes="2", gpus=4, stats=_TWO_NODE_STATS, start=0, end=100,
                           duration=100, jobid_raw="1", cluster="c", user="alice")
        _narrow_records({"1": record}, ["1"], nodename, gpu_ids)

    with pytest.raises(JobscopeError) as excinfo:
        narrow(nodename="n9")
    assert "'n9'" in str(excinfo.value) and "n1, n2" in str(excinfo.value)

    # Every miss, not just an all-miss: the 9 is what needs saying in "0,9".
    with pytest.raises(JobscopeError) as excinfo:
        narrow(gpu_ids=("0", "9"))
    assert "'9'" in str(excinfo.value) and "0, 1" in str(excinfo.value)


# --- a source that does not cover every card ---------------------------------------

def _gap_gpus():
    """Three jobs, one card each, on two hosts -- two of them on the blind host."""
    from jobscope.running import Gpu
    return {"u1": Gpu("u1", 1, "goodhost", 0, "GPU 0"),
            "u2": Gpu("u2", 2, "blindhost", 0, "GPU 0"),
            "u3": Gpu("u3", 3, "blindhost", 1, "GPU 1")}


def test_cards_the_chosen_source_does_not_cover_are_named(capsys):
    """The report: `-p kempner -a --ts --eff all` counted 3 jobs under --gpu-source nvml
    and 1 under dcgm, because dcgm-exporter was down on one host of the partition. The
    jobs vanish from the series and therefore from --eff's counts, so the difference has
    to be stated -- otherwise the two invocations disagree with nothing to explain it."""
    samples = {"u1": {100: {"duty": 50.0}}}          # u2/u3 came back empty
    select_mod._note_gpu_series_gap(_gap_gpus(), samples, dcgm.catalog().default_specs)
    err = capsys.readouterr().err
    assert "2 of 3 job(s) have no GPU metrics" in err
    assert "blindhost" in err and "goodhost" not in err
    # The actionable half: the cards are discovered through the nvml join either way, so
    # one missing from dcgm is usually present in nvml rather than genuinely idle.
    assert "--gpu-source" in err
    assert "probe" in err


def test_full_coverage_says_nothing(capsys):
    """A note that fires when nothing is wrong is worse than none."""
    samples = {u: {100: {"duty": 1.0}} for u in ("u1", "u2", "u3")}
    select_mod._note_gpu_series_gap(_gap_gpus(), samples, dcgm.catalog().default_specs)
    assert capsys.readouterr().err == ""


def test_the_running_branch_supplies_per_node_figures(monkeypatch):
    """The gap a cluster run found and the tests did not: per_node was computed in
    running.py and never put into the chunk, so --per-node showed dashes for every
    profiling column on a live job while the finished path was fine.

    Asserted on the chunk, which is the contract, rather than on the helper -- the helper
    was already right.
    """
    _patch_running(monkeypatch)
    selected = resolve(Request(mode=RUNNING, user="alice"), _cfg(), None, 1, dcgm.catalog().default_specs)
    (_jobids, _records, dcgm_data), = list(selected.chunks)
    per_node = dcgm_data["7"].per_node
    assert per_node, "the running branch supplied no per-node figures"
    # One node in the fixture, and its pooled figure is the job's since that is the whole
    # set -- the two reductions have to agree when the subset is everything.
    assert list(per_node) == ["node01"]
    assert per_node["node01"]["SM_ACT%"] == dcgm_data["7"].overall["SM_ACT%"]


def test_resolve_reports_whether_the_values_span_whole_runtimes(monkeypatch):
    """The fact both the Sampled line and the summary's weighting are read off.

    Settled here because the renderer cannot ask: it picks the tallies' unit -- counts or
    resource-hours -- before the first chunk arrives, and only here are the records in hand
    *and* the job-count cap applied.
    """
    running = _record("RUNNING", {"total_time": 600, "nodes": {}}, jobid="1")
    running = dataclasses.replace(running, end=None)
    done = _record("COMPLETED", {"total_time": 600, "nodes": {}}, jobid="2")
    assert running.unfinished and not done.unfinished          # the premise

    for records, average, expected in (({"2": done}, False, True),
                                       ({"1": running}, False, False),
                                       ({"1": running}, True, True),
                                       # Mixed: conservative, since a table cannot be half
                                       # weighted by hours.
                                       ({"1": running, "2": done}, False, False)):
        monkeypatch.setattr(select_mod, "select_jobs",
                            lambda *a, **kw: (list(records), "2 job ID(s)"))
        monkeypatch.setattr(select_mod, "fetch", lambda *a, **kw: dict(records))
        monkeypatch.setattr(select_mod, "_enrich", lambda chunks, *a, **kw: iter(chunks))
        request = Request(mode=JOBIDS, jobids=list(records), average=average)
        resolved = resolve(request, _cfg(), None, 1, None)
        assert resolved.folded is expected, (records, average)


def test_a_window_selection_is_folded_by_construction(monkeypatch, stream_window):
    """states_for() returns only finished states and fetch_window filters again on
    JobRecord.unfinished, so every record in a window selection already spans its whole
    runtime."""
    stream_window([(["1"], {})])
    monkeypatch.setattr(select_mod, "_enrich", lambda chunks, *a, **kw: iter(chunks))
    resolved = resolve(Request(mode=FINISHED, user="alice"), _cfg(), None, 1, None)
    assert resolved.folded is True


def _many_jobs(n):
    return {i: {"jobid": str(i), "user": "alice", "node": "node%02d" % i, "name": "train",
                "start_epoch": 1000, "elapsed_seconds": 600}
            for i in range(1, n + 1)}


class _ManyCardClient(FakeRunningClient):
    """One card per job, so collect_averaged actually fans out per (job, metric)."""

    def __init__(self, n, host_series=True):
        super().__init__()
        self.n = n
        self.host_series = host_series

    def query(self, query, at, timeout=None):
        self.queries.append(query)
        if "nvidia_gpu_jobId" in query:
            return [{"metric": {"uuid": "U%d" % i, "host": "node%02d:9445" % i,
                                "minor_number": "0"},
                     "value": [at, "%d" % i]} for i in range(1, self.n + 1)]
        if "cgroup_" in query:
            return super().query(query, at, timeout) if self.host_series else []
        # Only the cards this query actually named, so a batch's fan-out stays its own.
        asked = re.findall(r"U\d+", query)
        return [{"metric": {"uuid": u, "UUID": u}, "value": [at, "90"]} for u in asked]


def _resolve_running_for(monkeypatch, jobs, client, cfg=None, **kw):
    monkeypatch.setattr(select_mod, "fetch_jobs", lambda sel, t: dict(jobs))
    monkeypatch.setattr(select_mod, "client_from_config", lambda c, t: client)
    request = Request(mode=RUNNING, user="alice", **kw)
    return resolve(request, cfg or _cfg(), None, 4, dcgm.catalog().default_specs,
                   host_specs=list((cfg or _cfg()).metrics.host_summary))


def test_the_running_view_yields_a_batch_at_a_time(monkeypatch):
    """The whole point: rows reach the screen while the rest is still being queried.

    SummaryRenderer.add prints and flushes each row, so what turns a wide sweep from a
    blank wait into a table filling in is this generator handing over more than once.
    The contract is slurm.fetch_chunks': ids in display order, records cumulative, and
    concatenating every batch reproduces the selection exactly.
    """
    jobs = _many_jobs(20)
    resolved = _resolve_running_for(monkeypatch, jobs, _ManyCardClient(20), average=True)
    seen, sizes = [], []
    for batch, records, dcgm_data in resolved.chunks:
        assert batch, "an empty batch would make the renderer print nothing"
        seen.extend(batch)
        sizes.append((len(records), len(dcgm_data)))
    assert len(sizes) > 1, "one chunk is the eager behaviour this replaces"
    assert len(seen) == 20 and len(set(seen)) == 20
    assert seen == sorted(seen, key=lambda j: select_mod.job_sort_key({"jobid": j}))
    # Cumulative, never shrinking -- the renderer reads only the ids it is handed, but
    # the dicts are shared across yields.
    assert sizes == sorted(sizes)


def test_instant_yields_one_chunk(monkeypatch):
    """collect_instant is a single grouped query covering every card, so batching it would
    add one query per batch to a path that already returns in about a second."""
    jobs = _many_jobs(20)
    resolved = _resolve_running_for(monkeypatch, jobs, _ManyCardClient(20), average=False)
    assert len(list(resolved.chunks)) == 1


def test_the_running_sampled_line_follows_the_reduction(monkeypatch):
    """Everything squeue returns is still running, so the reduction is a choice. The
    header has to describe the choice that was made -- Used/GPU-hr beside "most recent
    scrape" is a contradiction, not a nuance."""
    for average, folded in ((True, True), (False, False)):
        resolved = _resolve_running_for(monkeypatch, _many_jobs(3), _ManyCardClient(3),
                                        average=average)
        assert resolved.folded is folded
        sampled = dict(resolved.context).get("Sampled", "")
        assert ("most recent scrape" in sampled) is not folded, (average, sampled)


def test_the_safety_net_refuses_a_sweep_past_the_limit(monkeypatch):
    """A backstop against a typo'd selection, not a cost policy -- so it names the flags
    that would narrow it rather than just refusing."""
    cfg = dataclasses.replace(
        _cfg(), defaults=dataclasses.replace(_cfg().defaults, max_running_jobs=5))
    with pytest.raises(JobscopeError) as exc:
        _resolve_running_for(monkeypatch, _many_jobs(6), _ManyCardClient(6), cfg=cfg,
                             average=True)
    message = str(exc.value)
    assert "6 running jobs" in message and "limit 5" in message
    assert "--min-elapsed" in message and "max_running_jobs" in message


def test_the_cost_estimate_prints_once_and_only_when_it_is_worth_saying(monkeypatch,
                                                                       capsys):
    """Pacing a wide sweep is the difference between slow and hung, so say the size up
    front. On a narrow one the fan-out finishes before anyone wonders and the note is
    noise."""
    wide = _resolve_running_for(monkeypatch, _many_jobs(60), _ManyCardClient(60),
                               average=True)
    capsys.readouterr()
    list(wide.chunks)
    err = capsys.readouterr().err
    assert err.count("running jobs -- about") == 1, err
    assert "Ctrl-C" in err and "--min-elapsed" in err

    narrow = _resolve_running_for(monkeypatch, _many_jobs(3), _ManyCardClient(3),
                                  average=True)
    capsys.readouterr()
    list(narrow.chunks)
    assert "about" not in capsys.readouterr().err


def test_the_host_gap_note_prints_once_not_once_per_batch(monkeypatch, capsys):
    """note_missing_host_series counts jobs -- "N running job(s) have no CPU%/MEM%" -- so
    inside the batch loop it would print once per batch and each count would be wrong.
    Hoisted past the final yield, which also puts it between the rows and the summary."""
    jobs = _many_jobs(20)
    resolved = _resolve_running_for(monkeypatch, jobs,
                                    _ManyCardClient(20, host_series=False), average=True)
    capsys.readouterr()
    list(resolved.chunks)
    err = capsys.readouterr().err
    assert err.count("have no CPU%/MEM%") == 1, err
    assert "20 running job(s)" in err, err


def test_batching_does_not_change_the_summary(monkeypatch):
    """Arrival order must not touch arithmetic. The tallies accumulate across add() calls
    and the aggregators run per batch, so this is the property that makes streaming a
    presentation change rather than a numerical one: the same selection rendered in one
    batch and in fourteen must produce the same table and the same pooled figures.
    """
    from jobscope.report import RenderOptions, SummaryRenderer

    def render(chunk_size):
        monkeypatch.setattr(select_mod, "RUNNING_JOBS_PER_CHUNK", chunk_size)
        jobs = _many_jobs(20)
        resolved = _resolve_running_for(monkeypatch, jobs, _ManyCardClient(20),
                                        average=True)
        out = io.StringIO()
        renderer = SummaryRenderer(resolved.context,
                                   RenderOptions(view="all", header=True,
                                                 time_weighted=resolved.folded), out)
        for batch, records, dcgm_data in resolved.chunks:
            renderer.add(build_rows(batch, records, dcgm_data))
        renderer.finish()
        return out.getvalue()

    streamed = render(3)          # seven batches
    at_once = render(999)         # one
    assert streamed == at_once
