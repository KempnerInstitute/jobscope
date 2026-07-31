"""Tests for the selection layer that hides sacct and squeue behind one interface."""

import dataclasses
import time

from jobscope import select as select_mod
from jobscope.dcgm import DEFAULT_SPECS
from jobscope.sacct import TIMESTAMP_FORMAT
from jobscope.select import (
    BLOB_SPECS,
    FINISHED,
    JOBIDS,
    RUNNING,
    Request,
    resolve,
    sacct_selection,
)

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


def test_historical_yields_one_chunk_per_batch(monkeypatch, gpu_record):
    rec2 = dataclasses.replace(gpu_record, jobid="101")
    records = {"100": gpu_record, "101": rec2}
    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: (["100", "101"], "last 1 day"))
    monkeypatch.setattr(select_mod, "fetch_chunks",
                        lambda ids, t: iter([(["100"], records), (["101"], records)]))
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, t: object())
    monkeypatch.setattr(select_mod, "compute_dcgm",
                        lambda r, ids, *a, **k: {j: ({"SM_ACT%": 1.0}, {}) for j in ids})

    selected = resolve(Request(mode=FINISHED, user="alice"), _cfg(), None, 1, DEFAULT_SPECS)
    chunks = list(selected.chunks)
    assert [ids for ids, _r, _d in chunks] == [["100"], ["101"]]
    # Each chunk arrives with its own metrics attached, so rendering can stream.
    assert all(dcgm for _i, _r, dcgm in chunks)


def test_historical_skips_prometheus_without_specs(monkeypatch, gpu_record):
    """`--cpu` over finished jobs must stay offline."""
    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: (["100"], "x"))
    monkeypatch.setattr(select_mod, "fetch", lambda i, t: {"100": gpu_record})
    monkeypatch.setattr(select_mod, "fetch_chunks",
                        lambda i, t: iter([(["100"], {"100": gpu_record})]))

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


def test_no_matching_jobs_returns_none(monkeypatch, capsys):
    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: ([], "last 1 day"))
    assert resolve(Request(mode=FINISHED, user="nobody"), _cfg(), None, 1, None) is None
    assert "No matching jobs" in capsys.readouterr().err


def test_no_matching_jobs_names_all_users(monkeypatch, capsys):
    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: ([], "last 1 day"))
    resolve(Request(mode=FINISHED, all_users=True, user=None), _cfg(), None, 1, None)
    assert "all users" in capsys.readouterr().err


# --- the squeue branch ------------------------------------------------------

class FakeLiveClient:
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


def _patch_live(monkeypatch, client=None):
    client = client or FakeLiveClient()
    monkeypatch.setattr(select_mod, "fetch_jobs", lambda sel, t: dict(JOBS))
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, t: client)
    return client


def test_running_yields_records_that_look_finished(monkeypatch):
    """The squeue branch must be indistinguishable from sacct downstream."""
    _patch_live(monkeypatch)
    selected = resolve(Request(mode=RUNNING, user="alice"), _cfg(), None, 1, DEFAULT_SPECS)
    (jobids, records, dcgm_data), = list(selected.chunks)
    assert jobids == ["7"]
    record = records["7"]
    assert record.state == "RUNNING" and record.jobid_raw == "7"
    # A reconstructed blob, so blob_metrics works exactly as for a stored one:
    # cpu = 100*150/(100*2) = 75, mem = 100*8/16 = 50, gpu = 90, gmem = 40/80 = 50.
    from jobscope.blob import blob_metrics
    assert blob_metrics(record.stats) == (75, 50, 90, 50)
    assert dcgm_data["7"][0]["SM_ACT%"] == 80.0


def test_running_provides_the_per_gpu_metrics(monkeypatch):
    _patch_live(monkeypatch)
    selected = resolve(Request(mode=RUNNING, user="alice"), _cfg(), None, 1, DEFAULT_SPECS)
    (_ids, _records, dcgm_data), = list(selected.chunks)
    per_gpu = dcgm_data["7"][1]
    # Keyed (node, minor) as the detail renderer and the blob both expect.
    assert per_gpu[("node01", "2")]["SM_ACT%"] == 80.0


def test_running_without_specs_still_builds_the_blob(monkeypatch):
    """--cpu needs CPU%, which comes from the reconstructed blob."""
    client = _patch_live(monkeypatch)
    selected = resolve(Request(mode=RUNNING, user="alice"), _cfg(), None, 1, None)
    (_ids, records, _d), = list(selected.chunks)
    from jobscope.blob import blob_metrics
    assert blob_metrics(records["7"].stats)[:2] == (75, 50)
    # ... but it does not pay for the DCGM profiling queries it will not print.
    assert not any("DCGM_FI" in q for q in client.queries)
    assert BLOB_SPECS and all(s.key in ("duty", "mem", "memtot") for s in BLOB_SPECS)


def test_running_context_names_the_owner_for_explicit_ids(monkeypatch):
    _patch_live(monkeypatch)
    selected = resolve(Request(mode=RUNNING, jobids=["7"], user=None),
                       _cfg(), None, 1, DEFAULT_SPECS)
    assert ("User", "alice") in selected.context


def test_running_context_says_all_users(monkeypatch):
    _patch_live(monkeypatch)
    selected = resolve(Request(mode=RUNNING, all_users=True, user=None),
                       _cfg(), None, 1, DEFAULT_SPECS)
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
    assert "add -a to include every user" in err


def test_both_branches_yield_the_same_chunk_shape(monkeypatch, gpu_record):
    """The contract that lets one renderer serve both sources."""
    _patch_live(monkeypatch)
    live = list(resolve(Request(mode=RUNNING, user="alice"),
                        _cfg(), None, 1, DEFAULT_SPECS).chunks)

    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: (["100"], "x"))
    monkeypatch.setattr(select_mod, "fetch", lambda i, t: {"100": gpu_record})
    monkeypatch.setattr(select_mod, "compute_dcgm", lambda r, ids, *a, **k: {"100": ({}, {})})
    past = list(resolve(Request(mode=JOBIDS, jobids=["100"], user="alice"),
                        _cfg(), None, 1, DEFAULT_SPECS).chunks)

    for chunks in (live, past):
        (jobids, records, dcgm_data), = chunks
        assert isinstance(jobids, list) and isinstance(records, dict)
        assert isinstance(dcgm_data, dict)
        jid = jobids[0]
        assert records[jid].jobid_raw          # both carry the raw ID for the joins
        overall, per_gpu = dcgm_data[jid]
        assert isinstance(overall, dict) and isinstance(per_gpu, dict)
