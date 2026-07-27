"""Tests for job selection, the bulk fetch, and the subprocess helper."""

import errno

import pytest

from jobscope import sacct
from jobscope.errors import JobscopeError
from jobscope.sacct import (
    Selection,
    days_to_window,
    default_user,
    end_of_day,
    epoch,
    fetch,
    query_jobid,
    run_capture,
    select_jobs,
)

from .conftest import GPU_STATS, make_blob


def test_default_user_from_env(monkeypatch):
    monkeypatch.setenv("USER", "carol")
    assert default_user() == "carol"


def test_days_to_window_orders():
    start, end = days_to_window(3)
    assert start < end


def test_end_of_day_from_date():
    assert end_of_day("2026-07-15") == "2026-07-16T00:00:00"


def test_end_of_day_from_datetime_closes_that_calendar_day():
    assert end_of_day("2026-07-15T14:30:00") == "2026-07-16T00:00:00"


def test_end_of_day_unparseable_returns_none():
    assert end_of_day("now-2days") is None


def test_epoch():
    assert epoch("Unknown") is None
    assert epoch("") is None
    assert epoch("garbage") is None
    assert isinstance(epoch("2020-01-01T00:00:00"), int)


def test_query_jobid_debrackets_arrays():
    assert query_jobid("18114115_[0-719%64]") == "18114115"
    assert query_jobid("18114115_3") == "18114115_3"
    assert query_jobid("12345") == "12345"


def test_run_capture_success():
    assert run_capture(["echo", "hi"], None, "echo") == "hi\n"


def test_run_capture_nonzero_raises():
    with pytest.raises(JobscopeError):
        run_capture(["false"], None, "false")


def test_run_capture_soft_returns_none():
    assert run_capture(["false"], None, "false", soft=True) is None


def test_run_capture_missing_command():
    with pytest.raises(JobscopeError):
        run_capture(["jobscope_no_such_command_xyz"], None, "missing")


def test_run_capture_timeout():
    with pytest.raises(JobscopeError) as exc:
        run_capture(["sleep", "3"], 0.2, "sleep")
    assert "timed out" in str(exc.value)


def test_select_jobs_explicit_ids_bypass():
    ids, desc = select_jobs(Selection(user="alice", jobids=["5", "6"]), None)
    assert ids == ["5", "6"]
    assert desc == "2 job ID(s)"


def test_select_jobs_parses_and_skips_pending(monkeypatch):
    monkeypatch.setattr(sacct, "run_capture",
                        lambda *a, **k: "100|COMPLETED\n101|RUNNING\n102|PENDING\n")
    ids, desc = select_jobs(Selection(user="alice"), None)
    assert ids == ["100", "101"]
    assert desc == "now-30days .. now"


def test_select_jobs_lastn(monkeypatch):
    monkeypatch.setattr(sacct, "run_capture", lambda *a, **k: "100|COMPLETED\n101|RUNNING\n")
    ids, desc = select_jobs(Selection(user="alice", lastn=1), None)
    assert ids == ["101"]
    assert desc == "last 1 jobs"


def test_select_jobs_days_desc(monkeypatch):
    monkeypatch.setattr(sacct, "run_capture", lambda *a, **k: "100|COMPLETED\n")
    ids, desc = select_jobs(
        Selection(user="alice", days=3, starttime="2026-01-01T00:00:00", endtime="now"), None)
    assert desc == "last 3 days"


def test_fetch_parses_record(monkeypatch):
    blob = make_blob(GPU_STATS)
    line = "|".join(["100", "COMPLETED", "train", "01:00:00", "1",
                     "billing=2,cpu=2,gres/gpu=2,mem=16G",
                     "2020-01-01T00:00:00", "2020-01-01T01:00:00", "100", "odyssey",
                     "alice", blob])
    monkeypatch.setattr(sacct, "run_capture", lambda *a, **k: line + "\n")
    records = fetch(["100"], None)
    record = records["100"]
    assert record.state == "COMPLETED"
    assert record.name == "train"
    assert record.gpus == 2
    assert record.duration == 3600
    assert record.user == "alice"
    assert record.stats == GPU_STATS


def test_fetch_empty():
    assert fetch([], None) == {}


def test_fetch_aliases_array_base_id(monkeypatch):
    # sacct returns the record keyed under the base id; fetch must alias it back
    # to the bracketed id the caller asked for.
    line = "|".join(["18114115", "COMPLETED", "arr", "00:10:00", "1", "gres/gpu=1",
                     "2020-01-01T00:00:00", "2020-01-01T00:10:00", "18114115",
                     "odyssey", "alice", make_blob(GPU_STATS)])
    monkeypatch.setattr(sacct, "run_capture", lambda *a, **k: line + "\n")
    records = fetch(["18114115_[0-719%64]"], None)
    assert "18114115_[0-719%64]" in records
    assert records["18114115_[0-719%64]"].state == "COMPLETED"


def _record_line(jobid: str, blob: str, name: str = "train") -> str:
    return "|".join([jobid, "COMPLETED", name, "01:00:00", "1", "gres/gpu=1",
                     "2020-01-01T00:00:00", "2020-01-01T01:00:00", jobid,
                     "odyssey", "alice", blob]) + "\n"


def _fake_fetch_run_capture(calls, blob):
    """A run_capture stub that answers any chunked -j query and records it."""
    def fake(cmd, timeout, what, soft=False):
        assert cmd[:2] == ["sacct", "-j"]
        assert len(cmd[2]) <= sacct.JOBID_ARG_LIMIT
        ids = cmd[2].split(",")
        calls.append(ids)
        return "".join(_record_line(jid, blob) for jid in ids)
    return fake


def test_chunk_jobids_empty():
    assert sacct.chunk_jobids([], 64) == []


def test_chunk_jobids_single_oversized_id():
    # An id longer than the limit cannot be split; it gets its own chunk.
    assert sacct.chunk_jobids(["a" * 100], 8) == [["a" * 100]]


def test_chunk_jobids_boundary():
    assert sacct.chunk_jobids(["aaa", "bbb"], 7) == [["aaa", "bbb"]]
    assert sacct.chunk_jobids(["aaa", "bbb", "c"], 7) == [["aaa", "bbb"], ["c"]]


def test_chunk_jobids_preserves_order_and_membership():
    ids = [str(1000 + i) for i in range(100)]
    chunks = sacct.chunk_jobids(ids, 16)
    assert [jid for chunk in chunks for jid in chunk] == ids
    assert all(len(",".join(chunk)) <= 16 for chunk in chunks)


def test_chunk_jobids_max_count():
    ids = ["1", "2", "3", "4", "5"]
    assert sacct.chunk_jobids(ids, 1024, max_count=2) == [["1", "2"], ["3", "4"], ["5"]]
    # the byte limit still wins when it is the tighter bound
    assert sacct.chunk_jobids(["aaa", "bbb", "ccc"], 7, max_count=10) == [["aaa", "bbb"], ["ccc"]]


def test_fetch_chunks_large_id_list(monkeypatch):
    # 4,000 8-digit ids join to ~36 KB -- over the 16 KB argv cap, so fetch
    # must split the -j query instead of building one oversized argv token.
    ids = [str(10_000_000 + i) for i in range(4_000)]
    blob = make_blob(GPU_STATS)
    calls = []
    monkeypatch.setattr(sacct, "run_capture", _fake_fetch_run_capture(calls, blob))
    records = fetch(ids, None)
    assert len(calls) >= 2
    assert [jid for chunk in calls for jid in chunk] == ids
    assert len(records) == len(ids)
    assert records["10000000"].state == "COMPLETED"
    assert records["10003999"].state == "COMPLETED"


def test_fetch_merges_records_across_chunks(monkeypatch):
    # limit 8 forces the two 8-byte ids into separate chunks; both parses must
    # merge into one dict rather than the second overwriting the first.
    monkeypatch.setattr(sacct, "JOBID_ARG_LIMIT", 8)
    blob = make_blob(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        jid = cmd[2]
        assert "," not in jid
        return _record_line(jid, blob, name="job-%s" % jid)

    monkeypatch.setattr(sacct, "run_capture", fake)
    records = fetch(["10000001", "10000002"], None)
    assert records["10000001"].name == "job-10000001"
    assert records["10000002"].name == "job-10000002"


def test_fetch_array_alias_survives_chunking(monkeypatch):
    # The bracketed id and the other id land in different chunks; the alias
    # back to the requested bracketed id must still resolve afterward.
    monkeypatch.setattr(sacct, "JOBID_ARG_LIMIT", 8)
    blob = make_blob(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], blob)

    monkeypatch.setattr(sacct, "run_capture", fake)
    records = fetch(["18114115_[0-719%64]", "99999999"], None)
    assert "18114115_[0-719%64]" in records
    assert records["99999999"].state == "COMPLETED"


def test_fetch_small_list_single_call(monkeypatch, capsys):
    # The common case: everything fits in one chunk -- exactly one sacct call
    # and no progress chatter on stderr.
    blob = make_blob(GPU_STATS)
    calls = []
    monkeypatch.setattr(sacct, "run_capture", _fake_fetch_run_capture(calls, blob))
    records = fetch(["100", "101"], None)
    assert len(calls) == 1
    assert len(records) == 2
    assert capsys.readouterr().err == ""


def test_fetch_progress_notes(monkeypatch, capsys):
    monkeypatch.setattr(sacct, "JOBID_ARG_LIMIT", 8)
    blob = make_blob(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], blob)

    monkeypatch.setattr(sacct, "run_capture", fake)
    fetch(["10000001", "10000002"], None)
    err = capsys.readouterr().err
    assert "2 jobs selected -- fetching sacct data in 2 batches" in err
    assert "batch 1/2" not in err  # throttled: below the NOTE_EVERY cadence
    assert "batch 2/2 done (2/2 jobs)" in err  # the final batch always notes


def test_fetch_chunks_empty():
    assert list(sacct.fetch_chunks([], None)) == []


def test_fetch_chunks_single_chunk_no_notes(monkeypatch, capsys):
    blob = make_blob(GPU_STATS)
    calls = []
    monkeypatch.setattr(sacct, "run_capture", _fake_fetch_run_capture(calls, blob))
    chunks = list(sacct.fetch_chunks(["100", "101"], None))
    assert len(chunks) == 1
    ready, records = chunks[0]
    assert ready == ["100", "101"]
    assert "100" in records and "101" in records
    assert len(calls) == 1
    assert capsys.readouterr().err == ""


def test_fetch_chunks_yields_per_batch_in_order(monkeypatch):
    monkeypatch.setattr(sacct, "JOBID_ARG_LIMIT", 8)
    blob = make_blob(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        assert "," not in cmd[2]
        return _record_line(cmd[2], blob)

    monkeypatch.setattr(sacct, "run_capture", fake)
    chunks = list(sacct.fetch_chunks(["10000001", "10000002"], None))
    assert [ready for ready, _ in chunks] == [["10000001"], ["10000002"]]
    # the records dict is cumulative: the last yield sees every job so far
    assert "10000001" in chunks[1][1] and "10000002" in chunks[1][1]


def test_fetch_chunks_alias_applied_at_yield(monkeypatch):
    monkeypatch.setattr(sacct, "JOBID_ARG_LIMIT", 8)
    blob = make_blob(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], blob)

    monkeypatch.setattr(sacct, "run_capture", fake)
    chunks = list(sacct.fetch_chunks(["18114115_[0-719%64]", "99999999"], None))
    ready, records = chunks[0]
    assert ready == ["18114115_[0-719%64]"]
    assert "18114115_[0-719%64]" in records


def test_fetch_chunks_prefix_preserves_global_order(monkeypatch):
    # The two bracketed ids share a base queried in chunk 1, but the middle
    # id's base is only queried in chunk 2 -- the third id must wait so that
    # the concatenated yields equal the input order.
    monkeypatch.setattr(sacct, "JOBID_ARG_LIMIT", 8)
    blob = make_blob(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], blob)

    monkeypatch.setattr(sacct, "run_capture", fake)
    requested = ["18114115_[0-9]", "99999999", "18114115_[10-19]"]
    chunks = list(sacct.fetch_chunks(requested, None))
    assert [ready for ready, _ in chunks] == [
        ["18114115_[0-9]"], ["99999999", "18114115_[10-19]"]]


def test_fetch_chunks_progress_notes(monkeypatch, capsys):
    monkeypatch.setattr(sacct, "JOBID_ARG_LIMIT", 8)
    blob = make_blob(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], blob)

    monkeypatch.setattr(sacct, "run_capture", fake)
    list(sacct.fetch_chunks(["10000001", "10000002"], None))
    err = capsys.readouterr().err
    assert "2 jobs selected -- fetching sacct data in 2 batches" in err
    assert "batch 1/2" not in err  # throttled: below the NOTE_EVERY cadence
    assert "batch 2/2 done (2/2 jobs)" in err  # the final batch always notes


def test_fetch_chunks_uses_jobs_per_chunk(monkeypatch):
    # The count bound splits even when the byte limit is nowhere near binding.
    monkeypatch.setattr(sacct, "JOBS_PER_CHUNK", 2)
    blob = make_blob(GPU_STATS)
    calls = []
    monkeypatch.setattr(sacct, "run_capture", _fake_fetch_run_capture(calls, blob))
    list(sacct.fetch_chunks(["1", "2", "3", "4", "5"], None))
    assert [len(c) for c in calls] == [2, 2, 1]


def test_fetch_chunks_notes_throttled(monkeypatch, capsys):
    monkeypatch.setattr(sacct, "JOBS_PER_CHUNK", 1)
    monkeypatch.setattr(sacct, "NOTE_EVERY", 2)
    blob = make_blob(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], blob)

    monkeypatch.setattr(sacct, "run_capture", fake)
    list(sacct.fetch_chunks(["1", "2", "3", "4"], None))
    err = capsys.readouterr().err
    assert "4 jobs selected -- fetching sacct data in 4 batches" in err
    assert "batch 2/4 done (2/4 jobs)" in err   # crossed the 2-job cadence
    assert "batch 4/4 done (4/4 jobs)" in err   # crossing + final
    assert "batch 1/4" not in err and "batch 3/4" not in err


def test_fetch_chunks_final_note_always_prints(monkeypatch, capsys):
    monkeypatch.setattr(sacct, "JOBS_PER_CHUNK", 1)
    blob = make_blob(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], blob)

    monkeypatch.setattr(sacct, "run_capture", fake)
    list(sacct.fetch_chunks(["1", "2", "3"], None))
    err = capsys.readouterr().err
    assert "batch 3/3 done (3/3 jobs)" in err  # completion note despite no cadence crossing
    assert "batch 1/3" not in err and "batch 2/3" not in err


def test_run_capture_oserror_raises_jobscope_error(monkeypatch):
    def boom(*a, **k):
        raise OSError(errno.E2BIG, "Argument list too long")

    monkeypatch.setattr(sacct.subprocess, "Popen", boom)
    with pytest.raises(JobscopeError) as exc:
        run_capture(["sacct", "-j", "1"], None, "sacct query")
    assert "narrow" in str(exc.value).lower()


def test_run_capture_oserror_soft_returns_none(monkeypatch):
    def boom(*a, **k):
        raise OSError(errno.E2BIG, "Argument list too long")

    monkeypatch.setattr(sacct.subprocess, "Popen", boom)
    assert run_capture(["sacct", "-j", "1"], None, "sacct query", soft=True) is None
