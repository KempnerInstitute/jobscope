"""Tests for job selection, the bulk fetch, and the subprocess helper."""

import errno
import time

import pytest

from jobscope import slurm
from jobscope.errors import JobscopeError
from jobscope.slurm import (
    TIMESTAMP_FORMAT,
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

from .conftest import GPU_STATS, make_jobstats


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


def test_select_jobs_skips_jobs_that_have_not_finished(monkeypatch):
    """`finished` means finished: a running job has no final numbers to report.

    The -s filter should keep these out at the sacct end; the client-side skip is
    the belt-and-braces behind it, and this pins it.
    """
    monkeypatch.setattr(slurm, "run_capture",
                        lambda *a, **k: "100|COMPLETED\n101|RUNNING\n"
                                        "102|PENDING\n103|SUSPENDED\n")
    ids, desc = select_jobs(Selection(user="alice"), None)
    assert ids == ["100"]
    assert desc == "now-30days .. now, completed"


def _rows(n, first=100):
    """``n`` COMPLETED selection rows, the shape _query_ids parses."""
    return "".join("%d|COMPLETED\n" % (first + i) for i in range(n))


def test_an_ordinary_selection_draws_no_breadth_note(monkeypatch, capsys):
    """The note is worth nothing if it fires on normal use. A busy GPU partition here
    runs ~100 jobs a day, so the common case must stay silent."""
    monkeypatch.setattr(slurm, "run_capture", lambda *a, **k: _rows(200))
    ids, _ = select_jobs(Selection(user="alice", partition="kempner", days=1), None)
    assert len(ids) == 200
    assert capsys.readouterr().err == ""


def test_a_broad_selection_is_named_before_the_batches_start(monkeypatch, capsys):
    """Said at selection time, not fetch time: the count cannot be known any earlier,
    and saying it early is what leaves room to abort before the batches run."""
    monkeypatch.setattr(slurm, "run_capture",
                        lambda *a, **k: _rows(slurm.LARGE_SELECTION))
    # -a unsets user; the CLI rejects the two together, so do not model both.
    ids, _ = select_jobs(Selection(user=None, all_users=True), None)
    err = capsys.readouterr().err
    assert len(ids) == slurm.LARGE_SELECTION
    assert "%d jobs is a broad selection" % slurm.LARGE_SELECTION in err
    assert "stay in memory" in err
    # On the default 30-day window, so the flag that shortens it is worth naming.
    assert "-D 1 for a single day" in err
    # It advises, it does not cap: the ids all come back.
    assert ids[0] == "100"


def test_the_breadth_note_offers_only_narrowings_not_already_in_use(monkeypatch, capsys):
    """Repeating a flag back to someone who just typed it reads as though it did not
    take effect, so -p is not offered to a selection that already named one."""
    monkeypatch.setattr(slurm, "run_capture",
                        lambda *a, **k: _rows(slurm.LARGE_SELECTION))
    select_jobs(Selection(user="alice", partition="kempner", days=1, lastn=None), None)
    err = capsys.readouterr().err
    assert "-p PARTITION" not in err        # already scoped to one
    assert "-D 1" not in err                # already a one-day window
    assert "-N to take only the newest N" in err   # the one thing left to suggest


def test_the_breadth_note_offers_only_a_smaller_n_under_lastn(monkeypatch, capsys):
    """Two traps here, both found by running it.

    _query_lastn writes the span it walked back onto `selection`, so asking about the
    window *after* the query offered a shorter one the user never chose -- the offers
    are therefore taken before the queries run.

    And under -N the count is pinned at exactly lastn, so a partition or a shorter
    window would change which jobs come back, not how many. Only a smaller -N helps.
    """
    monkeypatch.setattr(slurm, "run_capture",
                        lambda *a, **k: _rows(slurm.LARGE_SELECTION))
    select_jobs(Selection(user="alice", lastn=slurm.LARGE_SELECTION), None)
    err = capsys.readouterr().err
    assert "a smaller -N" in err
    assert "window" not in err and "-p PARTITION" not in err


def test_the_breadth_note_does_not_respell_an_explicit_window(monkeypatch, capsys):
    """Caught by running it: -S/-E was offered back to a selection that had just used
    -S/-E. A window the user chose can still be too wide, so the advice stays -- but it
    asks for a shorter one instead of naming the flags they typed."""
    monkeypatch.setattr(slurm, "run_capture",
                        lambda *a, **k: _rows(slurm.LARGE_SELECTION))
    select_jobs(Selection(user=None, all_users=True, starttime="now-1hours",
                          endtime="now"), None)
    err = capsys.readouterr().err
    assert "-S/-E" not in err and "-D 1" not in err
    assert "a shorter window" in err


def test_select_jobs_lastn(monkeypatch):
    monkeypatch.setattr(slurm, "run_capture",
                        lambda *a, **k: "100|COMPLETED\n101|COMPLETED\n")
    ids, desc = select_jobs(Selection(user="alice", lastn=1), None)
    assert ids == ["101"]                       # ascending, so the tail is newest
    assert desc == "last 1 job, completed"      # singular


def test_lastn_stops_after_one_day_when_that_is_enough(monkeypatch):
    """Scanning the month to find one job is the slowest way to answer the cheapest
    question: measured, one day of a busy partition took 1.2s where thirty timed out."""
    windows = []

    def fake(cmd, *a, **k):
        windows.append((cmd[cmd.index("-S") + 1], cmd[cmd.index("-E") + 1]))
        return "100|COMPLETED\n101|COMPLETED\n"

    monkeypatch.setattr(slurm, "run_capture", fake)
    selection = Selection(user="alice", lastn=1)
    ids, _desc = select_jobs(selection, None)
    assert ids == ["101"]
    assert len(windows) == 1                    # one day sufficed
    # Each query covers a single day, and the reported window is that day.
    start, end = windows[0]
    span = time.mktime(time.strptime(end, TIMESTAMP_FORMAT)) - \
        time.mktime(time.strptime(start, TIMESTAMP_FORMAT))
    assert abs(span - 86400) < 5
    assert (selection.starttime, selection.endtime) == (start, end)


def test_each_step_queries_exactly_one_day_further_back(monkeypatch):
    """Day by day, and each query covers only the new day.

    Re-scanning from now every step would re-list the same jobs over and over, which
    is what makes the naive widening expensive: the cost grows with the span.
    """
    spans, starts = [], []

    def fake(cmd, *a, **k):
        start, end = cmd[cmd.index("-S") + 1], cmd[cmd.index("-E") + 1]
        starts.append(start)
        spans.append(time.mktime(time.strptime(end, TIMESTAMP_FORMAT)) -
                     time.mktime(time.strptime(start, TIMESTAMP_FORMAT)))
        return ""                               # never enough, so it keeps walking

    monkeypatch.setattr(slurm, "run_capture", fake)
    selection = Selection(user="alice", lastn=1)
    select_jobs(selection, None)
    assert len(spans) == slurm.DEFAULT_LOOKBACK_DAYS
    assert all(abs(span - 86400) < 5 for span in spans)       # one day each, not growing
    # And walking backwards: every step starts earlier than the one before.
    assert starts == sorted(starts, reverse=True)


def test_lastn_walks_back_until_it_has_enough(monkeypatch):
    calls = []

    def fake(cmd, *a, **k):
        calls.append(cmd[cmd.index("-S") + 1])
        # Nothing until the third day back, then two jobs.
        return "100|COMPLETED\n101|COMPLETED\n" if len(calls) >= 3 else ""

    monkeypatch.setattr(slurm, "run_capture", fake)
    selection = Selection(user="alice", lastn=2)
    ids, _desc = select_jobs(selection, None)
    assert ids == ["100", "101"]
    assert len(calls) == 3
    # Three days covered, reported as one window ending now.
    span = time.mktime(time.strptime(selection.endtime, TIMESTAMP_FORMAT)) - \
        time.mktime(time.strptime(selection.starttime, TIMESTAMP_FORMAT))
    assert abs(span - 3 * 86400) < 5


def test_older_days_are_prepended_so_the_newest_are_kept(monkeypatch):
    """The trim takes the tail, so the accumulated list has to stay ascending.

    Walking backwards yields older jobs later, so appending would make the trim keep
    the oldest jobs while claiming to show the newest.
    """
    days = []

    def fake(cmd, *a, **k):
        days.append(1)
        return {1: "300|COMPLETED\n", 2: "200|COMPLETED\n"}.get(len(days), "100|COMPLETED\n")

    monkeypatch.setattr(slurm, "run_capture", fake)
    ids, _desc = select_jobs(Selection(user="alice", lastn=2), None)
    assert ids == ["200", "300"]                # the two newest, oldest-first


def test_a_job_spanning_two_days_is_not_counted_twice(monkeypatch):
    """sacct matches a job in every window it ran through, so slices overlap."""
    def fake(cmd, *a, **k):
        return "100|COMPLETED\n"               # the same job every day

    monkeypatch.setattr(slurm, "run_capture", fake)
    ids, _desc = select_jobs(Selection(user="alice", lastn=3), None)
    assert ids == ["100"]


def test_lastn_keeps_a_narrow_answer_when_a_wider_query_times_out(monkeypatch):
    """Some jobs beat an error, and the header says how far back it managed to look."""
    calls = []

    def fake(cmd, *a, **k):
        calls.append(1)
        if len(calls) == 1:
            return "100|COMPLETED\n"
        raise JobscopeError("sacct query timed out after 60s")

    monkeypatch.setattr(slurm, "run_capture", fake)
    ids, _desc = select_jobs(Selection(user="alice", lastn=5), None)
    assert ids == ["100"]                       # not an exception
    assert len(calls) == 2


def test_lastn_still_raises_when_even_the_narrowest_window_fails(monkeypatch):
    def boom(*a, **k):
        raise JobscopeError("sacct query timed out after 60s")

    monkeypatch.setattr(slurm, "run_capture", boom)
    with pytest.raises(JobscopeError, match="timed out"):
        select_jobs(Selection(user="alice", lastn=5), None)


def test_select_jobs_days_desc(monkeypatch):
    monkeypatch.setattr(slurm, "run_capture", lambda *a, **k: "100|COMPLETED\n")
    ids, desc = select_jobs(
        Selection(user="alice", days=3, starttime="2026-01-01T00:00:00", endtime="now"), None)
    assert desc == "last 3 days, completed"


def test_fetch_parses_record(monkeypatch):
    summary = make_jobstats(GPU_STATS)
    line = "|".join(["100", "COMPLETED", "train", "01:00:00", "1",
                     "billing=2,cpu=2,gres/gpu=2,mem=16G",
                     "2020-01-01T00:00:00", "2020-01-01T01:00:00", "100", "odyssey",
                     "alice", summary])
    monkeypatch.setattr(slurm, "run_capture", lambda *a, **k: line + "\n")
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
                     "odyssey", "alice", make_jobstats(GPU_STATS)])
    monkeypatch.setattr(slurm, "run_capture", lambda *a, **k: line + "\n")
    records = fetch(["18114115_[0-719%64]"], None)
    assert "18114115_[0-719%64]" in records
    assert records["18114115_[0-719%64]"].state == "COMPLETED"


def _record_line(jobid: str, summary: str, name: str = "train") -> str:
    return "|".join([jobid, "COMPLETED", name, "01:00:00", "1", "gres/gpu=1",
                     "2020-01-01T00:00:00", "2020-01-01T01:00:00", jobid,
                     "odyssey", "alice", summary]) + "\n"


def _fake_fetch_run_capture(calls, summary):
    """A run_capture stub that answers any chunked -j query and records it."""
    def fake(cmd, timeout, what, soft=False):
        assert cmd[:2] == ["sacct", "-j"]
        assert len(cmd[2]) <= slurm.JOBID_ARG_LIMIT
        ids = cmd[2].split(",")
        calls.append(ids)
        return "".join(_record_line(jid, summary) for jid in ids)
    return fake


def test_chunk_jobids_empty():
    assert slurm.chunk_jobids([], 64) == []


def test_chunk_jobids_single_oversized_id():
    # An id longer than the limit cannot be split; it gets its own chunk.
    assert slurm.chunk_jobids(["a" * 100], 8) == [["a" * 100]]


def test_chunk_jobids_boundary():
    assert slurm.chunk_jobids(["aaa", "bbb"], 7) == [["aaa", "bbb"]]
    assert slurm.chunk_jobids(["aaa", "bbb", "c"], 7) == [["aaa", "bbb"], ["c"]]


def test_chunk_jobids_preserves_order_and_membership():
    ids = [str(1000 + i) for i in range(100)]
    chunks = slurm.chunk_jobids(ids, 16)
    assert [jid for chunk in chunks for jid in chunk] == ids
    assert all(len(",".join(chunk)) <= 16 for chunk in chunks)


def test_chunk_jobids_max_count():
    ids = ["1", "2", "3", "4", "5"]
    assert slurm.chunk_jobids(ids, 1024, max_count=2) == [["1", "2"], ["3", "4"], ["5"]]
    # the byte limit still wins when it is the tighter bound
    assert slurm.chunk_jobids(["aaa", "bbb", "ccc"], 7, max_count=10) == [["aaa", "bbb"], ["ccc"]]


def test_fetch_chunks_large_id_list(monkeypatch):
    # 4,000 8-digit ids join to ~36 KB -- over the 16 KB argv cap, so fetch
    # must split the -j query instead of building one oversized argv token.
    ids = [str(10_000_000 + i) for i in range(4_000)]
    summary = make_jobstats(GPU_STATS)
    calls = []
    monkeypatch.setattr(slurm, "run_capture", _fake_fetch_run_capture(calls, summary))
    records = fetch(ids, None)
    assert len(calls) >= 2
    assert [jid for chunk in calls for jid in chunk] == ids
    assert len(records) == len(ids)
    assert records["10000000"].state == "COMPLETED"
    assert records["10003999"].state == "COMPLETED"


def test_fetch_merges_records_across_chunks(monkeypatch):
    # limit 8 forces the two 8-byte ids into separate chunks; both parses must
    # merge into one dict rather than the second overwriting the first.
    monkeypatch.setattr(slurm, "JOBID_ARG_LIMIT", 8)
    summary = make_jobstats(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        jid = cmd[2]
        assert "," not in jid
        return _record_line(jid, summary, name="job-%s" % jid)

    monkeypatch.setattr(slurm, "run_capture", fake)
    records = fetch(["10000001", "10000002"], None)
    assert records["10000001"].name == "job-10000001"
    assert records["10000002"].name == "job-10000002"


def test_fetch_array_alias_survives_chunking(monkeypatch):
    # The bracketed id and the other id land in different chunks; the alias
    # back to the requested bracketed id must still resolve afterward.
    monkeypatch.setattr(slurm, "JOBID_ARG_LIMIT", 8)
    summary = make_jobstats(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], summary)

    monkeypatch.setattr(slurm, "run_capture", fake)
    records = fetch(["18114115_[0-719%64]", "99999999"], None)
    assert "18114115_[0-719%64]" in records
    assert records["99999999"].state == "COMPLETED"


def test_fetch_small_list_single_call(monkeypatch, capsys):
    # The common case: everything fits in one chunk -- exactly one sacct call
    # and no progress chatter on stderr.
    summary = make_jobstats(GPU_STATS)
    calls = []
    monkeypatch.setattr(slurm, "run_capture", _fake_fetch_run_capture(calls, summary))
    records = fetch(["100", "101"], None)
    assert len(calls) == 1
    assert len(records) == 2
    assert capsys.readouterr().err == ""


def test_fetch_progress_notes(monkeypatch, capsys):
    monkeypatch.setattr(slurm, "JOBID_ARG_LIMIT", 8)
    summary = make_jobstats(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], summary)

    monkeypatch.setattr(slurm, "run_capture", fake)
    fetch(["10000001", "10000002"], None)
    err = capsys.readouterr().err
    assert "2 jobs selected -- fetching sacct data in 2 batches" in err
    assert "batch 1/2" not in err  # throttled: below the NOTE_EVERY cadence
    assert "batch 2/2 done (2/2 jobs)" in err  # the final batch always notes


def test_fetch_chunks_empty():
    assert list(slurm.fetch_chunks([], None)) == []


def test_fetch_chunks_single_chunk_no_notes(monkeypatch, capsys):
    summary = make_jobstats(GPU_STATS)
    calls = []
    monkeypatch.setattr(slurm, "run_capture", _fake_fetch_run_capture(calls, summary))
    chunks = list(slurm.fetch_chunks(["100", "101"], None))
    assert len(chunks) == 1
    ready, records = chunks[0]
    assert ready == ["100", "101"]
    assert "100" in records and "101" in records
    assert len(calls) == 1
    assert capsys.readouterr().err == ""


def test_fetch_chunks_yields_per_batch_in_order(monkeypatch):
    monkeypatch.setattr(slurm, "JOBID_ARG_LIMIT", 8)
    summary = make_jobstats(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        assert "," not in cmd[2]
        return _record_line(cmd[2], summary)

    monkeypatch.setattr(slurm, "run_capture", fake)
    chunks = list(slurm.fetch_chunks(["10000001", "10000002"], None))
    assert [ready for ready, _ in chunks] == [["10000001"], ["10000002"]]
    # the records dict is cumulative: the last yield sees every job so far
    assert "10000001" in chunks[1][1] and "10000002" in chunks[1][1]


def test_fetch_chunks_alias_applied_at_yield(monkeypatch):
    monkeypatch.setattr(slurm, "JOBID_ARG_LIMIT", 8)
    summary = make_jobstats(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], summary)

    monkeypatch.setattr(slurm, "run_capture", fake)
    chunks = list(slurm.fetch_chunks(["18114115_[0-719%64]", "99999999"], None))
    ready, records = chunks[0]
    assert ready == ["18114115_[0-719%64]"]
    assert "18114115_[0-719%64]" in records


def test_fetch_chunks_prefix_preserves_global_order(monkeypatch):
    # The two bracketed ids share a base queried in chunk 1, but the middle
    # id's base is only queried in chunk 2 -- the third id must wait so that
    # the concatenated yields equal the input order.
    monkeypatch.setattr(slurm, "JOBID_ARG_LIMIT", 8)
    summary = make_jobstats(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], summary)

    monkeypatch.setattr(slurm, "run_capture", fake)
    requested = ["18114115_[0-9]", "99999999", "18114115_[10-19]"]
    chunks = list(slurm.fetch_chunks(requested, None))
    assert [ready for ready, _ in chunks] == [
        ["18114115_[0-9]"], ["99999999", "18114115_[10-19]"]]


def test_fetch_chunks_progress_notes(monkeypatch, capsys):
    monkeypatch.setattr(slurm, "JOBID_ARG_LIMIT", 8)
    summary = make_jobstats(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], summary)

    monkeypatch.setattr(slurm, "run_capture", fake)
    list(slurm.fetch_chunks(["10000001", "10000002"], None))
    err = capsys.readouterr().err
    assert "2 jobs selected -- fetching sacct data in 2 batches" in err
    assert "batch 1/2" not in err  # throttled: below the NOTE_EVERY cadence
    assert "batch 2/2 done (2/2 jobs)" in err  # the final batch always notes


def test_fetch_chunks_uses_jobs_per_chunk(monkeypatch):
    # The count bound splits even when the byte limit is nowhere near binding.
    monkeypatch.setattr(slurm, "JOBS_PER_CHUNK", 2)
    summary = make_jobstats(GPU_STATS)
    calls = []
    monkeypatch.setattr(slurm, "run_capture", _fake_fetch_run_capture(calls, summary))
    list(slurm.fetch_chunks(["1", "2", "3", "4", "5"], None))
    assert [len(c) for c in calls] == [2, 2, 1]


def test_fetch_chunks_notes_throttled(monkeypatch, capsys):
    monkeypatch.setattr(slurm, "JOBS_PER_CHUNK", 1)
    monkeypatch.setattr(slurm, "NOTE_EVERY", 2)
    summary = make_jobstats(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], summary)

    monkeypatch.setattr(slurm, "run_capture", fake)
    list(slurm.fetch_chunks(["1", "2", "3", "4"], None))
    err = capsys.readouterr().err
    assert "4 jobs selected -- fetching sacct data in 4 batches" in err
    assert "batch 2/4 done (2/4 jobs)" in err   # crossed the 2-job cadence
    assert "batch 4/4 done (4/4 jobs)" in err   # crossing + final
    assert "batch 1/4" not in err and "batch 3/4" not in err


def test_fetch_chunks_final_note_always_prints(monkeypatch, capsys):
    monkeypatch.setattr(slurm, "JOBS_PER_CHUNK", 1)
    summary = make_jobstats(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        return _record_line(cmd[2], summary)

    monkeypatch.setattr(slurm, "run_capture", fake)
    list(slurm.fetch_chunks(["1", "2", "3"], None))
    err = capsys.readouterr().err
    assert "batch 3/3 done (3/3 jobs)" in err  # completion note despite no cadence crossing
    assert "batch 1/3" not in err and "batch 2/3" not in err


def test_run_capture_oserror_raises_jobscope_error(monkeypatch):
    def boom(*a, **k):
        raise OSError(errno.E2BIG, "Argument list too long")

    monkeypatch.setattr(slurm.subprocess, "Popen", boom)
    with pytest.raises(JobscopeError) as exc:
        run_capture(["sacct", "-j", "1"], None, "sacct query")
    assert "narrow" in str(exc.value).lower()


def test_run_capture_oserror_soft_returns_none(monkeypatch):
    def boom(*a, **k):
        raise OSError(errno.E2BIG, "Argument list too long")

    monkeypatch.setattr(slurm.subprocess, "Popen", boom)
    assert run_capture(["sacct", "-j", "1"], None, "sacct query", soft=True) is None


def test_the_state_filter_is_always_passed_to_sacct(monkeypatch):
    """Without -s, sacct returns running jobs, which is what made `finished` wrong."""
    seen = {}

    def fake(cmd, *a, **k):
        seen["cmd"] = cmd
        return "100|COMPLETED\n"

    monkeypatch.setattr(slurm, "run_capture", fake)
    select_jobs(Selection(user="alice"), None)
    assert "-s" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("-s") + 1] == "COMPLETED"
    select_jobs(Selection(user="alice", state="failed,timeout"), None)
    passed = seen["cmd"][seen["cmd"].index("-s") + 1].split(",")
    assert "TIMEOUT" in passed and "FAILED" in passed and "COMPLETED" not in passed


def test_state_groups_are_separable_and_composable():
    from jobscope.slurm import states_for
    assert states_for("completed") == ("COMPLETED",)
    assert "TIMEOUT" not in states_for("failed")      # no longer an umbrella
    assert set(states_for("failed,timeout")) == set(states_for("failed")) | set(
        states_for("timeout"))
    assert set(states_for("all")) > set(states_for("failed,timeout"))
    # Nothing live is selectable, however it is spelled.
    for live in ("running", "RUNNING", "pending"):
        with pytest.raises(JobscopeError, match="jobscope running"):
            states_for(live)
    with pytest.raises(JobscopeError, match="unknown"):
        states_for("nonsense")
