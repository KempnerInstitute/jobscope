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


# --- the window, cut into slices --------------------------------------------

def _at(stamp: str) -> str:
    """A TIMESTAMP_FORMAT stamp, so window_slices can read it back."""
    return time.strftime(TIMESTAMP_FORMAT, time.strptime(stamp, TIMESTAMP_FORMAT))


def test_a_one_day_window_is_a_single_slice():
    """The common case must not pay for the machinery: -D 1 is one call, as before."""
    slices = slurm.window_slices(_at("2026-08-10T00:00:00"), _at("2026-08-11T00:00:00"))
    assert slices == [("2026-08-10T00:00:00", "2026-08-11T00:00:00")]


def test_a_week_is_seven_slices_that_meet_without_overlapping():
    slices = slurm.window_slices(_at("2026-08-04T00:00:00"), _at("2026-08-11T00:00:00"))
    assert len(slices) == 7
    # Each slice starts exactly where the last ended: a gap loses jobs, an overlap
    # fetches them twice.
    assert all(a[1] == b[0] for a, b in zip(slices, slices[1:]))
    assert (slices[0][0], slices[-1][1]) == ("2026-08-04T00:00:00", "2026-08-11T00:00:00")


def test_a_ragged_window_keeps_its_ends():
    """The last slice is short rather than the window being rounded out to a day."""
    slices = slurm.window_slices(_at("2026-08-09T06:00:00"), _at("2026-08-11T09:30:00"))
    assert (slices[0][0], slices[-1][1]) == ("2026-08-09T06:00:00", "2026-08-11T09:30:00")
    assert len(slices) == 3


@pytest.mark.parametrize("start,end", [
    ("now-30days", "now"),          # the default lookback, sacct's own relative form
    ("2026-08-10", "2026-08-11"),   # a bare date, which epoch() does not read
])
def test_a_window_that_cannot_be_read_is_left_whole(start, end):
    """Unreadable is not unusable: sacct still understands these, so pass them through
    as one slice rather than refusing or guessing at a cut."""
    assert slurm.window_slices(start, end) == [(start, end)]


def test_an_inverted_or_empty_window_is_one_slice():
    same = _at("2026-08-10T00:00:00")
    assert slurm.window_slices(same, same) == [(same, same)]
    assert len(slurm.window_slices(_at("2026-08-11T00:00:00"), same)) == 1


# --- streaming a window, one sacct call per slice ----------------------------

def _window_stub(monkeypatch, by_slice, calls=None):
    """Answer each window fetch from ``by_slice``, keyed by the slice's -S value."""
    summary = make_jobstats(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        assert cmd[:2] == ["sacct", "-X"], cmd
        start = cmd[cmd.index("-S") + 1]
        if calls is not None:
            calls.append(start)
        ids = by_slice.get(start)
        if ids is None:
            return None                     # a slice that failed
        return "".join(_record_line(j, summary) for j in ids)

    monkeypatch.setattr(slurm, "run_capture", fake)


def _week(**by_day):
    """``{slice start: ids}`` for a week beginning 2026-08-04, keyed d0..d6."""
    return {"2026-08-0%dT00:00:00" % (4 + int(k[1:])): v for k, v in by_day.items()}


def _selection():
    return Selection(user="alice", starttime=_at("2026-08-04T00:00:00"),
                     endtime=_at("2026-08-11T00:00:00"))


def test_fetch_window_yields_one_slice_at_a_time(monkeypatch):
    calls = []
    _window_stub(monkeypatch, _week(d0=["100"], d1=["101"], d2=["102"]), calls)
    chunks = list(slurm.fetch_window(_selection(), None))
    assert [ids for ids, _r in chunks] == [["100"], ["101"], ["102"]]
    assert len(calls) == 7          # every slice is asked, even the empty ones


def test_one_slice_is_still_streamed_in_chunks(monkeypatch):
    """How much is *fetched* and how much is *yielded* are different questions.

    One sacct call per slice is right for the server, but run_capture buffers that call
    whole -- so yielding a slice in one piece made a default -D 1 a single chunk of nine
    thousand jobs, and every metric query for all of them ran before the first row could
    be drawn. This is the regression guard for that: a slice is handed out in chunks.
    """
    monkeypatch.setattr(slurm, "STREAM_CHUNK", 3)
    ids = ["10%02d" % i for i in range(7)]
    _window_stub(monkeypatch, _week(d0=ids))
    chunks = list(slurm.fetch_window(_selection(), None))
    assert [len(batch) for batch, _r in chunks] == [3, 3, 1]
    # Order is preserved across the chunk boundaries, and nothing is dropped or repeated.
    assert [j for batch, _r in chunks for j in batch] == ids
    # Each chunk carries only its own records.
    assert all(set(recs) == set(batch) for batch, recs in chunks)


def test_fetch_window_records_are_per_slice_not_cumulative(monkeypatch):
    """The other half of what made a wide selection expensive: fetch_chunks kept every
    record to the end of the run, and nothing downstream ever needed it to."""
    _window_stub(monkeypatch, _week(d0=["100"], d1=["101"]))
    chunks = list(slurm.fetch_window(_selection(), None))
    assert list(chunks[0][1]) == ["100"]
    assert list(chunks[1][1]) == ["101"]


def test_fetch_window_dedups_a_job_that_spans_a_boundary(monkeypatch):
    """sacct returns such a job from both slices; the first to reach it owns it, or the
    report would show the same job twice."""
    _window_stub(monkeypatch, _week(d0=["100", "200"], d1=["200", "201"]))
    chunks = list(slurm.fetch_window(_selection(), None))
    assert [ids for ids, _r in chunks] == [["100", "200"], ["201"]]


def test_fetch_window_skips_a_job_that_has_not_finished(monkeypatch):
    """The same belt-and-braces the id listing used to apply behind the -s filter."""
    summary = make_jobstats(GPU_STATS)

    def fake(cmd, timeout, what, soft=False):
        if cmd[cmd.index("-S") + 1] != "2026-08-04T00:00:00":
            return ""
        return (_record_line("100", summary)
                + _record_line("101", summary).replace("|COMPLETED|", "|RUNNING|"))

    monkeypatch.setattr(slurm, "run_capture", fake)
    chunks = list(slurm.fetch_window(_selection(), None))
    assert [ids for ids, _r in chunks] == [["100"]]


def test_a_failed_slice_is_named_and_the_rest_still_arrive(monkeypatch, capsys):
    """A hole the reader is told about beats both a silent hole and no report at all."""
    _window_stub(monkeypatch, _week(d0=["100"], d2=["102"]))   # d1 missing -> None
    chunks = list(slurm.fetch_window(_selection(), None))
    assert [ids for ids, _r in chunks] == [["100"], ["102"]]
    err = capsys.readouterr().err
    assert "that span is missing from this report" in err
    assert "2026-08-05" in err          # names which span, not just that one failed


def test_a_single_slice_window_says_nothing(monkeypatch, capsys):
    """The slice note is for a window worth explaining, not for every -D 1."""
    _window_stub(monkeypatch, {"2026-08-10T00:00:00": ["100"]})
    selection = Selection(user="alice", starttime=_at("2026-08-10T00:00:00"),
                          endtime=_at("2026-08-11T00:00:00"))
    assert [ids for ids, _r in slurm.fetch_window(selection, None)] == [["100"]]
    assert capsys.readouterr().err == ""


def test_the_window_fetch_and_the_id_listing_select_the_same_jobs(monkeypatch):
    """They must agree on *which* jobs, or a fetch returns rows the listing never
    offered. One filter builder feeds both; this pins that it stays that way."""
    selection = Selection(user="alice", partition="kempner", account="lab",
                          state="failed", starttime="2026-08-04T00:00:00",
                          endtime="2026-08-05T00:00:00")
    listing = slurm._select_cmd(selection, *selection.window())
    fetching = slurm._window_fetch_cmd(selection, *selection.window())
    # Everything up to the output format is identical.
    assert listing[:listing.index("--noheader")] == fetching[:fetching.index("--noheader")]
    assert "-r" in fetching and "kempner" in fetching
    assert "-X" in fetching          # allocations only; steps would multiply the rows


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


# --- hostlist expansion -------------------------------------------------------
#
# squeue's %N is compressed and every comparison downstream is against one label value,
# so an unexpanded list matches nothing. Parsed in-process rather than forked out to
# `scontrol show hostnames`: that fork costs ~36ms, and a partition selection holds
# enough distinct lists to make it the most expensive thing in the report -- 177 real
# lists on this cluster expand in 2.5ms total, and were checked against scontrol with
# zero mismatches.

@pytest.mark.parametrize("text,expected", [
    ("n1", ("n1",)),
    ("n1,n2", ("n1", "n2")),
    ("holygpu8a[05304,06504]", ("holygpu8a05304", "holygpu8a06504")),
    ("node[1-3]", ("node1", "node2", "node3")),
    # Zero-padding follows the low end, which is how Slurm writes it.
    ("node[08-11]", ("node08", "node09", "node10", "node11")),
    ("n[1-2,5]", ("n1", "n2", "n5")),
    ("a[1-2]x", ("a1x", "a2x")),
    # A top-level comma separates hosts; one inside brackets is part of the range.
    ("n1,m[2-3],p9", ("n1", "m2", "m3", "p9")),
])
def test_a_nodelist_expands_without_a_subprocess(text, expected, monkeypatch):
    monkeypatch.setattr(slurm, "run_capture",
                        lambda *a, **k: pytest.fail("should not have forked scontrol"))
    slurm._HOSTLIST.clear()
    assert slurm.expand_nodelist(text) == expected


@pytest.mark.parametrize("text", [
    "node[a-c]",            # non-numeric range
    "node[3-1]",            # descending
    "a[1-2]b[3-4]",         # two groups
    "node[[1-2]]",          # nested
])
def test_a_form_the_parser_cannot_be_sure_of_falls_back(text, monkeypatch):
    """None rather than a guess: these names decide whether a card belongs to a job, so a
    wrong expansion silently drops or steals one. Declining costs a fork."""
    assert slurm._expand_ranges(text) is None
    called = []
    monkeypatch.setattr(slurm, "run_capture",
                        lambda *a, **k: called.append(1) or "fallback1 fallback2")
    slurm._HOSTLIST.clear()
    assert slurm.expand_nodelist(text) == ("fallback1", "fallback2")
    assert called, "the declined form has to reach scontrol"


def test_an_expanded_nodelist_is_cached(monkeypatch):
    slurm._HOSTLIST.clear()
    assert slurm.expand_nodelist("n[1-2]") == ("n1", "n2")
    monkeypatch.setattr(slurm, "run_capture",
                        lambda *a, **k: pytest.fail("cached, so nothing to re-read"))
    assert slurm.expand_nodelist("n[1-2]") == ("n1", "n2")
