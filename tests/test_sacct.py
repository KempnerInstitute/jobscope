"""Tests for job selection, the bulk fetch, and the subprocess helper."""

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
