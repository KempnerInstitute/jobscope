"""Tests for CLI argument handling, validation, and dispatch."""

import argparse

import pytest

from jobscope import cli
from jobscope.cli import _inject_default_subcommand, _prepare_selection, main
from jobscope.errors import JobscopeError


def test_inject_default_subcommand():
    assert _inject_default_subcommand([]) == ["summary"]
    assert _inject_default_subcommand(["-D", "3"]) == ["summary", "-D", "3"]
    assert _inject_default_subcommand(["12345"]) == ["summary", "12345"]
    assert _inject_default_subcommand(["summary", "-D", "3"]) == ["summary", "-D", "3"]
    assert _inject_default_subcommand(["plot"]) == ["plot"]
    assert _inject_default_subcommand(["--version"]) == ["--version"]
    assert _inject_default_subcommand(["-h"]) == ["-h"]


def _sel_args(**kw):
    base = dict(user="alice", jobids=[], account=None, partition=None, state="all",
                lastn=None, days=None, starttime=None, endtime=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_default_scope_is_one_day():
    assert _prepare_selection(_sel_args()).days == 1


def test_jobids_skip_default_scope():
    assert _prepare_selection(_sel_args(jobids=["1"])).days is None


def test_days_must_be_positive():
    with pytest.raises(JobscopeError):
        _prepare_selection(_sel_args(days=0))


def test_days_conflicts_with_lastn():
    with pytest.raises(JobscopeError):
        _prepare_selection(_sel_args(days=3, lastn=5))


def test_days_conflicts_with_window():
    with pytest.raises(JobscopeError):
        _prepare_selection(_sel_args(days=3, starttime="2026-01-01T00:00:00"))


def test_lastn_must_be_positive():
    with pytest.raises(JobscopeError):
        _prepare_selection(_sel_args(lastn=-1))


def test_describe_command(capsys):
    main(["describe"])
    assert "CPU%" in capsys.readouterr().out


def test_describe_dcgm_ext(capsys):
    main(["describe", "--dcgm", "--ext"])
    out = capsys.readouterr().out
    assert "DCGM GPU metrics" in out
    assert "28 metrics" in out


def test_config_example(capsys):
    main(["config", "--example"])
    assert "[prometheus]" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "jobscope" in capsys.readouterr().out


def test_summary_cpu_offline(monkeypatch, capsys, cpu_record):
    monkeypatch.setattr(cli, "select_jobs", lambda selection, timeout: (["200"], "last 1 day"))
    monkeypatch.setattr(cli, "fetch", lambda ids, timeout: {"200": cpu_record})
    main(["summary", "--cpu", "-D", "1", "-u", "bob"])
    out = capsys.readouterr().out
    assert "200" in out and "CPU%" in out


def test_summary_gpu_wires_dcgm(monkeypatch, capsys, gpu_record):
    monkeypatch.setattr(cli, "select_jobs", lambda selection, timeout: (["100"], "x"))
    monkeypatch.setattr(cli, "fetch", lambda ids, timeout: {"100": gpu_record})
    monkeypatch.setattr(cli, "client_from_config", lambda cfg, timeout: object())
    overall = {"SM_ACT%": 60.0, "OCC%": 20.0, "TENSOR%": 5.0, "DRAM%": 10.0, "POWER_W": 400.0}
    monkeypatch.setattr(cli, "compute_dcgm", lambda *a, **k: {"100": (overall, {})})
    main(["summary", "--gpu", "-u", "alice", "100"])
    out = capsys.readouterr().out
    assert "SM_ACT%" in out and "60.0" in out


def test_dcgm_ts_requires_one_jobid(capsys):
    with pytest.raises(SystemExit):
        main(["dcgm", "--ts", "1", "2"])
    assert "one job at a time" in capsys.readouterr().err


def test_no_matching_jobs(monkeypatch, capsys):
    monkeypatch.setattr(cli, "select_jobs", lambda selection, timeout: ([], "last 1 day"))
    main(["summary", "--cpu", "-u", "nobody"])
    assert "No matching jobs" in capsys.readouterr().err


def test_dcgm_table(monkeypatch, capsys, gpu_record):
    monkeypatch.setattr(cli, "select_jobs", lambda selection, timeout: (["100"], "x"))
    monkeypatch.setattr(cli, "fetch", lambda ids, timeout: {"100": gpu_record})
    monkeypatch.setattr(cli, "client_from_config", lambda cfg, timeout: object())
    per_gpu = {("node01", "0"): {"SM_ACT%": 80.0}}
    monkeypatch.setattr(cli, "compute_dcgm", lambda *a, **k: {"100": ({}, per_gpu)})
    main(["dcgm", "-u", "alice", "100"])
    out = capsys.readouterr().out
    assert "Job 100" in out and "SM_ACT%" in out


def test_dcgm_timeseries(monkeypatch, capsys, gpu_record):
    monkeypatch.setattr(cli, "select_jobs", lambda selection, timeout: (["100"], "x"))
    monkeypatch.setattr(cli, "fetch", lambda ids, timeout: {"100": gpu_record})

    class Client:
        sampling_period = 60

        def query(self, query, at, timeout=None):
            return [{"metric": {"uuid": "U0", "host": "node01:9400", "minor_number": "0"}}]

        def query_range(self, query, start, end, step, timeout=None):
            return ([{"metric": {"UUID": "U0"}, "values": [[1000, "0.8"]]}]
                    if "DCGM_FI_PROF_SM_ACTIVE" in query else [])

    monkeypatch.setattr(cli, "client_from_config", lambda cfg, timeout: Client())
    main(["dcgm", "--ts", "-u", "alice", "100"])
    out = capsys.readouterr().out
    assert "EPOCH" in out and "80.0" in out


def test_config_path(capsys):
    main(["config", "--path"])
    assert "config.toml" in capsys.readouterr().out


def test_config_summary(capsys):
    main(["config"])
    assert "config path" in capsys.readouterr().out
