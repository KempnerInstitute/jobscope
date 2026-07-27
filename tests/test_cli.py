"""Tests for CLI argument handling, validation, and dispatch."""

import argparse
import dataclasses

import pytest

from jobscope import cli
from jobscope.cli import _inject_default_subcommand, _prepare_selection, build_parser, main
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
    base = dict(user="alice", jobids=[], jobids_opt=None, account=None, partition=None,
                state="all", lastn=None, days=None, starttime=None, endtime=None)
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


def test_starttime_alone_selects_single_day():
    sel = _prepare_selection(_sel_args(starttime="2026-07-15"))
    assert sel.starttime == "2026-07-15"
    assert sel.endtime == "2026-07-16T00:00:00"


def test_starttime_with_endtime_left_alone():
    sel = _prepare_selection(_sel_args(starttime="2026-07-15", endtime="2026-07-20"))
    assert sel.endtime == "2026-07-20"


def test_starttime_relative_leaves_window_open():
    sel = _prepare_selection(_sel_args(starttime="now-2days"))
    assert sel.endtime is None


def test_args_order_independent():
    _, subparsers = build_parser()
    summary = subparsers.choices["summary"]
    a1 = summary.parse_intermixed_args(["-D", "3", "111", "222"])
    a2 = summary.parse_intermixed_args(["111", "-D", "3", "222"])
    a3 = summary.parse_intermixed_args(["111", "222", "-D", "3"])
    assert a1.jobids == a2.jobids == a3.jobids == ["111", "222"]
    assert a1.days == a2.days == a3.days == 3


def test_jobid_flag_merges_with_positional():
    _, subparsers = build_parser()
    summary = subparsers.choices["summary"]
    for argv in (["111", "222"], ["-j", "111", "-j", "222"]):
        args = summary.parse_intermixed_args(argv)
        args.user = "alice"
        assert _prepare_selection(args).jobids == ["111", "222"]
    # Mixing the positional and -j still collects both IDs.
    mixed = summary.parse_intermixed_args(["-j", "111", "222"])
    mixed.user = "alice"
    assert set(_prepare_selection(mixed).jobids) == {"111", "222"}


def test_jobids_warn_on_ignored_selectors(capsys):
    sel = _prepare_selection(_sel_args(jobids=["1"], days=3))
    err = capsys.readouterr().err
    assert "ignoring time selectors" in err and "-D/--days" in err
    assert sel.jobids == ["1"] and sel.days is None


def test_main_jobid_intermixed_with_flags(monkeypatch, cpu_record):
    seen = {}

    def fake_select(selection, timeout):
        seen["jobids"] = list(selection.jobids)
        return (list(selection.jobids), "1 job ID(s)")

    monkeypatch.setattr(cli, "select_jobs", fake_select)
    monkeypatch.setattr(cli, "fetch", lambda ids, timeout: {i: cpu_record for i in ids})
    # JOBID before a flag, and no subcommand -> exercises injection + intermixed parse.
    main(["--cpu", "111", "-u", "bob"])
    assert seen["jobids"] == ["111"]


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
    monkeypatch.setattr(cli, "fetch_chunks",
                        lambda ids, timeout: iter([(list(ids), {"200": cpu_record})]))
    main(["summary", "--cpu", "-D", "1", "-u", "bob"])
    out = capsys.readouterr().out
    assert "200" in out and "CPU%" in out


def test_summary_streams_chunks_per_batch(monkeypatch, capsys, gpu_record):
    rec2 = dataclasses.replace(gpu_record, jobid="101", name="eval")
    records = {"100": gpu_record, "101": rec2}
    monkeypatch.setattr(cli, "select_jobs",
                        lambda selection, timeout: (["100", "101"], "last 1 day"))
    monkeypatch.setattr(cli, "fetch_chunks",
                        lambda ids, timeout: iter([(["100"], records), (["101"], records)]))
    clients = []
    monkeypatch.setattr(cli, "client_from_config",
                        lambda cfg, timeout: clients.append(1) or object())
    overall = {"SM_ACT%": 60.0, "OCC%": 20.0, "TENSOR%": 5.0, "DRAM%": 10.0, "POWER_W": 400.0}
    seen_chunks = []

    def fake_compute(records_arg, chunk_ids, *a, **k):
        seen_chunks.append(list(chunk_ids))
        return {j: (overall, {}) for j in chunk_ids}

    monkeypatch.setattr(cli, "compute_dcgm", fake_compute)
    main(["summary", "--gpu", "-D", "1", "-u", "alice"])
    out = capsys.readouterr().out
    assert seen_chunks == [["100"], ["101"]]  # DCGM computed per chunk
    assert len(clients) == 1                  # Prometheus client created once
    assert "100" in out and "101" in out
    assert out.count("Mean:") == 1
    assert out.index("Mean:") > out.index("101")


def test_summary_gpu_chunk_without_gpu_defers_client(monkeypatch, capsys, gpu_record, cpu_record):
    monkeypatch.setattr(cli, "select_jobs",
                        lambda selection, timeout: (["200", "100"], "last 1 day"))
    monkeypatch.setattr(cli, "fetch_chunks",
                        lambda ids, timeout: iter([
                            (["200"], {"200": cpu_record}),
                            (["100"], {"200": cpu_record, "100": gpu_record})]))
    clients = []
    monkeypatch.setattr(cli, "client_from_config",
                        lambda cfg, timeout: clients.append(1) or object())
    monkeypatch.setattr(cli, "compute_dcgm", lambda *a, **k: {"100": ({}, {})})
    main(["summary", "--gpu", "-D", "1", "-u", "alice"])
    out = capsys.readouterr().out
    assert len(clients) == 1  # not created for the cpu-only chunk, once for the gpu one
    assert "100" in out


def test_explicit_jobids_do_not_stream(monkeypatch, capsys, cpu_record):
    called = {}

    def fake_fetch(ids, timeout):
        called["ids"] = list(ids)
        return {"111": cpu_record}

    def no_stream(*a, **k):
        raise AssertionError("fetch_chunks must not be used for explicit job IDs")

    monkeypatch.setattr(cli, "fetch", fake_fetch)
    monkeypatch.setattr(cli, "fetch_chunks", no_stream)
    main(["summary", "--cpu", "111", "-u", "bob"])
    assert called["ids"] == ["111"]
    assert "111" in capsys.readouterr().out


def test_detail_streams_chunks(monkeypatch, capsys, cpu_record):
    rec2 = dataclasses.replace(cpu_record, jobid="201")
    monkeypatch.setattr(cli, "select_jobs",
                        lambda selection, timeout: (["200", "201"], "last 1 day"))
    monkeypatch.setattr(cli, "fetch_chunks",
                        lambda ids, timeout: iter([
                            (["200"], {"200": cpu_record}),
                            (["201"], {"200": cpu_record, "201": rec2})]))
    main(["detail", "--cpu", "-D", "1", "-u", "bob"])
    out = capsys.readouterr().out
    assert out.index("Job 200") < out.index("Job 201")


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
