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
    # The count is derived from the catalog, so it cannot drift out of date.
    from jobscope.dcgm import ALL_SPECS
    assert "%d metrics" % len(ALL_SPECS) in out


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


# --- the live subcommand ----------------------------------------------------

def _live_args(**kw):
    base = dict(jobids=[], jobids_opt=None, partition=None, user=None, all_users=False,
                min_elapsed="1h")
    base.update(kw)
    return argparse.Namespace(**base)


def test_live_is_a_known_subcommand():
    assert "live" in cli.SUBCOMMANDS
    assert _inject_default_subcommand(["live", "-a"]) == ["live", "-a"]
    _, subparsers = build_parser()
    assert "live" in subparsers.choices


def test_live_defaults_to_the_current_user():
    assert cli._live_selection(_live_args(user="alice")).user == "alice"


def test_live_all_users_clears_the_user_filter():
    assert cli._live_selection(_live_args(all_users=True)).user is None


def test_live_all_users_conflicts_with_user():
    with pytest.raises(JobscopeError):
        cli._live_selection(_live_args(all_users=True, user="alice"))


def test_live_rejects_a_bad_runtime_floor_even_with_no_jobs():
    # Validated up front, so a typo is never silently ignored.
    with pytest.raises(JobscopeError):
        cli._live_selection(_live_args(user="alice", min_elapsed="1 hour"))


def test_live_jobids_bypass_the_filters(capsys):
    selection = cli._live_selection(_live_args(jobids=["100_6"], partition="kempner"))
    err = capsys.readouterr().err
    assert "ignoring the -p/-u/-a filters" in err
    assert selection.jobids == ["100_6"] and selection.user is None


def test_live_jobid_flag_merges_with_positional():
    _, subparsers = build_parser()
    args = subparsers.choices["live"].parse_intermixed_args(["-j", "111", "222"])
    args.user = "alice"
    assert set(cli._live_selection(args).jobids) == {"111", "222"}


def test_live_args_order_independent():
    _, subparsers = build_parser()
    live = subparsers.choices["live"]
    a1 = live.parse_intermixed_args(["-p", "kempner", "111"])
    a2 = live.parse_intermixed_args(["111", "-p", "kempner"])
    assert a1.jobids == a2.jobids == ["111"] and a1.partition == a2.partition == "kempner"


def test_live_describe_needs_no_cluster_access(capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("--describe must not query squeue or Prometheus")

    monkeypatch.setattr(cli, "fetch_jobs", boom)
    monkeypatch.setattr(cli, "client_from_config", boom)
    main(["live", "--describe"])
    out = capsys.readouterr().out
    assert "jobscope live columns" in out and "GPU%" in out


def test_live_describe_reflects_the_column_selection(capsys):
    main(["live", "--all", "--describe"])
    out = capsys.readouterr().out
    assert "SM_ACT%" in out and "MEM%" in out
    # --all pulls in the extended catalog...
    assert "NVLINK_MBs" in out
    # ...but never the delta-reduced counter, which a snapshot cannot express.
    assert "ENERGY_kWh" not in out


class _LiveClient:
    """Prometheus stand-in for the live path: one GPU on one job."""

    sampling_period = 60

    def query(self, query, at, timeout=None):
        if "nvidia_gpu_jobId" in query:
            return [{"metric": {"uuid": "U0", "host": "node01:9445", "minor_number": "3"},
                     "value": [at, "4.2e+07"]}]
        if "DCGM_FI_PROF_SM_ACTIVE" in query:
            return [{"metric": {"UUID": "U0"}, "value": [at, "0.776"]}]
        return []

    def query_range(self, query, start, end, step, timeout=None):
        return ([{"metric": {"UUID": "U0"}, "values": [[1000, "0.8"]]}]
                if "DCGM_FI_PROF_SM_ACTIVE" in query else [])


_LIVE_JOB = {42000000: {"jobid": "100_6", "user": "alice", "node": "node01", "name": "train",
                        "start_epoch": 1000, "elapsed_seconds": 3600}}


def test_live_table_renders(monkeypatch, capsys):
    monkeypatch.setattr(cli, "fetch_jobs", lambda selection, timeout: dict(_LIVE_JOB))
    monkeypatch.setattr(cli, "client_from_config", lambda cfg, timeout: _LiveClient())
    main(["live", "-j", "100_6"])
    out = capsys.readouterr().out
    assert "100_6" in out and "alice" in out
    assert "GPU 3" in out and "77.6" in out


def test_live_reports_the_jobs_owner_not_the_filter(monkeypatch, capsys):
    # With explicit JOBIDs the -u filter is bypassed, so naming a user would lie.
    monkeypatch.setattr(cli, "fetch_jobs", lambda selection, timeout: dict(_LIVE_JOB))
    monkeypatch.setattr(cli, "client_from_config", lambda cfg, timeout: _LiveClient())
    main(["live", "-j", "100_6"])
    assert "alice" in capsys.readouterr().out


def test_live_timeseries_ignores_avg_with_a_note(monkeypatch, capsys):
    monkeypatch.setattr(cli, "fetch_jobs", lambda selection, timeout: dict(_LIVE_JOB))
    monkeypatch.setattr(cli, "client_from_config", lambda cfg, timeout: _LiveClient())
    main(["live", "-j", "100_6", "--ts", "--avg"])
    captured = capsys.readouterr()
    assert "--avg ignored with --ts" in captured.err
    assert captured.out.startswith("JOBID,EPOCH,TIME,NODE,GPU,")


def test_live_no_matching_jobs_is_not_an_error(monkeypatch, capsys):
    monkeypatch.setattr(cli, "fetch_jobs", lambda selection, timeout: {})
    main(["live", "-a"])
    assert "No running jobs match" in capsys.readouterr().err


# --- reconstructing the blob for running jobs -------------------------------

def _running_record(gpu_record):
    return dataclasses.replace(gpu_record, jobid="300", state="RUNNING", stats={})


def test_running_job_blob_is_reconstructed(monkeypatch, capsys, gpu_record):
    """A running job has no stored blob, so GPU%/GMEM% would otherwise be '-'."""
    running = _running_record(gpu_record)
    monkeypatch.setattr(cli, "select_jobs", lambda selection, timeout: (["300"], "last 1 day"))
    monkeypatch.setattr(cli, "fetch_chunks",
                        lambda ids, timeout: iter([(["300"], {"300": running})]))
    monkeypatch.setattr(cli, "client_from_config", lambda cfg, timeout: object())
    monkeypatch.setattr(cli, "compute_dcgm", lambda *a, **k: {"300": ({}, {})})

    def fake_fill(records, ids, client, timeout=None, workers=1):
        records["300"].stats = gpu_record.stats
        return 1

    monkeypatch.setattr(cli, "fill_running", fake_fill)
    main(["summary", "--gpu", "-D", "1", "-u", "alice"])
    out = capsys.readouterr().out
    # The blob fixture yields gpu=70, gmem=50 (see tests/conftest.py).
    assert "70" in out and "50" in out


def test_offline_view_warns_when_no_endpoint_can_fill(monkeypatch, capsys, gpu_record):
    running = _running_record(gpu_record)
    monkeypatch.setattr(cli, "select_jobs", lambda selection, timeout: (["300"], "last 1 day"))
    monkeypatch.setattr(cli, "fetch_chunks",
                        lambda ids, timeout: iter([(["300"], {"300": running})]))

    def no_endpoint(cfg, timeout):
        raise JobscopeError("no Prometheus endpoint configured")

    monkeypatch.setattr(cli, "client_from_config", no_endpoint)
    main(["summary", "--cpu", "-D", "1", "-u", "alice"])
    err = capsys.readouterr().err
    assert "no Prometheus" in err and "blank" in err


def test_finished_job_blob_is_never_recomputed(monkeypatch, capsys, gpu_record):
    """A completed job must report what Slurm stored, not a fresh query."""
    monkeypatch.setattr(cli, "select_jobs", lambda selection, timeout: (["100"], "last 1 day"))
    monkeypatch.setattr(cli, "fetch_chunks",
                        lambda ids, timeout: iter([(["100"], {"100": gpu_record})]))
    monkeypatch.setattr(cli, "client_from_config", lambda cfg, timeout: object())
    monkeypatch.setattr(cli, "compute_dcgm", lambda *a, **k: {"100": ({}, {})})

    def boom(*a, **k):
        raise AssertionError("a stored blob must not be refetched")

    monkeypatch.setattr(cli, "fill_running", boom)
    main(["summary", "--gpu", "-D", "1", "-u", "alice"])
    assert "100" in capsys.readouterr().out


def test_config_path(capsys):
    main(["config", "--path"])
    assert "config.toml" in capsys.readouterr().out


def test_config_summary(capsys):
    main(["config"])
    assert "config path" in capsys.readouterr().out
