"""Tests for the argument tree: mode resolution, validation, and dispatch."""

import argparse
import dataclasses
import sys

import pytest

from jobscope import cli
from jobscope import select as select_mod
from jobscope.cli import build_parser, build_request, default_mode, main, resolve_argv
from jobscope.errors import JobscopeError
from jobscope.select import FINISHED, JOBIDS, RUNNING

# --- level 1: which jobs ----------------------------------------------------

def test_bare_invocation_is_running():
    assert resolve_argv([]) == [RUNNING]


def test_a_window_flag_selects_finished():
    assert resolve_argv(["-D", "3"]) == [FINISHED, "-D", "3"]
    assert resolve_argv(["-N", "20"]) == [FINISHED, "-N", "20"]
    assert resolve_argv(["-S", "2026-07-15"]) == [FINISHED, "-S", "2026-07-15"]
    assert resolve_argv(["-t", "failed"]) == [FINISHED, "-t", "failed"]
    assert resolve_argv(["--days=3"]) == [FINISHED, "--days=3"]


def test_option_values_are_not_mistaken_for_jobids():
    """The bug this replaced: a bare word can be an option's value.

    Scanning argv for non-dash words to spot a JOBID sent `-p kempner_eng` to
    sacct, because "kempner_eng" looks exactly like a job ID from the outside.
    Only argparse knows which flags take an argument, so the mode is keyed on flag
    names alone and JOBIDs are recognised after parsing.
    """
    assert default_mode(["-p", "kempner_eng", "--min-elapsed", "0s"]) == RUNNING
    assert resolve_argv(["-a", "-p", "kempner_eng"]) == [RUNNING, "-a", "-p", "kempner_eng"]


def test_explicit_mode_words_pass_through():
    assert resolve_argv(["running", "--per-gpu"]) == [RUNNING, "--per-gpu"]
    assert resolve_argv(["finished", "-D", "3"]) == [FINISHED, "-D", "3"]


def test_utilities_and_help_pass_through():
    for argv in (["plot"], ["describe", "--dcgm"], ["config"], ["-h"], ["--version"]):
        assert resolve_argv(list(argv)) == argv


def test_a_jobid_needs_no_mode_word():
    # Parsed under `running`; build_request switches to JOBIDS once it sees one.
    assert resolve_argv(["35244230"]) == [RUNNING, "35244230"]
    assert _request(jobids=["35244230"]).mode == JOBIDS


# --- deprecated aliases -----------------------------------------------------

@pytest.mark.parametrize("old,expected,spelling", [
    ("summary", [FINISHED], "the default"),
    ("detail", [RUNNING, "--per-gpu"], "--per-gpu"),
    ("dcgm", [FINISHED, "--dcgm"], "--dcgm"),
    ("live", [RUNNING], "running"),
])
def test_deprecated_subcommands_still_resolve(old, expected, spelling, capsys):
    argv = resolve_argv([old] + (["-D", "1"] if old in ("summary", "dcgm") else []))
    assert argv[0] == expected[0]
    assert spelling in capsys.readouterr().err


def test_deprecated_dcgm_keeps_its_extended_catalog():
    assert "--dcgm" in resolve_argv(["dcgm", "-N", "2"])


def test_deprecated_live_is_always_running():
    # Even though `live` never accepted window flags, be explicit about the mode.
    assert resolve_argv(["live", "-a"])[0] == RUNNING


# --- level 2/3: validation --------------------------------------------------

def _args(**kw):
    base = dict(mode=RUNNING, jobids=[], jobids_opt=None, days=None, lastn=None,
                starttime=None, endtime=None, min_elapsed=None, partition=None,
                user="alice", all_users=False, account=None, state=None,
                per_gpu=False, ts=False, view=None, dcgm=False, avg=False,
                diagnose=False, diag_short=None, header=True, csv=False, step=None,
                timeout=None, workers=None, config_path=None, explicit_mode=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _request(**kw):
    return build_request(_args(**kw))


def test_finished_defaults_to_one_day():
    assert _request(mode=FINISHED).days == 1


def test_running_needs_no_window():
    request = _request(mode=RUNNING)
    assert request.days is None and request.live


def test_explicit_jobids_take_no_default_window():
    assert _request(jobids=["1"]).days is None


@pytest.mark.parametrize("flag,kw", [
    ("-D/--days", {"days": 3}), ("-N/--lastn", {"lastn": 5}),
    ("-S/--starttime", {"starttime": "2026-07-15"}), ("-E/--endtime", {"endtime": "2026-07-16"}),
])
def test_window_flags_are_rejected_for_running(flag, kw):
    with pytest.raises(JobscopeError) as exc:
        _request(mode=RUNNING, **kw)
    # The message must name the flag and offer the fix, not just refuse.
    assert flag in str(exc.value) and "finished" in str(exc.value)


def test_state_is_rejected_for_running():
    with pytest.raises(JobscopeError) as exc:
        _request(mode=RUNNING, state="failed")
    assert "-t/--state" in str(exc.value)


def test_avg_is_rejected_for_finished():
    with pytest.raises(JobscopeError) as exc:
        _request(mode=FINISHED, avg=True)
    assert "--avg" in str(exc.value)


def test_avg_is_accepted_for_running():
    assert _request(mode=RUNNING, avg=True).average is True


def test_days_must_be_positive():
    with pytest.raises(JobscopeError):
        _request(mode=FINISHED, days=0)


def test_days_conflicts_with_lastn_and_window():
    with pytest.raises(JobscopeError):
        _request(mode=FINISHED, days=3, lastn=5)
    with pytest.raises(JobscopeError):
        _request(mode=FINISHED, days=3, starttime="2026-01-01")


def test_lastn_must_be_positive():
    with pytest.raises(JobscopeError):
        _request(mode=FINISHED, lastn=-1)


def test_all_users_conflicts_with_user():
    with pytest.raises(JobscopeError):
        _request(all_users=True, user="alice")


def test_all_users_clears_the_user():
    request = _request(all_users=True, user=None)
    assert request.all_users and request.user is None


def test_bad_min_elapsed_is_rejected():
    with pytest.raises(JobscopeError):
        _request(min_elapsed="1 hour")


def test_jobids_warn_about_ignored_filters(capsys):
    request = _request(jobids=["1"], days=3, partition="kempner")
    err = capsys.readouterr().err
    assert "explicit JOBIDs" in err and "-D/--days" in err and "-p/--partition" in err
    assert request.mode == JOBIDS


def test_partition_survives_in_both_modes():
    assert _request(mode=RUNNING, partition="kempner").partition == "kempner"
    assert _request(mode=FINISHED, partition="kempner").partition == "kempner"


# --- granularity and columns compose ---------------------------------------

@pytest.mark.parametrize("flag", ["--per-gpu", "--hwdetail"])
def test_hwdetail_is_still_accepted_as_the_old_spelling(flag):
    """The rename must not break a script, alias or shell history line."""
    _, subparsers = build_parser()
    args = subparsers.choices[RUNNING].parse_intermixed_args([flag])
    assert args.per_gpu is True


def test_per_gpu_and_ts_are_mutually_exclusive():
    _, subparsers = build_parser()
    with pytest.raises(SystemExit):
        subparsers.choices[RUNNING].parse_intermixed_args(["--per-gpu", "--ts"])


def test_cpu_and_gpu_are_mutually_exclusive():
    _, subparsers = build_parser()
    with pytest.raises(SystemExit):
        subparsers.choices[RUNNING].parse_intermixed_args(["--cpu", "--gpu"])


@pytest.mark.parametrize("mode", [RUNNING, FINISHED])
@pytest.mark.parametrize("argv", [
    [], ["--per-gpu"], ["--ts"], ["--cpu"], ["--gpu"], ["--dcgm"], ["--diagnose"],
    ["--per-gpu", "--dcgm"], ["--ts", "--dcgm"], ["--gpu", "--dcgm", "--diagnose"],
    ["-p", "kempner"], ["-a"], ["--csv"], ["-n"],
])
def test_every_option_parses_in_every_mode(mode, argv):
    """The point of the restructure: no option is stranded on one mode."""
    _, subparsers = build_parser()
    args = subparsers.choices[mode].parse_intermixed_args(argv)
    assert args.mode == mode


def test_args_order_independent():
    _, subparsers = build_parser()
    finished = subparsers.choices[FINISHED]
    a1 = finished.parse_intermixed_args(["-D", "3", "111", "222"])
    a2 = finished.parse_intermixed_args(["111", "-D", "3", "222"])
    a3 = finished.parse_intermixed_args(["111", "222", "-D", "3"])
    assert a1.jobids == a2.jobids == a3.jobids == ["111", "222"]
    assert a1.days == a2.days == a3.days == 3


def test_jobid_flag_merges_with_positional():
    _, subparsers = build_parser()
    for argv in (["111", "222"], ["-j", "111", "-j", "222"], ["-j", "111", "222"]):
        args = subparsers.choices[FINISHED].parse_intermixed_args(argv)
        args.mode, args.user = FINISHED, "alice"
        assert set(build_request(args).jobids) == {"111", "222"}


def test_diag_short_replaces_the_old_min_runtime():
    _, subparsers = build_parser()
    args = subparsers.choices[FINISHED].parse_intermixed_args(["--diag-short", "300"])
    assert args.diag_short == 300
    # --min-runtime now means the runtime floor, not the DIAG threshold.
    args = subparsers.choices[RUNNING].parse_intermixed_args(["--min-runtime", "5m"])
    assert args.min_elapsed == "5m"


# --- dispatch ---------------------------------------------------------------

def test_describe_command(capsys):
    main(["describe"])
    assert "CPU%" in capsys.readouterr().out


def test_describe_dcgm_ext(capsys):
    main(["describe", "--dcgm", "--ext"])
    out = capsys.readouterr().out
    assert "DCGM GPU metrics" in out
    from jobscope.dcgm import ALL_SPECS
    assert "%d metrics" % len(ALL_SPECS) in out


def test_config_example(capsys):
    main(["config", "--example"])
    assert "[prometheus]" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "jobscope" in capsys.readouterr().out


def _patch_sacct(monkeypatch, records, chunks=None, ids=None):
    """Stub the sacct side of the selection layer."""
    ids = ids if ids is not None else list(records)
    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, timeout: (ids, "last 1 day"))
    monkeypatch.setattr(select_mod, "fetch", lambda i, timeout: records)
    monkeypatch.setattr(select_mod, "fetch_chunks",
                        lambda i, timeout: iter(chunks if chunks is not None
                                                else [(list(i), records)]))


def test_finished_cpu_view(monkeypatch, capsys, cpu_record):
    _patch_sacct(monkeypatch, {"200": cpu_record})
    main(["finished", "--cpu", "-D", "1", "-u", "bob"])
    out = capsys.readouterr().out
    assert "200" in out and "CPU%" in out and "SM_ACT%" not in out


def test_finished_streams_chunks_per_batch(monkeypatch, capsys, gpu_record):
    rec2 = dataclasses.replace(gpu_record, jobid="101", name="eval")
    records = {"100": gpu_record, "101": rec2}
    _patch_sacct(monkeypatch, records, chunks=[(["100"], records), (["101"], records)],
                 ids=["100", "101"])
    clients = []
    monkeypatch.setattr(select_mod, "client_from_config",
                        lambda cfg, timeout: clients.append(1) or object())
    overall = {"SM_ACT%": 60.0, "OCC%": 20.0, "TENSOR%": 5.0, "DRAM%": 10.0, "POWER_W": 400.0}
    seen = []

    def fake_compute(records_arg, chunk_ids, *a, **k):
        seen.append(list(chunk_ids))
        return {j: (overall, {}) for j in chunk_ids}

    monkeypatch.setattr(select_mod, "compute_dcgm", fake_compute)
    main(["finished", "-D", "1", "-u", "alice"])
    out = capsys.readouterr().out
    assert seen == [["100"], ["101"]]      # DCGM computed per chunk, still streaming
    assert len(clients) == 1               # one Prometheus client
    # One footer, printed once at the end, after every streamed chunk.
    assert out.count("Used/") == 1 and out.index("Used/") > out.index("101")


def test_explicit_jobids_do_not_stream(monkeypatch, capsys, cpu_record):
    def no_stream(*a, **k):
        raise AssertionError("fetch_chunks must not be used for explicit job IDs")

    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: (["111"], "1 job ID(s)"))
    monkeypatch.setattr(select_mod, "fetch", lambda i, t: {"111": cpu_record})
    monkeypatch.setattr(select_mod, "fetch_chunks", no_stream)
    main(["--cpu", "111"])
    assert "111" in capsys.readouterr().out


def test_per_gpu_renders_one_row_per_gpu(monkeypatch, capsys, gpu_record):
    _patch_sacct(monkeypatch, {"100": gpu_record})
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, timeout: object())
    monkeypatch.setattr(select_mod, "compute_dcgm",
                        lambda *a, **k: {"100": ({}, {("node01", "0"): {"SM_ACT%": 80.0}})})
    main(["finished", "--per-gpu", "-D", "1", "-u", "alice"])
    out = capsys.readouterr().out
    assert "Job 100" in out and "NODE" in out and "node01" in out


def test_no_matching_jobs_is_not_an_error(monkeypatch, capsys):
    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, timeout: ([], "last 1 day"))
    main(["finished", "--cpu", "-u", "nobody"])
    assert "No matching jobs" in capsys.readouterr().err


def test_finished_timeseries(monkeypatch, capsys, gpu_record):
    monkeypatch.setattr(select_mod, "select_jobs", lambda sel, t: (["100"], "x"))
    monkeypatch.setattr(select_mod, "fetch", lambda i, t: {"100": gpu_record})

    class Client:
        sampling_period = 60

        def query(self, query, at, timeout=None):
            return [{"metric": {"uuid": "U0", "host": "node01:9400", "minor_number": "0"}}]

        def query_range(self, query, start, end, step, timeout=None):
            return ([{"metric": {"UUID": "U0"}, "values": [[1000, "0.8"]]}]
                    if "DCGM_FI_PROF_SM_ACTIVE" in query else [])

    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, timeout: Client())
    main(["--ts", "100"])
    out = capsys.readouterr().out
    assert out.startswith("JOBID,EPOCH,TIME,NODE,GPU,") and "80.0" in out


# --- the running branch -----------------------------------------------------

GIB = 1024 ** 3


class _LiveClient:
    """Prometheus stand-in for the squeue branch: one GPU on one job.

    Serves the NVML and cgroup series as well as a DCGM one, because the live path
    reconstructs the utilization blob from them -- without those, a running job has
    no per-GPU rows to show under --per-gpu.
    """

    sampling_period = 60

    def query(self, query, at, timeout=None):
        if "nvidia_gpu_jobId" in query:
            return [{"metric": {"uuid": "U0", "host": "node01:9445", "minor_number": "3"},
                     "value": [at, "4.2e+07"]}]
        nvml = {"nvidia_gpu_duty_cycle": 90,
                "nvidia_gpu_memory_used_bytes": 40 * GIB,
                "nvidia_gpu_memory_total_bytes": 80 * GIB}
        for name, value in nvml.items():
            if name in query:
                return [{"metric": {"uuid": "U0"}, "value": [at, str(value)]}]
        cgroup = {"cgroup_cpus": 2, "cgroup_cpu_total_seconds": 5400,
                  "cgroup_memory_rss_bytes": 8 * GIB,
                  "cgroup_memory_total_bytes": 16 * GIB}
        for name, value in cgroup.items():
            if name in query:
                # The jobid label matters: these are batched into one query per
                # field across every job, then demultiplexed on it.
                return [{"metric": {"host": "node01:9306", "jobid": "42000000"},
                         "value": [at, str(value)]}]
        if "DCGM_FI_PROF_SM_ACTIVE" in query:
            return [{"metric": {"UUID": "U0"}, "value": [at, "0.776"]}]
        return []

    def query_range(self, query, start, end, step, timeout=None):
        return ([{"metric": {"UUID": "U0"}, "values": [[1000, "0.8"]]}]
                if "DCGM_FI_PROF_SM_ACTIVE" in query else [])


_LIVE_JOB = {42000000: {"jobid": "100_6", "user": "alice", "node": "node01", "name": "train",
                        "start_epoch": 1000, "elapsed_seconds": 3600}}


def _patch_squeue(monkeypatch):
    monkeypatch.setattr(select_mod, "fetch_jobs", lambda sel, timeout: dict(_LIVE_JOB))
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, timeout: _LiveClient())


def test_running_table(monkeypatch, capsys):
    _patch_squeue(monkeypatch)
    main(["running", "-j", "100_6"])
    out = capsys.readouterr().out
    assert "100_6" in out and "alice" in out and "RUNNING" in out and "77.6" in out
    # CPU% comes from cgroup_*: 100 * 5400 / (3600 * 2) = 75.
    assert "75" in out


def test_running_is_the_default_mode(monkeypatch, capsys):
    _patch_squeue(monkeypatch)

    def no_sacct(*a, **k):
        raise AssertionError("a bare invocation must not reach sacct")

    monkeypatch.setattr(select_mod, "select_jobs", no_sacct)
    main([])
    assert "RUNNING" in capsys.readouterr().out


def test_running_per_gpu(monkeypatch, capsys):
    """The per-GPU granularity for a running job, off the reconstructed blob."""
    _patch_squeue(monkeypatch)
    main(["running", "--per-gpu", "-j", "100_6"])
    out = capsys.readouterr().out
    assert "Job 100_6" in out and "node01" in out
    assert "NODE" in out and "GPU" in out
    assert "77.6" in out                    # the DCGM column, keyed (node, minor)


def test_running_timeseries(monkeypatch, capsys):
    _patch_squeue(monkeypatch)
    main(["running", "--ts", "-j", "100_6"])
    out = capsys.readouterr().out
    assert out.startswith("JOBID,EPOCH,TIME,NODE,GPU,")


def test_running_no_matching_jobs(monkeypatch, capsys):
    monkeypatch.setattr(select_mod, "fetch_jobs", lambda sel, timeout: {})
    main(["running", "-a"])
    assert "No running jobs match" in capsys.readouterr().err


def test_diagnose_ignored_for_the_cpu_view(monkeypatch, capsys, cpu_record):
    _patch_sacct(monkeypatch, {"200": cpu_record})
    main(["finished", "--cpu", "--diagnose", "-D", "1", "-u", "bob"])
    assert "ignoring it for --cpu" in capsys.readouterr().err


def test_running_blob_is_reconstructed(monkeypatch, capsys, gpu_record):
    """A running job selected by ID has no blob, so it is rebuilt from Prometheus."""
    running = dataclasses.replace(gpu_record, jobid="300", state="RUNNING", stats={})
    _patch_sacct(monkeypatch, {"300": running})
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, timeout: object())
    monkeypatch.setattr(select_mod, "compute_dcgm", lambda *a, **k: {"300": ({}, {})})

    def fake_fill(records, ids, client, timeout=None, workers=1):
        records["300"].stats = gpu_record.stats
        return 1

    monkeypatch.setattr(select_mod, "fill_running", fake_fill)
    main(["300"])
    out = capsys.readouterr().out
    assert "70" in out and "50" in out      # the blob fixture's gpu/gmem


def test_offline_view_warns_when_no_endpoint_can_fill(monkeypatch, capsys, gpu_record):
    running = dataclasses.replace(gpu_record, jobid="300", state="RUNNING", stats={})
    _patch_sacct(monkeypatch, {"300": running})

    def no_endpoint(cfg, timeout):
        raise JobscopeError("no Prometheus endpoint configured")

    monkeypatch.setattr(select_mod, "client_from_config", no_endpoint)
    main(["finished", "--cpu", "-D", "1", "-u", "alice"])
    err = capsys.readouterr().err
    assert "no Prometheus" in err and "blank" in err


def test_finished_job_blob_is_never_recomputed(monkeypatch, capsys, gpu_record):
    _patch_sacct(monkeypatch, {"100": gpu_record})
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, timeout: object())
    monkeypatch.setattr(select_mod, "compute_dcgm", lambda *a, **k: {"100": ({}, {})})

    def boom(*a, **k):
        raise AssertionError("a stored blob must not be refetched")

    monkeypatch.setattr(select_mod, "fill_running", boom)
    main(["finished", "-D", "1", "-u", "alice"])
    assert "100" in capsys.readouterr().out


def test_workers_must_be_positive(monkeypatch, capsys, cpu_record):
    _patch_sacct(monkeypatch, {"200": cpu_record})
    with pytest.raises(SystemExit):
        main(["finished", "--workers", "0", "-D", "1"])
    assert "--workers" in capsys.readouterr().err


def test_config_path(capsys):
    main(["config", "--path"])
    assert "config.toml" in capsys.readouterr().out


def test_config_summary(capsys):
    main(["config"])
    assert "config path" in capsys.readouterr().out


def test_cli_module_exposes_the_mode_words():
    assert cli.MODES == (RUNNING, FINISHED)


# --- the runtime floor ------------------------------------------------------

def test_min_elapsed_defaults_to_ten_minutes():
    """A job still loading data reads as idle, so the floor hides the youngest."""
    from jobscope.config import DEFAULT_MIN_ELAPSED
    assert DEFAULT_MIN_ELAPSED == "10m"
    assert _request(min_elapsed=None).min_elapsed == 600


def test_min_elapsed_comes_from_config_when_unset():
    import dataclasses as dc

    from jobscope import config
    base = config.get_config()
    cfg = dc.replace(base, defaults=dc.replace(base.defaults, min_elapsed="45m"))
    assert build_request(_args(min_elapsed=None), cfg).min_elapsed == 45 * 60


def test_the_flag_overrides_the_configured_floor():
    import dataclasses as dc

    from jobscope import config
    base = config.get_config()
    cfg = dc.replace(base, defaults=dc.replace(base.defaults, min_elapsed="45m"))
    assert build_request(_args(min_elapsed="30s"), cfg).min_elapsed == 30


def test_zero_disables_the_floor():
    assert _request(min_elapsed="0s").min_elapsed == 0


def test_a_bad_configured_floor_blames_the_config():
    import dataclasses as dc

    from jobscope import config
    base = config.get_config()
    cfg = dc.replace(base, defaults=dc.replace(base.defaults, min_elapsed="1 hour"))
    with pytest.raises(JobscopeError) as exc:
        build_request(_args(min_elapsed=None), cfg)
    # Not phrased as a bad command line, since the command line was fine.
    assert "config file" in str(exc.value) and "min_elapsed" in str(exc.value)


# --- when to tint -----------------------------------------------------------

def _color_args(**kw):
    base = dict(csv=False, no_color=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_color_needs_a_terminal(monkeypatch):
    """Escape codes in a redirected file are corruption, not decoration."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(cli.sys, "stdout", type("S", (), {"isatty": lambda self: True})())
    assert cli._want_color(_color_args()) is True
    monkeypatch.setattr(cli.sys, "stdout", type("S", (), {"isatty": lambda self: False})())
    assert cli._want_color(_color_args()) is False


def test_csv_and_no_color_and_the_env_var_all_disable_it(monkeypatch):
    monkeypatch.setattr(cli.sys, "stdout", type("S", (), {"isatty": lambda self: True})())
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert cli._want_color(_color_args(csv=True)) is False
    assert cli._want_color(_color_args(no_color=True)) is False
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli._want_color(_color_args()) is False


def test_a_stdout_without_isatty_is_treated_as_not_a_terminal(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(cli.sys, "stdout", object())
    assert cli._want_color(_color_args()) is False


# --- which weighting the mean footer uses -----------------------------------

def _options_for(argv, monkeypatch):
    """The RenderOptions handle_report builds for `argv`, without running a query."""
    captured = {}

    class FakeRenderer:
        def __init__(self, context, options, **kw):
            captured["options"] = options

        def add(self, *a, **kw):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(cli, "SummaryRenderer", FakeRenderer)
    # --per-gpu routes to the other renderer, whose finish() would otherwise object
    # to a --nodename that matched nothing in an empty fake selection.
    monkeypatch.setattr(cli, "DetailRenderer", FakeRenderer)
    monkeypatch.setattr(cli, "resolve",
                        lambda *a, **kw: select_mod.Resolved(context=[], chunks=[]))
    main(argv)
    return captured["options"]


def test_finished_jobs_are_weighted_by_resource_time(monkeypatch):
    """A finished job's values span its whole runtime, so runtime is a valid weight."""
    assert _options_for(["finished", "-D", "1"], monkeypatch).time_weighted is True


def test_an_explicit_job_id_is_weighted_by_resource_time(monkeypatch):
    assert _options_for(["35244230"], monkeypatch).time_weighted is True


def test_the_running_snapshot_is_not_weighted_by_time(monkeypatch):
    """Every value is the same instant; elapsed time is not evidence about it."""
    assert _options_for(["running"], monkeypatch).time_weighted is False


def test_running_avg_is_weighted_by_time(monkeypatch):
    """--avg folds each job over its own runtime, which restores the premise."""
    assert _options_for(["running", "--avg"], monkeypatch).time_weighted is True


def test_the_efficiency_bars_are_on_by_default(monkeypatch):
    assert _options_for(["finished", "-D", "1"], monkeypatch).plot_avgeff


def test_no_plot_switches_the_bars_off(monkeypatch):
    assert not _options_for(["finished", "-D", "1", "--no-plot"], monkeypatch).plot_avgeff


def test_the_old_plot_flag_is_accepted_and_notes(monkeypatch, capsys):
    """It shipped for one commit; a command naming it should still run."""
    for spelling in ("--plot_avgeff", "--plot-avgeff"):
        assert _options_for(["finished", "-D", "1", spelling], monkeypatch).plot_avgeff
        assert "is the default now" in capsys.readouterr().err


def test_the_timeseries_path_cannot_reach_the_summary_renderer(monkeypatch):
    """--ts is a per-scrape CSV, so no section furniture can leak into it.

    Guaranteed structurally rather than by a flag check: handle_report routes --ts to
    select.emit_timeseries, which never constructs a SummaryRenderer. This pins that
    routing, since a future refactor could quietly reintroduce one.
    """
    called = {}
    monkeypatch.setattr(cli, "emit_timeseries",
                        lambda *a, **kw: called.setdefault("ts", True))
    monkeypatch.setattr(cli, "SummaryRenderer",
                        lambda *a, **kw: pytest.fail("--ts built a SummaryRenderer"))
    monkeypatch.setattr(cli, "resolve",
                        lambda *a, **kw: pytest.fail("--ts resolved a summary chunk"))
    main(["finished", "-D", "1", "--ts", "--csv"])
    assert called == {"ts": True}


def _request_for(argv, monkeypatch):
    """The Request handle_report builds for `argv`, without running a query."""
    captured = {}

    def fake(request, *a, **kw):
        captured["req"] = request
        return select_mod.Resolved(context=[], chunks=[])

    monkeypatch.setattr(cli, "resolve", fake)
    monkeypatch.setattr(cli, "SummaryRenderer",
                        lambda *a, **kw: type("R", (), {"add": lambda *x: None,
                                                        "finish": lambda *x: None})())
    main(argv)
    return captured["req"]


def test_finished_defaults_to_completed_only(monkeypatch):
    """`finished` used to include running jobs, which have no final numbers."""
    assert _request_for(["finished", "-D", "1"], monkeypatch).state == "completed"


def test_dash_t_reaches_the_request_verbatim(monkeypatch):
    """sacct.states_for does the validating, so the CLI passes the string through."""
    got = _request_for(["finished", "-D", "1", "-t", "failed,timeout"], monkeypatch)
    assert got.state == "failed,timeout"


def test_dash_t_still_implies_the_finished_mode():
    assert resolve_argv(["-t", "timeout"]) == [FINISHED, "-t", "timeout"]


def test_an_explicit_running_plus_state_is_rejected(capsys):
    with pytest.raises(SystemExit):
        main(["running", "-t", "completed"])
    assert "does not apply to running jobs" in capsys.readouterr().err


def test_nodename_reaches_the_renderer(monkeypatch):
    got = _options_for(["-j", "1", "--per-gpu", "--nodename=holygpu8a10401"], monkeypatch)
    assert got.nodename == "holygpu8a10401"


def test_node_is_accepted_as_an_alias(monkeypatch):
    assert _options_for(["-j", "1", "--per-gpu", "--node", "n1"],
                        monkeypatch).nodename == "n1"


def test_nodename_without_a_per_gpu_view_is_rejected(capsys):
    """The per-job table's NODE column is a count, so there is no name to match."""
    with pytest.raises(SystemExit):
        main(["-j", "1", "--nodename=n1"])
    err = capsys.readouterr().err
    assert "needs --per-gpu or --ts" in err


def test_nodename_reaches_the_timeseries_emitter(monkeypatch):
    """--ts carries a NODE column too, so the filter applies to it."""
    captured = {}
    monkeypatch.setattr(cli, "emit_timeseries",
                        lambda *a, **kw: captured.setdefault("options", a[-1]))
    main(["-j", "1", "--ts", "--nodename=n1"])
    assert captured["options"].nodename == "n1"


# --- the help narrows to what the invocation can use ------------------------

def _help_for(argv, capsys):
    """``(options_shown, hidden_flags)`` for ``jobscope <argv> --help``.

    The body is cut at the footer, which names the hidden flags and would otherwise
    make every "not offered" assertion pass by accident.
    """
    with pytest.raises(SystemExit):
        main(list(argv) + ["--help"])
    out = capsys.readouterr().out
    marker = "hiding "
    if marker not in out:
        return out, []
    body, footer = out[:out.index(marker)], out[out.index(marker):]
    hidden = [f.strip() for f in footer.split(".\n")[0].split(":", 1)[1].split(",")]
    return body, hidden


def test_the_help_hides_the_flags_an_explicit_jobid_makes_inert(capsys):
    """The example that prompted this: 30 options for a command that can use 17."""
    body, hidden = _help_for(["-j", "36441613", "--per-gpu"], capsys)
    # The job ID *is* the selection, so no window and no filter can narrow it.
    for flag in ("--days", "--lastn", "--starttime", "--endtime",
                 "--partition", "--user", "--all-users", "--account", "--state"):
        assert flag in hidden
        assert flag not in body
    assert "--ts" in hidden           # mutually exclusive with --per-gpu
    assert "--step" in hidden         # only emit_timeseries reads it
    assert "--avg" in hidden          # running only
    # What remains is what this command actually honours.
    for flag in ("--nodename", "--dcgm", "--csv", "--no-plot", "--per-gpu"):
        assert flag in body


def test_the_hidden_set_is_exactly_what_the_run_would_reject_or_ignore(capsys):
    """No taste involved: every hidden flag maps to an error or an ignore.

    --state is in the list because select.py returns explicit IDs as-is, so the -s
    filter never runs -- the same reason build_request now names it in its note.
    """
    assert {name for name, _ in cli._JOBID_IGNORES} == {
        "-D/--days", "-N/--lastn", "-S/--starttime", "-E/--endtime",
        "-p/--partition", "-u/--user", "-a/--all-users", "-A/--account",
        "-t/--state"}


def test_running_hides_the_past_window_flags(capsys):
    _, hidden = _help_for(["running"], capsys)
    assert {"--days", "--lastn", "--starttime", "--endtime", "--state"} <= set(hidden)
    assert "--min-elapsed" not in hidden   # the running view is the one that uses it


def test_finished_hides_the_running_only_flags(capsys):
    _, hidden = _help_for(["finished"], capsys)
    assert {"--avg", "--min-elapsed"} <= set(hidden)
    assert "--days" not in hidden


def test_ts_hides_what_the_series_drops(capsys):
    """emit_timeseries notes these as dropped; the help should not offer them."""
    body, hidden = _help_for(["--ts"], capsys)
    assert {"--cpu", "--gpu", "--diagnose", "--per-gpu", "--no-plot"} <= set(hidden)
    assert "--step" in body      # --ts is the only thing that reads it
    assert "--nodename" in body  # the series carries a NODE column, so it filters


def test_cpu_hides_the_gpu_only_columns(capsys):
    _, hidden = _help_for(["--cpu"], capsys)
    assert {"--dcgm", "--diagnose", "--diag-short"} <= set(hidden)


def test_csv_hides_what_a_csv_cannot_carry(capsys):
    _, hidden = _help_for(["--csv"], capsys)
    assert {"--no-color", "--no-plot"} <= set(hidden)


def test_the_per_job_view_hides_nodename(capsys):
    """It raises without --per-gpu, so offering it is a dead end."""
    _, hidden = _help_for(["running"], capsys)
    assert "--nodename" in hidden


def test_help_all_hides_nothing(capsys):
    with pytest.raises(SystemExit):
        main(["-j", "1", "--per-gpu", "--help-all"])
    body = capsys.readouterr().out
    assert "hiding" not in body
    for flag in ("--days", "--partition", "--avg", "--step", "--ts"):
        assert flag in body


def test_the_narrowed_help_names_the_way_back(capsys):
    """Nothing is invisible: the footer counts what went and how to get it back."""
    with pytest.raises(SystemExit):
        main(["-j", "1", "--help"])
    out = capsys.readouterr().out
    assert "hiding " in out and "--help-all for the full list" in out


def test_the_top_level_help_is_untouched(capsys):
    """`jobscope --help` has no flags to narrow against; it lists the subcommands."""
    with pytest.raises(SystemExit):
        main(["--help"])
    body = capsys.readouterr().out
    assert "{running,finished,plot,describe,config}" in body
    assert "hiding" not in body


def test_a_tail_argparse_cannot_parse_narrows_nothing():
    _, subparsers = build_parser()
    assert cli.narrow_help(subparsers.choices[FINISHED], ["-D", "notanumber"], False) == []


def test_narrowing_leaves_a_formattable_usage_line(capsys):
    """A suppressed member of a mutually exclusive group used to crash the formatter.

    argparse renders "[--per-gpu | --ts]" from the group while building the option
    list from the visible actions; hide one and the two disagree, which trips an
    assert (and an emptied group raises outright).
    """
    for argv in (["--ts"], ["--per-gpu"], ["--cpu", "--ts"], ["-j", "1", "--per-gpu"]):
        body, _ = _help_for(argv, capsys)
        assert body.startswith("usage: jobscope")


# --- --plot_ts: the time-series chart in one command -------------------------

_TS_HEAD = "JOBID,EPOCH,TIME,NODE,GPU,GPU%,GMEM%\n"


def _ts_rows(jobids=("100",), nodes=("node01",), gpus=("0", "1", "2", "3")):
    return "".join(
        "%s,%d,2020-01-01T00:%02d:00,%s,%s,%d,%d\n" % (j, 1000 + 60 * t, t, n, g, 90 + t, 50 + t)
        for j in jobids for n in nodes for g in gpus for t in range(3))


def _fake_ts(monkeypatch, body):
    """Stand in for emit_timeseries, writing `body` to wherever it was told to."""
    def emit(*a, **kw):
        (kw.get("out") or sys.stdout).write(_TS_HEAD + body)
    monkeypatch.setattr(cli, "emit_timeseries", emit)


@pytest.mark.parametrize("flag", ["--plot-ts", "--plot_ts"])
def test_plot_ts_is_the_timeseries_plus_a_chart(flag):
    """It implies --ts, so every --ts path -- schema, --step, --nodename -- applies."""
    _, subparsers = build_parser()
    args = subparsers.choices[RUNNING].parse_intermixed_args([flag])
    assert args.plot_ts is True


def test_plot_ts_charts_instead_of_writing_the_csv(monkeypatch, capsys):
    _fake_ts(monkeypatch, _ts_rows())
    main(["-j", "1", "--plot_ts"])
    out = capsys.readouterr().out
    assert "JOBID,EPOCH" not in out      # the CSV went to the chart, not to stdout
    assert "┤" in out and "GPU%" in out


def test_plot_ts_takes_its_column_count_from_the_data(monkeypatch, capsys):
    """No --gpu to type: the CSV already says how many there are."""
    monkeypatch.setenv("COLUMNS", "210")     # wide enough for all four abreast
    _fake_ts(monkeypatch, _ts_rows(gpus=("0", "1", "2", "3")))
    main(["-j", "1", "--plot_ts"])
    titles = [ln for ln in capsys.readouterr().out.splitlines() if "gpu0" in ln]
    assert titles and all(("gpu%d" % g) in titles[0] for g in range(4))


def test_plot_ts_narrows_the_columns_to_the_terminal(monkeypatch, capsys):
    """Same 4 -> fewer degradation the --per-gpu charts have; the rest wrap."""
    monkeypatch.setenv("COLUMNS", "100")
    _fake_ts(monkeypatch, _ts_rows(gpus=("0", "1", "2", "3")))
    main(["-j", "1", "--plot_ts"])
    titles = [ln for ln in capsys.readouterr().out.splitlines() if "gpu0" in ln]
    assert "gpu3" not in titles[0]


def test_plot_ts_needs_a_nodename_when_the_job_spanned_nodes(monkeypatch, capsys):
    """The multinode branch charts one metric per node, which is not what was asked."""
    _fake_ts(monkeypatch, _ts_rows(nodes=("node01", "node02")))
    with pytest.raises(SystemExit):
        main(["-j", "1", "--plot_ts"])
    err = capsys.readouterr().err
    assert "node01, node02" in err and "--nodename" in err


def test_plot_ts_needs_no_nodename_for_a_single_node_job(monkeypatch, capsys):
    _fake_ts(monkeypatch, _ts_rows(nodes=("node01",)))
    main(["-j", "1", "--plot_ts"])
    assert "┤" in capsys.readouterr().out


def test_plot_ts_refuses_several_jobs(monkeypatch, capsys):
    """Series key on (NODE, GPU) alone, so two jobs on one GPU would become one line."""
    _fake_ts(monkeypatch, _ts_rows(jobids=("100", "101")))
    with pytest.raises(SystemExit):
        main(["finished", "-D", "1", "--plot_ts"])
    err = capsys.readouterr().err
    assert "one job" in err and "100, 101" in err and "-j JOBID" in err


def test_plot_ts_and_csv_contradict(capsys):
    with pytest.raises(SystemExit):
        main(["-j", "1", "--plot_ts", "--csv"])
    assert "drop --csv" in capsys.readouterr().err


@pytest.mark.parametrize("other", ["--ts", "--per-gpu"])
def test_plot_ts_is_exclusive_with_the_other_granularities(other):
    _, subparsers = build_parser()
    with pytest.raises(SystemExit):
        subparsers.choices[RUNNING].parse_intermixed_args(["--plot-ts", other])


def test_plot_ts_help_hides_what_it_rules_out(capsys):
    _, hidden = _help_for(["-j", "1", "--plot-ts"], capsys)
    assert {"--csv", "--ts", "--per-gpu"} <= set(hidden)
    assert "--nodename" not in hidden      # the flag it points you at


# --- --ts / --plot_ts take an optional window --------------------------------

@pytest.mark.parametrize("spec,seconds", [
    ("30s", 30), ("90m", 5400), ("1h", 3600), ("2d", 172800),
])
def test_the_ts_window_is_a_duration(spec, seconds):
    _, subparsers = build_parser()
    args = subparsers.choices[RUNNING].parse_intermixed_args(["--ts", spec])
    assert cli._ts_window(args) == seconds


def test_no_window_means_the_whole_run():
    _, subparsers = build_parser()
    args = subparsers.choices[RUNNING].parse_intermixed_args(["--ts"])
    assert args.ts is True and cli._ts_window(args) is None


def test_a_jobid_written_after_ts_is_still_a_jobid(capsys):
    """--ts grew an optional value, and must not eat the job that follows it.

    The grammars are disjoint -- a window carries a unit, a job ID never does -- so
    the value is handed back rather than guessed at.
    """
    _, subparsers = build_parser()
    args = subparsers.choices[RUNNING].parse_intermixed_args(["--ts", "36441613"])
    cli._reclaim_jobid_after_ts(args)
    assert args.jobids == ["36441613"] and args.ts is True
    assert cli._ts_window(args) is None
    assert "as a job ID" in capsys.readouterr().err     # and it says which reading


@pytest.mark.parametrize("jobid", ["36441613", "36609689_1", "100.batch"])
def test_array_tasks_and_steps_are_reclaimed_too(jobid, capsys):
    _, subparsers = build_parser()
    args = subparsers.choices[RUNNING].parse_intermixed_args(["--ts", jobid])
    cli._reclaim_jobid_after_ts(args)
    assert args.jobids == [jobid]


def test_a_window_is_not_mistaken_for_a_jobid(capsys):
    _, subparsers = build_parser()
    args = subparsers.choices[RUNNING].parse_intermixed_args(["--ts", "1h"])
    cli._reclaim_jobid_after_ts(args)
    assert args.jobids == [] and cli._ts_window(args) == 3600
    assert capsys.readouterr().err == ""


def test_an_unparseable_window_says_what_a_window_looks_like():
    _, subparsers = build_parser()
    args = subparsers.choices[RUNNING].parse_intermixed_args(["--ts", "abc"])
    with pytest.raises(JobscopeError) as exc:
        cli._ts_window(args)
    assert "duration with a unit" in str(exc.value) and "1h" in str(exc.value)


def test_plot_ts_takes_the_same_window():
    _, subparsers = build_parser()
    args = subparsers.choices[RUNNING].parse_intermixed_args(["--plot_ts", "1h"])
    assert cli._ts_window(args) == 3600


def test_the_window_reaches_the_emitter(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "emit_timeseries",
                        lambda *a, **kw: captured.setdefault("options", a[-1]))
    main(["-j", "1", "--ts", "1h"])
    assert captured["options"].window == 3600


def test_the_chart_says_which_window_it_is_showing(monkeypatch, capsys):
    """Otherwise a windowed chart is indistinguishable from a whole-run one: the x
    axis counts minutes from the window's own start either way."""
    _fake_ts(monkeypatch, _ts_rows())
    main(["-j", "1", "--plot_ts", "1h"])
    assert "last 1h" in capsys.readouterr().out


def test_stats_summarizes_instead_of_writing_the_csv(monkeypatch, capsys):
    _fake_ts(monkeypatch, _ts_rows(gpus=("0",)))
    main(["-j", "1", "--ts", "--stats"])
    out = capsys.readouterr().out
    assert "JOBID,EPOCH" not in out              # the CSV became the summary
    assert "MEAN" in out and "GPU%" in out


def test_stats_needs_a_timeseries(capsys):
    with pytest.raises(SystemExit):
        main(["-j", "1", "--stats"])
    assert "add --ts" in capsys.readouterr().err


def test_stats_is_redundant_with_plot_ts(monkeypatch, capsys):
    """The chart already prints min/mean/max/last under it."""
    _fake_ts(monkeypatch, _ts_rows(gpus=("0",)))
    main(["-j", "1", "--plot_ts", "--stats"])
    assert "already prints" in capsys.readouterr().err


def test_stats_composes_with_the_window(monkeypatch, capsys):
    captured = {}

    def emit(*a, **kw):
        captured["options"] = a[-1]
        kw["out"].write(_TS_HEAD + _ts_rows(gpus=("0",)))

    monkeypatch.setattr(cli, "emit_timeseries", emit)
    main(["-j", "1", "--ts", "30m", "--stats"])
    assert captured["options"].window == 1800      # the window narrowed the query
    assert "MEAN" in capsys.readouterr().out       # and the summary still rendered
