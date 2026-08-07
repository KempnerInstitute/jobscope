"""Tests for the argument tree: mode resolution, validation, and dispatch."""

import argparse
import re
import dataclasses
import sys

import pytest

from jobscope import cli, dcgm
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
    for argv in (["plot"], ["describe", "--metrics"], ["config"], ["-h"], ["--version"]):
        assert resolve_argv(list(argv)) == argv


def test_a_jobid_needs_no_mode_word():
    # Parsed under `running`; build_request switches to JOBIDS once it sees one.
    assert resolve_argv(["35244230"]) == [RUNNING, "35244230"]
    assert _request(jobids=["35244230"]).mode == JOBIDS


# --- the modes ---------------------------------------------------------------

def test_doctor_was_renamed_to_probe(capsys):
    """It is in RETIRED for the same reason as the others: without an entry the word
    falls through as a would-be JOBID and Slurm answers "Bad job/step specified"."""
    with pytest.raises(SystemExit):
        main(["doctor"])
    err = capsys.readouterr().err
    assert "renamed" in err and "'probe'" in err


@pytest.mark.parametrize("flag", sorted(cli.RETIRED_FLAGS))
def test_every_retired_flag_names_its_replacement(flag, capsys):
    """The table's whole reason for existing: a retired spelling must not become
    argparse's bare "unrecognized arguments", which says nothing about what to type.
    Parametrized over the table itself, so an entry added without an argparse action
    to back it fails here rather than the next time someone types it.

    Tried bare and then under `probe`, because a retired spelling belongs wherever the
    live one did -- --validate was a probe flag, so that is the parser its action is
    registered on. The assertion is that *some* invocation names the replacement; an
    entry backed by no action anywhere still fails.
    """
    for argv in ([flag], ["probe", flag]):
        capsys.readouterr()
        try:
            main(argv)
        except SystemExit:
            pass
        if "no longer a flag" in capsys.readouterr().err:
            break
    else:
        raise AssertionError("%s is in RETIRED_FLAGS with no action to back it" % flag)


def test_an_old_subcommand_is_now_an_ordinary_word():
    """summary/detail/dcgm/live used to be rewritten to flags with a note.

    They are gone. The word is no longer special, so it falls through as a would-be
    JOBID and argparse rejects it downstream -- which is what an unrecognised first
    word should do, rather than quietly meaning something.
    """
    for old in ("summary", "detail", "dcgm", "live"):
        argv = resolve_argv([old])
        assert argv[0] in (RUNNING, FINISHED)
        assert old in argv[1:]


# --- level 2/3: validation --------------------------------------------------

def _args(**kw):
    base = dict(mode=RUNNING, jobids=[], jobids_opt=None, days=None, lastn=None,
                starttime=None, endtime=None, min_elapsed=None, partition=None,
                user="alice", all_users=False, account=None, state=None,
                per_gpu=False, ts=False, view=None, dcgm=False, runtime_avg=False,
                instant=False,
                header=True, csv=False, step=None,
                timeout=None, workers=None, config_path=None, explicit_mode=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _request(**kw):
    return build_request(_args(**kw))


def test_finished_defaults_to_one_day():
    assert _request(mode=FINISHED).days == 1


def test_running_needs_no_window():
    request = _request(mode=RUNNING)
    assert request.days is None and request.running


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


def test_the_running_request_averages_by_default():
    """Averaging over each runtime is what a reader almost always wants; the cost is paced
    rather than capped, so nothing count-dependent rides along."""
    assert _request(mode=RUNNING).average is True


def test_runtime_avg_says_the_default_out_loud_and_instant_declines_it():
    assert _request(mode=RUNNING, runtime_avg=True).average is True
    assert _request(mode=RUNNING, instant=True).average is False


def test_the_default_average_reaches_an_explicit_jobid_too():
    """An explicit JOBID can name a job that is still running, and folding its window is
    what the default now does. A JOBID selection carries whatever states the ids have, so
    the fold is settled per record (JobRecord.unfinished), not here."""
    assert _request(jobids=["36978909"]).average is True


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

def test_per_gpu_is_the_only_spelling():
    """--hwdetail was the pre-rename name for it, and is gone."""
    _, subparsers = build_parser()
    assert subparsers.choices[RUNNING].parse_intermixed_args(["--per-gpu"]).per_gpu
    with pytest.raises(SystemExit):
        subparsers.choices[RUNNING].parse_intermixed_args(["--hwdetail"])


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
    [], ["--per-gpu"], ["--ts"], ["--cpu"], ["--gpu"], ["--all-metrics"],
    ["--per-gpu", "--all-metrics"], ["--ts", "--all-metrics"], ["--gpu", "--all-metrics"],
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


# --- dispatch ---------------------------------------------------------------

def test_describe_command(capsys):
    main(["describe"])
    assert "CPU%" in capsys.readouterr().out


def test_describe_dcgm_ext(capsys):
    main(["describe", "--metrics", "--all-metrics"])
    out = capsys.readouterr().out
    assert "DCGM GPU metrics" in out
    assert "%d metrics" % len(dcgm.catalog().all_specs) in out


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
    assert out.startswith("JOBID,USER,EPOCH,TIME,NODE,GPU,") and "80.0" in out


# --- the running branch -----------------------------------------------------

GIB = 1024 ** 3


class _RunningClient:
    """Prometheus stand-in for the squeue branch: one GPU on one job.

    Serves the NVML and cgroup series as well as a DCGM one, because the live path
    reconstructs the utilization summary from them -- without those, a running job has
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
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, timeout: _RunningClient())


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
    """The per-GPU granularity for a running job, off the reconstructed summary."""
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
    assert out.startswith("JOBID,USER,EPOCH,TIME,NODE,GPU,")


def test_running_no_matching_jobs(monkeypatch, capsys):
    monkeypatch.setattr(select_mod, "fetch_jobs", lambda sel, timeout: {})
    main(["running", "-a"])
    assert "No running jobs match" in capsys.readouterr().err


def test_running_jobstats_summary_is_reconstructed(monkeypatch, capsys, gpu_record):
    """A running job selected by ID has no jobstats summary, so it is rebuilt from Prometheus."""
    running = dataclasses.replace(gpu_record, jobid="300", state="RUNNING", stats={})
    _patch_sacct(monkeypatch, {"300": running})
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, timeout: object())
    monkeypatch.setattr(select_mod, "compute_dcgm", lambda *a, **k: {"300": ({}, {})})

    def fake_fill(records, ids, client, timeout=None, workers=1, force=False,
                  average=False):
        records["300"].stats = gpu_record.stats
        return 1

    monkeypatch.setattr(select_mod, "fill_running", fake_fill)
    main(["300"])
    out = capsys.readouterr().out
    assert "70" in out and "50" in out      # the jobstats summary fixture's gpu/gmem


def test_offline_view_warns_when_no_endpoint_can_fill(monkeypatch, capsys, gpu_record):
    running = dataclasses.replace(gpu_record, jobid="300", state="RUNNING", stats={})
    _patch_sacct(monkeypatch, {"300": running})

    def no_endpoint(cfg, timeout):
        raise JobscopeError("no Prometheus endpoint configured")

    monkeypatch.setattr(select_mod, "client_from_config", no_endpoint)
    main(["finished", "--cpu", "-D", "1", "-u", "alice"])
    err = capsys.readouterr().err
    assert "no Prometheus" in err and "blank" in err


def test_a_finished_jobs_summary_is_never_recomputed(monkeypatch, capsys, gpu_record):
    _patch_sacct(monkeypatch, {"100": gpu_record})
    monkeypatch.setattr(select_mod, "client_from_config", lambda cfg, timeout: object())
    monkeypatch.setattr(select_mod, "compute_dcgm", lambda *a, **k: {"100": ({}, {})})

    def boom(*a, **k):
        raise AssertionError("a stored summary must not be refetched")

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

def _options_for(argv, monkeypatch, folded=None):
    """The RenderOptions handle_report builds for `argv`, without running a query.

    `folded` is what the fake resolve() reports: whether every value spans its job's whole
    runtime. None mirrors the real thing -- a squeue selection folds unless --instant said
    otherwise, and a window selection is finished by construction and always folded --
    leaving it worth naming only for explicit JOBIDs, whose records can be either.
    """
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

    def fake_resolve(request, *a, **kw):
        settled = request.average if request.running else True
        return select_mod.Resolved(context=[], chunks=[],
                                   folded=folded if folded is not None else settled)

    monkeypatch.setattr(cli, "resolve", fake_resolve)
    main(argv)
    return captured["options"]


@pytest.mark.parametrize("level_flag", ("--per-node", "--per-gpu"))
def test_the_detail_views_are_told_the_selection_size(monkeypatch, level_flag):
    """They print a job's block before knowing whether another follows, so whether to draw
    per-unit charts or one aggregate summary cannot be counted -- it has to be handed over.
    resolve() knows it eagerly; without this wiring every sweep looks like a single job.
    """
    seen = {}

    class FakeDetail:
        def __init__(self, context, options, out=None, **kw):
            seen.update(kw)

        def add(self, *a, **kw):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(cli, "DetailRenderer", FakeDetail)
    monkeypatch.setattr(cli, "resolve",
                        lambda *a, **kw: select_mod.Resolved(context=[], chunks=[],
                                                             folded=True, total=37))
    main(["-p", "kempner", "-a", level_flag])
    assert seen.get("total") == 37, seen
    assert "specs" in seen, seen        # and the spec list the tallies are built from


def test_finished_jobs_are_weighted_by_resource_time(monkeypatch):
    """A finished job's values span its whole runtime, so runtime is a valid weight."""
    assert _options_for(["finished", "-D", "1"], monkeypatch).time_weighted is True


def test_a_finished_job_id_is_weighted_by_resource_time(monkeypatch):
    assert _options_for(["35244230"], monkeypatch, folded=True).time_weighted is True


def test_a_running_job_named_by_id_is_averaged_by_default(monkeypatch):
    """Averaging over the runtime is the default now, so the two commands that name one
    running job agree. --instant is the way back, and it is not time-weighted: every value
    is the same moment, which no amount of elapsed time makes representative."""
    assert _options_for(["35244230"], monkeypatch).time_weighted is True
    assert _options_for(["35244230", "--instant"], monkeypatch,
                        folded=False).time_weighted is False


@pytest.mark.parametrize("folded", (False, True))
def test_the_weighting_and_the_sampled_line_answer_from_one_fact(monkeypatch, folded):
    """Both say whether the values span whole runtimes, and a table cannot be described
    two ways: Used/GPU-hr beside "most recent scrape" is a contradiction, not a nuance.
    """
    from jobscope import dcgm as dcgm_mod
    from jobscope.report import averaging_note, sampled_pair
    argv = ["35244230"] + ([] if folded else ["--instant"])
    weighted = _options_for(argv, monkeypatch, folded=folded).time_weighted
    sampled = sampled_pair(dcgm_mod.catalog().default_specs, not folded, folded)[0][1]
    instant = "most recent scrape" in sampled
    assert weighted is not instant, sampled
    assert ("most recent scrape" in averaging_note(weighted)) is instant


def test_the_running_view_averages_over_the_runtime_by_default(monkeypatch):
    """The default flipped: one scrape of a bursty job is a coin toss across its whole
    range, and a reader who did not ask for that should not silently get it."""
    assert _options_for(["running"], monkeypatch).time_weighted is True


def test_instant_declines_the_average(monkeypatch):
    """The fast path stays reachable: one query per metric however many jobs."""
    assert _options_for(["running", "--instant"], monkeypatch).time_weighted is False


def test_runtime_avg_and_instant_are_mutually_exclusive(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _options_for(["running", "--runtime-avg", "--instant"], monkeypatch)
    assert "not allowed with" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ("--runtime-avg", "--instant"))
def test_neither_span_flag_applies_to_a_window_selection(monkeypatch, flag):
    """A window holds only finished jobs, whose values already span their runtimes --
    there is no newer scrape to read and nothing left to average."""
    with pytest.raises(JobscopeError) as exc:
        _request(mode=FINISHED, **{flag.lstrip("-").replace("-", "_"): True})
    assert flag in str(exc.value)


def test_the_efficiency_bars_are_on_by_default(monkeypatch):
    assert _options_for(["finished", "-D", "1"], monkeypatch).plot_avgeff


def test_no_plot_switches_the_bars_off(monkeypatch):
    assert not _options_for(["finished", "-D", "1", "--no-plot"], monkeypatch).plot_avgeff


def test_the_old_plot_flag_is_gone():
    """--plot_avgeff only ever printed "that is the default now"."""
    _, subparsers = build_parser()
    for spelling in ("--plot_avgeff", "--plot-avgeff"):
        with pytest.raises(SystemExit):
            subparsers.choices[RUNNING].parse_intermixed_args([spelling])


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
    """slurm.states_for does the validating, so the CLI passes the string through."""
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


def test_nodename_and_gpuid_reach_resolve_on_the_summary(monkeypatch):
    """They used to be rejected here: the per-job table has no row to filter.

    It has numbers to narrow instead -- select.py restricts the stats the summary is
    computed from -- so the flags apply to every view now, and the header says so.
    """
    seen = {}

    def fake_resolve(request, cfg, timeout, workers, specs, nodename=None, gpu_ids=(),
                     host_specs=None):
        seen["nodename"], seen["gpu_ids"] = nodename, gpu_ids
        return None

    monkeypatch.setattr(cli, "resolve", fake_resolve)
    main(["-j", "1", "--nodename=n1", "--gpuid", "0,1"])
    assert seen == {"nodename": "n1", "gpu_ids": ("0", "1")}


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
    # --runtime-avg is NOT hidden here, though it used to be: an explicit JOBID can name
    # that is still running, and folding its window is exactly what --runtime-avg is for. The
    # rule is "hidden when this command would reject it", and this one no longer does.
    assert "--runtime-avg" not in hidden
    # What remains is what this command actually honours.
    for flag in ("--nodename", "--all-metrics", "--csv", "--no-plot", "--per-gpu",
                 "--runtime-avg"):
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
    assert {"--runtime-avg", "--min-elapsed"} <= set(hidden)
    assert "--days" not in hidden


def test_ts_hides_what_the_series_drops(capsys):
    """emit_timeseries notes these as dropped; the help should not offer them."""
    body, hidden = _help_for(["--ts"], capsys)
    assert {"--cpu", "--gpu", "--per-gpu", "--no-plot"} <= set(hidden)
    assert "--step" in body      # --ts is the only thing that reads it
    assert "--nodename" in body  # the series carries a NODE column, so it filters


def test_cpu_hides_the_gpu_only_columns(capsys):
    _, hidden = _help_for(["--cpu"], capsys)
    assert "--all-metrics" in set(hidden)


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
    for flag in ("--days", "--partition", "--runtime-avg", "--step", "--ts"):
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
    assert "{running,finished,plot,describe,config,probe}" in body
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

_TS_HEAD = "JOBID,USER,EPOCH,TIME,NODE,GPU,GPU%,GMEM%\n"


def _ts_rows(jobids=("100",), nodes=("node01",), gpus=("0", "1", "2", "3")):
    return "".join(
        "%s,alice,%d,2020-01-01T00:%02d:00,%s,%s,%d,%d\n"
        % (j, 1000 + 60 * t, t, n, g, 90 + t, 50 + t)
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
    assert "JOBID,USER" not in out       # the CSV went to the chart, not to stdout
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


def test_the_overlay_needs_no_nodename_because_each_node_is_a_row(monkeypatch, capsys):
    """The guard --plot_ts raises is the one --plot_ts_overlay exists to lift."""
    monkeypatch.setenv("COLUMNS", "200")
    _fake_ts(monkeypatch, _ts_rows(nodes=("node01", "node02")))
    main(["-j", "1", "--plot_ts_overlay"])
    out = capsys.readouterr().out
    assert "┤" in out
    assert [ln.strip() for ln in out.splitlines()].count("node01") == 1
    assert [ln.strip() for ln in out.splitlines()].count("node02") == 1
    # The header cannot name one node when the rows below it name two.
    assert "2 nodes" in out


def test_the_overlay_still_refuses_several_jobs(monkeypatch, capsys):
    """Lifting the node guard does not lift the job one: render_line keys its series on
    (NODE, GPU) alone, so two jobs sharing a card would concatenate into one line."""
    rows = _ts_rows(nodes=("node01",))
    _fake_ts(monkeypatch, rows.replace("100,alice", "101,alice", 1))
    with pytest.raises(SystemExit):
        main(["-j", "1", "--plot_ts_overlay"])
    assert "charts one job" in capsys.readouterr().err


def test_the_overlay_folds_into_plot_ts_so_its_guards_apply(monkeypatch, capsys):
    """handle_report sets args.plot_ts from it, which is what makes every --ts path --
    the schema, the window, --csv -- apply unchanged. The message must still name the
    flag that was actually typed."""
    _fake_ts(monkeypatch, _ts_rows(nodes=("node01",)))
    with pytest.raises(SystemExit):
        main(["-j", "1", "--plot_ts_overlay", "--csv"])
    assert "--plot_ts_overlay draws a chart" in capsys.readouterr().err


def test_plot_ts_charts_every_metric_by_default(monkeypatch, capsys):
    """Bare --plot_ts resolves to the combined/extended view, same as --all-metrics did
    before -- so the chart shows the extended catalog either way.

    It can, because each metric gets its own panel and its own axis: nothing here has
    to share a scale with anything else, which is what would make a wide set unreadable.
    """
    header = "JOBID,USER,EPOCH,TIME,NODE,GPU,GPU%,GMEM%,ENGINE%\n"
    body = "".join(
        "100,alice,%d,2020-01-01T00:%02d:00,node01,%s,%d,%d,%d\n"
        % (1000 + 60 * t, t, g, 90 + t, 50 + t, 30 + t)
        for g in ("0", "1", "2", "3") for t in range(3))

    def emit(*a, **kw):
        (kw.get("out") or sys.stdout).write(header + body)
    monkeypatch.setattr(cli, "emit_timeseries", emit)

    main(["-j", "1", "--plot_ts"])
    assert "ENGINE%" in capsys.readouterr().out

    main(["-j", "1", "--all-metrics", "--plot_ts"])
    assert "ENGINE%" in capsys.readouterr().out


def test_cpu_ts_no_longer_says_it_does_not_apply(monkeypatch, capsys):
    """--cpu now switches --ts to the CPU/MEM series instead of being dropped."""
    _fake_ts(monkeypatch, "")
    main(["-j", "1", "--cpu", "--ts"])
    assert "does not apply" not in capsys.readouterr().err


def test_cpu_and_dcgm_together_no_longer_conflict(monkeypatch, capsys):
    """--cpu --all-metrics together now means the combined view, not a dropped flag."""
    _fake_ts(monkeypatch, "")
    main(["-j", "1", "--cpu", "--all-metrics", "--ts"])
    assert "does not apply" not in capsys.readouterr().err


@pytest.mark.parametrize("flags,expect_combined,expect_specs_name", [
    ([], True, "key"),                    # bare --ts: combined + curated key metrics
    (["--cpu"], False, "default"),        # --cpu alone: cpu-only (specs unused, but this
                                          # is what the outer `specs` var resolves to)
    (["--all-metrics"], False, "all"),           # --all-metrics alone: gpu-only/extended, unchanged
    (["--cpu", "--all-metrics"], True, "all"),   # both: combined + extended
])
def test_ts_view_resolution_truth_table(monkeypatch, flags, expect_combined, expect_specs_name):
    """The one genuinely new piece of branching logic in this feature: which of
    cpu-only/gpu-only/combined --ts resolves to, and whether the GPU catalog is
    dcgm.catalog().key_specs (curated default) or dcgm.catalog().all_specs (--all-metrics), for every
    (--cpu, --all-metrics) combination."""
    captured = {}

    def emit(request, cfg, timeout, workers, specs, step, options, out=None):
        captured["specs"] = specs
        captured["combined"] = options.combined

    monkeypatch.setattr(cli, "emit_timeseries", emit)
    main(["-j", "1", "--ts"] + flags)
    assert captured["combined"] == expect_combined
    active = dcgm.catalog()
    expected = {"key": active.key_specs, "all": active.all_specs,
                "default": active.default_specs}[expect_specs_name]
    assert captured["specs"] == list(expected)


# --- which of the two band tables each view is graded by ---------------------

def _two_table_config():
    """A config whose two views disagree, so which one is in force is visible."""
    from jobscope import config as config_module
    return dataclasses.replace(
        config_module.get_config(),
        thresholds=config_module.Thresholds(by_metric={"CPU%": {"wasteful": 5.0}}),
        timeslice_thresholds=config_module.Thresholds(by_metric={"CPU%": {"wasteful": 8.0}}))


@pytest.mark.parametrize("argv,expected", [
    (["-j", "1"], 5.0),                       # the plain summary: whole elapsed time
    (["-j", "1", "--ts"], 8.0),               # a time slice
    (["-j", "1", "--ts", "30m"], 8.0),        # ... with a window
    (["-j", "1", "--plot_ts"], 8.0),          # --plot_ts *is* --ts
    (["-j", "1", "--ts", "--eff"], 8.0),
    (["-j", "1", "--ts", "--stats"], 8.0),
])
def test_each_view_is_graded_by_its_own_table(monkeypatch, argv, expected):
    """The crux of the two-table feature: the summary reads [thresholds.summary]
    and every --ts path reads [thresholds.timeslice]. Nothing else distinguishes
    them, so getting this wrong grades a job by the other view's numbers."""
    from jobscope import config as config_module
    config_module.set_config(_two_table_config())
    captured = {}

    def emit(request, cfg, timeout, workers, specs, step, options, out=None):
        captured["options"] = options

    class FakeRenderer:
        def __init__(self, context, options, **kw):
            captured["options"] = options

        def add(self, *a, **kw):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(cli, "emit_timeseries", emit)
    monkeypatch.setattr(cli, "SummaryRenderer", FakeRenderer)
    monkeypatch.setattr(cli, "resolve",
                        lambda *a, **kw: select_mod.Resolved(context=[], chunks=[]))
    # The options are captured in emit(); what runs after it consumes a CSV the
    # fake never wrote, so those stages are stubbed out rather than fed one.
    for stage in ("_plot_timeseries", "_eff_timeseries", "_stats_timeseries"):
        monkeypatch.setattr(cli, stage, lambda *a, **kw: None)
    main(argv)
    assert captured["options"].thresholds.edge("wasteful", "CPU%") == expected


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
    assert "JOBID,USER" not in out               # the CSV became the summary
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


@pytest.mark.parametrize("argv,level", [
    (["--stats"], "gpu"),          # bare keeps the per-GPU meaning it always had
    (["--stats", "node"], "node"),
    (["--stats", "job"], "job"),
])
def test_the_stats_level_is_the_flags_own_value(argv, level):
    """One flag for one axis: this was --stats/--stats-per-node/--stats-per-job."""
    _, subparsers = build_parser()
    assert subparsers.choices[RUNNING].parse_intermixed_args(argv).stats == level


def test_an_unknown_stats_level_is_rejected():
    """`choices` is also what keeps the optional value from eating a JOBID:
    `--stats 12345` is an invalid choice, not a misread selection."""
    _, subparsers = build_parser()
    for bad in (["--stats", "socket"], ["--stats", "12345"]):
        with pytest.raises(SystemExit):
            subparsers.choices[RUNNING].parse_intermixed_args(bad)


@pytest.mark.parametrize("argv,lead", [
    (["--stats"], "NODE:GPU"), (["--stats", "node"], "NODE"),
    (["--stats", "job"], "JOBID"),
])
def test_each_level_renders_through_the_ts_path(argv, lead, monkeypatch, capsys):
    _fake_ts(monkeypatch, _ts_rows(nodes=("node01", "node02"), gpus=("0", "1")))
    main(["-j", "1", "--ts"] + argv)
    assert capsys.readouterr().out.splitlines()[0].split()[0] == lead


def test_a_stats_level_still_needs_a_timeseries(capsys):
    with pytest.raises(SystemExit):
        main(["-j", "1", "--stats", "node"])
    assert "add --ts" in capsys.readouterr().err


def test_eff_groups_the_jobs(monkeypatch, capsys):
    _fake_ts(monkeypatch, _ts_rows(jobids=("100", "101"), nodes=("node01",), gpus=("0",)))
    main(["-p", "kempner", "-a", "--ts", "10m", "--eff"])
    out = capsys.readouterr().out
    assert "by best of" in out and "jobs" in out
    assert "JOBID,USER,EPOCH" not in out      # the CSV became the report


def test_eff_defaults_to_the_job_as_the_unit(monkeypatch, capsys):
    """"Which jobs are idle" is asked about jobs, so that is the level."""
    _fake_ts(monkeypatch, _ts_rows(nodes=("node01", "node02"), gpus=("0", "1")))
    main(["-j", "1", "--ts", "--eff"])
    assert "1 jobs" in capsys.readouterr().out       # not 2 nodes or 4 GPUs


def test_eff_can_group_nodes_instead(monkeypatch, capsys):
    _fake_ts(monkeypatch, _ts_rows(nodes=("node01", "node02"), gpus=("0", "1")))
    main(["-j", "1", "--ts", "--eff", "--stats", "node"])
    assert "2 nodes" in capsys.readouterr().out


def test_eff_needs_a_timeseries(capsys):
    with pytest.raises(SystemExit):
        main(["-j", "1", "--eff"])
    assert "add --ts" in capsys.readouterr().err


def test_the_good_jobs_are_listed_by_the_flags_own_value(capsys, monkeypatch):
    """--all-categories only ever qualified --eff, so it is its value now."""
    _fake_ts(monkeypatch, _ts_rows(nodes=("node01",), gpus=("0",)))
    main(["-j", "1", "--ts", "--eff", "all"])
    assert "by best of" in capsys.readouterr().out


def test_an_unknown_eff_value_is_rejected(capsys):
    with pytest.raises(SystemExit):
        main(["-j", "1", "--ts", "--eff", "everything"])
    assert "takes no value, or 'all'" in capsys.readouterr().err


@pytest.mark.parametrize("argv,hidden", [
    (["running"], {"--stats", "--eff"}),                      # both raise without --ts
    # (a bare [] would reach the top-level parser, which does not narrow)
    (["--ts"], {"--csv"}),          # a plain --ts already writes CSV; --csv adds nothing
    (["--plot-ts"], {"--stats", "--eff"}),                    # both silently ignored
])
def test_the_narrowed_help_hides_the_summarizers_that_would_not_run(argv, hidden, capsys):
    """The feature's whole claim is that it hides what would not have worked."""
    _, shown = _help_for(argv, capsys)
    assert hidden <= set(shown), (hidden - set(shown), shown)


def test_csv_is_hidden_under_a_plain_ts_but_not_under_stats(capsys):
    """A plain --ts already writes CSV, so --csv adds nothing there and the narrowed
    help says so. With --stats it is the flag that turns the summary table into CSV, so
    hiding it would hide something that works -- which a blanket rule would have done."""
    _, plain = _help_for(["--ts"], capsys)
    assert "--csv" in set(plain)
    _, with_stats = _help_for(["--ts", "--stats"], capsys)
    assert "--csv" not in set(with_stats)


def test_stats_csv_really_does_emit_csv(monkeypatch, capsys):
    """The measurement the guard above rests on: if this ever stops being true, the
    guard is wrong rather than merely redundant."""
    _fake_ts(monkeypatch, _ts_rows(nodes=("node01",), gpus=("0",)))
    main(["-j", "1", "--ts", "--stats", "--csv"])
    assert capsys.readouterr().out.splitlines()[0].startswith("NODE:GPU,METRIC,")


def test_eff_with_plot_ts_says_it_is_ignored(monkeypatch, capsys):
    _fake_ts(monkeypatch, _ts_rows(gpus=("0",)))
    main(["-j", "1", "--plot_ts", "--eff"])
    assert "ignoring --eff" in capsys.readouterr().err


# --- [defaults] days / state reach the Request -------------------------------

def test_the_default_window_and_state_come_from_config(monkeypatch):
    """Both already had flags but no config key -- and the state default hides
    failures, which is worth setting once per site rather than typing every run."""
    from jobscope import config as config_module
    config_module.set_config(dataclasses.replace(
        config_module.get_config(),
        defaults=config_module.Defaults(workers=8, timeout=60.0,
                                        days=7, state="all")))
    request = _request_for(["finished"], monkeypatch)
    assert request.days == 7 and request.state == "all"


def test_an_explicit_flag_still_beats_the_config(monkeypatch):
    from jobscope import config as config_module
    config_module.set_config(dataclasses.replace(
        config_module.get_config(),
        defaults=config_module.Defaults(workers=8, timeout=60.0,
                                        days=7, state="all")))
    request = _request_for(["finished", "-D", "2", "-t", "failed"], monkeypatch)
    assert request.days == 2 and request.state == "failed"


# --- [metrics] reach each view's spec list -----------------------------------

def test_each_view_gets_its_configured_metric_list(monkeypatch):
    """The summary, the time series and --all-metrics each read their own [metrics] key."""
    from jobscope import config as config_module
    from jobscope.dcgm import specs_named
    config_module.set_config(dataclasses.replace(
        config_module.get_config(),
        metrics=config_module.Metrics(
            summary=tuple(specs_named(["gpu", "sm_act", "mem", "memtot"])),
            timeseries=tuple(specs_named(["gpu", "occ"])),
            extended=tuple(specs_named(["gpu", "temp", "mem", "memtot"])))))
    captured = {}

    def emit(request, cfg, timeout, workers, specs, step, options, out=None):
        captured["ts"] = [s.header for s in specs]

    class FakeRenderer:
        def __init__(self, context, options, **kw):
            captured["summary"] = [s.header for s in kw.get("specs") or []]

        def add(self, *a, **kw):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(cli, "emit_timeseries", emit)
    monkeypatch.setattr(cli, "SummaryRenderer", FakeRenderer)
    monkeypatch.setattr(cli, "resolve",
                        lambda *a, **kw: select_mod.Resolved(context=[], chunks=[]))

    main(["-j", "1", "--ts"])
    assert captured["ts"] == ["GPU%", "OCC%"]
    main(["-j", "1"])
    assert "SM_ACT%" in captured["summary"] and "TENSOR%" not in captured["summary"]
    main(["-j", "1", "--all-metrics"])
    assert "TEMP_C" in captured["summary"] and "SM_ACT%" not in captured["summary"]


def test_a_comma_in_a_jobid_is_caught_and_points_at_gpuid(capsys):
    """`--gpu 0,1` is what people type, expecting `jobscope plot`'s GPU filter.

    --gpu takes no value here -- it picks the GPU *columns* -- so the list used to
    fall through to the JOBID positional: the run warned about a job named "0,1",
    charted every GPU, and exited 0. A silently wrong chart is worse than no chart.
    """
    with pytest.raises(SystemExit):
        main(["-j", "123", "--plot_ts", "--gpu", "0,1"])
    err = capsys.readouterr().err
    assert "not a job ID" in err and "--gpuid 0,1" in err

    # Without --gpu there is no GPU to suggest, but a comma is still not a job ID.
    with pytest.raises(SystemExit):
        main(["123,456"])
    err = capsys.readouterr().err
    assert "not a job ID" in err and "--gpuid" not in err





def test_config_shows_the_endpoint_redacted(monkeypatch, capsys):
    """It printed every band table but never the one setting that must be right
    first. The token is the reason it has to go through redact_url."""
    secret = "glc_configleaktoken"
    monkeypatch.setenv("JOBSCOPE_PROM_URL",
                       "https://1180804:%s@prom.grafana.net/api/prom" % secret)
    main(["config"])
    out = capsys.readouterr().out
    assert secret not in out
    assert "***@prom.grafana.net/api/prom" in out
    assert "$JOBSCOPE_PROM_URL" in out       # and where it came from
    assert "scrape" in out


def test_force_color_is_the_mirror_of_no_color(monkeypatch):
    """scripts/make_screenshots.sh exports FORCE_COLOR to capture coloured tables for
    the docs; before this it was ignored and every screenshot came out grey.

    Neither variable may override an explicit flag: escapes in a CSV are corruption
    whoever asked for them.
    """
    args = argparse.Namespace(csv=False, no_color=False)

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert cli._want_color(args) is False           # not a tty under pytest

    monkeypatch.setenv("FORCE_COLOR", "1")
    assert cli._want_color(args) is True

    monkeypatch.setenv("NO_COLOR", "1")
    assert cli._want_color(args) is False           # NO_COLOR wins over FORCE_COLOR

    monkeypatch.delenv("NO_COLOR")
    assert cli._want_color(argparse.Namespace(csv=True, no_color=False)) is False
    assert cli._want_color(argparse.Namespace(csv=False, no_color=True)) is False


# --- what the startup path is allowed to import -------------------------------------

def test_the_startup_path_does_not_import_the_network_stack():
    """`requests` cost 257ms of a 442ms import, and importlib.metadata another 44 -- both
    on every invocation, including --help, --version, config, describe and every usage
    error. Deferred into the paths that need them (prometheus._make_session and cli's
    --version action), which took `jobscope --help` from 0.47s to 0.20s.

    A subprocess, because this process has already imported everything the test suite
    touches; asserting on sys.modules here would measure pytest, not jobscope.
    """
    import subprocess
    import sys

    code = ("import sys; import jobscope.cli; "
            "print(' '.join(sorted(m for m in ('requests', 'urllib3', "
            "'importlib.metadata') if m in sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "", "startup imports %s -- keep these off the import path" % out


def test_version_still_prints_and_pulls_the_metadata_then():
    """The flag has to keep working: deferring the lookup must not turn --version into a
    lazy attribute nobody resolves."""
    import subprocess
    import sys

    # main() rather than -m: cli.py is reached through the console script and has no
    # __main__ block, so `-m jobscope.cli` prints nothing and would pass vacuously.
    out = subprocess.run(
        [sys.executable, "-c", "from jobscope.cli import main; main(['--version'])"],
        capture_output=True, text=True)
    printed = (out.stdout + out.stderr).strip()
    assert printed.startswith("jobscope "), printed
    assert printed.split()[1][0].isdigit(), printed


# --- probe: a target written after the flag instead of beside it ---------------
#
# Every probe check takes an optional value, so argparse stops consuming one the moment
# another flag intervenes: `--coverage --full kempner` left `kempner` unrecognised while
# `--coverage kempner --full` worked. The two orderings now mean the same thing.

def _probe(argv):
    from jobscope.cli import _reclaim_probe_target
    _, subparsers = build_parser()
    args = subparsers.choices["probe"].parse_intermixed_args(argv)
    _reclaim_probe_target(args)
    return args


def test_a_probe_target_beside_its_flag_is_unchanged():
    args = _probe(["--coverage", "kempner", "--full"])
    assert args.coverage == "kempner" and args.full and args.target is None


def test_a_probe_target_after_another_flag_is_handed_back():
    """The regression: this used to be 'unrecognized arguments: kempner'."""
    args = _probe(["--coverage", "--full", "kempner"])
    assert args.coverage == "kempner" and args.full and args.target is None


def test_the_two_orderings_agree():
    beside = _probe(["--coverage", "kempner", "--full"])
    after = _probe(["--coverage", "--full", "kempner"])
    assert (beside.coverage, beside.full) == (after.coverage, after.full)


def test_a_trailing_target_reaches_the_other_checks_too():
    """Not special-cased to --coverage; the same argparse behaviour affects all four."""
    assert _probe(["--metrics", "--full", "123"]).metrics == "123"
    assert _probe(["--toml", "--full", "123"]).toml == "123"


def test_a_target_with_no_check_names_the_checks_it_could_have_meant():
    with pytest.raises(JobscopeError) as exc:
        _probe(["kempner"])
    assert "names nothing on its own" in str(exc.value)
    assert "--coverage kempner" in str(exc.value)


def test_two_bare_checks_refuse_to_guess():
    """Both readings are equally good, so name them rather than pick one."""
    with pytest.raises(JobscopeError) as exc:
        _probe(["--metrics", "--coverage", "--full", "kempner"])
    assert "could belong to" in str(exc.value)
    assert "--coverage" in str(exc.value) and "--metrics" in str(exc.value)


def test_a_bare_check_with_no_target_is_left_alone():
    args = _probe(["--coverage", "--full"])
    assert args.coverage == "" and args.target is None


def test_handle_probe_actually_reclaims_the_target(monkeypatch):
    """The wiring, not just the helper. Without this, deleting the call from
    handle_probe leaves every test above passing while the feature is gone."""
    from jobscope import probe as probe_mod
    seen = {}

    def fake_run(out, cfg, config_path, timeout, **kw):
        seen.update(kw)
        return 0

    monkeypatch.setattr(probe_mod, "run", fake_run)
    main(["probe", "--coverage", "--full", "kempner"])
    assert seen["coverage"] == "kempner"
    assert seen["full"] is True


# --- the granularity axis: --per-job / --per-node / --per-gpu ------------------

def test_the_three_levels_are_mutually_exclusive():
    _, subparsers = build_parser()
    for pair in (["--per-job", "--per-node"], ["--per-node", "--per-gpu"],
                 ["--per-job", "--per-gpu"]):
        with pytest.raises(SystemExit):
            subparsers.choices[RUNNING].parse_intermixed_args(pair)


def test_per_job_is_the_default_and_nameable():
    """A lone --per-gpu gave no hint a scale existed. Naming the default costs one flag
    and lets a script say which level it means."""
    _, subparsers = build_parser()
    bare = subparsers.choices[RUNNING].parse_intermixed_args([])
    named = subparsers.choices[RUNNING].parse_intermixed_args(["--per-job"])
    assert not bare.per_node and not bare.per_gpu
    assert named.per_job and not named.per_node and not named.per_gpu


def test_per_node_narrows_the_help_like_per_gpu_does():
    """It is a detail view, so the series flags do not apply to it either."""
    from jobscope.cli import _inert_dests
    _, subparsers = build_parser()
    args = subparsers.choices[RUNNING].parse_intermixed_args(["--per-node"])
    args.explicit_mode = False
    inert = _inert_dests(args)
    assert "ts" in inert and "plot_ts" in inert
    # and it keeps --nodename, which does apply to a per-node row
    assert "nodename" not in inert


# --- --verify -------------------------------------------------------------------

def test_verify_requires_named_jobs(capsys):
    """It fetches every scrape of each job's series, which is right for the job you are
    about to act on and wrong for a partition. Named, not silently capped: a truncated
    answer is the one thing this view must not give."""
    with pytest.raises(SystemExit):
        main(["finished", "-D", "1", "--verify"])
    err = capsys.readouterr().err
    assert "--verify checks the jobs you name" in err
    assert "-j JOBID" in err


def test_verify_takes_a_window_through_the_same_machinery_as_ts(capsys):
    """One window grammar, not a second copy of it -- and going through _ts_value is what
    gives --verify the job-ID rescue every other window-carrying flag has. A bare number
    after the flag is a job ID, because a window always carries a unit."""
    from jobscope.cli import _reclaim_jobid_after_ts, _ts_window
    _, subparsers = build_parser()

    def parsed(argv):
        args = subparsers.choices[RUNNING].parse_intermixed_args(argv)
        _reclaim_jobid_after_ts(args)
        return args

    assert _ts_window(parsed(["--verify"])) is None
    assert _ts_window(parsed(["--verify", "2h"])) == 7200
    # The regression: this used to raise "--verify takes a duration with a unit".
    reclaimed = parsed(["--verify", "30012345"])
    assert reclaimed.jobids == ["30012345"] and reclaimed.verify is True
    assert "as a job ID" in capsys.readouterr().err


def test_verify_is_exclusive_with_the_other_granularities():
    _, subparsers = build_parser()
    for other in ("--ts", "--per-gpu", "--per-node", "--plot-ts"):
        with pytest.raises(SystemExit):
            subparsers.choices[RUNNING].parse_intermixed_args(["--verify", other])


def test_verify_grades_against_the_timeslice_bands():
    """A windowed view answers to [thresholds.timeslice], like every other one: a
    half-hour rung that catches a checkpoint pause should not answer to a
    nineteen-hour job's bar."""
    _, subparsers = build_parser()
    args = subparsers.choices[RUNNING].parse_intermixed_args(["--verify"])
    assert args.verify is True
    # The resolution itself lives in handle_report; this pins the flag that drives it.
    assert bool(args.verify) and not args.ts


# --- the help tree ------------------------------------------------------------------

def _help_text(argv, capsys):
    with pytest.raises(SystemExit):
        main(argv)
    return capsys.readouterr().out


def _flat(text):
    """Help text with its wrapping undone -- argparse breaks a help string wherever the
    column runs out, so a phrase to match for may straddle a newline."""
    return " ".join(text.split())


def test_h_is_the_summary_and_help_all_is_the_detail(capsys):
    """117 lines is not a thing anyone reads, and the detail was worth keeping -- so it
    moved to --help-all, which already existed as the everything view."""
    brief = _help_text(["running", "-h"], capsys)
    full = _help_text(["running", "--help-all"], capsys)
    assert len(brief.splitlines()) < len(full.splitlines())
    # --verify is the worked example: its summary is one line, its detail five.
    assert "check one job before acting on it" in _flat(brief)
    assert "longest unbroken idle stretch" not in _flat(brief)
    assert "longest unbroken idle stretch" in _flat(full)


def test_no_flag_is_visible_in_h_but_missing_from_help_all(capsys):
    """Shortening must not hide a flag, only its detail. --help-all is the full list."""
    brief = _help_text(["running", "-h"], capsys)
    full = _help_text(["running", "--help-all"], capsys)
    for flag in re.findall(r"^  (--[a-z-]+|-[a-zA-Z])", brief, re.M):
        assert flag in full, flag


@pytest.mark.parametrize("argv", (["--help"], ["running", "-h"], ["running", "--help-all"],
                                 ["-j", "1", "--per-gpu", "-h"], ["probe", "-h"]))
def test_the_split_marker_never_reaches_a_terminal(argv, capsys):
    """The failure mode a sentinel invites. Both brief_help and full_help remove it; a
    path that printed help without going through either would leak a NUL byte."""
    assert cli._HELP_SPLIT not in _help_text(argv, capsys)


def test_every_reporting_flag_belongs_to_an_axis():
    """A flag added later must not land in bare `options:`, which is what happened to ten
    of them while the report parser was shared through parents=: argparse recreates a
    parent's mutually exclusive groups on the child *parser*, losing their group.
    """
    _, subparsers = build_parser()
    sub = subparsers.choices[RUNNING]
    axes = {g.title: g for g in sub._action_groups if g.title in cli.AXES}
    assert set(axes) == set(cli.AXES), sorted(axes)
    placed = {id(a) for g in axes.values() for a in g._group_actions}
    # Everything except argparse's own -h and the two meta flags from `base`.
    meta = {"help", "help_all", "config_path", cli._RETIRED_DEST}
    for action in sub._actions:
        if action.dest in meta or action.help is argparse.SUPPRESS:
            continue
        assert id(action) in placed, action.dest


def test_every_axis_says_what_question_it_answers():
    """A title alone leaves the reader to infer the axis from the flags under it."""
    _, subparsers = build_parser()
    for group in subparsers.choices[RUNNING]._action_groups:
        if group.title in cli.AXES:
            assert group.description, group.title


def test_the_advertised_axes_are_the_group_titles(capsys):
    """The summary line named different things from the headings under it -- five things,
    four names, no correspondence."""
    _, subparsers = build_parser()
    titles = [g.title for g in subparsers.choices[RUNNING]._action_groups
              if g.title in cli.AXES]
    assert titles == list(cli.AXES)
    body = _help_text(["running", "-h"], capsys)
    for axis in cli.AXES:
        assert "\n%s:\n" % axis in body, axis
        assert "[%s]" % axis in body, axis        # and in the usage line


def test_the_front_door_says_where_the_reporting_flags_are(capsys):
    """It listed six subcommands and seven examples using -j, -p and --plot_ts, and showed
    none of those flags."""
    body = _help_text(["--help"], capsys)
    assert "jobscope probe -h" in body
    assert "jobscope finished -h" in body
    assert "narrowed to what that command can use" in body


def test_narrowing_still_works_on_top_of_shortening(capsys):
    """Two independent passes over the same action list; the footer is unchanged."""
    body = _help_text(["-j", "1", "--per-gpu", "-h"], capsys)
    assert "hiding" in body and "--help-all for the full list" in body
    assert "--days" not in body.split("hiding")[0]
