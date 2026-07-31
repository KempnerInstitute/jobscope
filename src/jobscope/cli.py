"""Command-line interface.

The argument tree has one axis per level, so that every option composes with every
selection::

    jobscope [MODE] [scope] [filters] [granularity] [columns] [--diagnose] [output]

``MODE`` is the first positional and answers *which jobs*: ``running`` (the
default), ``finished``, or one or more explicit ``JOBID``s. The granularity and
column flags answer *how to show them* and are shared by all three, which the
earlier per-view subcommands could not do -- ``live`` had no ``--cpu``, the
historical views had no ``--min-elapsed``, and so on.

``plot``, ``describe`` and ``config`` are utilities and take the first slot too.
The old ``summary``/``detail``/``dcgm``/``live`` subcommands survive as deprecated
aliases; see :data:`DEPRECATED`.
"""

import argparse
import os
import sys
from typing import Optional

from . import __version__, config, plot
from .dcgm import ALL_SPECS, DEFAULT_SPECS
from .errors import JobscopeError
from .live import parse_duration
from .report import (
    DetailRenderer,
    RenderOptions,
    SummaryRenderer,
    describe,
    describe_dcgm,
)
from .sacct import default_user, in_group
from .select import FINISHED, JOBIDS, RUNNING, Request, emit_timeseries, resolve

MODES = (RUNNING, FINISHED)
UTILITIES = ("plot", "describe", "config")

# Flags that select a past window; their presence means sacct rather than squeue.
_WINDOW_FLAGS = ("-D", "--days", "-N", "--lastn", "-S", "--starttime",
                 "-E", "--endtime", "-t", "--state")

# Old subcommand -> (extra argv, how to spell it now). Kept because
# `dcgm --ts --csv | jobscope plot` appears in the README and the docs, and
# contrib/jobscope_live.py shells out to `live`.
DEPRECATED = {
    "summary": ([], "the default"),
    "detail": (["--hwdetail"], "--hwdetail"),
    "dcgm": (["--dcgm"], "--dcgm"),
    "live": ([], "running"),
}

_DESC = (
    "Slurm job efficiency and GPU utilization reporting.\n"
    "\n"
    "  jobscope [running|finished|JOBID...] [filters] [granularity] [columns]\n"
    "\n"
    "One row per job by default, for the jobs running right now.")

_EPILOG = (
    "examples:\n"
    "  jobscope                          your running jobs\n"
    "  jobscope -p kempner -a            everyone on a partition, now\n"
    "  jobscope finished -D 3            your last 3 days\n"
    "  jobscope 30012345                 one job, running or finished\n"
    "  jobscope running --hwdetail       per-GPU rows\n"
    "  jobscope finished -D 7 --dcgm     the full metric catalog\n"
    "  jobscope 30012345 --ts | jobscope plot\n"
    "\n"
    "Flags and JOBIDs may be given in any order.")


def build_parser():
    """Construct the argument parser; return ``(parser, subparsers_action)``."""
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("-c", "--config", dest="config_path", metavar="PATH",
                      help="path to a jobscope config file (overrides $JOBSCOPE_CONFIG)")

    report = argparse.ArgumentParser(add_help=False)
    scope = report.add_argument_group("job selection")
    scope.add_argument("jobids", nargs="*", metavar="JOBID",
                       help="specific job IDs, running or finished")
    scope.add_argument("-j", "--jobid", action="append", dest="jobids_opt", metavar="JOBID",
                       help="a job ID (repeatable; alternative to the positional JOBID)")
    scope.add_argument("-D", "--days", type=int, metavar="N",
                       help="finished: jobs in the last N days (default: 1)")
    scope.add_argument("-N", "--lastn", type=int, metavar="N",
                       help="finished: the most recent N jobs")
    scope.add_argument("-S", "--starttime", metavar="TIME",
                       help="finished: window start, e.g. 2026-07-15 or "
                            "2026-07-15T09:00:00; without -E, that day alone")
    scope.add_argument("-E", "--endtime", metavar="TIME",
                       help="finished: window end, same format as -S")
    scope.add_argument("--min-elapsed", "--min-runtime", dest="min_elapsed",
                       metavar="DURATION", default=None,
                       help="running: only jobs running longer than this (default from "
                            "config: %s; e.g. '5m', '2h', '0s' for no floor)"
                            % config.DEFAULT_MIN_ELAPSED)

    filters = report.add_argument_group("filters")
    filters.add_argument("-p", "--partition", help="narrow to this partition")
    filters.add_argument("-u", "--user", help="user (default: current user, $USER)")
    filters.add_argument("-a", "--all-users", dest="all_users", action="store_true",
                         help="every user's jobs, not just your own")
    filters.add_argument("-A", "--account", help="narrow to this account")
    filters.add_argument("-t", "--state", choices=["all", "completed", "failed"],
                         default="all", help="finished: job state (default: all)")

    shape = report.add_argument_group("granularity and columns")
    grain = shape.add_mutually_exclusive_group()
    grain.add_argument("--hwdetail", action="store_true",
                       help="one row per GPU, with node name and GPU number")
    grain.add_argument("--ts", "--timeseries", dest="ts", action="store_true",
                       help="the per-scrape time series as CSV (pipes to 'jobscope plot')")
    block = shape.add_mutually_exclusive_group()
    block.add_argument("--cpu", action="store_const", const="cpu", dest="view",
                       help="CPU columns only")
    block.add_argument("--gpu", action="store_const", const="gpu", dest="view",
                       help="GPU columns only")
    shape.add_argument("--dcgm", "--ext", dest="dcgm", action="store_true",
                       help="the full DCGM metric catalog (clocks, temps, PCIe, NVLink, ...)")
    shape.add_argument("--avg", action="store_true",
                       help="running: fold each metric over the job's runtime, making the "
                            "values comparable to jobstats (default: the newest scrape)")
    shape.add_argument("--diagnose", action="store_true",
                       help="add an advisory DIAG column")
    shape.add_argument("--diag-short", dest="diag_short", type=int, default=None,
                       metavar="SECONDS",
                       help="jobs shorter than this get DIAG=short (default from config)")

    out = report.add_argument_group("output")
    out.add_argument("-n", "--noheader", dest="header", action="store_false",
                     help="suppress the header/context block")
    out.add_argument("--csv", action="store_true",
                     help="machine-readable output (pipe to 'jobscope plot')")
    out.add_argument("--step", type=int, default=None, metavar="SECONDS",
                     help="--ts sample interval (default: the scrape interval, widened on "
                          "long jobs to stay under Prometheus' point cap)")
    out.add_argument("--timeout", type=float, default=None,
                     help="seconds per sacct/squeue/Prometheus call "
                          "(default from config; 0 disables)")
    out.add_argument("--workers", type=int, default=None,
                     help="max concurrent Prometheus queries (default from config; the "
                          "queries are I/O-bound so >1 helps)")

    parser = argparse.ArgumentParser(
        prog="jobscope", description=_DESC, epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version="jobscope %s" % __version__)
    subparsers = parser.add_subparsers(dest="command")

    for mode in MODES:
        sub = subparsers.add_parser(
            mode, parents=[base, report], description=_DESC, epilog=_EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            help=("jobs running right now (the default)" if mode == RUNNING
                  else "finished jobs (default window: the last day)"))
        sub.set_defaults(func=handle_report, mode=mode, view=None, explicit_mode=False)

    p_plot = subparsers.add_parser(
        "plot", parents=[base], help="render '--csv' output as a terminal chart")
    plot.add_arguments(p_plot)
    p_plot.set_defaults(func=handle_plot)

    p_describe = subparsers.add_parser(
        "describe", parents=[base], help="describe the columns and metrics")
    p_describe.add_argument("--dcgm", action="store_true",
                            help="describe the DCGM metric catalog instead of the columns")
    p_describe.add_argument("--ext", "--extended", dest="ext", action="store_true",
                            help="with --dcgm, describe the full metric catalog")
    p_describe.add_argument("--diagnose", action="store_true",
                            help="also print the DIAG legend")
    p_describe.set_defaults(func=handle_describe)

    p_config = subparsers.add_parser(
        "config", parents=[base], help="show the config path or print an example")
    p_config.add_argument("--example", action="store_true",
                          help="print an example config file to stdout")
    p_config.add_argument("--path", action="store_true",
                          help="print the config path jobscope would read")
    p_config.set_defaults(func=handle_config)

    return parser, subparsers


def _selects_a_window(argv) -> bool:
    """Whether these arguments name a past window, i.e. imply sacct."""
    return any(arg.split("=")[0] in _WINDOW_FLAGS for arg in argv)


def default_mode(argv) -> str:
    """The mode to assume when the first word is not one.

    A past-window flag means ``finished``; everything else means ``running``.

    Deliberately keyed on flag *names* only. Scanning for bare words to spot a
    JOBID cannot work here -- option values are bare words too, so
    ``-p kempner_eng`` and ``--min-elapsed 0s`` would both look like job IDs, and
    argparse is the only thing that knows which flags take an argument. JOBIDs need
    no guess anyway: both modes share one parser, so the IDs land in ``args.jobids``
    either way and :func:`build_request` switches to the JOBIDS mode on sight.
    """
    return FINISHED if _selects_a_window(argv) else RUNNING


def mode_was_explicit(argv) -> bool:
    """Whether the user named the mode rather than letting it be inferred.

    It matters for JOBIDs: ``jobscope 12345`` looks the job up through sacct, which
    finds it running or finished, but ``jobscope running -j 12345`` asked for the
    live view of it -- an instant snapshot, with --avg available. Without this the
    explicit word would be silently discarded.
    """
    # `live` always meant running jobs, so it counts as naming the mode.
    return bool(argv) and (argv[0] in MODES or argv[0] == "live")


def resolve_argv(argv):
    """Normalize ``argv`` to ``[mode, ...]``, expanding the deprecated aliases.

    A bare JOBID keeps working as the first word, and no first word at all means
    ``running``. Deprecated subcommands are rewritten to their replacement flags
    with a note, so old invocations keep producing their old output.
    """
    if not argv:
        return [RUNNING]
    first = argv[0]
    if first in ("-h", "--help", "--version") or first in MODES or first in UTILITIES:
        return list(argv)
    if first in DEPRECATED:
        extra, spelling = DEPRECATED[first]
        rest = list(argv[1:]) + extra
        print("jobscope: note: '%s' is deprecated; use '%s'" % (first, spelling),
              file=sys.stderr)
        # `live` always meant running jobs; the others kept the sacct default.
        mode = RUNNING if first == "live" else default_mode(rest)
        return [mode] + rest
    return [default_mode(argv)] + list(argv)


def _apply_config(args) -> config.Config:
    path = getattr(args, "config_path", None)
    if path:
        config.set_config(config.load_config(path=path))
    return config.get_config()


def _timeout(args, cfg: config.Config):
    value = args.timeout if args.timeout is not None else cfg.defaults.timeout
    return value if value and value > 0 else None


def _workers(args, cfg: config.Config) -> int:
    value = args.workers if args.workers is not None else cfg.defaults.workers
    if value < 1:
        raise JobscopeError("--workers must be a positive integer")
    return value


# Flags that only mean something in one mode, as (both spellings, short form,
# attribute). Rejected rather than ignored, so a wrong combination is never
# silently dropped.
_FINISHED_ONLY = (("-D/--days", "-D", "days"), ("-N/--lastn", "-N", "lastn"),
                  ("-S/--starttime", "-S", "starttime"),
                  ("-E/--endtime", "-E", "endtime"))


def _check_other_users_allowed(args, cfg: config.Config, me: Optional[str]) -> None:
    """Gate -a/--all-users and a -u naming someone else on group membership.

    Advisory, not a privilege boundary: `sacct -a` and `squeue -u` show the same
    jobs to anyone who runs them directly. The point is to keep the cluster-wide
    views out of the way of people who have no use for them, and it is configurable
    (``[defaults] admin_group``, empty to disable) because the group name is
    site-specific.
    """
    group = cfg.defaults.admin_group
    if not group:
        return
    asking_for_others = args.all_users or (args.user and args.user != me)
    if not asking_for_others or in_group(group):
        return
    who = "every user's jobs" if args.all_users else "%s's jobs" % args.user
    flag = "-a/--all-users" if args.all_users else "-u/--user"
    raise JobscopeError(
        "%s asks for %s, which is limited to members of the '%s' group.\n"
        "Drop %s to report on your own%s."
        % (flag, who, group, flag,
           "" if args.all_users or not me else ", or pass -u %s" % me))


def _min_elapsed(args, cfg: config.Config) -> int:
    """The runtime floor in seconds: the flag if given, else the configured default.

    A bad value is reported against whichever supplied it, so a typo in the config
    file does not read as a bad command line.
    """
    if args.min_elapsed is not None:
        return parse_duration(args.min_elapsed)
    try:
        return parse_duration(cfg.defaults.min_elapsed)
    except JobscopeError as exc:
        raise JobscopeError("[defaults] min_elapsed in the config file is invalid: %s" % exc)


def build_request(args, cfg: Optional[config.Config] = None) -> Request:
    """Validate the flag combination and build the :class:`Request`."""
    cfg = cfg or config.get_config()
    jobids = list(args.jobids) + list(getattr(args, "jobids_opt", None) or [])
    # An explicit `running` keeps the live path even with JOBIDs, narrowing within
    # squeue; an inferred mode yields to the IDs, which sacct resolves either way.
    live_ids = jobids and args.mode == RUNNING and getattr(args, "explicit_mode", False)
    mode = args.mode if (live_ids or not jobids) else JOBIDS

    if mode == RUNNING:
        for flag, short, attr in _FINISHED_ONLY:
            if getattr(args, attr, None) is not None:
                raise JobscopeError(
                    "%s selects a past window, which does not apply to running jobs.\n"
                    "Use 'jobscope finished %s ...', or drop the flag." % (flag, short))
        if args.state != "all":
            raise JobscopeError("-t/--state does not apply to running jobs (all are RUNNING)")
    elif args.avg:
        raise JobscopeError(
            "--avg applies to running jobs only; a finished job's metrics are always "
            "folded over its runtime.")

    if args.days is not None:
        if args.days <= 0:
            raise JobscopeError("-D/--days must be a positive integer")
        if args.lastn is not None:
            raise JobscopeError("-D/--days cannot be combined with -N/--lastn")
        if args.starttime or args.endtime:
            raise JobscopeError("-D/--days sets the window; do not also pass -S/-E")
    if args.lastn is not None and args.lastn <= 0:
        raise JobscopeError("-N/--lastn must be a positive integer")
    if args.all_users and args.user:
        raise JobscopeError("-a/--all-users and -u/--user are mutually exclusive")
    # Explicit JOBIDs ignore the filters entirely (noted below), so there is
    # nothing to gate there -- only a filter can widen the selection to others.
    if not jobids:
        _check_other_users_allowed(args, cfg, default_user())

    if jobids:
        ignored = [name for name, on in (
            ("-D/--days", args.days is not None), ("-N/--lastn", args.lastn is not None),
            ("-S/--starttime", bool(args.starttime)), ("-E/--endtime", bool(args.endtime)),
            ("-p/--partition", bool(args.partition)), ("-u/--user", bool(args.user)),
            ("-a/--all-users", args.all_users), ("-A/--account", bool(args.account)),
        ) if on]
        if ignored:
            print("jobscope: note: explicit JOBIDs given; ignoring %s" % ", ".join(ignored),
                  file=sys.stderr)

    user = None
    if not jobids and not args.all_users:
        user = args.user or default_user()
        if not user:
            raise JobscopeError("could not determine the current user from $USER; "
                                "pass -u/--user or -a/--all-users")

    days = args.days
    if mode == FINISHED and days is None and args.lastn is None \
            and not args.starttime and not args.endtime:
        days = 1        # the default window for finished jobs

    return Request(
        mode=mode, jobids=jobids,
        days=days, lastn=args.lastn, starttime=args.starttime, endtime=args.endtime,
        state=args.state, user=user, all_users=args.all_users,
        account=args.account, partition=args.partition,
        min_elapsed=_min_elapsed(args, cfg), average=args.avg,
    )


def handle_report(args) -> None:
    """The one data path: select jobs, then render at the chosen granularity."""
    cfg = _apply_config(args)
    request = build_request(args, cfg)
    timeout = _timeout(args, cfg)
    workers = _workers(args, cfg)

    view = args.view or "all"
    diagnose = args.diagnose
    if view == "cpu" and diagnose and not args.ts:
        # --ts reports its own dropped flags below; do not say it twice.
        print("note: --diagnose describes GPU use; ignoring it for --cpu", file=sys.stderr)
        diagnose = False
    show_dcgm = view in ("all", "gpu")
    specs = ALL_SPECS if args.dcgm else DEFAULT_SPECS
    options = RenderOptions(
        view=view, show_dcgm=show_dcgm, diagnose=diagnose, csv=args.csv,
        header=args.header,
        min_runtime=(args.diag_short if args.diag_short is not None
                     else cfg.defaults.min_runtime))

    if args.ts:
        # The series is per-GPU per-scrape and carries no host or advisory columns,
        # so say what is being dropped rather than ignoring the flags.
        for flag, on in (("--cpu", view == "cpu"), ("--gpu", view == "gpu"),
                         ("--diagnose", args.diagnose)):
            if on:
                print("note: %s does not apply to --ts (a per-GPU metric series)" % flag,
                      file=sys.stderr)
        emit_timeseries(request, cfg, timeout, workers, specs, args.step, options)
        return

    # The detail granularity renders the fixed DETAIL_COLUMNS, so it takes no spec
    # list; the per-job one splices the profiling block from whichever was chosen.
    selected = resolve(request, cfg, timeout, workers, specs if show_dcgm else None)
    if selected is None:
        return
    renderer = (DetailRenderer(selected.context, options) if args.hwdetail
                else SummaryRenderer(selected.context, options, specs=specs))
    for chunk_ids, records, dcgm_data in selected.chunks:
        renderer.add(chunk_ids, records, dcgm_data)
    renderer.finish()


def handle_plot(args) -> None:
    _apply_config(args)
    plot.run(args)


def handle_describe(args) -> None:
    if args.dcgm:
        describe_dcgm(ALL_SPECS if args.ext else DEFAULT_SPECS)
    else:
        describe(args.diagnose)


def handle_config(args) -> None:
    if args.example:
        sys.stdout.write(config.example_config_text())
        return
    path = args.config_path or os.environ.get(config.CONFIG_ENV) or config.default_config_path()
    if args.path:
        print(path)
        return
    exists = os.path.exists(str(path))
    print("config path: %s%s" % (path, "" if exists else " (not present; using built-in defaults)"))
    print("print an example with: jobscope config --example")


def main(argv=None) -> None:
    raw = sys.argv[1:] if argv is None else list(argv)
    explicit = mode_was_explicit(raw)
    argv = resolve_argv(raw)
    parser, subparsers = build_parser()
    if argv[0] in subparsers.choices:
        # Parse on the chosen subparser so JOBIDs and flags may appear in any
        # order. parse_intermixed_args cannot run on the top parser, where the
        # mode is itself a positional.
        args = subparsers.choices[argv[0]].parse_intermixed_args(argv[1:])
        args.explicit_mode = explicit
    else:  # -h / --help / --version
        args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return
    try:
        handler(args)
    except JobscopeError as exc:
        print("jobscope: error: %s" % exc, file=sys.stderr)
        sys.exit(1)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
