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
import io
import os
import sys
from typing import List, Optional

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
from .sacct import DEFAULT_STATE, default_user
from .select import FINISHED, JOBIDS, RUNNING, Request, emit_timeseries, resolve

MODES = (RUNNING, FINISHED)
UTILITIES = ("plot", "describe", "config")

# Flags an explicit JOBID makes inert -- the IDs are the selection, so there is
# nothing left for a window or a filter to narrow. build_request names these in its
# "ignoring ..." note and narrow_help() hides them, from this one list, so the note
# and the help cannot drift apart.
_JOBID_IGNORES = (
    ("-D/--days", "days"), ("-N/--lastn", "lastn"),
    ("-S/--starttime", "starttime"), ("-E/--endtime", "endtime"),
    ("-p/--partition", "partition"), ("-u/--user", "user"),
    ("-a/--all-users", "all_users"), ("-A/--account", "account"),
    # Explicit IDs skip _select_cmd entirely (select.py returns them as-is), so the
    # -s state filter never runs. Silently ignored before this note existed.
    ("-t/--state", "state"),
)

# Flags that select a past window; their presence means sacct rather than squeue.
_WINDOW_FLAGS = ("-D", "--days", "-N", "--lastn", "-S", "--starttime",
                 "-E", "--endtime", "-t", "--state")

# Old subcommand -> (extra argv, how to spell it now). Kept because
# `dcgm --ts --csv | jobscope plot` appears in the README and the docs, and
# contrib/jobscope_live.py shells out to `live`.
DEPRECATED = {
    "summary": ([], "the default"),
    "detail": (["--per-gpu"], "--per-gpu"),
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
    "  jobscope running --per-gpu        one row per GPU\n"
    "  jobscope finished -D 7 --dcgm     the full metric catalog\n"
    "  jobscope 30012345 --ts | jobscope plot\n"
    "\n"
    "Flags and JOBIDs may be given in any order.")


class _HelpAll(argparse.Action):
    """``--help-all``: the unfiltered help.

    ``-h`` narrows itself to the flags the current invocation can actually use (see
    :func:`narrow_help`), so there has to be a way back to the full list.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        parser.exit()


def build_parser():
    """Construct the argument parser; return ``(parser, subparsers_action)``."""
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("-c", "--config", dest="config_path", metavar="PATH",
                      help="path to a jobscope config file (overrides $JOBSCOPE_CONFIG)")
    base.add_argument("--help-all", action=_HelpAll, nargs=0,
                      help="every option, including those the current flags rule out")

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
    # No argparse choices: the value composes with commas ("-t failed,timeout"), so
    # sacct.states_for validates it and can say what went wrong. Default None rather
    # than "completed" so `running` can tell an explicit -t from the default.
    filters.add_argument("-t", "--state", default=None, metavar="STATE",
                         help="finished: which endings to include -- completed "
                              "(default), failed, timeout, cancelled, or all; "
                              "comma-separated, e.g. -t failed,timeout")

    shape = report.add_argument_group("granularity and columns")
    grain = shape.add_mutually_exclusive_group()
    grain.add_argument("--per-gpu", "--hwdetail", dest="per_gpu", action="store_true",
                       help="one row per GPU, with node name and GPU number "
                            "(--hwdetail is the old name for it)")
    grain.add_argument("--ts", "--timeseries", dest="ts", action="store_true",
                       help="the per-scrape time series as CSV (pipes to 'jobscope plot')")
    grain.add_argument("--plot-ts", "--plot_ts", dest="plot_ts", action="store_true",
                       help="chart that time series instead of writing it: one panel per "
                            "metric, one column per GPU. Needs --nodename on a "
                            "multi-node job")
    shape.add_argument("--nodename", "--node", dest="nodename", default=None,
                       metavar="NODE",
                       help="--per-gpu / --ts: report only this node's GPUs")
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
    shape.add_argument("--no-plot", dest="no_plot", action="store_true",
                       help="omit the efficiency-bars section (shown by default)")
    # Superseded: the bars are the default now. Accepted so a command that named it
    # still runs, with one note, as the deprecated subcommand aliases do.
    shape.add_argument("--plot-avgeff", "--plot_avgeff", dest="plot_avgeff",
                       action="store_true", help=argparse.SUPPRESS)
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
    out.add_argument("--no-color", dest="no_color", action="store_true",
                     help="never tint utilization cells (also respects $NO_COLOR)")
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


def _was_given(args, dest: str) -> bool:
    """Whether *dest* carries a value the user supplied, rather than its default."""
    value = getattr(args, dest, None)
    return value is not None and value is not False and value != ""


def _inert_dests(args) -> set:
    """The options this invocation would reject or silently ignore.

    Every entry mirrors something the run itself already does -- an error raised by
    :func:`build_request` or :func:`handle_report`, a "note: ... ignoring" line, or a
    value no renderer on this path ever reads. That is the whole rule, and it is why
    the narrowed help can be trusted: it hides what would not have worked, not what
    someone judged uninteresting.
    """
    jobids = bool(args.jobids or getattr(args, "jobids_opt", None))
    running = args.mode == RUNNING and (getattr(args, "explicit_mode", False) or not jobids)
    hide = set()
    if jobids:  # build_request: "explicit JOBIDs given; ignoring ..."
        hide.update(dest for _, dest in _JOBID_IGNORES)
    if running:
        hide.update(dest for _, _, dest in _FINISHED_ONLY)  # raises: no past window
        hide.add("state")                                   # raises: all are RUNNING
    else:
        hide.add("avg")          # raises: a finished job is always folded over its runtime
        hide.add("min_elapsed")  # only ever reaches LiveSelection
    if args.ts or args.plot_ts:
        # emit_timeseries drops these with a note; the series has no host, advisory or
        # aggregate columns to put them in, and nothing is plotted. --nodename is not
        # among them: the series carries a NODE column, so the filter applies.
        hide.update({"view", "diagnose", "diag_short", "per_gpu", "no_plot"})
        if args.plot_ts:
            hide.update({"csv", "ts"})   # raises / mutually exclusive
        else:
            hide.add("plot_ts")
    else:
        hide.add("step")  # only emit_timeseries reads it
        hide.add("ts" if args.per_gpu else "nodename")
        if args.per_gpu:
            hide.add("plot_ts")
    if args.view == "cpu":
        # show_dcgm goes false, so the spec list is never built and DIAG has no GPU
        # metric to advise on.
        hide.update({"dcgm", "diagnose", "diag_short"})
    if args.csv:
        hide.update({"no_color", "no_plot"})  # both already inert for a CSV
    return hide


def _flag_name(action) -> str:
    """The long spelling of an option, for naming it in prose."""
    longs = [s for s in action.option_strings if s.startswith("--")]
    return longs[0] if longs else action.option_strings[0]


def narrow_help(sub, argv, explicit: bool) -> List[str]:
    """Hide the options *argv* rules out, and say so in the epilog.

    ``jobscope -j 36441613 --per-gpu -h`` printed all thirty options, twenty of which
    that command cannot use: every window flag (the job ID is the selection), every
    filter, ``--ts`` (mutually exclusive), ``--step`` (``--ts`` only), ``--avg``
    (running only). Reading the help for a command you have already half-written
    should not mean re-reading the ones you have ruled out.

    Returns the names hidden, in declaration order. A tail argparse cannot make sense
    of hides nothing: a full help is a worse answer than a narrow one, but a better
    one than a wrong one.
    """
    probe = [a for a in argv if a not in ("-h", "--help")]
    try:
        args, _ = sub.parse_known_intermixed_args(probe)
    except (SystemExit, argparse.ArgumentError):
        return []
    args.explicit_mode = explicit
    inert = _inert_dests(args)
    hidden = []
    for action in sub._actions:
        if action.dest in inert and action.help is not argparse.SUPPRESS:
            hidden.append(_flag_name(action))
            action.help = argparse.SUPPRESS
    # A mutually exclusive group with a suppressed member makes argparse's usage
    # formatter assert -- it renders "[--a | --b]" from the group while building the
    # option list from the visible actions, and the two then disagree. Drop the hidden
    # members from the group: the constraint is real but it no longer binds anything
    # the reader can see.
    # An emptied group raises instead, so it goes altogether.
    groups = []
    for group in getattr(sub, "_mutually_exclusive_groups", []):
        group._group_actions = [a for a in group._group_actions
                                if a.help is not argparse.SUPPRESS]
        if group._group_actions:
            groups.append(group)
    sub._mutually_exclusive_groups = groups
    if hidden:
        sub.epilog += (
            "\n\nhiding %d option(s) these flags rule out: %s.\n"
            "Pass --help-all for the full list." % (len(hidden), ", ".join(hidden)))
    return hidden


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
        if args.state is not None:
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

    if jobids:
        ignored = [name for name, dest in _JOBID_IGNORES if _was_given(args, dest)]
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
        state=args.state or DEFAULT_STATE, user=user, all_users=args.all_users,
        account=args.account, partition=args.partition,
        min_elapsed=_min_elapsed(args, cfg), average=args.avg,
    )


def _want_color(args) -> bool:
    """Whether to tint the table.

    Only for a table on a terminal: escape codes in a CSV or a redirected file are
    corruption, not decoration, and $NO_COLOR is the cross-tool way to say no.
    """
    if args.csv or args.no_color or os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _plot_timeseries(text: str, args) -> None:
    """Chart the series ``--plot_ts`` just emitted, in place of writing its CSV.

    The two guards are here rather than in the renderer because only the emitted CSV
    knows how many nodes and jobs it covers, and because the fix for each is a flag on
    this side of the pipe.
    """
    _columns, rows = plot.parse_csv(io.StringIO(text))
    if not rows:
        # emit_timeseries has already said why on stderr.
        return
    jobids = sorted({r.get("JOBID") for r in rows if r.get("JOBID")})
    if len(jobids) > 1:
        # render_line keys its series on (NODE, GPU) alone, so two jobs that shared a
        # GPU would concatenate into one line: a chart that looks right and is not.
        raise JobscopeError(
            "--plot_ts charts one job; this selection has %d (%s%s). Pick one with -j JOBID."
            % (len(jobids), ", ".join(jobids[:4]), ", ..." if len(jobids) > 4 else ""))
    nodes = sorted({r.get("NODE") for r in rows if r.get("NODE")})
    if len(nodes) > 1:
        raise JobscopeError(
            "--plot_ts charts one node; this job ran on %d: %s. Add --nodename=NODE."
            % (len(nodes), ", ".join(nodes)))
    plot.run(plot.default_args(kind="line", by="metric", columns=True,
                               no_color=args.no_color),
             fobj=io.StringIO(text))


def handle_report(args) -> None:
    """The one data path: select jobs, then render at the chosen granularity."""
    if args.plot_ts:
        # --plot_ts *is* --ts, with the CSV charted instead of written. Setting it here,
        # before anything reads it, means every --ts path applies unchanged: the schema,
        # --step, and the --nodename guard just below.
        if args.csv:
            raise JobscopeError("--plot_ts draws a chart; drop --csv, or drop --plot_ts "
                                "to keep the CSV")
        args.ts = True
    cfg = _apply_config(args)
    request = build_request(args, cfg)
    timeout = _timeout(args, cfg)
    workers = _workers(args, cfg)

    if args.plot_avgeff:
        print("note: --plot_avgeff is the default now; use --no-plot to omit the "
              "efficiency bars", file=sys.stderr)
    if args.nodename and not (args.per_gpu or args.ts):
        # The per-job table's NODE column is a count of nodes, not a name, so there is
        # nothing there to match; say so rather than filtering nothing.
        raise JobscopeError("--nodename needs --per-gpu or --ts, the views whose rows "
                            "carry a node name")
    view = args.view or "all"
    diagnose = args.diagnose
    if view == "cpu" and diagnose and not args.ts:
        # --ts reports its own dropped flags below; do not say it twice.
        print("note: --diagnose describes GPU use; ignoring it for --cpu", file=sys.stderr)
        diagnose = False
    show_dcgm = view in ("all", "gpu")
    specs = ALL_SPECS if args.dcgm else DEFAULT_SPECS
    # Weight the mean by resource-time wherever the values already span whole
    # runtimes: a finished job's blob does, an explicit job ID's reconstruction
    # does, and running --avg does. The bare running view is a snapshot of one
    # moment, which no amount of elapsed time makes representative.
    time_weighted = request.mode != RUNNING or request.average
    options = RenderOptions(
        view=view, show_dcgm=show_dcgm, diagnose=diagnose, csv=args.csv,
        header=args.header,
        min_runtime=(args.diag_short if args.diag_short is not None
                     else cfg.defaults.min_runtime),
        time_weighted=time_weighted, plot_avgeff=not args.no_plot,
        nodename=args.nodename,
        color=_want_color(args), thresholds=cfg.thresholds)

    if args.ts:
        # The series is per-GPU per-scrape and carries no host or advisory columns,
        # so say what is being dropped rather than ignoring the flags.
        for flag, on in (("--cpu", view == "cpu"), ("--gpu", view == "gpu"),
                         ("--diagnose", args.diagnose)):
            if on:
                print("note: %s does not apply to --ts (a per-GPU metric series)" % flag,
                      file=sys.stderr)
        if not args.plot_ts:
            emit_timeseries(request, cfg, timeout, workers, specs, args.step, options)
            return
        buffer = io.StringIO()
        emit_timeseries(request, cfg, timeout, workers, specs, args.step, options,
                        out=buffer)
        _plot_timeseries(buffer.getvalue(), args)
        return

    # The detail granularity renders the fixed DETAIL_COLUMNS, so it takes no spec
    # list; the per-job one splices the profiling block from whichever was chosen.
    selected = resolve(request, cfg, timeout, workers, specs if show_dcgm else None)
    if selected is None:
        return
    renderer = (DetailRenderer(selected.context, options) if args.per_gpu
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
        sub = subparsers.choices[argv[0]]
        if argv[0] in MODES and any(a in ("-h", "--help") for a in argv[1:]):
            # Narrow before parsing, because argparse prints the help and exits the
            # moment it reaches -h.
            narrow_help(sub, argv[1:], explicit)
        args = sub.parse_intermixed_args(argv[1:])
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
