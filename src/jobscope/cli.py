"""Command-line interface.

The argument tree has one axis per level, so that every option composes with every
selection::

    jobscope [MODE] [scope] [filters] [granularity] [columns] [output]

``MODE`` is the first positional and answers *which jobs*: ``running`` (the
default), ``finished``, or one or more explicit ``JOBID``s. The granularity and
column flags answer *how to show them* and are shared by all three, which the
earlier per-view subcommands could not do -- ``live`` had no ``--cpu``, the
historical views had no ``--min-elapsed``, and so on.

``plot``, ``describe``, ``config`` and ``probe`` are utilities and take the first
slot too.
"""

import argparse
import dataclasses
import io
import os
import re
import sys
from typing import List, Optional, Tuple

from . import config, dcgm, plot, probe, report, rows
from .errors import JobscopeError
from .models import GPU_LEVEL, JOB_LEVEL, NODE_LEVEL
from .report import (
    DetailRenderer,
    RenderOptions,
    SummaryRenderer,
    describe,
    describe_dcgm,
    timeseries_eff,
    timeseries_stats,
    verify_ladder,
)
from .running import format_duration, parse_duration
from .select import FINISHED, JOBIDS, RUNNING, Request, emit_timeseries, resolve
from .slurm import DEFAULT_STATE, default_user

MODES = (RUNNING, FINISHED)
UTILITIES = ("plot", "describe", "config", "probe")

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

# The subcommands that were rewritten into flags, and what replaced each. They are no
# longer accepted, but they stay named here: dropped silently, the word falls through
# as a would-be JOBID and the user gets "sacct: fatal: Bad job/step specified: dcgm",
# which says nothing about what to type instead.
RETIRED = {
    # Renamed rather than retired, but it lands here for the same reason: without an
    # entry the word falls through as a would-be JOBID and Slurm answers
    # "sacct: fatal: Bad job/step specified: doctor".
    "doctor": "probe",
    "summary": "the default (jobscope finished ...)",
    "detail": "--per-gpu",
    "dcgm": "--all-metrics",
    "live": "running",
}

# Flags that were retired, and what to type now. Kept as *defined* flags rather than
# simply deleted, so the message names the replacement instead of argparse's bare
# "unrecognized arguments". --dcgm is here because the word now belongs to a source
# (--gpu-source dcgm) and cannot also mean "the whole catalog": leaving it accepted
# would have made an existing flag quietly mean something else, which is worse than
# an error.
#
# The table is meant to be temporary: drop it whole at the next minor version bump, by
# when "unrecognized arguments" is the right answer because the spellings will have been
# gone for a release. The reason to drop it is help and API surface, not cost -- each
# entry measures ~3.5us against a startup dominated by importing requests.
RETIRED_FLAGS = {
    "--dcgm": "--all-metrics (or --gpu-source dcgm to pick the source)",
    "--ext": "--all-metrics",
    # "blob" said only that the thing was opaque. It is jobstats' per-job summary,
    # stored in sacct's AdminComment -- so the flag, the [gpu]/[host] source value and
    # the module all say jobstats now. See source.RETIRED_SOURCES for the config side.
    "--no-blob": "--no-jobstats",
    # Three flags for one axis, folded into the flag's own value.
    "--stats-per-node": "--stats node",
    "--stats-per-job": "--stats job",
    "--all-categories": "--eff all",
    # Renamed for its axis: it averages over time, not across the GPUs of a job.
    "--avg": "--runtime-avg",
    # Removed rather than renamed. Its three-way comparison rested on Slurm's
    # gres/gpuutil, which is a single point reading and not a mean -- TRESUsageInTot,
    # InAve, InMax and InMin all return the same number -- so the GPU row compared a
    # snapshot against a whole-run average and disagreed in proportion to how fast the
    # metric was moving. Two exporters measuring the same quantity is the comparison it
    # was reaching for.
    "--validate": "--gpu-source nvml against --gpu-source dcgm, and --verify's SWING "
                  "column to see whether either mean is reproducible",
}

# Not a real destination. Every retired spelling shares one, so a dead flag can never
# occupy the dest of a live one -- a rename whose old spelling kept the old dest would
# leave args.<old> readable as None, turning a missed call site from an AttributeError
# into a silent no-op.
_RETIRED_DEST = "_retired"


class _Retired(argparse.Action):
    """Fail with the replacement named, the way :data:`RETIRED` does for subcommands."""

    def __call__(self, parser, namespace, values, option_string=None):
        # Indexed, not .get(): every spelling registered with this action is a key, and
        # a default would answer a future entry with an unrelated replacement.
        raise JobscopeError(
            "%s is no longer a flag; use %s"
            % (option_string, RETIRED_FLAGS[option_string]))

# Flags that select a past window; their presence means sacct rather than squeue.
_WINDOW_FLAGS = ("-D", "--days", "-N", "--lastn", "-S", "--starttime",
                 "-E", "--endtime", "-t", "--state")

# The axis names here are the argument-group titles verbatim -- see _add_report_args and
# the test that pins them together. A summary line that named different things from the
# headings under it left the reader to work out the correspondence.
AXES = ("which jobs", "granularity", "columns", "when", "output")

_DESC = (
    "Slurm job efficiency and GPU utilization reporting.\n"
    "\n"
    "  jobscope [running|finished|JOBID...]\n"
    "           %s\n"
    "\n"
    "One row per job by default, for the jobs running right now."
    % " ".join("[%s]" % axis for axis in AXES))

# What a mode subparser shows instead: the axes move into its usage line, where argparse
# would otherwise print eleven lines of bracketed flag names -- and where an error message
# will reprint them, which usage=SUPPRESS would have cost.
_MODE_DESC = (
    "Slurm job efficiency and GPU utilization reporting.\n"
    "\n"
    "One row per job by default. Each axis below is one question; -h lists only the\n"
    "flags the command you are writing can actually use, --help-all lists them all.")


def _mode_usage(mode: str) -> str:
    return "jobscope %s [JOBID ...] %s" % (
        mode, " ".join("[%s]" % axis for axis in AXES))

_EPILOG = (
    "examples:\n"
    "  jobscope                             your running jobs\n"
    "  jobscope -j <jobid>                  one job, running or finished\n"
    "  jobscope -p <partition>              your jobs on one partition (-a for everyone)\n"
    "  jobscope finished -D 2               your last 2 days\n"
    "  jobscope finished -S 2026-08-01      from an explicit start date\n"
    "  jobscope -j <jobid> --plot_ts        chart its series, one panel per metric\n"
    "  jobscope -j <jobid> --plot_ts_overlay  the same series overlaid, one panel per GPU\n"
    "\n"
    "Flags and JOBIDs may be given in any order.")

# Only the top-level help carries the pointer: bare `jobscope --help` listed six
# subcommands and seven examples using -j, -p and --plot_ts, none of which it showed.
# Under a mode the reader has already followed it, and the axes are right there.
_TOP_EPILOG = (
    _EPILOG + "\n"
    "\n"
    "Flags live under the command, grouped by axis under a mode:\n"
    "  jobscope probe -h          check what this cluster exposes before reporting on it\n"
    "  jobscope finished -h       the flags for a report over a past window\n"
    "  jobscope <anything> -h     narrowed to what that command can use")


class _HelpFormat(argparse.RawDescriptionHelpFormatter):
    """Raw description, and a wider invocation column.

    At argparse's default of 24 the summaries did not fit beside their flags, so entries
    like ``-j JOBID, --jobid JOBID`` spent three lines saying one thing. The point of
    shortening them was to be scannable.
    """

    def __init__(self, prog, **kw):
        kw.setdefault("max_help_position", 34)
        super().__init__(prog, **kw)


class _Version(argparse.Action):
    """``--version``, resolved when asked rather than when the parser is built.

    argparse's own ``action="version"`` wants the string at build time, which means
    importing importlib.metadata on every invocation to serve a value only this flag
    prints -- 44ms, measured. See jobscope/__init__.py's __getattr__.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        from . import __version__
        print("jobscope %s" % __version__)
        parser.exit()


class _HelpAll(argparse.Action):
    """``--help-all``: the unfiltered help.

    ``-h`` narrows itself to the flags the current invocation can actually use (see
    :func:`narrow_help`), so there has to be a way back to the full list.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        full_help(parser)
        parser.print_help()
        parser.exit()


def _add_report_args(report) -> None:
    """Add the reporting flags to one mode subparser.

    Called per subparser rather than shared through ``parents=``: argparse's
    ``_add_container_actions`` recreates a parent's mutually exclusive groups by calling
    ``self.add_mutually_exclusive_group()`` on the child **parser**, not on the matching
    argument group. So every axis-defining flag -- ``--per-*``, ``--ts``, ``--cpu``/
    ``--gpu``, ``--runtime-avg``/``--instant`` -- lost its heading and landed in bare
    ``options:``, while the peripheral flags kept theirs. Ten flags in the wrong place is
    most of why the help did not read as a tree.

    Building per subparser also stops :func:`narrow_help` and :func:`brief_help` mutating
    actions that ``running`` and ``finished`` shared.
    """
    # One group per question, titled with the same words _DESC advertises, so the summary
    # line at the top and the headings under it name the same axes. argparse prints a
    # group's description under its title, which is where the question goes -- four groups
    # carried a title and no description, leaving the reader to infer the axis from the
    # flags rather than being told it.
    scope = report.add_argument_group(
        "which jobs", "which jobs to report on. With none of these: your own, running now")
    scope.add_argument("jobids", nargs="*", metavar="JOBID",
                       help="specific job IDs, running or finished")
    scope.add_argument("-j", "--jobid", action="append", dest="jobids_opt", metavar="JOBID",
                       help="a job ID (repeatable; alternative to the positional JOBID)")
    scope.add_argument("-D", "--days", type=int, metavar="N",
                       help="finished: jobs in the last N days (default: 1)")
    scope.add_argument("-N", "--lastn", type=int, metavar="N",
                       help="finished: the most recent N jobs")
    scope.add_argument("-S", "--starttime", metavar="TIME",
                       help=_help("finished: window start",
                                  "e.g. 2026-07-15 or 2026-07-15T09:00:00; without -E, "
                                  "that day alone"))
    scope.add_argument("-E", "--endtime", metavar="TIME",
                       help="finished: window end, same format as -S")
    scope.add_argument("--min-elapsed", dest="min_elapsed",
                       metavar="DURATION", default=None,
                       help=_help("running: only jobs older than this",
                                  "default from config: %s; e.g. '5m', '2h', '0s' for no "
                                  "floor" % config.DEFAULT_MIN_ELAPSED))

    # Same axis as the block above -- these narrow the same selection -- so one heading.
    filters = scope
    filters.add_argument("-p", "--partition", help="narrow to this partition")
    filters.add_argument("-u", "--user", help="user (default: current user, $USER)")
    filters.add_argument("-a", "--all-users", dest="all_users", action="store_true",
                         help="every user's jobs, not just your own")
    filters.add_argument("-A", "--account", help="narrow to this account")
    # No argparse choices: the value composes with commas ("-t failed,timeout"), so
    # sacct.states_for validates it and can say what went wrong. Default None rather
    # than "completed" so `running` can tell an explicit -t from the default.
    filters.add_argument("-t", "--state", default=None, metavar="STATE",
                         help=_help("finished: which endings to include",
                                    "completed (default), failed, timeout, cancelled, or "
                                    "all; comma-separated, e.g. -t failed,timeout"))

    shape = report.add_argument_group(
        "granularity", "one row per what -- or a per-scrape time series instead of rows")
    grain = shape.add_mutually_exclusive_group()
    # One axis, three points on it. --per-job is the default and a no-op; it exists so the
    # axis is visible -- a lone --per-gpu gave no hint that a scale existed at all -- and
    # so a script can say which level it means rather than relying on the absence of a flag.
    grain.add_argument("--per-job", dest="per_job", action="store_true",
                       help="one row per job (the default)")
    grain.add_argument("--per-node", dest="per_node", action="store_true",
                       help=_help("one row per node",
                                  "with its GPU figures pooled across the cards it holds "
                                  "-- 16 rows where --per-gpu gives 128"))
    grain.add_argument("--per-gpu", dest="per_gpu", action="store_true",
                       help="one row per GPU, with node name and GPU number")
    grain.add_argument("--verify", dest="verify", nargs="?", const=True,
                       default=False, metavar="WINDOW",
                       help=_help("check one job before acting on it",
                                  "per metric, the min/max and the mean over a ladder of "
                                  "windows, the share of samples under its cutoff, the "
                                  "longest unbroken idle stretch, and the shape that "
                                  "follows. Takes an optional window to bound the fetch "
                                  "on a very long job"))
    grain.add_argument("--ts", dest="ts", nargs="?", const=True,
                       default=False, metavar="WINDOW",
                       help=_help("the per-scrape time series as CSV",
                                  "pipes to 'jobscope plot'. Takes an optional window -- "
                                  "'--ts 1h' is the last hour of the run, not all of it"))
    grain.add_argument("--plot-ts", "--plot_ts", dest="plot_ts", nargs="?", const=True,
                       default=False, metavar="WINDOW",
                       help=_help("chart that series, one panel per metric",
                                  "one column per GPU. Takes the same optional window. "
                                  "Needs --nodename on a multi-node job"))
    grain.add_argument("--plot-ts-overlay", "--plot_ts_overlay", dest="plot_ts_overlay",
                       nargs="?", const=True, default=False, metavar="WINDOW",
                       help=_help("the same series overlaid, one panel per GPU",
                                  "every metric on a shared axis, one row per node -- so "
                                  "a multi-node job needs no --nodename. Answers 'did "
                                  "these move together', where --plot_ts answers 'how did "
                                  "this one move'. Watts are omitted (they cannot share an "
                                  "axis with percentages) and each panel legends its own "
                                  "metrics"))
    level = shape.add_mutually_exclusive_group()
    # One flag for one axis. These were --stats/--stats-per-node/--stats-per-job, three
    # spellings setting this same dest to three constants. `choices` also keeps the
    # optional value from swallowing a JOBID: `--stats 12345` is an invalid choice rather
    # than a window-style misreading, which is why this needs no _reclaim equivalent.
    level.add_argument("--stats", nargs="?", const="gpu", default=None,
                       choices=("gpu", "node", "job"), metavar="LEVEL",
                       help=_help("--ts: summarize the series instead of writing it",
                                  "min/mean/max/last per metric over the window, per GPU "
                                  "(default), or pooled per 'node', or across the whole "
                                  "job"))
    shape.add_argument("--eff", nargs="?", const=True, default=False,
                       metavar="all",
                       help=_help("--ts: group the jobs into efficiency categories",
                                  "wasteful, inefficient, needs improvement, average, "
                                  "good -- by each one's best metric. The cutoffs are per "
                                  "metric and come from [thresholds.timeslice] in your "
                                  "config; each heading states the ones it used. "
                                  "'--eff all' lists the 'good' jobs too, instead of "
                                  "counting them"))
    # Folded into the two flags above. Defined rather than deleted so the message names
    # the replacement; see RETIRED_FLAGS.
    for old in ("--stats-per-node", "--stats-per-job", "--all-categories"):
        shape.add_argument(old, dest=_RETIRED_DEST,
                           action=_Retired, nargs=0, help=argparse.SUPPRESS)
    shape.add_argument("--nodename", "--node", dest="nodename", default=None,
                       metavar="NODE",
                       help=_help("report only this node",
                                  "narrows any view, including the summary, whose numbers "
                                  "are recomputed over it"))
    shape.add_argument("--gpuid", dest="gpuid", default=None,
                       metavar="IDS",
                       help=_help("only these GPUs, comma-separated",
                                  "--gpuid 0,1; ids are per node. Same spelling as "
                                  "'jobscope plot --gpuid'"))
    cols = report.add_argument_group(
        "columns", "which measurements appear, and which exporter they come from")
    block = cols.add_mutually_exclusive_group()
    block.add_argument("--cpu", action="store_const", const="cpu", dest="view",
                       help="CPU columns only")
    block.add_argument("--gpu", action="store_const", const="gpu", dest="view",
                       help="GPU columns only")
    cols.add_argument("--gpu-source", dest="gpu_source", default=None,
                       metavar="SOURCE",
                      help=_help("where the GPU numbers come from",
                                 "dcgm, nvml or summary, comma-separated for an order "
                                 "(default from [gpu] source). Each column takes its own "
                                 "best available source, so naming one promotes it rather "
                                 "than dropping what it cannot serve. Naming it here also "
                                 "outranks the summary jobstats stored"))
    cols.add_argument("--all-metrics", dest="all_metrics",
                       action="store_true",
                      help=_help("every metric the chosen source publishes",
                                 "not just the default columns (clocks, temps, PCIe, "
                                 "NVLink, ...)"))
    cols.add_argument("--dcgm", "--ext", dest=_RETIRED_DEST, action=_Retired,
                       nargs=0, help=argparse.SUPPRESS)
    # "--avg" said which operation but not over what axis, and read as an average across
    # GPUs sitting next to --per-gpu. It averages over *time*, and that is the axis the
    # name has to carry: the summary block's own figures move by a third between the two
    # answers (see report.averaging_note).
    #
    # Now the default for running jobs: one scrape of a bursty job is a coin toss across
    # its whole range, and a reader who did not ask for that should not silently get it.
    # So --runtime-avg is a no-op, kept for the reason --per-job is (see the grain group):
    # the axis stays visible, and a script can say which reduction it means rather than
    # relying on the absence of a flag.
    when = report.add_argument_group(
        "when", "what span each value covers: a moment, or the whole runtime")
    span = when.add_mutually_exclusive_group()
    span.add_argument("--runtime-avg", dest="runtime_avg", action="store_true",
                      help=_help("running: average over the job's runtime",
                                 "the default; this only says so explicitly"))
    span.add_argument("--instant", dest="instant", action="store_true",
                      help=_help("running: the newest scrape instead",
                                 "one query per metric however many jobs, against one per "
                                 "job per metric, so this is the fast one; a bursty job "
                                 "then reads as whatever it was doing that second"))
    when.add_argument("--avg", dest=_RETIRED_DEST, action=_Retired,
                       nargs=0, help=argparse.SUPPRESS)
    cols.add_argument("--no-jobstats", dest="no_jobstats",
                       action="store_true",
                      help=_help("read CPU%%/MEM%%/GPU%%/GMEM%% from Prometheus always",
                                 "even for finished jobs, instead of the summary jobstats "
                                 "stored in sacct's AdminComment (slower; use to compare "
                                 "the two, or where jobstats is not deployed)"))
    cols.add_argument("--no-blob", dest=_RETIRED_DEST, action=_Retired, nargs=0,
                       help=argparse.SUPPRESS)
    out = report.add_argument_group(
        "output", "how the result is written, rather than what it contains")
    out.add_argument("--no-plot", dest="no_plot", action="store_true",
                     help="omit the efficiency-bars section (shown by default)")
    out.add_argument("-n", "--noheader", dest="header", action="store_false",
                     help="suppress the header/context block")
    out.add_argument("--csv", action="store_true",
                     help="machine-readable output (pipe to 'jobscope plot')")
    out.add_argument("--no-color", dest="no_color", action="store_true",
                     help="never tint utilization cells (also respects $NO_COLOR)")
    when.add_argument("--step", type=int, default=None, metavar="SECONDS",
                      help=_help("--ts: sample interval",
                                 "default: the scrape interval, widened on long jobs to "
                                 "stay under Prometheus' point cap"))
    out.add_argument("--timeout", type=float, default=None,
                     help=_help("seconds per sacct/squeue/Prometheus call",
                                "default from config; 0 disables"))
    out.add_argument("--workers", type=int, default=None,
                     help=_help("max concurrent Prometheus queries",
                                "default from config; the queries are I/O-bound so "
                                ">1 helps"))


def build_parser():
    """Construct the argument parser; return ``(parser, subparsers_action)``."""
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("-c", "--config", dest="config_path", metavar="PATH",
                      help=_help("path to a jobscope config file",
                                 "overrides $JOBSCOPE_CONFIG"))
    base.add_argument("--help-all", action=_HelpAll, nargs=0,
                      help=_help("every option",
                                 "including those the current flags rule out, and the "
                                 "full text of each"))

    parser = argparse.ArgumentParser(
        prog="jobscope", description=_DESC, epilog=_TOP_EPILOG,
        formatter_class=_HelpFormat)
    parser.add_argument("--version", action=_Version, nargs=0,
                        help="print the version and exit")
    subparsers = parser.add_subparsers(dest="command")

    for mode in MODES:
        sub = subparsers.add_parser(
            mode, parents=[base], description=_MODE_DESC, epilog=_EPILOG,
            usage=_mode_usage(mode), formatter_class=_HelpFormat,
            help=("jobs running right now (the default)" if mode == RUNNING
                  else "finished jobs (default window: the last day)"))
        _add_report_args(sub)
        sub.set_defaults(func=handle_report, mode=mode, view=None, explicit_mode=False)

    p_plot = subparsers.add_parser(
        "plot", parents=[base], help="render '--csv' output as a terminal chart")
    plot.add_arguments(p_plot)
    p_plot.set_defaults(func=handle_plot)

    p_describe = subparsers.add_parser(
        "describe", parents=[base], help="describe the columns and metrics")
    p_describe.add_argument("--metrics", dest="metrics", action="store_true",
                            help="describe the GPU metric catalog instead of the columns")
    p_describe.add_argument("--all-metrics", dest="all_metrics",
                            action="store_true",
                            help="the full catalog (implies --metrics)")
    p_describe.add_argument("--dcgm", "--ext", dest=_RETIRED_DEST, action=_Retired,
                            nargs=0, help=argparse.SUPPRESS)
    p_describe.set_defaults(func=handle_describe)

    p_config = subparsers.add_parser(
        "config", parents=[base], help="show the config path or print an example")
    p_config.add_argument("--example", action="store_true",
                          help="print an example config file to stdout")
    p_config.add_argument("--path", action="store_true",
                          help="print the config path jobscope would read")
    p_config.set_defaults(func=handle_config)

    p_probe = subparsers.add_parser(
        "probe", parents=[base],
        help="check what this cluster exposes and whether jobscope can read it")
    p_probe.add_argument("--metrics", nargs="?", const="", metavar="JOBID",
                          help="also list the metrics the server carries for a job, by "
                               "family (a recent single-GPU job if none is named); with "
                               "--full, each one's current value as the server returns it")
    p_probe.add_argument("--toml", nargs="?", const="", metavar="JOBID",
                          help="print the discovered metrics as an editable [metrics] "
                               "block, to append to a config file")
    p_probe.add_argument("--init", action="store_true",
                         help="write a config for this site from what was detected "
                              "(only if none exists; otherwise prints it)")
    p_probe.add_argument("--full", action="store_true",
                         help="more detail: with --init, every remaining knob, commented; "
                              "with --coverage, the hosts whose job-to-card mapping is "
                              "wrong; with --metrics, each series' current value")
    p_probe.add_argument("--validate", dest=_RETIRED_DEST, action=_Retired, nargs=0,
                         help=argparse.SUPPRESS)
    p_probe.add_argument("--coverage", nargs="?", const="", metavar="PARTITION",
                          help="which hosts publish the series behind each column, and "
                               "what is missing. With a PARTITION, compared against that "
                               "partition's own nodes and the absent ones are named")
    # Every check above takes an optional value, so argparse stops consuming one the
    # moment another flag intervenes: `--coverage --full kempner` left `kempner` as an
    # unrecognised argument. Accepting it here and handing it back (see _reclaim_probe_
    # target) makes the two orderings equivalent, which is what someone who wrote
    # `--coverage kempner --full` once will expect.
    p_probe.add_argument("target", nargs="?", default=None, metavar="PARTITION|JOBID",
                         help="the partition or job the chosen check applies to, for "
                              "writing it after the flag instead of beside it")
    p_probe.set_defaults(func=handle_probe)

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
    running view of it -- averaged over its runtime, with --instant available. Without this the
    explicit word would be silently discarded.
    """
    return bool(argv) and argv[0] in MODES


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
        if not jobids:
            # Raises: a window selection holds only finished jobs, and those are always
            # folded over their runtime. An explicit JOBID can name a job that is still
            # running, where --runtime-avg is exactly the right flag -- so it stays offered
            # there, matching the guard in build_request.
            hide.update(("runtime_avg", "instant"))
        hide.add("min_elapsed")  # only ever reaches RunningSelection
    if args.ts or args.plot_ts:
        # emit_timeseries drops these with a note; the series has no host, advisory or
        # aggregate columns to put them in, and nothing is plotted. --nodename is not
        # among them: the series carries a NODE column, so the filter applies.
        hide.update({"view", "per_gpu", "no_plot"})
        if args.plot_ts:
            # raises / exclusive / both noted as ignored below
            hide.update({"csv", "ts", "stats", "eff"})
        else:
            hide.add("plot_ts")
            # A plain --ts already writes CSV, so --csv adds nothing -- the two outputs
            # are byte-identical. Guarded, not blanket: with --stats (or --eff) the
            # flag is what turns the summary table into CSV, so it is doing something
            # there and must stay visible.
            if not (args.stats or args.eff):
                hide.add("csv")
    else:
        hide.add("step")  # only emit_timeseries reads it
        detail = args.per_gpu or args.per_node
        hide.add("ts" if detail else "nodename")
        if detail:
            hide.add("plot_ts")
        # All three summarize a series, so all three raise without one.
        hide.update({"stats", "eff"})
    if args.view == "cpu":
        # show_dcgm goes false, so no GPU spec list is built -- which makes both the
        # width of that list and where it would have been read from inert.
        hide.update({"all_metrics", "gpu_source"})
    if args.csv:
        hide.update({"no_color", "no_plot"})  # both already inert for a CSV
    return hide


def _flag_name(action) -> str:
    """The long spelling of an option, for naming it in prose."""
    longs = [s for s in action.option_strings if s.startswith("--")]
    return longs[0] if longs else action.option_strings[0]


# A flag's help carries two forms separated by this: the summary `-h` prints, and the
# detail `--help-all` adds. A control character, not punctuation, so no prose can contain
# one by accident -- and never written by hand, since _help() puts it there. Both
# brief_help and full_help remove it, so it cannot reach a terminal.
_HELP_SPLIT = "\x00"


def _help(summary: str, detail: str = "") -> str:
    """A flag's help as a one-line summary plus the detail behind ``--help-all``.

    Where the summary ends is an authoring decision rather than a rule applied to prose.
    Splitting on the first sentence fails on exactly the entries that most need
    shortening: ``--verify``'s first sentence runs four lines, and its summary ends at a
    colon. So the author says where, and a flag with nothing to add passes no detail and
    reads the same in both forms.
    """
    return summary + (_HELP_SPLIT + detail if detail else "")


def _rewrite_help(parser, pick) -> None:
    """Rebuild every two-form help string through ``pick(summary, detail)``.

    The same mechanism :func:`narrow_help` uses -- mutate ``action.help``, then let
    argparse print -- and safe for the same reason: help is printed once and the process
    exits. ``parents=`` shares action objects between subparsers, so a caller that wants a
    clean parser builds one (:func:`build_parser`), as the tests do.
    """
    for action in parser._actions:
        if isinstance(action.help, str) and _HELP_SPLIT in action.help:
            action.help = pick(*action.help.split(_HELP_SPLIT, 1))


def brief_help(parser) -> None:
    """Keep only each flag's summary, for ``-h``.

    117 lines is not a thing anyone reads. The detail is not deleted -- ``--help-all``
    already existed as the everything view, and is now where the paragraphs live.
    """
    _rewrite_help(parser, lambda summary, _detail: summary)


def full_help(parser) -> None:
    """Both halves, for ``--help-all``."""
    _rewrite_help(parser, lambda summary, detail: ("%s %s" % (summary, detail)).strip())


def narrow_help(sub, argv, explicit: bool) -> List[str]:
    """Hide the options *argv* rules out, and say so in the epilog.

    ``jobscope -j 36441613 --per-gpu -h`` printed all thirty options, twenty of which
    that command cannot use: every window flag (the job ID is the selection), every
    filter, ``--ts`` (mutually exclusive), ``--step`` (``--ts`` only), ``--runtime-avg``
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
    """Normalize ``argv`` to ``[mode, ...]``.

    A bare JOBID keeps working as the first word, and no first word at all means
    ``running``.
    """
    if not argv:
        return [RUNNING]
    first = argv[0]
    if first in ("-h", "--help", "--version") or first in MODES or first in UTILITIES:
        return list(argv)
    return [default_mode(argv)] + list(argv)


def _apply_config(args) -> config.Config:
    path = getattr(args, "config_path", None)
    source = getattr(args, "gpu_source", None)
    # Reloaded when --gpu-source is given even without -c, because the order has to be
    # in place before the config resolves a metric name -- see load_config.
    if path or source:
        config.set_config(config.load_config(path=path, gpu_source=source))
    cfg = config.get_config()
    # Every command that renders goes through here first, and tint() is called from
    # too many places to hand a palette to each -- so the colours are installed once,
    # here, rather than threaded.
    report.set_palette(cfg.palette)
    return cfg


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
    # A comma is never valid in a job ID, and there is one thing people type that
    # produces one: `--gpu 0,1`, expecting `jobscope plot`'s GPU filter. --gpu takes
    # no value here -- it picks the GPU *columns* -- so the list falls through to this
    # positional and the run charts every GPU while warning about a job named "0,1".
    # A silently wrong chart is worse than no chart.
    retired = [j for j in jobids if str(j) in RETIRED]
    if retired:
        first = str(retired[0])
        raise JobscopeError(
            "'%s' was renamed; use '%s'" % (first, RETIRED[first]) if first == "doctor"
            else "'%s' is no longer a subcommand; use %s" % (first, RETIRED[first]))
    listy = [j for j in jobids if "," in str(j)]
    if listy:
        hint = (" Did you mean --gpuid %s? --gpu selects the GPU columns and takes no"
                " value." % listy[0]) if args.view == "gpu" else ""
        raise JobscopeError("%s is not a job ID -- job IDs have no commas.%s"
                            % (", ".join(repr(j) for j in listy), hint))
    # An explicit `running` keeps the live path even with JOBIDs, narrowing within
    # squeue; an inferred mode yields to the IDs, which sacct resolves either way.
    running_ids = jobids and args.mode == RUNNING and getattr(args, "explicit_mode", False)
    mode = args.mode if (running_ids or not jobids) else JOBIDS

    if mode == RUNNING:
        for flag, short, attr in _FINISHED_ONLY:
            if getattr(args, attr, None) is not None:
                raise JobscopeError(
                    "%s selects a past window, which does not apply to running jobs.\n"
                    "Use 'jobscope finished %s ...', or drop the flag." % (flag, short))
        if args.state is not None:
            raise JobscopeError("-t/--state does not apply to running jobs (all are RUNNING)")
    elif (args.runtime_avg or args.instant) and mode != JOBIDS:
        # Scoped to a *window* selection, which by construction holds only finished
        # jobs. An explicit JOBID can name a job that is still running -- this used to
        # reject that with a message asserting the job had finished, which was both
        # false and a refusal of the one flag that would have folded its window.
        # JOBIDS carries whatever states the ids have, so the choice is per record;
        # see JobRecord.unfinished.
        flag = "--runtime-avg" if args.runtime_avg else "--instant"
        raise JobscopeError(
            "%s applies to running jobs only; a finished job's metrics are always "
            "averaged over its runtime, there being no newer scrape to read." % flag)

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
        days = cfg.defaults.days    # the default window for finished jobs

    return Request(
        mode=mode, jobids=jobids,
        days=days, lastn=args.lastn, starttime=args.starttime, endtime=args.endtime,
        state=args.state or cfg.defaults.state, user=user, all_users=args.all_users,
        account=args.account, partition=args.partition,
        min_elapsed=_min_elapsed(args, cfg),
        # Averaging over each runtime is the default; --instant declines it. There is
        # nothing for --runtime-avg to override, which is why it is a no-op -- see its help.
        average=not args.instant,
        no_jobstats=getattr(args, "no_jobstats", False),
    )


def _want_color(args) -> bool:
    """Whether to tint the table.

    Only for a table on a terminal: escape codes in a CSV or a redirected file are
    corruption, not decoration, and $NO_COLOR is the cross-tool way to say no.

    $FORCE_COLOR is its mirror, for the case where the destination is not a tty but
    the caller knows it wants colour anyway -- scripts/make_screenshots.sh piping into
    a recorder, or a CI job rendering docs. Never over --csv or --no-color: those are
    explicit, and a machine format with escapes in it is corrupt whoever asked.
    """
    if args.csv or args.no_color or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


# A job ID: digits, plus the _N of an array task and the .batch/.0 of a step.
_JOBID_RE = re.compile(r"^\d+([_.]\w+)*$")


def _plot_flag(args) -> str:
    """Which chart flag the user typed, for the messages that name one.

    handle_report folds --plot_ts_overlay into plot_ts, so every guard downstream would
    otherwise tell someone to "drop --plot_ts" about a flag they did not type.
    """
    return ("--plot_ts_overlay" if getattr(args, "plot_ts_overlay", False) is not False
            else "--plot_ts")


def _ts_value(args) -> Tuple[str, object]:
    """``(flag, value)`` for whichever of --ts / --plot_ts / --plot_ts_overlay carries one.

    The overlay is checked first because handle_report has already folded its value into
    ``plot_ts``: both then hold it, and the note this feeds has to name the flag the user
    actually typed.
    """
    if getattr(args, "plot_ts_overlay", False) not in (None, False, True):
        return "--plot_ts_overlay", args.plot_ts_overlay
    if args.plot_ts not in (None, False, True):
        return "--plot_ts", args.plot_ts
    if getattr(args, "verify", False) not in (None, False, True):
        return "--verify", args.verify
    return "--ts", args.ts


def _reclaim_jobid_after_ts(args) -> None:
    """A JOBID written after ``--ts`` is still a JOBID.

    The optional window means argparse now consumes the next bare word, so
    ``jobscope --ts 12345`` would take the job as its window. Handing it back is
    unambiguous rather than a guess: the two grammars are disjoint, because a window
    always carries a unit (30s/5m/2h/7d) and a job ID never does. Without this the
    flag would have quietly broken every ``--ts JOBID`` already in someone's history.
    """
    flag, value = _ts_value(args)
    if value in (None, False, True) or not _JOBID_RE.match(str(value)):
        return
    args.jobids = list(args.jobids) + [str(value)]
    # The overlay sets both, since its value was folded into plot_ts and either would
    # otherwise be left holding a job ID where a window belongs.
    for dest in {"--ts": ["ts"], "--plot_ts": ["plot_ts"],
                 "--verify": ["verify"],
                 "--plot_ts_overlay": ["plot_ts", "plot_ts_overlay"]}[flag]:
        setattr(args, dest, True)
    # Say which reading was taken. A bare number cannot be both a job ID and a
    # window, and someone who meant "60 minutes" should not have to work out from an
    # empty report that it was read as job 60.
    print("jobscope: note: read %r after %s as a job ID; a window needs a unit, "
          "e.g. %s 60m" % (str(value), flag, flag), file=sys.stderr)


def _ts_window(args) -> Optional[int]:
    """``--ts WINDOW`` / ``--plot_ts WINDOW`` in seconds, or None for the whole run."""
    flag, value = _ts_value(args)
    if value in (None, False, True):
        return None
    try:
        return parse_duration(value)
    except JobscopeError:
        raise JobscopeError(
            "%s takes a duration with a unit, e.g. %s 1h or %s 90m (got %r)"
            % (flag, flag, flag, value))


def _emitted_series(text: str):
    """``(columns, rows)`` from a series just emitted into a buffer, or ``None``.

    None means there was nothing to read; ``emit_timeseries`` has already said why on
    stderr, so every consumer below simply returns.
    """
    columns, rows = plot.parse_csv(io.StringIO(text))
    return (columns, rows) if rows else None


def _eff_timeseries(text: str, options, level: str, show_all: bool) -> None:
    """Sort the series --eff just emitted into categories."""
    found = _emitted_series(text)
    if found:
        timeseries_eff(found[1], found[0], options, level=level, show_all=show_all)


def _verify_series(text: str, options, cfg) -> None:
    """Summarize the series --verify just emitted as a window ladder."""
    found = _emitted_series(text)
    if found:
        verify_ladder(found[1], plot.metric_cols(found[0]), options,
                      windows=cfg.report.verify_windows)


def _stats_timeseries(text: str, options, level: str) -> None:
    """Summarize the series --stats just emitted, in place of writing its CSV."""
    found = _emitted_series(text)
    if found:
        timeseries_stats(found[1], plot.metric_cols(found[0]), options, level=level)


def _plot_timeseries(text: str, args) -> None:
    """Chart the series ``--plot_ts`` just emitted, in place of writing its CSV.

    The two guards are here rather than in the renderer because only the emitted CSV
    knows how many nodes and jobs it covers, and because the fix for each is a flag on
    this side of the pipe.
    """
    found = _emitted_series(text)
    if not found:
        return
    rows = found[1]
    jobids = sorted({r.get("JOBID") for r in rows if r.get("JOBID")})
    if len(jobids) > 1:
        # render_line keys its series on (NODE, GPU) alone, so two jobs that shared a
        # GPU would concatenate into one line: a chart that looks right and is not.
        raise JobscopeError(
            "%s charts one job; this selection has %d (%s%s). Pick one with -j JOBID."
            % (_plot_flag(args), len(jobids), ", ".join(jobids[:4]),
               ", ..." if len(jobids) > 4 else ""))
    overlay = getattr(args, "plot_ts_overlay", False) is not False
    nodes = sorted({r.get("NODE") for r in rows if r.get("NODE")})
    if len(nodes) > 1 and not overlay:
        # Lifted for the overlay, which gives each node its own row -- that is the
        # layout's reason to exist. The one-job guard above still applies to both:
        # render_line keys its series on (NODE, GPU) alone.
        raise JobscopeError(
            "--plot_ts charts one node; this job ran on %d: %s. Add --nodename=NODE,"
            " or use --plot_ts_overlay for a row per node."
            % (len(nodes), ", ".join(nodes)))
    # Name what is being charted. Without it a windowed chart is indistinguishable
    # from a whole-run one -- the x axis counts minutes from the window's own start.
    window = _ts_window(args)
    # A count once there is more than one, since the overlay labels each row with its
    # own node and naming only the first here would contradict the rows below it.
    where = nodes[0] if len(nodes) == 1 else ("%d nodes" % len(nodes) if nodes else "?")
    print("job %s  %s%s" % (jobids[0], where,
                            "  last %s" % format_duration(window) if window else ""))
    # The CSV is already curated to exactly the metrics --ts resolved to show
    # (KEY_SPECS, ALL_SPECS, or CPU%/MEM%), so charting everything present in it
    # is always correct -- there is no narrower in-CSV subset left to fall back to.
    #
    # One panel per metric, each on its own axis. That is what makes `all=True` safe:
    # POWER_W in watts beside percentages is fine when nothing shares a scale, which is
    # why the two belong together -- put watts on a shared axis and a 500 W line pins
    # the scale and flattens every percentage onto the floor. The overlaid single-panel
    # chart is one pipe away, and the README shows it:
    #   jobscope -j JOB --ts --csv | jobscope plot
    # --plot_ts_overlay differs by `by` alone: "gpu" puts the metrics on one shared axis
    # in a panel per GPU, and --columns then packs those panels into a row per node.
    # Everything else -- the curated metric set, all=True, the colour choice -- is the
    # same call, which is what keeps the two layouts describing the same series.
    plot.run(plot.default_args(kind="line", by="gpu" if overlay else "metric",
                               columns=True, no_color=args.no_color, all=True),
             fobj=io.StringIO(text))


def handle_report(args) -> None:
    """The one data path: select jobs, then render at the chosen granularity."""
    # --plot_ts_overlay *is* --plot_ts with a different layout, so it is folded into it
    # here -- before _reclaim_jobid_after_ts, which is the first thing to read the value.
    # Seven places key on args.plot_ts (the window carriers, the --csv/--stats/--eff
    # guards, the --ts implication, and the narrowed help); folding means every one of
    # them applies to the overlay untouched, and only _plot_timeseries has to know.
    if args.plot_ts_overlay is not False:
        args.plot_ts = args.plot_ts_overlay
    _reclaim_jobid_after_ts(args)
    if args.plot_ts:
        # --plot_ts *is* --ts, with the CSV charted instead of written. Setting it here,
        # before anything reads it, means every --ts path applies unchanged: the schema,
        # --step, and the --nodename guard just below.
        if args.csv:
            raise JobscopeError("%s draws a chart; drop --csv, or drop %s to keep the CSV"
                                % (_plot_flag(args), _plot_flag(args)))
        args.ts = True
    cfg = _apply_config(args)
    request = build_request(args, cfg)
    timeout = _timeout(args, cfg)
    workers = _workers(args, cfg)

    if args.eff and not args.ts:
        raise JobscopeError("--eff sorts a time series into efficiency categories; add "
                            "--ts (optionally with a window, e.g. --ts 10m)")
    if args.eff not in (False, True, "all"):
        raise JobscopeError("--eff takes no value, or 'all' to list the "
                            "'good' jobs too; got %r" % (args.eff,))
    if args.stats and not args.ts:
        raise JobscopeError("--stats summarizes a time series; add --ts (optionally with "
                            "a window, e.g. --ts 1h)")
    if args.stats and args.plot_ts:
        print("note: the chart already prints min/mean/max/last; ignoring --stats",
              file=sys.stderr)
    if args.eff and args.plot_ts:
        # The plot branch wins below, so say so rather than drop it silently.
        print("note: %s draws the series; ignoring --eff" % _plot_flag(args),
              file=sys.stderr)
    # --nodename and --gpuid apply everywhere now. On --per-gpu and --ts they filter
    # rows, which have a node and a GPU on them; on the summary there is no row to
    # filter, so select.py narrows the numbers the summary is computed *from*. Both
    # end up describing the same subset.
    view = args.view or "all"
    show_dcgm = view in ("all", "gpu")
    # Which metrics each view collects, from [metrics] -- the built-in lists when a
    # site has not said otherwise. --per-gpu is the exception and does not widen under
    # --all-metrics: report.detail_columns() derives its profiling block from the
    # resolved headers instead, so the block follows the source preference (which
    # changes how many columns still need querying) without following this list.
    specs = list(cfg.metrics.extended if args.all_metrics else cfg.metrics.summary)
    # --ts's own view resolution: combined (GPU + CPU%/MEM% together) is the
    # default -- bare --ts behaves as --cpu --all-metrics --ts would. --cpu alone (no
    # --all-metrics) narrows to CPU-only; --all-metrics alone (no --cpu) narrows to
    # GPU-only, unchanged from before this feature. --gpu has no say in this -- it
    # stays the inert flag it already was under --ts (see the notes below). Whenever
    # the mode is not cpu-only, the GPU catalog is [metrics] timeseries (a small
    # curated set by default), widening to [metrics] extended only when --all-metrics
    # was actually passed -- printing/plotting everything is opt-in, not the default.
    ts_cpu_only = (view == "cpu" and not args.all_metrics)
    ts_combined = not ts_cpu_only and not (view != "cpu" and args.all_metrics)
    ts_specs = specs if ts_cpu_only else list(
        cfg.metrics.extended if args.all_metrics else cfg.metrics.timeseries)
    options = RenderOptions(
        view=view, show_dcgm=show_dcgm, csv=args.csv, header=args.header,
        # time_weighted is left at its default here and set below, once resolve() has
        # both the records and the cap: no time-series path reads it, and neither the mode
        # nor the flag alone can answer it.
        plot_avgeff=not args.no_plot,
        nodename=args.nodename, gpu_ids=tuple(plot.gpu_list(args.gpuid)) if args.gpuid else (),
        window=_ts_window(args),
        color=_want_color(args), combined=ts_combined,
        worst_jobs=cfg.defaults.worst_jobs,
        long_running=parse_duration(cfg.defaults.long_running),
        sections=cfg.report.sections,
        # The host side of [metrics], the counterpart of `specs` above. Previously
        # always None, so CPU%/MEM% were fixed whatever the config said and a
        # [metrics.cgroup.<name>] definition could be defined but never selected.
        #
        # None under --gpu, mirroring `specs if show_dcgm else None` for the other side:
        # that view prints no host column, so there is none to attribute a source to and
        # none to warn about being blank.
        cgroup_specs=(list(cfg.metrics.host_extended if args.all_metrics
                           else cfg.metrics.host_summary)
                      if view in ("all", "cpu") else None),
        # The two views have their own band tables and inherit nothing from each
        # other, because a two-hour window that catches a checkpoint pause should
        # not answer to a nineteen-hour job's bar. --plot_ts set args.ts above, so
        # this one test covers every time-series path.
        thresholds=(cfg.timeslice_thresholds if (args.ts or args.verify)
                    else cfg.thresholds))

    if args.verify:
        # It fetches every scrape of each job's series, which is fine for the job you are
        # about to act on and not for a partition. Named rather than silently capped: a
        # truncated answer is the one thing this view must not give.
        if request.mode != JOBIDS:
            raise JobscopeError(
                "--verify checks the jobs you name; use -j JOBID (or give the IDs "
                "directly).\n"
                "It fetches every scrape of each job's series, which a window or "
                "partition selection should not do.")
        buffer = io.StringIO()
        emit_timeseries(request, cfg, timeout, workers, ts_specs, args.step, options,
                        out=buffer)
        _verify_series(buffer.getvalue(), options, cfg)
        return

    if args.ts:
        # --gpu still does not apply to --ts; --cpu/--dcgm are resolved above into
        # ts_cpu_only/ts_combined, so neither is a dropped flag any more (--cpu
        # --dcgm together, or neither, now both mean combined).
        if view == "gpu":
            print("note: --gpu does not apply to --ts (a per-scrape metric series)",
                  file=sys.stderr)
        if not (args.plot_ts or args.stats or args.eff):
            emit_timeseries(request, cfg, timeout, workers, ts_specs, args.step, options)
            return
        buffer = io.StringIO()
        emit_timeseries(request, cfg, timeout, workers, ts_specs, args.step, options,
                        out=buffer)
        if args.plot_ts:
            _plot_timeseries(buffer.getvalue(), args)
        elif args.eff:
            # The unit of "which jobs are idle" is the job, so that is the default
            # level; --stats node grades hosts on the same rule.
            _eff_timeseries(buffer.getvalue(), options, args.stats or "job",
                            args.eff == "all")
        else:
            _stats_timeseries(buffer.getvalue(), options, args.stats)
        return

    # The detail granularity derives its own columns from the resolved headers, so it
    # takes no spec list; the per-job one splices the profiling block from whichever
    # was chosen. Both still need `specs` collected -- that is what fills the cells.
    selected = resolve(request, cfg, timeout, workers, specs if show_dcgm else None,
                       nodename=args.nodename, gpu_ids=options.gpu_ids,
                       host_specs=options.cgroup_specs)
    if selected is None:
        return
    # Now, not before: whether the values span whole runtimes is a fact about the records,
    # and the mode is only a proxy for it. `jobscope -j ID` on a job that is still running
    # took the finished branch of `mode != RUNNING` and weighted a single scrape by hours
    # of elapsed time -- the combination RenderOptions.time_weighted documents as invalid,
    # and the one its Sampled line had been contradicting all along.
    options = dataclasses.replace(
        options, time_weighted=selected.folded)
    # JOB_LEVEL keeps the per-job table; the two detail levels share one renderer and
    # differ only in their identity prefix -- see report.detail_columns.
    level = (NODE_LEVEL if args.per_node else GPU_LEVEL if args.per_gpu else JOB_LEVEL)
    # `total` and `specs` reach the detail views so a multi-job listing can end in one
    # aggregate summary rather than a chart per job -- see DetailRenderer.
    renderer = (SummaryRenderer(selected.context, options, specs=specs)
                if level == JOB_LEVEL
                else DetailRenderer(selected.context, options, level=level, specs=specs,
                                    total=selected.total))
    for chunk_ids, records, dcgm_data in selected.chunks:
        # `level` travels with the call so a summary sweep does not build the per-unit
        # rows only a detail view reads -- see jobscope.rows.
        renderer.add(rows.build_rows(chunk_ids, records, dcgm_data, level))
    renderer.finish()


def handle_plot(args) -> None:
    _apply_config(args)
    plot.run(args)


def handle_describe(args) -> None:
    cfg = _apply_config(args)
    # --all-metrics implies --metrics: it means "the full catalog" on a report, and it
    # used to mean nothing at all here without --dcgm beside it. One word, one meaning.
    if args.metrics or args.all_metrics:
        wide = cfg.metrics.extended
        describe_dcgm(list(wide if args.all_metrics else cfg.metrics.summary),
                      extended=wide)
    else:
        describe()


def handle_probe(args) -> None:
    # --init is the one command whose named config file is *expected* to be absent --
    # that is the file it is about to write. Everywhere else a missing -c path is a
    # typo and must stay an error, so the tolerance is scoped to this flag.
    if args.init:
        try:
            cfg = _apply_config(args)
        except JobscopeError:
            # Fall back to built-in defaults, dropping only $JOBSCOPE_CONFIG -- the
            # env copy keeps $JOBSCOPE_PROM_URL, which is how --init finds the
            # endpoint it is about to describe. A bare load_config() would consult
            # the same missing path and raise again.
            env = {k: v for k, v in os.environ.items() if k != config.CONFIG_ENV}
            config.set_config(config.load_config(env=env))
            cfg = config.get_config()
    else:
        cfg = _apply_config(args)
    _reclaim_probe_target(args)
    # Each flag is None when absent, "" when given bare, and the job ID when given
    # one -- so the bare form means "pick a recent job for me".
    status = probe.run(sys.stdout, cfg, args.config_path, cfg.defaults.timeout,
                        metrics=args.metrics is not None,
                        toml=args.toml is not None,
                        init=args.init, full=args.full,
                        coverage=args.coverage,
                        jobid=args.metrics or args.toml or None)
    if status:
        raise SystemExit(status)


_PROBE_TAKES_A_TARGET = (("--coverage", "coverage"), ("--metrics", "metrics"),
                         ("--toml", "toml"))


def _reclaim_probe_target(args) -> None:
    """Give a trailing bare word to whichever probe check was written without one.

    ``--coverage kempner --full`` already worked, because argparse consumes the next
    word as the flag's optional value. ``--coverage --full kempner`` did not: the flag
    saw ``--full`` and took no value, leaving ``kempner`` unrecognised. Both orderings
    now mean the same thing.

    Unambiguous rather than a guess, because it is settled by *how many* checks were
    given bare. Exactly one, and the word belongs to it. None, and there is nothing it
    could modify. More than one, and the two readings are equally good -- so say so and
    name them instead of picking.

    The counterpart of :func:`_reclaim_jobid_after_ts`, which solves the opposite
    problem: there argparse consumed a value it should not have.
    """
    target = getattr(args, "target", None)
    if not target:
        return
    bare = [flag for flag, dest in _PROBE_TAKES_A_TARGET if getattr(args, dest, None) == ""]
    if len(bare) == 1:
        setattr(args, dict(_PROBE_TAKES_A_TARGET)[bare[0]], target)
        args.target = None
        return
    if not bare:
        raise JobscopeError(
            "%r names nothing on its own. Add the check it belongs to, e.g.\n"
            "  jobscope probe --coverage %s      # that partition's host coverage\n"
            "  jobscope probe --metrics %s       # what the server carries for that job"
            % (target, target, target))
    raise JobscopeError(
        "%r could belong to %s -- write it beside the one you mean, e.g. '%s %s'"
        % (target, " or ".join(bare), bare[0], target))


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
    cfg = config.load_config(str(path)) if exists else config.load_config()
    _print_endpoint(cfg)
    for view, bands in (("summary", cfg.thresholds),
                        ("timeslice", cfg.timeslice_thresholds)):
        print()
        _print_bands(view, bands)
    print()
    print("[metrics]  (GPU/DCGM only; CPU%/MEM% are fixed)")
    for view in ("summary", "timeseries", "extended"):
        specs = getattr(cfg.metrics, view)
        print("  %-11s %s" % (view, " ".join(s.header for s in specs)))
    print("  --per-gpu keeps its own fixed four: %s"
          % " ".join(dcgm.catalog().headers))
    print()
    print("[colors]")
    for role in config.COLOR_ROLES:
        print("  %-13s %s" % (role, cfg.palette.colors[role]))
    print()
    print("[defaults]")
    for key, value in sorted(vars(cfg.defaults).items()):
        print("  %-13s %s" % (key, value))
    print()
    print("[report]")
    print("  %-13s %s" % ("sections", " ".join(cfg.report.sections)))
    print()
    print("[plot]")
    # Resolved, so a short name in the file shows as the column header it charts --
    # which is what a CSV has to carry for the series to appear at all.
    print("  %-13s %s" % ("metrics", " ".join(cfg.plot.metrics)))
    print("  %-13s %s" % ("palette", " ".join(str(c) for c in cfg.plot.palette)))
    print("  %-13s %s" % ("max_rows", cfg.plot.max_rows))
    print("  %-13s %s" % ("panels", cfg.plot.panels))
    print()
    print("print an example with: jobscope config --example")


def _print_endpoint(cfg) -> None:
    """The endpoint, redacted, and which of the three sources supplied it.

    First because it is the one setting that has to be right before anything works,
    and it was the one thing this command did not show -- you could read every band
    table and still not know which server the numbers would come from.

    Always through redact_url: the URL commonly embeds a Grafana Cloud token, so it
    is a secret that happens to look like an address.
    """
    try:
        url, scrape = config.resolve_prometheus(cfg)
    except JobscopeError as exc:
        print("prometheus:  %s" % str(exc).splitlines()[0])
        print("             run 'jobscope probe' for the three ways to set it")
        return
    print("prometheus:  %s" % config.redact_url(url))
    source = config.endpoint_source(cfg)
    print("             %sscrape %ds"
          % ("from %s, " % source if source else "", scrape))


def _print_bands(view: str, bands) -> None:
    """One view's resolved band table, as a grid of edges by metric.

    Resolved, not as written: the two views inherit nothing from each other and a
    metric inherits the default for any edge it does not name, so what a site wrote
    and what it is graded by are two different things. This is the one place to see
    the second.
    """
    print("[thresholds.%s]" % view)
    rows = [("default", bands.edges())]
    rows += [(header, bands.edges(header)) for header in sorted(bands.by_metric)]
    name_width = max([len(name) for name, _edges in rows] + [len("metric")])
    widths = [max(len(key), max(len("%g" % r[i]) for _n, r in rows))
              for i, key in enumerate(config.EDGE_KEYS)]

    def line(name, cells):
        return "  %-*s  %s" % (name_width, name,
                               "  ".join(c.rjust(widths[i]) for i, c in enumerate(cells)))

    print(line("metric", config.EDGE_KEYS))
    for name, edges in rows:
        print(line(name, ["%g" % edge for edge in edges]))
    print("  POWER_W is a watt floor, not a band: red below %g W, green above."
          % bands.power_w)
    if bands.power_w_by_model:
        for model, floor in sorted(bands.power_w_by_model.items()):
            print("    %s: %g W" % (model, floor))


def main(argv=None) -> None:
    raw = sys.argv[1:] if argv is None else list(argv)
    explicit = mode_was_explicit(raw)
    argv = resolve_argv(raw)
    parser, subparsers = build_parser()
    # Parsing is inside the guard as well as handling: a retired flag raises from its
    # own argparse action (see _Retired), which is during parse, and a traceback is a
    # poor way to say "that flag has a new name".
    try:
        if argv[0] in subparsers.choices:
            # Parse on the chosen subparser so JOBIDs and flags may appear in any
            # order. parse_intermixed_args cannot run on the top parser, where the
            # mode is itself a positional.
            sub = subparsers.choices[argv[0]]
            if any(a in ("-h", "--help") for a in argv[1:]):
                # Before parsing, because argparse prints the help and exits the moment it
                # reaches -h. Narrowing is modes-only (it reasons about a report's flags);
                # shortening applies to every subcommand that has anything to shorten.
                if argv[0] in MODES:
                    narrow_help(sub, argv[1:], explicit)
                brief_help(sub)
            args = sub.parse_intermixed_args(argv[1:])
            args.explicit_mode = explicit
        else:  # -h / --help / --version
            args = parser.parse_args(argv)
    except JobscopeError as exc:
        print("jobscope: error: %s" % exc, file=sys.stderr)
        sys.exit(2)  # argparse's own exit code for a usage error
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
