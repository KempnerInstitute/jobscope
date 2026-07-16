"""Command-line interface: a single ``jobscope`` command with subcommands.

Subcommands: ``summary`` (default), ``detail``, ``dcgm``, ``plot``, ``describe``,
and ``config``. The data views share a common set of job selectors; ``summary``
is assumed when no subcommand is given, so ``jobscope -D 3`` still works.
"""

import argparse
import os
import sys

from . import __version__, config, plot
from .dcgm import ALL_SPECS, DEFAULT_SPECS, GPU_SUMMARY_SPECS, compute_dcgm
from .errors import JobscopeError
from .prometheus import client_from_config
from .report import (
    RenderOptions,
    context_pairs,
    dcgm_report,
    dcgm_timeseries,
    describe,
    describe_dcgm,
    detail,
    summarize,
)
from .sacct import Selection, days_to_window, default_user, end_of_day, fetch, select_jobs

SUBCOMMANDS = ("summary", "detail", "dcgm", "plot", "describe", "config")

_SUMMARY_DESC = (
    "Compact per-job utilization summary. CPU/MEM/GPU/GMEM come from the sacct\n"
    "AdminComment blob (offline); the gpu view also pulls time-averaged DCGM\n"
    "profiling columns (SM_ACT%/OCC%/TENSOR%/DRAM%/POWER_W) from Prometheus.")

_SUMMARY_EPILOG = (
    "examples:\n"
    "  jobscope 30012345              one job by ID\n"
    "  jobscope -j 30012345           the same job, using the -j flag\n"
    "  jobscope -D 3                  your jobs from the last 3 days\n"
    "  jobscope -N 20 --cpu           last 20 jobs, CPU columns (offline)\n"
    "  jobscope -u alice -D 7 --csv | jobscope plot\n"
    "\n"
    "Flags and JOBIDs may be given in any order.")


def build_parser():
    """Construct the argument parser; return ``(parser, subparsers_action)``.

    The subparsers action is returned so callers can parse on a specific
    subparser (``subparsers.choices[name]``) with ``parse_intermixed_args`` --
    which the top parser cannot do, since the subcommand is itself a positional.
    """
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("-c", "--config", dest="config_path", metavar="PATH",
                      help="path to a jobscope config file (overrides $JOBSCOPE_CONFIG)")

    selector = argparse.ArgumentParser(add_help=False)
    sel_scope = selector.add_argument_group("job selection")
    sel_scope.add_argument("jobids", nargs="*", metavar="JOBID",
                           help="specific job IDs (bypass time selection)")
    sel_scope.add_argument("-j", "--jobid", action="append", dest="jobids_opt", metavar="JOBID",
                           help="a job ID (repeatable; alternative to the positional JOBID)")
    sel_scope.add_argument("-N", "--lastn", type=int, metavar="N",
                           help="the most recent N jobs")
    sel_scope.add_argument("-D", "--days", type=int, metavar="N",
                           help="jobs in the last N days")
    sel_scope.add_argument("-S", "--starttime", metavar="TIME",
                           help="window start, e.g. 2026-07-15 or 2026-07-15T09:00:00 "
                                "(sacct format); without -E, selects just that day")
    sel_scope.add_argument("-E", "--endtime", metavar="TIME",
                           help="window end, same format as -S "
                                "(default: end of the -S day, else now)")

    sel_filter = selector.add_argument_group("filters")
    sel_filter.add_argument("-u", "--user", help="user (default: current user, $USER)")
    sel_filter.add_argument("-A", "--account", help="narrow to this account")
    sel_filter.add_argument("-p", "--partition", help="narrow to this partition")
    sel_filter.add_argument("-t", "--state", choices=["all", "completed", "failed"], default="all",
                            help="job state filter (default: all)")

    sel_output = selector.add_argument_group("output")
    sel_output.add_argument("-n", "--noheader", dest="header", action="store_false",
                            help="suppress the header/context block")
    sel_output.add_argument("--csv", action="store_true",
                            help="machine-readable output (pipe to 'jobscope plot')")
    sel_output.add_argument("--timeout", type=float, default=None,
                            help="seconds per sacct/Prometheus call (default from config; 0 disables)")
    sel_output.add_argument("--workers", type=int, default=None,
                            help="max concurrent Prometheus query-sets for the gpu view "
                                 "(default from config; the queries are I/O-bound so >1 helps)")

    parser = argparse.ArgumentParser(
        prog="jobscope",
        description="Slurm job efficiency and GPU utilization reporting.")
    parser.add_argument("--version", action="version", version="jobscope %s" % __version__)
    subparsers = parser.add_subparsers(dest="command")

    p_summary = subparsers.add_parser(
        "summary", parents=[base, selector], description=_SUMMARY_DESC,
        epilog=_SUMMARY_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="one row per job (default subcommand)")
    _add_view_options(p_summary)
    p_summary.set_defaults(func=handle_summary)

    p_detail = subparsers.add_parser(
        "detail", parents=[base, selector],
        help="per-node / per-GPU breakdown for each job")
    _add_view_options(p_detail)
    p_detail.set_defaults(func=handle_detail)

    p_dcgm = subparsers.add_parser(
        "dcgm", parents=[base, selector],
        help="per-GPU DCGM profiling table (one row per GPU)")
    p_dcgm.add_argument("--ext", "--extended", dest="ext", action="store_true",
                        help="show the full 28-metric catalog (clocks, temps, PCIe, NVLink, ...)")
    p_dcgm.add_argument("--ts", "--timeseries", dest="ts", action="store_true",
                        help="emit the raw per-scrape time series for one job as CSV "
                             "(pass exactly one JOBID)")
    p_dcgm.set_defaults(func=handle_dcgm)

    p_plot = subparsers.add_parser(
        "plot", parents=[base],
        help="render 'jobscope <view> --csv' output as a terminal chart")
    plot.add_arguments(p_plot)
    p_plot.set_defaults(func=handle_plot)

    p_describe = subparsers.add_parser(
        "describe", parents=[base], help="describe the columns and metrics")
    p_describe.add_argument("--dcgm", action="store_true",
                            help="describe the DCGM metric catalog instead of the summary columns")
    p_describe.add_argument("--ext", "--extended", dest="ext", action="store_true",
                            help="with --dcgm, describe all 28 metrics")
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


def _add_view_options(subparser: argparse.ArgumentParser) -> None:
    group = subparser.add_mutually_exclusive_group()
    group.add_argument("--cpu", action="store_const", const="cpu", dest="metrics",
                       help="CPU columns only (offline)")
    group.add_argument("--gpu", action="store_const", const="gpu", dest="metrics",
                       help="GPU columns: blob GPU%%/GMEM%% plus DCGM profiling from Prometheus (default)")
    group.add_argument("--cgpu", action="store_const", const="cgpu", dest="metrics",
                       help="both CPU and GPU blob columns (offline; no Prometheus)")
    subparser.add_argument("--diagnose", action="store_true",
                           help="add an advisory DIAG column (gpu view only)")
    subparser.add_argument("--min-runtime", dest="min_runtime", type=int, default=None,
                           help="jobs shorter than this many seconds get DIAG=short "
                                "(default from config)")


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


def _min_runtime(args, cfg: config.Config) -> int:
    value = getattr(args, "min_runtime", None)
    return value if value is not None else cfg.defaults.min_runtime


def _prepare_selection(args) -> Selection:
    user = args.user or default_user()
    if not user:
        raise JobscopeError("could not determine the current user from $USER; pass -u/--user")
    days, lastn = args.days, args.lastn
    start, end = args.starttime, args.endtime
    jobids = list(args.jobids) + list(getattr(args, "jobids_opt", None) or [])
    if jobids:
        # Explicit JOBIDs win; time selectors do not apply. Warn rather than
        # silently drop them, then proceed with the given IDs.
        ignored = [name for name, on in (
            ("-D/--days", days is not None), ("-N/--lastn", lastn is not None),
            ("-S/--starttime", bool(start)), ("-E/--endtime", bool(end)),
        ) if on]
        if ignored:
            print("jobscope: note: explicit JOBIDs given; ignoring time selectors (%s)"
                  % ", ".join(ignored), file=sys.stderr)
        return Selection(user=user, jobids=jobids, account=args.account,
                         partition=args.partition, state=args.state)
    if lastn is None and days is None and not start and not end:
        days = 1
    if days is not None:
        if days <= 0:
            raise JobscopeError("-D/--days must be a positive integer")
        if lastn is not None:
            raise JobscopeError("-D/--days cannot be combined with -N/--lastn")
        if start or end:
            raise JobscopeError("-D/--days sets the window; do not also pass -S/-E")
        start, end = days_to_window(days)
    elif start and not end:
        end = end_of_day(start)  # -S alone selects just that calendar day
    if lastn is not None and lastn <= 0:
        raise JobscopeError("-N/--lastn must be a positive integer")
    return Selection(user=user, jobids=jobids, account=args.account, partition=args.partition,
                     state=args.state, lastn=lastn, days=days, starttime=start, endtime=end)


def _no_jobs(selection: Selection, desc: str) -> None:
    print("No matching jobs for user '%s' (%s)." % (selection.user, desc), file=sys.stderr)


def _view_common(args):
    """Shared setup for the summary and detail views.

    Returns ``(jobids, records, dcgm_data, context, options)`` or None when the
    selection is empty.
    """
    cfg = _apply_config(args)
    selection = _prepare_selection(args)
    timeout = _timeout(args, cfg)
    workers = _workers(args, cfg)
    view = args.metrics or "gpu"
    diagnose = args.diagnose
    if view != "gpu" and diagnose:
        print("note: --diagnose applies only to the gpu view; ignoring", file=sys.stderr)
        diagnose = False
    show_dcgm = view == "gpu"

    jobids, desc = select_jobs(selection, timeout)
    if not jobids:
        _no_jobs(selection, desc)
        return None
    records = fetch(jobids, timeout)

    dcgm_data = {}
    if show_dcgm:
        gpu_jobs = [j for j in jobids if j in records and records[j].gpus]
        if gpu_jobs:
            client = client_from_config(cfg, timeout)
            dcgm_data = compute_dcgm(records, jobids, GPU_SUMMARY_SPECS, client, timeout, workers)

    context = context_pairs(selection, desc, records)
    options = RenderOptions(view=view, show_dcgm=show_dcgm, diagnose=diagnose,
                            csv=args.csv, header=args.header, min_runtime=_min_runtime(args, cfg))
    return jobids, records, dcgm_data, context, options


def handle_summary(args) -> None:
    result = _view_common(args)
    if result is not None:
        summarize(*result)


def handle_detail(args) -> None:
    result = _view_common(args)
    if result is not None:
        detail(*result)


def handle_dcgm(args) -> None:
    cfg = _apply_config(args)
    selection = _prepare_selection(args)
    timeout = _timeout(args, cfg)
    workers = _workers(args, cfg)
    if args.ts and len(selection.jobids) != 1:
        raise JobscopeError("dcgm --ts profiles one job at a time: pass exactly one JOBID")
    specs = ALL_SPECS if args.ext else DEFAULT_SPECS

    jobids, desc = select_jobs(selection, timeout)
    if not jobids:
        _no_jobs(selection, desc)
        return
    records = fetch(jobids, timeout)
    options = RenderOptions(view="gpu", show_dcgm=True, diagnose=False,
                            csv=args.csv, header=args.header, min_runtime=cfg.defaults.min_runtime)

    if args.ts:
        client = client_from_config(cfg, timeout)
        dcgm_timeseries(jobids, records, specs, client, timeout, options)
        return

    dcgm_data = {}
    gpu_jobs = [j for j in jobids if j in records and records[j].gpus]
    if gpu_jobs:
        client = client_from_config(cfg, timeout)
        dcgm_data = compute_dcgm(records, jobids, specs, client, timeout, workers)
    context = context_pairs(selection, desc, records)
    dcgm_report(jobids, records, dcgm_data, specs, context, options)


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


def _inject_default_subcommand(argv):
    """Prefix 'summary' when no subcommand is given, so ``jobscope -D 3`` works."""
    if not argv:
        return ["summary"]
    if argv[0] in ("-h", "--help", "--version") or argv[0] in SUBCOMMANDS:
        return argv
    return ["summary"] + argv


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    argv = _inject_default_subcommand(argv)
    parser, subparsers = build_parser()
    if argv[0] in subparsers.choices:
        # Parse on the chosen subparser so JOBIDs and flags may appear in any
        # order. parse_intermixed_args can't run on the top parser, where the
        # subcommand is itself a positional.
        args = subparsers.choices[argv[0]].parse_intermixed_args(argv[1:])
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
