"""Command-line interface: a single ``jobscope`` command with subcommands.

Subcommands: ``summary`` (default), ``detail``, ``dcgm``, ``live``, ``plot``,
``describe``, and ``config``. The historical views share a common set of sacct job
selectors; ``summary`` is assumed when no subcommand is given, so ``jobscope -D 3``
still works. ``live`` selects from ``squeue`` instead and so takes its own, smaller
set of selectors.
"""

import argparse
import os
import sys

from . import __version__, config, plot
from .dcgm import ALL_SPECS, DEFAULT_SPECS, GPU_SUMMARY_SPECS, compute_dcgm
from .errors import JobscopeError
from .live import (
    DEFAULT_MIN_ELAPSED,
    EXTENDED_LIVE_SPECS,
    LiveSelection,
    build_columns,
    collect_averaged,
    collect_instant,
    collect_timeseries,
    discover_gpus,
    fetch_jobs,
    parse_duration,
    specs_for,
)
from .live_blob import fill_running, note_offline_gap
from .prometheus import client_from_config
from .report import (
    DetailRenderer,
    RenderOptions,
    SummaryRenderer,
    context_pairs,
    dcgm_report,
    dcgm_timeseries,
    describe,
    describe_dcgm,
    describe_live,
    live_report,
    live_timeseries,
)
from .sacct import (
    Selection,
    days_to_window,
    default_user,
    end_of_day,
    fetch,
    fetch_chunks,
    select_jobs,
)

SUBCOMMANDS = ("summary", "detail", "dcgm", "live", "plot", "describe", "config")

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
    "A JOBID works whether the job is running or finished. Running jobs have no\n"
    "stored utilization blob yet, so the gpu view reconstructs CPU%/MEM%/GPU%/GMEM%\n"
    "from Prometheus; the offline --cpu/--cgpu views leave them blank.\n"
    "\n"
    "Flags and JOBIDs may be given in any order.")

_LIVE_DESC = (
    "Per-GPU metrics for the jobs running right now: squeue for selection,\n"
    "Prometheus for the numbers. One row per GPU (per MIG instance where used).\n"
    "\n"
    "By default every value is the newest single scrape, which will NOT agree with\n"
    "jobstats on a bursty job -- jobstats folds over the whole runtime. Pass --avg to\n"
    "do the same, averaging utilization and peaking memory as it does.")

_LIVE_EPILOG = (
    "examples:\n"
    "  jobscope live                        your running jobs, over 1h\n"
    "  jobscope live -j 30012345_6          one running job or array element\n"
    "  jobscope live -p kempner -a          every user's jobs in a partition\n"
    "  jobscope live --min-elapsed 5m       include jobs only 5 minutes in\n"
    "  jobscope live --avg                  runtime-folded (= jobstats)\n"
    "  jobscope live --describe             what each column means\n"
    "  jobscope live -j 30012345 --ts | jobscope plot\n"
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

    p_live = subparsers.add_parser(
        "live", parents=[base], description=_LIVE_DESC, epilog=_LIVE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="per-GPU metrics for the jobs running right now")
    live_scope = p_live.add_argument_group("job selection")
    live_scope.add_argument("jobids", nargs="*", metavar="JOBID",
                           help="specific running job IDs or array elements "
                                "(bypass the filters and the runtime floor)")
    live_scope.add_argument("-j", "--jobid", action="append", dest="jobids_opt", metavar="JOBID",
                            help="a running job ID (repeatable; alternative to the positional)")
    live_scope.add_argument("-p", "--partition", help="narrow to this partition")
    live_scope.add_argument("-u", "--user", help="user (default: current user, $USER)")
    live_scope.add_argument("-a", "--all-users", dest="all_users", action="store_true",
                            help="every user's jobs, not just your own")
    live_scope.add_argument("--min-elapsed", "--min-runtime", dest="min_elapsed",
                            metavar="DURATION", default=DEFAULT_MIN_ELAPSED,
                            help="only jobs running longer than this (default: %s; "
                                 "e.g. '5m', '2h', '0s' for no floor)" % DEFAULT_MIN_ELAPSED)
    live_cols = p_live.add_argument_group("columns")
    live_cols.add_argument("--all", "--ext", action="store_const", const="all", dest="view",
                           help="the extended DCGM catalog (clocks, temps, PCIe, NVLink, ...)")
    live_cols.add_argument("--avg", action="store_true",
                           help="fold each metric over each job's runtime, making the values "
                                "comparable to jobstats (default: an instantaneous snapshot)")
    live_cols.add_argument("--describe", action="store_true",
                           help="explain the columns and their source metrics, then exit")
    live_out = p_live.add_argument_group("output")
    live_out.add_argument("--ts", "--timeseries", dest="ts", action="store_true",
                          help="emit the per-scrape time series as CSV instead of a table; "
                               "same schema as 'jobscope dcgm --ts', so it pipes to "
                               "'jobscope plot'. Ignores --avg")
    live_out.add_argument("--step", type=int, default=None, metavar="SECONDS",
                          help="--ts sample interval (default: the scrape interval, widened on "
                               "long jobs to stay under Prometheus' point cap)")
    live_out.add_argument("-n", "--noheader", dest="header", action="store_false",
                          help="suppress the header/context block")
    live_out.add_argument("--csv", action="store_true",
                          help="machine-readable output")
    live_out.add_argument("--timeout", type=float, default=None,
                          help="seconds per squeue/Prometheus call (default from config; "
                               "0 disables)")
    live_out.add_argument("--workers", type=int, default=None,
                          help="max concurrent Prometheus queries for --avg/--ts "
                               "(default from config)")
    p_live.set_defaults(func=handle_live, view=None)

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


def _view_common(args, renderer_cls) -> None:
    """Fetch and render the summary/detail views batch by batch.

    Time-window selections stream: each sacct batch is fetched, its DCGM
    metrics computed, and its rows rendered before the next batch is queried,
    so large selections show results as they arrive. A mid-stream error (sacct
    timeout, missing Prometheus endpoint) can therefore surface after a partial
    table. Explicit-JOBID selections render in one pass, since their context
    header lists the owners of every record.
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
        return
    options = RenderOptions(view=view, show_dcgm=show_dcgm, diagnose=diagnose,
                            csv=args.csv, header=args.header, min_runtime=_min_runtime(args, cfg))

    if selection.jobids:
        records = fetch(jobids, timeout)
        context = context_pairs(selection, desc, records)
        chunks = iter([(jobids, records)])
    else:
        context = context_pairs(selection, desc, {})  # window branch never reads records
        chunks = fetch_chunks(jobids, timeout)

    renderer = renderer_cls(context, options)
    client = None
    for chunk_ids, records in chunks:
        dcgm_chunk = {}
        if show_dcgm and any(j in records and records[j].gpus for j in chunk_ids):
            if client is None:
                client = client_from_config(cfg, timeout)
            dcgm_chunk = compute_dcgm(records, chunk_ids, GPU_SUMMARY_SPECS,
                                      client, timeout, workers)
        client = _fill_running_blobs(records, chunk_ids, cfg, timeout, workers, client)
        renderer.add(chunk_ids, records, dcgm_chunk)
    renderer.finish()


def _fill_running_blobs(records, jobids, cfg, timeout, workers, client):
    """Rebuild the utilization blob for running jobs; return the client used.

    A running job has no stored blob yet, so CPU%/MEM%/GPU%/GMEM% would all be
    empty. Every input is in Prometheus, so fill them from there -- including for
    the otherwise-offline --cpu/--cgpu views, since those own the CPU%/MEM% columns
    and no other view would show them. An install with no endpoint configured stays
    fully offline: the fill is skipped with a note rather than an error.
    """
    if not any(j in records and records[j].state == "RUNNING" and not records[j].stats
               for j in jobids):
        return client
    if client is None:
        try:
            client = client_from_config(cfg, timeout)
        except JobscopeError:
            note_offline_gap(records, jobids)
            return None
    fill_running(records, jobids, client, timeout, workers)
    return client


def handle_summary(args) -> None:
    _view_common(args, SummaryRenderer)


def handle_detail(args) -> None:
    _view_common(args, DetailRenderer)


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
        # Fill the blob for running jobs, so the DUR_S/state context and any blob
        # column in this view read the same as they do for a finished job.
        fill_running(records, jobids, client, timeout, workers)
    context = context_pairs(selection, desc, records)
    dcgm_report(jobids, records, dcgm_data, specs, context, options)


def _live_selection(args) -> LiveSelection:
    """Build the squeue-side selection, validating the runtime floor."""
    jobids = list(args.jobids) + list(getattr(args, "jobids_opt", None) or [])
    if jobids and (args.partition or args.user or args.all_users):
        print("jobscope: note: explicit JOBIDs given; ignoring the -p/-u/-a filters",
              file=sys.stderr)
    if args.all_users and args.user:
        raise JobscopeError("-a/--all-users and -u/--user are mutually exclusive")
    # Validated even when no jobs match, so a typo is never silently ignored.
    min_elapsed = parse_duration(args.min_elapsed)
    user = None
    if not jobids and not args.all_users:
        user = args.user or default_user()
        if not user:
            raise JobscopeError(
                "could not determine the current user from $USER; pass -u/--user or -a/--all-users")
    return LiveSelection(jobids=jobids, partition=args.partition, user=user,
                         min_elapsed=min_elapsed)


def _live_context(selection: LiveSelection, jobs, gpus) -> list:
    """Context lines for the live header block.

    With explicit JOBIDs the -u/-p filters are bypassed, so name the jobs' actual
    owners rather than a filter that was not applied -- as context_pairs does for
    the historical views.
    """
    if selection.jobids:
        owners = sorted({job["user"] for job in jobs.values() if job.get("user")})
        user = ", ".join(owners) if owners else "(explicit job IDs)"
    else:
        user = selection.user or "(all users)"
    pairs = [("User", user)]
    if selection.partition:
        pairs.append(("Partition", selection.partition))
    pairs.append(("Select", selection.describe()))
    pairs.append(("GPUs", "%d across %d job(s)" % (len(gpus), len(jobs))))
    return pairs


def handle_live(args) -> None:
    cfg = _apply_config(args)
    specs = specs_for(args.view)

    if args.describe:
        describe_live(specs, average=args.avg, n_all=len(build_columns(EXTENDED_LIVE_SPECS)))
        return

    timeout = _timeout(args, cfg)
    workers = _workers(args, cfg)
    selection = _live_selection(args)
    options = RenderOptions(view="gpu", show_dcgm=True, csv=args.csv, header=args.header)

    jobs = fetch_jobs(selection, timeout)
    if not jobs:
        print("No running jobs match (%s)." % selection.describe(), file=sys.stderr)
        return

    client = client_from_config(cfg, timeout)
    gpus = discover_gpus(client, jobs, timeout)
    if not gpus:
        # Not an error: a CPU-only selection legitimately has no GPUs.
        print("No GPU data in Prometheus for these jobs (CPU-only, or not yet scraped).",
              file=sys.stderr)

    if args.ts:
        if args.avg:
            print("note: --avg ignored with --ts (the CSV carries every sample)", file=sys.stderr)
        samples = collect_timeseries(client, jobs, gpus, specs, timeout, workers, args.step)
        live_timeseries(jobs, samples, gpus, specs, options)
        return

    metrics = (collect_averaged(client, jobs, gpus, specs, timeout, workers) if args.avg
               else collect_instant(client, gpus, specs, timeout))
    live_report(jobs, metrics, gpus, specs, _live_context(selection, jobs, gpus),
                options, average=args.avg)


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
