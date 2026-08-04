"""Collect the per-scrape samples a ``--ts`` view renders.

This is the query half of the time-series views. It was inside the renderers, which
meant :mod:`jobscope.report` held a Prometheus client and a report could not be
produced from data you already had -- so a test had to fake a server to check a
column's formatting, and nothing could re-render a series without re-fetching it.

Everything here returns :class:`JobSeries`, and the renderers take an iterable of
them. The finished-job collectors are **generators**, deliberately: a partition-wide
``--ts`` sweep holds one job's samples at a time, exactly as the interleaved version
did, so the split costs no memory. The running collectors are not, because they were
already batched -- one ``host_stats_many`` for the whole selection, then a thread pool
-- and there is nothing to stream.

``--nodename`` and ``--gpuid`` are resolved here rather than by the caller. They are
not only filters: dropping the other units' UUIDs *before* the queries is what makes a
4-node job cost a quarter of the range queries instead of fetching three nodes' samples
to throw them away. :class:`UnitFilter` carries the bookkeeping that makes a name
matching nothing an error naming what *did* run, rather than an empty report.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from .cpu import CgroupSpec, chosen_specs, host_series, host_stats, host_stats_many
from .dcgm import MetricSpec, applicable_derived, discover_gpus, values_by_key
from .errors import JobscopeError
from .prometheus import PrometheusClient
from .running import RunningJob, job_sort_key, range_window
from .slurm import JobRecord

# uuid -> (node, minor, model)
GpuIdentity = Dict[str, Tuple[str, str, str]]


@dataclass
class JobSeries:
    """One job's collected samples, in the shape the renderers walk.

    Both halves are present so that one record serves the GPU-only, host-only and
    combined views; a view simply leaves the half it does not draw empty.
    """

    jobid: str
    user: str
    # GPU side. `gpus` identifies each card; `gpu` is uuid -> stamp -> {header: value}.
    gpus: GpuIdentity = field(default_factory=dict)
    gpu: Dict[str, Dict[int, dict]] = field(default_factory=dict)
    # Host side. `host` is node -> stamp -> {header: value}; `hosts` is the node rows
    # a host-only view emits, which is not the same thing -- a node can resolve a
    # divisor and still have returned no samples.
    host: Dict[str, Dict[int, dict]] = field(default_factory=dict)
    hosts: Tuple[str, ...] = ()


class UnitFilter:
    """``--nodename`` / ``--gpuid`` bookkeeping, shared by every collector.

    Tracks which nodes and GPUs the selection touched so that :meth:`check` can name
    them. A filter that matched nothing raises there rather than returning nothing,
    because an empty report reads as an idle node rather than as a typo.

    Checked after the last job rather than per job, because "job 7 has no GPU 3" is
    not an error -- only "nothing in this selection has a GPU 3" is. A generator
    cannot raise that on its own final step without doing it from inside the caller's
    render loop, so the caller calls :meth:`check`.
    """

    def __init__(self, nodename: Optional[str] = None, gpu_ids=None):
        self.nodename = nodename
        self.gpu_ids = tuple(gpu_ids) if gpu_ids else ()
        self.seen: set = set()          # node names
        self.seen_gpus: set = set()     # gpu ids, as strings -- MIG is "0.1"
        self.hit_gpus: set = set()      # of gpu_ids, the ones something matched
        self.matched = False

    def __bool__(self) -> bool:
        """Whether any filter is set at all -- the collectors' cheap early-out."""
        return bool(self.nodename or self.gpu_ids)

    def keep(self, uuid_to):
        """``uuid_to`` narrowed to the wanted node and GPUs, recording what was seen.

        Both dimensions at once, so a job filtered out by node never contributes its
        GPU ids to the "available" list -- naming GPUs from a node the user excluded
        would be a confusing answer to "which GPUs are there".
        """
        if self.nodename:
            self.seen.update(node for node, _minor, _model in uuid_to.values())
            uuid_to = {u: nm for u, nm in uuid_to.items() if nm[0] == self.nodename}
            if not uuid_to:
                return uuid_to
            self.matched = True
        if self.gpu_ids:
            self.seen_gpus.update(str(minor) for _n, minor, _m in uuid_to.values())
            uuid_to = {u: nm for u, nm in uuid_to.items() if str(nm[1]) in self.gpu_ids}
            self.hit_gpus.update(str(minor) for _n, minor, _m in uuid_to.values())
        return uuid_to

    def check(self) -> None:
        """Raise if a filter was given and nothing in the selection matched it."""
        if self.nodename and not self.matched:
            raise JobscopeError(
                "no rows for node %r in this selection; it ran on: %s"
                % (self.nodename, ", ".join(sorted(self.seen)) or "(none)"))
        missing = [g for g in self.gpu_ids if g not in self.hit_gpus]
        if missing:
            # Every one that matched nothing, not just the first: with a list it is
            # the typo in the middle that is hard to spot. Same rule as plot.run's.
            raise JobscopeError(
                "no rows for GPU %s in this selection; it used: %s"
                % (", ".join(repr(g) for g in missing),
                   ", ".join(sorted(self.seen_gpus)) or "(none)"))


def cgroup_hosts(nodes: dict, nodename: Optional[str]):
    """``(divisors, row hosts)`` for a cgroup series over ``nodes``.

    ``divisors`` is the per-node summary dict itself -- each spec names the field that
    divides it -- and the row set is the hosts that resolved a *core* count. That
    second part is deliberate and unchanged: it is also the "did anything resolve"
    guard, so a node reporting memory but no cores yields no rows, exactly as it did
    before the catalog existed.
    """
    hosts = [host for host, node in nodes.items() if node.get("cpus")]
    if nodename is None:
        return nodes, hosts
    if nodename not in hosts:
        return nodes, []
    return {nodename: nodes[nodename]}, [nodename]


def _distinct(specs: List[MetricSpec]) -> List[MetricSpec]:
    """One spec per Prometheus metric -- two specs can share a series (POWER_W/PWRmax_W)."""
    seen, out = set(), []
    for spec in specs:
        if spec.metric not in seen:
            seen.add(spec.metric)
            out.append(spec)
    return out


def _gpu_samples(client: PrometheusClient, specs: List[MetricSpec], uuids,
                 start: int, end: int, span: int,
                 timeout: Optional[float]) -> Dict[str, Dict[int, dict]]:
    """Range-query every spec over ``uuids``, as uuid -> stamp -> {header: value}."""
    regex = "^(" + "|".join(uuids) + ")$"
    series: Dict[str, dict] = {uuid: {} for uuid in uuids}
    for spec in _distinct(specs):
        for result in client.query_range(
                '%s{%s=~"%s"}' % (spec.metric, spec.uuid_label, regex),
                start, end, span, timeout):
            metric = result["metric"]
            uuid = metric.get(spec.uuid_label) or metric.get("uuid") or metric.get("UUID")
            if uuid not in series:
                continue
            for stamp, value in result["values"]:
                try:
                    series[uuid].setdefault(int(stamp), {})[spec.header] = \
                        float(value) * spec.scale
                except (TypeError, ValueError):
                    pass
    return series


def _apply_derived(specs: List[MetricSpec], series: Dict[str, Dict[int, dict]]) -> None:
    """Fill each derived column in place, per timestamp.

    Per timestamp rather than once per GPU, so a ratio like GMEM% tracks growth over
    the run instead of reporting one moment's value against every row.
    """
    derived = applicable_derived(specs)
    if not derived:
        return
    for by_stamp in series.values():
        for cells in by_stamp.values():
            keyed = values_by_key(specs, cells)
            for column in derived:
                cells[column.header] = column.fn(keyed)


def _divisors(record: JobRecord, client: PrometheusClient,
              timeout: Optional[float]) -> dict:
    """A finished job's per-node cpus/total_memory, from its jobstats summary or from Prometheus.

    A record here is usually a finished job with its jobstats summary already decoded, but an
    explicit ``-j ID`` can also return a job that is still RUNNING and has none yet --
    rebuild just the divisors, the same way job_ave_stats.synthesize_stats() rebuilds
    the whole summary for the running view. The CPU-seconds and RSS themselves still come
    from the range query, which this does not give at per-timestamp granularity.
    """
    nodes = (record.stats or {}).get("nodes") if record else None
    if not nodes and record and record.jobid_raw and record.duration:
        nodes = host_stats(record.jobid_raw, record.duration, record.end, client, timeout)
    return nodes or {}


def finished_gpu(jobids: List[str], records: Dict[str, JobRecord],
                 specs: List[MetricSpec], client: PrometheusClient,
                 timeout: Optional[float], nodename: Optional[str] = None,
                 gpu_ids=None, window: Optional[int] = None, step: Optional[int] = None,
                 with_host: bool = False,
                 host_specs: Optional[List[CgroupSpec]] = None,
                 match: Optional[UnitFilter] = None) -> Iterator[JobSeries]:
    """Yield each finished job's GPU samples, optionally with its hosts' alongside.

    ``with_host`` is the default ``--ts`` view: GPU and host samples share one window
    per job -- the same :func:`range_window` result feeds both queries -- so they land
    on the same timestamp grid with no separate alignment step.

    ``match`` is checked by the caller after the last job, not here: a generator that
    raised on its own final step would do so from inside the render loop.
    """
    match = match if match is not None else UnitFilter(nodename, gpu_ids)
    sampling_period = client.sampling_period
    cgroup = chosen_specs(host_specs)

    for jid in jobids:
        record = records.get(jid)
        gpus = discover_gpus(record, client, timeout) if record else []
        if not gpus:
            print("warn: job %s has no GPU samples" % jid, file=sys.stderr)
            continue
        uuid_to: GpuIdentity = {g["uuid"]: (g["node"], g["minor"], g.get("model", ""))
                                for g in gpus}
        if match:
            # Before the queries, not after: dropping the other units' UUIDs here
            # shrinks the regex, so a 4-node job costs a quarter of the range queries
            # instead of fetching three nodes' samples to throw them away.
            uuid_to = match.keep(uuid_to)
            if not uuid_to:
                continue
        start, span = range_window(record.start, record.end, window,
                                   sampling_period, step)
        series = _gpu_samples(client, specs, uuid_to, start, record.end, span, timeout)
        _apply_derived(specs, series)

        host: Dict[str, Dict[int, dict]] = {}
        if with_host:
            # No --nodename narrowing on the host side: a row's node comes from its
            # GPU, and those were narrowed above.
            divisors, hosts = cgroup_hosts(_divisors(record, client, timeout), None)
            if hosts:
                host = host_series(record.jobid_raw, divisors, start, record.end, span,
                                   sampling_period, client, timeout, cgroup)
        yield JobSeries(jobid=jid, user=record.user, gpus=uuid_to, gpu=series, host=host)


def finished_host(jobids: List[str], records: Dict[str, JobRecord],
                  client: PrometheusClient, timeout: Optional[float],
                  nodename: Optional[str] = None, window: Optional[int] = None,
                  step: Optional[int] = None,
                  host_specs: Optional[List[CgroupSpec]] = None,
                  match: Optional[UnitFilter] = None) -> Iterator[JobSeries]:
    """Yield each finished job's cgroup samples -- the ``--cpu --ts`` view.

    No GPU dimension, so :attr:`JobSeries.gpus` stays empty and the renderer leaves
    the GPU/MODEL columns blank, keeping one schema across every ``--ts`` view.
    """
    match = match if match is not None else UnitFilter(nodename)
    sampling_period = client.sampling_period
    cgroup = chosen_specs(host_specs)

    for jid in jobids:
        record = records.get(jid)
        nodes = _divisors(record, client, timeout) if record else {}
        if not nodes:
            print("warn: job %s has no CPU/memory records" % jid, file=sys.stderr)
            continue
        if match:
            match.seen.update(h for h, n in nodes.items() if n.get("cpus"))
        divisors, hosts = cgroup_hosts(nodes, match.nodename)
        if not hosts:
            continue
        if match:
            match.matched = True
        start, span = range_window(record.start, record.end, window,
                                   sampling_period, step)
        yield JobSeries(
            jobid=jid, user=record.user, hosts=tuple(hosts),
            host=host_series(record.jobid_raw, divisors, start, record.end, span,
                             sampling_period, client, timeout, cgroup))


def _running_windows(jobs: Dict[int, RunningJob], divisors: dict,
                     window: Optional[int], sampling_period: int,
                     step: Optional[int], match: UnitFilter, warn: bool):
    """``[(raw_jobid, divisors, hosts, begin, end, span)]`` for the jobs worth querying."""
    tasks = []
    for raw_jobid, job in jobs.items():
        start, elapsed = job.get("start_epoch"), job.get("elapsed_seconds")
        if not start or not elapsed or elapsed <= 0:
            if warn:
                print("note: skipping CPU/MEM series for job %s: unknown runtime"
                      % job["jobid"], file=sys.stderr)
            continue
        by_host = divisors.get(raw_jobid, {})
        if warn and not any(node.get("cpus") for node in by_host.values()):
            print("warn: job %s has no CPU/memory records" % job["jobid"], file=sys.stderr)
            continue
        if match:
            match.seen.update(h for h, n in by_host.items() if n.get("cpus"))
        job_divisors, hosts = cgroup_hosts(by_host, match.nodename)
        if not hosts:
            continue
        if match:
            match.matched = True
        end = start + elapsed
        begin, span = range_window(start, end, window, sampling_period, step)
        tasks.append((raw_jobid, job_divisors, hosts, begin, end, span))
    return tasks


def running_host(jobs: Dict[int, RunningJob], client: PrometheusClient,
                 timeout: Optional[float], workers: int,
                 nodename: Optional[str] = None, window: Optional[int] = None,
                 step: Optional[int] = None,
                 host_specs: Optional[List[CgroupSpec]] = None,
                 warn: bool = True,
                 match: Optional[UnitFilter] = None) -> Dict[int, JobSeries]:
    """Every running job's cgroup samples, keyed by raw job ID.

    Not a generator, unlike the finished collectors: the divisors come from one
    batched :func:`jobscope.cpu.host_stats_many` call across the whole selection, and
    the per-job range queries then run concurrently. There is nothing left to stream.

    ``warn`` is off for the combined view, whose GPU rows are the report -- a job with
    no cgroup data there loses two columns, not its row, so saying so would be noise.
    """
    match = match if match is not None else UnitFilter(nodename)
    sampling_period = client.sampling_period
    cgroup = chosen_specs(host_specs)
    at = int(time.time())
    elapsed_by_job = {raw: job["elapsed_seconds"] for raw, job in jobs.items()
                      if (job.get("elapsed_seconds") or 0) > 0}
    divisors = host_stats_many(elapsed_by_job, at, client, timeout)

    tasks = _running_windows(jobs, divisors, window, sampling_period, step, match, warn)
    if not tasks:
        return {}

    def run(task):
        raw_jobid, job_divisors, _hosts, begin, end, span = task
        return raw_jobid, host_series(str(raw_jobid), job_divisors, begin, end, span,
                                      sampling_period, client, timeout, cgroup)

    series: Dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        for raw_jobid, found in pool.map(run, tasks):
            series[raw_jobid] = found

    out: Dict[int, JobSeries] = {}
    for raw_jobid, _d, hosts, _b, _e, _s in tasks:
        if raw_jobid not in series:
            continue
        job = jobs[raw_jobid]
        out[raw_jobid] = JobSeries(jobid=job["jobid"], user=job.get("user", "?"),
                                   hosts=tuple(hosts), host=series[raw_jobid])
    return out


def running_host_order(jobs: Dict[int, RunningJob], collected: Dict[int, JobSeries]):
    """``collected``'s job IDs in display order -- by start time, as the tables use."""
    return sorted(collected, key=lambda j: job_sort_key(jobs[j]))
