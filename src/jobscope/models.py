"""The values a report is made of, and the three things "no number" can mean.

A utilization figure is missing for two opposite reasons, and collapsing them is
how a monitoring outage comes to look like a fleet of wasteful jobs:

``ok``        measured, and here is the number.
``na``        *not applicable* -- the resource does not exist. A CPU-only job has
              no GPU%. A fact about the job; the verdict around it stays valid.
``unknown``   *should exist, could not be read* -- exporter down, job older than
              Prometheus retention, query timed out, summary missing a denominator.
              A fact about our collection, not about the job.

Both render as ``-``, so nothing about the display changes. The difference is what
job_eff is allowed to conclude: an ``na`` metric is simply left out of the
ballot, where a job whose *voting* metrics are all ``unknown`` must not be
classified at all. It gets ``no-data`` and is excluded from the counts and the
averages, because "we did not measure this" and "this job wasted its allocation"
are not the same claim and only one of them gets someone an email.

The rule this replaces: ``classify()`` used ``default="wasteful"`` on an empty
ballot. That was written for the ``na`` case -- a CPU-only job in a GPU sweep
genuinely has no GPU work -- and was correct for it. It becomes wrong the moment
the empty ballot can also mean "Prometheus did not answer", which is routine once
every metric is a network query.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, Mapping, Optional, Tuple

OK = "ok"
NA = "na"
UNKNOWN = "unknown"

STATES: Tuple[str, ...] = (OK, NA, UNKNOWN)


@dataclass(frozen=True)
class Measure:
    """One metric's reading for one unit, or the reason there is not one."""

    value: Optional[float]
    state: str
    reason: str = ""

    @classmethod
    def reading(cls, value: float) -> "Measure":
        """Store the value exactly as given -- no float() coercion.

        jobstats rounds to whole percents on purpose and the renderers ``str()``
        them, so coercing would turn a displayed ``100`` into ``100.0`` in the CSV
        and in every table.
        """
        return cls(value, OK)

    @classmethod
    def not_applicable(cls, reason: str) -> "Measure":
        """The resource does not exist for this job."""
        return cls(None, NA, reason)

    @classmethod
    def unmeasured(cls, reason: str) -> "Measure":
        """It should exist and could not be read. Say why -- the reason is the
        difference between a bare dash and an actionable report."""
        return cls(None, UNKNOWN, reason)

    @property
    def known(self) -> bool:
        return self.state == OK

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError("unknown measure state: %r" % (self.state,))
        if (self.value is None) == (self.state == OK):
            raise ValueError(
                "a %r measure must %s carry a value" % (self.state,
                                                        "" if self.state == OK else "not"))


@dataclass(frozen=True)
class JobMetrics:
    """One job's readings, keyed by column header.

    Replaces the positional ``(cpu, mem, gpu, gmem)`` tuple, whose order was a
    contract three places had to agree on -- ``_BLOB_HEADERS.index()``, a ``zip``
    over four literals, and the detail view's row indices -- with no way to add a
    metric without touching all of them. The last of those is gone too: the detail
    columns are generated from the resolved headers by ``report.detail_columns()``,
    after the hand-written indices printed every profiling value one column to the
    left of its header under ``--gpu-source dcgm``.

    ``source`` names where the numbers came from, because with a jobstats summary fast path and
    a Prometheus path the same column can come from either and they do not always
    agree. A report that mixed them across jobs without saying so would make an
    apples-to-oranges comparison look like a finding.
    """

    by_header: Mapping[str, Measure]
    source: str = ""

    def __iter__(self) -> Iterator[Tuple[str, Measure]]:
        return iter(self.by_header.items())

    def __contains__(self, header: str) -> bool:
        return header in self.by_header

    def value(self, header: str) -> Optional[float]:
        """``header``'s number, or None for absent in any of its senses.

        Total on purpose: every display path already renders None as ``-``, so
        callers that only want to print keep working unchanged. Callers that must
        distinguish -- job_eff -- ask :meth:`state`.
        """
        found = self.by_header.get(header)
        return found.value if found is not None else None

    def state(self, header: str) -> str:
        """``ok``/``na``/``unknown`` for ``header``; ``unknown`` if never collected.

        Absent-from-this-source and read-but-unknown deliberately collapse: a DCGM
        column is not something the jobstats summary *failed* to measure, it is something the
        summary is not about, but no caller needs to tell those apart. What decides a
        `no-data` verdict is which columns the *series* carries -- see
        job_eff.unceilinged -- and that is asked of the column set, not of one
        job's readings. Read ``by_header`` directly if you ever need the difference.
        """
        found = self.by_header.get(header)
        return found.state if found is not None else UNKNOWN

    def known(self) -> Dict[str, float]:
        """Just the measured values, for arithmetic that cannot see a None."""
        return {h: m.value for h, m in self.by_header.items()
                if m.known and m.value is not None}


# Measure.reason is still recorded, and jobstats.py sets it on every unmeasured value --
# a JobMetrics knows *why* each gap is there. Nothing renders it yet: the planned
# line was "GPU% unknown -- no samples in window; job ended 2026-01-03, retention
# begins 2026-02-04", and the aggregator for it (dedupe the sentences, since one dead
# exporter produces the same one for every column it served) was written before the
# renderer and removed unused. Read `by_header` when that lands.


@dataclass(frozen=True)
class ReportContext:
    """What the header block says about a selection, without the selection.

    The renderer needs to print who and what was asked for, and whether any job is
    still running. It does not need to know that those answers came from an ``sacct``
    window, a ``squeue`` snapshot, or a list of job IDs -- and taking a ``Selection``
    to find out was the last thing making the render layer import the scheduler.

    ``window`` arrives formatted, because formatting it needs the selection's own
    clock arithmetic. Empty when the ``Select`` line already *is* the window, which is
    the case for an explicit ``-S/-E``.

    ``owners`` is populated only for an explicit-JOBID selection: there the -u/-A/-p
    filters were bypassed, so the header names the jobs' actual owners rather than a
    filter that was not applied.
    """

    desc: str = ""
    user: str = ""
    account: str = ""
    partition: str = ""
    window: str = ""
    owners: Tuple[str, ...] = ()
    explicit_jobids: bool = False
    # Whether any job has not ended, so its window is still filling. Two answers depend
    # on it and must not diverge: the Sampled line's span and whether the summary may
    # weight by resource-time. Conservative on a mixed selection -- one running job
    # among finished ones means a table cannot be half weighted by hours.
    unfinished: bool = False


# What a report is a row *about*: the job, or one of the two units it ran on. Here
# rather than beside either user because both layers speak it and neither owns it --
# `jobscope.rows` reads it to build only the per-unit tuples the view will look at, and
# `jobscope.report` reads it to pick the identity prefix. They used to hold a copy each,
# with `cli` naming the levels a third time in literals, and the failure mode of a
# disagreement was silent: an unrecognised level builds neither tuple and the detail
# view prints "(no jobstats data)" for a job that has plenty.
JOB_LEVEL = "job"
GPU_LEVEL = "gpu"
NODE_LEVEL = "node"


@dataclass(frozen=True)
class UnitRow:
    """One node, or one GPU: what the detail views print a line for.

    ``cells`` is keyed by column header -- ``CPU%``, ``CPU-MEM``, ``GPU%``,
    ``GPU-MEM``, ``GMEM%`` -- and deliberately *not* a tuple. What this replaces was a
    fixed 7-tuple with ``(node, unit, cpu%, cpu-mem, gpu%, gpu-mem, gmem%)`` written
    down in three places: the storage helpers that built it, the renderer's
    ``_NODE_INDEX``/``_GPU_INDEX`` constants, and a hand-kept prefix of seven Columns.
    A site whose exporter serves a different pair could not reconfigure any of it, at
    the one point in the codebase where every other column list is catalog-driven.

    The cells are strings, not numbers, because two of them are not numbers: ``GPU-MEM``
    is a ``used/total`` pair and ``CPU-MEM`` likewise. Their precision is a property of
    what jobstats stored -- see ``jobstats._ROUNDING`` -- so they are spelled where that
    is documented rather than re-derived by a renderer.

    ``unit`` is a card's minor number at GPU level and the count of cards pooled at node
    level. That single difference is why the two levels can share a renderer.
    """

    node: str
    unit: str
    cells: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JobRow:
    """One job, ready to render: identity, capacity, and the readings.

    What the renderers take instead of a ``JobRecord`` and a ``dcgm_data`` tuple. The
    difference is where the domain transformation happens. ``SummaryRenderer.add`` used
    to call ``jobstats_metrics()`` on a gzipped base64 blob mid-render, so a renderer
    could only be tested by building one, and the render layer had to import the
    scheduler's model and the storage format to do its job. :mod:`jobscope.rows` builds
    these instead; the renderers place cells.

    ``cores``/``memory`` are the *allocation*, not the usage -- what the summary
    weights a job's contribution by. They come from ``jobstats_capacity`` and are
    carried rather than recomputed because the stats dict they are read from is
    exactly what this type exists to keep out of the renderers.

    ``measured`` is the job-level exporter reading per header, and ``per_gpu`` /
    ``per_node`` the same broken down by card and by host. Kept separate from
    ``metrics`` -- the stored summary -- because which of the two wins a column is a
    resolution question the renderer must not re-answer per cell; see
    :func:`jobscope.rows.build_rows`.
    """

    jobid: str
    user: str = "?"
    state: str = "?"
    nodes: str = "-"
    name: str = "?"
    runtime: str = "-"
    # Identity beyond the job and its owner. Carried always and shown only under
    # --show: they cost nothing to fetch (sacct charges for rows, not columns) and a
    # row that had to be rebuilt to answer "which account was that?" would defeat the
    # point of building it once.
    account: str = ""
    partition: str = ""
    cluster: str = ""
    gpus: int = 0
    duration: Optional[int] = None
    # Whether the scheduler returned a record for this jobid at all. False means every
    # identity field above is the dash it defaults to. It is not the same question as
    # `has_summary`, and the summary's weighting distinguishes them: a job that was
    # never found is left out of the weighted mean silently, where one that was found
    # but has no usable duration is counted in the "not weighted" note.
    found: bool = True
    cores: int = 0
    memory: int = 0
    metrics: JobMetrics = field(default_factory=lambda: JobMetrics({}))
    # The stored summary broken out per card and per host, for the two detail levels.
    gpu_rows: Tuple["UnitRow", ...] = ()
    node_rows: Tuple["UnitRow", ...] = ()
    measured: Mapping[str, float] = field(default_factory=dict)
    # The subset of ``measured`` whose column the exporter outranks the stored summary
    # for. Settled when the row is built, not per cell: the row and the footer average
    # must not disagree about which number a job scored. See jobscope.rows._overrides.
    overrides: Mapping[str, float] = field(default_factory=dict)
    per_gpu: Mapping = field(default_factory=dict)
    per_node: Mapping = field(default_factory=dict)
    model: str = ""

    @property
    def has_summary(self) -> bool:
        """Whether jobstats stored a summary for this job.

        A job without one is only half measured: it can have exporter numbers but no
        CPU%/MEM%/GPU%/GMEM%. It still lists, because it is a real job; it does not
        vote, because a job cannot be compared with the rest on a metric it has no
        value for. See :meth:`jobscope.report.SummaryRenderer.add`.
        """
        return bool(self.metrics.by_header)
