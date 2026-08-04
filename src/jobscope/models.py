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

from dataclasses import dataclass
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
    over four literals, and ``DETAIL_COLUMNS``' row indices -- with no way to add a
    metric without touching all of them.

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
