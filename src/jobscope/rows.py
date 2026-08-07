"""Scheduler records and exporter readings, turned into what a renderer places.

The renderers used to do this themselves. ``SummaryRenderer.add`` called
``jobstats_metrics()`` on the gzipped base64 blob sacct hands over, mid-render, once
per job; ``DetailRenderer._rows_for`` called ``jobstats_detail()`` the same way. Two
consequences, both of which this module exists to end:

* A renderer could only be tested by building a ``JS1:`` AdminComment. Every test of
  a column, a band or a footer paid for a gzip round-trip through a storage format
  that has nothing to do with what it was checking.
* The render layer had to import the scheduler's ``JobRecord`` and the storage
  helpers to do its job, so "how a job is stored" and "how a job is shown" could not
  move independently. That is the coupling ``tests/test_layering.py`` documents and
  now enforces.

What is *not* here: formatting. A :class:`jobscope.models.JobRow` carries readings,
not cells. The one exception is the per-unit rows, whose cells are strings already --
see :class:`jobscope.models.UnitRow` for why.

The interesting decision in here is which source wins a column, in
:func:`_overrides`. It has to be made once per job, before anything is tallied, and
it used to live inline in the renderer.
"""

from typing import Dict, List, Mapping, Optional, Sequence

from . import dcgm
from .dcgm import JobGpuData, job_model
from .jobstats import (
    jobstats_capacity,
    jobstats_detail,
    jobstats_metrics,
    jobstats_per_node,
)
from .models import JobRow, ReportContext
from .slurm import JobRecord, Selection, format_window

# What a caller wants per-unit rows for, if anything. The summary view reads neither
# tuple, and building both for a thousands-of-job sweep is the single most expensive
# thing this module does -- measured at 63% of build_rows on a 2000-job selection, and
# worse the bigger the allocation. ``None`` builds both, for a caller that has not said.
JOB_LEVEL = "job"
GPU_LEVEL = "gpu"
NODE_LEVEL = "node"


def _overrides(measured: Mapping[str, float]) -> Dict[str, float]:
    """The queried values that outrank the stored summary, by header.

    An exporter named ahead of the summary (``--gpu-source dcgm``) owns its columns,
    so the queried value is the answer. ``dcgm._prefer_stored`` applies the same rule
    inside ``dcgm_for_job``, but cannot settle it for a running job: that summary is
    synthesized *after* the queries run, so it would win by arriving later.

    Restricted to columns with a resolved spec, which excludes the derived ``GMEM%``
    -- its inputs stay jobstats-backed, and a ratio assembled from a stored half and a
    queried half would belong to neither (see :mod:`jobscope.source`).

    Decided once per job rather than per cell, because the row and the footer average
    must not disagree about which number a job scored: a column overridden after it
    was tallied shows the measured value and averages the stored one.
    """
    catalog = dcgm.catalog()
    return {header: value for header, value in measured.items()
            if header in catalog.spec_by_header
            and header not in catalog.resolved.from_jobstats
            and value is not None}


def build_row(jobid: str, record: Optional[JobRecord],
              found: JobGpuData = None, level: Optional[str] = None) -> JobRow:
    """One job's :class:`~jobscope.models.JobRow`.

    ``record`` may be None -- a jobid the selection asked for that sacct did not
    return. The row still renders, as the identity dashes it always did, so a job
    does not silently vanish from a report that named it.

    ``level`` is the view the row is being built for, so only the per-unit rows that
    view will read are built. See :data:`JOB_LEVEL` on why that is worth a parameter.
    """
    found = found if found is not None else JobGpuData()
    gpus = record.gpus if record else 0
    stats = record.stats if record else None
    cores, memory = jobstats_capacity(stats)
    measured = found.overall or {}
    per_gpu = found.per_gpu or {}
    return JobRow(
        jobid=jobid,
        user=record.user if record else "?",
        state=record.state if record else "?",
        nodes=record.nodes if record else "-",
        name=record.name if record else "?",
        runtime=record.runtime if record else "-",
        gpus=gpus,
        duration=record.duration if record else None,
        found=record is not None,
        cores=cores,
        memory=memory,
        metrics=jobstats_metrics(stats, gpus),
        gpu_rows=(tuple(jobstats_detail(stats))
                  if level in (None, GPU_LEVEL) else ()),
        node_rows=(tuple(jobstats_per_node(stats))
                   if level in (None, NODE_LEVEL) else ()),
        measured=measured,
        overrides=_overrides(measured),
        per_gpu=per_gpu,
        per_node=found.per_node or {},
        model=job_model(per_gpu),
    )


def build_rows(jobids: Sequence[str], records: Mapping[str, JobRecord],
               dcgm_data: Mapping, level: Optional[str] = None) -> List[JobRow]:
    """The rows for one chunk, in the order ``jobids`` gives them.

    Order is the caller's: a chunk arrives already sorted the way the report prints,
    and re-deriving that here would be a second opinion about it.
    """
    return [build_row(jid, records.get(jid),
                      JobGpuData(*dcgm_data.get(jid, ())), level)
            for jid in jobids]


def any_unfinished(records: Mapping[str, JobRecord]) -> bool:
    """Whether any record has not ended, so its ``[start, end]`` window is still filling.

    The one place the question is asked of a set rather than a record. See
    :attr:`jobscope.models.ReportContext.unfinished` for what turns on the answer.
    """
    return any(record.unfinished for record in records.values())


def build_context(selection: Selection, desc: str,
                  records: Mapping[str, JobRecord]) -> ReportContext:
    """What the header block needs to know about ``selection``.

    Only an explicit-JOBID selection can name a job that has not ended, and it is the
    one case with real records in hand to ask: a window selection is finished by
    construction, so ``unfinished`` stays False there rather than being re-derived
    from records the caller may not have passed.
    """
    if selection.jobids:
        return ReportContext(
            desc=desc,
            owners=tuple(sorted({r.user for r in records.values() if r.user})),
            explicit_jobids=True,
            unfinished=any_unfinished(records),
        )
    # The dates behind "last 1 day" or "last 20 jobs", which the Select line does not
    # show: a -D window is computed from the clock, and a bare -N reaches back the
    # default lookback, so a reader could not otherwise tell what was scanned. Not for
    # an explicit -S/-E, where the Select line already is the window.
    dated = selection.days is not None or selection.lastn is not None
    return ReportContext(
        desc=desc,
        # -a/--all-users leaves `user` unset, so say so rather than printing None.
        user=selection.user or "(all users)",
        account=selection.account or "",
        partition=selection.partition or "",
        window=format_window(*selection.window()) if dated else "",
    )
