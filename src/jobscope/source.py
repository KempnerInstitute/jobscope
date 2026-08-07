"""Which source serves each column.

A column can have more than one candidate. ``GPU%`` is published by dcgm-exporter
as ``DCGM_FI_DEV_GPU_UTIL`` and by the nvidia exporter as
``nvidia_gpu_duty_cycle``, and for a finished job jobstats has already computed it
into the ``JS1:`` blob that sacct hands over for free. ``SM_ACT%`` has exactly one
candidate, because only DCGM publishes it at all.

So the choice is not a mode. It is one decision per column, and this module makes
it: given the candidate specs and an order of preference, return the winner for
each column. A preference is therefore a *reordering* rather than a switch --
asking for dcgm cannot take away a column dcgm does not publish, it can only move
dcgm ahead for the columns it does.

That arrangement was already here, hardcoded: ``JOBSTATS_BACKED_KEYS`` named three keys
the jobstats summary served, and every view either excluded them or let ``_prefer_stored``
overwrite them afterwards. The policy is the same; it is now data, so a site can
state it and a caller can override it.

Two sources, two costs, which is what makes the default order the right one:

``jobstats`` free. It is already in the sacct output, so preferring it is the
           difference between no Prometheus query and one per job. It can only
           serve what jobstats stored -- see :data:`JOBSTATS_COLUMNS`.
``dcgm``   complete. The whole profiling catalog, and the only source for SM
           activity, tensor pipes and DRAM.
``nvml``   partial: duty cycle, memory, power, temperature. Also *mandatory*
           regardless of this order, because ``nvidia_gpu_jobId`` is the only
           series that maps a card to a job -- see :func:`jobscope.config.gpu_join`
           and ``dcgm.discover_gpus``. Choosing dcgm here selects where the
           *numbers* come from; it does not remove the nvidia exporter.

One asymmetry is worth knowing before preferring dcgm for a column nvml also
serves. An ``nvidia_*`` series can be intersected with the join series -- they come
from one exporter and so share a label set -- which clips the window to the samples
where the job actually held the card, the way jobstats does it. ``DCGM_*`` series
carry different labels, so PromQL's ``and`` can never match and the job's window is
the only bound (see ``dcgm.ownership_clip``). Preferring dcgm for ``GPU%`` therefore
makes it behave like ``SM_ACT%`` already does rather than like the jobstats summary: bounded by
the window, not by ownership. Those nearly coincide -- Slurm held the card for the
job's whole window -- but not exactly: the exporter keeps publishing the previous owner's
labelled series for a scrape or two after that job ends, so the first samples of the
window can belong to someone else. Measured on one A100: a job's own 2501-second window
held two duty-cycle series, its own at 73.7 and the previous job's at 40.0.

GPU memory is deliberately not split across sources. ``GMEM%`` is ``GMEM_GB /
GMEM_TOTAL_GB``, so serving the two halves from different exporters would produce
a ratio belonging to neither -- exactly the unattributable number
:mod:`jobscope.extra_metric` refuses to manufacture. The pair moves together or
not at all, which on a cluster whose DCGM publishes no ``DCGM_FI_DEV_FB_TOTAL``
means it stays with jobstats-or-nvml. ``FB_USED_GB`` in the extended catalog is the
DCGM reading, under its own name, where nothing divides by it.
"""

from typing import Dict, FrozenSet, Iterable, List, NamedTuple, Sequence, Tuple

from .errors import JobscopeError

# Two axes, because the GPU columns and the host columns have different candidates and
# a site may want them ordered differently -- a cluster with dcgm-exporter but no
# cgroup exporter is ordinary. The summary is on both: it is the one source that spans
# them, which is what makes it the free default for each.
SOURCES: Tuple[str, ...] = ("jobstats", "dcgm", "nvml")
HOST_SOURCES: Tuple[str, ...] = ("jobstats", "cgroup", "slurm")
DEFAULT_PREFERENCE: Tuple[str, ...] = SOURCES
DEFAULT_HOST_PREFERENCE: Tuple[str, ...] = HOST_SOURCES

# The exporters, in the order a tie between them breaks. Kept apart from SOURCES
# because "which exporter serves this column" and "does the jobstats summary outrank it" are
# different questions, asked at different times: the first at config load, the
# second per job, once it is known whether the job has a jobstats summary at all.
EXPORTERS: Tuple[str, ...] = ("dcgm", "nvml", "cgroup", "slurm")

# What jobstats stored, by the column it serves. GMEM_TOTAL_GB is here because the
# summary records used *and* total -- which is why the jobstats summary can serve GMEM% when
# neither exporter half would have to be borrowed from the other.
JOBSTATS_COLUMNS: Dict[str, str] = {
    "GPU%": "gpu_utilization",
    "GMEM_GB": "gpu_used_memory",
    "GMEM_TOTAL_GB": "gpu_total_memory",
}

# The host columns the jobstats summary serves, which unlike the GPU ones have no single field
# behind them: ``jobstats_metrics`` divides cpu_time by cpus*total_time and used_memory by
# total_memory. So they are named rather than mapped -- resolution only needs to know
# the jobstats summary *can* serve them, and the jobstats path computes them itself.
JOBSTATS_HOST_COLUMNS: Tuple[str, ...] = ("CPU%", "MEM%")

# Source names that were renamed, and what to write now. The counterpart of
# cli.RETIRED_FLAGS: a config predating the rename names something that still exists
# under another name, so parse_preference points at it rather than only listing the
# valid set and leaving the reader to guess which one it became.
RETIRED_SOURCES: Dict[str, str] = {"blob": "jobstats"}


class Resolution(NamedTuple):
    """The winning source for every column, and the specs to query.

    ``specs`` is the best *exporter* candidate per column, in catalog order, and is
    what the query builder consumes. ``from_jobstats`` is the columns the jobstats summary wins,
    which is a separate field rather than an absence from ``specs`` because the two
    views want opposite things from it: the summary skips those columns entirely
    (the whole point -- no query), while ``--per-gpu`` and the extended catalog
    query them and then defer to the stored value, which is what ``_prefer_stored``
    has always done and why it explains the window boundary in its docstring.
    """

    specs: Tuple = ()
    from_jobstats: FrozenSet[str] = frozenset()
    preference: Tuple[str, ...] = DEFAULT_PREFERENCE

    def source_of(self, header: str) -> str:
        """Which source serves ``header`` -- for the report's provenance line."""
        if header in self.from_jobstats:
            return "jobstats"
        for spec in self.specs:
            if spec.column == header:
                return spec.family
        return "?"

    def leading_exporter(self) -> str:
        """The first *exporter* in the preference -- the source whose defaults apply.

        The summary is skipped because it is not an exporter and has no catalog of its
        own: it serves three columns and cannot answer "what does this source
        publish", which is the question a per-source default view asks.
        """
        for name in self.preference:
            if name in EXPORTERS:
                return name
        return EXPORTERS[0]

    def exporters_used(self) -> Tuple[str, ...]:
        """The families actually queried, in preference order.

        Not the same as the preference: asking for nvml first still queries dcgm
        for SM_ACT%, because nothing else publishes it.
        """
        used = {spec.family for spec in self.specs}
        return tuple(name for name in self.preference if name in used)

    def by_source(self) -> List[Tuple[str, Tuple[str, ...]]]:
        """``[(source, columns)]`` in preference order, for one line of output.

        Columns keep catalog order within each group -- driven off ``specs`` rather
        than off ``from_jobstats``, which is a set and would print them in whatever order
        it happened to hash.
        """
        groups: Dict[str, List[str]] = {}
        for spec in self.specs:
            name = "jobstats" if spec.column in self.from_jobstats else spec.family
            groups.setdefault(name, []).append(spec.column)
        return [(name, tuple(groups[name])) for name in self.preference if name in groups]


def parse_preference(value, where: str = "[gpu] source",
                     allowed: Sequence[str] = SOURCES) -> Tuple[str, ...]:
    """A source order from config or the command line.

    Accepts a list, or a string naming one source or several comma-separated. Any
    source left unnamed keeps its default position at the end, so naming one is
    enough to promote it -- ``--gpu-source dcgm`` means "dcgm first, then whatever
    you would have done", not "dcgm only". Nothing is silently dropped: a column
    whose only candidate is an unnamed source is still served.

    ``allowed`` is the axis: :data:`SOURCES` for the GPU columns, :data:`HOST_SOURCES`
    for CPU%/MEM%. Naming a source from the other axis is an error rather than a
    no-op, since ``[host] source = "dcgm"`` cannot mean anything.
    """
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        names = [str(part).strip() for part in value]
    else:
        raise JobscopeError("%s must be a source name or a list of them, not %r"
                            % (where, value))
    names = [name.lower() for name in names if name]
    if not names:
        raise JobscopeError("%s names no sources; it takes %s"
                            % (where, ", ".join(allowed)))
    retired = [name for name in names if name in RETIRED_SOURCES]
    if retired:
        # Named with its successor, not merely rejected: a config that predates the
        # rename is asking for something that still exists under another name, and
        # listing the valid set leaves the reader to guess which one it became.
        raise JobscopeError("%s: %s is no longer a source name; use %s"
                            % (where, ", ".join(sorted(set(retired))),
                               ", ".join(sorted({RETIRED_SOURCES[n] for n in retired}))))
    unknown = [name for name in names if name not in allowed]
    if unknown:
        # Named rather than ignored: a typo would otherwise read as a source that
        # simply had no data, which is indistinguishable from working.
        raise JobscopeError("%s has no source %s; it takes %s"
                            % (where, ", ".join(sorted(set(unknown))), ", ".join(allowed)))
    seen: List[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return tuple(seen) + tuple(name for name in allowed if name not in seen)


def merged_metrics(builtins: Sequence, extra: Sequence, inherit) -> Tuple:
    """The built-in catalog with ``extra`` overriding by key and appending the rest.

    Both families register site metrics the same way, so the rule lives here once
    rather than beside each catalog. What ``inherit`` decides differs -- which fields
    a site may override and which keep the built-in's purpose -- so that stays with
    the family that knows its own spec type.

    The ``pop`` does double duty and is the reason this is worth naming: it hands
    ``inherit`` the override *and* consumes it, so what is left in ``by_key`` at the
    end is exactly the extras that matched no built-in. Those append in declaration
    order, which is what keeps catalog order stable and sorts site metrics last.
    """
    by_key = {spec.key: spec for spec in extra}
    merged = [inherit(builtin, by_key.pop(builtin.key, None)) for builtin in builtins]
    return tuple(merged + [spec for spec in extra if spec.key in by_key])


def _rank(preference: Sequence[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(preference)}


def resolve(specs: Iterable, preference: Sequence[str] = DEFAULT_PREFERENCE,
            jobstats_columns: Iterable[str] = ()) -> Resolution:
    """One winning source per column.

    ``specs`` are the candidates -- several may name the same column, differing in
    family. For each column the best-ranked exporter wins; the jobstats summary then takes the
    column instead if it can serve it and outranks that exporter.

    ``jobstats_columns`` defaults to nothing so a caller that has no jobstats summary (a running
    job, or a finished one whose record carries no ``JS1:``) gets the exporter
    answer with no further branching. Pass :data:`JOBSTATS_COLUMNS` to let it compete.
    """
    order = tuple(preference)
    rank = _rank(order)
    servable = set(jobstats_columns)
    # Materialised once: `specs` may be a generator, and it is walked twice -- for
    # the candidate grouping and for the catalog position each winner sorts by.
    catalog = list(specs)
    position = {spec.key: i for i, spec in enumerate(catalog)}

    candidates: Dict[str, List] = {}
    for spec in catalog:
        candidates.setdefault(spec.column, []).append(spec)

    chosen: List = []
    from_jobstats: List[str] = []
    for column, options in candidates.items():
        # A family absent from the preference sorts last rather than being dropped:
        # the alternative is a column with candidates and no winner, which reads to
        # a user as "the exporter had no data".
        best = min(options, key=lambda s: (rank.get(s.family, len(order)),
                                           EXPORTERS.index(s.family)
                                           if s.family in EXPORTERS else len(EXPORTERS)))
        if column in servable and rank.get("jobstats", len(order)) < rank.get(
                best.family, len(order)):
            from_jobstats.append(column)
        chosen.append(best)

    # Catalog order, not dict order: this list becomes the column order, and the
    # catalog's order is the one every rendered table and Worst-jobs row already
    # reads in. See metrics.in_catalog_order for the same reasoning on ballots.
    chosen.sort(key=lambda s: position.get(s.key, len(position)))
    return Resolution(specs=tuple(chosen), from_jobstats=frozenset(from_jobstats),
                      preference=order)
