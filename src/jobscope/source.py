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

That arrangement was already here, hardcoded: ``BLOB_BACKED_KEYS`` named three keys
the blob served, and every view either excluded them or let ``_prefer_stored``
overwrite them afterwards. The policy is the same; it is now data, so a site can
state it and a caller can override it.

Two sources, two costs, which is what makes the default order the right one:

``blob``   free. It is already in the sacct output, so preferring it is the
           difference between no Prometheus query and one per job. It can only
           serve what jobstats stored -- see :data:`BLOB_COLUMNS`.
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
the only bound (see ``running.clip_to_job``). Preferring dcgm for ``GPU%`` therefore
makes it behave like ``SM_ACT%`` already does rather than like the blob: bounded by
the window, not by ownership. On a running job those coincide, since Slurm held the
card for the whole window.

GPU memory is deliberately not split across sources. ``GMEM%`` is ``GMEM_GB /
GMEM_TOTAL_GB``, so serving the two halves from different exporters would produce
a ratio belonging to neither -- exactly the unattributable number
:mod:`jobscope.extra_metric` refuses to manufacture. The pair moves together or
not at all, which on a cluster whose DCGM publishes no ``DCGM_FI_DEV_FB_TOTAL``
means it stays with blob-or-nvml. ``FB_USED_GB`` in the extended catalog is the
DCGM reading, under its own name, where nothing divides by it.
"""

from typing import Dict, FrozenSet, Iterable, List, NamedTuple, Sequence, Tuple

from .errors import JobscopeError

# Two axes, because the GPU columns and the host columns have different candidates and
# a site may want them ordered differently -- a cluster with dcgm-exporter but no
# cgroup exporter is ordinary. The blob is on both: it is the one source that spans
# them, which is what makes it the free default for each.
SOURCES: Tuple[str, ...] = ("blob", "dcgm", "nvml")
HOST_SOURCES: Tuple[str, ...] = ("blob", "cgroup", "slurm")
DEFAULT_PREFERENCE: Tuple[str, ...] = SOURCES
DEFAULT_HOST_PREFERENCE: Tuple[str, ...] = HOST_SOURCES

# The exporters, in the order a tie between them breaks. Kept apart from SOURCES
# because "which exporter serves this column" and "does the blob outrank it" are
# different questions, asked at different times: the first at config load, the
# second per job, once it is known whether the job has a blob at all.
EXPORTERS: Tuple[str, ...] = ("dcgm", "nvml", "cgroup", "slurm")

# What jobstats stored, by the column it serves. GMEM_TOTAL_GB is here because the
# blob records used *and* total -- which is why the blob can serve GMEM% when
# neither exporter half would have to be borrowed from the other.
BLOB_COLUMNS: Dict[str, str] = {
    "GPU%": "gpu_utilization",
    "GMEM_GB": "gpu_used_memory",
    "GMEM_TOTAL_GB": "gpu_total_memory",
}

# The host columns the blob serves, which unlike the GPU ones have no single field
# behind them: ``blob_metrics`` divides cpu_time by cpus*total_time and used_memory by
# total_memory. So they are named rather than mapped -- resolution only needs to know
# the blob *can* serve them, and the blob path computes them itself.
BLOB_HOST_COLUMNS: Tuple[str, ...] = ("CPU%", "MEM%")


class Resolution(NamedTuple):
    """The winning source for every column, and the specs to query.

    ``specs`` is the best *exporter* candidate per column, in catalog order, and is
    what the query builder consumes. ``from_blob`` is the columns the blob wins,
    which is a separate field rather than an absence from ``specs`` because the two
    views want opposite things from it: the summary skips those columns entirely
    (the whole point -- no query), while ``--per-gpu`` and the extended catalog
    query them and then defer to the stored value, which is what ``_prefer_stored``
    has always done and why it explains the window boundary in its docstring.
    """

    specs: Tuple = ()
    from_blob: FrozenSet[str] = frozenset()
    preference: Tuple[str, ...] = DEFAULT_PREFERENCE

    def source_of(self, header: str) -> str:
        """Which source serves ``header`` -- for the report's provenance line."""
        if header in self.from_blob:
            return "blob"
        for spec in self.specs:
            if spec.column == header:
                return spec.family
        return "?"

    def leading_exporter(self) -> str:
        """The first *exporter* in the preference -- the source whose defaults apply.

        The blob is skipped because it is not an exporter and has no catalog of its
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
        than off ``from_blob``, which is a set and would print them in whatever order
        it happened to hash.
        """
        groups: Dict[str, List[str]] = {}
        for spec in self.specs:
            name = "blob" if spec.column in self.from_blob else spec.family
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


def _rank(preference: Sequence[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(preference)}


def resolve(specs: Iterable, preference: Sequence[str] = DEFAULT_PREFERENCE,
            blob_columns: Iterable[str] = ()) -> Resolution:
    """One winning source per column.

    ``specs`` are the candidates -- several may name the same column, differing in
    family. For each column the best-ranked exporter wins; the blob then takes the
    column instead if it can serve it and outranks that exporter.

    ``blob_columns`` defaults to nothing so a caller that has no blob (a running
    job, or a finished one whose record carries no ``JS1:``) gets the exporter
    answer with no further branching. Pass :data:`BLOB_COLUMNS` to let it compete.
    """
    order = tuple(preference)
    rank = _rank(order)
    servable = set(blob_columns)
    # Materialised once: `specs` may be a generator, and it is walked twice -- for
    # the candidate grouping and for the catalog position each winner sorts by.
    catalog = list(specs)
    position = {spec.key: i for i, spec in enumerate(catalog)}

    candidates: Dict[str, List] = {}
    for spec in catalog:
        candidates.setdefault(spec.column, []).append(spec)

    chosen: List = []
    from_blob: List[str] = []
    for column, options in candidates.items():
        # A family absent from the preference sorts last rather than being dropped:
        # the alternative is a column with candidates and no winner, which reads to
        # a user as "the exporter had no data".
        best = min(options, key=lambda s: (rank.get(s.family, len(order)),
                                           EXPORTERS.index(s.family)
                                           if s.family in EXPORTERS else len(EXPORTERS)))
        if column in servable and rank.get("blob", len(order)) < rank.get(
                best.family, len(order)):
            from_blob.append(column)
        chosen.append(best)

    # Catalog order, not dict order: this list becomes the column order, and the
    # catalog's order is the one every rendered table and Worst-jobs row already
    # reads in. See metrics.in_catalog_order for the same reasoning on ballots.
    chosen.sort(key=lambda s: position.get(s.key, len(position)))
    return Resolution(specs=tuple(chosen), from_blob=frozenset(from_blob),
                      preference=order)
