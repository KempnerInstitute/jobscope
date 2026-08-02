"""One place to ask what a metric is, across every family.

The catalogs live where their collectors do -- GPU metrics in :mod:`jobscope.dcgm`,
host metrics in :mod:`jobscope.cpu` -- because each knows how to query its own. What
did *not* have a home was the other half of a metric's identity: which ones a
verdict may be taken over, which get a Worst-jobs row, which are capacity readings
that must never count as work. That was spread across five hand-kept header tuples
in :mod:`jobscope.report`, one in :mod:`jobscope.plot` and one in
:mod:`jobscope.config`, none of which knew about the others.

They are now ``roles`` on the specs themselves, and this module is the only thing
that reads them. Adding a metric is one catalog entry; adding a *purpose* to an
existing metric is one word in its ``roles``.

The vocabulary, kept deliberately small:

``worst``     gets its own Worst-jobs ranking. Four of them, not every graded
              column: they say distinct things (duty cycle, SM residency, board
              watts, the host) where the DCGM catalog would add a dozen
              near-duplicates.
``resource``  one of the two distinct things a job holds, GPU and host. The
              ``gpu-cpu`` combined ranking is exactly these.
``memory``    a capacity reading, not a utilization one. Excluded from every
              verdict: a job that fills a card's memory and then computes nothing
              is idle, and letting GMEM% vote would call it busy.
``cap``       can only push a verdict *down*, never up -- POWER_W, whose floor is
              per GPU model. Not a percentage, so it has no tier of its own.
``split``     divides the worst band in two, so a job idle on the GPU but busy on
              the host reads differently from one idle on both. CPU% only.

``cap`` and ``split`` are read here so that :mod:`jobscope.config` can hand them to
a site later without the classifier growing a second source of truth; today they
name what the code already hardcodes.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from .cpu import CGROUP_METRICS
from .dcgm import DERIVED_COLUMNS, METRICS

WORST = "worst"
RESOURCE = "resource"
MEMORY = "memory"
CAP = "cap"
SPLIT = "split"

ROLES: Tuple[str, ...] = (WORST, RESOURCE, MEMORY, CAP, SPLIT)


def _catalog() -> List:
    """Every metric jobscope knows, GPU families first then host, in catalog order.

    Order matters and is not alphabetical: it is the order columns and Worst rows
    print in, and reproducing the old hand-written tuples exactly is what makes
    this a substitution rather than a change. GPU before host, because
    ``WORST_METRICS`` read ``GPU%, SM_ACT%, POWER_W, CPU%``.
    """
    return list(METRICS) + list(DERIVED_COLUMNS) + list(CGROUP_METRICS)


CATALOG: List = _catalog()

# Every spec that carries at least one role, so the common queries do not walk the
# full 37-entry catalog. Rebuilt from CATALOG rather than listed, so a role added
# to a spec is live immediately.
_BY_HEADER: Dict[str, object] = {spec.header: spec for spec in CATALOG}


def spec_for(header: str) -> Optional[object]:
    """The spec behind a column header, from any family, or None.

    Headers are unique across the catalogs -- there is a test -- so one flat
    mapping is enough and a caller never has to know which family it is asking
    about. That is the point: ``report`` should not import ``cpu`` to ask whether
    ``CPU%`` counts as memory.
    """
    return _BY_HEADER.get(header)


def with_role(role: str) -> List:
    """Every spec carrying ``role``, in catalog order."""
    return [spec for spec in CATALOG if role in getattr(spec, "roles", frozenset())]


def headers_with_role(role: str) -> Tuple[str, ...]:
    """The column headers carrying ``role``, in catalog order."""
    return tuple(spec.header for spec in with_role(role))


def has_role(header: str, role: str) -> bool:
    """Whether the metric behind ``header`` carries ``role``.

    False for an unknown header rather than an error: a ``--csv`` column a site
    added, or a derived one from a future family, is simply not that thing.
    """
    spec = spec_for(header)
    return bool(spec is not None and role in getattr(spec, "roles", frozenset()))


def label(header: str) -> str:
    """Short row-label form of a header -- ``SM_ACT%`` -> ``SM``.

    Falls back to the header without its trailing ``%`` for anything uncatalogued,
    which is what the old ``_WORST_SLUG.get(header, header.rstrip("%"))`` did.
    """
    spec = spec_for(header)
    return spec.label if spec is not None else header.rstrip("%")


def share_tag(header: str) -> str:
    """Suffix in a combined share -- ``35%gpu+24%cpu``.

    Total rather than a lookup that raises: ``_SHARE_TAG[header]`` was an
    unguarded ``KeyError`` waiting for the first metric given a Worst row without
    a matching tag entry.
    """
    spec = spec_for(header)
    return spec.share_tag if spec is not None else header.rstrip("%").lower()


def votable(headers: Sequence[str]) -> List[str]:
    """The columns a verdict may be taken over, in the order given.

    A percentage that is not a capacity reading. Watts are excluded by the ``%``
    test rather than by a role, since POWER_W's job is to cap rather than to vote.
    """
    return [h for h in headers if h.endswith("%") and not has_role(h, MEMORY)]
