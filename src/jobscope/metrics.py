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

``cap`` is documentation now rather than a lookup: ``[eff] floor`` names the
metrics that lower a verdict, so job_eff reads the resolved Thresholds and
not this role. It stays because it is still true of POWER_W, and a site adding a
second floor metric should be able to see that the shape has a name.

There was a ``split`` role too, for the mechanism that divided the worst band by
whether the host was busy. That mechanism is gone -- CPU% is an ordinary voter with
a ceiling -- so the role went with it rather than lingering as a description of
something the code no longer does.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from . import dcgm
from .cpu import CGROUP_METRICS

WORST = "worst"
RESOURCE = "resource"
MEMORY = "memory"
CAP = "cap"

ROLES: Tuple[str, ...] = (WORST, RESOURCE, MEMORY, CAP)


def _catalog() -> List:
    """Every metric jobscope knows, GPU families first then host, in catalog order.

    Order matters and is not alphabetical: it is the order columns and Worst rows
    print in, and reproducing the old hand-written tuples exactly is what makes
    this a substitution rather than a change. GPU before host, because
    ``WORST_METRICS`` read ``GPU%, SM_ACT%, POWER_W, CPU%``.

    Reads ``dcgm.ALL_SPECS`` -- the *resolved* candidates, one per column -- rather
    than ``dcgm.METRICS``, which holds every candidate and so has ``GPU%`` twice
    once two exporters offer it. Roles are a property of the column, so asking a
    duplicated header for its roles has no single answer. Via the module rather
    than a ``from`` import because ``_rebuild`` rebinds that name.
    """
    return list(dcgm.ALL_SPECS) + list(dcgm.DERIVED_COLUMNS) + list(CGROUP_METRICS)


CATALOG: List = _catalog()

# Header -> spec, across every family. Flat because headers are unique catalog-wide
# (there is a test), so a caller never has to know which family it is asking about.
_BY_HEADER: Dict[str, object] = {spec.header: spec for spec in CATALOG}


def rebuild() -> None:
    """Recompute the cross-family view after the catalogs change.

    ``[metrics.<family>.<name>]`` can add or replace entries at config-load time,
    which is after this module was imported. Without this the site metric would be
    invisible here -- no role lookup, no ``label``, and ``probe`` would still call
    it unnamed -- while working perfectly everywhere else, which is the confusing
    kind of half-broken.
    """
    global CATALOG, _BY_HEADER
    CATALOG = _catalog()
    _BY_HEADER = {spec.header: spec for spec in CATALOG}


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


def in_catalog_order(headers) -> List[str]:
    """``headers`` sorted the way columns print, unknown names last, order kept.

    Ballots are built from whatever a series carried, so their iteration order is
    incidental. Anything user-facing -- the "best of ..." line, a heading's criteria
    -- should read in the same order as the table beside it.
    """
    position = {spec.header: i for i, spec in enumerate(CATALOG)}
    known = [h for h in headers if h in position]
    unknown = [h for h in headers if h not in position]
    return sorted(known, key=position.__getitem__) + unknown
