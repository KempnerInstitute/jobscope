"""Tests for the cross-family metric registry.

The point of these is that the registry *reproduces* what the hand-kept header
tuples said, exactly. Where a test spells out a literal tuple, that literal is the
old table -- so if a role is added to the wrong spec, the test says which.
"""

import pytest

from jobscope import cpu, dcgm, metrics
from jobscope.dcgm import DERIVED_COLUMNS

# --- the tables it replaces -------------------------------------------------

def test_worst_metrics_reproduce_the_old_tuple_in_print_order():
    """Was `WORST_METRICS = ("GPU%", "SM_ACT%", "POWER_W", "CPU%")`. Order is the
    print order of the Worst rows, so it is part of the contract, not incidental."""
    assert metrics.headers_with_role(metrics.WORST) == ("GPU%", "SM_ACT%", "POWER_W", "CPU%")


def test_the_resource_pair_reproduces_combined_2():
    """Was `COMBINED_2 = ("GPU%", "CPU%")` -- the two distinct things a job holds."""
    assert metrics.headers_with_role(metrics.RESOURCE) == ("GPU%", "CPU%")


def test_memory_covers_the_two_the_old_skip_list_named():
    """Was `CLASSIFY_SKIP = ("GMEM%", "MEM%")`. It may now cover more -- that is the
    point -- but it must never cover less."""
    assert {"GMEM%", "MEM%"} <= set(metrics.headers_with_role(metrics.MEMORY))


def test_memory_also_covers_the_cgroup_columns_that_used_to_slip_through():
    """CACHE% and MEM_USED% are memory. Before the role existed, a cpu-only --ts
    series carrying CACHE% would have taken its verdict from it."""
    memory = set(metrics.headers_with_role(metrics.MEMORY))
    assert {"CACHE%", "MEM_USED%"} <= memory


def test_power_is_the_only_cap():
    """`cap` names the shape [eff] floor now implements: a metric that can only
    lower a verdict. POWER_W is the built-in one; the role is documentation, since
    job_eff reads the resolved floors rather than this."""
    assert metrics.headers_with_role(metrics.CAP) == ("POWER_W",)


# --- the short forms --------------------------------------------------------

@pytest.mark.parametrize("header,slug,tag", [
    ("GPU%", "GPU", "gpu"),
    ("SM_ACT%", "SM", "sm"),        # was an override in both old tables
    ("POWER_W", "POWER", "pw"),     # the two forms differ; both were overrides
    ("CPU%", "CPU", "cpu"),
])
def test_label_and_tag_reproduce_the_old_slug_and_share_tables(header, slug, tag):
    assert metrics.label(header) == slug
    assert metrics.share_tag(header) == tag


def test_an_uncatalogued_header_falls_back_instead_of_raising():
    """`_SHARE_TAG[header]` was an unguarded KeyError; the first metric given a
    Worst row without a matching tag entry would have crashed the report."""
    assert metrics.label("WIDGET%") == "WIDGET"
    assert metrics.share_tag("WIDGET%") == "widget"
    assert metrics.spec_for("WIDGET%") is None


def test_a_derived_column_resolves_like_a_queried_one():
    """GMEM% is computed, not fetched, and still has to answer role questions."""
    assert metrics.spec_for("GMEM%") is not None
    assert metrics.has_role("GMEM%", metrics.MEMORY)


# --- votable ---------------------------------------------------------------

def test_votable_keeps_percentages_and_drops_memory():
    assert metrics.votable(["GPU%", "GMEM%", "SM_ACT%", "MEM%"]) == ["GPU%", "SM_ACT%"]


def test_votable_drops_watts_on_the_percent_test():
    """POWER_W caps a verdict rather than voting; a raw wattage in the ballot would
    win classify()'s comparison outright whatever the GPU was doing."""
    assert "POWER_W" not in metrics.votable(["GPU%", "POWER_W"])


def test_votable_preserves_the_order_given():
    """Callers pass CSV column order and expect it back."""
    assert metrics.votable(["DRAM%", "GPU%", "SM_ACT%"]) == ["DRAM%", "GPU%", "SM_ACT%"]


def test_votable_keeps_an_unknown_percentage_column():
    """A site's own metric votes by default -- it is a utilization percentage until
    something says otherwise. Excluding it silently would be the worse failure."""
    assert "WIDGET%" in metrics.votable(["WIDGET%"])


def test_a_cpu_only_series_with_cache_votes_only_on_cpu():
    """The regression bar from the plan: this series must not take its verdict from
    CACHE%, which would make a job holding page cache look busy."""
    got = metrics.votable(["CPU%", "CPU_USER%", "CPU_SYS%", "CACHE%"])
    assert "CACHE%" not in got and "CPU%" in got


# --- catalog integrity ------------------------------------------------------

def test_headers_are_unique_across_every_family():
    """One flat header -> spec mapping is only valid if no two families collide;
    a duplicate would silently shadow one metric with another.

    Asserted over the *resolved* GPU catalog rather than the candidates. Two
    exporters may both offer GPU%, and jobscope.source picks one -- so a duplicate
    here would mean resolution failed to, which is the bug this guards."""
    headers = [s.header for s in dcgm.catalog().all_specs] + [d.header for d in DERIVED_COLUMNS] \
        + [c.header for c in cpu.catalog().metrics]
    assert len(headers) == len(set(headers))


def test_a_column_offered_by_two_exporters_resolves_to_one():
    """The reason the above is about ALL_SPECS: GPU% has an nvml and a dcgm candidate,
    and exactly one of them may be live at a time."""
    candidates = [s for s in dcgm.catalog().metrics if s.column == "GPU%"]
    assert {s.family for s in candidates} == {"nvml", "dcgm"}
    assert len([s for s in dcgm.catalog().all_specs if s.column == "GPU%"]) == 1


def test_every_role_used_in_the_catalog_is_a_declared_one():
    """A typo in a spec's roles would otherwise be a role nothing ever matches."""
    used = set()
    for spec in metrics.catalog().specs:
        used |= set(getattr(spec, "roles", frozenset()))
    assert used <= set(metrics.ROLES), "undeclared role(s): %s" % (used - set(metrics.ROLES))


def test_every_declared_role_is_actually_carried_by_something():
    """A role nothing claims is dead vocabulary."""
    for role in metrics.ROLES:
        assert metrics.with_role(role), "no metric carries %r" % role


def test_the_catalog_spans_all_three_sources():
    headers = {s.header for s in metrics.catalog().specs}
    assert {"SM_ACT%", "GPU%", "GMEM%", "CPU%"} <= headers


def test_has_role_is_false_for_an_unknown_header_rather_than_raising():
    assert metrics.has_role("NOSUCH%", metrics.MEMORY) is False
