"""Tests for the cgroup metric catalog and its time series (the host analogue of dcgm.py)."""

import pytest

from jobscope.cpu import (
    CGROUP_METRICS,
    DEFAULT_CGROUP_SPECS,
    RATE_LOOKBACK_SCRAPES,
    SPEC_BY_KEY,
    host_series,
    spec_named,
    specs_named,
)


class FakeClient:
    """A Prometheus stand-in driven by canned range results, keyed by metric then host.

    Dispatch is on the *catalog's* metric names rather than a hardcoded pair, so a
    spec whose query nothing answers shows up as an empty column here the same way
    it would against a real server -- see the note in tests/conftest.py about fakes
    that silently return [] for anything they were not taught.
    """

    def __init__(self, values=None, sampling_period=60):
        # {prometheus metric name: {host: [(ts, raw), ...]}}
        self.values = values or {}
        self.sampling_period = sampling_period
        self.calls = []

    def query_range(self, query, start, end, step, timeout=None):
        self.calls.append(query)
        for metric, by_host in self.values.items():
            # Substring rather than equality: the counters arrive wrapped in rate().
            if metric in query:
                return [{"metric": {"host": "%s:9100" % host},
                         "values": [[ts, str(v)] for ts, v in pts]}
                        for host, pts in by_host.items()]
        return []


CPU_SECS = "cgroup_cpu_total_seconds"
RSS = "cgroup_memory_rss_bytes"


# --- the catalog -----------------------------------------------------------

def test_catalog_keys_and_headers_are_unique():
    """Both are lookup keys -- a duplicate would silently shadow an entry."""
    keys = [spec.key for spec in CGROUP_METRICS]
    headers = [spec.header for spec in CGROUP_METRICS]
    assert len(keys) == len(set(keys))
    assert len(headers) == len(set(headers))


def test_default_specs_are_the_two_the_summary_has_always_shown():
    """CPU%/MEM% are the only cgroup metrics a stored blob can reconstruct, so they
    are the only ones outside the opt-in `all` group."""
    assert [spec.header for spec in DEFAULT_CGROUP_SPECS] == ["CPU%", "MEM%"]
    assert all(spec.group == "default" for spec in DEFAULT_CGROUP_SPECS)


def test_every_spec_names_a_denominator_that_is_a_blob_field():
    assert {spec.denom for spec in CGROUP_METRICS} == {"cpus", "total_memory"}


def test_spec_named_resolves_by_key_and_by_header():
    assert spec_named("cpu") is SPEC_BY_KEY["cpu"]
    assert spec_named("CPU%") is SPEC_BY_KEY["cpu"]
    assert spec_named("cpu%") is SPEC_BY_KEY["cpu"]
    assert spec_named("nonesuch") is None


def test_specs_named_returns_catalog_order_and_drops_duplicates():
    """Column order is a property of the report, not of how a site listed them."""
    specs = specs_named(["mem", "cpu", "MEM%", "nonesuch"])
    assert [spec.key for spec in specs] == ["cpu", "mem"]


# --- query construction ----------------------------------------------------

def test_a_counter_gets_a_rate_window_wider_than_the_display_step():
    """A range vector sized to exactly one scrape can contain 0-1 samples."""
    spec = SPEC_BY_KEY["cpu"]
    assert spec.query("12345", step=60, sampling_period=60) == (
        "rate(cgroup_cpu_total_seconds{jobid='12345',step='',task=''}[240s])")
    assert "[240s]" in spec.query("12345", step=30, sampling_period=60)
    # A wide display step already covers enough scrapes on its own.
    assert "[1000s]" in spec.query("12345", step=1000, sampling_period=60)


def test_the_rate_window_is_the_documented_multiple_of_the_scrape():
    spec = SPEC_BY_KEY["cpu"]
    assert "[%ds]" % (RATE_LOOKBACK_SCRAPES * 60) in spec.query("1", 60, 60)


def test_a_gauge_is_read_directly_with_no_rate():
    assert SPEC_BY_KEY["mem"].query("12345", step=60, sampling_period=60) == (
        "cgroup_memory_rss_bytes{jobid='12345',step='',task=''}")


@pytest.mark.parametrize("spec", CGROUP_METRICS, ids=lambda s: s.key)
def test_every_query_pins_the_job_level_cgroup(spec):
    """step/task pinned empty selects the job cgroup rather than a per-step one."""
    query = spec.query("12345", step=60, sampling_period=60)
    assert "jobid='12345',step='',task=''" in query
    assert spec.metric in query


# --- host_series -----------------------------------------------------------

def test_host_series_divides_each_sample_by_its_hosts_own_divisor():
    client = FakeClient({
        CPU_SECS: {"nodeA": [(1000, 0.5), (1060, 1.0)]},   # cores in use
        RSS: {"nodeA": [(1000, 4e9), (1060, 8e9)]},        # bytes RSS
    })
    series = host_series("12345", {"nodeA": {"cpus": 4, "total_memory": 8e9}},
                         start=1000, end=1060, step=60, sampling_period=60,
                         client=client, timeout=None)
    # Exact: DEFAULT_CGROUP_SPECS is CPU%/MEM% and nothing else, so a third default
    # metric should break this test rather than slip in unnoticed.
    assert series["nodeA"][1000] == {"CPU%": 12.5, "MEM%": 50.0}
    assert series["nodeA"][1060] == {"CPU%": 25.0, "MEM%": 100.0}


def test_host_series_defaults_to_the_default_group_only():
    client = FakeClient({CPU_SECS: {"nodeA": [(1000, 1.0)]}})
    host_series("12345", {"nodeA": {"cpus": 4}}, 1000, 1060, 60, 60, client, None)
    assert len(client.calls) == len(DEFAULT_CGROUP_SPECS)


def test_host_series_honours_a_widened_spec_list():
    """[metrics.cgroup] can ask for the user/system split, which is --ts only."""
    client = FakeClient({
        CPU_SECS: {"nodeA": [(1000, 2.0)]},
        "cgroup_cpu_user_seconds": {"nodeA": [(1000, 1.5)]},
        "cgroup_cpu_system_seconds": {"nodeA": [(1000, 0.5)]},
    })
    specs = specs_named(["cpu", "cpu_user", "cpu_sys"])
    series = host_series("12345", {"nodeA": {"cpus": 4}}, 1000, 1060, 60, 60,
                         client, None, specs)
    assert series["nodeA"][1000] == {"CPU%": 50.0, "CPU_USER%": 37.5, "CPU_SYS%": 12.5}


def test_a_missing_denominator_skips_only_that_metric():
    """A node reporting cores but not memory still gets its CPU columns."""
    client = FakeClient({
        CPU_SECS: {"nodeA": [(1000, 1.0)]},
        RSS: {"nodeA": [(1000, 4e9)]},
    })
    series = host_series("12345", {"nodeA": {"cpus": 4}},   # no total_memory
                         1000, 1060, 60, 60, client, None)
    assert series["nodeA"][1000] == {"CPU%": 25.0}


def test_host_series_skips_a_host_with_no_resolved_divisor():
    """A host missing from the divisor map is dropped rather than dividing by zero."""
    client = FakeClient({CPU_SECS: {"nodeA": [(1000, 1.0)], "nodeB": [(1000, 1.0)]}})
    series = host_series("12345", {"nodeA": {"cpus": 4}}, 1000, 1060, 60, 60,
                         client, None)
    assert "nodeA" in series and "nodeB" not in series


def test_a_zero_divisor_is_skipped_not_divided_by():
    client = FakeClient({CPU_SECS: {"nodeA": [(1000, 1.0)]}})
    series = host_series("12345", {"nodeA": {"cpus": 0}}, 1000, 1060, 60, 60,
                         client, None)
    assert series == {}


def test_host_series_strips_the_port_from_the_host_label():
    client = FakeClient({CPU_SECS: {"nodeA": [(1000, 1.0)]}})
    series = host_series("12345", {"nodeA": {"cpus": 4}}, 1000, 1060, 60, 60,
                         client, None)
    assert "nodeA" in series and "nodeA:9100" not in series


def test_host_series_tolerates_a_query_failure():
    class Boom(FakeClient):
        def query_range(self, *a, **kw):
            raise RuntimeError("prometheus is down")

    series = host_series("12345", {"nodeA": {"cpus": 4, "total_memory": 8e9}},
                         1000, 1060, 60, 60, Boom(), None)
    assert series == {}


def test_one_failing_metric_leaves_the_others_populated():
    """A partial answer is worth more than none -- the failed column is just absent."""
    class HalfDown(FakeClient):
        def query_range(self, query, *a, **kw):
            if RSS in query:
                raise RuntimeError("prometheus is down")
            return super().query_range(query, *a, **kw)

    client = HalfDown({CPU_SECS: {"nodeA": [(1000, 1.0)]}})
    series = host_series("12345", {"nodeA": {"cpus": 4, "total_memory": 8e9}},
                         1000, 1060, 60, 60, client, None)
    assert series["nodeA"][1000] == {"CPU%": 25.0}
