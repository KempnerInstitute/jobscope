"""Tests for the cgroup CPU%/MEM% time series (the host analogue of dcgm.py)."""

from jobscope.cpu import RATE_LOOKBACK_SCRAPES, _rate_window, cpu_query, host_series, mem_query


class FakeClient:
    """A Prometheus stand-in driven by canned range-query results, keyed by host."""

    def __init__(self, cpu_values=None, mem_values=None, sampling_period=60):
        self.cpu_values = cpu_values or {}   # {host: [(ts, raw), ...]}
        self.mem_values = mem_values or {}   # {host: [(ts, raw), ...]}
        self.sampling_period = sampling_period
        self.calls = []

    def query_range(self, query, start, end, step, timeout=None):
        self.calls.append(query)
        if "cgroup_cpu_total_seconds" in query:
            values = self.cpu_values
        elif "cgroup_memory_rss_bytes" in query:
            values = self.mem_values
        else:
            return []
        return [{"metric": {"host": "%s:9100" % host}, "values": [[ts, str(v)] for ts, v in pts]}
                for host, pts in values.items()]


def test_rate_window_never_shrinks_below_a_few_scrapes():
    """A range vector sized to exactly one scrape can contain 0-1 samples."""
    assert _rate_window(60, 60) == 60 * RATE_LOOKBACK_SCRAPES
    assert _rate_window(30, 60) == 60 * RATE_LOOKBACK_SCRAPES
    # A wide display step already covers enough scrapes on its own.
    assert _rate_window(1000, 60) == 1000


def test_cpu_query_uses_the_rate_window_not_the_raw_step():
    query = cpu_query("12345", step=60, sampling_period=60)
    assert "rate(cgroup_cpu_total_seconds{jobid='12345',step='',task=''}[240s])" == query


def test_mem_query_has_no_rate_its_a_gauge():
    assert mem_query("12345") == "cgroup_memory_rss_bytes{jobid='12345',step='',task=''}"


def test_host_series_divides_each_sample_by_its_hosts_own_divisor():
    client = FakeClient(
        cpu_values={"nodeA": [(1000, 0.5), (1060, 1.0)]},   # cores in use
        mem_values={"nodeA": [(1000, 4e9), (1060, 8e9)]},   # bytes RSS
    )
    series = host_series("12345", cpus_by_host={"nodeA": 4}, mem_total_by_host={"nodeA": 8e9},
                         start=1000, end=1060, step=60, sampling_period=60,
                         client=client, timeout=None)
    assert series["nodeA"][1000] == {"CPU%": 12.5, "MEM%": 50.0}
    assert series["nodeA"][1060] == {"CPU%": 25.0, "MEM%": 100.0}


def test_host_series_skips_a_host_with_no_resolved_divisor():
    """A host missing from the divisor map is dropped rather than dividing by zero."""
    client = FakeClient(cpu_values={"nodeA": [(1000, 1.0)], "nodeB": [(1000, 1.0)]})
    series = host_series("12345", cpus_by_host={"nodeA": 4}, mem_total_by_host={},
                         start=1000, end=1060, step=60, sampling_period=60,
                         client=client, timeout=None)
    assert "nodeA" in series and "nodeB" not in series


def test_host_series_strips_the_port_from_the_host_label():
    client = FakeClient(cpu_values={"nodeA": [(1000, 1.0)]})
    series = host_series("12345", cpus_by_host={"nodeA": 4}, mem_total_by_host={},
                         start=1000, end=1060, step=60, sampling_period=60,
                         client=client, timeout=None)
    assert "nodeA" in series and "nodeA:9100" not in series


def test_host_series_tolerates_a_query_failure():
    class Boom(FakeClient):
        def query_range(self, *a, **kw):
            raise RuntimeError("prometheus is down")

    series = host_series("12345", {"nodeA": 4}, {"nodeA": 8e9}, 1000, 1060, 60, 60,
                         Boom(), None)
    assert series == {}
