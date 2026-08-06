"""Tests for the Prometheus HTTP client (requests mocked)."""

import pytest

from jobscope import prometheus
from jobscope.config import Config, Defaults, Thresholds
from jobscope.prometheus import PrometheusClient, client_from_config


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeSession:
    """Stands in for a requests.Session, recording each call.

    Patched in through ``_make_session`` rather than over ``prometheus.requests``: the
    module no longer imports requests at all, precisely so that ``--help`` need not pay
    for it. ``calls`` is what lets a test assert the connection was *reused*.
    """

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.payload() if callable(self.payload) else self.payload)


def _client(monkeypatch, payload):
    session = FakeSession(payload)
    monkeypatch.setattr(PrometheusClient, "_make_session", lambda self: session)
    return PrometheusClient("http://p", 60, 30), session


def test_query_success(monkeypatch):
    client, session = _client(
        monkeypatch, {"status": "success", "data": {"result": [{"value": [1, "2"]}]}})
    assert client.query("up", 1000) == [{"value": [1, "2"]}]
    assert session.calls[0]["url"] == "http://p/api/v1/query"
    assert session.calls[0]["params"] == {"query": "up", "time": 1000}
    assert session.calls[0]["timeout"] == 30


def test_query_error_raises(monkeypatch):
    client, _ = _client(monkeypatch, {"status": "error", "error": "boom"})
    with pytest.raises(RuntimeError):
        client.query("up", 1000)


def test_query_range_success(monkeypatch):
    client, _ = _client(monkeypatch, {"status": "success", "data": {"result": [1, 2]}})
    assert client.query_range("up", 0, 10, 5) == [1, 2]


def test_query_range_error_returns_empty(monkeypatch):
    client, _ = _client(monkeypatch, {"status": "error"})
    assert client.query_range("up", 0, 10, 5) == []


def test_one_session_serves_every_query_on_a_thread(monkeypatch):
    """The whole point: the connection is reused rather than rebuilt per query. Measured
    against a live endpoint, a fresh connection costs 0.098s and a reused one 0.034s, so
    on a 209-query selection this is most of the wall clock."""
    built = []

    def make(self):
        session = FakeSession({"status": "success", "data": {"result": []}})
        built.append(session)
        return session

    monkeypatch.setattr(PrometheusClient, "_make_session", make)
    client = PrometheusClient("http://p", 60, 30)
    for _ in range(5):
        client.query("up", 1000)
    client.query_range("up", 0, 10, 5)
    assert len(built) == 1               # one connection, not six
    assert len(built[0].calls) == 6


def test_each_thread_gets_its_own_session(monkeypatch):
    """requests.Session is not documented thread-safe and dcgm.compute_dcgm fans out
    across a pool, so the reuse is per thread rather than one shared object."""
    import threading

    built = []
    lock = threading.Lock()

    def make(self):
        session = FakeSession({"status": "success", "data": {"result": []}})
        with lock:
            built.append(session)
        return session

    monkeypatch.setattr(PrometheusClient, "_make_session", make)
    client = PrometheusClient("http://p", 60, 30)
    threads = [threading.Thread(target=lambda: client.query("up", 1000)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 4               # one per thread, none shared


def test_the_module_does_not_import_requests():
    """257ms of a 442ms import, on a path that only the network views need. --help,
    --version, config, describe and every usage error used to pay for it."""
    import sys
    assert not hasattr(prometheus, "requests")
    # The client is constructible without it; only a real query pulls it in.
    assert PrometheusClient("http://p", 60, 30).url == "http://p"
    assert "jobscope.prometheus" in sys.modules


def _cfg(url):
    return Config(prometheus_url=url, sampling_period=45, sampling_period_explicit=True,
                  site_jobstats_config_path=None, thresholds=Thresholds(),
                  defaults=Defaults(workers=8, timeout=90.0))


def test_client_from_config_uses_defaults_timeout():
    client = client_from_config(_cfg("http://p:9090"))
    assert client.url == "http://p:9090"
    assert client.sampling_period == 45
    assert client.timeout == 90.0


def test_client_from_config_timeout_override():
    assert client_from_config(_cfg("http://p:9090"), timeout=5).timeout == 5


# --- pacing ------------------------------------------------------------------

class _Clock:
    """A fake clock, so a limiter test costs no wall time. Sleeping advances it."""

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _limiter(rate, burst):
    clock = _Clock()
    return prometheus.RateLimiter(rate, burst, monotonic=clock.monotonic,
                                  sleep=clock.sleep), clock


def test_the_burst_is_never_delayed():
    """What keeps pacing invisible for ordinary use: an explicit job ID is three queries,
    --verify a handful, a small partition under a hundred. None of them should wait."""
    limiter, clock = _limiter(rate=50, burst=200)
    for _ in range(200):
        limiter.acquire()
    assert clock.slept == []


def test_past_the_burst_it_paces_at_the_rate():
    limiter, clock = _limiter(rate=50, burst=4)
    for _ in range(4):
        limiter.acquire()
    limiter.acquire()
    assert clock.slept == [pytest.approx(1 / 50)]
    limiter.acquire()
    assert len(clock.slept) == 2 and clock.slept[-1] == pytest.approx(1 / 50)


def test_the_bucket_refills_but_does_not_bank_more_than_the_burst():
    """An idle stretch should not buy a second burst: the point is a ceiling on the rate
    the server sees, not on the average."""
    limiter, clock = _limiter(rate=10, burst=3)
    for _ in range(3):
        limiter.acquire()
    clock.now += 100.0                      # long idle -- 1000 tokens' worth of time
    for _ in range(3):
        limiter.acquire()                   # the bucket's worth, free again
    assert clock.slept == []
    limiter.acquire()                       # and no more than that
    assert clock.slept == [pytest.approx(1 / 10)]


def test_a_zero_rate_means_no_limiter_at_all(monkeypatch):
    """0 is a real setting: pace nothing. Expressed as a missing limiter rather than an
    infinite rate, so the query path costs nothing at all to skip it."""
    import dataclasses
    base = _cfg("http://p:9090")
    assert client_from_config(dataclasses.replace(base, max_queries_per_second=0),
                              5).limiter is None
    assert client_from_config(base, 5).limiter is not None


def test_a_client_built_directly_is_not_paced():
    """The default every other test in this file relies on -- and every library caller
    who never asked for pacing."""
    assert PrometheusClient("http://x", 60, 5).limiter is None


def test_a_refill_landing_a_hair_short_does_not_spin():
    """The bug this nearly shipped with. Refilling from two timestamps accumulates float
    error -- monotonic() 100.1 minus 100.0 is 0.09999999999999432 -- so a refill that
    should land on exactly one token lands just under. Comparing for >= 1.0 exactly then
    asked for a 5.7e-15 s wait, below the ULP of the clock reading, so the clock could not
    advance and the bucket span forever, sleeping ~0.
    """
    limiter, clock = _limiter(rate=10, burst=1)
    limiter.acquire()                       # empties the bucket at t=0
    clock.now = 100.0                       # exactly one token's worth, in awkward floats
    clock.now += 0.1
    limiter.acquire()                       # must return, not spin
    assert len(clock.slept) <= 1, clock.slept
