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
