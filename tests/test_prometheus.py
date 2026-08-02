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


def test_query_success(monkeypatch):
    captured = {}

    def fake_get(url, params, timeout):
        captured.update(url=url, params=params, timeout=timeout)
        return FakeResponse({"status": "success", "data": {"result": [{"value": [1, "2"]}]}})

    monkeypatch.setattr(prometheus.requests, "get", fake_get)
    result = PrometheusClient("http://p", 60, 30).query("up", 1000)
    assert result == [{"value": [1, "2"]}]
    assert captured["url"] == "http://p/api/v1/query"
    assert captured["params"] == {"query": "up", "time": 1000}
    assert captured["timeout"] == 30


def test_query_error_raises(monkeypatch):
    monkeypatch.setattr(prometheus.requests, "get",
                        lambda *a, **k: FakeResponse({"status": "error", "error": "boom"}))
    with pytest.raises(RuntimeError):
        PrometheusClient("http://p", 60, 30).query("up", 1000)


def test_query_range_success(monkeypatch):
    monkeypatch.setattr(prometheus.requests, "get",
                        lambda *a, **k: FakeResponse({"status": "success", "data": {"result": [1, 2]}}))
    assert PrometheusClient("http://p", 60, 30).query_range("up", 0, 10, 5) == [1, 2]


def test_query_range_error_returns_empty(monkeypatch):
    monkeypatch.setattr(prometheus.requests, "get",
                        lambda *a, **k: FakeResponse({"status": "error"}))
    assert PrometheusClient("http://p", 60, 30).query_range("up", 0, 10, 5) == []


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
