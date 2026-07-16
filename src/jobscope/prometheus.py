"""Thin Prometheus HTTP client for the DCGM metric queries.

Only the GPU and DCGM views reach the network; the CPU-only and offline blob
views never construct a client. The endpoint (which may carry a credential) is
resolved from configuration and never logged.
"""

from typing import List, Optional

import requests

from . import config


class PrometheusClient:
    """Minimal wrapper over the Prometheus HTTP query API."""

    def __init__(self, url: str, sampling_period: int, timeout: Optional[float]):
        self.url = url
        self.sampling_period = sampling_period
        self.timeout = timeout

    def query(self, query: str, at, timeout: Optional[float] = None) -> List[dict]:
        """Run an instant query at epoch ``at``; return the result list.

        Raises :class:`RuntimeError` when Prometheus reports a non-success status.
        """
        resp = requests.get(
            self.url + "/api/v1/query",
            params={"query": query, "time": at},
            timeout=self.timeout if timeout is None else timeout,
        )
        payload = resp.json()
        if payload.get("status") != "success":
            raise RuntimeError(payload.get("error", "prometheus query failed"))
        return payload["data"]["result"]

    def query_range(self, query: str, start, end, step,
                    timeout: Optional[float] = None) -> List[dict]:
        """Run a range query; return the result list, or [] on any non-success."""
        resp = requests.get(
            self.url + "/api/v1/query_range",
            params={"query": query, "start": start, "end": end, "step": step},
            timeout=self.timeout if timeout is None else timeout,
        )
        payload = resp.json()
        return payload["data"]["result"] if payload.get("status") == "success" else []


def client_from_config(cfg: Optional[config.Config] = None,
                       timeout: Optional[float] = None) -> PrometheusClient:
    """Build a client from configuration, resolving the endpoint on demand."""
    cfg = config.get_config() if cfg is None else cfg
    url, sampling_period = config.resolve_prometheus(cfg)
    if timeout is None:
        timeout = cfg.defaults.timeout
    return PrometheusClient(url, sampling_period, timeout)
