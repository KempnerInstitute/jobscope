"""Thin Prometheus HTTP client for the DCGM metric queries.

Only the GPU and DCGM views reach the network; the CPU-only and offline jobstats
views never construct a client. The endpoint (which may carry a credential) is
resolved from configuration and never logged -- ``jobscope probe`` is the sole
caller that shows it, and passes it through :func:`jobscope.config.redact_url`
first.

**``requests`` is not imported at module scope**, and must not be: it costs 257ms of
jobscope's import, and every path that never touches the network -- ``--help``,
``--version``, ``config``, ``describe``, every usage error -- used to pay it. The import
lives in :meth:`PrometheusClient._make_session`; ``tests/test_cli.py`` asserts it stays
off the startup path.
"""

import threading
from typing import List, Optional

from . import config


class PrometheusClient:
    """Minimal wrapper over the Prometheus HTTP query API.

    Holds a connection per thread, so a run pays one TLS handshake per worker rather
    than one per query. Measured against this cluster's endpoint, the same query costs
    0.098s on a fresh connection and 0.034s on a reused one -- roughly 65% of a query
    was handshake, repeated 209 times on a 109-job selection.
    """

    def __init__(self, url: str, sampling_period: int, timeout: Optional[float]):
        self.url = url
        self.sampling_period = sampling_period
        self.timeout = timeout
        # Per thread, not one shared: requests.Session is not documented thread-safe and
        # dcgm.compute_dcgm fans out across a pool. One session each keeps the reuse
        # without needing to reason about a shared connection pool. Never closed
        # explicitly -- the process is a CLI invocation, and a session's sockets go with
        # it; a shutdown hook would be ceremony for no gain.
        self._local = threading.local()

    def _make_session(self):
        """This thread's HTTP session. Overridden in tests to avoid the network.

        ``requests`` is imported here rather than at module scope because it costs
        257ms of jobscope's 442ms import and only this path needs it -- ``--help``,
        ``--version``, ``config``, ``describe`` and every usage error paid for it. The
        same lazy pattern :mod:`jobscope.plot` uses for plotext and rich.
        """
        import requests
        return requests.Session()

    @property
    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._make_session()
            self._local.session = session
        return session

    def query(self, query: str, at, timeout: Optional[float] = None) -> List[dict]:
        """Run an instant query at epoch ``at``; return the result list.

        Raises :class:`RuntimeError` when Prometheus reports a non-success status.
        """
        resp = self._session.get(
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
        resp = self._session.get(
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


def query_value(client: PrometheusClient, query: str, at, timeout: Optional[float]):
    """``[(labels, value), ...]`` for an instant query; ``[]`` on any failure.

    The collectors read one figure per series and do not want the envelope. They
    also must not abort a whole report because one job's query failed: a value the
    server cannot answer for becomes an absent column, which the report already
    renders as ``-`` rather than as a zero.
    """
    try:
        found = client.query(query, at, timeout)
    except Exception:
        return []
    out = []
    for series in found:
        try:
            out.append((series["metric"], float(series["value"][1])))
        except (KeyError, IndexError, TypeError, ValueError):
            out.append((series.get("metric", {}), None))
    return out
