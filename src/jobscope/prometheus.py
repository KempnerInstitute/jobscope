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
import time
from typing import Callable, List, Optional

from . import config


# A token is "whole enough" within this much. Refilling from two timestamps accumulates
# float error -- monotonic() 100.1 minus 100.0 is 0.09999999999999432, so a refill that
# should land on exactly one token lands a hair under it. Comparing for >= 1.0 exactly then
# computes a wait of 5.7e-15 s, which is below the ULP of the clock reading and so cannot
# advance it: the bucket spins, sleeping ~0 forever. A billionth of a token is far below
# anything the rate means and far above the error.
_TOKEN_EPS = 1e-9

# And a floor under the computed wait, so even a pathological rate cannot turn the loop
# into a busy-spin against a clock that has not moved.
_MIN_WAIT = 1e-4


class RateLimiter:
    """A token bucket: ``burst`` queries at full speed, then ``rate`` per second.

    The running view averages each job over its own runtime, and PromQL cannot vary a
    window per series, so that is one query per (job, metric) -- 628 queries for one
    110-job partition where the newest scrape is 12 for the whole selection. Unpaced,
    those arrive as fast as the worker pool can issue them, and the shared Prometheus
    absorbs the burst.

    ``burst`` is what keeps this invisible for ordinary use rather than a tax on it: an
    explicit job ID is three queries, ``--verify`` a handful, a small partition under a
    hundred -- all inside the bucket and never delayed. Only a genuinely wide sweep
    reaches the steady rate, and by then its rows are already streaming, so the pacing
    costs wall clock the reader can watch rather than a blank screen.

    Pacing spreads the queries out; it does not remove any. ``--min-elapsed`` is the only
    thing that reduces the count, by reducing the job count.

    Thread-safe, because :func:`jobscope.running.collect_averaged` drives it from a
    ThreadPoolExecutor. ``monotonic``/``sleep`` are injected so tests can drive a fake
    clock instead of waiting: a limiter test that really slept would add seconds to the
    suite for no extra confidence.
    """

    def __init__(self, rate: float, burst: int,
                 monotonic: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.rate = float(rate)
        self.burst = max(1, int(burst))
        self._monotonic = monotonic
        self._sleep = sleep
        self._tokens = float(self.burst)
        self._last = monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is free, then take it."""
        while True:
            with self._lock:
                now = self._monotonic()
                # Refill for the elapsed time, capped at the bucket size so an idle
                # stretch cannot bank unlimited credit.
                self._tokens = min(float(self.burst),
                                   self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0 - _TOKEN_EPS:
                    self._tokens = max(0.0, self._tokens - 1.0)
                    return
                # How long until the next whole token, computed inside the lock so two
                # threads cannot both decide the same token is theirs.
                wait = ((1.0 - self._tokens) / self.rate if self.rate > 0
                        else _MIN_WAIT * 10)
            self._sleep(max(wait, _MIN_WAIT))


class PrometheusClient:
    """Minimal wrapper over the Prometheus HTTP query API.

    Holds a connection per thread, so a run pays one TLS handshake per worker rather
    than one per query. Measured against this cluster's endpoint, the same query costs
    0.098s on a fresh connection and 0.034s on a reused one -- roughly 65% of a query
    was handshake, repeated 209 times on a 109-job selection.
    """

    def __init__(self, url: str, sampling_period: int, timeout: Optional[float],
                 limiter: Optional[RateLimiter] = None):
        self.url = url
        self.sampling_period = sampling_period
        self.timeout = timeout
        # None by default so a directly-constructed client -- which is what most tests
        # build -- never waits. client_from_config installs one from [prometheus].
        self.limiter = limiter
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
        if self.limiter is not None:
            self.limiter.acquire()
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
        if self.limiter is not None:
            self.limiter.acquire()
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
    limiter = (RateLimiter(cfg.max_queries_per_second, cfg.query_burst)
               if cfg.max_queries_per_second > 0 else None)
    return PrometheusClient(url, sampling_period, timeout, limiter)


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
