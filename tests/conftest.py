"""Shared fixtures and helpers for the jobscope test suite."""

import base64
import gzip
import json

import pytest

from jobscope import config as config_module
from jobscope import source
from jobscope import report, running, slurm
from jobscope.jobstats import GIB
from jobscope.slurm import JobRecord


def make_jobstats(stats: dict) -> str:
    """Encode a stats dict as the JS1:<base64 gzip JSON> AdminComment form."""
    return "JS1:" + base64.b64encode(gzip.compress(json.dumps(stats).encode())).decode()


# A two-GPU job on one node. Chosen so every derived metric is a clean number:
# cpu = 100*150/(100*2) = 75, mem = 100*8/16 = 50, gpu = (90+50)/2 = 70,
# gmem = 100*(48+32)/(80+80) = 50.
GPU_STATS = {
    "total_time": 100,
    "nodes": {
        "node01": {
            "total_time": 150,
            "cpus": 2,
            "used_memory": 8 * GIB,
            "total_memory": 16 * GIB,
            "gpu_utilization": {"0": 90, "1": 50},
            "gpu_used_memory": {"0": 48 * GIB, "1": 32 * GIB},
            "gpu_total_memory": {"0": 80 * GIB, "1": 80 * GIB},
        }
    },
}

# A CPU-only job: cpu = 100*50/(100*1) = 50, mem = 100*4/8 = 50, gpu/gmem absent.
CPU_STATS = {
    "total_time": 100,
    "nodes": {
        "node02": {
            "total_time": 50,
            "cpus": 1,
            "used_memory": 4 * GIB,
            "total_memory": 8 * GIB,
        }
    },
}

DEFAULT_CONFIG = config_module.Config(
    prometheus_url=None,
    sampling_period=60,
    sampling_period_explicit=False,
    site_jobstats_config_path=None,
    thresholds=config_module.Thresholds(),
    defaults=config_module.Defaults(workers=8, timeout=60.0),
)


def _reset_catalogs() -> None:
    """Built-in metrics under the built-in source order -- a complete catalog reset.

    Both halves matter: resetting the definitions while keeping whatever preference the
    last test named leaves the next one resolving columns to a source it never asked
    for, which is why test_source.py used to carry its own restore fixture.
    """
    config_module.build_catalogs({}, source.DEFAULT_PREFERENCE,
                                 source.DEFAULT_HOST_PREFERENCE)


def _clear_hostlist_caches() -> None:
    """Forget memoised node-list expansions between tests.

    Both caches are keyed by the compressed list, which is a stable key in production but
    not under test: a case that monkeypatches ``expand_nodelist`` would otherwise be served
    whatever an earlier test cached for the same string, making the pair order-dependent.
    """
    slurm._HOSTLIST.clear()
    running._JOB_NODES.clear()


@pytest.fixture(autouse=True)
def hermetic_config():
    """Pin the process-wide config to known defaults so tests never read ~/.config.

    Two other pieces of state get reset with it, both installed by a config load:

    * The palette -- ``report._SGR`` is set by ``cli._apply_config``, so a test that
      runs a command with a configured ``[colors]`` would otherwise leave every
      later test painting in its colours.
    * The metric catalogs -- both the ``[metrics.<family>.<name>]`` definitions and
      the ``[gpu]``/``[host]`` source order land on ``dcgm.catalog()`` and
      ``cpu.catalog()``, so a test that defines a site metric or names a source would
      otherwise leak it into every test after it, and into the catalog-shape
      assertions in particular. One slot each, replaced wholesale, so putting back
      the built-ins under the default order is the whole reset.
    """
    _reset_catalogs()
    config_module.set_config(DEFAULT_CONFIG)
    report.set_palette(DEFAULT_CONFIG.palette)
    _clear_hostlist_caches()
    yield
    config_module.reset_config()
    report.set_palette(config_module.Palette())
    _reset_catalogs()
    _clear_hostlist_caches()


@pytest.fixture
def gpu_record():
    return JobRecord(jobid="100", state="COMPLETED", name="train", runtime="01:00:00",
                     nodes="1", gpus=2, stats=GPU_STATS, start=1000, end=1100, duration=100,
                     jobid_raw="100", cluster="odyssey", user="alice")


@pytest.fixture
def cpu_record():
    return JobRecord(jobid="200", state="COMPLETED", name="prep", runtime="00:30:00",
                     nodes="1", gpus=0, stats=CPU_STATS, start=2000, end=2100, duration=100,
                     jobid_raw="200", cluster="odyssey", user="bob")
