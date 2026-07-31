"""Shared fixtures and helpers for the jobscope test suite."""

import base64
import gzip
import json

import pytest

from jobscope import config as config_module
from jobscope.blob import GIB
from jobscope.sacct import JobRecord


def make_blob(stats: dict) -> str:
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
    thresholds=config_module.Thresholds(gpu=25, gmem=20, cpu=25, mem=25, default=15),
    # admin_group empty so tests exercise the views without needing a group;
    # the gate itself is tested explicitly in test_cli.py.
    defaults=config_module.Defaults(workers=8, timeout=60.0, min_runtime=180,
                                    admin_group=""),
)


@pytest.fixture(autouse=True)
def hermetic_config():
    """Pin the process-wide config to known defaults so tests never read ~/.config."""
    config_module.set_config(DEFAULT_CONFIG)
    yield
    config_module.reset_config()


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
