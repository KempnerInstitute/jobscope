"""Tests for the DCGM metric catalog and the Prometheus join/compute logic."""

import dataclasses

from jobscope.blob import blob_metrics
from jobscope.dcgm import (
    ALL_SPECS,
    DCGM_HEADERS,
    DEFAULT_SPECS,
    GPU_SUMMARY_SPECS,
    METRICS,
    SPEC_BY_HEADER,
    compute_dcgm,
    dcgm_for_job,
    discover_gpus,
    format_by_header,
    format_value,
    gpu_minor_key,
    stored_utilization,
    window_query,
)


class FakeClient:
    """A Prometheus stand-in driven by canned GPU discovery and metric values."""

    def __init__(self, gpus, values, sampling_period=60, range_values=None):
        self.gpus = gpus                       # [(uuid, node, minor)]
        self.values = values                   # {metric_name: {uuid: raw_value}}
        self.range_values = range_values or {}  # {metric_name: {uuid: [(ts, raw)]}}
        self.sampling_period = sampling_period

    def query(self, query, at, timeout=None):
        if "nvidia_gpu_jobId" in query:
            return [{"metric": {"uuid": u, "host": node + ":9400", "minor_number": minor}}
                    for u, node, minor in self.gpus]
        for name, per in self.values.items():
            if name in query:
                return [{"metric": {"UUID": u}, "value": [at, str(v)]} for u, v in per.items()]
        return []

    def query_range(self, query, start, end, step, timeout=None):
        for name, per in self.range_values.items():
            if name in query:
                return [{"metric": {"UUID": u}, "values": [[ts, str(v)] for ts, v in pts]}
                        for u, pts in per.items()]
        return []


def test_catalog_shape():
    assert len(ALL_SPECS) == 28
    assert len(DEFAULT_SPECS) == 6
    assert len(GPU_SUMMARY_SPECS) == 5
    assert DCGM_HEADERS == ["SM_ACT%", "OCC%", "TENSOR%", "DRAM%", "POWER_W"]
    headers = [spec.header for spec in METRICS]
    assert len(headers) == len(set(headers))
    assert set(SPEC_BY_HEADER) == set(headers)
    assert any(spec.key == "duty" for spec in DEFAULT_SPECS)
    assert all(spec.key != "duty" for spec in GPU_SUMMARY_SPECS)


def test_format_value():
    assert format_value(SPEC_BY_HEADER["SM_ACT%"], 60.0) == "60.0"
    assert format_value(SPEC_BY_HEADER["POWER_W"], 400.6) == "401"
    assert format_value(SPEC_BY_HEADER["POWER_W"], None) == "-"
    assert format_by_header("SM_ACT%", None) == "-"


def test_gpu_minor_key():
    assert gpu_minor_key("3") == 3
    assert gpu_minor_key("x") == "x"


def test_window_query_reducers():
    assert window_query(SPEC_BY_HEADER["SM_ACT%"], ["U1", "U2"], 100) == \
        'avg_over_time((DCGM_FI_PROF_SM_ACTIVE{UUID=~"^(U1|U2)$"})[100s:])'
    assert window_query(SPEC_BY_HEADER["PWRmax_W"], ["U1"], 100) == \
        'max_over_time((DCGM_FI_DEV_POWER_USAGE{UUID=~"^(U1)$"})[100s:])'
    assert window_query(SPEC_BY_HEADER["ENERGY_kWh"], ["U1"], 100) == \
        ('(max_over_time((DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{UUID=~"^(U1)$"})[100s:]) - '
         'min_over_time((DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{UUID=~"^(U1)$"})[100s:]))')


def _client():
    return FakeClient(
        gpus=[("UUID-A", "node01", "0"), ("UUID-B", "node01", "1")],
        values={"DCGM_FI_PROF_SM_ACTIVE": {"UUID-A": 0.80, "UUID-B": 0.40},
                "DCGM_FI_DEV_POWER_USAGE": {"UUID-A": 300.0, "UUID-B": 500.0}})


def test_discover_gpus_sorted(gpu_record):
    gpus = discover_gpus(gpu_record, _client(), None)
    assert [g["minor"] for g in gpus] == ["0", "1"]
    assert gpus[0] == {"uuid": "UUID-A", "node": "node01", "minor": "0"}


def test_dcgm_for_job_scales_and_aggregates(gpu_record):
    overall, per_gpu = dcgm_for_job(gpu_record, DEFAULT_SPECS, _client(), None)
    assert overall["SM_ACT%"] == 60.0          # mean(80, 40), 0-1 fraction scaled to percent
    assert overall["POWER_W"] == 400.0         # mean(300, 500)
    assert per_gpu[("node01", "0")]["SM_ACT%"] == 80.0
    assert per_gpu[("node01", "1")]["POWER_W"] == 500.0


def test_utilization_comes_from_the_blob_not_a_recomputation(gpu_record):
    """A finished job must report the utilization Slurm stored, in every view.

    The fake client serves no duty samples at all, so this value can only have
    come from the blob -- and it is the same number blob_metrics gives the summary
    view, which is the point: the two cannot drift apart.
    """
    overall, per_gpu = dcgm_for_job(gpu_record, DEFAULT_SPECS, _client(), None)
    assert per_gpu[("node01", "0")]["GPU%"] == 90.0   # the blob's own per-GPU values
    assert per_gpu[("node01", "1")]["GPU%"] == 50.0
    assert overall["GPU%"] == 70.0                   # mean(90, 50)
    assert overall["GPU%"] == blob_metrics(gpu_record.stats)[2]


def test_running_job_keeps_the_prometheus_utilization(gpu_record):
    """With no blob there is nothing to defer to, so the query stands."""
    running = dataclasses.replace(gpu_record, state="RUNNING", stats={})
    overall, per_gpu = dcgm_for_job(running, DEFAULT_SPECS, _client(), None)
    # _client() serves no duty samples, so the column is simply absent rather
    # than silently borrowed from somewhere else.
    assert "GPU%" not in overall
    assert "GPU%" not in per_gpu[("node01", "0")]


def test_stored_utilization_is_keyed_by_node_and_minor_string(gpu_record):
    assert stored_utilization(gpu_record) == {("node01", "0"): 90.0, ("node01", "1"): 50.0}


def test_stored_utilization_empty_without_a_blob(gpu_record):
    assert stored_utilization(dataclasses.replace(gpu_record, stats={})) == {}


def test_dcgm_for_job_cpu_only(cpu_record):
    assert dcgm_for_job(cpu_record, DEFAULT_SPECS, _client(), None) == ({}, {})


def test_compute_dcgm_serial(gpu_record):
    results = compute_dcgm({"100": gpu_record}, ["100"], DEFAULT_SPECS, _client(), None, workers=1)
    assert set(results) == {"100"}
    assert results["100"][0]["SM_ACT%"] == 60.0


def test_compute_dcgm_threaded(gpu_record, cpu_record):
    records = {"100": gpu_record, "101": gpu_record, "200": cpu_record}
    results = compute_dcgm(records, ["100", "101", "200"], DEFAULT_SPECS, _client(), None, workers=4)
    assert set(results) == {"100", "101"}      # cpu job skipped
    assert results["101"][0]["POWER_W"] == 400.0


class _RaisingClient:
    sampling_period = 60

    def query(self, *a, **k):
        raise RuntimeError("prometheus down")


class _DiscoveryOnlyClient:
    """Discovery succeeds, but every metric query raises."""

    sampling_period = 60

    def query(self, query, at, timeout=None):
        if "nvidia_gpu_jobId" in query:
            return [{"metric": {"uuid": "U0", "host": "node01:9400", "minor_number": "0"}}]
        raise RuntimeError("metric down")


def test_dcgm_for_job_prometheus_down(gpu_record):
    assert dcgm_for_job(gpu_record, DEFAULT_SPECS, _RaisingClient(), None) == ({}, {})


def test_discover_gpus_prometheus_down(gpu_record):
    assert discover_gpus(gpu_record, _RaisingClient(), None) == []


def test_dcgm_for_job_no_uuids(gpu_record):
    client = FakeClient(gpus=[], values={})
    assert dcgm_for_job(gpu_record, DEFAULT_SPECS, client, None) == ({}, {})


def test_dcgm_for_job_metric_error_keeps_gpu(gpu_record):
    overall, per_gpu = dcgm_for_job(gpu_record, DEFAULT_SPECS, _DiscoveryOnlyClient(), None)
    # Every metric query fails, so nothing Prometheus-derived survives; the GPU row
    # itself is kept, carrying only what the stored blob already knew.
    assert set(overall) == {"GPU%"}
    assert set(per_gpu) == {("node01", "0")}
    assert set(per_gpu[("node01", "0")]) == {"GPU%"}


def test_dcgm_for_job_metric_error_on_a_running_job_yields_nothing(gpu_record):
    running = dataclasses.replace(gpu_record, state="RUNNING", stats={})
    overall, per_gpu = dcgm_for_job(running, DEFAULT_SPECS, _DiscoveryOnlyClient(), None)
    assert overall == {}
    assert per_gpu == {("node01", "0"): {}}
