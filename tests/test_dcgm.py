"""Tests for the DCGM metric catalog and the Prometheus join/compute logic."""

import dataclasses
import re

import pytest

from jobscope.dcgm import (
    ALL_SPECS,
    DCGM_HEADERS,
    DEFAULT_SPECS,
    GPU_SUMMARY_SPECS,
    JOBSTATS_BACKED_KEYS,
    KEY_SPECS,
    METRICS,
    MODEL_KEY,
    NAME_LABEL,
    SPEC_BY_HEADER,
    columns_for,
    compute_dcgm,
    dcgm_for_job,
    discover_gpus,
    format_by_header,
    format_value,
    gpu_minor_key,
    group_key,
    grouped_window_query,
    spec_named,
    stored_utilization,
    window_query,
)
from jobscope.jobstats import jobstats_metrics


class FakeClient:
    """A Prometheus stand-in driven by canned GPU discovery and metric values.

    Answers **grouped** queries the way a real server does, which is the whole
    point of the ``label_replace`` handling below: a fake that only knew per-metric
    queries would return nothing for a grouped one, the production code would fall
    back to per-metric, and every test would pass while measuring the path that is
    no longer used. ``grouped`` counts the ones it served, so a test can assert the
    query count actually fell.
    """

    def __init__(self, gpus, values, sampling_period=60, range_values=None,
                 support_grouping=True):
        self.gpus = gpus                       # [(uuid, node, minor)]
        self.values = values                   # {metric_name: {uuid: raw_value}}
        self.range_values = range_values or {}  # {metric_name: {uuid: [(ts, raw)]}}
        self.sampling_period = sampling_period
        # False stands in for a server that cannot do label_replace, to exercise
        # the per-metric fallback.
        self.support_grouping = support_grouping
        self.queries = []
        self.grouped = 0

    def _grouped_names(self, query):
        """The metric names a grouped query asks for, or None if it is per-metric."""
        if "label_replace" not in query:
            return None
        found = re.search(r'__name__=~"\^\(([^)]*)\)\$"', query)
        return found.group(1).split("|") if found else []

    def query(self, query, at, timeout=None):
        self.queries.append(query)
        if "nvidia_gpu_jobId" in query:
            return [{"metric": {"uuid": u, "host": node + ":9400", "minor_number": minor}}
                    for u, node, minor in self.gpus]
        names = self._grouped_names(query)
        if names is not None:
            if not self.support_grouping:
                return []
            self.grouped += 1
            # One row per (metric, card), with the name in the label the real
            # server keeps only because label_replace put it there.
            return [{"metric": {NAME_LABEL: name, "UUID": u}, "value": [at, str(v)]}
                    for name in names for u, v in self.values.get(name, {}).items()]
        for name, per in self.values.items():
            if name in query:
                return [{"metric": {"UUID": u}, "value": [at, str(v)]} for u, v in per.items()]
        return []

    def query_range(self, query, start, end, step, timeout=None):
        self.queries.append(query)
        for name, per in self.range_values.items():
            if name in query:
                return [{"metric": {"UUID": u}, "values": [[ts, str(v)] for ts, v in pts]}
                        for u, pts in per.items()]
        return []


def test_catalog_shape():
    assert len(ALL_SPECS) == 30
    assert len(DEFAULT_SPECS) == 7          # 5 profiling + the GPU memory pair
    assert len(GPU_SUMMARY_SPECS) == 4
    # The summary/detail DCGM columns exclude what the jobstats summary already supplies, so
    # neither GPU% nor the GMEM columns appear twice in those views. OCC% moved to
    # the "all" group -- --all-metrics only -- so it is not part of the default set.
    assert DCGM_HEADERS == ["SM_ACT%", "TENSOR%", "DRAM%", "POWER_W"]
    # METRICS is the *candidates*, so its headers repeat wherever two exporters
    # offer one column. Uniqueness is a property of the resolved view, which is what
    # SPEC_BY_HEADER and every consumer read -- see jobscope.source.
    headers = [spec.header for spec in ALL_SPECS]
    assert len(headers) == len(set(headers))
    assert set(SPEC_BY_HEADER) == set(headers)
    assert len(METRICS) > len(ALL_SPECS)
    assert any(spec.column == "GPU%" for spec in DEFAULT_SPECS)


def test_key_specs_is_the_curated_ts_default():
    """--ts/--plot_ts/--classify's default (no --dcgm/--ext): a small subset of
    DEFAULT_SPECS, not the full 8 -- notably no OCC% or the GPU memory pair."""
    assert [spec.header for spec in KEY_SPECS] == \
        ["GPU%", "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W"]
    assert set(KEY_SPECS) <= set(DEFAULT_SPECS)
    assert all(spec.key not in JOBSTATS_BACKED_KEYS for spec in GPU_SUMMARY_SPECS)


def test_dcgm_and_live_columns_are_identical():
    """A finished job and a running one must be described by the same columns."""
    from jobscope.running import DEFAULT_RUNNING_SPECS, build_columns
    assert columns_for(DEFAULT_SPECS) == build_columns(DEFAULT_RUNNING_SPECS)
    assert [h for _k, h, _d in columns_for(DEFAULT_SPECS)] == [
        "GPU%", "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W", "GMEM_GB", "GMEM%"]


def test_hidden_total_memory_is_queried_but_not_a_column():
    assert any(s.header == "GMEM_TOTAL_GB" for s in DEFAULT_SPECS)
    assert "GMEM_TOTAL_GB" not in [h for _k, h, _d in columns_for(DEFAULT_SPECS)]


def test_gpu_memory_comes_from_the_jobstats_summary_for_a_finished_job(gpu_record):
    """As with GPU%, a stored value is never recomputed -- see _prefer_stored."""
    overall, per_gpu = dcgm_for_job(gpu_record, DEFAULT_SPECS, _client(), None)
    # The summary holds 48 GiB used of 80 total on GPU 0, 32 of 80 on GPU 1.
    assert per_gpu[("node01", "0")]["GMEM_GB"] == 48.0
    assert per_gpu[("node01", "0")]["GMEM%"] == 60.0
    assert per_gpu[("node01", "1")]["GMEM%"] == 40.0
    # Job-level GMEM% sums used over sums total, exactly as jobstats_metrics does.
    assert overall["GMEM%"] == 50.0
    assert overall["GMEM%"] == jobstats_metrics(gpu_record.stats, gpu_record.gpus).value("GMEM%")


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
    assert gpus[0] == {"uuid": "UUID-A", "node": "node01", "minor": "0", "model": ""}


def test_dcgm_for_job_scales_and_aggregates(gpu_record):
    overall, per_gpu = dcgm_for_job(gpu_record, DEFAULT_SPECS, _client(), None)
    assert overall["SM_ACT%"] == 60.0          # mean(80, 40), 0-1 fraction scaled to percent
    assert overall["POWER_W"] == 400.0         # mean(300, 500)
    assert per_gpu[("node01", "0")]["SM_ACT%"] == 80.0
    assert per_gpu[("node01", "1")]["POWER_W"] == 500.0


def test_utilization_comes_from_the_jobstats_summary_not_a_recomputation(gpu_record):
    """A finished job must report the utilization Slurm stored, in every view.

    The fake client serves no duty samples at all, so this value can only have
    come from the jobstats summary -- and it is the same number jobstats_metrics gives the summary
    view, which is the point: the two cannot drift apart.
    """
    overall, per_gpu = dcgm_for_job(gpu_record, DEFAULT_SPECS, _client(), None)
    assert per_gpu[("node01", "0")]["GPU%"] == 90.0   # the summary's own per-GPU values
    assert per_gpu[("node01", "1")]["GPU%"] == 50.0
    assert overall["GPU%"] == 70.0                   # mean(90, 50)
    assert overall["GPU%"] == jobstats_metrics(gpu_record.stats, gpu_record.gpus).value("GPU%")


def test_running_job_keeps_the_prometheus_utilization(gpu_record):
    """With no jobstats summary there is nothing to defer to, so the query stands."""
    running = dataclasses.replace(gpu_record, state="RUNNING", stats={})
    overall, per_gpu = dcgm_for_job(running, DEFAULT_SPECS, _client(), None)
    # _client() serves no duty samples, so the column is simply absent rather
    # than silently borrowed from somewhere else.
    assert "GPU%" not in overall
    assert "GPU%" not in per_gpu[("node01", "0")]


def test_stored_utilization_is_keyed_by_node_and_minor_string(gpu_record):
    assert stored_utilization(gpu_record) == {("node01", "0"): 90.0, ("node01", "1"): 50.0}


def test_stored_utilization_empty_without_a_summary(gpu_record):
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
    # itself is kept, carrying only what the stored summary already knew.
    # GPU% and the GMEM columns survive because the jobstats summary supplies them.
    assert set(overall) == {"GPU%", "GMEM_GB", "GMEM_TOTAL_GB", "GMEM%"}
    assert set(per_gpu) == {("node01", "0")}
    assert set(per_gpu[("node01", "0")]) == {"GPU%", "GMEM_GB", "GMEM_TOTAL_GB", "GMEM%",
                                             MODEL_KEY}


def test_dcgm_for_job_metric_error_on_a_running_job_yields_nothing(gpu_record):
    running = dataclasses.replace(gpu_record, state="RUNNING", stats={})
    overall, per_gpu = dcgm_for_job(running, DEFAULT_SPECS, _DiscoveryOnlyClient(), None)
    assert overall == {}
    # The model rides with the row even when no metric survived: it identifies the
    # card, and POWER_W's floor depends on which one it was.
    assert per_gpu == {("node01", "0"): {MODEL_KEY: ""}}


# --- naming metrics from config ----------------------------------------------

@pytest.mark.parametrize("name,header", [
    ("gpu", "GPU%"), ("GPU", "GPU%"), ("GPU%", "GPU%"), ("duty", "GPU%"),
    ("sm_act", "SM_ACT%"), ("smact", "SM_ACT%"),
    ("power", "POWER_W"), ("power_w", "POWER_W"),
    ("energy", "ENERGY_kWh"), ("temp", "TEMP_C"),
])
def test_a_metric_answers_to_its_key_and_its_header(name, header):
    """Three forms per metric, derived from the catalog rather than listed, so a
    metric added to METRICS is nameable at once."""
    from jobscope.dcgm import spec_named
    assert spec_named(name).header == header


def test_an_unknown_name_resolves_to_nothing():
    from jobscope.dcgm import spec_named
    assert spec_named("gpuu") is None and spec_named("") is None


def test_every_offered_name_actually_resolves():
    """METRIC_NAMES is what an error message tells a user to choose from, so each
    one had better work."""
    from jobscope.dcgm import METRIC_NAMES, spec_named
    assert all(spec_named(n) is not None for n in METRIC_NAMES)
    assert len(METRIC_NAMES) == len(ALL_SPECS)


def test_specs_named_sorts_by_catalog_position():
    from jobscope.dcgm import specs_named
    assert [s.header for s in specs_named(["power", "gpu", "dram"])] == \
        ["GPU%", "DRAM%", "POWER_W"]


def test_specs_named_drops_duplicate_spellings_of_one_metric():
    from jobscope.dcgm import specs_named
    assert [s.header for s in specs_named(["gpu", "GPU%", "duty"])] == ["GPU%"]


def test_specs_named_skips_unknown_names():
    """Validation belongs to the caller, which can say which config key was wrong."""
    from jobscope.dcgm import specs_named
    assert [s.header for s in specs_named(["gpu", "nonsense"])] == ["GPU%"]


def test_the_live_view_drops_counter_deltas():
    """ENERGY_kWh is a difference over a finished window; the running view builds a
    summary from a window that has not finished, so the number would mean nothing."""
    from jobscope.dcgm import specs_named
    assert [s.header for s in specs_named(["gpu", "energy"])] == ["GPU%", "ENERGY_kWh"]
    assert [s.header for s in specs_named(["gpu", "energy"], running=True)] == ["GPU%"]


def test_the_built_in_lists_are_reproducible_by_name():
    """Which is what lets [metrics] express them, and what the shipped example
    config relies on."""
    from jobscope.dcgm import KEY_SPECS, specs_named
    assert [s.header for s in specs_named(["gpu", "sm_act", "tensor", "dram", "power"])] \
        == [s.header for s in KEY_SPECS]


# --- metric grouping --------------------------------------------------------

def test_grouped_query_names_every_metric_and_preserves_the_name_label():
    """label_replace is not decoration: avg_over_time drops __name__, so without it
    a response covering several metrics cannot be attributed to any of them."""
    q = grouped_window_query("avg", "UUID",
                             ["DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_PROF_DRAM_ACTIVE"],
                             ["U1", "U2"], 100)
    assert q == (
        'avg_over_time((label_replace({__name__=~"^(DCGM_FI_PROF_DRAM_ACTIVE|'
        'DCGM_FI_PROF_SM_ACTIVE)$",UUID=~"^(U1|U2)$"},'
        '"jsname","$1","__name__","(.*)"))[100s:])')


def test_the_delta_reducer_label_replaces_inside_both_halves():
    """max_over_time - min_over_time is two selectors; a name preserved in only one
    leaves half the response unattributable."""
    q = grouped_window_query("delta", "UUID", ["A", "B"], ["U1"], 100)
    assert q.count("label_replace") == 2
    assert q.startswith("(max_over_time(") and " - min_over_time(" in q


def test_group_key_splits_on_reducer_and_on_uuid_label():
    """Both matter: the reducer picks the function, and the label differs by family
    -- NVML uses lowercase uuid where DCGM uses uppercase UUID."""
    assert group_key(SPEC_BY_HEADER["SM_ACT%"]) == ("avg", "UUID")
    assert group_key(spec_named("duty")) == ("avg", "uuid")           # NVML
    assert group_key(spec_named("duty_dcgm")) == ("avg", "UUID")      # same column
    assert group_key(SPEC_BY_HEADER["PWRmax_W"]) == ("max", "UUID")
    assert group_key(SPEC_BY_HEADER["ENERGY_kWh"]) == ("delta", "UUID")


def test_grouping_cuts_the_query_count(gpu_record):
    """The whole point. 7 default specs share 3 (reducer, uuid_label) groups, so a
    job costs 1 discovery + 3 instead of 1 + 7."""
    client = _client()
    dcgm_for_job(gpu_record, DEFAULT_SPECS, client, None)
    assert client.grouped == 2      # avg/UUID and max/uuid; avg/uuid is a lone spec
    assert len(client.queries) < 1 + len(DEFAULT_SPECS)


def test_grouped_and_per_spec_produce_identical_values(gpu_record):
    """Batching changes how values are fetched, never which samples reduce into
    them -- so the two paths must agree exactly, not approximately."""
    grouped = dcgm_for_job(gpu_record, DEFAULT_SPECS, _client(), None)
    per_spec = dcgm_for_job(gpu_record, DEFAULT_SPECS,
                            FakeClient(gpus=[("UUID-A", "node01", "0"),
                                             ("UUID-B", "node01", "1")],
                                       values=_client().values,
                                       support_grouping=False), None)
    assert grouped == per_spec


def test_a_server_without_label_replace_falls_back_per_metric(gpu_record):
    """An unusable grouped response must cost a little speed, never a blank column."""
    client = FakeClient(gpus=[("UUID-A", "node01", "0"), ("UUID-B", "node01", "1")],
                        values={"DCGM_FI_PROF_SM_ACTIVE": {"UUID-A": 0.80, "UUID-B": 0.40}},
                        support_grouping=False)
    overall, per_gpu = dcgm_for_job(gpu_record, DEFAULT_SPECS, client, None)
    assert overall["SM_ACT%"] == 60.0
    assert per_gpu[("node01", "0")]["SM_ACT%"] == 80.0


def test_one_series_backing_two_specs_is_not_lost(gpu_record):
    """DCGM_FI_DEV_POWER_USAGE feeds POWER_W (avg) and PWRmax_W (max). They land in
    different groups today, but the metric->specs mapping must stay one-to-many."""
    client = _client()
    overall, _pg = dcgm_for_job(gpu_record, ALL_SPECS, client, None)
    assert overall["POWER_W"] == 400.0        # mean(300, 500)
    assert overall["PWRmax_W"] == 500.0       # max(300, 500)
