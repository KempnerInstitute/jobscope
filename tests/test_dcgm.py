"""Tests for the DCGM metric catalog and the Prometheus join/compute logic."""

import dataclasses
import re

import pytest

from jobscope import dcgm
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
    JobGpuData,
    columns_for,
    compute_dcgm,
    dcgm_for_job,
    discover_gpus,
    format_by_header,
    format_value,
    gpu_minor_key,
    group_key,
    grouped_window_query,
    pool_and_derive,
    pool_uuids,
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
        # Discovery, and only when the join *is* the selector. An nvml metric query also
        # names it, intersected after an `and` -- that is dcgm.ownership_clip -- so a bare
        # substring test served every nvml metric query the value-less discovery response.
        if "nvidia_gpu_jobId" in query and " and " not in query:
            return [{"metric": {"uuid": u, "host": node + ":9400", "minor_number": minor}}
                    for u, node, minor in self.gpus]
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
    """--ts/--plot_ts/--eff's default (no --all-metrics): a small subset of
    DEFAULT_SPECS, not the full 8 -- notably no OCC% or the GPU memory pair."""
    assert [spec.header for spec in KEY_SPECS] == \
        ["GPU%", "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W"]
    assert set(KEY_SPECS) <= set(DEFAULT_SPECS)
    assert all(spec.key not in JOBSTATS_BACKED_KEYS for spec in GPU_SUMMARY_SPECS)


def test_dcgm_and_live_columns_are_identical():
    """A finished job and a running one must be described by the same columns."""
    from jobscope.running import build_columns, default_running_specs
    assert columns_for(DEFAULT_SPECS) == build_columns(default_running_specs())
    assert [h for _k, h, _d in columns_for(DEFAULT_SPECS)] == [
        "GPU%", "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W", "GMEM_GB", "GMEM%"]


def test_hidden_total_memory_is_queried_but_not_a_column():
    assert any(s.header == "GMEM_TOTAL_GB" for s in DEFAULT_SPECS)
    assert "GMEM_TOTAL_GB" not in [h for _k, h, _d in columns_for(DEFAULT_SPECS)]


def test_gpu_memory_comes_from_the_jobstats_summary_for_a_finished_job(gpu_record):
    """As with GPU%, a stored value is never recomputed -- see _prefer_stored."""
    overall, per_gpu, _nodes = dcgm_for_job(gpu_record, DEFAULT_SPECS, _client(), None)
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


def test_instant_drops_the_reducer_for_an_unfinished_job():
    """The newest scrape, which is the same selector running.collect_instant builds --
    that identity is what makes a running job read the same via -j as via squeue."""
    assert window_query(SPEC_BY_HEADER["SM_ACT%"], ["U1", "U2"], 100, instant=True) == \
        'DCGM_FI_PROF_SM_ACTIVE{UUID=~"^(U1|U2)$"}'
    assert window_query(SPEC_BY_HEADER["PWRmax_W"], ["U1"], 100, instant=True) == \
        'DCGM_FI_DEV_POWER_USAGE{UUID=~"^(U1)$"}'


def test_instant_does_not_apply_to_a_counter_difference():
    """ENERGY_kWh is a delta. "The newest scrape" of a counter difference is not a
    smaller answer, it is none at all -- one sample has no difference to report -- so
    delta keeps its window even when everything beside it goes instant."""
    windowed = window_query(SPEC_BY_HEADER["ENERGY_kWh"], ["U1"], 100)
    assert window_query(SPEC_BY_HEADER["ENERGY_kWh"], ["U1"], 100, instant=True) == windowed


def _queries_for(record, **kw):
    """Every query dcgm_for_job issues for ``record``, so the window can be asserted."""
    asked = []

    class Recorder(FakeClient):
        def query(self, query, at, timeout=None):
            asked.append(query)
            return super().query(query, at, timeout)

    client = Recorder(gpus=[("UUID-A", "node01", "0")], values={})
    dcgm_for_job(record, DEFAULT_SPECS, client, None, **kw)
    # The discovery query keeps its window whatever the state -- it asks which cards the
    # job held, which is a fact about the run and not a reading.
    return [q for q in asked if "nvidia_gpu_jobId" not in q]


def test_a_finished_job_is_folded_over_its_runtime(gpu_record):
    """The reason the historical path folds at all: it makes the numbers comparable to
    the summary jobstats stores. Must not change."""
    assert all("_over_time" in q for q in _queries_for(gpu_record))


def test_an_unfinished_job_takes_the_newest_scrape(gpu_record):
    """The report: `jobscope -j <running job>` folded over the whole runtime while every
    other view of the same job showed the newest scrape, so the two disagreed with
    nothing on screen to say why. The window follows the record now."""
    running = dataclasses.replace(gpu_record, state="RUNNING", stats={})
    queries = _queries_for(running)
    assert queries and not any("_over_time" in q for q in queries)


def test_avg_folds_an_unfinished_job_after_all(gpu_record):
    """--runtime-avg is how the old behaviour stays reachable, and it is no longer refused for a
    job selected by an explicit JOBID."""
    running = dataclasses.replace(gpu_record, state="RUNNING", stats={})
    assert all("_over_time" in q for q in _queries_for(running, average=True))


@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "SUSPENDED", "REQUEUED",
                                   "CANCELLED by 64336", "COMPLETED", "TIMEOUT"])
def test_unfinished_is_decided_by_the_state_not_the_selection(gpu_record, state):
    """One definition, shared with the sacct filter -- see slurm.UNFINISHED_STATES. The
    decorated form matters: sacct writes "CANCELLED by 64336", which has ended."""
    record = dataclasses.replace(gpu_record, state=state)
    folded = all("_over_time" in q for q in _queries_for(record))
    assert folded is not record.unfinished


def test_grouped_query_stays_demultiplexable_when_instant():
    """label_replace sits inside the selector, so NAME_LABEL survives without the
    reduction -- the reduction is what drops __name__, and its absence cannot
    reintroduce that problem."""
    q = grouped_window_query("avg", "UUID", ["A", "B"], ["U1"], 100, instant=True)
    assert "_over_time" not in q
    assert NAME_LABEL in q


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
    overall, per_gpu, _nodes = dcgm_for_job(gpu_record, DEFAULT_SPECS, _client(), None)
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
    overall, per_gpu, _nodes = dcgm_for_job(gpu_record, DEFAULT_SPECS, _client(), None)
    assert per_gpu[("node01", "0")]["GPU%"] == 90.0   # the summary's own per-GPU values
    assert per_gpu[("node01", "1")]["GPU%"] == 50.0
    assert overall["GPU%"] == 70.0                   # mean(90, 50)
    assert overall["GPU%"] == jobstats_metrics(gpu_record.stats, gpu_record.gpus).value("GPU%")


def test_running_job_keeps_the_prometheus_utilization(gpu_record):
    """With no jobstats summary there is nothing to defer to, so the query stands."""
    running = dataclasses.replace(gpu_record, state="RUNNING", stats={})
    overall, per_gpu, _nodes = dcgm_for_job(running, DEFAULT_SPECS, _client(), None)
    # _client() serves no duty samples, so the column is simply absent rather
    # than silently borrowed from somewhere else.
    assert "GPU%" not in overall
    assert "GPU%" not in per_gpu[("node01", "0")]


def test_stored_utilization_is_keyed_by_node_and_minor_string(gpu_record):
    assert stored_utilization(gpu_record) == {("node01", "0"): 90.0, ("node01", "1"): 50.0}


def test_stored_utilization_empty_without_a_summary(gpu_record):
    assert stored_utilization(dataclasses.replace(gpu_record, stats={})) == {}


def test_dcgm_for_job_cpu_only(cpu_record):
    assert dcgm_for_job(cpu_record, DEFAULT_SPECS, _client(), None) == JobGpuData()


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
    assert dcgm_for_job(gpu_record, DEFAULT_SPECS, _RaisingClient(), None) == JobGpuData()


def test_discover_gpus_prometheus_down(gpu_record):
    assert discover_gpus(gpu_record, _RaisingClient(), None) == []


def test_dcgm_for_job_no_uuids(gpu_record):
    client = FakeClient(gpus=[], values={})
    assert dcgm_for_job(gpu_record, DEFAULT_SPECS, client, None) == JobGpuData()


def test_dcgm_for_job_metric_error_keeps_gpu(gpu_record):
    overall, per_gpu, _nodes = dcgm_for_job(gpu_record, DEFAULT_SPECS, _DiscoveryOnlyClient(), None)
    # Every metric query fails, so nothing Prometheus-derived survives; the GPU row
    # itself is kept, carrying only what the stored summary already knew.
    # GPU% and the GMEM columns survive because the jobstats summary supplies them.
    assert set(overall) == {"GPU%", "GMEM_GB", "GMEM_TOTAL_GB", "GMEM%"}
    assert set(per_gpu) == {("node01", "0")}
    assert set(per_gpu[("node01", "0")]) == {"GPU%", "GMEM_GB", "GMEM_TOTAL_GB", "GMEM%",
                                             MODEL_KEY}


def test_dcgm_for_job_metric_error_on_a_running_job_yields_nothing(gpu_record):
    running = dataclasses.replace(gpu_record, state="RUNNING", stats={})
    overall, per_gpu, _nodes = dcgm_for_job(running, DEFAULT_SPECS, _DiscoveryOnlyClient(), None)
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
    overall, per_gpu, _nodes = dcgm_for_job(gpu_record, DEFAULT_SPECS, client, None)
    assert overall["SM_ACT%"] == 60.0
    assert per_gpu[("node01", "0")]["SM_ACT%"] == 80.0


def test_one_series_backing_two_specs_is_not_lost(gpu_record):
    """DCGM_FI_DEV_POWER_USAGE feeds POWER_W (avg) and PWRmax_W (max). They land in
    different groups today, but the metric->specs mapping must stay one-to-many."""
    client = _client()
    overall, _pg, _nodes = dcgm_for_job(gpu_record, ALL_SPECS, client, None)
    assert overall["POWER_W"] == 400.0        # mean(300, 500)
    assert overall["PWRmax_W"] == 500.0       # max(300, 500)


# --- the label-based mapping on the finished path -------------------------------
#
# Synthetic throughout: hpc_job is not published on this cluster, so this path has never
# run against real data. See jobscope.config.Site on why a second mapping is worth having.

def _two_mapping_client(join_rows, labelled_rows, label="hpc_job"):
    class Client:
        sampling_period = 60

        def query(self, query, at, timeout=None):
            if label in query:
                return [{"metric": {"UUID": u, "gpu": str(g), "host": h,
                                    "modelName": "NVIDIA H200"}, "value": [at, "1"]}
                        for h, u, g in labelled_rows]
            return [{"metric": {"host": h, "uuid": u, "minor_number": str(m),
                                "name": "NVIDIA H200"}, "value": [at, "1"]}
                    for h, u, m in join_rows]

    return Client()


@pytest.fixture
def hpc_job(monkeypatch):
    monkeypatch.setattr("jobscope.config.gpu_job_label", lambda: "hpc_job")
    monkeypatch.setattr("jobscope.config.gpu_job_label_series",
                        lambda: "DCGM_FI_PROF_SM_ACTIVE")


def test_the_label_mapping_is_not_queried_when_unconfigured(gpu_record):
    asked = []

    class Client:
        sampling_period = 60

        def query(self, query, at, timeout=None):
            asked.append(query)
            return []

    discover_gpus(gpu_record, Client(), None)
    assert len(asked) == 1 and "hpc_job" not in asked[0]


def test_a_finished_job_the_join_missed_falls_back_to_the_label(gpu_record, hpc_job):
    gpus = discover_gpus(gpu_record, _two_mapping_client([], [("n1", "GPU-a", 2)]), None)
    assert [g["uuid"] for g in gpus] == ["GPU-a"]
    assert gpus[0]["minor"] == "2" and gpus[0]["node"] == "n1"


def test_the_join_s_answer_is_not_second_guessed(gpu_record, hpc_job):
    gpus = discover_gpus(gpu_record,
                         _two_mapping_client([("n1", "GPU-join", 0)],
                                             [("n1", "GPU-label", 0)]), None)
    assert [g["uuid"] for g in gpus] == ["GPU-join"]


def test_the_fallback_query_is_windowed_and_anchored_at_the_job_s_end(gpu_record, hpc_job):
    """Same protection as _jobid_query: a label still naming the job after it finished
    must not pull in a card that has since moved to someone else's work."""
    seen = {}

    class Client:
        sampling_period = 60

        def query(self, query, at, timeout=None):
            seen.setdefault("query", query if "hpc_job" in query else None)
            if "hpc_job" in query:
                seen["at"] = at
                seen["query"] = query
            return []

    discover_gpus(gpu_record, Client(), None)
    assert seen["at"] == gpu_record.end
    assert "max_over_time" in seen["query"]
    assert "[%ds:]" % gpu_record.duration in seen["query"]
    assert 'hpc_job="%s"' % gpu_record.jobid_raw in seen["query"]


# --- pooling to a node ----------------------------------------------------------
#
# A node figure is the same reduction as the job figure over a subset of UUIDs, which is
# why they share pool_uuids. The subset matters: per_gpu is keyed by (node, minor) and MIG
# siblings share a minor, so reducing *that* would report a partitioned card as one of its
# instances.

def _spec(header, agg="mean"):
    return next(s for s in ALL_SPECS if s.header == header and s.agg == agg)


def test_pool_uuids_uses_each_spec_s_own_agg():
    smact, power = _spec("SM_ACT%"), _spec("POWER_W")
    energy = _spec("ENERGY_kWh", agg="sum")
    pwrmax = _spec("PWRmax_W", agg="max")
    per_uuid = {"a": {"SM_ACT%": 10.0, "POWER_W": 100.0, "ENERGY_kWh": 1.0, "PWRmax_W": 300.0},
                "b": {"SM_ACT%": 30.0, "POWER_W": 200.0, "ENERGY_kWh": 2.0, "PWRmax_W": 400.0}}
    got = pool_uuids(per_uuid, [smact, power, energy, pwrmax], ["a", "b"])
    assert got["SM_ACT%"] == 20.0        # mean
    assert got["POWER_W"] == 150.0       # mean
    assert got["ENERGY_kWh"] == 3.0      # sum -- a node's energy is its cards' total
    assert got["PWRmax_W"] == 400.0      # max -- a peak is not averaged


def test_pool_uuids_counts_mig_siblings_separately():
    """The invariant. Two instances of one card share (node, minor), so a reduction over
    per_gpu would see one of them; over UUIDs it sees both.

    Here the mean of the two is 20 where either alone would read 10 or 30.
    """
    smact = _spec("SM_ACT%")
    per_uuid = {"MIG-a": {"SM_ACT%": 10.0}, "MIG-b": {"SM_ACT%": 30.0}}
    assert pool_uuids(per_uuid, [smact], ["MIG-a", "MIG-b"])["SM_ACT%"] == 20.0


def test_pool_uuids_ignores_uuids_it_has_no_values_for():
    smact = _spec("SM_ACT%")
    got = pool_uuids({"a": {"SM_ACT%": 40.0}}, [smact], ["a", "missing"])
    assert got["SM_ACT%"] == 40.0        # not 20.0, and not a KeyError


def test_pool_and_derive_takes_the_ratio_of_the_pooled_halves():
    """GMEM% is used/total, so it has to be derived *after* pooling. Deriving first and
    pooling after would average the per-card ratios, which is a different number:
    (10+70)/(100+80) is 44.4%, where the mean of 10% and 87.5% is 48.8%.
    """
    used, total = _spec("GMEM_GB", agg="max"), _spec("GMEM_TOTAL_GB", agg="max")
    specs = [used, total]
    per_uuid = {"a": {"GMEM_GB": 10.0, "GMEM_TOTAL_GB": 100.0},
                "b": {"GMEM_GB": 70.0, "GMEM_TOTAL_GB": 80.0}}
    got = pool_and_derive(per_uuid, specs, ["a", "b"])
    # GMEM_GB/GMEM_TOTAL_GB both agg="max" here, so the pooled pair is (70, 100).
    assert got["GMEM%"] == pytest.approx(70.0)
    assert got["GMEM%"] != pytest.approx((10 / 100 + 70 / 80) / 2 * 100)


def test_dcgm_for_job_returns_a_figure_per_node(gpu_record):
    overall, per_gpu, per_node = dcgm_for_job(gpu_record, DEFAULT_SPECS, _client(), None)
    # conftest's gpu_record is one node with two GPUs.
    assert list(per_node) == ["node01"]
    assert set(per_node["node01"]) & set(overall), "the same headers at both levels"


def test_the_node_figure_equals_the_job_figure_on_a_single_node_job(gpu_record):
    """The two reductions must agree when the subset is everything -- the same property
    the renderer test asserts one level up."""
    overall, _per_gpu, per_node = dcgm_for_job(gpu_record, DEFAULT_SPECS, _client(), None)
    for header, value in per_node["node01"].items():
        if isinstance(value, float) and header in overall:
            assert overall[header] == pytest.approx(value), header


def test_the_label_fallback_reaches_the_report_path_too(gpu_record, hpc_job):
    """The gap a review found: dcgm_for_job carried its own copy of discover_gpus' body,
    so the second mapping was live for --ts and probe --validate and inert for the whole
    report table -- the one place the feature was written for. Tests only exercised
    discover_gpus, so nothing caught it.
    """
    client = _two_mapping_client([], [("n1", "GPU-a", 2)])
    assert [g["uuid"] for g in discover_gpus(gpu_record, client, None)] == ["GPU-a"]
    # ...and the same job through the table path, which is what was broken.
    overall, per_gpu, per_node = dcgm_for_job(gpu_record, DEFAULT_SPECS, client, None)
    assert per_gpu, "the report path resolved no cards through the fallback"
    assert list(per_node) == ["n1"]


# --- the ownership clip: an nvml card carries its owner in a label -----------------

_NVML = [s for s in dcgm.METRICS if s.family == "nvml" and s.header == "GPU%"]


def test_an_nvml_selector_is_clipped_to_the_job_that_owned_the_card(gpu_record):
    """One card has one nvml series *per job that has held it* -- the owner is a ``jobid``
    label, not a selector. Measured on one live A100 over two hours: ten series for one
    card, means 0.00 to 100.00. Selecting by UUID alone takes them all, and a collector
    keyed by UUID keeps whichever lands last, which is how --gpu-source nvml reported
    GPU% 0 for a job running at 73%.
    """
    client = _client()
    dcgm_for_job(gpu_record, _NVML, client, None)
    metric_queries = [q for q in client.queries if "nvidia_gpu_duty_cycle" in q]
    assert metric_queries
    for q in metric_queries:
        assert "nvidia_gpu_jobId == 100" in q, q


def test_a_dcgm_selector_is_not_clipped(gpu_record):
    """DCGM series carry no job label, and its UUID/Hostname/gpu labels can never match
    the join's -- so `and` would match nothing and blank the column rather than narrow
    it. The window alone has to bound those."""
    client = _client()
    dcgm_for_job(gpu_record, [s for s in DEFAULT_SPECS if s.uuid_label == "UUID"],
                 client, None)
    for q in client.queries:
        if "DCGM_FI_" in q:
            assert "nvidia_gpu_jobId" not in q, q


def test_the_clip_sits_inside_the_label_replace():
    """`and` keeps only left-hand elements whose label set matches one on the right, and
    label_replace adds NAME_LABEL to the left alone -- so the outer form matches nothing
    and empties every column. Verified against the live server: outer 0 series, inner 2.
    """
    q = dcgm.grouped_window_query("avg", "uuid",
                                  ["nvidia_gpu_duty_cycle", "nvidia_gpu_power_usage_milliwatts"],
                                  ["UUID-A"], 100, clip="nvidia_gpu_jobId == 100")
    assert q.index("nvidia_gpu_jobId") < q.index(dcgm.NAME_LABEL), q
    assert q.count("label_replace") == 1
    # and the unclipped form is byte-identical to what it built before
    assert dcgm.grouped_window_query("avg", "uuid", ["a", "b"], ["UUID-A"], 100) == \
        dcgm.grouped_window_query("avg", "uuid", ["a", "b"], ["UUID-A"], 100, clip=None)


def test_ownership_clip_needs_both_an_nvml_label_and_a_job():
    assert dcgm.ownership_clip("uuid", "100") == "nvidia_gpu_jobId == 100"
    assert dcgm.ownership_clip("UUID", "100") is None      # wrong exporter family
    assert dcgm.ownership_clip("uuid", None) is None       # nothing to clip to
    assert dcgm.ownership_clip("uuid", "") is None


def test_a_value_less_row_is_skipped_not_raised():
    """A response row with no value is the case _store_value reports False for, so the
    caller retries per metric. It raised KeyError instead, aborting the whole report."""
    per_uuid = {"UUID-A": {}}
    spec = _NVML[0]
    assert dcgm._store_value(per_uuid, spec, {"metric": {"uuid": "UUID-A"}}) is False
    assert dcgm._store_value(per_uuid, spec, {"metric": {"uuid": "UUID-A"},
                                              "value": [0, "x"]}) is False
    assert per_uuid["UUID-A"] == {}
    assert dcgm._store_value(per_uuid, spec, {"metric": {"uuid": "UUID-A"},
                                              "value": [0, "42"]}) is True
    assert per_uuid["UUID-A"]["GPU%"] == 42.0


class _MultiOwnerClient(FakeClient):
    """A server where one card carries a series per past owner, as nvml really does.

    The foreign series is returned only when the query does *not* clip to the job, which
    is what makes this test measure the clip rather than the plumbing.
    """

    def __init__(self, gpus, values, mine, theirs):
        super().__init__(gpus, values)
        self.mine, self.theirs = mine, theirs

    def query(self, query, at, timeout=None):
        if "nvidia_gpu_duty_cycle" in query:
            self.queries.append(query)
            uuid = self.gpus[0][0]
            rows = [{"metric": {"uuid": uuid, "jobid": "100"},
                     "value": [at, str(self.mine)]}]
            if "nvidia_gpu_jobId" not in query:
                # A previous job on the same card. Same uuid, so a collector keyed by
                # uuid cannot tell them apart -- whichever lands last wins.
                rows.append({"metric": {"uuid": uuid, "jobid": "99"},
                             "value": [at, str(self.theirs)]})
            return rows
        return super().query(query, at, timeout)


def test_a_previous_owners_samples_do_not_reach_the_column(gpu_record):
    """The reported bug, end to end. `-j 37427690 --gpu-source nvml --runtime-avg` printed
    GPU% 0 in the table while the summary said 71% and the truth was 73 -- because the
    card's ten series (one per past owner, means 0.00 to 100.00) all matched the UUID
    selector and the last one written won.
    """
    client = _MultiOwnerClient([("UUID-A", "node01", "0")], {}, mine=73.0, theirs=0.0)
    # No stored summary: _prefer_stored would otherwise replace the queried GPU% with
    # jobstats' own, and this has to measure what the query returned.
    record = dataclasses.replace(gpu_record, stats={})
    overall, per_gpu, _per_node = dcgm_for_job(record, _NVML, client, None)
    assert overall["GPU%"] == pytest.approx(73.0)
    assert per_gpu[("node01", "0")]["GPU%"] == pytest.approx(73.0)
    # Not by luck of ordering: the foreign row was never offered, because the metric
    # query asked for one owner's samples. (The discovery query names the join too, as
    # its subject rather than as a clip.)
    metric_queries = [q for q in client.queries if "nvidia_gpu_duty_cycle" in q]
    assert metric_queries
    assert all("and nvidia_gpu_jobId == 100" in q for q in metric_queries), metric_queries


def test_the_grouped_query_is_clipped_too(gpu_record):
    """The branch the single-spec tests never reach. GPU%, POWER_W and TEMP_C are all
    nvml/avg, so they share one label_replace query -- and that builder took no clip at
    all, which is the form the reported bug actually ran through.
    """
    nvml_avg = [s for s in dcgm.METRICS
                if s.uuid_label == "uuid" and s.reducer == "avg"
                and s.header in ("GPU%", "POWER_W", "TEMP_C")]
    assert len(nvml_avg) >= 2, "need a real group, or this measures the per-spec path"
    client = _client()
    dcgm_for_job(gpu_record, nvml_avg, client, None)
    grouped = [q for q in client.queries if "label_replace" in q]
    assert grouped, client.queries
    for q in grouped:
        assert "and nvidia_gpu_jobId == 100" in q, q
        # Inside, or `and` sees NAME_LABEL on the left alone and matches nothing.
        assert q.index("nvidia_gpu_jobId") < q.index(NAME_LABEL), q
