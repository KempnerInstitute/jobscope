"""Tests for `jobscope probe` -- the naming convention, the probes, and the report."""

import io

import pytest

from jobscope import config as config_module
from jobscope import probe
from jobscope.config import redact_url
from jobscope.slurm import JobRecord

# --- the name mapping config depends on ------------------------------------

def test_a_catalogued_series_keeps_its_curated_short_name():
    """Not the mechanical derivation: `dcgm-sm_act` is the name already written in
    people's [thresholds] and [metrics], and SM_ACTIVE would derive `sm_active`."""
    assert probe.simple_name("DCGM_FI_PROF_SM_ACTIVE") == "dcgm-sm_act"
    assert probe.simple_name("nvidia_gpu_duty_cycle") == "nvml-gpu"
    assert probe.simple_name("cgroup_cpu_total_seconds") == "cgroup-cpu"


def test_an_uncatalogued_series_is_named_mechanically():
    assert probe.simple_name("cgroup_memsw_used_bytes") == "cgroup-memsw_used_bytes"
    assert probe.simple_name("DCGM_FI_DEV_XID_ERRORS") == "dcgm-xid_errors"
    assert probe.simple_name("nvidia_gpu_ecc_errors") == "nvml-ecc_errors"


def test_a_series_in_no_known_family_gets_no_name():
    """Naming it would imply jobscope knows how to join it to a job. It does not."""
    assert probe.simple_name("node_load1") is None
    assert probe.simple_name("up") is None


def test_nvml_and_dcgm_are_split_by_which_uuid_label_they_use():
    """The two families both describe GPUs and both export a duty cycle, so the
    split has to come from the catalog rather than from the metric name."""
    assert probe.family_of("nvidia_gpu_duty_cycle") == "nvml"
    assert probe.family_of("DCGM_FI_DEV_GPU_UTIL") == "dcgm"
    assert probe.catalog()["nvidia_gpu_duty_cycle"][0] == "nvml"
    assert probe.catalog()["DCGM_FI_PROF_SM_ACTIVE"][0] == "dcgm"


def test_the_longer_dcgm_prefixes_strip_before_the_bare_one():
    """DCGM_FI_PROF_ and DCGM_FI_DEV_ must win over DCGM_FI_, or every name keeps
    a stray `prof_`/`dev_`."""
    assert probe.simple_name("DCGM_FI_PROF_NEW_THING") == "dcgm-new_thing"
    assert probe.simple_name("DCGM_FI_DEV_NEW_THING") == "dcgm-new_thing"


def test_every_catalogued_series_yields_a_name():
    for raw in probe.catalog():
        assert probe.simple_name(raw), raw


def test_names_are_unique_so_config_cannot_be_ambiguous():
    names = [probe.simple_name(raw) for raw in probe.catalog()]
    assert len(names) == len(set(names))


# --- the credential must not be printed ------------------------------------

def test_redact_url_strips_an_embedded_credential():
    """The configured endpoint is a secret that looks like an address."""
    assert redact_url("https://1180804:glc_secrettoken@prom.grafana.net/api/prom") == (
        "https://***@prom.grafana.net/api/prom")


def test_redact_url_keeps_host_and_path_because_that_is_the_diagnostic():
    out = redact_url("https://user:pw@example.org:9090/api/v1/prom")
    assert "example.org:9090" in out and "/api/v1/prom" in out and "pw" not in out


@pytest.mark.parametrize("url", [
    "http://localhost:9090",
    "https://prom.internal/api/prom",
    "not-a-url",
    "",
])
def test_redact_url_leaves_a_credential_free_url_alone(url):
    assert redact_url(url) == url


def test_the_endpoint_is_redacted_in_the_report(monkeypatch):
    """The regression guard for the leak: probe is the only thing that prints the
    endpoint, and prometheus.py's contract is that it is never logged."""
    secret = "glc_averysecrettoken"

    class FakeClient:
        url = "https://1180804:%s@prom.grafana.net/api/prom" % secret
        sampling_period = 60

        def query(self, query, at, timeout=None):
            return [{"metric": {}, "value": [at, "1"]}]

    monkeypatch.setattr(probe, "_flavor", lambda url, timeout: "Grafana Mimir")
    monkeypatch.setattr("jobscope.prometheus.client_from_config",
                        lambda cfg, timeout: FakeClient())
    out = io.StringIO()
    probe.check_prometheus(out, object(), 30)
    assert secret not in out.getvalue()
    assert "***@prom.grafana.net/api/prom" in out.getvalue()


# --- retention probing ------------------------------------------------------

class LadderClient:
    """Answers only for ages at or below `depth_days`, with optional gaps."""

    url = "http://prom"
    sampling_period = 60

    def __init__(self, depth_days, gaps=(), now=1_700_000_000):
        self.depth_days = depth_days
        self.gaps = set(gaps)
        self.now = now
        self.asked = []

    def query(self, query, at, timeout=None):
        age = round((self.now - at) / 86400)
        self.asked.append(age)
        if age in self.gaps or age > self.depth_days:
            return []
        return [{"metric": {}, "value": [at, "1"]}]


def test_probe_retention_reports_the_deepest_age_that_answered(monkeypatch):
    client = LadderClient(depth_days=200)
    monkeypatch.setattr(probe.time, "time", lambda: client.now)
    assert probe.probe_retention(client, None) == 180


def test_probe_retention_stops_at_the_first_hit(monkeypatch):
    """Deepest-first and short-circuiting: the misses are ~0.1s while a hit costs
    2-3s against long-term storage, so the walk must not continue past one."""
    client = LadderClient(depth_days=200)
    monkeypatch.setattr(probe.time, "time", lambda: client.now)
    probe.probe_retention(client, None)
    assert client.asked == [730, 365, 180]


def test_a_gap_yields_a_conservative_answer_not_a_wrong_one(monkeypatch):
    """This site really does answer at 60/90/180d but not 45d. A gap must cost
    depth, never invent it."""
    client = LadderClient(depth_days=200, gaps={180})
    monkeypatch.setattr(probe.time, "time", lambda: client.now)
    assert probe.probe_retention(client, None) == 90


def test_probe_retention_returns_none_when_nothing_answers(monkeypatch):
    client = LadderClient(depth_days=-1)
    monkeypatch.setattr(probe.time, "time", lambda: client.now)
    assert probe.probe_retention(client, None) is None


def test_probe_retention_survives_a_query_that_raises(monkeypatch):
    class Boom(LadderClient):
        def query(self, query, at, timeout=None):
            age = round((self.now - at) / 86400)
            if age == 365:
                raise RuntimeError("upstream timeout")
            return super().query(query, at, timeout)

    client = Boom(depth_days=200)
    monkeypatch.setattr(probe.time, "time", lambda: client.now)
    assert probe.probe_retention(client, None) == 180


# --- the job sample ---------------------------------------------------------

SAMPLE = [
    ("101", "cpu=8,mem=64G", ""),
    ("102", "cpu=8,gres/gpu:nvidia_h200=1,gres/gpu=1,mem=64G", "JS1:abc"),
    ("103", "cpu=4,gres/gpu=2", "JS1:def"),
]


def test_recent_gpu_job_picks_one_with_gpus():
    assert probe._recent_gpu_job(SAMPLE) == "102"


def test_recent_gpu_job_is_none_when_the_sample_has_no_gpu_jobs():
    assert probe._recent_gpu_job([("101", "cpu=8", "")]) is None
    assert probe._recent_gpu_job([]) is None
    assert probe._recent_gpu_job(None) is None


def test_check_jobstats_counts_the_summaries_present():
    out = io.StringIO()
    assert probe.check_jobstats(out, SAMPLE) is True
    assert "2 of 3" in out.getvalue()


def test_check_jobstats_says_so_when_a_site_has_none():
    """Not a failure -- it costs the offline view and one oracle, nothing else."""
    out = io.StringIO()
    assert probe.check_jobstats(out, [("101", "cpu=8", "")]) is False
    text = out.getvalue()
    assert probe.ABSENT in text and "Prometheus" in text


def test_check_jobstats_distinguishes_sacct_failing_from_a_quiet_cluster():
    unavailable, quiet = io.StringIO(), io.StringIO()
    probe.check_jobstats(unavailable, None)
    probe.check_jobstats(quiet, [])
    assert "could not query sacct" in unavailable.getvalue()
    assert "no finished jobs" in quiet.getvalue()


# --- probe --toml: the editable name table ---------------------------------

class TomlClient:
    """Serves a fixed set of series names per family selector."""

    url = "http://prom"
    sampling_period = 60

    def __init__(self, by_family):
        self.by_family = by_family      # {"cgroup"|"nvml"|"dcgm": [series, ...]}

    def query(self, query, at, timeout=None):
        if "jobId" in query:            # GPU discovery
            return [{"metric": {"uuid": "GPU-1", "host": "n1:9400",
                                "minor_number": "0", "name": "NVIDIA A100"}}]
        for family, key in (("cgroup", "jobid="), ("nvml", "uuid=~"), ("dcgm", "UUID=~")):
            if key in query:
                return [{"metric": {"__name__": s}, "value": [at, "1"]}
                        for s in self.by_family.get(family, ())]
        return []


def _emit(by_family, record):
    out = io.StringIO()
    probe.emit_toml(out, TomlClient(by_family), record.jobid, None,
                     [(record.jobid, "gres/gpu=1", "JS1:x")])
    return out.getvalue()


def test_a_builtin_is_emitted_commented_so_its_name_is_visible(gpu_record, monkeypatch):
    """It already works; it is here so the name can be seen and renamed."""
    monkeypatch.setattr("jobscope.slurm.fetch", lambda ids, t: {gpu_record.jobid: gpu_record})
    text = _emit({"dcgm": ["DCGM_FI_PROF_SM_ACTIVE"]}, gpu_record)
    assert "# [metrics.dcgm.sm_act]" in text and "# built-in" in text
    assert "\n[metrics.dcgm.sm_act]" not in text     # never live


def test_an_uncatalogued_series_is_emitted_live(gpu_record, monkeypatch):
    """So a redirect into a config file is the only step -- no uncommenting."""
    monkeypatch.setattr("jobscope.slurm.fetch", lambda ids, t: {gpu_record.jobid: gpu_record})
    text = _emit({"dcgm": ["DCGM_FI_DEV_XID_ERRORS"]}, gpu_record)
    assert "\n[metrics.dcgm.xid_errors]        # new here" in text
    assert 'query  = "DCGM_FI_DEV_XID_ERRORS"' in text


def test_the_table_key_is_the_short_name_without_the_family(gpu_record, monkeypatch):
    """The family is already in the table path; repeating it would make the config
    name `dcgm-xid_errors` inside `[metrics.dcgm]`."""
    monkeypatch.setattr("jobscope.slurm.fetch", lambda ids, t: {gpu_record.jobid: gpu_record})
    text = _emit({"dcgm": ["DCGM_FI_DEV_XID_ERRORS"]}, gpu_record)
    assert "[metrics.dcgm.xid_errors]" in text and "[metrics.dcgm.dcgm-xid_errors]" not in text


def test_a_cgroup_count_is_not_given_an_invented_denominator(gpu_record, monkeypatch):
    """Every cgroup metric is divided by an allocation, and an OOM-kill count has
    none. A percentage of total bytes would be a number with no meaning."""
    monkeypatch.setattr("jobscope.slurm.fetch", lambda ids, t: {gpu_record.jobid: gpu_record})
    text = _emit({"cgroup": ["cgroup_memory_fail_count"]}, gpu_record)
    assert "not expressible here" in text
    assert "[metrics.cgroup.memory_fail_count]" not in text.replace("# ", "")
    assert "MEMORY_FAIL_COUNT%" not in text


@pytest.mark.parametrize("raw,kind,denom", [
    ("cgroup_cpu_user_seconds", "rate", "cpus"),
    ("cgroup_memsw_used_bytes", "gauge", "total_memory"),
])
def test_a_cgroup_shape_is_inferred_from_the_suffix(raw, kind, denom):
    assert probe._cgroup_fields(raw) == (kind, denom)


def test_a_cgroup_count_has_no_shape():
    assert probe._cgroup_fields("cgroup_memory_fail_count") is None


@pytest.mark.parametrize("raw,scale", [
    ("DCGM_FI_PROF_SM_ACTIVE", 100),        # a 0-1 fraction
    ("DCGM_FI_DEV_GPU_UTIL", 1),            # already a percentage
])
def test_a_gpu_scale_is_inferred_where_it_is_reliable(raw, scale):
    assert probe._gpu_scale(raw)[0] == scale


def test_an_unfamiliar_gpu_metric_says_to_check_its_units():
    """A wrong scale reads as a plausible number, so it is flagged rather than
    guessed."""
    _scale, note = probe._gpu_scale("DCGM_FI_DEV_ROW_REMAP_FAILURE")
    assert "CHECK" in note


def test_structural_series_are_listed_but_not_offered_as_metrics(gpu_record, monkeypatch):
    """cgroup_cpus is the denominator every CPU percentage divides by, not a metric.
    Listed rather than dropped, so a reader looking for it finds out where it went."""
    monkeypatch.setattr("jobscope.slurm.fetch", lambda ids, t: {gpu_record.jobid: gpu_record})
    text = _emit({"cgroup": ["cgroup_cpus", "cgroup_memory_rss_bytes"]}, gpu_record)
    assert "cgroup_cpus" in text and "denominator" in text
    assert "[metrics.cgroup.cpus]" not in text.replace("# ", "")


def test_the_other_sources_are_reference_only(gpu_record, monkeypatch):
    """slurm-* comes from sacct fields, so a query = "..." table cannot define one --
    printing syntax that fails would be worse than printing nothing."""
    monkeypatch.setattr("jobscope.slurm.fetch", lambda ids, t: {gpu_record.jobid: gpu_record})
    text = _emit({"cgroup": ["cgroup_memory_rss_bytes"]}, gpu_record)
    assert "slurm-cpu" in text and "slurm-gpuutil" in text
    assert "[metrics.slurm" not in text.replace("# ", "")


def test_the_output_is_a_loadable_config(gpu_record, monkeypatch, tmp_path,
                                        hermetic_config):
    """The whole point: `probe --toml >> config.toml` has to produce a config file,
    and the live blocks have to take effect."""
    from jobscope import dcgm
    monkeypatch.setattr("jobscope.slurm.fetch", lambda ids, t: {gpu_record.jobid: gpu_record})
    text = _emit({"cgroup": ["cgroup_memory_rss_bytes", "cgroup_memsw_used_bytes"],
                  "dcgm": ["DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_DEV_XID_ERRORS"]},
                 gpu_record)
    path = tmp_path / "generated.toml"
    path.write_text(text)
    config_module.set_config(config_module.load_config(str(path)))
    assert dcgm.spec_named("xid_errors").metric == "DCGM_FI_DEV_XID_ERRORS"
    from jobscope import cpu
    assert cpu.spec_named("memsw_used_bytes").denom == "total_memory"
    # And the commented built-in stayed a built-in, not a duplicate.
    assert len([s for s in dcgm.METRICS if s.key == "smact"]) == 1


def test_no_job_to_probe_yields_a_comment_not_a_crash(gpu_record):
    out = io.StringIO()
    assert probe.emit_toml(out, TomlClient({}), None, None, []) == 1
    assert out.getvalue().lstrip().startswith("#")


# --- detect_scrape: the trap is that the obvious method agrees with any guess ---

class _RawSampleClient:
    """Answers a range *selector* with real 30s samples, and query_range with the step.

    Modelled on the server: query_range aligns to whatever step it is given, which is
    why reading gaps from it "detects" the caller's own argument.
    """

    sampling_period = 60

    def query(self, query, at, timeout=None):
        if not query.endswith("[10m]"):
            return []
        return [{"metric": {}, "values": [[1000 + 30 * i, "1"] for i in range(20)]}]

    def query_range(self, query, start, end, step, timeout=None):
        return [{"metric": {}, "values": [[start + step * i, "1"] for i in range(20)]}]


def test_detect_scrape_reads_raw_samples_not_the_step():
    assert probe.detect_scrape(_RawSampleClient(), 30) == 30


def test_detect_scrape_ignores_a_long_gap_from_a_restart():
    """The mode, not the mean: one exporter restart would drag an average up."""
    class Client(_RawSampleClient):
        def query(self, query, at, timeout=None):
            if not query.endswith("[10m]"):
                return []
            stamps = [0, 60, 120, 900, 960, 1020, 1080]     # one 780s hole
            return [{"metric": {}, "values": [[t, "1"] for t in stamps]}]

    assert probe.detect_scrape(Client(), 30) == 60


def test_detect_scrape_returns_none_when_nothing_answers():
    class Silent:
        sampling_period = 60

        def query(self, query, at, timeout=None):
            return []

    assert probe.detect_scrape(Silent(), 30) is None


# --- measure_power_floors: report why, never guess ---------------------------

def _floor_client(idle, busy):
    class Client:
        sampling_period = 60

        def query(self, query, at, timeout=None):
            wanted = idle if "0.9" in query else busy
            return [{"metric": {"modelName": m}, "value": [at, str(v)]}
                    for m, v in wanted.items()]
    return Client()


def test_power_floors_put_the_floor_between_the_two_populations():
    floors = probe.measure_power_floors(_floor_client({"H200": 114.0}, {"H200": 253.0}), 30)
    value, why = floors["H200"]
    assert value == 180                       # (114 + 253) / 2, to the nearest 10
    assert "idle p90 114 W, busy p10 253 W" in why


def test_power_floors_omit_a_model_whose_populations_overlap():
    """A wrong floor silently caps healthy jobs at `inefficient`, so no number beats
    a guess. Two of this cluster's twelve models overlap at any given moment."""
    floors = probe.measure_power_floors(_floor_client({"A100": 97.0}, {"A100": 96.0}), 30)
    value, why = floors["A100"]
    assert value is None
    assert "overlap" in why and "97 W" in why and "96 W" in why


def test_power_floors_fall_back_to_a_margin_above_idle_when_nothing_was_busy():
    """Four of twelve models had no busy samples in the window -- an idle partition is
    the normal state, not an error."""
    floors = probe.measure_power_floors(_floor_client({"V100": 31.0}, {}), 30)
    value, why = floors["V100"]
    assert value == 40                        # 31 x 1.15, to the nearest 10
    assert "no busy samples" in why


# --- probe --init: the generated site config ---------------------------------

class _InitClient:
    """Enough of a server to answer every detector --init runs."""

    url = "http://prom:9090/api/v1"
    sampling_period = 60

    def query(self, query, at, timeout=None):
        if query.endswith("[10m]"):                       # detect_scrape
            return [{"metric": {}, "values": [[t, "1"] for t in range(0, 600, 30)]}]
        if "quantile" in query:                           # measure_power_floors
            watts = 40.0 if "0.9" in query else 200.0
            return [{"metric": {"modelName": "Tesla V100"}, "value": [at, str(watts)]}]
        if "count by" in query:                           # detect_labels
            label = query.split("(", 1)[1].split(")", 1)[0]
            return [{"metric": {label: "n1"}, "value": [at, "1"]}]
        return [{"metric": {}, "value": [at, "1"]}]


def _init_text(tmp_path, monkeypatch, full=False):
    monkeypatch.setattr(probe, "_scontrol_config", lambda t: {"ClusterName": "testbed"})
    monkeypatch.setattr(probe, "probe_series", lambda *a, **k: None)
    out = io.StringIO()
    probe.emit_config(out, _InitClient(), config_module.get_config(), 30, full=full)
    return out.getvalue()


def test_init_emits_only_detected_values(hermetic_config, tmp_path, monkeypatch):
    """Thresholds are absent on purpose: a band edge is a policy choice about what
    counts as waste, not a property of the cluster."""
    text = _init_text(tmp_path, monkeypatch)
    assert "[thresholds" not in text
    assert "sampling_period = 30" in text            # measured, not the client's 60
    assert 'host_label   = "host"' in text
    assert "testbed" in text                         # named the cluster it probed


def test_init_never_writes_the_endpoint(hermetic_config, tmp_path, monkeypatch):
    """The URL commonly embeds a Grafana Cloud token; a generated file must not be
    where that lands."""
    text = _init_text(tmp_path, monkeypatch)
    assert "prom:9090" not in text
    assert "No url" in text and "credential" in text


def test_init_parses_as_toml(hermetic_config, tmp_path, monkeypatch):
    """The whole point: this is a config file, not a report about one."""
    data = config_module._toml.loads(_init_text(tmp_path, monkeypatch))
    assert set(data) == {"prometheus", "site", "metrics", "eff"}
    assert data["eff"]["floor"]["power"]["Tesla V100"] == 120   # (40 + 200) / 2


def test_init_full_appends_only_commented_knobs(hermetic_config, tmp_path, monkeypatch):
    """--full appends the template's tuning half. It must stay parseable: two
    [prometheus] blocks in one file is a TOML error, not an override, so only the
    all-comments section below the divider can be concatenated."""
    text = _init_text(tmp_path, monkeypatch, full=True)
    assert text.count("[prometheus]") == 1
    data = config_module._toml.loads(text)
    assert set(data) == {"prometheus", "site", "metrics", "eff"}


def test_init_refuses_to_overwrite_an_existing_config(hermetic_config, tmp_path,
                                                     monkeypatch):
    """A config is hand-tuned within a week of being written, and the tuning has no
    other copy. So an existing file turns --init into stdout plus a note."""
    monkeypatch.setattr(probe, "_scontrol_config", lambda t: {})
    monkeypatch.setattr(probe, "probe_series", lambda *a, **k: None)
    path = tmp_path / "c.toml"
    path.write_text("# hand-tuned, do not lose\n")

    out, notes = io.StringIO(), io.StringIO()
    status = probe.write_config(out, notes, _InitClient(), config_module.get_config(),
                                str(path), 30, None, None, False)
    assert status == 0
    assert path.read_text() == "# hand-tuned, do not lose\n"      # untouched
    assert "already exists" in notes.getvalue()
    assert "[site]" in out.getvalue()                             # printed instead


def test_init_writes_when_nothing_is_there(hermetic_config, tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "_scontrol_config", lambda t: {})
    monkeypatch.setattr(probe, "probe_series", lambda *a, **k: None)
    path = tmp_path / "sub" / "c.toml"                 # parent does not exist either
    out, notes = io.StringIO(), io.StringIO()
    status = probe.write_config(out, notes, _InitClient(), config_module.get_config(),
                               str(path), 30, None, None, False)
    assert status == 0 and out.getvalue() == ""
    assert "[site]" in path.read_text()
    assert "wrote" in notes.getvalue()
    # And what it wrote is loadable.
    assert config_module.load_config(str(path)).site.host_label == "host"


# --- probe --coverage: per column, and what is missing -----------------------------

@pytest.mark.parametrize("state,up", [
    ("idle", True), ("mixed", True), ("allocated", True), ("completing", True),
    ("down", False), ("down*", False), ("drained", False), ("drng", False),
    ("maint", False), ("fail", False), ("unknown", False),
])
def test_only_nodes_that_could_report_are_counted(state, up):
    """Without this the report cries wolf: a partition with six `down` nodes names them
    against every column, burying the one node that is up and still missing a series --
    the only line worth acting on."""
    assert probe._is_up(state) is up


def test_every_source_is_reported_not_just_the_winner():
    """Candidates, not the resolved view: with only the winner shown,
    nvidia_gpu_duty_cycle is invisible whenever dcgm wins GPU%, so "would --gpu-source
    nvml cover more of my partition?" has no answer here -- which is the decision this
    report exists to inform."""
    by_family = probe._coverage_series()
    assert set(by_family) >= {"cgroup", "nvml", "dcgm"}
    # GPU% has a candidate under both exporters, and both are listed.
    gpu_families = {f for f, rows in by_family.items()
                    if any(col == "GPU%" for _s, col, _serving in rows)}
    assert gpu_families == {"nvml", "dcgm"}
    # slurm serves CPU%/MEM% from sacct, so it has no series and no host coverage.
    assert "slurm" not in by_family
    assert all(series for rows in by_family.values() for series, _c, _s in rows)


def test_exactly_one_source_is_marked_serving_per_column():
    """The mark has to name the *exporter* a running job reads. Keyed on source_of it
    would answer "jobstats" for the columns the stored summary wins -- and jobstats has
    no series, so every row here would be unmarked and the mark meaningless."""
    by_family = probe._coverage_series()
    serving = {}
    for family, rows in by_family.items():
        for _series, column, is_serving in rows:
            if is_serving:
                serving.setdefault(column, []).append(family)
    assert serving, "nothing marked serving"
    for column, families in serving.items():
        assert len(families) == 1, (column, families)
    # CPU% comes from cgroup even though the stored summary outranks it in the order.
    assert serving["CPU%"] == ["cgroup"]


def _coverage_report(monkeypatch, sinfo_lines, hosts_by_series, join=None):
    """Run report_column_coverage against a canned sinfo and a canned server.

    ``join`` is ``{host: [job id]}`` for the GPU join series, which is checked separately
    from the metric rows because its *values* are what matter -- see
    probe.check_gpu_join. Defaulted so the join is healthy on every host that publishes
    an nvidia series, which is what the metric-coverage tests below assume; they are not
    about the join and should not have to say so.
    """
    monkeypatch.setattr(probe, "run_capture",
                        lambda *a, **k: "\n".join(sinfo_lines) + "\n")
    if join is None:
        join = {h: [1] for h in hosts_by_series.get("nvidia_gpu", ())}

    class Client:
        def query(self, query, at, timeout=None):
            if "jobId" in query:
                return [{"metric": {"host": h}, "value": [at, str(j)]}
                        for h, jobs in join.items() for j in jobs]
            for series, hosts in hosts_by_series.items():
                if series in query:
                    return [{"metric": {"host": h}} for h in hosts]
            return []

    out = io.StringIO()
    probe.report_column_coverage(out, Client(), None, "somepart")
    return out.getvalue()


def test_a_node_up_and_missing_a_series_is_named_with_its_state(monkeypatch):
    """The report that prompted this: dcgm-exporter down on one host of a partition
    dropped 2 of 3 jobs from a --ts --eff run, and nothing said which host."""
    text = _coverage_report(
        monkeypatch,
        ["good mixed gpu:a100:4", "bad mixed gpu:a100:4", "dead down* gpu:a100:4"],
        {"DCGM_FI": ["good"], "nvidia_gpu": ["good", "bad"],
         "cgroup_": ["good", "bad"]})
    assert "bad" in text and "(mixed)" in text
    assert "no dcgm" in text
    # The down node is excluded, not listed against every column.
    assert "dead" not in text
    assert "1 down/drained, not counted" in text


def test_gpu_columns_are_not_reported_missing_on_cpu_only_nodes(monkeypatch):
    """A node with no GPU publishes no GPU series, correctly. Counting it as missing
    turns a CPU partition into a page of noise saying "these are CPU nodes"."""
    text = _coverage_report(
        monkeypatch, ["c1 mixed (null)", "c2 idle (null)"],
        {"cgroup_": ["c1", "c2"]})
    assert "no GPU nodes" in text
    assert "every serving series covers every node that is up" in text


@pytest.mark.parametrize("gres,is_mig", [
    ("gpu:nvidia_a100_3g.20gb:8", True),
    ("gpu:nvidia_a100_1g.5gb:56", True),
    ("gpu:nvidia_a100-sxm4-40gb:4", False),
    ("gpu:nvidia_h100_80gb_hbm3:4", False),
    ("(null)", False),
])
def test_mig_is_recognised_from_the_gres_profile(gres, is_mig):
    """`3g.20gb` is a MIG profile; `a100-sxm4-40gb` is a whole card whose name also has
    digits and a dash, so the pattern has to be the `Ng.Mgb` form specifically."""
    assert bool(probe._MIG_GRES.search(gres)) is is_mig


def test_mig_excuses_only_the_whole_device_column(monkeypatch):
    """Partitioning a card leaves no whole *device* to report a duty cycle for, so
    neither exporter publishes GPU% there -- not a misconfiguration and not fixable. The
    per-instance profiling and memory series come through fine, so a gap in *those* on a
    MIG node is still a real fault and must not be waved through."""
    only_gpu = _coverage_report(
        monkeypatch, ["m1 mixed gpu:nvidia_a100_3g.20gb:8"],
        {"DCGM_FI_PROF": ["m1"], "DCGM_FI_DEV_POWER": ["m1"],
         "nvidia_gpu_memory": ["m1"], "cgroup_": ["m1"]})
    assert "nothing unexplained" in only_gpu
    assert "have no GPU%: MIG partitions the card" in only_gpu

    also_profiling = _coverage_report(
        monkeypatch, ["m1 mixed gpu:nvidia_a100_3g.20gb:8"],
        {"nvidia_gpu_memory": ["m1"], "cgroup_": ["m1"]})
    assert "1 node(s) with an unexplained gap" in also_profiling
    assert "SM_ACT%" in also_profiling


def test_each_gap_is_explained_on_its_own_terms(monkeypatch):
    """An idle MIG node has two absences with two different explanations. Judging the
    node as a whole put it in the fault list for both -- the output that prompted this
    read "8 node(s) running jobs" over six idle ones."""
    text = _coverage_report(
        monkeypatch, ["m1 idle gpu:nvidia_a100_3g.20gb:8"],
        {"DCGM_FI_PROF": ["m1"], "DCGM_FI_DEV_POWER": ["m1"],
         "nvidia_gpu_memory": ["m1"]})
    assert "nothing unexplained" in text
    assert "have no GPU%: MIG partitions the card" in text
    assert "no job is running" in text


@pytest.mark.parametrize("state", ["idle", "reserved", "planned"])
def test_a_node_with_no_job_is_not_faulted_for_missing_cgroup(monkeypatch, state):
    """cgroup series are per running *job*, not per node, so a node with nothing running
    has none and its absence is the right answer. Keyed on whether a job runs rather than
    on `idle` alone: six `reserved` nodes were being reported as faults, outnumbering the
    one host that really was misconfigured."""
    text = _coverage_report(monkeypatch, ["n1 %s gpu:a100:4" % state],
                            {"DCGM_FI": ["n1"], "nvidia_gpu": ["n1"]})
    assert "no job is running, and cgroup series exist per running job" in text
    # Kept out of the fault list, which is what makes the fault list worth reading.
    assert "nothing unexplained" in text


@pytest.mark.parametrize("state", ["mixed", "allocated", "completing"])
def test_a_node_running_jobs_is_faulted_for_missing_cgroup(monkeypatch, state):
    """The other half: a job IS running there, so the series should exist."""
    text = _coverage_report(monkeypatch, ["n1 %s gpu:a100:4" % state],
                            {"DCGM_FI": ["n1"], "nvidia_gpu": ["n1"]})
    assert "1 node(s) with an unexplained gap" in text
    assert "no job is running" not in text


def test_a_real_fault_is_not_buried_by_explained_absences(monkeypatch):
    """Measured on five partitions: 6 reserved nodes and 1 genuinely misconfigured host.
    The fault has to come first and the rest collapse to a line."""
    text = _coverage_report(
        monkeypatch,
        ["bad mixed gpu:a100:4"] + ["r%d reserved gpu:a100:4" % i for i in range(6)],
        {"nvidia_gpu": ["bad"] + ["r%d" % i for i in range(6)],
         "DCGM_FI": ["r%d" % i for i in range(6)],
         "cgroup_": ["bad"]})
    fault_line = text.index("no dcgm")
    assert fault_line < text.index("no job is running")
    assert "1 node(s) with an unexplained gap" in text


def test_an_unknown_partition_says_how_to_list_them(monkeypatch):
    monkeypatch.setattr(probe, "run_capture", lambda *a, **k: "")
    with pytest.raises(probe.JobscopeError) as exc:
        probe.report_column_coverage(io.StringIO(), object(), None, "nope")
    assert "sinfo -o %R" in str(exc.value)


# --- the job-to-card join: present is not the same as current -----------------
#
# The gap these close. _coverage_series() enumerates metric specs and the join is not
# one -- it is [site] gpu_job_join -- so it never appeared in the table, while the nvml
# heading promised "the job-to-GPU join every source depends on". Measured on this
# cluster: the series was on every host and stale. No card claimed a job newer than
# 37239323 while jobs to 37366799 ran, and every GPU job started after the freeze
# reported blank GPU columns with nothing saying why.

def _join_check(monkeypatch, join_rows, squeue_lines, scope=None, full=False,
                sacct="COMPLETED|2026-08-04T14:37:04"):
    """Run check_gpu_join against a canned join series, squeue and sacct.

    Dispatched on the command, because the check now shells out to two different ones and
    a single canned answer would have sacct parsing squeue's rows.
    """
    def capture(cmd, timeout=None, what="", soft=False):
        if cmd[0] == "sacct":
            return (sacct + "\n") if sacct is not None else None
        return ("\n".join(squeue_lines) + "\n") if squeue_lines is not None else None

    monkeypatch.setattr(probe, "run_capture", capture)
    monkeypatch.setattr(probe, "expand_nodelist",
                        lambda text, timeout=None: tuple(
                            t for t in str(text).replace("[", "").replace("]", "").split(",") if t))

    class Client:
        def query(self, query, at, timeout=None):
            return [{"metric": {"host": h}, "value": [at, str(j)]} for h, j in join_rows]

    out = io.StringIO()
    probe.check_gpu_join(out, Client(), None, scope, full=full)
    return out.getvalue()


def test_the_join_series_now_appears_in_the_coverage_table(monkeypatch):
    """It was structurally absent: not a metric spec, so no row could exist for it."""
    text = _join_check(monkeypatch, [("n1", 100)], ["100|n1|gres/gpu:1"])
    assert "jobId" in text or "join" in text


def test_a_frozen_mapping_is_reported_with_both_job_ids(monkeypatch):
    """The regression. Cards name only old jobs; the running one is invisible.

    Both numbers are printed because the gap is the evidence -- a reader should not have
    to take "looks frozen" on trust.
    """
    text = _join_check(monkeypatch,
                       [("n1", 100), ("n2", 101)],
                       ["500|n1|gres/gpu:1", "501|n2|gres/gpu:1"])
    # The ceiling and the running id, which are the evidence. The exact wording depends
    # on whether sacct could date it -- see the two tests for that below.
    assert "101" in text and "501" in text
    assert "the mapping" in text
    assert "2 of 2 host(s)" in text


def test_a_healthy_mapping_says_so_and_does_not_cry_wolf(monkeypatch):
    text = _join_check(monkeypatch,
                       [("n1", 500), ("n2", 501)],
                       ["500|n1|gres/gpu:1", "501|n2|gres/gpu:1"])
    assert "every host running a GPU job is named by one of its own cards" in text
    assert "frozen" not in text and "FAIL" not in text


def test_a_host_with_no_running_gpu_job_is_not_a_fault(monkeypatch):
    """An idle card may legitimately still name its last occupant. Only hosts that *are*
    running a GPU job get judged, or this cries wolf on every drained node."""
    text = _join_check(monkeypatch, [("n1", 100)], ["500|n2|gres/gpu:1"])
    assert "FAIL" not in text
    assert "nothing to be checked against" in text


def test_a_cpu_only_job_is_not_counted_as_an_absence(monkeypatch):
    """It correctly claims no card. %b is read so this is not noise."""
    text = _join_check(monkeypatch, [("n1", 100)], ["500|n1|N/A"])
    assert "FAIL" not in text


def test_partial_coverage_is_reported_separately_from_blind(monkeypatch):
    """A host naming one of its two jobs is a different fault from naming neither."""
    text = _join_check(monkeypatch,
                       [("n1", 500)],
                       ["500|n1|gres/gpu:1", "501|n1|gres/gpu:1"])
    assert "name some of their jobs but not all" in text


def test_presence_without_squeue_does_not_claim_freshness(monkeypatch):
    """The distinction the original bug turned on: the series was present on every host.
    Without Slurm to compare against, say the values went unchecked rather than pass."""
    text = _join_check(monkeypatch, [("n1", 100)], None)
    assert "presence is not freshness" in text
    assert "frozen" not in text


def test_a_series_carrying_no_job_id_is_not_reported_as_absent(monkeypatch):
    """Present-but-broken and missing send someone to different places."""
    class Client:
        def query(self, query, at, timeout=None):
            return [{"metric": {"host": "n1"}}]          # no "value" at all

    out = io.StringIO()
    probe.check_gpu_join(out, Client(), None, None)
    text = out.getvalue()
    assert "none carrying a job id" in text
    assert "no series at all" not in text


def test_an_unreadable_join_series_is_a_hard_failure(monkeypatch):
    class Client:
        def query(self, query, at, timeout=None):
            raise RuntimeError("boom")

    out = io.StringIO()
    probe.check_gpu_join(out, Client(), None, None)
    assert "could not be queried" in out.getvalue()


def test_the_join_row_is_scoped_to_the_partition(monkeypatch):
    """Counted like the metric rows above it -- in scope / could publish it."""
    text = _join_check(monkeypatch, [("n1", 500), ("other", 9)],
                       ["500|n1|gres/gpu:1"], scope={"n1", "n2"})
    assert "1/2" in text


# --- dating the freeze, and the --full breakdown ------------------------------

def test_the_frozen_mapping_is_dated_not_just_numbered(monkeypatch):
    """An id is a puzzle, a timestamp is a ticket. sacct turns the newest claimed job
    into when the mapping stopped taking new ones."""
    text = _join_check(monkeypatch, [("n1", 100)], ["500|n1|gres/gpu:1"])
    assert "which ended 2026-08-04T14:37:04" in text
    assert "stopped taking new jobs then" in text


def test_a_still_running_newest_claim_is_not_dated(monkeypatch):
    """Its end is not when anything stopped, so fall back to the undated wording rather
    than invent a time."""
    text = _join_check(monkeypatch, [("n1", 100)], ["500|n1|gres/gpu:1"],
                       sacct="RUNNING|Unknown")
    assert "which ended" not in text
    assert "looks frozen" in text


def test_an_unavailable_sacct_still_reports_the_freeze(monkeypatch):
    """The verdict does not depend on being able to date it."""
    text = _join_check(monkeypatch, [("n1", 100)], ["500|n1|gres/gpu:1"], sacct=None)
    assert "looks frozen" in text
    assert "which ended" not in text


def test_without_full_the_host_list_is_replaced_by_a_pointer(monkeypatch):
    """Fleet-wide this runs to hundreds of lines, so it is gated -- but the reader has to
    learn it exists."""
    text = _join_check(monkeypatch, [("n1", 100)], ["500|n1|gres/gpu:1"])
    assert "--full lists the affected hosts" in text
    assert "cards claim" not in text


def test_full_lists_each_affected_host_with_both_sets(monkeypatch):
    """What goes in the ticket: what the host runs, and what its cards say instead."""
    text = _join_check(monkeypatch, [("n1", 100)], ["500|n1|gres/gpu:1"], full=True)
    assert "n1" in text and "runs 500" in text and "cards claim 100" in text


def test_full_splits_the_claims_by_kind(monkeypatch):
    """Naming a finished job and naming a job on another host are different exporter
    bugs; the host-level count conflates them."""
    text = _join_check(monkeypatch,
                       [("n1", 500), ("n1", 501), ("n1", 900)],
                       ["500|n1|gres/gpu:1", "501|n2|gres/gpu:1"], full=True)
    # 500 is on n1 (correct), 501 runs on n2 (misplaced), 900 is not running (finished).
    assert "3 claim(s)" in text
    assert "33% correct" in text
    assert "33% name a finished job" in text
    assert "33% name a job on another host" in text


def test_a_live_cpu_only_job_counts_as_misplaced_not_finished(monkeypatch):
    """The reason the running set is not filtered to GPU jobs: calling a live job
    'finished' would point at the wrong bug."""
    text = _join_check(monkeypatch, [("n1", 700)],
                       ["500|n1|gres/gpu:1", "700|n2|N/A"], full=True)
    assert "name a job on another host" in text
    assert "0% name a finished job" in text


def test_classify_claims_counts_per_claim_not_per_host():
    correct, finished, misplaced = probe._classify_claims(
        {"n1": {1, 2, 3}}, {"n1": {1}}, {1, 2})
    assert (correct, finished, misplaced) == (1, 1, 1)


# --- is there a second job-to-card mapping? -----------------------------------
#
# Whether one exists decides how bad a frozen join is: with one, a job the join missed
# still resolves; without one, the join is a single point of failure for every GPU column
# -- which is what it was on this cluster when it froze.

def _label_mapping(monkeypatch, label="", published=0):
    monkeypatch.setattr("jobscope.config.gpu_job_label", lambda: label)
    monkeypatch.setattr("jobscope.config.gpu_job_label_series",
                        lambda: "DCGM_FI_PROF_SM_ACTIVE")

    class Client:
        def query(self, query, at, timeout=None):
            # The real query is count(...) -- one scalar, not ~2000 labelled series.
            assert "hpc_job" not in query or query.startswith("count("), query
            return [{"metric": {}, "value": [at, str(published)]}] if "hpc_job" in query else []

    out = io.StringIO()
    probe.report_label_mapping(out, Client(), None)
    return out.getvalue()


def test_a_configured_second_mapping_is_reported(monkeypatch):
    text = _label_mapping(monkeypatch, label="hpc_job")
    assert "2nd job join" in text and "configured" in text


def test_no_second_mapping_says_the_join_is_a_single_point_of_failure(monkeypatch):
    """The state this cluster was in, and the reason the freeze was total."""
    text = _label_mapping(monkeypatch, label="")
    assert "single point of failure" in text
    assert "nvidia_gpu_jobId" in text


def test_an_unused_hpc_job_label_is_pointed_out(monkeypatch):
    """A site publishing it without telling jobscope has a fallback for one config key,
    and would otherwise never find out."""
    text = _label_mapping(monkeypatch, label="", published=12)
    assert "12 series carry an 'hpc_job' label" in text
    assert 'gpu_job_label = "hpc_job"' in text
    assert "single point of failure" not in text


def test_the_cross_check_normalises_host_names_like_the_report_does(monkeypatch):
    """A review found these disagreeing: the report path stripped the domain and probe
    did not, so at a site whose exporter labels are fully qualified probe reported a clean
    mapping for the very check the report tells you to run.

    One definition now (config.short_host), used on both sides.
    """
    text = _join_check(monkeypatch,
                       [("n1.cluster.example.edu", 500)],
                       ["500|n1|gres/gpu:1"])
    assert "every host running a GPU job is named by one of its own cards" in text
    assert "nothing to be checked against" not in text


# --- --metrics --full: the values, not just the names -------------------------

def test_a_single_gpu_job_is_preferred_for_the_sample():
    """Its families publish one series per name, so the values read as one reading per
    metric rather than the first of thirty-two."""
    sample = [("101", "cpu=8,mem=64G", ""),                      # no GPU
              ("102", "cpu=8,gres/gpu=4,mem=64G", "JS1:a"),      # 4 GPUs
              ("103", "cpu=4,gres/gpu=1,mem=8G", "JS1:b")]       # 1 GPU
    assert probe._gpu_job_candidates(sample) == ["103", "102"]
    assert probe._recent_gpu_job(sample) == "103"


def test_a_multi_gpu_job_is_still_offered_when_there_is_no_single_one():
    sample = [("102", "cpu=8,gres/gpu=4", "JS1:a")]
    assert probe._gpu_job_candidates(sample) == ["102"]


def test_a_sampled_value_shows_the_count_only_when_there_is_more_than_one():
    """One line per series would turn a 32-GPU job's ~500 series into a page."""
    assert probe._sampled(("1410", 1)) == "1410"
    assert probe._sampled(("1410", 4)) == "1410  (first of 4)"
    # --metrics without --full carries counts, not pairs, and prints no value cell.
    assert probe._sampled(3) == ""


def test_samples_are_raw_as_the_server_returned_them():
    """The point of this view is what the exporter publishes: scaling
    nvidia_gpu_memory_total_bytes to 80.0 GB would hide the thing being looked at."""
    class Client:
        def query(self, query, at, timeout=None):
            return [{"metric": {"__name__": "nvidia_gpu_memory_total_bytes"},
                     "value": [at, "85899345920"]}]

    got = probe._samples_for(Client(), "{}", 0, None)
    assert got["nvidia_gpu_memory_total_bytes"] == ("85899345920", 1)


def test_each_family_lists_only_its_own_series(monkeypatch):
    """The cgroup selector is {jobid="..."}, and the nvidia exporter labels its own series
    with a jobid too -- so every nvml series matched it and was listed under cgroup."""
    monkeypatch.setattr(probe, "run_capture", lambda *a, **k: None)

    class Client:
        def query(self, query, at, timeout=None):
            # One server, answering every selector with both families' series.
            return [{"metric": {"__name__": "cgroup_cpu_total_seconds"}, "value": [at, "5"]},
                    {"metric": {"__name__": "nvidia_gpu_duty_cycle"}, "value": [at, "99"]}]

    record = JobRecord(jobid="1", state="COMPLETED", name="j", runtime="1:00", nodes="1",
                       gpus=1, stats={}, start=0, end=100, duration=100, jobid_raw="1",
                       cluster="c", user="u")
    monkeypatch.setattr("jobscope.slurm.fetch", lambda ids, t: {"1": record})
    monkeypatch.setattr("jobscope.dcgm.discover_gpus",
                        lambda rec, c, t: [{"uuid": "GPU-a", "node": "n1", "minor": "0",
                                            "model": "H200"}])
    _rec, families = probe.probe_series(Client(), "1", None, samples=True)
    by_family = dict(families)
    assert list(by_family["cgroup"]) == ["cgroup_cpu_total_seconds"]
    assert list(by_family["nvml"]) == ["nvidia_gpu_duty_cycle"]
