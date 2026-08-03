"""Tests for `jobscope probe` -- the naming convention, the probes, and the report."""

import io

import pytest

from jobscope import config as config_module
from jobscope import probe
from jobscope.config import redact_url

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


def test_check_blob_counts_the_blobs_present():
    out = io.StringIO()
    assert probe.check_blob(out, SAMPLE) is True
    assert "2 of 3" in out.getvalue()


def test_check_blob_says_so_when_a_site_has_no_jobstats():
    """Not a failure -- it costs the offline view and one oracle, nothing else."""
    out = io.StringIO()
    assert probe.check_blob(out, [("101", "cpu=8", "")]) is False
    text = out.getvalue()
    assert probe.ABSENT in text and "Prometheus" in text


def test_check_blob_distinguishes_sacct_failing_from_a_quiet_cluster():
    unavailable, quiet = io.StringIO(), io.StringIO()
    probe.check_blob(unavailable, None)
    probe.check_blob(quiet, [])
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
    assert set(data) == {"prometheus", "site", "metrics", "classify"}
    assert data["classify"]["floor"]["power"]["Tesla V100"] == 120   # (40 + 200) / 2


def test_init_full_appends_only_commented_knobs(hermetic_config, tmp_path, monkeypatch):
    """--full appends the template's tuning half. It must stay parseable: two
    [prometheus] blocks in one file is a TOML error, not an override, so only the
    all-comments section below the divider can be concatenated."""
    text = _init_text(tmp_path, monkeypatch, full=True)
    assert text.count("[prometheus]") == 1
    data = config_module._toml.loads(text)
    assert set(data) == {"prometheus", "site", "metrics", "classify"}


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
