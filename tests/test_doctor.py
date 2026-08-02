"""Tests for `jobscope doctor` -- the naming convention, the probes, and the report."""

import io

import pytest

from jobscope import doctor
from jobscope.config import redact_url

# --- the name mapping config depends on ------------------------------------

def test_a_catalogued_series_keeps_its_curated_short_name():
    """Not the mechanical derivation: `dcgm-sm_act` is the name already written in
    people's [thresholds] and [metrics], and SM_ACTIVE would derive `sm_active`."""
    assert doctor.simple_name("DCGM_FI_PROF_SM_ACTIVE") == "dcgm-sm_act"
    assert doctor.simple_name("nvidia_gpu_duty_cycle") == "nvml-gpu"
    assert doctor.simple_name("cgroup_cpu_total_seconds") == "cgroup-cpu"


def test_an_uncatalogued_series_is_named_mechanically():
    assert doctor.simple_name("cgroup_memsw_used_bytes") == "cgroup-memsw_used_bytes"
    assert doctor.simple_name("DCGM_FI_DEV_XID_ERRORS") == "dcgm-xid_errors"
    assert doctor.simple_name("nvidia_gpu_temperature_celsius") == "nvml-temperature_celsius"


def test_a_series_in_no_known_family_gets_no_name():
    """Naming it would imply jobscope knows how to join it to a job. It does not."""
    assert doctor.simple_name("node_load1") is None
    assert doctor.simple_name("up") is None


def test_nvml_and_dcgm_are_split_by_which_uuid_label_they_use():
    """The two families both describe GPUs and both export a duty cycle, so the
    split has to come from the catalog rather than from the metric name."""
    assert doctor.family_of("nvidia_gpu_duty_cycle") == "nvml"
    assert doctor.family_of("DCGM_FI_DEV_GPU_UTIL") == "dcgm"
    assert doctor.catalog()["nvidia_gpu_duty_cycle"][0] == "nvml"
    assert doctor.catalog()["DCGM_FI_PROF_SM_ACTIVE"][0] == "dcgm"


def test_the_longer_dcgm_prefixes_strip_before_the_bare_one():
    """DCGM_FI_PROF_ and DCGM_FI_DEV_ must win over DCGM_FI_, or every name keeps
    a stray `prof_`/`dev_`."""
    assert doctor.simple_name("DCGM_FI_PROF_NEW_THING") == "dcgm-new_thing"
    assert doctor.simple_name("DCGM_FI_DEV_NEW_THING") == "dcgm-new_thing"


def test_every_catalogued_series_yields_a_name():
    for raw in doctor.catalog():
        assert doctor.simple_name(raw), raw


def test_names_are_unique_so_config_cannot_be_ambiguous():
    names = [doctor.simple_name(raw) for raw in doctor.catalog()]
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
    """The regression guard for the leak: doctor is the only thing that prints the
    endpoint, and prometheus.py's contract is that it is never logged."""
    secret = "glc_averysecrettoken"

    class FakeClient:
        url = "https://1180804:%s@prom.grafana.net/api/prom" % secret
        sampling_period = 60

        def query(self, query, at, timeout=None):
            return [{"metric": {}, "value": [at, "1"]}]

    monkeypatch.setattr(doctor, "_flavor", lambda url, timeout: "Grafana Mimir")
    monkeypatch.setattr("jobscope.prometheus.client_from_config",
                        lambda cfg, timeout: FakeClient())
    out = io.StringIO()
    doctor.check_prometheus(out, object(), 30)
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
    monkeypatch.setattr(doctor.time, "time", lambda: client.now)
    assert doctor.probe_retention(client, None) == 180


def test_probe_retention_stops_at_the_first_hit(monkeypatch):
    """Deepest-first and short-circuiting: the misses are ~0.1s while a hit costs
    2-3s against long-term storage, so the walk must not continue past one."""
    client = LadderClient(depth_days=200)
    monkeypatch.setattr(doctor.time, "time", lambda: client.now)
    doctor.probe_retention(client, None)
    assert client.asked == [730, 365, 180]


def test_a_gap_yields_a_conservative_answer_not_a_wrong_one(monkeypatch):
    """This site really does answer at 60/90/180d but not 45d. A gap must cost
    depth, never invent it."""
    client = LadderClient(depth_days=200, gaps={180})
    monkeypatch.setattr(doctor.time, "time", lambda: client.now)
    assert doctor.probe_retention(client, None) == 90


def test_probe_retention_returns_none_when_nothing_answers(monkeypatch):
    client = LadderClient(depth_days=-1)
    monkeypatch.setattr(doctor.time, "time", lambda: client.now)
    assert doctor.probe_retention(client, None) is None


def test_probe_retention_survives_a_query_that_raises(monkeypatch):
    class Boom(LadderClient):
        def query(self, query, at, timeout=None):
            age = round((self.now - at) / 86400)
            if age == 365:
                raise RuntimeError("upstream timeout")
            return super().query(query, at, timeout)

    client = Boom(depth_days=200)
    monkeypatch.setattr(doctor.time, "time", lambda: client.now)
    assert doctor.probe_retention(client, None) == 180


# --- the job sample ---------------------------------------------------------

SAMPLE = [
    ("101", "cpu=8,mem=64G", ""),
    ("102", "cpu=8,gres/gpu:nvidia_h200=1,gres/gpu=1,mem=64G", "JS1:abc"),
    ("103", "cpu=4,gres/gpu=2", "JS1:def"),
]


def test_recent_gpu_job_picks_one_with_gpus():
    assert doctor._recent_gpu_job(SAMPLE) == "102"


def test_recent_gpu_job_is_none_when_the_sample_has_no_gpu_jobs():
    assert doctor._recent_gpu_job([("101", "cpu=8", "")]) is None
    assert doctor._recent_gpu_job([]) is None
    assert doctor._recent_gpu_job(None) is None


def test_check_blob_counts_the_blobs_present():
    out = io.StringIO()
    assert doctor.check_blob(out, SAMPLE) is True
    assert "2 of 3" in out.getvalue()


def test_check_blob_says_so_when_a_site_has_no_jobstats():
    """Not a failure -- it costs the offline view and one oracle, nothing else."""
    out = io.StringIO()
    assert doctor.check_blob(out, [("101", "cpu=8", "")]) is False
    text = out.getvalue()
    assert doctor.ABSENT in text and "Prometheus" in text


def test_check_blob_distinguishes_sacct_failing_from_a_quiet_cluster():
    unavailable, quiet = io.StringIO(), io.StringIO()
    doctor.check_blob(unavailable, None)
    doctor.check_blob(quiet, [])
    assert "could not query sacct" in unavailable.getvalue()
    assert "no finished jobs" in quiet.getvalue()
