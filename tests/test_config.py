"""Tests for configuration loading and Prometheus endpoint resolution."""

import pytest

from jobscope import config as config_module
from jobscope.config import (
    Config,
    Defaults,
    Thresholds,
    example_config_text,
    grade_band,
    load_config,
    resolve_prometheus,
)
from jobscope.errors import JobscopeError

CONFIG_TOML = """\
[prometheus]
url = "https://file.example/api/prom"
sampling_period = 30
site_jobstats_config_path = "/opt/jobstats"

[thresholds]
red = 40
power_w = 250

[defaults]
workers = 4
timeout = 15
min_runtime = 90
"""


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(env={"XDG_CONFIG_HOME": str(tmp_path)})
    assert cfg.prometheus_url is None
    assert cfg.sampling_period == 60
    assert cfg.defaults.workers == 8
    assert cfg.thresholds.red == 10
    assert cfg.thresholds.power_w == 100


def test_load_from_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG_TOML)
    cfg = load_config(path=str(path), env={})
    assert cfg.prometheus_url == "https://file.example/api/prom"
    assert cfg.sampling_period == 30
    assert cfg.sampling_period_explicit is True
    assert cfg.site_jobstats_config_path == "/opt/jobstats"
    assert cfg.thresholds.red == 40
    assert cfg.thresholds.power_w == 250
    assert cfg.defaults == Defaults(workers=4, timeout=15.0, min_runtime=90)


def test_env_overrides_url(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG_TOML)
    cfg = load_config(path=str(path), env={"JOBSCOPE_PROM_URL": "https://env.example/api/prom"})
    assert cfg.prometheus_url == "https://env.example/api/prom"


def test_config_env_var_selects_file(tmp_path):
    path = tmp_path / "custom.toml"
    path.write_text(CONFIG_TOML)
    cfg = load_config(env={"JOBSCOPE_CONFIG": str(path)})
    assert cfg.sampling_period == 30


def test_missing_explicit_file_raises():
    with pytest.raises(JobscopeError):
        load_config(path="/nonexistent/jobscope/config.toml", env={})


def _cfg(**kw):
    base = dict(prometheus_url=None, sampling_period=60, sampling_period_explicit=False,
                site_jobstats_config_path=None,
                thresholds=Thresholds(10, 100),
                defaults=Defaults(8, 60.0, 180))
    base.update(kw)
    return Config(**base)


def test_resolve_prometheus_direct_url():
    assert resolve_prometheus(_cfg(prometheus_url="http://p:9090")) == ("http://p:9090", 60)


def test_resolve_prometheus_missing_raises(monkeypatch):
    # Disable auto-discovery so the suite never picks up a real jobstats/config.py
    # on the host running the tests.
    monkeypatch.setattr(config_module, "_discover_site_jobstats_dir", lambda: None)
    with pytest.raises(JobscopeError):
        resolve_prometheus(_cfg())


def test_resolve_prometheus_site_import(monkeypatch):
    monkeypatch.setattr(config_module, "_import_site_prometheus",
                        lambda path, required=True: ("http://site:9090", 30))
    cfg = _cfg(site_jobstats_config_path="/opt/jobstats")
    assert resolve_prometheus(cfg) == ("http://site:9090", 30)


def test_resolve_prometheus_site_keeps_explicit_period(monkeypatch):
    monkeypatch.setattr(config_module, "_import_site_prometheus",
                        lambda path, required=True: ("http://site:9090", 30))
    cfg = _cfg(site_jobstats_config_path="/opt/jobstats",
               sampling_period=120, sampling_period_explicit=True)
    assert resolve_prometheus(cfg) == ("http://site:9090", 120)


def test_resolve_prometheus_auto_discovers_site(monkeypatch, tmp_path):
    # With nothing else configured, jobscope discovers the config next to the
    # jobstats binary on PATH -- no config file or site_jobstats_config_path.
    (tmp_path / "config.py").write_text("PROM_SERVER='http://auto:9090'\nSAMPLING_PERIOD=45\n")
    monkeypatch.setattr(config_module, "_discover_site_jobstats_dir", lambda: str(tmp_path))
    assert resolve_prometheus(_cfg()) == ("http://auto:9090", 45)


def test_resolve_prometheus_explicit_site_overrides_discovery(monkeypatch, tmp_path):
    discovered = tmp_path / "discovered"
    discovered.mkdir()
    (discovered / "config.py").write_text("PROM_SERVER='http://discovered:9090'\n")
    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()
    (explicit_dir / "config.py").write_text("PROM_SERVER='http://explicit:9090'\n")
    monkeypatch.setattr(config_module, "_discover_site_jobstats_dir", lambda: str(discovered))
    url, _ = resolve_prometheus(_cfg(site_jobstats_config_path=str(explicit_dir)))
    assert url == "http://explicit:9090"


def test_discover_site_jobstats_dir(monkeypatch):
    monkeypatch.setattr(config_module.shutil, "which",
                        lambda name: "/opt/jobstats/bin/jobstats" if name == "jobstats" else None)
    assert config_module._discover_site_jobstats_dir() == "/opt/jobstats/bin"
    monkeypatch.setattr(config_module.shutil, "which", lambda name: None)
    assert config_module._discover_site_jobstats_dir() is None


def test_import_site_prometheus_reads_module(tmp_path):
    (tmp_path / "config.py").write_text("PROM_SERVER='http://site:9090'\nSAMPLING_PERIOD=30\n")
    url, sp = config_module._import_site_prometheus(str(tmp_path))
    assert url == "http://site:9090"
    assert sp == 30


def test_import_site_prometheus_reads_by_path_not_sys_path(tmp_path, monkeypatch):
    # A config.py named the same, earlier on sys.path, must not shadow the one at
    # the requested directory: the file is loaded by its full path.
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "config.py").write_text("PROM_SERVER='http://decoy:9090'\n")
    monkeypatch.syspath_prepend(str(decoy))
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.py").write_text("PROM_SERVER='http://target:9090'\n")
    url, _ = config_module._import_site_prometheus(str(target))
    assert url == "http://target:9090"


def test_import_site_prometheus_strict_missing_raises(tmp_path):
    with pytest.raises(JobscopeError):
        config_module._import_site_prometheus(str(tmp_path))  # no config.py present


def test_import_site_prometheus_lenient_missing(tmp_path):
    assert config_module._import_site_prometheus(str(tmp_path), required=False) == (None, None)


def test_import_site_prometheus_missing_prom_server(tmp_path):
    (tmp_path / "config.py").write_text("SAMPLING_PERIOD=30\n")
    assert config_module._import_site_prometheus(str(tmp_path)) == (None, 30)


def test_one_cutoff_covers_every_percent_metric():
    """Uniform on purpose: a reader should not carry a threshold per row.

    The per-metric values were never calibrated against each other, and having five
    of them is what made the summary table's cutoff column confusing.
    """
    t = Thresholds(red=10, power_w=100)
    for header in ("GPU%", "CPU%", "MEM%", "GMEM%", "SM_ACT%", "OCC%", "DRAM%"):
        assert t.cutoff(header) == 10
        assert t.grade(header, 9) == "red" and t.grade(header, 25) == "green"
    assert t.cutoff("POWER_W") == 100        # watts, its own knob
    assert t.cutoff("RUNTIME") is None       # not graded


def test_power_is_graded_in_watts_not_percent():
    """The one graded column that is not a percentage.

    A GPU below the floor is idle, which is the signal a duty cycle cannot fake: a
    job spinning on a trivial kernel reads busy on GPU% and draws idle watts.
    """
    t = Thresholds(red=10, power_w=100)
    assert t.grade("POWER_W", 73) == "red"        # measured idle floor
    assert t.grade("POWER_W", 135) == "yellow"    # below twice the floor
    assert t.grade("POWER_W", 289) == "green"     # the measured median
    # Without its own cutoff it would fall to `default`, i.e. 15 *watts*, and
    # nothing is ever below that -- every job would read green.
    assert grade_band(73, 10) == "green"


def test_power_floor_comes_from_the_config_file(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[thresholds]\npower_w = 150\n")
    assert load_config(str(path)).thresholds.power_w == 150.0


def test_a_column_with_no_cutoff_and_no_percent_is_ungraded():
    t = Thresholds(red=10)
    assert t.grade("RUNTIME", 5) == "" and t.grade("ENERGY_kWh", 0.3) == ""


def test_example_config_text():
    text = example_config_text()
    assert "[prometheus]" in text
    assert "JOBSCOPE_PROM_URL" in text


def test_a_config_still_setting_the_old_per_metric_keys_is_told(tmp_path, capsys):
    """Silently moving a tuned site to a different cutoff would be worse than noisy."""
    path = tmp_path / "c.toml"
    path.write_text("[thresholds]\ngpu = 25\nmem = 30\n")
    cfg = load_config(str(path))
    err = capsys.readouterr().err
    assert "gpu, mem no longer apply" in err and "'red'" in err
    assert cfg.thresholds.red == 10          # and the uniform cutoff is what applies
