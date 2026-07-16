"""Tests for configuration loading and Prometheus endpoint resolution."""

import sys

import pytest

from jobscope import config as config_module
from jobscope.config import (
    Config,
    Defaults,
    Thresholds,
    example_config_text,
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
gpu = 40
default = 5

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
    assert cfg.thresholds.gpu == 25


def test_load_from_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG_TOML)
    cfg = load_config(path=str(path), env={})
    assert cfg.prometheus_url == "https://file.example/api/prom"
    assert cfg.sampling_period == 30
    assert cfg.sampling_period_explicit is True
    assert cfg.site_jobstats_config_path == "/opt/jobstats"
    assert cfg.thresholds.gpu == 40
    assert cfg.thresholds.default == 5
    assert cfg.thresholds.gmem == 20  # unset -> built-in default
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
                thresholds=Thresholds(25, 20, 25, 25, 15),
                defaults=Defaults(8, 60.0, 180))
    base.update(kw)
    return Config(**base)


def test_resolve_prometheus_direct_url():
    assert resolve_prometheus(_cfg(prometheus_url="http://p:9090")) == ("http://p:9090", 60)


def test_resolve_prometheus_missing_raises():
    with pytest.raises(JobscopeError):
        resolve_prometheus(_cfg())


def test_resolve_prometheus_site_import(monkeypatch):
    monkeypatch.setattr(config_module, "_import_site_prometheus",
                        lambda path: ("http://site:9090", 30))
    cfg = _cfg(site_jobstats_config_path="/opt/jobstats")
    assert resolve_prometheus(cfg) == ("http://site:9090", 30)


def test_resolve_prometheus_site_keeps_explicit_period(monkeypatch):
    monkeypatch.setattr(config_module, "_import_site_prometheus",
                        lambda path: ("http://site:9090", 30))
    cfg = _cfg(site_jobstats_config_path="/opt/jobstats",
               sampling_period=120, sampling_period_explicit=True)
    assert resolve_prometheus(cfg) == ("http://site:9090", 120)


def test_import_site_prometheus_reads_module(tmp_path, monkeypatch):
    (tmp_path / "config.py").write_text("PROM_SERVER='http://site:9090'\nSAMPLING_PERIOD=30\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("config", None)
    try:
        url, sp = config_module._import_site_prometheus(str(tmp_path))
        assert url == "http://site:9090"
        assert sp == 30
    finally:
        sys.modules.pop("config", None)


def test_thresholds_red_map():
    red = Thresholds(25, 20, 25, 25, 15).red_map()
    assert red == {"GPU%": 25, "DUTY%": 25, "GMEM%": 20, "CPU%": 25, "MEM%": 25}


def test_example_config_text():
    text = example_config_text()
    assert "[prometheus]" in text
    assert "JOBSCOPE_PROM_URL" in text
