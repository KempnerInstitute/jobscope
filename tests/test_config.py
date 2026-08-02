"""Tests for configuration loading and Prometheus endpoint resolution."""

import pytest

from jobscope import config as config_module
from jobscope.config import (
    Config,
    Defaults,
    Thresholds,
    example_config_text,
    floor_band,
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
power_w = 250

[thresholds.summary.wasteful]
default = 5
cpu = 7
[thresholds.summary.inefficient]
default = 40
[thresholds.summary.improvement]
default = 60
[thresholds.summary.average]
default = 90

[thresholds.timeslice.wasteful]
default = 3

[defaults]
workers = 4
timeout = 15
"""


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(env={"XDG_CONFIG_HOME": str(tmp_path)})
    assert cfg.prometheus_url is None
    assert cfg.sampling_period == 60
    assert cfg.defaults.workers == 8
    assert cfg.thresholds.edges() == (2, 10, 20, 40)
    assert cfg.timeslice_thresholds.edges() == (2, 10, 20, 40)
    assert cfg.thresholds.power_w == 100


def test_load_from_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG_TOML)
    cfg = load_config(path=str(path), env={})
    assert cfg.prometheus_url == "https://file.example/api/prom"
    assert cfg.sampling_period == 30
    assert cfg.sampling_period_explicit is True
    assert cfg.site_jobstats_config_path == "/opt/jobstats"
    assert cfg.thresholds.edges() == (5, 40, 60, 90)
    assert cfg.thresholds.edges("CPU%") == (7, 40, 60, 90)   # its own wasteful edge
    assert cfg.thresholds.power_w == 250
    assert cfg.defaults == Defaults(workers=4, timeout=15.0)


def test_the_two_views_are_independent(tmp_path):
    """Nothing is inherited between them: the time slice keeps the built-in edges
    for everything the summary tuned, and shares only POWER_W's floor."""
    path = tmp_path / "config.toml"
    path.write_text(CONFIG_TOML)
    cfg = load_config(path=str(path), env={})
    assert cfg.timeslice_thresholds.edges() == (3, 10, 20, 40)   # its own, not (5,40,60,90)
    assert cfg.timeslice_thresholds.edges("CPU%") == (3, 10, 20, 40)   # summary's cpu=7 did not carry
    assert cfg.timeslice_thresholds.power_w == 250               # the floor is shared


def test_a_view_left_out_keeps_the_built_in_edges(tmp_path, capsys):
    """And says so once, since the two then grade the same job differently."""
    path = tmp_path / "c.toml"
    path.write_text("[thresholds.summary.wasteful]\ncpu = 5\n")
    cfg = load_config(str(path))
    assert cfg.thresholds.edge("wasteful", "CPU%") == 5
    assert cfg.timeslice_thresholds.edge("wasteful", "CPU%") == 2
    err = capsys.readouterr().err
    assert "[thresholds.summary] is set but [thresholds.timeslice] is not" in err


def test_both_views_set_says_nothing(tmp_path, capsys):
    path = tmp_path / "c.toml"
    path.write_text("[thresholds.summary.wasteful]\ncpu = 5\n"
                    "[thresholds.timeslice.wasteful]\ncpu = 8\n")
    load_config(str(path))
    assert "is not" not in capsys.readouterr().err


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
                thresholds=Thresholds(),
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


def test_the_default_edges_cover_every_percent_metric():
    """An unconfigured metric falls to `default`, which is what keeps the ~18 extra
    %-columns under --dcgm graded without any of them being listed."""
    t = Thresholds()
    for header in ("GPU%", "CPU%", "MEM%", "GMEM%", "SM_ACT%", "OCC%", "DRAM%"):
        assert t.tier(header, 9) == "inefficient"
        assert t.grade(header, 1) == "red"       # wasteful
        assert t.grade(header, 9) == "red"       # inefficient
        assert t.grade(header, 15) == "yellow"   # needs improvement
        assert t.grade(header, 25) == "green"    # average
        assert t.grade(header, 60) == "green"    # good
    assert t.grade("RUNTIME", 5) == ""            # not graded


def test_a_metric_is_graded_by_its_own_edges():
    """The point of the whole table: 4% is idle for a GPU and ordinary for a host."""
    t = Thresholds(by_metric={"CPU%": {"wasteful": 5.0}})
    assert t.tier("CPU%", 4.0) == "wasteful"
    assert t.tier("GPU%", 4.0) == "inefficient"      # same number, its own edge
    assert t.edge("wasteful", "CPU%") == 5 and t.edge("wasteful", "GPU%") == 2


def test_a_metric_naming_some_edges_falls_back_for_the_rest():
    t = Thresholds(by_metric={"CPU%": {"wasteful": 5.0}})
    assert t.edges("CPU%") == (5, 10, 20, 40)


def test_partial_defaults_are_filled_in():
    """A caller (or a TOML table) may name one edge; the other three still resolve."""
    assert Thresholds(defaults={"wasteful": 8}).edges() == (8, 10, 20, 40)


def test_power_is_never_tiered():
    """It is watts against a floor, so it has no place on the percent scale --
    tier() says so rather than banding 219 W as a very good percentage."""
    assert Thresholds().tier("POWER_W", 219.0) == ""


@pytest.mark.parametrize("written,header", [
    ("gpu", "GPU%"), ("GPU", "GPU%"), ("GPU%", "GPU%"), ("sm_act", "SM_ACT%"),
    ("cpu", "CPU%"), (" Dram ", "DRAM%"),
])
def test_metric_keys_are_written_the_way_metrics_are_talked_about(written, header):
    assert config_module.metric_header(written) == header


def test_power_has_no_yellow_band():
    """A floor asserts one thing -- below this is idle -- so it answers one.

    Yellow means "close to the cutoff", which for a percentage is up to twice it.
    An absolute floor has no such headroom: at 330 W, twice is 660 W and the card
    that floor exists for tops out near 480, so it could never read green.
    """
    t = Thresholds(power_w=100, power_w_by_model={"RTX": 330})
    assert {t.grade("POWER_W", w) for w in (0, 99, 135, 289, 700)} == {"red", "green"}
    assert t.grade("POWER_W", 100) == "green"     # at the floor, not below it
    assert floor_band(480, 330) == "green"        # the case that had no green at all
    # Percentages keep all five tiers: there is no floor, just a target to clear.
    assert t.grade("GPU%", 15) == "yellow"


def test_power_is_graded_in_watts_not_percent():
    """The one graded column that is not a percentage.

    A GPU below the floor is idle, which is the signal a duty cycle cannot fake: a
    job spinning on a trivial kernel reads busy on GPU% and draws idle watts.
    """
    t = Thresholds(power_w=100)
    assert t.grade("POWER_W", 73) == "red"        # measured idle floor
    assert t.grade("POWER_W", 289) == "green"     # the measured median


def test_power_floor_comes_from_the_config_file(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[thresholds]\npower_w = 150\n")
    assert load_config(str(path)).thresholds.power_w == 150.0


def test_a_column_with_no_percent_is_ungraded():
    t = Thresholds()
    assert t.grade("RUNTIME", 5) == "" and t.grade("ENERGY_kWh", 0.3) == ""


def test_example_config_text():
    text = example_config_text()
    assert "[prometheus]" in text
    assert "JOBSCOPE_PROM_URL" in text


def test_a_config_still_setting_the_old_top_level_keys_is_told(tmp_path, capsys):
    """Silently ignoring a tuned site's numbers would be worse than noisy -- and the
    note has to name the new home, because for the metric keys there is one."""
    path = tmp_path / "c.toml"
    path.write_text("[thresholds]\ngpu = 25\nmem = 30\n")
    cfg = load_config(str(path))
    err = capsys.readouterr().err
    assert "gpu, mem no longer apply at the top level" in err
    assert "[thresholds.summary.<edge>]" in err
    assert cfg.thresholds.edges() == (2, 10, 20, 40)   # the built-in edges apply


def test_the_flat_edge_keys_are_now_stale_too(tmp_path, capsys):
    """They lived at the top level for one release; they are per view now."""
    path = tmp_path / "c.toml"
    path.write_text("[thresholds]\nred = 25\ncpu = 5\nwasteful = 3\naverage = 50\n")
    cfg = load_config(str(path))
    err = capsys.readouterr().err
    assert "average, cpu, red, wasteful no longer apply at the top level" in err
    assert cfg.thresholds.edges() == (2, 10, 20, 40)


def test_power_w_is_not_flagged_as_stale(tmp_path, capsys):
    """It is the one top-level key that still means something."""
    path = tmp_path / "c.toml"
    path.write_text("[thresholds]\npower_w = 150\n")
    load_config(str(path))
    assert "no longer apply" not in capsys.readouterr().err


# --- rejecting a config that cannot mean what it says ------------------------

def test_the_old_block_scoped_one_level_short_is_rejected(tmp_path):
    """The likely migration slip: `[thresholds.summary] wasteful = 2` puts a number
    where a per-metric table belongs."""
    path = tmp_path / "c.toml"
    path.write_text("[thresholds.summary]\nwasteful = 2\n")
    with pytest.raises(JobscopeError, match="must be a table of per-metric cutoffs"):
        load_config(str(path))


def test_an_unknown_edge_name_is_rejected(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[thresholds.summary.wastefull]\ndefault = 2\n")
    with pytest.raises(JobscopeError, match="has no edge wastefull"):
        load_config(str(path))


def test_a_non_numeric_cutoff_is_rejected(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[thresholds.summary.wasteful]\ngpu = "two"\n')
    with pytest.raises(JobscopeError, match="is not a number"):
        load_config(str(path))


def test_edges_that_decrease_are_rejected(tmp_path):
    """Checked on the resolved edges, not the lines written: naming only
    `inefficient` composes it with the default `wasteful` and can invert them."""
    path = tmp_path / "c.toml"
    path.write_text("[thresholds.summary.inefficient]\ncpu = 1\n")
    with pytest.raises(JobscopeError, match=r"CPU% edges must not decrease"):
        load_config(str(path))


def test_equal_edges_are_allowed(tmp_path):
    """They collapse a band, which is a coherent thing to ask for."""
    path = tmp_path / "c.toml"
    path.write_text("[thresholds.summary.wasteful]\ndefault = 10\n")
    assert load_config(str(path)).thresholds.edges() == (10, 10, 20, 40)


def test_an_unknown_metric_key_is_dropped_with_a_note(tmp_path, capsys):
    """The %-normalisation would otherwise turn `gpuu` into a GPUU% nothing asks
    about, and the typo would look like a setting that simply had no effect."""
    path = tmp_path / "c.toml"
    path.write_text("[thresholds.summary.wasteful]\ngpuu = 2\n")
    cfg = load_config(str(path))
    assert "names no metric GPUU%" in capsys.readouterr().err
    assert dict(cfg.thresholds.by_metric) == {}


def test_the_blob_backed_metrics_are_known(tmp_path, capsys):
    """CPU% and MEM% are not in the DCGM catalog the validator is built from, and
    cpu is the single likeliest key a site sets."""
    path = tmp_path / "c.toml"
    path.write_text("[thresholds.summary.wasteful]\ncpu = 5\nmem = 3\n")
    cfg = load_config(str(path))
    assert "names no metric" not in capsys.readouterr().err
    assert cfg.thresholds.edge("wasteful", "CPU%") == 5
    assert cfg.thresholds.edge("wasteful", "MEM%") == 3


# --- per-architecture POWER_W floor -----------------------------------------

def test_the_power_floor_can_differ_per_gpu_model():
    """Idle draw is hardware, not policy: 27 W on a V100, 165 W on an RTX PRO 6000.

    One number is wrong at one end or the other, so the floor is looked up per model.
    """
    t = Thresholds(power_w=100, power_w_by_model={
        "NVIDIA RTX PRO 6000 Blackwell Server Edition": 330,
        "Tesla V100-PCIE-32GB": 45})
    assert t.floor_for("NVIDIA RTX PRO 6000 Blackwell Server Edition") == 330
    assert t.floor_for("Tesla V100-PCIE-32GB") == 45


@pytest.mark.parametrize("model", ["NVIDIA H100 80GB HBM3", None, ""])
def test_an_unlisted_or_unknown_model_falls_back_to_the_global_floor(model):
    t = Thresholds(power_w=100, power_w_by_model={"Tesla V100-PCIE-32GB": 45})
    assert t.floor_for(model) == 100


def test_the_per_model_table_is_read_from_config(tmp_path, monkeypatch):
    path = tmp_path / "c.toml"
    path.write_text('[thresholds]\npower_w = 100\n\n'
                    '[thresholds.power_w_by_model]\n"NVIDIA A40" = 40\n')
    cfg = config_module.load_config(path=str(path))
    assert cfg.thresholds.floor_for("NVIDIA A40") == 40
    assert cfg.thresholds.floor_for("NVIDIA A100-SXM4-40GB") == 100


def test_no_per_model_table_is_not_an_error(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[thresholds]\npower_w = 100\n")
    loaded = config_module.load_config(path=str(path))
    assert loaded.thresholds.floor_for("anything") == 100


# --- [metrics]: which metrics each view shows --------------------------------

@pytest.mark.parametrize("written", ["gpu", "GPU", "GPU%", "duty", " Gpu "])
def test_a_metric_is_nameable_by_any_of_its_forms(written, tmp_path):
    """The short header form, the header itself, and the catalog key all resolve,
    so a reader who knows the column can name it without learning a second word."""
    path = tmp_path / "c.toml"
    path.write_text('[metrics]\ntimeseries = ["%s"]\n' % written)
    specs = load_config(str(path)).metrics.timeseries
    assert [s.header for s in specs] == ["GPU%"]


def test_metrics_print_in_catalog_order_not_the_order_written(tmp_path):
    """Column order is a property of the report, not of how a site listed them --
    and it is what keeps a narrow selection a prefix of a wider one."""
    path = tmp_path / "c.toml"
    path.write_text('[metrics]\ntimeseries = ["dram", "gpu", "power", "sm_act"]\n')
    specs = load_config(str(path)).metrics.timeseries
    assert [s.header for s in specs] == ["GPU%", "SM_ACT%", "DRAM%", "POWER_W"]


def test_extended_takes_all_or_a_list(tmp_path):
    from jobscope.dcgm import ALL_SPECS
    path = tmp_path / "c.toml"
    path.write_text('[metrics]\nextended = "all"\n')
    assert len(load_config(str(path)).metrics.extended) == len(ALL_SPECS)
    path.write_text('[metrics]\nextended = ["gpu", "temp"]\n')
    # The blob-backed trio is added back (see below), so TEMP_C is what to check.
    assert "TEMP_C" in [s.header for s in load_config(str(path)).metrics.extended]


def test_an_unnamed_view_keeps_its_built_in_list(tmp_path):
    from jobscope.dcgm import DEFAULT_SPECS, KEY_SPECS
    path = tmp_path / "c.toml"
    path.write_text('[metrics]\ntimeseries = ["gpu"]\n')
    cfg = load_config(str(path))
    assert [s.header for s in cfg.metrics.summary] == [s.header for s in DEFAULT_SPECS]
    assert [s.header for s in cfg.metrics.timeseries] != [s.header for s in KEY_SPECS]


def test_the_blob_backed_metrics_cannot_be_dropped_from_the_summary(tmp_path):
    """GPU% and GMEM% are fixed columns of the summary and detail tables, not part
    of the configurable profiling block, and in the *running* view they come from
    Prometheus. Omitting them does not narrow the output, it blanks two columns --
    so the list is corrected rather than obeyed."""
    path = tmp_path / "c.toml"
    path.write_text('[metrics]\nsummary = ["sm_act"]\n')
    cfg = load_config(str(path))
    keys = {s.key for s in cfg.metrics.summary}
    assert {"duty", "mem", "memtot"} <= keys      # added back
    assert "smact" in keys                        # and what was asked for is kept
    # --ts has no fixed columns, so there a narrow list means exactly what it says.
    path.write_text('[metrics]\ntimeseries = ["sm_act"]\n')
    assert [s.header for s in load_config(str(path)).metrics.timeseries] == ["SM_ACT%"]


def test_an_unknown_metric_name_is_rejected_with_the_catalog(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[metrics]\nsummary = ["gpuu"]\n')
    with pytest.raises(JobscopeError, match="names no metric 'gpuu'"):
        load_config(str(path))


@pytest.mark.parametrize("body,match", [
    ('[metrics]\nsummary = "sm_act"\n', "must be a list"),
    ('[metrics]\nsummary = []\n', "is empty"),
    ('[metrics]\ndetail = ["gpu"]\n', "has no detail"),
])
def test_metrics_shape_errors(body, match, tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(body)
    with pytest.raises(JobscopeError, match=match):
        load_config(str(path))


# --- [colors] ---------------------------------------------------------------

def test_each_tier_gets_its_own_colour(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[colors]\nwasteful = "bright_red"\naverage = "cyan"\n')
    palette = load_config(str(path)).palette
    assert palette.for_tier("wasteful") == "bright_red"
    assert palette.for_tier("average") == "cyan"
    assert palette.for_tier("inefficient") == "red"      # untouched
    # "needs improvement" is spelled `improvement` in the config; both resolve.
    assert palette.for_tier("needs improvement") == "yellow"
    assert palette.for_tier("improvement") == "yellow"


def test_a_bucket_takes_the_colour_of_the_least_bad_tier_it_covers():
    """The three band columns span two tiers each, so one of them has to stand in --
    the milder, so "red" does not read as the more alarming wasteful."""
    from jobscope.config import Palette
    p = Palette(colors={"wasteful": "bright_red", "good": "blue"})
    assert p.for_bucket("red") == "red"        # inefficient's, not wasteful's
    assert p.for_bucket("green") == "blue"     # good's


def test_the_sgr_table_resolves_every_role_a_renderer_may_ask_for():
    """Call sites hold different things: a bucket for a cell, a tier for a
    --classify heading, long_running for a Wasteful entry."""
    from jobscope.config import Palette
    sgr = Palette(colors={"average": "cyan"}).sgr()
    for role in ("red", "yellow", "green", "wasteful", "needs improvement",
                 "improvement", "average", "good", "long_running"):
        assert sgr[role].startswith("\033["), role
    assert sgr["average"] == "\033[36m"        # cyan


def test_256_colour_codes_are_accepted(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[colors]\ngood = "color(33)"\n')
    palette = load_config(str(path)).palette
    assert palette.for_tier("good") == "color(33)"
    assert palette.sgr()["good"] == "\033[38;5;33m"


@pytest.mark.parametrize("body,match", [
    ('[colors]\nwasteful = "puce"\n', "is not a colour"),
    ('[colors]\nterrible = "red"\n', "has no terrible"),
])
def test_colour_errors(body, match, tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(body)
    with pytest.raises(JobscopeError, match=match):
        load_config(str(path))


# --- [defaults] additions ---------------------------------------------------

def test_the_new_defaults_are_read(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[defaults]\ndays = 7\nstate = "all"\nworst_jobs = 5\n'
                    'long_running = "90m"\n')
    dfl = load_config(str(path)).defaults
    assert (dfl.days, dfl.state, dfl.worst_jobs, dfl.long_running) == (7, "all", 5, "90m")


def test_the_state_default_is_checked_by_the_same_rule_as_the_flag(tmp_path):
    """Delegated to sacct.states_for, so the config and -t cannot come to disagree
    about the vocabulary -- including its refusal of live states."""
    path = tmp_path / "c.toml"
    path.write_text('[defaults]\nstate = "running"\n')
    with pytest.raises(JobscopeError, match="not a finished state"):
        load_config(str(path))
    path.write_text('[defaults]\nstate = "completed,failed"\n')   # comma-separated is fine
    assert load_config(str(path)).defaults.state == "completed,failed"


def test_worst_jobs_must_be_at_least_one(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[defaults]\nworst_jobs = 0\n")
    with pytest.raises(JobscopeError, match="at least 1"):
        load_config(str(path))


def test_the_shipped_example_reproduces_the_built_in_behaviour():
    """It spells every knob out, so it must spell out exactly the defaults --
    otherwise copying the template silently changes how jobs are graded."""
    import tempfile
    from jobscope.config import Metrics, Palette
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(example_config_text())
        path = fh.name
    cfg = load_config(path)
    assert [s.header for s in cfg.metrics.summary] == \
        [s.header for s in Metrics().summary]
    assert [s.header for s in cfg.metrics.timeseries] == \
        [s.header for s in Metrics().timeseries]
    assert dict(cfg.palette.colors) == dict(Palette().colors)
    assert (cfg.defaults.days, cfg.defaults.state, cfg.defaults.worst_jobs,
            cfg.defaults.long_running) == (1, "completed", 3, "3h")
