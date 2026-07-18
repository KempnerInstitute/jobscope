"""Configuration loading for jobscope.

Site-specific settings -- the Prometheus endpoint above all -- come from a TOML
file, an environment variable, or, for backward compatibility, an existing
jobstats ``config`` module. Nothing here ever prints the resolved Prometheus
URL, which commonly embeds a credential.
"""

import importlib
import importlib.resources
import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple

from .errors import JobscopeError

try:
    import tomllib as _toml
except ModuleNotFoundError:
    import tomli as _toml

CONFIG_ENV = "JOBSCOPE_CONFIG"
PROM_URL_ENV = "JOBSCOPE_PROM_URL"

DEFAULT_SAMPLING_PERIOD = 60
DEFAULT_WORKERS = 8
DEFAULT_TIMEOUT = 60.0
DEFAULT_MIN_RUNTIME = 180
DEFAULT_THRESHOLDS = {"gpu": 25.0, "gmem": 20.0, "cpu": 25.0, "mem": 25.0, "default": 15.0}


@dataclass(frozen=True)
class Thresholds:
    """Red cutoffs (percent) for plot color grading."""

    gpu: float
    gmem: float
    cpu: float
    mem: float
    default: float

    def red_map(self) -> dict:
        """Per-header red cutoffs for the graded percent columns."""
        return {"GPU%": self.gpu, "DUTY%": self.gpu, "GMEM%": self.gmem,
                "CPU%": self.cpu, "MEM%": self.mem}


@dataclass(frozen=True)
class Defaults:
    """Default values for CLI options that a site may want to override."""

    workers: int
    timeout: float
    min_runtime: int


@dataclass(frozen=True)
class Config:
    """Resolved jobscope settings."""

    prometheus_url: Optional[str]
    sampling_period: int
    sampling_period_explicit: bool
    site_jobstats_config_path: Optional[str]
    thresholds: Thresholds
    defaults: Defaults
    source_path: Optional[Path] = None


def default_config_path(env: Optional[Mapping[str, str]] = None) -> Path:
    """Path jobscope reads when neither an explicit path nor $JOBSCOPE_CONFIG is set."""
    env = os.environ if env is None else env
    base = env.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "jobscope" / "config.toml"


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as fh:
        return _toml.load(fh)


def load_config(path: Optional[str] = None,
                env: Optional[Mapping[str, str]] = None) -> Config:
    """Build a Config from a TOML file plus environment overrides.

    Resolution order for the file: ``path`` argument, then ``$JOBSCOPE_CONFIG``,
    then :func:`default_config_path`. A file named explicitly (argument or env
    var) must exist; the default path may be absent, in which case built-in
    defaults apply. The Prometheus URL prefers ``$JOBSCOPE_PROM_URL`` over the file.
    """
    env = os.environ if env is None else env
    explicit = path if path is not None else env.get(CONFIG_ENV)
    if explicit:
        chosen = Path(explicit)
        if not chosen.exists():
            raise JobscopeError("jobscope config not found: %s" % chosen)
    else:
        chosen = default_config_path(env)
    data = _read_toml(chosen) if chosen.exists() else {}

    prom = data.get("prometheus") or {}
    thr = data.get("thresholds") or {}
    dfl = data.get("defaults") or {}

    thresholds = Thresholds(
        gpu=float(thr.get("gpu", DEFAULT_THRESHOLDS["gpu"])),
        gmem=float(thr.get("gmem", DEFAULT_THRESHOLDS["gmem"])),
        cpu=float(thr.get("cpu", DEFAULT_THRESHOLDS["cpu"])),
        mem=float(thr.get("mem", DEFAULT_THRESHOLDS["mem"])),
        default=float(thr.get("default", DEFAULT_THRESHOLDS["default"])),
    )
    defaults = Defaults(
        workers=int(dfl.get("workers", DEFAULT_WORKERS)),
        timeout=float(dfl.get("timeout", DEFAULT_TIMEOUT)),
        min_runtime=int(dfl.get("min_runtime", DEFAULT_MIN_RUNTIME)),
    )
    return Config(
        prometheus_url=(env.get(PROM_URL_ENV) or prom.get("url")) or None,
        sampling_period=int(prom.get("sampling_period", DEFAULT_SAMPLING_PERIOD)),
        sampling_period_explicit="sampling_period" in prom,
        site_jobstats_config_path=prom.get("site_jobstats_config_path"),
        thresholds=thresholds,
        defaults=defaults,
        source_path=chosen if chosen.exists() else None,
    )


def _discover_site_jobstats_dir() -> Optional[str]:
    """Directory of the ``jobstats`` binary on ``PATH``, expected to hold the
    site ``config.py`` (with ``PROM_SERVER``); None when jobstats isn't on PATH.

    Keying on the jobstats binary keeps jobscope site-agnostic -- nothing
    site-specific ships in the package -- and is a strong signal: we only import
    a ``config.py`` that sits beside a real jobstats install the user already has,
    never a stray file at a fixed system path.
    """
    path = shutil.which("jobstats")
    return os.path.dirname(path) if path else None


def resolve_prometheus(cfg: Config) -> Tuple[str, int]:
    """Return ``(url, sampling_period)`` for Prometheus.

    When no URL is configured directly, the site jobstats config supplies it: an
    explicit ``site_jobstats_config_path`` (which must import cleanly), otherwise
    the directory of the ``jobstats`` binary auto-discovered on ``PATH`` (skipped
    silently when it holds no usable config). Raises :class:`JobscopeError` with
    actionable guidance when none is available. The URL can embed a credential,
    so callers must never log or print it.
    """
    url = cfg.prometheus_url
    sampling_period = cfg.sampling_period
    if not url:
        explicit = cfg.site_jobstats_config_path
        site_path = explicit or _discover_site_jobstats_dir()
        if site_path:
            site_url, site_sp = _import_site_prometheus(site_path, required=bool(explicit))
            if site_url:
                url = site_url
                if site_sp and not cfg.sampling_period_explicit:
                    sampling_period = int(site_sp)
    if not url:
        raise JobscopeError(_no_endpoint_message(cfg))
    return url, sampling_period


def _import_site_prometheus(config_path: str,
                            required: bool = True) -> Tuple[Optional[str], Optional[int]]:
    """Read ``(PROM_SERVER, SAMPLING_PERIOD)`` from a jobstats ``config.py``.

    The file at ``<config_path>/config.py`` is loaded directly by path, so it
    never depends on ``sys.path`` order and never shadows (or is shadowed by)
    another module named ``config``. With ``required=False`` a missing or
    unimportable file yields ``(None, None)`` instead of raising -- used for the
    automatic default path, which must not break jobscope where it is absent.
    """
    cfg_file = os.path.join(config_path, "config.py")
    if not os.path.isfile(cfg_file):
        if required:
            raise JobscopeError("no jobstats config.py found in %r" % config_path)
        return None, None
    try:
        spec = importlib.util.spec_from_file_location("_jobscope_site_config", cfg_file)
        site = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(site)
    except Exception as exc:
        if required:
            raise JobscopeError(
                "could not import the site jobstats config from %r: %s" % (config_path, exc))
        return None, None
    return getattr(site, "PROM_SERVER", None), getattr(site, "SAMPLING_PERIOD", None)


def _no_endpoint_message(cfg: Config) -> str:
    target = cfg.source_path or default_config_path()
    return (
        "no Prometheus endpoint configured; the GPU and DCGM views require one.\n"
        "(No 'jobstats' binary with a usable config.py was found on your PATH.)\n"
        "Fix any one of:\n"
        "  - set the JOBSCOPE_PROM_URL environment variable, or\n"
        '  - add [prometheus] url = "https://.../api/prom" to %s, or\n'
        "  - set [prometheus] site_jobstats_config_path to a dir holding a jobstats config.py.\n"
        "The offline --cpu / --cgpu views need no Prometheus." % target)


_active: Optional[Config] = None


def get_config() -> Config:
    """Return the process-wide Config, loading it from disk on first use."""
    global _active
    if _active is None:
        _active = load_config()
    return _active


def set_config(cfg: Config) -> None:
    """Install a Config (used by the CLI after --config, and by tests)."""
    global _active
    _active = cfg


def reset_config() -> None:
    """Forget the cached Config so the next :func:`get_config` reloads it."""
    global _active
    _active = None


def example_config_text() -> str:
    """The bundled config.example.toml as text."""
    return (importlib.resources.files("jobscope") / "config.example.toml").read_text()
