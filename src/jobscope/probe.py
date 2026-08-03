"""``jobscope probe`` -- what this cluster exposes, and whether jobscope can read it.

Two jobs, deliberately kept in one command because a new site needs both at once:

* **Diagnosis.** Is Slurm answering, is a config file being read, is the Prometheus
  endpoint reachable, are the label conventions the ones the collectors assume.
  Every check is independently failable -- a missing endpoint must not stop the
  Slurm section from reporting, since the whole point is to run *before* anything
  works.
* **Discovery.** Which metrics this server actually carries for a real job, printed
  with the short name ``[metrics]`` and ``[thresholds]`` take. jobscope's catalog is
  what *some* cluster exported; a different site has a different set, and until you
  can see the difference you cannot configure around it.

Nothing here writes: ``probe`` is safe to run anywhere, and the flag that edits a
config file is a separate, later thing.

Two findings from this site shaped the code. The backend is Grafana Mimir, not
Prometheus, so ``/api/v1/status/tsdb`` and ``/status/runtimeinfo`` both 404 and
retention cannot be read from an API -- it has to be probed. And a TRES appearing
in ``AccountingStorageTRES`` does not mean it is populated: ``energy`` is listed
here while ``ConsumedEnergy`` reads 0, because ``AcctGatherEnergyType`` is null.
So every capability is confirmed against a real job rather than believed from
configuration.
"""

import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

from . import config, extra_metric
from .cpu import CGROUP_METRICS
from .dcgm import METRICS as GPU_METRICS
from .errors import JobscopeError
from .slurm import run_capture

# Status markers. Deliberately words rather than colour: probe output gets pasted
# into issues and email, where colour does not survive.
OK = "ok"
ABSENT = "absent"
NEW = "new"
FAIL = "FAIL"
WARN = "warn"

# How far back to look for data when probing retention, deepest first. Reported as
# "found at N", never as an exact edge: a gap at one instant (a scrape outage, a
# restart) is indistinguishable from the end of retention with a single query, so
# claiming a precise boundary would be a guess dressed as a measurement.
#
# Deepest first because the walk stops at the first hit, and that ordering is both
# cheaper and more accurate. A query that finds nothing returns in ~0.1s where one
# that finds something costs 2-3s against long-term storage, so the misses are
# nearly free; and stopping at the first hit reports the *deepest* data found
# rather than the shallowest.
RETENTION_LADDER_DAYS = (730, 365, 180, 90, 60, 30, 14, 7, 1)

# The window the capability sample covers. Two hours of one cluster is a few
# thousand jobs and returns in ~0.2s, where two days is 114k rows and 5s -- and
# every question here ("are blobs being written", "find me a GPU job") is answered
# just as well by a recent sample as by an exhaustive one. Widened once when a
# quiet cluster returns nothing.
SAMPLE_WINDOWS = ("now-2hours", "now-2days")

# The series each family is recognised by, longest prefix first so DCGM's two
# sub-prefixes strip before the bare one.
FAMILY_PREFIXES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("cgroup", ("cgroup_",)),
    ("nvml", ("nvidia_gpu_",)),
    ("dcgm", ("DCGM_FI_PROF_", "DCGM_FI_DEV_", "DCGM_FI_")),
)


def _catalog() -> Dict[str, Tuple[str, str]]:
    """``{prometheus series: (family, short name)}`` for everything jobscope knows.

    Built from the two catalogs rather than spelled out, so a metric added to
    either is recognised here immediately. One series can back more than one spec
    (``DCGM_FI_DEV_POWER_USAGE`` feeds both POWER_W and PWRmax_W); the first wins,
    which is the ``default``-group one and so the name a reader expects.
    """
    found: Dict[str, Tuple[str, str]] = {}
    for spec in CGROUP_METRICS:
        found.setdefault(spec.metric, ("cgroup", spec.key))
    for spec in GPU_METRICS:
        family = spec.family
        header = spec.header.lower()
        short = header[:-1] if header.endswith("%") else spec.key
        found.setdefault(spec.metric, (family, short))
    return found


def catalog() -> Dict[str, Tuple[str, str]]:
    """The catalog as it stands *now*, including anything ``[metrics]`` defined.

    Computed per call rather than cached at import: config-defined metrics are
    registered at load time, which is after this module is imported, and a stale
    snapshot would report a metric the site had just named as still unnamed --
    working everywhere except in the command whose job is to tell you about it.
    ``--metrics`` runs once per invocation, so there is nothing to cache.
    """
    return _catalog()


def family_of(raw: str) -> Optional[str]:
    """The family a raw series name belongs to, or None if jobscope cannot place it."""
    for family, prefixes in FAMILY_PREFIXES:
        if any(raw.startswith(prefix) for prefix in prefixes):
            return family
    return None


def catalog_name(raw: str) -> Optional[str]:
    """The name config accepts for a series, or None if jobscope has none.

    Distinct from :func:`simple_name`, which will happily *derive* a plausible name
    for an unknown series. That derivation is right for suggesting what to call
    something and wrong for a listing: a derived name is indistinguishable from a
    real one, and config would reject it as unknown. Only catalogued series have a
    name, so only they get one here.
    """
    known = catalog().get(raw)
    return "%s-%s" % known if known else None


def name_parts(raw: str) -> Optional[Tuple[str, str]]:
    """``(family, short name)`` for a series, or None if jobscope cannot place it.

    Split out because the two halves are wanted separately: the listing joins them
    with a hyphen, while a ``[metrics.<family>.<name>]`` table already carries the
    family in its path and needs only the short half as its key.
    """
    known = catalog().get(raw)
    if known:
        return known
    family = family_of(raw)
    if family is None:
        return None
    for _f, prefixes in FAMILY_PREFIXES:
        for prefix in prefixes:
            if raw.startswith(prefix):
                return family, raw[len(prefix):].lower()
    return None


def simple_name(raw: str) -> Optional[str]:
    """The ``family-short`` name config uses for a raw Prometheus series.

    A catalogued series keeps its curated short name, so ``DCGM_FI_PROF_SM_ACTIVE``
    stays ``dcgm-sm_act`` rather than becoming ``dcgm-sm_active`` -- the name that
    is already in people's config files and in ``[thresholds]``. Anything else is
    derived mechanically by stripping the family prefix and lowercasing, which is
    reversible enough that a site can match it back to the series by eye.

    None for a series in no known family: naming it would imply jobscope knows how
    to join it to a job, and it does not.
    """
    parts = name_parts(raw)
    return "%s-%s" % parts if parts else None


# --- rendering -------------------------------------------------------------

def _line(out, label: str, text: str) -> None:
    print("%-11s %s" % (label, text), file=out)


def _cont(out, text: str) -> None:
    print("%-11s %s" % ("", text), file=out)


# --- slurm -----------------------------------------------------------------

def _scontrol_config(timeout: Optional[float]) -> Dict[str, str]:
    """``scontrol show config`` as a dict, or ``{}`` when it cannot be run."""
    try:
        out = run_capture(["scontrol", "show", "config"], timeout, "scontrol", soft=True)
    except JobscopeError:
        return {}
    if not out:
        return {}
    found = {}
    for line in out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            found[key.strip()] = value.strip()
    return found


def check_slurm(out, timeout: Optional[float]) -> Dict[str, bool]:
    """Report Slurm availability and which accounting sources it populates.

    Returns the capability flags the ``slurm-*`` metrics depend on, so the caller
    can tell the difference between "not collected here" and "jobscope cannot read
    it" -- a distinction a site administrator can act on and a user cannot.
    """
    caps = {"sacct": False, "squeue": False, "acct_cpu": False,
            "gpuutil": False, "gpumem": False, "energy": False}

    for name in ("sacct", "squeue"):
        try:
            answered = run_capture([name, "--version"], timeout, name, soft=True)
        except JobscopeError:      # not on PATH at all
            answered = None
        caps[name] = bool(answered)
    present = [n for n in ("sacct", "squeue") if caps[n]]
    missing = [n for n in ("sacct", "squeue") if not caps[n]]
    _line(out, "slurm", ", ".join("%s %s" % (n, OK) for n in present) or "none found")
    if missing:
        _cont(out, "%s: %s not on PATH -- %s views unavailable"
              % (FAIL, ", ".join(missing),
                 "running" if "squeue" in missing else "finished"))

    conf = _scontrol_config(timeout)
    if not conf:
        _cont(out, "%s scontrol show config unavailable; accounting capabilities unknown" % WARN)
        return caps

    gather = conf.get("JobAcctGatherType", "")
    caps["acct_cpu"] = bool(gather) and not gather.endswith("/none")
    _cont(out, "%s -> TotalCPU, MaxRSS %s"
          % (gather or "JobAcctGatherType unset", OK if caps["acct_cpu"] else ABSENT))

    tres = conf.get("AccountingStorageTRES", "")
    caps["gpuutil"] = "gres/gpuutil" in tres
    caps["gpumem"] = "gres/gpumem" in tres
    gpu_names = [n for n, on in (("slurm-gpuutil", caps["gpuutil"]),
                                 ("slurm-gpumem", caps["gpumem"])) if on]
    _cont(out, "gres/gpuutil, gres/gpumem %s"
          % ("accounted -> " + ", ".join(gpu_names) if gpu_names
             else "%s -- Slurm has no GPU utilization to cross-check against" % ABSENT))

    energy = conf.get("AcctGatherEnergyType", "")
    caps["energy"] = bool(energy) and energy not in ("(null)", "acct_gather_energy/none")
    _cont(out, "AcctGatherEnergyType %s -> slurm-energy %s"
          % (energy or "unset", OK if caps["energy"] else ABSENT))
    return caps


def sample_jobs(timeout: Optional[float]) -> Optional[List[Tuple[str, str, str]]]:
    """A recent sample of finished jobs as ``(raw jobid, AllocTRES, AdminComment)``.

    One sacct call answers every job-shaped question probe has -- whether blobs
    are being written, and which job to probe the metric catalog with -- so it runs
    once and is passed around rather than each check paying for its own scan.
    ``None`` when sacct could not be run at all, which the caller reports
    differently from an empty cluster.
    """
    for window in SAMPLE_WINDOWS:
        try:
            found = run_capture(
                ["sacct", "-X", "-a", "-S", window, "-E", "now", "-s", "COMPLETED",
                 "--noheader", "-P", "-o", "JobIDRaw,AllocTRES,AdminComment"],
                timeout, "sacct", soft=True)
        except JobscopeError:
            return None
        if found is None:
            return None
        rows = [tuple(ln.split("|", 2)) for ln in found.splitlines() if ln.strip()]
        rows = [r for r in rows if len(r) == 3]
        if rows:
            return rows
    return []


def check_blob(out, sample: Optional[List[Tuple[str, str, str]]]) -> bool:
    """Whether sacct is carrying jobstats ``JS1:`` blobs -- the optional fast path.

    Absent is not a failure: it costs the offline CPU view and the second oracle,
    and everything else comes from Prometheus regardless. Said plainly here because
    a site that could turn jobstats on may want to know it is missing out.
    """
    if sample is None:
        _line(out, "blob", "%s could not query sacct for AdminComment" % WARN)
        return False
    if not sample:
        _line(out, "blob", "%s no finished jobs in the sample window to check" % WARN)
        return False
    blobs = sum(1 for _jid, _tres, comment in sample if comment.startswith("JS1:"))
    if blobs:
        _line(out, "blob", "JS1: on %d of %d recent jobs -> offline --cpu view and a "
                           "jobstats cross-check are available" % (blobs, len(sample)))
        return True
    _line(out, "blob", "%s no JS1: blobs in AdminComment -- every metric comes from "
                       "Prometheus (no offline view)" % ABSENT)
    return False


# --- config ----------------------------------------------------------------

def check_config(out, path: Optional[str]) -> None:
    resolved = path or os.environ.get(config.CONFIG_ENV) or config.default_config_path()
    if not os.path.exists(str(resolved)):
        _line(out, "config", "%s (not present; built-in defaults in use)" % resolved)
        return
    try:
        cfg = config.load_config(str(resolved))
    except JobscopeError as exc:
        _line(out, "config", "%s %s: %s" % (FAIL, resolved, exc))
        return
    sections = []
    if cfg.thresholds.by_metric or cfg.timeslice_thresholds.by_metric:
        sections.append("[thresholds]")
    if cfg.source_path:
        sections.append("[prometheus]" if cfg.prometheus_url else "")
    _line(out, "config", "%s  %s" % (resolved, " ".join(s for s in sections if s) or "(defaults)"))


# --- prometheus ------------------------------------------------------------

def _flavor(url: str, timeout: Optional[float]) -> str:
    """Which TSDB is behind the endpoint, from ``/status/buildinfo``.

    Worth naming because it predicts which other endpoints exist: Mimir and Thanos
    answer the query API but not ``/status/tsdb``, which is where retention would
    otherwise come from.
    """
    import requests
    try:
        resp = requests.get(url + "/api/v1/status/buildinfo", timeout=timeout or 30)
        data = (resp.json() or {}).get("data") or {}
    except Exception:
        return ""
    return str(data.get("application") or "Prometheus")


def probe_retention(client, timeout: Optional[float]) -> Optional[int]:
    """Deepest age in days at which a core series still answers, or None.

    Walks a fixed ladder deepest-first and stops at the first hit, rather than
    bisecting. A bisect assumes the data is contiguous and it is not: this site
    answers at 60, 90 and 180 days but not at 45, because a single instant query
    lands in whatever gap a scrape outage left. A ladder is robust to that, and the
    answer is honest about being a lower bound rather than an edge.
    """
    now = int(time.time())
    for days in RETENTION_LADDER_DAYS:
        try:
            if client.query("count(cgroup_cpus)", now - days * 86400, timeout):
                return days
        except Exception:
            continue
    return None


def check_prometheus(out, cfg, timeout: Optional[float]):
    """Report endpoint reachability, flavour, scrape period and retention depth.

    Returns the client for the caller to reuse, or None when there is no usable
    endpoint -- in which case the message is config.py's own, which already lists
    the three ways to fix it.
    """
    from .prometheus import client_from_config
    try:
        client = client_from_config(cfg, timeout)
    except JobscopeError as exc:
        _line(out, "prometheus", "%s %s" % (FAIL, str(exc).splitlines()[0]))
        for line in str(exc).splitlines()[1:]:
            _cont(out, line)
        return None

    # Redacted: the configured URL commonly embeds a Grafana Cloud token in its
    # netloc, so it is a secret that happens to look like an address. The host and
    # path survive, which is what someone chasing a wrong endpoint actually needs.
    shown = config.redact_url(client.url)
    try:
        client.query("up", int(time.time()), timeout)
    except Exception as exc:
        _line(out, "prometheus", "%s %s unreachable: %s" % (FAIL, shown, str(exc)[:80]))
        return None

    flavor = _flavor(client.url, timeout)
    _line(out, "prometheus", "%s  reachable%s" % (shown, ", " + flavor if flavor else ""))
    retention = probe_retention(client, timeout)
    _cont(out, "scrape %ds; data found as far back as %s"
          % (client.sampling_period,
             "%dd" % retention if retention else "%s nothing older than 1d" % WARN))
    if retention:
        _cont(out, "jobs older than that have no Prometheus metrics -- "
                   "they report no-data, not 0%")
    return client


# Label names worth trying when the configured one answers nothing. Ordered by how
# likely each is to be the right answer: `instance` is what a stock Prometheus calls
# the scrape target, where jobstats' own exporter says `host`.
_LABEL_CANDIDATES = {
    "host": ("instance", "host", "node", "nodename"),
    "jobid": ("jobid", "slurm_job_id", "job_id", "job"),
}


def _label_works(client, label: str, metric: str, at, timeout) -> bool:
    """Whether ``metric`` actually carries a non-empty ``label`` on this server."""
    try:
        found = client.query("count by (%s) (%s)" % (label, metric), at, timeout)
    except Exception:
        return False
    return bool([s for s in found if s.get("metric", {}).get(label)])


def detect_labels(client, timeout: Optional[float]) -> Dict[str, Optional[str]]:
    """``{setting: the value that works}`` for the three joins, or None where none does.

    The detection was already here, inside check_labels, and thrown away: it worked out
    that a server uses ``instance`` and then said so in prose. `--init` needs the
    answer, so the search returns it and the printer renders it.

    Keyed by ``[site]`` field name. A configured value that works is kept even when a
    candidate would also work -- a site that set something deliberately should not have
    it second-guessed.
    """
    site = config.get_config().site
    now = int(time.time())
    found: Dict[str, Optional[str]] = {}
    for field, configured, kind in (("host_label", site.host_label, "host"),
                                    ("jobid_label", site.jobid_label, "jobid")):
        if _label_works(client, configured, "cgroup_cpus", now, timeout):
            found[field] = configured
            continue
        found[field] = next(
            (c for c in _LABEL_CANDIDATES[kind]
             if c != configured and _label_works(client, c, "cgroup_cpus", now, timeout)),
            None)
    # The GPU join is a metric name, not a label, so there is nothing to substitute:
    # either the series exists here or a port needs a human to name its equivalent.
    found["gpu_job_join"] = (site.gpu_job_join
                             if _label_works(client, "uuid", site.gpu_job_join, now, timeout)
                             else None)
    return found


def check_labels(out, client, timeout: Optional[float]) -> Dict[str, Optional[str]]:
    """Whether the join labels the collectors assume are the ones in use here.

    Checks the labels ``[site]`` actually configures, not a hardcoded set -- so a
    site that has overridden one gets told whether the override is *right*, which is
    the only version of this check worth running.

    Worth running because every one of these fails silently. The collectors read the
    host label off every series and split a ``:port`` from it; a stock Prometheus
    calls that ``instance``, and reading the wrong one leaves every node as ``?``,
    misses the cgroup divisor lookup, and returns blank CPU%/MEM% with no error at
    all. Where the configured name answers nothing, the working alternative is named.

    Returns :func:`detect_labels`' findings for ``--init`` to serialise.
    """
    site = config.get_config().site
    working = detect_labels(client, timeout)
    checks = (("host_label", site.host_label, "node names on cgroup series"),
              ("jobid_label", site.jobid_label, "job join for cgroup series"),
              ("gpu_job_join", "uuid", "GPU join for NVML series"))
    seen: List[str] = []
    for field, shown, _what in checks:
        if working.get(field) == (site.gpu_job_join if field == "gpu_job_join"
                                  else getattr(site, field)):
            seen.append("%s %s" % (shown, OK))
            continue
        alt = working.get(field)
        seen.append("%s %s%s" % (shown, ABSENT,
                                 " -- this server uses %r; set [site] %s"
                                 % (alt, field) if alt else ""))
    _line(out, "labels", ";  ".join(seen))
    _cont(out, "(%s)" % ", ".join(what for _f, _s, what in checks))
    _report_sources(out)
    return working


def _report_sources(out) -> None:
    """Which source serves which GPU column, and the one that is not optional.

    Worth stating next to the labels because the GPU join is the reason: the job-to-
    card mapping is a *value* on an nvidia-exporter series, and nothing in the DCGM
    catalog carries a job label. So a site can prefer dcgm for every number and still
    needs the nvidia exporter running -- which is not obvious from a config that says
    ``source = "dcgm"``, and is exactly the kind of thing a port discovers the hard way.
    """
    from . import config as config_module
    from . import dcgm
    resolution = dcgm.RESOLVED
    _line(out, "sources", "preference %s" % ", ".join(resolution.preference))
    for name, columns in resolution.by_source():
        shown = list(columns)[:6]
        _cont(out, "%-5s serves %s%s" % (name, " ".join(shown),
                                         " ..." if len(columns) > len(shown) else ""))
    _cont(out, "the %s join is required whatever the preference: it is the only"
               " job-to-card mapping" % config_module.gpu_join())


# --- detecting the numbers a config file needs -------------------------------

def detect_scrape(client, timeout: Optional[float]) -> Optional[int]:
    """The server's real scrape interval, from raw sample spacing, or None.

    **Not** via ``query_range``: that aligns its result to the ``step`` you pass, so
    the gaps come back as whatever you asked for. Measured on this cluster, the same
    server "reports" 15s for step=15 and 60s for step=60 -- a detector that agrees
    with any guess is not one.

    A range *selector* in an instant query returns the raw samples untouched, so the
    gaps are the server's own. The mode rather than the mean, because a restarted
    exporter leaves one long gap that would drag an average up.
    """
    now = int(time.time())
    for metric in ("nvidia_gpu_duty_cycle", "cgroup_cpus", "up"):
        try:
            found = client.query("%s[10m]" % metric, now, timeout)
        except Exception:
            continue
        gaps: Counter = Counter()
        for series in found[:50]:
            stamps = [int(v[0]) for v in series.get("values", [])]
            gaps.update(b - a for a, b in zip(stamps, stamps[1:]))
        if gaps:
            return gaps.most_common(1)[0][0]
    return None


# How the floor sits between the two measured populations, and how far above a lone
# idle figure to put it when nothing was busy. 15% clears the jitter on an idle board
# without reaching the bottom of the busy range on any model measured here.
_IDLE_MARGIN = 1.15


def measure_power_floors(client, timeout: Optional[float],
                         window: str = "1h") -> Dict[str, Tuple[Optional[int], str]]:
    """``{model: (floor_watts_or_None, why)}`` -- the per-model idle power floor.

    Runs the method docs/config.md documents: split every card of a model on whether
    its SMs were doing anything, and put the floor between the idle 90th percentile
    and the busy 10th. Two queries for the whole fleet.

    **Only about half the models measure cleanly at any one time**, so this reports
    why rather than guessing. Measured here, 12 models: 6 clean, 4 with no busy
    samples at all, 2 whose populations overlapped. A wrong floor is worse than no
    floor -- it silently caps healthy jobs at `inefficient` -- so an unclean model
    returns None and the reason, for the caller to emit as a comment.

    Over a window rather than an instant for the same reason: an instant snapshot of
    this fleet left three models unmeasurable and inverted a fourth that the window
    resolved.
    """
    now = int(time.time())

    def quantile(q: float, busy: str) -> Dict[str, float]:
        expr = ("quantile by (modelName) (%s, avg_over_time("
                "(DCGM_FI_DEV_POWER_USAGE and on(UUID) DCGM_FI_PROF_SM_ACTIVE %s)[%s:5m]))"
                % (q, busy, window))
        try:
            found = client.query(expr, now, timeout)
        except Exception:
            return {}
        out = {}
        for series in found:
            model = series.get("metric", {}).get("modelName", "")
            try:
                if model:
                    out[model] = float(series["value"][1])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        return out

    idle, busy = quantile(0.9, "== 0"), quantile(0.1, "> 0")
    floors: Dict[str, Tuple[Optional[int], str]] = {}
    for model in sorted(set(idle) | set(busy)):
        lo, hi = idle.get(model), busy.get(model)
        if lo is None:
            floors[model] = (None, "no idle samples in the last %s" % window)
        elif hi is None:
            floors[model] = (int(round(lo * _IDLE_MARGIN / 10.0) * 10),
                             "idle p90 %.0f W; no busy samples, so %+d%% above idle"
                             % (lo, round((_IDLE_MARGIN - 1) * 100)))
        elif hi <= lo:
            floors[model] = (None, "idle p90 %.0f W and busy p10 %.0f W overlap; "
                                   "measure again over a longer window" % (lo, hi))
        else:
            floors[model] = (int(round((lo + hi) / 2 / 10.0) * 10),
                             "idle p90 %.0f W, busy p10 %.0f W" % (lo, hi))
    return floors


# --- metric discovery ------------------------------------------------------

def _recent_gpu_job(sample: Optional[List[Tuple[str, str, str]]]) -> Optional[str]:
    """A recently finished GPU job from the sample, when the caller named none."""
    for jobid, tres, _comment in sample or ():
        if "gres/gpu=" in tres and jobid.isdigit():
            return jobid
    return None


def _names_for(client, selector: str, at, timeout) -> Dict[str, int]:
    """``{series name: series count}`` for everything matching ``selector``."""
    found = client.query("count by (__name__) (%s)" % selector, at, timeout)
    return {s["metric"].get("__name__", "?"): int(float(s["value"][1])) for s in found}


def probe_series(client, jobid: Optional[str], timeout: Optional[float],
                 sample=None):
    """``(record, [(family, {series: count})])`` for one job, or None.

    Shared by the listing and the TOML emitter so both describe the same server.
    Keyed on a real job rather than on the whole server because presence in isolation
    is not the question: a metric that exists cluster-wide but carries nothing for a
    GPU job is no use, and one absent on this hardware (DFMA% on an A100, exported on
    H100) should say so rather than look healthy and then render blank forever.
    """
    from .dcgm import discover_gpus
    from .slurm import fetch

    jobid = jobid or _recent_gpu_job(sample)
    if not jobid:
        return None
    records = fetch([jobid], timeout)
    record = records.get(jobid) or next(iter(records.values()), None)
    if record is None:
        raise JobscopeError("no such job: %s" % jobid)

    site = config.get_config().site
    selectors = [("cgroup", "{%s=\"%s\"}" % (site.jobid_label, record.jobid_raw))]
    gpus = discover_gpus(record, client, timeout)
    if gpus:
        uuids = "|".join(g["uuid"] for g in gpus)
        selectors.append(("nvml", '{uuid=~"%s"}' % uuids))
        selectors.append(("dcgm", '{UUID=~"%s"}' % uuids))

    families = []
    for family, selector in selectors:
        try:
            families.append((family, _names_for(client, selector, record.end, timeout)))
        except Exception:
            families.append((family, {}))
    return record, families


def discover_metrics(out, client, jobid: Optional[str], timeout: Optional[float],
                     sample: Optional[List[Tuple[str, str, str]]] = None) -> int:
    """Print every series this server carries for one job, by family, with config names.

    Keyed on a real job rather than on the whole server because presence in
    isolation is not the question -- a metric that exists cluster-wide but carries
    nothing for a GPU job is no use, and one absent on this hardware (DFMA% on an
    A100, exported on H100) should say ``absent`` rather than appear healthy and
    then render blank forever.
    """
    probed = probe_series(client, jobid, timeout, sample)
    if probed is None:
        print("\nno recently finished GPU job to probe with; name one:"
              "\n  jobscope probe --metrics JOBID", file=out)
        return 1
    record, families = probed

    print(file=out)
    print("metrics carried for job %s (%s, %d GPU(s), ran %s)"
          % (record.jobid, record.state, record.gpus, record.runtime), file=out)
    if record.gpus and len(families) == 1:
        print("  (no GPUs discovered -- CPU-only job, or no samples in its window)",
              file=out)

    catalogued_by_family: Dict[str, List[str]] = {}
    for raw, (family, _short) in catalog().items():
        catalogued_by_family.setdefault(family, []).append(raw)

    for family, found in families:
        known = catalogued_by_family.get(family, [])
        missing = [raw for raw in known if raw not in found]
        print("\n%s -- %d catalogued, %d present, %d not in jobscope's catalog"
              % (family, len(known), len(known) - len(missing),
                 sum(1 for raw in found if raw not in catalog())), file=out)
        # The name column carries the name **config actually takes**, and nothing
        # else. An uncatalogued series gets NEW rather than a mechanically derived
        # name, because a derived name looks exactly like a real one and is not:
        # putting it in [thresholds] today would be rejected as unknown. What it
        # needs is a [metrics] entry, which is what the footer explains.
        rows = ([(catalog_name(raw) or NEW, raw, "") for raw in sorted(found)]
                + [(catalog_name(raw) or NEW, raw, "%s here" % ABSENT)
                   for raw in sorted(missing)])
        # Widths from the content: names run from `nvml-gpu` to
        # `dcgm-uncorrectable_remapped_rows`, so a fixed column either wastes half
        # the line or lets the long ones collide with the series beside them.
        name_w = max((len(name) for name, _r, _s in rows), default=1)
        raw_w = max((len(raw) for _n, raw, _s in rows), default=1)
        for name, raw, status in rows:
            print("  %-*s  %-*s  %s" % (name_w, name, raw_w, raw, status), file=out)

    _print_config_guide(out)
    return 0


def _print_config_guide(out) -> None:
    """How to turn the listing above into config -- the reason for printing it.

    Spelled out because the mapping is not guessable: the left column is what
    ``[metrics]``, ``[thresholds]`` and ``[classify]`` accept, and a ``new`` row
    needs a definition before any of them will take it.
    """
    print("""
Mapping this to ~/.config/jobscope/config.toml
----------------------------------------------
The left column is a metric's jobscope name. Config takes the part after the
family prefix -- `dcgm-sm_act` is written `sm_act` -- because the names are
unique across families.

  [metrics]                            # which metrics each view collects/shows
  summary    = ["gpu", "sm_act", "tensor", "power"]
  timeseries = ["gpu", "sm_act", "power"]     # --ts / --plot_ts / --classify
  extended   = "all"                          # --dcgm

  [thresholds.summary.wasteful]        # per-metric band edges, per view
  default = 2                          # every metric not named below
  sm_act  = 3                          # this one alone
  [thresholds.timeslice.wasteful]      # --ts's own edges; inherits nothing
  default = 2

  [classify]                           # which metrics decide a verdict
  vote = ["gpu", "sm_act", "cpu"]      # best-of-N; omit to use every percentage
  [classify.floor.power]               # can only *lower* a verdict
  default = 100                        # watts; below this, cap at inefficient

Either spelling works -- `sm_act` or `dcgm-sm_act` -- since the names are unique
across families. Run `jobscope config` to see how every value actually resolved,
which is the quickest way to confirm an edit landed.

A row marked "%s" is a series your server exports that jobscope has no name for.
Give it one with a [metrics.<family>.<name>] table -- the family says which label
carries the GPU UUID (dcgm = UUID, nvml = uuid), and the name becomes the config
name:

  [metrics.dcgm.gpu_util]              # -> the name "gpu_util"
  query    = "DCGM_FI_DEV_GPU_UTIL"    # the series, exactly as listed above
  header   = "GPU_UTIL%%"               # column heading (default: the name, upper)
  decimals = 0
  reducer  = "avg"                     # avg (default) | max | delta
  scale    = 1                         # multiplier on the raw value
  agg      = "mean"                    # across a job's GPUs: mean | sum | max

  [metrics.cgroup.swap]                # -> "swap"
  query  = "cgroup_memsw_used_bytes"
  header = "SWAP%%"
  denom  = "total_memory"              # cgroup values divide by an allocation
  kind   = "gauge"                     # gauge | rate (a counter)

A defined metric joins the **extended** catalog, so it shows under --dcgm, or in
any view that names it: extended = ["sm_act", "gpu_util"]. It never joins the
default view on its own -- defining one cannot silently widen every report, or
the queries every sweep pays for.

The same table with a *built-in's* name **overrides** it, which is how a cluster
whose exporter uses different series names ports without a patch. An override
changes only what it names -- header, tier, roles and the rest are inherited, so
pointing cpu at another series does not rename the CPU%% column or take it out of
the classifier:

  [metrics.cgroup.cpu]
  query = "container_cpu_usage_seconds_total"

A row marked "%s" is in jobscope's catalog but your server does not carry it --
usually hardware, e.g. DFMA%% exists on H100 and not on A100. Nothing to do; the
column stays blank.""" % (NEW, ABSENT), file=out)


# --- entry point -----------------------------------------------------------

def run(out, cfg, config_path: Optional[str], timeout: Optional[float],
        metrics: bool = False, validate: bool = False, toml: bool = False,
        jobid: Optional[str] = None, init: bool = False, full: bool = False) -> int:
    """Print the report. Returns a process exit status."""
    # With --toml, stdout has to be a config file and nothing else: the documented
    # move is `jobscope probe --toml >> config.toml`, and a diagnosis section
    # appended ahead of it is prose where TOML belongs. The checks still run, and
    # still print -- to stderr, where a redirect leaves them visible.
    # With --toml or --init, stdout is a config file and nothing else.
    notes = sys.stderr if (toml or init) else out
    check_slurm(notes, timeout)
    sample = sample_jobs(timeout)
    check_blob(notes, sample)
    check_config(notes, config_path)
    client = check_prometheus(notes, cfg, timeout)
    if client is None:
        print("\nThe Slurm sections above are unaffected; only the metric views need "
              "an endpoint.", file=notes)
        return 1
    check_labels(notes, client, timeout)
    if init:
        return write_config(out, notes, client, cfg, config_path, timeout, jobid,
                            sample, full)
    if toml:
        return emit_toml(out, client, jobid, timeout, sample)
    if validate:
        target = jobid or _recent_gpu_job(sample)
        if not target:
            print("\nno recently finished GPU job to compare; name one:"
                  "\n  jobscope probe --validate JOBID", file=out)
            return 1
        return extra_metric.validate(out, target, client, timeout)
    if metrics:
        return discover_metrics(out, client, jobid, timeout, sample)
    print("\nNext:\n"
          "  jobscope probe --init       write a config for this site from the above\n"
          "  jobscope probe --metrics    what this server carries, and its config names\n"
          "  jobscope probe --toml       the same as an editable [metrics] block\n"
          "  jobscope probe --validate   compare Prometheus against Slurm's accounting",
          file=out)
    return 0


def write_config(out, notes, client, cfg, config_path: Optional[str],
                 timeout: Optional[float], jobid: Optional[str], sample,
                 full: bool) -> int:
    """Write the generated config to the default path, or print it if one is there.

    Writing is the whole point -- "one command to set up a site" is not one command if
    it ends in a redirect -- but never *over* a file. A config is hand-tuned within a
    week of being written, and a tool that silently replaces it has destroyed work
    that has no other copy. So an existing file turns this into stdout plus a note,
    which is the behaviour ``--toml`` has always had.
    """
    import io as _io

    buffer = _io.StringIO()
    emit_config(buffer, client, cfg, timeout, jobid, sample, full)
    text = buffer.getvalue()

    # The same precedence load_config reads: the -c argument, then $JOBSCOPE_CONFIG,
    # then the default path. Writing somewhere other than where jobscope will look for
    # it is the one outcome that would make this command actively misleading.
    path = (config_path or os.environ.get(config.CONFIG_ENV)
            or str(config.default_config_path()))
    if os.path.exists(path):
        print("\nnote: %s already exists, so this is stdout rather than a write.\n"
              "      Compare it, or redirect if you mean to replace it." % path,
              file=notes)
        out.write(text)
        return 0
    directory = os.path.dirname(path)
    try:
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
    except OSError as exc:
        print("\ncannot write %s: %s -- here it is instead." % (path, exc), file=notes)
        out.write(text)
        return 1
    print("\nwrote %s (%d lines)\n\n"
          "  next: jobscope config                 # what it resolves to\n"
          "        jobscope -j <a recent job>      # try it\n"
          "        jobscope probe --init --full    # the same, plus every knob commented"
          % (path, len(text.splitlines())), file=notes)
    return 0


# --- the generated site config ------------------------------------------------

def _present_series(client, jobid: Optional[str], timeout: Optional[float],
                    sample=None) -> Optional[set]:
    """Every series name this server carries for one real job, or None if unprobed."""
    probed = probe_series(client, jobid, timeout, sample)
    if probed is None:
        return None
    _record, families = probed
    return {name for _family, names in families for name in names}


def _view_metrics(view: str, present: Optional[set]) -> Tuple[List[str], List[str]]:
    """``(kept, dropped)`` config names for ``view``, against what the server carries.

    Narrowing matters for a site without one of the exporters: a metric jobscope asks
    for and the server does not have renders as a blank column on every report,
    forever, with nothing saying why. Naming the dropped ones in a comment is what
    makes that recoverable when the exporter arrives later.
    """
    specs = getattr(config.get_config().metrics, view)
    if present is None:
        return [_config_name(s) for s in specs], []
    kept = [_config_name(s) for s in specs if s.metric in present]
    dropped = [_config_name(s) for s in specs if s.metric not in present]
    return kept, dropped


def _config_name(spec) -> str:
    """The name a generated config should call ``spec`` by, never an ambiguous one.

    A spec's key is the short name to write, except where the *other* catalog claims
    it too: the DCGM key for GMEM_GB is ``mem``, which is also the cgroup key for MEM%.
    A generated file has to load, and ``[metrics]`` rejects that name rather than
    guessing -- so fall back to the header form, which is unique catalog-wide.
    """
    from . import cpu
    if cpu.spec_named(spec.key) is not None:
        return spec.header.lower()
    return spec.key


def _toml_str(value: str) -> str:
    return '"%s"' % value.replace("\\", "\\\\").replace('"', '\\"')


def emit_config(out, client, cfg, timeout: Optional[float],
                jobid: Optional[str] = None, sample=None, full: bool = False) -> None:
    """Write a minimal config for this site, from what was actually detected.

    Only detected values. Thresholds are absent on purpose: a band edge is a policy
    choice about what counts as waste, not a property of the cluster, and a generated
    file that quietly set them would be the tool deciding site policy.

    No ``url`` line either. The endpoint comes from jobstats auto-discovery or
    ``$JOBSCOPE_PROM_URL``, and writing it here would put a Grafana Cloud token in a
    file -- which is what ``[prometheus] url``'s own comment warns against.
    """
    cluster = ""
    try:
        cluster = _scontrol_config(timeout).get("ClusterName", "")
    except Exception:
        pass
    print("# jobscope site configuration, generated by 'jobscope probe --init'%s.\n"
          "#\n"
          "# Detected values only. Thresholds are deliberately absent: a band edge is a\n"
          "# policy choice about what counts as waste, not a property of this cluster, so\n"
          "# jobscope's built-ins apply until you set them. 'jobscope config --example'\n"
          "# prints every knob with its reasoning.\n"
          "#\n"
          "# Re-run to see fresh values; this file is never rewritten in place."
          % (" on %s" % cluster if cluster else ""), file=out)

    scrape = detect_scrape(client, timeout)
    print("\n[prometheus]", file=out)
    if scrape:
        print("sampling_period = %d       # measured from raw sample spacing" % scrape,
              file=out)
    else:
        print("# sampling_period       # could not measure it; %d is the built-in default"
              % client.sampling_period, file=out)
    print("# No url: it comes from $JOBSCOPE_PROM_URL or the jobstats config.py, and\n"
          "# writing it here would put a credential in a file. 'jobscope config' shows\n"
          "# which of those answered.", file=out)

    labels = detect_labels(client, timeout)
    print("\n[site]", file=out)
    for field, what in (("host_label", "node names on cgroup series"),
                        ("jobid_label", "job join for cgroup series"),
                        ("gpu_job_join", "the only job-to-GPU join there is")):
        value = labels.get(field)
        if value:
            print("%-12s = %-22s # confirmed: %s" % (field, _toml_str(value), what),
                  file=out)
        else:
            print("# %-10s   -- nothing answered; %s is missing here. jobscope cannot\n"
                  "#                guess this one: name the equivalent series yourself."
                  % (field, what), file=out)

    present = _present_series(client, jobid, timeout, sample)
    print("\n[metrics]", file=out)
    if present is None:
        print("# Not narrowed: no recently finished GPU job to probe with. Re-run as\n"
              "#   jobscope probe --init JOBID\n"
              "# to drop metrics this server does not carry.", file=out)
    for view in ("summary", "timeseries"):
        kept, dropped = _view_metrics(view, present)
        print("%-11s = [%s]" % (view, ", ".join(_toml_str(k) for k in kept)), file=out)
        if dropped:
            print("#             dropped, not carried by this server: %s"
                  % ", ".join(dropped), file=out)

    floors = measure_power_floors(client, timeout)
    print("\n[classify.floor.power]", file=out)
    print("default = %g              # watts; below this a GPU counts as idle"
          % config.DEFAULT_POWER_W, file=out)
    if not floors:
        print("# No per-model figures: DCGM power or SM-activity series not found.",
              file=out)
    for model, (value, why) in sorted(floors.items()):
        if value is None:
            print("# %-46s -- %s" % (_toml_str(model), why), file=out)
        else:
            print("%-48s = %-5d # %s" % (_toml_str(model), value, why), file=out)

    if full:
        print("\n\n# " + "#" * 74, file=out)
        print("#\n#   Below: every remaining knob, commented, from\n"
              "#   'jobscope config --example'. Uncomment what you want to tune.\n#",
              file=out)
        print("# " + "#" * 74, file=out)
        # Only the template's tuning half, which is entirely comments -- appending its
        # live [prometheus]/[metrics] blocks would redeclare tables already written
        # above, and a table declared twice is a TOML parse error rather than an
        # override.
        text = config.example_config_text()
        marker = "Everything above is enough to run"
        tail = text.split(marker, 1)
        if len(tail) == 2:
            print("\n" + tail[1].split("\n", 1)[1].rstrip(), file=out)


# --- the editable name table ------------------------------------------------

# Series jobscope reads for something other than a measurement, so a metric table
# for them would be noise: the job-to-GPU join, the two cgroup denominators every
# percentage is divided by, and the identity labels. Listed rather than filtered out
# silently -- a reader looking for `cgroup_cpus` should find out where it went.
_STRUCTURAL = {
    "cgroup_cpus": "denominator for the CPU percentages",
    "cgroup_memory_total_bytes": "denominator for the memory percentages",
    "cgroup_uid": "identity, not a measurement",
    "nvidia_gpu_jobId": "the job-to-GPU join; see [site] gpu_job_join",
    "nvidia_gpu_jobUid": "identity, not a measurement",
}


def _cgroup_fields(raw: str) -> Optional[Tuple[str, str]]:
    """``(kind, denom)`` for a cgroup series, or None if jobscope cannot model it.

    Inferred from the suffix, which is reliable for the two shapes that fit: a
    ``_seconds`` counter is a rate over allocated cores, a ``_bytes`` gauge is a level
    over allocated memory.

    None for anything else, and that matters more than a guess would. A cgroup metric
    is *divided* by an allocation field, so there is no such thing as leaving the
    denominator out -- and there is no allocation to divide an OOM-kill count by.
    ``cgroup_memory_fail_count`` as a percentage of total bytes is a number with no
    meaning, and emitting one would be worse than emitting nothing.
    """
    if raw.endswith("_seconds"):
        return "rate", "cpus"
    if raw.endswith("_bytes"):
        return "gauge", "total_memory"
    return None


def _gpu_scale(raw: str) -> Tuple[float, str]:
    """``(scale, note)`` guessed from a GPU series' name.

    DCGM's PROF metrics are 0-1 fractions and want 100; its DEV utilisations are
    already percentages. Anything else gets 1 and a note, because a wrong scale is
    the kind of error that reads as a plausible number.
    """
    if "_PROF_" in raw and raw.endswith(("_ACTIVE", "_OCCUPANCY")):
        return 100, ""
    if raw.endswith("_UTIL"):
        return 1, ""
    return 1, "  # CHECK: raw units -- set scale if this is a fraction or bytes"


def emit_toml(out, client, jobid: Optional[str], timeout: Optional[float],
              sample=None) -> int:
    """Print the discovered metrics as an editable ``[metrics]`` block.

    Two states per series, and the difference is the point:

    * **Catalogued** -- emitted commented out. It already works; it is here so the
      name is *visible*, and so renaming it is an edit rather than a guess.
    * **Uncatalogued** -- emitted live. Uncommenting is not required; the block is
      already active, so redirecting this into a config file adds every metric the
      server has.

    Renaming is editing the table key. That is not a special feature -- the key *is*
    the config name, which is what [metrics.<family>.<name>] made true.

    Prints to stdout and writes nothing. Appending to a file someone has hand-edited
    is not a thing to do without their eyes on it, and a redirect is one character.
    """
    probed = probe_series(client, jobid, timeout, sample)
    if probed is None:
        print("# no recently finished GPU job to probe with; name one:\n"
              "#   jobscope probe --toml JOBID", file=out)
        return 1
    record, families = probed

    print("# jobscope metric definitions, generated by 'jobscope probe --toml'.\n"
          "#\n"
          "# Probed against job %s (%s, %d GPU(s)) on this cluster's Prometheus.\n"
          "# Left column of each table path is the family, which fixes how the series\n"
          "# joins to a job. The table KEY is the name config uses -- rename it freely.\n"
          "#\n"
          "# Blocks that are commented out are jobscope built-ins: they already work,\n"
          "# and are shown so their names are visible and editable. Blocks that are\n"
          "# live are series this server has that jobscope does not name yet -- append\n"
          "# this file to your config and they take effect.\n"
          "#\n"
          "#   jobscope probe --toml >> ~/.config/jobscope/config.toml\n"
          "#\n"
          "# A defined metric joins the *extended* catalog: it shows under --dcgm, or in\n"
          "# any view that names it. It never joins the default view on its own."
          % (record.jobid, record.state, record.gpus), file=out)

    for family, found in families:
        if not found:
            continue
        print("\n\n# %s %s" % ("-" * 8, family), file=out)
        print("# %s" % _FAMILY_NOTE.get(family, ""), file=out)
        structural = [raw for raw in sorted(found) if raw in _STRUCTURAL]
        for raw in sorted(found):
            if raw in _STRUCTURAL:
                continue
            _emit_block(out, family, raw)
        if structural:
            print("\n# jobscope reads these for something other than a measurement,"
                  "\n# so they are not metrics you can select:", file=out)
            for raw in structural:
                print("#   %-30s %s" % (raw, _STRUCTURAL[raw]), file=out)

    _emit_other_sources(out)
    return 0


_FAMILY_NOTE = {
    "cgroup": "per-job host metrics, joined on the jobid label. Values are divided by\n"
              "# an allocation field, which `denom` names.",
    "nvml": "GPU metrics from the nvidia exporter, joined on the lowercase `uuid`\n"
            "# label via the job-to-GPU series.",
    "dcgm": "GPU metrics from dcgm-exporter, joined on the uppercase `UUID` label.",
}


def _emit_block(out, family: str, raw: str) -> None:
    """One ``[metrics.<family>.<name>]`` table, commented iff already catalogued."""
    parts = name_parts(raw)
    if parts is None:
        return
    _f, short = parts
    builtin = raw in catalog()
    hide = "# " if builtin else ""
    lines = ['[metrics.%s.%s]' % (family, short), 'query  = "%s"' % raw]
    tail, note = ("        # built-in" if builtin else "        # new here"), ""
    if family == "cgroup":
        shape = _cgroup_fields(raw)
        if shape is None:
            # Not modellable: every cgroup metric is divided by an allocation, and
            # this one has nothing to divide by. Commented out with the reason rather
            # than emitted with an invented denominator.
            print("\n# %s -- not expressible here. jobscope divides every cgroup metric"
                  "\n#   by an allocation (cpus or total_memory), and a count has none to"
                  "\n#   divide by. Reading it needs code, not config." % raw, file=out)
            return
        kind, denom = shape
        lines.append('header = "%s%%"' % short.upper())
        lines.append('kind   = "%s"' % kind)
        lines.append('denom  = "%s"' % denom)
    else:
        scale, note = _gpu_scale(raw)
        percent = scale == 100 or raw.endswith("_UTIL")
        lines.append('header = "%s%s"' % (short.upper(), "%" if percent else ""))
        lines.append('scale  = %g%s' % (scale, note))
    print("\n" + hide + lines[0] + tail, file=out)
    for line in lines[1:]:
        print(hide + line, file=out)


def _emit_other_sources(out) -> None:
    """The non-Prometheus names, for reference only.

    Reference only because they are derived from sacct fields rather than a PromQL
    query, so a ``query = "..."`` table cannot define one -- printing syntax that
    fails would be worse than printing nothing. Their names are listed because the
    question this command answers is "what is everything called", and these are part
    of the answer.
    """
    print("""

# -------- slurm and jobstats: names only, not definable here
#
# These come from sacct rather than Prometheus -- Slurm's own accounting and the
# jobstats blob -- so a `query = "..."` table cannot describe one. Shown because they
# are part of the mapping, and they are what 'jobscope probe --validate' compares
# against. Adding to this set is a code change, not a config one.
#
#   slurm-cpu        TotalCPU / CPUTime            (what `seff` reports)
#   slurm-cpu_user   UserCPU / CPUTime
#   slurm-cpu_sys    SystemCPU / CPUTime
#   slurm-mem        TRESUsageInTot mem / ReqMem   (summed RSS, not MaxRSS)
#   slurm-gpuutil    TRESUsageInTot gres/gpuutil   (summed across cards; divided
#                                                   by the count)
#   slurm-gpumem     TRESUsageInTot gres/gpumem    (likewise per card)
#   slurm-disk       TRESUsageInTot fs/disk
#   slurm-energy     ConsumedEnergyRaw             (needs AcctGatherEnergyType)
#   jobstats-*       the JS1: AdminComment blob    (absent at non-jobstats sites)""",
          file=out)
