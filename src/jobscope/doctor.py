"""``jobscope doctor`` -- what this cluster exposes, and whether jobscope can read it.

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

Nothing here writes: ``doctor`` is safe to run anywhere, and the flag that edits a
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
import time
from typing import Dict, List, Optional, Tuple

from . import config, extra_metric
from .cpu import CGROUP_METRICS
from .dcgm import METRICS as GPU_METRICS
from .errors import JobscopeError
from .sacct import run_capture

# Status markers. Deliberately words rather than colour: doctor output gets pasted
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
        family = "nvml" if spec.uuid_label == "uuid" else "dcgm"
        header = spec.header.lower()
        short = header[:-1] if header.endswith("%") else spec.key
        found.setdefault(spec.metric, (family, short))
    return found


CATALOG: Dict[str, Tuple[str, str]] = _catalog()


def family_of(raw: str) -> Optional[str]:
    """The family a raw series name belongs to, or None if jobscope cannot place it."""
    for family, prefixes in FAMILY_PREFIXES:
        if any(raw.startswith(prefix) for prefix in prefixes):
            return family
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
    known = CATALOG.get(raw)
    if known:
        return "%s-%s" % known
    family = family_of(raw)
    if family is None:
        return None
    for _f, prefixes in FAMILY_PREFIXES:
        for prefix in prefixes:
            if raw.startswith(prefix):
                return "%s-%s" % (family, raw[len(prefix):].lower())
    return None


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

    One sacct call answers every job-shaped question doctor has -- whether blobs
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


def check_labels(out, client, timeout: Optional[float]) -> None:
    """Whether the join labels the collectors assume are the ones in use here.

    Checks the labels ``[site]`` actually configures, not a hardcoded set -- so a
    site that has overridden one gets told whether the override is *right*, which is
    the only version of this check worth running.

    Worth running because every one of these fails silently. The collectors read the
    host label off every series and split a ``:port`` from it; a stock Prometheus
    calls that ``instance``, and reading the wrong one leaves every node as ``?``,
    misses the cgroup divisor lookup, and returns blank CPU%/MEM% with no error at
    all. The ``instance`` fallback is suggested by name because that is the single
    most likely correct answer.
    """
    site = config.get_config().site
    checks = ((site.host_label, "cgroup_cpus", "node names on cgroup series"),
              (site.jobid_label, "cgroup_cpus", "job join for cgroup series"),
              ("uuid", site.gpu_job_join, "GPU join for NVML series"))
    now = int(time.time())
    seen: List[str] = []
    for label, metric, _what in checks:
        try:
            found = client.query("count by (%s) (%s)" % (label, metric), now, timeout)
        except Exception:
            found = []
        if [s for s in found if s.get("metric", {}).get(label)]:
            seen.append("%s %s" % (label, OK))
            continue
        alt = ""
        if label == site.host_label:
            for candidate in ("instance", "host", "node", "nodename"):
                if candidate == label:
                    continue
                try:
                    if client.query("count by (%s) (%s)" % (candidate, metric), now, timeout):
                        alt = " -- this server uses %r; set [site] host_label" % candidate
                        break
                except Exception:
                    continue
        seen.append("%s %s%s" % (label, ABSENT, alt))
    _line(out, "labels", ";  ".join(seen))
    _cont(out, "(%s)" % ", ".join(what for _l, _m, what in checks))


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


def discover_metrics(out, client, jobid: Optional[str], timeout: Optional[float],
                     sample: Optional[List[Tuple[str, str, str]]] = None) -> int:
    """Print every series this server carries for one job, by family, with config names.

    Keyed on a real job rather than on the whole server because presence in
    isolation is not the question -- a metric that exists cluster-wide but carries
    nothing for a GPU job is no use, and one absent on this hardware (DFMA% on an
    A100, exported on H100) should say ``absent`` rather than appear healthy and
    then render blank forever.
    """
    from .dcgm import discover_gpus
    from .sacct import fetch

    jobid = jobid or _recent_gpu_job(sample)
    if not jobid:
        print("\nno recently finished GPU job to probe with; name one:"
              "\n  jobscope doctor --metrics JOBID", file=out)
        return 1
    records = fetch([jobid], timeout)
    record = records.get(jobid) or next(iter(records.values()), None)
    if record is None:
        raise JobscopeError("no such job: %s" % jobid)

    print(file=out)
    print("metrics carried for job %s (%s, %d GPU(s), ran %s)"
          % (record.jobid, record.state, record.gpus, record.runtime), file=out)

    groups = [("cgroup", '{jobid="%s"}' % record.jobid_raw)]
    gpus = discover_gpus(record, client, timeout)
    if gpus:
        uuids = "|".join(g["uuid"] for g in gpus)
        groups.append(("nvml", '{uuid=~"%s"}' % uuids))
        groups.append(("dcgm", '{UUID=~"%s"}' % uuids))
    else:
        print("  (no GPUs discovered -- CPU-only job, or no samples in its window)", file=out)

    catalogued_by_family: Dict[str, List[str]] = {}
    for raw, (family, _short) in CATALOG.items():
        catalogued_by_family.setdefault(family, []).append(raw)

    for family, selector in groups:
        try:
            found = _names_for(client, selector, record.end, timeout)
        except Exception as exc:
            print("\n%s -- query failed: %s" % (family, str(exc)[:70]), file=out)
            continue
        known = catalogued_by_family.get(family, [])
        missing = [raw for raw in known if raw not in found]
        print("\n%s -- %d catalogued, %d present, %d not in jobscope's catalog"
              % (family, len(known), len(known) - len(missing),
                 sum(1 for raw in found if raw not in CATALOG)), file=out)
        rows = ([(simple_name(raw), raw, OK if raw in CATALOG else NEW)
                 for raw in sorted(found)]
                + [(simple_name(raw), raw, "%s (not exported here)" % ABSENT)
                   for raw in sorted(missing)])
        # Widths from the content: the derived names run from `nvml-gpu` to
        # `dcgm-uncorrectable_remapped_rows`, so a fixed column either wastes half
        # the line or lets the long ones collide with the series beside them.
        name_w = max((len(name or "-") for name, _r, _s in rows), default=1)
        raw_w = max((len(raw) for _n, raw, _s in rows), default=1)
        for name, raw, status in rows:
            print("  %-*s  %-*s  %s" % (name_w, name or "-", raw_w, raw, status), file=out)

    print("\n%s = in jobscope's catalog and present   %s = present, jobscope has no name "
          "for it yet\n%s = catalogued but this server does not carry it"
          % (OK, NEW, ABSENT), file=out)
    return 0


# --- entry point -----------------------------------------------------------

def run(out, cfg, config_path: Optional[str], timeout: Optional[float],
        metrics: bool = False, validate: bool = False,
        jobid: Optional[str] = None) -> int:
    """Print the report. Returns a process exit status."""
    check_slurm(out, timeout)
    sample = sample_jobs(timeout)
    check_blob(out, sample)
    check_config(out, config_path)
    client = check_prometheus(out, cfg, timeout)
    if client is None:
        print("\nThe Slurm sections above are unaffected; only the metric views need "
              "an endpoint.", file=out)
        return 1
    check_labels(out, client, timeout)
    if validate:
        target = jobid or _recent_gpu_job(sample)
        if not target:
            print("\nno recently finished GPU job to compare; name one:"
                  "\n  jobscope doctor --validate JOBID", file=out)
            return 1
        return extra_metric.validate(out, target, client, timeout)
    if metrics:
        return discover_metrics(out, client, jobid, timeout, sample)
    print("\nRun 'jobscope doctor --metrics' to list the metrics this server carries "
          "for a real job,\nand 'jobscope doctor --validate' to compare Prometheus "
          "against Slurm's own accounting.", file=out)
    return 0
