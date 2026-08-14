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

import fnmatch
import os
import re
import sys
import textwrap
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

from . import config
from .cpu import CGROUP_METRICS
from .dcgm import METRICS as GPU_METRICS
from .errors import JobscopeError
from .running import requested_gpus
from .slurm import UNFINISHED_STATES, expand_nodelist, run_capture

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
# every question here ("are summaries being written", "find me a GPU job") is answered
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

    One sacct call answers every job-shaped question probe has -- whether summaries
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


def check_jobstats(out, sample: Optional[List[Tuple[str, str, str]]]) -> bool:
    """Whether sacct is carrying jobstats ``JS1:`` summaries -- the optional fast path.

    Absent is not a failure: it costs the offline CPU view and the second oracle,
    and everything else comes from Prometheus regardless. Said plainly here because
    a site that could turn jobstats on may want to know it is missing out.
    """
    if sample is None:
        _line(out, "jobstats", "%s could not query sacct for AdminComment" % WARN)
        return False
    if not sample:
        _line(out, "jobstats", "%s no finished jobs in the sample window to check" % WARN)
        return False
    summaries = sum(1 for _jid, _tres, comment in sample if comment.startswith("JS1:"))
    if summaries:
        _line(out, "jobstats", "JS1: on %d of %d recent jobs -> offline --cpu view and a "
                           "jobstats cross-check are available" % (summaries, len(sample)))
        return True
    _line(out, "jobstats", "%s no JS1: summaries in AdminComment -- every metric comes from "
                       "Prometheus (no offline view)" % ABSENT)
    return False


# --- config ----------------------------------------------------------------

def check_config(out, path: Optional[str]) -> None:
    resolved = config.resolve_config_path(path)
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
    _report_coverage(out, client, timeout)
    _report_sources(out)
    return working


# One representative series per family, and the column that goes blank when it is not
# there. Chosen as the series the *default* view queries, so thin coverage here means
# thin coverage of the report rather than of some corner of the catalog.
_COVERAGE: Tuple[Tuple[str, str, str], ...] = (
    ("cgroup", "cgroup_cpu_total_seconds", "running CPU%/MEM%"),
    ("nvml", "nvidia_gpu_duty_cycle", "the GPU join, and so every GPU column"),
    ("dcgm", "DCGM_FI_PROF_SM_ACTIVE", "SM_ACT%/TENSOR%/DRAM%"),
)


def host_coverage(client, timeout: Optional[float]) -> Dict[str, Optional[int]]:
    """``{family: hosts reporting it}``, or None per family the query failed for."""
    host_label = config.get_config().site.host_label
    now = int(time.time())
    found: Dict[str, Optional[int]] = {}
    for family, series, _what in _COVERAGE:
        try:
            rows = client.query("count(count by (%s) (%s))" % (host_label, series),
                                now, timeout)
        except Exception:
            found[family] = None      # asked and could not tell, which is not zero
            continue
        found[family] = int(float(rows[0]["value"][1])) if rows else 0
    return found


def _report_coverage(out, client, timeout: Optional[float]) -> None:
    """How many hosts each exporter reports on, and what thin coverage costs.

    Presence is not coverage, and the labels check above cannot tell the difference:
    it asks whether a label answers *anywhere*, so two stale hosts out of four hundred
    read as ``jobid ok``. That is how a cluster whose cgroup exporter is deployed on
    two nodes looks healthy here while every running job shows CPU% as ``-``.

    So the count, next to the widest family as a yardstick -- a bare "2 hosts" means
    nothing without knowing the fleet is 417.
    """
    counts = host_coverage(client, timeout)
    widest = max((n for n in counts.values() if n), default=0)
    _line(out, "coverage", ";  ".join(
        "%s on %s host(s)" % (family, "?" if counts.get(family) is None
                              else counts[family])
        for family, _series, _what in _COVERAGE))
    for family, _series, what in _COVERAGE:
        count = counts.get(family) or 0
        # A tenth of the widest exporter is the line: below that the family is not
        # "deployed with gaps", it is absent with a couple of leftovers reporting.
        if widest and count * 10 < widest:
            _cont(out, "%s is thin (%d of %d): %s will read \"-\"%s"
                       % (family, count, widest, what,
                          " -- the JS1: summary covers finished jobs"
                          if family == "cgroup" else ""))




def _expand_partitions(spec: str, timeout: Optional[float]) -> List[str]:
    """Resolve a comma-separated spec, expanding any shell-style wildcard.

    ``sinfo -p`` takes a comma list natively but no globs, so ``kempner*`` reaches it
    as a literal name and matches nothing. Expanded here against ``sinfo -o %R``, which
    is also the list the error message can offer -- a typo and an unsupported pattern
    are indistinguishable to the caller otherwise.
    """
    wanted = [t.strip() for t in spec.split(",") if t.strip()]
    if not any(ch in t for t in wanted for ch in "*?["):
        return wanted
    listed = run_capture(["sinfo", "-h", "-o", "%R"], timeout, "sinfo", soft=True)
    known = sorted({line.strip() for line in (listed or "").splitlines() if line.strip()})
    out: List[str] = []
    for token in wanted:
        if not any(ch in token for ch in "*?["):
            out.append(token)
            continue
        hits = fnmatch.filter(known, token)
        if not hits:
            near = [k for k in known if k.startswith(token.split("*")[0][:4])]
            raise JobscopeError(
                "no partition matches %r.%s" % (token,
                    " Did you mean: %s?" % ", ".join(near[:8]) if near
                    else " 'sinfo -o %R' lists them."))
        out.extend(hits)
    # Deduplicated, order preserved: two patterns may overlap, and sinfo would then
    # count the same nodes twice.
    seen, unique = set(), []
    for name in out:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _partition_nodes(partition: str, timeout: Optional[float]):
    """``{node: (slurm state, has_gpu, is_mig)}`` for ``partition``.

    ``%n`` rather than ``%N``, so Slurm expands its own bracketed hostlist -- the
    comparison downstream is against one label value at a time, and
    ``holygpu8a[19102,19302]`` matches nothing.

    The gres column is read because a node with no GPU correctly publishes no GPU
    series. Without it a CPU-only partition reports every GPU column missing on every
    node, which is a page of noise saying only "these are CPU nodes".
    """
    out = run_capture(["sinfo", "-h", "-p", partition, "-o", "%n %T %G"],
                      timeout, "sinfo", soft=True)
    nodes = {}
    for line in (out or "").splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2:
            gres = parts[2] if len(parts) > 2 else ""
            nodes[parts[0]] = (parts[1], "gpu:" in gres, bool(_MIG_GRES.search(gres)))
    if not nodes:
        raise JobscopeError(
            "no nodes in partition %r. 'sinfo -o %%R' lists the partitions." % partition)
    return nodes


# A MIG profile in Slurm's gres string: `gpu:nvidia_a100_3g.20gb:8`. Partitioning a card
# means there is no longer a whole *device* to report a duty cycle for, so neither
# exporter publishes one -- measured, DCGM_FI_DEV_GPU_UTIL and nvidia_gpu_duty_cycle are
# both absent on every MIG host here while the profiling and memory series are complete.
_MIG_GRES = re.compile(r"\d+g\.\d+gb")

# Columns MIG structurally cannot serve. GPU% is a whole-device duty cycle; the profiling
# ratios and the memory pair are per instance and come through fine. A future whole-device
# column would need adding here -- there is no role in the catalog that says "device-wide",
# so this cannot be derived.
_MIG_BLIND_COLUMNS = ("GPU%",)


# States where the node is not serving at all, so absent metrics are the expected answer
# rather than a gap. The trailing `*$~#` Slurm appends (unresponsive, maint, power-saving)
# is stripped first. `inval` is a node whose registration Slurm rejected -- up in name
# only.
_DOWN_STATES = ("down", "drain", "drng", "fail", "maint", "unk", "boot", "pow",
                "inval", "future", "perfctrs")


# States in which a job is actually running on the node. Only these can have cgroup
# series: those exist per running *job*, not per node, so `idle`, `reserved` and
# `planned` legitimately have none. Reserved nodes are otherwise up -- their GPU
# exporters publish normally -- so they are counted, just not faulted for cgroup.
_BUSY_STATES = ("alloc", "mix", "comp")


def _strip_state(state: str) -> str:
    return state.rstrip("*$~#+").lower()


def _is_up(state: str) -> bool:
    return not _strip_state(state).startswith(_DOWN_STATES)


def _runs_jobs(state: str) -> bool:
    return _strip_state(state).startswith(_BUSY_STATES)


def _series_hosts(client, series: str, timeout: Optional[float]):
    """The set of hosts publishing ``series``, or None when the query failed.

    None is not an empty set: "asked and could not tell" must not be reported as
    "nothing covers this", the same distinction Measure draws for a reading.
    """
    label = config.get_config().site.host_label
    try:
        rows = client.query("count by (%s) (%s)" % (label, series),
                            int(time.time()), timeout)
    except Exception:
        return None
    return {str(r["metric"].get(label, "?")).split(":")[0] for r in rows}


# The order sources are reported in: host axis first, then the two GPU exporters, which
# is the order the columns themselves print in.
_SOURCE_ORDER = ("cgroup", "nvml", "dcgm")

_SOURCE_WHAT = {
    "cgroup": "CPU%/MEM%, published per running job",
    "nvml": "duty cycle, memory, and the job-to-GPU join every source depends on",
    "dcgm": "the profiling catalog",
}


def join_owners(client, timeout: Optional[float]):
    """``(owners, rows seen)`` from the GPU join series, or None if it could not be read.

    ``owners`` is ``{host: {job id each of its cards claims}}``. An instant query, because
    the question is what the mapping says *now*.

    The row count is returned alongside because an empty ``owners`` has two causes worth
    telling apart: the series is absent, or it is present and no row carried a usable job
    ID. The second is a broken exporter rather than a missing one, and "no series at all"
    would be the wrong thing to tell someone about to go looking for it.
    """
    try:
        rows = client.query(config.gpu_join(), int(time.time()), timeout)
    except Exception:
        return None
    owners: Dict[str, set] = {}
    for row in rows:
        try:
            # Scientific notation is possible (3.4853925e+07), hence float first --
            # the same coercion jobscope.running.discover_gpus does.
            jobid = int(float(row["value"][1]))
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        owners.setdefault(config.short_host(config.host_of(row["metric"])),
                          set()).add(jobid)
    return owners, len(rows)


def running_gpu_hosts(timeout: Optional[float]):
    """``({host: {raw job id}}, {every running job id})``, or None if squeue failed.

    ``%A`` is the raw per-element job ID, which is what the join series reports -- ``%i``
    would miss every array element. ``%b`` identifies a GPU request: a CPU-only job
    correctly claims no card, and counting it as an absence would be noise.

    The second set is *not* filtered to GPU jobs, because it answers a different
    question. Classifying a claim needs to know whether the job it names is running
    anywhere at all -- a card naming a live CPU-only job is a misplaced claim, and
    calling it a finished one would point at the wrong exporter bug.
    """
    out = run_capture(["squeue", "-h", "-t", "RUNNING", "-o", "%A|%N|%b"],
                      timeout, "squeue", soft=True)
    if out is None:
        return None
    hosts: Dict[str, set] = {}
    live: set = set()
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3 or not parts[0].strip().isdigit():
            continue
        jobid = int(parts[0].strip())
        live.add(jobid)
        if not requested_gpus({"gres": parts[2]}):
            continue
        for host in expand_nodelist(parts[1].strip(), timeout):
            # Normalised on both sides through the one definition, or an FQDN-labelled
            # site gets a clean report from the very check the report path points at.
            hosts.setdefault(config.short_host(host), set()).add(jobid)
    return hosts, live


def job_ended(jobid: int, timeout: Optional[float]) -> str:
    """When ``jobid`` ended, as sacct reports it, or "" if that cannot be established.

    Turns the frozen-mapping finding from an opaque job ID into a time, which is the form
    it has to be in for anyone to act on it: "the mapping stopped taking new jobs around
    2026-08-04T15:43" is a ticket, "newest claimed is 37239323" is a puzzle.

    Empty for a job still running -- then its end is not when anything stopped -- and for
    one sacct has forgotten.
    """
    out = run_capture(["sacct", "-X", "-n", "-P", "-j", str(jobid), "-o", "State,End"],
                      timeout, "sacct", soft=True)
    for line in (out or "").splitlines():
        state, _, end = line.partition("|")
        if state.strip().upper().startswith(UNFINISHED_STATES):
            return ""
        if end.strip() and end.strip() != "Unknown":
            return end.strip()
    return ""


def _classify_claims(owners: Dict[str, set], by_host: Dict[str, set],
                     live: set) -> Tuple[int, int, int]:
    """``(correct, finished, misplaced)`` counted per claim, not per host.

    A different denominator from the host summary and a finer question. The host count
    says how much of the report is affected; this says which exporter bug to chase --
    cards reporting a job that has *ended* and cards reporting a job that is running
    somewhere *else* are different failures, and the host-level view conflates them.
    """
    correct = finished = misplaced = 0
    for host, claimed in owners.items():
        for jobid in claimed:
            if jobid in by_host.get(host, ()):
                correct += 1
            elif jobid in live:
                misplaced += 1
            else:
                finished += 1
    return correct, finished, misplaced


def report_label_mapping(out, client, timeout: Optional[float]) -> None:
    """Whether a second, label-based job-to-card mapping exists here.

    Worth a line whichever way it goes, because it decides how bad a frozen join is. With
    one, a job the join missed still resolves; without one, the join is a single point of
    failure for every GPU column -- which is what it turned out to be on this cluster.

    Says so even when unconfigured, and checks whether the label is published anyway: a
    site running dcgm-exporter's HPC job mapping without telling jobscope about it has a
    fallback available for the cost of one config key, and would otherwise never find out.
    """
    label = config.gpu_job_label()
    series = config.gpu_job_label_series()
    if label:
        _cont(out, "  %-34s %-14s configured ([site] gpu_job_label)"
                   % ("%s{%s}" % (series[:20], label), "2nd job join"))
        return
    # Unconfigured. dcgm-exporter's own name for it, which is what a site is most likely
    # to be publishing without having said so.
    try:
        # count(), not the series themselves: this needs one integer, and SM_ACTIVE is
        # ~2000 series each carrying a full DCGM label set -- a megabyte or two of JSON
        # parsed to print a number. The same trick _series_hosts already uses.
        rows = client.query('count(%s{hpc_job!=""})' % series, int(time.time()), timeout)
        published = int(float(rows[0]["value"][1])) if rows else 0
    except Exception:
        return
    if published:
        _cont(out, "%s %d series carry an 'hpc_job' label that jobscope is not using --"
                   % (WARN, published))
        _cont(out, "    set [site] gpu_job_label = \"hpc_job\" for a second job-to-card "
                   "mapping,")
        _cont(out, "    which answers for jobs the %s join misses" % config.gpu_join())
    else:
        _cont(out, "  %-34s %-14s none -- %s is the single point of failure"
                   % ("(no second mapping)", "2nd job join", config.gpu_join()))


def check_gpu_join(out, client, timeout: Optional[float], scope=None,
                   full: bool = False) -> None:
    """Whether the job-to-card mapping is present *and* current.

    The gap this closes. :func:`_coverage_series` enumerates metric specs, and the join
    is not one -- it is ``[site] gpu_job_join`` -- so it never appeared in the coverage
    table at all, while the ``nvml`` block's own heading promised "the job-to-GPU join
    every source depends on". Every column could read 89/89 with no job resolving.

    Presence alone would not have caught it either. Measured on this cluster: the series
    was on every host and *stale* -- no card claimed a job newer than 37239323 while jobs
    to 37366799 were running, 76% named a job that had already finished, and 55% of hosts
    running a GPU job named none of their own. Every GPU job started after the mapping
    froze reported blank GPU columns, and nothing said why.

    So the check is a comparison, not a count: do the job IDs the cards claim match the
    jobs Slurm says are on that host. A host with no running GPU job is not a fault --
    idle cards may legitimately still name their last occupant -- so only hosts that
    *are* running one are judged.
    """
    sample = join_owners(client, timeout)
    join = config.gpu_join()
    if sample is None:
        _cont(out, "%s %-34s could not be queried -- no job resolves to a card without it"
                   % (FAIL, join[:33]))
        return
    owners, rows = sample
    if scope:
        owners = {h: js for h, js in owners.items() if h in scope}
    if not owners:
        _cont(out, "%s %-34s %s -- every GPU column will be blank"
                   % (FAIL, join[:33],
                      "%d series, none carrying a job id" % rows if rows
                      else "no series at all"))
        return

    # Counted the way the rows above are, so the join reads as one of them rather than
    # as a footnote: in scope / could publish it.
    count = "%d/%d" % (len(owners), len(scope)) if scope else str(len(owners))
    _cont(out, "  %-34s %-14s %s" % (join[:33], "job->card join", count))
    # Immediately after, because the two are one question: how many mappings are there,
    # and is the report resting on a single one.
    report_label_mapping(out, client, timeout)

    sampled = running_gpu_hosts(timeout)
    if sampled is None:
        _cont(out, "%s squeue unavailable, so the mapping's *values* were not checked "
                   "-- presence is not freshness" % WARN)
        return
    by_host, live = sampled

    judged = {h: js for h, js in by_host.items() if h in owners and js}
    if not judged:
        _cont(out, "  no GPU job running on these hosts, so the mapping has nothing to "
                   "be checked against")
        return
    blind = sorted(h for h, js in judged.items() if not (js & owners[h]))
    partial = sorted(h for h, js in judged.items()
                     if (js & owners[h]) and (js - owners[h]))
    newest_claimed = max((j for js in owners.values() for j in js), default=0)
    newest_running = max((j for js in judged.values() for j in js), default=0)

    if not blind and not partial:
        _cont(out, "  every host running a GPU job is named by one of its own cards")
        if not full:
            # Nothing is broken *for this report*, but the mapping can still be carrying
            # stale claims on idle cards -- which --full will show. Said only here,
            # because below the FAIL line it would compete with the actual fault.
            _cont(out, "  --full splits the claims by kind, including any stale ones on "
                       "idle cards")
            return
    else:
        _cont(out, "%s %d of %d host(s) running a GPU job: their cards name none of it"
                   % (FAIL if blind else WARN, len(blind), len(judged)))
        if partial:
            _cont(out, "  %d more name some of their jobs but not all" % len(partial))
        if newest_claimed and newest_running > newest_claimed:
            # The decisive shape: a mapping that stopped ingesting has a hard ceiling,
            # where merely-idle cards would leave a spread. Said with both numbers because
            # the gap is the evidence -- a reader should not have to take a verdict on
            # trust -- and dated, because an id is a puzzle where a time is a ticket.
            ended = job_ended(newest_claimed, timeout)
            _cont(out, "  newest job id any card claims is %d%s,"
                       % (newest_claimed, ", which ended %s" % ended if ended else ""))
            _cont(out, "    but %d is running -- the mapping %s"
                       % (newest_running,
                          "stopped taking new jobs then" if ended else "looks frozen"))
        _cont(out, "  jobs newer than that report blank GPU columns; jobscope has no "
                   "other job-to-card mapping")
        if not full:
            _cont(out, "  --full lists the affected hosts and splits the claims by kind")
            return
    correct, finished, misplaced = _classify_claims(owners, by_host, live)
    total = correct + finished + misplaced
    if total:
        _cont(out, "  %d claim(s): %d%% correct, %d%% name a finished job, %d%% name a "
                   "job on another host"
                   % (total, round(100 * correct / total), round(100 * finished / total),
                      round(100 * misplaced / total)))
        # Named rather than left for the reader to weigh, because the two are different
        # exporter bugs: a mapping that never clears keeps naming jobs that have ended,
        # while mis-attribution puts a *live* job on the wrong host -- and that half is
        # the one that produced wrong numbers rather than missing ones.
        if finished or misplaced:
            _cont(out, "  mostly %s"
                       % ("a mapping that never clears -- cards still name jobs that "
                          "ended" if finished >= misplaced else
                          "mis-attribution -- cards name live jobs that run elsewhere"))
    for host in blind + partial:
        stale = sorted(owners[host] - by_host.get(host, set()))
        _cont(out, "    %-22s runs %s;  cards claim %s"
                   % (host, " ".join(str(j) for j in sorted(judged[host])),
                      " ".join(str(j) for j in stale) or "(nothing)"))


def _coverage_series():
    """``{family: [(series, column, serving)]}`` over every *candidate*, not just winners.

    Candidates rather than the resolved view, because comparing sources is the decision
    this report exists to inform: with only the winner shown, ``nvidia_gpu_duty_cycle``
    is invisible whenever dcgm wins GPU%, so "would --gpu-source nvml cover more of my
    partition?" has no answer here. ``serving`` marks the one actually in use.

    ``slurm`` candidates are skipped: they come from sacct, so they have no series and no
    host coverage to report. Same for the stored summary -- noted in prose instead.
    """
    from . import cpu, dcgm
    host, gpu = cpu.catalog(), dcgm.catalog()
    columns = {spec.column for spec in gpu.default_specs}
    by_family: Dict[str, List[Tuple[str, str, bool]]] = {}

    def winner(resolution, column: str) -> str:
        """The family of the *exporter* resolved to ``column``.

        Not ``source_of``, which answers ``jobstats`` for the columns the stored summary
        wins -- and jobstats has no series, so that would leave every row here unmarked
        and the mark meaningless. The resolved spec list holds exporters only, which is
        exactly the question host coverage can answer.
        """
        for spec in resolution.specs:
            if spec.column == column:
                return spec.family
        return ""

    for spec in host.candidates:
        if spec.column in ("CPU%", "MEM%") and spec.metric:
            by_family.setdefault(spec.family, []).append(
                (spec.metric, spec.column,
                 winner(host.resolved, spec.column) == spec.family))
    for spec in gpu.metrics:
        if spec.column in columns and spec.metric:
            by_family.setdefault(spec.family, []).append(
                (spec.metric, spec.column,
                 winner(gpu.resolved, spec.column) == spec.family))
    return by_family


def _column_order(columns) -> List[str]:
    """``columns`` in the order the table prints them, not alphabetically."""
    from . import cpu, dcgm
    order = ([s.column for s in cpu.catalog().resolved.specs]
             + [s.column for s in dcgm.catalog().all_specs])
    rank = {c: i for i, c in enumerate(order)}
    return sorted(columns, key=lambda c: rank.get(c, len(rank)))


def report_column_coverage(out, client, timeout: Optional[float],
                           partition: str = "", full: bool = False) -> int:
    """Per-series host coverage for every source, and by name whatever is missing.

    The coverage line in the main report gives one number per exporter, which hides gaps
    *within* a family: it picks one representative series, so a family whose members have
    different coverage reads as uniformly fine. Measured here, dcgm-exporter publishes
    ``DCGM_FI_PROF_SM_ACTIVE`` on 437 hosts and ``DCGM_FI_DEV_GPU_UTIL`` on 416, because
    MIG nodes have no whole-device duty cycle. One number per family hid a 22-host hole
    in the first column anyone reads.

    Grouped by source and covering every candidate, so all three are answerable side by
    side -- which is the decision this informs. ``*`` marks the series currently serving
    its column; the others are what a different ``--gpu-source`` would read.

    Node state is read alongside, because without it this cries wolf. A ``down`` node
    reports nothing by definition, and a node with no job running has no cgroup series to
    publish -- neither is a fault, and on five partitions here they outnumbered the one
    genuinely misconfigured host eight to one.
    """
    from . import dcgm
    by_family = _coverage_series()
    names = _expand_partitions(partition, timeout) if partition else []
    nodes = _partition_nodes(",".join(names), timeout) if names else {}
    states = {n: st for n, (st, _g, _m) in nodes.items()}
    up = {n for n, (st, _g, _m) in nodes.items() if _is_up(st)}
    gpu_up = {n for n in up if nodes[n][1]}
    mig = {n for n in up if nodes[n][2]}
    skipped = len(nodes) - len(up)

    if partition:
        _line(out, "coverage", "%d partition(s): %s" % (len(names), ", ".join(names)))
        _cont(out, "%d of %d node(s) up%s"
                   % (len(up), len(nodes),
                      " (%d down/drained, not counted)" % skipped if skipped else ""))
    else:
        _line(out, "coverage", "cluster-wide (name a partition to see which nodes are "
                               "missing)")

    absent_by_node: Dict[str, List[str]] = {}
    families: Dict[str, str] = {}
    for family in _SOURCE_ORDER:
        rows = by_family.get(family)
        if not rows:
            continue
        print("", file=out)
        _line(out, family, "-- %s" % _SOURCE_WHAT.get(family, ""))
        for series, column, serving in rows:
            hosts = _series_hosts(client, series, timeout)
            mark = "*" if serving else " "
            if hosts is None:
                _cont(out, "%s %-34s %-14s query failed" % (mark, series[:33], column))
                continue
            if partition:
                # Scoped to the nodes that could publish it: a GPU series over the GPU
                # nodes, so a count is not diluted by CPU-only partition members.
                scope = up if family == "cgroup" else gpu_up
                count = ("%d/%d" % (len(hosts & scope), len(scope)) if scope
                         else "-  (no GPU nodes)")
                # Only the serving series can break a report, so only it makes a node a
                # fault. The rest are here to be compared, not to raise alarms.
                if serving:
                    for node in sorted(scope - hosts):
                        absent_by_node.setdefault(node, []).append(column)
                        families[column] = family
            else:
                count = str(len(hosts))
            _cont(out, "%s %-34s %-14s %s" % (mark, series[:33], column, count))
        if family == "nvml":
            # Here rather than after the table because this block's heading is what
            # promises the join, and because the join is not a metric spec -- it is
            # [site] gpu_job_join, so _coverage_series() cannot produce a row for it.
            check_gpu_join(out, client, timeout, gpu_up if partition else None,
                           full=full)

    print("", file=out)
    _cont(out, "* serving that column now. Unmarked rows are what another --gpu-source "
               "would read.")
    stored = sorted(dcgm.catalog().resolved.from_jobstats)
    if stored:
        _cont(out, "jobstats also serves %s for a *finished* job -- stored per job in "
                   "sacct," % ", ".join(stored))
        _cont(out, "  so it has no host coverage; the rows above are what a running job "
                   "falls back to.")
    if not partition:
        _cont(out, "  jobscope probe --coverage PARTITION")
        return 0

    if not absent_by_node:
        _cont(out, "every serving series covers every node that is up")
        return 0

    # Classified per *gap*, not per node: an idle MIG node has two absences with two
    # different explanations, and judging the node as a whole put it in the fault list
    # for both. So each (node, column) is asked separately whether it is expected, and a
    # node is a fault only if something unexplained is left.
    faults: Dict[str, List[str]] = {}
    # {reason key: (explanation, {columns}, {nodes})}. Keyed so two nodes absent for the
    # same reason share one entry, and the explanation travels with it -- "no job running
    # (idle)" alone did not say *which* columns were affected or why that follows.
    expected: Dict[str, Tuple[str, set, set]] = {}

    def note_expected(key: str, explanation: str, column: str, node: str) -> None:
        _text, cols, hosts = expected.setdefault(key, (explanation, set(), set()))
        cols.add(column)
        hosts.add(node)

    for node in sorted(absent_by_node):
        state = states.get(node, "?")
        for column in absent_by_node[node]:
            family = families[column]
            if family == "cgroup" and not _runs_jobs(state):
                note_expected(
                    "nojob",
                    "no job is running, and cgroup series exist per running job rather "
                    "than per node", column, node)
            elif node in mig and column in _MIG_BLIND_COLUMNS:
                note_expected(
                    "mig",
                    "MIG partitions the card, so there is no whole device for either "
                    "exporter to report a duty cycle on", column, node)
            else:
                faults.setdefault(node, []).append(column)

    print("", file=out)
    if faults:
        _line(out, "missing", "%d node(s) with an unexplained gap:" % len(faults))
        for node in sorted(faults):
            cols = faults[node]
            fams = sorted({families[c] for c in cols})
            _cont(out, "%-16s %-11s no %s: %s"
                       % (node, "(%s)" % states.get(node, "?"), "/".join(fams),
                          ", ".join(cols)))
    else:
        _line(out, "missing", "nothing unexplained")

    for i, key in enumerate(sorted(expected)):
        explanation, cols, hosts = expected[key]
        if i == 0:
            print("", file=out)      # the two verdicts are different claims
        label = "expected" if i == 0 else ""
        header = ("%d node(s) have no %s: %s."
                  % (len(hosts), "/".join(_column_order(cols)), explanation))
        for j, line in enumerate(textwrap.wrap(header, 92)):
            (_line(out, label, line) if j == 0 and label else _cont(out, line))
        # Every node named, wrapped rather than truncated: the list is what you act on,
        # and an elided one cannot be pasted into scontrol or a ticket.
        for line in textwrap.wrap(", ".join(sorted(hosts)), 88):
            _cont(out, "  " + line)
    return 0


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
    resolution = dcgm.catalog().resolved
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

# How many sampled jobs to try before settling for one whose cards were not found. Each
# try is one sacct call plus one discovery query (~130ms together), and on a healthy
# mapping the first candidate answers -- this bounds the cost where the mapping is stale
# and every recent job fails to resolve.
_PROBE_TRIES = 5


def _gpu_job_candidates(sample: Optional[List[Tuple[str, str, str]]]) -> List[str]:
    """Recently finished GPU jobs to probe with, best first.

    **Single-GPU jobs first.** Every series a family publishes then has exactly one
    instance for the job, so the sampled values read as one reading per metric instead of
    the first of thirty-two -- which is what makes the table answerable at a glance.
    Multi-GPU jobs follow, rather than being excluded, so a site that runs none still gets
    an answer.

    A list rather than one id because whether a job's cards can be *found* is not visible
    here: the join mapping may not name it, and then only the cgroup family has anything
    to show. The caller walks these until one resolves.
    """
    single, multi = [], []
    for jobid, tres, _comment in sample or ():
        if "gres/gpu=" not in tres or not jobid.isdigit():
            continue
        (single if "gres/gpu=1," in tres + "," else multi).append(jobid)
    return single + multi


def _recent_gpu_job(sample: Optional[List[Tuple[str, str, str]]]) -> Optional[str]:
    """The best single candidate, for callers that do not walk the list."""
    candidates = _gpu_job_candidates(sample)
    return candidates[0] if candidates else None


def _names_for(client, selector: str, at, timeout) -> Dict[str, int]:
    """``{series name: series count}`` for everything matching ``selector``."""
    found = client.query("count by (__name__) (%s)" % selector, at, timeout)
    return {s["metric"].get("__name__", "?"): int(float(s["value"][1])) for s in found}


def _samples_for(client, selector: str, at, timeout) -> Dict[str, Tuple[str, int]]:
    """``{series name: (first value, how many series carry that name)}``.

    The same one query per family the name listing makes, keeping the samples instead of
    counting them and throwing them away -- measured at 39-224ms, against a `count by`
    that is cheaper only because it discards the answer.

    Values **raw**, exactly as the server returned them: this exists to show what the
    exporter publishes, so scaling them into the catalog's units would hide the thing
    being looked at (``nvidia_gpu_memory_total_bytes`` reads 85899345920, not 80.0).

    The *first* value, with the count beside it, rather than one line per series. A
    32-GPU job carries ~500 series over ~30 names, and a line each turns a table you can
    read into a page you cannot.
    """
    try:
        found = client.query(selector, at, timeout)
    except Exception:
        return {}
    out: Dict[str, Tuple[str, int]] = {}
    for row in found:
        name = row["metric"].get("__name__", "?")
        try:
            value = str(row["value"][1])
        except (KeyError, IndexError, TypeError):
            continue
        if name in out:
            out[name] = (out[name][0], out[name][1] + 1)
        else:
            out[name] = (value, 1)
    return out


def probe_series(client, jobid: Optional[str], timeout: Optional[float],
                 sample=None, samples: bool = False):
    """``(record, [(family, {series: count})])`` for one job, or None.

    Shared by the listing and the TOML emitter so both describe the same server.
    Keyed on a real job rather than on the whole server because presence in isolation
    is not the question: a metric that exists cluster-wide but carries nothing for a
    GPU job is no use, and one absent on this hardware (DFMA% on an A100, exported on
    H100) should say so rather than look healthy and then render blank forever.
    """
    from .dcgm import discover_gpus
    from .slurm import fetch

    # Named: exactly that job, and its failure to resolve is the answer. Unnamed: walk
    # the candidates, because a job whose cards no mapping names shows only the cgroup
    # family and would look like a server missing its GPU exporters. Bounded, since each
    # try costs one sacct call and one discovery query.
    wanted = [jobid] if jobid else _gpu_job_candidates(sample)[:_PROBE_TRIES]
    if not wanted:
        return None
    record = gpus = None
    for candidate in wanted:
        records = fetch([candidate], timeout)
        found = records.get(candidate) or next(iter(records.values()), None)
        if found is None:
            if jobid:
                raise JobscopeError("no such job: %s" % jobid)
            continue
        record, gpus = found, discover_gpus(found, client, timeout)
        if gpus or jobid:
            break
    if record is None:
        return None

    site = config.get_config().site
    selectors = [("cgroup", "{%s=\"%s\"}" % (site.jobid_label, record.jobid_raw))]
    if gpus:
        uuids = "|".join(g["uuid"] for g in gpus)
        selectors.append(("nvml", '{uuid=~"%s"}' % uuids))
        selectors.append(("dcgm", '{UUID=~"%s"}' % uuids))

    families = []
    read = _samples_for if samples else _names_for
    for family, selector in selectors:
        try:
            found = read(client, selector, record.end, timeout)
        except Exception:
            found = {}
        # Kept to the family that was asked for. The cgroup selector is `{jobid="..."}`,
        # and the nvidia exporter labels its own series with a jobid too -- so it matched
        # every nvml series as well and listed them under the cgroup heading. The selector
        # cannot express "cgroup series" without hard-coding a name prefix into a query, so
        # the classifier that already knows the answer does the filtering.
        families.append((family, {name: value for name, value in found.items()
                                  if family_of(name) == family}))
    return record, families


def _sampled(entry) -> str:
    """The value cell for one series name, or "" when only names were read.

    ``--metrics`` alone asks for counts, ``--metrics --full`` for the samples, so the
    per-family mapping holds either an int or a ``(value, count)`` pair. Formatting from
    the shape rather than threading the flag down here keeps the row builder one branch.
    """
    if not isinstance(entry, tuple):
        return ""
    value, count = entry
    return value if count == 1 else "%s  (first of %d)" % (value, count)


def discover_metrics(out, client, jobid: Optional[str], timeout: Optional[float],
                     sample: Optional[List[Tuple[str, str, str]]] = None,
                     full: bool = False) -> int:
    """Print every series this server carries for one job, by family, with config names.

    Keyed on a real job rather than on the whole server because presence in
    isolation is not the question -- a metric that exists cluster-wide but carries
    nothing for a GPU job is no use, and one absent on this hardware (DFMA% on an
    A100, exported on H100) should say ``absent`` rather than appear healthy and
    then render blank forever.
    """
    probed = probe_series(client, jobid, timeout, sample, samples=full)
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
        rows = ([(catalog_name(raw) or NEW, raw, _sampled(found[raw]))
                 for raw in sorted(found)]
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
    ``[metrics]``, ``[thresholds]`` and ``[eff]`` accept, and a ``new`` row
    needs a definition before any of them will take it.
    """
    target = config.resolve_config_path()
    print("""
Mapping this to %s
----------------------------------------------
The left column is a metric's jobscope name. Config takes the part after the
family prefix -- `dcgm-sm_act` is written `sm_act` -- because the names are
unique across families.

  [metrics]                            # which metrics each view collects/shows
  summary    = ["gpu", "sm_act", "tensor", "power"]
  timeseries = ["gpu", "sm_act", "power"]     # --ts / --plot_ts / --eff
  extended   = "all"                          # --all-metrics

  [thresholds.summary.wasteful]        # per-metric band edges, per view
  default = 2                          # every metric not named below
  sm_act  = 3                          # this one alone
  [thresholds.timeslice.wasteful]      # --ts's own edges; inherits nothing
  default = 2

  [eff]                                # which metrics decide a verdict
  vote = ["gpu", "sm_act", "cpu"]      # best-of-N; omit to use every percentage
  [eff.floor.power]                    # can only *lower* a verdict
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
the efficiency ballot:

  [metrics.cgroup.cpu]
  query = "container_cpu_usage_seconds_total"

A row marked "%s" is in jobscope's catalog but your server does not carry it --
usually hardware, e.g. DFMA%% exists on H100 and not on A100. Nothing to do; the
column stays blank.""" % (target, NEW, ABSENT), file=out)


# --- entry point -----------------------------------------------------------

def run(out, cfg, config_path: Optional[str], timeout: Optional[float],
        metrics: bool = False, toml: bool = False,
        jobid: Optional[str] = None, init: bool = False, full: bool = False,
        coverage: Optional[str] = None) -> int:
    """Print the report. Returns a process exit status."""
    # With --toml, stdout has to be a config file and nothing else: the documented
    # move is `jobscope probe --toml >> config.toml`, and a diagnosis section
    # appended ahead of it is prose where TOML belongs. The checks still run, and
    # still print -- to stderr, where a redirect leaves them visible.
    # With --toml or --init, stdout is a config file and nothing else.
    notes = sys.stderr if (toml or init) else out
    check_slurm(notes, timeout)
    sample = sample_jobs(timeout)
    check_jobstats(notes, sample)
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
    if metrics:
        return discover_metrics(out, client, jobid, timeout, sample, full=full)
    if coverage is not None:
        print("", file=out)
        return report_column_coverage(out, client, timeout, coverage, full=full)
    print("\nNext:\n"
          "  jobscope probe --init       write a config for this site from the above\n"
          "  jobscope probe --metrics    what this server carries, and its config names\n"
          "  jobscope probe --toml       the same as an editable [metrics] block\n"
          "  jobscope probe --coverage   per-column host coverage, and what is missing",
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

    # Somewhere load_config will read: the -c argument, then $JOBSCOPE_CONFIG, then
    # init_target_path. Writing somewhere other than where jobscope will look for it is
    # the one outcome that would make this command actively misleading.
    #
    # Not resolve_config_path: inside a checkout that answers with the tracked
    # jobscope.toml, and generated output should not land in a file several admins share
    # and review. init_target_path picks the git-ignored config.toml above it, which
    # jobscope reads first anyway -- so this is still the file that wins.
    path = config_path or str(config.init_target_path())
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
    print("\n[eff.floor.power]", file=out)
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
          "#   jobscope probe --toml >> %s\n"
          "#\n"
          "# A defined metric joins the *extended* catalog: it shows under --dcgm, or in\n"
          "# any view that names it. It never joins the default view on its own."
          % (record.jobid, record.state, record.gpus, config.resolve_config_path()),
          file=out)

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
# jobstats summary -- so a `query = "..."` table cannot describe one. Shown because they
# are part of the mapping, and they are what [host] source = slurm reads
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
