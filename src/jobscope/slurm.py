"""Slurm as a data source: which jobs exist, and what Slurm itself knows about them.

Named for the scheduler rather than for ``sacct`` because both of Slurm's job
listings belong here -- ``sacct`` for finished jobs and ``squeue``, via
:mod:`jobscope.running`, for the ones running now. Everything downstream asks this
module "which jobs", and asks Prometheus what they did.

Jobs are selected with one ``sacct`` query, then all of their data is retrieved
with bulk ``sacct -j`` queries, no per-job jobstats calls and no job-count
cap. When the id list would exceed the kernel's per-argument size limit the
``-j`` query is split into batches (see :data:`JOBID_ARG_LIMIT`), with progress
notes on stderr. The AdminComment blob returned by the bulk query is decoded
into each :class:`JobRecord`.
"""

import getpass
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Optional, Tuple

from . import config
from .errors import JobscopeError
from .jobstats import decode_admin_comment, gpu_model_from_tres, gpus_from_tres

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

# What each -t name covers. Separated rather than lumped into one "failed" bucket
# because they are different problems: a timeout usually means the walltime or the
# resource request was wrong, a cancellation is a person, and a true failure is the
# job itself. RUNNING and PENDING appear in none of them -- `finished` means finished,
# and `jobscope running` is the running view.
STATE_GROUPS = {
    "completed": ("COMPLETED",),
    "failed": ("FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"),
    "timeout": ("TIMEOUT", "DEADLINE"),
    "cancelled": ("CANCELLED", "PREEMPTED", "REVOKED"),
}
# Every terminal state jobscope knows, which is what -t all selects.
FINISHED_STATES = tuple(state for group in STATE_GROUPS.values() for state in group)
# Not selectable: they are not finished. Named so the error can say where to look.
LIVE_STATES = ("running", "pending", "suspended", "requeued")

DEFAULT_STATE = config.DEFAULT_STATE

# How far back a request that names no window reaches. Resolved to real timestamps by
# select.sacct_selection, so the header can print dates rather than "now-30days".
DEFAULT_LOOKBACK_DAYS = 30
# Last-resort fallbacks, for a Selection built without going through that resolution.
DEFAULT_START = "now-%ddays" % DEFAULT_LOOKBACK_DAYS
DEFAULT_END = "now"


def states_for(spec: Optional[str]) -> Tuple[str, ...]:
    """The sacct states a ``-t`` value selects, e.g. ``failed,timeout``.

    ``all`` (and an unset value) means every finished state. Always an explicit list,
    never "no filter": without one sacct returns jobs that are still running, and a
    running job has no final numbers to report.
    """
    if spec in (None, "", "all"):
        return FINISHED_STATES
    chosen: List[str] = []
    for name in str(spec).split(","):
        name = name.strip().lower()
        if not name:
            continue
        if name in LIVE_STATES:
            raise JobscopeError(
                "-t %s is not a finished state; use 'jobscope running' for live jobs"
                % name)
        if name not in STATE_GROUPS:
            raise JobscopeError(
                "unknown -t value %r: choose from %s, or 'all' (comma-separated)"
                % (name, ", ".join(sorted(STATE_GROUPS))))
        chosen.extend(state for state in STATE_GROUPS[name] if state not in chosen)
    if not chosen:
        raise JobscopeError("-t needs at least one state")
    return tuple(chosen)

# Chunk bounds for the batched sacct -j queries, which is now the *explicit JOBID*
# path only -- a window selection is sliced by time instead, see SLICE_SECONDS.
# JOBS_PER_CHUNK is the primary bound: a sacct call costs 85-145 ms whatever the id
# count (measured; an older note here said ~35 ms, which no longer holds), so small
# batches stream first rows sooner at a cost that is real but bounded by how many ids
# were named. JOBID_ARG_LIMIT backstops the Linux per-argument cap (MAX_ARG_STRLEN,
# 128 KiB) for pathologically long ids. NOTE_EVERY throttles the stderr progress notes
# to one per that many jobs.
JOBID_ARG_LIMIT = 16384
JOBS_PER_CHUNK = 200
NOTE_EVERY = 4096

# The fields one bulk query asks for. Shared by the two bulk paths so a column added
# for one cannot go missing from the other.
#
# **AdminComment stays last.** It is the one field that may itself contain a '|' (it is
# a base64 blob), so the parser splits with a maxsplit that stops before it and takes
# the remainder whole. Move it and every job's stored summary is truncated at its first
# stray character.
FETCH_FIELDS = ("JobID,State,JobName,Elapsed,NNodes,AllocTRES,"
                "Start,End,JobIDRaw,Cluster,User,Account,Partition,AdminComment")

# Derived, never written down twice. The maxsplit and this list disagreeing is the one
# failure here that does not raise: the parser's length guard would `continue` on every
# row, and the report would come back empty for a selection that matched thousands of
# jobs. Counting the fields it was actually asked for makes that unrepresentable.
FETCH_FIELD_COUNT = FETCH_FIELDS.count(",") + 1

# How much of a window one sacct call may cover.
#
# Both sacct's wall clock *and its memory* scale with the rows it returns, and the
# output format changes neither: measured cluster-wide over one day, 193,982 rows cost
# 1.66 GB of RSS inside sacct whether it was asked for twelve fields or for `JobID`
# alone. A week of that is ~1.4 M rows, or ~11 GB, which nothing about the field list
# or the id batching can help -- the id-batched pass never reached those rows, because
# the pass that listed the ids had already materialised every one of them.
#
# So the window is the unit that has to be cut, and a day is the cut because
# _query_lastn already walks back a day at a time for the same reason, and its
# measurement -- "a day took 1.2s and thirty days did not return inside 60" -- is this
# same curve seen from the other end.
SLICE_SECONDS = 86400

# How many jobs a streamed chunk carries, which is a different question from how many
# one sacct call fetches: this one is about how soon the first row appears, not about
# what the server is asked.
#
# Sized for latency, which is the only thing it controls. dcgm.compute_dcgm is a barrier
# -- every job in a chunk finishes before any of its rows is drawn -- so the chunk is
# exactly the wait before the table starts moving, at roughly two queries a job against
# [prometheus] max_queries_per_second.
#
# This was briefly 200, inherited from the id-batched path where a chunk *was* one sacct
# call and the number answered an argv-size question instead. Measured on one user's
# 8983-job day:
#
#   chunk   queries per 1000 jobs   first row
#     200                   2,011        8.0s
#      25                   2,092        4.9s   <-
#   (before any batching:   3,000)
#
# Smaller chunks cost discovery queries, since dcgm.discover_gpus_batch buckets per
# chunk and sacct returns a slice ordered by job id rather than tightly by time -- a
# 25-job chunk still spans a couple of hours here, so it needs more than one bucket.
# Measured, that is 4% more queries for a table that starts moving in half the time,
# and still 30% below what per-job discovery cost.
#
# It does not go lower because the remaining wait is not the chunk: of those 4.9s,
# 1.1s is the sacct slice and ~0.5s is startup, both fixed. Eight, what the running
# path uses (select.RUNNING_JOBS_PER_CHUNK), would buy about a second more and pay
# another few percent for it -- that path can afford eight because a running selection
# is hundreds of jobs rather than thousands.
STREAM_CHUNK = 25

# States meaning the job has not ended. `startswith`, because sacct decorates some
# states with detail ("CANCELLED by 64336"). One definition, because two places ask:
# _query_ids excludes these from a `finished` selection, and JobRecord.unfinished
# decides whether a metric may be folded over the job's window.
UNFINISHED_STATES = ("PENDING", "RUNNING", "SUSPENDED", "REQUEUED")

# When to say the selection is broad enough to be worth narrowing. There is no
# job-count cap and this does not introduce one -- the run proceeds -- but past a few
# thousand jobs the cost stops being invisible: records accumulate for the whole run
# (~1 KB each, measured), so it is worth naming before the batches start rather than
# after someone has waited. Chosen well above any real per-partition day (a busy GPU
# partition here runs ~100 jobs/day) and well below a cluster-wide sweep, which returned
# 315,702 jobs for a single day when this was measured -- so it fires on the selections
# that are broad by accident, not on ordinary ones.
LARGE_SELECTION = 5000


@dataclass
class Selection:
    """Inputs that determine which jobs to report on."""

    user: Optional[str]
    jobids: List[str] = field(default_factory=list)
    account: Optional[str] = None
    partition: Optional[str] = None
    state: str = DEFAULT_STATE
    lastn: Optional[int] = None
    days: Optional[int] = None
    starttime: Optional[str] = None
    endtime: Optional[str] = None
    all_users: bool = False   # sacct -a; `user` is then unset

    def window(self) -> Tuple[str, str]:
        """The ``(start, end)`` actually queried, defaults filled in.

        The header reports this, so it is the same pair the sacct call receives: a
        window shown but not used is worse than none.
        """
        return (self.starttime or DEFAULT_START, self.endtime or DEFAULT_END)


@dataclass
class JobRecord:
    """Everything jobscope needs about one job, from the bulk sacct query."""

    jobid: str
    state: str
    name: str
    runtime: str
    nodes: str
    gpus: int
    stats: dict
    start: Optional[int]
    end: int
    duration: Optional[int]
    jobid_raw: str
    cluster: str
    user: str
    # Identity the table shows only under --show, but collected always: sacct charges
    # for rows, not for columns (ten extra fields cost 0.05s against an 8587-row
    # query's 0.36s), so varying the field list by flag would buy nothing and give the
    # two bulk paths two different record shapes.
    account: str = ""
    partition: str = ""
    # The card the job ran on, as Slurm spells it ("nvidia_h100_80gb_hbm3"). Read out
    # of the AllocTRES this record already parses for ``gpus``, so it costs no field
    # and no query -- which is what keeps GPU_TYPE on screen under --cpu, the view
    # that talks to no exporter at all. Empty for a CPU job, and empty when the job's
    # cards disagree; see :func:`jobscope.jobstats.gpu_model_from_tres`.
    gpu_model: str = ""

    @property
    def unfinished(self) -> bool:
        """Whether the job has not ended, so its ``[start, end]`` window is still growing.

        What decides whether a metric can be *folded* over that window: a mean over a
        window that is still filling is not the same question as a mean over a finished
        one. Asked of the record rather than of the selection mode, so a job reads the
        same whichever way it was selected -- see :func:`jobscope.dcgm.dcgm_for_job`.
        """
        return self.state.upper().startswith(UNFINISHED_STATES)


def default_user() -> Optional[str]:
    """Current user from $USER, falling back to getpass; None if undeterminable."""
    user = os.environ.get("USER")
    if not user:
        try:
            user = getpass.getuser()
        except Exception:
            user = None
    return user


def format_window(start: str, end: str) -> str:
    """``2026-07-30 14:33 .. 2026-07-31 14:33`` for display.

    Seconds are dropped -- nobody selects a window to the second -- and anything that
    is not a timestamp (sacct's relative forms, a bare date) passes through as given
    rather than being guessed at.
    """
    def show(value):
        for fmt in (TIMESTAMP_FORMAT, "%Y-%m-%d"):
            try:
                return time.strftime("%Y-%m-%d %H:%M", time.strptime(value, fmt))
            except (ValueError, TypeError):
                continue
        return str(value)

    return "%s .. %s" % (show(start), show(end))


def days_to_window(days: int) -> Tuple[str, str]:
    """``(start, end)`` sacct timestamps for the last ``days`` days ending now."""
    return day_slice(days, 0)


def day_slice(from_days: int, to_days: int) -> Tuple[str, str]:
    """``(start, end)`` for the span from ``from_days`` ago to ``to_days`` ago.

    One clock reading for both ends, so a slice cannot straddle the instant between
    two calls to ``time()`` and leave a gap.
    """
    now = time.time()
    return (time.strftime(TIMESTAMP_FORMAT, time.localtime(now - from_days * 86400)),
            time.strftime(TIMESTAMP_FORMAT, time.localtime(now - to_days * 86400)))


def window_slices(start: str, end: str) -> List[Tuple[str, str]]:
    """``[start, end]`` cut into :data:`SLICE_SECONDS`-wide pieces, oldest first.

    The window returned unchanged, as a single slice, when either end is a form sacct
    accepts but :func:`epoch` cannot read -- ``now-30days`` and bare dates both reach
    here, and a window that cannot be cut is still a window that can be queried. Also
    when the two are the same instant or inverted, where cutting has nothing to do.

    Slices meet rather than overlap, so a job is returned by two of them only when it
    genuinely spans the boundary. :func:`fetch_window` dedups those.
    """
    first, last = epoch(start), epoch(end)
    if first is None or last is None or last <= first:
        return [(start, end)]
    def stamp(at: int) -> str:
        return time.strftime(TIMESTAMP_FORMAT, time.localtime(at))

    slices, at = [], first
    while at < last:
        nxt = min(at + SLICE_SECONDS, last)
        slices.append((stamp(at), stamp(nxt)))
        at = nxt
    return slices


def end_of_day(start: str) -> Optional[str]:
    """sacct end-time that closes the calendar day of ``start`` (the following
    midnight), so ``-S <date>`` alone selects just that day.

    Returns None when ``start`` is not an ISO date/datetime we can parse (e.g. a
    relative form like ``now-2days``), leaving the window open-ended.
    """
    for fmt in (TIMESTAMP_FORMAT, "%Y-%m-%d"):
        try:
            dt = datetime.strptime(start, fmt)
            break
        except ValueError:
            continue
    else:
        return None
    nxt = (dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return nxt.strftime(TIMESTAMP_FORMAT)


def epoch(value: str) -> Optional[int]:
    """sacct timestamp to int epoch; 'Unknown'/'' to None."""
    if not value or value == "Unknown":
        return None
    try:
        return int(time.mktime(time.strptime(value, TIMESTAMP_FORMAT)))
    except Exception:
        return None


def query_jobid(jobid: str) -> str:
    """The JobID to pass to ``sacct -j``.

    Array jobs cancelled or pending before expansion appear as
    ``BASE_[range%throttle]`` (e.g. ``18114115_[0-719%64]``); ``sacct -j`` cannot
    parse that element spec, so query the base id instead; it returns the same
    record. Plain ids and expanded tasks (``BASE_N``) pass through unchanged.
    """
    return jobid.split("_", 1)[0] if "_[" in jobid else jobid


def chunk_jobids(ids: List[str], limit: int,
                 max_count: Optional[int] = None) -> List[List[str]]:
    """Split ids into runs whose comma-joined form stays within limit bytes.

    Order is preserved and every id appears exactly once. An id longer than
    the limit gets its own chunk (it cannot be split). With ``max_count``, a
    chunk also holds at most that many ids.
    """
    chunks: List[List[str]] = []
    current: List[str] = []
    joined = 0
    for jobid in ids:
        added = len(jobid) + (1 if current else 0)  # +1 for the comma
        if current and (joined + added > limit
                        or (max_count is not None and len(current) >= max_count)):
            chunks.append(current)
            current, joined = [], 0
            added = len(jobid)
        current.append(jobid)
        joined += added
    if current:
        chunks.append(current)
    return chunks


def run_capture(cmd: List[str], timeout: Optional[float], what: str,
                soft: bool = False) -> Optional[str]:
    """Run ``cmd`` and capture stdout with a hard, process-group timeout.

    A child holding the pipe cannot hang us (the whole group is killed).
    ``timeout=None`` disables the limit. With ``soft=True`` a timeout or non-zero
    exit returns None instead of raising (used for best-effort calls).
    """
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 universal_newlines=True, start_new_session=True)
    except FileNotFoundError:
        raise JobscopeError("command not found: %s" % cmd[0])
    except OSError as exc:  # e.g. E2BIG when the argv exceeds the kernel limit
        if soft:
            return None
        raise JobscopeError(
            "%s could not be started (%s) -- if the argument list is too long,\n"
            "narrow the selection with -N, -D, or -S/-E." % (what, exc))
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        if soft:
            return None
        raise JobscopeError(
            "%s timed out after %gs; the selection likely spans too many jobs for\n"
            "sacct to return in time. Narrow it (in increasing order of help):\n"
            "  -N N             fewer jobs   - your most recent N (e.g. -N 50)\n"
            "  -D N             fewer days   - the last N days     (e.g. -D 3)\n"
            "  -S DATE -E DATE  a narrow explicit window (best), e.g.\n"
            "                   -S 2026-05-26 -E 2026-06-02\n"
            "Or raise/disable the cap with --timeout SECONDS (--timeout 0 disables it)."
            % (what, timeout))
    if proc.returncode != 0:
        if soft:
            return None
        raise JobscopeError("%s failed (exit %d): %s"
                            % (what, proc.returncode, (err or "").strip()))
    return out


_HOSTLIST: Dict[str, Tuple[str, ...]] = {}

# One bracket group of digits, commas and dashes, with nothing bracketed after it. Anything
# outside that -- nested groups, two groups in one name, a non-numeric range -- is declined
# rather than guessed at, and goes to scontrol. Being narrow is the point: a wrong expansion
# would attribute a card to the wrong job silently, where a decline just costs a fork.
_RANGE_GROUP = re.compile(r"^([^\[\]]*)\[([0-9,\-]+)\]([^\[\]]*)$")


def _split_hosts(text: str) -> List[str]:
    """``text`` split on its top-level commas: a comma inside ``[]`` is part of a range."""
    parts, depth, current = [], 0, []
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [p for p in parts if p]


def _expand_ranges(text: str) -> Optional[Tuple[str, ...]]:
    """``holygpu8a[05304,06504]`` -> the hostnames, or None if this cannot be sure.

    None rather than a best guess for anything the pattern does not cover, because the
    caller uses these names to decide whether a card belongs to a job: a wrong expansion
    drops a card that was really the job's, or keeps one that was not, and both are silent.
    Declining costs one subprocess.

    Zero-padding follows the low end of each range, which is how Slurm writes them --
    ``node[08-11]`` is ``node08``..``node11``, not ``node8``.
    """
    hosts: List[str] = []
    for entry in _split_hosts(text):
        if "[" not in entry:
            hosts.append(entry)
            continue
        match = _RANGE_GROUP.match(entry)
        if match is None:
            return None
        prefix, body, suffix = match.groups()
        for item in body.split(","):
            low, dash, high = item.partition("-")
            if not low.isdigit() or (dash and not high.isdigit()):
                return None
            if not dash:
                hosts.append(prefix + low + suffix)
                continue
            start, end = int(low), int(high)
            if end < start:
                return None
            hosts.extend(prefix + str(n).zfill(len(low)) + suffix
                         for n in range(start, end + 1))
    return tuple(hosts)


def expand_nodelist(nodelist: str, timeout: Optional[float] = None) -> Tuple[str, ...]:
    """Every host a Slurm nodelist names, bracketed ranges expanded.

    ``squeue``'s ``%N`` is compressed -- ``holygpu8a[05304,06504]`` -- and every
    comparison downstream is against one label value at a time, so an unexpanded list
    matches nothing. :func:`jobscope.probe._partition_nodes` sidesteps this by asking
    ``sinfo`` for ``%n``, which Slurm expands itself; ``squeue`` offers no such field,
    so the expansion has to be asked for.

    Unbracketed lists take the fast path and cost no subprocess, which is most of them:
    a single-node job is the common case, and it is also what the cache below is for --
    the same compressed list recurs across every element of an array job.

    Best-effort: an unparseable or unexpandable list yields the comma-split original
    rather than raising. A cross-check that cannot run should weaken to "no opinion",
    never to a wrong opinion about which host a card belongs to.
    """
    text = (nodelist or "").strip()
    if not text or text in ("(null)", "None", "n/a"):
        return ()
    if "[" not in text:
        return tuple(part for part in text.split(",") if part)
    if text in _HOSTLIST:
        return _HOSTLIST[text]
    parsed = _expand_ranges(text)
    if parsed is not None:
        _HOSTLIST[text] = parsed
        return parsed
    # Only the forms the parser declines reach scontrol. That fork costs ~36ms, and a
    # partition-wide selection holds enough distinct bracketed lists to make it the most
    # expensive thing in the report -- 27 lists, ~1.0s, measured on this cluster, against
    # 0.3ms for all 27 parsed in-process.
    out = run_capture(["scontrol", "show", "hostnames", text], timeout,
                      "scontrol show hostnames", soft=True)
    hosts = tuple((out or "").split())
    if not hosts:
        # Keep the brackets out of the result: a literal "holygpu8a[1,2]" compared
        # against a host label is a silent false negative, where an empty tuple is an
        # honest "could not tell".
        hosts = ()
    _HOSTLIST[text] = hosts
    return hosts


def _window_filters(selection: Selection, start: str, end: str) -> List[str]:
    """The sacct flags that narrow a window selection, without an output format.

    Shared by the two commands that query a window -- the id listing and the bulk
    fetch -- because they must select the *same* jobs. They drifting apart is how a
    fetch would return rows the listing never offered, or miss rows it did.

    Every one of these is pushed to slurmdbd rather than filtered here: -X alone is
    the difference between one row per job and one per job step.
    """
    cmd = ["sacct", "-X", "-S", start, "-E", end]
    # -a spans every user; otherwise scope to one. Mutually exclusive by
    # construction -- the CLI rejects -a together with -u.
    cmd += ["-a"] if selection.all_users else ["-u", selection.user]
    if selection.account:
        cmd += ["-A", selection.account]
    if selection.partition:
        cmd += ["-r", selection.partition]
    # Always filtered by state, so sacct never hands back a job that is still
    # running: `finished` reporting a RUNNING job was the bug this closed.
    cmd += ["-s", ",".join(states_for(selection.state))]
    return cmd


def _select_cmd(selection: Selection, start: str, end: str) -> List[str]:
    """The sacct command that lists candidate job IDs for a window."""
    return _window_filters(selection, start, end) + [
        "--noheader", "-P", "-o", "JobID,State"]


def _window_fetch_cmd(selection: Selection, start: str, end: str) -> List[str]:
    """The sacct command that fetches full records for a window, in one call.

    :func:`_select_cmd` plus the columns, which is the whole difference: measured on
    an 8587-job selection the ten extra fields cost 0.05s against that listing's
    0.36s, while the 43 ``sacct -j`` calls the listing existed to feed cost 3.6s. The
    columns are very nearly free; the round trips are not.
    """
    return _window_filters(selection, start, end) + [
        "--noheader", "-P", "--units=G", "-o", FETCH_FIELDS]


def _query_ids(selection: Selection, start: str, end: str,
               timeout: Optional[float]) -> List[str]:
    """Finished job IDs in ``[start, end]``, oldest first as sacct returns them."""
    out = run_capture(_select_cmd(selection, start, end), timeout, "sacct query")
    ids = []
    for line in out.splitlines():
        parts = line.split("|", 1)
        # Belt and braces behind the -s filter: whatever sacct returns, a job that has
        # not finished has no final numbers and does not belong here.
        if len(parts) == 2 and not parts[1].upper().startswith(UNFINISHED_STATES):
            ids.append(parts[0])
    return ids


def _query_lastn(selection: Selection, timeout: Optional[float]) -> List[str]:
    """Walk back a day at a time until ``lastn`` jobs have been found.

    ``-N`` asks for the most recent few jobs and sacct has no "last N", so a window
    has to be scanned and trimmed. Two things make the naive version slow: the default
    lookback is a month, and listing job IDs costs roughly in proportion to the span
    -- on one busy partition a day took 1.2s and thirty days did not return inside 60.

    So each step queries **one day**, the day before the last one, and accumulates.
    Reaching five days back costs five one-day queries rather than one-, two-, three-,
    four- and five-day scans, which would re-list the same jobs five times over.

    The span covered is written back onto ``selection``, so the header reports how far
    back it actually looked.
    """
    ids: List[str] = []
    seen = set()
    for day in range(DEFAULT_LOOKBACK_DAYS):
        start, end = day_slice(day + 1, day)
        try:
            found = _query_ids(selection, start, end, timeout)
        except JobscopeError:
            # A later day timing out should not throw away what earlier days found:
            # some jobs beat an error, and the header says how far back it managed.
            if not ids:
                raise
            break
        # Walking backwards, so each day is older than the one before: prepend to keep
        # the list ascending, as the caller's [-lastn:] trim expects. A job that ran
        # across a boundary matches both days, hence the dedup.
        ids[:0] = [jid for jid in found if jid not in seen]
        seen.update(found)
        selection.starttime = start
        selection.endtime = selection.endtime or end     # the first slice ends at now
        if len(ids) >= selection.lastn:
            break
    return ids


def describe_window(selection: Selection) -> str:
    """How a window selection reads in the header, **without querying for it**.

    Separated from :func:`select_jobs` because the streaming window path
    (:func:`fetch_window`) needs the header before it has asked sacct anything, and
    the whole point of that path is that there is no listing pass to take it from.
    ``-N`` keeps its description in ``select_jobs``: "last 20 jobs" is a fact about
    what the day-walk found, not about the window.
    """
    if selection.days is not None:
        desc = "last %d day%s" % (selection.days, "s" if selection.days != 1 else "")
    else:
        desc = format_window(*selection.window())
    if selection.state != "all":
        desc += ", %s" % selection.state
    return desc


def select_jobs(selection: Selection, timeout: Optional[float]) -> Tuple[List[str], str]:
    """Return ``(jobids, description)``. Per-job data is fetched afterward in bulk.

    The **explicit-JOBID and -N paths only**, now that a window selection streams
    through :func:`fetch_window` without listing its ids first. Both genuinely need
    the list up front: -N has to trim to the newest N, and explicit ids are the
    selection.
    """
    if selection.jobids:
        return list(selection.jobids), "%d job ID(s)" % len(selection.jobids)

    # Snapshot the narrowings before the queries run: _query_lastn writes the span it
    # walked back onto `selection`, so asking afterwards would offer a window the user
    # never chose.
    narrowings = _narrowings(selection)

    # Only when no window was named at all: with -S or -E given, the user has said
    # where to look and the day-walk would quietly ignore it.
    if selection.lastn is not None and not selection.starttime and not selection.endtime:
        ids = _query_lastn(selection, timeout)
    else:
        ids = _query_ids(selection, *selection.window(), timeout=timeout)

    if selection.lastn is not None:
        ids = ids[-selection.lastn:]  # sacct lists ascending, so the last N are newest
        desc = "last %d job%s" % (selection.lastn,
                                  "s" if selection.lastn != 1 else "")
        if selection.state != "all":
            desc += ", %s" % selection.state
    else:
        desc = describe_window(selection)
    _note_if_broad(ids, narrowings)
    return ids, desc


def _narrowings(selection: Selection) -> List[str]:
    """The narrowings ``selection`` is not already using, worded for what it did ask.

    Repeating a flag back to someone who just typed it reads as though it did not take
    effect, so a selection that named a window is asked for a shorter one rather than
    having -D/-S/-E spelled at it again.
    """
    if selection.lastn is not None:
        # -N pins the count at exactly lastn, so neither a partition nor a shorter
        # window reduces it -- those change which jobs come back, not how many.
        return ["a smaller -N"]
    offers = []
    if not selection.partition:
        offers.append("-p PARTITION")
    if selection.days is None and not (selection.starttime or selection.endtime):
        offers.append("-D 1 for a single day")     # still on the default 30-day window
    elif (selection.days or 0) > 1 or selection.starttime or selection.endtime:
        offers.append("a shorter window")          # -D 1 is already the floor
    offers.append("-N to take only the newest N")
    return offers


def _note_if_broad(ids: List[str], narrowings: List[str]) -> None:
    """Say when a selection is broad enough to be worth narrowing, and how.

    Here rather than in :func:`fetch_chunks` so it lands *before* the batches start:
    the count is only knowable after the selection query, so this is the earliest it
    can be said, and saying it early is the whole point -- it leaves room to abort.
    """
    if len(ids) < LARGE_SELECTION:
        return
    # Opens on the count rather than repeating "N jobs selected", which the batching
    # note on the very next line already says.
    print("note: %d jobs is a broad selection -- all of them stay in memory until the"
          " run ends, so this will be slow.%s"
          % (len(ids), " Narrow it with %s." % _joined(narrowings) if narrowings else ""),
          file=sys.stderr)


def _joined(items: List[str]) -> str:
    """``"a, b or c"`` -- for offering alternatives in a sentence."""
    if len(items) < 2:
        return "".join(items)
    return "%s or %s" % (", ".join(items[:-1]), items[-1])


def _parse_fetch_lines(out: str, records: Dict[str, JobRecord]) -> None:
    """Parse one bulk-query output into records, keyed by JobID.

    The split is bounded by :data:`FETCH_FIELD_COUNT` rather than by a literal, so the
    field list is the only place the shape is written down. The order below still has
    to match that list -- but a *count* that drifts is the silent failure, and this is
    what stops it.
    """
    for line in out.splitlines():
        parts = line.split("|", FETCH_FIELD_COUNT - 1)
        if len(parts) < FETCH_FIELD_COUNT:
            continue
        (jobid, state, name, elapsed, nnodes, alloc_tres,
         start, end, jobid_raw, cluster, user, account, partition, admin) = parts
        start_epoch = epoch(start)
        end_epoch = epoch(end) or int(time.time())  # running job -> now
        records[jobid] = JobRecord(
            jobid=jobid,
            state=(state.split() or ["?"])[0],
            name=name or "?",
            runtime=elapsed or "-",
            nodes=nnodes or "-",
            gpus=gpus_from_tres(alloc_tres),
            stats=decode_admin_comment(admin),
            start=start_epoch,
            end=end_epoch,
            duration=(max(end_epoch - start_epoch, 1) if start_epoch else None),
            jobid_raw=jobid_raw,
            cluster=cluster,
            user=user or "?",
            account=account,
            partition=partition,
            gpu_model=gpu_model_from_tres(alloc_tres),
        )


def fetch_window(selection: Selection, timeout: Optional[float],
                 ) -> Iterator[Tuple[List[str], Dict[str, JobRecord]]]:
    """Stream a window selection one time slice at a time, ``(ids, records)`` a slice.

    The same shape :func:`fetch_chunks` yields, so ``_enrich`` cannot tell which one it
    is draining -- but selected and fetched in *one* call per slice rather than a
    listing pass followed by a call per 200 ids. That listing pass was not a saving:
    it already materialised every row in the window (1.66 GB of sacct RSS for a
    cluster-wide day, output format irrelevant), and the id batching then chunked only
    the pass that had no memory problem. See :data:`SLICE_SECONDS`.

    Unlike ``fetch_chunks`` the records dict is **per chunk, not cumulative**. Nothing
    downstream indexes outside the ids it was handed, and holding every record to the
    end of the run is the other half of what made a wide selection expensive.

    **A slice is how much is fetched; it is not how much is yielded.** Those have to be
    separate, and conflating them cost this function its streaming: one sacct call is
    right for the *server* -- it is the whole point of cutting by window -- but
    ``run_capture`` buffers that call whole, so yielding once per slice meant a default
    ``-D 1`` produced a single chunk of nine thousand jobs. Every metric query for all
    of them then ran before the first row could be drawn, turning a report that used to
    start printing in seconds into one that printed nothing for minutes. So the slice's
    records are handed out :data:`STREAM_CHUNK` at a time, which costs nothing and puts
    the granularity back where it was.

    A slice that fails is reported and skipped rather than ending the run: the rest of
    the window is still worth having, and saying which span is missing is better than
    either a silent hole or nothing at all.
    """
    slices = window_slices(*selection.window())
    if len(slices) > 1:
        print("note: querying %d day-slices of the window, one sacct call each"
              % len(slices), file=sys.stderr)
    seen: set = set()
    for start, end in slices:
        out = run_capture(_window_fetch_cmd(selection, start, end), timeout,
                          "sacct query", soft=True)
        if out is None:
            print("note: no data for %s -- that span is missing from this report."
                  " Narrow the window or raise --timeout to include it."
                  % format_window(start, end), file=sys.stderr)
            continue
        found: Dict[str, JobRecord] = {}
        _parse_fetch_lines(out, found)
        # Two filters, both of which the id listing used to apply. A job spanning a
        # slice boundary is returned by both slices, and the first to reach it owns it
        # or it would be rendered twice; and belt-and-braces behind -s, a job that has
        # not ended has no final numbers and does not belong in a finished report.
        ids = [jid for jid, record in found.items()
               if jid not in seen and not record.unfinished]
        if not ids:
            continue
        seen.update(ids)
        # Handed out in chunks, not in one piece -- see the docstring. Sacct returns a
        # slice in ascending order and this preserves it, so a chunk is also a narrow
        # span of time, which is what keeps the batched GPU discovery inside it cheap.
        for at in range(0, len(ids), STREAM_CHUNK):
            batch = ids[at:at + STREAM_CHUNK]
            yield batch, {jid: found[jid] for jid in batch}


def fetch_chunks(jobids: List[str], timeout: Optional[float],
                 ) -> Iterator[Tuple[List[str], Dict[str, JobRecord]]]:
    """Fetch job data batch by batch, yielding ``(ready_ids, records)`` pairs.

    ``ready_ids`` is the next run of requested ids (in input order) whose data
    has been queried, so callers can render incrementally; ``records`` is the
    cumulative dict shared across yields (do not mutate). Concatenating every
    ``ready_ids`` reproduces ``jobids`` exactly, so emission order always
    matches the input. Array ids are de-bracketed to their base for the query
    (see query_jobid) and aliased back to the requested id at emission.
    ``timeout`` applies to each batch call. When there is more than one batch,
    stderr gets one upfront note (printed lazily, on first iteration), then a
    progress note per :data:`NOTE_EVERY` fetched jobs and on the final batch.
    """
    if not jobids:
        return

    query_ids, seen = [], set()
    for jobid in jobids:
        base = query_jobid(jobid)
        if base not in seen:
            seen.add(base)
            query_ids.append(base)

    chunks = chunk_jobids(query_ids, JOBID_ARG_LIMIT, JOBS_PER_CHUNK)
    if len(chunks) > 1:
        print("note: %d jobs selected -- fetching sacct data in %d batches"
              % (len(query_ids), len(chunks)), file=sys.stderr)
    records: Dict[str, JobRecord] = {}
    queried: set = set()
    pos, done = 0, 0
    for i, chunk in enumerate(chunks, 1):
        out = run_capture(
            ["sacct", "-j", ",".join(chunk), "-X", "-P", "-n", "--units=G",
             "-o", FETCH_FIELDS],
            timeout, "sacct query")
        _parse_fetch_lines(out, records)
        queried.update(chunk)
        done += len(chunk)
        if len(chunks) > 1 and (done // NOTE_EVERY > (done - len(chunk)) // NOTE_EVERY
                                or i == len(chunks)):
            print("note: batch %d/%d done (%d/%d jobs)"
                  % (i, len(chunks), done, len(query_ids)), file=sys.stderr)
        # Emit the longest run of not-yet-emitted requested ids whose base has
        # been queried; a base shared across distant requested ids can defer
        # ids to a later yield, but never reorder them.
        ready: List[str] = []
        while pos < len(jobids) and query_jobid(jobids[pos]) in queried:
            requested = jobids[pos]
            base = query_jobid(requested)
            if requested not in records and base in records:
                records[requested] = records[base]
            ready.append(requested)
            pos += 1
        yield ready, records


def fetch(jobids: List[str], timeout: Optional[float]) -> Dict[str, JobRecord]:
    """Bulk sacct query for all jobs, keyed by JobID.

    Drains :func:`fetch_chunks`: array ids are de-bracketed to their base for
    the query (see query_jobid) and aliased back to the originally-requested id,
    and id lists whose joined form would exceed the kernel's per-argument limit
    are queried in batches (with progress notes on stderr); ``timeout`` applies
    to each batch call.
    """
    records: Dict[str, JobRecord] = {}
    for _ready, chunk_records in fetch_chunks(jobids, timeout):
        records = chunk_records
    return records
