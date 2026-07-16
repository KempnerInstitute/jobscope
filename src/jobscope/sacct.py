"""Job selection and the single bulk sacct fetch.

Jobs are selected with one ``sacct`` query, then all of their data is retrieved
with a second bulk ``sacct -j`` query -- no per-job jobstats calls and no
job-count cap. The AdminComment blob returned by the second query is decoded into
each :class:`JobRecord`.
"""

import getpass
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .blob import decode_admin_comment, gpus_from_tres
from .errors import JobscopeError

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

FAILED_STATES = ("FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL,CANCELLED,"
                 "DEADLINE,BOOT_FAIL,PREEMPTED")


@dataclass
class Selection:
    """Inputs that determine which jobs to report on."""

    user: str
    jobids: List[str] = field(default_factory=list)
    account: Optional[str] = None
    partition: Optional[str] = None
    state: str = "all"
    lastn: Optional[int] = None
    days: Optional[int] = None
    starttime: Optional[str] = None
    endtime: Optional[str] = None


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


def default_user() -> Optional[str]:
    """Current user from $USER, falling back to getpass; None if undeterminable."""
    user = os.environ.get("USER")
    if not user:
        try:
            user = getpass.getuser()
        except Exception:
            user = None
    return user


def days_to_window(days: int) -> Tuple[str, str]:
    """``(start, end)`` sacct timestamps for the last ``days`` days ending now."""
    now = time.time()
    return (time.strftime(TIMESTAMP_FORMAT, time.localtime(now - days * 86400)),
            time.strftime(TIMESTAMP_FORMAT, time.localtime(now)))


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
    parse that element spec, so query the base id instead -- it returns the same
    record. Plain ids and expanded tasks (``BASE_N``) pass through unchanged.
    """
    return jobid.split("_", 1)[0] if "_[" in jobid else jobid


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
            "%s timed out after %gs -- the selection likely spans too many jobs for\n"
            "sacct to return in time. Narrow it (in increasing order of help):\n"
            "  -N N             fewer jobs   -- your most recent N (e.g. -N 50)\n"
            "  -D N             fewer days   -- the last N days     (e.g. -D 3)\n"
            "  -S DATE -E DATE  a narrow explicit window (best) -- e.g.\n"
            "                   -S 2026-05-26 -E 2026-06-02\n"
            "Or raise/disable the cap with --timeout SECONDS (--timeout 0 disables it)."
            % (what, timeout))
    if proc.returncode != 0:
        if soft:
            return None
        raise JobscopeError("%s failed (exit %d): %s"
                            % (what, proc.returncode, (err or "").strip()))
    return out


def select_jobs(selection: Selection, timeout: Optional[float]) -> Tuple[List[str], str]:
    """Return ``(jobids, description)``. Per-job data is fetched afterward in bulk."""
    if selection.jobids:
        return list(selection.jobids), "%d job ID(s)" % len(selection.jobids)

    start = selection.starttime or "now-30days"
    end = selection.endtime or "now"
    cmd = ["sacct", "-u", selection.user, "-X", "-S", start, "-E", end,
           "--noheader", "-P", "-o", "JobID,State"]
    if selection.account:
        cmd += ["-A", selection.account]
    if selection.partition:
        cmd += ["-r", selection.partition]
    if selection.state == "completed":
        cmd += ["-s", "COMPLETED"]
    elif selection.state == "failed":
        cmd += ["-s", FAILED_STATES]
    out = run_capture(cmd, timeout, "sacct query")

    ids = []
    for line in out.splitlines():
        parts = line.split("|", 1)
        if len(parts) == 2 and not parts[1].startswith("PENDING"):
            ids.append(parts[0])

    if selection.lastn is not None:
        ids = ids[-selection.lastn:]  # sacct lists ascending, so the last N are the most recent
        desc = "last %d jobs" % selection.lastn
    elif selection.days is not None:
        desc = "last %d day%s" % (selection.days, "s" if selection.days != 1 else "")
    else:
        desc = "%s .. %s" % (start, end)
    if selection.state != "all":
        desc += ", %s" % selection.state
    return ids, desc


def fetch(jobids: List[str], timeout: Optional[float]) -> Dict[str, JobRecord]:
    """One bulk sacct query for all jobs, keyed by JobID.

    Array ids are de-bracketed to their base for the query (see query_jobid), then
    each record is also aliased back to the originally-requested id so a caller that
    looks it up by that id still resolves it.
    """
    records: Dict[str, JobRecord] = {}
    if not jobids:
        return records

    query_ids, seen = [], set()
    for jobid in jobids:
        base = query_jobid(jobid)
        if base not in seen:
            seen.add(base)
            query_ids.append(base)

    # AdminComment (a '|'-free base64 blob) is queried last so a fixed maxsplit is safe.
    out = run_capture(
        ["sacct", "-j", ",".join(query_ids), "-X", "-P", "-n", "--units=G",
         "-o", "JobID,State,JobName,Elapsed,NNodes,AllocTRES,"
               "Start,End,JobIDRaw,Cluster,User,AdminComment"],
        timeout, "sacct query")

    for line in out.splitlines():
        parts = line.split("|", 11)
        if len(parts) < 12:
            continue
        (jobid, state, name, elapsed, nnodes, alloc_tres,
         start, end, jobid_raw, cluster, user, admin) = parts
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
        )
    for requested in jobids:
        base = query_jobid(requested)
        if requested not in records and base in records:
            records[requested] = records[base]
    return records
