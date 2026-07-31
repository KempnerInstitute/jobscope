"""Job selection and the bulk sacct fetch.

Jobs are selected with one ``sacct`` query, then all of their data is retrieved
with bulk ``sacct -j`` queries, no per-job jobstats calls and no job-count
cap. When the id list would exceed the kernel's per-argument size limit the
``-j`` query is split into batches (see :data:`JOBID_ARG_LIMIT`), with progress
notes on stderr. The AdminComment blob returned by the bulk query is decoded
into each :class:`JobRecord`.
"""

import getpass
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Optional, Tuple

from .blob import decode_admin_comment, gpus_from_tres
from .errors import JobscopeError

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

# What each -t name covers. Separated rather than lumped into one "failed" bucket
# because they are different problems: a timeout usually means the walltime or the
# resource request was wrong, a cancellation is a person, and a true failure is the
# job itself. RUNNING and PENDING appear in none of them -- `finished` means finished,
# and `jobscope running` is the live view.
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

DEFAULT_STATE = "completed"

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

# Chunk bounds for the batched sacct -j queries. JOBS_PER_CHUNK is the primary
# bound: sacct costs ~35 ms per call regardless of id count, so small batches
# stream first rows sooner at negligible overhead. JOBID_ARG_LIMIT backstops the
# Linux per-argument cap (MAX_ARG_STRLEN, 128 KiB) for pathologically long ids.
# NOTE_EVERY throttles the stderr progress notes to one per that many jobs.
JOBID_ARG_LIMIT = 16384
JOBS_PER_CHUNK = 200
NOTE_EVERY = 4096


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


def _select_cmd(selection: Selection, start: str, end: str) -> List[str]:
    """The sacct command that lists candidate job IDs for a window."""
    cmd = ["sacct", "-X", "-S", start, "-E", end,
           "--noheader", "-P", "-o", "JobID,State"]
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


def _query_ids(selection: Selection, start: str, end: str,
               timeout: Optional[float]) -> List[str]:
    """Finished job IDs in ``[start, end]``, oldest first as sacct returns them."""
    out = run_capture(_select_cmd(selection, start, end), timeout, "sacct query")
    ids = []
    for line in out.splitlines():
        parts = line.split("|", 1)
        # Belt and braces behind the -s filter: whatever sacct returns, a job that has
        # not finished has no final numbers and does not belong here.
        if len(parts) == 2 and not parts[1].upper().startswith(
                ("PENDING", "RUNNING", "SUSPENDED", "REQUEUED")):
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


def select_jobs(selection: Selection, timeout: Optional[float]) -> Tuple[List[str], str]:
    """Return ``(jobids, description)``. Per-job data is fetched afterward in bulk."""
    if selection.jobids:
        return list(selection.jobids), "%d job ID(s)" % len(selection.jobids)

    # Only when no window was named at all: with -S or -E given, the user has said
    # where to look and the day-walk would quietly ignore it.
    if selection.lastn is not None and not selection.starttime and not selection.endtime:
        ids = _query_lastn(selection, timeout)
    else:
        ids = _query_ids(selection, *selection.window(), timeout=timeout)

    start, end = selection.window()
    if selection.lastn is not None:
        ids = ids[-selection.lastn:]  # sacct lists ascending, so the last N are newest
        desc = "last %d job%s" % (selection.lastn,
                                  "s" if selection.lastn != 1 else "")
    elif selection.days is not None:
        desc = "last %d day%s" % (selection.days, "s" if selection.days != 1 else "")
    else:
        desc = format_window(start, end)
    if selection.state != "all":
        desc += ", %s" % selection.state
    return ids, desc


def _parse_fetch_lines(out: str, records: Dict[str, JobRecord]) -> None:
    """Parse one bulk-query output into records, keyed by JobID."""
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
        # AdminComment (a '|'-free base64 blob) is queried last so a fixed maxsplit is safe.
        out = run_capture(
            ["sacct", "-j", ",".join(chunk), "-X", "-P", "-n", "--units=G",
             "-o", "JobID,State,JobName,Elapsed,NNodes,AllocTRES,"
                   "Start,End,JobIDRaw,Cluster,User,AdminComment"],
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
