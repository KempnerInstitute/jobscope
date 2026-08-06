#!/usr/bin/env python3
"""Deprecated wrapper for ``jobscope running``.

This started life as a standalone script (squeue + Prometheus, no install, running
on the login node's system python). Its logic now lives in the package -- see
``src/jobscope/running.py`` -- and "which jobs" is now a mode word rather than a
subcommand, so the running view is ``jobscope running``. This wrapper stays so
existing command lines and scripts keep working.

Flag translation:
  (mode)             ->  running             the running selection
  --min-runtime DUR  ->  --min-elapsed DUR   rewritten; the alias is gone
  no -u given        ->  --all-users         this script showed every user;
                                             jobscope defaults to you
  --all / --ext      ->  --dcgm              the full metric catalog

Prefer ``jobscope running`` directly. Unlike this script it needs an installed
jobscope, because the package requires Python >= 3.9 -- see the message below if
you are on an older interpreter.

Note the granularity changed with the argument tree: ``jobscope running`` prints
one row per job. For the per-GPU rows this script used to show, add ``--per-gpu``.

Author: Bala Desinghu, Senior AI/HPC Research Computing Engineer, Kempner Institute, Harvard
"""

import os
import sys

_MIN_PYTHON = (3, 9)

_NO_PACKAGE = """\
jobscope_live.py now delegates to `jobscope running`, which needs the installed
jobscope package (Python >= {min}; this interpreter is {have}).

  pipx install {repo}          # or: pip install --user {repo}
  jobscope running -a -p <partition>

To run straight from a checkout instead:

  python{min} -m venv .venv && .venv/bin/pip install -e {repo}
  .venv/bin/jobscope running -a -p <partition>
"""


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _translate(argv):
    """Map this script's old flags onto the current ones.

    ``--min-runtime`` is rewritten to ``--min-elapsed``: jobscope accepted it as an
    alias until the flag surface was pruned, and this wrapper exists precisely so
    that pruning does not reach the people still typing the old script's spellings.
    Two things also differ in meaning: this script showed every user's jobs unless
    ``-u`` narrowed it, where jobscope defaults to your own; and the extended catalog
    moved from ``--all`` to ``--dcgm``.
    """
    out = []
    for a in argv:
        if a in ("--all", "--ext"):
            out.append("--dcgm")
        elif a == "--min-runtime":
            out.append("--min-elapsed")
        elif a.startswith("--min-runtime="):
            out.append("--min-elapsed=" + a.split("=", 1)[1])
        else:
            out.append(a)
    selects_user = any(a in ("-u", "--user", "-a", "--all-users")
                       or a.startswith("--user=") for a in out)
    return out if selects_user else out + ["--all-users"]


def main() -> int:
    if sys.version_info < _MIN_PYTHON:
        sys.stderr.write(_NO_PACKAGE.format(
            min="%d.%d" % _MIN_PYTHON,
            have="%d.%d.%d" % sys.version_info[:3],
            repo=_repo_root()))
        return 1
    try:
        from jobscope.cli import main as jobscope_main
    except ImportError:
        # Fall back to a sibling checkout before giving up, so the wrapper still
        # works in a dev tree where the package was never installed.
        sys.path.insert(0, os.path.join(_repo_root(), "src"))
        try:
            from jobscope.cli import main as jobscope_main
        except ImportError:
            sys.stderr.write(_NO_PACKAGE.format(
                min="%d.%d" % _MIN_PYTHON,
                have="%d.%d.%d" % sys.version_info[:3],
                repo=_repo_root()))
            return 1

    sys.stderr.write("note: jobscope_live.py is deprecated; use 'jobscope running'\n")
    jobscope_main(["running"] + _translate(sys.argv[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
