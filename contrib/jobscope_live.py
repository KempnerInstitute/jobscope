#!/usr/bin/env python3
"""Deprecated wrapper for ``jobscope live``.

This started life as a standalone script (squeue + Prometheus, no install, running
on the login node's system python). Its logic now lives in the package, as
``jobscope live`` -- see ``src/jobscope/live.py``. This wrapper stays so existing
command lines and scripts keep working; it translates the two flags whose names
changed and hands off.

Flag translation:
  --min-runtime DUR  ->  --min-elapsed DUR   (the package's --min-runtime is the
                                              DIAG threshold, in seconds)
  no -u given        ->  --all-users         (this script defaulted to every
                                              user; jobscope live defaults to you)

Prefer ``jobscope live`` directly. Unlike this script it needs an installed
jobscope, because the package requires Python >= 3.9 -- see the message below if
you are on an older interpreter.

Author: Bala Desinghu, Senior AI/HPC Research Computing Engineer, Kempner Institute, Harvard
"""

import os
import sys

_MIN_PYTHON = (3, 9)

_NO_PACKAGE = """\
jobscope_live.py now delegates to `jobscope live`, which needs the installed
jobscope package (Python >= {min}; this interpreter is {have}).

  pipx install {repo}          # or: pip install --user {repo}
  jobscope live -a -p <partition>

To run straight from a checkout instead:

  python{min} -m venv .venv && .venv/bin/pip install -e {repo}
  .venv/bin/jobscope live -a -p <partition>
"""


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _translate(argv):
    """Map this script's old flag names onto the `jobscope live` ones.

    ``--min-runtime`` needs no rewriting -- `jobscope live` accepts it as an alias
    of ``--min-elapsed``. Only the all-users default actually differs: this script
    showed every user's jobs unless ``-u`` narrowed it, where `jobscope live`
    defaults to your own.
    """
    selects_user = any(a in ("-u", "--user", "-a", "--all-users")
                       or a.startswith("--user=") for a in argv)
    return list(argv) if selects_user else list(argv) + ["--all-users"]


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

    sys.stderr.write("note: jobscope_live.py is deprecated; use 'jobscope live'\n")
    jobscope_main(["live"] + _translate(sys.argv[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
