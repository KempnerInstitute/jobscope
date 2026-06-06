#!/usr/bin/env bash
# install_venv.sh - build the jobstats_plot virtualenv at a base dir YOU choose and
# record it in plot_util/venv_path.conf. jobstats_plot reads that file and runs
# itself under <base>/.venv automatically -- no PATH/launcher setup needed; just
# run jobstats_plot (by path, or however you put it on your PATH).
#
# Usage:
#   bash plot_util/setup/install_venv.sh                # prompt for the base dir
#   bash plot_util/setup/install_venv.sh /path/to/base  # non-interactive
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENVCONF="$ROOT/plot_util/venv_path.conf"
# Canonical shared deployment -- the default base dir (its .venv is the shared venv).
SHARED_ROOT="/n/holylfs06/LABS/kempner_shared/Everyone/cluster_scripts/job_eff/kempner-jobstats"
DEFAULT_BASE="$SHARED_ROOT"
have() { command -v "$1" >/dev/null 2>&1; }

case "${1:-}" in
  -h|--help) sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 0 ;;
esac

if ! have uv; then
  echo "ERROR: 'uv' is required to build the .venv (install: https://docs.astral.sh/uv/)." >&2
  exit 1
fi

# 1) Base dir: $1, else prompt (TTY), else default.
if [ -n "${1:-}" ]; then
  BASE="$1"
elif [ -t 0 ]; then
  read -r -p "Base dir for the venv (created at <base>/.venv)? [$DEFAULT_BASE]: " ANS || ANS=""
  BASE="${ANS:-$DEFAULT_BASE}"
else
  echo "No path given and not interactive; using default base: $DEFAULT_BASE"
  BASE="$DEFAULT_BASE"
fi
BASE="${BASE/#\~/$HOME}"
case "$BASE" in /*) ;; *) BASE="$PWD/$BASE" ;; esac
VENV="$BASE/.venv"

# 2) Build/refresh the venv with the plot dependencies.
if [ -x "$VENV/bin/python" ]; then
  echo "Reusing existing virtualenv at $VENV (delete it to rebuild from scratch)."
else
  echo "Building Python 3.12 virtualenv at $VENV ..."
  uv venv --python 3.12 "$VENV"
fi
echo "Installing plotext + rich into $VENV ..."
VIRTUAL_ENV="$VENV" uv pip install "plotext>=5.3" "rich>=13"

# 3) Record it (replaces whatever venv_path.conf pointed at before) so
# jobstats_plot resolves this venv.
cat > "$VENVCONF" <<EOF
# jobstats_plot venv location. The tool runs \$venv_env_path/.venv/bin/python.
# Rewritten by setup/install_venv.sh.
venv_env_path=$BASE
EOF
echo "Recorded venv_env_path=$BASE in $VENVCONF."

# 4) Verify.
if "$VENV/bin/python" -c 'import plotext, rich' 2>/dev/null; then
  echo "OK: plotext + rich importable in $VENV."
else
  echo "WARNING: plotext/rich not importable in $VENV." >&2
fi

cat <<EOF

Done. jobstats_plot will now run under $VENV (it reads $VENVCONF and re-execs into
that venv) -- no PATH or launcher setup needed:

  ./kempner_jobstats --dcgm --ts --csv JOBID | ./plot_util/jobstats_plot --compact

(run by path, or add the repo dirs to your PATH yourself).
EOF
