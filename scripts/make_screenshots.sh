#!/usr/bin/env bash
# Regenerate the README screenshots in docs/ as SVG (four of them).
#
# Captures each plot's colored terminal output and converts it to SVG with rich
# (no dependency beyond what jobscope already needs). Requires jobscope installed
# (or on PATH) and a configured Prometheus endpoint (JOBSCOPE_PROM_URL or a
# jobscope config), since both screenshots use the DCGM views.
#
# Environment overrides:
#   JOBSCOPE     jobscope executable (default: jobscope)
#   PYBIN        python for ansi2svg (default: python3)
#   TS_JOB       GPU job id for the time-series screenshot (required)
#   TS_USER      its owner, masked out of the shot (default: read from sacct)
#   AGG_SELECT   selector for the aggregated bars (default: -N 15)
#   AGG_USER     user for the aggregated bars (optional; kept out of the title)
#   SUM_SELECT   selector for the summary-table shot (default: -p kempner_h100 -D 2)
#   SUM_USER     user for it (optional). Pick someone with a spread of efficiencies:
#                the shot is there to show red beside green, and a uniformly busy
#                selection demonstrates nothing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p docs

JOBSCOPE="${JOBSCOPE:-jobscope}"
PYBIN="${PYBIN:-python3}"
ansi2svg() { "$PYBIN" "$ROOT/scripts/ansi2svg.py" "$1" "$2"; }

export FORCE_COLOR=1
TS_JOB="${TS_JOB:?set TS_JOB to a GPU job id for the time-series screenshot}"
AGG_SELECT="${AGG_SELECT:--N 15}"
AGG_USER_ARGS=()
[ -n "${AGG_USER:-}" ] && AGG_USER_ARGS=(-u "$AGG_USER")
SUM_SELECT="${SUM_SELECT:--p kempner_h100 -D 2}"
SUM_USER_ARGS=()
[ -n "${SUM_USER:-}" ] && SUM_USER_ARGS=(-u "$SUM_USER")

echo "[1/4] time series  (job $TS_JOB)"
"$JOBSCOPE" -j "$TS_JOB" --ts --csv \
  | "$JOBSCOPE" plot --width 90 --height 18 \
  | ansi2svg docs/timeseries.svg "jobscope -j $TS_JOB --ts --csv | jobscope plot"

echo "[2/4] aggregated bars  ($AGG_SELECT)"
"$JOBSCOPE" "${AGG_USER_ARGS[@]}" --gpu $AGG_SELECT --csv \
  | "$JOBSCOPE" plot --kind bars \
  | ansi2svg docs/aggregated.svg "jobscope --gpu $AGG_SELECT --csv | jobscope plot --kind bars"

# The two table screenshots. FORCE_COLOR is what makes these coloured at all: the
# report tints only a tty, and here stdout is a pipe. `mask` keeps real usernames out
# of the docs, padded so the columns still line up.
mask() { sed -e "s/$1/alice /g"; }

# The job's OWNER, not whoever is running this: masking $(id -un) leaked the owner of
# a job belonging to someone else, which is exactly the case a screenshot of a
# real job is likely to be.
TS_USER="${TS_USER:-$(sacct -X -j "$TS_JOB" -o User -n 2>/dev/null | tr -d ' ' | head -1)}"
[ -n "$TS_USER" ] || { echo "cannot determine the owner of job $TS_JOB; set TS_USER" >&2; exit 1; }

echo "[3/4] one job  (job $TS_JOB, owner masked)"
"$JOBSCOPE" -j "$TS_JOB" \
  | mask "$TS_USER" \
  | grep -v "use each metric\|(wasteful/red/yellow)\|^  USED is\|^  bands catch" \
  | ansi2svg docs/onejob.svg "jobscope -j $TS_JOB"

echo "[4/4] partition summary  ($SUM_SELECT)"
# Trimmed to a spread of rows: the full selection is tens of jobs and 2700px tall,
# where what the shot has to show is red beside green.
"$JOBSCOPE" finished $SUM_SELECT "${SUM_USER_ARGS[@]}" \
  | mask "${SUM_USER:-$(id -un)}" \
  | "$PYBIN" "$ROOT/scripts/trim_rows.py" --column 8 --keep 6 \
  | ansi2svg docs/summary.svg "jobscope finished $SUM_SELECT"

echo "done."
