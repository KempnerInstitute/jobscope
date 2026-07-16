#!/usr/bin/env bash
# Regenerate the README screenshots in docs/ as SVG.
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
#   AGG_SELECT   selector for the aggregated bars (default: -N 15)
#   AGG_USER     user for the aggregated bars (optional; kept out of the title)
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

echo "[1/2] time series  (job $TS_JOB)"
"$JOBSCOPE" dcgm --ts --csv "$TS_JOB" \
  | "$JOBSCOPE" plot --width 90 --height 18 \
  | ansi2svg docs/timeseries.svg "jobscope dcgm --ts --csv $TS_JOB | jobscope plot"

echo "[2/2] aggregated bars  ($AGG_SELECT)"
"$JOBSCOPE" "${AGG_USER_ARGS[@]}" --gpu $AGG_SELECT --csv \
  | "$JOBSCOPE" plot --kind bars \
  | ansi2svg docs/aggregated.svg "jobscope --gpu $AGG_SELECT --csv | jobscope plot --kind bars"

echo "done."
