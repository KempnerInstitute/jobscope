#!/usr/bin/env bash
# Regenerate the README screenshots in docs/ as SVG.
#
# Captures each plot's colored terminal output and converts it to SVG with rich
# (rich.Text.from_ansi -> Console(record=True).save_svg) -- no extra dependency
# beyond what jobstats_plot already needs (python3.12 + rich). Re-run after
# changing the plots, or with different JOBIDs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source setup/env.sh
mkdir -p docs

# stdin: ANSI text | $1: output .svg | $2: title  (helper keeps stdin = the pipe)
ansi2svg() { python3.12 "$ROOT/setup/ansi2svg.py" "$1" "$2"; }

TS_JOB="${TS_JOB:-19375791}"
AGG_DAYS="${AGG_DAYS:-3}"

echo "[1/2] time series  (job $TS_JOB)"
FORCE_COLOR=1 kempner_jobstats "$TS_JOB" --dcgm --csv --ts \
  | FORCE_COLOR=1 jobstats_plot \
  | ansi2svg docs/timeseries.svg "kempner_jobstats $TS_JOB --dcgm --csv --ts | jobstats_plot"

echo "[2/2] aggregated bars  (last $AGG_DAYS days)"
FORCE_COLOR=1 kempner_jobstats -D "$AGG_DAYS" --csv \
  | FORCE_COLOR=1 jobstats_plot --kind bars \
  | ansi2svg docs/aggregated.svg "kempner_jobstats -D$AGG_DAYS --csv | jobstats_plot --kind bars"

echo "done."
