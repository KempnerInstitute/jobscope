#!/usr/bin/env bash
# Regenerate the README screenshots in docs/ as SVG.
#
# Captures each plot's colored terminal output and converts it to SVG with rich
# (rich.Text.from_ansi -> Console(record=True).save_svg) -- no extra dependency
# beyond what jobstats_plot already needs (python3.12 + rich). Re-run after
# changing the plots, or with different JOBIDs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p docs

# Run the tools straight from the repo: kempner_jobstats on its system python3,
# jobstats_plot (and ansi2svg) on the shared .venv python (plotext + rich).
PYBIN="$ROOT/.venv/bin/python"; [ -x "$PYBIN" ] || PYBIN="$(command -v python3.12)"
kempner_jobstats() { "$ROOT/kempner_jobstats" "$@"; }
jobstats_plot()    { "$PYBIN" "$ROOT/plot_util/jobstats_plot" "$@"; }
# stdin: ANSI text | $1: output .svg | $2: title  (helper keeps stdin = the pipe)
ansi2svg() { "$PYBIN" "$ROOT/plot_util/setup/ansi2svg.py" "$1" "$2"; }

export FORCE_COLOR=1   # force colored output through the non-tty pipes (above are functions)
TS_JOB="${TS_JOB:-19375791}"
AGG_DAYS="${AGG_DAYS:-3}"

echo "[1/2] time series  (job $TS_JOB)"
kempner_jobstats "$TS_JOB" --dcgm --csv --ts \
  | jobstats_plot \
  | ansi2svg docs/timeseries.svg "kempner_jobstats $TS_JOB --dcgm --csv --ts | jobstats_plot"

echo "[2/2] aggregated bars  (last $AGG_DAYS days)"
kempner_jobstats -D "$AGG_DAYS" --csv \
  | jobstats_plot --kind bars \
  | ansi2svg docs/aggregated.svg "kempner_jobstats -D$AGG_DAYS --csv | jobstats_plot --kind bars"

echo "done."
