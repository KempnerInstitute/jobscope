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
#   TS_JOB       GPU job id for the time-series screenshot (required). Pick a busy
#                one -- a flat line at 0% shows nothing about the chart.
#   TS_NODE      its node, for --nodename (required if the job spans several)
#   TS_GPUS      which cards to chart, e.g. "0,1" (default: all of them). Each card is
#                a panel, so a 4-GPU node is 2700px of screenshot -- two is plenty to
#                show what the chart is.
#   ONE_JOB      job id for the single-job table shot (default: $TS_JOB). Worth a
#                different job: the interesting table is a mediocre job, where the
#                interesting chart is a busy one.
#   AGG_SELECT   selector for the aggregated bars (default: -N 15)
#   AGG_USER     user for the aggregated bars (optional; kept out of the title)
#   SUM_SELECT   selector for the summary-table shot (default: -p kempner_h100 -D 1)
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
TS_NODE_ARGS=()
[ -n "${TS_NODE:-}" ] && TS_NODE_ARGS=(--nodename "$TS_NODE")
TS_GPU_ARGS=()
[ -n "${TS_GPUS:-}" ] && TS_GPU_ARGS=(--gpuid "$TS_GPUS")
ONE_JOB="${ONE_JOB:-$TS_JOB}"
AGG_SELECT="${AGG_SELECT:--N 15}"
AGG_USER_ARGS=()
[ -n "${AGG_USER:-}" ] && AGG_USER_ARGS=(-u "$AGG_USER")
SUM_SELECT="${SUM_SELECT:--p kempner_h100 -D 1}"
SUM_USER_ARGS=()
[ -n "${SUM_USER:-}" ] && SUM_USER_ARGS=(-u "$SUM_USER")

# --plot_ts directly, not `--ts --csv | plot`. The two render differently -- --plot_ts
# is one panel per metric, the pipe defaults to one panel per GPU -- so capturing the
# pipe while captioning it --plot_ts would show a chart the caption cannot produce.
# The caption is the command, verbatim.
echo "[1/4] time series  (job $TS_JOB)"
TS_CMD=(-j "$TS_JOB" "${TS_NODE_ARGS[@]}" "${TS_GPU_ARGS[@]}" --plot_ts)
"$JOBSCOPE" "${TS_CMD[@]}" \
  | ansi2svg docs/timeseries.svg "jobscope ${TS_CMD[*]}"

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
ONE_USER="${ONE_USER:-$(sacct -X -j "$ONE_JOB" -o User -n 2>/dev/null | tr -d ' ' | head -1)}"
[ -n "$ONE_USER" ] || { echo "cannot determine the owner of job $ONE_JOB; set ONE_USER" >&2; exit 1; }

echo "[3/4] one job  (job $ONE_JOB, owner masked)"
"$JOBSCOPE" -j "$ONE_JOB" \
  | mask "$ONE_USER" \
  | grep -v "use each metric\|(wasteful/red/yellow)\|^  USED is\|^  bands catch" \
  | ansi2svg docs/onejob.svg "jobscope -j $ONE_JOB"

echo "[4/4] partition summary  ($SUM_SELECT)"
# Trimmed to a spread of rows: the full selection is tens of jobs and 2700px tall,
# where what the shot has to show is red beside green.
"$JOBSCOPE" finished $SUM_SELECT "${SUM_USER_ARGS[@]}" \
  | mask "${SUM_USER:-$(id -un)}" \
  | "$PYBIN" "$ROOT/scripts/trim_rows.py" --column 8 --keep 6 \
  | ansi2svg docs/summary.svg "jobscope finished $SUM_SELECT"
# The caption deliberately omits -u $SUM_USER: the README documents the per-user
# default, and naming a real account in a screenshot caption is what `mask` above
# exists to prevent.

echo "done."
