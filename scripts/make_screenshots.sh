#!/usr/bin/env bash
# Regenerate the README's four sample outputs: two charts as SVG in docs/, and the two
# table views as plain text in docs/ for pasting into the README's fenced blocks.
#
# The charts are captured in colour and converted to SVG with rich (no dependency beyond
# what jobscope already needs). The two table views are NOT images: a ~124-column table
# shrunk to a README's display width is unreadable, where a fenced block renders in the
# reader's own font at full size. They are emitted as text for that reason, and the
# README inlines the contents -- so after running this, diff docs/*.txt against the
# blocks under "One job" and "Your finished jobs in a partition" and paste in what moved.
#
# Requires jobscope installed (or on PATH) and a configured Prometheus endpoint
# (JOBSCOPE_PROM_URL or a jobscope config), since all four use the DCGM views.
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
#   ONE_JOB      job id for the single-job table capture (default: $TS_JOB). Worth a
#                different job: the interesting table is a mediocre job, where the
#                interesting chart is a busy one.
#   AGG_SELECT   selector for the aggregated bars (default: -N 15)
#   AGG_USER     user for the aggregated bars (optional; kept out of the title)
#   SUM_SELECT   selector for the summary-table capture (default: -p kempner_h100 -D 1)
#   SUM_USER     user for it (optional). Pick someone with a spread of efficiencies:
#                the capture is there to show a wasteful job beside a busy one, and a
#                uniformly busy selection demonstrates nothing.
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

# The piped form, captioned as the pipe. --plot_ts renders the same series as one
# panel per metric; this is the overlaid single-panel chart, which is the more
# readable picture for a screenshot and the one the README shows. The caption is the
# command verbatim either way -- a caption you cannot paste is worse than none.
echo "[1/4] time series  (job $TS_JOB)"
TS_CMD=(-j "$TS_JOB" "${TS_NODE_ARGS[@]}" "${TS_GPU_ARGS[@]}" --ts --csv)
"$JOBSCOPE" "${TS_CMD[@]}" \
  | "$JOBSCOPE" plot --width 90 --height 18 \
  | ansi2svg docs/timeseries.svg "jobscope ${TS_CMD[*]} | jobscope plot"

echo "[2/4] aggregated bars  ($AGG_SELECT)"
"$JOBSCOPE" "${AGG_USER_ARGS[@]}" --gpu $AGG_SELECT --csv \
  | "$JOBSCOPE" plot --kind bars \
  | ansi2svg docs/aggregated.svg "jobscope --gpu $AGG_SELECT --csv | jobscope plot --kind bars"

# The two table views, as text for the README's fenced blocks rather than as images.
# NO_COLOR beats the FORCE_COLOR exported above (cli._want_color tests it first), which
# is what keeps escape codes out of a block that has to paste as plain text. `mask` keeps
# real usernames out of the docs, padded so the columns still line up.
mask() { sed -e "s/$1/alice /g"; }
# Trailing padding is invisible in a terminal but shows up as diff noise in a text file.
text() { sed -e 's/[[:space:]]*$//' > "$1"; echo "  wrote $1"; }

# The job's OWNER, not whoever is running this: masking $(id -un) leaked the owner of
# a job belonging to someone else, which is exactly the case a capture of a
# real job is likely to be.
ONE_USER="${ONE_USER:-$(sacct -X -j "$ONE_JOB" -o User -n 2>/dev/null | tr -d ' ' | head -1)}"
[ -n "$ONE_USER" ] || { echo "cannot determine the owner of job $ONE_JOB; set ONE_USER" >&2; exit 1; }

echo "[3/4] one job  (job $ONE_JOB, owner masked)"
NO_COLOR=1 "$JOBSCOPE" -j "$ONE_JOB" \
  | mask "$ONE_USER" \
  | text docs/onejob.txt

echo "[4/4] partition summary  ($SUM_SELECT)"
# Trimmed to a spread of rows: the full selection runs to tens of jobs, where what this
# has to show is one wasteful job beside a busy one.
NO_COLOR=1 "$JOBSCOPE" finished $SUM_SELECT "${SUM_USER_ARGS[@]}" \
  | mask "${SUM_USER:-$(id -un)}" \
  | "$PYBIN" "$ROOT/scripts/trim_rows.py" --column 8 --keep 6 \
  | text docs/summary.txt
# The README's heading for this block deliberately omits -u $SUM_USER: it documents the
# per-user default, and naming a real account is what `mask` above exists to prevent.

echo "done."
