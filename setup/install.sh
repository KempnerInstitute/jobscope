#!/usr/bin/env bash
# install.sh - set up kempner-jobstats: put the tools on PATH and (optionally)
# install the plotting dependencies. Non-destructive: it prints what to do and
# only edits ~/.bashrc / creates symlinks when you ask it to.
#
#   kempner_jobstats   -- the core scanner (any python3 >= 3.6; needs only sacct +
#                         the jobstats `config`; no extra packages)
#   jobstats_plot      -- optional terminal plots (python3.12 + plotext + rich)
#
# Usage:
#   bash setup/install.sh            # show options + status
#   bash setup/install.sh --path     # append `source .../setup/env.sh` to ~/.bashrc
#   bash setup/install.sh --symlink  # symlink both tools into ~/.local/bin
#   bash setup/install.sh --deps     # install plotext + rich for python3.12 (uv or pip)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLOT="$ROOT/plot_util/jobstats_plot"
CORE="$ROOT/kempner_jobstats"

have() { command -v "$1" >/dev/null 2>&1; }

show_status() {
  echo "kempner-jobstats: $ROOT"
  echo
  echo "Core tool : $CORE"
  echo "Plot tool : $PLOT  (needs python3.12 + plotext + rich)"
  echo
  echo "python3.12 : $(command -v python3.12 || echo 'NOT FOUND')"
  echo "uv         : $(command -v uv || echo 'not found (optional)')"
  if have python3.12; then
    if python3.12 -c 'import plotext, rich' 2>/dev/null; then
      echo "plot deps  : OK (plotext + rich importable on python3.12)"
    else
      echo "plot deps  : MISSING (run: bash setup/install.sh --deps)"
    fi
  fi
  cat <<EOF

Put the tools on PATH (pick one):
  source $ROOT/setup/env.sh           # this shell only
  bash setup/install.sh --path        # add that line to ~/.bashrc (permanent)
  bash setup/install.sh --symlink     # symlink into ~/.local/bin

Plot dependencies (pick one):
  uv run --script $PLOT ...           # zero install: uv reads the PEP 723 metadata
  bash setup/install.sh --deps        # install plotext + rich for python3.12
  singularity build --fakeroot jobstats_plot.sif setup/jobstats_plot.def   # container

Quick check:
  source $ROOT/setup/env.sh
  kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --compact
EOF
}

add_path() {
  local line="source \"$ROOT/setup/env.sh\""
  if grep -qsF "$ROOT/setup/env.sh" "$HOME/.bashrc"; then
    echo "~/.bashrc already sources setup/env.sh -- nothing to do."
  else
    printf '\n# kempner-jobstats tools on PATH\n%s\n' "$line" >> "$HOME/.bashrc"
    echo "Appended to ~/.bashrc: $line"
    echo "Open a new shell or run: source ~/.bashrc"
  fi
}

symlink() {
  mkdir -p "$HOME/.local/bin"
  ln -sf "$CORE" "$HOME/.local/bin/kempner_jobstats"
  ln -sf "$PLOT" "$HOME/.local/bin/jobstats_plot"
  echo "Symlinked kempner_jobstats and jobstats_plot into ~/.local/bin"
  case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) echo "Note: ~/.local/bin is not on \$PATH yet." ;; esac
}

deps() {
  if have uv; then
    echo "Installing plotext + rich with uv (python 3.12)..."
    uv pip install --python 3.12 plotext rich || uv pip install --system plotext rich
  elif have python3.12; then
    echo "Installing plotext + rich with pip --user (python3.12)..."
    python3.12 -m pip install --user plotext rich
  else
    echo "ERROR: need python3.12 (or uv) to install the plot dependencies." >&2
    exit 1
  fi
  echo "Done. Verify: python3.12 -c 'import plotext, rich'"
}

case "${1:-}" in
  --path)    add_path ;;
  --symlink) symlink ;;
  --deps)    deps ;;
  ""|--help|-h) show_status ;;
  *) echo "unknown option: $1"; echo "use: --path | --symlink | --deps | --help"; exit 2 ;;
esac
