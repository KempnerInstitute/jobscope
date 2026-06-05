# Add the kempner-jobstats tools to your PATH for this shell.
#   source setup/env.sh
# Then run `kempner_jobstats` and `jobstats_plot` by name from anywhere.
# (Add the same line to ~/.bashrc to make it permanent, or use setup/install.sh.)

# Resolve the repo root whether sourced from bash or zsh.
if [ -n "${BASH_SOURCE:-}" ]; then
  _KJ_SELF="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  _KJ_SELF="${(%):-%x}"
else
  _KJ_SELF="$0"
fi
_KJ_ROOT="$(cd "$(dirname "$_KJ_SELF")/.." && pwd)"

case ":$PATH:" in
  *":$_KJ_ROOT:"*) ;;                         # already on PATH
  *) export PATH="$_KJ_ROOT:$_KJ_ROOT/plot_util:$PATH" ;;
esac
unset _KJ_SELF _KJ_ROOT
