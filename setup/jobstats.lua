-- Lmod modulefile for kempner-jobstats.
-- Install: copy (or symlink) this file into a directory on your MODULEPATH, e.g.
--   mkdir -p ~/privatemodules/kempner-jobstats
--   ln -s <repo>/setup/jobstats.lua ~/privatemodules/kempner-jobstats/default.lua
--   module use ~/privatemodules           # then: module load kempner-jobstats
-- It derives the repo root from its own location (keep it under <repo>/setup/).

local self = myFileName()                                  -- .../<repo>/setup/jobstats.lua
local root = self:gsub("/setup/[^/]+$", "")

whatis("kempner-jobstats: per-job CPU/GPU efficiency from Slurm + optional terminal plots")
help([[
Adds kempner_jobstats (core scanner) and jobstats_plot (optional plots) to PATH.

  kempner_jobstats --gpu -D 5
  kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --compact

jobstats_plot needs python3.12 + plotext + rich (see <repo>/setup/install.sh),
or run it from the Singularity image (setup/jobstats_plot.def).
]])

prepend_path("PATH", root)
prepend_path("PATH", pathJoin(root, "plot_util"))
