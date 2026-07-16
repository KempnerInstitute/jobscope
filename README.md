# jobscope

`jobscope` reports the efficiency of completed Slurm jobs. It decodes the
utilization data Slurm already stores in each job's `sacct` AdminComment
(CPU / memory / GPU / GPU-memory), enriches GPU jobs with DCGM profiling metrics
pulled from Prometheus, and can render any view as a terminal chart.

For completed jobs the CPU/MEM/GPU/GMEM numbers match `jobstats`, because
jobscope decodes the same stored blob -- but in one bulk `sacct` query, with no
per-job calls and no job-count cap.

## Screenshots

<table width="800">
  <tr><td><strong>Per-job DCGM time series</strong></td></tr>
  <tr><td><img src="docs/timeseries.svg" alt="per-job DCGM time series" width="800"></td></tr>
  <tr><td><strong>Aggregated utilization across jobs</strong></td></tr>
  <tr><td><img src="docs/aggregated.svg" alt="aggregated mean-utilization bars" width="800"></td></tr>
</table>

```text
$ jobscope -S 2026-06-01 -E 2026-06-02
  User:      bdesinghu
  Select:    2026-06-01 .. 2026-06-02
JOBID        STATE     GPUS GPU%   GMEM%   SM_ACT%  OCC%    TENSOR%  DRAM%   POWER_W  RUNTIME      NAME
-------------------------------------------------------------------------------------------------------
17752673     COMPLETED 1    92     6       77.1     24.9    1.5      15.0    452      02:58:26     combo-pile
18074321     COMPLETED 1    98     8       49.3     5.8     1.0      6.6     181      01:37:53     vae-wt103
18074326     COMPLETED 1    93     6       76.7     24.8    1.5      14.8    455      09:28:41     combo-wt103
-------------------------------------------------------------------------------------------------------
Mean:                       94     7       67.7     18.5    1.3      12.1    363
```

## Requirements

- Python 3.9+
- Slurm with `sacct`, where the jobstats-style AdminComment blob is populated
  (needed by every view).
- For the GPU and DCGM views only: a Prometheus endpoint serving the DCGM
  (`DCGM_FI_*`) and `nvidia_gpu_*` series that jobstats scrapes. The offline
  `--cpu` / `--cgpu` views never contact it.

## Install

```bash
pip install .

# for development
pip install -e '.[dev]'
pytest
```

(A PyPI release will make `pip install jobscope` available later.)

## Configuration

The GPU and DCGM views need a Prometheus endpoint serving the DCGM and
`nvidia_gpu_*` series. (The offline `--cpu` / `--cgpu` views, and everything under
`describe` and `config`, need nothing.) Provide the endpoint one of these ways.

```bash
# Preferred: environment variable (keeps a credential out of any file)
export JOBSCOPE_PROM_URL="https://USER:TOKEN@prometheus.example.net/api/prom"
```

**Kempner AI Cluster users:** the endpoint is already installed on the cluster
under `/usr/local/bin` (the jobstats `config.py`), and jobscope reads it
**automatically** when nothing else is configured -- so you need no config file
and never handle the URL or token. Just run `jobscope`. (To point at a different
jobstats install, set `site_jobstats_config_path` in the config file below.)

On any other cluster, put your settings in that same config file
(`~/.config/jobscope/config.toml`, or wherever `$JOBSCOPE_CONFIG` points):

```bash
jobscope config --example > ~/.config/jobscope/config.toml   # then edit it
jobscope config                                              # show the path in use
```

The config file also sets the DCGM sampling period, plot color thresholds, and
default timeout / worker counts -- see `jobscope config --example` for the full,
commented template.

The Prometheus URL commonly embeds a credential: jobscope never prints it, and a
`config.toml` in a repo checkout is git-ignored. On sites already running
jobstats, `site_jobstats_config_path` reuses that install's `PROM_SERVER`, so the
secret is never copied.

## Quick start

```bash
jobscope -D 3                     # summary of your jobs over the last 3 days
jobscope --cgpu -D 2              # CPU + GPU summary, fully offline (no Prometheus)
jobscope --diagnose -D 5          # add an advisory GPU diagnosis column
jobscope detail JOBID             # per-node / per-GPU breakdown
jobscope dcgm --ext JOBID         # full per-GPU DCGM profiling table
jobscope dcgm --ts --csv JOBID | jobscope plot --compact   # time-series chart
```

`jobscope` alone is shorthand for `jobscope summary`, so the selectors below work
with or without a subcommand.

## Subcommands

| Command | Purpose |
|---|---|
| `jobscope summary` | one row per job (the default) |
| `jobscope detail` | per-node / per-GPU breakdown |
| `jobscope dcgm` | per-GPU DCGM profiling table (`--ext` for all 28 metrics, `--ts` for raw time series) |
| `jobscope plot` | render `--csv` output as a terminal chart |
| `jobscope describe` | plain-English column and metric reference (`--dcgm` for the catalog) |
| `jobscope config` | show the config path or print an example |

Selectors shared by `summary`, `detail`, and `dcgm`: `-N` (last N jobs), `-D`
(last N days), `-S`/`-E` (explicit window), `-u`/`-A`/`-p`/`-t` (user / account /
partition / state), `--csv`, and explicit `JOBID`s. With no scope at all, the
last day is used.

## Views

- `--gpu` (default) shows GPU blob columns plus time-averaged DCGM profiling
  (`SM_ACT%`/`OCC%`/`TENSOR%`/`DRAM%`/`POWER_W`) pulled from Prometheus.
- `--cgpu` and `--cpu` are offline views based only on the stored blob.
- `--diagnose` adds a short advisory GPU diagnosis label.

Run `jobscope describe` for column definitions and `jobscope describe --dcgm
--ext` for the full metric catalog.

## Plotting

`jobscope plot` renders `jobscope <view> --csv` output as terminal bar gauges,
histograms, heatmaps, and time-series line charts. The chart kind is
auto-detected from the CSV columns; override with `--kind`.

```bash
jobscope --gpu  --csv JOBID      | jobscope plot                 # bar gauges (one job)
jobscope --gpu  --csv -D 7       | jobscope plot --kind hist      # distribution (many jobs)
jobscope dcgm   --csv -D 7       | jobscope plot                 # heatmap (jobs/GPUs x metrics)
jobscope dcgm --ts --csv JOBID   | jobscope plot --compact        # time series
```

`jobscope plot` reads `summary`, `dcgm`, and `dcgm --ts` CSV; the `detail` CSV is
for machine consumption, not charts. Do not pass `-n` when piping to `jobscope
plot` -- the plot needs the CSV header row.

## Contrib

`contrib/jobstats_extended.py` is a site-specific prototype that folds DCGM
metrics into the jobstats blob itself. It depends on an upstream jobstats install
and is not part of the package -- see [`contrib/README.md`](contrib/README.md).

## References

- [FASRC jobstats documentation](https://docs.rc.fas.harvard.edu/kb/jobstats/)
- [Princeton jobstats](https://princetonuniversity.github.io/jobstats/)
- For live monitoring, use [KempnerPulse](https://github.com/KempnerInstitute/kempnerpulse)
