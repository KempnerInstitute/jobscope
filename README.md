# kempner-jobstats

`kempner_jobstats` is a command-line tool for reviewing Slurm job efficiency of
CPU and GPU devices after a job finishes. Scan many jobs quickly with the summary
views (`--gpu`/`--cpu`/`--cgpu`), then drop into `--dcgm` for the detailed per-GPU
DCGM profiling metrics.

It uses the same jobstats data sources, so its numbers line up with `jobstats`.
For live monitoring, use
[KempnerPulse](https://github.com/KempnerInstitute/kempnerpulse).

## Screenshots

Per-job time series — `kempner_jobstats JOBID --dcgm --csv --ts | jobstats_plot`:

![per-job DCGM time series](docs/timeseries.svg)

Aggregated utilization (mean across jobs) — `kempner_jobstats -D3 --csv | jobstats_plot --kind bars`:

![aggregated mean-utilization bars](docs/aggregated.svg)

(Regenerate with `bash setup/make_screenshots.sh`.)

## Setup

```bash
git clone https://github.com/KempnerInstitute/kempner-jobstats
cd kempner-jobstats
source setup/env.sh          # add kempner_jobstats + jobstats_plot to $PATH (this shell)
```

`source setup/env.sh` puts the repo root and `plot_util/` on `$PATH`, so you can run
`kempner_jobstats` and `jobstats_plot` by name. For a permanent setup, symlinks, or an
Lmod module, run `bash setup/install.sh`. The optional `jobstats_plot` also needs
Python 3.12 + plotext + rich — see
[Installing the plot dependencies](#installing-the-plot-dependencies).

On the cluster you can use the shared deploy instead of cloning:

```bash
source /n/holylfs06/LABS/kempner_shared/Everyone/cluster_scripts/job_eff/kempner-jobstats/setup/env.sh
```
## Quick Start

Start with a GPU summary for recent jobs:

```bash
kempner_jobstats -D 5
```

`--gpu` is the default, so this shows GPU jobs from the last 5 days with the
DCGM utilization columns included automatically. For a fast, offline overview of
all jobs (CPU + GPU blob columns, no Prometheus), use `--cgpu`.

Then inspect one job in detail (per-GPU DCGM metrics):

```bash
kempner_jobstats --dcgm --ext 17487044
```

Add a simple advisory label:

```bash
kempner_jobstats --gpu --diagnose -D 5
```

## Which Tool Should I Use?

| Need | Command |
|---|---|
| See recent GPU jobs with real activity metrics (default) | `kempner_jobstats -D 5` |
| Fast, offline overview of all jobs (CPU + GPU, no DCGM) | `kempner_jobstats --cgpu -D 5` |
| CPU-only jobs/columns | `kempner_jobstats --cpu -D 5` |
| Grade GPU jobs with a `DIAG` tag | `kempner_jobstats --diagnose -D 5` |
| Inspect jobs per GPU (full DCGM catalog) | `kempner_jobstats --dcgm --ext JOBID` |
| Export raw GPU time series | `kempner_jobstats --dcgm --ts JOBID > job.csv` |

## Reading GPU Metrics

`GPU%` alone can be misleading. It means a kernel was running, not that the GPU
was doing useful work.

The most useful columns are:

| Column | Meaning |
|---|---|
| `GPU%` / `DUTY%` | Time when at least one GPU kernel was running. Coarse signal. |
| `SM_ACT%` | How much the GPU cores were active. Best quick efficiency signal. |
| `OCC%` | How full the cores were. Low values often mean small kernels or weak parallelism. |
| `TENSOR%` | Tensor-core activity. Near zero on ML training often means no mixed precision. |
| `DRAM%` | HBM memory-bandwidth activity. High values can indicate a memory-bound job. |
| `POWER_W` | Mean board power. Low power usually confirms the GPU was mostly idle. |

Common patterns:

| Pattern | What it looks like |
|---|---|
| Idle | Low `GPU%`, low `SM_ACT%`, low `POWER_W` |
| Underfed | High `GPU%`, low `SM_ACT%`, low `OCC%` |
| Low occupancy | High `SM_ACT%`, low `OCC%` |
| Memory bound | High `DRAM%` relative to compute columns |
| No tensor cores | High `SM_ACT%`, near-zero `TENSOR%` on an ML workload |

## `kempner_jobstats`

Use this to scan jobs in bulk. It reads each job's stored `sacct`
`AdminComment` jobstats blob in one query.

Common commands:

```bash
kempner_jobstats                         # your GPU jobs from the last 1 day (default --gpu, DCGM included)
kempner_jobstats -D 7                    # your GPU jobs from the last 7 days
kempner_jobstats -N 20                   # GPU jobs among your 20 most recent
kempner_jobstats -A kempner_dev -D 7     # account jobs from the last 7 days
kempner_jobstats -p kempner_h100 -D 7    # partition jobs from the last 7 days
kempner_jobstats -t failed -D 7          # failed jobs from the last 7 days
kempner_jobstats --cgpu -D 7             # all jobs, CPU+GPU columns, fast & offline (no DCGM)
kempner_jobstats --cpu -D 7              # CPU columns only, offline
kempner_jobstats --diagnose -D 5         # GPU jobs with DIAG labels
kempner_jobstats -d JOBID                # per-node / per-GPU breakdown
kempner_jobstats --csv -D 7 > jobs.csv   # CSV output
kempner_jobstats --dcgm JOBID            # per-GPU DCGM table (default 6 metrics)
kempner_jobstats --dcgm --ext -D 1       # per-GPU table, full 28-metric catalog
kempner_jobstats --dcgm --ts JOBID > ts.csv  # raw per-scrape time series
```

Useful options:

| Option | Meaning |
|---|---|
| `JOBID ...` | Report specific job IDs instead of selecting by time. |
| `-u USER` | Select another user's jobs. |
| `-A ACCOUNT` | Filter by Slurm account. |
| `-p PARTITION` | Filter by partition. |
| `-t STATE` | Filter by state: `all`, `completed`, or `failed`. |
| `-D DAYS` | Select jobs from the last N days. |
| `-N N` | Select the most recent N jobs. |
| `-S TIME -E TIME` | Select an explicit time window. |
| `--cpu`, `--gpu`, `--cgpu` | Choose output columns. `--gpu` is the default and adds the DCGM columns `SM_ACT%`, `OCC%`, `TENSOR%`, `DRAM%`, and `POWER_W` from Prometheus (GPU jobs only). `--cpu` (CPU only) and `--cgpu` (CPU + GPU) are blob-only and offline. |
| `--diagnose` | Add an advisory `DIAG` label (implies `--gpu`). |
| `--dcgm` | Per-GPU DCGM profiling table (one row per GPU). Its own view; overrides `--cpu/--gpu/--cgpu` and `-d`. See below. |
| `--ext` | With `--dcgm`, show the full 28-metric catalog instead of the default 6. |
| `--ts` | With `--dcgm`, emit the raw per-scrape time series as CSV for a single job (pass exactly one JOBID). |
| `-d`, `--details` | Show per-node / per-GPU details. |
| `--csv` | Print machine-readable CSV. |
| `--describe` | Explain output columns and exit (add `--dcgm`/`--ext` for the DCGM catalog). |

Run the full help at any time:

```bash
kempner_jobstats --help
```

### Per-GPU DCGM details (`--dcgm`)

Use this after the summary points to a job worth investigating. `--dcgm` switches
to a per-GPU table (one row per GPU) of time-averaged DCGM profiling metrics. It
follows the same job selection as the rest of `kempner_jobstats` (a JOBID, `-N`,
`-D`, `-S/-E`, `-A`, `-p`, ...), not just explicit job IDs. The one exception is
`--ts`, which profiles a single job and requires exactly one JOBID.

```bash
kempner_jobstats --dcgm JOBID1 JOBID2     # default 6 metrics, per GPU
kempner_jobstats --dcgm --ext JOBID       # full 28-metric catalog
kempner_jobstats --dcgm -D 1              # every GPU job from the last day
kempner_jobstats --dcgm --csv JOBID > out.csv  # one row per job/GPU
kempner_jobstats --dcgm --ts JOBID > ts.csv    # raw time series (single job only)
kempner_jobstats --dcgm --describe --ext  # explain all metrics
```

Default columns:

| Column | Meaning |
|---|---|
| `DUTY%` | Same coarse duty-cycle signal as `GPU%`. |
| `SM_ACT%` | SM/core activity. |
| `OCC%` | SM occupancy. |
| `TENSOR%` | Tensor-core activity. |
| `DRAM%` | HBM bandwidth activity. |
| `POWER_W` | Mean board power. |

`--ext` adds the full catalog (ENGINE/HMMA/IMMA/DFMA/FP16/FP32/FP64/MEMCP/PWRmax/
ENERGY/FB_*/PCIE_*/NVLINK/clocks/temps/ENC/DEC).

## Plotting (optional): `jobstats_plot`

`jobstats_plot` turns any `kempner_jobstats --csv` output into a terminal graph. It
is a **separate, optional** tool: `kempner_jobstats` stays dependency-free and is not
affected by it. Pipe the CSV in (without `-n`, so the header is included):

```bash
kempner_jobstats --gpu  --csv JOBID       | jobstats_plot   # bar gauges (one job)
kempner_jobstats --gpu  --csv -D 7        | jobstats_plot   # GPU% histogram (many jobs)
kempner_jobstats --dcgm --csv -D 7        | jobstats_plot   # heatmap (jobs x metrics)
kempner_jobstats --dcgm --ts --csv JOBID  | jobstats_plot   # time-series line + per-metric stats
jobstats_plot -f saved.csv --kind heat                        # from a saved CSV file
```

The chart type is auto-detected from the columns; override with `--kind
auto|bars|heat|hist|line`. Other options: `--metric NAME` (histogram metric / line
metrics), `--by metric|gpu` (line faceting), `--compact` (line: one sparkline row
per metric, own scale), `--all` (line: every metric),
`--marker braille|dot|hd|fhd` (line style; braille = thin, the default),
`--width`/`--height`, `--max-rows` (heatmap cap), `--no-color` (also honors
`$NO_COLOR`), `--config` (pull color thresholds from the jobstats config).
`--help` for the full list.

By default `--dcgm --ts` draws **one panel per GPU** with all metrics on a shared
axis (`--by gpu`). Use `--by metric` to give **each metric its own panel and
y-axis** (so metrics in similar ranges, e.g. OCC% vs SM_ACT%, don't overlap; with
multiple GPUs each metric panel draws a line per GPU). A **multi-node** job facets
by node (one panel per node, a line per GPU, single metric). Narrow large jobs with
`--node NODE` (drill into one node) and/or `--gpu N` — these filters also apply to
the heatmap — or a single `--metric`. Stacked panels are capped (a note tells you
when). For a dense overview, `--compact` collapses each metric to a single braille
sparkline row (its own scale + min-max range) instead of full-height panels.

### Installing the plot dependencies

`jobstats_plot` needs Python 3.12 with `plotext` and `rich`. `kempner_jobstats`
itself needs none of this and runs on the system Python. Pick whichever install
fits you (in rough order of convenience):

**1. uv (recommended).** `jobstats_plot` carries inline [PEP 723] metadata, so
[uv](https://docs.astral.sh/uv/) auto-provisions Python 3.12 + plotext + rich
(cached after the first run) — no manual setup:

```bash
kempner_jobstats --dcgm --ts --csv JOBID | uv run --script plot_util/jobstats_plot --compact
```

**2. uv shared venv** (one install for all users; point the deploy at it):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv plotext rich
kempner_jobstats ... --csv | ./.venv/bin/python plot_util/jobstats_plot --compact
```

**3. pip --user** (quick, per-user):

```bash
/usr/bin/python3.12 -m pip install --user plotext rich
kempner_jobstats ... --csv | jobstats_plot
```

**4. Singularity container** (no host Python/uv needed; portable). Build once
(`setup/jobstats_plot.def` is in the repo), then pipe CSV into it — it needs no
Slurm/Prometheus/config inside:

```bash
# from the repo root:
singularity build --fakeroot jobstats_plot.sif setup/jobstats_plot.def
kempner_jobstats --dcgm --ts --csv JOBID | singularity run jobstats_plot.sif --compact
```

[PEP 723]: https://peps.python.org/pep-0723/

## Requirements

- Run this tool on a system where `jobstats` is installed.
- `kempner_jobstats` in the `--cpu` / `--cgpu` views only needs `sacct`.
- `kempner_jobstats` (default `--gpu`), `--diagnose`, and `--dcgm` query the
  jobstats Prometheus endpoint.

## References

- [FASRC jobstats documentation](https://docs.rc.fas.harvard.edu/kb/jobstats/)
- [Princeton jobstats](https://princetonuniversity.github.io/jobstats/)
- [KempnerPulse for live monitoring](https://github.com/KempnerInstitute/kempnerpulse)


