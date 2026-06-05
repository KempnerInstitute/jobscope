# kempner-jobstats

`jobstats_history` is a command-line tool for reviewing Slurm job efficiency of
CPU and GPU devices after a job finishes. Scan many jobs quickly with the summary
views (`--gpu`/`--cpu`/`--cgpu`), then drop into `--dcgm` for the detailed per-GPU
DCGM profiling metrics.

It uses the same jobstats data sources, so its numbers line up with `jobstats`.
For live monitoring, use
[KempnerPulse](https://github.com/KempnerInstitute/kempnerpulse).

## Setup

Update your local checkout before running the tools:

```bash
git clone https://github.com/KempnerInstitute/kempner-jobstats
cd kempner-jobstats
```
If you don't want to install, set up the path
```bash
export PATH=$PATH:/n/holylfs06/LABS/kempner_shared/Everyone/cluster_scripts/job_eff/kempner-jobstats
```
## Quick Start

Start with a GPU summary for recent jobs:

```bash
./jobstats_history -D 5
```

`--gpu` is the default, so this shows GPU jobs from the last 5 days with the
DCGM utilization columns included automatically. For a fast, offline overview of
all jobs (CPU + GPU blob columns, no Prometheus), use `--cgpu`.

Then inspect one job in detail (per-GPU DCGM metrics):

```bash
./jobstats_history --dcgm --ext 17487044
```

Add a simple advisory label:

```bash
./jobstats_history --gpu --diagnose -D 5
```

## Which Tool Should I Use?

| Need | Command |
|---|---|
| See recent GPU jobs with real activity metrics (default) | `./jobstats_history -D 5` |
| Fast, offline overview of all jobs (CPU + GPU, no DCGM) | `./jobstats_history --cgpu -D 5` |
| CPU-only jobs/columns | `./jobstats_history --cpu -D 5` |
| Grade GPU jobs with a `DIAG` tag | `./jobstats_history --diagnose -D 5` |
| Inspect jobs per GPU (full DCGM catalog) | `./jobstats_history --dcgm --ext JOBID` |
| Export raw GPU time series | `./jobstats_history --dcgm --ts JOBID > job.csv` |

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

## `jobstats_history`

Use this to scan jobs in bulk. It reads each job's stored `sacct`
`AdminComment` jobstats blob in one query.

Common commands:

```bash
./jobstats_history                         # your GPU jobs from the last 1 day (default --gpu, DCGM included)
./jobstats_history -D 7                    # your GPU jobs from the last 7 days
./jobstats_history -N 20                   # GPU jobs among your 20 most recent
./jobstats_history -A kempner_dev -D 7     # account jobs from the last 7 days
./jobstats_history -p kempner_h100 -D 7    # partition jobs from the last 7 days
./jobstats_history -t failed -D 7          # failed jobs from the last 7 days
./jobstats_history --cgpu -D 7             # all jobs, CPU+GPU columns, fast & offline (no DCGM)
./jobstats_history --cpu -D 7              # CPU columns only, offline
./jobstats_history --diagnose -D 5         # GPU jobs with DIAG labels
./jobstats_history -d JOBID                # per-node / per-GPU breakdown
./jobstats_history --csv -D 7 > jobs.csv   # CSV output
./jobstats_history --dcgm JOBID            # per-GPU DCGM table (default 6 metrics)
./jobstats_history --dcgm --ext -D 1       # per-GPU table, full 28-metric catalog
./jobstats_history --dcgm --ts JOBID > ts.csv  # raw per-scrape time series
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
./jobstats_history --help
```

### Per-GPU DCGM details (`--dcgm`)

Use this after the summary points to a job worth investigating. `--dcgm` switches
to a per-GPU table (one row per GPU) of time-averaged DCGM profiling metrics. It
follows the same job selection as the rest of `jobstats_history` (a JOBID, `-N`,
`-D`, `-S/-E`, `-A`, `-p`, ...), not just explicit job IDs. The one exception is
`--ts`, which profiles a single job and requires exactly one JOBID.

```bash
./jobstats_history --dcgm JOBID1 JOBID2     # default 6 metrics, per GPU
./jobstats_history --dcgm --ext JOBID       # full 28-metric catalog
./jobstats_history --dcgm -D 1              # every GPU job from the last day
./jobstats_history --dcgm --csv JOBID > out.csv  # one row per job/GPU
./jobstats_history --dcgm --ts JOBID > ts.csv    # raw time series (single job only)
./jobstats_history --dcgm --describe --ext  # explain all metrics
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

`jobstats_plot` turns any `jobstats_history --csv` output into a terminal graph. It
is a **separate, optional** tool: `jobstats_history` stays dependency-free and is not
affected by it. Pipe the CSV in (without `-n`, so the header is included):

```bash
./jobstats_history --gpu  --csv JOBID       | ./jobstats_plot   # bar gauges (one job)
./jobstats_history --gpu  --csv -D 7        | ./jobstats_plot   # GPU% histogram (many jobs)
./jobstats_history --dcgm --csv -D 7        | ./jobstats_plot   # heatmap (jobs x metrics)
./jobstats_history --dcgm --ts --csv JOBID  | ./jobstats_plot   # time-series line + sparklines
./jobstats_plot -f saved.csv --kind heat                        # from a saved CSV file
```

The chart type is auto-detected from the columns; override with `--kind
auto|bars|heat|hist|line`. Other options: `--metric NAME` (histogram metric / line
metrics), `--all` (line: every metric), `--width`/`--height`, `--max-rows` (heatmap
cap), `--no-color` (also honors `$NO_COLOR`), `--config` (pull color thresholds from
the jobstats config). `--help` for the full list.

**Requirements:** `jobstats_plot` needs Python 3.12 with `plotext` and `rich`:

```bash
/usr/bin/python3.12 -m pip install --user plotext rich
```

(A Singularity container bundling these is a planned secondary option — it only needs
`plotext`/`rich` and reads the CSV on stdin, so it requires no Slurm/Prometheus access.)

## Requirements

- Run this tool on a system where `jobstats` is installed.
- `jobstats_history` in the `--cpu` / `--cgpu` views only needs `sacct`.
- `jobstats_history` (default `--gpu`), `--diagnose`, and `--dcgm` query the
  jobstats Prometheus endpoint.

## References

- [FASRC jobstats documentation](https://docs.rc.fas.harvard.edu/kb/jobstats/)
- [Princeton jobstats](https://princetonuniversity.github.io/jobstats/)
- [KempnerPulse for live monitoring](https://github.com/KempnerInstitute/kempnerpulse)


