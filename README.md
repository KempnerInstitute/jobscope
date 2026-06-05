# kempner-jobstats

Two command-line tools for reviewing Slurm job efficiency of CPU and GPU devices after a job finishes.

- `jobstats_history`: scan many jobs quickly.
- `jobstats_dcgm`: inspect detailed GPU metrics for specific job IDs.

Both tools use the same jobstats data sources, so their numbers should line up
with `jobstats`.  For live monitoring, use
[KempnerPulse](https://github.com/KempnerInstitute/kempnerpulse). These two tools are excellent for extracting the CPU and GPU metrics of finished jobs.

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

Then inspect one job in detail:

```bash
./jobstats_dcgm --all 17487044
```

Add a simple advisory label:

```bash
./jobstats_history --gpu --diagnose -D 5
./jobstats_dcgm --diagnose 17487044
```

## Which Tool Should I Use?

| Need | Command |
|---|---|
| See recent GPU jobs with real activity metrics (default) | `./jobstats_history -D 5` |
| Fast, offline overview of all jobs (CPU + GPU, no DCGM) | `./jobstats_history --cgpu -D 5` |
| CPU-only jobs/columns | `./jobstats_history --cpu -D 5` |
| Grade GPU jobs with a `DIAG` tag | `./jobstats_history --diagnose -D 5` |
| Inspect one job per GPU | `./jobstats_dcgm --all JOBID` |
| Export raw GPU time series | `./jobstats_dcgm --ts JOBID > job.csv` |

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
| `-d`, `--details` | Show per-node / per-GPU details. |
| `--csv` | Print machine-readable CSV. |
| `--describe` | Explain output columns and exit. |

Run the full help at any time:

```bash
./jobstats_history --help
```

## `jobstats_dcgm`

Use this after `jobstats_history` points to a job worth investigating. It shows
time-averaged DCGM metrics per GPU.

Common commands:

```bash
./jobstats_dcgm --all JOBID           # all available GPU metrics
./jobstats_dcgm JOBID1 JOBID2         # several jobs
./jobstats_dcgm --diagnose JOBID      # add DIAG labels
./jobstats_dcgm --csv JOBID > out.csv # one row per job/GPU
./jobstats_dcgm --ts JOBID > ts.csv   # raw per-scrape time series
./jobstats_dcgm --describe --all      # explain all metrics
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

Run the full help at any time:

```bash
./jobstats_dcgm --help
```

## Requirements

- Run these tools on a system where `jobstats` is installed.
- `jobstats_history` in the `--cpu` / `--cgpu` views only needs `sacct`.
- `jobstats_history` (default `--gpu`), `jobstats_history --diagnose`, and
  `jobstats_dcgm` query the jobstats Prometheus endpoint.
- `jobstats_dcgm` requires at least one job ID unless you use `--describe`.

## References

- [FASRC jobstats documentation](https://docs.rc.fas.harvard.edu/kb/jobstats/)
- [Princeton jobstats](https://princetonuniversity.github.io/jobstats/)
- [KempnerPulse for live monitoring](https://github.com/KempnerInstitute/kempnerpulse)


