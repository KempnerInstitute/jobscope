# kempner-jobstats

Two command-line tools for reviewing Slurm job efficiency after a job finishes.

- `jobstats_history`: scan many jobs quickly.
- `jobstats_dcgm`: inspect detailed GPU metrics for specific job IDs.

Both tools use the same jobstats data sources, so their numbers should line up
with `jobstats`.

## Quick Start

Start with a GPU summary for recent jobs:

```bash
./jobstats_history --gpu --dcgm -D 5
```

This shows GPU jobs from the last 5 days and adds DCGM utilization columns.

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
| See recent jobs | `./jobstats_history` |
| See recent GPU jobs | `./jobstats_history --gpu -D 5` |
| Add real GPU activity metrics | `./jobstats_history --gpu --dcgm -D 5` |
| Grade GPU jobs with a `DIAG` tag | `./jobstats_history --gpu --diagnose -D 5` |
| Inspect one job per GPU | `./jobstats_dcgm JOBID` |
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
./jobstats_history                         # your jobs from the last 1 day
./jobstats_history -D 7                    # your jobs from the last 7 days
./jobstats_history -N 20                   # your 20 most recent jobs
./jobstats_history -A kempner_dev -D 7     # account jobs from the last 7 days
./jobstats_history -p kempner_h100 -D 7    # partition jobs from the last 7 days
./jobstats_history -t failed -D 7          # failed jobs from the last 7 days
./jobstats_history --gpu --dcgm -D 5       # GPU jobs with DCGM columns
./jobstats_history --gpu --diagnose -D 5   # GPU jobs with DIAG labels
./jobstats_history -d --gpu JOBID          # per-node / per-GPU breakdown
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
| `--cpu`, `--gpu`, `--full` | Choose output columns. `--full` is the default. |
| `--dcgm` | Add `SM_ACT%`, `OCC%`, `TENSOR%`, `DRAM%`, and `POWER_W`. |
| `--diagnose` | Add an advisory `DIAG` label. |
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
./jobstats_dcgm JOBID                 # default GPU metrics
./jobstats_dcgm JOBID1 JOBID2         # several jobs
./jobstats_dcgm --diagnose JOBID      # add DIAG labels
./jobstats_dcgm --all JOBID           # all available metrics
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
- `jobstats_history` without `--dcgm` or `--diagnose` only needs `sacct`.
- `jobstats_history --dcgm`, `jobstats_history --diagnose`, and
  `jobstats_dcgm` query the jobstats Prometheus endpoint.
- `jobstats_dcgm` requires at least one job ID unless you use `--describe`.

## References

- FASRC jobstats documentation:
  <https://docs.rc.fas.harvard.edu/kb/jobstats/>
- Princeton jobstats:
  <https://princetonuniversity.github.io/jobstats/>
- KempnerPulse for live monitoring:
  <https://github.com/KempnerInstitute/kempnerpulse>


