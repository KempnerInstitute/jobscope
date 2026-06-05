# kempner-jobstats

`kempner_jobstats` is a read-only command-line tool for reviewing completed Slurm
jobs on the Kempner cluster. It scans many jobs quickly from stored jobstats data,
adds live DCGM GPU profiling where available, and can export CSV for terminal
plots.

For live monitoring, use
[KempnerPulse](https://github.com/KempnerInstitute/kempnerpulse).

## What It Does

| Need | Command |
|---|---|
| Recent GPU jobs with DCGM activity metrics | `kempner_jobstats -D 5` |
| Fast offline CPU + GPU summary | `kempner_jobstats --cgpu -D 5` |
| CPU-only summary | `kempner_jobstats --cpu -D 5` |
| Advisory GPU diagnosis labels | `kempner_jobstats --diagnose -D 5` |
| Per-node / per-GPU breakdown | `kempner_jobstats -d JOBID` |
| Per-GPU DCGM profiling table | `kempner_jobstats --dcgm --ext JOBID` |
| Raw DCGM time series | `kempner_jobstats --dcgm --ts --csv JOBID > ts.csv` |
| Terminal plots from CSV | `kempner_jobstats --csv -D 7 > jobs.csv` then `jobstats_plot -f jobs.csv` |

## Setup

```bash
git clone https://github.com/KempnerInstitute/kempner-jobstats
cd kempner-jobstats
source setup/env.sh
```

`source setup/env.sh` adds `kempner_jobstats` and `jobstats_plot` to your `$PATH`
for the current shell. For the shared cluster deploy, permanent PATH setup,
plotting dependencies, containers, and cluster requirements, see
[`setup/README.md`](setup/README.md).

## Quick Start

Scan recent GPU jobs:

```bash
kempner_jobstats -D 5
```

Inspect one job with the full per-GPU DCGM catalog:

```bash
kempner_jobstats --dcgm --ext JOBID
```

Export a time series and plot it:

```bash
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --compact
```

Common selectors work across views:

```bash
kempner_jobstats -N 20
kempner_jobstats -A kempner_dev -D 7
kempner_jobstats -p kempner_h100 -t failed -D 7
kempner_jobstats -S YYYY-MM-DD -E YYYY-MM-DD --csv
```

Run `kempner_jobstats --help` for options and `kempner_jobstats --describe` for
column definitions.

## Output Views

- `--gpu` is the default. It shows GPU jobs with jobstats blob columns plus live
  DCGM columns: `SM_ACT%`, `OCC%`, `TENSOR%`, `DRAM%`, and `POWER_W`.
- `--cgpu` and `--cpu` are blob-only, offline views that do not query Prometheus.
- `--diagnose` adds a `DIAG` label such as `idle`, `underfed`, `low-occ`,
  `mem-bound`, `no-tensor`, or `ok`.
- `--dcgm` switches to one row per GPU; add `--ext` for the full metric catalog
  or `--ts --csv JOBID` for raw per-scrape samples.
- `--csv` makes any view machine-readable and pipeable to `jobstats_plot`.

Metric shorthand:

| Column | Read as |
|---|---|
| `GPU%` / `DUTY%` | A kernel was running; useful but coarse. |
| `SM_ACT%` | GPU core activity; best quick efficiency signal. |
| `OCC%` | How full the cores were. |
| `TENSOR%` | Tensor-core activity. |
| `DRAM%` | HBM memory-bandwidth activity. |
| `POWER_W` | Mean board power. |

## Plots

`jobstats_plot` renders `kempner_jobstats --csv` output as terminal charts and
auto-detects bar gauges, histograms, heatmaps, and time-series lines.

```bash
kempner_jobstats --gpu  --csv JOBID      | jobstats_plot
kempner_jobstats --gpu  --csv -D 7       | jobstats_plot
kempner_jobstats --dcgm --csv -D 7       | jobstats_plot
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --by metric
```

For plot dependencies, chart kinds, faceting, filters, and color options, see
[`plot_util/README.md`](plot_util/README.md).

## Screenshots

Per-job DCGM time series:

![per-job DCGM time series](docs/timeseries.svg)

Aggregated utilization across jobs:

![aggregated mean-utilization bars](docs/aggregated.svg)

Regenerate these with `bash setup/make_screenshots.sh`.

## References

- [FASRC jobstats documentation](https://docs.rc.fas.harvard.edu/kb/jobstats/)
- [Princeton jobstats](https://princetonuniversity.github.io/jobstats/)
