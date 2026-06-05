# kempner-jobstats

**Synopsis:** `kempner_jobstats` is a read-only Slurm job-efficiency scanner for
completed Kempner jobs. It reads existing jobstats data from `sacct`, adds DCGM
GPU metrics from Prometheus when available, and emits concise tables or CSV for
`jobstats_plot`.

Example Usage:

Scan recent GPU jobs:

```bash
kempner_jobstats -D 3  # last 3 days of DCGM GPU job metrics
```

Plot a particular job's DCGM metrics:

```bash
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --compact
```

For live monitoring, use
[KempnerPulse](https://github.com/KempnerInstitute/kempnerpulse).

## Screenshots

Per-job DCGM time series:

![per-job DCGM time series](docs/timeseries.svg)

Aggregated utilization across jobs:

![aggregated mean-utilization bars](docs/aggregated.svg)

Regenerate these with `bash setup/make_screenshots.sh`.

## Setup

```bash
git clone https://github.com/KempnerInstitute/kempner-jobstats
cd kempner-jobstats
source setup/env.sh
```

`source setup/env.sh` adds `kempner_jobstats` and `jobstats_plot` to your `$PATH`
for the current shell. For the shared cluster deploy, permanent PATH setup,
plot dependencies, containers, and cluster requirements, see
[`setup/README.md`](setup/README.md).

## Common Commands

| Need | Command |
|---|---|
| Recent GPU jobs with DCGM activity metrics | `kempner_jobstats -D 3` |
| Fast offline CPU + GPU summary | `kempner_jobstats --cgpu -D 5` |
| CPU-only summary | `kempner_jobstats --cpu -D 5` |
| Advisory GPU diagnosis labels | `kempner_jobstats --diagnose -D 5` |
| Per-node / per-GPU breakdown | `kempner_jobstats -d JOBID` |
| Per-GPU DCGM profiling table | `kempner_jobstats --dcgm --ext JOBID` |
| Raw DCGM time series CSV | `kempner_jobstats --dcgm --ts --csv JOBID > ts.csv` |
| Plot saved CSV | `jobstats_plot -f ts.csv --compact` |

Selectors such as `-N`, `-D`, `-S/-E`, `-u`, `-A`, `-p`, and `-t` work across
views. Run `kempner_jobstats --help` for options and `kempner_jobstats --describe`
for column definitions.

## Views

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

`jobstats_plot` renders `kempner_jobstats --csv` output as terminal bar gauges,
histograms, heatmaps, and time-series plots.

```bash
kempner_jobstats --gpu  --csv JOBID      | jobstats_plot
kempner_jobstats --gpu  --csv -D 7       | jobstats_plot
kempner_jobstats --dcgm --csv -D 7       | jobstats_plot
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --by metric
```

For plot dependencies, chart kinds, faceting, filters, and color options, see
[`plot_util/README.md`](plot_util/README.md).

## References

- [FASRC jobstats documentation](https://docs.rc.fas.harvard.edu/kb/jobstats/)
- [Princeton jobstats](https://princetonuniversity.github.io/jobstats/)
