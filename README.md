# kempner-jobstats

`kempner_jobstats` is a read-only Slurm job-efficiency scanner for completed Kempner jobs. It reads existing jobstats data from `sacct`, adds DCGM GPU metrics from Prometheus when available, and emits concise tables or CSV for `jobstats_plot`.

Example Usage:
```bash
kempner_jobstats -D 3  # last 3 days of GPU job metrics
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot # Plot the JOBID GPU metrics
```

## Screenshots

<table>
  <tr>
    <td><strong>Per-job DCGM time series</strong></td>
    <td><strong>Aggregated utilization across jobs</strong></td>
  </tr>
  <tr>
    <td><img src="docs/timeseries.svg" alt="per-job DCGM time series" width="420"></td>
    <td><img src="docs/aggregated.svg" alt="aggregated mean-utilization bars" width="420"></td>
  </tr>
</table>

Regenerate these with `bash plot_util/setup/make_screenshots.sh`.

## Setup

```bash
git clone https://github.com/KempnerInstitute/kempner-jobstats
cd kempner-jobstats
# run by path (no setup needed):
./kempner_jobstats --dcgm --ts --csv JOBID | ./plot_util/jobstats_plot --compact
# or add both to your PATH so the examples below work by name:
export PATH="$PWD:$PWD/plot_util:$PATH"
```

No install step: `kempner_jobstats` runs on the system `python3`, and
`jobstats_plot` finds its plotting venv automatically -- it reads
`plot_util/venv_path.conf` and re-execs under that venv (default: the shared
cluster `.venv`). The examples below write the bare names `kempner_jobstats` and
`jobstats_plot`; those resolve once the repo and `plot_util/` are on your `$PATH`
(the `export` line above), otherwise prefix them with `./` and `./plot_util/`. To
build your own venv, and for plot dependencies and cluster requirements, see
[`plot_util/setup/README.md`](plot_util/setup/README.md).

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
- For live monitoring, use [KempnerPulse](https://github.com/KempnerInstitute/kempnerpulse)
