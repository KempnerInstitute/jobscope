# kempner-jobstats

`kempner_jobstats` summarizes completed Kempner Slurm jobs. It reads existing
jobstats data from `sacct`, adds DCGM GPU metrics from Prometheus when available,
and can emit CSV for terminal plots.

## Quick Start

```bash
git clone https://github.com/KempnerInstitute/kempner-jobstats
cd kempner-jobstats

./kempner_jobstats -D 3
./kempner_jobstats --dcgm --ts --csv JOBID | ./plot_util/jobstats_plot --compact
```

`kempner_jobstats` does not need an install step. Run it by path, or add the repo
and plotting directory to your `PATH`:

```bash
export PATH="$PWD:$PWD/plot_util:$PATH"
```

For plotting dependencies, including how to use your own venv, see
[`plot_util/README.md`](plot_util/README.md).

## Common Commands

| Need | Command |
|---|---|
| Recent GPU jobs | `kempner_jobstats -D 3` |
| CPU + GPU summary without Prometheus | `kempner_jobstats --cgpu -D 5` |
| CPU-only summary | `kempner_jobstats --cpu -D 5` |
| GPU diagnosis labels | `kempner_jobstats --diagnose -D 5` |
| Per-node / per-GPU detail | `kempner_jobstats -d JOBID` |
| Per-GPU DCGM table | `kempner_jobstats --dcgm --ext JOBID` |
| Raw DCGM time series CSV | `kempner_jobstats --dcgm --ts --csv JOBID > ts.csv` |
| Plot saved CSV | `jobstats_plot -f ts.csv --compact` |

Selectors such as `-N`, `-D`, `-S/-E`, `-u`, `-A`, `-p`, and `-t` work across
views. Run `kempner_jobstats --help` for all options and
`kempner_jobstats --describe` for column definitions.

## Views

- `--gpu` is the default view for GPU jobs. It includes DCGM metrics when
  Prometheus data is available.
- `--cgpu` and `--cpu` are offline views based only on stored jobstats data.
- `--diagnose` adds a short GPU diagnosis label.
- `--dcgm` shows one row per GPU. Add `--ext` for more metrics or `--ts --csv`
  for raw time-series samples.
- `--csv` makes any view machine-readable and pipeable to `jobstats_plot`.

## Plots

`jobstats_plot` renders `kempner_jobstats --csv` output as terminal bars,
histograms, heatmaps, and time-series plots.

```bash
kempner_jobstats --gpu  --csv JOBID      | jobstats_plot
kempner_jobstats --gpu  --csv -D 7       | jobstats_plot
kempner_jobstats --dcgm --csv -D 7       | jobstats_plot
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --by metric
```

See [`plot_util/README.md`](plot_util/README.md) for plotting setup and options.

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

Regenerate screenshots with `bash plot_util/setup/make_screenshots.sh`.

## References

- [FASRC jobstats documentation](https://docs.rc.fas.harvard.edu/kb/jobstats/)
- [Princeton jobstats](https://princetonuniversity.github.io/jobstats/)
- For live monitoring, use [KempnerPulse](https://github.com/KempnerInstitute/kempnerpulse)
