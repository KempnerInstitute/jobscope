# plot_util/ - `jobstats_plot`

`jobstats_plot` turns `kempner_jobstats --csv` output into terminal charts. It is
optional and separate from the core scanner; the two tools are connected only by
CSV.

```bash
kempner_jobstats <view> --csv | jobstats_plot [options]
jobstats_plot -f saved.csv [options]
```

Do not use `-n` with `kempner_jobstats --csv`; `jobstats_plot` needs the header
row. Plot dependencies are covered in [`../setup/README.md`](../setup/README.md).

Quickest no-install run:

```bash
kempner_jobstats --dcgm --ts --csv JOBID | uv run --script plot_util/jobstats_plot --compact
```

## Examples

```bash
kempner_jobstats --gpu  --csv JOBID      | jobstats_plot              # bars
kempner_jobstats --gpu  --csv -D 7       | jobstats_plot              # histogram
kempner_jobstats --dcgm --csv -D 7       | jobstats_plot              # heatmap
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --by metric  # time series
jobstats_plot -f saved.csv --kind heat
```

## Chart Kinds

`--kind auto` is the default.

| Input | Default chart | Shows |
|---|---|---|
| One summary job | `bars` | utilization gauges |
| Many summary jobs | `hist` | distribution of one metric |
| `--dcgm --csv` | `heat` | jobs/GPUs by metric |
| `--dcgm --ts --csv` | `line` | time series plus min/mean/max/last |

Override with `--kind bars`, `--kind hist`, `--kind heat`, or `--kind line`.

## Useful Options

| Option | Use |
|---|---|
| `--metric NAME[,NAME...]` | choose histogram metric or line metrics |
| `--by gpu` | time series: one panel per GPU, shared metric axis |
| `--by metric` | time series: one panel per metric, separate y-axis |
| `--compact` | time series: one sparkline row per metric |
| `--all` | time series: draw every available metric |
| `--node NODE` / `--gpu N` | filter large jobs or heatmaps |
| `--width N` / `--height N` | control chart size |
| `--max-rows N` | cap heatmap rows |
| `--no-color` | disable color; also honors `$NO_COLOR` |
| `--config` | use jobstats color thresholds |

Percent metrics are colored red/yellow/green by threshold. Non-percent metrics
such as `POWER_W`, memory, clocks, and temperatures are shown as plain values.

Run `jobstats_plot --help` for the authoritative option list.
