# plot_util

`jobstats_plot` draws terminal charts from `kempner_jobstats --csv` output. It is
optional; the core `kempner_jobstats` scanner works without plotting
dependencies.

```bash
kempner_jobstats <view> --csv | jobstats_plot [options]
jobstats_plot -f saved.csv [options]
```

Do not use `-n` with `kempner_jobstats --csv`; plots need the CSV header row.

## Setup

The shared repo already points `jobstats_plot` at the shared plotting venv:

```bash
./kempner_jobstats --dcgm --ts --csv JOBID | ./plot_util/jobstats_plot --compact
```

To use your own venv, build it and record its base directory in
`plot_util/venv_path.conf`:

```bash
bash plot_util/setup/install_venv.sh /path/to/base
./kempner_jobstats --dcgm --ts --csv JOBID | ./plot_util/jobstats_plot --compact
```

The venv is created at `/path/to/base/.venv`. `jobstats_plot` reads
`plot_util/venv_path.conf` and re-runs itself under that venv when `plotext` and
`rich` are not already importable. The setup script requires `uv`.

`venv_path.conf` is tracked and repo-wide, so changing it in a shared checkout
changes the plotting venv for everyone using that checkout.

For a one-off run without changing the config:

```bash
./kempner_jobstats --dcgm --ts --csv JOBID | uv run --script plot_util/jobstats_plot --compact
```

Setup script details are in [`setup/README.md`](setup/README.md).

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

| Input | Default chart |
|---|---|
| One summary job | `bars` |
| Many summary jobs | `hist` |
| `--dcgm --csv` | `heat` |
| `--dcgm --ts --csv` | `line` |

Override with `--kind bars`, `--kind hist`, `--kind heat`, or `--kind line`.

## Useful Options

| Option | Use |
|---|---|
| `--metric NAME[,NAME...]` | choose metrics |
| `--by gpu` / `--by metric` | split time-series panels |
| `--compact` | show compact time-series sparklines |
| `--all` | draw every available metric |
| `--node NODE` / `--gpu N` | filter large jobs |
| `--width N` / `--height N` | control chart size |
| `--max-rows N` | cap heatmap rows |
| `--no-color` | disable color |
| `--config` | use jobstats color thresholds |

Run `jobstats_plot --help` for the full option list.
