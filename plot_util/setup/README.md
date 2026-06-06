# plot_util/setup

Setup helpers for `jobstats_plot`.

`jobstats_plot` needs Python 3.12 with `plotext` and `rich`. The core
`kempner_jobstats` scanner does not use these dependencies.

## Files

| File | Purpose |
|---|---|
| `../venv_path.conf` | records `venv_env_path=<base>` for the plotting venv |
| `install_venv.sh` | creates `<base>/.venv` and updates `venv_path.conf` |
| `make_screenshots.sh` | regenerates `docs/timeseries.svg` and `docs/aggregated.svg` |
| `ansi2svg.py` | converts ANSI output to SVG for screenshots |

## Venv

Use the shared default venv:

```bash
./kempner_jobstats --dcgm --ts --csv JOBID | ./plot_util/jobstats_plot --compact
```

Build and use your own venv:

```bash
bash plot_util/setup/install_venv.sh /path/to/base
```

This requires `uv`. It creates `/path/to/base/.venv`, installs `plotext` and
`rich`, and rewrites `plot_util/venv_path.conf`. After that, `jobstats_plot`
will automatically run under that venv.

`venv_path.conf` is tracked and repo-wide, so changing it in a shared checkout
changes the plotting venv for everyone using that checkout.

One-off run without changing `venv_path.conf`:

```bash
./kempner_jobstats --dcgm --ts --csv JOBID | uv run --script plot_util/jobstats_plot --compact
```

## Screenshots

```bash
bash plot_util/setup/make_screenshots.sh
TS_JOB=12345678 AGG_DAYS=5 bash plot_util/setup/make_screenshots.sh
```

## Verify

```bash
./kempner_jobstats --describe | head
./kempner_jobstats --gpu -D 1
./kempner_jobstats --dcgm --ts --csv JOBID | ./plot_util/jobstats_plot --compact
```
