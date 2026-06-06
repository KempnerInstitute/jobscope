# plot_util/setup/ - venv setup for jobstats_plot

`jobstats_plot` needs Python 3.12 with `plotext` + `rich`; the core
`kempner_jobstats` scanner needs none of that. `jobstats_plot` finds its venv
itself -- it reads `plot_util/venv_path.conf` (`venv_env_path=<base>`) and re-execs
under `<base>/.venv/bin/python`. So there is no PATH or launcher setup: build a
venv, point the config at it, and run `jobstats_plot` by path.

| File | Purpose |
|---|---|
| `../venv_path.conf` | tracked: `venv_env_path=<base>` -- the venv `jobstats_plot` uses (default: shared) |
| `install_venv.sh` | build a venv at `<base>/.venv` and write `venv_env_path` into the config |
| `make_screenshots.sh` | regenerate `docs/timeseries.svg` and `docs/aggregated.svg` |
| `ansi2svg.py` | ANSI -> SVG helper used by `make_screenshots.sh` |

## Requirements

`kempner_jobstats` runs on the system `python3` and uses the cluster's existing
jobstats stack:

- Slurm `sacct` for job selection and stored jobstats blobs.
- The jobstats `config` module for Prometheus settings and thresholds.
- Prometheus DCGM time series for `--gpu`, `--dcgm`, and `--diagnose`.

The offline `--cpu` and `--cgpu` views need only `sacct` and the stored jobstats
blob. `jobstats_plot` is optional and needs Python 3.12 with `plotext` and `rich`.

Data flow:

```text
sacct AdminComment blob + Prometheus DCGM metrics -> kempner_jobstats --csv -> jobstats_plot
```

## How jobstats_plot finds its venv

On startup `jobstats_plot` resolves its plotting deps, first match wins:

1. If `plotext` + `rich` are already importable (you ran it under the venv, or via
   `uv run`), it runs in place.
2. Otherwise it reads `venv_env_path` from `../venv_path.conf` and re-execs under
   `<venv_env_path>/.venv/bin/python` (falling back to the shared deployment's
   `.venv` if the config is missing).

`venv_path.conf` is tracked and defaults to the shared cluster venv; it is
repo-global, so changing it affects everyone using this repo.

## Usage

Shared default -- nothing to build; the config already points at the shared
`.venv`, so just run the tools by path:

```bash
./kempner_jobstats --dcgm --ts --csv JOBID | ./plot_util/jobstats_plot --compact
```

(Add the repo to your `$PATH` yourself if you prefer to run them by name.)

Your own venv -- build it and point the config at it (overwrites the previous
`venv_env_path`):

```bash
bash plot_util/setup/install_venv.sh /path/to/base   # builds <base>/.venv, records it
bash plot_util/setup/install_venv.sh                 # or: prompt; default = shared base
./kempner_jobstats --dcgm --ts --csv JOBID | ./plot_util/jobstats_plot --compact
```

Zero install -- let `uv` provision Python 3.12 + deps on demand (no config used):

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
