# setup/ - installation and environment

Use this directory to put `kempner_jobstats` and `jobstats_plot` on your `$PATH`,
install optional plotting dependencies, build the plot container, or regenerate
README screenshots.

| File | Purpose |
|---|---|
| `env.sh` | add the repo tools to `$PATH` for the current shell |
| `install.sh` | helper for PATH setup, symlinks, and plot dependencies |
| `jobstats_plot.def` | Singularity definition for `jobstats_plot` |
| `make_screenshots.sh` | regenerate `docs/timeseries.svg` and `docs/aggregated.svg` |

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

## PATH Setup

Current shell only:

```bash
source setup/env.sh
```

Shared cluster deploy, no clone needed:

```bash
source /n/holylfs06/LABS/kempner_shared/Everyone/cluster_scripts/job_eff/kempner-jobstats/setup/env.sh
```

Permanent setup:

```bash
bash setup/install.sh --path      # append source line to ~/.bashrc
bash setup/install.sh --symlink   # create ~/.local/bin symlinks
```

Run `bash setup/install.sh` with no arguments to print PATH and dependency
status.

## Plot Dependencies

Recommended: use `uv` and let `jobstats_plot` provision Python 3.12 plus
dependencies on demand:

```bash
kempner_jobstats --dcgm --ts --csv JOBID | uv run --script plot_util/jobstats_plot --compact
```

Shared venv:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv plotext rich
kempner_jobstats --csv -D 7 | ./.venv/bin/python plot_util/jobstats_plot
```

Per-user pip:

```bash
bash setup/install.sh --deps
```

## Singularity Plot Container

The container runs only `jobstats_plot`; CSV still comes from `kempner_jobstats`
on the host.

```bash
singularity build --fakeroot jobstats_plot.sif setup/jobstats_plot.def
kempner_jobstats --dcgm --ts --csv JOBID | singularity run jobstats_plot.sif --compact
```

`*.sif` is git-ignored.

## Screenshots

```bash
bash setup/make_screenshots.sh
TS_JOB=12345678 AGG_DAYS=5 bash setup/make_screenshots.sh
```

## Verify

```bash
source setup/env.sh
kempner_jobstats --describe | head
kempner_jobstats --gpu -D 1
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --compact
```
