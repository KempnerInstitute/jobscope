# setup/ — installation & environment

Everything needed to put the tools on your `$PATH` and (for `jobstats_plot`)
install its Python dependencies, or build the Singularity image.

| File | Purpose |
|---|---|
| `env.sh` | `source` it to add the tools to `$PATH` for the current shell |
| `install.sh` | one-stop helper: PATH (bashrc / symlinks) + plot deps |
| `jobstats_plot.def` | Singularity definition for the container build |
| `make_screenshots.sh` + `ansi2svg.py` | regenerate the README screenshots (`docs/*.svg`) |

`kempner_jobstats` (the core scanner) needs nothing beyond what `jobstats` already
provides (`sacct`, the `config` module, and `requests` for the Prometheus paths) —
it runs on the system `python3`. Only `jobstats_plot` (optional terminal plots)
needs **Python 3.12 + plotext + rich**.

---

## Cluster requirements

`kempner_jobstats` is a thin **read-only client** over the cluster's existing
jobstats and Prometheus stack. It collects nothing itself.

External infrastructure must already exist:

- **jobstats** installed on the cluster. It provides the `config` module
  (`PROM_SERVER`, `SAMPLING_PERIOD`, thresholds) and writes each job's metrics
  into the Slurm `sacct` `AdminComment` blob (`JS1:...` payload).
- **Slurm / `sacct`** for job selection, state, and reading the `AdminComment`
  blob.
- **Prometheus** for DCGM GPU time series used by `--gpu`, `--dcgm`, and
  `--diagnose`. The offline `--cpu` and `--cgpu` views need only `sacct`.

Data flow:

```text
dcgm-exporter (per GPU node) -> Prometheus
                                   |
sacct AdminComment blob -----------+-> kempner_jobstats --csv -> jobstats_plot
```

Python requirements:

| Tool | Interpreter | Packages |
|---|---|---|
| `kempner_jobstats` | system `python3` | standard library + `requests` on Prometheus paths |
| `jobstats_plot` | `python3.12` | `plotext` + `rich` |

---

## 1. Put the tools on your PATH

### a. This shell only

```bash
source setup/env.sh
kempner_jobstats --gpu -D 5
```

`env.sh` derives the repo root from its own location and prepends the repo root and
`plot_util/` to `$PATH`. Nothing is written to disk.

### b. Permanently (append to ~/.bashrc)

```bash
bash setup/install.sh --path      # appends `source <repo>/setup/env.sh` to ~/.bashrc
```

Idempotent — it won't add the line twice. Open a new shell (or `source ~/.bashrc`).

### c. Symlinks into ~/.local/bin

```bash
bash setup/install.sh --symlink   # ~/.local/bin/{kempner_jobstats,jobstats_plot}
```

Use this if `~/.local/bin` is already on your `$PATH`.

### d. Shared cluster deploy (no clone)

```bash
source /n/holylfs06/LABS/kempner_shared/Everyone/cluster_scripts/job_eff/kempner-jobstats/setup/env.sh
```

Run `bash setup/install.sh` with no arguments at any time to print current status
(PATH, python3.12, uv, and whether the plot deps are importable) and all options.

---

## 2. Install the plot dependencies (only for `jobstats_plot`)

Pick one — in rough order of convenience.

### a. uv — zero install (recommended)

`jobstats_plot` carries inline [PEP 723](https://peps.python.org/pep-0723/)
metadata, so [uv](https://docs.astral.sh/uv/) provisions Python 3.12 + plotext +
rich on demand (cached after the first run):

```bash
kempner_jobstats --dcgm --ts --csv JOBID | uv run --script plot_util/jobstats_plot --compact
```

### b. uv shared venv (one install, e.g. for a deploy)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv plotext rich
kempner_jobstats ... --csv | ./.venv/bin/python plot_util/jobstats_plot
```

### c. pip --user (per user)

```bash
bash setup/install.sh --deps        # or: /usr/bin/python3.12 -m pip install --user plotext rich
```

### d. Singularity container — see §3.

---

## 3. Singularity container (`jobstats_plot.def`)

A tiny image (`python:3.12-slim` + plotext + rich, installed with uv) that runs
**only** `jobstats_plot`. Because it consumes the CSV on stdin, it needs **no host
access** — no Slurm/`sacct`, no Prometheus, no `config`, no secrets inside.

Build (from the **repo root**; rootless `--fakeroot`; needs network for the base
image + wheels):

```bash
singularity build --fakeroot jobstats_plot.sif setup/jobstats_plot.def
```

The build is much faster with a local-disk cache (the network FS is slow for the
many small files of an image extract):

```bash
export SINGULARITY_TMPDIR=/tmp/$USER/sing-tmp SINGULARITY_CACHEDIR=/tmp/$USER/sing-cache
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"
singularity build --fakeroot jobstats_plot.sif setup/jobstats_plot.def
```

Run it (data still comes from `kempner_jobstats` on the host):

```bash
kempner_jobstats --dcgm --ts --csv JOBID | singularity run jobstats_plot.sif --compact
kempner_jobstats --dcgm --csv -D 7       | singularity run jobstats_plot.sif --kind heat
```

`*.sif` is git-ignored — build it where you need it.

---

## 4. Regenerating the README screenshots

`make_screenshots.sh` re-renders `docs/timeseries.svg` and `docs/aggregated.svg`
from real jobs, converting the colored terminal output to SVG with rich
(`ansi2svg.py`, no extra dependency):

```bash
bash setup/make_screenshots.sh                 # uses the default JOBID / window
TS_JOB=12345678 AGG_DAYS=5 bash setup/make_screenshots.sh   # override
```

---

## 5. Verify

```bash
source setup/env.sh
kempner_jobstats --describe | head
kempner_jobstats --gpu -D 1
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --compact
```
