# plot_util/ — `jobstats_plot`

`jobstats_plot` turns `kempner_jobstats --csv` output into a terminal graph. It is
a **separate, optional** tool: the core scanner stays dependency-free and is not
affected by it. They are coupled only by the CSV pipe.

```bash
kempner_jobstats <view> --csv | jobstats_plot [options]
jobstats_plot -f saved.csv [options]            # or from a saved file
```

Needs **Python 3.12 + plotext + rich** — see
[`../setup/README.md`](../setup/README.md) (uv / pip / container). The quickest:

```bash
kempner_jobstats --dcgm --ts --csv JOBID | uv run --script plot_util/jobstats_plot --compact
```

> Pipe **without** `-n` so the column header row is included (the parser needs it).

---

## What gets plotted (chart kinds)

The kind is **auto-detected from the CSV columns**; override with `--kind`.

| `--kind` | Best input | What it shows |
|---|---|---|
| `bars` | one job, or many (mean) | horizontal utilization gauges per metric, colored red/yellow/green by threshold |
| `hist` | many jobs (summary CSV) | distribution of one metric across the selection |
| `heat` | `--dcgm --csv` (per-GPU) | table of jobs/GPUs × metrics, each cell background-colored by value |
| `line` | `--dcgm --ts --csv` (one job) | time-series chart over the job's window + a min/mean/max/last summary |
| `auto` *(default)* | — | line if the CSV has `EPOCH`/`TIME`; heat if it has `NODE`+`GPU`+`DUR_S`; else summary → `bars` (1 job) or `hist` (many) |

```bash
kempner_jobstats --gpu  --csv JOBID      | jobstats_plot              # -> bars (one job)
kempner_jobstats --gpu  --csv -D 7       | jobstats_plot              # -> histogram (many jobs)
kempner_jobstats --gpu  --csv -D 7       | jobstats_plot --kind bars  # -> mean-of-N bars
kempner_jobstats --dcgm --csv -D 7       | jobstats_plot              # -> heatmap (jobs/GPUs x metrics)
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot              # -> time-series line chart
```

---

## The time-series view (`--dcgm --ts`)

This is where the layout adapts to the job's shape. Defaults to **`--by gpu`**.

- **`--by gpu`** *(default)* — one panel per GPU, all metrics on a shared y-axis.
  Compact; metrics in very different ranges can overlap.
- **`--by metric`** — one panel per metric, each with its **own y-axis** (so
  OCC% vs SM_ACT%, or TENSOR% vs DRAM%, don't crowd each other). With multiple
  GPUs, each metric panel draws a line per GPU.
- **`--compact`** — one **braille sparkline row per metric** (own scale +
  `min–max, mean`), instead of full-height panels. Densest overview.
- **multi-GPU** — `--by gpu`/`--by metric` facet accordingly; lines are colored
  per GPU where a panel holds several.
- **multi-node** — facets by node (one panel per node, a line per GPU, a single
  metric). Stacked panels are capped (a note tells you when).

Narrowing & styling:

- `--node NODE` — only that node (also drills a multi-node job down to per-metric panels).
- `--gpu N` — only that GPU index. (`--node`/`--gpu` also filter the heatmap.)
- `--metric A,B` — restrict to these metrics (single one for `hist`).
- `--all` — draw every metric in the catalog, not just the default 4.
- `--marker braille|dot|hd|fhd` — line style; `braille` (default) = thin lines,
  `hd` = thick blocks.

```bash
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --by metric
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --compact
kempner_jobstats --dcgm --ts --csv JOBID | jobstats_plot --node holygpu8a11302 --gpu 0
```

---

## All options

| Option | Meaning |
|---|---|
| `FILE` / `-f FILE` | read CSV from a file instead of stdin |
| `--kind auto\|bars\|heat\|hist\|line` | chart type (default: auto-detect) |
| `--metric NAME[,NAME...]` | metric(s): one for `hist`, comma-list for `line` |
| `--by metric\|gpu` | line faceting (default `gpu`) |
| `--compact` | line: one sparkline row per metric (own scale) |
| `--all` | line: draw every metric, not just the default set |
| `--marker braille\|dot\|hd\|fhd` | line marker (default `braille`) |
| `--node NODE` | plot only this node (where the CSV has a `NODE` column) |
| `--gpu N` | plot only this GPU index |
| `--width N` / `--height N` | plot size in characters (plotext charts) |
| `--max-rows N` | heatmap row cap (default 40) |
| `--no-color` | disable color (also respects `$NO_COLOR`) |
| `--config` | pull color thresholds from the jobstats `config` module |

`--help` for the authoritative list.

---

## Color

Percent metrics are graded red / yellow / green by threshold (built-in defaults,
or the jobstats `config` values with `--config`). Non-percent columns (POWER_W,
*_GB, clocks, temps) are shown as plain values. `--no-color` / `$NO_COLOR` turns
color off. When stdout is not a terminal, set `FORCE_COLOR=1` to keep ANSI color
(this is how `setup/make_screenshots.sh` captures the SVGs).
