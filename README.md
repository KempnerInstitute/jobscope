# jobscope

`jobscope` tells you how well a Slurm job used what it asked for. It reads the
utilization Slurm already stores in each job's `sacct` record — CPU, memory, GPU,
GPU memory — adds DCGM profiling metrics for GPU jobs, and can chart any of it in the
terminal.

For completed jobs the CPU/MEM/GPU/GMEM numbers match `jobstats`, because jobscope
decodes the same stored data — but in one bulk `sacct` query, with no per-job calls
and no cap on how many jobs you look at.

## Screenshots

<!-- Image paths are RELATIVE on purpose. An absolute
     raw.githubusercontent.com/.../main/... URL can only point at main, so a
     screenshot added or regenerated on a branch reads as "image not found" until
     the branch merges -- and a regenerated one silently serves main's old version,
     which is worse. Relative paths resolve against whatever ref you are viewing.
     The cost is that PyPI does not resolve them, so images do not render there. -->
<table width="800">
  <tr><td><strong>Per-job DCGM time series</strong></td></tr>
  <tr><td><img src="docs/timeseries.svg" alt="per-job DCGM time series" width="800"></td></tr>
  <tr><td><strong>Aggregated utilization across jobs</strong></td></tr>
  <tr><td><img src="docs/aggregated.svg" alt="aggregated mean-utilization bars" width="800"></td></tr>
</table>

## Install

Needs Python 3.9+. Install with [`uv`](https://docs.astral.sh/uv/), which provisions
its own Python and puts `jobscope` on your `PATH` — no virtualenv to create or
activate, and it never touches the system Python (too old on most clusters to build
this project anyway).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once, if you lack uv
uv tool install jobscope
```

Restart your shell so `uv` is on your `PATH`. Later, `uv tool upgrade jobscope` or
`uv tool uninstall jobscope`. (Other ways to get `uv`: `pipx install uv`,
`brew install uv`, or the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/).)

If your cluster has no jobscope configuration yet, that is a one-time setup — two
commands, once:

```bash
jobscope probe        # what does this cluster expose, and can jobscope read it
jobscope probe --init # write a config from what it just found
```

`probe` reads your cluster and reports what it found; `--init` turns that into a
config file (and refuses to overwrite one that already exists). Details, and what
each detected setting means, in [`docs/admin.md`](docs/admin.md).

## One job

<img src="docs/onejob.svg" alt="jobscope -j 36770231" width="900">

The row is the job. `USED` under it is resource-time that did work and its share of
what was allocated — so `GPU% 3.3h (28%)` means the job held GPU-hours of which 28%
were busy. The bars draw the same figures, and everything is tinted by the band it
falls in: red is pathological, green is fine.

Read `GMEM% 70%` beside `GPU% 28%` as a job that filled the cards' memory and then
barely computed — the shape a too-small batch or a data-loading bottleneck makes.
That is why the last line reads `average` rather than `good`.

`jobscope 36770231` works too; `-j` is there so you can repeat it for several jobs.

## Your finished jobs in a partition

```bash
jobscope finished -p kempner_h100          # the last day (the default)
jobscope finished -p kempner_h100 -D 3     # widen it to three days
```

<img src="docs/summary.svg" alt="jobscope finished -p kempner_h100" width="900">

`finished` is the mode; without it, bare `jobscope` shows what is **running** now. The
header always restates the window it actually scanned, so a report cannot claim a range
it did not read.

Three things to read here, and the third is the point:

1. **The rows**, tinted per metric. The spread is the story — `GPU% 0` on one job and
   `70` on another means the problem is not the partition, it is particular jobs.
2. **`1. Summary by metric`** pools every job: `GPU% 1.8h (41%)` is the GPU-time that
   did work across the whole selection, and `RED / YELLOW / GREEN` count how many jobs
   fell in each band. A low `USED` with no red jobs means everyone wastes a little; red
   jobs with a decent `USED` means a few jobs waste a lot. Those need different
   conversations.
3. **`3. Problem jobs`** names them. Each row is one measure, with the count that
   tripped it and the cutoff used, then the worst offenders by wasted resource-time —
   `36738257:0%:0.1h(00:06:58)` is that job at 0%, 0.1 GPU-hours wasted, over a
   seven-minute run. `Wasteful all` is the jobs that failed every measure at once,
   which is where to start.

`-D 3` above widens the window; the next section covers the rest.

## Selecting by date

```bash
jobscope finished -D 3                        # the last 3 days
jobscope finished -N 20                       # the most recent 20 jobs, however far back
jobscope finished -S 2026-08-01               # that calendar day alone
jobscope finished -S 2026-07-30 -E 2026-08-01 # an explicit window
jobscope finished -t failed                   # or timeout, cancelled, all
```

Notes worth having:

- **`-S` alone means that day**, not "since then" — the header confirms it as
  `2026-08-01 00:00 .. 2026-08-02 00:00`. Add `-E` for a range.
- Times take `2026-08-01` or `2026-08-01T09:00:00`.
- **`-t` defaults to completed only.** Failed and timed-out jobs are hidden until you
  ask, because a crash at 30 seconds is not an efficiency problem. `-t all` shows
  everything.
- `-N` widens its own window until it has enough jobs, so it reaches as far back as it
  needs to.

Selection is per-user by default — your own jobs.

## Narrowing to one node or GPU

A multi-node job reports one set of numbers for the whole allocation, which hides an
uneven one. `--nodename` recomputes them for a single node:

```
$ jobscope -j 36770231
36770231  alice  COMPLETED  NODE 2  #GPU 8  ...  GPU% 28  ...

$ jobscope -j 36770231 --nodename holygpu8a15401
  Node:      holygpu8a15401 only
36770231  alice  COMPLETED  NODE 1  #GPU 4  ...  GPU% 43  ...
```

28% across the job, 43% on that node — the other node was doing much less, which the
whole-job figure averages away. `NODE` and `#GPU` shrink with the filter, and the
header names it so a narrowed report is never mistaken for a whole-job one.

`--gpuid` narrows to particular cards:

```bash
jobscope -j 36770231 --nodename holygpu8a15401 --gpuid 0,1
```

GPU ids are **per node** — every node numbers its cards from 0, so `--gpuid 0` on a
two-node job keeps two cards. An id that matches nothing is an error naming what the
job did use, rather than a quietly shorter report.

## Per-GPU rows

```
$ jobscope -j 36770231 --per-gpu --nodename holygpu8a15401
  NODE             GPU  CPU%   CPU-MEM       GPU%   GPU-MEM        GMEM%  SM_ACT%  TENSOR%  DRAM%  POWER_W
  holygpu8a15401   0    6.6%   225GB/800GB   43.8%  55.6GB/79.6GB  69.8%  6.4      3.5      2.7    150
  holygpu8a15401   1    6.6%   225GB/800GB   43.8%  55.5GB/79.6GB  69.7%  6.2      3.4      2.6    163
  holygpu8a15401   2    6.6%   225GB/800GB   43.8%  55.5GB/79.6GB  69.7%  6.2      3.4      2.6    154
  holygpu8a15401   3    6.6%   225GB/800GB   40.4%  55.5GB/79.6GB  69.7%  6.0      3.3      2.5    147

  Efficiency by GPU on holygpu8a15401  (filled = used, grey = idle)
    GPU 0                                        GPU 1
         GPU%   44%  ███████████████░░░░░░░░░          GPU%   44%  ███████████████░░░░░░░░░
        GMEM%   70%  ████████████████████████░        GMEM%   70%  ████████████████████████░
```

One row per GPU, with absolute memory beside the percentages. This is how you spot one
straggling card in an otherwise busy job — four cards at 44% each is a job-wide
bottleneck, whereas three at 90% and one at 5% is a distribution problem.

On a multi-node job the footer charts each **node** instead; add `--nodename` to drop
to per-GPU as above.

## The time-series chart

Everything above is one average per job. `--plot_ts` shows the run over time, which is
what distinguishes "used half the GPU throughout" from "used all of it for half the
run":

```bash
jobscope -j 36788818_3 --plot_ts
```

<img src="docs/timeseries.svg" alt="jobscope -j 36788818_3 --plot_ts" width="900">

One panel per GPU, every metric on a shared axis, with min/mean/max/last underneath.
This job is worth reading closely: `GPU%` holds around 90 and `SM_ACT%` around 70, so
the card is genuinely busy — but `TENSOR%` is flat at 2. It is compute-bound on
arithmetic the tensor cores never see, which no single average would have told you and
which is the difference between "this job is fine" and "this job could be much
faster". The step up at the left is start-up: data loading, before any of it counts.

An idle job is a flat line along the bottom. A job that stalls periodically is a comb.

`36788818_3` is an array element, and jobscope takes that spelling directly —
internally Slurm calls it job `36788829`, which you never have to know.

A few variations:

```bash
jobscope -j 36788818_3 --plot_ts 30m                     # just the last 30 minutes
jobscope -j 36770231 --nodename holygpu8a15401 --plot_ts # a multi-node job
jobscope -j 36770231 --nodename holygpu8a15401 --plot_ts --gpuid 0,1
```

`--nodename` is **required** on a multi-node job: the chart keys on GPU, so two nodes'
card 0 would otherwise merge into one line. `--gpuid` keeps the picture readable when a
node holds many cards — each one is another panel.

For the numbers rather than the picture, `--ts` writes the same series as CSV:

```
$ jobscope -j 36788818_3 --ts 20m --csv
JOBID,USER,EPOCH,TIME,NODE,GPU,MODEL,GPU%,SM_ACT%,TENSOR%,DRAM%,POWER_W,CPU%,MEM%
36788818_3,alice,1785713214,2026-08-02T19:26:54,holygpu8a11202,2,NVIDIA H100 80GB HBM3,90,71.6,2.0,46.8,402,99,10
```

One row per GPU per scrape. Pipe it to `jobscope plot` for other chart shapes, or into
whatever you normally use:

```bash
jobscope -j 36788818_3 --ts --csv | jobscope plot --compact
```

`--stats` summarises the window instead — min/mean/max/last per GPU per metric — and
`--classify` sorts a selection's jobs into efficiency categories.

## Fewer or more columns

```bash
jobscope                       # no mode word: what is running now, not history
jobscope -j 36770231 --cpu     # CPU% and MEM% only, no GPU columns
jobscope -j 36770231 --gpu     # the GPU side only
jobscope -j 36770231 --dcgm    # the full DCGM catalog: clocks, temps, PCIe, NVLink
```

`--cpu` on finished jobs needs no Prometheus at all, so it is the fast one on a wide
selection. `--dcgm` widens from four profiling metrics to about thirty; every one of
them is defined in [`docs/reference.md`](docs/reference.md), or run
`jobscope describe --dcgm --ext`.

## Where to look next

| document | what is in it |
|---|---|
| [`docs/reference.md`](docs/reference.md) | every flag, column, band and plot layout |
| [`docs/admin.md`](docs/admin.md) | setting up a cluster: `probe`, the endpoint, configuration |
| [`docs/metrics.md`](docs/metrics.md) | how each number is measured, and how to check one by hand |
| [`CHANGELOG.md`](CHANGELOG.md) | what changed — read before upgrading if you script against `--csv` |

`jobscope describe` prints the column definitions without leaving the terminal, and
`jobscope --help` narrows to whatever command you are part-way through writing.

## Contrib

`contrib/jobstats_extended.py` is a site-specific prototype that folds DCGM metrics
into the jobstats blob itself. It depends on an upstream jobstats install and is not
part of the package; see [`contrib/README.md`](contrib/README.md).

## References

- [FASRC jobstats documentation](https://docs.rc.fas.harvard.edu/kb/jobstats/)
- [Princeton jobstats](https://princetonuniversity.github.io/jobstats/)
- For live monitoring, use [KempnerPulse](https://github.com/KempnerInstitute/kempnerpulse)
