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

```bash
jobscope -j 36770231
```

```
  User:      alice
  Select:    1 job ID(s)
JOBID        USER         STATE     NODE  CPU%   MEM%   #GPU  GPU%   GMEM%   SM_ACT%  TENSOR%  DRAM%   POWER_W  RUNTIME
----------------------------------------------------------------------------------------------------------------------------
36770231     alice        COMPLETED 2     6      28     8     28     70      6.0      3.3      2.6     149      01:29:10

1. Summary by metric
-----------------------------------------------------------------------------------------------------------------------------
  red/yellow/green use each metric's own cutoffs -- CPU% 5/10/20, MEM% 2/10/20, GPU% 2/10/20, GMEM% 2/10/20, SM_ACT% 2/10/20,
    TENSOR% 2/10/20, DRAM% 2/10/20 (wasteful/red/yellow); POWER_W red below 100 W, green above, no yellow. Counts are jobs.
  USED is resource-time that did work, and its share of the allocation -- for POWER_W, the time spent above that floor.
  bands catch pathological jobs, USED measures efficiency: no red with a low USED means every job wastes a little
METRIC   USED            RED  YELLOW  GREEN
CPU%     11.4h (6%)      1    0       0
MEM%     665.8GBh (28%)  0    0       1
GPU%     3.3h (28%)      0    0       1
GMEM%    8.3h (70%)      0    0       1
SM_ACT%  0.7h (6%)       1    0       0
TENSOR%  0.4h (3%)       1    0       0
DRAM%    0.3h (3%)       1    0       0
POWER_W  11.9h (100%)    0    0       1

2. Average efficiency  (filled = used, grey = idle)
---------------------------------------------------
     CPU%    6%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     MEM%   28%  ██████████░░░░░░░░░░░░░░░░░░░░░░░░
     GPU%   28%  ██████████░░░░░░░░░░░░░░░░░░░░░░░░
    GMEM%   70%  ████████████████████████░░░░░░░░░░
  SM_ACT%    6%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  TENSOR%    3%  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    DRAM%    3%  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

Classified by best of GPU%, SM_ACT%, TENSOR%, DRAM%, CPU% (CPU% can vote no higher than inefficient; POWER_W lowers it below the floor).
Classification: average (20-40%)
```

The row is the job. `USED` under it is resource-time that did work and its share of
what was allocated — so `GPU% 3.3h (28%)` means the job held GPU-hours of which 28%
were busy. The bars draw the same figures. On a terminal every figure is also tinted
by the band it falls in — red is pathological, green is fine — which is the one thing
these plain-text blocks cannot show you.

Read `GMEM% 70%` beside `GPU% 28%` as a job that filled the cards' memory and then
barely computed — the shape a too-small batch or a data-loading bottleneck makes.
That is why the last line reads `average` rather than `good`.

`jobscope 36770231` works too; `-j` is there so you can repeat it for several jobs.

## Your finished jobs in a partition

```bash
jobscope finished -p kempner_h100 -D 1     # the last day (-D 1 is the default)
jobscope finished -p kempner_h100 -D 3     # widen it to three days
```

```
  User:      alice
  Partition: kempner_h100
  Select:    last 1 day, completed
  Window:    2026-08-01 23:49 .. 2026-08-02 23:49
JOBID        USER         STATE     NODE  CPU%   MEM%   #GPU  GPU%   GMEM%   SM_ACT%  TENSOR%  DRAM%   POWER_W  RUNTIME
----------------------------------------------------------------------------------------------------------------------------
36738257     alice        COMPLETED 1     2      1      1     0      1       0.4      0.0      0.4     94       00:06:58
36739730     alice        COMPLETED 1     91     2      1     67     24      34.3     1.7      24.4    255      00:34:33
36759002     alice        COMPLETED 1     78     2      1     56     60      24.5     2.1      21.4    226      00:48:07
36765938     alice        COMPLETED 1     13     1      1     2      19      0.0      0.0      0.0     74       00:02:22
36775213     alice        COMPLETED 1     14     1      1     0      21      0.2      0.0      0.0     84       00:06:01
36828732     alice        COMPLETED 1     10     1      1     1      21      0.1      0.0      0.0     85       00:09:36
... 21 more rows ...

1. Summary by metric
-----------------------------------------------------------------------------------------------------------------------------
Used/GPU-hr:                              69     20           39     39      17.4     1.3      14.3    182
  red/yellow/green use each metric's own cutoffs -- CPU% 5/10/20, MEM% 2/10/20, GPU% 2/10/20, GMEM% 2/10/20, SM_ACT% 2/10/20,
    TENSOR% 2/10/20, DRAM% 2/10/20 (wasteful/red/yellow); POWER_W red below 100 W, green above, no yellow. Counts are jobs.
  USED is resource-time that did work, and its share of the allocation -- for POWER_W, the time spent above that floor.
  bands catch pathological jobs, USED measures efficiency: no red with a low USED means every job wastes a little
METRIC   USED           RED  YELLOW  GREEN
CPU%     2.9h (69%)     8    2       8
MEM%     84.3GBh (20%)  17   0       1
GPU%     1.7h (39%)     11   0       7
GMEM%    1.7h (39%)     2    6       10
SM_ACT%  0.8h (18%)     11   3       4
TENSOR%  0.1h (1%)      18   0       0
DRAM%    0.6h (15%)     11   4       3
POWER_W  3.3h (77%)     10   0       8

2. Average efficiency  (filled = used, grey = idle)
---------------------------------------------------
     CPU%   69%  ███████████████████████░░░░░░░░░░░
     MEM%   20%  ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░
     GPU%   39%  █████████████░░░░░░░░░░░░░░░░░░░░░
    GMEM%   39%  █████████████░░░░░░░░░░░░░░░░░░░░░
  SM_ACT%   18%  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  TENSOR%    1%  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    DRAM%   15%  █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

3. Problem jobs
-----------------------------------------------------------------------------------------------
Wasteful GPU (11/18): GPU < 2%
  alice | 36828732:1%:0.2h(00:09:36), 36836738:0%:0.1h(00:02:55), 36822372:0%:0.1h(00:08:30)
Wasteful SM (11/18): SM < 2%
  alice | 36828732:0%:0.2h(00:09:36), 36822372:0%:0.1h(00:08:30), 36827829:0%:0.1h(00:07:27)
Wasteful POWER (10/18): POWER < 100W
  alice | 36828732:85W:0.2h(00:09:36), 36836738:83W:0.1h(00:02:55), 36822372:80W:0.1h(00:08:30)
Wasteful CPU (8/18): CPU < 5%
  alice | 36738257:2%:0.1h(00:06:58)
Wasteful gpu-cpu (1): GPU < 2%, CPU < 5%
  alice | 36738257:gpu0%/cpu2%(00:06:58)
Wasteful all (1): GPU < 2%, SM < 2%, POWER < 100W, CPU < 5%
  alice | 36738257:gpu0%/sm0%/pw94W/cpu2%(00:06:58)
Jobs: cpu-jobs=18  gpu-jobs=18  gpus=20  no-blob=9
```

`finished` is the mode; without it, bare `jobscope` shows what is **running** now. The
header always restates the window it actually scanned, so a report cannot claim a range
it did not read.

Three things to read here, and the third is the point:

1. **The rows**, one per job. The spread is the story — `GPU% 0` on one job and `67` on
   another means the problem is not the partition, it is particular jobs.
2. **`1. Summary by metric`** pools every job: `GPU% 1.7h (39%)` is the GPU-time that
   did work across the whole selection, and `RED / YELLOW / GREEN` count how many jobs
   fell in each band. A low `USED` with no red jobs means everyone wastes a little; red
   jobs with a decent `USED` means a few jobs waste a lot. Those need different
   conversations.
3. **`3. Problem jobs`** names them. Each row is one measure, with the count that
   tripped it and the cutoff used, then the worst offenders by wasted resource-time —
   `36828732:1%:0.2h(00:09:36)` is that job at 1%, 0.2 GPU-hours wasted, over a
   nine-minute run. `Wasteful all` is the jobs that failed every measure at once, which
   is where to start.

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

Everything above is one average per job. A time series shows the run over time, which
is what distinguishes "used half the GPU throughout" from "used all of it for half the
run". `--ts --csv` writes the series and `jobscope plot` charts it:

```bash
jobscope -j 36788818_3 --ts --csv | jobscope plot
```

<img src="docs/timeseries.svg" alt="jobscope -j 36788818_3 --ts --csv | jobscope plot" width="900">

One panel per GPU, every metric on a shared axis, with min/mean/max/last underneath.
This job is worth reading closely: `GPU%` holds around 90 and `SM_ACT%` around 70, so
the card is genuinely busy — but `TENSOR%` is flat at 2. It is compute-bound on
arithmetic the tensor cores never see, which no single average would have told you and
which is the difference between "this job is fine" and "this job could be much
faster". The step up at the left is start-up: data loading, before any of it counts.

An idle job is a flat line along the bottom. A job that stalls periodically is a comb.

`36788818_3` is an array element, and jobscope takes that spelling directly —
internally Slurm calls it job `36788829`, which you never have to know.

**`--plot_ts` does the same thing in one command**, laid out as one panel *per metric*
rather than per GPU — each on its own axis, so watts and percentages can share a chart:

```bash
jobscope -j 36788818_3 --plot_ts                          # one panel per metric
jobscope -j 36788818_3 --plot_ts 30m                      # just the last 30 minutes
jobscope -j 36770231 --nodename holygpu8a15401 --plot_ts  # a multi-node job
```

`--nodename` is **required** on a multi-node job: the chart keys on GPU, so two nodes'
card 0 would otherwise merge into one line. `--gpuid` narrows to particular cards when
a node holds many.

The CSV is useful on its own:

```
$ jobscope -j 36788818_3 --ts 20m --csv
JOBID,USER,EPOCH,TIME,NODE,GPU,MODEL,GPU%,SM_ACT%,TENSOR%,DRAM%,POWER_W,CPU%,MEM%
36788818_3,alice,1785713214,2026-08-02T19:26:54,holygpu8a11202,2,NVIDIA H100 80GB HBM3,90,71.6,2.0,46.8,402,99,10
```

One row per GPU per scrape, so it goes straight into whatever you normally use. Or add
`--compact` for one sparkline row per metric:

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
