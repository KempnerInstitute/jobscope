# jobscope reference

Every command, flag and column. The [README](../README.md) covers day-to-day use;
this is the full spec.

- [Command shape](#command-shape) · [Columns](#columns) · [The report block](#the-report-block)
- [Thresholds](#thresholds) · [Display config](#display-config)
- [Running jobs](#running-jobs) · [Time series](#time-series) · [Plotting](#plotting)

How each number is *measured* is [`metrics.md`](metrics.md). Site setup is
[`admin.md`](admin.md).

---

## Command shape

One axis per level, so every option composes with every selection:

```
jobscope [MODE] [scope] [filters] [granularity] [columns] [output]
```

Flags and job IDs may be given in any order.

### Which jobs

| word | meaning |
|---|---|
| `running` | jobs running now, via `squeue` (**the default**) |
| `finished` | finished jobs, via `sacct`; default window the last day |
| `JOBID ...` | specific jobs, running or finished (`-j` also works) |

**Scope** (`finished` only): `-D N` days, `-N n` last n jobs, `-S`/`-E` an explicit
window. Any of these without a mode word implies `finished`.

The header states the window actually scanned, since `last 1 day` does not say which
day and a `-D` window moves with the clock:

```
  Select:    last 1 day, completed
  Window:    2026-07-30 11:25 .. 2026-07-31 11:25
```

`-N` walks backwards a day at a time until it has enough jobs, so it stays fast on a
busy partition; the `Window` line reports how far back it went. For an explicit
`-S`/`-E` the `Select` line already *is* the window, so it is not repeated.

**There is no cap on how many jobs a selection returns.** Past a few thousand jobscope
warns on stderr and names the narrowings you are not using, then proceeds — a cap would
silently answer a different question. The cost is real: records are fetched in batches
of 200 and all stay in memory (~1 KB each). One day of *every* partition here selects
over 300,000 jobs.

### Filters

`-p` partition · `-u` user · `-a` all users · `-A` account · `-t` how the job ended
(`finished` only) · `--min-elapsed` runtime floor (`running` only, default 10m — a job
still loading data reads as idle; `0s` disables it).

**`finished` means finished.** Completed jobs only by default, never running ones — a
running job has no final numbers, and mixing it in distorts every figure. `-t` selects
other endings:

| `-t` | Slurm states |
|---|---|
| `completed` (default) | `COMPLETED` |
| `failed` | `FAILED`, `OUT_OF_MEMORY`, `NODE_FAIL`, `BOOT_FAIL` |
| `timeout` | `TIMEOUT`, `DEADLINE` |
| `cancelled` | `CANCELLED`, `PREEMPTED`, `REVOKED` |
| `all` | every state above |

Comma-separated to combine (`-t failed,timeout`). The groups are separate because they
are different problems: a timeout means the walltime or the request was wrong, a
cancellation is a person, a failure is the job. An explicit job ID is never filtered
this way, so `jobscope <jobid>` still reports a job running right now.

### Granularity and columns

| option | effect |
|---|---|
| *(default)* / `--per-job` | one row per job |
| `--per-node` | one row per node, GPU figures pooled across its cards |
| `--per-gpu` | one row per GPU, with node name and GPU number |
| `--verify [WINDOW]` | check one job before acting: min/max, a mean per window rung, share of samples under the cutoff, longest unbroken idle stretch |
| `--ts [WINDOW]` | the per-scrape time series as CSV |
| `--stats` | with `--ts`: summarize it — min/mean/max/last per GPU per metric |
| `--stats node` / `--stats job` | the same, pooled per node / across the job |
| `--eff` | with `--ts`: sort jobs into efficiency categories, worst first |
| `--plot-ts [WINDOW]` | that series charted: one panel per metric, one column per GPU |
| `--plot-ts-overlay [WINDOW]` | overlaid: one panel per GPU, shared axis, one row per node |
| `--cpu` / `--gpu` | narrow the columns to one resource |
| `--all-metrics` | the full DCGM catalog |
| `--runtime-avg` | running: average over the job's runtime (the default, said explicitly) |
| `--instant` | running: the newest scrape instead — one query per metric, so the fast one |

**Output:** `--csv`, `-n`/`--noheader`, `--step` (with `--ts`), `--timeout`,
`--workers`, `--no-color`, `--no-plot`, `-c`.

`--nodename NODE` (or `--node`) and `--gpuid N,N` narrow to one node or specific cards
on **every** view — see [Narrowing](#narrowing-to-a-node-or-a-card).

### Help

`-h` gives a one-line summary per flag; `--help-all` adds the full text. Under a mode
the flags are grouped by the question they answer — **which jobs**, **granularity**,
**columns**, **when**, **output** — and the usage line names those groups.

`-h` also hides what the flags you have already typed rule out, and the footer names
every one it hid:

```console
$ jobscope -j 36441613 --per-gpu -h
... 21 options ...
hiding 15 option(s) these flags rule out: --days, --lastn, ... Pass --help-all for the full list.
```

The rule is mechanical: a flag is hidden exactly when this command would reject or
ignore it. A job ID *is* the selection, so no window or filter can narrow it further.

---

## Columns

Every per-job view prints the same columns, so a job reads identically whether it has
finished or is still running:

```
JOBID  USER  STATE  NODE  CPU%  MEM%  #GPU  GPU%  GMEM%  SM_ACT%  OCC%  TENSOR%  DRAM%  POWER_W  RUNTIME
```

`NODE` is the node count, `#GPU` the allocated GPU count. `CPU%`/`MEM%` sit beside
`SM_ACT%` deliberately: a GPU job with low `GPU%` and high `CPU%` is held up on the
host.

`--cpu` narrows to the host columns — for *finished* jobs that needs no Prometheus at
all. `--gpu` narrows to the GPU columns. `--all-metrics` widens the profiling block.

---

## The report block

After the job listing come three numbered sections: **Summary by metric**, **Average
efficiency** (bars), and **Problem jobs**. `[report] sections` chooses which appear and
in what order; they are numbered as printed. `--noheader` drops the headings and keeps
the data; `--csv` and `--ts` get none of it.

```
1. Summary by metric
METRIC   USED                JOBS  GOOD  OK   BAD
CPU%     633.1h (10%)        315   2     158  155
GPU%     359.7h (75%)        315   305   7    3
SM_ACT%  321.5h (66%)        421   372   16   33
POWER_W  422.1h (92% >100W)  412   294   77   41
```

**There is no per-job mean, on purpose.** Utilization is bimodal — jobs cluster near
0% or 100% — so an average describes a job that does not exist. On the day above `GPU%`
averaged 82% per job while the GPUs, pooled, ran at 44%.

`USED` is used resource-time over allocated. Each metric is measured against the
resource it is a percentage *of*:

| metric | resource |
|---|---|
| `CPU%` | allocated core-hours |
| `MEM%` | allocated host GB-hours |
| `GPU%`, `GMEM%`, every DCGM column | allocated GPU-hours |

So denominators differ by row on purpose, and reading down `USED` is the fastest way to
see which resource a selection wasted.

`JOBS` is how many jobs reported that metric; the three band counts sum to it, and it
differs per row wherever coverage does. A metric no job reported is left out rather
than shown as zeros. Default bands are `GOOD` above 20%, `OK` above 10%, `BAD`
otherwise, with `POWER_W` graded in watts against its floor.

### Reading it: green is not efficient

**`green` means "not pathological", not "efficient".** With an `inefficient` edge of 10,
a job at 21% is green while wasting four fifths of its cores — so a selection can be
half idle with nearly every job green:

```
METRIC  USED        JOBS  GOOD  OK  BAD
CPU%    87.7 (51%)  14    13    1   0
```

Every job there used about half its cores, so none is below 10, yet half the allocation
went unused. Read `USED` for efficiency and the bands for *where* the waste is:

| pattern | meaning | what to do |
|---|---|---|
| red band holds a large **resource** share | a few jobs waste a lot | find them — they are in `Worst` |
| no red band but a low `USED` | every job wastes a little | nothing to escalate; over-requesting is the habit |

The band cells give each band's share of the **jobs** and of the **resource-time**
(`13 (4%)/54%` is 13 jobs, 4% of jobs, holding 54% of the GPU-hours). The gap between
those two numbers is the finding; either alone conceals it.

### Problem jobs

```
Worst GPU (20/77):   alice| 36337337:0%:8h(08:00:29), 36337338:0%:8h(08:00:26)
Worst both (19):     alice| 36337338:gpu0/cpu1(08:00:26)
Jobs:                cpu-jobs=77  gpu-jobs=77  gpus=119  no-jobstats=13
```

Ranked by resource-time *wasted* rather than held, so a long job at a mediocre rate
outranks a short one at zero. One row per measure — duty cycle, SM residency, board
watts, the host — and a row with no red job is omitted. Jobs are grouped under their
owner; each entry reads `jobid:value:wasted(elapsed)`. **A job that ran over three
hours prints in red**: a brief bad job costs little, hours of idle hardware do not come
back. Only red jobs are candidates — a 95%-efficient job can idle 50 GPU-hours just by
being enormous.

Two combined rows follow. `Worst both:` covers the two distinct resources, GPU and CPU;
`Worst all:` covers all four measures. **A combined row lists only jobs red in *every*
measure it names**, which is what makes them unarguable — and means they are often
absent, which is itself the answer. The measures are in different units and cannot be
added, so each job's waste is expressed as a share of the selection's total waste in
that measure and the shares summed. Each cell prints the job's **value** in every metric
named (`36337338:gpu0/sm0/pw70W/cpu1`).

`Jobs:` — the two job counts differ whenever the selection mixes CPU-only and GPU work:
a CPU-only job has no `GPU%` to pool, so it is absent rather than counted as zero. A
GPU job that sat idle *is* counted, as 0%. DCGM rows can legitimately cover more jobs
than `gpu-jobs=`, since a job with no stored summary still has Prometheus data.

### POWER_W

Graded in **watts** against a floor that can differ per GPU model
(`[eff.floor.power]`). Idle draw is hardware: measured on one cluster it ran
from **27 W on a V100 to 165 W on an RTX PRO 6000**, so an idle RTX draws more than a
working V100 and no single number judges both.

Pick a floor by measuring *both* sides — idle 90th percentile against busy 10th. A flat
150 W would have called busy H200s (10th percentile 122 W) idle. Watts are the one idle
signal a duty cycle cannot fake: a job holding a trivial kernel resident reads busy on
`GPU%` while drawing idle watts.

`POWER_W` is the one table row with **no bar**. Its "used" is time spent above the
floor — a detector reading, not a fraction of a resource. On GPUs idling at 119 W it
fills to 100% beside `SM_ACT%` at 2%, reading as the healthiest metric while describing
the same idle GPUs.

### The efficiency bars

```
2. Average efficiency  (filled = used, grey = idle)
     CPU%    6%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     GPU%   28%  ██████████░░░░░░░░░░░░░░░░░░░░░░░░
    GMEM%   70%  ████████████████████████░░░░░░░░░░
```

Bar length is the pooled utilization tinted by its band — the `USED` column drawn
rather than tabulated. Shown by default; `--no-plot` omits them. Follows the table's
metric set, so `--cpu`/`--gpu`/`--all-metrics` narrow or widen it. The closing line says
what the bars were averaged over: "average efficiency" does not say average over what,
and the two answers differ by a third on the same partition.

Not emitted with `--csv` or `--ts`. Distinct from `jobscope plot`, which charts the
per-job CSV; this needs no pipe and no plotting libraries.

### A single job gets the table too

`jobscope <jobid>` prints the metric table for one job, which is how you see which band
each number falls in. Each metric shows a single `1` marking its band. The rest of the
block is suppressed: for one job the pooled row is that job's own row repeated and every
count is 1.

### Highlighting

On a terminal every `%` cell is tinted by efficiency — **red** at or below that metric's
`inefficient` edge, **yellow** at or below `improvement`, **green** above. The pooled
footer is tinted too, and the same cutoffs define the band tallies. `--per-gpu` rows are
graded by the same helper, so one GPU cannot read green in one table and red in another.

---

## Narrowing to a node or a card

```bash
jobscope -j JOBID --nodename NODE                  # the summary, one node
jobscope -j JOBID --nodename NODE --gpuid 0,1      # ... and two of its cards
jobscope -j JOBID --ts 30m --csv --gpuid 2
```

On `--per-gpu` and `--ts` these filter rows. The summary has no such row, so jobscope
narrows the numbers it is computed *from* — the stored summary is already per node and
per GPU, so `CPU%` becomes that node's CPU-seconds over its own cores. The `NODE` and
`#GPU` columns shrink with them.

That is how you find a straggler:

```
$ jobscope -j 36770231                              GPU%=28   #GPU=8  NODE=2
$ jobscope -j 36770231 --nodename holygpu8a15401    GPU%=43   #GPU=4  NODE=1
```

— half the allocation was doing most of the work, which the whole-job 28 hides. The
header says what was filtered (`Node: holygpu8a15401 only`), since a narrowed summary
otherwise looks like a whole-job one.

Both filters run **before** the queries, so one node of a four-node job costs a quarter
of the range queries. GPU ids are per node — nodes number cards from 0, so `--gpuid 0`
on a two-node job keeps two cards. MIG instances are addressed as they print, `0.1`. A
name or id matching nothing is an error naming what *was* touched; an empty report would
read as an idle node rather than a typo.

Note it is `--gpuid`, not `--gpu`: `--gpu` selects the GPU *columns* and takes no value.

### Per-node charts

For a **single job**, `--per-gpu` ends each block with the efficiency chart repeated per
node, packed as many per row as the terminal fits (up to four; a block is ~55 characters,
so 132 columns gives two). Off a terminal the layout is fixed at two, so redirected output
does not change shape with `$COLUMNS`.

**For more than one job the charts are replaced by one aggregate block** after the last
listing. Per job they answer the same question a dozen times and leave no reading of the
selection anywhere. The aggregate is over **jobs**, not printed unit rows, so `--per-node`
and the default view report identical summaries.

With one node in play the unit drops to the GPU, since the card is then the only thing
distinguishing the rows.

---

## Thresholds

Every `%` metric is banded into five tiers by four edges. These are what `--ts --eff`
reports and what tints the tables and `jobscope plot`:

| tier | default range | colour |
|---|---|---|
| `wasteful` | `< 2%` | red |
| `inefficient` | `2–10%` | red |
| `needs improvement` | `10–20%` | yellow |
| `average` | `20–40%` | green |
| `good` | `> 40%` | green |

The edges are **not uniform**: "below 2%" *excludes* 2, while every band above includes
its top. 10.0 is inefficient; 10.1 needs improvement.

**Edges are per metric**, because the metrics do not mean the same thing: a GPU job
legitimately holds cores it never uses, so `CPU%` at 4% is ordinary where `GPU%` at 4% is
idle, and `SM_ACT%` sits structurally below `GPU%` on the same work. `default` covers
every metric you do not name.

**There are two tables and nothing is inherited between them.**
`[thresholds.summary]` grades the plain report (one average over the whole runtime);
`[thresholds.timeslice]` grades `--ts`/`--plot-ts`/`--eff` (samples pooled inside a
window). A two-hour slice catching a checkpoint pause is not a two-hour idle job.

```toml
[thresholds.summary.wasteful]
default = 2
cpu     = 5      # a GPU job idling its cores is normal; 2% would flag them all
[thresholds.timeslice.wasteful]
default = 2
cpu     = 8      # a short window dips further than a whole-job average

[thresholds]
power_w = 100    # POWER_W, in WATTS -- below this a GPU counts as idle
```

A per-model floor goes under `[eff.floor.power]`; see
[`admin.md`](admin.md#choosing-a-power-floor).

Metric names are the column header, lowercase and without the `%` — `gpu`, `cpu`,
`sm_act`, `dram`. Edges must not decrease within a metric once resolved against the
defaults; jobscope rejects a config where they do. `POWER_W` has no bands, only a floor,
shared by both views since idle draw is a property of the hardware.

Run `jobscope config` to print both tables as they resolve.

---

## Display config

`[metrics]` picks the GPU/DCGM metrics per view. Order does not matter — columns print
in catalog order. `jobscope describe --all-metrics` lists the catalog.

```toml
[metrics]
summary    = ["sm_act", "tensor", "dram", "power"]
timeseries = ["gpu", "sm_act", "tensor", "dram", "power"]
extended   = "all"
```

Two things it does not reach: **`--per-gpu` keeps a fixed four** (`SM_ACT%`, `TENSOR%`,
`DRAM%`, `POWER_W`) because its rows are addressed by position, and **the CPU side is
fixed** at `CPU%`/`MEM%` because there are two cgroup queries and no catalog to choose
from.

`[colors]` gives each tier a colour, read by both the tables and `jobscope plot`. Takes
the eight ANSI names and their `bright_` variants, or `color(N)` for the 256-colour cube.
Two tiers sharing a colour is the default, not a requirement — give all five distinct
values for a colourblind-safe palette.

```toml
[colors]
wasteful = "bright_red"
improvement = "yellow"     # the "needs improvement" tier
long_running = "magenta"   # an entry past [defaults] long_running
```

The `--csv` band fields keep the names `red=`/`yellow=`/`green=` whatever you set, so a
script reading them does not break when you recolour the display.

`[report] sections` chooses which parts of the block print, and in what order:

```toml
[report]
sections = ["problems", "metrics"]   # lead with the jobs, skip the bars
```

`[plot]` sets chart defaults, shared by `jobscope plot` and `--plot-ts`:

```toml
[plot]
metrics  = ["gpu", "sm_act", "dram"]   # what a time series draws without --metric
palette  = [196, 46, 33, 208]          # one 256-colour code per series, cycled
max_rows = 40                          # heatmap row cap (--max-rows overrides)
panels   = 12                          # side-by-side panels before --by drops some
```

A metric name the CSV does not carry is skipped rather than charted empty, which is how
one list serves files with different columns.

---

## Utilities

| Command | Purpose |
|---|---|
| `jobscope plot` | render `--csv` output as a terminal chart |
| `jobscope describe` | plain-English column and metric reference (`--metrics` for the catalog) |
| `jobscope config` | show the config path or print an example |
| `jobscope probe` | check what this cluster exposes and whether jobscope can read it |

---

## Running jobs

`jobscope` with no arguments answers "what is happening on the GPUs *now*", selecting
from `squeue` and averaging each job over its own runtime.

```bash
jobscope                          # your running jobs
jobscope -p kempner -a            # every user in a partition
jobscope --instant                # the newest scrape instead of the runtime average
jobscope --min-elapsed 0s         # no runtime floor at all
```

**Why the average is the default.** A snapshot lands wherever the job happens to be, so
it will not match jobstats on a bursty job — one alternating compute with gaps is
genuinely bimodal, and a single scrape can read `GPU% 0` on a GPU averaging ~88%. Reach
for `--instant` when you want the fast query or the current moment specifically, and
`--ts` to see the phases themselves.

`CPU%`/`MEM%` are cumulative by nature — CPU-seconds over elapsed × cores, and peak RSS
— so they read the same in both modes; only the GPU columns follow the choice.

On a **MIG node** `--per-gpu` and `--ts` show the instances, but the DCGM columns read
`-`: NVML identifies an instance by a `MIG-…` UUID where DCGM reports the physical
`GPU-…` one, and nothing in the metrics maps between them.

---

## Useful commands

```bash
# what is running right now
jobscope                                          # your jobs
jobscope -p kempner_eng -a                        # everyone on a partition

# what already ran
jobscope finished                                 # your last day
jobscope finished -S 2026-07-26 -p kempner_eng    # a fixed window on one partition

# one job in detail
jobscope -j 36499551_64                           # summary, with efficiency bars
jobscope -j 36612315 --plot-ts 60m                # its metrics charted, last hour
jobscope -j 36441613 --ts 60m --stats job         # the last hour, averaged

# a whole partition, triaged
jobscope -p kempner_h100 -a --ts 60m --eff
jobscope -p kempner_h100 -a --ts 10m --eff --csv > triage.csv
```

The window goes on whichever flag you are already using — `--plot-ts 60m`, not
`--ts 60m --plot-ts`, since `--plot-ts` *is* `--ts` with the chart in place of the CSV.
`--eff` implies `--stats job`.

---

## Time series

### A window: `--ts 1h`

Both time-series flags take an optional window and give the **last** N of the run. It
narrows the range *queries*, not the rows afterwards, so an hour of a day-long job costs
a twenty-fourth of the samples — and the step is measured over the span actually
queried, so a window keeps native scrape resolution where the whole run would have been
coarsened.

The window lands on the run's own sample grid, so `--ts 1h` returns exactly the rows a
full `--ts` would have.

**A window needs its unit** — `1h`, `90m`, `30s`, `2d`. That is what keeps
`jobscope --ts 36441613` working: a job ID never carries a unit, so a bare number is
handed back as the job it looks like. `--ts 60` is *job 60*; write `--ts 60m`.

### Averages: `--stats`

```console
$ jobscope -j 36612315 --ts 60m --stats
  NODE:GPU          METRIC    N    MIN   MEAN    MAX   LAST
  holygpu8a17601:2  GPU%     61   17.0   20.6   22.0   21.0
  holygpu8a17601:2  POWER_W  61  119.0  119.9  120.0  120.0
```

Computed from the samples `--ts` already fetched, so **no extra queries** and the numbers
cannot disagree with the series. `N` is the sample count — worth a glance, since it says
whether the window had data.

`--stats node` pools each host's GPUs, `--stats job` pools every card. Each level names
what it pooled. Pooling is over the *samples*, not over per-GPU means, so a card the
exporter missed for half the window carries half the weight rather than counting as a
full peer.

This is a plain mean of samples, not each metric's own reducer — the tables peak memory
where this averages it. That is the honest reading of "the average over this window".

### Triage: `--eff`

```console
$ jobscope -p kempner_h100 -a --ts 10m --eff
  142 jobs, by best of GPU%, SM_ACT%, OCC%, TENSOR%, DRAM% (POWER_W caps the verdict when idle)

  wasteful (GPU% <2%, SM_ACT% <2%, ...)  15 jobs
    JOBID       NODES  GPUS  USER   GPU%  SM_ACT%  POWER_W
    36229482    1      2     bob     0.0      0.0       70
  good (best of ...: >40%)  98 jobs
    (--eff all to list them)
```

Each heading states the rule it applied, so the criteria never has to be looked up.
Categories use the [threshold](#thresholds) tiers from `[thresholds.timeslice]`.

**A job is judged on its best metric** — the same rule as "every metric is below X" read
from the other end. `GMEM%` sits out: reserving 80 GB and computing nothing is still
computing nothing. **`POWER_W` plays no part in the category**; it is graded on its own
terms, in watts against a per-model floor, and folding that into a rule expressed in
percent would mean picking one number for every architecture.

`good` collapses to a count by default — on a healthy partition it is most of the output
and none of the point; `--eff all` lists it. `--csv` gives one row per job, carrying
`GMEM%` and `POWER_W` even though neither votes on the label: a row you will sort or join
on should say what was measured.

---

## Plotting

`jobscope plot` renders `--csv` output as terminal bar gauges, histograms, heatmaps and
line charts. The kind is auto-detected from the columns; override with `--kind`.

```bash
jobscope --gpu --csv JOBID                 | jobscope plot              # bar gauges
jobscope --gpu --csv -D 7                  | jobscope plot --kind hist  # distribution
jobscope finished --all-metrics --csv -D 7 | jobscope plot              # heatmap
jobscope JOBID --ts                        | jobscope plot --compact    # time series
```

Time-series charts default to `GPU%`, `GMEM%`, `SM_ACT%`, `OCC%`, `TENSOR%`, `DRAM%`;
`--metric` picks columns and `--all` charts every numeric one. `GPU%` and `GMEM%` lead
because they are the *resources* — how busy the card is and how full — where
`OCC%`/`TENSOR%`/`DRAM%` describe how the SMs were used, which only means something once
the GPU is known to be busy. `GMEM%` catches a failure none of the others do: `GPU%` 96
with `GMEM%` 3 is under-batched.

Do not pass `-n` when piping to `jobscope plot` — it needs the CSV header row. `plot`
reads `summary`, `dcgm` and `dcgm --ts` CSV; the `detail` CSV is for machine consumption.

### Layout

`--by metric` gives one panel per variable, each with its own y-axis — what a mixed set
needs, since `GPU%` spans 0–100 where `OCC%` spans 0–21 and a shared axis crushes them
into the bottom sixth. `--by gpu` (the default) shares one axis per GPU. `--compact`
gives one sparkline row per series. `--gpuid` takes a comma list and each card named
becomes a column; `--columns` asks for the grid without naming them.

**`--plot-ts` is that whole pipeline in one command**, and behaves identically — same
schema, `--step`, window and `--nodename` filter. It charts one job on one node and says
so rather than guessing: several nodes without `--nodename` names them and asks for one,
several jobs points at `-j`. That guard matters because series key on `(node, GPU)`, so
two jobs sharing a card would join into one line.

### `--plot-ts-overlay`

`--plot-ts` gives each metric its own panel and axis, answering "how did this one move".
The overlay answers "did these move together" — every metric on one shared axis, a panel
per GPU, a row per node. It needs no `--nodename`, since a row per node *is* the layout.

```bash
jobscope -j 36770231 --plot-ts-overlay        # 2 nodes x 4 GPUs = two rows of four
```

Two consequences of the shared axis, both stated in the output:

- **Watts are omitted.** A 400 W line pins the scale and flattens every percentage onto
  the floor. `--plot-ts` is where watts get their own panel.
- **Each panel carries its own legend.** With `--no-color` the markers are identical, so
  the legend names traces without distinguishing them — use `--plot-ts` without colour.
