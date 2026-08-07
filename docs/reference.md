# jobscope reference

Every column, flag, band and plot layout. The [README](../README.md) covers the
handful of commands you run day to day; this is what each of them can be told to do
and exactly what comes back.

- [The argument tree](#the-argument-tree)
- [Columns](#columns)
- [Utilities](#utilities)
- [Running jobs](#running-jobs)
- [Useful commands](#useful-commands)
- [Plotting](#plotting)

How each number is *measured* -- which source wins, how a metric is reduced over time
and across GPUs, the raw-vs-display job ID rule, MIG limits -- is
[`metrics.md`](metrics.md). Site setup and configuration is [`admin.md`](admin.md).

---


## The argument tree

One axis per level, so every option composes with every selection:

```
jobscope [MODE] [scope] [filters] [granularity] [columns] [output]
```

**Level 1 — which jobs.** The first word, defaulting to `running`:

| word | meaning |
|---|---|
| `running` | jobs running now, via `squeue` (**the default**) |
| `finished` | finished jobs, via `sacct`; default window the last day |
| `JOBID ...` | specific jobs, running or finished (`-j` also works) |

**Level 2 — scope** (`finished` only): `-D N` days, `-N n` last n jobs, `-S`/`-E`
an explicit window. Passing one of these without a mode word implies `finished`,
so `jobscope -D 3` still means what it always did.

The header states the window it actually scanned, since `last 1 day` does not say
*which* day and a `-D` window moves with the clock:

```
  Select:    last 1 day, completed
  Window:    2026-07-30 11:25 .. 2026-07-31 11:25
```

A bare `-N` gets one too, and its span is decided at query time. sacct has no
"last N", so a window has to be scanned and trimmed -- and listing job IDs costs
roughly in proportion to the span: one day of `kempner_eng` takes ~1.2s where thirty
days does not return inside the 60s cap. So `-N` walks backwards **one day at a
time**, and each query covers only the new day, stopping as soon as enough jobs have
been found.

`-N 1` on a busy partition therefore makes one query, and `-N 3` for a user whose
newest job is five days old makes six, in under two seconds. Re-scanning from now at
every step would re-list the same jobs over and over, which is what made the naive
version slow. The `Window` line reports how far back it went. For an explicit `-S`/`-E` the `Select` line already *is* the window, so it
is not repeated; that also shows how `-S DATE` alone was widened to the whole
calendar day.

There is **no cap on how many jobs a selection may return**. Past a few thousand,
jobscope says so on stderr and names the narrowings you are not already using — the
run still proceeds, because a cap would silently answer a different question than the
one you asked. The cost past that point is real: the per-job data is fetched in
batches of 200, and every record stays in memory until the run ends (roughly 1 KB
each, measured). A cluster-wide sweep with no `-p` and a wide window is the case to
avoid — one day of *every* partition here selects over 300,000 jobs.

**Filters**, every mode: `-p` partition, `-u` user, `-a` all users, `-A` account,
`-t` how the job ended (`finished` only, see below), `--min-elapsed` runtime floor
(`running` only, default 10m -- a job still loading data reads as idle;
`[defaults] min_elapsed` changes it, `0s` disables it).

**`finished` means finished.** It reports **completed jobs only** by default, and
never jobs that are still running -- a running job has no final numbers, so mixing
it into a report of finished ones distorts every figure. `-t` selects other endings:

| `-t` | Slurm states |
|---|---|
| `completed` (default) | `COMPLETED` |
| `failed` | `FAILED`, `OUT_OF_MEMORY`, `NODE_FAIL`, `BOOT_FAIL` |
| `timeout` | `TIMEOUT`, `DEADLINE` |
| `cancelled` | `CANCELLED`, `PREEMPTED`, `REVOKED` |
| `all` | every state above |

Comma-separated to combine: `jobscope finished -t failed,timeout`. The groups are
separate because they are different problems -- a timeout usually means the walltime
or the resource request was wrong, a cancellation is a person, and a failure is the
job. Passing a running state (`-t running`) is an error pointing at `jobscope running`.

An explicit job ID is never filtered this way, so `jobscope <jobid>` still reports a
job that is running right now.

**Level 3 — granularity** (pick one) and **columns**:

| option | effect |
|---|---|
| *(default)* | one row per job |
| `--verify [WINDOW]` | check one job before acting on it: min/max, a mean per window rung, the share of samples under the metric's cutoff, the longest unbroken idle stretch, and the shape that follows |
| `--per-job` | one row per job (the default) |
| `--per-node` | one row per node, its GPU figures pooled across the cards it holds |
| `--per-gpu` | one row per GPU, with node name and GPU number (see below) |
| `--ts [WINDOW]` | the per-scrape time series as CSV; `--ts 1h` is the last hour of the run |
| `--stats` | with `--ts`: summarize that series instead -- min/mean/max/last per GPU per metric |
| `--stats node` | the same, pooled per node |
| `--stats job` | the same, pooled across every node and GPU |
| `--eff` | with `--ts`: sort the jobs into efficiency categories, worst first |
| `--plot-ts [WINDOW]` | that time series charted instead: one panel per metric, one column per GPU |
| `--plot-ts-overlay [WINDOW]` | the same series overlaid: one panel per GPU, every metric on a shared axis, one row per node |
| `--cpu` / `--gpu` | narrow the columns to one resource |
| `--all-metrics` | the full DCGM metric catalog |
| `--runtime-avg` | running: average each metric over the job's runtime — the default, said explicitly |
| `--instant` | running: the newest scrape instead — one query per metric however many jobs, so the fast one |

**Output** — `--csv`, `-n`, `--step` (with `--ts`), `--timeout`, `--workers`, `-c`.
`--help-all` prints every option; plain `-h` narrows to the ones the current flags
leave usable (see below).

Flags and JOBIDs may be given in any order. A `JOBID` works whether the job is
running or finished: Slurm only stores the utilization summary when a job *ends*, so
for a running one jobscope reconstructs `CPU%`/`MEM%`/`GPU%`/`GMEM%` from the same
Prometheus metrics jobstats falls back to. With no Prometheus endpoint configured
those columns stay blank and say so.

`jobscope running -j ID` and `jobscope ID` differ in how the job is *found* -- the first
through `squeue`, the second through `sacct` -- but no longer in what its numbers mean.
A job that has not ended is averaged over its runtime either way, and `--instant` reads
its newest scrape either way; a finished job is always averaged. The header's `Sampled:`
line states which of the two you are looking at. The second form used to fold a *running*
job silently, which made the two views of one job disagree with nothing to explain it --
now they agree because both average, and both say so.

### The help is a tree: modes, then axes

`jobscope --help` lists the modes and the utilities. Under a mode the reporting flags
are grouped by the question they answer, and the usage line names those groups:

```console
$ jobscope running -h
usage: jobscope running [JOBID ...] [which jobs] [granularity] [columns] [when] [output]
...
granularity:
  one row per what -- or a per-scrape time series instead of rows

  --per-job             one row per job (the default)
  --per-node            one row per node
  ...
```

Five axes, each with the question it answers under its heading: **which jobs**,
**granularity**, **columns**, **when**, **output**. The names in the usage line are the
headings verbatim, and a test pins them together so the summary cannot drift from the
sections it summarises.

`-h` prints a **one-line summary** per flag; `--help-all` adds the full text. `--verify`
is the extreme case -- one line against five. The detail is not gone, only one command
away, and `--help-all` was already the everything view.

### `-h` also narrows to the command you are writing

Thirty options is a lot to re-read when most of them cannot apply. So `-h` answers
for *this* invocation, hiding what it has already ruled out:

```console
$ jobscope -j 36441613 --per-gpu -h
... 21 options ...
hiding 15 option(s) these flags rule out: --days, --lastn, --starttime, --endtime,
--min-elapsed, --partition, --user, --all-users, --account, --state, --ts, --plot-ts,
--stats, --eff, --step.
Pass --help-all for the full list.
```

The rule is mechanical, not editorial: **a flag is hidden exactly when this command
would reject it or ignore it.** The job ID *is* the selection, so no window or filter
can narrow it further (jobscope says so at runtime too); `--ts` is mutually exclusive
with `--per-gpu`; `--step` is read only by the time series. `--runtime-avg` is dropped
only for a window selection, which holds finished jobs alone -- an explicit JOBID can
name a running one, so it stays offered there. Nothing is hidden for being merely
uninteresting, and the footer names every one that went.

`jobscope --help` has no flags to narrow against, so it lists the modes and utilities
and points at where the reporting flags live (`jobscope finished -h`, or `-h` after any
command) and at `jobscope probe -h`. `--help-all` is the
way back to all thirty, with their full text, from anywhere.


## Columns

Every per-job view prints the same columns, so a job reads identically whether it
has finished or is still running:

```
JOBID  USER  STATE  NODE  CPU%  MEM%  #GPU  GPU%  GMEM%  SM_ACT%  OCC%  TENSOR%  DRAM%  POWER_W  RUNTIME
```

`NODE` is the node count and `#GPU` the allocated GPU count. `CPU%`/`MEM%` sit
beside `SM_ACT%` deliberately: a GPU job whose `GPU%` is low and `CPU%` is high is
held up on the host, and no single view used to show both.

After the job listing come three numbered sections:

```

1. Summary by metric
------------------------------------------------------------------------------------------------
Used/GPU-hr:                              10     6            75   49     66.0  20.6  18.1  14.5
  Averaged over time: every value below is each job's mean over its whole runtime. Pooled across jobs by resource-time, so a
    10-hour job weighs ten times a 1-hour one.
  JOBS is how many jobs reported the metric and GOOD/OK/BAD sum to it, so the total differs per row wherever coverage does. Banded
    by GOOD above 20%, OK above 10%, BAD otherwise; POWER_W is GOOD above 100 W and BAD below, never OK.
  USED is the resource-time that did work, over what was allocated -- for POWER_W, the time spent above that watt floor.
  bands locate the waste and USED measures it: no red jobs but a low USED means every job wastes a little, rather than a few jobs
    wasting a lot
METRIC   USED                JOBS  GOOD  OK   BAD
CPU%     633.1h (10%)        315   2     158  155
MEM%     5.9TBh (6%)         315   4     8    303
GPU%     359.7h (75%)        315   305   7    3
GMEM%    235.6h (49%)        315   12    9    294
SM_ACT%  321.5h (66%)        421   372   16   33
OCC%     102.7h (21%)        421   31    267  123
TENSOR%  87.5h (18%)         421   2     10   409
DRAM%    67.7h (14%)         421   19    18   384
POWER_W  422.1h (92% >100W)  412   294   77   41

2. Average efficiency  (filled = used, grey = idle)
---------------------------------------------------
  bars are the USED share above -- averaged over
    each job's whole runtime
     CPU%   10%  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     MEM%    6%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     GPU%   75%  ██████████████████████████░░░░░░░░
    GMEM%   49%  █████████████████░░░░░░░░░░░░░░░░░
  SM_ACT%   66%  ███████████████████████░░░░░░░░░░░
     OCC%   21%  ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░
  TENSOR%   18%  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    DRAM%   14%  █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

3. Problem jobs
--------------------------------------------------------------------------------------------
Worst GPU (20/77):   alice| 36337337:0%:8h(08:00:29), 36337338:0%:8h(08:00:26)
Worst SM (20/77):    alice| 36337337:0%:8h(08:00:29), 36337292:0%:8h(08:00:21)
Worst POWER (20/77): alice| 36337337:68W:8h(08:00:29), 36337338:70W:8h(08:00:26)
Worst CPU (50/77):   alice| 36337338:1%:64.1h(08:00:26), 36337292:1%:64h(08:00:21)
Worst both (19):     alice| 36337338:gpu0/cpu1(08:00:26), 36337292:gpu0/cpu1(08:00:21)
Worst all (19):      alice| 36337338:gpu0/sm0/pw70W/cpu1(08:00:26)
Jobs:                cpu-jobs=77  gpu-jobs=77  gpus=119  no-jobstats=13
```

**There is no per-job mean**, on purpose. Utilization is bimodal -- jobs cluster
near 0% or near 100% -- so an average of them describes a job that does not
exist. On the day above `GPU%` averaged 82% per job while the GPUs, pooled,
ran at 44%.

`Used/GPU-hr:` is a ratio rather than a centre: used resource-time over allocated
resource-time, aligned under the columns above it.

**One table row per graded metric**, and the set follows the view: eight by
default, `CPU%`/`MEM%` under `--cpu`, six under `--gpu`, the full catalog under
`--all-metrics`. A metric no job reported is left out rather than shown as zeros. Each
metric is measured against the resource it is a percentage *of*:

| metric | resource |
|---|---|
| `CPU%` | allocated core-hours |
| `MEM%` | allocated host GB-hours |
| `GPU%`, `GMEM%`, and every DCGM column | allocated GPU-hours |

So the denominators differ by row on purpose, and reading down the `USED` column
is the fastest way to see which resource a selection actually wasted: above, the
GPUs ran at 44% while the *cores* managed 5% and the *tensor cores* 10%.

By default one set of cutoffs covers every metric -- **`GOOD` above 20%, `OK` above
10%, `BAD` otherwise** -- and `POWER_W` is `BAD` below 100 W. `JOBS` is how many jobs
reported that metric, which the three counts sum to; it differs per row wherever
coverage does, since a job whose GPU memory resolved but whose duty cycle did not is
graded by `GMEM%` and not by `GPU%`. The cutoffs are
per metric, though, so a site can give `CPU%` a different bar from `GPU%`; the
legend above the table states whichever ones are in force. See
[Thresholds](#thresholds).

The band cells are plain job counts. Their resource shares stay in the `--csv`
output for anyone scripting them.

**`green` means "not pathological", not "efficient".** With an `inefficient` edge of
10 a job at 21% is green while wasting four fifths of its cores, so a selection can
be half idle with nearly every job green:

```
  Averaged over time: every value below is each job's mean over its whole runtime. Pooled across jobs by resource-time, so a
    10-hour job weighs ten times a 1-hour one.
  JOBS is how many jobs reported the metric and GOOD/OK/BAD sum to it, so the total differs per row wherever coverage does. Banded
    by GOOD above 20%, OK above 10%, BAD otherwise; POWER_W is GOOD above 100 W and BAD below, never OK.
  USED is the resource-time that did work, over what was allocated -- for POWER_W, the time spent above that watt floor.
  bands locate the waste and USED measures it: no red jobs but a low USED means every job wastes a little, rather than a few jobs
    wasting a lot
METRIC  USED        JOBS  GOOD  OK  BAD
CPU%    87.7 (51%)  14    13    1   0
```

That is not a contradiction: every job there used about half its cores, so none is
below 10, yet half the allocation went unused. Read `USED` for efficiency and the
bands for *where* the waste is:

| pattern | meaning | what to do |
|---|---|---|
| red band holds a large **resource** share | a few jobs waste a lot | go find those jobs -- they are in `Worst` |
| no red band but a low `USED` | every job wastes a little | nothing to escalate; over-requesting is the habit |

On the partition above, `GPU%` is the first pattern (4% of jobs holding 53% of the
GPU-hours in red) and `CPU%` the second.

The three band cells give each band's share of the **jobs** and of the
**resource-time** (`13 (4%)/54%` is 13 jobs, 4% of the jobs, holding 54% of the
GPU-hours). The gap between those two numbers is the finding, and either alone
conceals it. On a terminal the band cells are printed in their own colours and
`USED` is tinted by that metric's pooled grade, so a wasted resource is a red
line in the block.

The `Worst` rows name the offenders, ranked by resource-time *wasted* rather than
held, so a long job at a mediocre rate outranks a short one at zero. There is one
row per measure -- duty cycle, SM residency, board watts, and the host -- and any
row with no red job is omitted.

Jobs are grouped under their owner, since one user usually owns several of them, and
each entry reads `jobid:value:wasted(elapsed)`. **A job that ran over three hours is
printed in red**: a brief bad job costs little, whereas hours of idle hardware do not
come back.

`POWER_W` is graded in **watts**, not percent, and against a floor that can differ
per GPU model (`[thresholds.power_w_by_model]`). Idle draw is hardware: measured on one
cluster it ran from **27 W on a V100 to 165 W on an RTX PRO 6000**, so an idle RTX draws
more than a working V100 and no single number can judge both. The model comes from the
exporter's own label, on a query jobscope already makes, so it costs nothing.

Pick a floor by measuring *both* sides -- idle 90th percentile against busy 10th -- not
just the idle one: on that cluster a flat 150 W would have called busy H200s (10th
percentile 122 W) idle. `jobscope config --example` carries the method and a worked
table. Below `[thresholds] power_w`
(default 100) a GPU counts as idle, so its waste is the GPU-hours held while below
that floor. Watts are the one idle signal a duty cycle cannot fake -- a job holding
a trivial kernel resident reads busy on `GPU%` while drawing idle watts. On the day
above the two agreed: the four lowest-power jobs sat at 73-74 W with `GPU% 0` and
`SM_ACT% 0.0`.

Two combined rows follow. `Worst both:` covers the two distinct *resources*, GPU
and CPU; `Worst all:` covers all four measures, so a job that looks bad by every
measure rises -- at the cost that three of its four terms describe the same GPUs,
which weights GPU idleness 3:1 against CPU idleness. The measures are in different
units and cannot simply be added -- any exchange rate would be invented, and a
wrong one decides the ranking by itself -- so each job's waste is expressed as a
share of the selection's total waste in that measure and the shares are summed.
Each cell prints the job's **value** in every metric the row names
(`36337338:gpu0/sm0/pw70W/cpu1`), all of them under their cutoffs, which is what put
the job there. The order still carries the ranking. Printing the waste shares instead
was actively misleading: `12%gpu` reads as a utilization of 12%, the inverse of the
row's meaning.

**A combined row lists only jobs red in *every* measure it names.** `Worst both:`
means idle by GPU *and* CPU; `Worst all:` means idle by all four. That is what makes
them unarguable -- and it means they are often absent, which is itself the answer:
nothing was bad by every measure at once. A job that wastes GPU-time while keeping
its cores busy stays on `Worst GPU:` alone, where it belongs. The give-away for the
older behaviour was a `0%pw` term appearing in `Worst all:` -- a job on a
power-inclusive list that was not drawing idle power.

Only red jobs are candidates at all: a 95%-efficient job can idle 50 GPU-hours just
by being enormous, and there is nothing to act on there.

The cutoffs come from `[thresholds]` in your config -- the same ones that tint
the cells and colour `jobscope plot`, so the block is a tally of what you can
already see.

The **running** view averages each job over its own runtime by default, so it gets the
same resource-hour form as a finished selection (`Used/GPU-hr:`). Under `--instant`, or
past the job-count limit, it reports resource *counts* instead (`Used/GPU:`, and `USED` in
GPUs / cores / GB rather than hours): those numbers are one scrape at a single moment, so
weighting them by elapsed time would claim that instant represents the whole run.

The two job counts on `Jobs:` differ whenever the selection mixes CPU-only and GPU
work: a CPU-only job has no `GPU%` to pool, so it is absent from the GPU figures
rather than counted as zero. A GPU job that sat idle *is* counted, as 0%.
`no-runtime=N` appears when a job had no elapsed time to weight by. A metric's own
denominator is its table row, so the DCGM rows can legitimately cover more jobs
than `gpu-jobs=` -- a job with no stored summary still has Prometheus data.
`ENERGY_kWh` and `PWRmax_W` have no pooled form -- one is a per-job total and the
other a peak -- so they fall back to the plain per-job figure.
`jobscope plot` skips every footer rather than charting them as jobs.

The sections answer three different questions -- how was each metric used, how do
they compare, and which jobs are the problem -- and each is ruled to its own width.
Numbering runs over the sections actually printed: `--no-plot` leaves `1.` and `2.`
rather than a gap, and a single job has no problem-jobs section so it gets only the
first two. `--noheader` drops the headings and rules and keeps the data; `--csv` and
`--ts` get none of it, staying flat machine formats.

### The efficiency bars

Comparing eight utilization percentages by eye is what a bar chart is for, so the bars
are shown by default (`--no-plot` omits them). They are numbered as printed rather
than fixed, since `[report] sections` chooses which sections appear and in what
order:

```
2. Average efficiency  (filled = used, grey = idle)
---------------------------------------------------
  bars are the USED share above -- averaged over
    each job's whole runtime
     CPU%    6%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     MEM%   28%  ██████████░░░░░░░░░░░░░░░░░░░░░░░░
     GPU%   28%  ██████████░░░░░░░░░░░░░░░░░░░░░░░░
    GMEM%   70%  ████████████████████████░░░░░░░░░░
  SM_ACT%    6%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  TENSOR%    3%  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    DRAM%    3%  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

The value sits next to its label rather than after the bar, so the numbers line up in
a column instead of following a ragged edge.

The closing line says what the bars were averaged over: whether each job contributed its
newest scrape or a mean over its whole runtime, and whether the pooling weighted those by
the resources held now or by resource-time. "Average efficiency" does not say average over
what, and the two answers differ by a third on the same partition. `--noheader` drops it
with the other furniture.

Bar length is the pooled utilization and the filled run is tinted by its band, so
this is the `USED` column drawn rather than tabulated -- the bar and the `USED`
percentage are the same figure. It follows the table's metric set, so `--cpu`, `--gpu` and `--all-metrics`
narrow or widen it too, and it prints for a single job as well -- there it is that
job's profile across metrics.

`POWER_W` is the one table row with no bar. Its "used" is the time spent *above* the
watt floor, which is a detector reading rather than a fraction of a resource: on a
partition of GPUs idling at 119 W it fills to 100% beside `SM_ACT%` at 2%, reading as
the healthiest metric while describing the same idle GPUs. The table row says the
same thing without inviting that comparison.

Not emitted with `--csv` or `--ts`. Distinct from `jobscope plot`, which charts the
per-job CSV; this needs no pipe and no plotting libraries.

### A single job gets the table too

`jobscope <jobid>` prints the metric table for that one job, which is how you see
which band each of its numbers falls in -- the row itself gives the values but not
where they sit:

```
$ jobscope 35244230
35244230     alice        COMPLETED 1     11     3      1     78     2       64.3  ...
  Averaged over time: every value below is each job's mean over its whole runtime. Pooled across jobs by resource-time, so a
    10-hour job weighs ten times a 1-hour one.
  JOBS is how many jobs reported the metric and GOOD/OK/BAD sum to it, so the total differs per row wherever coverage does. Banded
    by GOOD above 20%, OK above 10%, BAD otherwise; POWER_W is GOOD above 100 W and BAD below, never OK.
  USED is the resource-time that did work, over what was allocated -- for POWER_W, the time spent above that watt floor.
  bands locate the waste and USED measures it: no red jobs but a low USED means every job wastes a little, rather than a few jobs
    wasting a lot
METRIC   USED         JOBS  GOOD  OK  BAD
CPU%     0.1h (11%)   1     0     1   0
MEM%     0.3GBh (3%)  1     0     0   1
GPU%     <0.1h (78%)  1     1     0   0
GMEM%    <0.1h (2%)   1     0     0   1
SM_ACT%  0.2h (64%)   1     1     0   0
OCC%     <0.1h (14%)  1     0     1   0
TENSOR%  <0.1h (3%)   1     0     0   1
DRAM%    <0.1h (9%)   1     0     0   1
```

One job, so each metric has a single `1` marking its band: this one used its GPU
and its SMs well, was middling on cores and occupancy, and barely touched the GPU
memory or the tensor cores. The rest of the block is suppressed, because for one job
the pooled row is that job's own row repeated, a `Worst` row names it again, and
every job count is 1.

- `--cpu` narrows to the host columns. For *finished* jobs that needs no Prometheus
  at all; a running job's `CPU%` comes from `cgroup_*`, so it does.
- `--gpu` narrows to the GPU columns and the profiling block.
- `--all-metrics` widens the profiling block to the full catalog.

### Highlighting

On a terminal, every `%` cell is tinted by how efficient it is -- **red** at or below
that metric's `inefficient` edge, **yellow** at or below its `improvement` edge,
**green** above -- so an idle job is a red row and a healthy one is green. The pooled
footer row is tinted too, and the same cutoffs define the band tallies, so a wasteful
selection is obvious at a glance and quantified one line below. `--per-gpu`'s rows are
graded by the same helper, so one GPU cannot read green in one table and red in the
other.

### `--per-gpu`: per-node charts and `--nodename`

Sixteen rows of twelve columns do not answer "which node is the slow one", so **for a
single job** each block ends with the efficiency chart repeated **per node**:

```
  Efficiency by node  (filled = used, grey = idle)
    holygpu8a15401                                            holygpu8a17203
         CPU%    7%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░            CPU%    6%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         GPU%   43%  ███████████████░░░░░░░░░░░░░░░░░░░            GPU%   13%  █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        GMEM%   70%  ████████████████████████░░░░░░░░░░           GMEM%   70%  ████████████████████████░░░░░░░░░░
      SM_ACT%    6%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░         SM_ACT%    6%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        DRAM%    3%  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░           DRAM%    3%  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
      ... one bar per graded column
```

Groups are packed side by side, **as many per row as the terminal fits**, up to four.
A block is about 55 characters, so 132 columns gives two, 171 gives three and 229 or
more gives four -- a four-node job then reads as a single row of bars. Off a terminal
the layout is fixed at two, so redirected output does not change shape with whatever
`$COLUMNS` happened to be.

A node's value is the mean over its GPU rows, which within a node *is* the pooled
figure. `CPU%` and `CPU-MEM` are already per-node figures repeated on each row, so
averaging leaves them unchanged.

**For more than one job the charts are replaced by one aggregate block** -- the same
`Summary by metric`, `Average efficiency` and `Problem jobs` sections the per-job view
prints, once, after the last listing. Per job they answer the same question a dozen
times over and leave no reading of the selection anywhere; a sweep wants "how is this
selection doing", and a single job wants "which of my nodes is slow".

The aggregate is over **jobs**, not over the printed unit rows, so `--per-node` and the
default view report identical summaries for one selection and differ only in row
granularity. It is text only: its CSV form would be new `Stat`/`Worst`/`Jobs` rows, and
`--per-gpu --csv` stays what it was.

**With one node in play the unit drops to the GPU**, since the card is then the only
thing distinguishing the rows -- either because the job ran on one node, or because
`--nodename` selected one:

```bash
jobscope -j 36441613 --per-gpu --nodename=holygpu8a10401   # 4 per-GPU charts
```

`--nodename` (or `--node`) filters the rows to that node and drops jobs that never
touched it. A name matching nothing is an error listing the nodes the selection *did*
touch -- an empty report would read as an idle node rather than a typo.

It applies to **`--ts` as well**, which carries the same `NODE` column:

```bash
jobscope -j 36441613 --ts --nodename=holygpu8a10401 | jobscope plot --kind line
```

There the filter runs *before* the queries, so one node of a four-node job costs a
quarter of the Prometheus range queries rather than fetching all four and discarding
three. The CSV schema is untouched, so it still pipes to `jobscope plot`; a name that
matches nothing writes no header at all, since a header with no rows under it reads as
an idle node and gives `plot` nothing to chart.

`--gpuid` narrows the other dimension, and both work on **every** view:

```
jobscope -j JOBID --nodename NODE                    # the summary, one node
jobscope -j JOBID --nodename NODE --gpuid 0,1        # ... and two of its cards
jobscope -j JOBID --nodename NODE --plot_ts --gpuid 0,1
jobscope -j JOBID --ts 30m --csv --gpuid 2
```

On `--per-gpu` and `--ts` they filter rows, which carry a node and a card. The
summary has no such row, so jobscope narrows the numbers it is computed *from* —
the stored summary is per node and per GPU already, so CPU% becomes that node's
CPU-seconds over its own cores, and GPU% the mean over the cards that remain. The
`NODE` and `#GPU` columns shrink with them.

That is how you find a straggler. On a two-node job:

```
$ jobscope -j 36770231                              GPU%=28   #GPU=8  NODE=2
$ jobscope -j 36770231 --nodename holygpu8a15401    GPU%=43   #GPU=4  NODE=1
```

— half the allocation was doing most of the work, which the whole-job 28 hides.

Because a narrowed summary otherwise looks exactly like a whole-job one, the header
says what was filtered:

```
  Node:      holygpu8a15401 only
  GPUs:      0, 1 only (per node)
```

GPU ids are per node — nodes number their cards from 0, so `--gpuid 0` on a two-node
job keeps two cards. Both filters run before the queries, and both name **every** id
that matched nothing rather than quietly reporting a subset: in `--gpuid 0,9` it is
the `9` you need told about. MIG instances are addressed as they print, `0.1`.

Note it is `--gpuid`, not `--gpu`: `--gpu` selects the GPU *columns* and takes no
value. Writing `--gpu 0,1` used to leave `0,1` to be read as a job ID; it is now an
error that points here.

The charts follow the view, so `--cpu` narrows them to `CPU%` and `--all-metrics` widens
them, and `--no-plot` omits them.

<a id="thresholds"></a>
The cutoffs are site-tunable in `[thresholds]`, and they are the same ones
`jobscope plot` grades with, so a job red in a chart is red in the table.

Every `%` metric is banded into five tiers by four edges, and both the edges and
the tiers are what `--ts --eff` reports:

| tier | default range | colour |
|---|---|---|
| `wasteful` | `< 2%` | red |
| `inefficient` | `2-10%` | red |
| `needs improvement` | `10-20%` | yellow |
| `average` | `20-40%` | green |
| `good` | `> 40%` | green |

**The edges are per metric**, because the metrics do not mean the same thing: a GPU
job legitimately holds cores it never uses, so `CPU%` at 4% is ordinary where `GPU%`
at 4% is idle, and `SM_ACT%` sits structurally below `GPU%` on the same work.
`default` covers every metric you do not name, which is what keeps the ~18 extra
columns under `--all-metrics` graded without listing them.

**There are two tables, one per view, and nothing is inherited between them:**
`[thresholds.summary]` grades the plain report (one average over each job's whole
elapsed runtime) and `[thresholds.timeslice]` grades `--ts` / `--plot_ts` /
`--eff` (samples pooled inside a window). A two-hour slice that catches a
checkpoint pause is not a two-hour idle job, so the two can want different bars. A
table you leave out keeps the built-in edges; it does not copy the other one, and
jobscope prints one note when you have set only one of them.

```toml
[thresholds.summary.wasteful]
default = 2
cpu     = 5      # a GPU job idling its cores is normal; 2% would flag them all
sm_act  = 3
[thresholds.timeslice.wasteful]
default = 2
cpu     = 8      # a short window dips further than a whole-job average

[thresholds]
power_w = 100    # POWER_W, in WATTS -- below this a GPU counts as idle
```

Metric names are the column header, lowercase and without the `%` -- `gpu`, `cpu`,
`sm_act`, `dram`. Edges must not decrease within a metric once resolved against the
defaults it falls back to, and jobscope rejects a config where they do rather than
grade by a band nothing can reach.

`POWER_W` has no bands at all, just a floor, because watts are not a percentage --
and it is the one idle signal a duty cycle cannot fake, since a job spinning on a
trivial kernel reads busy on `GPU%` while drawing idle watts. It is shared by both
views: idle draw is a property of the hardware.

Run `jobscope config` to print both tables as they actually resolve.

### Which metrics, and what colour

Two more sections cover what the report *shows* rather than how it grades.

`[metrics]` picks the GPU/DCGM metrics per view — `summary` for the per-job table's
profiling block, `timeseries` for `--ts`/`--plot_ts`/`--eff`, and `extended`
for what `--all-metrics` widens to. Name them by their short name, the same ones
`[thresholds]` takes:

```toml
[metrics]
summary    = ["sm_act", "tensor", "dram", "power"]
timeseries = ["gpu", "sm_act", "tensor", "dram", "power"]
extended   = "all"
```

Order does not matter — columns always print in catalog order.
`jobscope describe --all-metrics` lists the catalog with descriptions.

Two things it deliberately does not reach. **`--per-gpu` keeps a fixed four**
(`SM_ACT%`, `TENSOR%`, `DRAM%`, `POWER_W`): its rows are addressed by position, so
its width is not free. And **the CPU side is fixed** at `CPU%`/`MEM%`, because there
are exactly two cgroup queries and no catalog to choose from. Naming a jobstats-backed
metric (`gpu`, `mem`) in `summary` is harmless — `GPU%` and `GMEM%` have their own
columns already — and leaving one out is corrected rather than obeyed, since in the
running view those columns come from Prometheus and would otherwise read `-`.

`[colors]` gives each classified tier a colour, read by both the tables and
`jobscope plot`, so a job is the same colour in either:

```toml
[colors]
wasteful     = "bright_red"
inefficient  = "red"
improvement  = "yellow"    # the "needs improvement" tier
average      = "cyan"
good          = "blue"
long_running = "magenta"   # an entry that ran past [defaults] long_running
```

Takes the eight ANSI names and their `bright_` variants, or `color(N)` for the
256-colour cube. Two tiers sharing a colour is the default, not a requirement — give
all five distinct values for a colourblind-safe palette.

The summary table's `RED`/`YELLOW`/`GREEN` columns keep those names whatever you
set, and so do the `red=`/`yellow=`/`green=` fields of the `--csv` output: those are
the three band *counts*, so a script reading them does not break when you recolour
the display.

`[report] sections` chooses which parts of the block below the job table print, and
in what order — `metrics` (Summary by metric), `efficiency` (the utilization bars)
and `problems` (the Wasteful rows). Leave one out to drop it; list them differently
to reorder. Sections are numbered as printed, so the numbers follow your order.

```toml
[report]
sections = ["problems", "metrics"]   # lead with the jobs, skip the bars
```

`[plot]` sets the chart defaults, shared by `jobscope plot` and `--plot_ts` so a
chart looks the same whichever way it was drawn:

```toml
[plot]
metrics  = ["gpu", "sm_act", "dram"]   # what a time series draws without --metric
palette  = [196, 46, 33, 208]          # one 256-colour code per series, cycled
max_rows = 40                          # heatmap row cap (--max-rows overrides)
panels   = 12                          # side-by-side panels before --by drops some
```

`metrics` takes the same short names as `[metrics]` and `[thresholds]`, or plain
column headers. A name the CSV does not carry is skipped rather than charted empty —
which is how one list serves files with different columns.


## Utilities

| Command | Purpose |
|---|---|
| `jobscope plot` | render `--csv` output as a terminal chart |
| `jobscope describe` | plain-English column and metric reference (`--metrics` for the catalog) |
| `jobscope config` | show the config path or print an example |
| `jobscope probe` | check what this cluster exposes and whether jobscope can read it |


## Running jobs

`jobscope` with no arguments answers "what is happening on the GPUs *now*". It
selects from `squeue` and, by default, reports the newest single scrape -- so
unlike the historical modes it is a snapshot, not a job-length average.

```bash
jobscope                          # your running jobs over 1h
jobscope -j 12345_6               # one running job or array element
jobscope -p kempner -a            # every user in a partition
jobscope --min-elapsed 0s         # no runtime floor at all
jobscope --instant                # the newest scrape instead of the runtime average
jobscope --per-gpu                # one row per GPU
jobscope --ts -j 12345 | jobscope plot
```

This is why the average is the default. A snapshot lands wherever the job happens to be,
so it will **not** match jobstats on a bursty job -- one that alternates compute with gaps
is genuinely bimodal, and a single scrape can read `GPU% 0` on a GPU averaging ~88%. Reach
for `--instant` when you want the fast query or the current moment specifically, and `--ts`
to see the phases themselves.

`CPU%`/`MEM%` are cumulative by nature -- CPU-seconds over elapsed x cores, and
peak RSS -- so they read the same in both modes; only the GPU columns follow the
average-versus-`--instant` choice.

On a MIG node `--per-gpu` and `--ts` show the instances; the DCGM columns read
`-` there, because NVML identifies an instance by a `MIG-…` UUID where DCGM
reports the physical `GPU-…` one and nothing in the metrics maps between them.


## Useful commands

A working set, in the order you would reach for them.

```bash
# what is running right now
jobscope                                          # your jobs
jobscope -p kempner_eng -a                        # everyone on a partition

# what already ran
jobscope finished                                 # your last day
jobscope finished -D 1                            # the same, explicitly
jobscope finished -S 2026-07-26 -p kempner_eng    # a fixed window on one partition
jobscope -p kempner_eng -a --runtime-avg          # running, averaged over each runtime
                                                  # (running-only: a finished job is
                                                  #  always averaged already)

# one job in detail
jobscope -j 36499551_64                           # summary, with efficiency bars
jobscope -j 36612315 --plot_ts                    # its metrics charted over time
jobscope -j 36612315 --plot_ts 60m                # the same, last hour only
jobscope -j 36441613 --ts 60m --stats job     # the last hour, averaged

# a whole partition, triaged
jobscope -p kempner_h100 -a --ts 60m --eff
jobscope -p kempner_h100 -a --ts 10m --eff --csv > triage.csv
```

The window goes on whichever flag you are already using -- `--plot_ts 60m`, not
`--ts 60m --plot_ts`, since `--plot_ts` *is* `--ts` with the chart in place of the CSV
and the two are mutually exclusive.

Two further notes. `--ts WINDOW` narrows the Prometheus queries rather than filtering
rows, so a short window over a busy partition is cheap -- 145 jobs in about 16s. And
`--eff` implies `--stats job`, so the two do not need to be given together.


## Plotting

`jobscope plot` renders `jobscope <view> --csv` output as terminal bar gauges,
histograms, heatmaps, and time-series line charts. The chart kind is
auto-detected from the CSV columns; override with `--kind`.

Time-series charts show `GPU%`, `GMEM%`, `SM_ACT%`, `OCC%`, `TENSOR%` and `DRAM%` by
default; `--metric` picks specific columns and `--all` charts every numeric one.
`GPU%` and `GMEM%` lead because they are the *resources* -- how busy the card is and
how full -- where `OCC%`/`TENSOR%`/`DRAM%` describe how the SMs were used, which only
means something once the GPU is known to be busy. `GMEM%` also catches a failure none
of the others do: `GPU%` 96 with `GMEM%` 3 is under-batched.

```bash
jobscope --gpu  --csv JOBID      | jobscope plot                 # bar gauges (one job)
jobscope --gpu  --csv -D 7       | jobscope plot --kind hist      # distribution (many jobs)
jobscope finished --all-metrics --csv -D 7 | jobscope plot              # heatmap (jobs/GPUs x metrics)
jobscope JOBID --ts             | jobscope plot --compact        # time series
```

### Which time-series layout

`--by metric` gives one panel per variable, each with its own y-axis, which is what a
mixed set needs: on a typical job `GPU%` spans 0-100 where `OCC%` spans 0-21 and
`DRAM%` 0-17, so a shared axis crushes three of them into the bottom sixth. `--by gpu`
(the default) puts every metric on one shared axis per GPU, and `--compact` gives one
sparkline row per series.

**`--gpu` takes a comma list, and each GPU named becomes a column.** So four cards on a
node read side by side, one row per metric:

```bash
jobscope -j 36441613 --nodename holygpu8a10501 --ts \
  | jobscope plot --by metric --gpuid 0,1,2,3
```

`--columns` asks for the same grid without naming the cards. And since every part of
that pipeline after `--ts` is mechanical -- the CSV already says how many GPUs there
are -- **`--plot_ts` is the whole thing in one command**:

```bash
jobscope -j 36441613 --nodename holygpu8a10501 --plot_ts
jobscope -j 36606149 --plot_ts        # single-node job: no --nodename needed
```

It *is* `--ts`, with the CSV charted rather than written, so the schema, `--step`,
the window below and the `--nodename` filter all behave the same.

#### Overlaid instead: `--plot-ts-overlay`

`--plot_ts` gives each metric its own panel and its own y-axis, which answers "how did
this one move". The overlay answers the other question -- "did these move together" --
by putting every metric on one shared axis, a panel per GPU, and a row per node:

```bash
jobscope -j 36770231 --plot-ts-overlay        # 2 nodes x 4 GPUs = two rows of four
```

It needs no `--nodename`: a row per node is the layout, so the guard `--plot_ts` raises
on a multi-node job does not apply. It is exactly `jobscope plot --by gpu --columns` on
the same CSV, and delegates to it.

Two consequences of the shared axis, both stated in the output:

- **Watts are omitted.** `POWER_W` against percentages means a 400 W line pins the
  scale and flattens every percentage onto the floor. `--plot_ts` is where watts get
  their own panel.
- **Each panel carries its own legend**, naming the metrics where they are drawn.
  plotext draws it inside the axes and one row per label, so the panels need more room
  than the per-metric grid's: they get a wider floor (fewer abreast on a narrow
  terminal, wrapped) and a height that grows with the metric set. Six metrics need
  eleven rows; at ten plotext drops the legend silently, which is why the height is not
  fixed. With `--no-color` the markers are identical, so the legend names the traces
  without distinguishing them -- `--plot_ts` is the one to use without colour.

#### A window: `--ts 1h`

Both flags take an optional window, and give the **last** N of the run:

```bash
jobscope -j 36441613 --nodename holygpu8a10501 --ts 1h        # 244 rows, not 5632
jobscope -j 36441613 --nodename holygpu8a10501 --plot_ts 30m
```

The window lands on the run's own sample grid, so `--ts 1h` returns *exactly* the rows
a full `--ts` would have -- fetching a window and slicing a whole series agree. That
alignment is not free: Prometheus anchors a range query's points at its start, so an
unaligned window relabels every sample, and because a running job's end is "now" its
grid would drift with the clock between invocations.

It narrows the range *queries*, not the rows afterwards, so an hour of a day-long job
costs a twenty-fourth of the samples to fetch -- and the step is measured over the span
actually queried, so a window keeps the native scrape resolution where the whole run
would have been coarsened. The chart names the window it drew, since the x axis counts
minutes from the window's own start either way.

#### Averages over the window: `--stats`

`--ts` gives every sample; `--stats` gives what they add up to, one row per GPU per
metric:

```console
$ jobscope -j 36612315 --ts 60m --stats
  NODE:GPU          METRIC    N    MIN   MEAN    MAX   LAST
  holygpu8a17601:2  GPU%     61   17.0   20.6   22.0   21.0
  holygpu8a17601:2  SM_ACT%  61    1.3    1.3    1.6    1.3
  holygpu8a17601:2  POWER_W  61  119.0  119.9  120.0  120.0
```

Computed from the samples `--ts` already fetched, so there are **no extra queries** and
the numbers cannot disagree with the series they summarize. `N` is the sample count,
which is worth a glance: it says whether the window actually had data. `MEAN` is tinted
by its band, as `USED` is in the summary table, and `--csv` emits the same rows for
scripting.

`--stats node` pools the job's GPUs on each host, and `--stats job` pools
every card it held:

```console
$ jobscope -j 36441613 --ts 20m --stats node
  NODE            GPUS  METRIC    N    MIN   MEAN    MAX   LAST
  holygpu8a10302  4     GPU%     84   17.0   93.1  100.0  100.0

$ jobscope -j 36441613 --ts 20m --stats job
  JOBID     NODES  GPUS  METRIC     N    MIN   MEAN    MAX   LAST
  36441613  4      16    GPU%     336    0.0   93.4  100.0  100.0
```

Each level names what it pooled -- a node mean over one GPU and over sixteen must not
read the same. Pooling is over the *samples*, not over per-GPU means, so a card the
exporter missed for half the window carries half the weight instead of counting as a
full peer. `JOBID` leads the narrower levels only when the series covers more than one
job.

Note this is a plain mean of samples, not each metric's own reducer -- the tables peak
memory where this averages it. That is the honest reading of "the average over this
window". `--plot_ts` already prints the same figures under its charts, so `--stats`
adds nothing there and says so.

#### Triaging a partition: `--eff`

`--stats job` over a partition is 145 jobs x 8 metrics and no verdict.
`--eff` sorts them instead:

```console
$ jobscope -p kempner_h100 -a --ts 10m --eff --all-metrics
  142 jobs, by best of GPU%, SM_ACT%, OCC%, TENSOR%, DRAM% (POWER_W caps the verdict when idle)

  wasteful (GPU% <2%, SM_ACT% <2%, OCC% <2%, TENSOR% <2%, DRAM% <2%)  15 jobs
    JOBID       NODES  GPUS  USER   GPU%  SM_ACT%  OCC%  TENSOR%  DRAM%  POWER_W
    36229482    1      2     bob     0.0      0.0   0.0      0.0    0.0       70
    36438938_1  1      1     carol   0.0      0.0   0.0      0.0    0.0      118
  inefficient (best of GPU%, SM_ACT%, OCC%, TENSOR%, DRAM%: 2-10%)  4 jobs
    ...
  good (best of GPU%, SM_ACT%, OCC%, TENSOR%, DRAM%: >40%)  98 jobs
    (--eff all to list them)
```

Each heading states the rule it applied, so the criteria never has to be looked up
in the config. The default edges:

| category | best metric |
|---|---|
| wasteful | `< 2%` |
| inefficient | `2-10%` |
| needs improvement | `10-20%` |
| average | `20-40%` |
| good | `> 40%` |

These come from `[thresholds.timeslice]` and are per metric, so a heading can show
`GPU% <2%, CPU% <5%` where a site has set them apart -- and the ranges collapse back
to one, exactly as above, wherever the metrics agree. See
[Thresholds](#thresholds).

The edges are not uniform, so they are worth stating: "below 2%" *excludes* 2, while
every band above it *includes* its top. 10.0 is inefficient; 10.1 needs improvement.

**A job is judged on its best metric**, which is the same rule as "every metric is
below X" read from the other end -- the AND the `Worst all` row uses, generalised to
five bands rather than a second notion of idle. `GMEM%` sits out: reserving 80GB and
computing nothing is still computing nothing. The header names the metrics actually
used, since `--all-metrics` widens the set.

**`POWER_W` plays no part in the category.** It is graded on its own terms, in watts
against a floor that depends on the card -- see below -- and folding a per-model
quantity into a rule expressed in percent could only be done by picking one number for
every architecture, which is the thing that does not work.

`good` collapses to a count by default, since on a healthy partition it is most of the
output and none of the point; `--eff all` lists it. `--stats node`
grades hosts on the same rule.

The rows are aligned columns under a header, and the identity block names whatever
was judged -- `JOBID` per job, `NODE` under `--stats node`, `NODE:GPU` per card.

For storing or post-processing, `--csv` gives one row per job -- id, user, every metric
the series carried, and the label last:

```console
$ jobscope -p kempner_h100 -a --ts 10m --eff --csv
JOBID,USER,GPU%,SM_ACT%,OCC%,TENSOR%,DRAM%,POWER_W,GMEM_GB,GMEM%,LABEL
36229482,bob,0.0,0.0,0.0,0.0,0.0,69.5,0.5,0.6,wasteful
36638420_2,dana,77.5,76.3,30.5,40.0,41.1,587.9,60.5,76.0,good
```

It carries `GMEM%` and `POWER_W` even though neither votes on the label: a row you are
going to sort or join on should say what was measured. Nothing names the partition,
because `-p` already fixed it for every row.

**A window needs its unit** -- `1h`, `90m`, `30s`, `2d`, the same vocabulary
`--min-elapsed` uses. That is what keeps `jobscope --ts 36441613` working: a job ID
never carries a unit, so a bare number after the flag is handed back as the job it
looks like, with a note saying so. `--ts 60` is therefore *job 60*; write `--ts 60m`
for an hour. It charts one job on one node, and says
so rather than guessing: several nodes without `--nodename` names them and asks for
one, and several jobs points at `-j`. That second guard matters because the chart
would otherwise be quietly wrong -- series key on `(node, GPU)`, so two jobs that
shared a card would join into one line.

```
GMEM%
              gpu0                        gpu1                        gpu2
    ┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐
88.8┤▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│88.8┤▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│88.8┤▛▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀│
```

Only when they are named. Without `--gpu` the cards stay overlaid as lines in one
panel, which is the view that answers "did one of them diverge" -- and a grid of six
metrics by four GPUs is 24 panels, better asked for than arrived at. As many columns
are drawn as the terminal fits; below that the rest wrap onto a further row, the same
way the `--per-gpu` charts pack.

`jobscope plot` reads `summary`, `dcgm`, and `dcgm --ts` CSV; the `detail` CSV is
for machine consumption, not charts. Do not pass `-n` when piping to `jobscope
plot`: the plot needs the CSV header row.
