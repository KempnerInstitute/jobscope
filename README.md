# jobscope

`jobscope` reports the efficiency of completed Slurm jobs. It decodes the
utilization data Slurm already stores in each job's `sacct` AdminComment
(CPU / memory / GPU / GPU-memory), enriches GPU jobs with DCGM profiling metrics
pulled from Prometheus, and can render any view as a terminal chart.

For completed jobs the CPU/MEM/GPU/GMEM numbers match `jobstats`, because
jobscope decodes the same stored blob, but in one bulk `sacct` query, with no
per-job calls and no job-count cap.

## Screenshots

<table width="800">
  <tr><td><strong>Per-job DCGM time series</strong></td></tr>
  <tr><td><img src="https://raw.githubusercontent.com/KempnerInstitute/jobscope/main/docs/timeseries.svg" alt="per-job DCGM time series" width="800"></td></tr>
  <tr><td><strong>Aggregated utilization across jobs</strong></td></tr>
  <tr><td><img src="https://raw.githubusercontent.com/KempnerInstitute/jobscope/main/docs/aggregated.svg" alt="aggregated mean-utilization bars" width="800"></td></tr>
</table>

```text
$ jobscope finished -S 2026-07-26 -E 2026-07-27
  User:      bdesinghu
  Select:    2026-07-26 .. 2026-07-27
JOBID        USER         STATE     NODE  CPU%   MEM%   #GPU  GPU%   GMEM%   SM_ACT%  OCC%    TENSOR%  DRAM%   POWER_W  RUNTIME
------------------------------------------------------------------------------------------------------------------------------------
35244230     bdesinghu    COMPLETED 1     11     3      1     78     2       64.3     14.4    2.9      8.9     406      00:09:30
35246690     bdesinghu    COMPLETED 1     11     2      1     63     3       44.3     8.8     0.2      4.3     338      00:12:07
35246691     bdesinghu    COMPLETED 1     11     3      1     67     2       43.8     8.7     0.2      4.4     341      00:13:05
35260825     bdesinghu    COMPLETED 1     18     2      1     20     1       3.3      0.5     0.0      0.4     106      00:04:18
35289592     bdesinghu    COMPLETED 1     13     12     1     95     3       83.9     20.6    0.2      15.4    580      02:08:33
                                            ... 12 more rows ...
------------------------------------------------------------------------------------------------------------------------------------
Used/GPU-hr:                              12     9            80     3       64.9     14.7    0.3      9.5     455
  red below 10%, yellow below 20%, green above; POWER_W red below 100 W. Counts are jobs.
  bands catch pathological jobs, IDLE measures efficiency: no red with a high IDLE means every job wastes a little
METRIC   IDLE            RED  YELLOW  GREEN
CPU%     49.3h (88%)     0    17      0
MEM%     534.4GBh (91%)  15   1       1
GPU%     1.4h (20%)      0    0       17
GMEM%    6.9h (97%)      17   0       0
SM_ACT%  2.5h (35%)      1    0       16
OCC%     6h (85%)        6    10      1
TENSOR%  7h (100%)       17   0       0
DRAM%    6.4h (91%)      14   3       0
Worst SM (1/17): 35260825 0.1h@3% bdesinghu
Jobs:            cpu-jobs=17  gpu-jobs=17  gpus=17
```

## Requirements

- Python 3.9+
- Slurm with `sacct`, where the jobstats-style AdminComment blob is populated
  (needed by every view).
- A Prometheus endpoint serving the DCGM (`DCGM_FI_*`), `nvidia_gpu_*` and
  `cgroup_*` series that jobstats scrapes. Needed for the GPU columns, and for any
  running job (whose blob does not exist yet). `finished --cpu` never contacts it.

## Install

jobscope is a command-line tool, so install it with
[`uv`](https://docs.astral.sh/uv/). `uv` provisions its own Python and puts the
`jobscope` executable on your `PATH`; there is no virtualenv to create or
activate, and it never touches the system Python (which on clusters like
FASRC/RHEL8 is too old to build `pyproject.toml` projects anyway).

Install `uv` once, if you don't already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell so `uv` is on your `PATH`. (Alternatives: `wget -qO-
https://astral.sh/uv/install.sh | sh`, `pipx install uv`, or `brew install uv`;
see the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/).)

Now install jobscope:

```bash
uv tool install jobscope      # from PyPI
uv tool install .             # from a source checkout
```

Upgrade or remove it later with `uv tool upgrade jobscope` or `uv tool uninstall
jobscope`.

For development from a checkout there's nothing to install; `uv` runs
everything straight from the source tree, creating the environment on demand:

```bash
uv run jobscope -D 3          # run the CLI from source
uv run --extra dev pytest     # run the test suite
```

## Configuration

The GPU columns, and every column for a running job, need a Prometheus endpoint
serving the DCGM, `nvidia_gpu_*` and `cgroup_*` series. (`finished --cpu`, and
everything under `describe` and `config`, need nothing.) Provide it one of these
ways.

```bash
# Preferred: environment variable (keeps a credential out of any file)
export JOBSCOPE_PROM_URL="https://USER:TOKEN@prometheus.example.net/api/prom"
```

**Kempner AI Cluster users:** the jobstats `config.py` sits beside the `jobstats`
binary on your `PATH`, and jobscope **auto-discovers it** when nothing else is
configured, so you need no config file and never handle the URL or token. Just
run `jobscope`. (This works at any jobstats site; to point at a different install,
set `site_jobstats_config_path` in the config file below.)

On any other cluster, put your settings in that same config file
(`~/.config/jobscope/config.toml`, or wherever `$JOBSCOPE_CONFIG` points):

```bash
jobscope config --example > ~/.config/jobscope/config.toml   # then edit it
jobscope config                                              # show the path in use
```

The config file also sets the DCGM sampling period, plot color thresholds, and
default timeout / worker counts; see `jobscope config --example` for the full,
commented template.

The Prometheus URL commonly embeds a credential: jobscope never prints it, and a
`config.toml` in a repo checkout is git-ignored. On sites already running
jobstats, `site_jobstats_config_path` reuses that install's `PROM_SERVER`, so the
secret is never copied.

## Quick start

```bash
jobscope                          # your running jobs, right now (the default)
jobscope -p kempner -a            # everyone on a partition, right now
jobscope finished -D 3            # your finished jobs over the last 3 days
jobscope 30012345                 # one job, running or finished
jobscope running --hwdetail       # per-GPU rows instead of per-job
jobscope finished -D 7 --dcgm     # the full DCGM metric catalog
jobscope 30012345 --ts | jobscope plot --compact    # time-series chart
```

## The argument tree

One axis per level, so every option composes with every selection:

```
jobscope [MODE] [scope] [filters] [granularity] [columns] [--diagnose] [output]
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

**Filters**, every mode: `-p` partition, `-u` user, `-a` all users, `-A` account,
`-t` state (`finished` only), `--min-elapsed` runtime floor (`running` only,
default 10m -- a job still loading data reads as idle; `[defaults] min_elapsed`
changes it, `0s` disables it).

**Level 3 — granularity** (pick one) and **columns**:

| option | effect |
|---|---|
| *(default)* | one row per job |
| `--hwdetail` | one row per GPU, with node name and GPU number |
| `--ts` | the per-scrape time series as CSV |
| `--cpu` / `--gpu` | narrow the columns to one resource |
| `--dcgm` | the full DCGM metric catalog |
| `--avg` | `running` only: fold over the runtime instead of a snapshot |

**Level 4** — `--diagnose`, which adds the advisory `DIAG` column at the end.

**Output** — `--csv`, `-n`, `--step` (with `--ts`), `--timeout`, `--workers`, `-c`.

Flags and JOBIDs may be given in any order. A `JOBID` works whether the job is
running or finished: Slurm only stores the utilization blob when a job *ends*, so
for a running one jobscope reconstructs `CPU%`/`MEM%`/`GPU%`/`GMEM%` from the same
Prometheus metrics jobstats falls back to. With no Prometheus endpoint configured
those columns stay blank and say so.

`jobscope running -j ID` differs from `jobscope ID`: the first reads the live view
of that job (an instant snapshot, with `--avg` available), the second looks it up
through `sacct` over its window.

## Columns

Every per-job view prints the same columns, so a job reads identically whether it
has finished or is still running:

```
JOBID  USER  STATE  NODE  CPU%  MEM%  #GPU  GPU%  GMEM%  SM_ACT%  OCC%  TENSOR%  DRAM%  POWER_W  RUNTIME
```

`NODE` is the node count and `#GPU` the allocated GPU count. `CPU%`/`MEM%` sit
beside `SM_ACT%` deliberately: a GPU job whose `GPU%` is low and `CPU%` is high is
held up on the host, and no single view used to show both.

With more than one job the table ends in footers:

```
Used/GPU-hr:                              5      4            43   30     37.5  11.4  9.9  8.2  272
  red below 10%, yellow below 20%, green above; POWER_W red below 100 W. Counts are jobs.
  bands catch pathological jobs, IDLE measures efficiency: no red with a high IDLE means every job wastes a little
METRIC   IDLE            RED  YELLOW  GREEN
CPU%     10034.5h (95%)  159  158     2
MEM%     129.6TBh (96%)  307  8       4
GPU%     408.8h (54%)    5    8       306
GMEM%    528.9h (70%)    299  9       11
SM_ACT%  452h (60%)      35   17      373
OCC%     660.6h (88%)    126  269     30
TENSOR%  672.3h (89%)    414  9       2
DRAM%    688.3h (91%)    389  18      18
Worst GPU (5/319):    35475803 142.9h@0% amazloumi  35476814 142h@0% amazloumi  36337781 47.7h@0% amazloumi
Worst SM (35/425):    35475803 142.9h@0% amazloumi  35476814 142h@0% amazloumi  36337781 47.7h@0% amazloumi
Worst POWER (41/425): 35475803 142.9h@73W amazloumi  35476814 142h@73W amazloumi  36337781 47.7h@73W amazloumi
Worst CPU (159/319):  35475803 2286.6h@0% amazloumi  35476814 2271.9h@0% amazloumi  36337781 763.7h@0% amazloumi
Worst both (159):     35475803 35%gpu+23%cpu  35476814 35%gpu+23%cpu  36337781 12%gpu+8%cpu
Worst all (210):      35475803 35%gpu+32%sm+40%pw+23%cpu  35476814 35%gpu+31%sm+40%pw+23%cpu
Jobs:                 cpu-jobs=319  gpu-jobs=319  gpus=381  no-runtime=5
```

**There is no per-job mean**, on purpose. Utilization is bimodal -- jobs cluster
near 0% or near 100% -- so an average of them describes a job that does not
exist. On the day above `GPU%` averaged 82% per job while the GPUs were 56% idle.

`Used/GPU-hr:` is a ratio rather than a centre: used resource-time over allocated
resource-time, aligned under the columns above it.

**One table row per graded metric**, and the set follows the view: eight by
default, `CPU%`/`MEM%` under `--cpu`, six under `--gpu`, the full catalog under
`--dcgm`. A metric no job reported is left out rather than shown as zeros. Each
metric is measured against the resource it is a percentage *of*:

| metric | resource |
|---|---|
| `CPU%` | allocated core-hours |
| `MEM%` | allocated host GB-hours |
| `GPU%`, `GMEM%`, and every DCGM column | allocated GPU-hours |

So the denominators differ by row on purpose, and reading down the `IDLE` column
is the fastest way to see which resource a selection actually wasted: above, the
GPUs were 56% idle while the *cores* were 95% idle and the *tensor cores* 90%.

One cutoff covers every metric -- **red below 10%, yellow below 20%, green above**
-- so there is no per-row threshold to carry and no cutoff column. `POWER_W` is red
below 100 W. The table prints a two-line legend saying so.

The band cells are plain job counts. Their resource shares stay in the `--csv`
output for anyone scripting them.

**`green` means "not pathological", not "efficient".** With a red cutoff of 10 a
job at 21% is green while wasting four fifths of its cores, so a selection can be
half idle with nearly every job green:

```
METRIC   IDLE            RED  YELLOW  GREEN
CPU%     84.3 (49%)      0    1       13
```

That is not a contradiction: every job there used about half its cores, so none is
below 10, yet half the allocation went unused. Read `IDLE` for efficiency and the
bands for *where* the waste is:

| pattern | meaning | what to do |
|---|---|---|
| red band holds a large **resource** share | a few jobs waste a lot | go find those jobs -- they are in `Worst` |
| no red band but a high `IDLE` | every job wastes a little | nothing to escalate; over-requesting is the habit |

On the partition above, `GPU%` is the first pattern (4% of jobs holding 53% of the
GPU-hours in red) and `CPU%` the second.

The three band cells give each band's share of the **jobs** and of the
**resource-time** (`13 (4%)/54%` is 13 jobs, 4% of the jobs, holding 54% of the
GPU-hours). The gap between those two numbers is the finding, and either alone
conceals it. On a terminal the band cells are printed in their own colours and
`IDLE` is tinted by that metric's pooled grade, so a wasted resource is a red
line in the block.

The `Worst` rows name the offenders, ranked by resource-time *wasted* rather than
held, so a long job at a mediocre rate outranks a short one at zero. There is one
row per measure -- duty cycle, SM residency, board watts, and the host -- and any
row with no red job is omitted.

`POWER_W` is graded in **watts**, not percent: below `[thresholds] power_w`
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
Every component is printed (`35%gpu+32%sm+40%pw+23%cpu`), so you can see which
measure put a job on the list. Only jobs red in at least one measure are
candidates: a 95%-efficient job can idle 50 GPU-hours just by being enormous, and
there is nothing to act on there.

The cutoffs come from `[thresholds]` in your config -- the same ones that tint
the cells and colour `jobscope plot`, so the block is a tally of what you can
already see.

The **running** view reports the same table over resource *counts* rather than
resource-hours (`Used/GPU:`, and `IDLE` in GPUs / cores / GB rather than hours).
Its numbers are one scrape at a single moment, so weighting them by elapsed time
would claim that instant represents the whole run; `--avg` folds each job over its
runtime and does get the hour-based form.

The two job counts on `Jobs:` differ whenever the selection mixes CPU-only and GPU
work: a CPU-only job has no `GPU%` to pool, so it is absent from the GPU figures
rather than counted as zero. A GPU job that sat idle *is* counted, as 0%.
`no-runtime=N` appears when a job had no elapsed time to weight by. A metric's own
denominator is its table row, so the DCGM rows can legitimately cover more jobs
than `gpu-jobs=` -- a job with no stored blob still has Prometheus data.
`ENERGY_kWh` and `PWRmax_W` have no pooled form -- one is a per-job total and the
other a peak -- so they fall back to the plain per-job figure.
`jobscope plot` skips every footer rather than charting them as jobs.

- `--cpu` narrows to the host columns. For *finished* jobs that needs no Prometheus
  at all; a running job's `CPU%` comes from `cgroup_*`, so it does.
- `--gpu` narrows to the GPU columns and the profiling block.
- `--dcgm` widens the profiling block to the full catalog.
- `--diagnose` appends the advisory `DIAG` column.

### Highlighting

On a terminal, every `%` cell is tinted by how efficient it is -- **red** below the
threshold, **yellow** below twice it, **green** above -- so an idle job is a red row
and a healthy one is green. The pooled footer row is tinted too, and the same
cutoffs define the band tallies, so a wasteful selection is obvious at a glance
and quantified one line below.

The cutoffs are per column and site-tunable in `[thresholds]`, and they are the
same ones `jobscope plot` grades with, so a job red in a chart is red in the table:

```
red = 10       every %-metric: red below 10, yellow below 20, green above
power_w = 100  POWER_W, in WATTS -- below this a GPU counts as idle
```

One cutoff rather than one per metric: a reader should not have to carry a different
threshold for each row of the summary table, and the old per-metric values were
never calibrated against each other. `POWER_W` is the exception because watts are
not a percentage -- and it is the one idle signal a duty cycle cannot fake, since a
job spinning on a trivial kernel reads busy on `GPU%` while drawing idle watts.

`[thresholds] gpu`, `gmem`, `cpu`, `mem` and `default` no longer do anything. A
config that still sets them prints one note saying so rather than silently changing
your cutoffs.

Colour is dropped automatically when the output is not a terminal, with `--csv`,
under `$NO_COLOR`, or with `--no-color` -- escape codes in a redirected file are
corruption, not decoration. For *why* a job is inefficient rather than just that it
is, add `--diagnose`, which tags each job `idle` / `underfed` / `low-occ` /
`mem-bound` / `no-tensor` / `ok`.

## Utilities

| Command | Purpose |
|---|---|
| `jobscope plot` | render `--csv` output as a terminal chart |
| `jobscope describe` | plain-English column and metric reference (`--dcgm` for the catalog) |
| `jobscope config` | show the config path or print an example |

## Running jobs

`jobscope` with no arguments answers "what is happening on the GPUs *now*". It
selects from `squeue` and, by default, reports the newest single scrape -- so
unlike the historical modes it is a snapshot, not a job-length average.

```bash
jobscope                          # your running jobs over 1h
jobscope -j 12345_6               # one running job or array element
jobscope -p kempner -a            # every user in a partition
jobscope --min-elapsed 0s         # no runtime floor at all
jobscope --avg                    # fold over each job's runtime (= jobstats)
jobscope --hwdetail               # per-GPU rows
jobscope --ts -j 12345 | jobscope plot
```

Because a snapshot lands wherever the job happens to be, it will **not** match
jobstats on a bursty job -- one that alternates compute with gaps is genuinely
bimodal, and a single scrape can read `GPU% 0` on a GPU averaging ~88%. Use
`--avg` for a jobstats-comparable number, or `--ts` to see the phases themselves.

`CPU%`/`MEM%` are cumulative by nature -- CPU-seconds over elapsed x cores, and
peak RSS -- so they read the same in both modes; only the GPU columns follow the
instant-versus-`--avg` choice.

On a MIG node `--hwdetail` and `--ts` show the instances; the DCGM columns read
`-` there, because NVML identifies an instance by a `MIG-…` UUID where DCGM
reports the physical `GPU-…` one and nothing in the metrics maps between them.

## Moving from the old subcommands

The old positional subcommands are deprecated and print a note, but still work:

| was | now |
|---|---|
| `jobscope summary -D 3` | `jobscope finished -D 3` |
| `jobscope detail JOBID` | `jobscope JOBID --hwdetail` |
| `jobscope dcgm --ext JOBID` | `jobscope JOBID --dcgm` |
| `jobscope dcgm --ts JOBID` | `jobscope JOBID --ts` |
| `jobscope live -a` | `jobscope -a` |
| `--cgpu` | the default (removed) |
| `--min-runtime 180` (DIAG cutoff) | `--diag-short 180` |

Note that bare `jobscope` now shows **running** jobs rather than the last day of
finished ones, and that `--min-runtime` now means the runtime floor
(`--min-elapsed`) in every mode.

## Reference

Run `jobscope describe` for column definitions and `jobscope describe --dcgm --ext`
for the full metric catalog.

For the pipeline behind those numbers — which source wins, how each metric is
reduced over time and across GPUs, the raw-vs-display job ID rule, MIG limits, and
how to verify a value by hand — see [`docs/metrics.md`](docs/metrics.md).

## Plotting

`jobscope plot` renders `jobscope <view> --csv` output as terminal bar gauges,
histograms, heatmaps, and time-series line charts. The chart kind is
auto-detected from the CSV columns; override with `--kind`.

Time-series charts show `GPU%`, `SM_ACT%`, `OCC%`, `TENSOR%` and `DRAM%` by
default; `--metric` picks specific columns and `--all` charts every numeric one.

```bash
jobscope --gpu  --csv JOBID      | jobscope plot                 # bar gauges (one job)
jobscope --gpu  --csv -D 7       | jobscope plot --kind hist      # distribution (many jobs)
jobscope dcgm   --csv -D 7       | jobscope plot                 # heatmap (jobs/GPUs x metrics)
jobscope dcgm --ts --csv JOBID   | jobscope plot --compact        # time series
```

`jobscope plot` reads `summary`, `dcgm`, and `dcgm --ts` CSV; the `detail` CSV is
for machine consumption, not charts. Do not pass `-n` when piping to `jobscope
plot`: the plot needs the CSV header row.

## Contrib

`contrib/jobstats_extended.py` is a site-specific prototype that folds DCGM
metrics into the jobstats blob itself. It depends on an upstream jobstats install
and is not part of the package; see [`contrib/README.md`](contrib/README.md).

## References

- [FASRC jobstats documentation](https://docs.rc.fas.harvard.edu/kb/jobstats/)
- [Princeton jobstats](https://princetonuniversity.github.io/jobstats/)
- For live monitoring, use [KempnerPulse](https://github.com/KempnerInstitute/kempnerpulse)
