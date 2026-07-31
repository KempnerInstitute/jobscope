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
GPU-hours:   7.1 alloc  5.7 used  1.4 idle (20%)
Core-hours:  56.2 alloc  6.9 used  49.3 idle (88%)
Bands GPU%:  red<25 1 job (6%)/0.1h (1%)  yellow<50 1 job (6%)/0.1h (2%)  green 15 jobs (88%)/6.9h (97%)
Bands CPU%:  red<10 0 jobs (0%)/0.0h (0%)  yellow<20 17 jobs (100%)/56.2h (100%)  green 0 jobs (0%)/0.0h (0%)
Worst:       35260825 0.1h@20% bdesinghu
Jobs:        cpu-jobs=17  gpu-jobs=17  gpus=17
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
Used/GPU-hr:                     3   5      34  14   29.8  ...
GPU-hours:   701.5 alloc  299.0 used  402.5 idle (57%)
Core-hours:  9875.7 alloc  492.3 used  9383.4 idle (95%)
Bands GPU%:  red<25 13 jobs (4%)/388.5h (55%)  yellow<50 4 jobs (1%)/0.4h (0%)  green 302 jobs (95%)/318.2h (45%)
Bands CPU%:  red<10 159 jobs (50%)/6249.1h (63%)  yellow<20 158 jobs (50%)/3366.7h (34%)  green 2 jobs (1%)/344.1h (3%)
Worst GPU:   35475803 142.9h@0% amazloumi  35476814 142.0h@0% amazloumi  36337781 43.8h@0% amazloumi
Worst CPU:   35475803 2286.6h@0% amazloumi  35476814 2271.9h@0% amazloumi  36337781 698.4h@0% amazloumi
Worst both:  35475803 35%gpu+24%cpu amazloumi  35476814 35%gpu+24%cpu amazloumi  36337781 11%gpu+7%cpu amazloumi
Jobs:        cpu-jobs=319  gpu-jobs=319  gpus=381  no-runtime=4
```

**There is no per-job mean**, on purpose. Utilization is bimodal -- jobs cluster
near 0% or near 100% -- so an average of them describes a job that does not
exist. On the day above it read 82%, while the partition was 66% idle.

`Used/GPU-hr:` is a ratio rather than a centre: used resource-time over allocated
resource-time. Each column is pooled over the resource *it* measures -- GPU-hours
for `GPU%`, core-hours for `CPU%`, GB-hours for `MEM%` -- so the row is the real
utilization of the pool and stays true whatever the distribution looks like.

Both resources are reported. A GPU job that holds 32 cores and uses two is
blocking other work from that node, and the GPU lines cannot show it -- above,
the GPUs were 57% idle while the *cores* were 95% idle. `--gpu` and `--cpu` narrow
the block to one.

The `Bands` rows are the part worth reading. Each gives a threshold band's share of the
**jobs** and of the **resource-time**, and the gap between those two numbers is
the finding: above, 4% of the jobs held 55% of the GPU-hours below 25%. Either
number alone conceals it.

The `Worst` rows then name the offenders, ranked by resource-time *wasted* rather
than held, so a long job at a mediocre rate outranks a short one at zero. There
is one per resource, plus a combined row. GPU-hours and core-hours cannot simply
be added -- any exchange rate between them would be invented, and a wrong one
decides the answer by itself -- so `Worst both:` expresses each job's waste as a
share of the selection's total waste in that resource and sums the two shares.
Both components are printed (`35%gpu+24%cpu`), so you can see which resource put
a job on the list. Only jobs red in at least one resource are candidates: a
95%-efficient job can idle 50 GPU-hours just by being enormous, and there is
nothing to act on there.

The cutoffs come from `[thresholds]` in your config -- the same ones that tint
the cells and colour `jobscope plot`, so the footer is a tally of what you can
already see.

The **running** view reports the same block over resource *counts* rather than
resource-hours (`Used/GPU:`, `GPUs: 20 alloc  17.3 used  2.7 idle (13%)`). `used`
is fractional there because it is GPU-equivalents busy, not whole GPUs. Its numbers are one scrape at a
single moment, so weighting them by elapsed time would claim that instant
represents the whole run; `--avg` folds each job over its runtime and does get
the hour-based form. A `--cpu` run switches the block to `CPU%` over core-hours.

The two job counts differ whenever the selection mixes CPU-only and GPU work: a
CPU-only job has no `GPU%` to pool, so it is absent from the GPU figures rather
than counted as zero. A GPU job that sat idle *is* counted, as 0%. `no-runtime=N`
appears when a job had no elapsed time to weight by and was left out.
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
gpu = 25      GPU% red below this, yellow below 50
gmem = 20     GMEM%
cpu = 10      CPU%  (low on purpose -- see below)
mem = 25      MEM%
default = 15  every other %-metric (SM_ACT%, OCC%, TENSOR%, DRAM%, ...)
```

`cpu` sits below the others because `CPU%` is the share of *allocated* cores a job
kept busy, and a GPU job legitimately keeps very few -- it asks for a batch of
cores and puts the work on the GPU. Measured over a day on one GPU partition,
`CPU%` had a median of 10 and a maximum of 18 across 396 jobs, so the old cutoff
of 25 marked every job red and distinguished nothing. Raise it on a CPU
partition, where a job that asks for cores is expected to use them.

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
