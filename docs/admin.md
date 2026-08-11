# Administering jobscope

For whoever sets jobscope up on a cluster: what the site must provide, how to point it
at Prometheus, and the reasoning behind the settings where the *why* is longer than the
setting.

Users need none of this. The [README](../README.md) is the day-to-day guide and
[`reference.md`](reference.md) is the full command spec.

- [What the cluster must provide](#what-the-cluster-must-provide) · [Setting up a site](#setting-up-a-site)
- [What `probe` reports](#what-probe-reports) · [`--coverage`](#--coverage-which-hosts-serve-each-column)
- [Pointing at Prometheus](#pointing-jobscope-at-prometheus) · [Configuration](#configuration)
- [Retired subcommands and flags](#retired-subcommands-and-flags)

---

## What the cluster must provide

- Python 3.9+
- Slurm with `sacct`, where the jobstats-style AdminComment summary is populated.
- A Prometheus endpoint serving the DCGM (`DCGM_FI_*`), `nvidia_gpu_*` and `cgroup_*`
  series. Needed for the GPU columns and for any running job. `finished --cpu` never
  contacts it.

`jobscope probe` checks all of this against the live cluster, which is where to start
rather than reading the list.

## Setting up a site

```bash
jobscope probe              # what does this cluster expose, and can jobscope read it
jobscope probe --init       # write a config from what it just found
```

`--init` writes to the path jobscope will read (`-c`, then `$JOBSCOPE_CONFIG`, then
`~/.config/jobscope/config.toml`) — but **only if nothing is there**. An existing file
turns it into stdout plus a note, because a config is hand-tuned within a week of being
written and that tuning has no other copy.

What it detects, all measured rather than assumed:

| setting | how |
|---|---|
| `[prometheus] sampling_period` | the spacing of raw samples — **not** `query_range`, which returns whatever step you pass |
| `[prometheus] max_queries_per_second` | pacing for the running view's per-job fan-out; `0` disables it |
| `[prometheus] query_burst` | how many queries run unpaced first — keeps small commands untouched |
| `[defaults] max_running_jobs` | the job count a running selection refuses past, as a typo backstop |
| `[defaults] verdict_window` | how far back `--eff`/`--verify` look with no window given (`180m`); a bare `--ts` still means the whole run |
| `[site]` labels | probed against real series, trying `instance`/`host`/`node`/`nodename` |
| `[metrics]` lists | narrowed to series this server carries, so a missing exporter does not leave columns blank forever |
| `[eff.floor.power]` | per GPU model: idle p90 vs busy p10, floor between them |

**Thresholds are deliberately absent.** A band edge is a policy choice about what counts
as waste, not a property of the cluster. `jobscope probe --init --full` appends every
remaining knob, commented.

The power floors are the part worth reading. The built-in flat 100 W is wrong for most
hardware, so `--init` measures each model. Roughly half measure cleanly at any given
moment, so it emits floors only for those and **comments the rest with the reason**:

```toml
"NVIDIA H100 80GB HBM3" = 130   # idle p90 111 W, busy p10 141 W
# "NVIDIA A100-SXM4-40GB"  -- idle p90 97 W and busy p10 96 W overlap;
#                             measure again over a longer window
```

**A wrong floor is worse than none**: it silently caps healthy jobs at `inefficient`.

## What `probe` reports

```bash
jobscope probe              # can jobscope reach Slurm and Prometheus, and how far back
jobscope probe --metrics    # every metric this server carries for a real job
jobscope probe --toml       # the same, as an editable [metrics] block
jobscope probe --coverage [PART]
```

`probe` reads nothing but your cluster and, apart from `--init`, writes nothing. Each
check fails independently, so it is useful precisely when jobscope does *not* yet work.

`--metrics` is the one to run before editing `[metrics]` or `[thresholds]`. It lists
each series alongside the short name config takes, marked `ok` (catalogued and present),
`new` (present, no jobscope name yet) or `absent` (catalogued but this server lacks it —
for example `DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE`, which A100s do not export and H100s
do). A metric jobscope names but your cluster lacks would otherwise render blank forever.

`--toml` turns that listing into config — built-ins commented out so their names are
visible and renameable, anything unnamed **live**, so a redirect is the only step:

```bash
jobscope probe --toml >> ~/.config/jobscope/config.toml
```

The table key *is* the config name, so renaming a metric is editing that key. Stdout is
only ever TOML and the diagnosis goes to stderr, which is what makes the redirect safe.

Two things it declines to guess rather than get wrong: a cgroup *count* (an OOM-kill
tally, say) is commented out, since every cgroup metric is divided by an allocation and
a count has none; and an unfamiliar GPU metric gets `scale = 1` with a `# CHECK` note,
because a wrong scale reads as a plausible value.

**The opposite flag.** `--no-dcgm` reads *nothing* from Prometheus: it drops the
exporter columns and reports only what the stored jobstats summary carried, which
arrived free with `sacct`. That is the lever for a selection wide enough to matter —
tens of thousands of queries become none — and the one to reach for before lowering
`max_queries_per_second`, which spreads the same queries out rather than removing them.
The two flags are refused together, since between them they would leave no source at
all.

**Checking the two sources agree.** `jobscope --no-jobstats` reads
CPU%/MEM%/GPU%/GMEM% from Prometheus even for finished jobs. Slower — the summary is one
free sacct field where this is several range queries per job — but it is what a site
without jobstats runs on. Across 51 finished jobs spanning CPU-only, single-GPU,
multi-GPU and multi-node shapes, the two agree here to within one point (memory columns
exactly).

To compare *exporters*, run the same report under `--gpu-source nvml` and
`--gpu-source dcgm`, and use `--verify --full`'s `SWING` column to see whether
either mean is reproducible at all. A plain `--verify` says so in words when the swing
is large enough to matter, without the column.

### `--coverage`: which hosts serve each column

The `coverage` line in the main report gives one number per exporter, which hides gaps
*within* a family. Measured on one cluster, `DCGM_FI_PROF_SM_ACTIVE` is on 437 hosts and
`DCGM_FI_DEV_GPU_UTIL` on 416 — the 22-host difference is MIG nodes, which have no
whole-device duty cycle. One number per exporter said nothing about that hole in GPU%.

`--coverage` counts **per series, grouped by source**, and lists every *candidate*, not
just the winner — because comparing sources is the decision it exists to inform. `*`
marks the series serving its column now; unmarked rows are what another `--gpu-source`
would read.

```console
$ jobscope probe --coverage kempner
coverage    1 partition(s): kempner
            22 of 28 node(s) up (6 down/drained, not counted)

nvml        -- duty cycle, memory, and the job-to-GPU join every source depends on
              nvidia_gpu_duty_cycle              GPU%           22/22
            * nvidia_gpu_memory_used_bytes       GMEM_GB        22/22

dcgm        -- the profiling catalog
            * DCGM_FI_DEV_GPU_UTIL               GPU%           21/22
            * DCGM_FI_PROF_SM_ACTIVE             SM_ACT%        21/22

missing     1 node(s) running jobs but not publishing every serving series:
            holygpu8a19102   (mixed)     no dcgm: GPU%, SM_ACT%, TENSOR%, DRAM%, POWER_W
```

Those two GPU% rows answer a question the per-exporter count cannot express: nvml covers
the node dcgm does not, so `--gpu-source nvml` would report on it.

Partition names accept a comma list and shell-style wildcards (`kempner_h*`), which
jobscope expands since `sinfo -p` does not.

**The `missing` section is the point** — over five kempner partitions it reduces 254
nodes to the one host worth acting on. Four classes are kept out of it, each judged per
*gap* rather than per node, since one node can have two absences with two explanations:

- **`down`/`drained`/`inval` nodes** — not serving, so nothing to measure.
- **GPU columns on nodes with no gpu gres** — a CPU-only partition would otherwise
  report every GPU column missing everywhere.
- **`CPU%`/`MEM%` where no job is running.** cgroup series exist per running *job*, so
  `idle`, `reserved` and `planned` nodes correctly have none. A node in `mixed`,
  `allocated` or `completing` **is** faulted.
- **`GPU%` on a MIG node.** Partitioning a card leaves no whole *device* to report a
  duty cycle for, so neither exporter publishes one. Scoped to `GPU%` alone: a MIG node
  missing `SM_ACT%` **is** a fault, because those series are per instance.

Each reason says which columns it costs and names **every** node rather than eliding a
tail — the list is what you paste into `scontrol` or a ticket.

## Pointing jobscope at Prometheus

```bash
# Preferred: environment variable (keeps a credential out of any file)
export JOBSCOPE_PROM_URL="https://USER:TOKEN@prometheus.example.net/api/prom"
```

**Kempner AI Cluster users:** the jobstats `config.py` sits beside the `jobstats` binary
on your `PATH` and jobscope **auto-discovers it**, so you need no config file and never
handle the URL or token. This works at any jobstats site; to point at a different
install, set `site_jobstats_config_path`.

On any other cluster, put settings in the config file
(`~/.config/jobscope/config.toml`, or wherever `$JOBSCOPE_CONFIG` points):

```bash
jobscope config --example > ~/.config/jobscope/config.toml   # then edit it
jobscope config                                              # show the path in use
```

The config file is also where reporting *policy* lives, so adapting jobscope to a site's
conventions is a TOML edit rather than a patch:

| section | what it sets |
|---|---|
| `[thresholds]` | the band edges, per metric and per view |
| `[gpu]` / `[host]` | which source serves each column |
| `[metrics]` | which GPU/DCGM metrics each view collects and shows |
| `[colors]` | the colour of each tier, in tables and charts alike |
| `[defaults]` | default window and state filter, timeouts, worker counts, row sizes |
| `[report]` | which sections print below the job table, and in what order |
| `[plot]` | chart defaults: which series, which colours, the row and panel caps |

`jobscope config` prints all of it as it actually resolves. The template resolves
exactly like no config file at all, so copying it is never a silent regrade.

The Prometheus URL commonly embeds a credential, so jobscope treats it as a secret: no
command prints it, and a `config.toml` in a repo checkout is git-ignored. The single
exception is `jobscope probe`, which masks it (`https://***@host/path`) so its output
stays safe to paste into a ticket.

**Widening to a partition.** `-a` widens a selection to every user. It is the one flag
that changes cost rather than presentation — a partition-wide sweep is hundreds of jobs,
each GPU job costing a set of range queries. `--workers` governs how many run at once.
`--cpu` needs no endpoint at all for finished jobs.

---

## Configuration

### `[site]` — label conventions

Only needed when porting to a cluster whose exporters label series differently. Every
one of these fails **silently** when wrong: read the wrong host label and every node
shows as `?`, the cgroup divisor lookup misses, and CPU%/MEM% come back blank with no
error at all. `jobscope probe` checks each against the live server for exactly that
reason — run it before editing anything here.

### Metric families, and which you can do without

| family | gives you | can you drop it? |
|---|---|---|
| `cgroup` | CPU%, MEM%, and the extended host columns | Yes — a GPU-only site |
| `nvml` | GPU%, GMEM% (3 specs) | **No** — see below |
| `dcgm` | SM_ACT%, TENSOR%, DRAM%, POWER_W … (27 specs) | Yes — no dcgm-exporter |

**`nvml` is load-bearing beyond its three metrics.** GPU series carry no job label, so
the only join between Slurm and *either* GPU exporter is `nvidia_gpu_jobId` — an
NVML-exporter series whose *value* is the job holding each card. Both `dcgm` and `nvml`
discovery go through it. A site with dcgm-exporter but no nvidia-exporter cannot
attribute GPU metrics to jobs at all; it would need a different join series set as
`[site] gpu_job_join`.

There is no "families" switch, because there need not be one: leave a family's metrics
out of `[metrics]` and they are never queried.

### `[thresholds]` — the band edges

The five tiers and their defaults are in
[`reference.md`](reference.md#thresholds). What matters when setting them:

**Red and yellow flag *pathological* jobs, not merely inefficient ones.** At an
`inefficient` edge of 10, a job at 21% paints green while leaving four fifths of its
allocation unused. That is the intended calibration — a report where most jobs are red
gets ignored — but it means the colours are not an efficiency score. For efficiency read
the **USED** column; for jobs worth someone's time read the Problem-jobs rows, which use
the stricter `wasteful` cutoff.

**Two tables, nothing inherited.** `[thresholds.summary]` grades one average over a
job's whole runtime; `[thresholds.timeslice]` grades samples pooled inside a `--ts`
window. Those are different statistics and can want different bars. A config that tunes
one and forgets the other would otherwise grade the same job differently depending on
whether `--ts` was passed, with nothing on screen to say why; jobscope prints a note when
only one is set. `[thresholds] edges = [2, 10, 20, 40]` sets both at once — use it unless
you have a reason to separate them.

**Per metric, because the metrics do not mean the same thing.** A GPU job legitimately
holds cores it never uses, so CPU% at 4% is ordinary where GPU% at 4% is idle. jobscope
ships exactly **one** such calibration: CPU% `wasteful = 5`.

**That calibration switches off for any edge you retune.** Writing
`edges = [3, 10, 20, 40]` means "3 for everything I did not name", and jobscope
overriding that with its own opinion is the kind of thing nobody can debug. Name CPU%
explicitly to keep a different value:

```toml
[thresholds]
edges = [3, 10, 20, 40]
[thresholds.summary.wasteful]
cpu = 5          # keep CPU% where it was
```

Edges must not decrease within a metric, counting the defaults it falls back to. Setting
only `inefficient.cpu = 1` against a `wasteful` of 2 would leave CPU% a band nothing can
land in; jobscope rejects that. Equal neighbours are allowed — they collapse a band on
purpose.

### `[eff]` — how metrics become a verdict

Two roles, and the asymmetry is the whole model:

- **`vote`** — best-of-N. One busy measure is enough to call a job not-idle, so a vote
  can only ever **raise** a verdict.
- **`floor`** — can only **lower** one.

A metric that must pull a verdict down cannot be a vote, because under best-of-N a low
reading is simply outvoted. Measured on 94 real GPU jobs, 8 looked healthy on their
percentages while the board sat below its idle floor.

Omit `vote` and jobscope derives it: every graded percentage that is not a capacity
reading. Set it to stop `--all-metrics` widening the ballot from four metrics to fifteen
— a job busy on ENC% alone would otherwise read `good`.

**Ceilings** cap how high a metric may vote without stopping it voting. CPU%'s is built
in at `inefficient`: a busy host is not evidence the cards were needed, so CPU% lifts a
GPU-idle job off `wasteful` but never calls it healthy. It still bands and colours on the
ordinary ladder — a job at CPU% 50 is genuinely half used and paints green. The ceiling
lifts when nothing else could carry a verdict, so a `--cpu` view and a genuinely CPU-only
job are judged on their own terms.

### Choosing a power floor

Idle draw is a property of the hardware, so one number cannot serve a mixed fleet — an
idle RTX PRO 6000 draws more than a working V100.

**Measure both sides.** A floor belongs between what a model draws idle and what it
draws working: take every GPU of that model at one instant, split on whether its duty
cycle is zero, and put the floor between the idle 90th percentile and the busy 10th.
Measuring only the idle side gives a number that flags working cards — on the fleet
below, a flat 150 W would have called busy H200s (busy p10 122 W) idle.

Keys go under `[eff.floor.power]`, alongside a `default`:

```toml
[eff.floor.power]
default = 100
"NVIDIA H100 80GB HBM3" = 130
```

(`[thresholds.power_w_by_model]` still works as the older spelling.)

Measured on one cluster, for illustration. **These are not defaults**, and hardware
elsewhere will differ:

| model | idle p90 | busy p10 | floor |
|---|---|---|---|
| NVIDIA RTX PRO 6000 Blackwell | 288 W | 373 W | 330 |
| NVIDIA H100 80GB HBM3 | 121 W | 141 W | 130 |
| NVIDIA H200 | 117 W | 122 W | 120 |
| NVIDIA A100-SXM4-80GB | 98 W | 99 W | 100 |
| NVIDIA A100-SXM4-40GB | 55 W | 121 W | 90 |
| NVIDIA A40 | 35 W | — | 40 |
| Tesla V100-PCIE-32GB | 39 W | — | 45 |

Keys are the model string the exporter itself reports — the `name` label on
`nvidia_gpu_*`, `modelName` on `DCGM_FI_*`. jobscope matches it exactly, so copy it
verbatim; `jobscope <job> --per-gpu` shows which cards a job ran on.

A bare `[eff.floor]` with nothing under it means **no floors**. That is legal, and it
re-opens what the floor closes: a job at GPU% 48 drawing 80 W goes back to reading
`good`.

### `[metrics]` — defining or repointing a metric

Run `jobscope probe --metrics` first, then `--toml` for an editable block.

A defined metric joins the **extended** catalog, so it appears under `--all-metrics` or
in any view that names it. It never joins the default view on its own — defining a
metric cannot silently widen every report, or the queries every sweep pays for.

The same table with a **built-in's** name overrides it, which is how a cluster whose
exporter uses different series names ports without patching jobscope. An override
changes only what it names; header, tier, group and roles are inherited, so repointing
`cgroup.cpu` does not rename the CPU% column or take it out of the efficiency ballot.

### `[colors]`

The summary table's `RED`/`YELLOW`/`GREEN` columns keep those names whatever you set, and
so do the `red=`/`yellow=`/`green=` fields of `--csv`: those are band *counts*, so a
script reading them does not break when you recolour the display.

Two tiers sharing a colour is the default, not a requirement — give all five distinct
values for a colourblind-safe palette.

### Unrecognized config sections

jobscope prints a note when a top-level section is one it does not read. A typo'd
`[promtheus]` used to cost a site every setting under it in silence. A note rather than
an error, because an unread section is inert and the rest of the file still resolves.

---

## Retired subcommands and flags

The positional subcommands are **gone**. Naming one is an error that says what to type
instead — they are kept as *defined* words rather than deleted, so the word does not
fall through as a would-be JOBID and get `sacct: fatal: Bad job/step specified`:

| was | now |
|---|---|
| `jobscope summary -D 3` | the default (`jobscope finished -D 3`) |
| `jobscope detail JOBID` | `jobscope JOBID --per-gpu` |
| `jobscope dcgm JOBID` | `jobscope JOBID --all-metrics` |
| `jobscope live -a` | `jobscope -a` (`running` is the default) |
| `jobscope doctor` | `jobscope probe` |

Retired flags, kept defined for the same reason so the message names the replacement:

| was | now |
|---|---|
| `--dcgm` | `--all-metrics` (or `--gpu-source dcgm` to pick the source) |
| `--ext` | `--all-metrics` |
| `--no-blob` | `--no-jobstats` |
| `--stats-per-node` / `--stats-per-job` | `--stats node` / `--stats job` |
| `--all-categories` | `--eff all` |
| `--avg` | `--runtime-avg` |

`--validate` was removed outright: its three-way comparison rested on Slurm's
`gres/gpuutil`, which is a single point reading and not a mean — `TRESUsageInTot`,
`InAve`, `InMax` and `InMin` all return the same number — so its GPU row compared a
snapshot against a whole-run average and disagreed in proportion to how fast the metric
was moving. What it was reaching for is two exporters measuring the same quantity:
`--gpu-source nvml` against `--gpu-source dcgm`, plus `--verify --full`'s
`SWING` column.

Older spellings not in that table (`--validate`, `--hwdetail`, `--timeseries`,
`--extended`, `--min-runtime`) are gone and give argparse's plain "unrecognized
arguments".
`--plot_ts`, `--plot_ts_overlay` and `--node` are still accepted, as underscore aliases
of `--plot-ts`, `--plot-ts-overlay` and `--nodename`.

Note that bare `jobscope` shows **running** jobs rather than the last day of finished
ones. `doctor` was renamed rather than retired — it probes the telemetry sources, and
the new name says so.
