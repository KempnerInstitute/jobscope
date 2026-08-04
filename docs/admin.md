# Administering jobscope

For whoever sets jobscope up on a cluster: what the site has to provide, how to point
it at Prometheus, how to generate a config, and the reasoning behind the settings where
the *why* is longer than the setting.

Users need none of this. The [README](../README.md) is the day-to-day guide, and
[`reference.md`](reference.md) is the full command and column spec.

- [What the cluster must provide](#what-the-cluster-must-provide)
- [Setting up a site](#setting-up-a-site-two-commands)
- [What `probe` reports](#what-probe-reports)
- [Pointing jobscope at Prometheus](#pointing-jobscope-at-prometheus)
- [Looking at a whole partition](#looking-at-a-whole-partition)
- [Configuration](#configuration) -- the per-section reasoning
- [Retired subcommands and flags](#retired-subcommands-and-flags)

---

## What the cluster must provide

- Python 3.9+
- Slurm with `sacct`, where the jobstats-style AdminComment summary is populated
  (needed by every view).
- A Prometheus endpoint serving the DCGM (`DCGM_FI_*`), `nvidia_gpu_*` and
  `cgroup_*` series that jobstats scrapes. Needed for the GPU columns, and for any
  running job (whose summary does not exist yet). `finished --cpu` never contacts it.

`jobscope probe` checks every one of these against the live cluster, which is where to
start rather than reading the list.

### Setting up a site: two commands

```bash
jobscope probe              # what does this cluster expose, and can jobscope read it
jobscope probe --init       # write a config from what it just found
```

`--init` turns the findings into a config instead of prose. It writes to the path
jobscope will read (`-c`, then `$JOBSCOPE_CONFIG`, then
`~/.config/jobscope/config.toml`) — but **only if nothing is there**. An existing
file turns it into stdout plus a note, because a config is hand-tuned within a week
of being written and that tuning has no other copy.

What it detects, all of it measured rather than assumed:

| setting | how |
|---|---|
| `[prometheus] sampling_period` | the spacing of raw samples — **not** `query_range`, which returns whatever step you pass |
| `[site]` labels | probed against real series, trying `instance`/`host`/`node`/`nodename` when the configured one answers nothing |
| `[metrics]` lists | narrowed to series this server actually carries, so a missing exporter does not leave columns blank forever |
| `[eff.floor.power]` | per GPU model: idle p90 vs busy p10, floor between them |

**Thresholds are deliberately absent.** A band edge is a policy choice about what
counts as waste, not a property of the cluster, so the built-ins apply until you set
them. `jobscope probe --init --full` appends every remaining knob, commented.

The power floors are the part worth reading. The built-in flat 100 W is wrong for most
hardware — an idle RTX PRO 6000 draws more than a working V100 — and `--init` measures
each model instead. Roughly half of them measure cleanly at any given moment, so it
emits floors only for those and **comments the rest with the reason**:

```toml
"NVIDIA H100 80GB HBM3" = 130   # idle p90 111 W, busy p10 141 W
# "NVIDIA A100-SXM4-40GB"  -- idle p90 97 W and busy p10 96 W overlap;
#                             measure again over a longer window
```

A wrong floor is worse than none: it silently caps healthy jobs at `inefficient`.

### What `probe` reports

```bash
jobscope probe              # can jobscope reach Slurm and Prometheus, and how far back
jobscope probe --metrics    # every metric this server carries for a real job
```

`probe` reads nothing but your cluster, and apart from `--init` writes nothing. Each check
fails independently -- so it is useful precisely when jobscope does *not* yet
work. It reports which Slurm accounting sources are populated, whether jobstats
summaries are being written, how far back Prometheus actually holds data, and whether
the join labels are the ones the collectors assume.

`--metrics` is the one to run before editing `[metrics]` or `[thresholds]`: it
lists each series alongside the short name config takes, and marks it `ok`
(catalogued and present), `new` (present, jobscope has no name for it yet) or
`absent` (catalogued but this server does not carry it -- for example
`DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE`, which A100s do not export and H100s do).
A metric jobscope names but your cluster lacks would otherwise just render blank
forever.

```bash
jobscope probe --toml               # the same, as an editable [metrics] block
jobscope probe --validate           # do the exporters agree with the scheduler?
```

`--toml` turns that listing into config. It prints one
`[metrics.<family>.<name>]` table per series -- built-ins commented out so their
names are visible and renameable, and anything jobscope has no name for **live**, so
a redirect is the only step:

```bash
jobscope probe --toml >> ~/.config/jobscope/config.toml
```

The table key *is* the config name, so renaming a metric is editing that key. Stdout
is only ever TOML; the diagnosis goes to stderr, which is what makes the redirect
safe. It writes nothing itself -- appending to a file you have hand-edited is your
call, not the tool's.

Two things it declines to guess rather than getting wrong. A cgroup *count* -- an
OOM-kill tally, say -- is commented out with the reason: every cgroup metric is
divided by an allocation, and a count has none, so a percentage of total bytes would
be a number with no meaning. And an unfamiliar GPU metric gets `scale = 1` with a
`# CHECK` note, because a wrong scale reads as a plausible value.

`jobscope --no-jobstats` is the other half of that: it reads CPU%/MEM%/GPU%/GMEM% from
Prometheus even for finished jobs, instead of the jobstats summary Slurm stored. Slower -- the
the summary is one free sacct field where this is several range queries per job -- but it
is what a site without jobstats runs on, and it is how you check the two agree.
Across 51 finished jobs spanning CPU-only, single-GPU, multi-GPU and multi-node
shapes, they agree here to within one point (memory columns exactly).

`--validate` puts one job's utilization side by side as each source measures it:
Prometheus, the jobstats summary, and Slurm's own `jobacct_gather` /
`AccountingStorageTRES` accounting, which needs neither of the other two.

Expect Slurm to read a little **higher**, and do not treat that as an error. Across
sixty finished GPU jobs here the median gap is +6 points on GPU% and +4.7 on CPU%,
in the same direction every time, because Slurm accounts only while a step is
running where NVML and cgroup average across the whole allocation -- setup,
teardown and idle gaps included. For an efficiency tool the wider denominator is
the point: allocated-but-idle time is exactly the waste jobscope is looking for.
A *large* gap on a single job usually means a long warm-up; a large gap across
many jobs is worth chasing.

### Pointing jobscope at Prometheus

The GPU columns, and every column for a running job, need a Prometheus endpoint
serving the DCGM, `nvidia_gpu_*` and `cgroup_*` series. (`finished --cpu`, and
everything under `describe`, `config` and the non-metric half of `probe`, need
nothing.) Provide it one of these ways.

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

The config file is also where the reporting *policy* lives, so adjusting jobscope
to a site's conventions is a TOML edit rather than a patch:

| section | what it sets |
|---|---|
| `[thresholds]` | the band edges, per metric and per view — see [Thresholds](#thresholds) |
| `[metrics]` | which GPU/DCGM metrics each view collects and shows |
| `[colors]` | the colour of each classified tier, in tables and in charts alike |
| `[defaults]` | the default window and state filter, timeouts, worker counts, and the Problem-jobs row size |
| `[report]` | which sections print below the job table, and in what order |
| `[plot]` | chart defaults: which series, which colours, the row and panel caps |

`jobscope config` prints all of it as it actually resolves; `jobscope config
--example` is the commented template, ordered required-first with everything below
its divider being tuning. [`docs/admin.md`](docs/admin.md) carries the reasoning
that is longer than the setting — how to choose a power floor, which metric families
you can do without, why the two band tables inherit nothing from each other.

The template resolves exactly like no config file at all, so copying it is never a
silent regrade.

The Prometheus URL commonly embeds a credential, so jobscope treats it as a
secret: no command prints it, and a `config.toml` in a repo checkout is
git-ignored. The single exception is `jobscope probe`, which has to name the
endpoint it is talking to and shows it with the credential replaced --
`https://***@prometheus.example.net/api/prom` -- so its output stays safe to
paste into a ticket. On sites already running jobstats,
`site_jobstats_config_path` reuses that install's `PROM_SERVER`, so the secret is
never copied.

## Looking at a whole partition

Everything in the README is scoped to your own jobs. `-a` widens a selection to every
user, which is the view for whoever is watching a partition rather than a job:

```bash
jobscope -p kempner_h100 -a              # every user's running jobs there
jobscope finished -p kempner_h100 -a -D 1  # and what finished in the last day
```

It is the one flag that changes cost rather than presentation. A partition-wide sweep
is hundreds of jobs instead of a handful, and each GPU job costs a set of Prometheus
range queries -- `--workers` governs how many run at once, and `[defaults] workers`
sets it per site. `--cpu` needs no endpoint at all for finished jobs, so it stays fast
however wide the selection.

## Configuration

### `[prometheus]` — the endpoint

The URL often embeds a credential (a Grafana Cloud token, say). Prefer
`$JOBSCOPE_PROM_URL` so the secret never lands in a file; a `url` in the config is
used only when that variable is unset. jobscope masks the credential wherever it
prints the endpoint — `jobscope probe` shows `https://***@host/path`.

If your site already runs jobstats, you usually need nothing here: jobscope finds the
`config.py` next to the `jobstats` binary on your `PATH` and reads `PROM_SERVER` from
it. `site_jobstats_config_path` is only for a `config.py` somewhere else.

### `[site]` — label conventions

Only needed when porting to a cluster whose exporters label series differently. Every
one of these fails **silently** when wrong: read the wrong host label and every node
shows as `?`, the cgroup divisor lookup misses, and CPU%/MEM% come back blank with no
error at all.

`jobscope probe` checks each against the live server for exactly that reason. Run it
before editing anything here — it names the label your server actually uses.

### Metric families, and which you can do without

Three families, and they are not symmetrically optional:

| family | gives you | can you drop it? |
|---|---|---|
| `cgroup` | CPU%, MEM%, and the extended host columns | Yes — a GPU-only site |
| `nvml` | GPU%, GMEM% (3 specs) | **No** — see below |
| `dcgm` | SM_ACT%, TENSOR%, DRAM%, POWER_W … (27 specs) | Yes — no dcgm-exporter |

**`nvml` is load-bearing beyond its three metrics.** GPU series carry no job label, so
the only join between Slurm and *either* GPU exporter is `nvidia_gpu_jobId` — a series
whose *value* is the job ID holding each card. That is an NVML-exporter series, and
both `dcgm` and `nvml` discovery go through it. A site with dcgm-exporter but no
nvidia-exporter cannot currently attribute GPU metrics to jobs at all; it would need a
different join series set as `[site] gpu_job_join`.

There is no "families" switch, because there does not need to be one: leave a family's
metrics out of `[metrics]` and they are never queried. `jobscope probe` reports which
exporters answer on your cluster, and `jobscope probe --metrics` lists every series
they carry for a real job.

### `[thresholds]` — the band edges

Every %-metric is graded into five tiers by four numbers:

```
        below wasteful   wasteful      (red)
     wasteful..inefficient   inefficient   (red)
  inefficient..improvement   needs improvement  (yellow)
   improvement..average      average       (green)
             above average   good          (green)
```

### What red actually catches

Red and yellow flag **pathological** jobs, not merely inefficient ones. At an
`inefficient` edge of 10, a job at 21% paints green while leaving four fifths of its
allocation unused. That is the intended calibration — a report where most jobs are red
gets ignored — but it means the colours are not an efficiency score.

For efficiency, read the **USED** column of the summary — resource-time that did
work, and its share of what was allocated. For the jobs actually worth someone's
time, read the Problem-jobs "Wasteful" rows, which use the stricter `wasteful`
cutoff.

### Why two tables, and why nothing is inherited

`[thresholds.summary]` grades one average over a job's whole elapsed runtime.
`[thresholds.timeslice]` grades samples pooled inside a `--ts` window. Those are
different statistics and can want different bars: a two-hour slice that catches a
checkpoint pause is not a two-hour idle job.

Nothing is copied between them. A config that tunes one and forgets the other would
otherwise grade the same job differently depending on whether `--ts` was passed, with
nothing on screen to say why. jobscope prints a note when only one is set.

`[thresholds] edges = [2, 10, 20, 40]` sets both at once — use it unless you have a
reason to separate them.

### Why per metric

The metrics do not mean the same thing. A GPU job legitimately holds cores it never
uses, so CPU% at 4% is ordinary where GPU% at 4% is idle; SM residency sits
structurally below GPU% on the same work.

jobscope ships exactly **one** such calibration: CPU% `wasteful = 5`. SM_ACT% is
graded on the shared ladder like everything else. The template suggests `sm_act = 3`
in a commented block because it reads better on real work, but it is a suggestion —
uncomment it and you have changed the grading, which is why it does not ship live.

**The calibration switches off for any edge you retune.** Writing
`edges = [3, 10, 20, 40]` means "3 for everything I did not name", and jobscope
overriding that with its own opinion is the kind of thing nobody can debug. So it
does not: CPU% moves to 3 with the rest. Name it explicitly to keep a different value:

```toml
[thresholds]
edges = [3, 10, 20, 40]
[thresholds.summary.wasteful]
cpu = 5          # keep CPU% where it was
```

Edges must not decrease within a metric, counting the defaults it falls back to.
Setting only `inefficient.cpu = 1` against a `wasteful` of 2 would leave CPU% a band
nothing can land in; jobscope rejects that rather than grade by it. Equal neighbours
are allowed — they collapse a band on purpose.

### `[eff]` — how metrics become an efficiency verdict

Two roles, and the asymmetry is the whole model:

- **`vote`** — best-of-N. One busy measure is enough to call a job not-idle, so a vote
  can only ever **raise** a verdict.
- **`floor`** — can only **lower** one.

A metric that must pull a verdict down cannot be a vote, because under best-of-N a low
reading is simply outvoted. Measured on 94 real GPU jobs here, 8 looked healthy on
their percentages while the board sat below its idle floor — a job spinning on a
trivial kernel reads busy on GPU% and draws idle watts, and watts are the one signal a
duty cycle cannot fake.

Omit `vote` and jobscope derives it: every graded percentage that is not a capacity
reading. That follows the catalog as it grows, which is usually what you want. Set it
to stop `--all-metrics` widening the ballot from four metrics to fifteen — a job busy on ENC%
alone would otherwise read `good`.

### Ceilings

A **ceiling** caps how high a metric may vote without stopping it voting. CPU%'s is
built in at `inefficient`: a busy host is not evidence the cards were needed, so CPU%
lifts a GPU-idle job off `wasteful` but never calls it healthy.

It still **bands and colours** on the ordinary ladder — a job at CPU% 50 is genuinely
half used and paints green. Two different questions about one number.

The ceiling lifts when nothing else could carry a verdict, so a `--cpu` view and a
genuinely CPU-only job are judged on their own terms rather than capped at
`inefficient` for having no GPU to justify.

### Choosing a power floor

Idle draw is a property of the hardware, so one number cannot serve a mixed fleet. An
idle RTX PRO 6000 draws more than a working V100 — a single floor is wrong at one end
or the other whatever value it takes.

**Measure both sides.** A floor belongs between what a model draws idle and what it
draws working: take every GPU of that model at one instant, split on whether its duty
cycle is zero, and put the floor between the idle 90th percentile and the busy 10th.
Measuring only the idle side gives a number that flags working cards — on the fleet
below, a flat 150 W would have called busy H200s (busy p10 122 W) idle.

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

A bare `[eff.floor]` with nothing under it means **no floors**. That is legal and
it re-opens what the floor closes: a job at GPU% 48 drawing 80 W goes back to reading
`good`.

### `[metrics]` — defining or repointing a metric

Run `jobscope probe --metrics` first: it lists every series your server carries for a
real job and marks `new` the ones with no jobscope name. `jobscope probe --toml`
emits that as an editable config block.

A defined metric joins the **extended** catalog, so it appears under `--all-metrics` or in any
view that names it. It never joins the default view on its own — defining a metric
cannot silently widen every report, or the queries every sweep pays for.

The same table with a **built-in's** name overrides it, which is how a cluster whose
exporter uses different series names ports without patching jobscope. An override
changes only what it names; header, tier, group and roles are inherited, so repointing
`cgroup.cpu` does not rename the CPU% column or take it out of the efficiency ballot.

### `[colors]`

The summary table's `RED`/`YELLOW`/`GREEN` columns keep those names whatever you set,
and so do the `red=`/`yellow=`/`green=` fields of `--csv`. Those are the three band
*counts*, so a script reading them does not break when you recolour the display.

Two tiers sharing a colour is the default, not a requirement — give all five distinct
values for a colourblind-safe palette.

## Retired subcommands and flags

The positional subcommands are **gone**. They were accepted with a deprecation note
for a while; naming one now is an error that tells you what to type instead:

| was | now |
|---|---|
| `jobscope summary -D 3` | `jobscope finished -D 3` |
| `jobscope detail JOBID` | `jobscope JOBID --per-gpu` |
| `jobscope dcgm --ext JOBID` | `jobscope JOBID --all-metrics` |
| `jobscope dcgm --ts JOBID` | `jobscope JOBID --ts` |
| `jobscope live -a` | `jobscope -a` |
| `jobscope doctor` | `jobscope probe` |
| `--cgpu` | the default |

Retired flag spellings, each a duplicate of the one beside it:

| was | now |
|---|---|
| `--hwdetail` | `--per-gpu` |
| `--min-runtime` | `--min-elapsed` |
| `--timeseries` | `--ts` |
| `--stats_per_node`, `--stats_per_job`, `--all_categories` | the `-` spellings |
| `--extended` | `--all-metrics` |
| `--plot_avgeff` | nothing — the bars are the default; `--no-plot` omits them |
| `--classify` | `--eff` — named for the question, not the mechanism |

`--plot_ts` and `--node` keep both spellings: they are what the docs and most
command lines actually use.

### Renamed config sections

**`[classify]` is `[eff]`.** This one needs acting on rather than just noting: it
would otherwise leave a site's own floors and ceilings unapplied and its jobs graded
by the built-in rule instead, and because `probe --init` used to write
`[classify.floor.power]`, generated configs are affected too. Rename the section; the
keys under it are unchanged.

jobscope prints a note when it sees a renamed section — and, since the same check
covers anything else it does not read, when it sees an unrecognized one. A typo'd
`[promtheus]` used to cost a site every setting under it in silence; it now says so.
Both stay notes rather than errors, because an unread section is inert and the rest of
the file still resolves.

Note that bare `jobscope` shows **running** jobs rather than the last day of
finished ones.

`doctor` was renamed rather than retired: it probes the telemetry sources, and the new
name says so. Same flags, plus `--init`.
