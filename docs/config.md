# Configuring jobscope

`jobscope config --example` prints a template with every knob in it. This file is the
reasoning behind the ones where the *why* is longer than the setting — kept here so
the template stays a template. Sections are in the same order as the file.

Run `jobscope config` at any point to see what your file actually resolves to. It
prints both band tables, the metric lists, and every default, after all overrides.

---

## `[prometheus]` — the endpoint

The URL often embeds a credential (a Grafana Cloud token, say). Prefer
`$JOBSCOPE_PROM_URL` so the secret never lands in a file; a `url` in the config is
used only when that variable is unset. jobscope masks the credential wherever it
prints the endpoint — `jobscope doctor` shows `https://***@host/path`.

If your site already runs jobstats, you usually need nothing here: jobscope finds the
`config.py` next to the `jobstats` binary on your `PATH` and reads `PROM_SERVER` from
it. `site_jobstats_config_path` is only for a `config.py` somewhere else.

## `[site]` — label conventions

Only needed when porting to a cluster whose exporters label series differently. Every
one of these fails **silently** when wrong: read the wrong host label and every node
shows as `?`, the cgroup divisor lookup misses, and CPU%/MEM% come back blank with no
error at all.

`jobscope doctor` checks each against the live server for exactly that reason. Run it
before editing anything here — it names the label your server actually uses.

## Metric families, and which you can do without

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
metrics out of `[metrics]` and they are never queried. `jobscope doctor` reports which
exporters answer on your cluster, and `jobscope doctor --metrics` lists every series
they carry for a real job.

## `[thresholds]` — the band edges

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

## `[classify]` — how metrics become a verdict

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
to stop `--dcgm` widening the ballot from four metrics to fifteen — a job busy on ENC%
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

A bare `[classify.floor]` with nothing under it means **no floors**. That is legal and
it re-opens what the floor closes: a job at GPU% 48 drawing 80 W goes back to reading
`good`.

## `[metrics]` — defining or repointing a metric

Run `jobscope doctor --metrics` first: it lists every series your server carries for a
real job and marks `new` the ones with no jobscope name. `jobscope doctor --toml`
emits that as an editable config block.

A defined metric joins the **extended** catalog, so it appears under `--dcgm` or in any
view that names it. It never joins the default view on its own — defining a metric
cannot silently widen every report, or the queries every sweep pays for.

The same table with a **built-in's** name overrides it, which is how a cluster whose
exporter uses different series names ports without patching jobscope. An override
changes only what it names; header, tier, group and roles are inherited, so repointing
`cgroup.cpu` does not rename the CPU% column or take it out of the classifier.

## `[colors]`

The summary table's `RED`/`YELLOW`/`GREEN` columns keep those names whatever you set,
and so do the `red=`/`yellow=`/`green=` fields of `--csv`. Those are the three band
*counts*, so a script reading them does not break when you recolour the display.

Two tiers sharing a colour is the default, not a requirement — give all five distinct
values for a colourblind-safe palette.
