# How jobscope gathers its metrics

Where each number comes from, how it is reduced over time and across GPUs, and which
source wins when two could answer. Read this when a number looks wrong, when two views
disagree, or before adding a metric.

What the report *prints* and how to read it is [`reference.md`](reference.md); site
setup is [`admin.md`](admin.md).

1. [Two sources](#1-two-sources-one-preference-order) · 2. [The exporters](#2-the-two-exporters) · 3. [Finding a job's GPUs](#3-finding-a-jobs-gpus)
4. [Reduction](#4-reduction-time-then-gpus) · 5. [Running jobs](#5-reconstructing-the-summary-for-a-running-job) · 6. [Why windows disagree](#6-why-a-recomputed-window-can-disagree)
7. [Reading the columns](#7-reading-the-columns-correctly) · 8. [MIG](#8-mig) · 9. [Verifying by hand](#9-verifying-a-number-by-hand) · 10. [Limits](#10-known-limits)

---

## 1. Two sources, one preference order

Everything jobscope prints comes from one of two places.

| | source | reached by | covers |
|---|---|---|---|
| **The jobstats summary** | `sacct` `AdminComment` | one bulk `sacct` call, no network | CPU%, MEM%, GPU%, GMEM% |
| **Prometheus** | DCGM + NVML exporters | HTTP query API | everything else, and all live data |

The summary is what jobstats stores when a job *ends*: `JS1:` followed by base64-encoded
gzipped JSON holding, per node, `total_time` (CPU-seconds), `cpus`, `used_memory`,
`total_memory`, and the per-GPU maps `gpu_utilization` / `gpu_used_memory` /
`gpu_total_memory`, plus a top-level `total_time` (elapsed wall seconds).

```json
{ "total_time": 12161,
  "nodes": { "holygpu8a10302": {
      "cpus": 16, "total_time": 12970.9,
      "used_memory": 2530840576, "total_memory": 137438953472,
      "gpu_utilization":  {"3": 93.1},
      "gpu_used_memory":  {"3": 19776995328},
      "gpu_total_memory": {"3": 150754820096} } } }
```

GPU maps are keyed by `minor_number` **as a string**.

### Preference order

One source of truth per number, chosen by job state:

| job state | CPU% / MEM% / GPU% / GMEM% / GMEM_GB | other DCGM columns |
|---|---|---|
| finished, summary present | **the jobstats summary**, always | Prometheus |
| finished, summary absent or `JS1:Short` | blank | Prometheus |
| running | **Prometheus, shaped as a summary** ([§5](#5-reconstructing-the-summary-for-a-running-job)) | Prometheus |

**A finished job's utilization is never recomputed.** The summary is what Slurm
recorded, so every view reports the same number, and re-deriving it would reintroduce
the boundary disagreement in [§6](#6-why-a-recomputed-window-can-disagree).

A site can override this per column with `[gpu] source` / `[host] source` — see
[`admin.md`](admin.md).

### One column set, one renderer

Every per-job report renders one row per job, so a job reads the same either side of
its end. Reports differ only in how jobs are selected (`sacct` versus `squeue`) and how
wide the profiling block is. Both paths yield the same chunks, so no renderer knows
which it got. Two details let the squeue side pass for the sacct side:

- Running-job records synthesize the summary Slurm has not written yet ([§5](#5-reconstructing-the-summary-for-a-running-job)).
- The `cgroup_*` queries are batched across every selected job — four queries total
  rather than four per job. Per-job round trips made a cluster-wide running view
  unusable: 8000 running jobs meant 32,000 queries.

`GMEM%` is derived (`GMEM_GB / GMEM_TOTAL_GB`) rather than queried, and `GMEM_TOTAL_GB`
is fetched only to feed it. The summary and detail views omit `GPU%` and the `GMEM`
columns from their *DCGM* set because they already render those from the summary — one
number, one column.

---

## 2. The two exporters

Both run on every GPU compute node, scraped locally and remote-written to Prometheus.

| | port | prefix | GPU UUID label | carries `minor_number`? |
|---|---|---|---|---|
| NVML exporter | 9445 | `nvidia_gpu_*` | lowercase `uuid` | **yes** |
| dcgm-exporter | 9400 | `DCGM_FI_*` | uppercase `UUID` | no (`gpu` is a different index) |

Three consequences, all of which the code works around:

- **The label case differs.** `DCGM_FI_PROF_SM_ACTIVE{UUID=...}` matches; `{uuid=...}`
  returns nothing.
- **A PromQL `and` cannot join across them.** `and` requires identical label sets, and
  `DCGM_FI_*` adds `Hostname`, `device`, `gpu`, `pci_bus_id`, `modelName`. So the
  ownership clip in [§4](#the-ownership-clip) applies to `nvidia_*` metrics only.
- **Only NVML knows the Slurm GPU number.** The DCGM `gpu` label is *not*
  `minor_number`. Everything is joined on UUID.

Host CPU and memory come from a third set, `cgroup_*`, which unlike the GPU series
carries a real `jobid` **label**.

---

## 3. Finding a job's GPUs

### The job ID is a metric value, not a label

`nvidia_gpu_jobId` reports the owning job as its **sample value**, so there is no label
to filter on:

```
nvidia_gpu_jobId{uuid="GPU-bab5106b-...", minor_number="0", host="holygpu8a10302"}  3.4853925e+07
```

It is exposed in scientific notation, hence parsed as `int(float(...))`.

Two strategies follow, and the difference matters:

- **Historical** filters server-side per job, evaluated at the job's end. Correct for a
  job whose window is known.
- **Running** issues one *unwindowed* instant query for all of `nvidia_gpu_jobId` and
  filters client-side. Not an optimization — a single GPU can host a dozen jobs in a
  day, so any windowed lookup would hand the same GPU to every job that touched it. One
  query returns ~2100 series in about 0.2 s however many jobs are selected.

### Raw versus display job IDs

**Every Prometheus series keys on the raw per-element job ID**, which differs from the
ID users type for array jobs:

```
sacct -j 34843528_6 -o JobID,JobIDRaw
34843528_6  | 34843629      <- what Prometheus stores
```

| specifier | meaning | array element | plain job |
|---|---|---|---|
| `%i` | display ID | `34843528_6` | `34622920` |
| `%A` | **raw per-element ID** | `34843629` | `34622920` |
| `%F` | array parent | `34843528` | `34622920` |

Using the display ID, or stripping it to the array parent, matches nothing. This applies
to `cgroup_*` too, despite its real `jobid` label: element `36410890_2` appears as
`jobid="36410916"`.

### GPU identity: UUID, not minor number

`minor_number` is **not unique per schedulable GPU** — two nodes each have a minor 0,
and on a MIG node every instance inherits its parent card's `minor_number` *and*
`ordinal`. Only the UUID is unique, and its prefix says what the device is: `GPU-…` for
a whole card, `MIG-…` for an instance. The running view keys rows by UUID and labels
them `GPU 0` or `MIG 0.1`.

---

## 4. Reduction: time, then GPUs

Two reductions apply to every Prometheus-derived number, and mixing them up is the most
common source of a surprising value.

### Over time

| reducer | PromQL | used for |
|---|---|---|
| `avg` | `avg_over_time(...)` | all utilization and power columns |
| `max` | `max_over_time(...)` | memory (`GMEM_GB`, `FB_USED_GB`, `PWRmax_W`) |
| `delta` | `max_over_time(...) - min_over_time(...)` | `ENERGY_kWh`, a monotonic counter |

**Utilization is averaged; memory is peaked**, mirroring jobstats, whose report labels
GPU memory "maximum used/total". Averaging a peak, or peaking an average, silently
changes the meaning.

The window is the job's runtime `[start, end]` — **once the job has ended.** A job still
running has no closed window to fold, so an unfinished job reports its **newest scrape**
and the reduction is dropped; `--runtime-avg` asks for the fold anyway. `delta` keeps
its window either way: one sample of a counter has no difference to report.

The choice is made per record, not per selection, so a set holding both states reports
each on its own rule. The header's `Sampled:` line states which you are looking at.
`CPU%`/`MEM%` are absent from that line on purpose — they are cumulative whatever the
window.

### The ownership clip

For `nvidia_*` metrics the selector is intersected with the job's ownership of the GPU:

```promql
avg_over_time((nvidia_gpu_duty_cycle{uuid=~"..."} and nvidia_gpu_jobId == 34843629)[7200s:])
```

**This is not an optimization.** The owning job is a *label* on the nvml series, so one
card carries one series **per job that has ever held it** — measured on one A100 over
two hours: ten `nvidia_gpu_duty_cycle` series for a single UUID, with means from 0.00 to
100.00. Selecting by UUID alone returns all of them and whichever arrives last wins. Nor
is it only a long-window problem: the exporter keeps publishing the previous owner's
series for a scrape or two after that job ends, so even a window exactly as long as the
job's runtime can contain a foreign series. Without the clip, one job running at 73%
reported `GPU% 0`.

`DCGM_FI_*` metrics cannot be clipped ([§2](#2-the-two-exporters)) and are bounded by
the window alone.

### Across a job's GPUs

**Within a job**, `GPU%` is the mean over the GPUs the summary reports, so a 4-GPU job
with one idle card reads 75%. `GMEM%` divides summed used by summed total, which is
capacity-weighted — the difference shows only on cards of unequal size. `ENERGY_kWh`
sums and `PWRmax_W` takes the max.

Aggregation runs over **UUIDs**, not `(node, minor)` pairs, so MIG siblings are not
silently dropped from a job-level mean.

**Across jobs there is deliberately no mean.** Utilization is bimodal — jobs cluster
near 0% or 100% — so the average lands where few jobs live and describes none of them.
Measured on one partition over one day, 385 GPU jobs split 302 at 75–100% against 11 at
0–5%; the per-job mean read 82% while the partition was 66% idle, and the per-job median
(89%) was worse still.

### The pooled row

What is printed instead is a pooled ratio: used resource-time over allocated
resource-time. Being a ratio of totals rather than a centre, it stays meaningful
whatever the shape of the distribution. Each column is pooled over the resource *it*
measures:

| column | weight |
|---|---|
| `GPU%`, `GMEM%`, DCGM mean metrics | GPU-seconds (`#GPU` × elapsed) |
| `CPU%` | core-seconds (allocated cores × elapsed) |
| `MEM%` | byte-seconds (allocated memory × elapsed) |

Weighting `CPU%` by core-seconds is exact, not merely reasonable: per job `CPU%` is
`100 × cpu_seconds / (elapsed × cores)`, so summing numerator and denominator across the
selection is identical to averaging the per-job values with weight `elapsed × cores`.

Including elapsed time keeps a swarm of short jobs from drowning out a long one: 100
five-minute jobs idling at 0% against one two-day job at 100% average to 1% per job, but
the long job is 85% of the GPU-hours.

Time weighting applies only where each value already spans its job's runtime. Under
`--instant` the weights are bare resource counts, because multiplying one scrape by two
days of elapsed time would assert that the instant represents those two days. That is
also what turns the label from `Used/GPU:` into `Used/GPU-hr:`.

### Which jobs contribute

| the job | contributes to the pooled GPU figure? |
|---|---|
| no GPU allocated | **no** — there is no GPU% to pool, and counting it would dilute |
| GPU allocated, sat idle | **yes, as 0** — this is the case worth finding |
| GPU allocated, no samples | **no** — absence of data is not evidence of 0% use |
| 4 allocated, 2 reported | the mean of the 2 that reported |

The last two lean the same way on purpose: jobscope never invents a zero for a GPU it
has no measurement of, because a retention gap would then read as waste that was never
observed. The cost is that such jobs quietly leave the figure, which is why the `Jobs:`
footer prints both totals.

A job with **no stored summary** is excluded from every tally and counted as
`no-jobstats=N`: without it a job has Prometheus numbers but no
`CPU%`/`MEM%`/`GPU%`/`GMEM%`, and letting it vote in the DCGM tallies alone put 117 jobs
behind `SM_ACT%` against 88 behind `GPU%` on one partition. It stays in the listing —
it ran. Note this is a *missing* summary, not a CPU-only one: a CPU-only job's summary
exists and simply carries no GPU data.

A job whose elapsed time is unknown cannot be placed on the resource-hour scale at all,
so it is dropped from the row and counted as `no-runtime=N`.

Bands are computed from the **stored** value, not the printed one. A job whose `OCC%`
prints as `15.0` may be 14.96 and therefore red against a cutoff of 15; banding the
display string would make the report depend on its own formatting.

> How the pooled row, the bands and the `Worst` rows are laid out and read is
> [`reference.md`](reference.md#the-report-block) — including why **green is not the
> same as efficient**.

---

## 5. Reconstructing the summary for a running job

Slurm writes the summary at job end, so a running job has none. jobscope rebuilds one
from Prometheus *in the summary's own shape*, so every view, `--csv` and `plot` work
unchanged.

| summary field | query | reducer |
|---|---|---|
| `cpus` | `cgroup_cpus{jobid='<raw>',step='',task=''}` | max |
| `total_time` (per node) | `cgroup_cpu_total_seconds{...}` | max |
| `used_memory` | `cgroup_memory_rss_bytes{...}` | max |
| `total_memory` | `cgroup_memory_total_bytes{...}` | max |
| `gpu_utilization` | `nvidia_gpu_duty_cycle and nvidia_gpu_jobId == <raw>` | **avg** |
| `gpu_used_memory` | `nvidia_gpu_memory_used_bytes and ...` | max |
| `gpu_total_memory` | `nvidia_gpu_memory_total_bytes and ...` | max |

Details that matter:

- The `jobid` matcher takes the **raw** ID ([§3](#raw-versus-display-job-ids)).
- `step=''` / `task=''` select the job-level cgroup rather than a per-step one. An `=''`
  matcher also matches the label being absent, which is the case on exporters that do
  not emit it.
- Values are rounded to the precision the stored summary uses, or reconstructed rows
  print `93.1386%` beside stored rows printing `93.1`.
- Failure degrades to blank columns. With no Prometheus endpoint configured at all the
  views say so, rather than printing dashes that look like idleness.

Because the GPU part uses exactly the query behind `--runtime-avg`, a running job's
`GPU%` equals the mean of its `--runtime-avg` values by construction.

---

## 6. Why a recomputed window can disagree

Worth understanding before trusting any recomputed utilization figure.

`GPU%` from the summary and `GPU%` recomputed from Prometheus are the same metric under
the same reducer, yet can differ by several points on a **short** job. The cause is the
window boundary: the summary is what jobstats computed at job end, while a recomputation
reconstructs the window from sacct's `Start` and `End`.

Measured on a 570 s job with one GPU (60 s scrape interval, so ~10 samples):

```
raw samples inside [start,end]:  100, 0, 94, 93, 89, 92, 79, 86, 86, 80
mean of those                    79.9
stored summary                   77.9
recomputed at sacct's End        72.5
recomputed 30 s earlier          77.7
```

One sample is worth roughly 8 points at this length, and the job's ramp-down sits just
outside the window. It is not the clip (clipped and unclipped both give 72.5) and not
subquery step alignment.

Two conclusions:

1. **Prefer the stored value when it exists** — the rule in [§1](#preference-order). It
   makes the views agree and removes the boundary question.
2. **Agreement improves with window length.** Across 1270 active GPUs, comparing
   `DCGM_FI_PROF_GR_ENGINE_ACTIVE` with `nvidia_gpu_duty_cycle`:

   | window | median &#124;Δ&#124; | p95 &#124;Δ&#124; | r |
   |---|---|---|---|
   | instant | 1.97 | 53.3 | 0.876 |
   | 5 min | 2.18 | 23.5 | 0.937 |
   | 1 hour | **1.19** | 7.2 | **0.9914** |

   The instantaneous outliers are sampling artifacts: the two exporters scrape on
   independent 60 s cadences.

---

## 7. Reading the columns correctly

`GPU%` and `SM_ACT%` are **not interchangeable**, and the gap between them is the useful
signal:

- `GPU%` / `ENGINE%` — *was the GPU busy at all* (any kernel resident)
- `SM_ACT%` — *how much of the GPU's width was engaged*
- `OCC%` — *how full the engaged SMs were*

Across 1270 active GPUs over a 1 h average, signed difference against `GPU%`:

| | mean signed Δ | below | above |
|---|---|---|---|
| `GR_ENGINE_ACTIVE` | −0.43 | 29% | 23% |
| `SM_ACTIVE` | **−19.86** | **93%** | 2% |

So `ENGINE%` is an unbiased stand-in for `GPU%`, while `SM_ACT%` runs ~20 points lower
and **must not be read as "GPU utilization"**. A job reading `91 / 65 / 36` for
GPU%/SM_ACT%/OCC% was never idle, but spread its kernels over about two-thirds of the
SMs and filled about a third of the warp slots — a single utilization number cannot show
that.

`GMEM_GB` (NVML) and `FB_USED_GB` (DCGM) both report used framebuffer from different
exporters and disagree by a few tenths of a GiB. `GMEM_GB` is the jobstats-comparable
one.

Note the `G`: a bare `MEM%` means **host** memory, so GPU memory is always `GMEM*`.

### Power

`POWER_W` is graded in watts against a floor, not banded as a percentage. Idle draw is a
property of the card — 27 W to 165 W across one fleet — so an idle RTX PRO 6000 outdraws
a working V100, and `[eff.floor.power]` sets it per model. The model is read
from the exporter's `name` label on a query both paths already make.

Its waste is the GPU-hours held while **below** the floor, all of it or none. Scaling by
how far below would imply 50 W wastes twice what 100 W does, and watts are not
utilization.

Why include it when it largely agrees with `GPU%`? It is the one idle signal a duty
cycle cannot fake: a job holding a trivial kernel resident reads busy on `GPU%` and draws
idle watts. Measured over one day the two did agree — the four lowest-power jobs sat at
73–74 W with `GPU% 0` — but power is not a restatement: r(POWER, GPU%) = 0.69 against
r(GPU%, SM_ACT%) = 0.76. It also covers 35 jobs the jobstats metrics miss.

---

## 8. MIG

On a partitioned node the two exporters disagree about what a GPU *is*: **NVML** reports
each MIG *instance* with a `MIG-…` UUID, each inheriting its parent's `minor_number`;
**dcgm-exporter** reports the *physical* card under its `GPU-…` UUID, distinguishing
instances by a separate `GPU_I_ID` label.

- **`--ts` is MIG-correct**: keyed by UUID, one row per instance, labelled `MIG n.i`. A
  slice's `memory_total` is the *slice*, so its `GMEM%` is per-slice.
- **DCGM columns read `-` on a MIG row.** A `MIG-…` UUID never equals a `GPU-…` one and
  nothing maps between them. Attributing the whole card's DCGM values to one slice would
  be actively misleading.
- **`GPU%` is unavailable on MIG.** NVML does not report `utilization.gpu` when MIG is
  enabled: on one node `jobId` returned 11 series while `duty_cycle` returned 3 — only
  its non-MIG cards. A fully partitioned node reports none.
- **The historical views collapse MIG rows**, keying per-GPU data by `(node, minor)`,
  which siblings share. `--ts` is the accurate view for MIG.

---

## 9. Verifying a number by hand

The exporters are reachable directly from a login node, bypassing Prometheus, the scrape
delay and the reduction machinery:

```bash
curl -s http://holygpu8a10302:9445/metrics | grep nvidia_gpu_duty_cycle
# nvidia_gpu_duty_cycle{minor_number="0",name="NVIDIA H200",uuid="GPU-bab5106b-..."} 99
```

To check a reduced value, resolve the raw ID first, then reproduce the query:

```bash
sacct -j 34843528_6 -o JobID,JobIDRaw,Start,End,Elapsed
```

```promql
nvidia_gpu_jobId == 34843629                       # which GPUs, and when they were owned
avg_over_time((nvidia_gpu_duty_cycle and nvidia_gpu_jobId == 34843629)[7200s:])
nvidia_gpu_duty_cycle{uuid="GPU-..."}              # every raw sample, to see the shape
```

`jobscope --ts` emits that last view as CSV, and pipes into `jobscope plot`.

The endpoint commonly embeds a credential, so jobscope never prints it; URLs reaching
help text or errors are masked. `jobscope probe` masks it too
(`https://***@host/path`), so its output is safe to paste into a ticket.

---

## 10. Known limits

- Recomputed utilization is boundary-sensitive on short jobs ([§6](#6-why-a-recomputed-window-can-disagree)).
  The preference order avoids it for finished jobs; there is no summary to prefer for
  running ones.
- DCGM columns cannot be clipped to GPU ownership ([§4](#the-ownership-clip)), so on a
  GPU that changed hands mid-window they may include a neighbouring job's samples.
  `nvidia_*` columns are clipped and do not have this problem.
- The running view's ownership query is not scoped to a cluster, so a Prometheus serving
  several clusters could in principle collide on job ID.
- MIG: no DCGM columns, no `GPU%`, and collapsed rows in the historical views ([§8](#8-mig)).
- **The running view's runtime average costs roughly one query per job per metric** —
  PromQL cannot vary a window per series — where the newest scrape is one query per
  metric for the whole selection. Measured on one 110-job partition: 628 queries against
  12.

  That cost is **paced, not capped**. The selection is queried in batches and each
  batch's rows print as they land, so a wide sweep is a table filling in rather than a
  blank wait — first row in about two seconds, summary at ten, on that partition.
  `[prometheus] max_queries_per_second` and `query_burst` hold the rate the server sees;
  the burst keeps this invisible for ordinary use. Pacing spreads queries out and does
  not remove any — `--min-elapsed` is the only thing that reduces the count.
  `--instant` declines the average outright and stays a single grouped query.

  Above `[defaults] max_running_jobs` (2000) a running selection is refused rather than
  swept: a backstop against a typo, not a cost policy.
