# How jobscope gathers its metrics

Reference for the data pipeline behind every column: where each number comes
from, how it is reduced over time and across GPUs, and which source wins when two
could answer. Read this when a number looks wrong, when two views disagree, or
before adding a metric.

Companion documents: `jobscope describe` (column reference), `jobscope describe
--dcgm --ext` (full metric catalog), `jobscope live --describe` (live columns).

---

## 1. Two sources, one preference order

Everything jobscope prints comes from one of two places.

| | source | reached by | covers |
|---|---|---|---|
| **The blob** | `sacct` `AdminComment` | one bulk `sacct` call, no network | CPU%, MEM%, GPU%, GMEM% |
| **Prometheus** | DCGM + NVML exporters | HTTP query API | everything else, and all live data |

**The blob** is what jobstats stores when a job *ends*: `JS1:` followed by
base64-encoded gzipped JSON. Decoded (`jobscope/blob.py`) it holds, per node,
`total_time` (CPU-seconds), `cpus`, `used_memory`, `total_memory`, and the per-GPU
maps `gpu_utilization` / `gpu_used_memory` / `gpu_total_memory`, plus a top-level
`total_time` (elapsed wall seconds).

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

The rule is one source of truth per number, chosen by job state:

| job state | CPU% / MEM% / GPU% / GMEM% | DCGM columns |
|---|---|---|
| finished, blob present | **the blob**, always | Prometheus |
| finished, blob absent or `JS1:Short` | blank | Prometheus |
| running | **Prometheus, shaped as a blob** (§5) | Prometheus |

A finished job's utilization is never recomputed. That is deliberate: the blob is
what Slurm recorded, so every view reports the same number, and re-deriving it
would reintroduce the disagreement described in §6.

---

## 2. The two exporters

Both run on every GPU compute node. An agent on each node scrapes its own
exporters (`url` label reads `http://localhost:<port>/metrics`) and remote-writes
to Prometheus.

| | port | prefix | GPU UUID label | carries `minor_number`? |
|---|---|---|---|---|
| NVML exporter | 9445 | `nvidia_gpu_*` | lowercase `uuid` | **yes** |
| dcgm-exporter | 9400 | `DCGM_FI_*` | uppercase `UUID` | no (`gpu` is a different index) |

Three consequences, all of which the code works around:

- **The label case differs.** `DCGM_FI_PROF_SM_ACTIVE{UUID=...}` matches;
  `{uuid=...}` returns nothing. `MetricSpec.uuid_label` records which to use.
- **A PromQL `and` cannot join across them.** `and` requires identical label sets,
  and `DCGM_FI_*` adds `Hostname`, `device`, `gpu`, `pci_bus_id`, `modelName`. So
  the runtime clip in §4 applies to `nvidia_*` metrics only.
- **Only NVML knows the Slurm GPU number.** The DCGM `gpu` label is *not*
  `minor_number`. Everything is therefore joined on UUID.

The full NVML set is small — `duty_cycle`, `memory_total_bytes`,
`memory_used_bytes`, `jobId`, `jobUid`, `num_devices`, `ecc_errors`,
`fanspeed_percent`, `temperature_celsius`, `power_usage_milliwatts`,
`last_error` — and dcgm-exporter supplies the ~33 `DCGM_FI_*` series behind the
profiling columns.

Host CPU and memory come from a third set, `cgroup_*`, which unlike the GPU series
carries a real `jobid` **label**: `cgroup_cpus`, `cgroup_cpu_total_seconds`,
`cgroup_memory_rss_bytes`, `cgroup_memory_total_bytes`.

---

## 3. Finding a job's GPUs

### The job ID is a metric value, not a label

`nvidia_gpu_jobId` reports the owning job as its **sample value**, so there is no
label to filter on:

```
nvidia_gpu_jobId{uuid="GPU-bab5106b-...", minor_number="0", host="holygpu8a10302"}  3.4853925e+07
```

It is exposed in scientific notation, hence parsed as `int(float(...))`.

Two strategies follow from that, and the difference matters:

- **Historical** (`jobscope/dcgm.py`) filters server-side per job:
  `max_over_time((nvidia_gpu_jobId{slurm_cluster=...} == <raw>)[<duration>s:])`
  evaluated at the job's end. Correct for a job whose window is known.
- **Live** (`jobscope/live.py`) issues one *unwindowed* instant query for all of
  `nvidia_gpu_jobId` and filters client-side by value. This is not an
  optimization — a single GPU can host a dozen jobs in a day, so any windowed
  lookup would hand the same GPU to every job that touched it. One query returns
  ~2100 series in about 0.2 s regardless of how many jobs are selected.

### Raw versus display job IDs

**Every Prometheus series keys on the raw per-element job ID**, which differs from
the ID users type for array jobs:

```
sacct -j 34843528_6 -o JobID,JobIDRaw
JobID       | JobIDRaw
34843528_6  | 34843629      <- what Prometheus stores
```

| specifier | meaning | array element | plain job |
|---|---|---|---|
| `%i` | display ID | `34843528_6` | `34622920` |
| `%A` | **raw per-element ID** | `34843629` | `34622920` |
| `%F` | array parent | `34843528` | `34622920` |

`%A` equals `%i` for non-array jobs, so one `squeue` call serves both. `sacct`
provides the same thing as `JobIDRaw` (`JobRecord.jobid_raw`). Using the display
ID, or stripping it to the array parent, matches nothing.

This applies to `cgroup_*` too, despite its having a real `jobid` label: array
element `36410890_2` appears as `jobid="36410916"`.

### GPU identity: UUID, not minor number

`minor_number` is **not unique per schedulable GPU**:

- two nodes each have a minor 0;
- on a MIG node every instance inherits its parent card's `minor_number` *and*
  `ordinal`, so a `3g.20gb` pair both report minor 0.

Only the UUID is unique, and its prefix says what the device is: `GPU-…` for a
whole card, `MIG-…` for an instance. The live view therefore keys rows by UUID and
labels them `GPU 0` or `MIG 0.1`, enumerating siblings that share a
`(job, host, minor)` by sorted UUID (NVML exposes no instance index; the ordering
is stable as long as the partitioning is).

---

## 4. Reduction: time, then GPUs

Two reductions apply to every Prometheus-derived number, and mixing them up is the
most common source of a surprising value.

### Over time (`MetricSpec.reducer`)

| reducer | PromQL | used for |
|---|---|---|
| `avg` | `avg_over_time(...)` | all utilization and power columns |
| `max` | `max_over_time(...)` | memory (`MEM_GB`, `FB_USED_GB`, `PWRmax_W`) |
| `delta` | `max_over_time(...) - min_over_time(...)` | `ENERGY_kWh`, a monotonic counter |

**Utilization is averaged; memory is peaked.** That mirrors jobstats, whose report
labels GPU memory "maximum used/total". Averaging a peak, or peaking an average,
silently changes the meaning — this is why the reducer lives in the spec rather
than at the call site.

The window is the job's runtime, `[start, end]`, expressed as a subquery
`[<duration>s:]` evaluated at `end`.

### The ownership clip

For `nvidia_*` metrics the selector is additionally intersected with the job's
ownership of the GPU:

```promql
avg_over_time((nvidia_gpu_duty_cycle{uuid=~"..."} and nvidia_gpu_jobId == 34843629)[7200s:])
```

This restricts the window to samples the job actually owned that GPU for, which
makes the window length a harmless upper bound. It works only because both series
come from the same exporter and so carry identical label sets. `DCGM_FI_*` metrics
cannot be clipped (§2) and are bounded by the window alone.

### Across a job's GPUs (`MetricSpec.agg`)

`mean` by default; `max` for peak-like metrics; `sum` for energy. The per-job
figure in the summary view uses this; the per-GPU rows in `detail`, `dcgm` and
`live` do not reduce across GPUs at all.

Note the aggregation runs over **UUIDs**, not over `(node, minor)` pairs, so MIG
siblings are not silently dropped from a job-level mean.

### Instant versus windowed

`jobscope live` defaults to the newest single scrape — no time reduction at all.
This is the only view that does, and it is why live numbers need not match
jobstats. `live --avg` applies the reductions above and does match.

---

## 5. Reconstructing the blob for a running job

Slurm writes the blob at job end, so a running job has none and its utilization
columns would be empty. `jobscope/live_blob.py` rebuilds one from Prometheus, in
the blob's own shape, so `blob_metrics` / `blob_detail` and therefore the summary
and detail views, `--csv`, `plot` and `--diagnose` all work unchanged.

| blob field | query | reducer |
|---|---|---|
| `cpus` | `cgroup_cpus{jobid='<raw>',step='',task=''}` | max |
| `total_time` (per node) | `cgroup_cpu_total_seconds{...}` | max |
| `used_memory` | `cgroup_memory_rss_bytes{...}` | max |
| `total_memory` | `cgroup_memory_total_bytes{...}` | max |
| `gpu_utilization` | `nvidia_gpu_duty_cycle and nvidia_gpu_jobId == <raw>` | **avg** |
| `gpu_used_memory` | `nvidia_gpu_memory_used_bytes and ...` | max |
| `gpu_total_memory` | `nvidia_gpu_memory_total_bytes and ...` | max |

Details that matter:

- The `jobid` matcher takes the **raw** ID (§3).
- `step=''` / `task=''` select the job-level cgroup rather than a per-step one. An
  `=''` matcher also matches the label being absent, which is the case on
  exporters that do not emit it, so the matcher is safe either way.
- Values are rounded to the precision the stored blob uses — byte counts to
  integers, utilization to one decimal — because `blob_detail` renders utilization
  with `%g` and unrounded floats print as `93.1386%`.
- Failure degrades to `{}`, i.e. the columns simply stay blank. If no Prometheus
  endpoint is configured at all, the views say so rather than printing dashes that
  look like idleness.

Because the GPU part uses exactly the query behind `live --avg`, a running job's
`GPU%` equals the mean of its `live --avg` `GPU%` values by construction.

---

## 6. Why a recomputed window can disagree

Worth understanding before trusting any recomputed utilization figure.

`GPU%` from the blob and `GPU%` recomputed from Prometheus are the same metric
under the same reducer, yet can differ by several points on a **short** job. The
cause is the window boundary: the blob is what jobstats computed at job end, while
a recomputation has to reconstruct the window from sacct's `Start` and `End`.

Measured on a 570 s job with one GPU (60 s scrape interval, so ~10 samples):

```
raw samples inside [start,end]:  100, 0, 94, 93, 89, 92, 79, 86, 86, 80
mean of those                    79.9
stored blob                      77.9
recomputed at sacct's End        72.5
recomputed 30 s earlier          77.7
```

One sample is worth roughly 8 points at this length, and the job's ramp-down sits
just outside the window. It is not the clip (clipped and unclipped both give 72.5)
and not subquery step alignment (an explicit `[570s:60s]` gives 72.5 too).

Two conclusions:

1. **Prefer the stored value when it exists** — which is the rule in §1. It makes
   the views agree and removes the boundary question entirely.
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

`GPU%` and `SM_ACT%` are not interchangeable, and the gap between them is the
useful signal:

- `GPU%` / `ENGINE%` — *was the GPU busy at all* (any kernel resident)
- `SM_ACT%` — *how much of the GPU's width was engaged*
- `OCC%` — *how full the engaged SMs were*

Across 1270 active GPUs over a 1 h average, signed difference against `GPU%`:

| | mean signed Δ | below | above |
|---|---|---|---|
| `GR_ENGINE_ACTIVE` | −0.43 | 29% | 23% |
| `SM_ACTIVE` | **−19.86** | **93%** | 2% |

So `ENGINE%` is an unbiased stand-in for `GPU%`, while `SM_ACT%` runs ~20 points
lower and **must not be read as "GPU utilization"**. A job reading
`91 / 65 / 36` for GPU%/SM_ACT%/OCC% was never idle, but spread its kernels over
only about two-thirds of the SMs and filled about a third of the warp slots — a
single utilization number cannot show that.

`MEM_GB` (NVML) and `FB_USED_GB` (DCGM) both report used framebuffer from
different exporters and disagree by a few tenths of a GiB. `MEM_GB` is the
jobstats-comparable one.

---

## 8. MIG

On a partitioned node the two exporters disagree about what a GPU *is*:

- **NVML** reports each MIG *instance*, with a `MIG-…` UUID, each inheriting its
  parent card's `minor_number` and `ordinal`.
- **dcgm-exporter** reports the *physical* card under its `GPU-…` UUID,
  distinguishing instances by a separate `GPU_I_ID` label.

What follows:

- **`jobscope live` is MIG-correct**: keyed by UUID, one row per instance, labelled
  `MIG n.i`. A slice's `memory_total` is the *slice* (e.g. 19.6 GB of a 40 GB
  card), so its `MEM%` is per-slice.
- **DCGM columns read `-` on a MIG row.** A `MIG-…` UUID never equals a `GPU-…`
  one, and nothing in the metrics maps between them — the instance shares its
  parent's `minor_number`, but nothing says which `GPU_I_ID` it is. Attributing
  the whole card's DCGM values to one slice would be actively misleading.
- **`GPU%` is unavailable on MIG.** NVML does not report `utilization.gpu` when MIG
  is enabled: on one node `jobId` and `memory_used_bytes` return 11 series each
  while `duty_cycle` returns 3 — only its non-MIG cards. A fully partitioned node
  reports none.
- **The historical views still collapse MIG rows.** `dcgm` and `detail` key per-GPU
  data by `(node, minor)`, which siblings share, so a four-instance job renders
  three rows. The blob has the same limitation, since it is keyed by minor number.
  `live` is the accurate view for MIG.

---

## 9. Verifying a number by hand

The exporters are reachable directly from a login node, which bypasses
Prometheus, the scrape delay and the reduction machinery:

```bash
curl -s http://holygpu8a10302:9445/metrics | grep nvidia_gpu_duty_cycle
# nvidia_gpu_duty_cycle{minor_number="0",name="NVIDIA H200",uuid="GPU-bab5106b-..."} 99
```

To check a reduced value, resolve the raw ID first, then reproduce the query:

```bash
sacct -j 34843528_6 -o JobID,JobIDRaw,Start,End,Elapsed
```

```promql
# which GPUs, and when they were owned
nvidia_gpu_jobId == 34843629

# the utilization figure, clipped to ownership
avg_over_time((nvidia_gpu_duty_cycle and nvidia_gpu_jobId == 34843629)[7200s:])

# every raw sample, to see the shape rather than the mean
nvidia_gpu_duty_cycle{uuid="GPU-..."}
```

`jobscope live --ts` emits exactly that last view as CSV, and pipes into
`jobscope plot`.

The endpoint is resolved from configuration (see the README) and commonly embeds a
credential, so jobscope never prints it; URLs reaching help text or error messages
are masked.

---

## 10. Known limits

- Recomputed utilization is boundary-sensitive on short jobs (§6). The preference
  order in §1 avoids it for finished jobs; there is no blob to prefer for running
  ones, where the window is `[start, now]` and the tail is still being written.
- DCGM columns cannot be clipped to GPU ownership (§4), so on a GPU that changed
  hands mid-window they may include a neighbouring job's samples. `nvidia_*`
  columns are clipped and do not have this problem.
- The live view's ownership query is not scoped to a cluster, so a Prometheus
  serving several clusters could in principle collide on job ID.
- MIG: no DCGM columns, no `GPU%`, and collapsed rows in the historical views (§8).
- `live --avg` without a filter fans out to roughly one query per job per metric.
  With a partition or user filter this is a few seconds; unfiltered across a busy
  cluster it is thousands of queries and there is no guard.
