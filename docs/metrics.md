# How jobscope gathers its metrics

Reference for the data pipeline behind every column: where each number comes
from, how it is reduced over time and across GPUs, and which source wins when two
could answer. Read this when a number looks wrong, when two views disagree, or
before adding a metric.

Companion documents: `jobscope describe` (column reference) and `jobscope describe
--dcgm --ext` (the full metric catalog). `--per-gpu` and `--ts` give per-GPU rows, and `--plot_ts` charts that series in
place of writing it. Both time-series flags take an optional window (`--ts 1h`),
which narrows the range queries to the end of the run rather than filtering rows,
and `--ts --stats` reduces the series to min/mean/max/last per GPU per metric
without querying anything further -- `--stats-per-node` and `--stats-per-job`
pool the same samples over a host's GPUs and over the whole job. `--classify`
turns those means into a verdict per job -- wasteful / inefficient / needs
improvement / average / good, from each job's *best* %-metric, with `GMEM%`
excluded and a `POWER_W` reading below the floor forcing wasteful.

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

| job state | CPU% / MEM% / GPU% / GMEM% / GMEM_GB | other DCGM columns |
|---|---|---|
| finished, blob present | **the blob**, always | Prometheus |
| finished, blob absent or `JS1:Short` | blank | Prometheus |
| running | **Prometheus, shaped as a blob** (§5) | Prometheus |

### One column set, one renderer

Every per-job report renders through `SummaryRenderer`, one row per job, so a job
reads the same either side of its end:

```
JOBID  USER  STATE  NODE  CPU%  MEM%  #GPU  GPU%  GMEM%  SM_ACT%  OCC%  TENSOR%  DRAM%  POWER_W  RUNTIME
```

Reports differ only in how jobs are selected (`sacct` versus `squeue`, behind
`jobscope/select.py`) and how wide the profiling block is (`--dcgm`). Because the
columns are a pure function of the spec list the renderer is handed, the modes
cannot drift apart.

`select.resolve` is what makes that true: it yields the same
`(jobids, records, dcgm_data)` chunks from either source, so no renderer knows
which it got. Two details let the squeue side pass for the sacct side:

- `live.live_records` synthesizes the blob Slurm has not written yet (§5), so a
  running job looks like a record with stored stats.
- `live_blob.host_stats_many` batches the `cgroup_*` queries across every selected
  job -- four queries in total rather than four per job. Those series are per-job
  and do not exist outside their job's lifetime, so one shared window (the longest
  job's) cannot pull another job's samples in. Per-job round trips made a
  cluster-wide live view unusable: 8000 running jobs meant 32000 queries.

`NODE` is the node count and `#GPU` the allocated GPU count. Under `running`,
`STATE` is always `RUNNING`, and `CPU%`/`MEM%` are cumulative in both modes --
CPU-seconds over elapsed x cores, and peak RSS, neither of which has an
instantaneous form -- while the GPU columns follow the instant-versus-`--avg`
choice.

**Per-GPU output** is `--per-gpu` and the `--ts` time series, which stay one row
per GPU. `--per-gpu` closes each job block with the efficiency chart repeated per
node -- or per GPU once a single node is in play, that being the only thing left that
distinguishes the rows. A node's value is the mean over its GPU rows, which within a
node is the pooled figure; `CPU%` is already a per-node number repeated on each row,
so averaging returns it unchanged. `--nodename=NODE` narrows the rows to one node,
in both views -- for `--ts` before the range queries are issued, so the skipped
nodes are never fetched. `--ts` keys by UUID throughout, so it is the accurate view on a MIG node;
`--per-gpu` keys by `(node, minor)` like the blob does, which MIG siblings share.

`GMEM%` is derived (`GMEM_GB / GMEM_TOTAL_GB`) rather than queried, and
`GMEM_TOTAL_GB` is fetched only to feed it, so it is not a column of its own. The
summary and detail views omit `GPU%` and the `GMEM` columns from their *DCGM* set
because they already render those from the blob -- one number, one column.

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
- **Running** (`jobscope/live.py`) issues one *unwindowed* instant query for all of
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
| `max` | `max_over_time(...)` | memory (`GMEM_GB`, `FB_USED_GB`, `PWRmax_W`) |
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

Two levels of averaging apply, and they answer different questions.

**Within a job**, `GPU%` is the mean over the GPUs the blob reports for it, so a
4-GPU job with one idle card reads 75%. `GMEM%` instead divides summed used by
summed total, which is capacity-weighted -- the difference only shows on cards of
unequal size.

**Across jobs**, there is deliberately **no mean**. Utilization is bimodal --
jobs cluster near 0% or near 100% -- so the average lands in a range where few
jobs live and describes none of them. Measured on one partition over one day,
385 GPU jobs split 302 at 75-100% against 11 at 0-5%; the per-job mean read 82%
while the partition was 66% idle, and the per-job median (89%) was worse still.

What is printed instead is a pooled ratio plus a distribution.

### The pooled row

`Used/GPU-hr:` (finished jobs, `running --avg`, an explicit job ID) or
`Used/GPU:` (the instantaneous running view) is used resource-time over allocated
resource-time. Being a ratio of totals rather than a centre, it stays meaningful
whatever the shape of the distribution. Each column is pooled over the resource
*it* measures:

| column | weight |
|---|---|
| `GPU%`, `GMEM%`, DCGM mean metrics | GPU-seconds (`#GPU` x elapsed) |
| `CPU%` | core-seconds (allocated cores x elapsed) |
| `MEM%` | byte-seconds (allocated memory x elapsed) |

Weighting `CPU%` by core-seconds is exact, not merely reasonable: per job `CPU%`
is `100 x cpu_seconds / (elapsed x cores)`, so summing numerator and denominator
across the selection is identical to averaging the per-job values with weight
`elapsed x cores`.

Including elapsed time is what keeps a swarm of short jobs from drowning out a
long one: 100 five-minute jobs idling at 0% against one two-day job at 100%
average to 1% per job, but the long job is 85% of the GPU-hours. Time weighting
is applied only where each value already spans its job's runtime. In the
instantaneous running view the weights are bare resource counts, because every
value there is a single scrape at the same moment and multiplying one by two days
of elapsed time would assert that the instant represents those two days.

`ENERGY_kWh` sums over a job's GPUs and `PWRmax_W` takes the max
(`MetricSpec.agg`), so neither has a pooled form; they fall back to the plain
per-job figure rather than a weighting that would mean nothing.

A job whose elapsed time is unknown cannot be placed on the resource-hour scale
at all, so it is dropped from the row and counted as `no-runtime=N` in the
`Jobs:` footer.

### The efficiency block

One table row per graded metric, in the same order as the columns above it, so
the block cannot drift from the table it summarizes. It prints for a **single job**
as well, where each metric shows a single `1` in the band its value falls in -- the
job row gives the numbers, the table says where they sit. For one job the rest of
the block is suppressed: the pooled row would repeat that job's own row, a `Worst`
row would name it again, and every job count would be 1. The set follows the view:
eight rows by default, `CPU%`/`MEM%` under `--cpu`, six under `--gpu`, the full
catalog (18) under `--dcgm`. A metric that no job reported is omitted rather than
printed as zeros, which would read as "nothing used it" instead of "nothing
measured it".

```
METRIC   IDLE            RED  YELLOW  GREEN
CPU%     10034.5h (95%)  159  158     2
MEM%     129.6TBh (96%)  307  8       4
GPU%     408.8h (54%)    5    8       306
GMEM%    528.9h (70%)    299  9       11
SM_ACT%  452h (60%)      35   17      373
```

Each metric is measured against the resource it is a percentage *of*, taken from
the same `_weights()` the pooled row uses so the two cannot disagree:

| metric | weight | unit |
|---|---|---|
| `CPU%` | allocated cores x elapsed | core-hours |
| `MEM%` | allocated host bytes x elapsed | GB-hours, promoted to TB-hours past four digits |
| `GPU%`, `GMEM%`, every DCGM `%` | allocated GPUs x elapsed | GPU-hours |

The denominators therefore differ by row, deliberately: reading down `IDLE` shows
which resource a selection actually wasted. Above, the GPUs were 56% idle while
the cores were 95% idle -- GPU jobs holding cores they never use, which blocks
other work from those nodes and no GPU row can show.

`IDLE` carries one decimal with trailing `.0` trimmed, in that row's own unit, plus
its share of the allocation. It is fractional even in the count form -- what is idle
is GPU-*equivalents*, not whole GPUs -- so rounding to an integer would make it
contradict the percentage beside it. `ALLOC` and `USED` were dropped: `IDLE` already
carries the same information in the form anyone acts on, and the band cells were
reduced to job counts for the same reason. Both survive in the CSV.

One cutoff, `[thresholds] red`, covers every percentage metric: red below it,
yellow below twice it, green above. Uniform on purpose -- the per-metric values it
replaced were never calibrated against each other, and carrying a different
threshold for each row is what made the old cutoff column confusing. `POWER_W` has
its own knob because watts are not a percentage. A three-line legend above the table
states the cutoffs, what `IDLE` counts, and why a green band is not the same as an
efficient one.

### Three sections

Everything after the per-job listing is presented as three numbered sections, ruled
to their own widths: **Summary by metric** (the pooled row and the table),
**Average efficiency** (the bars), and **Problem jobs** (the `Worst` rows and the
`Jobs:` counts). Numbering runs over the sections that actually have content, so a
suppressed or empty one leaves no gap -- a missing number would read as a failure.
Headings and rules follow `--noheader`; `--csv` and `--ts` carry none of it, `--ts`
structurally so, since it never constructs a `SummaryRenderer`.

### The efficiency bars

Shown by default, omitted with `--no-plot`. One horizontal bar per graded metric: length is the pooled
utilization, the filled run tinted by the band that value falls in. It is the `IDLE`
column read the other way round -- bar percent plus `IDLE` percent is 100 for every
metric, because both derive from `EfficiencyTally.pooled()` -- so a chart and the
table it sits under cannot disagree.

It reuses the table's metric list, so the set follows `--cpu` / `--gpu` / `--dcgm`
and omits whatever no job reported, minus `POWER_W`: it has a row but no bar, since
its "used" is time above the watt floor rather than a fraction of a resource, and
drawing that as an efficiency bar makes idle-but-powered GPUs look like the healthy
ones. A nonzero utilization always draws at least one block, since an empty bar
beside a "1%" contradicts itself.

Drawn with block characters and the report's own SGR codes rather than `rich`: the
report path is the common one and should not import a rendering library to print a
table. Suppressed under `--csv`, and its title follows `--noheader`.

### What green does not mean

Green is `>= 2x` the red cutoff, which is a low bar: with `cpu = 10` a job at 21%
is green while leaving four fifths of its cores unused. A selection can therefore
be half idle with almost every job green, which reads as a contradiction until the
two columns are separated:

```
METRIC   IDLE        RED  YELLOW  GREEN
CPU%     84.3 (49%)  0    1       13
```

Eleven jobs, each using about half its cores (12, 21, 47, 52, 55, 55, 55, 55, 56,
56, 56). None is below 10, so the red band is empty; pooled, 49% of the cores are
idle anyway.

`IDLE` is the efficiency measure. The bands say *where* the waste sits:

| pattern | reading |
|---|---|
| red band holds a large share of the **resource-time** | concentrated: a few jobs waste a lot, and `Worst` names them |
| red band empty but `IDLE` high | systemic: every job wastes a little, which is a habit rather than an incident |

Measured on one partition, `GPU%` showed the first (4% of jobs, 53% of the
GPU-hours, red) and `CPU%` the second. The bands are deliberately calibrated to
catch pathological jobs rather than to score efficiency, since the thresholds that
would score efficiency differ per workload -- inference, data prep and sparse HPC
all run legitimately low.

The three band cells give each band's share of the **jobs** and of the
**resource-time**: `13 (4%)/54%` is 13 jobs, 4% of those measured, holding 54% of
the GPU-hours. The gap between the two is the finding -- 4% of the jobs held 54%
of the GPU-hours below 25% -- and either share alone conceals it. The bands come
from `config.grade_band` and the site's `[thresholds]`, the same cutoffs that tint
the cells and colour `jobscope plot`, so the block is a tally of what is already
on screen rather than a second opinion.

Bands are computed from the **stored** value, not the printed one. A job whose
`OCC%` prints as `15.0` may be 14.96 and therefore red against a cutoff of 15;
banding the display string would make the report depend on its own formatting.

On a terminal each band cell is printed in its own colour and `IDLE` is tinted by
that metric's pooled grade. Colour is dropped for `--csv`, a non-tty and
`$NO_COLOR`, and the plain output is the tinted output minus the escapes -- the
final column is left unpadded so that stays exactly true.

A job with **no stored blob** is excluded from every tally and counted as
`no-blob=N`. Slurm writes the blob at job end, so without it a job has Prometheus
numbers but no `CPU%`/`MEM%`/`GPU%`/`GMEM%`; letting it vote in the DCGM tallies
alone put 117 jobs behind `SM_ACT%` against 88 behind `GPU%` on one partition, and a
job cannot be ranked against the rest on a metric it has no value for. It stays in
the listing regardless -- it ran, and its DCGM numbers are shown on its own row.
Note this is a *missing* blob, not a CPU-only one: a CPU-only job's blob exists and
simply carries no GPU data, so it still votes on `CPU%` and `MEM%`.

The `Worst` rows name the top few jobs by resource-time **wasted**,
`(1 - u) x weight`, not by resource-time held: a 100-hour job at 24% is a larger
finding than a 10-hour job at 0%.

Entries are grouped under their owner, in rank order rather than alphabetically, so
the first user named owns the worst job. Each reads `jobid:value:wasted(elapsed)`,
and a job whose elapsed time exceeds `report.LONG_RUNNING` (three hours) is printed
red: a brief bad job costs little next to hours of idle hardware. Lines wrap onto
continuation lines indented under the user column rather than running past the table,
and the wrap is measured on visible characters so the red escapes do not shorten it. There is one row per measure -- `GPU%`,
`SM_ACT%`, `POWER_W`, `CPU%` -- and a row is omitted when no job falls in that
measure's red band. Four rather than every graded column: these say distinct
things, while the DCGM catalog would add a dozen near-duplicates.

### Power, the one metric that is not a percentage

`POWER_W` is graded in watts against `[thresholds] power_w` (default 100). Before
that existed it fell through to the `%` default of 15 -- 15 *watts* -- so every
power cell graded green, in the table and in `jobscope plot`.

Its waste is the GPU-hours held while **below** the floor, all of it or none:

```
waste = weight if watts < power_w else 0
```

A floor asserts idle-or-not, and nothing finer is available. Scaling by how far
below would imply 50 W wastes twice what 100 W does, and watts are not utilization.

Why include it at all, when it largely agrees with `GPU%`? Because it is the one
idle signal a duty cycle cannot fake: a job holding a trivial kernel resident reads
busy on `GPU%` and draws idle watts. Measured over one day on kempner_eng the two
did agree -- the four lowest-power jobs sat at 73-74 W with `GPU% 0` and
`SM_ACT% 0.0`, and the top three of every ranking were the same jobs -- but power
is not a restatement of them: r(POWER, GPU%) = 0.69 and r(POWER, SM_ACT%) = 0.64,
against r(GPU%, SM_ACT%) = 0.76. It also covers 35 jobs the blob metrics miss (no
stored blob), though those held only 0.6 of 349.2 GPU-hours.

`POWER_W` does get a stats-table row. "Used watts" has no meaning as a total, but the
resource-time that drew *less* than the floor does, and that is what its `IDLE` counts
-- all-or-nothing per sample, where a percentage's `IDLE` takes a fraction of each.
Grading it as a proportion of the cutoff instead would imply 50 W wastes twice what
100 W does, and watts are not utilization. The bands still separate the near misses:
a GPU at 119 W is yellow, not red.

### The two combined rows

`Worst both:` ranks over the two distinct resources (`GPU%`, `CPU%`);
`Worst all:` over all four measures. The measures are in different units --
GPU-hours, core-hours, GPU-hours below a watt floor -- and cannot be added: any
exchange rate is invented, and on a GPU cluster a wrong one decides the ranking by
itself. Each job's waste is therefore normalised by the selection's own total waste
in that measure and the shares summed:

```
score = sum over measures of  waste(job, measure) / total_waste(measure)
```

Each cell prints the job's **value** in every metric the row names
(`36337338 gpu0 sm0 pw70W cpu1`), all of them under their cutoffs, which is what put
the job there; power carries its unit since watts are not a percentage. The order
still carries the ranking by summed waste share. Printing the shares instead was
actively misleading -- `12%gpu` reads as a utilization of 12%, the inverse of the
row's meaning. A row is omitted when any of its measures wasted nothing, since a
share of a zero total is undefined.

**Candidacy is a conjunction**: a job appears only if it is red in *every* metric the
row names. `Worst both:` is therefore "idle by GPU and by CPU", and `Worst all:`
"idle by all four". A disjunction put jobs on the four-metric row that were drawing
full power, recognisable by a `0%pw` component -- the row claimed more than it meant.
The cost is that a combined row is frequently absent, which is the honest answer when
no job is bad by every measure at once. Waste in a metric a job is *green* in is
still counted in that metric's own total and its own `Worst` row; it simply does not
earn a place in the conjunction.

Note what `Worst all:` costs: three of its four terms describe the same GPUs, so it
weights GPU idleness roughly 3:1 against CPU idleness. `Worst both:` is the fair
comparison between resources; `Worst all:` answers "worst by any measure".

Candidates are the jobs red in **at least one** resource, and for those the waste
in the *other* resource counts too even where they are green there -- it is real
waste; the red filter only decides who is a candidate. That filter is what keeps
the list actionable: a 95%-efficient job can idle 50 GPU-hours simply by being
enormous.



Which jobs contribute is the part worth being exact about:

| the job | contributes to the pooled GPU figure? |
|---|---|
| no GPU allocated | **no** -- there is no GPU% to pool, and counting it would dilute |
| GPU allocated, sat idle | **yes, as 0** -- this is the case worth finding, not hiding |
| GPU allocated, no samples | **no** -- absence of data is not evidence of 0% use |
| 4 allocated, 2 reported | the mean of the 2 that reported |

The last two lean the same way on purpose: jobscope never invents a zero for a
GPU it has no measurement of, because a retention gap or an unscraped short job
would then read as waste that was never observed. The cost is that such jobs
quietly leave the figure, which is why the `Jobs:` footer prints both totals --
`cpu-jobs=18 gpu-jobs=17` says one job's GPU use is unmeasured rather than zero.
The same jobs are absent from the band tally, so its shares are over measured
resource-time only.

`mean` by default; `max` for peak-like metrics; `sum` for energy. The per-job
figure in the summary view uses this; the per-GPU rows in `detail`, `dcgm` and
`live` do not reduce across GPUs at all.

Note the aggregation runs over **UUIDs**, not over `(node, minor)` pairs, so MIG
siblings are not silently dropped from a job-level mean.

### Instant versus windowed

`jobscope running` defaults to the newest single scrape — no time reduction at
all. It is the only mode that does, and it is why live numbers need not match
jobstats. `running --avg` applies the reductions above and does match.

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

Because the GPU part uses exactly the query behind `running --avg`, a running
job's `GPU%` equals the mean of its `running --avg` values by construction.

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

`GMEM_GB` (NVML) and `FB_USED_GB` (DCGM) both report used framebuffer from
different exporters and disagree by a few tenths of a GiB. `GMEM_GB` is the
jobstats-comparable one.

Note the `G`: a bare `MEM%` means **host** memory in the summary and detail views,
so GPU memory is always `GMEM*`. Reusing `MEM%` for GPU memory not only read as the
wrong quantity, it graded against the host threshold in `jobscope plot`.

---

## 8. MIG

On a partitioned node the two exporters disagree about what a GPU *is*:

- **NVML** reports each MIG *instance*, with a `MIG-…` UUID, each inheriting its
  parent card's `minor_number` and `ordinal`.
- **dcgm-exporter** reports the *physical* card under its `GPU-…` UUID,
  distinguishing instances by a separate `GPU_I_ID` label.

What follows:

- **`--ts` is MIG-correct**: keyed by UUID, one row per instance, labelled
  `MIG n.i`. A slice's `memory_total` is the *slice* (e.g. 19.6 GB of a 40 GB
  card), so its `GMEM%` is per-slice.
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
  `--ts` is the accurate view for MIG.

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

`jobscope --ts` emits exactly that last view as CSV, and pipes into
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
- The running view's ownership query is not scoped to a cluster, so a Prometheus
  serving several clusters could in principle collide on job ID.
- MIG: no DCGM columns, no `GPU%`, and collapsed rows in the historical views (§8).
- `running --avg` without a filter fans out to roughly one query per job per metric.
  With a partition or user filter this is a few seconds; unfiltered across a busy
  cluster it is thousands of queries and there is no guard.
