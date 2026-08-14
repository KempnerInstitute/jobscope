# Changelog

## Unreleased

### `PARTITION` and `GPU_TYPE` are now the last two columns

Every row already said what a job *did*. Neither said what it ran **on**, so a
selection that narrowed no partition could not tell its own rows apart — three jobs on
three different cards read identically:

```console
$ jobscope -j 38418744 -j 38418770 -j 38418771
JOBID     USER       STATE     ... RUNTIME     PARTITION        GPU_TYPE
38418744  bdesinghu  COMPLETED ... 00:00:02    kempner          A100
38418770  bdesinghu  COMPLETED ... 00:00:04    kempner_h200     H200
38418771  bdesinghu  COMPLETED ... 00:00:01    kempner_rtx      RTX6K
```

`GPU_TYPE` is the model number: `A100`, `H100`, `H200`, `RTX6K` — and `V100`, `L40S`,
`A40` on clusters that have them, because it is a rule (the first letters-then-digits
token) rather than a table of this cluster's cards. An unrecognised name truncates
rather than blanking, since an empty cell beside `#GPU` reads as "no GPU".

**It costs no query.** For a finished job it comes out of the `AllocTRES` sacct already
fetches for the GPU count, so it survives `--no-dcgm` and needs no Prometheus. A
running job has no such record — squeue's format carries no allocated-TRES field — so
it falls back to the model the exporter reports. Neither available prints `-`.

`--cpu` drops `GPU_TYPE` with the rest of the GPU block: a card model has no place in a
host view, and a running `--cpu` report queries no exporter, so the column would be
dashes all the way down. `PARTITION` is identity, not a GPU fact, and survives every
view. `--all-metrics` still widens the profiling block in the middle; the last two
columns stay last.

**Breaking: `--show partition` is gone.** `PARTITION` is a fixed column now, so the
keyword would offer a second copy of a column already on the table. It is an error
naming the valid set (`account`, `name`, `cluster`) rather than a flag that parses and
does nothing, and `--show all` now means `account, cluster`. `--per-gpu`/`--per-node`
block headers no longer carry the partition label; the header block states it instead.

### The header states the account and partition scope, always

`Account:` and `Partition:` used to print only when you passed `-A`/`-p`, and not at
all for an explicit job ID. So a report covering every account looked exactly like one
narrowed to yours, and `jobscope -j 38191538` never said where the job ran:

```console
$ jobscope finished -D 1
  User:      alice
  Account:   (all accounts)          # was: no line at all
  Partition: (all partitions)        # was: no line at all
  Select:    last 1 day, completed
  Window:    2026-08-12 11:25 .. 2026-08-13 11:25
```

This is what the `User` line has always done — `(all users)` under `-a` — and what the
`Window` line exists for: a saved report has to carry its own scope.

An explicit `JOBID` bypasses those filters, so its header **reports rather than
restates**: the account and partition come off the records, and several jobs spanning
two accounts name both.

```console
$ jobscope -j 38191538
  Account:   kempner_wharper_lab     # the job's, not a filter's
  Partition: kempner_h100
```

Nothing was read, nothing is claimed — if the scheduler returned no record, those two
lines are absent rather than guessed. `--csv` gains the two rows in its leading context
block; `jobscope plot` skips to `JOBID` and is unaffected.

The header lines are the per-selection answer — what the report *covers*. The
per-row answer is the `PARTITION` column, below.

### `-A/--account` now filters running jobs

It was accepted in every mode and silently dropped in one. `RunningSelection` had no
`account` field, so the flag never reached `squeue`, and `jobscope -A other_lab`
reported **every** account you could see, with no warning:

```console
$ jobscope -A kempner_lab           # was: every account, silently
```

The empty-result message names it too, so a selection that matched nothing says which
filter to drop: `no running jobs match (user alice, account kempner_lab, running,
longer than 10m)` and `Widen it: ...; or drop -A to search every account`.

Behaviour change for scripts: a running selection passing `-A` was getting every
account and now gets one.

### `--eff` and `--verify` now judge the last 3 hours by default

Given no window of their own, both used to reduce over the job's **whole runtime**.
That answers a different question from the one they are asked: a job that ran well for
two days and stalled an hour ago still averages well, so a verdict over its lifetime
says it is fine. They now look back over `[defaults] verdict_window`, `180m` by default:

```console
$ jobscope -j 38138738 --verify
  Window:   16:16 .. 19:16   3h01m          # was the job's full runtime
```

An explicit span still wins (`--verify 30m`), and the window is settable per site:

```toml
[defaults]
verdict_window = "180m"
```

**A bare `--ts` and `--plot-ts` are deliberately unchanged** and still mean the whole
run. Those dump or chart a series rather than judging one, and truncating them would
have silently changed what `jobscope plot` is handed.

One knock-on: `--verify`'s ladder shows the collected window as its first rung, so that
column is now `3h` rather than the full job — with `[report] verify_windows`' `2h` and
`30m` sitting inside it, which is what makes them rungs *below* it.

### New `--show`: account, partition, name and cluster as table columns

Every row already named the job and its owner. `--show account,partition` adds the
other two identity fields, `--show all` adds `account`, `partition` and `cluster`, and
an unknown keyword is an error naming the valid set. `all` deliberately leaves out the
job name — free text, often templated and longer than the other two together — which
stays available as `--show name`.

```console
$ jobscope finished -u alice --show account,partition
JOBID        USER         ACCOUNT              PARTITION        STATE     NODE  CPU% ...
```

They sit after `USER` in a fixed order regardless of how the keywords are typed, so two
runs of the same report can be diffed. `--all-metrics` is unaffected — it varies only
the profiling block — and `--csv` carries the full values. On `--per-gpu`/`--per-node`
they name the job's block rather than becoming columns, since a detail row is about one
card and the account is the same on all of them.

**Opt-in because they are wide.** Measured over 31,029 jobs, an account runs to 23
characters and a partition to 22. The table streams, so a column's width is fixed before
the first row is read and there is nothing to auto-size to: a long value overflows and
pushes the row right rather than being truncated. Nothing is lost or run together.

Neither field was previously collected — `sacct` gains `Account,Partition` and `squeue`
gains `%a|%P`, both fetched unconditionally, since sacct charges for rows and not for
columns. Both parsers now derive their field count from the format string they were
built for: a count that drifted did not raise, it made the length guard skip every row,
so a selection matching thousands of jobs would have come back empty in silence.

### A wide selection now costs a fraction of what it did, and says so up front

**Who this affects:** anyone selecting more than a few hundred finished jobs —
`jobscope finished -u someone`, a partition sweep, anything with a multi-day window.

**sacct is now queried once per day of the window, not once plus once per 200 jobs.**
Selecting 8 587 jobs used to be one listing call followed by 43 `sacct -j` calls: 44
round trips, ~4.0 s. It is now one call for a one-day window and seven for a week.

The listing pass was not buying anything. It already materialised every row in the
window — measured cluster-wide over one day, 193 982 rows cost 1.66 GB of RSS inside
`sacct`, and asking for `JobID` alone cost the same as asking for all twelve fields —
and the id batching then chunked the *second* pass, the one with no memory problem.
Cutting the window instead bounds both. Records are also held per slice rather than
accumulated for the whole run.

**Streaming granularity is now sized for latency**, which is the only thing it
controls: a slice is how much is fetched, not how much is yielded. `run_capture`
buffers an `sacct` call whole, so handing out a whole slice at once would mean every
metric query for nine thousand jobs running before the first row was drawn.

Records go downstream **25 at a time**, not the 200 the id-batched path used — that
number answered an argv-size question, back when a chunk *was* one sacct call. Since
`compute_dcgm` finishes a whole chunk before any of its rows is drawn, the chunk is
exactly the wait before the table starts moving. Measured on a 9 000-job day, the first
row now appears in **4.9 s rather than 8.0 s**, for 4% more queries (2 092 per 1 000
jobs against 2 011) — still 30% below what per-job discovery cost. The rest of that 4.9 s
is fixed: 1.1 s of sacct and ~0.5 s of startup.

Unchanged: explicit `JOBID`s and `-N` still list first, because both need every id up
front. A job spanning a slice boundary is returned by both slices and reported once. A
slice that fails is named and skipped rather than ending the run.

**New `--no-dcgm`.** Drops the columns an exporter serves and queries Prometheus not at
all. `CPU%`, `MEM%`, `GPU%` and `GMEM%` still print — they come from the jobstats
summary in `AdminComment`, which arrives free with `sacct`. On an 8 889-job selection
that is ~26 700 queries against none, and 1.8 s against about nine minutes. The cost is
`SM_ACT%`, `TENSOR%`, `DRAM%` and `POWER_W`. It is the exact opposite of
`--no-jobstats` and the two are refused together.

**GPU discovery is batched, so it no longer scales with the job count.** Finding which
cards ran a job was one query per job — 8 587 of them for the selection above, a third
of everything it issued. It is now one range query per hour of the window, however many
jobs there are. Measured on 300 real jobs: 300 queries and 78 s become 4 queries and
2.1 s, with identical card sets for all 300. End to end that selection went from 900
queries to 614.

A range query lands on a fixed step grid, so a job shorter than the scrape interval can
fall between two grid points; those fall back to the per-job query and stay exact. This
reverses one of three batching experiments recorded in `compute_dcgm` — that one
measured a single unbucketed query for the whole selection against *wall clock* on
25-and 120-job selections, and it lost. Bucketed, and measured against server load at
8 587 jobs, it wins. Both results are now recorded there.

**The broad-selection note now states the cost instead of the count.** It used to say a
selection was large and would be slow. It now projects the query count and the time at
the configured rate, and names the two levers:

```text
note: 8889 jobs, 8889 with GPUs -- about 17802 Prometheus queries, ~5.9 min at the
      configured 50 queries/s.
      --no-dcgm skips Prometheus entirely (jobstats columns only); [prometheus]
      max_queries_per_second trades wall clock for load on the server.
```

Projected from the first day-slice, not counted — counting means listing every row
first, which is the expensive thing. It runs high, which is the right direction for a
warning.

### Breaking: `--classify --csv` no longer emits `wasteful-cpu-gpu` or `wasteful-gpu`

**Who this affects:** anything that reads the `LABEL` column of
`jobscope … --ts --classify --csv` and matches on those two strings — a dashboard
query, an alerting rule, a `grep`, a `CASE` in SQL. If you only read the tables, or
only ever matched `wasteful`, nothing changes for you.

**What changed.** Classification used to have three mechanisms: best-of-N voting over
the percentage metrics, a `POWER_W` cap that could only *lower* a verdict, and a
`CPU%` *split* that divided the worst band in two — `wasteful-cpu-gpu` for a job idle
on both, `wasteful-gpu` for a job holding idle GPUs while its host was busy. The cap
and the split were hardcoded to those two metrics and could not be reached from
config.

There are now two ideas, both configurable under `[classify]`: a **vote** may only
raise a verdict, a **floor** may only lower one. `CPU%` became an ordinary voter
carrying a ceiling — it may vote a job no higher than `inefficient`, because a busy
host is not evidence that the GPU allocation was justified. That reproduces what the
split was for without a label of its own, so the two split names are gone.

**The `LABEL` column now takes exactly these values:**

```
wasteful   inefficient   needs improvement   average   good   no-data
```

**How the two removed values map.** Verified against the classifier, not inferred:

| was | now | why |
|---|---|---|
| `wasteful-cpu-gpu` | `wasteful` | `CPU%` below its 5% edge votes `wasteful` too, so the verdict is unchanged in substance |
| `wasteful-gpu` | `inefficient` | `CPU%` above 5% votes for its own band, capped at `inefficient` by its ceiling |

`wasteful-gpu` becomes `inefficient` for *every* CPU% above the 5% edge — 6%, 36% and
99% all land there. A job cannot fall out of the set: nothing that was labelled either
name now reads `average` or `good`.

**If you match on these strings**, the mechanical fix is:

```
wasteful-cpu-gpu  ->  wasteful
wasteful-gpu      ->  inefficient
```

Matching `^wasteful` used to catch both flavours and now catches only the first; if
you meant "GPUs were idle", match `wasteful|inefficient`.

**Why this was not made backward compatible.** Keeping the old names would have meant
keeping the split, which is the mechanism being removed — and the whole point is that
a site can now express its own rule in `[classify]` rather than inheriting two
hardcoded flavours of one band. Measured on 95 real GPU jobs before the change, 13%
depended on the cap or the split for their verdict; all of them still classify as a
problem, none leaked to `good`.

**Also worth knowing:** `no-data` is a value `LABEL` can now carry, for a unit whose
voting metrics could not be read at all (exporter down, or the job predates Prometheus
retention). It is deliberately not a tier — it has no band and no colour. A row
carrying it has empty metric cells rather than zeros, so a monitoring outage no longer
reads as a fleet of wasteful jobs. Anything that treats an unrecognised `LABEL` as an
error should be taught this one.

### `doctor` is now `probe`, and `probe --init` sets up a site

`doctor` said nothing about what it examines. It probes the telemetry sources — Slurm,
the jobstats blob, the Prometheus exporters — and reports what each can answer.
`jobscope doctor` is an error naming the new spelling.

Everything it discovered used to be printed and thrown away; a person porting jobscope
read the output and hand-wrote a config repeating it. `probe --init` writes that config
instead:

```
$ jobscope probe            # what does this cluster expose
$ jobscope probe --init     # write a config from what it just found
```

It writes only if nothing is there — an existing config turns it into stdout plus a
note, since a config is hand-tuned within a week and that tuning has no other copy.
`--full` appends every remaining knob, commented.

Detected: `sampling_period` (from raw sample spacing), the `[site]` join labels,
`[metrics]` narrowed to series this server carries, and per-model power floors.
Thresholds are deliberately absent — a band edge is policy, not a property of the
cluster.

The power floors matter most: the flat 100 W default is wrong for most hardware, and
about half of any fleet's models measure cleanly at once, so `--init` emits floors for
those and comments the rest with the reason. A wrong floor silently caps healthy jobs
at `inefficient`.

### `jobscope config` shows the endpoint

It printed every band table but never the Prometheus URL — the one setting that has to
be right first — nor which of the three sources supplied it. Redacted, because the URL
commonly embeds a credential.

### Breaking: the old subcommands and eight duplicate flag spellings are gone

`summary`, `detail`, `dcgm` and `live` were rewritten into flags with a deprecation
note. Naming one is now an error — one that says what to type instead, because the
alternative is the word falling through as a job ID and Slurm answering
`sacct: fatal: Bad job/step specified: dcgm`.

| was | now |
|---|---|
| `jobscope summary -D 3` | `jobscope finished -D 3` |
| `jobscope detail JOBID` | `jobscope JOBID --per-gpu` |
| `jobscope dcgm --ts JOBID` | `jobscope JOBID --ts` |
| `jobscope live -a` | `jobscope -a` |
| `--hwdetail` | `--per-gpu` |
| `--min-runtime` | `--min-elapsed` |
| `--timeseries` | `--ts` |
| `--extended` | `--ext` |
| `--stats_per_node`, `--stats_per_job`, `--all_categories` | the `-` spellings |
| `--plot_avgeff` | nothing — it only printed "that is the default now" |

`--plot_ts` and `--node` keep both spellings: those are what the docs and most
command lines actually use. `contrib/jobscope_live.py` now rewrites `--min-runtime`
rather than relying on the alias, so the old wrapper is unaffected.

### Fixed: `--ext` meant two different things

On a report `--ext` is `--dcgm`. Under `describe` it was a *separate* flag that did
nothing on its own — `jobscope describe --ext` printed the column list, and you
needed `describe --dcgm --ext` for the full catalog. `--ext` now implies `--dcgm`
there, so the word means "the full DCGM catalog" everywhere.

### `--nodename` and `--gpuid` now work on the summary too

They used to be rejected there — the per-job table has no row per unit to filter.
It has numbers to *narrow*, though: the stored blob is per node and per GPU already,
so restricting it before anything reads it makes the row, the per-metric table, the
bars and the verdict all describe the subset.

```
$ jobscope -j 36770231                              GPU%=28   #GPU=8  NODE=2
$ jobscope -j 36770231 --nodename holygpu8a15401    GPU%=43   #GPU=4  NODE=1
```

The DCGM columns are narrowed to the same cards, or the row would mix one node's
GPU% with every node's SM_ACT%. A narrowed summary looks exactly like a whole-job
one, so the header now names the filter (`Node: … only`).

GPU ids are per node: nodes number their cards from 0, so `--gpuid 0` on a two-node
job keeps two cards, not one.

### New: `--gpuid`, and `--gpu 0,1` is no longer silently wrong

`jobscope plot` has always taken `--gpu 0,1` to chart particular cards, so that is
what people type on a report command too. There `--gpu` is the flag that selects the
GPU *columns* and takes no value — so `0,1` fell through to the JOBID positional. The
run warned about a job named `0,1`, charted **every** GPU, and exited 0.

Two changes. `--gpuid` narrows the `--ts` family (`--ts`, `--plot_ts`, `--stats`,
`--classify`) to the named cards:

```
jobscope -j JOBID --nodename NODE --plot_ts --gpuid 0,1
jobscope -j JOBID --ts 30m --csv --gpuid 2
```

It filters before the queries, as `--nodename` does, and names every id that matched
nothing rather than silently charting a shorter list. MIG instances are addressed as
they print (`0.1`).

And a comma in a job ID is now an error — job IDs never contain one — pointing at
`--gpuid` when `--gpu` was given.

### The summary block's `IDLE` column is now `USED`

One number was being given three readings in a single screen. For a metric at 27%
utilization, the pooled `Used/GPU-hr:` row said `27`, the efficiency bar below drew
`27%`, and the table cell between them said `719.3h (73%)` — while taking its
*colour* from the 27. Same figure, two complements and one silent inversion.

The cell reports `USED` now: resource-time that did work, and its share of the
allocation. Everything in the block says the same thing, and the section is titled
"Average efficiency" either way. For `POWER_W` that means time spent *above* the
floor rather than below it.

`--csv` is unchanged — `Stat<METRIC>` rows still carry `allocated=`, `used=` and
`idle=`, so nothing scripted breaks. In the table, the allocation is `USED` over its
own percentage and the idle share is the remainder.

### `config.example.toml` reordered, and it no longer regrades on copy

The template is required-first now — Prometheus, then which jobs, then which metrics,
then how they read — with a divider below which everything is tuning. The prose that
was longer than the setting it explained moved to [`docs/admin.md`](docs/admin.md).
351 lines to 293, and 56 live settings to 31.

**One behaviour change if you copied the old template.** It shipped three live values
that were *opinions, not jobscope's defaults*, so copying it silently changed grading:

| | old template | now |
|---|---|---|
| `[thresholds.summary.wasteful] sm_act` | 3 | 2 (the shared ladder) |
| `[thresholds.timeslice.wasteful] cpu` | 8 | 5 (the built-in calibration) |
| `[thresholds.timeslice.wasteful] sm_act` | 3 | 2 |

Both are still in the file as commented suggestions one keystroke away, with the
reasoning in `docs/admin.md`. The template now resolves identically to running with
no config at all, and a test enforces that. If you want the old numbers, uncomment
those blocks. A config file you wrote yourself is unaffected.

### New: `[thresholds] edges`, the one-line ladder

```toml
[thresholds]
edges = [2, 10, 20, 40]   # wasteful, inefficient, needs improvement, average
```

Seeds both the summary and timeslice views, so the eight `[thresholds.<view>.<edge>]`
tables collapse to one line for a site that grades a window the same way it grades a
whole job. The per-view and per-metric tables still override it: narrowest wins.

Note that retuning an edge switches **off** jobscope's built-in per-metric
calibration for that edge — `edges = [3, …]` moves CPU% to 3 along with everything
else, because "3 for everything I did not name" is an instruction and overriding it
silently would be undebuggable. Name `cpu` explicitly to keep a different value.

### Other changes since v0.1.1

Not exhaustive; the breaking change above is the only one that needs action.

- `jobscope probe` — what this cluster exposes and whether jobscope can read it,
  with `--metrics` to list the server's series against jobscope's names,
  `--toml` to emit those names as an editable config block, and `--validate` to
  cross-check Slurm accounting, the jobstats blob and Prometheus on one job.
- `live` is now `running` throughout. `jobscope live` still works as a deprecated
  alias and still prints a note; it is out of `--help`.
- Absent data never reads as zero. A metric that is *not applicable* (a CPU-only job
  has no GPU%) and one that is merely *unknown* (nothing could be read) are now
  distinct, and only the first is excluded silently.
- New config sections: `[site]` (label conventions, so a port needs no patch),
  `[classify]` (vote / floor / ceiling), `[metrics.<family>.<name>]` (define or
  rename a metric), `[report]` (which sections print, in what order) and `[plot]`
  (chart defaults).
- `--no-blob` reads finished jobs from Prometheus instead of the sacct blob, so the
  two sources can be compared on the same jobs.
- Metrics are fetched grouped by reducer rather than one query each — a 51-job
  `--dcgm` sweep went from 12.8 s to 3.4 s.
- The `diag` subcommand is gone; jobs are classified instead.

## v0.1.1

Initial tagged releases; see the git history.
