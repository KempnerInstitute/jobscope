# Changelog

## Unreleased

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
was longer than the setting it explained moved to [`docs/config.md`](docs/config.md).
351 lines to 293, and 56 live settings to 31.

**One behaviour change if you copied the old template.** It shipped three live values
that were *opinions, not jobscope's defaults*, so copying it silently changed grading:

| | old template | now |
|---|---|---|
| `[thresholds.summary.wasteful] sm_act` | 3 | 2 (the shared ladder) |
| `[thresholds.timeslice.wasteful] cpu` | 8 | 5 (the built-in calibration) |
| `[thresholds.timeslice.wasteful] sm_act` | 3 | 2 |

Both are still in the file as commented suggestions one keystroke away, with the
reasoning in `docs/config.md`. The template now resolves identically to running with
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

- `jobscope doctor` — what this cluster exposes and whether jobscope can read it,
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
