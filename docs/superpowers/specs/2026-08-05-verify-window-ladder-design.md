# `--verify`: a per-job accuracy check across a window ladder

Status: approved, ready for an implementation plan
Date: 2026-08-05

## Why

The sweep is fast and sometimes wrong in the direction that gets a working job killed.
Measured on job 36664692, a 4-GPU H200 job, all three readings taken minutes apart:

```
instant (what the sweep shows)   GPU% 0     SM_ACT% 0.0    POWER_W 111
--avg (folded over the run)      GPU% 48    SM_ACT% 33.3   POWER_W 246
per-scrape series                mean 32-48% by window, max 100.0 in every
                                 window including the last 30 minutes
```

The instant reading is zero on two metrics — below the 2% wasteful edge, top of any
Problem-jobs list, an obvious kill. The job was computing in bursts and had saturated all
four cards within the previous half hour. `POWER_W 111` is near the H200 idle floor, so at
that instant the card genuinely was idle; it simply was not representative.

Across a partition, a single scrape sits more than 20 points from the job's own mean about
a quarter of the time (n=54, four partitions): `GPU%` median 5.0 / p90 36.0 / max 59.0;
`POWER_W` median 14 W / p90 133 W / max 192 W. Fine for triage. Not a basis for `scancel`.

`--verify` is the check to run on one job before acting on it.

## The finding that shapes the design

A single accurate number is still the wrong output. Same job, `GPU%` per GPU:

| window | n | mean across the 4 cards | max |
|---|---|---|---|
| whole run | 2442 | 49.2 / 48.1 / 47.7 / 48.6 | 100.0 |
| last 240m | 241 | 51.5 / 49.4 / 50.0 / 49.2 | 100.0 |
| last 120m | 121 | 45.1 / 44.4 / 44.1 / 46.0 | 100.0 |
| last 30m | 31 | 32.1 / 32.5 / 31.5 / 32.2 | 100.0 |

Reading down the ladder gives *bursty, duty declining, still working*. No single mean says
that at any accuracy. Two values carry the decision:

- **`max`** separates "dead" from "bursty". A view printing only means would have told you
  to kill this job.
- **the trend across rungs** separates "always been mediocre" from "was fine, stopped an
  hour ago" — the same lifetime mean covers both.

## What this is not

Not a recommendation engine. `review_reco` proposes text like *"This often means small
kernels, poor batching, or input starvation. Check batch size and data loading."* That is
confident about a cause the data cannot establish, and being wrong once in that register
is how a tool stops being trusted. `--verify` classifies the *shape* of the series, which
is derivable, and stops there.

## Naming

`probe --validate JOBID` already exists and means something adjacent: *do my three sources
agree about this job* — an instrumentation check for an admin. `--verify` means *how did
this job actually behave* — a decision check for whoever is about to cancel it. The two
names do not distinguish those, and `review_reco` proposed `explain` for this role.

Specced as `--verify` because that is the name in use in the discussion. If the collision
matters more than the continuity, `--explain` is the drop-in alternative and only the flag
string changes.

## CLI surface

```
--verify [WINDOW]     per-job accuracy check across a window ladder
```

Joins the mutually-exclusive `grain` group (`cli.py:208`), so it cannot combine with
`--ts`, `--per-gpu`, `--per-node`, or the plot flags.

`WINDOW` is parsed by the existing `_ts_window` (`cli.py:738`) — same `parse_duration`
call, same error text naming the flag. It **bounds the fetch**, for a job long enough that
the whole series is more than wanted; the ladder is then built inside it.

Works on running and finished jobs alike — `timeseries.finished_gpu` and the running
collectors both yield the same `JobSeries`, and neither the ladder nor the shape rules care
which. For a running job the widest rung ends at "now"; for a finished one, at its end.

`--verify` requires named jobs. A window or partition selection is an error:

```
--verify checks one job at a time; name the jobs with -j.
It fetches every scrape of each job's series, which a partition sweep should not do.
```

`--nodename` and `--gpuid` narrow it as they do elsewhere, dropping UUIDs before the
queries rather than filtering rows after.

## The window ladder

Default rungs: **whole run, last 2h, last 30m**. Configurable:

```toml
[report]
verify_windows = ["2h", "30m"]     # the whole run is always the first rung
```

A rung whose span is greater than or equal to the run's own length is **dropped**, not
printed as a duplicate of the whole run — otherwise every job shorter than two hours gets
three identical columns. A 20-minute job therefore shows one rung; a 40-hour job shows
three.

At 60 s scrapes the 30 m rung is 31 samples. Thin, and adequate for "is it moving" —
which is why `max` and the idle-stretch figure carry the decision rather than the mean
alone.

## Output

Extends the `--ts --stats` row identity (`report.timeseries_stats`) rather than inventing a
layout: same `NODE:GPU` / `METRIC` lead columns, different tail.

```
NODE:GPU        METRIC   N     MIN   MAX    run   2h    30m   BELOW  IDLEMAX  SHAPE
holygpu8a…:0    GPU%     2442  0.0   100.0  49.2  45.1  32.1   38%    4m      bursty
holygpu8a…:1    GPU%     2442  0.0   100.0  48.1  44.4  32.5   39%    4m      bursty
```

The rung columns are named by the configured rungs, so a site that changes
`verify_windows` changes these headers; `run` is always the first. Followed by:

- **`BELOW`** — the fraction of *measured* samples under the metric's cutoff, over the
  **widest rung fetched** (the whole run, or the `WINDOW` bound when one was given).
  `review_reco`'s "fraction of runtime below threshold".
- **`IDLEMAX`** — the longest unbroken stretch under that cutoff, as a duration.
  `review_reco`'s "longest continuous idle interval", and the most decision-relevant single
  number here: 4 minutes is bursty, 3 hours is wedged, and the same mean covers both.
- **`SHAPE`** — the classification below.

A provenance block follows the table, naming which source served each column via
`RESOLVED.source_of(header)`. `--verify` is the pre-action check; where the number came
from is part of the answer.

### Cutoff, and why gaps are not idle

"Under the cutoff" means below `thresholds.edge("wasteful", header)` for a percentage
metric, and below `thresholds.floor_of("POWER_W", model)` for `POWER_W` — its per-model
floor, since watts have no tier.

**A missing scrape is not an idle sample.** Absent stamps break an idle run rather than
extending it, and `BELOW` is a fraction of measured samples, not of elapsed time. Counting
gaps as idle would let an exporter outage read as a three-hour wedge — the same collapse
between "not measured" and "not working" that `models.Measure` exists to prevent. The
count of gaps is printed under the table so a thin series is visible rather than silently
averaged.

### Shape classification

Per (unit, metric), in terms of the existing tier machinery so a site's own
`[thresholds]` govern it:

Per (unit, metric). Tier names are `config.TIERS`' — worst to best: `wasteful`,
`inefficient`, `needs improvement`, `average`, `good`.

| shape | rule | reading |
|---|---|---|
| `no-data` | no measured samples for this unit and metric | not a shape; reuses `job_eff.NO_DATA` rather than inventing a sixth word for it |
| `flat-idle` | `tier(max) == "wasteful"` | nothing ever happened — **the only kill-candidate shape** |
| `bursty` | `tier(max) == "good"` and `tier(mean_widest)` in `{wasteful, inefficient}` | works hard in bursts |
| `declining` | two or more rungs, each rung's tier no better than the one before, and the narrowest rung's tier strictly worse than the widest's | was fine, is not now |
| `steady` | anything else | as advertised |

`declining` needs at least two rungs, so a job shorter than the narrowest configured rung
can never be `declining` — it has only the whole-run rung.

`declining` and `bursty` can both hold; `declining` is reported, being the actionable one.
Only `flat-idle` says "nothing ran" — and it is the one shape the instant reading on job
36664692 would have wrongly produced.

## Implementation

**No new collection.** `timeseries.finished_gpu` / `finished_host` / `running_host` already
yield `JobSeries` carrying every per-scrape sample, and `report.timeseries_stats` already
reduces one to min/mean/max/last — *"a window costs nothing extra to summarize"*
(`report.py:2307`). `--verify` is a new reducer plus a renderer over series that already
arrive.

- Fetch **once**, whole run (or the `WINDOW` bound), then slice per rung client-side.
  `running.range_window` guarantees this is sound: *"Aligned, `--ts 1h` returns exactly the
  rows a full `--ts` would have, so fetching a window and slicing a whole series agree."*
  So every rung is free.
- Pool with the existing `report.pool_samples`, which pools samples rather than per-GPU
  means — a card the exporter missed for half the window carries half the weight instead
  of counting as a full peer.
- `BELOW`, `IDLEMAX` and `SHAPE` are new functions over one unit's stamp-ordered samples.
  They belong beside `job_eff`'s policy code, not in the renderer: they read thresholds and
  return a figure, and `tests/test_layering.py` keeps rendering free of policy.

**Thresholds.** Graded against `cfg.timeslice_thresholds`, not `cfg.thresholds`, matching
every other windowed view (`cli.py:922`) — *"a two-hour window that catches a checkpoint
pause should not answer to a nineteen-hour job's bar"*.

**Cost.** One range query per (job, metric) over the fetch window; for job 36664692 that is
2442 stamps × 4 cards × 5 metrics, which is one `--ts` run. `review_reco`'s grouped range
queries would cut the per-metric fan-out and apply directly here.

## Testing

- The 36664692 case as a fixture: a series whose `max` is high, whose mean is mid, and
  whose last rung is low, asserted to classify `bursty` and **not** `flat-idle`. This is
  the regression that protects against a means-only view.
- A genuinely dead series (`max` under the wasteful edge) classifies `flat-idle`.
- A series busy then flat classifies `declining`, and a flat-busy one `steady`.
- A rung equal to or longer than the run is dropped, not duplicated: a 20-minute job shows
  one rung.
- Slicing a whole-run series for a rung equals a narrowed fetch for the same span — the
  `range_window` alignment guarantee, asserted rather than assumed.
- **Gaps are not idle**: a series with a hole in the middle does not report an `IDLEMAX`
  spanning it, and `BELOW` divides by measured samples. Fails if absent stamps are counted.
- `POWER_W` uses its per-model floor, not a percentage edge, and a model with a higher idle
  draw (RTX PRO 6000 at 165 W vs V100 at 27 W) changes the answer.
- `--verify` on a window selection raises with the message above.
- Grading uses `timeslice_thresholds`: a config where the two tables differ produces the
  timeslice answer.
- Provenance names the resolved source per column, and follows `--gpu-source`.

## Verification

1. `.venv/bin/python -m pytest tests/ -q`.
2. Confirm the editable install is live — `jobscope.cli.__file__` under `src/`.
3. `jobscope -j 36664692 --verify` reproduces the ladder in this document and classifies
   `bursty`, against `--ts --stats` at each rung by hand.
4. A job that is genuinely idle, confirming `flat-idle` and an `IDLEMAX` near its runtime.
5. `jobscope -j <JOBID> --verify 120m` bounds the fetch and drops the rungs that no longer
   fit.
6. `--gpu-source dcgm` and `nvml`, confirming the provenance block tracks the resolution.
