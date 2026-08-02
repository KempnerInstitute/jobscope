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
