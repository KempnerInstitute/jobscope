# Per-unit granularity and elapsed time in the detail tables

Status: approved, ready for an implementation plan
Date: 2026-08-05

## Why

A multi-node job has no node-level table. `--per-gpu` on a 16-node job with 8 cards each
prints 128 rows; the question being asked of it — *which node is the slow one* — wants 16.
The codebase already says so, in `_unit_charts`' docstring: *"Grouping by whatever
distinguishes them -- the node normally -- is what turns sixteen rows of twelve columns
into 'this node is the slow one'."* The efficiency **bars** already group by node. The
table cannot.

Two smaller gaps ride along:

- The detail tables carry no elapsed time. The per-job summary has `RUNTIME`; `--per-gpu`
  does not, and its CSV carries `JOBID` and nothing else per-job, so a consumer cannot
  recover it.
- The granularity axis is invisible. `--per-gpu` is a lone boolean; nothing indicates a
  per-job default exists or that these are points on one scale.

Out of scope, specified separately: a high-accuracy per-job pre-action view (`--verify`).
`--avg` is unchanged — the fast instant reading stays the default for sweeps.

## Decisions taken, with the reasoning

**Three boolean flags, not `--per LEVEL`.** `--stats` takes `{gpu,node,job}` as a value,
and the comment above it records that it was once three flags: *"These were
--stats/--stats-per-node/--stats-per-job, three spellings setting this same dest to three
constants."* That precedent argues for consolidating. It is outweighed by
`docs/admin.md:514,525`, which is a **migration table**: `jobscope detail JOBID` →
`--hwdetail` → `--per-gpu`. This flag is already the destination of two renames. A third
spelling would invalidate a table written for people who already moved once.

**Separate column sets per level, not one shared set.** Unifying would change the per-job
table's columns and its CSV, breaking anything parsing it, for a cosmetic gain. The
profiling block is shared; the identity prefix differs.

**Elapsed as a column, not in the per-job header line.** The header line
(`Job 36978909 [RUNNING] name`) is not emitted in CSV mode, and a CSV consumer is the
caller that cannot otherwise get elapsed. It repeats per row, which is accepted: `CPU%`
already does.

## CLI surface

The existing mutually-exclusive `grain` group (`cli.py:208`) gains two members:

```
--per-job     one row per job    (today's default, now nameable)
--per-node    one row per node   (new)
--per-gpu     one row per GPU    (unchanged)
```

`--per-job` selects the default and is otherwise a no-op. It exists so the axis is
discoverable and so scripts can be explicit.

`cli.py` resolves these to a level string `"job" | "node" | "gpu"`. `"job"` keeps
`SummaryRenderer`; `"node"` and `"gpu"` both use `DetailRenderer`, which takes the level.

Flag-narrowing (`cli.py:452-458`) treats `--per-node` exactly as `--per-gpu`: it hides
`--ts` and `--plot_ts`.

## The per-node row

```
NODE  #GPU  CPU%  CPU-MEM  GPU%  GPU-MEM  GMEM%  SM_ACT%  TENSOR%  DRAM%  POWER_W  RUNTIME
```

That is the **display order**. Identical in shape to the per-GPU row; the second column
changes from the GPU minor number to that node's card count, because a pooled row has to
say how many cards it pooled.

Row positions are *not* display order — `Column.index` decouples them, see the Renderer
section. By row position:

| row cell | per-gpu | per-node |
|---|---|---|
| 0 | node name | node name |
| 1 | `GPU` — minor number | `#GPU` — count of cards on this node, `0` if none |
| 2–3 | `CPU%`, `CPU-MEM` | identical — already per-node |
| 4–6 | `GPU%`, `GPU-MEM`, `GMEM%` for one card | pooled over the node's cards |
| 7 | `RUNTIME` — displayed last, not eighth | same |
| 8+ | profiling block, one card | profiling block, pooled |

`CPU%` and `CPU-MEM` need no aggregation: `jobstats_detail` already computes them per
node, which is exactly why they repeat on every GPU row today.

### Aggregation

Pooling uses each metric's declared `spec.agg` — `mean` for utilization, `max` for
`PWRmax_W` and the memory pair, `sum` for `ENERGY_kWh`. This is the same rule
`dcgm_for_job` already applies to produce the job-level figure (`dcgm.py:823-832`); a node
figure is that reduction over a subset of UUIDs.

Two invariants carried over, both load-bearing:

- **Reduce across UUIDs, never across `(node, minor)` keys.** MIG siblings share a minor,
  so `per_gpu` has already collapsed them — the last UUID written wins (`dcgm.py:818-822`).
  Pooling that dict would silently drop instances. `dcgm.py:825-827` states this for the
  job level and it applies identically here.
- **Derive `GMEM%` after pooling**, as summed-used over summed-total. A mean of per-card
  ratios is a different number; `running.py:573` — *"a ratio of means is not the mean of
  ratios."* `_add_derived` already does this correctly and is reused unchanged.

For a **finished** job the stored summary is already per-node — `stats["nodes"][<node>]`
carries `gpu_utilization`, `gpu_used_memory`, `gpu_total_memory` keyed by minor, plus
`cpus`/`total_time`/`used_memory`/`total_memory`. A new `jobstats.jobstats_per_node(stats)`
returns one row per node in the same 7-cell prefix shape `jobstats_detail` returns — cells
0–6, without `RUNTIME`, which the renderer appends. `GPU%` is the mean of that node's
utilizations, `GPU-MEM` summed used over summed total, `GMEM%` the ratio of those sums. It
reuses `bytes_to_gb` so the cells are spelled identically to the per-GPU rows.

A node with no GPU entries yields `#GPU` of `0` and `-` for cells 4–6, mirroring
`jobstats_detail`'s existing else-branch for a GPU-less node.

## Data flow

`dcgm_for_job` has `per_uuid` in hand, which is the only place the UUID-level values exist.
Per-node pooling therefore happens there, not in the renderer.

- Extract the reduction at `dcgm.py:823-832` into a helper taking a UUID subset and
  returning `{header: value}`. The job level calls it with every UUID, so existing
  behaviour runs the same code.
- `dcgm_for_job` and `compute_dcgm` return a third mapping, `per_node`, keyed by node name.
- The pair becomes a `NamedTuple` — `JobGpuData(overall, per_gpu, per_node)` — not a bare
  3-tuple. It stays index-compatible with the existing `[0]`/`[1]` reads, and the field
  names remove the positional guessing those reads currently require. `select.DcgmData`
  becomes `Dict[str, JobGpuData]`.

Every existing access site **indexes** (`[0]`, `[1]`) with a `({}, {})` default and none
unpacks the pair, so this is mechanical. The sites to update are the defaults and
annotations at `report.py:1283`, `1306`, `1340`, `1368`, `1371`, `1823`, `1826`, `1835`,
and `select.py:69`, `234`, `332`.

For the **running** path, `running.per_gpu_by_node_minor` (`running.py:597`) gains a
sibling that pools by `gpu.host` at UUID level — the `Gpu` records already carry `host`.

## Renderer

`detail_columns()` and `detail_gpu_headers()` — the seam added when the column-shift
defects were fixed — take a `level`. The profiling block is shared; only the identity
prefix differs. The generated-index property that closed those defects is preserved:
nothing writes a row index by hand.

**`RUNTIME` sits at row index 7, before the profiling block, but is ordered last as a
column.** `Column.index` already decouples display order from row position, and using it
here means the runtime cell's index does not depend on how many profiling columns there
are. The alternative — appending it after the block — would make its index vary with
`show_dcgm`, which is the exact shape of the bug just fixed.

The runtime cell is appended in `DetailRenderer._rows_for`, which runs unconditionally,
rather than in `extend_detail_row`, which is only called when `show_dcgm` is true.
Otherwise `--cpu --per-gpu` would show a `RUNTIME` column with no cell behind it.
`RUNTIME` carries group `"id"`, so `cols_for` keeps it in every view — matching
`SUMMARY_COLUMNS`.

`_unit_charts` must not take its by-GPU branch at level `"node"`. That branch triggers on
`len(nodes) == 1` and labels each group `"GPU <minor>"`; at node level a single-node job
has one row and would be labelled `GPU -`. At `"node"` it always groups by node.

`--nodename` filtering already works at both levels — it reads `_NODE_INDEX`, which stays
cell 0.

## Testing

- Per-node figures against hand-computed values from a two-node fixture, where the two
  nodes differ, so a swap or a job-level fallback fails.
- A MIG fixture — two UUIDs sharing one minor — proving pooling happens at UUID level.
  Fails if the implementation reduces `per_gpu`.
- `GMEM%` as ratio-of-sums, with a fixture where ratio-of-sums and mean-of-ratios differ
  (e.g. 10/100 and 70/80 on one node).
- A single-node job's per-node row equals its per-job GPU figures — the two reductions
  agree when the subset is everything.
- A CPU-only job: one row per node, GPU cells `-`, no crash on the absent `#GPU`.
- `RUNTIME` present in table and CSV at both detail levels, and under `--cpu` where the
  profiling block is absent.
- The three-preference alignment tests (`jobstats`/`dcgm`/`nvml`) extended to level
  `"node"`, including the CSV-by-name assertion.
- Default `--per-gpu` output byte-identical to current except for the added `RUNTIME`
  column; per-job output completely unchanged.

## Verification

1. `.venv/bin/python -m pytest tests/ -q`.
2. Confirm the editable install is live — `python -c "import jobscope.cli; print(jobscope.cli.__file__)"` should point into `src/`.
3. A real multi-node GPU job at all three levels, checking the per-node row against the
   per-GPU rows above it by hand:
   ```
   jobscope -j <JOBID> --per-job
   jobscope -j <JOBID> --per-node
   jobscope -j <JOBID> --per-gpu
   ```
4. The same three with `--csv`, confirming header and rows agree in width and `RUNTIME`
   is present.
5. `--gpu-source dcgm` at level `node`, confirming the alignment property still holds.
6. A MIG job at `--per-node`, confirming the pooled `GPU%` reflects every instance rather
   than one sibling.
