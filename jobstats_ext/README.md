# jobstats_extended

A local, non-invasive superset of the stock `jobstats` for completed Slurm
jobs. It subclasses `jobstats.Jobstats` (reusing all of its
sacct/Prometheus/serialization logic) and folds extra DCGM GPU profiling
metrics into the jobstats blob, keyed per-GPU by `minor_number` exactly like the
built-in `gpu_utilization`.

Extra metrics added (time-averaged over the job window):

- SM active (%)
- SM occupancy (%)
- Tensor pipe active (%)
- DRAM active (%)
- Power usage (W)

Because the `DCGM_FI_*` series live in a different exporter and carry
`UUID`/`gpu` rather than `minor_number`, the script maps `UUID -> (node, minor)`
via the `nvidia_gpu_jobId` companion series, then stores each DCGM metric back
under `minor_number` so the new keys align with `gpu_utilization` in the blob.

## Usage

```bash
./jobstats_extended.py <job_id> [<job_id> ...] [options]
```

## Examples

```bash
./jobstats_extended.py 18583067            # formatted report (like jobstats) + DCGM
./jobstats_extended.py --json 18583067     # extended blob as pretty JSON
./jobstats_extended.py -b 18583067         # JS1:<base64> blob (storage form)
./jobstats_extended.py --no-dcgm 18583067  # report/blob without the DCGM metrics
```

## Options

- `-j`, `--json` — emit the extended blob as pretty JSON instead of the report
- `-b`, `--base64` — emit the `JS1:<base64 gzip JSON>` blob (AdminComment storage form)
- `--no-dcgm` — skip the DCGM augmentation
- `-f`, `--force` — force recompute from Prometheus instead of the cached blob
- `-c`, `--cluster` — cluster name (default: from sacct)
- `-n`, `--no-color` — render the report without colorization
- `-d`, `--debug` — debug output

## Requirements

Runs alongside an existing `jobstats` install: it imports `jobstats`,
`output_formatters`, and `config` (`PROM_SERVER`) from the system path
(`/usr/local/bin`, `/usr/bin`), plus the `requests` library.

