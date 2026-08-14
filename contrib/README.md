# contrib

Site-specific and experimental tools that ship in the repository but are **not**
part of the installable `jobscope` package.

## Moved out: idle_sweep

The idle-job sweep that used to live here is now its own repository,
[`jobsweep`](https://github.com/KempnerInstitute/jobsweep), which depends on jobscope
rather than shipping inside it. It cancels jobs and writes to their owners, which is an
operator tool with a different audience and a different blast radius from a reporting
library; keeping it here meant a copy of it existed wherever jobscope did.

It still imports jobscope rather than shelling out to it -- `report.classify_units` exists
for that caller, so it gets values instead of parsing a display format back, and one
process means one rate limiter across the sweep and every verify.

## jobstats_extended.py

A local, non-invasive superset of the stock `jobstats` command. Unlike jobscope
(which decodes the stats blob Slurm already stored and reads DCGM metrics
independently), `jobstats_extended.py` subclasses `jobstats.Jobstats` and folds
extra DCGM profiling metrics (SM active, SM occupancy, tensor pipe, DRAM active,
power) **into** the jobstats blob, keyed per-GPU by `minor_number`. It can emit
the augmented `JS1:<base64 gzip JSON>` blob in the same storage form jobstats
writes to the sacct AdminComment.

It exists to prototype what an extended jobstats deployment would look like before
an administrator swaps the augmented blob into a cluster-wide install. Because it
imports the upstream `jobstats`, `output_formatters`, and `config` modules from
the system path, it only runs where a Princeton-style jobstats install is present
-- which is why it is kept here rather than in the general package.

```bash
./jobstats_extended.py 18583067            # formatted report (like jobstats) + DCGM
./jobstats_extended.py --json 18583067     # extended blob as pretty JSON
./jobstats_extended.py -b 18583067         # JS1:<base64> blob (storage form)
./jobstats_extended.py --no-dcgm 18583067  # report/blob without the DCGM metrics
```

For end-user reporting, prefer `jobscope --all-metrics`, which shows the full DCGM
metric catalog without needing the upstream jobstats package.
