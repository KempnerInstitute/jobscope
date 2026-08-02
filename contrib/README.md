# contrib

Site-specific and experimental tools that ship in the repository but are **not**
part of the installable `jobscope` package.

## jobscope_live.py — deprecated

A thin wrapper for `jobscope live`, kept so existing command lines keep working.

This began as a standalone script -- squeue plus Prometheus, no install needed,
running on the login node's system python -- for the question the historical views
could not answer: *what is running right now*. That logic now lives in the package
(`src/jobscope/live.py`), so this file only translates the flags whose names
changed and hands off:

| old | new | why |
|---|---|---|
| `--min-runtime 5m` | `--min-elapsed 5m` | same meaning, clearer name (the old spelling still works as an alias) |
| no `-u` meant every user | `-a`/`--all-users` | `jobscope live` defaults to your own jobs, like every other view (the wrapper passes `-a` for you) |

Prefer `jobscope live` directly. Unlike the original script it needs an installed
jobscope, because the package requires Python >= 3.9 where the script targeted
3.6; run the wrapper on an older interpreter and it prints install guidance rather
than a traceback.

```bash
jobscope live -p kempner -a       # what ./jobscope_live.py -p kempner used to print
jobscope live --describe          # what each column means
```

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

For end-user reporting, prefer `jobscope dcgm --ext`, which shows the full DCGM
metric catalog without needing the upstream jobstats package.
