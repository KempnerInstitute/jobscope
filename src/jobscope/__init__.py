"""jobscope: Slurm job efficiency and GPU utilization reporting.

Reads the utilization blob Slurm stores in each job's sacct AdminComment,
enriches it with DCGM GPU profiling metrics from Prometheus, and renders the
result as tables, CSV, or terminal charts.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jobscope")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
