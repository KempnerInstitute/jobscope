"""jobscope: Slurm job efficiency and GPU utilization reporting.

Reads the utilization summary Slurm stores in each job's sacct AdminComment,
enriches it with DCGM GPU profiling metrics from Prometheus, and renders the
result as tables, CSV, or terminal charts.
"""

__all__ = ["__version__"]


def __getattr__(name: str) -> str:
    """Resolve ``__version__`` on first access (PEP 562).

    ``importlib.metadata.version()`` costs 44ms, and it was paid by every invocation to
    serve a string only ``--version`` prints. Deferring it here rather than at each use
    site keeps ``from jobscope import __version__`` working unchanged. Note that such an
    import *triggers* this, so it must not sit at another module's top level or the cost
    comes straight back -- cli.py resolves it inside its --version action for that reason.
    """
    if name != "__version__":
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("jobscope")
    except PackageNotFoundError:
        return "0.0.0"
