"""The import graph, asserted.

Module boundaries decay silently: one convenient import and a collector is
formatting output, or a renderer is issuing queries. Nothing fails, nothing is
slower, and the next person to port jobscope to another cluster has to read the
whole package to find out where the queries live. So the layering is a test.

Layers, bottom up:

    errors, models                 leaves; import nothing from the package
    config, metrics, job_eff,      policy -- read settings, return decisions
      source
    prometheus                     the client
    summary, cpu, nvml, dcgm,         collectors -- turn a source into numbers
      slurm, running, timeseries,
      extra_metric, job_ave_stats,
      rows
    report, plot                   render -- turn numbers into text
    select, cli, probe             orchestration -- wire the above together

``rows`` is the newest and sits deliberately at the top of the collectors: it is what
turns a scheduler record and a batch of exporter readings into the ``models.JobRow`` a
renderer places. Rendering used to do that itself.

Three rules are enforced, because only these three have ever been broken: a collector
must not reach up into rendering, rendering must not reach down into the client, and
rendering must not reach down into the scheduler's model or the storage format. The
rest of the ordering is documentation.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "jobscope"

LEAVES = {"errors", "models"}
POLICY = {"config", "metrics", "job_eff", "source"}
COLLECTORS = {"jobstats", "cpu", "nvml", "dcgm", "slurm", "running", "timeseries",
              "extra_metric", "job_ave_stats", "rows"}
RENDER = {"report", "plot"}

# Empty, and the test below is what keeps it that way. report.py used to query inside
# the render pass -- five time-series functions took a PrometheusClient -- so this held
# {"report"} while that was true. jobscope.timeseries now issues those queries and the
# renderers take what it yields.
COLLECTS_WHILE_RENDERING: set = set()


def imports_of(stem: str) -> set:
    """The sibling modules ``stem`` imports, by name."""
    found = set()
    for node in ast.walk(ast.parse((SRC / (stem + ".py")).read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module:
                found.add(node.module)
            else:  # from . import config
                found |= {a.name for a in node.names}
    return found


def module_stems():
    return sorted(p.stem for p in SRC.glob("*.py") if not p.stem.startswith("__"))


def test_every_module_is_placed_in_a_layer():
    """A new module has to be classified here, or the rules below skip it silently."""
    placed = (LEAVES | POLICY | COLLECTORS | RENDER
              | {"prometheus", "select", "cli", "probe"})
    assert set(module_stems()) == placed


@pytest.mark.parametrize("stem", sorted(COLLECTORS))
def test_no_collector_imports_the_render_layer(stem):
    """A collector that formats is a collector that cannot be reused.

    This is the rule live_blob.py broke: it existed to build a jobstats-shaped dict
    purely so the summary renderers would accept it, which is why the shape of one
    site's accounting blob became the whole package's data model.
    """
    assert not imports_of(stem) & RENDER


@pytest.mark.parametrize("stem", sorted(RENDER))
def test_the_render_layer_does_not_query_prometheus(stem):
    """Rendering takes numbers, not a client -- so a report can be tested offline.

    It is also what lets --ts stream: the collectors are generators, and a renderer
    that held the client would have had to choose between querying per job inside its
    own loop (what it used to do) and buffering the whole sweep.
    """
    assert "prometheus" not in imports_of(stem)


def test_no_render_module_collects():
    """The exemption this test guards is empty; it must stay that way."""
    offenders = {s for s in RENDER if "prometheus" in imports_of(s)}
    assert offenders == COLLECTS_WHILE_RENDERING == set()


# What a renderer must not name: how a job is *stored* and how the scheduler *models*
# it. Two formatters are allowed through -- jobstats.bytes_to_gb and the per-unit
# header list -- because they are presentation, and the precision they encode belongs
# beside the shape that defines it.
STORAGE_NAMES = {"JobRecord", "Selection", "jobstats_metrics", "jobstats_detail",
                 "jobstats_per_node", "jobstats_capacity", "decode_admin_comment",
                 "narrow_stats", "gpu_ids_in", "gpu_count", "nodes_in"}


@pytest.mark.parametrize("stem", sorted(RENDER))
def test_the_render_layer_does_not_name_the_scheduler_or_the_storage(stem):
    """A renderer takes rows, not records.

    report.py used to call jobstats_metrics() on the gzipped base64 blob sacct stores
    in AdminComment, mid-render, once per job -- so a test of a column or a footer had
    to build one, and "how a job is stored" could not change without touching the
    renderers. jobscope.rows does that now and the renderers place cells.

    Names rather than modules, because the two formatters report.py still imports from
    jobstats are fine: it is the *model* and the *decoding* that must not be here.
    """
    named = set()
    for node in ast.walk(ast.parse((SRC / (stem + ".py")).read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            named |= {a.name for a in node.names}
    assert not named & STORAGE_NAMES, (
        "%s.py names %s -- take a models.JobRow instead"
        % (stem, ", ".join(sorted(named & STORAGE_NAMES))))


@pytest.mark.parametrize("stem", sorted(LEAVES))
def test_leaves_import_nothing_from_the_package(stem):
    assert not imports_of(stem)


def test_job_eff_renders_nothing():
    """The verdict is policy. It must stay callable without a terminal in sight."""
    assert not imports_of("job_eff") & (RENDER | {"prometheus"})
