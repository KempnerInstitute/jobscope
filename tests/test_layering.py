"""The import graph, asserted.

Module boundaries decay silently: one convenient import and a collector is
formatting output, or a renderer is issuing queries. Nothing fails, nothing is
slower, and the next person to port jobscope to another cluster has to read the
whole package to find out where the queries live. So the layering is a test.

Layers, bottom up:

    errors, models                 leaves; import nothing from the package
    config, metrics, classifier,   policy -- read settings, return decisions
      source
    prometheus                     the client
    summary, cpu, nvml, dcgm,         collectors -- turn a source into numbers
      slurm, running, timeseries,
      extra_metric, job_ave_stats
    report, plot                   render -- turn numbers into text
    select, cli, probe             orchestration -- wire the above together

Only two rules are enforced, because only two of them have ever been broken:
a collector must not reach up into rendering, and rendering must not reach down
into the client. The rest of the ordering is documentation.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "jobscope"

LEAVES = {"errors", "models"}
POLICY = {"config", "metrics", "classifier", "source"}
COLLECTORS = {"jobstats", "cpu", "nvml", "dcgm", "slurm", "running", "timeseries",
              "extra_metric", "job_ave_stats"}
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


@pytest.mark.parametrize("stem", sorted(LEAVES))
def test_leaves_import_nothing_from_the_package(stem):
    assert not imports_of(stem)


def test_the_classifier_renders_nothing():
    """The verdict is policy. It must stay callable without a terminal in sight."""
    assert not imports_of("classifier") & (RENDER | {"prometheus"})
