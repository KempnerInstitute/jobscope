"""Tests for the advisory GPU diagnosis tags."""

from jobscope.diagnose import diagnose, diagnose_dcgm


def test_short_when_under_min_runtime():
    assert diagnose(90, 90, 90, 90, duration=60, min_runtime=180) == "short"


def test_no_data():
    assert diagnose(None, None, None, None, duration=None, min_runtime=None) == "-"


def test_idle_and_underfed():
    assert diagnose(2, 0, 0, 0, None, None) == "idle"
    assert diagnose(10, 0, 0, 0, None, None) == "underfed"


def test_ok_when_busy_and_healthy():
    assert diagnose(80, 90, 90, 10, None, None) == "ok"


def test_low_occupancy():
    assert diagnose(80, 10, 90, 10, None, None) == "low-occ"


def test_mem_bound():
    assert diagnose(50, 90, 90, 90, None, None) == "mem-bound"


def test_no_tensor():
    assert diagnose(80, 90, 0, 10, None, None) == "no-tensor"


def test_multiple_tags_join():
    tags = diagnose(50, 10, 0, 90, None, None).split(",")
    assert set(tags) == {"low-occ", "mem-bound", "no-tensor"}


def test_diagnose_dcgm_reads_headers():
    values = {"SM_ACT%": 80, "OCC%": 90, "TENSOR%": 90, "DRAM%": 10}
    assert diagnose_dcgm(values, duration=1000, min_runtime=180) == "ok"
    assert diagnose_dcgm({}, duration=1000, min_runtime=180) == "-"
