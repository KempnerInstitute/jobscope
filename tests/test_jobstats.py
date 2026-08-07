"""Tests for the sacct AdminComment blob decoder and metrics."""

from jobscope import jobstats, models
from jobscope.jobstats import (
    GIB,
    bytes_to_gb,
    decode_admin_comment,
    gpus_from_tres,
    jobstats_detail,
    jobstats_metrics,
)

from .conftest import CPU_STATS, GPU_STATS, make_jobstats


def test_decode_round_trip():
    assert decode_admin_comment(make_jobstats(GPU_STATS)) == GPU_STATS


def test_decode_rejects_non_jobstats():
    assert decode_admin_comment("") == {}
    assert decode_admin_comment(None) == {}
    assert decode_admin_comment("JS1:Short") == {}
    assert decode_admin_comment("JS1:None") == {}
    assert decode_admin_comment("not a jobstats summary") == {}


def test_decode_bad_payload_is_empty():
    assert decode_admin_comment("JS1:not-valid-base64!!") == {}


def test_gpus_from_tres():
    assert gpus_from_tres("billing=2,cpu=2,gres/gpu=4,mem=16G") == 4
    assert gpus_from_tres("cpu=2,mem=16G") == 0
    assert gpus_from_tres("") == 0


def test_bytes_to_gb_trims_zeros():
    assert bytes_to_gb(8 * GIB) == "8GB"
    assert bytes_to_gb(0) == "0GB"
    assert bytes_to_gb(int(1.5 * GIB)) == "1.5GB"


def test_jobstats_metrics_gpu_job():
    got = jobstats_metrics(GPU_STATS, gpus=2)
    assert got.known() == {"CPU%": 75, "MEM%": 50, "GPU%": 70, "GMEM%": 50}


def test_jobstats_metrics_rounds_to_whole_percents_as_the_renderers_expect():
    """str() of a reading goes straight into the table and the CSV, so a float
    would print 100.0 where every other release printed 100."""
    got = jobstats_metrics(GPU_STATS, gpus=2)
    assert all(isinstance(v, int) for v in got.known().values())


def test_a_cpu_only_job_has_no_gpu_rather_than_an_unknown_one():
    """`na`, not `unknown`: the job was allocated no GPUs, so there is nothing
    missing. The distinction is what keeps the verdict valid."""
    got = jobstats_metrics(CPU_STATS, gpus=0)
    assert got.known() == {"CPU%": 50, "MEM%": 50}
    assert got.state("GPU%") == models.NA and got.state("GMEM%") == models.NA


def test_gpus_allocated_but_unsampled_is_unknown_not_na():
    """Allocated 2, summary has none: that is a gap in collection, and it must not
    read as 'this job has no GPU'."""
    got = jobstats_metrics(CPU_STATS, gpus=2)
    assert got.state("GPU%") == models.UNKNOWN
    assert "no GPU samples" in got.by_header["GPU%"].reason


def test_jobstats_metrics_empty():
    for stats in ({}, {"no_nodes": 1}):
        got = jobstats_metrics(stats)
        assert not got.by_header and got.known() == {}


def test_a_jobstats_summary_missing_its_core_count_is_unknown_not_zero():
    """This was `else 0`. A fabricated zero passes every is-it-measured guard
    downstream and lands in the summary averages as a real reading, so a job with
    a truncated blob used to be indistinguishable from a genuinely idle one."""
    got = jobstats_metrics({"total_time": 100, "nodes": {"n1": {"total_time": 50}}})
    assert got.state("CPU%") == models.UNKNOWN
    assert got.value("CPU%") is None
    assert "core count" in got.by_header["CPU%"].reason


def test_a_jobstats_summary_missing_its_memory_allocation_is_unknown_not_zero():
    got = jobstats_metrics({"total_time": 100, "nodes": {"n1": {"cpus": 2, "total_time": 100}}})
    assert got.state("MEM%") == models.UNKNOWN
    assert got.value("MEM%") is None


def _as_tuple(row):
    """A UnitRow flattened to what it used to be, so these read as they always did.

    The cells are keyed by header now -- see models.UnitRow -- but the *values* and the
    print order are unchanged, and stating them positionally here is what says so.
    """
    return (row.node, row.unit) + tuple(row.cells[h] for h in jobstats.UNIT_HEADERS)


def test_jobstats_detail_gpu_job():
    rows = jobstats_detail(GPU_STATS)
    assert [_as_tuple(r) for r in rows] == [
        ("node01", "0", "75.0%", "8GB/16GB", "90%", "48GB/80GB", "60.0%"),
        ("node01", "1", "75.0%", "8GB/16GB", "50%", "32GB/80GB", "40.0%"),
    ]


def test_jobstats_detail_cpu_only():
    assert [_as_tuple(r) for r in jobstats_detail(CPU_STATS)] == [
        ("node02", "-", "50.0%", "4GB/8GB", "-", "-", "-"),
    ]


def test_a_unit_row_is_addressed_by_header_not_by_position():
    """The point of the change: a renderer asks for CPU% by name.

    The seven-cell tuple this replaced was written down in the storage helpers, in the
    renderer's _NODE_INDEX/_GPU_INDEX, and in a hand-kept prefix of seven Columns, and a
    site could reconfigure none of them.
    """
    row = jobstats_detail(GPU_STATS)[0]
    assert row.node == "node01" and row.unit == "0"
    assert row.cells["GPU%"] == "90%"
    assert row.cells["CPU-MEM"] == "8GB/16GB"
    assert set(row.cells) == set(jobstats.UNIT_HEADERS)


def test_jobstats_detail_empty():
    assert jobstats_detail({}) == []
