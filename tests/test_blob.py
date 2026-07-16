"""Tests for the sacct AdminComment blob decoder and metrics."""

from jobscope.blob import (
    GIB,
    blob_detail,
    blob_metrics,
    bytes_to_gb,
    decode_admin_comment,
    gpus_from_tres,
)

from .conftest import CPU_STATS, GPU_STATS, make_blob


def test_decode_round_trip():
    assert decode_admin_comment(make_blob(GPU_STATS)) == GPU_STATS


def test_decode_rejects_non_blob():
    assert decode_admin_comment("") == {}
    assert decode_admin_comment(None) == {}
    assert decode_admin_comment("JS1:Short") == {}
    assert decode_admin_comment("JS1:None") == {}
    assert decode_admin_comment("not a blob") == {}


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


def test_blob_metrics_gpu_job():
    assert blob_metrics(GPU_STATS) == (75, 50, 70, 50)


def test_blob_metrics_cpu_only_has_no_gpu():
    assert blob_metrics(CPU_STATS) == (50, 50, None, None)


def test_blob_metrics_empty():
    assert blob_metrics({}) is None
    assert blob_metrics({"no_nodes": 1}) is None


def test_blob_detail_gpu_job():
    rows = blob_detail(GPU_STATS)
    assert rows == [
        ("node01", "0", "75.0%", "8GB/16GB", "90%", "48GB/80GB", "60.0%"),
        ("node01", "1", "75.0%", "8GB/16GB", "50%", "32GB/80GB", "40.0%"),
    ]


def test_blob_detail_cpu_only():
    assert blob_detail(CPU_STATS) == [
        ("node02", "-", "50.0%", "4GB/8GB", "-", "-", "-"),
    ]


def test_blob_detail_empty():
    assert blob_detail({}) == []
