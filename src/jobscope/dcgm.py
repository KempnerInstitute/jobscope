"""DCGM profiling metric catalog and the Prometheus join that populates it.

These metrics are not in the sacct blob; they come from the same Prometheus that
jobstats uses. Each value is the time-average (or max/delta) over the job's
``[start, end]`` window. GPUs are joined to the job by UUID via the
``nvidia_gpu_jobId`` companion series, because the DCGM ``gpu`` index and Slurm
``minor_number`` disagree and only the UUID is stable across the two exporters.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .prometheus import PrometheusClient
from .sacct import JobRecord


@dataclass(frozen=True)
class MetricSpec:
    """One DCGM metric and how to query, reduce, and display it.

    ``scale`` is the display multiplier (fraction to percent, MiB to GiB, mJ to
    kWh); ``decimals`` of 0 renders an integer; ``group`` is ``default`` (always
    shown) or ``all`` (only with the extended catalog). ``reducer`` collapses the
    window (avg | max | delta); ``agg`` reduces across a job's GPUs for the overall
    row (mean | sum | max); ``uuid_label`` is the Prometheus label holding the GPU
    UUID. The last three default to the common case and are set only where a metric
    differs.
    """

    key: str
    header: str
    metric: str
    scale: float
    decimals: int
    group: str
    reducer: str = "avg"
    agg: str = "mean"
    uuid_label: str = "UUID"


METRICS: List[MetricSpec] = [
    MetricSpec("duty", "DUTY%", "nvidia_gpu_duty_cycle", 1, 0, "default", uuid_label="uuid"),
    MetricSpec("smact", "SM_ACT%", "DCGM_FI_PROF_SM_ACTIVE", 100, 1, "default"),
    MetricSpec("occ", "OCC%", "DCGM_FI_PROF_SM_OCCUPANCY", 100, 1, "default"),
    MetricSpec("tensor", "TENSOR%", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", 100, 1, "default"),
    MetricSpec("dram", "DRAM%", "DCGM_FI_PROF_DRAM_ACTIVE", 100, 1, "default"),
    MetricSpec("power", "POWER_W", "DCGM_FI_DEV_POWER_USAGE", 1, 0, "default"),
    MetricSpec("engine", "ENGINE%", "DCGM_FI_PROF_GR_ENGINE_ACTIVE", 100, 1, "all"),
    MetricSpec("hmma", "HMMA%", "DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE", 100, 1, "all"),
    MetricSpec("imma", "IMMA%", "DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE", 100, 1, "all"),
    MetricSpec("dfma", "DFMA%", "DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE", 100, 1, "all"),
    MetricSpec("fp16", "FP16%", "DCGM_FI_PROF_PIPE_FP16_ACTIVE", 100, 1, "all"),
    MetricSpec("fp32", "FP32%", "DCGM_FI_PROF_PIPE_FP32_ACTIVE", 100, 1, "all"),
    MetricSpec("fp64", "FP64%", "DCGM_FI_PROF_PIPE_FP64_ACTIVE", 100, 1, "all"),
    MetricSpec("memcp", "MEMCP%", "DCGM_FI_DEV_MEM_COPY_UTIL", 1, 0, "all"),
    MetricSpec("pwrmax", "PWRmax_W", "DCGM_FI_DEV_POWER_USAGE", 1, 0, "all", reducer="max", agg="max"),
    MetricSpec("energy", "ENERGY_kWh", "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", 1 / 3.6e9, 3, "all",
               reducer="delta", agg="sum"),
    MetricSpec("fbused", "FB_USED_GB", "DCGM_FI_DEV_FB_USED", 1 / 1024, 1, "all"),
    MetricSpec("fbfree", "FB_FREE_GB", "DCGM_FI_DEV_FB_FREE", 1 / 1024, 1, "all"),
    MetricSpec("fbrsvd", "FB_RSVD_GB", "DCGM_FI_DEV_FB_RESERVED", 1 / 1024, 1, "all"),
    MetricSpec("pcietx", "PCIE_TX_MBs", "DCGM_FI_PROF_PCIE_TX_BYTES", 1e-6, 1, "all"),
    MetricSpec("pcierx", "PCIE_RX_MBs", "DCGM_FI_PROF_PCIE_RX_BYTES", 1e-6, 1, "all"),
    MetricSpec("nvlink", "NVLINK_MBs", "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL", 1 / 1024, 1, "all"),
    MetricSpec("smclk", "SMCLK_MHz", "DCGM_FI_DEV_SM_CLOCK", 1, 0, "all"),
    MetricSpec("memclk", "MEMCLK_MHz", "DCGM_FI_DEV_MEM_CLOCK", 1, 0, "all"),
    MetricSpec("temp", "TEMP_C", "DCGM_FI_DEV_GPU_TEMP", 1, 0, "all"),
    MetricSpec("memtemp", "MEMTEMP_C", "DCGM_FI_DEV_MEMORY_TEMP", 1, 0, "all"),
    MetricSpec("enc", "ENC%", "DCGM_FI_DEV_ENC_UTIL", 1, 0, "all"),
    MetricSpec("dec", "DEC%", "DCGM_FI_DEV_DEC_UTIL", 1, 0, "all"),
]

SPEC_BY_HEADER: Dict[str, MetricSpec] = {spec.header: spec for spec in METRICS}
DEFAULT_SPECS: List[MetricSpec] = [spec for spec in METRICS if spec.group == "default"]
ALL_SPECS: List[MetricSpec] = list(METRICS)

# GPU summary columns: the default group minus DUTY% (== the blob's GPU%, already shown).
GPU_SUMMARY_SPECS: List[MetricSpec] = [spec for spec in DEFAULT_SPECS if spec.key != "duty"]
DCGM_HEADERS: List[str] = [spec.header for spec in GPU_SUMMARY_SPECS]

DESCRIPTIONS: Dict[str, str] = {
    "DUTY%": "jobstats' GPU%. Coarse duty cycle: fraction of the run during which at least one "
             "kernel was executing on the GPU. Says the GPU was occupied in time, NOT how "
             "intensely -- a 1-thread kernel and a full-GPU kernel both read ~100%.",
    "SM_ACT%": "Fraction of time at least one warp was resident on an SM, averaged across all SMs. "
               "Distinguishes 'one SM busy' from 'all SMs busy' -- low while DUTY% is high means the "
               "GPU was barely loaded (parked / underfed).",
    "OCC%": "SM occupancy: the fraction of warp slots that were filled, averaged over SMs and "
            "time (active warps / the hardware max per SM). Low occupancy means kernels under-fill "
            "the GPU -- small launches, or register / shared-memory limits.",
    "TENSOR%": "Fraction of time the tensor-core pipe was active. High only for mixed-precision "
               "matmul-heavy work (fp16/bf16/tf32 training or inference); ~0 means the tensor cores "
               "sat idle.",
    "DRAM%": "Fraction of time the device-memory (HBM) interface was busy moving data -- a "
             "memory-bandwidth duty cycle. High while SM_ACT% is low suggests the job is "
             "memory-bound, not compute-bound.",
    "POWER_W": "Mean board power draw over the run, in watts. Compare to the GPU's TDP (~700 W for "
               "H100/H200, ~400 W for A100); near-idle watts mean the GPU was not really working.",
    "ENGINE%": "Fraction of time the graphics/compute engine had work in flight -- a finer-grained "
               "successor to the DUTY% duty cycle.",
    "HMMA%": "Tensor-core activity for half-precision matrix ops (fp16/bf16). A precision breakdown "
             "of TENSOR%.",
    "IMMA%": "Tensor-core activity for integer matrix ops (int8). Precision breakdown of TENSOR% -- "
             "nonzero for quantized / int8 inference.",
    "DFMA%": "Tensor-core activity for double-precision matrix ops (fp64). Precision breakdown of "
             "TENSOR% -- relevant to fp64 HPC on tensor cores.",
    "FP16%": "Fraction of time the (non-tensor) fp16 floating-point pipe was active.",
    "FP32%": "Fraction of time the (non-tensor) fp32 floating-point pipe was active -- the default "
             "precision for much numerical code.",
    "FP64%": "Fraction of time the fp64 (double-precision) pipe was active. High for "
             "double-precision HPC (CFD, MD, dense linear algebra).",
    "MEMCP%": "Percent of time the memory-copy engine was moving data (nvidia-smi's 'Memory' "
              "utilization). Not the same as DRAM bandwidth.",
    "PWRmax_W": "Peak board power seen during the run, in watts (vs POWER_W, the mean).",
    "ENERGY_kWh": "Total energy the GPU consumed over the run, in kWh (from the monotonic energy "
                  "counter, end minus start). Useful for cost / efficiency accounting.",
    "FB_USED_GB": "Mean GPU (framebuffer) memory in use over the run, in GiB. Complements jobstats' "
                  "GMEM%, which is the PEAK -- a job that loads a model then idles shows high peak but a "
                  "lower mean.",
    "FB_FREE_GB": "Mean free GPU memory over the run, in GiB.",
    "FB_RSVD_GB": "Mean GPU memory reserved by the driver/system over the run, in GiB (not available to "
                  "your job).",
    "PCIE_TX_MBs": "Mean PCIe transmit throughput (GPU -> host), in MB/s. A bottleneck if the GPU waits "
                   "on host transfers. (DCGM rate; treat the absolute value as approximate.)",
    "PCIE_RX_MBs": "Mean PCIe receive throughput (host -> GPU), in MB/s -- e.g. input batches streamed to "
                   "the GPU. (DCGM rate; absolute value approximate.)",
    "NVLINK_MBs": "Mean NVLink throughput, in MiB/s. ~0 for single-GPU jobs; nonzero indicates "
                  "multi-GPU communication (e.g. NCCL all-reduce). (DCGM rate; absolute approximate.)",
    "SMCLK_MHz": "Mean SM (core) clock frequency, in MHz. A low mean alongside high temperature / power "
                 "can indicate throttling.",
    "MEMCLK_MHz": "Mean memory clock frequency, in MHz.",
    "TEMP_C": "Mean GPU core temperature, in degrees Celsius.",
    "MEMTEMP_C": "Mean GPU memory (HBM) temperature, in degrees Celsius.",
    "ENC%": "Mean hardware video-encoder (NVENC) utilization. Usually 0 unless encoding video.",
    "DEC%": "Mean hardware video-decoder (NVDEC) utilization. Usually 0 unless decoding video.",
}


def format_value(spec: MetricSpec, value: Optional[float]) -> str:
    """Format one metric cell ('-' when missing), using the metric's decimals."""
    if value is None:
        return "-"
    if spec.decimals:
        return ("{:.%df}" % spec.decimals).format(value)
    return str(int(round(value)))


def format_by_header(header: str, value: Optional[float]) -> str:
    """Format a metric cell by header (summary/detail callers); see format_value."""
    return format_value(SPEC_BY_HEADER[header], value)


def gpu_minor_key(minor):
    """Numeric sort key for a GPU minor number; falls back to string."""
    return int(minor) if str(minor).isdigit() else minor


def window_query(spec: MetricSpec, uuids: List[str], duration: int) -> str:
    """PromQL that reduces ``spec`` over a ``duration``-second window for ``uuids``."""
    regex = "^(" + "|".join(uuids) + ")$"  # UUIDs are hex+hyphen, RE2-safe as-is
    selector = '%s{%s=~"%s"}' % (spec.metric, spec.uuid_label, regex)
    if spec.reducer == "avg":
        return "avg_over_time((%s)[%ds:])" % (selector, duration)
    if spec.reducer == "max":
        return "max_over_time((%s)[%ds:])" % (selector, duration)
    return "(max_over_time((%s)[%ds:]) - min_over_time((%s)[%ds:]))" % (
        selector, duration, selector, duration)


def _jobid_query(record: JobRecord) -> str:
    cluster = "slurm_cluster='%s'" % record.cluster if record.cluster else ""
    return ("max_over_time((nvidia_gpu_jobId{%s} == %s)[%ds:])"
            % (cluster, record.jobid_raw, record.duration))


def discover_gpus(record: JobRecord, client: PrometheusClient,
                  timeout: Optional[float]) -> List[dict]:
    """The GPUs that ran a job, as ``{uuid, node, minor}`` sorted by (node, minor).

    Empty for CPU-only jobs or when no GPU samples exist. Joins via
    ``nvidia_gpu_jobId`` -- the same mapping jobstats uses.
    """
    if not (record.gpus and record.jobid_raw and record.duration):
        return []
    try:
        found = client.query(_jobid_query(record), record.end, timeout)
    except Exception:
        return []
    gpus = []
    for series in found:
        metric = series["metric"]
        uuid = metric.get("uuid")
        if uuid:
            gpus.append({"uuid": uuid,
                         "node": metric.get("host", "?").split(":")[0],
                         "minor": str(metric.get("minor_number", "?"))})
    gpus.sort(key=lambda g: (g["node"], gpu_minor_key(g["minor"])))
    return gpus


def dcgm_for_job(record: JobRecord, specs: List[MetricSpec], client: PrometheusClient,
                 timeout: Optional[float]) -> Tuple[dict, dict]:
    """``(overall, per_gpu)`` metric dicts for one job over ``specs``.

    ``({}, {})`` when the job has no GPUs or no samples. ``overall`` is keyed by
    header; ``per_gpu`` is keyed by ``(node, minor)``. Series are joined to the
    job on UUID via ``nvidia_gpu_jobId``.
    """
    if not (record.gpus and record.jobid_raw and record.duration):
        return {}, {}
    try:
        found = client.query(_jobid_query(record), record.end, timeout)
    except Exception:
        return {}, {}
    gpus, uuids = [], []
    for series in found:
        metric = series["metric"]
        uuid = metric.get("uuid")
        if uuid:
            gpus.append((metric.get("host", "?").split(":")[0],
                         str(metric.get("minor_number", "?")), uuid))
            uuids.append(uuid)
    if not uuids:
        return {}, {}

    per_uuid: Dict[str, dict] = {uuid: {} for uuid in uuids}
    for spec in specs:
        try:
            result = client.query(window_query(spec, uuids, record.duration),
                                  record.end, timeout)
        except Exception:
            continue
        for series in result:
            metric = series["metric"]
            uuid = metric.get(spec.uuid_label) or metric.get("uuid") or metric.get("UUID")
            try:
                if uuid in per_uuid:
                    per_uuid[uuid][spec.header] = float(series["value"][1]) * spec.scale
            except (TypeError, ValueError):
                pass

    per_gpu = {(node, minor): per_uuid.get(uuid, {}) for node, minor, uuid in gpus}
    overall: Dict[str, float] = {}
    for spec in specs:
        values = [per_uuid[u][spec.header] for u in uuids if spec.header in per_uuid[u]]
        if values:
            overall[spec.header] = (sum(values) if spec.agg == "sum"
                                    else max(values) if spec.agg == "max"
                                    else sum(values) / len(values))
    return overall, per_gpu


def compute_dcgm(records: Dict[str, JobRecord], jobids: List[str],
                 specs: List[MetricSpec], client: PrometheusClient,
                 timeout: Optional[float], workers: int) -> Dict[str, Tuple[dict, dict]]:
    """Run :func:`dcgm_for_job` over every GPU job, concurrently.

    The per-job queries are network I/O-bound, so a thread pool overlaps them and
    speeds up wide selections even on one CPU core. The per-metric queries within a
    job stay sequential -- the parallelism is across jobs.
    """
    gpu_jobs = [jid for jid in jobids if jid in records and records[jid].gpus]
    if not gpu_jobs:
        return {}
    workers = max(1, min(workers, len(gpu_jobs)))
    if workers == 1:
        return {jid: dcgm_for_job(records[jid], specs, client, timeout) for jid in gpu_jobs}
    results: Dict[str, Tuple[dict, dict]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(dcgm_for_job, records[jid], specs, client, timeout): jid
                   for jid in gpu_jobs}
        for future, jid in futures.items():
            try:
                results[jid] = future.result()
            except Exception:
                results[jid] = ({}, {})
    return results
