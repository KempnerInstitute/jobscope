"""DCGM profiling metric catalog and the Prometheus join that populates it.

These metrics are not in the sacct blob; they come from the same Prometheus that
jobstats uses. Each value is the time-average (or max/delta) over the job's
``[start, end]`` window. GPUs are joined to the job by UUID via the
``nvidia_gpu_jobId`` companion series, because the DCGM ``gpu`` index and Slurm
``minor_number`` disagree and only the UUID is stable across the two exporters.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

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
    show: bool = True   # False = queried only to feed a derived column


METRICS: List[MetricSpec] = [
    MetricSpec("duty", "GPU%", "nvidia_gpu_duty_cycle", 1, 0, "default", uuid_label="uuid"),
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
    # NVML GPU memory, the pair jobstats reports as "GPU memory usage per node -
    # maximum used/total". Peaked, not averaged, so it is comparable to jobstats.
    # Named GMEM_* to match the blob-derived GMEM% of the summary and detail views:
    # a bare MEM% means HOST memory there, and reusing it for GPU memory both reads
    # as the wrong quantity and grades against the host threshold in plots.
    MetricSpec("mem", "GMEM_GB", "nvidia_gpu_memory_used_bytes", 1 / 1024 ** 3, 1, "default",
               reducer="max", agg="max", uuid_label="uuid"),
    MetricSpec("memtot", "GMEM_TOTAL_GB", "nvidia_gpu_memory_total_bytes", 1 / 1024 ** 3, 1,
               "default", reducer="max", agg="max", uuid_label="uuid", show=False),
]


class Derived(NamedTuple):
    """A column computed from queried metrics rather than fetched from Prometheus."""

    key: str
    header: str
    decimals: int
    deps: Tuple[str, ...]   # metric keys it needs; absent -> the column is skipped
    fn: Callable[[Dict[str, Optional[float]]], Optional[float]]
    source: str             # what it is computed from, for --describe


def _gmem_percent(values: Dict[str, Optional[float]]) -> Optional[float]:
    """GPU memory used as a percentage of that GPU's own total."""
    used, total = values.get("mem"), values.get("memtot")
    if used is None or not total:
        return None
    return used / total * 100


# Columns computed from other metrics. Shared by the dcgm and live views so both
# render the same set; each declares the metric keys it needs, so it appears only
# where those were actually collected. ``fn`` receives a dict keyed by metric key,
# which callers storing values by header must build first -- see values_by_key.
DERIVED_COLUMNS: List[Derived] = [
    Derived("gmempct", "GMEM%", 1, ("mem", "memtot"), _gmem_percent,
            "GMEM_GB / nvidia_gpu_memory_total"),
]


def applicable_derived(specs: List[MetricSpec]) -> List[Derived]:
    """The derived columns whose input metrics are all present in ``specs``."""
    fetched = {spec.key for spec in specs}
    return [d for d in DERIVED_COLUMNS if fetched.issuperset(d.deps)]


def columns_for(specs: List[MetricSpec]) -> List[Tuple[str, str, int]]:
    """``(key, header, decimals)`` per displayed column, derived ones included.

    Hidden specs (``show=False``) are dropped, and each derived column is inserted
    next to the metric it is computed from rather than trailing at the far end.
    """
    cols = [(s.key, s.header, s.decimals) for s in specs if s.show]
    for derived in applicable_derived(specs):
        from_deps = [i for i, (key, _h, _d) in enumerate(cols) if key in derived.deps]
        cols.insert(max(from_deps) + 1 if from_deps else len(cols),
                    (derived.key, derived.header, derived.decimals))
    return cols


def values_by_key(specs: List[MetricSpec], by_header: Dict[str, float]
                  ) -> Dict[str, Optional[float]]:
    """Re-key one GPU's values from header to metric key, for a derived ``fn``."""
    return {spec.key: by_header.get(spec.header) for spec in specs}


SPEC_BY_HEADER: Dict[str, MetricSpec] = {spec.header: spec for spec in METRICS}
DEFAULT_SPECS: List[MetricSpec] = [spec for spec in METRICS if spec.group == "default"]
ALL_SPECS: List[MetricSpec] = list(METRICS)

# Quantities the sacct blob already supplies, which the summary and detail views
# render from it directly (GPU%, GMEM%, and GPU-MEM). Excluded from those views'
# DCGM columns so a job does not get two columns for one number -- and for a
# finished job they would be the very same number, see _prefer_stored.
BLOB_BACKED_KEYS = ("duty", "mem", "memtot")
GPU_SUMMARY_SPECS: List[MetricSpec] = [spec for spec in DEFAULT_SPECS
                                       if spec.key not in BLOB_BACKED_KEYS]
DCGM_HEADERS: List[str] = [spec.header for spec in GPU_SUMMARY_SPECS]

# The column headers those keys produce, including the derived GMEM%. Renderers use
# this to keep a blob-backed quantity out of the profiling block.
DCGM_BLOB_HEADERS: Tuple[str, ...] = tuple(
    [spec.header for spec in METRICS if spec.key in BLOB_BACKED_KEYS]
    + [d.header for d in DERIVED_COLUMNS if set(d.deps) & set(BLOB_BACKED_KEYS)])

DESCRIPTIONS: Dict[str, str] = {
    "GPU%": "NVML's duty cycle: the fraction of the run during which at least one kernel was "
            "executing on the GPU, the same number jobstats reports. Says the GPU was occupied "
            "in time, NOT how intensely: a 1-thread kernel and a full-GPU kernel both read ~100%. "
            "For a finished job this comes from the stored blob, so every view agrees. NVML stops "
            "reporting it once MIG is enabled, so it is \"-\" on a MIG node.",
    "SM_ACT%": "Fraction of time at least one warp was resident on an SM, averaged across all "
               "SMs. Distinguishes 'one SM busy' from 'all SMs busy'; low while GPU% is high "
               "means the GPU was barely loaded (parked / underfed). Measured cluster-wide it runs "
               "~20 points BELOW GPU% on 93% of active GPUs, so do not read it as "
               "\"GPU utilization\"; the gap between the two is the diagnostic signal.",
    "OCC%": "SM occupancy: the fraction of warp slots that were filled, averaged over SMs and "
            "time (active warps / the hardware max per SM). Low occupancy means kernels under-fill "
            "the GPU: small launches, or register / shared-memory limits.",
    "TENSOR%": "Fraction of time the tensor-core pipe was active. High only for mixed-precision "
               "matmul-heavy work (fp16/bf16/tf32 training or inference); ~0 means the tensor cores "
               "sat idle.",
    "DRAM%": "Fraction of time the device-memory (HBM) interface was busy moving data, a "
             "memory-bandwidth duty cycle. High while SM_ACT% is low suggests the job is "
             "memory-bound, not compute-bound.",
    "POWER_W": "Mean board power draw over the run, in watts. Compare to the GPU's TDP (~700 W for "
               "H100/H200, ~400 W for A100); near-idle watts mean the GPU was not really working.",
    "ENGINE%": "Fraction of time the graphics/compute engine had work in flight, DCGM's "
               "finer-grained analogue of GPU%. Tracks it closely over a job-length window "
               "(median |delta| 1.2, r=0.99 at 1h), so it is a usable stand-in; the two disagree "
               "far more on a single scrape, as they are scraped independently.",
    "HMMA%": "Tensor-core activity for half-precision matrix ops (fp16/bf16). A precision breakdown "
             "of TENSOR%.",
    "IMMA%": "Tensor-core activity for integer matrix ops (int8). Precision breakdown of TENSOR%: "
             "nonzero for quantized / int8 inference.",
    "DFMA%": "Tensor-core activity for double-precision matrix ops (fp64). Precision breakdown of "
             "TENSOR%: relevant to fp64 HPC on tensor cores.",
    "FP16%": "Fraction of time the (non-tensor) fp16 floating-point pipe was active.",
    "FP32%": "Fraction of time the (non-tensor) fp32 floating-point pipe was active, the default "
             "precision for much numerical code.",
    "FP64%": "Fraction of time the fp64 (double-precision) pipe was active. High for "
             "double-precision HPC (CFD, MD, dense linear algebra).",
    "MEMCP%": "Percent of time the memory-copy engine was moving data (nvidia-smi's 'Memory' "
              "utilization). Not the same as DRAM bandwidth.",
    "PWRmax_W": "Peak board power seen during the run, in watts (vs POWER_W, the mean).",
    "ENERGY_kWh": "Total energy the GPU consumed over the run, in kWh (from the monotonic energy "
                  "counter, end minus start). Useful for cost / efficiency accounting.",
    "FB_USED_GB": "Mean GPU (framebuffer) memory in use over the run, in GiB. Complements jobstats' "
                  "GMEM%, which is the PEAK: a job that loads a model then idles shows high peak but a "
                  "lower mean.",
    "FB_FREE_GB": "Mean free GPU memory over the run, in GiB.",
    "FB_RSVD_GB": "Mean GPU memory reserved by the driver/system over the run, in GiB (not available to "
                  "your job).",
    "PCIE_TX_MBs": "Mean PCIe transmit throughput (GPU -> host), in MB/s. A bottleneck if the GPU waits "
                   "on host transfers. (DCGM rate; treat the absolute value as approximate.)",
    "PCIE_RX_MBs": "Mean PCIe receive throughput (host -> GPU), in MB/s, e.g. input batches streamed to "
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
    "GMEM_GB": "GPU framebuffer memory in use, in GiB, from the NVML exporter. Peaked rather than "
               "averaged, so it is jobstats' \"GPU memory usage per node - maximum used/total\". "
               "For a finished job this comes from the stored blob, so every view agrees.",
    "GMEM_TOTAL_GB": "Total framebuffer memory on the GPU, in GiB. Fetched only to derive GMEM%; on "
                     "a MIG instance this is the slice's share, not the physical card's.",
    "GMEM%": "GMEM_GB as a percent of that GPU's total memory, the per-GPU form of the summary "
             "view's GMEM%. Named GMEM% rather than MEM% because a bare MEM% means HOST memory "
             "elsewhere. On a MIG row the total is the slice's, so the percentage is per slice.",
}

def format_number(value: Optional[float], decimals: int, missing: str = "-") -> str:
    """Format one metric cell to ``decimals`` places; ``missing`` when value is None."""
    if value is None:
        return missing
    if decimals:
        return ("{:.%df}" % decimals).format(value)
    return str(int(round(value)))


def format_value(spec: MetricSpec, value: Optional[float]) -> str:
    """Format one metric cell ('-' when missing), using the metric's decimals."""
    return format_number(value, spec.decimals)


def format_by_header(header: str, value: Optional[float]) -> str:
    """Format a metric cell by header (summary/detail callers); see format_value."""
    return format_value(SPEC_BY_HEADER[header], value)


def gpu_minor_key(minor):
    """Numeric sort key for a GPU minor number; falls back to string."""
    return int(minor) if str(minor).isdigit() else minor


def window_query(spec: MetricSpec, uuids: List[str], duration: int,
                 clip: Optional[str] = None) -> str:
    """PromQL that reduces ``spec`` over a ``duration``-second window for ``uuids``.

    ``clip`` is an optional series to intersect the selector with, which restricts
    the window to the samples where that series also existed. Only usable when the
    two come from the same exporter, since PromQL's ``and`` requires identical
    label sets -- see :func:`jobscope.live.clip_to_job`.
    """
    regex = "^(" + "|".join(uuids) + ")$"  # UUIDs are hex+hyphen, RE2-safe as-is
    selector = '%s{%s=~"%s"}' % (spec.metric, spec.uuid_label, regex)
    if clip:
        selector = "%s and %s" % (selector, clip)
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
    ``nvidia_gpu_jobId``, the same mapping jobstats uses.
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
        # Aggregated across UUIDs rather than per_gpu keys: per_gpu is keyed by
        # (node, minor), which MIG siblings share, so averaging it would drop
        # instances.
        values = [per_uuid[u][spec.header] for u in uuids if spec.header in per_uuid[u]]
        if values:
            overall[spec.header] = (sum(values) if spec.agg == "sum"
                                    else max(values) if spec.agg == "max"
                                    else sum(values) / len(values))
    _prefer_stored(record, specs, per_gpu, overall)
    _add_derived(specs, per_gpu, overall)
    return overall, per_gpu


# Blob field -> the metric key it supersedes, and how a job-level figure is formed
# from the per-GPU values. GMEM_GB sums because the blob's GMEM% is the ratio of
# summed used to summed total across the job's GPUs.
_STORED_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("gpu_utilization", "duty", "mean"),
    ("gpu_used_memory", "mem", "sum"),
    ("gpu_total_memory", "memtot", "sum"),
)


def stored_per_gpu(record: JobRecord, field: str) -> Dict[Tuple[str, str], float]:
    """One per-GPU map from the job's stored blob, keyed ``(node, minor)``.

    Empty when the job has no blob, which is every running job -- Slurm writes it
    at job end.
    """
    found: Dict[Tuple[str, str], float] = {}
    for node, info in (record.stats or {}).get("nodes", {}).items():
        for minor, value in (info.get(field) or {}).items():
            try:
                found[(node, str(minor))] = float(value)
            except (TypeError, ValueError):
                continue
    return found


def stored_utilization(record: JobRecord) -> Dict[Tuple[str, str], float]:
    """Per-GPU utilization from the job's stored blob, keyed ``(node, minor)``."""
    return stored_per_gpu(record, "gpu_utilization")


def _prefer_stored(record: JobRecord, specs: List[MetricSpec],
                   per_gpu: Dict[Tuple[str, str], dict],
                   overall: Dict[str, float]) -> None:
    """Overwrite GPU% and GPU memory with the blob's values where it has them.

    These come from the same ``nvidia_gpu_*`` series under the same reducers, so
    they measure the same thing -- but the blob is what jobstats computed at job
    end, while recomputing here has to reconstruct the window from sacct's Start
    and End. On a short job that boundary is worth several points (a 570s job at a
    60s scrape interval has ~10 samples, so one sample in or out moves the mean by
    ~8), which showed up as this view disagreeing with the summary view's GPU% for
    the same job.

    Rather than tune the window to imitate an instant we cannot recover, defer to
    the stored value: a finished job then reports exactly what Slurm recorded, in
    every view. Running jobs have no blob, so they keep the Prometheus value --
    reconstructed by :mod:`jobscope.live_blob` for the blob columns, so those agree
    with each other by construction.
    """
    by_key = {spec.key: spec for spec in specs}
    for field, key, agg in _STORED_FIELDS:
        spec = by_key.get(key)
        if spec is None:
            continue
        stored = stored_per_gpu(record, field)
        if not stored:
            continue
        for node_minor, value in stored.items():
            if node_minor in per_gpu:
                per_gpu[node_minor][spec.header] = value * spec.scale
        total = sum(stored.values()) * spec.scale
        overall[spec.header] = total / len(stored) if agg == "mean" else total


def _add_derived(specs: List[MetricSpec], per_gpu: Dict[Tuple[str, str], dict],
                 overall: Dict[str, float]) -> None:
    """Fill in the derived columns, per GPU and for the job as a whole."""
    derived = applicable_derived(specs)
    if not derived:
        return
    for values in list(per_gpu.values()) + [overall]:
        keyed = values_by_key(specs, values)
        for column in derived:
            computed = column.fn(keyed)
            if computed is not None:
                values[column.header] = computed


def compute_dcgm(records: Dict[str, JobRecord], jobids: List[str],
                 specs: List[MetricSpec], client: PrometheusClient,
                 timeout: Optional[float], workers: int) -> Dict[str, Tuple[dict, dict]]:
    """Run :func:`dcgm_for_job` over every GPU job, concurrently.

    The per-job queries are network I/O-bound, so a thread pool overlaps them and
    speeds up wide selections even on one CPU core. The per-metric queries within a
    job stay sequential; the parallelism is across jobs.
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
