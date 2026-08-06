"""Tests for Slurm's own accounting as a cross-check source.

The fixture rows are copied from real sacct output on the cluster, because every
bug this module had came from a shape that looked implausible until it appeared:
a batch step that is not the job, a duration with four-digit days, and a memory
figure that means something different in two adjacent columns.
"""

import pytest

from jobscope import extra_metric as em

F = {name: i for i, name in enumerate(em.FIELDS)}


def row(**kw):
    """A sacct row as a field list, defaulting everything unset to blank."""
    cells = [""] * len(em.FIELDS)
    for name, value in kw.items():
        cells[F[name]] = value
    return cells


# --- duration parsing -------------------------------------------------------

@pytest.mark.parametrize("text,seconds", [
    ("00:00:00", 0),
    ("05:24:40", 5 * 3600 + 24 * 60 + 40),
    ("1-18:02:33", 86400 + 18 * 3600 + 2 * 60 + 33),
    ("4580-00:04:33", 4580 * 86400 + 4 * 60 + 33),      # four-digit days are real
    ("40:30.266", 40 * 60 + 30.266),                     # MM:SS -- forty minutes
    ("30.266", 30.266),
])
def test_parse_duration_handles_every_shape_sacct_emits(text, seconds):
    assert em.parse_duration(text) == pytest.approx(seconds)


def test_the_two_field_form_is_minutes_not_hours():
    """`40:30.266` is forty minutes. Reading the leftmost field as hours makes a
    40-minute job look like a 40-hour one, and its CPU% 60x too small."""
    assert em.parse_duration("40:30.266") < em.parse_duration("01:00:00")


@pytest.mark.parametrize("text", ["", None, "  ", "notatime", "a:b:c"])
def test_parse_duration_returns_none_rather_than_zero(text):
    """None means Slurm did not record it; zero would mean it recorded idleness."""
    assert em.parse_duration(text) is None


# --- size parsing -----------------------------------------------------------

@pytest.mark.parametrize("text,size", [
    ("1746956K", 1746956 * 1024),
    ("4302M", 4302 * 1024 ** 2),
    ("64G", 64 * 1024 ** 3),
    ("168857588K", 168857588 * 1024),
    ("382", 382),                     # bare = bytes (or a unitless count)
    ("0", 0),
])
def test_parse_size_handles_slurms_suffixes(text, size):
    assert em.parse_size(text) == pytest.approx(size)


@pytest.mark.parametrize("text", ["", None, "  ", "notasize"])
def test_parse_size_returns_none_for_a_blank(text):
    assert em.parse_size(text) is None


def test_parse_tres_splits_the_comma_list():
    got = em.parse_tres("cpu=00:16:35,energy=0,gres/gpuutil=382,mem=448032K")
    assert got["gres/gpuutil"] == "382"
    assert got["mem"] == "448032K"
    assert em.parse_tres("") == {}


# --- GPU counting -----------------------------------------------------------

def test_gpus_from_tres_reads_the_untyped_form():
    assert em.gpus_from_tres(
        "billing=3352,cpu=4,gres/gpu:nvidia_a100-sxm4-80gb=4,gres/gpu=4,mem=48G") == 4


def test_gpus_from_tres_falls_back_to_the_typed_form():
    """A site emitting only the typed spelling still gets a count."""
    assert em.gpus_from_tres("cpu=4,gres/gpu:nvidia_h200=2,mem=48G") == 2


def test_gpus_from_tres_is_zero_for_a_cpu_job():
    assert em.gpus_from_tres("cpu=8,mem=64G,node=1") == 0


# --- the multi-node trap ----------------------------------------------------

# Job 35358552 as sacct really reports it: 16 nodes, 1024 CPUs, 4.5 days. The
# batch step is the script on the head node and covers 64 CPUs; step .0 is the
# actual work. Taking .batch for the job would report CPU% 0.0 instead of 99.2.
MULTINODE = [
    row(JobID="35358552", TotalCPU="4580-00:04:33", UserCPU="4579-00:00:00",
        SystemCPU="1-00:04:33", CPUTime="4615-13:45:36", ReqMem="500G",
        NNodes="16", NCPUS="1024", AllocTRES="cpu=1024,mem=500G,node=16"),
    row(JobID="35358552.batch", TotalCPU="18:35.933", CPUTime="288-11:21:36",
        MaxRSS="448032K", NNodes="1", NCPUS="64",
        TRESUsageInTot="cpu=00:16:35,energy=0,mem=448032K,pages=22"),
    row(JobID="35358552.extern", TotalCPU="00:00:00", CPUTime="4615-13:45:36",
        NNodes="16", NCPUS="1024", TRESUsageInTot="energy=0"),
    row(JobID="35358552.0", TotalCPU="4579-23:45:57", CPUTime="4611-04:13:52",
        MaxRSS="174992K", NNodes="16", NCPUS="1024",
        TRESUsageInTot="cpu=4579-23:45:48,energy=0,mem=168857588K,pages=4545"),
]


def _metrics(rows):
    alloc = [r for r in rows if "." not in r[0]][0]
    steps = [r for r in rows if "." in r[0]
             and not any(r[0].endswith(s) for s in em.SKIP_STEPS)]
    return em._from_rows(alloc[0], alloc, steps)


def test_cpu_percent_comes_from_the_rolled_up_allocation_row():
    """Not from .batch, which on this job is 0.004% of the work."""
    assert _metrics(MULTINODE).cpu_pct == pytest.approx(99.2, abs=0.1)


def test_memory_uses_summed_rss_not_the_peak_single_task():
    """MaxRSS is one task's peak: 0.4G here, against a 500G allocation the job
    really filled to 161G. Using it would report 0.1% instead of 32%."""
    got = _metrics(MULTINODE)
    assert got.mem_pct == pytest.approx(32.2, abs=0.1)
    assert got.used_mem_bytes == pytest.approx(168857588 * 1024)
    assert got.max_rss_bytes == pytest.approx(448032 * 1024)   # kept, for display


def test_the_extern_step_is_ignored():
    """It wraps the allocation and reports zero work; including it drags maxima."""
    assert em.SKIP_STEPS == (".extern",)
    with_extern = _metrics(MULTINODE)
    without = _metrics([r for r in MULTINODE if not r[0].endswith(".extern")])
    assert with_extern == without


def test_a_cpu_only_job_has_no_gpu_readings():
    got = _metrics(MULTINODE)
    assert got.gpus == 0 and got.gpu_pct is None and got.gpu_mem_bytes is None


# --- the GPU sum ------------------------------------------------------------

# Job 35269459_0: 4x A100, and gres/gpuutil=382 is the *sum* across them.
FOUR_GPU = [
    row(JobID="35269459_0", TotalCPU="1-00:57:58", CPUTime="1-11:26:16",
        ReqMem="48G", NNodes="1", NCPUS="4",
        AllocTRES="cpu=4,gres/gpu:nvidia_a100-sxm4-80gb=4,gres/gpu=4,mem=48G,node=1"),
    row(JobID="35269459_0.batch", TotalCPU="1-00:57:58", MaxRSS="10055528K",
        NNodes="1", NCPUS="4",
        TRESUsageInTot="cpu=1-00:57:58,gres/gpumem=248460M,gres/gpuutil=382,"
                       "mem=10055528K"),
]


def test_gpu_percent_is_divided_by_the_card_count():
    """382 across 4 GPUs is ~95.5% each, not 382%. Invisible on a 1-GPU job,
    which is exactly how this survives review."""
    got = _metrics(FOUR_GPU)
    assert got.gpus == 4
    assert got.gpu_pct == pytest.approx(95.5, abs=0.1)


def test_gpu_memory_is_also_per_card():
    got = _metrics(FOUR_GPU)
    assert got.gpu_mem_bytes == pytest.approx(248460 * 1024 ** 2 / 4)


def test_a_single_gpu_job_needs_no_division_but_takes_the_same_path():
    one = [
        row(JobID="1", TotalCPU="40:30.266", CPUTime="05:24:40", ReqMem="64G",
            NNodes="1", NCPUS="8", AllocTRES="cpu=8,gres/gpu=1,mem=64G,node=1"),
        row(JobID="1.batch", MaxRSS="1746956K", NNodes="1", NCPUS="8",
            TRESUsageInTot="gres/gpuutil=96,gres/gpumem=4302M,mem=1746956K"),
    ]
    got = _metrics(one)
    assert got.gpu_pct == pytest.approx(96.0)
    assert got.cpu_pct == pytest.approx(12.5, abs=0.1)
    assert got.mem_pct == pytest.approx(2.6, abs=0.1)


# --- absent is not zero -----------------------------------------------------

def test_an_unrecorded_quantity_is_none_not_zero():
    """The distinction the whole cross-check rests on: a job Slurm has no memory
    figure for is unknown, not idle."""
    bare = [row(JobID="7", NNodes="1", NCPUS="8", AllocTRES="cpu=8,mem=64G")]
    got = _metrics(bare)
    assert got.cpu_pct is None and got.mem_pct is None and got.gpu_pct is None


def test_a_zero_denominator_yields_none_rather_than_dividing():
    rows = [row(JobID="8", TotalCPU="01:00:00", CPUTime="00:00:00", NNodes="1",
                NCPUS="8", AllocTRES="cpu=8")]
    assert _metrics(rows).cpu_pct is None


def test_energy_is_none_when_the_gatherer_is_off():
    """AcctGatherEnergyType null makes ConsumedEnergyRaw read 0 on every job; that
    is 'not measured', and reporting 0 J would be a claim nobody made."""
    rows = [row(JobID="9", NNodes="1", NCPUS="1", ConsumedEnergyRaw="0")]
    assert _metrics(rows).energy_j is None


def test_collect_returns_nothing_when_sacct_says_nothing(monkeypatch):
    monkeypatch.setattr(em, "run_capture", lambda *a, **kw: "")
    assert em.collect(["1"], None) == {}
    assert em.collect([], None) == {}


def test_collect_groups_steps_under_their_job(monkeypatch):
    text = "\n".join("|".join(r) for r in MULTINODE + FOUR_GPU)
    monkeypatch.setattr(em, "run_capture", lambda *a, **kw: text)
    got = em.collect(["35358552", "35269459_0"], None)
    assert set(got) == {"35358552", "35269459_0"}
    assert got["35358552"].cpu_pct == pytest.approx(99.2, abs=0.1)
    assert got["35269459_0"].gpu_pct == pytest.approx(95.5, abs=0.1)


def test_array_elements_stay_separate_jobs(monkeypatch):
    """`123_4` and `123_5` are different jobs; only `123_4.batch` folds into one."""
    rows = [
        row(JobID="123_4", TotalCPU="01:00:00", CPUTime="02:00:00", NNodes="1",
            NCPUS="1", AllocTRES="cpu=1"),
        row(JobID="123_4.batch", MaxRSS="100K", NNodes="1", NCPUS="1"),
        row(JobID="123_5", TotalCPU="00:30:00", CPUTime="02:00:00", NNodes="1",
            NCPUS="1", AllocTRES="cpu=1"),
    ]
    monkeypatch.setattr(em, "run_capture", lambda *a, **kw: "\n".join("|".join(r) for r in rows))
    got = em.collect(["123_4", "123_5"], None)
    assert set(got) == {"123_4", "123_5"}
    assert got["123_4"].cpu_pct == pytest.approx(50.0)
    assert got["123_5"].cpu_pct == pytest.approx(25.0)
