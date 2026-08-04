"""Per-column source resolution.

The policy these assert used to be three literals in ``dcgm.JOBSTATS_BACKED_KEYS`` plus a
hardcoded preference for the jobstats summary inside ``_prefer_stored``. It is now data, so the
thing worth testing is that the data reproduces the old behaviour by default and
changes only what a stated preference asks it to.
"""

import pytest

from jobscope import dcgm, source
from jobscope.errors import JobscopeError


@pytest.fixture(autouse=True)
def restore_preference():
    """Every test here moves module state, so put it back."""
    yield
    dcgm.set_preference(source.DEFAULT_PREFERENCE)


# --- parsing ----------------------------------------------------------------

def test_naming_one_source_promotes_it_and_keeps_the_rest():
    """A preference is a reordering, not a filter: a column whose only candidate is an
    unnamed source still has to be served, or naming dcgm would silently drop GMEM%."""
    assert source.parse_preference("dcgm") == ("dcgm", "jobstats", "nvml")
    assert source.parse_preference("nvml") == ("nvml", "jobstats", "dcgm")
    assert set(source.parse_preference("dcgm")) == set(source.SOURCES)


def test_an_order_may_be_given_in_full_or_comma_separated():
    assert source.parse_preference("jobstats,nvml") == ("jobstats", "nvml", "dcgm")
    assert source.parse_preference(["nvml", "dcgm"]) == ("nvml", "dcgm", "jobstats")


def test_a_repeated_name_is_not_repeated_in_the_order():
    assert source.parse_preference("dcgm,dcgm,jobstats") == ("dcgm", "jobstats", "nvml")


@pytest.mark.parametrize("bad", ["dgcm", "prometheus", "", [], 7, "dcgm,nvml,oops"])
def test_an_unknown_source_is_named_not_ignored(bad):
    """A typo would otherwise read as a source that simply had no data, which is
    indistinguishable from working."""
    with pytest.raises(JobscopeError):
        source.parse_preference(bad)


# --- resolution -------------------------------------------------------------

def test_the_default_order_reproduces_the_old_hardcoded_jobstats_list():
    """What JOBSTATS_BACKED_KEYS = ("duty", "mem", "memtot") used to say, by column."""
    dcgm.set_preference(source.DEFAULT_PREFERENCE)
    assert dcgm.RESOLVED.from_jobstats == {"GPU%", "GMEM_GB", "GMEM_TOTAL_GB"}
    assert dcgm.DCGM_HEADERS == ["SM_ACT%", "TENSOR%", "DRAM%", "POWER_W"]


def test_exactly_one_provider_wins_each_column():
    for spec in (dcgm.ALL_SPECS):
        assert len([s for s in dcgm.ALL_SPECS if s.column == spec.column]) == 1


def test_naming_an_exporter_takes_the_column_off_jobstats():
    """The point of the flag: on a finished job it has to actually change where GPU%
    comes from, rather than being quietly overridden by the stored value."""
    dcgm.set_preference(source.parse_preference("dcgm"))
    assert "GPU%" not in dcgm.RESOLVED.from_jobstats
    assert dcgm.RESOLVED.source_of("GPU%") == "dcgm"


def test_a_column_only_one_source_publishes_is_unaffected_by_the_order():
    """SM_ACT% has no nvml candidate, so asking for nvml cannot take it away."""
    dcgm.set_preference(source.parse_preference("nvml"))
    assert dcgm.RESOLVED.source_of("SM_ACT%") == "dcgm"
    dcgm.set_preference(source.parse_preference("dcgm"))
    assert dcgm.RESOLVED.source_of("SM_ACT%") == "dcgm"


def test_without_a_jobstats_summary_the_column_falls_through_to_an_exporter():
    """A running job, or a finished one with no JS1:, must get the exporter answer
    rather than a no-data column."""
    resolution = source.resolve(dcgm.METRICS, source.DEFAULT_PREFERENCE, jobstats_columns=())
    assert resolution.from_jobstats == frozenset()
    assert resolution.source_of("GPU%") in ("dcgm", "nvml")


def test_the_leading_exporter_skips_jobstats():
    """Per-source default views are per *exporter*: the jobstats summary has no catalog to take a
    default set from, serving three columns and nothing else."""
    assert source.resolve(dcgm.METRICS, ("jobstats", "dcgm", "nvml")).leading_exporter() == "dcgm"
    assert source.resolve(dcgm.METRICS, ("jobstats", "nvml", "dcgm")).leading_exporter() == "nvml"


# --- per-source default views ----------------------------------------------

def test_each_source_has_its_own_default_view():
    """nvml publishes no profiling metrics, so leading with it must not leave a summary
    asking for four columns it would render as "-"."""
    dcgm.set_preference(source.parse_preference("dcgm"))
    assert [s.column for s in dcgm.default_view("summary")] == [
        "GPU%", "SM_ACT%", "TENSOR%", "DRAM%", "POWER_W"]
    dcgm.set_preference(source.parse_preference("nvml"))
    nvml_view = [s.column for s in dcgm.default_view("summary")]
    assert "SM_ACT%" not in nvml_view and "GPU%" in nvml_view
    assert "GMEM_GB" in nvml_view


def test_all_metrics_is_scoped_to_the_active_source():
    """--all-metrics means everything *this* source publishes, not the other's."""
    dcgm.set_preference(source.parse_preference("nvml"))
    families = {s.family for s in dcgm.default_view("extended")}
    assert families == {"nvml"}


# --- the mixed set stays describable ---------------------------------------

def test_provenance_groups_columns_by_source_in_preference_order():
    dcgm.set_preference(source.DEFAULT_PREFERENCE)
    groups = dcgm.RESOLVED.by_source()
    assert [name for name, _ in groups] == ["jobstats", "dcgm"]
    jobstats_columns = dict(groups)["jobstats"]
    # Catalog order, not set order -- from_jobstats is a frozenset.
    assert jobstats_columns == ("GPU%", "GMEM_GB", "GMEM_TOTAL_GB")


def test_the_running_view_does_not_credit_a_jobstats_summary_it_cannot_have():
    """Slurm writes the jobstats summary at job *end*, so a running job's GPU% was measured by an
    exporter whatever the preference says. Naming the jobstats summary there would credit a source
    that had nothing to give -- the same error as claiming a window not scanned."""
    from jobscope.report import gpu_source_line
    dcgm.set_preference(source.DEFAULT_PREFERENCE)
    specs = dcgm.DEFAULT_SPECS
    assert "jobstats" in gpu_source_line(specs, have_jobstats=True)
    assert "jobstats" not in gpu_source_line(specs, have_jobstats=False)


def test_no_gpu_specs_means_no_provenance_line():
    """--cpu prints no GPU column, so there is nothing to state a source for."""
    from jobscope.report import gpu_source_line, source_pair
    assert gpu_source_line(None) == ""
    assert source_pair(None) == []


# --- the host axis ----------------------------------------------------------

def test_host_columns_have_their_own_axis():
    """CPU%/MEM% choose between the summary and the cgroup exporter, not between dcgm and
    nvml -- so naming a GPU source for them is an error rather than a no-op."""
    from jobscope import cpu
    assert source.parse_preference("cgroup", "[host] source",
                                   source.HOST_SOURCES) == ("cgroup", "jobstats", "slurm")
    with pytest.raises(JobscopeError):
        source.parse_preference("dcgm", "[host] source", source.HOST_SOURCES)
    assert cpu.RESOLVED.source_of("CPU%") == "jobstats"


def test_the_host_preference_moves_cpu_and_mem():
    from jobscope import cpu
    cpu.set_preference(("cgroup", "jobstats"))
    try:
        assert cpu.RESOLVED.source_of("CPU%") == "cgroup"
        assert cpu.RESOLVED.from_jobstats == frozenset()
    finally:
        cpu.set_preference(source.DEFAULT_HOST_PREFERENCE)


def test_the_source_line_names_the_host_columns_first():
    """They print first in the table, so they read first here too."""
    from jobscope import cpu
    from jobscope.report import gpu_source_line
    line = gpu_source_line(dcgm.DEFAULT_SPECS, host_specs=cpu.DEFAULT_CGROUP_SPECS)
    assert line.index("CPU%") < line.index("GPU%")
    assert "MEM%" in line


def test_a_name_claimed_by_both_catalogs_is_rejected_not_guessed():
    """`mem` is the DCGM key for GMEM_GB and the cgroup key for MEM%. Picking a side
    would be a coin toss that reads as working."""
    from jobscope import config as config_module
    from jobscope import cpu
    assert dcgm.spec_named("mem").header == "GMEM_GB"
    assert cpu.spec_named("mem").header == "MEM%"
    with pytest.raises(JobscopeError) as exc:
        config_module._metrics({"summary": ["cpu", "mem"]})
    assert "gmem_gb" in str(exc.value) and "mem%" in str(exc.value)


# --- slurm as a host source -------------------------------------------------

def _slurm_record(state, duration=3480):
    from jobscope.slurm import JobRecord
    return JobRecord(jobid="1", state=state, name="t", runtime="00:58:00", nodes="1",
                     gpus=1, stats={}, start=0, end=duration, duration=duration,
                     jobid_raw="1", cluster="", user="u")


def _slurm_metrics(total_cpu_s, duration=3480, cores=16):
    from jobscope.extra_metric import SlurmMetrics
    return SlurmMetrics(jobid="1", gpus=1, total_cpu_s=total_cpu_s,
                        cpu_time_s=duration * cores,
                        used_mem_bytes=8e9, req_mem_bytes=16e9)


@pytest.mark.parametrize("state", ["RUNNING", "COMPLETED"])
def test_zero_cpu_seconds_is_not_gathered_rather_than_idle(state):
    """A process that ran at all burns some CPU, so a literal zero means jobacct is not
    recording it. Rendering 0% would say "idle", which is the one thing jobscope must
    never say about a number it does not have. Measured on this cluster: TotalCPU=0 on a
    finished 128-core job whose summary reports CPU% 6."""
    from jobscope.job_ave_stats import accounted
    assert accounted(_slurm_metrics(0.0), _slurm_record(state)) is False
    assert accounted(_slurm_metrics(30000.0), _slurm_record(state)) is True


def test_slurm_figures_are_job_totals_under_one_entry():
    """sacct accounts CPU-seconds and memory per *job*, so there is no per-node split to
    reproduce -- and the entry is not named after a host, because labelling job totals
    with a hostname would claim a measurement Slurm did not make."""
    from jobscope.job_ave_stats import SLURM_NODE, slurm_host_map
    from jobscope.jobstats import jobstats_metrics
    nodes = slurm_host_map(_slurm_metrics(30000.0), 3480)
    assert list(nodes) == [SLURM_NODE]
    # 30000 cpu-seconds over 3480s x 16 cores = 53.9%, and 8/16 GiB = 50%.
    assert jobstats_metrics({"total_time": 3480, "nodes": nodes}, 1).known() == {
        "CPU%": 54, "MEM%": 50}


def test_naming_slurm_replaces_the_jobstats_summary_rather_than_filling_gaps():
    """Naming a source has to mean the numbers come from it, as --gpu-source dcgm does.
    Falling back to the jobstats summary would leave the header crediting slurm for jobstats figures."""
    from jobscope.job_ave_stats import apply_slurm_host
    record = _slurm_record("COMPLETED")
    record.stats = {"total_time": 3480,
                    "nodes": {"node01": {"total_time": 999.0, "cpus": 8,
                                         "used_memory": 1e9, "total_memory": 2e9,
                                         "gpu_utilization": {"0": 90}}}}
    # Slurm has nothing to give: the summary's host fields go anyway, and the GPU map stays.
    apply_slurm_host({"1": record}, ["1"], None, override=True)
    node = record.stats["nodes"]["node01"]
    assert "total_time" not in node and "cpus" not in node
    assert node["gpu_utilization"] == {"0": 90}


def test_a_blank_running_cpu_says_why(capsys):
    """The failure this was reported as: GPU columns full, CPU columns dashes, and
    nothing connecting that to an exporter. A running job has no jobstats summary to fall back on,
    so the note names the one source that could have served it."""
    from jobscope.job_ave_stats import note_missing_host_series
    running, finished = _slurm_record("RUNNING"), _slurm_record("COMPLETED")
    note_missing_host_series({"1": running}, ["1"])
    assert "no cgroup_* series covers them" in capsys.readouterr().err
    # Not for a finished job -- its jobstats summary supplies them -- nor when the fields arrived.
    note_missing_host_series({"1": finished}, ["1"])
    running.stats = {"total_time": 60, "nodes": {"n1": {"total_time": 30.0, "cpus": 1}}}
    note_missing_host_series({"1": running}, ["1"])
    assert capsys.readouterr().err == ""


def test_every_resolved_spec_agrees_with_its_uuid_label():
    """A spec claiming one family while querying the other returns rows that cannot be
    attributed to a card -- so MetricSpec checks the pair. Assert it stays checked."""
    from jobscope.dcgm import MetricSpec
    with pytest.raises(ValueError):
        MetricSpec("x", "X%", "nvidia_gpu_x", 1, 0, "all", family="nvml")
