"""The pre-action check's arithmetic, without a terminal in sight.

Everything here is a pure function over ``(stamp, value)`` pairs and a ``Thresholds``.
That is the point of it living in job_eff rather than in the renderer: the figures a
verdict rests on can be checked as numbers, and the one rule they all share -- that a
duration is *measured* time and never the wall clock -- is checkable in one place.
"""

from jobscope import job_eff
from jobscope.config import TIER_NAMES, Thresholds

STEP = 60


def pairs(values, step=STEP, start=1_000_000):
    """``(stamp, value)`` pairs, oldest first. None leaves the stamp out entirely."""
    return [(start + i * step, v) for i, v in enumerate(values) if v is not None]


# --- bucketing ----------------------------------------------------------------

def test_bucket_edges_tile_the_span_with_no_hole_and_no_overlap():
    edges = job_eff.bucket_edges(0, 100, 4)
    assert len(edges) == 4
    assert edges[0][0] == 0 and edges[-1][1] == 100
    for (_lo, hi), (next_lo, _next_hi) in zip(edges, edges[1:]):
        assert hi == next_lo


def test_a_bucket_nothing_was_measured_in_is_none_not_zero():
    """The distinction the whole view rests on: an exporter outage and an idle card are
    not the same event, and a display that draws them alike says they are."""
    # Samples only in the first half of the span.
    got = job_eff.bucket(pairs([5] * 10), 1_000_000, 1_000_000 + 1200, 4)
    assert got[0] == 5 and got[1] == 5
    assert got[2] is None and got[3] is None
    # A measured zero is a number, not a hole.
    zeros = job_eff.bucket(pairs([0] * 20), 1_000_000, 1_000_000 + 1200, 4)
    assert all(v == 0 for v in zeros)


def test_bucket_returns_one_value_per_bucket_whatever_the_sample_count():
    for n in (1, 3, 40, 200):
        assert len(job_eff.bucket(pairs([5] * 20), 1_000_000, 1_000_000 + 1200, n)) == n


def test_the_last_bucket_includes_the_closing_stamp():
    """Half-open everywhere else, closed at the end, or the newest sample -- the one a
    reader looks at first -- falls off the strip."""
    start, end = 1_000_000, 1_000_060
    assert job_eff.bucket([(end, 9.0)], start, end, 2)[-1] == 9.0


def test_bucket_takes_a_reducer_so_a_peak_survives_a_coarse_cell():
    stamped = pairs([0, 100, 0, 100])
    start, end = 1_000_000, 1_000_180
    assert job_eff.bucket(stamped, start, end, 1) == [50.0]
    assert job_eff.bucket(stamped, start, end, 1, reduce=max) == [100.0]


# --- the fixed axis -----------------------------------------------------------

def test_full_scale_is_fixed_rather_than_the_data_s_own_range():
    """A flat-idle series scaled to its own min..max draws as a full-height sawtooth,
    which is the one reading this view exists to make impossible."""
    t = Thresholds()
    assert job_eff.full_scale(t, "GPU%") == 100.0
    assert job_eff.full_scale(t, "CPU%") == 100.0
    # Watts have no percentage to be a fraction of, so the idle floor sits mid-axis.
    assert job_eff.full_scale(t, "POWER_W") == 2 * t.floor_of("POWER_W")
    # Nothing to draw against.
    assert job_eff.full_scale(t, "MEM%") is None
    assert job_eff.full_scale(t, "TEMP_C") is None


# --- measured time ------------------------------------------------------------

def test_idle_and_active_sum_to_what_was_measured_not_to_the_window():
    """below_share's own scenario, in durations: 50 idle of 100 measured, over a window
    of 150 scrapes. The missing 50 are their own figure and are in neither of the others,
    because folding them into idle reports an exporter's silence as idleness."""
    stamped = pairs(([1] * 50) + ([90] * 50))
    got = job_eff.measured_time(stamped, 2.0, STEP, expected=150)
    assert got.idle == 50 * STEP
    assert got.active == 50 * STEP
    assert got.missing == 50 * STEP
    assert got.idle + got.active == 100 * STEP != 150 * STEP


def test_missing_is_zero_without_something_to_compare_against():
    """A statement about what is known, not a claim that nothing is missing."""
    assert job_eff.measured_time(pairs([1] * 10), 2.0, STEP).missing == 0


def test_measured_time_has_no_answer_without_a_cutoff():
    assert job_eff.measured_time(pairs([1] * 10), None, STEP) is None
    assert job_eff.measured_time([], 2.0, STEP) is None


# --- when every unit was idle at once -----------------------------------------

def test_gpus_idle_at_different_times_were_never_idle_together():
    """Asked as `last_all_idle - first_all_idle` this would report the whole span. The
    job had a card working at every instant."""
    first = job_eff.idle_stamps(pairs(([1] * 50) + ([90] * 50)), 2.0)
    second = job_eff.idle_stamps(pairs(([90] * 50) + ([1] * 50)), 2.0)
    assert job_eff.concurrent_idle([first, second], STEP) == 0


def test_concurrent_idle_is_the_instants_they_shared():
    both = job_eff.idle_stamps(pairs(([90] * 40) + ([1] * 60)), 2.0)
    one = job_eff.idle_stamps(pairs(([90] * 70) + ([1] * 30)), 2.0)
    assert job_eff.concurrent_idle([both, one], STEP) == 30 * STEP


def test_a_unit_with_no_sample_at_an_instant_is_not_assumed_idle_through_it():
    idle = job_eff.idle_stamps(pairs([1] * 100), 2.0)
    # The same card, its exporter answering only every other scrape.
    patchy = job_eff.idle_stamps(pairs([1, None] * 50), 2.0)
    assert job_eff.concurrent_idle([idle, patchy], STEP) == 50 * STEP


# --- the transition -----------------------------------------------------------

def test_sustain_is_five_minutes_however_coarse_the_scrape():
    assert job_eff.sustain_samples(60) == 5
    assert job_eff.sustain_samples(30) == 10
    # Never one sample: a single reading cannot establish a state.
    assert job_eff.sustain_samples(600) == 3


def test_a_job_that_stopped_an_hour_ago_names_when():
    stamped = pairs(([90] * 150) + ([1] * 60))
    got = job_eff.went_idle(stamped, 2.0, STEP)
    assert got.seconds == 60 * STEP
    assert got.at == stamped[150][0]


def test_one_dip_between_batches_is_not_a_transition():
    """The figure longest_idle's docstring calls a job between batches."""
    assert job_eff.went_idle(pairs(([90] * 100) + ([1] * 3)), 2.0, STEP) is None


def test_a_job_still_working_has_not_gone_idle():
    assert job_eff.went_idle(pairs(([1] * 60) + ([90] * 150)), 2.0, STEP) is None


def test_a_series_that_never_ran_reports_no_transition():
    """flat-idle. "Went idle at" would describe a job that once worked."""
    assert job_eff.went_idle(pairs([1] * 200), 2.0, STEP) is None


def test_a_bursty_job_that_flatlined_still_reports_when_it_stopped():
    """The work before the tail is counted, not required to be consecutive. A job
    alternating 100 and 0 every scrape never has two adjacent samples above the cutoff,
    so a run-length test finds nothing to have stopped and calls a card that died two
    hours ago still working."""
    got = job_eff.went_idle(pairs(([100, 0] * 60) + ([0.2] * 120)), 2.0, STEP)
    assert got is not None and got.seconds == 121 * STEP


def test_one_spike_is_not_work_that_later_stopped():
    """What the consecutive test was reaching for, kept by counting to the same bar."""
    assert job_eff.went_idle(pairs([100] + [0.2] * 200), 2.0, STEP) is None


def test_a_gap_ends_the_idle_run_rather_than_extending_it():
    stamped = pairs([90] * 100) + pairs([1] * 10, start=1_000_000 + 100 * STEP + 3600)
    got = job_eff.went_idle(stamped, 2.0, STEP)
    assert got is not None and got.seconds == 10 * STEP


# --- the distribution ---------------------------------------------------------

def test_the_bins_are_the_tiers_and_all_of_them_are_present():
    got = job_eff.distribution([50.0] * 10, Thresholds(), "GPU%")
    assert [name for name, _count in got] == list(TIER_NAMES)
    assert sum(count for _name, count in got) == 10


def test_every_bin_agrees_with_the_tier_the_verdict_is_taken_through():
    """The "the histogram cannot disagree with the conclusion under it" claim, asserted
    rather than promised."""
    t = Thresholds()
    values = [0.0, 1.5, 2.0, 8.0, 12.0, 25.0, 45.0, 99.0]
    by_tier = {name: 0 for name in TIER_NAMES}
    for value in values:
        by_tier[t.tier("GPU%", value)] += 1
    assert job_eff.distribution(values, t, "GPU%") == [(n, by_tier[n]) for n in TIER_NAMES]


def test_a_floor_graded_metric_has_two_bins_not_five():
    """Watts have no tiers to bin by -- only a floor, per GPU model."""
    got = job_eff.distribution([80.0, 120.0, 500.0], Thresholds(), "POWER_W")
    assert got == [("below", 1), ("at or above", 2)]


def test_a_metric_with_no_cutoff_has_nothing_for_a_bin_to_mean():
    assert job_eff.distribution([50.0] * 10, Thresholds(), "MEM%") == []
    assert job_eff.distribution([], Thresholds(), "GPU%") == []


# --- the guards ---------------------------------------------------------------

def test_a_window_too_short_to_grade_says_so_and_suppresses():
    got = job_eff.qualifiers(n=6, expected=6, low=0.0, peak=97.0, swing=None,
                             pairs=5, limit=2.0)
    assert job_eff.THIN in got
    assert set(got) & job_eff.SUPPRESSING


def test_half_a_window_measured_is_not_a_description_of_the_window():
    got = job_eff.qualifiers(n=100, expected=241, low=0.0, peak=90.0, swing=1.0,
                             pairs=99, limit=2.0)
    assert job_eff.SPARSE in got and set(got) & job_eff.SUPPRESSING


def test_a_few_dropped_scrapes_qualify_rather_than_suppress():
    got = job_eff.qualifiers(n=200, expected=241, low=0.0, peak=90.0, swing=1.0,
                             pairs=199, limit=2.0)
    assert job_eff.GAPPY in got
    assert not set(got) & job_eff.SUPPRESSING


def test_a_full_window_raises_no_coverage_caveat():
    got = job_eff.qualifiers(n=241, expected=241, low=0.0, peak=90.0, swing=1.0,
                             pairs=240, limit=2.0)
    assert job_eff.GAPPY not in got and job_eff.SPARSE not in got


def test_nothing_heard_lately_is_caught_even_at_full_coverage():
    """The case coverage alone misses: the measured half looks busy and the tail is
    silence. Reported as idle, that is a verdict about a job nobody has heard from."""
    got = job_eff.qualifiers(n=120, expected=120, low=0.0, peak=90.0, swing=1.0,
                             pairs=119, limit=2.0, newest_age=7200, step=STEP)
    assert job_eff.STALE in got


def test_a_signal_faster_than_its_sampler_has_no_reproducible_mean():
    """The measured case: swing 67 over a 0-100 range."""
    assert job_eff.aliased(0.0, 100.0, 67.0, pairs=200) is True


def test_ordinary_training_with_a_periodic_sync_is_not_aliased():
    """20-100 in 20-point steps. A guard that fires here teaches readers to skip guards."""
    assert job_eff.aliased(20.0, 100.0, 20.0, pairs=200) is False


def test_a_signal_confined_to_one_band_is_not_judged_on_its_swing():
    """Inside a single band a mean is reproducible to within that band whatever the
    swing -- and the ratio is 0/0 on a constant series."""
    assert job_eff.aliased(90.0, 95.0, 4.0, pairs=200) is False
    assert job_eff.aliased(50.0, 50.0, 0.0, pairs=200) is False


def test_too_few_jumps_to_take_a_median_of():
    assert job_eff.aliased(0.0, 100.0, 67.0, pairs=3) is False


def test_a_low_mean_on_a_series_that_peaked_is_flagged_as_having_run():
    """wasteful and idle are not the same finding, and only one of them licenses a kill."""
    assert job_eff.PEAKED in job_eff.qualifiers(
        n=200, expected=200, low=0.0, peak=100.0, swing=1.0, pairs=199, limit=2.0)
    assert job_eff.PEAKED not in job_eff.qualifiers(
        n=200, expected=200, low=0.0, peak=0.5, swing=0.1, pairs=199, limit=2.0)


def test_a_clean_steady_series_raises_nothing():
    assert job_eff.qualifiers(n=200, expected=200, low=60.0, peak=62.0, swing=0.5,
                              pairs=199, limit=2.0, newest_age=0) == [job_eff.PEAKED]
