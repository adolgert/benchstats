"""Step 7: bootstrap confidence intervals.

The coverage tests are the central tests of the project. Each coverage test
is a rate test. Where the method should cover, the test tolerates a coverage
just under the nominal 95% and must catch a lower one. Negative controls show
the tests can fail: they claim a coverage well below 95% and must catch 90%.
"""

import numpy as np
import pytest
from scipy import stats

from benchstats import resampling
from support import synthetic
from support.design import ALPHA, BETA, check_rate, n_for_rate


def median(x, axis=-1):
    return np.median(x, axis=axis)


# ---------------------------------------------------------------- tier 1


def test_bootstrap_ci_matches_scipy(rng):
    x = rng.lognormal(size=60)
    seed = int(rng.integers(2**32))
    ours = resampling.bootstrap_ci((x,), median, n_resamples=999, rng=np.random.default_rng(seed))
    theirs = stats.bootstrap((x,), median, n_resamples=999, rng=np.random.default_rng(seed))
    assert ours.low == theirs.confidence_interval.low
    assert ours.high == theirs.confidence_interval.high
    assert ours.estimate == np.median(x)
    assert ours.mc_se_low > 0 and ours.mc_se_high > 0


def test_bootstrap_ci_takes_two_sample_statistics(rng):
    a, b = rng.lognormal(size=50), 2 * rng.lognormal(size=60)

    def log_ratio(x, y, axis=-1):
        return np.log(np.median(y, axis=axis) / np.median(x, axis=axis))

    ci = resampling.bootstrap_ci((a, b), log_ratio, n_resamples=999, rng=rng)
    assert ci.estimate == pytest.approx(log_ratio(a, b))
    assert ci.low < ci.estimate < ci.high


def test_bootstrap_ci_accepts_statistics_without_axis(rng):
    x = rng.lognormal(size=30)
    ci = resampling.bootstrap_ci((x,), lambda s: float(np.median(s)), n_resamples=200, method="percentile", rng=rng)
    assert ci.low <= np.median(x) <= ci.high


def test_bootstrap_ci_rejects_bad_input(rng):
    with pytest.raises(ValueError, match="tuple"):
        resampling.bootstrap_ci(np.ones(5), median, rng=rng)
    with pytest.raises(ValueError, match="finite"):
        resampling.bootstrap_ci((np.array([1.0, np.nan, 2.0]),), median, rng=rng)
    with pytest.raises(ValueError, match="confidence"):
        resampling.bootstrap_ci((np.ones(5),), median, confidence=1.5, rng=rng)
    with pytest.raises(ValueError, match="Generator"):
        resampling.bootstrap_ci((np.ones(5),), median, rng=1)


def test_stationary_indices_extremes(rng):
    # Block length 1: every position starts a new block.
    idx = resampling.stationary_indices(50, 1.0, 20, rng)
    assert idx.shape == (20, 50) and idx.min() >= 0 and idx.max() < 50
    # A very long block length: one block per resample, wrapping around.
    idx = resampling.stationary_indices(50, 1e12, 20, rng, size=80)
    expected = (idx[:, :1] + np.arange(80)) % 50
    np.testing.assert_array_equal(idx, expected)


def test_stationary_indices_runs_continue_blocks(rng):
    idx = resampling.stationary_indices(1000, 10.0, 50, rng)
    continues = np.diff(idx, axis=1) % 1000 == 1
    # Each step continues the block with probability 0.9, apart from random restarts that land next door.
    assert 0.85 < continues.mean() < 0.95


def test_block_length_matches_arch(rng):
    arch = pytest.importorskip("arch.bootstrap")
    for phi in (0.0, 0.5, 0.9):
        x = synthetic.correlated(synthetic.GPDTail(0.0, 1.0, 1.0), phi, 500, rng)
        expected = max(1.0, float(arch.optimal_block_length(np.log(x))["stationary"].iloc[0]))
        assert resampling.block_length(np.log(x)) == pytest.approx(expected, rel=1e-10)


def test_block_bootstrap_ci_is_percentile_of_resamples(rng):
    x = rng.lognormal(size=200)
    ci = resampling.block_bootstrap_ci((x,), median, block_length=5.0, n_resamples=999, rng=rng)
    assert ci.method == "stationary percentile"
    assert ci.low < np.median(x) < ci.high


def _grouped(sizes):
    values = np.concatenate([np.full(size, float(g + 1)) for g, size in enumerate(sizes)])
    labels = np.concatenate([np.full(size, g) for g, size in enumerate(sizes)])
    return values, labels


@pytest.mark.parametrize("sizes", [(4, 4, 4, 4), (2, 5, 3)])
def test_hierarchical_resamples_whole_groups(rng, sizes):
    values, labels = _grouped(sizes)
    seen = []

    def check(x):
        counts = {v: int((x == v).sum()) for v in np.unique(x)}
        for v, count in counts.items():
            assert count % sizes[int(v) - 1] == 0  # whole groups only
        seen.append(len(x))
        return float(np.mean(x))

    resampling.hierarchical_bootstrap_ci((values,), (labels,), check, resample_within=None, n_resamples=50, rng=rng)
    assert len(seen) >= 50


def test_hierarchical_resamples_within_groups(rng):
    values = np.arange(1.0, 13.0)
    labels = np.repeat([0, 1, 2], 4)
    resampled = []

    def keep(x, axis=-1):
        resampled.append(np.array(x))
        return np.mean(x, axis=axis)

    resampling.hierarchical_bootstrap_ci((values,), (labels,), keep, resample_within="iid", n_resamples=200, rng=rng)
    rows = np.concatenate(resampled[1:])  # the first call is the estimate
    for row in rows:
        blocks = row.reshape(3, 4)
        for block in blocks:  # every value in a block comes from one group
            assert len({int((v - 1) // 4) for v in block}) == 1


def test_hierarchical_needs_two_groups(rng):
    with pytest.raises(ValueError, match="two groups"):
        resampling.hierarchical_bootstrap_ci((np.ones(5),), (np.zeros(5, int),), median, rng=rng)


# ---------------------------------------------------------------- tier 2: coverage

FLOOR = synthetic.FloorPlusDetours(1.0, 0.05, (0.1, 0.02), (0.2, 1.0))
N_RESAMPLES = 1999  # resamples per interval in coverage tests

COVERS_CATCH = 0.90  # every "covers" test must catch a coverage of 90%
NEGATIVE_CATCH = 0.90  # every negative control must catch a coverage of 90%


def _covers(ci, truth):
    return ci.low <= truth <= ci.high


def independent_coverage(dist, n, n_datasets, rng):
    truth = dist.median()
    return np.array(
        [
            _covers(resampling.bootstrap_ci((dist.sample(n, rng),), median, n_resamples=N_RESAMPLES, rng=rng), truth)
            for _ in range(n_datasets)
        ],
        float,
    )


# Pilot coverages (scripts/pilot.py test_resampling, 1,000 datasets) are in
# comments; tolerated levels sit a few standard errors below them.
INDEPENDENT_CASES = {
    "gpd_tail xi=0.3": (synthetic.GPDTail(0.3, 1.0, 1.0), 100, 0.925),  # pilot 0.944
    "floor_plus_detours": (FLOOR, 100, 0.925),  # pilot 0.942
}


@pytest.mark.statistical
@pytest.mark.parametrize("name", list(INDEPENDENT_CASES))
def test_bca_interval_for_median_covers(name, rng):
    dist, n, tolerated = INDEPENDENT_CASES[name]
    m = n_for_rate(tolerated, COVERS_CATCH, ALPHA, BETA)
    covered = int(independent_coverage(dist, n, m, rng).sum())
    result = check_rate(covered, m, tolerated, ALPHA, "lower")
    assert result.passed, result


CORRELATED_PHI = 0.7
CORRELATED_N = 1000
BURSTY = synthetic.BurstyModel(
    quiet=synthetic.FloorPlusDetours(1.0, 0.05, (0.02,), (0.2,)),
    noisy=synthetic.FloorPlusDetours(1.0, 0.05, (0.8,), (0.2,)),
    stay_quiet=0.99,
    stay_noisy=0.95,
)


def _dependent_series(name, rng):
    if name == "correlated":
        return synthetic.correlated(FLOOR, CORRELATED_PHI, CORRELATED_N, rng), FLOOR.median()
    return synthetic.bursty(BURSTY, CORRELATED_N, rng), BURSTY.marginal.median()


def dependent_coverage(name, method, n_datasets, rng):
    covered = np.empty(n_datasets)
    for i in range(n_datasets):
        x, truth = _dependent_series(name, rng)
        if method == "independent":
            ci = resampling.bootstrap_ci((x,), median, n_resamples=N_RESAMPLES, rng=rng)
        else:
            length = resampling.block_length(np.log(x))
            ci = resampling.block_bootstrap_ci((x,), median, block_length=length, n_resamples=N_RESAMPLES, rng=rng)
        covered[i] = _covers(ci, truth)
    return covered


# Negative controls: the independent bootstrap ignores dependence and should
# under-cover. Pilot coverages: correlated 0.675 (1,000 datasets), bursty
# 0.645 (200 datasets).
NEGATIVE_DEPENDENT = {"correlated": 0.75, "bursty": 0.75}


@pytest.mark.statistical
@pytest.mark.parametrize("name", list(NEGATIVE_DEPENDENT))
def test_independent_bootstrap_undercovers_dependent_draws(name, rng):
    tolerated = NEGATIVE_DEPENDENT[name]
    m = n_for_rate(tolerated, NEGATIVE_CATCH, ALPHA, BETA)
    covered = int(dependent_coverage(name, "independent", m, rng).sum())
    result = check_rate(covered, m, tolerated, ALPHA, "upper")
    assert result.passed, result


# The block bootstrap under-covers a little in small samples. Pilot coverage
# on the correlated data: 0.916. Tolerate 0.89 and catch 0.84.
BLOCK_TOLERATED, BLOCK_CATCH = 0.89, 0.84


@pytest.mark.statistical
def test_block_bootstrap_covers_correlated_draws(rng):
    m = n_for_rate(BLOCK_TOLERATED, BLOCK_CATCH, ALPHA, BETA)
    covered = int(dependent_coverage("correlated", "blocks", m, rng).sum())
    result = check_rate(covered, m, BLOCK_TOLERATED, ALPHA, "lower")
    assert result.passed, result


INVOCATION_SD = 0.02
ITERATIONS = 20


def hierarchical_coverage(n_invocations, method, n_datasets, rng):
    truth = synthetic.hierarchical_median(FLOOR, INVOCATION_SD)
    covered = np.empty(n_datasets)
    for i in range(n_datasets):
        draws = synthetic.hierarchical(
            FLOOR,
            n_builds=1,
            n_invocations=n_invocations,
            n_iterations=ITERATIONS,
            build_sd=0.0,
            invocation_sd=INVOCATION_SD,
            rng=rng,
        )
        if method == "flat":
            ci = resampling.bootstrap_ci((draws.time,), median, n_resamples=N_RESAMPLES, rng=rng)
        else:
            within = None if method == "hierarchical-whole" else "iid"
            ci = resampling.hierarchical_bootstrap_ci(
                (draws.time,), (draws.invocation,), median, resample_within=within, n_resamples=N_RESAMPLES, rng=rng
            )
        covered[i] = _covers(ci, truth)
    return covered


# Pilot coverages (1,000 datasets) for 10 and 30 invocations of 20 draws:
# resampling invocations and then draws within them ("iid", the default)
# 0.930 and 0.942; resampling whole invocations only 0.904 and 0.935. With
# 10 invocations the test tolerates 0.90 and must catch 0.85; with 30 it
# tolerates 0.92 and must catch 0.87.
HIERARCHICAL_CASES = {10: (0.90, 0.85), 30: (0.92, 0.87)}
# The flat bootstrap ignores the invocations. Pilot coverage 0.51 for both.
NEGATIVE_HIERARCHICAL = 0.60


@pytest.mark.statistical
@pytest.mark.parametrize("n_invocations", list(HIERARCHICAL_CASES))
def test_hierarchical_bootstrap_covers(n_invocations, rng):
    tolerated, catch = HIERARCHICAL_CASES[n_invocations]
    m = n_for_rate(tolerated, catch, ALPHA, BETA)
    covered = int(hierarchical_coverage(n_invocations, "hierarchical", m, rng).sum())
    result = check_rate(covered, m, tolerated, ALPHA, "lower")
    assert result.passed, result


@pytest.mark.statistical
@pytest.mark.parametrize("n_invocations", list(HIERARCHICAL_CASES))
def test_flat_bootstrap_undercovers_hierarchical_draws(n_invocations, rng):
    m = n_for_rate(NEGATIVE_HIERARCHICAL, NEGATIVE_CATCH, ALPHA, BETA)
    covered = int(hierarchical_coverage(n_invocations, "flat", m, rng).sum())
    result = check_rate(covered, m, NEGATIVE_HIERARCHICAL, ALPHA, "upper")
    assert result.passed, result


# ---------------------------------------------------------------- tier 2: Monte Carlo error

# Endpoint Monte Carlo error at 9,999 resamples, as a fraction of the
# half-width: pilot median 0.017, 99th percentile 0.047, maximum below 0.08
# (400 datasets). The bound is 8%; the rate of meeting it must be at least
# 0.99 and the test must catch 0.90.
MC_FRACTION = 0.08
MC_P0, MC_P1 = 0.99, 0.90  # tolerated and must-catch rates of meeting that bound


def mc_ratio(n_datasets, rng):
    ratios = np.empty(n_datasets)
    for i in range(n_datasets):
        ci = resampling.bootstrap_ci((FLOOR.sample(100, rng),), median, rng=rng)  # default n_resamples
        half_width = (ci.high - ci.low) / 2
        ratios[i] = max(ci.mc_se_low, ci.mc_se_high) / half_width
    return ratios


@pytest.mark.statistical
def test_endpoint_monte_carlo_error_is_small_at_default_resamples(rng):
    m = n_for_rate(MC_P0, MC_P1, ALPHA, BETA)
    met = int((mc_ratio(m, rng) < MC_FRACTION).sum())
    result = check_rate(met, m, MC_P0, ALPHA, "lower")
    assert result.passed, result


PILOTS = {
    **{
        f"coverage {name}": (lambda m, rng, name=name: independent_coverage(INDEPENDENT_CASES[name][0], INDEPENDENT_CASES[name][1], m, rng))
        for name in INDEPENDENT_CASES
    },
    **{
        f"coverage {name} {method}": (lambda m, rng, name=name, method=method: dependent_coverage(name, method, m, rng))
        for name in ("correlated", "bursty")
        for method in ("independent", "blocks")
    },
    **{
        f"coverage hierarchical G={g} {method}": (lambda m, rng, g=g, method=method: hierarchical_coverage(g, method, m, rng))
        for g in HIERARCHICAL_CASES
        for method in ("flat", "hierarchical", "hierarchical-whole")
    },
    "mc ratio (mean is the ratio)": mc_ratio,
}
