"""Step 5: time order: dependence, warmup and drift."""

import math

import numpy as np
import pytest
from scipy import stats

from benchstats import characterization
from support import synthetic
from support.design import ALPHA, BETA, check_mean_error, check_rate, n_for_mean_error, n_for_rate

# ---------------------------------------------------------------- tier 1


def _direct_rank_autocorrelation(x, max_lag):
    r = stats.rankdata(x)
    r = r - r.mean()
    return np.array([np.sum(r[: r.size - k] * r[k:]) / np.sum(r * r) for k in range(1, max_lag + 1)])


def test_rank_autocorrelation_matches_direct_sum(rng):
    x = rng.lognormal(size=200)
    np.testing.assert_allclose(characterization.rank_autocorrelation(x, 10), _direct_rank_autocorrelation(x, 10), atol=1e-12)


def test_rank_autocorrelation_ignores_increasing_transforms(rng):
    x = synthetic.correlated(synthetic.GPDTail(0.3, 1.0, 1.0), 0.6, 300, rng)
    base = characterization.rank_autocorrelation(x, 8)
    for transformed in (np.log(x), x**3, np.exp(x / x.max()), 5 + 2 * x):
        np.testing.assert_allclose(characterization.rank_autocorrelation(transformed, 8), base, atol=1e-12)


def test_integrated_autocorrelation_time_follows_sokal_window(rng):
    x = synthetic.correlated(synthetic.GPDTail(0.0, 1.0, 1.0), 0.7, 2000, rng)
    rho = _direct_rank_autocorrelation(x, 1000)
    tau = 1 + 2 * np.cumsum(rho)
    window = next(w for w in range(1, 1001) if w >= 5 * tau[w - 1])
    assert characterization.integrated_autocorrelation_time(x) == pytest.approx(tau[window - 1])


def test_effective_sample_size_is_at_most_n(rng):
    x = rng.lognormal(size=500)
    tau = characterization.integrated_autocorrelation_time(x)
    ess = characterization.effective_sample_size(x)
    assert ess <= 500
    if tau > 1:
        assert ess == pytest.approx(500 / tau)


def _direct_mser(x, batch):
    means = np.log(x[: (x.size // batch) * batch]).reshape(-1, batch).mean(axis=1)
    values = [np.var(means[d:]) / (means.size - d) for d in range(means.size // 2 + 1)]
    return batch * int(np.argmin(values))


def test_warmup_mser_matches_direct_loop(rng):
    for _ in range(5):
        x = synthetic.add_warmup(rng.lognormal(size=203), np.arange(203), amplitude=rng.uniform(0, 3), decay=10.0)
        assert characterization.warmup_mser(x) == _direct_mser(x, 5)
        assert characterization.warmup_mser(x, batch=3) == _direct_mser(x, 3)


def test_warmup_mser_drops_an_obvious_transient():
    x = 1 + 0.02 * np.sin(1.7 * np.arange(200))
    x[:20] *= 3.0
    assert characterization.warmup_mser(x) == 20


def test_drift_slope_is_exact_on_a_noiseless_trend():
    slope = 2e-5
    x = np.exp(slope * np.arange(500))
    assert characterization.drift_slope(x) == pytest.approx(100 * math.expm1(1000 * slope))


def test_time_order_functions_reject_bad_input():
    with pytest.raises(ValueError, match="max_lag"):
        characterization.rank_autocorrelation([1.0, 2.0, 3.0], 3)
    with pytest.raises(ValueError, match="all be equal"):
        characterization.integrated_autocorrelation_time([1.0] * 10)
    with pytest.raises(ValueError, match="positive"):
        characterization.drift_slope([1.0, -1.0, 2.0])


# ---------------------------------------------------------------- tier 2: dependence

HEAVY = synthetic.GPDTail(0.3, 1.0, 1.0)
LAGS = np.arange(1, 6)
ACF_N = 1000
# Pilot: sd of the error at lags 1..5. The estimator's downward bias is of
# order tau_int / n; tolerated levels are the pilot's |mean| plus three
# standard errors, rounded up. Each must catch 0.03 more.
ACF_PILOT = {
    0.0: {"sd": (0.031, 0.032, 0.030, 0.031, 0.031), "tolerated": 0.005},
    0.5: {"sd": (0.029, 0.035, 0.040, 0.040, 0.040), "tolerated": 0.010},
    0.9: {"sd": (0.017, 0.030, 0.041, 0.049, 0.055), "tolerated": 0.025},
}


def acf_errors(phi, n_datasets, rng):
    truth = synthetic.correlated_rank_autocorrelation(phi, LAGS)
    return np.array(
        [characterization.rank_autocorrelation(synthetic.correlated(HEAVY, phi, ACF_N, rng), LAGS[-1]) - truth for _ in range(n_datasets)]
    )


@pytest.mark.statistical
@pytest.mark.parametrize("phi", list(ACF_PILOT))
def test_rank_autocorrelation_recovers_copula_value(phi, rng):
    alpha = ALPHA / LAGS.size
    tolerated = ACF_PILOT[phi]["tolerated"]
    sizes = [n_for_mean_error(sd, tolerated, tolerated + 0.03, alpha, BETA) for sd in ACF_PILOT[phi]["sd"]]
    errors = acf_errors(phi, max(sizes), rng)
    for k, lag in enumerate(LAGS):
        result = check_mean_error(errors[: sizes[k], k], tolerated, alpha)
        assert result.passed, f"lag {lag}: {result}"


TAU_N = 5000
# Relative error of tau_int against the value implied by the copula. The Sokal
# window truncates the sum and the autocorrelations are biased down, so the
# tolerated level comes from the pilot; each must catch 10% more.
TAU_PILOT = {
    0.0: {"sd": 0.065, "tolerated": 0.010},
    0.5: {"sd": 0.099, "tolerated": 0.020},
    0.9: {"sd": 0.243, "tolerated": 0.030},
}


def tau_errors(phi, n_datasets, rng):
    truth = synthetic.correlated_tau_int(phi)
    return np.array(
        [
            characterization.integrated_autocorrelation_time(synthetic.correlated(HEAVY, phi, TAU_N, rng)) / truth - 1
            for _ in range(n_datasets)
        ]
    )


@pytest.mark.statistical
@pytest.mark.parametrize("phi", list(TAU_PILOT))
def test_integrated_autocorrelation_time_recovers_copula_value(phi, rng):
    pilot = TAU_PILOT[phi]
    m = n_for_mean_error(pilot["sd"], pilot["tolerated"], pilot["tolerated"] + 0.10, ALPHA, BETA)
    result = check_mean_error(tau_errors(phi, m, rng), pilot["tolerated"], ALPHA)
    assert result.passed, result


BURSTY = synthetic.BurstyModel(
    quiet=synthetic.FloorPlusDetours(1.0, 0.05, (0.02,), (0.2,)),
    noisy=synthetic.FloorPlusDetours(1.0, 0.05, (0.5,), (0.2,)),
    stay_quiet=0.98,
    stay_noisy=0.9,
)
BURSTY_N = 1000
DETECTION_LEVEL = 0.05  # level of the detection rule tau_int > null quantile


@pytest.mark.statistical
def test_bursty_dependence_is_detected(rng):
    """Power check: tau_int exceeds its 95% null quantile with probability >= 1 - beta.

    Under independence the ranks are a random permutation, so the null
    distribution of tau_int depends only on n and is simulated exactly.
    """
    null = [characterization.integrated_autocorrelation_time(rng.permutation(BURSTY_N).astype(float)) for _ in range(2000)]
    critical = np.quantile(null, 1 - DETECTION_LEVEL)
    n = n_for_rate(1 - BETA, 0.75, ALPHA, BETA)
    detected = sum(characterization.integrated_autocorrelation_time(synthetic.bursty(BURSTY, BURSTY_N, rng)) > critical for _ in range(n))
    result = check_rate(detected, n, 1 - BETA, ALPHA, "lower")
    assert result.passed, result


# ---------------------------------------------------------------- tier 2: warmup

WARMUP_BASE = synthetic.FloorPlusDetours(1.0, 0.05, (0.1,), (0.2,))
WARMUP_N = 500


def warmup_series(rng, *, early_burst):
    x = synthetic.add_warmup(WARMUP_BASE.sample(WARMUP_N, rng), np.arange(WARMUP_N), amplitude=0.5, decay=20.0)
    if early_burst:
        burst = slice(100, 130)  # a burst of detours early in the steady state
        x[burst] += rng.exponential(0.5, size=30) * (rng.random(30) < 0.6)
    return x


def warmup_errors(early_burst, n_datasets, rng, *, truncate=True):
    truth = WARMUP_BASE.median()
    errors = np.empty(n_datasets)
    for i in range(n_datasets):
        x = warmup_series(rng, early_burst=early_burst)
        d = characterization.warmup_mser(x) if truncate else 0
        errors[i] = np.median(x[d:]) / truth - 1
    return errors


# Relative error of the median after truncation. Pilot: without truncation
# the error is about +0.47% (no burst) and +0.59% (burst), so the test must
# catch a bias of 0.4%. The tolerated level covers the pilot's +0.13% after
# truncation.
WARMUP_PILOT = {
    False: {"sd": 0.0017, "tolerated": 0.002},
    True: {"sd": 0.0015, "tolerated": 0.002},
}
WARMUP_CATCH = 0.004


@pytest.mark.statistical
@pytest.mark.parametrize("early_burst", [False, True], ids=["plain", "early-burst"])
def test_mser_truncation_recovers_steady_state_median(early_burst, rng):
    pilot = WARMUP_PILOT[early_burst]
    m = n_for_mean_error(pilot["sd"], pilot["tolerated"], WARMUP_CATCH, ALPHA, BETA)
    result = check_mean_error(warmup_errors(early_burst, m, rng), pilot["tolerated"], ALPHA)
    assert result.passed, result


# ---------------------------------------------------------------- tier 2: drift

DRIFT_N = 1000
DRIFT_CHANGE = 0.02  # 2% rise over the experiment


def drift_errors(phi, n_datasets, rng):
    errors = np.empty(n_datasets)
    for i in range(n_datasets):
        base = synthetic.correlated(WARMUP_BASE, phi, DRIFT_N, rng) if phi else WARMUP_BASE.sample(DRIFT_N, rng)
        x, slope = synthetic.add_drift(base, np.arange(DRIFT_N), total_change=DRIFT_CHANGE)
        errors[i] = characterization.drift_slope(x) - 100 * math.expm1(1000 * slope)
    return errors


# Errors in percent per 1,000 draws (the true value is about 2.0). The pilot
# shows no bias beyond its standard error (0.006 and 0.014); the tolerated
# level is three standard errors. Must catch 0.2 more.
DRIFT_PILOT = {
    0.0: {"sd": 0.19, "tolerated": 0.02},
    0.7: {"sd": 0.45, "tolerated": 0.05},
}
DRIFT_CATCH_ABOVE_TOLERATED = 0.2


@pytest.mark.statistical
@pytest.mark.parametrize("phi", list(DRIFT_PILOT), ids=["independent", "correlated"])
def test_drift_slope_recovers_trend(phi, rng):
    pilot = DRIFT_PILOT[phi]
    m = n_for_mean_error(pilot["sd"], pilot["tolerated"], pilot["tolerated"] + DRIFT_CATCH_ABOVE_TOLERATED, ALPHA, BETA)
    result = check_mean_error(drift_errors(phi, m, rng), pilot["tolerated"], ALPHA)
    assert result.passed, result


PILOTS = {
    **{
        f"acf phi={phi} lag={lag}": (lambda m, rng, phi=phi, k=k: acf_errors(phi, m, rng)[:, k])
        for phi in ACF_PILOT
        for k, lag in enumerate(LAGS)
    },
    **{f"tau phi={phi}": (lambda m, rng, phi=phi: tau_errors(phi, m, rng)) for phi in TAU_PILOT},
    **{f"warmup burst={b}": (lambda m, rng, b=b: warmup_errors(b, m, rng)) for b in WARMUP_PILOT},
    **{f"warmup burst={b} untruncated": (lambda m, rng, b=b: warmup_errors(b, m, rng, truncate=False)) for b in WARMUP_PILOT},
    **{f"drift phi={phi}": (lambda m, rng, phi=phi: drift_errors(phi, m, rng)) for phi in DRIFT_PILOT},
}
