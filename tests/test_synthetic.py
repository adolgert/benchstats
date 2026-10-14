"""Step 1b: the generators match their known true values.

Tier 1 checks closed forms against numerical integration. Tier 2 checks each
generator's marginal against its known CDF with a Kolmogorov-Smirnov test, the
correlated generator against its known rank autocorrelation, and shows the
checks have power by catching a broken generator.
"""

import math

import numpy as np
import pytest
from scipy import integrate, stats

from support import synthetic
from support.design import ALPHA, BETA, check_mean_error, check_rate, n_for_mean_error, n_for_rate

DETOURS = synthetic.FloorPlusDetours(1.0, 0.05, (0.1, 0.02), (0.2, 1.0))
POWER_CHECK = synthetic.FloorPlusDetours(1.0, 0.05, (0.3,), (0.2,))
KS_SIZE = 80_000  # draws per KS check; large enough to catch a 10% error in the detour mean


# ---------------------------------------------------------------- tier 1: closed forms


def test_gpd_closed_form_lmoments_match_integration():
    gpd = synthetic.GPDTail(0.3, 2.0, 0.5)
    integrated = synthetic.Mixture((gpd,), (1.0,)).lmoments()
    np.testing.assert_allclose(gpd.lmoments(), integrated, rtol=1e-7)


def test_floor_plus_detours_mean_matches_integration():
    assert DETOURS.lmoments()[0] == pytest.approx(DETOURS.mean(), rel=1e-12)


def test_floor_plus_detours_pdf_integrates_to_cdf():
    for x in (1.01, 1.05, 1.3, 2.0):
        value, _ = integrate.quad(DETOURS.pdf, 1.0, x, points=[1.05], limit=200)
        assert value == pytest.approx(DETOURS.cdf(x), abs=1e-10)


@pytest.mark.parametrize("u", [1.05, 1.2, 2.0])
def test_exceedance_closed_forms_match_cdf(u):
    survival = lambda x: 1 - DETOURS.cdf(x)  # noqa: E731
    assert DETOURS.exceedance_rate(u) == pytest.approx(survival(u), rel=1e-10)
    integral, _ = integrate.quad(survival, u, np.inf, limit=200)
    assert DETOURS.mean_excess(u) == pytest.approx(integral / survival(u), rel=1e-8)


def test_exceedance_closed_forms_refuse_low_thresholds():
    with pytest.raises(ValueError, match="floor"):
        DETOURS.exceedance_rate(1.01)


def test_ppf_inverts_cdf():
    q = np.array([1e-6, 0.01, 0.5, 0.9, 0.999999])
    for dist in (DETOURS, synthetic.two_floors((1.0, 1.3), (0.05, 0.05), (0.6, 0.4))):
        np.testing.assert_allclose(dist.cdf(dist.ppf(q)), q, rtol=1e-9, atol=1e-12)


def test_bursty_stationary_probabilities():
    model = synthetic.BurstyModel(DETOURS, POWER_CHECK, stay_quiet=0.95, stay_noisy=0.8)
    quiet, noisy = model.stationary
    assert quiet == pytest.approx(0.2 / 0.25)
    # Stationary: the flow out of quiet equals the flow into it.
    assert quiet * 0.05 == pytest.approx(noisy * 0.2)


def test_hierarchical_median_without_offsets_is_base_median():
    assert synthetic.hierarchical_median(DETOURS, 1e-9) == pytest.approx(DETOURS.median(), rel=1e-9)


def test_warmup_decays_to_steady_state():
    time = np.ones(50)
    warmed = synthetic.add_warmup(time, np.arange(50), amplitude=1.0, decay=3.0)
    assert warmed[0] == pytest.approx(2.0)
    assert warmed[-1] == pytest.approx(1.0, abs=1e-6)


def test_drift_is_linear_in_log_time():
    time, slope = synthetic.add_drift(np.ones(101), np.arange(101), total_change=0.02)
    np.testing.assert_allclose(np.diff(np.log(time)), slope)
    assert time[-1] == pytest.approx(1.02)


def test_correlated_rank_autocorrelation_formula_at_zero_and_one():
    assert synthetic.correlated_rank_autocorrelation(0.0, 1) == 0.0
    assert synthetic.correlated_rank_autocorrelation(1.0, 1) == pytest.approx(1.0)


# ---------------------------------------------------------------- tier 2: marginals


def _thinned(series, step):
    """Every step-th draw: dependence at that lag is below 1e-9, so draws are independent."""
    return series[::step]


BURSTY = synthetic.BurstyModel(DETOURS, POWER_CHECK, stay_quiet=0.95, stay_noisy=0.8)

MARGINAL_CASES = {
    "floor_plus_detours": (DETOURS, lambda rng: DETOURS.sample(KS_SIZE, rng)),
    "gpd_tail": (synthetic.GPDTail(0.3), lambda rng: synthetic.GPDTail(0.3).sample(KS_SIZE, rng)),
    "two_floors": (
        synthetic.two_floors((1.0, 1.3), (0.05, 0.05), (0.6, 0.4)),
        lambda rng: synthetic.two_floors((1.0, 1.3), (0.05, 0.05), (0.6, 0.4)).sample(KS_SIZE, rng),
    ),
    # Lag 200 of phi = 0.9 has correlation 0.9^200 < 1e-9.
    "correlated": (DETOURS, lambda rng: _thinned(synthetic.correlated(DETOURS, 0.9, 200 * 10_000, rng), 200)),
    # The chain's second eigenvalue is 0.75; 0.75^100 < 1e-12.
    "bursty": (BURSTY.marginal, lambda rng: _thinned(synthetic.bursty(BURSTY, 100 * 10_000, rng), 100)),
}


@pytest.mark.statistical
@pytest.mark.parametrize("name", MARGINAL_CASES)
def test_marginal_matches_known_cdf(name, rng):
    dist, draw = MARGINAL_CASES[name]
    x = draw(rng)
    result = stats.ks_1samp(x, dist.cdf)
    assert result.pvalue >= ALPHA, f"KS statistic {result.statistic:.4g}, p = {result.pvalue:.3g}"


@pytest.mark.statistical
def test_hierarchical_marginal_matches_known_cdf(rng):
    # One draw per build gives independent draws from the marginal over offsets.
    draws = synthetic.hierarchical(DETOURS, n_builds=20_000, n_invocations=1, n_iterations=1, build_sd=0.03, invocation_sd=0.04, rng=rng)
    result = stats.ks_1samp(draws.time, lambda x: synthetic.hierarchical_cdf(DETOURS, 0.05, x))
    assert result.pvalue >= ALPHA, f"KS statistic {result.statistic:.4g}, p = {result.pvalue:.3g}"


# Pilot (scripts/pilot.py): sd of the lag-k Spearman error on n = 2,000 draws.
# The estimator's downward bias is of order tau_int / n, about 0.006 at phi = 0.9.
RANK_CASES = {
    0.5: {"sd": (0.021, 0.026, 0.028)},
    0.9: {"sd": (0.012, 0.022, 0.028)},
}
RANK_N = 2000
RANK_LAGS = (1, 2, 3)
RANK_TOLERATED = 0.01
RANK_CATCH = 0.03


def rank_errors(phi, lag, n_datasets, rng):
    truth = synthetic.correlated_rank_autocorrelation(phi, lag)
    errors = np.empty(n_datasets)
    for i in range(n_datasets):
        x = synthetic.correlated(DETOURS, phi, RANK_N, rng)
        errors[i] = stats.spearmanr(x[:-lag], x[lag:]).statistic - truth
    return errors


@pytest.mark.statistical
@pytest.mark.parametrize("phi", RANK_CASES)
def test_correlated_matches_known_rank_autocorrelation(phi, rng):
    alpha = ALPHA / len(RANK_LAGS)
    for lag, sd in zip(RANK_LAGS, RANK_CASES[phi]["sd"]):
        m = n_for_mean_error(sd, RANK_TOLERATED, RANK_CATCH, alpha, BETA)
        result = check_mean_error(rank_errors(phi, lag, m, rng), RANK_TOLERATED, alpha)
        assert result.passed, f"lag {lag}: {result}"


POWER_P0, POWER_P1 = 1 - BETA, 0.75


@pytest.mark.statistical
def test_ks_check_catches_broken_generator(rng):
    """A detour mean 10% too large is caught with probability at least 1 - beta."""
    broken = POWER_CHECK.broken(1.1)
    n = n_for_rate(POWER_P0, POWER_P1, ALPHA, BETA)
    caught = sum(stats.ks_1samp(broken.sample(KS_SIZE, rng), POWER_CHECK.cdf).pvalue < ALPHA for _ in range(n))
    result = check_rate(caught, n, POWER_P0, ALPHA, "lower")
    assert result.passed, result
    assert math.isfinite(result.p_value)


PILOTS = {
    f"rank phi={phi} lag={lag}": (lambda n, rng, phi=phi, lag=lag: rank_errors(phi, lag, n, rng)) for phi in RANK_CASES for lag in RANK_LAGS
}
