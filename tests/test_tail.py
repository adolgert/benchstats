"""Step 4: the detour tail."""

import numpy as np
import pytest
from scipy import stats

from benchstats import characterization
from support import synthetic
from support.design import ALPHA, BETA, check_mean_error, n_for_mean_error

# ---------------------------------------------------------------- tier 1


@pytest.mark.parametrize("xi", [-0.3, 0.0, 0.2, 0.4, 0.9])
@pytest.mark.parametrize("sigma", [0.5, 2.0])
def test_exact_lmoments_give_exact_parameters(xi, sigma):
    l1, l2, _, _ = synthetic.GPDTail(xi, sigma).lmoments()
    fitted_xi, fitted_sigma = characterization.gpd_from_lmoments(l1, l2)
    assert fitted_xi == pytest.approx(xi, abs=1e-12)
    assert fitted_sigma == pytest.approx(sigma, rel=1e-12)


def test_fit_uses_sample_lmoments_of_excesses(rng):
    excess = rng.exponential(size=50)
    l1, l2 = stats.lmoment(excess, order=[1, 2], standardize=False)
    assert characterization.gpd_fit_lmoments(excess) == pytest.approx((2 - l1 / l2, (l1 / l2 - 1) * l1))


def test_fit_rejects_bad_excesses():
    with pytest.raises(ValueError, match="non-negative"):
        characterization.gpd_fit_lmoments([1.0, -0.5, 2.0])
    with pytest.raises(ValueError, match="all be equal"):
        characterization.gpd_fit_lmoments([1.0, 1.0, 1.0])


def test_tail_fit_counts_and_mean_excess(rng):
    x = 1.0 + rng.exponential(0.1, size=200)
    fit = characterization.tail_fit(x, 1.1)
    above = x[x > 1.1]
    assert fit.n_exceed == above.size
    assert fit.rate == pytest.approx(above.size / 200)
    # With the L-moment fit, sigma / (1 - xi) is exactly the mean excess.
    assert fit.mean_excess == pytest.approx((above - 1.1).mean())
    with pytest.raises(ValueError, match="at least 2 draws above"):
        characterization.tail_fit(x, x.max())


def test_threshold_scan_matches_tail_fit(rng):
    x = 1.0 + rng.exponential(0.1, size=300)
    thresholds = np.array([1.05, 1.1, 1.2, x.max() + 1])
    scan = characterization.threshold_scan(x, thresholds)
    for i, u in enumerate(thresholds[:3]):
        fit = characterization.tail_fit(x, u)
        assert scan.n_exceed[i] == fit.n_exceed
        assert scan.xi[i] == pytest.approx(fit.xi)
        assert scan.modified_scale[i] == pytest.approx(fit.sigma - fit.xi * u)
        assert scan.mean_excess[i] == pytest.approx(fit.mean_excess)
    assert scan.n_exceed[3] == 0
    assert np.isnan(scan.xi[3]) and np.isnan(scan.mean_excess[3])


# ---------------------------------------------------------------- tier 2: GPD recovery

# Errors: xi absolute, sigma relative. The plan's example levels for xi are
# tolerate 0.01 and catch 0.05; sigma tolerates its pilot bias and must catch
# 0.04 more. Tolerated levels are the pilot's |mean error| plus three
# standard errors, rounded up, and at least the plan's 0.01 for xi.
# Pilot: scripts/pilot.py test_tail, 2,000 datasets.
GPD_PILOT = {
    (0.0, 200): {"sd": (0.080, 0.109), "tolerated": (0.010, 0.010)},
    (0.0, 1000): {"sd": (0.037, 0.048), "tolerated": (0.010, 0.005)},
    (0.2, 200): {"sd": (0.086, 0.113), "tolerated": (0.015, 0.015)},
    (0.2, 1000): {"sd": (0.038, 0.049), "tolerated": (0.010, 0.005)},
    (0.4, 200): {"sd": (0.094, 0.115), "tolerated": (0.025, 0.025)},
    (0.4, 1000): {"sd": (0.048, 0.054), "tolerated": (0.010, 0.010)},
}
CATCH_ABOVE_TOLERATED = 0.04


def gpd_errors(xi, n_excess, n_datasets, rng):
    dist = synthetic.GPDTail(xi, 2.0)
    errors = np.empty((n_datasets, 2))
    for i in range(n_datasets):
        fitted_xi, fitted_sigma = characterization.gpd_fit_lmoments(dist.sample(n_excess, rng))
        errors[i] = fitted_xi - xi, fitted_sigma / 2.0 - 1
    return errors


@pytest.mark.statistical
@pytest.mark.parametrize("case", list(GPD_PILOT), ids=[f"xi{x}-n{n}" for x, n in GPD_PILOT])
def test_gpd_fit_recovers_parameters(case, rng):
    xi, n_excess = case
    alpha = ALPHA / 2
    tolerated = GPD_PILOT[case]["tolerated"]
    sizes = [n_for_mean_error(sd, b, b + CATCH_ABOVE_TOLERATED, alpha, BETA) for sd, b in zip(GPD_PILOT[case]["sd"], tolerated)]
    errors = gpd_errors(xi, n_excess, max(sizes), rng)
    for k, name in enumerate(("xi", "sigma")):
        result = check_mean_error(errors[: sizes[k], k], tolerated[k], alpha)
        assert result.passed, f"{name}: {result}"


# ---------------------------------------------------------------- tier 2: detours

ONE_SOURCE = synthetic.FloorPlusDetours(1.0, 0.05, (0.1,), (0.2,))
ONE_SOURCE_N = 2000
# Errors: xi absolute; exceedance rate and mean excess relative. The rate and
# the mean excess are unbiased (the excess is exactly exponential above
# floor + jitter), so they tolerate no bias and must catch 5%. xi tolerates
# its pilot bias and must catch 0.05 more.
ONE_SOURCE_PILOT = {
    1.1: {"sd": (0.098, 0.082, 0.086), "xi_tolerated": 0.010},
    1.3: {"sd": (0.175, 0.137, 0.140), "xi_tolerated": 0.035},
}


def one_source_errors(u, n_datasets, rng):
    rate, excess = ONE_SOURCE.exceedance_rate(u), ONE_SOURCE.mean_excess(u)
    errors = np.empty((n_datasets, 3))
    for i in range(n_datasets):
        fit = characterization.tail_fit(ONE_SOURCE.sample(ONE_SOURCE_N, rng), u)
        errors[i] = fit.xi, fit.rate / rate - 1, fit.mean_excess / excess - 1
    return errors


@pytest.mark.statistical
@pytest.mark.parametrize("u", list(ONE_SOURCE_PILOT))
def test_one_exponential_source_gives_xi_zero_and_exact_tail(u, rng):
    alpha = ALPHA / 3
    pilot = ONE_SOURCE_PILOT[u]
    levels = [(pilot["xi_tolerated"], pilot["xi_tolerated"] + 0.05), (0.0, 0.05), (0.0, 0.05)]
    sizes = [n_for_mean_error(sd, b, c, alpha, BETA) for sd, (b, c) in zip(pilot["sd"], levels)]
    errors = one_source_errors(u, max(sizes), rng)
    for k, name in enumerate(("xi", "exceedance rate", "mean excess")):
        result = check_mean_error(errors[: sizes[k], k], levels[k][0], alpha)
        assert result.passed, f"{name}: {result}"


TWO_SOURCES = synthetic.FloorPlusDetours(1.0, 0.05, (0.1, 0.02), (0.1, 0.5))
TWO_SOURCES_N = 20_000
SCAN_THRESHOLDS = (1.1, 1.3, 1.6, 2.0)
# Relative errors of the scan's mean excess: unbiased, must catch 5%.
SCAN_SD = (0.043, 0.065, 0.087, 0.135)


def scan_errors(n_datasets, rng):
    truth = np.array([TWO_SOURCES.mean_excess(u) for u in SCAN_THRESHOLDS])
    errors = np.empty((n_datasets, len(SCAN_THRESHOLDS)))
    for i in range(n_datasets):
        scan = characterization.threshold_scan(TWO_SOURCES.sample(TWO_SOURCES_N, rng), SCAN_THRESHOLDS)
        errors[i] = scan.mean_excess / truth - 1
    return errors


def test_two_source_mean_excess_rises_toward_larger_mean():
    curve = [TWO_SOURCES.mean_excess(u) for u in SCAN_THRESHOLDS]
    assert np.all(np.diff(curve) > 0)
    assert curve[-1] < 0.5 and curve[-1] > 0.45


@pytest.mark.statistical
def test_threshold_scan_follows_exact_mean_excess_curve(rng):
    alpha = ALPHA / len(SCAN_THRESHOLDS)
    sizes = [n_for_mean_error(sd, 0.0, 0.05, alpha, BETA) for sd in SCAN_SD]
    errors = scan_errors(max(sizes), rng)
    for k, u in enumerate(SCAN_THRESHOLDS):
        result = check_mean_error(errors[: sizes[k], k], 0.0, alpha)
        assert result.passed, f"threshold {u}: {result}"


PILOTS = {
    **{
        f"gpd xi={xi} n={n} {name}": (lambda m, rng, xi=xi, n=n, k=k: gpd_errors(xi, n, m, rng)[:, k])
        for xi, n in GPD_PILOT
        for k, name in enumerate(("xi", "sigma"))
    },
    **{
        f"one source u={u} {name}": (lambda m, rng, u=u, k=k: one_source_errors(u, m, rng)[:, k])
        for u in ONE_SOURCE_PILOT
        for k, name in enumerate(("xi", "rate", "mean excess"))
    },
    **{f"scan u={u}": (lambda m, rng, k=k: scan_errors(m, rng)[:, k]) for k, u in enumerate(SCAN_THRESHOLDS)},
}
