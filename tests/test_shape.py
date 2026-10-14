"""Step 6: shape: density of log times, modes, and the dip test."""

import sys

import numpy as np
import pytest
from scipy import stats

from benchstats import characterization
from support import synthetic
from support.design import ALPHA, BETA, check_mean_error, check_rate, n_for_mean_error, n_for_rate

pytest.importorskip("KDEpy")

PROMINENCE = 0.1

# ---------------------------------------------------------------- tier 1


def _two_gaussians(grid, weights=(0.7, 0.3), means=(0.0, 3.0), sd=0.5):
    return sum(w * stats.norm.pdf(grid, m, sd) for w, m in zip(weights, means))


def test_find_modes_on_exact_mixture_density():
    grid = np.linspace(-4, 7, 4001)
    modes = characterization.find_modes(grid, _two_gaussians(grid), min_prominence=PROMINENCE)
    np.testing.assert_allclose(modes.location, [0.0, 3.0], atol=0.01)
    # The minimum between the components splits off a sliver of each tail.
    np.testing.assert_allclose(modes.mass, [0.7, 0.3], atol=0.002)
    assert modes.mass.sum() == pytest.approx(1.0)


def test_find_modes_prominence_floor_removes_small_bumps():
    grid = np.linspace(-4, 7, 4001)
    density = _two_gaussians(grid, weights=(0.97, 0.03))
    assert characterization.find_modes(grid, density, min_prominence=0.01).location.size == 2
    assert characterization.find_modes(grid, density, min_prominence=0.1).location.size == 1


def test_find_modes_detects_peaks_at_the_grid_ends():
    grid = np.linspace(0, 1, 101)
    modes = characterization.find_modes(grid, np.exp(-grid), min_prominence=PROMINENCE)
    assert modes.location.tolist() == [0.0]


def test_find_modes_rejects_bad_input():
    with pytest.raises(ValueError, match="same length"):
        characterization.find_modes(np.arange(5.0), np.ones(4), min_prominence=0.1)
    with pytest.raises(ValueError, match="increasing"):
        characterization.find_modes(np.array([0.0, 2.0, 1.0]), np.ones(3), min_prominence=0.1)
    with pytest.raises(ValueError, match="fraction"):
        characterization.find_modes(np.arange(5.0), np.ones(5), min_prominence=2.0)


def _deterministic_two_floor_sample():
    """Quantiles of a two-floor mixture: a fixed sample with two clear modes."""
    dist = synthetic.two_floors((1.0, 1.3), (0.05, 0.05), (0.6, 0.4))
    return dist.ppf((np.arange(400) + 0.5) / 400)


def test_log_density_is_a_density_on_the_log_scale():
    x = _deterministic_two_floor_sample()
    grid, density = characterization.log_density(x, grid_size=512)
    assert grid.size == 512
    assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=1e-3)
    assert grid.min() < np.log(x.min()) and grid.max() > np.log(x.max())
    with pytest.raises(ValueError, match="bandwidth"):
        characterization.log_density(x, bandwidth="wide")


def test_mode_count_scan_merges_modes_at_large_bandwidths():
    x = _deterministic_two_floor_sample()
    counts = characterization.mode_count_scan(x, [0.02, 0.05, 1.0], min_prominence=PROMINENCE)
    assert counts.tolist() == [2, 2, 1]


def test_dip_test_wraps_diptest():
    diptest = pytest.importorskip("diptest")
    x = _deterministic_two_floor_sample()
    assert characterization.dip_test(x) == pytest.approx(diptest.diptest(x))


def test_dip_test_explains_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "diptest", None)
    with pytest.raises(ImportError, match=r"benchstats\[diptest\]"):
        characterization.dip_test([1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------- tier 2: dip test

DIP_LEVEL = 0.05
DIP_N = 200
UNIMODAL = {
    "uniform": stats.uniform(1.0, 0.5),
    "gpd_tail": synthetic.GPDTail(0.2, 1.0, 1.0),
    "floor_plus_detours": synthetic.FloorPlusDetours(1.0, 0.05, (0.1, 0.02), (0.2, 1.0)),
}


def _draw(dist, n, rng):
    return dist.rvs(size=n, random_state=rng) if hasattr(dist, "rvs") else dist.sample(n, rng)


def _rejection_count(dist, n_datasets, rng):
    return sum(characterization.dip_test(_draw(dist, DIP_N, rng))[1] < DIP_LEVEL for _ in range(n_datasets))


@pytest.mark.statistical
def test_dip_test_is_calibrated_under_uniform(rng):
    pytest.importorskip("diptest")
    n = n_for_rate(DIP_LEVEL, 0.08, ALPHA, BETA, side="two-sided")
    result = check_rate(_rejection_count(UNIMODAL["uniform"], n, rng), n, DIP_LEVEL, ALPHA, "two-sided")
    assert result.passed, result


@pytest.mark.statistical
@pytest.mark.parametrize("name", ["gpd_tail", "floor_plus_detours"])
def test_dip_test_is_conservative_for_other_unimodal_shapes(name, rng):
    pytest.importorskip("diptest")
    n = n_for_rate(DIP_LEVEL, 2 * DIP_LEVEL, ALPHA, BETA)
    result = check_rate(_rejection_count(UNIMODAL[name], n, rng), n, DIP_LEVEL, ALPHA, "upper")
    assert result.passed, result


# Two floors 0.6% apart with jitter 0.5% (a gap of 0.1% between the jitter
# bands), weights 0.6 and 0.4, n = 200. Pilot power 0.79 (1,000 datasets); at
# a 0.8% separation the power is already 1. The test tolerates 0.70 and must
# catch 0.40, far above the power of about alpha a broken call would have.
DIP_POWER_CASE = synthetic.two_floors((1.0, 1.006), (0.005, 0.005), (0.6, 0.4))
DIP_POWER_TOLERATED, DIP_POWER_CATCH = 0.70, 0.40


@pytest.mark.statistical
def test_dip_test_has_power_on_two_floors(rng):
    pytest.importorskip("diptest")
    n = n_for_rate(DIP_POWER_TOLERATED, DIP_POWER_CATCH, ALPHA, BETA)
    result = check_rate(_rejection_count(DIP_POWER_CASE, n, rng), n, DIP_POWER_TOLERATED, ALPHA, "lower")
    assert result.passed, result


# ---------------------------------------------------------------- tier 2: modes

MODE_N = 500
TWO_FLOORS = synthetic.two_floors((1.0, 1.3), (0.05, 0.05), (0.6, 0.4))
FINDS_P0, FINDS_P1 = 0.95, 0.80  # tolerated and must-catch rates of reporting the right count


def _mode_count(dist, rng):
    grid, density = characterization.log_density(_draw(dist, MODE_N, rng))
    return characterization.find_modes(grid, density, min_prominence=PROMINENCE)


@pytest.mark.statistical
def test_two_floors_give_two_modes(rng):
    n = n_for_rate(FINDS_P0, FINDS_P1, ALPHA, BETA)
    count = sum(_mode_count(TWO_FLOORS, rng).location.size == 2 for _ in range(n))
    result = check_rate(count, n, FINDS_P0, ALPHA, "lower")
    assert result.passed, result


@pytest.mark.statistical
@pytest.mark.parametrize("name", ["gpd_tail", "floor_plus_detours"])
def test_unimodal_generators_give_one_mode(name, rng):
    n = n_for_rate(FINDS_P0, FINDS_P1, ALPHA, BETA)
    count = sum(_mode_count(UNIMODAL[name], rng).location.size == 1 for _ in range(n))
    result = check_rate(count, n, FINDS_P0, ALPHA, "lower")
    assert result.passed, result


def mode_errors(n_datasets, rng):
    """Errors of the two modes' log locations and masses, over datasets that found two."""
    locations, masses = synthetic.two_floor_modes(TWO_FLOORS)
    errors = []
    while len(errors) < n_datasets:
        modes = _mode_count(TWO_FLOORS, rng)
        if modes.location.size == 2:
            errors.append(np.r_[modes.location - locations, modes.mass - masses])
    return np.array(errors)


# Pilot (1,000 datasets): sd of the errors (location 1, location 2, mass 1,
# mass 2). The pilot bias is below 0.0001 for locations and 0.002 for masses;
# kernel smoothing leaks a little mass across the minimum. Locations
# tolerate 0.001 and must catch 0.005 (a tenth of the log jitter width);
# masses tolerate 0.005 and must catch 0.035.
MODE_PILOT = {"sd": (0.0009, 0.0008, 0.023, 0.022), "tolerated": (0.001, 0.001, 0.005, 0.005)}
MODE_CATCH_ABOVE = (0.004, 0.004, 0.03, 0.03)


@pytest.mark.statistical
def test_two_floor_mode_locations_and_masses(rng):
    alpha = ALPHA / 4
    sizes = [n_for_mean_error(sd, b, b + c, alpha, BETA) for sd, b, c in zip(MODE_PILOT["sd"], MODE_PILOT["tolerated"], MODE_CATCH_ABOVE)]
    errors = mode_errors(max(sizes), rng)
    for k, name in enumerate(("location 1", "location 2", "mass 1", "mass 2")):
        result = check_mean_error(errors[: sizes[k], k], MODE_PILOT["tolerated"][k], alpha)
        assert result.passed, f"{name}: {result}"


def dip_power(n_datasets, rng):
    return np.array([characterization.dip_test(_draw(DIP_POWER_CASE, DIP_N, rng))[1] < DIP_LEVEL for _ in range(n_datasets)], float)


PILOTS = {
    **{f"mode error {k}": (lambda m, rng, k=k: mode_errors(m, rng)[:, k]) for k in range(4)},
    "dip power (mean is the power)": dip_power,
    **{
        f"dip rejection {name} (mean is the rate)": (
            lambda m, rng, name=name: np.array(
                [characterization.dip_test(_draw(UNIMODAL[name], DIP_N, rng))[1] < DIP_LEVEL for _ in range(m)], float
            )
        )
        for name in UNIMODAL
    },
    **{
        f"one mode {name} (mean is the rate)": (
            lambda m, rng, name=name: np.array([_mode_count(UNIMODAL[name], rng).location.size == 1 for _ in range(m)], float)
        )
        for name in ("gpd_tail", "floor_plus_detours")
    },
}
