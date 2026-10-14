"""Step 3: L-moments."""

import itertools
import math

import numpy as np
import pytest

from benchstats import characterization
from support import synthetic
from support.design import ALPHA, BETA, check_mean_error, n_for_mean_error


def _lmoments_by_definition(x):
    """Unbiased sample L-moments as averages over all subsets (Hosking 1990, eq. 2.1)."""
    x = np.sort(np.asarray(x, dtype=float))
    out = []
    for r in range(1, 5):
        weights = [(-1) ** k * math.comb(r - 1, k) for k in range(r)]  # on X_{r-k:r}
        total = 0.0
        subsets = list(itertools.combinations(x, r))
        for subset in subsets:
            ordered = sorted(subset)
            total += sum(w * ordered[r - 1 - k] for k, w in enumerate(weights))
        out.append(total / (r * len(subsets)))
    return np.array(out)


# ---------------------------------------------------------------- tier 1


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0], [2.5, 5 / 6, 0.0, 0.0]),
        ([1.0, 1.0, 2.0, 5.0], [9 / 4, 13 / 12, 3 / 4, 1 / 4]),  # tied values
    ],
)
def test_lmoments_hand_computed(x, expected):
    result = characterization.lmoments(x)
    np.testing.assert_allclose([result.l1, result.l2, result.l3, result.l4], expected, atol=1e-14)


def test_lmoments_match_definition_on_random_tied_inputs(rng):
    for size in (4, 5, 9):
        x = rng.integers(1, 5, size=size).astype(float)  # many ties
        if np.all(x == x[0]):
            x[0] += 1
        result = characterization.lmoments(x)
        np.testing.assert_allclose([result.l1, result.l2, result.l3, result.l4], _lmoments_by_definition(x), atol=1e-12)


def test_lmoment_ratios_are_ratios_of_lmoments(rng):
    x = rng.lognormal(size=50)
    m, r = characterization.lmoments(x), characterization.lmoment_ratios(x)
    assert r.cv == pytest.approx(m.l2 / m.l1)
    assert r.skewness == pytest.approx(m.l3 / m.l2)
    assert r.kurtosis == pytest.approx(m.l4 / m.l2)


def test_fewer_than_four_draws_raise():
    with pytest.raises(ValueError, match="at least 4"):
        characterization.lmoments([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="at least 4"):
        characterization.lmoment_ratios([1.0, 2.0, 3.0])


def test_bad_input_raises():
    with pytest.raises(ValueError, match="finite"):
        characterization.lmoments([1.0, 2.0, np.inf, 4.0])
    with pytest.raises(ValueError, match="1-D"):
        characterization.lmoments(np.ones((2, 4)))
    with pytest.raises(ValueError, match="positive"):
        characterization.lmoment_ratios([1.0, 2.0, -3.0, 4.0])


def test_affine_and_reflection_invariance(rng):
    x = rng.lognormal(size=40)
    a, b = rng.normal(), rng.uniform(0.1, 10)
    base = characterization.lmoments(x)
    shifted = characterization.lmoments(a + b * x)
    assert shifted.l2 == pytest.approx(b * base.l2)
    assert shifted.l3 / shifted.l2 == pytest.approx(base.l3 / base.l2)
    assert shifted.l4 / shifted.l2 == pytest.approx(base.l4 / base.l2)
    reflected = characterization.lmoments(-x)
    assert reflected.l3 / reflected.l2 == pytest.approx(-base.l3 / base.l2)
    assert reflected.l4 / reflected.l2 == pytest.approx(base.l4 / base.l2)


# ---------------------------------------------------------------- tier 2

DISTRIBUTIONS = {
    "gpd xi=0": synthetic.GPDTail(0.0, 1.0, 1.0),
    "gpd xi=0.2": synthetic.GPDTail(0.2, 1.0, 1.0),
    "gpd xi=0.4": synthetic.GPDTail(0.4, 1.0, 1.0),
    "floor_plus_detours": synthetic.FloorPlusDetours(1.0, 0.05, (0.1, 0.02), (0.2, 1.0)),
}
QUANTITIES = ("l1", "l2", "tau3", "tau4")

# Must-catch levels. lambda_1 and lambda_2 are unbiased, so they tolerate no
# bias; their errors are relative. tau3 and tau4 are ratios with a small
# finite-sample bias; their errors are absolute and their tolerated levels
# come from the pilot (scripts/pilot.py test_lmoments, 2,000 datasets):
# the pilot's |mean error| plus three standard errors, rounded up.
CATCH_RELATIVE = 0.02
CATCH_ABOVE_TOLERATED = 0.02

# (distribution, n) -> sd of each quantity's error, and tolerated bias of tau3 and tau4.
PILOT = {
    ("gpd xi=0", 100): {"sd": (0.050, 0.117, 0.049, 0.046), "tolerated": (0.010, 0.010)},
    ("gpd xi=0", 1000): {"sd": (0.016, 0.036, 0.015, 0.014), "tolerated": (0.005, 0.005)},
    ("gpd xi=0.2", 100): {"sd": (0.072, 0.165, 0.062, 0.070), "tolerated": (0.015, 0.015)},
    ("gpd xi=0.2", 1000): {"sd": (0.023, 0.053, 0.020, 0.023), "tolerated": (0.005, 0.005)},
    ("gpd xi=0.4", 100): {"sd": (0.121, 0.281, 0.084, 0.104), "tolerated": (0.025, 0.030)},
    ("gpd xi=0.4", 1000): {"sd": (0.044, 0.098, 0.033, 0.043), "tolerated": (0.005, 0.010)},
    ("floor_plus_detours", 100): {"sd": (0.019, 0.462, 0.105, 0.102), "tolerated": (0.050, 0.045)},
    ("floor_plus_detours", 1000): {"sd": (0.007, 0.147, 0.029, 0.033), "tolerated": (0.005, 0.005)},
}


def _truth(dist):
    l1, l2, l3, l4 = dist.lmoments()
    return np.array([l1, l2, l3 / l2, l4 / l2])


def lmoment_errors(name, n, n_datasets, rng):
    """Errors of (lambda_1, lambda_2) relative to truth, and of (tau3, tau4) absolute."""
    dist = DISTRIBUTIONS[name]
    truth = _truth(dist)
    errors = np.empty((n_datasets, 4))
    for i in range(n_datasets):
        r = characterization.lmoment_ratios(dist.sample(n, rng))
        errors[i] = np.array([r.l1, r.l2, r.skewness, r.kurtosis]) - truth
    errors[:, :2] /= truth[:2]
    return errors


def _levels(case, k):
    if k < 2:
        return 0.0, CATCH_RELATIVE
    tolerated = PILOT[case]["tolerated"][k - 2]
    return tolerated, tolerated + CATCH_ABOVE_TOLERATED


@pytest.mark.statistical
@pytest.mark.parametrize("case", list(PILOT), ids=[f"{d}-n{n}" for d, n in PILOT])
def test_lmoment_estimators_have_small_bias(case, rng):
    name, n = case
    alpha = ALPHA / len(QUANTITIES)
    sizes = [n_for_mean_error(sd, *_levels(case, k), alpha, BETA) for k, sd in enumerate(PILOT[case]["sd"])]
    errors = lmoment_errors(name, n, max(sizes), rng)
    for k, quantity in enumerate(QUANTITIES):
        tolerated = _levels(case, k)[0]
        result = check_mean_error(errors[: sizes[k], k], tolerated, alpha)
        assert result.passed, f"{quantity}: {result}"


PILOTS = {
    f"{name} n={n} {q}": (lambda m, rng, name=name, n=n, k=k: lmoment_errors(name, n, m, rng)[:, k])
    for name, n in PILOT
    for k, q in enumerate(QUANTITIES)
}
