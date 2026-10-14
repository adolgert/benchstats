"""Step 1d: exact tests of the sizing functions and pass/fail rules.

The pass/fail rules are checked with fake tests that fail with a known
probability. Because the fakes are Bernoulli draws, the rules' error rates
follow exactly from the binomial distribution, so the chain of tests of
tests ends here with exact arithmetic.
"""

import math

import numpy as np
import pytest
from scipy import stats

from support.design import check_mean_error, check_rate, n_for_mean_error, n_for_rate, rejection_probability


def _exact_failure_rate(n, p0, true_p, alpha, side):
    """Probability that check_rate fails, summed over every possible count."""
    return sum(stats.binom.pmf(k, n, true_p) for k in range(n + 1) if not check_rate(k, n, p0, alpha, side).passed)


# ---------------------------------------------------------------- n_for_rate


def test_n_for_rate_hand_computed_upper():
    # n = 8: P(X >= 7 | 0.5) = 9/256 <= 0.05 and P(X >= 7 | 0.9) = 0.813 >= 0.8.
    # n = 4..7 reach at most 0.59 power, so 8 is the smallest.
    assert n_for_rate(0.5, 0.9, 0.05, 0.2) == 8


def test_n_for_rate_hand_computed_lower():
    assert n_for_rate(0.5, 0.1, 0.05, 0.2) == 8


@pytest.mark.parametrize(
    ("p0", "p1", "expected"),
    [(0.05, 0.08, 1400), (0.95, 0.90, 550)],
)
def test_n_for_rate_matches_plan_examples(p0, p1, expected):
    """The plan's normal-approximation examples with alpha = 0.001, beta = 0.05."""
    assert n_for_rate(p0, p1, 0.001, 0.05) == pytest.approx(expected, rel=0.1)


@pytest.mark.parametrize(
    ("p0", "p1", "side"),
    [(0.05, 0.08, "upper"), (0.95, 0.90, "lower"), (0.05, 0.10, "two-sided"), (0.5, 0.3, "lower")],
)
def test_n_for_rate_gives_exact_size_and_power(p0, p1, side):
    alpha, beta = 0.001, 0.05
    n = n_for_rate(p0, p1, alpha, beta, side=side)
    assert _exact_failure_rate(n, p0, p0, alpha, side) <= alpha
    assert _exact_failure_rate(n, p0, p1, alpha, side) >= 1 - beta
    assert rejection_probability(n, p0, p1, alpha, side) == pytest.approx(_exact_failure_rate(n, p0, p1, alpha, side))
    # One fewer dataset falls short of the power, unless the power is sawtoothed.
    smaller = [rejection_probability(k, p0, p1, alpha, side) for k in range(max(1, n - 50), n)]
    assert max(smaller) < 1 - beta


def test_n_for_rate_rejects_bad_rates():
    with pytest.raises(ValueError, match="strictly between"):
        n_for_rate(0.0, 0.1, 0.05, 0.05)
    with pytest.raises(ValueError, match="differ"):
        n_for_rate(0.1, 0.1, 0.05, 0.05)


# ---------------------------------------------------------------- check_rate


def test_check_rate_hand_computed():
    # P(X >= 9 | n=10, p=0.5) = 11/1024; P(X >= 8) = 56/1024.
    fails = check_rate(9, 10, 0.5, 0.05, "upper")
    assert not fails.passed
    assert fails.p_value == pytest.approx(11 / 1024)
    passes = check_rate(8, 10, 0.5, 0.05, "upper")
    assert passes.passed
    assert passes.p_value == pytest.approx(56 / 1024)
    # P(X <= 1 | n=10, p=0.5) = 11/1024, doubled for two sides.
    assert check_rate(1, 10, 0.5, 0.05, "two-sided").p_value == pytest.approx(22 / 1024)


@pytest.mark.parametrize("true_rate", [0.05, 0.08])
@pytest.mark.parametrize("side", ["upper", "lower", "two-sided"])
def test_fake_bernoulli_tests_fail_at_exact_rate(true_rate, side):
    """A fake test that fails with probability exactly 5% or 8%, judged at p0 = 5%."""
    n, p0, alpha = 400, 0.05, 0.01
    rate = _exact_failure_rate(n, p0, true_rate, alpha, side)
    assert rate == pytest.approx(rejection_probability(n, p0, true_rate, alpha, side))
    if true_rate == p0:
        assert rate <= alpha
    elif side == "upper":
        assert rate > alpha  # the rule notices a rate that is too high
    elif side == "lower":
        assert rate < alpha  # and ignores it when only low rates matter


def test_check_rate_rejects_bad_input():
    with pytest.raises(ValueError, match="side"):
        check_rate(1, 10, 0.5, 0.05, "both")
    with pytest.raises(ValueError, match="between 0 and n"):
        check_rate(11, 10, 0.5, 0.05, "upper")


# ---------------------------------------------------------------- error tests


def test_n_for_mean_error_hand_computed():
    # ((1.6449 + 1.6449) * 1 / 0.5)^2 = 43.29, rounded up.
    assert n_for_mean_error(1.0, 0.0, 0.5, 0.05, 0.05) == 44
    # The tolerated bias only shrinks the gap: ((z + z) * 2 / (0.3 - 0.1))^2 = 1082.2.
    assert n_for_mean_error(2.0, 0.1, 0.3, 0.05, 0.05) == 1083


def test_n_for_mean_error_gives_exact_power_with_known_standard_error():
    """With the standard error known, the rule's power follows from the normal cdf."""
    sd, tolerated, catch, alpha, beta = 0.3, 0.01, 0.05, 0.001, 0.05
    m = n_for_mean_error(sd, tolerated, catch, alpha, beta)
    se = sd / math.sqrt(m)
    limit = tolerated + stats.norm.ppf(1 - alpha) * se
    power = stats.norm.sf((limit - catch) / se) + stats.norm.cdf((-limit - catch) / se)
    size = stats.norm.sf((limit - tolerated) / se) + stats.norm.cdf((-limit - tolerated) / se)
    assert power >= 1 - beta
    assert size <= alpha * 1.0001


def test_check_mean_error_hand_computed():
    errors = np.array([1.0, 2.0, 3.0, 4.0])  # mean 2.5, standard error 0.6455
    result = check_mean_error(errors, tolerated=1.0, alpha=0.05)
    assert result.mean == pytest.approx(2.5)
    assert result.standard_error == pytest.approx(math.sqrt(5 / 3) / 2)
    assert result.limit == pytest.approx(1.0 + 1.6448536 * 0.6454972, rel=1e-6)
    assert not result.passed
    assert check_mean_error(errors, tolerated=2.0, alpha=0.05).passed
    assert not check_mean_error(-errors, tolerated=1.0, alpha=0.05).passed


def test_check_mean_error_rejects_bad_input():
    with pytest.raises(ValueError, match="at least two"):
        check_mean_error([1.0], tolerated=0.1, alpha=0.05)
    with pytest.raises(ValueError, match="finite"):
        check_mean_error([1.0, np.nan], tolerated=0.1, alpha=0.05)
