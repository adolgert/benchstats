"""Sizing and pass/fail rules for statistical tests.

There are two kinds of statistical test.

Rate tests check a probability: a false-alarm rate, a coverage, or a power.
Each states the rate p0 it tolerates and the wrong rate p1 it must catch.
``n_for_rate`` sizes the test and ``check_rate`` judges it. Both use the
exact binomial distribution.

Error tests check an estimator's average error. Each states the bias it
tolerates and the larger bias it must catch. ``n_for_mean_error`` sizes the
test and ``check_mean_error`` judges it, with the normal approximation.

Error budget (decision D5): a full tier-2 run should fail by chance with
probability at most ``FAMILY_ALPHA``. Each statistical test item gets
``ALPHA = FAMILY_ALPHA / N_STATISTICAL_TESTS``. A test item that makes several
checks divides its ``ALPHA`` among them. A tier-1 test keeps
``N_STATISTICAL_TESTS`` equal to the number of collected tier-2 items.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

FAMILY_ALPHA = 0.05
N_STATISTICAL_TESTS = 63
ALPHA = FAMILY_ALPHA / N_STATISTICAL_TESTS
BETA = 0.05

_SIDES = ("upper", "lower", "two-sided")


@dataclass(frozen=True)
class RateCheck:
    passed: bool
    count: int
    n: int
    p0: float
    side: str
    p_value: float


@dataclass(frozen=True)
class MeanErrorCheck:
    passed: bool
    mean: float
    standard_error: float
    tolerated: float
    limit: float


def _infer_side(p0, p1):
    if p1 == p0:
        raise ValueError("p1 must differ from p0")
    return "upper" if p1 > p0 else "lower"


def _upper_tail(count, n, p):
    """P(X >= count) for X ~ Binomial(n, p)."""
    return stats.binom.sf(np.asarray(count) - 1, n, p)


def _lower_tail(count, n, p):
    """P(X <= count) for X ~ Binomial(n, p)."""
    return stats.binom.cdf(count, n, p)


def _p_value(count, n, p0, side):
    if side == "upper":
        return _upper_tail(count, n, p0)
    if side == "lower":
        return _lower_tail(count, n, p0)
    # Equal-tailed: twice the smaller tail, capped at 1.
    return np.minimum(1.0, 2 * np.minimum(_upper_tail(count, n, p0), _lower_tail(count, n, p0)))


def rejection_probability(n, p0, p, alpha, side):
    """Exact probability that check_rate fails when the true rate is p."""
    counts = np.arange(n + 1)
    reject = _p_value(counts, n, p0, side) < alpha
    return float(stats.binom.pmf(counts, n, p)[reject].sum())


def n_for_rate(p0, p1, alpha, beta, *, side=None, n_max=200_000):
    """Number of simulated datasets for a rate test.

    The test tolerates rate p0 and must catch rate p1. It returns the smallest
    n at which ``check_rate`` fails with probability at least 1 - beta when
    the true rate is p1. Its false-failure rate at p0 is at most alpha by
    construction. ``side`` defaults to the direction from p0 to p1; a
    two-sided test splits alpha equally between the tails.
    """
    for p in (p0, p1):
        if not 0 < p < 1:
            raise ValueError("rates must lie strictly between 0 and 1")
    side = side or _infer_side(p0, p1)
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}")
    # Start from half the normal approximation and step up exactly.
    z = stats.norm.ppf(1 - alpha / (2 if side == "two-sided" else 1)) * math.sqrt(p0 * (1 - p0))
    z += stats.norm.ppf(1 - beta) * math.sqrt(p1 * (1 - p1))
    n = max(1, int((z / (p1 - p0)) ** 2 / 2))
    while n <= n_max:
        if rejection_probability(n, p0, p1, alpha, side) >= 1 - beta:
            return n
        n += 1
    raise ValueError(f"no n up to {n_max} gives the requested power")


def check_rate(count, n, p0, alpha, side):
    """Pass/fail rule for a rate test.

    ``side="upper"`` fails when the observed rate is significantly above p0,
    ``"lower"`` when it is significantly below, and ``"two-sided"`` when
    either holds (alpha split equally). The p-value is exact.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}")
    if not 0 <= count <= n:
        raise ValueError("count must lie between 0 and n")
    p_value = float(_p_value(count, n, p0, side))
    return RateCheck(p_value >= alpha, int(count), int(n), p0, side, p_value)


MIN_DATASETS = 30


def n_for_mean_error(sd, tolerated, catch, alpha, beta):
    """Number of simulated datasets M for an error test.

    M = ((z_{1-alpha} + z_{1-beta}) * sd / (catch - tolerated))**2, where sd
    is the estimator's standard deviation at the test's sample size,
    measured once in a pilot run. M is at least MIN_DATASETS, so that the
    standard error in ``check_mean_error`` is itself estimated well.
    """
    if not 0 <= tolerated < catch:
        raise ValueError("need 0 <= tolerated < catch")
    if sd <= 0:
        raise ValueError("sd must be positive")
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - beta)
    return max(MIN_DATASETS, math.ceil((z * sd / (catch - tolerated)) ** 2))


def check_mean_error(errors, tolerated, alpha):
    """Pass/fail rule for an error test.

    Fails when the mean error lies beyond +/- tolerated by more than
    z_{1-alpha} standard errors.
    """
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1 or errors.size < 2:
        raise ValueError("errors must be 1-D with at least two values")
    if not np.all(np.isfinite(errors)):
        raise ValueError("errors must be finite")
    mean = float(errors.mean())
    se = float(errors.std(ddof=1) / math.sqrt(errors.size))
    limit = tolerated + stats.norm.ppf(1 - alpha) * se
    return MeanErrorCheck(abs(mean) <= limit, mean, se, tolerated, limit)
