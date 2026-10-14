"""Characterization: describe one timing distribution.

Functions take plain 1-D float arrays of raw times in seconds, in run order
where order matters. Each docstring says whether the function works on raw
times or takes logs itself. Sections:

- L-moments (location, scale, shape)
- the detour tail (generalized Pareto fit and threshold scan)
- time order (rank autocorrelation, warmup, drift)
- modes (log density, mode finding, dip test)

This module imports nothing from the rest of the package.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import signal, stats

# ---------------------------------------------------------------- input checks


def _as_sample(x, name="x", *, positive=False, min_size=1):
    """A 1-D finite float64 array, or ValueError naming the failed check."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"{name}: must be 1-D")
    if x.size < min_size:
        raise ValueError(f"{name}: needs at least {min_size} values, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name}: all values must be finite")
    if positive and not np.all(x > 0):
        raise ValueError(f"{name}: all values must be positive")
    return x


# ---------------------------------------------------------------- L-moments


@dataclass(frozen=True)
class LMoments:
    l1: float
    l2: float
    l3: float
    l4: float


@dataclass(frozen=True)
class LMomentRatios:
    l1: float
    l2: float
    cv: float  # tau = l2 / l1
    skewness: float  # tau3 = l3 / l2
    kurtosis: float  # tau4 = l4 / l2


def lmoments(x):
    """Sample L-moments lambda_1 to lambda_4 (unbiased estimators).

    Works on raw or log times. Needs at least four values.
    """
    x = _as_sample(x, min_size=4)
    values = stats.lmoment(x, order=[1, 2, 3, 4], standardize=False)
    return LMoments(*(float(v) for v in values))


def lmoment_ratios(x):
    """lambda_1, lambda_2, the L-CV, the L-skewness tau3 and the L-kurtosis tau4.

    Takes raw times; the L-CV lambda_2 / lambda_1 assumes positive values.
    Needs at least four values.
    """
    x = _as_sample(x, positive=True, min_size=4)
    l1, l2, t3, t4 = stats.lmoment(x, order=[1, 2, 3, 4], standardize=True)
    return LMomentRatios(float(l1), float(l2), float(l2 / l1), float(t3), float(t4))


# ---------------------------------------------------------------- the detour tail


@dataclass(frozen=True)
class TailFit:
    threshold: float
    n_exceed: int
    rate: float  # fraction of draws above the threshold
    xi: float
    sigma: float
    mean_excess: float  # sigma / (1 - xi)


@dataclass(frozen=True)
class ThresholdScan:
    threshold: np.ndarray
    n_exceed: np.ndarray
    xi: np.ndarray
    modified_scale: np.ndarray  # sigma - xi * threshold
    mean_excess: np.ndarray


def gpd_from_lmoments(l1, l2):
    """Generalized Pareto shape xi and scale sigma from the first two L-moments.

    xi = 2 - l1 / l2 and sigma = (1 - xi) * l1. Valid for xi < 1.
    """
    if not (l1 > 0 and l2 > 0):
        raise ValueError("l1 and l2 must be positive")
    xi = 2.0 - l1 / l2
    return xi, (1.0 - xi) * l1


def gpd_fit_lmoments(excess):
    """Fit a generalized Pareto distribution to excesses over a threshold.

    Takes excesses x - u >= 0 in seconds. Returns (xi, sigma) from the first
    two sample L-moments. Valid for xi < 1, where the mean exists.
    """
    excess = _as_sample(excess, "excess", min_size=2)
    if np.any(excess < 0):
        raise ValueError("excess: values must be non-negative")
    if np.all(excess == excess[0]):
        raise ValueError("excess: values must not all be equal")
    l1, l2 = stats.lmoment(excess, order=[1, 2], standardize=False)
    return gpd_from_lmoments(float(l1), float(l2))


def tail_fit(x, threshold):
    """Detour tail above a threshold: how often and how large.

    Takes raw times. Returns the threshold, the number of draws above it, the
    exceedance rate, the generalized Pareto xi and sigma of the excesses, and
    the mean excess sigma / (1 - xi). For one exponential detour source,
    xi = 0 and the mean excess is the mean detour size. A person chooses the
    threshold; the median plus a few MADs is a first one to look at.
    """
    x = _as_sample(x, positive=True)
    above = x[x > threshold]
    if above.size < 2:
        raise ValueError(f"threshold: needs at least 2 draws above it, got {above.size}")
    xi, sigma = gpd_fit_lmoments(above - threshold)
    return TailFit(float(threshold), int(above.size), above.size / x.size, xi, sigma, sigma / (1 - xi))


def threshold_scan(x, thresholds):
    """Stability and mean-residual-life diagnostics over several thresholds.

    Takes raw times. For each threshold u, returns the count above it, xi,
    the modified scale sigma - xi * u, and the mean excess. Thresholds with
    fewer than two draws above them give NaN. See Coles (2001), section 4.3.

    One exponential detour source: xi stays near 0 and the mean excess stays
    flat. Several sources of different sizes: the mean excess rises toward
    the largest source's mean, and xi is above 0 at moderate thresholds and
    falls toward 0 at high ones. A power-law tail: xi stays at the same
    positive value.
    """
    x = _as_sample(x, positive=True)
    thresholds = _as_sample(thresholds, "thresholds")
    n_exceed = np.array([(x > u).sum() for u in thresholds])
    xi = np.full(thresholds.size, np.nan)
    sigma = np.full(thresholds.size, np.nan)
    for i, u in enumerate(thresholds):
        excess = x[x > u] - u
        if excess.size >= 2 and not np.all(excess == excess[0]):
            xi[i], sigma[i] = gpd_fit_lmoments(excess)
    return ThresholdScan(thresholds, n_exceed, xi, sigma - xi * thresholds, sigma / (1 - xi))


# ---------------------------------------------------------------- time order


def _rank_autocorrelation_all(x, max_lag):
    """Rank autocorrelation at lags 0..max_lag, by FFT."""
    r = stats.rankdata(x)
    r = r - r.mean()
    n = r.size
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(r, size)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), size)[: max_lag + 1]
    return acov / acov[0]


def rank_autocorrelation(x, max_lag):
    """Autocorrelation of the ranks of x in run order, at lags 1..max_lag.

    Takes raw times in run order. Large detours cannot dominate it, and it
    does not change under any increasing transform such as log. Uses the
    usual estimator with denominator n, which is biased downward by order 1/n.
    """
    x = _as_sample(x, min_size=3)
    if not 1 <= max_lag < x.size:
        raise ValueError("max_lag: must be between 1 and len(x) - 1")
    if np.all(x == x[0]):
        raise ValueError("x: values must not all be equal")
    return _rank_autocorrelation_all(x, max_lag)[1:]


def integrated_autocorrelation_time(x, *, c=5.0):
    """tau_int = 1 + 2 * sum of rank autocorrelations up to a window W.

    Takes raw times in run order. W is chosen by Sokal's rule: the smallest
    W with W >= c * tau_int(W). For independent draws tau_int is near 1.
    """
    x = _as_sample(x, min_size=4)
    if np.all(x == x[0]):
        raise ValueError("x: values must not all be equal")
    max_lag = x.size // 2
    rho = _rank_autocorrelation_all(x, max_lag)[1:]
    tau = 1 + 2 * np.cumsum(rho)
    window = np.arange(1, max_lag + 1)
    ok = np.flatnonzero(window >= c * tau)
    return float(tau[ok[0]] if ok.size else tau[-1])


def effective_sample_size(x):
    """n / tau_int, never more than n.

    Takes raw times in run order. Negatively correlated draws would give
    tau_int below 1; the result is then capped at n.
    """
    x = _as_sample(x, min_size=4)
    tau = integrated_autocorrelation_time(x)
    return float(x.size / tau) if tau > 1 else float(x.size)


def warmup_mser(x, batch=5):
    """MSER truncation point (White 1997): how many leading draws to drop.

    Takes raw times of one invocation in run order and works on their logs,
    which reduces the pull of large detours. Batch means of ``batch`` draws
    are formed, and the truncation d minimizes the variance of the remaining
    batch means divided by their count squared, over d up to half the
    batches. Returns the number of draws to drop (a multiple of ``batch``).
    """
    x = _as_sample(x, positive=True, min_size=2 * batch)
    if batch < 1:
        raise ValueError("batch: must be at least 1")
    n_batches = x.size // batch
    means = np.log(x[: n_batches * batch]).reshape(n_batches, batch).mean(axis=1)
    # Suffix sums give the statistic for every truncation d at once.
    tail = means[::-1]
    count = np.arange(1, n_batches + 1)
    s1 = np.cumsum(tail)[::-1]
    s2 = np.cumsum(tail * tail)[::-1]
    remaining = count[::-1]
    mser = (s2 - s1 * s1 / remaining) / remaining**2
    d = int(np.argmin(mser[: n_batches // 2 + 1]))
    return d * batch


def drift_slope(x):
    """Theil-Sen slope of log time against run order, as percent change per 1,000 draws.

    Takes raw times in run order. Descriptive only: a p-value would assume
    independent draws. Uses scipy.stats.theilslopes, which holds all pairs
    in memory, so it suits up to several thousand draws.
    """
    x = _as_sample(x, positive=True, min_size=3)
    slope = stats.theilslopes(np.log(x)).slope
    return 100.0 * math.expm1(1000.0 * slope)


# ---------------------------------------------------------------- modes


@dataclass(frozen=True)
class Modes:
    location: np.ndarray  # log seconds
    height: np.ndarray  # density at each mode
    mass: np.ndarray  # probability between the minima on either side
    left: np.ndarray  # log seconds of the left boundary
    right: np.ndarray  # log seconds of the right boundary


def log_density(x, *, bandwidth="silverman", grid_size=1024):
    """Gaussian kernel density estimate of log times.

    Takes raw times and works on their logs, where one bandwidth suits both
    the narrow floor and the long tail. ``bandwidth`` is "silverman",
    "scott", "isj" (improved Sheather-Jones, Botev et al. 2010), or a number
    on the log scale. Returns (grid, density) with the grid in log seconds.
    Needs KDEpy (``benchstats[kde]``).

    The default is Silverman's rule, not ISJ. Timing data have a hard floor,
    a sharp edge in the density, and ISJ answers a sharp edge with a
    bandwidth tens of times smaller than Silverman's, which splits one floor
    into dozens of spurious modes. Use ``mode_count_scan`` to check that a
    mode persists at smaller bandwidths.
    """
    x = _as_sample(x, positive=True, min_size=2)
    try:
        from KDEpy import FFTKDE
    except ImportError as error:
        raise ImportError("log_density needs KDEpy: pip install 'benchstats[kde]'") from error
    bw = {"isj": "ISJ", "silverman": "silverman", "scott": "scott"}.get(bandwidth, bandwidth)
    if isinstance(bw, str) and bw not in ("ISJ", "silverman", "scott"):
        raise ValueError("bandwidth: must be 'isj', 'silverman', 'scott' or a positive number")
    if not isinstance(bw, str) and not bw > 0:
        raise ValueError("bandwidth: must be positive")
    grid, density = FFTKDE(kernel="gaussian", bw=bw).fit(np.log(x)).evaluate(grid_size)
    return grid, density


def find_modes(grid, density, *, min_prominence):
    """Local maxima of a density with a prominence floor, and the mass of each.

    ``min_prominence`` is a fraction of the highest density value. Each
    mode's mass is the integral of the density between the minima on either
    side, as a fraction of the integral over the whole grid. Works on any
    scale; with ``log_density`` the locations are log seconds.
    """
    grid = _as_sample(grid, "grid", min_size=3)
    density = _as_sample(density, "density", min_size=3)
    if grid.size != density.size:
        raise ValueError("grid and density must have the same length")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("grid: must be increasing")
    if not 0 <= min_prominence < 1:
        raise ValueError("min_prominence: must be a fraction in [0, 1)")
    peaks, _ = signal.find_peaks(np.r_[0.0, density, 0.0], prominence=min_prominence * density.max())
    peaks = peaks - 1  # undo the zero padding, which lets peaks sit at the ends
    bounds = [0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        bounds.append(a + int(np.argmin(density[a : b + 1])))
    bounds.append(grid.size - 1)
    total = np.trapezoid(density, grid)
    mass = np.array([np.trapezoid(density[lo : hi + 1], grid[lo : hi + 1]) for lo, hi in zip(bounds[:-1], bounds[1:])])
    return Modes(grid[peaks], density[peaks], mass / total, grid[bounds[:-1]], grid[bounds[1:]])


def mode_count_scan(x, bandwidths, *, min_prominence, grid_size=1024):
    """Number of modes of the log density at each bandwidth (log scale).

    Takes raw times. A mode that persists over a range of bandwidths is more
    credible than one seen at a single bandwidth. Needs KDEpy.
    """
    counts = []
    for bw in bandwidths:
        grid, density = log_density(x, bandwidth=float(bw), grid_size=grid_size)
        counts.append(find_modes(grid, density, min_prominence=min_prominence).location.size)
    return np.array(counts)


def dip_test(x):
    """Hartigan's dip statistic and its p-value, as (dip, p_value).

    Works on the scale given; pass log times to match ``log_density``. The
    p-value is computed against the uniform distribution, the least
    favorable unimodal case, so other unimodal shapes are rejected less
    often than the nominal level. Needs the GPL-licensed diptest package
    (``benchstats[diptest]``).
    """
    x = _as_sample(x, min_size=4)
    try:
        import diptest
    except ImportError as error:
        raise ImportError("dip_test needs the diptest package: pip install 'benchstats[diptest]'") from error
    dip, p_value = diptest.diptest(x)
    return float(dip), float(p_value)
