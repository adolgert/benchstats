"""Resampling: bootstrap confidence intervals.

All functions take data as a tuple of samples and a ``statistic(*samples)``,
the form scipy.stats.bootstrap uses, so a two-sample statistic such as a
ratio of medians needs no special code. A statistic that accepts an ``axis``
keyword is evaluated on many resamples at once; otherwise it is called once
per resample.

- ``bootstrap_ci``: independent draws (wraps scipy.stats.bootstrap).
- ``block_bootstrap_ci``: correlated draws (stationary bootstrap).
- ``hierarchical_bootstrap_ci``: draws grouped by invocation or build.

Every interval reports the Monte Carlo standard error of its endpoints.

This module imports nothing from the rest of the package.
"""

import inspect
import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

N_BATCHES = 10  # batches of resamples for the Monte Carlo error of endpoints
_CHUNK = 2_000_000  # values per chunk when evaluating resamples together


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    confidence: float
    method: str
    n_resamples: int
    mc_se_low: float  # Monte Carlo standard error of low
    mc_se_high: float  # Monte Carlo standard error of high


# ---------------------------------------------------------------- input checks


def _as_samples(data, *, min_size=2):
    if not isinstance(data, tuple | list) or len(data) == 0:
        raise ValueError("data: must be a non-empty tuple of samples")
    out = []
    for i, sample in enumerate(data):
        sample = np.asarray(sample, dtype=np.float64)
        if sample.ndim != 1:
            raise ValueError(f"data[{i}]: must be 1-D")
        if sample.size < min_size:
            raise ValueError(f"data[{i}]: needs at least {min_size} values")
        if not np.all(np.isfinite(sample)):
            raise ValueError(f"data[{i}]: all values must be finite")
        out.append(sample)
    return tuple(out)


def _check_common(confidence, n_resamples, rng):
    if not 0 < confidence < 1:
        raise ValueError("confidence: must lie strictly between 0 and 1")
    if int(n_resamples) < 2 * N_BATCHES:
        raise ValueError(f"n_resamples: must be at least {2 * N_BATCHES}")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng: must be a numpy.random.Generator")


def _takes_axis(statistic):
    try:
        return "axis" in inspect.signature(statistic).parameters
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- shared pieces


def _evaluate(statistic, resampled):
    """statistic on each row of the 2-D resampled arrays, as a 1-D array."""
    if _takes_axis(statistic):
        return np.asarray(statistic(*resampled, axis=-1), dtype=float).reshape(-1)
    n = resampled[0].shape[0]
    return np.array([statistic(*(r[i] for r in resampled)) for i in range(n)], dtype=float)


def _bootstrap_distribution(n_resamples, row_values, draw):
    """Bootstrap distribution, made chunk by chunk to bound memory.

    ``draw(rows)`` resamples ``rows`` times and returns the statistic for
    each; ``row_values`` is the number of values in one resample of all
    samples together.
    """
    chunk = max(1, _CHUNK // max(row_values, 1))
    out = []
    done = 0
    while done < n_resamples:
        rows = min(chunk, n_resamples - done)
        out.append(draw(rows))
        done += rows
    return np.concatenate(out)


def _percentile_interval(boot, confidence):
    tail = (1 - confidence) / 2
    low, high = np.quantile(boot, [tail, 1 - tail])
    return float(low), float(high), tail, 1 - tail


def _endpoint_mc_se(boot, levels):
    """Monte Carlo standard error of quantiles of the bootstrap distribution.

    Splits the resamples into N_BATCHES batches, takes each batch's quantile
    at each level, and divides the spread of the batch quantiles by
    sqrt(N_BATCHES).
    """
    batches = np.array_split(boot, N_BATCHES)
    q = np.array([np.quantile(b, levels) for b in batches])
    return tuple(float(v) for v in q.std(axis=0, ddof=1) / math.sqrt(N_BATCHES))


def _attained_level(boot, value):
    """Fraction of the bootstrap distribution at or below value, kept off 0 and 1."""
    level = np.mean(boot <= value)
    edge = 0.5 / boot.size
    return float(min(max(level, edge), 1 - edge))


# ---------------------------------------------------------------- independent draws


def bootstrap_ci(data, statistic, *, confidence=0.95, n_resamples=9999, method="BCa", paired=False, rng):
    """Bootstrap confidence interval for independent draws.

    Wraps scipy.stats.bootstrap, which offers "BCa", "percentile" and
    "basic" intervals. ``paired=True`` resamples the samples together, row
    by row. Works on whatever scale the statistic expects.
    """
    data = _as_samples(data)
    _check_common(confidence, n_resamples, rng)
    if method not in ("BCa", "percentile", "basic"):
        raise ValueError("method: must be 'BCa', 'percentile' or 'basic'")
    vectorized = _takes_axis(statistic)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.DegenerateDataWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        result = stats.bootstrap(
            data,
            statistic,
            n_resamples=int(n_resamples),
            vectorized=vectorized,
            paired=paired,
            confidence_level=confidence,
            method=method,
            rng=rng,
        )
    low, high = float(result.confidence_interval.low), float(result.confidence_interval.high)
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"the {method} interval cannot be computed for these data (degenerate resamples); use more draws")
    boot = np.asarray(result.bootstrap_distribution, dtype=float).reshape(-1)
    estimate = float(_evaluate(statistic, tuple(d[np.newaxis, :] for d in data))[0])
    if method == "basic":
        levels = (_attained_level(boot, 2 * estimate - high), _attained_level(boot, 2 * estimate - low))
        se_high, se_low = _endpoint_mc_se(boot, levels)
    else:
        levels = (_attained_level(boot, low), _attained_level(boot, high))
        se_low, se_high = _endpoint_mc_se(boot, levels)
    return ConfidenceInterval(estimate, low, high, confidence, method, int(n_resamples), se_low, se_high)


# ---------------------------------------------------------------- correlated draws


def stationary_indices(n, block_length, n_resamples, rng, *, size=None):
    """Index matrix (n_resamples, size) for the stationary bootstrap of a length-n series.

    Politis and Romano (1994): blocks start at uniform random positions,
    have geometric lengths with mean ``block_length``, and wrap around the
    end of the series. ``size`` defaults to n.
    """
    if block_length < 1:
        raise ValueError("block_length: must be at least 1")
    size = n if size is None else int(size)
    p_new = 1.0 / block_length
    new_block = rng.random((n_resamples, size)) < p_new
    new_block[:, 0] = True
    starts = rng.integers(0, n, size=(n_resamples, size))
    position = np.broadcast_to(np.arange(size), (n_resamples, size))
    last_start = np.maximum.accumulate(np.where(new_block, position, 0), axis=1)
    start_value = np.take_along_axis(starts, last_start, axis=1)
    return (start_value + position - last_start) % n


def block_bootstrap_ci(data, statistic, *, block_length, confidence=0.95, n_resamples=9999, paired=False, rng):
    """Percentile interval from the stationary bootstrap, for correlated draws.

    Each sample is resampled in blocks of random length with mean
    ``block_length``, keeping run order within blocks. With ``paired=True``
    all samples share the same indices. Percentile intervals only: BCa is
    defined for independent resampling. Choose ``block_length`` with
    ``block_length(x)``.
    """
    data = _as_samples(data)
    _check_common(confidence, n_resamples, rng)
    if paired and len({d.size for d in data}) != 1:
        raise ValueError("data: paired samples must have the same length")
    row_values = sum(d.size for d in data)

    def draw(rows):
        if paired:
            idx = stationary_indices(data[0].size, block_length, rows, rng)
            resampled = tuple(d[idx] for d in data)
        else:
            resampled = tuple(d[stationary_indices(d.size, block_length, rows, rng)] for d in data)
        return _evaluate(statistic, resampled)

    boot = _bootstrap_distribution(int(n_resamples), row_values, draw)
    estimate = float(_evaluate(statistic, tuple(d[np.newaxis, :] for d in data))[0])
    low, high, q_low, q_high = _percentile_interval(boot, confidence)
    se_low, se_high = _endpoint_mc_se(boot, (q_low, q_high))
    return ConfidenceInterval(estimate, low, high, confidence, "stationary percentile", int(n_resamples), se_low, se_high)


def block_length(x):
    """Block length for the stationary bootstrap (Politis and White 2004).

    Includes the correction of Patton, Politis and White (2009). This is a
    port of arch.bootstrap.optimal_block_length (stationary column),
    including its choice of the tuning lag m: the first lag that starts a run
    of k_n autocorrelations inside +/- 2 sqrt(log10(n)/n). Works on the
    scale given; pass log times so detours do not dominate. Returns a float
    of at least 1.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 10:
        raise ValueError("x: must be 1-D with at least 10 values")
    if not np.all(np.isfinite(x)):
        raise ValueError("x: all values must be finite")
    n = x.size
    eps = x - x.mean()
    b_max = math.ceil(min(3 * math.sqrt(n), n / 3))
    kn = max(5, int(math.log10(n)))
    m_max = math.ceil(math.sqrt(n)) + kn
    band = 2 * math.sqrt(math.log10(n) / n)
    acv = np.array([eps[k:] @ eps[: n - k] for k in range(m_max + 1)]) / n
    abs_acorr = np.array(
        [
            abs(eps[k:] @ eps[: n - k]) / math.sqrt((eps[k + 1 :] @ eps[k + 1 :]) * (eps[: n - k - 1] @ eps[: n - k - 1]))
            for k in range(m_max + 1)
        ]
    )
    opt_m = None
    for i in range(kn, m_max + 1):
        if np.all(abs_acorr[i - kn : i] < band):
            opt_m = i - kn
            break
    m = min(2 * max(opt_m, 1) if opt_m is not None else m_max, m_max)
    k = np.arange(1, m + 1)
    lam = np.where(k / m <= 0.5, 1.0, 2 * (1 - k / m))
    g = float(np.sum(2 * lam * k * acv[1 : m + 1]))
    long_run = float(acv[0] + np.sum(2 * lam * acv[1 : m + 1]))
    if long_run <= 0:
        return 1.0
    b = (2 * g * g / (2 * long_run * long_run)) ** (1 / 3) * n ** (1 / 3)
    return float(max(1.0, min(b, b_max)))


# ---------------------------------------------------------------- hierarchical draws


def _group_rows(sample, labels):
    """Split a sample into its groups, keeping the order within each group."""
    labels = np.asarray(labels)
    if labels.shape != sample.shape:
        raise ValueError("groups: each label array must align with its sample")
    if labels.dtype.kind not in "iu":
        raise ValueError("groups: labels must be integers")
    idx = np.argsort(labels, kind="stable")
    _, starts = np.unique(labels[idx], return_index=True)
    return np.split(sample[idx], starts[1:])


def _hierarchical_draw(groups, rows, resample_within, within_block, rng):
    """One sample's resamples: whole groups with replacement, then optionally within."""
    n_groups = len(groups)
    sizes = {g.size for g in groups}
    chosen = rng.integers(0, n_groups, size=(rows, n_groups))
    if len(sizes) == 1:
        matrix = np.stack(groups)  # (n_groups, m)
        m = matrix.shape[1]
        picked = matrix[chosen]  # (rows, n_groups, m)
        if resample_within == "iid":
            inner = rng.integers(0, m, size=picked.shape)
            picked = np.take_along_axis(picked, inner, axis=2)
        elif resample_within == "blocks":
            inner = stationary_indices(m, within_block, rows * n_groups, rng).reshape(picked.shape)
            picked = np.take_along_axis(picked, inner, axis=2)
        return picked.reshape(rows, n_groups * m)
    # Unequal group sizes give resamples of different lengths: one list entry each.
    out = []
    for r in range(rows):
        parts = []
        for g in chosen[r]:
            values = groups[g]
            if resample_within == "iid":
                values = values[rng.integers(0, values.size, size=values.size)]
            elif resample_within == "blocks":
                values = values[stationary_indices(values.size, within_block, 1, rng)[0]]
            parts.append(values)
        out.append(np.concatenate(parts))
    return out


def hierarchical_bootstrap_ci(
    data,
    groups,
    statistic,
    *,
    resample_within="iid",
    within_block_length=None,
    confidence=0.95,
    n_resamples=9999,
    rng,
):
    """Percentile interval that resamples whole groups (invocations or builds).

    ``groups`` holds one integer label array per sample, aligned with it,
    such as ``group_labels(m, "invocation")``. Groups are resampled with
    replacement. ``resample_within`` is "iid" (resample draws within each
    chosen group), "blocks" (stationary bootstrap within each group, mean
    block length ``within_block_length``) or None (keep each group whole).
    Each sample's groups are resampled independently.

    The default comes from the coverage tests: for the median with 10 and
    30 invocations of 20 draws, "iid" covered 93% and 94% at 95% nominal,
    and None covered 90% and 93%.
    """
    data = _as_samples(data)
    _check_common(confidence, n_resamples, rng)
    if not isinstance(groups, tuple | list) or len(groups) != len(data):
        raise ValueError("groups: must hold one label array per sample")
    if resample_within not in (None, "iid", "blocks"):
        raise ValueError("resample_within: must be None, 'iid' or 'blocks'")
    if resample_within == "blocks" and within_block_length is None:
        raise ValueError("within_block_length: needed when resample_within='blocks'")
    split = [_group_rows(d, g) for d, g in zip(data, groups)]
    for i, parts in enumerate(split):
        if len(parts) < 2:
            raise ValueError(f"groups[{i}]: needs at least two groups")
    vectorized = all(len({p.size for p in parts}) == 1 for parts in split) and _takes_axis(statistic)
    row_values = sum(d.size for d in data)

    def draw(rows):
        resampled = [_hierarchical_draw(parts, rows, resample_within, within_block_length, rng) for parts in split]
        if vectorized:
            return _evaluate(statistic, tuple(resampled))
        return np.array([statistic(*(np.asarray(r[i]) for r in resampled)) for i in range(rows)], dtype=float)

    boot = _bootstrap_distribution(int(n_resamples), row_values, draw)
    estimate = float(_evaluate(statistic, tuple(d[np.newaxis, :] for d in data))[0])
    low, high, q_low, q_high = _percentile_interval(boot, confidence)
    se_low, se_high = _endpoint_mc_se(boot, (q_low, q_high))
    method = "hierarchical percentile" + (f", {resample_within} within" if resample_within else "")
    return ConfidenceInterval(estimate, low, high, confidence, method, int(n_resamples), se_low, se_high)
