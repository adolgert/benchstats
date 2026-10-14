"""Decision: compare two versions, and plan how many runs a comparison needs.

``compare`` turns a Measurements record into a statement such as "B is 4.1%
slower than A (95% CI +2.3% to +5.9%)" and a verdict judged against an
effect size delta chosen before the data are seen. ``plan_runs`` uses pilot
data to find how many runs give power 1 - beta against delta.

This is the only core module that imports the others.
"""

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy import optimize, stats

from benchstats import characterization, collection, resampling

DESIGNS = ("independent", "blocks", "hierarchical", "paired")
EFFECTS = ("ratio", "difference")


class DesignWarning(UserWarning):
    """The data look inconsistent with the chosen design."""


# ---------------------------------------------------------------- statistics


@dataclass(frozen=True)
class Statistic:
    """A summary of one variant's raw times, evaluated along an axis.

    ``function(x, axis=-1)`` takes raw times in seconds. ``effect`` is the
    default effect scale: "ratio" for location-like statistics, "difference"
    for shape statistics. ``log_value`` says the function already returns a
    log, so a difference of values is a log ratio.
    """

    name: str
    function: Callable
    effect: str
    log_value: bool = False


def median_time():
    """The median of log time; its effect is the log ratio of medians (default)."""
    return Statistic("median", lambda x, axis=-1: np.median(np.log(x), axis=axis), "ratio", log_value=True)


def quantile_time(q):
    """A high quantile of time (tail latency), compared as a ratio."""
    if not 0 < q < 1:
        raise ValueError("q: must lie strictly between 0 and 1")
    return Statistic(f"{100 * q:g}th percentile", lambda x, axis=-1: np.quantile(np.log(x), q, axis=axis), "ratio", log_value=True)


def exceedance_rate(threshold):
    """The fraction of draws above a fixed threshold in seconds, compared as a ratio."""
    return Statistic(f"rate above {threshold:g} s", lambda x, axis=-1: np.mean(x > threshold, axis=axis), "ratio")


def mean_excess(threshold):
    """The mean of x - threshold over draws above a fixed threshold, compared as a ratio."""

    def function(x, axis=-1):
        above = x > threshold
        return np.sum(np.where(above, x - threshold, 0.0), axis=axis) / np.sum(above, axis=axis)

    return Statistic(f"mean excess over {threshold:g} s", function, "ratio")


def tail_shape(threshold):
    """The generalized Pareto shape xi of excesses over a fixed threshold, compared as a difference."""

    def one(row):
        return characterization.gpd_fit_lmoments(row[row > threshold] - threshold)[0]

    def function(x, axis=-1):
        return np.apply_along_axis(one, axis, x)

    return Statistic(f"xi above {threshold:g} s", function, "difference")


def l_kurtosis():
    """The L-kurtosis tau4 of raw times, compared as a difference."""
    return Statistic("L-kurtosis", lambda x, axis=-1: stats.lmoment(x, order=4, axis=axis), "difference")


def _effect_function(statistic, effect):
    """Two-sample statistic: candidate minus baseline on the effect scale."""
    f = statistic.function
    take_log = effect == "ratio" and not statistic.log_value

    def effect_of(a, b, axis=-1):
        va, vb = f(a, axis=axis), f(b, axis=axis)
        return np.log(vb) - np.log(va) if take_log else vb - va

    return effect_of


# ---------------------------------------------------------------- verdicts


class Outcome(StrEnum):
    SLOWER = "slower"
    FASTER = "faster"
    NO_CHANGE = "no change"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    less_than_delta: bool  # only for slower or faster: the whole interval is inside +/- d


def effect_threshold(delta, effect="ratio"):
    """d = log(1 + delta) for ratios and d = delta for differences."""
    if effect not in EFFECTS:
        raise ValueError(f"effect: must be one of {EFFECTS}")
    if not delta > 0:
        raise ValueError("delta: must be positive")
    return math.log1p(delta) if effect == "ratio" else float(delta)


def verdict(low, high, delta, *, effect="ratio"):
    """Turn an interval [low, high] for the effect into one of four verdicts.

    With d = effect_threshold(delta, effect):

    - low > 0: slower; flagged "by less than delta" if high < d.
    - high < 0: faster; flagged "by less than delta" if low > -d.
    - -d < low <= 0 <= high < d: no change as large as delta.
    - anything else: inconclusive, more runs needed.

    For "difference" effects, "slower" means the candidate's statistic is larger.
    """
    if not low <= high:
        raise ValueError("need low <= high")
    d = effect_threshold(delta, effect)
    if low > 0:
        return Verdict(Outcome.SLOWER, high < d)
    if high < 0:
        return Verdict(Outcome.FASTER, low > -d)
    if -d < low and high < d:
        return Verdict(Outcome.NO_CHANGE, False)
    return Verdict(Outcome.INCONCLUSIVE, False)


# ---------------------------------------------------------------- compare


@dataclass(frozen=True)
class Comparison:
    benchmark: str
    baseline: str
    candidate: str
    statistic: str
    effect: str
    design: str
    estimate: float  # candidate minus baseline, on the effect scale
    low: float
    high: float
    confidence: float
    delta: float
    verdict: Verdict
    statement: str
    warnings: tuple[str, ...]
    n_baseline: int
    n_candidate: int
    mc_se_low: float
    mc_se_high: float


def _percent(log_ratio):
    return 100.0 * math.expm1(log_ratio)


def _statement(baseline, candidate, statistic, effect, estimate, low, high, confidence, delta, v):
    level = f"{100 * confidence:g}% CI"
    if effect == "ratio":
        pct = _percent(estimate)
        direction = "slower" if pct >= 0 else "faster"
        head = f"{candidate} is {abs(pct):.1f}% {direction} than {baseline} ({level} {_percent(low):+.1f}% to {_percent(high):+.1f}%)"
        size = f"{100 * delta:g}%"
    else:
        direction = "higher" if estimate >= 0 else "lower"
        head = f"{statistic} is {abs(estimate):.3g} {direction} in {candidate} than in {baseline} ({level} {low:+.3g} to {high:+.3g})"
        size = f"{delta:g}"
    tail = {
        Outcome.SLOWER: "slower" + (f", by less than {size}" if v.less_than_delta else ""),
        Outcome.FASTER: "faster" + (f", by less than {size}" if v.less_than_delta else ""),
        Outcome.NO_CHANGE: f"no change as large as {size}",
        Outcome.INCONCLUSIVE: "inconclusive, more runs needed",
    }[v.outcome]
    return f"{head}. Verdict: {tail}."


def _adjacent_pairs(m, baseline, candidate, include_warmup):
    """Indices into flatten(baseline) and flatten(candidate) of adjacent pairs in run order."""
    order_a = collection.run_order(m, variant=baseline, include_warmup=include_warmup)
    order_b = collection.run_order(m, variant=candidate, include_warmup=include_warmup)
    order = np.concatenate([order_a, order_b])
    is_b = np.r_[np.zeros(order_a.size, bool), np.ones(order_b.size, bool)]
    position = np.r_[np.arange(order_a.size), np.arange(order_b.size)]
    idx = np.argsort(order, kind="stable")
    is_b, position = is_b[idx], position[idx]
    pairs_a, pairs_b = [], []
    i = 0
    while i < idx.size - 1:
        if is_b[i] != is_b[i + 1]:
            a, b = (i, i + 1) if not is_b[i] else (i + 1, i)
            pairs_a.append(position[a])
            pairs_b.append(position[b])
            i += 2
        else:
            i += 1
    return np.array(pairs_a, dtype=int), np.array(pairs_b, dtype=int)


def _dependence_note(name, x):
    """A warning text if x is clearly correlated in run order, else None.

    Clear means both: the lag-1 rank autocorrelation is more than three null
    standard errors above zero, and tau_int is above 1.5.
    """
    if x.size < 20 or np.all(x == x[0]):
        return None
    rho1 = characterization.rank_autocorrelation(x, 1)[0]
    tau = characterization.integrated_autocorrelation_time(x)
    if rho1 * math.sqrt(x.size) > 3 and tau > 1.5:
        return f"{name} are correlated in run order (tau_int = {tau:.1f}); consider design='blocks'"
    return None


def _design_notes(m, design, level, variants, series, include_warmup):
    notes = []
    if design in ("independent", "paired"):
        for name, x in series.items():
            note = _dependence_note(name, x)
            if note:
                notes.append(note)
    if design != "hierarchical":
        for name in variants:
            n_groups = int(collection.group_labels(m, "invocation", variant=name, include_warmup=include_warmup).max()) + 1
            if n_groups > 1:
                notes.append(f"{name} has {n_groups} invocations; consider design='hierarchical'")
    if design == "hierarchical":
        for name in variants:
            n_groups = int(collection.group_labels(m, level, variant=name, include_warmup=include_warmup).max()) + 1
            if n_groups < 10:
                notes.append(f"{name} has only {n_groups} {level}s; hierarchical intervals may under-cover")
    return notes


def compare(
    m,
    *,
    baseline,
    candidate,
    delta,
    statistic=None,
    effect=None,
    confidence=0.95,
    design="independent",
    level="invocation",
    resample_within="iid",
    block_length=None,
    n_resamples=9999,
    include_warmup=False,
    rng,
):
    """Compare a candidate variant with a baseline variant in one record.

    ``statistic`` is a Statistic, by default ``median_time()``, whose effect
    is the log ratio of medians, reported as a percent change. ``effect`` is
    "ratio" or "difference" and defaults to the statistic's own. ``delta``
    is the effect size that matters, chosen before the data are seen: a
    fraction such as 0.02 for ratios, or a value on the statistic's scale
    for differences.

    ``design`` is chosen by the caller:

    - "independent": BCa bootstrap of independent draws.
    - "blocks": stationary bootstrap for draws correlated in run order;
      ``block_length`` defaults to the larger Politis-White choice of the
      two variants' log times.
    - "hierarchical": resamples whole groups (``level`` "invocation" or
      "build") and then draws within them (``resample_within``: "iid",
      "blocks" or None).
    - "paired": for interleaved runs; each baseline draw is matched with the
      adjacent candidate draw in run order and pairs are resampled together.

    Warns with DesignWarning when the data contradict the design, and never
    switches designs.
    """
    if design not in DESIGNS:
        raise ValueError(f"design: must be one of {DESIGNS}")
    statistic = statistic or median_time()
    effect = effect or statistic.effect
    effect_threshold(delta, effect)  # checks delta and effect
    a = collection.flatten(m, variant=baseline, include_warmup=include_warmup)
    b = collection.flatten(m, variant=candidate, include_warmup=include_warmup)
    if a.size < 2 or b.size < 2:
        raise ValueError("each variant needs at least two draws")
    effect_of = _effect_function(statistic, effect)
    series = {f"{baseline} draws": a, f"{candidate} draws": b}

    if design == "independent":
        ci = resampling.bootstrap_ci((a, b), effect_of, confidence=confidence, n_resamples=n_resamples, rng=rng)
    elif design == "blocks":
        length = block_length or max(resampling.block_length(np.log(a)), resampling.block_length(np.log(b)))
        ci = resampling.block_bootstrap_ci((a, b), effect_of, block_length=length, confidence=confidence, n_resamples=n_resamples, rng=rng)
    elif design == "hierarchical":
        groups = tuple(collection.group_labels(m, level, variant=v, include_warmup=include_warmup) for v in (baseline, candidate))
        if any(g.max() < 1 for g in groups):
            raise ValueError(f"design 'hierarchical' needs at least two {level}s per variant")
        ci = resampling.hierarchical_bootstrap_ci(
            (a, b),
            groups,
            effect_of,
            resample_within=resample_within,
            within_block_length=block_length,
            confidence=confidence,
            n_resamples=n_resamples,
            rng=rng,
        )
    else:
        ia, ib = _adjacent_pairs(m, baseline, candidate, include_warmup)
        if ia.size < 2:
            raise ValueError("design 'paired' needs interleaved runs: fewer than two adjacent pairs found")
        a, b = a[ia], b[ib]
        series = {"the paired differences": np.log(b) - np.log(a)}
        ci = resampling.bootstrap_ci((a, b), effect_of, confidence=confidence, n_resamples=n_resamples, paired=True, rng=rng)

    if not all(np.isfinite([ci.estimate, ci.low, ci.high])):
        raise ValueError(f"the {statistic.name} effect is not finite on these data; try effect='difference'")
    notes = _design_notes(m, design, level, (baseline, candidate), series, include_warmup)
    for note in notes:
        warnings.warn(note, DesignWarning, stacklevel=2)
    v = verdict(ci.low, ci.high, delta, effect=effect)
    text = _statement(baseline, candidate, statistic.name, effect, ci.estimate, ci.low, ci.high, confidence, delta, v)
    return Comparison(
        benchmark=m.benchmark,
        baseline=baseline,
        candidate=candidate,
        statistic=statistic.name,
        effect=effect,
        design=design,
        estimate=ci.estimate,
        low=ci.low,
        high=ci.high,
        confidence=confidence,
        delta=delta,
        verdict=v,
        statement=text,
        warnings=tuple(notes),
        n_baseline=int(a.size),
        n_candidate=int(b.size),
        mc_se_low=ci.mc_se_low,
        mc_se_high=ci.mc_se_high,
    )


def bonferroni_confidence(confidence, k):
    """Per-comparison confidence 1 - (1 - confidence) / k for a suite of k comparisons."""
    if k < 1:
        raise ValueError("k: must be at least 1")
    return 1 - (1 - confidence) / k


def compare_suite(records, *, confidence=0.95, **kwargs):
    """Compare K benchmarks, widening each interval to confidence 1 - alpha/K (Bonferroni).

    The chance of at least one false verdict across the suite is then at
    most 1 - confidence. Other keyword arguments go to ``compare``.
    """
    records = list(records)
    level = bonferroni_confidence(confidence, len(records))
    return [compare(m, confidence=level, **kwargs) for m in records]


# ---------------------------------------------------------------- variance components


@dataclass(frozen=True)
class VarianceComponents:
    between_build: float  # standard deviations of log time; NaN if not estimable
    between_invocation: float
    within_invocation: float
    n_builds: int
    n_invocations: int
    iterations_per_invocation: float  # median


_MAD_SCALE = 1.4826022185056018


def _mad_sd(values):
    values = np.asarray(values, dtype=float)
    return _MAD_SCALE * float(np.median(np.abs(values - np.median(values))))


def _group_medians(y, labels):
    return np.array([np.median(y[labels == g]) for g in range(int(labels.max()) + 1)])


def variance_components(m, *, variant=None, include_warmup=False):
    """Between-build, between-invocation and within-invocation spread of log time.

    Standard deviations from medians and scaled MADs, so detours pull less.
    The spread of group medians includes the noise of the medians themselves;
    that part is subtracted with the normal approximation, where a median of
    J draws has variance (pi/2) s^2 / J. A component that cannot be estimated
    (one build, or one invocation per build) is NaN.
    """
    y = np.log(collection.flatten(m, variant=variant, include_warmup=include_warmup))
    inv = collection.group_labels(m, "invocation", variant=variant, include_warmup=include_warmup)
    bld = collection.group_labels(m, "build", variant=variant, include_warmup=include_warmup)
    n_inv = int(inv.max()) + 1
    n_bld = int(bld.max()) + 1
    sizes = np.bincount(inv)
    inv_median = _group_medians(y, inv)
    within = _mad_sd(y - inv_median[inv])

    inv_build = np.array([bld[inv == g][0] for g in range(n_inv)])
    per_build = np.bincount(inv_build, minlength=n_bld)
    noise_of_median = (math.pi / 2) * within**2 / float(np.median(sizes))
    if np.any(per_build >= 2):
        build_of_inv_median = _group_medians(inv_median, inv_build)
        spread = _mad_sd(inv_median - build_of_inv_median[inv_build])
        between_inv = math.sqrt(max(0.0, spread**2 - noise_of_median))
    else:
        build_of_inv_median = inv_median
        between_inv = math.nan
    if n_bld >= 2:
        noise = (math.pi / 2) * ((0.0 if math.isnan(between_inv) else between_inv**2) + noise_of_median)
        noise /= float(np.median(per_build))
        between_bld = math.sqrt(max(0.0, _mad_sd(build_of_inv_median) ** 2 - noise))
    else:
        between_bld = math.nan
    return VarianceComponents(between_bld, between_inv, within, n_bld, n_inv, float(np.median(sizes)))


def allocate(components, *, cost_per_invocation, cost_per_iteration, max_iterations=10_000):
    """Iterations per invocation from two-stage sampling (Cochran 1977).

    J* = sqrt(c_inv s_within^2 / (c_iter s_between^2)), rounded and at least
    1. The invocation cost includes process start and warmup. When the
    between-invocation spread is zero, returns ``max_iterations``.
    """
    if not (cost_per_invocation > 0 and cost_per_iteration > 0):
        raise ValueError("costs must be positive")
    s_w, s_b = components.within_invocation, components.between_invocation
    if math.isnan(s_b):
        raise ValueError("components: between-invocation spread is not estimable; the pilot needs several invocations")
    if s_b == 0:
        return int(max_iterations)
    j = math.sqrt(cost_per_invocation * s_w**2 / (cost_per_iteration * s_b**2))
    return int(min(max_iterations, max(1, round(j))))


# ---------------------------------------------------------------- planning runs


def uniform_slowdown(delta):
    """Effect model: multiply every candidate time by 1 + delta."""
    return lambda x, rng: x * (1 + delta)


def added_detours(rate, mean):
    """Effect model: add an exponential detour with the given mean to a fraction ``rate`` of draws."""
    if not 0 < rate <= 1 or not mean > 0:
        raise ValueError("need 0 < rate <= 1 and mean > 0")
    return lambda x, rng: x + (rng.random(x.size) < rate) * rng.exponential(mean, size=x.size)


@dataclass(frozen=True)
class RunPlan:
    runs: int  # draws per variant, or invocations per variant for "hierarchical"
    unit: str  # "draws" or "invocations"
    iterations_per_invocation: int  # 1 unless the design is hierarchical
    power: float  # estimated power at ``runs``
    power_se: float  # Monte Carlo standard error of ``power``
    target_power: float
    history: tuple[tuple[int, float], ...]  # (runs, estimated power) for each size tried


def _simulated_record(baseline, candidate, *, invocation=None, iteration=None, interleave=False):
    n_a, n_b = baseline.size, candidate.size
    variant = np.r_[np.full(n_a, "A"), np.full(n_b, "B")]
    if interleave:
        order = np.r_[2 * np.arange(n_a), 2 * np.arange(n_b) + 1]
    else:
        order = np.arange(n_a + n_b)
    return collection.make_measurements(
        np.r_[baseline, candidate],
        variant=variant,
        invocation=invocation,
        iteration=iteration,
        order=order,
        benchmark="simulated",
        created="simulated",
    )


def _simulate(pilot_series, groups, design, runs, iterations, block, effect_model, rng):
    """One simulated experiment of the given size, resampled from the pilot."""
    n = pilot_series.size
    if design == "hierarchical":
        chosen = rng.integers(0, len(groups), size=2 * runs)
        draws = np.concatenate([groups[g][rng.integers(0, groups[g].size, size=iterations)] for g in chosen])
        half = runs * iterations
        a, b = draws[:half], effect_model(draws[half:], rng)
        invocation = np.repeat(np.r_[np.arange(runs), np.arange(runs)], iterations)
        iteration = np.tile(np.arange(iterations), 2 * runs)
        return _simulated_record(a, b, invocation=invocation, iteration=iteration)
    if design == "blocks":
        idx = resampling.stationary_indices(n, block, 2, rng, size=runs)
        a, b = pilot_series[idx[0]], effect_model(pilot_series[idx[1]], rng)
        return _simulated_record(a, b)
    a = pilot_series[rng.integers(0, n, size=runs)]
    b = effect_model(pilot_series[rng.integers(0, n, size=runs)], rng)
    return _simulated_record(a, b, interleave=design == "paired")


def _size_from_power_curve(history, n_sim, target, *, fallback):
    """Smallest size where the fitted curve Phi(a + b sqrt(size)) reaches target.

    For a test statistic that is roughly normal, power is
    Phi(delta sqrt(n) / sigma - z), so this curve has the right shape.
    """
    sizes = np.array(sorted(history), dtype=float)
    hits = np.array([history[int(k)] for k in sizes]) * n_sim

    def negative_log_likelihood(params):
        p = np.clip(stats.norm.cdf(params[0] + params[1] * np.sqrt(sizes)), 1e-12, 1 - 1e-12)
        return -np.sum(hits * np.log(p) + (n_sim - hits) * np.log1p(-p))

    fit = optimize.minimize(negative_log_likelihood, x0=[-2.0, 0.2], method="Nelder-Mead")
    a, b = fit.x
    if not fit.success or b <= 0:
        return int(fallback)
    return max(2, math.ceil(((stats.norm.ppf(target) - a) / b) ** 2))


def plan_runs(
    pilot,
    *,
    delta,
    statistic=None,
    effect=None,
    alpha=0.05,
    beta=0.2,
    design="independent",
    effect_model=None,
    n_sim=200,
    n_resamples=999,
    iterations_per_invocation=None,
    start=16,
    max_runs=1_000_000,
    variant=None,
    rng,
):
    """How many runs give power 1 - beta to call a true slowdown of delta "slower".

    Power is estimated by simulation from the pilot. Each simulated
    experiment resamples the pilot with the same design, applies
    ``effect_model`` (default: multiply times by 1 + delta) to make a
    candidate, runs ``compare`` at confidence 1 - alpha, and records whether
    the verdict is "slower". The search doubles the number of runs from
    ``start`` until the power reaches 1 - beta, adds two sizes inside the
    last doubling, and fits the power curve Phi(a + b sqrt(runs)) to every
    size tried by maximum likelihood. The plan is the smallest size the
    curve puts at 1 - beta, and its reported power is a fresh estimate at
    that size. Fitting all sizes avoids the optimism of stopping at the first
    size whose noisy estimate reaches the target.

    For "hierarchical", runs are invocations per variant, each with
    ``iterations_per_invocation`` draws (default: the pilot's median); the
    other designs count draws per variant.

    Resampling a pilot treats it as the population: a short pilot
    underrepresents rare detours, so planned power for tail statistics comes
    out optimistic. A pilot of several invocations reduces this.
    """
    if design not in DESIGNS:
        raise ValueError(f"design: must be one of {DESIGNS}")
    if not 0 < alpha < 1 or not 0 < beta < 1:
        raise ValueError("alpha and beta must lie strictly between 0 and 1")
    effect_model = effect_model or uniform_slowdown(delta)
    series = collection.flatten(pilot, variant=variant)
    labels = collection.group_labels(pilot, "invocation", variant=variant)
    groups = [series[labels == g] for g in range(int(labels.max()) + 1)]
    if design == "hierarchical":
        if len(groups) < 2:
            raise ValueError("design 'hierarchical' needs a pilot with several invocations")
        iterations = int(iterations_per_invocation or np.median([g.size for g in groups]))
    else:
        iterations = 1
    block = resampling.block_length(np.log(series)) if design == "blocks" else None
    target = 1 - beta
    history = {}

    def power(runs):
        hits = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DesignWarning)
            for _ in range(n_sim):
                record = _simulate(series, groups, design, runs, iterations, block, effect_model, rng)
                try:
                    result = compare(
                        record,
                        baseline="A",
                        candidate="B",
                        delta=delta,
                        statistic=statistic,
                        effect=effect,
                        confidence=1 - alpha,
                        design=design,
                        n_resamples=n_resamples,
                        rng=rng,
                    )
                except ValueError:
                    continue  # no interval at this size counts as a miss
                hits += result.verdict.outcome == Outcome.SLOWER
        history[runs] = hits / n_sim
        return history[runs]

    low, high = None, max(2, int(start))
    while power(high) < target:
        low, high = high, 2 * high
        if high > max_runs:
            raise ValueError(f"power stays below {target:g} up to {max_runs} runs")
    low = low or 1
    for fraction in (1 / 3, 2 / 3):
        size = int(round(low + fraction * (high - low)))
        if 2 <= size not in history:
            power(size)
    runs = _size_from_power_curve(history, n_sim, target, fallback=high)
    history.pop(runs, None)  # estimate the power at the chosen size afresh
    p = power(runs)
    return RunPlan(
        runs=runs,
        unit="invocations" if design == "hierarchical" else "draws",
        iterations_per_invocation=iterations,
        power=p,
        power_se=math.sqrt(p * (1 - p) / n_sim),
        target_power=target,
        history=tuple(sorted(history.items())),
    )
