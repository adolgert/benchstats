# benchstats: reference for quick use

Python ≥ 3.12; base dependencies numpy and scipy ≥ 1.15. Extras: `kde`
(KDEpy), `diptest` (GPL), `hdf5` (pandas, PyTables), `plots` (matplotlib).
Times are float64 seconds. Every stochastic function takes a required keyword
`rng: numpy.random.Generator`; there is no global random state. Results with
more than two values are frozen dataclasses. Invalid input raises `ValueError`
naming the failed check. Readable explanation: `docs/guide.md`.

Workflow, as a loop through a pilot: **decide → plan (from pilot) → run
(external) → store → describe → compare**; an inconclusive verdict sends you
back to plan.

| Module | Holds |
| --- | --- |
| `collection` | `Measurements` record, I/O, `flatten` (store) |
| `characterization` | L-moments, GPD tail, time order, modes (describe); takes 1-D arrays |
| `resampling` | BCa, stationary, hierarchical bootstrap (used by plan and compare) |
| `decision` | `compare`, `verdict`, suites (compare); `variance_components`, `allocate`, `plan_runs` (plan) |
| `plots` | matplotlib figures |

Only `decision` and `plots` import other package modules.

## Quick start

```python
import numpy as np
from benchstats import collection, characterization as ch, decision

rng = np.random.default_rng()
m = collection.read_table("timings.csv", {"ns": "time", "commit": "variant", "process": "invocation"},
                          benchmark="parse", time_unit="ns")
a = collection.flatten(m, variant="A")                       # run order, warmup excluded
print(ch.lmoment_ratios(a), ch.integrated_autocorrelation_time(a))
r = decision.compare(m, baseline="A", candidate="B", delta=0.02, design="hierarchical", rng=rng)
print(r.statement)       # "B is 3.6% slower than A (95% CI +1.5% to +6.5%). Verdict: slower."
plan = decision.plan_runs(pilot, delta=0.02, design="hierarchical", rng=rng)   # pilot: a Measurements
```

## 1. Decide (before seeing data)

- `delta` (δ): smallest effect that matters. Ratio effects: a fraction (0.02 = 2%), threshold d = log(1+δ). Difference effects: d = δ on the statistic's scale.
- `alpha`: false-verdict rate; `compare` uses `confidence = 1 − alpha` (default 0.95). Two-sided.
- `beta`: miss rate at δ; `plan_runs` default 0.2 (power 0.8).
- `statistic`: a `decision.Statistic(name, function(x, axis=-1), effect, log_value)`. Factories:
  - `median_time()` (default): median of log time; effect = log ratio of medians.
  - `quantile_time(q)`: q-quantile of log time; ratio.
  - `exceedance_rate(u)`: P(X > u); ratio (fails if a resample has no exceedances; use `effect="difference"`).
  - `mean_excess(u)`: E[X − u | X > u]; ratio.
  - `tail_shape(u)`: GPD ξ of excesses over u (L-moment fit); difference.
  - `l_kurtosis()`: τ4; difference.
- `design`: `"independent"` | `"blocks"` | `"hierarchical"` | `"paired"` (see Compare).

## 2. Plan

- `variance_components(m, *, variant=None, include_warmup=False) -> VarianceComponents(between_build, between_invocation, within_invocation, n_builds, n_invocations, iterations_per_invocation)`. SDs of log time from medians and scaled MADs. Between-group spread is corrected for median noise by the normal approximation var(median of J) = (π/2)s²/J. Not estimable → NaN.
- `allocate(components, *, cost_per_invocation, cost_per_iteration, max_iterations=10_000) -> int`. J* = √(c_inv s²_within / (c_iter s²_between)), rounded, ≥ 1 (Cochran 1977). s_between = 0 → `max_iterations`; NaN → error.
- `plan_runs(pilot, *, delta, statistic=None, effect=None, alpha=0.05, beta=0.2, design="independent", effect_model=None, n_sim=200, n_resamples=999, iterations_per_invocation=None, start=16, max_runs=1_000_000, variant=None, rng) -> RunPlan(runs, unit, iterations_per_invocation, power, power_se, target_power, history)`.
  - Power = P(verdict "slower") by simulation: resample the pilot under the design (iid; stationary with the pilot's block length; invocations with replacement then J draws iid within; interleaved pairs), apply `effect_model` to the candidate, run `compare`.
  - Search: from `start`, double while power < 1 − β (or halve while power ≥ 1 − β) to get a bracket (low, high]; add its 1/3 and 2/3 points; fit Φ(a + b√N) by binomial maximum likelihood over all sizes; take the smallest N on the curve at 1 − β, clipped to the bracket; re-estimate power there with fresh simulations (reported `power`, `power_se`).
  - `unit` is `"draws"` per variant, or `"invocations"` per variant for `"hierarchical"` (with `iterations_per_invocation`, default the pilot's median).
  - Effect models: `uniform_slowdown(delta)` (default; x·(1+δ)), `added_detours(rate, mean)` (adds Exp(mean) to a fraction `rate`), or any `f(times, rng) -> times`.
  - Caveat: the pilot is treated as the population. With 1,000-draw pilots, true power at the planned size had mean 0.79 (target 0.80), between-pilot SD 0.04. Tail statistics plan optimistically from short pilots.

## 3. Run (external; not in the library)

Separate processes (invocations); interleave variants; record each draw, not loop averages (else set `executions_per_draw`); keep warmup draws; record run order and environment. Sequential A-then-B under a 2% drift gave false verdicts 96% of the time; interleaved + `"paired"` gave 4.6%.

## 4. Store

`Measurements` (frozen; arrays read-only), long format, one row per draw: `time` float64 > 0, `variant` str, `build`, `invocation`, `iteration`, `order` non-negative int64, `warmup` bool; metadata `benchmark`, `executions_per_draw`, `environment` (str→str), `created` (ISO 8601), `source`. Checks: equal lengths; finite positive times; unique `order`; within each (variant, build, invocation) iterations are 0, 1, 2, … in run order.

- `make_measurements(time, *, variant=None, build=None, invocation=None, iteration=None, order=None, warmup=None, benchmark, executions_per_draw=1, environment=None, created=None, source="")`. Defaults: variant "A" (a scalar string is broadcast), build 0, invocation 0, order = position, iteration = position within invocation by order, warmup False.
- `read_table(path, columns, *, benchmark, time_unit="s", executions_per_draw=1, environment=None)`: CSV from other tools; `columns` maps file names → record names (`"time"` required); `time_unit` ∈ {s, ms, us, ns}.
- `save_csv(m, path)` / `load_csv(path)`: CSV + `<path>.json` metadata sidecar; times written `%.17g`, bit-exact round trip.
- `save_hdf5(m, path, *, key="draws")` / `load_hdf5(...)`: pandas table format, `data_columns=True` (h5py-readable compound dataset at `<key>/table`), metadata JSON in attribute `benchstats_metadata`. Needs `hdf5` extra.
- `flatten(m, *, variant=None, include_warmup=False) -> ndarray`: times in run order; the only place the hierarchy collapses. `variant` required if several.
- `group_labels(m, level, *, variant=None, include_warmup=False)`: 0..G−1 labels for `"build"` or `"invocation"` (distinct across builds), aligned with `flatten`.
- `run_order(m, ...)`: `order` values aligned with `flatten`.
- `mark_warmup(m, counts) -> Measurements`: first `counts` iterations per invocation; `counts` is an int or `{(variant, build, invocation): int}`; replaces existing marks.
- `describe(m) -> str`: one-line summary.

## 5. Describe (`characterization`; 1-D arrays of raw times)

L-moments (wrap `scipy.stats.lmoment`, unbiased estimators, n ≥ 4):
- `lmoments(x) -> LMoments(l1, l2, l3, l4)`; any scale.
- `lmoment_ratios(x) -> LMomentRatios(l1, l2, cv, skewness, kurtosis)`; cv = λ2/λ1, skewness = τ3, kurtosis = τ4; x > 0. Measured bias of τ3, τ4: ≤ 0.04 at n = 100, ≤ 0.005 at n = 1,000.

Detour tail (GPD by L-moments, valid ξ < 1; threshold chosen by you, e.g. median + k·MAD):
- `gpd_from_lmoments(l1, l2) -> (xi, sigma)`: ξ = 2 − λ1/λ2, σ = (1 − ξ)λ1.
- `gpd_fit_lmoments(excess) -> (xi, sigma)`; excess ≥ 0, n ≥ 2.
- `tail_fit(x, threshold) -> TailFit(threshold, n_exceed, rate, xi, sigma, mean_excess)`; mean_excess = σ/(1−ξ) = sample mean excess.
- `threshold_scan(x, thresholds) -> ThresholdScan(threshold, n_exceed, xi, modified_scale, mean_excess)`; arrays, NaN where < 2 exceedances; modified scale σ − ξu. Reading: one exponential source → ξ ≈ 0, flat mean excess; several sources → rising mean excess, ξ > 0 falling to 0; power law → constant ξ > 0.

Time order (x in run order):
- `rank_autocorrelation(x, max_lag) -> ndarray` (lags 1..max_lag); ACF of ranks, denominator n (downward bias O(τ_int/n)); invariant to increasing transforms.
- `integrated_autocorrelation_time(x, *, c=5.0) -> float`: τ_int = 1 + 2Σ_{k≤W} ρ_k on ranks, W = smallest with W ≥ c·τ_int(W) (Sokal).
- `effective_sample_size(x)`: n/τ_int, capped at n.
- `warmup_mser(x, batch=5) -> int`: MSER-5 truncation on log x for one invocation, d ≤ half the batches; returns draws to drop. Over-truncates short, detour-heavy invocations.
- `drift_slope(x) -> float`: Theil–Sen slope of log x vs index, as % change per 1,000 draws; no p-value; O(n²) memory.

Modes (log scale; `kde` extra):
- `log_density(x, *, bandwidth="silverman", grid_size=1024) -> (grid, density)`: KDEpy FFTKDE, Gaussian kernel, on log x; bandwidth "silverman" | "scott" | "isj" | float. Do not use ISJ on timing data: the floor's sharp edge drives it to a tiny bandwidth (10–130 spurious modes on unimodal data).
- `find_modes(grid, density, *, min_prominence) -> Modes(location, height, mass, left, right)`: `scipy.signal.find_peaks` with prominence ≥ `min_prominence`·max density (0.1 works); mass = integral between adjacent minima, normalized.
- `mode_count_scan(x, bandwidths, *, min_prominence, grid_size=1024) -> ndarray`.
- `dip_test(x) -> (dip, p_value)`: Hartigan, via `diptest` (GPL extra); scale as given; conservative for non-uniform unimodal shapes.

Design choice from the description: τ_int ≫ 1 → `"blocks"`; several invocations/builds → `"hierarchical"`; interleaved runs with drift → `"paired"`; else `"independent"`.

## 6. Compare

`compare(m, *, baseline, candidate, delta, statistic=None, effect=None, confidence=0.95, design="independent", level="invocation", resample_within="iid", block_length=None, n_resamples=9999, include_warmup=False, rng) -> Comparison(benchmark, baseline, candidate, statistic, effect, design, estimate, low, high, confidence, delta, verdict, statement, warnings, n_baseline, n_candidate, mc_se_low, mc_se_high)`. Effect = candidate − baseline on the effect scale.

| design | interval |
| --- | --- |
| independent | `bootstrap_ci`, BCa, two independent samples |
| blocks | `block_bootstrap_ci`, stationary bootstrap, percentile; `block_length` default max of Politis–White on log times of each variant |
| hierarchical | `hierarchical_bootstrap_ci` over `level` groups ("invocation" or "build"), then `resample_within` ("iid" default, "blocks", None), percentile |
| paired | greedy adjacent A/B pairs in run order, BCa with `paired=True`; errors if < 2 pairs |

`verdict(low, high, delta, *, effect="ratio") -> Verdict(outcome, less_than_delta)`; `Outcome` ∈ {SLOWER, FASTER, NO_CHANGE, INCONCLUSIVE}:
- low > 0 → SLOWER (less_than_delta if high < d); high < 0 → FASTER (if low > −d);
- −d < low ≤ 0 ≤ high < d → NO_CHANGE; else INCONCLUSIVE. "Slower" = candidate statistic larger.

`DesignWarning` (also in `.warnings`): lag-1 rank ACF > 3/√n and τ_int > 1.5 under independent/paired; > 1 invocation without hierarchical; < 10 groups with hierarchical. The design is never switched.

Suites: `compare_suite(records, *, confidence=0.95, **compare_kwargs)` uses Bonferroni, per-comparison confidence `bonferroni_confidence(confidence, k) = 1 − (1 − confidence)/k`; FWER ≤ α.

Measured (tier-2 pilots), nominal 95% for the median: BCa n = 100: 94%. AR(1) copula φ = 0.7, n = 1,000: stationary 92%, iid 68%. Hierarchical, 20 draws/invocation: 10 invocations 93%, 30 invocations 94% (whole invocations only: 90%, 93%); flat iid 51%. False-verdict rate with no change, α = 0.05: independent 4.6%, blocks 6.5%, paired-on-drift 4.6%.

## Resampling primitives (`resampling`)

All take `data` as a tuple of 1-D samples and `statistic(*samples)`; a statistic with an `axis` keyword is vectorized, otherwise called per resample. All return `ConfidenceInterval(estimate, low, high, confidence, method, n_resamples, mc_se_low, mc_se_high)`; endpoint Monte Carlo SE from 10 batches of resamples (at 9,999 resamples it was < 5% of the half-width in 99% of cases).

- `bootstrap_ci(data, statistic, *, confidence=0.95, n_resamples=9999, method="BCa", paired=False, rng)`: wraps `scipy.stats.bootstrap` ("BCa" | "percentile" | "basic"); raises if BCa is degenerate.
- `block_bootstrap_ci(data, statistic, *, block_length, confidence=0.95, n_resamples=9999, paired=False, rng)`: stationary bootstrap (Politis–Romano), percentile.
- `block_length(x) -> float`: Politis–White (2004) with Patton et al. (2009), ported from `arch.bootstrap.optimal_block_length` (stationary), floored at 1.
- `stationary_indices(n, block_length, n_resamples, rng, *, size=None) -> (n_resamples, size)` indices.
- `hierarchical_bootstrap_ci(data, groups, statistic, *, resample_within="iid", within_block_length=None, confidence=0.95, n_resamples=9999, rng)`: `groups` is one int label array per sample; ≥ 2 groups; equal group sizes are vectorized.

## Plots (`plots`; `plots` extra)

`run_sequence(m, *, variant=None, include_warmup=True, ax=None)`, `density(x, *, min_prominence=0.1, bandwidth="silverman", ax=None)`, `lmoment_ratio_diagram(samples, *, labels=None, ax=None)` (GPD curve τ4 = τ3(1+5τ3)/(5+τ3)), `threshold_scan(x, thresholds, *, axes=None)`, `rank_autocorrelation(x, max_lag=40, *, ax=None)`, `comparison_table(comparisons, *, ax=None)`.

## Testing

`pytest` (tier 1, ~11 s, exact). `pytest -m statistical` (tier 2, 61 tests, ~26 min): each test states the quantity, tolerated level, must-catch level, α, β; sizes from `tests/support/design.py` (`n_for_rate`, `check_rate` exact binomial; `n_for_mean_error`, `check_mean_error` normal, ≥ 30 datasets); family-wise false-failure budget 5% (α per test = 0.05/61). Generators with exact truths: `tests/support/synthetic.py`. Fresh seed per test, printed on failure; replay `pytest <id> -m '' --seed=<s>`. Re-measure pilot-derived tolerances: `python scripts/pilot.py <test_module>`.
