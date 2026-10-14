# A guide to benchstats

This guide explains how to use benchstats to measure the speed of code and to
compare two versions of it. It follows the order in which you do the work. Each
section says what the step is for, which functions do it, and what can go wrong.

## The workflow

A benchmark experiment answers two questions. The first question is what the
timings of one version look like. The second question is whether a new version
is faster or slower than an old one by an amount that matters.

The work has six steps:

1. **Decide.** Choose what you want to learn and how sure you need to be.
2. **Plan.** Work out how many runs you need, using data from a small pilot run.
3. **Run.** Time the code. This step happens outside the library.
4. **Store.** Put every timing into one record, with labels that say where it came from.
5. **Describe.** Look at the timings of one version: their center, spread, shape, tail, and how they change over time.
6. **Compare.** Decide whether one version differs from another by an amount that matters.

The steps form a loop, not a straight line. Planning needs data, so you start
with a short pilot run. You store the pilot and describe it, and then you plan
the full experiment from it. After the full run you store, describe, and
compare again. If the comparison is inconclusive, the plan tells you how many
more runs to make.

```
decide ──> pilot run ──> store ──> describe ──> plan ──> full run ──> store ──> describe ──> compare
                                                  ^                                            │
                                                  └──────────── inconclusive ──────────────────┘
```

The library has four main modules. They do not map one to one onto the steps:

- `benchstats.collection` holds the record and handles storing (Store).
- `benchstats.characterization` holds the descriptive statistics (Describe).
- `benchstats.resampling` holds the bootstrap methods. Both planning and comparison use them.
- `benchstats.decision` holds both the planner (Plan) and the comparison (Compare).

There is also `benchstats.plots`, which draws figures. It needs matplotlib.

All times are in seconds. Every function that uses random numbers takes a
`numpy.random.Generator` as the keyword argument `rng`. There is no hidden
random state, so you can repeat any result by reusing the seed.

## Install

```sh
pip install -e .                        # numpy and scipy only
pip install -e '.[kde,hdf5,plots]'      # density estimates, HDF5 files, figures
pip install -e '.[diptest]'             # the dip test, which is GPL-licensed
```

## 1. Decide

Make these choices before you look at any data. The error rates in the later
steps are only true if the choices come first. If you choose the size of change
that "matters" after seeing the result, the stated error rates no longer hold.

**The effect size, δ.** This is the smallest change you care about. For speed
it is usually a fraction, such as 0.02 for a 2% change. A smaller δ needs more
runs, often many more, because the number of runs grows with 1/δ².

**The false-alarm rate, α.** This is the chance of reporting a change when there
is none. The default is 0.05, which matches a 95% confidence interval.

**The miss rate, β.** This is the chance of missing a real change of size δ.
The power of the experiment is 1 − β. The planner defaults to β = 0.2, which
means 80% power.

**The statistic.** This is the number you compute for each version. The default
is the median of the log of the time. Comparing two medians on the log scale
gives the ratio of the medians, which the library reports as a percent change.
Other choices measure the tail instead of the center:

| Statistic | Function | What it measures | Effect scale |
| --- | --- | --- | --- |
| Median time | `decision.median_time()` | The typical time | ratio |
| A high quantile | `decision.quantile_time(0.99)` | Tail latency | ratio |
| Exceedance rate | `decision.exceedance_rate(u)` | How often a draw is slower than `u` seconds | ratio |
| Mean excess | `decision.mean_excess(u)` | How much slower those draws are, on average | ratio |
| Tail shape ξ | `decision.tail_shape(u)` | How heavy the tail is above `u` | difference |
| L-kurtosis τ4 | `decision.l_kurtosis()` | How peaked and heavy-tailed the whole distribution is | difference |

A ratio effect compares the log of the two values. A difference effect
subtracts them. For a difference effect, δ is on the scale of the statistic
itself, such as 0.1 for ξ.

**The design.** This says how the draws depend on each other. It decides how
the bootstrap resamples them. You can only choose it well after you describe
the pilot, so the Describe and Compare sections return to it.

| Design | Use it when |
| --- | --- |
| `"independent"` | Draws do not depend on their neighbors, and all draws come from one process. |
| `"blocks"` | Neighboring draws are correlated, for example because detours come in bursts. |
| `"hierarchical"` | Draws come from several processes (invocations) or builds, and each one has its own offset. |
| `"paired"` | The two versions ran interleaved, one after the other, and the machine's speed drifts. |

## 2. Plan

Planning answers two questions. How many runs give enough power? And how should
the runs be split between process starts and repetitions inside a process?

### Run a pilot first

The planner learns the noise from a pilot run. Make the pilot look like the real
experiment. Use the same machine, the same harness, and several invocations
(separate process starts). A pilot of a few hundred to a thousand draws is a
reasonable size. Store and describe the pilot before you plan from it.

### Split runs between invocations and iterations

Each process start has a cost: loading, compiling, and warming up. Each
iteration inside a process has a smaller cost. Some of the noise varies between
processes and some varies within a process. `variance_components` measures both
parts, and `allocate` turns them into the number of iterations per invocation
that gives the most information for the cost:

```python
from benchstats import decision

vc = decision.variance_components(pilot, variant="A")
iterations = decision.allocate(vc, cost_per_invocation=0.5, cost_per_iteration=0.001)
```

`variance_components` returns standard deviations of log time between builds,
between invocations, and within invocations. It uses medians and scaled median
absolute deviations, so rare detours pull on it less. It subtracts the part of
the spread between invocation medians that comes from noise within each
invocation. A component it cannot estimate, such as the spread between builds
when there is only one build, is NaN.

`allocate` uses the standard result from two-stage sampling (Cochran 1977). The
number of iterations per invocation is the square root of
(cost per invocation × within variance) / (cost per iteration × between
variance), rounded, and at least 1. If the spread between invocations is zero,
it returns `max_iterations`.

### Find the number of runs

`plan_runs` estimates power by simulation. It resamples the pilot to make a
fake baseline. It makes a fake candidate by applying the effect you want to
detect. Then it runs `compare` on the pair and records whether the verdict is
"slower". It repeats this many times at each size:

```python
plan = decision.plan_runs(pilot, delta=0.02, design="independent", rng=rng)
print(plan.runs, plan.unit, plan.power, plan.power_se)
```

The search starts at 16 runs. If the power there is too low, it doubles the
size until the power reaches 1 − β. If the power is already high enough, it
halves the size until the power falls short. Either way it ends with two sizes,
one too small and one large enough, and it adds two sizes between them. Then it
fits the curve
power = Φ(a + b√N) to every size it tried. That curve is the right shape when
the test statistic is roughly normal. The plan is the smallest size where the
curve reaches 1 − β, kept between the two bracketing sizes. Finally, it estimates the power at that size again with
fresh simulations. The reported power and its standard error come from that
fresh estimate.

The result counts draws per version for most designs. For the hierarchical
design it counts invocations per version, each with
`iterations_per_invocation` draws.

The effect model says what the change looks like. The default,
`uniform_slowdown(delta)`, multiplies every time by 1 + δ. For tail statistics,
`added_detours(rate, mean)` adds an exponential delay to a fraction of the
draws. You can pass any function of `(times, rng)` that returns new times.

### What planning cannot do

The planner treats the pilot as if it were the whole population. A short pilot
may miss rare detours, so planned power for tail statistics comes out too
optimistic. Even for the median, the true power at the planned size varies from
pilot to pilot. In our tests, with pilots of 1,000 draws, the true power at the
planned size averaged 0.79 against a target of 0.80, with a standard deviation
of about 0.04 between pilots. If the stakes are high, plan for a little more
power than you need, or use a larger pilot.

## 3. Run (outside the library)

The library does not time code. It reads timings that another tool produced.
How you collect them matters as much as how you analyze them. A good harness
does these things:

- **It starts several processes.** Each process start is an invocation. Effects
  such as memory layout and code placement change between processes, and only
  several invocations reveal them.
- **It interleaves the versions.** Run A, then B, then A, then B, in the same
  session. If you run all of A first and then all of B, any slow drift in the
  machine's speed looks like a difference between the versions. In our tests, a
  steady 2% drift produced a false verdict 96% of the time when the versions ran
  one after the other. With interleaved runs and the paired design, the rate
  stayed at the nominal 5%.
- **It records every draw.** Do not average inside the harness. Averages hide
  the detours, which are often what you most want to see. If the harness must
  time a loop and report the mean, record how many executions each draw
  averages.
- **It keeps warmup draws.** Mark them later rather than throwing them away.
- **It records the run order** and the machine details: host, CPU, compiler
  flags, and anything else that could change the result.

## 4. Store

### The record

All timings of one benchmark live in one `Measurements` record. It is a table
with one row per draw:

| Column | Type | Meaning |
| --- | --- | --- |
| `time` | float | Seconds for this draw. |
| `variant` | string | Which version ran, such as "A", "B", or a commit id. |
| `build` | integer | Which build of that version. |
| `invocation` | integer | Which process run within the build. |
| `iteration` | integer | Position within the invocation, starting at 0. |
| `order` | integer | Position in the run order of the whole experiment. |
| `warmup` | boolean | Whether the draw belongs to the warmup. |

The record also holds metadata: the benchmark name, the number of executions per
draw, a dictionary describing the environment, the creation time, and the
source.

The record checks itself when it is built. All columns must have the same
length. Times must be finite and positive. Labels must be non-negative
integers. Run-order values must be unique. Within each invocation, iterations
must run 0, 1, 2, and so on without gaps, in run order. If a check fails, you
get a ValueError that names the check. After it is built, the record's arrays
are read-only, so no function can change a timing in place.

### Building a record

`make_measurements` builds a record from arrays. Labels you leave out get
defaults: one variant ("A"), one build, one invocation, run order equal to
position, and iterations counted within each invocation in run order.

```python
from benchstats import collection

m = collection.make_measurements(
    times,
    variant=variants,
    invocation=invocations,
    benchmark="parse",
    environment={"host": "lab-3", "cpu": "x86-64"},
)
```

`read_table` reads a CSV file that another tool wrote. You give it a mapping
from the file's column names to the record's column names. The `time_unit`
argument converts from "ms", "us", or "ns" to seconds:

```python
m = collection.read_table(
    "timings.csv",
    {"ns": "time", "commit": "variant", "process": "invocation"},
    benchmark="parse",
    time_unit="ns",
)
```

### Saving and loading

`save_csv` writes one CSV file of draws and a JSON file of metadata beside it,
with the same name and the suffix ".json". Times are written with 17
significant digits, so `load_csv` gives back exactly the same numbers.

`save_hdf5` and `load_hdf5` store the draws as a pandas table in an HDF5 file,
with the metadata as an attribute. Every column is its own field, so other HDF5
tools can open the file without pandas. These two functions need the `hdf5`
extra.

### Getting plain arrays out

The descriptive functions and the bootstrap take plain arrays, not records.
`flatten` returns the times of one version as a 1-D array in run order. It is
the only place where the hierarchy of builds and invocations is collapsed. It
leaves out warmup draws unless you ask for them:

```python
a = collection.flatten(m, variant="A")
labels = collection.group_labels(m, "invocation", variant="A")  # aligned with a
```

`group_labels` returns integer labels for builds or invocations, in the same
order as `flatten`. `run_order` returns the run-order positions in that order.

`mark_warmup` returns a new record with the first iterations of each
invocation marked as warmup. You can give one count for all invocations, or a
dictionary from (variant, build, invocation) to a count. The original record
does not change.

## 5. Describe

The descriptive functions take a 1-D array of raw times in seconds, in run order
where order matters. Some of them take the log internally; each docstring says
so. None of them makes a decision, so none of them has an error rate. They help
you understand the data and choose the design for the comparison.

### Center, spread, and shape: L-moments

L-moments summarize a distribution in the same way as the mean, standard
deviation, skewness, and kurtosis. The difference is that L-moments exist
whenever the mean exists, and outliers do not dominate them. Timing data have
long tails, so this matters.

```python
from benchstats import characterization as ch

r = ch.lmoment_ratios(a)
print(r.l1, r.l2, r.cv, r.skewness, r.kurtosis)
```

`lmoments` returns λ1 to λ4. `lmoment_ratios` returns λ1, λ2, the L-CV
(λ2/λ1), the L-skewness τ3, and the L-kurtosis τ4. Both need at least four
draws. The estimators of λ1 and λ2 are unbiased. The ratios τ3 and τ4 have a
small bias in small samples. In our tests the bias was up to 0.04 at 100 draws
and below 0.005 at 1,000 draws.

### The detour tail

Most timings sit near a floor. A few are much slower because something
interrupted the code: an interrupt, a page fault, a context switch. We call
these detours. The tail functions measure how often detours happen and how
large they are.

Pick a threshold above the floor. A first one to try is the median plus a few
median absolute deviations. `tail_fit` then counts the draws above the
threshold and fits a generalized Pareto distribution to the excesses:

```python
fit = ch.tail_fit(a, threshold)
print(fit.rate, fit.mean_excess, fit.xi)
```

The fit uses the first two L-moments of the excesses. The shape is
ξ = 2 − λ1/λ2 and the scale is σ = (1 − ξ)λ1. The mean excess σ/(1 − ξ) is
then exactly the average excess. If detours come from one source with an
exponential size, ξ is 0 and the mean excess is the mean detour size.

The library does not choose the threshold for you. Instead, `threshold_scan`
fits many thresholds at once, and you read the pattern:

- If ξ stays near 0 and the mean excess stays flat as the threshold rises, the
  detours look like one exponential source.
- If the mean excess rises with the threshold and ξ falls toward 0 at high
  thresholds, there are several detour sources of different sizes.
- If ξ stays at the same positive value, the tail follows a power law.

A positive ξ at one threshold does not by itself mean a power law. Several
exponential sources can produce it.

### Time order

Draws that depend on their neighbors carry less information than independent
draws. If you treat them as independent, your intervals come out too narrow. In
our tests, an independent bootstrap on correlated draws covered the true median
only 65% to 68% of the time instead of 95%.

`rank_autocorrelation` gives the autocorrelation of the ranks of the draws at
each lag. Ranks keep large detours from dominating it, and it does not change
if you take logs. `integrated_autocorrelation_time` adds up those
autocorrelations into τ_int. For independent draws, τ_int is near 1. A value of
5 means the draws carry about as much information as one fifth as many
independent draws. `effective_sample_size` returns n/τ_int, capped at n.

`warmup_mser` finds where the warmup ends in one invocation. It uses the MSER-5
rule from simulation output analysis on log times. It returns the number of
leading draws to drop. Apply it to each invocation separately, then pass the
counts to `collection.mark_warmup`. On short invocations with frequent
detours, it can drop too much, up to half of the invocation. Check its answer
against a run-sequence plot.

`drift_slope` gives the slope of log time against run order, as a percent
change per 1,000 draws. It uses the Theil–Sen estimator, which ignores
outliers. It is descriptive only and has no p-value, because a p-value would
assume independent draws. It holds all pairs of draws in memory, so use it on
a few thousand draws at a time.

### Modes

Some benchmarks have two or more typical times, for example because the code
takes one of two paths. A single median hides this.

`log_density` estimates the density of log time with a Gaussian kernel. On the
log scale, one bandwidth suits both the narrow floor and the long tail.
`find_modes` finds the peaks of that density, with a floor on how prominent a
peak must be, and reports the probability mass between the low points on
either side of each peak:

```python
grid, density = ch.log_density(a)
modes = ch.find_modes(grid, density, min_prominence=0.1)
print(np.exp(modes.location), modes.mass)
```

The default bandwidth is Silverman's rule. The improved Sheather–Jones rule
(ISJ) is available, but it does badly on timing data. The hard floor makes a
sharp edge in the density, and ISJ answers a sharp edge with a very small
bandwidth. In our tests it split single-mode data into 10 to 130 false modes.
`mode_count_scan` counts modes across a range of bandwidths. A mode that
persists over a range of bandwidths is more believable than one that appears at
only one bandwidth.

`dip_test` runs Hartigan's dip test of unimodality. It needs the `diptest`
extra, which is GPL-licensed. Its p-value is computed against the uniform
distribution, which is the hardest single-mode case. So for other single-mode
shapes it rejects less often than its nominal level. A second mode may be a
second code path, or it may be a cluster of detours of similar size. The mode
locations, the masses, and the tail fit help you tell these apart.

### Choosing the design from the description

Use the description to choose the design for the comparison:

- If the rank autocorrelation is clearly positive and τ_int is well above 1,
  use `"blocks"`.
- If you have several invocations or builds, use `"hierarchical"`.
- If the versions were interleaved and there is drift, use `"paired"`.
- Otherwise, use `"independent"`.

### Figures

`benchstats.plots` draws a run-sequence plot colored by invocation, the
density of log times with its modes, the L-moment ratio diagram, the threshold
scan, the rank autocorrelation, and a table of comparisons. Each function
draws on a matplotlib Axes you pass in, or makes a new figure.

## 6. Compare

### One comparison

`compare` takes a record, the names of two versions, and δ. It returns an
interval for the effect, a verdict, and a sentence:

```python
result = decision.compare(m, baseline="A", candidate="B", delta=0.02, design="hierarchical", rng=rng)
print(result.statement)
# B is 3.6% slower than A (95% CI +1.5% to +6.5%). Verdict: slower.
```

The effect is always the candidate minus the baseline, on the effect scale.
For the median, that is the log ratio of the medians. The result holds the
estimate, the interval, the verdict, the sentence, the number of draws used for
each version, any warnings, and the Monte Carlo standard error of each end of
the interval.

### The verdict

The verdict turns the interval into one of four results. Let d = log(1 + δ) for
ratio effects and d = δ for difference effects:

| Interval [low, high] | Verdict |
| --- | --- |
| low > 0 | Slower. Flagged "by less than δ" if high < d. |
| high < 0 | Faster. Flagged "by less than δ" if low > −d. |
| −d < low ≤ 0 ≤ high < d | No change as large as δ. |
| anything else | Inconclusive: more runs are needed. |

"Slower" means the candidate's statistic is larger. For a difference effect,
such as ξ, that means a heavier tail rather than a slower time.

The "no change" verdict needs the whole interval to sit inside ±d. It is a
positive finding: any change is smaller than δ. "Inconclusive" means the
interval is too wide to say either way. You need more runs, and `plan_runs`
can tell you how many.

### How the interval is computed

The interval comes from a bootstrap. The design chooses the kind of bootstrap:

- `"independent"` uses the bias-corrected and accelerated (BCa) bootstrap from
  SciPy. Each version's draws are resampled separately.
- `"blocks"` uses the stationary bootstrap of Politis and Romano. It resamples
  blocks of neighboring draws, with random block lengths, so that the
  dependence inside each block is kept. The mean block length comes from the
  rule of Politis and White, applied to the log times. The interval is a
  percentile interval.
- `"hierarchical"` resamples whole invocations (or builds, with
  `level="build"`) with replacement. It then resamples the draws inside each
  chosen invocation. The interval is a percentile interval.
- `"paired"` matches each baseline draw with the neighboring candidate draw in
  run order. It resamples the pairs together, using the BCa bootstrap.

Each interval also reports the Monte Carlo standard error of its two ends,
which it estimates from batches of resamples. If that error is large compared
with the width of the interval, use more resamples. At the default of 9,999
resamples, the error was below 5% of the half-width in 99% of our test cases.

### How well the intervals work

The test suite measures how often each interval covers the true value. These
are the coverages of nominal 95% intervals for the median in our tests:

| Situation | Method | Coverage |
| --- | --- | --- |
| Independent draws, n = 100 | BCa | 94% |
| Correlated draws (Gaussian AR(1) copula, φ = 0.7), n = 1,000 | stationary bootstrap | 92% |
| Same correlated draws | independent bootstrap | 68% (wrong design) |
| 10 invocations of 20 draws | hierarchical | 93% |
| 30 invocations of 20 draws | hierarchical | 94% |
| Same invocations | independent bootstrap | 51% (wrong design) |

The wrong designs show why the design matters. The block bootstrap and the
hierarchical bootstrap with few invocations cover a little less than they
claim. So their false-alarm rates are a little higher than α. In our tests, the
block design gave false verdicts about 6.5% of the time at α = 0.05.

### Warnings

`compare` warns when the data look inconsistent with the design you chose. It
warns when draws are clearly correlated under the independent or paired
design, when a version has several invocations but the design is not
hierarchical, and when a hierarchical comparison has fewer than ten groups. It
never changes the design on its own. The warnings appear as `DesignWarning`
and also in the result's `warnings` field.

### Many benchmarks at once

When you compare many benchmarks, the chance that at least one verdict is
false grows with their number. `compare_suite` handles this with the
Bonferroni correction. For K benchmarks, it widens each interval to confidence
1 − α/K. Then the chance of any false verdict across the whole suite is at most
α. This is conservative when there are many benchmarks.

## How we know the statistics are right

The test suite has two tiers. The first tier runs on every `pytest` call and
takes seconds. It checks exact results: hand-computed values, comparisons with
reference implementations, and the rules about which modules may import which.

The second tier runs with `pytest -m statistical` and takes about 26 minutes.
It checks statistical claims by simulation. It uses generators of synthetic
timing data whose true values are known exactly, from closed forms or
numerical integration.

Each statistical test states five things: the quantity it checks, the level it
tolerates, the wrong level it must catch, the false-failure rate α, and the
miss rate β. The number of simulated datasets follows from those. Some tests
check rates, such as coverage; they use exact binomial calculations. Others
check average errors; they use the normal approximation. Many tests are
negative controls. They show that a wrong method fails, so the test is able to
fail.

All the statistical tests together fail by chance at most 5% of the time. Each
test draws a fresh random seed and prints it if it fails. To replay a failure,
run `pytest <test id> -m '' --seed=<seed>`. Where a tolerated level came from a
pilot simulation, `python scripts/pilot.py <test module>` measures it again.

## Known limits

- The library does not time code. It reads timings from other tools.
- It does not choose tail thresholds. It gives diagnostics, and you choose.
- The planner treats the pilot as the population. A short pilot gives
  optimistic power, especially for tail statistics.
- MSER can drop too much warmup on short invocations with many detours.
- `drift_slope` uses memory that grows with the square of the number of draws.
- The block and hierarchical bootstraps cover a little less than their nominal
  level, as the table above shows.
- The dip test depends on a GPL-licensed package. It is an optional extra.
