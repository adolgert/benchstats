# benchstats

Statistics for benchmark timings. The toolkit keeps every timing and answers
two questions with error rates fixed in advance:

1. What does the timing distribution look like? Location and spread, shape
   (L-moments), the detour tail, modes, and time order (dependence, warmup,
   drift).
2. Does version B differ from version A by an amount that matters? An
   interval for the effect, judged against an effect size δ chosen before the
   data are seen, and a planner for how many runs that judgment needs.

## Install

```sh
pip install -e .                       # numpy and scipy only
pip install -e '.[kde,hdf5,plots]'     # density estimates, HDF5 files, figures
pip install -e '.[diptest]'            # Hartigan's dip test (GPL; optional)
```

For development, `uv sync --group dev` installs everything the tests use.

## Modules

| Module | What it does |
| --- | --- |
| `benchstats.collection` | The `Measurements` record: one row per draw with variant, build, invocation, iteration, run order and warmup labels. Read CSV from other tools, save and load CSV+JSON or HDF5, and `flatten` to plain arrays. |
| `benchstats.characterization` | Functions of plain arrays: L-moments, generalized Pareto tail fit and threshold scan, rank autocorrelation, τ_int, MSER warmup, Theil–Sen drift, log density, modes, dip test. |
| `benchstats.resampling` | Bootstrap intervals for independent (BCa), correlated (stationary bootstrap) and hierarchical (resample invocations) draws, each with the Monte Carlo error of its endpoints. |
| `benchstats.decision` | `compare` with four verdicts, Bonferroni suites, variance components, iterations per invocation, and `plan_runs`. |
| `benchstats.plots` | Optional figures (matplotlib). |

Only `decision` (and `plots`) import other modules of the package; a test
enforces this.

## Example

```python
import numpy as np
from benchstats import collection, decision

m = collection.read_table(
    "timings.csv",
    {"ns": "time", "commit": "variant", "process": "invocation"},
    benchmark="parse",
    time_unit="ns",
)
result = decision.compare(
    m, baseline="abc123", candidate="def456", delta=0.02,
    design="hierarchical", rng=np.random.default_rng(),
)
print(result.statement)
# def456 is 4.1% slower than abc123 (95% CI +2.3% to +5.9%). Verdict: slower.
```

`python examples/walkthrough.py` runs the whole workflow on simulated data.

## Tests

```sh
pytest                     # tier 1: exact tests, seconds
pytest -m statistical      # tier 2: simulation-based tests, minutes
```

Every tier-2 test states the quantity it checks, the level it tolerates, the
discrepancy it must catch, α and β; the number of simulated datasets follows
(`tests/support/design.py`). The whole tier-2 run fails by chance with
probability at most 5% (decision D5). Each test draws a fresh seed and prints
it on failure; replay with `pytest <test id> -m '' --seed=<seed>`.

Tolerated levels that come from pilot runs can be re-measured with
`python scripts/pilot.py <test module>`.
