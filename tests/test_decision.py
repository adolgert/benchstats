"""Step 8: comparing two versions."""

import math
import warnings

import numpy as np
import pytest

from benchstats import collection, decision
from benchstats.decision import Outcome
from support import synthetic
from support.design import ALPHA, BETA, check_rate, n_for_rate

D = math.log1p(0.05)  # d for delta = 5% on the ratio scale

# ---------------------------------------------------------------- tier 1: verdict


@pytest.mark.parametrize(
    ("low", "high", "outcome", "small"),
    [
        (0.01, 0.03, Outcome.SLOWER, True),
        (0.01, 0.06, Outcome.SLOWER, False),
        (0.01, D, Outcome.SLOWER, False),  # high = d is not "less than delta"
        (-0.03, -0.01, Outcome.FASTER, True),
        (-0.06, -0.01, Outcome.FASTER, False),
        (-D, -0.01, Outcome.FASTER, False),
        (-0.03, 0.03, Outcome.NO_CHANGE, False),
        (0.0, 0.03, Outcome.NO_CHANGE, False),  # low = 0 is not "slower"
        (-0.03, 0.0, Outcome.NO_CHANGE, False),  # high = 0 is not "faster"
        (-0.06, 0.03, Outcome.INCONCLUSIVE, False),
        (-0.03, 0.06, Outcome.INCONCLUSIVE, False),
        (-D, 0.03, Outcome.INCONCLUSIVE, False),  # low = -d is not inside
        (-0.03, D, Outcome.INCONCLUSIVE, False),  # high = d is not inside
        (-0.1, 0.1, Outcome.INCONCLUSIVE, False),
    ],
)
def test_verdict_table_on_ratio_scale(low, high, outcome, small):
    assert decision.verdict(low, high, 0.05) == decision.Verdict(outcome, small)


def test_verdict_on_difference_scale_uses_delta_itself():
    assert decision.verdict(0.01, 0.049, 0.05, effect="difference").less_than_delta
    assert decision.verdict(-0.049, 0.049, 0.05, effect="difference").outcome == Outcome.NO_CHANGE
    # 0.0488 < log(1.05) < 0.049 < 0.05: the scales differ.
    assert decision.verdict(-0.0489, 0.0489, 0.05, effect="ratio").outcome == Outcome.INCONCLUSIVE


def test_verdict_rejects_bad_input():
    with pytest.raises(ValueError, match="low <= high"):
        decision.verdict(0.1, 0.0, 0.05)
    with pytest.raises(ValueError, match="delta"):
        decision.verdict(0.0, 0.1, 0.0)
    with pytest.raises(ValueError, match="effect"):
        decision.verdict(0.0, 0.1, 0.05, effect="percent")


# ---------------------------------------------------------------- tier 1: statistics


@pytest.mark.parametrize(
    "statistic",
    [
        decision.median_time(),
        decision.quantile_time(0.9),
        decision.exceedance_rate(1.1),
        decision.mean_excess(1.05),
        decision.tail_shape(1.05),
        decision.l_kurtosis(),
    ],
    ids=lambda s: s.name,
)
def test_statistics_evaluate_row_by_row(statistic, rng):
    x = synthetic.FloorPlusDetours(1.0, 0.05, (0.3,), (0.2,)).sample(4 * 200, rng).reshape(4, 200)
    rows = np.array([statistic.function(row) for row in x])
    np.testing.assert_allclose(statistic.function(x, axis=-1), rows)


def test_median_effect_is_log_ratio_of_medians(rng):
    a, b = rng.lognormal(size=101), 1.1 * rng.lognormal(size=101)
    effect = decision._effect_function(decision.median_time(), "ratio")
    assert effect(a, b) == pytest.approx(math.log(np.median(b) / np.median(a)))


def _record(a, b, **labels):
    variant = np.r_[np.full(a.size, "A"), np.full(b.size, "B")]
    return collection.make_measurements(np.r_[a, b], variant=variant, benchmark="bench", **labels)


# ---------------------------------------------------------------- tier 1: compare


def test_compare_reports_percent_change_and_verdict(rng):
    base = synthetic.FloorPlusDetours(1.0, 0.05)
    m = _record(base.sample(300, rng), 1.1 * base.sample(300, rng))
    result = decision.compare(m, baseline="A", candidate="B", delta=0.02, n_resamples=999, rng=rng)
    assert result.verdict.outcome == Outcome.SLOWER
    assert 9.0 < 100 * math.expm1(result.estimate) < 11.0
    assert result.statement.startswith("B is ")
    assert "% slower than A (95% CI +" in result.statement
    assert result.statement.endswith("Verdict: slower.")


def test_compare_difference_statement(rng):
    base = synthetic.FloorPlusDetours(1.0, 0.05, (0.3,), (0.2,))
    m = _record(base.sample(300, rng), base.sample(300, rng))
    result = decision.compare(m, baseline="A", candidate="B", delta=0.1, statistic=decision.l_kurtosis(), n_resamples=999, rng=rng)
    assert result.effect == "difference"
    assert "L-kurtosis is" in result.statement


def test_compare_warns_on_correlated_draws_with_independent_design(rng):
    base = synthetic.FloorPlusDetours(1.0, 0.05)
    a = synthetic.correlated(base, 0.9, 400, rng)
    b = synthetic.correlated(base, 0.9, 400, rng)
    with pytest.warns(decision.DesignWarning, match="correlated in run order"):
        result = decision.compare(_record(a, b), baseline="A", candidate="B", delta=0.02, n_resamples=999, rng=rng)
    assert any("blocks" in w for w in result.warnings)


def test_compare_warns_on_invocations_without_hierarchical_design(rng):
    base = synthetic.FloorPlusDetours(1.0, 0.05)
    invocation = np.tile(np.repeat([0, 1, 2], 20), 2)
    m = _record(base.sample(60, rng), base.sample(60, rng), invocation=invocation)
    with pytest.warns(decision.DesignWarning, match="consider design='hierarchical'"):
        decision.compare(m, baseline="A", candidate="B", delta=0.02, n_resamples=999, rng=rng)


def test_compare_never_switches_design(rng):
    base = synthetic.FloorPlusDetours(1.0, 0.05)
    m = _record(base.sample(100, rng), base.sample(100, rng))
    with pytest.raises(ValueError, match="interleaved"):
        decision.compare(m, baseline="A", candidate="B", delta=0.02, design="paired", rng=rng)
    with pytest.raises(ValueError, match="two invocations"):
        decision.compare(m, baseline="A", candidate="B", delta=0.02, design="hierarchical", rng=rng)


def test_adjacent_pairs_follow_run_order():
    # Run order: A B B A A B A
    variant = ["A", "B", "B", "A", "A", "B", "A"]
    m = collection.make_measurements(np.arange(1.0, 8.0), variant=variant, benchmark="b")
    ia, ib = decision._adjacent_pairs(m, "A", "B", include_warmup=False)
    a = collection.flatten(m, variant="A")[ia]
    b = collection.flatten(m, variant="B")[ib]
    # Pairs (1, 2), (3, 4), (5, 6) by time value; the last A is unpaired.
    np.testing.assert_array_equal(a, [1.0, 4.0, 5.0])
    np.testing.assert_array_equal(b, [2.0, 3.0, 6.0])


def test_compare_suite_widens_intervals(rng):
    base = synthetic.FloorPlusDetours(1.0, 0.05)
    records = [_record(base.sample(80, rng), base.sample(80, rng)) for _ in range(4)]
    results = decision.compare_suite(records, baseline="A", candidate="B", delta=0.02, n_resamples=999, rng=rng)
    assert [r.confidence for r in results] == [pytest.approx(1 - 0.05 / 4)] * 4
    assert decision.bonferroni_confidence(0.9, 10) == pytest.approx(0.99)


# ---------------------------------------------------------------- tier 2: false verdicts

COMPARE_LEVEL = 0.05  # each comparison at 95% confidence
FALSE_CATCH = 2 * COMPARE_LEVEL
N_RESAMPLES = 1999
FLOOR = synthetic.FloorPlusDetours(1.0, 0.05, (0.1, 0.02), (0.2, 1.0))
DRIFT_PAIRS = 200


def no_change_record(design, rng):
    if design == "independent":
        return _record(FLOOR.sample(100, rng), FLOOR.sample(100, rng))
    if design == "blocks":
        return _record(synthetic.correlated(FLOOR, 0.5, 1000, rng), synthetic.correlated(FLOOR, 0.5, 1000, rng))
    if design == "hierarchical":
        a, b = (
            synthetic.hierarchical(FLOOR, n_builds=1, n_invocations=30, n_iterations=20, build_sd=0.0, invocation_sd=0.02, rng=rng)
            for _ in range(2)
        )
        return _record(a.time, b.time, invocation=np.r_[a.invocation, b.invocation])
    # Drift: a steady 2% rise over the experiment, with A and B interleaved
    # ("paired") or with all A runs first ("sequential").
    order = np.arange(2 * DRIFT_PAIRS)
    time, _ = synthetic.add_drift(FLOOR.sample(order.size, rng), order, total_change=0.02)
    is_b = order % 2 == 1 if design == "paired" else order >= DRIFT_PAIRS
    return collection.make_measurements(time, variant=np.where(is_b, "B", "A"), order=order, benchmark="drift")


def false_verdicts(design, n_datasets, rng):
    analysis = "independent" if design == "sequential" else design
    out = np.empty(n_datasets)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", decision.DesignWarning)
        for i in range(n_datasets):
            result = decision.compare(
                no_change_record(design, rng), baseline="A", candidate="B", delta=0.02, design=analysis, n_resamples=N_RESAMPLES, rng=rng
            )
            out[i] = result.verdict.outcome in (Outcome.SLOWER, Outcome.FASTER)
    return out


# With no true change, the rate of "slower" or "faster" verdicts is at most
# alpha. Pilot rates (scripts/pilot.py test_decision, 1,000 datasets) are in
# comments; where a design is known to under-cover a little, the tolerated
# rate sits above alpha but every test must catch 2 alpha.
NO_CHANGE_CASES = {
    "independent": 0.05,  # pilot 0.050
    "blocks": 0.06,  # pilot 0.064
    "hierarchical": 0.07,  # pilot 0.071
    "paired": 0.05,  # pilot 0.047, on interleaved drift
}


@pytest.mark.statistical
@pytest.mark.parametrize("design", list(NO_CHANGE_CASES))
def test_no_change_gives_few_false_verdicts(design, rng):
    tolerated = NO_CHANGE_CASES[design]
    n = n_for_rate(tolerated, FALSE_CATCH, ALPHA, BETA)
    count = int(false_verdicts(design, n, rng).sum())
    result = check_rate(count, n, tolerated, ALPHA, "upper")
    assert result.passed, result


# Negative control: all A runs first, then all B, analyzed as independent.
# The drift alone makes B look about 1% slower. Pilot rate 1.0; the test
# claims at least 0.5 and must catch 2 alpha.
SEQUENTIAL_TOLERATED = 0.5


@pytest.mark.statistical
def test_sequential_runs_under_drift_give_false_verdicts(rng):
    n = n_for_rate(SEQUENTIAL_TOLERATED, FALSE_CATCH, ALPHA, BETA)
    count = int(false_verdicts("sequential", n, rng).sum())
    result = check_rate(count, n, SEQUENTIAL_TOLERATED, ALPHA, "lower")
    assert result.passed, result


PILOTS = {
    f"false verdicts {design} (mean is the rate)": (lambda m, rng, d=design: false_verdicts(d, m, rng))
    for design in [*NO_CHANGE_CASES, "sequential"]
}
