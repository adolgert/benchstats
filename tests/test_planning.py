"""Step 9: how many runs. The end-to-end test is the most important test in the project."""

import math
import warnings

import numpy as np
import pytest

from benchstats import collection, decision
from benchstats.decision import Outcome
from support import synthetic
from support.design import ALPHA, BETA, check_mean_error, check_rate, n_for_mean_error, n_for_rate

# ---------------------------------------------------------------- tier 1


def _components(within, between):
    return decision.VarianceComponents(math.nan, between, within, 1, 10, 20.0)


@pytest.mark.parametrize(
    ("c_inv", "c_iter", "within", "between", "expected"),
    [
        (100.0, 1.0, 0.1, 0.05, 20),  # sqrt(100 * 0.01 / 0.0025) = 20
        (10.0, 1.0, 0.1, 0.1, 3),  # sqrt(10) = 3.16
        (1.0, 1.0, 0.01, 0.1, 1),  # sqrt(0.01) = 0.1, at least 1
        (50.0, 2.0, 0.3, 0.1, 15),  # sqrt(25 * 9) = 15
    ],
)
def test_allocate_hand_computed(c_inv, c_iter, within, between, expected):
    assert decision.allocate(_components(within, between), cost_per_invocation=c_inv, cost_per_iteration=c_iter) == expected


def test_allocate_edge_cases():
    assert decision.allocate(_components(0.1, 0.0), cost_per_invocation=10, cost_per_iteration=1, max_iterations=500) == 500
    with pytest.raises(ValueError, match="not estimable"):
        decision.allocate(_components(0.1, math.nan), cost_per_invocation=10, cost_per_iteration=1)
    with pytest.raises(ValueError, match="positive"):
        decision.allocate(_components(0.1, 0.1), cost_per_invocation=0, cost_per_iteration=1)


def test_variance_components_on_constructed_record():
    """Each invocation repeats the log pattern (-a, 0, a) around its own offset."""
    a = 0.01
    offsets = np.array([-0.04, -0.02, 0.0, 0.02, 0.04, 0.01, -0.01])
    pattern = np.array([-a, 0.0, a])
    log_time = (offsets[:, None] + pattern[None, :]).ravel()
    invocation = np.repeat(np.arange(offsets.size), pattern.size)
    m = collection.make_measurements(np.exp(log_time), invocation=invocation, benchmark="b")
    vc = decision.variance_components(m)
    within = 1.4826022185056018 * a  # residuals are the pattern; median |pattern| = a
    spread = 1.4826022185056018 * 0.02  # median |offset - 0| = 0.02
    assert vc.within_invocation == pytest.approx(within)
    assert vc.between_invocation == pytest.approx(math.sqrt(spread**2 - (math.pi / 2) * within**2 / 3))
    assert math.isnan(vc.between_build)
    assert (vc.n_builds, vc.n_invocations, vc.iterations_per_invocation) == (1, 7, 3.0)


def test_effect_models(rng):
    x = np.ones(10_000)
    np.testing.assert_allclose(decision.uniform_slowdown(0.05)(x, rng), 1.05)
    detoured = decision.added_detours(0.1, 0.5)(x, rng)
    assert np.all(detoured >= 1.0)
    assert 0.08 < np.mean(detoured > 1.0) < 0.12


def test_plan_runs_reports_a_consistent_plan(rng):
    base = synthetic.FloorPlusDetours(1.0, 0.05)
    pilot = collection.make_measurements(base.sample(200, rng), benchmark="pilot")
    plan = decision.plan_runs(pilot, delta=0.05, n_sim=40, n_resamples=200, rng=rng)
    assert plan.unit == "draws" and plan.iterations_per_invocation == 1
    assert plan.target_power == pytest.approx(0.8)
    assert dict(plan.history)[plan.runs] == plan.power
    assert plan.power_se == pytest.approx(math.sqrt(plan.power * (1 - plan.power) / 40))
    # The search starts at 16 and the plan comes from the fitted power curve.
    sizes = [runs for runs, _ in plan.history]
    assert 16 in sizes and plan.runs >= 2


def test_plan_runs_hierarchical_counts_invocations(rng):
    base = synthetic.FloorPlusDetours(1.0, 0.05)
    draws = synthetic.hierarchical(base, n_builds=1, n_invocations=8, n_iterations=10, build_sd=0, invocation_sd=0.01, rng=rng)
    pilot = collection.make_measurements(draws.time, invocation=draws.invocation, benchmark="pilot")
    plan = decision.plan_runs(pilot, delta=0.05, design="hierarchical", n_sim=20, n_resamples=200, rng=rng)
    assert plan.unit == "invocations" and plan.iterations_per_invocation == 10


# ---------------------------------------------------------------- tier 2: end to end

GENERATOR = synthetic.FloorPlusDetours(1.0, 0.05, (0.1, 0.02), (0.2, 1.0))
PILOT_SIZE = 1000
DELTA = 0.01
PLAN_ALPHA, PLAN_BETA = 0.05, 0.2
PLAN_SIMS = 200
N_RESAMPLES = 1999
FRESH_PER_PILOT = 200  # fresh experiments per pilot to measure the true power

# The planned power is 1 - PLAN_BETA = 0.8. A short pilot stands in for the
# population, so the true power at the planned size varies from pilot to
# pilot: over 10 pilots of 1,000 draws (scripts/pilot.py test_planning) it
# averaged 0.79 with sd 0.042, from 0.71 to 0.85. One pilot cannot tell that
# spread from a real shortfall, so the test is an error test over many
# pilots: each pilot's true power minus 0.8 is one error. It tolerates a mean
# shortfall of 0.05 and must catch the plan's shortfall of 10 points. The sd
# of one error (pilot spread plus Monte Carlo error from PLAN_SIMS and
# FRESH_PER_PILOT) is about 0.05.
POWER_ERROR_SD = 0.05
POWER_TOLERATED, POWER_CATCH = 0.05, 0.10
FALSE_TOLERATED, FALSE_CATCH = PLAN_ALPHA, 2 * PLAN_ALPHA


def _plan(rng):
    pilot = collection.make_measurements(GENERATOR.sample(PILOT_SIZE, rng), benchmark="pilot")
    return decision.plan_runs(pilot, delta=DELTA, alpha=PLAN_ALPHA, beta=PLAN_BETA, n_sim=PLAN_SIMS, rng=rng)


def _fresh_outcomes(runs, factor, n_experiments, rng):
    """Verdicts on fresh experiments of the planned size, drawn from the generator."""
    outcomes = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", decision.DesignWarning)
        for _ in range(n_experiments):
            a, b = GENERATOR.sample(runs, rng), factor * GENERATOR.sample(runs, rng)
            m = collection.make_measurements(np.r_[a, b], variant=np.r_[np.full(runs, "A"), np.full(runs, "B")], benchmark="fresh")
            result = decision.compare(
                m, baseline="A", candidate="B", delta=DELTA, confidence=1 - PLAN_ALPHA, n_resamples=N_RESAMPLES, rng=rng
            )
            outcomes.append(result.verdict.outcome)
    return outcomes


@pytest.mark.statistical
def test_planned_runs_reach_planned_power(rng):
    """End to end: plan from a pilot, then check the plan on fresh data from the generator.

    With a true slowdown of delta, the rate of "slower" verdicts must reach
    the planned power on average over pilots. With no change, the rate of
    "slower" or "faster" verdicts must be at most alpha.
    """
    alpha = ALPHA / 2
    n_pilots = n_for_mean_error(POWER_ERROR_SD, POWER_TOLERATED, POWER_CATCH, alpha, BETA)
    n_false = n_for_rate(FALSE_TOLERATED, FALSE_CATCH, alpha, BETA)
    per_pilot = math.ceil(n_false / n_pilots)
    errors, false, runs = [], 0, []
    for _ in range(n_pilots):
        plan = _plan(rng)
        runs.append(plan.runs)
        slower = sum(o == Outcome.SLOWER for o in _fresh_outcomes(plan.runs, 1 + DELTA, FRESH_PER_PILOT, rng))
        errors.append(slower / FRESH_PER_PILOT - plan.target_power)
        false += sum(o in (Outcome.SLOWER, Outcome.FASTER) for o in _fresh_outcomes(plan.runs, 1.0, per_pilot, rng))
    power = check_mean_error(errors, POWER_TOLERATED, alpha)
    assert power.passed, f"planned sizes {runs}; power minus target: {power}"
    calibration = check_rate(false, per_pilot * n_pilots, FALSE_TOLERATED, alpha, "upper")
    assert calibration.passed, f"planned sizes {runs}; {calibration}"


def true_power_at_plan(n_pilots, rng, n_experiments=FRESH_PER_PILOT):
    """For each of several pilots: the true power at the planned size."""
    powers = []
    for _ in range(n_pilots):
        plan = _plan(rng)
        outcomes = _fresh_outcomes(plan.runs, 1 + DELTA, n_experiments, rng)
        powers.append(np.mean([o == Outcome.SLOWER for o in outcomes]))
        print(f"  planned {plan.runs} draws (estimated power {plan.power:.3f}); true power {powers[-1]:.3f}")
    return np.array(powers)


PILOTS = {"true power at planned size (one value per pilot)": true_power_at_plan}
