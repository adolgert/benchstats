"""Step 11: plots. Smoke tests only: each plot runs on small inputs without error."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("KDEpy")

import matplotlib.pyplot as plt  # noqa: E402

from benchstats import collection, decision, plots  # noqa: E402
from support import synthetic  # noqa: E402

BASE = synthetic.FloorPlusDetours(1.0, 0.05, (0.2,), (0.3,))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _record(rng):
    a, b = BASE.sample(60, rng), BASE.sample(60, rng)
    invocation = np.tile(np.repeat([0, 1, 2], 20), 2)
    return collection.make_measurements(np.r_[a, b], variant=np.repeat(["A", "B"], 60), invocation=invocation, benchmark="smoke")


def test_run_sequence(rng):
    plots.run_sequence(_record(rng), variant="A")


def test_density(rng):
    plots.density(BASE.sample(200, rng))


def test_lmoment_ratio_diagram(rng):
    plots.lmoment_ratio_diagram([BASE.sample(100, rng), synthetic.GPDTail(0.2).sample(100, rng)], labels=["floor", "gpd"])


def test_threshold_scan(rng):
    plots.threshold_scan(BASE.sample(500, rng), np.array([1.06, 1.1, 1.2, 1.4]))


def test_rank_autocorrelation(rng):
    plots.rank_autocorrelation(BASE.sample(100, rng), max_lag=20)


def test_comparison_table(rng):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", decision.DesignWarning)
        results = decision.compare_suite([_record(rng), _record(rng)], baseline="A", candidate="B", delta=0.02, n_resamples=200, rng=rng)
    plots.comparison_table(results)
