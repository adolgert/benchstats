"""Test conventions shared by every tier.

Tier 1 runs on every ``pytest`` call. Tier 2 holds the tests marked
``statistical`` and runs with ``pytest -m statistical``.

Every test that needs random numbers takes the ``rng`` fixture. It draws a
fresh seed each run and writes the seed into the failure report. To replay a
failure, run the same test with ``--seed=<value>``.
"""

import secrets
import sys
from pathlib import Path

import numpy as np
import pytest

# Lets test modules import the support package as ``support``.
sys.path.insert(0, str(Path(__file__).parent))

pytest_plugins = ["pytester"]

_SEED_KEY = pytest.StashKey[int]()


def pytest_addoption(parser):
    parser.addoption(
        "--seed",
        type=int,
        default=None,
        help="Replay tests that use the rng fixture with this seed.",
    )


@pytest.fixture
def rng(request):
    """A numpy Generator with a fresh seed, or the seed given by --seed."""
    seed = request.config.getoption("--seed")
    if seed is None:
        seed = secrets.randbits(64)
    request.node.stash[_SEED_KEY] = seed
    return np.random.default_rng(seed)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    seed = item.stash.get(_SEED_KEY, None)
    if seed is not None and report.failed:
        message = f"seed={seed}\nreplay with: pytest '{item.nodeid}' -m '' --seed={seed}"
        if hasattr(report.longrepr, "addsection"):
            report.longrepr.addsection("benchstats seed", message)
        else:
            report.sections.append(("benchstats seed", message))
    return report
