"""Step 0: package conventions, import direction, and seed replay."""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from support import design

PACKAGE = Path(__file__).parents[1] / "src" / "benchstats"
LEAF_MODULES = ("collection", "characterization", "resampling")
CORE_MODULES = (*LEAF_MODULES, "decision")


def _package_imports(module):
    """Names of benchstats modules that a module's source imports."""
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "benchstats":
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or (node.module or "").split(".")[0] == "benchstats":
                found.update(f"{node.module or ''}:{alias.name}" for alias in node.names)
    return found


@pytest.mark.parametrize("module", LEAF_MODULES)
def test_leaf_modules_import_nothing_from_the_package(module):
    assert _package_imports(module) == set()


def test_decision_imports_only_the_other_core_modules():
    imported = _package_imports("decision")
    names = {name.split(":")[-1].split(".")[-1] for name in imported}
    assert names <= set(LEAF_MODULES)


def test_package_init_imports_no_modules():
    assert _package_imports("__init__") == set()


_FAILING_TEST = """
import pytest

@pytest.mark.statistical
def test_deliberately_fails(rng):
    value = rng.random()
    assert value > 2, f"value={value!r}"
"""


def test_failing_statistical_test_prints_seed_and_replays(pytester):
    conftest = (Path(__file__).parent / "conftest.py").read_text()
    pytester.makeconftest(conftest.replace('pytest_plugins = ["pytester"]', ""))
    pytester.makepyfile(test_inner=_FAILING_TEST)
    pytester.makeini("[pytest]\nmarkers =\n    statistical: tier 2\n")

    first = pytester.runpytest_subprocess("-m", "statistical")
    first.assert_outcomes(failed=1)
    text = first.stdout.str()
    seed = re.search(r"seed=(\d+)", text).group(1)
    value = re.search(r"value=([0-9.e-]+)", text).group(1)
    assert f"--seed={seed}" in text

    replay = pytester.runpytest_subprocess("-m", "statistical", f"--seed={seed}")
    replay.assert_outcomes(failed=1)
    assert f"value={value}" in replay.stdout.str()


def test_error_budget_counts_every_statistical_test():
    """ALPHA = FAMILY_ALPHA / N_STATISTICAL_TESTS needs the right count (decision D5)."""
    tests_dir = Path(__file__).parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "statistical", "-p", "no:cacheprovider", str(tests_dir)],
        capture_output=True,
        text=True,
        cwd=tests_dir.parent,
    )
    collected = [line for line in result.stdout.splitlines() if "::" in line]
    assert len(collected) == design.N_STATISTICAL_TESTS, (
        f"{len(collected)} statistical tests collected; set N_STATISTICAL_TESTS in tests/support/design.py to match"
    )
