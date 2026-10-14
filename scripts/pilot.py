"""Pilot runs for the error tests: measure each estimator's mean error and sd.

Each test module that sizes error tests from a pilot exposes ``PILOTS``, a
mapping from a label to ``errors(n_datasets, rng)``. This script runs them and
prints the mean error, its standard error and the sd, next to which the
recorded values in the test module can be checked.

    python scripts/pilot.py test_lmoments [--datasets 500] [--match text] [--seed 1]
"""

import argparse
import importlib
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modules", nargs="+", help="test modules, such as test_lmoments")
    parser.add_argument("--datasets", type=int, default=500)
    parser.add_argument("--match", default="", help="run only pilots whose label contains this text")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    for name in args.modules:
        module = importlib.import_module(name)
        for label, errors_of in module.PILOTS.items():
            if args.match not in label:
                continue
            start = time.perf_counter()
            errors = np.asarray(errors_of(args.datasets, rng), dtype=float)
            mean, sd = errors.mean(), errors.std(ddof=1)
            print(
                f"{name}  {label:45s} mean {mean:+.5f} (se {sd / math.sqrt(errors.size):.5f})  "
                f"sd {sd:.5f}  [{time.perf_counter() - start:.1f} s]"
            )


if __name__ == "__main__":
    main()
