"""A walk through the toolkit on simulated timings.

Simulates two versions of a benchmark (B is 3% slower, with occasional
detours), saves and reloads the record, describes the baseline's
distribution, compares B with A, and plans the size of the next experiment.

    python examples/walkthrough.py [--plots walkthrough.png]
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from benchstats import characterization, collection, decision


def simulate(rng, *, slowdown, invocations=6, iterations=60):
    """Floor 1 ms, 50 us of jitter, a 10% chance of a 200 us detour, per-invocation offsets."""
    shape = (invocations, iterations)
    floor = 1e-3 * np.exp(rng.normal(0, 0.02, size=(invocations, 1)))
    detour = rng.exponential(2e-4, size=shape) * (rng.random(shape) < 0.1)
    warmup = 1 + 0.5 * np.exp(-np.arange(iterations) / 3.0)
    return (floor + rng.uniform(0, 5e-5, size=shape) + detour) * warmup * (1 + slowdown)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", help="save a figure of the baseline to this file (needs matplotlib)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # Collection: interleave the invocations of A and B in run order.
    a, b = simulate(rng, slowdown=0.0), simulate(rng, slowdown=0.03)
    n_inv, n_iter = a.shape
    time = np.stack([a, b], axis=1).reshape(-1)  # A0, B0, A1, B1, ... by invocation
    variant = np.repeat(np.tile(["A", "B"], n_inv), n_iter)
    invocation = np.repeat(np.repeat(np.arange(n_inv), 2), n_iter)
    m = collection.make_measurements(time, variant=variant, invocation=invocation, benchmark="walkthrough")

    with tempfile.TemporaryDirectory() as tmp:
        collection.save_csv(m, Path(tmp) / "draws.csv")
        m = collection.load_csv(Path(tmp) / "draws.csv")
    print(collection.describe(m))

    # Warmup: MSER per invocation, then mark it in a new record.
    counts = {}
    for v in ("A", "B"):
        x = collection.flatten(m, variant=v, include_warmup=True)
        labels = collection.group_labels(m, "invocation", variant=v, include_warmup=True)
        for g in np.unique(labels):
            counts[(v, 0, int(g))] = characterization.warmup_mser(x[labels == g])
    m = collection.mark_warmup(m, counts)
    print(f"warmup draws dropped per invocation: {sorted(set(counts.values()))}")

    # Characterization of the baseline.
    x = collection.flatten(m, variant="A")
    r = characterization.lmoment_ratios(x)
    print(f"\nA: median {np.median(x) * 1e3:.4f} ms, L-CV {r.cv:.4f}, L-skewness {r.skewness:.3f}, L-kurtosis {r.kurtosis:.3f}")
    u = np.median(x) + 5 * 1.4826 * np.median(np.abs(x - np.median(x)))
    tail = characterization.tail_fit(x, u)
    print(f"detours above {u * 1e3:.4f} ms: rate {tail.rate:.3f}, mean excess {tail.mean_excess * 1e6:.0f} us, xi {tail.xi:+.2f}")
    print(
        f"tau_int {characterization.integrated_autocorrelation_time(x):.2f}, drift {characterization.drift_slope(x):+.2f}% per 1,000 draws"
    )
    try:
        grid, density = characterization.log_density(x)
        modes = characterization.find_modes(grid, density, min_prominence=0.1)
        print(f"modes at {np.round(np.exp(modes.location) * 1e3, 4)} ms holding {np.round(modes.mass, 3)}")
    except ImportError as error:
        print(f"(skipping modes: {error})")

    # Decision: the caller picks the design; several invocations call for "hierarchical".
    print()
    for design in ("independent", "hierarchical"):
        result = decision.compare(m, baseline="A", candidate="B", delta=0.01, design=design, rng=rng)
        print(f"{design:>12}: {result.statement}")

    # Planning: spread of log time, iterations per invocation, and runs for 1% at 80% power.
    vc = decision.variance_components(m, variant="A")
    j = decision.allocate(vc, cost_per_invocation=0.5, cost_per_iteration=1e-3)
    print(f"\nlog-time spread: between invocations {vc.between_invocation:.4f}, within {vc.within_invocation:.4f}")
    print(f"iterations per invocation for a 0.5 s process start: {j}")
    plan = decision.plan_runs(m, variant="A", delta=0.01, design="hierarchical", n_sim=100, rng=rng)
    print(f"plan: {plan.runs} {plan.unit} per variant of {plan.iterations_per_invocation} iterations", end=" ")
    print(f"(power {plan.power:.2f} +/- {plan.power_se:.2f})")

    if args.plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from benchstats import plots

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        plots.run_sequence(m, variant="A", ax=axes[0, 0])
        plots.density(x, ax=axes[0, 1])
        plots.rank_autocorrelation(x, ax=axes[1, 0])
        plots.lmoment_ratio_diagram([x], labels=["A"], ax=axes[1, 1])
        fig.tight_layout()
        fig.savefig(args.plots, dpi=120)
        print(f"\nsaved {args.plots}")


if __name__ == "__main__":
    main()
