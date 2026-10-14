"""Plots: figures on top of the other modules. Needs matplotlib (``benchstats[plots]``).

Each function draws on a given matplotlib Axes, or makes a new figure, and
returns the Axes it drew on.
"""

import numpy as np

from benchstats import characterization, collection


def _axes(ax):
    if ax is not None:
        return ax
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("plots need matplotlib: pip install 'benchstats[plots]'") from error
    return plt.figure().add_subplot()


def run_sequence(m, *, variant=None, include_warmup=True, ax=None):
    """Time against run order, colored by invocation."""
    ax = _axes(ax)
    order = collection.run_order(m, variant=variant, include_warmup=include_warmup)
    time = collection.flatten(m, variant=variant, include_warmup=include_warmup)
    labels = collection.group_labels(m, "invocation", variant=variant, include_warmup=include_warmup)
    ax.scatter(order, time, c=labels, cmap="tab20", s=6)
    ax.set_xlabel("run order")
    ax.set_ylabel("time (s)")
    ax.set_yscale("log")
    return ax


def density(x, *, min_prominence=0.1, bandwidth="silverman", ax=None):
    """Density of log times, with the modes marked."""
    ax = _axes(ax)
    grid, dens = characterization.log_density(x, bandwidth=bandwidth)
    modes = characterization.find_modes(grid, dens, min_prominence=min_prominence)
    ax.plot(grid, dens)
    ax.plot(modes.location, modes.height, "v")
    for loc, height, mass in zip(modes.location, modes.height, modes.mass):
        ax.annotate(f"{mass:.0%}", (loc, height), textcoords="offset points", xytext=(0, 6), ha="center")
    ax.set_xlabel("log time (log s)")
    ax.set_ylabel("density")
    return ax


def lmoment_ratio_diagram(samples, *, labels=None, ax=None):
    """L-skewness against L-kurtosis, with the generalized Pareto and normal references.

    ``samples`` is a sequence of raw-time arrays; each becomes one point.
    """
    ax = _axes(ax)
    t3 = np.linspace(-0.2, 0.95, 200)
    ax.plot(t3, t3 * (1 + 5 * t3) / (5 + t3), label="generalized Pareto")
    ax.plot([0.0], [0.1226], "s", label="normal")
    ax.plot([1 / 3], [1 / 6], "D", label="exponential")
    for i, x in enumerate(samples):
        r = characterization.lmoment_ratios(x)
        ax.plot(r.skewness, r.kurtosis, "o", label=None if labels is None else labels[i])
    ax.set_xlabel("L-skewness tau3")
    ax.set_ylabel("L-kurtosis tau4")
    ax.legend()
    return ax


def threshold_scan(x, thresholds, *, axes=None):
    """xi and mean excess against the threshold. Returns two Axes."""
    if axes is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(2, 1, sharex=True)
    scan = characterization.threshold_scan(x, thresholds)
    axes[0].plot(scan.threshold, scan.xi, "o-")
    axes[0].axhline(0.0, color="gray", lw=0.8)
    axes[0].set_ylabel("xi")
    axes[1].plot(scan.threshold, scan.mean_excess, "o-")
    axes[1].set_ylabel("mean excess (s)")
    axes[1].set_xlabel("threshold (s)")
    return axes


def rank_autocorrelation(x, max_lag=40, *, ax=None):
    """Rank autocorrelation against lag, with the +/- 2/sqrt(n) band for independent draws."""
    ax = _axes(ax)
    x = np.asarray(x, dtype=float)
    rho = characterization.rank_autocorrelation(x, min(max_lag, x.size - 1))
    lags = np.arange(1, rho.size + 1)
    ax.vlines(lags, 0, rho)
    band = 2 / np.sqrt(x.size)
    ax.axhspan(-band, band, color="gray", alpha=0.2)
    ax.set_xlabel("lag")
    ax.set_ylabel("rank autocorrelation")
    return ax


def comparison_table(comparisons, *, ax=None):
    """A table of a suite's comparisons: benchmark, change, interval, verdict."""
    ax = _axes(ax)
    rows = []
    for c in comparisons:
        if c.effect == "ratio":
            cells = [f"{100 * np.expm1(v):+.1f}%" for v in (c.estimate, c.low, c.high)]
        else:
            cells = [f"{v:+.3g}" for v in (c.estimate, c.low, c.high)]
        rows.append([c.benchmark, c.statistic, *cells, c.verdict.outcome.value])
    ax.axis("off")
    ax.table(cellText=rows, colLabels=["benchmark", "statistic", "change", "low", "high", "verdict"], loc="center")
    return ax
