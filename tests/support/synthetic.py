"""Synthetic timing data with known true values.

Marginal distributions are small frozen dataclasses with ``cdf``, ``ppf``,
``sample`` and closed-form or numerically integrated true values. True values
never come from a large simulation through the same code path.

Dependence and structure are added by functions that take a marginal:
``correlated``, ``bursty``, ``hierarchical``, ``add_warmup`` and ``add_drift``.

All times are float64 seconds.
"""

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy import integrate, optimize, signal, special, stats

# ---------------------------------------------------------------- helpers


def _shifted_legendre_integral(r, u):
    """Q_r(u) = integral from 0 to u of the shifted Legendre polynomial P*_{r-1}.

    For r >= 2, lambda_r = -integral of Q_r(F(x)) dx over the support.
    """
    coeffs = special.sh_legendre(r - 1).integ()  # zero constant term
    return coeffs(u)


def _ppf_by_root(dist, q, lower, upper):
    """Invert dist.cdf by bracketing root finding, elementwise."""
    q = np.asarray(q, dtype=float)
    result = optimize.elementwise.find_root(
        lambda x, target: dist.cdf(x) - target,
        (np.full_like(q, lower), np.full_like(q, upper)),
        args=(q,),
        tolerances={"xatol": 1e-14, "xrtol": 1e-13},
    )
    return result.x


class _Marginal:
    """Shared machinery for marginals defined by cdf and pdf."""

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        if np.any((q <= 0) | (q >= 1)):
            raise ValueError("ppf needs 0 < q < 1")
        return _ppf_by_root(self, q, self.support_min, self._upper_bracket())

    def _upper_bracket(self):
        upper = self.support_min + 1.0
        while self.cdf(upper) < 1 - 1e-15:
            upper = self.support_min + 2 * (upper - self.support_min)
        return upper

    def median(self):
        return float(self.ppf(0.5))

    def _integrate(self, f):
        """Integral of f(x) dx from the support minimum to infinity."""
        total = 0.0
        breaks = list(self._breakpoints()) + [np.inf]
        for a, b in zip(breaks[:-1], breaks[1:]):
            value, _ = integrate.quad(f, a, b, limit=500, epsabs=1e-13, epsrel=1e-10)
            total += value
        return total

    def _breakpoints(self):
        return (self.support_min,)

    def lmoments(self):
        """True L-moments lambda_1..lambda_4 by numerical integration."""
        l1 = self.support_min + self._integrate(lambda x: 1 - self.cdf(x))
        rest = [-self._integrate(lambda x, r=r: _shifted_legendre_integral(r, self.cdf(x))) for r in (2, 3, 4)]
        return np.array([l1, *rest])

    def log_mad_scale(self):
        """Scaled MAD of log X: 1.4826 * median |log X - median log X|."""
        m = math.log(self.median())

        def mass_within(d):
            return float(self.cdf(math.exp(m + d)) - self.cdf(math.exp(m - d))) - 0.5

        d = optimize.brentq(mass_within, 1e-12, 50.0, xtol=1e-14)
        return 1.4826022185056018 * d


# ---------------------------------------------------------------- marginals


@dataclass(frozen=True)
class FloorPlusDetours(_Marginal):
    """A fixed floor, uniform jitter of width ``jitter``, and at most one detour.

    Detour source k is chosen with probability ``probs[k]`` and adds an
    exponential delay with mean ``means[k]``. With probability
    1 - sum(probs) there is no detour.
    """

    floor: float
    jitter: float
    probs: tuple[float, ...] = ()
    means: tuple[float, ...] = ()

    def __post_init__(self):
        if self.floor <= 0 or self.jitter <= 0:
            raise ValueError("floor and jitter must be positive")
        if len(self.probs) != len(self.means):
            raise ValueError("probs and means must have the same length")
        if any(p < 0 for p in self.probs) or sum(self.probs) > 1:
            raise ValueError("probs must be non-negative and sum to at most 1")
        if any(mu <= 0 for mu in self.means):
            raise ValueError("detour means must be positive")

    @property
    def support_min(self):
        return self.floor

    def _breakpoints(self):
        return (self.floor, self.floor + self.jitter)

    def sample(self, n, rng):
        jitter = rng.uniform(0.0, self.jitter, size=n)
        p_none = 1.0 - sum(self.probs)
        source = rng.choice(len(self.probs) + 1, size=n, p=[p_none, *self.probs])
        means = np.array([0.0, *self.means])
        detour = rng.exponential(1.0, size=n) * means[source]
        return self.floor + jitter + detour

    def _uniform_plus_exp_cdf(self, t, mu):
        """P(J + E <= t) with J ~ U(0, w) and E ~ Exp(mean mu)."""
        w = self.jitter
        t = np.asarray(t, dtype=float)
        inside = (t - mu * -np.expm1(-np.clip(t, 0, None) / mu)) / w
        beyond = 1 - (mu / w) * np.expm1(w / mu) * np.exp(-t / mu)
        return np.where(t <= 0, 0.0, np.where(t <= w, inside, beyond))

    def _uniform_plus_exp_pdf(self, t, mu):
        w = self.jitter
        t = np.asarray(t, dtype=float)
        inside = -np.expm1(-np.clip(t, 0, None) / mu) / w
        beyond = (1 / w) * np.expm1(w / mu) * np.exp(-t / mu)
        return np.where(t <= 0, 0.0, np.where(t <= w, inside, beyond))

    def cdf(self, x):
        t = np.asarray(x, dtype=float) - self.floor
        p_none = 1.0 - sum(self.probs)
        result = p_none * np.clip(t / self.jitter, 0.0, 1.0)
        for p, mu in zip(self.probs, self.means):
            result = result + p * self._uniform_plus_exp_cdf(t, mu)
        return result

    def pdf(self, x):
        t = np.asarray(x, dtype=float) - self.floor
        p_none = 1.0 - sum(self.probs)
        result = np.where((t > 0) & (t <= self.jitter), p_none / self.jitter, 0.0)
        for p, mu in zip(self.probs, self.means):
            result = result + p * self._uniform_plus_exp_pdf(t, mu)
        return result

    def mean(self):
        return self.floor + self.jitter / 2 + sum(p * mu for p, mu in zip(self.probs, self.means))

    def _excess_terms(self, u):
        """Unnormalized weights p_k c_k exp(-(u - floor)/mu_k) of the excess mixture."""
        if u < self.floor + self.jitter:
            raise ValueError("closed forms hold only above floor + jitter")
        w = self.jitter
        return np.array([p * (mu / w) * math.expm1(w / mu) * math.exp(-(u - self.floor) / mu) for p, mu in zip(self.probs, self.means)])

    def exceedance_rate(self, u):
        """P(X > u), exact for u >= floor + jitter."""
        return float(self._excess_terms(u).sum())

    def mean_excess(self, u):
        """E[X - u | X > u], exact for u >= floor + jitter.

        Above floor + jitter the excess is a mixture of exponentials with
        weights proportional to ``_excess_terms`` and means ``means``.
        """
        terms = self._excess_terms(u)
        return float(terms @ np.array(self.means) / terms.sum())

    def broken(self, factor):
        """The same distribution with every detour mean multiplied by factor."""
        return FloorPlusDetours(self.floor, self.jitter, self.probs, tuple(factor * mu for mu in self.means))


@dataclass(frozen=True)
class GPDTail:
    """Generalized Pareto draws with shape xi, scale sigma, and location loc."""

    shape: float
    scale: float = 1.0
    loc: float = 0.0

    @cached_property
    def _dist(self):
        return stats.genpareto(c=self.shape, loc=self.loc, scale=self.scale)

    @property
    def support_min(self):
        return self.loc

    def sample(self, n, rng):
        return self._dist.rvs(size=n, random_state=rng)

    def cdf(self, x):
        return self._dist.cdf(x)

    def pdf(self, x):
        return self._dist.pdf(x)

    def ppf(self, q):
        return self._dist.ppf(q)

    def median(self):
        return float(self._dist.median())

    def lmoments(self):
        """Closed-form L-moments, valid for shape < 1."""
        xi, s = self.shape, self.scale
        l1 = self.loc + s / (1 - xi)
        l2 = s / ((1 - xi) * (2 - xi))
        t3 = (1 + xi) / (3 - xi)
        t4 = (1 + xi) * (2 + xi) / ((3 - xi) * (4 - xi))
        return np.array([l1, l2, t3 * l2, t4 * l2])


@dataclass(frozen=True)
class Mixture(_Marginal):
    """A mixture of marginals with the given weights."""

    components: tuple
    weights: tuple[float, ...]

    def __post_init__(self):
        if len(self.components) != len(self.weights):
            raise ValueError("components and weights must have the same length")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-12):
            raise ValueError("weights must sum to 1")

    @property
    def support_min(self):
        return min(c.support_min for c in self.components)

    def _breakpoints(self):
        points = set()
        for c in self.components:
            points.update(getattr(c, "_breakpoints", lambda c=c: (c.support_min,))())
        return tuple(sorted(points))

    def sample(self, n, rng):
        which = rng.choice(len(self.components), size=n, p=self.weights)
        out = np.empty(n)
        for k, c in enumerate(self.components):
            mask = which == k
            out[mask] = c.sample(int(mask.sum()), rng)
        return out

    def cdf(self, x):
        return sum(w * c.cdf(x) for w, c in zip(self.weights, self.components))

    def pdf(self, x):
        return sum(w * c.pdf(x) for w, c in zip(self.weights, self.components))


def two_floors(floors, jitters, weights):
    """Two code paths with different floors, each with uniform jitter.

    When the floors are well separated relative to the jitter, there are two
    modes. On the log scale, path k's mode sits near log(floor_k + jitter_k/2)
    and holds probability weights[k]; see ``two_floor_modes``.
    """
    components = tuple(FloorPlusDetours(f, w) for f, w in zip(floors, jitters))
    return Mixture(components, tuple(weights))


def two_floor_modes(mixture):
    """True (log location, mass) of each mode of a two_floors mixture."""
    locations = np.array([math.log(c.floor + c.jitter / 2) for c in mixture.components])
    return locations, np.array(mixture.weights)


# ---------------------------------------------------------------- dependence


def correlated(marginal, phi, n, rng):
    """Draws in run order with dependence from a Gaussian AR(1) copula.

    The latent series starts in its stationary distribution, so every draw
    has the marginal exactly.
    """
    if not -1 < phi < 1:
        raise ValueError("phi must lie in (-1, 1)")
    shocks = rng.standard_normal(n)
    scale = math.sqrt(1 - phi * phi)
    # z[0] = shocks[0]; z[t] = phi * z[t-1] + scale * shocks[t].
    rest = signal.lfilter([scale], [1.0, -phi], shocks[1:], zi=[phi * shocks[0]])[0]
    z = np.r_[shocks[0], rest]
    return marginal.ppf(np.clip(stats.norm.cdf(z), 1e-16, 1 - 1e-16))


def correlated_rank_autocorrelation(phi, lags):
    """True rank autocorrelation of ``correlated`` at the given lags."""
    lags = np.asarray(lags)
    return (6 / math.pi) * np.arcsin(phi**lags / 2)


def correlated_tau_int(phi, max_lag=100_000):
    """True integrated autocorrelation time of the ranks of ``correlated``."""
    rho = correlated_rank_autocorrelation(phi, np.arange(1, max_lag + 1))
    return 1 + 2 * float(rho.sum())


@dataclass(frozen=True)
class BurstyModel:
    """A two-state Markov chain: quiet and noisy regimes with their own marginals."""

    quiet: object
    noisy: object
    stay_quiet: float
    stay_noisy: float

    @property
    def stationary(self):
        """Stationary probabilities of (quiet, noisy)."""
        leave_q, leave_n = 1 - self.stay_quiet, 1 - self.stay_noisy
        return leave_n / (leave_q + leave_n), leave_q / (leave_q + leave_n)

    @property
    def marginal(self):
        return Mixture((self.quiet, self.noisy), self.stationary)


def bursty(model, n, rng):
    """Draws whose detours arrive in clusters.

    The chain starts in its stationary distribution, so every draw has
    ``model.marginal`` exactly.
    """
    state = np.empty(n, dtype=bool)  # True means noisy
    u = rng.random(n)
    state[0] = u[0] < model.stationary[1]
    for t in range(1, n):
        stay = model.stay_noisy if state[t - 1] else model.stay_quiet
        state[t] = state[t - 1] if u[t] < stay else not state[t - 1]
    out = np.empty(n)
    out[~state] = model.quiet.sample(int((~state).sum()), rng)
    out[state] = model.noisy.sample(int(state.sum()), rng)
    return out


# ---------------------------------------------------------------- structure


@dataclass(frozen=True)
class HierarchicalDraws:
    """Times with build, invocation and iteration labels, in run order."""

    time: np.ndarray
    build: np.ndarray
    invocation: np.ndarray
    iteration: np.ndarray


def hierarchical(base, *, n_builds, n_invocations, n_iterations, build_sd, invocation_sd, rng):
    """Builds and invocations, each with a random multiplicative offset.

    Each build multiplies its times by exp(N(0, build_sd^2)), each invocation
    by exp(N(0, invocation_sd^2)), around within-invocation draws from
    ``base``. Invocations are numbered within each build.
    """
    shape = (n_builds, n_invocations, n_iterations)
    log_offset = rng.normal(0.0, build_sd, size=(n_builds, 1, 1)) + rng.normal(0.0, invocation_sd, size=(n_builds, n_invocations, 1))
    time = base.sample(math.prod(shape), rng).reshape(shape) * np.exp(log_offset)
    build, invocation, iteration = np.indices(shape)
    return HierarchicalDraws(time.ravel(), build.ravel(), invocation.ravel(), iteration.ravel())


def hierarchical_cdf(base, total_sd, x):
    """True CDF of all draws from ``hierarchical``, over random offsets.

    log X = Z + log Y with Z ~ N(0, total_sd^2) independent of Y ~ base, so
    P(X <= x) = E_Z[F_Y(x exp(-Z))], computed by Gauss-Hermite quadrature.
    total_sd^2 = build_sd^2 + invocation_sd^2.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    weights = weights / weights.sum()
    x = np.asarray(x, dtype=float)
    return base.cdf(x[..., np.newaxis] * np.exp(-total_sd * nodes)) @ weights


def hierarchical_median(base, total_sd):
    """True median of all draws from ``hierarchical``, over random offsets."""
    m = base.median()
    lo, hi = m * math.exp(-10 * total_sd - 1), m * math.exp(10 * total_sd + 1)
    return optimize.brentq(lambda x: float(hierarchical_cdf(base, total_sd, x)) - 0.5, lo, hi, xtol=1e-15, rtol=1e-13)


def add_warmup(time, iteration, *, amplitude, decay):
    """Multiply each draw by 1 + amplitude * exp(-iteration / decay).

    The steady-state distribution is the distribution of ``time``.
    """
    return np.asarray(time) * (1 + amplitude * np.exp(-np.asarray(iteration) / decay))


def add_drift(time, order, *, total_change):
    """A steady trend in run order: log time rises linearly.

    Over the whole experiment the times rise by the factor 1 + total_change.
    Returns the new times and the true slope of log time per draw.
    """
    order = np.asarray(order, dtype=float)
    span = order.max() - order.min()
    slope = math.log1p(total_change) / span
    return np.asarray(time) * np.exp(slope * (order - order.min())), slope
