"""Maturation-delay models -- the ``d_s`` column of a simulated Stream.

Each builder returns an integer delay array of length ``n``. A structurally
censored tail (label never matures in-window) is encoded with ``NEVER``.

``informative_delay`` deliberately makes the delay depend on the latent reward
(violating Assumption D); it also returns the per-round maturation propensity
``P(d_s ≤ horizon)`` so the IPCW correction can be supplied to the monitor.
"""
from __future__ import annotations

import numpy as np

from lamprec._release import withheld
from lamprec.data.event import NEVER


def _censor(delay: np.ndarray, rng: np.random.Generator, frac: float) -> np.ndarray:
    if frac > 0:
        delay = delay.copy()
        delay[rng.random(delay.size) < frac] = NEVER
    return delay


def constant_delay(n: int, d: int) -> np.ndarray:
    return np.full(n, int(d))


def exponential_delay(n: int, mean: float, rng: np.random.Generator,
                      censored_frac: float = 0.0) -> np.ndarray:
    d = np.ceil(rng.exponential(mean, n)).astype(int)
    return _censor(d, rng, censored_frac)


def lognormal_delay(n: int, median: float, sigma: float, rng: np.random.Generator,
                    censored_frac: float = 0.0) -> np.ndarray:
    d = np.ceil(rng.lognormal(np.log(max(median, 1e-6)), sigma, n)).astype(int)
    return _censor(d, rng, censored_frac)


def heavytail_delay(n: int, scale: float, alpha: float, rng: np.random.Generator,
                    censored_frac: float = 0.05) -> np.ndarray:
    """Pareto (heavy-tailed) maturation; a censored tail is on by default."""
    withheld("lamprec.sim.delay.heavytail_delay")


def informative_delay(n: int, quality: np.ndarray, base_mean: float,
                      rng: np.random.Generator, strength: float = 4.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Delay correlated with the latent relevance (Assumption-D violation).

    High-relevance rounds mature fast; low-relevance rounds mature slowly, so
    at any time the matured set over-represents high relevance, biasing the
    naive estimate and breaking the naive martingale's coverage. Returns
    ``(delay, delay_means)`` -- the per-round exponential-delay mean, i.e. the
    oracle delay LAW, from which the maturation propensity is evaluated at the
    ELAPSED LAG ``π^mat_{t,s} = P(d_s ≤ t−s)``, matching the paper's definition
    (a fixed-window propensity is horizon-inconsistent).
    """
    withheld("lamprec.sim.delay.informative_delay")


def mat_prob_at_lag(delay_means: np.ndarray, lag) -> np.ndarray:
    """``P(d_s ≤ lag)`` under the exponential delay law (elementwise)."""
    return 1.0 - np.exp(-np.asarray(lag, dtype=float) / delay_means)


def mat_prob_fixed_lag(delay_means: np.ndarray, lam: float) -> np.ndarray:
    """Horizon-consistent per-round propensity for an EMA estimator.

    Evaluates the delay CDF at the estimator's own effective horizon
    ``L = ⌈1/(1−λ)⌉`` -- the dominant lag of the forgetting weights -- so the
    IPCW weight is right where the EMA mass sits.
    """
    withheld("lamprec.sim.delay.mat_prob_fixed_lag")
