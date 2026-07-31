"""Drift regimes -- relevance ``μ_s(a)`` and policy ``π_s(a)`` trajectories.

Each builder returns ``(mu, pi)`` of shape ``(n, K)``; the gold value is
``θ_s = Σ_a π_s(a) μ_s(a)`` (computed in ``sim/stream.py``). A policy is an
interpolation between uniform and near-greedy controlled by a per-round
*quality* ``q_s ∈ [0,1]`` -- raising ``q`` concentrates mass on high-relevance
items and raises ``θ``.
"""
from __future__ import annotations

import numpy as np

from lamprec._release import withheld


def _greedy(mu: np.ndarray, temp: float = 0.05) -> np.ndarray:
    """Near-greedy softmax policy over a relevance vector."""
    z = mu / temp
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def _policy_from_quality(mu: np.ndarray, q: float) -> np.ndarray:
    """Interpolate uniform → near-greedy by quality ``q ∈ [0,1]``."""
    K = mu.shape[0]
    return (1.0 - q) * np.full(K, 1.0 / K) + q * _greedy(mu)


def _base_relevance(K: int, rng: np.random.Generator, mean: float = 0.3,
                    spread: float = 0.22) -> np.ndarray:
    return np.clip(rng.normal(mean, spread, K), 0.02, 0.98)


def stationary(n: int, K: int, rng: np.random.Generator, quality: float = 0.6,
               mean: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    mu0 = _base_relevance(K, rng, mean)
    pi0 = _policy_from_quality(mu0, quality)
    return np.tile(mu0, (n, 1)), np.tile(pi0, (n, 1))


def abrupt_redeploy(n: int, K: int, rng: np.random.Generator, redeploy_at: int,
                    q_before: float = 0.45, q_after: float = 0.75,
                    decay: float = 0.15, mean: float = 0.3
                    ) -> tuple[np.ndarray, np.ndarray]:
    """A better policy ships at ``redeploy_at``: θ rises, jumps, then decays."""
    withheld("lamprec.sim.drift.abrupt_redeploy")


def popularity_shift(n: int, K: int, rng: np.random.Generator, quality: float = 0.7,
                     shift_rate: float = 0.6, mean: float = 0.3
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Relevance drifts (a slow rotation of which items are good) while the
    policy stays greedy on the *initial* relevance → silent staleness decay."""
    withheld("lamprec.sim.drift.popularity_shift")


def drifting(n: int, K: int, rng: np.random.Generator, drift_per_step: float,
             quality0: float = 0.55, amp: float = 0.25, mean: float = 0.3
             ) -> tuple[np.ndarray, np.ndarray]:
    """Oscillating drift whose per-step ``|Δθ|`` matches ``drift_per_step``."""
    withheld("lamprec.sim.drift.drifting")


def quality_step(n: int, K: int, rng: np.random.Generator, drop_at: int,
                 q_high: float = 0.7, q_low: float = 0.55, mean: float = 0.3
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Abrupt quality drop at ``drop_at`` (a degradation to be alarmed on)."""
    withheld("lamprec.sim.drift.quality_step")


def performative_loop(n: int, K: int, rng: np.random.Generator, kappa: float,
                      quality: float = 0.7, eta: float = 0.1, perf: float = 0.5,
                      mean: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Decision-dependent relevance: recommendations feed back into popularity.

    ``κ < 1`` is contractive (popularity settles); ``κ ≥ 1`` is expansive
    (feedback amplification / popularity collapse) → growing drift.
    """
    withheld("lamprec.sim.drift.performative_loop")
