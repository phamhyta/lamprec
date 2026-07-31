"""Forgetting / drift controller -- Corollary 1a.

Closed-form optimal forgetting rate and the online drift plug-in. Minimizing

    MSE(β) ≈ V_t²/β²  +  (C_t/2)·β,     β = 1-λ

gives ``1 - λ* ≍ (V_t²/C_t)^{1/3}``, ``n_eff* ≍ (C_t/V_t²)^{1/3}`` and the
achievable ``RMSE(λ*) ≍ C_t^{1/3} V_t^{1/3}`` -- the ``V_t^{1/3}`` error rate.
"""
from __future__ import annotations

import numpy as np

from lamprec._release import withheld


def optimal_lambda(V: float, C: float) -> float:
    """Closed-form optimal forgetting factor ``λ*`` from Cor. 1a.

    ``V`` is the local per-step drift, ``C = Θ(σ_g² + ρ̄ σ_r²)`` the variance
    constant. Returns ``λ* = 1 - (V²/C)^{1/3}`` clipped to ``(0, 1)``.
    """
    if V <= 0:
        return 0.999
    beta = (V ** 2 / max(C, 1e-12)) ** (1.0 / 3.0)
    return float(np.clip(1.0 - beta, 1e-3, 1.0 - 1e-3))


def n_eff_star(V: float, C: float) -> float:
    """Optimal effective memory ``n_eff* ≍ (C/V²)^{1/3} ∝ V^{-2/3}``."""
    if V <= 0:
        return np.inf
    return float((max(C, 1e-12) / V ** 2) ** (1.0 / 3.0))


def rmse_star(V: float, C: float) -> float:
    """Achievable RMSE at ``λ*``: ``C^{1/3} V^{1/3}`` (the ``V^{1/3}`` rate)."""
    return float((max(C, 0.0) ** (1.0 / 3.0)) * (max(V, 0.0) ** (1.0 / 3.0)))


def mse_curve(V: float, C: float, lambdas: np.ndarray) -> np.ndarray:
    """Theoretical MSE(λ) = (V·λ/(1-λ))² + C·(1-λ)/(1+λ) -- the U-curve."""
    withheld("lamprec.core.drift.mse_curve")


def estimate_drift(theta_hat: np.ndarray, window: int = 50,
                   noise_floor: float | None = None) -> np.ndarray:
    """Online plug-in ``V̂_t`` from successive estimate gaps.

    Smooths ``|θ̂_t - θ̂_{t-1}|`` over a trailing window; if ``noise_floor`` is
    given it is subtracted (the part of the gap due to estimator noise rather
    than genuine drift). Returns a per-round drift estimate.
    """
    withheld("lamprec.core.drift.estimate_drift")
