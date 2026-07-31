"""Forgetting-rate theory (Cor. 1a) and the examination factorization."""
from __future__ import annotations

import numpy as np
import pytest

from lamprec._release import NOTICE
from lamprec.core.drift import n_eff_star, optimal_lambda, rmse_star
from lamprec.core.slate import gamma_cascade, position_propensity


def test_optimal_lambda_monotone_in_drift():
    """More drift → forget faster (smaller λ*)."""
    lams = [optimal_lambda(V, C=0.05) for V in (1e-4, 1e-3, 1e-2)]
    assert lams[0] > lams[1] > lams[2]


def test_rate_exponents():
    """n_eff* ∝ V^{-2/3} and RMSE* ∝ V^{1/3} (log-log slopes)."""
    V = np.array([1e-4, 1e-3, 1e-2])
    n_eff = np.array([n_eff_star(v, 0.05) for v in V])
    rmse = np.array([rmse_star(v, 0.05) for v in V])
    s_n = np.diff(np.log(n_eff)) / np.diff(np.log(V))
    s_r = np.diff(np.log(rmse)) / np.diff(np.log(V))
    assert np.allclose(s_n, -2.0 / 3.0, atol=1e-6)
    assert np.allclose(s_r, 1.0 / 3.0, atol=1e-6)


def test_cascade_weights_decay_and_bound():
    g = gamma_cascade(10, decay=0.85)
    assert g[0] == 1.0 and np.all(np.diff(g) < 0)
    pe = position_propensity(np.full(10, 0.1), g)
    assert np.all((pe > 0) & (pe <= 0.1))


@pytest.mark.skip(reason=NOTICE)
def test_mse_curve_minimized_near_optimal_lambda():
    """The theoretical MSE(λ) U-curve attains its minimum near λ*."""


@pytest.mark.skip(reason=NOTICE)
def test_factorized_propensity_beats_slate():
    """The per-position overlap floor dominates the per-slate floor."""
