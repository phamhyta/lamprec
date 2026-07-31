"""Core correctness: schema, rectifier identity, and the release gate.

The full suite (PPI identity, EMA recursion, anytime-valid coverage under
out-of-order maturation, aGRAPA-vs-fixed width) ships with the released
implementation; here we exercise what the public skeleton implements and assert
the withheld components raise the release notice.
"""
from __future__ import annotations

import numpy as np
import pytest

from lamprec import Stream
from lamprec._release import NOTICE, NotReleasedError
from lamprec.core.estimator import LampRecConfig, rectifier, run_estimator


def ppi_stream(n: int, theta: float, rho_strength: float, delay_mean: float,
               seed: int, censored_frac: float = 0.0) -> Stream:
    """Stationary stream with known constant value ``theta``.

    ``c_s ~ Bernoulli(theta)`` is the (delayed) reward; the proxy tracks it with
    strength ``rho_strength``. ``p = e = 1`` so the internal rectifier is
    ``g_s - c_s``.
    """
    rng = np.random.default_rng(seed)
    c = (rng.random(n) < theta).astype(float)                 # true reward
    g = np.clip(theta + rho_strength * (c - theta)
                + (1 - rho_strength) * rng.normal(0, 0.15, n), 0, 1)
    delay = rng.exponential(delay_mean, n).astype(int)
    if censored_frac > 0:
        cens = rng.random(n) < censored_frac
        delay[cens] = 10 ** 9                                  # never matures
    return Stream(
        g=g, g_action=g, r_obs=c, p=np.ones(n), e=np.ones(n),
        delay=delay, theta=np.full(n, theta),
    )


def test_stream_schema_validates_lengths():
    s = ppi_stream(n=50, theta=0.4, rho_strength=0.7, delay_mean=3.0, seed=0)
    assert s.n == 50
    assert s.maturation_time.shape == (50,)
    with pytest.raises(ValueError):
        Stream(g=np.zeros(5), p=np.ones(4), e=np.ones(5), r_obs=np.zeros(5),
               delay=np.zeros(5, dtype=int), theta=np.zeros(5))


def test_rectifier_identity_at_unit_propensities():
    """At ``p = e = 1`` the rectifier reduces to ``g_action - r_obs``."""
    s = ppi_stream(n=200, theta=0.3, rho_strength=0.8, delay_mean=5.0, seed=1)
    r, omega = rectifier(s, tau_min=1.0)
    assert np.allclose(r, s.g_action - s.r_obs)
    assert np.allclose(omega, 1.0)


def test_withheld_estimator_raises_release_notice():
    s = ppi_stream(n=20, theta=0.4, rho_strength=0.7, delay_mean=2.0, seed=2)
    with pytest.raises(NotReleasedError, match="not public yet"):
        run_estimator(s, LampRecConfig())
    assert "released in this repository once the paper is accepted" in NOTICE


@pytest.mark.skip(reason=NOTICE)
def test_ema_recursion_matches_bruteforce():
    """Streaming θ̂_t equals the direct λ-weighted formula (no delay)."""


@pytest.mark.skip(reason=NOTICE)
def test_ppi_identity_recovers_theta():
    """On a stationary stream θ̂_t concentrates on the true value."""


@pytest.mark.skip(reason=NOTICE)
def test_proxy_reduces_variance():
    """Higher proxy↔reward correlation shrinks the final-estimate spread."""


@pytest.mark.skip(reason=NOTICE)
def test_anytime_valid_coverage_under_delay():
    """The θ-CS covers the true value time-uniformly under out-of-order maturation."""


@pytest.mark.skip(reason=NOTICE)
def test_agrapa_narrows_width_vs_fixed():
    """The adaptive aGRAPA bet gives narrower intervals than the fixed bet."""
