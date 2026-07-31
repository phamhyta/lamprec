"""Simulator sanity; the end-to-end unbiasedness suite ships on release."""
from __future__ import annotations

import numpy as np
import pytest

from lamprec._release import NOTICE, NotReleasedError
from lamprec.sim import drift as drift_mod
from lamprec.sim import stream as stream_mod


def test_stationary_regime_shapes_and_simplex():
    rng = np.random.default_rng(0)
    mu, pi = drift_mod.stationary(100, 20, rng)
    assert mu.shape == (100, 20) and pi.shape == (100, 20)
    assert np.allclose(pi.sum(axis=1), 1.0)
    assert np.all((mu >= 0) & (mu <= 1))


def test_higher_quality_raises_policy_value():
    rng = np.random.default_rng(1)
    mu, _ = drift_mod.stationary(1, 20, rng)
    lo = (drift_mod._policy_from_quality(mu[0], 0.2) * mu[0]).sum()
    hi = (drift_mod._policy_from_quality(mu[0], 0.9) * mu[0]).sum()
    assert hi > lo


def test_simulate_is_withheld():
    with pytest.raises(NotReleasedError, match="not public yet"):
        stream_mod.simulate("stationary", "exponential")


@pytest.mark.skip(reason=NOTICE)
def test_pipeline_unbiased_stationary():
    """The full DR pipeline is unbiased for θ on a stationary stream."""


@pytest.mark.skip(reason=NOTICE)
def test_examination_debias_is_load_bearing():
    """Dropping the 1/e debias re-introduces the position bias."""


@pytest.mark.skip(reason=NOTICE)
def test_redeploy_makes_theta_jump():
    """The abrupt-redeploy regime produces a visible jump in gold θ."""


@pytest.mark.skip(reason=NOTICE)
def test_proxy_quality_controls_rectifier_spread():
    """Raising proxy_rho shrinks the rectifier's standard deviation."""


@pytest.mark.skip(reason=NOTICE)
def test_misspecification_robustness():
    """Graceful degradation under noisy propensities and a wrong cascade decay."""


@pytest.mark.skip(reason=NOTICE)
def test_shrinkage_helps_low_signal_never_hurts():
    """James-Stein debias shrinkage matches naive on low-signal streams."""
