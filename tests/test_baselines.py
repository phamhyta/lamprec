"""Baselines run on the common Stream; the comparative claims ship on release."""
from __future__ import annotations

import numpy as np
import pytest

from lamprec import Stream
from lamprec._release import NOTICE, NotReleasedError
from lamprec.baselines import ALL_BASELINES, naive_feedback, pape


def _stream(seed=0, n=300):
    rng = np.random.default_rng(seed)
    r = (rng.random(n) < 0.3).astype(float)
    g = np.clip(0.3 + 0.5 * (r - 0.3) + rng.normal(0, 0.1, n), 0, 1)
    return Stream(g=g, p=np.ones(n), e=np.full(n, 0.6), r_obs=0.6 * r,
                  delay=rng.exponential(4.0, n).astype(int),
                  theta=np.full(n, 0.3))


def test_naive_and_pape_run_and_shape():
    s = _stream()
    for fn in (naive_feedback, pape):
        out = fn(s)
        assert out["theta_hat"].shape == (s.n,)
        assert np.all(np.isfinite(out["theta_hat"]))


def test_naive_biased_low_by_examination():
    """The naive click average sits near ``e·θ``, below the true θ."""
    s = _stream()
    final = naive_feedback(s, lam=0.995)["theta_hat"][-1]
    assert final < 0.25          # ≈ 0.6 · 0.3 = 0.18 ≪ θ = 0.3


def test_withheld_baselines_raise_release_notice():
    s = _stream()
    for name in ("eta", "iupm", "av_dr", "opecs"):
        with pytest.raises(NotReleasedError):
            ALL_BASELINES[name](s)


@pytest.mark.skip(reason=NOTICE)
def test_all_baselines_run_and_shape():
    """Every baseline consumes the same Stream and returns (n,) estimates."""


@pytest.mark.skip(reason=NOTICE)
def test_lamprec_beats_naive_mae():
    """LAMP-Rec's point estimate has lower MAE than the naive click average."""


@pytest.mark.skip(reason=NOTICE)
def test_lamprec_point_estimate_beats_av_dr():
    """The proxy control variate reduces MAE against proxy-free AV-DR."""


@pytest.mark.skip(reason=NOTICE)
def test_lamprec_cs_competitive_with_av_dr():
    """The proxy-anchored CS is no wider than the proxy-free betting CS."""
