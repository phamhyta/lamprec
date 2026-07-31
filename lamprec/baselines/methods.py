"""Comparison methods, all consuming the common Stream at equal labels-spent.

Because every method sees the *same* matured labels of the *same* stream, the
"equal labels-spent" protocol holds by construction. Point-estimate methods
return ``{'theta_hat': (n,)}``; interval methods also return ``query_t/lo/hi``.

  naive_feedback       -- EMA of raw matured clicks (position-biased, laggy)
  prequential_delayed  -- rolling mean of matured clicks (river); honest but laggy
  prequential_debiased -- rolling mean of examination-debiased clicks (fair)
  adwin_track          -- ADWIN adaptive-window tracker on the debiased reward
  eta                  -- elapsed-time-sampling: inverse-maturity reweighting
  pape                 -- static label-free proxy estimate with a frozen offset
  iupm                 -- OT (quantile) label transport + heuristic-SD band
  opecs                -- arrival-ordered off-policy CS (nearest CS competitor)
  av_dr                -- anytime-valid position-debiased DR-OPE WITHOUT the proxy
"""
from __future__ import annotations

import numpy as np

from lamprec._release import withheld
from lamprec.data.event import Stream


def _matured_ema(stream: Stream, value: np.ndarray, lam: float,
                 weight: np.ndarray | None = None) -> np.ndarray:
    """EMA over matured rounds of ``value`` (optionally extra per-round weight).

    Same injection trick as the LAMP-Rec estimator: a round maturing at ``t``
    enters with forgetting weight ``λ^{d_s}`` and decays by ``λ`` thereafter.
    Value is carried forward between maturations.
    """
    n = stream.n
    groups = stream.groups
    num = den = 0.0
    out = np.empty(n)
    for t in range(n):
        num *= lam
        den *= lam
        for s in groups[t]:
            w = lam ** int(stream.delay[s])
            if weight is not None:
                w *= weight[s]
            num += w * value[s]
            den += w
        out[t] = num / den if den > 0 else 0.0
    return out


def naive_feedback(stream: Stream, lam: float = 0.97) -> dict:
    """Average the realized delayed clicks directly -- no debiasing."""
    return {"theta_hat": _matured_ema(stream, stream.r_obs, lam)}


def prequential_delayed(stream: Stream, window: int = 200) -> dict:
    """Test-then-train rolling mean of matured clicks, via river's RollingMean."""
    withheld("lamprec.baselines.methods.prequential_delayed")


def prequential_debiased(stream: Stream, window: int = 200) -> dict:
    """FAIR prequential: rolling mean of the EXAMINATION-DEBIASED reward
    ``r^obs/e`` over matured rounds (via river) -- the honest delay-aware
    competitor."""
    withheld("lamprec.baselines.methods.prequential_debiased")


def adwin_track(stream: Stream, delta: float = 0.05) -> dict:
    """ADWIN adaptive-window tracker (Bifet & Gavaldà 2007) on the debiased
    reward -- the canonical drift-adaptive competitor to geometric forgetting.
    A point estimate only: no interval, no proxy variance reduction, and no
    delay/maturation model."""
    withheld("lamprec.baselines.methods.adwin_track")


def eta(stream: Stream, lam: float = 0.97, mean_delay: float | None = None) -> dict:
    """Elapsed-Time-sAmpling: reweight matured clicks by inverse maturity prob.

    Corrects the delayed-feedback censoring lag (not the examination bias),
    using an exponential maturity model evaluated at each round's observed
    delay.
    """
    withheld("lamprec.baselines.methods.eta")


def pape(stream: Stream, lam: float = 0.97, calib_frac: float = 0.2) -> dict:
    """Static label-free estimate: proxy EMA shifted by a frozen calibration
    offset (proxy↔truth gap measured once, never recalibrated)."""
    n = stream.n
    c = int(n * calib_frac)
    # debiased reward on the calibration window (uses the few early labels once)
    e = np.clip(stream.e, 1e-3, 1.0)
    matured_c = stream.maturation_time[:c] < n
    truth_c = np.mean((stream.r_obs[:c] / e[:c])[matured_c]) if matured_c.any() else 0.0
    offset = truth_c - np.mean(stream.g[:c])
    # frozen-offset proxy EMA
    g_ema = np.empty(n)
    num = den = 0.0
    for t in range(n):
        num = lam * num + stream.g[t]
        den = lam * den + 1.0
        g_ema[t] = num / den
    return {"theta_hat": g_ema + offset}


def iupm(stream: Stream, lam: float = 0.97, calib_frac: float = 0.2,
         query_times: np.ndarray | None = None) -> dict:
    """Optimal-transport label transport + heuristic-SD band.

    Learns a 1-D OT (quantile) map proxy→debiased-reward on a frozen
    calibration window, then *transports* it forward by applying the stale map
    to the current proxy. Bias accumulates as the proxy distribution drifts
    from calibration; the interval is a heuristic ``±1.96·SD`` with no coverage
    guarantee.
    """
    withheld("lamprec.baselines.methods.iupm")


def av_dr(stream: Stream, lam: float = 0.97, tau_min: float = 0.05,
          alpha: float = 0.05, query_times: np.ndarray | None = None,
          bet: str = "agrapa") -> dict:
    """Anytime-valid position-debiased DR-OPE WITHOUT the label-free proxy.

    Point estimate = EMA of the debiased reward ``r^obs/e`` over matured rounds;
    CS = the same betting engine as LAMP-Rec applied to those samples (no proxy
    control variate, so wider). Isolates the proxy's variance contribution.
    """
    withheld("lamprec.baselines.methods.av_dr")


def opecs(stream: Stream, lam: float = 0.97, tau_min: float = 0.05,
          alpha: float = 0.05, query_times: np.ndarray | None = None,
          bet: str = "agrapa", mat_prob: np.ndarray | None = None,
          rho_max: float = 2.0) -> dict:
    """Off-the-shelf off-policy confidence sequence (arrival order).

    The nearest prior CS competitor: an anytime-valid betting CS for the
    position-debiased off-policy value, processing the logged samples in
    **arrival (serve-index) order** with **no maturation model**. Passing
    ``mat_prob`` adds the IPCW maturation-propensity correction while keeping
    the arrival ordering.
    """
    withheld("lamprec.baselines.methods.opecs")


ALL_BASELINES = {
    "naive_feedback": naive_feedback,
    "prequential_delayed": prequential_delayed,
    "prequential_debiased": prequential_debiased,
    "adwin_track": adwin_track,
    "eta": eta,
    "pape": pape,
    "iupm": iupm,
    "opecs": opecs,
    "av_dr": av_dr,
}
