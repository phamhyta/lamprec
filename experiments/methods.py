"""Shared method registry used by the RQ runners (paper row order)."""
from __future__ import annotations

import numpy as np

from lamprec._release import withheld
from lamprec.baselines import (adwin_track, av_dr, eta, iupm, naive_feedback,
                               pape, prequential_debiased, prequential_delayed)

LAM = 0.99
TAU_MIN = 0.05


def lamprec_point(stream, lam: float = LAM, tau_min: float = TAU_MIN):
    """MSE-optimal point estimate: variance-aware shrinkage of the examination
    debias + online proxy recalibration. Falls back to the naive/AV-DR level on
    real logs with no informative proxy, and keeps the full PPI gain under
    drift; the confidence sequence still uses the unbiased rectifier."""
    withheld("experiments.methods.lamprec_point")


# Point-estimate methods for the MAE table, in paper row order. The first block
# is uncorrected/biased; the middle block is FAIR examination-corrected
# trackers; the last block is the DR estimators.
POINT_METHODS = {
    "Naive-Feedback": lambda s: naive_feedback(s, lam=LAM)["theta_hat"],
    "Prequential-Delayed": lambda s: prequential_delayed(s, window=200)["theta_hat"],
    "ETA": lambda s: eta(s, lam=LAM)["theta_hat"],
    "Prequential-Debiased": lambda s: prequential_debiased(s, window=200)["theta_hat"],
    "ADWIN": lambda s: adwin_track(s)["theta_hat"],
    "PAPE": lambda s: pape(s, lam=LAM)["theta_hat"],
    "IUPM": lambda s: iupm(s, lam=LAM)["theta_hat"],
    "AV-DR": lambda s: av_dr(s, lam=LAM, tau_min=TAU_MIN,
                             query_times=np.array([s.n - 1]))["theta_hat"],
    "LAMP-Rec": lamprec_point,
}
