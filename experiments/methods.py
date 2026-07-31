"""Shared method registry used by the RQ runners (paper row order)."""
from __future__ import annotations

import json
import os

import numpy as np

from experiments.protocol import RESULTS
from lamprec.__about__ import SEEDS, TUNING_SEEDS
from lamprec._release import withheld
from lamprec.baselines import (adwin_track, av_dr, eta, iupm, naive_feedback,
                               pape, prequential_debiased, prequential_delayed)

# Hyperparameters are frozen by experiments/tuning.py on TUNING_SEEDS (100-109),
# which are disjoint from the reported eval SEEDS -- see results/tuning/ for the
# selection rules and evidence. Fallback = the same values, so a missing tuning
# artifact never silently changes the reported numbers.
assert not set(SEEDS) & set(TUNING_SEEDS), "tuning/eval seed split violated"
_SEL_PATH = os.path.join(RESULTS, "tuning", "selected.json")
if os.path.exists(_SEL_PATH):
    with open(_SEL_PATH) as f:
        _SEL = json.load(f)
    LAM = float(_SEL["lam_synthetic"])
    TAU_MIN = float(_SEL["tau_min"])
else:                                     # pre-tuning fallback (same values)
    LAM = 0.99
    TAU_MIN = 0.05


def av_dr_shrink(stream, lam: float = LAM, tau_min: float = TAU_MIN):
    """AV-DR with LAMP-Rec's variance-aware debias shrinkage (fairness row).

    Same shrunken examination-debiased matured EMA as LAMP-Rec, with the proxy
    zeroed out. Isolates what the *shrinkage* contributes vs what the *proxy*
    contributes.
    """
    withheld("experiments.methods.av_dr_shrink")


def lamprec_point(stream, lam: float = LAM, tau_min: float = TAU_MIN):
    """MSE-optimal point estimate: variance-aware shrinkage of the examination
    debias + slope-only centered power-tuning of the proxy (β → 0 when the
    proxy is uninformative, killing the proxy term identically), with the
    anytime-valid examination gate armed. Reduces to the naive/AV-DR matured
    EMA on real logs with no informative proxy, keeps the full PPI gain under
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
    "AV-DR (+shrink)": av_dr_shrink,
    "LAMP-Rec": lamprec_point,
}
