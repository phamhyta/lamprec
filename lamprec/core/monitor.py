"""Algorithm 1 orchestrator: run the LAMP-Rec monitor over a Stream.

Bundles the streaming point estimate (:mod:`lamprec.core.estimator`) with the
anytime-valid confidence sequence (:mod:`lamprec.core.confseq`) and the
degradation alarm into one call -- the "method" the experiments invoke.
"""
from __future__ import annotations

import numpy as np

from lamprec.core.confseq import CSConfig, alarm_time, run_confseq
from lamprec.core.estimator import LampRecConfig, run_estimator
from lamprec.data.event import Stream


def run_monitor(stream: Stream, cfg: LampRecConfig | None = None,
                cs: CSConfig | None = None, query_times: np.ndarray | None = None,
                threshold: float | None = None) -> dict:
    """Run the full monitor; return point estimate, CS over time, and alarm time.

    Keys: ``theta_hat, g_hat, r_hat, n_eff_r`` (per round); ``t, lo, hi`` (CS at
    query times); ``alarm`` (first query time the CS upper bound < ``threshold``,
    else -1).
    """
    cfg = cfg or LampRecConfig()
    cs = cs or CSConfig()
    point = run_estimator(stream, cfg)
    seq = run_confseq(stream, cfg, cs, query_times=query_times)
    out = {**point, **seq}
    if threshold is not None:
        out["alarm"] = alarm_time(seq["t"], seq["hi"], threshold)
    return out
