#!/usr/bin/env python3
"""RQ7 -- Robustness to misspecification (breaks the "circular simulator" worry).

The simulator otherwise satisfies the method's assumptions exactly. Here we feed
the estimator MIS-SPECIFIED nuisances and measure graceful (or not) degradation:

(a) Estimated propensities: log-normal noise on the logging/examination
    propensities. LAMP-Rec degrades gracefully and stays at the proxy-free
    AV-DR level.
(b) Wrong examination model: the estimator assumes a cascade with the WRONG
    decay. The 1/e debias then over-corrects; the defense rows show the
    estimated examination curve (``estimate_exam``) and the anytime-valid gate
    (``exam_gate``) neutralizing the misspecification, with the overlap floor
    τ_min (Thm 4a) as the rank-free fallback.

    uv run python experiments/rq7_misspecification.py
"""
from __future__ import annotations

from lamprec._release import withheld

SEEDS_N = 8


def _base(seed):
    withheld("experiments.rq7_misspecification._base")


def _sr(s, tau=0.05, **flags):
    withheld("experiments.rq7_misspecification._sr")


def propensity_noise():
    withheld("experiments.rq7_misspecification.propensity_noise")


def exam_misspec():
    withheld("experiments.rq7_misspecification.exam_misspec")


def main():
    withheld("experiments.rq7_misspecification.main")


if __name__ == "__main__":
    main()
