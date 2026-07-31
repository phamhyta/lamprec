#!/usr/bin/env python3
"""RQ5 -- Ablation of the forced components, optimal forgetting, drift scaling.

(a) ablation table: MAE / coverage / width on a drifting stream for the full
    monitor and each ablation (−IPS/examination, −proxy, λ=1, proxy-only,
    fixed bet, untuned τ_min).
(b) λ U-curve: MAE vs the forgetting factor, with the closed-form λ* (Cor. 1a).
(c) drift scaling: the MSE-minimizing error vs drift V tracks V^{2/3}.

    uv run python experiments/rq5_ablation.py
"""
from __future__ import annotations

from lamprec._release import withheld

SEEDS_N = 8


def _abl_stream(seed, n=2000):
    withheld("experiments.rq5_ablation._abl_stream")


def _eval(stream, cfg, cs, qt, point_key="theta_hat"):
    withheld("experiments.rq5_ablation._eval")


def ablation_table():
    withheld("experiments.rq5_ablation.ablation_table")


def lambda_sweep():
    withheld("experiments.rq5_ablation.lambda_sweep")


def drift_scaling():
    withheld("experiments.rq5_ablation.drift_scaling")


def main():
    withheld("experiments.rq5_ablation.main")


if __name__ == "__main__":
    main()
