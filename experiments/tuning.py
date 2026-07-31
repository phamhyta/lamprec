#!/usr/bin/env python3
"""Pre-registered hyperparameter selection on a dedicated tuning-seed block.

Every default the runners consume is selected HERE, on ``TUNING_SEEDS``
(100-109), which are excluded from every reported table (eval ``SEEDS`` =
{0..9}). The selection rules are stated a priori:

(1) ``lam`` (forgetting factor).
    - Synthetic streams: grid argmin of tracking MAE on the tuning seeds,
      cross-checked against the closed-form λ* of Cor. 1a.
    - Real logs: bandwidth match to the estimand's own smoothing horizon --
      a rule tied to the estimand definition, involving no data at all.

(2) ``tau_min`` (overlap floor).
    - Default config: τ_min stays the overlap floor (Thm 4a); the
      examination-misspecification defense is the estimated examination curve,
      not the clip.
    - Safe fallback (no rank signal available): minimax over the pre-registered
      misspecification stress grid on tuning seeds.

    uv run python experiments/tuning.py
"""
from __future__ import annotations

from lamprec._release import withheld

LAM_GRID = [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
TAU_GRID = [0.05, 0.10, 0.15, 0.20, 0.30]
DECAY_STRESS = [0.80, 0.78, 0.75, 0.70, 0.60]   # true decay is 0.80
FRAC_GRID = [0.02, 0.05, 0.10, 0.15, 0.25, 0.40]  # union alpha-split candidates
REAL_GOLD_SMOOTH = 200                            # the Setup's gold-θ horizon


def lam_real_rule(smooth: int = REAL_GOLD_SMOOTH) -> float:
    """Bandwidth-matched λ for real logs: ``n_eff(λ) = (1+λ)/(1-λ) = smooth``.

    A data-free rule tied to the estimand definition (the smoothed gold value).
    """
    return (smooth - 1.0) / (smooth + 1.0)


def select_lambda() -> dict:
    """Grid selection of the forgetting factor on the tuning seeds."""
    withheld("experiments.tuning.select_lambda")


def select_tau_safe() -> dict:
    """Minimax overlap floor over the misspecification stress grid."""
    withheld("experiments.tuning.select_tau_safe")


def select_alpha_split() -> dict:
    """Union-bound alpha-split selection (width, not coverage, is at stake)."""
    withheld("experiments.tuning.select_alpha_split")


def main():
    withheld("experiments.tuning.main")


if __name__ == "__main__":
    main()
