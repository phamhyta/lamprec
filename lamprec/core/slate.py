"""Examination-factorized rectifier -- Proposition 4b.

Under a position-based / cascade examination model ``e_s = γ_{rank(a_s)}`` the
slate observation propensity factorizes ``p_s e_s = π_t(a_s|x_s)·γ_{rank(a_s)}``.
Writing the rectifier **per (item, position)** rather than per slate replaces the
exponentially-small *slate* propensity in the overlap floor with the much larger
*position* propensity::

    Var_rectifier = O(1 / τ^pos_min)   instead of   O(1 / τ^slate_min),
    τ^pos_min ≫ τ^slate_min.

This module provides the cascade examination weights and the two propensity
conventions the "slate vs position-factorized" ablation contrasts.
"""
from __future__ import annotations

import numpy as np

from lamprec._release import withheld


def gamma_cascade(n_positions: int, decay: float = 0.85) -> np.ndarray:
    """Cascade examination weights ``γ_k = decay^k`` over positions ``0..K-1``."""
    return decay ** np.arange(n_positions, dtype=float)


def position_propensity(pi_action: np.ndarray, gamma_rank: np.ndarray) -> np.ndarray:
    """Per-(item,position) overlap ``π(a_s|x_s)·γ_{rank}`` (Prop. 4b numerator)."""
    return np.clip(pi_action, 0.0, 1.0) * np.clip(gamma_rank, 0.0, 1.0)


def slate_propensity(pi_action: np.ndarray, gamma_rank: np.ndarray,
                     slate_size: int) -> np.ndarray:
    """Slate-level overlap: the per-position propensity shrunk by the slate."""
    withheld("lamprec.core.slate.slate_propensity")


def overlap_floor(pe: np.ndarray, quantile: float = 0.05) -> float:
    """A data-driven overlap floor ``τ_min`` = low quantile of ``p·e`` (Thm 4a)."""
    withheld("lamprec.core.slate.overlap_floor")
