"""Anytime-valid confidence sequence -- Algorithm 1 component C3 (Theorem 2).

A betting (hedged-capital) confidence sequence for a bounded mean, built on the
*rectifier* in **maturation order** and combined by a union bound with a CS for
the proxy mean to give an anytime-valid CS for ``θ_t = E[g] - E[r]``.

Validity (Ville's inequality): for a fixed true mean ``μ`` and predictable bets,
the wealth ``K_j(μ) = ∏ (1 ± ω_j (u_j - μ))`` is a nonnegative martingale with
``K_0 = 1``, so ``P(∃ j : K_j(μ) ≥ 1/α) ≤ α``. The CS is ``{μ : K_j(μ) < 1/α}``.
Processing the rectifier in **maturation order** (not arrival order) and folding
the geometric forgetting weight into the predictable bet ``ω_j`` is exactly the
out-of-sequence construction Theorem 2 asks for; bets remain predictable so the
martingale property is untouched. Forgetting down-weights stale rounds (tracking)
without affecting coverage of a constant mean.

Bridging the *windowed* rectifier mean to the moving ``θ_t`` uses the Theorem-1
drift inflation ``V̄·λ/(1-λ)`` (Remark 2a); pass ``drift_budget`` to enable it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lamprec._release import withheld
from lamprec.core.estimator import LampRecConfig
from lamprec.data.event import Stream

_EPS = 1e-3          # keeps betting factors strictly positive
_VAR_FLOOR = 1e-4    # floor on the plug-in variance used by the aGRAPA bet


def _betting_interval(
    u: np.ndarray, w: np.ndarray, alpha: float, grid: np.ndarray,
    bet: str = "agrapa", c: float = 1.0 - _EPS, lam0: float = 0.5,
) -> tuple[float, float]:
    """Hedged-capital CS for the weighted mean of ``u ∈ [0,1]`` (in given order).

    ``w`` are predictable forgetting weights in ``(0,1]``; ``grid`` is the set of
    candidate means in ``[0,1]`` tested by Ville's rule. Returns ``(lo, hi)`` --
    the smallest/largest grid mean not yet excluded.
    """
    withheld("lamprec.core.confseq._betting_interval")


def betting_mean_interval(values: np.ndarray, weights: np.ndarray, alpha: float,
                          lo_bound: float, hi_bound: float, grid_size: int = 81,
                          bet: str = "agrapa", fixed_lam: float = 0.5
                          ) -> tuple[float, float]:
    """Anytime-valid CS for the weighted mean of bounded ``values ∈ [lo,hi]``.

    A thin public wrapper over the hedged-capital engine: rescale to ``[0,1]``,
    run the betting interval, map back. Used by the AV-DR baseline (betting CS on
    the position-debiased reward, with no proxy control variate).
    """
    withheld("lamprec.core.confseq.betting_mean_interval")


@dataclass
class CSConfig:
    alpha: float = 0.05
    grid_size: int = 81
    bet: str = "agrapa"          # 'agrapa' (adaptive) or 'fixed' (ablation)
    fixed_lam: float = 0.5       # bet fraction for the 'fixed' ablation
    drift_budget: float | None = None   # V̄ for the θ_t inflation (Remark 2a)
    rho_max: float = 1.0         # cap on maturation inflation ρ^mat (range bound)
    proxy_alpha_frac: float = 0.25  # union-bound split: proxy gets α·frac (it is
                                    # tight, so the rectifier keeps the rest)
    window_mult: float = 6.0     # CS window = window_mult/(1-λ) recent rounds
    method: str = "union"        # 'union' = proxy-CS ⊕ rectifier-CS (Thm 2): the
                                 # robust, distribution-free-valid DEFAULT.
                                 # 'joint' = control-variate PPI++ betting CS:
                                 # tighter at high proxy quality -- an opt-in.
    joint_beta_shrink: float = 1.0  # shrink the control-variate coefficient toward 0
    joint_inflate: float = 1.4      # finite-sample half-width inflation for the
                                    # joint (control-variate) interval


def _proxy_interval(stream: Stream, cfg: LampRecConfig, cs: CSConfig,
                    t: int) -> tuple[float, float]:
    """CS for ``E_w[g]`` from all rounds ``s ≤ t`` (no delay)."""
    withheld("lamprec.core.confseq._proxy_interval")


def _rectifier_interval(stream: Stream, cfg: LampRecConfig, cs: CSConfig,
                        t: int) -> tuple[float, float]:
    """CS for ``E_w[r]`` from rounds matured by ``t``, in maturation order."""
    withheld("lamprec.core.confseq._rectifier_interval")


def _theta_interval_joint(stream: Stream, cfg: LampRecConfig, cs: CSConfig,
                          t: int) -> tuple[float, float]:
    """Control-variate (PPI++) betting CS for ``θ_t`` -- ONE interval.

    Uses the proxy at the shown action as a control variate centred on a
    predictable all-rounds proxy baseline, realising the ``√(1-ρ²)`` variance
    reduction inside a single interval (no union split).
    """
    withheld("lamprec.core.confseq._theta_interval_joint")


def _theta_interval_union(stream: Stream, cfg: LampRecConfig, cs: CSConfig,
                          t: int) -> tuple[float, float]:
    """Union-bound CS (Thm 2): proxy-CS ⊕ rectifier-CS."""
    withheld("lamprec.core.confseq._theta_interval_union")


def theta_interval(stream: Stream, cfg: LampRecConfig, cs: CSConfig,
                   t: int) -> tuple[float, float]:
    """Anytime-valid CS for ``θ_t`` at query time ``t`` (dispatch on ``cs.method``)."""
    if cs.method == "union":
        return _theta_interval_union(stream, cfg, cs, t)
    return _theta_interval_joint(stream, cfg, cs, t)


def run_confseq(stream: Stream, cfg: LampRecConfig, cs: CSConfig,
                query_times: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Compute the θ-CS over time.

    ``query_times`` defaults to every round when ``n`` is small, else a strided
    subsample (the CS is recomputed per query in O(|M_t|·grid)). Returns the
    query times and per-query ``(lo, hi)``.
    """
    withheld("lamprec.core.confseq.run_confseq")


def alarm_time(query_times: np.ndarray, hi: np.ndarray, threshold: float) -> int:
    """First query time whose CS upper bound drops below ``threshold`` (else -1).

    Degradation alarm: the monitor is *certain* (anytime-valid) that θ_t has
    fallen below the quality floor when the whole interval is under it.
    """
    fired = np.nonzero(hi < threshold)[0]
    return int(query_times[fired[0]]) if fired.size else -1
