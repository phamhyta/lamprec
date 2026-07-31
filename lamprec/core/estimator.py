"""LAMP-Rec point estimate -- Algorithm 1, components C1 / C2 / C4.

The estimand is the running current-policy value ``θ_t = V(π_t)`` (unnormalized
DCG / CTR). The PPI / DR identity is the spine::

    θ = E[g] - E[r],     r_s = (ĝ(x_s,a_s) - r^obs_s) / (p_s e_s)

so ``θ̂_t = ĝ_t - r̂_t`` where

    ĝ_t = Σ_{s≤t}   λ^{t-s} g_s              / Σ_{s≤t}   λ^{t-s}          (C1, every round)
    r̂_t = Σ_{s∈M_t} λ^{t-s} ρ^mat_s r_s      / Σ_{s∈M_t} λ^{t-s} ρ^mat_s  (C2, matured only)

``ĝ_t`` is a plain exponential moving average. ``r̂_t`` is the same, except a
round only *enters* the sums when its label matures (at ``t = s + d_s``), where
it is injected with forgetting weight ``λ^{d_s}`` and decays by ``λ`` thereafter
-- which reproduces ``λ^{t-s}`` at every later ``t`` in O(1) per event. Out-of-
order maturation is automatic: rounds enter in maturation order, not arrival
order.

Overlap is enforced by clipping the IPS weight (Thm 4a)::

    ips_w_s = min(1/(p_s e_s), 1/τ_min).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lamprec._release import withheld
from lamprec.data.event import Stream


def recalibrate_proxy(stream: Stream, lam: float, tau_min: float, *,
                      importance: bool = True, examination: bool = True
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Online affine recalibration of the (frozen) proxy from matured labels.

    Fits a predictable ``(a_t, b_t)`` mapping ``ĝ → debiased reward`` by an
    exponentially-weighted regression over matured rounds, and returns the
    recalibrated ``(g'_direct, g'_action) = a_t + b_t·(g, g_action)``. Because
    the same ``(a_t,b_t)`` multiply both the proxy mean and the rectifier
    baseline, they cancel in expectation -- recalibration preserves
    unbiasedness and is pure variance reduction.
    """
    withheld("lamprec.core.estimator.recalibrate_proxy")


def shrunk_rectifier(stream: Stream, cfg: "LampRecConfig") -> np.ndarray:
    """Rectifier with variance-aware (James-Stein) examination-debiasing shrinkage.

    The ``1/e`` examination debias removes bias but adds variance; on low-signal
    streams that variance dominates and a biased low-variance estimate has lower
    MSE. The debiasing strength is shrunk toward ``1`` (no debias) by a
    predictable positive-part James-Stein weight computed from matured rounds
    only -- full debias when the examination bias is strong and well-estimated,
    no debias when the adjustment is swamped by noise.
    """
    withheld("lamprec.core.estimator.shrunk_rectifier")


def rectifier(stream: Stream, tau_min: float, *, importance: bool = True,
              examination: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-round position-debiased DR rectifier ``r_s`` and its IPS weight ``ω_s``.

    The unbiased control-variate form (target-on-logging or off-policy)::

        imp_s = π_target(a_s)/p_s              (policy importance weight, =1 on-policy)
        ω_s   = min(imp_s / e_s, 1/τ_min)      (clipped exposure-debiased IPS weight)
        r_s   = imp_s · ĝ(x_s,a_s)  −  ω_s · r^obs_s

    so ``θ̂ = ĝ − r̂`` is unbiased for ``E_{a~π_t}[μ(a)]``: the examination
    debias ``1/e`` acts on the (position-biased) reward only, never on the proxy
    baseline. Clipping ``ω_s`` at ``1/τ_min`` enforces overlap (Thm 4a).

    Ablation flags: ``importance=False`` drops the policy weight (uses
    ``imp=1``); ``examination=False`` drops the ``1/e`` debias (so the
    position bias in ``r^obs`` re-enters and the estimate becomes biased).
    """
    p = np.clip(stream.p, 1e-12, None)
    e = np.clip(stream.e, 1e-12, None)
    imp = (stream.pi_target / p) if importance else np.ones(stream.n)
    inv_e = (1.0 / e) if examination else np.ones(stream.n)
    omega = np.minimum(imp * inv_e, 1.0 / tau_min)
    r = imp * stream.g_action - omega * stream.r_obs
    return r, omega


@dataclass
class LampRecConfig:
    """Knobs for the LAMP-Rec monitor."""

    lam: float = 0.97               # forgetting factor λ ∈ (0,1)
    tau_min: float = 0.05           # overlap floor for IPS clipping (Thm 4a)
    mat_prob: np.ndarray | None = None  # per-round maturation propensity π^mat (IPCW); None→1
    importance: bool = True         # use policy importance weight (ablation off)
    examination: bool = True        # use 1/e position debias (ablation off)
    recalibrate: bool = False       # online affine proxy recalibration (variance ↓
                                    # under drift; unbiasedness preserved)
    shrink_debias: bool = False     # variance-aware James-Stein shrinkage of the
                                    # examination debias
    adaptive_lam: bool = False      # EXPERIMENTAL online plug-in λ̂ (a well-chosen
                                    # fixed λ remains the default)
    weight_floor: float = 1e-12     # numerical floor on weight denominators


@dataclass
class EstimatorState:
    """Streaming sufficient statistics for ``θ̂_t``; advanced one round at a time."""

    lam: float
    num_g: float = 0.0     # Σ λ^{t-s} g_s
    den_g: float = 0.0     # Σ λ^{t-s}
    num_r: float = 0.0     # Σ_{M_t} λ^{t-s} ρ^mat_s r_s
    den_r: float = 0.0     # Σ_{M_t} λ^{t-s} ρ^mat_s
    sumw_r: float = 0.0    # Σ_{M_t} λ^{t-s}            (for n_eff of the rectifier)
    sumw2_r: float = 0.0   # Σ_{M_t} (λ^{t-s})^2

    def g_hat(self) -> float:
        return self.num_g / max(self.den_g, 1e-12)

    def r_hat(self) -> float:
        return self.num_r / self.den_r if self.den_r > 0 else 0.0

    def theta_hat(self) -> float:
        return self.g_hat() - self.r_hat()

    def n_eff(self) -> float:
        """Effective # matured labels ``(Σw)² / Σw²`` (geometric ≈ (1+λ)/(1-λ))."""
        return (self.sumw_r ** 2) / self.sumw2_r if self.sumw2_r > 0 else 0.0


def run_estimator(stream: Stream, cfg: LampRecConfig) -> dict[str, np.ndarray]:
    """Run the streaming point estimate over the whole stream.

    Returns arrays over ``t = 0..n-1``: ``theta_hat, g_hat, r_hat, n_eff_r,
    lam``. O(1) amortized per round (each round injected into the rectifier
    once, at its maturation).
    """
    withheld("lamprec.core.estimator.run_estimator")
