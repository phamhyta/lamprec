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


def estimate_examination(stream: Stream, lam_e: float = 0.999,
                         min_count: float = 20.0,
                         decay_clip: tuple[float, float] = (0.4, 0.99)
                         ) -> np.ndarray:
    """Online estimate of the examination curve from the matured labels.

    Positions are randomized independently of relevance, so the position-wise
    means of matured labels point-identify the TRUE examination curve
    regardless of the assumed decay. Per-position EW means are updated in
    maturation order (predictable), pooled by a one-parameter cascade fit; the
    assumed ``stream.e`` is kept as the cold-start fallback until every used
    position has enough effective count. A plug-in nuisance, reported with the
    same estimated-nuisance caveat as the maturation model.
    """
    withheld("lamprec.core.estimator.estimate_examination")


def gated_examination(stream: Stream, alpha_spec: float = 0.05) -> np.ndarray:
    """Anytime-valid specification gate on the assumed examination model.

    A sequential test-by-betting of label-value independence that exploits the
    logged position randomization: under H0 (assumed ``e`` correct) the betting
    increment is conditionally mean-zero regardless of drift, and two one-sided
    wealths give a time-uniform level-``alpha_spec`` test by Ville. On
    rejection, rounds maturing after the stopping time switch to the estimated
    curve of :func:`estimate_examination`; if the test never rejects, the
    assumed model is used throughout (its correct-model optimality is kept).
    """
    withheld("lamprec.core.estimator.gated_examination")


def power_tune_proxy(stream: Stream, lam: float, tau_min: float, *,
                     importance: bool = True, examination: bool = True
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Slope-only, centered power-tuning of the proxy (PPI++ coefficient).

    Replaces the affine recalibration's ``(a_t, b_t)`` with
    ``g'_s = β_s (g_s − m_s)`` where ``m_s`` is a predictable EW proxy mean and
    ``β_s = clip(Cov/Var, [0,1])`` the variance-optimal control-variate
    coefficient from labels matured before ``s`` (no intercept). Unbiasedness
    is untouched for any ``(β, m)`` path; when the proxy carries no signal
    ``β→0`` kills BOTH terms identically, so the estimator reduces to the
    proxy-free matured EMA.
    """
    withheld("lamprec.core.estimator.power_tune_proxy")


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
    mat_mean: np.ndarray | None = None  # per-round MEAN of an exponential delay law
                                    # (oracle or estimated). When set, the IPCW
                                    # weight is evaluated at the ELAPSED LAG
                                    # π^mat_{t,s} = 1−exp(−(t−s)/mean_s) -- the
                                    # paper's definition. Takes precedence over
                                    # mat_prob.
    importance: bool = True         # use policy importance weight (ablation off)
    examination: bool = True        # use 1/e position debias (ablation off)
    recalibrate: bool = False       # online affine proxy recalibration. Superseded
                                    # by power_tune; kept for the ablation
                                    # comparison.
    power_tune: bool = False        # slope-only centered power-tuning (PPI++):
                                    # β_t (g−m_t) with predictable (β, m); reduces
                                    # to the proxy-free matured EMA when β→0. The
                                    # point-estimate default.
    shrink_debias: bool = False     # variance-aware James-Stein shrinkage of the
                                    # examination debias
    estimate_exam: bool = False     # estimate the examination curve online from
                                    # the matured labels' position-wise means
                                    # (randomized ranks identify it)
    exam_gate: bool = False         # anytime-valid spec test of the assumed
                                    # curve; switches to the estimated curve only
                                    # on rejection
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
