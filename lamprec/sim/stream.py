"""Synthetic streaming-recommender generator with exact ground-truth ``θ_t``.

Given relevance/policy trajectories ``(mu, pi)`` (from :mod:`lamprec.sim.drift`)
and a delay array (from :mod:`lamprec.sim.delay`), :func:`build_stream` emits a
:class:`~lamprec.data.event.Stream`:

* shown action ``a_s ~ π_s``; logging propensity ``p_s = π_s(a_s)`` (on-policy);
* position ``rank_s`` with cascade examination ``e_s = decay^{rank}``;
* graded relevance with a tunable proxy correlation ``proxy_rho`` -- the direct
  proxy ``g_s = Σ_a π_s(a) ĝ_s(a)`` and the at-action proxy ``ĝ_s(a_s)``;
* gold ``θ_s = Σ_a π_s(a) μ_s(a)`` (eval only).

By construction the position-debiased DR estimate ``θ̂ = ĝ - r̂`` is unbiased
for the forgetting-weighted mean of ``θ_s``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lamprec._release import withheld
from lamprec.data.event import Stream
from lamprec.sim import delay as delay_mod
from lamprec.sim import drift as drift_mod


@dataclass
class SimConfig:
    n: int = 1000
    K: int = 20                      # number of items
    exam_positions: int = 10         # distinct positions an item can land in
    exam_decay: float = 0.80         # cascade γ_k = decay^k (e ∈ [~0.13, 1])
    proxy_rho: float = 0.75          # target corr(ĝ, μ); 1.0 → perfect proxy
    seed: int = 0


def _sample_actions(pi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cum = np.cumsum(pi, axis=1)
    u = rng.random(pi.shape[0])
    a = (cum < u[:, None]).sum(axis=1)
    return np.clip(a, 0, pi.shape[1] - 1)


def build_stream(mu: np.ndarray, pi: np.ndarray, delay: np.ndarray,
                 cfg: SimConfig, rng: np.random.Generator) -> Stream:
    """Assemble observables + gold ``θ_t`` from trajectories and delays.

    The proxy is a FROZEN deployed-model snapshot calibrated on the initial
    relevance -- under drift it goes stale (``E[g] ≠ θ_t``) and the delayed
    labels re-anchor the estimate; ``proxy_rho`` controls the proxy↔reward
    correlation and hence the rectifier variance.
    """
    withheld("lamprec.sim.stream.build_stream")


_REGIMES = {
    "stationary": drift_mod.stationary,
    "abrupt_redeploy": drift_mod.abrupt_redeploy,
    "popularity_shift": drift_mod.popularity_shift,
    "drifting": drift_mod.drifting,
    "quality_step": drift_mod.quality_step,
    "performative_loop": drift_mod.performative_loop,
}
_DELAYS = {
    "constant": delay_mod.constant_delay,
    "exponential": delay_mod.exponential_delay,
    "lognormal": delay_mod.lognormal_delay,
    "heavytail": delay_mod.heavytail_delay,
}


def simulate(regime: str = "stationary", delay_model: str = "exponential",
             cfg: SimConfig | None = None, regime_kwargs: dict | None = None,
             delay_kwargs: dict | None = None) -> Stream:
    """Build a Stream end-to-end: drift regime + delay model + observables."""
    withheld("lamprec.sim.stream.simulate")


def misspecify(stream: Stream, prop_noise: float = 0.0,
               exam_decay_wrong: float | None = None, seed: int = 0) -> Stream:
    """Hand the estimator MIS-SPECIFIED propensities/examination (stress test).

    Rewards and gold ``θ_t`` are unchanged (they came from the TRUE world);
    only the ``p``/``e`` the estimator *sees* are corrupted -- multiplicative
    log-normal noise on the propensities, or a cascade examination model with
    the WRONG decay.
    """
    withheld("lamprec.sim.stream.misspecify")


def simulate_informative_delay(cfg: SimConfig, base_mean: float = 8.0,
                               strength: float = 4.0, regime: str = "stationary",
                               regime_kwargs: dict | None = None
                               ) -> tuple[Stream, np.ndarray]:
    """Stream whose delay correlates with the reward (Assumption-D violation).

    Returns ``(stream, mat_prob)``; pass ``mat_prob`` to the monitor's config to
    apply the IPCW correction that restores coverage.
    """
    withheld("lamprec.sim.stream.simulate_informative_delay")
