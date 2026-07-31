"""Maturation-law estimation from the stream itself (no oracle propensity).

Replaces the simulator-supplied delay law with a predictable estimate: rounds
are bucketed by the action-proxy quantile (the proxy correlates with the latent
relevance that drives the delay), and each bucket's law is estimated by the
standard censored-data construction under administrative censoring. Estimates
refresh on a coarse time grid; a round receives the estimate frozen at the last
grid point at or before its maturation time, so the weight applied at injection
uses strictly-past data (predictable). The estimate identifies the
bucket-MARGINAL delay law: residual delay-reward coupling *within* a bucket is
not corrected -- reported as a point-estimate correction, not under Theorem 2.
"""
from __future__ import annotations

import numpy as np

from lamprec._release import withheld
from lamprec.data.event import Stream


def estimate_delay_means(stream: Stream, lag: float, n_buckets: int = 4,
                         warmup: int = 100, grid_step: int = 50) -> np.ndarray:
    """Per-round predictable pseudo-mean encoding the ESTIMATED ``F̂_b(lag)``.

    Nonparametric per-bucket binomial estimate of the fraction matured within
    ``lag``, returned as the pseudo-mean that makes the estimator's exponential
    formula reproduce ``F̂`` at that lag. Bucket quantiles are frozen on the
    warmup prefix; rounds maturing before the first refresh get no correction.

    MEASURED LIMITATION (reported, not hidden): when the bucketing feature is a
    noisy *relevance* proxy, the estimated correction can amplify the very
    selection it is meant to correct -- the delay model must be estimated from
    features that actually determine the delay.
    """
    withheld("lamprec.core.maturation.estimate_delay_means")
