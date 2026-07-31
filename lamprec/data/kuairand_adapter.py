"""KuaiRand adapter -> the common Stream schema.

KuaiRand ships logged short-video interactions; its **random-exposure** log is the
gold slice (items shown uniformly, so the click mean is an unbiased policy value).

Download (KuaiRand-Pure / -1K, a few hundred MB):
    https://kuairand.com/    (or the KuaiRand GitHub release)
    # place the random-exposure log under data/cache/kuairand/log_random.csv
    # expected columns: user_id, video_id, time_ms, is_click[, propensity]

Mapping mirrors the OBP adapter (see :mod:`lamprec.data.obp_adapter`): video_id
-> action, is_click -> reward, uniform propensity on the random slice, a frozen
per-video CTR proxy, injected maturation delay, and a smoothed running debiased
reward as the gold θ_t. For external validation only; the synthetic backbone
runs without it.
"""
from __future__ import annotations

import os

from lamprec._release import withheld
from lamprec.data.event import Stream

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "cache", "kuairand")


def load_kuairand(filename: str = "log_random.csv", calib_frac: float = 0.1,
                  delay_median: float = 6.0, delay_sigma: float = 0.8,
                  smooth: int = 200, seed: int = 0, max_rows: int = 40000,
                  path: str | None = None) -> Stream:
    """Load a KuaiRand random-exposure log into a Stream. Raises if absent."""
    withheld("lamprec.data.kuairand_adapter.load_kuairand")


def available(filename: str = "log_random.csv") -> bool:
    return os.path.exists(os.path.join(CACHE, filename))
