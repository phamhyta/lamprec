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

# Parsed + time-sorted CSVs, keyed by absolute path. Pure performance cache (the
# 1.2M-row log would otherwise be re-parsed per seed); no protocol change.
_DF_CACHE: dict = {}


def _load_df(path: str):
    import pandas as pd

    key = os.path.abspath(path)
    if key not in _DF_CACHE:
        df = pd.read_csv(path)
        if "time_ms" in df:
            df = df.sort_values("time_ms")
        _DF_CACHE[key] = df.reset_index(drop=True)
    return _DF_CACHE[key]


def load_kuairand(filename: str = "log_random.csv", calib_frac: float = 0.1,
                  delay_median: float = 6.0, delay_sigma: float = 0.8,
                  smooth: int = 200, seed: int = 0, max_rows: int = 40000,
                  reward: str = "click", path: str | None = None) -> Stream:
    """Load a KuaiRand random-exposure log into a Stream. Raises if absent.

    ``reward`` selects the observed reward:

    * ``"click"`` (default): binary ``is_click`` -- the original column.
    * ``"watch_ratio"``: graded engagement ``clip(play_time_ms/duration_ms, 0, 1)``
      -- the paper's graded-relevance (DCG) estimand on real data
      ("KuaiRand-Engage"), with a frozen per-video mean-watch-ratio proxy.
    """
    withheld("lamprec.data.kuairand_adapter.load_kuairand")


def available(filename: str = "log_random.csv") -> bool:
    return os.path.exists(os.path.join(CACHE, filename))
