"""Open Bandit Dataset (OBP) adapter -> the common Stream schema.

The OBP library is intentionally NOT used (it hard-pins ancient scipy); we read
the released CSVs directly. Each CSV is one (behavior_policy, campaign) log with
columns at least: ``timestamp, item_id, position, click, propensity_score`` plus
user features.

Download (small public release; ~tens of MB):
    https://research.zozo.com/data.html        (Open Bandit Dataset)
    # place CSVs under data/cache/obp/{behavior_policy}/{campaign}.csv

Mapping to :class:`~lamprec.data.event.Stream`:
  * ``action``   <- item_id ; ``rank`` <- position ; ``r_obs`` <- click
  * ``p``        <- propensity_score (logged) ; ``e`` <- cascade γ_{rank}
  * proxy ``ĝ``  <- frozen per-item empirical CTR fit on a calibration prefix
                   (the "deployed model" snapshot that goes stale)
  * ``delay``    <- injected maturation (OBP has no native label delay)
  * gold ``θ_t`` <- heavily-smoothed running debiased reward r_obs/e; on the
                   random-exposure ('random') slice this is the unbiased
                   uniform-policy value.

The headline experiments run on the synthetic backbone; this adapter is for
external validation when the CSV is present.
"""
from __future__ import annotations

import os

import numpy as np

from lamprec._release import withheld
from lamprec.data.event import Stream

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "cache", "obp")


def _cascade_e(position: np.ndarray, decay: float = 0.85) -> np.ndarray:
    return decay ** np.clip(position.astype(int), 0, None)


def load_obp(behavior_policy: str = "random", campaign: str = "all",
             calib_frac: float = 0.1, delay_mean: float = 8.0,
             smooth: int = 200, seed: int = 0, path: str | None = None) -> Stream:
    """Load an OBP CSV into a Stream. Raises FileNotFoundError if absent."""
    withheld("lamprec.data.obp_adapter.load_obp")


def available(behavior_policy: str = "random", campaign: str = "all") -> bool:
    return os.path.exists(os.path.join(CACHE, behavior_policy, f"{campaign}.csv"))


def available_redeploy(campaign: str = "all") -> bool:
    return available("random", campaign) and available("bts", campaign)


def load_obp_redeploy(n_each: int = 5000, campaign: str = "all", calib_frac: float = 0.1,
                      delay_mean: float = 8.0, smooth: int = 300, seed: int = 0) -> Stream:
    """Real-data policy redeploy: concatenate OBP ``random`` → ``bts`` logs.

    The deployed policy switches at the boundary (a genuine redeploy with REAL
    propensities and clicks); the proxy is a per-item CTR frozen on the
    pre-redeploy period, so it goes stale after the switch and the delayed
    labels must re-anchor -- the drifting-monitoring scenario on real logs.
    """
    withheld("lamprec.data.obp_adapter.load_obp_redeploy")
