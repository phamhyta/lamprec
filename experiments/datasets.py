"""Stream sources for the experiment runners.

The synthetic simulator is the backbone (it alone gives ground-truth θ_t). For
the main MAE table we expose several synthetic profiles whose delay/feedback
characteristics mimic the named logged streams; each is flagged
``synthetic=True`` so the honest-reconciliation step can mark which columns are
real. Real OBP / KuaiRand adapters override the matching source when their data
is present under ``data/cache/``.
"""
from __future__ import annotations

from lamprec._release import withheld

# Operating points approximating each stream's profile (proxy quality + delay
# law). Synthetic stand-ins until the real data is loaded.
MAIN_PROFILES = {
    "Synthetic": dict(proxy_rho=0.80, delay_model="exponential",
                      delay_kwargs={"mean": 8.0}),
    "OBP":       dict(proxy_rho=0.78, delay_model="exponential",
                      delay_kwargs={"mean": 6.0}),
    "KuaiRand":  dict(proxy_rho=0.82, delay_model="lognormal",
                      delay_kwargs={"median": 6.0, "sigma": 0.8}),
    # Graded watch-ratio engagement on the same KuaiRand random log (the paper's
    # DCG estimand on real data); synthetic stand-in until the log is cached.
    "KuaiRand-Engage": dict(proxy_rho=0.65, delay_model="lognormal",
                            delay_kwargs={"median": 6.0, "sigma": 0.8}),
}

# A-priori row budget for KuaiRand-Engage, fixed BEFORE any accuracy number was
# seen (largest candidate whose full 10-seed RQ1 column fits the time budget).
ENGAGE_MAX_ROWS = 200000


def _real_stream(name: str, seed: int):
    """Load the logged OBP/KuaiRand stream if its data is cached, else None."""
    try:
        if name == "OBP":
            from lamprec.data import obp_adapter
            if obp_adapter.available():
                return obp_adapter.load_obp(seed=seed)
        if name == "KuaiRand":
            from lamprec.data import kuairand_adapter
            if kuairand_adapter.available():
                return kuairand_adapter.load_kuairand(seed=seed)
        if name == "KuaiRand-Engage":
            from lamprec.data import kuairand_adapter
            if kuairand_adapter.available():
                return kuairand_adapter.load_kuairand(
                    seed=seed, reward="watch_ratio", max_rows=ENGAGE_MAX_ROWS)
    except Exception as exc:                              # noqa: BLE001
        print(f"  [warn] {name} adapter failed ({exc}); using synthetic profile")
    return None


def main_stream(name: str, seed: int, n: int = 2000):
    """A drifting stream for the main running-accuracy (MAE) table.

    Uses the real logged stream when its data is cached under ``data/cache/``;
    otherwise a synthetic profile matched to that stream. Quality drifts
    (popularity shift with a frozen policy → silent staleness), so a *tracking*
    monitor is needed: static baselines drift off while LAMP-Rec re-anchors.
    """
    withheld("experiments.datasets.main_stream")


def tracking_stream(seed: int, n: int = 1000, redeploy: int = 600):
    """Abrupt-redeploy stream for the tracking figure (θ rises, jumps, decays)."""
    withheld("experiments.datasets.tracking_stream")
