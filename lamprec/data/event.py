"""Common streaming-log schema shared by every data source.

The synthetic simulator and the real-data adapters (OBP, KuaiRand) all emit a
:class:`Stream`: a columnar container of per-round *observables* plus a gold
ground-truth running value ``theta`` that a monitor is **not** allowed to see
(it exists only for evaluation -- MAE-vs-truth and coverage).

Streaming recommender log, round ``s = 0 .. n-1``:

============  ===============================================================
``context``   feature vector ``x_s`` (used by some baselines; the LAMP-Rec
              estimator needs only ``g/p/e/r/delay``).
``g``         label-free proxy ``g_s = Σ_a π_t(a|x_s) ĝ(x_s,a) ∈ [0,1]``,
              available immediately (no feedback).
``p``         logging propensity ``p_s = π_s(a_s|x_s) ∈ (0,1]``.
``e``         examination / position propensity ``e_s ∈ (0,1]``.
``rank``      0-based position of the shown item (cascade factorization).
``r_obs``     observed (position-biased) reward of the shown action. Its value
              exists at logging time but is **revealed** to the monitor only
              once the round matures.
``delay``     maturation delay ``d_s ≥ 0`` in rounds. Round ``s`` matures at
              ``s + delay[s]``. If ``s + delay[s] >= n`` the label never
              arrives within the observation window (structurally censored).
============  ===============================================================

``theta[s]`` is the running current-policy value ``θ_s = V(π_s)`` -- the
estimand. Gold only; never passed to a monitor.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from lamprec._release import withheld

# A delay this large (relative to horizon) marks a structurally-censored round
# whose label never matures inside the observation window.
NEVER: int = 1 << 30


@dataclass
class Stream:
    """Columnar streaming-recommender log with gold ground truth for eval."""

    g: np.ndarray          # (n,) DIRECT proxy g_s = Σ_a π_t(a|x_s) ĝ(x_s,a), immediate
    p: np.ndarray          # (n,) logging propensity p_s = π_s(a_s|x_s)
    e: np.ndarray          # (n,) examination propensity e_s
    r_obs: np.ndarray      # (n,) observed reward of shown action, revealed at maturation
    delay: np.ndarray      # (n,) int maturation delay in rounds
    theta: np.ndarray      # (n,) gold running value V(π_s) -- EVAL ONLY
    g_action: np.ndarray | None = None   # (n,) proxy at shown action ĝ(x_s,a_s); default = g
    pi_target: np.ndarray | None = None  # (n,) target-policy prob of shown action; default = p (on-policy)
    rank: np.ndarray | None = None       # (n,) int position; default all 0
    context: np.ndarray | None = None    # (n, d) features; optional
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.g = np.asarray(self.g, dtype=float)
        n = self.g.shape[0]
        self.p = np.asarray(self.p, dtype=float)
        self.e = np.asarray(self.e, dtype=float)
        self.r_obs = np.asarray(self.r_obs, dtype=float)
        self.delay = np.asarray(self.delay)
        self.theta = np.asarray(self.theta, dtype=float)
        if self.g_action is None:
            self.g_action = self.g.copy()       # top-1 approximation: ĝ(x,a_s) ≈ g_s
        self.g_action = np.asarray(self.g_action, dtype=float)
        if self.pi_target is None:
            self.pi_target = self.p.copy()      # on-policy monitoring: target = logging
        self.pi_target = np.asarray(self.pi_target, dtype=float)
        if self.rank is None:
            self.rank = np.zeros(n, dtype=int)
        self.rank = np.asarray(self.rank, dtype=int)
        for name in ("p", "e", "r_obs", "delay", "theta", "g_action", "pi_target", "rank"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(f"Stream.{name} has length {arr.shape[0]}, expected {n}")
        if self.context is not None:
            self.context = np.asarray(self.context, dtype=float)
            if self.context.shape[0] != n:
                raise ValueError("Stream.context first axis must equal n")
        # Maturation time s + d_s (clamped); censored rounds map to >= n.
        mt = np.arange(n) + np.minimum(self.delay, NEVER)
        self._mature_t = mt.astype(np.int64)
        self._groups: list[np.ndarray] | None = None  # built lazily

    # -- sizes -------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.g.shape[0]

    @property
    def maturation_time(self) -> np.ndarray:
        """``s + delay[s]`` per round (>= n means never matures in-window)."""
        return self._mature_t

    # -- maturation bookkeeping -------------------------------------------
    def matured_mask(self, t: int) -> np.ndarray:
        """Boolean mask of the matured set ``M_t = {s : s + d_s <= t}``."""
        return self._mature_t <= t

    @property
    def groups(self) -> list[np.ndarray]:
        """``groups[t]`` = rounds ``s`` whose label matures exactly at ``t``.

        Out-of-order arrival is implicit: a round logged early with a long
        delay lands in a later bucket than a round logged later with a short
        delay -- exactly the regime the certified path must survive.
        """
        if self._groups is None:
            n = self.n
            buckets: list[list[int]] = [[] for _ in range(n)]
            mt = self._mature_t
            for s in range(n):
                t = int(mt[s])
                if t < n:
                    buckets[t].append(s)
            self._groups = [np.asarray(b, dtype=int) for b in buckets]
        return self._groups

    def iter_events(self):
        """Yield ``(t, matured_s)`` for ``t = 0 .. n-1`` in arrival order.

        ``matured_s`` is the (possibly empty) array of rounds whose labels
        arrive at ``t``. The proxy observable for round ``t`` itself is always
        available at ``t`` (index it directly on the arrays).
        """
        groups = self.groups
        for t in range(self.n):
            yield t, groups[t]

    # -- transforms --------------------------------------------------------
    def replace(self, **changes) -> "Stream":
        """Return a copy with some columns/metadata replaced (immutable-style)."""
        return dataclasses.replace(self, **changes)

    def subwindow(self, start: int, stop: int) -> "Stream":
        """Slice rounds ``[start, stop)`` and re-base delays to the window.

        Labels that would mature past the window end become censored.
        """
        withheld("lamprec.data.event.Stream.subwindow")
