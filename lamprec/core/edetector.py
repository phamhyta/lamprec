"""Changepoint e-detector for degradation alarms.

A CS-crossing alarm repurposes an *estimation* object for *detection* and pays
for it: pre-change samples leave the betting wealth in debt that post-change
samples must repay before the bound moves. A changepoint e-detector restarts
its bets at every candidate changepoint, so the post-change term carries no
debt (Shiryaev-Roberts / e-detector construction).

Statistic (fixed-prior mixture, computed recursively):

    R_t = e_t · (R_{t−1} + π_t),        R_0 = 0,   π_t = 1/N  (uniform prior),

with one-sided e-factors from predictable bets on a predictable running
envelope; Ville gives the genuine time-uniform false-alarm guarantee
``P(∃t: R_t ≥ 1/α) ≤ α`` -- no union bound over restart times, no debt.
"""
from __future__ import annotations

import numpy as np

from lamprec._release import withheld


def edetector_trajectory(values: np.ndarray, event_times: np.ndarray, n: int,
                         mu0: float, alpha: float = 0.05,
                         lam_stats: float = 0.97) -> np.ndarray:
    """Round-indexed trajectory ``R_t`` of the one-sided degradation e-detector.

    ``values[j]`` is revealed at round ``event_times[j]`` (pass maturation
    times for the label channel, arrival times for the proxy channel; events
    must be sorted by ``event_times``). The alarm certifies "the conditional
    mean of THIS channel has fallen below ``mu0``" -- time-uniformly at level
    ``alpha`` by Ville. The bet is a predictable aGRAPA-style plug-in, zero
    while the recent mean sits at/above the floor, positive once it dips.
    """
    withheld("lamprec.core.edetector.edetector_trajectory")


def alarm_from_trajectory(R: np.ndarray, alpha: float, start: int = 0) -> int:
    """First round ``t ≥ start`` with ``R_t ≥ 1/α`` (else -1)."""
    fired = np.nonzero(R[start:] >= 1.0 / alpha)[0]
    return int(fired[0] + start) if fired.size else -1
