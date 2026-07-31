#!/usr/bin/env python3
"""RQ3 -- Degradation alarms: detection delay at MATCHED false-alarm rate.

An abrupt ~15% θ drop is injected mid-stream. Two fixes over a naive
single-operating-point comparison:

(1) PROTOCOL. Detection delay is only comparable at equal false-alarm rate.
    Each method's alarm margin is selected on CALIBRATION null streams (seeds
    disjoint from everything reported) as the smallest margin whose
    Clopper-Pearson 95% upper bound on the FAR is ≤ α; evaluation streams then
    report the realized FAR and the median detection delay at that frozen
    margin, plus the full delay-vs-FAR curve. Per-stream floors come from the
    stream's own causal pre-drop window -- never from the scored data.

(2) DETECTOR. The CS-crossing alarm carries pre-change wealth debt; the
    changepoint e-detector (lamprec/core/edetector.py) restarts its bets at
    every candidate changepoint (fixed-prior mixture, Ville-valid FAR ≤ α) and
    runs on two channels: the LABEL channel (control-variate pseudo-outcome at
    maturation times -- certifies a θ degradation) and the zero-delay PROXY
    channel (arrival times -- early warning). A 0.5/0.5 convex mixture of the
    two supermartingales is again a supermartingale: one Ville application, no
    alpha split. The same e-detector on AV-DR's proxy-free debiased stream
    isolates what the proxy contributes.

    uv run python experiments/rq3_alarm.py
"""
from __future__ import annotations

from lamprec._release import withheld

CAL_SEEDS = list(range(100, 200))     # margin calibration (nulls only, unreported)
EVAL_SEEDS = list(range(0, 50))       # reported FAR (nulls) + DD (drops)
MARGINS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
CAL_WIN = (100, 400)                  # causal pre-drop floor window
FAR_FROM = 400                        # null alarms counted from here

E_ROWS = ("LAMP-Rec e-det (label+proxy)", "LAMP-Rec e-det (label)",
          "AV-DR e-det")
CS_ROWS = ("LAMP-Rec CS", "AV-DR CS")
PT_ROWS = ("Naive (sensitive)", "Naive (smoothed)")
ROWS = E_ROWS + CS_ROWS + PT_ROWS


def _stream(seed, drop):
    withheld("experiments.rq3_alarm._stream")


def _channels(s):
    withheld("experiments.rq3_alarm._channels")


def _win_mean(values, times):
    withheld("experiments.rq3_alarm._win_mean")


def _alarm_time(ch, row, margin, from_t):
    withheld("experiments.rq3_alarm._alarm_time")


def _cp_upper_np(k, n, delta=0.05):
    withheld("experiments.rq3_alarm._cp_upper_np")


def main():
    withheld("experiments.rq3_alarm.main")


if __name__ == "__main__":
    main()
