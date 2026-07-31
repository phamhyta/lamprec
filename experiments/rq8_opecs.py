#!/usr/bin/env python3
"""RQ8 -- Head-to-head against the nearest CS competitor (off-policy CS).

The certified path is a betting confidence sequence indexed by *maturation*
order, valid under out-of-order delayed labels. The nearest prior is an
arrival-ordered off-policy CS with no maturation model. This RQ benchmarks the
interval directly, as a component ladder:

    OPE-CS   (arrival order, no IPCW, no proxy)   -- the off-the-shelf competitor
    AV-DR    (+ maturation ordering)              -- isolates the ordering
    LAMP-Rec (+ IPCW rho + label-free proxy)      -- the full monitor

    uv run python experiments/rq8_opecs.py
"""
from __future__ import annotations

from lamprec._release import withheld

LAM, TAU = 0.99, 0.05
STRENGTHS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]   # informative-delay severity sweep


def _make(seed, strength):
    withheld("experiments.rq8_opecs._make")


def _cs(method, stream, mat_prob):
    withheld("experiments.rq8_opecs._cs")


def _sweep():
    withheld("experiments.rq8_opecs._sweep")


def _bias():
    withheld("experiments.rq8_opecs._bias")


def main():
    withheld("experiments.rq8_opecs.main")


if __name__ == "__main__":
    main()
