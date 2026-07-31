#!/usr/bin/env python3
"""RQ2 -- Time-uniform coverage and interval width.

Empirical coverage of each method's interval against the nominal 1-α, swept over
a grid of nominal levels, on stationary streams (θ constant → the windowed
estimand equals θ, so a valid CS must cover it). LAMP-Rec and AV-DR are
anytime-valid (should track the diagonal); IUPM's heuristic-SD band ignores the
nominal target and under-covers.

    uv run python experiments/rq2_coverage.py
"""
from __future__ import annotations

from lamprec._release import withheld

NOMINAL = [0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]


def _stream(seed):
    withheld("experiments.rq2_coverage._stream")


def main():
    withheld("experiments.rq2_coverage.main")


if __name__ == "__main__":
    main()
