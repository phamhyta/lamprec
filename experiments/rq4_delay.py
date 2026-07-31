#!/usr/bin/env python3
"""RQ4 -- Proxy gain and delay as a resource.

Panel (a): proxy↔reward correlation ρ vs the variance/width reduction; the
rectifier-std ratio tracks the √(1-ρ²) PPI envelope (the point-estimate gain),
while the anytime-valid CS-width ratio is the (conservative, union-bound) gain.
Panel (b): the CS-width reduction grows with mean maturation delay -- more
unlabeled rounds per matured label keep the proxy mean tight.

    uv run python experiments/rq4_delay.py
"""
from __future__ import annotations

from lamprec._release import withheld

RHOS = [0.40, 0.55, 0.70, 0.80, 0.86, 0.92]
DELAYS = [1, 4, 10, 25, 60]
SEEDS_N = 8


def _width_ratio(rho, delay, seed):
    withheld("experiments.rq4_delay._width_ratio")


def main():
    withheld("experiments.rq4_delay.main")


if __name__ == "__main__":
    main()
