#!/usr/bin/env python3
"""RQ1 -- Running-accuracy tracking at equal label budget.

Produces (a) the main MAE table (methods × streams, mean±95% CI over seeds) and
(b) tracking curves across an abrupt redeploy for the tracking figure.

    uv run python experiments/rq1_tracking.py
"""
from __future__ import annotations

from lamprec._release import withheld


def mae_table():
    """MAE (×10⁻³) per (method, stream), mean±95% CI over seeds.

    Columns are marked synthetic with a trailing ``*`` when the named real log
    (OBP/KuaiRand) is not present under data/cache/ -- so the table never
    implies real data it does not have. The mark drops automatically once the
    adapter loads the logged stream.
    """
    withheld("experiments.rq1_tracking.mae_table")


def tracking_curves(seed: int = 0):
    """One-seed curves of θ_t and a few methods across the redeploy (for the fig)."""
    withheld("experiments.rq1_tracking.tracking_curves")


def main():
    withheld("experiments.rq1_tracking.main")


if __name__ == "__main__":
    main()
