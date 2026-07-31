#!/usr/bin/env python3
"""RQ6 -- Performative drift, informative delay, and cost.

Empirical coverage across five stress regimes: a baseline, a contractive
performative loop (κ<1), an expansive loop (κ≥1), and informative delay (delay
correlated with reward, Assumption-D violation) without and with the IPCW
maturation-propensity correction. Coverage is unconditional in the drift: it
should hold in all but the uncorrected informative-delay case, which the
ρ-correction restores.

    uv run python experiments/rq6_robustness.py
"""
from __future__ import annotations

from lamprec._release import withheld


def _drift_budget(stream):
    withheld("experiments.rq6_robustness._drift_budget")


def _coverage_run(make_stream, mat_mean_from=None):
    withheld("experiments.rq6_robustness._coverage_run")


def _perf(kappa):
    withheld("experiments.rq6_robustness._perf")


def _baseline(seed):
    withheld("experiments.rq6_robustness._baseline")


def _info(seed):
    withheld("experiments.rq6_robustness._info")


def _info_bias():
    withheld("experiments.rq6_robustness._info_bias")


def main():
    withheld("experiments.rq6_robustness.main")


if __name__ == "__main__":
    main()
