#!/usr/bin/env python3
"""results/ -> paper anchors.json + regenerated tables + claimed-vs-measured diff.

Assembles every RQ summary into a single anchors file that the figure
generators read, regenerates the main and ablation tables from the measured
numbers, and prints a claimed-vs-measured diff so the paper prose can be
reconciled to reality (honest reporting).

    uv run python experiments/report/build_anchors.py
"""
from __future__ import annotations

import json
import os

from lamprec._release import withheld

CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(CODE, "results")


def _load(rq, name="summary.json"):
    p = os.path.join(RESULTS, rq, name)
    if not os.path.exists(p):
        print(f"  [warn] missing {p} -- skipping")
        return None
    with open(p) as f:
        return json.load(f)


def build_anchors() -> dict:
    """Assemble all RQ summaries into one anchors dict."""
    withheld("experiments.report.build_anchors.build_anchors")


def emit_tab_main(anchors) -> str:
    """Regenerate the main MAE table from measured numbers."""
    withheld("experiments.report.build_anchors.emit_tab_main")


def emit_tab_ablation(anchors) -> str:
    """Regenerate the ablation table from measured numbers."""
    withheld("experiments.report.build_anchors.emit_tab_ablation")


def diff_report(anchors):
    """Print a claimed-vs-measured diff for prose reconciliation."""
    withheld("experiments.report.build_anchors.diff_report")


def main():
    withheld("experiments.report.build_anchors.main")


if __name__ == "__main__":
    main()
