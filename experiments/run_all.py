#!/usr/bin/env python3
"""Run every RQ end-to-end, writing results/<rq>/summary.json.

    uv run python experiments/run_all.py
"""
from __future__ import annotations

import importlib
import time

RUNNERS = [
    ("RQ1 tracking", "experiments.rq1_tracking"),
    ("RQ2 coverage", "experiments.rq2_coverage"),
    ("RQ3 alarm", "experiments.rq3_alarm"),
    ("RQ4 delay", "experiments.rq4_delay"),
    ("RQ5 ablation", "experiments.rq5_ablation"),
    ("RQ6 robustness", "experiments.rq6_robustness"),
    ("RQ7 misspecification", "experiments.rq7_misspecification"),
    ("RQ8 OPE-CS ladder", "experiments.rq8_opecs"),
]


def main():
    total = time.time()
    for label, mod in RUNNERS:
        t0 = time.time()
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        importlib.import_module(mod).main()
        print(f"[{label} done in {time.time()-t0:.1f}s]")
    print(f"\nAll RQs complete in {time.time()-total:.1f}s (CPU only).")


if __name__ == "__main__":
    main()
