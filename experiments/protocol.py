"""Experiment harness: seed loop, aggregation (mean±95% CI), results I/O.

Every RQ runner writes ``results/<rq>/summary.json`` (committed) and optionally
``results/<rq>/seed_*.json`` (git-ignored raw). The equal-labels-spent protocol
holds by construction: all methods consume the same matured labels of the same
stream.
"""
from __future__ import annotations

import json
import os

import numpy as np

from lamprec._release import withheld

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def write_json(rq: str, name: str, obj) -> str:
    d = os.path.join(RESULTS, rq)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    with open(path, "w") as f:
        json.dump(_jsonable(obj), f, indent=2)
    return path


def read_json(rq: str, name: str):
    with open(os.path.join(RESULTS, rq, name)) as f:
        return json.load(f)


def summarize(per_seed: list[dict], keys: list[str]) -> dict:
    """Mean ± 95% CI half-width across seeds for each metric key.

    Returns ``{key: {"mean": m, "ci": h, "n": k}}``; nested per-method dicts
    (``{method: {metric: value}}``) are summarized per (method, metric).
    """
    withheld("experiments.protocol.summarize")


def run_seeds(fn, seeds) -> list[dict]:
    """Run ``fn(seed) -> dict`` over seeds, returning the per-seed results."""
    return [fn(int(s)) for s in seeds]
