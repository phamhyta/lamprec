#!/usr/bin/env python3
"""RQ3 -- Degradation alarms: false-alarm rate vs detection delay.

An abrupt ~15% θ drop is injected mid-stream. A monitor alarms when it is
*certain* quality has fallen below a floor ``θ_min`` (LAMP-Rec/AV-DR: CS upper
bound < θ_min; naive: point estimate < θ_min). We measure detection delay on
degradation streams and the false-alarm rate on null (no-drop) streams. The
anytime-valid CS controls FAR at α while detecting near the maturation horizon;
naive must trade over-firing against detecting too late.

    uv run python experiments/rq3_alarm.py
"""
from __future__ import annotations

from lamprec._release import withheld

MARGIN = 0.08
CAL = slice(100, 400)


def _stream(seed, drop):
    withheld("experiments.rq3_alarm._stream")


def _floor(values_over_qt, qt):
    withheld("experiments.rq3_alarm._floor")


def main():
    withheld("experiments.rq3_alarm.main")


if __name__ == "__main__":
    main()
