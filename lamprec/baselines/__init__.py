"""Comparison methods, all consuming the common Stream at equal labels-spent."""
from lamprec.baselines.methods import (ALL_BASELINES, adwin_track, av_dr, eta,
                                       iupm, naive_feedback, opecs, pape,
                                       prequential_debiased,
                                       prequential_delayed)

__all__ = ["ALL_BASELINES", "naive_feedback", "prequential_delayed",
           "prequential_debiased", "adwin_track", "eta", "pape", "iupm",
           "opecs", "av_dr"]
