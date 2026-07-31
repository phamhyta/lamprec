"""LAMP-Rec: label-free, anytime-valid performance monitoring for streaming
recommenders -- the experiment apparatus behind the paper.

The package is dataset-agnostic: every source emits a
:class:`lamprec.data.event.Stream`, and the monitor + baselines consume only
that schema.
"""
from lamprec.data.event import Stream

__all__ = ["Stream"]
__version__ = "0.2.0"
