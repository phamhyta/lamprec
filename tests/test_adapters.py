"""Adapter interface: availability probes work; parsers ship on release."""
from __future__ import annotations

import pytest

from lamprec._release import NOTICE, NotReleasedError
from lamprec.data import kuairand_adapter, obp_adapter


def test_available_probes_return_bool():
    assert isinstance(obp_adapter.available(), bool)
    assert isinstance(kuairand_adapter.available(), bool)


def test_loaders_are_withheld():
    with pytest.raises(NotReleasedError, match="not public yet"):
        obp_adapter.load_obp()
    with pytest.raises(NotReleasedError, match="not public yet"):
        kuairand_adapter.load_kuairand()


@pytest.mark.skip(reason=NOTICE)
def test_obp_adapter_parses(tmp_path):
    """A synthetic OBP-shaped CSV round-trips into a valid Stream."""


@pytest.mark.skip(reason=NOTICE)
def test_kuairand_adapter_parses(tmp_path):
    """A synthetic KuaiRand-shaped CSV round-trips into a valid Stream."""


@pytest.mark.skip(reason=NOTICE)
def test_kuairand_watch_ratio_parses(tmp_path):
    """reward='watch_ratio': graded r_obs in [0,1], mean-fallback proxy, rho meta."""


@pytest.mark.skip(reason=NOTICE)
def test_adapters_raise_when_absent(tmp_path):
    """Missing cache files raise a clear FileNotFoundError with download steps."""


@pytest.mark.skip(reason=NOTICE)
def test_adapter_stream_runs_through_monitor(tmp_path):
    """An adapter-emitted Stream runs through the full monitor end to end."""
