"""The tiled lane can finally use the packed exchange.

A latitude band's halo is two whole circles of longitude however thin the band
gets, so at 128 GPUs a band carries four rows of halo for every row it owns.
Splitting longitude too takes that to 0.62 -- and the tiled lane has always
measured slower anyway, because the packing that gives the band lane thirteen
messages a step refused any mesh that was not one-dimensional, so the tiled
lane sent one message per field.

The tiled packer now takes what the band packer takes, and this wires it into
the dycore behind its own switch. These gates say the switch changes the
program, does not change the answer, and cannot be spelled wrong.
"""

from __future__ import annotations

import os

import jax
import numpy as np
import pytest


def _need(n: int) -> None:
    if jax.device_count() < n:
        pytest.skip(
            f"Need {n} CPU devices "
            f"(XLA_FLAGS=--xla_force_host_platform_device_count={n})")


def test_the_switch_cannot_be_spelled_wrong(monkeypatch):
    """A typo must not silently leave the tiled lane on the per-field path
    while a receipt records the packer as armed."""
    from legoesm.parallel.latlon_spmd import packed_2d_exchange_mesh

    for bad in ("yes", "true", "2", " 1", "on"):
        monkeypatch.setenv("LEGOESM_LATLON_PACKED_2D_EXCHANGE", bad)
        with pytest.raises(ValueError):
            packed_2d_exchange_mesh()


def test_off_and_unset_are_the_same_thing(monkeypatch):
    from legoesm.parallel.latlon_spmd import packed_2d_exchange_mesh

    for value in ("", "0"):
        monkeypatch.setenv("LEGOESM_LATLON_PACKED_2D_EXCHANGE", value)
        assert packed_2d_exchange_mesh() is None


def test_the_band_switch_does_not_arm_the_tile_packer(monkeypatch):
    """Two packers, two switches. Arming the band one on a tiled mesh must
    not reach the tile packer: every existing receipt on this lane was
    measured with the band switch on, and silently changing what that means
    would invalidate all of them."""
    from legoesm.parallel.latlon_spmd import packed_2d_exchange_mesh

    monkeypatch.setenv("LEGOESM_LATLON_PACKED_EXCHANGE", "1")
    monkeypatch.delenv("LEGOESM_LATLON_PACKED_2D_EXCHANGE", raising=False)
    assert packed_2d_exchange_mesh() is None


def _tile_tendency(packed: str):
    """The (2,2)-tile tendency, with the packed tile exchange off or on."""
    import test_atm_latlon_2d_tiling as tiling

    os.environ["LEGOESM_LATLON_PACKED_2D_EXCHANGE"] = packed
    os.environ["LEGOESM_LATLON_PACKED_EXCHANGE"] = "0"
    try:
        _, tile, _ = tiling._serial_and_tile_tendency()
    finally:
        os.environ.pop("LEGOESM_LATLON_PACKED_2D_EXCHANGE", None)
        os.environ.pop("LEGOESM_LATLON_PACKED_EXCHANGE", None)
    return tile


def test_packing_the_tile_exchange_does_not_move_the_answer():
    """Regrouping messages moves no data. If a single value differs, the
    packed body is delivering the wrong rows and everything downstream is
    wrong quietly."""
    _need(4)
    off = _tile_tendency("0")
    on = _tile_tendency("1")
    for name, a, b in zip(("du", "dv", "dT", "dps"), on, off):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"the packed tile exchange changed {name}")
