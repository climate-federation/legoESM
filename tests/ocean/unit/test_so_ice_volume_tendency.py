"""The band integral must use true cell area, not a cos(lat) proxy.

This is the check the original inline probe lacked: on a mesh whose cell area
is deliberately NOT proportional to cos(lat), the two weightings must give
different answers, and ``band_ice_volume`` must track the true-area one.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SRC = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
        / "ocean_fidelity" / "so_ice_volume_tendency.py")
_spec = importlib.util.spec_from_file_location("so_ice_volume_tendency", _SRC)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _snapshot(tmp_path, lat, conc, thick, name="snapshot_day0001.npz"):
    p = tmp_path / name
    np.savez(p, lat_T=lat, land_mask=np.ones_like(lat),
             ice_concentration=conc, ice_thickness=thick)
    return p


def test_true_area_weighting_differs_from_coslat_and_is_used(tmp_path):
    # Two cells in the band. Cell 0 has 3x the area of cell 1, but the SAME
    # latitude, so a cos(lat) weight would call them equal.
    lat = np.array([[-60.0, -60.0]])
    area = np.array([[3.0, 1.0]])
    conc = np.array([[1.0, 1.0]])
    thick = np.array([[2.0, 0.0]])  # ice only in the BIG cell

    got = mod.band_ice_volume(_snapshot(tmp_path, lat, conc, thick), area, -45.0)

    # True area: (3*2 + 1*0)/4 = 1.5. A cos(lat) weight would give 1.0.
    assert got == pytest.approx(1.5)
    assert got != pytest.approx(1.0)


def test_mesh_shape_mismatch_is_fatal_not_broadcast(tmp_path):
    lat = np.array([[-60.0, -60.0]])
    snap = _snapshot(tmp_path, lat, np.ones_like(lat), np.ones_like(lat))
    with pytest.raises(SystemExit, match="does not match snapshot grid"):
        mod.band_ice_volume(snap, np.ones((4, 4)), -45.0)


def test_empty_band_raises_rather_than_returning_nan(tmp_path):
    lat = np.array([[10.0, 20.0]])
    snap = _snapshot(tmp_path, lat, np.ones_like(lat), np.ones_like(lat))
    with pytest.raises(SystemExit, match="no wet cells"):
        mod.band_ice_volume(snap, np.ones_like(lat), -45.0)
