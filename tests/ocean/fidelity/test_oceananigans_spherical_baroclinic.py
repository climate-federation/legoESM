"""Unit tests for the spherical baroclinic-adjustment comparator (the SPHERE analog).

Pins the front ramp + the lat-lon grid/IC builder so the spherical-metric test bed
(does the full-velocity vertadv fix hold on the lat-lon C-grid?) stays valid.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.fixture(scope="module")
def sb():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.compare_oceananigans_spherical_baroclinic as mod  # type: ignore
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def test_ramp_front(sb):
    assert sb._ramp(-sb.DY, sb.DY) == pytest.approx(0.0)
    assert sb._ramp(0.0, sb.DY) == pytest.approx(0.5)
    assert sb._ramp(sb.DY, sb.DY) == pytest.approx(1.0)
    assert sb.DB == pytest.approx(sb.DY * sb.M2)


def test_setup_is_latlon_with_vertadv_fix(sb):
    grid, wall, z, state, model = sb.build_setup()
    # Real lat-lon grid: cos_lat varies (NOT the Cartesian cos_lat≡1).
    cos_lat = np.asarray(grid.cos_lat)
    assert cos_lat.std() > 1e-4, "expected a spherical (varying cos_lat) grid"
    # The faithful §5 closure default is ON.
    assert model.config.weno_vertadv_full_velocity is True
    # The matched vertex-f Coriolis.
    assert model.config.coriolis_scheme == "matsuno_split"


def test_set_ic_stratified_from_rest(sb):
    grid, wall, z, state, model = sb.build_setup()
    state = sb.set_ic(grid, z, state)
    T = np.asarray(state.T.data)
    assert T.shape[2] == sb.NZ
    ci, cj = T.shape[0] // 2, T.shape[1] // 2
    assert float(T[ci, cj, 0]) > float(T[ci, cj, -1])     # k=0 surface warmer (stratified)
    assert float(np.abs(np.asarray(state.u.data)).max()) == pytest.approx(0.0)  # rest start
