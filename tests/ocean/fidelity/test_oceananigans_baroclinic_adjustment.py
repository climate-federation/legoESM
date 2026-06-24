"""Unit tests for the baroclinic-adjustment comparison driver (the §5 precursor).

Pins the pure helpers (the linear front ramp + the grid/IC builder shapes) so the
fast CPU test bed for the §5-class baroclinic-eddy blow-up stays valid.
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
def ba():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.compare_oceananigans_baroclinic_adjustment as mod  # type: ignore
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def test_ramp_front(ba):
    # ramp(y, dy): 0 for y < -dy/2, y/dy+1/2 in between, 1 for y > dy/2.
    assert ba._ramp(-ba.DY, ba.DY) == pytest.approx(0.0)
    assert ba._ramp(0.0, ba.DY) == pytest.approx(0.5)
    assert ba._ramp(ba.DY, ba.DY) == pytest.approx(1.0)
    assert ba.DB == pytest.approx(ba.DY * ba.M2)


def test_setup_builds_3d_stratified(ba):
    grid, wall, z, state, model = ba.build_setup()
    assert z.n_levels == ba.NZ
    state = ba.set_ic(grid, z, state)
    T = np.asarray(state.T.data)
    assert T.shape[2] == ba.NZ
    # Stratified: at a WET interior column, warmer at the surface (k=0) than the
    # bottom (k=0) since b = N^2 z increases toward z=0 and T = T_ref + b/(g a).
    ci, cj = T.shape[0] // 2, T.shape[1] // 2
    assert float(T[ci, cj, 0]) > float(T[ci, cj, -1])  # k=0 surface warmer
    # Rest start (matches Oceananigans `set!(model, b=bᵢ)`): u=v=0, the jet spins
    # up by geostrophic adjustment in the run, NOT imposed at t=0.
    assert float(np.abs(np.asarray(state.u.data)).max()) == pytest.approx(0.0)
