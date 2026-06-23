"""Unit tests for the internal_tide comparator (first TOPOGRAPHY + tidal-forcing case).

Pins the ridge shape + the bump-bathymetry builder. The comparator EXPOSES a z-star
coordinate limitation (legoESM under-generates the topographic internal tide vs the
z-level oracle) — see docs/ocean_fidelity/oceananigans_reproduction_scoreboard.md.
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
def it():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.compare_oceananigans_internal_tide as mod  # type: ignore
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def test_ridge_shape(it):
    assert it._hill(0.0) == pytest.approx(it.H0)              # peak at x=0
    assert it._hill(it.WIDTH) == pytest.approx(it.H0 * np.exp(-0.5))
    assert it._hill(10 * it.WIDTH) < 1e-6 * it.H0             # decays far from the ridge


def test_tidal_params(it):
    # M2 period, excursion parameter → forcing amplitude (A₂) sane and finite.
    assert it.T2 == pytest.approx(12.421 * 3600.0)
    assert it.U2 > 0 and np.isfinite(it.A2)


def test_setup_has_bump_bathymetry(it):
    grid, wall, z, state, model = it.build_setup()
    Hb = np.asarray(state.H_bathy.data)
    # The ridge raises the bottom by ~H0 in x (NOT a flat bottom).
    assert (Hb[0].max() - Hb[0].min()) > 0.8 * it.H0
    assert Hb.max() == pytest.approx(it.H, rel=1e-3)         # full depth away from the ridge
    # 2D x-z setup: thin y, the real NX/NZ.
    assert Hb.shape == (it.NY, it.NX)
