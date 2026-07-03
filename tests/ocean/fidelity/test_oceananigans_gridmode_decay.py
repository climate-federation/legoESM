"""Unit tests for the 2Δx grid-mode decay gate's pure helpers.

Pins the metric (amp2dx recovers a known Nyquist amplitude) and the seed (an
index-space 2Δx-in-lon checkerboard adds Nyquist power where there was none), so the
fast §5-residual gate stays valid without needing the Oceananigans reference.
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
def gmd():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.compare_oceananigans_gridmode_decay as mod  # type: ignore
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def test_amp2dx_recovers_nyquist_amplitude(gmd):
    """A pure (-1)^j checkerboard of amplitude A has 2Δx-amplitude exactly A; a
    smooth (low-wavenumber) field has ~zero Nyquist content."""
    nlon = gmd.M_NYQ * 2
    j = np.arange(nlon)
    checker = 0.7 * ((-1.0) ** j)[None, :] * np.ones((10, 1))
    assert gmd.amp2dx(checker) == pytest.approx(0.7, abs=1e-12)
    smooth = np.cos(2 * np.pi * np.arange(nlon) / nlon)[None, :] * np.ones((10, 1))
    assert gmd.amp2dx(smooth) < 1e-12


def test_seed_injects_2dx_mode(gmd):
    """seed_gridmode adds a 2Δx-in-lon mode where the bickley v IC had ~none."""
    bk = gmd.bk
    grid, _wall, _z, state, _model = bk.build_bickley()
    state = bk.set_bickley_ic(grid, state)
    v0 = np.asarray(state.v.data)[:, :, 0]
    amp_before = gmd.amp2dx(v0)
    seeded = gmd.seed_gridmode(grid, state)
    v1 = np.asarray(seeded.v.data)[:, :, 0]
    amp_after = gmd.amp2dx(v1)
    # The smooth bickley IC has negligible Nyquist content; the seed dominates it.
    assert amp_after > 100 * (amp_before + 1e-12)
    assert amp_after > 1e-4
