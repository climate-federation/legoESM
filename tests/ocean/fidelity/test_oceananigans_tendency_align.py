"""Unit test for the tendency-match driver's grid-alignment helper.

The Phase-1 tendency-match harness (compare_oceananigans_tendency.py) aligns a
legoESM (lat, lon) field — which carries wall rows + a periodic-lon wrap — onto an
Oceananigans (lat, lon) reference by searching a lat-offset + lon-roll, validated
on the IC. This pins that helper so the alignment logic can't silently regress.
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
def tendency_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.compare_oceananigans_tendency as mod  # type: ignore
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def test_best_align_recovers_known_map(tendency_module):
    rng = np.random.default_rng(0)
    nlat, nlon = 12, 16
    # Periodic-in-lon reference (so the legoESM wrap column is a true duplicate and
    # a lon-roll is well defined). legoESM field: ocn at lat-offset 1 (wall rows
    # top+bottom) + the periodic wrap column.
    ocn = np.sin(np.linspace(0, 4 * np.pi, nlon))[None, :] * rng.standard_normal((nlat, 1))
    lego = np.zeros((nlat + 2, nlon + 1))
    lego[1:1 + nlat, :nlon] = ocn
    lego[:, nlon] = lego[:, 0]                           # periodic wrap column
    la0, roll, corr = tendency_module._best_align(lego, ocn, "test")
    assert corr > 0.999, f"alignment corr {corr} too low"
    assert la0 == 1, f"lat offset {la0} != 1"
    rec = np.roll(lego, roll, axis=1)[la0:la0 + nlat, :nlon]
    assert np.corrcoef(rec.ravel(), ocn.ravel())[0, 1] > 0.999


def test_best_align_handles_constant_field(tendency_module):
    # A constant (zero-variance) field must not crash and returns a default.
    lego = np.ones((6, 9))
    ocn = np.ones((4, 8))
    la0, roll, corr = tendency_module._best_align(lego, ocn, "const")
    assert isinstance(la0, int) and isinstance(roll, int)
