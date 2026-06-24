"""Unit tests for the geostrophic-adjustment comparison driver's helpers.

The geostrophic-adjustment harness is a deterministic test of the free-surface
predictor-corrector COUPLING (the §5-residual node). This pins its pure helpers
(the Gaussian-bump IC and the (lat,lon) alignment search).
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
def ga(tmp_path_factory):
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.compare_oceananigans_geostrophic_adjustment as mod  # type: ignore
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def test_bump_peaks_at_centre(ga):
    assert ga._bump(0.0, 0.0) == pytest.approx(ga.ETA0)
    # decays away from centre and is symmetric
    assert ga._bump(30.0, 0.0) < ga.ETA0
    assert ga._bump(30.0, 0.0) == pytest.approx(ga._bump(0.0, 30.0))
    assert ga._bump(0.0, 30.0) == pytest.approx(ga._bump(0.0, -30.0))


def test_align_recovers_offset(ga):
    rng = np.random.default_rng(1)
    nlat, nlon = 10, 14
    ocn = np.sin(np.linspace(0, 4 * np.pi, nlon))[None, :] * rng.standard_normal((nlat, 1))
    lego = np.zeros((nlat + 2, nlon + 1))
    lego[1:1 + nlat, :nlon] = ocn
    lego[:, nlon] = lego[:, 0]
    corr, la0, roll = ga._align(lego, ocn)
    assert corr > 0.999 and la0 == 1
