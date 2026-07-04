"""NEMO ln_rnf_depth_ini per-cell runoff spread-depth map
(forcing/runoff_depth.py) — locked against an F90 transliteration of
sbcrnf.F90's initialisation block."""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.forcing.runoff_depth import nemo_runoff_depth_map


def _f90_reference(rnf_monthly, H, dep_max=150.0, rnf_max=0.05):
    """Literal transliteration: zrnfcl max-loop + WHERE + bottom clamp."""
    zrnfcl = np.zeros_like(rnf_monthly[0])
    for jm in range(rnf_monthly.shape[0]):
        zrnfcl = np.maximum(zrnfcl, rnf_monthly[jm])
    zacoef = dep_max / rnf_max
    h_rnf = np.where(zrnfcl > 0.0, zacoef * zrnfcl, 1.0)
    out = h_rnf.copy()
    ny, nx = h_rnf.shape
    for jj in range(ny):
        for ji in range(nx):
            if zrnfcl[jj, ji] > 0.0:
                out[jj, ji] = min(h_rnf[jj, ji], H[jj, ji])
    return out


def test_matches_f90_transliteration():
    rng = np.random.default_rng(0)
    rnf = np.zeros((12, 6, 8))
    # a few rivers with seasonal cycles; one Amazon-strength, one Arctic
    rnf[:, 2, 3] = 0.05 * (0.5 + 0.5 * np.sin(np.arange(12)))  # big river
    rnf[:, 4, 6] = 1.0e-3 * (1 + np.arange(12) % 3)            # small river
    rnf[:, 1, 1] = 0.2                                          # > rnf_max
    H = rng.uniform(20.0, 4000.0, (6, 8))
    got = nemo_runoff_depth_map(rnf, H)
    ref = _f90_reference(rnf, H)
    # the module also floors dry/clamped cells at 1 m (h_dry); the F90
    # reference floors implicitly via ELSEWHERE=1 — apply the same floor
    np.testing.assert_allclose(got, np.maximum(ref, 1.0), rtol=1e-13)


def test_amazon_vs_arctic_contrast():
    rnf = np.zeros((12, 2, 2))
    rnf[:, 0, 0] = 0.05       # Amazon-strength -> 150 m
    rnf[:, 1, 1] = 1.0e-3     # Arctic river    -> 3 m
    H = np.full((2, 2), 4000.0)
    h = nemo_runoff_depth_map(rnf, H)
    assert h[0, 0] == pytest.approx(150.0)
    assert h[1, 1] == pytest.approx(3.0)
    assert h[0, 1] == pytest.approx(1.0)          # no runoff -> 1 m


def test_bottom_clamp_and_floor():
    rnf = np.zeros((12, 1, 2))
    rnf[:, 0, 0] = 0.05
    H = np.array([[35.0, 0.0]])
    h = nemo_runoff_depth_map(rnf, H)
    assert h[0, 0] == pytest.approx(35.0)         # capped at local depth
    assert h[0, 1] == pytest.approx(1.0)          # dry floor


def test_annual_field_and_bad_shape_raise():
    H = np.full((3, 3), 100.0)
    h = nemo_runoff_depth_map(np.full((3, 3), 0.01), H)   # month-free OK
    assert np.allclose(h, 30.0)
    with pytest.raises(ValueError, match="shape"):
        nemo_runoff_depth_map(np.zeros((5, 4)), H)
    with pytest.raises(ValueError, match="must be > 0"):
        nemo_runoff_depth_map(np.zeros((12, 3, 3)), H, rnf_max=0.0)
