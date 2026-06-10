"""iter73: the gnomonic_ed cdgrid extended-grid halo must NOT collapse at the
cube corners.

The earlier construct→extended-θ halo (`_gnomonic_ed_6face_from_theta` with a
halo-extended ``theta``) could not extend the edge-PERPENDICULAR direction — it
pins the W/E edges to the boundary meridians (where the gnomonic coord
``pp2 = -tan(0.75π)·rsq3`` is CONSTANT for all lat), so the ±1 halo ring
collapsed onto the boundary at the 4 cube corners (zero-width halo cells →
duplicate grid nodes → degenerate corner metrics).

The fix (`_gnomonic_ed_faces_from_angle_1d` + `_gnomonic_ed_extrap1d`) builds the
ed extended grids as the SAME-FACE continuation via the tested forward map
`face_gnomonic_to_lonlat` with the separable 1D ed angle array extrapolated for
the halo — the direct ed analog of equiangular's ``linspace`` extension.

These pins guard that:
  * the halo ring steps OFF the boundary everywhere (no zero-width halo cells);
  * the in-domain block is EXACTLY the gnomonic_ed corners / 2× supergrid
    (refinement consistency with the A-grid);
  * the extended grids are mutually consistent (supergrid even nodes == corners,
    padded-supergrid interior == supergrid).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (  # noqa: E402
    gnomonic_ed_corner_ext_lonlat,
    gnomonic_ed_supergrid_lonlat,
    gnomonic_ed_padded_supergrid_lonlat,
    make_fv3_native_grid,
    gnomonic_ed_remap_to_create,
)


def _gcd(la1, lo1, la2, lo2):
    a = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@pytest.mark.parametrize("n", [24, 36])
def test_corner_ext_halo_does_not_collapse(n):
    """Every halo-vs-first-interior node spacing must be a real (≈1 cell)
    great-circle distance — never zero (the corner-collapse signature)."""
    lon, lat = (np.asarray(a) for a in gnomonic_ed_corner_ext_lonlat(n))
    assert lon.shape == (6, n + 3, n + 3)
    # interior cell size scale (mid-edge, in-domain) for a relative floor
    mid = _gcd(lat[:, n // 2 + 1, n // 2 + 1], lon[:, n // 2 + 1, n // 2 + 1],
               lat[:, n // 2 + 2, n // 2 + 1], lon[:, n // 2 + 2, n // 2 + 1]).mean()
    for f in range(6):
        for (a_idx, b_idx) in ((0, 1), (-1, -2)):  # W/E halo rows vs first interior
            s_i = _gcd(lat[f, a_idx, :], lon[f, a_idx, :],
                       lat[f, b_idx, :], lon[f, b_idx, :])
            s_j = _gcd(lat[f, :, a_idx], lon[f, :, a_idx],
                       lat[f, :, b_idx], lon[f, :, b_idx])
            assert s_i.min() > 0.3 * mid, (
                f"face{f} i-halo collapsed: min spacing {s_i.min():.2e} "
                f"<< cell {mid:.2e}")
            assert s_j.min() > 0.3 * mid, (
                f"face{f} j-halo collapsed: min spacing {s_j.min():.2e} "
                f"<< cell {mid:.2e}")


def test_corner_ext_interior_matches_corners():
    """Refinement consistency: corner_ext[1:-1, 1:-1] == the gnomonic_ed corners."""
    n = 36
    lon, lat = (np.asarray(a) for a in gnomonic_ed_corner_ext_lonlat(n))
    lc, la = (np.asarray(a) for a in gnomonic_ed_remap_to_create(
        *make_fv3_native_grid(n, grid_type=0)))
    sep = _gcd(la, lc, lat[:, 1:-1, 1:-1], lon[:, 1:-1, 1:-1])
    assert sep.max() < 1e-12, f"corner_ext interior diverges from corners by {sep.max():.2e}"


def test_padded_supergrid_interior_matches_supergrid_and_corners():
    n = 36
    plon, plat = (np.asarray(a) for a in gnomonic_ed_padded_supergrid_lonlat(n))
    slon, slat = (np.asarray(a) for a in gnomonic_ed_supergrid_lonlat(n))
    assert plon.shape == (6, 2 * n + 3, 2 * n + 3)
    sep = _gcd(slat, slon, plat[:, 1:-1, 1:-1], plon[:, 1:-1, 1:-1])
    assert sep.max() < 1e-12, f"padded_sg interior != supergrid ({sep.max():.2e})"
    # supergrid even nodes reproduce the corners
    lc, la = (np.asarray(a) for a in gnomonic_ed_remap_to_create(
        *make_fv3_native_grid(n, grid_type=0)))
    sep2 = _gcd(la, lc, slat[:, ::2, ::2], slon[:, ::2, ::2])
    assert sep2.max() < 1e-12, f"supergrid even nodes != corners ({sep2.max():.2e})"
