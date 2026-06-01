"""iter62: first brick of the gated gnomonic_ed grid wiring —
`_compute_gnomonic_ed_lonlat(n)` cell centers (FV3 operational grid_type=0).

Pins that the gnomonic_ed center routine (a) returns the same (6,n,n) shape
as the equiangular `_compute_gnomonic_lonlat` it will replace, (b) is on the
sphere, (c) is genuinely the gnomonic_ed grid (distinct from equiangular and
carrying the √2 / near-uniform signature to cell centers), so a future
`create_cubed_sphere(gnomonic="ed")` inherits the FV3-faithful grid.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (  # noqa: E402
    _compute_gnomonic_ed_lonlat, _compute_gnomonic_lonlat, create_cubed_sphere,
)
from legoesm.grids.halo import compute_cross_face_continuity  # noqa: E402


def _gcd(la1, lo1, la2, lo2):
    dl = lo2 - lo1
    dla = la2 - la1
    a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dl / 2) ** 2
    return 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def test_shape_and_on_sphere():
    n = 24
    lon, lat = _compute_gnomonic_ed_lonlat(n)
    assert lon.shape == (6, n, n) and lat.shape == (6, n, n)
    lon, lat = np.asarray(lon), np.asarray(lat)
    assert np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))
    assert np.all(np.abs(lat) <= np.pi / 2 + 1e-9)
    assert np.all((lon >= -1e-9) & (lon <= 2 * np.pi + 1e-9))


def test_distinct_from_equiangular():
    """gnomonic_ed centers must differ from equiangular — same grid would
    mean the routine silently fell back to grid_type=2."""
    n = 24
    lon_ed, lat_ed = (np.asarray(a) for a in _compute_gnomonic_ed_lonlat(n))
    lon_eq, lat_eq = (np.asarray(a) for a in _compute_gnomonic_lonlat(n))
    # geodesic separation between the two center sets, max over all cells
    sep = _gcd(lat_ed, lon_ed, lat_eq, lon_eq)
    assert sep.max() > 1e-3, (
        "gnomonic_ed centers coincide with equiangular — not the operational grid")


def test_faces_at_create_positions():
    """After remap, gnomonic_ed faces must sit at create_cubed_sphere's face
    positions (else the halo tables are inconsistent).  Equatorial faces 0-3
    are checked by centre lon/lat; polar faces 4-5 by centre lat=±90 (centre
    lon is degenerate at a pole)."""
    n = 24
    lon_ed, lat_ed = (np.asarray(a) for a in _compute_gnomonic_ed_lonlat(n))
    g = create_cubed_sphere(n)
    lon_c, lat_c = np.asarray(g.lon), np.asarray(g.lat)

    def center(lon, lat, f):
        x = np.cos(lat[f]) * np.cos(lon[f]); y = np.cos(lat[f]) * np.sin(lon[f])
        z = np.sin(lat[f]); xm, ym, zm = x.mean(), y.mean(), z.mean()
        r = np.sqrt(xm ** 2 + ym ** 2 + zm ** 2)
        return np.degrees(np.arctan2(ym, xm)) % 360, np.degrees(np.arcsin(zm / r))

    for f in range(4):  # equatorial: lon + lat
        le, la = center(lon_ed, lat_ed, f); lc, lat_cc = center(lon_c, lat_c, f)
        assert abs(((le - lc + 180) % 360) - 180) < 2.0 and abs(la - lat_cc) < 2.0, (
            f"gnomonic_ed face {f} at ({le:.0f},{la:.0f}) != create ({lc:.0f},{lat_cc:.0f})")
    for f in (4, 5):  # polar: lat only
        _, la = center(lon_ed, lat_ed, f); _, lat_cc = center(lon_c, lat_c, f)
        assert abs(la - lat_cc) < 2.0 and abs(abs(la) - 90.0) < 2.0, (
            f"gnomonic_ed polar face {f} lat {la:.0f} != create {lat_cc:.0f}")


def test_seam_continuous_in_create_topology():
    """The remapped gnomonic_ed grid must be seam-continuous through create's
    halo tables (the cross-face metric uses pad_halo_4d).  A wrong face
    permutation/orientation would show large cross-seam jumps."""
    n = 24
    _, lat_ed = _compute_gnomonic_ed_lonlat(n)
    m = compute_cross_face_continuity(np.asarray(lat_ed))  # lat = geographic-continuous
    assert m["max_ratio"] < 2.5, (
        f"gnomonic_ed seam continuity {m['max_ratio']:.2f} > 2.5 — face "
        f"permutation/orientation inconsistent with create's halo topology")


def test_center_spacing_carries_sqrt2_signature():
    """The √2 max/min spacing signature of gnomonic_ed should be visible in
    the centre-to-centre great-circle distances too (equiangular is ~1.4 on
    dx but its cell ASPECT is 1.40; gnomonic_ed keeps aspect ~1.06).  Here we
    check the center grid is the equal-edge family: max/min center spacing
    along a face row is close to √2, not larger."""
    n = 48
    lon, lat = (np.asarray(a) for a in _compute_gnomonic_ed_lonlat(n))
    # i-direction center spacing on face 0
    dx = _gcd(lat[0, :-1, :], lon[0, :-1, :], lat[0, 1:, :], lon[0, 1:, :])
    ratio = dx.max() / dx.min()
    assert 1.3 < ratio < 1.55, (
        f"gnomonic_ed center i-spacing ratio {ratio:.3f} outside the √2 "
        f"equal-edge band [1.3,1.55]")
