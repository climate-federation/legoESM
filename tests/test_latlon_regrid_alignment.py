"""Regression guard: the lat-lon cross-grid regrid must land features at the
SAME geographic longitude as the cube/icosa grids.

fv3_faithful (latlon-translation report): the lat-lon FV grid stores winds on
``lon in [0, 360)`` at native (72, 144), while the cube/icosa snapshots regrid
to the common ``(181, 360)`` canvas with lon CELL-centered on ``[-180, 180)``
(``regridding.py``: ``linspace(-180,180,n_lon,endpoint=False)+180/n_lon``).  The
old lat-lon path only ROLLED the native array (kept 144 columns) and stored it
under a 360-point lon label, so the cross-grid comparison plotted a 144-wide
field on a 360-wide axis and continents/jets appeared LONGITUDE-TRANSLATED by
tens of degrees relative to the other grids.

``_regrid_latlon_to_181x360`` bilinearly interpolates (periodic in lon) onto the
exact cube/icosa target.  These tests fail if a refactor reverts the lat-lon
branch to a roll-only path: they assert (a) the helper output is on the (181,360)
canvas for 2D and 3D, (b) a localized feature at a known geographic longitude
lands at the matching cube-canvas column to within one grid cell (no gross
translation), and (c) both regrid call sites route the lat-lon branch through the
interpolating helper rather than ``_roll_lon_to_pm180``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.matrix.run_atmosphere_test_matrix import (
    _regrid_latlon_to_181x360,
    _regrid_2d,
    _regrid_3d_level,
)

# The exact common canvas the cube/icosa regrid to (regridding.py).
_CUBE_LON = np.linspace(-180.0, 180.0, 360, endpoint=False) + 180.0 / 360.0


def _native_bump(truth_lon: float, nlat: int = 72, nlon: int = 144):
    """A localized Gaussian bump at ``truth_lon`` (deg E) on a native lat-lon
    grid with lon in ``[0, 360)`` — i.e. exactly the ``grid.lon`` convention the
    lat-lon FV state carries."""
    lat = np.linspace(-90.0, 90.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    lo, la = np.meshgrid(lon, lat)
    d = np.minimum((lo - truth_lon) % 360.0, (truth_lon - lo) % 360.0)
    field = np.exp(-((d / 15.0) ** 2 + ((la - 40.0) / 10.0) ** 2))
    return field, lon, lat


@pytest.mark.parametrize("truth_lon", [20.0, 200.0, -90.0, 0.0, 179.0])
def test_latlon_regrid_lands_at_geographic_longitude(truth_lon):
    field, lon, lat = _native_bump(truth_lon)
    out = _regrid_latlon_to_181x360(field, lon, lat)
    assert out.shape == (181, 360), f"want (181,360) canvas, got {out.shape}"
    j = int(np.unravel_index(np.argmax(out), out.shape)[1])
    # geographic longitude of the cube-canvas column the bump landed in
    got = ((_CUBE_LON[j] + 180.0) % 360.0) - 180.0
    err = abs(((got - truth_lon + 180.0) % 360.0) - 180.0)
    assert err <= 1.0, (
        f"bump at {truth_lon} deg E landed at cube-canvas lon {got:.1f} "
        f"(err {err:.2f} deg > 1 cell): the lat-lon regrid is TRANSLATED "
        "relative to the cube/icosa canvas — use _regrid_latlon_to_181x360, "
        "not a roll-only path."
    )


def test_latlon_regrid_handles_3d():
    field2d, lon, lat = _native_bump(20.0)
    nlev = 5
    field3d = np.repeat(field2d[..., None], nlev, axis=2)
    out = _regrid_latlon_to_181x360(field3d, lon, lat)
    assert out.shape == (181, 360, nlev), f"3D canvas wrong: {out.shape}"
    # each level identical (input was level-invariant) and aligned
    j = int(np.unravel_index(np.argmax(out[..., 0]), out[..., 0].shape)[1])
    got = ((_CUBE_LON[j] + 180.0) % 360.0) - 180.0
    assert abs(((got - 20.0 + 180.0) % 360.0) - 180.0) <= 1.0
    assert np.allclose(out[..., 0], out[..., -1])


def test_regrid_dispatch_uses_canvas_for_latlon():
    """2D and 3D dispatch must route the lat-lon branch through the (181,360)
    canvas regrid (shape (181,360)), not leave the field at native width."""
    field2d, lon, lat = _native_bump(20.0)
    out2d = _regrid_2d(field2d, lon, lat, "latlon")
    assert out2d.shape == (181, 360)
    out3d = _regrid_3d_level(field2d[..., None], lon, lat, "latlon")
    assert out3d.shape[:2] == (181, 360)


def test_gaussian_dispatch_shares_canvas():
    """The Gaussian/spectral branch (non-uniform lat, uniform lon) must also
    land on the (181,360) canvas — it previously interpolated only the lat axis
    and kept the native (~64) lon width, leaving the spectral snapshot narrower
    and longitude-shifted vs the other grids."""
    # Gaussian-like non-uniform latitudes, uniform lon in [0,360).
    nlat, nlon = 32, 64
    lat_g = np.degrees(np.arcsin(np.linspace(-0.98, 0.98, nlat)))  # clustered
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    lo, la = np.meshgrid(lon, lat_g)
    d = np.minimum((lo - 20.0) % 360.0, (20.0 - lo) % 360.0)
    field = np.exp(-((d / 20.0) ** 2 + ((la - 40.0) / 15.0) ** 2))
    out2d = _regrid_2d(field, lon, lat_g, "gaussian")
    assert out2d.shape == (181, 360), f"gaussian 2D canvas wrong: {out2d.shape}"
    out3d = _regrid_3d_level(field[..., None], lon, lat_g, "gaussian")
    assert out3d.shape[:2] == (181, 360)
    j = int(np.unravel_index(np.argmax(out2d), out2d.shape)[1])
    got = ((_CUBE_LON[j] + 180.0) % 360.0) - 180.0
    assert abs(((got - 20.0 + 180.0) % 360.0) - 180.0) <= 6.0  # ~1 coarse cell


def test_latlon_regrid_no_polar_extrapolation_overshoot():
    """The canvas reaches the poles (+-90) but source grids stop short of them
    (lat-lon cell-centers ~+-88.75; gaussian lats ~+-85).  Uncorrected, the polar
    target rows fell outside the source range and RegularGridInterpolator LINEAR-
    EXTRAPOLATED, which overshot: it drove nonnegative magnitudes (wind_speed)
    NEGATIVE and inflated extrema in the polar rows on the lat-lon and spectral
    regrid paths (cube/icosa use bounded weights and stayed clean -> the observed
    cross-grid inconsistency).  The regrid must instead clamp the target latitude
    to the source span (bounded nearest-edge hold).  A nonnegative field whose
    minimum sits AT the source edge is the sharpest probe: any extrapolation past
    the edge necessarily goes negative."""
    for name, lat_src in (
        ("gaussian", np.degrees(np.arcsin(np.linspace(-0.98, 0.98, 32)))),
        ("latlon", np.linspace(-88.75, 88.75, 72)),
    ):
        nlon = 64
        lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
        # >= 0 everywhere on the source, == 0 exactly at the southern edge, with a
        # positive meridional slope -> linear extrapolation toward -90 goes < 0.
        field = (lat_src[:, None] - lat_src.min()) + 0.0 * lon[None, :]
        out = _regrid_latlon_to_181x360(field, lon, lat_src)
        assert np.nanmin(out) >= -1e-9, (
            f"[{name}] polar extrapolation drove a nonnegative field negative "
            f"(min={np.nanmin(out):.4g}) — clamp target lat to the source range."
        )
        # No overshoot beyond the source extrema on either side.
        assert np.nanmax(out) <= float(field.max()) + 1e-9, (
            f"[{name}] regrid overshot the source maximum "
            f"({np.nanmax(out):.4g} > {field.max():.4g})."
        )


def test_source_no_roll_only_latlon_branch():
    """Source guard: neither regrid dispatcher may send the lat-lon branch to a
    roll-only path — that re-introduces the gross longitude translation."""
    src = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "matrix" / "run_atmosphere_test_matrix.py"
    ).read_text()
    assert src.count("_regrid_latlon_to_181x360(") >= 3, (
        "both _regrid_2d and _regrid_3d_level lat-lon branches (and the helper) "
        "must use _regrid_latlon_to_181x360 to share the cube/icosa canvas."
    )
    assert "if False and coord_kind" not in src, (
        "dead roll-only lat-lon branch left in the regrid dispatch."
    )
