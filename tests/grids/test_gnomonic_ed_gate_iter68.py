"""iter68: the gated create_cubed_sphere(gnomonic="ed") path — assembling all
the gnomonic_ed builders into a production CubedSphereGrid.

Pins that the FV3 operational gnomonic_ed grid builds with the right signature
(√2 dx ratio, 1.06 cell aspect, sphere-closing area, finite metrics/offsets),
that the equiangular default is unaffected (distinct, coarser conditioning), and
that the ed path correctly rejects the incompatible Schmidt/shift transforms.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere  # noqa: E402
from legoesm import constants  # noqa: E402

R = float(constants.R_earth)


def test_ed_builds_with_fv3_signature():
    g = create_cubed_sphere(24, dtype=np.float64, gnomonic="ed")
    dx = np.asarray(g.dx); dy = np.asarray(g.dy); area = np.asarray(g.area)
    # area closes to the sphere
    assert abs(area.sum() / (4 * np.pi * R ** 2) - 1.0) < 1e-9
    # √2 edge-ratio + 1.06 aspect signatures
    assert abs(dx.max() / dx.min() - np.sqrt(2.0)) < 0.05
    aspect = (np.maximum(dx, dy) / np.minimum(dx, dy)).max()
    assert aspect < 1.10, f"gnomonic_ed cell aspect {aspect:.3f} not ~1.06"
    # all assembled fields finite
    for fld in ("lon", "lat", "angle", "angle_padded", "hx_ext", "hy_ext",
                "halo_interp_offsets", "x_cart", "f"):
        assert np.all(np.isfinite(np.asarray(getattr(g, fld)))), fld


def test_equiangular_default_distinct_and_coarser():
    """Default path is unaffected: equiangular cells are markedly more
    distorted (aspect ~1.4) than gnomonic_ed (~1.06)."""
    g = create_cubed_sphere(24, dtype=np.float64)
    dx = np.asarray(g.dx); dy = np.asarray(g.dy)
    aspect = (np.maximum(dx, dy) / np.minimum(dx, dy)).max()
    assert aspect > 1.3, f"equiangular aspect {aspect:.3f} unexpectedly uniform"


def test_ed_rejects_schmidt_and_shift():
    with pytest.raises(ValueError):
        create_cubed_sphere(24, gnomonic="ed", stretch_fac=2.0)
    with pytest.raises(ValueError):
        create_cubed_sphere(24, gnomonic="ed", shift_fac=18.0)


def test_unknown_gnomonic_raises():
    with pytest.raises(ValueError):
        create_cubed_sphere(24, gnomonic="banana")


def test_cdgrid_auto_follows_provenance_no_second_flag():
    """create_cubed_sphere_cdgrid(grid) with NO explicit gnomonic flag must
    follow the base grid's STATIC provenance, so an ed A-grid never silently
    pairs with an equiangular C/D grid (the iter72 corruption path: model
    constructors call it with just `grid`).  Phase-1 FV3-native work replaced
    the old dx/dy aspect-ratio inference with the provenance read, and a
    contradicting explicit request is now a hard error rather than a silent
    metric-family mix."""
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

    g_ed = create_cubed_sphere(24, dtype=np.float64, gnomonic="ed")
    cd_auto = create_cubed_sphere_cdgrid(g_ed)                       # auto
    cd_ed = create_cubed_sphere_cdgrid(g_ed, gnomonic="ed")          # explicit ed
    # auto on an ed grid must reproduce the EXPLICIT ed cdgrid exactly...
    assert np.allclose(np.asarray(cd_auto.dxc), np.asarray(cd_ed.dxc), atol=1e-12), (
        "auto did not follow ed provenance on an ed base grid")
    # ...an explicitly CONTRADICTING request is the corruption path — hard error
    with pytest.raises(ValueError, match="provenance"):
        create_cubed_sphere_cdgrid(g_ed, gnomonic="equiangular")

    g_eq = create_cubed_sphere(24, dtype=np.float64)
    cd_eq_auto = create_cubed_sphere_cdgrid(g_eq)  # auto → equiangular
    cd_eq_ex = create_cubed_sphere_cdgrid(g_eq, gnomonic="equiangular")
    assert np.allclose(np.asarray(cd_eq_auto.dxc), np.asarray(cd_eq_ex.dxc), atol=1e-12), (
        "auto did not follow equiangular provenance on an equiangular base grid")
    # ed and equiangular metric families must stay distinguishable
    assert not np.allclose(np.asarray(cd_ed.dxc), np.asarray(cd_eq_auto.dxc), atol=1.0), (
        "ed and equiangular cdgrids are indistinguishable — provenance is moot")
