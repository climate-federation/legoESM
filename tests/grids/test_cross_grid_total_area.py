"""Cross-grid domain-metric consistency.

Every cell-centered global grid must sum its cell area to 4πR² and
satisfy the area-weighted moments expected of an unbiased sphere
covering. This test pins the consistency the audit script
``scripts/audit_grid_metrics.py`` enforces so a regression in any one
grid factory is caught at unit-test time.

The dominant historical break was the uniform lat-lon factory's
midpoint cell-area formula ``R² dφ dλ cos(φ)`` overshooting by
``dφ²/24`` — 80 ppm at n_lat=72, 300 ppm at n_lat=36 — see
``_exact_uniform_cell_area_lat`` in ``legoesm.grids.latlon``.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.voronoi import create_voronoi_mesh

R = float(constants.R_earth)
SPHERE_AREA = 4.0 * np.pi * R * R

# fp32 storage policy: stored area arrays have ~1.2e-7 relative round-off.
# Tolerance covers that plus a small margin.
TOL_TOTAL = 2.0e-7
TOL_SIN = 2.0e-7


@pytest.mark.parametrize(
    "name,builder",
    [
        ("cubed_sphere/C24", lambda: create_cubed_sphere(24)),
        ("cubed_sphere/C36", lambda: create_cubed_sphere(36)),
        ("latlon/36x72", lambda: create_latlon_grid(36, 72)),
        ("latlon/72x144", lambda: create_latlon_grid(72, 144)),
        ("icosahedral/ico3", lambda: create_voronoi_mesh(3)),
        ("icosahedral/ico5", lambda: create_voronoi_mesh(5)),
        ("spectral/T21", lambda: create_gaussian_grid(21)),
        ("spectral/T42", lambda: create_gaussian_grid(42)),
    ],
)
def test_total_area_matches_sphere(name, builder):
    grid = builder()
    total = float(np.asarray(grid.grid_area).sum())
    drel = (total - SPHERE_AREA) / SPHERE_AREA
    assert abs(drel) < TOL_TOTAL, (
        f"{name}: total_area drel={drel:+.3e} exceeds {TOL_TOTAL:.0e} "
        f"(sum={total:.6e}, 4πR²={SPHERE_AREA:.6e})"
    )


@pytest.mark.parametrize(
    "name,builder",
    [
        ("cubed_sphere/C24", lambda: create_cubed_sphere(24)),
        ("latlon/72x144", lambda: create_latlon_grid(72, 144)),
        ("icosahedral/ico5", lambda: create_voronoi_mesh(5)),
        ("spectral/T21", lambda: create_gaussian_grid(21)),
    ],
)
def test_area_weighted_sin_lat_is_zero(name, builder):
    """∫sin(φ) over sphere = 0 — any unbiased covering must match."""
    grid = builder()
    area = np.asarray(grid.grid_area).ravel()
    lat = np.asarray(grid.grid_lat).ravel()
    mean = float((np.sin(lat) * area).sum() / area.sum())
    assert abs(mean) < TOL_SIN, (
        f"{name}: <sin lat>={mean:+.3e} exceeds {TOL_SIN:.0e}"
    )


def test_uniform_latlon_area_exact_sum():
    """Direct check on the helper formula 2 R² dλ sin(dφ/2) cos(φ_c)."""
    grid = create_latlon_grid(180, 360)
    total = float(np.asarray(grid.grid_area).sum())
    drel = (total - SPHERE_AREA) / SPHERE_AREA
    # 180x360 stays inside the fp32 cast precision band as well.
    assert abs(drel) < TOL_TOTAL, (
        f"180x360 latlon total_area drel={drel:+.3e} > {TOL_TOTAL:.0e}"
    )


def test_cross_grid_total_area_agreement_at_matrix_resolutions():
    """Atmosphere-matrix grids agree on total area to fp32 precision."""
    grids = {
        "cubed_sphere/C36": create_cubed_sphere(36),
        "latlon/72x144": create_latlon_grid(72, 144),
        "icosahedral/ico5": create_voronoi_mesh(5),
        "spectral/T21": create_gaussian_grid(21),
    }
    totals = {n: float(np.asarray(g.grid_area).sum()) for n, g in grids.items()}
    ref_name, ref_val = next(iter(totals.items()))
    for n, v in totals.items():
        drel = (v - ref_val) / ref_val
        assert abs(drel) < TOL_TOTAL, (
            f"{n} vs {ref_name}: drel={drel:+.3e} > {TOL_TOTAL:.0e} "
            f"({v:.6e} vs {ref_val:.6e})"
        )
