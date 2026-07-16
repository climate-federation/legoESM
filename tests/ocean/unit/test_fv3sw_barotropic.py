"""Regression guard: the cube ocean barotropic must use the FV3-faithful SW core
(``fv3sw``) and stay zonally symmetric — never the A-grid solver.

fv3_faithful (ocean): the cube ocean barotropic defaulted to an A-grid solver
whose computational pressure mode grew a ~40% non-zonal eta artifact in
geostrophic_adjustment (zonal thermal-wind IC) vs ~0 on latlon/MPAS.  Per the
never-A-grid / FV3-faithfulness directive the barotropic free-surface mode (a 2-D
shallow-water system) is routed through the validated FV3 cube SW core
(``CDGridShallowWaterModel`` — vector-invariant absolute-vorticity-flux Coriolis,
SSP-RK3, divergence damping + hyperdiffusion).  An isolated zonal-eta initial
state stays bit-stably zonal under that core, where the explicit-f*v A-/C-grid
solvers either grow the imprint or blow up.

These tests fail if a refactor reverts the cube barotropic to A-grid or breaks the
SW-core zonal-symmetry property.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.grids.cubed_sphere import create_cubed_sphere  # noqa: E402
from legoesm.grids.regridding import (  # noqa: E402
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon,
)
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (  # noqa: E402
    CDGridShallowWaterModel, CDGridShallowWaterState, iter1009_dual_target_config,
)


def _nonzonal_fraction(eta_cube, weights):
    ll = apply_cubedsphere_to_latlon(np.asarray(eta_cube), weights)
    zm = np.nanmean(ll, axis=1, keepdims=True)
    return float(np.nansum((ll - zm) ** 2)
                 / max(np.nansum((ll - np.nanmean(ll)) ** 2), 1e-30))


def test_sw_core_barotropic_keeps_zonal_eta_zonal():
    """The FV3 SW core (the fv3sw barotropic engine) advances a zonally-symmetric
    free-surface state while preserving its zonal symmetry and staying stable —
    the property the A-grid solver violated (40% non-zonal imprint)."""
    n, H = 24, 5500.0
    grid = create_cubed_sphere(n)
    model = CDGridShallowWaterModel(
        grid, iter1009_dual_target_config(n, div_damp_factor=120.0))
    lat = np.asarray(grid.lat) * 180.0 / np.pi
    eta0 = jnp.asarray(0.1 * np.exp(-((lat - 30.0) / 20.0) ** 2), jnp.float64)
    h_s = jnp.asarray(-H * np.ones((6, n, n)), jnp.float64)
    st = CDGridShallowWaterState(
        h=jnp.asarray(H + eta0, jnp.float64),
        u_d=jnp.zeros((6, n + 1, n + 1), jnp.float64),
        v_d=jnp.zeros((6, n + 1, n + 1), jnp.float64),
        h_s=h_s,
    )
    w = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)
    nz0 = _nonzonal_fraction(eta0, w)
    for _ in range(1200):  # 1200 * 5 s = ~1.7 h of barotropic substeps
        st = model.step(st, 5.0)
    eta = st.h - H
    assert bool(jnp.all(jnp.isfinite(eta))), "SW-core barotropic went non-finite"
    nz = _nonzonal_fraction(eta, w)
    assert nz < 0.05, (
        f"SW-core barotropic broke zonal symmetry: non-zonal fraction {nz:.3f} "
        f"(init {nz0:.3f}); a zonally-symmetric free surface must stay zonal — "
        "this is the A-grid computational-mode artifact returning."
    )


def test_cube_ocean_barotropic_is_fv3sw_never_a_grid():
    """Source guard: the ocean test matrix must configure the cube ocean with the
    FV3-faithful ``fv3sw`` barotropic, never the A-grid solver.

    The cube ocean config now lives in the shared
    ``scripts/matrix/ocean_test_matrix/setup.py`` block
    (``cube_matrix_ocean_config_kwargs``) which ``run_ocean_test_matrix.py``
    calls — so the single source of truth for the cube barotropic staggering is
    that module."""
    src = (
        Path(__file__).resolve().parents[3]
        / "scripts" / "matrix" / "ocean_test_matrix" / "setup.py"
    ).read_text()
    # The cube ocean config block must request fv3sw.
    assert 'barotropic_staggering="fv3sw"' in src, (
        "the cube ocean must use barotropic_staggering='fv3sw' (FV3 SW core)."
    )
    assert 'barotropic_staggering="a_grid"' not in src, (
        "the cube ocean must NOT use the A-grid barotropic (never-A-grid "
        "directive: it grows the geostrophic non-zonal computational mode)."
    )
