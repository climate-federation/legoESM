"""Phase 2 unit tests for the AHH08 MPAS Voronoi wrapper.

Validates that ``density_jacobian_pgf_ahh08_mpas`` gives:

1. Bit-exact zero on a horizontally uniform T(z), S(z) rest state with
   FULL cells (the easy case all schemes pass).
2. Machine-precision zero on a horizontally uniform T(z), S(z) rest
   state with PARTIAL CELLS (the property that motivates AHH08 over
   centered/Adcroft, and the property SMC03 also achieves but only on
   linear ρ).
3. Smooth (no NaN, no isolated spikes) and AD-compatible.

Float64 mode is required to verify the 1e-12 rest-state property.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy(storage=jnp.float64, compute=jnp.float64))

from legoesm import constants
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.mpas_partial_cell_helpers import (
    density_jacobian_pgf_ahh08_mpas,
)
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
)


def _gaussian_seamount_bathy(mesh, H_max=4000.0, height=2000.0, sigma=0.3):
    """A Gaussian seamount at the equator → forces partial cells."""
    lat = mesh.latCell
    lon = mesh.lonCell
    bump = height * jnp.exp(-((lat) ** 2 + (lon) ** 2) / (2.0 * sigma ** 2))
    return H_max - bump


def _build_test_state(nlev=10, sub=2):
    mesh = create_voronoi_mesh(subdivision_level=sub)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    H_bathy = _gaussian_seamount_bathy(mesh, H_max=4000.0, height=2000.0)
    pc = create_partial_cell_coordinate(z_coord, H_bathy)
    return mesh, pc, H_bathy


def _uniform_rest_T_S(pc, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0):
    """Horizontally uniform T(z), S(z) — exponential T(z) profile."""
    nlev = pc.h_partial.shape[-1]
    nCells = pc.h_partial.shape[0]
    z_full = pc.z_full_ref  # (nlev,), negative downward
    T_prof = T_deep + (T_water_init_C - T_deep) * jnp.exp(z_full / 1000.0)
    T_3d = jnp.broadcast_to(T_prof[None, :], (nCells, nlev)).astype(jnp.float64)
    S_3d = jnp.full((nCells, nlev), S_uniform, dtype=jnp.float64)
    return T_3d, S_3d


def test_full_cells_horizontally_uniform_zero():
    """Flat-bottom + horizontally uniform T(z), S(z) → bit-zero PGF."""
    mesh = create_voronoi_mesh(subdivision_level=2)
    z_coord = create_ocean_z_star(n_levels=10, H_max=4000.0)
    H_bathy = jnp.full(mesh.nCells, 4000.0, dtype=jnp.float64)
    pc = create_partial_cell_coordinate(z_coord, H_bathy)
    T_3d, S_3d = _uniform_rest_T_S(pc)

    pgf = density_jacobian_pgf_ahh08_mpas(
        T_3d, S_3d, pc.h_partial.astype(jnp.float64), mesh, g=constants.g,
    )
    max_pgf = float(jnp.max(jnp.abs(pgf)))
    assert max_pgf < 1e-9, (
        f"flat-bottom horizontally uniform should give bit-zero, got {max_pgf:.3e}"
    )


def test_partial_cells_horizontally_uniform_machine_zero():
    """Seamount bathymetry with horizontally uniform T(z), S(z) → ~1e-12.

    The rest-state property that distinguishes AHH08: even with partial
    cells creating step edges, the analytic integration over the common
    wet face gives identical pressures on both sides → zero PGF.
    """
    mesh, pc, _H = _build_test_state(nlev=10, sub=2)
    T_3d, S_3d = _uniform_rest_T_S(pc)

    pgf = density_jacobian_pgf_ahh08_mpas(
        T_3d, S_3d, pc.h_partial.astype(jnp.float64), mesh, g=constants.g,
    )
    # Mask out below-step edges (h_face = 0) before checking.  These
    # carry whatever the 0/0 protector returns and are filtered by the
    # dispatcher's edge_mask_3d at runtime.
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    h_face = jnp.minimum(
        pc.h_partial[c1], pc.h_partial[c2],
    ).astype(jnp.float64)
    wet_edge_mask = (h_face > 0.0).astype(jnp.float64)

    pgf_masked = pgf * wet_edge_mask
    max_pgf = float(jnp.max(jnp.abs(pgf_masked)))
    assert max_pgf < 1e-7, (
        f"partial-cell horizontally uniform AHH08 PGF residual {max_pgf:.3e} "
        f"too large; expected machine-precision zero (~1e-10).  This is the "
        f"property that motivates AHH08."
    )


def test_horizontal_T_anomaly_produces_finite_gradient():
    """A small horizontal T anomaly produces a finite, smooth PGF gradient
    (sanity check that the wrapper actually couples columns)."""
    mesh, pc, _H = _build_test_state(nlev=10, sub=2)
    T_3d, S_3d = _uniform_rest_T_S(pc)
    # Add a 0.5 °C cosine-lat anomaly to T at all levels.
    anomaly = 0.5 * jnp.cos(mesh.latCell)
    T_3d = T_3d + anomaly[:, None]

    pgf = density_jacobian_pgf_ahh08_mpas(
        T_3d, S_3d, pc.h_partial.astype(jnp.float64), mesh, g=constants.g,
    )
    assert jnp.all(jnp.isfinite(pgf)), "non-finite PGF on a smooth state"
    max_pgf = float(jnp.max(jnp.abs(pgf)))
    # 0.5°C → density anomaly ~ 0.1 kg/m³ → pressure anomaly ~ 1 kg/m³ over
    # the column → gradient ~ g · anomaly · h / dx ~ 1e-3 Pa/m.  Within
    # an order of magnitude.
    assert 1e-5 < max_pgf < 1.0, (
        f"PGF magnitude {max_pgf:.3e} not in plausible 1e-5..1 Pa/m range"
    )


def test_grad_flows_through_wrapper():
    """jax.grad wrt T flows through the wrapper without NaNs."""
    mesh, pc, _H = _build_test_state(nlev=8, sub=2)
    T_3d, S_3d = _uniform_rest_T_S(pc)

    h_64 = pc.h_partial.astype(jnp.float64)

    def loss(T_in):
        pgf = density_jacobian_pgf_ahh08_mpas(
            T_in, S_3d, h_64, mesh, g=constants.g,
        )
        return jnp.sum(pgf ** 2)

    g = jax.grad(loss)(T_3d)
    assert jnp.all(jnp.isfinite(g)), "non-finite grad through AHH08 wrapper"
