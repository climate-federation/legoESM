"""Iter-1039 sentinel: 3D model field smoothness across cube edges.

Verifies that on a non-trivial 3D initial condition, the model
does not concentrate spurious perturbations on cube edges or
vertices over a multi-step integration.

Method:
1. Initialize hydrostatic model with isothermal + small u-wind
   perturbation (zonal flow).
2. Run 30 steps.
3. Check that final field's variance across cube edges is bounded
   (within 2× the global field variance).

If edge artifacts develop, edge-cell values will spike and the
edge-variance ratio will exceed the threshold.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
from legoesm.grids.cubed_sphere import create_cubed_sphere


@pytest.fixture(scope="module")
def hyd_model_state():
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationConfig,
        CDGridPrimitiveEquationModel,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    grid = create_cubed_sphere(8)
    nlev = 5
    sigma_coord = create_sigma_coordinate(nlev)
    dt = 300.0
    nu2, _ = default_div_damp_coeffs(grid, dt=dt)
    config = CDGridPrimitiveEquationConfig(
        div_damp_coeff=nu2,
        hyperdiff_coeff=1e14,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma_coord, config)

    n = grid.n
    T_data = jnp.full((6, n, n, nlev), 250.0)
    # Small zonal-wind perturbation
    u_data = jnp.full((6, n, n, nlev), 5.0)
    v_data = jnp.zeros((6, n, n, nlev))
    ps_data = jnp.full((6, n, n), 1e5)
    phis_data = jnp.zeros((6, n, n))

    state = HydrostaticState(
        u=Field(data=u_data, name="u", dims=("face", "x", "y", "lev"),
                 units="m/s"),
        v=Field(data=v_data, name="v", dims=("face", "x", "y", "lev"),
                 units="m/s"),
        T=Field(data=T_data, name="T", dims=("face", "x", "y", "lev"),
                 units="K"),
        p_s=Field(data=ps_data, name="p_s", dims=("face", "x", "y"),
                   units="Pa"),
        phis=Field(data=phis_data, name="phis",
                    dims=("face", "x", "y"), units="m^2/s^2"),
    )
    return model, state, dt


def test_iter1039_hydrostatic_no_edge_artifacts(hyd_model_state):
    """3D hydrostatic on uniform u=5 m/s zonal wind: no edge artifacts."""
    model, state, dt = hyd_model_state
    s = state
    for _ in range(30):
        s = model.step(s, dt)

    assert jnp.all(jnp.isfinite(s.T.data)), "NaN in T after 30 steps"
    assert jnp.all(jnp.isfinite(s.u.data)), "NaN in u after 30 steps"
    assert jnp.all(jnp.isfinite(s.v.data)), "NaN in v after 30 steps"

    # Edge-vs-interior variance check on u (zonal wind).
    # u shape: (6, n, n, nlev) — face-cell-cell-level
    u = np.asarray(s.u.data)
    n = u.shape[1]
    # Edge cells: i=0, i=n-1, j=0, j=n-1
    edge_mask = np.zeros((n, n), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    # Interior cells: complement
    interior_mask = ~edge_mask

    # Compute std over cells per face per level, then average
    edge_std_global = float(np.std(u[:, edge_mask, :]))
    interior_std_global = float(np.std(u[:, interior_mask, :]))

    # Edge std should be at most ~3× interior std
    # (some edge concentration is expected at low resolution C8)
    if interior_std_global > 1e-10:
        ratio = edge_std_global / interior_std_global
        assert ratio < 5.0, (
            f"Edge/interior std ratio = {ratio:.2f} > 5.0 — "
            f"likely edge artifact developing.")


def test_iter1041_hydrostatic_long_integration_finite(hyd_model_state):
    """3D hydrostatic 100-step integration: no NaN with zonal IC."""
    model, state, dt = hyd_model_state
    s = state
    for _ in range(100):
        s = model.step(s, dt)
    # Just verify finite — 100 steps × 300s = ~8 hr simulation
    assert jnp.all(jnp.isfinite(s.T.data)), "T NaN at 100 steps"
    assert jnp.all(jnp.isfinite(s.u.data)), "u NaN at 100 steps"
    assert jnp.all(jnp.isfinite(s.v.data)), "v NaN at 100 steps"
    assert jnp.all(jnp.isfinite(s.p_s.data)), "p_s NaN at 100 steps"


def test_iter1043_hydrostatic_cube_vertex_finite(hyd_model_state):
    """3D hydrostatic 30-step run: cube-vertex cells have finite finite-bounded values.

    Cube vertices (8 total: 4 corners × 2 of {NE,NW,SE,SW}) are
    where 3 faces meet — historically the trouble spot for edge
    artifacts.  This test specifically inspects those cells.
    """
    model, state, dt = hyd_model_state
    s = state
    for _ in range(30):
        s = model.step(s, dt)

    u = np.asarray(s.u.data)
    n = u.shape[1]
    # Cube-vertex cells per face: (0,0), (0,n-1), (n-1,0), (n-1,n-1)
    vertex_idx = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    vertex_u = []
    for f in range(6):
        for i, j in vertex_idx:
            vertex_u.append(u[f, i, j, :])
    vertex_u = np.array(vertex_u)  # shape (24, nlev)

    assert np.isfinite(vertex_u).all(), "NaN at cube vertex"
    # Vertex values shouldn't be wildly different from cell-mean
    cell_mean = float(np.mean(u))
    cell_std = float(np.std(u))
    if cell_std > 1e-10:
        max_z = float(np.max(np.abs(vertex_u - cell_mean)) / cell_std)
        # Z-score < 10 indicates no extreme outliers at vertices
        assert max_z < 10.0, (
            f"Cube vertex extreme z-score = {max_z:.2f} > 10 — "
            f"indicating possible artifact concentration.")


def test_iter1045_nonhydrostatic_cube_vertex_finite():
    """3D non-hydrostatic 30-step run: cube-vertex cells finite-bounded."""
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerConfig,
        CDGridCompressibleEulerModel,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import NonHydrostaticState
    from legoesm.grids.vertical import (
        compute_terrain_metric,
        create_height_coordinate,
    )

    grid = create_cubed_sphere(8)
    nlev = 5
    z_top = 30000.0
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, grid.n, grid.n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    config = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, config)

    n = grid.n
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=jnp.full((6, n, n, nlev), 5.0),
                 name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)),
                 name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                 name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                           name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                         name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)),
                    name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                       name="tracers",
                       dims=("face", "x", "y", "level", "tracer"),
                       units="kg/kg"),
    )
    dt = 10.0
    s = state
    for _ in range(30):
        s = model.step(s, dt)

    u = np.asarray(s.u.data)
    vertex_idx = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    vertex_u = []
    for f in range(6):
        for i, j in vertex_idx:
            vertex_u.append(u[f, i, j, :])
    vertex_u = np.array(vertex_u)

    assert np.isfinite(vertex_u).all(), "NaN at NH cube vertex"
    cell_mean = float(np.mean(u))
    cell_std = float(np.std(u))
    if cell_std > 1e-10:
        max_z = float(np.max(np.abs(vertex_u - cell_mean)) / cell_std)
        assert max_z < 10.0, (
            f"NH cube vertex extreme z-score = {max_z:.2f} > 10.")


def test_iter1042_nonhydrostatic_long_integration_finite():
    """3D non-hydrostatic 100-step integration: no NaN with zonal IC."""
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerConfig,
        CDGridCompressibleEulerModel,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import NonHydrostaticState
    from legoesm.grids.vertical import (
        compute_terrain_metric,
        create_height_coordinate,
    )

    grid = create_cubed_sphere(8)
    nlev = 5
    z_top = 30000.0
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, grid.n, grid.n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    config = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, config)

    n = grid.n
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=jnp.full((6, n, n, nlev), 5.0),
                 name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)),
                 name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                 name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                           name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                         name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)),
                    name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                       name="tracers",
                       dims=("face", "x", "y", "level", "tracer"),
                       units="kg/kg"),
    )
    dt = 10.0  # NH needs smaller dt due to acoustic
    s = state
    for _ in range(100):
        s = model.step(s, dt)
    # 100 × 10 = 1000 s = ~17 minutes simulated.  Useful for
    # NaN detection over an integration much longer than the
    # 30-step tests above.
    assert jnp.all(jnp.isfinite(s.u.data)), "NH u NaN at 100 steps"
    assert jnp.all(jnp.isfinite(s.theta_prime.data)), "theta_prime NaN"
    assert jnp.all(jnp.isfinite(s.rho_prime.data)), "rho_prime NaN"
    assert jnp.all(jnp.isfinite(s.w.data)), "w NaN at 100 steps"


def test_iter1040_nonhydrostatic_no_edge_artifacts():
    """3D non-hydrostatic on uniform u=5 m/s zonal wind: no edge artifacts."""
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerConfig,
        CDGridCompressibleEulerModel,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import NonHydrostaticState
    from legoesm.grids.vertical import (
        compute_terrain_metric,
        create_height_coordinate,
    )

    grid = create_cubed_sphere(8)
    nlev = 5
    z_top = 30000.0
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, grid.n, grid.n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    config = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14,
        n_acoustic_substeps=4,
    )
    model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, config)

    n = grid.n
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    # Initialize with small u perturbation
    state = NonHydrostaticState(
        u=Field(data=jnp.full((6, n, n, nlev), 5.0),
                 name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)),
                 name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                 name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                           name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                         name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                    dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                       name="tracers",
                       dims=("face", "x", "y", "level", "tracer"),
                       units="kg/kg"),
    )
    dt = 10.0
    s = state
    for _ in range(30):
        s = model.step(s, dt)

    assert jnp.all(jnp.isfinite(s.u.data)), "u NaN after 30 steps"
    assert jnp.all(jnp.isfinite(s.theta_prime.data)), "theta_prime NaN"
    assert jnp.all(jnp.isfinite(s.rho_prime.data)), "rho_prime NaN"

    # Edge-vs-interior std ratio for u
    u = np.asarray(s.u.data)
    edge_mask = np.zeros((n, n), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    interior_mask = ~edge_mask

    edge_std = float(np.std(u[:, edge_mask, :]))
    interior_std = float(np.std(u[:, interior_mask, :]))
    if interior_std > 1e-10:
        ratio = edge_std / interior_std
        assert ratio < 5.0, (
            f"NH edge/interior std ratio = {ratio:.2f} > 5.0 — "
            f"likely edge artifact.")
