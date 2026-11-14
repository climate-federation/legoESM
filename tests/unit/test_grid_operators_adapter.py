"""B2: a real grid, adapter-wrapped, satisfies the GridOperators contract."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.operator_adapters import (
    LatLonCGridOperators,
    latlon_cgrid_operators,
)
from legoesm.grids.operator_protocol import (
    GridOperators,
    validate_grid_operators,
)
from legoesm.grids.operators_latlon_cgrid import (
    curl_vertex_cgrid,
    divergence_cgrid,
    gradient_x_cgrid,
    gradient_y_cgrid,
    interp_cell_to_uface,
    pad_ns_scalar,
)


def _grid_and_ops():
    from legoesm.grids.latlon import create_latlon_grid

    grid = create_latlon_grid(16)
    return grid, LatLonCGridOperators(grid)


def test_adapter_satisfies_the_contract() -> None:
    """validate_grid_operators accepts the adapter; it is a GridOperators."""
    _grid, ops = _grid_and_ops()
    validate_grid_operators(ops)  # raises if any operator is missing/mis-arity
    assert isinstance(ops, GridOperators)  # structural (attribute presence)
    assert latlon_cgrid_operators(_grid).grid is _grid


def test_methods_delegate_byte_identically_to_free_functions() -> None:
    """Each adapter method is a thin wrapper over the shared free operator."""
    grid, ops = _grid_and_ops()
    scalar = jnp.cos(grid.grid_lat) * jnp.sin(grid.grid_lon)

    # gradient -> face-staggered (u, v); reuse them for divergence/vorticity
    gx, gy = ops.gradient(scalar)
    assert jnp.array_equal(gx, gradient_x_cgrid(scalar, grid))
    assert jnp.array_equal(gy, gradient_y_cgrid(scalar, grid))

    assert jnp.array_equal(ops.divergence(gx, gy), divergence_cgrid(gx, gy, grid))
    assert jnp.array_equal(ops.vorticity(gx, gy), curl_vertex_cgrid(gx, gy, grid))
    assert jnp.array_equal(
        ops.interpolate(scalar, "center", "uface"), interp_cell_to_uface(scalar)
    )
    assert jnp.array_equal(ops.halo_fill(scalar), pad_ns_scalar(scalar, grid))


def test_adapter_vface_interp_is_grid_aware_not_gridless() -> None:
    """Regression: the adapter's interpolate(.., 'vface') uses the grid-AWARE
    pad_ns north/south form (interp_cell_to_vface(field, grid)), which DIFFERS at the
    pole/fold from the legacy grid=None pole-inert form interp_cell_to_vface(field).

    A dycore that needs the grid=None form (e.g. the lat-lon SW mass flux) must call
    the free function directly, NOT route vface through this adapter — routing it
    changed the north-row value (7.3 vs 0.0) and would alter dh_dt on a tripolar grid.
    """
    from legoesm.grids.operators_latlon_cgrid import interp_cell_to_vface

    grid, ops = _grid_and_ops()
    field = jnp.cos(grid.grid_lat) + 5.0  # non-zero pole rows so the BC shows
    via_ops = ops.interpolate(field, "center", "vface")
    assert jnp.array_equal(via_ops, interp_cell_to_vface(field, grid))  # grid-aware
    assert not jnp.allclose(via_ops, interp_cell_to_vface(field))       # != grid=None


@pytest.mark.parametrize(
    "src,dst",
    [
        ("center", "corner"),  # unsupported destination
        ("vface", "uface"),    # non-center source must NOT silently run cell->face
        (None, "vface"),       # missing source
        ("center", None),      # missing destination
    ],
)
def test_unsupported_interpolation_stagger_raises(src, dst) -> None:
    """Both ends of the stagger pair are validated — no silent mis-location."""
    _grid, ops = _grid_and_ops()
    with pytest.raises(ValueError, match="unsupported interpolation"):
        ops.interpolate(jnp.zeros((4, 4)), src, dst)


# --- cubed-sphere C-D grid adapter (B2 replication) ---

def _cube_grid_and_ops():
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.operator_adapters import CubedSphereCDGridOperators

    grid = create_cubed_sphere(8)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    return grid, cdgrid, CubedSphereCDGridOperators(cdgrid)


def test_cube_adapter_satisfies_the_contract() -> None:
    from legoesm.grids.operator_adapters import cubed_sphere_cdgrid_operators

    grid, cdgrid, ops = _cube_grid_and_ops()
    validate_grid_operators(ops)
    assert isinstance(ops, GridOperators)
    assert cubed_sphere_cdgrid_operators(cdgrid).cdgrid is cdgrid


def test_cube_methods_delegate_byte_identically() -> None:
    from legoesm.core.operators_cdgrid import (
        interp_center_to_corner,
        pad_halo_auto,
        cgrid_divergence,
        cgrid_gradient_2d,
    )

    grid, cdgrid, ops = _cube_grid_and_ops()
    scalar = jnp.cos(grid.grid_lat) * jnp.sin(grid.grid_lon)

    # scalar-only operators
    assert jnp.array_equal(ops.halo_fill(scalar), pad_halo_auto(scalar, cdgrid))
    assert jnp.array_equal(
        ops.interpolate(scalar, "center", "corner"),
        interp_center_to_corner(scalar, cdgrid),
    )
    # gradient -> C-grid winds; reuse for divergence
    g_ad = ops.gradient(scalar)
    g_dir = cgrid_gradient_2d(scalar, cdgrid)
    for a, b in zip(jax.tree.leaves(g_ad), jax.tree.leaves(g_dir)):
        assert jnp.array_equal(a, b)
    uc, vc = g_ad
    assert jnp.array_equal(
        ops.divergence(uc, vc), cgrid_divergence(uc, vc, cdgrid)
    )


def test_cube_unsupported_interpolation_raises() -> None:
    _grid, _cdgrid, ops = _cube_grid_and_ops()
    with pytest.raises(ValueError, match="unsupported interpolation"):
        ops.interpolate(jnp.zeros((6, 8, 8)), "center", "uface")


# --- MPAS/Voronoi edge-normal (TRiSK) adapter (B2 replication) ---

def _mpas_mesh_and_ops():
    from legoesm.grids.operator_adapters import MPASEdgeOperators
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    return mesh, MPASEdgeOperators(mesh)


def _eq(a, b):
    return jnp.array_equal(a, b, equal_nan=True)


def test_mpas_edge_adapter_satisfies_edge_contract() -> None:
    from legoesm.grids.operator_adapters import mpas_edge_operators
    from legoesm.grids.operator_protocol import (
        EdgeOperators,
        validate_edge_operators,
        validate_grid_operators,
    )

    mesh, ops = _mpas_mesh_and_ops()
    validate_edge_operators(ops)            # satisfies the edge-normal contract
    assert isinstance(ops, EdgeOperators)
    assert mpas_edge_operators(mesh).mesh is mesh
    # ... and does NOT satisfy the component-velocity GridOperators contract
    # (divergence is 1-arg edge-normal, not 2-arg (u, v)).
    with pytest.raises(TypeError):
        validate_grid_operators(ops)


def test_mpas_edge_methods_delegate_byte_identically() -> None:
    from legoesm.core.operators_voronoi import (
        cell_to_edge_avg,
        curl_vertex,
        divergence_cell,
        gradient_edge,
        tangential_velocity,
    )

    mesh, ops = _mpas_mesh_and_ops()
    # smooth edge-normal velocity + cell scalar from mesh coordinates
    u_edge = jnp.sin(mesh.lonEdge) * jnp.cos(mesh.latEdge)
    phi_cell = jnp.cos(mesh.lonCell) * jnp.sin(mesh.latCell)

    assert _eq(ops.divergence(u_edge), divergence_cell(u_edge, mesh))
    assert _eq(ops.gradient(phi_cell), gradient_edge(phi_cell, mesh))
    assert _eq(ops.vorticity(u_edge), curl_vertex(u_edge, mesh))
    assert _eq(ops.tangential(u_edge), tangential_velocity(u_edge, mesh))
    assert _eq(ops.cell_to_edge(phi_cell), cell_to_edge_avg(phi_cell, mesh))


def test_mpas_edge_methods_dispatch_to_3d_for_columns() -> None:
    """A column field (nEdges/nCells, nlev) routes to the *_3d TRiSK kernels —
    the SAME adapter backs the SW (2-D) and PE/ocean (3-D) MPAS dycores."""
    from legoesm.core.operators_voronoi import (
        cell_to_edge_avg_3d,
        curl_vertex_3d,
        divergence_cell_3d,
        gradient_edge_3d,
        tangential_velocity_3d,
    )

    mesh, ops = _mpas_mesh_and_ops()
    nlev = 5
    sigma = jnp.linspace(0.1, 1.0, nlev)
    u3 = (jnp.sin(mesh.lonEdge) * jnp.cos(mesh.latEdge))[:, None] * sigma[None, :]
    phi3 = (jnp.cos(mesh.lonCell) * jnp.sin(mesh.latCell))[:, None] * sigma[None, :]

    assert ops.divergence(u3).ndim == 2  # column output, not a slice
    assert _eq(ops.divergence(u3), divergence_cell_3d(u3, mesh))
    assert _eq(ops.gradient(phi3), gradient_edge_3d(phi3, mesh))
    assert _eq(ops.vorticity(u3), curl_vertex_3d(u3, mesh))
    assert _eq(ops.tangential(u3), tangential_velocity_3d(u3, mesh))
    assert _eq(ops.cell_to_edge(phi3), cell_to_edge_avg_3d(phi3, mesh))


def test_mpas_edge_rejects_rank3_bundle() -> None:
    """A rank-3 tracer/channel bundle must be flattened by the caller — the
    column kernels broadcast assuming rank 2, so the adapter raises instead of
    silently mis-broadcasting."""
    mesh, ops = _mpas_mesh_and_ops()
    tracers = jnp.zeros((mesh.nCells, 5, 3))  # (nCells, nlev, n_tracers)
    with pytest.raises(ValueError, match="rank-3"):
        ops.gradient(tracers)
    with pytest.raises(ValueError, match="rank-3"):
        ops.cell_to_edge(tracers)


# ---------------------------------------------------------------------------
# Neighbour-sum adapters: same contracts, same operators (fp64)
# ---------------------------------------------------------------------------

def test_neighbour_adapters_satisfy_contracts_and_match() -> None:
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.operator_adapters import (
        MPASEdgeOperators,
        latlon_neighbour_operators,
        mpas_neighbour_operators,
    )
    from legoesm.grids.operator_protocol import validate_edge_operators

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        grid = create_latlon_grid(16, 32)
        new, old = latlon_neighbour_operators(grid), LatLonCGridOperators(grid)
        validate_grid_operators(new)
        rng = np.random.default_rng(0)
        nj, ni = grid.n_lat, grid.n_lon
        scalar = jnp.asarray(rng.standard_normal((nj, ni)))
        gx = jnp.asarray(rng.standard_normal((nj, ni)))
        gx = jnp.concatenate([gx, gx[:, :1]], axis=1)
        gy = jnp.asarray(rng.standard_normal((nj + 1, ni))).at[0].set(0.0).at[-1].set(0.0)

        def rel(a, b):
            return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))

        for a, b in zip(new.gradient(scalar), old.gradient(scalar)):
            assert rel(a, b) <= 1e-13
        assert rel(new.divergence(gx, gy), old.divergence(gx, gy)) <= 1e-13
        assert rel(new.vorticity(gx, gy), old.vorticity(gx, gy)) <= 1e-13

        mesh, mold = _mpas_mesh_and_ops()
        mnew = mpas_neighbour_operators(mesh)
        validate_edge_operators(mnew)
        assert isinstance(mold, MPASEdgeOperators)
        u_edge = jnp.sin(mesh.lonEdge) * jnp.cos(mesh.latEdge)
        phi_cell = jnp.cos(mesh.lonCell) * jnp.sin(mesh.latCell)
        assert _eq(mnew.divergence(u_edge), mold.divergence(u_edge))
        assert _eq(mnew.gradient(phi_cell), mold.gradient(phi_cell))
        assert _eq(mnew.vorticity(u_edge), mold.vorticity(u_edge))
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_mpas_neighbour_adapter_delegates_columns_to_parent() -> None:
    from legoesm.grids.operator_adapters import mpas_neighbour_operators

    mesh, parent = _mpas_mesh_and_ops()
    ops = mpas_neighbour_operators(mesh)
    sigma = jnp.linspace(0.1, 1.0, 4)
    u3 = (jnp.sin(mesh.lonEdge) * jnp.cos(mesh.latEdge))[:, None] * sigma[None, :]
    phi3 = (jnp.cos(mesh.lonCell) * jnp.sin(mesh.latCell))[:, None] * sigma[None, :]
    assert _eq(ops.divergence(u3), parent.divergence(u3))
    assert _eq(ops.gradient(phi3), parent.gradient(phi3))
    assert _eq(ops.vorticity(u3), parent.vorticity(u3))
    with pytest.raises(ValueError):
        ops.divergence(u3[..., None])


def test_latlon_neighbour_adapter_delegates_levels_to_parent() -> None:
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.operator_adapters import latlon_neighbour_operators

    grid = create_latlon_grid(16, 32)
    ops, parent = latlon_neighbour_operators(grid), LatLonCGridOperators(grid)
    rng = np.random.default_rng(1)
    s3 = jnp.asarray(rng.standard_normal((16, 32, 3)))
    u3 = jnp.asarray(rng.standard_normal((16, 33, 3)))
    v3 = jnp.asarray(rng.standard_normal((17, 32, 3)))
    for a, b in zip(ops.gradient(s3), parent.gradient(s3)):
        assert _eq(a, b)
    assert _eq(ops.divergence(u3, v3), parent.divergence(u3, v3))
    assert _eq(ops.vorticity(u3, v3), parent.vorticity(u3, v3))
