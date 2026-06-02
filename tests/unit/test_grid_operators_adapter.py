"""B2: a real grid, adapter-wrapped, satisfies the GridOperators contract."""

from __future__ import annotations

import jax
import jax.numpy as jnp
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
        _interp_center_to_corner,
        _pad_halo_auto,
        cgrid_divergence,
        cgrid_gradient_2d,
    )

    grid, cdgrid, ops = _cube_grid_and_ops()
    scalar = jnp.cos(grid.grid_lat) * jnp.sin(grid.grid_lon)

    # scalar-only operators
    assert jnp.array_equal(ops.halo_fill(scalar), _pad_halo_auto(scalar, cdgrid))
    assert jnp.array_equal(
        ops.interpolate(scalar, "center", "corner"),
        _interp_center_to_corner(scalar, cdgrid),
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
