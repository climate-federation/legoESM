"""B2: a real grid, adapter-wrapped, satisfies the GridOperators contract."""

from __future__ import annotations

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
