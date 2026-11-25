"""Column-lane atmosphere <-> lat-lon 3-D ocean: the remap is a reshape.

The lat-lon column lane's mesh is the lat-lon grid's cells raveled row-major;
make_grid_remapper maps the two by one weight of 1.0 per cell.  Tiny grids,
no model, login-trivial.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.atmosphere.dynamics.gcm.latlon_cgrid_column import (
    build_latlon_column_mesh,
)
from legoesm.coupler.grid_remap import (
    attach_wet_masks, make_grid_remapper, remap_field, remap_surface_fields,
)
from legoesm.grids.latlon import create_latlon_grid

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def grids():
    g = create_latlon_grid(6, 12)
    return build_latlon_column_mesh(g), g


def test_reshape_round_trip_is_exact(grids):
    mesh, g = grids
    rem = make_grid_remapper(mesh, g)
    assert rem.reshape and not rem.identity
    x = jnp.asarray(np.random.default_rng(0).normal(size=mesh.nCells))
    on_ocean = remap_field(x, rem.a2o)
    np.testing.assert_array_equal(on_ocean, x.reshape(g.n_lat, g.n_lon))
    np.testing.assert_array_equal(remap_field(on_ocean, rem.o2a), x)
    # leading axes (levels) and the pytree path go through the same map
    x3 = jnp.stack([x, 2 * x])
    np.testing.assert_array_equal(
        remap_field(x3, rem.a2o), x3.reshape(2, g.n_lat, g.n_lon))
    out = remap_surface_fields({"f": x}, rem.a2o)
    np.testing.assert_array_equal(out["f"], x.reshape(g.n_lat, g.n_lon))


def test_reshape_conserves_the_exchanged_flux_integral(grids):
    mesh, g = grids
    rem = make_grid_remapper(mesh, g)
    flux = jnp.asarray(np.random.default_rng(1).normal(size=mesh.nCells))
    # every cell's flux x area is the same number on both sides, so the area
    # integral is equal in any one summation order
    atm = flux * mesh.areaCell
    ocn = remap_field(flux, rem.a2o) * g.grid_area
    np.testing.assert_array_equal(ocn.reshape(-1), atm)
    assert float(jnp.sum(atm)) == float(jnp.sum(ocn.reshape(-1)))


def test_gradient_through_the_reshape(grids):
    mesh, g = grids
    rem = make_grid_remapper(mesh, g)
    w = jnp.asarray(g.grid_area)

    def loss(x):
        y = remap_field(x, rem.a2o)
        return jnp.sum(jnp.sin(y) * w) + jnp.sum(remap_field(y ** 2, rem.o2a))

    x = jnp.asarray(np.random.default_rng(2).normal(size=mesh.nCells))
    check_grads(loss, (x,), order=2, modes=("fwd", "rev"))
    assert float(jnp.max(jnp.abs(jax.grad(loss)(x)))) > 0.0


def test_wet_mask_rebake_leaves_the_reshape_unchanged(grids):
    mesh, g = grids
    rem = make_grid_remapper(mesh, g)
    wet = np.ones((g.n_lat, g.n_lon))
    wet[0] = 0.0
    assert attach_wet_masks(rem, mesh, g, wet) is rem


def test_a_mesh_that_is_not_the_ravel_is_not_a_reshape(grids):
    mesh, g = grids
    perm = np.random.default_rng(3).permutation(mesh.nCells)
    shuffled = mesh._replace(latCell=mesh.latCell[perm], lonCell=mesh.lonCell[perm])
    with pytest.raises(NotImplementedError):
        make_grid_remapper(shuffled, g)
    with pytest.raises(NotImplementedError):
        make_grid_remapper(mesh._replace(areaCell=mesh.areaCell * 1.001), g)
