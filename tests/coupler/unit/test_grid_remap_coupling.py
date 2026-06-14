"""Differentiable atm<->ocean grid coupling (coupler.grid_remap).

Proves the model can mix-and-match a different atmosphere grid with a different
ocean grid through a remap that is (a) DIFFERENTIABLE (jax.grad flows through the
ocean->atm remap back to the ocean field) and (b) CONSERVATIVE for the
atm->ocean flux direction (global area-integral preserved), plus the same-grid
identity short-circuit.  Login-trivial: tiny grids, no model spin-up, < 5 s.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _kernel_cell_area(grid):
    """Cell area as the conservative kernel defines it: (sin-band) x dlon.

    NOT grid.area — that carries a ~79 ppm metric correction that the overlap
    weights deliberately omit, so the conservation identity is exact only
    against this sin-band area.
    """
    from legoesm.grids.conservative_regrid import cell_edges_1d
    lat_e = np.asarray(grid.lat_v, dtype=np.float64)               # (n_lat+1,)
    lon_e = cell_edges_1d(np.asarray(grid.lon), periodic_lon=True)  # (n_lon+1,)
    lat_band = np.sin(lat_e[1:]) - np.sin(lat_e[:-1])              # (n_lat,)
    lon_band = lon_e[1:] - lon_e[:-1]                              # (n_lon,)
    return lat_band[:, None] * lon_band[None, :]                   # (n_lat, n_lon)


class TestDifferentiableGridRemap(unittest.TestCase):
    def setUp(self):
        from legoesm.grids.latlon import create_latlon_grid
        # Two DIFFERENT regular lat-lon resolutions, both global.
        self.atm = create_latlon_grid(n_lat=18, n_lon=36)   # 10 deg
        self.ocean = create_latlon_grid(n_lat=12, n_lon=24)  # 15 deg

    def test_remapper_shapes(self):
        from legoesm.coupler.grid_remap import make_grid_remapper, remap_field
        gr = make_grid_remapper(self.atm, self.ocean)
        self.assertFalse(gr.identity)
        sst_ocean = jnp.asarray(np.random.RandomState(0).rand(12, 24))
        sst_atm = remap_field(sst_ocean, gr.o2a)
        self.assertEqual(sst_atm.shape, (18, 36))
        flux_atm = jnp.asarray(np.random.RandomState(1).rand(18, 36))
        flux_ocean = remap_field(flux_atm, gr.a2o)
        self.assertEqual(flux_ocean.shape, (12, 24))

    def test_identity_when_same_grid(self):
        """atm_grid is ocean_grid -> identity remapper, remap is pass-through."""
        from legoesm.coupler.grid_remap import make_grid_remapper, remap_field
        gr = make_grid_remapper(self.atm, self.atm)
        self.assertTrue(gr.identity)
        self.assertIsNone(gr.a2o)
        self.assertIsNone(gr.o2a)
        f = jnp.asarray(np.random.RandomState(2).rand(18, 36))
        # remap_field with None weights is an exact pass-through (byte-identical).
        self.assertTrue(jnp.array_equal(remap_field(f, gr.o2a), f))

    def test_differentiable_ocean_to_atm(self):
        """jax.grad of an atm-grid loss flows back to the ocean SST field."""
        from legoesm.coupler.grid_remap import make_grid_remapper, remap_field
        gr = make_grid_remapper(self.atm, self.ocean)
        sst_ocean = jnp.asarray(np.random.RandomState(3).rand(12, 24))

        def loss(sst):
            return jnp.sum(remap_field(sst, gr.o2a) ** 2)

        g = jax.grad(loss)(sst_ocean)
        self.assertEqual(g.shape, sst_ocean.shape)
        self.assertTrue(jnp.all(jnp.isfinite(g)))
        self.assertFalse(jnp.allclose(g, 0.0))
        # JIT-clean (no host callbacks / data-dependent control flow in apply).
        gj = jax.jit(jax.grad(loss))(sst_ocean)
        self.assertTrue(jnp.allclose(g, gj))

    def test_flux_conservation_atm_to_ocean(self):
        """Global area-integral of a flux is preserved by the atm->ocean remap."""
        from legoesm.coupler.grid_remap import make_grid_remapper, remap_field
        gr = make_grid_remapper(self.atm, self.ocean)
        flux_atm = jnp.asarray(np.random.RandomState(4).rand(18, 36))
        flux_ocean = remap_field(flux_atm, gr.a2o)
        int_atm = float(jnp.sum(flux_atm * jnp.asarray(_kernel_cell_area(self.atm))))
        int_ocean = float(jnp.sum(flux_ocean * jnp.asarray(_kernel_cell_area(self.ocean))))
        self.assertAlmostEqual(int_atm, int_ocean, delta=abs(int_atm) * 1e-10)

    def test_constant_field_preserved(self):
        """A constant flux maps to the same constant (weights sum to 1 per cell)."""
        from legoesm.coupler.grid_remap import make_grid_remapper, remap_field
        gr = make_grid_remapper(self.atm, self.ocean)
        const = jnp.full((18, 36), 3.5)
        out = remap_field(const, gr.a2o)
        self.assertTrue(jnp.allclose(out, 3.5, atol=1e-10))

    def test_structurally_equal_latlon_is_identity(self):
        """Two independently-built identical lat-lon grids -> identity (no remap)."""
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.coupler.grid_remap import make_grid_remapper
        a = create_latlon_grid(n_lat=18, n_lon=36)
        b = create_latlon_grid(n_lat=18, n_lon=36)
        self.assertTrue(make_grid_remapper(a, b).identity)


class _FakeNonLatLonGrid:
    """Duck-typed grid lacking lat-lon geometry (stands in for cube/MPAS)."""
    def __init__(self, shape):
        self.grid_shape_2d = shape


class TestGridRemapperDispatch(unittest.TestCase):
    """The dispatcher couples same-family grids directly and never silently
    routes through an intermediate lat-lon grid."""

    def test_cross_family_raises_not_implemented(self):
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.coupler.grid_remap import make_grid_remapper
        atm = _FakeNonLatLonGrid((6, 4, 4))           # cube-like
        ocean = create_latlon_grid(n_lat=8, n_lon=16)  # lat-lon
        with self.assertRaises(NotImplementedError) as cm:
            make_grid_remapper(atm, ocean)
        # No silent lat-lon fallback: it must explicitly refuse.
        self.assertIn("cross-family", str(cm.exception).lower())

    def test_same_family_nonlatlon_raises_direct_remap(self):
        from legoesm.coupler.grid_remap import make_grid_remapper
        atm = _FakeNonLatLonGrid((6, 8, 8))
        ocean = _FakeNonLatLonGrid((6, 4, 4))
        with self.assertRaises(NotImplementedError) as cm:
            make_grid_remapper(atm, ocean)
        msg = str(cm.exception).lower()
        self.assertIn("direct", msg)
        self.assertIn("lat-lon", msg)  # message warns NOT to route through lat-lon


class TestRemapSurfaceFields(unittest.TestCase):
    """remap_surface_fields remaps 2-D surface leaves, passes others through,
    and is differentiable."""

    def setUp(self):
        from legoesm.grids.latlon import create_latlon_grid
        self.atm = create_latlon_grid(n_lat=18, n_lon=36)
        self.ocean = create_latlon_grid(n_lat=12, n_lon=24)

    def test_remaps_2d_passes_through_scalars(self):
        from legoesm.coupler.grid_remap import make_grid_remapper, remap_surface_fields
        gr = make_grid_remapper(self.atm, self.ocean)
        obj = {
            "sw_down": jnp.asarray(np.random.RandomState(5).rand(18, 36)),
            "co2_ppmv": jnp.asarray(415.0),       # scalar -> pass through
            "profile": jnp.asarray(np.arange(20.0)),  # 1-D -> pass through
        }
        out = remap_surface_fields(obj, gr.a2o)
        self.assertEqual(out["sw_down"].shape, (12, 24))
        self.assertTrue(jnp.array_equal(out["co2_ppmv"], obj["co2_ppmv"]))
        self.assertTrue(jnp.array_equal(out["profile"], obj["profile"]))

    def test_identity_pass_through(self):
        from legoesm.coupler.grid_remap import remap_surface_fields
        obj = {"f": jnp.asarray(np.random.RandomState(6).rand(18, 36))}
        out = remap_surface_fields(obj, None)  # identity remapper
        self.assertTrue(jnp.array_equal(out["f"], obj["f"]))

    def test_differentiable(self):
        from legoesm.coupler.grid_remap import make_grid_remapper, remap_surface_fields
        gr = make_grid_remapper(self.atm, self.ocean)
        field = jnp.asarray(np.random.RandomState(7).rand(18, 36))

        def loss(f):
            out = remap_surface_fields({"f": f}, gr.a2o)
            return jnp.sum(out["f"] ** 2)

        g = jax.grad(loss)(field)
        self.assertEqual(g.shape, field.shape)
        self.assertTrue(jnp.all(jnp.isfinite(g)))
        self.assertFalse(jnp.allclose(g, 0.0))


if __name__ == "__main__":
    unittest.main()
