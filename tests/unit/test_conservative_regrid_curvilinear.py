"""Conservative, differentiable regular-lat-lon <-> tripole remap
(fine-quadrature first-order; ``conservative_regrid_curvilinear``).

Gates the Phase-2 cross-grid coupling for a regular lat-lon atmosphere and a
curvilinear tripole ocean:
  * partition of unity (weights sum to 1 per destination cell) — exact;
  * constant-field preservation — machine-exact at any n_sub;
  * global-integral conservation of the atm->ocean (regular->tripole) flux
    direction — first-order, convergent in n_sub;
  * differentiability of the apply (jax.grad flows through the remap);
  * the coupler dispatch (make_grid_remapper) routes regular-atm + tripole-ocean
    to the curvilinear remap and still raises for unsupported pairs.

Login-trivial sizes, but builds grids + JAX -> run on a compute node (sbatch).
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _grids(n_lat_atm=12, n_lat_trip=18):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.tripole import create_synthetic_tripole
    atm = create_latlon_grid(n_lat_atm)                 # regular LatLonGrid
    trip = create_synthetic_tripole(n_lat=n_lat_trip)   # active-fold C-grid
    return atm, trip


class TestCurvilinearRemap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.atm, cls.trip = _grids()

    def _a2o(self, n_sub=6):
        from legoesm.grids.conservative_regrid_curvilinear import (
            make_regular_to_curvilinear_weights,
        )
        return make_regular_to_curvilinear_weights(self.atm, self.trip, n_sub=n_sub)

    def _o2a(self, n_sub=6):
        from legoesm.grids.conservative_regrid_curvilinear import (
            make_curvilinear_to_regular_weights,
        )
        return make_curvilinear_to_regular_weights(self.trip, self.atm, n_sub=n_sub)

    def _weight_sum_per_dst(self, w):
        ws = np.asarray(w.weights)
        dst = np.asarray(w.dst_idx_flat)
        n = int(w.n_dst_cells)
        s = np.zeros(n)
        np.add.at(s, dst, ws)
        return s

    def test_partition_of_unity_a2o(self):
        """FLUX direction (atm -> tripole) sums to 1 per tripole cell — constant
        preserved, no spurious flux gradients (robust to eORCA degenerate cells;
        true-area_T normalisation would blow up there)."""
        w = self._a2o()
        s = self._weight_sum_per_dst(w)
        covered = s > 0
        self.assertTrue(covered.all(), "some tripole cell received no source")
        np.testing.assert_allclose(s[covered], 1.0, atol=1e-12)

    def test_global_flux_conservation_a2o_first_order(self):
        """FLUX direction conserves the global area-weighted integral the OCEAN
        budget uses (sum_d F[d]*area_T[d] ~ sum_s F_atm[s]*area_atm[s]) to FIRST
        ORDER — the residual is the small nearest-centre vs true-area_T
        discrepancy (zero for a uniform flux)."""
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        w = self._a2o()
        lat = np.asarray(self.atm.lat); lon = np.asarray(self.atm.lon)
        LON, LAT = np.meshgrid(lon, lat)
        F_atm = (2.0 + np.sin(LAT) * np.cos(LON)).astype(np.float64)
        F_oce = np.asarray(apply_conservative_regrid(jnp.asarray(F_atm), w))
        area_atm = np.asarray(self.atm.area)
        area_T = np.asarray(self.trip.area_T)
        lhs = float((F_oce * area_T).sum())
        rhs = float((F_atm * area_atm).sum())
        rel = abs(lhs - rhs) / abs(rhs)
        self.assertLess(rel, 0.05, f"a2o global flux conservation poor (rel={rel})")

    def test_a2o_full_coverage_when_tripole_finer(self):
        """The realistic case: a COARSE atm source and a FINER tripole ocean.
        Every tripole cell must receive flux (the coverage fallback samples any
        cell the source-tiling misses) — no zero-flux cold-spots."""
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.grids.tripole import create_synthetic_tripole
        from legoesm.grids.conservative_regrid_curvilinear import (
            make_regular_to_curvilinear_weights,
        )
        atm = create_latlon_grid(8)                  # coarse: 8x16
        trip = create_synthetic_tripole(n_lat=40)    # fine: 40x80
        w = make_regular_to_curvilinear_weights(atm, trip)
        dst = np.asarray(w.dst_idx_flat)
        covered = np.zeros(int(w.n_dst_cells), dtype=bool)
        covered[dst] = True
        self.assertTrue(covered.all(),
                        f"{(~covered).sum()} tripole cells got no flux")

    def test_partition_of_unity_o2a(self):
        """STATE direction (tripole -> atm) sums to 1 per atm cell (constant
        preserved exactly — the right property for interpolating SST)."""
        w = self._o2a()
        s = self._weight_sum_per_dst(w)
        covered = s > 0
        self.assertTrue(covered.all())
        np.testing.assert_allclose(s[covered], 1.0, atol=1e-12)

    def test_constant_field_preserved_o2a(self):
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        w_o2a = self._o2a()
        src2 = jnp.full(w_o2a.src_shape, -1.25, dtype=jnp.float64)
        out2 = apply_conservative_regrid(src2, w_o2a)
        self.assertEqual(tuple(out2.shape), tuple(w_o2a.dst_shape))
        np.testing.assert_allclose(np.asarray(out2), -1.25, atol=1e-12)

    def test_a2o_constant_flux_preserved(self):
        """Partition-of-unity => a uniform atm flux maps to the SAME uniform
        flux on every tripole cell (exact; no spurious gradients)."""
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        w = self._a2o()
        out = np.asarray(apply_conservative_regrid(
            jnp.full(w.src_shape, 3.7, dtype=jnp.float64), w))
        np.testing.assert_allclose(out, 3.7, atol=1e-10)

    def test_a2o_converges_and_bounded(self):
        """Regular->tripole converges as the quadrature refines (n_sub 12 close
        to 6) AND stays within the source range (partition-of-unity => weighted
        average => no overshoot)."""
        from legoesm.grids.conservative_regrid import apply_conservative_regrid

        lat = np.asarray(self.atm.lat)
        lon = np.asarray(self.atm.lon)
        LON, LAT = np.meshgrid(lon, lat)
        field = (1.0 + 0.5 * np.sin(LAT) * np.cos(LON)).astype(np.float64)

        w6 = self._a2o(n_sub=6)
        w12 = self._a2o(n_sub=12)
        out6 = np.asarray(apply_conservative_regrid(jnp.asarray(field), w6))
        out12 = np.asarray(apply_conservative_regrid(jnp.asarray(field), w12))
        rel = np.abs(out12 - out6).max() / (np.abs(field).max())
        self.assertLess(rel, 0.05, f"a2o not converging in n_sub (rel={rel})")
        # Weighted average => bounded by the source extremes.
        self.assertGreaterEqual(out12.min(), field.min() - 1e-9)
        self.assertLessEqual(out12.max(), field.max() + 1e-9)

    def test_grad_flows_through_remap(self):
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        w = self._o2a()

        def loss(src):
            return jnp.sum(apply_conservative_regrid(src, w) ** 2)

        src0 = jnp.ones(w.src_shape) * 0.3
        g = jax.grad(loss)(src0)
        self.assertTrue(np.all(np.isfinite(np.asarray(g))))
        self.assertFalse(jnp.allclose(g, 0.0))

    def test_dispatch_regular_atm_tripole_ocean(self):
        from legoesm.coupler.grid_remap import make_grid_remapper
        rem = make_grid_remapper(self.atm, self.trip)
        self.assertFalse(rem.identity)
        self.assertIsNotNone(rem.a2o)
        self.assertIsNotNone(rem.o2a)
        # a2o maps atm(src) -> ocean(dst); o2a the reverse.
        self.assertEqual(tuple(rem.a2o.src_shape),
                         (int(self.atm.n_lat), int(self.atm.n_lon)))
        self.assertEqual(tuple(rem.a2o.dst_shape),
                         (int(self.trip.n_lat), int(self.trip.n_lon)))

    def test_dispatch_rejects_tripole_tripole(self):
        from legoesm.coupler.grid_remap import make_curvilinear_latlon_remapper
        with self.assertRaises(NotImplementedError):
            make_curvilinear_latlon_remapper(self.trip, self.trip)


if __name__ == "__main__":
    unittest.main()
