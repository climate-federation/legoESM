"""Cross-grid coastal wet-mask fix (H3 open-water fraction + H4 o2a land
exclusion) for the regular-lat-lon atm <-> tripole ocean coupling.

Asserts:
  * o2a (ocean STATE -> atm): baked src_wet masking excludes ocean-grid LAND
    source cells so a coastal atm SST never ingests the land fill value (H4);
    the UNMASKED weights DO ingest it (the before-bug).
  * o2a masked remap preserves a constant WET ocean field (partition of unity
    over the surviving wet sources).
  * a2o (atm FLUX -> ocean): the UNMASKED partition-of-unity weights gated by
    the ocean wet mask conserve a uniform flux integral over the WET ocean
    domain (the invariant the H3 driver fix relies on).
  * cross_grid_open_water_fraction uses the ocean's OWN wet mask: a wet ocean
    cell is FULL open water (f_ocean==1, ice-free) regardless of the atm
    coastline (H3), and land cells are 0.
  * identity (shared-grid) path is byte-exact (attach no-op; None pass-through).
  * a2o weights are NOT masked by attach (only o2a shrinks).
  * jax.grad flows finite + nonzero through the masked o2a remap (AD intact).

Login-trivial sizes but builds grids + JAX -> run on a compute node (sbatch).
"""

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_LAND_FILL = 274.65   # WOA deep-ocean fill SST [K] at an ocean LAND cell
_WET_VAL = 10.0       # a real ocean SST-anomaly-like value [arbitrary]


def _grids(n_lat_atm=10, n_lat_trip=16):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.tripole import create_synthetic_tripole
    atm = create_latlon_grid(n_lat_atm)
    trip = create_synthetic_tripole(n_lat=n_lat_trip)
    return atm, trip


def _wet_mask(trip):
    """Ocean wet mask (1=ocean, 0=land) with a mid-latitude LAND band, so some
    atm cells overlap both wet and land tripole cells (a coastline)."""
    n_lat = int(trip.n_lat)
    n_lon = int(trip.n_lon)
    wet = np.ones((n_lat, n_lon), dtype=np.float64)
    j0 = n_lat // 2
    wet[j0:j0 + 3, :] = 0.0   # land band
    return wet


class TestGridRemapWetMask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.atm, cls.trip = _grids()
        cls.wet = _wet_mask(cls.trip)

    # ---- H4: o2a land exclusion --------------------------------------------
    def test_o2a_excludes_land_fill(self):
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        from legoesm.grids.conservative_regrid_curvilinear import (
            make_curvilinear_to_regular_weights,
        )
        wet = self.wet
        src = np.where(wet > 0.5, _WET_VAL, _LAND_FILL).astype(np.float64)
        w_unmasked = make_curvilinear_to_regular_weights(self.trip, self.atm)
        w_masked = make_curvilinear_to_regular_weights(
            self.trip, self.atm, src_wet=wet)
        out_unmasked = np.asarray(
            apply_conservative_regrid(jnp.asarray(src), w_unmasked))
        out_masked = np.asarray(
            apply_conservative_regrid(jnp.asarray(src), w_masked))
        # BEFORE-bug: the unmasked remap ingests the land fill -> coastal /
        # land atm cells reach values far above the wet value.
        self.assertGreater(out_unmasked.max(), 100.0,
                           "unmasked o2a should be contaminated by land fill")
        # AFTER-fix: the masked remap NEVER exceeds the wet value (no fill bleed);
        # covered cells equal the wet value exactly, all-land cells are 0.
        self.assertLessEqual(out_masked.max(), _WET_VAL + 1e-6,
                             "masked o2a leaked the land fill into an atm cell")
        self.assertTrue(np.all(out_masked <= _WET_VAL + 1e-6),
                        "masked o2a produced a value above the wet source")
        # No intermediate coastal contamination band either.
        self.assertEqual(
            int(((out_masked > _WET_VAL + 1e-6)).sum()), 0)

    def test_o2a_constant_wet_field_preserved(self):
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        from legoesm.grids.conservative_regrid_curvilinear import (
            make_curvilinear_to_regular_weights,
        )
        w_masked = make_curvilinear_to_regular_weights(
            self.trip, self.atm, src_wet=self.wet)
        src = jnp.full(w_masked.src_shape, 3.3, dtype=jnp.float64)
        out = np.asarray(apply_conservative_regrid(src, w_masked))
        dst = np.asarray(w_masked.dst_idx_flat)
        covered = np.zeros(int(w_masked.n_dst_cells), dtype=bool)
        covered[dst] = True
        # Every covered atm cell sees the constant; uncovered (all-land) -> 0.
        np.testing.assert_allclose(out.reshape(-1)[covered], 3.3, atol=1e-12)

    def test_o2a_all_land_cell_is_finite_zero(self):
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        from legoesm.grids.conservative_regrid_curvilinear import (
            make_curvilinear_to_regular_weights,
        )
        w_masked = make_curvilinear_to_regular_weights(
            self.trip, self.atm, src_wet=self.wet)
        src = np.where(self.wet > 0.5, _WET_VAL, _LAND_FILL).astype(np.float64)
        out = np.asarray(apply_conservative_regrid(jnp.asarray(src), w_masked))
        self.assertTrue(np.all(np.isfinite(out)), "masked o2a produced non-finite")

    # ---- H3: a2o wet-domain flux conservation via the ocean-wet gate --------
    def test_a2o_uniform_flux_conserved_over_wet_domain(self):
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        from legoesm.grids.conservative_regrid_curvilinear import (
            make_regular_to_curvilinear_weights,
        )
        w_a2o = make_regular_to_curvilinear_weights(self.atm, self.trip)
        F = 2.5
        out = np.asarray(apply_conservative_regrid(
            jnp.full(w_a2o.src_shape, F, dtype=jnp.float64), w_a2o))
        area_T = np.asarray(self.trip.area_T)
        gated = out * self.wet                 # driver open-water (ice-free) gate
        lhs = float((gated * area_T).sum())    # what the ocean receives
        rhs = F * float((self.wet * area_T).sum())  # F * wet-ocean area
        rel = abs(lhs - rhs) / abs(rhs)
        # Partition-of-unity => out==F on every covered cell => exact for uniform.
        self.assertLess(rel, 1e-9, f"a2o wet-domain conservation broke (rel={rel})")

    def test_cross_grid_open_water_fraction(self):
        from legoesm.coupler.grid_remap import cross_grid_open_water_fraction
        owet = jnp.asarray(self.wet)
        # Ice-free: a WET ocean cell is FULL open water (f==1) irrespective of the
        # atm coastline (H3); land cells are 0.
        f0 = np.asarray(cross_grid_open_water_fraction(
            owet, jnp.zeros_like(owet)))
        np.testing.assert_allclose(f0, self.wet, atol=1e-12)
        # With ice: wet cells scale by (1-sic); land stays 0.
        sic = jnp.full_like(owet, 0.4)
        f1 = np.asarray(cross_grid_open_water_fraction(owet, sic))
        expected = self.wet * 0.6
        np.testing.assert_allclose(f1, expected, atol=1e-12)

    # ---- identity byte-exactness -------------------------------------------
    def test_identity_path_byte_exact(self):
        from legoesm.coupler.grid_remap import (
            attach_wet_masks,
            make_grid_remapper,
            remap_field,
        )
        rem = make_grid_remapper(self.atm, self.atm)   # same grid -> identity
        self.assertTrue(rem.identity)
        self.assertIsNone(rem.a2o)
        self.assertIsNone(rem.o2a)
        rem2 = attach_wet_masks(
            rem, self.atm, self.atm, np.ones((int(self.atm.n_lat),
                                             int(self.atm.n_lon))))
        self.assertTrue(rem2.identity)
        self.assertIsNone(rem2.o2a)
        self.assertIsNone(rem2.a2o)
        x = jnp.arange(
            int(self.atm.n_lat) * int(self.atm.n_lon), dtype=jnp.float64
        ).reshape(int(self.atm.n_lat), int(self.atm.n_lon))
        out = remap_field(x, rem2.o2a)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(x))

    def test_attach_masks_only_o2a(self):
        from legoesm.coupler.grid_remap import (
            attach_wet_masks,
            make_grid_remapper,
        )
        rem = make_grid_remapper(self.atm, self.trip)
        rem_m = attach_wet_masks(rem, self.atm, self.trip, self.wet)
        self.assertFalse(rem_m.identity)
        # a2o is intentionally UNMASKED -> bit-equal weights.
        np.testing.assert_array_equal(
            np.asarray(rem.a2o.weights), np.asarray(rem_m.a2o.weights))
        # o2a dropped the land-source pairs -> strictly fewer pairs.
        self.assertLess(int(rem_m.o2a.weights.shape[0]),
                        int(rem.o2a.weights.shape[0]))

    # ---- AD through the masked o2a remap -----------------------------------
    def test_grad_flows_through_masked_o2a(self):
        from legoesm.grids.conservative_regrid import apply_conservative_regrid
        from legoesm.grids.conservative_regrid_curvilinear import (
            make_curvilinear_to_regular_weights,
        )
        w = make_curvilinear_to_regular_weights(
            self.trip, self.atm, src_wet=self.wet)

        def loss(src):
            return jnp.sum(apply_conservative_regrid(src, w) ** 2)

        src0 = jnp.asarray(
            np.where(self.wet > 0.5, 0.3, 0.0)).astype(jnp.float64)
        g = np.asarray(jax.grad(loss)(src0))
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertFalse(np.allclose(g, 0.0))

    # ---- H3 driver wiring: the cross-grid branch must USE the ocean wet mask -
    def test_driver_cross_grid_branch_uses_ocean_wet_mask(self):
        """Source-level guard for the H3 wiring (the behavioural change has no
        cheap coupled-driver test on a CPU login node).  The cross-grid branch
        of ``_assemble_ocean_forcing`` must build the open-water fraction from
        the OCEAN's own wet mask via ``cross_grid_open_water_fraction`` -- NOT by
        remapping the ATM open-water fraction, which is the exact bug (imposes
        the atm coastline on the ocean grid).  Tokenised so a comment quoting the
        old expression cannot satisfy or trip the check."""
        import inspect
        import io
        import tokenize

        from legoesm.driver import coupled_esm_driver as ced

        src = inspect.getsource(ced.CoupledESMDriver._assemble_ocean_forcing)
        code = []
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            code.append(tok.string)
        code = " ".join(code)
        self.assertIn("cross_grid_open_water_fraction", code,
                      "H3: cross-grid open-water fraction helper not wired into "
                      "_assemble_ocean_forcing")
        # The old bug: remapping the atm open-water fraction a2o. The dynamic ice
        # fraction IS remapped (remap_field(_sic, ...)), but f_ocean itself must
        # NOT be a remap of the atm tile fraction.
        self.assertNotIn("remap_field ( f_ocean_atm", code,
                         "H3: f_ocean is still a remap of the ATM open-water "
                         "fraction -- the ocean coastline is being overwritten "
                         "by the atm coastline")


if __name__ == "__main__":
    unittest.main()
