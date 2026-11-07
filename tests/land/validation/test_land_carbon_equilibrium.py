"""Tests for the land-carbon equilibrium validator.

Pure-logic tests (config build, pixel catalog, assessment arithmetic,
dispatch guards) run anywhere.  ``test_smoke_integration`` JIT-compiles a
tiny 2-year single-pixel run and is meant for a compute node.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_HARNESS = (Path(__file__).resolve().parents[3]
            / "scripts" / "validate" / "land_carbon_equilibrium.py")


def _load_harness():
    spec = importlib.util.spec_from_file_location("land_carbon_equilibrium", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["land_carbon_equilibrium"] = mod
    spec.loader.exec_module(mod)
    return mod


lce = _load_harness()


def _steady_annual(ny=6):
    """A synthetic, internally consistent equilibrium annual record.

    NEE == 0 (GPP == Ra + Rh, and Rh == NPP == total litterfall); allocation
    closes; every pool's input == output including the coarse-woody-debris
    humification split (wood_litter = wood_to_som + R_het_cwd).
    """
    const = lambda v: np.full(ny, float(v))
    npp, r_auto, humif = 500.0, 500.0, 0.3
    a = dict(a_fol=75.0, a_lab=42.5, a_root=95.6, a_wood=286.9)  # sum == 500
    # Litterfall == NPP; each biomass pool in == out.
    lab_release = a["a_lab"]                        # labile in == out
    leaf_litter = a["a_fol"] + lab_release          # foliage in == out
    root_litter = a["a_root"]                       # root in == out
    wood_litter = a["a_wood"]                       # wood in == out
    wood_to_som = humif * wood_litter
    r_het_cwd = wood_litter - wood_to_som
    lit_to_som = 50.0
    r_het_som = lit_to_som + wood_to_som            # SOM in == out
    r_het_lit = (leaf_litter + root_litter) - lit_to_som  # litter in == out
    r_het = r_het_lit + r_het_som + r_het_cwd        # == NPP == 500
    d = dict(
        gpp=const(npp + r_auto), npp=const(npp), r_auto=const(r_auto),
        r_maint=const(300.0), r_growth=const(200.0), r_het=const(r_het),
        r_het_lit=const(r_het_lit), r_het_som=const(r_het_som),
        r_het_cwd=const(r_het_cwd),
        nee=const(0.0), nee_model=const(0.0),
        a_fol=const(a["a_fol"]), a_lab=const(a["a_lab"]),
        a_root=const(a["a_root"]), a_wood=const(a["a_wood"]),
        lab_release=const(lab_release), leaf_litter=const(leaf_litter),
        root_litter=const(root_litter), wood_litter=const(wood_litter),
        wood_to_som=const(wood_to_som), lit_to_som=const(lit_to_som),
        alloc_resid=const(0.0),
        lai_sum=const(4.0 * 100), lai_max=const(4.0), nsteps=const(100.0),
        C_lab=const(500.0), C_fol=const(400.0), C_root=const(1500.0),
        C_wood=const(18000.0), C_lit=const(800.0),
        # SOM resolved into active/slow/passive (A1: slow/passive inert == 0).
        C_som_active=const(11000.0), C_som_slow=const(0.0),
        C_som_passive=const(0.0),
    )
    return d


class TestPixelCatalog(unittest.TestCase):

    def test_pixels_reference_valid_pfts_textures_biomes(self):
        from legoesm.land.surface_params import CLM5_PFT_NAMES
        from legoesm.land.soil_texture import SOIL_TEXTURE_VG
        for (name, lat, lon, pft, texture, T_init, precip, ft, biome) in lce.PIXELS:
            self.assertIn(pft, CLM5_PFT_NAMES, name)
            self.assertIn(texture, SOIL_TEXTURE_VG, name)
            self.assertIn(biome, lce.LITERATURE, name)
            self.assertTrue(-90 <= lat <= 90 and precip > 0)

    def test_build_pixel_config(self):
        for (name, lat, lon, pft, texture, T_init, precip, ft, biome) in lce.PIXELS:
            cfg = lce.build_pixel_config(pft, texture, ft, 8, 3.0, biome)
            self.assertEqual(cfg.carbon.scheme, "differland")
            self.assertTrue(cfg.stomata.enabled)
            self.assertEqual(cfg.thermal.enable_freeze_thaw, ft)
            self.assertGreater(cfg.stomata.Vc_max25, 0.0)
            # Woody stands seed a biome-appropriate wood pool; herbaceous none.
            if lce.is_woody(pft):
                self.assertGreater(cfg.carbon.C_wood_init, 0.0)
            else:
                self.assertEqual(cfg.carbon.C_wood_init, 0.0)

    def test_pixels_run_the_drainage_limiter_at_the_calibrated_value(self):
        # Calibrated at 0.5; must not inherit the RichardsConfig default (0.0).
        for (name, lat, lon, pft, texture, T_init, precip, ft, biome) in lce.PIXELS:
            cfg = lce.build_pixel_config(pft, texture, ft, 8, 3.0, biome)
            self.assertEqual(cfg.richards.fc_drain_saturation, 0.5, name)


class TestAssess(unittest.TestCase):

    def test_steady_state_passes_all_hard_checks(self):
        d = _steady_annual()
        rec = lce.assess_pixel("temperate_deciduous", "temperate_forest", d, None)
        for k, v in rec["checks"].items():
            self.assertTrue(v, f"hard check {k} failed on steady synthetic input")
        self.assertAlmostEqual(rec["wood_in_out_ratio"], 1.0, places=3)
        self.assertAlmostEqual(rec["som_in_out_ratio"], 1.0, places=3)
        self.assertAlmostEqual(sum(rec["alloc_frac"].values()), 1.0, places=6)
        self.assertAlmostEqual(rec["cue"], 0.5, places=3)

    def test_herbaceous_wood_check_flags_woody_grass(self):
        """A grass pixel that accumulated wood must fail the herbaceous check."""
        d = _steady_annual()
        # temperate_grassland is herbaceous; C_wood=18 kgC should flag.
        rec = lce.assess_pixel("temperate_grassland", "grassland", d, None)
        self.assertTrue(rec["is_herbaceous"])
        self.assertFalse(rec["checks"]["herbaceous_no_wood"])

    def test_allocation_residual_detected(self):
        d = _steady_annual()
        d["alloc_resid"] = np.full_like(d["alloc_resid"], 5.0)  # broken closure
        rec = lce.assess_pixel("temperate_deciduous", "temperate_forest", d, None)
        self.assertFalse(rec["checks"]["allocation_closes"])


class TestDispatch(unittest.TestCase):

    def test_only_unknown_pixel_raises(self):
        with self.assertRaises(SystemExit):
            lce.main(["--only", "does_not_exist", "--years", "1"])


class TestSmokeIntegration(unittest.TestCase):
    """Tiny JIT-compiled run — compute-node scale (skipped-fast if it errors
    on an unusual backend, but should pass on CPU/GPU)."""

    def test_two_year_tropical(self):
        cfg = lce.build_pixel_config(
            "broadleaf_evergreen_tropical", "clay_loam", False, 6, 2.0,
            "tropical_forest")
        # Tiny smoke: 2 spin-up + 2 verify years (exercises the semi-analytic
        # slow-pool reset path + both scan phases).
        annual, final_state, final_carbon = lce.run_pixel(
            cfg, lat_deg=2.0, lon_deg=-60.0, T_init=298.0,
            precip_rate=6e-5, n_spinup=2, dt=7200.0, n_verify=2)
        rec = lce.assess_pixel("tropical_rainforest", "tropical_forest",
                               annual, final_carbon)
        # Allocation must close to machine precision and pools stay physical.
        self.assertTrue(rec["checks"]["allocation_closes"],
                        f"alloc_resid={rec['alloc_resid']:.3e}")
        self.assertTrue(rec["checks"]["pools_nonneg"])
        self.assertTrue(rec["checks"]["npp_le_gpp"])
        # Reconstructed-diagnostic NEE is informational (residual-sensitive on
        # the coupled model); only guard against gross divergence.
        self.assertLess(rec["nee_diag_rel_mismatch"], 1.5)
        self.assertTrue(np.isfinite(rec["gpp"]) and rec["gpp"] >= 0.0)
        for v in final_carbon:
            self.assertTrue(np.all(np.isfinite(np.asarray(v))))
        # A2: the three live SOM pools are all positive after the analytic
        # forward-substitution reset + verify segment, and the total exceeds the
        # active pool alone (the slow/passive pools carry real stock).
        for p in ("C_som_active", "C_som_slow", "C_som_passive"):
            self.assertTrue(np.all(np.asarray(getattr(final_carbon, p)) > 0.0), p)
        from legoesm.land.carbon.config import som_total
        self.assertGreater(
            float(np.asarray(som_total(final_carbon)).reshape(-1)[0]),
            float(np.asarray(final_carbon.C_som_active).reshape(-1)[0]))


if __name__ == "__main__":
    unittest.main()
