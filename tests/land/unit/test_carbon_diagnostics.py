"""Direct test for legoesm.land.carbon_diagnostics.reconstruct_carbon_diagnostics.

Integration-flavoured (runs one coupled land step + reconstruction); compute-
node scale.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.land.carbon.config import CarbonDiagnostics
from legoesm.land.carbon.carbon_cycle import init_carbon_state
from legoesm.land.lmip_forcing import make_synthetic_lmip_forcing
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.multilayer_land import (
    step_multilayer_land, init_multilayer_land_state)
from legoesm.land.carbon_diagnostics import reconstruct_carbon_diagnostics

_HARNESS = (Path(__file__).resolve().parents[3]
            / "scripts" / "validate" / "land_carbon_equilibrium.py")


def _load_harness():
    spec = importlib.util.spec_from_file_location("lce", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lce"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestReconstructCarbonDiagnostics(unittest.TestCase):

    def test_one_step_reconstruction(self):
        lce = _load_harness()
        cfg = lce.build_pixel_config(
            "broadleaf_deciduous_temperate", "loam", False, 6, 2.0,
            "temperate_forest")
        state = init_multilayer_land_state(1, cfg, T_init=283.0)
        carbon = init_carbon_state((1,), cfg.carbon)
        lat_jnp = jnp.asarray([45.0 * jnp.pi / 180.0])
        grid = make_soil_grid(cfg.soil_grid)
        root_frac = jnp.exp(-grid.z_node / cfg.root_depth)
        root_frac = root_frac / jnp.sum(root_frac)
        dt = 3600.0
        forcing = make_synthetic_lmip_forcing(
            45.0 * jnp.pi / 180.0, 0.0, jnp.asarray(180.0), jnp.asarray(12.0))
        new_state, _resp, _cn = step_multilayer_land(
            state, forcing, cfg, 1.0, dt, lat=lat_jnp, carbon_state=carbon,
            doy=jnp.asarray(180.0))
        diag = reconstruct_carbon_diagnostics(
            new_state, forcing, carbon, cfg, root_frac, cfg.theta_wp,
            cfg.theta_fc, cfg.beta_min, lat_jnp, jnp.asarray(180.0), dt,
            spatial=False)
        self.assertIsInstance(diag, CarbonDiagnostics)
        for f in diag._fields:
            self.assertTrue(jnp.all(jnp.isfinite(getattr(diag, f))), f)
        # Allocation closes to machine precision.
        alloc = diag.a_fol + diag.a_lab + diag.a_root + diag.a_wood
        np.testing.assert_allclose(
            np.asarray(alloc), np.asarray(jnp.maximum(diag.npp, 0.0)),
            rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
