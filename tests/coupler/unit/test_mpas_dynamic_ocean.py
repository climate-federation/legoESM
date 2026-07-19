"""Voronoi/MPAS 3-D dynamic-ocean coupler wiring: ``_init_mpas_dynamic_ocean``.

Exercises the driver-side wiring in isolation (no full atm/land/ice spin-up),
mirroring ``test_tripole_dynamic_ocean`` for the MPAS path:
  * the dynamic-ocean dispatch ACCEPTS a VoronoiMesh ocean grid and builds the
    OMIP NEMO-match MPAS recipe with the two coupled overlays
    (surface_forcing scheme='none' + normalize_freshwater=True);
  * the stratified rest cold-start IC is finite, physical, all-ocean;
  * ``_ocean_surface_KuvC`` reads back (nCells,) SST [K] + Perot-reconstructed
    surface currents (geographic east/north);
  * ONE ocean step from the cold start does not NaN (stability sanity).

Small ico level-2 mesh (162 cells) — fast, CPU-only.
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

_NLEV = 5
_H_MAX = 4000.0
_T_SFC_MEAN = 293.0          # K


def _build_mpas_driver(level=2):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.driver.coupled_config import CoupledConfig

    mesh = create_voronoi_mesh(level)
    cfg = CoupledConfig(
        ocean_mode="dynamic", ocean_ic="rest",
        ocean_nlev=_NLEV, ocean_H_max_m=_H_MAX, ocean_dt_s=300.0,
    )
    drv = CoupledESMDriver.__new__(CoupledESMDriver)
    drv.coupled_cfg = cfg
    drv._ocean_grid = mesh
    # Dispatch through the public entry so we also cover the VoronoiMesh branch.
    drv._init_dynamic_ocean(_T_SFC_MEAN)
    return drv, mesh


class TestMPASDynamicOcean(unittest.TestCase):

    def test_dispatch_builds_mpas_with_coupled_overlays(self):
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        drv, _ = _build_mpas_driver()
        self.assertTrue(drv._is_dynamic_ocean)
        self.assertTrue(drv._ocean_is_mpas)
        self.assertIsInstance(drv._ocean_model, MPASOceanModel)
        cfg = drv._ocean_model.config
        # The two coupled-run overlays on the proven OMIP recipe.
        self.assertEqual(cfg.physics.surface_forcing.scheme, "none")
        self.assertTrue(cfg.normalize_freshwater)

    def test_cold_start_ic_finite_and_all_ocean(self):
        drv, mesh = _build_mpas_driver()
        st = drv._ocean_state
        T = np.asarray(st.T.data)
        S = np.asarray(st.S.data)
        self.assertEqual(T.shape, (mesh.nCells, _NLEV))
        self.assertTrue(np.all(np.isfinite(T)) and np.all(np.isfinite(S)))
        # Zero-velocity rest start.
        self.assertTrue(np.all(np.asarray(st.u.data) == 0.0))
        # land_lat_threshold=90 => all-ocean aquaplanet (mask==1 everywhere).
        frac = float(jnp.mean(drv._ocean_land_mask))
        self.assertAlmostEqual(frac, 1.0, places=6)
        # Surface layer near the atmosphere's mean SST (293 K -> ~20 degC).
        self.assertGreater(float(T[:, 0].mean()), 15.0)
        self.assertLess(float(T[:, 0].mean()), 25.0)

    def test_surface_readback_sst_and_currents(self):
        drv, mesh = _build_mpas_driver()
        sst, u_e, v_n = drv._ocean_surface_KuvC()
        for name, a in (("sst", sst), ("u_east", u_e), ("v_north", v_n)):
            arr = np.asarray(a)
            self.assertEqual(arr.shape, (mesh.nCells,), f"{name} wrong shape")
            self.assertTrue(np.all(np.isfinite(arr)), f"{name} non-finite")
        # SST is Kelvin (top-level T[degC] + T_freeze), physical band.
        self.assertGreater(float(np.asarray(sst).min()), 250.0)
        self.assertLess(float(np.asarray(sst).max()), 320.0)
        # Rest start => zero surface currents.
        self.assertTrue(np.allclose(np.asarray(u_e), 0.0))
        self.assertTrue(np.allclose(np.asarray(v_n), 0.0))

    def test_one_ocean_step_does_not_nan(self):
        drv, _ = _build_mpas_driver()
        st1 = drv._ocean_model.step(drv._ocean_state, 300.0)
        for name, arr in (("T", st1.T.data), ("S", st1.S.data),
                          ("u", st1.u.data), ("eta", st1.eta.data)):
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))),
                            f"{name} non-finite after one step")
        umax = float(jnp.max(jnp.abs(st1.u.data)))
        self.assertLess(umax, 5.0, f"max|u|={umax} m/s after one cold-start step")

    def test_ocean_steps_and_responds_to_wind_stress(self):
        """PROOF the ocean actually STEPS and responds (not a no-op): the coupled
        path forces the MPAS ocean with an EXTERNAL ``OceanSurfaceForcing``
        (config surface_forcing scheme='none'), so a nonzero wind stress must
        spin up edge-normal currents from the rest cold start — and the coupler
        surface readback (``_ocean_surface_KuvC``) must then reflect them."""
        from legoesm.ocean.state import OceanSurfaceForcing
        drv, mesh = _build_mpas_driver()
        _dt = drv._ocean_state.u.data.dtype          # match the ocean storage dtype
        z = jnp.zeros((mesh.nCells,), dtype=_dt)
        tau = jnp.full((mesh.nCells,), 0.1, dtype=_dt)  # 0.1 Pa uniform zonal stress
        sf = OceanSurfaceForcing(sw_down=z, q_net=z, tau_x=tau, tau_y=z,
                                 salt_flux=z, freshwater=None)
        self.assertEqual(drv._ocean_model.config.physics.surface_forcing.scheme,
                         "none")
        st = drv._ocean_state
        self.assertEqual(float(jnp.max(jnp.abs(st.u.data))), 0.0)  # rest start
        for _ in range(10):
            st = drv._ocean_model.step(st, 300.0, surface_forcing=sf)
        umax = float(jnp.max(jnp.abs(st.u.data)))
        self.assertTrue(np.isfinite(umax))
        self.assertGreater(umax, 0.0, "wind stress spun up NO current — the "
                           "external surface_forcing is not reaching the ocean")
        # The coupler reads currents back from the STEPPED state.
        drv._ocean_state = st
        _, u_east, v_north = drv._ocean_surface_KuvC()
        self.assertGreater(
            float(jnp.max(jnp.abs(u_east)) + jnp.max(jnp.abs(v_north))), 0.0,
            "surface-current readback is zero after a forced step")

    def test_dispatch_rejects_cube_like_grid(self):
        """A non-Voronoi, non-LatLon, non-fold ocean grid is still rejected."""
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        from legoesm.driver.coupled_config import CoupledConfig

        class _FakeCube:
            pass

        cfg = CoupledConfig(ocean_mode="dynamic", ocean_ic="rest")
        drv = CoupledESMDriver.__new__(CoupledESMDriver)
        drv.coupled_cfg = cfg
        drv._ocean_grid = _FakeCube()
        with self.assertRaises(ValueError):
            drv._init_dynamic_ocean(285.0)


if __name__ == "__main__":
    unittest.main()
