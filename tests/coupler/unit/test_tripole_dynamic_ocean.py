"""Phase-2 tripole-ocean coupler: ``_init_tripole_dynamic_ocean`` integration.

Exercises the driver-side wiring in isolation (no full atm/land/ice spin-up):
  * the dynamic-ocean guard ACCEPTS a tripole (active-fold) grid;
  * the OMIP-validated tripole recipe is built (adcroft PGF, implicit_cn);
  * the cold-start IC (NEMO mesh mask+bathy -> partial cells -> rest state) is
    finite and physical;
  * ONE ocean step from the cold start does not NaN (stability sanity — the
    standalone OMIP tripole cold-start is the validated target).

Builds a SYNTHETIC tripole geometry + a matching tiny synthetic NEMO mesh_mask
so no 484 MB ORCA1 file is needed.  Builds JAX ocean dynamics -> run on a
compute node (sbatch), not the login node.
"""

import sys
import unittest
from pathlib import Path
import tempfile

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _write_synthetic_nemo_mesh(path, n_lat, n_lon, nlev, H_max):
    """Tiny synthetic NEMO mesh_mask matching a create_synthetic_tripole(n_lat).

    Ocean everywhere except a southern land row + one interior land cell;
    e3t_0 uniform so H_bathy = e3t * (#wet levels)."""
    import xarray as xr
    dz = H_max / nlev
    tmaskutil = np.ones((n_lat, n_lon), dtype=np.float64)
    tmaskutil[0, :] = 0.0           # south land row
    tmaskutil[n_lat // 2, n_lon // 2] = 0.0  # interior island
    # Wet to a per-column depth: deeper in mid-basin, shallow near edges.
    kbot = np.full((n_lat, n_lon), nlev, dtype=int)
    kbot[:, 0] = max(2, nlev // 2)
    tmask = np.zeros((nlev, n_lat, n_lon), dtype=np.float64)
    for k in range(nlev):
        tmask[k] = ((k < kbot) & (tmaskutil > 0.5)).astype(np.float64)
    e3t0 = np.full((nlev, n_lat, n_lon), dz, dtype=np.float64)
    ds = xr.Dataset({
        "tmaskutil": (("y", "x"), tmaskutil),
        "tmask": (("z", "y", "x"), tmask),
        "e3t_0": (("z", "y", "x"), e3t0),
    })
    ds.to_netcdf(path, engine="scipy")


class TestTripoleDynamicOcean(unittest.TestCase):

    def _build(self, ocean_ic="rest"):
        from legoesm.grids.tripole import create_synthetic_tripole
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        from legoesm.driver.coupled_config import CoupledConfig

        n_lat, nlev, H_max = 12, 5, 2000.0
        grid = create_synthetic_tripole(n_lat=n_lat)
        n_lon = int(grid.n_lon)

        tmp = tempfile.mkdtemp()
        mesh = str(Path(tmp) / "synthetic_eorca_mesh.nc")
        _write_synthetic_nemo_mesh(mesh, n_lat, n_lon, nlev, H_max)

        cfg = CoupledConfig(
            ocean_mode="dynamic", ocean_ic=ocean_ic,
            ocean_nlev=nlev, ocean_H_max_m=H_max, ocean_dt_s=300.0,
            tripole_mesh_path=mesh,
        )
        drv = CoupledESMDriver.__new__(CoupledESMDriver)
        drv.coupled_cfg = cfg
        drv._ocean_grid = grid
        drv._init_tripole_dynamic_ocean()
        return drv

    def test_guard_accepts_tripole_and_builds_omip_recipe(self):
        drv = self._build()
        # OMIP-validated tripole recipe (NOT the lat-lon smc03/rk3 recipe).
        ocfg = drv._ocean_model.config
        self.assertEqual(ocfg.barotropic.barotropic_solver, "implicit_cn")
        self.assertEqual(ocfg.pgf_scheme, "adcroft")
        self.assertAlmostEqual(ocfg.C_smag_lap, 0.33, places=6)
        self.assertTrue(ocfg.implicit_vertical_mixing)
        self.assertEqual(ocfg.tracer_advection, "tvd")
        self.assertTrue(drv._is_dynamic_ocean)

    def test_cold_start_ic_finite_and_masked(self):
        drv = self._build()
        st = drv._ocean_state
        T = np.asarray(st.T.data)
        S = np.asarray(st.S.data)
        self.assertTrue(np.all(np.isfinite(T)))
        self.assertTrue(np.all(np.isfinite(S)))
        # Ocean fraction strictly between 0 and 1 (there IS land + ocean).
        frac = float(jnp.mean(drv._ocean_land_mask))
        self.assertGreater(frac, 0.0)
        self.assertLess(frac, 1.0)

    def test_one_ocean_step_does_not_nan(self):
        drv = self._build()
        st1 = drv._ocean_model.step(drv._ocean_state, 300.0)
        for name, arr in (("T", st1.T.data), ("S", st1.S.data),
                          ("u", st1.u.data), ("v", st1.v.data),
                          ("eta", st1.eta.data)):
            a = np.asarray(arr)
            self.assertTrue(np.all(np.isfinite(a)),
                            f"{name} has non-finite after one step")
        # Velocity sane for a single cold-start step (not a runaway).
        umax = float(jnp.max(jnp.abs(st1.u.data)))
        self.assertLess(umax, 5.0, f"max|u|={umax} m/s after one step")

    def test_rejects_unsupported_ocean_grid(self):
        """A cube/MPAS-like ocean grid (no active fold, not LatLonGrid) is
        rejected by the dynamic-ocean guard."""
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        from legoesm.driver.coupled_config import CoupledConfig

        class _FakeCube:  # neither LatLonGrid nor active-fold tripole
            pass

        cfg = CoupledConfig(ocean_mode="dynamic", ocean_ic="rest")
        drv = CoupledESMDriver.__new__(CoupledESMDriver)
        drv.coupled_cfg = cfg
        drv._ocean_grid = _FakeCube()
        with self.assertRaises(ValueError):
            drv._init_dynamic_ocean(285.0)


if __name__ == "__main__":
    unittest.main()
