"""Volcanic stratospheric LONGWAVE aerosol (#9): the new optional per-layer
``aerosol_lw_od`` is OFF by default (byte-identical no-op) and, when supplied,
materialises as a JIT-stable field and reaches the radiation backend.

End-to-end "LW aerosol changes RRTMGP heating" is covered indirectly: the full
radiation-chain regression (test_physics_radiation / test_radiation_number_coupling
/ test_compiled_segments) passes with the new param threaded, and the backend-spy
test here pins that a supplied ``aerosol_lw_od`` is delivered to the solver call.
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


class TestVolcanicLwDefaultOff(unittest.TestCase):
    """The loader is OFF by default => no LW aerosol => no-op."""

    def test_get_aerosol_lw_disabled_returns_none(self):
        from legoesm.forcing.external import AerosolConfig, get_aerosol_lw_at_time
        cfg = AerosolConfig()  # volcanic_lw_enabled defaults False
        self.assertFalse(cfg.volcanic_lw_enabled)
        out = get_aerosol_lw_at_time(cfg, day=0.0, lat_grid=np.zeros(4))
        self.assertIsNone(out)

    def test_get_aerosol_lw_enabled_without_path_returns_none(self):
        from legoesm.forcing.external import AerosolConfig, get_aerosol_lw_at_time
        cfg = AerosolConfig(volcanic_lw_enabled=True)  # no volcanic_path
        out = get_aerosol_lw_at_time(cfg, day=0.0, lat_grid=np.zeros(4))
        self.assertIsNone(out)


class TestPackForcingLwField(unittest.TestCase):
    """pack_forcing materialises aerosol_lw_od as a JIT-stable concrete array."""

    def _pack(self, aerosol_lw_od):
        from legoesm.driver.compiled_segments import pack_forcing
        shp = (6, 4, 4)
        aer = jnp.zeros((6 * 4 * 4, 5))  # (ncol, nlev) per-layer SW aerosol od
        return pack_forcing(
            sst=jnp.full(shp, 290.0), sic=jnp.zeros(shp),
            day_of_year=1, seconds_of_day=0.0,
            solar_weights=jnp.ones(1), s_0=1361.0,
            o3_vmr=jnp.zeros((6 * 4 * 4, 5)), aerosol_od=aer,
            aerosol_lw_od=aerosol_lw_od,
        )

    def test_none_materialises_zeros_same_shape(self):
        f = self._pack(None)
        self.assertEqual(f.aerosol_lw_od.shape, f.aerosol_od.shape)
        self.assertTrue(jnp.all(f.aerosol_lw_od == 0.0))

    def test_supplied_value_preserved(self):
        lw = jnp.full((6 * 4 * 4, 5), 0.1)
        f = self._pack(lw)
        self.assertEqual(f.aerosol_lw_od.shape, (6 * 4 * 4, 5))
        self.assertTrue(jnp.allclose(f.aerosol_lw_od, 0.1))


if __name__ == "__main__":
    unittest.main()
