"""Land runoff must reach the dynamic ocean — once, and in the right channel.

Regression for the water-budget leak where ``_assemble_ocean_forcing`` built
the 3D-ocean ``FreshwaterForcing`` with ``runoff=0`` (and ``ice_fw=0``), so
interactive land runoff (rivers) and ice melt computed by the coupler were
dropped before the prognostic ocean step.

The fix keeps ocean P-E CURRENT in the precip/evap channels and lags ONLY the
two non-ocean sub-channels of the blended surface response:

  * ``runoff``  <- ``river_runoff_flux``       (land rivers, depth-spread)
  * ``ice_fw``  <- ``ice_lake_freshwater_flux`` (ice melt + lake P-E, surface)

so ocean P-E is never reconstructed (no stale-P-E or coastal area-weight
residual), land runoff is depth-spread while ice melt stays at the surface,
and the aquaplanet path is byte-identical even under time-varying forcing.
"""

from __future__ import annotations

import types
import unittest

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.freshwater import net_freshwater_flux


def _fake_tile(shape, lhflx=50.0):
    """Minimal ocean-tile response with the fields _assemble reads."""
    z = jnp.zeros(shape)
    return types.SimpleNamespace(
        albedo=jnp.full(shape, 0.06),
        lw_up=jnp.full(shape, 400.0),
        shflx=jnp.full(shape, 10.0),
        lhflx=jnp.full(shape, lhflx),   # >0 => evaporation, drives evap channel
        tau_x=z, tau_y=z,
    )


def _assemble(prev, *, precip=2.0e-5, lhflx=50.0, shape=(3, 4)):
    """Call the real ``_assemble_ocean_forcing`` against a light stub self.

    ``prev`` is a stub lagged ``SurfaceToAtm`` (or ``None`` for the first step).
    """
    from unittest import mock
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    sst = jnp.full(shape, 290.0)
    cur = jnp.zeros(shape)
    stub = types.SimpleNamespace(
        _coupler_cfg=None,
        _last_sfc_response=prev,
        _ocean_surface_KuvC=lambda: (sst, cur, cur),
    )
    atm_forcing = types.SimpleNamespace(
        sw_down=jnp.full(shape, 300.0),
        lw_down=jnp.full(shape, 350.0),
        precip_total=jnp.full(shape, precip),   # kg/m²/s
    )
    with mock.patch(
        "legoesm.coupler.coupler.ocean_tile_response",
        return_value=_fake_tile(shape, lhflx=lhflx),
    ):
        sf, fw = CoupledESMDriver._assemble_ocean_forcing(stub, atm_forcing)
    evap = jnp.full(shape, lhflx) / constants.L_v
    return fw, atm_forcing.precip_total, evap


def _prev(shape=(3, 4), *, river=0.0, ice_lake=0.0):
    z = jnp.zeros(shape)
    return types.SimpleNamespace(
        freshwater_flux=z,  # unused by the split now, kept for realism
        river_runoff_flux=jnp.full(shape, river),
        ice_lake_freshwater_flux=jnp.full(shape, ice_lake),
        # Ice->ocean back-reaction channels (zero for the no-ice aquaplanet path
        # these tests exercise) — a real SurfaceToAtm always carries them.
        ocean_heat_extraction=z,
        salt_flux=z,
        ocean_stress_x=z,
        ocean_stress_y=z,
    )


class TestRunoffReachesOcean(unittest.TestCase):
    def test_no_prior_response_is_zero(self):
        # First step: nothing lagged yet -> both non-ocean channels zero.
        fw, precip, evap = _assemble(None)
        self.assertTrue(jnp.allclose(fw.runoff, 0.0))
        self.assertTrue(jnp.allclose(fw.ice_fw, 0.0))
        # Ocean P-E delivered current.
        self.assertTrue(jnp.allclose(net_freshwater_flux(fw), precip - evap, atol=1e-12))

    def test_aquaplanet_byte_identical_under_time_varying_forcing(self):
        # No land/ice/lake tile => lagged channels are zero EVEN when the
        # current precip/evap differ from the step that produced ``prev``.
        # Guards codex HIGH#1: ocean P-E must be CURRENT, never stale.
        prev = _prev(river=0.0, ice_lake=0.0)
        fw, precip, evap = _assemble(prev, precip=9.9e-5, lhflx=123.0)
        # EXACT equality (atol=rtol=0): the lagged channels must contribute
        # literally nothing and ocean P-E must be the current value, so the
        # delivered freshwater is bit-for-bit the legacy runoff=ice_fw=0 result.
        self.assertTrue(bool(jnp.array_equal(fw.runoff, jnp.zeros_like(fw.runoff))))
        self.assertTrue(bool(jnp.array_equal(fw.ice_fw, jnp.zeros_like(fw.ice_fw))))
        # Net == CURRENT ocean P-E (not the stale value from prev's step).
        self.assertTrue(bool(jnp.array_equal(net_freshwater_flux(fw), precip - evap)))

    def test_land_runoff_in_depth_spread_channel(self):
        # Land river runoff R lands on the depth-spread ``runoff`` channel; ice_fw
        # stays zero; total = current ocean P-E + R (counted once).
        R = 7.0e-6
        fw, precip, evap = _assemble(_prev(river=R))
        self.assertTrue(jnp.allclose(fw.runoff, R, atol=1e-12))
        self.assertTrue(jnp.allclose(fw.ice_fw, 0.0, atol=1e-12))
        self.assertTrue(
            jnp.allclose(net_freshwater_flux(fw), (precip - evap) + R, atol=1e-12)
        )

    def test_ice_melt_in_surface_channel_not_depth_spread(self):
        # Ice melt M lands on the SURFACE ``ice_fw`` channel, NOT the depth-spread
        # ``runoff`` channel — the ice_fw correctness the fix guarantees.
        M = 4.0e-6
        fw, precip, evap = _assemble(_prev(ice_lake=M))
        self.assertTrue(jnp.allclose(fw.runoff, 0.0, atol=1e-12))
        self.assertTrue(jnp.allclose(fw.ice_fw, M, atol=1e-12))
        self.assertTrue(
            jnp.allclose(net_freshwater_flux(fw), (precip - evap) + M, atol=1e-12)
        )

    def test_both_channels_together(self):
        # Land runoff and ice melt delivered simultaneously to distinct channels.
        R, M = 6.0e-6, 3.0e-6
        fw, precip, evap = _assemble(_prev(river=R, ice_lake=M))
        self.assertTrue(jnp.allclose(fw.runoff, R, atol=1e-12))
        self.assertTrue(jnp.allclose(fw.ice_fw, M, atol=1e-12))
        self.assertTrue(
            jnp.allclose(net_freshwater_flux(fw), (precip - evap) + R + M, atol=1e-12)
        )

    def test_runoff_sign_dilutes(self):
        # Positive land runoff must enter as positive (into-ocean) freshwater —
        # a sign flip here would concentrate salinity instead of diluting it.
        fw, _, _ = _assemble(_prev(river=5.0e-6))
        self.assertTrue(bool(jnp.all(fw.runoff > 0.0)))


if __name__ == "__main__":
    unittest.main()
