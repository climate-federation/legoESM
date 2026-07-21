"""3D-ocean surface forcing must not treat ice-covered cells as ice-free.

Regression for the energy+mass leak where ``_assemble_ocean_forcing`` built the
prognostic-ocean surface forcing from the FULL-CELL open-water (ocean-tile)
fluxes with NO ``f_ocean`` scaling and delivered the ice-melt freshwater
(``ice_fw``) WITHOUT the ice back-reaction heat / salt / stress the ice model
already computed on ``_last_sfc_response``.  Net over an ice-covered cell: the
ocean was heated / evaporated / wind-stressed as if ICE-FREE and the melt water
arrived without its melt heat.

The fix:

  * scales the open-water fluxes (q_net, sw_down, tau, evap, precip) by
    ``f_ocean = f_water*(1 - ice_concentration)`` so the ice-covered fraction is
    NOT forced by the open-ocean bulk fluxes; and
  * ADDS the ice->ocean back-reaction from the blended ``_last_sfc_response``
    (``ocean_heat_extraction`` -> negative q_net, ``salt_flux`` -> +into ocean,
    ``ocean_stress_x/y`` -> ``-stress`` in the ocean's ``-tau`` convention),
    pairing the melt/freeze freshwater with its heat and salt.
"""

from __future__ import annotations

import types
import unittest

import jax.numpy as jnp


def _fake_tile(shape, *, lhflx=50.0, tau_x=0.0, tau_y=0.0):
    """Minimal ocean-tile response with the fields ``_assemble`` reads."""
    z = jnp.zeros(shape)
    return types.SimpleNamespace(
        albedo=jnp.full(shape, 0.06),
        lw_up=jnp.full(shape, 400.0),
        shflx=jnp.full(shape, 10.0),
        lhflx=jnp.full(shape, lhflx),   # >0 => evaporation
        tau_x=jnp.full(shape, tau_x),
        tau_y=jnp.full(shape, tau_y),
    )


def _prev(shape, *, ice_lake=0.0, ohe=0.0, salt=0.0, stress_x=0.0, stress_y=0.0):
    """Stub blended ``SurfaceToAtm`` carrying the ice->ocean back-reaction."""
    z = jnp.zeros(shape)
    return types.SimpleNamespace(
        freshwater_flux=z,
        river_runoff_flux=z,
        ice_lake_freshwater_flux=jnp.full(shape, ice_lake),  # ice melt (+into ocean)
        ocean_heat_extraction=jnp.full(shape, ohe),          # +ocean LOSES heat
        salt_flux=jnp.full(shape, salt),                     # +into ocean
        ocean_stress_x=jnp.full(shape, stress_x),            # +eastward force ON ocean
        ocean_stress_y=jnp.full(shape, stress_y),
    )


def _assemble(conc, prev, *, precip=2.0e-5, lhflx=50.0, tau_x=0.0):
    """Call the real ``_assemble_ocean_forcing`` against a light stub self.

    ``conc`` is the prognostic ice concentration (shape (1, 2): an ice-free and
    an ice-covered cell); ``prev`` a stub lagged ``SurfaceToAtm``.
    """
    from unittest import mock
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    conc = jnp.asarray(conc)
    shape = conc.shape
    sst = jnp.full(shape, 290.0)
    cur = jnp.zeros(shape)
    z = jnp.zeros(shape)
    stub = types.SimpleNamespace(
        _coupler_cfg=None,
        _last_sfc_response=prev,
        _ocean_surface_KuvC=lambda: (sst, cur, cur),
        # Prognostic ice concentration + static tile config drive f_ocean.
        _sfc_state=types.SimpleNamespace(
            ice=types.SimpleNamespace(
                concentration=types.SimpleNamespace(data=conc))),
        _tile_config=types.SimpleNamespace(f_land=z, f_lake=z),
    )
    atm_forcing = types.SimpleNamespace(
        sw_down=jnp.full(shape, 300.0),
        lw_down=jnp.full(shape, 350.0),
        precip_total=jnp.full(shape, precip),
    )
    with mock.patch(
        "legoesm.coupler.coupler.ocean_tile_response",
        return_value=_fake_tile(shape, lhflx=lhflx, tau_x=tau_x),
    ):
        return CoupledESMDriver._assemble_ocean_forcing(stub, atm_forcing)


class TestOceanIceForcing(unittest.TestCase):
    def test_ice_cell_gets_open_water_flux_scaled_by_f_ocean(self):
        # Cell 0 ice-free (conc 0 => f_ocean 1); cell 1 conc 0.8 => f_ocean 0.2.
        conc = jnp.array([[0.0, 0.8]])
        sf, fw = _assemble(conc, _prev(conc.shape))  # zero back-reaction
        # q_net over the ice cell is 0.2x the ice-free cell (open-water scaling).
        self.assertNotAlmostEqual(float(sf.q_net[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(
            float(sf.q_net[0, 1]), 0.2 * float(sf.q_net[0, 0]), places=8
        )
        # Evaporation likewise: only the open-water fraction evaporates.
        self.assertNotAlmostEqual(float(fw.evap[0, 0]), 0.0, places=8)
        self.assertAlmostEqual(
            float(fw.evap[0, 1]), 0.2 * float(fw.evap[0, 0]), places=10
        )
        # Penetrating SW and precip carry the same open-water scaling.
        self.assertAlmostEqual(
            float(sf.sw_down[0, 1]), 0.2 * float(sf.sw_down[0, 0]), places=8
        )
        self.assertAlmostEqual(
            float(fw.precip[0, 1]), 0.2 * float(fw.precip[0, 0]), places=12
        )

    def test_ice_free_cell_byte_identical_to_full_cell(self):
        # conc 0 everywhere => f_ocean 1 => open-water fluxes unscaled and the
        # (zero) back-reaction leaves q_net exactly the full-cell open-water value.
        conc = jnp.zeros((1, 2))
        sf, _ = _assemble(conc, _prev(conc.shape))
        sw_net = jnp.full(conc.shape, 300.0) * (1.0 - 0.06)
        q_open = sw_net + 350.0 - 400.0 - 10.0 - 50.0
        self.assertTrue(bool(jnp.allclose(sf.q_net, q_open, atol=1e-9)))

    def test_melt_freshwater_arrives_with_heat_salt_and_stress(self):
        # Ice cell (conc 0.8) with melt water M, basal heat extraction H>0,
        # brine salt S_b, and ice stress T_s.  The melt water must NOT arrive
        # without its heat/salt/stress (the closed leak).
        conc = jnp.array([[0.0, 0.8]])
        M, H, S_b, T_s = 4.0e-6, 25.0, 1.0e-6, 0.05
        base_sf, base_fw = _assemble(conc, _prev(conc.shape))  # no back-reaction
        sf, fw = _assemble(
            conc,
            _prev(conc.shape, ice_lake=M, ohe=H, salt=S_b, stress_x=T_s),
            tau_x=0.0,
        )
        ice = (0, 1)
        # Melt freshwater delivered.
        self.assertAlmostEqual(float(fw.ice_fw[ice]), M, places=12)
        self.assertGreater(float(fw.ice_fw[ice]), 0.0)
        # ...and it is PAIRED with heat: q_net drops by exactly the extraction H
        # (ocean_heat_extraction is +ocean-loses; q_net is +into ocean => -H).
        self.assertAlmostEqual(
            float(sf.q_net[ice] - base_sf.q_net[ice]), -H, places=8
        )
        # Not "freshwater without heat": the back-reaction moved q_net.
        self.assertGreater(abs(float(sf.q_net[ice] - base_sf.q_net[ice])), 0.0)
        # Brine salt delivered +into ocean.
        self.assertAlmostEqual(float(sf.salt_flux[ice]), S_b, places=12)
        self.assertGreater(float(sf.salt_flux[ice]), 0.0)
        # Ice stress applied in the ocean's -tau convention (force ON ocean = +T_s
        # => feed -T_s so the ocean core's -tau yields +T_s); open-water tau is 0.
        self.assertAlmostEqual(float(sf.tau_x[ice]), -T_s, places=10)

    def test_no_ice_tile_is_full_cell(self):
        # Missing _sfc_state (no ice tile) => f_ocean falls back to 1.0.
        from unittest import mock
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver

        shape = (1, 2)
        z = jnp.zeros(shape)
        stub = types.SimpleNamespace(
            _coupler_cfg=None,
            _last_sfc_response=None,
            _ocean_surface_KuvC=lambda: (jnp.full(shape, 290.0), z, z),
        )
        atm_forcing = types.SimpleNamespace(
            sw_down=jnp.full(shape, 300.0),
            lw_down=jnp.full(shape, 350.0),
            precip_total=jnp.full(shape, 2.0e-5),
        )
        with mock.patch(
            "legoesm.coupler.coupler.ocean_tile_response",
            return_value=_fake_tile(shape),
        ):
            sf, _ = CoupledESMDriver._assemble_ocean_forcing(stub, atm_forcing)
        q_open = 300.0 * (1.0 - 0.06) + 350.0 - 400.0 - 10.0 - 50.0
        self.assertTrue(bool(jnp.allclose(sf.q_net, q_open, atol=1e-9)))


if __name__ == "__main__":
    unittest.main()
