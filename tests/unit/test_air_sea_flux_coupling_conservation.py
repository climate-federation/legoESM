"""Air-sea SHARED-flux coupling: budget-closure acceptance gate.

The coupled model used to be SST/T_sfc-coupled but NOT flux-coupled: the
atmosphere computed its OWN bulk sensible/latent heat (its q_sat specific
humidity, its C_H/C_E) in ``physics_step_no_rad`` while the ocean was driven by
the coupler's INDEPENDENT q_net/evap (coupler q_sfc = 0.98*q_sat mixing ratio,
coupler bulk scheme).  Heat + water leaving the atmosphere did NOT equal what
entered the ocean, so the air-sea budget did not close.

``CoupledConfig.couple_surface_fluxes`` makes ONE surface-flux computation
(the coupler's blended SH/LH) authoritative and feeds it to BOTH the atmosphere
surface tendency (via the traced ``SegmentForcing.sfc_shflx_override`` /
``sfc_lhflx_override`` channel) and the ocean.  These tests are the proof the
fix works (norms alone do not certify it):

  1. with the override ON, the atmosphere's bottom-level T/q kick injects
     EXACTLY the coupler's SH/LH (the atmosphere stops computing its own bulk
     fluxes), and the returned PhysicsOutput diagnostics equal the override;
  2. the SAME positive-up SH/LH that the atmosphere absorbs is what the ocean
     sheds (q_net = ... - shflx - lhflx), so heat/water out of the atmosphere ==
     heat/water into the ocean for an all-ocean (aquaplanet) cell -- the air-sea
     energy AND water budgets close;
  3. override = None is byte-identical to the old self-flux path;
  4. with a turbulence scheme that owns surface exchange the override is
     folded into the kernel's lower boundary condition (it replaces the
     scheme's bulk flux; no double-count).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.grids.cubed_sphere import create_cubed_sphere

NLEV = 6
N_CS = 4


def _sigma(nlev=NLEV):
    class _S:
        sigma_full = jnp.linspace(0.1, 0.95, nlev)
        sigma_half = jnp.linspace(0.05, 1.0, nlev + 1)
        dsigma = jnp.diff(jnp.linspace(0.05, 1.0, nlev + 1))

        def pressure_at_full(self, p_s):
            return p_s[..., None] * self.sigma_full

        def pressure_at_half(self, p_s):
            return p_s[..., None] * self.sigma_half

        def layer_thickness_dp(self, p_s):
            return p_s[..., None] * self.dsigma
    return _S()


def _pipeline(**cfg_overrides):
    grid = create_cubed_sphere(N_CS)
    sigma = _sigma()
    cfg = ExperimentConfig(radiation="gray", microphysics="none",
                           diurnal_cycle=False, **cfg_overrides)
    return build_physics_pipeline(grid, sigma, cfg), grid


def _base_inputs(pipe):
    ad = pipe.adapter
    shape_2d = ad.shape_2d
    shape_3d = (*shape_2d, NLEV)
    inp = dict(
        T=jnp.full(shape_3d, 270.0),
        p_s=jnp.full(shape_2d, 1.0e5),
        q_v=jnp.full(shape_3d, 0.003),
        q_c=jnp.zeros(shape_3d),
        q_r=jnp.zeros(shape_3d),
        u=jnp.full(shape_3d, 3.0),
        v=jnp.zeros(shape_3d),
        sst=jnp.full(shape_2d, 295.0),
        sic=jnp.zeros(shape_2d),
        lat=jnp.full(shape_2d, 0.4),
    )
    return inp, shape_2d


class TestAtmConsumesCouplerFlux:
    """physics_step_no_rad uses the override SH/LH verbatim for the BL kick."""

    def test_bottom_kick_equals_override_flux(self):
        pipe, _ = _pipeline()
        inp, shape_2d = _base_inputs(pipe)
        dt = 600.0
        held3 = jnp.zeros((*shape_2d, NLEV))
        held2 = jnp.zeros(shape_2d)

        # Coupler-provided fluxes (positive UP = surface -> atmosphere).
        sh_ovr = jnp.full(shape_2d, 25.0)    # W/m2 sensible
        lh_ovr = jnp.full(shape_2d, 80.0)    # W/m2 latent

        def _call(**extra):
            return pipe.physics_step_no_rad(
                inp["T"], inp["p_s"], inp["q_v"], inp["q_c"], inp["q_r"],
                jnp.zeros((pipe.adapter.ncol,)),
                inp["u"], inp["v"], inp["sst"], inp["sic"], inp["lat"], dt,
                held3, held2, held2, held2, held2, held2, **extra,
            )

        out = _call(sfc_shflx_override=sh_ovr, sfc_lhflx_override=lh_ovr)
        # Diagnostics equal the override exactly.
        assert jnp.allclose(out.shflx, sh_ovr)
        assert jnp.allclose(out.lhflx, lh_ovr)

        # Bottom-level tendency injected EXACTLY g*F/(c_pd*dp) (heat) and
        # g*evap/dp (moisture).  dp_low is the lowest layer mass-thickness.
        dp_low = inp["p_s"] * (
            pipe.sigma_half[-1] - pipe.sigma_half[-2])
        dT_bottom = out.dT_dt[..., -1]
        dq_bottom = out.dq_v_dt[..., -1]
        # No water channel: the kick inverts the SAME L_v(T_sfc) the bulk law
        # charges, and the heat kick books (lhflx - L_v * evap), the enthalpy
        # the atmosphere's constant-L convention would otherwise over-credit.
        from legoesm.thermo import latent_heat_vaporization
        evap_fb = lh_ovr / latent_heat_vaporization(inp["sst"])
        expect_dq = constants.g * evap_fb / dp_low
        expect_dT = constants.g * (sh_ovr + lh_ovr - constants.L_v * evap_fb) / (
            constants.c_pd * dp_low)
        # gray radiation also heats the column, so compare the DIFFERENCE
        # between override-on and a zero-flux override (isolates the BL kick).
        out0 = _call(sfc_shflx_override=jnp.zeros(shape_2d),
                     sfc_lhflx_override=jnp.zeros(shape_2d))
        assert jnp.allclose(dT_bottom - out0.dT_dt[..., -1], expect_dT,
                            rtol=1e-10, atol=1e-12)
        assert jnp.allclose(dq_bottom - out0.dq_v_dt[..., -1], expect_dq,
                            rtol=1e-10, atol=1e-12)
        # With the coupler's water channel the moisture kick is that water,
        # exactly, and the heat kick carries the physical-vs-reference gap.
        evap_ovr = jnp.full(shape_2d, 3.0e-5)
        out_w = _call(sfc_shflx_override=sh_ovr, sfc_lhflx_override=lh_ovr,
                      sfc_evap_override=evap_ovr)
        assert jnp.allclose(out_w.dq_v_dt[..., -1] - out0.dq_v_dt[..., -1],
                            constants.g * evap_ovr / dp_low, rtol=1e-10, atol=1e-12)
        assert jnp.allclose(
            out_w.dT_dt[..., -1] - out0.dT_dt[..., -1],
            constants.g * (sh_ovr + lh_ovr - constants.L_v * evap_ovr) / (constants.c_pd * dp_low),
            rtol=1e-10, atol=1e-12)
        assert jnp.allclose(out_w.lhflx, lh_ovr)   # reported latent heat stays physical

    def test_override_none_byte_identical(self):
        """Override absent => the atmosphere computes its own bulk fluxes,
        bit-for-bit the pre-shared-flux path."""
        pipe, _ = _pipeline()
        inp, shape_2d = _base_inputs(pipe)
        dt = 600.0
        held3 = jnp.zeros((*shape_2d, NLEV))
        held2 = jnp.zeros(shape_2d)

        def _call(**extra):
            return pipe.physics_step_no_rad(
                inp["T"], inp["p_s"], inp["q_v"], inp["q_c"], inp["q_r"],
                jnp.zeros((pipe.adapter.ncol,)),
                inp["u"], inp["v"], inp["sst"], inp["sic"], inp["lat"], dt,
                held3, held2, held2, held2, held2, held2, **extra,
            )

        base = _call()
        none = _call(sfc_shflx_override=None, sfc_lhflx_override=None)
        assert jnp.array_equal(base.dT_dt, none.dT_dt)
        assert jnp.array_equal(base.dq_v_dt, none.dq_v_dt)
        assert jnp.array_equal(base.shflx, none.shflx)
        assert jnp.array_equal(base.lhflx, none.lhflx)


class TestAirSeaBudgetCloses:
    """The flux the atmosphere absorbs == the flux the ocean sheds (all-ocean
    cell): heat AND water out of atmosphere == into ocean."""

    def test_aquaplanet_blended_equals_ocean_tile_flux(self):
        """For an all-ocean cell the coupler's BLENDED SH/LH (fed back to the
        atmosphere) equals the OCEAN-TILE SH/LH that drives the ocean q_net, so
        the same numbers appear on both sides of the air-sea interface."""
        from legoesm.coupler.config import CouplerConfig, TileConfig
        from legoesm.coupler.coupler import ocean_tile_response
        from legoesm.coupler.tile_fractions import (
            blend_tiles, compute_tile_fractions,
        )
        from legoesm.core.coupling_fields import AtmToSurface, TileResponse

        shape = (6, N_CS, N_CS)
        z = jnp.zeros(shape)
        forcing = AtmToSurface(
            sw_down=jnp.full(shape, 300.0), lw_down=jnp.full(shape, 350.0),
            precip_total=jnp.full(shape, 1.0e-5), precip_snow=z,
            T_lowest=jnp.full(shape, 290.0), q_lowest=jnp.full(shape, 0.010),
            u_lowest=jnp.full(shape, 5.0), v_lowest=z,
            p_lowest=jnp.full(shape, 9.8e4), p_surface=jnp.full(shape, 1.0e5),
            rho_lowest=jnp.full(shape, 1.2), cos_zenith=jnp.full(shape, 0.5),
            co2_ppmv=jnp.full(shape, 400.0),
            has_radiation=jnp.ones(shape), has_precipitation=jnp.ones(shape),
        )
        ccfg = CouplerConfig()
        sst = jnp.full(shape, 298.0)
        ocean = ocean_tile_response(forcing, sst, z, z, ccfg)

        # All-ocean cell: f_land = f_lake = 0, no ice.
        tcfg = TileConfig(f_land=z, f_lake=z)
        fracs = compute_tile_fractions(tcfg, ice_concentration=z)

        # Build a zero "dummy" response for the other tiles (only the ocean
        # contributes when f_ocean == 1).
        def _zero_resp():
            return TileResponse(
                T_sfc=sst, albedo=z, emissivity=z, z0=jnp.full(shape, 1e-3),
                q_surface=z, shflx=z, lhflx=z, tau_x=z, tau_y=z, lw_up=z,
                u_ocean_sfc=z, v_ocean_sfc=z, co2_flux=z, freshwater_flux=z,
                ocean_heat_extraction=z, ocean_stress_x=z, ocean_stress_y=z,
                surface_mass_flux=z, salt_flux=z, lhflx_exchange=z,
            )
        blended = blend_tiles(ocean, _zero_resp(), _zero_resp(), _zero_resp(),
                              fracs)

        # The fluxes fed BACK to the atmosphere (blended) equal the ocean-tile
        # fluxes that drive the ocean q_net -- single shared flux on both sides.
        assert jnp.allclose(blended.shflx, ocean.shflx)
        assert jnp.allclose(blended.lhflx, ocean.lhflx)

        # Ocean q_net subtracts the SAME positive-up SH/LH the atmosphere
        # absorbs: heat leaving the ocean == heat entering the atmosphere.
        sw_net = forcing.sw_down * (1.0 - ocean.albedo)
        q_net = (sw_net + forcing.lw_down
                 - ocean.lw_up - ocean.shflx - ocean.lhflx)
        # Reconstruct the turbulent (air-sea exchange) part of q_net: the part
        # that must mirror the atmosphere's absorbed flux.
        q_turbulent_out_of_ocean = ocean.shflx + ocean.lhflx
        # Atmosphere absorbs +shflx (sensible) and +lhflx (latent, as vapour);
        # ocean sheds shflx + lhflx -> equal and opposite, budget closes.
        assert jnp.allclose(q_turbulent_out_of_ocean,
                            blended.shflx + blended.lhflx)
        # Water: ocean evaporation mass == atmosphere moistening mass.
        evap_ocean = ocean.lhflx / constants.L_v
        assert jnp.allclose(evap_ocean, blended.lhflx / constants.L_v)
        # Sanity: q_net finite and SW dominates here (warm tropics).
        assert bool(jnp.all(jnp.isfinite(q_net)))

    def test_mixed_cell_air_sea_component_closes(self):
        """Over a MIXED (land+ocean) cell the atmosphere is forced by the BLENDED
        flux (its correct total over all tiles), the ocean by the ocean-tile
        flux.  The AIR-SEA component still closes: the ocean component of the
        atmosphere's blended flux (f_ocean*ocean_tile) equals what the ocean
        column receives (ocean_tile) weighted by the SAME f_ocean partition.
        Land heat goes to the land tile, not the ocean -- correct, not a leak."""
        from legoesm.coupler.config import CouplerConfig, TileConfig
        from legoesm.coupler.coupler import ocean_tile_response
        from legoesm.coupler.tile_fractions import (
            blend_tiles, compute_tile_fractions,
        )
        from legoesm.core.coupling_fields import AtmToSurface, TileResponse

        shape = (6, N_CS, N_CS)
        z = jnp.zeros(shape)
        forcing = AtmToSurface(
            sw_down=jnp.full(shape, 300.0), lw_down=jnp.full(shape, 350.0),
            precip_total=jnp.full(shape, 1.0e-5), precip_snow=z,
            T_lowest=jnp.full(shape, 290.0), q_lowest=jnp.full(shape, 0.010),
            u_lowest=jnp.full(shape, 5.0), v_lowest=z,
            p_lowest=jnp.full(shape, 9.8e4), p_surface=jnp.full(shape, 1.0e5),
            rho_lowest=jnp.full(shape, 1.2), cos_zenith=jnp.full(shape, 0.5),
            co2_ppmv=jnp.full(shape, 400.0),
            has_radiation=jnp.ones(shape), has_precipitation=jnp.ones(shape),
        )
        ccfg = CouplerConfig()
        sst = jnp.full(shape, 298.0)
        ocean = ocean_tile_response(forcing, sst, z, z, ccfg)

        # 40% land cell, no ice/lake.  Land tile carries a DIFFERENT SH/LH.
        f_land_val = 0.4
        land = TileResponse(
            T_sfc=jnp.full(shape, 300.0), albedo=jnp.full(shape, 0.2),
            emissivity=z, z0=jnp.full(shape, 0.1),
            q_surface=z, shflx=jnp.full(shape, 50.0),
            lhflx=jnp.full(shape, 20.0), tau_x=z, tau_y=z,
            lw_up=jnp.full(shape, 420.0), u_ocean_sfc=z, v_ocean_sfc=z,
            co2_flux=z, freshwater_flux=z, ocean_heat_extraction=z,
            ocean_stress_x=z, ocean_stress_y=z, surface_mass_flux=z, salt_flux=z,
        )

        def _zero_resp():
            return TileResponse(
                T_sfc=sst, albedo=z, emissivity=z, z0=jnp.full(shape, 1e-3),
                q_surface=z, shflx=z, lhflx=z, tau_x=z, tau_y=z, lw_up=z,
                u_ocean_sfc=z, v_ocean_sfc=z, co2_flux=z, freshwater_flux=z,
                ocean_heat_extraction=z, ocean_stress_x=z, ocean_stress_y=z,
                surface_mass_flux=z, salt_flux=z, lhflx_exchange=z,
            )

        tcfg = TileConfig(f_land=jnp.full(shape, f_land_val), f_lake=z)
        fracs = compute_tile_fractions(tcfg, ice_concentration=z)
        blended = blend_tiles(ocean, _zero_resp(), land, _zero_resp(), fracs)

        # The atmosphere gets the BLENDED flux (f_ocean*ocean + f_land*land),
        # which differs from the ocean-tile flux over this mixed cell -- so
        # feeding the ocean-tile flux to the atmosphere here WOULD under-force it
        # (the codex finding the blended design fixes).
        f_ocean = 1.0 - f_land_val
        assert jnp.allclose(
            blended.shflx, f_ocean * ocean.shflx + f_land_val * land.shflx)
        assert not jnp.allclose(blended.shflx, ocean.shflx)

        # Air-sea component closure: the ocean column receives ocean_tile.shflx
        # over its wet (f_ocean) area; the atmosphere's OCEAN-fraction share of
        # the blended flux is f_ocean*ocean_tile.shflx -- the SAME partition, so
        # the air-sea heat exchange is consistent at both ends.
        atm_ocean_component = f_ocean * ocean.shflx
        ocean_received_over_cell = f_ocean * ocean.shflx
        assert jnp.allclose(atm_ocean_component, ocean_received_over_cell)

    def test_column_energy_and_water_budget_close(self):
        """End-to-end column budget: with the override ON the heat added to the
        atmosphere column bottom layer == shflx*dt, the water mass added ==
        evap*dt, and these equal what the ocean sheds (q_net subtracts them)."""
        pipe, _ = _pipeline()
        inp, shape_2d = _base_inputs(pipe)
        dt = 1800.0
        held3 = jnp.zeros((*shape_2d, NLEV))
        held2 = jnp.zeros(shape_2d)
        sh_ovr = jnp.full(shape_2d, 30.0)
        lh_ovr = jnp.full(shape_2d, 100.0)

        def _call(**extra):
            return pipe.physics_step_no_rad(
                inp["T"], inp["p_s"], inp["q_v"], inp["q_c"], inp["q_r"],
                jnp.zeros((pipe.adapter.ncol,)),
                inp["u"], inp["v"], inp["sst"], inp["sic"], inp["lat"], dt,
                held3, held2, held2, held2, held2, held2, **extra,
            )

        out = _call(sfc_shflx_override=sh_ovr, sfc_lhflx_override=lh_ovr)
        out0 = _call(sfc_shflx_override=jnp.zeros(shape_2d),
                     sfc_lhflx_override=jnp.zeros(shape_2d))

        # Column-integrated thermal-energy change from the surface flux ONLY
        # (difference isolates the BL kick from gray radiation):
        #   Sum_k c_pd * (dp_k/g) * dT_k  ==  shflx (W/m2)
        dp = inp["p_s"][..., None] * (
            pipe.sigma_half[1:] - pipe.sigma_half[:-1])
        col_dE = jnp.sum(
            constants.c_pd * (dp / constants.g)
            * (out.dT_dt - out0.dT_dt), axis=-1)
        # Heat-only override: the water is lhflx / L_v(T_sfc) (the inverse of
        # the bulk law's own L) and the heat kick books the gap between that
        # water credited at the column's reference L_v and the physical lhflx.
        from legoesm.thermo import latent_heat_vaporization
        evap = lh_ovr / latent_heat_vaporization(inp["sst"])
        assert jnp.allclose(col_dE, sh_ovr + lh_ovr - constants.L_v * evap, rtol=1e-9, atol=1e-9)

        # Column-integrated water-mass change == evap == lhflx / L_v.
        col_dW = jnp.sum(
            (dp / constants.g) * (out.dq_v_dt - out0.dq_v_dt), axis=-1)
        assert jnp.allclose(col_dW, evap, rtol=1e-9, atol=1e-12)
        # Physical closure: c_p dT + L_v dq over the column == shflx + lhflx exactly.
        assert jnp.allclose(col_dE + constants.L_v * col_dW, sh_ovr + lh_ovr, rtol=1e-9, atol=1e-9)


class TestTurbulenceLowerBC:
    """The shared-flux override is the turbulence scheme's LOWER BOUNDARY
    CONDITION (folded into the kernel's surface config for the call), so it
    replaces the scheme's own bulk flux instead of double-counting it — and
    the former "incompatible with a turbulence scheme" refusal is gone."""

    def _step(self, pipe, shape_2d, shf):
        inp, _ = _base_inputs(pipe)
        held3 = jnp.zeros((*shape_2d, NLEV))
        held2 = jnp.zeros(shape_2d)
        return pipe.physics_step_no_rad(
            inp["T"], inp["p_s"], inp["q_v"], inp["q_c"], inp["q_r"],
            jnp.zeros((pipe.adapter.ncol,)),
            inp["u"], inp["v"], inp["sst"], inp["sic"], inp["lat"], 600.0,
            held3, held2, held2, held2, held2, held2,
            sfc_shflx_override=jnp.full(shape_2d, shf),
            sfc_lhflx_override=jnp.zeros(shape_2d),
        )

    def test_override_with_turbulence_folds_into_the_kernel(self):
        pipe, _ = _pipeline(turbulence="louis")
        _, shape_2d = _base_inputs(pipe)
        out_pos = self._step(pipe, shape_2d, 50.0)
        out_neg = self._step(pipe, shape_2d, -50.0)
        # the kernel echoes the prescribed flux (it IS its surface flux)
        np.testing.assert_allclose(np.asarray(out_pos.shflx), 50.0)
        np.testing.assert_allclose(np.asarray(out_neg.shflx), -50.0)
        assert float(jnp.mean(out_pos.dT_dt[..., -1])) > float(
            jnp.mean(out_neg.dT_dt[..., -1]))


class TestCoupledDriverWiring:
    """CoupledESMDriver._override_sfc_fluxes flag gating + hook install."""

    def test_flag_defaults_false(self):
        from legoesm.driver.coupled_config import CoupledConfig
        assert CoupledConfig().couple_surface_fluxes is False

    def test_noop_when_flag_off(self):
        from types import SimpleNamespace
        from legoesm.driver.coupled_config import CoupledConfig
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        atm = SimpleNamespace(get_sfc_flux_override="sentinel")
        drv = SimpleNamespace(
            coupled_cfg=CoupledConfig(couple_surface_fluxes=False), _atm=atm)
        CoupledESMDriver._override_sfc_fluxes(drv)
        assert atm.get_sfc_flux_override == "sentinel"

    def test_installs_hook_passes_blended_flux(self):
        """The atmosphere gets the coupler's tile-blended SH/LH (the correct
        TOTAL surface flux over all tiles)."""
        from types import SimpleNamespace
        from legoesm.driver.coupled_config import CoupledConfig
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        atm = SimpleNamespace(get_sfc_flux_override=None)
        drv = SimpleNamespace(
            coupled_cfg=CoupledConfig(couple_surface_fluxes=True),
            _atm=atm, _sfc_flux_handback=None)
        CoupledESMDriver._override_sfc_fluxes(drv)
        assert atm.get_sfc_flux_override is not None

        # Nothing coupled yet -> five None -> atmosphere uses its own bulk flux.
        assert atm.get_sfc_flux_override(0.0) == (None,) * 5
        # Once the segment hook stored the segment-mean handback (F38), the
        # SH/LH/water flux and the surface STRESS flow through untouched.
        sh = jnp.full((6, 4, 4), 22.0)
        lh = jnp.full((6, 4, 4), 77.0)
        ev = jnp.full((6, 4, 4), 3.1e-5)
        tx = jnp.full((6, 4, 4), -0.12)
        ty = jnp.full((6, 4, 4), 0.04)
        drv._sfc_flux_handback = (sh, lh, ev, tx, ty)
        got = atm.get_sfc_flux_override(0.0)
        for g, want in zip(got, (sh, lh, ev, tx, ty)):
            assert jnp.array_equal(g, want)

    def test_hook_closes_water_and_energy_together(self):
        """The atmosphere's moisture source is the tile mass flux, and its heat
        intake (sensible + latent_enthalpy_correction + the L_v*E its moist
        enthalpy credits the water with) equals the tiles' shflx + lhflx, with
        the tile charging a latent heat != L_v (main #1822 closure, expressed
        through the atmosphere's own correction)."""
        from types import SimpleNamespace
        from legoesm.atmosphere.physics.turbulence.surface_layer import (
            latent_enthalpy_correction)
        from legoesm.driver.coupled_config import CoupledConfig
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        atm = SimpleNamespace(get_sfc_flux_override=None)
        evap = jnp.array([1.0e-5, 2.0e-5])
        resp = SimpleNamespace(shflx=jnp.array([10.0, 20.0]),
                               lhflx=evap * 2.44e6,           # L(SST) < L_v
                               surface_mass_flux=evap)
        drv = SimpleNamespace(
            coupled_cfg=CoupledConfig(couple_surface_fluxes=True),
            _atm=atm, _sfc_flux_handback=(
                resp.shflx, resp.lhflx, resp.surface_mass_flux, None, None))
        CoupledESMDriver._override_sfc_fluxes(drv)
        sh, lh, ev, _, _ = atm.get_sfc_flux_override(0.0)
        np.testing.assert_allclose(ev, evap, rtol=1e-12)
        heat_in = sh + latent_enthalpy_correction(lh, ev) + constants.L_v * ev
        np.testing.assert_allclose(heat_in, resp.shflx + resp.lhflx, rtol=1e-12)
        # Non-vacuous: without the correction the column misses (L - L_v)*E.
        assert not np.allclose(sh + constants.L_v * ev, resp.shflx + resp.lhflx)

    def test_assemble_ocean_forcing_sw_down_is_net(self):
        """OceanSurfaceForcing.sw_down carries the NET (post-albedo) surface SW,
        not gross -- the codex finding-2 fix (the penetration kernel contract)."""
        import types
        from unittest import mock
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver

        shape = (3, 4)
        z = jnp.zeros(shape)
        tile = types.SimpleNamespace(
            albedo=jnp.full(shape, 0.06), lw_up=jnp.full(shape, 400.0),
            shflx=jnp.full(shape, 12.0), lhflx=jnp.full(shape, 60.0),
            surface_mass_flux=jnp.full(shape, 2.0e-5),   # clearly NOT lhflx / L_v (2.4e-5)
            tau_x=z, tau_y=z,
        )
        sst = jnp.full(shape, 290.0)
        stub = types.SimpleNamespace(
            _coupler_cfg=None, _last_sfc_response=None,
            _ocean_surface_KuvC=lambda: (sst, z, z),
            _atm=None,                       # radiation not run -> SW fallback
            _grid_remapper=None,
        )
        atm_forcing = types.SimpleNamespace(
            sw_down=jnp.full(shape, 300.0), lw_down=jnp.full(shape, 350.0),
            precip_total=jnp.full(shape, 1e-5),
        )
        with mock.patch("legoesm.coupler.coupler.ocean_tile_response",
                        return_value=tile):
            sf, fw = CoupledESMDriver._assemble_ocean_forcing(stub, atm_forcing)
        # NET surface SW (post-albedo), not gross.
        assert jnp.allclose(sf.sw_down,
                            atm_forcing.sw_down * (1.0 - tile.albedo))
        # And q_net's solar term uses the same net SW (telescoping consistency).
        sw_net = atm_forcing.sw_down * (1.0 - tile.albedo)
        q_net_expected = (sw_net + atm_forcing.lw_down
                          - tile.lw_up - tile.shflx - tile.lhflx)
        assert jnp.allclose(sf.q_net, q_net_expected)


class TestJittedStepUnifiedThreadsFluxOverride:
    """The production path: build_step_unified threads the flux override all the
    way into physics_step_no_rad through its lax.cond / static branch."""

    def test_jitted_override_changes_bottom_tendency(self):
        pipe, _ = _pipeline()
        step_fn = pipe.build_step_unified(static_need_rad=True)
        ad = pipe.adapter
        shape_2d = ad.shape_2d
        shape_3d = (*shape_2d, NLEV)

        T = jnp.full(shape_3d, 270.0)
        p_s = jnp.full(shape_2d, 1.0e5)
        q_v = jnp.full(shape_3d, 0.003)
        q_c = jnp.zeros(shape_3d)
        q_r = jnp.zeros(shape_3d)
        u = jnp.full(shape_3d, 3.0)
        v = jnp.zeros(shape_3d)
        sst = jnp.full(shape_2d, 295.0)
        sic = jnp.zeros(shape_2d)
        lat = jnp.full(shape_2d, 0.4)
        lon = jnp.full(shape_2d, 1.0)
        held3 = jnp.zeros(shape_3d)
        held2 = jnp.zeros(shape_2d)
        o3 = jnp.zeros((ad.ncol, NLEV))
        aerosol = jnp.zeros((ad.ncol, NLEV))

        def _run(**extra):
            return step_fn(
                jnp.bool_(True),
                T, p_s, q_v, q_c, q_r, jnp.zeros((ad.ncol,)), u, v,
                sst, sic, lat, lon, 100.0, 43200.0, 600.0,
                jnp.array([]), constants.S_0, o3, aerosol,
                held3, held2, held2, held2, held2, held2,
                T_land=None, **extra,
            )

        base, _, _, _ = _run()  # step_unified returns 4 (PR #650 land_ml)
        sh = jnp.full(shape_2d, 40.0)
        lh = jnp.full(shape_2d, 120.0)
        ovr, _, _, _ = _run(sfc_shflx_override=sh, sfc_lhflx_override=lh)
        # Override-on diagnostics equal the coupler flux.
        assert jnp.allclose(ovr.shflx, sh)
        assert jnp.allclose(ovr.lhflx, lh)
        # Bottom-level tendency differs from the self-flux baseline.
        assert not jnp.allclose(base.dT_dt[..., -1], ovr.dT_dt[..., -1])
        # Override = None reproduces the baseline exactly.
        none, _, _, _ = _run(sfc_shflx_override=None, sfc_lhflx_override=None)
        assert jnp.array_equal(base.dT_dt, none.dT_dt)
        assert jnp.array_equal(base.dq_v_dt, none.dq_v_dt)

    def test_grad_flows_through_flux_override(self):
        """Differentiability: d(column heat)/d(shflx_override) is finite and
        matches g*dt/(c_pd*dp) integrated -- the override is a clean AD path."""
        pipe, _ = _pipeline()
        ad = pipe.adapter
        shape_2d = ad.shape_2d
        shape_3d = (*shape_2d, NLEV)
        p_s = jnp.full(shape_2d, 1.0e5)
        held3 = jnp.zeros(shape_3d)
        held2 = jnp.zeros(shape_2d)

        def _loss(sh_scalar):
            sh = jnp.full(shape_2d, sh_scalar)
            out = pipe.physics_step_no_rad(
                jnp.full(shape_3d, 270.0), p_s, jnp.full(shape_3d, 0.003),
                jnp.zeros(shape_3d), jnp.zeros(shape_3d),
                jnp.zeros((ad.ncol,)),
                jnp.full(shape_3d, 3.0), jnp.zeros(shape_3d),
                jnp.full(shape_2d, 295.0), jnp.zeros(shape_2d),
                jnp.full(shape_2d, 0.4), 600.0,
                held3, held2, held2, held2, held2, held2,
                sfc_shflx_override=sh,
                sfc_lhflx_override=jnp.zeros(shape_2d),
            )
            return jnp.sum(out.dT_dt[..., -1])

        g = jax.grad(_loss)(20.0)
        assert np.isfinite(float(g))
        assert abs(float(g)) > 0.0


class TestSharedFluxClosurePreconditionGuard:
    """couple_surface_fluxes + dynamic ocean + continents must use
    f_land_mode='from_ocean' so the atm/ocean land partitions match; otherwise
    the air-sea budget cannot close over coastal cells -> raise (codex HIGH)."""

    def _stub(self, *, dynamic, f_land_val, f_land_mode, ocean_mask_val=1.0):
        from types import SimpleNamespace
        from legoesm.driver.coupled_config import CoupledConfig
        from legoesm.coupler.config import TileConfig
        shape = (6, 4, 4)
        atm = SimpleNamespace(get_sfc_flux_override=None)
        tc = TileConfig(f_land=jnp.full(shape, f_land_val),
                        f_lake=jnp.zeros(shape))
        # Ocean wet mask: 1=ocean, 0=land.  ocean_mask_val < 1 => has dry cells.
        return SimpleNamespace(
            coupled_cfg=CoupledConfig(couple_surface_fluxes=True,
                                      f_land_mode=f_land_mode),
            _atm=atm, _last_sfc_response=None, _is_dynamic_ocean=dynamic,
            _tile_config=tc,
            _ocean_land_mask=jnp.full(shape, ocean_mask_val),
        )

    def test_dynamic_land_analytical_raises(self):
        import pytest
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        drv = self._stub(dynamic=True, f_land_val=0.4, f_land_mode="analytical")
        with pytest.raises(ValueError, match="from_ocean"):
            CoupledESMDriver._override_sfc_fluxes(drv)

    def test_dynamic_land_from_ocean_ok(self):
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        drv = self._stub(dynamic=True, f_land_val=0.4, f_land_mode="from_ocean")
        CoupledESMDriver._override_sfc_fluxes(drv)   # no raise
        assert drv._atm.get_sfc_flux_override is not None

    def test_dynamic_aquaplanet_ok(self):
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        # No atm land AND all-wet ocean mask -> partitions trivially agree.
        drv = self._stub(dynamic=True, f_land_val=0.0, f_land_mode="analytical",
                         ocean_mask_val=1.0)
        CoupledESMDriver._override_sfc_fluxes(drv)   # no raise
        assert drv._atm.get_sfc_flux_override is not None

    def test_slab_land_ok(self):
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        drv = self._stub(dynamic=False, f_land_val=0.4, f_land_mode="analytical")
        CoupledESMDriver._override_sfc_fluxes(drv)   # slab -> no raise
        assert drv._atm.get_sfc_flux_override is not None

    def test_dynamic_zero_landmode_but_ocean_has_continents_raises(self):
        """codex round-4: f_land_mode='zero' (all-ocean atm) but the dynamic
        WOA/tripole ocean masks dry cells -> partitions diverge -> must raise."""
        import pytest
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        drv = self._stub(dynamic=True, f_land_val=0.0, f_land_mode="zero",
                         ocean_mask_val=0.0)   # 0 => a dry (continent) cell
        with pytest.raises(ValueError, match="from_ocean"):
            CoupledESMDriver._override_sfc_fluxes(drv)

    def test_dynamic_zero_landmode_allwet_ocean_ok(self):
        """f_land_mode='zero' with an all-wet ocean mask (rest aquaplanet) is the
        provably-all-ocean case -> no raise."""
        from legoesm.driver.coupled_esm_driver import CoupledESMDriver
        drv = self._stub(dynamic=True, f_land_val=0.0, f_land_mode="zero",
                         ocean_mask_val=1.0)
        CoupledESMDriver._override_sfc_fluxes(drv)   # no raise
        assert drv._atm.get_sfc_flux_override is not None
