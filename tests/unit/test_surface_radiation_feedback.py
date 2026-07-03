"""Cluster-A wiring: coupler dynamic surface albedo / skin-T -> atm radiation.

The ocean/sea-ice/land tile models compute dynamic surface albedo (sea-ice melt
feedback, zenith ocean albedo, snow brightening) and a prognostic skin
temperature, which the coupler tile-blends.  Previously these were silently
dropped — the atmosphere radiation used frozen config scalars, and the feedback
hook (``driver._sfc_T_override``) was a dead attribute ModelDriver never read.

These tests pin the new traced ``SegmentForcing`` channel and the opt-in
coupled-driver wiring (default off ⇒ byte-identical to the pre-feedback runs).
The actual radiation use of the override is pinned in
``test_land_surface.py`` (compute_radiation_core + step_unified).
"""
from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from legoesm import constants
from legoesm.driver.compiled_segments import SegmentForcing, pack_forcing
from legoesm.driver.coupled_config import CoupledConfig
from legoesm.driver.coupled_esm_driver import CoupledESMDriver


class TestSegmentForcingSurfaceOverride:
    def test_pack_forcing_defaults_none(self):
        f = pack_forcing(
            sst=0.0, sic=0.0, day_of_year=1.0, seconds_of_day=0.0,
            solar_weights=[1.0], s_0=1361.0, o3_vmr=[0.0], aerosol_od=[0.0],
        )
        assert f.sfc_albedo_override is None
        assert f.sfc_T_override is None
        assert "sfc_albedo_override" in SegmentForcing._fields
        assert "sfc_T_override" in SegmentForcing._fields

    def test_pack_forcing_carries_override(self):
        alb = jnp.full((2, 2), 0.5)
        T = jnp.full((2, 2), 250.0)
        f = pack_forcing(
            sst=jnp.zeros((2, 2)), sic=jnp.zeros((2, 2)),
            day_of_year=1.0, seconds_of_day=0.0,
            solar_weights=[1.0], s_0=1361.0,
            o3_vmr=[0.0], aerosol_od=[0.0],
            sfc_albedo_override=alb, sfc_T_override=T,
        )
        assert jnp.array_equal(f.sfc_albedo_override, alb)
        assert jnp.array_equal(f.sfc_T_override, T)


class TestCoupledConfigFlag:
    def test_flag_defaults_false(self):
        assert CoupledConfig().couple_surface_radiation is False


class TestOverrideSfc:
    """``CoupledESMDriver._override_sfc`` flag gating + seed + passthrough.

    Exercised on a lightweight stub (the method only touches coupled_cfg,
    _atm, _last_sfc_response, atm_config) to avoid a full coupled setup."""

    def _atm_config(self, radiation="rrtmgp"):
        return SimpleNamespace(albedo_ice=0.65, albedo_ocean=0.06,
                               T_ice=271.35, radiation=radiation)

    def _drv(self, atm, radiation="rrtmgp"):
        return SimpleNamespace(
            coupled_cfg=CoupledConfig(couple_surface_radiation=True),
            _atm=atm, _last_sfc_response=None,
            atm_config=self._atm_config(radiation),
        )

    def _physics_mock(self, f_land=None):
        # Lightweight PhysicsPipeline carrying only the attrs
        # ``static_surface_emissivity`` reads (the RRTMGP seed now reconstructs
        # the EXACT static blend radiation emits with via this method).
        from legoesm.driver.physics_pipeline import PhysicsPipeline
        p = PhysicsPipeline.__new__(PhysicsPipeline)
        p.emissivity_ice = constants.emissivity_ice
        p.emissivity_ocean = constants.emissivity_ocean
        p.emissivity_land = 0.96
        p.f_land = f_land
        return p

    def _atm(self, sst, sic, f_land=None):
        return SimpleNamespace(
            get_sfc_override=None, get_sst_sic=lambda d: (sst, sic),
            physics=self._physics_mock(f_land=f_land),
        )

    def test_noop_when_flag_off(self):
        atm = SimpleNamespace(get_sfc_override="sentinel")
        drv = SimpleNamespace(
            coupled_cfg=CoupledConfig(couple_surface_radiation=False),
            _atm=atm,
        )
        CoupledESMDriver._override_sfc(drv)
        # Untouched (still the sentinel, hook not installed).
        assert atm.get_sfc_override == "sentinel"

    def test_seeds_static_blend_before_first_response(self):
        # RRTMGP (conservative) path: seed returns a grid-shaped emissivity array.
        sst = jnp.full((2, 2), 290.0)
        sic = jnp.zeros((2, 2))
        atm = self._atm(sst, sic)
        drv = self._drv(atm, "rrtmgp")
        CoupledESMDriver._override_sfc(drv)
        assert atm.get_sfc_override is not None
        alb, T, emis = atm.get_sfc_override(0.0)
        # Seed = static blend (all-ocean here): albedo 0.06, T_sfc = sst.
        assert alb is not None and T is not None
        assert jnp.allclose(alb, 0.06)
        assert jnp.allclose(T, 290.0)
        # RRTMGP seed emissivity is a GRID-SHAPED static array (NOT None) so the
        # override leaf keeps a constant pytree shape from segment 0 (no
        # recompile).  It is the pipeline's EXACT static blend (so it matches the
        # lw_net inversion); all-ocean here -> emissivity_ocean.
        assert emis is not None and emis.shape == sst.shape
        assert jnp.allclose(emis, constants.emissivity_ocean)

    def test_seed_emissivity_includes_land_blend(self):
        # Pre-first-response, RRTMGP emits with the seed override; that seed must
        # be the SAME emissivity the lw_net inversion reconstructs with — the
        # pipeline's land-inclusive blend — or land cells bias the initial
        # lw_down (codex round-6 regression).
        sst = jnp.full((2, 2), 290.0)
        sic = jnp.zeros((2, 2))
        f_land = jnp.ones((2, 2))   # all land
        atm = self._atm(sst, sic, f_land=f_land)
        drv = self._drv(atm, "rrtmgp")
        CoupledESMDriver._override_sfc(drv)
        _, _, emis = atm.get_sfc_override(0.0)
        # All-land -> the configured land emissivity (0.96), NOT the ocean value.
        assert jnp.allclose(emis, 0.96)
        assert not jnp.allclose(emis, constants.emissivity_ocean)

    def test_passes_through_sfc_response(self):
        # RRTMGP: dynamic albedo + radiative-equivalent T_rad + emissivity flow.
        sst = jnp.full((2, 2), 290.0)
        sic = jnp.zeros((2, 2))
        atm = self._atm(sst, sic)
        drv = self._drv(atm, "rrtmgp")
        CoupledESMDriver._override_sfc(drv)
        drv._last_sfc_response = SimpleNamespace(
            albedo=jnp.full((2, 2), 0.5), T_sfc=jnp.full((2, 2), 250.0),
            T_rad=jnp.full((2, 2), 248.0), emissivity=jnp.full((2, 2), 0.985),
        )
        alb, T, emis = atm.get_sfc_override(0.0)
        assert jnp.array_equal(alb, jnp.full((2, 2), 0.5))
        # RRTMGP gets the radiative-equivalent T_rad (NOT the soil-side T_sfc).
        assert jnp.array_equal(T, jnp.full((2, 2), 248.0))
        assert jnp.array_equal(emis, jnp.full((2, 2), 0.985))

    def test_gray_uses_brightness_temp_and_no_emissivity_override(self):
        # Gray (idealized eps=1) cannot honour the canopy eps_col, so it gets a
        # black-surface BRIGHTNESS temperature from the full upward flux
        # (sigma*T_bb^4 = LW_out) — NOT T_rad (which would over-emit by 1/eps_col)
        # and NOT the aerodynamic/sensible-heat T_sfc (the canopy air-space Tc).
        # No emissivity override — gray keeps its own eps=1.
        sst = jnp.full((2, 2), 290.0)
        sic = jnp.zeros((2, 2))
        atm = self._atm(sst, sic)
        drv = self._drv(atm, "gray")
        CoupledESMDriver._override_sfc(drv)
        # Seed: gray emissivity override is None; T is the static blend (sst).
        alb, T, emis = atm.get_sfc_override(0.0)
        assert emis is None
        assert jnp.allclose(T, 290.0)
        # Response: gray gets the brightness temp T_bb = (lw_up/sigma)^0.25,
        # NOT T_rad (248) or the aerodynamic T_sfc (250); None emissivity.
        lw_up = jnp.full((2, 2), 390.0)
        drv._last_sfc_response = SimpleNamespace(
            albedo=jnp.full((2, 2), 0.5), T_sfc=jnp.full((2, 2), 250.0),
            T_rad=jnp.full((2, 2), 248.0), emissivity=jnp.full((2, 2), 0.985),
            lw_up=lw_up,
        )
        alb, T, emis = atm.get_sfc_override(0.0)
        T_bb = (lw_up / constants.sigma_sb) ** 0.25
        assert jnp.allclose(T, T_bb)
        assert not jnp.allclose(T, 248.0)   # not T_rad
        # And gray emits the full upward flux exactly: sigma*T_bb^4 == lw_up.
        assert jnp.allclose(constants.sigma_sb * T ** 4, lw_up, atol=1e-6)
        assert emis is None


class TestStaticSurfaceEmissivity:
    """``PhysicsPipeline.static_surface_emissivity`` — the EXACT ocean/ice(/land)
    emissivity blend radiation emits with absent a coupler override, used by the
    coupled drivers to invert held lw_net_sfc back to gross lw_down without an
    O(1 W/m^2) bias.  Exercised on a lightweight instance (only the emissivity_*
    and f_land attributes matter)."""

    def _pipe(self, f_land=None):
        from legoesm.driver.physics_pipeline import PhysicsPipeline
        p = PhysicsPipeline.__new__(PhysicsPipeline)
        p.emissivity_ice = 0.95
        p.emissivity_ocean = 0.97
        p.emissivity_land = 0.96
        p.f_land = f_land
        return p

    def test_ocean_ice_blend_uses_configured_values(self):
        from legoesm.forcing.surface_utils import blend_surface_property
        p = self._pipe()
        sic = jnp.array([0.0, 0.5, 1.0])
        eps = p.static_surface_emissivity(sic, land_active=False)
        # Configured (0.95/0.97), NOT a constant approximation.
        assert jnp.allclose(eps, blend_surface_property(sic, 0.95, 0.97))
        assert jnp.allclose(eps[0], 0.97)   # all ocean
        assert jnp.allclose(eps[2], 0.95)   # all ice

    def test_land_blend_applied_when_active(self):
        from legoesm.forcing.surface_utils import blend_surface_property
        f_land = jnp.array([0.0, 1.0, 0.5])
        p = self._pipe(f_land=f_land)
        sic = jnp.zeros(3)
        eps = p.static_surface_emissivity(sic, land_active=True)
        base = blend_surface_property(sic, 0.95, 0.97)
        assert jnp.allclose(eps, f_land * 0.96 + (1.0 - f_land) * base)
        assert jnp.allclose(eps[1], 0.96)   # all land -> land emissivity

    def test_land_blend_skipped_when_inactive(self):
        # f_land set but land_active False (no land skin temp) -> no land term,
        # matching compute_radiation_core's f_land-AND-T_land gate.
        p = self._pipe(f_land=jnp.array([1.0, 1.0]))
        sic = jnp.zeros(2)
        eps = p.static_surface_emissivity(sic, land_active=False)
        assert jnp.allclose(eps, 0.97)      # pure ocean, no land blend
