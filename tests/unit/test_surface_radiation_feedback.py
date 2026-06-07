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

    def _atm_config(self):
        return SimpleNamespace(albedo_ice=0.65, albedo_ocean=0.06, T_ice=271.35)

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
        sst = jnp.full((2, 2), 290.0)
        sic = jnp.zeros((2, 2))
        atm = SimpleNamespace(
            get_sfc_override=None, get_sst_sic=lambda d: (sst, sic),
        )
        drv = SimpleNamespace(
            coupled_cfg=CoupledConfig(couple_surface_radiation=True),
            _atm=atm, _last_sfc_response=None, atm_config=self._atm_config(),
        )
        CoupledESMDriver._override_sfc(drv)
        assert atm.get_sfc_override is not None
        alb, T = atm.get_sfc_override(0.0)
        # Seed = static blend (all-ocean here): albedo 0.06, T_sfc = sst.
        assert alb is not None and T is not None
        assert jnp.allclose(alb, 0.06)
        assert jnp.allclose(T, 290.0)

    def test_passes_through_sfc_response(self):
        sst = jnp.full((2, 2), 290.0)
        sic = jnp.zeros((2, 2))
        atm = SimpleNamespace(
            get_sfc_override=None, get_sst_sic=lambda d: (sst, sic),
        )
        drv = SimpleNamespace(
            coupled_cfg=CoupledConfig(couple_surface_radiation=True),
            _atm=atm, _last_sfc_response=None, atm_config=self._atm_config(),
        )
        CoupledESMDriver._override_sfc(drv)
        # Once a coupler response exists, its dynamic albedo / skin-T flow.
        drv._last_sfc_response = SimpleNamespace(
            albedo=jnp.full((2, 2), 0.5), T_sfc=jnp.full((2, 2), 250.0),
        )
        alb, T = atm.get_sfc_override(0.0)
        assert jnp.array_equal(alb, jnp.full((2, 2), 0.5))
        assert jnp.array_equal(T, jnp.full((2, 2), 250.0))
