"""Direct unit tests for the DINO experiment module.

Exercises the leaf symbols of ``legoesm.ocean.experiments.dino`` —
config defaults, analytical bathymetry, wind-stress interpolation,
T*/S* targets, equatorial 1D profiles, top-layer tendencies, the
Lévy z* helper wrapper, and the lat-lon initial-state construction.
The MPAS path is exercised in a separate test that is skipped when
``scipy`` is unavailable, since regional Voronoi mesh creation depends
on it.

All tests run without JIT and at small problem sizes so the suite
stays cheap.  Production-scale integration of DINO is covered by
``scripts/run/run_dino.py`` and the experiment registry; this file is
the slopbuster-mandated direct-coverage entry point.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.latlon import create_mercator_grid
from legoesm.ocean.experiments import dino
from legoesm.ocean.experiments import AVAILABLE_EXPERIMENTS
from legoesm.ocean.experiments.dino import (
    DINO_L2_RECIPES,
    DINO_RECIPES,
    DINOConfig,
    EXPERIMENT_CONFIG,
    create_dino_z_star,
    create_initial_conditions,
    dino_S_profile_1d,
    dino_S_star,
    dino_T_profile_1d,
    dino_T_star_annual_mean,
    dino_bathymetry,
    dino_config_for_recipe,
    dino_initial_T_S,
    dino_lat_lon_grid,
    dino_lat_lon_initial_state_arrays,
    dino_lat_lon_model_config,
    dino_top_layer_S_tendency,
    dino_top_layer_T_tendency,
    dino_top_layer_u_tendency,
    dino_wind_stress,
)
from legoesm.ocean.init_latlon_cgrid import partial_periodic_seam_wall_latlon
from legoesm.ocean.vertical import create_levy_stretched_z_star


# ---------------------------------------------------------------------
# Config + registry
# ---------------------------------------------------------------------

class TestDINOConfig:
    def test_default_construct(self):
        cfg = DINOConfig()
        assert cfg.n_levels == 36
        assert cfg.H_deep == 4000.0
        assert cfg.H_shallow == 2000.0
        assert cfg.dz_min == 10.0
        # Paper-specific reference constants differ from project canonical
        # values by <0.1% — flagged in the docstring.
        assert cfg.rho_0 == pytest.approx(1026.0)
        assert cfg.c_p == pytest.approx(3991.86)
        # Wind knots: 7 (lat, tau) pairs.
        assert len(cfg.wind_tau_lats_deg) == 7
        assert len(cfg.wind_tau_values) == 7

    def test_mpas_equatorial_visc_boost_propagates(self):
        """The MPAS-only equatorial viscosity boost
        (``mpas_equatorial_visc_boost``) must reach the ``MPASOceanConfig``.
        Without it the forced f→0 equatorial jet runs away on the implicit-CN
        MPAS path (|u| 1.8→7.6 m/s by day 20 → NaN by day 30; fastest edges
        all at |lat|<3°). It mirrors the lat-lon ``A_h_eq_boost`` mechanism.
        ``physics`` is left off so the stub mesh only needs ``areaCell``.
        """
        import dataclasses
        from types import SimpleNamespace
        mesh = SimpleNamespace(areaCell=jnp.full((64,), 1.0e10))  # ~100 km cells
        cfg = DINOConfig()
        assert cfg.mpas_equatorial_visc_boost == pytest.approx(8.0)
        mc, phys = dino.dino_mpas_model_config(mesh, cfg, physics=False)
        assert mc.equatorial_visc_boost == pytest.approx(
            cfg.mpas_equatorial_visc_boost)
        assert phys is None
        # Override (the run_dino --mpas-eq-visc-boost flag) propagates. Use a
        # value distinct from the 8.0 default so the assert is non-vacuous.
        cfg2 = dataclasses.replace(cfg, mpas_equatorial_visc_boost=12.0)
        mc2, _ = dino.dino_mpas_model_config(mesh, cfg2, physics=False)
        assert mc2.equatorial_visc_boost == pytest.approx(12.0)

    def test_mpas_physics_is_wired_into_model_config(self):
        """The MPAS model gates KPP/GM-Redi/convection on
        ``config.physics is not None``; dino_mpas_model_config MUST wire the
        physics into the returned MPASOceanConfig (regression: it used to return
        physics only as the 2nd value, which run_dino drops -> dycore-only MPAS).
        """
        from types import SimpleNamespace
        mesh = SimpleNamespace(areaCell=jnp.full((64,), 1.0e10))
        mc, phys = dino.dino_mpas_model_config(mesh, DINOConfig(), physics=True)
        assert phys is not None
        assert mc.physics is not None, "MPAS physics not wired into the model config"
        assert mc.physics is phys
        assert mc.physics.vertical_mixing.scheme in ("kpp", "tke", "constant")
        # physics=False stays dycore-only.
        mc0, phys0 = dino.dino_mpas_model_config(mesh, DINOConfig(), physics=False)
        assert phys0 is None and mc0.physics is None

    def test_vmix_scheme_default_and_dispatch(self):
        """The shared vertical-mixing helper selects the paper's TKE closure by
        default (with the paper background visc/diff + convective ceiling) and
        KPP on request, and raises on an unknown scheme (dispatch hardening)."""
        import dataclasses
        cfg = DINOConfig()
        # Default is the stable KPP closure (TKE = paper's scheme but unstable
        # in our 1deg DINO past ~day 40 — see DINOConfig.vmix_scheme).
        assert cfg.vmix_scheme == "kpp"
        vm_kpp = dino._dino_vertical_mixing_config(cfg)
        assert vm_kpp.scheme == "kpp"
        assert vm_kpp.kpp.K_bg == pytest.approx(cfg.K_v_bg)
        # The paper-faithful TKE config maps the paper background + ceiling, and
        # prandtl_mode="constant" is required or kappaH_min/kappaM_max are dead.
        cfg_tke = dataclasses.replace(cfg, vmix_scheme="tke")
        vm = dino._dino_vertical_mixing_config(cfg_tke)
        assert vm.scheme == "tke"
        assert vm.tke.prandtl_mode == "constant"
        # MOMENTUM floor = the EFFECTIVE A_v (the SW-corner stabilizer = 5e-4, 4×
        # the paper) so TKE is runnable; the TRACER floor stays at the paper
        # K_v_bg, so the thermocline mixing is unchanged (see A_v_bg_effective).
        assert vm.tke.kappaM_min == pytest.approx(cfg_tke.A_v_bg_effective)
        assert cfg_tke.A_v_bg_effective == pytest.approx(5.0e-4)
        assert vm.tke.kappaH_min == pytest.approx(cfg.K_v_bg)
        assert vm.tke.kappaM_max == pytest.approx(cfg.K_conv)
        assert vm.tke.bg_diff_scale == pytest.approx(0.0)  # constant bg (no Bryan-Lewis)
        # "constant" = background-only (used to unify vmix across grids).
        vm_c = dino._dino_vertical_mixing_config(
            dataclasses.replace(cfg, vmix_scheme="constant"))
        assert vm_c.scheme == "constant"
        # richardson (≈ Oceananigans RiBasedVerticalDiffusivity) and catke
        # (Oceananigans CATKEVerticalDiffusivity) are the L2 Oceananigans-card
        # closures; both wire with the paper background floors.
        vm_ri = dino._dino_vertical_mixing_config(
            dataclasses.replace(cfg, vmix_scheme="richardson"))
        assert vm_ri.scheme == "richardson"
        assert vm_ri.richardson.K_bg == pytest.approx(cfg.K_v_bg)
        assert vm_ri.richardson.A_bg == pytest.approx(cfg.A_v_bg_effective)
        vm_catke = dino._dino_vertical_mixing_config(
            dataclasses.replace(cfg, vmix_scheme="catke"))
        assert vm_catke.scheme == "catke"
        with pytest.raises(ValueError):
            dino._dino_vertical_mixing_config(
                dataclasses.replace(cfg, vmix_scheme="bogus"))

    def test_registry_entry(self):
        assert "dino" in AVAILABLE_EXPERIMENTS
        assert AVAILABLE_EXPERIMENTS["dino"] is EXPERIMENT_CONFIG
        assert EXPERIMENT_CONFIG["name"] == "dino"
        assert EXPERIMENT_CONFIG["config_class"] is DINOConfig
        # Grid support matches the implemented paths.
        gs = EXPERIMENT_CONFIG["grid_support"]
        assert gs["latlon"] is True
        assert gs["mpas"] is True
        assert gs["cubed_sphere"] is False
        assert gs["spectral"] is False


# ---------------------------------------------------------------------
# L2 model recipes (DINO two-level intercomparison)
# ---------------------------------------------------------------------

class TestDINORecipes:
    """The recipe overlay (``DINO_RECIPES`` + ``dino_config_for_recipe``) selects
    each model's canonical blocks as a PURE CONFIG on ``DINOConfig``, threads
    them into the lat-lon model config, and raises on an unknown recipe
    (dispatch hardening — a typo must never silently pick a default)."""

    def test_unknown_recipe_raises(self):
        with pytest.raises(ValueError):
            dino_config_for_recipe("bogus_model")

    def test_catalog_membership(self):
        assert set(DINO_RECIPES) == {
            "legoesm_default", "nemo_paper", "veros", "mitgcm", "oceananigans"}
        assert set(DINO_L2_RECIPES) == {"veros", "mitgcm", "oceananigans"}
        assert set(DINO_L2_RECIPES) <= set(DINO_RECIPES)

    def test_legoesm_default_is_identity(self):
        # The identity card = a bare DINOConfig (Wright + KPP), so a no-op run
        # reproduces the production default exactly.
        assert dino_config_for_recipe("legoesm_default") == DINOConfig()

    def test_recipe_overlay_selects_documented_blocks(self):
        # EOS + vertical mixing + tracer advection = the per-model DINO choices.
        nemo = dino_config_for_recipe("nemo_paper")
        assert (nemo.eos, nemo.vmix_scheme, nemo.tracer_advection) == (
            "nemo_seos", "tke", "tvd")
        # nemo_paper must use the geometric S-EOS depth (NEMO gdept) — the DINO
        # tendency certificate showed insitu carries a 10x depth-proportional
        # error in the thermobaric term.  Other recipes keep the insitu default.
        assert nemo.eos_depth == "geometric"
        assert dino_config_for_recipe("legoesm_default").eos_depth == "insitu"
        veros = dino_config_for_recipe("veros")
        assert (veros.eos, veros.vmix_scheme, veros.tracer_advection,
                veros.barotropic_solver) == (
            "veros_nonlin2", "tke", "superbee", "rigid_lid")
        mit = dino_config_for_recipe("mitgcm")
        assert (mit.eos, mit.vmix_scheme, mit.tracer_advection,
                mit.momentum_advection, mit.outer_integrator,
                mit.barotropic_solver) == (
            "unesco80", "kpp", "dst3_multidim", "flux_form", "ab2",
            "implicit_unsplit")
        ocn = dino_config_for_recipe("oceananigans")
        assert (ocn.eos, ocn.vmix_scheme, ocn.tracer_advection,
                ocn.momentum_advection, ocn.outer_integrator) == (
            "veros_gsw", "catke", "weno7", "weno7", "ab2")

    def test_base_override_preserved(self):
        # A recipe overlay keeps the non-scheme setup fields of the base config.
        import dataclasses
        base = dataclasses.replace(DINOConfig(), dt=1800.0)
        cfg = dino_config_for_recipe("mitgcm", base=base)
        assert cfg.dt == pytest.approx(1800.0)     # setup field preserved
        assert cfg.eos == "unesco80"               # recipe field applied

    def test_default_model_config_scheme_identity_unchanged(self):
        # Behavior preservation: a bare DINOConfig still yields the legoESM DINO
        # dycore identity (the newly threaded fields default to the prior values,
        # so existing runs are byte-identical).
        cfg = DINOConfig()
        grid = dino_lat_lon_grid(cfg, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, cfg, physics=False)
        assert mc.flat_get("momentum_advection") == "vector_invariant"
        assert mc.flat_get("coriolis_scheme") == "matsuno_split"
        assert mc.flat_get("outer_integrator") == "forward_euler"

    def test_mitgcm_threads_into_model_config(self):
        # The MITgcm card's flux-form / AB2 / unsplit-FS blocks reach the model.
        cfg = dino_config_for_recipe("mitgcm")
        grid = dino_lat_lon_grid(cfg, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, cfg, physics=False)
        assert mc.flat_get("momentum_advection") == "flux_form"
        assert mc.flat_get("momentum_flux_scheme") == "centered"
        assert mc.flat_get("coriolis_scheme") == "explicit_ab2"
        assert mc.flat_get("outer_integrator") == "ab2"
        assert mc.flat_get("ab2_scope") == "total"
        assert mc.flat_get("barotropic_solver") == "implicit_unsplit"
        assert mc.flat_get("eos") == "unesco80"

    def test_veros_rigid_lid_forces_ab2_stack(self):
        # The Veros card selects rigid_lid; the builder ALWAYS overlays the
        # coordinated Veros-faithful ab2 stack on that path (ab2 + explicit_ab2 +
        # advective scope), independent of the DINOConfig integrator defaults.
        cfg = dino_config_for_recipe("veros")
        grid = dino_lat_lon_grid(cfg, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, cfg, physics=False)
        assert mc.flat_get("barotropic_solver") == "rigid_lid"
        assert mc.flat_get("outer_integrator") == "ab2"
        assert mc.flat_get("coriolis_scheme") == "explicit_ab2"
        assert mc.flat_get("ab2_scope") == "advective"
        assert mc.flat_get("eos") == "veros_nonlin2"
        # rigid_lid force-disables the F_slow AB2 flag (its validation rejects
        # ab2_scope="advective"; the streamfunction projection has no barotropic
        # inertial mode to time-center).
        assert mc.flat_get("barotropic_slow_forcing_ab2") is False

    def test_oceananigans_card_ab2_centers_barotropic_slow_forcing(self):
        # Root-cause guard for the DINO 'oceananigans'-card barotropic blowup
        # (dino_l2_bisect o_ctl: |eta| 6 m by day 15, growth rate ∝ dt, basin-
        # scale off-equatorial quadrupole).  Under coriolis_scheme="explicit_ab2"
        # + barotropic_solver="implicit_cn" the CN predictor gates its FB
        # Coriolis off (_cori_fac=0) and the outer AB2 keeps the barotropic
        # increment un-extrapolated, so WITHOUT barotropic_slow_forcing_ab2 the
        # barotropic-mode Coriolis integrates forward-Euler — unconditionally
        # unstable, |G| = sqrt(1 + (f·dt)²) per step.  The card must carry the
        # Oceananigans-faithful Gᵁ AB2 time-centering (as the validated-stable
        # Silvestri §5 jet stack does).
        cfg = dino_config_for_recipe("oceananigans")
        assert cfg.barotropic_slow_forcing_ab2 is True
        grid = dino_lat_lon_grid(cfg, n_lon=10)
        mc, _ = dino_lat_lon_model_config(grid, cfg, physics=False)
        assert mc.flat_get("coriolis_scheme") == "explicit_ab2"
        assert mc.flat_get("barotropic_solver") == "implicit_cn"
        assert mc.flat_get("outer_integrator") == "ab2"
        assert mc.flat_get("ab2_scope") == "total"
        assert mc.flat_get("barotropic_slow_forcing_ab2") is True


# ---------------------------------------------------------------------
# Bathymetry (paper Appendix A)
# ---------------------------------------------------------------------

class TestDinoBathymetry:
    def test_interior_is_deep(self):
        # Point well inside the basin (no taper, no sill).
        H = float(dino_bathymetry(jnp.array(-25.0), jnp.array(0.0)))
        assert H == pytest.approx(DINOConfig().H_deep, rel=1e-3)

    def test_boundary_is_shallow(self):
        cfg = DINOConfig()
        # Exactly at the western wall.
        H = float(dino_bathymetry(
            jnp.array(cfg.lon_west_deg), jnp.array(0.0),
        ))
        assert H == pytest.approx(cfg.H_shallow, rel=1e-3)

    def test_drake_sill_shallows_basin(self):
        # The Drake sill is a Gaussian ring anchored at the western wall
        # (sill_lon_m_deg) and extending eastward over
        # ``sill_gaussian_width_s`` degrees. At lon=sill_lon_m_deg
        # exactly, ``sill_taper`` is 0 (left edge of smooth-step), so
        # we sample slightly east of the sill anchor where the taper is
        # active. In the channel band the open-flow bathymetry would be
        # H_deep, so any sill effect shoals it below H_deep.
        cfg = DINOConfig()
        lon_inside_sill = cfg.sill_lon_m_deg + 0.5 * cfg.sill_gaussian_width_s
        H_sill_region = float(dino_bathymetry(
            jnp.array(lon_inside_sill),
            jnp.array(cfg.sill_lat_m_deg),
        ))
        H_open_channel = float(dino_bathymetry(
            jnp.array(-25.0),
            jnp.array(cfg.sill_lat_m_deg),
        ))
        assert H_sill_region < H_open_channel
        assert H_sill_region >= cfg.H_sill - 100.0

    def test_shape_broadcast(self):
        lon = jnp.linspace(-50.0, 0.0, 10)
        lat = jnp.linspace(-70.0, 70.0, 12)[:, None]
        H = dino_bathymetry(lon[None, :], lat)
        assert H.shape == (12, 10)
        assert jnp.all(jnp.isfinite(H))


# ---------------------------------------------------------------------
# Wind stress
# ---------------------------------------------------------------------

class TestDinoWindStress:
    def test_hits_knots(self):
        cfg = DINOConfig()
        for lat_knot, tau_knot in zip(cfg.wind_tau_lats_deg, cfg.wind_tau_values):
            tau = float(dino_wind_stress(jnp.array(lat_knot)))
            assert tau == pytest.approx(tau_knot, abs=1e-6)

    def test_outside_range_clips(self):
        cfg = DINOConfig()
        # Beyond the southernmost knot — extrapolation clips to the
        # endpoint per the cubic-smooth-step formulation.
        tau_far_south = float(dino_wind_stress(jnp.array(-89.0)))
        assert tau_far_south == pytest.approx(cfg.wind_tau_values[0], abs=1e-6)
        tau_far_north = float(dino_wind_stress(jnp.array(89.0)))
        assert tau_far_north == pytest.approx(cfg.wind_tau_values[-1], abs=1e-6)


# ---------------------------------------------------------------------
# Restoring targets (paper eqs B1-B2)
# ---------------------------------------------------------------------

class TestRestoringTargets:
    def test_T_star_warmest_at_equator(self):
        cfg = DINOConfig()
        T_eq = float(dino_T_star_annual_mean(jnp.array(0.0)))
        T_north = float(dino_T_star_annual_mean(jnp.array(cfg.lat_max_deg)))
        T_south = float(dino_T_star_annual_mean(jnp.array(-cfg.lat_max_deg)))
        assert T_eq > T_north
        assert T_eq > T_south
        # Equatorial target matches the configured value when lat=0.
        assert T_eq == pytest.approx(cfg.T_star_eq, abs=0.5)

    def test_S_star_dip_at_equator(self):
        cfg = DINOConfig()
        # Equatorial Gaussian dip means S* at the equator is BELOW the
        # cosine-profile peak by approximately ``S_star_eq_dip_amp``.
        S_eq = float(dino_S_star(jnp.array(0.0)))
        S_north = float(dino_S_star(jnp.array(cfg.lat_max_deg)))
        assert S_eq < cfg.S_star_eq  # dip subtracted
        # Northern boundary target ~ S_star_n.
        assert S_north == pytest.approx(cfg.S_star_n, abs=0.5)


# ---------------------------------------------------------------------
# Equatorial 1D T(z), S(z) profiles (paper eq D2-D3)
# ---------------------------------------------------------------------

class TestProfiles1D:
    def test_T_decreasing_with_depth(self):
        z = jnp.linspace(0.0, 4000.0, 41)
        T = dino_T_profile_1d(z)
        # Sea-surface warmer than abyss, abyss roughly 3-5 C.
        T_np = np.asarray(T)
        assert T_np[0] > T_np[-1]
        assert 2.0 < T_np[-1] < 5.5
        assert 20.0 < T_np[0] < 30.0

    def test_S_finite_and_bounded(self):
        z = jnp.linspace(0.0, 4000.0, 41)
        S = np.asarray(dino_S_profile_1d(z))
        assert np.all(np.isfinite(S))
        assert (S > 30.0).all() and (S < 38.0).all()


# ---------------------------------------------------------------------
# Top-layer tendencies (paper eqs 7-9)
# ---------------------------------------------------------------------

class TestTopLayerTendencies:
    def test_T_relaxes_toward_target(self):
        cfg = DINOConfig()
        dz0 = 10.0
        # Cold ocean below warm target: positive tendency.
        dT_dt_warm = float(dino_top_layer_T_tendency(
            T_sfc_C=0.0, T_star=20.0, Q_sr=0.0, dz_0=dz0, cfg=cfg,
        ))
        # Warm ocean below cold target: negative tendency.
        dT_dt_cold = float(dino_top_layer_T_tendency(
            T_sfc_C=20.0, T_star=0.0, Q_sr=0.0, dz_0=dz0, cfg=cfg,
        ))
        assert dT_dt_warm > 0.0
        assert dT_dt_cold < 0.0

    def test_S_relaxes_toward_target(self):
        cfg = DINOConfig()
        dz0 = 10.0
        dS_dt = float(dino_top_layer_S_tendency(
            S_surface=34.0, S_star=36.0, dz_0=dz0, cfg=cfg,
        ))
        assert dS_dt > 0.0  # freshening deficit → S increases

    def test_u_proportional_to_tau(self):
        cfg = DINOConfig()
        dz0 = 10.0
        du_a = float(dino_top_layer_u_tendency(0.1, dz0, cfg))
        du_b = float(dino_top_layer_u_tendency(0.2, dz0, cfg))
        assert du_b == pytest.approx(2.0 * du_a, rel=1e-6)
        # Magnitude check: τ/(ρ·dz) ≈ 0.1 / (1026 · 10) ≈ 9.7e-6 m/s².
        expected = 0.1 / (cfg.rho_0 * dz0)
        assert du_a == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------
# Lévy z* helper (vertical.py) + DINO wrapper
# ---------------------------------------------------------------------

class TestLevyZStar:
    def test_endpoints_and_monotone(self):
        # 36-level DINO grid: dz_min constraint is dz/dk at k=1, which
        # matches dz[0] (forward difference) closely when the stretching
        # parameters keep d²z/dk² small over the top layer. Tight check
        # is therefore valid only at the design n_levels=36.
        n_levels = 36
        zc = create_levy_stretched_z_star(
            n_levels=n_levels, H_max=4000.0, dz_min=10.0,
            k_th=float(n_levels - 1), a_cr=10.5,
        )
        assert zc.n_levels == n_levels
        assert float(zc.z_half_ref[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(zc.z_half_ref[-1]) == pytest.approx(-4000.0, abs=1e-9)
        dz = np.asarray(zc.dz_ref)
        assert (dz > 0).all()
        assert dz[0] == pytest.approx(10.0, rel=0.05)
        assert dz[-1] > dz[0]
        assert dz.sum() == pytest.approx(4000.0, rel=1e-6)

    def test_low_resolution_endpoints(self):
        # At coarse resolutions the forward-difference dz[0] can differ
        # from the analytical dz/dk constraint, but endpoints and total
        # depth must still match exactly.
        zc = create_levy_stretched_z_star(
            n_levels=12, H_max=4000.0, dz_min=10.0,
            k_th=11.0, a_cr=10.5,
        )
        assert zc.n_levels == 12
        assert float(zc.z_half_ref[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(zc.z_half_ref[-1]) == pytest.approx(-4000.0, abs=1e-9)
        dz = np.asarray(zc.dz_ref)
        assert (dz > 0).all()
        assert dz.sum() == pytest.approx(4000.0, rel=1e-6)

    def test_create_dino_z_star_defaults(self):
        cfg = DINOConfig()
        zc = create_dino_z_star(cfg)
        assert zc.n_levels == cfg.n_levels == 36
        assert float(zc.z_half_ref[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(zc.z_half_ref[-1]) == pytest.approx(-cfg.H_deep, abs=1e-9)
        # Top layer ~ dz_min.
        assert float(zc.dz_ref[0]) == pytest.approx(cfg.dz_min, rel=0.05)

    def test_input_validation(self):
        with pytest.raises(ValueError):
            create_levy_stretched_z_star(1, 4000.0, 10.0, 35.0, 10.5)
        with pytest.raises(ValueError):
            create_levy_stretched_z_star(36, 0.0, 10.0, 35.0, 10.5)
        with pytest.raises(ValueError):
            create_levy_stretched_z_star(36, 4000.0, 0.0, 35.0, 10.5)


# ---------------------------------------------------------------------
# Seam-wall helper (lat-lon C-grid)
# ---------------------------------------------------------------------

class TestPartialPeriodicSeamLatLon:
    def test_open_band_stays_ocean(self):
        cfg = DINOConfig()
        grid = create_mercator_grid(
            n_lon=10,
            lat_max_deg=cfg.lat_max_deg,
            lon_west_deg=cfg.lon_west_deg,
            lon_east_deg=cfg.lon_east_deg,
        )
        mask = np.asarray(partial_periodic_seam_wall_latlon(
            grid,
            open_lat_south_deg=cfg.channel_lat_south_deg,
            open_lat_north_deg=cfg.channel_lat_north_deg,
            seam_column_index=0,
        ))
        # Outside the open band, the seam column is land.
        lat_1d_deg = np.degrees(np.asarray(grid.lat))
        for j, lat_deg in enumerate(lat_1d_deg):
            in_band = (
                cfg.channel_lat_south_deg
                <= lat_deg
                <= cfg.channel_lat_north_deg
            )
            if in_band:
                assert mask[j, 0] == 1.0
            else:
                assert mask[j, 0] == 0.0


# ---------------------------------------------------------------------
# Initial-state construction (Mercator path)
# ---------------------------------------------------------------------

class TestLatLonInitialState:
    def test_T_S_field_shapes(self):
        # Use a coarser grid for test speed (4-level z, 6 lon, ~12 lat).
        cfg = DINOConfig()
        grid = dino_lat_lon_grid(cfg, n_lon=6)
        zc = create_levy_stretched_z_star(
            n_levels=4, H_max=cfg.H_deep, dz_min=400.0,
            k_th=3.0, a_cr=2.0,
        )
        T, S, H_bathy, land_mask = dino_lat_lon_initial_state_arrays(
            grid, zc, cfg,
        )
        n_lat, n_lon = grid.n_lat, grid.n_lon
        assert T.shape == (n_lat, n_lon, 4)
        assert S.shape == (n_lat, n_lon, 4)
        assert H_bathy.shape == (n_lat, n_lon)
        assert land_mask.shape == (n_lat, n_lon)
        # Ocean depth on wet cells in [H_shallow, H_deep] (sill carve-out
        # may shoal below H_shallow on a few sill-ring cells — tolerate).
        ocean = np.asarray(land_mask) > 0.5
        Hb = np.asarray(H_bathy)[ocean]
        assert Hb.max() <= cfg.H_deep + 1e-6
        assert Hb.min() > 0.0

    def test_initial_T_S_columns(self):
        cfg = DINOConfig()
        zc = create_levy_stretched_z_star(
            n_levels=8, H_max=cfg.H_deep, dz_min=50.0,
            k_th=7.0, a_cr=2.0,
        )
        # Equator column: should match the 1D profile (uses cfg.lat_max_deg
        # in the meridional weight, so |φ|=0 ⇒ full equatorial profile).
        T_2d, S_2d = dino_initial_T_S(jnp.array(0.0), zc.z_full_ref, cfg)
        # Surface warmer than abyss.
        T_col = np.asarray(T_2d)
        assert T_col.shape == (8,)
        assert T_col[0] > T_col[-1]


# ---------------------------------------------------------------------
# create_initial_conditions dispatch
# ---------------------------------------------------------------------

class TestCreateInitialConditionsDispatch:
    def test_rejects_unknown_grid_type(self):
        cfg = DINOConfig()
        grid = dino_lat_lon_grid(cfg, n_lon=4)
        zc = create_levy_stretched_z_star(
            n_levels=4, H_max=cfg.H_deep, dz_min=400.0,
            k_th=3.0, a_cr=2.0,
        )
        with pytest.raises(ValueError, match="Unknown grid_type"):
            create_initial_conditions("cubed_sphere", grid, zc, cfg)


# ---------------------------------------------------------------------
# MPAS partial-periodic seam wall (optional — needs scipy)
# ---------------------------------------------------------------------

scipy = pytest.importorskip("scipy", reason="MPAS Voronoi mesh needs scipy")


class TestMPASSeamWall:
    def test_open_band_keeps_seam_cells_ocean(self):
        from legoesm.grids.voronoi import create_regional_voronoi_mesh
        from legoesm.ocean.init_mpas import partial_periodic_seam_wall_mpas
        cfg = DINOConfig()
        try:
            mesh = create_regional_voronoi_mesh(
                lon_range=(cfg.lon_west_deg, cfg.lon_east_deg),
                lat_range=(-cfg.lat_max_deg, cfg.lat_max_deg),
                resolution_km=200.0,
                periodic_x=True,
            )
        except (ValueError, RuntimeError) as exc:
            pytest.skip(f"Voronoi mesh generation unstable on this "
                        f"platform: {exc}")
        mask = np.asarray(partial_periodic_seam_wall_mpas(
            mesh,
            open_lat_south_deg=cfg.channel_lat_south_deg,
            open_lat_north_deg=cfg.channel_lat_north_deg,
            seam_lon_deg=cfg.lon_west_deg,
        ))
        lat_deg = np.degrees(np.asarray(mesh.latCell))
        lon_deg = (np.degrees(np.asarray(mesh.lonCell)) + 180.0) % 360.0 - 180.0
        # Cells near the seam outside the open band: expect at least
        # one to be land (mask = 0). Inside the band: expect at least
        # one near-seam cell to stay ocean.
        near_seam_dist_deg = (
            (lon_deg - cfg.lon_west_deg) % 360.0
        )
        seam_strip = near_seam_dist_deg < 5.0
        outside_band = (lat_deg < cfg.channel_lat_south_deg) | (
            lat_deg > cfg.channel_lat_north_deg
        )
        inside_band = (lat_deg >= cfg.channel_lat_south_deg) & (
            lat_deg <= cfg.channel_lat_north_deg
        )
        if (seam_strip & outside_band).any():
            assert (mask[seam_strip & outside_band] < 0.5).any()
        if (seam_strip & inside_band).any():
            assert (mask[seam_strip & inside_band] > 0.5).any()


# ---------------------------------------------------------------------
# Rigid-lid island topology (the ACC is the channel island constant)
# ---------------------------------------------------------------------

class TestDINORigidLidIslandTopology:
    """The re-entrant southern channel makes the DINO basin multiply-connected.

    The seam-wall land (column 0) is split by the open channel band into a
    NORTHERN segment and a SOUTHERN segment — two disconnected land masses.
    That is the Veros ACC topology: with two islands the rigid-lid streamfunction
    system has one free island circulation constant, and that constant IS the
    depth-integrated ACC transport.  This is why ``rigid_lid`` is the natural
    barotropic solver here (the ACC is a NULL mode of the free-surface eta solve,
    which is what made implicit_cn / explicit_substep under-/over-shoot + ring in
    the barotropic-solver audit).
    """

    def test_seam_wall_splits_into_two_land_masses(self):
        from legoesm.ocean.dynamics.rigid_lid_islands import _label_islands

        cfg = DINOConfig()
        grid = dino_lat_lon_grid(cfg, n_lon=20)           # small + cheap
        zc = create_levy_stretched_z_star(
            n_levels=4, H_max=cfg.H_deep, dz_min=400.0, k_th=3.0, a_cr=2.0)
        _, _, _, land_mask = dino_lat_lon_initial_state_arrays(grid, zc, cfg)
        cell_land = np.asarray(land_mask) < 0.5

        # All land is the seam column; the channel band rows are open there.
        assert cell_land[:, 1:].sum() == 0, "DINO land must be the seam column only"
        lat = np.degrees(np.asarray(grid.lat))
        chan = (lat > cfg.channel_lat_south_deg) & (lat < cfg.channel_lat_north_deg)
        assert not cell_land[chan, 0].any(), "channel band seam must be open ocean"
        assert cell_land[~chan, 0].any(), "seam outside the channel must be land"

        labels, nisle = _label_islands(cell_land, periodic_x=True)
        # Two land masses (northern + southern seam wall) ⇒ one free ACC constant.
        assert nisle == 2, (
            f"DINO re-entrant channel must yield exactly 2 land masses "
            f"(N + S seam walls) for the ACC island constant; got nisle={nisle}")
        # The two islands are the seam segments north / south of the channel.
        north = labels[(lat > cfg.channel_lat_north_deg), 0]
        south = labels[(lat < cfg.channel_lat_south_deg), 0]
        assert set(np.unique(north[north > 0])).isdisjoint(
            set(np.unique(south[south > 0]))), "N and S walls must be distinct islands"


class TestDINORigidLidFaithfulStack:
    """Selecting rigid_lid auto-applies the Veros-faithful coordinated stack.

    A BARE barotropic_solver="rigid_lid" flip runs away (568 Sv → NaN): the
    bottom-drag depth-mean reaches the streamfunction barotropic balance only via
    the du_diss fold, which is gated on ab2_scope="advective".  So
    dino_lat_lon_model_config pairs rigid_lid with ab2 + explicit_ab2 +
    ab2_scope="advective" + dt_mom_ratio (veros_acc_recipe.py); other solvers keep
    the forward_euler defaults.
    """

    def _model_cfg(self, solver):
        cfg = DINOConfig(barotropic_solver=solver)
        grid = dino_lat_lon_grid(cfg, n_lon=6)
        mcfg, _ = dino.dino_lat_lon_model_config(grid, cfg, physics=True)
        return mcfg

    def test_rigid_lid_applies_faithful_stack(self):
        m = self._model_cfg("rigid_lid")
        assert m.barotropic.barotropic_solver == "rigid_lid"
        assert m.outer_integrator == "ab2"
        assert m.coriolis_scheme == "explicit_ab2"
        assert m.ab2_scope == "advective"
        assert m.dt_mom_ratio == DINOConfig().rigid_lid_dt_mom_ratio == 9.0

    def test_implicit_cn_keeps_forward_euler_defaults(self):
        m = self._model_cfg("implicit_cn")
        assert m.outer_integrator == "forward_euler"
        assert m.coriolis_scheme == "matsuno_split"
        assert m.ab2_scope == "total"
        assert m.dt_mom_ratio == 1.0

    def test_rigid_lid_dt_mom_ratio_is_tunable(self):
        cfg = DINOConfig(barotropic_solver="rigid_lid", rigid_lid_dt_mom_ratio=5.0)
        grid = dino_lat_lon_grid(cfg, n_lon=6)
        mcfg, _ = dino.dino_lat_lon_model_config(grid, cfg, physics=True)
        assert mcfg.dt_mom_ratio == 5.0
# Level-1 exactness: seasonal forcing (usrdef_sbc ln_ann_cyc) + salt flux
# ---------------------------------------------------------------------

class TestSeasonalForcing:
    def test_seasonal_cosine_phases(self):
        from legoesm.ocean.experiments.dino import dino_seasonal_cosines
        day = 86400.0
        # c1 peaks at 21 June (day 171 of the 360-day year), c2 at 21 July.
        c1, c2 = dino_seasonal_cosines(171.0 * 24.0 * 3600.0)
        assert float(c1) == pytest.approx(1.0, abs=1e-12)
        c1_w, _ = dino_seasonal_cosines((171.0 + 180.0) * day)
        assert float(c1_w) == pytest.approx(-1.0, abs=1e-12)
        _, c2_p = dino_seasonal_cosines(201.0 * day)
        assert float(c2_p) == pytest.approx(1.0, abs=1e-12)
        # periodic over the 360-day year
        c1_a, _ = dino_seasonal_cosines(10.0 * day)
        c1_b, _ = dino_seasonal_cosines((10.0 + 360.0) * day)
        assert float(c1_a) == pytest.approx(float(c1_b), abs=1e-12)

    def test_T_star_seasonal_asymmetry_and_mean_consistency(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_T_star_annual_mean, dino_T_star_seasonal,
            dino_seasonal_cosines,
        )
        cfg = DINOConfig()
        day = 86400.0
        lat = jnp.array([-70.0, 0.0, 70.0])
        # boundary swings: north amp 3.0, south amp 0.5 (oracle asymmetry)
        T_jul = dino_T_star_seasonal(lat, 201.0 * day, cfg)   # c2 = +1
        T_jan = dino_T_star_seasonal(lat, 21.0 * day, cfg)    # c2 = -1
        assert float(T_jul[2] - T_jan[2]) == pytest.approx(2 * 3.0, abs=1e-9)
        assert float(T_jan[0] - T_jul[0]) == pytest.approx(2 * 0.5, abs=1e-9)
        # equator: profile=1 -> T* = T_eq, season-independent
        assert float(T_jul[1]) == pytest.approx(float(T_jan[1]), abs=1e-12)
        # zero-phase (c2=0) equals the annual-mean form
        # c2 = 0 at day 201 - 90 = 111
        t0 = 111.0 * day
        _, c2 = dino_seasonal_cosines(t0)
        assert abs(float(c2)) < 1e-9
        np.testing.assert_allclose(
            np.asarray(dino_T_star_seasonal(lat, t0, cfg)),
            np.asarray(dino_T_star_annual_mean(lat, cfg)), rtol=1e-9)

    def test_Q_sr_seasonal_mean_matches_annual_quadrature(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_Q_sr_annual_mean, dino_Q_sr_seasonal,
        )
        cfg = DINOConfig()
        lat = jnp.linspace(-70.0, 70.0, 15)
        # daily samples at t = d days reproduce the annual-mean fn's
        # quadrature EXACTLY (same discrete phases)
        days = np.arange(1, 361, dtype=np.float64)
        acc = np.zeros(15)
        for d in days:
            acc += np.asarray(dino_Q_sr_seasonal(lat, d * 86400.0, cfg))
        np.testing.assert_allclose(
            acc / 360.0, np.asarray(dino_Q_sr_annual_mean(lat, cfg)),
            rtol=1e-10)
        # polar night: high south lat in southern winter (c1=+1) -> 0
        q = dino_Q_sr_seasonal(jnp.array([-70.0]), 171.0 * 86400.0, cfg)
        assert float(q[0]) == 0.0

    def test_apply_seasonal_requires_time_and_changes_forcing(self):
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_grid,
            dino_lat_lon_state, dino_lat_lon_surface_forcing_arrays,
            apply_dino_lat_lon_surface_forcing,
        )
        cfg = dataclasses.replace(DINOConfig(), forcing_annual_cycle=True)
        z = create_dino_z_star(cfg)
        g = dino_lat_lon_grid(cfg, n_lon=8)
        st = dino_lat_lon_state(g, z, cfg)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
        with pytest.raises(ValueError, match="t_seconds"):
            apply_dino_lat_lon_surface_forcing(st, frc, z, cfg, 2700.0)
        day = 86400.0
        s_jun = apply_dino_lat_lon_surface_forcing(
            st, frc, z, cfg, 2700.0, t_seconds=171.0 * day)
        s_dec = apply_dino_lat_lon_surface_forcing(
            st, frc, z, cfg, 2700.0, t_seconds=351.0 * day)
        assert float(jnp.max(jnp.abs(s_jun.T.data - s_dec.T.data))) > 0.0
        # flag OFF: t_seconds ignored -> bit-identical to the legacy call
        cfg0 = dataclasses.replace(cfg, forcing_annual_cycle=False)
        a = apply_dino_lat_lon_surface_forcing(
            st, frc, z, cfg0, 2700.0, t_seconds=171.0 * day)
        b = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg0, 2700.0)
        np.testing.assert_array_equal(np.asarray(a.T.data),
                                      np.asarray(b.T.data))

    def test_mpas_apply_rejects_seasonal(self):
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            DINOConfig, apply_dino_mpas_surface_forcing,
        )
        cfg = dataclasses.replace(DINOConfig(), forcing_annual_cycle=True)
        with pytest.raises(NotImplementedError, match="lat-lon"):
            apply_dino_mpas_surface_forcing(None, None, None, cfg, 2700.0)

    def test_salt_and_heat_flux_coefficients_match_nemo(self):
        """A_S == |rn_srp| and A_theta == |rn_trp|; the top-layer salinity
        tendency equals NEMO's sfx/(rho0*e3t) with sfx = srp*(SSS-S*)."""
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_top_layer_S_tendency,
        )
        cfg = DINOConfig()
        assert cfg.A_S == pytest.approx(3.858e-3)     # -rn_srp
        assert cfg.A_theta == pytest.approx(40.0)     # -rn_trp
        S, S_star, dz0 = 35.4, 35.0, 10.0
        ours = float(dino_top_layer_S_tendency(
            jnp.asarray(S), jnp.asarray(S_star), dz0, cfg))
        srp = -3.858e-3
        nemo = srp * (S - S_star) / (cfg.rho_0 * dz0)
        assert ours == pytest.approx(nemo, rel=1e-12)
        assert ours < 0.0    # SSS above target -> freshening flux (sign walk)


class TestWindThroughStep:
    def test_taum_boost_field(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_surface_forcing_arrays,
        )
        cfg = DINOConfig()
        g = dino_lat_lon_grid(cfg, n_lon=8)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
        tau = np.asarray(frc["tau_u_cell_2d"][:, 0])
        taum = np.asarray(frc["taum_2d"][:, 0])
        west = tau > 0.0
        np.testing.assert_allclose(taum[west], 1.3 * np.abs(tau[west]),
                                   rtol=1e-12)
        np.testing.assert_allclose(taum[~west], np.abs(tau[~west]),
                                   rtol=1e-12)
        assert west.any() and (~west).any()

    def test_tke_taum_override_changes_K(self):
        from legoesm.ocean.physics.vertical_mixing.tke import (
            tke_vertical_mixing,
        )
        from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
        nlev = 10
        shape = (4, 3, nlev)
        rng = np.random.default_rng(0)
        T = jnp.asarray(20.0 - 15.0 * np.linspace(0, 1, nlev)[None, None, :]
                        * np.ones(shape))
        S = jnp.full(shape, 35.0)
        u = jnp.zeros(shape); v = jnp.zeros(shape)
        rho = jnp.asarray(1026.0 - 2.0 * np.linspace(0, 1, nlev))[None, None, :] * jnp.ones(shape)
        dz_half = jnp.full(shape[:-1] + (nlev - 1,), 50.0)
        tau = jnp.full(shape[:-1], 0.2)
        out_plain = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, tke_old=None,
            tau_x_surface=tau, tau_y_surface=jnp.zeros_like(tau),
            dt=86400.0, cfg=TKEConfig(), rho_0=1026.0, g=9.80665,
            n_iterations=2)
        out_boost = tke_vertical_mixing(
            u, v, T, S, rho, dz_half, tke_old=None,
            tau_x_surface=tau, tau_y_surface=jnp.zeros_like(tau),
            taum_surface=1.3 * tau,
            dt=86400.0, cfg=TKEConfig(), rho_0=1026.0, g=9.80665,
            n_iterations=2)
        dK = float(jnp.max(jnp.abs(out_boost.K_H - out_plain.K_H)))
        assert dK > 0.0    # boosted taum must strengthen the TKE input

    def test_apply_skips_wind_when_through_step(self):
        import dataclasses
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_grid,
            dino_lat_lon_state, dino_lat_lon_surface_forcing_arrays,
            apply_dino_lat_lon_surface_forcing,
        )
        cfg_off = DINOConfig()
        cfg_on = dataclasses.replace(cfg_off, wind_through_step=True)
        z = create_dino_z_star(cfg_off)
        g = dino_lat_lon_grid(cfg_off, n_lon=8)
        st = dino_lat_lon_state(g, z, cfg_off)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg_off)
        s_off = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg_off, 2700.0)
        s_on = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg_on, 2700.0)
        # flag ON: the applicator leaves u untouched (the step owns wind)
        np.testing.assert_array_equal(np.asarray(s_on.u.data),
                                      np.asarray(st.u.data))
        assert float(jnp.max(jnp.abs(s_off.u.data - st.u.data))) > 0.0

    def test_step_forcing_carries_wind_and_taum(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_surface_forcing_arrays,
            dino_step_surface_forcing,
        )
        cfg = DINOConfig()
        g = dino_lat_lon_grid(cfg, n_lon=8)
        frc = dino_lat_lon_surface_forcing_arrays(g, cfg)
        sf = dino_step_surface_forcing(frc)
        # atmospheric convention: the core applies -tau (ocean reaction)
        np.testing.assert_array_equal(np.asarray(sf.tau_x),
                                      -np.asarray(frc["tau_u_cell_2d"]))
        np.testing.assert_array_equal(np.asarray(sf.taum),
                                      np.asarray(frc["taum_2d"]))
        assert sf.q_net is None and sf.sw_down is None

    def test_model_step_applies_wind_from_rest(self):
        import dataclasses
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_grid,
            dino_lat_lon_state, dino_lat_lon_model_config,
            dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing,
        )
        cfg = dataclasses.replace(DINOConfig(), wind_through_step=True)
        z = create_dino_z_star(cfg)
        g = dino_lat_lon_grid(cfg, n_lon=10)
        st = dino_lat_lon_state(g, z, cfg)
        model = LatLonCGridOceanModel(
            g, z, dino_lat_lon_model_config(g, cfg, physics=True)[0])
        sf = dino_step_surface_forcing(
            dino_lat_lon_surface_forcing_arrays(g, cfg))
        s1 = model.step(st, dt=cfg.dt, surface_forcing=sf)
        u_top = np.asarray(s1.u.data[..., 0])
        assert np.isfinite(u_top).all()
        # westerlies (tau>0 around -45 lat) accelerate +u at the top layer,
        # with the UNBOOSTED momentum stress: pin the magnitude against the
        # analytic Euler kick tau*dt/(rho0*dz0) (RK3 staging + the implicit
        # friction shave it somewhat; a x1.3 taum leak into momentum or a
        # doubled application would leave the [0.5, 1.1] band).
        lat = np.degrees(np.asarray(g.lat))
        j = int(np.argmin(np.abs(lat - (-45.0))))
        got = u_top[j, 1:-1].mean()
        tau_row = float(np.asarray(
            dino_lat_lon_surface_forcing_arrays(g, cfg)["tau_u_cell_2d"])[j, 0])
        expect = tau_row * cfg.dt / (cfg.rho_0 * float(z.dz_ref[0]))
        assert got > 0.0
        assert 0.5 * expect < got < 1.1 * expect, (got, expect)


def test_S_star_boundary_targets_match_oracle():
    """codex catch (dino-s12 #1): rn_sstar_s=35.0, rn_sstar_n=35.1 — the
    DINOConfig defaults had them SWAPPED.  Pin the boundary values (the
    cos-profile term vanishes at |phi|=70 where cos(2*pi*70/140)=-1 makes
    the profile factor 0; the Gaussian dip is negligible there)."""
    from legoesm.ocean.experiments.dino import DINOConfig, dino_S_star
    cfg = DINOConfig()
    assert cfg.S_star_s == 35.0 and cfg.S_star_n == 35.1
    s_s = float(dino_S_star(jnp.asarray(-70.0), cfg))
    s_n = float(dino_S_star(jnp.asarray(70.0), cfg))
    assert s_s == pytest.approx(35.0, abs=1e-6)
    assert s_n == pytest.approx(35.1, abs=1e-6)


def test_mpas_builder_rejects_fct_family_tracer_advection():
    """The r1_exact preset (fct2) must fail LOUDLY at the MPAS config
    builder with actionable guidance, not deep in model construction."""
    import pytest as _pytest

    from legoesm.ocean.experiments.dino import (
        dino_mpas_model_config, dino_r1_exact_config,
    )

    cfg = dino_r1_exact_config()
    with _pytest.raises(ValueError, match="lat-lon only"):
        dino_mpas_model_config(None, cfg, physics=False)


class TestMaskedZco:
    """NEMO ln_zco full-cell masking (zgr_msk_top_bot) for DINO."""

    def _z_and_bowl(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_bowl,
            dino_lat_lon_grid,
        )
        cfg = DINOConfig()
        z = create_dino_z_star(cfg)
        g = dino_lat_lon_grid(cfg, n_lon=12)
        return cfg, z, np.asarray(dino_lat_lon_bowl(g, cfg))

    @staticmethod
    def _mi96_f90(jpk, H, dzmin, kth, acr):
        """INDEPENDENT numpy transliteration of zgr_lib.F90 mi96_1d
        (kkconst=0, ph_co=0 — the 1-D reference ladder DINO's ln_zco
        uses): returns (pdepw_1d[jpk], pdept_1d[jpk]) positive down."""
        import math
        jpkm1 = jpk - 1
        za1 = ((dzmin - H / jpkm1)
               / (math.tanh((1 - kth) / acr)
                  - acr / jpkm1 * (math.log(math.cosh((jpk - kth) / acr))
                                   - math.log(math.cosh((1 - kth) / acr)))))
        za0 = dzmin - za1 * math.tanh((1 - kth) / acr)
        zsur = -za0 - za1 * acr * math.log(math.cosh((1 - kth) / acr))
        w = np.array([zsur + za0 * k
                      + za1 * acr * math.log(math.cosh((k - kth) / acr))
                      for k in range(1, jpk + 1)])
        t = np.array([zsur + za0 * (k + 0.5)
                      + za1 * acr * math.log(math.cosh((k + 0.5 - kth) / acr))
                      for k in range(1, jpk + 1)])
        return w, t

    def test_ladder_matches_f90_and_reference_run(self):
        """The masked-zco ladder must equal the mi96_1d transliteration
        AND the reference run's deptht (first/last wet values hardcoded
        from DINO_1m_grid_T.nc, float32 storage)."""
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_vertical,
        )
        import dataclasses
        cfg = dataclasses.replace(DINOConfig(),
                                  vertical_coordinate="masked_zco")
        g = dino_lat_lon_grid(cfg, n_lon=12)
        coord = dino_lat_lon_vertical(g, cfg)
        assert coord.n_levels == cfg.n_levels - 1     # NEMO jpk dummy level

        w_f90, t_f90 = self._mi96_f90(
            cfg.n_levels, cfg.H_deep, cfg.dz_min, float(cfg.k_th),
            cfg.a_cr)
        np.testing.assert_allclose(
            np.abs(np.asarray(coord.z_half_ref))[1:], w_f90[1:], rtol=1e-9)
        np.testing.assert_allclose(
            np.abs(np.asarray(coord.z_full_ref)), t_f90[:-1], rtol=1e-9)
        # reference-run oracle (deptht, f32): first two + last wet centre
        np.testing.assert_allclose(
            np.abs(np.asarray(coord.z_full_ref))[[0, 1, -1]],
            [5.0335817, 15.322634, 3757.309], rtol=1e-6)

    def test_snap_rule_matches_f90_transliteration(self):
        """k_bot per usrdef_zgr.F90 zgr_msk_top_bot:
        WHERE( pdept(jk) < H .AND. H <= pdept(jk+1) ) k_bot = jk,
        with pdept from the INDEPENDENT mi96 transliteration."""
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_bowl, dino_lat_lon_grid,
            dino_lat_lon_vertical,
        )
        cfg = dataclasses.replace(DINOConfig(),
                                  vertical_coordinate="masked_zco")
        g = dino_lat_lon_grid(cfg, n_lon=12)
        H = np.asarray(dino_lat_lon_bowl(g, cfg))
        coord = dino_lat_lon_vertical(g, cfg)

        _, pdept = self._mi96_f90(
            cfg.n_levels, cfg.H_deep, cfg.dz_min, float(cfg.k_th),
            cfg.a_cr)
        jpkm1 = cfg.n_levels - 1
        k_bot = np.zeros(H.shape, dtype=int)   # 0 = land (k_top=0)
        for jk in range(jpkm1):                # NEMO: 1..jpkm1 wet
            sel = (pdept[jk] < H) & (H <= pdept[jk + 1])
            k_bot[sel] = jk + 1                # NEMO 1-based level count
        np.testing.assert_array_equal(
            np.asarray(coord.bottom_level) + 1, k_bot)

    def test_full_cells_and_snap_depth(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_bowl, dino_lat_lon_grid,
            dino_lat_lon_vertical, dino_masked_zco_coordinate,
        )
        cfg = dataclasses.replace(DINOConfig(),
                                  vertical_coordinate="masked_zco")
        g = dino_lat_lon_grid(cfg, n_lon=12)
        H = np.asarray(dino_lat_lon_bowl(g, cfg))
        coord = dino_lat_lon_vertical(g, cfg)
        H_snap = np.asarray(coord.h_partial).sum(-1)
        h = np.asarray(coord.h_partial)
        dz = np.asarray(coord.dz_ref)
        # every wet cell is a FULL cell; below-bottom cells are zero
        is_full = np.isclose(h, dz[None, None, :], rtol=0, atol=1e-9)
        is_zero = h == 0.0
        assert bool(np.all(is_full | is_zero))
        # and the Jacobian is 1 at eta=0 on wet columns
        from legoesm.ocean.vertical import compute_ocean_jacobian
        J = np.asarray(compute_ocean_jacobian(
            jnp.zeros(H.shape), jnp.asarray(H_snap), coord))
        wet = np.asarray(coord.bottom_level) >= 0
        np.testing.assert_allclose(J[wet], 1.0, rtol=0, atol=1e-12)

    def test_dispatch(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_vertical,
        )
        from legoesm.ocean.vertical import (
            OceanPartialCellCoordinate, OceanZStarCoordinate,
        )
        cfg = DINOConfig()
        g = dino_lat_lon_grid(cfg, n_lon=12)
        assert isinstance(dino_lat_lon_vertical(g, cfg),
                          OceanZStarCoordinate)
        cfg2 = dataclasses.replace(cfg, vertical_coordinate="masked_zco")
        assert isinstance(dino_lat_lon_vertical(g, cfg2),
                          OceanPartialCellCoordinate)
        cfg3 = dataclasses.replace(cfg, vertical_coordinate="sigma")
        with pytest.raises(ValueError, match="vertical_coordinate"):
            dino_lat_lon_vertical(g, cfg3)

    def test_state_H_bathy_matches_coordinate(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_state,
            dino_lat_lon_vertical,
        )
        cfg = dataclasses.replace(DINOConfig(),
                                  vertical_coordinate="masked_zco")
        g = dino_lat_lon_grid(cfg, n_lon=12)
        z = dino_lat_lon_vertical(g, cfg)
        st = dino_lat_lon_state(g, z, cfg)
        # state fields are stored float32 -> f32-appropriate tolerance
        np.testing.assert_allclose(
            np.asarray(st.H_bathy.data),
            np.asarray(z.h_partial).sum(-1), rtol=1e-6)

    def test_mpas_builder_rejects_masked_zco(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_mpas_model_config,
        )
        cfg = dataclasses.replace(DINOConfig(),
                                  vertical_coordinate="masked_zco")
        with pytest.raises(ValueError, match="lat-lon only"):
            dino_mpas_model_config(None, cfg, physics=False)


class TestIsoneutralRediOnly:
    """lateral_tracer_mixing='isoneutral' — NEMO ln_traldf_iso(+msc)."""

    def _preset_model_cfg(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            dino_lat_lon_grid, dino_lat_lon_model_config,
            dino_r1_exact_config,
        )
        cfg = dino_r1_exact_config()
        g = dino_lat_lon_grid(cfg, n_lon=12)
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        return cfg, g, mc

    def test_preset_builds_redi_only(self):
        cfg, g, mc = self._preset_model_cfg()
        gm = mc.gm_redi
        assert gm is not None
        assert gm.kappa_GM == 0.0
        assert gm.kappa_Redi > 0.0
        assert gm.kappa_redi_lat_scaling is True
        assert gm.S_max == 0.01
        assert gm.slope_density == "neutral"
        assert gm.implicit_K33 is True
        assert not gm.visbeck.enabled and not gm.treguier.enabled
        assert mc.K_h == 0.0                    # no iso-level double-count
        # kappa_Redi equals the legacy K_h coefficient (½·U_T·R·dλ)
        import numpy as _np

        from legoesm import constants as _c
        expect = 0.5 * cfg.U_T * _c.R_earth * float(_np.asarray(g.dlon))
        assert gm.kappa_Redi == pytest.approx(expect, rel=1e-12)

    def test_legacy_default_unchanged(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_model_config,
        )
        cfg = DINOConfig()          # geopotential + use_gm_redi=True
        g = dino_lat_lon_grid(cfg, n_lon=12)
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        assert mc.K_h > 0.0
        assert mc.gm_redi.visbeck.enabled       # historical adaptive path
        assert mc.gm_redi.kappa_redi_lat_scaling is False

    def test_unknown_mixing_raises(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_model_config,
        )
        cfg = dataclasses.replace(DINOConfig(),
                                  lateral_tracer_mixing="epineutral")
        g = dino_lat_lon_grid(cfg, n_lon=12)
        with pytest.raises(ValueError, match="lateral_tracer_mixing"):
            dino_lat_lon_model_config(g, cfg, physics=True)

    def test_iso_plus_eiv_rejected(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_model_config,
        )
        cfg = dataclasses.replace(
            DINOConfig(), lateral_tracer_mixing="isoneutral",
            use_gm_redi=True)
        g = dino_lat_lon_grid(cfg, n_lon=12)
        with pytest.raises(ValueError, match="EIV"):
            dino_lat_lon_model_config(g, cfg, physics=True)

    def test_static_kappa_override_row_scaling(self):
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            _static_kappa_redi_override,
        )
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid,
        )
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        g = dino_lat_lon_grid(DINOConfig(), n_lon=12)
        gm_on = GMRediConfig(kappa_Redi=100.0, kappa_redi_lat_scaling=True)
        arr = _static_kappa_redi_override(gm_on, g)
        assert arr.shape == (g.n_lat, 12)
        lat = np.asarray(g.lat)
        # grid.lat is stored float32 -> f32-appropriate tolerance
        np.testing.assert_allclose(
            np.asarray(arr)[:, 0], 100.0 * np.cos(lat), rtol=1e-6)
        gm_off = GMRediConfig(kappa_Redi=100.0)
        assert _static_kappa_redi_override(gm_off, g) is None

    def test_mpas_builder_rejects_isoneutral(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_mpas_model_config,
        )
        cfg = dataclasses.replace(DINOConfig(),
                                  lateral_tracer_mixing="isoneutral",
                                  use_gm_redi=False)
        with _pytest_raises_valueerror("MPAS DINO path"):
            dino_mpas_model_config(None, cfg, physics=False)

    def test_lat_scaling_rejects_2d_latitudes(self):
        from types import SimpleNamespace

        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            _static_kappa_redi_override,
        )
        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        gm = GMRediConfig(kappa_Redi=100.0, kappa_redi_lat_scaling=True)
        fake = SimpleNamespace(lat=np.zeros((4, 5)), n_lon=5)
        with pytest.raises(ValueError, match="1-D latitudes"):
            _static_kappa_redi_override(gm, fake)


def _pytest_raises_valueerror(match):
    return pytest.raises(ValueError, match=match)


class TestSlopeLimitNemoCap:
    """slope_limit='nemo_cap': slope capped at S_max, taper == 1."""

    def _w_triads(self, slope_limit):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            _w_triad_slopes_tapers,
        )
        rng = np.random.default_rng(3)
        n_lat, n_lon, nlev = 4, 5, 6
        drho_dx_u = jnp.asarray(rng.normal(size=(n_lat, n_lon + 1, nlev)))
        drho_dy_v = jnp.asarray(rng.normal(size=(n_lat + 1, n_lon, nlev)))
        # weak stratification -> raw slopes far beyond S_max
        drho_dz_w = jnp.full((n_lat, n_lon, nlev - 1), -1e-3)
        return _w_triad_slopes_tapers(
            drho_dx_u, drho_dy_v, drho_dz_w, n_lat, n_lon,
            S_max=0.01, taper_width_frac=0.1, slope_limit=slope_limit)

    def test_cap_bounds_slopes_and_unit_tapers(self):
        out = self._w_triads("nemo_cap")
        slopes, tapers = out[:8], out[8:]
        for sl in slopes:
            assert float(jnp.abs(sl).max()) <= 0.01 + 1e-15
        for tp in tapers:
            np.testing.assert_array_equal(np.asarray(tp), 1.0)

    def test_dm95_tapers_at_steep_slopes(self):
        """dm95 mode: the taper suppresses the flux at steep slopes
        (~0.5 at the in-situ clip point) — the behaviour nemo_cap
        replaces with unit tapers."""
        out = self._w_triads("dm95_taper")
        tapers = out[8:]
        assert float(jnp.stack(tapers).min()) <= 0.55
        assert float(jnp.stack(tapers).max()) < 1.0

    def test_unknown_slope_limit_raises(self):
        with pytest.raises(ValueError, match="slope_limit"):
            self._w_triads("gerdes")

    def test_centered_scheme_rejects_cap(self):
        from legoesm.ocean.experiments.dino import (
            dino_lat_lon_grid, dino_lat_lon_state, dino_lat_lon_vertical,
            dino_r1_exact_config,
        )
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            gm_redi_tracer_tendency_latlon,
        )
        cfg = dino_r1_exact_config()
        g = dino_lat_lon_grid(cfg, n_lon=12)
        z = dino_lat_lon_vertical(g, cfg)
        st = dino_lat_lon_state(g, z, cfg)
        from legoesm.ocean.experiments.dino import dino_lat_lon_model_config
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        gm_bad = mc.gm_redi._replace(slope_scheme="centered")
        with pytest.raises(ValueError, match="nemo_cap"):
            gm_redi_tracer_tendency_latlon(
                st.T.data, st.S.data, st.eta.data, st.H_bathy.data,
                g, z, gm_bad, eos=mc.eos)

    def test_preset_selects_cap(self):
        from legoesm.ocean.experiments.dino import (
            dino_lat_lon_grid, dino_lat_lon_model_config,
            dino_r1_exact_config,
        )
        cfg = dino_r1_exact_config()
        assert cfg.redi_slope_limit == "nemo_cap"
        g = dino_lat_lon_grid(cfg, n_lon=12)
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        assert mc.gm_redi.slope_limit == "nemo_cap"

    def test_mpas_gm_redi_rejects_nemo_cap(self):
        """MPAS centred GM/Redi must raise on slope_limit='nemo_cap'
        rather than silently keep the DM95 taper (codex r7 P1)."""
        from types import SimpleNamespace

        from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
        from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
            gm_redi_tracer_tendency_mpas,
        )
        cfg = GMRediConfig(slope_limit="nemo_cap")
        with pytest.raises(NotImplementedError, match="slope_limit"):
            # entry-guard fires before any mesh access
            gm_redi_tracer_tendency_mpas(
                None, None, None, None, None, None, cfg)


class TestNemoCentredBarotropic:
    """dynspg_ts ln_bt_fw=F + nn_bt_flt=1 forward-frame reduction."""

    def test_boxcar_window_centred_at_new_time(self):
        from legoesm.ocean.dynamics.barotropic_common import (
            compute_nemo_boxcar_centred_weights,
        )
        n = 12
        w, w_total, w_tr, n_loop = compute_nemo_boxcar_centred_weights(
            n, jnp.float64)
        w = np.asarray(w)
        assert n_loop == w.size
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-14)
        # F90 transliteration: zwgt1(jn)=1 where |jn - n|/n < 0.5
        jn = np.arange(1, n_loop + 1, dtype=float)
        expect = (np.abs(jn - n) / n < 0.5).astype(float)
        expect = expect / expect.sum()
        np.testing.assert_allclose(w, expect, rtol=1e-14)
        # centroid at the baroclinic step (tau = 1)
        centroid = (w * jn / n).sum()
        np.testing.assert_allclose(centroid, 1.0, rtol=0, atol=0.05)
        # window extends past the step but not to 2n
        assert n < n_loop < 2 * n

    def test_transport_weights_continuity_telescoping(self):
        """w_transport[j] = sum(w[j:]) / n — the unique choice with
        div(Hu_avg) == (eta_old - eta_avg)/dt (uniform tracer)."""
        from legoesm.ocean.dynamics.barotropic_common import (
            compute_nemo_boxcar_centred_weights,
        )
        n = 9
        w, _, w_tr, n_loop = compute_nemo_boxcar_centred_weights(
            n, jnp.float64)
        w, w_tr = np.asarray(w), np.asarray(w_tr)
        expect = np.array([w[i:].sum() for i in range(n_loop)]) / n
        np.testing.assert_allclose(w_tr, expect, rtol=1e-14)

    def test_auto_substeps_formula(self):
        from legoesm.ocean.dynamics.barotropic_common import (
            nemo_auto_substeps,
        )
        import math

        from legoesm import constants as _c
        dt, H, e1, e2 = 2700.0, 4000.0, 1.1e5, 1.1e5
        inv = 1.0 / e1 ** 2 + 1.0 / e2 ** 2
        n = nemo_auto_substeps(dt, H, inv, float(_c.g), cmax=0.8)
        zcu = math.sqrt(float(_c.g) * H * inv)
        assert n == math.ceil(dt / 0.8 * zcu)
        with pytest.raises(ValueError, match="n="):
            nemo_auto_substeps(1e-6, H, inv, float(_c.g), cmax=0.8)

    def test_mpas_builder_rejects_centred_barotropic(self):
        import dataclasses

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_mpas_model_config,
        )
        cfg = dataclasses.replace(
            DINOConfig(), barotropic_time_filter="nemo_boxcar_centred")
        with pytest.raises(ValueError, match="barotropic"):
            dino_mpas_model_config(None, cfg, physics=False)
        cfg2 = dataclasses.replace(DINOConfig(), barotropic_auto_cmax=0.8)
        with pytest.raises(ValueError, match="barotropic"):
            dino_mpas_model_config(None, cfg2, physics=False)

    def test_preset_selects_centred_explicit(self):
        from legoesm.ocean.experiments.dino import (
            dino_lat_lon_grid, dino_lat_lon_model_config,
            dino_r1_exact_config,
        )
        import dataclasses

        # The PRESET keeps implicit_cn: the centred window needs the MLF
        # before-state start (job 8826132 NaN; NEMO's own namelist:
        # "model crashes if ln_bt_fw=T"). The blocks remain selectable:
        cfg = dino_r1_exact_config()
        assert cfg.barotropic_solver == "implicit_cn"
        cfg = dataclasses.replace(
            cfg, barotropic_solver="explicit_substep",
            barotropic_time_filter="nemo_boxcar_centred",
            barotropic_auto_cmax=0.8)
        g = dino_lat_lon_grid(cfg, n_lon=12)
        mc, _ = dino_lat_lon_model_config(g, cfg, physics=True)
        assert mc.barotropic.barotropic_solver == "explicit_substep"
        assert mc.barotropic.barotropic_time_filter == "nemo_boxcar_centred"
        # auto count: static int equal to the ln_bt_auto helper's value
        from legoesm.ocean.experiments.dino import _dino_barotropic_substeps
        assert isinstance(mc.barotropic.n_barotropic_substeps, int)
        assert (mc.barotropic.n_barotropic_substeps
                == _dino_barotropic_substeps(g, cfg) >= 2)
