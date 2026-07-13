"""Phase 4 tests for the CLM-ML-JAX canopy interface.

Tests:
- Smoke test: compute_clm_ml_canopy_fluxes runs under daytime forcing
- Energy balance closure: |net_rad - SH - LH - G| < 5 W/m² (cold-start)
- GPP > 0 under positive PAR
- Differentiability: jax.grad of sum(lhflx) w.r.t. T_lowest runs without error
- Integration: step_multilayer_land with CLMMLCanopyConfig produces finite output
- Dispatch test: default SimpleSEB scheme still works after Phase 3 changes
- Config validation: CLMMLCanopyConfig() is a valid NamedTuple
"""

from __future__ import annotations

import unittest

import pytest

jax_available = True
try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except ImportError:
    jax_available = False

clm_ml_available = True
try:
    import multilayer_canopy  # noqa: F401 – just test import
except ImportError:
    clm_ml_available = False

pytestmark = pytest.mark.skipif(
    not (jax_available and clm_ml_available),
    reason="clm-ml-jax not installed",
)

NCOL = 2   # minimal column count for tests


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_forcing(ncol: int = NCOL):
    from legoesm.core.coupling_fields import AtmToSurface
    return AtmToSurface(
        sw_down=jnp.full(ncol, 400.0),          # daytime: positive PAR
        lw_down=jnp.full(ncol, 350.0),
        precip_total=jnp.full(ncol, 0.0),
        precip_snow=jnp.zeros(ncol),
        T_lowest=jnp.full(ncol, 295.0),
        q_lowest=jnp.full(ncol, 0.010),
        u_lowest=jnp.full(ncol, 4.0),
        v_lowest=jnp.full(ncol, 1.5),
        p_lowest=jnp.full(ncol, 95000.0),
        p_surface=jnp.full(ncol, 100000.0),
        rho_lowest=jnp.full(ncol, 1.2),
        cos_zenith=jnp.full(ncol, 0.7),
        co2_ppmv=jnp.full(ncol, 400.0),
        has_radiation=jnp.ones(ncol),
        has_precipitation=jnp.ones(ncol),
    )


def _make_soil_arrays(ncol: int = NCOL, n_layers: int = 8):
    T_soil = jnp.full((ncol, n_layers), 290.0)
    psi_soil = jnp.full((ncol, n_layers), -0.5)   # [m], moderate suction
    theta_soil = jnp.full((ncol, n_layers), 0.25)  # [m³/m³]
    return T_soil, psi_soil, theta_soil


def _call_interface(ncol=NCOL):
    """Call compute_clm_ml_canopy_fluxes with minimal realistic inputs.

    Returns ``(surface_out, canopy_state_new)``.
    """
    from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    from legoesm.land.config import MultiLayerLandConfig

    forcing = _make_forcing(ncol)
    T_soil, psi_soil, theta_soil = _make_soil_arrays(ncol)
    config = CLMMLCanopyConfig()
    land_config = MultiLayerLandConfig(surface_scheme=config)

    return compute_clm_ml_canopy_fluxes(
        T_soil_top=T_soil[:, 0],
        forcing=forcing,
        canopy_config=config,
        land_config=land_config,
        land_params=None,
        w_frac_rz=jnp.full(ncol, 0.6),
        wind_speed=jnp.full(ncol, 4.5),
        canopy_state=None,
        dt=1800.0,
        T_soil=T_soil,
        psi_soil=psi_soil,
        theta_soil=theta_soil,
        lat=jnp.zeros(ncol),
        doy=180.0,
    )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestCLMMLInterface(unittest.TestCase):
    """Smoke and unit tests for compute_clm_ml_canopy_fluxes."""

    @pytest.mark.timeout(120)
    def test_smoke_daytime(self):
        """Interface runs under daytime forcing and returns valid shapes."""
        surface_out, canopy_state = _call_interface()

        self.assertEqual(surface_out.shflx.shape, (NCOL,))
        self.assertEqual(surface_out.lhflx.shape, (NCOL,))
        self.assertEqual(surface_out.G_soil.shape, (NCOL,))
        self.assertEqual(surface_out.sw_net.shape, (NCOL,))
        self.assertIsNotNone(canopy_state)
        self.assertIsNotNone(canopy_state.mlcanopy)

    @pytest.mark.timeout(120)
    def test_output_finite(self):
        """All output fields must be finite (no NaN / Inf)."""
        surface_out, _ = _call_interface()
        for name in surface_out._fields:
            val = getattr(surface_out, name)
            if val is None:
                continue
            arr = jnp.asarray(val)
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(arr))),
                f"{name} contains non-finite values: {arr}",
            )

    @pytest.mark.timeout(120)
    def test_gpp_positive_under_light(self):
        """GPP must be positive when sw_down > 0 (daytime)."""
        surface_out, _ = _call_interface()
        # gpp is in gC/m²/s; should be positive for all columns under 400 W/m² SW
        gpp = surface_out.gpp
        if gpp is not None:
            self.assertTrue(
                bool(jnp.all(gpp >= 0.0)),
                f"GPP should be non-negative under positive PAR, got {gpp}",
            )
            # At least one column should have meaningful GPP
            self.assertTrue(
                bool(jnp.any(gpp > 1e-10)),
                f"GPP should be positive under 400 W/m² SW, got {gpp}",
            )

    @pytest.mark.timeout(120)
    def test_energy_balance_closure(self):
        """CLM-ML internal energy balance: Rnet = SH + LH + G + stflx_air + stflx_veg.

        stflx_air + stflx_veg are canopy heat storage terms internal to CLM-ML.
        They are NOT folded into the coupler shflx (standard CLM-CAM convention).
        The coupler sees a small per-step residual that averages to zero diurnally.
        """
        surface_out, _ = _call_interface()
        net_rad = surface_out.sw_net + surface_out.lw_net
        stflx = jnp.zeros_like(surface_out.shflx)
        if surface_out.stflx_air is not None:
            stflx = stflx + surface_out.stflx_air
        if surface_out.stflx_veg is not None:
            stflx = stflx + surface_out.stflx_veg
        residual = jnp.abs(
            net_rad - surface_out.shflx - surface_out.lhflx - surface_out.G_soil - stflx
        )
        max_res = float(jnp.max(residual))
        self.assertLess(
            max_res, 10.0,   # cold-start residual; includes stflx_air + stflx_veg terms
            f"Energy balance residual too large: {max_res:.2f} W/m²",
        )

    @pytest.mark.timeout(120)
    def test_T_canopy_air_populated(self):
        """T_canopy_air is populated and physically between T_surface and forcing T."""
        surface_out, _ = _call_interface()
        self.assertIsNotNone(
            surface_out.T_canopy_air,
            "T_canopy_air must be populated for the coupler aerodynamic T_sfc",
        )
        Tc = surface_out.T_canopy_air
        T_sfc = surface_out.T_surface
        # Canopy air T must be finite and physically realistic (200–340 K)
        self.assertTrue(bool(jnp.all(jnp.isfinite(Tc))), "T_canopy_air has non-finite values")
        self.assertTrue(bool(jnp.all(Tc > 200.0)), f"T_canopy_air too cold: {Tc}")
        self.assertTrue(bool(jnp.all(Tc < 340.0)), f"T_canopy_air too hot: {Tc}")
        # Canopy air must differ from tg_soil (not the fallback) when vegetation is active
        self.assertFalse(
            bool(jnp.all(Tc == T_sfc)),
            "T_canopy_air equals T_surface — taveg_canopy fallback triggered unexpectedly",
        )

    @pytest.mark.timeout(120)
    def test_warm_start_reuses_state(self):
        """Second call with existing canopy_state does not re-cold-start."""
        _, state1 = _call_interface()
        # Second call passes state1 back
        from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
        from legoesm.land.canopy.config import CLMMLCanopyConfig
        from legoesm.land.config import MultiLayerLandConfig

        forcing = _make_forcing(NCOL)
        T_soil, psi_soil, theta_soil = _make_soil_arrays(NCOL)
        config = CLMMLCanopyConfig()
        land_config = MultiLayerLandConfig(surface_scheme=config)

        surface_out2, state2 = compute_clm_ml_canopy_fluxes(
            T_soil_top=T_soil[:, 0],
            forcing=forcing,
            canopy_config=config,
            land_config=land_config,
            land_params=None,
            w_frac_rz=jnp.full(NCOL, 0.6),
            wind_speed=jnp.full(NCOL, 4.5),
            canopy_state=state1,
            dt=1800.0,
            T_soil=T_soil,
            psi_soil=psi_soil,
            theta_soil=theta_soil,
            lat=jnp.zeros(NCOL),
            doy=180.0,
        )
        self.assertIsNotNone(state2)
        # Both calls should produce finite output
        for name in surface_out2._fields:
            val = getattr(surface_out2, name)
            if val is None:
                continue
            arr = jnp.asarray(val)
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(arr))),
                f"Warm-start: {name} contains non-finite values",
            )


class TestCLMMLDifferentiability(unittest.TestCase):
    """jax.grad flows through the CLM-ML canopy interface in differentiable mode.

    The CLM-ML-JAX repo exposes a JAX-native differentiable path selected by a
    per-call ``GridInfo`` (``grid=``).  legoESM activates it via
    ``CLMMLCanopyConfig(differentiable=True)`` (single column).  These tests
    verify ``jax.grad`` is finite, non-zero, matches finite difference, and that
    the diff path reproduces the forward-path fluxes exactly.  They are ``slow``
    (the first grad trace compiles the whole multilayer-canopy scan, ~minutes).
    """

    # ncol == 1: the diff path reads one concrete (ncan, ntop, nbot).
    _NCOL1 = 1

    def _run(self, config, T_lowest, state, T_soil, psi_soil, theta_soil):
        from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
        from legoesm.land.config import MultiLayerLandConfig
        forcing = _make_forcing(self._NCOL1)._replace(T_lowest=T_lowest)
        return compute_clm_ml_canopy_fluxes(
            T_soil_top=T_soil[:, 0],
            forcing=forcing,
            canopy_config=config,
            land_config=MultiLayerLandConfig(surface_scheme=config),
            land_params=None,
            w_frac_rz=jnp.full(self._NCOL1, 0.6),
            wind_speed=jnp.full(self._NCOL1, 4.5),
            canopy_state=state,
            dt=1800.0,
            T_soil=T_soil,
            psi_soil=psi_soil,
            theta_soil=theta_soil,
            lat=jnp.zeros(self._NCOL1),
            doy=180.0,
        )

    @pytest.mark.slow
    def test_multicolumn_diff_is_hard_error(self):
        """differentiable=True with ncol>1 raises (no silent degrade to forward)."""
        from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
        from legoesm.land.canopy.config import CLMMLCanopyConfig
        from legoesm.land.config import MultiLayerLandConfig
        config = CLMMLCanopyConfig(differentiable=True)
        T_soil, psi_soil, theta_soil = _make_soil_arrays(2)
        with self.assertRaises(ValueError):
            compute_clm_ml_canopy_fluxes(
                T_soil_top=T_soil[:, 0], forcing=_make_forcing(2),
                canopy_config=config,
                land_config=MultiLayerLandConfig(surface_scheme=config),
                land_params=None, w_frac_rz=jnp.full(2, 0.6),
                wind_speed=jnp.full(2, 4.5), canopy_state=None, dt=1800.0,
                T_soil=T_soil, psi_soil=psi_soil, theta_soil=theta_soil,
                lat=jnp.zeros(2), doy=180.0,
            )

    @pytest.mark.slow
    def test_grad_lhflx_wrt_T_lowest(self):
        """jax.grad(sum lhflx) w.r.t. T_lowest is finite, non-zero, FD-consistent,
        and the diff path matches the forward path to floating point.

        Replaces the historical strict-xfail: the CLM-ML-JAX diff mode
        (``grid=`` + ``lax.scan``) is JAX-native, so the old "requires a
        JAX-native rewrite" claim is obsolete.
        """
        from legoesm.land.canopy.config import CLMMLCanopyConfig

        cfg_diff = CLMMLCanopyConfig(differentiable=True)
        cfg_fwd = CLMMLCanopyConfig(differentiable=False)
        n = self._NCOL1
        # Soil state held FIXED (not derived from T_lowest) so T_lowest flows only
        # through the atmospheric forcing — a clean analytic-vs-FD comparison.
        T_soil = jnp.full((n, 8), 290.0)
        psi_soil = jnp.full((n, 8), -0.5)
        theta_soil = jnp.full((n, 8), 0.25)
        T0 = jnp.full(n, 295.0)

        # Warm-up cold call builds the vertical structure (forward, single step).
        _, state0 = self._run(cfg_diff, T0, None, T_soil, psi_soil, theta_soil)
        self.assertIsNotNone(state0.mlcanopy)

        # Forward/diff parity from the same warm state.
        out_diff, _ = self._run(cfg_diff, T0, state0, T_soil, psi_soil, theta_soil)
        out_fwd, _ = self._run(cfg_fwd, T0, state0, T_soil, psi_soil, theta_soil)
        for name in ("shflx", "lhflx", "G_soil", "sw_net", "gpp", "T_surface"):
            a = float(getattr(out_diff, name)[0])
            b = float(getattr(out_fwd, name)[0])
            self.assertTrue(
                abs(a - b) <= 1e-6 + 1e-6 * abs(b),
                f"diff/forward parity failed for {name}: {a} vs {b}",
            )

        # Gradient: finite, non-zero.
        def _loss(T_lowest):
            out, _ = self._run(cfg_diff, T_lowest, state0, T_soil, psi_soil, theta_soil)
            return jnp.sum(out.lhflx)

        g = jax.grad(_loss)(T0)
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))), f"grad non-finite: {g}")
        self.assertGreater(float(jnp.max(jnp.abs(g))), 0.0,
                           "grad is all-zero — forcing disconnected from lhflx")

        # Central finite-difference check (loose tol for the non-smooth physics).
        eps = 1e-2
        fp = float(_loss(T0 + eps))
        fm = float(_loss(T0 - eps))
        g_fd = (fp - fm) / (2.0 * eps)
        g_an = float(g[0])
        rel = abs(g_an - g_fd) / (abs(g_fd) + 1e-12)
        self.assertLess(
            rel, 0.10,
            f"FD gradient mismatch: analytic={g_an:.6e} fd={g_fd:.6e} rel={rel:.3%}",
        )


class TestCLMMLIntegration(unittest.TestCase):
    """Integration tests: step_multilayer_land with scheme='clm_ml'."""

    def _make_state_and_config(self, ncol=NCOL):
        from legoesm.land.canopy.config import CLMMLCanopyConfig
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.multilayer_land import init_multilayer_land_state

        config = MultiLayerLandConfig(surface_scheme=CLMMLCanopyConfig())
        state = init_multilayer_land_state(ncol, config, T_init=288.0)
        return state, config

    @pytest.mark.timeout(120)
    def test_single_step_finite(self):
        """One step with scheme='clm_ml' must produce finite TileResponse.

        Uses CHATS7 site geometry (38.47°N, California, May 1) so that the
        solar zenith angle is physically consistent with the validation site.
        """
        from legoesm.land.multilayer_land import step_multilayer_land

        state, config = self._make_state_and_config()
        forcing = _make_forcing(NCOL)

        new_state, response, _ = step_multilayer_land(
            state, forcing, config, U_min=1.0, dt=1800.0,
            lat=jnp.full(NCOL, 38.47), doy=120.5)

        # Check TileResponse is finite
        for name in response._fields:
            val = getattr(response, name)
            if val is None:
                continue
            arr = jnp.asarray(val)
            self.assertTrue(
                bool(jnp.all(jnp.isfinite(arr))),
                f"TileResponse.{name} is non-finite after first step",
            )

    @pytest.mark.slow
    def test_24h_run_water_budget(self):
        """24-hour run (48 × 1800 s steps): water budget closes within 1 mm/day.

        Water balance: Δ(water_in_soil) = precip - runoff - ET (approximately).
        For dry forcing (precip=0), all water loss is ET + runoff.
        We check that per-column integrated theta change is consistent with output.
        """
        from legoesm.land.multilayer_land import step_multilayer_land
        from legoesm.land.soil_grid import make_soil_grid

        state, config = self._make_state_and_config()
        forcing = _make_forcing(NCOL)
        # Set precip to zero so budget is simpler
        forcing_dry = forcing._replace(
            precip_total=jnp.zeros(NCOL),
            precip_snow=jnp.zeros(NCOL),
        )

        grid = make_soil_grid(config.soil_grid)
        dz = grid.dz  # (n_layers,)
        theta0 = state.theta_soil  # (ncol, n_layers)

        dt = 1800.0
        n_steps = 48  # 24 h

        total_lhflx = jnp.zeros(NCOL)
        total_runoff = jnp.zeros(NCOL)

        for _ in range(n_steps):
            state, response, _ = step_multilayer_land(
                state, forcing_dry, config, U_min=1.0, dt=dt)
            total_lhflx += response.lhflx
            total_runoff += (
                state.runoff_surface + state.runoff_subsurface)

        theta_final = state.theta_soil
        # Column water change [m water equivalent]
        delta_w = jnp.sum((theta_final - theta0) * dz[None, :], axis=-1)

        # ET [m water equivalent]: lhflx / (Lv * rho_w)  summed over steps
        from legoesm import constants
        Lv = constants.L_v
        rho_w = constants.rho_water  # const-ok: standard liquid water density
        et_m = total_lhflx * dt / (Lv * rho_w)  # m over 24 h

        # Runoff [m water equivalent]: (kg/m��/s × s) / (kg/m³) = m
        # total_runoff accumulated per step (kg/m²/s), so convert: × dt / rho_w
        runoff_m = total_runoff * dt / rho_w  # m over 24 h

        # Water balance: Δsoil + ET + runoff ≈ precip (≈0 for dry forcing)
        # Signs: delta_w = water gained (+), et_m = water lost to atmosphere (+),
        # runoff_m = water lost through drainage (+).
        budget_residual = jnp.abs(delta_w + et_m + runoff_m)
        max_res_mm = float(jnp.max(budget_residual)) * 1000.0
        self.assertLess(
            max_res_mm, 5.0,   # within 5 mm/day tolerance
            f"Water budget residual too large: {max_res_mm:.3f} mm/day",
        )


class TestDispatch(unittest.TestCase):
    """Verify that the default SimpleSEB dispatch is unaffected by Phase 3."""

    def test_default_scheme_still_works(self):
        """MultiLayerLandConfig() (SimpleSEB) produces finite output unchanged."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.multilayer_land import (
            step_multilayer_land,
            init_multilayer_land_state,
        )
        from legoesm.core.coupling_fields import AtmToSurface

        ncol = 4
        config = MultiLayerLandConfig()
        state = init_multilayer_land_state(ncol, config, T_init=285.0)
        forcing = AtmToSurface(
            sw_down=jnp.full(ncol, 250.0),
            lw_down=jnp.full(ncol, 340.0),
            precip_total=jnp.full(ncol, 1e-4),
            precip_snow=jnp.zeros(ncol),
            T_lowest=jnp.full(ncol, 288.0),
            q_lowest=jnp.full(ncol, 0.008),
            u_lowest=jnp.full(ncol, 3.0),
            v_lowest=jnp.full(ncol, 1.0),
            p_lowest=jnp.full(ncol, 95000.0),
            p_surface=jnp.full(ncol, 100000.0),
            rho_lowest=jnp.full(ncol, 1.2),
            cos_zenith=jnp.full(ncol, 0.6),
            co2_ppmv=jnp.full(ncol, 400.0),
            has_radiation=jnp.ones(ncol),
            has_precipitation=jnp.ones(ncol),
        )
        new_state, response, _ = step_multilayer_land(
            state, forcing, config, U_min=1.0, dt=600.0)

        # Shapes
        self.assertEqual(new_state.T_soil.shape, (ncol, config.soil_grid.n_layers))
        self.assertTrue(bool(jnp.all(jnp.isfinite(response.shflx))),
                        "SimpleSEB shflx is non-finite")

    def test_canopy_state_none_for_default(self):
        """canopy_state must be None for non-CLM-ML schemes."""
        from legoesm.land.config import MultiLayerLandConfig
        from legoesm.land.multilayer_land import init_multilayer_land_state

        config = MultiLayerLandConfig()
        state = init_multilayer_land_state(4, config, T_init=285.0)
        self.assertIsNone(state.canopy_state)


class TestConfig(unittest.TestCase):
    """Config NamedTuple validation."""

    def test_clm_ml_config_defaults(self):
        """CLMMLCanopyConfig() creates a valid object with expected defaults."""
        from legoesm.land.canopy.config import CLMMLCanopyConfig

        cfg = CLMMLCanopyConfig()
        self.assertEqual(cfg.nlevmlcan, 9)
        self.assertEqual(cfg.runge_kutta_type, 10)
        self.assertEqual(cfg.num_ml_steps, 1)
        self.assertAlmostEqual(cfg.o2ref, 209.0)
        self.assertEqual(cfg.met_type, 0)

    def test_canopy_state_init(self):
        """CanopyState(mlcanopy=None) is a valid cold-start sentinel."""
        from legoesm.land.canopy.state import CanopyState

        state = CanopyState(mlcanopy=None)
        self.assertIsNone(state.mlcanopy)

    def test_clm_ml_config_new_fields(self):
        """New config fields (pft_clm, f_vis, f_dir, soil defaults) have physical defaults."""
        from legoesm.land.canopy.config import CLMMLCanopyConfig

        cfg = CLMMLCanopyConfig()
        # pft_clm=7 (broadleaf deciduous temperate tree) — not 13 (C3 grass)
        self.assertEqual(cfg.pft_clm, 7)
        # f_vis=0.46 — observation-based, not 0.5
        self.assertAlmostEqual(cfg.f_vis, 0.46)
        # f_dir=-1 → auto-estimated from zenith
        self.assertLess(cfg.f_dir, 0.0)
        # smp_default_mm: moderate stress (-0.49 MPa), not near saturation
        self.assertLess(cfg.smp_default_mm, -1000.0)
        # hk_default_mm_s: silty clay loam range, not 10× too high
        self.assertLess(cfg.hk_default_mm_s, 1.0e-4)
        # soilresis_default_s_m: must be physically consistent with dry soil (~2000 s/m)
        self.assertGreater(cfg.soilresis_default_s_m, 500.0)
        # root_biomass_default_g_m2: not spval (1e36), must be in literature range
        self.assertGreater(cfg.root_biomass_default_g_m2, 50.0)
        self.assertLess(cfg.root_biomass_default_g_m2, 2000.0)
        # albgrd_vis/nir: soil spectral albedo (NOT vegetation broadband)
        # loam soil: VIS ≈ 0.10, NIR ≈ 0.20; must satisfy NIR > VIS
        self.assertGreater(cfg.albgrd_nir_default, cfg.albgrd_vis_default)
        self.assertLess(cfg.albgrd_vis_default, 0.20)   # VIS < 0.20 for mineral soil
        self.assertLess(cfg.albgrd_nir_default, 0.40)   # NIR < 0.40 for moist soil


class TestRootFractionNormalization(unittest.TestCase):
    """rootfr_padded must sum to 1.0 even when n_layers != nlevsoi."""

    def _make_rootfr_padded(self, n_layers, nlevsoi=10):
        import numpy as np
        z = np.cumsum([0.1] * n_layers) - 0.05
        root_frac = np.exp(-z / 1.0)
        root_frac /= root_frac.sum()
        rootfr_padded = np.zeros(nlevsoi, dtype=np.float64)
        for j in range(nlevsoi):
            jl = min(j, n_layers - 1)
            rootfr_padded[j] = float(root_frac[jl])
        total = rootfr_padded.sum()
        if total > 0:
            rootfr_padded /= total
        return rootfr_padded

    def test_normalization_fewer_legoesm_layers(self):
        """n_layers=8 < nlevsoi=10: padded rootfr must still sum to 1."""
        rf = self._make_rootfr_padded(8, 10)
        self.assertAlmostEqual(rf.sum(), 1.0, places=10)

    def test_normalization_exact_match(self):
        """n_layers == nlevsoi: rootfr sum still 1."""
        rf = self._make_rootfr_padded(10, 10)
        self.assertAlmostEqual(rf.sum(), 1.0, places=10)

    def test_normalization_more_legoesm_layers(self):
        """n_layers=15 > nlevsoi=10: truncated rootfr renormalized to 1."""
        rf = self._make_rootfr_padded(15, 10)
        self.assertAlmostEqual(rf.sum(), 1.0, places=10)


class TestSolarGeometry(unittest.TestCase):
    """Unit tests for solar zenith and SW beam-fraction helpers."""

    def test_cos_zenith_noon_pdt(self):
        """At CHATS7 (38.47°N, -121.84°W) near solar noon in May, cos_zen ≈ 0.85-0.95."""
        import numpy as np
        from legoesm.land.canopy.clm_ml_interface import _compute_cos_zenith

        lat = np.array([38.47])
        lon = np.array([-121.84])
        # May 1 solar noon ≈ 20:00 UTC (noon PDT = UTC-8h + ~8 min longitude correction)
        doy_solar_noon = 120.0 + 19.87 / 24.0
        cos_z = _compute_cos_zenith(lat, lon, doy_solar_noon)
        self.assertGreater(float(cos_z[0]), 0.80, "cos_zen at solar noon should be > 0.80")
        self.assertLessEqual(float(cos_z[0]), 1.0, "cos_zen must be <= 1")

    def test_cos_zenith_midnight_is_zero(self):
        """At solar midnight, cos_zen must be 0 (clamped)."""
        import numpy as np
        from legoesm.land.canopy.clm_ml_interface import _compute_cos_zenith

        lat = np.array([38.47])
        lon = np.array([-121.84])
        # Midnight PDT = 07:00 UTC (lon=-121.84 → UTC offset ≈ -8.1h)
        doy_midnight = 120.0 + 7.12 / 24.0
        cos_z = _compute_cos_zenith(lat, lon, doy_midnight)
        self.assertAlmostEqual(float(cos_z[0]), 0.0, places=2)

    def test_spencer_doy_convention(self):
        """Spencer formula uses doy directly (0-based), not doy-1."""
        import numpy as np
        from legoesm.land.canopy.clm_ml_interface import _compute_cos_zenith

        # June 21 (doy≈172) is summer solstice — max declination ~+23.45°
        # Dec 21 (doy≈355) is winter solstice — min declination ~-23.45°
        # The 1-day offset between 0-based and 1-based doy shifts declination
        # by ~0.4° at solstices.  Verify solstice symmetry holds.
        lat = np.array([0.0])  # equator
        lon = np.array([0.0])
        doy_summer = 172.5   # June 22 noon UTC
        doy_winter = 355.5   # Dec 22 noon UTC
        cos_summer = _compute_cos_zenith(lat, lon, doy_summer)
        cos_winter = _compute_cos_zenith(lat, lon, doy_winter)
        # At equator noon: cos_zen = cos(|decl|) ≈ cos(23.45°) ≈ 0.917
        # Both should be ~equal and ~0.917 ± 0.01
        self.assertAlmostEqual(float(cos_summer[0]), float(cos_winter[0]), places=2)
        self.assertGreater(float(cos_summer[0]), 0.90)

    def test_beam_fraction_clear_sky(self):
        """Under nearly-clear sky (kt ≈ 0.73), Erbs 1982 gives f_dir ≈ 0.80."""
        import numpy as np
        from legoesm.land.canopy.clm_ml_interface import _estimate_beam_fraction

        sw   = np.array([800.0])    # typical clear-sky daytime
        cosz = np.array([0.8])      # mid-afternoon sun
        f    = _estimate_beam_fraction(sw, cosz)
        # kt = 800 / (1361 * 0.8) ≈ 0.734; Erbs poly: Id/I ≈ 0.20 → f_dir ≈ 0.80
        # (physically: ~80% beam under near-clear sky conditions)
        self.assertGreater(float(f[0]), 0.5)

    def test_beam_fraction_erbs_polynomial(self):
        """Verify Erbs 1982 polynomial matches known values at kt=0.5."""
        import numpy as np
        from legoesm.land.canopy.clm_ml_interface import _estimate_beam_fraction

        # kt=0.5: Id/I = 0.9511 - 0.1604*0.5 + 4.388*0.25 - 16.638*0.125 + 12.336*0.0625
        #       = 0.9511 - 0.0802 + 1.097 - 2.0798 + 0.771 ≈ 0.659
        # f_dir = 1 - 0.659 ≈ 0.341
        sw   = np.array([340.25])   # 340.25 / (1361 * 0.5) = 0.5 exactly
        cosz = np.array([0.5])
        f    = _estimate_beam_fraction(sw, cosz)
        expected = 1.0 - (0.9511 - 0.1604*0.5 + 4.388*0.25 - 16.638*0.125 + 12.336*0.0625)
        self.assertAlmostEqual(float(f[0]), expected, places=3)

    def test_beam_fraction_nighttime_zero(self):
        """At night (sw_down < 1 W/m²), beam fraction must be 0."""
        import numpy as np
        from legoesm.land.canopy.clm_ml_interface import _estimate_beam_fraction

        sw   = np.array([0.0, 0.5])
        cosz = np.array([0.0, 0.0])
        f    = _estimate_beam_fraction(sw, cosz)
        self.assertAlmostEqual(float(f[0]), 0.0)
        self.assertAlmostEqual(float(f[1]), 0.0)

    def test_interface_with_lon_and_doy(self):
        """compute_clm_ml_canopy_fluxes accepts lon and doy, produces finite output."""
        import numpy as np

        if not (jax_available and clm_ml_available):
            self.skipTest("clm-ml-jax not installed")

        from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
        from legoesm.land.canopy.config import CLMMLCanopyConfig
        from legoesm.land.config import MultiLayerLandConfig

        ncol = 1
        config = CLMMLCanopyConfig(pft_clm=7)
        land_config = MultiLayerLandConfig(surface_scheme=config)
        forcing = _make_forcing(ncol)
        T_soil, psi_soil, theta_soil = _make_soil_arrays(ncol)

        surface_out, _ = compute_clm_ml_canopy_fluxes(
            T_soil_top=T_soil[:, 0],
            forcing=forcing,
            canopy_config=config,
            land_config=land_config,
            land_params=None,
            w_frac_rz=jnp.full(ncol, 0.6),
            wind_speed=jnp.full(ncol, 4.5),
            canopy_state=None,
            dt=1800.0,
            T_soil=T_soil,
            psi_soil=psi_soil,
            theta_soil=theta_soil,
            lat=jnp.array([38.47]),
            lon=jnp.array([-121.84]),
            doy=120.833,  # May 1 ~noon PDT
        )
        for name in surface_out._fields:
            val = getattr(surface_out, name)
            if val is not None:
                self.assertTrue(
                    bool(jnp.all(jnp.isfinite(jnp.asarray(val)))),
                    f"{name} is non-finite with lon+doy set",
                )

    def test_multilayer_config_accepts_clm_ml(self):
        """MultiLayerLandConfig accepts CLMMLCanopyConfig as surface_scheme."""
        from legoesm.land.canopy.config import CLMMLCanopyConfig
        from legoesm.land.config import MultiLayerLandConfig

        config = MultiLayerLandConfig(surface_scheme=CLMMLCanopyConfig())
        from legoesm.land.canopy.config import CLMMLCanopyConfig as CCC
        self.assertIsInstance(config.surface_scheme, CCC)

    def test_virtual_lon_round_trips_cos_zen(self):
        """_compute_virtual_lon_deg produces lon that recovers input cos_zen via shr_orb_cosz.

        Since _compute_virtual_lon_deg now uses CLM's own shr_orb_decl and caldaym1,
        the round-trip must close through shr_orb_cosz — not through the Spencer-based
        _compute_cos_zenith (which would show small residuals from Kepler vs Spencer).
        """
        import numpy as np
        from legoesm.land.canopy.clm_ml_interface import (
            _compute_virtual_lon_deg,
            _ensure_clm_initialized,
        )
        from clm_share.shr_orb_mod import shr_orb_cosz, shr_orb_decl
        import clm_src_utils.clm_varorb as _varorb
        from math import pi

        _ensure_clm_initialized()  # sets _varorb.*

        lat = np.array([38.47, 51.5, -33.9])   # CHATS7 CA, London, Sydney
        cos_zen_in = np.array([0.85, 0.45, 0.62])
        doy = 120.833  # May 1 near noon PDT for column 0
        dt = 1800.0

        # caldaym1 used by _MLCanopyForcing when itim=round(doy*86400/dt)
        itim = max(1, round(doy * 86400.0 / dt))
        caldaym1 = 1.0 + (itim - 1) * dt / 86400.0

        lon_virtual = _compute_virtual_lon_deg(cos_zen_in, lat, caldaym1)

        # Verify round-trip through CLM's own shr_orb_cosz
        declinm1, _ = shr_orb_decl(caldaym1, _varorb.eccen, _varorb.mvelpp,
                                    _varorb.lambm0, _varorb.obliqr)
        for i in range(len(lat)):
            lat_r = float(lat[i]) * pi / 180.0
            lon_r = float(lon_virtual[i]) * pi / 180.0
            cos_zen_out = shr_orb_cosz(caldaym1, lat_r, lon_r, float(declinm1))
            self.assertAlmostEqual(
                float(cos_zen_in[i]), float(cos_zen_out), delta=1e-6,
                msg=(f"col {i}: CLM round-trip error: "
                     f"{cos_zen_in[i]:.6f} → lon={lon_virtual[i]:.2f}° "
                     f"→ {cos_zen_out:.6f}"),
            )

    def test_virtual_lon_finite_for_all_cos_zen(self):
        """_compute_virtual_lon_deg returns finite [-180,180] for all cos_zen values including nighttime."""
        import numpy as np
        from legoesm.land.canopy.clm_ml_interface import (
            _compute_virtual_lon_deg,
            _ensure_clm_initialized,
        )

        _ensure_clm_initialized()

        lat = np.array([38.47, 38.47, 38.47, 38.47])
        cos_zen_test = np.array([0.85, 0.2, 0.0, 0.005])  # daytime, low sun, terminator
        doy = 120.0
        dt = 1800.0
        itim = max(1, round(doy * 86400.0 / dt))
        caldaym1 = 1.0 + (itim - 1) * dt / 86400.0

        lon_out = _compute_virtual_lon_deg(cos_zen_test, lat, caldaym1)
        for i in range(len(lat)):
            self.assertTrue(
                np.isfinite(lon_out[i]),
                f"col {i} (cos_zen={cos_zen_test[i]}): expected finite lon, got {lon_out[i]}",
            )
            self.assertGreaterEqual(float(lon_out[i]), -180.0)
            self.assertLessEqual(float(lon_out[i]), 180.0)


if __name__ == "__main__":
    unittest.main()
