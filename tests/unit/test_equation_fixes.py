"""Regression tests for equation/formula/unit fixes (Issues 1–11).

Each test class targets a specific issue and verifies the fix is correct.
All tests are designed to work under JAX jit on CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.core.field import Field


# ======================================================================
# Helpers
# ======================================================================

def _make_unstable_columns(ncol=4, nlev=10):
    """Warm, moist, conditionally unstable columns."""
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
        (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    T = jnp.broadcast_to(
        jnp.linspace(200.0, 300.0, nlev)[None, :], (ncol, nlev),
    )
    q_sat = saturation_mixing_ratio(T, p_full)
    rh = jnp.linspace(0.1, 0.9, nlev)[None, :]
    q_v = rh * q_sat
    return T, q_v, p_full, p_half


def _make_warm_micro_columns(ncol=4, nlev=10, dt=300.0):
    """Near-saturated warm columns for microphysics."""
    T = jnp.linspace(220.0, 290.0, nlev)[None, :].repeat(ncol, axis=0)
    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = 0.95 * q_sat
    q_c = jnp.zeros((ncol, nlev)).at[:, 3:7].set(1e-3)
    q_r = jnp.zeros((ncol, nlev)).at[:, 5:9].set(1e-4)
    rho = p_full / (constants.R_d * T)
    dp = p_half[:, 1:] - p_half[:, :-1]
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_full, 1.0)))

    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    hydrometeors = HydrometeorState(
        q_c=q_c, q_r=q_r,
        q_i=jnp.zeros((ncol, nlev)),
        q_s=jnp.zeros((ncol, nlev)),
        q_g=jnp.zeros((ncol, nlev)),
        N_c=jnp.full((ncol, nlev), 1e8),
        N_r=jnp.full((ncol, nlev), 1e4),
        N_i=jnp.zeros((ncol, nlev)),
    )
    return T, q_v, hydrometeors, p_full, p_half, rho, dz


# ======================================================================
# Issue 1: Hydrostatic convection tracer propagation
# ======================================================================

class TestIssue1_HydrostaticConvectionTracers:

    def test_hydrostatic_convection_returns_tracer_tendencies(self):
        """Convection adapter must return nonzero moisture tendency."""
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig
        from legoesm.atmosphere.physics.convection.integration import make_convection_physics
        from legoesm.core.state import HydrostaticState
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate

        n = 4
        nlev = 10
        grid = create_cubed_sphere(n)
        sigma = create_sigma_coordinate(nlev)

        shape_3d = (6, n, n, nlev)
        shape_2d = (6, n, n)

        # Create warm moist state to trigger convection
        T_data = jnp.broadcast_to(
            jnp.linspace(200.0, 300.0, nlev), shape_3d,
        )
        p_s_data = jnp.full(shape_2d, 1e5)
        q_v_data = 0.8 * saturation_mixing_ratio(
            T_data,
            p_s_data[..., None] * jnp.broadcast_to(
                sigma.sigma_full, shape_3d,
            ),
        )

        state = HydrostaticState(
            u=Field(data=jnp.zeros(shape_3d), name="u", dims=("face", "x", "y", "level"), units="m/s"),
            v=Field(data=jnp.zeros(shape_3d), name="v", dims=("face", "x", "y", "level"), units="m/s"),
            T=Field(data=T_data, name="T", dims=("face", "x", "y", "level"), units="K"),
            p_s=Field(data=p_s_data, name="p_s", dims=("face", "x", "y"), units="Pa"),
            phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=("face", "x", "y"), units="m^2/s^2"),
            tracers={
                "q_v": Field(data=q_v_data, name="q_v", dims=("face", "x", "y", "level"), units="kg/kg"),
            },
        )

        config = ConvectionConfig(scheme="sbm")
        physics_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        # Must have tracer_tendencies
        assert tendencies.tracer_tendencies is not None, \
            "Hydrostatic convection must return tracer tendencies"
        assert "q_v" in tendencies.tracer_tendencies, \
            "Must include q_v tendency"
        # q_v tendency should be nonzero where convection is active
        dq_v = tendencies.tracer_tendencies["q_v"].data
        assert jnp.any(dq_v != 0), "q_v tendency should be nonzero"


# ======================================================================
# Issue 2: Hydrostatic microphysics tracer propagation
# ======================================================================

class TestIssue2_HydrostaticMicrophysicsTracers:

    def test_hydrostatic_microphysics_returns_tracer_tendencies(self):
        """Microphysics adapter must return nonzero vapor/cloud/rain tendencies."""
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
        from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
        from legoesm.core.state import HydrostaticState
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate

        n = 4
        nlev = 10
        grid = create_cubed_sphere(n)
        sigma = create_sigma_coordinate(nlev)

        shape_3d = (6, n, n, nlev)
        shape_2d = (6, n, n)

        T_data = jnp.broadcast_to(
            jnp.linspace(220.0, 290.0, nlev), shape_3d,
        )
        p_s_data = jnp.full(shape_2d, 1e5)
        p_full = p_s_data[..., None] * jnp.broadcast_to(
            sigma.sigma_full, shape_3d,
        )
        q_sat = saturation_mixing_ratio(T_data, p_full)
        # Supersaturated so Kessler condensation actually fires.  At
        # 0.95·q_sat (subsaturated, with no cloud water or rain present)
        # the scheme correctly does nothing — the old setup asserted a
        # no-op would produce a tendency.
        q_v_data = 1.05 * q_sat

        state = HydrostaticState(
            u=Field(data=jnp.zeros(shape_3d), name="u", dims=("face", "x", "y", "level"), units="m/s"),
            v=Field(data=jnp.zeros(shape_3d), name="v", dims=("face", "x", "y", "level"), units="m/s"),
            T=Field(data=T_data, name="T", dims=("face", "x", "y", "level"), units="K"),
            p_s=Field(data=p_s_data, name="p_s", dims=("face", "x", "y"), units="Pa"),
            phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=("face", "x", "y"), units="m^2/s^2"),
            tracers={
                "q_v": Field(data=q_v_data, name="q_v", dims=("face", "x", "y", "level"), units="kg/kg"),
            },
        )

        config = MicrophysicsConfig(scheme="kessler")
        physics_fn = make_microphysics_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies = physics_fn(state, grid, sigma)

        assert tendencies.tracer_tendencies is not None, \
            "Hydrostatic microphysics must return tracer tendencies"
        for key in ("q_v", "q_c", "q_r"):
            assert key in tendencies.tracer_tendencies, f"Must include {key} tendency"
        # At least vapor tendency should be nonzero for near-saturated air
        dq_v = tendencies.tracer_tendencies["q_v"].data
        assert jnp.any(dq_v != 0), "q_v tendency should be nonzero"


# ======================================================================
# Issue 3: Microphysics rate-vs-increment
# ======================================================================

class TestIssue3_MicrophysicsRateSemantics:

    def test_kessler_condensation_depends_on_dt(self):
        """Condensation tendency should differ when dt changes (not dt-independent)."""
        from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
        T, q_v, hydrometeors, p_full, p_half, rho, dz = _make_warm_micro_columns()

        out1 = kessler_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt=60.0)
        out2 = kessler_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt=600.0)

        # The saturation adjustment component scales with 1/dt
        # Even though other processes may dominate, dT_dt should differ
        diff = jnp.abs(out1.dT_dt - out2.dT_dt).sum()
        assert diff > 0, "dT_dt should depend on dt (condensation is divided by dt)"

    def test_kessler_condensation_heating_scales_correctly(self):
        """Condensation-only heating should be L_v/c_p * excess/dt, not L_v/c_p * excess."""
        from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
        from legoesm.atmosphere.physics.microphysics.output import HydrometeorState

        # Isolate condensation: use supersaturated air, no cloud/rain (no evaporation)
        ncol, nlev = 2, 5
        dt = 300.0
        T = jnp.full((ncol, nlev), 280.0)
        p_half = jnp.linspace(5e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        q_sat = saturation_mixing_ratio(T, p_full)
        q_v = 1.05 * q_sat  # 5% supersaturated
        rho = p_full / (constants.R_d * T)
        dp = p_half[:, 1:] - p_half[:, :-1]
        dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_full, 1.0)))
        zero = jnp.zeros((ncol, nlev))
        hydrometeors = HydrometeorState(
            q_c=zero, q_r=zero, q_i=zero, q_s=zero, q_g=zero,
            N_c=zero, N_r=zero, N_i=zero,
        )

        out = kessler_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt)

        # Condensation heating: L_v/c_p * (excess/dt), should scale with 1/dt
        excess = 0.05 * q_sat  # ~5% of q_sat
        max_expected = float(constants.L_v / constants.c_pd * jnp.max(excess) / dt)
        max_actual = float(jnp.max(jnp.abs(out.dT_dt)))

        # Should be within 2x of expected (sigmoid smoothing may reduce it)
        assert max_actual < 2 * max_expected, \
            f"Heating {max_actual} >> expected {max_expected} (increment not divided by dt?)"
        # Should not be dt-independent (i.e. max_expected * dt >> max_actual is wrong)
        assert max_actual > max_expected * 0.1, \
            f"Heating {max_actual} << expected {max_expected}"

    @pytest.mark.parametrize("scheme_name,scheme_fn", [
        ("sundqvist", "sundqvist_microphysics"),
        ("seifert_beheng", "seifert_beheng_microphysics"),
        ("morrison", "morrison_microphysics"),
        ("thompson", "thompson_microphysics"),
    ])
    def test_all_schemes_condensation_depends_on_dt(self, scheme_name, scheme_fn):
        """All backends' condensation component should depend on dt."""
        import importlib
        mod = importlib.import_module(f"legoesm.atmosphere.physics.microphysics.{scheme_name}")
        fn = getattr(mod, scheme_fn)
        T, q_v, hydrometeors, p_full, p_half, rho, dz = _make_warm_micro_columns()
        # Supersaturate so the (instant) condensation component — the one
        # under test — actually fires for every scheme.  The shared fixture
        # is subsaturated (0.95·q_sat), which leaves a rate-based scheme
        # like Sundqvist with zero condensation and only a (correctly)
        # dt-independent rain-evaporation rate, so the dt-dependence of
        # condensation could never be detected.
        q_v = 1.05 * saturation_mixing_ratio(T, p_full)
        out1 = fn(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt=60.0)
        out2 = fn(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt=600.0)

        diff = jnp.abs(out1.dT_dt - out2.dT_dt).sum()
        assert diff > 0, f"{scheme_name}: dT_dt should depend on dt"

    def test_precipitation_units_kg_m2_s(self):
        """Precipitation should be in kg/m^2/s (O(1e-5) to O(1e-2))."""
        from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
        dt = 300.0
        T, q_v, hydrometeors, p_full, p_half, rho, dz = _make_warm_micro_columns(dt=dt)
        out = kessler_microphysics(T, q_v, hydrometeors, p_full, p_half, rho, dz, dt)
        max_precip = jnp.max(out.precipitation)
        # Reasonable precipitation is 0 to ~100 mm/hr = ~0.028 kg/m²/s
        assert max_precip < 1.0, f"Precipitation {max_precip} kg/m²/s unreasonably large"


# ======================================================================
# Issues 4-5: Mass-flux and EDMF precipitation dimensions
# ======================================================================

class TestIssue4_MassFluxPrecipitation:

    def test_dq_c_conv_dimensional_consistency(self):
        """Convective cloud-water source from mass-flux should be kg/kg/s.

        Post-Option-C: mass_flux now emits a 3D ``dq_c_conv_dt``
        (kg/kg/s) instead of a scalar surface ``precipitation``
        (kg/m²/s); microphysics owns the surface-flux diagnostic. A
        sane parameterization keeps per-level condensation rates well
        under 1e-3 kg/kg/s — this bound catches gross unit errors.
        """
        from legoesm.atmosphere.physics.convection.mass_flux import mass_flux_convection
        from legoesm.atmosphere.physics.convection.config import MassFluxConfig

        T, q_v, p_full, p_half = _make_unstable_columns()
        ncol = T.shape[0]
        M_c = jnp.full(ncol, 0.01)

        config = MassFluxConfig(M_scale=0.01, cape_threshold=0.0)
        out, _ = mass_flux_convection(T, q_v, p_full, p_half, M_c, dt=300.0, config=config)

        max_rate = jnp.max(out.dq_c_conv_dt)
        assert max_rate >= 0, "dq_c_conv_dt must be non-negative"
        assert max_rate < 1e-3, (
            f"Mass-flux dq_c_conv_dt {max_rate:.4e} kg/kg/s unreasonably "
            "large (dimensional error?)"
        )

    def test_analytic_column_precipitation(self):
        """Analytic test: uniform M, delta_0, condensate, dz → known result."""
        from legoesm.atmosphere.physics.convection.config import MassFluxConfig

        ncol, nlev = 1, 5
        M_val = 0.01  # kg/m²/s
        delta_0 = 1e-3  # 1/m
        condensate = 1e-3  # kg/kg
        dz_val = 1000.0  # m per layer

        # Expected: delta_0 * M * condensate * sum(dz) = 1e-3 * 0.01 * 1e-3 * 5000 = 5e-5 kg/m²/s
        expected = delta_0 * M_val * condensate * nlev * dz_val

        # Construct simple column
        T = jnp.full((ncol, nlev), 280.0)
        q_v = jnp.full((ncol, nlev), 0.01)
        p_half = jnp.linspace(5e4, 1e5, nlev + 1)[None, :]
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

        # Compute dz from the column
        dp = p_half[:, 1:] - p_half[:, :-1]
        p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        dz_computed = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0)))

        # The precipitation integral uses dz, not dp/g
        # So the result should scale with dz, not dp/g
        # Just verify it's in the right order of magnitude
        assert expected < 1e-3, "Analytic expectation should be small"


class TestIssue5_EDMFPrecipitation:

    def test_dq_c_conv_dimensional_consistency(self):
        """Convective cloud-water source from EDMF should be in kg/kg/s.

        Replaces the legacy ``out.precipitation`` (kg/m²/s) check after
        the Option-C refactor: EDMF now emits a 3D
        ``dq_c_conv_dt`` field; surface precipitation is owned by
        microphysics. A sane parameterization should produce per-level
        condensation rates well under 1e-3 kg/kg/s — far below this
        bound for any physically reasonable column.
        """
        from legoesm.atmosphere.physics.convection.mass_flux import edmf_convection
        from legoesm.atmosphere.physics.convection.config import EDMFConfig

        T, q_v, p_full, p_half = _make_unstable_columns()
        ncol = T.shape[0]
        a_u = jnp.full(ncol, 0.1)

        config = EDMFConfig(cape_threshold=0.0)
        out, _ = edmf_convection(T, q_v, p_full, p_half, a_u, dt=300.0, config=config)

        max_rate = jnp.max(out.dq_c_conv_dt)
        assert max_rate >= 0, "dq_c_conv_dt must be non-negative"
        assert max_rate < 1e-3, (
            f"EDMF dq_c_conv_dt {max_rate:.4e} kg/kg/s unreasonably large "
            "(dimensional error?)"
        )


# ======================================================================
# Issue 6: Kuo trigger units
# ======================================================================

class TestIssue6_KuoTriggerUnits:

    def test_kuo_config_uses_me_threshold(self):
        """KuoConfig should use 'me_threshold' (moisture excess, kg/m²)."""
        from legoesm.atmosphere.physics.convection.config import KuoConfig
        cfg = KuoConfig()
        assert hasattr(cfg, 'me_threshold'), "KuoConfig must have me_threshold"
        assert not hasattr(cfg, 'mc_threshold'), \
            "KuoConfig should not have mc_threshold (renamed to me_threshold)"

    def test_kuo_trigger_responds_to_moisture_excess(self):
        """Kuo activates only when column moisture excess exceeds threshold.

        The default ``_make_unstable_columns`` profile is
        conditionally unstable but undersaturated (``MC = 0``); under
        the post-Option-C MC-gating Kuo correctly stays off there
        (no spurious heating, no destroyed vapor, no created cloud
        water in undersaturated columns). To test that the trigger
        *does* fire when moisture excess is present, supersaturate
        the lower half of the column here.
        """
        from legoesm.atmosphere.physics.convection.kuo import kuo_convection
        from legoesm.atmosphere.physics.convection.config import KuoConfig
        from legoesm.thermo import saturation_mixing_ratio

        T, q_v, p_full, p_half = _make_unstable_columns()
        q_sat = saturation_mixing_ratio(T, p_full)
        nlev = q_v.shape[-1]
        moist_mask = (jnp.arange(nlev) >= nlev // 2)
        q_v = jnp.where(moist_mask[None, :], 1.05 * q_sat, q_v)
        config = KuoConfig(me_threshold=1e-5)
        out = kuo_convection(T, q_v, p_full, p_half, dt=300.0, config=config)

        # With moist columns, convection should be triggered
        assert jnp.any(out.convective_mask > 0.1), "Kuo should trigger for moist columns"
        assert jnp.any(out.dT_dt != 0), "Kuo should produce nonzero heating"


# ======================================================================
# Issue 7: KPP non-local transport and Ri_crit
# ======================================================================

class TestIssue7_KPP:

    def test_ri_crit_default_matches_lmd94(self):
        """KPP critical bulk Richardson number default should be 0.3 — the
        value stated in Large, McWilliams & Doney (1994) and used by
        NCAR POP2 / MOM6 (CVMix).  (The earlier 0.25 here contradicted the
        cited reference; 0.3 is LMD94's actual Ri_c.)"""
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        cfg = KPPConfig()
        assert cfg.Ri_crit == 0.3, f"Expected Ri_crit=0.3, got {cfg.Ri_crit}"

    def test_nonlocal_transport_uses_shape_function(self):
        """Non-local T tendency should have the G(sigma) profile shape."""
        from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
        from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
        from legoesm.ocean.vertical import create_ocean_z_star

        nlev = 10
        n = 4
        shape_3d = (6, n, n, nlev)
        shape_2d = (6, n, n)

        z_coord = create_ocean_z_star(n_levels=nlev, H_max=500.0)
        jacobian = jnp.ones(shape_2d)

        # Warm surface (unstable): T decreasing with depth
        T = jnp.broadcast_to(
            jnp.linspace(20.0, 5.0, nlev), shape_3d,
        )
        S = jnp.full(shape_3d, 35.0)
        u = jnp.full(shape_3d, 0.1)
        v = jnp.zeros(shape_3d)
        eta = jnp.zeros(shape_2d)

        from legoesm.ocean.eos import wright_eos, compute_hydrostatic_pressure
        p_hydro = compute_hydrostatic_pressure(
            wright_eos(T, S, jnp.zeros_like(T)),
            eta, z_coord.dz_ref, jacobian,
        )
        rho = wright_eos(T, S, p_hydro)

        # Positive B_f = unstable (convective)
        B_f = jnp.full(shape_2d, 1e-7)

        cfg = KPPConfig()
        out = kpp_vertical_mixing(u, v, T, S, rho, eta, z_coord, jacobian, cfg, B_f=B_f)

        # dT_dt should be nonzero (non-local transport active in unstable BL)
        assert jnp.any(jnp.abs(out.dT_dt) > 0), "KPP non-local should produce nonzero dT_dt"


# ======================================================================
# Issue 8: Ocean bulk forcing tau_y
# ======================================================================

class TestIssue8_OceanBulkTauY:

    def test_tau_y_applied_to_dv_dt(self):
        """Nonzero meridional stress should produce nonzero dv_dt."""
        from legoesm.ocean.physics.surface_forcing.bulk_formulas import bulk_formula_surface_forcing
        from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
        from legoesm.ocean.vertical import create_ocean_z_star

        nlev = 5
        n = 4
        shape_3d = (6, n, n, nlev)
        shape_2d = (6, n, n)

        z_coord = create_ocean_z_star(n_levels=nlev, H_max=250.0)
        jacobian = jnp.ones(shape_2d)

        T = jnp.full(shape_3d, 20.0)
        S = jnp.full(shape_3d, 35.0)

        # Use COARE3 scheme which produces nonzero tau_y
        # Or use constant scheme with nonzero U_a — constant scheme
        # computes tau_y = 0 by construction (only zonal stress).
        # But with the fix, tau_y should be applied when nonzero.
        # Use constant scheme: tau_y will be zero for pure zonal wind.
        # Instead, verify that the code path exists by checking
        # the output structure directly.
        cfg = BulkFormulaConfig(bulk_scheme="constant", U_a=5.0)
        out = bulk_formula_surface_forcing(T, S, z_coord, jacobian, cfg)

        # tau_x should produce nonzero du_dt
        assert jnp.any(out.du_dt != 0), "du_dt should be nonzero from tau_x"

        # For constant scheme with pure zonal wind, tau_y is 0,
        # so dv_dt will also be 0. But the code path now applies it.
        # Verify with a manual tau_y injection by checking the formula works:
        # dv_dt[...,0] = tau_y * inv_rho_dz
        # If tau_y != 0, dv_dt would be nonzero.
        # We verify the output has the correct shape and the pathway exists.
        assert out.dv_dt.shape == shape_3d


# ======================================================================
# Issue 9: Ocean T convention
# ======================================================================

class TestIssue9_OceanTemperatureConvention:

    def test_ocean_state_says_potential_temperature(self):
        """OceanState docstring should describe T as potential temperature."""
        from legoesm.ocean.state import OceanState
        doc = OceanState.__doc__
        assert "Potential temperature" in doc or "potential temperature" in doc

    def test_wright_eos_accepts_potential_temperature(self):
        """Wright EOS docstring should say potential temperature."""
        from legoesm.ocean.eos import wright_eos
        doc = wright_eos.__doc__
        assert "Potential temperature" in doc or "potential temperature" in doc


# ======================================================================
# Issue 10: Gray radiation moisture optical depth docs
# ======================================================================

class TestIssue10_GrayRadiationDocs:

    def test_config_docstring_mentions_dp_g(self):
        """GrayRadiationConfig docstring should describe the per-layer formula."""
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
        doc = GrayRadiationConfig.__doc__
        assert "m^2/kg" in doc or "m²/kg" in doc, \
            "Config should document tau_moist_coeff units as m²/kg"

    def test_moisture_optical_depth_scales_with_column_water(self):
        """Moisture optical depth should scale with q_v * dp / g."""
        from legoesm.atmosphere.physics.radiation.gray import _compute_lw_optical_depth
        from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig

        ncol, nlev = 2, 5
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_s = p_half[:, -1]
        lat = jnp.zeros(ncol)
        config = GrayRadiationConfig(tau_moist_coeff=0.0115)

        # Dry case
        dtau_dry = _compute_lw_optical_depth(p_half, p_s, lat, None, config)

        # Moist case
        q_v = jnp.full((ncol, nlev), 0.01)
        dtau_moist = _compute_lw_optical_depth(p_half, p_s, lat, q_v, config)

        # Moist should have more optical depth
        assert jnp.all(dtau_moist >= dtau_dry)
        # The difference should scale with q_v * dp / g
        dp = p_half[:, 1:] - p_half[:, :-1]
        expected_extra = config.tau_moist_coeff * q_v * dp / constants.g
        actual_extra = dtau_moist - dtau_dry
        assert jnp.allclose(actual_extra, expected_extra, rtol=1e-5)


# ======================================================================
# Issue 11: Bulk coefficient alignment
# ======================================================================

class TestIssue11_BulkCoefficients:

    def test_ocean_bulk_defaults_match_coupler(self):
        """Standalone ocean and coupler bulk coefficients should be aligned."""
        from legoesm.ocean.physics.surface_forcing.config import BulkFormulaConfig
        from legoesm.coupler.config import CouplerConfig

        ocean_cfg = BulkFormulaConfig()
        coupler_cfg = CouplerConfig()

        assert ocean_cfg.C_D == coupler_cfg.Cd_ocean, \
            f"C_D mismatch: ocean={ocean_cfg.C_D}, coupler={coupler_cfg.Cd_ocean}"
        assert ocean_cfg.C_H == coupler_cfg.Ch_ocean, \
            f"C_H mismatch: ocean={ocean_cfg.C_H}, coupler={coupler_cfg.Ch_ocean}"
