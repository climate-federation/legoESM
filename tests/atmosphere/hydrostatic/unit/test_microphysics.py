"""Unit tests for the microphysics module.

Tests all 6 backends (Kessler, Sundqvist, Seifert-Beheng, Morrison,
Thompson, ML emulator) and the integration bridge for hydrostatic
and non-hydrostatic dycores.
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.microphysics.config import (
    MicrophysicsConfig,
    KesslerConfig,
    SundqvistConfig,
    SeifertBehengConfig,
    MorrisonConfig,
    ThompsonConfig,
    MicrophysicsMLEmulatorConfig,
)
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    make_zero_hydrometeors,
    make_zero_output,
    sedimentation_tendency,
)
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.atmosphere.physics.microphysics.ml_emulator import (
    ml_microphysics,
    MicrophysicsEmulator,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.thermo import saturation_mixing_ratio
from legoesm import constants


# ======================================================================
# Test helpers
# ======================================================================

def _make_warm_columns(ncol=4, nlev=10):
    """Create warm, near-saturated columns for testing warm-rain schemes."""
    # Temperature profile: 290K at surface, 220K at top
    T = jnp.linspace(220.0, 290.0, nlev)[None, :].repeat(ncol, axis=0)
    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    # Near-saturated vapor
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = 0.95 * q_sat

    # Cloud and rain water at mid levels
    q_c = jnp.zeros((ncol, nlev))
    q_c = q_c.at[:, 3:7].set(1e-3)
    q_r = jnp.zeros((ncol, nlev))
    q_r = q_r.at[:, 5:9].set(1e-4)

    rho = p_full / (constants.R_d * T)
    dp = p_half[:, 1:] - p_half[:, :-1]
    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_full, 1.0))
    dz = jnp.abs(dz)

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


def _make_cold_columns(ncol=4, nlev=10):
    """Create cold columns with ice and snow for testing mixed-phase schemes."""
    T = jnp.linspace(200.0, 265.0, nlev)[None, :].repeat(ncol, axis=0)
    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = 0.9 * q_sat

    q_c = jnp.zeros((ncol, nlev)).at[:, 6:9].set(5e-4)
    q_r = jnp.zeros((ncol, nlev)).at[:, 7:9].set(5e-5)
    q_i = jnp.zeros((ncol, nlev)).at[:, 2:6].set(1e-4)
    q_s = jnp.zeros((ncol, nlev)).at[:, 3:7].set(5e-5)
    q_g = jnp.zeros((ncol, nlev)).at[:, 4:7].set(2e-5)

    rho = p_full / (constants.R_d * T)
    dp = p_half[:, 1:] - p_half[:, :-1]
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.clip(p_full, 1.0)))

    hydrometeors = HydrometeorState(
        q_c=q_c, q_r=q_r, q_i=q_i, q_s=q_s, q_g=q_g,
        N_c=jnp.full((ncol, nlev), 1e8),
        N_r=jnp.full((ncol, nlev), 1e4),
        N_i=jnp.full((ncol, nlev), 1e3),
    )

    return T, q_v, hydrometeors, p_full, p_half, rho, dz


def _make_evaporation_columns(ncol=4, nlev=10):
    """Create warm, subsaturated columns with rain to isolate evaporation."""
    T = jnp.full((ncol, nlev), 290.0)
    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = 0.2 * q_sat

    q_r = jnp.full((ncol, nlev), 1e-3)
    z = jnp.zeros((ncol, nlev))
    hydrometeors = HydrometeorState(
        q_c=z,
        q_r=q_r,
        q_i=z,
        q_s=z,
        q_g=z,
        N_c=jnp.full((ncol, nlev), 1e8),
        N_r=jnp.full((ncol, nlev), 1e4),
        N_i=jnp.full((ncol, nlev), 1e3),
    )

    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 500.0)
    return T, q_v, hydrometeors, p_full, p_half, rho, dz


# ======================================================================
# Output helpers
# ======================================================================

class TestOutputHelpers:

    def test_make_zero_hydrometeors_shape(self):
        h = make_zero_hydrometeors(4, 10)
        assert h.q_c.shape == (4, 10)
        assert jnp.all(h.q_c == 0)

    def test_make_zero_output_shape(self):
        out = make_zero_output(4, 10)
        assert out.dT_dt.shape == (4, 10)
        assert out.precipitation.shape == (4,)

    def test_sedimentation_tendency_shape(self):
        q = jnp.ones((4, 10)) * 1e-4
        rho = jnp.ones((4, 10)) * 1.2
        V_t = jnp.ones((4, 10)) * 5.0
        dz = jnp.ones((4, 10)) * 500.0
        tend = sedimentation_tendency(q, rho, V_t, dz)
        assert tend.shape == (4, 10)
        assert jnp.all(jnp.isfinite(tend))

    def test_sedimentation_cfl_positivity(self):
        """With dt and a large V_t·dt/dz, q_new must stay non-negative.

        Without the dt-aware flux limiter introduced in clean_physics
        iter-1, ``q + dt · tendency`` can go negative when the local
        Courant number ``V_t·dt/dz > 1``.
        """
        q = jnp.ones((1, 10)) * 1e-4
        rho = jnp.ones((1, 10)) * 1.2
        V_t = jnp.ones((1, 10)) * 50.0   # huge fall speed
        dz = jnp.ones((1, 10)) * 100.0
        dt = 60.0                          # V_t·dt/dz = 30, far above CFL

        tend = sedimentation_tendency(q, rho, V_t, dz, dt=dt)
        q_new = q + dt * tend
        # With the limiter, positivity holds for ANY local Courant.
        assert jnp.all(q_new >= -1.0e-12)

    def test_sedimentation_surface_flux_conservation(self):
        """Surface flux must equal the column-integrated mass removed.

        With ``return_surface_flux=True`` the precip diagnostic matches
        the dt-limited removal, so column water conservation is exact.
        """
        q = jnp.ones((2, 10)) * 1e-3
        rho = jnp.ones((2, 10)) * 1.2
        V_t = jnp.ones((2, 10)) * 5.0
        dz = jnp.ones((2, 10)) * 500.0
        dt = 60.0

        tend, surf_flux = sedimentation_tendency(
            q, rho, V_t, dz, dt=dt, return_surface_flux=True,
        )
        # Column-integrated mass change per area [kg/m²].
        column_mass_before = jnp.sum(q * rho * dz, axis=-1)
        column_mass_after = jnp.sum((q + dt * tend) * rho * dz, axis=-1)
        delta_per_step = column_mass_before - column_mass_after
        # Surface flux × dt should match the column mass removed.
        assert jnp.allclose(surf_flux * dt, delta_per_step, rtol=1.0e-10)


# ======================================================================
# Kessler tests
# ======================================================================

class TestKessler:

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert out.dT_dt.shape == T.shape
        assert out.precipitation.shape == (T.shape[0],)

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        for field in out:
            if field is None:  # optional fields (e.g. 2-moment number tendencies)
                continue
            assert jnp.all(jnp.isfinite(field)), f"Non-finite in {field}"

    def test_dry_air_zero_tendency(self):
        ncol, nlev = 4, 10
        T = jnp.full((ncol, nlev), 280.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        q_v = jnp.zeros((ncol, nlev))
        rho = p_full / (constants.R_d * T)
        dz = jnp.full((ncol, nlev), 500.0)
        h = make_zero_hydrometeors(ncol, nlev)
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        # With zero moisture, all ice/snow/graupel/number tendencies are exact zero
        assert float(jnp.max(jnp.abs(out.dq_i_dt))) == 0.0
        assert float(jnp.max(jnp.abs(out.dq_s_dt))) == 0.0
        assert float(jnp.max(jnp.abs(out.dq_g_dt))) == 0.0

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()

        def loss(T_in):
            out = kessler_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))

    def test_evaporation_enthalpy_balance(self):
        """Evaporation cooling should balance vapor tendency latent energy."""
        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
        assert float(jnp.max(jnp.abs(residual))) < 1e-6

    def test_total_water_conservation_under_heavy_clamp(self):
        """Total water ``q_v + q_c + q_r`` is conserved by the
        in-scheme tendencies in a HYDROSTATIC column (``dz = dp /
        (ρ g)``).  Sedimentation redistributes vertically and the
        column total changes only through the surface precipitation
        flux.  Constructed so accretion drives the q_c clamp.  Locks
        in the conservation-respecting scaling of matched sink/source
        pairs added by the iter-35 consolidation onto
        ``donor_clamp_scale`` — without it, scaled sinks and unscaled
        sources would lose mass.  Uses a relative tolerance so the
        test passes in both fp32 (default) and fp64
        (``JAX_ENABLE_X64=1``) modes.
        """
        ncol, nlev = 2, 10
        T = jnp.full((ncol, nlev), 290.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        rho = p_full / (constants.R_d * T)
        dp = p_half[:, 1:] - p_half[:, :-1]
        # Hydrostatic layer thickness: dp = ρ g dz so the layer mass
        # ``ρ·dz`` matches the coupler's ``dp/g``.  A non-hydrostatic
        # mismatch would produce a fake conservation residual that
        # has nothing to do with the scheme.
        dz = dp / (rho * constants.g)
        q_sat = saturation_mixing_ratio(T, p_full)
        q_v = q_sat * 1.001
        h = make_zero_hydrometeors(ncol, nlev)
        h = h._replace(
            q_c=jnp.full_like(T, 1e-4),
            q_r=jnp.full_like(T, 5e-3),
        )
        dt = 1200.0
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=dt)
        col_tend = jnp.sum(
            (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt) * dp / constants.g,
            axis=-1,
        )
        residual = col_tend + out.precipitation
        scale = jnp.maximum(jnp.abs(out.precipitation), 1.0e-10)
        rel = jnp.max(jnp.abs(residual) / scale)
        # 1e-6 covers fp32 (~1e-7 epsilon) with margin; in fp64 we get
        # ~1e-15 which trivially clears.
        assert float(rel) < 1.0e-6


# ======================================================================
# Sundqvist tests
# ======================================================================

class TestSundqvist:

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert out.dT_dt.shape == T.shape
        assert out.precipitation.shape == (T.shape[0],)

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        for field in out:
            if field is None:  # optional fields (e.g. 2-moment number tendencies)
                continue
            assert jnp.all(jnp.isfinite(field))

    def test_autoconversion_does_not_drive_qc_negative(self):
        """With dt large enough that auto_rate·dt > 1, an explicit Euler
        step on q_c must NOT drive q_c below zero.  Default
        ``auto_rate = 1e-3`` and ``dt = 1800 s`` give ``auto_rate·dt = 1.8``;
        without the donor clamp added in clean_physics iter-22, the
        autoconversion sink would exceed available ``q_c`` and the
        explicit update ``q_c + dt·dq_c_dt`` would go negative."""
        ncol, nlev = 4, 10
        T = jnp.full((ncol, nlev), 280.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        rho = p_full / (constants.R_d * T)
        dz = jnp.full((ncol, nlev), 500.0)
        q_sat = saturation_mixing_ratio(T, p_full)
        q_v = jnp.full_like(T, 0.5) * q_sat  # subsat, no new condensation
        h = make_zero_hydrometeors(ncol, nlev)
        # Seed a thin cloud water layer
        q_c = jnp.full_like(T, 1.0e-4)
        h = h._replace(q_c=q_c)
        dt = 1800.0  # 30-min step → auto_rate·dt = 1.8
        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=dt)
        q_c_new = q_c + dt * out.dq_c_dt
        # Positivity must hold for any explicit Euler step
        assert jnp.all(q_c_new >= -1.0e-12)

    def test_below_RH_crit_no_condensation(self):
        ncol, nlev = 4, 10
        T = jnp.full((ncol, nlev), 280.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        q_sat = saturation_mixing_ratio(T, p_full)
        q_v = 0.5 * q_sat  # well below RH_crit=0.8
        rho = p_full / (constants.R_d * T)
        dz = jnp.full((ncol, nlev), 500.0)
        h = make_zero_hydrometeors(ncol, nlev)
        # Use very high sharpness to make the sigmoid effectively a step
        config = SundqvistConfig(rh_crit=0.8, sigmoid_sharpness=200.0)
        out = sundqvist_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
        # Condensation should be much smaller than saturated case
        out_sat = sundqvist_microphysics(T, q_sat, h, p_full, p_half, rho, dz, dt=10.0, config=config)
        ratio = float(jnp.max(jnp.abs(out.dq_c_dt))) / float(jnp.max(jnp.abs(out_sat.dq_c_dt)) + 1e-20)
        assert ratio < 0.1

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()

        def loss(T_in):
            out = sundqvist_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))


# ======================================================================
# Seifert-Beheng tests
# ======================================================================

class TestSeifertBeheng:

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert out.dT_dt.shape == T.shape
        assert out.dN_c_dt.shape == T.shape
        assert out.dN_r_dt.shape == T.shape

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        for field in out:
            if field is None:  # optional fields (e.g. 2-moment number tendencies)
                continue
            assert jnp.all(jnp.isfinite(field))

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()

        def loss(T_in):
            out = seifert_beheng_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))

    def test_evaporation_enthalpy_balance(self):
        """Evaporation cooling should balance vapor tendency latent energy."""
        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
        assert float(jnp.max(jnp.abs(residual))) < 1e-6

    def test_total_water_conservation_under_heavy_clamp(self):
        """Seifert-Beheng total water conservation in a hydrostatic
        column under heavy q_c clamp activity.  Locks in the iter-35
        consolidation onto ``donor_clamp_scale``.  Uses a relative
        tolerance so the test passes in both fp32 and fp64."""
        ncol, nlev = 2, 10
        T = jnp.full((ncol, nlev), 290.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        rho = p_full / (constants.R_d * T)
        dp = p_half[:, 1:] - p_half[:, :-1]
        dz = dp / (rho * constants.g)
        q_sat = saturation_mixing_ratio(T, p_full)
        q_v = q_sat * 1.001
        h = make_zero_hydrometeors(ncol, nlev)
        h = h._replace(
            q_c=jnp.full_like(T, 1e-4),
            q_r=jnp.full_like(T, 5e-3),
            N_c=jnp.full_like(T, 1.0e8),
            N_r=jnp.full_like(T, 1.0e3),
        )
        out = seifert_beheng_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=1200.0)
        total_tend = out.dq_v_dt + out.dq_c_dt + out.dq_r_dt
        col = jnp.sum(total_tend * dp / constants.g, axis=-1)
        residual = col + out.precipitation
        scale = jnp.maximum(jnp.abs(out.precipitation), 1.0e-10)
        rel = jnp.max(jnp.abs(residual) / scale)
        assert float(rel) < 1.0e-6


# ======================================================================
# Morrison tests
# ======================================================================

class TestMorrison:

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert out.dT_dt.shape == T.shape
        assert out.dq_i_dt.shape == T.shape
        assert out.dN_i_dt.shape == T.shape

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        for field in out:
            if field is None:  # optional fields (e.g. 2-moment number tendencies)
                continue
            assert jnp.all(jnp.isfinite(field))

    def test_ice_only_below_freezing(self):
        """Ice tendencies should be near-zero when T > T_freeze everywhere."""
        ncol, nlev = 4, 10
        T = jnp.full((ncol, nlev), 290.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        q_v = 0.5 * saturation_mixing_ratio(T, p_full)
        rho = p_full / (constants.R_d * T)
        dz = jnp.full((ncol, nlev), 500.0)
        h = make_zero_hydrometeors(ncol, nlev)
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        # Ice tendencies should be very small above freezing
        assert float(jnp.max(jnp.abs(out.dq_i_dt))) < 1e-6

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()

        def loss(T_in):
            out = morrison_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))

    def test_differentiable_zero_sink_qv_clamp(self):
        """A clear (no-sink) column must not produce NaN gradients
        through the q_v donor clamp.  The naive
        ``q_v / max(0, 1e-30)`` form has VJP ~``-q_v / 1e-60`` which
        overflows in fp32 → NaN even though ``min(1, huge) = 1`` kills
        the forward value.  iter-31 / iter-32 AD-safe floor fixes this."""
        ncol, nlev = 4, 10
        T = jnp.full((ncol, nlev), 290.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        rho = p_full / (constants.R_d * T)
        dz = jnp.full((ncol, nlev), 500.0)
        # Subsat clear column: no condensation, no deposition
        q_sat = saturation_mixing_ratio(T, p_full)
        q_v = 0.3 * q_sat
        h = make_zero_hydrometeors(ncol, nlev)

        def loss(q_v_in):
            out = morrison_microphysics(
                T, q_v_in, h, p_full, p_half, rho, dz, dt=100.0,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(q_v)
        assert jnp.all(jnp.isfinite(grad))

    def test_morrison_fp32_grad_finite_with_tiny_qv_sink(self):
        """Morrison forward+backward fp32 gradients on a cold column
        with a TRULY TINY but positive vapor sink path.

        Inputs are constructed so the q_v clamp sees a nonzero but
        very small ``qv_sink_dt`` (below the iter-32 1e-15 floor):
            T = 240 K (cold, ice physics active)
            N_i = 1e3 /kg (so safe_pow(N_i, 1/3) > 0 → dq_i_dep > 0)
            q_v ≈ q_sat_ice · (1 + 1e-7)  (S_i tiny positive)
        Forward measurement: ``dq_i_dt ≈ 1.2e-18`` → ``qv_sink_dt ≈
        1.2e-17`` which is below 1e-15 (active clamp path).

        Before iter-35 (legacy 1e-30 floor) the fp32 VJP NaN'd.  After
        the consolidated ``donor_clamp_scale`` (1e-15 floor) the
        gradient is finite — confirmed end-to-end through the full
        Morrison routine, not just the helper in isolation."""
        from legoesm.thermo import saturation_mixing_ratio_ice
        from legoesm.atmosphere.physics.microphysics.output import (
            HydrometeorState,
        )
        ncol, nlev = 4, 10
        dt32 = jnp.float32
        T = jnp.full((ncol, nlev), 240.0, dtype=dt32)
        p_half = jnp.linspace(
            1.0e4, 1.0e5, nlev + 1, dtype=dt32,
        )[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        rho = (p_full / (constants.R_d * T)).astype(dt32)
        dz = jnp.full((ncol, nlev), 500.0, dtype=dt32)
        q_sat_i = saturation_mixing_ratio_ice(T, p_full)
        # q_v just above q_sat_ice → tiny S_i.
        q_v = (q_sat_i * jnp.float32(1.0 + 1e-7)).astype(dt32)
        z32 = jnp.zeros((ncol, nlev), dtype=dt32)
        # N_i > 0 so dq_i_dep ∝ N_i^(1/3) is positive (engages the
        # tiny-sink clamp path).  q_i = 0 — the q_i_min_growth floor
        # in dq_i_dep gives a tiny but positive deposition rate.
        h = HydrometeorState(
            q_c=z32, q_r=z32, q_i=z32, q_s=z32, q_g=z32,
            N_c=z32, N_r=z32,
            N_i=jnp.full((ncol, nlev), 1.0e3, dtype=dt32),
        )

        # Verify the forward path actually creates a non-zero but
        # tiny sink — otherwise the test would not exercise the
        # active clamp branch.
        out_fwd = morrison_microphysics(
            T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
        )
        assert float(jnp.max(jnp.abs(out_fwd.dq_v_dt))) > 0.0, (
            "Test setup error: no q_v sink — clamp path not exercised"
        )
        assert float(jnp.max(jnp.abs(out_fwd.dq_v_dt))) < 1e-10, (
            "Test setup error: q_v sink too large — not in tiny regime"
        )

        def loss(q_v_in):
            out = morrison_microphysics(
                T, q_v_in, h, p_full, p_half, rho, dz, dt=10.0,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(q_v)
        assert jnp.all(jnp.isfinite(grad))

    def test_qv_clamp_divisor_floor_protects_VJP(self):
        """Exercises the ACTUAL production donor_clamp_scale helper
        (used by Morrison and Thompson q_v clamps) at a tiny positive
        sink in fp32.

        Codex stop-time review chain (iter-31 → iter-32 → iter-33):
        - iter-31 added a boolean ``sink > 0`` guard.  Did NOT cover
          tiny-positive sinks where ``sink_dt²`` underflows the fp32
          VJP.
        - iter-32 replaced the boolean guard with a divisor floor.
        - iter-33 noted the previous test did not call production
          code.  This iter-34 test calls ``donor_clamp_scale`` from
          ``_warm_rain.py`` directly — the SAME helper now used by
          Morrison and Thompson q_v clamps.
        """
        from legoesm.atmosphere.physics.microphysics._warm_rain import (
            donor_clamp_scale,
        )

        # fp32 inputs, tiny positive sink that — without the floor —
        # would overflow the VJP in fp32.  ``sink·dt = 1e-24 ≪ 1e-15``
        # floor.
        q_v = jnp.asarray(1.0e-3, dtype=jnp.float32)
        sink_total = jnp.asarray(1.0e-25, dtype=jnp.float32)
        dt = jnp.asarray(10.0, dtype=jnp.float32)

        # Production helper called directly.
        grad_q = jax.grad(
            lambda q: donor_clamp_scale(jnp.clip(q, 0.0), sink_total, dt)
        )(q_v)
        grad_s = jax.grad(
            lambda s: donor_clamp_scale(jnp.clip(q_v, 0.0), s, dt)
        )(sink_total)

        assert jnp.isfinite(grad_q), f"q_v gradient NaN: {grad_q}"
        assert jnp.isfinite(grad_s), f"sink gradient NaN: {grad_s}"
        # In the floor regime the scale is exactly 1.0 (min), so its
        # gradient is gated to zero — physically correct "no scaling
        # in inactive regime".
        assert float(jnp.abs(grad_q)) < 1.0e-3
        assert float(jnp.abs(grad_s)) < 1.0e-3

        # Also verify the helper produces a non-trivial scale in the
        # ACTIVE regime (very large sink, small q_v).
        q_v_small = jnp.asarray(1.0e-6, dtype=jnp.float32)
        sink_big = jnp.asarray(1.0e-3, dtype=jnp.float32)
        active_scale = donor_clamp_scale(q_v_small, sink_big, dt)
        # sink·dt = 1e-2 > 1e-15 → divisor = 1e-2, scale = min(1, 1e-4)
        # = 1e-4
        assert float(active_scale) == pytest.approx(1.0e-4, rel=1e-3)

    def test_evaporation_enthalpy_balance(self):
        """Warm-rain evaporation cooling should close latent energy tendency."""
        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
        assert float(jnp.max(jnp.abs(residual))) < 5e-3

    def test_total_water_conservation_under_heavy_clamp(self):
        """Morrison total water conservation in a hydrostatic column
        under heavy q_c / q_i / q_s clamp activity.  Locks in the
        iter-35 consolidation onto ``donor_clamp_scale``.  Uses a
        relative tolerance so the test passes in both fp32 and fp64."""
        ncol, nlev = 2, 10
        T = jnp.full((ncol, nlev), 280.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        rho = p_full / (constants.R_d * T)
        dp = p_half[:, 1:] - p_half[:, :-1]
        dz = dp / (rho * constants.g)
        q_sat = saturation_mixing_ratio(T, p_full)
        q_v = q_sat * 1.001
        h = make_zero_hydrometeors(ncol, nlev)
        h = h._replace(
            q_c=jnp.full_like(T, 1e-4),
            q_r=jnp.full_like(T, 5e-3),
            q_i=jnp.full_like(T, 1e-4),
            q_s=jnp.full_like(T, 1e-3),
            N_i=jnp.full_like(T, 1.0e4),
        )
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=1200.0)
        total_tend = (
            out.dq_v_dt + out.dq_c_dt + out.dq_r_dt
            + out.dq_i_dt + out.dq_s_dt
        )
        col = jnp.sum(total_tend * dp / constants.g, axis=-1)
        residual = col + out.precipitation
        scale = jnp.maximum(jnp.abs(out.precipitation), 1.0e-10)
        rel = jnp.max(jnp.abs(residual) / scale)
        assert float(rel) < 1.0e-6


# ======================================================================
# Thompson tests
# ======================================================================

class TestThompson:

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert out.dT_dt.shape == T.shape
        assert out.dq_g_dt.shape == T.shape

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        for field in out:
            if field is None:  # optional fields (e.g. 2-moment number tendencies)
                continue
            assert jnp.all(jnp.isfinite(field))

    def test_graupel_from_riming(self):
        """With strong riming, graupel tendencies should be nonzero."""
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()
        # Boost cloud water and ice to promote riming
        h = h._replace(
            q_c=jnp.full_like(h.q_c, 5e-3),
            q_i=jnp.full_like(h.q_i, 5e-3),
            q_s=jnp.full_like(h.q_s, 5e-3),
        )
        config = ThompsonConfig(
            rime_coeff=10.0,
            rime_to_graupel_threshold=1e-6,
        )
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0, config=config)
        assert float(jnp.max(jnp.abs(out.dq_g_dt))) > 0

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_cold_columns()

        def loss(T_in):
            out = thompson_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))

    def test_evaporation_enthalpy_balance(self):
        """Warm-rain evaporation cooling should close latent energy tendency."""
        T, q_v, h, p_full, p_half, rho, dz = _make_evaporation_columns()
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0)
        residual = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
        assert float(jnp.max(jnp.abs(residual))) < 5e-3

    def test_total_water_conservation_under_heavy_clamp(self):
        """Thompson total water conservation in a hydrostatic column
        under heavy q_c / q_i / q_s / graupel clamp activity.  Locks
        in the iter-35 consolidation onto ``donor_clamp_scale``.  Uses
        a relative tolerance so the test passes in both fp32 and fp64."""
        ncol, nlev = 2, 10
        T = jnp.full((ncol, nlev), 270.0)
        p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        rho = p_full / (constants.R_d * T)
        dp = p_half[:, 1:] - p_half[:, :-1]
        dz = dp / (rho * constants.g)
        q_sat = saturation_mixing_ratio(T, p_full)
        q_v = q_sat * 1.001
        h = make_zero_hydrometeors(ncol, nlev)
        h = h._replace(
            q_c=jnp.full_like(T, 1e-4),
            q_r=jnp.full_like(T, 1e-3),
            q_i=jnp.full_like(T, 1e-4),
            q_s=jnp.full_like(T, 5e-4),
            q_g=jnp.full_like(T, 5e-4),
            N_i=jnp.full_like(T, 1.0e4),
        )
        out = thompson_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=1200.0)
        total_tend = (
            out.dq_v_dt + out.dq_c_dt + out.dq_r_dt
            + out.dq_i_dt + out.dq_s_dt + out.dq_g_dt
        )
        col = jnp.sum(total_tend * dp / constants.g, axis=-1)
        residual = col + out.precipitation
        scale = jnp.maximum(jnp.abs(out.precipitation), 1.0e-10)
        rel = jnp.max(jnp.abs(residual) / scale)
        assert float(rel) < 1.0e-6


# ======================================================================
# ML Emulator tests
# ======================================================================

class TestMLEmulator:

    def _make_model(self, config=None):
        if config is None:
            config = MicrophysicsMLEmulatorConfig()
        key = jax.random.PRNGKey(config.seed)
        return MicrophysicsEmulator(
            config.n_input, config.n_hidden, config.n_layers,
            config.n_output, key=key,
        )

    def test_output_shapes(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = MicrophysicsMLEmulatorConfig()
        model = self._make_model(config)
        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
                              config=config, model=model)
        assert out.dT_dt.shape == T.shape
        assert out.precipitation.shape == (T.shape[0],)

    def test_precipitation_non_negative(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = MicrophysicsMLEmulatorConfig()
        model = self._make_model(config)
        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
                              config=config, model=model)
        assert jnp.all(out.precipitation >= 0)

    def test_nonzero_tendencies(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = MicrophysicsMLEmulatorConfig()
        model = self._make_model(config)
        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
                              config=config, model=model)
        # Untrained model will still produce nonzero output
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_finite_outputs(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = MicrophysicsMLEmulatorConfig()
        model = self._make_model(config)
        out = ml_microphysics(T, q_v, h, p_full, p_half, rho, dz, dt=10.0,
                              config=config, model=model)
        for field in out:
            if field is None:  # optional fields (e.g. 2-moment number tendencies)
                continue
            assert jnp.all(jnp.isfinite(field))

    def test_differentiable(self):
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = MicrophysicsMLEmulatorConfig()
        model = self._make_model(config)

        def loss(T_in):
            out = ml_microphysics(T_in, q_v, h, p_full, p_half, rho, dz, dt=10.0,
                                  config=config, model=model)
            return jnp.sum(out.dT_dt ** 2)

        grad = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad))


# ======================================================================
# Integration bridge tests
# ======================================================================

class TestIntegrationHydrostatic:

    @pytest.fixture
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)
        return state, grid, sigma

    def test_hydrostatic_shapes(self, setup):
        state, grid, sigma = setup
        config = MicrophysicsConfig(scheme="kessler")
        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
        tend = physics_fn(state, grid, sigma)
        assert tend.dT_dt.data.shape == state.T.data.shape

    def test_none_scheme_zeros(self, setup):
        state, grid, sigma = setup
        config = MicrophysicsConfig(scheme="none")
        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
        tend = physics_fn(state, grid, sigma)
        assert float(jnp.max(jnp.abs(tend.dT_dt.data))) == 0.0

    def test_grad_through_hydrostatic(self, setup):
        state, grid, sigma = setup
        config = MicrophysicsConfig(scheme="kessler")
        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend = physics_fn(s, grid, sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad))

    def test_fast_sbm_through_production_pipeline(self, setup):
        # The bin scheme must run end-to-end through the REAL hydrostatic
        # microphysics factory (not just _get_microphysics_fn): physical
        # shapes, finite tendencies, nonnegative precip.
        state, grid, sigma = setup
        config = MicrophysicsConfig(scheme="fast_sbm")
        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
        tend = physics_fn(state, grid, sigma)
        assert tend.dT_dt.data.shape == state.T.data.shape
        assert jnp.all(jnp.isfinite(tend.dT_dt.data))
        # Moisture tendencies flow via the tracer-tendency dict.
        assert tend.tracer_tendencies is not None
        assert jnp.all(jnp.isfinite(tend.tracer_tendencies["q_v"].data))

    def test_fast_sbm_grad_through_hydrostatic(self, setup):
        state, grid, sigma = setup
        config = MicrophysicsConfig(scheme="fast_sbm")
        physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend = physics_fn(s, grid, sigma)
            return jnp.sum(tend.dT_dt.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad))


class TestIntegrationNonhydrostatic:

    @pytest.fixture
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import (
            create_height_coordinate,
            compute_terrain_metric,
        )
        from legoesm.core.field import Field
        from legoesm.core.state import NonHydrostaticState

        grid = create_cubed_sphere(8)
        height_coord = create_height_coordinate(10, 10000.0)
        z_s = jnp.zeros((6, grid.n, grid.n))
        terrain_metric = compute_terrain_metric(z_s, height_coord)

        nlev = height_coord.n_levels
        n = grid.n
        shape_3d = (6, n, n, nlev)
        shape_w = (6, n, n, nlev + 1)
        shape_2d = (6, n, n)
        n_tracers = 3

        state = NonHydrostaticState(
            u=Field(data=jnp.zeros(shape_3d), name="u",
                    dims=("face", "x", "y", "level"), units="m/s"),
            v=Field(data=jnp.zeros(shape_3d), name="v",
                    dims=("face", "x", "y", "level"), units="m/s"),
            w=Field(data=jnp.zeros(shape_w), name="w",
                    dims=("face", "x", "y", "level_half"), units="m/s"),
            theta_prime=Field(data=jnp.zeros(shape_3d), name="theta_prime",
                              dims=("face", "x", "y", "level"), units="K"),
            rho_prime=Field(data=jnp.zeros(shape_3d), name="rho_prime",
                            dims=("face", "x", "y", "level"), units="kg/m^3"),
            phis=Field(data=jnp.zeros(shape_2d), name="phis",
                       dims=("face", "x", "y"), units="m^2/s^2"),
            tracers=Field(
                data=jnp.zeros((*shape_3d, n_tracers)),
                name="tracers",
                dims=("face", "x", "y", "level", "tracer"),
                units="kg/kg",
            ),
        )
        return state, grid, height_coord, terrain_metric

    def test_nonhydrostatic_shapes(self, setup):
        state, grid, hc, tm = setup
        config = MicrophysicsConfig(scheme="kessler")
        physics_fn = make_microphysics_physics(config, "nonhydrostatic", dt=1.0)
        tend = physics_fn(state, grid, hc, tm)
        assert tend.dtheta_prime_dt.data.shape == state.theta_prime.data.shape
        assert tend.dtracers_dt.data.shape == state.tracers.data.shape

    def test_nonhydrostatic_nonzero_heating(self, setup):
        state, grid, hc, tm = setup
        # Add moisture to trigger microphysics
        tracers = state.tracers.data.at[..., 0].set(0.01)
        state = state._replace(tracers=state.tracers.replace(data=tracers))
        config = MicrophysicsConfig(scheme="kessler")
        physics_fn = make_microphysics_physics(config, "nonhydrostatic", dt=1.0)
        tend = physics_fn(state, grid, hc, tm)
        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))
        assert jnp.all(jnp.isfinite(tend.dtracers_dt.data))


class TestSchemeSelection:

    def test_scheme_selection(self):
        """All 6 scheme strings are accepted by the factory."""
        for scheme in ["kessler", "sundqvist", "seifert_beheng",
                       "morrison", "thompson", "ml_emulator", "none"]:
            config = MicrophysicsConfig(scheme=scheme)
            physics_fn = make_microphysics_physics(config, "hydrostatic", dt=300.0)
            assert callable(physics_fn)

    def test_invalid_scheme_raises(self):
        config = MicrophysicsConfig(scheme="invalid")
        with pytest.raises(ValueError, match="Unknown microphysics scheme"):
            make_microphysics_physics(config, "hydrostatic", dt=300.0)

    def test_invalid_model_type_raises(self):
        config = MicrophysicsConfig(scheme="kessler")
        with pytest.raises(ValueError, match="Unknown model_type"):
            make_microphysics_physics(config, "invalid", dt=300.0)


class TestKesslerNonhydrostatic:

    def test_kessler_nonhydrostatic_tendencies(self):
        """Kessler microphysics via integration bridge on nonhydrostatic state."""
        from legoesm.atmosphere.physics.microphysics.config import KesslerConfig, MicrophysicsConfig
        from legoesm.atmosphere.physics.microphysics.integration import make_microphysics_physics
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import (
            create_height_coordinate,
            compute_terrain_metric,
        )
        from legoesm.core.field import Field
        from legoesm.core.state import NonHydrostaticState

        grid = create_cubed_sphere(8)
        hc = create_height_coordinate(10, 10000.0)
        z_s = jnp.zeros((6, grid.n, grid.n))
        tm = compute_terrain_metric(z_s, hc)

        nlev = hc.n_levels
        n = grid.n
        shape_3d = (6, n, n, nlev)
        shape_w = (6, n, n, nlev + 1)
        shape_2d = (6, n, n)

        state = NonHydrostaticState(
            u=Field(data=jnp.zeros(shape_3d), name="u",
                    dims=("face", "x", "y", "level"), units="m/s"),
            v=Field(data=jnp.zeros(shape_3d), name="v",
                    dims=("face", "x", "y", "level"), units="m/s"),
            w=Field(data=jnp.zeros(shape_w), name="w",
                    dims=("face", "x", "y", "level_half"), units="m/s"),
            theta_prime=Field(data=jnp.zeros(shape_3d), name="theta_prime",
                              dims=("face", "x", "y", "level"), units="K"),
            rho_prime=Field(data=jnp.zeros(shape_3d), name="rho_prime",
                            dims=("face", "x", "y", "level"), units="kg/m^3"),
            phis=Field(data=jnp.zeros(shape_2d), name="phis",
                       dims=("face", "x", "y"), units="m^2/s^2"),
            tracers=Field(
                data=jnp.zeros((*shape_3d, 3)),
                name="tracers",
                dims=("face", "x", "y", "level", "tracer"),
                units="kg/kg",
            ),
        )

        config = KesslerConfig()
        micro_config = MicrophysicsConfig(scheme="kessler", kessler=config)
        physics_fn = make_microphysics_physics(micro_config, model_type="nonhydrostatic", dt=1.0)
        tend = physics_fn(state, grid, hc, tm)
        assert tend.dtheta_prime_dt.data.shape == shape_3d
        assert tend.dtracers_dt.data.shape == (*shape_3d, 3)
        assert jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data))


# ======================================================================
# AMIP integration tests (checkpoint save/load with q_c/q_r)
# ======================================================================

class TestCheckpointWithHydrometeors:
    """Test checkpoint save/load roundtrip with q_c and q_r fields."""

    @pytest.fixture
    def setup(self, tmp_path):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
        from legoesm.forcing.amip_config import (
            AMIPExperimentConfig, save_checkpoint, load_checkpoint,
        )

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)
        n = grid.n
        nlev = 10
        q_v = jnp.full((6, n, n, nlev), 0.005)
        q_c = jnp.full((6, n, n, nlev), 1e-4)
        q_r = jnp.full((6, n, n, nlev), 5e-5)
        config = AMIPExperimentConfig(
            resolution=n, nlev=nlev, microphysics="kessler",
        )
        return state, q_v, q_c, q_r, config, grid, sigma, tmp_path

    def test_roundtrip_with_hydrometeors(self, setup):
        state, q_v, q_c, q_r, config, grid, sigma, tmp_path = setup
        from legoesm.forcing.amip_config import save_checkpoint, load_checkpoint
        path = tmp_path / "ckpt.npz"
        save_checkpoint(path, state, q_v, step=100, day=10.0, config=config,
                        q_c=q_c, q_r=q_r)

        loaded = load_checkpoint(path, grid, sigma)
        assert len(loaded) == 9  # state, q_v, step, day, config, diag, q_c, q_r, carry_aux
        _, _, step, day, _, _, q_c_loaded, q_r_loaded, _ = loaded
        assert step == 100
        assert day == 10.0
        assert q_c_loaded is not None
        assert q_r_loaded is not None
        assert float(jnp.max(jnp.abs(q_c_loaded - q_c))) < 1e-10
        assert float(jnp.max(jnp.abs(q_r_loaded - q_r))) < 1e-10

    def test_roundtrip_without_hydrometeors(self, setup):
        """Old checkpoints without q_c/q_r should load with None."""
        state, q_v, q_c, q_r, config, grid, sigma, tmp_path = setup
        from legoesm.forcing.amip_config import save_checkpoint, load_checkpoint
        path = tmp_path / "ckpt_old.npz"
        # Save without q_c/q_r (old-style)
        save_checkpoint(path, state, q_v, step=50, day=5.0, config=config)

        loaded = load_checkpoint(path, grid, sigma)
        _, _, _, _, _, _, q_c_loaded, q_r_loaded, _ = loaded
        assert q_c_loaded is None
        assert q_r_loaded is None

    def test_config_microphysics_field_roundtrip(self, setup):
        state, q_v, q_c, q_r, config, grid, sigma, tmp_path = setup
        from legoesm.forcing.amip_config import save_checkpoint, load_checkpoint
        path = tmp_path / "ckpt_cfg.npz"
        save_checkpoint(path, state, q_v, step=10, day=1.0, config=config,
                        q_c=q_c, q_r=q_r)
        _, _, _, _, restored_config, _, _, _, _ = load_checkpoint(path, grid, sigma)
        assert restored_config.microphysics == "kessler"


class TestAMIPMicrophysicsConfig:
    """Test AMIPExperimentConfig microphysics field."""

    def test_default_microphysics_none(self):
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        config = AMIPExperimentConfig()
        assert config.microphysics == "none"

    def test_kessler_microphysics(self):
        from legoesm.forcing.amip_config import AMIPExperimentConfig
        config = AMIPExperimentConfig(microphysics="kessler")
        assert config.microphysics == "kessler"

    def test_config_json_roundtrip(self):
        from legoesm.forcing.amip_config import (
            AMIPExperimentConfig, config_to_dict, config_from_dict,
        )
        config = AMIPExperimentConfig(microphysics="sundqvist")
        d = config_to_dict(config)
        assert d["microphysics"] == "sundqvist"
        restored = config_from_dict(d)
        assert restored.microphysics == "sundqvist"


class TestMicrophysicsInPhysicsStep:
    """Test microphysics backend dispatch for AMIP-like operator-split setup."""

    def test_kessler_in_column_physics(self):
        """Kessler produces non-trivial tendencies on moist columns."""
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = KesslerConfig()
        out = kessler_microphysics(T, q_v, h, p_full, p_half, rho, dz,
                                   dt=600.0, config=config)
        # Should have nonzero cloud water and rain tendencies
        assert float(jnp.max(jnp.abs(out.dq_c_dt))) > 0
        assert float(jnp.max(jnp.abs(out.dq_r_dt))) > 0
        # Latent heating should be nonzero
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 0

    def test_multi_step_stability(self):
        """Multiple Kessler steps shouldn't produce NaN."""
        T, q_v, h, p_full, p_half, rho, dz = _make_warm_columns()
        config = KesslerConfig()
        dt = 600.0
        q_c = h.q_c
        q_r = h.q_r

        for _ in range(10):
            h_step = h._replace(q_c=q_c, q_r=q_r)
            out = kessler_microphysics(T, q_v, h_step, p_full, p_half, rho, dz,
                                       dt=dt, config=config)
            q_v = jnp.maximum(q_v + dt * out.dq_v_dt, 0.0)
            q_c = jnp.maximum(q_c + dt * out.dq_c_dt, 0.0)
            q_r = jnp.maximum(q_r + dt * out.dq_r_dt, 0.0)
            T = T + dt * out.dT_dt

        assert jnp.all(jnp.isfinite(T))
        assert jnp.all(jnp.isfinite(q_v))
        assert jnp.all(jnp.isfinite(q_c))
        assert jnp.all(jnp.isfinite(q_r))
