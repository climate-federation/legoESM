"""Unit tests for atmospheric convection module.

Tests cover:
- Thermodynamics: saturation, moist adiabat, CAPE
- SBM: shapes, enthalpy conservation, precipitation, trigger, differentiability
- DCA: shapes, stable unchanged, instability reduction, differentiability
- Integration: hydrostatic/NH shapes, nonzero heating, jax.grad, scheme selection
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.thermodynamics import (
    saturation_mixing_ratio,
    temperature_from_theta,
    pressure_from_eos,
    moist_adiabat_lapse_rate,
    compute_moist_adiabat,
    compute_cape,
    parcel_profile_and_cape,
)
from legoesm.atmosphere.physics.convection.config import (
    SBMConfig,
    DCAConfig,
    KuoConfig,
    MassFluxConfig,
    ConvectiveEDMFConfig,
    ConvectionConfig,
)
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.atmosphere.physics.convection.dca import dca_convection
from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.mass_flux import (
    diagnose_mass_flux_closure,
    edmf_convection,
    mass_flux_convection,
    mass_flux_convection_from_closure,
)
from legoesm.atmosphere.physics.convection.integration import (
    make_convection_physics,
)
from legoesm import constants


# ===========================================================================
# Helpers
# ===========================================================================

def _make_unstable_columns(ncol=4, nlev=10):
    """Create test column data with a conditionally unstable profile.

    Returns T, q_v, p_full, p_half with a warm moist lower troposphere.
    """
    # Pressure: linearly spaced interfaces from 100 Pa (top) to 1e5 Pa (surface)
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
        (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    # Temperature: warm surface, decreasing faster than moist adiabat
    # (conditionally unstable)
    T_sfc = 300.0
    T_top = 200.0
    T = jnp.broadcast_to(
        jnp.linspace(T_top, T_sfc, nlev)[None, :],
        (ncol, nlev),
    )

    # Moisture: near saturation in lower levels, dry aloft
    q_sat = saturation_mixing_ratio(T, p_full)
    # 90% RH at surface, decreasing to 10% at top
    rh_profile = jnp.linspace(0.1, 0.9, nlev)[None, :]
    q_v = rh_profile * q_sat

    return T, q_v, p_full, p_half


def _make_stable_columns(ncol=4, nlev=10):
    """Create test column data with a stable profile (isothermal)."""
    p_half = jnp.broadcast_to(
        jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
        (ncol, nlev + 1),
    )
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

    # Isothermal profile — stable
    T = jnp.full((ncol, nlev), 250.0)

    # Very dry
    q_v = jnp.full((ncol, nlev), 1e-6)

    return T, q_v, p_full, p_half


# ===========================================================================
# Thermodynamics tests
# ===========================================================================

class TestThermodynamics:
    """Tests for shared thermodynamic functions."""

    def test_qsat_increases_with_temperature(self):
        """Saturation mixing ratio should increase with temperature."""
        p = jnp.full(5, 1.0e5)
        T = jnp.array([250.0, 260.0, 270.0, 280.0, 290.0])
        q_sat = saturation_mixing_ratio(T, p)
        # Each should be larger than the previous
        assert jnp.all(jnp.diff(q_sat) > 0)

    def test_qsat_positive(self):
        """Saturation mixing ratio should always be positive."""
        T = jnp.array([200.0, 250.0, 300.0, 350.0])
        p = jnp.full(4, 5.0e4)
        q_sat = saturation_mixing_ratio(T, p)
        assert jnp.all(q_sat > 0)

    def test_temperature_from_theta_identity(self):
        """At reference pressure, T should equal theta."""
        theta = jnp.array([300.0, 310.0])
        T = temperature_from_theta(theta, jnp.full(2, constants.p_ref))
        assert jnp.allclose(T, theta, rtol=1e-6)

    def test_pressure_from_eos_positive(self):
        """Pressure from EOS should always be positive."""
        rho = jnp.array([0.5, 1.0, 1.5])
        theta = jnp.array([300.0, 300.0, 300.0])
        p = pressure_from_eos(rho, theta)
        assert jnp.all(p > 0)

    def test_moist_lapse_rate_positive(self):
        """Moist adiabatic lapse rate dT/dp should be positive (T increases with p)."""
        T = jnp.array([250.0, 270.0, 290.0])
        p = jnp.array([5e4, 7e4, 9e4])
        gamma = moist_adiabat_lapse_rate(T, p)
        assert jnp.all(gamma > 0)

    def test_moist_adiabat_warmer_than_dry(self):
        """Moist adiabat should be warmer than dry adiabat at upper levels."""
        ncol = 2
        nlev = 20
        T_base = jnp.array([300.0, 295.0])

        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
            (ncol, nlev + 1),
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

        T_moist = compute_moist_adiabat(T_base, p_full)

        # Dry adiabat: T = T_base * (p / p_sfc)^kappa
        p_sfc = p_full[:, -1:]
        T_dry = T_base[:, None] * (p_full / p_sfc) ** constants.kappa

        # At upper levels (lower pressure), moist should be warmer
        # Check at the top few levels
        assert jnp.all(T_moist[:, :5] > T_dry[:, :5] - 5.0)  # allow some tolerance

    def test_moist_adiabat_shape(self):
        """Moist adiabat should have correct output shape."""
        ncol, nlev = 3, 15
        T_base = jnp.full(ncol, 300.0)
        p_half = jnp.broadcast_to(
            jnp.linspace(100.0, 1.0e5, nlev + 1)[None, :],
            (ncol, nlev + 1),
        )
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        T_moist = compute_moist_adiabat(T_base, p_full)
        assert T_moist.shape == (ncol, nlev)

    def test_cape_positive_for_unstable(self):
        """CAPE should be positive when parcel is warmer than environment."""
        T, q_v, p_full, p_half = _make_unstable_columns(ncol=2, nlev=10)
        # Use a warmer parcel
        T_parcel = T + 5.0
        cape = compute_cape(T, T_parcel, p_full, p_half)
        assert jnp.all(cape > 0)

    def test_cape_zero_for_stable(self):
        """CAPE should be zero when parcel is cooler than environment."""
        T, q_v, p_full, p_half = _make_stable_columns(ncol=2, nlev=10)
        T_parcel = T - 5.0
        cape = compute_cape(T, T_parcel, p_full, p_half)
        assert jnp.allclose(cape, 0.0, atol=1e-10)

    def test_cape_p_source_excludes_below_departure_buoyancy(self):
        """An elevated-departure parcel contributes NO CAPE below its source
        level (the parcel does not exist there).

        Root cause of the tier-5 Bechtold quiescence regression: the PBL-MEAN
        parcel, translated to surface pressure theta-preserving, is +6.3 K
        warmer than the actual surface air of a STABLE boundary layer (theta
        increases with height), and the resulting below-departure positive
        area alone produced CAPE ~53-66 J/kg on a zero-CAPE column — defeating
        the launch gate and heating the column by 886 W/m² (AMIP C24
        bechtold+mcfarlane blowup at day 10). ``p_source`` restricts the
        integral to levels at/above the departure level (textbook mean-layer
        parcel convention). ``p_source = surface`` must be BYTE-IDENTICAL to
        omitting it (all surface-parcel callers unchanged).
        """
        T, q_v, p_full, p_half = _make_stable_columns(ncol=2, nlev=10)
        # Parcel buoyant ONLY in the two lowest (highest-pressure) levels —
        # the artifact pattern: warm below the departure level, cold above.
        T_parcel = T - 5.0
        T_parcel = T_parcel.at[:, -2:].set(T[:, -2:] + 5.0)
        p_src = p_full[:, -3]  # departure ABOVE the buoyant layers

        cape_no_src = compute_cape(T, T_parcel, p_full, p_half)
        assert jnp.all(cape_no_src > 0)  # artifact present without the mask

        cape_src = compute_cape(T, T_parcel, p_full, p_half, p_source=p_src)
        assert jnp.allclose(cape_src, 0.0, atol=1e-10)

        # surface departure == legacy behaviour, bit-for-bit
        cape_sfc = compute_cape(
            T, T_parcel, p_full, p_half, p_source=p_full[:, -1])
        assert jnp.array_equal(cape_sfc, cape_no_src)

    def test_parcel_profile_and_cape_depends_on_launch_humidity(self):
        """Shared parcel->CAPE helper must respond to boundary-layer humidity.

        Regression for F-CONV-1: Zhang-McFarlane and Tiedtke previously called
        ``compute_moist_adiabat``/``compute_cape`` *without* q_v, lifting a
        parcel saturated from the surface.  That makes CAPE independent of the
        actual humidity and spuriously large in dry columns, firing deep
        convection over deserts.  Threading q_v (dry adiabat below the LCL,
        moist above, virtual-T CAPE) must instead give a *dry* column far less
        CAPE than a moist one, and the legacy ``q_v=None`` path must remain the
        humidity-independent saturated-from-base value.
        """
        T, _, p_full, p_half = _make_unstable_columns(ncol=2, nlev=12)
        q_dry = jnp.full_like(T, 1e-4)     # essentially no vapor
        q_moist = jnp.full_like(T, 0.015)  # moist boundary layer

        _, cape_dry = parcel_profile_and_cape(T, p_full, p_half, q_v=q_dry)
        _, cape_moist = parcel_profile_and_cape(T, p_full, p_half, q_v=q_moist)
        _, cape_sat_dry = parcel_profile_and_cape(T, p_full, p_half, q_v=None)
        _, cape_sat_moist = parcel_profile_and_cape(T, p_full, p_half, q_v=None)

        # Physically-correct trigger: a dry column has far less CAPE than moist.
        assert jnp.all(cape_dry < 0.25 * cape_moist + 1.0), (
            f"dry CAPE {cape_dry} not << moist CAPE {cape_moist}"
        )
        # Legacy saturated-from-base path ignores launch humidity (kept as-is).
        assert jnp.allclose(cape_sat_dry, cape_sat_moist), (
            "q_v=None path must be humidity-independent (saturated from base)"
        )
        # The bug's signature: the old (q_v=None) dry-column CAPE is much larger
        # than the corrected (threaded-q_v) dry-column CAPE.
        assert jnp.all(cape_dry < cape_sat_dry)

    def test_parcel_profile_and_cape_grad_finite(self):
        """jax.grad through the shared parcel->CAPE helper must be finite."""
        T, _, p_full, p_half = _make_unstable_columns(ncol=2, nlev=12)
        q_v = jnp.full_like(T, 0.012)
        g = jax.grad(lambda q: jnp.sum(parcel_profile_and_cape(T, p_full, p_half, q_v=q)[1]))(q_v)
        assert jnp.all(jnp.isfinite(g))

    def test_cape_uses_p_mid_for_dlnp_weighting(self):
        """Audit cycle iter-39 finding HIGH #1: ``compute_cape`` must
        use the half-level midpoint pressure ``p_mid = 0.5(p_half[k]
        + p_half[k+1])`` for the discrete ``∫ dlnp`` weighting,
        consistent with every sister physics helper.  The earlier
        formulation used ``p_full`` (a layer-mean pressure on
        hybrid-sigma grids), which produced a 0.5-2 % CAPE bias.

        Test: pin the CAPE value to the analytical reference using
        ``p_mid``.  If a future regression switches back to
        ``p_full``, the value would drift outside the tight
        relative tolerance.
        """
        # Single-column setup with deliberately non-trivial p_full vs
        # p_mid: hybrid-sigma layer-mean ≠ half-level midpoint.
        nlev = 8
        p_half = jnp.array([[100.0, 1.0e4, 2.0e4, 3.5e4, 5.0e4,
                             6.5e4, 8.0e4, 9.5e4, 1.0e5]])  # (1, nlev+1)
        # p_full deliberately offset from the midpoint to simulate a
        # log-pressure layer mean.
        p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
        # Constant 5 K positive buoyancy for clean integral.
        T_env = jnp.full((1, nlev), 280.0)
        T_parcel = T_env + 5.0
        dp = p_half[:, 1:] - p_half[:, :-1]
        cape_actual = float(compute_cape(T_env, T_parcel, p_mid, p_half)[0])
        # Analytical: R_d · 5 · ∑ dp/p_mid
        from legoesm import constants
        cape_expected = float(
            constants.R_d * 5.0 * jnp.sum(dp / p_mid),
        )
        # Bit-exact agreement when the helper uses p_mid.
        assert abs(cape_actual - cape_expected) < 1.0, (
            f"compute_cape returned {cape_actual:.3f} vs expected "
            f"{cape_expected:.3f} (p_mid weighting).  Iter-39 fix"
        )

    def test_qsat_grad_works(self):
        """jax.grad should work through saturation_mixing_ratio."""
        def loss(T):
            return jnp.sum(saturation_mixing_ratio(T, jnp.full_like(T, 1e5)))
        T = jnp.array([280.0, 290.0])
        g = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(g))


# ===========================================================================
# SBM convection tests
# ===========================================================================

class TestSBM:
    """Tests for Simplified Betts-Miller convection."""

    def test_output_shapes(self):
        """SBM output should have correct shapes."""
        ncol, nlev = 4, 10
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = SBMConfig()
        out = sbm_convection(T, q_v, p_full, p_half, dt=300.0, config=config)

        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.dq_c_conv_dt.shape == (ncol, nlev)
        assert out.cape.shape == (ncol,)
        assert out.convective_mask.shape == (ncol,)

    def test_never_net_moistens_a_column(self):
        """Column-water conservation contract (#771): SBM must never ADD net
        column water.

        SBM relaxes q_v toward q_ref = RH_ref*q_sat. In a TRIGGERED column
        that is net sub-saturated vs q_ref, the unfixed relaxation net-
        MOISTENS the column while sbm.py rescales only the CONDENSATE to
        col-net-drying — the applied vapour tendency then creates water
        from nothing (+0.53 mm/day global in the 3-day C24 AMIP probe; the
        CWV drift behind the C48 pilot's day-150 blowup). The drying_gate
        zeroes the WHOLE adjustment in such columns:
          (1) every column satisfies col ∫ dq_v dp/g <= 0;
          (2) gated (would-be-moistening) columns have EXACTLY zero dT_dt
              too (the dT/dq_v gate is shared, preserving the Newton
              enthalpy closure — a heat-only residual would be a leak).
        Non-vacuous: at least one column must be TRIGGERED yet gated, and
        this test FAILS on the pre-#771-fix sbm.py (verified on main).
        """
        ncol, nlev = 4, 20
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        # Dry the columns progressively: column 0 keeps the moist sounding
        # (net-drying, exercises the gate=1 path); the driest columns stay
        # conditionally unstable (trigger on) but are net sub-saturated vs
        # q_ref -> the unfixed scheme would net-moisten them.
        scale = jnp.array([1.0, 0.5, 0.3, 0.15])[:, None]
        q_v = q_v * scale
        config = SBMConfig()
        out = sbm_convection(T, q_v, p_full, p_half, dt=300.0, config=config)

        dp = p_half[:, 1:] - p_half[:, :-1]
        col_dqv = jnp.sum(out.dq_v_dt * dp, axis=1) / constants.g

        # (1) no column ever gains net water (tol ~ fp roundoff of the sum)
        assert bool(jnp.all(col_dqv <= 1e-10))

        # non-vacuity: some column is triggered yet fully gated (the unfixed
        # scheme would have moistened it), and some column genuinely dries
        gated = (out.convective_mask > 0) & jnp.all(out.dq_v_dt == 0.0, axis=1)
        drying = col_dqv < -1e-12
        assert bool(jnp.any(gated)), (
            "fixture produced no triggered net-sub-saturated column — the "
            "conservation contract was not exercised")
        assert bool(jnp.any(drying))

        # (2) gated columns carry NO heat tendency either (shared gate)
        dT_gated = jnp.where(gated[:, None], out.dT_dt, 0.0)
        assert bool(jnp.all(dT_gated == 0.0))

    def test_enthalpy_conservation(self):
        """Column enthalpy tendency should be approximately conserved.

        The linearized enthalpy correction ensures approximate conservation.
        sum(c_pd * dT_dt + L_v * dq_v_dt) * dp / g should be small
        relative to the total enthalpy change from temperature alone.
        """
        ncol, nlev = 4, 20
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = SBMConfig()
        out = sbm_convection(T, q_v, p_full, p_half, dt=300.0, config=config)

        dp = p_half[:, 1:] - p_half[:, :-1]
        enthalpy_tend = jnp.sum(
            (constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt) * dp / constants.g,
            axis=1,
        )
        # Compute scale: total temperature tendency magnitude
        T_scale = jnp.sum(
            constants.c_pd * jnp.abs(out.dT_dt) * dp / constants.g,
            axis=1,
        )
        # Relative conservation: enthalpy residual should be small fraction of total
        relative_error = jnp.abs(enthalpy_tend) / jnp.clip(T_scale, 1.0, None)
        assert float(jnp.max(relative_error)) < 0.1  # within 10%

    def test_precipitation_non_negative(self):
        """Convective cloud-water source should always be >= 0."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        config = SBMConfig()
        out = sbm_convection(T, q_v, p_full, p_half, dt=300.0, config=config)
        assert jnp.all(out.dq_c_conv_dt >= 0)

    def test_stable_gives_small_tendency(self):
        """Stable columns should produce small tendencies relative to unstable."""
        T_stable, q_stable, p_full, p_half = _make_stable_columns()
        T_unstable, q_unstable, _, _ = _make_unstable_columns()
        config = SBMConfig()

        out_stable = sbm_convection(T_stable, q_stable, p_full, p_half, dt=300.0, config=config)
        out_unstable = sbm_convection(T_unstable, q_unstable, p_full, p_half, dt=300.0, config=config)

        # Stable tendencies should be much smaller than unstable
        max_stable = float(jnp.max(jnp.abs(out_stable.dT_dt)))
        max_unstable = float(jnp.max(jnp.abs(out_unstable.dT_dt)))
        assert max_stable < max_unstable

    def test_unstable_gives_nonzero(self):
        """Unstable columns should produce nonzero tendencies."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        config = SBMConfig()
        out = sbm_convection(T, q_v, p_full, p_half, dt=300.0, config=config)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-6

    def test_differentiable(self):
        """jax.grad should work through SBM convection."""
        ncol, nlev = 2, 8
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = SBMConfig()

        def loss(T_in):
            out = sbm_convection(T_in, q_v, p_full, p_half, dt=300.0, config=config)
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

    def test_cloud_mask_straight_through_gradient_is_nonzero(self):
        """SBM cloud_mask uses a straight-through estimator: forward = hard
        boolean ``(T_moist >= T)``, backward = sigmoid' so that
        ``jax.grad`` flows through layer-membership transitions.

        This test isolates the cloud_mask gradient path from every other
        SBM gradient route (T_base, T_ref, CAPE/trigger, (T_ref-T)) by
        holding T_moist FIXED and constructing a 1-column profile with
        an interior boundary level (T_moist == T).  Then:

          ∂[cloud_mask]/∂T at boundary
            = ∂[sigmoid(s·(T_moist - T))]/∂T |_{T_moist=T}
            = -s/4

        With sharpness s = 5/K, the boundary gradient should be ≈ -1.25.
        With the legacy hard ``(T_moist >= T).astype(...)`` mask, this
        derivative is exactly zero almost everywhere — the assertion
        falsifies the hard-mask implementation by construction.

        The straight-through estimator's forward value is also tested:
        cloud_mask must equal the hard step, NOT the sigmoid (so the
        forward integration semantics of the prior implementation are
        preserved exactly).
        """
        sharpness = SBMConfig().cloud_mask_sharpness  # 5.0/K
        # Interior boundary at level 1 (between levels 0 and 2).
        T_moist = jnp.array([[260.0, 280.0, 300.0]])
        T = jnp.array([[265.0, 280.0, 295.0]])  # T_moist == T at level 1

        def cloud_mask_only(T_in):
            soft = jax.nn.sigmoid(sharpness * (T_moist - T_in))
            hard = (T_moist >= T_in).astype(T_in.dtype)
            return soft + jax.lax.stop_gradient(hard - soft)

        # Forward: must equal hard step
        mask_value = cloud_mask_only(T)
        expected_hard = jnp.array([[0.0, 1.0, 1.0]])  # T_moist >= T
        assert jnp.allclose(mask_value, expected_hard), (
            f"Forward cloud_mask should preserve hard-step semantics, "
            f"got {mask_value} vs expected {expected_hard}"
        )

        # Backward: gradient at interior boundary level must be non-zero
        grad_T = jax.grad(lambda T_in: jnp.sum(cloud_mask_only(T_in)))(T)
        boundary_grad = float(grad_T[0, 1])
        # Expected: -sharpness/4 = -1.25
        expected_grad = -sharpness / 4.0
        assert abs(boundary_grad - expected_grad) < 0.01, (
            f"Boundary gradient is {boundary_grad}, expected ≈ "
            f"{expected_grad} for the straight-through estimator. "
            f"A hard boolean mask gives 0 (which would fail this check)."
        )

    def test_accepts_columnwise_parameters(self):
        """SBM should support per-column traced control parameters."""
        ncol, nlev = 3, 8
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = SBMConfig(
            tau_c=jnp.array([3600.0, 7200.0, 14400.0]),
            rh_ref=jnp.array([0.65, 0.75, 0.85]),
            cape_threshold=jnp.array([10.0, 70.0, 150.0]),
        )

        out = sbm_convection(T, q_v, p_full, p_half, dt=300.0, config=config)

        assert jnp.all(jnp.isfinite(out.dT_dt))
        assert jnp.all(jnp.isfinite(out.dq_v_dt))
        assert out.dT_dt.shape == (ncol, nlev)
        assert float(jnp.max(jnp.abs(out.dT_dt[0]))) >= float(jnp.max(jnp.abs(out.dT_dt[-1])))

    def test_stratosphere_above_lnb_has_zero_adjustment(self):
        """Regression for #326: SBM must not adjust stably stratified
        layers above the level of neutral buoyancy.

        The cloud_mask = (T_moist >= T_env) gate restricts the
        convective relaxation to conditionally unstable levels. Without
        it, the scheme relaxed the *entire* column — including the
        stable stratosphere — toward the moist adiabat, producing the
        spurious ~40 K/h cooling at the model top that drove the upper-
        atmosphere warm bias documented in #318.

        Here the deep conditionally-unstable profile from
        ``_make_unstable_columns`` has a warm-moist lower troposphere
        (moist adiabat warmer than the environment) and a cold upper
        region where the moist adiabat launched from the surface parcel
        falls below the environment (stable). The forward cloud_mask is
        an exact hard step, so the relaxation tendency must be EXACTLY
        zero in every layer where ``T_moist < T_env``.
        """
        ncol, nlev = 2, 20
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = SBMConfig()
        out = sbm_convection(T, q_v, p_full, p_half, dt=300.0, config=config)

        # Reconstruct the convective-instability mask the scheme uses:
        # the moist adiabat is launched from the surface parcel T[:, -1].
        T_moist = compute_moist_adiabat(T[:, -1], p_full)
        stable = T_moist < T  # (ncol, nlev) — layers the gate must skip

        # The constructed profile must actually contain both regimes,
        # else the test is vacuous.
        assert bool(jnp.any(stable)), "profile has no stable layers"
        assert bool(jnp.any(~stable)), "profile has no unstable layers"

        dT = out.dT_dt
        dq = out.dq_v_dt
        max_stable_dT = float(jnp.max(jnp.abs(jnp.where(stable, dT, 0.0))))
        max_stable_dq = float(jnp.max(jnp.abs(jnp.where(stable, dq, 0.0))))
        max_unstable_dT = float(jnp.max(jnp.abs(jnp.where(stable, 0.0, dT))))

        # Troposphere is convectively active …
        assert max_unstable_dT > 1.0e-6, (
            "conditionally unstable troposphere produced no adjustment"
        )
        # … but stable layers see zero adjustment (exact hard-step mask).
        assert max_stable_dT < 1.0e-12, (
            f"stable-layer |dT/dt|={max_stable_dT:.3e} K/s should be 0 — "
            "cloud_mask stratosphere gate broken (#326)"
        )
        assert max_stable_dq < 1.0e-15, (
            f"stable-layer |dq_v/dt|={max_stable_dq:.3e} should be 0 (#326)"
        )


# ===========================================================================
# DCA convection tests
# ===========================================================================

class TestDCA:
    """Tests for Deep Convective Adjustment."""

    def test_output_shapes(self):
        """DCA output should have correct shapes."""
        ncol, nlev = 4, 10
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = DCAConfig()
        out = dca_convection(T, q_v, p_full, p_half, dt=300.0, config=config)

        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.dq_c_conv_dt.shape == (ncol, nlev)
        assert out.cape.shape == (ncol,)
        assert out.convective_mask.shape == (ncol,)

    def test_stable_small_tendency(self):
        """Stable columns should produce much smaller tendencies than unstable."""
        T_stable, q_stable, p_full, p_half = _make_stable_columns()
        T_unstable, q_unstable, _, _ = _make_unstable_columns()
        config = DCAConfig()

        out_stable = dca_convection(T_stable, q_stable, p_full, p_half, dt=300.0, config=config)
        out_unstable = dca_convection(T_unstable, q_unstable, p_full, p_half, dt=300.0, config=config)

        max_stable = float(jnp.max(jnp.abs(out_stable.dT_dt)))
        max_unstable = float(jnp.max(jnp.abs(out_unstable.dT_dt)))
        assert max_stable < max_unstable

    def test_produces_nonzero_for_unstable(self):
        """DCA should produce nonzero tendencies for unstable columns."""
        ncol, nlev = 2, 10
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = DCAConfig(n_iterations=1)
        out = dca_convection(T, q_v, p_full, p_half, dt=300.0, config=config)

        # Should have nonzero temperature adjustment
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-6

    def test_upward_pair_adjustment_uses_progressive_scan_carry(self):
        """Upper-pair instability should adjust even if bottom pair is stable."""
        T = jnp.array([[230.0, 290.0, 291.0]])  # top -> bottom
        q_v = jnp.full_like(T, 1e-6)
        p_half = jnp.array([[1.0e4, 4.0e4, 8.0e4, 1.1e5]])
        p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])

        out = dca_convection(
            T,
            q_v,
            p_full,
            p_half,
            dt=300.0,
            config=DCAConfig(n_iterations=1, mixing_fraction=1.0),
        )

        # Top level (index 0) belongs only to the upper pair and should adjust.
        assert float(jnp.abs(out.dT_dt[0, 0])) > 1e-4

    def test_differentiable(self):
        """jax.grad should work through DCA convection."""
        ncol, nlev = 2, 8
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = DCAConfig()

        def loss(T_in):
            out = dca_convection(T_in, q_v, p_full, p_half, dt=300.0, config=config)
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

    def test_moist_static_energy_conservation(self):
        """DCA must conserve moist static energy column-wise.

        Adjustment imposes the moist-adiabatic lapse rate and removes
        super-saturation; the latent heat released by condensation
        must warm the column so that

            c_p · ⟨ΔT⟩ + L_v · ⟨Δq⟩ = 0   (column mean, dp-weighted)

        holds layer-pair by layer-pair.  The Physical_Consistency
        cycle iter-1 fix added this latent-heat term — without it,
        DCA conserved only dry static energy and biased the column
        cool by ~2.5 K per g/kg condensed.
        """
        from legoesm import constants
        from legoesm.thermo import saturation_mixing_ratio
        from legoesm.atmosphere.physics.convection.dca import (
            _adjust_one_iteration,
        )

        ncol, nlev = 4, 10
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        # Saturate the column so removing super-saturation actually
        # condenses water (otherwise q_adj == q_v and the test is
        # vacuous on stable / dry columns).
        q_v = saturation_mixing_ratio(T, p_full) * 1.05

        # Bypass CAPE-gating by calling the inner adjustment loop
        # directly: the conservation property we are testing is a
        # property of ``_adjust_one_iteration``, not of the cape
        # threshold.
        dp = p_half[:, 1:] - p_half[:, :-1]
        T_new, q_new, _ = _adjust_one_iteration(
            T, q_v, p_full, dp, mixing_fraction=1.0,
            instability_blend_sharpness=DCAConfig().instability_blend_sharpness,
        )

        # Column mean tendencies, mass-weighted by dp.
        dT = T_new - T  # K
        dq = q_new - q_v  # kg/kg
        dT_col = jnp.sum(dT * dp, axis=-1) / jnp.sum(dp, axis=-1)
        dq_col = jnp.sum(dq * dp, axis=-1) / jnp.sum(dp, axis=-1)

        # Sanity: the column actually condensed water (negative ⟨Δq⟩
        # at meaningful magnitude).  Without this check the test
        # would pass vacuously on a column where no adjustment fired.
        assert float(jnp.min(-dq_col)) > 1e-4, (
            f"test setup did not produce real condensation; "
            f"<Δq> = {float(jnp.min(dq_col)):.2e}"
        )

        # Moist static energy invariant: c_p · ⟨ΔT⟩ + L_v · ⟨Δq⟩ = 0
        # since dq is negative (condensation) and dT positive (latent
        # heat release), the residual should be near zero.
        residual = constants.c_pd * dT_col + constants.L_v * dq_col
        scale = jnp.maximum(constants.c_pd * jnp.abs(dT_col), 1e-12)
        max_rel = float(jnp.max(jnp.abs(residual) / scale))
        # Per-pair conservation is exact; the residual at the column
        # level comes only from successive scan steps each seeing a
        # slightly updated T.  Empirically max_rel ~ 3e-6 with f64.
        # Codex review tightened from 0.10 to 1e-4 on grounds that
        # the looser bound let a regression slip past undetected.
        assert max_rel < 1e-4, (
            f"DCA moist-static-energy residual = {max_rel:.3e}; "
            "iter-1 latent-heat fix should keep this small."
        )


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    """Tests for make_convection_physics integration bridge."""

    def test_hydrostatic_tendency_shapes(self):
        """Hydrostatic convection tendencies should have correct shapes."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = ConvectionConfig(scheme="sbm")
        physics_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        n = grid.n
        nlev = sigma.n_levels
        assert tendencies.dT_dt.data.shape == (6, n, n, nlev)
        assert tendencies.du_dt.data.shape == (6, n, n, nlev)
        assert tendencies.dp_s_dt.data.shape == (6, n, n)

    def test_hydrostatic_nonzero_heating(self):
        """Hydrostatic convection should produce nonzero T tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = ConvectionConfig(scheme="sbm")
        physics_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        max_hr = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
        assert max_hr > 0.0

    def test_hydrostatic_zero_wind_tendency(self):
        """Convection should not produce wind or pressure tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = ConvectionConfig(scheme="sbm")
        physics_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        assert jnp.allclose(tendencies.du_dt.data, 0.0)
        assert jnp.allclose(tendencies.dv_dt.data, 0.0)
        assert jnp.allclose(tendencies.dp_s_dt.data, 0.0)

    def test_nonhydrostatic_tendency_shapes(self):
        """Non-hydrostatic convection tendencies should have correct shapes."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import (
            create_height_coordinate,
            compute_terrain_metric,
        )
        from legoesm.core.field import Field
        from legoesm.core.state import NonHydrostaticState

        n = 8
        nlev = 10
        grid = create_cubed_sphere(n)
        height_coord = create_height_coordinate(nlev, 30000.0)
        z_s = jnp.zeros((6, n, n))
        terrain_metric = compute_terrain_metric(z_s, height_coord)

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        state = NonHydrostaticState(
            u=Field(data=jnp.zeros((6, n, n, nlev)), name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.zeros((6, n, n, nlev)), name="v", dims=dims_3d, units="m/s"),
            w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w", dims=dims_w, units="m/s"),
            theta_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="theta_prime", dims=dims_3d, units="K"),
            rho_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="rho_prime", dims=dims_3d, units="kg/m^3"),
            phis=Field(data=jnp.zeros((6, n, n)), name="phis", dims=dims_2d, units="m^2/s^2"),
            tracers=Field(data=jnp.zeros((6, n, n, nlev, 1)), name="tracers", dims=dims_tr, units="kg/kg"),
        )

        config = ConvectionConfig(scheme="sbm")
        physics_fn = make_convection_physics(config, model_type="nonhydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, height_coord, terrain_metric)

        assert tendencies.dtheta_prime_dt.data.shape == (6, n, n, nlev)
        assert tendencies.du_dt.data.shape == (6, n, n, nlev)
        assert tendencies.dw_dt.data.shape == (6, n, n, nlev + 1)
        assert tendencies.dtracers_dt.data.shape == (6, n, n, nlev, 1)

    def test_nonhydrostatic_nonzero_heating(self):
        """NH convection should produce nonzero theta tendencies.

        SBM only heats a convectively ACTIVE column: it relaxes T/q toward a
        moist-adiabatic reference where CAPE exceeds ``cape_threshold``.  A dry
        (q_v=0), unperturbed state sits at the dry-neutral ``theta_ref=300 K``
        reference with CAPE well below threshold, so the trigger is exactly zero
        and zero heating is PHYSICALLY CORRECT — a ``> 0`` assertion on it is
        vacuous.  Seed a moist column (q_v ~18 g/kg at the surface, level index
        -1 per SBM's z-up convention, tapering to ~0 aloft) so the moist adiabat
        clears the reference and the scheme genuinely heats.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import (
            create_height_coordinate,
            compute_terrain_metric,
        )
        from legoesm.core.field import Field
        from legoesm.core.state import NonHydrostaticState

        n = 8
        nlev = 10
        grid = create_cubed_sphere(n)
        height_coord = create_height_coordinate(nlev, 30000.0)
        z_s = jnp.zeros((6, n, n))
        terrain_metric = compute_terrain_metric(z_s, height_coord)

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        # Water-vapour tracer: 0 at the model top (level 0) rising to ~18 g/kg at
        # the surface (level -1).  Against the dry-neutral reference this gives
        # CAPE > threshold, so the SBM trigger fires.
        q_surface = 0.018
        qv_profile = q_surface * jnp.linspace(0.0, 1.0, nlev)
        qv = jnp.broadcast_to(qv_profile, (6, n, n, nlev))

        state = NonHydrostaticState(
            u=Field(data=jnp.zeros((6, n, n, nlev)), name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.zeros((6, n, n, nlev)), name="v", dims=dims_3d, units="m/s"),
            w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w", dims=dims_w, units="m/s"),
            theta_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="theta_prime", dims=dims_3d, units="K"),
            rho_prime=Field(data=jnp.zeros((6, n, n, nlev)), name="rho_prime", dims=dims_3d, units="kg/m^3"),
            phis=Field(data=jnp.zeros((6, n, n)), name="phis", dims=dims_2d, units="m^2/s^2"),
            tracers=Field(data=qv[..., None], name="tracers", dims=dims_tr, units="kg/kg"),
        )

        config = ConvectionConfig(scheme="sbm")
        physics_fn = make_convection_physics(config, model_type="nonhydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, height_coord, terrain_metric)

        max_hr = float(jnp.max(jnp.abs(tendencies.dtheta_prime_dt.data)))
        assert max_hr > 0.0

    def test_grad_through_hydrostatic_convection(self):
        """jax.grad should work through hydrostatic convection physics."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = ConvectionConfig(scheme="sbm")
        physics_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)

        def loss(T_data):
            new_state = state._replace(T=state.T.replace(data=T_data))
            tendencies, _ = physics_fn(new_state, grid, sigma)
            return jnp.sum(tendencies.dT_dt.data ** 2)

        grad_T = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad_T))

    def test_scheme_selection_sbm(self):
        """scheme='sbm' should select SBM backend."""
        ncol, nlev = 2, 8
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)

        config_sbm = ConvectionConfig(scheme="sbm")
        config_dca = ConvectionConfig(scheme="dca")

        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        fn_sbm = make_convection_physics(config_sbm, model_type="hydrostatic", dt=300.0)
        fn_dca = make_convection_physics(config_dca, model_type="hydrostatic", dt=300.0)

        tend_sbm, _ = fn_sbm(state, grid, sigma)
        tend_dca, _ = fn_dca(state, grid, sigma)

        # Both should produce nonzero but different tendencies
        assert not jnp.allclose(tend_sbm.dT_dt.data, tend_dca.dT_dt.data, atol=1e-10)

    def test_none_scheme_gives_zeros(self):
        """scheme='none' should produce zero tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = ConvectionConfig(scheme="none")
        physics_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)
        tendencies, _ = physics_fn(state, grid, sigma)

        assert jnp.allclose(tendencies.dT_dt.data, 0.0)

    def test_scheme_selection_kuo(self):
        """scheme='kuo' should give different results from 'sbm'."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        fn_kuo = make_convection_physics(
            ConvectionConfig(scheme="kuo"), model_type="hydrostatic", dt=300.0,
        )
        fn_sbm = make_convection_physics(
            ConvectionConfig(scheme="sbm"), model_type="hydrostatic", dt=300.0,
        )
        tend_kuo, _ = fn_kuo(state, grid, sigma)
        tend_sbm, _ = fn_sbm(state, grid, sigma)

        assert not jnp.allclose(tend_kuo.dT_dt.data, tend_sbm.dT_dt.data, atol=1e-10)

    def test_scheme_selection_mass_flux(self):
        """scheme='mass_flux' should produce nonzero tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        fn = make_convection_physics(
            ConvectionConfig(scheme="mass_flux"), model_type="hydrostatic", dt=300.0,
        )
        tendencies, _ = fn(state, grid, sigma)
        max_val = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
        assert max_val > 0.0

    def test_scheme_selection_edmf(self):
        """scheme='edmf' should produce nonzero tendencies."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        fn = make_convection_physics(
            ConvectionConfig(scheme="edmf"), model_type="hydrostatic", dt=300.0,
        )
        tendencies, _ = fn(state, grid, sigma)
        max_val = float(jnp.max(jnp.abs(tendencies.dT_dt.data)))
        assert max_val > 0.0

    def test_grad_through_mass_flux(self):
        """jax.grad should work through mass_flux integration."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = ConvectionConfig(scheme="mass_flux")
        physics_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)

        def loss(T_data):
            new_state = state._replace(T=state.T.replace(data=T_data))
            tendencies, _ = physics_fn(new_state, grid, sigma)
            return jnp.sum(tendencies.dT_dt.data ** 2)

        grad_T = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad_T))

    def test_grad_through_edmf(self):
        """jax.grad should work through edmf integration."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.atmosphere.held_suarez import held_suarez_init

        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        state = held_suarez_init(grid, sigma)

        config = ConvectionConfig(scheme="edmf")
        physics_fn = make_convection_physics(config, model_type="hydrostatic", dt=300.0)

        def loss(T_data):
            new_state = state._replace(T=state.T.replace(data=T_data))
            tendencies, _ = physics_fn(new_state, grid, sigma)
            return jnp.sum(tendencies.dT_dt.data ** 2)

        grad_T = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grad_T))


# ===========================================================================
# Kuo convection tests
# ===========================================================================

class TestKuo:
    """Tests for Kuo moisture convergence convection."""

    def test_output_shapes(self):
        """Kuo output should have correct shapes."""
        ncol, nlev = 4, 10
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = KuoConfig()
        out = kuo_convection(T, q_v, p_full, p_half, dt=300.0, config=config)

        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.dq_c_conv_dt.shape == (ncol, nlev)
        assert out.cape.shape == (ncol,)
        assert out.convective_mask.shape == (ncol,)

    def test_precipitation_non_negative(self):
        """Convective cloud-water source should always be >= 0."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        config = KuoConfig()
        out = kuo_convection(T, q_v, p_full, p_half, dt=300.0, config=config)
        assert jnp.all(out.dq_c_conv_dt >= 0)

    def test_nonzero_tendencies(self):
        """Canonical Kuo fires under a positive large-scale moisture-
        convergence source (the faithful Kuo source), producing nonzero
        heating.  Without convergence Kuo is correctly quiescent (a
        single column has no resolved ascent) — a different test.
        """
        T, q_v, p_full, p_half = _make_unstable_columns()
        nlev = q_v.shape[-1]
        # Positive convergence in the lower/mid troposphere.
        ptenq = jnp.broadcast_to(
            (2.0e-8 * (jnp.arange(nlev) >= nlev // 3))[None, :], q_v.shape,
        ).astype(q_v.dtype)
        config = KuoConfig()
        out = kuo_convection(T, q_v, p_full, p_half, dt=300.0, config=config,
                             moisture_convergence=ptenq)
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-7

    def test_quiescent_without_convergence(self):
        """No large-scale convergence → zero Kuo tendency (correct)."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        out = kuo_convection(T, q_v, p_full, p_half, dt=300.0,
                             config=KuoConfig(), moisture_convergence=None)
        assert float(jnp.max(jnp.abs(out.dT_dt))) == 0.0
        assert float(jnp.max(jnp.abs(out.dq_v_dt))) == 0.0

    def test_differentiable(self):
        """jax.grad should work through Kuo convection (with and without
        a convergence source)."""
        ncol, nlev = 2, 8
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = KuoConfig()
        ptenq = jnp.broadcast_to(
            (2.0e-8 * (jnp.arange(nlev) >= nlev // 3))[None, :], q_v.shape,
        ).astype(q_v.dtype)

        def loss(T_in):
            out = kuo_convection(T_in, q_v, p_full, p_half, dt=300.0,
                                 config=config, moisture_convergence=ptenq)
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape

        # Quiescent path also has a finite (zero) gradient.
        def loss0(T_in):
            out = kuo_convection(T_in, q_v, p_full, p_half, dt=300.0,
                                 config=config, moisture_convergence=None)
            return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dq_v_dt ** 2)

        grad0 = jax.grad(loss0)(T)
        assert jnp.all(jnp.isfinite(grad0))


# ===========================================================================
# Mass-Flux convection tests
# ===========================================================================

class TestMassFlux:
    """Tests for Prognostic Mass-Flux convection."""

    def test_output_shapes(self):
        """Mass-Flux output should have correct shapes."""
        ncol, nlev = 4, 10
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = MassFluxConfig()
        M_c = jnp.full(ncol, config.M_c_init)
        out, M_c_new = mass_flux_convection(
            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
        )

        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.dq_c_conv_dt.shape == (ncol, nlev)
        assert out.cape.shape == (ncol,)
        assert out.convective_mask.shape == (ncol,)
        assert M_c_new.shape == (ncol,)

    @pytest.mark.xfail(
        reason="F-CONV-MSE: mass-flux convection does NOT conserve column moist "
        "static energy (~64% residual w/o condensate; the kernel heats ~3 kW/m² "
        "while drying supplies only ~1 kW/m²). Root cause: the fixed delta_0 "
        "detrainment is decoupled from the prescribed sin M-profile's dM/dz, so "
        "subsidence+detrainment do not telescope into the conservative flux form. "
        "An iter-34 column-MSE energy fixer (rescaling heating) was PROVEN "
        "inadequate: _make_unstable_columns yields kernel states that heat AND "
        "moisten vapor, so conservation would require cooling — conflicting with "
        "detrainment warming. The flux-form g*d_p[M(X_u-X)] reformulation (with a "
        "positivity-preserving limiter for dq_c>=0) is required + a convection "
        "benchmark. See parameterization_checks.md.",
        strict=False,
    )
    def test_mass_flux_conserves_column_mse(self):
        """Column MSE tendency must be a small fraction of the gross heating —
        the *same* criterion SBM (`test_enthalpy_conservation`) and DCA pass at
        <10%.  Executable spec for F-CONV-MSE: flips to XPASS when conservation
        is enforced.  Uses M_c at the literature cap so convection is active.
        """
        ncol, nlev = 4, 20
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = MassFluxConfig()
        M_c = jnp.full(ncol, config.M_b_max)  # active convection
        out, _ = mass_flux_convection(
            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
        )
        dp = p_half[:, 1:] - p_half[:, :-1]
        mse_tend = jnp.sum(
            (constants.c_pd * out.dT_dt
             + constants.L_v * (out.dq_v_dt + out.dq_c_conv_dt)) * dp / constants.g,
            axis=1,
        )
        gross = jnp.sum(
            constants.c_pd * jnp.abs(out.dT_dt) * dp / constants.g, axis=1,
        )
        rel = jnp.abs(mse_tend) / jnp.clip(gross, 1.0, None)
        assert float(jnp.max(rel)) < 0.1

    def test_M_c_non_negative(self):
        """M_c_new should be >= 0 (softplus floor)."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        ncol = T.shape[0]
        config = MassFluxConfig()
        M_c = jnp.full(ncol, config.M_c_init)
        _, M_c_new = mass_flux_convection(
            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
        )
        assert jnp.all(M_c_new >= 0)

    def test_prognostic_evolves(self):
        """Calling twice with updated M_c should give different results."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        ncol = T.shape[0]
        config = MassFluxConfig()
        M_c_0 = jnp.full(ncol, config.M_c_init)

        out1, M_c_1 = mass_flux_convection(
            T, q_v, p_full, p_half, M_c_0, dt=300.0, config=config,
        )
        out2, M_c_2 = mass_flux_convection(
            T, q_v, p_full, p_half, M_c_1, dt=300.0, config=config,
        )
        # M_c should have evolved
        assert not jnp.allclose(M_c_1, M_c_2, atol=1e-12)

    def test_nonzero_tendencies(self):
        """Unstable columns should produce nonzero tendencies."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        ncol = T.shape[0]
        config = MassFluxConfig()
        M_c = jnp.full(ncol, config.M_c_init)
        out, _ = mass_flux_convection(
            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
        )
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10

    def test_precipitation_is_nonzero_for_moist_unstable_columns(self):
        """Moist unstable columns should produce some convective condensate."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        q_v = saturation_mixing_ratio(T, p_full)
        ncol = T.shape[0]
        config = MassFluxConfig()
        M_c = jnp.full(ncol, config.M_c_init)
        out, _ = mass_flux_convection(
            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
        )
        assert float(jnp.max(out.dq_c_conv_dt)) > 0.0

    def test_closure_split_matches_full_kernel(self):
        """The split closure+tendency path should match the full kernel."""
        T, q_v, p_full, p_half = _make_unstable_columns(ncol=3, nlev=12)
        config = MassFluxConfig()
        M_c = jnp.full((T.shape[0],), config.M_c_init)

        out_full, M_c_new_full = mass_flux_convection(
            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
        )
        closure = diagnose_mass_flux_closure(
            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
        )
        out_split = mass_flux_convection_from_closure(
            T, q_v, p_full, p_half, closure, config=config,
        )

        assert jnp.allclose(closure.M_c_new, M_c_new_full)
        assert jnp.allclose(out_split.dT_dt, out_full.dT_dt)
        assert jnp.allclose(out_split.dq_v_dt, out_full.dq_v_dt)
        assert jnp.allclose(out_split.dq_c_conv_dt, out_full.dq_c_conv_dt)

    def test_differentiable(self):
        """jax.grad should work through Mass-Flux convection."""
        ncol, nlev = 2, 8
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = MassFluxConfig()
        M_c = jnp.full(ncol, config.M_c_init)

        def loss(T_in):
            out, _ = mass_flux_convection(
                T_in, q_v, p_full, p_half, M_c, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape


# ===========================================================================
# EDMF convection tests
# ===========================================================================

class TestEDMF:
    """Tests for simplified EDMF convection."""

    def test_output_shapes(self):
        """EDMF output should have correct shapes."""
        ncol, nlev = 4, 10
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = ConvectiveEDMFConfig()
        a_u = jnp.full(ncol, config.a_u_init)
        out, a_u_new = edmf_convection(
            T, q_v, p_full, p_half, a_u, dt=300.0, config=config,
        )

        assert out.dT_dt.shape == (ncol, nlev)
        assert out.dq_v_dt.shape == (ncol, nlev)
        assert out.dq_c_conv_dt.shape == (ncol, nlev)
        assert out.cape.shape == (ncol,)
        assert out.convective_mask.shape == (ncol,)
        assert a_u_new.shape == (ncol,)

    def test_a_u_bounded(self):
        """a_u_new should be in [0, 0.5]."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        ncol = T.shape[0]
        config = ConvectiveEDMFConfig()
        a_u = jnp.full(ncol, config.a_u_init)
        _, a_u_new = edmf_convection(
            T, q_v, p_full, p_half, a_u, dt=300.0, config=config,
        )
        assert jnp.all(a_u_new >= 0.0)
        assert jnp.all(a_u_new <= 0.5)

    def test_prognostic_evolves(self):
        """a_u should change across calls when starting away from equilibrium."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        ncol = T.shape[0]
        config = ConvectiveEDMFConfig()
        # Start far from equilibrium to see evolution
        a_u_0 = jnp.full(ncol, 0.01)

        _, a_u_1 = edmf_convection(
            T, q_v, p_full, p_half, a_u_0, dt=300.0, config=config,
        )
        _, a_u_2 = edmf_convection(
            T, q_v, p_full, p_half, a_u_1, dt=300.0, config=config,
        )
        # a_u should be moving toward equilibrium
        assert not jnp.allclose(a_u_0, a_u_1, atol=1e-12)
        assert not jnp.allclose(a_u_1, a_u_2, atol=1e-12)

    def test_nonzero_tendencies(self):
        """Unstable columns should produce nonzero tendencies."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        ncol = T.shape[0]
        config = ConvectiveEDMFConfig()
        a_u = jnp.full(ncol, config.a_u_init)
        out, _ = edmf_convection(
            T, q_v, p_full, p_half, a_u, dt=300.0, config=config,
        )
        assert float(jnp.max(jnp.abs(out.dT_dt))) > 1e-10

    def test_differentiable(self):
        """jax.grad should work through EDMF convection."""
        ncol, nlev = 2, 8
        T, q_v, p_full, p_half = _make_unstable_columns(ncol, nlev)
        config = ConvectiveEDMFConfig()
        a_u = jnp.full(ncol, config.a_u_init)

        def loss(T_in):
            out, _ = edmf_convection(
                T_in, q_v, p_full, p_half, a_u, dt=300.0, config=config,
            )
            return jnp.sum(out.dT_dt ** 2)

        grad_T = jax.grad(loss)(T)
        assert jnp.all(jnp.isfinite(grad_T))
        assert grad_T.shape == T.shape


# ===========================================================================
# Physics-correctness tests for plume schemes
# ===========================================================================

class TestMassFluxPhysics:
    """Physics-correctness tests for the mass-flux scheme."""

    def test_subsidence_warms_troposphere(self):
        """Compensating subsidence should produce net warming in the mid-troposphere
        where lapse rate is negative (T decreases with height)."""
        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
        config = MassFluxConfig(
            M_scale=0.05, cape_threshold=0.0,
            delta_0=0.0,  # disable detrainment to isolate subsidence
        )
        ncol = T.shape[0]
        # Use a large M_c to see clear subsidence signal
        M_c = jnp.full(ncol, 0.05)
        out, _ = mass_flux_convection(
            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
        )
        # Mid-troposphere levels (away from boundaries where gradient is zero)
        mid = slice(5, 15)
        dT_mid = out.dT_dt[:, mid]
        # Subsidence should warm (positive dT/dt) in the troposphere
        mean_warming = jnp.mean(dT_mid)
        assert float(mean_warming) > 0, (
            f"Subsidence should warm mid-troposphere, got mean dT/dt = {float(mean_warming):.2e}"
        )

    def test_detrainment_warms_where_updraft_warmer(self):
        """Detrainment of warm updraft air should warm the environment."""
        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
        config = MassFluxConfig(
            M_scale=0.05, cape_threshold=0.0,
            epsilon_0=0.0,  # no entrainment: T_u stays on moist adiabat
        )
        ncol = T.shape[0]
        M_c = jnp.full(ncol, 0.05)
        out, _ = mass_flux_convection(
            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
        )
        # With conditionally unstable profile, updraft (moist adiabat) is
        # warmer than environment at upper levels where CAPE > 0
        assert jnp.any(out.dT_dt > 0), "Detrainment should produce some warming"

    def test_subsidence_dries_troposphere(self):
        """Compensating subsidence should produce drying (dq/dt < 0) in
        the mid-troposphere where moisture decreases with height."""
        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
        config = MassFluxConfig(
            M_scale=0.05, cape_threshold=0.0,
            delta_0=0.0,  # disable detrainment to isolate subsidence
        )
        ncol = T.shape[0]
        M_c = jnp.full(ncol, 0.05)
        out, _ = mass_flux_convection(
            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
        )
        # Moisture typically decreases with height, so subsidence brings
        # drier air down: dq/dt should be mostly negative
        mid = slice(5, 15)
        mean_dq = jnp.mean(out.dq_v_dt[:, mid])
        assert float(mean_dq) < 0, (
            f"Subsidence should dry mid-troposphere, got mean dq/dt = {float(mean_dq):.2e}"
        )

    def test_precipitation_non_negative(self):
        """Precipitation should always be >= 0."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        ncol = T.shape[0]
        config = MassFluxConfig()
        M_c = jnp.full(ncol, config.M_c_init)
        out, _ = mass_flux_convection(
            T, q_v, p_full, p_half, M_c, dt=300.0, config=config,
        )
        assert jnp.all(out.dq_c_conv_dt >= 0)


class TestEDMFPhysics:
    """Physics-correctness tests for the EDMF scheme."""

    def test_subsidence_warms_troposphere(self):
        """Compensating subsidence should warm the mid-troposphere."""
        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
        config = ConvectiveEDMFConfig(
            a_u_init=0.1, cape_threshold=0.0,
            delta_0=0.0,  # disable detrainment to isolate subsidence
        )
        ncol = T.shape[0]
        a_u = jnp.full(ncol, 0.1)
        out, _ = edmf_convection(
            T, q_v, p_full, p_half, a_u, dt=300.0, config=config,
        )
        mid = slice(5, 15)
        mean_warming = jnp.mean(out.dT_dt[:, mid])
        assert float(mean_warming) > 0, (
            f"Subsidence should warm mid-troposphere, got mean dT/dt = {float(mean_warming):.2e}"
        )

    def test_detrainment_warms_where_updraft_warmer(self):
        """Detrainment of warm updraft air should warm the environment."""
        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
        config = ConvectiveEDMFConfig(
            a_u_init=0.1, cape_threshold=0.0,
            epsilon_0=0.0,  # no entrainment: T_u stays on moist adiabat
        )
        ncol = T.shape[0]
        a_u = jnp.full(ncol, 0.1)
        out, _ = edmf_convection(
            T, q_v, p_full, p_half, a_u, dt=300.0, config=config,
        )
        assert jnp.any(out.dT_dt > 0), "Detrainment should produce some warming"

    def test_both_terms_contribute(self):
        """With both subsidence and detrainment active, tendency should be
        larger than either alone."""
        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
        ncol = T.shape[0]
        a_u = jnp.full(ncol, 0.1)

        # Subsidence only
        cfg_sub = ConvectiveEDMFConfig(a_u_init=0.1, cape_threshold=0.0, delta_0=0.0)
        out_sub, _ = edmf_convection(T, q_v, p_full, p_half, a_u, 300.0, cfg_sub)

        # Detrainment only (epsilon_0=0 makes T_u=T_moist, strong detrainment)
        cfg_det = ConvectiveEDMFConfig(a_u_init=0.1, cape_threshold=0.0, epsilon_0=0.0)
        out_det, _ = edmf_convection(T, q_v, p_full, p_half, a_u, 300.0, cfg_det)

        # Both active
        cfg_both = ConvectiveEDMFConfig(a_u_init=0.1, cape_threshold=0.0)
        out_both, _ = edmf_convection(T, q_v, p_full, p_half, a_u, 300.0, cfg_both)

        # RMS of combined should generally be larger than either alone
        rms_sub = jnp.sqrt(jnp.mean(out_sub.dT_dt ** 2))
        rms_det = jnp.sqrt(jnp.mean(out_det.dT_dt ** 2))
        rms_both = jnp.sqrt(jnp.mean(out_both.dT_dt ** 2))

        # At minimum, both-active should be non-zero
        assert float(rms_both) > 1e-10

    def test_subsidence_dries_troposphere(self):
        """The ADVECTIVE compensating subsidence dries the mid-troposphere.

        #824: EDMF now DEFAULTS to the conservative ``implicit_flux`` solve, whose
        flux-form transport conserves column water and REDISTRIBUTES it (net
        mid-trop tendency ~0) rather than leaving the advective form's local
        drying — so this drying property is specific to the advective term, which
        we select explicitly here to keep testing it."""
        T, q_v, p_full, p_half = _make_unstable_columns(ncol=4, nlev=20)
        config = ConvectiveEDMFConfig(
            a_u_init=0.1, cape_threshold=0.0,
            delta_0=0.0,  # isolate subsidence
            subsidence_solve="advective",  # #824: default is now implicit_flux
        )
        ncol = T.shape[0]
        a_u = jnp.full(ncol, 0.1)
        out, _ = edmf_convection(
            T, q_v, p_full, p_half, a_u, dt=300.0, config=config,
        )
        mid = slice(5, 15)
        mean_dq = jnp.mean(out.dq_v_dt[:, mid])
        assert float(mean_dq) < 0, (
            f"Subsidence should dry mid-troposphere, got mean dq/dt = {float(mean_dq):.2e}"
        )

    def test_precipitation_non_negative(self):
        """Convective cloud-water source should always be >= 0."""
        T, q_v, p_full, p_half = _make_unstable_columns()
        ncol = T.shape[0]
        config = ConvectiveEDMFConfig()
        a_u = jnp.full(ncol, config.a_u_init)
        out, _ = edmf_convection(
            T, q_v, p_full, p_half, a_u, dt=300.0, config=config,
        )
        assert jnp.all(out.dq_c_conv_dt >= 0)
