"""Unit tests for shared thermodynamic utilities."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
)


class TestReconstructHalfLevelPressure:
    """Tests for hydrostatic interface-pressure reconstruction."""

    def test_shape_and_monotonicity(self):
        ncol, nlev = 3, 5
        p_full = jnp.broadcast_to(
            jnp.linspace(2.0e4, 1.0e5, nlev)[None, :], (ncol, nlev)
        )
        rho_full = jnp.broadcast_to(
            jnp.linspace(0.3, 1.2, nlev)[None, :], (ncol, nlev)
        )
        z_half = jnp.broadcast_to(
            jnp.linspace(30000.0, 0.0, nlev + 1)[None, :], (ncol, nlev + 1)
        )

        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p_full, rho_full=rho_full, z_half=z_half
        )

        assert p_half.shape == (ncol, nlev + 1)
        assert jnp.all(p_half > 0.0)
        # Top-to-bottom indexing -> pressure should increase with level index.
        assert jnp.all(p_half[:, 1:] > p_half[:, :-1])

    def test_works_with_4d_columns(self):
        nlev = 4
        p_full = jnp.ones((6, 2, 2, nlev)) * jnp.linspace(1.5e4, 9.5e4, nlev)
        rho_full = jnp.ones_like(p_full) * jnp.linspace(0.2, 1.0, nlev)
        z_half = jnp.ones((6, 2, 2, nlev + 1)) * jnp.linspace(25000.0, 0.0, nlev + 1)

        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p_full, rho_full=rho_full, z_half=z_half
        )

        assert p_half.shape == (6, 2, 2, nlev + 1)
        assert jnp.all(jnp.isfinite(p_half))


class TestPressureFromEOS:
    """Robustness checks for EOS pressure reconstruction."""

    def test_handles_unphysical_inputs_with_positive_finite_output(self):
        rho = jnp.array([1.0, -1.0e-3, 0.0, 2.0])
        theta = jnp.array([300.0, 280.0, -20.0, 0.0])
        p = pressure_from_eos(rho, theta)
        assert jnp.all(jnp.isfinite(p))
        assert jnp.all(p > 0.0)


class TestBoltonLCLTemperature:
    """Direct tests for the promoted-public Bolton (1980) Eq. 22 helper
    (formerly private; now the single shared implementation consumed by
    ``compute_moist_adiabat`` and ``convection._plume.compute_lcl``)."""

    def test_saturated_parcel_returns_parcel_temperature(self):
        from legoesm.thermo import saturation_mixing_ratio
        from legoesm.atmosphere.physics.thermodynamics import (
            bolton_lcl_temperature,
        )
        T = jnp.asarray([290.0])
        p = jnp.asarray([1.0e5])
        q_sat = saturation_mixing_ratio(T, p)
        T_lcl = bolton_lcl_temperature(T, p, q_sat)
        assert float(jnp.abs(T_lcl[0] - T[0])) < 1e-6

    def test_bolton_reference_parcel(self):
        """T=290 K, q=8 g/kg, p=1000 hPa -> T_LCL ~ 283 K (Bolton 1980,
        same reference case as the plume compute_lcl test)."""
        from legoesm.atmosphere.physics.thermodynamics import (
            bolton_lcl_temperature,
        )
        T_lcl = bolton_lcl_temperature(
            jnp.asarray([290.0]), jnp.asarray([1.0e5]), jnp.asarray([8.0e-3]),
        )
        assert 281.0 < float(T_lcl[0]) < 285.0

    def test_matches_plume_compute_lcl(self):
        """``compute_lcl`` must delegate to this helper (dedup guard):
        identical T_lcl for identical parcels."""
        from legoesm.atmosphere.physics.thermodynamics import (
            bolton_lcl_temperature,
        )
        from legoesm.atmosphere.physics.convection._plume import compute_lcl
        T = jnp.asarray([288.0, 300.0])
        p = jnp.asarray([9.8e4, 1.005e5])
        q = jnp.asarray([6.0e-3, 14.0e-3])
        p_full = jnp.linspace(5.0e3, 1.0e5, 12)[None, :].repeat(2, axis=0)
        lcl = compute_lcl(T, q, p, p_full)
        T_ref = bolton_lcl_temperature(T, p, q)
        assert float(jnp.max(jnp.abs(lcl.T_lcl - T_ref))) == 0.0


class TestLatentHeatVaporization:
    """Kirchhoff L(T) helper — the single shared implementation for the
    Kuo / ZM-dilute / Emanuel temperature-dependent latent heat."""

    def test_anchor_and_slope(self):
        from legoesm import constants
        from legoesm.atmosphere.physics.thermodynamics import (
            latent_heat_vaporization,
        )
        T = jnp.asarray([constants.T_freeze, constants.T_freeze + 10.0])
        L = latent_heat_vaporization(T)
        # L(T_freeze) = L_v exactly.
        assert float(L[0]) == constants.L_v
        # Slope = -(c_pw - c_pv) J/kg/K.
        slope = float((L[1] - L[0]) / 10.0)
        assert abs(slope + (constants.c_pw - constants.c_pv)) < 1e-9

    def test_c_liquid_override_changes_slope(self):
        """Emanuel passes its tunable CL in place of c_pw."""
        from legoesm import constants
        from legoesm.atmosphere.physics.thermodynamics import (
            latent_heat_vaporization,
        )
        T = jnp.asarray([constants.T_freeze + 20.0])
        c_l = 2500.0
        L = latent_heat_vaporization(T, c_liquid=c_l)
        expected = constants.L_v - (c_l - constants.c_pv) * 20.0
        assert abs(float(L[0]) - expected) < 1e-6
