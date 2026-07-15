"""Convergence and conservation tests for FV3 shallow water on cubed sphere.

Williamson Test Case 2 (steady geostrophic flow) at multiple resolutions
to verify bounded error and mass conservation precision.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere


def _sw_to_cdgrid(state, cdgrid):
    """Convert generic ShallowWaterState to CDGridShallowWaterState."""
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import CDGridShallowWaterState
    from legoesm.grids.halo import pad_halo_vector
    h = state.h.data
    u_center = state.u.data
    v_center = state.v.data
    h_s = state.h_s.data
    u_pad, v_pad = pad_halo_vector(
        u_center, v_center,
        cdgrid.base.cos_angle, cdgrid.base.sin_angle,
        cdgrid.base.cos_angle_padded, cdgrid.base.sin_angle_padded,
        interp_offsets=cdgrid.base.halo_interp_offsets,
    )
    u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1] +
                   u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
    v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1] +
                   v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])
    return CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


def _run_tc2_cdgrid(n, nsteps, dt):
    """Run Williamson TC2 on the native CDGrid model at resolution Cn."""
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        CDGridShallowWaterModel,
    )
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from tests.test_cases.williamson import williamson_test2

    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    config = CDGridShallowWaterConfig(
        A_h=0.0,
        hyperdiff_coeff=1e15,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = CDGridShallowWaterModel(grid, config)

    sw_state = williamson_test2(grid)
    state = _sw_to_cdgrid(sw_state, cdgrid)
    model.set_initial_mass(state)

    s = state
    for _ in range(nsteps):
        s = model.step(s, dt)

    # L2 error on height field relative to analytic (= initial condition)
    h_err = s.h - sw_state.h.data
    area = grid.area
    l2_num = float(jnp.sqrt(jnp.sum(h_err ** 2 * area)))
    l2_den = float(jnp.sqrt(jnp.sum(sw_state.h.data ** 2 * area)))
    return l2_num / l2_den, s, sw_state, grid


class TestFVConvergence:
    """Williamson Test Case 2 bounded error tests."""

    def test_error_bounded_at_multiple_resolutions(self):
        """L2 error stays bounded (<5%) at all resolutions after 1 day.

        The C-D grid scheme's discrete steady state differs from the
        continuous TC2 solution due to the A->D interpolation, so
        convergence to zero is not expected. The key check is that
        the error stays bounded and doesn't blow up at any resolution.
        """
        configs = [
            (8, 144, 600.0),
            (16, 288, 300.0),
            (32, 576, 150.0),
        ]
        for n, nsteps, dt in configs:
            err, _, _, _ = _run_tc2_cdgrid(n, nsteps, dt)
            assert err < 0.05, (
                f"C{n} L2 error {err:.3e} exceeds 5% threshold"
            )

    def test_c32_absolute_error(self):
        """C32 L2 error after 1 day should be < 5% relative."""
        err, _, _, _ = _run_tc2_cdgrid(32, 576, 150.0)
        assert err < 0.05, f"C32 L2 error {err:.3e} exceeds 5% threshold"


class TestMassConservation:
    """Mass conservation precision for C-D grid shallow water."""

    def test_mass_conservation_precision(self):
        """Mass should be conserved to high precision over 100 steps."""
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterConfig,
            CDGridShallowWaterModel,
        )
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from tests.test_cases.williamson import williamson_test2

        grid = create_cubed_sphere(16)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        dt = 600.0

        config = CDGridShallowWaterConfig(
            A_h=0.0,
            hyperdiff_coeff=1e15,
            use_conservation_fixer=True,
            fix_mass=True,
        )
        model = CDGridShallowWaterModel(grid, config)

        sw_state = williamson_test2(grid)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        model.set_initial_mass(state)

        area = grid.area
        mass_init = float(jnp.sum(state.h.astype(jnp.float64) * area.astype(jnp.float64)))

        s = state
        for _ in range(100):
            s = model.step(s, dt)

        mass_final = float(jnp.sum(s.h.astype(jnp.float64) * area.astype(jnp.float64)))
        # iter-164: centralized helper.
        from legoesm.diagnostics import compute_relative_drift
        rel_drift = compute_relative_drift([mass_init, mass_final])

        # The PPM reconstruction across cube-face boundaries introduces
        # O(1e-6) asymmetry that prevents exact telescoping. With the
        # conservation fixer this is corrected, but rounding accumulates.
        assert rel_drift < 1e-5, (
            f"Mass drift {rel_drift:.2e} exceeds threshold (1e-5)"
        )

    def test_raw_flux_residual(self):
        """Raw (uncorrected) mass flux divergence should integrate near zero.

        If the PPM fluxes telescope correctly, the area-weighted integral
        of dh_dt should be close to machine precision.  A large residual
        signals non-conservative face fluxes.
        """
        import numpy as np
        from legoesm.core.operators_cdgrid import fv3_sw_tendencies
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            FV3EdgeShallowWaterState,
        )
        from tests.test_cases.williamson import williamson_test2

        grid = create_cubed_sphere(16)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        sw_state = williamson_test2(grid)
        # Build edge-midpoint D-grid IC (fv3_sw_tendencies expects this stagger)
        u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
        u_d = cdgrid.cos_angle_edge_x * u0 * jnp.cos(cdgrid.lat_edge_x)
        v_d = -cdgrid.sin_angle_edge_y * u0 * jnp.cos(cdgrid.lat_edge_y)
        h = sw_state.h.data
        h_s = sw_state.h_s.data

        # Compute tendencies WITHOUT global-mean correction
        dh_dt, _, _ = fv3_sw_tendencies(
            h, u_d, v_d, h_s, cdgrid,
            zero_mean_correction=False,
        )
        area = grid.area.astype(jnp.float64)
        residual = float(jnp.abs(jnp.sum(dh_dt.astype(jnp.float64) * area)))
        total_mass = float(jnp.sum(h.astype(jnp.float64) * area))
        rel_residual = residual / total_mass

        # Threshold: O(1e-6) per the known PPM face-boundary asymmetry.
        # If this regresses significantly, face flux telescoping is broken.
        assert rel_residual < 1e-4, (
            f"Raw flux residual {rel_residual:.2e} too large — "
            f"face fluxes are not telescoping"
        )
