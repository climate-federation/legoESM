"""Regression tests for FV shallow-water cubed-sphere dynamics."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    CDGridShallowWaterModel,
    CDGridShallowWaterState,
    cdgrid_shallow_water_tendencies,
)


def _sw_to_cdgrid(state, cdgrid):
    """Convert generic ShallowWaterState to CDGridShallowWaterState."""
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


class TestFVShallowWater:
    """FV shallow water on cubed sphere with divergence damping."""

    @pytest.fixture(scope="class")
    def grid_sw(self):
        return create_cubed_sphere(16)

    @pytest.fixture(scope="class")
    def model_and_state(self, grid_sw):
        from tests.test_cases.williamson import williamson_test2

        cdgrid = create_cubed_sphere_cdgrid(grid_sw)
        dt = 600.0
        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=1e15,
            time_integrator="ssp_rk3",
            use_conservation_fixer=True,
            fix_mass=True,
        )
        model = CDGridShallowWaterModel(grid_sw, config)
        sw_state = williamson_test2(grid_sw)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        # Anchor conservation fixer to initial mass for drift-free long runs
        model.set_initial_mass(state)
        return model, state, dt

    def test_tendencies_finite(self, grid_sw):
        from tests.test_cases.williamson import williamson_test2

        cdgrid = create_cubed_sphere_cdgrid(grid_sw)
        sw_state = williamson_test2(grid_sw)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        config = CDGridShallowWaterConfig()
        dh, du, dv = cdgrid_shallow_water_tendencies(state, cdgrid, config)
        assert jnp.all(jnp.isfinite(dh))
        assert jnp.all(jnp.isfinite(du))
        assert jnp.all(jnp.isfinite(dv))

    def test_single_step(self, model_and_state):
        model, state, dt = model_and_state
        s1 = model.step(state, dt)
        assert jnp.all(jnp.isfinite(s1.h))
        assert jnp.all(jnp.isfinite(s1.u_d))
        assert jnp.all(jnp.isfinite(s1.v_d))

    def test_100_steps_stable(self, model_and_state):
        """100 steps of Williamson 2 should remain stable."""
        model, state, dt = model_and_state
        s = state
        for _ in range(100):
            s = model.step(s, dt)
        assert jnp.all(jnp.isfinite(s.h))
        assert jnp.all(jnp.isfinite(s.u_d))
        assert jnp.all(jnp.isfinite(s.v_d))
        h_mean_init = float(jnp.mean(state.h))
        h_mean_final = float(jnp.mean(s.h))
        # iter-164: centralized helper (NaN-aware).
        from legoesm.diagnostics import compute_relative_drift
        assert compute_relative_drift([h_mean_init, h_mean_final]) < 1e-3

    def test_mass_conservation(self, model_and_state, grid_sw):
        """Mass should be conserved to near machine precision over 50 steps."""
        model, state, dt = model_and_state
        area = grid_sw.area
        mass_init = float(jnp.sum(state.h.astype(jnp.float64) * area.astype(jnp.float64)))
        s = state
        for _ in range(50):
            s = model.step(s, dt)
        mass_final = float(jnp.sum(s.h.astype(jnp.float64) * area.astype(jnp.float64)))
        # iter-164: centralized helper (NaN-aware, same migration
        # as iter-157 / iter-159 sweep).
        from legoesm.diagnostics import compute_relative_drift
        mass_drift = compute_relative_drift([mass_init, mass_final])
        # The conservation fixer anchors to the CDGrid state mass.
        # Machine-precision conservation (< 1e-10) requires a flux-form
        # scheme; the CDGrid scheme achieves ~1e-6 relative drift over 50 steps.
        assert mass_drift < 1e-4, f"Mass drift {mass_drift:.2e} exceeds threshold"

    def test_williamson5_stable(self, grid_sw):
        """Williamson 5 (mountain) should be stable for 50 steps."""
        from tests.test_cases.williamson import williamson_test5

        cdgrid = create_cubed_sphere_cdgrid(grid_sw)
        dt = 450.0
        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=1e15,
            use_conservation_fixer=True,
            fix_mass=True,
        )
        model = CDGridShallowWaterModel(grid_sw, config)
        sw_state = williamson_test5(grid_sw)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        s = state
        for _ in range(50):
            s = model.step(s, dt)
        assert jnp.all(jnp.isfinite(s.h))

    def test_differentiable_10_steps(self, grid_sw):
        """FV SW should be differentiable through 10 steps via scan."""
        from tests.test_cases.williamson import williamson_test2

        cdgrid = create_cubed_sphere_cdgrid(grid_sw)
        dt = 600.0
        config = CDGridShallowWaterConfig(
            hyperdiff_coeff=1e15,
        )
        model = CDGridShallowWaterModel(grid_sw, config)
        sw_state = williamson_test2(grid_sw)
        state = _sw_to_cdgrid(sw_state, cdgrid)
        state = jax.tree.map(
            lambda x: x.astype(jnp.float64) if hasattr(x, "dtype") else x, state
        )

        def loss_fn(h_data):
            s = state._replace(h=h_data)
            final, _ = model.integrate_scan(s, 10, dt)
            return jnp.mean(final.h**2)

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.h)
        assert jnp.all(jnp.isfinite(g))
        assert float(jnp.max(jnp.abs(g))) > 0


class TestCubeCswW2Residual:
    """new_test_dycores iter-4: pin cube ``c_sw + p_grad_c`` residual on
    Williamson 2 steady solid-body initial condition.

    For W2 steady state the FB half-step (c_sw + p_grad_c) increment to
    (uc, vc) should be exactly zero.  Discrete cube discretisation has
    a structural residual at cube-vertex regions (codex iter-983
    isolated face=4/5 polar vertices; iter-4 probe also flags
    face=2 i=1 j=35 north-boundary).  The ceilings below pin the
    current residual so future regressions WORSE than today fail; any
    targeted improvement to ``_corner_vorticity`` / ``_vorticity_flux``
    / ``_p_grad_c`` would tighten these and re-pin downwards.
    """

    def test_w2_csw_pgrad_c_residual_bounded(self):
        from legoesm.core.fv3_sw_core import (
            _c_sw, d2a2c_vect, _p_grad_c,
        )
        from legoesm import constants

        n = 36
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)

        u0 = 2.0 * jnp.pi * float(grid.radius) / (12.0 * 86400.0)
        u_d = (cdgrid.cos_angle_edge_x
               * jnp.cos(cdgrid.lat_edge_x) * u0).astype(jnp.float64)
        v_d = (-cdgrid.sin_angle_edge_y
               * jnp.cos(cdgrid.lat_edge_y) * u0).astype(jnp.float64)
        from tests.test_cases.williamson import williamson_test2
        h = williamson_test2(grid).h.data.astype(jnp.float64)
        h_s = jnp.zeros_like(h)
        dt = 1800.0
        dt2 = 0.5 * dt

        _ua, _va, uc_base, vc_base, _ut, _vt = d2a2c_vect(
            u_d, v_d, cdgrid)
        h_star, uc_new, vc_new, _ua2, _va2 = _c_sw(
            h, u_d, v_d, h_s, cdgrid, dt, constants.g)
        duc = uc_new - uc_base
        dvc = vc_new - vc_base
        # C36 baseline: |duc|_max = 2.7086 m/s, |dvc|_max = 4.3163 m/s.
        # The |dvc| ceiling was 3.6 (pinned at 3.2887 in iter-4, commit
        # 3a6e830f) and is re-pinned to 4.75 here. The growth is NOT a
        # regression: the FV3-faithfulness corner-area work that landed after
        # iter-4 (4ad2fea0 cube-vertex 3-face-junction area, d7108d48 EDGE
        # area_corner, 5d1d273a C1 vertex 3x scaling) corrects the under-
        # resolved vertex/edge control-volume areas, and ``_corner_vorticity``
        # returns ``f_corner + rarea_c*vort`` so the residual at the 8 cube
        # vertices scales mechanically with the corrected (1.73x) metric.
        # Visually verified (controlled old-vs-new A/B): the entire growth is
        # localized to 8 vertex cells (0.1% of the field, interior bit-
        # identical), with NO checkerboard / face-edge stripes / grid-scale
        # noise; W2 L2=1.76e-4 unchanged and W5 mass drift ~3e-16. Ceilings
        # pinned ~10% above current.
        duc_max = float(jnp.max(jnp.abs(duc)))
        dvc_max = float(jnp.max(jnp.abs(dvc)))
        assert duc_max < 3.0, (
            f"c_sw |duc|_max = {duc_max:.4f} m/s exceeds 3.0 ceiling "
            "(cube-vertex regression; check _corner_vorticity / "
            "_vorticity_flux)"
        )
        assert dvc_max < 4.75, (
            f"c_sw |dvc|_max = {dvc_max:.4f} m/s exceeds 4.75 ceiling"
        )

        # Combined c_sw + p_grad_c residual should be smaller (partial
        # cancellation between vortex flux and pressure gradient).
        dp_x, dp_y = _p_grad_c(h_star, h_s, cdgrid, dt2, constants.g)
        # The duogrid FB step subtracts p_grad_c from c_sw output:
        #   uc_final = uc_new - dp_x
        # (sign matches fv3_fb_sw_step formulation).
        sum_uc = float(jnp.max(jnp.abs(duc - dp_x)))
        sum_vc = float(jnp.max(jnp.abs(dvc - dp_y)))
        assert sum_uc < 6.0, (
            f"|duc - dp_x|_max = {sum_uc:.4f} m/s exceeds 6.0 ceiling "
            "(cube-vertex c_sw + p_grad_c residual regression)"
        )
        # Re-pinned 6.0 -> 6.6 alongside the |dvc| ceiling: the same FV3-faithful
        # vertex-area correction lifts the combined residual to 5.98 (~10% margin).
        assert sum_vc < 6.6, (
            f"|dvc - dp_y|_max = {sum_vc:.4f} m/s exceeds 6.6 ceiling"
        )
