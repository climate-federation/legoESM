"""Differentiability tests for ocean dynamics and EOS.

Categories:
  3a) Ocean dynamics — cubed-sphere
  3b) Ocean dynamics — lat-lon
  3c) Ocean dynamics — MPAS
  3d) Equation of state
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field


def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac*100:.1f}% non-zero (need {min_nonzero_frac*100:.0f}%)"
    )


# ============================================================================
# 3a  Ocean dynamics — cubed-sphere
# ============================================================================

class TestCubedSphereOcean:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init import rest_state_ocean
        from legoesm.ocean.state import OceanConfig
        from legoesm.ocean.dynamics.ocean_model import OceanModel

        n, nlev = 4, 3
        grid = create_cubed_sphere(n)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        config = OceanConfig(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
        )
        self.model = OceanModel(grid, z_coord, config)
        self.state = rest_state_ocean(grid, z_coord)
        self.dt = 300.0

    def test_grad_wrt_T(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            out = model.step(s, dt)
            return jnp.sum(out.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "CS Ocean single step w.r.t. T")


# ============================================================================
# 3c  Ocean dynamics — MPAS
# ============================================================================

class TestMPASOcean:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel

        nlev = 3
        mesh = create_voronoi_mesh(2)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        self.model = MPASOceanModel(mesh, z_coord)
        self.state = rest_state_mpas_ocean(mesh, z_coord)
        self.dt = 300.0

    def test_grad_wrt_T(self):
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            out = model.step(s, dt)
            return jnp.sum(out.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "MPAS Ocean single step w.r.t. T")


# ============================================================================
# 3d  Equation of state
# ============================================================================

class TestEOS:

    def test_wright_eos_grad_T(self):
        from legoesm.ocean.eos import wright_eos

        T = jnp.array(20.0)
        S = jnp.array(35.0)
        p = jnp.array(0.0)

        drho_dT = jax.grad(lambda T: wright_eos(T, S, p))(T)
        assert jnp.isfinite(drho_dT), "drho/dT is not finite"
        # Warmer water is lighter (thermal expansion)
        assert drho_dT < 0, f"drho/dT should be negative, got {drho_dT}"

    def test_wright_eos_grad_S(self):
        from legoesm.ocean.eos import wright_eos

        T = jnp.array(20.0)
        S = jnp.array(35.0)
        p = jnp.array(0.0)

        drho_dS = jax.grad(lambda S: wright_eos(T, S, p))(S)
        assert jnp.isfinite(drho_dS), "drho/dS is not finite"
        # Saltier water is heavier (haline contraction)
        assert drho_dS > 0, f"drho/dS should be positive, got {drho_dS}"


# ============================================================================
# 3e  Barotropic solver differentiability
# ============================================================================

class TestBarotropicSolver:

    @pytest.fixture(autouse=True)
    def setup(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init import rest_state_ocean
        from legoesm.ocean.state import OceanConfig
        from legoesm.ocean.dynamics.ocean_model import OceanModel

        n, nlev = 4, 3
        grid = create_cubed_sphere(n)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        config = OceanConfig(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
            differentiable_barotropic=True,  # use lax.scan for AD
        )
        self.model = OceanModel(grid, z_coord, config)
        self.state = rest_state_ocean(grid, z_coord)
        self.dt = 300.0

    def test_grad_wrt_eta(self):
        """Gradient w.r.t. sea surface height through barotropic solver."""
        model, state, dt = self.model, self.state, self.dt

        def loss(eta_data):
            s = state._replace(eta=state.eta.replace(data=eta_data))
            out = model.step(s, dt)
            return jnp.sum(out.eta.data ** 2)

        grad = jax.grad(loss)(state.eta.data)
        assert_gradient_ok(grad, "Barotropic solver w.r.t. eta")

    def test_grad_3_steps(self):
        """Multi-step gradient through ocean model with differentiable barotropic.

        Regression for the ``cast_pytree(..., "storage")`` round-trip:
        ``OceanModel.step`` upcasts to compute precision on entry and
        used to leave the output at compute precision because the
        storage cast defaulted to ``allow_downcast=False``.  Under
        ``JAX_ENABLE_X64`` that turned fp32 inputs into fp64 outputs and
        ``jax.lax.scan`` rejected the carry-dtype mismatch.  The fix
        passes ``allow_downcast=True`` for the storage cast so the
        output dtype matches the input dtype.
        """
        model, state, dt = self.model, self.state, self.dt

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            def body(carry, _):
                return model.step(carry, dt), None
            s_final, _ = jax.lax.scan(body, s, None, length=3)
            return jnp.sum(s_final.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "Ocean 3 steps (diff barotropic) w.r.t. T")


# ============================================================================
# 3f  Ocean with physics (vertical mixing)
# ============================================================================

class TestOceanPhysics:

    def test_grad_with_vertical_mixing(self):
        """Gradient through ocean step with vertical mixing enabled."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init import rest_state_ocean
        from legoesm.ocean.state import OceanConfig
        from legoesm.ocean.dynamics.ocean_model import OceanModel

        n, nlev = 4, 3
        grid = create_cubed_sphere(n)
        z_coord = create_ocean_z_star(nlev, H_max=500.0)
        config = OceanConfig(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
            K_v=1e-4,  # Vertical diffusivity enabled
            A_v=1e-3,  # Vertical viscosity enabled
        )
        model = OceanModel(grid, z_coord, config)
        state = rest_state_ocean(grid, z_coord)

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            out = model.step(s, 300.0)
            return jnp.sum(out.T.data ** 2)

        grad = jax.grad(loss)(state.T.data)
        assert_gradient_ok(grad, "Ocean + vertical mixing w.r.t. T")
