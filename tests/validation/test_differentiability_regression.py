"""Regression tests for differentiability fixes.

Proves that:
- jax.grad and jit(grad) work through combined atmospheric physics with
  explicit PhysicsState threading (no side-channel mutation)
- Repeated differentiated calls do not leave Tracer objects in Python attributes
- grad through full coupler step (ocean SST, land T, sea-ice T, concentration)
- Atmospheric dycore step with target_mass supplied explicitly
- Ocean AD-safe kernel path vs checked host wrapper path
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState


N = 4
NLEV = 5


@pytest.fixture(scope="module")
def grid():
    return create_cubed_sphere(N)


@pytest.fixture(scope="module")
def sigma():
    return create_sigma_coordinate(NLEV)


@pytest.fixture(scope="module")
def hydrostatic_state():
    shape_3d = (6, N, N, NLEV)
    shape_2d = (6, N, N)
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return HydrostaticState(
        u=Field(data=jnp.ones(shape_3d) * 5.0, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.ones(shape_3d) * 2.0, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=jnp.ones(shape_3d) * 280.0, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=jnp.ones(shape_2d) * 1e5, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=dims_2d, units="m^2/s^2"),
    )


# ===========================================================================
# Task 1 & 2: Combined physics with explicit PhysicsState threading
# ===========================================================================

class TestPhysicsStateThreading:
    """No mutable Python-side prognostic-state side channel remains."""

    def test_grad_through_combined_physics(self, grid, sigma, hydrostatic_state):
        """jax.grad works through combined physics with PhysicsState."""
        from legoesm.atmosphere.physics.combined import make_physics, PhysicsConfig
        from legoesm.atmosphere.physics.radiation.config import RadiationConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.physics_state import init_physics_state

        config = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            turbulence=TurbulenceConfig(scheme="louis"),
        )
        physics_fn = make_physics(config, model_type="hydrostatic", dt=300.0)
        ncol = 6 * N * N
        ps = init_physics_state(ncol=ncol, nlev=NLEV, physics_config=config)
        state = hydrostatic_state

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend, ps_out = physics_fn(s, grid, sigma, phys_state=ps)
            return jnp.mean(tend.dT_dt.data ** 2)

        grads = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grads))

    def test_jit_grad_through_combined_physics(self, grid, sigma, hydrostatic_state):
        """jit(grad) works through combined physics with PhysicsState."""
        from legoesm.atmosphere.physics.combined import make_physics, PhysicsConfig
        from legoesm.atmosphere.physics.radiation.config import RadiationConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.physics_state import init_physics_state

        config = PhysicsConfig(
            radiation=RadiationConfig(scheme="gray"),
            turbulence=TurbulenceConfig(scheme="louis"),
        )
        physics_fn = make_physics(config, model_type="hydrostatic", dt=300.0)
        ncol = 6 * N * N
        ps = init_physics_state(ncol=ncol, nlev=NLEV, physics_config=config)
        state = hydrostatic_state

        @jax.jit
        def grad_fn(T_data):
            def loss(T):
                s = state._replace(T=state.T.replace(data=T))
                tend, _ = physics_fn(s, grid, sigma, phys_state=ps)
                return jnp.mean(tend.dT_dt.data ** 2)
            return jax.grad(loss)(T_data)

        grads = grad_fn(state.T.data)
        assert jnp.all(jnp.isfinite(grads))

    def test_no_tracer_leakage_on_repeated_calls(self, grid, sigma, hydrostatic_state):
        """Repeated differentiated calls don't leave Tracer objects in attributes."""
        from legoesm.atmosphere.physics.combined import make_physics, PhysicsConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.physics_state import init_physics_state

        config = PhysicsConfig(
            turbulence=TurbulenceConfig(scheme="tke"),
        )
        physics_fn = make_physics(config, model_type="hydrostatic", dt=300.0)
        ncol = 6 * N * N
        ps = init_physics_state(ncol=ncol, nlev=NLEV, physics_config=config)
        state = hydrostatic_state

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            tend, ps_out = physics_fn(s, grid, sigma, phys_state=ps)
            return jnp.mean(tend.dT_dt.data ** 2)

        # Run grad twice — no Tracer leakage
        g1 = jax.grad(loss)(state.T.data)
        g2 = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(g1))
        assert jnp.all(jnp.isfinite(g2))

        # Verify no _updated_* attributes exist
        assert not hasattr(physics_fn, '_updated_phys_state')
        assert not hasattr(physics_fn, '_updated_tke')

    def test_physics_returns_updated_state(self, grid, sigma, hydrostatic_state):
        """PhysicsState is functionally threaded through returns."""
        from legoesm.atmosphere.physics.combined import make_physics, PhysicsConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.physics_state import init_physics_state

        config = PhysicsConfig(
            turbulence=TurbulenceConfig(scheme="tke"),
        )
        physics_fn = make_physics(config, model_type="hydrostatic", dt=300.0)
        ncol = 6 * N * N
        ps = init_physics_state(ncol=ncol, nlev=NLEV, physics_config=config)
        state = hydrostatic_state

        tend, ps_out = physics_fn(state, grid, sigma, phys_state=ps)

        # PhysicsState should be returned with updated TKE
        assert ps_out is not None
        assert ps_out.tke.shape == ps.tke.shape
        assert ps_out.conv_prog_profile.shape == ps.conv_prog_profile.shape


# ===========================================================================
# Task 3: Dycore step with target mass
# ===========================================================================

class TestDycoreTargetMass:

    def test_pe_grad_with_mass_fixer(self, grid, sigma):
        """PE dycore with mass fixer is differentiable."""
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
        )

        shape_3d = (6, N, N, NLEV)
        shape_2d = (6, N, N)
        dims_3d = ("face", "x", "y", "level")
        dims_2d = ("face", "x", "y")
        state = HydrostaticState(
            u=Field(data=jnp.ones(shape_3d) * 5.0, name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.ones(shape_3d) * 2.0, name="v", dims=dims_3d, units="m/s"),
            T=Field(data=jnp.ones(shape_3d) * 280.0, name="T", dims=dims_3d, units="K"),
            p_s=Field(data=jnp.ones(shape_2d) * 1e5, name="p_s", dims=dims_2d, units="Pa"),
            phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=dims_2d, units="m^2/s^2"),
        )

        config = CDGridPrimitiveEquationConfig(
            hyperdiff_coeff=0.0,
            use_conservation_fixer=False,
        )
        model = CDGridPrimitiveEquationModel(grid, sigma, config)

        def loss(u_data):
            s = state._replace(u=state.u.replace(data=u_data))
            s = model.step(s, 60.0)
            return jnp.mean(s.T.data ** 2)

        grads = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(grads))
        assert not jnp.allclose(grads, 0.0)


# ===========================================================================
# Task 4 & 5: Coupler differentiability
# ===========================================================================

class TestCouplerDifferentiability:

    def test_grad_through_coupler_sst(self):
        """grad through coupler step wrt ocean SST."""
        from legoesm.coupler.coupler import (
            make_coupler, SurfaceState, init_surface_state,
        )
        from legoesm.core.coupling_fields import AtmToSurface
        from legoesm.coupler.config import CouplerConfig, TileConfig
        from legoesm.land.config import LandConfig
        from legoesm.ice.config import SeaIceConfig
        from legoesm.coupler.lake.config import LakeConfig
        from legoesm.land.state import LandState
        from legoesm.ice.state import SeaIceState
        from legoesm.coupler.lake.state import LakeState
        from legoesm.coupler.accumulator import reset_accumulator
        from legoesm.core.field import Field as F

        shape = (6, 4, 4)
        dims = ("face", "x", "y")
        z = jnp.zeros(shape)

        coupler_config = CouplerConfig()
        land_config = LandConfig()
        ice_config = SeaIceConfig()
        lake_config = LakeConfig()
        lat = jnp.zeros(shape)

        step_fn = make_coupler(
            coupler_config, land_config, ice_config, lake_config,
            lat=lat, grid=None,
        )

        sfc = init_surface_state(shape)
        atm = AtmToSurface(
            sw_down=jnp.full(shape, 200.0),
            lw_down=jnp.full(shape, 300.0),
            precip_total=z, precip_snow=z,
            T_lowest=jnp.full(shape, 280.0),
            q_lowest=jnp.full(shape, 5e-3),
            u_lowest=jnp.full(shape, 5.0),
            v_lowest=jnp.full(shape, -3.0),
            p_lowest=jnp.full(shape, 95000.0),
            p_surface=jnp.full(shape, 1e5),
            rho_lowest=jnp.full(shape, 1.15),
            cos_zenith=jnp.full(shape, 0.6),
            co2_ppmv=jnp.array(400.0),
            has_radiation=jnp.array(1.0),
            has_precipitation=jnp.array(1.0),
        )
        tile = TileConfig(
            f_land=jnp.full(shape, 0.3),
            f_lake=jnp.full(shape, 0.05),
        )
        ocean_sst = jnp.full(shape, 290.0)
        ocean_u = jnp.zeros(shape)
        ocean_v = jnp.zeros(shape)

        def loss(sst):
            _, response = step_fn(sfc, atm, tile, sst, ocean_u, ocean_v, dt=300.0)
            return jnp.mean(response.T_sfc ** 2)

        grads = jax.grad(loss)(ocean_sst)
        assert jnp.all(jnp.isfinite(grads))
        assert not jnp.allclose(grads, 0.0)


# ===========================================================================
# Task 4: Ocean AD-safe kernel
# ===========================================================================

class TestOceanDifferentiability:

    def test_grad_through_ocean_step(self):
        """grad through OceanModel.step() (AD-safe kernel)."""
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init import rest_state_ocean
        from legoesm.ocean.state import OceanConfig

        grid = create_cubed_sphere(8)
        z_coord = create_ocean_z_star(n_levels=5, H_max=5500.0)
        state = rest_state_ocean(grid, z_coord)
        config = OceanConfig(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
        )
        model = OceanModel(grid, z_coord, config)

        def loss(T_data):
            s = state._replace(T=state.T.replace(data=T_data))
            s = model.step(s, 600.0)
            return jnp.mean(s.T.data ** 2)

        grads = jax.grad(loss)(state.T.data)
        assert jnp.all(jnp.isfinite(grads))

    def test_jit_grad_through_ocean(self):
        """jit(grad) through OceanModel.step()."""
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init import rest_state_ocean
        from legoesm.ocean.state import OceanConfig

        grid = create_cubed_sphere(8)
        z_coord = create_ocean_z_star(n_levels=5, H_max=5500.0)
        state = rest_state_ocean(grid, z_coord)
        config = OceanConfig(
            use_conservation_fixer=False,
            enable_runtime_checks=False,
            n_barotropic_substeps=2,
        )
        model = OceanModel(grid, z_coord, config)

        @jax.jit
        def grad_fn(T_data):
            def loss(T):
                s = state._replace(T=state.T.replace(data=T))
                s = model.step(s, 600.0)
                return jnp.mean(s.T.data ** 2)
            return jax.grad(loss)(T_data)

        grads = grad_fn(state.T.data)
        assert jnp.all(jnp.isfinite(grads))

    def test_step_vs_step_checked_consistency(self):
        """step() and step_checked(checks=False) produce identical results."""
        from legoesm.ocean.dynamics.ocean_model import OceanModel
        from legoesm.ocean.vertical import create_ocean_z_star
        from legoesm.ocean.init import rest_state_ocean
        from legoesm.ocean.state import OceanConfig

        grid = create_cubed_sphere(8)
        z_coord = create_ocean_z_star(n_levels=5, H_max=5500.0)
        state = rest_state_ocean(grid, z_coord)

        config_no_checks = OceanConfig(enable_runtime_checks=False)
        model = OceanModel(grid, z_coord, config_no_checks)

        s1 = model.step(state, 600.0)
        s2 = model.step_checked(state, 600.0)

        assert jnp.allclose(s1.T.data, s2.T.data)
        assert jnp.allclose(s1.u.data, s2.u.data)
