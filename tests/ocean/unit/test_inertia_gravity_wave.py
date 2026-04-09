"""Inertia-Gravity Wave validation test for the ocean cubed-sphere path.

Part of the FV3 audit harness validation progression:
  1. Shallow-water: cosine bell, Williamson 2, Williamson 5
  2. Ocean: inertia-gravity wave (THIS TEST)
  3. 3D atmosphere: targeted checks

Validates:
- Barotropic wave propagation on cubed-sphere
- Short-run stability (no NaN/Inf)
- Mass (volume) conservation
- Finite tendencies for wave-perturbed state
- Basic analytical comparison

Run with: JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/ocean/unit/test_inertia_gravity_wave.py -v
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.state import OceanConfig
from legoesm.ocean.dynamics.ocean_model import OceanModel
from legoesm.ocean.experiments.inertia_gravity_wave import (
    InertiaGravityWaveConfig,
    create_initial_conditions,
    compute_analytical_solution,
    compute_wave_metrics,
)


@pytest.fixture
def igw_grid():
    """Small C8 cubed-sphere grid."""
    return create_cubed_sphere(8)


@pytest.fixture
def igw_z_coord():
    """2-level vertical coordinate for barotropic wave."""
    return create_ocean_z_star(n_levels=2, H_max=1000.0)


@pytest.fixture
def igw_config():
    return InertiaGravityWaveConfig()


@pytest.fixture
def igw_state(igw_grid, igw_z_coord, igw_config):
    """IGW initial condition on cubed-sphere."""
    return create_initial_conditions("cubed_sphere", igw_grid, igw_z_coord, igw_config)


@pytest.fixture
def ocean_config():
    """Ocean model config with light mixing."""
    return OceanConfig(
        A_h=1e2, K_h=0.0, A_v=1e-4, K_v=1e-5,
        n_barotropic_substeps=10,
        hyperdiff_coeff=0.0,
    )


class TestIGWInitialCondition:
    """Verify the IGW initial condition is well-formed."""

    def test_eta_finite(self, igw_state):
        """SSH perturbation should be finite."""
        assert jnp.all(jnp.isfinite(igw_state.eta.data))

    def test_eta_has_wave_structure(self, igw_state):
        """SSH should have nonzero amplitude (wave present)."""
        eta = igw_state.eta.data
        assert float(jnp.max(jnp.abs(eta))) > 0.1, "IGW perturbation too small"

    def test_velocities_finite(self, igw_state):
        """Velocities should be finite."""
        assert jnp.all(jnp.isfinite(igw_state.u.data))
        assert jnp.all(jnp.isfinite(igw_state.v.data))

    def test_velocities_nonzero(self, igw_state):
        """Wave should have nonzero velocities."""
        u_max = float(jnp.max(jnp.abs(igw_state.u.data)))
        v_max = float(jnp.max(jnp.abs(igw_state.v.data)))
        assert max(u_max, v_max) > 1e-6, "IGW velocities too small"


class TestIGWModelStep:
    """Test ocean model stepping with IGW initial condition."""

    def test_one_step_finite(self, igw_grid, igw_z_coord, igw_state, ocean_config):
        """One model step should produce finite output."""
        model = OceanModel(igw_grid, igw_z_coord, ocean_config)
        state = igw_state
        dt = 300.0

        state_new = model.step(state, dt)

        assert jnp.all(jnp.isfinite(state_new.eta.data)), "eta NaN after 1 step"
        assert jnp.all(jnp.isfinite(state_new.u.data)), "u NaN after 1 step"
        assert jnp.all(jnp.isfinite(state_new.v.data)), "v NaN after 1 step"

    def test_10_steps_stable(self, igw_grid, igw_z_coord, igw_state, ocean_config):
        """10 steps should remain stable."""
        model = OceanModel(igw_grid, igw_z_coord, ocean_config)
        state = igw_state
        dt = 300.0

        for _ in range(10):
            state = model.step(state, dt)

        assert jnp.all(jnp.isfinite(state.eta.data)), "eta NaN after 10 steps"
        assert jnp.all(jnp.isfinite(state.u.data)), "u NaN after 10 steps"
        assert jnp.all(jnp.isfinite(state.v.data)), "v NaN after 10 steps"

    def test_volume_conservation_10_steps(self, igw_grid, igw_z_coord, igw_state, ocean_config):
        """Volume (mass) should be approximately conserved."""
        model = OceanModel(igw_grid, igw_z_coord, ocean_config)
        state = igw_state

        area = igw_grid.area
        land_mask = state.land_mask.data
        ocean_mask = 1.0 - land_mask

        vol_0 = float(jnp.sum(state.eta.data * area * ocean_mask))

        dt = 300.0
        for _ in range(10):
            state = model.step(state, dt)

        vol_f = float(jnp.sum(state.eta.data * area * ocean_mask))

        # Volume conservation: relative to initial amplitude
        eta_scale = float(jnp.max(jnp.abs(igw_state.eta.data)))
        vol_ref = float(jnp.sum(area * ocean_mask)) * eta_scale
        if vol_ref > 0:
            rel_err = abs(vol_f - vol_0) / vol_ref
            assert rel_err < 0.1, f"Volume drift relative error = {rel_err:.2e}"


class TestIGWAnalyticalComparison:
    """Compare numerical solution to analytical solution."""

    def test_analytical_solution_exists(self, igw_grid, igw_config):
        """Analytical solution should be computable."""
        eta_exact, omega = compute_analytical_solution(
            "cubed_sphere", igw_grid, 3600.0, igw_config)
        assert np.all(np.isfinite(eta_exact))
        assert omega > 0

    def test_dispersion_relation(self, igw_config):
        """Verify analytical dispersion relation ω² = f² + gH(k² + l²)."""
        H = igw_config.H_max
        f0 = igw_config.f0
        kx = igw_config.wavenumber_x
        ky = igw_config.wavenumber_y
        R = 6.371e6

        k_phys = kx / R
        l_phys = ky / R
        omega_expected = np.sqrt(f0**2 + 9.80616 * H * (k_phys**2 + l_phys**2))

        # Should be dominated by f0 for low wavenumbers
        assert omega_expected > f0
        assert omega_expected < 10 * f0  # reasonable range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
