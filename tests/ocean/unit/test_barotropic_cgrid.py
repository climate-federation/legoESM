"""Tests for the C-grid barotropic solver on cubed-sphere (#182).

Validates:
- Stability (no NaN/Inf)
- Mass (volume) conservation
- Wave amplitude compared to A-grid solver
- Reduced diffusion (alpha=0.01 vs 0.05)
- Config validation
- Backward compatibility (A-grid default unchanged)

Run with: JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/ocean/unit/test_barotropic_cgrid.py -v
"""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.state import OceanConfig
from legoesm.ocean.dynamics.ocean_model import OceanModel
from legoesm.ocean.experiments.inertia_gravity_wave import (
    InertiaGravityWaveConfig,
    create_initial_conditions,
)


@pytest.fixture
def grid():
    return create_cubed_sphere(8)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=2, H_max=1000.0)


@pytest.fixture
def igw_state(grid, z_coord):
    return create_initial_conditions(
        "cubed_sphere", grid, z_coord, InertiaGravityWaveConfig(),
    )


def _make_config(staggering, alpha=None):
    """Build OceanConfig with the given barotropic staggering."""
    if alpha is None:
        alpha = 0.01 if staggering == "c_grid" else 0.05
    return OceanConfig(
        A_h=1e2, K_h=0.0, A_v=1e-4, K_v=1e-5,
        n_barotropic_substeps=10,
        barotropic_staggering=staggering,
        barotropic_diffusion_alpha=alpha,
    )


class TestCGridBarotropicStability:
    """C-grid solver should produce finite output."""

    def test_one_step_finite(self, grid, z_coord, igw_state):
        config = _make_config("c_grid")
        model = OceanModel(grid, z_coord, config)
        state_new = model.step(igw_state, 300.0)

        assert jnp.all(jnp.isfinite(state_new.eta.data)), "eta NaN"
        assert jnp.all(jnp.isfinite(state_new.u.data)), "u NaN"
        assert jnp.all(jnp.isfinite(state_new.v.data)), "v NaN"

    def test_10_steps_stable(self, grid, z_coord, igw_state):
        config = _make_config("c_grid")
        model = OceanModel(grid, z_coord, config)
        state = igw_state
        for _ in range(10):
            state = model.step(state, 300.0)

        assert jnp.all(jnp.isfinite(state.eta.data)), "eta NaN after 10 steps"
        assert jnp.all(jnp.isfinite(state.u.data)), "u NaN after 10 steps"


class TestCGridConservation:
    """Volume should be approximately conserved."""

    def test_volume_conservation(self, grid, z_coord, igw_state):
        config = _make_config("c_grid")
        model = OceanModel(grid, z_coord, config)
        state = igw_state
        area = grid.area

        vol_0 = float(jnp.sum(state.eta.data * area))
        for _ in range(10):
            state = model.step(state, 300.0)
        vol_f = float(jnp.sum(state.eta.data * area))

        eta_scale = float(jnp.max(jnp.abs(igw_state.eta.data)))
        vol_ref = float(jnp.sum(area)) * eta_scale
        if vol_ref > 0:
            rel_err = abs(vol_f - vol_0) / vol_ref
            assert rel_err < 0.1, f"Volume drift relative error = {rel_err:.2e}"


class TestCGridReducedDiffusion:
    """C-grid should work with alpha=0.01 (matching lat-lon/MPAS)."""

    def test_low_diffusion_stable(self, grid, z_coord, igw_state):
        """alpha=0.01 should be stable with C-grid (not with A-grid)."""
        config = _make_config("c_grid", alpha=0.01)
        model = OceanModel(grid, z_coord, config)
        state = igw_state
        for _ in range(10):
            state = model.step(state, 300.0)

        assert jnp.all(jnp.isfinite(state.eta.data))
        # Wave amplitude should be better preserved with lower diffusion
        eta_max = float(jnp.max(jnp.abs(state.eta.data)))
        assert eta_max > 0.01, f"Wave completely damped: max|eta|={eta_max}"


class TestCGridWaveAmplitude:
    """C-grid with alpha=0.01 should preserve more wave amplitude than A-grid with alpha=0.05."""

    def test_cgrid_less_damped(self, grid, z_coord, igw_state):
        dt = 300.0
        n_steps = 10

        # A-grid with its required higher diffusion
        config_a = _make_config("a_grid", alpha=0.05)
        model_a = OceanModel(grid, z_coord, config_a)
        state_a = igw_state
        for _ in range(n_steps):
            state_a = model_a.step(state_a, dt)

        # C-grid with lower diffusion
        config_c = _make_config("c_grid", alpha=0.01)
        model_c = OceanModel(grid, z_coord, config_c)
        state_c = igw_state
        for _ in range(n_steps):
            state_c = model_c.step(state_c, dt)

        eta_max_a = float(jnp.max(jnp.abs(state_a.eta.data)))
        eta_max_c = float(jnp.max(jnp.abs(state_c.eta.data)))

        # C-grid with lower diffusion should preserve more amplitude
        assert eta_max_c > eta_max_a, (
            f"C-grid (alpha=0.01) amplitude {eta_max_c:.4f} should exceed "
            f"A-grid (alpha=0.05) amplitude {eta_max_a:.4f}"
        )


class TestConfigValidation:
    """Config validation for barotropic_staggering field."""

    def test_valid_a_grid(self, grid, z_coord):
        config = _make_config("a_grid")
        OceanModel(grid, z_coord, config)  # should not raise

    def test_valid_c_grid(self, grid, z_coord):
        config = _make_config("c_grid")
        OceanModel(grid, z_coord, config)  # should not raise

    def test_invalid_staggering(self, grid, z_coord):
        config = OceanConfig(barotropic_staggering="b_grid")
        with pytest.raises(ValueError, match="barotropic_staggering"):
            OceanModel(grid, z_coord, config)

    def test_default_is_a_grid(self):
        config = OceanConfig()
        assert config.barotropic_staggering == "a_grid"


class TestBackwardCompatibility:
    """Default config should use A-grid solver (no regression)."""

    def test_default_matches_a_grid(self, grid, z_coord, igw_state):
        config_default = OceanConfig(
            A_h=1e2, K_h=0.0, A_v=1e-4, K_v=1e-5,
            n_barotropic_substeps=10,
        )
        config_a = OceanConfig(
            A_h=1e2, K_h=0.0, A_v=1e-4, K_v=1e-5,
            n_barotropic_substeps=10,
            barotropic_staggering="a_grid",
        )

        model_default = OceanModel(grid, z_coord, config_default)
        model_a = OceanModel(grid, z_coord, config_a)

        state_default = model_default.step(igw_state, 300.0)
        state_a = model_a.step(igw_state, 300.0)

        # Should be bitwise identical
        assert jnp.array_equal(state_default.eta.data, state_a.eta.data)
        assert jnp.array_equal(state_default.u.data, state_a.u.data)
        assert jnp.array_equal(state_default.v.data, state_a.v.data)
