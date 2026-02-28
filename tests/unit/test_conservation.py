"""Unit tests for conservation fixers."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.core.conservation import (
    fix_mass_shallow_water,
    fix_energy_shallow_water,
    apply_conservation_fixer,
    compute_conservation_diagnostics,
)
from legoesm.core.operators import global_integral
from legoesm.grids.cubed_sphere import create_cubed_sphere


class TestConservationFixers:
    """Tests for mass and energy conservation fixers."""

    @pytest.fixture
    def grid(self):
        return create_cubed_sphere(8)

    @pytest.fixture
    def state(self, grid):
        shape = (6, 8, 8)
        dims = ("face", "x", "y")
        return ShallowWaterState(
            h=Field(data=jnp.ones(shape) * 1e4, name="h", dims=dims, units="m"),
            u=Field(data=jnp.ones(shape) * 10.0, name="u", dims=dims, units="m/s"),
            v=Field(data=jnp.ones(shape) * 5.0, name="v", dims=dims, units="m/s"),
            h_s=Field(data=jnp.zeros(shape), name="h_s", dims=dims, units="m"),
        )

    def test_mass_fixer_exact(self, grid, state):
        """Mass fixer should restore exact mass."""
        # Perturb mass
        key = jax.random.PRNGKey(0)
        perturbation = jax.random.normal(key, state.h.shape) * 10.0
        state_perturbed = state._replace(
            h=state.h.replace(data=state.h.data + perturbation)
        )

        state_fixed = fix_mass_shallow_water(state_perturbed, state, grid)

        mass_original = global_integral(state.h, grid)
        mass_fixed = global_integral(state_fixed.h, grid)
        # float32 limits precision to ~1e-7 relative; mass fixer is exact
        # in infinite precision, but float32 accumulation rounds off
        assert jnp.allclose(mass_fixed, mass_original, rtol=1e-5)

    def test_mass_fixer_preserves_gradients(self, grid, state):
        """Mass fixer should be differentiable."""
        def loss(h_data):
            state_new = state._replace(h=state.h.replace(data=h_data + 1.0))
            state_fixed = fix_mass_shallow_water(state_new, state, grid)
            return jnp.sum(state_fixed.h.data ** 2)

        grads = jax.grad(loss)(state.h.data)
        assert jnp.all(jnp.isfinite(grads))

    def test_energy_fixer_exact(self, grid, state):
        """Energy fixer should restore exact total energy."""
        # Perturb velocities (simulating a time step that drifts energy)
        state_perturbed = state._replace(
            u=state.u.replace(data=state.u.data * 1.05),
            v=state.v.replace(data=state.v.data * 0.97),
        )

        state_fixed = fix_energy_shallow_water(state_perturbed, state, grid)

        def total_energy(s):
            h, u, v, h_s = s.h.data, s.u.data, s.v.data, s.h_s.data
            ke = 0.5 * h * (u**2 + v**2)
            pe = 0.5 * 9.80616 * (h + h_s)**2
            return jnp.sum((ke + pe) * grid.area)

        E_old = total_energy(state)
        E_fixed = total_energy(state_fixed)
        assert jnp.allclose(E_fixed, E_old, rtol=1e-5), (
            f"Energy not restored: old={float(E_old):.6e}, fixed={float(E_fixed):.6e}"
        )

    def test_energy_fixer_large_perturbation(self, grid, state):
        """Energy fixer should handle >1% drift (no clamping)."""
        # Large velocity perturbation: ~21% KE increase
        state_perturbed = state._replace(
            u=state.u.replace(data=state.u.data * 1.10),
            v=state.v.replace(data=state.v.data * 1.10),
        )

        state_fixed = fix_energy_shallow_water(state_perturbed, state, grid)

        def total_energy(s):
            h, u, v, h_s = s.h.data, s.u.data, s.v.data, s.h_s.data
            ke = 0.5 * h * (u**2 + v**2)
            pe = 0.5 * 9.80616 * (h + h_s)**2
            return jnp.sum((ke + pe) * grid.area)

        E_old = total_energy(state)
        E_fixed = total_energy(state_fixed)
        assert jnp.allclose(E_fixed, E_old, rtol=1e-5), (
            f"Energy not restored for large drift: "
            f"old={float(E_old):.6e}, fixed={float(E_fixed):.6e}"
        )

    def test_energy_fixer_no_nan_when_pe_exceeds_total(self, grid, state):
        """Energy fixer must not produce NaN when PE > E_old."""
        # After mass fixer, h could increase enough that PE_new > E_old.
        # Simulate this by inflating h while keeping velocities small.
        state_high_pe = state._replace(
            h=state.h.replace(data=state.h.data * 2.0),
            u=state.u.replace(data=state.u.data * 0.01),
            v=state.v.replace(data=state.v.data * 0.01),
        )

        state_fixed = fix_energy_shallow_water(state_high_pe, state, grid)

        assert jnp.all(jnp.isfinite(state_fixed.u.data)), "u contains NaN/Inf"
        assert jnp.all(jnp.isfinite(state_fixed.v.data)), "v contains NaN/Inf"

    def test_energy_fixer_preserves_gradients(self, grid, state):
        """Energy fixer should be differentiable."""
        def loss(u_data):
            state_new = state._replace(
                u=state.u.replace(data=u_data * 1.02),
            )
            state_fixed = fix_energy_shallow_water(state_new, state, grid)
            return jnp.sum(state_fixed.u.data ** 2)

        grads = jax.grad(loss)(state.u.data)
        assert jnp.all(jnp.isfinite(grads)), "Gradients through energy fixer are not finite"

    def test_conservation_diagnostics(self, grid, state):
        """Diagnostics should return reasonable values."""
        diag = compute_conservation_diagnostics(state, grid)
        assert 'total_mass' in diag
        assert 'total_energy' in diag
        assert float(diag['total_mass']) > 0
        assert float(diag['total_energy']) > 0
