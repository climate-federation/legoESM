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
