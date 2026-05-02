"""Smoke tests for atmosphere/physics/learned_column.py."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.learned_column import (
    build_column_physics,
    make_column_physics_fn,
)
from legoesm.atmosphere.physics.neural_physics import NeuralPhysics


def test_build_column_physics_returns_neural_physics():
    nlev = 8
    key = jax.random.PRNGKey(0)
    model = build_column_physics(nlev=nlev, hidden_dim=16, n_layers=2, key=key)
    assert isinstance(model, NeuralPhysics)
    assert model.nlev == nlev


def test_build_column_physics_residual_scale_passes_through():
    nlev = 4
    key = jax.random.PRNGKey(1)
    model = build_column_physics(
        nlev=nlev, hidden_dim=8, n_layers=2, residual_scale=0.05, key=key
    )
    assert isinstance(model, NeuralPhysics)


def test_make_column_physics_fn_returns_callable():
    """Check that make_column_physics_fn produces a callable physics fn.

    We don't exercise the full spectral round-trip here (that needs a
    GaussianGrid + SpectralHydrostaticState) — just that the factory
    returns a function with the documented signature.
    """
    from legoesm.grids.gaussian import create_gaussian_grid

    nlev = 4
    key = jax.random.PRNGKey(0)
    model = build_column_physics(nlev=nlev, hidden_dim=8, n_layers=2, key=key)
    grid = create_gaussian_grid(n_max=10)

    fn = make_column_physics_fn(model, grid)
    assert callable(fn)
