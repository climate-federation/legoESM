"""Shared test fixtures for legoESM."""

from pathlib import Path

import pytest
import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere, CubedSphereGrid
from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState


RESULTS_SUBDIRS = (
    "atmosphere/shallow_water",
    "atmosphere/hydrostatic",
    "atmosphere/nonhydrostatic",
    "ocean",
    "land",
    "sea_ice",
)


@pytest.fixture(scope="session", autouse=True)
def _ensure_results_tree() -> Path:
    """Ensure structured component result directories exist."""
    root = Path("results")
    for rel in RESULTS_SUBDIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def small_grid() -> CubedSphereGrid:
    """A small C8 grid for fast unit tests."""
    return create_cubed_sphere(8)


@pytest.fixture(scope="session")
def medium_grid() -> CubedSphereGrid:
    """A medium C24 grid for integration tests."""
    return create_cubed_sphere(24)


@pytest.fixture
def random_field(small_grid) -> Field:
    """A random scalar field on the small grid."""
    key = jax.random.PRNGKey(42)
    data = jax.random.normal(key, shape=(6, 8, 8))
    return Field(data=data, name="test", dims=("face", "x", "y"), units="1")


@pytest.fixture
def constant_state(small_grid) -> ShallowWaterState:
    """A constant state for testing conservation."""
    shape = (6, 8, 8)
    return ShallowWaterState(
        h=Field(data=jnp.ones(shape) * 1e4, name="h", dims=("face", "x", "y"), units="m"),
        u=Field(data=jnp.zeros(shape), name="u", dims=("face", "x", "y"), units="m/s"),
        v=Field(data=jnp.zeros(shape), name="v", dims=("face", "x", "y"), units="m/s"),
        h_s=Field(data=jnp.zeros(shape), name="h_s", dims=("face", "x", "y"), units="m"),
    )
