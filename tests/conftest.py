"""Shared test fixtures for legoESM."""

from pathlib import Path

import pytest
import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere, CubedSphereGrid
from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState


def _neutralize_broken_mps_gelu_patch() -> None:
    """Undo ``jax-mps``'s broken monkeypatch of ``jax.nn.gelu`` / ``dot_product_attention``.

    The installed ``jax-mps`` plugin (Apple Metal) monkeypatches ``jax.nn.gelu``
    and ``jax.nn.dot_product_attention`` to route through fused ``mps.*``
    primitives — but those primitives ship NO vmap batching rule, so ANY
    ``vmap``-ed network (every ML-emulator: microphysics, GWD, SFNO) dies with
    ``NotImplementedError: Batching rule for 'mps.gelu' not implemented`` even
    under ``JAX_PLATFORMS=cpu`` (Metal is unused/broken here — see the
    metal-backend memory).  The fused kernel is worthless on CPU, so restore the
    stock JAX implementations the plugin saved.  No-op when the plugin is absent
    (e.g. Linux CI) or has not patched.
    """
    try:
        from jax_plugins.mps import ops as _mps_ops
    except Exception:
        return
    # The patch is applied LAZILY on first backend init; force it so the plugin
    # has saved the originals (``_gelu_original`` / ``_sdpa_original``) before we
    # read them back.
    jnp.zeros(())
    import jax.nn as _jnn
    from jax._src.nn import functions as _nnf

    original_gelu = getattr(_mps_ops, "_gelu_original", None)
    if original_gelu is not None and getattr(_jnn.gelu, "_mps_patched", False):
        _jnn.gelu = original_gelu
        _nnf.gelu = original_gelu
    original_sdpa = getattr(_mps_ops, "_sdpa_original", None)
    if original_sdpa is not None and getattr(
        _jnn.dot_product_attention, "_mps_patched", False
    ):
        _jnn.dot_product_attention = original_sdpa
        _nnf.dot_product_attention = original_sdpa


_neutralize_broken_mps_gelu_patch()


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


@pytest.fixture
def conservation_gate():
    """The shared matrix conservation gates, so pytest tests assert PASS/FAIL
    with the SAME gates the test-matrix runners use (one source of truth).

    Returns the ``legoesm.experiments.matrix.gates`` module:

        def test_mass_conserved(conservation_gate):
            ok, notes = conservation_gate.mass_gate(
                True, "", mass_series, component="atmosphere")
            assert ok, notes
    """
    from legoesm.experiments.matrix import gates
    return gates
