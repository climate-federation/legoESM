"""FV3_3D iter 822: mixed_layer_height_fv3 (θ-jump detection).

h_ML = z[k*] where k* = argmin{k : θ(k) − θ(surface) > 0.5 K}.

Tests
-----

1. ``test_ml_well_mixed_then_jump``: uniform surface, sharp jump aloft.
2. ``test_ml_no_jump_top``: monotone θ within threshold → top.
3. ``test_ml_custom_threshold``: tighter threshold picks lower level.
4. ``test_ml_immediate_jump``: large surface inversion → bottom level.
5. ``test_ml_shapes_batched``: 3-D batched (n_x, n_y, km) → (n_x, n_y).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import mixed_layer_height_fv3


def test_ml_well_mixed_then_jump():
    """Surface θ=300, uniform to 1000 m, jump to θ=304 at 1500 m.
    Threshold 0.5 K triggers at the jump level."""
    z = jnp.array([50.0, 500.0, 1000.0, 1500.0, 3000.0])
    theta = jnp.array([300.0, 300.0, 300.0, 304.0, 310.0])
    h = mixed_layer_height_fv3(theta, z)
    assert float(h) == 1500.0


def test_ml_no_jump_top():
    """θ within threshold throughout → top of column."""
    z = jnp.array([50.0, 500.0, 1000.0, 1500.0, 3000.0])
    theta = jnp.array([300.0, 300.1, 300.2, 300.3, 300.4])  # max Δ = 0.4 < 0.5
    h = mixed_layer_height_fv3(theta, z)
    assert float(h) == 3000.0


def test_ml_custom_threshold():
    """Tighter threshold (0.1 K) picks earlier level."""
    z = jnp.array([50.0, 500.0, 1000.0, 1500.0, 3000.0])
    theta = jnp.array([300.0, 300.2, 300.5, 301.0, 305.0])
    h_default = mixed_layer_height_fv3(theta, z)
    h_tight = mixed_layer_height_fv3(theta, z, theta_jump_thresh=0.1)
    # Default 0.5: Δθ at k=2 is 0.5 (NOT > 0.5 strict) → first >0.5 at k=3 (z=1500)
    assert float(h_default) == 1500.0
    # Tight 0.1: Δθ at k=1 is 0.2 > 0.1 → first at z=500
    assert float(h_tight) == 500.0


def test_ml_immediate_jump():
    """Large surface inversion → first level above surface."""
    z = jnp.array([50.0, 500.0, 1000.0, 1500.0])
    theta = jnp.array([288.0, 295.0, 300.0, 305.0])  # Δθ at k=1 is 7 K
    h = mixed_layer_height_fv3(theta, z)
    assert float(h) == 500.0


def test_ml_shapes_batched():
    """3-D batched (n_x, n_y, km) → (n_x, n_y)."""
    rng = np.random.default_rng(seed=822)
    n_x, n_y, km = 4, 5, 20
    z = jnp.cumsum(
        jnp.asarray(rng.uniform(100.0, 500.0, size=(n_x, n_y, km))), axis=-1
    )
    # Build θ: surface 300, jumping aloft
    theta = 300.0 + jnp.linspace(0.0, 30.0, km)
    theta = jnp.broadcast_to(theta, (n_x, n_y, km))
    h = mixed_layer_height_fv3(theta, z)
    assert h.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(h))
