"""FV3_3D iter 817: effective_inflow_layer_fv3 (Thompson 2007).

EIL = { k : CAPE(k) ≥ 100 AND |CIN(k)| ≤ 250 }.

Tests
-----

1. ``test_eil_midlayer``: middle layers qualify → z_bot/z_top in mid.
2. ``test_eil_none_qualify_nan``: no layer qualifies → both NaN.
3. ``test_eil_single_layer``: one layer qualifies → z_bot=z_top.
4. ``test_eil_bottom_up``: surface-based EIL.
5. ``test_eil_shapes_batched``: 3-D batched (n_x, n_y, km) → (n_x, n_y).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import effective_inflow_layer_fv3


def test_eil_midlayer():
    """Levels 2-4 qualify; bottom and top of layer."""
    z = jnp.array([100.0, 500.0, 1500.0, 3000.0, 6000.0, 10_000.0])
    # CAPE high in middle, low elsewhere
    cape = jnp.array([50.0, 80.0, 500.0, 1000.0, 800.0, 50.0])
    # |CIN| low in middle, high elsewhere
    cin_mag = jnp.array([300.0, 280.0, 100.0, 50.0, 80.0, 400.0])
    z_bot, z_top = effective_inflow_layer_fv3(cape, cin_mag, z)
    # Valid mask: F, F, T, T, T, F → idx 2..4
    assert float(z_bot) == 1500.0
    assert float(z_top) == 6000.0


def test_eil_none_qualify_nan():
    """All CAPE < 100 → no EIL → NaN."""
    z = jnp.array([100.0, 500.0, 1500.0, 3000.0])
    cape = jnp.array([50.0, 30.0, 20.0, 10.0])
    cin_mag = jnp.array([0.0, 50.0, 100.0, 200.0])
    z_bot, z_top = effective_inflow_layer_fv3(cape, cin_mag, z)
    assert jnp.isnan(z_bot)
    assert jnp.isnan(z_top)


def test_eil_single_layer():
    """Only one level qualifies → z_bot=z_top."""
    z = jnp.array([100.0, 500.0, 1500.0, 3000.0])
    cape = jnp.array([50.0, 30.0, 500.0, 50.0])
    cin_mag = jnp.array([0.0, 50.0, 100.0, 200.0])
    z_bot, z_top = effective_inflow_layer_fv3(cape, cin_mag, z)
    assert float(z_bot) == 1500.0
    assert float(z_top) == 1500.0


def test_eil_bottom_up():
    """Surface-based EIL: lowest 3 levels qualify."""
    z = jnp.array([100.0, 500.0, 1500.0, 3000.0, 6000.0])
    cape = jnp.array([500.0, 600.0, 700.0, 50.0, 50.0])
    cin_mag = jnp.array([100.0, 80.0, 60.0, 300.0, 400.0])
    z_bot, z_top = effective_inflow_layer_fv3(cape, cin_mag, z)
    assert float(z_bot) == 100.0
    assert float(z_top) == 1500.0


def test_eil_shapes_batched():
    """3-D batched: (n_x, n_y, km) → (n_x, n_y)."""
    rng = np.random.default_rng(seed=817)
    n_x, n_y, km = 4, 5, 20
    z = jnp.cumsum(
        jnp.asarray(rng.uniform(100.0, 600.0, size=(n_x, n_y, km))), axis=-1
    )
    cape = jnp.asarray(rng.uniform(0.0, 2000.0, size=(n_x, n_y, km)))
    cin = jnp.asarray(rng.uniform(0.0, 500.0, size=(n_x, n_y, km)))
    z_bot, z_top = effective_inflow_layer_fv3(cape, cin, z)
    assert z_bot.shape == (n_x, n_y)
    assert z_top.shape == (n_x, n_y)
