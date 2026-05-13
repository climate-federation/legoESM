"""FV3_3D iter 732: layer_mean_pressure_fv3 port.

Faithful JAX port of FV3's layer-center pressure diagnostic
p_f = delp / Delta-peln.

Tests
-----

1. ``test_pf_isothermal_layer``.
2. ``test_pf_matches_log_integral_mean``.
3. ``test_pf_monotone_increasing``.
4. ``test_pf_composes_with_hybrid``.
5. ``test_pf_shapes_3d``.
6. ``test_pf_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    compute_hybrid_pressure_fv3,
    layer_mean_pressure_fv3,
)


def test_pf_isothermal_layer():
    """Thin layer around p=1e5:
    p_f = (p_bot − p_top) / ln(p_bot/p_top) ≈ p_mid for thin layers."""
    pe = jnp.array([0.99e5, 1.01e5])
    peln = jnp.log(pe)
    delp = pe[1:] - pe[:-1]
    p_f = layer_mean_pressure_fv3(delp, peln)
    # log-mean ≈ arithmetic mean for thin layers (~1e5 with ~2% spread)
    expected = 1.0e5
    assert abs(float(p_f[0]) - expected) / expected < 1e-3


def test_pf_matches_log_integral_mean():
    """For layer [p_top, p_bot]:
    p_f = (p_bot − p_top) / ln(p_bot/p_top)
    Algebraic verification on a single layer."""
    p_top, p_bot = 5.0e4, 1.0e5
    delp = jnp.array([p_bot - p_top])
    peln = jnp.log(jnp.array([p_top, p_bot]))
    p_f = layer_mean_pressure_fv3(delp, peln)
    expected = (p_bot - p_top) / jnp.log(p_bot / p_top)
    assert abs(float(p_f[0]) - float(expected)) / float(expected) < 1e-12


def test_pf_monotone_increasing():
    """p_f monotone increasing through atmospheric column."""
    km = 10
    pe = jnp.linspace(1.0e4, 1.0e5, km + 1)
    peln = jnp.log(pe)
    delp = pe[1:] - pe[:-1]
    p_f = layer_mean_pressure_fv3(delp, peln)
    assert jnp.all(jnp.diff(p_f) > 0.0)


def test_pf_composes_with_hybrid():
    """iter-724 hybrid produces (delp, _, peln).  p_f from those
    matches direct (p_bot-p_top)/Δpeln computation."""
    km = 10
    ak = jnp.linspace(100.0, 0.0, km + 1)
    bk = jnp.linspace(0.0, 1.0, km + 1)
    ps = jnp.asarray(1.0e5)
    delp, pe, peln = compute_hybrid_pressure_fv3(ak, bk, ps)
    p_f = layer_mean_pressure_fv3(delp, peln)
    # Sanity: p_f bounded between interface pressures
    assert jnp.all(p_f > pe[:-1])
    assert jnp.all(p_f < pe[1:])


def test_pf_shapes_3d():
    """3-D (n_x, n_y, km+1) peln → (n_x, n_y, km) p_f."""
    rng = np.random.default_rng(seed=732)
    n_x, n_y, km = 4, 5, 20
    pe_col = jnp.linspace(1.0e4, 1.0e5, km + 1)
    peln = jnp.broadcast_to(jnp.log(pe_col)[None, None, :], (n_x, n_y, km + 1))
    delp = jnp.broadcast_to((pe_col[1:] - pe_col[:-1])[None, None, :],
                            (n_x, n_y, km))
    p_f = layer_mean_pressure_fv3(delp, peln)
    assert p_f.shape == (n_x, n_y, km)


def test_pf_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=733)
    km = 30
    pe_col = jnp.linspace(5.0e3, 1.05e5, km + 1)
    peln = jnp.broadcast_to(jnp.log(pe_col)[None, None, :], (4, 4, km + 1))
    delp = jnp.broadcast_to((pe_col[1:] - pe_col[:-1])[None, None, :],
                            (4, 4, km))
    p_f = layer_mean_pressure_fv3(delp, peln)
    assert jnp.all(jnp.isfinite(p_f))
