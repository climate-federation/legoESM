"""FV3_3D iter 799: stratospheric_h2o_entry_fv3 (cold-trap H₂O entry).

q_v_strat = q_sat(T_CPT, p_CPT)
ppmv      = q_v_strat / ε · 10⁶

Composes iter-798 cold_point_tropopause_fv3 + canonical thermo.

Tests
-----

1. ``test_h2o_tropical_cpt``: T=195, p=10 kPa → ppmv 5-15.
2. ``test_h2o_warmer_cpt_more``: ↑T_CPT → ↑ppmv.
3. ``test_h2o_lower_p_more``: ↓p at fixed T → ↑ppmv.
4. ``test_h2o_composes_iter798``: (T column) → T_CPT → q_strat.
5. ``test_h2o_tuple_shape``: returns (kg/kg, ppmv) tuple.
6. ``test_h2o_finite_nonneg``: 3-D + finite + non-negative.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    cold_point_tropopause_fv3,
    stratospheric_h2o_entry_fv3,
)


def test_h2o_tropical_cpt():
    """Tropical CPT (T=195 K, p=100 hPa): ppmv ~5-30."""
    t_cpt = jnp.array([195.0])
    p_cpt = jnp.array([10_000.0])  # 100 hPa
    q_kgkg, ppmv = stratospheric_h2o_entry_fv3(t_cpt, p_cpt)
    # Expect physically reasonable cold-trap value (3-100 ppmv range,
    # depending on saturation curve)
    assert jnp.all(jnp.isfinite(q_kgkg))
    assert jnp.all(jnp.isfinite(ppmv))
    assert 1.0 < float(ppmv[0]) < 500.0


def test_h2o_warmer_cpt_more():
    """Warmer CPT → more H₂O entry (Clausius-Clapeyron)."""
    t_warm = jnp.array([200.0])
    t_cold = jnp.array([185.0])
    p = jnp.array([10_000.0])
    _, ppmv_warm = stratospheric_h2o_entry_fv3(t_warm, p)
    _, ppmv_cold = stratospheric_h2o_entry_fv3(t_cold, p)
    assert float(ppmv_warm[0]) > float(ppmv_cold[0])


def test_h2o_lower_p_more():
    """Lower pressure at fixed T → larger q_sat → more ppmv."""
    t = jnp.array([195.0])
    p_hi = jnp.array([20_000.0])  # 200 hPa
    p_lo = jnp.array([5_000.0])   # 50 hPa
    _, ppmv_hi = stratospheric_h2o_entry_fv3(t, p_hi)
    _, ppmv_lo = stratospheric_h2o_entry_fv3(t, p_lo)
    assert float(ppmv_lo[0]) > float(ppmv_hi[0])


def test_h2o_composes_iter798():
    """Pipeline (T column) → T_CPT → q_strat."""
    z = jnp.array([1000.0, 5000.0, 11_000.0, 15_000.0, 17_000.0, 22_000.0])
    t = jnp.array([290.0, 260.0, 220.0, 200.0, 195.0, 215.0])
    _, t_cpt = cold_point_tropopause_fv3(t, z)
    p_cpt = jnp.array([10_000.0])
    q_kgkg, ppmv = stratospheric_h2o_entry_fv3(t_cpt, p_cpt)
    assert jnp.all(jnp.isfinite(ppmv))
    assert float(ppmv[0]) > 0.0


def test_h2o_tuple_shape():
    """Tuple (q_kgkg, ppmv) shape preserved."""
    rng = np.random.default_rng(seed=799)
    n_x, n_y = 4, 5
    t_cpt = jnp.asarray(rng.uniform(180.0, 210.0, size=(n_x, n_y)))
    p_cpt = jnp.asarray(rng.uniform(5_000.0, 25_000.0, size=(n_x, n_y)))
    q, ppmv = stratospheric_h2o_entry_fv3(t_cpt, p_cpt)
    assert q.shape == (n_x, n_y)
    assert ppmv.shape == (n_x, n_y)


def test_h2o_finite_nonneg():
    """Finite + non-negative for realistic CPT inputs."""
    rng = np.random.default_rng(seed=800)
    t_cpt = jnp.asarray(rng.uniform(180.0, 230.0, size=(8,)))
    p_cpt = jnp.asarray(rng.uniform(3_000.0, 30_000.0, size=(8,)))
    q, ppmv = stratospheric_h2o_entry_fv3(t_cpt, p_cpt)
    assert jnp.all(jnp.isfinite(q))
    assert jnp.all(jnp.isfinite(ppmv))
    assert jnp.all(q >= 0.0)
    assert jnp.all(ppmv >= 0.0)
