"""FV3_3D iter 813: sweat_index_fv3.

SWEAT = 12·Td850 + 20·(TT-49) + 2·U850 + U500 + 125·(sin(D500-D850)+0.2)

Miller (1972) with conditional gating.

Tests
-----

1. ``test_sweat_severe_all_terms``: classic severe → SWEAT > 300.
2. ``test_sweat_td_negative_zeros_term1``: Td<0 → 12·Td term=0.
3. ``test_sweat_tt_below_49_zeros_term2``: TT<49 → 20·(TT-49) term=0.
4. ``test_sweat_invalid_shear_zeros_term5``: bad shear → 125·sin term=0.
5. ``test_sweat_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import sweat_index_fv3


def test_sweat_severe_all_terms():
    """Classic tornadic Great Plains: Td=18°C, TT=55, U850=30, U500=50,
    D850=180 (S), D500=270 (W). Veering 90° with strong shear → SWEAT > 300."""
    td850 = jnp.array([18.0])
    tt = jnp.array([55.0])
    u850 = jnp.array([30.0])
    u500 = jnp.array([50.0])
    d850 = jnp.array([180.0])
    d500 = jnp.array([270.0])
    sweat = sweat_index_fv3(td850, tt, u850, u500, d850, d500)
    # term1=216, term2=120, term3=60, term4=50, term5=125·(sin(90°)+0.2)=125·1.2=150
    # SWEAT = 216+120+60+50+150 = 596
    np.testing.assert_allclose(np.asarray(sweat), [596.0], rtol=1e-12)
    assert float(sweat[0]) > 300.0


def test_sweat_td_negative_zeros_term1():
    """Td<0 → 12·Td term = 0."""
    td850 = jnp.array([-5.0])
    tt = jnp.array([55.0])
    u850 = jnp.array([30.0])
    u500 = jnp.array([50.0])
    d850 = jnp.array([180.0])
    d500 = jnp.array([270.0])
    sweat = sweat_index_fv3(td850, tt, u850, u500, d850, d500)
    # term1=0, term2=120, term3=60, term4=50, term5=150 → 380
    np.testing.assert_allclose(np.asarray(sweat), [380.0], rtol=1e-12)


def test_sweat_tt_below_49_zeros_term2():
    """TT < 49 → 20·(TT-49) term = 0."""
    td850 = jnp.array([18.0])
    tt = jnp.array([40.0])
    u850 = jnp.array([30.0])
    u500 = jnp.array([50.0])
    d850 = jnp.array([180.0])
    d500 = jnp.array([270.0])
    sweat = sweat_index_fv3(td850, tt, u850, u500, d850, d500)
    # term1=216, term2=0, term3=60, term4=50, term5=150 → 476
    np.testing.assert_allclose(np.asarray(sweat), [476.0], rtol=1e-12)


def test_sweat_invalid_shear_zeros_term5():
    """Bad shear direction (no veering) → 125·sin term=0."""
    td850 = jnp.array([18.0])
    tt = jnp.array([55.0])
    u850 = jnp.array([30.0])
    u500 = jnp.array([50.0])
    # D850=180, D500=170 (backing, not veering) → invalid
    d850 = jnp.array([180.0])
    d500 = jnp.array([170.0])
    sweat = sweat_index_fv3(td850, tt, u850, u500, d850, d500)
    # term1=216, term2=120, term3=60, term4=50, term5=0 → 446
    np.testing.assert_allclose(np.asarray(sweat), [446.0], rtol=1e-12)


def test_sweat_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=813)
    n_x, n_y = 6, 8
    td850 = jnp.asarray(rng.uniform(-5.0, 25.0, size=(n_x, n_y)))
    tt = jnp.asarray(rng.uniform(35.0, 65.0, size=(n_x, n_y)))
    u850 = jnp.asarray(rng.uniform(0.0, 60.0, size=(n_x, n_y)))
    u500 = jnp.asarray(rng.uniform(0.0, 100.0, size=(n_x, n_y)))
    d850 = jnp.asarray(rng.uniform(0.0, 360.0, size=(n_x, n_y)))
    d500 = jnp.asarray(rng.uniform(0.0, 360.0, size=(n_x, n_y)))
    sweat = sweat_index_fv3(td850, tt, u850, u500, d850, d500)
    assert sweat.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(sweat))
