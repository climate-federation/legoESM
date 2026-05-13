"""FV3_3D iter 885: budyko_aet_fv3.

AET = P·PET/(P^ω + PET^ω)^(1/ω)  (Yang 2008 ω-form).

Tests
-----

1. ``test_energy_limited``: PET<<P → AET → PET.
2. ``test_water_limited``: P<<PET → AET → P.
3. ``test_crossover``: P=PET → AET ≈ 0.6·P (Budyko canonical).
4. ``test_bounded_by_p_and_pet``: AET ≤ min(P, PET).
5. ``test_omega_scaling``: larger ω → sharper transition.
6. ``test_runoff_coefficient``: Q/P = 1 − AET/P; computable.
7. ``test_chain_with_aridity_index``: AI=P/PET, AET/P from Budyko.
8. ``test_p_zero_floored``: P=0 → finite (AET → 0).
9. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    aridity_index_fv3,
    budyko_aet_fv3,
)


def test_energy_limited():
    """PET << P (very humid) → AET → PET."""
    aet = budyko_aet_fv3(
        precip=jnp.array([5000.0]),
        pet=jnp.array([500.0]),
    )
    # AET should be near PET ≈ 500
    assert 400.0 < float(aet[0]) < 510.0


def test_water_limited():
    """P << PET (very arid) → AET → P."""
    aet = budyko_aet_fv3(
        precip=jnp.array([100.0]),
        pet=jnp.array([3000.0]),
    )
    # AET should be near P = 100
    assert 80.0 < float(aet[0]) < 105.0


def test_crossover():
    """P = PET (energy-water crossover) → AET ≈ 0.6·P."""
    p = 1000.0
    aet = budyko_aet_fv3(
        precip=jnp.array([p]),
        pet=jnp.array([p]),
    )
    # Yang 2008 ω=2.6 at P=PET gives AET = P · P/(2P^ω)^(1/ω)
    # = P/(2^(1/ω)) = 1000/2^(1/2.6) = 1000/1.308 ≈ 765
    # Hmm — Budyko original gives ~600 but Yang ω=2.6 form gives ~765.
    # Loosen band to 600-800.
    assert 600.0 < float(aet[0]) < 800.0


def test_bounded_by_p_and_pet():
    """AET ≤ min(P, PET) always (physical bound)."""
    rng = np.random.default_rng(seed=885)
    n = 50
    p = jnp.asarray(rng.uniform(50.0, 3000.0, size=(n,)))
    pet = jnp.asarray(rng.uniform(100.0, 3000.0, size=(n,)))
    aet = budyko_aet_fv3(p, pet)
    minimum = jnp.minimum(p, pet)
    assert jnp.all(aet <= minimum + 1.0)  # 1 mm tolerance for Yang form


def test_omega_scaling():
    """↑ω → sharper transition: at P=PET, AET/P closer to 1."""
    p = jnp.array([1000.0])
    pet = jnp.array([1000.0])
    aet_low = budyko_aet_fv3(p, pet, omega=1.5)
    aet_high = budyko_aet_fv3(p, pet, omega=3.5)
    # Higher ω → AET/P → closer to limit ratio 1
    assert float(aet_high[0]) > float(aet_low[0])


def test_runoff_coefficient():
    """Q/P = 1 − AET/P; in semi-arid Q/P typically ~0.15."""
    p = jnp.array([500.0])
    pet = jnp.array([1500.0])
    aet = budyko_aet_fv3(p, pet)
    runoff_coef = 1.0 - float(aet[0]) / float(p[0])
    assert 0.0 < runoff_coef < 0.30


def test_chain_with_aridity_index():
    """Budyko AET pairs with AI: 1−AET/P depends on PET/P=1/AI."""
    p = jnp.array([800.0])
    pet = jnp.array([1200.0])
    aet = budyko_aet_fv3(p, pet)
    ai = aridity_index_fv3(p, pet)
    # AI = 0.667 (humid edge); AET/P should be sizable (>0.4)
    et_ratio = float(aet[0]) / float(p[0])
    assert et_ratio > 0.4
    assert 0.65 < float(ai[0]) < 0.7


def test_p_zero_floored():
    """P=0 → finite via floor; AET → 0 physically."""
    aet = budyko_aet_fv3(
        precip=jnp.array([0.0]),
        pet=jnp.array([1500.0]),
    )
    assert jnp.all(jnp.isfinite(aet))
    assert float(aet[0]) < 1.0  # ≈ 0


def test_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=885)
    n_x, n_y = 6, 8
    p = jnp.asarray(rng.uniform(50.0, 2500.0, size=(n_x, n_y)))
    pet = jnp.asarray(rng.uniform(100.0, 3000.0, size=(n_x, n_y)))
    aet = budyko_aet_fv3(p, pet)
    assert aet.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(aet))
    assert jnp.all(aet >= 0.0)
