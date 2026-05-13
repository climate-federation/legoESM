"""FV3_3D iter 797: lapse_rate_tropopause_fv3 (WMO 1957).

Tests
-----

1. ``test_lrt_standard_atm``: ICAO standard atm → trop ~11 km.
2. ``test_lrt_no_inversion``: monotone decreasing T → top.
3. ``test_lrt_deep_inversion``: all dT/dz > thresh → bottom midpoint.
4. ``test_lrt_custom_thresh``: 1 K/km threshold picks lower level.
5. ``test_lrt_complementary_to_iter796``: compare with dynamic tropopause.
6. ``test_lrt_shapes_batched``: 3-D columns (n_x, n_y, km) → (n_x, n_y).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    dynamic_tropopause_fv3,
    lapse_rate_tropopause_fv3,
)


def test_lrt_standard_atm():
    """ICAO standard atmosphere: T decreases ~6.5 K/km to 11 km then isothermal.
    Expected z_LRT ≈ 11 km (midpoint of crossing layer)."""
    # 6 levels: 1, 5, 9, 11, 13, 15 km
    z = jnp.array([1000.0, 5000.0, 9000.0, 11_000.0, 13_000.0, 15_000.0])
    # T: surface 288 K, lapse 6.5 K/km in trop, isothermal above 11 km
    t = jnp.array([281.5, 255.5, 229.5, 216.5, 216.5, 216.5])
    z_lrt = lapse_rate_tropopause_fv3(t, z)
    # Trop crossing at midpoint of (9 km, 11 km) or (11 km, 13 km)
    # Lapse rates: layer1-2: (255.5-281.5)/4000 = -6.5e-3 (below thresh)
    # layer2-3: -6.5e-3 (below), layer3-4: 0 (above), layer4-5: 0 (above), layer5-6: 0
    # First above at idx 2 → z_mid between 9 km and 11 km = 10 km
    assert 9000.0 < float(z_lrt) <= 12_000.0


def test_lrt_no_inversion():
    """Monotone lapse rate −6.5 K/km everywhere → no tropopause → top midpoint."""
    z = jnp.array([1000.0, 5000.0, 10_000.0, 15_000.0, 20_000.0])
    t = jnp.array([281.5, 255.5, 223.0, 190.5, 158.0])  # 6.5 K/km lapse
    z_lrt = lapse_rate_tropopause_fv3(t, z)
    # All dT/dz = -6.5e-3 < -2e-3 → no crossing → top midpoint = (15+20)/2 = 17.5 km
    assert float(z_lrt) == 17_500.0


def test_lrt_deep_inversion():
    """All dT/dz > thresh → bottom midpoint."""
    z = jnp.array([1000.0, 5000.0, 10_000.0, 15_000.0, 20_000.0])
    t = jnp.array([200.0, 210.0, 220.0, 230.0, 240.0])  # T↑ with z always
    z_lrt = lapse_rate_tropopause_fv3(t, z)
    # All dT/dz = +2.5e-3 > -2e-3 → first crossing at idx 0 → midpoint(1, 5) = 3 km
    assert float(z_lrt) == 3000.0


def test_lrt_custom_thresh():
    """Tighter threshold picks lower level."""
    z = jnp.array([1000.0, 5000.0, 9000.0, 11_000.0, 13_000.0])
    t = jnp.array([281.5, 255.5, 229.5, 220.0, 216.0])
    # Lapse rates: -6.5e-3, -6.5e-3, -4.75e-3, -2e-3
    z_default = lapse_rate_tropopause_fv3(t, z, dT_dz_thresh=-2e-3)
    z_tighter = lapse_rate_tropopause_fv3(t, z, dT_dz_thresh=-5e-3)
    # Tighter (-5e-3) catches earlier crossing
    assert float(z_tighter) <= float(z_default)


def test_lrt_complementary_to_iter796():
    """LRT and DT both find tropopause-like levels in similar range."""
    # Build a sounding with stratospheric stability transition
    z = jnp.array([1000.0, 5000.0, 9000.0, 11_000.0, 13_000.0, 15_000.0, 20_000.0])
    t = jnp.array([288.0, 262.0, 236.0, 220.0, 216.5, 216.5, 220.0])
    # Build matching PV (low in trop, high in strato)
    pv = jnp.array([0.3e-6, 0.5e-6, 1.0e-6, 1.8e-6, 3.0e-6, 8.0e-6, 12.0e-6])
    z_lrt = lapse_rate_tropopause_fv3(t, z)
    z_dt = dynamic_tropopause_fv3(pv, z)
    # Both should be in 9-13 km range (upper troposphere)
    assert 9000.0 < float(z_lrt) < 14_000.0
    assert 9000.0 < float(z_dt) < 14_000.0


def test_lrt_shapes_batched():
    """3-D batched columns: (n_x, n_y, km) → (n_x, n_y)."""
    rng = np.random.default_rng(seed=797)
    n_x, n_y, km = 4, 5, 30
    z = jnp.cumsum(
        jnp.asarray(rng.uniform(200.0, 800.0, size=(n_x, n_y, km))), axis=-1
    )
    # Build T column: T = 290 - 6.5e-3·z for z<10 km, isothermal above
    z_flat = z.reshape(-1, km)
    t_flat = jnp.where(z_flat < 10_000.0, 290.0 - 6.5e-3 * z_flat, 290.0 - 65.0)
    t = t_flat.reshape(n_x, n_y, km)
    z_lrt = lapse_rate_tropopause_fv3(t, z)
    assert z_lrt.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(z_lrt))
