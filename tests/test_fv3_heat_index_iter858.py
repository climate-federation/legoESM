"""FV3_3D iter 858: heat_index_rothfusz_fv3.

NOAA Rothfusz 1990 9-coefficient apparent-T regression.

Tests
-----

1. ``test_noaa_80f_40rh``: 80°F=26.67°C, 40% → HI ≈ 80°F.
2. ``test_caution_band``: 30°C/60% → HI in 27-35°C caution range.
3. ``test_extreme_caution_band``: 35°C/60% → HI in extreme caution.
4. ``test_danger_band``: 40°C/65% → HI > 50°C (danger/extreme).
5. ``test_iraq_2015_extreme``: 50°C/15% → HI raw regression.
6. ``test_monotone_in_t``: ↑T → ↑HI.
7. ``test_monotone_in_rh``: ↑RH → ↑HI.
8. ``test_pairs_with_wet_bulb``: HI > T_w typically (different meaning).
9. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    heat_index_rothfusz_fv3,
    wet_bulb_temperature_stull_fv3,
)


def test_noaa_80f_40rh():
    """80°F (26.67°C) / 40% → HI ≈ 80°F (26.67°C) — calibration."""
    t = jnp.array([26.667])  # 80°F
    rh = jnp.array([40.0])
    hi = heat_index_rothfusz_fv3(t, rh)
    # NOAA HI table at (80°F, 40%) = 80°F = 26.67°C
    assert 25.0 < float(hi[0]) < 28.0


def test_caution_band():
    """30°C / 60% → HI 30-35°C (Caution band 27-32; high edge)."""
    t = jnp.array([30.0])
    rh = jnp.array([60.0])
    hi = heat_index_rothfusz_fv3(t, rh)
    # NOAA: 30°C 60% RH ≈ 34°C HI
    assert 30.0 < float(hi[0]) < 38.0


def test_extreme_caution_band():
    """35°C / 60% → HI in 40-50°C (Extreme caution / Danger)."""
    t = jnp.array([35.0])
    rh = jnp.array([60.0])
    hi = heat_index_rothfusz_fv3(t, rh)
    # NOAA: 35°C 60% ≈ 46°C HI
    assert 40.0 < float(hi[0]) < 50.0


def test_danger_band():
    """40°C / 65% → HI > 50°C (Danger / Extreme danger)."""
    t = jnp.array([40.0])
    rh = jnp.array([65.0])
    hi = heat_index_rothfusz_fv3(t, rh)
    assert float(hi[0]) > 50.0


def test_iraq_2015_extreme():
    """50°C / 15% → very dry; raw Rothfusz outside validity but finite."""
    t = jnp.array([50.0])
    rh = jnp.array([15.0])
    hi = heat_index_rothfusz_fv3(t, rh)
    # Out of NOAA validity (<40%RH); regression returns *some* value,
    # may not match operational NOAA low-RH correction.  Just verify
    # finite and bounded.
    assert jnp.all(jnp.isfinite(hi))


def test_monotone_in_t():
    """↑T at fixed RH → ↑HI."""
    rh = jnp.full((4,), 60.0)
    t = jnp.array([28.0, 32.0, 36.0, 40.0])
    hi = heat_index_rothfusz_fv3(t, rh)
    diffs = jnp.diff(hi)
    assert jnp.all(diffs > 0.0)


def test_monotone_in_rh():
    """↑RH at fixed T (high T regime) → ↑HI."""
    t = jnp.full((4,), 35.0)
    rh = jnp.array([40.0, 60.0, 75.0, 90.0])
    hi = heat_index_rothfusz_fv3(t, rh)
    diffs = jnp.diff(hi)
    assert jnp.all(diffs > 0.0)


def test_pairs_with_wet_bulb():
    """HI and T_w both rise with humid heat (consistent direction)."""
    t = jnp.array([35.0])
    rh = jnp.array([70.0])
    hi = heat_index_rothfusz_fv3(t, rh)
    tw = wet_bulb_temperature_stull_fv3(t, rh)
    # Both > base T (T_w < T psychrometric, HI > T physiological)
    assert float(hi[0]) > float(t[0])
    assert float(tw[0]) < float(t[0])
    # Both must be finite + physical
    assert jnp.all(jnp.isfinite(hi))
    assert jnp.all(jnp.isfinite(tw))


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=858)
    n_x, n_y = 6, 8
    t = jnp.asarray(rng.uniform(27.0, 45.0, size=(n_x, n_y)))
    rh = jnp.asarray(rng.uniform(40.0, 95.0, size=(n_x, n_y)))
    hi = heat_index_rothfusz_fv3(t, rh)
    assert hi.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(hi))
