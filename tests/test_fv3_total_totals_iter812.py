"""FV3_3D iter 812: total_totals_fv3.

TT = (T_850 + T_d850) − 2·T_500   (Miller 1972).

Tests
-----

1. ``test_tt_known``: analytic value.
2. ``test_tt_unit_invariant``: TT_K = TT_C (Celsius-shift cancels).
3. ``test_tt_cold_t500_higher``: ↓T_500 → ↑TT.
4. ``test_tt_moist_td_higher``: ↑T_d850 → ↑TT.
5. ``test_tt_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import total_totals_fv3


def test_tt_known():
    """Severe-storm sample: T850=22, Td850=18, T500=−20 °C → TT=22+18+40=80? No.

    Recompute: TT = (T850 + Td850) − 2·T500.  Use Celsius:
    T850=22, Td850=18, T500=−20: TT = (22 + 18) − 2·(−20) = 40 + 40 = 80.
    """
    t850 = jnp.array([22.0]) + 273.15
    td850 = jnp.array([18.0]) + 273.15
    t500 = jnp.array([-20.0]) + 273.15
    tt = total_totals_fv3(t850, td850, t500)
    # Celsius form: TT = 80
    # Kelvin form: same (TT is unit-invariant)
    np.testing.assert_allclose(np.asarray(tt), [80.0], rtol=1e-12)


def test_tt_unit_invariant():
    """TT identical regardless of K/°C input."""
    t850_C = jnp.array([22.0, 25.0, 15.0])
    td850_C = jnp.array([18.0, 22.0, 5.0])
    t500_C = jnp.array([-20.0, -25.0, -15.0])
    tt_C = total_totals_fv3(t850_C, td850_C, t500_C)
    tt_K = total_totals_fv3(t850_C + 273.15, td850_C + 273.15, t500_C + 273.15)
    np.testing.assert_allclose(np.asarray(tt_K), np.asarray(tt_C), rtol=1e-12)


def test_tt_cold_t500_higher():
    """↓T_500 → ↑TT (steeper mid-trop lapse rate)."""
    t850 = jnp.array([290.0])
    td850 = jnp.array([285.0])
    t500_warm = jnp.array([260.0])
    t500_cold = jnp.array([245.0])
    tt_warm = total_totals_fv3(t850, td850, t500_warm)
    tt_cold = total_totals_fv3(t850, td850, t500_cold)
    assert float(tt_cold[0]) > float(tt_warm[0])


def test_tt_moist_td_higher():
    """↑T_d850 → ↑TT (moister BL)."""
    t850 = jnp.array([290.0])
    t500 = jnp.array([250.0])
    td_dry = jnp.array([275.0])
    td_moist = jnp.array([288.0])
    tt_dry = total_totals_fv3(t850, td_dry, t500)
    tt_moist = total_totals_fv3(t850, td_moist, t500)
    assert float(tt_moist[0]) > float(tt_dry[0])


def test_tt_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=812)
    n_x, n_y = 6, 8
    t850 = jnp.asarray(rng.uniform(280.0, 300.0, size=(n_x, n_y)))
    td850 = jnp.asarray(rng.uniform(270.0, 295.0, size=(n_x, n_y)))
    t500 = jnp.asarray(rng.uniform(240.0, 260.0, size=(n_x, n_y)))
    tt = total_totals_fv3(t850, td850, t500)
    assert tt.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(tt))
