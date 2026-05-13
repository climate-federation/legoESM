"""FV3_3D iter 832: clausius_clapeyron_dqdt_fv3.

dq/dT |_RH = RH · q_sat · L_v / (R_v · T²).

Tests
-----

1. ``test_earth_surface_rate``: T=288 K, RH=1, p=1e5 → relative
   rate dq/q/dT ≈ 6.5–7.5 %/K (canonical CC).
2. ``test_finite_diff_matches``: dq/dT via finite diff of
   thermo.saturation_mixing_ratio matches analytic to 1%.
3. ``test_rh_linear``: dq/dT|RH=0.5 = 0.5 · dq/dT|RH=1.
4. ``test_positive``: dq/dT > 0 always (warming → wetter air).
5. ``test_t_zero_floored``: T=0 → finite (no NaN).
6. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants, thermo
from legoesm.grids.cubed_sphere import clausius_clapeyron_dqdt_fv3


def test_earth_surface_rate():
    """T=288, RH=1, p=1e5: dq/q/dT = L_v/(R_v·T²) ≈ 6.7 %/K."""
    t = jnp.array([288.0])
    p = jnp.array([1.0e5])
    dqdt = clausius_clapeyron_dqdt_fv3(t, p)
    q_sat = thermo.saturation_mixing_ratio(t, p)
    rel = float(dqdt[0] / q_sat[0])
    assert 0.060 < rel < 0.080, f"CC rate {rel*100:.2f} %/K outside 6-8% band"


def test_finite_diff_matches():
    """Finite diff of q_sat matches analytic dq_sat/dT to 1%."""
    t = jnp.array([280.0, 290.0, 300.0])
    p = jnp.full_like(t, 1.0e5)
    dqdt_analytic = clausius_clapeyron_dqdt_fv3(t, p)
    dt = 0.1
    q_plus = thermo.saturation_mixing_ratio(t + dt, p)
    q_minus = thermo.saturation_mixing_ratio(t - dt, p)
    dqdt_fd = (q_plus - q_minus) / (2.0 * dt)
    np.testing.assert_allclose(
        np.asarray(dqdt_analytic), np.asarray(dqdt_fd), rtol=2e-2
    )


def test_rh_linear():
    """dq/dT|RH=0.5 = 0.5 · dq/dT|RH=1."""
    t = jnp.array([290.0])
    p = jnp.array([1.0e5])
    dqdt_full = clausius_clapeyron_dqdt_fv3(t, p, rh=1.0)
    dqdt_half = clausius_clapeyron_dqdt_fv3(t, p, rh=0.5)
    np.testing.assert_allclose(
        np.asarray(dqdt_half), 0.5 * np.asarray(dqdt_full), rtol=1e-12
    )


def test_positive():
    """dq/dT > 0 always (warming → wetter air)."""
    t = jnp.linspace(200.0, 320.0, 13)
    p = jnp.full_like(t, 1.0e5)
    dqdt = clausius_clapeyron_dqdt_fv3(t, p)
    assert jnp.all(dqdt > 0.0)


def test_t_zero_floored():
    """T=0 → finite via t_floor (no NaN)."""
    t = jnp.array([0.0])
    p = jnp.array([1.0e5])
    dqdt = clausius_clapeyron_dqdt_fv3(t, p)
    assert jnp.all(jnp.isfinite(dqdt))


def test_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=832)
    n_x, n_y, n_z = 4, 5, 6
    t = jnp.asarray(rng.uniform(230.0, 310.0, size=(n_x, n_y, n_z)))
    p = jnp.asarray(rng.uniform(2.0e4, 1.0e5, size=(n_x, n_y, n_z)))
    rh = jnp.asarray(rng.uniform(0.0, 1.0, size=(n_x, n_y, n_z)))
    dqdt = clausius_clapeyron_dqdt_fv3(t, p, rh)
    assert dqdt.shape == (n_x, n_y, n_z)
    assert jnp.all(jnp.isfinite(dqdt))
    assert jnp.all(dqdt >= 0.0)
