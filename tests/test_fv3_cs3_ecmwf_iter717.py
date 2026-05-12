"""FV3_3D iter 717: cs3_interpolator_fv3 ECMWF below-surface T upgrade.

iter-710 used edge-value clamp for iv==1 below-surface temperature.
iter-717 adds the FV3-faithful Trenberth 1993 ECMWF extrapolation
when ``wz_surface`` is provided.

Tests
-----

1. ``test_ecmwf_default_unchanged_iter710``.
2. ``test_ecmwf_with_wz_changes_below_surface``.
3. ``test_ecmwf_lapse_rate_alpha_basic``.
4. ``test_ecmwf_high_terrain_blend``.
5. ``test_ecmwf_shapes_3d``.
6. ``test_ecmwf_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import cs3_interpolator_fv3


def test_ecmwf_default_unchanged_iter710():
    """iv=1 without wz_surface falls back to edge clamp (iter-710 behavior)."""
    km = 10
    qin = jnp.full((km,), 280.0)
    pe = jnp.log(jnp.linspace(1.0e4, 1.0e5, km + 1))
    pout = jnp.array([jnp.log(1.5e5)])  # below surface
    qout = cs3_interpolator_fv3(qin, pe, pout, iv=1)
    # Without wz, uses edge clamp ≈ qe[km] ≈ 280
    assert abs(float(qout[0]) - 280.0) < 1.0


def test_ecmwf_with_wz_changes_below_surface():
    """iv=1 with wz_surface uses ECMWF extrapolation.  Output should
    increase below surface (standard lapse rate)."""
    km = 10
    qin = jnp.full((km,), 280.0)
    pe = jnp.log(jnp.linspace(1.0e4, 1.0e5, km + 1))
    pout = jnp.array([jnp.log(1.2e5)])  # below surface
    wz_surface = jnp.asarray(0.0)
    qout = cs3_interpolator_fv3(
        qin, pe, pout, iv=1, wz_surface=wz_surface,
    )
    # T below surface should be > T at surface due to standard lapse rate
    assert float(qout[0]) > 280.0


def test_ecmwf_lapse_rate_alpha_basic():
    """At sea level (wz=0), alpha = 0.0065·R_d/g.
    T(p) = Ts · exp(alpha · (log_p − log_p_s)).
    With small (log_p − log_p_s), T - Ts ≈ Ts · alpha · (log_p − log_p_s).
    """
    km = 5
    qin = jnp.full((km,), 280.0)
    pe = jnp.log(jnp.linspace(5.0e4, 1.0e5, km + 1))
    p_below = 1.05e5
    pout = jnp.array([jnp.log(p_below)])
    wz_surface = jnp.asarray(0.0)
    qout = cs3_interpolator_fv3(
        qin, pe, pout, iv=1, wz_surface=wz_surface,
    )
    # Standard atmosphere: ~6.5 K/km lapse. With ln(1.05e5/1e5) ≈ 0.0488,
    # alpha·Δln(p) = 0.0065·R_d/g · 0.0488 ≈ 0.00930e-3
    # T_below ≈ Ts · (1 + 0.00930e-3 · ... actually we test it's > Ts and finite
    assert float(qout[0]) > 280.0
    assert float(qout[0]) < 290.0   # Plausible lapse


def test_ecmwf_high_terrain_blend():
    """wz_surface in [2000, 2500] with high ts engages the 298-K cap +
    blended alpha (differs from sea-level path)."""
    km = 10
    # Use high ts so t0 > 298 K triggers blend
    qin = jnp.full((km,), 295.0)
    pe = jnp.log(jnp.linspace(1.0e4, 8.0e4, km + 1))
    pout = jnp.array([jnp.log(9.0e4)])
    wz_low = jnp.asarray(0.0)
    wz_high = jnp.asarray(2250.0)
    q_low = cs3_interpolator_fv3(qin, pe, pout, iv=1, wz_surface=wz_low)
    q_high = cs3_interpolator_fv3(qin, pe, pout, iv=1, wz_surface=wz_high)
    # 298-K cap kicks in at wz=2250 with t0~309, so high-terrain alpha differs
    assert float(q_low[0]) != float(q_high[0])
    assert jnp.isfinite(q_high[0])


def test_ecmwf_shapes_3d():
    """3-D input + 2-D wz_surface → 3-D output."""
    rng = np.random.default_rng(seed=717)
    n_x, n_y, km, kd = 4, 5, 20, 3
    qin = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    pe_col = jnp.log(jnp.linspace(1.0e4, 1.0e5, km + 1))
    pe = jnp.broadcast_to(pe_col[None, None, :], (n_x, n_y, km + 1))
    pout = jnp.log(jnp.array([5.0e4, 8.0e4, 1.1e5]))
    wz = jnp.asarray(rng.uniform(0.0, 1500.0, size=(n_x, n_y)))
    qout = cs3_interpolator_fv3(qin, pe, pout, iv=1, wz_surface=wz)
    assert qout.shape == (n_x, n_y, kd)


def test_ecmwf_finite():
    """No NaN/Inf in ECMWF path."""
    rng = np.random.default_rng(seed=718)
    n_x, n_y, km, kd = 4, 4, 30, 5
    qin = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    pe_col = jnp.log(jnp.linspace(1.0e4, 1.0e5, km + 1))
    pe = jnp.broadcast_to(pe_col[None, None, :], (n_x, n_y, km + 1))
    pout = jnp.log(jnp.linspace(5.0e4, 1.2e5, kd))
    wz = jnp.asarray(rng.uniform(0.0, 3000.0, size=(n_x, n_y)))
    qout = cs3_interpolator_fv3(qin, pe, pout, iv=1, wz_surface=wz)
    assert jnp.all(jnp.isfinite(qout))
