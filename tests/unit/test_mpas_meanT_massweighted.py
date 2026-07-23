"""Mass-weighted global mean T diagnostic (2026-07-23 hybrid-vs-sigma fix).

The MPAS day-line / timeseries ``mean T`` was an UNWEIGHTED mean over
(cells, levels).  On the stretched hybrid coordinate the many thin warm
near-surface levels each get one equal vote, overstating the global mean by
~+9 K relative to the mass-weighted value on the SAME state (quantified on
the hybrid-L20 day-8 checkpoint: 268.2 unweighted vs 259.0 mass-weighted),
while uniform sigma is nearly unbiased (+0.1 K).  Hybrid-vs-sigma stability
curves were therefore incomparable — reported as an apparent ~9 K "bug".

These tests pin the corrected convention, mean_T = sum(T*dp)/sum(dp) with
dp from the coordinate's own ``pressure_at_half``:
  - on the hybrid grid it differs from the unweighted mean in the expected
    direction for a warm-below profile (unweighted is warmer);
  - on the uniform-sigma grid with uniform p_s the two agree exactly;
  - the arithmetic matches an explicit hand sum.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels


def _mass_weighted_mean(T, coord, p_s):
    """The exact convention model_driver's MPAS diagnostics use."""
    p_half = coord.pressure_at_half(p_s)
    dp = p_half[..., 1:] - p_half[..., :-1]
    return float(jnp.sum(T * dp) / jnp.sum(dp))


def _warm_below_profile(coord, p_s, ncell):
    """T decreasing linearly with height (in pressure): T = 200 + 90*p/p_s."""
    p_full = coord.pressure_at_full(p_s)
    return 200.0 + 90.0 * p_full / p_s[..., None]


def test_hybrid_unweighted_mean_is_biased_warm():
    coord = make_hybrid_levels(20, p_top_Pa=200.0, stretching=2.0)
    ncell = 8
    p_s = jnp.full((ncell,), 1.0e5)
    T = _warm_below_profile(coord, p_s, ncell)
    unweighted = float(jnp.mean(T))
    mw = _mass_weighted_mean(T, coord, p_s)
    # Stretching packs thin near-surface (warm) levels -> equal-vote mean
    # sits several K above the mass-weighted truth.
    assert unweighted - mw > 2.0, (unweighted, mw)


def test_sigma_uniform_mean_matches_massweighted():
    coord = create_sigma_coordinate(30)
    ncell = 8
    p_s = jnp.full((ncell,), 1.0e5)
    T = _warm_below_profile(coord, p_s, ncell)
    unweighted = float(jnp.mean(T))
    mw = _mass_weighted_mean(T, coord, p_s)
    np.testing.assert_allclose(mw, unweighted, atol=0.2)


def test_massweighted_matches_explicit_hand_sum():
    coord = create_sigma_coordinate(2)
    p_s = jnp.array([1.0e5, 8.0e4])
    # Uniform sigma spacing: dp = dsigma*p_s with equal dsigma per level, so
    # the dsigma factor cancels and columns weight by p_s alone (works for
    # any sigma_top, incl. the 0.01 default).
    T = jnp.array([[300.0, 200.0], [280.0, 240.0]])
    expected = ((300.0 + 200.0) * 1.0 + (280.0 + 240.0) * 0.8) \
        / (2 * 1.0 + 2 * 0.8)
    np.testing.assert_allclose(
        _mass_weighted_mean(T, coord, p_s), expected, rtol=1e-6)


def test_hybrid_and_sigma_agree_on_same_state_massweighted():
    """The point of the fix: the SAME physical profile scores the same
    mass-weighted mean on both coordinate families (to interpolation-free
    analytic tolerance), where the unweighted means disagree by many K."""
    ncell = 4
    p_s = jnp.full((ncell,), 1.0e5)
    hyb = make_hybrid_levels(20, p_top_Pa=200.0, stretching=2.0)
    sig = create_sigma_coordinate(20)
    T_h = _warm_below_profile(hyb, p_s, ncell)
    T_s = _warm_below_profile(sig, p_s, ncell)
    mw_h = _mass_weighted_mean(T_h, hyb, p_s)
    mw_s = _mass_weighted_mean(T_s, sig, p_s)
    # T is linear in p, so sum(T*dp)/sum(dp) is nearly discretization-
    # independent (exact up to the midpoint-rule level placement).
    assert abs(mw_h - mw_s) < 1.0, (mw_h, mw_s)
    unw_gap = abs(float(jnp.mean(T_h)) - float(jnp.mean(T_s)))
    assert unw_gap > 2.0, unw_gap
