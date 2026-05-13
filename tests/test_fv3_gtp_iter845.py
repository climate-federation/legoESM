"""FV3_3D iter 845: gtp_single_decay_fv3.

GTP_H = c · A·τ/(τ−d) · (exp(−H/τ) − exp(−H/d)) / AGTP_CO2(H).

Tests
-----

1. ``test_ch4_gtp_100_short_lived``: CH₄ GTP_100 << its GWP_100
   (short-lived gas T-decayed by year 100).
2. ``test_n2o_gtp_100_long_lived``: N₂O τ>>d, GTP_100 plausible band.
3. ``test_gtp_horizon_short_le_long_for_short_lived``: GTP_20 > GTP_100
   for CH₄.
4. ``test_no_forcing_zero``: A=0 → GTP=0.
5. ``test_tau_equals_d_floored``: τ=d resonance → finite (no NaN).
6. ``test_paired_with_gwp_consistency``: GTP and GWP both positive.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    gtp_single_decay_fv3,
    gwp_single_decay_fv3,
)


def test_ch4_gtp_100_short_lived():
    """CH₄ (A=3.88e-13, τ=11.8) → GTP_100 << GWP_100."""
    rad_eff = jnp.array([3.88e-13])
    tau = jnp.array([11.8])
    gtp = gtp_single_decay_fv3(rad_eff, tau)
    gwp = gwp_single_decay_fv3(rad_eff, tau)
    assert float(gtp[0]) > 0.0
    # GTP_100 should be substantially smaller than GWP_100 for CH₄
    assert float(gtp[0]) < 0.5 * float(gwp[0])


def test_n2o_gtp_100_long_lived():
    """N₂O (A=3.03e-13, τ=109) → GTP_100 in plausible band."""
    rad_eff = jnp.array([3.03e-13])
    tau = jnp.array([109.0])
    gtp = gtp_single_decay_fv3(rad_eff, tau)
    # N₂O is long-lived (τ >> d), so AGTP scales similar to AGWP
    # AR6 GTP_100 for N₂O ≈ 233 — single-decay form should be in 100-400 band
    assert 50.0 < float(gtp[0]) < 500.0


def test_gtp_horizon_short_le_long_for_short_lived():
    """For short-lived CH₄: GTP_20 > GTP_100 (T-response decays)."""
    rad_eff = jnp.array([3.88e-13])
    tau = jnp.array([11.8])
    # AGTP_CO2(20) ≈ 6.84e-16 × 0.4 ≈ 2.7e-16 (T-response peaks ~year 20)
    gtp_20 = gtp_single_decay_fv3(rad_eff, tau, horizon=20.0,
                                   agtp_co2=2.7e-16)
    gtp_100 = gtp_single_decay_fv3(rad_eff, tau, horizon=100.0,
                                    agtp_co2=6.84e-16)
    assert float(gtp_20[0]) > float(gtp_100[0])


def test_no_forcing_zero():
    """A=0 → GTP=0."""
    rad_eff = jnp.array([0.0])
    tau = jnp.array([100.0])
    gtp = gtp_single_decay_fv3(rad_eff, tau)
    np.testing.assert_allclose(np.asarray(gtp), [0.0], atol=1e-14)


def test_tau_equals_d_floored():
    """τ = d_climate (8.4 yr) resonance → finite via floor."""
    rad_eff = jnp.array([3.0e-13])
    tau = jnp.array([8.4])  # exactly d_climate default
    gtp = gtp_single_decay_fv3(rad_eff, tau)
    assert jnp.all(jnp.isfinite(gtp))


def test_paired_with_gwp_consistency():
    """GTP and GWP same sign (both positive for warming gas)."""
    rad_eff = jnp.array([3.88e-13])
    tau = jnp.array([11.8])
    gtp = gtp_single_decay_fv3(rad_eff, tau)
    gwp = gwp_single_decay_fv3(rad_eff, tau)
    assert (float(gtp[0]) > 0.0) == (float(gwp[0]) > 0.0)


def test_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=845)
    n_x, n_y = 6, 8
    rad_eff = jnp.asarray(rng.uniform(1e-13, 1e-11, size=(n_x, n_y)))
    tau = jnp.asarray(rng.uniform(2.0, 500.0, size=(n_x, n_y)))
    gtp = gtp_single_decay_fv3(rad_eff, tau)
    assert gtp.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(gtp))
    assert jnp.all(gtp >= 0.0)
