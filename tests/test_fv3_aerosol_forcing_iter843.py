"""FV3_3D iter 843: aerosol_forcing_fv3.

ΔF_aero = − 20·τ_aero + (−0.45)·ln(N_d/N_d_ref).

Tests
-----

1. ``test_direct_only_present_day``: τ=0.02, ratio=1 → ΔF=−0.40 (ERFari).
2. ``test_indirect_only_present_day``: τ=0, ratio=1.5 → ΔF=−0.45·ln(1.5)≈−0.18.
3. ``test_no_perturbation_zero``: τ=0, ratio=1 → ΔF=0.
4. ``test_total_ar6_central``: combined gives ≈−1.1 W/m² (AR5/AR6).
5. ``test_dampens_ghg``: ΔF_aero + ΔF_CO₂(present-day) ≈ AR6 net.
6. ``test_ratio_zero_floored``: N_d=0 → finite (no NaN).
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    aerosol_forcing_fv3,
    radiative_forcing_co2_fv3,
)


def test_direct_only_present_day():
    """τ=0.02 (present-day mean), no indirect → ΔF=−0.40 W/m²."""
    tau = jnp.array([0.02])
    f = aerosol_forcing_fv3(tau, n_cdnc_ratio=jnp.array([1.0]))
    np.testing.assert_allclose(np.asarray(f), [-0.40], rtol=1e-12)


def test_indirect_only_present_day():
    """τ=0, ratio=1.5 → ΔF = −0.45·ln(1.5) ≈ −0.182 W/m²."""
    tau = jnp.array([0.0])
    f = aerosol_forcing_fv3(tau, n_cdnc_ratio=jnp.array([1.5]))
    expected = -0.45 * float(jnp.log(1.5))
    np.testing.assert_allclose(np.asarray(f), [expected], rtol=1e-12)
    assert -0.2 < float(f[0]) < -0.15


def test_no_perturbation_zero():
    """τ=0, ratio=1 → ΔF=0."""
    tau = jnp.array([0.0])
    f = aerosol_forcing_fv3(tau, n_cdnc_ratio=jnp.array([1.0]))
    np.testing.assert_allclose(np.asarray(f), [0.0], atol=1e-14)


def test_total_ar6_central():
    """Both effects → ΔF ≈ −1.06 W/m² (AR6 ERFari+aci median)."""
    # Tuned: τ=0.035 (direct=−0.7), ratio≈2.04 (indirect=−0.32) gives ~−1.0
    # Use AR6-consistent: pick (τ, ratio) that gives ≈−1.0.
    tau = jnp.array([0.011])    # direct=−0.22 (ERFari)
    ratio = jnp.array([6.5])    # indirect=−0.45·ln(6.5)≈−0.842 (ERFaci)
    f = aerosol_forcing_fv3(tau, ratio)
    assert -1.2 < float(f[0]) < -0.9


def test_dampens_ghg():
    """Present-day GHG (+2.21 from CO₂) + aerosol (−1.06) ≈ +1.15."""
    f_co2 = radiative_forcing_co2_fv3(jnp.array([420.0]))    # ≈+2.21
    f_aero = aerosol_forcing_fv3(
        jnp.array([0.011]), n_cdnc_ratio=jnp.array([6.5]),
    )
    net = f_co2 + f_aero
    assert 1.0 < float(net[0]) < 1.4


def test_ratio_zero_floored():
    """N_d=0 → finite via ratio_floor (no NaN)."""
    tau = jnp.array([0.0])
    f = aerosol_forcing_fv3(tau, n_cdnc_ratio=jnp.array([0.0]))
    assert jnp.all(jnp.isfinite(f))


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=843)
    n_x, n_y = 6, 8
    tau = jnp.asarray(rng.uniform(0.0, 0.1, size=(n_x, n_y)))
    ratio = jnp.asarray(rng.uniform(0.5, 3.0, size=(n_x, n_y)))
    f = aerosol_forcing_fv3(tau, ratio)
    assert f.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(f))
