"""FV3_3D iter 825: cloud_optical_thickness_fv3 (Slingo 1989).

τ = 1.5 · LWP / (ρ_water · r_eff).

Tests
-----

1. ``test_tau_sc_deck``: LWP=0.1, r_eff=10 μm → τ=15.
2. ``test_tau_thin_cirrus``: LWP=0.005, r_eff=30 μm → τ≈0.25.
3. ``test_tau_zero_lwp``: LWP=0 → τ=0.
4. ``test_tau_monotone_lwp``: ↑LWP → ↑τ.
5. ``test_tau_monotone_reff``: ↑r_eff → ↓τ.
6. ``test_tau_r_eff_floored``: r_eff=0 → finite via floor.
7. ``test_tau_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import cloud_optical_thickness_fv3


def test_tau_sc_deck():
    """Sc deck: LWP=0.1 kg/m² (100 g/m²), r_eff=10 μm → τ=15."""
    lwp = jnp.array([0.1])
    r_eff = jnp.array([1.0e-5])
    tau = cloud_optical_thickness_fv3(lwp, r_eff)
    np.testing.assert_allclose(np.asarray(tau), [15.0], rtol=1e-12)


def test_tau_thin_cirrus():
    """Thin cirrus: LWP=0.005 kg/m², r_eff=30 μm → τ ≈ 0.25."""
    lwp = jnp.array([0.005])
    r_eff = jnp.array([3.0e-5])
    tau = cloud_optical_thickness_fv3(lwp, r_eff)
    np.testing.assert_allclose(np.asarray(tau), [0.25], rtol=1e-12)


def test_tau_zero_lwp():
    """LWP=0 → τ=0."""
    lwp = jnp.zeros((3,))
    r_eff = jnp.array([1e-5, 2e-5, 3e-5])
    tau = cloud_optical_thickness_fv3(lwp, r_eff)
    np.testing.assert_allclose(np.asarray(tau), jnp.zeros((3,)), atol=1e-15)


def test_tau_monotone_lwp():
    """↑LWP → ↑τ at fixed r_eff."""
    lwp = jnp.array([0.01, 0.05, 0.1, 0.2])
    r_eff = jnp.full((4,), 1.0e-5)
    tau = cloud_optical_thickness_fv3(lwp, r_eff)
    assert jnp.all(jnp.diff(tau) > 0.0)


def test_tau_monotone_reff():
    """↑r_eff → ↓τ at fixed LWP."""
    lwp = jnp.full((4,), 0.1)
    r_eff = jnp.array([5e-6, 1e-5, 2e-5, 4e-5])
    tau = cloud_optical_thickness_fv3(lwp, r_eff)
    assert jnp.all(jnp.diff(tau) < 0.0)


def test_tau_r_eff_floored():
    """r_eff=0 → τ huge but finite via floor."""
    lwp = jnp.array([0.1])
    r_eff = jnp.array([0.0])
    tau = cloud_optical_thickness_fv3(lwp, r_eff, r_eff_floor=1e-9)
    assert jnp.all(jnp.isfinite(tau))
    # 1.5·0.1/(1000·1e-9) = 1.5e5
    assert float(tau[0]) > 1e4


def test_tau_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=825)
    n_x, n_y = 6, 8
    lwp = jnp.asarray(rng.uniform(0.0, 0.5, size=(n_x, n_y)))
    r_eff = jnp.asarray(rng.uniform(5e-6, 5e-5, size=(n_x, n_y)))
    tau = cloud_optical_thickness_fv3(lwp, r_eff)
    assert tau.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(tau))
    assert jnp.all(tau >= 0.0)
