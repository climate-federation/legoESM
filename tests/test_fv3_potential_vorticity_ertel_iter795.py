"""FV3_3D iter 795: potential_vorticity_ertel_fv3 (PV = η·∂θ/∂z/ρ).

Composes iter-794 absolute_vorticity_fv3.

Tests
-----

1. ``test_pv_midlat_troposphere``: η=1e-4, ρ=1, ∂θ/∂z=5e-3 → 0.5 PVU.
2. ``test_pv_stratosphere``: η=1e-4, ρ=0.1, ∂θ/∂z=2e-2 → 20 PVU.
3. ``test_pv_zero_eta``: η=0 → PV=0.
4. ``test_pv_monotonic_inputs``: ↑η→↑PV, ↑∂θ/∂z→↑PV, ↑ρ→↓PV.
5. ``test_pv_composes_iter794``: (ζ, f, ρ, ∂θ/∂z) → η → PV.
6. ``test_pv_zero_rho_floored``: ρ=0 → finite via rho_floor.
7. ``test_pv_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    absolute_vorticity_fv3,
    potential_vorticity_ertel_fv3,
)


def test_pv_midlat_troposphere():
    """Mid-lat trop: η=1e-4, ρ=1, ∂θ/∂z=5e-3 → PV = 5e-7 = 0.5 PVU."""
    eta = jnp.array([1.0e-4])
    rho = jnp.array([1.0])
    dtheta_dz = jnp.array([5.0e-3])
    pv = potential_vorticity_ertel_fv3(eta, rho, dtheta_dz)
    pvu = pv * 1e6  # convert to PVU
    np.testing.assert_allclose(np.asarray(pvu), [0.5], rtol=1e-12)


def test_pv_stratosphere():
    """Stratosphere: η=1e-4, ρ=0.1, ∂θ/∂z=2e-2 → PV = 2e-5 = 20 PVU."""
    eta = jnp.array([1.0e-4])
    rho = jnp.array([0.1])
    dtheta_dz = jnp.array([2.0e-2])
    pv = potential_vorticity_ertel_fv3(eta, rho, dtheta_dz)
    pvu = pv * 1e6
    np.testing.assert_allclose(np.asarray(pvu), [20.0], rtol=1e-12)


def test_pv_zero_eta():
    """η=0 → PV=0."""
    eta = jnp.zeros((3,))
    rho = jnp.array([1.0, 0.5, 0.1])
    dtheta_dz = jnp.array([5e-3, 1e-2, 2e-2])
    pv = potential_vorticity_ertel_fv3(eta, rho, dtheta_dz)
    np.testing.assert_allclose(np.asarray(pv), jnp.zeros((3,)), atol=1e-15)


def test_pv_monotonic_inputs():
    """↑η → ↑PV; ↑∂θ/∂z → ↑PV; ↑ρ → ↓PV."""
    base_eta = jnp.array([1e-4, 1e-4, 1e-4])
    base_rho = jnp.array([1.0, 1.0, 1.0])
    base_dt = jnp.array([5e-3, 5e-3, 5e-3])
    pv_base = potential_vorticity_ertel_fv3(base_eta, base_rho, base_dt)
    assert jnp.all(potential_vorticity_ertel_fv3(base_eta * 2, base_rho, base_dt) > pv_base)
    assert jnp.all(potential_vorticity_ertel_fv3(base_eta, base_rho, base_dt * 2) > pv_base)
    assert jnp.all(potential_vorticity_ertel_fv3(base_eta, base_rho * 2, base_dt) < pv_base)


def test_pv_composes_iter794():
    """Pipeline (ζ, f, ρ, ∂θ/∂z) → η → PV."""
    zeta = jnp.array([1.0e-5])
    f = jnp.array([1.0e-4])
    rho = jnp.array([1.0])
    dtheta_dz = jnp.array([5.0e-3])
    eta = absolute_vorticity_fv3(zeta, f)  # = 1.1e-4
    pv = potential_vorticity_ertel_fv3(eta, rho, dtheta_dz)
    # 1.1e-4 · 5e-3 / 1 = 5.5e-7 = 0.55 PVU
    np.testing.assert_allclose(np.asarray(pv * 1e6), [0.55], rtol=1e-12)


def test_pv_zero_rho_floored():
    """ρ=0 → PV huge but finite via rho_floor."""
    eta = jnp.array([1.0e-4])
    rho = jnp.array([0.0])
    dtheta_dz = jnp.array([5.0e-3])
    pv = potential_vorticity_ertel_fv3(eta, rho, dtheta_dz, rho_floor=1e-6)
    assert jnp.all(jnp.isfinite(pv))
    # 1e-4 · 5e-3 / 1e-6 = 0.5 (huge in PVU: 5e5 PVU vs ~1 PVU normal)
    assert float(pv[0]) > 0.1


def test_pv_shapes_3d_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=795)
    n_x, n_y, km = 4, 5, 20
    eta = jnp.asarray(rng.uniform(-2e-4, 2e-4, size=(n_x, n_y, km)))
    rho = jnp.asarray(rng.uniform(0.05, 1.3, size=(n_x, n_y, km)))
    dtheta_dz = jnp.asarray(rng.uniform(-1e-2, 3e-2, size=(n_x, n_y, km)))
    pv = potential_vorticity_ertel_fv3(eta, rho, dtheta_dz)
    assert pv.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(pv))
