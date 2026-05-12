"""FV3_3D iter 769: lcl_state_fv3 + MSE-conservation cross-check.

Single-call (T, p, q, z) → (T_LCL, p_LCL, z_LCL) triad that composes
iter-763 + iter-764 + iter-765.  Critical property: MSE is conserved
on the dry adiabat below LCL, so MSE(parcel) ≡ MSE(LCL).

Tests
-----

1. ``test_lcl_state_matches_separate_calls``: composing iter-763/764/765
   directly gives the same triad as ``lcl_state_fv3``.
2. ``test_lcl_state_mse_conservation``: MSE(parcel) − MSE(LCL) ≈ 0
   over a realistic sounding (this is the key regression guard).
3. ``test_lcl_state_dse_conservation``: DSE(parcel) − DSE(LCL) ≈ 0
   (exact by construction of iter-765).
4. ``test_lcl_state_shapes_3d``: 3-D shapes for all three outputs.
5. ``test_lcl_state_finite``.
6. ``test_lcl_state_above_parcel``: z_LCL > z, p_LCL < p, T_LCL < T.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    dry_static_energy_fv3,
    lcl_height_fv3,
    lcl_pressure_fv3,
    lcl_state_fv3,
    lcl_temperature_fv3,
    moist_static_energy_fv3,
)


def _sample_parcels():
    """Return a realistic sounding of parcel (T, p, q, z) tuples."""
    T = jnp.array([300.0, 295.0, 290.0, 285.0, 280.0])
    p_pa = jnp.array([101325.0, 95000.0, 85000.0, 75000.0, 65000.0])
    q = jnp.array([0.018, 0.014, 0.010, 0.006, 0.003])
    z = jnp.array([10.0, 500.0, 1500.0, 2500.0, 3500.0])
    return T, p_pa, q, z


def test_lcl_state_matches_separate_calls():
    """Triad output identical to iter-763/764/765 invoked separately."""
    T, p_pa, q, z = _sample_parcels()
    t_lcl, p_lcl, z_lcl = lcl_state_fv3(T, p_pa, q, z)
    # Reference computation via separate calls
    p_mb = p_pa / 100.0
    t_lcl_ref = lcl_temperature_fv3(T, p_mb, q)
    p_lcl_ref = lcl_pressure_fv3(T, p_pa, t_lcl_ref)
    z_lcl_ref = lcl_height_fv3(z, T, t_lcl_ref)
    np.testing.assert_allclose(np.asarray(t_lcl), np.asarray(t_lcl_ref), atol=1e-12)
    np.testing.assert_allclose(np.asarray(p_lcl), np.asarray(p_lcl_ref), atol=1e-9)
    np.testing.assert_allclose(np.asarray(z_lcl), np.asarray(z_lcl_ref), atol=1e-9)


def test_lcl_state_mse_conservation():
    """MSE(parcel) − MSE(LCL) ≈ 0 (analytical identity).

    On the dry adiabat below LCL:
      - q is conserved (no condensation)
      - DSE = c_pd·T + g·z is conserved (definition of iter-765)
      → MSE = DSE + L_v·q is conserved.

    Numerical drift should be at float64 roundoff level (≲ 1e-6 J/kg
    relative to ~3e5 J/kg MSE values).
    """
    T, p_pa, q, z = _sample_parcels()
    t_lcl, _, z_lcl = lcl_state_fv3(T, p_pa, q, z)
    mse_parcel = moist_static_energy_fv3(T, z, q)
    mse_lcl = moist_static_energy_fv3(t_lcl, z_lcl, q)
    diff = jnp.abs(mse_parcel - mse_lcl)
    # Float64 roundoff: MSE values are O(3e5 J/kg), expect <1e-6 J/kg drift
    assert jnp.all(diff < 1e-6), f"max MSE drift = {float(jnp.max(diff))} J/kg"


def test_lcl_state_dse_conservation():
    """DSE(parcel) − DSE(LCL) ≈ 0 exactly by iter-765 construction."""
    T, p_pa, q, z = _sample_parcels()
    t_lcl, _, z_lcl = lcl_state_fv3(T, p_pa, q, z)
    dse_parcel = dry_static_energy_fv3(T, z)
    dse_lcl = dry_static_energy_fv3(t_lcl, z_lcl)
    diff = jnp.abs(dse_parcel - dse_lcl)
    assert jnp.all(diff < 1e-6), f"max DSE drift = {float(jnp.max(diff))} J/kg"


def test_lcl_state_shapes_3d():
    """3-D shapes preserved for all three outputs."""
    rng = np.random.default_rng(seed=769)
    n_x, n_y, km = 4, 5, 20
    T = jnp.asarray(rng.uniform(240.0, 300.0, size=(n_x, n_y, km)))
    p = jnp.asarray(rng.uniform(10_000.0, 100_000.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.001, 0.020, size=(n_x, n_y, km)))
    z = jnp.asarray(rng.uniform(0.0, 10_000.0, size=(n_x, n_y, km)))
    t_lcl, p_lcl, z_lcl = lcl_state_fv3(T, p, q, z)
    assert t_lcl.shape == (n_x, n_y, km)
    assert p_lcl.shape == (n_x, n_y, km)
    assert z_lcl.shape == (n_x, n_y, km)


def test_lcl_state_finite():
    """No NaN/Inf for realistic atmospheric range."""
    rng = np.random.default_rng(seed=770)
    T = jnp.asarray(rng.uniform(220.0, 305.0, size=(8, 30)))
    p = jnp.asarray(rng.uniform(5_000.0, 105_000.0, size=(8, 30)))
    q = jnp.asarray(rng.uniform(1e-6, 0.025, size=(8, 30)))
    z = jnp.asarray(rng.uniform(0.0, 15_000.0, size=(8, 30)))
    t_lcl, p_lcl, z_lcl = lcl_state_fv3(T, p, q, z)
    assert jnp.all(jnp.isfinite(t_lcl))
    assert jnp.all(jnp.isfinite(p_lcl))
    assert jnp.all(jnp.isfinite(z_lcl))


def test_lcl_state_above_parcel():
    """Physical sanity: T_LCL<T, p_LCL<p, z_LCL>z for moist parcels."""
    T, p_pa, q, z = _sample_parcels()
    t_lcl, p_lcl, z_lcl = lcl_state_fv3(T, p_pa, q, z)
    assert jnp.all(t_lcl < T)
    assert jnp.all(p_lcl < p_pa)
    assert jnp.all(z_lcl > z)
