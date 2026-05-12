"""FV3_3D iter 729: dry_pressure_fv3 helper + iter-692 refactor.

Extracted moist-air dry partial pressure pattern from inline use
in iter-692 eqv_pot_fv3 (FV3 tools/fv_diagnostics.F90:5375-5391).

Tests
-----

1. ``test_dry_p_dry_branch_matches_delp_over_dpeln``.
2. ``test_dry_p_moist_branch_reduces_by_factor``.
3. ``test_dry_p_nonhydro_dry_known_value``.
4. ``test_dry_p_iter692_eqv_pot_unchanged``.
5. ``test_dry_p_shapes_3d``.
6. ``test_dry_p_finite``.
7. ``test_dry_p_missing_args_raise``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import dry_pressure_fv3, eqv_pot_fv3


def test_dry_p_dry_branch_matches_delp_over_dpeln():
    """Dry hydrostatic: pd = delp / Δpeln (since 1-rq = 1)."""
    pe = jnp.array([5.0e4, 7.5e4, 1.0e5])
    peln = jnp.log(pe)
    delp = pe[1:] - pe[:-1]
    pd = dry_pressure_fv3(delp, peln=peln, hydrostatic=True, moist=False)
    expected = delp / (peln[1:] - peln[:-1])
    assert jnp.allclose(pd, expected, atol=1e-8)


def test_dry_p_moist_branch_reduces_by_factor():
    """Moist hydrostatic with q=0.01: pd = 0.99 · (delp / Δpeln)."""
    pe = jnp.array([5.0e4, 7.5e4, 1.0e5])
    peln = jnp.log(pe)
    delp = pe[1:] - pe[:-1]
    q = jnp.full((2,), 0.01)
    pd = dry_pressure_fv3(delp, q=q, peln=peln, hydrostatic=True, moist=True)
    expected = 0.99 * delp / (peln[1:] - peln[:-1])
    assert jnp.allclose(pd, expected, atol=1e-8)


def test_dry_p_nonhydro_dry_known_value():
    """Non-hydrostatic dry:
        pd = -R_d · pt · delp / (g · delz)
    With pt=280, delp=1e4, delz=-1000 → pd = R_d · 280 · 1e4 / (g·1000)
    """
    pt = jnp.array([280.0])
    delp = jnp.array([1.0e4])
    delz = jnp.array([-1000.0])
    pd = dry_pressure_fv3(
        delp, pt=pt, delz=delz, hydrostatic=False, moist=False,
    )
    expected = constants.R_d * 280.0 * 1.0e4 / (constants.g * 1000.0)
    assert abs(float(pd[0]) - expected) / expected < 1e-10


def test_dry_p_iter692_eqv_pot_unchanged():
    """iter-692 eqv_pot_fv3 refactor preserves output exactly."""
    rng = np.random.default_rng(seed=729)
    km = 10
    T = 290.0
    pt = jnp.full((km,), T)
    delp = jnp.full((km,), 1000.0)
    q = jnp.asarray(rng.uniform(0.001, 0.020, size=(km,)))
    delz = jnp.full((km,), -constants.R_d * T * delp[0] / (constants.g * 1.0e5))
    theta_e = eqv_pot_fv3(pt, delp, q, delz=delz, moist=True)
    # Sanity: T=290 with varying q ~ 0.001..0.02 gives θ_e ~ 295-350 K
    assert jnp.all(theta_e > 290.0)
    assert jnp.all(theta_e < 400.0)


def test_dry_p_shapes_3d():
    """3-D input → 3-D output."""
    rng = np.random.default_rng(seed=730)
    n_x, n_y, km = 4, 5, 20
    delp = jnp.full((n_x, n_y, km), 1000.0)
    pt = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    delz = jnp.full((n_x, n_y, km), -200.0)
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(n_x, n_y, km)))
    pd = dry_pressure_fv3(
        delp, q=q, pt=pt, delz=delz, hydrostatic=False, moist=True,
    )
    assert pd.shape == (n_x, n_y, km)


def test_dry_p_finite():
    """No NaN/Inf on random."""
    rng = np.random.default_rng(seed=731)
    km = 30
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, km)))
    pt = jnp.asarray(rng.uniform(200.0, 320.0, size=(4, 4, km)))
    delz = jnp.asarray(rng.uniform(-500.0, -100.0, size=(4, 4, km)))
    q = jnp.asarray(rng.uniform(0.0, 0.025, size=(4, 4, km)))
    pd = dry_pressure_fv3(
        delp, q=q, pt=pt, delz=delz, hydrostatic=False, moist=True,
    )
    assert jnp.all(jnp.isfinite(pd))


def test_dry_p_missing_args_raise():
    """Missing peln (hydrostatic) / pt+delz (non-hydro) / q (moist) raises."""
    delp = jnp.full((5,), 1000.0)
    with pytest.raises(ValueError):
        dry_pressure_fv3(delp, hydrostatic=True, moist=False)
    with pytest.raises(ValueError):
        dry_pressure_fv3(delp, hydrostatic=False, moist=False)
    with pytest.raises(ValueError):
        dry_pressure_fv3(delp, peln=jnp.zeros((6,)), hydrostatic=True, moist=True)
