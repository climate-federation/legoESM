"""FV3_3D iter 716: eqv_pot_bolton_fv3 port.

Faithful JAX port of FV3 Bolton-form ``eqv_pot``
(tools/fv_diagnostics.F90:5421-5497, Xi.Chen + SJL variant).
Alternative to iter-692 simplified S.-J. Lin form.

Tests
-----

1. ``test_bolton_dry_isothermal_at_p_ref``.
2. ``test_bolton_dry_poisson_form``.
3. ``test_bolton_moist_gt_dry``.
4. ``test_bolton_hydrostatic_branch``.
5. ``test_bolton_shapes_3d``.
6. ``test_bolton_finite``.
7. ``test_bolton_missing_arg_raises``.
8. ``test_bolton_differs_from_sjl``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import eqv_pot_bolton_fv3, eqv_pot_fv3


def test_bolton_dry_isothermal_at_p_ref():
    """Dry, q=0, p_mb=1000 → θ_e = T exactly (Poisson factor = 1)."""
    km = 5
    T = 280.0
    pt = jnp.full((km,), T)
    q = jnp.zeros((km,))
    delp = jnp.full((km,), 1000.0)
    # Need p_mb = 1000: non-hydro pd = 1e5 Pa, delz chosen accordingly
    # p_mb = 0.01 · (-R_d/g · delp/delz · T · (1+0·q))
    # Want p_mb = 1000 → -R_d·delp/(g·delz)·T = 1e5
    # delz = -R_d·T·delp/(g·1e5)
    delz = jnp.full((km,), -constants.R_d * T * 1000.0 / (constants.g * 1.0e5))
    theta_e = eqv_pot_bolton_fv3(pt, delp, q, delz=delz, moist=False)
    assert jnp.all(jnp.abs(theta_e - T) < 1e-10 * T)


def test_bolton_dry_poisson_form():
    """Dry θ_e = T · (1000/p_mb)^kappa."""
    km = 5
    T = 250.0
    pt = jnp.full((km,), T)
    q = jnp.zeros((km,))
    delp = jnp.full((km,), 2000.0)
    p_target_pa = 5.0e4  # 500 mb
    delz = jnp.full((km,), -constants.R_d * T * delp[0] / (constants.g * p_target_pa))
    theta_e = eqv_pot_bolton_fv3(pt, delp, q, delz=delz, moist=False)
    expected = T * (1000.0 / 500.0) ** constants.kappa
    assert jnp.all(jnp.abs(theta_e - expected) / expected < 1e-10)


def test_bolton_moist_gt_dry():
    """Moist Bolton θ_e > dry θ_e for q > 0."""
    km = 5
    T = 290.0
    pt = jnp.full((km,), T)
    q_moist = jnp.full((km,), 0.015)
    q_dry = jnp.zeros((km,))
    delp = jnp.full((km,), 1000.0)
    delz = jnp.full((km,), -constants.R_d * T * delp[0] / (constants.g * 1.0e5))
    theta_e_moist = eqv_pot_bolton_fv3(pt, delp, q_moist, delz=delz, moist=True)
    theta_e_dry = eqv_pot_bolton_fv3(pt, delp, q_dry, delz=delz, moist=False)
    assert jnp.all(theta_e_moist > theta_e_dry)


def test_bolton_hydrostatic_branch():
    """Hydrostatic branch via peln."""
    km = 5
    T = 280.0
    pt = jnp.full((km,), T)
    q = jnp.zeros((km,))
    pe = jnp.array([5.0e4, 6.0e4, 7.0e4, 8.0e4, 9.0e4, 1.0e5])
    peln = jnp.log(pe)
    delp = pe[1:] - pe[:-1]
    theta_e = eqv_pot_bolton_fv3(pt, delp, q, peln=peln, hydrostatic=True, moist=False)
    assert jnp.all(jnp.isfinite(theta_e))
    assert jnp.all(theta_e > 0.0)


def test_bolton_shapes_3d():
    """3-D input → 3-D output."""
    rng = np.random.default_rng(seed=716)
    n_x, n_y, km = 4, 5, 30
    pt = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    delz = jnp.full((n_x, n_y, km), -200.0)
    theta_e = eqv_pot_bolton_fv3(pt, delp, q, delz=delz, moist=True)
    assert theta_e.shape == (n_x, n_y, km)


def test_bolton_finite():
    """No NaN/Inf on random inputs."""
    rng = np.random.default_rng(seed=717)
    km = 30
    pt = jnp.asarray(rng.uniform(220.0, 320.0, size=(4, 4, km)))
    q = jnp.asarray(rng.uniform(1e-6, 0.025, size=(4, 4, km)))
    delp = jnp.asarray(rng.uniform(500, 2000, size=(4, 4, km)))
    delz = jnp.asarray(rng.uniform(-500, -100, size=(4, 4, km)))
    theta_e = eqv_pot_bolton_fv3(pt, delp, q, delz=delz, moist=True)
    assert jnp.all(jnp.isfinite(theta_e))


def test_bolton_missing_arg_raises():
    """Missing peln (hydrostatic) / delz (non-hydro) raises."""
    km = 5
    pt = jnp.full((km,), 280.0)
    q = jnp.zeros((km,))
    delp = jnp.full((km,), 1000.0)
    with pytest.raises(ValueError):
        eqv_pot_bolton_fv3(pt, delp, q, hydrostatic=True)
    with pytest.raises(ValueError):
        eqv_pot_bolton_fv3(pt, delp, q, hydrostatic=False)


def test_bolton_differs_from_sjl():
    """For moist tropical conditions Bolton and SJL θ_e differ
    (Bolton uses T_LCL formula, SJL uses simpler L_v/(cp·T))."""
    km = 5
    T = 295.0
    pt = jnp.full((km,), T)
    q = jnp.full((km,), 0.018)
    delp = jnp.full((km,), 1000.0)
    delz = jnp.full((km,), -constants.R_d * T * delp[0] / (constants.g * 1.0e5))
    theta_e_bolton = eqv_pot_bolton_fv3(pt, delp, q, delz=delz, moist=True)
    theta_e_sjl = eqv_pot_fv3(pt, delp, q, delz=delz, moist=True)
    diff = float(jnp.max(jnp.abs(theta_e_bolton - theta_e_sjl)))
    assert diff > 0.1  # K-level difference expected
