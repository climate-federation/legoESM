"""FV3_3D iter 692: eqv_pot_fv3 port.

Faithful JAX port of FV3 ``eqv_pot`` (tools/fv_diagnostics.F90:
5341-5419).  Equivalent potential temperature θ_e.

Tests
-----

1. ``test_eqv_pot_dry_isothermal_at_p_ref``.
2. ``test_eqv_pot_dry_poisson_form``.
3. ``test_eqv_pot_moist_gt_dry``.
4. ``test_eqv_pot_hydrostatic_branch``.
5. ``test_eqv_pot_shapes_3d``.
6. ``test_eqv_pot_finite``.
7. ``test_eqv_pot_missing_arg_raises``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import eqv_pot_fv3


def test_eqv_pot_dry_isothermal_at_p_ref():
    """Dry, q=0, p = 1e5 Pa → θ_e = T exactly."""
    km = 5
    T = 280.0
    pt = jnp.full((km,), T)
    q = jnp.zeros((km,))
    delp = jnp.full((km,), 1000.0)
    # Choose delz so non-hydrostatic pd = 1e5
    # pd = -R_d·T·1·delp / (g·delz) = 1e5
    # delz = -R_d·T·delp / (g·1e5)
    delz = jnp.full((km,), -constants.R_d * T * 1000.0 / (constants.g * 1.0e5))
    theta_e = eqv_pot_fv3(pt, delp, q, delz=delz, moist=False)
    assert jnp.all(jnp.abs(theta_e - T) < 1e-10 * T)


def test_eqv_pot_dry_poisson_form():
    """Dry θ_e at known pd:  θ_e = T·(1e5/pd)^kappa."""
    km = 5
    T = 250.0
    pt = jnp.full((km,), T)
    q = jnp.zeros((km,))
    delp = jnp.full((km,), 2000.0)
    p_target = 5.0e4  # 500 hPa
    delz = jnp.full((km,), -constants.R_d * T * delp[0] / (constants.g * p_target))
    theta_e = eqv_pot_fv3(pt, delp, q, delz=delz, moist=False)
    expected = T * (1.0e5 / p_target) ** constants.kappa
    assert jnp.all(jnp.abs(theta_e - expected) / expected < 1e-10)


def test_eqv_pot_moist_gt_dry():
    """Moist θ_e > dry θ_e for q > 0 (latent heat boost)."""
    km = 5
    T = 290.0
    pt = jnp.full((km,), T)
    q_moist = jnp.full((km,), 0.015)  # 15 g/kg, typical tropical
    q_dry = jnp.zeros((km,))
    delp = jnp.full((km,), 1000.0)
    delz = jnp.full((km,), -constants.R_d * T * delp[0] / (constants.g * 1.0e5))
    theta_e_moist = eqv_pot_fv3(pt, delp, q_moist, delz=delz, moist=True)
    theta_e_dry = eqv_pot_fv3(pt, delp, q_dry, delz=delz, moist=False)
    assert jnp.all(theta_e_moist > theta_e_dry)
    # ~ 40-50 K higher for tropical conditions
    assert jnp.all(theta_e_moist - theta_e_dry > 20.0)


def test_eqv_pot_hydrostatic_branch():
    """Hydrostatic branch via peln (Δpeln = ln(p[k+1]/p[k]))."""
    km = 5
    T = 280.0
    pt = jnp.full((km,), T)
    q = jnp.zeros((km,))
    # Monotone increasing pe (top → bottom).  5 layers.
    pe = jnp.array([5.0e4, 6.0e4, 7.0e4, 8.0e4, 9.0e4, 1.0e5])
    peln = jnp.log(pe)
    delp = pe[1:] - pe[:-1]
    theta_e = eqv_pot_fv3(pt, delp, q, peln=peln, hydrostatic=True, moist=False)
    # pd ≈ delp/dpeln ≈ p_mean.  Layer-pd varies per layer.
    assert jnp.all(jnp.isfinite(theta_e))
    assert jnp.all(theta_e > 0.0)


def test_eqv_pot_shapes_3d():
    """3-D input → 3-D output."""
    rng = np.random.default_rng(seed=692)
    n_x, n_y, km = 4, 5, 30
    pt = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    delz = jnp.full((n_x, n_y, km), -200.0)
    theta_e = eqv_pot_fv3(pt, delp, q, delz=delz, moist=True)
    assert theta_e.shape == (n_x, n_y, km)


def test_eqv_pot_finite():
    """No NaN/Inf on random inputs."""
    rng = np.random.default_rng(seed=693)
    km = 30
    pt = jnp.asarray(rng.uniform(220.0, 320.0, size=(4, 4, km)))
    q = jnp.asarray(rng.uniform(1e-6, 0.025, size=(4, 4, km)))
    delp = jnp.asarray(rng.uniform(500, 2000, size=(4, 4, km)))
    delz = jnp.asarray(rng.uniform(-500, -100, size=(4, 4, km)))
    theta_e = eqv_pot_fv3(pt, delp, q, delz=delz, moist=True)
    assert jnp.all(jnp.isfinite(theta_e))


def test_eqv_pot_missing_arg_raises():
    """Missing peln in hydrostatic mode raises."""
    km = 5
    pt = jnp.full((km,), 280.0)
    q = jnp.zeros((km,))
    delp = jnp.full((km,), 1000.0)
    with pytest.raises(ValueError):
        eqv_pot_fv3(pt, delp, q, hydrostatic=True)
    with pytest.raises(ValueError):
        eqv_pot_fv3(pt, delp, q, hydrostatic=False)
