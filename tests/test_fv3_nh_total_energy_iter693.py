"""FV3_3D iter 693: nh_total_energy_fv3 port.

Faithful JAX port of FV3 ``nh_total_energy``
(tools/fv_diagnostics.F90:5501-5571).  Vertically-integrated total
energy per column.

Tests
-----

1. ``test_te_isothermal_at_rest_zero_hs``.
2. ``test_te_moist_gt_dry``.
3. ``test_te_kinetic_contribution``.
4. ``test_te_positive_atmosphere``.
5. ``test_te_shapes_3d``.
6. ``test_te_finite``.
7. ``test_te_moist_requires_q``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import nh_total_energy_fv3


def test_te_isothermal_at_rest_zero_hs():
    """Isothermal column at rest, hs=0:
       TE = cv·T·p_s/g + Σ delp/g·phi_avg

    With uniform delz and increasing phiz from 0 at surface, this is
    a positive finite value.  Sanity check value scale.
    """
    km = 30
    T = 250.0
    pt = jnp.full((km,), T)
    delp = jnp.full((km,), 3000.0)         # p_s = 90 kPa
    delz = jnp.full((km,), -300.0)         # 9 km column
    hs = jnp.asarray(0.0)
    ua = jnp.zeros((km,))
    va = jnp.zeros((km,))
    w = jnp.zeros((km,))
    te = nh_total_energy_fv3(ua, va, w, pt, delp, delz, hs, moist_phys=False)
    # Dry-thermal piece dominates: TE ~ cv·T·p_s/g
    cv = constants.c_pd - constants.R_d
    expected_lower = cv * T * 90000.0 / constants.g
    assert float(te) > expected_lower * 0.99
    # +geopotential piece ~ 0.5·g·H·p_s/g = 0.5·H·p_s ~ 0.5·9000·90000 = 4e8
    expected_upper = expected_lower + 0.5 * 9000.0 * 90000.0 * 1.5
    assert float(te) < expected_upper


def test_te_moist_gt_dry():
    """Moist branch with q>0 → TE > dry branch by L_v·Σ delp·q / g."""
    km = 10
    pt = jnp.full((km,), 290.0)
    delp = jnp.full((km,), 5000.0)
    delz = jnp.full((km,), -500.0)
    hs = jnp.asarray(0.0)
    ua = jnp.zeros((km,))
    va = jnp.zeros((km,))
    w = jnp.zeros((km,))
    q = jnp.full((km,), 0.010)             # 10 g/kg
    te_dry = nh_total_energy_fv3(ua, va, w, pt, delp, delz, hs, moist_phys=False)
    te_moist = nh_total_energy_fv3(
        ua, va, w, pt, delp, delz, hs, q_sphum=q, moist_phys=True,
    )
    delta_expected = constants.L_v * float(jnp.sum(delp * q)) / constants.g
    delta_actual = float(te_moist - te_dry)
    assert abs(delta_actual - delta_expected) / delta_expected < 1e-10


def test_te_kinetic_contribution():
    """Compare zero-wind vs uniform-wind: KE adds 0.5·U²·p_s/g."""
    km = 10
    pt = jnp.full((km,), 280.0)
    delp = jnp.full((km,), 5000.0)
    delz = jnp.full((km,), -500.0)
    hs = jnp.asarray(0.0)
    U = 20.0
    te0 = nh_total_energy_fv3(
        jnp.zeros((km,)), jnp.zeros((km,)), jnp.zeros((km,)),
        pt, delp, delz, hs, moist_phys=False,
    )
    te_u = nh_total_energy_fv3(
        jnp.full((km,), U), jnp.zeros((km,)), jnp.zeros((km,)),
        pt, delp, delz, hs, moist_phys=False,
    )
    p_s = 50000.0
    delta_expected = 0.5 * U * U * p_s / constants.g
    delta_actual = float(te_u - te0)
    assert abs(delta_actual - delta_expected) / delta_expected < 1e-10


def test_te_positive_atmosphere():
    """Realistic atmosphere column → TE > 0."""
    km = 30
    pt = jnp.linspace(220.0, 290.0, km)
    delp = jnp.full((km,), 3000.0)
    delz = jnp.full((km,), -300.0)
    hs = jnp.asarray(0.0)
    ua = jnp.full((km,), 10.0)
    va = jnp.full((km,), 5.0)
    w = jnp.zeros((km,))
    q = jnp.full((km,), 0.005)
    te = nh_total_energy_fv3(
        ua, va, w, pt, delp, delz, hs, q_sphum=q, moist_phys=True,
    )
    assert float(te) > 0.0
    # Realistic order: 1e9 J/m^2 for full troposphere
    assert 1e8 < float(te) < 1e10


def test_te_shapes_3d():
    """3-D input → 2-D output."""
    rng = np.random.default_rng(seed=693)
    n_x, n_y, km = 4, 5, 30
    ua = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    w = jnp.asarray(rng.normal(scale=0.1, size=(n_x, n_y, km)))
    pt = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 3000.0)
    delz = jnp.full((n_x, n_y, km), -300.0)
    hs = jnp.asarray(rng.uniform(0, 5000, size=(n_x, n_y)))
    te = nh_total_energy_fv3(ua, va, w, pt, delp, delz, hs, moist_phys=False)
    assert te.shape == (n_x, n_y)


def test_te_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=694)
    km = 30
    pt = jnp.asarray(rng.uniform(220, 300, size=(4, 4, km)))
    delp = jnp.full((4, 4, km), 3000.0)
    delz = jnp.full((4, 4, km), -300.0)
    hs = jnp.zeros((4, 4))
    ua = jnp.asarray(rng.normal(scale=10.0, size=(4, 4, km)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(4, 4, km)))
    w = jnp.zeros((4, 4, km))
    q = jnp.asarray(rng.uniform(1e-6, 0.02, size=(4, 4, km)))
    te = nh_total_energy_fv3(
        ua, va, w, pt, delp, delz, hs, q_sphum=q, moist_phys=True,
    )
    assert jnp.all(jnp.isfinite(te))


def test_te_moist_requires_q():
    """moist_phys=True without q_sphum raises."""
    km = 5
    z = jnp.zeros((km,))
    pt = jnp.full((km,), 280.0)
    delp = jnp.full((km,), 1000.0)
    delz = jnp.full((km,), -300.0)
    hs = jnp.asarray(0.0)
    with pytest.raises(ValueError):
        nh_total_energy_fv3(z, z, z, pt, delp, delz, hs, moist_phys=True)
