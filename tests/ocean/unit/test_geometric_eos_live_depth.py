"""Live z* depth for the geometric EOS path (#1226).

NEMO's ``eos_insitu`` evaluates at ``zh = gdept(ji,jj,jk,Knn)`` — the LIVE
``gdept_0*(1+r3t)``, ``r3t = ssh/ht_0`` (eosbn2.F90:541).  Feeding the STATIC
reference ladder omitted a stretch of up to 1.206 m on the DINO twin and put a
depth-structured 2.559e-6 into the density anomaly ``prd``, which was the
residual floor under all four ``ldf_slp`` slope rows.

These tests pin the three behaviours that adversarial review identified as the
risky part of the fix — the GATING, not the physics:

  * ``linear_free_surface`` (NEMO ``key_linssh``): the column NEVER stretches,
    so the live ladder must equal the static one exactly;
  * no ``t_depth_ref`` (every non-NEMO-bridged recipe): bit-identical to the
    pre-fix behaviour, so the change cannot perturb unrelated configs;
  * a fidelity ladder present: the stretch IS applied.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.eos import nemo_bn2_depth_ladders, nemo_bn2_live_ladders
from legoesm.ocean.vertical import create_z_star_from_thicknesses


def _z_coord(nlev=6, dz=100.0):
    dz_ref = jnp.full((nlev,), dz)
    gdept = jnp.cumsum(dz_ref) - 0.5 * dz_ref
    return create_z_star_from_thicknesses(dz_ref, t_depth_ref_m=gdept)


def test_live_ladder_applies_the_z_star_stretch():
    z = _z_coord()
    H = jnp.asarray([[600.0]])
    eta = jnp.asarray([[6.0]])                    # r3t = 0.01
    static, _ = nemo_bn2_depth_ladders(z)
    live, _ = nemo_bn2_live_ladders(z, eta, H)
    np.testing.assert_allclose(
        np.asarray(live)[0, 0], np.asarray(static) * 1.01, rtol=1e-12)


def test_linear_free_surface_column_never_stretches():
    """NEMO key_linssh: domqco inactive => r3t == 0 => gdept(Kmm) == gdept_0.

    Regression for adversarial-review finding F1: the live helper applied the
    stretch unconditionally, which would give the density an eta dependence
    NEMO does not have. compute_ocean_jacobian already special-cases this flag.
    """
    z = _z_coord()._replace(linear_free_surface=True)
    H = jnp.asarray([[600.0]])
    static, static_w = nemo_bn2_depth_ladders(z)
    live, live_w = nemo_bn2_live_ladders(z, jnp.asarray([[6.0]]), H)
    np.testing.assert_array_equal(np.asarray(live), np.asarray(static))
    np.testing.assert_array_equal(np.asarray(live_w), np.asarray(static_w))


def test_dry_column_is_inert_and_gradients_stay_finite():
    """H_bathy == 0 must not produce inf/NaN forward OR in the backward pass."""
    z = _z_coord()
    H = jnp.asarray([[0.0, 600.0]])
    eta = jnp.asarray([[1.0, 6.0]])
    live, _ = nemo_bn2_live_ladders(z, eta, H)
    assert np.all(np.isfinite(np.asarray(live)))

    def loss(e, h):
        return jnp.sum(nemo_bn2_live_ladders(z, e, h)[0] ** 2)

    g_eta, g_H = jax.grad(loss, argnums=(0, 1))(eta, H)
    assert np.all(np.isfinite(np.asarray(g_eta)))
    assert np.all(np.isfinite(np.asarray(g_H)))
    assert float(np.asarray(g_eta)[0, 0]) == 0.0      # dry column inert
    assert float(np.asarray(g_H)[0, 0]) == 0.0


def test_stretch_floor_keeps_depth_positive():
    """eta + H <= 0 would give a NON-POSITIVE geometric depth.

    That silently flips the sign of the S-EOS compressibility term, so the
    helper floors the stretch. Unclamped callers degrade loudly-wrong rather
    than silently-plausible.
    """
    z = _z_coord()
    live, _ = nemo_bn2_live_ladders(
        z, jnp.asarray([[-700.0]]), jnp.asarray([[600.0]]))
    assert float(np.asarray(live).min()) > 0.0


def test_gm_redi_density_is_bit_identical_without_a_fidelity_ladder():
    """No t_depth_ref => the pre-fix static behaviour, EXACTLY.

    Regression for adversarial-review finding F2: the fix must not perturb the
    ~every non-NEMO-bridged recipe, and must not put the PGF and GM/Redi on
    different EOS depths within one timestep.
    """
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        gm_redi_density_and_jacobian,
    )
    from legoesm.ocean.vertical import create_ocean_z_star

    nlev = 5
    z = create_ocean_z_star(n_levels=nlev, H_max=500.0)
    assert getattr(z, "t_depth_ref", None) is None      # the gated case
    shape = (2, 3, nlev)
    T = jnp.asarray(np.linspace(20.0, 4.0, nlev))[None, None, :] * jnp.ones(shape)
    S = jnp.full(shape, 35.0)
    H = jnp.full((2, 3), 500.0)
    mask = jnp.ones((2, 3))

    rho_a, _ = gm_redi_density_and_jacobian(
        T, S, jnp.zeros((2, 3)), H, None, z, eos="nemo_seos",
        eos_linear=None, mask=mask, eos_depth="geometric")
    # a LARGE eta must not move the density at all on this coordinate
    rho_b, _ = gm_redi_density_and_jacobian(
        T, S, jnp.full((2, 3), 25.0), H, None, z, eos="nemo_seos",
        eos_linear=None, mask=mask, eos_depth="geometric")
    np.testing.assert_array_equal(np.asarray(rho_a), np.asarray(rho_b))


def test_gm_redi_density_does_move_with_eta_when_a_ladder_is_carried():
    """The complement: with a fidelity ladder the live stretch IS applied.

    Without this the previous test could pass vacuously (e.g. if eta never
    reached the EOS at all).
    """
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        gm_redi_density_and_jacobian,
    )
    nlev = 6
    z = _z_coord(nlev=nlev)
    shape = (1, 1, nlev)
    T = jnp.asarray(np.linspace(20.0, 4.0, nlev))[None, None, :]
    S = jnp.full(shape, 35.0)
    H = jnp.asarray([[600.0]])
    mask = jnp.ones((1, 1))
    kw = dict(eos="nemo_seos", eos_linear=None, mask=mask,
              eos_depth="geometric")
    rho_0eta, _ = gm_redi_density_and_jacobian(
        T, S, jnp.zeros((1, 1)), H, None, z, **kw)
    rho_eta, _ = gm_redi_density_and_jacobian(
        T, S, jnp.asarray([[6.0]]), H, None, z, **kw)
    assert float(np.abs(np.asarray(rho_eta) - np.asarray(rho_0eta)).max()) > 0.0
