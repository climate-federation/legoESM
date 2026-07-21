"""Regression: the DYNAMIC depth-dependent reference-density branch of
``iterate_eos_and_pressure_anomaly`` reads ``mask``.

Bug (Prove-It): the helper opened with ``del mask`` (the arg was believed
unused), but when ``use_depth_dependent_ref=True`` AND ``is_active_3d`` is
None the fallback wet-cell weight is built from ``mask``:

    wet = jnp.broadcast_to(mask[..., None].astype(rho.dtype), rho.shape)

so that branch raised ``UnboundLocalError: local variable 'mask' referenced
before assignment``.  This test drives exactly that branch and asserts a
finite ``p'`` comes back.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)

_G = constants.g
_RHO0 = 1026.0


def _run(*, use_depth_dependent_ref, is_active_3d):
    rng = np.random.default_rng(0)
    nlev = 8
    T = jnp.asarray(_RHO0 + rng.normal(size=(3, 4, nlev)))   # rho == T
    S = jnp.zeros_like(T)
    mask = jnp.ones(T.shape[:-1])
    dz = jnp.full(nlev, 55.0)
    return iterate_eos_and_pressure_anomaly(
        T, S, mask, lambda f: f,
        lambda T_, S_, p_: T_,                    # identity EOS: rho == T
        dz, _RHO0, _G,
        use_depth_dependent_ref=use_depth_dependent_ref,
        is_active_3d=is_active_3d,
    )


def test_dynamic_ref_no_active_mask_uses_mask_arg():
    """use_depth_dependent_ref=True, is_active_3d=None -> mask fallback branch.

    Before the fix this raised UnboundLocalError (mask was ``del``-eted).
    """
    rho, rho_prime, p_prime = _run(
        use_depth_dependent_ref=True, is_active_3d=None)
    assert bool(jnp.all(jnp.isfinite(p_prime)))
    # dynamic ref subtracts the per-level wet-cell mean, so rho' has ~zero
    # horizontal mean at each level (mask all-wet here).
    horiz_mean = jnp.mean(rho_prime, axis=(0, 1))
    np.testing.assert_allclose(np.asarray(horiz_mean), 0.0, atol=1e-10)


def test_dynamic_ref_with_active_3d_still_finite():
    """The is_active_3d path (mask NOT used) stays finite — guards against a
    fix that broke the sibling branch."""
    active = jnp.ones((3, 4, 8), bool)
    _, _, p_prime = _run(
        use_depth_dependent_ref=True, is_active_3d=active)
    assert bool(jnp.all(jnp.isfinite(p_prime)))
