"""FV3_3D iter 295: linear-in-|dt| scaling and sign symmetry of
the iter-187 smag_vort cap formula.

The iter-187 smag_vort cap (FV3 sw_core.F90:1795-1809) is::

    smag_vort = |dt| * sqrt(delpc² + ζ²)

This has two basic invariants by construction:

* **Linear in |dt|**: scaling dt → α*dt produces |α| × smag_vort.
* **Sign-flip symmetric**: smag_vort(-delpc, -ζ) = smag_vort(delpc, ζ)
  since the inputs enter only via squares.

iter-285 pinned non-negativity, sqrt(0) AD-safety, and rtol=1e-12
agreement with naive sqrt at non-rest.  iter-295 fills the
orthogonal scaling and sign-symmetry slots.

Tests
-----

1. ``test_smag_vort_linear_in_dt`` — α*dt → |α| × smag_vort.
2. ``test_smag_vort_sign_symmetric`` — smag_vort(-delpc, -ζ)
   = smag_vort(delpc, ζ).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _safe_smag_vort(delpc, zeta, dt):
    """Reproduce the iter-183 AD-safe form (mirror of the
    in-source pattern at compressible_euler_cdgrid.py:692-697).
    """
    smag_arg = delpc ** 2 + zeta ** 2
    safe_arg = jnp.where(smag_arg > 0.0, smag_arg, 1.0)
    smag_root = jnp.where(
        smag_arg > 0.0, jnp.sqrt(safe_arg), 0.0,
    )
    return jnp.abs(dt) * smag_root


@pytest.mark.parametrize("alpha", [-2.5, 0.5, 1.0, 3.0])
def test_smag_vort_linear_in_dt(alpha):
    """smag_vort(α*dt) = |α| * smag_vort(dt) bit-for-bit."""
    rng = np.random.default_rng(seed=295)
    delpc = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(6, 9, 9, 5)))
    zeta = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(6, 9, 9, 5)))
    dt = 10.0

    smag_base = _safe_smag_vort(delpc, zeta, dt)
    smag_alpha = _safe_smag_vort(delpc, zeta, alpha * dt)

    np.testing.assert_allclose(
        np.asarray(smag_alpha), abs(alpha) * np.asarray(smag_base),
        rtol=1e-14,
        err_msg=(
            f"smag_vort must scale linearly with |dt|; alpha="
            f"{alpha} should give |alpha| * smag_vort exactly."
        ),
    )


def test_smag_vort_sign_symmetric():
    """smag_vort(-delpc, -ζ) = smag_vort(delpc, ζ).

    Both inputs enter only via squared terms in the radicand.
    """
    rng = np.random.default_rng(seed=295)
    delpc = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(6, 9, 9, 5)))
    zeta = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=(6, 9, 9, 5)))
    dt = 10.0

    smag_pos = _safe_smag_vort(delpc, zeta, dt)
    smag_neg = _safe_smag_vort(-delpc, -zeta, dt)
    np.testing.assert_array_equal(
        np.asarray(smag_neg), np.asarray(smag_pos),
    )

    # Mixed signs (only one flipped) also invariant — squares.
    smag_mix1 = _safe_smag_vort(-delpc, zeta, dt)
    smag_mix2 = _safe_smag_vort(delpc, -zeta, dt)
    np.testing.assert_array_equal(
        np.asarray(smag_mix1), np.asarray(smag_pos),
    )
    np.testing.assert_array_equal(
        np.asarray(smag_mix2), np.asarray(smag_pos),
    )
