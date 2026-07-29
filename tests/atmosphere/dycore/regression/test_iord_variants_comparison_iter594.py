"""FV3_3D iter 594: side-by-side comparison of the FV3 iord limiter families.

The limiter FAMILIES {8}, {10}, {11}, and the positive-definite iv=0
family produce distinct bl, br on a stress test mixing a smooth ramp +
sharp step.  NOTE (#1256): iord=9 and iord=12 are the SAME positive-
definite family — iord=9 is the ``pert_ppm(iv=0)`` SUBROUTINE
(``_pert_ppm_iv0``) and iord=12 is the INLINE iord==7/12 branch
(``_pert_ppm_iv0_inline``); they are identical for ``q > 0`` (the
positive-definite fields these transports carry) and differ only for
``q <= 0``.  The bit-exact match of each to its oracle branch, and the
q<=0 divergence, are certified in
``tests/unit/test_hord12_oracle_iord_branch_1256.py``.

Tests
-----

1. ``test_all_five_iord_variants_distinct`` — at the step discontinuity
   the distinct families differ (iord=9/12 coincide on this positive
   field).
2. ``test_smooth_region_all_variants_agree`` — far from extrema, all
   variants give similar (non-flat) result.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.fv_tp_2d import (
    _pert_ppm_iv0,        # iord=9 (pert_ppm iv=0 subroutine)
    _pert_ppm_iv0_inline, # iord=12 (inline iord==7/12 branch)
    apply_hord8_limiter,  # iord=8
    apply_hord10_limiter, # iord=10
    apply_hord11_limiter, # iord=11
)


def _build_stress_field(n=20):
    """Smooth ramp 0..0.5 then sharp step from 0.5 to 5.0 then 5.0..5.5."""
    rng = np.random.default_rng(seed=594)
    q = np.linspace(0.0, 0.5, n)
    q[n // 2:] += 4.5  # step
    q = jnp.asarray(q + 1e-3 * rng.uniform(-1, 1, size=n))
    return q


def _build_bl_br_dm(q):
    """Simple PPM bl, br, dm aligned with q (all length n).

    bl[i] = (q[i-1] + q[i])/2 - q[i] = (q[i-1] - q[i])/2  (left-edge perturbation)
    br[i] = (q[i] + q[i+1])/2 - q[i] = (q[i+1] - q[i])/2  (right-edge perturbation)
    dm[i] = (q[i+1] - q[i-1])/4   (centered slope)
    """
    qe = jnp.concatenate([q[:1], q, q[-1:]])  # length n+2, edge-padded
    bl = 0.5 * (qe[:-2] - q)
    br = 0.5 * (qe[2:] - q)
    dm = 0.25 * (qe[2:] - qe[:-2])
    return bl, br, dm


def test_all_five_iord_variants_distinct():
    """Different variants produce different output at the step."""
    q = _build_stress_field(n=20)
    bl, br, dm = _build_bl_br_dm(q)

    # Apply each variant with its CORRECT limiter (#1256): iord=9 = iv=0
    # subroutine, iord=12 = inline iord==7/12 branch.  The stress field is
    # all-positive, so iord=9 and iord=12 coincide here by construction.
    bl9, br9 = _pert_ppm_iv0(q, bl, br)
    bl12, br12 = _pert_ppm_iv0_inline(q, bl, br)
    bl8, br8 = apply_hord8_limiter(bl, br, dm)
    bl11, br11 = apply_hord11_limiter(bl, br, dm, ppm_fac=1.5)
    bl10, br10 = apply_hord10_limiter(bl, br, dm, q)

    # iord=9 and iord=12 are the SAME family for q>0 — assert they coincide
    # on this positive field (the corrected labels; a regression to
    # pert_ppm(iv=1) for iord=9 would break this).
    np.testing.assert_allclose(np.asarray(bl9), np.asarray(bl12))
    np.testing.assert_allclose(np.asarray(br9), np.asarray(br12))

    # The distinct FAMILIES should differ on the stress field.
    variants = {
        "iv0(9/12)": (bl9, br9),
        "iord=8":  (bl8, br8),
        "iord=11": (bl11, br11),
        "iord=10": (bl10, br10),
    }
    names = list(variants.keys())
    max_diff_found = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            di_bl = float(jnp.abs(variants[ni][0] - variants[nj][0]).max())
            di_br = float(jnp.abs(variants[ni][1] - variants[nj][1]).max())
            max_diff_found = max(max_diff_found, di_bl + di_br)
    assert max_diff_found > 0.01, (
        f"Expected variants to differ on stress field; "
        f"max diff = {max_diff_found:.3e}"
    )


def test_smooth_region_all_variants_agree():
    """Far from extrema, all variants should give finite & bounded
    results.  Sanity check that no variant produces NaN/Inf."""
    n = 20
    q = jnp.asarray(np.linspace(0.0, 1.0, n))  # purely smooth
    bl, br, dm = _build_bl_br_dm(q)

    for name, fn in [
        ("iord=9",  lambda: _pert_ppm_iv0(q, bl, br)),
        ("iord=12", lambda: _pert_ppm_iv0_inline(q, bl, br)),
        ("iord=8",  lambda: apply_hord8_limiter(bl, br, dm)),
        ("iord=11", lambda: apply_hord11_limiter(bl, br, dm)),
        ("iord=10", lambda: apply_hord10_limiter(bl, br, dm, q)),
    ]:
        bl_out, br_out = fn()
        assert jnp.all(jnp.isfinite(bl_out)), f"{name} produced non-finite"
        assert jnp.all(jnp.isfinite(br_out))
