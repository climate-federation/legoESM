"""#1256: certify hord=12 against the ACTUAL oracle iord=12 branch.

The audit (#1256) suspected the ``hord==12 -> _pert_ppm_iv0`` dispatch was
mislabelled.  Resolved by reading the on-repo Zenodo duo extract
``scripts/validate/fv3_native/fv3_tpcore_duo_extract.F90``:

* iord==9/13 (extract line 590) run the bare ``bl=al-q``/``br=al_R-q`` and
  then call the ``pert_ppm`` SUBROUTINE with ``iv=0`` (extract:1233-1291),
  which flattens ``bl=br=0`` whenever ``a0<=0`` before the extremum clip.
  ``_pert_ppm_iv0`` transcribes that subroutine — so hord=9 is faithful.
* iord==7/12 (extract:560-583) is a SEPARATE INLINE positive-definite
  branch that shares the extremum clip but has NO ``q<=0 -> flatten`` (and
  no ``q>0`` gate).  ``_pert_ppm_iv0_inline`` transcribes that branch — so
  hord=12 must use it, NOT the subroutine.

The two coincide for ``q > 0`` (the positive-definite fields these
transports carry) and differ only for ``q <= 0``; using the subroutine
form for hord=12 was therefore a real-but-practically-inert infidelity.
These tests are the oracle certificate: each Python limiter must match a
direct transcription of its Fortran branch bit-for-bit (including q<=0),
and the two must provably diverge somewhere on q<=0 (non-vacuous).

Run with JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.fv_tp_2d import _pert_ppm_iv0, _pert_ppm_iv0_inline

_R12 = 1.0 / 12.0


def _oracle_subroutine_iv0(q, bl, br):
    """Direct transcription of pert_ppm(iv=0), extract:1233-1291
    (the iord==9/13 call).  Element-wise numpy on (q, bl, br)."""
    q = np.asarray(q, float)
    al = np.asarray(bl, float).copy()   # Fortran names bl->al, br->ar
    ar = np.asarray(br, float).copy()
    out_al = al.copy()
    out_ar = ar.copy()
    for i in range(q.size):
        if q.flat[i] <= 0.0:
            out_al.flat[i] = 0.0
            out_ar.flat[i] = 0.0
        else:
            a4 = -3.0 * (ar.flat[i] + al.flat[i])
            da1 = ar.flat[i] - al.flat[i]
            if abs(da1) < -a4:
                fmin = q.flat[i] + 0.25 / a4 * da1 ** 2 + a4 * _R12
                if fmin < 0.0:
                    if ar.flat[i] > 0.0 and al.flat[i] > 0.0:
                        out_ar.flat[i] = 0.0
                        out_al.flat[i] = 0.0
                    elif da1 > 0.0:
                        out_ar.flat[i] = -2.0 * al.flat[i]
                    else:
                        out_al.flat[i] = -2.0 * ar.flat[i]
    return out_al, out_ar


def _oracle_inline_iord7_12(q, bl, br):
    """Direct transcription of the inline iord==7/12 branch,
    extract:560-583.  NO q<=0 flatten; NO q>0 gate."""
    q = np.asarray(q, float)
    al = np.asarray(bl, float).copy()
    ar = np.asarray(br, float).copy()
    out_al = al.copy()
    out_ar = ar.copy()
    for i in range(q.size):
        a4 = -3.0 * (al.flat[i] + ar.flat[i])
        da1 = ar.flat[i] - al.flat[i]
        ext5 = ar.flat[i] * al.flat[i] > 0.0
        ext6 = abs(da1) < -a4
        if ext6:
            if q.flat[i] + 0.25 / a4 * da1 ** 2 + a4 * _R12 < 0.0:
                if ext5:
                    out_ar.flat[i] = 0.0
                    out_al.flat[i] = 0.0
                elif da1 > 0.0:
                    out_ar.flat[i] = -2.0 * al.flat[i]
                else:
                    out_al.flat[i] = -2.0 * ar.flat[i]
    return out_al, out_ar


def _random_inputs(seed, n=4000):
    rng = np.random.default_rng(seed)
    # bl/br spanning the extremum-clip regimes; q spanning BOTH signs so the
    # q<=0 divergence is exercised.
    bl = rng.uniform(-2.0, 2.0, n)
    br = rng.uniform(-2.0, 2.0, n)
    q = rng.uniform(-1.0, 3.0, n)
    return q, bl, br


def test_inline_matches_oracle_iord7_12_branch():
    q, bl, br = _random_inputs(1256)
    bl_o, br_o = _oracle_inline_iord7_12(q, bl, br)
    bl_p, br_p = _pert_ppm_iv0_inline(jnp.asarray(q), jnp.asarray(bl),
                                      jnp.asarray(br))
    np.testing.assert_allclose(np.asarray(bl_p), bl_o, rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(br_p), br_o, rtol=0, atol=1e-13)


def test_subroutine_matches_oracle_pert_ppm_iv0():
    q, bl, br = _random_inputs(9013)
    bl_o, br_o = _oracle_subroutine_iv0(q, bl, br)
    bl_p, br_p = _pert_ppm_iv0(jnp.asarray(q), jnp.asarray(bl),
                               jnp.asarray(br))
    np.testing.assert_allclose(np.asarray(bl_p), bl_o, rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(br_p), br_o, rtol=0, atol=1e-13)


def test_hord9_and_hord12_agree_for_positive_q():
    """The two families are identical wherever q > 0 (the physical regime)."""
    rng = np.random.default_rng(77)
    q = jnp.asarray(rng.uniform(0.05, 3.0, 4000))   # strictly positive
    bl = jnp.asarray(rng.uniform(-2.0, 2.0, 4000))
    br = jnp.asarray(rng.uniform(-2.0, 2.0, 4000))
    bl9, br9 = _pert_ppm_iv0(q, bl, br)
    bl12, br12 = _pert_ppm_iv0_inline(q, bl, br)
    np.testing.assert_allclose(np.asarray(bl9), np.asarray(bl12),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(br9), np.asarray(br12),
                               rtol=0, atol=0)


def test_hord9_and_hord12_diverge_for_nonpositive_q():
    """Non-vacuous: they MUST differ somewhere on q <= 0 (else the #1256
    fix would be a no-op and the subroutine form would have been fine)."""
    # Construct a case with q<=0 AND a clipped extremum, where the
    # subroutine flattens to 0 but the inline branch keeps a -2x fix.
    q = jnp.asarray([-0.5, 0.0, -1.0, -0.2])
    bl = jnp.asarray([0.9, -0.8, 1.2, 0.3])
    br = jnp.asarray([-0.9, 0.8, -1.3, 0.4])
    bl9, br9 = _pert_ppm_iv0(q, bl, br)
    bl12, br12 = _pert_ppm_iv0_inline(q, bl, br)
    # Subroutine zeros every q<=0 cell; inline does not => at least one differs.
    diff = (np.abs(np.asarray(bl9) - np.asarray(bl12))
            + np.abs(np.asarray(br9) - np.asarray(br12))).max()
    assert diff > 1e-6, (
        "hord=9 and hord=12 identical on q<=0 — the inline branch is not "
        "actually distinct from the subroutine (#1256 fix vacuous)")
    # And the subroutine really did flatten the q<=0 cells to zero.
    q_np = np.asarray(q)
    assert np.allclose(np.asarray(bl9)[q_np <= 0.0], 0.0)
    assert np.allclose(np.asarray(br9)[q_np <= 0.0], 0.0)
