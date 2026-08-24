"""c2l_ord4_face: vectorized authority vs an independent Fortran-loop
transcription (fv_grid_utils.F90:2407-2531, bounded_domain branch), plus a
non-vacuity check (a shifted stencil must disagree) and the ng>=2 guard.
The JAX twin (``c2l_ord4_face_jax``) is gated here too: jax==numpy on a random
a-matrix (same amat both lanes -- a np->jnp transcription check, not a physics
check), jit==eager, the NaN-outside-compute contract, and differentiability
w.r.t. the winds.
"""
from __future__ import annotations

import numpy as np
import pytest

# the FV3 duo lane is fp64; enable x64 explicitly so the JAX-twin numerical
# identity is tested at the production precision, not JAX's default float32.
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

from legoesm.grids.fv3_native_ext_vector import (  # noqa: E402
    c2l_ord4_face,
    c2l_ord4_face_jax,
)


def _fortran_loop_ref(u, v, amat, n, ng):
    """Independent triple-loop transcription of the bounded_domain branch, one
    level; numpy index = fortran index - lo, lo = 1 - ng."""
    a11, a12, a21, a22 = amat
    c1, c2 = 1.125, -0.125
    lo = 1 - ng
    m_a = n + 2 * ng
    npx, npy = n + 1, n + 1
    is_, ie, js, je = 1, n, 1, n
    utmp = np.full((m_a, m_a), np.nan)
    vtmp = np.full((m_a, m_a), np.nan)
    for j in range(max(1, js), min(npy - 1, je) + 1):
        for i in range(max(1, is_), min(npx - 1, ie) + 1):
            utmp[i - lo, j - lo] = (c2 * (u[i - lo, j - 1 - lo] + u[i - lo, j + 2 - lo])
                                    + c1 * (u[i - lo, j - lo] + u[i - lo, j + 1 - lo]))
            vtmp[i - lo, j - lo] = (c2 * (v[i - 1 - lo, j - lo] + v[i + 2 - lo, j - lo])
                                    + c1 * (v[i - lo, j - lo] + v[i + 1 - lo, j - lo]))
    ua = np.full((m_a, m_a), np.nan)
    va = np.full((m_a, m_a), np.nan)
    for j in range(js, je + 1):
        for i in range(is_, ie + 1):
            ua[i - lo, j - lo] = (a11[i - lo, j - lo] * utmp[i - lo, j - lo]
                                  + a12[i - lo, j - lo] * vtmp[i - lo, j - lo])
            va[i - lo, j - lo] = (a21[i - lo, j - lo] * utmp[i - lo, j - lo]
                                  + a22[i - lo, j - lo] * vtmp[i - lo, j - lo])
    return ua, va


def _shifted_stencil_variant(u, v, amat, n, ng):
    """Deliberately wrong vectorized port: stencil shifted +1 (j+1..j+4)."""
    a11, a12, a21, a22 = amat
    c1, c2 = 1.125, -0.125
    s, e = ng, ng + n
    cs = slice(s, e)
    utmp = (c2 * (u[cs, s + 1:e + 1] + u[cs, s + 4:e + 4])
            + c1 * (u[cs, s + 2:e + 2] + u[cs, s + 3:e + 3]))
    vtmp = (c2 * (v[s + 1:e + 1, cs] + v[s + 4:e + 4, cs])
            + c1 * (v[s + 2:e + 2, cs] + v[s + 3:e + 3, cs]))
    ua = np.full((n + 2 * ng, n + 2 * ng), np.nan)
    va = np.full((n + 2 * ng, n + 2 * ng), np.nan)
    ua[cs, cs] = a11[cs, cs] * utmp + a12[cs, cs] * vtmp
    va[cs, cs] = a21[cs, cs] * utmp + a22[cs, cs] * vtmp
    return ua, va


def test_c2l_ord4_face_matches_fortran_loops():
    n, ng = 8, 3
    m_a, m_b = n + 2 * ng, n + 2 * ng + 1
    rng = np.random.default_rng(42)
    u = rng.standard_normal((m_a, m_b))
    v = rng.standard_normal((m_b, m_a))
    amat = tuple(rng.standard_normal((m_a, m_a)) for _ in range(4))

    ua, va = c2l_ord4_face(u, v, amat, n, ng)
    ua_ref, va_ref = _fortran_loop_ref(u, v, amat, n, ng)

    cs = slice(ng, ng + n)
    np.testing.assert_allclose(ua[cs, cs], ua_ref[cs, cs], rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(va[cs, cs], va_ref[cs, cs], rtol=0.0, atol=1e-13)

    # everything outside the compute domain is NaN
    outside = np.ones((m_a, m_a), dtype=bool)
    outside[cs, cs] = False
    assert np.isnan(ua[outside]).all()
    assert np.isnan(va[outside]).all()

    # non-vacuity: the shifted stencil must disagree with the loop reference
    ua_bad, va_bad = _shifted_stencil_variant(u, v, amat, n, ng)
    assert np.abs(ua_bad[cs, cs] - ua_ref[cs, cs]).max() > 1e-6
    assert np.abs(va_bad[cs, cs] - va_ref[cs, cs]).max() > 1e-6


def test_c2l_ord4_face_requires_ng_ge_2():
    n, ng = 8, 1
    m_a, m_b = n + 2 * ng, n + 2 * ng + 1
    rng = np.random.default_rng(0)
    u = rng.standard_normal((m_a, m_b))
    v = rng.standard_normal((m_b, m_a))
    amat = tuple(rng.standard_normal((m_a, m_a)) for _ in range(4))
    with pytest.raises(ValueError):
        c2l_ord4_face(u, v, amat, n, ng)


# ---- JAX twin: jax==numpy, jit==eager, differentiable ----
# Random a-matrix + winds, exactly like the NumPy-authority tests above: this is
# a np->jnp transcription-parity check, so both lanes get the SAME amat and it
# only needs to exercise all four (a11..a22) terms -- the production gridstruct
# builder is not usable at this small n (west halo ng+1=4 unsupported at n=8),
# and a "real" metric adds nothing a random one does not for a parity test.

N4, NG4 = 8, 3
M_A, M_B = N4 + 2 * NG4, N4 + 2 * NG4 + 1


@pytest.fixture(scope="module")
def real_amat_case():
    """Random a-matrix (M_A, M_A) x4 and random D-grid winds, seeded."""
    rng = np.random.default_rng(7)
    amat = tuple(rng.standard_normal((M_A, M_A)) for _ in range(4))
    u = rng.standard_normal((M_A, M_B))
    v = rng.standard_normal((M_B, M_A))
    return u, v, amat


def test_c2l_ord4_face_jax_matches_numpy_and_jit_equals_eager(real_amat_case):
    import jax
    u, v, amat = real_amat_case
    ua_np, va_np = c2l_ord4_face(u, v, amat, N4, NG4)   # fresh authority call
    ua_j, va_j = c2l_ord4_face_jax(u, v, amat, N4, NG4)
    assert np.asarray(ua_j).dtype == np.float64, "jax twin demoted to float32"
    # full-array compare (assert_allclose counts NaN==NaN as equal), then pin
    # that the NaNs sit EXACTLY outside the compute block -- the consumer
    # contract -- not merely at matching spots.
    np.testing.assert_allclose(np.asarray(ua_j), ua_np, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(va_j), va_np, rtol=0, atol=1e-12)
    cs = slice(NG4, NG4 + N4)
    outside = np.ones((M_A, M_A), dtype=bool)
    outside[cs, cs] = False
    assert np.isnan(np.asarray(ua_j)[outside]).all()
    assert np.isnan(np.asarray(va_j)[outside]).all()
    assert np.isfinite(np.asarray(ua_j)[cs, cs]).all()
    assert np.isfinite(np.asarray(va_j)[cs, cs]).all()
    f = jax.jit(c2l_ord4_face_jax, static_argnums=(3, 4))
    ua_t, va_t = f(u, v, amat, N4, NG4)
    np.testing.assert_allclose(np.asarray(ua_t), np.asarray(ua_j), rtol=0,
                               atol=1e-12)
    np.testing.assert_allclose(np.asarray(va_t), np.asarray(va_j), rtol=0,
                               atol=1e-12)


def test_c2l_ord4_face_jax_requires_ng_ge_2():
    n, ng = 8, 1
    m_a, m_b = n + 2 * ng, n + 2 * ng + 1
    rng = np.random.default_rng(0)
    u = rng.standard_normal((m_a, m_b))
    v = rng.standard_normal((m_b, m_a))
    amat = tuple(rng.standard_normal((m_a, m_a)) for _ in range(4))
    with pytest.raises(ValueError):
        c2l_ord4_face_jax(u, v, amat, n, ng)


def test_c2l_ord4_face_jax_differentiable_wrt_winds(real_amat_case):
    import jax.numpy as jnp
    from tests.grids.fv3_gate_helpers import (
        assert_fd_gap_at_roundoff_floor,
        gated_check_grads,
    )
    u, v, amat = real_amat_case
    cs = slice(NG4, NG4 + N4)

    def f(uu, vv):
        ua, va = c2l_ord4_face_jax(uu, vv, amat, N4, NG4)
        # compute block only: outside is NaN BY CONTRACT -- values, not
        # derivatives (the NaN base is a constant), so slicing it away keeps
        # both the FD side and the cotangents finite.
        return ua[cs, cs], va[cs, cs]

    args = (jnp.asarray(u), jnp.asarray(v))
    # LINEAR in (u, v) -- the a-matrix is a fixed coefficient field -- so the
    # eps^2 truncation ladder has no signal on these operands (its
    # precondition would refuse) and the affine-case instrument is the
    # roundoff-floor gate: still FD-vs-VJP, still independent of the AD.
    # margin/atol are provisional headroom (>=10x the paper-derived roundoff);
    # the gate passes empirically, and the floor helper has no measure mode to
    # pin them tighter, so they stay as conservative headroom.
    gated_check_grads("c2l_ord4_face_jax u/v", f, args, order=2,
                      modes=("fwd", "rev"), eps=1e-3, atol=1e-11, rtol=1e-6)
    assert_fd_gap_at_roundoff_floor("c2l_ord4_face_jax u/v", f, args,
                                    margin=10.0, eps=1e-3)
