"""c2l_ord4_face: vectorized authority vs an independent Fortran-loop
transcription (fv_grid_utils.F90:2407-2531, bounded_domain branch), plus a
non-vacuity check (a shifted stencil must disagree) and the ng>=2 guard.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.grids.fv3_native_ext_vector import c2l_ord4_face


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
