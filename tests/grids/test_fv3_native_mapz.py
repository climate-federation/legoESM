"""Direct tests for the ``fv_mapz`` port.

The Fortran oracle cannot be called from here, so these are TRUTH-TIER
checks (identities the remap must satisfy for any correct implementation)
plus structural checks that pin the two places where a plausible-looking
but wrong port would still pass an identity test:

* ``scalar_profile`` and ``cs_profile`` must NOT be interchangeable;
* ``map1_q2`` must divide by the caller's ``dp2``, not by
  ``pe2(k+1)-pe2(k)``.

Every non-vacuity guard below is an assertion that the thing being tested
would FAIL if the feature were removed -- a conservation test on a remap
that is secretly the identity proves nothing.
"""
import numpy as np
import pytest

from legoesm.core.fv3_native_eta import set_eta_analytic
from legoesm.core.fv3_native_mapz import (
    T_MIN,
    cs_limiters,
    cs_profile,
    lagrangian_to_eulerian,
    map1_ppm,
    map1_q2,
    map_scalar,
    pad1,
    ppm_profile_is_unported,
    scalar_profile,
    unpad1,
)

KM = 5
AKAP = 2.0 / 7.0


def _lagrangian_edges(im=4, km=KM, ps=1000.0e2, ptop=500.0e2, deform=0.30):
    """A Lagrangian interface set that is NOT the Eulerian one.

    ``deform`` skews the layer thicknesses column by column so the rezone
    genuinely has to move mass; deform=0 collapses it to the identity and
    every conservation test below becomes vacuous.
    """
    ak, bk, ptop_e, _ = set_eta_analytic(km)
    pe_eul = ak + bk * ps                                   # (km+1,)
    dp_eul = np.diff(pe_eul)
    pe1 = np.zeros((im, km + 2), dtype=np.float64)          # 1-based
    pe2 = np.zeros((im, km + 2), dtype=np.float64)
    for i in range(im):
        w = 1.0 + deform * np.cos(np.arange(km) + 1.7 * i)
        dp = dp_eul * w
        dp *= (ps - ptop) / dp.sum()                        # same column mass
        pe1[i, 1:] = np.concatenate([[ptop], ptop + np.cumsum(dp)])
        pe2[i, 1:] = pe_eul
    assert ptop_e == ptop
    return pe1, pe2, ak, bk


def _column(im=4, km=KM, amp=8.0):
    """A 1-based (im, km+1) field with real vertical structure."""
    q = np.zeros((im, km + 1), dtype=np.float64)
    for i in range(im):
        q[i, 1:] = 250.0 + amp * np.sin(np.arange(km) * 1.3 + 0.4 * i)
    return q


# ---------------------------------------------------------------------- #
# the rezone
# ---------------------------------------------------------------------- #

def test_identity_remap_reproduces_the_input_column():
    """pe2 == pe1 must return the cell means, to rounding.

    This is the one case where the answer is known analytically: with
    pl=0 and pr=1 the parabola mean collapses to ``AL + (AR-AL)/2 + A6/6``
    which is ``a4(1)`` by the definition of A6.
    """
    pe1, _, _, _ = _lagrangian_edges()
    q1 = _column()
    out = map_scalar(pe1, q1, pe1, KM, KM, 1, 9, T_MIN)
    assert np.allclose(out[:, 1:], q1[:, 1:], rtol=0.0, atol=1e-11)


def test_rezone_conserves_the_column_integral():
    pe1, pe2, _, _ = _lagrangian_edges()
    q1 = _column()
    out = map_scalar(pe1, q1, pe2, KM, KM, 1, 9, T_MIN)
    dp1 = pe1[:, 2:] - pe1[:, 1:-1]
    dp2 = pe2[:, 2:] - pe2[:, 1:-1]
    src = (q1[:, 1:] * dp1).sum(axis=1)
    dst = (out[:, 1:] * dp2).sum(axis=1)
    assert np.allclose(dst, src, rtol=2e-14, atol=0.0)


def test_rezone_actually_moves_the_field():
    """Non-vacuity for the two tests above.

    If the deformed and Eulerian grids happened to coincide, or if the
    rezone silently copied its input, the identity and conservation
    checks would both pass while proving nothing.
    """
    pe1, pe2, _, _ = _lagrangian_edges()
    q1 = _column()
    out = map_scalar(pe1, q1, pe2, KM, KM, 1, 9, T_MIN)
    assert np.max(np.abs(out[:, 1:] - q1[:, 1:])) > 1e-3


def _monotone_column(im=4, km=KM):
    """A strictly increasing column -- no interior local extremum."""
    q = np.zeros((im, km + 1), dtype=np.float64)
    for i in range(im):
        q[i, 1:] = 220.0 + 9.0 * np.arange(km) + 0.5 * i
    return q


def test_rezone_overshoot_stays_small_on_a_monotone_column():
    """kord=9 is NOT bound-preserving, not even on monotone data.

    RETRACTED PREMISE: an earlier version asserted the output lay inside
    the source range. It does not, and the reason is in the port's own
    docstring -- ``q(i,1)`` and ``q(i,km+1)`` are NEVER clamped
    (fv_mapz.F90:1949-1984 clamps interfaces 2..km only), so the
    one-sided cubic extrapolation at the top and bottom interfaces
    escapes the data range by design. ``cs_limiters(iv=1)`` then bounds
    the PARABOLA between AL and AR, not AL and AR themselves.

    So the testable property is that the excursion is small, and (below)
    that the column integral is exact. Clipping it would be MORE monotone
    than the oracle and would fail parity.
    """
    pe1, pe2, _, _ = _lagrangian_edges()
    q1 = _monotone_column()
    out = map_scalar(pe1, q1, pe2, KM, KM, 1, 9, T_MIN)
    lo = q1[:, 1:].min(axis=1)[:, None]
    hi = q1[:, 1:].max(axis=1)[:, None]
    span = (hi - lo)
    assert np.all(out[:, 1:] >= lo - 0.05 * span)
    assert np.all(out[:, 1:] <= hi + 0.05 * span)


def test_tracer_lane_iv0_is_positive_definite():
    """iv=0 IS the bound that actually holds: positivity (`:2244-2247`).

    ``cs_limiters(iv=0)`` plus the ``a4(3,km) = max(0, ...)`` bottom rule
    exist precisely because the unconstrained scheme above can undershoot.
    A tracer that went negative here would be a real defect.
    """
    pe1, pe2, _, _ = _lagrangian_edges()
    q1 = np.zeros((4, KM + 1), dtype=np.float64)
    q1[:, 1:] = np.array([1e-6, 8e-3, 2e-5, 9e-3, 1e-7])   # spiky, tiny
    dp2 = np.zeros((4, KM + 1), dtype=np.float64)
    dp2[:, 1:] = pe2[:, 2:] - pe2[:, 1:-1]
    out = map1_q2(pe1, q1, pe2, dp2, KM, KM, 0, 9, 0.0)
    assert np.all(out[:, 1:] >= 0.0), f"negative tracer: {out.min()}"
    dp1 = pe1[:, 2:] - pe1[:, 1:-1]
    assert np.allclose((out[:, 1:] * dp2[:, 1:]).sum(axis=1),
                       (q1[:, 1:] * dp1).sum(axis=1), rtol=1e-13)


def test_rezone_overshoots_a_sharp_interior_extremum():
    """PPM is NOT bound-preserving at a local extremum, and that is right.

    At km=5 the interior loop covers k=3 ONLY, and for iv != 0 no
    ``cs_limiters`` call is made there (fv_mapz.F90:2237 is gated on
    iv==0). So the cell-3 parabola keeps a peak above its own mean and
    the remapped value can exceed the source maximum. An implementation
    that clipped this would be MORE monotone than the oracle and would
    fail parity -- the property to protect is conservation, which is
    asserted alongside.
    """
    pe1, pe2, _, _ = _lagrangian_edges()
    u1 = np.zeros((4, KM + 1), dtype=np.float64)
    u1[:, 1:] = np.array([-4.0, 12.0, 30.0, 22.0, 6.0])   # sharp max at k=3
    out = map1_ppm(pe1, u1, pe2, KM, KM, -1, 9)
    dp1 = pe1[:, 2:] - pe1[:, 1:-1]
    dp2 = pe2[:, 2:] - pe2[:, 1:-1]
    assert np.allclose((out[:, 1:] * dp2).sum(axis=1),
                       (u1[:, 1:] * dp1).sum(axis=1), rtol=2e-14)
    over = out[:, 1:].max() - u1[:, 1:].max()
    assert over > 0.0, "no overshoot: something is clipping the parabola"
    assert over < 0.05 * u1[:, 1:].max(), f"overshoot {over} is not PPM-sized"


def test_map1_q2_divides_by_the_callers_dp2():
    """``map1_q2`` must use ``dp2``, not ``pe2(k+1)-pe2(k)``.

    They are equal on a consistent grid, so the only way to see the
    difference is to hand it an inconsistent one. A port that ignored
    ``dp2`` would return the ``map_scalar`` answer here.
    """
    pe1, pe2, _, _ = _lagrangian_edges()
    q1 = _column(amp=0.004)
    dp2_true = np.zeros((q1.shape[0], KM + 1), dtype=np.float64)
    dp2_true[:, 1:] = pe2[:, 2:] - pe2[:, 1:-1]
    ref = map1_q2(pe1, q1, pe2, dp2_true, KM, KM, 0, 9, 0.0)
    skewed = dp2_true.copy()
    skewed[:, 1:] *= 2.0
    got = map1_q2(pe1, q1, pe2, skewed, KM, KM, 0, 9, 0.0)
    moved = np.abs(got[:, 1:] - ref[:, 1:]) > 1e-12
    assert moved.any(), "map1_q2 ignored dp2"


# ---------------------------------------------------------------------- #
# the profile builders
# ---------------------------------------------------------------------- #

def _a4(q1, km=KM):
    im = q1.shape[0]
    a4 = np.zeros((5, im, km + 1), dtype=np.float64)
    a4[1] = q1
    return a4


def test_scalar_and_cs_profile_are_not_the_same_function():
    """The ``qmin`` clause exists only in ``scalar_profile``.

    Build a column whose k=3 cell is BOTH a local extremum and colder
    than ``qmin``, while k=2 and k=4 are NOT extrema -- otherwise the
    2-delta-z clauses that BOTH routines carry fire first.

    TWO EARLIER COLUMNS FAILED THIS, for different reasons, and both are
    worth remembering:
      [300, 305, 100, 306, 301] -- k=2 is also an extremum, so cs's
          ``extm(k) .and. extm(k-1)`` clause flattens too.
      [300, 290, 100, 200, 210] -- k=2/k=4 are clean, but the drop to 100
          is so deep that cs's Huynh clamp drives AL and AR onto the cell
          mean anyway, giving the SAME (100, 100, 0). A "different code
          path" is not the same as "a different answer".
    """
    im = 2
    delp = np.zeros((im, KM + 1), dtype=np.float64)
    delp[:, 1:] = 1.0
    q = np.array([350.0, 230.0, 170.0, 210.0, 320.0])
    q1 = np.zeros((im, KM + 1), dtype=np.float64)
    q1[:, 1:] = q

    a_s, a_c = _a4(q1), _a4(q1)
    scalar_profile(a_s, delp, KM, 1, 9, qmin=T_MIN)
    cs_profile(a_c, delp, KM, 1, 9)

    # k=3 is the only layer the interior loop touches at km=5. Flattening
    # means AL = AR = the CELL MEAN of layer 3, i.e. q[2] (0-based).
    qbar3 = q[2]
    assert np.allclose(a_s[2, :, 3], qbar3), "scalar_profile did not flatten"
    assert np.allclose(a_s[3, :, 3], qbar3)
    assert np.allclose(a_s[4, :, 3], 0.0)
    assert not np.allclose(a_c[2, :, 3], a_s[2, :, 3]), \
        "cs_profile flattened too -- it has no qmin clause"
    assert not np.allclose(a_c[4, :, 3], 0.0), \
        "cs_profile's A6 collapsed to 0 -- the clamp, not the qmin clause"


def test_cs_profile_kord15_lets_ext5_swallow_the_ext6_arm():
    """cs_profile nests kord=15 differently from scalar_profile.

    ``scalar_profile`` (:2176-2195) is a FLAT elseif chain, so
    ``elseif (ext6)`` is reachable when ext5(k) is set and no neighbour
    is. ``cs_profile`` (:2607-2623) wraps the ext5 tests INSIDE
    ``if (ext5(k))``, so that same point is left completely untouched.
    Column supplied by the adversarial reviewer; it is the only one in
    this file that separates the two nestings.
    """
    delp = np.zeros((1, KM + 1), dtype=np.float64)
    delp[0, 1:] = [0.24887241281639058, 0.1673993544216956,
                   1.7437252316008056, 0.16335454668312105,
                   2.384564998820821]
    q1 = np.zeros((1, KM + 1), dtype=np.float64)
    q1[0, 1:] = [-4.566530047187453, -0.061160100798463395,
                 4.344253573583179, 1.950142859220142,
                 -0.7382462074722012]
    a4 = _a4(q1)
    cs_profile(a4, delp, KM, 1, 15)
    # ext5 = (F, F, T, F, F): ext5(3) set, neither neighbour set, so the
    # oracle's `if (ext5)` swallows the ext6 arm and k=3 is untouched.
    assert not np.allclose(a4[2, 0, 3], a4[1, 0, 3]), \
        "cs_profile flattened k=3 at kord=15 -- ext5 has no neighbour here"
    assert abs(a4[4, 0, 3]) > 1.0, \
        "A6 is ~0 at k=3: the ext6 clamp fired, but cs_profile blocks it"


def test_scalar_profile_kord15_does_reach_the_ext6_arm():
    """The other side of the nesting difference -- non-vacuity for it."""
    delp = np.zeros((1, KM + 1), dtype=np.float64)
    delp[0, 1:] = [0.24887241281639058, 0.1673993544216956,
                   1.7437252316008056, 0.16335454668312105,
                   2.384564998820821]
    q1 = np.zeros((1, KM + 1), dtype=np.float64)
    q1[0, 1:] = [-4.566530047187453, -0.061160100798463395,
                 4.344253573583179, 1.950142859220142,
                 -0.7382462074722012]
    a_s, a_c = _a4(q1), _a4(q1)
    scalar_profile(a_s, delp, KM, 1, 15, qmin=-1e30)   # qmin cannot fire
    cs_profile(a_c, delp, KM, 1, 15)
    assert not np.allclose(a_s[2:5, 0, 3], a_c[2:5, 0, 3]), \
        "the two kord=15 nestings gave the same answer -- one is wrong"


def test_kord12_uses_the_6_minus_3_grouping_in_both_routines():
    """fv_mapz.F90:2141 and :2571 both write ``6.*q - 3.*(AL+AR)``.

    Checked BITWISE: the two groupings are algebraically identical, so an
    ``allclose`` here would pass with the wrong one.
    """
    delp = np.zeros((1, KM + 1), dtype=np.float64)
    delp[0, 1:] = [0.9, 1.1, 1.0, 1.3, 0.8]
    q1 = np.zeros((1, KM + 1), dtype=np.float64)
    q1[0, 1:] = [12.7, 3.1, 19.9, 5.3, 14.2]
    a_s, a_c = _a4(q1), _a4(q1)
    scalar_profile(a_s, delp, KM, 1, 12, qmin=-1e30)
    cs_profile(a_c, delp, KM, 1, 12)
    assert a_s[4, 0, 3].tobytes() == a_c[4, 0, 3].tobytes(), \
        "kord=12 A6 differs between the routines -- one used 3.*(2.*q-...)"


def test_scalar_and_cs_profile_agree_where_the_qmin_clause_cannot_fire():
    """Above ``qmin`` the two differ only by the A6 FP grouping."""
    im = 3
    delp = np.zeros((im, KM + 1), dtype=np.float64)
    delp[:, 1:] = np.array([0.9, 1.1, 1.0, 1.3, 0.8])
    q1 = _column(im=im, amp=6.0)
    a_s, a_c = _a4(q1), _a4(q1)
    scalar_profile(a_s, delp, KM, 1, 9, qmin=T_MIN)
    cs_profile(a_c, delp, KM, 1, 9)
    assert np.allclose(a_s[1:5], a_c[1:5], rtol=1e-13, atol=1e-11)


def test_cs_limiters_iv1_flattens_a_non_bracketing_cell():
    """iv=1 recomputes the extremum test and ignores ``extm``."""
    a4 = np.zeros((5, 2), dtype=np.float64)
    a4[1] = 10.0
    a4[2] = 11.0        # both edges above the mean -> (q-AL)(q-AR) > 0
    a4[3] = 12.0
    a4[4] = 3.0 * (2.0 * 10.0 - (11.0 + 12.0))
    cs_limiters(a4, np.zeros(2, dtype=bool), 1)      # extm all False
    assert np.allclose(a4[2], 10.0)
    assert np.allclose(a4[3], 10.0)
    assert np.allclose(a4[4], 0.0)


def test_cs_limiters_iv2_uses_the_precomputed_extm():
    """The ``else`` arm flattens on ``extm`` alone -- the iv=1/iv=2 split."""
    base = np.zeros((5, 2), dtype=np.float64)
    base[1] = 10.0
    base[2] = 9.5       # brackets the mean, so iv=1 would NOT flatten
    base[3] = 10.5
    base[4] = 0.0
    on = base.copy()
    cs_limiters(on, np.ones(2, dtype=bool), 2)
    assert np.allclose(on[2], 10.0) and np.allclose(on[3], 10.0)
    off = base.copy()
    cs_limiters(off, np.zeros(2, dtype=bool), 2)
    assert np.allclose(off[2], 9.5), "extm=False must not flatten"


def test_low_kord_is_refused_rather_than_silently_rerouted():
    with pytest.raises(ValueError, match="ppm_profile"):
        ppm_profile_is_unported(4)
    pe1, pe2, _, _ = _lagrangian_edges()
    with pytest.raises(ValueError, match="ppm_profile"):
        map_scalar(pe1, _column(), pe2, KM, KM, 1, 7, T_MIN)
    ppm_profile_is_unported(9)          # the pinned lane must pass


def test_map1_ppm_wind_lane_conserves_momentum():
    """iv=-1, kord_mt=9 -- the u/v lane (fv_mapz.F90:547, :567)."""
    pe1, pe2, _, _ = _lagrangian_edges()
    u1 = _monotone_column() - 240.0            # spans both signs
    out = map1_ppm(pe1, u1, pe2, KM, KM, -1, 9)
    dp1 = pe1[:, 2:] - pe1[:, 1:-1]
    dp2 = pe2[:, 2:] - pe2[:, 1:-1]
    assert np.allclose((out[:, 1:] * dp2).sum(axis=1),
                       (u1[:, 1:] * dp1).sum(axis=1), rtol=2e-14)


def test_map1_ppm_iv_minus1_sign_rules_change_the_boundary_layers():
    """iv=-1's edge rules (fv_mapz.F90:2024-2027, :2248-2251).

    ``if (AL*qbar <= 0) AL = 0`` at k=1, and the mirrored ``a4(3,km)``
    rule at the bottom, are the wind lane's ONLY behavioural difference
    from iv=1. They fire only where the reconstructed edge disagrees in
    sign with its cell mean, which depends on the whole tridiagonal
    solve -- so this sweeps sign-crossing columns and requires at least
    one to separate the two, rather than betting on a single hand-picked
    column (the first attempt bet wrong: AL and qbar were both negative).
    """
    dp1 = np.zeros((1, KM + 1), dtype=np.float64)
    dp1[:, 1:] = 1.0
    columns = [
        [0.5, 20.0, 28.0, 32.0, 36.0],
        [-0.5, -20.0, -28.0, -32.0, -36.0],
        [30.0, 22.0, 12.0, 2.0, -0.5],
        [-2.0, 18.0, 26.0, 30.0, 34.0],
        [0.2, -14.0, -22.0, -26.0, -30.0],
    ]
    separated = []
    for col in columns:
        q1 = np.zeros((1, KM + 1), dtype=np.float64)
        q1[0, 1:] = col
        a_w, a_t = _a4(q1), _a4(q1)
        cs_profile(a_w, dp1, KM, -1, 9)
        cs_profile(a_t, dp1, KM, 1, 9)
        if not np.allclose(a_w[1:5, 0, :], a_t[1:5, 0, :]):
            separated.append(col)
    assert separated, (
        "no sign-crossing column separated iv=-1 from iv=1 -- either the "
        "edge rules are missing or every column here has AL and qbar of "
        "the same sign")


def test_profile_refuses_a_km_its_layer_blocks_would_overlap():
    a4 = _a4(np.ones((2, 4)), km=3)
    delp = np.ones((2, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="km=3"):
        scalar_profile(a4, delp, 3, 1, 9, qmin=T_MIN)


# ---------------------------------------------------------------------- #
# the driver
# ---------------------------------------------------------------------- #

def _face(n=7, ng=3, km=KM, ps0=1000.0e2, deform=0.25):
    """One face's arrays in the fv3_native_state_3d layout.

    ``n`` deliberately does NOT equal ``km+1``: ``pe``/``peln`` are stored
    ``(i,k,j)`` while ``delp``/``pk`` are ``(i,j,k)``, so at n == km+1 a
    missing transpose is shape-compatible and silently wrong. It cost one
    round-trip here; keep n != km+1.
    """
    m_a = n + 2 * ng
    m_b = m_a + 1
    ak, bk, ptop, _ = set_eta_analytic(km)
    ia = ng

    ps2d = ps0 + 30.0e2 * np.cos(
        np.arange(n)[:, None] * 0.7 + np.arange(n)[None, :] * 0.4)

    # Lagrangian interfaces: the Eulerian ones, skewed, same column mass.
    pe = np.zeros((n + 2, km + 1, n + 2), dtype=np.float64)
    for jj in range(n + 2):
        for ii in range(n + 2):
            ps = ps0 + 30.0e2 * np.cos((ii - 1) * 0.7 + (jj - 1) * 0.4)
            dp = np.diff(ak + bk * ps)
            dp = dp * (1.0 + deform * np.cos(np.arange(km) + 0.3 * ii))
            dp *= (ps - ptop) / dp.sum()
            pe[ii, :, jj] = np.concatenate([[ptop], ptop + np.cumsum(dp)])

    peln = np.log(pe[1:n + 1, :, 1:n + 1])            # (i, k, j)
    pe_ikj = pe[1:n + 1, :, 1:n + 1]
    pk = np.zeros((m_a, m_a, km + 1), dtype=np.float64)
    pk[ia:ia + n, ia:ia + n, :] = (pe_ikj ** AKAP).transpose(0, 2, 1)

    delp = np.zeros((m_a, m_a, km), dtype=np.float64)
    delp[ia:ia + n, ia:ia + n, :] = np.diff(pe_ikj, axis=1).transpose(0, 2, 1)

    pkz = np.zeros((n, n, km), dtype=np.float64)
    pkc = pk[ia:ia + n, ia:ia + n, :]
    pkz[:] = ((pkc[:, :, 1:] - pkc[:, :, :-1])
              / (AKAP * (peln[:, 1:, :] - peln[:, :-1, :]
                         ).transpose(0, 2, 1)))

    # pt as virtual POTENTIAL temperature, i.e. T_v / pkz.
    tv = 250.0 + 20.0 * np.cos(np.arange(km) * 1.1)[None, None, :] \
        + 4.0 * np.cos(np.arange(n))[:, None, None]
    pt = np.zeros((m_a, m_a, km), dtype=np.float64)
    pt[ia:ia + n, ia:ia + n, :] = tv / pkz

    u = np.zeros((m_a, m_b, km), dtype=np.float64)
    v = np.zeros((m_b, m_a, km), dtype=np.float64)
    u[ia:ia + n, ia:ia + n + 1, :] = 15.0 + 5.0 * np.cos(
        np.arange(km) * 0.9)[None, None, :]
    v[ia:ia + n + 1, ia:ia + n, :] = -3.0 + 2.0 * np.sin(
        np.arange(km) * 0.6)[None, None, :]
    ps = np.zeros((m_a, m_a), dtype=np.float64)

    # omga must be supplied: fv_mapz.F90:504-523 rewrites it on every
    # last_step, so the driver refuses to skip it silently. Give it real
    # vertical structure so the interpolation cannot pass by being a no-op.
    omga = np.zeros((m_a, m_a, km), dtype=np.float64)
    omga[ia:ia + n, ia:ia + n, :] = (
        -0.4 + 0.25 * np.arange(km))[None, None, :] + 0.01 * np.arange(
            n)[:, None, None]

    return dict(pe=pe, peln=peln, pk=pk, pkz=pkz, delp=delp, pt=pt,
                u=u, v=v, ps=ps, omga=omga, ak=ak, bk=bk, ptop=ptop,
                akap=AKAP, cp=1004.6, r_vir=0.0, km=km, n=n, ng=ng,
                kord_mt=9, kord_tm=-9, kord_tr=9), ps2d


def test_driver_puts_delp_on_the_eulerian_coordinate():
    face, _ = _face()
    n, ng, km = face["n"], face["ng"], face["km"]
    ak, bk, ptop = face["ak"], face["bk"], face["ptop"]
    before = face["delp"][ng:ng + n, ng:ng + n, :].copy()

    lagrangian_to_eulerian(**face, q=[])

    ia = ng
    ps = face["ps"][ia:ia + n, ia:ia + n]
    want = (np.diff(ak)[None, None, :]
            + np.diff(bk)[None, None, :] * ps[:, :, None])
    got = face["delp"][ia:ia + n, ia:ia + n, :]
    assert np.allclose(got, want, rtol=1e-14)
    # non-vacuity: the Lagrangian delp really was different
    assert np.max(np.abs(before - got)) > 1.0
    # and the column mass is untouched
    assert np.allclose(got.sum(axis=2), ps - ptop, rtol=1e-13)


def test_driver_conserves_the_column_heat_integral():
    """INT(T dlnp) is what kord_tm<0 conserves -- not INT(theta dp)."""
    face, _ = _face()
    n, ng, km = face["n"], face["ng"], face["km"]
    ia = ng
    pkc = face["pk"][ia:ia + n, ia:ia + n, :].copy()
    pln0 = face["peln"].copy()
    pt0 = face["pt"][ia:ia + n, ia:ia + n, :].copy()
    # T_v = theta_v * pkz_lagrangian (fv_mapz.F90:215)
    lag_pkz = ((pkc[:, :, 1:] - pkc[:, :, :-1])
               / (face["akap"] * (pln0[:, 1:, :] - pln0[:, :-1, :]
                                  ).transpose(0, 2, 1)))
    tv0 = pt0 * lag_pkz
    dlnp0 = (pln0[:, 1:, :] - pln0[:, :-1, :]).transpose(0, 2, 1)
    src = (tv0 * dlnp0).sum(axis=2)

    lagrangian_to_eulerian(**face, q=[])

    pln1 = face["peln"]
    dlnp1 = (pln1[:, 1:, :] - pln1[:, :-1, :]).transpose(0, 2, 1)
    dst = (face["pt"][ia:ia + n, ia:ia + n, :] * dlnp1).sum(axis=2)
    assert np.allclose(dst, src, rtol=1e-12)


def test_driver_leaves_pt_as_temperature_at_last_step():
    """last_step=True must NOT divide pt by the new pkz."""
    face, _ = _face()
    n, ng = face["n"], face["ng"]
    ia = ng
    lagrangian_to_eulerian(**face, q=[])
    t = face["pt"][ia:ia + n, ia:ia + n, :]
    assert 180.0 < t.min() and t.max() < 350.0, \
        f"pt looks like theta, not T: [{t.min()}, {t.max()}]"


def test_driver_returns_theta_when_not_last_step():
    """The two arms differ by exactly one division by the NEW pkz.

    NOT asserted as "theta is large": FV3's pkz is ``p**kappa`` (not
    ``(p/p0)**kappa``), so theta_v here is ~10, not ~300. An earlier
    version of this test asserted ``> 400`` and failed on correct code.
    """
    n, ng = 7, 3
    ia = ng
    a, _ = _face()
    b, _ = _face()
    lagrangian_to_eulerian(**a, q=[])                       # last_step=True -> T
    lagrangian_to_eulerian(**b, q=[], last_step=False)      # -> theta_v
    assert np.allclose(a["pkz"], b["pkz"], rtol=0.0, atol=0.0)
    assert np.allclose(b["pt"][ia:ia + n, ia:ia + n, :],
                       a["pt"][ia:ia + n, ia:ia + n, :] / a["pkz"],
                       rtol=1e-15)
    assert not np.allclose(b["pt"][ia:ia + n, ia:ia + n, :],
                           a["pt"][ia:ia + n, ia:ia + n, :])


def test_driver_updates_pe_peln_pk_consistently():
    face, _ = _face()
    n, ng, km = face["n"], face["ng"], face["km"]
    ia = ng
    lagrangian_to_eulerian(**face, q=[])
    pe_int = face["pe"][1:n + 1, :, 1:n + 1]              # (i, k, j)
    assert np.allclose(np.exp(face["peln"]), pe_int, rtol=1e-13)
    assert np.allclose(face["pk"][ia:ia + n, ia:ia + n, :],
                       (pe_int ** face["akap"]).transpose(0, 2, 1),
                       rtol=1e-13)


def test_driver_remaps_the_winds():
    face, _ = _face()
    n, ng = face["n"], face["ng"]
    ia = ng
    u0 = face["u"][ia:ia + n, ia:ia + n + 1, :].copy()
    v0 = face["v"][ia:ia + n + 1, ia:ia + n, :].copy()
    lagrangian_to_eulerian(**face, q=[])
    assert np.max(np.abs(face["u"][ia:ia + n, ia:ia + n + 1, :] - u0)) > 1e-6
    assert np.max(np.abs(face["v"][ia:ia + n + 1, ia:ia + n, :] - v0)) > 1e-6


def test_driver_remaps_both_tracers_of_the_pinned_lane():
    """nr = 2 on this deck (ncnst=3, dnats=1) -- test both, separately.

    A single-tracer test cannot see a loop that remaps only ``q[0]`` and
    leaves the rest, which is the natural off-by-one here.
    """
    face, _ = _face()
    n, ng, km = face["n"], face["ng"], face["km"]
    ia = ng
    m_a = n + 2 * ng
    qs = []
    for scale, phase in ((1e-3, 1.4), (4e-4, 0.3)):
        qq = np.zeros((m_a, m_a, km), dtype=np.float64)
        qq[ia:ia + n, ia:ia + n, :] = scale * (
            1.0 + 0.5 * np.cos(np.arange(km) * phase))[None, None, :]
        qs.append(qq)
    before = [qq[ia:ia + n, ia:ia + n, :].copy() for qq in qs]
    dp0 = face["delp"][ia:ia + n, ia:ia + n, :].copy()
    src = [(b * dp0).sum(axis=2) for b in before]

    lagrangian_to_eulerian(**face, q=qs)

    dp1 = face["delp"][ia:ia + n, ia:ia + n, :]
    for iq, (qq, b, s) in enumerate(zip(qs, before, src)):
        now = qq[ia:ia + n, ia:ia + n, :]
        assert np.allclose((now * dp1).sum(axis=2), s, rtol=1e-12), \
            f"tracer {iq} not conserved"
        assert np.max(np.abs(now - b)) > 1e-9, f"tracer {iq} never remapped"


def test_driver_remaps_omega_onto_the_new_cell_centres():
    """fv_mapz.F90:504-523 -- and it must not be a silent no-op."""
    face, _ = _face()
    n, ng = face["n"], face["ng"]
    ia = ng
    before = face["omga"][ia:ia + n, ia:ia + n, :].copy()
    lagrangian_to_eulerian(**face, q=[])
    after = face["omga"][ia:ia + n, ia:ia + n, :]
    assert np.max(np.abs(after - before)) > 1e-6, "omega was not remapped"
    # The interpolation is a convex combination of the OLD column, so it
    # cannot leave that column's range.
    assert after.min() >= before.min() - 1e-12
    assert after.max() <= before.max() + 1e-12


def test_driver_requires_omga_at_last_step():
    face, _ = _face()
    face.pop("omga")
    with pytest.raises(ValueError, match="omga"):
        lagrangian_to_eulerian(**face, q=[])


def test_driver_uses_the_declared_sphum_index_not_tracer_zero():
    """fv_mapz.F90:975 divides by the EXPLICIT sphum tracer.

    With q[0] dry and q[1] = 0.5, taking tracer 0 leaves T_v unchanged
    while the oracle divides by 1.5 -- a 33% error that no conservation
    test would see.
    """
    n, ng, km = 7, 3, KM
    ia, m_a = ng, n + 2 * ng

    def _tracers():
        dry = np.zeros((m_a, m_a, km), dtype=np.float64)
        wet = np.zeros((m_a, m_a, km), dtype=np.float64)
        wet[ia:ia + n, ia:ia + n, :] = 0.5
        return [dry, wet]

    dry_face, _ = _face()                      # r_vir is already 0.0
    lagrangian_to_eulerian(**dry_face, q=_tracers())
    moist_face, _ = _face()
    moist_face["r_vir"] = 1.0
    lagrangian_to_eulerian(**moist_face, q=_tracers(), sphum_index=1)
    got = moist_face["pt"][ia:ia + n, ia:ia + n, :]
    want = dry_face["pt"][ia:ia + n, ia:ia + n, :] / 1.5
    assert np.allclose(got, want, rtol=1e-14)

    bad, _ = _face()
    bad["r_vir"] = 1.0
    with pytest.raises(ValueError, match="sphum_index"):
        lagrangian_to_eulerian(**bad, q=_tracers())


@pytest.mark.parametrize("override, needle, ntracer", [
    # hydrostatic=False is PORTED now (w/delz remap, w_limiter, NH pkz);
    # its argument requirements are asserted in
    # test_driver_nh_requires_its_arguments below.
    (dict(consv=1.0), "consv", 0),
    # fillz sits INSIDE the `elseif (nq > 0)` arm (fv_mapz.F90:330-336), so
    # fill=True is only refusable when there are tracers -- with none it is
    # unreachable upstream and must NOT raise (asserted separately in
    # test_driver_guards_do_not_over_refuse_a_non_last_step_call).
    (dict(fill=True), "fillz", 1),
    (dict(kord_tm=9), "kord_tm", 0),
    (dict(do_sat_adj=True), "do_sat_adj", 0),
])
def test_driver_refuses_every_unported_lane(override, needle, ntracer):
    face, _ = _face()
    face.update(override)
    m_a = face["n"] + 2 * face["ng"]
    tr = [np.zeros((m_a, m_a, face["km"]), dtype=np.float64)
          for _ in range(ntracer)]
    with pytest.raises(NotImplementedError, match=needle):
        lagrangian_to_eulerian(**face, q=tr)


def test_driver_nh_requires_its_arguments():
    """The NH lane must not run on defaults: w/delz/ws/rdgas/grav are
    all required (a silently-defaulted rdgas would be a wrong pkz, not
    an error), and a negative kord_wz selects the refused iv=-3 arm."""
    face, _ = _face()
    face["hydrostatic"] = False
    with pytest.raises(ValueError, match="hydrostatic=False needs"):
        lagrangian_to_eulerian(**face, q=[])
    nh = _nh_face()
    nh["kord_wz"] = -9
    with pytest.raises(NotImplementedError, match="iv"):
        lagrangian_to_eulerian(**nh, q=[])


def test_driver_refuses_a_moist_lane_without_any_tracers():
    face, _ = _face()
    face["r_vir"] = 0.6077
    with pytest.raises(ValueError, match="r_vir"):
        lagrangian_to_eulerian(**face, q=[])


def test_iv_minus3_is_refused_rather_than_invented():
    """fv_mapz.F90:2320-2339 reads an UNASSIGNED gam(i,km) at :2358.

    Zero-filling it would turn undefined oracle behaviour into a
    confident number. scalar_profile has no -3 arm at all (:1877 tests
    only -2), so there it must fall through to the default solve.
    """
    q1 = _column(im=2)
    delp = np.zeros((2, KM + 1), dtype=np.float64)
    delp[:, 1:] = 1.0
    with pytest.raises(NotImplementedError, match="iv=-3"):
        cs_profile(_a4(q1), delp, KM, -3, 9, qs=np.zeros(2))
    ref, got = _a4(q1), _a4(q1)
    scalar_profile(ref, delp, KM, 1, 9, qmin=T_MIN)
    scalar_profile(got, delp, KM, -3, 9, qmin=T_MIN)   # must NOT raise
    assert np.allclose(ref[1:5], got[1:5])


def test_pad_roundtrip():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert np.array_equal(unpad1(pad1(a)), a)
    assert np.array_equal(pad1(a)[:, 0], np.zeros(3))


# ---------------------------------------------------------------------- #
# driver wiring: recompute each output from the ported kernels directly
#
# SCOPE, stated plainly: these pin the DRIVER'S WIRING -- which kernel it
# calls, with which iv/kord/coordinate/divisor -- not the kernel maths.
# They kill the mutation class the adversarial reviewer demonstrated
# (map_scalar <-> map1_ppm, iv=0 <-> iv=1, iv=-1 <-> iv=1, kord 9 <-> 12,
# a doubled pkz, omega read at the interface instead of the midpoint), all
# of which previously stayed green. Anchoring the kernel MATHS needs a
# Fortran-produced golden and is tracked separately.
# ---------------------------------------------------------------------- #

def _pn2_for_row(face, j, n, ak, bk):
    """Rebuild the Eulerian ln(p) target for row j, as the driver does.

    Endpoints are COPIED from peln (fv_mapz.F90:297-298), not recomputed
    from ptop/ps -- reproducing that here is what makes the comparison
    exact rather than approximate.
    """
    km = face["km"]
    peln0 = face["_peln0"]
    pe0 = face["_pe0"]
    pn2 = np.zeros((n, km + 2), dtype=np.float64)
    pn2[:, 1] = peln0[:, 0, j - 1]
    pn2[:, km + 1] = peln0[:, km, j - 1]
    for k in range(2, km + 1):
        pn2[:, k] = np.log(ak[k - 1] + bk[k - 1] * pe0[1:n + 1, km, j])
    return pn2


def _face_with_snapshots():
    face, _ = _face()
    face["_pe0"] = face["pe"].copy()
    face["_peln0"] = face["peln"].copy()
    face["_pk0"] = face["pk"].copy()
    face["_pt0"] = face["pt"].copy()
    face["_u0"] = face["u"].copy()
    face["_v0"] = face["v"].copy()
    face["_omga0"] = face["omga"].copy()
    return face


def _call(face, **kw):
    """Invoke the driver with the snapshot keys stripped out."""
    args = {k: v for k, v in face.items() if not k.startswith("_")}
    return lagrangian_to_eulerian(**args, **kw)


def test_driver_pt_equals_a_direct_map_scalar_call():
    """kord_tm<0 selects map_scalar in ln(p) -- not map1_ppm in p.

    Both conserve their own integral, so the heat-integral test alone
    cannot tell them apart (reviewer's counterexample: 182.01792793 vs
    181.60677638 on a cold column).
    """
    face = _face_with_snapshots()
    n, ng, km = face["n"], face["ng"], face["km"]
    ia, akap = ng, face["akap"]
    ak, bk = face["ak"], face["bk"]
    _call(face, q=[])

    j = 3
    pk0 = face["_pk0"][ia:ia + n, ia + j - 1, :]
    pln0 = face["_peln0"][:, :, j - 1]
    lag_pkz = (pk0[:, 1:] - pk0[:, :-1]) / (akap * (pln0[:, 1:] - pln0[:, :-1]))
    tv = face["_pt0"][ia:ia + n, ia + j - 1, :] * lag_pkz
    want = unpad1(map_scalar(pad1(pln0), pad1(tv),
                             _pn2_for_row(face, j, n, ak, bk),
                             km, km, 1, abs(face["kord_tm"]), T_MIN))
    got = face["pt"][ia:ia + n, ia + j - 1, :]
    assert np.array_equal(got, want), \
        f"driver pt != map_scalar(iv=1, ln p): max |d| = {np.abs(got-want).max()}"


def test_driver_u_equals_a_direct_map1_ppm_call_including_the_je_plus_1_row():
    """The u remap runs for j = js..je+1 (fv_mapz.F90:195, :547).

    The north row is the whole reason the j-loop extends one past je, and
    it is remapped against the deliberately OLD pe(...,j-1). Ending the
    loop at je leaves it untouched and every previous wind test still
    passed.
    """
    face = _face_with_snapshots()
    n, ng, km = face["n"], face["ng"], face["km"]
    ia = ng
    ak, bk = face["ak"], face["bk"]
    pe0, u0 = face["_pe0"], face["_u0"]
    _call(face, q=[])

    for j in (2, n + 1):                       # an interior row and je+1
        jd, jpe = ia + j - 1, j
        pe1 = pad1(pe0[1:n + 1, :, jpe])
        pe0_u = np.zeros((n, km + 2), dtype=np.float64)
        pe3_u = np.zeros((n, km + 2), dtype=np.float64)
        pe0_u[:, 1] = pe0[1:n + 1, 0, jpe]
        for k in range(2, km + 2):
            pe0_u[:, k] = 0.5 * (pe0[1:n + 1, k - 1, jpe - 1] + pe1[:, k])
        for k in range(1, km + 2):
            bkh = 0.5 * bk[k - 1]
            pe3_u[:, k] = ak[k - 1] + bkh * (pe0[1:n + 1, km, jpe - 1]
                                             + pe1[:, km + 1])
        want = unpad1(map1_ppm(pe0_u, pad1(u0[ia:ia + n, jd, :]), pe3_u,
                               km, km, -1, face["kord_mt"]))
        got = face["u"][ia:ia + n, jd, :]
        assert np.array_equal(got, want), f"u row j={j} mismatch"

    assert not np.array_equal(face["u"][ia:ia + n, ia + n, :],
                              u0[ia:ia + n, ia + n, :]), \
        "the j=je+1 u row was never remapped -- the loop stopped at je"


def test_driver_v_equals_a_direct_map1_ppm_call_over_is_to_ie_plus_1():
    """v spans i = is..ie+1 and is skipped at j = je+1 (:551-570)."""
    face = _face_with_snapshots()
    n, ng, km = face["n"], face["ng"], face["km"]
    ia, npv = ng, n + 1
    ak, bk = face["ak"], face["bk"]
    pe0, v0 = face["_pe0"], face["_v0"]
    _call(face, q=[])

    j, jd, jpe = 3, ia + 2, 3
    pe0_v = np.zeros((npv, km + 2), dtype=np.float64)
    pe3_v = np.zeros((npv, km + 2), dtype=np.float64)
    pe0_v[:, 1] = pe0[1:1 + npv, 0, jpe]
    pe3_v[:, 1] = ak[0]
    for k in range(2, km + 2):
        bkh = 0.5 * bk[k - 1]
        pe0_v[:, k] = 0.5 * (pe0[0:npv, k - 1, jpe] + pe0[1:1 + npv, k - 1, jpe])
        pe3_v[:, k] = ak[k - 1] + bkh * (pe0[0:npv, km, jpe]
                                         + pe0[1:1 + npv, km, jpe])
    want = unpad1(map1_ppm(pe0_v, pad1(v0[ia:ia + npv, jd, :]), pe3_v,
                           km, km, -1, face["kord_mt"]))
    assert np.array_equal(face["v"][ia:ia + npv, jd, :], want)
    # v is NOT remapped on the j=je+1 pass
    assert np.array_equal(face["v"][ia:ia + npv, ia + n, :],
                          v0[ia:ia + npv, ia + n, :]), \
        "v was remapped at j=je+1, but :551 guards it with `if (j < je+1)`"


def test_driver_tracer_equals_a_direct_map1_q2_call_with_iv0():
    """The tracer lane is map1_q2(iv=0, qmin=0.) with the driver's dp2.

    iv=1 conserves too, so the conservation test could not see it; on the
    reviewer's spiky column the iv=1 mutant reaches -3.5e-3.
    """
    face = _face_with_snapshots()
    n, ng, km = face["n"], face["ng"], face["km"]
    ia = ng
    ak, bk = face["ak"], face["bk"]
    m_a = n + 2 * ng
    qv = np.zeros((m_a, m_a, km), dtype=np.float64)
    qv[ia:ia + n, ia:ia + n, :] = np.array(
        [1e-6, 8e-3, 2e-5, 9e-3, 1e-7])[None, None, :]
    q0 = qv.copy()
    pe0 = face["_pe0"]
    _call(face, q=[qv])

    j, jd, jpe = 3, ia + 2, 3
    pe1 = pad1(pe0[1:n + 1, :, jpe])
    pe2 = np.zeros((n, km + 2), dtype=np.float64)
    pe2[:, 1] = face["ptop"]
    pe2[:, km + 1] = pe0[1:n + 1, km, jpe]
    for k in range(2, km + 1):
        pe2[:, k] = ak[k - 1] + bk[k - 1] * pe0[1:n + 1, km, jpe]
    dp2 = np.zeros((n, km + 1), dtype=np.float64)
    for k in range(1, km + 1):
        dp2[:, k] = pe2[:, k + 1] - pe2[:, k]
    want = unpad1(map1_q2(pe1, pad1(q0[ia:ia + n, jd, :]), pe2, dp2,
                          km, km, 0, face["kord_tr"], 0.0))
    assert np.array_equal(qv[ia:ia + n, jd, :], want)
    assert np.all(qv[ia:ia + n, ia:ia + n, :] >= 0.0), \
        "tracer went negative -- the driver is not passing iv=0"


def test_driver_pkz_matches_its_interface_formula_exactly():
    """pkz = (pk2(k+1)-pk2(k)) / (akap*(peln_new(k+1)-peln_new(k))).

    A doubled pkz left every earlier assertion green, because the only
    other test that touched pkz checked a self-consistent RATIO.
    """
    face = _face_with_snapshots()
    n, ng, km = face["n"], face["ng"], face["km"]
    ia, akap = ng, face["akap"]
    _call(face, q=[])
    pkc = face["pk"][ia:ia + n, ia:ia + n, :]
    pln = face["peln"]
    want = ((pkc[:, :, 1:] - pkc[:, :, :-1])
            / (akap * (pln[:, 1:, :] - pln[:, :-1, :]).transpose(0, 2, 1)))
    assert np.array_equal(face["pkz"], want)
    # ...and it is a real number, not a leftover zero fill.
    assert face["pkz"].min() > 1.0


def test_driver_omega_interpolates_at_the_new_logp_midpoints():
    """fv_mapz.F90:507 uses 0.5*(peln_new(k)+peln_new(k+1)).

    Reading the top interface instead stays inside the convex range, so
    the range check could not see it (reviewer: max error 0.186).
    """
    face = _face_with_snapshots()
    n, ng, km = face["n"], face["ng"], face["km"]
    ia = ng
    pe0_old_all = face["_peln0"]
    om0 = face["_omga0"]
    _call(face, q=[])

    j, jd = 3, ng + 2
    pln_new = face["peln"][:, :, j - 1]
    mid = 0.5 * (pln_new[:, :-1] + pln_new[:, 1:])
    pe0_old = pe0_old_all[:, :, j - 1]
    pe3 = np.zeros((n, km + 2), dtype=np.float64)
    for k in range(2, km + 2):
        pe3[:, k] = om0[ia:ia + n, jd, k - 2]

    want = np.zeros((n, km), dtype=np.float64)
    for i in range(n):
        k_next = 1
        for nn in range(1, km + 1):
            for k in range(k_next, km + 1):
                if (mid[i, nn - 1] <= pe0_old[i, k]
                        and mid[i, nn - 1] >= pe0_old[i, k - 1]):
                    want[i, nn - 1] = (
                        pe3[i, k] + (pe3[i, k + 1] - pe3[i, k])
                        * (mid[i, nn - 1] - pe0_old[i, k - 1])
                        / (pe0_old[i, k] - pe0_old[i, k - 1]))
                    k_next = k
                    break
    assert np.array_equal(face["omga"][ia:ia + n, jd, :], want)


def test_driver_requires_q_explicitly():
    """No default: skipping the nr=2 tracer lane must be a visible choice."""
    face, _ = _face()
    with pytest.raises(TypeError):
        lagrangian_to_eulerian(**face)


def test_driver_guards_do_not_over_refuse_a_non_last_step_call():
    """consv and sphum are read ONLY inside `if (last_step)` (:628, :964).

    Refusing them on a non-last_step call would be stricter than the
    oracle -- a guard that fires where upstream does nothing is as much a
    divergence as one that fails to fire.
    """
    a, _ = _face()
    lagrangian_to_eulerian(**a, q=[], last_step=False, consv=1.0)
    b, _ = _face()
    b["r_vir"] = 1.0
    m_a = b["n"] + 2 * b["ng"]
    tr = [np.zeros((m_a, m_a, b["km"]), dtype=np.float64) for _ in range(2)]
    lagrangian_to_eulerian(**b, q=tr, last_step=False)
    c, _ = _face()
    lagrangian_to_eulerian(**c, q=[], fill=True)     # fillz needs nq > 0


# ------------------------------------------------------------- NH remap

def _nh_face(w_const=None, seed=59, deform=0.25):
    """The `_face` fixture plus make_nh delz, a w field and the ws BC.

    ``w_const`` fills w with one constant (the iv=-2 constant-
    preservation control); None gives it vertical structure.
    ``deform=0`` makes the Lagrangian and Eulerian coordinates coincide,
    so every remap is an identity -- the lens that isolates the
    conversions from the rezone.
    """
    from legoesm.grids.fv3_native_gridstruct import FV3_GRAV, FV3_RDGAS

    face, ps2d = _face(deform=deform)
    n, ng, km = face["n"], face["ng"], face["km"]
    ia = ng
    rng = np.random.default_rng(seed)
    m_a = n + 2 * ng

    # T_v back from theta (pt = T_v/pkz in the fixture).
    tv = face["pt"][ia:ia + n, ia:ia + n, :] * face["pkz"]
    dpeln = np.diff(face["peln"], axis=1).transpose(0, 2, 1)
    delz = -(FV3_RDGAS / FV3_GRAV) * tv * dpeln

    w = np.zeros((m_a, m_a, km), dtype=np.float64)
    if w_const is None:
        w[ia:ia + n, ia:ia + n, :] = 0.3 * rng.standard_normal((n, n, km))
    else:
        w[ia:ia + n, ia:ia + n, :] = w_const
    ws = (np.full((n, n), w_const) if w_const is not None
          else 0.1 * rng.standard_normal((n, n)))
    face.update(hydrostatic=False, w=w, delz=delz, ws=ws, kord_wz=9,
                w_limiter=False, rdgas=FV3_RDGAS, grav=FV3_GRAV)
    return face


def test_driver_nh_conserves_delz_and_preserves_constant_w():
    """Three NH facts in one integration: (a) the delz remap conserves
    the column height integral EXACTLY (specific volume is remapped
    mass-weighted over an unchanged column mass); (b) a CONSTANT w with
    a matching ws bottom BC comes back as the same constant (the iv=-2
    reconstruction of a constant is the constant); (c) the returned pkz
    is the NH ideal-gas form of the RETURNED delp/delz/pt to the last
    bit (r_vir=0 makes the :975 conversion an identity)."""
    face = _nh_face(w_const=5.0)
    n, ng, km = face["n"], face["ng"], face["km"]
    ia = ng
    dz_before = face["delz"].sum(axis=2).copy()

    lagrangian_to_eulerian(**face, q=[])

    dz_after = face["delz"].sum(axis=2)
    assert np.abs(dz_after - dz_before).max() < 1e-9 * np.abs(
        dz_before).max()
    assert face["delz"].max() < 0.0
    wwin = face["w"][ia:ia + n, ia:ia + n, :]
    assert np.abs(wwin - 5.0).max() < 1e-11, np.abs(wwin - 5.0).max()

    from legoesm.grids.fv3_native_gridstruct import FV3_GRAV, FV3_RDGAS
    rrg = -FV3_RDGAS / FV3_GRAV
    want = np.exp(face["akap"] * np.log(
        rrg * face["delp"][ia:ia + n, ia:ia + n, :] / face["delz"]
        * face["pt"][ia:ia + n, ia:ia + n, :]))
    assert np.allclose(face["pkz"], want, rtol=1e-14, atol=0.0)


def test_driver_nh_theta_conversion_uses_the_ideal_gas_pkz():
    """codex NH r3 #1: the NH theta_v -> T_v conversion is
    ``pt *= exp(k1k*log(rrg*delp/delz*pt))`` (fv_mapz.F90:231-232,
    k1k = rdgas/cv_air), NOT the hydrostatic Dpk/(akap*Dpeln) form.
    At deform=0 the remap is an identity and r_vir=0 makes :975 an
    identity too, so the returned pt IS the converted value -- compared
    here against a direct transcription on the captured inputs."""
    from legoesm.grids.fv3_native_gridstruct import FV3_GRAV, FV3_RDGAS

    face = _nh_face(w_const=0.0, deform=0.0)
    n, ng, km = face["n"], face["ng"], face["km"]
    ia = ng
    pt_in = np.array(face["pt"][ia:ia + n, ia:ia + n, :], copy=True)
    dp_in = np.array(face["delp"][ia:ia + n, ia:ia + n, :], copy=True)
    dz_in = np.array(face["delz"], copy=True)
    # The hydrostatic-form conversion, for the non-vacuity contrast.
    pkc = face["pk"][ia:ia + n, ia:ia + n, :]
    pln = face["peln"]
    hydro_form = pt_in * ((pkc[:, :, 1:] - pkc[:, :, :-1])
                          / (AKAP * np.diff(pln, axis=1).transpose(0, 2, 1)))

    lagrangian_to_eulerian(**face, q=[])

    k1k = FV3_RDGAS / (face["cp"] - FV3_RDGAS)
    rrg = -FV3_RDGAS / FV3_GRAV
    want = pt_in * np.exp(k1k * np.log(rrg * dp_in / dz_in * pt_in))
    got = face["pt"][ia:ia + n, ia:ia + n, :]
    # 1e-11: at deform=0 the coordinates coincide ALGEBRAICALLY but the
    # cumsum-built pe1 and the ak+bk*ps pe2 differ in the last bits, so
    # the "identity" remap still moves pt by ~1e-12 relative.
    assert np.allclose(got, want, rtol=1e-11, atol=0.0)
    # The two conversion forms genuinely differ at this discretisation
    # (~2.8e-4 relative on the thickest log-layer) -- the assert above
    # is not satisfiable by the hydro form.
    assert np.abs(want - hydro_form).max() > 1e-5 * np.abs(want).max()


def test_driver_nh_w_limiter_top_escape_valve():
    """codex NH r3 #5: a large bottom-layer violation cascades UP the
    column and must hit the :408-416 top valve at exactly 2*w_max --
    with the spilled momentum above that DISCARDED (the valve is the
    one deliberately non-conserving branch)."""
    from legoesm.core.fv3_native_mapz import W_MAX_MAPZ

    face = _nh_face(w_const=0.0, deform=0.0)
    n, ng, km = face["n"], face["ng"], face["km"]
    ia = ng
    # Bottom-layer monster: the up pass clamps k=km-1..1 and dumps the
    # excess into k=0, where only the valve can stop it.
    face["w"][ia + 1, ia + 1, km - 1] = 5.0e4
    face["ws"][:] = 0.0
    face["w_limiter"] = True

    lagrangian_to_eulerian(**face, q=[])

    wcol = face["w"][ia + 1, ia + 1, :]
    assert wcol[0] == 2.0 * W_MAX_MAPZ, wcol
    assert np.all(wcol[1:] <= W_MAX_MAPZ + 1e-12), wcol
    # Momentum was genuinely LOST at the valve (non-conservation is the
    # documented intent of :408-416).
    dp2 = face["delp"][ia + 1, ia + 1, :]
    assert (wcol * dp2).sum() < 5.0e4 * dp2[km - 1] * 0.5


def test_driver_nh_w_limiter_clamps_and_conserves_momentum():
    """An interior w = 200 violates w_max = 90 (fv_mapz.F90:51); the
    limiter must clamp it and push the excess into the neighbours so
    the column integral w*dp2 is unchanged (momentum-conserving by
    construction; the top escape valve does not fire at these values)."""
    face = _nh_face(w_const=0.0)
    n, ng, km = face["n"], face["ng"], face["km"]
    ia = ng
    face["w"][ia + 2, ia + 3, 2] = 200.0
    face["ws"][:] = 0.0
    face["w_limiter"] = True

    lagrangian_to_eulerian(**face, q=[])

    wwin = face["w"][ia:ia + n, ia:ia + n, :]
    from legoesm.core.fv3_native_mapz import W_MAX_MAPZ
    assert wwin.max() <= W_MAX_MAPZ + 1e-12, wwin.max()
    # Momentum: recompute dp2 from the returned (Eulerian) delp.
    dp2 = face["delp"][ia:ia + n, ia:ia + n, :]
    col = (wwin * dp2).sum(axis=2)
    # The un-limited run carries the same remap; only the limiter differs.
    face2 = _nh_face(w_const=0.0)
    face2["w"][ia + 2, ia + 3, 2] = 200.0
    face2["ws"][:] = 0.0
    lagrangian_to_eulerian(**face2, q=[])
    w2 = face2["w"][ia:ia + n, ia:ia + n, :]
    col2 = (w2 * face2["delp"][ia:ia + n, ia:ia + n, :]).sum(axis=2)
    assert np.abs(col - col2).max() < 1e-9 * max(np.abs(col2).max(), 1.0)
    # Non-vacuity: the limiter really fired.
    assert w2.max() > W_MAX_MAPZ
    assert not np.array_equal(wwin, w2)
