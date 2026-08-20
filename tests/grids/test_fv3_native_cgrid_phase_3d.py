"""Unit 3: the km-general six-face C-grid phase.

This composes kernels that are already bit-exact against the Fortran, so
what needs testing is the COMPOSITION: that the k cadence is right, that
per-level results land in the right level, that stagger mismatches raise
rather than broadcast, and that km=1 through this path reproduces the
certified km=1 kernel call exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.fv3_native_cgrid_phase_3d import CSW_OUT_2D, csw_phase_3d
from legoesm.core.fv3_native_state_3d import build_state_3d, field_shape

N, NG = 12, 3   # 2*ng < n required, and the context widens ng to 4
KM = 3


def _compute(ctx, a):
    """Restrict to the COMPUTE window.

    build_kinked_corner_lonlat writes `sentinel` into the corner-diagonal
    halo regions, so c_sw output there is legitimately non-finite. Scoring
    the full padded array would (a) fail a finiteness assertion for the
    wrong reason and (b) make an equality assertion vacuous, because
    assert_array_equal treats NaN == NaN as equal -- a comparison over
    mostly-NaN halos can pass while proving nothing.
    """
    bd = ctx["bd"]
    i0, j0 = bd.is_ - bd.isd, bd.js - bd.jsd
    ni, nj = bd.ie - bd.is_ + 1, bd.je - bd.js + 1
    return a[i0:i0 + ni, j0:j0 + nj]


@pytest.fixture(scope="module")
def ctx():
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    return build_six_face_duo_context(N, NG, oracle_conventions=True)


def _seeded_state(km, seed=0):
    """Distinct values per face AND per level, so a face mix-up or a level
    mix-up cannot hide behind symmetry."""
    rng = np.random.default_rng(seed)
    st = build_state_3d(N, NG, km)
    for t, face in enumerate(st):
        for name in face:
            a = face[name]
            for k in range(km):
                a[:, :, k] = (1.0 + 0.01 * t + 0.1 * k
                              + 1e-3 * rng.standard_normal(a.shape[:2]))
        face["delp"][:] = np.abs(face["delp"]) + 1.0   # positive thickness
        face["pt"][:] = np.abs(face["pt"]) + 280.0
    return st


def test_shapes_and_levels(ctx):
    st = _seeded_state(KM)
    outs = csw_phase_3d(ctx, st, dt2=10.0, km=KM)
    assert len(outs) == 6
    for acc in outs:
        assert set(acc) == set(CSW_OUT_2D)
        assert acc["delpc"].shape == field_shape("delp", N, NG, KM)
        assert acc["uc"].shape == field_shape("uc", N, NG, KM)
        assert acc["vc"].shape == field_shape("vc", N, NG, KM)
        assert acc["divg_d"].shape == field_shape("divgd", N, NG, KM)
        for name, a in acc.items():
            assert a.dtype == np.float64
            win = _compute(ctx, a)
            assert np.all(np.isfinite(win)), (
                f"{name}: non-finite inside the compute window")


def test_each_level_is_computed_from_its_own_level_only(ctx):
    """THE cadence test. Perturb ONE level of the input; only that level
    of the output may move. If the k-loop leaked (or `fort` broadcast a
    column), other levels would change too."""
    st = _seeded_state(KM)
    base = csw_phase_3d(ctx, st, dt2=10.0, km=KM)

    st2 = _seeded_state(KM)
    st2[0]["delp"][:, :, 1] += 0.5           # face 1, level 1 only
    pert = csw_phase_3d(ctx, st2, dt2=10.0, km=KM)

    d = np.abs(_compute(ctx, pert[0]["delpc"])
               - _compute(ctx, base[0]["delpc"]))
    assert d[:, :, 1].max() > 1e-12, "level 1 must respond"
    assert d[:, :, 0].max() == 0.0, "level 0 must be untouched"
    assert d[:, :, 2].max() == 0.0, "level 2 must be untouched"


def test_perturbing_one_face_does_not_change_another(ctx):
    """c_sw is face-local; cross-face coupling only enters at the halo
    exchanges and the barriers, which are units 4-5."""
    st = _seeded_state(KM)
    base = csw_phase_3d(ctx, st, dt2=10.0, km=KM)
    st2 = _seeded_state(KM)
    st2[2]["pt"][:, :, 0] += 1.0
    pert = csw_phase_3d(ctx, st2, dt2=10.0, km=KM)
    assert np.abs(_compute(ctx, pert[2]["ptc"])
                  - _compute(ctx, base[2]["ptc"])).max() > 1e-12
    for t in (0, 1, 3, 4, 5):
        assert np.array_equal(_compute(ctx, pert[t]["ptc"]),
                              _compute(ctx, base[t]["ptc"]))


def test_km1_through_the_3d_path_matches_the_certified_2d_kernel(ctx):
    """Stage-3 style check against the thing that carries the oracle
    certificate: the 3-D assembler must add cadence, not change math."""
    from legoesm.core.fv3_native_sw_core import c_sw

    st = _seeded_state(1, seed=3)
    outs = csw_phase_3d(ctx, st, dt2=7.5, km=1)
    for t in range(6):
        direct = c_sw(st[t]["delp"][:, :, 0], st[t]["pt"][:, :, 0],
                      st[t]["w"][:, :, 0], st[t]["u"][:, :, 0],
                      st[t]["v"][:, :, 0], ctx["gs6"][t], ctx["bd"],
                      N + 1, N + 1, 7.5, nord=2, duogrid=True)
        for name in CSW_OUT_2D:
            got, want = outs[t][name][:, :, 0], direct[name]
            # Non-vacuity: assert_array_equal treats NaN == NaN as equal,
            # so require the compute window to be FINITE before trusting
            # the equality over the padded array.
            assert np.all(np.isfinite(_compute(ctx, want))), (
                f"face {t + 1} {name}: reference is non-finite in the "
                f"compute window; the equality below would be vacuous")
            np.testing.assert_array_equal(
                got, want,
                err_msg=f"face {t + 1} {name}: 3-D path diverged from the "
                        f"certified 2-D kernel at km=1")


def test_duogrid_flag_is_actually_forwarded():
    """c_sw's `duogrid` DEFAULTS TO FALSE. This session already lost a
    measurement to an unpassed default, so prove the flag reaches the
    kernel.

    Builds its OWN plain context instead of using the module fixture,
    because the fixture is bounded and `d2a2c_vect` has no bounded port.

    THE IMPLICATION IS ONE-WAY. fv_arrays.F90:1512 is
    `bounded_domain = regional .or. nested .or. duogrid`, so duogrid
    forces bounded -- but bounded does NOT force duogrid: regional and
    nested runs are bounded with duogrid false. An earlier version of
    this docstring claimed `duogrid=False, bounded_domain=True` was
    unreachable, which reverses the implication; that pair is exactly the
    regional/nested category, and it is the category `d2a2c_vect`'s
    NotImplementedError legitimately marks as un-ported (upstream calls
    it with a literal .false. and routes bounded non-duo through
    divergence_corner_nest, sw_core.F90:150-160).

    km=1 corpus migration (2026-08-11): c_sw now REFUSES the
    upstream-impossible `duogrid=T, bounded=F` pair, so the old
    outputs-differ A/B can no longer run on a plain context.  The
    forwarding proof is now the raise itself: it is ONE-VARIABLE (the
    duogrid argument alone separates the arm that runs from the arm
    that raises), and the guard sits inside c_sw, so it can only fire
    if csw_phase_3d handed the flag down to c_sw.

    SCOPE (GLM r1 finding 2): this certifies the 3-D ASSEMBLER ->
    c_sw hop only — the hop this test was written for (an unpassed
    default in csw_phase_3d).  Forwarding INSIDE c_sw to
    d2a2c_vect_duo/divergence_corner_duo is certified separately by
    test_duo_branch_engages (spy + consumption) and by the bit-exact
    c_sw oracle.
    """
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    ctx = build_six_face_duo_context(N, NG)
    st = _seeded_state(KM, seed=5)
    off = csw_phase_3d(ctx, _seeded_state(KM, seed=5), dt2=10.0, km=KM,
                       duogrid=False)
    assert np.all(np.isfinite(_compute(ctx, off[0]["delpc"])))
    with pytest.raises(ValueError, match=r"fv_arrays\.F90:1512"):
        csw_phase_3d(ctx, st, dt2=10.0, km=KM, duogrid=True)


def test_km_above_the_remap_window_is_refused(ctx):
    with pytest.raises(ValueError, match="fv_mapz"):
        csw_phase_3d(ctx, build_state_3d(N, NG, 4), dt2=1.0, km=8)
