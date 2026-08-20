"""The moist (zvir) coupling in the NumPy SPECIFICATION lane.

Phase 1 was hydrostatic virtual-temperature coupling; the NH arm
followed (fv_dynamics.F90:299-322 recomputes ``pkz`` with the
``(1+dp1)`` factor inside its log).  ``consv_te`` (the energy fixer,
fv_mapz.F90:628-747) stays refused, and the refusal is gated here so an
unported arm cannot run silently.

THE REGRESSION GATE THAT MATTERS MOST: enabling this path must leave
the ADIABATIC lane BIT-IDENTICAL. That lane carries a certified
1.1866e-09 agreement with the pinned Fortran; if ``zvir = 0`` stopped
taking ``pt /= pkz`` and started taking ``pt *= (1.+0)/pkz``, the
result would round twice and the certification would silently move.
``test_dry_lane_is_bitwise_under_the_moist_patch`` pins that.
"""
from __future__ import annotations

import numpy as np
import pytest
from legoesm.core.fv3_native_dynamics import pt_to_theta_v

N, NG, KM = 4, 3, 5
MA = N + 2 * NG


def _pt_pkz(seed=3):
    rng = np.random.default_rng(seed)
    pt = 280.0 + 5.0 * rng.standard_normal((MA, MA, KM))
    pkz = 0.9 + 0.2 * rng.random((N, N, KM))
    return pt, pkz


def test_dry_lane_is_bitwise_under_the_moist_patch():
    """The dry lane must still be EXACTLY ``pt/pkz``.

    THE OLD DOCSTRING HERE WAS FALSE (job 9442478). It said a
    zeros-array equivalent "computes a reciprocal and then multiplies --
    two roundings where the dry lane has one -- so it is NOT
    bit-identical". That described the OLD association
    ``pt *= (1.0+dp1)/pkz``; once it was corrected to the oracle's
    ``(pt*(1+dp1))/pkz``, ``1.0+0.0`` is exactly 1.0 and ``win*1.0`` is
    exact, so the zeros array IS bit-identical. Both facts are asserted
    below instead of described.
    """
    win = (slice(NG, NG + N), slice(NG, NG + N), slice(None))

    # 1. the dry arm is a single division -- the arithmetic the
    #    certified 1.1866e-09 parity was measured on.
    pt_a, pkz = _pt_pkz()
    expect = pt_a[win] / pkz
    pt_dry = pt_a.copy()
    pt_to_theta_v(pt_dry, pkz, n=N, ng=NG)
    assert np.array_equal(pt_dry[win], expect), (
        "the dry lane is no longer a single division -- the certified "
        "1.1866e-09 parity was measured on this exact arithmetic")

    # 2. and a zeros dp1 now agrees with it BITWISE. If this ever goes
    #    red the moist association has changed and the dry lane moved
    #    with it, which is the failure this file exists to catch.
    pt_zero = pt_a.copy()
    pt_to_theta_v(pt_zero, pkz, n=N, ng=NG, dp1=np.zeros((N, N, KM)))
    assert np.array_equal(pt_zero[win], pt_dry[win])

    # 3. ANTI-VACUITY: a real dp1 must move it.
    pt_wet = pt_a.copy()
    pt_to_theta_v(pt_wet, pkz, n=N, ng=NG,
                  dp1=np.full((N, N, KM), 0.0077))
    assert not np.array_equal(pt_wet[win], pt_dry[win])


def test_moist_arm_matches_the_oracle_expression():
    """pt = pt*(1.+dp1)/pkz — fv_dynamics.F90:402, USE_COND undefined."""
    pt, pkz = _pt_pkz(seed=7)
    rng = np.random.default_rng(11)
    dp1 = 0.61 * (1e-3 * rng.random((N, N, KM)))     # zvir*q, q ~ 1e-3
    win = (slice(NG, NG + N), slice(NG, NG + N), slice(None))
    expect = pt[win] * (1.0 + dp1) / pkz
    got = pt.copy()
    pt_to_theta_v(got, pkz, n=N, ng=NG, dp1=dp1)
    assert np.array_equal(got[win], expect)
    # ...and it must actually DIFFER from the dry answer, or the test
    # certifies nothing about the coupling.
    dry = pt.copy()
    pt_to_theta_v(dry, pkz, n=N, ng=NG)
    assert not np.array_equal(got[win], dry[win]), "vacuous: dp1 did nothing"


def test_halo_is_untouched_by_either_arm():
    """Both arms are COMPUTE-WINDOW ONLY (the oracle loops is..ie)."""
    for dp1 in (None, np.full((N, N, KM), 1e-3)):
        pt, pkz = _pt_pkz(seed=5)
        before = pt.copy()
        pt_to_theta_v(pt, pkz, n=N, ng=NG, dp1=dp1)
        halo = np.ones((MA, MA), bool)
        halo[NG:NG + N, NG:NG + N] = False
        assert np.array_equal(pt[halo, :], before[halo, :])


def _minimal_step_kwargs(**over):
    """Arguments that reach the entry guards and nothing beyond them.

    The guards under test all fire before ctx/state are read (the only
    prior checks are k_split/n_split, the six-face length check, and
    require_uniform_damping_lane), so dummies suffice -- and if a guard
    is ever MOVED below the dynamics, these calls stop raising, which is
    exactly what a source-grep test could not detect.
    """
    six = [{} for _ in range(6)]
    kw = dict(ctx={}, state=six, press=[{} for _ in range(6)],
              bdt=60.0, km=KM, k_split=1, n_split=2, ptop=100.0,
              ak=np.zeros(KM + 1), bk=np.zeros(KM + 1), akap=2.0 / 7.0,
              cp_air=1004.6, kord_mt=9, kord_tm=-9, kord_tr=9,
              q=[[np.zeros((MA, MA, KM))] for _ in range(6)],
              zvir=0.61, sphum_index=0, hydrostatic=True)
    kw.update(over)
    return kw


@pytest.mark.parametrize("bad,match", [
    (None, "sphum_index"),
    (True, "sphum_index"),
    (1.5, "sphum_index"),
    (-1, "out of range"),
    (99, "out of range"),
])
def test_sphum_index_is_validated_not_guessed(bad, match):
    """BEHAVIOURAL (codex MAJOR 2026-08-19): the first version of this
    test parametrized five bad values, never passed them anywhere, and
    grepped the source instead -- it would have passed with the guard
    deleted. It now calls the function."""
    from legoesm.core.fv3_native_dynamics import fv_dynamics_step
    with pytest.raises(ValueError, match=match):
        fv_dynamics_step(**_minimal_step_kwargs(sphum_index=bad))


def test_sphum_index_accepts_a_numpy_integer():
    """``np.int64`` is what a config-driven caller carries.

    The guard was widened from ``isinstance(int)`` to
    ``operator.index``; rejecting a numpy integer would be a spurious
    raise, and nothing else exercises the dynamics-lane guard's ACCEPT
    path (GLM MINOR, job 9442483).  Reaching past the guard is enough:
    the minimal fixture cannot run a step, so any later failure is not
    a ValueError about sphum_index.
    """
    from legoesm.core.fv3_native_dynamics import fv_dynamics_step
    with pytest.raises(Exception) as ei:
        fv_dynamics_step(**_minimal_step_kwargs(sphum_index=np.int64(0)))
    assert not (isinstance(ei.value, ValueError)
                and "sphum_index" in str(ei.value)), \
        f"numpy integer was rejected by the index guard: {ei.value}"


def test_no_tracers_with_zvir_is_refused():
    from legoesm.core.fv3_native_dynamics import fv_dynamics_step
    with pytest.raises(ValueError, match="no tracers|nq > 0"):
        fv_dynamics_step(**_minimal_step_kwargs(q=[[] for _ in range(6)]))


def test_unported_moist_arms_still_refuse():
    """consv_te must RAISE, not be greppable.

    The NH moist arm USED to be here; it is enabled now (the pkz is
    recomputed with the (1+dp1) factor inside its log,
    fv_dynamics.F90:299-322).  What it leaves behind is a requirement,
    not a refusal: the NH arm needs ``delz`` on every face, and the
    minimal fixture carries none, so it must be caught rather than
    indexed into.
    """
    from legoesm.core.fv3_native_dynamics import fv_dynamics_step
    with pytest.raises(NotImplementedError, match="consv_te"):
        fv_dynamics_step(**_minimal_step_kwargs(zvir=0.0, consv_te=1.0))
    with pytest.raises(ValueError, match="delz"):
        fv_dynamics_step(**_minimal_step_kwargs(hydrostatic=False))
