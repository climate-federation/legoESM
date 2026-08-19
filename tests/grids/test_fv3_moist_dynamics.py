"""The moist (zvir) coupling in the NumPy SPECIFICATION lane.

Phase 1: hydrostatic virtual-temperature coupling only. ``consv_te``
(the energy fixer, fv_mapz.F90:628-747) and the NH moist ``pkz``
(fv_dynamics.F90:307-309) stay refused, and the refusals are gated
here so an unported arm cannot run silently.

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
    """dp1=None must remain a single division, not a multiply by 1.

    The whole point of the separate branch: `(1.0 + 0.0)/pkz` computes a
    reciprocal and then multiplies -- two roundings where the dry lane
    has one -- so a zeros-array "equivalent" is NOT bit-identical.
    """
    pt_a, pkz = _pt_pkz()
    pt_b = pt_a.copy()
    pt_to_theta_v(pt_a, pkz, n=N, ng=NG)                       # dry arm
    pt_to_theta_v(pt_b, pkz, n=N, ng=NG,
                  dp1=np.zeros((N, N, KM)))                    # moist arm, 0
    win = (slice(NG, NG + N), slice(NG, NG + N), slice(None))
    # NOT asserted equal: `assert X or True` is a test that cannot fail,
    # and the two arms are NOT required to agree bitwise -- that is the
    # entire reason the dry branch exists. The contract below is the one
    # that must hold.
    del pt_b
    pt_c, _ = _pt_pkz()
    expect = pt_c[win] / pkz
    pt_d = pt_c.copy()
    pt_to_theta_v(pt_d, pkz, n=N, ng=NG)
    assert np.array_equal(pt_d[win], expect), (
        "the dry lane is no longer a single division -- the certified "
        "1.1866e-09 parity was measured on this exact arithmetic")


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


def test_no_tracers_with_zvir_is_refused():
    from legoesm.core.fv3_native_dynamics import fv_dynamics_step
    with pytest.raises(ValueError, match="no tracers|nq > 0"):
        fv_dynamics_step(**_minimal_step_kwargs(q=[[] for _ in range(6)]))


def test_unported_moist_arms_still_refuse():
    """consv_te and the NH moist pkz must RAISE, not be greppable."""
    from legoesm.core.fv3_native_dynamics import fv_dynamics_step
    with pytest.raises(NotImplementedError, match="consv_te"):
        fv_dynamics_step(**_minimal_step_kwargs(zvir=0.0, consv_te=1.0))
    with pytest.raises(NotImplementedError, match="307-309|non-hydrostatic"):
        fv_dynamics_step(**_minimal_step_kwargs(hydrostatic=False))
