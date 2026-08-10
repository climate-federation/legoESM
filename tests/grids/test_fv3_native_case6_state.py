"""Williamson case-6 (Rossby-Haurwitz wave 4) six-face IC certificate.

``case6_six_face_state`` is a literal port of the pinned oracle's
``tools/test_cases.F90:1213-1270``.  There is no zero-step Fortran dump
for this case yet (the Zenodo ``C48.sw.case6.alpha0.duo.hord8``
reference ships a day-100 RESTART and daily means, not an IC), so this
file certifies the two analytic formulas by INDEPENDENT routes rather
than against a binary fixture:

* the WIND is certified against the published Williamson et al. (1992)
  case-6 streamfunction

      psi = -a^2 w sin(p) + a^2 K cos^R(p) sin(p) cos(R L)

  differentiated NUMERICALLY (central differences in lon/lat), i.e.
  u = -(1/a) d(psi)/dp, v = (1/(a cos p)) d(psi)/dL.  The port's closed
  form is the analytic derivative of exactly this psi, so agreement to
  the finite-difference truncation error is a real cert: a flipped sign,
  a wrong power of cos, or R/2R swapped in either component fails it by
  orders of magnitude.  Both sides go through the SAME certified
  ``analytic_swcore_state`` covariant projection, so the comparison
  isolates the wind formula and nothing else (one variable).

* the HEIGHT is certified at TRANSCRIPTION level only, against the same
  A/B/C written in a different algebraic factoring
  (``cos^{2R-2}`` instead of ``cos^{2R} * cos^{-2}``, and the
  ``((R+1) cos p)^2`` square expanded).  That catches a mistyped
  coefficient or exponent; it does NOT independently establish the
  formula itself.  Stated here rather than implied -- the binary IC
  cert against a zero-step Fortran run is the follow-up that closes it.

* the gravity FLAVOUR is guarded: ``gh0 = 8e3 * Grav`` enters ``delp``
  additively (no cancellation, unlike case 2), so using
  ``legoesm.constants.g`` instead of the oracle's GFS ``FV3_GRAV``
  would shift every height by several m^2/s^2.
"""

import numpy as np
import pytest
from legoesm.core.fv3_native_duo_stepper import (
    build_six_face_duo_context,
    case6_six_face_state,
)
from legoesm.grids.fv3_native_gridstruct import (
    BIG_NUMBER,
    FV3_GRAV,
    FV3_OMEGA,
    FV3_RADIUS_M,
    analytic_swcore_state,
)

N = 12
NG = 3

# Sentinel detector.  MUST sit BELOW gridstruct.BIG_NUMBER (1.0e8) -- an
# earlier 1e10 threshold sat two orders above it and masked nothing.
_SENTINEL = 0.5 * BIG_NUMBER

# test_cases.F90:1216-1218
R_WAVE = 4.0
OMG = 7.848e-6
RK = 7.848e-6
GH0 = 8.0e3 * FV3_GRAV          # :1215


@pytest.fixture(scope="module")
def ctx():
    return build_six_face_duo_context(N, NG, ext_exclude=("divgd", "cvec"))


@pytest.fixture(scope="module")
def states(ctx):
    return case6_six_face_state(ctx)


def _psi(lon, lat):
    """Williamson et al. (1992) case-6 streamfunction."""
    a_r = FV3_RADIUS_M
    return (-a_r * a_r * OMG * np.sin(lat)
            + a_r * a_r * RK * np.cos(lat) ** R_WAVE * np.sin(lat)
            * np.cos(R_WAVE * lon))


def _psi_fd_wind(ll):
    """(u_east, v_north) from psi by CENTRAL DIFFERENCES.

    Deliberately NOT the analytic derivative -- that is what is under
    test.  h is at the sqrt(eps)-optimal scale for a second-order
    central difference on an O(1) angular argument.
    """
    lon = ll[..., 0]
    lat = ll[..., 1]
    a_r = FV3_RADIUS_M
    h = 1e-6
    dpsi_dlat = (_psi(lon, lat + h) - _psi(lon, lat - h)) / (2.0 * h)
    dpsi_dlon = (_psi(lon + h, lat) - _psi(lon - h, lat)) / (2.0 * h)
    uu = -dpsi_dlat / a_r
    vv = dpsi_dlon / (a_r * np.cos(lat))
    return uu, vv


def _gh_refactored(ll):
    """The case-6 height, same coefficients, different factoring."""
    lon = ll[..., 0]
    lat = ll[..., 1]
    a_r = FV3_RADIUS_M
    c = np.cos(lat)
    r = R_WAVE
    # A: cos^{2R} * cos^{-2} folded into cos^{2R-2}
    a_t = (0.5 * OMG * (2.0 * FV3_OMEGA + OMG) * c ** 2
           + 0.25 * RK * RK * ((r + 1.0) * c ** (2.0 * r + 2.0)
                               + (2.0 * r * r - r - 2.0) * c ** (2.0 * r)
                               - 2.0 * r * r * c ** (2.0 * r - 2.0)))
    # B: ((R+1) cos p)^2 expanded
    b_t = ((2.0 * (FV3_OMEGA + OMG) * RK / ((r + 1.0) * (r + 2.0)))
           * (c ** r * (r * r + 2.0 * r + 2.0)
              - (r + 1.0) ** 2 * c ** (r + 2.0)))
    c_t = 0.25 * RK * RK * ((r + 1.0) * c ** (2.0 * r + 2.0)
                            - (r + 2.0) * c ** (2.0 * r))
    gh = GH0 + a_r * a_r * (a_t
                            + b_t * np.cos(r * lon)
                            + c_t * np.cos(2.0 * r * lon))
    return gh, np.ones_like(gh)


def test_wind_matches_the_published_streamfunction(ctx, states):
    """The port's closed-form wind IS curl(psi): compare against a
    numerical curl of psi pushed through the SAME covariant projection.

    Scale is set by the field's own peak (RH-4 winds reach tens of m/s).
    The 1e-7 threshold is a GUARD tolerance, not the method's truncation
    bound -- observed agreement is ~1e-10 (codex r1 #9), so the guard
    has three orders of headroom while a formula error still fails by
    O(1).

    SCOPE (codex r1 #7): both sides run through the SAME
    analytic_swcore_state covariant projection, so this certifies the
    WIND FORMULA on top of that projection, and a defect INSIDE the
    projection (e.g. a reversed edge tangent) would cancel here.  The
    projection itself is certified independently and against a binary
    oracle: the DCMIP16_BC IC built through these same helpers matches
    the Fortran oracle's zero-step restart at ~1e-14 on all six faces
    (ic_face_map_parity), which a reversed tangent could not survive.
    """
    worst = 0.0
    for t, gs in enumerate(ctx["gs6"]):
        ref = analytic_swcore_state(gs, wind_fn=_psi_fd_wind,
                                    scalars_fn=_gh_refactored)
        for comp in ("u", "v"):
            got = np.asarray(states[t][comp])
            want = np.asarray(ref[comp])
            # Filter BOTH sides (the winds carry D-grid corner sentinels
            # on both sides; see the delp comment in the height test).
            ok = (np.isfinite(got) & np.isfinite(want)
                  & (np.abs(got) < _SENTINEL)
                  & (np.abs(want) < _SENTINEL))
            assert ok.sum() > 0
            scale = max(np.abs(want[ok]).max(), 1.0)
            rel = np.abs(got[ok] - want[ok]).max() / scale
            worst = max(worst, float(rel))
            assert rel < 1e-7, (t, comp, rel)
    # Non-vacuity: the compared fields are not trivially zero.
    peaks = []
    for t in range(6):
        uu = np.asarray(states[t]["u"])
        peaks.append(float(np.abs(uu[np.abs(uu) < _SENTINEL]).max()))
    assert max(peaks) > 10.0, peaks
    print(f"case6 wind vs numerical curl(psi): worst rel {worst:.3e}, "
          f"peak |u| {max(peaks):.3f} m/s")


def test_height_matches_the_refactored_coefficients(ctx, states):
    """Transcription-level cert of A, B, C (see module docstring)."""
    for t, gs in enumerate(ctx["gs6"]):
        ref = analytic_swcore_state(gs, wind_fn=_psi_fd_wind,
                                    scalars_fn=_gh_refactored)
        got = np.asarray(states[t]["delp"])
        want = np.asarray(ref["delp"])
        # Filter BOTH sides.  case6_six_face_state refills delp/pt on the
        # full lattice while the reference here still comes straight from
        # analytic_swcore_state, which leaves BIG_NUMBER in the corner
        # diagonals -- so a got-only filter compares 7.8e4 against 1e8 and
        # scores a rel of 0.999 on a field that is correct everywhere the
        # model reads it.
        ok = (np.isfinite(got) & np.isfinite(want)
              & (np.abs(got) < _SENTINEL) & (np.abs(want) < _SENTINEL))
        assert ok.sum() > 0
        rel = np.abs(got[ok] - want[ok]).max() / np.abs(want[ok]).max()
        assert rel < 1e-12, (t, rel)


def test_height_uses_the_oracle_gravity_flavour(ctx, states):
    """gh0 = 8e3 * Grav enters delp additively; certify the FLAVOUR by
    direct field comparison, not by a range check.

    (codex r1 #8: the previous version proved the two constants differ
    and then asserted a 70000..85000 range that BOTH flavours satisfy --
    it could not fail on the substitution it named.  This one rebuilds
    the expected field with each flavour and requires an exact match to
    the GFS one and a nonzero distance from the legoESM one.)
    """
    from legoesm import constants
    from legoesm.core.williamson_sw_analytic import (
        RH4_MEAN_DEPTH_M,
        rossby_haurwitz_4_geopotential,
    )

    assert abs(FV3_GRAV - 9.80665) < 1e-12   # const-ok: gfs_constants.h Grav
    assert abs(FV3_GRAV - constants.g) > 1e-4, (
        "FV3_GRAV and constants.g have converged; this guard is now "
        "vacuous and the flavour must be re-established another way")

    expected_shift = RH4_MEAN_DEPTH_M * abs(FV3_GRAV - constants.g)
    for t, gs in enumerate(ctx["gs6"]):
        got = np.asarray(states[t]["delp"])
        want_gfs = rossby_haurwitz_4_geopotential(
            gs["agrid_lon"], gs["agrid_lat"], radius=FV3_RADIUS_M,
            omega=FV3_OMEGA, gh0=RH4_MEAN_DEPTH_M * FV3_GRAV)
        want_lego = rossby_haurwitz_4_geopotential(
            gs["agrid_lon"], gs["agrid_lat"], radius=FV3_RADIUS_M,
            omega=FV3_OMEGA, gh0=RH4_MEAN_DEPTH_M * constants.g)
        ok = np.abs(got) < _SENTINEL
        assert ok.sum() > 0
        # Exact: the constructor IS this expression with the GFS flavour.
        assert np.array_equal(got[ok], want_gfs[ok]), t
        # And the wrong flavour is detectably different everywhere.
        d = np.abs(got[ok] - want_lego[ok]).min()
        assert d > 0.9 * expected_shift, (t, d, expected_shift)
    print(f"case6 gravity flavour: gh0={GH0:.6f}, "
          f"flavour separation {expected_shift:.4f} m^2/s^2")


def test_compute_domain_is_finite_and_positive(states):
    sl = slice(NG, NG + N)
    for t in range(6):
        delp = np.asarray(states[t]["delp"])[sl, sl]
        assert np.isfinite(delp).all()
        assert (delp > 0).all(), delp.min()
        for comp in ("u", "v"):
            f = np.asarray(states[t][comp])[sl, sl]
            assert np.isfinite(f).all()
            assert np.abs(f).max() < 200.0, np.abs(f).max()


def test_state_advances_one_step_conserving_mass(ctx, states):
    """The IC is USABLE by the stepper, and the first d_sw1/d_sw2 pair
    conserves mass exactly through the averaged cross-face fluxes -- the
    same killer invariant the case-2 state is held to."""
    from legoesm.core.fv3_native_duo_stepper import (
        csw_step_sixface,
        dsw12_step_sixface,
    )

    csw = csw_step_sixface(ctx, states, dt2=112.5)
    outs = dsw12_step_sixface(ctx, states, csw, dt=225.0)
    sl = slice(NG, NG + N)
    m0 = m1 = 0.0
    for t in range(6):
        area = ctx["gs6"][t]["area"][sl, sl]
        m0 += float((states[t]["delp"][sl, sl] * area).sum())
        m1 += float((outs[t]["delp"][sl, sl] * area).sum())
    assert np.isfinite(m1)
    assert abs(m1 - m0) / abs(m0) < 1e-13, (m0, m1, m1 - m0)
