"""DCMIP16_BC analytic profiles (test_cases.F90:6774-6853).

These are the pointwise formulas of the test_case=-13 initial condition. The
grid assembly is separate; what is testable here is the mathematics, plus the
handful of values the oracle pins exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.fv3_native_dcmip16_bc import (
    GFDL_CONSTANTS,
    GFS_CONSTANTS,
    P0_PA,
    PTROP_PA,
    QT,
    T0_K,
    _it,
    newton_height,
    pressure,
    sphum,
    temperature,
    uwind,
)

C = GFS_CONSTANTS   # the constant set the matched reference run was linked with


def test_t0_is_275_exactly():
    """:6490 -- T0 = 0.5*(Te+Tp) = 0.5*(310+240). The oracle's own comment
    says the DCMIP document gets this wrong, so it is worth pinning."""
    assert T0_K == 275.0


def test_pressure_at_the_surface_is_exactly_p0():
    """:6803 with z=0: Tir=0 so Ti1=Ti2=0 and p = p0*exp(0).

    True for EVERY latitude, which is what makes ps uniform (:6528-6532).
    """
    for lat in (0.0, 0.3, -0.7, 1.0, -1.2):
        assert pressure(0.0, lat, C) == P0_PA


def test_it_at_the_equator():
    """cos(0)=1 so IT = 1 - KK/(KK+2) = 1 - 3/5 = 0.4 exactly."""
    assert _it(0.0, C) == pytest.approx(0.4, abs=0.0, rel=1e-15)


def test_uwind_is_identically_zero_at_the_equator():
    """:6816 -- UU carries (cos^2 - cos^4), which is 0 at lat=0, so
    ur = -omega*R + sqrt((omega*R)^2) = 0. Exact, not approximate."""
    for z in (0.0, 1000.0, 15000.0):
        t = temperature(z, 0.0, C)
        assert uwind(z, t, 0.0, C) == 0.0


def test_newton_height_inverts_the_pressure_profile():
    """The solve is the inverse of `pressure`, so round-tripping must return
    the target to the convergence tolerance.

    Checked in PRESSURE space: 1e-6 m of height is ~1e-5 Pa here, so a
    height-space tolerance would be the weaker statement.
    """
    for lat in (0.0, np.pi / 4, -np.pi / 4, 1.3):
        for p_target in (5.0e4, 6.0e4, 7.0e4, 8.0e4, 9.0e4):
            z = newton_height(p_target, lat, 0.0, C)
            assert pressure(z, lat, C) == pytest.approx(p_target, rel=1e-12)
            assert z > 0.0          # every target is above the surface


def test_newton_height_is_monotone_in_pressure():
    """Lower pressure must be higher up. Catches a sign error in the update
    that a single round-trip could still satisfy."""
    lat = np.pi / 4
    z = [newton_height(p, lat, 0.0, C) for p in (9e4, 8e4, 7e4, 6e4, 5e4)]
    assert all(b > a for a, b in zip(z, z[1:]))


def test_sphum_is_the_stratospheric_floor_above_the_tropopause():
    """:6849 -- hard switch at p > ptrop, no blending."""
    assert sphum(PTROP_PA, P0_PA, 0.0, C) == QT          # not >, so floor
    assert sphum(PTROP_PA * 0.5, P0_PA, 0.0, C) == QT
    assert sphum(PTROP_PA * 1.001, P0_PA, 0.0, C) > QT   # just inside moist


def test_sphum_peaks_at_the_equatorial_surface():
    """q0*exp(-(lat/phiW)^4)*exp(-((eta-1)*p0/pw)^2): both factors are 1 at
    lat=0, p=ps, so the value there is exactly q0."""
    from legoesm.core.fv3_native_dcmip16_bc import Q0
    assert sphum(P0_PA, P0_PA, 0.0, C) == pytest.approx(Q0, rel=1e-15)
    # and it decreases away from the equator
    assert sphum(P0_PA, P0_PA, 0.8, C) < sphum(P0_PA, P0_PA, 0.2, C)


def test_temperature_is_between_the_pole_and_equator_parameters():
    """Tp=240, Te=310 bracket the surface field; a sign slip in IT would
    push values outside immediately."""
    for lat in (0.0, 0.5, 1.0, -1.0):
        t = temperature(0.0, lat, C)
        assert 230.0 < t < 320.0


def test_the_two_fms_constant_sets_give_different_answers():
    """The whole reason the constants are an argument.

    If these ever agree, the confound guard has been defeated and a port
    could be compared against the wrong planet without noticing.
    """
    lat = np.pi / 4
    z = 5000.0
    assert pressure(z, lat, GFS_CONSTANTS) != pressure(z, lat, GFDL_CONSTANTS)
    assert temperature(z, lat, GFS_CONSTANTS) != temperature(z, lat, GFDL_CONSTANTS)
    # and the difference is FAR above any parity tolerance -- this is why it
    # matters (radius differs by 200 m, gravity by 6.8e-4 relative)
    rel = abs(pressure(z, lat, GFS_CONSTANTS) / pressure(z, lat, GFDL_CONSTANTS) - 1)
    assert rel > 1e-6, f"constant sets differ by only {rel:.3g}"


def test_zvir_and_rp_follow_the_oracle_definitions():
    """:6525 zvir = rvgas/rdgas - 1;  :6496 Rp = radius/10."""
    assert GFS_CONSTANTS.zvir == 461.50 / 287.05 - 1.0   # const-ok: oracle FMS values
    assert GFS_CONSTANTS.rp == 6.3712e6 / 10.0           # const-ok: oracle FMS values
    # ppcenter is (20E, 40N) in RADIANS
    lon_c, lat_c = GFS_CONSTANTS.ppcenter
    assert lon_c == pytest.approx(np.deg2rad(20.0), rel=1e-15)
    assert lat_c == pytest.approx(np.deg2rad(40.0), rel=1e-15)


def test_gfdl_reference_column_at_45n():
    """Independent cross-check.

    These interface heights were computed by a separate agent from the same
    Fortran, in float64 with the GFDL constant set and the npz=5 ak/bk
    (pe = 50000..100000 Pa). Agreeing with an independently-derived column is
    a stronger statement about the formulas than any self-consistency test in
    this file.
    """
    lat = np.deg2rad(45.0)
    pe = np.array([50000.0, 60000.0, 70000.0, 80000.0, 90000.0, 100000.0])
    want = np.array([5379.7676, 4017.9501, 2837.6363, 1793.0448, 854.1188, 0.0])

    z = np.zeros(6)
    for k in range(4, -1, -1):                 # march upward from the surface
        z[k] = newton_height(pe[k], lat, z[k + 1], GFDL_CONSTANTS)

    np.testing.assert_allclose(z, want, atol=1e-3, rtol=0)


def test_the_profiles_are_vectorised_over_z():
    """The assembly evaluates whole columns/faces at once; a formula that
    only works on scalars would force a Python loop over 48x48x6."""
    z = np.array([0.0, 1000.0, 5000.0, 12000.0])
    lat = np.full_like(z, 0.3)
    assert pressure(z, lat, C).shape == z.shape
    assert temperature(z, lat, C).shape == z.shape
    assert np.all(np.diff(pressure(z, lat, C)) < 0.0)   # p decreases with z
