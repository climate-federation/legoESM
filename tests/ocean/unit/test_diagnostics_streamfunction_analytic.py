"""Analytic ground-truth tests for the MHT / MOC / BSF section integrals.

Written to settle whether a day-1 ``MHT NH peak = 19.18 PW`` (obs ~1.8 PW) and
``global-MOC [-401, 768] Sv`` come from a defect in the DIAGNOSTIC or from the
cold-start transient in the VELOCITY field.  The verdict must not rest on "the
number looks wrong", so every case here has a value computable by hand.

The decisive one is ``test_mht_is_reference_independent_only_when_mass_balanced``:
``meridional_heat_transport`` integrates ``rho0*cp*v*theta_v*h_v*dx_v`` with
theta in **degC**, which is the conventional choice ONLY because a full zonal
integral at a latitude carries ~zero net mass flux.  When the net mass flux is
NOT zero -- exactly the cold-start barotropic state -- the degC reference adds a
spurious ``rho0*cp*theta_ref*(net volume flux)`` term.  These tests measure that
term rather than asserting it.
"""

from __future__ import annotations

import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.diagnostics_streamfunction import (
    barotropic_streamfunction,
    meridional_heat_transport,
    moc_streamfunction,
)

N_LAT, N_LON, NLEV = 8, 12, 3
_PW = 1.0e15
_SV = 1.0e6


class _Grid:
    """Minimal regular lat-lon grid proxy (the analytic branch of the helper)."""

    radius = constants.R_earth
    dlon = 2.0 * np.pi / N_LON
    dlat = np.pi / N_LAT

    def __init__(self):
        self.lat = np.linspace(-np.pi / 2 + 0.5 * self.dlat,
                               np.pi / 2 - 0.5 * self.dlat, N_LAT)
        self.lat_v = np.concatenate([[self.lat[0] - 0.5 * self.dlat],
                                     self.lat + 0.5 * self.dlat])
        self.dy = np.full((N_LAT,), self.radius * self.dlat)


def _uniform_fields(v0=0.1, theta0=10.0, h0=100.0):
    v = np.full((N_LAT + 1, N_LON, NLEV), v0)
    theta = np.full((N_LAT, N_LON, NLEV), theta0)
    h = np.full((N_LAT, N_LON, NLEV), h0)
    mask = np.ones((N_LAT, N_LON))
    return v, theta, h, mask


# ------------------------------------------------------------------ MHT


def test_mht_matches_the_hand_computed_value_on_a_uniform_section():
    """MHT(j) = rho0*cp*v*theta*h*(sum_x dx_v)*nlev, computable by hand.

    Pins rho*cp applied ONCE, the thickness weighting, and the zonal dx
    integral.  A missing h or dx, or a doubled rho*cp, fails here.
    """
    g = _Grid()
    v0, theta0, h0 = 0.1, 10.0, 100.0
    v, theta, h, mask = _uniform_fields(v0, theta0, h0)
    mht_PW, lat_deg = meridional_heat_transport(v, theta, h, mask, g)

    j = N_LAT // 2                       # an interior v-row (poles are zeroed)
    dx = g.radius * g.dlon * np.cos(g.lat_v[j])
    expect_W = (constants.rho_ocean * constants.c_sw
                * v0 * theta0 * h0 * dx * N_LON * NLEV)
    assert mht_PW[j] == pytest.approx(expect_W / _PW, rel=1e-12)
    assert lat_deg.shape == (N_LAT + 1,)


def test_mht_scales_linearly_in_each_factor():
    """Doubling v, theta, or h must exactly double MHT (no hidden square)."""
    g = _Grid()
    v, theta, h, mask = _uniform_fields()
    b = meridional_heat_transport(v, theta, h, mask, g)[0][N_LAT // 2]
    for scaled in (
        meridional_heat_transport(2 * v, theta, h, mask, g)[0][N_LAT // 2],
        meridional_heat_transport(v, 2 * theta, h, mask, g)[0][N_LAT // 2],
        meridional_heat_transport(v, theta, 2 * h, mask, g)[0][N_LAT // 2],
    ):
        assert scaled == pytest.approx(2.0 * b, rel=1e-12)


def test_mht_poles_carry_no_flux():
    g = _Grid()
    v, theta, h, mask = _uniform_fields()
    mht_PW, _ = meridional_heat_transport(v, theta, h, mask, g)
    assert mht_PW[0] == 0.0 and mht_PW[-1] == 0.0


def test_mht_is_reference_independent_only_when_mass_balanced():
    """THE decisive test for the 19 PW question.

    With ZERO net mass flux at a latitude, adding a constant to theta must not
    change MHT (degC vs K is then arbitrary).  With NON-zero net mass flux --
    the cold-start barotropic state -- the shift changes MHT by EXACTLY
    rho0*cp*dTheta*(net volume flux), which is the spurious term.
    """
    g = _Grid()
    _, theta, h, mask = _uniform_fields()
    j = N_LAT // 2
    dTheta = 5.0

    # (a) mass-balanced: half the cells flow north, half south
    v_bal = np.zeros((N_LAT + 1, N_LON, NLEV))
    v_bal[:, : N_LON // 2, :] = +0.1
    v_bal[:, N_LON // 2:, :] = -0.1
    a0 = meridional_heat_transport(v_bal, theta, h, mask, g)[0][j]
    a1 = meridional_heat_transport(v_bal, theta + dTheta, h, mask, g)[0][j]
    assert a1 == pytest.approx(a0, abs=1e-12), (
        "with zero net mass flux the degC reference must not matter")

    # (b) NOT mass-balanced: uniform northward flow
    v_net = np.full((N_LAT + 1, N_LON, NLEV), 0.1)
    b0 = meridional_heat_transport(v_net, theta, h, mask, g)[0][j]
    b1 = meridional_heat_transport(v_net, theta + dTheta, h, mask, g)[0][j]
    dx = g.radius * g.dlon * np.cos(g.lat_v[j])
    net_vol = 0.1 * 100.0 * dx * N_LON * NLEV                  # m^3/s
    expect = constants.rho_ocean * constants.c_sw * dTheta * net_vol / _PW
    assert (b1 - b0) == pytest.approx(expect, rel=1e-10)
    assert abs(b1 - b0) > 0.0, "the unbalanced case must be reference-dependent"


def test_spurious_mht_per_sverdrup_of_net_mass_flux():
    """Quantify the sensitivity: PW of spurious MHT per Sv of net mass flux.

    This is the conversion that decides whether a 19 PW day-1 value is
    explicable by the observed hundreds of Sv of cold-start transport.
    """
    theta_ref = 15.0                       # representative upper-ocean degC
    pw_per_sv = constants.rho_ocean * constants.c_sw * theta_ref * _SV / _PW
    # ~0.061 PW per Sv at 15 degC
    assert 0.05 < pw_per_sv < 0.07
    # 19 PW would then need this many Sv of NET (not gross) mass flux:
    sv_needed = 19.0 / pw_per_sv
    assert 250.0 < sv_needed < 400.0


# ------------------------------------------------------------------ MOC


def test_moc_matches_the_hand_computed_value_on_a_uniform_section():
    """psi(j,k) = -cumsum_z(sum_x v*h_v*dx_v); top level is one layer's worth."""
    g = _Grid()
    v0, h0 = 0.1, 100.0
    v, theta, h, mask = _uniform_fields(v0, 10.0, h0)
    psi = moc_streamfunction(v, h, None, None, mask, g)
    j = N_LAT // 2
    dx = g.radius * g.dlon * np.cos(g.lat_v[j])
    one_layer_Sv = v0 * h0 * dx * N_LON / _SV
    assert abs(psi[j, 0]) == pytest.approx(one_layer_Sv, rel=1e-10)
    assert abs(psi[j, -1]) == pytest.approx(NLEV * one_layer_Sv, rel=1e-10)


def test_moc_magnitude_is_linear_in_velocity():
    """A 100x velocity gives a 100x psi -- so O(100) Sv psi implies O(100x)
    excess velocity, not a diagnostic normalisation error."""
    g = _Grid()
    v, theta, h, mask = _uniform_fields()
    p1 = np.abs(moc_streamfunction(v, h, None, None, mask, g)).max()
    p2 = np.abs(moc_streamfunction(100 * v, h, None, None, mask, g)).max()
    assert p2 == pytest.approx(100.0 * p1, rel=1e-10)


# ------------------------------------------------------------------ BSF


def test_bsf_matches_the_hand_computed_value_on_a_uniform_section():
    """psi_bt = -cumsum_lat(sum_z h_u*u * dy); uniform u gives a linear ramp."""
    g = _Grid()
    u0, h0 = 0.1, 100.0
    u = np.full((N_LAT, N_LON + 1, NLEV), u0)
    h = np.full((N_LAT, N_LON, NLEV), h0)
    mask = np.ones((N_LAT, N_LON))
    psi = barotropic_streamfunction(u, h, mask, g)
    dy = g.radius * g.dlat * 0.5
    step_Sv = u0 * h0 * NLEV * dy / _SV
    d = np.diff(psi[:, 0])
    assert np.allclose(d, d[0], rtol=1e-10)
    assert abs(d[0]) == pytest.approx(step_Sv, rel=1e-10)


def test_bsf_is_linear_in_velocity():
    g = _Grid()
    h = np.full((N_LAT, N_LON, NLEV), 100.0)
    mask = np.ones((N_LAT, N_LON))
    u = np.full((N_LAT, N_LON + 1, NLEV), 0.1)
    p1 = np.abs(barotropic_streamfunction(u, h, mask, g)).max()
    p2 = np.abs(barotropic_streamfunction(50 * u, h, mask, g)).max()
    assert p2 == pytest.approx(50.0 * p1, rel=1e-10)
