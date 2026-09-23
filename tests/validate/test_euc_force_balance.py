"""Manufactured-solution tests for the EUC force-balance probe's operators.

A probe is untrusted code until it reproduces answers that are known without
it.  The two operators that carry every number this probe prints are a zonal
derivative on a C grid and a vertical stress divergence, and both have closed
forms on simple fields.  Each test below pins one of those; the pair marked
NON-VACUITY exists to show the tests can actually fail, since a derivative
test that passes when the metric is wrong proves nothing.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_SRC = (pathlib.Path(__file__).resolve().parents[2]
        / "scripts" / "validate" / "ocean_fidelity" / "euc_force_balance.py")


def _mod():
    import sys
    sys.path.insert(0, str(_SRC.parent))
    spec = importlib.util.spec_from_file_location("euc_force_balance", _SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()


def test_zonal_derivative_is_exact_on_a_linear_field():
    """A field rising by a fixed amount per cell has a constant gradient."""
    nj, ni = 3, 12
    dx = 4000.0
    slope = 0.25                                   # units per metre of x
    x = np.arange(ni) * dx
    phi = np.tile(x * slope, (nj, 1))
    dxw = np.full((nj, ni), dx)
    got = M._dphi_dx(phi, dxw, dxw)
    # The periodic wrap makes the first and last columns meaningless; the
    # interior must be exact.
    assert np.allclose(got[:, 1:-1], slope, rtol=0, atol=1e-12)


def test_zonal_derivative_respects_unequal_face_widths():
    """NON-VACUITY: the metric is load-bearing, not decoration.

    The same field differenced with a doubled span must give exactly half the
    gradient.  If this passes while the previous test also passes, the two
    dx arguments are genuinely being used rather than ignored.
    """
    nj, ni = 2, 10
    phi = np.tile(np.arange(ni, dtype=float), (nj, 1))
    one = np.ones((nj, ni))
    a = M._dphi_dx(phi, one, one)
    b = M._dphi_dx(phi, one, 3.0 * one)
    assert np.allclose(a[:, 1:-1], 1.0)
    assert np.allclose(b[:, 1:-1], 0.5)


def test_stress_divergence_matches_the_analytic_parabola():
    """u = c z^2 with constant Kv has d/dz(Kv du/dz) = 2 c Kv everywhere."""
    nk = 40
    z_if = np.linspace(5.0, 395.0, nk - 1)
    z_c = np.concatenate([[0.0], 0.5 * (z_if[:-1] + z_if[1:]), [400.0]])
    assert z_c.size == nk
    c, kv0 = 3e-6, 1e-3
    u = np.tile((c * z_c ** 2)[None, None, :], (2, 3, 1))
    kv = np.full((2, 3, nk - 1), kv0)
    got = M._ddz_flux(u, kv, z_c, z_if)
    interior = got[:, :, 2:-2]
    assert np.isfinite(interior).all()
    assert np.allclose(interior, 2.0 * c * kv0, rtol=2e-2)


def test_stress_divergence_vanishes_on_uniform_flow():
    """No shear, no friction -- whatever the viscosity profile is."""
    nk = 20
    z_if = np.linspace(5.0, 195.0, nk - 1)
    z_c = np.concatenate([[0.0], 0.5 * (z_if[:-1] + z_if[1:]), [200.0]])
    u = np.full((2, 2, nk), 0.4)
    kv = np.abs(np.random.default_rng(0).normal(1e-3, 3e-4, (2, 2, nk - 1)))
    got = M._ddz_flux(u, kv, z_c, z_if)
    assert np.allclose(got[:, :, 1:-1], 0.0, atol=1e-18)


def test_pressure_anomaly_is_the_running_integral_to_the_cell_centre():
    """Constant density anomaly integrates to (drho/rho0) * depth at centres."""
    nk = 6
    dz = np.full((1, 1, nk), 10.0)
    rho0, drho = 1025.0, 2.0
    rho = np.full((1, 1, nk), rho0 + drho)
    got = M._hydrostatic_p_prime(rho, dz, rho0)
    centres = np.cumsum(np.full(nk, 10.0)) - 5.0     # 5, 15, 25, ...
    assert np.allclose(got[0, 0], (drho / rho0) * centres, rtol=1e-12)


def test_pressure_anomaly_is_zero_for_a_resting_reference_ocean():
    got = M._hydrostatic_p_prime(np.full((1, 1, 4), 1025.0),
                                 np.full((1, 1, 4), 25.0), 1025.0)
    assert np.allclose(got, 0.0, atol=1e-15)


def test_box_profile_refuses_an_empty_box_rather_than_returning_nan():
    """A silent NaN here would look like a masked ocean, not a bad selector."""
    lat = np.full((4, 4), 40.0)
    lon = np.full((4, 4), 220.0)
    with pytest.raises(SystemExit):
        M._box_profile(np.zeros((4, 4, 3)), lat, lon, 220.0, 1.0)


def test_box_profile_averages_only_the_selected_band():
    lat = np.array([[0.5, 0.5], [30.0, 30.0]])
    lon = np.full((2, 2), 220.0)
    f = np.zeros((2, 2, 2))
    f[0, :, :] = 1.0          # equatorial row
    f[1, :, :] = 99.0         # must not contribute
    got = M._box_profile(f, lat, lon, 220.0, 1.0)
    assert np.allclose(got, 1.0)
