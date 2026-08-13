"""The Poincare channel mode must actually solve the linear shallow-water
equations -- otherwise the IGW channel case gates against a wrong oracle.

WHY THIS FILE EXISTS. ``run_ocean_test_matrix.igw_channel_mode`` is the
EXACT solution the ``inertia_gravity_wave_channel`` case scores itself
against: its L2 gate, its frequency gate and its mode-purity gate are all
statements about agreement with this closed form. The block comment above
that function claimed the form had been verified by substitution and
pointed at this file -- which did not exist (found 2026-08-13). A case
whose oracle is unverified can be confidently, quietly wrong, so the
substitution is done here for real.

WHAT IS CHECKED, and why each is not redundant:

1. The three linear shallow-water residuals vanish. This is the only check
   that the closed form is a SOLUTION rather than merely a plausible-looking
   trigonometric expression. Derivatives are taken by central differences of
   the SHIPPED function -- not by re-deriving the algebra here, which would
   only test that two copies of the same mistake agree.

2. ``v`` vanishes on both walls. The channel's boundary condition, and the
   reason the meridional wavenumber is quantised as ``l = n pi / Ly``. A
   plane wave passes check 1 and fails this one, which is exactly the error
   the global sphere case makes.

3. The dispersion relation the case's own parameters carry is
   ``omega^2 = f^2 + gH(k^2 + l^2)``. Checks 1 and 2 constrain the amplitude
   ratios; this pins the frequency the phase-fit gate is read against.

4. The mode is periodic in x over ``Lx`` and returns to itself after one
   period. If it were not, "1.25 periods" would not be the quantity the case
   thinks it is running.

Requires float64: the residual in check 1 is a difference of terms that
cancel to ~1e-9 of their own size, which fp32 cannot express.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from legoesm import constants as C

_REPO = Path(__file__).resolve().parents[3]


def _matrix():
    """The ocean matrix module, for the SHIPPED mode and its parameters."""
    if "_rm_igwc" in sys.modules:
        return sys.modules["_rm_igwc"]
    p = _REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py"
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("_rm_igwc", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rm_igwc"] = mod
    spec.loader.exec_module(mod)
    return mod


#: Cell counts only select the geometry helper's spacing; the mode itself is
#: continuous, so any pair works. These match the case's coarse arm.
_NLAT, _NLON = 20, 40


def _params():
    M = _matrix()
    lx, ly, k, l_y, omega, period = M._igw_channel_params(_NLAT, _NLON)
    return M, lx, ly, k, l_y, omega, period


def _sample_points(lx, ly, n=9):
    """Interior points, deliberately NOT on any node of sin/cos(l y).

    A test evaluated only where the mode's y-structure is zero would pass
    for a form whose y-dependence is wrong.
    """
    x = np.linspace(0.07 * lx, 0.93 * lx, n)
    y = np.linspace(0.11 * ly, 0.89 * ly, n)
    return np.meshgrid(x, y, indexing="ij")


def _swe_residuals(mode, X, Y, t0, k, l_y, period):
    """Max relative residual of the three linear shallow-water equations.

    Derivatives by central differences of whatever ``mode(x, y, t)`` is
    handed in, so the SAME machinery scores the shipped form and the
    deliberately-wrong controls below. Each residual is normalised by the
    largest TERM in its own equation, making the number a statement about
    cancellation rather than about the amplitude the mode is scaled to.
    """
    M = _matrix()
    f, g, H = M._IGWC_F0, C.g, M._IGWC_H
    # Steps small enough that truncation is far below the tolerance and
    # large enough that round-off is not: ~1e-4 of a wavelength / period.
    hx = 1.0e-4 * (2.0 * np.pi / k)
    hy = 1.0e-4 * (np.pi / l_y)
    ht = 1.0e-4 * period

    def d(comp, arg, h):
        p = [X, Y, np.full_like(X, t0)]
        m = [X, Y, np.full_like(X, t0)]
        p[arg] = p[arg] + h
        m[arg] = m[arg] - h
        return (np.asarray(mode(*p)[comp]) - np.asarray(mode(*m)[comp])) \
            / (2.0 * h)

    _eta, u, v = (np.asarray(a) for a in mode(X, Y, t0))
    r1 = d(1, 2, ht) - f * v + g * d(0, 0, hx)
    r2 = d(2, 2, ht) + f * u + g * d(0, 1, hy)
    r3 = d(0, 2, ht) + H * (d(1, 0, hx) + d(2, 1, hy))
    s1 = max(np.max(np.abs(f * v)), np.max(np.abs(g * d(0, 0, hx))))
    s2 = max(np.max(np.abs(f * u)), np.max(np.abs(g * d(0, 1, hy))))
    s3 = max(np.max(np.abs(d(0, 2, ht))), np.max(np.abs(H * d(1, 0, hx))))
    return (np.max(np.abs(r1)) / s1, np.max(np.abs(r2)) / s2,
            np.max(np.abs(r3)) / s3)


def test_the_mode_solves_the_linear_shallow_water_equations():
    """u_t - f v + g eta_x = 0, v_t + f u + g eta_y = 0,
    eta_t + H (u_x + v_y) = 0, by central differences of the shipped form.
    """
    M, lx, ly, k, l_y, _omega, period = _params()
    X, Y = _sample_points(lx, ly)
    res = _swe_residuals(
        lambda x, y, t: M.igw_channel_mode(x, y, t, _NLAT, _NLON),
        X, Y, 0.317 * period, k, l_y, period)
    assert max(res) < 1e-6, res


@pytest.mark.parametrize("component", [0, 1, 2])
def test_the_residual_check_can_fail(component):
    """NON-VACUITY. Break ONE amplitude by 5% and the residual must blow up.

    A residual test that cannot fail would certify any expression at all.
    Scaling a single component is the smallest perturbation that still
    violates the coupling, and it is applied to each of eta, u and v in
    turn so no equation is left unconstrained.
    """
    M, lx, ly, k, l_y, _omega, period = _params()
    X, Y = _sample_points(lx, ly)

    def broken(x, y, t):
        out = list(M.igw_channel_mode(x, y, t, _NLAT, _NLON))
        out[component] = out[component] * 1.05
        return tuple(out)

    res = _swe_residuals(broken, X, Y, 0.317 * period, k, l_y, period)
    assert max(res) > 1e-3, res


def test_a_plane_wave_would_fail_the_wall_condition():
    """The wall check is the one a plane wave cannot pass.

    A plane wave IS a solution of the f-plane equations -- it passes the
    residual check -- and is still the wrong answer here, because the
    channel has walls. That is exactly the error the global sphere case
    makes, so it is the right negative control for the boundary condition.
    """
    M, lx, ly, k, l_y, omega, period = _params()
    f, g = M._IGWC_F0, C.g
    X, Y = _sample_points(lx, ly)
    t0 = 0.317 * period

    def plane(x, y, t):
        th = k * x + l_y * y - omega * t
        den = omega ** 2 - f ** 2
        return (np.cos(th),
                (g / den) * (omega * k * np.cos(th) - f * l_y * np.sin(th)),
                (g / den) * (omega * l_y * np.cos(th) + f * k * np.sin(th)))

    # It solves the equations ...
    assert max(_swe_residuals(plane, X, Y, t0, k, l_y, period)) < 1e-6
    # ... and still drives flow straight through the southern wall.
    v_wall = plane(X, np.zeros_like(Y), t0)[2]
    assert np.max(np.abs(v_wall)) > 0.1 * np.max(np.abs(plane(X, Y, t0)[2]))


def test_v_vanishes_on_both_walls():
    M, lx, ly, k, l_y, omega, period = _params()
    X, _Y = _sample_points(lx, ly)
    for wall in (0.0, ly):
        for t in (0.0, 0.317 * period, 0.5 * period):
            v = np.asarray(
                M.igw_channel_mode(X, np.full_like(X, wall), t,
                                   _NLAT, _NLON)[2])
            # Scaled against the mode's own peak meridional velocity, not
            # against an absolute floor.
            assert np.max(np.abs(v)) < 1e-12 * M._IGWC_V_AMP


def test_the_dispersion_relation_is_the_one_the_gate_reads():
    M, lx, ly, k, l_y, omega, _period = _params()
    expect = np.sqrt(M._IGWC_F0 ** 2 + C.g * M._IGWC_H * (k ** 2 + l_y ** 2))
    assert abs(omega - expect) <= 1e-14 * expect
    # And the quantisation that makes the wall condition possible.
    assert abs(l_y - np.pi * M._IGWC_N / ly) <= 1e-14 * l_y
    assert abs(k - 2.0 * np.pi * M._IGWC_M / lx) <= 1e-14 * k


def test_the_mode_is_periodic_in_x_and_in_time():
    M, lx, ly, k, l_y, omega, period = _params()
    X, Y = _sample_points(lx, ly)
    t0 = 0.317 * period
    for i, name in enumerate(("eta", "u", "v")):
        a = np.asarray(M.igw_channel_mode(X, Y, t0, _NLAT, _NLON)[i])
        b = np.asarray(M.igw_channel_mode(X + lx, Y, t0, _NLAT, _NLON)[i])
        c = np.asarray(M.igw_channel_mode(X, Y, t0 + period,
                                          _NLAT, _NLON)[i])
        scale = max(np.max(np.abs(a)), 1e-300)
        assert np.max(np.abs(a - b)) < 1e-9 * scale, name
        assert np.max(np.abs(a - c)) < 1e-9 * scale, name


def test_a_quarter_period_is_not_the_initial_condition():
    """The case runs 1.25 periods precisely so "close to exact" cannot
    degenerate into "close to your own IC". Pin that the exact solution has
    in fact moved by then."""
    M, lx, ly, _k, _l, _omega, period = _params()
    X, Y = _sample_points(lx, ly)
    e0 = np.asarray(M.igw_channel_mode(X, Y, 0.0, _NLAT, _NLON)[0])
    e1 = np.asarray(M.igw_channel_mode(X, Y, M._IGWC_PERIODS * period,
                                       _NLAT, _NLON)[0])
    rel = np.sqrt(np.mean((e1 - e0) ** 2)) / np.sqrt(np.mean(e0 ** 2))
    assert rel > 1.0, rel
