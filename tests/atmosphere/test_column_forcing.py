"""Unit tests for :mod:`legoesm.atmosphere.forcing.idealized.column_forcing`.

Pins Stage 4 (assembly half) of ``docs/COMPARE_REANALYSIS.md``: building a
steady SCMForcing from a flagged GCM column's large-scale state — Coriolis,
ω→w conversion (reusing the shared diagnostic), steady callables, and the
surface-prescription dispatch (raise on unknown / inconsistent).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.forcing.idealized.column_forcing import (
    ColumnLargeScaleState,
    build_column_scm_forcing,
    coriolis_f_c,
    subsidence_w_from_omega,
)
from legoesm.atmosphere.physics._shared import diagnose_grid_w_from_omega


def _column(nlev=5, **overrides):
    base = dict(
        lat_rad=jnp.deg2rad(30.0),
        T=jnp.full((nlev,), 280.0),
        p_full=jnp.linspace(2.0e4, 1.0e5, nlev),
        q_v=jnp.full((nlev,), 5e-3),
    )
    base.update(overrides)
    return ColumnLargeScaleState(**base)


def test_coriolis_equator_and_30N():
    assert coriolis_f_c(0.0) == pytest.approx(0.0, abs=1e-15)
    f30 = coriolis_f_c(float(jnp.deg2rad(30.0)))
    # rel=1e-5 is fp32-safe: this file is in CI's fp32-by-default unit tier, where
    # 2·Ω·sin(30°) carries ~5e-8 relative float32 roundoff; 1e-5 stays deterministic
    # at fp32 AND x64 and still catches a wrong factor/latitude/sign. (The prior
    # rel=1e-10 passed only via the session-wide x64 leak — order-dependent.)
    assert f30 == pytest.approx(2.0 * constants.Omega * 0.5, rel=1e-5)
    # Southern hemisphere flips sign.
    assert coriolis_f_c(float(jnp.deg2rad(-30.0))) == pytest.approx(-f30, rel=1e-5)


def test_subsidence_w_from_omega_matches_shared():
    nlev = 4
    omega = jnp.array([0.1, 0.05, -0.02, 0.0])  # Pa/s, positive = sinking
    T = jnp.full((nlev,), 270.0)
    p = jnp.linspace(3.0e4, 9.0e4, nlev)
    q = jnp.full((nlev,), 4e-3)
    w = subsidence_w_from_omega(omega, T, p, q)
    ref = diagnose_grid_w_from_omega(omega[None, :], T[None, :], p[None, :], q[None, :])[0]
    assert jnp.allclose(w, ref)
    # Positive omega (sinking) -> negative w (downward).
    assert float(w[0]) < 0.0


def test_build_from_omega_sets_subsidence_callable():
    nlev = 5
    omega = jnp.full((nlev,), 0.05)
    ls = _column(nlev=nlev, omega=omega)
    forcing = build_column_scm_forcing(ls)
    assert forcing.subsidence_w is not None
    w = forcing.subsidence_w(0.0)
    ref = subsidence_w_from_omega(omega, ls.T, ls.p_full, ls.q_v)
    assert jnp.allclose(w, ref)
    # Steady: time argument is ignored.
    assert jnp.allclose(forcing.subsidence_w(1234.0), w)


def test_build_from_explicit_subsidence_w():
    nlev = 5
    w_in = jnp.linspace(-0.02, 0.0, nlev)
    ls = _column(nlev=nlev, subsidence_w=w_in)
    forcing = build_column_scm_forcing(ls)
    assert jnp.allclose(forcing.subsidence_w(0.0), w_in)


def test_both_omega_and_w_raises():
    nlev = 5
    ls = _column(
        nlev=nlev, omega=jnp.zeros((nlev,)), subsidence_w=jnp.zeros((nlev,))
    )
    with pytest.raises(ValueError, match="either omega or subsidence_w"):
        build_column_scm_forcing(ls)


def test_geostrophic_and_advective_profiles_wrapped():
    nlev = 5
    u_geo = jnp.full((nlev,), 8.0)
    theta_adv = jnp.full((nlev,), -1e-5)
    ls = _column(
        nlev=nlev, u_geo=u_geo, theta_adv=theta_adv,
        subsidence_w=jnp.zeros((nlev,)),
    )
    forcing = build_column_scm_forcing(ls)
    assert jnp.allclose(forcing.u_geo(0.0), u_geo)
    assert jnp.allclose(forcing.theta_adv(99.0), theta_adv)
    assert forcing.v_geo is None
    assert forcing.qv_adv is None
    assert forcing.f_c == pytest.approx(coriolis_f_c(float(ls.lat_rad)))


def test_prescribe_T_s():
    ls = _column(prescribe="T_s", T_s=301.0, subsidence_w=jnp.zeros((5,)))
    forcing = build_column_scm_forcing(ls)
    assert forcing.prescribe == "T_s"
    assert float(forcing.T_s(0.0)) == pytest.approx(301.0)


def test_prescribe_fluxes():
    ls = _column(
        prescribe="fluxes", w_th_s=0.02, w_qv_s=1e-5, subsidence_w=jnp.zeros((5,))
    )
    forcing = build_column_scm_forcing(ls)
    assert forcing.prescribe == "fluxes"
    assert float(forcing.w_th_s(0.0)) == pytest.approx(0.02)


def test_prescribe_T_s_without_value_raises():
    ls = _column(prescribe="T_s", subsidence_w=jnp.zeros((5,)))  # T_s missing
    with pytest.raises(ValueError, match="T_s"):
        build_column_scm_forcing(ls)


def test_unknown_prescribe_raises():
    ls = _column(prescribe="bogus", subsidence_w=jnp.zeros((5,)))
    with pytest.raises(ValueError):
        build_column_scm_forcing(ls)


def test_no_subsidence_requires_explicit_opt_in():
    # Default: silently dropping subsidence is rejected.
    with pytest.raises(ValueError, match="allow_no_subsidence"):
        build_column_scm_forcing(_column())
    # Explicit opt-in is allowed.
    forcing = build_column_scm_forcing(_column(), allow_no_subsidence=True)
    assert forcing.subsidence_w is None


def test_prescribe_none_with_surface_field_raises():
    ls = _column(subsidence_w=jnp.zeros((5,)), T_s=300.0)  # prescribe defaults none
    with pytest.raises(ValueError, match="prescribe='none'"):
        build_column_scm_forcing(ls)


def test_prescribe_T_s_with_flux_extra_raises():
    ls = _column(subsidence_w=jnp.zeros((5,)), prescribe="T_s", T_s=300.0, w_th_s=0.01)
    with pytest.raises(ValueError, match="conflicts with flux"):
        build_column_scm_forcing(ls)


def test_prescribe_fluxes_with_T_s_raises():
    ls = _column(subsidence_w=jnp.zeros((5,)), prescribe="fluxes", w_th_s=0.01, T_s=300.0)
    with pytest.raises(ValueError, match="conflicts with T_s"):
        build_column_scm_forcing(ls)


def test_subsidence_w_from_omega_bad_qv_shape_raises():
    nlev = 4
    omega = jnp.zeros((nlev,))
    T = jnp.full((nlev,), 270.0)
    p = jnp.linspace(3.0e4, 9.0e4, nlev)
    bad_q = jnp.zeros((nlev + 2,))  # wrong length
    with pytest.raises(ValueError, match="q_v"):
        subsidence_w_from_omega(omega, T, p, bad_q)


def test_malformed_profile_shape_raises():
    nlev = 5
    bad_u = jnp.zeros((nlev + 1,))  # wrong length
    ls = _column(nlev=nlev, subsidence_w=jnp.zeros((nlev,)), u_geo=bad_u)
    with pytest.raises(ValueError, match="u_geo"):
        build_column_scm_forcing(ls)
