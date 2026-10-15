"""Kinematic pressure diagnostic + exact column budgets of the spectral plane LES.

* The projection potential of the rotational-form RHS is the DYNAMIC pressure
  ``P = p/ρ₀ + ½|u|²``; ``kinematic_pressure`` returns ``p/ρ₀``. Checked against the
  analytic pressure of a steady 2-D Taylor–Green (Euler) cell, where ``P`` and ``p``
  differ at O(1), with second-order convergence in the vertical.
* ``step(return_budget=True)`` integrates the boundary/body sources through the
  same RK stage combination as the state, so the column θ/u/v budgets close to
  round-off (JAX_ENABLE_X64) — including the MOST prescribed-cooling surface, whose
  flux changes from stage to stage.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402
from legoesm.atmosphere.les_suite.emit import covariance  # noqa: E402


def _tg_grid(nz, nx=32, ny=4, length=1000.0):
    cfg = sl.SpectralLESConfig(nx=nx, ny=ny, nz=nz, Lx=length, Ly=length * ny / nx,
                               Lz=length,
                               dealias=True, spectral_filter=False)
    return sl.make_grid(cfg, dtype=jnp.float64)


def _taylor_green(g, amp=1.0):
    """Steady 2-D Euler cell in (x, z): ψ = (A/m) sin(kx) sin(mz), u = ψ_z,
    w = −ψ_x, ω = −K²ψ ⇒ u×ω = ∇(−K²ψ²/2), so P = −K²ψ²/2 and
    p/ρ₀ = P − |u|²/2 (w = 0 on both walls)."""
    cfg = g.cfg
    k = 2.0 * np.pi / cfg.Lx
    m = np.pi / cfg.Lz
    x = (np.arange(cfg.nx) * cfg.Lx / cfg.nx)[None, :, None]
    zc = np.asarray(g.z_c)[None, None, :]
    zf = np.asarray(g.z_f)[None, None, :]
    shape = (cfg.ny, cfg.nx, cfg.nz)
    u = np.broadcast_to(amp * np.sin(k * x) * np.cos(m * zc), shape)
    w = np.broadcast_to(-amp * (k / m) * np.cos(k * x) * np.sin(m * zf),
                        (cfg.ny, cfg.nx, cfg.nz + 1))
    psi = (amp / m) * np.sin(k * x) * np.sin(m * zc)
    uc2 = u ** 2 + (amp * (k / m) * np.cos(k * x) * np.sin(m * zc)) ** 2
    p = np.broadcast_to(-0.5 * (k ** 2 + m ** 2) * psi ** 2 - 0.5 * uc2, shape)
    return jnp.asarray(u), jnp.zeros(shape), jnp.asarray(w), p


def _dev(f):
    f = np.asarray(f)
    return f - f.mean(axis=(0, 1), keepdims=True)


def _tg_errors(nz):
    g = _tg_grid(nz)
    u, v, w, p_true = _taylor_green(g)
    cu, cv, cw = sl.advection(u, v, w, g)
    cw = cw.at[..., 0].set(0.0).at[..., -1].set(0.0)
    big_p = np.asarray(sl.pressure_potential(cu, cv, cw, 1.0, g))
    ke = np.asarray(sl.resolved_kinetic_energy(u, v, w, g))
    scale = np.max(np.abs(_dev(p_true)))
    err_p = np.max(np.abs(_dev(big_p - ke) - _dev(p_true))) / scale
    err_big = np.max(np.abs(_dev(big_p) - _dev(p_true))) / scale
    return err_p, err_big


def test_kinematic_pressure_matches_taylor_green_and_converges():
    e32, big32 = _tg_errors(32)
    e64, big64 = _tg_errors(64)
    assert e64 < 2e-3, e64
    assert 3.0 < e32 / e64 < 5.0, (e32, e64)   # second order in dz
    # the projection potential itself is NOT p: it carries +½|u|² (O(1) off)
    assert big64 > 0.3, big64


def test_resolved_kinetic_energy_is_alias_free():
    # u = cos(6·2πx/L) on 16 points: u² holds wavenumber 12 > Nyquist (8), which a
    # coarse-grid square folds onto wavenumber 4. The de-aliased product (the one the
    # rotational advection uses) keeps only the resolvable part: ½u² → ¼ exactly.
    g = _tg_grid(4, nx=16, ny=16)
    x = np.arange(16) * g.cfg.Lx / 16
    u = jnp.asarray(np.broadcast_to(
        np.cos(6 * 2 * np.pi * x / g.cfg.Lx)[None, :, None], (16, 16, 4)))
    z = jnp.zeros((16, 16, 4))
    ke = np.asarray(sl.resolved_kinetic_energy(u, z, jnp.zeros((16, 16, 5)), g))
    np.testing.assert_allclose(ke, 0.25, atol=1e-13)
    naive = 0.5 * np.asarray(u) ** 2
    assert np.max(np.abs(naive - 0.25)) > 0.2      # the aliased mode the fix removes


def test_kinematic_pressure_is_projection_of_full_rhs_minus_ke():
    g = _tg_grid(16, nx=16, ny=16)
    key = jax.random.PRNGKey(3)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    shp = (16, 16, 16)
    u = jax.random.normal(k1, shp)
    v = jax.random.normal(k2, shp)
    w = jax.random.normal(k3, (16, 16, 17)).at[..., 0].set(0.0).at[..., -1].set(0.0)
    u, v, w = sl.project(u, v, w, 1.0, g)
    st = sl.SpectralLESState(u=u, v=v, w=w, rhs_u_prev=u * 0, rhs_v_prev=v * 0,
                             rhs_w_prev=w * 0)
    p = sl.kinematic_pressure(st, g, (1.0, 0.0), 1e-4)
    ru, rv, rw, *_ = sl.rhs(u, v, w, g, (1.0, 0.0), 1e-4)
    ref = (sl.pressure_potential(ru, rv, rw, 1.0, g)
           - sl.resolved_kinetic_energy(u, v, w, g))
    np.testing.assert_allclose(np.asarray(p), np.asarray(ref), rtol=0, atol=1e-12)
    # pressure–strain trace <p'∇·u'> vanishes for the projected (divergence-free) field
    terms = [covariance(np.asarray(p), np.asarray(f)) for f in (
        sl.ddx(u, g), sl.ddy(v, g), sl.ddz_f2c(w, g.dz))]
    trace = terms[0] + terms[1] + terms[2]
    assert np.max(np.abs(trace)) < 1e-10 * max(np.max(np.abs(t)) for t in terms)


def _stable_case_state(nx=12, nz=24, ug=8.0, theta_air=266.0):
    cfg = sl.SpectralLESConfig(
        nx=nx, ny=nx, nz=nz, Lx=400.0, Ly=400.0, Lz=400.0, z0=0.1, dealias=True,
        smagorinsky_dynamic=False, sgs_model="vreman", time_scheme="rk3",
        buoyancy=True, theta_ref0=265.0, nu_floor=0.01)
    g = sl.make_grid(cfg, dtype=jnp.float64)
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    shp = (nx, nx, nz)
    th = theta_air + 0.01 * g.z_c + 0.1 * jax.random.normal(k1, shp)
    u = ug + 0.5 * jax.random.normal(k2, shp)
    v = 0.5 * jax.random.normal(k3, shp)
    w = jnp.zeros((nx, nx, nz + 1))
    u, v, w = sl.project(u, v, w, 1.0, g)
    st = sl.SpectralLESState(u=u, v=v, w=w, rhs_u_prev=u * 0, rhs_v_prev=v * 0,
                             rhs_w_prev=w * 0, theta=th, rhs_theta_prev=th * 0)
    return g, st


def _col(f, g):
    return float(g.dz * jnp.sum(f.mean((0, 1))))


@pytest.mark.parametrize("t_sfc", [None, 263.0])
def test_column_budgets_close_to_roundoff(t_sfc):
    g, st = _stable_case_state()
    f_cor, u_geo, q0, dt = 1.39e-4, (8.0, 0.0), -0.01, 0.5
    step = jax.jit(lambda s, ts: sl.step(s, g, dt, u_geo, f_cor, sfc_theta_flux=q0,
                                         t_sfc=ts, return_budget=True))
    c0 = [_col(f, g) for f in (st.theta, st.u, st.v)]
    acc = np.zeros(len(sl.BUDGET_TERMS))
    for _ in range(20):
        st, _us, b = step(st, None if t_sfc is None else jnp.asarray(t_sfc))
        acc += np.asarray(b)
    c1 = [_col(f, g) for f in (st.theta, st.u, st.v)]
    t = dict(zip(sl.BUDGET_TERMS, acc))
    d_th, d_u, d_v = (c1[i] - c0[i] for i in range(3))
    assert abs(t["sfc_heat_flux"]) > 1e-3          # a real surface exchange
    # round-off of a column sum scales with the column CONTENT (∫θdz ~ 1e5 K m)
    tol = [1e-13 * abs(c) for c in c1]
    res = (d_th - t["sfc_heat_flux"],
           d_u - (t["sfc_stress_x"] + t["coriolis_u"] + t["body_force_u"]),
           d_v - (t["sfc_stress_y"] + t["coriolis_v"] + t["body_force_v"]))
    for r, tl, d in zip(res, tol, (d_th, d_u, d_v)):
        assert abs(r) <= tl, (res, tol)
        assert abs(r) < 1e-6 * abs(d), (res, d)   # and negligible against the change


def test_budget_flag_leaves_the_state_bit_identical():
    g, st = _stable_case_state()
    a, _ = sl.step(st, g, 0.5, (8.0, 0.0), 1.39e-4, t_sfc=jnp.asarray(263.0))
    b, _, _ = sl.step(st, g, 0.5, (8.0, 0.0), 1.39e-4, t_sfc=jnp.asarray(263.0),
                      return_budget=True)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_return_budget_rejects_ab2():
    cfg = sl.SpectralLESConfig(nx=8, ny=8, nz=8, Lx=100.0, Ly=100.0, Lz=100.0,
                               time_scheme="ab2")
    g = sl.make_grid(cfg, dtype=jnp.float64)
    z = jnp.zeros((8, 8, 8))
    st = sl.SpectralLESState(u=z, v=z, w=jnp.zeros((8, 8, 9)), rhs_u_prev=z,
                             rhs_v_prev=z, rhs_w_prev=jnp.zeros((8, 8, 9)))
    with pytest.raises(ValueError):
        sl.step(st, g, 0.1, (0.0, 0.0), 0.0, return_budget=True)


def test_surface_boundary_matches_most_and_prescribed():
    g, st = _stable_case_state()
    cd, q = sl.surface_boundary(st.u, st.v, st.theta, g, -0.01, None)
    assert cd is None and q == -0.01
    cd, q = sl.surface_boundary(st.u, st.v, st.theta, g, -0.01, jnp.asarray(263.0))
    spd = jnp.mean(jnp.sqrt(st.u[..., 0] ** 2 + st.v[..., 0] ** 2 + 1e-12))
    _us, _ts, q_ref, cd_ref = sl.most_surface_flux(
        spd, jnp.mean(st.theta[..., 0]), 263.0, g.z_c[0], 0.1, 265.0)
    assert float(q) < 0.0     # warm air over a colder surface → downward heat flux
    np.testing.assert_allclose(float(q), float(q_ref), rtol=1e-12)
    np.testing.assert_allclose(float(cd), float(cd_ref), rtol=1e-12)
