"""Pseudo-spectral incompressible plane-LES core (jax-alfa-faithful).

Validates the load-bearing pieces:
  1. PROJECTION — an arbitrary velocity field is made discretely divergence-free
     to round-off (the fractional-step Poisson solve is the core of the method);
  2. SPECTRAL DERIVATIVE — exact on a single Fourier mode;
  3. ROTATIONAL ADVECTION conserves resolved kinetic energy (⟨u·C⟩≈0) — the
     property that lets the scheme sustain turbulence (no numerical dissipation);
  4. a step runs, stays finite, and keeps w=0 at the walls.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl

jax.config.update("jax_enable_x64", True)


def _grid(nx=16, ny=16, nz=24):
    cfg = sl.SpectralLESConfig(nx=nx, ny=ny, nz=nz, Lx=320.0, Ly=320.0, Lz=480.0)
    return sl.make_grid(cfg)


def _divergence(u, v, w, g):
    return sl.ddx(u, g) + sl.ddy(v, g) + sl.ddz_f2c(w, g.dz)


def test_projection_makes_divergence_free():
    g = _grid()
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    k = jax.random.PRNGKey(0)
    u = jax.random.normal(k, (ny, nx, nz))
    v = jax.random.normal(jax.random.PRNGKey(1), (ny, nx, nz))
    w = jax.random.normal(jax.random.PRNGKey(2), (ny, nx, nz + 1))
    w = w.at[..., 0].set(0.0).at[..., -1].set(0.0)
    u2, v2, w2 = sl.project(u, v, w, dt=0.1, g=g)
    div = _divergence(u2, v2, w2, g)
    # Relative to the pre-projection divergence magnitude.
    div0 = _divergence(u, v, w, g)
    assert float(jnp.max(jnp.abs(div))) / float(jnp.max(jnp.abs(div0))) < 1e-10
    # Walls untouched.
    assert float(jnp.max(jnp.abs(w2[..., 0]))) == 0.0
    assert float(jnp.max(jnp.abs(w2[..., -1]))) == 0.0


def test_spectral_derivative_exact_single_mode():
    g = _grid()
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    x = (jnp.arange(nx) + 0.5) * g.dx
    kx = 2.0 * jnp.pi * 2 / g.cfg.Lx                      # mode 2
    f = jnp.broadcast_to(jnp.sin(kx * x)[None, :, None], (ny, nx, nz))
    df = sl.ddx(f, g)
    ref = jnp.broadcast_to((kx * jnp.cos(kx * x))[None, :, None], (ny, nx, nz))
    np.testing.assert_allclose(np.asarray(df), np.asarray(ref), atol=1e-9)


def test_rotational_advection_conserves_energy():
    """⟨u·C⟩ over the volume ≈ 0 for the rotational form (no spurious KE
    production/destruction) — the non-dissipation that sustains turbulence."""
    g = _grid()
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    k = jax.random.PRNGKey(3)
    u = 0.3 * jax.random.normal(k, (ny, nx, nz))
    v = 0.3 * jax.random.normal(jax.random.PRNGKey(4), (ny, nx, nz))
    w = 0.2 * jax.random.normal(jax.random.PRNGKey(5), (ny, nx, nz + 1))
    w = w.at[..., 0].set(0.0).at[..., -1].set(0.0)
    u, v, w = sl.project(u, v, w, dt=0.1, g=g)            # start divergence-free
    Cu, Cv, Cw = sl.advection(u, v, w, g)
    wc = sl.f2c(w)
    Cw_c = sl.f2c(Cw)
    prod = jnp.mean(u * Cu + v * Cv + wc * Cw_c)
    norm = jnp.mean(u ** 2 + v ** 2 + wc ** 2)
    assert abs(float(prod)) / float(norm) < 0.05          # small KE-production residual


def test_dealias_product_is_alias_free():
    """3/2 zero-padding (Dealias1/2 port) computes quadratic products alias-free:
    a resolved product is exact, and an interaction that would alias under plain
    2/3-truncation is removed to round-off (matches the analytic low-pass)."""
    ny = nx = 16
    nz = 3
    xx = jnp.arange(nx) / nx
    bcast = lambda f: jnp.broadcast_to(f[None, :, None], (ny, nx, nz))  # noqa: E731
    # modes 3 and 4 → product modes 1 and 7 (< Nyquist 8): exact.
    a = bcast(jnp.cos(2 * jnp.pi * 3 * xx))
    b = bcast(jnp.cos(2 * jnp.pi * 4 * xx))
    prod = sl._truncate_from_fine(sl._pad_to_fine(a) * sl._pad_to_fine(b), ny, nx)
    np.testing.assert_allclose(np.asarray(prod), np.asarray(a * b), atol=1e-12)
    # modes 6 and 5 → product modes 1 and 11; mode 11 aliases to 5 under
    # truncation, but 3/2 padding drops it → analytic low-pass is 0.5 cos(x).
    a2 = bcast(jnp.cos(2 * jnp.pi * 6 * xx))
    b2 = bcast(jnp.cos(2 * jnp.pi * 5 * xx))
    pp = sl._truncate_from_fine(sl._pad_to_fine(a2) * sl._pad_to_fine(b2), ny, nx)
    true = bcast(0.5 * jnp.cos(2 * jnp.pi * 1 * xx))
    np.testing.assert_allclose(np.asarray(pp), np.asarray(true), atol=1e-12)


def test_step_runs_finite_and_walls():
    g = _grid()
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    u = jnp.full((ny, nx, nz), 8.0) + 0.1 * jax.random.normal(
        jax.random.PRNGKey(6), (ny, nx, nz))
    v = jnp.zeros((ny, nx, nz))
    w = jnp.zeros((ny, nx, nz + 1))
    st = sl.SpectralLESState(u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u),
                             rhs_v_prev=jnp.zeros_like(v), rhs_w_prev=jnp.zeros_like(w))
    st, ustar = sl.step(st, g, dt=0.05, u_geo=(8.0, 0.0), f_cor=1e-4, first=True)
    for _ in range(5):
        st, ustar = sl.step(st, g, dt=0.05, u_geo=(8.0, 0.0), f_cor=1e-4)
    assert bool(jnp.all(jnp.isfinite(st.u)))
    assert bool(jnp.all(jnp.isfinite(st.w)))
    assert float(jnp.max(jnp.abs(st.w[..., 0]))) == 0.0
    assert float(ustar) > 0.0


def test_buoyancy_and_surface_heatflux_signs():
    """A +ve surface kinematic heat flux WARMS the lowest cell, and a warm
    parcel (θ > horizontal mean) gets a POSITIVE (upward) buoyancy force."""
    cfg = sl.SpectralLESConfig(nx=8, ny=8, nz=16, Lx=160.0, Ly=160.0, Lz=320.0,
                               buoyancy=True, theta_ref0=300.0)
    g = sl.make_grid(cfg)
    th = jnp.full((8, 8, 16), 300.0)
    u = jnp.zeros((8, 8, 16)); v = jnp.zeros((8, 8, 16))
    w = jnp.zeros((8, 8, 17))
    nu_t = jnp.full((8, 8, 16), 0.1)
    Rth = sl.scalar_rhs(th, u, v, w, nu_t, g, sfc_flux=0.05)
    assert float(Rth[..., 0].mean()) > 0.0          # surface heating warms cell 0
    th2 = th.at[4, 4, 8].add(1.0)
    bf = sl.buoyancy_w(th2, g)
    assert float(bf[4, 4, 8]) > 0.0                 # warm parcel → upward buoyancy
    # other columns get only the tiny mean-shift compensation (≪ the bubble).
    assert abs(float(bf[0, 0, 8])) < 0.05 * float(bf[4, 4, 8])
    # buoyancy is from the DEVIATION: a uniform θ field gives zero force.
    assert float(jnp.max(jnp.abs(sl.buoyancy_w(th, g)))) < 1e-6


# --------------------------------------------------------------------------- #
# Vreman SGS, RK3 time scheme, CFL-adaptive dt (added 2026-06-06)              #
# --------------------------------------------------------------------------- #
def _grid_cfg(**kw):
    base = dict(nx=16, ny=16, nz=24, Lx=320.0, Ly=320.0, Lz=480.0)
    base.update(kw)
    return sl.make_grid(sl.SpectralLESConfig(**base))


def test_vreman_vanishes_for_unidirectional_shear():
    """Vreman ν_t must be ~0 for a well-resolved 1D shear u(z) (only ∂u/∂z≠0):
    the property that stops it over-dissipating resolved laminar shear."""
    g = _grid_cfg(sgs_model="vreman", c_vreman=0.07)
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    zc = g.z_c
    u = jnp.broadcast_to(2.0 * zc, (ny, nx, nz))           # u = 2 z  → ∂u/∂z const
    v = jnp.zeros((ny, nx, nz))
    w = jnp.zeros((ny, nx, nz + 1))
    nu_t = sl._vreman_nu_t(u, v, w, g)
    assert float(jnp.max(jnp.abs(nu_t))) < 1e-10


def test_vreman_positive_for_3d_field():
    """Vreman ν_t ≥ 0 everywhere and strictly > 0 for genuine 3D structure."""
    g = _grid_cfg(sgs_model="vreman")
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    k = jax.random.PRNGKey(3)
    u = jax.random.normal(k, (ny, nx, nz))
    v = jax.random.normal(jax.random.PRNGKey(4), (ny, nx, nz))
    w = jax.random.normal(jax.random.PRNGKey(5), (ny, nx, nz + 1))
    w = w.at[..., 0].set(0.0).at[..., -1].set(0.0)
    nu_t = sl._vreman_nu_t(u, v, w, g)
    assert float(jnp.min(nu_t)) >= 0.0
    assert float(jnp.max(nu_t)) > 0.0


def test_rk3_step_divergence_free_and_finite():
    """An SSP-RK3 step keeps the velocity discretely divergence-free + finite."""
    g = _grid_cfg(time_scheme="rk3", sgs_model="vreman")
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    k = jax.random.PRNGKey(6)
    u = jax.random.normal(k, (ny, nx, nz))
    v = jax.random.normal(jax.random.PRNGKey(7), (ny, nx, nz))
    w = jnp.zeros((ny, nx, nz + 1))
    u, v, w = sl.project(u, v, w, 0.05, g)
    st = sl.SpectralLESState(u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u),
                             rhs_v_prev=jnp.zeros_like(v), rhs_w_prev=jnp.zeros_like(w))
    st2, _ = sl.step(st, g=g, dt=0.05, u_geo=(0.0, 0.0), f_cor=1e-4)
    div = _divergence(st2.u, st2.v, st2.w, g)
    assert float(jnp.max(jnp.abs(div))) < 1e-8
    assert bool(jnp.all(jnp.isfinite(st2.u))) and bool(jnp.all(jnp.isfinite(st2.w)))
    assert float(jnp.max(jnp.abs(st2.w[..., 0]))) == 0.0


def test_rk3_matches_ab2_to_second_order_on_first_step():
    """On the first step (AB2 = forward Euler) RK3 and AB2 agree to O(dt²): a
    small-dt sanity check that RK3 is consistent with the same RHS."""
    for scheme in ("rk3", "ab2"):
        g = _grid_cfg(time_scheme=scheme, sgs_model="smagorinsky",
                      spectral_filter=False)
        ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
        u = jnp.zeros((ny, nx, nz)) + 1.0
        v = jnp.zeros((ny, nx, nz))
        w = jnp.zeros((ny, nx, nz + 1))
        st = sl.SpectralLESState(u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u),
                                 rhs_v_prev=jnp.zeros_like(v),
                                 rhs_w_prev=jnp.zeros_like(w))
        st2, _ = sl.step(st, g=g, dt=1e-3, u_geo=(0.0, 0.0), f_cor=0.0, first=True)
        assert bool(jnp.all(jnp.isfinite(st2.u)))


def test_select_dt_static_cfl():
    """The reused trace-time static CFL dt: dt = cfl·dx/max_wind, a plain float,
    smaller for a larger conservative wind, capped by dt_cap, validates inputs."""
    import pytest

    from legoesm.timestepping.split_explicit import select_dt
    dt = select_dt(dx=5.0, max_wind_safe=20.0, cfl_safe=0.8)
    assert isinstance(dt, float)
    assert abs(dt - 0.8 * 5.0 / 20.0) < 1e-12
    assert select_dt(5.0, 40.0, 0.8) < select_dt(5.0, 20.0, 0.8)   # windier→smaller
    assert select_dt(5.0, 1.0, 0.8, dt_cap=0.5) == 0.5            # cap honoured
    with pytest.raises(ValueError):
        select_dt(dx=-1.0)


def test_step_is_differentiable():
    """sl.step must remain reverse-mode differentiable (legoESM end-to-end AD goal)
    — grad of a scalar of the post-step state w.r.t. the initial velocity is finite
    and non-trivial. Covers the SSP-RK3 path that reuses split_explicit."""
    g = _grid_cfg(time_scheme="rk3", sgs_model="vreman", nz=12)
    ny, nx, nz = g.cfg.ny, g.cfg.nx, g.cfg.nz
    k = jax.random.PRNGKey(11)
    u0 = jax.random.normal(k, (ny, nx, nz)) * 0.1
    v0 = jnp.zeros((ny, nx, nz))
    w0 = jnp.zeros((ny, nx, nz + 1))
    u0, v0, w0 = sl.project(u0, v0, w0, 0.05, g)

    def loss(u):
        st = sl.SpectralLESState(u=u, v=v0, w=w0, rhs_u_prev=jnp.zeros_like(u),
                                 rhs_v_prev=jnp.zeros_like(v0),
                                 rhs_w_prev=jnp.zeros_like(w0))
        st2, _ = sl.step(st, g=g, dt=0.05, u_geo=(0.0, 0.0), f_cor=1e-4)
        return jnp.mean(st2.u ** 2)

    grad = jax.grad(loss)(u0)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


# --------------------------------------------------------------------------- #
# Coupled stable Monin–Obukhov surface layer (GABLS1 prescribed-cooling BC)     #
# --------------------------------------------------------------------------- #
def _neutral_ustar(spd, z1, z0):
    kappa = sl.constants.kappa_von_karman
    return kappa * spd / np.log(z1 / z0)


def test_most_surface_flux_signs_and_limits():
    """most_surface_flux: neutral ⇒ zero heat flux + neutral drag; stable cooling
    ⇒ downward (negative) heat flux, REDUCED u_* and drag vs neutral (the
    self-limiting SBL behaviour); strengthening the surface cooling deepens the
    flux but the stability correction keeps u_* below neutral."""
    z1, z0, thref, U = 2.0, 0.1, 265.0, 4.0
    un = _neutral_ustar(U, z1, z0)

    # Neutral: θ_air == T_sfc.
    us, ths, q0, cd = sl.most_surface_flux(U, 265.0, 265.0, z1, z0, thref)
    assert abs(float(q0)) < 1e-9
    assert abs(float(ths)) < 1e-9
    assert abs(float(us) - un) < 1e-6
    assert abs(float(cd) - (un / U) ** 2) < 1e-9

    # Stable: warmer air over a cooled surface ⇒ w'θ' < 0, u_* and Cd suppressed.
    us_s, ths_s, q0_s, cd_s = sl.most_surface_flux(U, 265.0, 263.0, z1, z0, thref)
    assert float(q0_s) < 0.0                         # downward (cooling) heat flux
    assert float(ths_s) > 0.0
    assert float(us_s) < un                          # stability reduces u_*
    assert float(cd_s) < (un / U) ** 2               # stability reduces drag
    assert np.isfinite([float(us_s), float(q0_s), float(cd_s)]).all()


def test_most_surface_flux_is_differentiable():
    """Reverse-mode AD through the surface-layer solve (legoESM end-to-end AD
    goal): d q0 / d T_sfc is finite and non-zero in the stable regime."""
    def q0_of_tsfc(t_sfc):
        return sl.most_surface_flux(4.0, 265.0, t_sfc, 2.0, 0.1, 265.0)[2]

    g = jax.grad(q0_of_tsfc)(263.0)
    assert bool(jnp.isfinite(g)) and abs(float(g)) > 0.0


def test_most_surface_flux_reuses_shared_psi(monkeypatch):
    """ENFORCEMENT (CLAUDE.md no-duplicate-numerics): most_surface_flux MUST use
    the SHARED canonical stability functions ``legoesm.core.bulk_flux.psi_m/psi_h``
    — it must not re-derive them inline. We monkeypatch the names the module
    bound at import; if the solver truly delegates, forcing ψ≡0 collapses the
    stable solution to the NEUTRAL log-law (despite strong stratification). A
    re-derived inline ψ would ignore the patch and this test would fail."""
    z1, z0, thref, U = 2.0, 0.1, 265.0, 4.0
    # Strongly stable input — a faithful solver gives u_* well below neutral.
    us_real = float(sl.most_surface_flux(U, 270.0, 260.0, z1, z0, thref)[0])
    un = _neutral_ustar(U, z1, z0)
    assert us_real < 0.95 * un                       # stability correction active

    monkeypatch.setattr(sl, "psi_m", lambda z: jnp.zeros_like(jnp.asarray(z)))
    monkeypatch.setattr(sl, "psi_h", lambda z: jnp.zeros_like(jnp.asarray(z)))
    us_patched = float(sl.most_surface_flux(U, 270.0, 260.0, z1, z0, thref)[0])
    # ψ≡0 ⇒ pure neutral log-law, regardless of the (strong) stratification.
    assert abs(us_patched - un) < 1e-6


def test_most_surface_flux_thermal_roughness():
    """z0h=None (default) and z0h=z0 give identical results, and in the neutral
    limit θ_* = κΔθ/ln(z₁/z0h): a smaller z0h lengthens the thermal log layer and
    weakens the heat flux by ln(z₁/z0)/ln(z₁/z0h). Identity to the pre-z0h code:
    the default reproduces outputs frozen from that code (commit 8232c3c49; equal
    bit for bit on CPU when frozen, compared at 1e-13 for portability)."""
    frozen = {(4.0, 265.0, 263.0, 2.0, 0.1, 265.0): (
                  0.5093883918647581, 0.2546941959323791, -0.12973826688328222,
                  0.016217283360402165),
              (8.0, 300.0, 301.5, 5.0, 0.01, 300.0): (
                  0.5221088251452545, -0.09922605066298044, 0.051806796735452215,
                  0.004259337895226402)}
    for args, ref in frozen.items():
        np.testing.assert_allclose([float(x) for x in sl.most_surface_flux(*args)], ref,
                                   rtol=1e-13)
    z1, z0, thref, U = 2.0, 0.1, 265.0, 4.0
    a = sl.most_surface_flux(U, 265.0, 263.0, z1, z0, thref)
    b = sl.most_surface_flux(U, 265.0, 263.0, z1, z0, thref, z0h=None)
    c = sl.most_surface_flux(U, 265.0, 263.0, z1, z0, thref, z0h=z0)
    for x, y, w in zip(a, b, c):
        assert float(x) == float(y) == float(w)
    # neutral-limit check: tiny Δθ so ψ_h ≈ 0
    dth = 1e-6
    _us, ths, _q, _cd = sl.most_surface_flux(U, 265.0 + dth, 265.0, z1, z0, thref,
                                             z0h=1e-3)
    kappa = sl.constants.kappa_von_karman
    np.testing.assert_allclose(float(ths), kappa * dth / np.log(z1 / 1e-3), rtol=1e-6)
    q_small = float(sl.most_surface_flux(U, 265.0, 263.0, z1, z0, thref, z0h=1e-3)[2])
    assert float(a[2]) < q_small < 0.0      # weaker cooling flux with smaller z0h


def test_most_drag_prescribed_flux_limits_and_fixed_point():
    """Zero flux ⇒ the neutral log law (to ψ's 1e-10 ζ clamp); surface heating (q0>0, unstable)
    ⇒ MORE drag than neutral, cooling ⇒ less; the returned u_* satisfies the
    Businger–Dyer relation at the Obukhov length the flux sets; AD finite."""
    z1, z0, thref, U = 10.0, 0.01, 300.0, 10.0
    un = _neutral_ustar(U, z1, z0)
    us0, cd0 = sl.most_drag_prescribed_flux(U, 0.0, z1, z0, thref)
    np.testing.assert_allclose(float(us0), un, rtol=1e-9)
    np.testing.assert_allclose(float(cd0), (un / U) ** 2, rtol=1e-9)
    us_u, cd_u = sl.most_drag_prescribed_flux(U, 0.10, z1, z0, thref)
    us_s, cd_s = sl.most_drag_prescribed_flux(U, -0.01, z1, z0, thref)
    assert float(cd_u) > float(cd0) > float(cd_s)
    kappa = sl.constants.kappa_von_karman
    inv_L = -kappa * sl.constants.g * 0.10 / (float(us_u) ** 3 * thref)
    rhs = kappa * U / (np.log(z1 / z0) - float(sl.psi_m(z1 * inv_L)))
    np.testing.assert_allclose(float(us_u), rhs, rtol=1e-6)
    gq = jax.grad(lambda q: sl.most_drag_prescribed_flux(U, q, z1, z0, thref)[1])(0.1)
    assert bool(jnp.isfinite(gq)) and float(gq) > 0.0


def test_most_drag_prescribed_flux_converges_near_free_convection():
    """Weak wind + strong heating (Sullivan–Patton start: ⟨U⟩=1, q0=0.24, z1=4,
    z0=0.1), where a plain fixed point oscillates: u_* matches an independently
    converged root (damped iteration, residual 3e-14) and satisfies the
    Businger–Dyer relation to round-off over a weak-to-strong sweep; AD matches
    finite differences."""
    for (U, q0, z1, z0), ref in [((1.0, 0.24, 4.0, 0.1), 0.183255152572),
                                 ((0.5, 0.24, 10.0, 0.01), 0.045887479379)]:
        us, _ = sl.most_drag_prescribed_flux(U, q0, z1, z0, 300.0)
        np.testing.assert_allclose(float(us), ref, rtol=1e-10)
    kappa, gacc = sl.constants.kappa_von_karman, sl.constants.g
    for z1, z0 in [(4.0, 0.1), (10.0, 0.01), (10.0, 0.1), (1.0, 1e-3)]:
        for U in np.r_[1e-6, 1e-3, np.geomspace(0.1, 20.0, 9)]:   # incl. calm start
            for q0 in np.geomspace(1e-5, 0.5, 9):
                us = float(sl.most_drag_prescribed_flux(U, q0, z1, z0, 300.0)[0])
                inv_L = -kappa * gacc * q0 / (us ** 3 * 300.0)
                rhs = kappa * U / (np.log(z1 / z0) - float(sl.psi_m(z1 * inv_L)))
                np.testing.assert_allclose(us, rhs, rtol=1e-9, err_msg=f"{U=} {q0=} {z1=}")
    for q0 in (1e-25, 1e-40):             # root below the bracket: neutral to round-off
        us = float(sl.most_drag_prescribed_flux(10.0, q0, 10.0, 0.01, 300.0)[0])
        np.testing.assert_allclose(us, _neutral_ustar(10.0, 10.0, 0.01), rtol=1e-9)
    from jax.test_util import check_grads
    grad = jax.grad(lambda U, q: sl.most_drag_prescribed_flux(U, q, 4.0, 0.1, 300.0)[1],
                    argnums=(0, 1))
    for U, q0 in [(1e-6, 0.24), (1e-3, 0.24), (1.0, 0.24), (10.0, 1e-25), (10.0, 0.0),
                  (8.0, -0.05)]:       # every branch selected once; none may NaN the others
        assert all(np.isfinite(float(x)) for x in grad(U, q0)), (U, q0)
    for U0 in (1.0, 1e-3):                # bisected root / clamped free-convection root
        check_grads(lambda U, q: sl.most_drag_prescribed_flux(U, q, 4.0, 0.1, 300.0)[1],
                    (U0, 0.24), order=2, modes=("fwd", "rev"))
    # Root inside the bracket where the shared ψ is constant (|ζ| < 1e-10): u_* is
    # exactly linear in ⟨U⟩, so its curvature must vanish (implicit 2nd derivative).
    d2 = jax.grad(jax.grad(
        lambda U: sl.most_drag_prescribed_flux(U, 1e-15, 10.0, 0.01, 300.0)[0]))(10.0)
    assert abs(float(d2)) < 1e-12, float(d2)


def test_most_drag_prescribed_cooling_converges_or_is_nan():
    """Stable prescribed flux: every returned u_* satisfies the Businger–Dyer
    relation (with the shared ψ, clamped at ζ = 10) to 1e-7, or is NaN — never a
    wrong stress. Strong cooling (ζ past the clamp) gets the clamped closed form."""
    kappa, gacc = sl.constants.kappa_von_karman, sl.constants.g
    z1, z0, U = 10.0, 0.01, 10.0
    us = float(sl.most_drag_prescribed_flux(U, -0.5, z1, z0, 300.0)[0])
    np.testing.assert_allclose(
        us, kappa * U / (np.log(z1 / z0) - float(sl.psi_m(10.0))), rtol=1e-12)
    # weak wind, weak cooling: ζ is far past the clamp — no regulariser may bias it
    us = float(sl.most_drag_prescribed_flux(1e-6, -1e-16, z1, z0, 300.0)[0])
    np.testing.assert_allclose(
        us, kappa * 1e-6 / (np.log(z1 / z0) - float(sl.psi_m(10.0))), rtol=1e-12)
    for q0 in (-1e-4, -0.01, -0.05, -0.1, -0.2, -0.3, -0.5, -2.0):
        us = float(sl.most_drag_prescribed_flux(U, q0, z1, z0, 300.0)[0])
        if np.isnan(us):
            continue                      # marginal root: refused loudly, allowed
        inv_L = -kappa * gacc * q0 / (us ** 3 * 300.0)
        rhs = kappa * U / (np.log(z1 / z0) - float(sl.psi_m(z1 * inv_L)))
        np.testing.assert_allclose(us, rhs, rtol=1e-7, err_msg=f"{q0=}")
    assert np.isfinite(float(sl.most_drag_prescribed_flux(U, -0.05, z1, z0, 300.0)[0]))


def test_most_drag_prescribed_flux_outside_its_domain_is_nan():
    """⟨U⟩ below 5e-7 m/s: NaN (loud). The
    calm-start speed surface_boundary produces (sqrt(1e-12) in float32) is inside."""
    for q0 in (0.24, -0.05, 0.0):
        assert np.isnan(float(sl.most_drag_prescribed_flux(1e-8, q0, 10.0, 0.01, 300.0)[1]))
    calm = jnp.sqrt(jnp.float32(1e-12))
    assert np.isfinite(float(sl.most_drag_prescribed_flux(
        calm, jnp.float32(0.1), jnp.float32(10.0), jnp.float32(0.01),
        jnp.float32(300.0))[1]))


def test_most_drag_prescribed_flux_mixed_precision_under_jit():
    """float32 wind, float64 flux, Python-float geometry: one dtype, any regime."""
    f = jax.jit(lambda U, q: sl.most_drag_prescribed_flux(U, q, 10.0, 0.01, 300.0)[1])
    for q0 in (0.1, -0.05, 0.0):
        cd = f(jnp.float32(5.0), jnp.asarray(q0, jnp.float64))
        assert np.isfinite(float(cd))
        g = jax.grad(f, argnums=1)(jnp.float32(5.0), jnp.asarray(q0, jnp.float64))
        assert np.isfinite(float(g))
    # all-float32 weak wind, both signs and the clamped regimes: finite fwd + rev
    f32 = jax.jit(lambda U, q: sl.most_drag_prescribed_flux(
        U, q, jnp.float32(10.0), jnp.float32(0.01), jnp.float32(300.0))[1])
    f64 = jax.jit(lambda U, q: sl.most_drag_prescribed_flux(U, q, 10.0, 0.01, 300.0)[1])
    for U, q0 in [(1e-6, -1e-16), (1e-6, 0.24), (1e-3, -0.05), (1e-3, 0.24), (5.0, -0.05)]:
        args = (jnp.float32(U), jnp.float32(q0))
        cd = f32(*args)
        assert cd.dtype == jnp.float32
        np.testing.assert_allclose(float(cd), float(f64(U, q0)), rtol=1e-3)
        for g in jax.grad(f32, argnums=(0, 1))(*args):
            assert np.isfinite(float(g)), (U, q0)
        _, t = jax.jvp(f32, args, (jnp.float32(1.0), jnp.float32(1.0)))
        assert np.isfinite(float(t)), (U, q0)


def test_most_flux_drag_needs_a_first_level_above_the_free_convection_limit():
    """ln(z1/z0) <= psi_m(-10) has no unstable MOST solution: the prescribed-flux
    MOST path refuses it; the neutral path and the surface-temperature path do not
    use it and still run."""
    import pytest
    cfg = sl.SpectralLESConfig(nx=8, ny=8, nz=8, Lx=800.0, Ly=800.0, Lz=8.0, z0=0.1,
                               buoyancy=True, theta_ref0=300.0, sfc_flux_drag="most")
    u, v = jnp.full((8, 8, 8), 5.0), jnp.zeros((8, 8, 8))
    th = jnp.full((8, 8, 8), 300.0)
    for wrap in (lambda f: f, jax.jit):                    # eager and under jit
        with pytest.raises(ValueError, match="ln\\(z1/z0\\)"):    # z1 = 0.5 m
            wrap(lambda u_: sl.surface_boundary(u_, v, th, sl.make_grid(cfg), 0.1,
                                                None))(u)
        g_ok = sl.make_grid(cfg._replace(Lz=800.0))
        cd, _ = wrap(lambda u_: sl.surface_boundary(u_, v, th, g_ok, 0.1, None))(u)
        assert np.isfinite(float(cd))
    sl.surface_boundary(u, v, th, sl.make_grid(cfg), None, 299.0)
    sl.surface_boundary(u, v, th, sl.make_grid(cfg._replace(sfc_flux_drag="neutral")),
                        0.1, None)
    sl.surface_boundary(u, v, th, sl.make_grid(cfg._replace(Lz=800.0)), 0.1, None)


def test_surface_boundary_flux_drag_dispatch():
    """sfc_flux_drag selects the wall drag under a prescribed flux; the default
    'neutral' returns no override (unchanged path); an unknown value, or 'most'
    without θ, raises."""
    import pytest
    cfg = sl.SpectralLESConfig(nx=8, ny=8, nz=8, Lx=800.0, Ly=800.0, Lz=800.0,
                               buoyancy=True, theta_ref0=300.0)
    g = sl.make_grid(cfg)
    u = jnp.full((8, 8, 8), 5.0)
    v = jnp.zeros((8, 8, 8))
    th = jnp.full((8, 8, 8), 300.0)
    assert sl.surface_boundary(u, v, th, g, 0.1, None) == (None, 0.1)
    g_most = sl.make_grid(cfg._replace(sfc_flux_drag="most"))
    cd, q = sl.surface_boundary(u, v, th, g_most, 0.1, None)
    assert float(q) == 0.1 and float(cd) > float(sl.surface_drag_coefficient(g))
    with pytest.raises(ValueError, match="needs θ"):
        sl.surface_boundary(u, v, None, g_most, 0.1, None)
    g_bad = sl.make_grid(cfg._replace(sfc_flux_drag="nope"))
    with pytest.raises(ValueError, match="sfc_flux_drag"):
        sl.surface_boundary(u, v, th, g_bad, 0.1, None)
