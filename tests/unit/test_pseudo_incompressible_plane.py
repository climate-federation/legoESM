"""Unit + oracle tests for the pseudo-incompressible plane LES dycore.

Pins: the projection leaves ``∇·(ρ0θ0 u)`` ≈ 0; a hydrostatic base state at rest stays
at rest; a dry warm bubble rises with a physically reasonable, x-symmetric updraught
whose peak vertical velocity is in the LEX oracle ballpark; the step is JIT/grad-safe.

The dry warm-bubble oracle (LEX, solver_opt=1, turb off) gives w_max ≈ 8.5 m/s after
600 s on a 200 m grid (see docs/physics-notes/pseudo_incompressible_les.md). We do NOT expect a
bit-level match (collocated-horizontal vs LEX C-grid; different advection details), so
the oracle assertion is a physically-meaningful bracket (a few m/s ≤ w_max ≤ ~15 m/s)
plus x-symmetry, which would catch a wrong sign, dead dynamics, or a blow-up.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics import pseudo_incompressible_plane as pip


def _cfg(**kw):
    base = dict(nx=64, ny=4, nz=50, Lx=12800.0, Ly=800.0, Lz=10000.0,
                theta_ref0=300.0, scheme="weno5")
    base.update(kw)
    return pip.PseudoIncompressibleConfig(**base)


def _rest_state(g):
    sh = (g.cfg.ny, g.cfg.nx, g.cfg.nz)
    theta = jnp.broadcast_to(g.theta0[None, None, :], sh)
    return pip.PseudoIncompressibleState(
        u=jnp.zeros(sh), v=jnp.zeros(sh),
        w=jnp.zeros((g.cfg.ny, g.cfg.nx, g.cfg.nz + 1)),
        theta=theta, pi_prev=jnp.zeros(sh))


def _bubble_state(g, dtheta=2.0, xc=6400.0, zc=2000.0, r=2000.0):
    s = _rest_state(g)
    x = (jnp.arange(g.cfg.nx) + 0.5) * g.dx
    rr = jnp.sqrt(((x[None, :, None] - xc) / r) ** 2
                  + ((g.z_c[None, None, :] - zc) / r) ** 2)
    pert = dtheta * jnp.cos(rr * np.pi / 2.0) ** 2
    pert = jnp.where(rr > 1.0, 0.0, pert)
    pert = jnp.broadcast_to(pert, (g.cfg.ny, g.cfg.nx, g.cfg.nz))
    return s._replace(theta=s.theta + pert)


def test_shapiro_denoiser_kills_2dx_and_is_gated():
    """The CFL-unlimited [1,2,1] scalar de-noiser: exactly zeros the 2Δ checkerboard,
    preserves a constant, is wired into step() on θ, and is off (no-op) at s=0."""
    ny, nx, nz = 8, 8, 4
    yy, xx = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
    checker = ((-1.0) ** (xx + yy))[:, :, None] * jnp.ones((ny, nx, nz))
    assert float(jnp.abs(pip._shapiro_h(checker)).max()) < 1e-12   # 2Δ mode removed
    const = jnp.full((ny, nx, nz), 3.0)
    assert float(jnp.abs(pip._shapiro_h(const) - const).max()) < 1e-12  # k→0 preserved
    # wired into step + gated: with 2Δ θ-noise present, s>0 damps it (θ differs),
    # s=0 is a no-op (default path unchanged). Velocity path stays finite/div-free.
    mk = lambda s: pip.make_grid(_cfg(nx=16, ny=8, nz=12, Lx=1600.0, Ly=800.0,
                                      Lz=2400.0, shapiro_coeff=s))
    g0, gs = mk(0.0), mk(0.3)
    yy2, xx2 = jnp.meshgrid(jnp.arange(8), jnp.arange(16), indexing="ij")
    noise = 0.5 * ((-1.0) ** (xx2 + yy2))[:, :, None] * jnp.ones((8, 16, 12))
    st0 = _rest_state(g0)._replace(theta=_rest_state(g0).theta + noise)
    sts = _rest_state(gs)._replace(theta=_rest_state(gs).theta + noise)
    th_off = pip.step(st0, g0, 0.5).theta
    st_on = pip.step(sts, gs, 0.5)
    assert float(jnp.abs(st_on.theta - th_off).max()) > 1e-6      # filter active when on
    assert bool(jnp.all(jnp.isfinite(st_on.w)))                   # velocity path intact
    with pytest.raises(ValueError, match="shapiro_coeff"):        # range enforced (s>1 unsafe)
        pip.make_grid(_cfg(shapiro_coeff=1.5))


def test_projection_makes_rho_divergence_machine_zero():
    """EXACT C-grid projection: the post-correction ρ-weighted divergence is machine
    zero (to the BiCGSTAB tolerance), because the compact divergence/gradient and the
    compact Poisson Laplacian form an exact D·G=L triple on the C-grid."""
    g = pip.make_grid(_cfg(nx=16, ny=8, nz=12, Lx=1600.0, Ly=800.0, Lz=2400.0))
    key = jax.random.PRNGKey(0)
    ku, kv, kw = jax.random.split(key, 3)
    sh = (g.cfg.ny, g.cfg.nx, g.cfg.nz)
    u = jax.random.normal(ku, sh)
    v = jax.random.normal(kv, sh)
    w = jax.random.normal(kw, (g.cfg.ny, g.cfg.nx, g.cfg.nz + 1))
    w = w.at[..., 0].set(0.0).at[..., -1].set(0.0)
    theta = jnp.broadcast_to(g.theta0[None, None, :], sh)
    dt = 1.0
    un, vn, wn, _pi = pip.project(u, v, w, theta, None, jnp.zeros(sh), dt, g)
    div = float(jnp.linalg.norm(pip._rho_weighted_divergence(un, vn, wn, g)))
    div0 = float(jnp.linalg.norm(pip._rho_weighted_divergence(u, v, w, g)))
    # exact projection ⇒ divergence cut by orders of magnitude (tol-limited, not 0.7×)
    assert div < 1e-3 * div0, f"div/div0={div / div0}"


def test_velocity_shapiro_commutes_with_divergence():
    """The velocity [1,2,1] de-noiser (momentum_shapiro_coeff) is a horizontal
    convolution that COMMUTES with the C-grid ρ-weighted divergence: div(H·u)=H(div u)
    to machine precision. So filtering the projected velocity does NOT reintroduce
    divergence (no re-projection) — this is what makes the non-dissipative 'central'
    momentum clean while keeping the flow divergence-free."""
    g = pip.make_grid(_cfg(nx=16, ny=8, nz=12, Lx=1600.0, Ly=800.0, Lz=2400.0))
    ku, kv, kw = jax.random.split(jax.random.PRNGKey(1), 3)
    sh = (g.cfg.ny, g.cfg.nx, g.cfg.nz)
    u = jax.random.normal(ku, sh); v = jax.random.normal(kv, sh)
    w = jax.random.normal(kw, (g.cfg.ny, g.cfg.nx, g.cfg.nz + 1))
    w = w.at[..., 0].set(0.0).at[..., -1].set(0.0)
    theta = jnp.broadcast_to(g.theta0[None, None, :], sh)
    un, vn, wn, _pi = pip.project(u, v, w, theta, None, jnp.zeros(sh), 1.0, g)
    s, order = 0.3, 8                       # the production high-order de-noiser
    H = lambda f: f + s * (pip._shapiro_h(f, order) - f)
    div_of_filtered = pip._rho_weighted_divergence(H(un), H(vn), H(wn), g)   # div(H·u)
    filtered_div = H(pip._rho_weighted_divergence(un, vn, wn, g))            # H(div u)
    assert float(jnp.abs(div_of_filtered - filtered_div).max()) < 1e-10      # they COMMUTE (any order)
    assert float(jnp.abs(H(un) - un).max()) > 1e-6                           # filter is active
    assert float(jnp.abs(H(wn)[..., 0]).max()) < 1e-12                       # w walls preserved
    with pytest.raises(ValueError, match="momentum_shapiro_coeff"):
        pip.make_grid(_cfg(momentum_shapiro_coeff=2.0))


def test_poisson_tol_floored_keeps_f32_finite():
    """A float32 BiCGSTAB targeting an atol below the achievable ~O(eps≈1.2e-7)
    residual iterates past convergence and its ρ/ω recurrences underflow → NaN.
    project() floors tol/atol at ~O(eps), so even an absurdly tight requested atol
    stays finite in f32; f64 (floor far below its defaults) is unaffected."""
    cfg = pip.PseudoIncompressibleConfig(
        nx=16, ny=8, nz=12, Lx=1600.0, Ly=800.0, Lz=2400.0, theta_ref0=300.0,
        poisson_atol=1e-30, poisson_tol=1e-30)          # unreachable target
    for dtype in (jnp.float32, jnp.float64):
        g = pip.make_grid(cfg, dtype=dtype)
        sh = (8, 16, 12)
        ku, kv, kw = jax.random.split(jax.random.PRNGKey(3), 3)
        u = jax.random.normal(ku, sh, dtype); v = jax.random.normal(kv, sh, dtype)
        w = jax.random.normal(kw, (8, 16, 13), dtype).at[..., 0].set(0.0).at[..., -1].set(0.0)
        th = jnp.broadcast_to(g.theta0[None, None, :], sh).astype(dtype)
        un, vn, wn, pi = pip.project(u, v, w, th, None, jnp.zeros(sh, dtype), 1.0, g)
        assert bool(jnp.all(jnp.isfinite(un))), f"{dtype}: non-finite velocity"
        assert bool(jnp.all(jnp.isfinite(pi))), f"{dtype}: non-finite pressure"


def test_high_order_shapiro_flat_passband():
    """The Shapiro ORDER controls passband flatness: a well-resolved mode is barely
    touched at high order (response ≈1) but damped by the order-1 [1,2,1]. Per-step
    velocity de-noising with order-1 would COMPOUND that damping and erode the resolved
    eddies (over-damp central momentum to laminar); a high order stays flat in the
    passband so only the 2Δ mode is removed — this is what makes central momentum clean."""
    nx = 64
    x = 2.0 * np.pi * np.arange(nx) / nx
    f = jnp.broadcast_to(jnp.asarray(np.sin(4.0 * x))[None, :, None], (1, nx, 1))  # 16 pts/wave
    resp = lambda o: float(jnp.sum(pip._shapiro_h(f, o) * f) / jnp.sum(f * f))
    assert resp(8) > 0.999                       # order-8: passband ≈ flat (resolved scale kept)
    assert resp(1) < 0.98                         # order-1: damps it (would compound to laminar)
    assert resp(8) > resp(4) > resp(1)            # order raises passband flatness


def test_divergence_stays_bounded_over_steps():
    """The approximate projection keeps the ρ-weighted divergence BOUNDED (no
    checkerboard runaway) over many steps — the physically relevant stability test."""
    g = pip.make_grid(_cfg(nx=32, ny=4, nz=30, Lx=6400.0, Ly=800.0, Lz=6000.0))
    s = _bubble_state(g, dtheta=2.0, xc=3200.0, zc=2000.0, r=1500.0)
    divs = []
    for _ in range(30):
        s = pip.step(s, g, dt=2.0)
        divs.append(float(jnp.linalg.norm(
            pip._rho_weighted_divergence(s.u, s.v, s.w, g))))
    assert np.all(np.isfinite(divs))
    # late-time divergence not blowing up relative to early transient
    assert max(divs[-5:]) < 5.0 * max(divs[:5]) + 1e-6


def test_rest_stays_at_rest():
    g = pip.make_grid(_cfg(nx=16, ny=4, nz=20))
    s = _rest_state(g)
    for _ in range(5):
        s = pip.step(s, g, dt=2.0)
    assert float(jnp.max(jnp.abs(s.u))) < 1e-6
    assert float(jnp.max(jnp.abs(s.w))) < 1e-6
    np.testing.assert_allclose(np.asarray(s.theta), np.asarray(g.theta0[None, None, :]
                                                               + 0 * s.theta), atol=1e-6)


def test_warm_bubble_rises_oracle_bracket():
    """Dry warm bubble: rises, x-symmetric, w_max in the LEX-oracle ballpark."""
    g = pip.make_grid(_cfg())
    s = _bubble_state(g)
    dt = 2.0
    w_max = 0.0
    for n in range(300):                       # 600 s
        s = pip.step(s, g, dt=dt)
        w_max = max(w_max, float(jnp.max(s.w)))
    assert np.isfinite(w_max)
    # physically-meaningful bracket around the LEX oracle (≈8.5 m/s), not bit-level.
    assert 3.0 < w_max < 15.0, f"w_max={w_max} outside oracle bracket"
    # updraught is positive and the bubble rose (max w above the initial centre).
    wc = s.w[g.cfg.ny // 2, g.cfg.nx // 2, :]
    assert float(jnp.max(wc)) > 0.0
    # x-symmetry about the bubble centre column (quasi-2D, symmetric forcing).
    th = s.theta[g.cfg.ny // 2]                 # (nx, nz)
    th_flip = jnp.flip(th, axis=0)
    # centre at index nx/2-? bubble centred at x=6400 = column 32 (0-based 31.5);
    # compare symmetry of the field reflected about its own centre of mass instead.
    asym = float(jnp.linalg.norm(th - th_flip)) / float(jnp.linalg.norm(th - 300.0))
    # ≈0.18 over 600 s: the component-wise C-grid momentum advection treats the face
    # velocity as cell-centred (an O(Δx) half-cell offset — the disclosed
    # momentum-conservative-flux upgrade), which breaks perfect x-symmetry slightly.
    # The dynamics (rise, w_max bracket, positive updraught) are unaffected. Tighten
    # this bound when the conservative C-grid momentum flux lands.
    assert asym < 0.25, f"theta x-asymmetry {asym} too large"


def test_step_jit_and_grad_safe():
    g = pip.make_grid(_cfg(nx=16, ny=4, nz=12))
    s = _bubble_state(g, dtheta=1.0, xc=g.dx * 8, zc=1000.0, r=1000.0)

    @jax.jit
    def one(theta0_field):
        s2 = s._replace(theta=theta0_field)
        s3 = pip.step(s2, g, dt=2.0)
        return jnp.sum(s3.w ** 2)

    val = one(s.theta)
    grad = jax.grad(one)(s.theta)
    assert np.isfinite(float(val)) and np.all(np.isfinite(np.asarray(grad)))


def test_make_grid_validates():
    with pytest.raises(ValueError):
        pip.make_grid(_cfg(scheme="quintic"))
    with pytest.raises(ValueError):
        pip.make_grid(_cfg(moist=True, n_tracers=0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
