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

from legoesm.atmosphere.dynamics import spectral_les_plane as sl

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
