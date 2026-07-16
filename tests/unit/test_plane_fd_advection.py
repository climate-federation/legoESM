"""Unit tests for the shared plane flux-form FD advection.

Pins the advection invariants: a constant field has zero tendency under any
divergent velocity (flow-divergence correction); a uniform translation advects a
tracer in the right direction; pure horizontal advection conserves the domain
integral (periodic flux form); no flux crosses the rigid z walls; jit/grad safe.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.les import plane_fd_advection as adv

SCHEMES = ("upwind", "van_leer", "weno5", "weno7", "weno9", "central")


def test_central_is_nondissipative_and_weno_order_reduces_diffusion():
    """-u·∂φ/∂x of sin(kx) is pure -uk·cos(kx) — a scheme's DISSIPATION is the
    in-phase sin(kx) component it adds. Central adds none (energy-conserving, the
    fix for the pseudo core over-smoothing vs the spectral core); WENO is
    upwind-dissipative, and higher order (weno5→7→9) reduces that dissipation."""
    ny, nx, nz = 4, 32, 4
    x = 2.0 * np.pi * np.arange(nx) / nx
    s = np.sin(6.0 * x)                             # a high-k mode (~5 pts/wave) where upwind bites
    phi = jnp.broadcast_to(jnp.asarray(s)[None, :, None], (ny, nx, nz))
    u = jnp.ones((ny, nx, nz)); v = jnp.zeros((ny, nx, nz)); w = jnp.zeros((ny, nx, nz + 1))

    def dissipation(scheme):
        t = np.asarray(adv.advect_scalar(phi, u, v, w, 1.0, 1.0, 1.0, scheme))[0, :, 0]
        return abs(float(np.sum(t * s) / np.sum(s ** 2)))   # in-phase (dissipative) coeff

    d = {sc: dissipation(sc) for sc in ("central", "weno5", "weno7", "weno9")}
    assert d["central"] < 1e-9                      # central: NO numerical dissipation (energy-conserving)
    assert d["weno5"] > 1e-3                         # weno5 is strongly upwind-dissipative at high k
    assert d["weno9"] < d["weno7"] < d["weno5"]     # higher order => less dissipation


def _grid(ny=16, nx=16, nz=12):
    return dict(ny=ny, nx=nx, nz=nz, dx=50.0, dy=50.0, dz=40.0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_constant_field_zero_tendency(scheme):
    """A spatially constant scalar must have ZERO advective tendency even when the
    velocity field is divergent (the φ∇·u correction cancels the flux divergence)."""
    g = _grid()
    key = jax.random.PRNGKey(0)
    ku, kv, kw = jax.random.split(key, 3)
    u = jax.random.normal(ku, (g["ny"], g["nx"], g["nz"]))
    v = jax.random.normal(kv, (g["ny"], g["nx"], g["nz"]))
    w = jax.random.normal(kw, (g["ny"], g["nx"], g["nz"] + 1))
    w = w.at[..., 0].set(0.0).at[..., -1].set(0.0)
    phi = jnp.full((g["ny"], g["nx"], g["nz"]), 3.14)
    t = adv.advect_scalar(phi, u, v, w, g["dx"], g["dy"], g["dz"], scheme)
    assert float(jnp.max(jnp.abs(t))) < 1e-9


@pytest.mark.parametrize("scheme", SCHEMES)
def test_uniform_advection_sign(scheme):
    """Uniform +x wind on a Gaussian bump ⇒ tendency moves the bump in −x at the
    leading edge (−u ∂φ/∂x): tendency<0 where ∂φ/∂x>0."""
    g = _grid()
    x = (jnp.arange(g["nx"]) + 0.5) * g["dx"]
    bump = jnp.exp(-((x - x.mean()) / (3 * g["dx"])) ** 2)
    phi = jnp.broadcast_to(bump[None, :, None], (g["ny"], g["nx"], g["nz"]))
    u = jnp.ones_like(phi)            # +x wind, magnitude 1
    v = jnp.zeros_like(phi)
    w = jnp.zeros((g["ny"], g["nx"], g["nz"] + 1))
    t = adv.advect_scalar(phi, u, v, w, g["dx"], g["dy"], g["dz"], scheme)
    dphidx = (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1)) / (2 * g["dx"])
    # t ≈ −u ∂φ/∂x ⇒ strong NEGATIVE correlation with ∂φ/∂x (robust across schemes,
    # incl. diffusive upwind, vs a brittle per-cell sign match).
    corr = float(jnp.sum(t * dphidx) / (jnp.linalg.norm(t) * jnp.linalg.norm(dphidx)))
    assert corr < -0.9


@pytest.mark.parametrize("scheme", SCHEMES)
def test_horizontal_conservation(scheme):
    """Divergence-free horizontal flow ⇒ domain integral tendency ≈ 0 (conservation)."""
    g = _grid()
    key = jax.random.PRNGKey(3)
    phi = jax.random.uniform(key, (g["ny"], g["nx"], g["nz"]))
    u = jnp.full_like(phi, 2.0)       # uniform ⇒ ∇·u = 0
    v = jnp.full_like(phi, -1.0)
    w = jnp.zeros((g["ny"], g["nx"], g["nz"] + 1))
    t = adv.advect_scalar(phi, u, v, w, g["dx"], g["dy"], g["dz"], scheme)
    assert abs(float(jnp.sum(t))) < 1e-7 * float(jnp.sum(jnp.abs(phi)))


@pytest.mark.parametrize("scheme", SCHEMES)
def test_no_flux_through_walls(scheme):
    """No advective flux crosses the rigid z walls: the CONSERVATIVE flux part
    ``−∇·(uφ) = t − φ∇·u`` integrates to zero over the domain (it telescopes to the
    wall fluxes, which are zero by ``w=0`` at the walls). (The full advective
    tendency does NOT conserve when ∇·u≠0 — that is the φ∇·u term, by design.)"""
    g = _grid()
    zc = (jnp.arange(g["nz"]) + 0.5) * g["dz"]
    phi = jnp.broadcast_to(jnp.sin(zc / zc[-1] * np.pi)[None, None, :],
                           (g["ny"], g["nx"], g["nz"]))
    u = jnp.zeros_like(phi)
    v = jnp.zeros_like(phi)
    wf = jnp.zeros((g["ny"], g["nx"], g["nz"] + 1)).at[..., 1:-1].set(0.5)
    t = adv.advect_scalar(phi, u, v, wf, g["dx"], g["dy"], g["dz"], scheme)
    div = adv.divergence_centre(u, v, wf, g["dx"], g["dy"], g["dz"])
    flux_div_integral = float(jnp.sum(t - phi * div))      # = ∫ −∇·(uφ)
    assert abs(flux_div_integral) < 1e-7 * float(jnp.sum(jnp.abs(phi)))


@pytest.mark.parametrize("scheme", SCHEMES)
def test_vertical_advection_sign(scheme):
    """Uniform updraught w>0 on a z-varying scalar ⇒ t ≈ −w ∂φ/∂z: strong negative
    correlation with ∂φ/∂z (pins the _flux_div_z upwind side + direction)."""
    g = _grid()
    zc = (jnp.arange(g["nz"]) + 0.5) * g["dz"]
    bump = jnp.exp(-((zc - zc.mean()) / (3 * g["dz"])) ** 2)
    phi = jnp.broadcast_to(bump[None, None, :], (g["ny"], g["nx"], g["nz"]))
    u = jnp.zeros_like(phi)
    v = jnp.zeros_like(phi)
    w = jnp.full((g["ny"], g["nx"], g["nz"] + 1), 0.5).at[..., 0].set(0.0).at[..., -1].set(0.0)
    t = adv.advect_scalar(phi, u, v, w, g["dx"], g["dy"], g["dz"], scheme)
    dphidz = jnp.zeros_like(phi).at[..., 1:-1].set((phi[..., 2:] - phi[..., :-2]) / (2 * g["dz"]))
    interior = (slice(None), slice(None), slice(1, -1))
    ti, di = t[interior], dphidz[interior]
    corr = float(jnp.sum(ti * di) / (jnp.linalg.norm(ti) * jnp.linalg.norm(di)))
    assert corr < -0.85


def test_momentum_advection_sign():
    """Self-advection of a u-bump on a uniform +x background ⇒ au ≈ −U ∂u/∂x:
    negative correlation with ∂u/∂x (pins advect_momentum direction, not just shape)."""
    g = _grid()
    x = (jnp.arange(g["nx"]) + 0.5) * g["dx"]
    bump = 0.1 * jnp.exp(-((x - x.mean()) / (3 * g["dx"])) ** 2)
    U = 2.0
    u = jnp.broadcast_to((U + bump)[None, :, None], (g["ny"], g["nx"], g["nz"]))
    v = jnp.zeros_like(u)
    w = jnp.zeros((g["ny"], g["nx"], g["nz"] + 1))
    au, _av, _aw = adv.advect_momentum(u, v, w, g["dx"], g["dy"], g["dz"], "weno5")
    dudx = (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2 * g["dx"])
    mask = jnp.abs(dudx) > 1e-6
    corr = float(jnp.sum(au[mask] * dudx[mask])
                 / (jnp.linalg.norm(au[mask]) * jnp.linalg.norm(dudx[mask])))
    assert corr < -0.9


def test_momentum_shapes_and_walls():
    """advect_momentum returns correctly-shaped tendencies; w-tendency maps to faces."""
    g = _grid()
    key = jax.random.PRNGKey(5)
    ku, kv, kw = jax.random.split(key, 3)
    u = jax.random.normal(ku, (g["ny"], g["nx"], g["nz"]))
    v = jax.random.normal(kv, (g["ny"], g["nx"], g["nz"]))
    w = jax.random.normal(kw, (g["ny"], g["nx"], g["nz"] + 1)).at[..., 0].set(0.0).at[..., -1].set(0.0)
    au, av, aw = adv.advect_momentum(u, v, w, g["dx"], g["dy"], g["dz"], "weno5")
    assert au.shape == u.shape and av.shape == v.shape
    assert aw.shape == w.shape


@pytest.mark.parametrize("scheme", SCHEMES)
def test_cgrid_constant_field_zero_tendency(scheme):
    """C-grid (vel_at_faces=True): a constant scalar advected by face velocities has
    zero tendency under a divergent field (compact φ∇·u correction cancels the flux)."""
    g = _grid()
    key = jax.random.PRNGKey(1)
    ku, kv, kw = jax.random.split(key, 3)
    u = jax.random.normal(ku, (g["ny"], g["nx"], g["nz"]))      # x-face velocities
    v = jax.random.normal(kv, (g["ny"], g["nx"], g["nz"]))
    w = jax.random.normal(kw, (g["ny"], g["nx"], g["nz"] + 1))
    w = w.at[..., 0].set(0.0).at[..., -1].set(0.0)
    phi = jnp.full((g["ny"], g["nx"], g["nz"]), -2.5)
    t = adv.advect_scalar(phi, u, v, w, g["dx"], g["dy"], g["dz"], scheme,
                          vel_at_faces=True)
    assert float(jnp.max(jnp.abs(t))) < 1e-9


def test_cgrid_divergence_compact_vs_collocated():
    """C-grid compact divergence uses one-sided face differences (≠ centred 2Δ)."""
    g = _grid(8, 8, 6)
    key = jax.random.PRNGKey(4)
    u = jax.random.normal(key, (g["ny"], g["nx"], g["nz"]))
    v = jnp.zeros_like(u)
    w = jnp.zeros((g["ny"], g["nx"], g["nz"] + 1))
    d_face = adv.divergence_centre(u, v, w, g["dx"], g["dy"], g["dz"], vel_at_faces=True)
    expect = (u - jnp.roll(u, 1, axis=1)) / g["dx"]
    np.testing.assert_allclose(np.asarray(d_face), np.asarray(expect), rtol=1e-12)


def test_jit_and_grad_safe():
    g = _grid(8, 8, 6)
    key = jax.random.PRNGKey(9)
    phi = jax.random.normal(key, (g["ny"], g["nx"], g["nz"]))
    u = jnp.ones_like(phi)
    v = jnp.zeros_like(phi)
    w = jnp.zeros((g["ny"], g["nx"], g["nz"] + 1))

    @jax.jit
    def loss(s):
        t = adv.advect_scalar(s * phi, u, v, w, g["dx"], g["dy"], g["dz"], "weno5")
        return jnp.sum(t ** 2)

    assert np.isfinite(float(loss(1.0)))
    assert np.isfinite(float(jax.grad(loss)(1.0)))


def test_unknown_scheme_raises():
    g = _grid(8, 8, 6)
    z = jnp.zeros((g["ny"], g["nx"], g["nz"]))
    w = jnp.zeros((g["ny"], g["nx"], g["nz"] + 1))
    with pytest.raises(ValueError):
        adv.advect_scalar(z, z, z, w, g["dx"], g["dy"], g["dz"], "quintic")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
