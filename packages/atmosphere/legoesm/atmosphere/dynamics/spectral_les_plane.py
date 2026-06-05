"""Pseudo-spectral INCOMPRESSIBLE boundary-layer LES on the doubly-periodic plane.

Faithful in algorithm to the **jax-alfa** LES oracle (Basu; the validation
reference for legoESM's ABL LES): horizontal derivatives are exact in Fourier
space (``rfft2``), the vertical is second-order finite difference on a grid
STAGGERED in ``w``, advection is the rotational (vorticity) form ``C = ω × u``
(de-aliased by the 3/2 rule), and incompressibility is enforced each step by a
fractional-step pressure PROJECTION (spectral-horizontal + tridiagonal-vertical
Poisson). Time stepping is Adams–Bashforth-2.

Why this core (vs the compressible plane dycore)
-----------------------------------------------
The compressible plane dycore needs acoustic off-centring + biharmonic hyperdiff
for stability; that numerical dissipation caps the effective Reynolds number
below the turbulence-sustaining threshold, so resolved ABL turbulence laminarises
(``docs/les_plane_turbulence_notes.md``). A pseudo-spectral incompressible solver
has NO acoustic mode and NO numerical hyperdiffusion — the ONLY dissipation is
the physical SGS model — so the effective Re is high and turbulence sustains,
exactly as in the oracle. This module is the faithful path to a quantitative
oracle match; it hosts the same wall-damped Smagorinsky family as the oracle
(constant-coefficient here; the Bou-Zeid scale-dependent dynamic coefficient
``_compute_scale_dependent_dynamic_smag_cs_plane`` is wired in a follow-up).

Grid / staggering
-----------------
Physical layout ``(ny, nx, nz)`` (``y`` axis 0, ``x`` axis 1), matching the rest
of the plane code. ``u, v`` live at cell CENTRES ``z_c = (k+½)Δz`` (k=0..nz-1);
``w`` lives at FACES ``z_f = kΔz`` (k=0..nz) with the rigid-lid / ground BC
``w[0] = w[nz] = 0``. Uniform ``Δz`` (oracle convention). Horizontal is periodic.

Status: NEUTRAL ABL (no buoyancy) — the cleanest oracle-comparison case. θ /
buoyancy for SBL/CBL is a follow-up; the projection, advection and wall model are
buoyancy-independent.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants


# --------------------------------------------------------------------------- #
# Config + grid + state                                                        #
# --------------------------------------------------------------------------- #
class SpectralLESConfig(NamedTuple):
    nx: int
    ny: int
    nz: int
    Lx: float
    Ly: float
    Lz: float
    z0: float = 0.1                 # aerodynamic roughness [m]
    c_s: float = 0.16              # Smagorinsky coefficient (wall-damped)
    wall_damping: bool = True      # cap l_m at κz near the surface (Mason 1989)
    dealias: bool = True           # 3/2-rule (2/3 truncation) de-aliasing
    nu_molecular: float = 0.0      # optional explicit viscosity (usually 0)


class SpectralLESGrid(NamedTuple):
    cfg: SpectralLESConfig
    dx: float
    dy: float
    dz: float
    z_c: jax.Array                 # (nz,) centre heights
    z_f: jax.Array                 # (nz+1,) face heights
    kx: jax.Array                  # (ny, nx//2+1) rad/m, x-wavenumber
    ky: jax.Array                  # (ny, nx//2+1) rad/m, y-wavenumber
    k2: jax.Array                  # kx²+ky²
    dealias_mask: jax.Array        # (ny, nx//2+1) 2/3 truncation mask


class SpectralLESState(NamedTuple):
    u: jax.Array                   # (ny, nx, nz)   centres
    v: jax.Array                   # (ny, nx, nz)   centres
    w: jax.Array                   # (ny, nx, nz+1) faces, w[..,0]=w[..,nz]=0
    rhs_u_prev: jax.Array          # AB2 previous tendencies
    rhs_v_prev: jax.Array
    rhs_w_prev: jax.Array


def make_grid(cfg: SpectralLESConfig, dtype=jnp.float64) -> SpectralLESGrid:
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    dx, dy, dz = cfg.Lx / nx, cfg.Ly / ny, cfg.Lz / nz
    z_c = (jnp.arange(nz, dtype=dtype) + 0.5) * dz
    z_f = jnp.arange(nz + 1, dtype=dtype) * dz
    # rfft2 over (y, x): axis-0 (y) full length ny, axis-1 (x) reduced to nx//2+1.
    kx_1d = 2.0 * jnp.pi * jnp.fft.rfftfreq(nx, d=dx).astype(dtype)      # (nx//2+1,)
    ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy).astype(dtype)       # (ny,)
    # Zero the Nyquist wavenumbers: the spectral derivative of the Nyquist mode
    # is not representable on the grid (it would alias), so it must be excluded
    # from BOTH the gradient/divergence AND the pressure Laplacian k² to keep the
    # projection EXACTLY consistent (div∘grad = Laplacian). Without this the
    # corrected field retains an O(10%) Nyquist divergence.
    if nx % 2 == 0:
        kx_1d = kx_1d.at[-1].set(0.0)
    if ny % 2 == 0:
        ky_1d = ky_1d.at[ny // 2].set(0.0)
    kx = jnp.broadcast_to(kx_1d[None, :], (ny, kx_1d.size))
    ky = jnp.broadcast_to(ky_1d[:, None], (ny, kx_1d.size))
    k2 = kx ** 2 + ky ** 2
    # 2/3-rule de-alias mask (truncate |k| above 2/3 Nyquist on each axis).
    cut_x = nx // 3
    cut_y = ny // 3
    iy = jnp.arange(ny)
    fold_y = jnp.minimum(iy, ny - iy)
    mask = ((fold_y[:, None] <= cut_y) & (jnp.arange(kx_1d.size)[None, :] <= cut_x)
            ).astype(dtype)
    return SpectralLESGrid(cfg=cfg, dx=dx, dy=dy, dz=dz, z_c=z_c, z_f=z_f,
                           kx=kx, ky=ky, k2=k2, dealias_mask=mask)


# --------------------------------------------------------------------------- #
# Horizontal spectral derivatives (exact)                                      #
# --------------------------------------------------------------------------- #
def _fft(f):
    return jnp.fft.rfft2(f, axes=(0, 1))


def _ifft(fh, ny, nx):
    return jnp.fft.irfft2(fh, axes=(0, 1), s=(ny, nx))


def ddx(f, g: SpectralLESGrid):
    """∂/∂x via spectral (exact); Nyquist x-mode killed for a real derivative."""
    ny, nx = f.shape[0], f.shape[1]
    fh = _fft(f) * (1j * g.kx[..., None])
    return _ifft(fh, ny, nx)


def ddy(f, g: SpectralLESGrid):
    ny, nx = f.shape[0], f.shape[1]
    fh = _fft(f) * (1j * g.ky[..., None])
    return _ifft(fh, ny, nx)


def _dealias(f, g: SpectralLESGrid):
    if not g.cfg.dealias:
        return f
    ny, nx = f.shape[0], f.shape[1]
    return _ifft(_fft(f) * g.dealias_mask[..., None], ny, nx)


# --------------------------------------------------------------------------- #
# Vertical staggering helpers (uniform Δz)                                     #
# --------------------------------------------------------------------------- #
def c2f(fc):
    """Centre (nz) → interior face (nz-1) average."""
    return 0.5 * (fc[..., :-1] + fc[..., 1:])


def f2c(ff):
    """Face (nz+1) → centre (nz) average."""
    return 0.5 * (ff[..., :-1] + ff[..., 1:])


def ddz_c2f(fc, dz):
    """∂/∂z of a CENTRE field at interior FACES (nz-1) = (f[k]-f[k-1])/Δz."""
    return (fc[..., 1:] - fc[..., :-1]) / dz


def ddz_f2c(ff, dz):
    """∂/∂z of a FACE field at CENTRES (nz) = (f[k+1]-f[k])/Δz."""
    return (ff[..., 1:] - ff[..., :-1]) / dz


# --------------------------------------------------------------------------- #
# Strain rate + Smagorinsky eddy viscosity                                     #
# --------------------------------------------------------------------------- #
def _strain(u, v, w, g: SpectralLESGrid):
    """Full resolved strain ``S_ij`` and ``|S|=√(2 S_ij S_ij)`` at CENTRES."""
    dz = g.dz
    dudx, dvdx, dwdx = ddx(u, g), ddx(v, g), ddx(w, g)   # dwdx at faces
    dudy, dvdy, dwdy = ddy(u, g), ddy(v, g), ddy(w, g)
    dwdz_c = ddz_f2c(w, dz)                               # ∂w/∂z at centres
    # ∂u/∂z, ∂v/∂z at faces → centre.
    dudz_f = jnp.pad(ddz_c2f(u, dz), ((0, 0), (0, 0), (1, 1)), mode="edge")
    dvdz_f = jnp.pad(ddz_c2f(v, dz), ((0, 0), (0, 0), (1, 1)), mode="edge")
    dudz_c = f2c(dudz_f)
    dvdz_c = f2c(dvdz_f)
    dwdx_c = f2c(dwdx)
    dwdy_c = f2c(dwdy)
    S11, S22, S33 = dudx, dvdy, dwdz_c
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz_c + dwdx_c)
    S23 = 0.5 * (dvdz_c + dwdy_c)
    Smag = jnp.sqrt(jnp.maximum(
        2.0 * (S11 ** 2 + S22 ** 2 + S33 ** 2
               + 2.0 * (S12 ** 2 + S13 ** 2 + S23 ** 2)), 1e-30))
    return (S11, S22, S33, S12, S13, S23), Smag


def eddy_viscosity(u, v, w, g: SpectralLESGrid):
    """Wall-damped constant-coefficient Smagorinsky ``ν_t=(C_s l)²|S|`` at centres.

    ``l = min(C_s Δ, κ z)`` (Mason 1989) with ``Δ=(Δx Δy Δz)^⅓``. This is the
    oracle's static-SGS option; the scale-dependent dynamic coefficient is wired
    separately."""
    _, Smag = _strain(u, v, w, g)
    delta = (g.dx * g.dy * g.dz) ** (1.0 / 3.0)
    l_smag = g.cfg.c_s * delta
    if g.cfg.wall_damping:
        kappa = constants.kappa_von_karman
        l_m = jnp.minimum(l_smag, kappa * g.z_c)            # (nz,)
    else:
        l_m = l_smag
    return (l_m ** 2) * Smag                                # (ny,nx,nz)


# --------------------------------------------------------------------------- #
# Rotational-form advection (de-aliased)                                        #
# --------------------------------------------------------------------------- #
def advection(u, v, w, g: SpectralLESGrid):
    """Rotational (vorticity) form ``C = ω × u`` — the energy-conserving form the
    oracle uses (the Bernoulli ½|u|² term is absorbed by the pressure). Returns
    ``(Cu, Cv, Cw)`` as MINUS the advective tendency (i.e. the RHS contribution
    ``-C``). ``Cu, Cv`` at centres, ``Cw`` at interior faces."""
    dz = g.dz
    # de-alias the velocities entering the products (3/2-rule analogue).
    ud, vd = _dealias(u, g), _dealias(v, g)
    wd = _dealias(w, g)
    wc = f2c(wd)                                            # w at centres
    # Vorticity components.
    dudy, dvdx = ddy(ud, g), ddx(vd, g)
    omega_z = dvdx - dudy                                   # centres
    dudz_f = jnp.pad(ddz_c2f(ud, dz), ((0, 0), (0, 0), (1, 1)), mode="edge")
    dvdz_f = jnp.pad(ddz_c2f(vd, dz), ((0, 0), (0, 0), (1, 1)), mode="edge")
    dwdx_f, dwdy_f = ddx(wd, g), ddy(wd, g)                 # faces
    omega_x_f = dvdz_f - dwdy_f                             # faces
    omega_y_f = dwdx_f - dudz_f                             # faces
    # Centre tendencies: Cu = (ω × u)_x = ω_y w − ω_z v ; Cv = ω_z u − ω_x w.
    omega_x_c = f2c(omega_x_f)
    omega_y_c = f2c(omega_y_f)
    Cu = omega_y_c * wc - omega_z * vd
    Cv = omega_z * ud - omega_x_c * wc
    # Face tendency: Cw = (ω × u)_z = ω_x v − ω_y u, with u,v averaged to faces.
    uf = jnp.pad(c2f(ud), ((0, 0), (0, 0), (1, 1)))         # 0 at walls
    vf = jnp.pad(c2f(vd), ((0, 0), (0, 0), (1, 1)))
    omega_x_face = jnp.pad(c2f(omega_x_c), ((0, 0), (0, 0), (1, 1)), mode="edge")
    omega_y_face = jnp.pad(c2f(omega_y_c), ((0, 0), (0, 0), (1, 1)), mode="edge")
    Cw = omega_x_face * vf - omega_y_face * uf              # faces (nz+1)
    # de-alias the products back to the resolved grid.
    Cu, Cv = _dealias(Cu, g), _dealias(Cv, g)
    Cw = _dealias(Cw, g)
    return Cu, Cv, Cw


# --------------------------------------------------------------------------- #
# SGS stress divergence + MOST wall model                                      #
# --------------------------------------------------------------------------- #
def sgs_and_wall(u, v, w, nu_t, g: SpectralLESGrid, u_geo):
    """SGS force ``∂_j(2 ν_t S_ij)`` with the surface vertical momentum flux
    replaced by the Monin–Obukhov (neutral) wall stress.

    Returns ``(Fu, Fv, Fw)`` (centres, centres, faces) and the surface friction
    velocity ``u_*`` (diagnostic)."""
    dz = g.dz
    kappa = constants.kappa_von_karman
    (S11, S22, S33, S12, S13, S23), _ = _strain(u, v, w, g)
    tau11, tau22, tau33 = 2 * nu_t * S11, 2 * nu_t * S22, 2 * nu_t * S33
    tau12 = 2 * nu_t * S12
    tau13, tau23 = 2 * nu_t * S13, 2 * nu_t * S23           # centres
    # Horizontal divergence of the stress (spectral, exact).
    Fu = ddx(tau11, g) + ddy(tau12, g)
    Fv = ddx(tau12, g) + ddy(tau22, g)
    # Vertical SGS momentum flux on FACES: τ_i3 = 2 ν_t S_i3, interpolated to
    # faces; the SURFACE face value is the MOST wall stress (not the interior
    # closure), the rigid-lid top is no-stress.
    nu_t_f = c2f(nu_t)
    dudz_f = ddz_c2f(u, dz)
    dvdz_f = ddz_c2f(v, dz)
    tau13_f = nu_t_f * dudz_f                               # interior faces (nz-1)
    tau23_f = nu_t_f * dvdz_f
    # MOST neutral wall stress at the first centre level z_c[0].
    u1, v1 = u[..., 0], v[..., 0]
    spd1 = jnp.sqrt(u1 ** 2 + v1 ** 2 + 1e-12)
    Cd = (kappa / jnp.log(g.z_c[0] / g.cfg.z0)) ** 2
    tau_w_x = -Cd * spd1 * u1                               # surface stress (kinematic)
    tau_w_y = -Cd * spd1 * v1
    u_star = (Cd ** 0.5) * jnp.sqrt(jnp.mean(spd1 ** 2))
    # Assemble full-face stress: [wall, interior, top=0].
    z = jnp.zeros_like(u1)[..., None]
    tau13_full = jnp.concatenate([(-tau_w_x)[..., None], tau13_f, z], axis=-1)
    tau23_full = jnp.concatenate([(-tau_w_y)[..., None], tau23_f, z], axis=-1)
    # ∂τ_i3/∂z at centres.
    Fu = Fu + ddz_f2c(tau13_full, dz)
    Fv = Fv + ddz_f2c(tau23_full, dz)
    # Vertical SGS force on interior FACES: ∂_x τ13 + ∂_y τ23 + ∂_z τ33.
    #   τ13_f, τ23_f live on interior faces (nz-1); τ33 at centres (nz).
    Fw_int = (ddx(tau13_f, g) + ddy(tau23_f, g)
              + ddz_c2f(tau33, dz))                        # (ny, nx, nz-1)
    Fw = jnp.pad(Fw_int, ((0, 0), (0, 0), (1, 1)))         # 0 at the walls → nz+1
    return Fu, Fv, Fw, u_star


# --------------------------------------------------------------------------- #
# Pressure projection (fractional step)                                        #
# --------------------------------------------------------------------------- #
def project(u_s, v_s, w_s, dt, g: SpectralLESGrid):
    """Enforce ``∇·u=0`` by a pressure projection. Solve per horizontal
    wavenumber a tridiagonal vertical Poisson ``φ''−k²φ = div(u*)/dt`` with
    Neumann (``w=0``) walls, then ``u = u* − dt ∇φ``."""
    ny, nx, nz = u_s.shape
    dz = g.dz
    uh = _fft(u_s) * (1j * g.kx[..., None])
    vh = _fft(v_s) * (1j * g.ky[..., None])
    # divergence at centres: i kx û + i ky v̂ + ∂w/∂z|_c
    wzh = _fft(ddz_f2c(w_s, dz))
    div_h = uh + vh + wzh                                   # (ny, nxr, nz) complex
    rhs = div_h / dt
    # Tridiagonal in z (centres), Neumann walls: φ[-1]=φ[0], φ[nz]=φ[nz-1].
    k2 = g.k2[..., None]                                    # (ny, nxr, 1)
    inv_dz2 = 1.0 / dz ** 2
    a = jnp.full((ny, g.kx.shape[1], nz), inv_dz2, dtype=rhs.dtype)   # sub
    c = jnp.full((ny, g.kx.shape[1], nz), inv_dz2, dtype=rhs.dtype)   # super
    b = -(2.0 * inv_dz2 + k2) * jnp.ones_like(a)
    # Neumann: top/bottom diagonal loses one neighbour coupling.
    b = b.at[..., 0].set(-(inv_dz2 + k2[..., 0]))
    b = b.at[..., -1].set(-(inv_dz2 + k2[..., 0]))
    a = a.at[..., 0].set(0.0)
    c = c.at[..., -1].set(0.0)
    # k²=0 modes (the global mean + the Nyquist-zeroed columns) still need their
    # VERTICAL Poisson solved (dw/dz ≠ 0 there); the pure-Neumann vertical matrix
    # is singular by one constant, so pin only the GAUGE — φ[0]=0 — and keep the
    # interior vertical Laplacian. (Pinning the whole mode to 0 would leave the
    # vertical divergence of these columns uncorrected — an O(5%) leak.)
    sing = (g.k2 == 0.0)                                   # (ny, nxr)
    s0 = sing[..., None]
    b = b.at[..., 0].set(jnp.where(sing, 1.0, b[..., 0]))
    c = c.at[..., 0].set(jnp.where(sing, 0.0, c[..., 0]))
    rhs = rhs.at[..., 0].set(jnp.where(sing, 0.0 + 0.0j, rhs[..., 0]))
    del s0
    phi_h = _thomas_complex(a, b, c, rhs)
    # u = u* - dt ∇φ.  Horizontal grad spectral; vertical grad to faces.
    phi = _ifft(phi_h, ny, nx)
    u_new = u_s - dt * ddx(phi, g)
    v_new = v_s - dt * ddy(phi, g)
    dphidz_f = jnp.pad(ddz_c2f(phi, dz), ((0, 0), (0, 0), (1, 1)))    # 0 at walls
    w_new = w_s - dt * dphidz_f
    w_new = w_new.at[..., 0].set(0.0).at[..., -1].set(0.0)
    return u_new, v_new, w_new


def _thomas_complex(a, b, c, d):
    """Thomas tridiagonal solve along the LAST axis (complex RHS, real coeffs).

    Static Python loop over ``n`` (= nz, small) so it fully unrolls under JIT —
    robust (no scan-reconstruction indexing) and exact for the projection."""
    n = d.shape[-1]
    cp = [None] * n
    dp = [None] * n
    cp[0] = c[..., 0] / b[..., 0]
    dp[0] = d[..., 0] / b[..., 0]
    for k in range(1, n):
        m = b[..., k] - a[..., k] * cp[k - 1]
        cp[k] = c[..., k] / m
        dp[k] = (d[..., k] - a[..., k] * dp[k - 1]) / m
    x = [None] * n
    x[n - 1] = dp[n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return jnp.stack(x, axis=-1)


# --------------------------------------------------------------------------- #
# One AB2 time step                                                            #
# --------------------------------------------------------------------------- #
def rhs(u, v, w, g: SpectralLESGrid, u_geo, f_cor, force=(0.0, 0.0)):
    """Momentum RHS = -advection + SGS force + Coriolis + a constant body force.

    ``f_cor``≠0 drives a geostrophic/Ekman balance toward ``u_geo=(ug,vg)``; a
    constant ``force=(fx,fy)`` drives a pressure-gradient channel (``fx=u_*²/Lz``
    gives a target ``u_*`` and a log-law equilibrium in a few eddy turnovers —
    the clean Monin–Obukhov validation case)."""
    Cu, Cv, Cw = advection(u, v, w, g)
    nu_t = eddy_viscosity(u, v, w, g)
    Fu, Fv, Fw, u_star = sgs_and_wall(u, v, w, nu_t, g, u_geo)
    ug, vg = u_geo
    Ru = Cu + Fu + f_cor * (v - vg) + force[0]
    Rv = Cv + Fv - f_cor * (u - ug) + force[1]
    Rw = Cw + Fw
    Rw = Rw.at[..., 0].set(0.0).at[..., -1].set(0.0)
    return Ru, Rv, Rw, u_star


def step(state: SpectralLESState, g: SpectralLESGrid, dt: float,
         u_geo, f_cor: float, first: bool = False, force=(0.0, 0.0)):
    """One Adams–Bashforth-2 step + pressure projection. ``first`` uses forward
    Euler (no previous RHS yet)."""
    u, v, w = state.u, state.v, state.w
    Ru, Rv, Rw, u_star = rhs(u, v, w, g, u_geo, f_cor, force=force)
    if first:
        au, av, aw = Ru, Rv, Rw
    else:
        au = 1.5 * Ru - 0.5 * state.rhs_u_prev
        av = 1.5 * Rv - 0.5 * state.rhs_v_prev
        aw = 1.5 * Rw - 0.5 * state.rhs_w_prev
    u_s = u + dt * au
    v_s = v + dt * av
    w_s = w + dt * aw
    w_s = w_s.at[..., 0].set(0.0).at[..., -1].set(0.0)
    u_n, v_n, w_n = project(u_s, v_s, w_s, dt, g)
    return SpectralLESState(u=u_n, v=v_n, w=w_n,
                            rhs_u_prev=Ru, rhs_v_prev=Rv, rhs_w_prev=Rw), u_star
