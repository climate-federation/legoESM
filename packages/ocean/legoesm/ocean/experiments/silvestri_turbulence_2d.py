"""Silvestri et al. 2024 §4 — 2D decaying homogeneous turbulence test.

A standalone Cartesian doubly-periodic 2D Navier–Stokes solver (vorticity form)
that drives the CANONICAL WENO vorticity-flux reconstruction kernels
(``weno_reconstruct_split``) to reproduce the paper's §4 comparison: WENO vs the
Leith closure vs DNS, and the {ζ;u} (W*V) vs {ζ;ζ} (W*D) smoothness distinction.

This is NOT the ocean model — §4 is a Cartesian periodic box (2π×2π, non-
rotating), a different geometry from the lat-lon channel. Per the reproduction
plan (TEST CASE A), the faithful test of the *scheme* is to exercise the WENO
vorticity-flux kernel in a 2D harness; mimicry-glue lives here, not in the model.

Equations (non-dimensional, Eq 46-47): ∂ₜζ + u·∇ζ = (1/Re)∇²ζ, ∇·u=0,
with u=(−∂ψ/∂y, ∂ψ/∂x), ∇²ψ=−ζ. Re=3.3e4. Advection is flux-form
∂ₜζ = −∂ₓ(uζ) − ∂ᵧ(vζ) (since ∇·u=0), with ζ at faces reconstructed by WENO
upwinding (the scheme under test) or 2nd-order centered (DNS/Leith). Velocity
from vorticity and the molecular Laplacian are spectral (FFT). SSP-RK3 in time.

IC (Ishiko et al. 2009, Eqs 48-50): narrow-band spectrum
E(k)=½ a_s k_p⁻¹ (k/k_p)⁷ exp[−7/2 (k/k_p)²], a_s=16/3, k_p=12; ζ̂=[k/π E(k)]^½ e^{iφ}
with random phases (real ζ); û=i k_y/k² ζ̂, v̂=−i k_x/k² ζ̂.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from legoesm.core.weno import weno_reconstruct_split, weno_upwind


# --- Scheme matrix (Table 1): vorticity-flux variants -----------------------
class Turb2DScheme(NamedTuple):
    name: str
    advection: str          # "centered" (DNS/Leith) or "weno"
    weno_order: int         # 5 or 9 (ignored for centered)
    weno_smoothness: str    # "split" ({ζ;u}) or "standard" ({ζ;ζ})
    leith_C: float          # >0 → add Leith viscosity (centered cases)


SILVESTRI_TURB2D_SCHEMES = {
    "DNS":  Turb2DScheme("DNS",  "centered", 0, "standard", 0.0),
    "Leith1": Turb2DScheme("Leith1", "centered", 0, "standard", 1.0),
    "Leith2": Turb2DScheme("Leith2", "centered", 0, "standard", 2.0),
    "W5D":  Turb2DScheme("W5D",  "weno", 5, "standard", 0.0),
    "W9D":  Turb2DScheme("W9D",  "weno", 9, "standard", 0.0),
    "W5V":  Turb2DScheme("W5V",  "weno", 5, "split", 0.0),
    "W9V":  Turb2DScheme("W9V",  "weno", 9, "split", 0.0),
}


def _wavenumbers(N: int, L: float = 2.0 * np.pi):
    k1 = np.fft.fftfreq(N, d=L / N) * 2.0 * np.pi      # angular [1/length]
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    k2_inv = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0)
    return kx, ky, k2, k2_inv


def ishiko_initial_vorticity(N: int, *, k_p: float = 12.0, a_s: float = 16.0 / 3.0,
                             seed: int = 0) -> np.ndarray:
    """Ishiko (2009) narrow-band initial vorticity field ζ(x,y) (real, N×N)."""
    kx, ky, k2, _ = _wavenumbers(N)
    k = np.sqrt(k2)
    with np.errstate(divide="ignore", invalid="ignore"):
        E = 0.5 * a_s * (1.0 / k_p) * (k / k_p) ** 7 * np.exp(-3.5 * (k / k_p) ** 2)
    E = np.where(k > 0, E, 0.0)
    amp = np.sqrt(np.maximum(k / np.pi * E, 0.0))      # |ζ̂|
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(N, N))
    zeta_hat = amp * np.exp(1j * phase)
    zeta = np.real(np.fft.ifft2(zeta_hat))
    # Re-symmetrise so the field is exactly real & zero-mean.
    zeta = zeta - zeta.mean()
    return zeta


def velocity_from_vorticity(zeta: jnp.ndarray, kx, ky, k2_inv):
    """Divergence-free velocity (u,v) at cell centres from vorticity (spectral).
    ψ̂=ζ̂/k², u=−∂ψ/∂y, v=∂ψ/∂x."""
    zhat = jnp.fft.fft2(zeta)
    # ζ = ∂ₓv − ∂ᵧu = ∇²ψ ⇒ ψ̂ = −ζ̂/k² (the minus is essential: ψ̂=+ζ̂/k²
    # inverts the velocity so curl(u,v) returns −ζ and reverses the advection).
    psi_hat = -zhat * k2_inv
    u = jnp.real(jnp.fft.ifft2(-1j * ky * psi_hat))
    v = jnp.real(jnp.fft.ifft2(1j * kx * psi_hat))
    return u, v


def _weno_face_recon(field, psi, vel_face, axis, order):
    """WENO upwind reconstruction of ``field`` at the faces along ``axis``
    (periodic), with smoothness from ``psi``, selected by ``vel_face`` sign.

    Face j sits between cells j-1 and j. Returns the face values (same shape).
    """
    hw = {5: 3, 7: 4, 9: 5}[order]
    phi_st = [jnp.roll(field, hw - s, axis=axis) for s in range(2 * hw)]
    psi_st = [jnp.roll(psi, hw - s, axis=axis) for s in range(2 * hw)]
    f_plus, f_minus = weno_reconstruct_split(phi_st, psi_st, order=order)
    return weno_upwind(f_plus, f_minus, vel_face)


def _flux_div_weno(zeta, u, v, order, smoothness):
    """−∂ₓ(uζ) − ∂ᵧ(vζ) with ζ WENO-reconstructed at faces (the scheme under test)."""
    # Face velocities (2-pt average of cell-centred u,v; periodic).
    u_xface = 0.5 * (jnp.roll(u, 1, axis=0) + u)        # x-faces (axis 0)
    v_yface = 0.5 * (jnp.roll(v, 1, axis=1) + v)        # y-faces (axis 1)
    psi_x = u if smoothness == "split" else zeta
    psi_y = v if smoothness == "split" else zeta
    zeta_xface = _weno_face_recon(zeta, psi_x, u_xface, axis=0, order=order)
    zeta_yface = _weno_face_recon(zeta, psi_y, v_yface, axis=1, order=order)
    Fx = u_xface * zeta_xface                            # flux at x-faces
    Fy = v_yface * zeta_yface
    N = zeta.shape[0]
    dx = 2.0 * np.pi / N
    div = (jnp.roll(Fx, -1, axis=0) - Fx) / dx + (jnp.roll(Fy, -1, axis=1) - Fy) / dx
    return -div


def _flux_div_centered(zeta, u, v):
    """2nd-order centered flux divergence (DNS / Leith base; dispersive)."""
    u_xface = 0.5 * (jnp.roll(u, 1, axis=0) + u)
    v_yface = 0.5 * (jnp.roll(v, 1, axis=1) + v)
    zeta_xface = 0.5 * (jnp.roll(zeta, 1, axis=0) + zeta)
    zeta_yface = 0.5 * (jnp.roll(zeta, 1, axis=1) + zeta)
    Fx = u_xface * zeta_xface
    Fy = v_yface * zeta_yface
    N = zeta.shape[0]
    dx = 2.0 * np.pi / N
    div = (jnp.roll(Fx, -1, axis=0) - Fx) / dx + (jnp.roll(Fy, -1, axis=1) - Fy) / dx
    return -div


def _laplacian_spectral(zeta, k2):
    return jnp.real(jnp.fft.ifft2(-k2 * jnp.fft.fft2(zeta)))


def _leith_viscosity_term(zeta, C, dx):
    """Leith Laplacian dissipation ∇·(ν_L ∇ζ), ν_L=(C dx/π)³ |∇ζ| (2D Leith, Eq A1)."""
    dzdx = (jnp.roll(zeta, -1, axis=0) - jnp.roll(zeta, 1, axis=0)) / (2 * dx)
    dzdy = (jnp.roll(zeta, -1, axis=1) - jnp.roll(zeta, 1, axis=1)) / (2 * dx)
    grad = jnp.sqrt(dzdx ** 2 + dzdy ** 2 + 1e-30)
    nu = (C * dx / np.pi) ** 3 * grad
    # ∇·(ν ∇ζ) via centered flux of ν·∇ζ.
    fx = 0.5 * (nu + jnp.roll(nu, -1, axis=0)) * (jnp.roll(zeta, -1, axis=0) - zeta) / dx
    fy = 0.5 * (nu + jnp.roll(nu, -1, axis=1)) * (jnp.roll(zeta, -1, axis=1) - zeta) / dx
    return (fx - jnp.roll(fx, 1, axis=0)) / dx + (fy - jnp.roll(fy, 1, axis=1)) / dx


def make_rhs(scheme: Turb2DScheme, Re: float, N: int):
    kx, ky, k2, k2_inv = (jnp.asarray(a) for a in _wavenumbers(N))
    dx = 2.0 * np.pi / N

    def rhs(zeta):
        u, v = velocity_from_vorticity(zeta, kx, ky, k2_inv)
        if scheme.advection == "weno":
            adv = _flux_div_weno(zeta, u, v, scheme.weno_order, scheme.weno_smoothness)
        else:
            adv = _flux_div_centered(zeta, u, v)
        visc = (1.0 / Re) * _laplacian_spectral(zeta, k2)
        if scheme.leith_C > 0.0:
            visc = visc + _leith_viscosity_term(zeta, scheme.leith_C, dx)
        return adv + visc

    return rhs


def step_ssp_rk3(zeta, dt, rhs):
    z1 = zeta + dt * rhs(zeta)
    z2 = 0.75 * zeta + 0.25 * (z1 + dt * rhs(z1))
    return (1.0 / 3.0) * zeta + (2.0 / 3.0) * (z2 + dt * rhs(z2))


def total_energy_enstrophy(zeta, N):
    """Domain-integrated KE and enstrophy of a 2D periodic vorticity field.
    KE = ½∫|u|², enstrophy = ½∫ζ²; both per unit area × area."""
    kx, ky, k2, k2_inv = (jnp.asarray(a) for a in _wavenumbers(N))
    u, v = velocity_from_vorticity(zeta, kx, ky, k2_inv)
    cell = (2.0 * np.pi / N) ** 2
    ke = float(0.5 * jnp.sum(u ** 2 + v ** 2) * cell)
    ens = float(0.5 * jnp.sum(zeta ** 2) * cell)
    return ke, ens
