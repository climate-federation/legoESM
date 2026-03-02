"""Implicit vertical diffusion using the Thomas algorithm.

Solves the 1D diffusion equation:

    dφ/dt = (1/ρ) d/dz [K dφ/dz]

using backward Euler time stepping, producing a tridiagonal system
solved via forward-sweep / back-substitution with jax.lax.scan.

The bottom boundary condition applies a prescribed surface flux.
The top boundary condition is zero flux (no diffusion through the top).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def implicit_vertical_diffusion(
    phi: jax.Array,
    K_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dz_half: jax.Array,
    dt: float,
    surface_flux: jax.Array,
) -> jax.Array:
    """Apply implicit vertical diffusion to a column field.

    Parameters
    ----------
    phi : jax.Array
        Field to diffuse, shape (ncol, nlev). Level index 0 is top,
        index nlev-1 is bottom (nearest surface).
    K_half : jax.Array
        Eddy diffusivity at half-levels (interfaces between full levels),
        shape (ncol, nlev-1). K_half[k] is between full levels k and k+1.
    rho : jax.Array
        Air density at full levels, shape (ncol, nlev).
    dz : jax.Array
        Layer thickness at full levels, shape (ncol, nlev).
    dz_half : jax.Array
        Distance between full-level centers (half-level spacing),
        shape (ncol, nlev-1). dz_half[k] = distance from center of
        level k to center of level k+1.
    dt : float
        Time step [s].
    surface_flux : jax.Array
        Bottom boundary flux, shape (ncol,). Positive upward.
        Units: [phi_units * kg/m^2/s] (i.e., ρ * K * dφ/dz).

    Returns
    -------
    jax.Array
        Diffused field, shape (ncol, nlev).
    """
    ncol, nlev = phi.shape

    # Build tridiagonal coefficients (all positive).
    # The system is: -a[k]*phi[k-1] + b[k]*phi[k] - c[k]*phi[k+1] = rhs[k]

    # a[k] = dt * K_half[k-1] / (rho[k] * dz[k] * dz_half[k-1])  for k=1..nlev-1
    a = jnp.zeros((ncol, nlev))
    a = a.at[:, 1:].set(
        dt * K_half / (rho[:, 1:] * dz[:, 1:] * dz_half)
    )

    # c[k] = dt * K_half[k] / (rho[k] * dz[k] * dz_half[k])  for k=0..nlev-2
    c = jnp.zeros((ncol, nlev))
    c = c.at[:, :-1].set(
        dt * K_half / (rho[:, :-1] * dz[:, :-1] * dz_half)
    )

    # Diagonal: b[k] = 1 + a[k] + c[k]
    b = 1.0 + a + c

    # Right-hand side
    rhs = phi.copy()
    # Surface flux at bottom level (index nlev-1)
    rhs = rhs.at[:, -1].add(dt * surface_flux / (rho[:, -1] * dz[:, -1]))

    # --- Thomas algorithm via jax.lax.scan ---
    # Transpose to (nlev, ncol) for scan
    a_T = jnp.moveaxis(a, 1, 0)  # (nlev, ncol)
    b_T = jnp.moveaxis(b, 1, 0)
    c_T = jnp.moveaxis(c, 1, 0)
    rhs_T = jnp.moveaxis(rhs, 1, 0)

    # Forward sweep: eliminate sub-diagonal.
    # For the system -a[k]*x[k-1] + b[k]*x[k] - c[k]*x[k+1] = d[k],
    # Gaussian elimination gives:
    #   w = a[k] / b'[k-1]
    #   b'[k] = b[k] - w * c[k-1]
    #   d'[k] = d[k] + w * d'[k-1]

    init = (b_T[0], rhs_T[0])
    c_prev = c_T[:-1]  # c[0]..c[nlev-2] for levels 1..nlev-1

    def forward_step(carry, inputs):
        b_prev_mod, rhs_prev_mod = carry
        a_k, b_k, c_prev_k, rhs_k = inputs

        w = a_k / jnp.clip(b_prev_mod, 1e-30, None)
        b_k_mod = b_k - w * c_prev_k
        rhs_k_mod = rhs_k + w * rhs_prev_mod
        return (b_k_mod, rhs_k_mod), (b_k_mod, rhs_k_mod)

    _, (b_mod_rest, rhs_mod_rest) = jax.lax.scan(
        forward_step,
        init,
        (a_T[1:], b_T[1:], c_prev, rhs_T[1:]),
    )

    # Full modified arrays: prepend level 0
    b_mod = jnp.concatenate([b_T[0:1], b_mod_rest], axis=0)  # (nlev, ncol)
    rhs_mod = jnp.concatenate([rhs_T[0:1], rhs_mod_rest], axis=0)

    # --- Back substitution: scan from bottom to top ---
    # phi[nlev-1] = d'[nlev-1] / b'[nlev-1]
    phi_bottom = rhs_mod[-1] / jnp.clip(b_mod[-1], 1e-30, None)

    # phi[k] = (d'[k] + c[k] * phi[k+1]) / b'[k]
    def back_step(phi_below, inputs):
        b_k, c_k, rhs_k = inputs
        phi_k = (rhs_k + c_k * phi_below) / jnp.clip(b_k, 1e-30, None)
        return phi_k, phi_k

    _, phi_upper_rev = jax.lax.scan(
        back_step,
        phi_bottom,
        (b_mod[-2::-1], c_T[-2::-1], rhs_mod[-2::-1]),
    )

    # Assemble result
    phi_upper = phi_upper_rev[::-1]  # (nlev-1, ncol), top to bottom
    phi_new = jnp.concatenate([phi_upper, phi_bottom[None]], axis=0)  # (nlev, ncol)

    return jnp.moveaxis(phi_new, 0, 1)  # (ncol, nlev)
