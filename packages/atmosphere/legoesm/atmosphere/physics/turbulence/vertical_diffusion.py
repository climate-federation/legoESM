"""Implicit vertical diffusion using the Thomas algorithm.

Solves the 1D diffusion equation in flux form:

    dφ/dt = (1/ρ) d/dz [ρ K dφ/dz]

using backward Euler time stepping, producing a tridiagonal system
solved via forward-sweep / back-substitution with jax.lax.scan.

The flux F = ρ K dφ/dz is evaluated on half-levels (interfaces) using
arithmetic-mean interface density ρ_half = 0.5 (ρ[k] + ρ[k+1]).  Without
the interface density the scheme reduces to a kinematic diffusivity
``K/(ρ dz dz_half)`` discretization that is inconsistent with the
surface-flux boundary condition ``F_sfc = ρ K dφ/dz`` and silently mixes
mass-weighted and kinematic fluxes (audit 2026-05-12 finding HIGH #1).

The bottom boundary condition applies a prescribed surface flux.
The top boundary condition is zero flux (no diffusion through the top).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants

__physics_contract__ = {
    "summary": (
        "Implicit (backward-Euler) flux-form vertical diffusion of a column "
        "scalar via the Thomas tridiagonal algorithm; no-flux top boundary "
        "and a prescribed surface-flux bottom boundary. A theta variant "
        "diffuses potential temperature so a dry adiabat stays neutral."
    ),
    "inputs": {
        "phi": "same as diffused field (m/s, K, or kg/kg)",
        "K_half": "m^2/s", "rho": "kg/m^3", "dz": "m", "dz_half": "m",
        "dt": "s",
        "surface_flux": "phi_unit*kg/m^2/s (= rho K dphi/dz, positive upward)",
    },
    "outputs": {"phi_new": "same as phi (diffused field)"},
    "sign_convention": (
        "Down-gradient: flux F = rho K dphi/dz on interfaces, tendency "
        "dphi/dt = (1/rho) dF/dz. The top interface is no-flux; the bottom "
        "receives the PRESCRIBED surface_flux (positive upward = a source "
        "when nonzero). z increases upward; level index 0 is the top, -1 the "
        "surface. With surface_flux=0 the mass-weighted column integral "
        "Sum(rho dz phi) is conserved to machine precision (flux form); the "
        "theta variant conserves the mass-weighted column theta but NOT "
        "column enthalpy (Exner varies with height)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Thomas (1949) tridiagonal algorithm; backward-Euler flux-form "
        "vertical diffusion (Richtmyer & Morton 1967)"
    ),
    "idealized_test": (
        "surface_flux=0 -> Sum(rho dz phi) conserved to machine precision "
        "(no-flux top+bottom, flux form); constant-K diffusion relaxes an "
        "arbitrary profile toward the mass-weighted column mean."
    ),
}

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)


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
    dtype = phi.dtype
    dt = jnp.asarray(dt, dtype=dtype)

    # Interface density for the flux ρ_half K_half dφ/dz on half-levels.
    rho_half = 0.5 * (rho[:, :-1] + rho[:, 1:])

    # Build tridiagonal coefficients (all positive).
    # The system is: -a[k]*phi[k-1] + b[k]*phi[k] - c[k]*phi[k+1] = rhs[k]
    # with F_{k-1/2} = rho_half[k-1/2] * K_half[k-1/2] * (phi[k]-phi[k-1]) / dz_half[k-1/2]
    # and -dphi/dt[k] = (F_{k+1/2} - F_{k-1/2}) / (rho[k] dz[k]).

    # a[k] = dt * rho_half[k-1] * K_half[k-1] / (rho[k] * dz[k] * dz_half[k-1])
    a = jnp.zeros((ncol, nlev), dtype=dtype)
    a = a.at[:, 1:].set(
        dt * rho_half * K_half / (rho[:, 1:] * dz[:, 1:] * dz_half)
    )

    # c[k] = dt * rho_half[k] * K_half[k] / (rho[k] * dz[k] * dz_half[k])
    c = jnp.zeros((ncol, nlev), dtype=dtype)
    c = c.at[:, :-1].set(
        dt * rho_half * K_half / (rho[:, :-1] * dz[:, :-1] * dz_half)
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

        w = a_k / jnp.clip(b_prev_mod, _TINY, None)
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
    phi_bottom = rhs_mod[-1] / jnp.clip(b_mod[-1], _TINY, None)

    # phi[k] = (d'[k] + c[k] * phi[k+1]) / b'[k]
    def back_step(phi_below, inputs):
        b_k, c_k, rhs_k = inputs
        phi_k = (rhs_k + c_k * phi_below) / jnp.clip(b_k, _TINY, None)
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


def implicit_vertical_diffusion_theta(
    T: jax.Array,
    K_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dz_half: jax.Array,
    p_full: jax.Array,
    dt: float,
    surface_flux_T: jax.Array,
) -> jax.Array:
    """Implicit vertical diffusion of temperature via potential temperature.

    Diffusing absolute T is unphysical: even a dry adiabat (dθ/dz = 0)
    has dT/dz ≈ -g/c_p ≈ -9.8 K/km, so applying ``K dT/dz`` mixes a
    neutrally stratified column into an unphysical isothermal state.
    The correct conserved variable for dry mixing is potential temperature
    θ = T (p_ref/p)^κ — its gradient vanishes on a dry adiabat.

    Conservation caveat (not an energy-conserving scheme).  Diffusing θ in
    flux form conserves the mass-weighted column θ, ``Σ ρ dz θ``, but NOT
    the column enthalpy ``Σ ρ dz c_p T``, because the Exner function
    π = (p/p_ref)^κ relating T = π θ varies with height, so redistributing θ
    redistributes enthalpy unequally between layers.  This is an accepted
    approximation here.  A strictly energy-conserving variant would instead
    diffuse the dry static energy s = c_p T + g z (whose flux-form mixing
    conserves ``Σ ρ dz s`` exactly); switching the diffused variable to DSE
    is a validated follow-up, not done in this helper.

    This helper:

    1. Converts T → θ using the column pressure.
    2. Converts the surface T flux to a θ flux at the lowest interface,
       ``F_θ_sfc = F_T_sfc / exner_sfc`` where ``exner_sfc =
       (p_low/p_ref)^κ`` is the Exner function at the lowest full level
       (a column-bottom proxy when ``p_sfc`` is not threaded through).
    3. Diffuses θ via :func:`implicit_vertical_diffusion`.
    4. Converts the diffused θ back to T using the same Exner factor.

    Parameters
    ----------
    T : jax.Array, shape (ncol, nlev)
        Temperature [K].
    K_half : jax.Array, shape (ncol, nlev-1)
        Eddy heat diffusivity on interfaces [m²/s].
    rho : jax.Array, shape (ncol, nlev)
        Full-level air density [kg/m³].
    dz : jax.Array, shape (ncol, nlev)
        Full-level layer thickness [m].
    dz_half : jax.Array, shape (ncol, nlev-1)
        Distance between adjacent full-level centers [m].
    p_full : jax.Array, shape (ncol, nlev)
        Full-level pressure [Pa].
    dt : float
        Time step [s].
    surface_flux_T : jax.Array, shape (ncol,)
        Surface sensible-heat flux divided by ``c_pd`` [K kg/m²/s],
        positive upward.  ``shflx [W/m²] / c_pd`` is what the surface
        layer already returns.

    Returns
    -------
    jax.Array, shape (ncol, nlev)
        Diffused temperature [K].
    """
    p_safe = jnp.clip(p_full, 1.0, None)
    exner = (p_safe / constants.p_ref) ** constants.kappa
    exner_safe = jnp.clip(exner, 1.0e-6, None)
    theta = T / exner_safe
    # Surface-Exner proxy: use the lowest FULL-level Exner (exner[:, -1]) as
    # a stand-in for the surface Exner.  Because p_low < p_surface, this proxy
    # is biased low, so F_θ_sfc = F_T_sfc / exner_sfc is biased slightly HIGH
    # (the injected surface θ-flux is a touch too large).  Threading a true
    # p_surface through and using (p_sfc/p_ref)^κ would remove this bias.
    exner_sfc = exner_safe[:, -1]
    # F_T_sfc has units [K · kg/m²/s] = ρ K dT/dz.  Diffusing θ requires
    # the surface θ flux F_θ_sfc = F_T_sfc / exner_sfc.
    surface_flux_theta = surface_flux_T / exner_sfc
    theta_new = implicit_vertical_diffusion(
        theta, K_half, rho, dz, dz_half, dt, surface_flux_theta,
    )
    return theta_new * exner
