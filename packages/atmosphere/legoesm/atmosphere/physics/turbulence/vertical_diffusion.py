"""Implicit vertical diffusion using the Thomas algorithm.

Solves the 1D diffusion equation in flux form:

    dφ/dt = (1/ρ) d/dz [ρ K dφ/dz]

using backward Euler time stepping, producing a tridiagonal system
solved via the SHARED batched Thomas solver
:func:`legoesm.timestepping.tridiagonal.thomas_solve_batched` (the same
solver used by CLUBB and the convective mass-flux scheme — no hand-rolled
forward/back sweeps here).

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
from legoesm.atmosphere.physics._shared import exner_function
from legoesm.timestepping.tridiagonal import thomas_solve_batched

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
    # Coerce every coefficient input to the diffused field's working ``dtype``. They may
    # arrive WIDER than ``phi`` — e.g. a per-column ``C_K`` diagnosed in float64 by the
    # LES-informed correction loop (→ a float64 ``K_half``), or float64 grid metrics —
    # feeding a float32 finite-volume run. Without this the tridiagonal-coefficient
    # scatters implicitly downcast float64→float32, a JAX FutureWarning that will become
    # an error. Behavior-preserving: the coefficients were already built in ``dtype``, so
    # the implicit downcast happened anyway — this only makes it explicit + consistent.
    K_half = jnp.asarray(K_half, dtype=dtype)
    rho = jnp.asarray(rho, dtype=dtype)
    dz = jnp.asarray(dz, dtype=dtype)
    dz_half = jnp.asarray(dz_half, dtype=dtype)
    surface_flux = jnp.asarray(surface_flux, dtype=dtype)

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

    # --- Solve the tridiagonal system via the SHARED batched Thomas solver ---
    # Our system is  -a[k]·x[k-1] + b[k]·x[k] - c[k]·x[k+1] = rhs[k]; the
    # shared solver expects  sub[k]·x[k-1] + diag[k]·x[k] + sup[k]·x[k+1] =
    # d[k]  with the system on the LAST axis (columns batched over leading
    # axes) — so pass sub = -a, diag = b, sup = -c.  ``a[:, 0]`` and
    # ``c[:, -1]`` are already zero (no-flux top, surface-flux bottom BC).
    # Same solver as CLUBB (thomas_solve) and convective mass_flux
    # (thomas_solve_batched); deletes the previously duplicated ~60 LOC of
    # forward-elimination / back-substitution lax.scan sweeps.
    return thomas_solve_batched(-a, b, -c, rhs)


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
    # Exner Pi = (p/p_ref)^kappa via the canonical helper (same 1 Pa floor).
    exner = exner_function(p_full)
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


def diagnostic_heat_flux_full(
    theta: jax.Array,
    dz_half: jax.Array,
    Kh_half: jax.Array,
    gamma_theta_half: jax.Array | None = None,
) -> jax.Array:
    """Diagnostic kinematic heat flux ``⟨w'θ'⟩ = −Kh·(∂θ/∂z − γ)`` on FULL levels.

    The ONE shared reduction for the LES-suite Q1 diagnostic score
    (``TurbulenceOutput.wtheta_flux``) — every K-closure exposes its flux through
    this helper so local and nonlocal schemes are scored apples-to-apples.

    Sign convention (matches the atm column: index 0 = model top, index −1 =
    surface; ``dz_half > 0``): ``∂θ/∂z`` is positive when θ increases upward, so a
    stable layer (``∂θ/∂z > 0``) with ``γ = 0`` gives a downward (negative) flux —
    correct down-gradient sign. A local closure passes ``gamma_theta_half=None``
    (γ ≡ 0); a nonlocal closure passes its counter-gradient ``γ`` [K/m] so a CBL
    mixed layer carries an UPWARD flux against a weakly stable gradient.

    The interface flux (``nlev−1`` values) is averaged to full levels exactly the
    way ``Kh_full`` is derived from ``Kh_half`` in every scheme (interior mean,
    edge interfaces copied) so it co-locates with the full-level θ the diagnostic
    score compares against. PURE diagnostic — the tendencies come from the implicit
    flux-divergence solve, never from this value, so it can change no run.
    """
    dtheta_dz_half = (theta[:, :-1] - theta[:, 1:]) / dz_half
    if gamma_theta_half is not None:
        dtheta_dz_half = dtheta_dz_half - gamma_theta_half
    flux_half = -Kh_half * dtheta_dz_half  # (ncol, nlev-1) interface flux [K m/s]
    flux_interior = 0.5 * (flux_half[:, :-1] + flux_half[:, 1:])
    return jnp.concatenate(
        [flux_half[:, :1], flux_interior, flux_half[:, -1:]], axis=1,
    )
