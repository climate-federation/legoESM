"""Sea ice dynamics: EVP momentum solver and stress divergence.

Solves the ice momentum equation using Elastic-Viscous-Plastic (EVP)
subcycling (Hunke & Dukowicz 1997):

    m du/dt = tau_air + tau_ocean - mfk x u + div(sigma)

where m = rho_ice * h is ice mass per unit area, tau_air and tau_ocean
are wind and ocean drag stresses, f is the Coriolis parameter,
and sigma is the internal stress tensor from the VP/EVP rheology.

The EVP solver subcycles N_evp times per dynamical timestep. Each
subcycle updates stress via elastic relaxation toward the VP solution,
then advances velocity semi-implicitly.

Uses ``jax.lax.fori_loop`` (production) or ``jax.lax.scan``
(differentiable) following the barotropic substep pattern.

All functions are JAX-compatible (differentiable, JIT-friendly).

References
----------
- Hunke, E. C. & Dukowicz, J. K. (1997): An elastic-viscous-plastic model
  for sea ice dynamics. J. Phys. Oceanogr., 27, 1849-1867.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators import gradient_x, gradient_y
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo, pad_halo_vector
from legoesm.ice.rheology import (
    ice_strength,
    strain_rates,
    evp_stress_update,
)


# ==============================================================================
# Raw-array operator wrappers (avoid Field allocation in fori_loop)
# ==============================================================================

def _gradient_x_raw(data: jnp.ndarray, grid: CubedSphereGrid) -> jnp.ndarray:
    f = Field(data=data, name="f", dims=("face", "x", "y"), units="")
    return gradient_x(f, grid).data


def _gradient_y_raw(data: jnp.ndarray, grid: CubedSphereGrid) -> jnp.ndarray:
    f = Field(data=data, name="f", dims=("face", "x", "y"), units="")
    return gradient_y(f, grid).data


# ==============================================================================
# Stress divergence
# ==============================================================================

def stress_divergence(
    sigma_11: jnp.ndarray,
    sigma_22: jnp.ndarray,
    sigma_12: jnp.ndarray,
    grid: CubedSphereGrid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Compute the divergence of the internal stress tensor.

    F_x = d(sigma_11)/dx + d(sigma_12)/dy
    F_y = d(sigma_12)/dx + d(sigma_22)/dy

    The stress components are treated as scalars in local grid-aligned
    coordinates. At cubed-sphere face boundaries, this introduces an
    O(dx) error in the tensor rotation, which is acceptable at coarse
    resolution and is damped by the EVP's elastic relaxation.

    For higher accuracy, a full tensor halo exchange would rotate
    (sigma_11, sigma_22, sigma_12) between faces. This simpler
    approach avoids that complexity.

    Parameters
    ----------
    sigma_11, sigma_22, sigma_12 : arrays (6, n, n)
        Stress tensor components [N/m].
    grid : CubedSphereGrid

    Returns
    -------
    Fx, Fy : arrays (6, n, n)
        Internal stress force per unit area [N/m^2].
    """
    # d(sigma_11)/dx + d(sigma_12)/dy
    Fx = _gradient_x_raw(sigma_11, grid) + _gradient_y_raw(sigma_12, grid)
    # d(sigma_12)/dx + d(sigma_22)/dy
    Fy = _gradient_x_raw(sigma_12, grid) + _gradient_y_raw(sigma_22, grid)

    return Fx, Fy


# ==============================================================================
# Free-drift velocity
# ==============================================================================

def free_drift_velocity(
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    wind_u: jnp.ndarray,
    wind_v: jnp.ndarray,
    drag_ocean: float = 5.5e-3,
    drag_atm: float = 1.3e-3,
    rho_air: float = 1.225,
    rho_ice: float = 917.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute diagnostic ice velocity via linear drag combination.

    **This is a heuristic placeholder**, not a physically faithful
    free-drift solver.  A proper free-drift model would solve the
    steady-state momentum balance including Coriolis and turning
    angles.  The current formulation is a simple linear combination:

        u_ice = drag_ocean * u_ocean + drag_atm * (rho_air/rho_ice) * u_wind

    Parameters
    ----------
    ocean_u, ocean_v : arrays
        Ocean surface currents [m/s].
    wind_u, wind_v : arrays
        Atmospheric wind [m/s].
    drag_ocean, drag_atm : float
        Drag coefficients.
    rho_air, rho_ice : float
        Densities [kg/m^3].

    Returns
    -------
    u_ice, v_ice : arrays
        Diagnostic ice velocity [m/s].
    """
    ratio = drag_atm * rho_air / rho_ice
    u_ice = drag_ocean * ocean_u + ratio * wind_u
    v_ice = drag_ocean * ocean_v + ratio * wind_v
    return u_ice, v_ice


# ==============================================================================
# Air and ocean stress on ice
# ==============================================================================

def air_ice_stress(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    wind_u: jnp.ndarray,
    wind_v: jnp.ndarray,
    rho_air: float = 1.225,
    C_ai: float = 1.3e-3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute air-ice drag stress.

    tau_ai = rho_air * C_ai * |u_wind - u_ice| * (u_wind - u_ice)

    Parameters
    ----------
    u_ice, v_ice : arrays
        Ice velocity [m/s].
    wind_u, wind_v : arrays
        Atmospheric wind [m/s].
    rho_air : float
    C_ai : float
        Air-ice drag coefficient.

    Returns
    -------
    tau_x, tau_y : arrays
        Air stress on ice [N/m^2].
    """
    du = wind_u - u_ice
    dv = wind_v - v_ice
    speed = jnp.sqrt(du ** 2 + dv ** 2 + 1e-10)
    tau_x = rho_air * C_ai * speed * du
    tau_y = rho_air * C_ai * speed * dv
    return tau_x, tau_y


def ocean_ice_stress(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    rho_ocean: float = 1025.0,
    C_oi: float = 5.5e-3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute ocean-ice drag stress.

    tau_oi = rho_ocean * C_oi * |u_ocean - u_ice| * (u_ocean - u_ice)

    Parameters
    ----------
    u_ice, v_ice : arrays
        Ice velocity [m/s].
    ocean_u, ocean_v : arrays
        Ocean surface currents [m/s].
    rho_ocean : float
    C_oi : float
        Ocean-ice drag coefficient.

    Returns
    -------
    tau_x, tau_y : arrays
        Ocean stress on ice [N/m^2].
    """
    du = ocean_u - u_ice
    dv = ocean_v - v_ice
    speed = jnp.sqrt(du ** 2 + dv ** 2 + 1e-10)
    tau_x = rho_ocean * C_oi * speed * du
    tau_y = rho_ocean * C_oi * speed * dv
    return tau_x, tau_y


# ==============================================================================
# EVP solver
# ==============================================================================

def evp_solver(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    sigma_11: jnp.ndarray,
    sigma_22: jnp.ndarray,
    sigma_12: jnp.ndarray,
    h_ice: jnp.ndarray,
    concentration: jnp.ndarray,
    wind_u: jnp.ndarray,
    wind_v: jnp.ndarray,
    ocean_u: jnp.ndarray,
    ocean_v: jnp.ndarray,
    grid: CubedSphereGrid,
    dt: float,
    N_evp: int = 120,
    e_yield: float = 2.0,
    P_star: float = 2.75e4,
    C_strength: float = 20.0,
    T_evp: float = 0.36,
    rho_ice: float = 917.0,
    rho_air: float = 1.225,
    rho_ocean: float = 1025.0,
    C_ai: float = 1.3e-3,
    C_oi: float = 5.5e-3,
    differentiable: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the EVP subcycled momentum solver.

    Subcycles N_evp times per dynamical timestep dt. Each subcycle:
    1. Compute strain rates from current velocity
    2. Update stress via EVP relaxation
    3. Compute stress divergence
    4. Advance velocity semi-implicitly with Coriolis

    Parameters
    ----------
    u_ice, v_ice : arrays (6, n, n)
        Initial ice velocity [m/s].
    sigma_11, sigma_22, sigma_12 : arrays (6, n, n)
        Initial stress tensor [N/m].
    h_ice : array (6, n, n)
        Ice thickness [m].
    concentration : array (6, n, n)
        Ice concentration [0-1].
    wind_u, wind_v : arrays (6, n, n)
        Atmospheric wind [m/s].
    ocean_u, ocean_v : arrays (6, n, n)
        Ocean surface currents [m/s].
    grid : CubedSphereGrid
    dt : float
        Full dynamical timestep [s].
    N_evp : int
        Number of EVP subcycles.
    e_yield : float
        Yield curve eccentricity.
    P_star, C_strength : float
        Ice strength parameters.
    T_evp : float
        EVP damping timescale ratio.
    rho_ice, rho_air, rho_ocean : float
        Densities [kg/m^3].
    C_ai, C_oi : float
        Drag coefficients.
    differentiable : bool
        Use scan (True) or fori_loop (False).

    Returns
    -------
    u_new, v_new : arrays (6, n, n)
        Updated ice velocity [m/s].
    sigma_11_new, sigma_22_new, sigma_12_new : arrays (6, n, n)
        Updated stress tensor [N/m].
    """
    dt_s = dt / N_evp  # subcycle timestep

    # Ice mass per unit ice-covered area: rho_ice · h.  In the CICE
    # equation of motion, both the wind/ocean stress AND the mass scale
    # by concentration A: m_grid · du/dt = A · tau_a + A · tau_o + ...,
    # which simplifies to rho_ice · h · du/dt = tau_a + tau_o + ...
    # because air_ice_stress / ocean_ice_stress already return stress
    # per unit ice-covered area (no A factor).  Including concentration
    # in m_ice without also weighting the stresses introduces a 1/A
    # over-acceleration in the marginal-ice zone.  An earlier audit
    # iteration claimed the original ``rho_ice · max(h, 0.01)`` was
    # missing concentration weighting; Codex GPT-5 review caught the
    # bookkeeping mistake, and the original form is correct.
    m_ice = rho_ice * jnp.maximum(h_ice, 0.01)

    # Ice strength (constant during subcycling)
    P = ice_strength(h_ice, concentration, P_star, C_strength)

    # Coriolis parameter at cell centers
    f = grid.f.astype(u_ice.dtype)  # (6, n, n)
    alpha = 0.5 * f * dt_s
    coriolis_denom = 1.0 + alpha ** 2

    # Ice mask: only compute dynamics where ice exists
    ice_mask = concentration > 0.01

    def substep_body(i, carry):
        u_c, v_c, s11_c, s22_c, s12_c = carry

        # 1. Strain rates from current velocity
        eps_11, eps_22, eps_12 = strain_rates(u_c, v_c, grid)

        # 2. EVP stress update
        s11_new, s22_new, s12_new = evp_stress_update(
            s11_c, s22_c, s12_c,
            eps_11, eps_22, eps_12,
            P, e_yield, T_evp, dt_s, N_evp,
        )

        # 3. Stress divergence
        Fx, Fy = stress_divergence(s11_new, s22_new, s12_new, grid)

        # 4. External forces (recomputed with current velocity)
        tau_air_x, tau_air_y = air_ice_stress(
            u_c, v_c, wind_u, wind_v, rho_air, C_ai,
        )
        tau_ocean_x, tau_ocean_y = ocean_ice_stress(
            u_c, v_c, ocean_u, ocean_v, rho_ocean, C_oi,
        )

        # 5. Total force per unit mass (excluding Coriolis)
        ax = (tau_air_x + tau_ocean_x + Fx) / m_ice
        ay = (tau_air_y + tau_ocean_y + Fy) / m_ice

        # 6. Semi-implicit velocity update with Coriolis
        rhs_u = u_c + dt_s * ax + alpha * v_c
        rhs_v = v_c + dt_s * ay - alpha * u_c
        u_new = (rhs_u + alpha * rhs_v) / coriolis_denom
        v_new = (rhs_v - alpha * rhs_u) / coriolis_denom

        # Zero velocity / stress where no ice.  Cast the boolean mask
        # to float once and multiply — fuses naturally with the
        # following stage and gives gradients a smooth zero (vs. the
        # branchless ``select`` that ``jnp.where`` lowers to).
        ice_mask_f = ice_mask.astype(u_new.dtype)
        u_new = u_new * ice_mask_f
        v_new = v_new * ice_mask_f
        s11_new = s11_new * ice_mask_f
        s22_new = s22_new * ice_mask_f
        s12_new = s12_new * ice_mask_f

        return (u_new, v_new, s11_new, s22_new, s12_new)

    init_carry = (u_ice, v_ice, sigma_11, sigma_22, sigma_12)

    if differentiable:
        def scan_body(carry, _):
            new_carry = substep_body(0, carry)
            return new_carry, None

        (u_f, v_f, s11_f, s22_f, s12_f), _ = jax.lax.scan(
            scan_body, init_carry, xs=None, length=N_evp,
        )
    else:
        u_f, v_f, s11_f, s22_f, s12_f = jax.lax.fori_loop(
            0, N_evp, substep_body, init_carry,
        )

    return u_f, v_f, s11_f, s22_f, s12_f
