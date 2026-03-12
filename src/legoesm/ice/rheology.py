"""VP/EVP sea ice rheology: constitutive law and ice strength.

Implements the Elastic-Viscous-Plastic (EVP) rheology of Hunke & Dukowicz
(1997) for sea ice dynamics. The yield curve is an ellipse in principal
stress space with eccentricity *e* (default 2).

Key functions:
- ``ice_strength``: Hibler (1979) P = P* h exp(-C(1-A))
- ``strain_rates``: Symmetric strain rate tensor from velocity gradients
- ``delta_deformation``: Deformation rate invariant
- ``vp_stress``: Viscous-plastic stress from VP constitutive law
- ``evp_stress_update``: Single EVP subcycle stress update

All functions are JAX-compatible (differentiable, JIT-friendly).

References
----------
- Hibler, W. D. III (1979): A dynamic thermodynamic sea ice model.
  J. Phys. Oceanogr., 9, 815-846.
- Hunke, E. C. & Dukowicz, J. K. (1997): An elastic-viscous-plastic model
  for sea ice dynamics. J. Phys. Oceanogr., 27, 1849-1867.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo, pad_halo_vector


# ==============================================================================
# Ice strength
# ==============================================================================

def ice_strength(
    h: jnp.ndarray,
    A: jnp.ndarray,
    P_star: float = 2.75e4,
    C_strength: float = 20.0,
) -> jnp.ndarray:
    """Compute ice strength following Hibler (1979).

    P = P* * h * exp(-C * (1 - A))

    Parameters
    ----------
    h : array
        Mean ice thickness [m].
    A : array
        Ice concentration [0-1].
    P_star : float
        Ice strength parameter [N/m^2].
    C_strength : float
        Exponential decay constant.

    Returns
    -------
    P : array
        Ice strength [N/m].
    """
    return P_star * h * jnp.exp(-C_strength * (1.0 - A))


# ==============================================================================
# Strain rates
# ==============================================================================

def strain_rates(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    grid: CubedSphereGrid,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the symmetric strain rate tensor on the cubed sphere.

    eps_11 = du/dx
    eps_22 = dv/dy
    eps_12 = 0.5 * (du/dy + dv/dx)

    Uses centered finite differences with proper vector halo exchange
    at face boundaries.

    Parameters
    ----------
    u_ice, v_ice : array (6, n, n)
        Ice velocity components [m/s].
    grid : CubedSphereGrid

    Returns
    -------
    eps_11, eps_22, eps_12 : arrays (6, n, n)
        Strain rate tensor components [1/s].
    """
    # Vector halo exchange for correct cross-face rotation
    u_pad, v_pad = pad_halo_vector(
        u_ice, v_ice,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )

    # Centered differences
    du_dx = (u_pad[:, 2:, 1:-1] - u_pad[:, :-2, 1:-1]) / grid.dx
    du_dy = (u_pad[:, 1:-1, 2:] - u_pad[:, 1:-1, :-2]) / grid.dy
    dv_dx = (v_pad[:, 2:, 1:-1] - v_pad[:, :-2, 1:-1]) / grid.dx
    dv_dy = (v_pad[:, 1:-1, 2:] - v_pad[:, 1:-1, :-2]) / grid.dy

    eps_11 = du_dx
    eps_22 = dv_dy
    eps_12 = 0.5 * (du_dy + dv_dx)

    return eps_11, eps_22, eps_12


# ==============================================================================
# Deformation invariant
# ==============================================================================

def delta_deformation(
    eps_11: jnp.ndarray,
    eps_22: jnp.ndarray,
    eps_12: jnp.ndarray,
    e_yield: float = 2.0,
    Delta_min: float = 2.0e-9,
) -> jnp.ndarray:
    """Compute the deformation rate invariant Delta.

    Delta = sqrt((eps_11 + eps_22)^2 + (1/e^2)*((eps_11-eps_22)^2 + 4*eps_12^2))

    Regularized with Delta_min for numerical stability.

    Parameters
    ----------
    eps_11, eps_22, eps_12 : arrays
        Strain rate tensor components.
    e_yield : float
        Yield curve eccentricity.
    Delta_min : float
        Minimum deformation rate [1/s].

    Returns
    -------
    Delta : array
        Deformation rate invariant [1/s].
    """
    divergence = eps_11 + eps_22
    shear = (eps_11 - eps_22) ** 2 + 4.0 * eps_12 ** 2
    Delta_sq = divergence ** 2 + shear / (e_yield ** 2)
    return jnp.maximum(jnp.sqrt(jnp.maximum(Delta_sq, 0.0)), Delta_min)


# ==============================================================================
# VP stress tensor
# ==============================================================================

def vp_stress(
    eps_11: jnp.ndarray,
    eps_22: jnp.ndarray,
    eps_12: jnp.ndarray,
    P: jnp.ndarray,
    Delta: jnp.ndarray,
    e_yield: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the VP stress tensor.

    zeta = P / (2 * Delta)        (bulk viscosity)
    eta  = zeta / e^2             (shear viscosity)

    sigma_11 = 2*eta*eps_11 + (zeta-eta)*(eps_11+eps_22) - P/2
    sigma_22 = 2*eta*eps_22 + (zeta-eta)*(eps_11+eps_22) - P/2
    sigma_12 = 2*eta*eps_12

    Parameters
    ----------
    eps_11, eps_22, eps_12 : arrays
        Strain rate tensor components.
    P : array
        Ice strength [N/m].
    Delta : array
        Deformation rate invariant [1/s].
    e_yield : float
        Yield curve eccentricity.

    Returns
    -------
    sigma_11, sigma_22, sigma_12 : arrays
        Stress tensor components [N/m].
    """
    zeta = P / (2.0 * Delta)
    eta = zeta / (e_yield ** 2)

    trace = eps_11 + eps_22
    sigma_11 = 2.0 * eta * eps_11 + (zeta - eta) * trace - P / 2.0
    sigma_22 = 2.0 * eta * eps_22 + (zeta - eta) * trace - P / 2.0
    sigma_12 = 2.0 * eta * eps_12

    return sigma_11, sigma_22, sigma_12


# ==============================================================================
# EVP subcycle stress update
# ==============================================================================

def evp_stress_update(
    sigma_11: jnp.ndarray,
    sigma_22: jnp.ndarray,
    sigma_12: jnp.ndarray,
    eps_11: jnp.ndarray,
    eps_22: jnp.ndarray,
    eps_12: jnp.ndarray,
    P: jnp.ndarray,
    e_yield: float,
    T_evp: float,
    dt_s: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single EVP subcycle stress update (Hunke & Dukowicz 1997).

    Each stress component is relaxed toward the VP solution:
        sigma_new = (1/(1 + E_factor)) * (sigma_old + E_factor * sigma_VP)

    where E_factor = dt_s / (2 * T_damp) and T_damp is the EVP
    damping timescale.

    Parameters
    ----------
    sigma_11, sigma_22, sigma_12 : arrays
        Current stress tensor [N/m].
    eps_11, eps_22, eps_12 : arrays
        Current strain rate tensor [1/s].
    P : array
        Ice strength [N/m].
    e_yield : float
        Yield curve eccentricity.
    T_evp : float
        EVP damping timescale ratio (T_damp = T_evp * N_evp * dt_s).
    dt_s : float
        EVP subcycle timestep [s].

    Returns
    -------
    sigma_11_new, sigma_22_new, sigma_12_new : arrays
        Updated stress tensor [N/m].
    """
    Delta = delta_deformation(eps_11, eps_22, eps_12, e_yield)

    # VP target stress
    s11_vp, s22_vp, s12_vp = vp_stress(
        eps_11, eps_22, eps_12, P, Delta, e_yield,
    )

    # EVP relaxation factor
    # T_damp = T_evp * dt_subcycle_total; but we express per substep:
    # E_factor = dt_s / (2 * T_damp) = 1 / (2 * T_evp * N_evp)
    # Simplified: use T_evp as the ratio dt_s / T_damp directly
    E_factor = 1.0 / (2.0 * T_evp)
    denom = 1.0 + E_factor

    sigma_11_new = (sigma_11 + E_factor * s11_vp) / denom
    sigma_22_new = (sigma_22 + E_factor * s22_vp) / denom
    sigma_12_new = (sigma_12 + E_factor * s12_vp) / denom

    return sigma_11_new, sigma_22_new, sigma_12_new
