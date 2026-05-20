"""VP/EVP sea ice rheology: constitutive law and ice strength.

Implements two ice-stress relaxation laws sharing the same VP target:

- **EVP** (Hunke & Dukowicz 1997): elastic-viscous-plastic relaxation
  parameterised by the damping ratio ``T_evp`` and the subcycle count
  ``N_evp``.
- **mEVP** (Bouillon et al. 2013; Kimmritz et al. 2015): modified-EVP
  pseudo-time relaxation parameterised by ``alpha_mevp`` (stress) and
  ``beta_mevp`` (velocity). At convergence the system reproduces the
  implicit VP solution; ``alpha`` and ``beta`` only control the
  pseudo-time relaxation rate and are not subject to the EVP elastic
  CFL constraint.

The yield curve is an ellipse in principal stress space with
eccentricity *e* (default 2).

Key functions:
- ``ice_strength``: Hibler (1979) P = P* h exp(-C(1-A))
- ``strain_rates``: Symmetric strain rate tensor from velocity gradients
- ``delta_deformation``: Deformation rate invariant
- ``vp_stress``: Viscous-plastic stress from VP constitutive law
- ``evp_stress_update``: Single EVP subcycle stress update
- ``mevp_stress_update``: Single mEVP pseudo-time stress update

All functions are JAX-compatible (differentiable, JIT-friendly).

References
----------
- Hibler, W. D. III (1979): A dynamic thermodynamic sea ice model.
  J. Phys. Oceanogr., 9, 815-846.
- Hunke, E. C. & Dukowicz, J. K. (1997): An elastic-viscous-plastic model
  for sea ice dynamics. J. Phys. Oceanogr., 27, 1849-1867.
- Bouillon, S., T. Fichefet, V. Legat, G. Madec (2013): The
  elastic-viscous-plastic method revisited. Ocean Modelling, 71, 2-12.
- Kimmritz, M., S. Danilov, M. Losch (2015): On the convergence of the
  modified elastic-viscous-plastic method for solving the sea ice
  momentum equation. J. Comput. Phys., 296, 90-100.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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
    N_evp: int,
    Delta_min: float = 2.0e-9,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single EVP subcycle stress update (Hunke & Dukowicz 1997).

    Backward-Euler relaxation of each stress component toward the VP
    target:

        sigma_new = (1/(1 + E_factor)) · (sigma_old + E_factor · sigma_VP)

    where ``E_factor = dt_s / (2 · T_damp)`` and the EVP damping
    timescale ``T_damp = T_evp · dt_dyn = T_evp · N_evp · dt_s``.  This
    gives ``E_factor = 1 / (2 · T_evp · N_evp)``.  An earlier
    formulation used ``E_factor = 1 / (2 · T_evp)``, which omits the
    ``N_evp`` factor and over-relaxes by O(N_evp×) per subcycle —
    defeating the elastic regularisation that keeps EVP stable.

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
        EVP damping ratio T_damp / dt_dyn (Hunke & Dukowicz E_y).
        Default 0.36 in CICE.
    dt_s : float
        EVP subcycle timestep [s].  Reserved for an explicit
        ``dt_s / (2·T_damp)`` form; kept in the signature so callers
        do not need to be rewritten when that path is added.
    N_evp : int
        Number of EVP subcycles per dynamic step.  Required to recover
        the correct relaxation timescale.
    Delta_min : float
        Deformation-rate regulariser [1/s] passed through to
        :func:`delta_deformation`.  Threaded from ``SeaIceConfig.Delta_min``
        so users can tune the EVP plastic-yield smoothness from the
        config rather than relying on the hard-coded default.

    Returns
    -------
    sigma_11_new, sigma_22_new, sigma_12_new : arrays
        Updated stress tensor [N/m].
    """
    del dt_s  # currently unused; see docstring
    Delta = delta_deformation(eps_11, eps_22, eps_12, e_yield, Delta_min)

    # VP target stress
    s11_vp, s22_vp, s12_vp = vp_stress(
        eps_11, eps_22, eps_12, P, Delta, e_yield,
    )

    # EVP relaxation factor: E = dt_s / (2 · T_damp) with T_damp =
    # T_evp · N_evp · dt_s  ⇒  E = 1 / (2 · T_evp · N_evp).
    E_factor = 1.0 / (2.0 * T_evp * float(N_evp))
    denom = 1.0 + E_factor

    sigma_11_new = (sigma_11 + E_factor * s11_vp) / denom
    sigma_22_new = (sigma_22 + E_factor * s22_vp) / denom
    sigma_12_new = (sigma_12 + E_factor * s12_vp) / denom

    return sigma_11_new, sigma_22_new, sigma_12_new


# ==============================================================================
# mEVP pseudo-time stress update
# ==============================================================================

def mevp_stress_update(
    sigma_11: jnp.ndarray,
    sigma_22: jnp.ndarray,
    sigma_12: jnp.ndarray,
    eps_11: jnp.ndarray,
    eps_22: jnp.ndarray,
    eps_12: jnp.ndarray,
    P: jnp.ndarray,
    e_yield: float,
    alpha: float,
    Delta_min: float = 2.0e-9,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single mEVP pseudo-time stress update (Bouillon 2013 / Kimmritz 2015).

    Pseudo-time relaxation toward the VP target with a single
    relaxation parameter ``alpha``:

        σ^(p+1) = (1 − 1/α) · σ^p + (1/α) · σ_VP(ε(u^p))

    Equivalent to the EVP elastic relaxation with ``E_factor = 1 /
    (alpha − 1)``, but with no implicit dependence on the subcycle
    count or a damping-ratio parameter — the pseudo-time iteration
    converges to the implicit VP solution as ``N_mevp → ∞`` regardless
    of the physical timestep.

    Stability (Kimmritz 2015): ``alpha · beta ≥ (e_yield²/4) · γ²``
    where ``γ = ζ · Δt / (ρ_ice · h · L²)`` is the dimensionless
    viscosity-stride product.  The bound is not enforced here — it
    depends on the realised ``ζ = P / (2 Δ)`` and grid spacing, so a
    static check would be either too conservative or unreliable.  The
    CICE / FESOM default ``alpha = beta = 500`` covers typical Arctic
    regimes with Δt ≤ 1 h on 50 km grids.  For larger Δt, finer grids,
    or thinner marginal ice raise ``alpha`` and ``beta`` and confirm
    convergence by running with doubled ``N_mevp``.

    Parameters
    ----------
    sigma_11, sigma_22, sigma_12 : arrays
        Current pseudo-time stress tensor [N/m].
    eps_11, eps_22, eps_12 : arrays
        Strain rate tensor at the current pseudo-time velocity ``u^p``
        [1/s].
    P : array
        Ice strength [N/m].
    e_yield : float
        Yield curve eccentricity.
    alpha : float
        mEVP stress-relaxation parameter (dimensionless, typically
        ≥ 100, default 500).
    Delta_min : float
        Deformation-rate regulariser [1/s] passed through to
        :func:`delta_deformation`.

    Returns
    -------
    sigma_11_new, sigma_22_new, sigma_12_new : arrays
        Updated stress tensor [N/m].

    Raises
    ------
    ValueError
        If ``alpha < 1`` (relaxation factor ``1/alpha > 1`` produces
        anti-relaxation past the VP target, including the ``alpha=0``
        divide-by-zero corner).

    Notes
    -----
    ``e_yield``, ``alpha``, and ``Delta_min`` are **static** Python
    scalars: they appear in the relaxation factor and the validation
    branches.  Pass them as Python ``float`` constants, never as
    JAX-traced arrays.  Under ``jax.jit`` declare them with
    ``static_argnames`` so the Python control flow does not run on
    tracers.
    """
    if not np.isfinite(alpha) or alpha < 1.0:
        raise ValueError(
            f"mevp_stress_update: alpha must be a finite scalar >= 1 to "
            f"contract toward the VP target; got {alpha}.  "
            f"alpha=NaN/inf silently poisons the stress; "
            f"alpha=0 divides by zero; alpha<1 extrapolates past the VP "
            f"target (anti-relaxation)."
        )

    Delta = delta_deformation(eps_11, eps_22, eps_12, e_yield, Delta_min)

    s11_vp, s22_vp, s12_vp = vp_stress(
        eps_11, eps_22, eps_12, P, Delta, e_yield,
    )

    inv_alpha = 1.0 / alpha
    one_minus = 1.0 - inv_alpha
    sigma_11_new = one_minus * sigma_11 + inv_alpha * s11_vp
    sigma_22_new = one_minus * sigma_22 + inv_alpha * s22_vp
    sigma_12_new = one_minus * sigma_12 + inv_alpha * s12_vp

    return sigma_11_new, sigma_22_new, sigma_12_new
