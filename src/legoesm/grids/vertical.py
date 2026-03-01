"""Vertical coordinates for legoESM.

Two vertical coordinate systems are provided:

1. **SigmaCoordinate** (pressure-based, for hydrostatic primitive equations):
   σ = p / p_s, terrain-following in pressure space.

2. **HeightCoordinate** (height-based, for non-hydrostatic compressible Euler):
   z* = H·(z - z_s)/(H - z_s), terrain-following in physical height space.
   Includes 1D reference state profiles (rho_0, theta_0, pi_0) for
   reference-state subtraction.

3. **TerrainMetric**: 3D coordinate metric terms that depend on surface
   elevation z_s(x, y). Used with HeightCoordinate for terrain-following
   transformations.

All functions are pure (no side effects) and compatible with jax.jit,
jax.grad, jax.vmap, and jax.lax.scan.

References
----------
- Simmons & Burridge (1981): An Energy and Angular-Momentum Conserving
  Vertical Finite-Difference Scheme and Hybrid Vertical Coordinates.
- Klemp et al. (2007): A Terrain-Following Coordinate with Smoothed
  Coordinate Surfaces.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants


class SigmaCoordinate(NamedTuple):
    """Sigma vertical coordinate definition.

    Levels are indexed top-to-bottom: k=0 is the model top (σ≈0),
    k=n_levels-1 is the lowest level (σ≈1).

    Fields
    ------
    n_levels : int
        Number of full levels.
    sigma_full : jax.Array
        Sigma values at full (mid-) levels, shape (n_levels,).
        These are cell centers where prognostic variables live.
    sigma_half : jax.Array
        Sigma values at half (interface) levels, shape (n_levels+1,).
        sigma_half[0] = 0 (model top), sigma_half[-1] = 1 (surface).
    dsigma : jax.Array
        Layer thickness in sigma, shape (n_levels,).
        dsigma[k] = sigma_half[k+1] - sigma_half[k].
    ln_ratio : jax.Array
        Log ratio ln(σ_{k+1/2} / σ_{k-1/2}), shape (n_levels,).
        Used in geopotential computation.
    alpha : jax.Array
        Simmons-Burridge coefficient ln(σ_{k+1/2} / σ_full_k), shape (n_levels,).
        Used in geopotential computation.
    fractional_sigma : jax.Array
        (σ_{k+1/2} - σ_top) / (1 - σ_top) for k=1..n_levels, shape (n_levels,).
        Used in sigma-dot computation.
    dsigma_full : jax.Array
        Differences between full levels: diff(sigma_full), shape (n_levels-1,).
        Used in vertical advection.
    """
    n_levels: int
    sigma_full: jax.Array
    sigma_half: jax.Array
    dsigma: jax.Array
    ln_ratio: jax.Array
    alpha: jax.Array
    fractional_sigma: jax.Array
    dsigma_full: jax.Array


def create_sigma_coordinate(
    n_levels: int,
    sigma_top: float = 0.01,
) -> SigmaCoordinate:
    """Create a uniformly spaced sigma coordinate.

    Parameters
    ----------
    n_levels : int
        Number of vertical levels.
    sigma_top : float
        Sigma value at the model top. Default 0.01 corresponds to
        p_top ≈ 10 hPa when p_s = 1000 hPa. Must be > 0 to avoid
        log(0) in the hydrostatic geopotential integration.

    Returns
    -------
    SigmaCoordinate
        The vertical coordinate definition.
    """
    # Always use float32 for sigma coordinate arrays. The PE and tracer
    # transport models run in float32, and mixing float64 sigma arrays
    # with float32 state arrays triggers scatter-cast warnings.
    sigma_half = jnp.linspace(sigma_top, 1.0, n_levels + 1, dtype=jnp.float32)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    dsigma = sigma_half[1:] - sigma_half[:-1]

    # Precompute constants for geopotential (avoids log calls in hot loop)
    sigma_half_safe = jnp.clip(sigma_half, 1e-30, None)
    ln_ratio = jnp.log(sigma_half_safe[1:] / sigma_half_safe[:-1])  # (n_levels,)
    alpha = jnp.log(sigma_half_safe[1:] / sigma_full)  # (n_levels,)

    # Precompute fractional sigma for sigma-dot computation
    sigma_range = 1.0 - sigma_top
    fractional_sigma = (sigma_half[1:] - sigma_top) / sigma_range  # (n_levels,)

    # Precompute full-level differences for vertical advection
    dsigma_full = jnp.diff(sigma_full)  # (n_levels-1,)

    return SigmaCoordinate(
        n_levels=n_levels,
        sigma_full=sigma_full,
        sigma_half=sigma_half,
        dsigma=dsigma,
        ln_ratio=ln_ratio,
        alpha=alpha,
        fractional_sigma=fractional_sigma,
        dsigma_full=dsigma_full,
    )


def pressure_from_sigma(
    sigma: jax.Array,
    p_s: jax.Array,
) -> jax.Array:
    """Compute pressure from sigma and surface pressure.

    p = σ · p_s

    Parameters
    ----------
    sigma : jax.Array
        Sigma values, shape (nlev,) or (nlev+1,).
    p_s : jax.Array
        Surface pressure, shape (6, n, n).

    Returns
    -------
    jax.Array
        Pressure, shape (6, n, n, nlev) or (6, n, n, nlev+1).
    """
    # Broadcast: p_s[..., None] is (6,n,n,1), sigma is (nlev,) -> (6,n,n,nlev)
    return p_s[..., None] * sigma[None, None, None, :]


def compute_geopotential(
    T: jax.Array,
    p_s: jax.Array,
    sigma_coord: SigmaCoordinate,
    phis: jax.Array,
) -> jax.Array:
    """Compute geopotential at full levels via Simmons-Burridge integration.

    Uses the standard Simmons & Burridge (1981) hydrostatic integration
    with the proper α_k correction for placing the geopotential at
    full (mid-layer) levels rather than interfaces. This gives much
    better accuracy than simple interface averaging, especially for the
    top layers where σ spacing is large in ln(p).

    The geopotential at full level k is:

        Φ_k = Φ_s + R_d Σ_{k'=k+1}^{nlev-1} T_{k'} ln(σ_{k'+1/2}/σ_{k'-1/2})
              + α_k R_d T_k

    where the Simmons-Burridge coefficient:

        α_k = 1 - (σ_{k-1/2} / Δσ_k) ln(σ_{k+1/2} / σ_{k-1/2})

    This coefficient accounts for the sub-layer vertical distribution
    of pressure. For thin layers (Δσ/σ → 0), α_k → 0.5·ln(σ_{k+1/2}/σ_{k-1/2}),
    recovering the interface-average. For thick layers (like the top level),
    α_k differs significantly from the simple average and gives much
    better accuracy.

    Parameters
    ----------
    T : jax.Array
        Temperature, shape (6, n, n, nlev).
    p_s : jax.Array
        Surface pressure, shape (6, n, n).
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    phis : jax.Array
        Surface geopotential (g·z_s), shape (6, n, n).

    Returns
    -------
    jax.Array
        Geopotential at full levels, shape (6, n, n, nlev).
    """
    R_d = constants.R_d

    # Use precomputed log ratios and alpha coefficients
    ln_ratio = sigma_coord.ln_ratio  # (nlev,)
    alpha = sigma_coord.alpha  # (nlev,)

    # Geopotential thickness of each full layer (interface to interface):
    # ΔΦ_k = R_d · T_k · ln(σ_{k+1/2} / σ_{k-1/2})
    dPhi = R_d * T * ln_ratio[None, None, None, :]  # (6,n,n,nlev)

    # Geopotential at the BOTTOM interface of each layer:
    # Φ_bottom_interface[k] = phis + Σ_{k'=k}^{nlev-1} ΔΦ_{k'}
    # (i.e., the interface ABOVE layer k, which is the bottom of layer k-1)
    # Build from bottom: cumulative sum from k to nlev-1
    dPhi_reversed = dPhi[..., ::-1]
    cumsum_reversed = jnp.cumsum(dPhi_reversed, axis=-1)
    cumsum = cumsum_reversed[..., ::-1]  # cumsum[k] = sum from k to nlev-1

    # Interface geopotentials: Φ_interface[k] = phis + Σ_{k'=k}^{nlev-1} ΔΦ_{k'}
    Phi_above = phis[..., None] + cumsum  # (6,n,n,nlev) — top of each layer

    # Full-level geopotential using Simmons-Burridge:
    # Φ_k = Φ_interface[k+1] + α_k · R_d · T_k
    # where Φ_interface[k+1] is the BOTTOM interface of layer k.
    #
    # Φ_interface[k+1] for k=0..nlev-2 is Phi_above[k+1]
    # Φ_interface[nlev] (surface) = phis
    Phi_below = jnp.concatenate(
        [Phi_above[..., 1:], phis[..., None]], axis=-1
    )  # (6,n,n,nlev) — bottom interface of each layer

    Phi_full = Phi_below + alpha[None, None, None, :] * R_d * T  # (6,n,n,nlev)

    return Phi_full


def compute_sigma_dot(
    div_3d: jax.Array,
    sigma_coord: SigmaCoordinate,
) -> jax.Array:
    """Diagnose sigma-dot (vertical velocity in sigma coordinates).

    From the continuity equation in sigma coordinates with σ_top > 0:

        ∂(ln p_s)/∂t + div(v) + ∂σ̇/∂σ = 0

    where ∂(ln p_s)/∂t = -D_total / (1 - σ_top).

    Integrating from the top boundary (σ = σ_top, where σ̇ = 0):

        σ̇_{k+1/2} = (σ_{k+1/2} - σ_top) / (1 - σ_top) · D_total
                     - Σ_{k'=0}^{k} div(v_k') · Δσ_k'

    where D_total = Σ div(v_k) · Δσ_k.

    The factor (σ_{k+1/2} - σ_top) / (1 - σ_top) ensures exact
    discrete continuity closure: the residual
    ∂(ln p_s)/∂t + div(v_k) + (σ̇_{k+1/2} - σ̇_{k-1/2})/Δσ_k = 0
    holds to machine precision at every level.

    Boundary conditions: σ̇ = 0 at σ = σ_top (top) and σ = 1 (surface).

    Parameters
    ----------
    div_3d : jax.Array
        Horizontal divergence at each level, shape (6, n, n, nlev).
    sigma_coord : SigmaCoordinate
        Vertical coordinate.

    Returns
    -------
    jax.Array
        Sigma-dot at half (interface) levels, shape (6, n, n, nlev+1).
        Index 0 = model top (σ̇=0), index nlev = surface (σ̇=0).
    """
    dsigma = sigma_coord.dsigma  # (nlev,)

    # Weighted divergence: D_k * Δσ_k
    div_dsigma = div_3d * dsigma[None, None, None, :]  # (6,n,n,nlev)

    # Column-integrated divergence: D_total = Σ D_k * Δσ_k
    D_total = jnp.sum(div_dsigma, axis=-1, keepdims=True)  # (6,n,n,1)

    # Cumulative sum from top: Σ_{k'=0}^{k} D_k' * Δσ_k'
    cumsum_div = jnp.cumsum(div_dsigma, axis=-1)  # (6,n,n,nlev)

    # σ̇ at interfaces 1..nlev:
    # σ̇_{k+1/2} = (σ_{k+1/2} - σ_top) / (1 - σ_top) · D_total - cumsum_div[k]
    # Use precomputed fractional_sigma
    fractional_sigma = sigma_coord.fractional_sigma[None, None, None, :]  # (1,1,1,nlev)

    sigma_dot_inner = (
        fractional_sigma * D_total - cumsum_div
    )  # (6,n,n,nlev)

    # Prepend top (σ̇=0)
    shape_2d = div_3d.shape[:3]  # (6, n, n)
    zero_top = jnp.zeros((*shape_2d, 1))
    sigma_dot = jnp.concatenate([zero_top, sigma_dot_inner], axis=-1)  # (6,n,n,nlev+1)

    # Force bottom boundary (should be zero by construction, enforce for safety)
    sigma_dot = sigma_dot.at[..., -1].set(0.0)

    return sigma_dot


def vertical_advection(
    field: jax.Array,
    sigma_dot: jax.Array,
    sigma_coord: SigmaCoordinate,
) -> jax.Array:
    """Compute vertical advection using the advective form with upwind.

    Computes: -σ̇ · ∂f/∂σ

    This is the **advective form**, appropriate for non-mass-weighted
    variables (u, v, T). The flux form -∂(σ̇·f)/∂σ would add a spurious
    term -f·∂σ̇/∂σ that causes exponential instability.

    σ̇ is interpolated from half-levels to full levels, and the vertical
    gradient uses upwind differencing:
    - σ̇ > 0 (downward): ∂f/∂σ ≈ (f_k - f_{k-1}) / Δσ  (backward)
    - σ̇ < 0 (upward):   ∂f/∂σ ≈ (f_{k+1} - f_k) / Δσ  (forward)

    Parameters
    ----------
    field : jax.Array
        Field to advect, shape (6, n, n, nlev).
    sigma_dot : jax.Array
        Sigma-dot at interfaces, shape (6, n, n, nlev+1).
    sigma_coord : SigmaCoordinate
        Vertical coordinate (provides sigma_full and dsigma).

    Returns
    -------
    jax.Array
        Vertical advection tendency, shape (6, n, n, nlev).
    """
    # Interpolate σ̇ from half-levels to full levels
    sigma_dot_full = 0.5 * (sigma_dot[..., :-1] + sigma_dot[..., 1:])  # (6,n,n,nlev)

    # Backward difference: ∂f/∂σ ≈ (f_k - f_{k-1}) / (σ_k - σ_{k-1})
    # Pad top with zero-gradient BC: f_{-1} = f_0
    dsigma_bwd = sigma_coord.dsigma_full  # (nlev-1,) precomputed diff(sigma_full)
    df_bwd = jnp.diff(field, axis=-1)  # (6,n,n,nlev-1) f_{k+1} - f_k
    # At k=0 (top level): backward gradient = 0 (zero-gradient BC)
    grad_bwd = jnp.concatenate(
        [jnp.zeros((*field.shape[:3], 1)),
         df_bwd / dsigma_bwd[None, None, None, :]],
        axis=-1,
    )  # (6,n,n,nlev)

    # Forward difference: ∂f/∂σ ≈ (f_{k+1} - f_k) / (σ_{k+1} - σ_k)
    # At k=nlev-1 (bottom level): forward gradient = 0 (zero-gradient BC)
    grad_fwd = jnp.concatenate(
        [df_bwd / dsigma_bwd[None, None, None, :],
         jnp.zeros((*field.shape[:3], 1))],
        axis=-1,
    )  # (6,n,n,nlev)

    # Upwind selection: σ̇ > 0 (downward) → backward, σ̇ < 0 (upward) → forward
    grad = jnp.where(sigma_dot_full > 0, grad_bwd, grad_fwd)

    # Tendency: -σ̇ · ∂f/∂σ
    return -sigma_dot_full * grad


def compute_pressure_velocity(
    sigma_dot: jax.Array,
    p_s: jax.Array,
    dp_s_dt: jax.Array,
    sigma_coord: SigmaCoordinate,
) -> jax.Array:
    """Compute pressure velocity omega (ω = dp/dt) at full levels.

    ω = σ · (dp_s/dt) + p_s · σ̇

    This is needed for the adiabatic heating term κ·T·ω/p.

    Parameters
    ----------
    sigma_dot : jax.Array
        Sigma-dot at interfaces, shape (6, n, n, nlev+1).
    p_s : jax.Array
        Surface pressure, shape (6, n, n).
    dp_s_dt : jax.Array
        Surface pressure tendency, shape (6, n, n).
    sigma_coord : SigmaCoordinate
        Vertical coordinate.

    Returns
    -------
    jax.Array
        Pressure velocity (omega) at full levels, shape (6, n, n, nlev).
    """
    sigma_full = sigma_coord.sigma_full  # (nlev,)

    # Average sigma_dot to full levels
    sigma_dot_full = 0.5 * (sigma_dot[..., :-1] + sigma_dot[..., 1:])  # (6,n,n,nlev)

    # ω = σ · dp_s/dt + p_s · σ̇
    omega = (
        sigma_full[None, None, None, :] * dp_s_dt[..., None]
        + p_s[..., None] * sigma_dot_full
    )

    return omega


# ==============================================================================
# Height-based vertical coordinate (non-hydrostatic)
# ==============================================================================

class HeightCoordinate(NamedTuple):
    """Height-based terrain-following vertical coordinate (z-star).

    The z-star coordinate is defined as:

        z* = H · (z - z_s) / (H - z_s)

    where H is the model top height and z_s is the surface elevation.
    z* ranges from 0 at the surface to H at the model top.

    Levels are indexed top-to-bottom: k=0 is the model top (z*=H),
    k=n_levels-1 is the lowest level (z* near 0).

    The reference state (rho_0, theta_0, pi_0) is a 1D hydrostatically
    balanced profile used for reference-state subtraction in the
    compressible Euler equations.

    Fields
    ------
    n_levels : int
        Number of full levels.
    H : float
        Model top height [m].
    z_full : jax.Array
        z* at full (mid-) levels [m], shape (nlev,). Top-to-bottom.
    z_half : jax.Array
        z* at half (interface) levels [m], shape (nlev+1,). Top-to-bottom.
        z_half[0] = H (model top), z_half[-1] = 0 (surface).
    dz : jax.Array
        Layer thickness dz* [m], shape (nlev,).
        dz[k] = z_half[k] - z_half[k+1] (positive since top-to-bottom).
    dz_half : jax.Array
        Distance between adjacent full levels [m], shape (nlev-1,).
        dz_half[k] = z_full[k] - z_full[k+1].
    rho_ref : jax.Array
        Reference density at full levels [kg/m^3], shape (nlev,).
    theta_ref : jax.Array
        Reference potential temperature at full levels [K], shape (nlev,).
    exner_ref : jax.Array
        Reference Exner function at full levels [-], shape (nlev,).
    exner_ref_half : jax.Array
        Reference Exner function at half levels [-], shape (nlev+1,).
    rho_ref_half : jax.Array
        Reference density at half levels [kg/m^3], shape (nlev+1,).
    """
    n_levels: int
    H: float
    z_full: jax.Array
    z_half: jax.Array
    dz: jax.Array
    dz_half: jax.Array
    rho_ref: jax.Array
    theta_ref: jax.Array
    exner_ref: jax.Array
    exner_ref_half: jax.Array
    rho_ref_half: jax.Array


class TerrainMetric(NamedTuple):
    """Terrain-following coordinate metric terms.

    These are 3D arrays that depend on horizontal position through
    the surface elevation z_s(x, y).

    The Jacobian J = dz/dz* = (H - z_s) / H maps between z* and
    physical z. All vertical derivatives in z* must be divided by J
    to get derivatives in physical z.

    Fields
    ------
    z_s : jax.Array
        Surface elevation [m], shape (6, n, n).
    jacobian : jax.Array
        dz/dz* = (H - z_s) / H, shape (6, n, n). Always > 0.
    z_full_3d : jax.Array
        Physical z at full levels [m], shape (6, n, n, nlev).
    z_half_3d : jax.Array
        Physical z at half levels [m], shape (6, n, n, nlev+1).
    """
    z_s: jax.Array
    jacobian: jax.Array
    z_full_3d: jax.Array
    z_half_3d: jax.Array


def _default_theta_ref(z: jax.Array) -> jax.Array:
    """Default reference potential temperature: isothermal at 300 K.

    For an isothermal atmosphere T=T0, the potential temperature is:
        theta(z) = T0 * (p0/p(z))^kappa
    But for simplicity, we use a constant theta_0 = 300 K.
    """
    return jnp.full_like(z, 300.0)


def compute_reference_state(
    z: jax.Array,
    theta_ref_fn: Callable[[jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute 1D reference state by integrating hydrostatic balance.

    Given theta_0(z), integrate the hydrostatic equation downward from
    the model top:

        d(pi_0)/dz = -g / (c_p · theta_0(z))

    where pi = (p / p_0)^kappa is the dimensionless Exner function.

    Then recover density from the equation of state:

        rho_0 = p_0 · pi_0^(c_v/R_d) / (R_d · theta_0)

    Parameters
    ----------
    z : jax.Array
        Height values [m], shape (n_points,). Must be sorted
        top-to-bottom (decreasing).
    theta_ref_fn : callable
        Function theta_0(z) -> potential temperature [K].

    Returns
    -------
    rho_0 : jax.Array
        Reference density [kg/m^3], shape (n_points,).
    theta_0 : jax.Array
        Reference potential temperature [K], shape (n_points,).
    exner_0 : jax.Array
        Reference Exner function [-], shape (n_points,).
    """
    g = constants.g
    c_p = constants.c_pd
    c_v = constants.c_vd
    R_d = constants.R_d
    p_0 = constants.p_ref

    theta_0 = theta_ref_fn(z)

    # Integrate d(pi)/dz = -g / (c_p * theta_0) downward from top.
    # Use trapezoidal rule: pi[k+1] = pi[k] + (-g/(c_p*theta_avg)) * (z[k+1]-z[k])
    # Note: z is top-to-bottom, so z[k+1] < z[k], and dz = z[k+1]-z[k] < 0.
    # This means pi increases downward (as expected).

    # Start with pi at model top. Use a reasonable value:
    # T_top = theta_top * pi_top => pi_top = T_top / theta_top
    # For ~40km top, T ~ 250K, theta ~ 1000K => pi ~ 0.25
    # Better: use standard atmosphere pressure at model top.
    # p_top = p_0 * exp(-g * z_top / (R_d * T_avg))
    z_top = z[0]
    T_avg = 250.0  # rough average temperature for scale height
    p_top = p_0 * jnp.exp(-g * z_top / (R_d * T_avg))
    pi_top = (p_top / p_0) ** (R_d / c_p)

    # Integrate downward level by level
    n = z.shape[0]
    dz_vals = jnp.diff(z)  # (n-1,) — negative since z decreasing

    # Trapezoidal integration of -g / (c_p * theta_0)
    integrand = -g / (c_p * theta_0)  # (n,)
    integrand_avg = 0.5 * (integrand[:-1] + integrand[1:])  # (n-1,)
    d_pi = integrand_avg * dz_vals  # (n-1,)

    # Cumulative sum gives pi at each level
    pi_increments = jnp.cumsum(d_pi)  # (n-1,)
    exner_0 = jnp.concatenate([jnp.array([pi_top]), pi_top + pi_increments])

    # Recover density from equation of state:
    # p = p_0 * pi^(c_p/R_d)
    # rho = p / (R_d * T) = p / (R_d * theta * pi)
    pressure = p_0 * exner_0 ** (c_p / R_d)
    rho_0 = pressure / (R_d * theta_0 * exner_0)

    return rho_0, theta_0, exner_0


def create_height_coordinate(
    n_levels: int,
    H: float,
    theta_ref_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> HeightCoordinate:
    """Create a uniformly-spaced height-based (z-star) vertical coordinate.

    Parameters
    ----------
    n_levels : int
        Number of full vertical levels.
    H : float
        Model top height [m].
    theta_ref_fn : callable, optional
        Function theta_0(z) -> potential temperature [K].
        Default: constant 300 K (isothermal reference).

    Returns
    -------
    HeightCoordinate
        The vertical coordinate with precomputed reference state.
    """
    if theta_ref_fn is None:
        theta_ref_fn = _default_theta_ref

    # z* grid: top-to-bottom (z_half[0] = H, z_half[-1] = 0)
    z_half = jnp.linspace(H, 0.0, n_levels + 1, dtype=jnp.float32)
    z_full = 0.5 * (z_half[:-1] + z_half[1:])  # (nlev,)
    dz = z_half[:-1] - z_half[1:]  # (nlev,) positive
    dz_half = z_full[:-1] - z_full[1:]  # (nlev-1,) positive

    # Compute reference state at full and half levels
    rho_ref, theta_ref, exner_ref = compute_reference_state(
        z_full, theta_ref_fn
    )
    rho_ref_half, _, exner_ref_half = compute_reference_state(
        z_half, theta_ref_fn
    )

    return HeightCoordinate(
        n_levels=n_levels,
        H=H,
        z_full=z_full,
        z_half=z_half,
        dz=dz,
        dz_half=dz_half,
        rho_ref=rho_ref,
        theta_ref=theta_ref,
        exner_ref=exner_ref,
        exner_ref_half=exner_ref_half,
        rho_ref_half=rho_ref_half,
    )


def compute_terrain_metric(
    z_s: jax.Array,
    height_coord: HeightCoordinate,
) -> TerrainMetric:
    """Compute 3D terrain-following coordinate metrics.

    The z-star to physical z transformation is:

        z(z*, x, y) = z_s(x, y) + z* · (H - z_s(x, y)) / H

    The Jacobian is:

        J = dz/dz* = (H - z_s) / H

    Parameters
    ----------
    z_s : jax.Array
        Surface elevation [m], shape (6, n, n).
    height_coord : HeightCoordinate
        Vertical coordinate.

    Returns
    -------
    TerrainMetric
        3D coordinate metric terms.
    """
    H = height_coord.H

    # Jacobian: J = (H - z_s) / H, shape (6, n, n)
    jacobian = (H - z_s) / H

    # Physical z at full levels: z = z_s + z* * (H - z_s) / H
    # z_full is (nlev,), z_s is (6,n,n) -> z_full_3d is (6,n,n,nlev)
    z_full_3d = (
        z_s[..., None]
        + height_coord.z_full[None, None, None, :] * jacobian[..., None]
    )

    # Physical z at half levels: same formula with z_half
    z_half_3d = (
        z_s[..., None]
        + height_coord.z_half[None, None, None, :] * jacobian[..., None]
    )

    return TerrainMetric(
        z_s=z_s,
        jacobian=jacobian,
        z_full_3d=z_full_3d,
        z_half_3d=z_half_3d,
    )
