"""Vertical coordinates for legoESM.

Three vertical coordinate systems are provided:

1. **SigmaCoordinate** (pressure-based, for hydrostatic primitive equations):
   σ = p / p_s, terrain-following in pressure space.

2. **HybridSigmaPressureCoordinate** (hybrid σ-p, for hydrostatic PE):
   p(k) = A(k)·p_ref + B(k)·p_s.  Near the top, levels are pure
   pressure surfaces (B→0); near the surface, pure sigma (A→0, B→1).
   Eliminates pressure-gradient force errors over steep topography.

3. **HeightCoordinate** (height-based, for non-hydrostatic compressible Euler):
   z* = H·(z - z_s)/(H - z_s), terrain-following in physical height space.
   Includes 1D reference state profiles (rho_0, theta_0, pi_0) for
   reference-state subtraction.

4. **TerrainMetric**: 3D coordinate metric terms that depend on surface
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

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)

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

    # ------------------------------------------------------------------
    # VerticalCoordProtocol methods
    # ------------------------------------------------------------------

    def pressure_at_full(self, p_s):
        return pressure_from_sigma(self.sigma_full, p_s)

    def pressure_at_half(self, p_s):
        return pressure_from_sigma(self.sigma_half, p_s)

    def layer_thickness_dp(self, p_s):
        return self.dsigma * p_s[..., None]


def create_sigma_coordinate(
    n_levels: int,
    sigma_top: float = 0.01,
    dtype=None,
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
    dtype : jnp.dtype or None
        Dtype for coordinate arrays. If None, uses the precision
        policy's compute dtype (defaults to float32 when no policy
        is active). Explicit dtype overrides the policy.

    Returns
    -------
    SigmaCoordinate
        The vertical coordinate definition.
    """
    # Resolve dtype from precision policy if not explicitly provided.
    # Default to float32 for backward compatibility (PE and tracer
    # transport models run in float32, and mixing float64 sigma arrays
    # with float32 state arrays triggers scatter-cast warnings).
    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().compute
        except Exception:
            dtype = jnp.float32
    sigma_half = jnp.linspace(sigma_top, 1.0, n_levels + 1, dtype=dtype)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    dsigma = sigma_half[1:] - sigma_half[:-1]

    # Precompute constants for geopotential (avoids log calls in hot loop)
    sigma_half_safe = jnp.clip(sigma_half, _TINY, None)
    ln_ratio = jnp.log(sigma_half_safe[1:] / sigma_half_safe[:-1])  # (n_levels,)
    # Exact Simmons-Burridge (1981) alpha coefficient:
    #   α_k = 1 - (σ_{k-1/2} / Δσ_k) * ln(σ_{k+1/2} / σ_{k-1/2})
    alpha = 1.0 - (sigma_half_safe[:-1] / dsigma) * ln_ratio  # (n_levels,)

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
    # Broadcast: p_s[..., None] is (...,1), sigma is (nlev,) -> (...,nlev)
    return p_s[..., None] * sigma


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
    dPhi = R_d * T * ln_ratio  # (...,nlev)

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

    Phi_full = Phi_below + alpha * R_d * T  # (...,nlev)

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
    div_dsigma = div_3d * dsigma  # (...,nlev)

    # Cumulative sum from top: Σ_{k'=0}^{k} D_k' * Δσ_k'
    # ``cumsum_div[..., -1]`` is exactly ``Σ D_k * Δσ_k`` — extract it
    # rather than calling ``jnp.sum`` independently.  Under any sharding
    # of the level (axis -1) this drops the per-call cross-level
    # collective from 2 to 1.  Mirrors iter-50/51 in the spectral PE.
    cumsum_div = jnp.cumsum(div_dsigma, axis=-1)  # (...,nlev)
    D_total = cumsum_div[..., -1:]  # (...,1)

    # σ̇ at interfaces 1..nlev:
    # σ̇_{k+1/2} = (σ_{k+1/2} - σ_top) / (1 - σ_top) · D_total - cumsum_div[k]
    # Use precomputed fractional_sigma
    fractional_sigma = sigma_coord.fractional_sigma  # (nlev,)

    sigma_dot_inner = (
        fractional_sigma * D_total - cumsum_div
    )  # (...,nlev)

    # Prepend top (σ̇=0) and force bottom boundary (zero by construction).
    # Drop the (∼0) last element + pad with zeros on both ends in one
    # ``jnp.pad`` — replaces alloc-zeros + concatenate + scatter (3 HLO
    # ops) with slice + Pad (2 HLO ops).
    pad_axes = ((0, 0),) * (sigma_dot_inner.ndim - 1)
    sigma_dot = jnp.pad(sigma_dot_inner[..., :-1], (*pad_axes, (1, 1)))

    return sigma_dot


def compute_sigma_dot_and_total(
    div_3d: jax.Array,
    sigma_coord: SigmaCoordinate,
) -> tuple[jax.Array, jax.Array]:
    """Diagnose sigma-dot AND return the column-integrated divergence.

    Identical to :func:`compute_sigma_dot` for the σ̇ output, but also
    returns the column-integrated divergence ``D_total = Σ div_k · Δσ_k``
    (shape ``(..., 1)``).  Callers that need both σ̇ and ``D_total``
    (e.g. the cubed-sphere FV3 PE non-hybrid path, which uses the
    column sum for ``dp_s/dt = -p_s · D_total / (1 - σ_top)``) can use
    this single call instead of running both ``jnp.sum`` and
    ``compute_sigma_dot`` — saving one cross-level collective per call.

    See :func:`compute_sigma_dot` for full documentation.

    Returns
    -------
    sigma_dot : jax.Array, shape (..., nlev+1)
    D_total : jax.Array, shape (..., 1)
    """
    dsigma = sigma_coord.dsigma  # (nlev,)
    div_dsigma = div_3d * dsigma  # (..., nlev)
    cumsum_div = jnp.cumsum(div_dsigma, axis=-1)  # (..., nlev)
    D_total = cumsum_div[..., -1:]  # (..., 1)

    fractional_sigma = sigma_coord.fractional_sigma  # (nlev,)
    sigma_dot_inner = fractional_sigma * D_total - cumsum_div

    pad_axes = ((0, 0),) * (sigma_dot_inner.ndim - 1)
    sigma_dot = jnp.pad(sigma_dot_inner[..., :-1], (*pad_axes, (1, 1)))

    return sigma_dot, D_total


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

    # Backward / forward differences both consume ``df_bwd / dsigma_bwd``
    # — compute the inner gradient once and pad along the trailing
    # axis instead of building two padded arrays via fresh
    # ``jnp.zeros`` + concatenate.  Single Pad HLO op each.
    dsigma_bwd = sigma_coord.dsigma_full  # (nlev-1,) precomputed diff(sigma_full)
    df_bwd = jnp.diff(field, axis=-1)  # (...,nlev-1) f_{k+1} - f_k
    df_over = df_bwd / dsigma_bwd
    pad_axes = ((0, 0),) * (df_over.ndim - 1)
    # At k=0 (top level): backward gradient = 0 (zero-gradient BC)
    grad_bwd = jnp.pad(df_over, (*pad_axes, (1, 0)))
    # At k=nlev-1 (bottom level): forward gradient = 0 (zero-gradient BC)
    grad_fwd = jnp.pad(df_over, (*pad_axes, (0, 1)))

    # Upwind selection: σ̇ > 0 (downward) → backward, σ̇ < 0 (upward) → forward
    grad = jnp.where(sigma_dot_full > 0, grad_bwd, grad_fwd)

    # Tendency: -σ̇ · ∂f/∂σ
    return -sigma_dot_full * grad


def vertical_advection_theta(
    T: jax.Array,
    sigma_dot: jax.Array,
    p_s: jax.Array,
    sigma_coord: SigmaCoordinate,
) -> jax.Array:
    """Combined vertical advection + adiabatic σ̇ term for temperature.

    Instead of computing  -σ̇·∂T/∂σ  and  κ·T·σ̇/σ  separately (which
    involves catastrophic cancellation at upper levels where 1/σ → ∞),
    this function advects potential temperature θ and converts back:

        -σ̇·∂T/∂σ + κ·T·σ̇/σ  =  -(p/p₀)^κ · σ̇·∂θ/∂σ

    This eliminates the 1/σ amplification and is numerically stable at
    all levels.

    Parameters
    ----------
    T : jax.Array
        Temperature, shape (..., nlev).
    sigma_dot : jax.Array
        Sigma-dot at interfaces, shape (..., nlev+1).
    p_s : jax.Array
        Surface pressure, shape (...).
    sigma_coord : SigmaCoordinate
        Vertical coordinate.

    Returns
    -------
    jax.Array
        Combined tendency: -σ̇·∂T/∂σ + κ·T·σ̇/σ, shape (..., nlev).
    """
    kappa = constants.kappa
    sigma_full = sigma_coord.sigma_full  # (nlev,)
    P_0 = 1.0e5

    # Pressure at full levels
    p_full = sigma_full * p_s[..., None]  # (..., nlev)

    # Potential temperature: θ = T · (p₀/p)^κ
    theta = T * (P_0 / jnp.maximum(p_full, 1.0)) ** kappa

    # Advect θ: -σ̇ · ∂θ/∂σ  (using same upwind scheme)
    adv_theta = vertical_advection(theta, sigma_dot, sigma_coord)

    # Convert back: tendency_T = (p/p₀)^κ · adv_θ
    exner = (p_full / P_0) ** kappa  # (p/p₀)^κ
    return exner * adv_theta


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
        sigma_full * dp_s_dt[..., None]
        + p_s[..., None] * sigma_dot_full
    )

    return omega


# ==============================================================================
# Hybrid sigma-pressure vertical coordinate
# ==============================================================================


class HybridSigmaPressureCoordinate(NamedTuple):
    """Hybrid sigma-pressure vertical coordinate.

    Pressure at level k: p(k) = A(k) * p_ref + B(k) * p_s

    Near the model top B → 0 (pure pressure levels, no terrain influence).
    Near the surface A → 0, B → 1 (terrain-following sigma coordinate).
    This eliminates the large pressure-gradient force errors that pure
    sigma coordinates produce over steep topography.

    Levels are indexed top-to-bottom: k=0 is the model top, k=nlev-1
    is the lowest level.

    Fields
    ------
    n_levels : int
        Number of full levels.
    A_half : jax.Array
        Interface A coefficients, shape (nlev+1,). A_half[0] = p_top/p_ref.
    B_half : jax.Array
        Interface B coefficients, shape (nlev+1,). B_half[0] = 0 (top).
    A_full : jax.Array
        Mid-level A coefficients, shape (nlev,).
    B_full : jax.Array
        Mid-level B coefficients, shape (nlev,).
    p_ref : float
        Reference surface pressure [Pa], typically 1e5.
    dA : jax.Array
        A_half[k+1] - A_half[k], shape (nlev,).
    dB : jax.Array
        B_half[k+1] - B_half[k], shape (nlev,).
    B_range : float
        B_half[-1] - B_half[0], typically 1.0.
    ln_ratio_ref : jax.Array
        Precomputed log(p_half[k+1]/p_half[k]) at p_s = p_ref, shape (nlev,).
    alpha_ref : jax.Array
        Simmons-Burridge alpha at p_s = p_ref, shape (nlev,).
    dsigma_eff : jax.Array
        Effective layer thickness dA + dB for semi-implicit, shape (nlev,).
    """
    n_levels: int
    A_half: jax.Array
    B_half: jax.Array
    A_full: jax.Array
    B_full: jax.Array
    p_ref: float
    dA: jax.Array
    dB: jax.Array
    B_range: float
    ln_ratio_ref: jax.Array
    alpha_ref: jax.Array
    dsigma_eff: jax.Array

    # ------------------------------------------------------------------
    # VerticalCoordProtocol methods
    # ------------------------------------------------------------------

    def pressure_at_full(self, p_s):
        return pressure_from_hybrid(self, p_s, full=True)

    def pressure_at_half(self, p_s):
        return pressure_from_hybrid(self, p_s, full=False)

    def layer_thickness_dp(self, p_s):
        return dp_from_hybrid(self, p_s)

    # ------------------------------------------------------------------
    # Sigma-compatibility views
    # ------------------------------------------------------------------
    @property
    def sigma_half(self):
        """Effective interface sigma coefficients A_half + B_half.

        This preserves compatibility with utilities that only require a
        monotonic sigma-like coordinate shape and do not assume pure-sigma
        pressure dependence.
        """
        return self.A_half + self.B_half

    @property
    def sigma_full(self):
        """Effective full-level sigma coefficients A_full + B_full."""
        return self.A_full + self.B_full

    @property
    def dsigma(self):
        """Effective layer thickness coefficients dA + dB."""
        return self.dsigma_eff

    @property
    def dsigma_full(self):
        """Difference between full-level sigma coefficients."""
        return jnp.diff(self.sigma_full)

    @property
    def ln_ratio(self):
        """Log ratio ln(p_half[k+1]/p_half[k]) at reference pressure.

        Provides SigmaCoordinate-compatible access for routines like
        ``compute_geopotential`` that expect a ``ln_ratio`` attribute.
        """
        return self.ln_ratio_ref

    @property
    def alpha(self):
        """Simmons-Burridge alpha coefficient at reference pressure.

        Provides SigmaCoordinate-compatible access for routines like
        ``compute_geopotential`` that expect an ``alpha`` attribute.
        """
        return self.alpha_ref

    @property
    def fractional_sigma(self):
        """Fractional sigma for sigma-dot computation.

        fractional_sigma[k] = (sigma_half[k+1] - sigma_top) / (1 - sigma_top)

        This is the same quantity precomputed in SigmaCoordinate and
        required by ``compute_sigma_dot``.
        """
        s_half = self.sigma_half  # A_half + B_half
        sigma_top = s_half[0]
        sigma_range = s_half[-1] - sigma_top
        sigma_range = jnp.where(sigma_range > 0, sigma_range, 1.0)
        return (s_half[1:] - sigma_top) / sigma_range


def create_hybrid_coordinate(
    n_levels: int,
    A_half: jax.Array,
    B_half: jax.Array,
    p_ref: float = 1e5,
    dtype=None,
) -> HybridSigmaPressureCoordinate:
    """Create a hybrid sigma-pressure coordinate from A/B coefficients.

    Parameters
    ----------
    n_levels : int
        Number of full levels.
    A_half : array-like
        Interface A coefficients, shape (nlev+1,).
    B_half : array-like
        Interface B coefficients, shape (nlev+1,).
    p_ref : float
        Reference surface pressure [Pa].
    dtype : jnp.dtype or None
        Dtype for coordinate arrays. If None, uses the precision
        policy's compute dtype (defaults to float32 when no policy
        is active).

    Returns
    -------
    HybridSigmaPressureCoordinate
    """
    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().compute
        except Exception:
            dtype = jnp.float32
    A_half = jnp.asarray(A_half, dtype=dtype)
    B_half = jnp.asarray(B_half, dtype=dtype)

    A_full = 0.5 * (A_half[:-1] + A_half[1:])
    B_full = 0.5 * (B_half[:-1] + B_half[1:])
    dA = A_half[1:] - A_half[:-1]
    dB = B_half[1:] - B_half[:-1]
    B_range = float(B_half[-1] - B_half[0])

    # Reference pressures at p_s = p_ref
    p_half_ref = (A_half + B_half) * p_ref
    p_full_ref = (A_full + B_full) * p_ref
    p_half_ref_safe = jnp.clip(p_half_ref, 1e-10, None)
    p_full_ref_safe = jnp.clip(p_full_ref, 1e-10, None)

    ln_ratio_ref = jnp.log(p_half_ref_safe[1:] / p_half_ref_safe[:-1])
    # Exact Simmons-Burridge (1981) alpha:
    dP_ref = p_half_ref_safe[1:] - p_half_ref_safe[:-1]
    alpha_ref = 1.0 - (p_half_ref_safe[:-1] / dP_ref) * ln_ratio_ref

    dsigma_eff = dA + dB

    return HybridSigmaPressureCoordinate(
        n_levels=n_levels,
        A_half=A_half,
        B_half=B_half,
        A_full=A_full,
        B_full=B_full,
        p_ref=p_ref,
        dA=dA,
        dB=dB,
        B_range=B_range,
        ln_ratio_ref=ln_ratio_ref,
        alpha_ref=alpha_ref,
        dsigma_eff=dsigma_eff,
    )


def make_hybrid_levels(
    n_levels: int,
    p_top_Pa: float = 200.0,
    p_ref: float = 1e5,
    transition_exponent: int = 3,
    stretching: float = 0.0,
) -> HybridSigmaPressureCoordinate:
    """Generate hybrid coordinate with smooth sigma-to-pressure transition.

    Uses the parameterization:
        eta = stretched(linspace(0, 1, nlev+1))
        B = eta^exponent
        A = eta - B + (p_top/p_ref) * (1 - eta)

    At reference p_s = p_ref and zero stretching, levels are uniformly
    spaced in pressure. The exponent controls the sigma-to-pressure
    transition height. The stretching parameter concentrates levels near
    the surface for boundary layer resolution.

    Parameters
    ----------
    n_levels : int
        Number of full levels.
    p_top_Pa : float
        Model top pressure [Pa]. Default 200 Pa (2 hPa).
    p_ref : float
        Reference surface pressure [Pa].
    transition_exponent : int
        Power for B(eta) = eta^exponent. 1 = pure sigma, 3 = typical.
    stretching : float
        Sinh-based stretching parameter (>= 0). 0 = uniform spacing,
        2-3 = enhanced boundary layer resolution. The stretching maps
        eta -> sinh(s*eta)/sinh(s), concentrating levels near eta=1
        (the surface).

    Returns
    -------
    HybridSigmaPressureCoordinate
    """
    import numpy as np

    eta = np.linspace(0.0, 1.0, n_levels + 1)
    if stretching > 0:
        # Concentrate levels near eta=1 (surface) for BL resolution
        eta = 1.0 - np.sinh(stretching * (1.0 - eta)) / np.sinh(stretching)
    B_half = eta ** transition_exponent
    A_half = eta - B_half + (p_top_Pa / p_ref) * (1.0 - eta)

    return create_hybrid_coordinate(n_levels, A_half, B_half, p_ref)


def standard_hybrid_levels(
    n_levels: int = 40,
    p_ref: float = 1e5,
) -> HybridSigmaPressureCoordinate:
    """Create standard hybrid levels with good defaults for any resolution.

    Provides well-tested level designs for common configurations:

    - **L20**: p_top=1000 Pa (10 hPa), moderate stretching.
      5 BL levels, 10 troposphere, 5 stratosphere.
    - **L40** (default): p_top=200 Pa (2 hPa), enhanced BL resolution.
      ~8 BL levels, ~16 troposphere, ~8 UTLS, ~8 stratosphere.
    - **L60**: p_top=10 Pa (0.1 hPa), high BL + stratospheric resolution.
      ~10 BL levels, ~20 troposphere, ~15 UTLS, ~15 stratosphere.
    - **Other**: automatically selects p_top and stretching based on nlev.

    Parameters
    ----------
    n_levels : int
        Number of full levels. Default 40.
    p_ref : float
        Reference surface pressure [Pa].

    Returns
    -------
    HybridSigmaPressureCoordinate
    """
    if n_levels <= 25:
        # Moderate resolution: 10 hPa top, mild stretching
        p_top = 1000.0
        stretching = 1.5
        transition_exponent = 2
    elif n_levels <= 45:
        # Standard resolution: 2 hPa top, good BL resolution
        p_top = 200.0
        stretching = 2.5
        transition_exponent = 3
    else:
        # High resolution: 0.1 hPa top, strong BL + strat resolution
        p_top = 10.0
        stretching = 2.5
        transition_exponent = 3

    return make_hybrid_levels(
        n_levels, p_top_Pa=p_top, p_ref=p_ref,
        transition_exponent=transition_exponent, stretching=stretching,
    )


def hybrid_from_sigma(
    sigma_coord: SigmaCoordinate,
    p_ref: float = 1e5,
) -> HybridSigmaPressureCoordinate:
    """Convert a SigmaCoordinate to hybrid form (A=0, B=sigma).

    Useful for validation: a pure-sigma hybrid should give identical
    results to the original SigmaCoordinate.

    Parameters
    ----------
    sigma_coord : SigmaCoordinate
        Existing sigma coordinate.
    p_ref : float
        Reference surface pressure [Pa].

    Returns
    -------
    HybridSigmaPressureCoordinate
    """
    import numpy as np

    A_half = np.zeros(sigma_coord.n_levels + 1)
    B_half = np.array(sigma_coord.sigma_half)

    return create_hybrid_coordinate(sigma_coord.n_levels, A_half, B_half, p_ref)


def pressure_from_hybrid(
    coord: HybridSigmaPressureCoordinate,
    p_s: jax.Array,
    full: bool = True,
) -> jax.Array:
    """Compute pressure at full or half levels from hybrid coordinate.

    p = A * p_ref + B * p_s

    Parameters
    ----------
    coord : HybridSigmaPressureCoordinate
    p_s : jax.Array
        Surface pressure, shape (...,).
    full : bool
        If True, return full-level pressures; if False, half-level.

    Returns
    -------
    jax.Array
        Pressure, shape (..., nlev) or (..., nlev+1).
    """
    A = coord.A_full if full else coord.A_half
    B = coord.B_full if full else coord.B_half
    return A * coord.p_ref + B * p_s[..., None]


def dp_from_hybrid(
    coord: HybridSigmaPressureCoordinate,
    p_s: jax.Array,
) -> jax.Array:
    """Compute layer pressure thickness from hybrid coordinate.

    dp_k = dA_k * p_ref + dB_k * p_s

    Parameters
    ----------
    coord : HybridSigmaPressureCoordinate
    p_s : jax.Array
        Surface pressure, shape (...,).

    Returns
    -------
    jax.Array
        Layer pressure thickness, shape (..., nlev).
    """
    return coord.dA * coord.p_ref + coord.dB * p_s[..., None]


def compute_geopotential_hybrid(
    T: jax.Array,
    p_s: jax.Array,
    coord: HybridSigmaPressureCoordinate,
    phis: jax.Array,
) -> jax.Array:
    """Compute geopotential using Simmons-Burridge for hybrid coordinates.

    Same algorithm as compute_geopotential but with pressure-dependent
    log ratios and alpha coefficients computed from the hybrid pressure.

    Parameters
    ----------
    T : jax.Array
        Temperature, shape (..., nlev).
    p_s : jax.Array
        Surface pressure, shape (...,).
    coord : HybridSigmaPressureCoordinate
    phis : jax.Array
        Surface geopotential, shape (...,).

    Returns
    -------
    jax.Array
        Geopotential at full levels, shape (..., nlev).
    """
    R_d = constants.R_d

    # Compute hybrid pressures at interfaces and full levels
    p_half = pressure_from_hybrid(coord, p_s, full=False)  # (..., nlev+1)
    p_full = pressure_from_hybrid(coord, p_s, full=True)   # (..., nlev)

    p_half_safe = jnp.clip(p_half, 1e-10, None)
    p_full_safe = jnp.clip(p_full, 1e-10, None)

    # Log ratios and exact Simmons-Burridge alpha — spatially dependent
    ln_ratio = jnp.log(p_half_safe[..., 1:] / p_half_safe[..., :-1])  # (..., nlev)
    dp = p_half_safe[..., 1:] - p_half_safe[..., :-1]
    alpha = 1.0 - (p_half_safe[..., :-1] / dp) * ln_ratio  # (..., nlev)

    # Geopotential thickness of each full layer
    dPhi = R_d * T * ln_ratio  # (..., nlev)

    # Cumulative sum from bottom: Phi_above[k] = sum from k to nlev-1
    dPhi_reversed = dPhi[..., ::-1]
    cumsum_reversed = jnp.cumsum(dPhi_reversed, axis=-1)
    cumsum = cumsum_reversed[..., ::-1]

    Phi_above = phis[..., None] + cumsum

    # Bottom interface of each layer
    Phi_below = jnp.concatenate(
        [Phi_above[..., 1:], phis[..., None]], axis=-1
    )

    Phi_full = Phi_below + alpha * R_d * T

    return Phi_full


def compute_mass_flux_hybrid(
    div_3d: jax.Array,
    p_s: jax.Array,
    coord: HybridSigmaPressureCoordinate,
) -> tuple[jax.Array, jax.Array]:
    """Compute vertical mass flux at half-levels for hybrid coordinates.

    Returns ``(mass_flux, D_total_p)``:

        F_{k+1/2} = B_{k+1/2} * D_total_p - cumsum(D_k * dp_k)[k]
        D_total_p = sum(D_k * dp_k)

    where dp_k is the layer pressure thickness.  ``D_total_p`` is
    returned so the caller (e.g. ``spectral_pe_tendencies`` step 8)
    can reuse it for the surface-pressure tendency without recomputing
    the column sum — saves one cross-level collective per RK3 stage
    under spectral level-sharding.

    Boundary conditions: F = 0 at top and surface.

    Parameters
    ----------
    div_3d : jax.Array
        Horizontal divergence at each level, shape (..., nlev).
    p_s : jax.Array
        Surface pressure, shape (...,).
    coord : HybridSigmaPressureCoordinate

    Returns
    -------
    mass_flux : jax.Array
        Mass flux at half-levels, shape (..., nlev+1). Units: Pa/s.
    D_total_p : jax.Array
        Column-integrated mass-weighted divergence, shape (..., 1).
    """
    dp = dp_from_hybrid(coord, p_s)  # (..., nlev)

    # Mass-weighted divergence
    div_dp = div_3d * dp  # (..., nlev)

    # Cumulative sum from top — its last entry is ``D_total_p``, so we
    # extract that rather than calling ``jnp.sum`` independently.  Under
    # level-sharding this drops the per-stage cross-level collective from
    # 2 (sum + cumsum) to 1 (cumsum reuses its own last index).
    cumsum_div = jnp.cumsum(div_dp, axis=-1)  # (..., nlev)
    D_total_p = cumsum_div[..., -1:]  # (..., 1)

    # Mass flux at interfaces 1..nlev
    # F_{k+1/2} = (B_{k+1/2} - B_top) / B_range * D_total_p - cumsum_k
    B_top = coord.B_half[0]
    frac_B = (coord.B_half[1:] - B_top) / coord.B_range  # (nlev,)
    mass_flux_inner = frac_B * D_total_p - cumsum_div  # (..., nlev)

    # Prepend top (F=0) and force bottom boundary (zero by construction).
    # Drop the (∼0) last element + pad with zeros on both ends in one
    # ``jnp.pad`` — replaces alloc-zeros + concatenate + scatter (3 HLO
    # ops) with slice + Pad (2 HLO ops).
    pad_axes = ((0, 0),) * (mass_flux_inner.ndim - 1)
    mass_flux = jnp.pad(mass_flux_inner[..., :-1], (*pad_axes, (1, 1)))

    return mass_flux, D_total_p


def vertical_advection_hybrid(
    field: jax.Array,
    mass_flux: jax.Array,
    p_s: jax.Array,
    coord: HybridSigmaPressureCoordinate,
) -> jax.Array:
    """Compute vertical advection in hybrid coordinates.

    Computes: -F_full * df/dp

    where F is the mass flux interpolated from half-levels to full levels
    and df/dp uses upwind differencing in pressure space.

    Parameters
    ----------
    field : jax.Array
        Field to advect, shape (..., nlev).
    mass_flux : jax.Array
        Mass flux at half-levels from compute_mass_flux_hybrid,
        shape (..., nlev+1).
    p_s : jax.Array
        Surface pressure, shape (...,).
    coord : HybridSigmaPressureCoordinate

    Returns
    -------
    jax.Array
        Vertical advection tendency, shape (..., nlev).
    """
    # Interpolate mass flux to full levels
    F_full = 0.5 * (mass_flux[..., :-1] + mass_flux[..., 1:])  # (..., nlev)

    # Pressure at full levels
    p_full = pressure_from_hybrid(coord, p_s, full=True)  # (..., nlev)
    dp_full = jnp.diff(p_full, axis=-1)  # (..., nlev-1)
    dp_full_safe = jnp.clip(jnp.abs(dp_full), 1e-10, None)

    # Vertical gradient df/dp with upwind differencing.  Pad along the
    # trailing axis instead of allocating fresh ``zeros`` and
    # concatenating — single Pad HLO op each.
    df = jnp.diff(field, axis=-1)  # (..., nlev-1)
    df_over = df / dp_full_safe
    pad_axes = ((0, 0),) * (df_over.ndim - 1)
    grad_bwd = jnp.pad(df_over, (*pad_axes, (1, 0)))
    grad_fwd = jnp.pad(df_over, (*pad_axes, (0, 1)))

    # Upwind: F > 0 (downward mass flux) → backward difference
    grad = jnp.where(F_full > 0, grad_bwd, grad_fwd)

    return -F_full * grad


def compute_omega_hybrid(
    mass_flux: jax.Array,
    p_s: jax.Array,
    dp_s_dt: jax.Array,
    coord: HybridSigmaPressureCoordinate,
) -> jax.Array:
    """Compute pressure velocity omega for hybrid coordinates.

    omega_k = B_full_k * dp_s/dt + F_k_full

    where F_k_full is the mass flux interpolated to full levels.

    Parameters
    ----------
    mass_flux : jax.Array
        Mass flux at half-levels, shape (..., nlev+1).
    p_s : jax.Array
        Surface pressure, shape (...,).
    dp_s_dt : jax.Array
        Surface pressure tendency, shape (...,).
    coord : HybridSigmaPressureCoordinate

    Returns
    -------
    jax.Array
        Pressure velocity at full levels, shape (..., nlev).
    """
    F_full = 0.5 * (mass_flux[..., :-1] + mass_flux[..., 1:])  # (..., nlev)

    omega = coord.B_full * dp_s_dt[..., None] + F_full

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
    # Use JAX default dtype (float64 when x64 is enabled, float32 otherwise)
    z_half = jnp.linspace(H, 0.0, n_levels + 1)
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

    # Jacobian: J = (H - z_s) / H, shape same as z_s
    jacobian = (H - z_s) / H

    # Physical z at full levels: z = z_s + z* * (H - z_s) / H
    # z_full is (nlev,).  z_s can be (6,n,n) or (nCells,) — broadcast generically.
    n_spatial = z_s.ndim  # 3 for cubed-sphere, 1 for MPAS
    z_full_bc = height_coord.z_full.reshape((1,) * n_spatial + (-1,))
    z_half_bc = height_coord.z_half.reshape((1,) * n_spatial + (-1,))
    z_full_3d = z_s[..., None] + z_full_bc * jacobian[..., None]
    z_half_3d = z_s[..., None] + z_half_bc * jacobian[..., None]

    return TerrainMetric(
        z_s=z_s,
        jacobian=jacobian,
        z_full_3d=z_full_3d,
        z_half_3d=z_half_3d,
    )
