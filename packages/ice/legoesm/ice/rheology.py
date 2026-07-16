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

Faithfulness
------------
The constitutive closed forms are pinned by
``tests/ice/unit/test_ice_rheology_faithful.py`` against an independent scalar
oracle (rel 1e-9):

  * ``ice_strength`` (Hibler 1979): P = P* h exp(-C (1 - A)).
  * ``delta_deformation`` (Hibler 1979): Delta =
    sqrt(max((e11+e22)^2 + ((e11-e22)^2 + 4 e12^2)/e^2, Delta_min^2)).
  * ``vp_stress`` (Hibler 1979 / Hunke-Dukowicz 1997): zeta = P/(2 Delta),
    eta = zeta/e^2; sigma_11 = 2 eta e11 + (zeta-eta)(e11+e22) - P/2, etc.  The
    pin includes the DEFINING yield-curve identity — the stress lies exactly on
    the ellipse ((sigma_I+P/2)/(P/2))^2 + (sigma_II e/(P/2))^2 = 1 when Delta is
    un-regularized — and the sigma = -P/2 I rest state at zero strain.
  * ``evp_stress_update`` (Hunke-Dukowicz 1997): sigma_new =
    (sigma + E sigma_VP)/(1 + E), E = 1/(2 T_evp N_evp); iterating to the
    sigma_VP fixed point.
  * ``mevp_stress_update`` (Bouillon 2013 / Kimmritz 2015): sigma^(p+1) =
    (1 - 1/alpha) sigma^p + (1/alpha) sigma_VP; the alpha < 1 anti-relaxation
    guard raises.

The P*, C, e, Delta_min, T_evp, N_evp, alpha coefficients are transcribed as
independent oracle literals and canaried against the ``SeaIceConfig`` defaults
(config == literal == value).  DEPARTURE / guard: ``delta_deformation`` floors
the sqrt ARGUMENT at Delta_min^2 (not the result at Delta_min) — forward-
identical but AD-safe, so the VP/EVP/mEVP stress gradients stay finite at the
zero-strain rest state (where sqrt'(0) would otherwise poison the VJP); this is
pinned by a dedicated zero-strain gradient test.

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

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo_vector
from legoesm.grids.halo_latlon import pad_halo_vector_latlon
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.ice.config import SeaIceConfig

# Canonical EVP rheology defaults live on SeaIceConfig (single source of truth
# callers pass in); kwarg signatures default to these so no empirical literal is
# buried in a signature.
_RHEO_DEFAULTS = SeaIceConfig()


def _is_latlon_grid(grid) -> bool:
    return isinstance(grid, LatLonGrid)


def _is_voronoi_mesh(grid) -> bool:
    return isinstance(grid, VoronoiMesh)


def cell_gradient_voronoi(
    f_cell: jnp.ndarray,
    mesh: VoronoiMesh,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cell-centered Green-Gauss gradient on a Voronoi mesh.

    ∂f/∂x|_C = (1/A_C) Σ_e f_e · n_e_x · L_e · s_{e,C}
    ∂f/∂y|_C = (1/A_C) Σ_e f_e · n_e_y · L_e · s_{e,C}

    where ``f_e = 0.5 · (f_{c1} + f_{c2})`` is the edge-averaged
    value, ``n_e = (cos(angleEdge), sin(angleEdge))`` is the
    edge-normal direction (oriented c1→c2 in ``cellsOnEdge``),
    ``L_e = dvEdge`` is the edge length, ``s_{e,C} =
    edgeSignOnCell`` is +1 if the edge-normal points out of C,
    −1 otherwise, and ``A_C = areaCell``.

    Boundary edges (single-cell) use ``f_e = f_{c1}`` (one-sided).

    Returns
    -------
    df_dx, df_dy : array ``(nCells,)``
    """
    eoc = mesh.edgesOnCell                 # (maxEdges, nCells)
    sign = mesh.edgeSignOnCell             # (maxEdges, nCells)
    mask = (eoc >= 0).astype(f_cell.dtype)
    eoc_safe = jnp.maximum(eoc, 0)

    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    c1_safe = jnp.maximum(c1, 0)
    c2_safe = jnp.maximum(c2, 0)
    interior = c2 >= 0
    f_edge = jnp.where(
        interior, 0.5 * (f_cell[c1_safe] + f_cell[c2_safe]), f_cell[c1_safe],
    )

    cos_a = jnp.cos(mesh.angleEdge)
    sin_a = jnp.sin(mesh.angleEdge)
    dv = mesh.dvEdge

    f_e_x = f_edge * cos_a * dv
    f_e_y = f_edge * sin_a * dv

    f_e_x_gathered = f_e_x[eoc_safe]       # (maxEdges, nCells)
    f_e_y_gathered = f_e_y[eoc_safe]

    flux_x = sign * f_e_x_gathered * mask
    flux_y = sign * f_e_y_gathered * mask

    df_dx = jnp.sum(flux_x, axis=0) / mesh.areaCell
    df_dy = jnp.sum(flux_y, axis=0) / mesh.areaCell
    return df_dx, df_dy


def _strain_rates_voronoi(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    mesh: VoronoiMesh,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Strain rate tensor on a Voronoi mesh via Green-Gauss cell gradients.

    Cell-centered (u, v) on the MPAS mesh use the same local
    east-north basis as cubed-sphere / lat-lon.  The Green-Gauss
    gradient is a first-order-accurate cell-centered estimator
    that is exact for linear fields.

    Returns
    -------
    eps_11, eps_22, eps_12 : array ``(nCells,)``
    """
    du_dx, du_dy = cell_gradient_voronoi(u_ice, mesh)
    dv_dx, dv_dy = cell_gradient_voronoi(v_ice, mesh)
    eps_11 = du_dx
    eps_22 = dv_dy
    eps_12 = 0.5 * (du_dy + dv_dx)
    return eps_11, eps_22, eps_12


# ==============================================================================
# Ice strength
# ==============================================================================

def ice_strength(
    h: jnp.ndarray,
    A: jnp.ndarray,
    P_star: float = _RHEO_DEFAULTS.P_star,
    C_strength: float = _RHEO_DEFAULTS.C_strength,
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

def _strain_rates_cubed_sphere(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    grid: CubedSphereGrid,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Strain rate tensor on the cubed sphere (centered FD, halo-2 vector exchange)."""
    u_pad, v_pad = pad_halo_vector(
        u_ice, v_ice,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )
    du_dx = (u_pad[:, 2:, 1:-1] - u_pad[:, :-2, 1:-1]) / grid.dx
    du_dy = (u_pad[:, 1:-1, 2:] - u_pad[:, 1:-1, :-2]) / grid.dy
    dv_dx = (v_pad[:, 2:, 1:-1] - v_pad[:, :-2, 1:-1]) / grid.dx
    dv_dy = (v_pad[:, 1:-1, 2:] - v_pad[:, 1:-1, :-2]) / grid.dy
    eps_11 = du_dx
    eps_22 = dv_dy
    eps_12 = 0.5 * (du_dy + dv_dx)
    return eps_11, eps_22, eps_12


def _strain_rates_latlon(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    grid: LatLonGrid,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Strain rate tensor on a lat-lon A-grid with spherical metric (F10).

    Centered finite differences in (x = r·cosθ·dλ, y = r·dθ)
    coordinates.  ``grid.dx`` is the 2-cell zonal distance
    ``2 · r · cosθ · dλ`` and ``grid.dy`` is the 2-cell meridional
    distance ``2 · r · dθ``, so the centered differences give the
    ``(1/h)∂/∂`` velocity-gradient terms

        du_dx = (u[i, j+1] - u[i, j-1]) / dx(i)  = (1/(r cosθ)) ∂u/∂λ
        dv_dy = (v[i+1, j] - v[i-1, j]) / dy(i)  = (1/r) ∂v/∂θ        ...

    The rate-of-strain tensor on a sphere (orthogonal curvilinear coords
    h1 = r cosθ, h2 = r) additionally carries spherical-METRIC terms from the
    Christoffel symbols (``∂h1/∂θ = -r sinθ``):

        eps_11 = du_dx - (v · tanθ) / r
        eps_22 = dv_dy                       (no metric term; ∂h2/∂λ = 0)
        eps_12 = 0.5 (du_dy + dv_dx) + (u · tanθ) / (2 r)

    These were previously omitted (valid only at low latitude); including them
    makes divergence / shear correct away from the equator (F10).

    The metric coefficient uses the EXACT ``tanθ`` (MITgcm SEAICE
    ``k2 = -tanφ/a`` convention) on every cell-center row — it is NOT clipped,
    so 1° and finer grids (centers at ±89.5° and beyond) get the correct
    polar value rather than a capped one.  ``create_latlon_grid`` places cell
    centers strictly inside the poles, so ``tanθ`` is finite; ``step_sea_ice``
    rejects (with a clear error) any lat-lon grid whose rows reach the exact
    pole (``cosθ → 0``) under EVP/mEVP.  NOTE: this metric correction does not
    resolve the missing tripolar fold; very-high-latitude lat-lon EVP should
    still be cross-checked against the cubed-sphere / MPAS backends (visual
    ice-drift verification recommended; see CLAUDE.md spatial-artifact guidance).

    Polar rows: the halo padder folds across the pole; centered differences
    then see physical neighbours.

    Parameters
    ----------
    u_ice, v_ice : array ``(n_lat, n_lon)``
        Ice velocity components [m/s].
    grid : LatLonGrid

    Returns
    -------
    eps_11, eps_22, eps_12 : arrays ``(n_lat, n_lon)``
        Strain rate tensor components [1/s].
    """
    u_pad, v_pad = pad_halo_vector_latlon(u_ice, v_ice, halo=1)
    dx = grid.dx                       # (n_lat, n_lon)
    dy = grid.dy[:, None]              # broadcast to (n_lat, 1)

    du_dx = (u_pad[1:-1, 2:] - u_pad[1:-1, :-2]) / dx
    du_dy = (u_pad[2:, 1:-1] - u_pad[:-2, 1:-1]) / dy
    dv_dx = (v_pad[1:-1, 2:] - v_pad[1:-1, :-2]) / dx
    dv_dy = (v_pad[2:, 1:-1] - v_pad[:-2, 1:-1]) / dy

    # Spherical-metric coefficient tanθ / r = sinθ / (r cosθ).  Computed via
    # sin/cos with |cosθ| floored at 1e-12 ONLY at the exact pole: real
    # cell-centered grids have cosθ >= ~0.009 even at 89.5°, so the floor never
    # binds and the coefficient is the EXACT tanθ; it only prevents inf/NaN if a
    # degenerate grid places a row exactly at ±90° (a NaN-safety floor, not a
    # tunable, and NOT the rejected hard clip — it does not cap real-grid rows).
    cos_lat = jnp.cos(grid.lat)
    cos_safe = jnp.where(jnp.abs(cos_lat) < 1e-12, 1e-12, cos_lat)
    metric = (jnp.sin(grid.lat) / cos_safe / grid.radius)[:, None]   # (n_lat, 1)

    eps_11 = du_dx - v_ice * metric
    eps_22 = dv_dy
    eps_12 = 0.5 * (du_dy + dv_dx) + 0.5 * u_ice * metric
    return eps_11, eps_22, eps_12


def strain_rates(
    u_ice: jnp.ndarray,
    v_ice: jnp.ndarray,
    grid,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the symmetric strain rate tensor.

    Grid-agnostic dispatcher:
        * ``CubedSphereGrid`` → cubed-sphere FD with halo-2 vector
          exchange + cross-face rotation.
        * ``LatLonGrid`` → A-grid centered FD in local-Cartesian
          (r·cosθ·dλ, r·dθ) coordinates.
        * ``VoronoiMesh`` → Green-Gauss cell-centered gradient via
          existing MPAS connectivity (``edgesOnCell``,
          ``edgeSignOnCell``, ``angleEdge``).

    eps_11 = du/dx, eps_22 = dv/dy, eps_12 = 0.5·(du/dy + dv/dx).
    """
    if _is_voronoi_mesh(grid):
        return _strain_rates_voronoi(u_ice, v_ice, grid)
    if _is_latlon_grid(grid):
        return _strain_rates_latlon(u_ice, v_ice, grid)
    if not isinstance(grid, CubedSphereGrid):
        raise TypeError(
            f"unsupported grid {type(grid).__name__} for strain_rates"
        )
    return _strain_rates_cubed_sphere(u_ice, v_ice, grid)


# ==============================================================================
# Deformation invariant
# ==============================================================================

def delta_deformation(
    eps_11: jnp.ndarray,
    eps_22: jnp.ndarray,
    eps_12: jnp.ndarray,
    e_yield: float = _RHEO_DEFAULTS.e_yield,
    Delta_min: float = _RHEO_DEFAULTS.Delta_min,
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
    # Floor the sqrt *argument* at Delta_min**2 (not the result at Delta_min).
    # Forward-identical — sqrt(max(Delta_sq, Delta_min**2)) == max(sqrt(Delta_sq),
    # Delta_min) — but AD-safe: the argument is >= Delta_min**2 > 0 so sqrt is
    # never differentiated at 0.  The previous max(sqrt(max(Delta_sq, 0)),
    # Delta_min) form returned a NaN gradient at zero strain (rest state / cold
    # start, eps_ij == 0): sqrt'(0) = inf and the outer max routes a zero
    # selector into it -> 0*inf = NaN, poisoning every VP/EVP/mEVP stress
    # gradient on the first backward pass.
    return jnp.sqrt(jnp.maximum(Delta_sq, Delta_min ** 2))


# ==============================================================================
# VP stress tensor
# ==============================================================================

def vp_stress(
    eps_11: jnp.ndarray,
    eps_22: jnp.ndarray,
    eps_12: jnp.ndarray,
    P: jnp.ndarray,
    Delta: jnp.ndarray,
    e_yield: float = _RHEO_DEFAULTS.e_yield,
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
    Delta_min: float = _RHEO_DEFAULTS.Delta_min,
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

    **Per-dynamic-step relaxation (Hunke & Dukowicz E_evp behaviour).**
    Accumulating this subcycle map over ``N_evp`` subcycles gives a
    total relaxation of ``1 − (1 + E_factor)**(-N_evp) ≈ 1 − exp(−1 /
    (2·T_evp))`` toward the VP target — which is *independent of*
    ``N_evp`` (more subcycles integrate the SAME elastic-damping ODE
    more accurately, they do not relax further).  With the CICE default
    ``T_evp = 0.36`` this is ≈ 0.75 per dynamic step, so the stress does
    NOT reach the full VP/plastic solution in a single ``evp_solver``
    call: it converges to ``σ = −P/2·I`` (rest state) over several
    dynamic steps as ``σ`` is carried forward.  This is the intended
    elastic-memory design, not under-convergence.  ``T_evp`` is the
    damping ratio ``T_damp / dt_dyn``; smaller ``T_evp`` → faster
    per-step relaxation toward plastic but stiffer elastic waves.

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
    Delta_min: float = _RHEO_DEFAULTS.Delta_min,
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
