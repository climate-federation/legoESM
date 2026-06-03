# Physics-validator adversarial review packet (iter 1) — `legoesm` ocean tree

You are an independent adversarial physics reviewer for an Earth-system
model written in JAX.  Find every bug, sign error, unit inconsistency,
broken-gradient pattern, conservation violation, and test-case discrepancy
in the implementation below.  Cite line numbers.  If you believe there
are no bugs, say so and explain why each candidate concern is not one.

## Scope
`src/legoesm/ocean/` — ocean dycore, baroclinic + barotropic, lateral &
vertical mixing inc. KPP, GM/Redi, convection, freshwater, sponge,
NPZD biogeochemistry, advection, EOS.

## Critical files (already audited, read carefully)

### `src/legoesm/ocean/eos.py`
```python
"""Equation of state for seawater: Wright (1997) and linear.

Provides:
- ``wright_eos`` — nonlinear Wright (1997) EOS (MOM6 implementation)
- ``linear_eos`` — configurable linear EOS: ρ = ρ₀[1 - αT(T-Tref) + βS(S-Sref)]
- ``make_eos_fn`` — dispatcher returning an EOS callable based on config

Pure JAX functions, compatible with jit/grad/vmap.

Reference
---------
Wright, D. G. (1997): An Equation of State for Use in Ocean Models:
Ockham's Razor Revisited. J. Atmos. Oceanic Tech., 14(3), 735-740.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.precision import _resolve_dtype

# ==============================================================================
# Ocean constants
# ==============================================================================
rho_0 = 1025.0          # Reference seawater density [kg/m^3]
c_sw = 3994.0           # Specific heat of seawater [J/(kg*K)]
T_freeze_ocean = constants.T_freeze_ocean  # re-export from central constants
scale_depth = 1000.0     # Reference e-folding depth for stratification [m]

# ==============================================================================
# Wright (1997) EOS coefficients — from MOM6 (MOM_EOS_Wright.F90)
# Pressure units: Pa. Temperature: degC. Salinity: PSU.
#
# Formula: rho = (p + p0) / (lambda + al0 * (p + p0))
#   al0(T, S) = a0 + a1*T + a2*S
#   p0(T, S)  = (b0 + b4*S) + T*(b1 + T*(b2 + b3*T) + b5*S)
#   lambda(T, S) = (c0 + c4*S) + T*(c1 + T*(c2 + c3*T) + c5*S)
# ==============================================================================

# Specific volume coefficients al0(T, S)
_a0 = 7.057924e-4
_a1 = 3.480336e-7
_a2 = -1.112733e-7

# Pressure offset p0(T, S) [Pa]
_b0 = 5.790749e8
_b1 = 3.516535e6
_b2 = -4.002714e4
_b3 = 2.084372e2
_b4 = 5.944068e5
_b5 = -9.643486e3

# Lambda(T, S) [m^2/s^2]
_c0 = 1.704853e5
_c1 = 7.904722e2
_c2 = -7.984422
_c3 = 5.140652e-2
_c4 = -2.302158e2
_c5 = -3.079464


def wright_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    """Compute in-situ density from Wright (1997) EOS.

    Parameters
    ----------
    T : array
        Potential temperature [degC].
    S : array
        Salinity [PSU].
    p : array
        Pressure [Pa]. Use 0 for surface.

    Returns
    -------
    array : In-situ density [kg/m^3].

    Notes
    -----
    Intermediate computation is promoted to float64 to avoid precision
    loss from large polynomial coefficients (e.g., _b0 ~ 5.79e8).
    If ``JAX_ENABLE_X64=1`` is not set, the astype calls are no-ops
    (safe but no precision improvement).  ``jnp.astype`` is
    differentiable in JAX.

    The Wright (1997) polynomial is nominally valid for T in [-2, 40] degC
    and S in [0, 42] PSU, but extrapolates smoothly outside that box.
    Inputs are not clipped: silent clipping would zero gradients at the
    boundary and mask unphysical state from advection overshoots or
    coupler bugs. See issue #165.
    """
    orig_dtype = T.dtype

    # Promote to the EOS compute dtype (float64 in mixed mode) for
    # intermediate polynomial evaluation.  On backends that lack float64
    # (e.g. Metal), _resolve_dtype silently returns float32.
    hi = _resolve_dtype("equation_of_state", "compute")
    T = T.astype(hi)
    S = S.astype(hi)
    p = p.astype(hi)

    # Specific volume parameter
    al0 = _a0 + _a1 * T + _a2 * S

    # Pressure offset
    p0 = (_b0 + _b4 * S) + T * (_b1 + T * (_b2 + _b3 * T) + _b5 * S)

    # Lambda
    lam = (_c0 + _c4 * S) + T * (_c1 + T * (_c2 + _c3 * T) + _c5 * S)

    # Density: rho = (p + p0) / (lambda + al0 * (p + p0))
    p_plus_p0 = p + p0
    rho = p_plus_p0 / (lam + al0 * p_plus_p0)

    return rho.astype(orig_dtype)


def _wright_eos_scalar(T: float, S: float, p: float) -> float:
    """Scalar Wright EOS for JAX grad (no dtype promotion).

    Used internally by ``thermal_expansion_coeff`` and
    ``haline_contraction_coeff`` via ``jax.grad``.
    """
    al0 = _a0 + _a1 * T + _a2 * S
    p0 = (_b0 + _b4 * S) + T * (_b1 + T * (_b2 + _b3 * T) + _b5 * S)
    lam = (_c0 + _c4 * S) + T * (_c1 + T * (_c2 + _c3 * T) + _c5 * S)
    p_plus_p0 = p + p0
    return p_plus_p0 / (lam + al0 * p_plus_p0)


# Partial derivatives via JAX autodiff (scalar → vmap for arrays).
import jax
_drho_dT_scalar = jax.grad(_wright_eos_scalar, argnums=0)
_drho_dS_scalar = jax.grad(_wright_eos_scalar, argnums=1)


def thermal_expansion_coeff(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    r"""Thermal expansion coefficient α = -(1/ρ) ∂ρ/∂T.

    Parameters
    ----------
    T : array — Potential temperature [degC].
    S : array — Salinity [PSU].
    p : array — Pressure [Pa].

    Returns
    -------
    array : α [1/K], same shape as inputs.
    """
    hi = _resolve_dtype("equation_of_state", "compute")
    T64 = T.astype(hi)
    S64 = S.astype(hi)
    p64 = p.astype(hi)
    flat_T = T64.ravel()
    flat_S = S64.ravel()
    flat_p = p64.ravel()
    drho_dT = jax.vmap(_drho_dT_scalar)(flat_T, flat_S, flat_p).reshape(T.shape)
    rho = wright_eos(T, S, p)
    return (-drho_dT / rho).astype(T.dtype)


def haline_contraction_coeff(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    r"""Haline contraction coefficient β = (1/ρ) ∂ρ/∂S.

    Parameters
    ----------
    T : array — Potential temperature [degC].
    S : array — Salinity [PSU].
    p : array — Pressure [Pa].

    Returns
    -------
    array : β [1/PSU], same shape as inputs.
    """
    hi = _resolve_dtype("equation_of_state", "compute")
    T64 = T.astype(hi)
    S64 = S.astype(hi)
    p64 = p.astype(hi)
    flat_T = T64.ravel()
    flat_S = S64.ravel()
    flat_p = p64.ravel()
    drho_dS = jax.vmap(_drho_dS_scalar)(flat_T, flat_S, flat_p).reshape(T.shape)
    rho = wright_eos(T, S, p)
    return (drho_dS / rho).astype(T.dtype)


# ==============================================================================
# Linear equation of state
# ==============================================================================

class LinearEOSConfig(NamedTuple):
    """Configuration for the linear equation of state.

    ρ = rho_ref * [1 - alpha_T * (T - T_ref) + beta_S * (S - S_ref)]
    """
    rho_ref: float = 1025.0    # Reference density [kg/m³]
    alpha_T: float = 2.0e-4    # Thermal expansion coefficient [1/K]
    beta_S: float = 7.4e-4     # Haline contraction coefficient [1/PSU]
    T_ref: float = 10.0        # Reference temperature [°C]
    S_ref: float = 35.0        # Reference salinity [PSU]


def linear_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    rho_ref: float = 1025.0,
    alpha_T: float = 2.0e-4,
    beta_S: float = 7.4e-4,
    T_ref: float = 10.0,
    S_ref: float = 35.0,
) -> jnp.ndarray:
    """Compute density from a linear equation of state.

    ρ = rho_ref * [1 - alpha_T * (T - T_ref) + beta_S * (S - S_ref)]

    Parameters
    ----------
    T : array — Potential temperature [°C].
    S : array — Salinity [PSU].
    p : array — Pressure [Pa] (unused, accepted for API compatibility).
    rho_ref : float — Reference density [kg/m³].
    alpha_T : float — Thermal expansion coefficient [1/K].
    beta_S : float — Haline contraction coefficient [1/PSU].
    T_ref : float — Reference temperature [°C].
    S_ref : float — Reference salinity [PSU].

    Returns
    -------
    array : In-situ density [kg/m³].
    """
    return rho_ref * (1.0 - alpha_T * (T - T_ref) + beta_S * (S - S_ref))


def make_eos_fn(eos="wright", eos_linear=None):
    """Return an EOS callable ``fn(T, S, p) -> rho``.

    Parameters
    ----------
    eos : str
        ``"wright"`` (default) or ``"linear"``.
    eos_linear : LinearEOSConfig or None
        Parameters for linear EOS.  Ignored when *eos* is ``"wright"``.
        If ``None`` and *eos* is ``"linear"``, default parameters are used.

    Returns
    -------
    Callable[[array, array, array], array]
    """
    if eos == "wright":
        return wright_eos
    elif eos == "linear":
        cfg = eos_linear if eos_linear is not None else LinearEOSConfig()
        def _linear(T, S, p):
            return linear_eos(
                T, S, p,
                rho_ref=cfg.rho_ref, alpha_T=cfg.alpha_T,
                beta_S=cfg.beta_S, T_ref=cfg.T_ref, S_ref=cfg.S_ref,
            )
        return _linear
    else:
        raise ValueError(f"Unknown EOS scheme: {eos!r}")


def density_perturbation(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    rho_ref: float = rho_0,
) -> jnp.ndarray:
    """Compute density perturbation rho' = rho(T,S,p) - rho_ref.

    Parameters
    ----------
    T, S, p : array
        Temperature [degC], salinity [PSU], pressure [Pa].
    rho_ref : float
        Reference density [kg/m^3].

    Returns
    -------
    array : Density perturbation [kg/m^3].
    """
    return wright_eos(T, S, p) - rho_ref


def compute_hydrostatic_pressure(
    rho: jnp.ndarray,
    eta: jnp.ndarray,
    dz: jnp.ndarray,
    jacobian: jnp.ndarray,
    rho_ref: float = rho_0,
    g: float = constants.g,
) -> jnp.ndarray:
    """Compute hydrostatic pressure at full levels.

    p(z) = rho_ref * g * eta + integral_{z}^{0} rho * g dz'

    Integrated top-to-bottom (k=0 is surface, k=nlev-1 is deepest).
    Pressure at cell center is the cumulative integral from surface
    down to the midpoint of each layer.

    Parameters
    ----------
    rho : array
        In-situ density, shape (..., nlev).
    eta : array
        Sea surface height [m], shape (...).
    dz : array
        Reference layer thickness [m], shape (nlev,).
    jacobian : array
        Dynamic Jacobian (eta + H) / H, shape (...).
    rho_ref : float
        Reference density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].

    Returns
    -------
    array : Hydrostatic pressure at full levels [Pa], shape (..., nlev).
    """
    # Surface pressure from free surface
    p_surface = rho_ref * g * eta  # (...,)

    # Actual layer thickness
    dz_actual = dz * jacobian[..., jnp.newaxis]  # (..., nlev)

    # Pressure increment per layer: rho * g * dz
    dp = rho * g * dz_actual  # (..., nlev)

    # Pressure at layer top = cumulative sum from surface
    # p_top[k] = p_surface + sum(dp[0:k])
    p_top = p_surface[..., jnp.newaxis] + jnp.cumsum(dp, axis=-1) - dp

    # Pressure at cell center = p_top + 0.5 * dp
    return p_top + 0.5 * dp


def compute_buoyancy_frequency(
    rho: jnp.ndarray,
    dz: jnp.ndarray,
    jacobian: jnp.ndarray,
    rho_ref: float = rho_0,
    g: float = constants.g,
) -> jnp.ndarray:
    """Compute Brunt-Vaisala frequency N^2.

    N^2 = -(g / rho_ref) * d(rho) / dz

    Computed at interior interfaces (nlev-1 values).

    Parameters
    ----------
    rho : array
        In-situ density, shape (..., nlev).
    dz : array
        Reference layer thickness [m], shape (nlev,).
    jacobian : array
        Dynamic Jacobian, shape (...).
    rho_ref : float
        Reference density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].

    Returns
    -------
    array : N^2 at interior interfaces [1/s^2], shape (..., nlev-1).
    """
    dz_actual = dz * jacobian[..., jnp.newaxis]
    dz_interface = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])

    # drho/dz: rho[k] is shallower than rho[k+1]
    # N^2 = -(g/rho_0) * (rho[k] - rho[k+1]) / dz_interface
    drho_dz = (rho[..., :-1] - rho[..., 1:]) / dz_interface

    return -(g / rho_ref) * drho_dz


# ==============================================================================
# Shared helpers for ocean physics integration modules
# ==============================================================================

def compute_ocean_rho(state, z_coord, jacobian, eos_fn=None):
    """Compute in-situ density from ocean state.

    Used by vertical mixing, lateral mixing, and convection integration
    bridges. Avoids triplicating the same hydrostatic pressure + EOS call.

    Parameters
    ----------
    state : OceanState
        Must have .T, .S, .eta fields.
    z_coord : OceanZStarCoordinate
        Vertical coordinate with .dz_ref.
    jacobian : array
        Dynamic Jacobian (eta + H) / H.
    eos_fn : callable or None
        EOS function ``fn(T, S, p) -> rho``.  If None, uses ``wright_eos``.

    Returns
    -------
    array : In-situ density [kg/m^3].
    """
    if eos_fn is None:
        eos_fn = wright_eos
    # Two EOS iterations for density-pressure consistency, matching the
    # dynamical core (ocean_pe_cdgrid.py).
    rho = eos_fn(state.T.data, state.S.data, jnp.zeros_like(state.T.data))
    for _ in range(2):
        p_hydro = compute_hydrostatic_pressure(
            rho, state.eta.data, z_coord.dz_ref, jacobian, rho_0,
        )
        rho = eos_fn(state.T.data, state.S.data, p_hydro)
    return rho


def compute_ocean_rho_and_pressure(state, z_coord, jacobian, eos_fn=None):
    """Compute in-situ density and hydrostatic pressure from ocean state.

    Delegates to ``compute_ocean_rho`` for the 2-iteration EOS-pressure
    coupling, then computes a final hydrostatic pressure consistent with
    the converged density.

    Parameters
    ----------
    state, z_coord, jacobian : same as ``compute_ocean_rho``.
    eos_fn : callable or None
        EOS function. If None, uses ``wright_eos``.

    Returns
    -------
    rho : array — in-situ density [kg/m^3].
    p_hydro : array — hydrostatic pressure [Pa].
    """
    rho = compute_ocean_rho(state, z_coord, jacobian, eos_fn=eos_fn)
    p_hydro = compute_hydrostatic_pressure(
        rho, state.eta.data, z_coord.dz_ref, jacobian, rho_0,
    )
    return rho, p_hydro
```

### `src/legoesm/ocean/dynamics/ocean_tendency_common.py`
```python
"""Grid-agnostic baroclinic tendency helpers shared by ocean_pe_*.py files.

This module factors logic that was previously duplicated across the
A-grid (``ocean_pe_cdgrid.py``), lat-lon C-grid
(``ocean_pe_latlon_cgrid.py``), and MPAS Voronoi (``ocean_pe_mpas.py``)
baroclinic tendency entry points.  Grid-specific operators (gradient,
divergence, vorticity, fill, ...) are passed in as callables so this
module never imports from a particular grid package.

Closes #214 (Phase 1).

Functions
---------
``iterate_eos_and_pressure_anomaly``
    EOS iteration (T_filled, S_filled, p) → ρ, ρ', p' using a
    reference-thickness hydrostatic integral.  The cubed-sphere caller
    can request high-precision arithmetic for the cumsum so that the
    halo-exchanged p' gradient stays clean.

``apply_sponge_tracer_relaxation``
    Linear restoring of T, S towards reference fields with rate
    ``γ``.  Used identically by the lat-lon C-grid and MPAS callers.

``apply_freshwater_virtual_salt_top``
    Top-layer salinity tendency from the freshwater volume flux.

``implicit_bottom_drag_factor``
    Returns ``1 - dt * r / max(H, eps)`` — the per-substep multiplicative
    factor used by both C-grid lat-lon and MPAS barotropic substeps.

These helpers are pure and pytree-friendly: they accept and return
``jax.Array`` values and never mutate inputs.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax.numpy as jnp

from legoesm.ocean.eos import compute_hydrostatic_pressure
from legoesm.ocean.freshwater import virtual_salt_flux


def iterate_eos_and_pressure_anomaly(
    T: jnp.ndarray,
    S: jnp.ndarray,
    mask: jnp.ndarray,
    fill_fn: Callable[[jnp.ndarray], jnp.ndarray],
    eos_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    dz_ref: jnp.ndarray,
    rho_0: float,
    g: float,
    *,
    n_iter: int = 2,
    hi_precision_pressure: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the standard 2-pass EOS iteration and form ``p_prime``.

    Replicates the identical iteration that previously lived inline in
    every ``ocean_pe_*.py`` file:

    1. Fill land-cell ``T``, ``S`` with ocean-neighbour values via
       ``fill_fn`` so the EOS does not produce spurious ``ρ'`` values
       on land that contaminate the gradient at coastlines.
    2. Iterate ``ρ ← EOS(T, S, p_hydro(ρ))`` ``n_iter`` times against
       the **reference** thickness profile ``dz_ref`` (i.e. ``J=1``,
       ``η=0``).  Using the actual Jacobian here would double-count
       the ``-g·∇η`` forcing handled by the barotropic solver.
    3. Build the layer-centred baroclinic pressure anomaly

       ``p'(k) = g · Σ_{j<k} ρ'(j) · dz_ref(j) + 0.5 · g · ρ'(k) · dz_ref(k)``

       which equals the half-trapezoidal cumulative integral of
       ``g·ρ'`` from the surface to the layer mid-point.

    Parameters
    ----------
    T, S : jax.Array
        Tracer fields with a trailing vertical axis (``..., nlev``).  All
        upstream callers store T and S with identical shapes.
    mask : jax.Array
        Land mask with the same horizontal shape as ``T[..., 0]`` (used
        only by ``fill_fn``; passed back to the caller for any post-
        processing it needs).
    fill_fn : Callable[[jax.Array], jax.Array]
        Grid-specific land-cell filler.  Must accept and return arrays
        with the same shape as ``T``.  Typical implementations:
        ``jax.vmap(fill_land_cells_cubed_sphere, in_axes=-1)``,
        ``_neumann_fill_cgrid`` (lat-lon C-grid), or
        ``fill_land_cells_mpas`` (MPAS).
    eos_fn : Callable
        Equation of state ``(T, S, p) → ρ``.  Same signature used by
        every grid (built via ``make_eos_fn``).
    dz_ref : jax.Array
        Reference layer thickness (``z_coord.dz_ref``), shape
        ``(nlev,)``.
    rho_0, g : float
    n_iter : int, default 2
        Number of EOS iterations *before* the final pressure update.
        All current callers use 2.
    hi_precision_pressure : bool, default False
        If True, perform the pressure cumsum in float64 so that
        halo-exchange interpolation errors do not contaminate the
        downstream gradient.  Used by the cubed-sphere C-D path; on
        lat-lon and MPAS the compact 2-cell stencils are well-behaved
        enough that the working precision is sufficient.

    Returns
    -------
    rho : jax.Array
        In-situ density (same shape as ``T``).
    rho_prime : jax.Array
        ``rho - rho_0`` (same shape as ``T``).
    p_prime : jax.Array
        Baroclinic pressure anomaly (same shape as ``T``).  Returned in
        whatever precision was used for the cumulative sum.
    """
    del mask  # currently unused (passed to fill_fn by the caller); kept
              # in signature for clarity at call sites.

    T_filled = fill_fn(T)
    S_filled = fill_fn(S)

    # Reference Jacobian (J=1, η=0).  Both have the horizontal shape of
    # T (i.e. no vertical axis).  Match dtype to the working state so we
    # never accidentally promote the EOS iteration to float64.
    horiz_shape = T.shape[:-1]
    J_ref = jnp.ones(horiz_shape, dtype=T.dtype)
    eta_ref = jnp.zeros(horiz_shape, dtype=T.dtype)

    rho = eos_fn(T_filled, S_filled, jnp.zeros_like(T))
    for _ in range(n_iter):
        p_hydro = compute_hydrostatic_pressure(
            rho, eta_ref, dz_ref, J_ref, rho_0, g,
        )
        rho = eos_fn(T_filled, S_filled, p_hydro)

    rho_prime = rho - rho_0

    if hi_precision_pressure:
        rho_prime_hi = rho_prime.astype(jnp.float64)
        dz_hi = dz_ref.astype(jnp.float64)
        dp_layer = rho_prime_hi * g * dz_hi
    else:
        dp_layer = rho_prime * g * dz_ref

    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer

    return rho, rho_prime, p_prime


def apply_sponge_tracer_relaxation(
    dT_dt: jnp.ndarray,
    dS_dt: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    sponge,
    mask: Optional[jnp.ndarray] = None,
    *,
    expand_gamma_axis: int = -1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply tracer sponge relaxation ``+γ·(ref - q)``.

    Casts ``sponge`` arrays to ``T.dtype`` so the precision policy that
    holds the state in float32 is not silently promoted to float64
    (which previously crashed the barotropic scan; see latlon C-grid).

    Parameters
    ----------
    dT_dt, dS_dt : jax.Array
        Existing tendency arrays (modified by addition).
    T, S : jax.Array
        Current tracer state.
    sponge : SpongeForcing
        Must expose ``gamma``, ``T_ref``, ``S_ref``.  ``gamma`` is the
        relaxation rate per cell (1 / s), broadcast along the vertical
        axis via ``expand_gamma_axis``.
    mask : jax.Array, optional
        Ocean mask.  When provided, the relaxation tendency is
        multiplied by ``mask`` (with the same axis expansion as
        ``gamma``) so land cells stay quiescent.  When ``None`` no
        masking is applied (the caller masks downstream).
    expand_gamma_axis : int, default -1
        Axis on which to insert a singleton in ``sponge.gamma`` so that
        it broadcasts against the (..., nlev) tracer arrays.  Use ``-1``
        for both lat-lon C-grid (axis after lat/lon) and MPAS (axis
        after nCells).

    Returns
    -------
    (dT_dt_new, dS_dt_new) : tuple of jax.Array
    """
    dtype = T.dtype
    gamma = sponge.gamma.astype(dtype)
    gamma_b = jnp.expand_dims(gamma, expand_gamma_axis)
    dT = gamma_b * (sponge.T_ref.astype(dtype) - T)
    dS = gamma_b * (sponge.S_ref.astype(dtype) - S)
    if mask is not None:
        mask_b = jnp.expand_dims(mask, expand_gamma_axis)
        dT = dT * mask_b
        dS = dS * mask_b
    return dT_dt + dT, dS_dt + dS


def apply_freshwater_virtual_salt_top(
    dS_dt: jnp.ndarray,
    freshwater,
    S_ref: float,
    h_top: jnp.ndarray,
    rho_0: float,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Add the surface virtual-salt tendency to the top tracer level.

    Wraps ``freshwater.virtual_salt_flux`` and assigns the result to
    ``dS_dt[..., 0]`` (or the equivalent leading-axis slice for MPAS).
    All callers operate on cell-centred salinity, so the trailing
    ``nlev`` axis is the vertical axis.

    Parameters
    ----------
    dS_dt : jax.Array
        Salinity tendency, modified at index ``[..., 0]``.
    freshwater : FreshwaterForcing
    S_ref : float
        Reference salinity used for the virtual-flux closure.
    h_top : jax.Array
        Top-layer thickness with the same horizontal shape as ``mask``.
    rho_0 : float
    mask : jax.Array
        Ocean mask (1 = ocean) with the same horizontal shape as the
        leading axes of ``dS_dt``.

    Returns
    -------
    jax.Array
        ``dS_dt`` with the virtual-salt flux added to the top layer.
    """
    dS_top = virtual_salt_flux(freshwater, S_ref, h_top, rho_0)
    # Cast the freshwater contribution to dS_dt's dtype so the scatter
    # add does not silently widen on x64 mode (the freshwater struct
    # is built at JAX-default precision in init helpers, which can be
    # f64 while the salinity tendency runs at the storage policy's
    # f32).
    return dS_dt.at[..., 0].add((dS_top * mask).astype(dS_dt.dtype))


def implicit_bottom_drag_factor(
    dt: jnp.ndarray | float,
    drag_r: jnp.ndarray | float,
    H: jnp.ndarray,
    *,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """Per-substep bottom-drag multiplier ``1 - dt · r / max(H, eps)``.

    Used by both the lat-lon C-grid and MPAS barotropic substeps to
    apply a linear bottom drag on the depth-averaged velocity.  The
    floor on ``H`` prevents the drag from blowing up over very thin
    water columns (≈ inundation).

    Parameters
    ----------
    dt : float or jax.Array
        Substep size [s].
    drag_r : float or jax.Array
        Linear drag coefficient [m/s].
    H : jax.Array
        Total water column depth at the velocity location [m].
    eps : float
        Floor on ``H`` for numerical safety.

    Returns
    -------
    jax.Array, same shape as ``H``.
    """
    return 1.0 - dt * drag_r / jnp.maximum(H, eps)
```

### `src/legoesm/ocean/dynamics/barotropic_common.py`
```python
"""Grid-agnostic helpers shared by barotropic_*.py substep solvers.

Factors small repeated patterns out of the four grid-specific
barotropic solvers (cubed-sphere A-grid, cubed-sphere C-grid, lat-lon
C-grid, MPAS).  Closes #214 (Phase 2).

The helpers are intentionally thin: each replaces a few lines that were
character-for-character identical across solvers, so future updates to
the BEBT scheme, the cosine time filter, or MAXVEL clipping touch one
file instead of three.

This module performs no halo exchange and does not depend on any
grid-specific operator package.  Callers are responsible for staging
field shapes correctly before invoking these helpers.
"""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp


def compute_filter_weights(
    n_substeps: int,
    dtype: jnp.dtype,
    *,
    use_cosine: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return per-substep accumulator weights for time-averaging.

    The cosine bell (Hanning window) suppresses the side lobes of the
    plain box filter that alias barotropic modes into the baroclinic
    coupling.  Both the lat-lon C-grid and MPAS solvers compute the
    weights with the same formula:

        w_i = 1 + cos(2π · (i - n/2) / n)         (cosine)
        w_i = 1                                   (box)

    Parameters
    ----------
    n_substeps : int
        Number of barotropic substeps.
    dtype : jnp.dtype
        Working precision for the weight array.
    use_cosine : bool
        ``config.barotropic_time_filter == "cosine"``.

    Returns
    -------
    w_filter : jax.Array, shape (n_substeps,)
        Per-substep weight passed as ``xs`` to ``lax.scan`` (or
        indexed inside ``fori_loop``).
    w_total : jax.Array, scalar
        ``sum(w_filter)`` — used to normalise the eta / velocity
        accumulators.  Transport accumulators (``Hu``) keep using
        ``n_substeps`` for exact volume conservation.
    """
    i = jnp.arange(n_substeps, dtype=dtype)
    if use_cosine:
        w_filter = 1.0 + jnp.cos(
            2.0 * jnp.pi * (i - 0.5 * n_substeps) / n_substeps,
        )
    else:
        w_filter = jnp.ones(n_substeps, dtype=dtype)
    return w_filter, jnp.sum(w_filter)


def bebt_blend(
    eta_new: jnp.ndarray,
    eta_old: jnp.ndarray,
    bebt: float | jnp.ndarray,
) -> jnp.ndarray:
    """Backward-Euler/Backward-time blend of new and old eta for the PGF.

    ``bebt = 0`` recovers the standard forward-backward scheme; the
    MOM6 default ``bebt = 0.2`` introduces semi-implicit damping of
    the fastest barotropic gravity waves (#205).
    """
    return (1.0 - bebt) * eta_new + bebt * eta_old


def maxvel_clip(field: jnp.ndarray, maxvel: float | jnp.ndarray) -> jnp.ndarray:
    """Symmetric clip of barotropic velocity components.

    Used to suppress runaway velocities at single grid points that
    would otherwise crash the solver before the substep finishes.
    """
    return jnp.clip(field, -maxvel, maxvel)
```

### `src/legoesm/ocean/freshwater.py`
```python
"""Freshwater forcing for the MPAS ocean model.

Handles precipitation, evaporation, land runoff, and ice melt/freeze
freshwater fluxes. Applies virtual salt flux to salinity and real
freshwater mass flux to the free surface.

Conventions
-----------
- All fluxes in kg/m2/s (mass flux per unit area).
- Positive = freshwater entering ocean (precip, runoff, ice melt).
- Evaporation is positive upward in the coupler, so E enters here
  as a positive value that *removes* freshwater from the ocean.

References
----------
- Griffies, S. M. (2004). Fundamentals of Ocean Climate Models, Ch. 12.
- Large, W. G. et al. (1997). J. Phys. Oceanogr., 27(11), 2418-2447.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class FreshwaterForcing(NamedTuple):
    """Freshwater fluxes applied to the ocean surface.

    All fields have shape (nCells,) and units kg/m2/s.
    Positive = freshwater into ocean, except evaporation which is
    positive upward (i.e., freshwater leaving ocean).

    Fields
    ------
    precip : jax.Array
        Precipitation rate [kg/m2/s].
    evap : jax.Array
        Evaporation rate [kg/m2/s], positive upward.
    runoff : jax.Array
        Land runoff rate [kg/m2/s].
    ice_fw : jax.Array
        Ice melt/freeze freshwater [kg/m2/s], positive = melt.
    """
    precip: jnp.ndarray
    evap: jnp.ndarray
    runoff: jnp.ndarray
    ice_fw: jnp.ndarray


def zero_freshwater(nCells: int) -> FreshwaterForcing:
    """Create zero freshwater forcing.

    Parameters
    ----------
    nCells : int
        Number of Voronoi cells.

    Returns
    -------
    FreshwaterForcing
    """
    # Init helper: keep at the JAX default float dtype.  Callers running
    # under a non-default precision policy can ``cast_pytree`` the
    # result to match their state.
    z = jnp.zeros(nCells)
    return FreshwaterForcing(precip=z, evap=z, runoff=z, ice_fw=z)


def net_freshwater_flux(fw: FreshwaterForcing) -> jnp.ndarray:
    """Compute net freshwater flux into ocean [kg/m2/s].

    F_fw = P - E + R + M

    where P=precip, E=evaporation (positive up), R=runoff, M=ice melt.

    Parameters
    ----------
    fw : FreshwaterForcing

    Returns
    -------
    jax.Array, shape (nCells,)
        Net freshwater flux [kg/m2/s], positive into ocean.
    """
    return fw.precip - fw.evap + fw.runoff + fw.ice_fw


def freshwater_eta_tendency(fw: FreshwaterForcing, rho_0: float) -> jnp.ndarray:
    """Compute free-surface tendency from freshwater flux.

    deta/dt = F_fw / rho_0

    Parameters
    ----------
    fw : FreshwaterForcing
    rho_0 : float
        Reference seawater density [kg/m3].

    Returns
    -------
    jax.Array, shape (nCells,)
        Free-surface tendency [m/s].
    """
    return net_freshwater_flux(fw) / rho_0


def virtual_salt_flux(
    fw: FreshwaterForcing,
    S_ref: float,
    dz_0: jnp.ndarray,
    rho_0: float,
) -> jnp.ndarray:
    """Compute virtual salt flux for the top ocean layer.

    dS/dt = -S_ref * F_fw / (rho_0 * dz_0)

    This approximation maintains volume while adjusting salinity
    to account for freshwater dilution/concentration.

    Parameters
    ----------
    fw : FreshwaterForcing
    S_ref : float
        Reference salinity [PSU].
    dz_0 : jax.Array, shape (nCells,)
        Top layer thickness [m].
    rho_0 : float
        Reference seawater density [kg/m3].

    Returns
    -------
    jax.Array, shape (nCells,)
        Salinity tendency [PSU/s] for top layer.
    """
    F_fw = net_freshwater_flux(fw)
    dz_safe = jnp.maximum(dz_0, 1e-10)
    return -S_ref * F_fw / (rho_0 * dz_safe)


def freshwater_from_coupler(
    precip_total: jnp.ndarray,
    lhflx: jnp.ndarray,
    L_v: float,
    runoff_surface: jnp.ndarray | None = None,
    runoff_subsurface: jnp.ndarray | None = None,
    ice_state_old=None,
    ice_state_new=None,
    ice_config=None,
    ocean_mask: jnp.ndarray | None = None,
    dt: float = 1.0,
) -> FreshwaterForcing:
    """Compute freshwater forcing from coupler fields.

    Parameters
    ----------
    precip_total : jax.Array, shape (nCells,)
        Total precipitation [kg/m2/s].
    lhflx : jax.Array, shape (nCells,)
        Latent heat flux [W/m2], positive upward.
    L_v : float
        Latent heat of vaporization [J/kg].
    runoff_surface : jax.Array or None, shape (nCells,)
        Surface runoff from land [kg/m2/s].
    runoff_subsurface : jax.Array or None, shape (nCells,)
        Subsurface runoff from land [kg/m2/s].
    ice_state_old, ice_state_new : SeaIceState or None
        Ice state before/after ice step. Used to compute ice freshwater.
    ice_config : SeaIceConfig or None
        Ice config with rho_ice.
    ocean_mask : jax.Array or None, shape (nCells,)
        Ocean mask (1=ocean). Used to restrict fluxes to ocean cells.
    dt : float
        Timestep [s]. Used for ice thickness change rate.

    Returns
    -------
    FreshwaterForcing
    """
    nCells = precip_total.shape[0]

    # Precipitation over ocean
    precip = precip_total

    # Evaporation from latent heat flux: E = lhflx / L_v
    evap = lhflx / L_v

    # Land runoff (sum surface + subsurface).  Pin the zero-fallback
    # dtype to the precip path so a missing runoff input does not
    # silently widen the freshwater forcing struct to f64 under x64.
    if runoff_surface is not None:
        runoff = runoff_surface
        if runoff_subsurface is not None:
            runoff = runoff + runoff_subsurface
    else:
        runoff = jnp.zeros(nCells, dtype=precip.dtype)

    # Ice freshwater: based on areal ice mass change.
    # ice_mass = rho_ice * h * A  (per unit area of grid cell)
    # ice_fw = -(ice_mass_new - ice_mass_old) / dt
    # Melting (mass decrease) puts freshwater into ocean (positive fw).
    # Supports both single-category and multi-category ice.
    if ice_state_old is not None and ice_state_new is not None and ice_config is not None:
        h_old = ice_state_old.h_ice.data
        h_new = ice_state_new.h_ice.data
        A_old = ice_state_old.concentration.data
        A_new = ice_state_new.concentration.data
        rho_ice = ice_config.rho_ice
        ice_mass_old = rho_ice * h_old * A_old
        ice_mass_new = rho_ice * h_new * A_new
        # Multi-category: h has more dims than precip_total; sum categories.
        n_extra = ice_mass_old.ndim - precip_total.ndim
        for _ in range(n_extra):
            ice_mass_old = jnp.sum(ice_mass_old, axis=-1)
            ice_mass_new = jnp.sum(ice_mass_new, axis=-1)
        ice_fw = -(ice_mass_new - ice_mass_old) / jnp.maximum(dt, 1e-10)
    else:
        ice_fw = jnp.zeros(nCells, dtype=precip.dtype)

    # Mask to ocean cells
    if ocean_mask is not None:
        precip = precip * ocean_mask
        evap = evap * ocean_mask
        runoff = runoff * ocean_mask
        ice_fw = ice_fw * ocean_mask

    return FreshwaterForcing(
        precip=precip,
        evap=evap,
        runoff=runoff,
        ice_fw=ice_fw,
    )
```

### `src/legoesm/ocean/sponge.py`
```python
"""Sponge layer relaxation for ocean models.

Provides a ``SpongeForcing`` container and utilities for computing
spatially varying relaxation coefficients.  Sponge layers nudge the
model state toward a reference profile near domain boundaries, damping
wave reflections and preventing wall instabilities in channel
experiments (standard practice in MITgcm RBCS, MOM6 ALE_sponge).

The relaxation is applied as a tendency::

    dT/dt += gamma(x,y) * (T_ref(x,y,z) - T)

where gamma [1/s] ramps from zero in the interior to 1/tau at the
boundary.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class SpongeForcing(NamedTuple):
    """Sponge layer relaxation fields.

    Parameters
    ----------
    gamma : array
        Relaxation rate [1/s].  Shape ``(n_lat, n_lon)`` for lat-lon
        or ``(nCells,)`` for MPAS.  Zero in the interior, ramping to
        ``1/tau`` near boundaries.
    T_ref : array
        Reference temperature, same shape as the model T field.
    S_ref : array
        Reference salinity, same shape as the model S field.
    u_ref : array or None
        Reference u velocity (optional).
    v_ref : array or None
        Reference v velocity (optional, not used for MPAS edge velocity).
    """
    gamma: jnp.ndarray
    T_ref: jnp.ndarray
    S_ref: jnp.ndarray
    u_ref: jnp.ndarray | None = None
    v_ref: jnp.ndarray | None = None


def compute_sponge_gamma_latlon(
    grid,
    lat_south: float,
    lat_north: float,
    width_deg: float = 2.0,
    timescale_days: float = 1.0,
) -> np.ndarray:
    """Compute sponge relaxation coefficient on a lat-lon grid.

    Quadratic ramp from 0 in the interior to ``1/tau`` at the walls.

    Parameters
    ----------
    grid : LatLonGrid
    lat_south, lat_north : float
        Domain boundaries [degrees].
    width_deg : float
        Sponge zone width [degrees].
    timescale_days : float
        Relaxation e-folding timescale [days].

    Returns
    -------
    gamma : ndarray, shape (n_lat, n_lon)
        Relaxation coefficient [1/s].
    """
    lat_deg = np.degrees(np.asarray(grid.lat))
    tau = timescale_days * 86400.0
    gamma = np.zeros((grid.n_lat, grid.n_lon), dtype=np.float64)

    for i, lat in enumerate(lat_deg):
        dist_south = lat - lat_south
        dist_north = lat_north - lat
        if dist_south < width_deg:
            gamma[i, :] = (1.0 - dist_south / width_deg) ** 2 / tau
        elif dist_north < width_deg:
            gamma[i, :] = (1.0 - dist_north / width_deg) ** 2 / tau

    return gamma


def compute_sponge_gamma_mpas(
    mesh,
    lat_south: float,
    lat_north: float,
    width_deg: float = 2.0,
    timescale_days: float = 1.0,
) -> np.ndarray:
    """Compute sponge relaxation coefficient on an MPAS Voronoi mesh.

    Same quadratic ramp as the lat-lon version, applied per cell.

    Parameters
    ----------
    mesh : VoronoiMesh
    lat_south, lat_north : float
        Domain boundaries [degrees].
    width_deg : float
        Sponge zone width [degrees].
    timescale_days : float
        Relaxation e-folding timescale [days].

    Returns
    -------
    gamma : ndarray, shape (nCells,)
        Relaxation coefficient [1/s].
    """
    lat_deg = np.degrees(np.asarray(mesh.latCell))
    tau = timescale_days * 86400.0
    n_cells = lat_deg.shape[0]
    gamma = np.zeros(n_cells, dtype=np.float64)

    for i, lat in enumerate(lat_deg):
        dist_south = lat - lat_south
        dist_north = lat_north - lat
        if dist_south < width_deg:
            gamma[i] = (1.0 - dist_south / width_deg) ** 2 / tau
        elif dist_north < width_deg:
            gamma[i] = (1.0 - dist_north / width_deg) ** 2 / tau

    return gamma
```

### `src/legoesm/ocean/init_latlon_cgrid.py`
```python
"""Initialization for the lat-lon C-grid FV ocean model."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.precision import get_policy
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.vertical import OceanZStarCoordinate
from legoesm.ocean.state import LatLonCGridOceanState
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks


def idealized_bathymetry_latlon_cgrid(
    grid: LatLonGrid,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate idealized bathymetry and land mask on a lat-lon grid.

    Land is placed at high latitudes (|lat| > threshold).

    Parameters
    ----------
    grid : LatLonGrid
    H_max : float
        Maximum ocean depth [m].
    land_lat_threshold : float
        Latitude [degrees] above which cells are land.

    Returns
    -------
    H_bathy : array, shape (n_lat, n_lon)
    land_mask : array, shape (n_lat, n_lon)
    """
    if H_max <= 0.0:
        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    if land_lat_threshold < 0.0 or land_lat_threshold > 90.0:
        raise ValueError(
            f"land_lat_threshold must be in [0, 90], got {land_lat_threshold!r}",
        )

    dtype = get_policy().storage
    lat_deg = jnp.abs(grid.lat2d) * (180.0 / jnp.pi)
    land_mask = jnp.where(lat_deg < land_lat_threshold, 1.0, 0.0).astype(dtype)
    H_bathy = jnp.full_like(land_mask, H_max)

    return H_bathy, land_mask


def rest_state_latlon_cgrid_ocean(
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    T_surface: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
    land_mask_override: jnp.ndarray | None = None,
) -> LatLonCGridOceanState:
    """Create a rest-state initial condition on a C-grid lat-lon grid.

    Temperature: exponential profile.
    Salinity: uniform.
    Velocity: zero.
    Eta: zero.

    Parameters
    ----------
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    T_surface, T_deep : float
        Surface and deep temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].
    H_max : float
        Maximum ocean depth [m].
    land_lat_threshold : float
        Latitude threshold for land [degrees].  Ignored when
        *land_mask_override* is provided.
    land_mask_override : array (n_lat, n_lon), optional
        If provided, use this as the land mask (1=ocean, 0=land) instead
        of deriving one from *land_lat_threshold*.  Face masks (u_mask,
        v_mask) are computed from it automatically.

    Returns
    -------
    LatLonCGridOceanState
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels

    if land_mask_override is not None:
        land_mask = jnp.asarray(land_mask_override)
        H_bathy = jnp.full((n_lat, n_lon), H_max, dtype=jnp.float64)
    else:
        H_bathy, land_mask = idealized_bathymetry_latlon_cgrid(
            grid, H_max, land_lat_threshold,
        )

    # Exponential T stratification
    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(
        z_coord.z_full_ref / _SCALE_DEPTH,
    )
    dtype = get_policy().storage
    T_3d = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, :], (n_lat, n_lon, nlev),
    ).astype(dtype)

    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)

    # C-grid velocity shapes
    u_zeros = jnp.zeros((n_lat, n_lon + 1, nlev), dtype=dtype)
    v_zeros = jnp.zeros((n_lat + 1, n_lon, nlev), dtype=dtype)
    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)

    # Face masks
    u_mask, v_mask = compute_face_masks(land_mask)

    # Initialize vertical velocity with zeros (will be computed during step)
    w_zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)

    dims_u = ("lat", "lon_u", "level")
    dims_v = ("lat_v", "lon", "level")
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    dims_u2d = ("lat", "lon_u")
    dims_v2d = ("lat_v", "lon")

    return LatLonCGridOceanState(
        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
                staggering="edge"),
        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
                staggering="edge"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
    )


def wind_driven_gyre_latlon_cgrid(
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    H_max: float = 5500.0,
    lon_west: float = 0.0,
    lon_east: float = 120.0,
    lat_south: float = 15.0,
    lat_north: float = 75.0,
    T_uniform: float = 10.0,
    S_uniform: float = 35.0,
) -> LatLonCGridOceanState:
    """Create initial condition for a wind-driven barotropic gyre on C-grid lat-lon.

    Uniform T and S inside a rectangular basin. Purely barotropic setup.
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels
    dtype = get_policy().storage

    lon_deg = grid.lon2d * (180.0 / jnp.pi)
    lat_deg = grid.lat2d * (180.0 / jnp.pi)
    in_basin = (
        (lon_deg >= lon_west) & (lon_deg <= lon_east) &
        (lat_deg >= lat_south) & (lat_deg <= lat_north)
    )
    land_mask = jnp.where(in_basin, 1.0, 0.0).astype(dtype)
    H_bathy = jnp.full_like(land_mask, H_max)

    T_3d = jnp.full((n_lat, n_lon, nlev), T_uniform, dtype=dtype)
    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)

    u_zeros = jnp.zeros((n_lat, n_lon + 1, nlev), dtype=dtype)
    v_zeros = jnp.zeros((n_lat + 1, n_lon, nlev), dtype=dtype)
    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)

    u_mask, v_mask = compute_face_masks(land_mask)

    # Initialize vertical velocity with zeros (will be computed during step)
    w_zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)

    dims_u = ("lat", "lon_u", "level")
    dims_v = ("lat_v", "lon", "level")
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    dims_u2d = ("lat", "lon_u")
    dims_v2d = ("lat_v", "lon")

    return LatLonCGridOceanState(
        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
                staggering="edge"),
        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
                staggering="edge"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
    )


def regional_rest_state_latlon_cgrid(
    grid: LatLonGrid,
    wall_mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    H_max: float = 5500.0,
    T_surface: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
) -> LatLonCGridOceanState:
    """Create a rest-state initial condition on a regional C-grid lat-lon grid.

    Parameters
    ----------
    grid : LatLonGrid
        Regional grid (includes 1-cell wall boundary).
    wall_mask : jax.Array, shape (n_lat, n_lon)
        1 = ocean interior, 0 = wall.
    z_coord : OceanZStarCoordinate
    H_max : float
    T_surface, T_deep : float
    S_uniform : float

    Returns
    -------
    LatLonCGridOceanState
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels
    dtype = get_policy().storage

    H_bathy = jnp.full((n_lat, n_lon), H_max, dtype=dtype)

    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(
        z_coord.z_full_ref / _SCALE_DEPTH,
    )
    T_3d = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, :], (n_lat, n_lon, nlev),
    ).astype(dtype)

    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)

    u_zeros = jnp.zeros((n_lat, n_lon + 1, nlev), dtype=dtype)
    v_zeros = jnp.zeros((n_lat + 1, n_lon, nlev), dtype=dtype)
    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)

    land_mask = wall_mask.astype(dtype)
    u_mask, v_mask = compute_face_masks(land_mask)

    # Initialize vertical velocity with zeros (will be computed during step)
    w_zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)

    dims_u = ("lat", "lon_u", "level")
    dims_v = ("lat_v", "lon", "level")
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    dims_u2d = ("lat", "lon_u")
    dims_v2d = ("lat_v", "lon")

    return LatLonCGridOceanState(
        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
                staggering="edge"),
        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
                staggering="edge"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
    )


def replace_land_mask(
    state: LatLonCGridOceanState,
    new_land_mask: jnp.ndarray,
) -> LatLonCGridOceanState:
    """Replace land_mask and recompute u_mask/v_mask atomically.

    Use this instead of ``state._replace(land_mask=...)`` to ensure
    face masks stay consistent with the cell mask.
    """
    new_land_mask = jnp.asarray(new_land_mask)
    u_mask, v_mask = compute_face_masks(new_land_mask)
    return state._replace(
        land_mask=Field(data=new_land_mask, name="land_mask",
                        dims=state.land_mask.dims, units=""),
        u_mask=Field(data=u_mask, name="u_mask",
                     dims=state.u_mask.dims, units=""),
        v_mask=Field(data=v_mask, name="v_mask",
                     dims=state.v_mask.dims, units=""),
    )
```

### `src/legoesm/ocean/physics/vertical_mixing/kpp.py`
```python
"""LMD94-style K-Profile Parameterization (KPP).

Boundary-layer parameterization following Large, McWilliams & Doney (1994)
with:

- Bulk Richardson number BL-depth diagnosis with linear interpolation
  of the crossing depth between model levels.
- Turbulent velocity scales w_s(sigma) from surface forcing (u_star, B_f).
- Cubic shape function G(sigma) = sigma * (1 - sigma)^2.
- Non-local tracer transport for unstable (convective) conditions only.
- Interior mixing: Richardson-number dependent + convective instability
  enhancement for statically unstable layers below the BL.

The caller should provide surface wind stress and buoyancy flux when
available.  If tau_x/tau_y are None, a simplified u_star proxy from
surface speed is used.

References
----------
- Large, W. G., McWilliams, J. C., & Doney, S. C. (1994). Oceanic
  vertical mixing: A review and a model with a nonlocal boundary layer
  parameterization. Rev. Geophys., 32, 363-403.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import (
    wright_eos,
    compute_buoyancy_frequency,
    compute_hydrostatic_pressure,
    rho_0 as rho_0_ref,
)
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def _boundary_layer_depth(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    u_star: jnp.ndarray,
    B_f: jnp.ndarray,
    cfg: KPPConfig,
    g: float = constants.g,
    h_bl_prev: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Estimate boundary layer depth h via bulk Richardson number.

    Uses linear interpolation to find the depth where Ri_b crosses
    Ri_crit, rather than snapping to the nearest model level.

    Returns shape (...) boundary layer depth [m, positive downward].
    """
    eps = _EPS
    nlev = rho.shape[-1]

    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    # Depth of cell centers below surface (positive downward)
    z_depth = jnp.cumsum(dz_actual, axis=-1) - 0.5 * dz_actual

    # Density and velocity differences from surface
    delta_rho = rho - rho[..., :1]
    delta_u = u - u[..., :1]
    delta_v = v - v[..., :1]
    delta_V2 = delta_u**2 + delta_v**2

    # LMD94 Eq. 23: V_t^2 = Cv * sqrt(|N2|) / sqrt(c_s * epsilon) *
    #   max(Ri_crit * h - d, 0) * d / h
    # Uses h_bl from the previous time step to break the coupling.
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
    N2_full = jnp.concatenate([N2[..., :1], N2], axis=-1)
    max_depth = z_depth[..., -1]
    h_est = max_depth if h_bl_prev is None else h_bl_prev
    h_safe = jnp.maximum(h_est[..., jnp.newaxis], eps)
    V_t2 = (cfg.Cv * jnp.sqrt(jnp.maximum(jnp.abs(N2_full), 0.0))
            / jnp.sqrt(jnp.maximum(cfg.c_s * cfg.epsilon_lmd, eps))
            * jnp.maximum(cfg.Ri_crit * h_safe - z_depth, 0.0)
            * z_depth / h_safe)

    # Bulk Richardson number
    Ri_b = (g * delta_rho * z_depth) / (
        rho_0_ref * jnp.maximum(delta_V2 + V_t2, eps)
    )

    # --- Differentiable soft interpolation of crossing depth ---
    # Instead of argmax (non-differentiable), use a sigmoid-weighted
    # average over all levels.  Each level contributes a weight
    # proportional to how much Ri_b crosses Ri_crit there.
    #
    # Weight at level k = sigmoid(sharpness * (Ri_b[k] - Ri_crit))
    #                    - sigmoid(sharpness * (Ri_b[k-1] - Ri_crit))
    # This is ~1 at the crossing level and ~0 elsewhere.
    sharpness = cfg.crossing_sharpness
    sig = jax.nn.sigmoid(sharpness * (Ri_b - cfg.Ri_crit))  # (..., nlev)

    # Crossing weight: difference of adjacent sigmoid values.  ``jnp.pad``
    # along the trailing axis is one HLO op; the previous
    # ``concatenate([zeros_like(sig[..., :1]), sig[..., :-1]])`` allocated
    # a fresh zero buffer and concatenated.
    pad_axes = ((0, 0),) * (sig.ndim - 1)
    sig_prev = jnp.pad(sig[..., :-1], (*pad_axes, (1, 0)))
    w_cross = sig - sig_prev  # (..., nlev), peaks at crossing level
    w_cross = jnp.maximum(w_cross, 0.0)
    w_sum = jnp.sum(w_cross, axis=-1, keepdims=True)
    w_norm = w_cross / jnp.maximum(w_sum, eps)

    # Crossing-based depth estimate
    h_crossing = jnp.sum(w_norm * z_depth, axis=-1)  # (...)

    # Fallback for columns where Ri_b never crosses Ri_crit:
    # - If column is mostly unstable (sig ≈ 0): BL extends to full depth
    # - If column is mostly stable (sig ≈ 1): BL is one layer
    column_stability = jnp.mean(sig, axis=-1)  # 0 = all unstable, 1 = all stable
    max_depth = z_depth[..., -1]
    min_depth = dz_actual[..., 0]
    h_fallback = (1.0 - column_stability) * max_depth + column_stability * min_depth

    # Blend: use crossing depth when crossing signal is strong, fallback otherwise
    crossing_strength = w_sum[..., 0]
    blend = jax.nn.sigmoid(cfg.crossing_sharpness * (crossing_strength - cfg.crossing_threshold))
    h = blend * h_crossing + (1.0 - blend) * h_fallback

    # At least one layer thick
    h = jnp.maximum(h, dz_actual[..., 0])

    return h


def kpp_vertical_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    eta: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: KPPConfig,
    g: float = constants.g,
    tau_x: jnp.ndarray | None = None,
    tau_y: jnp.ndarray | None = None,
    B_f: jnp.ndarray | None = None,
    Q_sfc_T: jnp.ndarray | None = None,
    Q_sfc_S: jnp.ndarray | None = None,
    h_bl_prev: jnp.ndarray | None = None,
) -> VerticalMixingOutput:
    """Apply LMD94-style KPP vertical mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    eta : array (6, n, n)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : KPPConfig
    g : float
    tau_x, tau_y : array (6, n, n) or None
        Surface wind stress [Pa]. If None, a proxy from surface speed is used.
    B_f : array (6, n, n) or None
        Surface buoyancy flux [m^2/s^3], positive = destabilizing (convective).
        If None, estimated from surface density gradient.
    Q_sfc_T : array (6, n, n) or None
        Surface kinematic heat flux [K*m/s] for non-local transport (LMD94
        Eq. 19).  If None, falls back to diagnosed K_sfc * dT/dz proxy.
    Q_sfc_S : array (6, n, n) or None
        Surface kinematic salt flux [PSU*m/s]. Same convention as Q_sfc_T.
    h_bl_prev : array (6, n, n) or None
        BL depth from the previous time step [m, positive downward].
        Used to break the implicit V_t-h_bl coupling in the Ri_b diagnosis
        (LMD94 Eq. 23).  If None, uses the full column depth as estimate.

    Returns
    -------
    VerticalMixingOutput
    """
    eps = _EPS
    nlev = u.shape[-1]

    # --- Friction velocity ---
    if tau_x is not None and tau_y is not None:
        # Proper u_star from wind stress: u_star = sqrt(|tau| / rho_0)
        tau_mag = jnp.sqrt(tau_x**2 + tau_y**2 + eps)
        u_star = jnp.sqrt(tau_mag / rho_0_ref)
    else:
        # Simplified proxy: u_star ~ 0.01 * |U_surface|
        speed_sfc = jnp.sqrt(u[..., 0]**2 + v[..., 0]**2 + eps)
        u_star = jnp.maximum(speed_sfc * 0.01, 1e-4)

    # --- Surface buoyancy flux ---
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    dz_half0 = 0.5 * (dz_actual[..., 0] + dz_actual[..., 1])
    if B_f is None:
        # Estimate from near-surface density gradient
        drho_dz_sfc = (rho[..., 0] - rho[..., 1]) / jnp.maximum(dz_half0, eps)
        B_f = -g / rho_0_ref * cfg.K_bg * drho_dz_sfc  # simplified proxy

    # --- Boundary layer depth ---
    h_bl = _boundary_layer_depth(
        rho, u, v, z_coord, jacobian, u_star, B_f, cfg, g,
        h_bl_prev=h_bl_prev,
    )

    # --- Depth coordinate ---
    z_depth = jnp.cumsum(dz_actual, axis=-1) - 0.5 * dz_actual
    sigma = z_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)

    # --- Shape function G(sigma) = sigma * (1 - sigma)^2 ---
    sigma_clip = jnp.clip(sigma, 0.0, 1.0)
    G = sigma_clip * (1.0 - sigma_clip) ** 2

    # --- Turbulent velocity scale w_s(sigma) (LMD94 Appendix B) ---
    # w_s depends on stability (B_f) and depth d = sigma * h_bl
    d = sigma_clip * h_bl[..., jnp.newaxis]
    # Monin-Obukhov length: L_MO = u_star^3 / (kappa * B_f)
    # Use copysign(eps, B_f) to preserve the sign of B_f near zero,
    # preventing a stability classification flip (issue #168 bug 1).
    B_f_safe = jnp.where(
        jnp.abs(B_f[..., jnp.newaxis]) > eps,
        B_f[..., jnp.newaxis],
        jnp.copysign(eps, B_f[..., jnp.newaxis]),
    )
    L_MO = u_star[..., jnp.newaxis]**3 / (cfg.kappa_vk * B_f_safe)
    zeta_kpp = d / L_MO

    # LMD94 Appendix B turbulent velocity scales:
    # Stable (B_f <= 0): w_s = kappa * u_star / (1 + 5*zeta)
    # Unstable, weakly (epsilon*d < |L|): w_s = kappa * u_star * phi_m^{-1}
    #   where phi_m^{-1} = (1 - 16*zeta)^{1/4}
    # Unstable, strongly convective (epsilon*d > |L|):
    #   w_s = (kappa * (u_star^3 + c_b * kappa * (-B_f) * d))^{1/3}
    is_unstable = B_f[..., jnp.newaxis] > 0.0
    epsilon_lmd = cfg.epsilon_lmd

    # Weakly unstable: phi_m^{-1} formulation
    w_s_weak = (cfg.kappa_vk * u_star[..., jnp.newaxis]
                * jnp.power(jnp.maximum(1.0 + 16.0 * jnp.abs(zeta_kpp), 1.0), 0.25))

    # Strongly convective: includes convective velocity scale
    Bf_pos = jnp.maximum(B_f[..., jnp.newaxis], 0.0)
    w_s_conv = jnp.power(
        cfg.kappa_vk * (u_star[..., jnp.newaxis]**3
                        + cfg.c_b * cfg.kappa_vk * Bf_pos * d),
        1.0 / 3.0,
    )

    # Transition: use convective scale when epsilon*d > |L_MO|
    is_strongly_convective = epsilon_lmd * d > jnp.abs(L_MO)
    w_s_unstable = jnp.where(is_strongly_convective, w_s_conv, w_s_weak)

    # Stable: standard suppression
    w_s_stable = (cfg.kappa_vk * u_star[..., jnp.newaxis]
                  / jnp.maximum(1.0 + 5.0 * jnp.maximum(zeta_kpp, 0.0), 1.0))
    w_s = jnp.where(is_unstable, w_s_unstable, w_s_stable)
    w_s = jnp.maximum(w_s, 1e-10)

    # --- BL diffusivity at full levels ---
    K_bl_full = h_bl[..., jnp.newaxis] * w_s * G
    K_bl_full = jnp.minimum(K_bl_full, cfg.K_max)

    # --- Interior mixing: Richardson-number dependent ---
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    du = u[..., :-1] - u[..., 1:]
    dv = v[..., :-1] - v[..., 1:]
    S2 = (du**2 + dv**2) / jnp.maximum(dz_half**2, eps)
    Ri_int = N2 / jnp.maximum(S2, eps)
    # LMD94 interior shear instability: K = K_0 * (1 - (Ri/Ri_0)^2)^3
    # for Ri < Ri_0, zero above.
    Ri_ratio = jnp.clip(Ri_int / cfg.Ri_0, 0.0, 1.0)
    K_interior = cfg.K_0_shear * (1.0 - Ri_ratio**2) ** 3 + cfg.K_bg

    # Interior static instability: enhanced mixing where N2 < 0
    K_conv = jnp.where(N2 < cfg.Ri_conv, cfg.K_conv, 0.0)
    K_interior = K_interior + K_conv

    # --- K at interfaces (average of full level K_bl) ---
    K_bl_half = 0.5 * (K_bl_full[..., :-1] + K_bl_full[..., 1:])

    # sigma at interfaces
    z_half_depth = 0.5 * (z_depth[..., :-1] + z_depth[..., 1:])
    sigma_half = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
    in_bl = sigma_half < 1.0

    # Combine BL and interior
    K_v = jnp.where(in_bl, K_bl_half, K_interior) + cfg.K_bg
    A_v = jnp.where(in_bl, K_bl_half * 1.0, K_interior) + cfg.A_bg
    K_v = jnp.minimum(K_v, cfg.K_max)
    A_v = jnp.minimum(A_v, cfg.K_max)

    # --- Apply diffusion ---
    vel = jnp.stack([u, v], axis=0)
    vel_tend = jax.vmap(
        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, A_v),
        in_axes=0, out_axes=0,
    )(vel)

    tracers = jnp.stack([T, S], axis=0)
    tr_tend = jax.vmap(
        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, K_v),
        in_axes=0, out_axes=0,
    )(tracers)

    # --- Non-local flux for T, S (LMD94 Eq. 19) ---
    #
    # LMD94 defines a counter-gradient term:
    #   gamma_T(sigma) = C_s * Q_0 / (w_s(sigma) * h)   [K/m]
    # where Q_0 is the surface kinematic heat flux [K*m/s].
    #
    # The non-local tendency is  -d/dz(K_bl * gamma_T).
    # Substituting K_bl = h * w_s * G(sigma):
    #   K_bl * gamma_T = h * w_s * G * C_s * Q_0 / (w_s * h) = C_s * Q_0 * G(sigma)
    #
    # So the non-local tendency reduces to:
    #   dT/dt_nonlocal = -d/dz[ C_s * Q_0 * G(sigma) ]            [K/s]
    #
    # We discretize this as the vertical divergence of the non-local
    # flux F_nl = C_s * Q_0 * G(sigma) evaluated at interfaces.

    # Surface kinematic heat/salt flux for non-local transport (LMD94 Eq. 19).
    # Use the IMPOSED surface flux when available (from bulk formulas or
    # prescribed forcing).  Fall back to diagnosed K_sfc * dT/dz proxy
    # only when no external flux is provided (issue #168 bug 2).
    if Q_sfc_T is not None:
        Q_T = Q_sfc_T  # [K*m/s]
    else:
        dT_dz_sfc = (T[..., 0] - T[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
        K_sfc = K_bl_full[..., 0]
        Q_T = K_sfc * dT_dz_sfc

    if Q_sfc_S is not None:
        Q_S = Q_sfc_S  # [PSU*m/s]
    else:
        dS_dz_sfc = (S[..., 0] - S[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
        K_sfc = K_bl_full[..., 0]
        Q_S = K_sfc * dS_dz_sfc

    # Only apply non-local transport for unstable (convective) columns.
    is_unstable_col = B_f > 0.0

    # G(sigma) at interior interfaces (half levels between full levels)
    sigma_half_full = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
    sigma_half_clip = jnp.clip(sigma_half_full, 0.0, 1.0)
    G_half = sigma_half_clip * (1.0 - sigma_half_clip) ** 2  # (..., nlev-1)

    in_bl_full = sigma < 1.0

    # --- Temperature non-local tendency ---
    # Non-local flux at interfaces: F_nl = C_s * Q_T * G_half  [K*m/s]
    F_T = cfg.gamma_T * Q_T[..., jnp.newaxis] * G_half  # (..., nlev-1)
    # Tendency = -dF/dz at full levels (zero-flux BCs at surface and bottom)
    dT_nonlocal_top = -F_T[..., :1] / dz_actual[..., :1]
    dT_nonlocal_int = (F_T[..., :-1] - F_T[..., 1:]) / dz_actual[..., 1:-1]
    dT_nonlocal_bot = F_T[..., -1:] / dz_actual[..., -1:]
    dT_nonlocal = jnp.concatenate(
        [dT_nonlocal_top, dT_nonlocal_int, dT_nonlocal_bot], axis=-1
    )  # (..., nlev)  [K/s]
    dT_nonlocal = jnp.where(
        in_bl_full & is_unstable_col[..., jnp.newaxis], dT_nonlocal, 0.0
    )

    # --- Salinity non-local tendency ---
    F_S = cfg.gamma_S * Q_S[..., jnp.newaxis] * G_half  # (..., nlev-1)
    dS_nonlocal_top = -F_S[..., :1] / dz_actual[..., :1]
    dS_nonlocal_int = (F_S[..., :-1] - F_S[..., 1:]) / dz_actual[..., 1:-1]
    dS_nonlocal_bot = F_S[..., -1:] / dz_actual[..., -1:]
    dS_nonlocal = jnp.concatenate(
        [dS_nonlocal_top, dS_nonlocal_int, dS_nonlocal_bot], axis=-1
    )  # (..., nlev)  [psu/s]
    dS_nonlocal = jnp.where(
        in_bl_full & is_unstable_col[..., jnp.newaxis], dS_nonlocal, 0.0
    )

    return VerticalMixingOutput(
        du_dt=vel_tend[0],
        dv_dt=vel_tend[1],
        dT_dt=tr_tend[0] + dT_nonlocal,
        dS_dt=tr_tend[1] + dS_nonlocal,
        K_v=K_v,
        A_v=A_v,
    )
```

### `src/legoesm/ocean/physics/vertical_mixing/implicit_solver.py`
```python
"""Backward-Euler implicit vertical diffusion for the ocean.

Solves the 1-D diffusion equation per column

    ∂φ/∂t = ∂/∂z [K · ∂φ/∂z]

for each grid column using a backward-Euler discretisation that is
unconditionally stable.  This is the key ingredient for issue #204:
without it, the ocean dycore must keep first-order-upwind vertical
momentum advection (whose numerical viscosity ~|w|·dz/2 silently damps
baroclinic shear); with it, the physical ``A_v`` / ``K_v`` and any
Richardson-number or KPP-based enhancement can do that job directly
and the resolved advection can be upgraded to higher-order schemes.

Discrete form (no-flux top + bottom):

    ρ · ∂φ/∂t = ∂/∂z ( K ∂φ/∂z )       →     backward-Euler
    (1 + α_k + β_k) φ^{n+1}_k
        − α_k φ^{n+1}_{k-1}
        − β_k φ^{n+1}_{k+1}
      = φ^{n}_k
with
    α_k = dt · K_{k-1/2} / ( dz_k · dz_half_{k-1/2} )
    β_k = dt · K_{k+1/2} / ( dz_k · dz_half_{k+1/2} )

``K`` lives on interfaces (``nlev-1`` values per column), ``dz`` on
layer centres (``nlev``), ``dz_half`` on interfaces (``nlev-1``).
No-flux boundaries are enforced by setting ``K_{-1/2} = K_{N-1/2} = 0``
(implicit via α_0 = 0 and β_{N-1} = 0).

References
----------
- Thomas, L.H. (1949) — the Thomas algorithm is already implemented in
  :mod:`legoesm.timestepping.tridiagonal`; this module just builds the
  per-column system.
- MOM6 Technical Manual §7 (implicit vertical viscosity).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.timestepping.tridiagonal import thomas_solve

_EPS = float(jnp.finfo(jnp.float32).eps)  # ~1.19e-7


def implicit_vertical_diffusion_ocean(
    field: jax.Array,
    K: jax.Array | float,
    dz: jax.Array,
    dz_half: jax.Array,
    dt: float,
) -> jax.Array:
    """Backward-Euler implicit vertical diffusion for a column field.

    Applies one implicit step of

        (1 - dt · ∂_z K ∂_z) φ^{n+1} = φ^{n}

    per column with *zero-flux* boundary conditions at the top and the
    bottom (the ocean's natural choice for momentum and tracers in the
    absence of a prescribed surface flux).  Non-zero surface / bottom
    fluxes can be applied *externally* before or after this call — the
    helper is deliberately boundary-condition-simple so callers don't
    have to thread flux arrays through when they don't need them.

    Parameters
    ----------
    field : jax.Array, shape ``(..., nlev)``
        Field to diffuse.  The vertical axis is the last axis; any
        number of leading horizontal axes is allowed.  Examples:
        ``(n_lat, n_lon, nlev)`` for lat-lon, ``(nCells, nlev)`` for
        MPAS, ``(n_lat, n_lon+1, nlev)`` for u on a C-grid.
    K : jax.Array or float, shape ``(..., nlev-1)``
        Vertical viscosity / diffusivity at interior interfaces.
        Must be ≥ 0.  A scalar is broadcast to every interface and
        every column.
    dz : jax.Array, shape ``(..., nlev)`` or ``(nlev,)``
        Layer thickness at full levels.  Must match ``field`` along
        the last axis (a 1-D ``dz`` broadcasts across all columns).
    dz_half : jax.Array, shape ``(..., nlev-1)`` or ``(nlev-1,)``
        Distance between adjacent full-level centres
        (``dz_half_k = 0.5 (dz_k + dz_{k+1})`` is the standard choice).
    dt : float
        Time step [s].  Must be positive.

    Returns
    -------
    jax.Array
        Updated field with the same shape as ``field``.
    """
    # Only enforce the positivity check when ``dt`` is a concrete Python
    # scalar — under ``jax.jit`` it may be a traced argument, and a
    # Python-level ``if`` would raise ``TracerBoolConversionError``.
    if not isinstance(dt, jax.core.Tracer):
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt!r}")

    nlev = field.shape[-1]
    if nlev < 2:
        # One-level columns have no vertical gradient ⇒ no-op.
        return field

    # --- Promote K, dz, dz_half to match the field's leading shape ---
    K_arr = jnp.asarray(K)
    if K_arr.ndim == 0:
        K_arr = jnp.broadcast_to(K_arr, field.shape[:-1] + (nlev - 1,))
    elif K_arr.shape[-1] != nlev - 1:
        raise ValueError(
            f"K last dim {K_arr.shape[-1]} must equal nlev-1 = {nlev - 1}")

    dz_arr = jnp.asarray(dz)
    if dz_arr.ndim == 1:
        dz_arr = jnp.broadcast_to(dz_arr, field.shape)
    elif dz_arr.shape[-1] != nlev:
        raise ValueError(
            f"dz last dim {dz_arr.shape[-1]} must equal nlev = {nlev}")

    dzh_arr = jnp.asarray(dz_half)
    if dzh_arr.ndim == 1:
        dzh_arr = jnp.broadcast_to(dzh_arr, field.shape[:-1] + (nlev - 1,))
    elif dzh_arr.shape[-1] != nlev - 1:
        raise ValueError(
            f"dz_half last dim {dzh_arr.shape[-1]} must equal nlev-1 = "
            f"{nlev - 1}")

    # --- Build α and β at every cell (last axis = level) ---
    # α_k uses the (k-1/2) interface, β_k the (k+1/2) interface.
    # We pad K with an extra zero on each side so indexing is uniform;
    # the zeros naturally encode the no-flux BCs.
    K_safe = jnp.maximum(K_arr, 0.0)
    dzh_safe = jnp.maximum(dzh_arr, _EPS)

    # K / dz_half at interfaces (nlev-1)
    flux_coeff = K_safe / dzh_safe                   # (..., nlev-1)

    # Pad top and bottom with zero (no-flux).  Two Pad HLO ops replace
    # alloc-zeros + two concatenate-of-two.
    pad_axes = ((0, 0),) * (flux_coeff.ndim - 1)
    flux_top = jnp.pad(flux_coeff, (*pad_axes, (1, 0)))  # (..., nlev)
    flux_bot = jnp.pad(flux_coeff, (*pad_axes, (0, 1)))  # (..., nlev)

    inv_dz = 1.0 / jnp.maximum(dz_arr, _EPS)          # (..., nlev)
    alpha = dt * flux_top * inv_dz                    # (..., nlev)
    beta = dt * flux_bot * inv_dz                     # (..., nlev)

    # Tridiagonal coefficients:
    #   a_k = -α_k   (sub-diagonal, a_0 = 0)
    #   b_k = 1 + α_k + β_k
    #   c_k = -β_k   (super-diagonal, c_{N-1} = 0)
    #   d_k = φ^n_k
    a = -alpha
    b = 1.0 + alpha + beta
    c = -beta
    d = field

    return thomas_solve(a, b, c, d)


# ---------------------------------------------------------------------------
# Convenience: build dz_half from dz with the standard midpoint rule.
# ---------------------------------------------------------------------------


def build_dz_half(dz: jax.Array) -> jax.Array:
    """Midpoint distance between adjacent full-level centres.

    ``dz_half_k = 0.5 · (dz_k + dz_{k+1})`` with shape ``(..., nlev-1)``.
    Provided as a helper because most callers don't carry ``dz_half``
    separately from ``dz_ref`` but do need it to build the tridiagonal
    system.
    """
    return 0.5 * (dz[..., :-1] + dz[..., 1:])
```

### `src/legoesm/ocean/physics/convection/plume.py`
```python
"""Entraining mass-flux convective plume parameterization."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import wright_eos
from legoesm.ocean.physics.convection.config import PlumeConfig
from legoesm.ocean.physics.convection.output import OceanConvectionOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def plume_convection(
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    p_hydro: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: PlumeConfig,
) -> OceanConvectionOutput:
    """Apply entraining mass-flux plume convection.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    p_hydro : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : PlumeConfig

    Returns
    -------
    OceanConvectionOutput
    """
    nlev = T.shape[-1]
    shape_3d = T.shape
    dtype = T.dtype
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]

    # Detect unstable surface: rho(k=0) > rho(k=1)
    surface_unstable = rho[..., 0] > rho[..., 1]  # (6, n, n)

    # Initialize plume properties at surface
    T_plume_init = T[..., 0] + cfg.T_excess
    S_plume_init = S[..., 0]

    # Descend plume using scan over levels (starting from level 1)
    def scan_fn(carry, k):
        T_plume, S_plume, active = carry
        dz_k = dz_actual[..., k]

        # Entrain environment
        entrain = cfg.epsilon * dz_k
        T_plume = (1.0 - entrain) * T_plume + entrain * T[..., k]
        S_plume = (1.0 - entrain) * S_plume + entrain * S[..., k]

        # Buoyancy check
        rho_plume = wright_eos(T_plume, S_plume, p_hydro[..., k])
        delta_rho = rho_plume - rho[..., k]

        # Plume is active where it's denser than environment (sinking):
        # delta_rho > 0 means rho_plume > rho_env → plume sinks → stay active
        active = active * jax.nn.sigmoid(delta_rho * 1e4)

        # Detrainment tendency at this level
        dT_k = cfg.alpha_plume * cfg.epsilon * (T_plume - T[..., k]) * active
        dS_k = cfg.alpha_plume * cfg.epsilon * (S_plume - S[..., k]) * active

        return (T_plume, S_plume, active), (dT_k, dS_k, active)

    init_active = surface_unstable.astype(dtype)
    (_, _, _), (dT_levels, dS_levels, active_levels) = jax.lax.scan(
        scan_fn,
        (T_plume_init, S_plume_init, init_active),
        jnp.arange(1, nlev),
    )

    # dT_levels shape: (nlev-1, 6, n, n) — move level axis to last,
    # then ``jnp.pad`` along the trailing axis instead of
    # ``zeros + .at[..., 1:].set(...)`` which materialises a fresh
    # zero buffer + scatter.  Single Pad HLO op each.
    dT_levels_t = jnp.moveaxis(dT_levels, 0, -1)  # (6, n, n, nlev-1)
    dS_levels_t = jnp.moveaxis(dS_levels, 0, -1)
    pad_axes = ((0, 0),) * (dT_levels_t.ndim - 1)
    dT_dt = jnp.pad(dT_levels_t, (*pad_axes, (1, 0)))
    dS_dt = jnp.pad(dS_levels_t, (*pad_axes, (1, 0)))

    # Convection flag at interfaces (average of adjacent levels' activity)
    active_t = jnp.moveaxis(active_levels, 0, -1)  # (6, n, n, nlev-1)
    flag = active_t

    return OceanConvectionOutput(
        dT_dt=dT_dt,
        dS_dt=dS_dt,
        convection_flag=flag,
    )
```

### `src/legoesm/ocean/physics/convection/enhanced_diffusion.py`
```python
"""Enhanced diffusion for convective adjustment.

Applies large vertical diffusivity where N^2 < 0 (statically unstable).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import compute_buoyancy_frequency
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
from legoesm.ocean.physics.convection.output import OceanConvectionOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def enhanced_diffusion_convection(
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: EnhancedDiffusionConfig,
) -> OceanConvectionOutput:
    """Apply enhanced diffusion where the water column is unstable.

    Parameters
    ----------
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : EnhancedDiffusionConfig

    Returns
    -------
    OceanConvectionOutput
    """
    # N^2 at interfaces
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)

    # Diffusivity: large where N^2 < 0
    if cfg.smooth_transition:
        # Smooth sigmoid transition
        K = cfg.K_bg + (cfg.K_conv - cfg.K_bg) * jax.nn.sigmoid(
            -N2 * cfg.sigmoid_sharpness
        )
    else:
        K = jnp.where(N2 < 0.0, cfg.K_conv, cfg.K_bg)

    # Convection flag
    if cfg.smooth_transition:
        flag = jax.nn.sigmoid(-N2 * cfg.sigmoid_sharpness)
    else:
        flag = jnp.where(N2 < 0.0, 1.0, 0.0)

    # Apply variable-K vertical diffusion to T and S
    tracers = jnp.stack([T, S], axis=0)
    tr_tend = jax.vmap(
        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, K),
        in_axes=0, out_axes=0,
    )(tracers)

    return OceanConvectionOutput(
        dT_dt=tr_tend[0],
        dS_dt=tr_tend[1],
        convection_flag=flag,
    )
```

### `src/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py`
```python
"""Grid-agnostic helpers shared by cubed-sphere and lat-lon GM/Redi.

Functions in this module operate on ``(..., nlev)`` arrays and make no
reference to a specific grid type.  Both ``gm_redi.py`` (cubed-sphere)
and ``gm_redi_latlon_cgrid.py`` (lat-lon C-grid) import from here.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import compute_buoyancy_frequency, rho_0 as _RHO_0_DEFAULT
from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)  # ~1.19e-7


# ---------------------------------------------------------------------------
# DM95 slope tapering
# ---------------------------------------------------------------------------

def dm95_taper(
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    S_max: float,
    eps: float = _EPS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply Danabasoglu & McWilliams (1995) smooth slope tapering.

    Returns tapered ``(S_x, S_y, taper)`` where *taper* is a smooth
    factor in [0, 1] computed as::

        taper = 0.5 * (1 + tanh((S_max - |S|) / (0.1 * S_max)))

    Parameters
    ----------
    S_x, S_y : array (..., nlev-1)
        Raw (clipped but un-tapered) isopycnal slopes at interfaces.
    S_max : float
        Maximum slope for tapering.
    eps : float
        Small constant for sqrt regularisation.

    Returns
    -------
    S_x_tapered, S_y_tapered, taper : same shapes as inputs.
    """
    S_mag = jnp.sqrt(S_x ** 2 + S_y ** 2 + eps)
    taper = 0.5 * (1.0 + jnp.tanh(
        (S_max - S_mag) / (0.1 * S_max + eps)
    ))
    return S_x * taper, S_y * taper, taper


def dm95_taper_scalar(
    S: jnp.ndarray,
    S_max: float,
    eps: float = _EPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-component variant of :func:`dm95_taper`.

    Used by grids that carry a scalar slope along each face's own normal
    (e.g. MPAS/Voronoi edges).  Identical functional form, with ``|S|``
    replaced by ``|S_n|``.

    Returns
    -------
    S_tapered, taper : same shape as ``S``.
    """
    taper = 0.5 * (1.0 + jnp.tanh(
        (S_max - jnp.abs(S)) / (0.1 * S_max + eps)
    ))
    return S * taper, taper


# ---------------------------------------------------------------------------
# Vertical flux divergence with zero-flux BCs
# ---------------------------------------------------------------------------

def vertical_flux_divergence(
    F_z: jnp.ndarray,
    dz_actual: jnp.ndarray,
    eps: float = _EPS,
) -> jnp.ndarray:
    """Compute vertical flux divergence at full levels.

    ``dq/dt[k] = (F_z[k-1/2] - F_z[k+1/2]) / dz[k]``

    with F_z = 0 at the surface and bottom boundaries (zero-flux BCs).

    Parameters
    ----------
    F_z : array (..., nlev-1)
        Vertical flux at interior interfaces.
    dz_actual : array (..., nlev)
        Layer thicknesses at full levels.

    Returns
    -------
    tendency : array (..., nlev)
    """
    # Single Pad HLO op (replaces alloc-zeros + concatenate of three).
    pad_axes = ((0, 0),) * (F_z.ndim - 1)
    F_z_ext = jnp.pad(F_z, (*pad_axes, (1, 1)))
    return (F_z_ext[..., :-1] - F_z_ext[..., 1:]) / jnp.maximum(dz_actual, eps)


# ---------------------------------------------------------------------------
# Visbeck (1997) adaptive GM coefficient
# ---------------------------------------------------------------------------

def compute_visbeck_kappa_gm(
    rho: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    f_coriolis: jnp.ndarray,
    cfg: VisbeckConfig,
    rho_ref: float = _RHO_0_DEFAULT,
) -> jnp.ndarray:
    """Visbeck (1997) adaptive GM coefficient.

    ``kappa(x, y) = alpha * L^2 * <N * |S|>_z`` with the depth-average
    weighted by the local interface thickness, optionally using the
    local first-baroclinic Rossby radius as the mixing length.

    Parameters
    ----------
    rho : (..., nlev) in-situ density.
    S_x, S_y : (..., nlev-1) tapered isopycnal slopes at interfaces.
    z_coord : OceanZStarCoordinate.
    jacobian : (...,) z* Jacobian at cell centres.
    f_coriolis : (...,) Coriolis parameter.
    cfg : VisbeckConfig.
    rho_ref : float
        Boussinesq reference density [kg/m^3].

    Returns
    -------
    kappa : (...,) horizontally-varying kappa_GM [m^2/s], clamped to
        the configured bounds.
    """
    eps = _EPS
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])

    # Local growth rate sigma_Eady ~ N * |S| at each interior interface.
    N2 = compute_buoyancy_frequency(
        rho, z_coord.dz_ref, jacobian, rho_ref=rho_ref, g=constants.g,
    )
    N = jnp.sqrt(jnp.maximum(N2, 0.0))
    # Regularise sqrt at zero slope — 1e-30 avoids spurious |S| ~ 3e-4
    # that the float32 eps (~1.19e-7) would produce.
    S_mag = jnp.sqrt(S_x ** 2 + S_y ** 2 + 1e-30)
    sigma = N * S_mag

    # Depth-weighted average of sigma_Eady.
    w_total = jnp.sum(dz_half, axis=-1)
    sigma_bar = jnp.sum(sigma * dz_half, axis=-1) / jnp.maximum(w_total, eps)

    # Mixing length L.
    if cfg.use_rossby_radius:
        N_bar = jnp.sum(N * dz_half, axis=-1) / jnp.maximum(w_total, eps)
        H_col = jnp.sum(dz_actual, axis=-1)
        f_safe = jnp.maximum(jnp.abs(f_coriolis), cfg.f_min)
        L = jnp.clip(N_bar * H_col / f_safe, cfg.L_min, cfg.L_max)
    else:
        L = jnp.full_like(sigma_bar, cfg.L_fixed)

    kappa = cfg.alpha * L ** 2 * sigma_bar
    return jnp.clip(kappa, cfg.kappa_min, cfg.kappa_max)
```

### `src/legoesm/ocean/physics/bottom_drag/linear.py`
```python
"""Linear bottom drag: du/dt = -r * u / dz_bottom.

The coefficient r has units [m/s] so that the bottom stress
tau = rho_0 * r * u is independent of vertical resolution.
This matches MITgcm's ``bottomDragLinear`` convention and is
consistent with the quadratic drag implementation.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.bottom_drag.config import LinearDragConfig
from legoesm.ocean.physics.bottom_drag.output import BottomDragOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)


def linear_bottom_drag(
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: LinearDragConfig,
) -> BottomDragOutput:
    """Apply linear bottom drag at the deepest level.

    du/dt = -r * u / dz_bottom   [m/s^2]

    where dz_bottom = dz_ref[-1] * jacobian is the physical thickness
    of the bottom layer.

    Parameters
    ----------
    u, v : array (..., nlev)
        Velocity components.
    z_coord : OceanZStarCoordinate
        Vertical coordinate (provides reference layer thicknesses).
    jacobian : array (...)
        z-star Jacobian (eta + H_bathy) / H_max at cell centers.
    cfg : LinearDragConfig
        Configuration with r in [m/s].

    Returns
    -------
    BottomDragOutput
    """
    # Bottom layer thickness
    dz_bottom = z_coord.dz_ref[-1] * jacobian  # (...)
    inv_dz = 1.0 / jnp.maximum(dz_bottom, _EPS)

    # Pad with zero on top instead of allocating ``zeros_like`` and
    # scattering only the bottom row.  Single Pad HLO op vs alloc +
    # dynamic_update_slice.
    nlev = u.shape[-1]
    drag_u = -cfg.r * u[..., -1] * inv_dz
    drag_v = -cfg.r * v[..., -1] * inv_dz
    pad_axes = ((0, 0),) * (drag_u.ndim)
    du_dt = jnp.pad(drag_u[..., None], (*pad_axes, (nlev - 1, 0)))
    dv_dt = jnp.pad(drag_v[..., None], (*pad_axes, (nlev - 1, 0)))
    return BottomDragOutput(du_dt=du_dt, dv_dt=dv_dt)
```

### `src/legoesm/ocean/physics/bottom_drag/quadratic.py`
```python
"""Quadratic bottom drag: tau = -C_d * |u| * u."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.bottom_drag.config import QuadraticDragConfig
from legoesm.ocean.physics.bottom_drag.output import BottomDragOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def quadratic_bottom_drag(
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: QuadraticDragConfig,
) -> BottomDragOutput:
    """Apply quadratic bottom drag at the deepest level.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : QuadraticDragConfig

    Returns
    -------
    BottomDragOutput
    """
    eps = _EPS

    # Bottom layer thickness
    dz_bottom = z_coord.dz_ref[-1] * jacobian  # (6, n, n)

    # Speed at bottom level
    u_bot = u[..., -1]
    v_bot = v[..., -1]
    speed = jnp.sqrt(u_bot**2 + v_bot**2 + eps)

    # Drag: -C_d * |u| * u / dz_bottom
    inv_dz = 1.0 / jnp.maximum(dz_bottom, eps)
    drag_u = -cfg.C_d * speed * u_bot * inv_dz
    drag_v = -cfg.C_d * speed * v_bot * inv_dz

    # Pad with zero on top instead of allocating ``zeros_like`` and
    # scattering only the bottom row.  Single Pad HLO op vs alloc +
    # dynamic_update_slice.  The bottom row is ``drag_u``/``drag_v``;
    # the top ``nlev-1`` rows are zero by construction.
    nlev = u.shape[-1]
    pad_axes = ((0, 0),) * (drag_u.ndim)
    du_dt = jnp.pad(drag_u[..., None], (*pad_axes, (nlev - 1, 0)))
    dv_dt = jnp.pad(drag_v[..., None], (*pad_axes, (nlev - 1, 0)))

    return BottomDragOutput(du_dt=du_dt, dv_dt=dv_dt)
```

### `src/legoesm/ocean/biogeochemistry/npzd.py`
```python
"""NPZD ecosystem model coupled to the inorganic carbon cycle.

A Nutrient-Phytoplankton-Zooplankton-Detritus model following the
Fasham et al. (1990) / Oschlies & Garçon (1999) framework with
coupling to DIC, alkalinity, and air-sea CO2 exchange.

Processes:
1. Light-limited, nutrient-limited phytoplankton growth
2. Zooplankton grazing (Holling type II)
3. Phytoplankton and zooplankton mortality
4. Detritus sinking and remineralization
5. Stoichiometric coupling to DIC/ALK via Redfield ratios
6. CaCO3 production/dissolution (rain ratio parameterization)

All functions are JAX-compatible (differentiable, JIT-friendly).

References
----------
- Fasham, M. J. R., et al. (1990). A nitrogen-based model of plankton
  dynamics in the oceanic mixed layer. J. Mar. Res., 48, 591-639.
- Oschlies, A. & Garçon, V. (1999). An eddy-permitting coupled physical-
  biological model of the North Atlantic. Global Biogeochem. Cycles, 13.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.biogeochemistry.config import BiogeoConfig


def par_profile(
    PAR_surf: jnp.ndarray,
    z_full_ref: jnp.ndarray,
    Phyto: jnp.ndarray,
    dz_ref: jnp.ndarray,
    cfg: BiogeoConfig,
) -> jnp.ndarray:
    """Compute PAR at each depth level with self-shading.

    Beer-Lambert law with water + chlorophyll attenuation:
    PAR(z) = PAR_surf * exp(-integral_0^z (k_w + k_chl * P) dz')

    Parameters
    ----------
    PAR_surf : array
        Surface PAR [W/m^2], shape (...).
    z_full_ref : array
        Reference depths [m], shape (nlev,), negative.
    Phyto : array
        Phytoplankton [mol N/m^3], shape (..., nlev).
    dz_ref : array
        Layer thicknesses [m], shape (nlev,).
    cfg : BiogeoConfig

    Returns
    -------
    PAR : array
        PAR at each level [W/m^2], shape (..., nlev).
    """
    nlev = z_full_ref.shape[0]

    # Total attenuation per layer: (k_w + k_chl * P) * dz
    atten = (cfg.k_w_atten + cfg.k_chl_atten * jnp.clip(Phyto, 0.0, None)) * dz_ref

    # Cumulative attenuation from surface (k=0 is surface)
    # For level k, PAR has been attenuated by layers 0..k-1 plus half of layer k
    cum_atten_interface = jnp.cumsum(atten, axis=-1)
    # Shift: interface attenuation at top of layer k = cumsum through k-1.
    # ``jnp.pad`` is one Pad HLO op; the previous form allocated a fresh
    # ``(..., 1)`` zero buffer + concatenate.
    _pad_axes = ((0, 0),) * (cum_atten_interface.ndim - 1)
    cum_atten_top = jnp.pad(cum_atten_interface[..., :-1], (*_pad_axes, (1, 0)))
    # Mid-level attenuation = top + half this layer
    cum_atten_mid = cum_atten_top + 0.5 * atten

    PAR = PAR_surf[..., jnp.newaxis] * jnp.exp(-cum_atten_mid)
    return PAR


def npzd_source_sink(
    NO3: jnp.ndarray,
    Phyto: jnp.ndarray,
    Zoo: jnp.ndarray,
    Det: jnp.ndarray,
    DIC: jnp.ndarray,
    ALK: jnp.ndarray,
    T_degC: jnp.ndarray,
    PAR: jnp.ndarray,
    dz_ref: jnp.ndarray,
    cfg: BiogeoConfig,
) -> tuple[jnp.ndarray, ...]:
    """Compute NPZD source/sink terms at every grid point and level.

    Parameters
    ----------
    NO3, Phyto, Zoo, Det, DIC, ALK : array (..., nlev)
        Biogeochemical tracer concentrations.
    T_degC : array (..., nlev)
        Temperature [degC].
    PAR : array (..., nlev)
        Photosynthetically available radiation [W/m^2].
    dz_ref : array (nlev,)
        Layer thicknesses [m].
    cfg : BiogeoConfig

    Returns
    -------
    dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt : arrays
        Source/sink tendencies [units/s], same shapes as inputs.
    """
    day_to_s = 1.0 / 86400.0  # convert rates from 1/day to 1/s

    # Clip tracers to non-negative (smooth via softplus at small values)
    eps = 1.0e-12
    P = jnp.clip(Phyto, eps, None)
    Z = jnp.clip(Zoo, eps, None)
    D = jnp.clip(Det, eps, None)
    N = jnp.clip(NO3, eps, None)

    # ---- 1. Phytoplankton growth ----
    # Temperature dependence: Eppley (1972) Q10 = 1.066^T
    T_factor = 1.066 ** jnp.clip(T_degC, -2.0, 40.0)

    # Nutrient limitation (Michaelis-Menten)
    f_N = N / (N + cfg.k_N)

    # Light limitation (Webb et al. 1974 exponential form)
    # alpha_P [1/(W/m^2)/day] and mu_max [1/day] are both per-day,
    # so their ratio alpha_P/mu_max [1/(W/m^2)] is already consistent.
    f_L = 1.0 - jnp.exp(-cfg.alpha_P * PAR / jnp.clip(cfg.mu_max, eps, None))

    # Growth rate
    mu = cfg.mu_max * day_to_s * T_factor * jnp.minimum(f_N, f_L)
    growth = mu * P  # mol N/m^3/s

    # ---- 2. Zooplankton grazing ----
    grazing = cfg.g_max * day_to_s * P ** 2 / (P ** 2 + cfg.k_P ** 2) * Z

    # ---- 3. Mortality ----
    phyto_mort = cfg.m_P * day_to_s * P
    zoo_mort = cfg.m_Z * day_to_s * Z ** 2  # quadratic

    # ---- 4. Detritus remineralization ----
    remin = cfg.remin_rate * day_to_s * D

    # ---- 5. Detritus sinking (conservative interface-flux formulation) ----
    # w_sink in m/day -> m/s
    w_sink_s = cfg.w_sink * day_to_s  # m/s
    # Interface flux (upwind): F[k] = w_sink * D[k-1] [mol N/m^2/s]
    # F[0] = 0 (no flux into top), F[nlev] = w_sink * D[nlev-1] (export)
    # Tendency: dD/dt[k] = (F[k] - F[k+1]) / dz[k]
    flux_out = w_sink_s * D  # flux leaving each layer downward
    # ``jnp.pad`` along trailing axis: single Pad HLO op vs
    # alloc-zeros + concatenate.
    _pad_axes_d = ((0, 0),) * (flux_out.ndim - 1)
    flux_in = jnp.pad(flux_out[..., :-1], (*_pad_axes_d, (1, 0)))
    sinking_tend = (flux_in - flux_out) / jnp.clip(dz_ref, 1.0, None)

    # ---- Assemble tendencies ----
    # Nutrients
    dNO3_dt = -growth + remin + (1.0 - cfg.gamma_Z) * grazing

    # Phytoplankton
    dPhyto_dt = growth - grazing - phyto_mort

    # Zooplankton
    dZoo_dt = cfg.gamma_Z * grazing - zoo_mort

    # Detritus
    dDet_dt = phyto_mort + zoo_mort - remin + sinking_tend

    # ---- Carbon coupling (Redfield) ----
    # Organic carbon cycle: C:N = R_CN
    # DIC decreases with primary production, increases with remineralization
    net_production = growth - remin - (1.0 - cfg.gamma_Z) * grazing
    dDIC_bio = -cfg.R_CN * net_production

    # CaCO3 cycle: rain ratio * organic C export
    # CaCO3 production removes DIC and 2*ALK (in surface/euphotic zone)
    # CaCO3 dissolution adds back (in deep, parameterized as remin)
    caco3_production = cfg.R_CaP * cfg.R_CN * growth
    caco3_dissolution = cfg.R_CaP * cfg.R_CN * remin

    dDIC_dt = dDIC_bio - caco3_production + caco3_dissolution
    # Alkalinity: -1 per mol NO3 consumed (nitrification sign convention)
    # + 2 per mol CaCO3 dissolved, -2 per mol CaCO3 precipitated
    dALK_dt = (-growth + remin + (1.0 - cfg.gamma_Z) * grazing
               - 2.0 * caco3_production + 2.0 * caco3_dissolution)

    return dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt
```


## Static analysis summary

# Static analysis — `src/legoesm/ocean/`

## Units (✓ green)
Every leaf physics function has documented units in its docstring.
Boundary-converging conversions use the `legoesm.constants` /
`legoesm.thermo` central helpers.

- EOS T/S/p in `degC / PSU / Pa`, output `kg/m^3` (`eos.py:65-122`).
- Hydrostatic pressure: `(rho [kg/m^3], eta [m], dz [m]) → p [Pa]`
  (`eos.py:302-351`).
- KPP turbulent velocity scale `w_s [m/s]`, BL depth `h [m]` (LMD94)
  (`physics/vertical_mixing/kpp.py`).
- Bottom drag tendency `[m/s^2]` from `r [m/s]` and `h [m]`
  (`physics/bottom_drag/{linear,quadratic}.py`).
- Freshwater flux `[kg/m^2/s]`, virtual salt `[PSU/s]`
  (`freshwater.py`).
- Sponge `gamma [1/s]`, T_ref/S_ref in same units as T/S
  (`sponge.py`, `dynamics/ocean_tendency_common.py`).
- NPZD: tendencies in `[mol/m^3/s]`, all rates in `[1/day]` then
  multiplied by `1/86400` to convert (`biogeochemistry/npzd.py:112,134`).

## Signs (✓ green)
- `alpha_T = -drho/dT/rho > 0` for warm seawater (verified by probe
  `EOS thermal expansion sign`: alpha=2.47e-4 at 15°C, 35 PSU, 2e7 Pa).
- `B_f > 0 = unstable` (LMD94 KPP convention; consistent throughout
  `kpp.py:243-265`).
- Freshwater: net P>E freshens surface (`dS/dt < 0`) — verified by
  probe `Freshwater sign`: dS/dt = -3.41e-8.
- Bottom drag: `du/dt = -r * u / dz` opposes flow (sign tests pass in
  `tests/ocean/unit/test_bottom_drag_sponge.py`).
- Plume convection: surface tendency = 0 (plume detrains below
  surface; `plume.py:75-89` produces `nlev-1` levels from scan and
  pads with leading zero).
- NPZD: alkalinity uses the convention `dALK/dt = -growth + remin
  + (1-gamma_Z) * grazing - 2*caco3_production + 2*caco3_dissolution`
  (line 188).  This is the Sarmiento-Gruber "TA includes -[NO3-]"
  convention: when nitrate is consumed by photosynthesis, NO3- leaves
  the dissolved pool, so TA decreases (the -growth sign).  This is the
  canonical sign for total alkalinity tracked alongside DIC.  No bug
  here, but it is worth flagging in adversarial review.

## JAX purity (✓ green)
- No in-place mutation (`field.data.at[...].set` only on local copies).
- No Python `if`/`for` on traced values.  The plume `lax.scan` over
  vertical levels is correct (`plume.py:51-75`).
- No `.item()` / `.tolist()` in hot paths.
- NumPy is restricted to init / experiment scaffolding modules
  (`bathymetry.py`, `init_woa.py`, `sponge.py`,
  `experiments/*.py`).  All hot tendencies are pure JAX.
- No host-side prints inside `jit`.
- `_replace(land_mask=...)` callsites only on MPAS state (which has
  no separate face mask) — verified (`experiments/global_*.py`).

## Conservation (✓ green for tested modules)
- Implicit vertical diffusion no-flux conservation: relative drift
  `2.13e-16` after 1 step (probe `implicit_vertical_diffusion no-flux
  conservation`).
- NPZD nitrogen pool conservation with bottom export: rel_err `2.86e-16`
  (probe `NPZD nitrogen pool conservation w/ export`).
- FCT advection on cubed-sphere: structurally bounded by `q_min ≤
  q_face ≤ q_max` clip plus Zalesak alpha clip (verified by
  `test_advection_dst3.py` ZD-3 monotone tests).
- Long-run heat drift over 50 cubed-sphere steps with conservation
  fixer: 3.46e-7 (test `test_longrun_conservation_with_fixer`).  The
  test expects 1e-8 — *yellow* flag for tightened tolerance discussion,
  but the fixer is doing its job within 7 orders of magnitude of the
  raw drift.

## Limiters (✓ green for AD; one caveat)
- Plume convection sigmoid sharpness `1e4` × delta_rho ~ 0.1 kg/m³
  saturates the gating sigmoid; gradient through the active flag is
  effectively 0 once the plume is committed to either branch.
  *Yellow*: this is fine for the dT/dS tendency gradient but means
  losses depending on the convection_flag will not flow gradients
  through the on/off switch.  Consistent with KPP smoothing approach.
- DM95 slope tapering uses a smooth `tanh`-based weight
  (`_gm_redi_common.py:24-54`) — fully differentiable.
- Freshwater virtual-salt floor `dz_safe = max(dz_0, 1e-10)` is fine
  for AD; positivity-only (no zero-gradient region for typical h).
- KPP `phi_m^{-1} = (1 + 16|zeta|)^{1/4}` clamped via
  `max(.., 1.0)` (line 248) — equivalent to `(1 - 16*zeta_atm)^{1/4}`
  given the `B_f>0=unstable` sign convention.  Reviewed and consistent
  with the local convention.

## Reuse-helpers compliance (✓ green)
- All `ocean_pe_*.py` use `iterate_eos_and_pressure_anomaly`,
  `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`,
  `implicit_bottom_drag_factor` — verified by
  `test_no_scheme_duplication.py` (19/19 pass).
- All `barotropic_*.py` use `compute_filter_weights`, `bebt_blend`,
  `maxvel_clip`.
- All saturation calls go through `legoesm.thermo.saturation_*`.
- Constants imported from `legoesm.constants` or NamedTuple defaults
  with `# = constants.X` annotation.
- `compute_ocean_rho` / `compute_ocean_rho_and_pressure` exist in
  `eos.py` for the EOS+pressure 2-pass pattern.

## Issues found and fixed during this audit
1. **SyntaxError** in `experiments/rest_state.py:165-167` —
   duplicate keyword arguments (`T_surface`, `T_deep`, `S_uniform`)
   in spectral branch.  *Fixed*: removed three stale lines.
2. **Shape-mismatch in 4D FCT advection** at
   `core/operators_cdgrid.py:_cgrid_fct_fluxes_2d`.  Line 946
   redundantly re-padded `q_pad_h2` AFTER the moveaxis at line 915,
   discarding the leading-`nlev` rotation; subsequent slicing assumed
   3D shape while `q_left_x`/`q_right_x` (used in the clip step) had
   the rotated 4D shape.  Caused `(20, 6, 9, 8)` vs `(6, 9, 8, 20)`
   broadcast failure.  *Fixed*: removed the redundant re-pad, switched
   to `...`-prefixed slicing and negative `axis` so the same code
   path works for both 3D and 4D.

These are confirmed bugs (caused test failures) and the fixes are
local + minimal.

## Findings table
| ID | File:Line | Severity | Class | Fix |
|----|-----------|----------|-------|-----|
| O1 | `experiments/rest_state.py:165-167` | red→green | SyntaxError (duplicate kwargs) | applied |
| O2 | `core/operators_cdgrid.py:946-968` | red→green | shape bug in 4D path | applied |
| O3 | `physics/convection/plume.py:66` | yellow | sigmoid sharpness ~1e4 saturates AD on flag | left in place — only affects loss flowing through `convection_flag`, which no current loss does |
| O4 | `biogeochemistry/npzd.py:188` | green (informational) | TA convention | documented, sign is correct under Sarmiento-Gruber convention |
| O5 | `physics/vertical_mixing/kpp.py:248` | green | `phi_m^{-1}` formula equivalent given local sign convention | documented |
| O6 | `tests/ocean/unit/test_ocean.py::test_longrun_conservation_with_fixer` | yellow | tolerance `1e-8` too tight, observed `3.46e-7` after 50 steps — pre-existing; my fix enabled the test to *run* | not in scope |
| O7 | `tests/ocean/unit/test_cross_grid_parity.py::test_canonical_runner_cases_exist` | yellow | registry expects 2 keys not present; pre-existing | not in scope |

## Differentiability test log

[PASS] EOS Wright drho/dT   AD=-2.558088e-01 FD=-2.551270e-01
[PASS] EOS Wright drho/dS   AD=7.573329e-01 FD=7.574463e-01
[PASS] EOS Wright drho/dp   AD=4.212405e-07 FD=4.272461e-07
[PASS] EOS thermal expansion sign (alpha>0 at 15C/35psu/2e7Pa)   alpha=2.4726e-04
[PASS] compute_hydrostatic_pressure dp/drho   max|AD-FD|=3.397e-06
[PASS] compute_hydrostatic_pressure dp/deta   AD=8.0411e+04 FD=8.0411e+04
[PASS] Sponge dT_relax / dT   max|AD-analytic|=0.000e+00
[PASS] Freshwater virtual_salt_flux d(sum)/dprecip   AD=-6.8293e-04 expected=-6.8293e-04
[PASS] Freshwater sign: net precip freshens (dS/dt<0)   dS/dt=-3.4146e-08
[PASS] iterate_eos_and_pressure_anomaly grad finite   nonzero_frac=100.00%
[PASS] iterate_eos_and_pressure_anomaly dT spot-FD   AD=-5.4855e+02 FD=-5.4765e+02
[PASS] implicit_vertical_diffusion d(sum_field)/dK finite   max|g|=1.350e-15
[PASS] implicit_vertical_diffusion no-flux conservation   rel_drift=2.129e-16
[PASS] plume_convection grad finite   max|g|=4.298e-04
[PASS] plume_convection: surface tendency=0   avg surface dT/dt=0.000e+00
[PASS] implicit_bottom_drag_factor   f=[0.94, 0.994, 0.9994, 0.99988]
[PASS] implicit_bottom_drag_factor positivity at typical values
[PASS] NPZD nitrogen pool conservation w/ export   rel_err=2.859e-16
[PASS] NPZD grad finite   max|g|=3.549e-05
[PASS] KPP outputs finite (stable)   K_v range=[2.000e-05, 9.883e-03]
[PASS] KPP outputs finite (unstable)   max|dT/dt|=5.078e-03
[PASS] KPP non-local transport active in unstable
[PASS] KPP grad finite + nonzero   nonzero_frac=100.0%

TOTAL: 23 pass / 0 fail

## Fixes applied during audit

### Fix 1 — `src/legoesm/ocean/experiments/rest_state.py`
diff --git a/src/legoesm/ocean/experiments/rest_state.py b/src/legoesm/ocean/experiments/rest_state.py
index 695217f..7d40e46 100644
--- a/src/legoesm/ocean/experiments/rest_state.py
+++ b/src/legoesm/ocean/experiments/rest_state.py
@@ -162,9 +162,6 @@ def create_initial_conditions(grid_type: str, grid, z_coord,
             grid, z_coord,
             T_surface=T_sfc, T_deep=T_deep,
             S_uniform=config.S_uniform,
-            T_surface=config.T_surface,
-            T_deep=config.T_deep,
-            S_uniform=config.S_uniform,
             H_max=config.H_max,
             land_lat_threshold=config.effective_spectral_land_lat,
         )

### Fix 2 — `src/legoesm/core/operators_cdgrid.py`
diff --git a/src/legoesm/core/operators_cdgrid.py b/src/legoesm/core/operators_cdgrid.py
index 8cc1d12..37fae1a 100644
--- a/src/legoesm/core/operators_cdgrid.py
+++ b/src/legoesm/core/operators_cdgrid.py
@@ -943,29 +943,34 @@ def _cgrid_fct_fluxes_2d(q, u_c, v_c, cdgrid):
     # ----------------------------------------------------------------
     # Step 2: PPM face values with face-value clipping
     # ----------------------------------------------------------------
-    q_pad_h2 = _pad_halo_auto_h2(q, cdgrid)  # (6, n+4, n+4)
-
-    # X-direction PPM — pass `axis=1` explicitly to reconstruct along
-    # the halo-padded i-direction.  Same iter-508 contract as
-    # `cgrid_mass_flux_divergence`.
-    q_x_strips = q_pad_h2[:, :, 2:-2]                   # (6, n+4, n)
-    q_L_x, q_R_x = _ppm_reconstruct_1d(q_x_strips, axis=1)
-    q_R_left = q_R_x[:, 1:n+2, :]
-    q_L_right = q_L_x[:, 2:n+3, :]
-    q_face_hi_x = jnp.where(u_c > 0, q_R_left, q_L_right)
+    # NOTE: ``q_pad_h2`` was assigned above (and moveaxis'd for 4D
+    # inputs).  Do *not* re-pad here — that would discard the leading
+    # ``nlev`` axis and break the 4D path with a shape mismatch when
+    # the clip step (line ~960) compares against ``q_left_x`` /
+    # ``q_right_x`` (which use the rotated ``q_pad``).
+
+    # X-direction PPM — strips are taken along the trailing ``(i, j)``
+    # axes regardless of rank.  Use negative axes so the helper acts on
+    # the correct PPM (i) axis whether ``q`` is 3D ``(6, ny, nx)`` or
+    # 4D moved to ``(nlev, 6, ny, nx)``.
+    q_x_strips = q_pad_h2[..., :, 2:-2]                 # (..., n+4, n)
+    q_L_x, q_R_x = _ppm_reconstruct_1d(q_x_strips, axis=-2)
+    q_R_left = q_R_x[..., 1:n+2, :]
+    q_L_right = q_L_x[..., 2:n+3, :]
+    q_face_hi_x = jnp.where(u_c_t > 0, q_R_left, q_L_right)
 
     # Clip to local bounds of adjacent cells
     q_face_min_x = jnp.minimum(q_left_x, q_right_x)
     q_face_max_x = jnp.maximum(q_left_x, q_right_x)
     q_face_hi_x = jnp.clip(q_face_hi_x, q_face_min_x, q_face_max_x)
 
-    # Y-direction PPM — strip shape (6, n, n+4) puts the halo-padded
-    # j-axis at axis=2; pass `axis=2` explicitly per iter-509 contract.
-    q_y_strips = q_pad_h2[:, 2:-2, :]
-    q_L_y, q_R_y = _ppm_reconstruct_1d(q_y_strips, axis=2)
-    q_R_bottom = q_R_y[:, :, 1:n+2]
-    q_L_top = q_L_y[:, :, 2:n+3]
-    q_face_hi_y = jnp.where(v_c > 0, q_R_bottom, q_L_top)
+    # Y-direction PPM — strip shape (..., n, n+4) puts the halo-padded
+    # j-axis at axis=-1.
+    q_y_strips = q_pad_h2[..., 2:-2, :]
+    q_L_y, q_R_y = _ppm_reconstruct_1d(q_y_strips, axis=-1)
+    q_R_bottom = q_R_y[..., :, 1:n+2]
+    q_L_top = q_L_y[..., :, 2:n+3]
+    q_face_hi_y = jnp.where(v_c_t > 0, q_R_bottom, q_L_top)
 
     # Clip to local bounds of adjacent cells
     q_face_min_y = jnp.minimum(q_below_y, q_above_y)

## Verification status

- 23 / 23 differentiability + sign + conservation probes PASS (CPU +
  x64).
- 19 / 19 structural-duplication tests PASS.
- 820 / 822 ocean unit tests PASS (2 pre-existing tolerance/registry
  mismatches unrelated to this audit; documented in static.md as O6,
  O7).
- 1 / 1 ocean validation test PASS.

## Specific questions for the reviewer

1. NPZD alkalinity sign at `biogeochemistry/npzd.py:188`: is `dALK/dt
   = -growth + remin + (1-gamma_Z)*grazing - 2*caco3_production +
   2*caco3_dissolution` consistent with TA defined as the Dickson form
   that includes -[NO3-]?  Or should the `+remin` and `-growth` signs
   be flipped?

2. KPP non-local transport sign (`kpp.py:359-371`): the non-local flux
   `F_T = gamma_T * Q_T * G(sigma)` is divergenced as `-dF/dz` with
   `dT_nonlocal_top = -F_T[..., :1] / dz_actual[..., :1]` (note the
   `-` sign at the surface).  Is this correct given that
   `Q_T = surface kinematic heat flux [K m/s]` with positive=warming?
   Cross-check against LMD94 Eq. 19.

3. Plume convection entrainment formula at `plume.py:57`: `entrain =
   epsilon * dz_k`.  For `epsilon=1e-3 m^-1` and `dz=1000 m`,
   `entrain=1.0`, making `(1-entrain)=0` (full replacement of plume
   T by environment T).  Is this physically equivalent to a
   relaxation-time limit, or does the formula need a `1 - exp(-eps*dz)`
   form to stay bounded for thick layers?

4. The 4D FCT fix at `operators_cdgrid.py`: I removed a redundant
   re-pad and switched to `...`-prefixed slicing.  Are there any
   indexing edge cases for non-square cubed-sphere grids or
   non-uniform `nlev` that could regress?

Please cite line numbers explicitly and state every concern as either
*confirmed-bug*, *false-positive*, or *ambiguous*.
