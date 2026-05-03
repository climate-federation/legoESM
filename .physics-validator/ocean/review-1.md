Reading prompt from stdin...
OpenAI Codex v0.125.0 (research preview)
--------
workdir: /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
model: gpt-5.3-codex
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, /Users/pierregentine/.codex/memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019de8c2-2f27-7250-9845-7cbdaedc9d05
--------
user
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

codex
I’ll run an independent pass over `src/legoesm/ocean/` and validate your four flagged questions plus look for additional confirmed physics/units/gradient/conservation issues, with exact file+line citations from the current tree. I’m starting by reading the live files and cross-checking key formulas against their surrounding call paths.
web search: 
web search: 
exec
/bin/zsh -lc 'pwd && git status --short' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 M src/legoesm/core/operators_cdgrid.py
 M src/legoesm/ocean/experiments/rest_state.py
?? .physics-validator/ocean/

codex
I found the tree is dirty in exactly the files you said were patched (`operators_cdgrid.py`, `rest_state.py`), so I’ll treat those as intentional and proceed read-only. Next I’m pulling line-accurate views of the audited physics files and then I’ll run targeted conservation/gradient spot-check scripts.
exec
/bin/zsh -lc 'rg --files src/legoesm/ocean | wc -l && rg --files src/legoesm/ocean | head -n 40' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     108
src/legoesm/ocean/sponge.py
src/legoesm/ocean/init_latlon_cgrid.py
src/legoesm/ocean/bathymetry.py
src/legoesm/ocean/state.py
src/legoesm/ocean/advection.py
src/legoesm/ocean/init.py
src/legoesm/ocean/physics/combined.py
src/legoesm/ocean/physics/surface_forcing/output.py
src/legoesm/ocean/physics/surface_forcing/wind_profiles.py
src/legoesm/ocean/physics/surface_forcing/integration.py
src/legoesm/ocean/physics/surface_forcing/prescribed.py
src/legoesm/ocean/physics/surface_forcing/__init__.py
src/legoesm/ocean/physics/surface_forcing/config.py
src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py
src/legoesm/ocean/physics/surface_forcing/restoring.py
src/legoesm/ocean/physics/lateral_mixing/output.py
src/legoesm/ocean/physics/lateral_mixing/harmonic.py
src/legoesm/ocean/physics/lateral_mixing/gm_redi_mpas.py
src/legoesm/ocean/physics/lateral_mixing/biharmonic.py
src/legoesm/ocean/physics/lateral_mixing/integration.py
src/legoesm/ocean/physics/lateral_mixing/gm_redi.py
src/legoesm/ocean/physics/lateral_mixing/backscatter.py
src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py
src/legoesm/ocean/physics/lateral_mixing/__init__.py
src/legoesm/ocean/physics/lateral_mixing/config.py
src/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py
src/legoesm/ocean/physics/mixing.py
src/legoesm/ocean/physics/mpas_physics.py
src/legoesm/ocean/physics/shortwave_penetration.py
src/legoesm/ocean/physics/convection/output.py
src/legoesm/ocean/physics/convection/enhanced_diffusion.py
src/legoesm/ocean/physics/convection/plume.py
src/legoesm/ocean/physics/convection/integration.py
src/legoesm/ocean/physics/convection/__init__.py
src/legoesm/ocean/physics/convection/config.py
src/legoesm/ocean/physics/__init__.py
src/legoesm/ocean/physics/vertical_mixing/kpp.py
src/legoesm/ocean/physics/vertical_mixing/output.py
src/legoesm/ocean/physics/vertical_mixing/constant.py
src/legoesm/ocean/physics/vertical_mixing/richardson.py

exec
/bin/zsh -lc 'wc -l src/legoesm/core/operators_cdgrid.py src/legoesm/ocean/experiments/rest_state.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'wc -l src/legoesm/ocean/eos.py src/legoesm/ocean/dynamics/ocean_tendency_common.py src/legoesm/ocean/dynamics/barotropic_common.py src/legoesm/ocean/freshwater.py src/legoesm/ocean/sponge.py src/legoesm/ocean/init_latlon_cgrid.py src/legoesm/ocean/physics/vertical_mixing/kpp.py src/legoesm/ocean/physics/vertical_mixing/implicit_solver.py src/legoesm/ocean/physics/convection/plume.py src/legoesm/ocean/physics/convection/enhanced_diffusion.py src/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py src/legoesm/ocean/physics/bottom_drag/linear.py src/legoesm/ocean/physics/bottom_drag/quadratic.py src/legoesm/ocean/biogeochemistry/npzd.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
    2515 src/legoesm/core/operators_cdgrid.py
     366 src/legoesm/ocean/experiments/rest_state.py
    2881 total

 succeeded in 0ms:
     454 src/legoesm/ocean/eos.py
     279 src/legoesm/ocean/dynamics/ocean_tendency_common.py
      89 src/legoesm/ocean/dynamics/barotropic_common.py
     232 src/legoesm/ocean/freshwater.py
     130 src/legoesm/ocean/sponge.py
     306 src/legoesm/ocean/init_latlon_cgrid.py
     392 src/legoesm/ocean/physics/vertical_mixing/kpp.py
     175 src/legoesm/ocean/physics/vertical_mixing/implicit_solver.py
      99 src/legoesm/ocean/physics/convection/plume.py
      69 src/legoesm/ocean/physics/convection/enhanced_diffusion.py
     174 src/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py
      62 src/legoesm/ocean/physics/bottom_drag/linear.py
      58 src/legoesm/ocean/physics/bottom_drag/quadratic.py
     191 src/legoesm/ocean/biogeochemistry/npzd.py
    2710 total

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/eos.py | sed -n '1,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Equation of state for seawater: Wright (1997) and linear.
     2	
     3	Provides:
     4	- ``wright_eos`` — nonlinear Wright (1997) EOS (MOM6 implementation)
     5	- ``linear_eos`` — configurable linear EOS: ρ = ρ₀[1 - αT(T-Tref) + βS(S-Sref)]
     6	- ``make_eos_fn`` — dispatcher returning an EOS callable based on config
     7	
     8	Pure JAX functions, compatible with jit/grad/vmap.
     9	
    10	Reference
    11	---------
    12	Wright, D. G. (1997): An Equation of State for Use in Ocean Models:
    13	Ockham's Razor Revisited. J. Atmos. Oceanic Tech., 14(3), 735-740.
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	from typing import NamedTuple
    19	
    20	import jax.numpy as jnp
    21	
    22	from legoesm import constants
    23	from legoesm.core.precision import _resolve_dtype
    24	
    25	# ==============================================================================
    26	# Ocean constants
    27	# ==============================================================================
    28	rho_0 = 1025.0          # Reference seawater density [kg/m^3]
    29	c_sw = 3994.0           # Specific heat of seawater [J/(kg*K)]
    30	T_freeze_ocean = constants.T_freeze_ocean  # re-export from central constants
    31	scale_depth = 1000.0     # Reference e-folding depth for stratification [m]
    32	
    33	# ==============================================================================
    34	# Wright (1997) EOS coefficients — from MOM6 (MOM_EOS_Wright.F90)
    35	# Pressure units: Pa. Temperature: degC. Salinity: PSU.
    36	#
    37	# Formula: rho = (p + p0) / (lambda + al0 * (p + p0))
    38	#   al0(T, S) = a0 + a1*T + a2*S
    39	#   p0(T, S)  = (b0 + b4*S) + T*(b1 + T*(b2 + b3*T) + b5*S)
    40	#   lambda(T, S) = (c0 + c4*S) + T*(c1 + T*(c2 + c3*T) + c5*S)
    41	# ==============================================================================
    42	
    43	# Specific volume coefficients al0(T, S)
    44	_a0 = 7.057924e-4
    45	_a1 = 3.480336e-7
    46	_a2 = -1.112733e-7
    47	
    48	# Pressure offset p0(T, S) [Pa]
    49	_b0 = 5.790749e8
    50	_b1 = 3.516535e6
    51	_b2 = -4.002714e4
    52	_b3 = 2.084372e2
    53	_b4 = 5.944068e5
    54	_b5 = -9.643486e3
    55	
    56	# Lambda(T, S) [m^2/s^2]
    57	_c0 = 1.704853e5
    58	_c1 = 7.904722e2
    59	_c2 = -7.984422
    60	_c3 = 5.140652e-2
    61	_c4 = -2.302158e2
    62	_c5 = -3.079464
    63	
    64	
    65	def wright_eos(
    66	    T: jnp.ndarray,
    67	    S: jnp.ndarray,
    68	    p: jnp.ndarray,
    69	) -> jnp.ndarray:
    70	    """Compute in-situ density from Wright (1997) EOS.
    71	
    72	    Parameters
    73	    ----------
    74	    T : array
    75	        Potential temperature [degC].
    76	    S : array
    77	        Salinity [PSU].
    78	    p : array
    79	        Pressure [Pa]. Use 0 for surface.
    80	
    81	    Returns
    82	    -------
    83	    array : In-situ density [kg/m^3].
    84	
    85	    Notes
    86	    -----
    87	    Intermediate computation is promoted to float64 to avoid precision
    88	    loss from large polynomial coefficients (e.g., _b0 ~ 5.79e8).
    89	    If ``JAX_ENABLE_X64=1`` is not set, the astype calls are no-ops
    90	    (safe but no precision improvement).  ``jnp.astype`` is
    91	    differentiable in JAX.
    92	
    93	    The Wright (1997) polynomial is nominally valid for T in [-2, 40] degC
    94	    and S in [0, 42] PSU, but extrapolates smoothly outside that box.
    95	    Inputs are not clipped: silent clipping would zero gradients at the
    96	    boundary and mask unphysical state from advection overshoots or
    97	    coupler bugs. See issue #165.
    98	    """
    99	    orig_dtype = T.dtype
   100	
   101	    # Promote to the EOS compute dtype (float64 in mixed mode) for
   102	    # intermediate polynomial evaluation.  On backends that lack float64
   103	    # (e.g. Metal), _resolve_dtype silently returns float32.
   104	    hi = _resolve_dtype("equation_of_state", "compute")
   105	    T = T.astype(hi)
   106	    S = S.astype(hi)
   107	    p = p.astype(hi)
   108	
   109	    # Specific volume parameter
   110	    al0 = _a0 + _a1 * T + _a2 * S
   111	
   112	    # Pressure offset
   113	    p0 = (_b0 + _b4 * S) + T * (_b1 + T * (_b2 + _b3 * T) + _b5 * S)
   114	
   115	    # Lambda
   116	    lam = (_c0 + _c4 * S) + T * (_c1 + T * (_c2 + _c3 * T) + _c5 * S)
   117	
   118	    # Density: rho = (p + p0) / (lambda + al0 * (p + p0))
   119	    p_plus_p0 = p + p0
   120	    rho = p_plus_p0 / (lam + al0 * p_plus_p0)
   121	
   122	    return rho.astype(orig_dtype)
   123	
   124	
   125	def _wright_eos_scalar(T: float, S: float, p: float) -> float:
   126	    """Scalar Wright EOS for JAX grad (no dtype promotion).
   127	
   128	    Used internally by ``thermal_expansion_coeff`` and
   129	    ``haline_contraction_coeff`` via ``jax.grad``.
   130	    """
   131	    al0 = _a0 + _a1 * T + _a2 * S
   132	    p0 = (_b0 + _b4 * S) + T * (_b1 + T * (_b2 + _b3 * T) + _b5 * S)
   133	    lam = (_c0 + _c4 * S) + T * (_c1 + T * (_c2 + _c3 * T) + _c5 * S)
   134	    p_plus_p0 = p + p0
   135	    return p_plus_p0 / (lam + al0 * p_plus_p0)
   136	
   137	
   138	# Partial derivatives via JAX autodiff (scalar → vmap for arrays).
   139	import jax
   140	_drho_dT_scalar = jax.grad(_wright_eos_scalar, argnums=0)
   141	_drho_dS_scalar = jax.grad(_wright_eos_scalar, argnums=1)
   142	
   143	
   144	def thermal_expansion_coeff(
   145	    T: jnp.ndarray,
   146	    S: jnp.ndarray,
   147	    p: jnp.ndarray,
   148	) -> jnp.ndarray:
   149	    r"""Thermal expansion coefficient α = -(1/ρ) ∂ρ/∂T.
   150	
   151	    Parameters
   152	    ----------
   153	    T : array — Potential temperature [degC].
   154	    S : array — Salinity [PSU].
   155	    p : array — Pressure [Pa].
   156	
   157	    Returns
   158	    -------
   159	    array : α [1/K], same shape as inputs.
   160	    """
   161	    hi = _resolve_dtype("equation_of_state", "compute")
   162	    T64 = T.astype(hi)
   163	    S64 = S.astype(hi)
   164	    p64 = p.astype(hi)
   165	    flat_T = T64.ravel()
   166	    flat_S = S64.ravel()
   167	    flat_p = p64.ravel()
   168	    drho_dT = jax.vmap(_drho_dT_scalar)(flat_T, flat_S, flat_p).reshape(T.shape)
   169	    rho = wright_eos(T, S, p)
   170	    return (-drho_dT / rho).astype(T.dtype)
   171	
   172	
   173	def haline_contraction_coeff(
   174	    T: jnp.ndarray,
   175	    S: jnp.ndarray,
   176	    p: jnp.ndarray,
   177	) -> jnp.ndarray:
   178	    r"""Haline contraction coefficient β = (1/ρ) ∂ρ/∂S.
   179	
   180	    Parameters
   181	    ----------
   182	    T : array — Potential temperature [degC].
   183	    S : array — Salinity [PSU].
   184	    p : array — Pressure [Pa].
   185	
   186	    Returns
   187	    -------
   188	    array : β [1/PSU], same shape as inputs.
   189	    """
   190	    hi = _resolve_dtype("equation_of_state", "compute")
   191	    T64 = T.astype(hi)
   192	    S64 = S.astype(hi)
   193	    p64 = p.astype(hi)
   194	    flat_T = T64.ravel()
   195	    flat_S = S64.ravel()
   196	    flat_p = p64.ravel()
   197	    drho_dS = jax.vmap(_drho_dS_scalar)(flat_T, flat_S, flat_p).reshape(T.shape)
   198	    rho = wright_eos(T, S, p)
   199	    return (drho_dS / rho).astype(T.dtype)
   200	
   201	
   202	# ==============================================================================
   203	# Linear equation of state
   204	# ==============================================================================
   205	
   206	class LinearEOSConfig(NamedTuple):
   207	    """Configuration for the linear equation of state.
   208	
   209	    ρ = rho_ref * [1 - alpha_T * (T - T_ref) + beta_S * (S - S_ref)]
   210	    """
   211	    rho_ref: float = 1025.0    # Reference density [kg/m³]
   212	    alpha_T: float = 2.0e-4    # Thermal expansion coefficient [1/K]
   213	    beta_S: float = 7.4e-4     # Haline contraction coefficient [1/PSU]
   214	    T_ref: float = 10.0        # Reference temperature [°C]
   215	    S_ref: float = 35.0        # Reference salinity [PSU]
   216	
   217	
   218	def linear_eos(
   219	    T: jnp.ndarray,
   220	    S: jnp.ndarray,
   221	    p: jnp.ndarray,
   222	    rho_ref: float = 1025.0,
   223	    alpha_T: float = 2.0e-4,
   224	    beta_S: float = 7.4e-4,
   225	    T_ref: float = 10.0,
   226	    S_ref: float = 35.0,
   227	) -> jnp.ndarray:
   228	    """Compute density from a linear equation of state.
   229	
   230	    ρ = rho_ref * [1 - alpha_T * (T - T_ref) + beta_S * (S - S_ref)]
   231	
   232	    Parameters
   233	    ----------
   234	    T : array — Potential temperature [°C].
   235	    S : array — Salinity [PSU].
   236	    p : array — Pressure [Pa] (unused, accepted for API compatibility).
   237	    rho_ref : float — Reference density [kg/m³].
   238	    alpha_T : float — Thermal expansion coefficient [1/K].
   239	    beta_S : float — Haline contraction coefficient [1/PSU].
   240	    T_ref : float — Reference temperature [°C].
   241	    S_ref : float — Reference salinity [PSU].
   242	
   243	    Returns
   244	    -------
   245	    array : In-situ density [kg/m³].
   246	    """
   247	    return rho_ref * (1.0 - alpha_T * (T - T_ref) + beta_S * (S - S_ref))
   248	
   249	
   250	def make_eos_fn(eos="wright", eos_linear=None):
   251	    """Return an EOS callable ``fn(T, S, p) -> rho``.
   252	
   253	    Parameters
   254	    ----------
   255	    eos : str
   256	        ``"wright"`` (default) or ``"linear"``.
   257	    eos_linear : LinearEOSConfig or None
   258	        Parameters for linear EOS.  Ignored when *eos* is ``"wright"``.
   259	        If ``None`` and *eos* is ``"linear"``, default parameters are used.
   260	

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/eos.py | sed -n '260,470p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   260	
   261	    Returns
   262	    -------
   263	    Callable[[array, array, array], array]
   264	    """
   265	    if eos == "wright":
   266	        return wright_eos
   267	    elif eos == "linear":
   268	        cfg = eos_linear if eos_linear is not None else LinearEOSConfig()
   269	        def _linear(T, S, p):
   270	            return linear_eos(
   271	                T, S, p,
   272	                rho_ref=cfg.rho_ref, alpha_T=cfg.alpha_T,
   273	                beta_S=cfg.beta_S, T_ref=cfg.T_ref, S_ref=cfg.S_ref,
   274	            )
   275	        return _linear
   276	    else:
   277	        raise ValueError(f"Unknown EOS scheme: {eos!r}")
   278	
   279	
   280	def density_perturbation(
   281	    T: jnp.ndarray,
   282	    S: jnp.ndarray,
   283	    p: jnp.ndarray,
   284	    rho_ref: float = rho_0,
   285	) -> jnp.ndarray:
   286	    """Compute density perturbation rho' = rho(T,S,p) - rho_ref.
   287	
   288	    Parameters
   289	    ----------
   290	    T, S, p : array
   291	        Temperature [degC], salinity [PSU], pressure [Pa].
   292	    rho_ref : float
   293	        Reference density [kg/m^3].
   294	
   295	    Returns
   296	    -------
   297	    array : Density perturbation [kg/m^3].
   298	    """
   299	    return wright_eos(T, S, p) - rho_ref
   300	
   301	
   302	def compute_hydrostatic_pressure(
   303	    rho: jnp.ndarray,
   304	    eta: jnp.ndarray,
   305	    dz: jnp.ndarray,
   306	    jacobian: jnp.ndarray,
   307	    rho_ref: float = rho_0,
   308	    g: float = constants.g,
   309	) -> jnp.ndarray:
   310	    """Compute hydrostatic pressure at full levels.
   311	
   312	    p(z) = rho_ref * g * eta + integral_{z}^{0} rho * g dz'
   313	
   314	    Integrated top-to-bottom (k=0 is surface, k=nlev-1 is deepest).
   315	    Pressure at cell center is the cumulative integral from surface
   316	    down to the midpoint of each layer.
   317	
   318	    Parameters
   319	    ----------
   320	    rho : array
   321	        In-situ density, shape (..., nlev).
   322	    eta : array
   323	        Sea surface height [m], shape (...).
   324	    dz : array
   325	        Reference layer thickness [m], shape (nlev,).
   326	    jacobian : array
   327	        Dynamic Jacobian (eta + H) / H, shape (...).
   328	    rho_ref : float
   329	        Reference density [kg/m^3].
   330	    g : float
   331	        Gravitational acceleration [m/s^2].
   332	
   333	    Returns
   334	    -------
   335	    array : Hydrostatic pressure at full levels [Pa], shape (..., nlev).
   336	    """
   337	    # Surface pressure from free surface
   338	    p_surface = rho_ref * g * eta  # (...,)
   339	
   340	    # Actual layer thickness
   341	    dz_actual = dz * jacobian[..., jnp.newaxis]  # (..., nlev)
   342	
   343	    # Pressure increment per layer: rho * g * dz
   344	    dp = rho * g * dz_actual  # (..., nlev)
   345	
   346	    # Pressure at layer top = cumulative sum from surface
   347	    # p_top[k] = p_surface + sum(dp[0:k])
   348	    p_top = p_surface[..., jnp.newaxis] + jnp.cumsum(dp, axis=-1) - dp
   349	
   350	    # Pressure at cell center = p_top + 0.5 * dp
   351	    return p_top + 0.5 * dp
   352	
   353	
   354	def compute_buoyancy_frequency(
   355	    rho: jnp.ndarray,
   356	    dz: jnp.ndarray,
   357	    jacobian: jnp.ndarray,
   358	    rho_ref: float = rho_0,
   359	    g: float = constants.g,
   360	) -> jnp.ndarray:
   361	    """Compute Brunt-Vaisala frequency N^2.
   362	
   363	    N^2 = -(g / rho_ref) * d(rho) / dz
   364	
   365	    Computed at interior interfaces (nlev-1 values).
   366	
   367	    Parameters
   368	    ----------
   369	    rho : array
   370	        In-situ density, shape (..., nlev).
   371	    dz : array
   372	        Reference layer thickness [m], shape (nlev,).
   373	    jacobian : array
   374	        Dynamic Jacobian, shape (...).
   375	    rho_ref : float
   376	        Reference density [kg/m^3].
   377	    g : float
   378	        Gravitational acceleration [m/s^2].
   379	
   380	    Returns
   381	    -------
   382	    array : N^2 at interior interfaces [1/s^2], shape (..., nlev-1).
   383	    """
   384	    dz_actual = dz * jacobian[..., jnp.newaxis]
   385	    dz_interface = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
   386	
   387	    # drho/dz: rho[k] is shallower than rho[k+1]
   388	    # N^2 = -(g/rho_0) * (rho[k] - rho[k+1]) / dz_interface
   389	    drho_dz = (rho[..., :-1] - rho[..., 1:]) / dz_interface
   390	
   391	    return -(g / rho_ref) * drho_dz
   392	
   393	
   394	# ==============================================================================
   395	# Shared helpers for ocean physics integration modules
   396	# ==============================================================================
   397	
   398	def compute_ocean_rho(state, z_coord, jacobian, eos_fn=None):
   399	    """Compute in-situ density from ocean state.
   400	
   401	    Used by vertical mixing, lateral mixing, and convection integration
   402	    bridges. Avoids triplicating the same hydrostatic pressure + EOS call.
   403	
   404	    Parameters
   405	    ----------
   406	    state : OceanState
   407	        Must have .T, .S, .eta fields.
   408	    z_coord : OceanZStarCoordinate
   409	        Vertical coordinate with .dz_ref.
   410	    jacobian : array
   411	        Dynamic Jacobian (eta + H) / H.
   412	    eos_fn : callable or None
   413	        EOS function ``fn(T, S, p) -> rho``.  If None, uses ``wright_eos``.
   414	
   415	    Returns
   416	    -------
   417	    array : In-situ density [kg/m^3].
   418	    """
   419	    if eos_fn is None:
   420	        eos_fn = wright_eos
   421	    # Two EOS iterations for density-pressure consistency, matching the
   422	    # dynamical core (ocean_pe_cdgrid.py).
   423	    rho = eos_fn(state.T.data, state.S.data, jnp.zeros_like(state.T.data))
   424	    for _ in range(2):
   425	        p_hydro = compute_hydrostatic_pressure(
   426	            rho, state.eta.data, z_coord.dz_ref, jacobian, rho_0,
   427	        )
   428	        rho = eos_fn(state.T.data, state.S.data, p_hydro)
   429	    return rho
   430	
   431	
   432	def compute_ocean_rho_and_pressure(state, z_coord, jacobian, eos_fn=None):
   433	    """Compute in-situ density and hydrostatic pressure from ocean state.
   434	
   435	    Delegates to ``compute_ocean_rho`` for the 2-iteration EOS-pressure
   436	    coupling, then computes a final hydrostatic pressure consistent with
   437	    the converged density.
   438	
   439	    Parameters
   440	    ----------
   441	    state, z_coord, jacobian : same as ``compute_ocean_rho``.
   442	    eos_fn : callable or None
   443	        EOS function. If None, uses ``wright_eos``.
   444	
   445	    Returns
   446	    -------
   447	    rho : array — in-situ density [kg/m^3].
   448	    p_hydro : array — hydrostatic pressure [Pa].
   449	    """
   450	    rho = compute_ocean_rho(state, z_coord, jacobian, eos_fn=eos_fn)
   451	    p_hydro = compute_hydrostatic_pressure(
   452	        rho, state.eta.data, z_coord.dz_ref, jacobian, rho_0,
   453	    )
   454	    return rho, p_hydro

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/dynamics/ocean_tendency_common.py | sed -n '1,320p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Grid-agnostic baroclinic tendency helpers shared by ocean_pe_*.py files.
     2	
     3	This module factors logic that was previously duplicated across the
     4	A-grid (``ocean_pe_cdgrid.py``), lat-lon C-grid
     5	(``ocean_pe_latlon_cgrid.py``), and MPAS Voronoi (``ocean_pe_mpas.py``)
     6	baroclinic tendency entry points.  Grid-specific operators (gradient,
     7	divergence, vorticity, fill, ...) are passed in as callables so this
     8	module never imports from a particular grid package.
     9	
    10	Closes #214 (Phase 1).
    11	
    12	Functions
    13	---------
    14	``iterate_eos_and_pressure_anomaly``
    15	    EOS iteration (T_filled, S_filled, p) → ρ, ρ', p' using a
    16	    reference-thickness hydrostatic integral.  The cubed-sphere caller
    17	    can request high-precision arithmetic for the cumsum so that the
    18	    halo-exchanged p' gradient stays clean.
    19	
    20	``apply_sponge_tracer_relaxation``
    21	    Linear restoring of T, S towards reference fields with rate
    22	    ``γ``.  Used identically by the lat-lon C-grid and MPAS callers.
    23	
    24	``apply_freshwater_virtual_salt_top``
    25	    Top-layer salinity tendency from the freshwater volume flux.
    26	
    27	``implicit_bottom_drag_factor``
    28	    Returns ``1 - dt * r / max(H, eps)`` — the per-substep multiplicative
    29	    factor used by both C-grid lat-lon and MPAS barotropic substeps.
    30	
    31	These helpers are pure and pytree-friendly: they accept and return
    32	``jax.Array`` values and never mutate inputs.
    33	"""
    34	
    35	from __future__ import annotations
    36	
    37	from typing import Callable, Optional, Tuple
    38	
    39	import jax.numpy as jnp
    40	
    41	from legoesm.ocean.eos import compute_hydrostatic_pressure
    42	from legoesm.ocean.freshwater import virtual_salt_flux
    43	
    44	
    45	def iterate_eos_and_pressure_anomaly(
    46	    T: jnp.ndarray,
    47	    S: jnp.ndarray,
    48	    mask: jnp.ndarray,
    49	    fill_fn: Callable[[jnp.ndarray], jnp.ndarray],
    50	    eos_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    51	    dz_ref: jnp.ndarray,
    52	    rho_0: float,
    53	    g: float,
    54	    *,
    55	    n_iter: int = 2,
    56	    hi_precision_pressure: bool = False,
    57	) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    58	    """Run the standard 2-pass EOS iteration and form ``p_prime``.
    59	
    60	    Replicates the identical iteration that previously lived inline in
    61	    every ``ocean_pe_*.py`` file:
    62	
    63	    1. Fill land-cell ``T``, ``S`` with ocean-neighbour values via
    64	       ``fill_fn`` so the EOS does not produce spurious ``ρ'`` values
    65	       on land that contaminate the gradient at coastlines.
    66	    2. Iterate ``ρ ← EOS(T, S, p_hydro(ρ))`` ``n_iter`` times against
    67	       the **reference** thickness profile ``dz_ref`` (i.e. ``J=1``,
    68	       ``η=0``).  Using the actual Jacobian here would double-count
    69	       the ``-g·∇η`` forcing handled by the barotropic solver.
    70	    3. Build the layer-centred baroclinic pressure anomaly
    71	
    72	       ``p'(k) = g · Σ_{j<k} ρ'(j) · dz_ref(j) + 0.5 · g · ρ'(k) · dz_ref(k)``
    73	
    74	       which equals the half-trapezoidal cumulative integral of
    75	       ``g·ρ'`` from the surface to the layer mid-point.
    76	
    77	    Parameters
    78	    ----------
    79	    T, S : jax.Array
    80	        Tracer fields with a trailing vertical axis (``..., nlev``).  All
    81	        upstream callers store T and S with identical shapes.
    82	    mask : jax.Array
    83	        Land mask with the same horizontal shape as ``T[..., 0]`` (used
    84	        only by ``fill_fn``; passed back to the caller for any post-
    85	        processing it needs).
    86	    fill_fn : Callable[[jax.Array], jax.Array]
    87	        Grid-specific land-cell filler.  Must accept and return arrays
    88	        with the same shape as ``T``.  Typical implementations:
    89	        ``jax.vmap(fill_land_cells_cubed_sphere, in_axes=-1)``,
    90	        ``_neumann_fill_cgrid`` (lat-lon C-grid), or
    91	        ``fill_land_cells_mpas`` (MPAS).
    92	    eos_fn : Callable
    93	        Equation of state ``(T, S, p) → ρ``.  Same signature used by
    94	        every grid (built via ``make_eos_fn``).
    95	    dz_ref : jax.Array
    96	        Reference layer thickness (``z_coord.dz_ref``), shape
    97	        ``(nlev,)``.
    98	    rho_0, g : float
    99	    n_iter : int, default 2
   100	        Number of EOS iterations *before* the final pressure update.
   101	        All current callers use 2.
   102	    hi_precision_pressure : bool, default False
   103	        If True, perform the pressure cumsum in float64 so that
   104	        halo-exchange interpolation errors do not contaminate the
   105	        downstream gradient.  Used by the cubed-sphere C-D path; on
   106	        lat-lon and MPAS the compact 2-cell stencils are well-behaved
   107	        enough that the working precision is sufficient.
   108	
   109	    Returns
   110	    -------
   111	    rho : jax.Array
   112	        In-situ density (same shape as ``T``).
   113	    rho_prime : jax.Array
   114	        ``rho - rho_0`` (same shape as ``T``).
   115	    p_prime : jax.Array
   116	        Baroclinic pressure anomaly (same shape as ``T``).  Returned in
   117	        whatever precision was used for the cumulative sum.
   118	    """
   119	    del mask  # currently unused (passed to fill_fn by the caller); kept
   120	              # in signature for clarity at call sites.
   121	
   122	    T_filled = fill_fn(T)
   123	    S_filled = fill_fn(S)
   124	
   125	    # Reference Jacobian (J=1, η=0).  Both have the horizontal shape of
   126	    # T (i.e. no vertical axis).  Match dtype to the working state so we
   127	    # never accidentally promote the EOS iteration to float64.
   128	    horiz_shape = T.shape[:-1]
   129	    J_ref = jnp.ones(horiz_shape, dtype=T.dtype)
   130	    eta_ref = jnp.zeros(horiz_shape, dtype=T.dtype)
   131	
   132	    rho = eos_fn(T_filled, S_filled, jnp.zeros_like(T))
   133	    for _ in range(n_iter):
   134	        p_hydro = compute_hydrostatic_pressure(
   135	            rho, eta_ref, dz_ref, J_ref, rho_0, g,
   136	        )
   137	        rho = eos_fn(T_filled, S_filled, p_hydro)
   138	
   139	    rho_prime = rho - rho_0
   140	
   141	    if hi_precision_pressure:
   142	        rho_prime_hi = rho_prime.astype(jnp.float64)
   143	        dz_hi = dz_ref.astype(jnp.float64)
   144	        dp_layer = rho_prime_hi * g * dz_hi
   145	    else:
   146	        dp_layer = rho_prime * g * dz_ref
   147	
   148	    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
   149	    p_prime = p_prime + 0.5 * dp_layer
   150	
   151	    return rho, rho_prime, p_prime
   152	
   153	
   154	def apply_sponge_tracer_relaxation(
   155	    dT_dt: jnp.ndarray,
   156	    dS_dt: jnp.ndarray,
   157	    T: jnp.ndarray,
   158	    S: jnp.ndarray,
   159	    sponge,
   160	    mask: Optional[jnp.ndarray] = None,
   161	    *,
   162	    expand_gamma_axis: int = -1,
   163	) -> Tuple[jnp.ndarray, jnp.ndarray]:
   164	    """Apply tracer sponge relaxation ``+γ·(ref - q)``.
   165	
   166	    Casts ``sponge`` arrays to ``T.dtype`` so the precision policy that
   167	    holds the state in float32 is not silently promoted to float64
   168	    (which previously crashed the barotropic scan; see latlon C-grid).
   169	
   170	    Parameters
   171	    ----------
   172	    dT_dt, dS_dt : jax.Array
   173	        Existing tendency arrays (modified by addition).
   174	    T, S : jax.Array
   175	        Current tracer state.
   176	    sponge : SpongeForcing
   177	        Must expose ``gamma``, ``T_ref``, ``S_ref``.  ``gamma`` is the
   178	        relaxation rate per cell (1 / s), broadcast along the vertical
   179	        axis via ``expand_gamma_axis``.
   180	    mask : jax.Array, optional
   181	        Ocean mask.  When provided, the relaxation tendency is
   182	        multiplied by ``mask`` (with the same axis expansion as
   183	        ``gamma``) so land cells stay quiescent.  When ``None`` no
   184	        masking is applied (the caller masks downstream).
   185	    expand_gamma_axis : int, default -1
   186	        Axis on which to insert a singleton in ``sponge.gamma`` so that
   187	        it broadcasts against the (..., nlev) tracer arrays.  Use ``-1``
   188	        for both lat-lon C-grid (axis after lat/lon) and MPAS (axis
   189	        after nCells).
   190	
   191	    Returns
   192	    -------
   193	    (dT_dt_new, dS_dt_new) : tuple of jax.Array
   194	    """
   195	    dtype = T.dtype
   196	    gamma = sponge.gamma.astype(dtype)
   197	    gamma_b = jnp.expand_dims(gamma, expand_gamma_axis)
   198	    dT = gamma_b * (sponge.T_ref.astype(dtype) - T)
   199	    dS = gamma_b * (sponge.S_ref.astype(dtype) - S)
   200	    if mask is not None:
   201	        mask_b = jnp.expand_dims(mask, expand_gamma_axis)
   202	        dT = dT * mask_b
   203	        dS = dS * mask_b
   204	    return dT_dt + dT, dS_dt + dS
   205	
   206	
   207	def apply_freshwater_virtual_salt_top(
   208	    dS_dt: jnp.ndarray,
   209	    freshwater,
   210	    S_ref: float,
   211	    h_top: jnp.ndarray,
   212	    rho_0: float,
   213	    mask: jnp.ndarray,
   214	) -> jnp.ndarray:
   215	    """Add the surface virtual-salt tendency to the top tracer level.
   216	
   217	    Wraps ``freshwater.virtual_salt_flux`` and assigns the result to
   218	    ``dS_dt[..., 0]`` (or the equivalent leading-axis slice for MPAS).
   219	    All callers operate on cell-centred salinity, so the trailing
   220	    ``nlev`` axis is the vertical axis.
   221	
   222	    Parameters
   223	    ----------
   224	    dS_dt : jax.Array
   225	        Salinity tendency, modified at index ``[..., 0]``.
   226	    freshwater : FreshwaterForcing
   227	    S_ref : float
   228	        Reference salinity used for the virtual-flux closure.
   229	    h_top : jax.Array
   230	        Top-layer thickness with the same horizontal shape as ``mask``.
   231	    rho_0 : float
   232	    mask : jax.Array
   233	        Ocean mask (1 = ocean) with the same horizontal shape as the
   234	        leading axes of ``dS_dt``.
   235	
   236	    Returns
   237	    -------
   238	    jax.Array
   239	        ``dS_dt`` with the virtual-salt flux added to the top layer.
   240	    """
   241	    dS_top = virtual_salt_flux(freshwater, S_ref, h_top, rho_0)
   242	    # Cast the freshwater contribution to dS_dt's dtype so the scatter
   243	    # add does not silently widen on x64 mode (the freshwater struct
   244	    # is built at JAX-default precision in init helpers, which can be
   245	    # f64 while the salinity tendency runs at the storage policy's
   246	    # f32).
   247	    return dS_dt.at[..., 0].add((dS_top * mask).astype(dS_dt.dtype))
   248	
   249	
   250	def implicit_bottom_drag_factor(
   251	    dt: jnp.ndarray | float,
   252	    drag_r: jnp.ndarray | float,
   253	    H: jnp.ndarray,
   254	    *,
   255	    eps: float = 1e-10,
   256	) -> jnp.ndarray:
   257	    """Per-substep bottom-drag multiplier ``1 - dt · r / max(H, eps)``.
   258	
   259	    Used by both the lat-lon C-grid and MPAS barotropic substeps to
   260	    apply a linear bottom drag on the depth-averaged velocity.  The
   261	    floor on ``H`` prevents the drag from blowing up over very thin
   262	    water columns (≈ inundation).
   263	
   264	    Parameters
   265	    ----------
   266	    dt : float or jax.Array
   267	        Substep size [s].
   268	    drag_r : float or jax.Array
   269	        Linear drag coefficient [m/s].
   270	    H : jax.Array
   271	        Total water column depth at the velocity location [m].
   272	    eps : float
   273	        Floor on ``H`` for numerical safety.
   274	
   275	    Returns
   276	    -------
   277	    jax.Array, same shape as ``H``.
   278	    """
   279	    return 1.0 - dt * drag_r / jnp.maximum(H, eps)

exec
/bin/zsh -lc "rg -n \"implicit_bottom_drag_factor\\(|drag_factor|bottom_drag\" src/legoesm/ocean/dynamics -g'*.py'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/ocean/dynamics/barotropic_mpas.py:42:from legoesm.ocean.dynamics.ocean_tendency_common import implicit_bottom_drag_factor
src/legoesm/ocean/dynamics/barotropic_mpas.py:271:        if config.bottom_drag_r > 0:
src/legoesm/ocean/dynamics/barotropic_mpas.py:272:            u_bar_next = u_bar_next * implicit_bottom_drag_factor(
src/legoesm/ocean/dynamics/barotropic_mpas.py:273:                dt_baro, config.bottom_drag_r, H_e_c,
src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:1265:    if config.bottom_drag_r > 0:
src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:1273:            -config.bottom_drag_r * u[..., -1] / dz_bot_u)
src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:1275:            -config.bottom_drag_r * v[..., -1] / dz_bot_v)
src/legoesm/ocean/dynamics/ocean_tendency_common.py:27:``implicit_bottom_drag_factor``
src/legoesm/ocean/dynamics/ocean_tendency_common.py:250:def implicit_bottom_drag_factor(
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:35:from legoesm.ocean.dynamics.ocean_tendency_common import implicit_bottom_drag_factor
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:339:        if config.bottom_drag_r > 0:
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:340:            U_bar_new = U_bar_new * implicit_bottom_drag_factor(
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:341:                dt_s, config.bottom_drag_r, H_u,
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:343:            V_bar_new = V_bar_new * implicit_bottom_drag_factor(
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:344:                dt_s, config.bottom_drag_r, H_v,
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:321:    if config.bottom_drag_r > 0:
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:324:            -config.bottom_drag_r * u_3d[:, -1] / dz_bot_e * edge_mask)

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py | sed -n '300,380p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/dynamics/barotropic_mpas.py | sed -n '240,320p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   240	                + F_slow_u
   241	            ) * edge_mask
   242	        else:
   243	            v_t_old = tangential_velocity(u_bar_c, mesh)
   244	            u_bar_next = u_bar_c + dt_baro * (
   245	                -g * grad_eta + mesh.fEdge * v_t_old + F_slow_u
   246	            ) * edge_mask
   247	
   248	        # Divergence damping: add nu * grad(div(u_bar)) (#205).
   249	        if use_div_damp:
   250	            div_ubar = divergence_cell(u_bar_next * edge_mask, mesh) * mask
   251	            div_filled = _fill_land_cells_mpas(div_ubar, mask)
   252	            grad_div = gradient_edge(div_filled, mesh)
   253	            u_bar_next = (
   254	                u_bar_next + div_damp_coeff * div_damp_area_edge * grad_div
   255	            ) * edge_mask
   256	
   257	        # Barotropic-mode lateral viscosity on u_bar (TRiSK null branch).
   258	        # Forward-Euler Laplacian: stable while A_baro_visc * dt_baro / dx² < 0.5.
   259	        if use_baro_visc:
   260	            lap_u = vector_laplacian_del2(u_bar_next, mesh)
   261	            u_bar_next = (
   262	                u_bar_next + dt_baro * A_baro_visc * lap_u
   263	            ) * edge_mask
   264	
   265	        # Optional barotropic damping (Rayleigh drag)
   266	        if config.barotropic_damping > 0:
   267	            u_bar_next = u_bar_next * (1.0 - dt_baro * config.barotropic_damping)
   268	
   269	        # Bottom drag on barotropic velocity: -r * U_bar / H_total.
   270	        # r is in [m/s] — resolution-independent bottom stress.
   271	        if config.bottom_drag_r > 0:
   272	            u_bar_next = u_bar_next * implicit_bottom_drag_factor(
   273	                dt_baro, config.bottom_drag_r, H_e_c,
   274	            )
   275	
   276	        # MAXVEL clipping
   277	        if use_maxvel:
   278	            u_bar_next = maxvel_clip(u_bar_next, _maxvel)
   279	
   280	        # Barotropic Laplacian diffusion on eta (flux-form: conservative).
   281	        # Uses div(nu_edge * grad(eta)) instead of nu_cell * div(grad(eta))
   282	        # so that volume is exactly conserved by the divergence theorem.
   283	        if use_baro_diffusion:
   284	            eta_filled = _fill_land_cells_mpas(eta_next, mask)
   285	            grad_e = gradient_edge(eta_filled, mesh)
   286	            diff_flux = nu_dt_edge * grad_e * edge_mask
   287	            eta_next = (
   288	                eta_next + divergence_cell(diff_flux, mesh)
   289	            ) * mask
   290	            eta_next = _clamp_redistribute(eta_next, eta_floor, mask, _area_cell)
   291	
   292	        # Accumulate eta and u_bar with cosine filter weights
   293	        eta_sum_new = eta_sum_c + w_i * eta_next.astype(_eta_dtype)
   294	        ubar_sum_new = ubar_sum_c + w_i * u_bar_next.astype(_eta_dtype)
   295	
   296	        # Cast back to input dtype (mesh ops may promote to float64)
   297	        return (eta_next.astype(_eta_dtype), u_bar_next.astype(_ubar_dtype),
   298	                Hu_sum_new.astype(_eta_dtype),
   299	                eta_sum_new, ubar_sum_new), None
   300	
   301	    (eta_new, u_bar_new, Hu_sum_f, eta_sum_f, ubar_sum_f), _ = jax.lax.scan(
   302	        _substep, (eta, u_bar, Hu_sum, eta_sum, ubar_sum),
   303	        w_filter, length=n_substeps,
   304	    )
   305	
   306	    # Time-averaged barotropic fields
   307	    Hu_avg = Hu_sum_f / n_substeps  # transport: always box-filtered
   308	    eta_avg = eta_sum_f / w_total   # eta/velocity: cosine or box filtered
   309	    u_bar_avg = ubar_sum_f / w_total
   310	
   311	    return eta_avg, u_bar_avg, Hu_avg
   312	
   313	
   314	def reconcile_3d_velocity(u_3d, u_bar_old, u_bar_new, mesh, mask):
   315	    """Reconcile 3D velocity with updated barotropic velocity.
   316	
   317	    Preserves baroclinic structure (deviations from depth-mean) while
   318	    replacing the depth-mean with the barotropic solution:
   319	
   320	        u_3d_new = (u_3d - u_bar_old) + u_bar_new

 succeeded in 0ms:
   300	        # (wall BC); single Pad HLO op replaces alloc-zeros +
   301	        # concatenate-of-three (called every substep).
   302	        U_at_v_interior = 0.25 * (
   303	            U_bar_c[:-1, :-1] + U_bar_c[:-1, 1:]
   304	            + U_bar_c[1:, :-1] + U_bar_c[1:, 1:]
   305	        )
   306	        U_at_v = jnp.pad(U_at_v_interior, ((1, 1), (0, 0)))
   307	
   308	        # Forward-backward Coriolis (Matsuno) + PGF + slow forcing
   309	        U_bar_new = (U_bar_c + dt_s * (
   310	            f_u * V_at_u - g * deta_dx + F_slow_u
   311	        )) * u_mask
   312	
   313	        U_new_at_v_interior = 0.25 * (
   314	            U_bar_new[:-1, :-1] + U_bar_new[:-1, 1:]
   315	            + U_bar_new[1:, :-1] + U_bar_new[1:, 1:]
   316	        )
   317	        # Pole rows are zero; single Pad HLO op (substep hot path).
   318	        U_new_at_v = jnp.pad(U_new_at_v_interior, ((1, 1), (0, 0)))
   319	        V_bar_new = (V_bar_c + dt_s * (
   320	            -f_v * U_new_at_v - g * deta_dy + F_slow_v
   321	        )) * v_mask
   322	
   323	        # Divergence damping: grad(div(u_bar)) (#205)
   324	        if use_div_damp:
   325	            div_uv = divergence_cgrid(
   326	                U_bar_new, V_bar_new, grid,
   327	                u_mask=u_mask, v_mask=v_mask,
   328	            ).astype(eta.dtype)
   329	            grad_div_x = gradient_x_cgrid(div_uv * mask, grid).astype(eta.dtype)
   330	            grad_div_y = gradient_y_cgrid(div_uv * mask, grid).astype(eta.dtype)
   331	            U_bar_new = (
   332	                U_bar_new + div_damp_coeff * div_damp_area_u * grad_div_x
   333	            ) * u_mask
   334	            V_bar_new = (
   335	                V_bar_new + div_damp_coeff * div_damp_area_v * grad_div_y
   336	            ) * v_mask
   337	
   338	        # Bottom drag: -r * U_bar / H_total
   339	        if config.bottom_drag_r > 0:
   340	            U_bar_new = U_bar_new * implicit_bottom_drag_factor(
   341	                dt_s, config.bottom_drag_r, H_u,
   342	            )
   343	            V_bar_new = V_bar_new * implicit_bottom_drag_factor(
   344	                dt_s, config.bottom_drag_r, H_v,
   345	            )
   346	
   347	        # --- MAXVEL clipping: prevent runaway velocities ---
   348	        if use_maxvel:
   349	            U_bar_new = maxvel_clip(U_bar_new, _maxvel)
   350	            V_bar_new = maxvel_clip(V_bar_new, _maxvel)
   351	
   352	        # Optional Laplacian damping on eta (flux-form: conservative)
   353	        if config.barotropic_diffusion_alpha > 0.0:
   354	            grad_x = gradient_x_cgrid(eta_new * mask, grid)
   355	            grad_y = gradient_y_cgrid(eta_new * mask, grid)
   356	            flux_x = nu_face_u * grad_x * diff_u_mask
   357	            flux_y = nu_face_v * grad_y * diff_v_mask
   358	            eta_new = (
   359	                eta_new + divergence_cgrid(flux_x, flux_y, grid).astype(eta.dtype)
   360	            ) * mask
   361	            eta_new = _clamp_redistribute(eta_new, eta_floor, mask, _area)
   362	
   363	        # Accumulate eta, U_bar, V_bar with cosine filter weights
   364	        eta_sum_new = eta_sum_c + w_i * eta_new
   365	        U_sum_new = U_sum_c + w_i * U_bar_new
   366	        V_sum_new = V_sum_c + w_i * V_bar_new
   367	
   368	        return (eta_new, U_bar_new, V_bar_new,
   369	                Hu_sum_new, Hv_sum_new, eta_sum_new, U_sum_new, V_sum_new)
   370	
   371	    init_carry = (eta, U_bar, V_bar, Hu_sum, Hv_sum, eta_sum, U_sum, V_sum)
   372	
   373	    if config.differentiable_barotropic:
   374	        # scan path: pass filter weights as xs for cosine filtering
   375	        def scan_body(carry, w_i):
   376	            new_carry = substep_body(w_i, carry)
   377	            return new_carry, None
   378	
   379	        (eta_f, U_bar_f, V_bar_f,
   380	         Hu_sum_f, Hv_sum_f, eta_sum_f, U_sum_f, V_sum_f), _ = jax.lax.scan(

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/freshwater.py | sed -n '1,280p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Freshwater forcing for the MPAS ocean model.
     2	
     3	Handles precipitation, evaporation, land runoff, and ice melt/freeze
     4	freshwater fluxes. Applies virtual salt flux to salinity and real
     5	freshwater mass flux to the free surface.
     6	
     7	Conventions
     8	-----------
     9	- All fluxes in kg/m2/s (mass flux per unit area).
    10	- Positive = freshwater entering ocean (precip, runoff, ice melt).
    11	- Evaporation is positive upward in the coupler, so E enters here
    12	  as a positive value that *removes* freshwater from the ocean.
    13	
    14	References
    15	----------
    16	- Griffies, S. M. (2004). Fundamentals of Ocean Climate Models, Ch. 12.
    17	- Large, W. G. et al. (1997). J. Phys. Oceanogr., 27(11), 2418-2447.
    18	"""
    19	
    20	from __future__ import annotations
    21	
    22	from typing import NamedTuple
    23	
    24	import jax.numpy as jnp
    25	
    26	
    27	class FreshwaterForcing(NamedTuple):
    28	    """Freshwater fluxes applied to the ocean surface.
    29	
    30	    All fields have shape (nCells,) and units kg/m2/s.
    31	    Positive = freshwater into ocean, except evaporation which is
    32	    positive upward (i.e., freshwater leaving ocean).
    33	
    34	    Fields
    35	    ------
    36	    precip : jax.Array
    37	        Precipitation rate [kg/m2/s].
    38	    evap : jax.Array
    39	        Evaporation rate [kg/m2/s], positive upward.
    40	    runoff : jax.Array
    41	        Land runoff rate [kg/m2/s].
    42	    ice_fw : jax.Array
    43	        Ice melt/freeze freshwater [kg/m2/s], positive = melt.
    44	    """
    45	    precip: jnp.ndarray
    46	    evap: jnp.ndarray
    47	    runoff: jnp.ndarray
    48	    ice_fw: jnp.ndarray
    49	
    50	
    51	def zero_freshwater(nCells: int) -> FreshwaterForcing:
    52	    """Create zero freshwater forcing.
    53	
    54	    Parameters
    55	    ----------
    56	    nCells : int
    57	        Number of Voronoi cells.
    58	
    59	    Returns
    60	    -------
    61	    FreshwaterForcing
    62	    """
    63	    # Init helper: keep at the JAX default float dtype.  Callers running
    64	    # under a non-default precision policy can ``cast_pytree`` the
    65	    # result to match their state.
    66	    z = jnp.zeros(nCells)
    67	    return FreshwaterForcing(precip=z, evap=z, runoff=z, ice_fw=z)
    68	
    69	
    70	def net_freshwater_flux(fw: FreshwaterForcing) -> jnp.ndarray:
    71	    """Compute net freshwater flux into ocean [kg/m2/s].
    72	
    73	    F_fw = P - E + R + M
    74	
    75	    where P=precip, E=evaporation (positive up), R=runoff, M=ice melt.
    76	
    77	    Parameters
    78	    ----------
    79	    fw : FreshwaterForcing
    80	
    81	    Returns
    82	    -------
    83	    jax.Array, shape (nCells,)
    84	        Net freshwater flux [kg/m2/s], positive into ocean.
    85	    """
    86	    return fw.precip - fw.evap + fw.runoff + fw.ice_fw
    87	
    88	
    89	def freshwater_eta_tendency(fw: FreshwaterForcing, rho_0: float) -> jnp.ndarray:
    90	    """Compute free-surface tendency from freshwater flux.
    91	
    92	    deta/dt = F_fw / rho_0
    93	
    94	    Parameters
    95	    ----------
    96	    fw : FreshwaterForcing
    97	    rho_0 : float
    98	        Reference seawater density [kg/m3].
    99	
   100	    Returns
   101	    -------
   102	    jax.Array, shape (nCells,)
   103	        Free-surface tendency [m/s].
   104	    """
   105	    return net_freshwater_flux(fw) / rho_0
   106	
   107	
   108	def virtual_salt_flux(
   109	    fw: FreshwaterForcing,
   110	    S_ref: float,
   111	    dz_0: jnp.ndarray,
   112	    rho_0: float,
   113	) -> jnp.ndarray:
   114	    """Compute virtual salt flux for the top ocean layer.
   115	
   116	    dS/dt = -S_ref * F_fw / (rho_0 * dz_0)
   117	
   118	    This approximation maintains volume while adjusting salinity
   119	    to account for freshwater dilution/concentration.
   120	
   121	    Parameters
   122	    ----------
   123	    fw : FreshwaterForcing
   124	    S_ref : float
   125	        Reference salinity [PSU].
   126	    dz_0 : jax.Array, shape (nCells,)
   127	        Top layer thickness [m].
   128	    rho_0 : float
   129	        Reference seawater density [kg/m3].
   130	
   131	    Returns
   132	    -------
   133	    jax.Array, shape (nCells,)
   134	        Salinity tendency [PSU/s] for top layer.
   135	    """
   136	    F_fw = net_freshwater_flux(fw)
   137	    dz_safe = jnp.maximum(dz_0, 1e-10)
   138	    return -S_ref * F_fw / (rho_0 * dz_safe)
   139	
   140	
   141	def freshwater_from_coupler(
   142	    precip_total: jnp.ndarray,
   143	    lhflx: jnp.ndarray,
   144	    L_v: float,
   145	    runoff_surface: jnp.ndarray | None = None,
   146	    runoff_subsurface: jnp.ndarray | None = None,
   147	    ice_state_old=None,
   148	    ice_state_new=None,
   149	    ice_config=None,
   150	    ocean_mask: jnp.ndarray | None = None,
   151	    dt: float = 1.0,
   152	) -> FreshwaterForcing:
   153	    """Compute freshwater forcing from coupler fields.
   154	
   155	    Parameters
   156	    ----------
   157	    precip_total : jax.Array, shape (nCells,)
   158	        Total precipitation [kg/m2/s].
   159	    lhflx : jax.Array, shape (nCells,)
   160	        Latent heat flux [W/m2], positive upward.
   161	    L_v : float
   162	        Latent heat of vaporization [J/kg].
   163	    runoff_surface : jax.Array or None, shape (nCells,)
   164	        Surface runoff from land [kg/m2/s].
   165	    runoff_subsurface : jax.Array or None, shape (nCells,)
   166	        Subsurface runoff from land [kg/m2/s].
   167	    ice_state_old, ice_state_new : SeaIceState or None
   168	        Ice state before/after ice step. Used to compute ice freshwater.
   169	    ice_config : SeaIceConfig or None
   170	        Ice config with rho_ice.
   171	    ocean_mask : jax.Array or None, shape (nCells,)
   172	        Ocean mask (1=ocean). Used to restrict fluxes to ocean cells.
   173	    dt : float
   174	        Timestep [s]. Used for ice thickness change rate.
   175	
   176	    Returns
   177	    -------
   178	    FreshwaterForcing
   179	    """
   180	    nCells = precip_total.shape[0]
   181	
   182	    # Precipitation over ocean
   183	    precip = precip_total
   184	
   185	    # Evaporation from latent heat flux: E = lhflx / L_v
   186	    evap = lhflx / L_v
   187	
   188	    # Land runoff (sum surface + subsurface).  Pin the zero-fallback
   189	    # dtype to the precip path so a missing runoff input does not
   190	    # silently widen the freshwater forcing struct to f64 under x64.
   191	    if runoff_surface is not None:
   192	        runoff = runoff_surface
   193	        if runoff_subsurface is not None:
   194	            runoff = runoff + runoff_subsurface
   195	    else:
   196	        runoff = jnp.zeros(nCells, dtype=precip.dtype)
   197	
   198	    # Ice freshwater: based on areal ice mass change.
   199	    # ice_mass = rho_ice * h * A  (per unit area of grid cell)
   200	    # ice_fw = -(ice_mass_new - ice_mass_old) / dt
   201	    # Melting (mass decrease) puts freshwater into ocean (positive fw).
   202	    # Supports both single-category and multi-category ice.
   203	    if ice_state_old is not None and ice_state_new is not None and ice_config is not None:
   204	        h_old = ice_state_old.h_ice.data
   205	        h_new = ice_state_new.h_ice.data
   206	        A_old = ice_state_old.concentration.data
   207	        A_new = ice_state_new.concentration.data
   208	        rho_ice = ice_config.rho_ice
   209	        ice_mass_old = rho_ice * h_old * A_old
   210	        ice_mass_new = rho_ice * h_new * A_new
   211	        # Multi-category: h has more dims than precip_total; sum categories.
   212	        n_extra = ice_mass_old.ndim - precip_total.ndim
   213	        for _ in range(n_extra):
   214	            ice_mass_old = jnp.sum(ice_mass_old, axis=-1)
   215	            ice_mass_new = jnp.sum(ice_mass_new, axis=-1)
   216	        ice_fw = -(ice_mass_new - ice_mass_old) / jnp.maximum(dt, 1e-10)
   217	    else:
   218	        ice_fw = jnp.zeros(nCells, dtype=precip.dtype)
   219	
   220	    # Mask to ocean cells
   221	    if ocean_mask is not None:
   222	        precip = precip * ocean_mask
   223	        evap = evap * ocean_mask
   224	        runoff = runoff * ocean_mask
   225	        ice_fw = ice_fw * ocean_mask
   226	
   227	    return FreshwaterForcing(
   228	        precip=precip,
   229	        evap=evap,
   230	        runoff=runoff,
   231	        ice_fw=ice_fw,
   232	    )

codex
I’ve identified one concrete bug already in `freshwater_from_coupler`: `runoff_subsurface` is silently dropped when `runoff_surface` is `None`. I’ll keep scanning for similarly concrete issues and then classify your four specific concerns as confirmed/false-positive/ambiguous.
exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/sponge.py | sed -n '1,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Sponge layer relaxation for ocean models.
     2	
     3	Provides a ``SpongeForcing`` container and utilities for computing
     4	spatially varying relaxation coefficients.  Sponge layers nudge the
     5	model state toward a reference profile near domain boundaries, damping
     6	wave reflections and preventing wall instabilities in channel
     7	experiments (standard practice in MITgcm RBCS, MOM6 ALE_sponge).
     8	
     9	The relaxation is applied as a tendency::
    10	
    11	    dT/dt += gamma(x,y) * (T_ref(x,y,z) - T)
    12	
    13	where gamma [1/s] ramps from zero in the interior to 1/tau at the
    14	boundary.
    15	"""
    16	
    17	from __future__ import annotations
    18	
    19	from typing import NamedTuple
    20	
    21	import jax.numpy as jnp
    22	import numpy as np
    23	
    24	
    25	class SpongeForcing(NamedTuple):
    26	    """Sponge layer relaxation fields.
    27	
    28	    Parameters
    29	    ----------
    30	    gamma : array
    31	        Relaxation rate [1/s].  Shape ``(n_lat, n_lon)`` for lat-lon
    32	        or ``(nCells,)`` for MPAS.  Zero in the interior, ramping to
    33	        ``1/tau`` near boundaries.
    34	    T_ref : array
    35	        Reference temperature, same shape as the model T field.
    36	    S_ref : array
    37	        Reference salinity, same shape as the model S field.
    38	    u_ref : array or None
    39	        Reference u velocity (optional).
    40	    v_ref : array or None
    41	        Reference v velocity (optional, not used for MPAS edge velocity).
    42	    """
    43	    gamma: jnp.ndarray
    44	    T_ref: jnp.ndarray
    45	    S_ref: jnp.ndarray
    46	    u_ref: jnp.ndarray | None = None
    47	    v_ref: jnp.ndarray | None = None
    48	
    49	
    50	def compute_sponge_gamma_latlon(
    51	    grid,
    52	    lat_south: float,
    53	    lat_north: float,
    54	    width_deg: float = 2.0,
    55	    timescale_days: float = 1.0,
    56	) -> np.ndarray:
    57	    """Compute sponge relaxation coefficient on a lat-lon grid.
    58	
    59	    Quadratic ramp from 0 in the interior to ``1/tau`` at the walls.
    60	
    61	    Parameters
    62	    ----------
    63	    grid : LatLonGrid
    64	    lat_south, lat_north : float
    65	        Domain boundaries [degrees].
    66	    width_deg : float
    67	        Sponge zone width [degrees].
    68	    timescale_days : float
    69	        Relaxation e-folding timescale [days].
    70	
    71	    Returns
    72	    -------
    73	    gamma : ndarray, shape (n_lat, n_lon)
    74	        Relaxation coefficient [1/s].
    75	    """
    76	    lat_deg = np.degrees(np.asarray(grid.lat))
    77	    tau = timescale_days * 86400.0
    78	    gamma = np.zeros((grid.n_lat, grid.n_lon), dtype=np.float64)
    79	
    80	    for i, lat in enumerate(lat_deg):
    81	        dist_south = lat - lat_south
    82	        dist_north = lat_north - lat
    83	        if dist_south < width_deg:
    84	            gamma[i, :] = (1.0 - dist_south / width_deg) ** 2 / tau
    85	        elif dist_north < width_deg:
    86	            gamma[i, :] = (1.0 - dist_north / width_deg) ** 2 / tau
    87	
    88	    return gamma
    89	
    90	
    91	def compute_sponge_gamma_mpas(
    92	    mesh,
    93	    lat_south: float,
    94	    lat_north: float,
    95	    width_deg: float = 2.0,
    96	    timescale_days: float = 1.0,
    97	) -> np.ndarray:
    98	    """Compute sponge relaxation coefficient on an MPAS Voronoi mesh.
    99	
   100	    Same quadratic ramp as the lat-lon version, applied per cell.
   101	
   102	    Parameters
   103	    ----------
   104	    mesh : VoronoiMesh
   105	    lat_south, lat_north : float
   106	        Domain boundaries [degrees].
   107	    width_deg : float
   108	        Sponge zone width [degrees].
   109	    timescale_days : float
   110	        Relaxation e-folding timescale [days].
   111	
   112	    Returns
   113	    -------
   114	    gamma : ndarray, shape (nCells,)
   115	        Relaxation coefficient [1/s].
   116	    """
   117	    lat_deg = np.degrees(np.asarray(mesh.latCell))
   118	    tau = timescale_days * 86400.0
   119	    n_cells = lat_deg.shape[0]
   120	    gamma = np.zeros(n_cells, dtype=np.float64)
   121	
   122	    for i, lat in enumerate(lat_deg):
   123	        dist_south = lat - lat_south
   124	        dist_north = lat_north - lat
   125	        if dist_south < width_deg:
   126	            gamma[i] = (1.0 - dist_south / width_deg) ** 2 / tau
   127	        elif dist_north < width_deg:
   128	            gamma[i] = (1.0 - dist_north / width_deg) ** 2 / tau
   129	
   130	    return gamma

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/init_latlon_cgrid.py | sed -n '1,340p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Initialization for the lat-lon C-grid FV ocean model."""
     2	
     3	from __future__ import annotations
     4	
     5	import jax.numpy as jnp
     6	
     7	from legoesm.core.field import Field
     8	from legoesm.core.precision import get_policy
     9	from legoesm.grids.latlon import LatLonGrid
    10	from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
    11	from legoesm.ocean.vertical import OceanZStarCoordinate
    12	from legoesm.ocean.state import LatLonCGridOceanState
    13	from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
    14	
    15	
    16	def idealized_bathymetry_latlon_cgrid(
    17	    grid: LatLonGrid,
    18	    H_max: float = 5500.0,
    19	    land_lat_threshold: float = 80.0,
    20	) -> tuple[jnp.ndarray, jnp.ndarray]:
    21	    """Generate idealized bathymetry and land mask on a lat-lon grid.
    22	
    23	    Land is placed at high latitudes (|lat| > threshold).
    24	
    25	    Parameters
    26	    ----------
    27	    grid : LatLonGrid
    28	    H_max : float
    29	        Maximum ocean depth [m].
    30	    land_lat_threshold : float
    31	        Latitude [degrees] above which cells are land.
    32	
    33	    Returns
    34	    -------
    35	    H_bathy : array, shape (n_lat, n_lon)
    36	    land_mask : array, shape (n_lat, n_lon)
    37	    """
    38	    if H_max <= 0.0:
    39	        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    40	    if land_lat_threshold < 0.0 or land_lat_threshold > 90.0:
    41	        raise ValueError(
    42	            f"land_lat_threshold must be in [0, 90], got {land_lat_threshold!r}",
    43	        )
    44	
    45	    dtype = get_policy().storage
    46	    lat_deg = jnp.abs(grid.lat2d) * (180.0 / jnp.pi)
    47	    land_mask = jnp.where(lat_deg < land_lat_threshold, 1.0, 0.0).astype(dtype)
    48	    H_bathy = jnp.full_like(land_mask, H_max)
    49	
    50	    return H_bathy, land_mask
    51	
    52	
    53	def rest_state_latlon_cgrid_ocean(
    54	    grid: LatLonGrid,
    55	    z_coord: OceanZStarCoordinate,
    56	    T_surface: float = 20.0,
    57	    T_deep: float = 2.0,
    58	    S_uniform: float = 35.0,
    59	    H_max: float = 5500.0,
    60	    land_lat_threshold: float = 80.0,
    61	    land_mask_override: jnp.ndarray | None = None,
    62	) -> LatLonCGridOceanState:
    63	    """Create a rest-state initial condition on a C-grid lat-lon grid.
    64	
    65	    Temperature: exponential profile.
    66	    Salinity: uniform.
    67	    Velocity: zero.
    68	    Eta: zero.
    69	
    70	    Parameters
    71	    ----------
    72	    grid : LatLonGrid
    73	    z_coord : OceanZStarCoordinate
    74	    T_surface, T_deep : float
    75	        Surface and deep temperature [degC].
    76	    S_uniform : float
    77	        Uniform salinity [PSU].
    78	    H_max : float
    79	        Maximum ocean depth [m].
    80	    land_lat_threshold : float
    81	        Latitude threshold for land [degrees].  Ignored when
    82	        *land_mask_override* is provided.
    83	    land_mask_override : array (n_lat, n_lon), optional
    84	        If provided, use this as the land mask (1=ocean, 0=land) instead
    85	        of deriving one from *land_lat_threshold*.  Face masks (u_mask,
    86	        v_mask) are computed from it automatically.
    87	
    88	    Returns
    89	    -------
    90	    LatLonCGridOceanState
    91	    """
    92	    n_lat = grid.n_lat
    93	    n_lon = grid.n_lon
    94	    nlev = z_coord.n_levels
    95	
    96	    if land_mask_override is not None:
    97	        land_mask = jnp.asarray(land_mask_override)
    98	        H_bathy = jnp.full((n_lat, n_lon), H_max, dtype=jnp.float64)
    99	    else:
   100	        H_bathy, land_mask = idealized_bathymetry_latlon_cgrid(
   101	            grid, H_max, land_lat_threshold,
   102	        )
   103	
   104	    # Exponential T stratification
   105	    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(
   106	        z_coord.z_full_ref / _SCALE_DEPTH,
   107	    )
   108	    dtype = get_policy().storage
   109	    T_3d = jnp.broadcast_to(
   110	        T_profile[jnp.newaxis, jnp.newaxis, :], (n_lat, n_lon, nlev),
   111	    ).astype(dtype)
   112	
   113	    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)
   114	
   115	    # C-grid velocity shapes
   116	    u_zeros = jnp.zeros((n_lat, n_lon + 1, nlev), dtype=dtype)
   117	    v_zeros = jnp.zeros((n_lat + 1, n_lon, nlev), dtype=dtype)
   118	    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)
   119	
   120	    # Face masks
   121	    u_mask, v_mask = compute_face_masks(land_mask)
   122	
   123	    # Initialize vertical velocity with zeros (will be computed during step)
   124	    w_zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)
   125	
   126	    dims_u = ("lat", "lon_u", "level")
   127	    dims_v = ("lat_v", "lon", "level")
   128	    dims_3d = ("lat", "lon", "level")
   129	    dims_2d = ("lat", "lon")
   130	    dims_u2d = ("lat", "lon_u")
   131	    dims_v2d = ("lat_v", "lon")
   132	
   133	    return LatLonCGridOceanState(
   134	        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
   135	                staggering="edge"),
   136	        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
   137	                staggering="edge"),
   138	        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
   139	        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
   140	        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
   141	        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
   142	        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
   143	        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
   144	        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
   145	        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
   146	    )
   147	
   148	
   149	def wind_driven_gyre_latlon_cgrid(
   150	    grid: LatLonGrid,
   151	    z_coord: OceanZStarCoordinate,
   152	    H_max: float = 5500.0,
   153	    lon_west: float = 0.0,
   154	    lon_east: float = 120.0,
   155	    lat_south: float = 15.0,
   156	    lat_north: float = 75.0,
   157	    T_uniform: float = 10.0,
   158	    S_uniform: float = 35.0,
   159	) -> LatLonCGridOceanState:
   160	    """Create initial condition for a wind-driven barotropic gyre on C-grid lat-lon.
   161	
   162	    Uniform T and S inside a rectangular basin. Purely barotropic setup.
   163	    """
   164	    n_lat = grid.n_lat
   165	    n_lon = grid.n_lon
   166	    nlev = z_coord.n_levels
   167	    dtype = get_policy().storage
   168	
   169	    lon_deg = grid.lon2d * (180.0 / jnp.pi)
   170	    lat_deg = grid.lat2d * (180.0 / jnp.pi)
   171	    in_basin = (
   172	        (lon_deg >= lon_west) & (lon_deg <= lon_east) &
   173	        (lat_deg >= lat_south) & (lat_deg <= lat_north)
   174	    )
   175	    land_mask = jnp.where(in_basin, 1.0, 0.0).astype(dtype)
   176	    H_bathy = jnp.full_like(land_mask, H_max)
   177	
   178	    T_3d = jnp.full((n_lat, n_lon, nlev), T_uniform, dtype=dtype)
   179	    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)
   180	
   181	    u_zeros = jnp.zeros((n_lat, n_lon + 1, nlev), dtype=dtype)
   182	    v_zeros = jnp.zeros((n_lat + 1, n_lon, nlev), dtype=dtype)
   183	    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)
   184	
   185	    u_mask, v_mask = compute_face_masks(land_mask)
   186	
   187	    # Initialize vertical velocity with zeros (will be computed during step)
   188	    w_zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)
   189	
   190	    dims_u = ("lat", "lon_u", "level")
   191	    dims_v = ("lat_v", "lon", "level")
   192	    dims_3d = ("lat", "lon", "level")
   193	    dims_2d = ("lat", "lon")
   194	    dims_u2d = ("lat", "lon_u")
   195	    dims_v2d = ("lat_v", "lon")
   196	
   197	    return LatLonCGridOceanState(
   198	        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
   199	                staggering="edge"),
   200	        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
   201	                staggering="edge"),
   202	        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
   203	        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
   204	        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
   205	        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
   206	        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
   207	        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
   208	        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
   209	        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
   210	    )
   211	
   212	
   213	def regional_rest_state_latlon_cgrid(
   214	    grid: LatLonGrid,
   215	    wall_mask: jnp.ndarray,
   216	    z_coord: OceanZStarCoordinate,
   217	    H_max: float = 5500.0,
   218	    T_surface: float = 20.0,
   219	    T_deep: float = 2.0,
   220	    S_uniform: float = 35.0,
   221	) -> LatLonCGridOceanState:
   222	    """Create a rest-state initial condition on a regional C-grid lat-lon grid.
   223	
   224	    Parameters
   225	    ----------
   226	    grid : LatLonGrid
   227	        Regional grid (includes 1-cell wall boundary).
   228	    wall_mask : jax.Array, shape (n_lat, n_lon)
   229	        1 = ocean interior, 0 = wall.
   230	    z_coord : OceanZStarCoordinate
   231	    H_max : float
   232	    T_surface, T_deep : float
   233	    S_uniform : float
   234	
   235	    Returns
   236	    -------
   237	    LatLonCGridOceanState
   238	    """
   239	    n_lat = grid.n_lat
   240	    n_lon = grid.n_lon
   241	    nlev = z_coord.n_levels
   242	    dtype = get_policy().storage
   243	
   244	    H_bathy = jnp.full((n_lat, n_lon), H_max, dtype=dtype)
   245	
   246	    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(
   247	        z_coord.z_full_ref / _SCALE_DEPTH,
   248	    )
   249	    T_3d = jnp.broadcast_to(
   250	        T_profile[jnp.newaxis, jnp.newaxis, :], (n_lat, n_lon, nlev),
   251	    ).astype(dtype)
   252	
   253	    S_3d = jnp.full((n_lat, n_lon, nlev), S_uniform, dtype=dtype)
   254	
   255	    u_zeros = jnp.zeros((n_lat, n_lon + 1, nlev), dtype=dtype)
   256	    v_zeros = jnp.zeros((n_lat + 1, n_lon, nlev), dtype=dtype)
   257	    zeros_2d = jnp.zeros((n_lat, n_lon), dtype=dtype)
   258	
   259	    land_mask = wall_mask.astype(dtype)
   260	    u_mask, v_mask = compute_face_masks(land_mask)
   261	
   262	    # Initialize vertical velocity with zeros (will be computed during step)
   263	    w_zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)
   264	
   265	    dims_u = ("lat", "lon_u", "level")
   266	    dims_v = ("lat_v", "lon", "level")
   267	    dims_3d = ("lat", "lon", "level")
   268	    dims_2d = ("lat", "lon")
   269	    dims_u2d = ("lat", "lon_u")
   270	    dims_v2d = ("lat_v", "lon")
   271	
   272	    return LatLonCGridOceanState(
   273	        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
   274	                staggering="edge"),
   275	        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
   276	                staggering="edge"),
   277	        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
   278	        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
   279	        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
   280	        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
   281	        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
   282	        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
   283	        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
   284	        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
   285	    )
   286	
   287	
   288	def replace_land_mask(
   289	    state: LatLonCGridOceanState,
   290	    new_land_mask: jnp.ndarray,
   291	) -> LatLonCGridOceanState:
   292	    """Replace land_mask and recompute u_mask/v_mask atomically.
   293	
   294	    Use this instead of ``state._replace(land_mask=...)`` to ensure
   295	    face masks stay consistent with the cell mask.
   296	    """
   297	    new_land_mask = jnp.asarray(new_land_mask)
   298	    u_mask, v_mask = compute_face_masks(new_land_mask)
   299	    return state._replace(
   300	        land_mask=Field(data=new_land_mask, name="land_mask",
   301	                        dims=state.land_mask.dims, units=""),
   302	        u_mask=Field(data=u_mask, name="u_mask",
   303	                     dims=state.u_mask.dims, units=""),
   304	        v_mask=Field(data=v_mask, name="v_mask",
   305	                     dims=state.v_mask.dims, units=""),
   306	    )

exec
/bin/zsh -lc "rg -n \"H_bathy\\.data|jacobian|eta \\+ H_bathy|H_total\" src/legoesm/ocean -g'*.py' | head -n 120" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/ocean/diagnostics.py:26:    jacobian: jnp.ndarray,
src/legoesm/ocean/diagnostics.py:43:    jacobian : (...)
src/legoesm/ocean/diagnostics.py:46:        Coriolis parameter [1/s], same horizontal shape as ``jacobian``.
src/legoesm/ocean/diagnostics.py:59:        rho, z_coord.dz_ref, jacobian, rho_ref=rho_ref, g=constants.g,
src/legoesm/ocean/diagnostics.py:64:    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]  # (..., nlev)
src/legoesm/ocean/__init__.py:42:    compute_ocean_jacobian,
src/legoesm/ocean/__init__.py:129:    "compute_ocean_jacobian",
src/legoesm/ocean/physics/combined.py:126:            from legoesm.ocean.vertical import compute_ocean_jacobian
src/legoesm/ocean/physics/combined.py:127:            J = compute_ocean_jacobian(
src/legoesm/ocean/physics/combined.py:128:                state.eta.data, state.H_bathy.data, z_coord,
src/legoesm/ocean/dynamics/barotropic_cgrid.py:166:    H_bathy = cast(state.H_bathy.data, _M, "compute")
src/legoesm/ocean/dynamics/barotropic_cgrid.py:184:    H_total = jnp.maximum(jnp.sum(h_k, axis=-1), min_water_col)
src/legoesm/ocean/dynamics/barotropic_cgrid.py:185:    U_bar_cc = jnp.sum(u * h_k, axis=-1) / H_total * mask
src/legoesm/ocean/dynamics/barotropic_cgrid.py:186:    V_bar_cc = jnp.sum(v * h_k, axis=-1) / H_total * mask
src/legoesm/ocean/dynamics/barotropic_cgrid.py:216:        H_total_c = jnp.maximum(eta_c + H_bathy, min_water_col) * mask
src/legoesm/ocean/dynamics/barotropic_cgrid.py:219:        H_pad = pad_halo(H_total_c, interp_offsets=grid.halo_interp_offsets)
src/legoesm/ocean/conservation.py:108:            - state_new.H_bathy.data
src/legoesm/ocean/conservation.py:138:        state_old.H_bathy.data,
src/legoesm/ocean/conservation.py:144:        state_new.H_bathy.data,
src/legoesm/ocean/conservation.py:189:        state_old.H_bathy.data,
src/legoesm/ocean/conservation.py:195:        state_new.H_bathy.data,
src/legoesm/ocean/conservation.py:238:        state_old.eta.data, state_old.H_bathy.data, z_coord,
src/legoesm/ocean/conservation.py:259:            eta_floor = jnp.asarray(min_wc, dtype=eta_corrected.dtype) - state_new.H_bathy.data
src/legoesm/ocean/conservation.py:266:        eta_corrected, state_new.H_bathy.data, z_coord,
src/legoesm/ocean/dynamics/ocean_model_mpas.py:270:                T_new, S_new, state.eta.data, state.H_bathy.data,
src/legoesm/ocean/dynamics/ocean_model_mpas.py:292:            mask, state.eta.data, state.H_bathy.data,
src/legoesm/ocean/dynamics/ocean_model_mpas.py:340:            state.eta.data, state.H_bathy.data, z_coord,
src/legoesm/ocean/dynamics/ocean_model_mpas.py:344:            eta_new, state.H_bathy.data, z_coord,
src/legoesm/ocean/dynamics/ocean_model_mpas.py:352:        H_total = jnp.maximum(state.eta.data + state.H_bathy.data,
src/legoesm/ocean/dynamics/ocean_model_mpas.py:354:        H_e = 0.5 * (H_total[c1] + H_total[c2])
src/legoesm/ocean/init.py:54:    # (eta + H_bathy) / H_max is smooth across coastlines.  The land_mask
src/legoesm/ocean/physics/mpas_physics.py:19:from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
src/legoesm/ocean/physics/mpas_physics.py:105:        H_bathy = state.H_bathy.data
src/legoesm/ocean/physics/mpas_physics.py:116:        jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
src/legoesm/ocean/physics/mpas_physics.py:124:            dz_0_cell = z_coord.dz_ref[0] * jacobian  # (nCells,)
src/legoesm/ocean/physics/mpas_physics.py:184:            rho = compute_ocean_rho(state, z_coord, jacobian)
src/legoesm/ocean/physics/mpas_physics.py:186:                state.T.data, state.S.data, rho, z_coord, jacobian, cfg_c,
src/legoesm/ocean/dynamics/barotropic.py:5:    d(eta)/dt = -div(H_total * U_bar, H_total * V_bar)
src/legoesm/ocean/dynamics/barotropic.py:202:    H_bathy = cast(state.H_bathy.data, _M, "compute")
src/legoesm/ocean/dynamics/barotropic.py:216:    H_total = jnp.sum(h_k, axis=-1)                        # (6, n, n)
src/legoesm/ocean/dynamics/barotropic.py:218:    H_total = jnp.maximum(H_total, min_water_col)
src/legoesm/ocean/dynamics/barotropic.py:219:    U_bar = jnp.sum(u * h_k, axis=-1) / H_total * mask     # (6, n, n)
src/legoesm/ocean/dynamics/barotropic.py:220:    V_bar = jnp.sum(v * h_k, axis=-1) / H_total * mask
src/legoesm/ocean/dynamics/barotropic.py:264:        H_total_c = jnp.maximum(eta_c + H_bathy, min_water_col) * mask
src/legoesm/ocean/dynamics/barotropic.py:267:        # deta/dt = -div(H_total * U_bar, H_total * V_bar)
src/legoesm/ocean/dynamics/barotropic.py:273:        flux_u = H_total_c * U_bar_c * mask
src/legoesm/ocean/dynamics/barotropic.py:274:        flux_v = H_total_c * V_bar_c * mask
src/legoesm/ocean/dynamics/ocean_pe_cdgrid.py:47:    compute_ocean_jacobian,
src/legoesm/ocean/dynamics/ocean_pe_cdgrid.py:101:    H_bathy = state.H_bathy.data
src/legoesm/ocean/dynamics/ocean_pe_cdgrid.py:112:    J = compute_ocean_jacobian(
src/legoesm/ocean/dynamics/ocean_pe_cdgrid.py:219:    H_total = jnp.maximum(jnp.sum(h_k, axis=-1), min_water_col)
src/legoesm/ocean/dynamics/ocean_pe_cdgrid.py:220:    U_bar_a = jnp.sum(u_a * h_k, axis=-1) / H_total * mask
src/legoesm/ocean/dynamics/ocean_pe_cdgrid.py:221:    V_bar_a = jnp.sum(v_a * h_k, axis=-1) / H_total * mask
src/legoesm/ocean/physics/shortwave_penetration.py:62:    jacobian: jnp.ndarray,
src/legoesm/ocean/physics/shortwave_penetration.py:77:    jacobian : array, shape (...,)
src/legoesm/ocean/physics/shortwave_penetration.py:110:    dz_actual = z_coord_dz_ref * jacobian[..., jnp.newaxis]  # (..., nlev)
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:64:    compute_ocean_jacobian,
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:111:    H_bathy = state.H_bathy.data  # (nCells,)
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:123:    jacobian = compute_ocean_jacobian(
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:415:        u_prime_3d, dz_half, dz, jacobian=jacobian, coeff=config.A_v, is_edge=True,
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:423:            q, dz_half, dz, jacobian=jacobian, coeff=config.K_v, is_edge=False,
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:483:def _vertical_diffusion(field_3d, dz_half, dz, jacobian, coeff, is_edge, mesh):
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:491:    jacobian : jax.Array or None, shape (nCells,) or None
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:494:        If True, field lives on edges (use edge-averaged jacobian).
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:505:    # Scale dz by jacobian if available
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:506:    if jacobian is not None and not is_edge:
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:507:        J = jacobian[:, jnp.newaxis]  # (nCells, 1)
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:508:    elif jacobian is not None and is_edge:
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:511:        J = 0.5 * (jacobian[c1] + jacobian[c2])  # (nEdges,)
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:516:    dz_half_actual = dz_half * J if jacobian is not None else jnp.broadcast_to(
src/legoesm/ocean/dynamics/ocean_pe_mpas.py:519:    dz_actual = dz * J if jacobian is not None else jnp.broadcast_to(
src/legoesm/ocean/physics/bottom_drag/quadratic.py:18:    jacobian: jnp.ndarray,
src/legoesm/ocean/physics/bottom_drag/quadratic.py:27:    jacobian : array (6, n, n)
src/legoesm/ocean/physics/bottom_drag/quadratic.py:37:    dz_bottom = z_coord.dz_ref[-1] * jacobian  # (6, n, n)
src/legoesm/ocean/physics/convection/enhanced_diffusion.py:23:    jacobian: jnp.ndarray,
src/legoesm/ocean/physics/convection/enhanced_diffusion.py:33:    jacobian : array (6, n, n)
src/legoesm/ocean/physics/convection/enhanced_diffusion.py:41:    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
src/legoesm/ocean/physics/convection/enhanced_diffusion.py:61:        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, K),
src/legoesm/ocean/physics/surface_forcing/prescribed.py:20:    jacobian: jnp.ndarray,
src/legoesm/ocean/physics/surface_forcing/prescribed.py:31:    jacobian : array (6, n, n)
src/legoesm/ocean/physics/surface_forcing/prescribed.py:44:    dz_0 = z_coord.dz_ref[0] * jacobian  # (6, n, n)
src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py:26:    jacobian: jnp.ndarray,
src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py:38:    jacobian : array (6, n, n)
src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py:86:    dz_0 = z_coord.dz_ref[0] * jacobian
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:6:    d(eta)/dt = -div(H_total * U_bar, H_total * V_bar)   [cell centers]
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:125:    H_bathy = state.H_bathy.data
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:263:        H_total_c = jnp.maximum(eta_c + H_bathy, min_water_col) * mask
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:266:        H_u = 0.5 * (jnp.roll(H_total_c, 1, axis=1) + H_total_c)
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:270:        H_v_interior = 0.5 * (H_total_c[:-1] + H_total_c[1:])
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:338:        # Bottom drag: -r * U_bar / H_total
src/legoesm/ocean/physics/bottom_drag/integration.py:9:from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
src/legoesm/ocean/physics/bottom_drag/integration.py:53:        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
src/legoesm/ocean/physics/bottom_drag/integration.py:66:        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
src/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:210:    H_bathy = state.H_bathy.data
src/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:240:    H_total_old = jnp.maximum(eta_old + H_bathy, min_water_col)
src/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:241:    H_e_old = _edge_avg(H_total_old, mesh)
src/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:321:    H_total_new = jnp.maximum(eta_new + H_bathy, min_water_col)
src/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:322:    H_e_new = _edge_avg(H_total_new, mesh)
src/legoesm/ocean/init_mpas.py:45:    # Jacobian (eta + H_bathy) / H_max is smooth across coastlines.
src/legoesm/ocean/dynamics/ocean_pe_fc.py:34:    compute_ocean_jacobian,
src/legoesm/ocean/dynamics/ocean_pe_fc.py:73:    H_bathy = state.H_bathy.data
src/legoesm/ocean/dynamics/ocean_pe_fc.py:84:    J = compute_ocean_jacobian(
src/legoesm/ocean/dynamics/ocean_pe_fc.py:190:    H_total = jnp.maximum(jnp.sum(h_k, axis=-1), min_water_col)
src/legoesm/ocean/dynamics/ocean_pe_fc.py:191:    U_bar = jnp.sum(u * h_k, axis=-1) / H_total * mask
src/legoesm/ocean/dynamics/ocean_pe_fc.py:192:    V_bar = jnp.sum(v * h_k, axis=-1) / H_total * mask
src/legoesm/ocean/physics/convection/plume.py:20:    jacobian: jnp.ndarray,
src/legoesm/ocean/physics/convection/plume.py:31:    jacobian : array (6, n, n)
src/legoesm/ocean/physics/convection/plume.py:41:    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
src/legoesm/ocean/physics/surface_forcing/integration.py:9:from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
src/legoesm/ocean/physics/surface_forcing/integration.py:57:        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
src/legoesm/ocean/physics/surface_forcing/integration.py:88:        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
src/legoesm/ocean/physics/surface_forcing/integration.py:111:        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:46:from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:61:    jacobian: jnp.ndarray,
src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:74:    jacobian : (n_lat, n_lon)
src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:108:    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]  # (n_lat, n_lon, nlev-1)
src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:133:    jacobian: jnp.ndarray,
src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:157:    jacobian : (n_lat, n_lon)
src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:168:    dz_actual = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]  # (n_lat, n_lon, nlev)
src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:169:    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]  # (n_lat, n_lon, nlev-1)

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/vertical.py | sed -n '1,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Ocean z-star vertical coordinate.
     2	
     3	z* = H_max * (z + H) / (eta + H)
     4	
     5	where H is the local ocean depth (bathymetry) and eta is the
     6	time-varying sea surface height.
     7	
     8	Unlike the atmosphere's z-star (static terrain Jacobian), the ocean
     9	z-star has a DYNAMIC Jacobian J = (eta + H) / H that is recomputed
    10	at every timestep as eta evolves.
    11	
    12	Level convention: k=0 is surface, k=nlev-1 is deepest.
    13	Reference z values are negative (below sea level).
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	from typing import NamedTuple
    19	
    20	import jax.numpy as jnp
    21	
    22	
    23	class OceanZStarCoordinate(NamedTuple):
    24	    """Static vertical grid definition (independent of eta).
    25	
    26	    Levels indexed surface-to-bottom: k=0 is surface, k=nlev-1 is deepest.
    27	    Reference layer thicknesses assume eta=0 and flat bottom H_max.
    28	
    29	    Fields
    30	    ------
    31	    n_levels : int
    32	        Number of vertical levels.
    33	    H_max : float
    34	        Maximum ocean depth [m] (positive).
    35	    z_full_ref : array
    36	        Reference z* at full (cell center) levels [m], shape (nlev,).
    37	        Negative values (below sea level). z_full_ref[0] is shallowest.
    38	    z_half_ref : array
    39	        Reference z* at half (interface) levels [m], shape (nlev+1,).
    40	        z_half_ref[0] = 0 (surface), z_half_ref[-1] = -H_max (bottom).
    41	    dz_ref : array
    42	        Reference layer thickness [m], shape (nlev,). Positive.
    43	    dz_half_ref : array
    44	        Distance between adjacent full levels [m], shape (nlev-1,).
    45	    """
    46	    n_levels: int
    47	    H_max: float
    48	    z_full_ref: jnp.ndarray
    49	    z_half_ref: jnp.ndarray
    50	    dz_ref: jnp.ndarray
    51	    dz_half_ref: jnp.ndarray
    52	
    53	
    54	def create_ocean_z_star(
    55	    n_levels: int = 50,
    56	    H_max: float = 5500.0,
    57	    dz_surface: float = 10.0,
    58	    dz_deep: float = 200.0,
    59	) -> OceanZStarCoordinate:
    60	    """Create a stretched ocean z-star coordinate.
    61	
    62	    Uses hyperbolic tangent stretching: fine resolution near surface
    63	    (~dz_surface m), coarse at depth (~dz_deep m).
    64	
    65	    Parameters
    66	    ----------
    67	    n_levels : int
    68	        Number of vertical levels.
    69	    H_max : float
    70	        Maximum ocean depth [m].
    71	    dz_surface : float
    72	        Target layer thickness near surface [m].
    73	    dz_deep : float
    74	        Target layer thickness at depth [m].
    75	
    76	    Returns
    77	    -------
    78	    OceanZStarCoordinate : The vertical coordinate.
    79	    """
    80	    if n_levels < 1:
    81	        raise ValueError(
    82	            f"n_levels must be >= 1, got {n_levels!r}",
    83	        )
    84	    if H_max <= 0.0:
    85	        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    86	    if dz_surface <= 0.0:
    87	        raise ValueError(f"dz_surface must be > 0, got {dz_surface!r}")
    88	    if dz_deep <= 0.0:
    89	        raise ValueError(f"dz_deep must be > 0, got {dz_deep!r}")
    90	
    91	    # Stretched grid: dz grows smoothly from dz_surface to dz_deep.
    92	    # Use a normalized distribution then scale to match H_max.
    93	    from legoesm.core.precision import get_policy
    94	    k = jnp.arange(n_levels, dtype=get_policy().control)
    95	
    96	    # Layer thickness profile: linear growth from dz_surface to dz_deep
    97	    dz_raw = dz_surface + k * (dz_deep - dz_surface) / jnp.maximum(n_levels - 1.0, 1.0)
    98	
    99	    # Normalize so total thickness matches H_max
   100	    scale = H_max / jnp.sum(dz_raw)
   101	    dz_ref = dz_raw * scale
   102	
   103	    # Interface depths from cumulative sum (surface=0, bottom=-H_max)
   104	    z_half_ref = jnp.concatenate([
   105	        jnp.array([0.0], dtype=dz_ref.dtype),
   106	        -jnp.cumsum(dz_ref),
   107	    ])
   108	
   109	    # Full level depths (cell centers)
   110	    z_full_ref = 0.5 * (z_half_ref[:-1] + z_half_ref[1:])
   111	
   112	    # Layer thicknesses (positive)
   113	    dz_ref = z_half_ref[:-1] - z_half_ref[1:]  # positive since z[k] > z[k+1]
   114	
   115	    # Distance between full levels
   116	    dz_half_ref = z_full_ref[:-1] - z_full_ref[1:]  # positive
   117	
   118	    return OceanZStarCoordinate(
   119	        n_levels=n_levels,
   120	        H_max=H_max,
   121	        z_full_ref=z_full_ref,
   122	        z_half_ref=z_half_ref,
   123	        dz_ref=dz_ref,
   124	        dz_half_ref=dz_half_ref,
   125	    )
   126	
   127	
   128	def compute_layer_thickness(
   129	    eta: jnp.ndarray,
   130	    H_bathy: jnp.ndarray,
   131	    z_coord: OceanZStarCoordinate,
   132	    min_water_column_m: float | None = None,
   133	) -> jnp.ndarray:
   134	    """Compute actual layer thickness incorporating eta and bathymetry.
   135	
   136	    h_k = dz_ref[k] * (eta + H_bathy) / H_max
   137	
   138	    The dynamic Jacobian J = (eta + H_bathy) / H_max modifies
   139	    reference thicknesses to account for the actual water column.
   140	
   141	    Parameters
   142	    ----------
   143	    eta : array
   144	        Sea surface height [m], shape (...).
   145	    H_bathy : array
   146	        Local bathymetry depth [m], shape (...). Positive.
   147	    z_coord : OceanZStarCoordinate
   148	        Vertical coordinate.
   149	    min_water_column_m : float or None
   150	        Optional lower bound for local water-column thickness
   151	        ``eta + H_bathy`` [m]. When set, Jacobian/thickness values are
   152	        clipped to avoid dry or negative columns.
   153	
   154	    Returns
   155	    -------
   156	    array : Layer thickness [m], shape (..., nlev). Positive.
   157	    """
   158	    J = compute_ocean_jacobian(
   159	        eta, H_bathy, z_coord, min_water_column_m=min_water_column_m,
   160	    )
   161	    return z_coord.dz_ref * J[..., jnp.newaxis]
   162	
   163	
   164	def compute_ocean_jacobian(
   165	    eta: jnp.ndarray,
   166	    H_bathy: jnp.ndarray,
   167	    z_coord: OceanZStarCoordinate,
   168	    min_water_column_m: float | None = None,
   169	) -> jnp.ndarray:
   170	    """Compute the dynamic z-star Jacobian.
   171	
   172	    J = (eta + H_bathy) / H_max
   173	
   174	    This is recomputed at every timestep as eta evolves.
   175	
   176	    Parameters
   177	    ----------
   178	    eta : array
   179	        Sea surface height [m], shape (...).
   180	    H_bathy : array
   181	        Local bathymetry depth [m], shape (...). Positive.
   182	    z_coord : OceanZStarCoordinate
   183	        Vertical coordinate (provides H_max).
   184	    min_water_column_m : float or None
   185	        Optional lower bound for local water-column thickness
   186	        ``eta + H_bathy`` [m].
   187	
   188	    Returns
   189	    -------
   190	    array : Jacobian, shape (...).
   191	    """
   192	    water_col = eta + H_bathy
   193	    if min_water_column_m is not None:
   194	        min_col = jnp.asarray(min_water_column_m, dtype=water_col.dtype)
   195	        water_col = jnp.maximum(water_col, min_col)
   196	    return water_col / z_coord.H_max
   197	
   198	
   199	def upwind_vertical_gradient(
   200	    field: jnp.ndarray,
   201	    dz_half: jnp.ndarray,
   202	    w: jnp.ndarray,
   203	    *,
   204	    eps: float = 1.0e-12,
   205	) -> jnp.ndarray:
   206	    """Compute first-order upwind d(field)/dz at full levels.
   207	
   208	    Assumes levels are indexed surface-to-bottom (k=0 at surface).
   209	    Caller must use consistent coordinates: both ``dz_half`` and ``w``
   210	    should be in the same vertical coordinate (physical z or z*).
   211	
   212	    Parameters
   213	    ----------
   214	    field : array
   215	        Field at full levels, shape (..., nlev).
   216	    dz_half : array
   217	        Full-level spacing [m], shape (..., nlev-1). Positive.
   218	    w : array
   219	        Vertical velocity [m/s], shape (..., nlev).
   220	        Only its sign is used (upwind direction). Positive = upward.

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/vertical_mixing/kpp.py | sed -n '1,430p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""LMD94-style K-Profile Parameterization (KPP).
     2	
     3	Boundary-layer parameterization following Large, McWilliams & Doney (1994)
     4	with:
     5	
     6	- Bulk Richardson number BL-depth diagnosis with linear interpolation
     7	  of the crossing depth between model levels.
     8	- Turbulent velocity scales w_s(sigma) from surface forcing (u_star, B_f).
     9	- Cubic shape function G(sigma) = sigma * (1 - sigma)^2.
    10	- Non-local tracer transport for unstable (convective) conditions only.
    11	- Interior mixing: Richardson-number dependent + convective instability
    12	  enhancement for statically unstable layers below the BL.
    13	
    14	The caller should provide surface wind stress and buoyancy flux when
    15	available.  If tau_x/tau_y are None, a simplified u_star proxy from
    16	surface speed is used.
    17	
    18	References
    19	----------
    20	- Large, W. G., McWilliams, J. C., & Doney, S. C. (1994). Oceanic
    21	  vertical mixing: A review and a model with a nonlocal boundary layer
    22	  parameterization. Rev. Geophys., 32, 363-403.
    23	"""
    24	
    25	from __future__ import annotations
    26	
    27	import jax
    28	import jax.numpy as jnp
    29	
    30	from legoesm import constants
    31	from legoesm.ocean.eos import (
    32	    wright_eos,
    33	    compute_buoyancy_frequency,
    34	    compute_hydrostatic_pressure,
    35	    rho_0 as rho_0_ref,
    36	)
    37	from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
    38	from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
    39	from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput
    40	from legoesm.ocean.vertical import OceanZStarCoordinate
    41	
    42	_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)
    43	
    44	
    45	def _boundary_layer_depth(
    46	    rho: jnp.ndarray,
    47	    u: jnp.ndarray,
    48	    v: jnp.ndarray,
    49	    z_coord: OceanZStarCoordinate,
    50	    jacobian: jnp.ndarray,
    51	    u_star: jnp.ndarray,
    52	    B_f: jnp.ndarray,
    53	    cfg: KPPConfig,
    54	    g: float = constants.g,
    55	    h_bl_prev: jnp.ndarray | None = None,
    56	) -> jnp.ndarray:
    57	    """Estimate boundary layer depth h via bulk Richardson number.
    58	
    59	    Uses linear interpolation to find the depth where Ri_b crosses
    60	    Ri_crit, rather than snapping to the nearest model level.
    61	
    62	    Returns shape (...) boundary layer depth [m, positive downward].
    63	    """
    64	    eps = _EPS
    65	    nlev = rho.shape[-1]
    66	
    67	    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    68	    # Depth of cell centers below surface (positive downward)
    69	    z_depth = jnp.cumsum(dz_actual, axis=-1) - 0.5 * dz_actual
    70	
    71	    # Density and velocity differences from surface
    72	    delta_rho = rho - rho[..., :1]
    73	    delta_u = u - u[..., :1]
    74	    delta_v = v - v[..., :1]
    75	    delta_V2 = delta_u**2 + delta_v**2
    76	
    77	    # LMD94 Eq. 23: V_t^2 = Cv * sqrt(|N2|) / sqrt(c_s * epsilon) *
    78	    #   max(Ri_crit * h - d, 0) * d / h
    79	    # Uses h_bl from the previous time step to break the coupling.
    80	    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
    81	    N2_full = jnp.concatenate([N2[..., :1], N2], axis=-1)
    82	    max_depth = z_depth[..., -1]
    83	    h_est = max_depth if h_bl_prev is None else h_bl_prev
    84	    h_safe = jnp.maximum(h_est[..., jnp.newaxis], eps)
    85	    V_t2 = (cfg.Cv * jnp.sqrt(jnp.maximum(jnp.abs(N2_full), 0.0))
    86	            / jnp.sqrt(jnp.maximum(cfg.c_s * cfg.epsilon_lmd, eps))
    87	            * jnp.maximum(cfg.Ri_crit * h_safe - z_depth, 0.0)
    88	            * z_depth / h_safe)
    89	
    90	    # Bulk Richardson number
    91	    Ri_b = (g * delta_rho * z_depth) / (
    92	        rho_0_ref * jnp.maximum(delta_V2 + V_t2, eps)
    93	    )
    94	
    95	    # --- Differentiable soft interpolation of crossing depth ---
    96	    # Instead of argmax (non-differentiable), use a sigmoid-weighted
    97	    # average over all levels.  Each level contributes a weight
    98	    # proportional to how much Ri_b crosses Ri_crit there.
    99	    #
   100	    # Weight at level k = sigmoid(sharpness * (Ri_b[k] - Ri_crit))
   101	    #                    - sigmoid(sharpness * (Ri_b[k-1] - Ri_crit))
   102	    # This is ~1 at the crossing level and ~0 elsewhere.
   103	    sharpness = cfg.crossing_sharpness
   104	    sig = jax.nn.sigmoid(sharpness * (Ri_b - cfg.Ri_crit))  # (..., nlev)
   105	
   106	    # Crossing weight: difference of adjacent sigmoid values.  ``jnp.pad``
   107	    # along the trailing axis is one HLO op; the previous
   108	    # ``concatenate([zeros_like(sig[..., :1]), sig[..., :-1]])`` allocated
   109	    # a fresh zero buffer and concatenated.
   110	    pad_axes = ((0, 0),) * (sig.ndim - 1)
   111	    sig_prev = jnp.pad(sig[..., :-1], (*pad_axes, (1, 0)))
   112	    w_cross = sig - sig_prev  # (..., nlev), peaks at crossing level
   113	    w_cross = jnp.maximum(w_cross, 0.0)
   114	    w_sum = jnp.sum(w_cross, axis=-1, keepdims=True)
   115	    w_norm = w_cross / jnp.maximum(w_sum, eps)
   116	
   117	    # Crossing-based depth estimate
   118	    h_crossing = jnp.sum(w_norm * z_depth, axis=-1)  # (...)
   119	
   120	    # Fallback for columns where Ri_b never crosses Ri_crit:
   121	    # - If column is mostly unstable (sig ≈ 0): BL extends to full depth
   122	    # - If column is mostly stable (sig ≈ 1): BL is one layer
   123	    column_stability = jnp.mean(sig, axis=-1)  # 0 = all unstable, 1 = all stable
   124	    max_depth = z_depth[..., -1]
   125	    min_depth = dz_actual[..., 0]
   126	    h_fallback = (1.0 - column_stability) * max_depth + column_stability * min_depth
   127	
   128	    # Blend: use crossing depth when crossing signal is strong, fallback otherwise
   129	    crossing_strength = w_sum[..., 0]
   130	    blend = jax.nn.sigmoid(cfg.crossing_sharpness * (crossing_strength - cfg.crossing_threshold))
   131	    h = blend * h_crossing + (1.0 - blend) * h_fallback
   132	
   133	    # At least one layer thick
   134	    h = jnp.maximum(h, dz_actual[..., 0])
   135	
   136	    return h
   137	
   138	
   139	def kpp_vertical_mixing(
   140	    u: jnp.ndarray,
   141	    v: jnp.ndarray,
   142	    T: jnp.ndarray,
   143	    S: jnp.ndarray,
   144	    rho: jnp.ndarray,
   145	    eta: jnp.ndarray,
   146	    z_coord: OceanZStarCoordinate,
   147	    jacobian: jnp.ndarray,
   148	    cfg: KPPConfig,
   149	    g: float = constants.g,
   150	    tau_x: jnp.ndarray | None = None,
   151	    tau_y: jnp.ndarray | None = None,
   152	    B_f: jnp.ndarray | None = None,
   153	    Q_sfc_T: jnp.ndarray | None = None,
   154	    Q_sfc_S: jnp.ndarray | None = None,
   155	    h_bl_prev: jnp.ndarray | None = None,
   156	) -> VerticalMixingOutput:
   157	    """Apply LMD94-style KPP vertical mixing.
   158	
   159	    Parameters
   160	    ----------
   161	    u, v : array (6, n, n, nlev)
   162	    T, S : array (6, n, n, nlev)
   163	    rho : array (6, n, n, nlev)
   164	    eta : array (6, n, n)
   165	    z_coord : OceanZStarCoordinate
   166	    jacobian : array (6, n, n)
   167	    cfg : KPPConfig
   168	    g : float
   169	    tau_x, tau_y : array (6, n, n) or None
   170	        Surface wind stress [Pa]. If None, a proxy from surface speed is used.
   171	    B_f : array (6, n, n) or None
   172	        Surface buoyancy flux [m^2/s^3], positive = destabilizing (convective).
   173	        If None, estimated from surface density gradient.
   174	    Q_sfc_T : array (6, n, n) or None
   175	        Surface kinematic heat flux [K*m/s] for non-local transport (LMD94
   176	        Eq. 19).  If None, falls back to diagnosed K_sfc * dT/dz proxy.
   177	    Q_sfc_S : array (6, n, n) or None
   178	        Surface kinematic salt flux [PSU*m/s]. Same convention as Q_sfc_T.
   179	    h_bl_prev : array (6, n, n) or None
   180	        BL depth from the previous time step [m, positive downward].
   181	        Used to break the implicit V_t-h_bl coupling in the Ri_b diagnosis
   182	        (LMD94 Eq. 23).  If None, uses the full column depth as estimate.
   183	
   184	    Returns
   185	    -------
   186	    VerticalMixingOutput
   187	    """
   188	    eps = _EPS
   189	    nlev = u.shape[-1]
   190	
   191	    # --- Friction velocity ---
   192	    if tau_x is not None and tau_y is not None:
   193	        # Proper u_star from wind stress: u_star = sqrt(|tau| / rho_0)
   194	        tau_mag = jnp.sqrt(tau_x**2 + tau_y**2 + eps)
   195	        u_star = jnp.sqrt(tau_mag / rho_0_ref)
   196	    else:
   197	        # Simplified proxy: u_star ~ 0.01 * |U_surface|
   198	        speed_sfc = jnp.sqrt(u[..., 0]**2 + v[..., 0]**2 + eps)
   199	        u_star = jnp.maximum(speed_sfc * 0.01, 1e-4)
   200	
   201	    # --- Surface buoyancy flux ---
   202	    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
   203	    dz_half0 = 0.5 * (dz_actual[..., 0] + dz_actual[..., 1])
   204	    if B_f is None:
   205	        # Estimate from near-surface density gradient
   206	        drho_dz_sfc = (rho[..., 0] - rho[..., 1]) / jnp.maximum(dz_half0, eps)
   207	        B_f = -g / rho_0_ref * cfg.K_bg * drho_dz_sfc  # simplified proxy
   208	
   209	    # --- Boundary layer depth ---
   210	    h_bl = _boundary_layer_depth(
   211	        rho, u, v, z_coord, jacobian, u_star, B_f, cfg, g,
   212	        h_bl_prev=h_bl_prev,
   213	    )
   214	
   215	    # --- Depth coordinate ---
   216	    z_depth = jnp.cumsum(dz_actual, axis=-1) - 0.5 * dz_actual
   217	    sigma = z_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
   218	
   219	    # --- Shape function G(sigma) = sigma * (1 - sigma)^2 ---
   220	    sigma_clip = jnp.clip(sigma, 0.0, 1.0)
   221	    G = sigma_clip * (1.0 - sigma_clip) ** 2
   222	
   223	    # --- Turbulent velocity scale w_s(sigma) (LMD94 Appendix B) ---
   224	    # w_s depends on stability (B_f) and depth d = sigma * h_bl
   225	    d = sigma_clip * h_bl[..., jnp.newaxis]
   226	    # Monin-Obukhov length: L_MO = u_star^3 / (kappa * B_f)
   227	    # Use copysign(eps, B_f) to preserve the sign of B_f near zero,
   228	    # preventing a stability classification flip (issue #168 bug 1).
   229	    B_f_safe = jnp.where(
   230	        jnp.abs(B_f[..., jnp.newaxis]) > eps,
   231	        B_f[..., jnp.newaxis],
   232	        jnp.copysign(eps, B_f[..., jnp.newaxis]),
   233	    )
   234	    L_MO = u_star[..., jnp.newaxis]**3 / (cfg.kappa_vk * B_f_safe)
   235	    zeta_kpp = d / L_MO
   236	
   237	    # LMD94 Appendix B turbulent velocity scales:
   238	    # Stable (B_f <= 0): w_s = kappa * u_star / (1 + 5*zeta)
   239	    # Unstable, weakly (epsilon*d < |L|): w_s = kappa * u_star * phi_m^{-1}
   240	    #   where phi_m^{-1} = (1 - 16*zeta)^{1/4}
   241	    # Unstable, strongly convective (epsilon*d > |L|):
   242	    #   w_s = (kappa * (u_star^3 + c_b * kappa * (-B_f) * d))^{1/3}
   243	    is_unstable = B_f[..., jnp.newaxis] > 0.0
   244	    epsilon_lmd = cfg.epsilon_lmd
   245	
   246	    # Weakly unstable: phi_m^{-1} formulation
   247	    w_s_weak = (cfg.kappa_vk * u_star[..., jnp.newaxis]
   248	                * jnp.power(jnp.maximum(1.0 + 16.0 * jnp.abs(zeta_kpp), 1.0), 0.25))
   249	
   250	    # Strongly convective: includes convective velocity scale
   251	    Bf_pos = jnp.maximum(B_f[..., jnp.newaxis], 0.0)
   252	    w_s_conv = jnp.power(
   253	        cfg.kappa_vk * (u_star[..., jnp.newaxis]**3
   254	                        + cfg.c_b * cfg.kappa_vk * Bf_pos * d),
   255	        1.0 / 3.0,
   256	    )
   257	
   258	    # Transition: use convective scale when epsilon*d > |L_MO|
   259	    is_strongly_convective = epsilon_lmd * d > jnp.abs(L_MO)
   260	    w_s_unstable = jnp.where(is_strongly_convective, w_s_conv, w_s_weak)
   261	
   262	    # Stable: standard suppression
   263	    w_s_stable = (cfg.kappa_vk * u_star[..., jnp.newaxis]
   264	                  / jnp.maximum(1.0 + 5.0 * jnp.maximum(zeta_kpp, 0.0), 1.0))
   265	    w_s = jnp.where(is_unstable, w_s_unstable, w_s_stable)
   266	    w_s = jnp.maximum(w_s, 1e-10)
   267	
   268	    # --- BL diffusivity at full levels ---
   269	    K_bl_full = h_bl[..., jnp.newaxis] * w_s * G
   270	    K_bl_full = jnp.minimum(K_bl_full, cfg.K_max)
   271	
   272	    # --- Interior mixing: Richardson-number dependent ---
   273	    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
   274	    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
   275	    du = u[..., :-1] - u[..., 1:]
   276	    dv = v[..., :-1] - v[..., 1:]
   277	    S2 = (du**2 + dv**2) / jnp.maximum(dz_half**2, eps)
   278	    Ri_int = N2 / jnp.maximum(S2, eps)
   279	    # LMD94 interior shear instability: K = K_0 * (1 - (Ri/Ri_0)^2)^3
   280	    # for Ri < Ri_0, zero above.
   281	    Ri_ratio = jnp.clip(Ri_int / cfg.Ri_0, 0.0, 1.0)
   282	    K_interior = cfg.K_0_shear * (1.0 - Ri_ratio**2) ** 3 + cfg.K_bg
   283	
   284	    # Interior static instability: enhanced mixing where N2 < 0
   285	    K_conv = jnp.where(N2 < cfg.Ri_conv, cfg.K_conv, 0.0)
   286	    K_interior = K_interior + K_conv
   287	
   288	    # --- K at interfaces (average of full level K_bl) ---
   289	    K_bl_half = 0.5 * (K_bl_full[..., :-1] + K_bl_full[..., 1:])
   290	
   291	    # sigma at interfaces
   292	    z_half_depth = 0.5 * (z_depth[..., :-1] + z_depth[..., 1:])
   293	    sigma_half = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
   294	    in_bl = sigma_half < 1.0
   295	
   296	    # Combine BL and interior
   297	    K_v = jnp.where(in_bl, K_bl_half, K_interior) + cfg.K_bg
   298	    A_v = jnp.where(in_bl, K_bl_half * 1.0, K_interior) + cfg.A_bg
   299	    K_v = jnp.minimum(K_v, cfg.K_max)
   300	    A_v = jnp.minimum(A_v, cfg.K_max)
   301	
   302	    # --- Apply diffusion ---
   303	    vel = jnp.stack([u, v], axis=0)
   304	    vel_tend = jax.vmap(
   305	        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, A_v),
   306	        in_axes=0, out_axes=0,
   307	    )(vel)
   308	
   309	    tracers = jnp.stack([T, S], axis=0)
   310	    tr_tend = jax.vmap(
   311	        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, K_v),
   312	        in_axes=0, out_axes=0,
   313	    )(tracers)
   314	
   315	    # --- Non-local flux for T, S (LMD94 Eq. 19) ---
   316	    #
   317	    # LMD94 defines a counter-gradient term:
   318	    #   gamma_T(sigma) = C_s * Q_0 / (w_s(sigma) * h)   [K/m]
   319	    # where Q_0 is the surface kinematic heat flux [K*m/s].
   320	    #
   321	    # The non-local tendency is  -d/dz(K_bl * gamma_T).
   322	    # Substituting K_bl = h * w_s * G(sigma):
   323	    #   K_bl * gamma_T = h * w_s * G * C_s * Q_0 / (w_s * h) = C_s * Q_0 * G(sigma)
   324	    #
   325	    # So the non-local tendency reduces to:
   326	    #   dT/dt_nonlocal = -d/dz[ C_s * Q_0 * G(sigma) ]            [K/s]
   327	    #
   328	    # We discretize this as the vertical divergence of the non-local
   329	    # flux F_nl = C_s * Q_0 * G(sigma) evaluated at interfaces.
   330	
   331	    # Surface kinematic heat/salt flux for non-local transport (LMD94 Eq. 19).
   332	    # Use the IMPOSED surface flux when available (from bulk formulas or
   333	    # prescribed forcing).  Fall back to diagnosed K_sfc * dT/dz proxy
   334	    # only when no external flux is provided (issue #168 bug 2).
   335	    if Q_sfc_T is not None:
   336	        Q_T = Q_sfc_T  # [K*m/s]
   337	    else:
   338	        dT_dz_sfc = (T[..., 0] - T[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
   339	        K_sfc = K_bl_full[..., 0]
   340	        Q_T = K_sfc * dT_dz_sfc
   341	
   342	    if Q_sfc_S is not None:
   343	        Q_S = Q_sfc_S  # [PSU*m/s]
   344	    else:
   345	        dS_dz_sfc = (S[..., 0] - S[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
   346	        K_sfc = K_bl_full[..., 0]
   347	        Q_S = K_sfc * dS_dz_sfc
   348	
   349	    # Only apply non-local transport for unstable (convective) columns.
   350	    is_unstable_col = B_f > 0.0
   351	
   352	    # G(sigma) at interior interfaces (half levels between full levels)
   353	    sigma_half_full = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
   354	    sigma_half_clip = jnp.clip(sigma_half_full, 0.0, 1.0)
   355	    G_half = sigma_half_clip * (1.0 - sigma_half_clip) ** 2  # (..., nlev-1)
   356	
   357	    in_bl_full = sigma < 1.0
   358	
   359	    # --- Temperature non-local tendency ---
   360	    # Non-local flux at interfaces: F_nl = C_s * Q_T * G_half  [K*m/s]
   361	    F_T = cfg.gamma_T * Q_T[..., jnp.newaxis] * G_half  # (..., nlev-1)
   362	    # Tendency = -dF/dz at full levels (zero-flux BCs at surface and bottom)
   363	    dT_nonlocal_top = -F_T[..., :1] / dz_actual[..., :1]
   364	    dT_nonlocal_int = (F_T[..., :-1] - F_T[..., 1:]) / dz_actual[..., 1:-1]
   365	    dT_nonlocal_bot = F_T[..., -1:] / dz_actual[..., -1:]
   366	    dT_nonlocal = jnp.concatenate(
   367	        [dT_nonlocal_top, dT_nonlocal_int, dT_nonlocal_bot], axis=-1
   368	    )  # (..., nlev)  [K/s]
   369	    dT_nonlocal = jnp.where(
   370	        in_bl_full & is_unstable_col[..., jnp.newaxis], dT_nonlocal, 0.0
   371	    )
   372	
   373	    # --- Salinity non-local tendency ---
   374	    F_S = cfg.gamma_S * Q_S[..., jnp.newaxis] * G_half  # (..., nlev-1)
   375	    dS_nonlocal_top = -F_S[..., :1] / dz_actual[..., :1]
   376	    dS_nonlocal_int = (F_S[..., :-1] - F_S[..., 1:]) / dz_actual[..., 1:-1]
   377	    dS_nonlocal_bot = F_S[..., -1:] / dz_actual[..., -1:]
   378	    dS_nonlocal = jnp.concatenate(
   379	        [dS_nonlocal_top, dS_nonlocal_int, dS_nonlocal_bot], axis=-1
   380	    )  # (..., nlev)  [psu/s]
   381	    dS_nonlocal = jnp.where(
   382	        in_bl_full & is_unstable_col[..., jnp.newaxis], dS_nonlocal, 0.0
   383	    )
   384	
   385	    return VerticalMixingOutput(
   386	        du_dt=vel_tend[0],
   387	        dv_dt=vel_tend[1],
   388	        dT_dt=tr_tend[0] + dT_nonlocal,
   389	        dS_dt=tr_tend[1] + dS_nonlocal,
   390	        K_v=K_v,
   391	        A_v=A_v,
   392	    )

exec
/bin/zsh -lc "rg -n \"Q_sfc_T|kinematic heat flux|surface.*heat|dT_dt.*Q\" src/legoesm/ocean -g'*.py'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/ocean/state.py:77:        subsurface SW penetration heating.
src/legoesm/ocean/state.py:79:        Net surface heat flux (positive into ocean) [W/m²].
src/legoesm/ocean/simple_ocean_mpas.py:94:    dT_dt = (sw_net + lw_net - shflx - lhflx + config.Q_flux) / C_mix
src/legoesm/ocean/conservation.py:3:Volume (free surface), heat (T), and salt (S) conservation via
src/legoesm/ocean/simple_ocean.py:121:    dT_dt = (sw_net + lw_net - shflx - lhflx + config.Q_flux) / C_mix
src/legoesm/ocean/physics/surface_forcing/prescribed.py:1:"""Prescribed surface forcing: fixed wind stress and heat/freshwater fluxes."""
src/legoesm/ocean/physics/mpas_physics.py:150:                dT_dt = dT_dt.at[:, 0].add(cfg.Q_net * inv_rho_csw_dz * mask)
src/legoesm/ocean/physics/shortwave_penetration.py:1:"""Subsurface shortwave penetration heating.
src/legoesm/ocean/physics/surface_forcing/config.py:12:    Q_net: float = 0.0           # Net surface heat flux [W/m^2] (+ into ocean)
src/legoesm/ocean/physics/vertical_mixing/kpp.py:153:    Q_sfc_T: jnp.ndarray | None = None,
src/legoesm/ocean/physics/vertical_mixing/kpp.py:174:    Q_sfc_T : array (6, n, n) or None
src/legoesm/ocean/physics/vertical_mixing/kpp.py:175:        Surface kinematic heat flux [K*m/s] for non-local transport (LMD94
src/legoesm/ocean/physics/vertical_mixing/kpp.py:178:        Surface kinematic salt flux [PSU*m/s]. Same convention as Q_sfc_T.
src/legoesm/ocean/physics/vertical_mixing/kpp.py:319:    # where Q_0 is the surface kinematic heat flux [K*m/s].
src/legoesm/ocean/physics/vertical_mixing/kpp.py:335:    if Q_sfc_T is not None:
src/legoesm/ocean/physics/vertical_mixing/kpp.py:336:        Q_T = Q_sfc_T  # [K*m/s]

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/mpas_physics.py | sed -n '120,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   120	        # --- Prescribed wind / Q_net / E-P (also reused under "combined") ---
   121	        if apply_wind_block:
   122	            cfg = sf_config.prescribed
   123	
   124	            dz_0_cell = z_coord.dz_ref[0] * jacobian  # (nCells,)
   125	
   126	            # Compute cell-centered wind stress from latitude
   127	            from legoesm.ocean.physics.surface_forcing.wind_profiles import compute_wind_stress
   128	            tau_x, tau_y = compute_wind_stress(mesh.grid_lat, cfg)
   129	
   130	            # Project cell-centered wind stress onto edge normals.
   131	            # Average tau from the two cells sharing each edge, then dot
   132	            # with the edge-normal direction (angleEdge).
   133	            tau_x_e = 0.5 * (tau_x[c1] + tau_x[c2])
   134	            tau_y_e = 0.5 * (tau_y[c1] + tau_y[c2])
   135	            tau_n = (tau_x_e * jnp.cos(mesh.angleEdge)
   136	                     + tau_y_e * jnp.sin(mesh.angleEdge))
   137	
   138	            # Edge top-layer thickness
   139	            dz_0_e = 0.5 * (dz_0_cell[c1] + dz_0_cell[c2])
   140	            inv_rho_dz_e = 1.0 / (rho_0_ref * jnp.maximum(dz_0_e, 1e-10))
   141	
   142	            # Apply wind stress to top layer only
   143	            du_dt = du_dt.at[:, 0].add(tau_n * inv_rho_dz_e)
   144	
   145	            # Heat flux: dT/dt = Q_net / (rho_0 * c_sw * dz_0)
   146	            if cfg.Q_net != 0.0:
   147	                from legoesm.ocean.eos import c_sw
   148	                inv_rho_csw_dz = 1.0 / (
   149	                    rho_0_ref * c_sw * jnp.maximum(dz_0_cell, 1e-10))
   150	                dT_dt = dT_dt.at[:, 0].add(cfg.Q_net * inv_rho_csw_dz * mask)
   151	
   152	            # E-P virtual salt flux
   153	            if cfg.E_minus_P != 0.0:
   154	                inv_dz = 1.0 / jnp.maximum(dz_0_cell, 1e-10)
   155	                dS_dt = dS_dt.at[:, 0].add(
   156	                    state.S.data[:, 0] * cfg.E_minus_P * inv_dz * mask)
   157	
   158	        # --- T/S restoring (under "restoring" or "combined") ---
   159	        if apply_restoring:
   160	            from legoesm.ocean.physics.surface_forcing.restoring import (
   161	                restoring_surface_forcing,
   162	            )
   163	            cfg_r = sf_config.restoring
   164	            r_out = restoring_surface_forcing(
   165	                state.T.data, state.S.data, mesh, cfg_r,
   166	            )
   167	            # restoring_surface_forcing does not mask land; do it here so
   168	            # land-cell tracer values are not driven by the restoring term.
   169	            dT_dt = dT_dt + r_out.dT_dt * mask[:, None]
   170	            dS_dt = dS_dt + r_out.dS_dt * mask[:, None]
   171	
   172	        # --- Convective adjustment (enhanced diffusion where N²<0) ---
   173	        if apply_convection:
   174	            from legoesm.ocean.physics.convection.enhanced_diffusion import (
   175	                enhanced_diffusion_convection,
   176	            )
   177	            from legoesm.ocean.eos import compute_ocean_rho
   178	            cfg_c = conv_config.enhanced_diffusion
   179	            # Match the lat-lon convection integration: use the default
   180	            # (Wright) EOS for the ρ used in the static-stability check,
   181	            # even when the dycore is configured with linear EOS.  This
   182	            # is a known approximation — the EOS choice only affects the
   183	            # static-stability ranking, not the dycore tendencies.
   184	            rho = compute_ocean_rho(state, z_coord, jacobian)
   185	            c_out = enhanced_diffusion_convection(
   186	                state.T.data, state.S.data, rho, z_coord, jacobian, cfg_c,
   187	            )
   188	            dT_dt = dT_dt + c_out.dT_dt * mask[:, None]
   189	            dS_dt = dS_dt + c_out.dS_dt * mask[:, None]
   190	
   191	        return MPASOceanTendencies(
   192	            du_dt=Field(data=du_dt, name="du_dt",
   193	                        dims=("nEdges", "nlev"), units="m/s²"),
   194	            dT_dt=Field(data=dT_dt, name="dT_dt",
   195	                        dims=("nCells", "nlev"), units="degC/s"),
   196	            dS_dt=Field(data=dS_dt, name="dS_dt",
   197	                        dims=("nCells", "nlev"), units="PSU/s"),
   198	            deta_dt=Field(data=deta_dt, name="deta_dt",
   199	                          dims=("nCells",), units="m/s"),
   200	        )
   201	
   202	    return physics_fn

exec
/bin/zsh -lc "rg -n \"kpp_vertical_mixing\\(|Q_sfc_T|B_f\" src/legoesm/ocean/physics -g'*.py'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/ocean/physics/vertical_mixing/kpp.py:8:- Turbulent velocity scales w_s(sigma) from surface forcing (u_star, B_f).
src/legoesm/ocean/physics/vertical_mixing/kpp.py:52:    B_f: jnp.ndarray,
src/legoesm/ocean/physics/vertical_mixing/kpp.py:139:def kpp_vertical_mixing(
src/legoesm/ocean/physics/vertical_mixing/kpp.py:152:    B_f: jnp.ndarray | None = None,
src/legoesm/ocean/physics/vertical_mixing/kpp.py:153:    Q_sfc_T: jnp.ndarray | None = None,
src/legoesm/ocean/physics/vertical_mixing/kpp.py:171:    B_f : array (6, n, n) or None
src/legoesm/ocean/physics/vertical_mixing/kpp.py:174:    Q_sfc_T : array (6, n, n) or None
src/legoesm/ocean/physics/vertical_mixing/kpp.py:178:        Surface kinematic salt flux [PSU*m/s]. Same convention as Q_sfc_T.
src/legoesm/ocean/physics/vertical_mixing/kpp.py:204:    if B_f is None:
src/legoesm/ocean/physics/vertical_mixing/kpp.py:207:        B_f = -g / rho_0_ref * cfg.K_bg * drho_dz_sfc  # simplified proxy
src/legoesm/ocean/physics/vertical_mixing/kpp.py:211:        rho, u, v, z_coord, jacobian, u_star, B_f, cfg, g,
src/legoesm/ocean/physics/vertical_mixing/kpp.py:224:    # w_s depends on stability (B_f) and depth d = sigma * h_bl
src/legoesm/ocean/physics/vertical_mixing/kpp.py:226:    # Monin-Obukhov length: L_MO = u_star^3 / (kappa * B_f)
src/legoesm/ocean/physics/vertical_mixing/kpp.py:227:    # Use copysign(eps, B_f) to preserve the sign of B_f near zero,
src/legoesm/ocean/physics/vertical_mixing/kpp.py:229:    B_f_safe = jnp.where(
src/legoesm/ocean/physics/vertical_mixing/kpp.py:230:        jnp.abs(B_f[..., jnp.newaxis]) > eps,
src/legoesm/ocean/physics/vertical_mixing/kpp.py:231:        B_f[..., jnp.newaxis],
src/legoesm/ocean/physics/vertical_mixing/kpp.py:232:        jnp.copysign(eps, B_f[..., jnp.newaxis]),
src/legoesm/ocean/physics/vertical_mixing/kpp.py:234:    L_MO = u_star[..., jnp.newaxis]**3 / (cfg.kappa_vk * B_f_safe)
src/legoesm/ocean/physics/vertical_mixing/kpp.py:238:    # Stable (B_f <= 0): w_s = kappa * u_star / (1 + 5*zeta)
src/legoesm/ocean/physics/vertical_mixing/kpp.py:242:    #   w_s = (kappa * (u_star^3 + c_b * kappa * (-B_f) * d))^{1/3}
src/legoesm/ocean/physics/vertical_mixing/kpp.py:243:    is_unstable = B_f[..., jnp.newaxis] > 0.0
src/legoesm/ocean/physics/vertical_mixing/kpp.py:251:    Bf_pos = jnp.maximum(B_f[..., jnp.newaxis], 0.0)
src/legoesm/ocean/physics/vertical_mixing/kpp.py:335:    if Q_sfc_T is not None:
src/legoesm/ocean/physics/vertical_mixing/kpp.py:336:        Q_T = Q_sfc_T  # [K*m/s]
src/legoesm/ocean/physics/vertical_mixing/kpp.py:350:    is_unstable_col = B_f > 0.0
src/legoesm/ocean/physics/vertical_mixing/integration.py:91:        out = kpp_vertical_mixing(
src/legoesm/ocean/physics/vertical_mixing/config.py:33:    surface buoyancy flux (B_f) so that friction velocity and turbulent

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/vertical_mixing/integration.py | sed -n '1,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Factory for ocean vertical mixing physics."""
     2	
     3	from __future__ import annotations
     4	
     5	from typing import Callable
     6	
     7	from legoesm.grids.cubed_sphere import CubedSphereGrid
     8	from legoesm.ocean.eos import compute_ocean_rho as _compute_rho
     9	from legoesm.ocean.state import OceanState, OceanTendencies
    10	from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
    11	from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    12	
    13	
    14	def make_vertical_mixing_physics(
    15	    config: VerticalMixingConfig,
    16	) -> Callable:
    17	    """Create a vertical mixing physics function.
    18	
    19	    Parameters
    20	    ----------
    21	    config : VerticalMixingConfig
    22	
    23	    Returns
    24	    -------
    25	    Callable : physics_fn(state, grid, z_coord) -> OceanTendencies
    26	    """
    27	    scheme = config.scheme
    28	
    29	    if scheme == "none":
    30	        return _make_none()
    31	    elif scheme == "constant":
    32	        return _make_constant(config)
    33	    elif scheme == "richardson":
    34	        return _make_richardson(config)
    35	    elif scheme == "kpp":
    36	        return _make_kpp(config)
    37	    else:
    38	        raise ValueError(f"Unknown vertical mixing scheme: {scheme!r}")
    39	
    40	
    41	def _make_none() -> Callable:
    42	    def physics_fn(state: OceanState, grid: CubedSphereGrid,
    43	                   z_coord: OceanZStarCoordinate,
    44	                   surface_forcing=None) -> OceanTendencies:
    45	        return _zero_tendencies(state)
    46	    return physics_fn
    47	
    48	
    49	def _make_constant(config: VerticalMixingConfig) -> Callable:
    50	    from legoesm.ocean.physics.vertical_mixing.constant import constant_vertical_mixing
    51	    cfg = config.constant
    52	
    53	    def physics_fn(state: OceanState, grid: CubedSphereGrid,
    54	                   z_coord: OceanZStarCoordinate,
    55	                   surface_forcing=None) -> OceanTendencies:
    56	        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    57	        out = constant_vertical_mixing(
    58	            state.u.data, state.v.data, state.T.data, state.S.data,
    59	            z_coord, J, cfg,
    60	        )
    61	        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    62	    return physics_fn
    63	
    64	
    65	def _make_richardson(config: VerticalMixingConfig) -> Callable:
    66	    from legoesm.ocean.physics.vertical_mixing.richardson import richardson_vertical_mixing
    67	    cfg = config.richardson
    68	
    69	    def physics_fn(state: OceanState, grid: CubedSphereGrid,
    70	                   z_coord: OceanZStarCoordinate,
    71	                   surface_forcing=None) -> OceanTendencies:
    72	        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    73	        rho = _compute_rho(state, z_coord, J)
    74	        out = richardson_vertical_mixing(
    75	            state.u.data, state.v.data, state.T.data, state.S.data,
    76	            rho, z_coord, J, cfg,
    77	        )
    78	        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    79	    return physics_fn
    80	
    81	
    82	def _make_kpp(config: VerticalMixingConfig) -> Callable:
    83	    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    84	    cfg = config.kpp
    85	
    86	    def physics_fn(state: OceanState, grid: CubedSphereGrid,
    87	                   z_coord: OceanZStarCoordinate,
    88	                   surface_forcing=None) -> OceanTendencies:
    89	        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    90	        rho = _compute_rho(state, z_coord, J)
    91	        out = kpp_vertical_mixing(
    92	            state.u.data, state.v.data, state.T.data, state.S.data,
    93	            rho, state.eta.data, z_coord, J, cfg,
    94	        )
    95	        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    96	    return physics_fn
    97	
    98	
    99	
   100	def _zero_tendencies(state):
   101	    from legoesm.ocean.physics.combined import zero_ocean_tendencies
   102	    return zero_ocean_tendencies(state)
   103	
   104	
   105	def _wrap_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state):
   106	    from legoesm.ocean.physics.combined import wrap_ocean_tendencies
   107	    return wrap_ocean_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state)

exec
/bin/zsh -lc "rg -n \"kpp_vertical_mixing\\(\" -g'*.py' src/legoesm | cat" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/ocean/physics/vertical_mixing/kpp.py:139:def kpp_vertical_mixing(
src/legoesm/ocean/physics/vertical_mixing/integration.py:91:        out = kpp_vertical_mixing(

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/combined.py | sed -n '1,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Combined ocean physics orchestrator.
     2	
     3	Provides OceanPhysicsConfig and make_ocean_physics(), which create a
     4	single physics function combining vertical mixing, lateral mixing,
     5	surface forcing, bottom drag, and convection.
     6	"""
     7	
     8	from __future__ import annotations
     9	
    10	from typing import Callable, NamedTuple
    11	
    12	import jax.numpy as jnp
    13	
    14	from legoesm.core.field import Field
    15	from legoesm.grids.cubed_sphere import CubedSphereGrid
    16	from legoesm.ocean.state import OceanState, OceanSurfaceForcing, OceanTendencies
    17	from legoesm.ocean.vertical import OceanZStarCoordinate
    18	
    19	from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    20	from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    21	from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    22	from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    23	from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    24	from legoesm.ocean.physics.shortwave_penetration import (
    25	    ShortwavePenetrationConfig,
    26	    shortwave_penetration_tendency,
    27	)
    28	
    29	from legoesm.ocean.physics.vertical_mixing.integration import make_vertical_mixing_physics
    30	from legoesm.ocean.physics.lateral_mixing.integration import make_lateral_mixing_physics
    31	from legoesm.ocean.physics.surface_forcing.integration import make_surface_forcing_physics
    32	from legoesm.ocean.physics.bottom_drag.integration import make_bottom_drag_physics
    33	from legoesm.ocean.physics.convection.integration import make_convection_physics
    34	
    35	
    36	class OceanPhysicsConfig(NamedTuple):
    37	    """Unified ocean physics configuration.
    38	
    39	    Set the ``scheme`` field of any sub-config to ``"none"`` to disable
    40	    that module entirely.
    41	    """
    42	    vertical_mixing: VerticalMixingConfig = VerticalMixingConfig()
    43	    lateral_mixing: LateralMixingConfig = LateralMixingConfig()
    44	    surface_forcing: SurfaceForcingConfig = SurfaceForcingConfig()
    45	    bottom_drag: BottomDragConfig = BottomDragConfig()
    46	    convection: OceanConvectionConfig = OceanConvectionConfig()
    47	    shortwave_penetration: ShortwavePenetrationConfig | None = ShortwavePenetrationConfig()
    48	
    49	
    50	def make_ocean_physics(config: OceanPhysicsConfig) -> Callable:
    51	    """Create a combined ocean physics function.
    52	
    53	    The returned function calls each enabled physics module and sums
    54	    their tendencies.
    55	
    56	    Parameters
    57	    ----------
    58	    config : OceanPhysicsConfig
    59	
    60	    Returns
    61	    -------
    62	    Callable : physics_fn(state, grid, z_coord, surface_forcing=None) -> OceanTendencies
    63	    """
    64	    fns = []
    65	
    66	    if config.vertical_mixing.scheme != "none":
    67	        fns.append(make_vertical_mixing_physics(config.vertical_mixing))
    68	    if config.lateral_mixing.scheme != "none":
    69	        fns.append(make_lateral_mixing_physics(config.lateral_mixing))
    70	    if config.surface_forcing.scheme != "none":
    71	        fns.append(make_surface_forcing_physics(config.surface_forcing))
    72	    # Physics-level bottom drag is deprecated — use the dynamics-level
    73	    # ``bottom_drag_r`` field on the model config instead.  The dynamics
    74	    # path applies drag in both the baroclinic PE and the barotropic
    75	    # substeps, which is physically correct (MOM6 convention).
    76	    if config.bottom_drag.scheme != "none":
    77	        raise ValueError(
    78	            f"Physics-level bottom drag (scheme={config.bottom_drag.scheme!r}) "
    79	            "is deprecated. Use bottom_drag_r on your model config "
    80	            "(LatLonCGridOceanConfig or MPASOceanConfig) instead, which "
    81	            "applies drag in both the baroclinic PE and the barotropic "
    82	            "substeps (matching MOM6). Set BottomDragConfig(scheme='none') "
    83	            "in your OceanPhysicsConfig."
    84	        )
    85	    if config.convection.scheme != "none":
    86	        fns.append(make_convection_physics(config.convection))
    87	
    88	    sw_config = config.shortwave_penetration
    89	
    90	    def physics_fn(
    91	        state: OceanState,
    92	        grid: CubedSphereGrid,
    93	        z_coord: OceanZStarCoordinate,
    94	        surface_forcing: OceanSurfaceForcing | None = None,
    95	    ) -> OceanTendencies:
    96	        if not fns and sw_config is None:
    97	            return _zero_tendencies(state)
    98	
    99	        # Sum tendencies from all enabled sub-physics modules.
   100	        if fns:
   101	            first = fns[0](state, grid, z_coord, surface_forcing)
   102	            du_dt = first.du_dt.data
   103	            dv_dt = first.dv_dt.data
   104	            dT_dt = first.dT_dt.data
   105	            dS_dt = first.dS_dt.data
   106	            deta_dt = first.deta_dt.data
   107	
   108	            for fn in fns[1:]:
   109	                t = fn(state, grid, z_coord, surface_forcing)
   110	                du_dt = du_dt + t.du_dt.data
   111	                dv_dt = dv_dt + t.dv_dt.data
   112	                dT_dt = dT_dt + t.dT_dt.data
   113	                dS_dt = dS_dt + t.dS_dt.data
   114	                deta_dt = deta_dt + t.deta_dt.data
   115	        else:
   116	            z3 = jnp.zeros_like(state.u.data)
   117	            z2 = jnp.zeros_like(state.eta.data)
   118	            du_dt, dv_dt, dT_dt, dS_dt, deta_dt = z3, z3, z3, z3, z2
   119	
   120	        # Shortwave penetration: distribute SW heating through water column.
   121	        if (
   122	            sw_config is not None
   123	            and surface_forcing is not None
   124	            and surface_forcing.sw_down is not None
   125	        ):
   126	            from legoesm.ocean.vertical import compute_ocean_jacobian
   127	            J = compute_ocean_jacobian(
   128	                state.eta.data, state.H_bathy.data, z_coord,
   129	            )
   130	            sw_tend = shortwave_penetration_tendency(
   131	                surface_forcing.sw_down,
   132	                z_coord.dz_ref,
   133	                z_coord.z_half_ref,
   134	                J,
   135	                sw_config,
   136	            )
   137	            dT_dt = dT_dt + sw_tend
   138	
   139	        dims_3d = state.T.dims if hasattr(state.T, 'dims') else ("face", "x", "y", "level")
   140	        dims_2d = state.eta.dims if hasattr(state.eta, 'dims') else ("face", "x", "y")
   141	        return OceanTendencies(
   142	            du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
   143	            dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
   144	            dT_dt=Field(data=dT_dt, name="dT_dt", dims=dims_3d, units="degC/s"),
   145	            dS_dt=Field(data=dS_dt, name="dS_dt", dims=dims_3d, units="PSU/s"),
   146	            deta_dt=Field(data=deta_dt, name="deta_dt", dims=dims_2d, units="m/s"),
   147	            dH_bathy_dt=Field(
   148	                data=jnp.zeros_like(state.eta.data),
   149	                name="dH_bathy_dt", dims=dims_2d, units="m/s",
   150	            ),
   151	            dland_mask_dt=Field(
   152	                data=jnp.zeros_like(state.eta.data),
   153	                name="dland_mask_dt", dims=dims_2d, units="1/s",
   154	            ),
   155	        )
   156	
   157	    return physics_fn
   158	
   159	
   160	def _zero_tendencies(state: OceanState) -> OceanTendencies:
   161	    return zero_ocean_tendencies(state)
   162	
   163	
   164	# ---------------------------------------------------------------------------
   165	# Shared tendency helpers (used by all ocean physics integration modules)
   166	# ---------------------------------------------------------------------------
   167	
   168	def zero_ocean_tendencies(state: OceanState) -> OceanTendencies:
   169	    """Return zero tendencies matching *state* shapes."""
   170	    z3 = jnp.zeros_like(state.u.data)
   171	    z2 = jnp.zeros_like(state.eta.data)
   172	    dims_3d = state.u.dims if hasattr(state.u, 'dims') else ("face", "x", "y", "level")
   173	    dims_2d = state.eta.dims if hasattr(state.eta, 'dims') else ("face", "x", "y")
   174	    return OceanTendencies(
   175	        du_dt=Field(data=z3, name="du_dt", dims=dims_3d, units="m/s^2"),
   176	        dv_dt=Field(data=z3, name="dv_dt", dims=dims_3d, units="m/s^2"),
   177	        dT_dt=Field(data=z3, name="dT_dt", dims=dims_3d, units="degC/s"),
   178	        dS_dt=Field(data=z3, name="dS_dt", dims=dims_3d, units="PSU/s"),
   179	        deta_dt=Field(data=z2, name="deta_dt", dims=dims_2d, units="m/s"),
   180	        dH_bathy_dt=Field(data=z2, name="dH_bathy_dt", dims=dims_2d, units="m/s"),
   181	        dland_mask_dt=Field(data=z2, name="dland_mask_dt", dims=dims_2d, units="1/s"),
   182	    )
   183	
   184	
   185	def wrap_ocean_tendencies(
   186	    du_dt, dv_dt, dT_dt, dS_dt, state: OceanState,
   187	) -> OceanTendencies:
   188	    """Wrap raw tendency arrays into an ``OceanTendencies`` NamedTuple.
   189	
   190	    2-D fields (deta_dt, dH_bathy_dt, dland_mask_dt) are set to zero.
   191	    Any of du_dt … dS_dt may be ``None``, in which case the corresponding
   192	    tendency is set to zero.
   193	    """
   194	    z3 = jnp.zeros_like(state.u.data)
   195	    z2 = jnp.zeros_like(state.eta.data)
   196	    dims_3d = state.u.dims if hasattr(state.u, 'dims') else ("face", "x", "y", "level")
   197	    dims_2d = state.eta.dims if hasattr(state.eta, 'dims') else ("face", "x", "y")
   198	    return OceanTendencies(
   199	        du_dt=Field(data=du_dt if du_dt is not None else z3, name="du_dt", dims=dims_3d, units="m/s^2"),
   200	        dv_dt=Field(data=dv_dt if dv_dt is not None else z3, name="dv_dt", dims=dims_3d, units="m/s^2"),
   201	        dT_dt=Field(data=dT_dt if dT_dt is not None else z3, name="dT_dt", dims=dims_3d, units="degC/s"),
   202	        dS_dt=Field(data=dS_dt if dS_dt is not None else z3, name="dS_dt", dims=dims_3d, units="PSU/s"),
   203	        deta_dt=Field(data=z2, name="deta_dt", dims=dims_2d, units="m/s"),
   204	        dH_bathy_dt=Field(data=z2, name="dH_bathy_dt", dims=dims_2d, units="m/s"),
   205	        dland_mask_dt=Field(data=z2, name="dland_mask_dt", dims=dims_2d, units="1/s"),
   206	    )

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/vertical_mixing/implicit_solver.py | sed -n '1,240p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Backward-Euler implicit vertical diffusion for the ocean.
     2	
     3	Solves the 1-D diffusion equation per column
     4	
     5	    ∂φ/∂t = ∂/∂z [K · ∂φ/∂z]
     6	
     7	for each grid column using a backward-Euler discretisation that is
     8	unconditionally stable.  This is the key ingredient for issue #204:
     9	without it, the ocean dycore must keep first-order-upwind vertical
    10	momentum advection (whose numerical viscosity ~|w|·dz/2 silently damps
    11	baroclinic shear); with it, the physical ``A_v`` / ``K_v`` and any
    12	Richardson-number or KPP-based enhancement can do that job directly
    13	and the resolved advection can be upgraded to higher-order schemes.
    14	
    15	Discrete form (no-flux top + bottom):
    16	
    17	    ρ · ∂φ/∂t = ∂/∂z ( K ∂φ/∂z )       →     backward-Euler
    18	    (1 + α_k + β_k) φ^{n+1}_k
    19	        − α_k φ^{n+1}_{k-1}
    20	        − β_k φ^{n+1}_{k+1}
    21	      = φ^{n}_k
    22	with
    23	    α_k = dt · K_{k-1/2} / ( dz_k · dz_half_{k-1/2} )
    24	    β_k = dt · K_{k+1/2} / ( dz_k · dz_half_{k+1/2} )
    25	
    26	``K`` lives on interfaces (``nlev-1`` values per column), ``dz`` on
    27	layer centres (``nlev``), ``dz_half`` on interfaces (``nlev-1``).
    28	No-flux boundaries are enforced by setting ``K_{-1/2} = K_{N-1/2} = 0``
    29	(implicit via α_0 = 0 and β_{N-1} = 0).
    30	
    31	References
    32	----------
    33	- Thomas, L.H. (1949) — the Thomas algorithm is already implemented in
    34	  :mod:`legoesm.timestepping.tridiagonal`; this module just builds the
    35	  per-column system.
    36	- MOM6 Technical Manual §7 (implicit vertical viscosity).
    37	"""
    38	
    39	from __future__ import annotations
    40	
    41	import jax
    42	import jax.numpy as jnp
    43	
    44	from legoesm.timestepping.tridiagonal import thomas_solve
    45	
    46	_EPS = float(jnp.finfo(jnp.float32).eps)  # ~1.19e-7
    47	
    48	
    49	def implicit_vertical_diffusion_ocean(
    50	    field: jax.Array,
    51	    K: jax.Array | float,
    52	    dz: jax.Array,
    53	    dz_half: jax.Array,
    54	    dt: float,
    55	) -> jax.Array:
    56	    """Backward-Euler implicit vertical diffusion for a column field.
    57	
    58	    Applies one implicit step of
    59	
    60	        (1 - dt · ∂_z K ∂_z) φ^{n+1} = φ^{n}
    61	
    62	    per column with *zero-flux* boundary conditions at the top and the
    63	    bottom (the ocean's natural choice for momentum and tracers in the
    64	    absence of a prescribed surface flux).  Non-zero surface / bottom
    65	    fluxes can be applied *externally* before or after this call — the
    66	    helper is deliberately boundary-condition-simple so callers don't
    67	    have to thread flux arrays through when they don't need them.
    68	
    69	    Parameters
    70	    ----------
    71	    field : jax.Array, shape ``(..., nlev)``
    72	        Field to diffuse.  The vertical axis is the last axis; any
    73	        number of leading horizontal axes is allowed.  Examples:
    74	        ``(n_lat, n_lon, nlev)`` for lat-lon, ``(nCells, nlev)`` for
    75	        MPAS, ``(n_lat, n_lon+1, nlev)`` for u on a C-grid.
    76	    K : jax.Array or float, shape ``(..., nlev-1)``
    77	        Vertical viscosity / diffusivity at interior interfaces.
    78	        Must be ≥ 0.  A scalar is broadcast to every interface and
    79	        every column.
    80	    dz : jax.Array, shape ``(..., nlev)`` or ``(nlev,)``
    81	        Layer thickness at full levels.  Must match ``field`` along
    82	        the last axis (a 1-D ``dz`` broadcasts across all columns).
    83	    dz_half : jax.Array, shape ``(..., nlev-1)`` or ``(nlev-1,)``
    84	        Distance between adjacent full-level centres
    85	        (``dz_half_k = 0.5 (dz_k + dz_{k+1})`` is the standard choice).
    86	    dt : float
    87	        Time step [s].  Must be positive.
    88	
    89	    Returns
    90	    -------
    91	    jax.Array
    92	        Updated field with the same shape as ``field``.
    93	    """
    94	    # Only enforce the positivity check when ``dt`` is a concrete Python
    95	    # scalar — under ``jax.jit`` it may be a traced argument, and a
    96	    # Python-level ``if`` would raise ``TracerBoolConversionError``.
    97	    if not isinstance(dt, jax.core.Tracer):
    98	        if dt <= 0.0:
    99	            raise ValueError(f"dt must be > 0, got {dt!r}")
   100	
   101	    nlev = field.shape[-1]
   102	    if nlev < 2:
   103	        # One-level columns have no vertical gradient ⇒ no-op.
   104	        return field
   105	
   106	    # --- Promote K, dz, dz_half to match the field's leading shape ---
   107	    K_arr = jnp.asarray(K)
   108	    if K_arr.ndim == 0:
   109	        K_arr = jnp.broadcast_to(K_arr, field.shape[:-1] + (nlev - 1,))
   110	    elif K_arr.shape[-1] != nlev - 1:
   111	        raise ValueError(
   112	            f"K last dim {K_arr.shape[-1]} must equal nlev-1 = {nlev - 1}")
   113	
   114	    dz_arr = jnp.asarray(dz)
   115	    if dz_arr.ndim == 1:
   116	        dz_arr = jnp.broadcast_to(dz_arr, field.shape)
   117	    elif dz_arr.shape[-1] != nlev:
   118	        raise ValueError(
   119	            f"dz last dim {dz_arr.shape[-1]} must equal nlev = {nlev}")
   120	
   121	    dzh_arr = jnp.asarray(dz_half)
   122	    if dzh_arr.ndim == 1:
   123	        dzh_arr = jnp.broadcast_to(dzh_arr, field.shape[:-1] + (nlev - 1,))
   124	    elif dzh_arr.shape[-1] != nlev - 1:
   125	        raise ValueError(
   126	            f"dz_half last dim {dzh_arr.shape[-1]} must equal nlev-1 = "
   127	            f"{nlev - 1}")
   128	
   129	    # --- Build α and β at every cell (last axis = level) ---
   130	    # α_k uses the (k-1/2) interface, β_k the (k+1/2) interface.
   131	    # We pad K with an extra zero on each side so indexing is uniform;
   132	    # the zeros naturally encode the no-flux BCs.
   133	    K_safe = jnp.maximum(K_arr, 0.0)
   134	    dzh_safe = jnp.maximum(dzh_arr, _EPS)
   135	
   136	    # K / dz_half at interfaces (nlev-1)
   137	    flux_coeff = K_safe / dzh_safe                   # (..., nlev-1)
   138	
   139	    # Pad top and bottom with zero (no-flux).  Two Pad HLO ops replace
   140	    # alloc-zeros + two concatenate-of-two.
   141	    pad_axes = ((0, 0),) * (flux_coeff.ndim - 1)
   142	    flux_top = jnp.pad(flux_coeff, (*pad_axes, (1, 0)))  # (..., nlev)
   143	    flux_bot = jnp.pad(flux_coeff, (*pad_axes, (0, 1)))  # (..., nlev)
   144	
   145	    inv_dz = 1.0 / jnp.maximum(dz_arr, _EPS)          # (..., nlev)
   146	    alpha = dt * flux_top * inv_dz                    # (..., nlev)
   147	    beta = dt * flux_bot * inv_dz                     # (..., nlev)
   148	
   149	    # Tridiagonal coefficients:
   150	    #   a_k = -α_k   (sub-diagonal, a_0 = 0)
   151	    #   b_k = 1 + α_k + β_k
   152	    #   c_k = -β_k   (super-diagonal, c_{N-1} = 0)
   153	    #   d_k = φ^n_k
   154	    a = -alpha
   155	    b = 1.0 + alpha + beta
   156	    c = -beta
   157	    d = field
   158	
   159	    return thomas_solve(a, b, c, d)
   160	
   161	
   162	# ---------------------------------------------------------------------------
   163	# Convenience: build dz_half from dz with the standard midpoint rule.
   164	# ---------------------------------------------------------------------------
   165	
   166	
   167	def build_dz_half(dz: jax.Array) -> jax.Array:
   168	    """Midpoint distance between adjacent full-level centres.
   169	
   170	    ``dz_half_k = 0.5 · (dz_k + dz_{k+1})`` with shape ``(..., nlev-1)``.
   171	    Provided as a helper because most callers don't carry ``dz_half``
   172	    separately from ``dz_ref`` but do need it to build the tridiagonal
   173	    system.
   174	    """
   175	    return 0.5 * (dz[..., :-1] + dz[..., 1:])

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/convection/plume.py | sed -n '1,180p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Entraining mass-flux convective plume parameterization."""
     2	
     3	from __future__ import annotations
     4	
     5	import jax
     6	import jax.numpy as jnp
     7	
     8	from legoesm.ocean.eos import wright_eos
     9	from legoesm.ocean.physics.convection.config import PlumeConfig
    10	from legoesm.ocean.physics.convection.output import OceanConvectionOutput
    11	from legoesm.ocean.vertical import OceanZStarCoordinate
    12	
    13	
    14	def plume_convection(
    15	    T: jnp.ndarray,
    16	    S: jnp.ndarray,
    17	    rho: jnp.ndarray,
    18	    p_hydro: jnp.ndarray,
    19	    z_coord: OceanZStarCoordinate,
    20	    jacobian: jnp.ndarray,
    21	    cfg: PlumeConfig,
    22	) -> OceanConvectionOutput:
    23	    """Apply entraining mass-flux plume convection.
    24	
    25	    Parameters
    26	    ----------
    27	    T, S : array (6, n, n, nlev)
    28	    rho : array (6, n, n, nlev)
    29	    p_hydro : array (6, n, n, nlev)
    30	    z_coord : OceanZStarCoordinate
    31	    jacobian : array (6, n, n)
    32	    cfg : PlumeConfig
    33	
    34	    Returns
    35	    -------
    36	    OceanConvectionOutput
    37	    """
    38	    nlev = T.shape[-1]
    39	    shape_3d = T.shape
    40	    dtype = T.dtype
    41	    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    42	
    43	    # Detect unstable surface: rho(k=0) > rho(k=1)
    44	    surface_unstable = rho[..., 0] > rho[..., 1]  # (6, n, n)
    45	
    46	    # Initialize plume properties at surface
    47	    T_plume_init = T[..., 0] + cfg.T_excess
    48	    S_plume_init = S[..., 0]
    49	
    50	    # Descend plume using scan over levels (starting from level 1)
    51	    def scan_fn(carry, k):
    52	        T_plume, S_plume, active = carry
    53	        dz_k = dz_actual[..., k]
    54	
    55	        # Entrain environment
    56	        entrain = cfg.epsilon * dz_k
    57	        T_plume = (1.0 - entrain) * T_plume + entrain * T[..., k]
    58	        S_plume = (1.0 - entrain) * S_plume + entrain * S[..., k]
    59	
    60	        # Buoyancy check
    61	        rho_plume = wright_eos(T_plume, S_plume, p_hydro[..., k])
    62	        delta_rho = rho_plume - rho[..., k]
    63	
    64	        # Plume is active where it's denser than environment (sinking):
    65	        # delta_rho > 0 means rho_plume > rho_env → plume sinks → stay active
    66	        active = active * jax.nn.sigmoid(delta_rho * 1e4)
    67	
    68	        # Detrainment tendency at this level
    69	        dT_k = cfg.alpha_plume * cfg.epsilon * (T_plume - T[..., k]) * active
    70	        dS_k = cfg.alpha_plume * cfg.epsilon * (S_plume - S[..., k]) * active
    71	
    72	        return (T_plume, S_plume, active), (dT_k, dS_k, active)
    73	
    74	    init_active = surface_unstable.astype(dtype)
    75	    (_, _, _), (dT_levels, dS_levels, active_levels) = jax.lax.scan(
    76	        scan_fn,
    77	        (T_plume_init, S_plume_init, init_active),
    78	        jnp.arange(1, nlev),
    79	    )
    80	
    81	    # dT_levels shape: (nlev-1, 6, n, n) — move level axis to last,
    82	    # then ``jnp.pad`` along the trailing axis instead of
    83	    # ``zeros + .at[..., 1:].set(...)`` which materialises a fresh
    84	    # zero buffer + scatter.  Single Pad HLO op each.
    85	    dT_levels_t = jnp.moveaxis(dT_levels, 0, -1)  # (6, n, n, nlev-1)
    86	    dS_levels_t = jnp.moveaxis(dS_levels, 0, -1)
    87	    pad_axes = ((0, 0),) * (dT_levels_t.ndim - 1)
    88	    dT_dt = jnp.pad(dT_levels_t, (*pad_axes, (1, 0)))
    89	    dS_dt = jnp.pad(dS_levels_t, (*pad_axes, (1, 0)))
    90	
    91	    # Convection flag at interfaces (average of adjacent levels' activity)
    92	    active_t = jnp.moveaxis(active_levels, 0, -1)  # (6, n, n, nlev-1)
    93	    flag = active_t
    94	
    95	    return OceanConvectionOutput(
    96	        dT_dt=dT_dt,
    97	        dS_dt=dS_dt,
    98	        convection_flag=flag,
    99	    )

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/convection/config.py | sed -n '1,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Configuration for ocean convection schemes."""
     2	
     3	from __future__ import annotations
     4	
     5	from typing import NamedTuple
     6	
     7	
     8	class EnhancedDiffusionConfig(NamedTuple):
     9	    """Enhanced diffusion where N^2 < 0."""
    10	    K_conv: float = 1.0        # Convective diffusivity [m^2/s]
    11	    K_bg: float = 1e-5         # Background diffusivity [m^2/s]
    12	    smooth_transition: bool = True
    13	    sigmoid_sharpness: float = 1e6
    14	
    15	
    16	class PlumeConfig(NamedTuple):
    17	    """Entraining mass-flux convective plume."""
    18	    epsilon: float = 1e-3       # Entrainment rate [1/m]
    19	    alpha_plume: float = 0.1    # Detrainment tendency scaling
    20	    w_plume_min: float = 0.01   # Minimum plume velocity [m/s]
    21	    T_excess: float = 0.05      # Initial plume temperature excess [K]
    22	
    23	
    24	class OceanConvectionConfig(NamedTuple):
    25	    """Top-level ocean convection configuration."""
    26	    scheme: str = "none"  # "enhanced_diffusion", "plume", "none"
    27	    enhanced_diffusion: EnhancedDiffusionConfig = EnhancedDiffusionConfig()
    28	    plume: PlumeConfig = PlumeConfig()

exec
/bin/zsh -lc "rg -n \"plume\" tests/ocean -g'*.py'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
tests/ocean/unit/test_mpas_physics.py:215:            convection=OceanConvectionConfig(scheme="plume"),
tests/ocean/unit/test_mpas_physics.py:217:        with pytest.raises(NotImplementedError, match="plume"):

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/convection/integration.py | sed -n '1,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Factory for ocean convection physics."""
     2	
     3	from __future__ import annotations
     4	
     5	from typing import Callable
     6	
     7	import jax.numpy as jnp
     8	
     9	from legoesm.core.field import Field
    10	from legoesm.grids.cubed_sphere import CubedSphereGrid
    11	from legoesm.ocean.eos import (
    12	    compute_ocean_rho as _compute_rho,
    13	    compute_ocean_rho_and_pressure as _compute_rho_and_pressure,
    14	)
    15	from legoesm.ocean.state import OceanState, OceanTendencies
    16	from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian
    17	from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    18	
    19	
    20	def make_convection_physics(
    21	    config: OceanConvectionConfig,
    22	) -> Callable:
    23	    """Create an ocean convection physics function.
    24	
    25	    Parameters
    26	    ----------
    27	    config : OceanConvectionConfig
    28	
    29	    Returns
    30	    -------
    31	    Callable : physics_fn(state, grid, z_coord) -> OceanTendencies
    32	    """
    33	    scheme = config.scheme
    34	
    35	    if scheme == "none":
    36	        return _make_none()
    37	    elif scheme == "enhanced_diffusion":
    38	        return _make_enhanced_diffusion(config)
    39	    elif scheme == "plume":
    40	        return _make_plume(config)
    41	    else:
    42	        raise ValueError(f"Unknown ocean convection scheme: {scheme!r}")
    43	
    44	
    45	def _make_none() -> Callable:
    46	    def physics_fn(state: OceanState, grid: CubedSphereGrid,
    47	                   z_coord: OceanZStarCoordinate,
    48	                   surface_forcing=None) -> OceanTendencies:
    49	        return _zero_tendencies(state)
    50	    return physics_fn
    51	
    52	
    53	def _make_enhanced_diffusion(config: OceanConvectionConfig) -> Callable:
    54	    from legoesm.ocean.physics.convection.enhanced_diffusion import enhanced_diffusion_convection
    55	    cfg = config.enhanced_diffusion
    56	
    57	    def physics_fn(state: OceanState, grid: CubedSphereGrid,
    58	                   z_coord: OceanZStarCoordinate,
    59	                   surface_forcing=None) -> OceanTendencies:
    60	        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    61	        rho = _compute_rho(state, z_coord, J)
    62	        out = enhanced_diffusion_convection(
    63	            state.T.data, state.S.data, rho, z_coord, J, cfg,
    64	        )
    65	        z3 = jnp.zeros_like(state.u.data)
    66	        return _wrap_tendencies(z3, z3, out.dT_dt, out.dS_dt, state)
    67	    return physics_fn
    68	
    69	
    70	def _make_plume(config: OceanConvectionConfig) -> Callable:
    71	    from legoesm.ocean.physics.convection.plume import plume_convection
    72	    cfg = config.plume
    73	
    74	    def physics_fn(state: OceanState, grid: CubedSphereGrid,
    75	                   z_coord: OceanZStarCoordinate,
    76	                   surface_forcing=None) -> OceanTendencies:
    77	        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    78	        rho, p_hydro = _compute_rho_and_pressure(state, z_coord, J)
    79	        out = plume_convection(
    80	            state.T.data, state.S.data, rho, p_hydro, z_coord, J, cfg,
    81	        )
    82	        z3 = jnp.zeros_like(state.u.data)
    83	        return _wrap_tendencies(z3, z3, out.dT_dt, out.dS_dt, state)
    84	    return physics_fn
    85	
    86	
    87	
    88	def _zero_tendencies(state):
    89	    from legoesm.ocean.physics.combined import zero_ocean_tendencies
    90	    return zero_ocean_tendencies(state)
    91	
    92	
    93	def _wrap_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state):
    94	    from legoesm.ocean.physics.combined import wrap_ocean_tendencies
    95	    return wrap_ocean_tendencies(du_dt, dv_dt, dT_dt, dS_dt, state)

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py | sed -n '1,240p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Grid-agnostic helpers shared by cubed-sphere and lat-lon GM/Redi.
     2	
     3	Functions in this module operate on ``(..., nlev)`` arrays and make no
     4	reference to a specific grid type.  Both ``gm_redi.py`` (cubed-sphere)
     5	and ``gm_redi_latlon_cgrid.py`` (lat-lon C-grid) import from here.
     6	"""
     7	
     8	from __future__ import annotations
     9	
    10	import jax.numpy as jnp
    11	
    12	from legoesm import constants
    13	from legoesm.ocean.eos import compute_buoyancy_frequency, rho_0 as _RHO_0_DEFAULT
    14	from legoesm.ocean.physics.lateral_mixing.config import VisbeckConfig
    15	from legoesm.ocean.vertical import OceanZStarCoordinate
    16	
    17	_EPS = float(jnp.finfo(jnp.float32).eps)  # ~1.19e-7
    18	
    19	
    20	# ---------------------------------------------------------------------------
    21	# DM95 slope tapering
    22	# ---------------------------------------------------------------------------
    23	
    24	def dm95_taper(
    25	    S_x: jnp.ndarray,
    26	    S_y: jnp.ndarray,
    27	    S_max: float,
    28	    eps: float = _EPS,
    29	) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    30	    """Apply Danabasoglu & McWilliams (1995) smooth slope tapering.
    31	
    32	    Returns tapered ``(S_x, S_y, taper)`` where *taper* is a smooth
    33	    factor in [0, 1] computed as::
    34	
    35	        taper = 0.5 * (1 + tanh((S_max - |S|) / (0.1 * S_max)))
    36	
    37	    Parameters
    38	    ----------
    39	    S_x, S_y : array (..., nlev-1)
    40	        Raw (clipped but un-tapered) isopycnal slopes at interfaces.
    41	    S_max : float
    42	        Maximum slope for tapering.
    43	    eps : float
    44	        Small constant for sqrt regularisation.
    45	
    46	    Returns
    47	    -------
    48	    S_x_tapered, S_y_tapered, taper : same shapes as inputs.
    49	    """
    50	    S_mag = jnp.sqrt(S_x ** 2 + S_y ** 2 + eps)
    51	    taper = 0.5 * (1.0 + jnp.tanh(
    52	        (S_max - S_mag) / (0.1 * S_max + eps)
    53	    ))
    54	    return S_x * taper, S_y * taper, taper
    55	
    56	
    57	def dm95_taper_scalar(
    58	    S: jnp.ndarray,
    59	    S_max: float,
    60	    eps: float = _EPS,
    61	) -> tuple[jnp.ndarray, jnp.ndarray]:
    62	    """Single-component variant of :func:`dm95_taper`.
    63	
    64	    Used by grids that carry a scalar slope along each face's own normal
    65	    (e.g. MPAS/Voronoi edges).  Identical functional form, with ``|S|``
    66	    replaced by ``|S_n|``.
    67	
    68	    Returns
    69	    -------
    70	    S_tapered, taper : same shape as ``S``.
    71	    """
    72	    taper = 0.5 * (1.0 + jnp.tanh(
    73	        (S_max - jnp.abs(S)) / (0.1 * S_max + eps)
    74	    ))
    75	    return S * taper, taper
    76	
    77	
    78	# ---------------------------------------------------------------------------
    79	# Vertical flux divergence with zero-flux BCs
    80	# ---------------------------------------------------------------------------
    81	
    82	def vertical_flux_divergence(
    83	    F_z: jnp.ndarray,
    84	    dz_actual: jnp.ndarray,
    85	    eps: float = _EPS,
    86	) -> jnp.ndarray:
    87	    """Compute vertical flux divergence at full levels.
    88	
    89	    ``dq/dt[k] = (F_z[k-1/2] - F_z[k+1/2]) / dz[k]``
    90	
    91	    with F_z = 0 at the surface and bottom boundaries (zero-flux BCs).
    92	
    93	    Parameters
    94	    ----------
    95	    F_z : array (..., nlev-1)
    96	        Vertical flux at interior interfaces.
    97	    dz_actual : array (..., nlev)
    98	        Layer thicknesses at full levels.
    99	
   100	    Returns
   101	    -------
   102	    tendency : array (..., nlev)
   103	    """
   104	    # Single Pad HLO op (replaces alloc-zeros + concatenate of three).
   105	    pad_axes = ((0, 0),) * (F_z.ndim - 1)
   106	    F_z_ext = jnp.pad(F_z, (*pad_axes, (1, 1)))
   107	    return (F_z_ext[..., :-1] - F_z_ext[..., 1:]) / jnp.maximum(dz_actual, eps)
   108	
   109	
   110	# ---------------------------------------------------------------------------
   111	# Visbeck (1997) adaptive GM coefficient
   112	# ---------------------------------------------------------------------------
   113	
   114	def compute_visbeck_kappa_gm(
   115	    rho: jnp.ndarray,
   116	    S_x: jnp.ndarray,
   117	    S_y: jnp.ndarray,
   118	    z_coord: OceanZStarCoordinate,
   119	    jacobian: jnp.ndarray,
   120	    f_coriolis: jnp.ndarray,
   121	    cfg: VisbeckConfig,
   122	    rho_ref: float = _RHO_0_DEFAULT,
   123	) -> jnp.ndarray:
   124	    """Visbeck (1997) adaptive GM coefficient.
   125	
   126	    ``kappa(x, y) = alpha * L^2 * <N * |S|>_z`` with the depth-average
   127	    weighted by the local interface thickness, optionally using the
   128	    local first-baroclinic Rossby radius as the mixing length.
   129	
   130	    Parameters
   131	    ----------
   132	    rho : (..., nlev) in-situ density.
   133	    S_x, S_y : (..., nlev-1) tapered isopycnal slopes at interfaces.
   134	    z_coord : OceanZStarCoordinate.
   135	    jacobian : (...,) z* Jacobian at cell centres.
   136	    f_coriolis : (...,) Coriolis parameter.
   137	    cfg : VisbeckConfig.
   138	    rho_ref : float
   139	        Boussinesq reference density [kg/m^3].
   140	
   141	    Returns
   142	    -------
   143	    kappa : (...,) horizontally-varying kappa_GM [m^2/s], clamped to
   144	        the configured bounds.
   145	    """
   146	    eps = _EPS
   147	    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
   148	    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
   149	
   150	    # Local growth rate sigma_Eady ~ N * |S| at each interior interface.
   151	    N2 = compute_buoyancy_frequency(
   152	        rho, z_coord.dz_ref, jacobian, rho_ref=rho_ref, g=constants.g,
   153	    )
   154	    N = jnp.sqrt(jnp.maximum(N2, 0.0))
   155	    # Regularise sqrt at zero slope — 1e-30 avoids spurious |S| ~ 3e-4
   156	    # that the float32 eps (~1.19e-7) would produce.
   157	    S_mag = jnp.sqrt(S_x ** 2 + S_y ** 2 + 1e-30)
   158	    sigma = N * S_mag
   159	
   160	    # Depth-weighted average of sigma_Eady.
   161	    w_total = jnp.sum(dz_half, axis=-1)
   162	    sigma_bar = jnp.sum(sigma * dz_half, axis=-1) / jnp.maximum(w_total, eps)
   163	
   164	    # Mixing length L.
   165	    if cfg.use_rossby_radius:
   166	        N_bar = jnp.sum(N * dz_half, axis=-1) / jnp.maximum(w_total, eps)
   167	        H_col = jnp.sum(dz_actual, axis=-1)
   168	        f_safe = jnp.maximum(jnp.abs(f_coriolis), cfg.f_min)
   169	        L = jnp.clip(N_bar * H_col / f_safe, cfg.L_min, cfg.L_max)
   170	    else:
   171	        L = jnp.full_like(sigma_bar, cfg.L_fixed)
   172	
   173	    kappa = cfg.alpha * L ** 2 * sigma_bar
   174	    return jnp.clip(kappa, cfg.kappa_min, cfg.kappa_max)

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/biogeochemistry/npzd.py | sed -n '1,240p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""NPZD ecosystem model coupled to the inorganic carbon cycle.
     2	
     3	A Nutrient-Phytoplankton-Zooplankton-Detritus model following the
     4	Fasham et al. (1990) / Oschlies & Garçon (1999) framework with
     5	coupling to DIC, alkalinity, and air-sea CO2 exchange.
     6	
     7	Processes:
     8	1. Light-limited, nutrient-limited phytoplankton growth
     9	2. Zooplankton grazing (Holling type II)
    10	3. Phytoplankton and zooplankton mortality
    11	4. Detritus sinking and remineralization
    12	5. Stoichiometric coupling to DIC/ALK via Redfield ratios
    13	6. CaCO3 production/dissolution (rain ratio parameterization)
    14	
    15	All functions are JAX-compatible (differentiable, JIT-friendly).
    16	
    17	References
    18	----------
    19	- Fasham, M. J. R., et al. (1990). A nitrogen-based model of plankton
    20	  dynamics in the oceanic mixed layer. J. Mar. Res., 48, 591-639.
    21	- Oschlies, A. & Garçon, V. (1999). An eddy-permitting coupled physical-
    22	  biological model of the North Atlantic. Global Biogeochem. Cycles, 13.
    23	"""
    24	
    25	from __future__ import annotations
    26	
    27	import jax.numpy as jnp
    28	
    29	from legoesm.ocean.biogeochemistry.config import BiogeoConfig
    30	
    31	
    32	def par_profile(
    33	    PAR_surf: jnp.ndarray,
    34	    z_full_ref: jnp.ndarray,
    35	    Phyto: jnp.ndarray,
    36	    dz_ref: jnp.ndarray,
    37	    cfg: BiogeoConfig,
    38	) -> jnp.ndarray:
    39	    """Compute PAR at each depth level with self-shading.
    40	
    41	    Beer-Lambert law with water + chlorophyll attenuation:
    42	    PAR(z) = PAR_surf * exp(-integral_0^z (k_w + k_chl * P) dz')
    43	
    44	    Parameters
    45	    ----------
    46	    PAR_surf : array
    47	        Surface PAR [W/m^2], shape (...).
    48	    z_full_ref : array
    49	        Reference depths [m], shape (nlev,), negative.
    50	    Phyto : array
    51	        Phytoplankton [mol N/m^3], shape (..., nlev).
    52	    dz_ref : array
    53	        Layer thicknesses [m], shape (nlev,).
    54	    cfg : BiogeoConfig
    55	
    56	    Returns
    57	    -------
    58	    PAR : array
    59	        PAR at each level [W/m^2], shape (..., nlev).
    60	    """
    61	    nlev = z_full_ref.shape[0]
    62	
    63	    # Total attenuation per layer: (k_w + k_chl * P) * dz
    64	    atten = (cfg.k_w_atten + cfg.k_chl_atten * jnp.clip(Phyto, 0.0, None)) * dz_ref
    65	
    66	    # Cumulative attenuation from surface (k=0 is surface)
    67	    # For level k, PAR has been attenuated by layers 0..k-1 plus half of layer k
    68	    cum_atten_interface = jnp.cumsum(atten, axis=-1)
    69	    # Shift: interface attenuation at top of layer k = cumsum through k-1.
    70	    # ``jnp.pad`` is one Pad HLO op; the previous form allocated a fresh
    71	    # ``(..., 1)`` zero buffer + concatenate.
    72	    _pad_axes = ((0, 0),) * (cum_atten_interface.ndim - 1)
    73	    cum_atten_top = jnp.pad(cum_atten_interface[..., :-1], (*_pad_axes, (1, 0)))
    74	    # Mid-level attenuation = top + half this layer
    75	    cum_atten_mid = cum_atten_top + 0.5 * atten
    76	
    77	    PAR = PAR_surf[..., jnp.newaxis] * jnp.exp(-cum_atten_mid)
    78	    return PAR
    79	
    80	
    81	def npzd_source_sink(
    82	    NO3: jnp.ndarray,
    83	    Phyto: jnp.ndarray,
    84	    Zoo: jnp.ndarray,
    85	    Det: jnp.ndarray,
    86	    DIC: jnp.ndarray,
    87	    ALK: jnp.ndarray,
    88	    T_degC: jnp.ndarray,
    89	    PAR: jnp.ndarray,
    90	    dz_ref: jnp.ndarray,
    91	    cfg: BiogeoConfig,
    92	) -> tuple[jnp.ndarray, ...]:
    93	    """Compute NPZD source/sink terms at every grid point and level.
    94	
    95	    Parameters
    96	    ----------
    97	    NO3, Phyto, Zoo, Det, DIC, ALK : array (..., nlev)
    98	        Biogeochemical tracer concentrations.
    99	    T_degC : array (..., nlev)
   100	        Temperature [degC].
   101	    PAR : array (..., nlev)
   102	        Photosynthetically available radiation [W/m^2].
   103	    dz_ref : array (nlev,)
   104	        Layer thicknesses [m].
   105	    cfg : BiogeoConfig
   106	
   107	    Returns
   108	    -------
   109	    dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt : arrays
   110	        Source/sink tendencies [units/s], same shapes as inputs.
   111	    """
   112	    day_to_s = 1.0 / 86400.0  # convert rates from 1/day to 1/s
   113	
   114	    # Clip tracers to non-negative (smooth via softplus at small values)
   115	    eps = 1.0e-12
   116	    P = jnp.clip(Phyto, eps, None)
   117	    Z = jnp.clip(Zoo, eps, None)
   118	    D = jnp.clip(Det, eps, None)
   119	    N = jnp.clip(NO3, eps, None)
   120	
   121	    # ---- 1. Phytoplankton growth ----
   122	    # Temperature dependence: Eppley (1972) Q10 = 1.066^T
   123	    T_factor = 1.066 ** jnp.clip(T_degC, -2.0, 40.0)
   124	
   125	    # Nutrient limitation (Michaelis-Menten)
   126	    f_N = N / (N + cfg.k_N)
   127	
   128	    # Light limitation (Webb et al. 1974 exponential form)
   129	    # alpha_P [1/(W/m^2)/day] and mu_max [1/day] are both per-day,
   130	    # so their ratio alpha_P/mu_max [1/(W/m^2)] is already consistent.
   131	    f_L = 1.0 - jnp.exp(-cfg.alpha_P * PAR / jnp.clip(cfg.mu_max, eps, None))
   132	
   133	    # Growth rate
   134	    mu = cfg.mu_max * day_to_s * T_factor * jnp.minimum(f_N, f_L)
   135	    growth = mu * P  # mol N/m^3/s
   136	
   137	    # ---- 2. Zooplankton grazing ----
   138	    grazing = cfg.g_max * day_to_s * P ** 2 / (P ** 2 + cfg.k_P ** 2) * Z
   139	
   140	    # ---- 3. Mortality ----
   141	    phyto_mort = cfg.m_P * day_to_s * P
   142	    zoo_mort = cfg.m_Z * day_to_s * Z ** 2  # quadratic
   143	
   144	    # ---- 4. Detritus remineralization ----
   145	    remin = cfg.remin_rate * day_to_s * D
   146	
   147	    # ---- 5. Detritus sinking (conservative interface-flux formulation) ----
   148	    # w_sink in m/day -> m/s
   149	    w_sink_s = cfg.w_sink * day_to_s  # m/s
   150	    # Interface flux (upwind): F[k] = w_sink * D[k-1] [mol N/m^2/s]
   151	    # F[0] = 0 (no flux into top), F[nlev] = w_sink * D[nlev-1] (export)
   152	    # Tendency: dD/dt[k] = (F[k] - F[k+1]) / dz[k]
   153	    flux_out = w_sink_s * D  # flux leaving each layer downward
   154	    # ``jnp.pad`` along trailing axis: single Pad HLO op vs
   155	    # alloc-zeros + concatenate.
   156	    _pad_axes_d = ((0, 0),) * (flux_out.ndim - 1)
   157	    flux_in = jnp.pad(flux_out[..., :-1], (*_pad_axes_d, (1, 0)))
   158	    sinking_tend = (flux_in - flux_out) / jnp.clip(dz_ref, 1.0, None)
   159	
   160	    # ---- Assemble tendencies ----
   161	    # Nutrients
   162	    dNO3_dt = -growth + remin + (1.0 - cfg.gamma_Z) * grazing
   163	
   164	    # Phytoplankton
   165	    dPhyto_dt = growth - grazing - phyto_mort
   166	
   167	    # Zooplankton
   168	    dZoo_dt = cfg.gamma_Z * grazing - zoo_mort
   169	
   170	    # Detritus
   171	    dDet_dt = phyto_mort + zoo_mort - remin + sinking_tend
   172	
   173	    # ---- Carbon coupling (Redfield) ----
   174	    # Organic carbon cycle: C:N = R_CN
   175	    # DIC decreases with primary production, increases with remineralization
   176	    net_production = growth - remin - (1.0 - cfg.gamma_Z) * grazing
   177	    dDIC_bio = -cfg.R_CN * net_production
   178	
   179	    # CaCO3 cycle: rain ratio * organic C export
   180	    # CaCO3 production removes DIC and 2*ALK (in surface/euphotic zone)
   181	    # CaCO3 dissolution adds back (in deep, parameterized as remin)
   182	    caco3_production = cfg.R_CaP * cfg.R_CN * growth
   183	    caco3_dissolution = cfg.R_CaP * cfg.R_CN * remin
   184	
   185	    dDIC_dt = dDIC_bio - caco3_production + caco3_dissolution
   186	    # Alkalinity: -1 per mol NO3 consumed (nitrification sign convention)
   187	    # + 2 per mol CaCO3 dissolved, -2 per mol CaCO3 precipitated
   188	    dALK_dt = (-growth + remin + (1.0 - cfg.gamma_Z) * grazing
   189	               - 2.0 * caco3_production + 2.0 * caco3_dissolution)
   190	
   191	    return dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt

exec
/bin/zsh -lc "nl -ba src/legoesm/core/operators_cdgrid.py | sed -n '880,1045p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   880	    q : jax.Array, shape (6, n, n) or (6, n, n, nlev)
   881	    u_c : jax.Array, shape (6, n+1, n[, nlev])
   882	    v_c : jax.Array, shape (6, n, n+1[, nlev])
   883	    cdgrid : CubedSphereCDGrid
   884	
   885	    Returns
   886	    -------
   887	    dq_dt : jax.Array, shape matching ``q``.
   888	
   889	    References
   890	    ----------
   891	    - Colella & Woodward (1984): PPM reconstruction.
   892	    - Lin (2004): FV3 transport.
   893	    - Zalesak (1979): Fully multidimensional FCT.
   894	    """
   895	    n = cdgrid.n
   896	    area = cdgrid.base.area   # (6, n, n)
   897	    dy = cdgrid.dy_edge_x     # (6, n+1, n)
   898	    dx = cdgrid.dx_edge_y     # (6, n, n+1)
   899	
   900	    # Halo-padded fields (halo=1 for upwind / min-max stencil, halo=2 for PPM).
   901	    # ``_pad_halo_auto*`` already dispatch on ``q.ndim`` so 4D inputs use
   902	    # ``pad_halo_4d`` (one MPI message for all levels).
   903	    q_pad_full = _pad_halo_auto(q, cdgrid)        # (6, n+2, n+2[, nlev])
   904	    q_pad_h2_full = _pad_halo_auto_h2(q, cdgrid)  # (6, n+4, n+4[, nlev])
   905	
   906	    # For 4D inputs we move ``nlev`` to the leading position so that the
   907	    # rest of the body's spatial slicing — written with ``[..., ...]``
   908	    # prefixes — operates on the same trailing ``(i, j)`` axes regardless
   909	    # of rank.  This also keeps ``_ppm_reconstruct_1d`` (axis -1) acting on
   910	    # the correct PPM axis (j with halo, n with halo stripped).
   911	    is_4d = q.ndim == 4
   912	    if is_4d:
   913	        q_t = jnp.moveaxis(q, -1, 0)
   914	        q_pad = jnp.moveaxis(q_pad_full, -1, 0)
   915	        q_pad_h2 = jnp.moveaxis(q_pad_h2_full, -1, 0)
   916	        u_c_t = jnp.moveaxis(u_c, -1, 0)
   917	        v_c_t = jnp.moveaxis(v_c, -1, 0)
   918	    else:
   919	        q_t = q
   920	        q_pad = q_pad_full
   921	        q_pad_h2 = q_pad_h2_full
   922	        u_c_t = u_c
   923	        v_c_t = v_c
   924	
   925	    # ----------------------------------------------------------------
   926	    # Step 1: First-order upwind fluxes (inherently monotone for CFL<1)
   927	    # ----------------------------------------------------------------
   928	    q_left_x = q_pad[..., :-1, 1:-1]
   929	    q_right_x = q_pad[..., 1:, 1:-1]
   930	    q_face_low_x = jnp.where(u_c_t > 0, q_left_x, q_right_x)
   931	
   932	    q_below_y = q_pad[..., 1:-1, :-1]
   933	    q_above_y = q_pad[..., 1:-1, 1:]
   934	    q_face_low_y = jnp.where(v_c_t > 0, q_below_y, q_above_y)
   935	
   936	    flux_low_x = q_face_low_x * u_c_t * dy
   937	    flux_low_y = q_face_low_y * v_c_t * dx
   938	
   939	    net_low_x = flux_low_x[..., 1:, :] - flux_low_x[..., :-1, :]
   940	    net_low_y = flux_low_y[..., 1:] - flux_low_y[..., :-1]
   941	    dq_low = -(net_low_x + net_low_y) / area
   942	
   943	    # ----------------------------------------------------------------
   944	    # Step 2: PPM face values with face-value clipping
   945	    # ----------------------------------------------------------------
   946	    # NOTE: ``q_pad_h2`` was assigned above (and moveaxis'd for 4D
   947	    # inputs).  Do *not* re-pad here — that would discard the leading
   948	    # ``nlev`` axis and break the 4D path with a shape mismatch when
   949	    # the clip step (line ~960) compares against ``q_left_x`` /
   950	    # ``q_right_x`` (which use the rotated ``q_pad``).
   951	
   952	    # X-direction PPM — strips are taken along the trailing ``(i, j)``
   953	    # axes regardless of rank.  Use negative axes so the helper acts on
   954	    # the correct PPM (i) axis whether ``q`` is 3D ``(6, ny, nx)`` or
   955	    # 4D moved to ``(nlev, 6, ny, nx)``.
   956	    q_x_strips = q_pad_h2[..., :, 2:-2]                 # (..., n+4, n)
   957	    q_L_x, q_R_x = _ppm_reconstruct_1d(q_x_strips, axis=-2)
   958	    q_R_left = q_R_x[..., 1:n+2, :]
   959	    q_L_right = q_L_x[..., 2:n+3, :]
   960	    q_face_hi_x = jnp.where(u_c_t > 0, q_R_left, q_L_right)
   961	
   962	    # Clip to local bounds of adjacent cells
   963	    q_face_min_x = jnp.minimum(q_left_x, q_right_x)
   964	    q_face_max_x = jnp.maximum(q_left_x, q_right_x)
   965	    q_face_hi_x = jnp.clip(q_face_hi_x, q_face_min_x, q_face_max_x)
   966	
   967	    # Y-direction PPM — strip shape (..., n, n+4) puts the halo-padded
   968	    # j-axis at axis=-1.
   969	    q_y_strips = q_pad_h2[..., 2:-2, :]
   970	    q_L_y, q_R_y = _ppm_reconstruct_1d(q_y_strips, axis=-1)
   971	    q_R_bottom = q_R_y[..., :, 1:n+2]
   972	    q_L_top = q_L_y[..., :, 2:n+3]
   973	    q_face_hi_y = jnp.where(v_c_t > 0, q_R_bottom, q_L_top)
   974	
   975	    # Clip to local bounds of adjacent cells
   976	    q_face_min_y = jnp.minimum(q_below_y, q_above_y)
   977	    q_face_max_y = jnp.maximum(q_below_y, q_above_y)
   978	    q_face_hi_y = jnp.clip(q_face_hi_y, q_face_min_y, q_face_max_y)
   979	
   980	    flux_hi_x = q_face_hi_x * u_c_t * dy
   981	    flux_hi_y = q_face_hi_y * v_c_t * dx
   982	
   983	    net_hi_x = flux_hi_x[..., 1:, :] - flux_hi_x[..., :-1, :]
   984	    net_hi_y = flux_hi_y[..., 1:] - flux_hi_y[..., :-1]
   985	    dq_hi = -(net_hi_x + net_hi_y) / area
   986	
   987	    # ----------------------------------------------------------------
   988	    # Step 3: Zalesak-style blending (high ← low fallback)
   989	    #
   990	    # Compute anti-diffusive tendency = dq_hi - dq_low.
   991	    # Limit it so that the total tendency dq_low + alpha * (dq_hi - dq_low)
   992	    # does not push q outside [q_min, q_max] for any cell, even for a
   993	    # unit-CFL step where the tendency acts as a full update.
   994	    #
   995	    # Since the actual dt is always ≤ dx/u (CFL), limiting for dt=1
   996	    # (i.e. treating the tendency as the update) is strictly conservative.
   997	    # ----------------------------------------------------------------
   998	    ad = dq_hi - dq_low  # anti-diffusive tendency
   999	
  1000	    # Local min/max including all face-adjacent neighbours
  1001	    q_min = q_t
  1002	    q_max = q_t
  1003	    q_min = jnp.minimum(q_min, q_pad[..., :-2, 1:-1])   # west
  1004	    q_min = jnp.minimum(q_min, q_pad[..., 2:, 1:-1])    # east
  1005	    q_min = jnp.minimum(q_min, q_pad[..., 1:-1, :-2])   # south
  1006	    q_min = jnp.minimum(q_min, q_pad[..., 1:-1, 2:])    # north
  1007	    q_max = jnp.maximum(q_max, q_pad[..., :-2, 1:-1])
  1008	    q_max = jnp.maximum(q_max, q_pad[..., 2:, 1:-1])
  1009	    q_max = jnp.maximum(q_max, q_pad[..., 1:-1, :-2])
  1010	    q_max = jnp.maximum(q_max, q_pad[..., 1:-1, 2:])
  1011	
  1012	    # How much room does the low-order update leave?
  1013	    # After applying dq_low, q would be at q + dq_low (for unit "dt").
  1014	    # We allow the anti-diffusive part to bring it to at most q_max
  1015	    # and at least q_min.
  1016	    q_td = q_t + dq_low   # provisional (unit-step low-order update)
  1017	
  1018	    room_up = q_max - q_td     # how much we can still increase
  1019	    room_dn = q_td - q_min     # how much we can still decrease
  1020	
  1021	    # Per-cell blending factor alpha ∈ [0, 1]:
  1022	    # If ad > 0 (high-order wants to increase), alpha = room_up / ad
  1023	    # If ad < 0 (high-order wants to decrease), alpha = room_dn / |ad|
  1024	    # If ad == 0, alpha = 1 (no correction needed)
  1025	    #
  1026	    # NOTE: jnp.where evaluates BOTH branches for all elements before
  1027	    # selecting.  Division by ad when ad ≈ 0 produces inf/NaN values
  1028	    # that are discarded in the forward pass but propagate through
  1029	    # jax.grad.  Use safe denominators clamped away from zero so that
  1030	    # the unevaluated branch never divides by zero.
  1031	    eps = 1.0e-30
  1032	    safe_ad_pos = jnp.maximum(ad, eps)    # always > 0 — safe for branch ad > eps
  1033	    safe_ad_neg = jnp.minimum(ad, -eps)   # always < 0 — safe for branch ad < -eps
  1034	    alpha = jnp.where(
  1035	        ad > eps,
  1036	        jnp.minimum(1.0, room_up / safe_ad_pos),
  1037	        jnp.where(
  1038	            ad < -eps,
  1039	            jnp.minimum(1.0, room_dn / (-safe_ad_neg)),
  1040	            1.0,
  1041	        ),
  1042	    )
  1043	    alpha = jnp.clip(alpha, 0.0, 1.0)
  1044	
  1045	    result = dq_low + alpha * ad

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/experiments/rest_state.py | sed -n '130,210p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   130	            grid, z_coord,
   131	            T_surface=config.T_surface,
   132	            T_deep=config.T_deep,
   133	            S_uniform=config.S_uniform,
   134	            H_max=config.H_max,
   135	            land_lat_threshold=land_lat,
   136	        )
   137	
   138	    elif grid_type == "latlon":
   139	        from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
   140	        return rest_state_latlon_cgrid_ocean(
   141	            grid, z_coord,
   142	            T_surface=T_sfc, T_deep=T_deep,
   143	            S_uniform=config.S_uniform,
   144	            H_max=config.H_max,
   145	            land_lat_threshold=land_lat,
   146	        )
   147	
   148	    elif grid_type == "mpas":
   149	        from legoesm.ocean.init_mpas import rest_state_mpas_ocean
   150	        return rest_state_mpas_ocean(
   151	            grid, z_coord,
   152	            T_surface=config.T_surface,
   153	            T_deep=config.T_deep,
   154	            S_uniform=config.S_uniform,
   155	            H_max=config.H_max,
   156	            land_lat_threshold=land_lat,
   157	        )
   158	
   159	    elif grid_type == "spectral":
   160	        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
   161	        return rest_state_spectral_ocean(
   162	            grid, z_coord,
   163	            T_surface=T_sfc, T_deep=T_deep,
   164	            S_uniform=config.S_uniform,
   165	            H_max=config.H_max,
   166	            land_lat_threshold=config.effective_spectral_land_lat,
   167	        )
   168	
   169	    else:
   170	        raise ValueError(f"Unknown grid type: {grid_type}")
   171	
   172	
   173	def create_forcings(grid_type: str, grid, config: RestStateConfig = None):
   174	    """Create forcing functions for rest state experiment.
   175	
   176	    The rest state experiment has no external forcings - this is the point!
   177	    The ocean should remain in rest state under its own dynamics.
   178	
   179	    Returns
   180	    -------
   181	    None
   182	        No forcings for this experiment
   183	    """
   184	    return None
   185	
   186	
   187	def create_domain_config(config: RestStateConfig = None) -> Dict[str, Any]:
   188	    """Create domain configuration parameters.
   189	
   190	    Returns
   191	    -------
   192	    Dict[str, Any]
   193	        Domain configuration parameters for this experiment
   194	    """
   195	    if config is None:
   196	        config = RestStateConfig()
   197	
   198	    return {
   199	        "H_max": config.H_max,
   200	        "land_lat_threshold": config.land_lat_threshold,
   201	        "spectral_land_lat_threshold": config.spectral_land_lat_threshold,
   202	        "description": "Global ocean with polar land masses",
   203	    }
   204	
   205	
   206	def compute_drift_metrics(diagnostics: Dict[str, list], H_max: float) -> Dict[str, float]:
   207	    """Compute drift metrics specific to rest state validation.
   208	
   209	    For rest state, we care about absolute drift rather than relative drift
   210	    since initial values are often zero (e.g., mean_eta = 0).

exec
/bin/zsh -lc "python - <<'PY'
import jax.numpy as jnp
from legoesm.ocean.freshwater import freshwater_from_coupler
p=jnp.array([0.0,0.0])
lh=jnp.array([0.0,0.0])
fw=freshwater_from_coupler(p, lh, L_v=2.5e6, runoff_surface=None, runoff_subsurface=jnp.array([1.0,2.0]))
print('runoff', fw.runoff)
print('net', fw.precip-fw.evap+fw.runoff+fw.ice_fw)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 127 in 0ms:
zsh:1: command not found: python

codex
I’m now validating candidate issues with executable spot checks. The first check confirms whether subsurface runoff is dropped when `runoff_surface` is absent.
exec
/bin/zsh -lc "python3 - <<'PY'
import jax.numpy as jnp
from legoesm.ocean.freshwater import freshwater_from_coupler
p=jnp.array([0.0,0.0])
lh=jnp.array([0.0,0.0])
fw=freshwater_from_coupler(p, lh, L_v=2.5e6, runoff_surface=None, runoff_subsurface=jnp.array([1.0,2.0]))
print('runoff', fw.runoff)
print('net', fw.precip-fw.evap+fw.runoff+fw.ice_fw)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 676ms:
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
ModuleNotFoundError: No module named 'legoesm'

exec
/bin/zsh -lc "PYTHONPATH=src python3 - <<'PY'
import jax.numpy as jnp
from legoesm.ocean.freshwater import freshwater_from_coupler
p=jnp.array([0.0,0.0])
lh=jnp.array([0.0,0.0])
fw=freshwater_from_coupler(p, lh, L_v=2.5e6, runoff_surface=None, runoff_subsurface=jnp.array([1.0,2.0]))
print('runoff', fw.runoff)
print('net', fw.precip-fw.evap+fw.runoff+fw.ice_fw)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 885ms:
runoff [0. 0.]
net [0. 0.]

exec
/bin/zsh -lc "rg -n \"class LatLonGrid|def create_latlon_grid|LatLonGrid\\(\" src/legoesm/grids/latlon.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
25:class LatLonGrid(NamedTuple):
103:def create_latlon_grid(
167:    return LatLonGrid(
306:    grid = LatLonGrid(

exec
/bin/zsh -lc "nl -ba src/legoesm/grids/latlon.py | sed -n '90,190p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
    90	    @property
    91	    def grid_shape_2d(self) -> tuple[int, ...]:
    92	        return (self.n_lat, self.n_lon)
    93	
    94	    def to_columns(self, field):
    95	        extra = field.shape[2:]
    96	        return field.reshape(self.n_lat * self.n_lon, *extra)
    97	
    98	    def from_columns(self, cols):
    99	        extra = cols.shape[1:]
   100	        return cols.reshape(self.n_lat, self.n_lon, *extra)
   101	
   102	
   103	def create_latlon_grid(
   104	    n_lat: int,
   105	    n_lon: int | None = None,
   106	    radius: float = constants.R_earth,
   107	    omega: float = constants.Omega,
   108	    dtype=None,
   109	) -> LatLonGrid:
   110	    """Create a latitude-longitude grid.
   111	
   112	    Parameters
   113	    ----------
   114	    n_lat : int
   115	        Number of latitude points.
   116	    n_lon : int, optional
   117	        Number of longitude points. Defaults to 2 * n_lat.
   118	    radius : float
   119	        Sphere radius [m].
   120	    omega : float
   121	        Rotation rate [rad/s].
   122	
   123	    Returns
   124	    -------
   125	    LatLonGrid
   126	    """
   127	    if n_lon is None:
   128	        n_lon = 2 * n_lat
   129	
   130	    if dtype is None:
   131	        try:
   132	            from legoesm.core.precision import get_policy
   133	            dtype = get_policy().storage
   134	        except Exception:
   135	            dtype = jnp.float32
   136	
   137	    dlat = jnp.pi / n_lat
   138	    dlon = 2.0 * jnp.pi / n_lon
   139	
   140	    # Cell-centered latitudes: avoid exact poles
   141	    lat = jnp.linspace(
   142	        -jnp.pi / 2.0 + dlat / 2.0,
   143	        jnp.pi / 2.0 - dlat / 2.0,
   144	        n_lat,
   145	    )
   146	    lon = jnp.linspace(0.0, 2.0 * jnp.pi - dlon, n_lon)
   147	
   148	    lat2d, lon2d = jnp.meshgrid(lat, lon, indexing="ij")  # (n_lat, n_lon)
   149	
   150	    cos_lat = jnp.maximum(jnp.cos(lat), 1e-10)
   151	    sin_lat = jnp.sin(lat)
   152	
   153	    # Coriolis parameter
   154	    f = 2.0 * omega * sin_lat[:, None] * jnp.ones((1, n_lon))
   155	
   156	    # Metric terms — dx and dy span 2 cells (for centered differences)
   157	    # Single-cell width in lon = R * dlon * cos(lat)
   158	    # Two-cell span: dx = R * 2*dlon * cos(lat)
   159	    dx = radius * 2.0 * dlon * cos_lat[:, None] * jnp.ones((1, n_lon))
   160	    dy = float(radius * 2.0 * dlat)
   161	
   162	    # Cell area
   163	    area = radius**2 * dlat * dlon * cos_lat[:, None] * jnp.ones((1, n_lon))
   164	    total_area = jnp.sum(area)
   165	
   166	    _c = lambda a: a.astype(dtype) if hasattr(a, 'astype') else a
   167	    return LatLonGrid(
   168	        n_lat=n_lat,
   169	        n_lon=n_lon,
   170	        radius=float(radius),
   171	        lat=_c(lat),
   172	        lon=_c(lon),
   173	        lat2d=_c(lat2d),
   174	        lon2d=_c(lon2d),
   175	        cos_lat=_c(cos_lat),
   176	        sin_lat=_c(sin_lat),
   177	        f=_c(f),
   178	        dx=_c(dx),
   179	        dy=dy,
   180	        area=_c(area),
   181	        total_area=total_area,
   182	        dlon=float(dlon),
   183	        dlat=float(dlat),
   184	    )
   185	
   186	
   187	def create_regional_latlon_grid(
   188	    n_lat: int,
   189	    n_lon: int,
   190	    lat_south: float,

exec
/bin/zsh -lc "PYTHONPATH=src python3 - <<'PY'
import jax.numpy as jnp
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.core.precision import get_policy

grid=create_latlon_grid(8,16)
z=create_ocean_z_star(n_levels=5,H_max=5000)
mask=jnp.ones((8,16),dtype=jnp.float32)
state=rest_state_latlon_cgrid_ocean(grid,z,land_mask_override=mask)
print('policy storage', get_policy().storage)
print('H_bathy dtype', state.H_bathy.data.dtype)
print('eta dtype', state.eta.data.dtype)
print('land dtype', state.land_mask.data.dtype)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 1017ms:
/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/ocean/init_latlon_cgrid.py:98: UserWarning: Explicitly requested dtype float64 requested in full is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  H_bathy = jnp.full((n_lat, n_lon), H_max, dtype=jnp.float64)
policy storage <class 'jax.numpy.float32'>
H_bathy dtype float32
eta dtype float32
land dtype float32

exec
/bin/zsh -lc "PYTHONPATH=src JAX_ENABLE_X64=1 python3 - <<'PY'
import jax.numpy as jnp
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.core.precision import get_policy

grid=create_latlon_grid(8,16)
z=create_ocean_z_star(n_levels=5,H_max=5000)
mask=jnp.ones((8,16),dtype=jnp.float32)
state=rest_state_latlon_cgrid_ocean(grid,z,land_mask_override=mask)
print('policy storage', get_policy().storage)
print('H_bathy dtype', state.H_bathy.data.dtype)
print('eta dtype', state.eta.data.dtype)
print('land dtype', state.land_mask.data.dtype)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 1038ms:
policy storage <class 'jax.numpy.float32'>
H_bathy dtype float64
eta dtype float32
land dtype float32

web search: 
web search: Large McWilliams Doney 1994 nonlocal flux sign Q0 positive upward
exec
/bin/zsh -lc "rg -n \"n_substeps|barotropic_substeps|barotropic_time_filter|compute_filter_weights\" src/legoesm/ocean -g'*.py'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/ocean/mpas_config.py:47:    n_barotropic_substeps : int
src/legoesm/ocean/mpas_config.py:86:    n_barotropic_substeps: int = 30
src/legoesm/ocean/mpas_config.py:128:    barotropic_time_filter: str = "cosine"  # "box" or "cosine"
src/legoesm/ocean/dynamics/barotropic_mpas.py:39:    compute_filter_weights,
src/legoesm/ocean/dynamics/barotropic_mpas.py:45:def barotropic_substeps_mpas(
src/legoesm/ocean/dynamics/barotropic_mpas.py:51:    n_substeps,
src/legoesm/ocean/dynamics/barotropic_mpas.py:74:    n_substeps : int
src/legoesm/ocean/dynamics/barotropic_mpas.py:189:    use_cosine_filter = config.barotropic_time_filter == "cosine"
src/legoesm/ocean/dynamics/barotropic_mpas.py:190:    w_filter, w_total = compute_filter_weights(
src/legoesm/ocean/dynamics/barotropic_mpas.py:191:        n_substeps, eta.dtype, use_cosine=use_cosine_filter,
src/legoesm/ocean/dynamics/barotropic_mpas.py:303:        w_filter, length=n_substeps,
src/legoesm/ocean/dynamics/barotropic_mpas.py:307:    Hu_avg = Hu_sum_f / n_substeps  # transport: always box-filtered
src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:41:    barotropic_substeps_latlon_cgrid,
src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:228:        if config.n_barotropic_substeps < 1:
src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:230:                f"n_barotropic_substeps must be >= 1, got "
src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:231:                f"{config.n_barotropic_substeps!r}",
src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:297:        n_sub = self.config.n_barotropic_substeps
src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:311:                f"Suggest n_barotropic_substeps >= {n_min} "
src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:485:            dt_s = dt / self.config.n_barotropic_substeps
src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:486:            state_new, (Hu_avg, Hv_avg) = barotropic_substeps_latlon_cgrid(
src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:487:                state_mid, dt_s, self.config.n_barotropic_substeps,
src/legoesm/ocean/dynamics/barotropic_common.py:24:def compute_filter_weights(
src/legoesm/ocean/dynamics/barotropic_common.py:25:    n_substeps: int,
src/legoesm/ocean/dynamics/barotropic_common.py:42:    n_substeps : int
src/legoesm/ocean/dynamics/barotropic_common.py:47:        ``config.barotropic_time_filter == "cosine"``.
src/legoesm/ocean/dynamics/barotropic_common.py:51:    w_filter : jax.Array, shape (n_substeps,)
src/legoesm/ocean/dynamics/barotropic_common.py:57:        ``n_substeps`` for exact volume conservation.
src/legoesm/ocean/dynamics/barotropic_common.py:59:    i = jnp.arange(n_substeps, dtype=dtype)
src/legoesm/ocean/dynamics/barotropic_common.py:62:            2.0 * jnp.pi * (i - 0.5 * n_substeps) / n_substeps,
src/legoesm/ocean/dynamics/barotropic_common.py:65:        w_filter = jnp.ones(n_substeps, dtype=dtype)
src/legoesm/ocean/dynamics/barotropic_cgrid.py:126:def barotropic_substeps_cgrid(
src/legoesm/ocean/dynamics/barotropic_cgrid.py:129:    n_substeps: int,
src/legoesm/ocean/dynamics/barotropic_cgrid.py:153:    n_substeps : int
src/legoesm/ocean/dynamics/barotropic_cgrid.py:272:            scan_body, (eta, U_bar, V_bar), xs=None, length=n_substeps,
src/legoesm/ocean/dynamics/barotropic_cgrid.py:276:            0, n_substeps, substep_body, (eta, U_bar, V_bar),
src/legoesm/ocean/dynamics/ocean_model.py:37:from legoesm.ocean.dynamics.barotropic import barotropic_substeps
src/legoesm/ocean/dynamics/ocean_model.py:38:from legoesm.ocean.dynamics.barotropic_cgrid import barotropic_substeps_cgrid
src/legoesm/ocean/dynamics/ocean_model.py:145:        if config.n_barotropic_substeps < 1:
src/legoesm/ocean/dynamics/ocean_model.py:147:                "n_barotropic_substeps must be >= 1, got "
src/legoesm/ocean/dynamics/ocean_model.py:148:                f"{config.n_barotropic_substeps!r}",
src/legoesm/ocean/dynamics/ocean_model.py:355:        dt_s = dt / self.config.n_barotropic_substeps
src/legoesm/ocean/dynamics/ocean_model.py:357:            state_new = barotropic_substeps_cgrid(
src/legoesm/ocean/dynamics/ocean_model.py:359:                dt_s, self.config.n_barotropic_substeps,
src/legoesm/ocean/dynamics/ocean_model.py:363:            state_new = barotropic_substeps(
src/legoesm/ocean/dynamics/ocean_model.py:365:                dt_s, self.config.n_barotropic_substeps,
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:32:    compute_filter_weights,
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:88:def barotropic_substeps_latlon_cgrid(
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:91:    n_substeps: int,
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:107:    n_substeps : int
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:229:    use_cosine_filter = config.barotropic_time_filter == "cosine"
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:230:    w_filter, w_total = compute_filter_weights(
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:231:        n_substeps, eta.dtype, use_cosine=use_cosine_filter,
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:381:            scan_body, init_carry, xs=w_filter, length=n_substeps,
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:391:            0, n_substeps, fori_body, init_carry,
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:395:    Hu_avg = Hu_sum_f / n_substeps
src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:396:    Hv_avg = Hv_sum_f / n_substeps
src/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:47:``barotropic_substeps_mpas`` line 221+).
src/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:206:        :func:`barotropic_substeps_mpas` interface).
src/legoesm/ocean/dynamics/spectral_ocean_pe.py:677:        if self.config.n_barotropic_substeps != 1:
src/legoesm/ocean/dynamics/spectral_ocean_pe.py:679:                "SpectralOceanConfig.n_barotropic_substeps is currently ignored "
src/legoesm/ocean/dynamics/spectral_ocean_pe.py:728:        if config.n_barotropic_substeps < 1:
src/legoesm/ocean/dynamics/spectral_ocean_pe.py:730:                "n_barotropic_substeps must be >= 1, got "
src/legoesm/ocean/dynamics/spectral_ocean_pe.py:731:                f"{config.n_barotropic_substeps!r}",
src/legoesm/ocean/dynamics/ocean_model_mpas.py:34:    barotropic_substeps_mpas,
src/legoesm/ocean/dynamics/ocean_model_mpas.py:174:        n_sub = self.config.n_barotropic_substeps
src/legoesm/ocean/dynamics/ocean_model_mpas.py:187:                f"Suggest n_barotropic_substeps >= {n_min} "
src/legoesm/ocean/dynamics/ocean_model_mpas.py:302:        n_sub = config.n_barotropic_substeps
src/legoesm/ocean/dynamics/ocean_model_mpas.py:332:            eta_new, u_bar_new, Hu_avg = barotropic_substeps_mpas(
src/legoesm/ocean/dynamics/eta_floor.py:31:    ``allreduce`` calls.  At ``n_substeps≈30`` and ``n_iter=3`` that is
src/legoesm/ocean/state.py:100:    n_barotropic_substeps: int = 30
src/legoesm/ocean/state.py:176:    n_barotropic_substeps: int = 1
src/legoesm/ocean/state.py:248:    n_barotropic_substeps: int = 30
src/legoesm/ocean/state.py:484:    n_barotropic_substeps: int = 30
src/legoesm/ocean/state.py:495:    barotropic_time_filter: str = "cosine"  # "box" or "cosine" (shaped filter for time-averaging)
src/legoesm/ocean/dynamics/barotropic.py:161:def barotropic_substeps(
src/legoesm/ocean/dynamics/barotropic.py:164:    n_substeps: int,
src/legoesm/ocean/dynamics/barotropic.py:183:    n_substeps : int
src/legoesm/ocean/dynamics/barotropic.py:332:            scan_body, (eta, U_bar, V_bar), xs=None, length=n_substeps,
src/legoesm/ocean/dynamics/barotropic.py:338:            0, n_substeps, substep_body, (eta, U_bar, V_bar),
src/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:241:        Same return signature as ``barotropic_substeps_latlon_cgrid``.

exec
/bin/zsh -lc "PYTHONPATH=src python3 - <<'PY'
import jax.numpy as jnp
from legoesm.ocean.dynamics.barotropic_common import compute_filter_weights
w,tot=compute_filter_weights(1,jnp.float32,use_cosine=True)
print('w',w,'sum',tot)
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 877ms:
w [0.] sum 0.0

exec
/bin/zsh -lc 'rg -n "min_water_column|bottom_drag_r" src/legoesm/ocean/state.py src/legoesm/ocean/*config.py src/legoesm/ocean/dynamics/*config*.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 0ms:
zsh:1: no matches found: src/legoesm/ocean/dynamics/*config*.py

exec
/bin/zsh -lc 'rg -n "min_water_column|bottom_drag_r|barotropic_time_filter|n_barotropic_substeps" src/legoesm/ocean/state.py src/legoesm/ocean/mpas_config.py src/legoesm/ocean/spectral_config.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 2 in 0ms:
rg: src/legoesm/ocean/spectral_config.py: No such file or directory (os error 2)
src/legoesm/ocean/mpas_config.py:30:    bottom_drag_r : float
src/legoesm/ocean/mpas_config.py:47:    n_barotropic_substeps : int
src/legoesm/ocean/mpas_config.py:59:    min_water_column_m : float
src/legoesm/ocean/mpas_config.py:81:    bottom_drag_r: float = 0.0
src/legoesm/ocean/mpas_config.py:86:    n_barotropic_substeps: int = 30
src/legoesm/ocean/mpas_config.py:114:    min_water_column_m: float = 0.5
src/legoesm/ocean/mpas_config.py:128:    barotropic_time_filter: str = "cosine"  # "box" or "cosine"
src/legoesm/ocean/state.py:100:    n_barotropic_substeps: int = 30
src/legoesm/ocean/state.py:115:    min_water_column_m: float = 0.5
src/legoesm/ocean/state.py:176:    n_barotropic_substeps: int = 1
src/legoesm/ocean/state.py:187:    min_water_column_m: float = 0.5
src/legoesm/ocean/state.py:248:    n_barotropic_substeps: int = 30
src/legoesm/ocean/state.py:257:    min_water_column_m: float = 0.5
src/legoesm/ocean/state.py:479:    bottom_drag_r: float = 0.0
src/legoesm/ocean/state.py:484:    n_barotropic_substeps: int = 30
src/legoesm/ocean/state.py:495:    barotropic_time_filter: str = "cosine"  # "box" or "cosine" (shaped filter for time-averaging)
src/legoesm/ocean/state.py:497:    min_water_column_m: float = 0.5

web search: 
web search: 
exec
/bin/zsh -lc "PYTHONPATH=src python3 - <<'PY'
import jax.numpy as jnp
from legoesm.ocean.dynamics.ocean_tendency_common import iterate_eos_and_pressure_anomaly
from legoesm.ocean.eos import wright_eos
T=jnp.ones((2,2,3),dtype=jnp.float32)
S=35*jnp.ones_like(T)
mask=jnp.ones((2,2),dtype=jnp.float32)
fill=lambda q:q
dz=jnp.array([10.,20.,30.],dtype=jnp.float32)
_ = iterate_eos_and_pressure_anomaly(T,S,mask,fill,wright_eos,dz,1025.0,9.81,hi_precision_pressure=True)
print('done')
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 607ms:
/opt/homebrew/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  return lax_numpy.astype(self, dtype, copy=copy, device=device)
done

codex
I’m seeing a second concrete issue class: hard-coded `float64` in some ocean paths triggers dtype-policy mismatches and warnings under default `JAX_ENABLE_X64=0`. I’m scanning for all such occurrences now to separate true bugs from intentional high-precision paths.
exec
/bin/zsh -lc "rg -n \"jnp\\.float64|dtype=float64|astype\\(jnp\\.float64\\)\" src/legoesm/ocean src/legoesm/core/operators_cdgrid.py" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/ocean/init_latlon_cgrid.py:98:        H_bathy = jnp.full((n_lat, n_lon), H_max, dtype=jnp.float64)
src/legoesm/ocean/dynamics/ocean_model_mpas.py:468:            _f64 = jnp.float64
src/legoesm/ocean/experiments/inertia_gravity_wave.py:290:    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
src/legoesm/ocean/experiments/phillips_two_layer.py:234:    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
src/legoesm/ocean/dynamics/spectral_ocean_pe.py:316:    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
src/legoesm/ocean/dynamics/spectral_ocean_pe.py:948:    T_hat = sh_analysis_3d(grid, T_grid.astype(jnp.float64))
src/legoesm/ocean/dynamics/spectral_ocean_pe.py:949:    S_hat = sh_analysis_3d(grid, S_grid.astype(jnp.float64))
src/legoesm/ocean/dynamics/spectral_ocean_pe.py:951:    H_bathy_hat = sh_analysis(grid, H_bathy_grid.astype(jnp.float64))
src/legoesm/ocean/dynamics/sfno_ocean.py:245:            eta_hat = sh_analysis(grid, eta_new.astype(jnp.float64))
src/legoesm/ocean/dynamics/sfno_ocean.py:269:                T_hat = sh_analysis_3d(grid, T_new.astype(jnp.float64))
src/legoesm/ocean/dynamics/sfno_ocean.py:278:                S_hat = sh_analysis_3d(grid, S_new.astype(jnp.float64))
src/legoesm/ocean/dynamics/ocean_tendency_common.py:142:        rho_prime_hi = rho_prime.astype(jnp.float64)
src/legoesm/ocean/dynamics/ocean_tendency_common.py:143:        dz_hi = dz_ref.astype(jnp.float64)

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/dynamics/ocean_model_mpas.py | sed -n '430,520p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   430	                    tr, w, h_k_old, dt)
   431	            else:
   432	                vert_flux_div = flux_form_vertical_tracer_advection(tr, w)
   433	
   434	            # Full flux-form tracer update:
   435	            # h_new * T_new = h_old * T_mid - dt * vert - dt * horiz
   436	            hT_new = h_k_old * tr - dt * vert_flux_div - dt * div_hut
   437	            tr_new = hT_new / jnp.maximum(h_k_new, 1e-10)
   438	            # Preserve pre-step land values instead of zeroing them.
   439	            # Zeroing T, S on land each step and then averaging those
   440	            # zeros into coastal cells via the Neumann fill produced a
   441	            # cold/fresh front that propagated into the interior
   442	            # one-cell-per-step. See issue #164. Matches the lat-lon
   443	            # pattern in ocean_model_latlon_cgrid.py:493.
   444	            tr_new = jnp.where(mask_3d > 0.5, tr_new, tr)
   445	
   446	            if tr_name == 'T':
   447	                T_corrected = tr_new
   448	            else:
   449	                S_corrected = tr_new
   450	
   451	        # Final state construction with explicit land masking
   452	        T_final = T_corrected
   453	        S_final = S_corrected
   454	        
   455	        state_new = MPASOceanState(
   456	            u=state.u.replace(data=u_3d_new),
   457	            T=state.T.replace(data=T_final),
   458	            S=state.S.replace(data=S_final),
   459	            eta=state.eta.replace(data=eta_new * mask),
   460	            w=state.w.replace(data=w),
   461	            H_bathy=state.H_bathy,
   462	            land_mask=state.land_mask,
   463	        )
   464	
   465	        # 10. Conservation fixers (#166: pass expected forcing so fixer
   466	        # only removes numerical drift, not the forcing itself)
   467	        if config.use_conservation_fixer:
   468	            _f64 = jnp.float64
   469	            wa = mask.astype(_f64)[:, jnp.newaxis] * mesh.areaCell.astype(_f64)[:, jnp.newaxis]
   470	            expected_dHeat = jnp.sum(
   471	                tend.dT_dt.data.astype(_f64) * h_k_old.astype(_f64) * wa
   472	            ) * dt
   473	            expected_dSalt = jnp.sum(
   474	                tend.dS_dt.data.astype(_f64) * h_k_old.astype(_f64) * wa
   475	            ) * dt
   476	            state_new = mpas_ocean_conservation_fixer(
   477	                state_new, state, mesh, z_coord, config,
   478	                expected_dHeat=expected_dHeat,
   479	                expected_dSalt=expected_dSalt,
   480	            )
   481	
   482	        return cast_pytree(state_new, None, "storage")
   483	
   484	    def step_checked(
   485	        self,
   486	        state: MPASOceanState,
   487	        dt: float,
   488	        freshwater=None,
   489	        surface_forcing=None,
   490	        sponge=None,
   491	    ) -> MPASOceanState:
   492	        """Advance one timestep with host-side runtime validation.
   493	
   494	        Unlike the previous implementation which silently clipped tracers,
   495	        this raises on out-of-bounds values so the caller sees the failure.
   496	        """
   497	        if not self._cfl_checked:
   498	            self.check_barotropic_cfl(dt)
   499	            self._cfl_checked = True
   500	        state_new = self.step(state, dt, freshwater=freshwater,
   501	                              surface_forcing=surface_forcing,
   502	                              sponge=sponge)
   503	        if self.config.enable_runtime_checks:
   504	            self._assert_runtime_invariants(state_new)
   505	        return state_new
   506	
   507	    def _assert_runtime_invariants(self, state: MPASOceanState) -> None:
   508	        """Host-side runtime checks (matching cubed-sphere ocean model).
   509	
   510	        Fuses the finite-check, T-min/max, and S-min/max reductions
   511	        into a single ``jnp.stack`` + ``np.asarray`` host transfer so
   512	        the runtime checks cost one GPU stall per step instead of
   513	        five.  Uses ``jnp.where``-masked ``nanmin``/``nanmax`` to
   514	        avoid the boolean indexing path (``state.T.data[wet]`` allocates
   515	        a dynamically-shaped array that cannot be JIT'd; the masked
   516	        reductions are equivalent and stay on device).
   517	        """
   518	        mask = state.land_mask.data
   519	        wet = mask > 0.5
   520	        wet3 = wet[..., jnp.newaxis] if state.T.data.ndim > 1 else wet

web search: 
web search: nitrate uptake effect on total alkalinity +1 per mol NO3
web search: 
web search: ocean biogeochemical model dALK nitrate uptake sign -dNO3
exec
/bin/zsh -lc "rg -n \"dALK|alkalinity|NO3\" tests/ocean -g'*.py'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
tests/ocean/unit/test_ocean_biogeochemistry.py:126:        assert abiotic_state.NO3 is None
tests/ocean/unit/test_ocean_biogeochemistry.py:135:        for field in ["DIC", "ALK", "NO3", "Phyto", "Zoo", "Det"]:
tests/ocean/unit/test_ocean_biogeochemistry.py:141:        """NO3 should increase from surface to deep."""
tests/ocean/unit/test_ocean_biogeochemistry.py:142:        NO3 = npzd_state.NO3
tests/ocean/unit/test_ocean_biogeochemistry.py:143:        mean_surf = float(jnp.mean(NO3[:, 0]))
tests/ocean/unit/test_ocean_biogeochemistry.py:144:        mean_deep = float(jnp.mean(NO3[:, -1]))
tests/ocean/unit/test_ocean_biogeochemistry.py:260:        """Increasing alkalinity should decrease pCO2 (more CO3^2-)."""
tests/ocean/unit/test_ocean_biogeochemistry.py:393:        NO3 = jnp.full(shape, 10e-3)
tests/ocean/unit/test_ocean_biogeochemistry.py:402:        result = npzd_source_sink(NO3, P, Z, D, DIC, ALK, T, PAR, dz, cfg)
tests/ocean/unit/test_ocean_biogeochemistry.py:411:        NO3 = jnp.full(shape, 10e-3)
tests/ocean/unit/test_ocean_biogeochemistry.py:420:        result = npzd_source_sink(NO3, P, Z, D, DIC, ALK, T, PAR, dz, cfg)
tests/ocean/unit/test_ocean_biogeochemistry.py:438:        NO3 = jnp.full(shape, 10e-3)
tests/ocean/unit/test_ocean_biogeochemistry.py:440:            NO3, P, Z, D, DIC, ALK, T, zero_PAR, dz, cfg,
tests/ocean/unit/test_ocean_biogeochemistry.py:446:        zero_NO3 = jnp.full(shape, 1e-15)
tests/ocean/unit/test_ocean_biogeochemistry.py:449:            zero_NO3, P, Z, D, DIC, ALK, T, PAR, dz, cfg,
tests/ocean/unit/test_ocean_biogeochemistry.py:473:        assert new_state.NO3 is None
tests/ocean/unit/test_ocean_biogeochemistry.py:535:        for field in ["DIC", "ALK", "NO3", "Phyto", "Zoo", "Det"]:
tests/ocean/unit/test_ocean_biogeochemistry.py:552:        for field in ["DIC", "ALK", "NO3", "Phyto", "Zoo", "Det"]:
tests/ocean/unit/test_ocean_biogeochemistry.py:570:        for field in ["DIC", "ALK", "NO3", "Phyto", "Zoo", "Det"]:

exec
/bin/zsh -lc "nl -ba tests/ocean/unit/test_ocean_biogeochemistry.py | sed -n '360,470p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   360	            ALK_surf=jnp.array(2.3),
   361	            T_surf=jnp.array(20.0),
   362	            S_surf=jnp.array(35.0),
   363	            U10=jnp.array(7.0),
   364	        )
   365	        assert jnp.isfinite(diag.pCO2_ocean)
   366	        assert jnp.isfinite(diag.pH)
   367	        assert jnp.isfinite(diag.flux_co2)
   368	        assert jnp.isfinite(diag.k_w)
   369	
   370	
   371	# ============================================================================
   372	# NPZD
   373	# ============================================================================
   374	
   375	
   376	class TestNPZD:
   377	    def test_par_decreases_with_depth(self, z_ref):
   378	        z_full, dz = z_ref
   379	        cfg = BiogeoConfig(scheme="npzd")
   380	        PAR_surf = jnp.array([200.0])
   381	        Phyto = jnp.full((1, 10), 0.1e-3)
   382	        PAR = par_profile(PAR_surf, z_full, Phyto, dz, cfg)
   383	        assert PAR.shape == (1, 10)
   384	        # Surface should be brightest
   385	        assert float(PAR[0, 0]) > float(PAR[0, -1])
   386	        # Should be positive everywhere
   387	        assert jnp.all(PAR > 0)
   388	
   389	    def test_source_sink_shapes(self, z_ref):
   390	        z_full, dz = z_ref
   391	        cfg = BiogeoConfig(scheme="npzd")
   392	        shape = (4, 10)
   393	        NO3 = jnp.full(shape, 10e-3)
   394	        P = jnp.full(shape, 0.1e-3)
   395	        Z = jnp.full(shape, 0.05e-3)
   396	        D = jnp.full(shape, 0.01e-3)
   397	        DIC = jnp.full(shape, 2.1)
   398	        ALK = jnp.full(shape, 2.3)
   399	        T = jnp.full(shape, 20.0)
   400	        PAR = jnp.full(shape, 100.0)
   401	
   402	        result = npzd_source_sink(NO3, P, Z, D, DIC, ALK, T, PAR, dz, cfg)
   403	        assert len(result) == 6
   404	        for r in result:
   405	            assert r.shape == shape
   406	
   407	    def test_source_sink_finite(self, z_ref):
   408	        z_full, dz = z_ref
   409	        cfg = BiogeoConfig(scheme="npzd")
   410	        shape = (4, 10)
   411	        NO3 = jnp.full(shape, 10e-3)
   412	        P = jnp.full(shape, 0.1e-3)
   413	        Z = jnp.full(shape, 0.05e-3)
   414	        D = jnp.full(shape, 0.01e-3)
   415	        DIC = jnp.full(shape, 2.1)
   416	        ALK = jnp.full(shape, 2.3)
   417	        T = jnp.full(shape, 20.0)
   418	        PAR = jnp.full(shape, 100.0)
   419	
   420	        result = npzd_source_sink(NO3, P, Z, D, DIC, ALK, T, PAR, dz, cfg)
   421	        for r in result:
   422	            assert jnp.all(jnp.isfinite(r))
   423	
   424	    def test_growth_requires_light_and_nutrients(self, z_ref):
   425	        """Zero PAR or zero nutrients should give zero growth."""
   426	        z_full, dz = z_ref
   427	        cfg = BiogeoConfig(scheme="npzd")
   428	        shape = (1, 10)
   429	        P = jnp.full(shape, 0.5e-3)
   430	        Z = jnp.full(shape, 0.05e-3)
   431	        D = jnp.full(shape, 0.01e-3)
   432	        DIC = jnp.full(shape, 2.1)
   433	        ALK = jnp.full(shape, 2.3)
   434	        T = jnp.full(shape, 20.0)
   435	
   436	        # No light: phytoplankton should not grow
   437	        zero_PAR = jnp.zeros(shape)
   438	        NO3 = jnp.full(shape, 10e-3)
   439	        _, dP_dark, _, _, _, _ = npzd_source_sink(
   440	            NO3, P, Z, D, DIC, ALK, T, zero_PAR, dz, cfg,
   441	        )
   442	        # Growth term should be near zero; only losses
   443	        assert float(jnp.mean(dP_dark)) <= 0
   444	
   445	        # No nutrients: phytoplankton should not grow
   446	        zero_NO3 = jnp.full(shape, 1e-15)
   447	        PAR = jnp.full(shape, 200.0)
   448	        _, dP_no_N, _, _, _, _ = npzd_source_sink(
   449	            zero_NO3, P, Z, D, DIC, ALK, T, PAR, dz, cfg,
   450	        )
   451	        assert float(jnp.mean(dP_no_N)) <= 0
   452	
   453	
   454	# ============================================================================
   455	# Step function
   456	# ============================================================================
   457	
   458	
   459	class TestStepAbiotic:
   460	    def test_step_shapes(self, z_ref):
   461	        z_full, dz = z_ref
   462	        cfg = BiogeoConfig(scheme="abiotic")
   463	        state = init_biogeo_state((4, 10), z_full, cfg)
   464	        T = jnp.full((4, 10), 20.0)
   465	        S = jnp.full((4, 10), 35.0)
   466	        mask = jnp.ones(4)
   467	
   468	        new_state, diag = step_ocean_biogeochemistry(
   469	            state, T, S, dz, z_full, mask, dt=3600.0, cfg=cfg,
   470	        )

web search: 
web search: OCMIP alkalinity nitrate uptake +1
web search: 
web search: alkalinity tendency nitrate uptake remineralization sign ocean model equation dALK = -dNO3
exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/biogeochemistry/config.py | sed -n '1,240p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Configuration and state containers for ocean biogeochemistry.
     2	
     3	Two schemes:
     4	- ``"abiotic"``: DIC + alkalinity only. Carbonate chemistry with air-sea
     5	  CO2 exchange but no biology. Suitable for carbon-cycle experiments.
     6	- ``"npzd"``: Nutrient-Phytoplankton-Zooplankton-Detritus ecosystem model
     7	  coupled to the inorganic carbon cycle. Adds N, P, Z, D tracers with
     8	  light-limited growth, grazing, mortality, and remineralization.
     9	- ``"none"``: Disabled (default).
    10	"""
    11	
    12	from __future__ import annotations
    13	
    14	from typing import NamedTuple
    15	
    16	import jax
    17	import jax.numpy as jnp
    18	
    19	
    20	class BiogeoConfig(NamedTuple):
    21	    """Ocean biogeochemistry configuration.
    22	
    23	    Parameters
    24	    ----------
    25	    scheme : str
    26	        ``"none"``, ``"abiotic"``, or ``"npzd"``.
    27	
    28	    Carbonate chemistry (abiotic + npzd)
    29	    -------------------------------------
    30	    pCO2_atm : float
    31	        Atmospheric CO2 partial pressure [uatm]. Default 400.
    32	    wind_speed : float
    33	        Constant 10-m wind speed [m/s] for gas exchange (used when
    34	        no atmospheric coupling provides wind). Default 7.0.
    35	    K_h_bio : float
    36	        Horizontal diffusivity for biogeo tracers [m^2/s]. Default 1e3.
    37	    K_v_bio : float
    38	        Vertical diffusivity for biogeo tracers [m^2/s]. Default 1e-4.
    39	
    40	    NPZD parameters
    41	    ---------------
    42	    mu_max : float
    43	        Maximum phytoplankton growth rate [1/day]. Default 1.5.
    44	    k_N : float
    45	        Nutrient half-saturation [mol N/m^3]. Default 0.7e-3.
    46	    k_PAR : float
    47	        Light half-saturation [W/m^2]. Default 30.0.
    48	    alpha_P : float
    49	        Initial slope of P-I curve [1/(W/m^2)/day]. Default 0.025.
    50	    g_max : float
    51	        Maximum zooplankton grazing rate [1/day]. Default 0.6.
    52	    k_P : float
    53	        Grazing half-saturation [mol N/m^3]. Default 0.2e-3.
    54	    gamma_Z : float
    55	        Zooplankton assimilation efficiency. Default 0.7.
    56	    m_P : float
    57	        Phytoplankton linear mortality [1/day]. Default 0.05.
    58	    m_Z : float
    59	        Zooplankton quadratic mortality [1/(mol N/m^3)/day]. Default 0.2.
    60	    remin_rate : float
    61	        Detritus remineralization rate [1/day]. Default 0.05.
    62	    w_sink : float
    63	        Detritus sinking speed [m/day]. Default 10.0.
    64	    k_w_atten : float
    65	        Seawater light attenuation coefficient [1/m]. Default 0.04.
    66	    k_chl_atten : float
    67	        Chlorophyll self-shading [m^2/(mol N)]. Default 25.0.
    68	    R_CN : float
    69	        Redfield C:N ratio [mol C / mol N]. Default 6.625.
    70	    R_ON : float
    71	        Redfield O2:N ratio [mol O2 / mol N]. Default 10.625.
    72	    R_CaP : float
    73	        Rain ratio (CaCO3 production / organic C export). Default 0.07.
    74	
    75	    Initial conditions
    76	    ------------------
    77	    DIC_init : float
    78	        Initial DIC [mol C/m^3]. Default 2.1 (~ 2100 umol/kg).
    79	    ALK_init : float
    80	        Initial alkalinity [mol eq/m^3]. Default 2.3 (~ 2300 umol/kg).
    81	    NO3_init_surf : float
    82	        Initial surface NO3 [mol N/m^3]. Default 5e-3.
    83	    NO3_init_deep : float
    84	        Initial deep NO3 [mol N/m^3]. Default 30e-3.
    85	    Phyto_init : float
    86	        Initial phytoplankton [mol N/m^3]. Default 0.1e-3.
    87	    Zoo_init : float
    88	        Initial zooplankton [mol N/m^3]. Default 0.05e-3.
    89	    Det_init : float
    90	        Initial detritus [mol N/m^3]. Default 0.01e-3.
    91	    """
    92	    scheme: str = "none"
    93	
    94	    # Carbonate chemistry
    95	    pCO2_atm: float = 400.0
    96	    wind_speed: float = 7.0
    97	    K_h_bio: float = 1.0e3
    98	    K_v_bio: float = 1.0e-4
    99	
   100	    # NPZD
   101	    mu_max: float = 1.5
   102	    k_N: float = 0.7e-3
   103	    k_PAR: float = 30.0
   104	    alpha_P: float = 0.025
   105	    g_max: float = 0.6
   106	    k_P: float = 0.2e-3
   107	    gamma_Z: float = 0.7
   108	    m_P: float = 0.05
   109	    m_Z: float = 0.2
   110	    remin_rate: float = 0.05
   111	    w_sink: float = 10.0
   112	    k_w_atten: float = 0.04
   113	    k_chl_atten: float = 25.0
   114	    R_CN: float = 6.625
   115	    R_ON: float = 10.625
   116	    R_CaP: float = 0.07
   117	
   118	    # Initial conditions
   119	    DIC_init: float = 2.1
   120	    ALK_init: float = 2.3
   121	    NO3_init_surf: float = 5.0e-3
   122	    NO3_init_deep: float = 30.0e-3
   123	    Phyto_init: float = 0.1e-3
   124	    Zoo_init: float = 0.05e-3
   125	    Det_init: float = 0.01e-3
   126	
   127	
   128	class OceanBiogeoState(NamedTuple):
   129	    """Ocean biogeochemistry tracer state.
   130	
   131	    All fields have shape matching the ocean grid:
   132	    - Cubed-sphere: (6, n, n, nlev)
   133	    - MPAS: (nCells, nlev)
   134	
   135	    Fields are None when the corresponding scheme does not use them.
   136	    """
   137	    DIC: jax.Array          # Dissolved inorganic carbon [mol C/m^3]
   138	    ALK: jax.Array          # Total alkalinity [mol eq/m^3]
   139	    NO3: jax.Array | None = None  # Nitrate [mol N/m^3], NPZD only
   140	    Phyto: jax.Array | None = None  # Phytoplankton [mol N/m^3], NPZD only
   141	    Zoo: jax.Array | None = None    # Zooplankton [mol N/m^3], NPZD only
   142	    Det: jax.Array | None = None    # Detritus [mol N/m^3], NPZD only
   143	
   144	
   145	class BiogeoTendencies(NamedTuple):
   146	    """Source/sink tendencies for biogeochemistry tracers.
   147	
   148	    Same shapes as OceanBiogeoState. These are the
   149	    biogeochemistry-only tendencies (gas exchange, biology, etc.).
   150	    Advection/diffusion tendencies are handled by the ocean dynamics.
   151	    """
   152	    dDIC_dt: jax.Array
   153	    dALK_dt: jax.Array
   154	    dNO3_dt: jax.Array | None = None
   155	    dPhyto_dt: jax.Array | None = None
   156	    dZoo_dt: jax.Array | None = None
   157	    dDet_dt: jax.Array | None = None
   158	
   159	
   160	class AirSeaCO2Diagnostics(NamedTuple):
   161	    """Diagnostics from the air-sea CO2 flux calculation."""
   162	    pCO2_ocean: jax.Array   # Ocean surface pCO2 [uatm]
   163	    pH: jax.Array           # Surface pH
   164	    flux_co2: jax.Array     # Air-sea CO2 flux [mol C/m^2/s], positive into ocean
   165	    k_w: jax.Array          # Piston velocity [m/s]
   166	
   167	
   168	def init_biogeo_state(
   169	    shape_3d: tuple,
   170	    z_full_ref: jax.Array,
   171	    cfg: BiogeoConfig,
   172	) -> OceanBiogeoState | None:
   173	    """Initialize biogeochemistry state.
   174	
   175	    Parameters
   176	    ----------
   177	    shape_3d : tuple
   178	        Shape of 3D ocean arrays, e.g. (6, n, n, nlev) or (nCells, nlev).
   179	    z_full_ref : jax.Array
   180	        Reference depths [m], shape (nlev,), negative values.
   181	    cfg : BiogeoConfig
   182	        Configuration.
   183	
   184	    Returns
   185	    -------
   186	    OceanBiogeoState or None
   187	        None if scheme is "none".
   188	    """
   189	    if cfg.scheme == "none":
   190	        return None
   191	
   192	    nlev = shape_3d[-1]
   193	
   194	    DIC = jnp.full(shape_3d, cfg.DIC_init)
   195	    ALK = jnp.full(shape_3d, cfg.ALK_init)
   196	
   197	    if cfg.scheme == "abiotic":
   198	        return OceanBiogeoState(DIC=DIC, ALK=ALK)
   199	
   200	    elif cfg.scheme == "npzd":
   201	        # NO3: linear increase with depth from surface to deep
   202	        # z_full_ref is negative, so deeper = more negative
   203	        z_norm = jnp.clip(-z_full_ref / 1000.0, 0.0, 1.0)  # 0 at surface, 1 at 1000m+
   204	        NO3_profile = cfg.NO3_init_surf + (cfg.NO3_init_deep - cfg.NO3_init_surf) * z_norm
   205	        NO3 = jnp.broadcast_to(
   206	            NO3_profile.reshape((1,) * (len(shape_3d) - 1) + (nlev,)),
   207	            shape_3d,
   208	        )
   209	
   210	        Phyto = jnp.full(shape_3d, cfg.Phyto_init)
   211	        Zoo = jnp.full(shape_3d, cfg.Zoo_init)
   212	        Det = jnp.full(shape_3d, cfg.Det_init)
   213	
   214	        return OceanBiogeoState(
   215	            DIC=DIC, ALK=ALK, NO3=NO3,
   216	            Phyto=Phyto, Zoo=Zoo, Det=Det,
   217	        )
   218	
   219	    raise ValueError(f"Unknown biogeochemistry scheme: {cfg.scheme!r}")

exec
/bin/zsh -lc "rg -n \"ALK|alk\" src/legoesm/ocean/biogeochemistry -g'*.py'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/ocean/biogeochemistry/gas_exchange.py:72:    ALK_surf: jnp.ndarray,
src/legoesm/ocean/biogeochemistry/gas_exchange.py:89:    ALK_surf : array
src/legoesm/ocean/biogeochemistry/gas_exchange.py:90:        Surface alkalinity [mol eq/m^3].
src/legoesm/ocean/biogeochemistry/gas_exchange.py:108:    pCO2_ocean, pH = solve_carbonate_system(DIC_surf, ALK_surf, T_surf, S_surf, rho_sw)
src/legoesm/ocean/biogeochemistry/config.py:4:- ``"abiotic"``: DIC + alkalinity only. Carbonate chemistry with air-sea
src/legoesm/ocean/biogeochemistry/config.py:79:    ALK_init : float
src/legoesm/ocean/biogeochemistry/config.py:80:        Initial alkalinity [mol eq/m^3]. Default 2.3 (~ 2300 umol/kg).
src/legoesm/ocean/biogeochemistry/config.py:120:    ALK_init: float = 2.3
src/legoesm/ocean/biogeochemistry/config.py:138:    ALK: jax.Array          # Total alkalinity [mol eq/m^3]
src/legoesm/ocean/biogeochemistry/config.py:153:    dALK_dt: jax.Array
src/legoesm/ocean/biogeochemistry/config.py:195:    ALK = jnp.full(shape_3d, cfg.ALK_init)
src/legoesm/ocean/biogeochemistry/config.py:198:        return OceanBiogeoState(DIC=DIC, ALK=ALK)
src/legoesm/ocean/biogeochemistry/config.py:215:            DIC=DIC, ALK=ALK, NO3=NO3,
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:81:    ALK_surf = state.ALK[..., 0]
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:86:        DIC_surf, ALK_surf, T_surf, S_surf, U10,
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:97:    dALK_dt = jnp.zeros(shape, dtype=_state_dtype)
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:113:         dDIC_bio, dALK_bio) = npzd_source_sink(
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:115:            state.DIC, state.ALK, T_degC, PAR, dz_ref, cfg,
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:119:        dALK_dt = dALK_dt + dALK_bio * mask_3d
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:127:    dALK_dt = dALK_dt * mask_3d
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:131:        dALK_dt=dALK_dt,
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:202:    ALK_new = _soft_pos(state.ALK + dt * tend.dALK_dt)
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:217:        ALK=ALK_new,
src/legoesm/ocean/biogeochemistry/__init__.py:5:- ``"abiotic"``: DIC + alkalinity with carbonate chemistry and air-sea CO2.
src/legoesm/ocean/biogeochemistry/npzd.py:5:coupling to DIC, alkalinity, and air-sea CO2 exchange.
src/legoesm/ocean/biogeochemistry/npzd.py:12:5. Stoichiometric coupling to DIC/ALK via Redfield ratios
src/legoesm/ocean/biogeochemistry/npzd.py:87:    ALK: jnp.ndarray,
src/legoesm/ocean/biogeochemistry/npzd.py:97:    NO3, Phyto, Zoo, Det, DIC, ALK : array (..., nlev)
src/legoesm/ocean/biogeochemistry/npzd.py:109:    dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt : arrays
src/legoesm/ocean/biogeochemistry/npzd.py:180:    # CaCO3 production removes DIC and 2*ALK (in surface/euphotic zone)
src/legoesm/ocean/biogeochemistry/npzd.py:188:    dALK_dt = (-growth + remin + (1.0 - cfg.gamma_Z) * grazing
src/legoesm/ocean/biogeochemistry/npzd.py:191:    return dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt
src/legoesm/ocean/biogeochemistry/carbonate.py:4:computing surface ocean pCO2 from DIC and total alkalinity,
src/legoesm/ocean/biogeochemistry/carbonate.py:15:  inorganic carbon, alkalinity, and equations for K1 and K2.
src/legoesm/ocean/biogeochemistry/carbonate.py:145:    ALK: jnp.ndarray,
src/legoesm/ocean/biogeochemistry/carbonate.py:161:    ALK : array
src/legoesm/ocean/biogeochemistry/carbonate.py:162:        Total alkalinity [mol eq/m^3].
src/legoesm/ocean/biogeochemistry/carbonate.py:179:    ALK_kg = ALK / rho_sw
src/legoesm/ocean/biogeochemistry/carbonate.py:186:    # Carbonate alkalinity: A_C = ALK - A_B
src/legoesm/ocean/biogeochemistry/carbonate.py:192:    # Borate alkalinity using initial [H+]
src/legoesm/ocean/biogeochemistry/carbonate.py:195:    # Carbonate alkalinity
src/legoesm/ocean/biogeochemistry/carbonate.py:196:    A_C = ALK_kg - A_B

exec
/bin/zsh -lc "PYTHONPATH=src python3 - <<'PY'
import jax.numpy as jnp
from legoesm.ocean.biogeochemistry.npzd import npzd_source_sink
from legoesm.ocean.biogeochemistry.config import BiogeoConfig
cfg=BiogeoConfig(scheme='npzd',R_CaP=0.0)
shape=(1,1)
NO3=jnp.array([[1e-2]])
P=jnp.array([[1e-3]])
Z=jnp.array([[0.]])
D=jnp.array([[0.]])
DIC=jnp.array([[2.1]])
ALK=jnp.array([[2.3]])
T=jnp.array([[20.]])
PAR=jnp.array([[200.]])
dz=jnp.array([10.])
res=npzd_source_sink(NO3,P,Z,D,DIC,ALK,T,PAR,dz,cfg)
(dNO3,dP,dZ,dD,dDIC,dALK)=res
print('dNO3',float(dNO3[0,0]))
print('dALK',float(dALK[0,0]))
print('dALK + dNO3',float((dALK+dNO3)[0,0]))
print('dALK - dNO3',float((dALK-dNO3)[0,0]))
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 1111ms:
dNO3 -5.825560123184914e-08
dALK -5.825560123184914e-08
dALK + dNO3 -1.1651120246369828e-07
dALK - dNO3 0.0

web search: 
web search: create_ocean_z_star dz_deep default 200 epsilon 1e-3
exec
/bin/zsh -lc "rg -n \"alkalinity|dALK|NO3 consumed|potential alkalinity|TA includes\" docs src -g'*.md' -g'*.py' | head -n 200" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/ocean/biogeochemistry/gas_exchange.py:90:        Surface alkalinity [mol eq/m^3].
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:97:    dALK_dt = jnp.zeros(shape, dtype=_state_dtype)
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:113:         dDIC_bio, dALK_bio) = npzd_source_sink(
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:119:        dALK_dt = dALK_dt + dALK_bio * mask_3d
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:127:    dALK_dt = dALK_dt * mask_3d
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:131:        dALK_dt=dALK_dt,
src/legoesm/ocean/biogeochemistry/carbon_cycle.py:202:    ALK_new = _soft_pos(state.ALK + dt * tend.dALK_dt)
src/legoesm/ocean/biogeochemistry/npzd.py:5:coupling to DIC, alkalinity, and air-sea CO2 exchange.
src/legoesm/ocean/biogeochemistry/npzd.py:109:    dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt : arrays
src/legoesm/ocean/biogeochemistry/npzd.py:186:    # Alkalinity: -1 per mol NO3 consumed (nitrification sign convention)
src/legoesm/ocean/biogeochemistry/npzd.py:188:    dALK_dt = (-growth + remin + (1.0 - cfg.gamma_Z) * grazing
src/legoesm/ocean/biogeochemistry/npzd.py:191:    return dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt
src/legoesm/ocean/biogeochemistry/carbonate.py:4:computing surface ocean pCO2 from DIC and total alkalinity,
src/legoesm/ocean/biogeochemistry/carbonate.py:15:  inorganic carbon, alkalinity, and equations for K1 and K2.
src/legoesm/ocean/biogeochemistry/carbonate.py:162:        Total alkalinity [mol eq/m^3].
src/legoesm/ocean/biogeochemistry/carbonate.py:186:    # Carbonate alkalinity: A_C = ALK - A_B
src/legoesm/ocean/biogeochemistry/carbonate.py:192:    # Borate alkalinity using initial [H+]
src/legoesm/ocean/biogeochemistry/carbonate.py:195:    # Carbonate alkalinity
src/legoesm/ocean/biogeochemistry/__init__.py:5:- ``"abiotic"``: DIC + alkalinity with carbonate chemistry and air-sea CO2.
src/legoesm/ocean/biogeochemistry/config.py:4:- ``"abiotic"``: DIC + alkalinity only. Carbonate chemistry with air-sea
src/legoesm/ocean/biogeochemistry/config.py:80:        Initial alkalinity [mol eq/m^3]. Default 2.3 (~ 2300 umol/kg).
src/legoesm/ocean/biogeochemistry/config.py:138:    ALK: jax.Array          # Total alkalinity [mol eq/m^3]
src/legoesm/ocean/biogeochemistry/config.py:153:    dALK_dt: jax.Array

web search: 
web search: LMD94 Eq.19 nonlocal term sign dT/dt = -d/dz(wT)
exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/vertical_mixing/config.py | sed -n '1,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Configuration for ocean vertical mixing schemes."""
     2	
     3	from __future__ import annotations
     4	
     5	from typing import NamedTuple
     6	
     7	
     8	class ConstantVerticalMixingConfig(NamedTuple):
     9	    """Constant-coefficient vertical mixing."""
    10	    A_v: float = 1e-3   # Vertical viscosity [m^2/s]
    11	    K_v: float = 1e-4   # Vertical diffusivity [m^2/s]
    12	
    13	
    14	class RichardsonVerticalMixingConfig(NamedTuple):
    15	    """Pacanowski & Philander (1981) Richardson-number dependent mixing."""
    16	    K_0: float = 5e-3    # Maximum diffusivity [m^2/s]
    17	    alpha: float = 5.0   # Stability parameter
    18	    n: int = 2           # Exponent
    19	    K_bg: float = 1e-5   # Background diffusivity [m^2/s]
    20	    A_bg: float = 1e-4   # Background viscosity [m^2/s]
    21	    Pr_t: float = 10.0   # Turbulent Prandtl number
    22	
    23	
    24	class KPPConfig(NamedTuple):
    25	    """LMD94-style K-Profile Parameterization.
    26	
    27	    Extends the Large, McWilliams & Doney (1994) KPP boundary-layer
    28	    parameterization with linear interpolation for h_bl, proper
    29	    turbulent velocity scales, and interior convective instability
    30	    handling.
    31	
    32	    The caller should supply surface wind stress (tau_x, tau_y) and
    33	    surface buoyancy flux (B_f) so that friction velocity and turbulent
    34	    velocity scales can be computed correctly.  If these are not
    35	    provided, simplified proxies from the ocean state are used.
    36	    """
    37	    Ri_crit: float = 0.25   # Critical bulk Richardson number (LMD94)
    38	    Cv: float = 1.6          # Unresolved shear coefficient
    39	    kappa_vk: float = 0.4    # von Karman constant
    40	    K_max: float = 1.0       # Maximum diffusivity [m^2/s]
    41	    K_bg: float = 1e-5       # Background diffusivity [m^2/s]
    42	    A_bg: float = 1e-4       # Background viscosity [m^2/s]
    43	    gamma_T: float = 6.33    # Non-local transport coefficient for T
    44	    gamma_S: float = 6.33    # Non-local transport coefficient for S
    45	    K_conv: float = 1.0      # Convective mixing diffusivity [m^2/s]
    46	    Ri_conv: float = 0.0     # Ri threshold for convective instability
    47	    K_0_shear: float = 5e-3  # LMD94 interior shear instability peak K [m^2/s]
    48	    Ri_0: float = 0.7        # LMD94 critical Ri for interior shear mixing
    49	    c_s: float = 98.96       # LMD94 parameter for V_t^2 (Appendix B)
    50	    c_b: float = 0.599       # LMD94 convective velocity scale parameter
    51	    epsilon_lmd: float = 0.1  # LMD94 surface-layer fraction (App. A/B)
    52	    crossing_sharpness: float = 20.0  # Sigmoid sharpness for h_bl crossing-depth selector
    53	    crossing_threshold: float = 0.1   # Crossing-strength threshold for h_bl blend
    54	
    55	
    56	class VerticalMixingConfig(NamedTuple):
    57	    """Top-level vertical mixing configuration."""
    58	    scheme: str = "constant"  # "constant", "richardson", "kpp", "none"
    59	    constant: ConstantVerticalMixingConfig = ConstantVerticalMixingConfig()
    60	    richardson: RichardsonVerticalMixingConfig = RichardsonVerticalMixingConfig()
    61	    kpp: KPPConfig = KPPConfig()

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/vertical_mixing/richardson.py | sed -n '1,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Pacanowski & Philander (1981) Richardson-number dependent mixing.
     2	
     3	K = K_0 / (1 + alpha * Ri)^n + K_bg
     4	A = K * Pr_t + A_bg
     5	"""
     6	
     7	from __future__ import annotations
     8	
     9	import jax
    10	import jax.numpy as jnp
    11	
    12	from legoesm.ocean.eos import compute_buoyancy_frequency
    13	from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
    14	from legoesm.ocean.physics.vertical_mixing.config import RichardsonVerticalMixingConfig
    15	from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput
    16	from legoesm.ocean.vertical import OceanZStarCoordinate
    17	
    18	_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)
    19	
    20	
    21	def richardson_vertical_mixing(
    22	    u: jnp.ndarray,
    23	    v: jnp.ndarray,
    24	    T: jnp.ndarray,
    25	    S: jnp.ndarray,
    26	    rho: jnp.ndarray,
    27	    z_coord: OceanZStarCoordinate,
    28	    jacobian: jnp.ndarray,
    29	    cfg: RichardsonVerticalMixingConfig,
    30	) -> VerticalMixingOutput:
    31	    """Apply Richardson-number dependent vertical mixing.
    32	
    33	    Parameters
    34	    ----------
    35	    u, v : array (6, n, n, nlev)
    36	    T, S : array (6, n, n, nlev)
    37	    rho : array (6, n, n, nlev)
    38	        In-situ density.
    39	    z_coord : OceanZStarCoordinate
    40	    jacobian : array (6, n, n)
    41	    cfg : RichardsonVerticalMixingConfig
    42	
    43	    Returns
    44	    -------
    45	    VerticalMixingOutput
    46	    """
    47	    eps = _EPS
    48	
    49	    # N^2 at interfaces
    50	    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
    51	
    52	    # Shear^2 at interfaces
    53	    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    54	    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    55	    du = u[..., :-1] - u[..., 1:]
    56	    dv = v[..., :-1] - v[..., 1:]
    57	    S2 = (du**2 + dv**2) / jnp.maximum(dz_half**2, eps)
    58	
    59	    # Richardson number
    60	    Ri = N2 / jnp.maximum(S2, eps)
    61	    Ri = jnp.maximum(Ri, 0.0)  # Clip negative Ri (unstable → max mixing)
    62	
    63	    # Diffusivity and viscosity at interfaces
    64	    K_v = cfg.K_0 / (1.0 + cfg.alpha * Ri) ** cfg.n + cfg.K_bg
    65	    A_v = K_v * cfg.Pr_t + cfg.A_bg
    66	
    67	    # Apply variable-K vertical diffusion
    68	    vel = jnp.stack([u, v], axis=0)
    69	    vel_tend = jax.vmap(
    70	        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, A_v),
    71	        in_axes=0, out_axes=0,
    72	    )(vel)
    73	
    74	    tracers = jnp.stack([T, S], axis=0)
    75	    tr_tend = jax.vmap(
    76	        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, K_v),
    77	        in_axes=0, out_axes=0,
    78	    )(tracers)
    79	
    80	    return VerticalMixingOutput(
    81	        du_dt=vel_tend[0],
    82	        dv_dt=vel_tend[1],
    83	        dT_dt=tr_tend[0],
    84	        dS_dt=tr_tend[1],
    85	        K_v=K_v,
    86	        A_v=A_v,
    87	    )

exec
/bin/zsh -lc "PYTHONPATH=src python3 - <<'PY'
import jax.numpy as jnp
from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.vertical import create_ocean_z_star

z=create_ocean_z_star(n_levels=6,H_max=1000,dz_surface=50,dz_deep=200)
shape=(1,1,1,6)
# stably stratified rho increasing with depth
T=jnp.array([[[[20,18,16,14,12,10]]]],dtype=jnp.float32)
S=jnp.full(shape,35.0,dtype=jnp.float32)
rho=jnp.array([[[[1024,1025,1026,1027,1028,1029]]]],dtype=jnp.float32)
u=jnp.zeros(shape,dtype=jnp.float32)
v=jnp.zeros(shape,dtype=jnp.float32)
eta=jnp.zeros((1,1,1),dtype=jnp.float32)
J=jnp.ones((1,1,1),dtype=jnp.float32)
out=kpp_vertical_mixing(u,v,T,S,rho,eta,z,J,KPPConfig())
print('K_v min/max',float(out.K_v.min()),float(out.K_v.max()))
print('A_v min/max',float(out.A_v.min()),float(out.A_v.max()))
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 2123ms:
K_v min/max 1.9999999494757503e-05 0.013193544931709766
A_v min/max 0.00010999999358318746 0.013283545151352882

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/convection/output.py | sed -n '1,120p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Output container for ocean convection."""
     2	
     3	from __future__ import annotations
     4	
     5	from typing import NamedTuple
     6	
     7	import jax.numpy as jnp
     8	
     9	
    10	class OceanConvectionOutput(NamedTuple):
    11	    """Ocean convection tendencies and diagnostics.
    12	
    13	    Tendencies have shape (6, n, n, nlev).
    14	    convection_flag has shape (6, n, n, nlev-1) at interfaces.
    15	    """
    16	    dT_dt: jnp.ndarray
    17	    dS_dt: jnp.ndarray
    18	    convection_flag: jnp.ndarray   # 1 where convection is active

exec
/bin/zsh -lc 'PYTHONPATH=src pytest -q tests/ocean/unit/test_advection_dst3.py -k "zd3 or fct"' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 127 in 0ms:
zsh:1: command not found: pytest

exec
/bin/zsh -lc ls in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
CLAUDE.md
FIXME_test_audit_prompt.md
NEXT_STEPS.md
README.md
SPECIFICATION.md
config
diagnostics
docs
evaluations
legoESM.png
output
pyproject.toml
results
scripts
src
tests

exec
/bin/zsh -lc 'uv --version' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 127 in 0ms:
zsh:1: command not found: uv

exec
/bin/zsh -lc 'PYTHONPATH=src python3 -m pytest --version' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 0ms:
/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest

exec
/bin/zsh -lc 'rg -n "def _cgrid_fct_fluxes_2d|def cgrid_fct_fluxes|def cgrid_mass_flux_divergence" src/legoesm/core/operators_cdgrid.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
710:def cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid,
851:def _cgrid_fct_fluxes_2d(q, u_c, v_c, cdgrid):

exec
/bin/zsh -lc "rg -n \"class CubedSphereCDGrid|def create_cubed_sphere_cdgrid|create_cdgrid|CDGrid\" src/legoesm -g'*.py'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/supported_matrix.py:40:        "cdgrid_shallow_water", "CDGridShallowWaterModel",
src/legoesm/supported_matrix.py:45:        "cdgrid_primitive_equations", "CDGridPrimitiveEquationModel",
src/legoesm/supported_matrix.py:50:        "cdgrid_compressible_euler", "CDGridCompressibleEulerModel",
src/legoesm/supported_matrix.py:175:        "CDGridShallowWaterModel",
src/legoesm/supported_matrix.py:176:        "Use CDGridShallowWaterModel explicitly.",
src/legoesm/supported_matrix.py:179:        "CDGridPrimitiveEquationModel",
src/legoesm/supported_matrix.py:180:        "Use CDGridPrimitiveEquationModel explicitly.",
src/legoesm/supported_matrix.py:183:        "CDGridPrimitiveEquationConfig",
src/legoesm/supported_matrix.py:184:        "Use CDGridPrimitiveEquationConfig explicitly.",
src/legoesm/supported_matrix.py:187:        "CDGridCompressibleEulerModel",
src/legoesm/supported_matrix.py:188:        "Use CDGridCompressibleEulerModel explicitly.",
src/legoesm/supported_matrix.py:191:        "CDGridShallowWaterModel",
src/legoesm/supported_matrix.py:192:        "The FV cubed-sphere solver is CDGridShallowWaterModel.",
src/legoesm/supported_matrix.py:195:        "CDGridPrimitiveEquationModel",
src/legoesm/supported_matrix.py:196:        "The FV cubed-sphere solver is CDGridPrimitiveEquationModel.",
src/legoesm/supported_matrix.py:199:        "CDGridCompressibleEulerModel",
src/legoesm/supported_matrix.py:200:        "The FV cubed-sphere solver is CDGridCompressibleEulerModel.",
src/legoesm/supported_matrix.py:203:        "CDGridPrimitiveEquationConfig",
src/legoesm/supported_matrix.py:204:        "The FV cubed-sphere config is CDGridPrimitiveEquationConfig.",
src/legoesm/supported_matrix.py:207:        "CDGridCompressibleEulerConfig",
src/legoesm/supported_matrix.py:208:        "The FV cubed-sphere config is CDGridCompressibleEulerConfig.",
src/legoesm/supported_matrix.py:211:        "CDGridShallowWaterModel",
src/legoesm/supported_matrix.py:212:        "The C-grid cubed-sphere solver is CDGridShallowWaterModel.",
src/legoesm/supported_matrix.py:215:        "CDGridPrimitiveEquationModel",
src/legoesm/supported_matrix.py:216:        "The C-grid cubed-sphere solver is CDGridPrimitiveEquationModel.",
src/legoesm/supported_matrix.py:219:        "CDGridCompressibleEulerModel",
src/legoesm/supported_matrix.py:220:        "The C-grid cubed-sphere solver is CDGridCompressibleEulerModel.",
src/legoesm/ocean/dynamics/barotropic_cgrid.py:33:from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
src/legoesm/ocean/dynamics/barotropic_cgrid.py:110:    cdgrid : CubedSphereCDGrid
src/legoesm/ocean/dynamics/barotropic_cgrid.py:131:    cdgrid: CubedSphereCDGrid,
src/legoesm/ocean/dynamics/barotropic_cgrid.py:155:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv_tp_2d.py:20:from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
src/legoesm/core/fv_tp_2d.py:252:    # bypasses the entire block.  Production CDGrid uses
src/legoesm/core/fv_tp_2d.py:623:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv_tp_2d.py:880:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:30:from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:499:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:684:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:883:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:1117:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:1148:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:1326:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:1377:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:1565:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:1765:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:1848:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:1870:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:2023:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:2040:    #      `CDGridShallowWaterModel`, which DOES apply hyperdiff, or
src/legoesm/core/operators_cdgrid.py:2050:            f"on a cubed-sphere SW path, use `CDGridShallowWaterModel` "
src/legoesm/core/operators_cdgrid.py:2101:        # production CDGrid mass-flux PPM picks up Fortran's iord<7
src/legoesm/core/operators_cdgrid.py:2248:        # `CDGridShallowWaterConfig.dddmp_prod`
src/legoesm/core/operators_cdgrid.py:2445:    cdgrid : CubedSphereCDGrid
src/legoesm/core/operators_cdgrid.py:2480:    cdgrid : CubedSphereCDGrid
src/legoesm/ocean/dynamics/ocean_pe_cdgrid.py:42:from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
src/legoesm/ocean/dynamics/ocean_pe_cdgrid.py:73:    cdgrid: CubedSphereCDGrid,
src/legoesm/ocean/dynamics/ocean_pe_cdgrid.py:88:    cdgrid : CubedSphereCDGrid
src/legoesm/driver/component_factory.py:193:            CDGridShallowWaterModel, CDGridShallowWaterConfig,
src/legoesm/driver/component_factory.py:195:        cfg = CDGridShallowWaterConfig(
src/legoesm/driver/component_factory.py:201:        return CDGridShallowWaterModel(grid, cfg)
src/legoesm/driver/component_factory.py:205:            CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
src/legoesm/driver/component_factory.py:207:        cfg = CDGridPrimitiveEquationConfig(
src/legoesm/driver/component_factory.py:214:        return CDGridPrimitiveEquationModel(grid, sigma, cfg)
src/legoesm/driver/component_factory.py:218:            CDGridCompressibleEulerModel, CDGridCompressibleEulerConfig,
src/legoesm/driver/component_factory.py:220:        cfg = CDGridCompressibleEulerConfig(
src/legoesm/driver/component_factory.py:240:        return CDGridCompressibleEulerModel(grid, height_coord, terrain_metric, cfg)
src/legoesm/core/fv3_sw_core.py:28:from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:63:    cdgrid : CubedSphereCDGrid (must have an active duogrid with ng>=halo)
src/legoesm/core/fv3_sw_core.py:161:    cdgrid : CubedSphereCDGrid (must have an active duogrid with ng≥3)
src/legoesm/core/fv3_sw_core.py:286:    cdgrid : CubedSphereCDGrid (must have an active duogrid with ng>=3)
src/legoesm/core/fv3_sw_core.py:352:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:841:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:1221:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:1306:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:1577:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:2056:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:2174:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:2281:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:2290:        `CDGridShallowWaterConfig.div_damp`).  The FB chain's
src/legoesm/core/fv3_sw_core.py:2343:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:2702:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:2898:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:2908:        `div_damp` and share `CDGridShallowWaterConfig.div_damp`:
src/legoesm/core/fv3_sw_core.py:3141:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_sw_core.py:3148:        `div_damp` and share `CDGridShallowWaterConfig.div_damp`).
src/legoesm/cli.py:132:    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import CDGridShallowWaterModel as ShallowWaterModel, CDGridShallowWaterConfig as ShallowWaterConfig
src/legoesm/cli.py:254:    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import CDGridShallowWaterModel as ShallowWaterModel
src/legoesm/core/fv3_d_sw5_corner_divergence.py:70:    cdgrid : CubedSphereCDGrid
src/legoesm/atmosphere/__init__.py:4:    CDGridShallowWaterModel as ShallowWaterModel,
src/legoesm/core/fv3_del6_vt_flux.py:37:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_del6_vt_flux.py:72:    # rest of the CDGrid operators (discrete Stokes theorem etc).
src/legoesm/core/fv3_del6_vt_flux.py:107:    cdgrid : CubedSphereCDGrid
src/legoesm/core/fv3_del6_vt_flux.py:241:    cdgrid : CubedSphereCDGrid
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:65:    CubedSphereCDGrid,
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:92:class CDGridPrimitiveEquationConfig(NamedTuple):
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:155:    cdgrid: CubedSphereCDGrid,
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:156:    config: CDGridPrimitiveEquationConfig = CDGridPrimitiveEquationConfig(),
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:171:    cdgrid : CubedSphereCDGrid
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:172:    config : CDGridPrimitiveEquationConfig
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:628:    cdgrid: CubedSphereCDGrid,
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:665:class CDGridPrimitiveEquationModel(IntegrationMixin):
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:678:    config : CDGridPrimitiveEquationConfig, optional
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:685:        config: CDGridPrimitiveEquationConfig | None = None,
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:689:        self.config = config or CDGridPrimitiveEquationConfig()
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:903:    cdgrid: CubedSphereCDGrid,
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:904:    config: CDGridPrimitiveEquationConfig = CDGridPrimitiveEquationConfig(),
src/legoesm/atmosphere/dynamics/primitive_eq_cdgrid.py:971:    cdgrid: CubedSphereCDGrid,
src/legoesm/config.py:92:        #   shallow_water  + cdgrid   → CDGridShallowWaterModel
src/legoesm/config.py:94:        #   hydrostatic    + cdgrid   → CDGridPrimitiveEquationModel
src/legoesm/config.py:96:        #   nonhydrostatic + cdgrid   → CDGridCompressibleEulerModel
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:39:    CubedSphereCDGrid,
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:51:class CDGridShallowWaterState(NamedTuple):
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:65:class CDGridShallowWaterConfig(NamedTuple):
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:68:    Iter-1009 dual-target preset (production CDGrid path):
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:70:        cfg = CDGridShallowWaterConfig(
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:100:    #      file, which is invoked by `CDGridShallowWaterModel.
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:259:    # in `fv3_sw_tendencies` from the iter-892 CDGrid PPM
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:268:    # Motivation: the iter-892 CDGrid PPM is NOT true FV3
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:345:    # `CDGridShallowWaterModel` default-config users (Codex pass-3
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:346:    # finding).  Advanced `CDGridShallowWaterModel` callers wanting
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:406:) -> CDGridShallowWaterConfig:
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:446:    CDGridShallowWaterConfig
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:471:    return CDGridShallowWaterConfig(
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:485:    state: CDGridShallowWaterState,
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:486:    cdgrid: CubedSphereCDGrid,
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:487:    config: CDGridShallowWaterConfig = CDGridShallowWaterConfig(),
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:496:    state : CDGridShallowWaterState
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:497:    cdgrid : CubedSphereCDGrid
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:498:    config : CDGridShallowWaterConfig
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:511:    # warning at `CDGridShallowWaterModel.__init__` catches the
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:513:    _default_dddmp_prod = CDGridShallowWaterConfig._field_defaults[
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:540:    # change `CDGridShallowWaterModel(default_config)` behaviour
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:541:    # (Codex pass-3 finding).  Advanced `CDGridShallowWaterModel`
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:565:class CDGridShallowWaterModel(IntegrationMixin):
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:575:    config : CDGridShallowWaterConfig, optional
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:581:        config: CDGridShallowWaterConfig | None = None,
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:585:        self.config = config or CDGridShallowWaterConfig()
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:588:        # to a non-default value, because `CDGridShallowWaterModel`
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:594:        _default_dddmp_prod = CDGridShallowWaterConfig._field_defaults[
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:599:                f"CDGridShallowWaterConfig.dddmp_prod="
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:601:                f"CDGridShallowWaterModel instance, but this model's "
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:608:                f"`CDGridShallowWaterModel` users wanting adaptive "
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:614:    def set_initial_mass(self, state: CDGridShallowWaterState):
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:618:    def _sync_dgrid_boundary(self, state: CDGridShallowWaterState):
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:694:    def tendencies(self, state: CDGridShallowWaterState):
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:702:        self, state: CDGridShallowWaterState, dt: float,
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:703:    ) -> CDGridShallowWaterState:
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:713:            return CDGridShallowWaterState(
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:768:    config : CDGridShallowWaterConfig, optional
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:774:        self.config = config or CDGridShallowWaterConfig()
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:847:            # via `CDGridShallowWaterConfig`.  Default OFF preserves
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:858:            # enable it via `CDGridShallowWaterConfig`.  Default OFF.
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:921:        self.config = config or CDGridShallowWaterConfig()
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:993:                    # CDGrid path picks up Fortran's iord<7 cube-edge
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:996:                    # contract is now superseded — production CDGrid
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:1001:                    # production CDGrid PPM so the LEFT cube-edge override
src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:1008:                    # Fortran-faithful flag to the production CDGrid PPM.
src/legoesm/atmosphere/dynamics/primitive_eq_latlon_cgrid.py:644:        Physics is evaluated inside each RK stage (matching the CDGrid
src/legoesm/atmosphere/dynamics/primitive_eq_latlon_cgrid.py:796:        Physics is evaluated inside each RK stage (matching CDGrid PE),
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:56:    CubedSphereCDGrid,
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:75:class CDGridCompressibleEulerConfig(NamedTuple):
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:106:    cdgrid: CubedSphereCDGrid,
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:107:    config: CDGridCompressibleEulerConfig,
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:118:    cdgrid : CubedSphereCDGrid
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:119:    config : CDGridCompressibleEulerConfig
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:492:class CDGridCompressibleEulerModel(IntegrationMixin):
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:500:    config : CDGridCompressibleEulerConfig, optional
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:508:        config: CDGridCompressibleEulerConfig | None = None,
src/legoesm/atmosphere/dynamics/compressible_euler_cdgrid.py:512:        self.config = config or CDGridCompressibleEulerConfig()
src/legoesm/grids/cubed_sphere_cdgrid.py:38:class CubedSphereCDGrid(NamedTuple):
src/legoesm/grids/cubed_sphere_cdgrid.py:529:def create_cubed_sphere_cdgrid(
src/legoesm/grids/cubed_sphere_cdgrid.py:533:) -> CubedSphereCDGrid:
src/legoesm/grids/cubed_sphere_cdgrid.py:551:    CubedSphereCDGrid
src/legoesm/grids/cubed_sphere_cdgrid.py:1176:    return CubedSphereCDGrid(
src/legoesm/grids/cubed_sphere_cdgrid.py:1247:    cdgrid : CubedSphereCDGrid
src/legoesm/atmosphere/dynamics/__init__.py:27:``shallow_water``            cdgrid          CDGridShallowWaterModel
src/legoesm/atmosphere/dynamics/__init__.py:31:``hydrostatic``              cdgrid          CDGridPrimitiveEquationModel
src/legoesm/atmosphere/dynamics/__init__.py:36:``nonhydrostatic``           cdgrid          CDGridCompressibleEulerModel
src/legoesm/atmosphere/dynamics/__init__.py:47:  → use ``CDGridShallowWaterModel``
src/legoesm/atmosphere/dynamics/__init__.py:49:  ``CGPrimitiveEquationModel`` → use ``CDGridPrimitiveEquationModel``
src/legoesm/atmosphere/dynamics/__init__.py:51:  ``CGCompressibleEulerModel`` → use ``CDGridCompressibleEulerModel``
src/legoesm/atmosphere/dynamics/__init__.py:73:    "CDGridShallowWaterModel": ("legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid", "CDGridShallowWaterModel"),
src/legoesm/atmosphere/dynamics/__init__.py:74:    "CDGridShallowWaterConfig": ("legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid", "CDGridShallowWaterConfig"),
src/legoesm/atmosphere/dynamics/__init__.py:76:    "CDGridShallowWaterState": ("legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid", "CDGridShallowWaterState"),
src/legoesm/atmosphere/dynamics/__init__.py:78:    "CDGridPrimitiveEquationModel": ("legoesm.atmosphere.dynamics.primitive_eq_cdgrid", "CDGridPrimitiveEquationModel"),
src/legoesm/atmosphere/dynamics/__init__.py:79:    "CDGridPrimitiveEquationConfig": ("legoesm.atmosphere.dynamics.primitive_eq_cdgrid", "CDGridPrimitiveEquationConfig"),
src/legoesm/atmosphere/dynamics/__init__.py:84:    "CDGridCompressibleEulerModel": ("legoesm.atmosphere.dynamics.compressible_euler_cdgrid", "CDGridCompressibleEulerModel"),
src/legoesm/atmosphere/dynamics/__init__.py:85:    "CDGridCompressibleEulerConfig": ("legoesm.atmosphere.dynamics.compressible_euler_cdgrid", "CDGridCompressibleEulerConfig"),
src/legoesm/atmosphere/dynamics/__init__.py:143:    "ShallowWaterModel": "CDGridShallowWaterModel",
src/legoesm/atmosphere/dynamics/__init__.py:144:    "PrimitiveEquationModel": "CDGridPrimitiveEquationModel",
src/legoesm/atmosphere/dynamics/__init__.py:145:    "PrimitiveEquationConfig": "CDGridPrimitiveEquationConfig",
src/legoesm/atmosphere/dynamics/__init__.py:146:    "CompressibleEulerModel": "CDGridCompressibleEulerModel",
src/legoesm/atmosphere/dynamics/__init__.py:147:    "FVShallowWaterModel": "CDGridShallowWaterModel",
src/legoesm/atmosphere/dynamics/__init__.py:148:    "FVPrimitiveEquationModel": "CDGridPrimitiveEquationModel",
src/legoesm/atmosphere/dynamics/__init__.py:149:    "FVCompressibleEulerModel": "CDGridCompressibleEulerModel",
src/legoesm/atmosphere/dynamics/__init__.py:150:    "FVPrimitiveEquationConfig": "CDGridPrimitiveEquationConfig",
src/legoesm/atmosphere/dynamics/__init__.py:151:    "FVCompressibleEulerConfig": "CDGridCompressibleEulerConfig",
src/legoesm/atmosphere/dynamics/__init__.py:152:    "CGShallowWaterCubedModel": "CDGridShallowWaterModel",
src/legoesm/atmosphere/dynamics/__init__.py:153:    "CGPrimitiveEquationModel": "CDGridPrimitiveEquationModel",
src/legoesm/atmosphere/dynamics/__init__.py:154:    "CGCompressibleEulerModel": "CDGridCompressibleEulerModel",
src/legoesm/atmosphere/dynamics/__init__.py:450:        "cdgrid_shallow_water": "CDGridShallowWaterModel",
src/legoesm/atmosphere/dynamics/__init__.py:451:        "cdgrid_primitive_equations": "CDGridPrimitiveEquationModel",
src/legoesm/atmosphere/dynamics/__init__.py:452:        "cdgrid_compressible_euler": "CDGridCompressibleEulerModel",
src/legoesm/grids/cubed_sphere.py:729:    CubedSphereGrid or (CubedSphereGrid, CubedSphereCDGrid)
src/legoesm/grids/duogrid.py:1097:    Uses exact D-grid edge angles from CubedSphereCDGrid (not averaged
src/legoesm/grids/__init__.py:6:    CubedSphereCDGrid,

web search: 
web search: mass flux plume parameterization temperature tendency should conserve tracer
exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py | sed -n '200,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   200	        self.grid = grid
   201	        self.z_coord = z_coord
   202	        self.config = config or LatLonCGridOceanConfig()
   203	        self._validate_config(self.config)
   204	        self._cfl_checked = False
   205	
   206	        if self.config.physics is not None:
   207	            from legoesm.ocean.physics.combined import make_ocean_physics
   208	            self._physics_fn = make_ocean_physics(self.config.physics)
   209	        else:
   210	            self._physics_fn = None
   211	
   212	    @staticmethod
   213	    def _validate_config(config: LatLonCGridOceanConfig) -> None:
   214	        """Validate configuration ranges."""
   215	        nonnegative = {
   216	            "A_h": config.A_h,
   217	            "B_h": config.B_h,
   218	            "K_h": config.K_h,
   219	            "A_v": config.A_v,
   220	            "K_v": config.K_v,
   221	            "hyperdiff_coeff": config.hyperdiff_coeff,
   222	            "barotropic_diffusion_alpha": config.barotropic_diffusion_alpha,
   223	        }
   224	        for name, value in nonnegative.items():
   225	            if value < 0.0:
   226	                raise ValueError(f"{name} must be >= 0, got {value!r}")
   227	
   228	        if config.n_barotropic_substeps < 1:
   229	            raise ValueError(
   230	                f"n_barotropic_substeps must be >= 1, got "
   231	                f"{config.n_barotropic_substeps!r}",
   232	            )
   233	        if config.barotropic_diffusion_dt_ref <= 0.0:
   234	            raise ValueError(
   235	                f"barotropic_diffusion_dt_ref must be > 0, got "
   236	                f"{config.barotropic_diffusion_dt_ref!r}",
   237	            )
   238	        if config.min_water_column_m <= 0.0:
   239	            raise ValueError(
   240	                f"min_water_column_m must be > 0, got "
   241	                f"{config.min_water_column_m!r}",
   242	            )
   243	        _valid_fw = {"none", "virtual_salt_flux"}
   244	        if config.freshwater_closure not in _valid_fw:
   245	            raise ValueError(
   246	                f"freshwater_closure must be one of {_valid_fw}, "
   247	                f"got {config.freshwater_closure!r}",
   248	            )
   249	        if config.max_abs_eta_m <= 0.0:
   250	            raise ValueError(
   251	                f"max_abs_eta_m must be > 0, got {config.max_abs_eta_m!r}")
   252	        if config.temperature_min_c >= config.temperature_max_c:
   253	            raise ValueError(
   254	                f"temperature_min_c ({config.temperature_min_c}) must be "
   255	                f"< temperature_max_c ({config.temperature_max_c})")
   256	        if config.salinity_min_psu >= config.salinity_max_psu:
   257	            raise ValueError(
   258	                f"salinity_min_psu ({config.salinity_min_psu}) must be "
   259	                f"< salinity_max_psu ({config.salinity_max_psu})")
   260	        _valid_solvers = {"explicit_substep", "implicit_cn"}

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/dynamics/ocean_model_mpas.py | sed -n '140,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   140	            )
   141	
   142	        # Precompute upwind-of-upwind cell indices for TVD advection.
   143	        # This is a one-time mesh topology operation stored as static data.
   144	        if self.config.tracer_advection == "tvd":
   145	            self._upup_pos, self._upup_neg = compute_upup_cells(mesh)
   146	        else:
   147	            self._upup_pos = None
   148	            self._upup_neg = None
   149	
   150	        if self.config.physics is not None:
   151	            from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
   152	            self._physics_fn = make_mpas_ocean_physics(self.config.physics)
   153	        else:
   154	            self._physics_fn = None
   155	
   156	    def check_barotropic_cfl(self, dt: float) -> float:
   157	        """Check barotropic CFL and warn if marginal or unstable.
   158	
   159	        Parameters
   160	        ----------
   161	        dt : float
   162	            Baroclinic timestep [s].
   163	
   164	        Returns
   165	        -------
   166	        cfl : float
   167	            Barotropic CFL number.
   168	        """
   169	        import math
   170	        import warnings
   171	
   172	        g = self.config.g
   173	        H_max = self.z_coord.H_max
   174	        n_sub = self.config.n_barotropic_substeps
   175	        dx_min = float(jnp.min(self.mesh.dcEdge))
   176	
   177	        c_baro = math.sqrt(g * H_max)
   178	        dt_baro = dt / n_sub
   179	        cfl = c_baro * dt_baro / dx_min
   180	
   181	        if cfl > 0.8:
   182	            n_min = math.ceil(c_baro * dt / (0.8 * dx_min))
   183	            warnings.warn(
   184	                f"Barotropic CFL = {cfl:.2f} (> 0.8) — may be unstable. "
   185	                f"c_baro={c_baro:.1f} m/s, dx_min={dx_min:.0f} m, "
   186	                f"dt_baro={dt_baro:.1f} s. "
   187	                f"Suggest n_barotropic_substeps >= {n_min} "
   188	                f"(currently {n_sub}).",
   189	                stacklevel=2,
   190	            )
   191	        return cfl
   192	
   193	    def tendencies(
   194	        self,
   195	        state: MPASOceanState,
   196	        freshwater: FreshwaterForcing | None = None,
   197	        surface_forcing=None,
   198	        sponge=None,
   199	    ) -> MPASOceanTendencies:
   200	        """Compute baroclinic tendencies."""
   201	        return mpas_ocean_baroclinic_tendencies(
   202	            state, self.mesh, self.z_coord, self.config,
   203	            freshwater=freshwater,
   204	            physics_fn=self._physics_fn,
   205	            surface_forcing=surface_forcing,
   206	            sponge=sponge,
   207	        )
   208	
   209	    @partial(jax.jit, static_argnums=(0,))
   210	    def step(
   211	        self,
   212	        state: MPASOceanState,
   213	        dt: float,
   214	        freshwater: FreshwaterForcing | None = None,
   215	        surface_forcing=None,
   216	        sponge=None,
   217	    ) -> MPASOceanState:
   218	        """Advance one full timestep (baroclinic + barotropic).
   219	
   220	        Parameters

exec
/bin/zsh -lc 'rg -n "_validate_config|n_barotropic_substeps must" src/legoesm/ocean/dynamics/ocean_model_mpas.py src/legoesm/ocean/mpas_config.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 exited 1 in 0ms:
exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/mpas_config.py | sed -n '1,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Configuration for MPAS Voronoi-mesh ocean model.
     2	
     3	Provides config NamedTuples for the full 3D MPAS ocean dynamics
     4	and for simplified (slab/fixed) ocean modes on Voronoi meshes.
     5	"""
     6	
     7	from __future__ import annotations
     8	
     9	from typing import NamedTuple
    10	
    11	
    12	class MPASOceanConfig(NamedTuple):
    13	    """Configuration for MPAS ocean primitive equation solver.
    14	
    15	    Fields
    16	    ------
    17	    g : float
    18	        Gravitational acceleration [m/s²].
    19	    rho_0 : float
    20	        Reference seawater density [kg/m³].
    21	    A_h : float
    22	        Horizontal (harmonic) viscosity [m²/s].
    23	    B_h : float
    24	        Horizontal biharmonic viscosity [m⁴/s].  Applied as a constant-
    25	        coefficient ``B_h * del4(u)`` operator.
    26	    C_smag : float
    27	        Smagorinsky coefficient (dimensionless, typical 0.01-0.15).
    28	        When > 0, enables flow-dependent biharmonic Smagorinsky viscosity
    29	        ``-del2(A_smag * del2(u))`` where ``A_smag = (C_smag * Δ)² |D|``.
    30	    bottom_drag_r : float
    31	        Linear bottom drag coefficient [m/s].  Applied as ``-r * u / dz_bot``
    32	        at the bottom level in the baroclinic tendency and as
    33	        ``-r * U_bar / H`` in the barotropic substeps.  Matches MITgcm's
    34	        ``bottomDragLinear`` convention.
    35	    K_h : float
    36	        Horizontal tracer diffusivity [m²/s].
    37	    K_bih : float
    38	        Horizontal biharmonic tracer diffusivity [m⁴/s].  Applied as
    39	        ``-K_bih * ∇⁴(tr)`` on cell-centered tracers using the scalar
    40	        bilaplacian ``bilaplacian_cell_3d`` (two applications of the
    41	        cell-centered scalar Laplacian ``div(grad(·))``).  Scale-selective
    42	        dissipation with weaker damping of large scales than ``K_h``.
    43	    A_v : float
    44	        Vertical viscosity [m²/s].
    45	    K_v : float
    46	        Vertical tracer diffusivity [m²/s].
    47	    n_barotropic_substeps : int
    48	        Number of barotropic (free-surface) substeps per baroclinic step.
    49	    pv_scheme : str
    50	        PV flux scheme: "energy" or "enstrophy".
    51	    use_conservation_fixer : bool
    52	        Apply conservation fixers after each step.
    53	    fix_volume : bool
    54	        Fix volume (eta) conservation.
    55	    fix_heat : bool
    56	        Fix heat (T) conservation.
    57	    fix_salt : bool
    58	        Fix salt (S) conservation.
    59	    min_water_column_m : float
    60	        Minimum allowed water column depth [m].
    61	    barotropic_damping : float
    62	        Rayleigh damping coefficient for barotropic mode [1/s].
    63	    freshwater_closure : str
    64	        Freshwater closure: "none", "virtual_salt_flux", or "real_freshwater".
    65	        "virtual_salt_flux": apply virtual salt flux to S, real mass flux to eta.
    66	        "real_freshwater": apply real freshwater mass to eta and dilution to S.
    67	        "none": ignore freshwater forcing.
    68	    S_ref : float
    69	        Reference salinity [PSU] for virtual salt flux.
    70	    tracer_advection : str
    71	        Tracer advection scheme: "upwind" or "tvd".
    72	        "upwind" uses first-order donor-cell reconstruction.
    73	        "tvd" uses second-order Van Leer limiter (less diffusive,
    74	        monotone) for both horizontal and vertical advection.
    75	    """
    76	    g: float = 9.80616           # = constants.g
    77	    rho_0: float = 1025.0        # = eos.rho_0
    78	    A_h: float = 1.0e4
    79	    B_h: float = 0.0
    80	    C_smag: float = 0.0
    81	    bottom_drag_r: float = 0.0
    82	    K_h: float = 0.0
    83	    K_bih: float = 0.0
    84	    A_v: float = 1.0e-3
    85	    K_v: float = 1.0e-4
    86	    n_barotropic_substeps: int = 30
    87	    # Default to enstrophy-conserving PV flux — avoids the ζ-checkerboard
    88	    # null mode of the energy-conserving scheme (Ringler et al. 2010).
    89	    # Use "energy" if total-KE conservation is required and the ζ null
    90	    # mode can be controlled by other means.
    91	    pv_scheme: str = "enstrophy"
    92	    apvm_dt: float = 0.0  # APVM damping timescale [s]; set to baroclinic dt
    93	                          # to enable the Anticipated PV Method upstream
    94	                          # bias (damps ζ-checkerboard null mode of the
    95	                          # energy-conserving PV flux). 0 = disabled.
    96	                          # Set automatically by the test matrix to dt.
    97	    pv_alpha: float = 1.0  # Weight on energy-conserving flux when
    98	                           # ``pv_scheme == "mixed"``: α·energy + (1−α)·enstrophy.
    99	                           # α=1.0 recovers pure energy; α=0.0 pure enstrophy.
   100	                           # Ignored for pv_scheme in {"energy", "enstrophy"}.
   101	    K_zeta_bih: float = 0.0  # Biharmonic dissipation on relative vorticity ζ
   102	                             # [m⁴/s]. Adds −K_ζ·∇⁴ζ to the vorticity
   103	                             # equation (scale-selective damping of grid-scale
   104	                             # ζ patterns), applied as a tangential-gradient
   105	                             # force on the momentum equation. Targets the
   106	                             # ζ-checkerboard null mode of the energy-
   107	                             # conserving PV flux — invisible to B_h·∇⁴u
   108	                             # because the null mode lives in the kernel of
   109	                             # the discrete curl. 0 = disabled.
   110	    use_conservation_fixer: bool = False
   111	    fix_volume: bool = True
   112	    fix_heat: bool = True
   113	    fix_salt: bool = True
   114	    min_water_column_m: float = 0.5
   115	    barotropic_damping: float = 0.0
   116	    barotropic_diffusion_alpha: float = 0.01
   117	    barotropic_diffusion_dt_ref: float = 60.0
   118	    barotropic_div_damp: float = 0.0  # Divergence damping on barotropic velocity (dimensionless)
   119	    barotropic_u_viscosity: float = 0.0  # Lateral viscosity on u_bar [m²/s].
   120	                                         # Damps the TRiSK rotational null branch
   121	                                         # (Thuburn 2008; Ringler et al. 2010) on
   122	                                         # hexagonal C-grids — invisible to eta
   123	                                         # diffusion and divergence damping.
   124	                                         # 0 = disabled; typical 1e3-1e4 m²/s
   125	                                         # for global ico4 (~460 km) meshes.
   126	    bebt: float = 0.2               # Semi-implicit barotropic PGF [0,1]. 0=forward-backward, 0.2=MOM6 default.
   127	    maxvel_barotropic: float = 0.0  # Velocity clipping [m/s]. 0=disabled.
   128	    barotropic_time_filter: str = "cosine"  # "box" or "cosine"
   129	    semi_implicit_coriolis: bool = True
   130	    # Barotropic solver selection.  ``"explicit_substep"`` (default) uses
   131	    # the existing forward-backward substep loop with cosine filter.
   132	    # ``"implicit_cn"`` uses a single-step Crank-Nicolson free surface
   133	    # with PCG Helmholtz solve, mirroring the lat-lon implementation
   134	    # (docs/issues/barotropic_mode_noise.md).  Eliminates the TRiSK
   135	    # rotational null branch (Thuburn 2008; Ringler+ 2010 §6) by
   136	    # construction; no substepping or time filter needed.
   137	    barotropic_solver: str = "explicit_substep"
   138	    # Implicit-CN knobs (only used when ``barotropic_solver = 'implicit_cn'``).
   139	    # 0.5 = pure Crank-Nicolson (2nd-order, no implicit damping); 1.0 =
   140	    # fully backward (1st-order, max damping).  0.55 is the standard
   141	    # MITgcm/MPAS-O choice — slightly past CN to suppress chequerboard
   142	    # while staying close to 2nd-order in time.
   143	    barotropic_implicit_theta_eta: float = 0.55
   144	    barotropic_implicit_theta_pgf: float = 0.55
   145	    barotropic_implicit_pcg_tol: float = 1.0e-10
   146	    barotropic_implicit_pcg_maxiter: int = 200
   147	    freshwater_closure: str = "virtual_salt_flux"
   148	    S_ref: float = 35.0
   149	    physics: object = None  # OceanPhysicsConfig or None
   150	    eos: str = "wright"    # "wright" or "linear"
   151	    eos_linear: object = None  # LinearEOSConfig when eos="linear"
   152	    gm_redi: object = None  # GMRediConfig — None disables GM/Redi.  Only the
   153	                            # 'centered' slope_scheme is implemented on MPAS;
   154	                            # 'triads' raises NotImplementedError (Phase 5
   155	                            # of docs/ocean_experiments/gm_redi_mpas_plan.md).
   156	    tracer_advection: str = "upwind"
   157	    # Runtime bounds checks (matching cubed-sphere ocean)
   158	    enable_runtime_checks: bool = False
   159	    temperature_min_c: float = -5.0
   160	    temperature_max_c: float = 45.0
   161	    salinity_min_psu: float = 0.0
   162	    salinity_max_psu: float = 50.0
   163	    # Leith viscosity coefficient (Leith 1996).  When > 0 enables
   164	    # flow-adaptive biharmonic viscosity ``-∇²(A_L ∇²u)`` with
   165	    # ``A_L = (C_L · Δ_e)³ · |∇ζ|`` (or ``sqrt(|∇ζ|² + |∇δ|²)`` when
   166	    # ``C_leith_modified = True``).  Typical values: 1.0–2.0.  Historically
   167	    # appended at the END of the NamedTuple for positional-call stability.
   168	    # Note: all in-tree constructors use keyword arguments and the
   169	    # ``K_bih`` field inserted above deliberately mirrors the field order
   170	    # of :class:`LatLonCGridOceanConfig`; positional stability is not
   171	    # relied on at MPAS call sites.
   172	    C_leith: float = 0.0
   173	    C_leith_modified: bool = False
   174	
   175	
   176	class MPASSimpleOceanConfig(NamedTuple):
   177	    """Configuration for simplified ocean on Voronoi mesh.
   178	
   179	    Mirrors SimpleOceanConfig but operates on (nCells,) arrays.
   180	
   181	    Fields
   182	    ------
   183	    mode : str
   184	        Ocean mode: "fixed", "slab", or "two_layer".
   185	    sst_constant : float
   186	        Fixed SST [K] (for "fixed" mode).
   187	    h_mix : float
   188	        Mixed layer depth [m].
   189	    rho_ocean : float
   190	        Seawater density [kg/m³].
   191	    c_ocean : float
   192	        Specific heat of seawater [J/(kg·K)].
   193	    Q_flux : float
   194	        Prescribed ocean heat flux [W/m²].
   195	    albedo_ocean : float
   196	        Ocean albedo.
   197	    emissivity_ocean : float
   198	        Ocean emissivity.
   199	    Cd_ocean : float
   200	        Drag coefficient over ocean.
   201	    Ch_ocean : float
   202	        Heat transfer coefficient over ocean.
   203	    U_min : float
   204	        Minimum wind speed [m/s].
   205	    T_freeze : float
   206	        Freezing temperature [K].
   207	    h_deep : float
   208	        Deep layer depth [m] (for "two_layer" mode).
   209	    k_mix : float
   210	        Vertical mixing coefficient [m²/s] (for "two_layer" mode).
   211	    restore_deep : bool
   212	        Restore deep layer toward reference (for "two_layer" mode).
   213	    T_deep_ref : float
   214	        Deep layer reference temperature [K].
   215	    tau_deep : float
   216	        Deep layer restoring timescale [s].
   217	    """
   218	    mode: str = "fixed"
   219	    sst_constant: float = 300.0
   220	    h_mix: float = 50.0

exec
/bin/zsh -lc "PYTHONPATH=src python3 - <<'PY'
import jax.numpy as jnp
from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.vertical import create_ocean_z_star

cfg=KPPConfig(K_bg=1e-5)
z=create_ocean_z_star(n_levels=5,H_max=1000,dz_surface=100,dz_deep=100)
shape=(1,1,1,5)
# column with neutral strat + no shear -> interior should hit background
u=v=jnp.zeros(shape,dtype=jnp.float32)
T=jnp.zeros(shape,dtype=jnp.float32)
S=jnp.full(shape,35.0,dtype=jnp.float32)
rho=jnp.array([[[[1025,1025.1,1025.2,1025.3,1025.4]]]],dtype=jnp.float32)
eta=jnp.zeros((1,1,1),dtype=jnp.float32)
J=jnp.ones((1,1,1),dtype=jnp.float32)
out=kpp_vertical_mixing(u,v,T,S,rho,eta,z,J,cfg,B_f=jnp.full((1,1,1),-1e-8))
print('cfg.K_bg=',cfg.K_bg)
print('min K_v=',float(jnp.min(out.K_v)))
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 2179ms:
cfg.K_bg= 1e-05
min K_v= 1.9999999494757503e-05

exec
/bin/zsh -lc "PYTHONPATH=src python3 - <<'PY'
import jax.numpy as jnp
from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.vertical import create_ocean_z_star

z=create_ocean_z_star(n_levels=6,H_max=600,dz_surface=100,dz_deep=100)
shape=(1,1,1,6)
u=v=jnp.zeros(shape,dtype=jnp.float32)
T=jnp.linspace(10,8,6,dtype=jnp.float32).reshape(shape)
S=jnp.full(shape,35.0,dtype=jnp.float32)
rho=jnp.linspace(1026,1028,6,dtype=jnp.float32).reshape(shape)
eta=jnp.zeros((1,1,1),dtype=jnp.float32)
J=jnp.ones((1,1,1),dtype=jnp.float32)
cfg=KPPConfig(K_bg=0.0,A_bg=0.0,K_0_shear=0.0,K_conv=0.0)
Bf=jnp.full((1,1,1),1e-7,dtype=jnp.float32)
for Q in (-1e-5,1e-5):
    out=kpp_vertical_mixing(u,v,T,S,rho,eta,z,J,cfg,B_f=Bf,Q_sfc_T=jnp.full((1,1,1),Q,dtype=jnp.float32),Q_sfc_S=jnp.zeros((1,1,1),dtype=jnp.float32))
    print('Q',Q,'top dT',float(out.dT_dt[0,0,0,0]),'next',float(out.dT_dt[0,0,0,1]))
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 1544ms:
Q -1e-05 top dT -3.4161164421675494e-06 next 3.4623749343154486e-06
Q 1e-05 top dT -3.508633426463348e-06 next 3.4623749343154486e-06

exec
/bin/zsh -lc 'rg -n "class OceanSurfaceForcing|tau_x|tau_y|Q_net|E_minus_P|sw_down" src/legoesm/ocean/state.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
66:class OceanSurfaceForcing(NamedTuple):
75:    sw_down : array or None
80:    tau_x, tau_y : array or None
85:    sw_down: object = None       # jnp.ndarray | None
87:    tau_x: object = None         # jnp.ndarray | None
88:    tau_y: object = None         # jnp.ndarray | None

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/state.py | sed -n '50,110p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
    50	
    51	class OceanTendencies(NamedTuple):
    52	    """Tendencies for the ocean primitive equations.
    53	
    54	    Same structure as OceanState. Static fields (H_bathy, land_mask)
    55	    have zero tendencies, matching the phis pattern in the atmosphere.
    56	    """
    57	    du_dt: Field
    58	    dv_dt: Field
    59	    dT_dt: Field
    60	    dS_dt: Field
    61	    deta_dt: Field
    62	    dH_bathy_dt: Field
    63	    dland_mask_dt: Field
    64	
    65	
    66	class OceanSurfaceForcing(NamedTuple):
    67	    """External atmospheric/surface forcing data for ocean physics.
    68	
    69	    Carries coupler-provided fields into the ocean physics pipeline.
    70	    All fields are optional (None means not available).  Shape of 2D
    71	    fields matches the horizontal grid; 3D fields add a level axis.
    72	
    73	    Fields
    74	    ------
    75	    sw_down : array or None
    76	        Downwelling shortwave at sea surface [W/m²].  Needed for
    77	        subsurface SW penetration heating.
    78	    q_net : array or None
    79	        Net surface heat flux (positive into ocean) [W/m²].
    80	    tau_x, tau_y : array or None
    81	        Surface wind stress components [Pa].
    82	    freshwater : array or None
    83	        Net freshwater flux into ocean (P - E + R + M) [kg/m²/s].
    84	    """
    85	    sw_down: object = None       # jnp.ndarray | None
    86	    q_net: object = None         # jnp.ndarray | None
    87	    tau_x: object = None         # jnp.ndarray | None
    88	    tau_y: object = None         # jnp.ndarray | None
    89	    freshwater: object = None    # jnp.ndarray | None
    90	
    91	
    92	class OceanConfig(NamedTuple):
    93	    """Configuration for the ocean model."""
    94	    g: float = 9.80616           # = constants.g
    95	    rho_0: float = 1025.0        # = eos.rho_0
    96	    A_h: float = 1.0e4           # Horizontal viscosity [m^2/s]
    97	    K_h: float = 0.0           # Horizontal tracer diffusivity [m^2/s]
    98	    A_v: float = 1.0e-3          # Vertical viscosity [m^2/s]
    99	    K_v: float = 1.0e-4          # Vertical tracer diffusivity [m^2/s]
   100	    n_barotropic_substeps: int = 30
   101	    hyperdiff_coeff: float = 0.0
   102	    use_conservation_fixer: bool = False
   103	    fix_volume: bool = True
   104	    fix_heat: bool = True
   105	    fix_salt: bool = True
   106	    # 2-D Laplacian damping used in barotropic subcycling.
   107	    # Per-substep coefficient is alpha * (dt_s / dt_ref) * area * laplacian(...),
   108	    # so the damping is explicitly dt-scaled and tunable.
   109	    # Cubed-sphere default is higher (0.05) than other grids (0.01)
   110	    # to suppress the face-boundary feedback instability.  See #100.

web search: 
web search: 
exec
/bin/zsh -lc "rg -n \"class OceanState\" src/legoesm/ocean/state.py && nl -ba src/legoesm/ocean/state.py | sed -n '1,55p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
19:class OceanState(NamedTuple):
     1	"""Ocean state containers.
     2	
     3	All states are registered as JAX pytrees via NamedTuple + Field,
     4	consistent with ShallowWaterState, HydrostaticState, and
     5	NonHydrostaticState in core/state.py.
     6	"""
     7	
     8	from __future__ import annotations
     9	
    10	from typing import NamedTuple
    11	
    12	from legoesm.core.field import Field
    13	
    14	
    15	# ==============================================================================
    16	# FV Ocean State (cubed-sphere)
    17	# ==============================================================================
    18	
    19	class OceanState(NamedTuple):
    20	    """State for the ocean primitive equations on the cubed-sphere.
    21	
    22	    3D fields: shape (6, n, n, nlev).
    23	    2D fields: shape (6, n, n).
    24	
    25	    Fields
    26	    ------
    27	    u : Field
    28	        Zonal velocity [m/s]. Prognostic. Shape (6, n, n, nlev).
    29	    v : Field
    30	        Meridional velocity [m/s]. Prognostic. Shape (6, n, n, nlev).
    31	    T : Field
    32	        Potential temperature [degC]. Prognostic. Shape (6, n, n, nlev).
    33	    S : Field
    34	        Salinity [PSU]. Prognostic. Shape (6, n, n, nlev).
    35	    eta : Field
    36	        Sea surface height [m]. Prognostic. Shape (6, n, n).
    37	    H_bathy : Field
    38	        Bathymetry depth [m]. Static (positive downward). Shape (6, n, n).
    39	    land_mask : Field
    40	        Ocean mask. Static. 1=ocean, 0=land. Shape (6, n, n).
    41	    """
    42	    u: Field
    43	    v: Field
    44	    T: Field
    45	    S: Field
    46	    eta: Field
    47	    H_bathy: Field
    48	    land_mask: Field
    49	
    50	
    51	class OceanTendencies(NamedTuple):
    52	    """Tendencies for the ocean primitive equations.
    53	
    54	    Same structure as OceanState. Static fields (H_bathy, land_mask)
    55	    have zero tendencies, matching the phis pattern in the atmosphere.

exec
/bin/zsh -lc "nl -ba src/legoesm/core/field.py | sed -n '1,110p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Field: A coordinate-aware array registered as a JAX pytree.
     2	
     3	The Field is the fundamental data container in legoESM. It wraps a JAX array
     4	with metadata (name, dimensions, units, staggering) and is registered as a JAX
     5	pytree so that jit, grad, vmap, and scan all work transparently.
     6	"""
     7	
     8	from __future__ import annotations
     9	
    10	from typing import Any
    11	
    12	import jax
    13	import jax.numpy as jnp
    14	
    15	
    16	class Field:
    17	    """A coordinate-aware JAX array.
    18	
    19	    Fields carry metadata alongside their numerical data, enabling:
    20	    - Automatic dimension tracking through operations
    21	    - Unit-aware diagnostics and I/O
    22	    - Staggering information for C-grid operators
    23	
    24	    As a JAX pytree, Field works seamlessly with all JAX transformations:
    25	    jit, grad, vmap, scan, checkpoint, etc.
    26	
    27	    Parameters
    28	    ----------
    29	    data : jax.Array
    30	        The numerical data.
    31	    name : str
    32	        Variable name (e.g., "potential_temperature").
    33	    dims : tuple of str
    34	        Dimension names (e.g., ("face", "x", "y")).
    35	    units : str
    36	        Physical units (e.g., "K", "m/s").
    37	    long_name : str, optional
    38	        Human-readable description.
    39	    staggering : str, optional
    40	        Grid staggering: "cell", "edge", or "vertex". Default "cell".
    41	    """
    42	
    43	    __slots__ = ("data", "name", "dims", "units", "long_name", "staggering")
    44	
    45	    def __init__(
    46	        self,
    47	        data: jax.Array,
    48	        name: str = "",
    49	        dims: tuple[str, ...] = (),
    50	        units: str = "",
    51	        long_name: str = "",
    52	        staggering: str = "cell",
    53	    ):
    54	        object.__setattr__(self, "data", data)
    55	        object.__setattr__(self, "name", name)
    56	        object.__setattr__(self, "dims", dims)
    57	        object.__setattr__(self, "units", units)
    58	        object.__setattr__(self, "long_name", long_name)
    59	        object.__setattr__(self, "staggering", staggering)
    60	
    61	    def __setattr__(self, key: str, value: Any) -> None:
    62	        raise AttributeError("Field is immutable. Use .replace() to create a new Field.")
    63	
    64	    # ---- JAX pytree registration ----
    65	
    66	    def tree_flatten(self):
    67	        """Flatten for JAX: data is the dynamic leaf, metadata is static.
    68	
    69	        All metadata is part of aux_data (static). For tree_map to work
    70	        between two pytrees, their aux_data must match exactly. Ensure that
    71	        tendencies returned by physics modules use state.field.replace(data=...)
    72	        to preserve metadata compatibility.
    73	        """
    74	        children = (self.data,)
    75	        aux_data = (self.name, self.dims, self.units, self.long_name, self.staggering)
    76	        return children, aux_data
    77	
    78	    @classmethod
    79	    def tree_unflatten(cls, aux_data, children):
    80	        """Reconstruct Field from flattened representation."""
    81	        (data,) = children
    82	        name, dims, units, long_name, staggering = aux_data
    83	        return cls(data=data, name=name, dims=dims, units=units,
    84	                   long_name=long_name, staggering=staggering)
    85	
    86	    # ---- Convenience methods ----
    87	
    88	    def replace(self, **kwargs) -> Field:
    89	        """Create a new Field with some attributes replaced."""
    90	        return Field(
    91	            data=kwargs.get("data", self.data),
    92	            name=kwargs.get("name", self.name),
    93	            dims=kwargs.get("dims", self.dims),
    94	            units=kwargs.get("units", self.units),
    95	            long_name=kwargs.get("long_name", self.long_name),
    96	            staggering=kwargs.get("staggering", self.staggering),
    97	        )
    98	
    99	    @property
   100	    def shape(self) -> tuple[int, ...]:
   101	        return self.data.shape
   102	
   103	    @property
   104	    def dtype(self):
   105	        return self.data.dtype
   106	
   107	    @property
   108	    def ndim(self) -> int:
   109	        return self.data.ndim
   110	

exec
/bin/zsh -lc "PYTHONPATH=src python3 - <<'PY'
import jax.numpy as jnp
from legoesm.ocean.physics.vertical_mixing.integration import make_vertical_mixing_physics
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.state import OceanState, OceanSurfaceForcing
from legoesm.core.field import Field

n=3;nlev=5
shape=(6,n,n,nlev)
shape2=(6,n,n)
state=OceanState(
    u=Field(jnp.zeros(shape,dtype=jnp.float32),name='u'),
    v=Field(jnp.zeros(shape,dtype=jnp.float32),name='v'),
    T=Field(jnp.ones(shape,dtype=jnp.float32)*10,name='T'),
    S=Field(jnp.ones(shape,dtype=jnp.float32)*35,name='S'),
    eta=Field(jnp.zeros(shape2,dtype=jnp.float32),name='eta'),
    H_bathy=Field(jnp.ones(shape2,dtype=jnp.float32)*5000,name='H_bathy'),
    land_mask=Field(jnp.ones(shape2,dtype=jnp.float32),name='mask'),
)
z=create_ocean_z_star(n_levels=nlev,H_max=5000)
fn=make_vertical_mixing_physics(VerticalMixingConfig(scheme='kpp'))
out0=fn(state,None,z,None)
forcing=OceanSurfaceForcing(q_net=jnp.ones(shape2,dtype=jnp.float32)*200,tau_x=jnp.ones(shape2,dtype=jnp.float32),tau_y=jnp.zeros(shape2,dtype=jnp.float32))
out1=fn(state,None,z,forcing)
print('max |du diff|',float(jnp.max(jnp.abs(out0.du_dt.data-out1.du_dt.data))))
print('max |dT diff|',float(jnp.max(jnp.abs(out0.dT_dt.data-out1.dT_dt.data))))
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 2710ms:
max |du diff| 0.0
max |dT diff| 0.0

web search: 
web search: 
exec
/bin/zsh -lc "PYTHONPATH=src python3 - <<'PY'
# quick consistency check for NPZD alkalinity sign implementation
import jax.numpy as jnp
from legoesm.ocean.biogeochemistry.npzd import npzd_source_sink
from legoesm.ocean.biogeochemistry.config import BiogeoConfig
cfg=BiogeoConfig(scheme='npzd',R_CaP=0.0,g_max=0.0,m_P=0.0,m_Z=0.0,remin_rate=0.0)
shape=(1,1)
NO3=jnp.array([[1e-2]])
P=jnp.array([[1e-3]])
Z=jnp.array([[1e-12]])
D=jnp.array([[1e-12]])
DIC=jnp.array([[2.1]])
ALK=jnp.array([[2.3]])
T=jnp.array([[20.]])
PAR=jnp.array([[200.]])
dz=jnp.array([10.])
dNO3,_,_,_,_,dALK=npzd_source_sink(NO3,P,Z,D,DIC,ALK,T,PAR,dz,cfg)
print('dNO3',float(dNO3[0,0]))
print('dALK',float(dALK[0,0]))
PY" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 691ms:
dNO3 -5.825560123184914e-08
dALK -5.825560123184914e-08

web search: 
web search: 
exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py | sed -n '390,402p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/dynamics/barotropic_common.py | sed -n '52,74p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   390	         Hu_sum_f, Hv_sum_f, eta_sum_f, U_sum_f, V_sum_f) = jax.lax.fori_loop(
   391	            0, n_substeps, fori_body, init_carry,
   392	        )
   393	
   394	    # Time-averaged barotropic transport (always box-filtered)
   395	    Hu_avg = Hu_sum_f / n_substeps
   396	    Hv_avg = Hv_sum_f / n_substeps
   397	
   398	    # Time-averaged eta and velocity (cosine or box filtered)
   399	    eta_avg = eta_sum_f / w_total
   400	    U_bar_avg = U_sum_f / w_total
   401	    V_bar_avg = V_sum_f / w_total
   402	

 succeeded in 0ms:
    52	        Per-substep weight passed as ``xs`` to ``lax.scan`` (or
    53	        indexed inside ``fori_loop``).
    54	    w_total : jax.Array, scalar
    55	        ``sum(w_filter)`` — used to normalise the eta / velocity
    56	        accumulators.  Transport accumulators (``Hu``) keep using
    57	        ``n_substeps`` for exact volume conservation.
    58	    """
    59	    i = jnp.arange(n_substeps, dtype=dtype)
    60	    if use_cosine:
    61	        w_filter = 1.0 + jnp.cos(
    62	            2.0 * jnp.pi * (i - 0.5 * n_substeps) / n_substeps,
    63	        )
    64	    else:
    65	        w_filter = jnp.ones(n_substeps, dtype=dtype)
    66	    return w_filter, jnp.sum(w_filter)
    67	
    68	
    69	def bebt_blend(
    70	    eta_new: jnp.ndarray,
    71	    eta_old: jnp.ndarray,
    72	    bebt: float | jnp.ndarray,
    73	) -> jnp.ndarray:
    74	    """Backward-Euler/Backward-time blend of new and old eta for the PGF.

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/dynamics/barotropic_mpas.py | sed -n '304,312p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   304	    )
   305	
   306	    # Time-averaged barotropic fields
   307	    Hu_avg = Hu_sum_f / n_substeps  # transport: always box-filtered
   308	    eta_avg = eta_sum_f / w_total   # eta/velocity: cosine or box filtered
   309	    u_bar_avg = ubar_sum_f / w_total
   310	
   311	    return eta_avg, u_bar_avg, Hu_avg
   312	

codex
I’ve completed the targeted execution checks. Next I’m consolidating findings by severity with exact line references and explicit `confirmed-bug` / `false-positive` / `ambiguous` labels for each concern (including your 4 questions).
exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/vertical_mixing/kpp.py | sed -n '276,304p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   276	    dv = v[..., :-1] - v[..., 1:]
   277	    S2 = (du**2 + dv**2) / jnp.maximum(dz_half**2, eps)
   278	    Ri_int = N2 / jnp.maximum(S2, eps)
   279	    # LMD94 interior shear instability: K = K_0 * (1 - (Ri/Ri_0)^2)^3
   280	    # for Ri < Ri_0, zero above.
   281	    Ri_ratio = jnp.clip(Ri_int / cfg.Ri_0, 0.0, 1.0)
   282	    K_interior = cfg.K_0_shear * (1.0 - Ri_ratio**2) ** 3 + cfg.K_bg
   283	
   284	    # Interior static instability: enhanced mixing where N2 < 0
   285	    K_conv = jnp.where(N2 < cfg.Ri_conv, cfg.K_conv, 0.0)
   286	    K_interior = K_interior + K_conv
   287	
   288	    # --- K at interfaces (average of full level K_bl) ---
   289	    K_bl_half = 0.5 * (K_bl_full[..., :-1] + K_bl_full[..., 1:])
   290	
   291	    # sigma at interfaces
   292	    z_half_depth = 0.5 * (z_depth[..., :-1] + z_depth[..., 1:])
   293	    sigma_half = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
   294	    in_bl = sigma_half < 1.0
   295	
   296	    # Combine BL and interior
   297	    K_v = jnp.where(in_bl, K_bl_half, K_interior) + cfg.K_bg
   298	    A_v = jnp.where(in_bl, K_bl_half * 1.0, K_interior) + cfg.A_bg
   299	    K_v = jnp.minimum(K_v, cfg.K_max)
   300	    A_v = jnp.minimum(A_v, cfg.K_max)
   301	
   302	    # --- Apply diffusion ---
   303	    vel = jnp.stack([u, v], axis=0)
   304	    vel_tend = jax.vmap(

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/freshwater.py | sed -n '185,199p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   185	    # Evaporation from latent heat flux: E = lhflx / L_v
   186	    evap = lhflx / L_v
   187	
   188	    # Land runoff (sum surface + subsurface).  Pin the zero-fallback
   189	    # dtype to the precip path so a missing runoff input does not
   190	    # silently widen the freshwater forcing struct to f64 under x64.
   191	    if runoff_surface is not None:
   192	        runoff = runoff_surface
   193	        if runoff_subsurface is not None:
   194	            runoff = runoff + runoff_subsurface
   195	    else:
   196	        runoff = jnp.zeros(nCells, dtype=precip.dtype)
   197	
   198	    # Ice freshwater: based on areal ice mass change.
   199	    # ice_mass = rho_ice * h * A  (per unit area of grid cell)

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/init_latlon_cgrid.py | sed -n '94,102p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
    94	    nlev = z_coord.n_levels
    95	
    96	    if land_mask_override is not None:
    97	        land_mask = jnp.asarray(land_mask_override)
    98	        H_bathy = jnp.full((n_lat, n_lon), H_max, dtype=jnp.float64)
    99	    else:
   100	        H_bathy, land_mask = idealized_bathymetry_latlon_cgrid(
   101	            grid, H_max, land_lat_threshold,
   102	        )

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/vertical_mixing/integration.py | sed -n '82,97p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
    82	def _make_kpp(config: VerticalMixingConfig) -> Callable:
    83	    from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
    84	    cfg = config.kpp
    85	
    86	    def physics_fn(state: OceanState, grid: CubedSphereGrid,
    87	                   z_coord: OceanZStarCoordinate,
    88	                   surface_forcing=None) -> OceanTendencies:
    89	        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    90	        rho = _compute_rho(state, z_coord, J)
    91	        out = kpp_vertical_mixing(
    92	            state.u.data, state.v.data, state.T.data, state.S.data,
    93	            rho, state.eta.data, z_coord, J, cfg,
    94	        )
    95	        return _wrap_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
    96	    return physics_fn
    97	

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/biogeochemistry/npzd.py | sed -n '184,191p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   184	
   185	    dDIC_dt = dDIC_bio - caco3_production + caco3_dissolution
   186	    # Alkalinity: -1 per mol NO3 consumed (nitrification sign convention)
   187	    # + 2 per mol CaCO3 dissolved, -2 per mol CaCO3 precipitated
   188	    dALK_dt = (-growth + remin + (1.0 - cfg.gamma_Z) * grazing
   189	               - 2.0 * caco3_production + 2.0 * caco3_dissolution)
   190	
   191	    return dNO3_dt, dPhyto_dt, dZoo_dt, dDet_dt, dDIC_dt, dALK_dt

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/vertical_mixing/kpp.py | sed -n '359,378p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   359	    # --- Temperature non-local tendency ---
   360	    # Non-local flux at interfaces: F_nl = C_s * Q_T * G_half  [K*m/s]
   361	    F_T = cfg.gamma_T * Q_T[..., jnp.newaxis] * G_half  # (..., nlev-1)
   362	    # Tendency = -dF/dz at full levels (zero-flux BCs at surface and bottom)
   363	    dT_nonlocal_top = -F_T[..., :1] / dz_actual[..., :1]
   364	    dT_nonlocal_int = (F_T[..., :-1] - F_T[..., 1:]) / dz_actual[..., 1:-1]
   365	    dT_nonlocal_bot = F_T[..., -1:] / dz_actual[..., -1:]
   366	    dT_nonlocal = jnp.concatenate(
   367	        [dT_nonlocal_top, dT_nonlocal_int, dT_nonlocal_bot], axis=-1
   368	    )  # (..., nlev)  [K/s]
   369	    dT_nonlocal = jnp.where(
   370	        in_bl_full & is_unstable_col[..., jnp.newaxis], dT_nonlocal, 0.0
   371	    )
   372	
   373	    # --- Salinity non-local tendency ---
   374	    F_S = cfg.gamma_S * Q_S[..., jnp.newaxis] * G_half  # (..., nlev-1)
   375	    dS_nonlocal_top = -F_S[..., :1] / dz_actual[..., :1]
   376	    dS_nonlocal_int = (F_S[..., :-1] - F_S[..., 1:]) / dz_actual[..., 1:-1]
   377	    dS_nonlocal_bot = F_S[..., -1:] / dz_actual[..., -1:]
   378	    dS_nonlocal = jnp.concatenate(

exec
/bin/zsh -lc "nl -ba src/legoesm/ocean/physics/convection/plume.py | sed -n '54,59p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
    54	
    55	        # Entrain environment
    56	        entrain = cfg.epsilon * dz_k
    57	        T_plume = (1.0 - entrain) * T_plume + entrain * T[..., k]
    58	        S_plume = (1.0 - entrain) * S_plume + entrain * S[..., k]
    59	

codex
1. `confirmed-bug` — NPZD alkalinity N-cycle sign is flipped for total alkalinity.  
Reference: [npzd.py:186](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/biogeochemistry/npzd.py:186), [npzd.py:188](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/biogeochemistry/npzd.py:188).  
Why: code makes `dALK_dt` track `dNO3_dt` (organic term), i.e. nitrate uptake lowers ALK. For Dickson TA (`... - [NO3-]`), this should be opposite sign (`-dNO3_dt`) for the N-cycle part.

2. `confirmed-bug` — KPP interior/background diffusivity is double-counted.  
Reference: [kpp.py:282](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:282), [kpp.py:297](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:297), [kpp.py:298](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:298).  
Why: `K_interior` already includes `+ cfg.K_bg`, then `K_v` adds `+ cfg.K_bg` again. This gives floor `2*K_bg` (I verified min `K_v=2e-5` for `K_bg=1e-5`).

3. `confirmed-bug` — `runoff_subsurface` is dropped when `runoff_surface` is `None`.  
Reference: [freshwater.py:191](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/freshwater.py:191), [freshwater.py:196](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/freshwater.py:196).  
Why: logic only adds subsurface runoff inside the `runoff_surface is not None` branch.

4. `confirmed-bug` — cosine barotropic filter degenerates for `n_substeps=1` and causes divide-by-zero in averages.  
Reference: [barotropic_common.py:60](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/dynamics/barotropic_common.py:60), [barotropic_common.py:66](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/dynamics/barotropic_common.py:66), [barotropic_latlon_cgrid.py:399](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:399), [barotropic_mpas.py:308](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/dynamics/barotropic_mpas.py:308).  
Why: for `n_substeps=1`, `w_filter=[0]`, `w_total=0`, then `eta_sum/w_total` and velocity averages divide by zero.

5. `confirmed-bug` — KPP integration path ignores provided surface forcing (`tau`, `q_net`, etc.).  
Reference: [vertical_mixing/integration.py:88](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/integration.py:88), [vertical_mixing/integration.py:91](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/integration.py:91), [kpp.py:150](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:150).  
Why: wrapper accepts `surface_forcing` but never passes any forcing-derived terms to `kpp_vertical_mixing`.

6. `confirmed-bug` — lat-lon rest-state override path hardcodes `float64` bathymetry against policy dtype.  
Reference: [init_latlon_cgrid.py:98](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/init_latlon_cgrid.py:98).  
Why: causes mixed precision under x64 and warning spam under default x32.

Your 4 specific concerns:

1. NPZD alkalinity sign at `npzd.py:188`: `confirmed-bug` (for Dickson TA).  
`+growth` / `-remin` / `-(1-gamma)grazing` is the correct N-cycle sign if ALK is total alkalinity including `-[NO3-]`.

2. KPP non-local sign at `kpp.py:359-371`: `false-positive`.  
The implemented divergence form is internally consistent with `dT/dt = -(F_{k+1/2}-F_{k-1/2})/dz`; surface term sign is correct for that convention.

3. Plume entrainment `entrain = epsilon*dz` at `plume.py:56-58`: `ambiguous`.  
It is first-order valid for small `epsilon*dz`; for coarse/thick layers it can exceed 1 and become unphysical. A bounded form (`1-exp(-epsilon*dz)` or clip) is safer.

4. 4D FCT patch edge cases in `operators_cdgrid.py`: `false-positive` for the two cases you asked.  
Non-square face geometry is not a supported cubed-sphere shape here, and the `...`/negative-axis rewrite is consistent for both 3D and 4D rank handling. No additional indexing break is evident from this patch.

Assumptions and gaps:
- I could not run the project pytest suite in this shell because `pytest` is not installed.
- Findings above are from direct code inspection plus executable spot checks in ad hoc scripts.
2026-05-02T13:08:16.683746Z ERROR codex_core::session: failed to record rollout items: thread 019de8c2-2f27-7250-9845-7cbdaedc9d05 not found
tokens used
335,301
1. `confirmed-bug` — NPZD alkalinity N-cycle sign is flipped for total alkalinity.  
Reference: [npzd.py:186](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/biogeochemistry/npzd.py:186), [npzd.py:188](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/biogeochemistry/npzd.py:188).  
Why: code makes `dALK_dt` track `dNO3_dt` (organic term), i.e. nitrate uptake lowers ALK. For Dickson TA (`... - [NO3-]`), this should be opposite sign (`-dNO3_dt`) for the N-cycle part.

2. `confirmed-bug` — KPP interior/background diffusivity is double-counted.  
Reference: [kpp.py:282](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:282), [kpp.py:297](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:297), [kpp.py:298](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:298).  
Why: `K_interior` already includes `+ cfg.K_bg`, then `K_v` adds `+ cfg.K_bg` again. This gives floor `2*K_bg` (I verified min `K_v=2e-5` for `K_bg=1e-5`).

3. `confirmed-bug` — `runoff_subsurface` is dropped when `runoff_surface` is `None`.  
Reference: [freshwater.py:191](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/freshwater.py:191), [freshwater.py:196](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/freshwater.py:196).  
Why: logic only adds subsurface runoff inside the `runoff_surface is not None` branch.

4. `confirmed-bug` — cosine barotropic filter degenerates for `n_substeps=1` and causes divide-by-zero in averages.  
Reference: [barotropic_common.py:60](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/dynamics/barotropic_common.py:60), [barotropic_common.py:66](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/dynamics/barotropic_common.py:66), [barotropic_latlon_cgrid.py:399](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:399), [barotropic_mpas.py:308](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/dynamics/barotropic_mpas.py:308).  
Why: for `n_substeps=1`, `w_filter=[0]`, `w_total=0`, then `eta_sum/w_total` and velocity averages divide by zero.

5. `confirmed-bug` — KPP integration path ignores provided surface forcing (`tau`, `q_net`, etc.).  
Reference: [vertical_mixing/integration.py:88](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/integration.py:88), [vertical_mixing/integration.py:91](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/integration.py:91), [kpp.py:150](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/physics/vertical_mixing/kpp.py:150).  
Why: wrapper accepts `surface_forcing` but never passes any forcing-derived terms to `kpp_vertical_mixing`.

6. `confirmed-bug` — lat-lon rest-state override path hardcodes `float64` bathymetry against policy dtype.  
Reference: [init_latlon_cgrid.py:98](/Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My%20Drive/Code/legoESM/src/legoesm/ocean/init_latlon_cgrid.py:98).  
Why: causes mixed precision under x64 and warning spam under default x32.

Your 4 specific concerns:

1. NPZD alkalinity sign at `npzd.py:188`: `confirmed-bug` (for Dickson TA).  
`+growth` / `-remin` / `-(1-gamma)grazing` is the correct N-cycle sign if ALK is total alkalinity including `-[NO3-]`.

2. KPP non-local sign at `kpp.py:359-371`: `false-positive`.  
The implemented divergence form is internally consistent with `dT/dt = -(F_{k+1/2}-F_{k-1/2})/dz`; surface term sign is correct for that convention.

3. Plume entrainment `entrain = epsilon*dz` at `plume.py:56-58`: `ambiguous`.  
It is first-order valid for small `epsilon*dz`; for coarse/thick layers it can exceed 1 and become unphysical. A bounded form (`1-exp(-epsilon*dz)` or clip) is safer.

4. 4D FCT patch edge cases in `operators_cdgrid.py`: `false-positive` for the two cases you asked.  
Non-square face geometry is not a supported cubed-sphere shape here, and the `...`/negative-axis rewrite is consistent for both 3D and 4D rank handling. No additional indexing break is evident from this patch.

Assumptions and gaps:
- I could not run the project pytest suite in this shell because `pytest` is not installed.
- Findings above are from direct code inspection plus executable spot checks in ad hoc scripts.
