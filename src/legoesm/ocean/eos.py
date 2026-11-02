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
# Re-export central physical constants so existing call sites
# (``from legoesm.ocean.eos import rho_0, c_sw, ...``) keep working.
# Per CLAUDE.md the canonical values live in ``legoesm.constants``;
# the prior literal definitions here violated the
# constant-discipline rule and could silently drift from the central
# values.  Module-level binding to ``constants.X`` keeps a single
# source of truth.
rho_0 = constants.rho_ocean         # Reference seawater density [kg/m^3]
c_sw = constants.c_sw               # Specific heat of seawater [J/(kg*K)]
T_freeze_ocean = constants.T_freeze_ocean  # Freezing point of seawater [K]
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
    rho_ref: float = constants.rho_ocean
    alpha_T: float = 2.0e-4    # Thermal expansion coefficient [1/K]
    beta_S: float = 7.4e-4     # Haline contraction coefficient [1/PSU]
    T_ref: float = 10.0        # Reference temperature [°C]
    S_ref: float = 35.0        # Reference salinity [PSU]


def linear_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    rho_ref: float = rho_0,
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
    h_actual: jnp.ndarray | None = None,
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
        Reference layer thickness [m], shape (nlev,).  Ignored when
        ``h_actual`` is provided.
    jacobian : array
        Dynamic Jacobian, shape (...).  Ignored when ``h_actual`` is
        provided.
    rho_ref : float
        Reference density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].
    h_actual : array or None
        Optional pre-computed per-cell layer thickness, shape
        (..., nlev).  When provided, used directly; when None, the
        legacy formula ``dz * jacobian[..., None]`` is used.

        This is the partial-cells extension point: callers using an
        ``OceanPartialCellCoordinate`` should pass
        ``h_actual = compute_layer_thickness(eta, H_bathy, coord)``
        to integrate pressure with the correct partial bottom-cell
        thickness.  Cells below the seafloor have h_actual=0, so they
        contribute zero pressure increment automatically.

    Returns
    -------
    array : Hydrostatic pressure at full levels [Pa], shape (..., nlev).
    """
    # Surface pressure from free surface
    p_surface = rho_ref * g * eta  # (...,)

    # Actual layer thickness — pre-computed (partial cells) or
    # dz * jacobian (legacy z*).
    if h_actual is None:
        h_actual = dz * jacobian[..., jnp.newaxis]  # (..., nlev)

    # Pressure increment per layer: rho * g * h
    dp = rho * g * h_actual  # (..., nlev)

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

def _maybe_partial_h_actual(state, z_coord):
    """Return per-cell h_actual when z_coord is a partial-cell coord,
    else None (caller falls back to dz * jacobian).

    Routed through the local import to avoid a circular dependency:
    eos.py imports vertical.py would create a cycle through state.py.
    """
    from legoesm.ocean.vertical import (
        OceanPartialCellCoordinate, compute_layer_thickness,
    )
    if isinstance(z_coord, OceanPartialCellCoordinate):
        return compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord,
        )
    return None


def compute_ocean_rho(state, z_coord, jacobian, eos_fn=None):
    """Compute in-situ density from ocean state.

    Used by vertical mixing, lateral mixing, and convection integration
    bridges. Avoids triplicating the same hydrostatic pressure + EOS call.

    Dispatches on coord type:

    - ``OceanZStarCoordinate``: legacy path, uses ``dz_ref * jacobian``
      for layer thickness.  Bit-exact unchanged.
    - ``OceanPartialCellCoordinate``: passes per-cell ``h_partial *
      (eta+H_bathy)/H_bathy`` to the hydrostatic integrator so the
      partial bottom cell's contribution is correct.

    Parameters
    ----------
    state : OceanState
        Must have .T, .S, .eta, .H_bathy fields.
    z_coord : OceanZStarCoordinate or OceanPartialCellCoordinate
        Vertical coordinate.
    jacobian : array
        Dynamic Jacobian.  For pure z*: (eta + H) / H_max.  For
        partial cells: (eta + H_bathy) / H_bathy.  Caller is expected
        to use ``compute_ocean_jacobian`` which dispatches.
    eos_fn : callable or None
        EOS function ``fn(T, S, p) -> rho``.  If None, uses ``wright_eos``.

    Returns
    -------
    array : In-situ density [kg/m^3].
    """
    if eos_fn is None:
        eos_fn = wright_eos
    h_actual = _maybe_partial_h_actual(state, z_coord)
    # Two EOS iterations for density-pressure consistency, matching the
    # dynamical core (ocean_pe_cdgrid.py).
    rho = eos_fn(state.T.data, state.S.data, jnp.zeros_like(state.T.data))
    for _ in range(2):
        p_hydro = compute_hydrostatic_pressure(
            rho, state.eta.data, z_coord.dz_ref, jacobian, rho_0,
            h_actual=h_actual,
        )
        rho = eos_fn(state.T.data, state.S.data, p_hydro)
    return rho


def compute_ocean_rho_and_pressure(state, z_coord, jacobian, eos_fn=None):
    """Compute in-situ density and hydrostatic pressure from ocean state.

    Dispatches on coord type — see ``compute_ocean_rho``.

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
    h_actual = _maybe_partial_h_actual(state, z_coord)
    p_hydro = compute_hydrostatic_pressure(
        rho, state.eta.data, z_coord.dz_ref, jacobian, rho_0,
        h_actual=h_actual,
    )
    return rho, p_hydro


# --- from our tree (MARBL work), re-added after the colleague merge 2026-10-16 ---
def locally_referenced_buoyancy_frequency(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p_half: jnp.ndarray,
    dz_half: jnp.ndarray,
    eos_fn=None,
    rho_ref: float = rho_0,
    g: float = constants.g,
) -> jnp.ndarray:
    """Locally referenced N^2 at interior interfaces.

    Both cells adjacent to interface k+1/2 are evaluated at the interface
    pressure, so the compressibility of seawater does not enter N^2.
    Differencing in-situ density (``compute_buoyancy_frequency``) adds
    ~ g^2 / c_s^2 ~ 4e-5 s^-2 to N^2 everywhere.

    Parameters
    ----------
    T, S : array (..., nlev)
        Potential temperature [degC], salinity [PSU]; k=0 is the surface.
    p_half : array (..., nlev-1)
        Pressure at interior interfaces [Pa].
    dz_half : array (..., nlev-1)
        Distance between adjacent cell centres [m].
    eos_fn : callable or None
        ``fn(T, S, p) -> rho``; defaults to ``wright_eos``.

    Returns
    -------
    array : N^2 [1/s^2], shape (..., nlev-1), positive for stable stratification.
    """
    if eos_fn is None:
        eos_fn = wright_eos
    rho_up = eos_fn(T[..., :-1], S[..., :-1], p_half)
    rho_dn = eos_fn(T[..., 1:], S[..., 1:], p_half)
    return (g / rho_ref) * (rho_dn - rho_up) / dz_half


def adiabatic_temperature_gradient(
    S: jnp.ndarray,
    T: jnp.ndarray,
    p_bar: jnp.ndarray,
) -> jnp.ndarray:
    """Adiabatic lapse rate of seawater [degC/bar] (Bryden 1973; UNESCO 1983).

    T is in-situ temperature [degC], p_bar sea pressure [bar].
    UNESCO check value: S = 40, T = 40 degC, p = 1000 bar -> 3.255976e-3 degC/bar.
    """
    # Bryden's coefficients are per 1000 dbar (= 100 bar) with p in dbar.
    k = 1.0e-2
    Sm = S - 35.0
    p = p_bar * 10.0  # dbar
    return k * (0.35803e-1 + T * (0.85258e-2 + T * (-0.68360e-4 + T * 0.66228e-6))
                + Sm * (0.18932e-2 - 0.42393e-4 * T)
                + p * (0.18741e-4 + T * (-0.67795e-6 + T * (0.87330e-8 - T * 0.54481e-10))
                       + Sm * (-0.11351e-6 + 0.27759e-8 * T)
                       + p * (-0.46206e-9 + T * (0.18676e-10 - T * 0.21687e-12))))


def potential_temperature(
    S: jnp.ndarray,
    T: jnp.ndarray,
    p_bar: jnp.ndarray,
    p_ref_bar: jnp.ndarray,
) -> jnp.ndarray:
    """Temperature of a parcel moved adiabatically from p_bar to p_ref_bar [degC].

    Fofonoff (1977) fourth-order Runge-Kutta integration of the adiabatic
    lapse rate.  With p_bar = 0 and p_ref_bar = sea pressure this converts
    potential temperature to in-situ temperature.
    UNESCO check value: S = 40, T = 40 degC, p = 1000 bar, p_ref = 0 -> 36.89073 degC.
    """
    sqrt2 = jnp.sqrt(2.0)
    dp = p_ref_bar - p_bar
    p_mid = p_bar + 0.5 * dp
    dtheta = dp * adiabatic_temperature_gradient(S, T, p_bar)
    theta = T + 0.5 * dtheta
    q = dtheta
    dtheta = dp * adiabatic_temperature_gradient(S, theta, p_mid)
    theta = theta + (1.0 - 1.0 / sqrt2) * (dtheta - q)
    q = (2.0 - sqrt2) * dtheta + (-2.0 + 3.0 / sqrt2) * q
    dtheta = dp * adiabatic_temperature_gradient(S, theta, p_mid)
    theta = theta + (1.0 + 1.0 / sqrt2) * (dtheta - q)
    q = (2.0 + sqrt2) * dtheta + (-2.0 - 3.0 / sqrt2) * q
    dtheta = dp * adiabatic_temperature_gradient(S, theta, p_ref_bar)
    return theta + (dtheta - 2.0 * q) / 6.0
