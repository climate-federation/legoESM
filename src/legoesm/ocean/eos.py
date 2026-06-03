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


def eos_density_derivatives(
    eos_fn,
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Locally-referenced EOS partial derivatives ``(∂ρ/∂T, ∂ρ/∂S)``.

    Returns the two partial derivatives of an arbitrary EOS callable
    ``eos_fn(T, S, p) -> ρ`` evaluated at the LOCAL pressure ``p`` — the
    ingredients of the locally-referenced *neutral* density gradient
    ``∇_neutral ρ = (∂ρ/∂T)·∇T + (∂ρ/∂S)·∇S`` used by isoneutral mixing
    (Veros ``get_drhodT`` / ``get_drhodS`` at ``abs(zt)``;
    ``veros/core/isoneutral/isoneutral.py:40-41``).

    Relationship to the thermodynamic coefficients already in this module:
    ``∂ρ/∂T = −ρ·α`` (``thermal_expansion_coeff``) and
    ``∂ρ/∂S = +ρ·β`` (``haline_contraction_coeff``).  Rather than restrict
    to those wright-only helpers, this differentiates the *selected*
    ``eos_fn`` directly via :func:`jax.grad` so EVERY dispatchable EOS
    (wright, linear, unesco80, veros_nonlin2/3) is supported with one code
    path — exactly the autodiff α/β already use internally for wright.

    Holding ``p`` fixed during the differentiation is deliberate and
    faithful: the neutral gradient is the density change at constant
    (local) reference pressure, so the adiabatic compressibility term
    ``∂ρ/∂p·∂p/∂z`` — which makes the in-situ ``∂ρ/∂z`` ~4× too steep — is
    excluded by construction.

    Parameters
    ----------
    eos_fn : Callable[[array, array, array], array]
        Equation of state (e.g. from :func:`make_eos_fn`).
    T, S, p : array
        Potential temperature [°C], salinity [PSU], pressure [Pa] at the
        SAME points; identical shapes.

    Returns
    -------
    drho_dT : array — ``∂ρ/∂T`` [kg/m³/K], same shape as inputs.
    drho_dS : array — ``∂ρ/∂S`` [kg/m³/(g/kg)], same shape as inputs.

    Notes
    -----
    Fully ``jax.grad``-safe: only EOS evaluations + autodiff, no
    ``where``/``cond`` on traced values.  Promotes to the EOS compute dtype
    (float64 in mixed mode) for the polynomial evaluation, matching α/β.
    """
    hi = _resolve_dtype("equation_of_state", "compute")
    T64 = T.astype(hi)
    S64 = S.astype(hi)
    p64 = p.astype(hi)
    flat_T = T64.ravel()
    flat_S = S64.ravel()
    flat_p = p64.ravel()

    def _scalar(t, s, pp):
        # eos_fn is array-shaped; wrap scalars in length-1 arrays so the
        # promotion-to-float64 astype inside the EOS sees an ndarray.
        return eos_fn(t[None], s[None], pp[None])[0]

    drho_dT = jax.vmap(jax.grad(_scalar, argnums=0))(
        flat_T, flat_S, flat_p
    ).reshape(T.shape)
    drho_dS = jax.vmap(jax.grad(_scalar, argnums=1))(
        flat_T, flat_S, flat_p
    ).reshape(T.shape)
    return drho_dT.astype(T.dtype), drho_dS.astype(T.dtype)


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


# ==============================================================================
# UNESCO 1980 EOS coefficients (international one-atmosphere standard;
# Fofonoff & Millard 1983 UNESCO Tech. Papers in Marine Science No. 44).
# Veros's ``eq_of_state_type=3`` is the closely-related Jackett &
# McDougall 1995 polynomial, which modifies a small subset of these
# coefficients to better match measurements but agrees to within
# ~0.001 kg/m³ at typical ocean T/S. For bit-exact Veros parity, read
# Veros's polynomial directly; for tier-2 PGF-tendency comparison the
# UNESCO 1980 form should land inside the discretisation noise floor.
# ==============================================================================

# rho_w(T) — density of pure water [kg/m³]
_UN80_A0 = 999.842594
_UN80_A1 = 6.793952e-2
_UN80_A2 = -9.095290e-3
_UN80_A3 = 1.001685e-4
_UN80_A4 = -1.120083e-6
_UN80_A5 = 6.536332e-9

# A(T)·S coefficients
_UN80_B0 = 8.24493e-1
_UN80_B1 = -4.0899e-3
_UN80_B2 = 7.6438e-5
_UN80_B3 = -8.2467e-7
_UN80_B4 = 5.3875e-9

# B(T)·S^(3/2) coefficients
_UN80_C0 = -5.72466e-3
_UN80_C1 = 1.0227e-4
_UN80_C2 = -1.6546e-6

# C·S^2 coefficient
_UN80_D0 = 4.8314e-4

# K0(T) — secant bulk modulus of pure water at p=0 [bar]
_UN80_E0 = 19652.21
_UN80_E1 = 148.4206
_UN80_E2 = -2.327105
_UN80_E3 = 1.360477e-2
_UN80_E4 = -5.155288e-5

# KS(T) — salinity correction to K at p=0 [bar / PSU]
_UN80_F0 = 54.6746
_UN80_F1 = -0.603459
_UN80_F2 = 1.09987e-2
_UN80_F3 = -6.1670e-5

# S^(3/2) correction to K at p=0 [bar / PSU^(3/2)]
_UN80_G0 = 7.944e-2
_UN80_G1 = 1.6483e-2
_UN80_G2 = -5.3009e-4

# K_p0(T) — pressure correction (coefficient of p) [bar / bar = dimensionless]
_UN80_H0 = 3.239908
_UN80_H1 = 1.43713e-3
_UN80_H2 = 1.16092e-4
_UN80_H3 = -5.77905e-7

# K_pS(T)·S correction (coefficient of p·S)
_UN80_I0 = 2.2838e-3
_UN80_I1 = -1.0981e-5
_UN80_I2 = -1.6078e-6

# K_pS^(3/2) correction (coefficient of p·S^(3/2))
_UN80_J0 = 1.91075e-4

# K_pp(T) — p² coefficient
_UN80_K0 = 8.50935e-5
_UN80_K1 = -6.12293e-6
_UN80_K2 = 5.2787e-8

# K_ppS·S — p²·S coefficient
_UN80_M0 = -9.9348e-7
_UN80_M1 = 2.0816e-8
_UN80_M2 = 9.1697e-10


def unesco80_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    """UNESCO 1980 international one-atmosphere equation of state.

    Parameters
    ----------
    T : array
        Potential temperature [°C]. Valid range: -2 to 40 °C.
    S : array
        Practical salinity [PSU]. Valid range: 0 to 42 PSU.
    p : array
        Sea pressure (gauge; 0 at surface) [Pa]. Valid range: 0 to
        10000 dbar = 1e8 Pa.

    Returns
    -------
    array
        In-situ density [kg/m³]. Reference values:

        - ``rho(0, 0, 0)   ≈ 999.842594``
        - ``rho(0, 35, 0)  ≈ 1028.106331``
        - ``rho(20, 35, 0) ≈ 1024.78``

    Notes
    -----
    All intermediate computations are promoted to float64 to avoid
    precision loss from the large polynomial coefficients (e.g.
    ``E0 ≈ 1.97e4``); the function is differentiable end-to-end.

    The UNESCO 1980 polynomial uses pressure in **bar** internally; the
    input ``p`` (in Pa) is converted via ``p_bar = p · 1e-5``.

    For Veros's ``eq_of_state_type=3`` (JM95) parity at the per-tendency
    level, this implementation should land inside the
    discretisation-truncation noise budget for typical ocean conditions.
    Bit-exact parity requires the exact Veros / JM95 coefficient table
    and is tracked as a follow-up under Phase G.1b in the audit doc.
    """
    T = T.astype(jnp.float64)
    S = S.astype(jnp.float64)
    # Convert pressure Pa → bar (UNESCO convention).
    p_bar = p.astype(jnp.float64) * 1e-5
    # Ensure non-negative salinity in the polynomial (clip floor at 0).
    S_safe = jnp.maximum(S, 0.0)
    S_sqrt = jnp.sqrt(S_safe)

    # Density of pure water at p = 0
    rho_w = (
        _UN80_A0
        + T * (_UN80_A1
        + T * (_UN80_A2
        + T * (_UN80_A3
        + T * (_UN80_A4
        + T * _UN80_A5))))
    )

    # Salinity correction at p = 0
    A_T = (
        _UN80_B0
        + T * (_UN80_B1
        + T * (_UN80_B2
        + T * (_UN80_B3
        + T * _UN80_B4)))
    )
    B_T = (
        _UN80_C0
        + T * (_UN80_C1
        + T * _UN80_C2)
    )
    rho_0 = rho_w + A_T * S + B_T * S_safe * S_sqrt + _UN80_D0 * S * S

    # Secant bulk modulus K(T, S, p)
    K0_T = (
        _UN80_E0
        + T * (_UN80_E1
        + T * (_UN80_E2
        + T * (_UN80_E3
        + T * _UN80_E4)))
    )
    KS_T = (
        _UN80_F0
        + T * (_UN80_F1
        + T * (_UN80_F2
        + T * _UN80_F3))
    )
    KS32_T = (
        _UN80_G0
        + T * (_UN80_G1
        + T * _UN80_G2)
    )
    K_p0 = K0_T + KS_T * S + KS32_T * S_safe * S_sqrt

    Kp_T = (
        _UN80_H0
        + T * (_UN80_H1
        + T * (_UN80_H2
        + T * _UN80_H3))
    )
    KpS_T = (
        _UN80_I0
        + T * (_UN80_I1
        + T * _UN80_I2)
    )
    K_p1 = Kp_T + KpS_T * S + _UN80_J0 * S_safe * S_sqrt

    Kpp_T = (
        _UN80_K0
        + T * (_UN80_K1
        + T * _UN80_K2)
    )
    KppS_T = (
        _UN80_M0
        + T * (_UN80_M1
        + T * _UN80_M2)
    )
    K_p2 = Kpp_T + KppS_T * S

    K = K_p0 + K_p1 * p_bar + K_p2 * p_bar * p_bar

    rho = rho_0 / (1.0 - p_bar / K)
    return rho


# ==============================================================================
# Veros "nonlinear EOS variant 2" (Vallis 2008) — eq_of_state_type=3 in
# Veros (despite the "_eq2" file name, this is what
# ``veros/core/density/get_rho.py`` dispatches to for type=3).
#
# Quadratic in T with a small T² nonlinearity, linear salinity term, and
# pressure dependence with a coupled T·z correction:
#
#     rho_anom = -(grav·z/cs0² + βT·(1 - γs·grav·z·ρ0)·θ + βTs·θ²/2
#                  - βS·(S - S0)) · ρ0
#
# with z = depth (positive downward), θ = T - theta0. Source:
# ``veros/core/density/nonlinear_eq2.py``.
# ==============================================================================


class VerosNonlin2Config(NamedTuple):
    """Veros eq_of_state_type=3 (Vallis 2008 nonlin2) coefficients.

    Defaults taken verbatim from
    ``veros/core/density/nonlinear_eq2.py``.
    """
    rho_0: float = 1024.0
    theta0_C: float = 283.0 - 273.15   # 9.85 °C
    S0: float = 35.0
    grav: float = 9.81
    cs0: float = 1490.0                  # speed of sound [m/s]
    betaT: float = 1.67e-4
    betaTs: float = 1.0e-5
    betaS: float = 0.78e-3
    gammas: float = 1.1e-8
    z0: float = 0.0


def veros_nonlin2_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    cfg: VerosNonlin2Config | None = None,
) -> jnp.ndarray:
    """Veros ``eq_of_state_type=3`` density (Vallis 2008 nonlin2).

    Parameters
    ----------
    T : array — potential temperature [°C]
    S : array — salinity [g/kg or PSU]
    p : array — **pressure in Pa** (the legoESM EOS API convention).
        Internally converted to Veros's "depth in m" convention via
        ``depth_m ≈ p / (rho_0 * g)``. This matches Veros's actual
        usage which passes ``abs(vs.zt)`` (depth in meters) as the
        ``press`` argument to ``get_rho``; that is hydrostatically
        equivalent to ``p_Pa / (rho_0 g)``.

    Returns
    -------
    array
        In-situ density [kg/m³]. (Veros's
        ``nonlin2_eq_of_state_rho`` returns the anomaly; this function
        adds ``rho_0`` so the legoESM EOS API contract stays consistent.)
    """
    cfg = cfg if cfg is not None else VerosNonlin2Config()
    # Convert pressure (Pa) → depth (m) using ``depth ≈ p / (rho_0 g)``.
    # For a hydrostatic ocean column this is exact at the cell centre
    # where ``p_hydro = rho_0 g z``. At depths where rho deviates from
    # rho_0 the equivalence introduces a sub-percent error well below
    # the discretisation noise.
    depth_m = p / (cfg.rho_0 * cfg.grav)
    zz = -depth_m - cfg.z0   # Veros sign convention (negative below surface)
    thetas = T - cfg.theta0_C
    rho_anom = -(
        cfg.grav * zz / (cfg.cs0 * cfg.cs0)
        + cfg.betaT * (1.0 - cfg.gammas * cfg.grav * zz * cfg.rho_0) * thetas
        + 0.5 * cfg.betaTs * thetas * thetas
        - cfg.betaS * (S - cfg.S0)
    ) * cfg.rho_0
    return rho_anom + cfg.rho_0


# ==============================================================================
# Veros "nonlinear EOS variant 3" — eq_of_state_type=4 in Veros.
# Quadratic-in-T (with optional T^2 nonlinearity), zero salinity contribution,
# zero pressure dependency. Despite Veros calling this "nonlinear", it's a
# simple polynomial that captures the leading T-dependence of seawater
# density without invoking UNESCO 1980 / JM95.
#
# Source: veros/core/density/nonlinear_eq3.py:
#   rho = -(betaT·(T - theta0) + betaTs·(T - theta0)^2
#           - betaS·(S - S0)) · rho_0
# Defaults from the Veros source (used by veros.setups.acc.ACCSetup):
#   rho_0 = 1024, theta0 = 9.85 [°C] (= 283 K - T_freeze),
#   S0 = 35, betaT = 1.67e-4, betaTs = 5e-6, betaS = 0
#
# Because betaS = 0 by default, salinity is effectively a passive tracer in
# this EOS — exactly the convention Veros ACC relies on (uniform 35 PSU init).
# ==============================================================================


class VerosNonlin3Config(NamedTuple):
    """Veros eq_of_state_type=3 (nonlinear, T-only) coefficients."""
    rho_0: float = 1024.0
    theta0_C: float = 283.0 - 273.15      # 9.85 °C
    S0: float = 35.0
    betaT: float = 1.67e-4
    betaTs: float = 5.0e-6                # = 1e-5 / 2 (Veros source convention)
    betaS: float = 0.0                    # NOTE: zero by default in Veros nonlin3


def veros_nonlin3_eos(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p: jnp.ndarray,
    cfg: VerosNonlin3Config | None = None,
) -> jnp.ndarray:
    """Veros ``eq_of_state_type=4`` density (quadratic in T, no S, no p).

    Returns
    -------
    array
        In-situ density [kg/m³]. Pressure ``p`` is accepted for API
        symmetry but ignored — this EOS has no pressure dependence.

    Notes
    -----
    Mirrors ``veros/core/density/nonlinear_eq3.py`` exactly.
    NOTE: Veros's ``eq_of_state_type=3`` dispatches to ``nonlin2`` (see
    ``veros/core/density/get_rho.py``), so this function — despite
    being a port of ``nonlinear_eq3.py`` — corresponds to ``type=4``.
    For ``ACCSetup`` (which uses ``type=3``) see :func:`veros_nonlin2_eos`.
    """
    cfg = cfg if cfg is not None else VerosNonlin3Config()
    thetas = T - cfg.theta0_C
    rho_anom = -(
        cfg.betaT * thetas
        + cfg.betaTs * thetas * thetas
        - cfg.betaS * (S - cfg.S0)
    ) * cfg.rho_0
    # Veros returns the density anomaly (rho - rho_0); to match the legoESM
    # EOS API (in-situ rho) we add rho_0 back here.
    return rho_anom + cfg.rho_0


# Single source of truth for the dispatchable EOS scheme literals. Referenced
# by both make_eos_fn (unknown-scheme ValueError) and config validators
# (fail-fast at construction) so the valid set is never duplicated.
VALID_EOS_SCHEMES = frozenset(
    {"wright", "linear", "unesco80", "veros_nonlin2", "veros_nonlin3"}
)


def make_eos_fn(eos="wright", eos_linear=None,
                eos_veros_nonlin2: VerosNonlin2Config | None = None,
                eos_veros_nonlin3: VerosNonlin3Config | None = None):
    """Return an EOS callable ``fn(T, S, p) -> rho``.

    Parameters
    ----------
    eos : str
        ``"wright"`` (default, Wright 1997), ``"linear"``, or
        ``"unesco80"`` (UNESCO 1980 polynomial — close approximation
        to Veros's ``eq_of_state_type=3`` JM95 form, within ~0.001 kg/m³
        at typical ocean T/S; bit-exact Veros parity requires reading
        Veros's polynomial coefficients directly).
    eos_linear : LinearEOSConfig or None
        Parameters for linear EOS.  Ignored unless *eos* is ``"linear"``.
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
    elif eos == "unesco80":
        return unesco80_eos
    elif eos == "veros_nonlin2":
        cfg = eos_veros_nonlin2 if eos_veros_nonlin2 is not None else VerosNonlin2Config()
        def _veros_nl2(T, S, p):
            return veros_nonlin2_eos(T, S, p, cfg=cfg)
        return _veros_nl2
    elif eos == "veros_nonlin3":
        cfg = eos_veros_nonlin3 if eos_veros_nonlin3 is not None else VerosNonlin3Config()
        def _veros_nl3(T, S, p):
            return veros_nonlin3_eos(T, S, p, cfg=cfg)
        return _veros_nl3
    else:
        raise ValueError(
            f"Unknown EOS scheme: {eos!r}. Valid schemes: "
            f"{sorted(VALID_EOS_SCHEMES)}."
        )


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


def compute_buoyancy_frequency_adiabatic(
    T: jnp.ndarray,
    S: jnp.ndarray,
    p_cell: jnp.ndarray,
    dz: jnp.ndarray,
    jacobian: jnp.ndarray,
    eos_fn=None,
    rho_ref: float = rho_0,
    g: float = constants.g,
) -> jnp.ndarray:
    r"""Static-stability ``N^2`` via adiabatic parcel displacement.

    This is the **true static stability** used by Veros
    (``veros/core/thermodynamics.py:99-103``) and the cleaner form
    requested for oracle fidelity: instead of differencing the *in-situ*
    densities ``rho(T[k], S[k], p[k]) - rho(T[k+1], S[k+1], p[k+1])``
    (which carries the compressibility difference between two different
    reference pressures and is therefore biased ~6x too stable), both
    parcels are evaluated at the **upper cell's reference pressure**
    ``p_cell[k]`` so only the (potential-) density contrast remains:

    .. math::

        N^2[k] = -\frac{g}{\rho_0}\,
            \frac{\rho(T_{k+1}, S_{k+1}, p_k) - \rho(T_k, S_k, p_k)}
                 {\Delta z_{int}[k]}

    With ``z`` increasing upward and legoESM's ``k=0`` at the surface,
    a statically *unstable* column (denser water displaced over lighter)
    gives ``N^2 < 0``. **Unlike** :func:`compute_buoyancy_frequency` and
    the legacy ``tke._compute_N2``, this is **not** clipped at zero — the
    sign is the convection trigger, so it must be allowed to go negative.

    The choice of the *upper cell-centre* pressure ``p_cell[k]`` (rather
    than the interface pressure) matches Veros exactly: Veros passes
    ``press = abs(zt)`` (the cell-centre geometric depth of the upper
    cell) and compares ``get_rho(T[k+1], S[k+1], press[k])`` against the
    upper cell's own in-situ density ``rho[k] = get_rho(T[k], S[k],
    press[k])`` — i.e. both at ``press[k]``.

    Differentiability: the only operations are the (differentiable) EOS
    evaluations and arithmetic — no ``where``/``cond`` on traced values —
    so this is fully ``jax.grad``-safe (``d N^2 / dT`` etc. flow through
    the EOS at the displaced pressure).

    Parameters
    ----------
    T, S : array
        Potential temperature [degC] / salinity [PSU] at cell centres,
        shape ``(..., nlev)``.
    p_cell : array
        Hydrostatic pressure [Pa] at cell centres, shape ``(..., nlev)``
        (e.g. from :func:`compute_ocean_rho_and_pressure`). The pressure
        of the *upper* cell of each interface, ``p_cell[..., :-1]``, is
        used as the common reference pressure for both displaced parcels.
    dz : array
        Reference layer thickness [m], shape ``(nlev,)``.
    jacobian : array
        Dynamic Jacobian, shape ``(...)``.
    eos_fn : callable or None
        EOS ``fn(T, S, p) -> rho``. If None, uses :func:`wright_eos`.
    rho_ref : float
        Reference density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].

    Returns
    -------
    array : ``N^2`` at interior interfaces [1/s^2], shape ``(..., nlev-1)``.
        **Signed** (negative where statically unstable).
    """
    if eos_fn is None:
        eos_fn = wright_eos

    dz_actual = dz * jacobian[..., jnp.newaxis]
    dz_interface = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])

    # Common reference pressure = the UPPER cell's centre pressure (Veros
    # press[k] = abs(zt[k])).
    p_ref_int = p_cell[..., :-1]

    # Upper parcel (k) and lower parcel (k+1), BOTH at the upper pressure.
    rho_upper = eos_fn(T[..., :-1], S[..., :-1], p_ref_int)
    rho_lower = eos_fn(T[..., 1:], S[..., 1:], p_ref_int)

    # drho/dz with z positive upward; (rho_upper - rho_lower)/dz. For a
    # stable column rho_upper < rho_lower -> drho/dz < 0 -> N^2 > 0.
    drho_dz = (rho_upper - rho_lower) / dz_interface
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
