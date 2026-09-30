"""Monin-Obukhov bulk air-sea flux algorithms.

Provides stability-dependent transfer coefficients using iterative
Monin-Obukhov similarity theory (MOST). Three schemes:

1. ``constant`` — Fixed neutral transfer coefficients (no iteration)
2. ``coare3`` — COARE 3.0 (Fairall et al. 2003): Charnock + smooth-flow
   roughness, Fairall Kansas + free-convective unstable blend, BH91-form
   native stable branch (selectable via ``stability_scheme``)
3. ``large_yeager`` — Large & Yeager 2009 (CORE/OMIP): empirical C_DN(U_10N)
   with high-wind quintic correction, stability-dependent Stanton/Dalton
   numbers

The iterative Obukhov length loop uses ``jax.lax.fori_loop`` for
full JAX differentiability (compatible with jax.grad, jax.jit).

The ``large_yeager`` path is the OMIP-2 protocol bulk formula (Griffies
2016 §2.2 mandates Large & Yeager 2009).

References
----------
- Fairall, C. W., et al. (2003). Bulk parameterization of air-sea fluxes:
  Updates and verification for the COARE algorithm. J. Climate, 16, 571-591.
- Large, W. G., & Yeager, S. G. (2009). The global climatology of an
  interannually varying air-sea flux data set. Climate Dynamics, 33,
  341-364. doi:10.1007/s00382-008-0441-3.
- Businger, J. A., et al. (1971). Flux-profile relationships in the
  atmospheric surface layer. J. Atmos. Sci., 28, 181-189.
- Dyer, A. J. (1974). A review of flux-profile relationships. Boundary-Layer
  Meteorol., 7, 363-372.
- Beljaars, A. C. M., & Holtslag, A. A. M. (1991). Flux parameterization over
  land surfaces for atmospheric models. J. Appl. Meteorol., 30, 327-341.
- Grachev, A. A., Andreas, E. L., Fairall, C. W., Guest, P. S., & Persson,
  P. O. G. (2007). SHEBA flux-profile relationships in the stable atmospheric
  boundary layer. Boundary-Layer Meteorol., 124, 315-333.
  doi:10.1007/s10546-007-9177-6.
- Gryanik, V. M., Lupkes, C., Grachev, A., & Sidorenko, D. (2020). New modified
  and extended stability functions for the stable boundary layer based on SHEBA
  and parametrizations of bulk transfer coefficients for climate models.
  J. Atmos. Sci., 77, 2687-2716. doi:10.1175/JAS-D-19-0255.1.

The stable-regime (zeta>0) similarity functions are selectable via the
``stability_scheme`` argument to :func:`psi_m` / :func:`psi_h` /
:func:`psi_m_coare` / :func:`psi_h_coare` / :func:`compute_most_fluxes`; the
unstable branch is never affected by the selector — Businger-Dyer on the
``psi_m``/``psi_h`` path, the Fairall (1996/2003) Kansas + free-convective
blend on the COARE path.  On ``coare3`` the default ``"dyer1974"`` is a
sentinel for the byte-identical COARE-native stable form (itself the BH91 fit
with rounded constants — see :func:`psi_m_coare`).  Coefficient values
cross-checked against CliMA ``SurfaceFluxes.jl`` (``UniversalFunctions``).
"""

from __future__ import annotations

import math
import numbers

import jax
import jax.numpy as jnp

from legoesm import constants

# Physical constants
KAPPA = constants.kappa_vk  # von Kármán constant (0.4)
G = constants.g
NU_AIR = constants.nu_air  # kinematic viscosity of air [m²/s]

# Known bulk-flux scheme names (union across all surface-flux dispatchers):
#   "constant"     — fixed neutral transfer coefficients (no iteration)
#   "most"         — iterative MOST with fixed roughness (constant-z0 stability)
#   "coare3"       — COARE 3.0
#   "large_yeager" — Large & Yeager 2009 (OMIP)
#   "large_yeager_cesm" — CESM/CIME ``shr_flux_atmOcn`` (Large & Pond 1981/82
#                         + LY04 neutral coefficients, 2 fixed iterations);
#                         see ``compute_sam_oceflx_fluxes(variant="cesm")``
#   "nemo_si3_constant" — NEMO SI3 constant-coefficient ice/ocean bulk identity
_VALID_BULK_SCHEMES = (
    "constant", "most", "coare3", "large_yeager", "large_yeager_cesm",
    "nemo_si3_constant",
)


# ============================================================================
# Stable-regime stability-function coefficients (empirical fits)
# ============================================================================
# ``stability_scheme`` selects the STABLE-branch (zeta = z/L > 0) Monin-Obukhov
# similarity functions psi_m, psi_h. The UNSTABLE branch (zeta < 0) stays
# Businger-Dyer for every scheme. These are EMPIRICAL fit coefficients (not
# physical constants), tabulated here as module-level named constants -- not
# inline literals -- each under its published-reference provenance block. Values
# cross-checked against CliMA SurfaceFluxes.jl (UniversalFunctions / ClimaParams).
_VALID_STABILITY_SCHEMES = (
    "dyer1974",
    "beljaars_holtslag1991",
    "grachev2007_sheba",
    "gryanik2020",
)

# --- Dyer (1974) linear stable functions ---
# psi_m(zeta) = psi_h(zeta) = -beta*zeta (the historical default). Dyer (1974).
_DYER_STABLE_BETA = 5.0

# --- Businger-Dyer (1971) / Dyer (1974) UNSTABLE-branch coefficient ---
# The universal gamma in the unstable dimensionless gradient functions
# phi_m = (1 - gamma*zeta)^{-1/4}, phi_h = (1 - gamma*zeta)^{-1/2} that
# integrate to the Businger-Dyer psi_m/psi_h; the historical value is 16.
# Businger et al. (1971) J. Atmos. Sci. 28, 181-189; Dyer (1974) BLM 7, 363-372.
# UNSTABLE branch only (zeta < 0); the beljaars/grachev/gryanik schemes carry
# their OWN published fits and are NOT reparameterised by this coefficient.
_DYER_UNSTABLE_GAMMA = 16.0

# --- Beljaars & Holtslag (1991) stable functions ---
#   psi_m(zeta) = -(a*zeta + b*(zeta - c/d)*exp(-d*zeta) + b*c/d)
#   psi_h(zeta) = -((1 + 2a*zeta/3)^{3/2} + b*(zeta - c/d)*exp(-d*zeta) + b*c/d - 1)
_BH91_A = 1.0
_BH91_B = 0.667
_BH91_C = 5.0
_BH91_D = 0.35

# --- Grachev et al. (2007) SHEBA stable functions ---
# Momentum = their Eq. 12 (x = (1+zeta)^{1/3}, B_m = (1/b_m - 1)^{1/3});
# heat = their Eq. 13 (B_h = (c_h^2 - 4)^{1/2}); b_m = a_m/6.5. Rational/log-
# arctan fits valid to very large zeta (Arctic sea-ice / strongly-stable SBL).
_GRACHEV_A_M = 5.0
_GRACHEV_B_M = 5.0 / 6.5  # = 0.7692307692307693
_GRACHEV_A_H = 5.0
_GRACHEV_B_H = 5.0
_GRACHEV_C_H = 3.0
# Grachev et al. (2007) Eq. 13 assumes phi_h(0) = 1 — there is NO neutral-
# Prandtl prefactor in the paper (d psi_h/d zeta|0+ = -a_h = -5), and CliMA
# SurfaceFluxes.jl likewise pins Pr_0 = 1.0 for Grachev (CreateParametersExt:
# "the formulation assumes phi_h(0) = 1.0").  Pr0 = 0.98 belongs to the
# Gryanik et al. (2020) modification only (see _GRYANIK_PR0 below).
_GRACHEV_PR0 = 1.0        # neutral turbulent Prandtl number (paper + CliMA)

# --- Gryanik et al. (2020) modified SHEBA stable functions ---
#   psi_m(zeta) = -3*(a_m/b_m)*((1 + b_m*zeta)^{1/3} - 1)
#   psi_h(zeta) = -Pr0*(a_h/b_h)*ln(1 + b_h*zeta)
_GRYANIK_A_M = 5.0
_GRYANIK_B_M = 0.3
_GRYANIK_A_H = 5.0
_GRYANIK_B_H = 0.4
_GRYANIK_PR0 = 0.98

# --- Thermal/momentum roughness ratio z0h/z0 ---
# Ratio of the thermal (scalar) roughness length z0h (= z0_t = z0_q) to the
# aerodynamic momentum roughness length z0 for the FIXED-roughness MOST path
# ("constant"/"most"): z0_t = z0 * _Z0H_Z0_RATIO_DEFAULT.  ~0.1 is the typical
# land value (Garratt 1992 §4; the Zilitinkevich 1995 kB^-1 = ln(z0/z0h) family
# spans ~0.01-1 over natural surfaces).  A LARGER ratio raises z0_t, SHRINKS the
# heat log-law denominator ln(z_t/z0_t), RAISES the heat exchange coefficient
# and thereby STRENGTHENS the sensible/latent flux.  COARE 3.0 and large_yeager
# compute their OWN scalar roughness (Fairall smooth-flow Re fit / LY09
# coefficient space) and DO NOT read this ratio.
_Z0H_Z0_RATIO_DEFAULT = 0.1

# Safety floor for the base of the Beljaars-Holtslag psi_h ^{3/2} power. In the
# valid domain (zeta > 0) the base 1 + 2a*zeta/3 >= 1, so the floor is inert; it
# only guards a direct out-of-domain (zeta <= 0) call from a NaN value/gradient
# (fractional power of a non-positive base) -- the "clamp the base before the
# power" AD-safety rule.
_BH91_PSIH_BASE_FLOOR = 1e-6


def surface_reference_state(T, z_ref, z_low=None):
    """Pair model-level height with dry-adiabatic surface-referenced air T.

    Observed forcing (z_low=None) already follows its supplied reference
    convention. Model-level forcing must move BOTH height and temperature.
    """
    if z_low is None:
        return T, z_ref
    # Sensible heat is positive UPWARD: H = rho cp Ch U (T_sfc - T_air).
    # Bringing air down warms it (+g z/cp), reducing upward H; land loses H
    # through -H in its energy budget. Height simultaneously reduces exchange.
    return T + (constants.g / constants.c_pd) * z_low, z_low


def apply_gustiness(u: jax.Array, v: jax.Array, gustiness: float) -> jax.Array:
    """Effective surface wind with a sub-grid convective gustiness floor.

    ``|U|_eff = sqrt(u^2 + v^2 + u_gust^2)``.  The resolved grid-mean wind misses
    sub-grid wind variability — boundary-layer convective gustiness — which in
    calm/convective regions (the tropics, where the mean wind is light but deep
    convection drives gusts) is the dominant contributor to the air-sea latent
    and sensible flux.  Omitting it (or using a ~1 m/s numerical floor) yields
    ~1/3 of the realistic surface evaporation at light winds, starving the
    hydrological cycle (measured: coupled hfls ~35 vs ~80 W/m^2 at a 1.9 m/s
    mean surface wind).  Same algebraic form as the numerical wind floor
    ``sqrt(u^2+v^2+U_min^2)`` — a larger, physically-motivated ``u_gust``
    subsumes it.

    The floor lives INSIDE the single ``sqrt`` (not a two-stage
    ``sqrt(sqrt(u^2+v^2)^2 + g^2)``): with ``gustiness > 0`` the argument is
    strictly positive everywhere, so the gradient is finite even at exact calm
    wind ``u = v = 0`` (a bare ``sqrt(u^2+v^2)`` would give a NaN gradient
    there).  AD-safe for the differentiable coupler.

    Parameters
    ----------
    u, v : array
        Resolved grid-mean surface wind components [m/s].  A caller holding only
        the wind SPEED magnitude ``s`` passes ``apply_gustiness(s, 0.0, gust)``
        (``sqrt(s^2 + gust^2)``), which is equally AD-safe.
    gustiness : float
        Gustiness floor ``u_gust`` [m/s] (~5 m/s, Wing 2018 RCEMIP1).

    Returns
    -------
    array
        Effective wind speed for the bulk flux [m/s].

    References
    ----------
    - Beljaars (1995), QJRMS 121, 255-270 — convective gustiness in bulk fluxes.
    - Wing et al. (2018), GMD 11, 793-813 — RCEMIP1 5 m/s gustiness floor.
    """
    return jnp.sqrt(u ** 2 + v ** 2 + gustiness ** 2)


# Floors for the neutral log-law drag. _LN_RATIO_FLOOR is the same 0.5 the
# iterative solver applies to its own denominator (``_denom_floor`` in
# compute_most_fluxes), so the neutral limit is bounded exactly as the in-loop
# form is; _Z0_FLOOR_M matches the roughness floor used there.
_LN_RATIO_FLOOR = 0.5
_Z0_FLOOR_M = 1.0e-12


def neutral_drag_coefficient(z_ref, z0):
    """Neutral-limit bulk drag coefficient ``Cd = (kappa / ln(z_ref/z0))^2``.

    The canonical home for the neutral log-law drag, so a single-column model,
    the global model and the CRM cannot each grow their own copy. It is the
    zero-stability limit of the profile this module already integrates: with
    ``psi_m = 0`` the iterative solver's ``u* = kappa*U/ln(z_ref/z0)``
    (the ``u_star`` initialisation below) is exactly ``sqrt(Cd)*U``.

    That in-loop expression is deliberately NOT rewritten in terms of this
    helper: ``kappa*U/ln`` and ``sqrt((kappa/ln)^2)*U`` are algebraically equal
    but not bit-identical, and that code path is shared by the ocean, sea-ice,
    land and coupler surface schemes. ``tests`` pins the two forms to agree.

    Parameters
    ----------
    z_ref : array or float
        Height at which the wind is evaluated [m]. Use the height of the model
        level whose wind is actually passed to the flux routine, not a nominal
        10 m, or the drag will be inconsistent with that wind.
    z0 : array or float
        Aerodynamic roughness length [m].

    Returns
    -------
    array
        Neutral drag coefficient [-].
    """
    z_ref = jnp.asarray(z_ref)
    z0 = jnp.asarray(z0, dtype=z_ref.dtype)
    ln_ratio = jnp.log(z_ref / jnp.maximum(z0, _Z0_FLOOR_M))
    return (KAPPA / jnp.maximum(ln_ratio, _LN_RATIO_FLOOR)) ** 2


def validate_bulk_scheme(scheme: str) -> None:
    """Raise ``ValueError`` on an unknown bulk-flux scheme name.

    Every surface-flux dispatcher gates on ``bulk_scheme`` with
    ``if scheme in (<MOST schemes>): ... else: <constant>``.  Without this guard
    a *typo'd* scheme silently falls through to the constant-coefficient branch
    and runs the wrong air-sea physics (CLAUDE.md dispatch rule: factories must
    raise on unknown schemes).  ``scheme`` is a static Python string resolved at
    trace time, so this validates at function entry — never inside a traced /
    ``jit`` body.
    """
    if scheme not in _VALID_BULK_SCHEMES:
        raise ValueError(
            f"Unknown bulk_scheme {scheme!r}; expected one of "
            f"{_VALID_BULK_SCHEMES}."
        )


def validate_stability_scheme(stability_scheme: str) -> None:
    """Raise ``ValueError`` on an unknown stable-regime stability scheme.

    ``stability_scheme`` selects the STABLE-branch (zeta > 0) ``psi_m``/``psi_h``
    forms; it is a static Python string resolved at trace time, so this validates
    at function entry -- never inside a traced / ``jit`` body. Dispatch-hardening
    (CLAUDE.md): a typo'd name must fail LOUDLY rather than silently fall through
    to a default branch and run the wrong surface-layer physics.
    """
    if stability_scheme not in _VALID_STABILITY_SCHEMES:
        raise ValueError(
            f"Unknown stability_scheme {stability_scheme!r}; expected one of "
            f"{_VALID_STABILITY_SCHEMES}."
        )


def large_yeager_neutral_cd(wind, *, nemo_parity: bool = False):
    """Large & Yeager (2009) Eq. 6 neutral 10-m drag coefficient ``C_DN``.

    .. math::
        C_{DN} = \\left(\\frac{2.7}{U} + 0.142 + \\frac{U}{13.09}
                  - 3.14807\\times10^{-10}\\,U^{6}\\right)\\times10^{-3}

    Default (``nemo_parity=False``, the MOST-solver convention): clipped to
    ``[0.5e-3, 3.0e-3]``.  The ``-3.14807e-10·U⁶`` high-wind term is LY09's
    correction over LY04 (which over-estimated drag at ``U > 30 m/s``) and is
    REQUIRED by the OMIP-2 protocol (Griffies 2016 §2.2).  ``wind`` is the
    10-m wind speed [m/s]; floored to 0.5 to avoid the ``1/U`` blow-up at calm
    winds.  The canonical drag law shared by the MOST flux solver and the OMIP-2
    air-sea bulk formulas (no per-component re-derivation).

    ``nemo_parity=True`` reproduces NEMO/aerobulk ``cd_n10_ncar`` EXACTLY
    (sbcblk_algo_ncar.F90): the same polynomial, but (a) a constant cyclone
    plateau ``2.34e-3`` for ``U >= 33 m/s`` instead of letting the ``U⁶`` term
    pull the polynomial down, and (b) ONLY the NEMO floor ``Cx_min = 1e-4`` —
    no upper clip, so the calm-wind ``2.7/U`` enhancement (up to ``5.54e-3``
    at the 0.5 m/s floor) is kept.  The two conventions differ only for
    ``U < ~0.97 m/s`` (where the 3.0e-3 clip binds) and ``U > 33 m/s``.
    """
    U = jnp.maximum(jnp.asarray(wind), 0.5)
    C_DN = (
        2.7 / U + 0.142 + U / 13.09 - 3.14807e-10 * U ** 6
    ) * 1e-3
    if nemo_parity:
        C_DN = jnp.where(U >= 33.0, 2.34e-3, C_DN)
        return jnp.maximum(C_DN, 0.1e-3)
    return jnp.clip(C_DN, 0.5e-3, 3.0e-3)


# ============================================================================
# Stability functions (Businger-Dyer unstable; selectable stable regime)
# ============================================================================
# Convention: zeta = z/L is the Monin-Obukhov stability parameter (L the Obukhov
# length); zeta < 0 unstable, zeta > 0 stable. Every psi satisfies psi(0) = 0
# and, in the stable regime, decreases monotonically (more negative) with
# increasing zeta -- a more-stable column raises the log-law denominator
# ``ln(z/z0) - psi`` and thereby REDUCES the drag/exchange coefficient and the
# surface fluxes (the physically-required stable-regime suppression).
#
# The private ``_<scheme>_psi_{m,h}`` helpers below evaluate ONLY the stable
# branch and are always called with a strictly-positive ``zeta_pos`` (floored at
# 1e-10 by :func:`psi_m` / :func:`psi_h`), so no fractional power / log / cbrt
# ever sees a non-positive argument. Together with the ``jnp.where`` split this
# keeps the masked (inactive) branch finite in BOTH value and gradient -- the
# classic AD-safe double-branch construction.


def _gryanik_psi_m(zeta_pos):
    """Gryanik et al. (2020) stable momentum psi_m(zeta), zeta = zeta_pos > 0."""
    a_m, b_m = _GRYANIK_A_M, _GRYANIK_B_M
    return -3.0 * (a_m / b_m) * (jnp.cbrt(1.0 + b_m * zeta_pos) - 1.0)


def _gryanik_psi_h(zeta_pos):
    """Gryanik et al. (2020) stable heat psi_h(zeta), zeta > 0."""
    a_h, b_h = _GRYANIK_A_H, _GRYANIK_B_H
    return -_GRYANIK_PR0 * (a_h / b_h) * jnp.log1p(b_h * zeta_pos)


def _beljaars_holtslag_psi_m(zeta_pos):
    """Beljaars & Holtslag (1991) stable momentum psi_m(zeta), zeta > 0."""
    a, b, c, d = _BH91_A, _BH91_B, _BH91_C, _BH91_D
    return -(
        a * zeta_pos
        + b * (zeta_pos - c / d) * jnp.exp(-d * zeta_pos)
        + b * c / d
    )


def _beljaars_holtslag_psi_h(zeta_pos):
    """Beljaars & Holtslag (1991) stable heat psi_h(zeta), zeta > 0."""
    a, b, c, d = _BH91_A, _BH91_B, _BH91_C, _BH91_D
    # Clamp the ^{3/2} base to a positive floor BEFORE the power (AD-safety):
    # in-domain (zeta > 0) the base 1 + 2a*zeta/3 >= 1 so the floor is inert.
    base = jnp.maximum(1.0 + (2.0 / 3.0) * a * zeta_pos, _BH91_PSIH_BASE_FLOOR)
    return -(
        base ** 1.5
        + b * (zeta_pos - c / d) * jnp.exp(-d * zeta_pos)
        + b * c / d
        - 1.0
    )


def _grachev_psi_m(zeta_pos):
    """Grachev et al. (2007) SHEBA stable momentum psi_m(zeta), zeta > 0 (Eq. 12)."""
    a_m, b_m = _GRACHEV_A_M, _GRACHEV_B_M
    # Compile-time scalar constants via ``math`` (weakly-typed Python floats, so
    # they never promote the traced float32 ``zeta_pos`` state to float64).
    B_m = (1.0 / b_m - 1.0) ** (1.0 / 3.0)
    sqrt3 = math.sqrt(3.0)
    one_plus_Bm = 1.0 + B_m
    quad_den = 1.0 - B_m + B_m * B_m
    atan_ref = math.atan((2.0 - B_m) / (sqrt3 * B_m))
    x = jnp.cbrt(1.0 + zeta_pos)  # (1 + zeta)^{1/3}
    linear = -3.0 * (a_m / b_m) * (x - 1.0)
    log_1 = 2.0 * jnp.log((x + B_m) / one_plus_Bm)
    log_2 = -jnp.log((x * x - x * B_m + B_m * B_m) / quad_den)
    atan_1 = jnp.arctan((2.0 * x - B_m) / (sqrt3 * B_m))
    bracket = log_1 + log_2 + 2.0 * sqrt3 * (atan_1 - atan_ref)
    return linear + (a_m * B_m) / (2.0 * b_m) * bracket


def _grachev_psi_h(zeta_pos):
    """Grachev et al. (2007) SHEBA stable heat psi_h(zeta), zeta > 0 (Eq. 13)."""
    a_h, b_h, c_h = _GRACHEV_A_H, _GRACHEV_B_H, _GRACHEV_C_H
    B_h = math.sqrt(c_h * c_h - 4.0)  # = sqrt(5); compile-time constant
    coeff = a_h / B_h - (b_h * c_h) / (2.0 * B_h)
    log_ref = math.log((c_h - B_h) / (c_h + B_h))
    fractional_logs = jnp.log(
        (2.0 * zeta_pos + c_h - B_h) / (2.0 * zeta_pos + c_h + B_h)
    ) - log_ref
    quadratic_log = (b_h / 2.0) * jnp.log1p(c_h * zeta_pos + zeta_pos * zeta_pos)
    return _GRACHEV_PR0 * (-coeff * fractional_logs - quadratic_log)


def _stable_psi_m(zeta_pos, stability_scheme, stable_beta=_DYER_STABLE_BETA):
    """Stable-branch (zeta > 0) psi_m for ``stability_scheme``.

    The SINGLE dispatch for the selectable stable momentum similarity
    function, consumed by :func:`psi_m` (Businger-Dyer unstable side) AND by
    :func:`psi_m_coare` (COARE unstable side, non-default schemes only) so
    the published fits are never duplicated.  ``stable_beta`` acts only on
    the linear ``dyer1974`` form.  Unknown scheme -> ``ValueError`` (dispatch
    hardening; static Python string, raises at trace time).
    """
    if stability_scheme == "dyer1974":
        return -stable_beta * zeta_pos
    elif stability_scheme == "beljaars_holtslag1991":
        return _beljaars_holtslag_psi_m(zeta_pos)
    elif stability_scheme == "grachev2007_sheba":
        return _grachev_psi_m(zeta_pos)
    elif stability_scheme == "gryanik2020":
        return _gryanik_psi_m(zeta_pos)
    raise ValueError(
        f"Unknown stability_scheme {stability_scheme!r}; expected one of "
        f"{_VALID_STABILITY_SCHEMES}."
    )


def _stable_psi_h(zeta_pos, stability_scheme, stable_beta=_DYER_STABLE_BETA):
    """Stable-branch (zeta > 0) psi_h for ``stability_scheme``.

    Heat/moisture twin of :func:`_stable_psi_m` (same dispatch contract).
    """
    if stability_scheme == "dyer1974":
        return -stable_beta * zeta_pos
    elif stability_scheme == "beljaars_holtslag1991":
        return _beljaars_holtslag_psi_h(zeta_pos)
    elif stability_scheme == "grachev2007_sheba":
        return _grachev_psi_h(zeta_pos)
    elif stability_scheme == "gryanik2020":
        return _gryanik_psi_h(zeta_pos)
    raise ValueError(
        f"Unknown stability_scheme {stability_scheme!r}; expected one of "
        f"{_VALID_STABILITY_SCHEMES}."
    )


def psi_m(zeta, stability_scheme="dyer1974", *,
          unstable_gamma=_DYER_UNSTABLE_GAMMA,
          stable_beta=_DYER_STABLE_BETA):
    """MOST momentum stability function psi_m(zeta).

    Convention zeta = z/L (>0 stable). Unstable (zeta < 0) is Businger-Dyer for
    every scheme:
        psi_m = 2 ln((1+x)/2) + ln((1+x^2)/2) - 2 arctan(x) + pi/2,
        with x = (1 - gamma zeta)^{1/4}, gamma = ``unstable_gamma`` (default 16).
    Stable (zeta > 0) is selected by ``stability_scheme``:

    - ``"dyer1974"``             : psi_m = -beta zeta (default; historical linear
      form with beta = ``stable_beta``, default 5)
    - ``"beljaars_holtslag1991"``: Beljaars & Holtslag (1991)
    - ``"grachev2007_sheba"``    : Grachev et al. (2007) SHEBA (Arctic/strong-stable)
    - ``"gryanik2020"``          : Gryanik et al. (2020) modified SHEBA

    ``unstable_gamma`` / ``stable_beta`` are the trainable Businger-Dyer
    coefficients (Businger et al. 1971 / Dyer 1974). They act ONLY when
    ``stability_scheme="dyer1974"`` (unstable gamma for zeta < 0, stable beta for
    zeta > 0). For the beljaars/grachev/gryanik schemes BOTH the stable form
    (their own published fit) AND the unstable branch (kept at the historical
    gamma = 16) are UNAFFECTED. Both default to the module constants so every
    existing caller is byte-identical.

    Safe double-branch construction: the unstable expression is evaluated on
    ``zeta_neg <= -1e-10`` and the stable expression on ``zeta_pos >= 1e-10`` so
    the masked branch never produces a NaN value or gradient under ``jnp.where``.
    """
    validate_stability_scheme(stability_scheme)
    zeta_c = jnp.clip(zeta, -10.0, 10.0)
    # Safe inputs: each branch only sees valid arguments
    zeta_neg = jnp.minimum(zeta_c, -1e-10)
    zeta_pos = jnp.maximum(zeta_c, 1e-10)

    # The trainable ``unstable_gamma`` reparameterises the Businger-Dyer UNSTABLE
    # branch ONLY for ``dyer1974`` (the scheme whose stable branch it is paired
    # with).  The beljaars/grachev/gryanik schemes keep the historical
    # gamma = 16 unstable branch (their PUBLISHED fits assume it), so a tuned
    # gamma does not silently perturb their unstable side.  Static Python select
    # on the (compile-time) scheme string -> no traced branch.
    _gamma = unstable_gamma if stability_scheme == "dyer1974" else _DYER_UNSTABLE_GAMMA
    x = jnp.power(1.0 - _gamma * zeta_neg, 0.25)
    unstable = (
        2.0 * jnp.log((1.0 + x) / 2.0)
        + jnp.log((1.0 + x ** 2) / 2.0)
        - 2.0 * jnp.arctan(x)
        + jnp.pi / 2.0
    )

    # Stable branch: static ``stability_scheme`` -> Python dispatch (only the
    # selected expression is traced), not ``jnp.where`` (which would trace all).
    stable = _stable_psi_m(zeta_pos, stability_scheme, stable_beta)

    return jnp.where(zeta_c < 0.0, unstable, stable)


def psi_h(zeta, stability_scheme="dyer1974", *,
          unstable_gamma=_DYER_UNSTABLE_GAMMA,
          stable_beta=_DYER_STABLE_BETA):
    """MOST heat/moisture stability function psi_h(zeta).

    Convention zeta = z/L (>0 stable). Unstable (zeta < 0) is Businger-Dyer for
    every scheme:
        psi_h = 2 ln((1+y)/2),  y = (1 - gamma zeta)^{1/2},
        gamma = ``unstable_gamma`` (default 16).
    Stable (zeta > 0) is selected by ``stability_scheme`` (same options as
    :func:`psi_m`; ``"dyer1974"`` default reproduces the historical
    -``stable_beta`` zeta with beta = 5). ``unstable_gamma`` / ``stable_beta``
    (Businger-Dyer / Dyer 1974 coefficients) act ONLY when
    ``stability_scheme="dyer1974"`` (both branches); the beljaars/grachev/gryanik
    schemes keep their own published stable fits AND the historical gamma = 16
    unstable branch. Both default to the module constants (byte-identical).
    """
    validate_stability_scheme(stability_scheme)
    zeta_c = jnp.clip(zeta, -10.0, 10.0)
    zeta_neg = jnp.minimum(zeta_c, -1e-10)
    zeta_pos = jnp.maximum(zeta_c, 1e-10)

    # See psi_m: the trainable unstable gamma applies ONLY to dyer1974; the
    # non-linear stable schemes keep the historical gamma = 16 unstable branch.
    _gamma = unstable_gamma if stability_scheme == "dyer1974" else _DYER_UNSTABLE_GAMMA
    y = jnp.sqrt(1.0 - _gamma * zeta_neg)
    unstable = 2.0 * jnp.log((1.0 + y) / 2.0)

    stable = _stable_psi_h(zeta_pos, stability_scheme, stable_beta)

    return jnp.where(zeta_c < 0.0, unstable, stable)


def psi_m_coare(zeta, stability_scheme="dyer1974"):
    """COARE 3.0 momentum stability function (Fairall et al. 1996, 2003).

    Unstable (ζ < 0): blend of the Kansas form and the free-convective
    form of Fairall et al. (1996):
        ψ_kansas: x = (1 − 15ζ)^{1/4} (standard Businger-Dyer shape)
        ψ_conv:   y = (1 − 10.15ζ)^{1/3}
                  ψ = 1.5 ln((1 + y + y²)/3) − √3 arctan((1 + 2y)/√3) + π/√3
        blend:    f = ζ²/(1 + ζ²);  ψ = (1 − f) ψ_kansas + f ψ_conv
    Stable (ζ > 0): COARE native stable form (``stability_scheme="dyer1974"``,
    the config default — see below)
        c = min(50, 0.35ζ)
        ψ = −[(1 + ζ) + 0.6667 (ζ − 14.28) e^{−c} + 8.525]

    ``stability_scheme`` swaps ONLY the stable (ζ > 0) branch, so the
    experiment-level ``surface_stability_scheme`` knob is not silently inert
    on a ``bulk_scheme="coare3"`` lane (it was: the 2026-08 dyer-vs-BH AMIP
    A/B was bit-identical because every executed psi call was this function).
    The unstable branch ALWAYS keeps the Fairall Kansas + free-convective
    blend that defines COARE.  Semantics of the selector here:

    - ``"dyer1974"`` (the ``SurfaceLayerConfig`` default) is a SENTINEL for
      "COARE native" and is byte-identical to the pre-selector behaviour.
      COARE 3.0's native stable form above IS the Beljaars & Holtslag (1991)
      fit with rounded constants (Fairall et al. 2003 adopted BH91 for the
      stable side: 0.6667 ≈ b = 2/3, 14.28 ≈ c/d = 5/0.35, 8.525 ≈ b·c/d − 1),
      so there is no meaningful linear ``−5ζ`` COARE variant to expose and the
      default must not silently change.
    - ``"beljaars_holtslag1991"`` swaps to the canonical BH91 constants — a
      rounding-level change vs native (max |Δψ| ≈ 4.5e-3 over ζ ∈ (0, 10]).
    - ``"grachev2007_sheba"`` / ``"gryanik2020"`` give the genuinely
      different strong-stability (SHEBA) tails.

    Safe branching (min/max on inputs) keeps gradients NaN-free in the
    inactive branch, matching :func:`psi_m`.
    """
    # Dispatch hardening AT ENTRY on the static string: without this a typo'd
    # scheme only raises after the whole unstable branch has been traced (and
    # never at all for a caller that reaches this function directly rather
    # than through compute_most_fluxes, which validates for itself).
    validate_stability_scheme(stability_scheme)
    zeta_c = jnp.clip(zeta, -10.0, 10.0)
    zeta_neg = jnp.minimum(zeta_c, -1e-10)
    zeta_pos = jnp.maximum(zeta_c, 1e-10)

    x = jnp.power(1.0 - 15.0 * zeta_neg, 0.25)
    psi_k = (
        2.0 * jnp.log((1.0 + x) / 2.0)
        + jnp.log((1.0 + x ** 2) / 2.0)
        - 2.0 * jnp.arctan(x)
        + jnp.pi / 2.0
    )
    y = jnp.power(1.0 - 10.15 * zeta_neg, 1.0 / 3.0)
    sqrt3 = jnp.sqrt(3.0)
    psi_c = (
        1.5 * jnp.log((1.0 + y + y ** 2) / 3.0)
        - sqrt3 * jnp.arctan((1.0 + 2.0 * y) / sqrt3)
        + jnp.pi / sqrt3
    )
    f = zeta_neg ** 2 / (1.0 + zeta_neg ** 2)
    unstable = (1.0 - f) * psi_k + f * psi_c

    if stability_scheme == "dyer1974":
        # COARE native stable branch (byte-identical default; see docstring).
        c = jnp.minimum(50.0, 0.35 * zeta_pos)
        stable = -(
            (1.0 + zeta_pos)
            + 0.6667 * (zeta_pos - 14.28) * jnp.exp(-c)
            + 8.525
        )
    else:
        stable = _stable_psi_m(zeta_pos, stability_scheme)
    return jnp.where(zeta_c < 0.0, unstable, stable)


def psi_h_coare(zeta, stability_scheme="dyer1974"):
    """COARE 3.0 heat/moisture stability function (Fairall et al. 1996, 2003).

    Unstable: Kansas ψ = 2 ln((1+x)/2), x = (1 − 15ζ)^{1/2}, blended with
    the free-convective form (y = (1 − 34.15ζ)^{1/3}) via f = ζ²/(1+ζ²).
    Stable (native): ψ = −[(1 + 2ζ/3)^{3/2} + 0.6667 (ζ − 14.28) e^{−c} + 8.525].

    ``stability_scheme`` swaps ONLY the stable (ζ > 0) branch; ``"dyer1974"``
    (the config default) is the SENTINEL for the byte-identical COARE native
    form, which is itself the Beljaars & Holtslag (1991) ψ_h with rounded
    constants — see :func:`psi_m_coare` for the full semantics and why.
    """
    validate_stability_scheme(stability_scheme)  # entry dispatch hardening
    zeta_c = jnp.clip(zeta, -10.0, 10.0)
    zeta_neg = jnp.minimum(zeta_c, -1e-10)
    zeta_pos = jnp.maximum(zeta_c, 1e-10)

    x = jnp.sqrt(1.0 - 15.0 * zeta_neg)
    psi_k = 2.0 * jnp.log((1.0 + x) / 2.0)
    y = jnp.power(1.0 - 34.15 * zeta_neg, 1.0 / 3.0)
    sqrt3 = jnp.sqrt(3.0)
    psi_c = (
        1.5 * jnp.log((1.0 + y + y ** 2) / 3.0)
        - sqrt3 * jnp.arctan((1.0 + 2.0 * y) / sqrt3)
        + jnp.pi / sqrt3
    )
    f = zeta_neg ** 2 / (1.0 + zeta_neg ** 2)
    unstable = (1.0 - f) * psi_k + f * psi_c

    if stability_scheme == "dyer1974":
        # COARE native stable branch (byte-identical default; see psi_m_coare).
        c = jnp.minimum(50.0, 0.35 * zeta_pos)
        stable = -(
            jnp.power(1.0 + 2.0 * zeta_pos / 3.0, 1.5)
            + 0.6667 * (zeta_pos - 14.28) * jnp.exp(-c)
            + 8.525
        )
    else:
        stable = _stable_psi_h(zeta_pos, stability_scheme)
    return jnp.where(zeta_c < 0.0, unstable, stable)


# --- COARE 3.0 wind-dependent Charnock ramp (Fairall et al. 2003 §3c) ---
_COARE_CHARNOCK_U_LO = 10.0   # [m/s] U_10N below which charnock stays at base
_COARE_CHARNOCK_U_HI = 18.0   # [m/s] U_10N at/above which charnock = hi value
_COARE_CHARNOCK_HI = 0.018    # charnock value at/above U_HI

# --- Scheme-native gustiness / calm-wind conventions (AeroBulk parity) ---
# COARE 3.0 has convective gustiness BUILT IN (aerobulk mod_blk_coare3p0:
# zi0 = 600 m, Beta0 = 1.25) plus a 0.2 m/s bulk-wind floor; LY09/NEMO ncar
# has NO gustiness but floors the bulk wind at 0.5 m/s.
_COARE_GUSTINESS_ZI = 600.0   # [m] aerobulk coare3p0 zi0 (BL scale height)
_COARE_UB_FLOOR = 0.2         # [m/s] aerobulk coare3p0 bulk-wind floor
_LY_UB_FLOOR = 0.5            # [m/s] LY09 / NEMO sbcblk_algo_ncar wind floor


def resolve_gustiness_w_zi(gustiness_w_zi, scheme) -> float:
    """The EFFECTIVE convective-gustiness BL depth z_i [m] for ``scheme``.

    ``None`` means scheme-native (AeroBulk/COARE parity): COARE 3.0 includes
    convective gustiness as part of the algorithm (zi = 600 m), the other
    schemes do not.  An explicit value (0.0 = off) overrides.  The SINGLE
    source for this resolution — used by :func:`compute_most_fluxes` itself
    and by the air-sea consistency guard, so "what does None mean" can never
    drift between the model and the validator.
    """
    if gustiness_w_zi is None:
        return _COARE_GUSTINESS_ZI if scheme == "coare3" else 0.0
    return float(gustiness_w_zi)


# ============================================================================
# Main MOST flux computation
# ============================================================================

def compute_most_fluxes(
    u_rel,
    v_rel,
    T_atm,
    q_atm,
    T_sfc,
    q_sfc,
    rho,
    z_ref=10.0,
    z_t=None,
    z_q=None,
    z0_init=1e-4,
    scheme="coare3",
    n_iter=5,
    charnock=0.011,
    L_latent=None,
    thermo_convention="legoesm",
    gustiness_w_zi=None,
    gustiness_beta=1.25,
    return_2m=False,
    z_diag=2.0,
    max_exchange_coeff=None,
    stability_scheme="dyer1974",
    return_convergence=False,
    *,
    unstable_gamma=_DYER_UNSTABLE_GAMMA,
    stable_beta=_DYER_STABLE_BETA,
    z0h_z0_ratio=_Z0H_Z0_RATIO_DEFAULT,
):
    """Compute stability-dependent bulk fluxes via iterative MOST.

    Uses ``jax.lax.fori_loop`` for the Obukhov length iteration,
    ensuring full JAX differentiability (jax.grad, jax.jit).

    Parameters
    ----------
    u_rel, v_rel : array
        Wind components relative to z_ref [m/s].
    T_atm : array
        Atmospheric temperature at z_t [K].
    q_atm : array
        Atmospheric specific humidity at z_q [kg/kg].
    T_sfc : array
        Surface temperature [K].
    q_sfc : array
        Surface saturation specific humidity [kg/kg].
    rho : array
        Air density at z_t [kg/m³].
    z_ref : float
        Reference height for the wind / momentum [m] (default 10).
    z_t : float or None
        Reference height for atmospheric temperature [m]. Defaults to
        ``z_ref`` (single-height mode).  For OMIP-2 / JRA55-do use **10.0**:
        the v1.4.0 files carry an explicit ``height = 10.0 m`` coordinate on
        ``tas`` and ``huss`` as well as on ``uas``/``vas`` (the CF
        ``comment`` string "usually, 2 meter" is CMOR-table boilerplate that
        contradicts the file's own coordinate), and FESOM2 forces the same
        dataset with ``ncar_bulk_z_tair = ncar_bulk_z_shum = 10.0``.  This
        docstring previously said 2.0; declaring the 10 m state at 2 m
        inflates the air-sea gradients by ~10 % of the turbulent fluxes.
    z_q : float or None
        Reference height for atmospheric specific humidity [m]. Defaults
        to ``z_ref`` (single-height mode).  10.0 for OMIP-2 / JRA55-do, per
        the note on ``z_t``.
    z0_init : float
        Initial momentum roughness length [m] (default 1e-4).
    scheme : str
        ``"constant"``, ``"most"``, ``"coare3"`` or ``"large_yeager"``.  The
        fixed-roughness log-law path (``"constant"``/``"most"``) uses
        ``z0_init`` (and ``z0h_z0_ratio`` for the scalar roughness) directly;
        ``coare3``/``large_yeager`` evolve the roughness in the iteration.
    n_iter : int
        Number of MOST iterations (default 5).
    charnock : float
        Charnock coefficient (COARE only, default 0.011).
    thermo_convention : str
        Constants set converting MOST scales into fluxes (#762):
        ``"legoesm"`` (default) = constant ``L_v`` / dry ``c_pd``;
        ``"aerobulk"`` = the NEMO/AeroBulk/COARE convention
        (SST-dependent ``L_vap(T_sfc)``, moist ``cp_air(q_atm)``) — up to
        ~3 % LH at warm SST and ~1-2 % SH in the humid tropics.  An
        explicit ``L_latent`` overrides the L choice either way.
    gustiness_w_zi : float or None
        COARE convective-gustiness BL depth z_i [m].  None (default) =
        scheme-native: 600 m for ``"coare3"`` (AeroBulk/Fairall 2003 —
        gustiness is part of the algorithm), 0 (off) for the others.
        Explicit 0.0 disables for any scheme.
    max_exchange_coeff : float or None
        Optional physical ceiling on the neutral-equivalent bulk transfer
        coefficient ``C = κ²/(denom_m·denom_h)``.  ``None`` (default) leaves
        the MOST log-law denominators floored at the standard ``0.5``
        (``C ≤ κ²/0.25 ≈ 0.64``), i.e. BYTE-IDENTICAL to the prior behaviour
        for every existing (ocean/atmosphere) caller.  When set to ``C_max``,
        each denominator is instead floored at ``max(0.5, κ/√C_max)`` so the
        implied ``C_d``, ``C_h`` and ``C_e`` cannot exceed ``C_max``.  This
        caps ``u*``, ``θ*`` and ``q*`` CONSISTENTLY (the fluxes stay linear in
        the T/q gradients, so a surface-energy-balance solve retains its
        self-limiting feedback) and removes the extreme-instability cold-start
        singularity where the floored ``0.5`` denominator lets a trivial
        ~0.6 g/kg humidity gradient generate a spurious ~4900 W/m² latent
        shock (``C_e≈0.64`` vs a physical ~3.4e-3).  Intended for the land
        surface tile, whose stiff thin top layer at ``dt_rad`` is unstable to
        that shock; ocean/atmosphere callers never approach the floor.
    stability_scheme : str
        Stable-regime (zeta > 0) similarity functions ``psi_m``/``psi_h``.
        ``"dyer1974"`` (default) is BYTE-IDENTICAL to the prior behaviour for
        every existing caller: the historical linear ``-5 zeta`` on the
        ``constant``/``most``/``large_yeager`` Businger-Dyer path, and the
        COARE NATIVE stable form on ``coare3`` (which is itself the Beljaars &
        Holtslag 1991 fit with rounded constants — Fairall et al. 2003 §3
        adopted BH91 for the stable side; see :func:`psi_m_coare`).  The
        non-linear alternatives ``"beljaars_holtslag1991"``,
        ``"grachev2007_sheba"`` and ``"gryanik2020"`` do not collapse the fluxes
        to zero under strong stability (Arctic sea-ice / nocturnal SBL); on
        ``coare3`` they swap ONLY the stable branch (the Fairall unstable blend
        is COARE-defining and always kept), so on that scheme
        ``"beljaars_holtslag1991"`` is a rounding-level change vs the default
        and the genuinely different tails are grachev/gryanik. The unstable
        branch (zeta < 0) stays Businger-Dyer (Fairall blend for coare3)
        regardless. Validated at function entry (a typo raises ``ValueError``).
    unstable_gamma : float
        Businger-Dyer / Dyer (1974) UNSTABLE-branch coefficient gamma in
        ``x = (1 - gamma zeta)^{1/4}`` (psi_m) / ``y = (1 - gamma zeta)^{1/2}``
        (psi_h) for zeta < 0.  Default 16 (byte-identical to the prior
        behaviour).  A LARGER gamma makes psi more positive under instability =>
        smaller log-law denominator => larger exchange coefficient => stronger
        fluxes.  Applies to the Businger-Dyer unstable branch used by the
        ``constant``/``most``/``large_yeager`` (non-COARE) path ONLY when
        ``stability_scheme="dyer1974"``; the non-linear stable schemes keep the
        historical gamma = 16 unstable branch and COARE 3.0 keeps its own Fairall
        (1996/2003) unstable coefficients.
    stable_beta : float
        Dyer (1974) STABLE-branch linear coefficient beta in
        ``psi = -beta zeta`` for zeta > 0, used ONLY when
        ``stability_scheme="dyer1974"``.  Default 5 (byte-identical).  A LARGER
        beta makes psi more negative under stability => larger log-law
        denominator => smaller exchange coefficient => weaker fluxes.  Ignored by
        the non-linear ``beljaars_holtslag1991``/``grachev2007_sheba``/
        ``gryanik2020`` stable forms (their own published fits) and by COARE 3.0.
    z0h_z0_ratio : float
        Thermal/momentum roughness ratio z0h/z0 for the FIXED-roughness path:
        z0_t = z0_q = z0_init * z0h_z0_ratio.  Default 0.1 (byte-identical to
        the prior hardcoded ``z0 * 0.1``).  Consumed ONLY by the
        ``constant``/``most`` (log-law fixed-roughness) branch.  COARE 3.0 and
        large_yeager compute their own scalar roughness in the loop and are
        EXACTLY invariant to this argument: their pre-loop z0_t seed is pinned to
        the historical 0.1 so a finite-``n_iter`` seed residual cannot leak the
        ratio into the ocean-scheme fluxes.  A LARGER ratio => larger z0_t =>
        smaller ln(z_t/z0_t) denom_h => larger heat exchange coefficient =>
        STRONGER sensible/latent flux (Garratt 1992; Zilitinkevich kB^-1 range
        ~0.01-1).
    return_convergence : bool
        When True, additionally return the MOST fixed-point convergence
        residual (the relative change in ``u*`` over the FINAL iteration).
        Default False = byte-identical to the prior signature.  The Obukhov
        iteration is a fixed ``n_iter`` ``fori_loop`` (AD-safe — a
        tolerance-based ``while_loop`` would break reverse-mode ``grad``), so
        this residual is the convergence CHECK: it is ~0 where the fixed
        iteration count converged and O(1) in the strong-stability columns
        where it did not (the gap vs a tolerance-guaranteed root solve, e.g.
        CliMA SurfaceFluxes.jl).  It does not affect the fluxes.

    Returns
    -------
    tau_x, tau_y : array
        Surface stress [Pa] (opposes wind direction).
    shflx : array
        Sensible heat flux [W/m²] (positive upward = surface warmer).
    lhflx : array
        Latent heat flux [W/m²] (positive upward = surface moister).
    ustar : array
        Friction velocity [m/s].
    most_residual : array
        ONLY when ``return_convergence=True`` (appended last, after ``T_2m``
        if ``return_2m`` is also set): the final-iteration relative ``u*``
        change [1], per column.
    """
    # Dispatch hardening (CLAUDE.md): ``scheme`` is a static Python string
    # resolved at trace time. Validate it at function entry so a typo'd name
    # (e.g. ``"coar3"``) fails LOUDLY instead of silently falling through the
    # ``else`` branch below to the constant-roughness MOST path and running the
    # wrong air-sea physics. ``coare3``/``large_yeager`` take dedicated
    # branches; ``constant``/``most`` are the (valid) fixed-roughness else path.
    validate_bulk_scheme(scheme)
    if scheme == "large_yeager_cesm":
        raise ValueError(
            "compute_most_fluxes does not implement 'large_yeager_cesm' (the "
            "CESM shr_flux_atmOcn law has its own fixed-iteration solver): "
            "call compute_sam_oceflx_fluxes(..., variant='cesm') — the "
            "surface-layer / coupler dispatchers do."
        )
    # Dispatch hardening (#762): the thermodynamic-convention selector is a
    # static string — a typo must fail LOUDLY, never silently run the other
    # constants set.
    if thermo_convention not in ("legoesm", "aerobulk"):
        raise ValueError(
            "thermo_convention must be 'legoesm' or 'aerobulk', got "
            f"{thermo_convention!r}"
        )
    # ``stability_scheme`` selects the STABLE-branch psi_m/psi_h; also a static
    # string, so validate it once here rather than inside the traced loop body.
    validate_stability_scheme(stability_scheme)

    # Scheme-native gustiness default (AeroBulk/COARE parity, static Python
    # resolved at trace time) via the single shared resolver — the air-sea
    # consistency guard uses the same one.
    gustiness_w_zi = resolve_gustiness_w_zi(gustiness_w_zi, scheme)

    # Resolve scalar reference heights. ``z_ref`` is the wind/momentum
    # height (always = z_u in the formulas below); z_t, z_q default to
    # z_ref so that single-height callers (lake, idealized adapter,
    # legacy tests) keep their existing behaviour bit-identically.
    z_u = z_ref
    if z_t is None:
        z_t = z_ref
    if z_q is None:
        z_q = z_ref

    wind_speed = jnp.sqrt(u_rel ** 2 + v_rel ** 2 + 1e-4)
    dT = T_sfc - T_atm
    dq = q_sfc - q_atm
    # Virtual-T moisture coefficient = 1/ε − 1 ≈ 0.6078 (canonical, not 0.61).
    _vT_coef = 1.0 / constants.epsilon - 1.0
    T_v = T_atm * (1.0 + _vT_coef * q_atm)

    # Log-law denominator floor. Default 0.5 (=> C ≤ κ²/0.25 ≈ 0.64, the
    # historical behaviour, byte-identical for every existing caller). When a
    # physical ceiling ``max_exchange_coeff`` is requested, floor each
    # denominator at κ/√C_max so the implied C_d/C_h/C_e ≤ C_max — a static
    # Python float, constant-folded, no retrace (feature-gating, not traced
    # selection). See the ``max_exchange_coeff`` docstring entry.
    if max_exchange_coeff is None:
        _denom_floor = 0.5
        _coeff_cap = None
    else:
        # Static feature-gate value (constant-folded). A traced/array scalar
        # would break the Python ``max``/``if`` below, so require a real Python
        # or numpy scalar and reject it loudly rather than silently mis-tracing.
        # ``bool`` is a numbers.Real subclass — exclude it explicitly. Then
        # guard finite > 0: C_max = 0 gives an infinite floor (silent zero
        # fluxes) and C_max < 0 gives ``sqrt`` of a negative (NaN).
        if isinstance(max_exchange_coeff, bool) or not isinstance(
            max_exchange_coeff, numbers.Real
        ):
            raise TypeError(
                "max_exchange_coeff must be a static real scalar or None, "
                f"got {type(max_exchange_coeff).__name__}"
            )
        max_exchange_coeff = float(max_exchange_coeff)
        if not (math.isfinite(max_exchange_coeff) and max_exchange_coeff > 0.0):
            raise ValueError(
                "max_exchange_coeff must be finite and > 0, "
                f"got {max_exchange_coeff}"
            )
        _denom_floor = max(0.5, KAPPA / (max_exchange_coeff ** 0.5))
        # Coefficient-space equivalent ceiling for the large_yeager branch,
        # which forms rd/rh/re directly (u*=rd·U, θ*=rh·dT, q*=re·dq) and never
        # touches the log-law denominators.  The log-law path caps the
        # "rd-equivalent" κ/denom at κ/_denom_floor, so bounding rd, rh, re at
        # the SAME value gives C_d=rd² , C_h=rd·rh , C_e=rd·re ≤ C_max
        # identically.  = min(√C_max, 0.8) so it never tightens below the
        # historical 0.5-floor behaviour.
        _coeff_cap = KAPPA / _denom_floor

    # Initialize with neutral log-law profile. The thermal (scalar) roughness
    # z0_t = z0_q = z0 * z0h_z0_ratio (default 0.1) is the trainable knob for
    # the FIXED-roughness "constant"/"most" path — that path carries the init
    # z0_t straight through to the flux via the log law.
    #
    # COARE 3.0 and large_yeager compute their OWN scalar roughness inside the
    # loop (Fairall smooth-flow Re fit / LY09 coefficient space), so z0h_z0_ratio
    # is not a physical parameter for them.  BUT the pre-loop z0_t also seeds the
    # first-iteration theta*/q* -> theta_v* -> Obukhov length -> zeta -> psi, and
    # with a finite ``n_iter`` (no exact convergence) that seed leaves a small
    # residual in the ocean-scheme fluxes.  To keep those schemes EXACTLY
    # invariant to z0h_z0_ratio (and byte-identical to the pre-#z0h behaviour),
    # seed their z0_t with the historical constant ratio, applying the tunable
    # ratio ONLY on the fixed-roughness path.  Static Python select on the
    # (compile-time) scheme string -> no traced branch, constant-folded.
    _init_z0h_ratio = (
        z0h_z0_ratio if scheme in ("constant", "most")
        else _Z0H_Z0_RATIO_DEFAULT
    )
    z0 = jnp.full_like(wind_speed, z0_init)
    z0_t = z0 * _init_z0h_ratio
    z0_q = z0_t

    ln_zu_z0 = jnp.log(z_u / jnp.maximum(z0, 1e-12))
    ln_zt_z0t = jnp.log(z_t / jnp.maximum(z0_t, 1e-12))
    ln_zq_z0q = jnp.log(z_q / jnp.maximum(z0_q, 1e-12))
    u_star = KAPPA * wind_speed / jnp.maximum(ln_zu_z0, _denom_floor)
    theta_star = KAPPA * dT / jnp.maximum(ln_zt_z0t, _denom_floor)
    q_star_val = KAPPA * dq / jnp.maximum(ln_zq_z0q, _denom_floor)

    # Carry leaves 7-8 (orthogonal MOST diagnostics unioned across branches):
    #  (7) U_eff, the bulk/effective wind incl. gustiness + scheme calm-wind
    #      floor, so the post-loop stress normalization tau = rho u*^2 u/U_eff
    #      (AeroBulk semantics: tau = rho Cd Ub u) uses the CONVERGED value;
    #      with gustiness off and no floor binding, U_eff == wind_speed
    #      byte-identically.  Seeded with the pre-loop wind_speed.
    #  (8) the MOST convergence residual (relative u* change), updated each
    #      iteration; seeded at 1.0 ("not yet converged").
    resid0 = jnp.ones_like(u_star)
    carry = (u_star, z0, z0_t, z0_q, theta_star, q_star_val, wind_speed, resid0)
    # The MOST iteration mixes the (possibly float32) input state with float64
    # physical constants (G, NU_AIR, c_pd via the virtual-T coefficient), so a
    # carry leaf would silently promote float32 -> float64 mid-loop and trip
    # ``fori_loop``'s equal-types invariant.  This only bites the float32
    # atmosphere coupled path; the OMIP ocean path runs float64 so the re-casts
    # below are no-ops (byte-identical).  Pin each leaf back to its input dtype.
    _carry_dtypes = tuple(c.dtype for c in carry)

    def body_fn(i, carry):
        u_star, z0, z0_t, z0_q, theta_star, q_star_val, _U_eff_prev, _resid = carry
        u_star_safe = jnp.maximum(u_star, 1e-6)

        # Virtual potential temperature scale (1/ε − 1 ≈ 0.6078)
        theta_v_star = theta_star + _vT_coef * T_atm * q_star_val

        # COARE 3.0 convective gustiness (scheme-native default: ON for
        # coare3 with z_i = 600 m per AeroBulk/Fairall 2003, OFF otherwise;
        # explicit gustiness_w_zi=0.0 disables).  Over
        # a calm but convectively-unstable warm ocean the mean wind alone gives
        # an anemic flux (the tropical hfls ~45 vs ~120 W/m² bias); the
        # free-convection velocity scale w* = (g·z_i·<w'θv'>/θv)^(1/3) adds a
        # sub-grid gust U_eff = sqrt(|U|² + (β·w*)²) (Fairall et al. 2003,
        # β~1.25, z_i = BL depth ~600 m).  <w'θv'> = u*·θv* (kinematic, upward
        # +; unstable only).
        if gustiness_w_zi > 0.0:
            wpthvp = jnp.maximum(u_star_safe * theta_v_star, 0.0)
            # cbrt(0) has an infinite derivative, so a stable column (buoyancy
            # flux floored to exactly 0) would inject a NaN reverse-mode
            # gradient into every trainable upstream (charnock, state).  Floor
            # the cbrt argument at a tiny positive constant: the max() kink
            # clamps the stable-side gradient to 0 (physically correct — no
            # free convection when stable) and the primal offset is w* ~ 1e-8
            # m/s, negligible next to the resolved wind.  (1e-24 <= 1e-6 safety
            # floor, exempt from the inline-coeff ratchet.)
            wstar = jnp.cbrt(
                jnp.maximum(G * gustiness_w_zi * wpthvp / T_v, 1e-24)
            )
            U_eff = jnp.sqrt(wind_speed ** 2 + (gustiness_beta * wstar) ** 2)
        else:
            U_eff = wind_speed

        # Scheme-native calm-wind floors on the bulk (effective) wind, per
        # the reference implementations (see module constants above).
        if scheme == "coare3":
            U_eff = jnp.maximum(U_eff, _COARE_UB_FLOOR)
        elif scheme == "large_yeager":
            U_eff = jnp.maximum(U_eff, _LY_UB_FLOOR)

        # Inverse Obukhov length: 1/L = −κ g θ_v* / (u*² T_v).
        #
        # The stability parameter ζ = z/L is the *only* way L enters this
        # solver, so we carry the reciprocal directly.  The reciprocal form
        # is singularity-free: the denominator u*² T_v is strictly positive
        # (u_star_safe ≥ 1e-6, T_v > 0), so there is no divide-by-zero and
        # no need for a sentinel or ``safe_divide``.  Crucially it gives the
        # correct neutral limit *and* its derivative: at exact neutral
        # (θ_v* = 0) ⇒ 1/L = 0 ⇒ ζ = 0 with the true finite sensitivity
        # dζ/dθ_v* = −z κ g /(u*² T_v).  Forming L = −u*²T_v/(κgθ_v*) first
        # and then z/L would instead either inject a NaN gradient (0·∞ from
        # the dead branch of a neutral-limit ``where``) or, if masked with a
        # constant ±1e6 sentinel, flatten dζ/dθ_v* to zero across the
        # neutral band and jump at the mask threshold.  Away from neutral ζ
        # equals the textbook z/L to round-off.
        inv_L = -KAPPA * G * theta_v_star / (u_star_safe ** 2 * T_v)

        # Stability parameters and ψ functions evaluated at each
        # measurement height. When z_t == z_q == z_u (single-height),
        # zeta_t == zeta_q == zeta_u and psi_h_t == psi_h_q.
        zeta_u = jnp.clip(z_u * inv_L, -10.0, 10.0)
        zeta_t = jnp.clip(z_t * inv_L, -10.0, 10.0)
        zeta_q = jnp.clip(z_q * inv_L, -10.0, 10.0)
        # Scheme-matched stability functions (static Python dispatch at trace
        # time): COARE 3.0 uses the Fairall 1996/2003 Kansas + free-convective
        # blend on the unstable side and the ``stability_scheme``-selectable
        # STABLE branch (default "dyer1974" = the byte-identical COARE native
        # stable form — itself BH91, see psi_m_coare); large_yeager/constant/
        # most keep Businger-Dyer with the same selectable stable branch
        # ("dyer1974" default = the historical -5*zeta that LY09/NEMO ncar
        # use).  Without threading the selector here the knob was silently
        # inert on every coare3 lane (the 2026-08 bit-identical AMIP A/B).
        if scheme == "coare3":
            psi_m_u = psi_m_coare(zeta_u, stability_scheme)
            psi_h_t = psi_h_coare(zeta_t, stability_scheme)
            psi_h_q = psi_h_coare(zeta_q, stability_scheme)
        else:
            psi_m_u = psi_m(zeta_u, stability_scheme,
                            unstable_gamma=unstable_gamma, stable_beta=stable_beta)
            psi_h_t = psi_h(zeta_t, stability_scheme,
                            unstable_gamma=unstable_gamma, stable_beta=stable_beta)
            psi_h_q = psi_h(zeta_q, stability_scheme,
                            unstable_gamma=unstable_gamma, stable_beta=stable_beta)

        # --- Roughness update (Python if resolved at trace time) ---
        if scheme == "coare3":
            # COARE 3.0: wind-dependent Charnock + smooth-flow regime.
            # Fairall et al. (2003) §3c ramp the Charnock parameter linearly
            # from the base value at U_10N <= 10 m/s to 0.018 at >= 18 m/s;
            # without it the high-wind drag is biased ~15% low (found by the
            # aerobulk coare3p0 oracle).  ``charnock`` may be a traced
            # trainable parameter, so use jnp ops throughout.
            U_10N_c = u_star_safe / KAPPA * jnp.log(
                10.0 / jnp.maximum(z0, 1e-12)
            )
            ramp = jnp.clip(
                (U_10N_c - _COARE_CHARNOCK_U_LO)
                / (_COARE_CHARNOCK_U_HI - _COARE_CHARNOCK_U_LO),
                0.0, 1.0,
            )
            charnock_hi = jnp.maximum(_COARE_CHARNOCK_HI, charnock)
            charnock_eff = charnock + ramp * (charnock_hi - charnock)
            z0_new = (
                charnock_eff * u_star_safe ** 2 / G
                + 0.11 * NU_AIR / u_star_safe
            )
            Re = u_star_safe * z0_new / NU_AIR
            z0_t_new = jnp.minimum(
                1.15e-4,
                5.5e-5 * jnp.power(jnp.maximum(Re, 0.01), -0.6),
            )
            z0_q_new = z0_t_new

        elif scheme == "large_yeager":
            # Large & Yeager 2009 (CORE/OMIP): iterate in coefficient space.
            # 1) Neutral 10-m wind from current u_star and z0
            U_10N = u_star_safe / KAPPA * jnp.log(
                10.0 / jnp.maximum(z0, 1e-12)
            )
            U_10N = jnp.clip(U_10N, 0.5, 50.0)

            # 2) LY09 neutral 10-m drag coefficient — the canonical shared drag
            # law (Large & Yeager 2009 Eq. 6, incl. the −3.14807e-10·U⁶ high-wind
            # correction + clip), NOT a per-component re-derivation.  U_10N is
            # already clipped to [0.5, 50], so the helper's 0.5 floor is a no-op.
            C_DN = large_yeager_neutral_cd(U_10N)

            # 3) Neutral exchange coefficients at 10 m
            rdn = jnp.sqrt(C_DN)
            # Stability-dependent 10-m Stanton/Dalton number (LY09 Table 4).
            # Use the wind-height stability for the unstable/stable branch.
            CHN10 = jnp.where(zeta_u < 0.0, 32.7e-3, 18.0e-3) * rdn
            CEN10 = 34.6e-3 * rdn
            rhn = CHN10 / rdn  # = ch_coeff
            ren = CEN10 / rdn  # = ce_coeff

            # 4) Shift the NEUTRAL 10-m coefficients to the measurement
            #    height AND the actual stability (LY04 Eq. 9-11 / LY09 §3;
            #    identical to NEMO sbcblk_algo_ncar):
            #    rd = rdn / (1 + rdn/κ · (ln(z_u/10) − ψ_m(z_u/L)))
            #    rh = rhn / (1 + rhn/κ · (ln(z_t/10) − ψ_h(z_t/L)))
            #    re = ren / (1 + ren/κ · (ln(z_q/10) − ψ_h(z_q/L)))
            #    The reference coefficient is neutral-10-m, so the stability
            #    term is ψ(z/L) alone.  Subtracting ψ(10/L) here (the old
            #    form) cancels the entire stability correction at z = 10 m
            #    and silently runs near-neutral exchange in stable/unstable
            #    air — caught by the aerobulk oracle (stable light-wind
            #    fluxes were up to ~10x too large).
            ln_zr_u = jnp.log(z_u / 10.0)
            ln_zr_t = jnp.log(z_t / 10.0)
            ln_zr_q = jnp.log(z_q / 10.0)

            rd = rdn / jnp.maximum(
                1.0 + rdn / KAPPA * (ln_zr_u - psi_m_u), 0.2
            )
            rh = rhn / jnp.maximum(
                1.0 + rhn / KAPPA * (ln_zr_t - psi_h_t), 0.2
            )
            re = ren / jnp.maximum(
                1.0 + ren / KAPPA * (ln_zr_q - psi_h_q), 0.2
            )

            # Optional transfer-coefficient ceiling (max_exchange_coeff). The LY
            # coefficient-space path bypasses the log-law denominator floor, so
            # apply the equivalent cap directly: rd, rh, re ≤ κ/_denom_floor
            # bounds C_d=rd², C_h=rd·rh, C_e=rd·re ≤ C_max — the SAME guarantee
            # the else-branch gets from _denom_floor. _coeff_cap is None (skip,
            # byte-identical) unless a ceiling was requested. z0_new below reads
            # the capped rd, so the roughness estimate stays consistent.
            if _coeff_cap is not None:
                rd = jnp.minimum(rd, _coeff_cap)
                rh = jnp.minimum(rh, _coeff_cap)
                re = jnp.minimum(re, _coeff_cap)

            # 5) Update scaling parameters directly from coefficients
            u_star_new = rd * U_eff
            theta_star_new = rh * dT
            q_star_new = re * dq

            # Still need z0 for the next iteration's U_10N estimate
            z0_new = z_u / jnp.exp(KAPPA / rd + psi_m_u)
            z0_new = jnp.clip(z0_new, 1e-12, 1.0)
            z0_t_new = z0_t  # not used in coefficient path
            z0_q_new = z0_q  # not used in coefficient path

            # Convergence residual (relative u* change) — see the common
            # return below; kept in sync so the carry structure matches.
            resid_new = jnp.abs(u_star_new - u_star) / jnp.maximum(
                jnp.abs(u_star_new), 1e-6)
            return (u_star_new, z0_new, z0_t_new, z0_q_new,
                    theta_star_new, q_star_new, U_eff, resid_new)

        else:
            z0_new = z0
            z0_t_new = z0_t
            z0_q_new = z0_q

        # For COARE and constant: transfer coefficients via log-law + stability,
        # with each variable referenced to its own measurement height.
        ln_zu_z0 = jnp.log(z_u / jnp.maximum(z0_new, 1e-12))
        ln_zt_z0t = jnp.log(z_t / jnp.maximum(z0_t_new, 1e-12))
        ln_zq_z0q = jnp.log(z_q / jnp.maximum(z0_q_new, 1e-12))

        denom_m = jnp.maximum(ln_zu_z0 - psi_m_u, _denom_floor)
        denom_h = jnp.maximum(ln_zt_z0t - psi_h_t, _denom_floor)
        denom_q = jnp.maximum(ln_zq_z0q - psi_h_q, _denom_floor)

        u_star_new = KAPPA * U_eff / denom_m
        theta_star_new = KAPPA * dT / denom_h
        q_star_new = KAPPA * dq / denom_q

        # MOST fixed-point convergence residual: the relative change in u*
        # over this iteration (small = converged).  DIAGNOSTIC ONLY — it does
        # not feed the fluxes, so the default (return_convergence=False) path
        # is byte-identical; it lets callers/tests detect columns where the
        # fixed n_iter under-converges (strong stability), the gap vs a
        # tolerance-based root solve (CliMA SurfaceFluxes.jl).
        resid_new = jnp.abs(u_star_new - u_star) / jnp.maximum(
            jnp.abs(u_star_new), 1e-6)

        return (u_star_new, z0_new, z0_t_new, z0_q_new,
                theta_star_new, q_star_new, U_eff, resid_new)

    def _body_fn_dtype_stable(i, carry):
        out = body_fn(i, carry)
        return tuple(jnp.asarray(o).astype(d)
                     for o, d in zip(out, _carry_dtypes))

    carry = jax.lax.fori_loop(0, n_iter, _body_fn_dtype_stable, carry)
    u_star, z0, z0_t, z0_q, theta_star, q_star_val, U_eff_final, most_residual = carry

    # Fluxes from scaling parameters.  Stress normalization: tau =
    # -rho u*^2 u/U_eff == -rho Cd U_eff u (AeroBulk/COARE: one factor of the
    # bulk wind incl. gust/floor, one raw wind component for direction and
    # magnitude); reduces to u/|U| exactly when U_eff == wind_speed.
    # Thermodynamic convention (#762) now selects only the heat capacity:
    # 'aerobulk' = NEMO/AeroBulk moist cp_air(q); 'legoesm' = dry c_pd.  The
    # latent heat is the Kirchhoff L_v(T_sfc) in BOTH (user decision
    # 2026-09-28); an explicit ``L_latent`` always wins (ice passes L_s, the
    # OMIP NEMO-parity path and the oracle tests inject their own).
    from legoesm.thermo import latent_heat_vaporization
    _L = latent_heat_vaporization(T_sfc) if L_latent is None else L_latent
    if thermo_convention == "aerobulk":
        from legoesm.thermo import moist_air_cp
        _cp = moist_air_cp(q_atm)
    else:
        _cp = constants.c_pd
    tau_x = -rho * u_star ** 2 * u_rel / U_eff_final
    tau_y = -rho * u_star ** 2 * v_rel / U_eff_final
    shflx = rho * _cp * u_star * theta_star
    lhflx = rho * _L * u_star * q_star_val

    if return_2m:
        # Air temperature at the diagnostic height (default 2 m) from the
        # converged MOST similarity profile: T(z) = T_sfc − (θ*/κ)·[ln(z/z0t) −
        # ψ_h(z/L)], which reduces to T_atm at z=z_t.  Over a warm ocean the
        # lowest model level (~100 m at nlev=20) reads colder than 2 m, so the
        # raw lowest-level "tas" exaggerates the cold/air-sea-gap bias — this
        # gives the physically correct CMIP 2 m value.  Recompute 1/L from the
        # converged scales (the iteration carries scales, not L).
        u_star_safe = jnp.maximum(u_star, 1e-6)
        theta_v_star = theta_star + _vT_coef * T_atm * q_star_val
        inv_L = -KAPPA * G * theta_v_star / (u_star_safe ** 2 * T_v)
        zeta_d = jnp.clip(z_diag * inv_L, -10.0, 10.0)
        # Scheme-match the diagnostic psi_h to the MAIN loop (static Python
        # dispatch): COARE 3.0 uses the Fairall free-convective psi_h_coare
        # WITH the same stability_scheme-selected stable branch as the main
        # loop, every other scheme uses the stability_scheme-selected psi_h.
        # Using the non-COARE psi_h here for coare3 made the 2 m T
        # inconsistent with the converged coare3 profile.
        _psi_h_d = (psi_h_coare(zeta_d, stability_scheme)
                    if scheme == "coare3"
                    else psi_h(zeta_d, stability_scheme,
                               unstable_gamma=unstable_gamma,
                               stable_beta=stable_beta))
        denom_d = jnp.log(z_diag / jnp.maximum(z0_t, 1e-12)) - _psi_h_d
        T_2m = T_sfc - (theta_star / KAPPA) * denom_d
        # Guard against profile extrapolation outside [T_atm, T_sfc].
        lo = jnp.minimum(T_atm, T_sfc)
        hi = jnp.maximum(T_atm, T_sfc)
        T_2m = jnp.clip(T_2m, lo, hi)
        if return_convergence:
            return tau_x, tau_y, shflx, lhflx, u_star, T_2m, most_residual
        return tau_x, tau_y, shflx, lhflx, u_star, T_2m

    if return_convergence:
        return tau_x, tau_y, shflx, lhflx, u_star, most_residual
    return tau_x, tau_y, shflx, lhflx, u_star


def sam_ocean_surface_q(
    T_sfc: jnp.ndarray,
    p_sfc: jnp.ndarray | float,
    salt_factor: float = 0.981,
) -> jnp.ndarray:
    """SAM ocean surface saturation specific humidity (``oceflx`` ``qs``).

    SAM sets the ocean surface humidity to the SATURATION value at the SST,
    reduced by a salinity factor (``qs = salt_factor · qsat(SST, p_sfc)``,
    ``salt_factor = 0.981`` ≈ the 2 % saturation reduction over saline
    ocean water).  At SST = 300 K, p_sfc = 1014.8 hPa this is ≈ 0.0217
    kg/kg — about 20 % larger than the hardcoded ``q_sfc = 0.018`` the
    RCEMIP plane harness used, so the latent-heat flux (and the whole
    WISHE feedback / RCE moisture budget) was biased low.

    Uses the shared :func:`legoesm.thermo.saturation_specific_humidity`
    (CLAUDE.md — no re-implemented Clausius–Clapeyron).

    Parameters
    ----------
    T_sfc : array
        Sea-surface temperature [K].
    p_sfc : array or float
        Surface pressure [Pa].
    salt_factor : float
        Salinity reduction of saturation humidity (SAM default 0.981).

    Returns
    -------
    array
        Surface saturation specific humidity [kg/kg].
    """
    from legoesm.thermo import saturation_specific_humidity

    return salt_factor * saturation_specific_humidity(T_sfc, p_sfc)


def ocean_surface_q_sat(
    T_sfc: jnp.ndarray,
    p_sfc: jnp.ndarray | float,
    *,
    thermo_convention: str = "legoesm",
    bulk_scheme: str = "constant",
    saline_factor: float = 1.0,
) -> jnp.ndarray:
    """Air-sea surface saturation MIXING RATIO with the convention-appropriate
    saturation curve (#762): the single q_sfc source shared by the coupler
    ocean tile and the standalone slab ocean, so the interface humidity is no
    longer split between Goff (coupler) and Tetens (slab) on the aerobulk MOST
    path.  (The atmosphere tiled surface-layer q_sfc is a separate site tracked
    for the same unification.)

    The WMO Goff (1957) curve is used under ``thermo_convention='aerobulk'`` on
    a MOST solver scheme (``'most'``/``'coare3'``/``'large_yeager'`` — the only
    schemes carrying the NEMO/AeroBulk constant set, matching the flux-formation
    convention); every other case keeps the Tetens
    :func:`legoesm.thermo.saturation_mixing_ratio`, so the default ``legoesm``
    convention and the fixed-coefficient ``constant`` closure are byte-identical.
    ``saline_factor`` applies the salinity reduction of q_sat (0.98 for the
    coupled ocean tile; 1.0 leaves it unscaled).

    Parameters
    ----------
    T_sfc : array
        Sea-surface temperature [K].
    p_sfc : array or float
        Surface pressure [Pa].
    thermo_convention : str
        ``'legoesm'`` (Tetens) or ``'aerobulk'`` (Goff on the MOST schemes).
    bulk_scheme : str
        Surface bulk scheme; Goff engages only on the MOST solvers.
    saline_factor : float
        Salinity reduction of q_sat (1.0 = none).

    Returns
    -------
    array
        Surface saturation mixing ratio [kg/kg].
    """
    from legoesm.thermo import (
        saturation_mixing_ratio, saturation_mixing_ratio_goff)

    _use_goff = (thermo_convention == "aerobulk"
                 and bulk_scheme in ("most", "coare3", "large_yeager"))
    _sat = saturation_mixing_ratio_goff if _use_goff else saturation_mixing_ratio
    return saline_factor * _sat(T_sfc, p_sfc)


# --- CESM coupler air-sea flux law (cime5.6.47 shr_flux_mod.F90) ---
# ``seq_flux_atmocn_minwind`` namelist default (CESM2 namelist_defaults_cam.xml).
_CESM_MINWIND = 0.5


def _sam_cdn(u10: jnp.ndarray) -> jnp.ndarray:
    """SAM/CESM neutral 10 m drag coefficient (``oceflx.f90`` ``cdn``).

    ``cdn(U) = 0.0027/U + 0.000142 + 0.0000764·U``.  ``U`` is floored at
    the SAM minimum wind ``umin = 1 m/s`` so the ``1/U`` term stays bounded
    (and AD-safe).
    """
    u = jnp.maximum(u10, 1.0)
    return _cesm_cdn(u)


def _cesm_cdn(u10: jnp.ndarray) -> jnp.ndarray:
    """CESM ``shr_flux_mod`` ``cdn(Umps)`` — the same Large & Pond fit with NO
    wind floor (the caller floors ``vmag`` at ``seq_flux_atmocn_minwind``)."""
    return 0.0027 / u10 + 0.000142 + 0.0000764 * u10


def compute_sam_oceflx_fluxes(
    u_atm: jnp.ndarray,
    v_atm: jnp.ndarray,
    theta_atm: jnp.ndarray,
    q_atm: jnp.ndarray,
    T_sfc: jnp.ndarray,
    q_sfc: jnp.ndarray,
    rho: jnp.ndarray,
    z_bot: jnp.ndarray | float,
    exner_sfc: jnp.ndarray | float = 1.0,
    wd: jnp.ndarray | float = 0.0,
    n_iter: int = 2,
    *,
    variant: str = "sam",
    minwind: float = _CESM_MINWIND,
    return_2m: bool = False,
    z_diag: float = 2.0,
) -> tuple[jnp.ndarray, ...]:
    """SAM ocean surface fluxes — faithful port of ``oceflx.f90`` (CESM1).

    ``variant`` selects the oracle (static Python string, dispatch-hardened):

    * ``"sam"`` (default, byte-identical): gSAM ``oceflx.f90`` — ``vmag``
      floored at 1 m/s, the same floor inside ``cdn``, dry ``c_pd``.
    * ``"cesm"``: CIME ``shr_flux_mod.F90::shr_flux_atmOcn`` (cime5.6.47, the
      CAM6/CESM2 coupler law) — ``vmag`` floored at ``minwind``
      (``seq_flux_atmocn_minwind``, CESM2 default 0.5 m/s), NO floor inside
      ``cdn``, and the moist ``cp = c_pd (1 + cpvir ssq)`` = ``c_pd + (c_pv -
      c_pd) q_sfc`` on the sensible heat.  Stability iteration, coefficients
      and flux algebra are otherwise identical (``flux_con_tol = 0``,
      ``flux_con_max_iter = 2`` -> exactly ``n_iter = 2`` passes;
      ``gust_fac = 0``; cold-air-outbreak modification off).  ``ssq`` is
      taken from ``q_sfc`` (the caller's saturation curve and salinity
      factor; CESM's own ``0.98 * 640380/exp(5107.4/T)`` fit is NOT
      re-derived here).  With ``return_2m=True`` the CESM ``tref`` 2 m
      temperature diagnostic (potential-to-temperature corrected) is appended.

    Iterative Monin–Obukhov bulk scheme with SAM's exact neutral transfer
    coefficients and Businger–Dyer stability functions:

    - ``vmag = max(1.0, sqrt(|U|² + wd²))`` — 1 m/s minimum, optional gust
      enhancement ``wd`` (SAM default 0; NOT the Wing-2018 5 m/s floor).
    - Neutral drag ``rdn² = cdn(U10)`` (:func:`_sam_cdn`).
    - Neutral Stanton ``rhn = 0.0327`` (unstable) / ``0.018`` (stable);
      neutral Dalton ``ren = 0.0346``.
    - ``n_iter`` (SAM uses 2) stability iterations: Obukhov ratio
      ``hol = κ g z (t*/θ + q*/(1/ε+q)) / u*²`` (clipped |hol|≤10), then
      the coefficients are shifted to the model level via
      ``r = rn / (1 + rn/κ·(ln(z/z_ref) − ψ))``.

    Fluxes are returned in legoESM's UPWARD-POSITIVE W/m² convention
    (surface warmer/moister ⇒ positive), matching
    :func:`simple_bulk_fluxes`:

        SHF = −ρ c_pd u* t*,   LHF = −ρ L_v u* q*,   τ = ρ u*²

    Parameters
    ----------
    u_atm, v_atm : array
        Lowest-level wind components [m/s].
    theta_atm : array
        Lowest-level POTENTIAL temperature [K] (SAM ``thbot``).
    q_atm : array
        Lowest-level specific humidity [kg/kg].
    T_sfc : array
        Surface temperature [K].  The surface POTENTIAL temperature
        ``θ_sfc = T_sfc / exner_sfc`` is formed for the sensible-heat
        gradient (Codex iter-5 HIGH: ``exner_sfc`` differs from 1 by
        ~0.4 % over the ocean ⇒ ~1 K offset in ``θ_sfc``, comparable to
        the air–sea disequilibrium and able to flip the stable/unstable
        classification near neutral — so do NOT use ``T_sfc`` as
        ``θ_sfc`` directly).
    q_sfc : array
        Surface saturation specific humidity [kg/kg] (saturation at the
        absolute SST; see :func:`sam_ocean_surface_q`).
    rho : array
        Air density at the lowest level [kg/m³].
    z_bot : array or float
        Lowest model-level height [m] (SAM ``zbot``).
    exner_sfc : array or float
        Surface Exner function ``(p_sfc/p_ref)^κ`` used to convert
        ``T_sfc`` to the surface potential temperature.  Default 1.0
        reproduces SAM's ``ts``-as-``θ`` approximation.
    wd : array or float
        Gust-enhancement wind [m/s] added in quadrature to ``vmag``
        (SAM ``wd``, default 0).
    n_iter : int
        Number of MO stability iterations (SAM uses 2).

    Returns
    -------
    tau_x, tau_y : array
        Surface stress [Pa] (opposes wind).
    shflx : array
        Sensible heat flux [W/m²] (positive upward = surface warmer).
    lhflx : array
        Latent heat flux [W/m²] (positive upward = surface moister).
    ustar : array
        Friction velocity [m/s].
    """
    if variant not in ("sam", "cesm"):
        raise ValueError(
            f"Unknown oceflx variant {variant!r}; expected 'sam' or 'cesm'.")
    karman = KAPPA
    eps_v = 1.0 / constants.epsilon - 1.0
    z_ref = 10.0
    umin = 1.0 if variant == "sam" else minwind
    _cdn = _sam_cdn if variant == "sam" else _cesm_cdn

    # Safe sqrt (+eps) so the calm-wind (u=v=wd=0) reverse-mode gradient
    # stays finite — bare sqrt(0) gives a 0·inf NaN even though the value
    # is floored to umin (Codex iter-5). eps ≪ umin² so the primal is
    # unchanged to round-off.
    vmag = jnp.maximum(
        umin, jnp.sqrt(u_atm ** 2 + v_atm ** 2 + wd ** 2 + 1e-12)
    )
    theta_sfc = T_sfc / exner_sfc             # surface potential temperature
    delt = theta_atm - theta_sfc             # SAM thbot - ts (pot-T diff)
    delq = q_atm - q_sfc                      # spec-humidity diff
    alz = jnp.log(z_bot / z_ref)

    # Businger–Dyer unstable stability integrals (SAM psimhu / psixhu).
    def psimhu(xd):
        return (
            jnp.log((1.0 + xd * (2.0 + xd)) * (1.0 + xd * xd) / 8.0)
            - 2.0 * jnp.arctan(xd) + 1.571
        )

    def psixhu(xd):
        return 2.0 * jnp.log((1.0 + xd * xd) / 2.0)

    # --- neutral first guess ---
    # Fortran ``sign(0.5,x)`` treats x==0 as POSITIVE ⇒ stable=1 at exact
    # neutral; ``where(x>=0,1,0)`` matches that (0.5+0.5*sign would give
    # 0.5 — a non-SAM averaged branch).
    stable = jnp.where(delt >= 0.0, 1.0, 0.0)
    rdn = jnp.sqrt(_cdn(vmag))
    rhn = (1.0 - stable) * 0.0327 + stable * 0.018
    ren = jnp.full_like(jnp.asarray(rdn), 0.0346)
    u_star = rdn * vmag
    t_star = rhn * delt
    q_star = ren * delq

    # --- MO stability iterations (SAM does 2) ---
    for _ in range(n_iter):
        hol = (
            karman * G * z_bot
            * (t_star / theta_atm + q_star / (1.0 / eps_v + q_atm))
            / jnp.maximum(u_star ** 2, 1e-12)
        )
        # jnp.clip (not sign*min(abs)) clamps hol to [-10, 10] with the SAME
        # forward values but a SMOOTH, nonzero gradient in (-10, 10): the old
        # sign*min(abs) form has a zero-gradient flat spot at hol=0 (neutral)
        # that severs d(flux)/d(state) sensitivity there under jax.grad.
        hol = jnp.clip(hol, -10.0, 10.0)
        stable = jnp.where(hol >= 0.0, 1.0, 0.0)
        xsq = jnp.maximum(jnp.sqrt(jnp.abs(1.0 - 16.0 * hol)), 1.0)
        xqq = jnp.sqrt(xsq)
        psimh = -5.0 * hol * stable + (1.0 - stable) * psimhu(xqq)
        psixh = -5.0 * hol * stable + (1.0 - stable) * psixhu(xqq)
        # Shift wind, recompute neutral coeffs at u10n, then shift all.
        rd = rdn / (1.0 + rdn / karman * (alz - psimh))
        u10n = vmag * rd / rdn
        rdn = jnp.sqrt(_cdn(u10n))
        rhn = (1.0 - stable) * 0.0327 + stable * 0.018
        ren = jnp.full_like(jnp.asarray(rdn), 0.0346)
        rd = rdn / (1.0 + rdn / karman * (alz - psimh))
        rh = rhn / (1.0 + rhn / karman * (alz - psixh))
        re = ren / (1.0 + ren / karman * (alz - psixh))
        u_star = rd * vmag
        t_star = rh * delt
        q_star = re * delq

    # --- fluxes (legoESM upward-positive W/m²) ---
    tau = rho * u_star ** 2
    tau_x = -tau * u_atm / vmag
    tau_y = -tau * v_atm / vmag
    # CESM: cp = cpdair*(1 + cpvir*ssq), cpvir = cpwv/cpdair - 1.
    cp = (constants.c_pd + (constants.c_pv - constants.c_pd) * q_sfc
          if variant == "cesm" else constants.c_pd)
    shflx = -rho * cp * u_star * t_star
    lhflx = -rho * constants.L_v * u_star * q_star
    if not return_2m:
        return tau_x, tau_y, shflx, lhflx, u_star
    # CESM ``tref`` (shr_flux_atmOcn diagnostics block): re-evaluate the
    # stability function at ztref, integrate the theta profile from zbot down
    # to ztref, then the 0.01 K/m potential-to-temperature correction.
    al2 = jnp.log(z_ref / z_diag)
    hol2 = hol * z_diag / z_bot
    xsq2 = jnp.maximum(1.0, jnp.sqrt(jnp.abs(1.0 - 16.0 * hol2)))
    xqq2 = jnp.sqrt(xsq2)
    psix2 = -5.0 * hol2 * stable + (1.0 - stable) * psixhu(xqq2)
    fac = (rh / karman) * (alz + al2 - psixh + psix2)
    tref = theta_atm - delt * fac - 0.01 * z_diag  # coeff-ok: CESM 0.01 K/m theta->T
    return tau_x, tau_y, shflx, lhflx, u_star, tref


def nemo_si3_constant_fluxes(
    u_air: jnp.ndarray,
    v_air: jnp.ndarray,
    theta_air: jnp.ndarray,
    q_air: jnp.ndarray,
    T_ice: jnp.ndarray,
    p_surface: jnp.ndarray,
    rho_air: jnp.ndarray,
    Cd: float,
    Ch: float,
    Ce: float,
) -> tuple[jnp.ndarray, ...]:
    """Executing constant-coefficient SI3 air--ice bulk core.

    Transcribes NEMO 5.0.2 ``sbcblk.F90:1085-1168,1231-1273`` and
    ``sbc_phy.F90:321-358,665-790``.  ``theta_air`` is already potential
    temperature, as at NEMO's ``blk_ice_1/2`` boundary.  Returned stress uses
    NEMO's air-to-ice sign; callers adapting to legoESM's atmospheric reaction
    convention negate it explicitly.

    Returns ``(tau_x, tau_y, wind, theta_ice, q_sat, dq_sat_dT,
    sensible, latent, dq_sensible_dT, dq_latent_dT)``.
    """
    from legoesm.thermo import nemo_si3_saturation_over_ice

    wind = jnp.sqrt(u_air * u_air + v_air * v_air)
    theta_ice = T_ice * (
        constants.p_ref / p_surface
    ) ** (constants.R_gas_molar / (constants.M_dry_air * constants.c_p_dry_air_nemo))
    q_sat, dq_sat_dT = nemo_si3_saturation_over_ice(T_ice, p_surface)
    rho_wind = rho_air * wind
    stress_scale = rho_wind * Cd
    tau_x = stress_scale * u_air
    tau_y = stress_scale * v_air
    sensible_scale = rho_wind * constants.c_p_air_ice_nemo * Ch
    latent_scale = rho_wind * constants.L_sub_nemo * Ce
    sensible = sensible_scale * (theta_ice - theta_air)
    latent = latent_scale * (q_sat - q_air)
    dq_sensible_dT = sensible_scale
    dq_latent_dT = latent_scale * dq_sat_dT
    return (
        tau_x, tau_y, wind, theta_ice, q_sat, dq_sat_dT,
        sensible, latent, dq_sensible_dT, dq_latent_dT,
    )


def simple_bulk_fluxes(
    u_lowest: jnp.ndarray,
    v_lowest: jnp.ndarray,
    T_lowest: jnp.ndarray,
    q_lowest: jnp.ndarray,
    T_sfc: jnp.ndarray,
    q_sfc: jnp.ndarray,
    rho: jnp.ndarray,
    wind_speed: jnp.ndarray,
    Cd: float,
    Ch: float,
    L_latent: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Simple bulk-aerodynamic surface fluxes with constant coefficients.

    Parameters
    ----------
    u_lowest, v_lowest : array
        Lowest-level wind components [m/s].
    T_lowest : array
        Lowest-level air temperature [K].
    q_lowest : array
        Lowest-level specific humidity [kg/kg].
    T_sfc : array
        Surface temperature [K].
    q_sfc : array
        Surface specific humidity [kg/kg].
    rho : array
        Air density at lowest level [kg/m3].
    wind_speed : array
        Wind speed (with minimum floor applied) [m/s].
    Cd : float
        Drag coefficient for momentum.
    Ch : float
        Transfer coefficient for heat and moisture.

    Returns
    -------
    tau_x, tau_y : array
        Surface stress [Pa] (opposes wind).
    shflx : array
        Sensible heat flux [W/m2] (positive upward = surface warmer).
    lhflx : array
        Latent heat flux [W/m2] (positive upward = surface moister).
    """
    from legoesm.thermo import latent_heat_vaporization
    _L = latent_heat_vaporization(T_sfc) if L_latent is None else L_latent
    tau_x = -rho * Cd * wind_speed * u_lowest
    tau_y = -rho * Cd * wind_speed * v_lowest
    shflx = rho * constants.c_pd * Ch * wind_speed * (T_sfc - T_lowest)
    lhflx = rho * _L * Ch * wind_speed * (q_sfc - q_lowest)
    return tau_x, tau_y, shflx, lhflx
