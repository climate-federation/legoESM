"""Physically-based aerosol activation (Abdul-Razzak & Ghan 2000, "ARG").

This module adds a **prognostic, physically-based** cloud-droplet activation
scheme alongside the climatological AOD→CDNC proxy in
:mod:`aerosol_activation` (Andreae 2009 power law).  The proxy carries no
size distribution, composition, or updraft dependence; the ARG scheme closes
the aerosol → cloud-droplet-number link through an explicit supersaturation
balance over a set of lognormal aerosol modes, exactly the gap flagged in the
proxy module docstring and present in CliMA CloudMicrophysics.jl.

Given an updraft velocity ``w``, temperature ``T``, pressure ``p`` and a set of
internally-mixed lognormal aerosol modes — each with number concentration
``N_a`` [1/m^3], geometric-mean **dry** radius ``r_g`` [m], geometric standard
deviation ``sigma_g`` [-] and hygroscopicity ``kappa`` [-] — the scheme solves
for the maximum in-cloud supersaturation ``S_max`` and the activated number
fraction per mode, summing to the total cloud-droplet number concentration
(CDNC).

Governing relations (pure ARG2000; verified term-by-term against CliMA
CloudMicrophysics.jl ``src/AerosolActivation.jl`` + ``src/Common.jl`` and the
paper):

  Koehler curvature (Kelvin) term      A     = 2 sigma_w / (rho_w R_v T)          [m]
  mode critical supersaturation        S_m,i = (2/sqrt(kappa_i)) (A/(3 r_g,i))^{3/2}
  thermodynamic coefficient            alpha = g L_v /(c_p R_v T^2) - g/(R_d T)    [1/m]
  water-supply coefficient             gamma = R_v T / e_s + eps L_v^2/(c_p p T)   [m^3/kg]
  condensation growth coefficient      G     = 1/[rho_w R_v T/(e_s D_v)
                                              + rho_w L_v/(k_a T)(L_v/(R_v T)-1)]   [m^2/s]
  ARG dimensionless groups             zeta  = (2 A/3) sqrt(alpha w / G)
                                       eta_i = (alpha w/G)^{3/2}/(2 pi rho_w gamma N_i)
  mode-width factors                   f_i   = 0.5 exp(2.5 ln^2 sigma_i)
                                       g_i   = 1 + 0.25 ln sigma_i
  maximum supersaturation
      1/S_max^2 = sum_i (1/S_m,i^2)[ f_i (zeta/eta_i)^{3/2}
                                   + g_i (S_m,i^2/(eta_i + 3 zeta))^{3/4} ]
  activated fraction of mode i         frac_i = 0.5 erfc( 2 ln(S_m,i/S_max)
                                                        / (3 sqrt(2) ln sigma_i) )
  total droplet number                 CDNC   = sum_i N_i frac_i                    [1/m^3]

Only the pure-ARG2000 limit is implemented (no pre-existing-hydrometeor
competition factor, no ``p_v/p_vs`` prefactor, no moist ``R_m``): these are
CliMA refinements that recover the textbook form when ``N_liq=N_ice=0`` and
``RH=1`` and are documented as the natural next step.

Faithfulness
------------
Oracle = the ARG2000 paper closed form.  The ALGEBRAIC STRUCTURE of the S_max
balance is a term-for-term transcription of the on-disk gSAM/M2005 reference
code ``MICRO_M2005/module_mp_graupel.f90:2307-2340`` (``IACT=2``, non-
equilibrium "DROPLET ACTIVATION FROM ABDUL-RAZZAK AND GHAN (2000)" path).  Each
symbol maps onto a gSAM line:

  A/AACT, alpha/ALPHA, gamma/GAMM, G/GG, zeta/PSI, eta_i/ETA, S_m,i/SM,
  f_i/F11, g_i/F21, the (1/S_m^2)[f(zeta/eta)^1.5 + g(S_m^2/(eta+3 zeta))^0.75]
  summand/DUM1, S_max/SMAX, the standard-erf activated fraction/(1-DERF1(UU)).

The multi-mode ``sum_i`` here is the ARG2000 Part-2 generalization of gSAM's
fixed two-mode (NANEW1/NANEW2) block.  This is a STRUCTURAL (form) match, NOT a
numerical/full-scheme reproduction: legoESM evaluates the same forms with its
own constants and thermodynamics, so the numbers differ from gSAM's.
``tests/unit/test_arg_activation_faithful.py`` pins legoESM's OWN closed form
against an independent NumPy transcription (rel 1e-9, single- AND two-mode) and
canaries the f/g/exponent constants against their literals.

SELECTED differences (not exhaustive; legoESM follows the ARG2000 PAPER / CliMA
constant-coefficient form, gSAM/M2005 substitutes empirical fits — documented,
NOT pinned; only the SIGVL and 3*sqrt(2) items below carry a test canary):
  * Kelvin term uses a CONSTANT surface tension ``constants.sigma_water``
    (=0.0728 N/m) via :func:`.._warm_rain.kelvin_coefficient`, vs gSAM's
    T-ramped ``SIGVL = 0.0761 - 1.55e-4*(T - T_freeze)`` [N/m].  (canaried)
  * the erfc denominator uses the EXACT ``3*sqrt(2)`` (=4.24264...), vs gSAM's
    rounded literal ``4.242``; legoESM is closer to the paper here.  (canaried)
  * ``G`` uses CONSTANT ``constants.D_vapor``/``constants.k_air`` (the ARG/CliMA
    published form, see :func:`condensation_growth_coeff_G`), vs gSAM's
    T,p-dependent ``DV`` and T-dependent ``KAP`` (=1.414e3*MU, Sutherland MU(T)).
  * saturation ``e_s`` is the shared legoESM Tetens curve
    (:func:`..thermo.saturation_vapor_pressure`), vs gSAM's POLYSVP Flatau
    polynomial.
  * gSAM's molar constants ``MW=0.018``/``MA=0.0284``/``RR=8.3187`` give
    effective ``R_d~=293``, ``epsilon~=0.634`` — legoESM uses ``constants.R_d``
    / ``constants.epsilon`` (both a few % smaller) in alpha/gamma.
  * ``S_m`` maps onto gSAM ``SM`` only under the identification
    ``kappa_i = BACT``; gSAM fixes ``BACT`` from a prescribed ammonium-sulfate
    composition (~0.509), whereas ``kappa_i`` is a per-mode config field.
  * the driving updraft: legoESM takes ``w`` directly (defaulting to a
    characteristic ``w_char_m_s``); gSAM builds ``DUM`` from the resolved plus
    an optional sub-grid velocity (0.10 m/s minimum), with cloud-water / upward-
    motion / ``IBASE`` cloud-edge gating and a SEPARATE equilibrium-interior
    activation path — none of that column logic is modelled here.
  * CDNC is per-VOLUME [1/m^3]; gSAM carries per-MASS number and divides by rho.
  * :func:`arg_cdnc` returns the DIAGNOSTIC total activated number; gSAM emits a
    source-only tendency ``max(0,(N_act - N_c)/dt)`` capped at the total aerosol
    (the ``>= N_c`` guard + cap are a coupling choice, not the S_max algebra).

References
----------
Abdul-Razzak, H. & Ghan, S. J. (2000): A parameterization of aerosol
activation, 2. Multiple aerosol types, J. Geophys. Res., 105(D5), 6837-6844,
doi:10.1029/1999JD901161.
Petters, M. D. & Kreidenweis, S. M. (2007): A single parameter representation
of hygroscopic growth and cloud condensation nucleus activity (kappa-Koehler),
Atmos. Chem. Phys., 7, 1961-1971, doi:10.5194/acp-7-1961-2007.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics._warm_rain import kelvin_coefficient
from legoesm.thermo import saturation_vapor_pressure

__physics_contract__ = {
    "summary": (
        "Physically-based aerosol activation (Abdul-Razzak & Ghan 2000): "
        "solve the ARG maximum-supersaturation balance over lognormal "
        "aerosol modes -> activated cloud-droplet number (CDNC)."
    ),
    "inputs": {
        "w": "m/s (updraft velocity, > 0)",
        "T": "K (air temperature)",
        "p": "Pa (air pressure)",
        "mode_number": "1/m^3 (per-mode aerosol number concentration >= 0)",
        "mode_r_g": "m (per-mode geometric-mean dry radius > 0)",
        "mode_sigma_g": "1 (per-mode geometric standard deviation > 1)",
        "mode_kappa": "1 (per-mode hygroscopicity > 0)",
    },
    "outputs": {"cdnc": "1/m^3 (activated cloud-droplet number concentration)"},
    "sign_convention": (
        "CDNC >= 0 and monotonically increasing in updraft w and in aerosol "
        "number N_a; activated fraction in [0, 1]; higher N_a lowers the "
        "activated FRACTION (supersaturation competition)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Abdul-Razzak & Ghan (2000), JGR 105(D5), 6837-6844, "
        "doi:10.1029/1999JD901161; kappa-Koehler: Petters & Kreidenweis "
        "(2007), ACP 7, 1961-1971"
    ),
    "idealized_test": (
        "single accumulation mode N_a=100 cm^-3, r_g=0.05 um, sigma_g=2.0, "
        "kappa=0.6, w=0.3 m/s, T=285 K, p=95000 Pa -> CDNC ~ 70 cm^-3 "
        "(activated fraction ~0.7); monotone in N_a and w; frac in [0,1]"
    ),
}

__param_spec__ = {
    "ActivationConfig": {
        "scheme_key": "atm.aerosol.ActivationConfig",
        "excluded": {},
        "params": {
            # Characteristic sub-grid updraft used to drive ARG when a
            # resolved / TKE-derived vertical velocity is not threaded to the
            # activation call site (standard GCM practice).
            "w_char_m_s": {
                "units": "m/s", "bounds": (0.01, 3.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "aerosol_activation",
                "reference": "ARG (2000) characteristic activation updraft",
                "shape": None,
            },
        },
    },
}

# --- ARG2000 lognormal mode-integral factors (Abdul-Razzak & Ghan 2000; ----
# --- coefficients match CliMA ClimaParams ARG2000_{f,g}_coeff_{1,2}) --------
# f_i = _ARG_F_PREFACTOR * exp(_ARG_F_LOGSQ_COEFF * ln^2 sigma_i)
# g_i = 1 + _ARG_G_LOG_COEFF * ln sigma_i
_ARG_F_PREFACTOR = 0.5           # ARG2000 f-coefficient 1
_ARG_F_LOGSQ_COEFF = 2.5         # ARG2000 f-coefficient 2 (multiplies ln^2 sigma)
_ARG_G_LOG_COEFF = 0.25          # ARG2000 g-coefficient 2 (multiplies ln sigma)
# S_max closed-form exponents (ClimaParams ARG2000_pow_1 / ARG2000_pow_2):
_ARG_SMAX_POW_1 = 1.5            # exponent of (zeta/eta_i) in the S_max sum
_ARG_SMAX_POW_2 = 0.75           # exponent of S_m^2/(eta_i + 3 zeta) in the sum

# --- unit conversions (exact) ---------------------------------------------
_PER_CM3_TO_PER_M3 = 1.0e6       # number concentration cm^-3 -> m^-3
_UM_TO_M = 1.0e-6                # radius micrometre -> metre

# --- AD / numerics floors (keep sqrt/log/erfc arguments away from 0) -------
_W_MIN_M_S = 1.0e-6              # updraft floor: alpha*w/G > 0 so eta, zeta finite
_N_A_FLOOR_M3 = 1.0             # per-mode number floor [1/m^3] (eta ~ 1/N_i)
_R_G_FLOOR_M = 1.0e-10          # dry-radius floor [m] (A/(3 r_g) finite)
_KAPPA_FLOOR = 1.0e-4          # hygroscopicity floor (S_m ~ 1/sqrt(kappa))
_SIGMA_G_FLOOR = 1.0 + 1.0e-3  # geometric-std floor (> 1; ln sigma > 0)
_INV_S2_FLOOR = 1.0e-30        # floor on sum 1/S_max^2 so S_max finite
_S_RATIO_FLOOR = 1.0e-30       # floor on S_m/S_max so ln(.) finite


class ActivationConfig(NamedTuple):
    """Aerosol -> cloud-droplet-number activation configuration.

    Selects between the climatological AOD proxy (``scheme="proxy"``, the
    current default — byte-identical to :func:`aerosol_activation.ccn_from_aod`
    / :func:`aerosol_activation.specified_nc_field`) and the physically-based
    ARG2000 modal activation (``scheme="arg"``).

    The ARG aerosol population is a set of lognormal modes stored as equal-
    length tuples (mode ``i`` = ``mode_number_cm3[i]`` etc.).  Number is given
    in cm^-3 and radius in micrometres for readable configs; both are converted
    to SI inside :func:`arg_cdnc`.

    Fields
    ------
    scheme : str
        ``"proxy"`` (default) or ``"arg"``.
    proxy : CCNFromAODConfig
        Configuration for the proxy path (unused when ``scheme="arg"``).
    mode_number_cm3, mode_r_g_um, mode_sigma_g, mode_kappa : tuple of float
        Per-mode lognormal parameters: number concentration [cm^-3],
        geometric-mean DRY radius [um], geometric standard deviation [-] (> 1),
        and kappa-Koehler hygroscopicity [-].  Equal length = number of modes.
        Default is one marine-ish accumulation mode.
    w_char_m_s : float
        Characteristic sub-grid updraft [m/s] used by :func:`activated_nc_field`
        when no resolved / TKE-derived updraft is threaded to the call site.
    """

    scheme: str = "proxy"
    # Nested proxy config; ``None`` resolves to the default ``CCNFromAODConfig``
    # lazily in the dispatch (avoids a module-level import-ordering coupling).
    proxy: object = None
    mode_number_cm3: tuple = (100.0,)
    mode_r_g_um: tuple = (0.05,)
    mode_sigma_g: tuple = (2.0,)
    mode_kappa: tuple = (0.6,)
    w_char_m_s: float = 0.3


def _default_proxy_config():
    """Return the default :class:`CCNFromAODConfig` (lazy import)."""
    from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
        CCNFromAODConfig,
    )
    return CCNFromAODConfig()


def _broadcast_mode(mode_1d: jnp.ndarray, field: jnp.ndarray) -> jnp.ndarray:
    """Reshape a per-mode 1-D array to broadcast against ``field``.

    ``field`` carries the spatial shape (e.g. ``(ncol, nlev)``); the returned
    array has shape ``(n_modes,) + (1,) * field.ndim`` so per-mode quantities
    broadcast cleanly and the mode axis (0) can be reduced.
    """
    return mode_1d.reshape((mode_1d.shape[0],) + (1,) * field.ndim)


def kohler_curvature_A(T: jnp.ndarray) -> jnp.ndarray:
    """Koehler curvature (Kelvin) length ``A = 2 sigma_w/(rho_w R_v T)`` [m].

    Thin alias for the SINGLE canonical :func:`.._warm_rain.kelvin_coefficient`
    (also used by fast-SBM nucleation, oracle AKOE) so the Kelvin term cannot
    drift between activation schemes.  Uses the specific gas constant of water
    vapour ``R_v = R/M_w`` (equivalent to CliMA ``2 sigma M_w/(rho_w R T)``).
    """
    return kelvin_coefficient(T)


def mode_critical_supersaturation(
    A: jnp.ndarray, r_g: jnp.ndarray, kappa: jnp.ndarray,
) -> jnp.ndarray:
    """kappa-Koehler critical supersaturation of a mode [fraction].

    ``S_m = (2/sqrt(kappa)) (A/(3 r_g))^{3/2}`` (ARG2000 Eq. for S_m with the
    kappa-Koehler ``B_i = kappa_i`` identification, Petters & Kreidenweis 2007).
    ``r_g`` is the geometric-mean DRY radius [m].
    """
    r_safe = jnp.maximum(r_g, _R_G_FLOOR_M)
    k_safe = jnp.maximum(kappa, _KAPPA_FLOOR)
    return (2.0 / jnp.sqrt(k_safe)) * (A / (3.0 * r_safe)) ** 1.5


def condensation_growth_coeff_G(
    T: jnp.ndarray, e_s: jnp.ndarray,
) -> jnp.ndarray:
    """Diffusional droplet-growth coefficient ``G`` [m^2/s].

    ``G = 1/[ rho_w R_v T/(e_s D_v) + rho_w L_v/(k_a T)(L_v/(R_v T) - 1) ]``
    (ARG2000 Eq. 16 / Pruppacher-Klett; the rho_w factor is applied here
    exactly once).  ``D_v`` = water-vapour diffusivity in air, ``k_a`` =
    thermal conductivity of air, ``e_s`` = saturation vapour pressure over
    liquid.

    RELATED IMPLEMENTATIONS (same F_D+F_K denominator, different refinements
    — kept separate DELIBERATELY, do not silently unify without a benchmark):
    ``fast_sbm/diffusional_growth.py`` uses a T,p-dependent ``D_v`` (oracle
    D_MY; ~2x this constant value at 500 hPa) + ventilation, and
    ``sdm/condensation.py`` adds the Fukuta-Walter Knudsen correction.  ARG2000
    (and CliMA CloudMicrophysics) use the constant-coefficient form below,
    faithful to the published closed-form S_max derivation.
    """
    vapor_term = constants.rho_water * constants.R_v * T / (e_s * constants.D_vapor)
    thermal_term = (constants.rho_water * constants.L_v / (constants.k_air * T)
                    * (constants.L_v / (constants.R_v * T) - 1.0))
    return 1.0 / (vapor_term + thermal_term)


def arg_max_supersaturation(
    w: jnp.ndarray,
    T: jnp.ndarray,
    p: jnp.ndarray,
    mode_number: jnp.ndarray,
    mode_r_g: jnp.ndarray,
    mode_sigma_g: jnp.ndarray,
    mode_kappa: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """ARG2000 maximum supersaturation ``S_max`` and per-mode ``S_m,i``.

    All ``mode_*`` inputs are 1-D arrays of length ``n_modes`` in SI units
    (number [1/m^3], radius [m], sigma [-], kappa [-]).  ``w``, ``T``, ``p``
    share an arbitrary spatial shape ``F``.

    Returns
    -------
    S_max : jnp.ndarray, shape ``F``
    S_m : jnp.ndarray, shape ``(n_modes,) + F``
    """
    # Positive-updraft floor: alpha*w/G must stay > 0 so zeta, eta are finite
    # and the (eta + 3 zeta) denominator never vanishes.
    w_safe = jnp.maximum(w, _W_MIN_M_S)
    e_s = saturation_vapor_pressure(T)

    A = kohler_curvature_A(T)                                     # F
    # alpha [1/m]: gravitational settling of supersaturation.
    alpha = (constants.g * constants.L_v
             / (constants.c_pd * constants.R_v * T ** 2.0)
             - constants.g / (constants.R_d * T))
    # gamma [m^3/kg]: change in supersaturation per unit condensed water.
    gamma = (constants.R_v * T / e_s
             + constants.epsilon * constants.L_v ** 2.0
             / (constants.c_pd * p * T))
    G = condensation_growth_coeff_G(T, e_s)                       # F

    # alpha*w/G [1/m^2] appears in both zeta and eta.
    aw_over_G = alpha * w_safe / G                                # F
    zeta = (2.0 * A / 3.0) * jnp.sqrt(aw_over_G)                  # F

    # Broadcast per-mode parameters against the spatial field (mode axis 0).
    n_i = _broadcast_mode(jnp.maximum(mode_number, _N_A_FLOOR_M3), T)
    rg_i = _broadcast_mode(mode_r_g, T)
    sig_i = _broadcast_mode(jnp.maximum(mode_sigma_g, _SIGMA_G_FLOOR), T)
    kap_i = _broadcast_mode(mode_kappa, T)

    S_m = mode_critical_supersaturation(A[None, ...], rg_i, kap_i)  # (M,)+F
    eta_i = aw_over_G[None, ...] ** 1.5 / (
        2.0 * jnp.pi * constants.rho_water * gamma[None, ...] * n_i)

    ln_sig = jnp.log(sig_i)
    f_i = _ARG_F_PREFACTOR * jnp.exp(_ARG_F_LOGSQ_COEFF * ln_sig ** 2.0)
    g_i = 1.0 + _ARG_G_LOG_COEFF * ln_sig

    # ARG closed form: 1/S_max^2 = sum_i (1/S_m,i^2) [ f (zeta/eta)^{3/2}
    #                              + g (S_m^2/(eta + 3 zeta))^{3/4} ].
    inv_sm2 = 1.0 / S_m ** 2.0
    term = f_i * (zeta[None, ...] / eta_i) ** _ARG_SMAX_POW_1 + g_i * (
        S_m ** 2.0 / (eta_i + 3.0 * zeta[None, ...])) ** _ARG_SMAX_POW_2
    inv_smax2 = jnp.sum(inv_sm2 * term, axis=0)                   # F
    S_max = jnp.sqrt(1.0 / jnp.maximum(inv_smax2, _INV_S2_FLOOR))  # F
    return S_max, S_m


def arg_activated_fraction(
    S_m: jnp.ndarray, S_max: jnp.ndarray, mode_sigma_g: jnp.ndarray,
) -> jnp.ndarray:
    """Activated number fraction per mode [0, 1] (ARG2000 erfc form).

    ``frac_i = 0.5 erfc( 2 ln(S_m,i/S_max) / (3 sqrt(2) ln sigma_i) )``.
    ``S_m`` has shape ``(n_modes,) + F``; ``S_max`` shape ``F``;
    ``mode_sigma_g`` is the broadcastable per-mode geometric std.
    """
    sig = jnp.maximum(mode_sigma_g, _SIGMA_G_FLOOR)
    ratio = jnp.maximum(S_m / S_max[None, ...], _S_RATIO_FLOOR)
    u = 2.0 * jnp.log(ratio) / (3.0 * jnp.sqrt(2.0) * jnp.log(sig))
    return 0.5 * jax.scipy.special.erfc(u)


def arg_cdnc(
    w: jnp.ndarray,
    T: jnp.ndarray,
    p: jnp.ndarray,
    mode_number: jnp.ndarray,
    mode_r_g: jnp.ndarray,
    mode_sigma_g: jnp.ndarray,
    mode_kappa: jnp.ndarray,
    *,
    return_diagnostics: bool = False,
):
    """Total activated cloud-droplet number concentration (CDNC) [1/m^3].

    Parameters
    ----------
    w, T, p : jnp.ndarray
        Updraft velocity [m/s], temperature [K], pressure [Pa]; common shape.
    mode_number : jnp.ndarray, shape (n_modes,)
        Per-mode aerosol number concentration [1/m^3].
    mode_r_g : jnp.ndarray, shape (n_modes,)
        Per-mode geometric-mean DRY radius [m].
    mode_sigma_g : jnp.ndarray, shape (n_modes,)
        Per-mode geometric standard deviation [-] (> 1).
    mode_kappa : jnp.ndarray, shape (n_modes,)
        Per-mode kappa-Koehler hygroscopicity [-].
    return_diagnostics : bool, default False
        When True also return ``(S_max, activated_fraction_per_mode)``.

    Returns
    -------
    cdnc : jnp.ndarray, shape of ``T``
        Total activated droplet number [1/m^3] = sum_i N_i frac_i.
    """
    mode_number = jnp.asarray(mode_number)
    mode_r_g = jnp.asarray(mode_r_g)
    mode_sigma_g = jnp.asarray(mode_sigma_g)
    mode_kappa = jnp.asarray(mode_kappa)

    S_max, S_m = arg_max_supersaturation(
        w, T, p, mode_number, mode_r_g, mode_sigma_g, mode_kappa)
    frac = arg_activated_fraction(
        S_m, S_max, _broadcast_mode(mode_sigma_g, T))
    # Weight by the ACTUAL per-mode number (unfloored) so a genuinely empty
    # mode contributes exactly zero droplets.
    n_i = _broadcast_mode(mode_number, T)
    cdnc = jnp.sum(jnp.clip(n_i, 0.0, None) * frac, axis=0)
    if return_diagnostics:
        return cdnc, S_max, frac
    return cdnc


def arg_cdnc_from_config(
    config: ActivationConfig,
    T: jnp.ndarray,
    p: jnp.ndarray,
    *,
    w: jnp.ndarray | float | None = None,
    aerosol_number: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """CDNC [1/m^3] from an :class:`ActivationConfig` and (T, p) fields.

    The aerosol population is taken from ``config`` (prescribed modes, cm^-3 /
    um converted to SI), UNLESS ``aerosol_number`` is supplied — the prognostic
    number [1/m^3] field then REPLACES the (single) mode-0 number, keeping that
    mode's prescribed ``r_g``, ``sigma_g`` and ``kappa`` shape.  This is the
    Part-2 -> Part-1 feed: a prognostic bulk aerosol drives ARG activation.

    ``w`` defaults to ``config.w_char_m_s`` (characteristic sub-grid updraft).
    """
    # Static mode-consistency validation: all four lognormal-parameter tuples
    # must describe the SAME set of modes (equal length), and ARG needs at
    # least one — an incomplete/empty config (e.g. an r_g without a matching
    # sigma_g/kappa) would otherwise enter broken/empty broadcasting.
    n_modes = len(config.mode_number_cm3)
    if not (len(config.mode_r_g_um) == len(config.mode_sigma_g)
            == len(config.mode_kappa) == n_modes):
        raise ValueError(
            "ActivationConfig mode_number_cm3 / mode_r_g_um / mode_sigma_g / "
            "mode_kappa must have equal length (one entry per lognormal mode); "
            f"got {len(config.mode_number_cm3)}, {len(config.mode_r_g_um)}, "
            f"{len(config.mode_sigma_g)}, {len(config.mode_kappa)}."
        )
    if n_modes == 0:
        raise ValueError(
            "ARG activation requires at least one configured lognormal aerosol "
            "mode in ActivationConfig (all mode_* tuples are empty)."
        )
    if w is None:
        w = config.w_char_m_s
    w_arr = jnp.broadcast_to(jnp.asarray(w, dtype=T.dtype), T.shape)

    r_g = jnp.asarray(config.mode_r_g_um, dtype=T.dtype) * _UM_TO_M
    sigma_g = jnp.asarray(config.mode_sigma_g, dtype=T.dtype)
    kappa = jnp.asarray(config.mode_kappa, dtype=T.dtype)

    if aerosol_number is None:
        number = jnp.asarray(config.mode_number_cm3, dtype=T.dtype) * _PER_CM3_TO_PER_M3
        return arg_cdnc(w_arr, T, p, number, r_g, sigma_g, kappa)

    # Prognostic single-mode feed: number is a per-cell field [1/m^3]; the
    # prognostic mode inherits the config mode-0 shape (guaranteed present by
    # the mode validation above).  vmap the per-cell single-mode activation.
    #
    # Dispatch hardening: the prognostic feed supports SINGLE-mode configs
    # only in this pass — silently dropping configured coarse/Aitken modes
    # from the S_max competition would be a silent physics change (a coarse
    # mode measurably suppresses accumulation-mode activation), so a
    # multi-mode config + prognostic feed must fail loudly.
    if n_modes > 1:
        raise NotImplementedError(
            "Prognostic aerosol_number feed supports a SINGLE configured "
            f"lognormal mode, got {n_modes} modes in ActivationConfig. "
            "Multi-mode competition with a prognostic mode-0 number is not "
            "yet implemented; drop the extra modes explicitly or disable the "
            "prognostic feed."
        )
    number_field = jnp.asarray(aerosol_number, dtype=T.dtype)
    # Layout guard: the flatten+vmap below validates only the total element
    # count, so a transposed (nlev, ncol) field would silently misassociate
    # aerosol with the wrong grid cells.  Shapes are static — enforce exact
    # agreement with the thermodynamic fields at trace time.
    if number_field.shape != T.shape:
        raise ValueError(
            f"aerosol_number shape {number_field.shape} must exactly match "
            f"the T/p field shape {T.shape} (same layout, no transposition — "
            "element count alone is not checked downstream)."
        )
    r_g0 = r_g[:1]
    sigma0 = sigma_g[:1]
    kappa0 = kappa[:1]

    def _single(w_c, T_c, p_c, n_c):
        return arg_cdnc(
            w_c[None], T_c[None], p_c[None],
            n_c[None], r_g0, sigma0, kappa0)[0]

    flat = (w_arr.reshape(-1), T.reshape(-1), p.reshape(-1),
            number_field.reshape(-1))
    cdnc_flat = jax.vmap(_single)(*flat)
    return cdnc_flat.reshape(T.shape)


def activated_nc_field(
    config: ActivationConfig,
    target_shape,
    *,
    aerosol_od: jnp.ndarray | None = None,
    T: jnp.ndarray | None = None,
    p: jnp.ndarray | None = None,
    w: jnp.ndarray | float | None = None,
    aerosol_number: jnp.ndarray | None = None,
    ccn_aod: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Specified cloud-droplet number field [1/m^3] — activation dispatch.

    Single selectable entry point routing between the climatological AOD proxy
    and physically-based ARG activation.  ``scheme="proxy"`` (default) is
    BYTE-IDENTICAL to :func:`aerosol_activation.specified_nc_field` with the
    default proxy config, preserving existing runs bit-for-bit.

    Parameters
    ----------
    config : ActivationConfig
        Selects the scheme and carries its parameters.
    target_shape : tuple of int
        Output ``(ncol, nlev)`` field shape the microphysics / radiation kernel
        expects.
    aerosol_od : jnp.ndarray, optional
        Per-layer aerosol optical depth (REQUIRED for ``scheme="proxy"``).
    T, p : jnp.ndarray, optional
        Temperature [K] and pressure [Pa] fields (REQUIRED for ``scheme="arg"``),
        each broadcastable to ``target_shape``.
    w : jnp.ndarray or float, optional
        Updraft [m/s] for ARG; defaults to ``config.w_char_m_s``.
    aerosol_number : jnp.ndarray, optional
        Prognostic aerosol number [1/m^3]; when supplied it drives ARG instead
        of the config's prescribed mode-0 number.
    ccn_aod : jnp.ndarray, optional
        Proxy column AOD ``(ncol,)`` overriding the ``aerosol_od`` sum.

    Returns
    -------
    n_c : jnp.ndarray, shape ``target_shape``
    """
    scheme = config.scheme
    if scheme == "proxy":
        if aerosol_od is None:
            raise ValueError(
                "activation scheme 'proxy' requires an aerosol_od field "
                "(enable external aerosol forcing --aerosol-forcing external, "
                "or disable --aerosol-ccn)."
            )
        from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
            specified_nc_field,
        )
        proxy_cfg = config.proxy if config.proxy is not None else _default_proxy_config()
        return specified_nc_field(jnp.asarray(aerosol_od), target_shape, proxy_cfg,
                                  ccn_aod=ccn_aod)
    if scheme == "arg":
        if T is None or p is None:
            raise ValueError(
                "activation scheme 'arg' requires T and p fields "
                "(updraft-driven supersaturation balance)."
            )
        T_f = jnp.broadcast_to(jnp.asarray(T), target_shape)
        p_f = jnp.broadcast_to(jnp.asarray(p), target_shape)
        return arg_cdnc_from_config(
            config, T_f, p_f, w=w, aerosol_number=aerosol_number)
    raise ValueError(
        f"unknown activation scheme {scheme!r}; expected 'proxy' or 'arg'."
    )
