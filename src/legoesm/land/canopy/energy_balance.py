"""Leaf and soil energy balance for the two-leaf canopy model.

Two schemes are provided:
  - BT (Bulk Transfer): direct specific-humidity-gradient formulation.
      LE = λ ρ (q_f - q_c) / (Rb + rs)
    Computationally efficient; default.
  - PM (Penman-Monteith): second-order Paw & Gao (1988) solution.
    More physical but requires second derivatives of sat. vapour pressure.

Canopy air-space update: conductance-weighted mixing of heat/vapour from
leaves, soil, and free atmosphere.

All functions are pure JAX, JIT-compatible, and differentiable.
Sources: DifferBESS/process/CanopyEnergyBalance.py, Soil.py, CarbonWaterFluxes.py
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from legoesm.land.canopy.stomatal import ball_berry_gs, medlyn_gs

# Module-local constants.
# NOTE: Stefan-Boltzmann and other canonical physical constants are imported
# from ``legoesm.constants`` — do not redefine them here.
_T0    = 273.15      # [K]  (kept local: used inside Tetens-form literals)
_Ps0   = 101325.0    # IUPAC STP pressure [Pa] used in the mol → m/s
                     # unit conversion factor 0.446; distinct from
                     # ``constants.p_ref`` (1e5 Pa hydrostatic reference).
_Lv    = 2.501e6     # latent heat of vaporisation at 0°C [J kg-1]


# ---------------------------------------------------------------------------
# Meteorological helpers
# ---------------------------------------------------------------------------

@jax.jit
def saturation_specific_humidity(T: jax.Array, p: jax.Array) -> jax.Array:
    """Saturation specific humidity [kg kg-1] from T [K] and p [Pa]."""
    e_s = 611.2 * jnp.exp(17.67 * (T - _T0) / ((T - _T0) + 243.5))
    return 0.622 * e_s / (p - (1.0 - 0.622) * e_s)


@jax.jit
def canopy_met_variables(
    Ps: jax.Array,
    Tc: jax.Array,
    q_c: jax.Array,
) -> tuple[jax.Array, ...]:
    """Meteorological variables for the canopy air space.

    Parameters
    ----------
    Ps  : atmospheric pressure [Pa]
    Tc  : canopy air temperature [K]
    q_c : canopy air specific humidity [kg kg-1]

    Returns
    -------
    e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma
      All in Pa (or Pa K-1 for derivatives; Pa K-2 for second derivative).
    """
    # Vapour pressure from specific humidity
    e_c  = q_c * Ps / (0.622 + (1.0 - 0.622) * q_c)
    # Saturation vapour pressure (Clausius-Clapeyron)
    TcC  = Tc - _T0
    es_c = 611.2 * jnp.exp(17.67 * TcC / (TcC + 243.5))

    VPD_c = es_c - e_c
    RH_c  = jnp.clip(e_c / jnp.maximum(es_c, 1e-6), 0.0, 1.0)

    # First derivative des/dT [Pa K-1]
    desTc  = es_c * 4098.0 * (TcC + 237.3) ** (-2)
    # Second derivative d²es/dT² [Pa K-2]
    ddesTc = 4098.0 * (
        desTc  * (TcC + 237.3) ** (-2)
        + (-2.0) * e_c * (TcC + 237.3) ** (-3)
    )

    # Latent heat and psychrometric constant
    lam   = _Lv - 2.361e3 * TcC
    gamma = 1004.0 / 0.622 * Ps / lam   # [Pa K-1]

    return e_c, es_c, VPD_c, RH_c, desTc, ddesTc, gamma


# ---------------------------------------------------------------------------
# Stomatal conductance dispatch (Ball-Berry or Medlyn)
# ---------------------------------------------------------------------------

def _compute_gs_and_ci(
    An: jax.Array,
    RH_c: jax.Array,
    VPD_c: jax.Array,
    Ca: jax.Array,
    Tf: jax.Array,
    Ps: jax.Array,
    m: jax.Array,
    b0: jax.Array,
    stomatal_model: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Stomatal conductance and intercellular CO2 closure for a leaf.

    Dispatches between Ball-Berry (RH-based) and Medlyn (VPD-based) based
    on the ``stomatal_model`` static argument.  Both models use the same
    per-leaf ``(m, b0)`` pair — ``m`` is interpreted as the Ball-Berry
    slope ``g1_bb`` for ``"ball_berry"`` and as the Medlyn slope ``g1_med``
    [kPa^0.5] for ``"medlyn"``; ``b0`` is the residual conductance in both
    cases.

    Parameters
    ----------
    An     : net assimilation [μmol CO2 / m^2 / s]
    RH_c   : canopy-air relative humidity [-]  (used by Ball-Berry)
    VPD_c  : canopy-air vapour pressure deficit [Pa]  (converted to
             kPa inside for Medlyn; unused by Ball-Berry)
    Ca     : ambient CO2 [μmol / mol]
    Tf     : leaf temperature [K]  (for mol → m/s unit conversion)
    Ps     : surface pressure [Pa]
    m, b0  : stomatal slope and intercept (see above)
    stomatal_model : ``"ball_berry"`` | ``"medlyn"`` — static Python string
                     captured in a ``functools.partial`` closure; never
                     traced by JAX.

    Returns
    -------
    (rs [s m-1], gs [m s-1], Ci [μmol mol-1])
    """
    if stomatal_model == "medlyn":
        VPD_kPa = jnp.maximum(VPD_c, 50.0) / 1000.0  # Pa → kPa, floor 0.05 kPa
        gs_mol = medlyn_gs(An, VPD_kPa, Ca, m, b0)
    else:
        gs_mol = ball_berry_gs(An, RH_c, Ca, m, b0)

    Ci = Ca - 1.6 * An / jnp.maximum(gs_mol, 1e-9)
    # Clip Ci to the physically reasonable C3 range; mixed-PFT C3/C4
    # is handled upstream in ``photosynthesis()`` via the continuous fC4
    # fraction, so the C4 bounds are not needed here.
    Ci = jnp.clip(Ci, 0.5 * Ca, 0.9 * Ca)

    # Unit conversion: mol m-2 s-1 → m s-1
    cf = 0.446 * (_T0 / Tf) * (Ps / _Ps0)
    rs = 1.0 / (gs_mol / cf * 1e-2)   # [s m-1]
    gs = 1.0 / rs                      # [m s-1]
    return rs, gs, Ci


# ---------------------------------------------------------------------------
# Leaf energy balance — BT (Bulk Transfer)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("stomatal_model",))
def leaf_energy_balance_bt(
    An: jax.Array,
    ASW: jax.Array,
    ALW: jax.Array,
    Tf: jax.Array,
    Ps: jax.Array,
    Ca: jax.Array,
    Tc: jax.Array,
    q_f: jax.Array,
    q_c: jax.Array,
    RH_c: jax.Array,
    VPD_c: jax.Array,
    lam: jax.Array,
    Cp: jax.Array,
    rhoa: jax.Array,
    Rb: jax.Array,
    m: jax.Array,
    b0: jax.Array,
    stomatal_model: str = "ball_berry",
) -> tuple[jax.Array, ...]:
    """Leaf energy balance via direct bulk transfer (BT).

    LE = λ ρ (q_f - q_c) / (Rb + rs)
    H  = Rn - LE
    Tf_new = Tc + Rb / (ρ Cp) * H

    Parameters
    ----------
    An     : net photosynthesis [μmol m-2 s-1]
    ASW    : absorbed shortwave [W m-2]
    ALW    : net absorbed longwave [W m-2]
    Tf     : leaf temperature [K]
    Ps     : pressure [Pa]
    Ca     : ambient CO2 [μmol mol-1]
    Tc     : canopy air temperature [K]
    q_f    : leaf saturation specific humidity [kg kg-1]
    q_c    : canopy air specific humidity [kg kg-1]
    RH_c   : canopy relative humidity [-]
    VPD_c  : canopy-air vapour pressure deficit [Pa] (used by Medlyn)
    lam    : latent heat of vaporisation [J kg-1]
    Cp     : specific heat of air [J kg-1 K-1]
    rhoa   : air density [kg m-3]
    Rb     : boundary-layer resistance [s m-1]
    m, b0  : stomatal slope and intercept (Ball-Berry or Medlyn; see
             ``_compute_gs_and_ci``)
    stomatal_model : ``"ball_berry"`` | ``"medlyn"`` — static argument.

    Returns
    -------
    Rn, LE, H, Tf_new, gs, Ci
    """
    rs, gs, Ci = _compute_gs_and_ci(
        An, RH_c, VPD_c, Ca, Tf, Ps, m, b0, stomatal_model)

    Rn = ASW + ALW
    LE = lam * rhoa * (q_f - q_c) / jnp.maximum(Rb + rs, 1e-6)

    # LE sign is not constrained here: negative LE = dew formation on the
    # leaf, positive LE = transpiration + evaporation.  The DifferBESS
    # daytime-only clamp ``LE ∈ [0, max(Rn, 0)]`` has been removed so the
    # model can represent nocturnal dew and non-stationary transitions.

    H  = Rn - LE
    # ``Tf_new`` is the leaf temperature satisfying sensible-flux closure
    # ``H = ρ Cp (Tf − Tc)/Rb``.  The earlier hard ``clip(dT, ±30)`` was
    # non-differentiable at the clamp boundary and silently degraded the
    # Newton Jacobian.  Leave ``Tf_new`` unclipped here — the Newton
    # driver ``solve_canopy_closure`` already damps ``Δx`` globally
    # (``clamp = 10 → 0.1`` across iterations), so step sizes remain
    # bounded without a local non-smooth clip.
    Tf_new = Tc + Rb * H / (rhoa * Cp)

    return Rn, LE, H, Tf_new, gs, Ci


# ---------------------------------------------------------------------------
# Leaf energy balance — PM (Penman-Monteith, second-order Paw & Gao 1988)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("stomatal_model",))
def leaf_energy_balance_pm(
    An: jax.Array,
    ASW: jax.Array,
    ALW: jax.Array,
    Tf: jax.Array,
    Ps: jax.Array,
    Ca: jax.Array,
    Tc: jax.Array,
    VPD_c: jax.Array,
    RH_c: jax.Array,
    desTc: jax.Array,
    ddesTc: jax.Array,
    gamma_c: jax.Array,
    Cp: jax.Array,
    rhoa: jax.Array,
    Rb: jax.Array,
    m: jax.Array,
    b0: jax.Array,
    stomatal_model: str = "ball_berry",
) -> tuple[jax.Array, ...]:
    """Leaf energy balance via second-order Penman-Monteith (Paw & Gao 1988).

    ``stomatal_model`` selects Ball-Berry or Medlyn; see
    ``leaf_energy_balance_bt`` for details.

    Returns
    -------
    Rn, LE, H, Tf_new, gs, Ci
    """
    rs, gs, Ci = _compute_gs_and_ci(
        An, RH_c, VPD_c, Ca, Tf, Ps, m, b0, stomatal_model)

    Rn = ASW + ALW
    rc = rs

    ddesTc_Rb2          = ddesTc * Rb**2
    gamma_Rb_rc          = gamma_c * (Rb + rc)
    rhoa_Cp_gamma_Rb_rc  = rhoa * Cp * gamma_Rb_rc

    a = 0.5 * ddesTc_Rb2 / rhoa_Cp_gamma_Rb_rc
    b = (-1.0
         - Rb * desTc / gamma_Rb_rc
         - ddesTc_Rb2 * Rn / rhoa_Cp_gamma_Rb_rc)
    c = (rhoa * Cp / gamma_Rb_rc * VPD_c
         + desTc * Rb / gamma_Rb_rc * Rn
         + 0.5 * ddesTc_Rb2 / rhoa_Cp_gamma_Rb_rc * Rn**2)

    disc = jnp.maximum(b**2 - 4.0 * a * c, 0.0)
    LE   = (-b + jnp.sign(b) * jnp.sqrt(disc)) / (2.0 * a)

    # No LE clamp — see notes in leaf_energy_balance_bt for rationale.

    H  = Rn - LE
    # Unclipped Tf update; see leaf_energy_balance_bt for rationale.
    Tf_new = Tc + Rb * H / (rhoa * Cp)

    return Rn, LE, H, Tf_new, gs, Ci


# ---------------------------------------------------------------------------
# Soil energy balance — BT
# ---------------------------------------------------------------------------

@jax.jit
def soil_energy_balance_bt(
    Ts: jax.Array,
    Tc: jax.Array,
    q_s: jax.Array,
    q_c: jax.Array,
    lam: jax.Array,
    rhoa: jax.Array,
    Cp: jax.Array,
    rah_soil: jax.Array,
    raw_soil: jax.Array,
    Rsoil: jax.Array,
    ASW_soil: jax.Array,
    ALW_soil: jax.Array,
) -> tuple[jax.Array, ...]:
    """Soil energy balance with prescribed skin temperature (BT).

    The soil skin ``Ts`` is supplied by the caller from the top layer of the
    multilayer soil thermal state (``T_soil[:, 0]``).  Turbulent fluxes are
    computed from explicit bulk-transfer formulas and ``G`` closes the
    surface energy budget as a residual — it is *not* parameterised as
    ``G_alpha · Rn``.  The resulting ``G`` is then fed as the upper
    boundary condition to ``solve_soil_thermal`` in the caller (the same
    pattern as ``multilayer_land.py``).

    ``Rsoil`` is the soil-dryness surface resistance
    ``raw_below · (1/fStress_soil - 1)`` added in series with ``raw_soil``.

    Returns
    -------
    Rn_soil, LE_soil, H_soil, G
    """
    Rn = ASW_soil + ALW_soil
    # Direct bulk-transfer turbulent fluxes — Ts is prescribed so no
    # root-finding for soil T is needed.
    LE = lam * rhoa * (q_s - q_c) / jnp.maximum(raw_soil + Rsoil, 1e-6)
    H  = rhoa * Cp * (Ts - Tc) / jnp.maximum(rah_soil, 1e-6)
    # G closes the surface energy budget as a residual — positive into soil.
    G  = Rn - LE - H
    return Rn, LE, H, G


# ---------------------------------------------------------------------------
# Soil energy balance — PM
# ---------------------------------------------------------------------------

@jax.jit
def soil_energy_balance_pm(
    Ts: jax.Array,
    Tc: jax.Array,
    q_s: jax.Array,
    q_c: jax.Array,
    lam: jax.Array,
    rhoa: jax.Array,
    Cp: jax.Array,
    rah_soil: jax.Array,
    raw_soil: jax.Array,
    Rsoil: jax.Array,
    ASW_soil: jax.Array,
    ALW_soil: jax.Array,
) -> tuple[jax.Array, ...]:
    """Soil energy balance with prescribed skin temperature (PM).

    With ``Ts`` prescribed by the caller, the second-order Penman-Monteith
    quadratic that originally solved for ``Ts`` is no longer needed — LE
    and H follow from explicit bulk-transfer formulas.  The PM variant is
    therefore numerically identical to the BT variant at the soil level;
    the dispatch is kept for API symmetry with the leaf pathway where PM
    and BT still differ.

    Returns
    -------
    Rn_soil, LE_soil, H_soil, G
    """
    Rn = ASW_soil + ALW_soil
    LE = lam * rhoa * (q_s - q_c) / jnp.maximum(raw_soil + Rsoil, 1e-6)
    H  = rhoa * Cp * (Ts - Tc) / jnp.maximum(rah_soil, 1e-6)
    G  = Rn - LE - H
    return Rn, LE, H, G


# ---------------------------------------------------------------------------
# Canopy air temperature and humidity update
# ---------------------------------------------------------------------------

@jax.jit
def canopy_air_update(
    Ta: jax.Array,
    q_atm: jax.Array,
    Tf_Sun: jax.Array,
    Tf_Sh: jax.Array,
    Ts: jax.Array,
    gs_Sun: jax.Array,
    gs_Sh: jax.Array,
    Rb_Sun: jax.Array,
    Rb_Sh: jax.Array,
    rah_above: jax.Array,
    raw_above: jax.Array,
    rah_below: jax.Array,
    raw_below: jax.Array,
    Rsoil: jax.Array,
    Ps: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Update canopy air temperature Tc and specific humidity q_c.

    Conductance-weighted mixing of above-canopy air, sunlit and shaded
    leaves, and soil — DifferBESS FULLY_COUPLED formulation.  Leaves and
    soil both communicate with the canopy air space (Tc, q_c).

    Returns
    -------
    Tc_new, q_c_new
    """
    ch_a   = 1.0 / jnp.maximum(rah_above, 1e-9)
    ch_sun = 1.0 / jnp.maximum(Rb_Sun,    1e-9)
    ch_sh  = 1.0 / jnp.maximum(Rb_Sh,     1e-9)
    ch_g   = 1.0 / jnp.maximum(rah_below, 1e-9)

    gs_Sun_safe = jnp.maximum(gs_Sun, 1e-9)
    gs_Sh_safe  = jnp.maximum(gs_Sh,  1e-9)
    cw_a   = 1.0 / jnp.maximum(raw_above, 1e-9)
    cw_sun = 1.0 / (Rb_Sun + 1.0 / gs_Sun_safe)
    cw_sh  = 1.0 / (Rb_Sh  + 1.0 / gs_Sh_safe)
    cw_g   = 1.0 / jnp.maximum(raw_below + Rsoil, 1e-9)

    q_f_Sun = saturation_specific_humidity(Tf_Sun, Ps)
    q_f_Sh  = saturation_specific_humidity(Tf_Sh,  Ps)
    q_s     = saturation_specific_humidity(Ts,     Ps)

    Tc_new = (ch_a * Ta + ch_sun * Tf_Sun + ch_sh * Tf_Sh + ch_g * Ts) / (
        ch_a + ch_sun + ch_sh + ch_g)
    q_c_new = (cw_a * q_atm + cw_sun * q_f_Sun + cw_sh * q_f_Sh + cw_g * q_s) / (
        cw_a + cw_sun + cw_sh + cw_g)

    return Tc_new, q_c_new
