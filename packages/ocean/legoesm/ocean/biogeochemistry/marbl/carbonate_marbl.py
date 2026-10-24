"""Seawater carbonate chemistry (OCMIP-2 / MARBL co2calc).

Equilibrium constants, pressure corrections and the total-alkalinity
expression follow MARBL's ``marbl_co2calc_mod.F90`` (OCMIP-2 co2calc,
Dickson & Goyet 1994 DOE handbook), so the MARBL port reproduces MARBL:

- K0 (Weiss 1974), fugacity factor ff (Weiss & Price 1980)
- K1, K2 (Lueker et al. 2000, total pH scale)
- KB (Dickson 1990), K1P/K2P/K3P (Millero 1995), KSi (Yao & Millero 1995)
- KW (Millero 1995), KS (Dickson 1990, free scale), KF (Dickson & Riley 1979)
- B_T (Uppstrom 1974), S_T (Morris & Riley 1966), F_T (Riley 1965)
- pressure corrections (Millero 1995, with CO2SYS typo fixes)
- CO3 at calcite / aragonite saturation (Mucci 1983, Millero 1979)

Concentrations inside this module are in mol/kg.  [H+] is found by a
safeguarded Newton/bisection iteration in ln[H+] with a fixed iteration
count, wrapped in ``jax.lax.custom_root`` so gradients follow from the
implicit-function theorem rather than from unrolling the iteration.

All functions are JAX-compatible (jit, vmap, grad).
"""

from __future__ import annotations
# NOTE: this is the ggn/bgc_modelling carbonate.py kept verbatim for the MARBL port
# (pressure-dependent constants, full-depth speciation, CO3 saturation).  The
# abiotic/NPZD schemes use ../carbonate.py (main's surface solver); reconciling the
# two solvers is a separate task.

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants

# Lower bounds MARBL applies before the solve (marbl_co2calc_mod: salt_min,
# dic_min, alk_min; the latter two are in mmol/m^3 there).
_SALT_MIN = 0.1
_DIC_MIN = _SALT_MIN / 35.0 * 1944.0e-3 / constants.rho_sw_marbl  # [mol/kg]
_ALK_MIN = _SALT_MIN / 35.0 * 2225.0e-3 / constants.rho_sw_marbl  # [mol/kg]

# Gas constant in cm^3 bar / (mol K) for the pressure corrections.
_R_CM3_BAR = constants.R_universal * 10.0
_LN10 = 2.302585092994046


class CarbonateCoeffs(NamedTuple):
    """Equilibrium constants [mol/kg units] and conservative totals [mol/kg]."""
    k0: jnp.ndarray   # CO2 solubility [mol/(kg atm)]
    ff: jnp.ndarray   # CO2 solubility incl. non-ideality and pH2O [mol/(kg atm)]
    k1: jnp.ndarray
    k2: jnp.ndarray
    kb: jnp.ndarray
    k1p: jnp.ndarray
    k2p: jnp.ndarray
    k3p: jnp.ndarray
    ksi: jnp.ndarray
    kw: jnp.ndarray
    ks: jnp.ndarray   # free pH scale
    kf: jnp.ndarray
    bt: jnp.ndarray   # total borate
    st: jnp.ndarray   # total sulfate
    ft: jnp.ndarray   # total fluoride


class CarbonateSpecies(NamedTuple):
    """Carbonate speciation [mol/kg] and pH (total scale)."""
    H2CO3: jnp.ndarray  # CO2* (dissolved CO2 + H2CO3)
    HCO3: jnp.ndarray
    CO3: jnp.ndarray
    pH: jnp.ndarray


# ============================================================================
# Individual constants (surface, p = 0)
# ============================================================================


def co2_solubility(T_degC: jnp.ndarray, S_psu: jnp.ndarray) -> jnp.ndarray:
    """Henry's law solubility of CO2 in seawater, K0 [mol/(kg atm)] (Weiss 1974)."""
    T_100 = (T_degC + constants.T_freeze) / 100.0
    ln_K0 = (
        -60.2409 + 93.4517 / T_100 + 23.3585 * jnp.log(T_100)
        + S_psu * (0.023517 - 0.023656 * T_100 + 0.0047036 * T_100 ** 2)
    )
    return jnp.exp(ln_K0)


def co2_solubility_moist_air(T_degC: jnp.ndarray, S_psu: jnp.ndarray) -> jnp.ndarray:
    """CO2 solubility for dry-air mole fraction at 1 atm moist air, ff [mol/(kg atm)].

    Weiss & Price (1980), eq. 13 with table 6: includes the fugacity
    correction and the water-vapour pressure.  [CO2*]_air = xCO2 * ff * P_atm.
    """
    T_100 = (T_degC + constants.T_freeze) / 100.0
    ln_ff = (
        -162.8301 + 218.2968 / T_100 + 90.9241 * jnp.log(T_100) - 1.47696 * T_100 ** 2
        + S_psu * (0.025695 - 0.025225 * T_100 + 0.0049867 * T_100 ** 2)
    )
    return jnp.exp(ln_ff)


def carbonate_equilibria(
    T_degC: jnp.ndarray,
    S_psu: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """K1, K2 of carbonic acid (Lueker et al. 2000, total pH scale) [mol/kg]."""
    T_K = T_degC + constants.T_freeze
    pK1 = (
        3633.86 / T_K - 61.2172 + 9.6777 * jnp.log(T_K)
        - 0.011555 * S_psu + 0.0001152 * S_psu ** 2
    )
    pK2 = (
        471.78 / T_K + 25.929 - 3.16967 * jnp.log(T_K)
        - 0.01781 * S_psu + 0.0001122 * S_psu ** 2
    )
    return 10.0 ** (-pK1), 10.0 ** (-pK2)


def borate_equilibrium(T_degC: jnp.ndarray, S_psu: jnp.ndarray) -> jnp.ndarray:
    """Borate dissociation constant K_B (Dickson 1990, total pH scale) [mol/kg]."""
    T_K = T_degC + constants.T_freeze
    sqrt_S = jnp.sqrt(jnp.maximum(S_psu, 0.0))
    ln_KB = (
        (-8966.90 - 2890.53 * sqrt_S - 77.942 * S_psu
         + 1.728 * S_psu * sqrt_S - 0.0996 * S_psu ** 2) / T_K
        + 148.0248 + 137.1942 * sqrt_S + 1.62142 * S_psu
        + (-24.4344 - 25.085 * sqrt_S - 0.2474 * S_psu) * jnp.log(T_K)
        + 0.053105 * sqrt_S * T_K
    )
    return jnp.exp(ln_KB)


def total_borate(S_psu: jnp.ndarray) -> jnp.ndarray:
    """Total boron [mol/kg] from chlorinity (Uppstrom 1974)."""
    return 0.000232 / 10.811 * (S_psu / 1.80655)


def _pressure_factor(T_degC, inv_RT, p_bar, dV_coefs, kappa_coefs):
    """K(p)/K(0) = exp((-dV + kappa p / 2) p / RT) (Millero 1995)."""
    dV = dV_coefs[0] + (dV_coefs[1] + dV_coefs[2] * T_degC) * T_degC
    kap = (kappa_coefs[0] + (kappa_coefs[1] + kappa_coefs[2] * T_degC) * T_degC) * 1.0e-3
    return jnp.exp((-dV + 0.5 * kap * p_bar) * p_bar * inv_RT)


def carbonate_coefficients(
    T_degC: jnp.ndarray,
    S_psu: jnp.ndarray,
    p_bar: jnp.ndarray | None = None,
) -> CarbonateCoeffs:
    """All equilibrium constants and totals (MARBL comp_co2calc_coeffs).

    Parameters
    ----------
    T_degC : array
        In-situ temperature [degC].
    S_psu : array
        Salinity [PSU]; bounded below by 0.1 as in MARBL.
    p_bar : array or None
        Sea pressure [bar] (gauge, 0 at the surface).  None skips the
        pressure correction (surface solve).
    """
    S = jnp.maximum(S_psu, _SALT_MIN)
    T_K = T_degC + constants.T_freeze
    inv_T = 1.0 / T_K
    ln_T = jnp.log(T_K)
    sqrt_S = jnp.sqrt(S)
    ionic = 19.924 * S / (1000.0 - 1.005 * S)
    sqrt_I = jnp.sqrt(ionic)
    chlorinity = S / 1.80655
    ln_1m = jnp.log(1.0 - 0.001005 * S)

    k0 = co2_solubility(T_degC, S)
    ff = co2_solubility_moist_air(T_degC, S)
    k1, k2 = carbonate_equilibria(T_degC, S)
    kb = borate_equilibrium(T_degC, S)

    k1p = jnp.exp(-4576.752 * inv_T + 115.525 - 18.453 * ln_T
                  + (-106.736 * inv_T + 0.69171) * sqrt_S
                  + (-0.65643 * inv_T - 0.01844) * S)
    k2p = jnp.exp(-8814.715 * inv_T + 172.0883 - 27.927 * ln_T
                  + (-160.340 * inv_T + 1.3566) * sqrt_S
                  + (0.37335 * inv_T - 0.05778) * S)
    k3p = jnp.exp(-3070.75 * inv_T - 18.141
                  + (17.27039 * inv_T + 2.81197) * sqrt_S
                  + (-44.99486 * inv_T - 0.09984) * S)
    ksi = jnp.exp(-8904.2 * inv_T + 117.385 - 19.334 * ln_T
                  + (-458.79 * inv_T + 3.5913) * sqrt_I
                  + (188.74 * inv_T - 1.5998) * ionic
                  + (-12.1652 * inv_T + 0.07871) * ionic ** 2
                  + ln_1m)
    kw = jnp.exp(-13847.26 * inv_T + 148.9652 - 23.6521 * ln_T
                 + (118.67 * inv_T - 5.977 + 1.0495 * ln_T) * sqrt_S
                 - 0.01615 * S)
    ks = jnp.exp(-4276.1 * inv_T + 141.328 - 23.093 * ln_T
                 + (-13856.0 * inv_T + 324.57 - 47.986 * ln_T) * sqrt_I
                 + (35474.0 * inv_T - 771.54 + 114.723 * ln_T) * ionic
                 - 2698.0 * inv_T * ionic * sqrt_I
                 + 1776.0 * inv_T * ionic ** 2
                 + ln_1m)

    if p_bar is not None:
        inv_RT = 1.0 / (_R_CM3_BAR * T_K)
        pf = lambda dV, kap: _pressure_factor(T_degC, inv_RT, p_bar, dV, kap)  # noqa: E731
        k1 = k1 * pf((-25.5, 0.1271, 0.0), (-3.08, 0.0877, 0.0))
        k2 = k2 * pf((-15.82, -0.0219, 0.0), (1.13, -0.1475, 0.0))
        kb = kb * pf((-29.48, 0.1622, -0.002608), (-2.84, 0.0, 0.0))
        k1p = k1p * pf((-14.51, 0.1211, -0.000321), (-2.67, 0.0427, 0.0))
        k2p = k2p * pf((-23.12, 0.1758, -0.002647), (-5.15, 0.09, 0.0))
        k3p = k3p * pf((-26.57, 0.202, -0.003042), (-4.08, 0.0714, 0.0))
        ksi = ksi * pf((-29.48, 0.1622, -0.002608), (-2.84, 0.0, 0.0))
        kw = kw * pf((-20.02, 0.1119, -0.001409), (-5.13, 0.0794, 0.0))
        ks = ks * pf((-18.03, 0.0466, 0.000316), (-4.53, 0.09, 0.0))

    # kf uses the (pressure-corrected) ks for the free -> total scale change,
    # as in MARBL.
    kf = jnp.exp(1590.2 * inv_T - 12.641 + 1.525 * sqrt_I + ln_1m
                 + jnp.log(1.0 + (0.1400 / 96.062) * chlorinity / ks))
    if p_bar is not None:
        kf = kf * pf((-9.78, -0.009, -0.000942), (-3.91, 0.054, 0.0))

    bt = total_borate(S)
    st = 0.14 / 96.062 * chlorinity
    ft = 0.000067 / 18.9984 * chlorinity
    return CarbonateCoeffs(k0, ff, k1, k2, kb, k1p, k2p, k3p, ksi, kw, ks, kf, bt, st, ft)


# ============================================================================
# [H+] solve
# ============================================================================


def _alkalinity_residual(H, dic, ta, pt, sit, c: CarbonateCoeffs):
    """TA(H) - TA: hco3 + 2 co3 + borate + oh + hpo4 + 2 po4 + silicate
    - hfree - hso4 - hf - h3po4 - TA (MARBL total_alkalinity, mol/kg)."""
    H2 = H * H
    H3 = H2 * H
    k12 = c.k1 * c.k2
    k12p = c.k1p * c.k2p
    k123p = k12p * c.k3p
    a = H3 + c.k1p * H2 + k12p * H + k123p
    b = H2 + c.k1 * H + k12
    sulf = 1.0 + c.st / c.ks
    return (c.k1 * dic * H / b
            + 2.0 * dic * k12 / b
            + c.bt * c.kb / (c.kb + H)
            + c.kw / H
            + pt * k12p * H / a
            + 2.0 * pt * k123p / a
            + sit * c.ksi / (c.ksi + H)
            - H / sulf
            - c.st / (1.0 + sulf * c.ks / H)
            - c.ft / (1.0 + c.kf / H)
            - pt * H3 / a
            - ta)


def solve_htotal(
    dic: jnp.ndarray,
    ta: jnp.ndarray,
    pt: jnp.ndarray,
    sit: jnp.ndarray,
    coeffs: CarbonateCoeffs,
    n_iter: int = 50,
    ph_lo: float = 2.0,
    ph_hi: float = 13.0,
) -> jnp.ndarray:
    """Total [H+] [mol/kg] satisfying the alkalinity balance.

    Safeguarded Newton in x = ln[H+] on the bracket pH in [ph_lo, ph_hi]:
    the residual decreases monotonically in [H+], so each step keeps the
    root bracketed and falls back to bisection when Newton leaves the
    bracket.  ``n_iter`` iterations are always performed (static shape).
    Gradients are implicit (``lax.custom_root``).

    Inputs are in mol/kg; ``dic`` and ``ta`` are bounded below and ``pt``,
    ``sit`` at zero as in MARBL.
    """
    dic = jnp.maximum(dic, _DIC_MIN)
    ta = jnp.maximum(ta, _ALK_MIN)
    pt = jnp.maximum(pt, 0.0)
    sit = jnp.maximum(sit, 0.0)

    def residual(x):
        return _alkalinity_residual(jnp.exp(x), dic, ta, pt, sit, coeffs)

    _dtype = jnp.result_type(dic, ta, pt, sit, *jax.tree_util.tree_leaves(coeffs))
    x_lo0 = jnp.full(dic.shape, -ph_hi * _LN10, dtype=_dtype)
    x_hi0 = jnp.full(dic.shape, -ph_lo * _LN10, dtype=_dtype)

    def solve(f, x0):
        def body(_, carry):
            x, lo, hi = carry
            fx, dfx = jax.jvp(f, (x,), (jnp.ones_like(x),))
            # residual decreases in x: f > 0 means the root lies above x
            above = fx > 0.0
            lo = jnp.where(above, x, lo)
            hi = jnp.where(above, hi, x)
            dfx_safe = jnp.where(dfx < 0.0, dfx, -1.0)
            x_newton = x - fx / dfx_safe
            ok = (dfx < 0.0) & (x_newton > lo) & (x_newton < hi)
            x = jnp.where(ok, x_newton, 0.5 * (lo + hi))
            return x, lo, hi

        x, _, _ = jax.lax.fori_loop(0, n_iter, body, (x0, x_lo0, x_hi0))
        return x

    def tangent_solve(g, y):
        # residual is elementwise, so its Jacobian is diagonal
        return y / g(jnp.ones_like(y))

    x0 = 0.5 * (x_lo0 + x_hi0)
    x = jax.lax.custom_root(residual, x0, solve, tangent_solve)
    return jnp.exp(x)


def carbonate_speciation(
    dic: jnp.ndarray,
    ta: jnp.ndarray,
    pt: jnp.ndarray,
    sit: jnp.ndarray,
    coeffs: CarbonateCoeffs,
    n_iter: int = 50,
) -> CarbonateSpecies:
    """H2CO3*, HCO3-, CO3-- [mol/kg] and pH from DIC, TA, PO4, SiO3 [mol/kg]."""
    H = solve_htotal(dic, ta, pt, sit, coeffs, n_iter=n_iter)
    dic_b = jnp.maximum(dic, _DIC_MIN)
    H2 = H * H
    denom = H2 + coeffs.k1 * H + coeffs.k1 * coeffs.k2
    return CarbonateSpecies(
        H2CO3=dic_b * H2 / denom,
        HCO3=dic_b * coeffs.k1 * H / denom,
        CO3=dic_b * coeffs.k1 * coeffs.k2 / denom,
        pH=-jnp.log10(H),
    )


def co3_saturation(
    T_degC: jnp.ndarray,
    S_psu: jnp.ndarray,
    p_bar: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """[CO3--] at calcite and aragonite saturation [mol/kg] (Mucci 1983).

    Ca is taken proportional to salinity: [Ca] = 0.01028 * S / 35 mol/kg.
    Omega = [CO3--] / [CO3--]_sat.
    """
    S = jnp.maximum(S_psu, _SALT_MIN)
    T_K = T_degC + constants.T_freeze
    log10_T = jnp.log10(T_K)
    inv_T = 1.0 / T_K
    sqrt_S = jnp.sqrt(S)
    S15 = sqrt_S * S
    log10_K_calc = (-171.9065 - 0.077993 * T_K + 2839.319 * inv_T + 71.595 * log10_T
                    + (-0.77712 + 0.0028426 * T_K + 178.34 * inv_T) * sqrt_S
                    - 0.07711 * S + 0.0041249 * S15)
    log10_K_arag = (-171.945 - 0.077993 * T_K + 2903.293 * inv_T + 71.595 * log10_T
                    + (-0.068393 + 0.0017276 * T_K + 88.135 * inv_T) * sqrt_S
                    - 0.10018 * S + 0.0059415 * S15)
    K_calc = 10.0 ** log10_K_calc
    K_arag = 10.0 ** log10_K_arag
    if p_bar is not None:
        inv_RT = 1.0 / (_R_CM3_BAR * T_K)
        K_calc = K_calc * _pressure_factor(T_degC, inv_RT, p_bar,
                                           (-48.76, 0.5304, 0.0), (-11.76, 0.3692, 0.0))
        K_arag = K_arag * _pressure_factor(T_degC, inv_RT, p_bar,
                                           (-45.96, 0.5304, 0.0), (-11.76, 0.3692, 0.0))
    inv_Ca = (35.0 / 0.01028) / S
    return K_calc * inv_Ca, K_arag * inv_Ca


# ============================================================================
# Surface pCO2 (mol/m^3 interface used by the npzd / abiotic schemes)
# ============================================================================


def solve_carbonate_system(
    DIC: jnp.ndarray,
    ALK: jnp.ndarray,
    T_degC: jnp.ndarray,
    S_psu: jnp.ndarray,
    rho_sw: float = constants.rho_ocean,
    PO4: jnp.ndarray | float = 0.0,
    SiO3: jnp.ndarray | float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Surface pCO2 [uatm] and pH from DIC, ALK (and PO4, SiO3) in mol/m^3.

    pCO2 = [CO2*] / K0 (Weiss 1974), consistent with
    ``gas_exchange.air_sea_co2_flux``.
    """
    to_kg = 1.0 / rho_sw
    DIC = jnp.asarray(DIC)
    pt = jnp.broadcast_to(jnp.asarray(PO4, dtype=DIC.dtype), DIC.shape)
    sit = jnp.broadcast_to(jnp.asarray(SiO3, dtype=DIC.dtype), DIC.shape)
    coeffs = carbonate_coefficients(T_degC, S_psu)
    spec = carbonate_speciation(DIC * to_kg, ALK * to_kg, pt * to_kg, sit * to_kg, coeffs)
    pCO2 = spec.H2CO3 / coeffs.k0 * 1.0e6
    return pCO2, spec.pH
