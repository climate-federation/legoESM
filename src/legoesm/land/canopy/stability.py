"""Monin-Obukhov similarity theory and canopy aerodynamics.

Provides:
  - Aerodynamic roughness / displacement from canopy height and LAI.
  - Above-canopy MOST stability iteration (4-regime, CLM5 formulation).
  - Below-canopy resistance (clumping-weighted Cs approach).
  - Leaf boundary-layer resistance from wind speed.

All functions are pure JAX, JIT-compatible, and differentiable.
Sources:
  DifferBESS/process/stability.py (Ryu et al. / CLM5 stability functions)
  DifferBESS/process/CarbonWaterFluxes.py (aerodynamic block)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

# Physical constants
_g   = 9.80665    # standard gravity [m s-2]
_kv  = 0.4        # von Kármán constant
_z0mg = 0.01      # bare-soil roughness length [m]
_ZETA_MAX_STABLE = 0.5
_CONV_BDY_HEIGHT = 1000.0  # convective boundary layer height [m]


# ---------------------------------------------------------------------------
# Saturation specific humidity helper
# ---------------------------------------------------------------------------

@jax.jit
def sat_specific_humidity(T: jax.Array, p: jax.Array) -> jax.Array:
    """Saturation specific humidity from temperature and pressure.

    Parameters
    ----------
    T : temperature [K]
    p : pressure [Pa]

    Returns
    -------
    q_sat [kg kg-1]
    """
    e_s = 611.2 * jnp.exp(17.67 * (T - 273.15) / ((T - 273.15) + 243.5))
    return 0.622 * e_s / (p - (1.0 - 0.622) * e_s)


# ---------------------------------------------------------------------------
# Aerodynamics: z0m and displacement height from canopy geometry
# ---------------------------------------------------------------------------

@jax.jit
def compute_aerodynamics(
    hc: jax.Array,
    LAI: jax.Array,
    rz0m: jax.Array,
    rd: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute roughness length z0m and displacement height d from hc and LAI.

    Effective greenness fraction (egvf) blends dense-canopy and bare-soil
    limits based on LAI.  From DifferBESS model.py:compute_default_aerodynamics.

    Parameters
    ----------
    hc    : canopy height [m], shape (ncol,)
    LAI   : leaf area index [m2/m2], shape (ncol,)
    rz0m  : z0m / hc ratio (PFT-specific), shape (ncol,)
    rd    : displacement height / hc ratio (PFT-specific), shape (ncol,)

    Returns
    -------
    z0m    : roughness length for momentum [m]
    displa : displacement height [m]
    """
    LAI_CRIT = 2.0
    LAI_filt = jnp.clip(LAI, 0.0, LAI_CRIT)
    egvf = (1.0 - jnp.exp(-LAI_filt)) / (1.0 - jnp.exp(-LAI_CRIT))

    displa = hc * rd * egvf
    z0m = jnp.exp(
        egvf * jnp.log(jnp.maximum(hc * rz0m, _z0mg))
        + (1.0 - egvf) * jnp.log(_z0mg)
    )
    return z0m, displa


# ---------------------------------------------------------------------------
# MOST stability functions
# ---------------------------------------------------------------------------

def _stability_func_momentum(zeta: jax.Array) -> jax.Array:
    """Ψ_m(ζ) — momentum stability function (unstable branch)."""
    chik2 = jnp.sqrt(1.0 - 16.0 * zeta)
    chik  = jnp.sqrt(chik2)
    return (2.0 * jnp.log((1.0 + chik) * 0.5)
            + jnp.log((1.0 + chik2) * 0.5)
            - 2.0 * jnp.arctan(chik)
            + jnp.pi * 0.5)


def _stability_func_heat(zeta: jax.Array) -> jax.Array:
    """Ψ_h(ζ) — heat stability function (unstable branch)."""
    chik2 = jnp.sqrt(1.0 - 16.0 * zeta)
    return 2.0 * jnp.log((1.0 + chik2) * 0.5)


def _friction_velocity(zldis: jax.Array, z0m: jax.Array,
                       obu: jax.Array, um: jax.Array) -> jax.Array:
    """Compute ustar using 4-regime stability functions (CLM5 / DifferBESS)."""
    zetam = 1.574  # momentum regime transition
    zeta = zldis / obu

    # Very unstable
    ustar1 = _kv * um / (
        jnp.log(-zetam * obu / z0m)
        - _stability_func_momentum(-zetam)
        + _stability_func_momentum(z0m / obu)
        + 1.14 * (jnp.cbrt(-zeta) - jnp.cbrt(zetam))
    )
    # Unstable
    ustar2 = _kv * um / (
        jnp.log(zldis / z0m)
        - _stability_func_momentum(zeta)
        + _stability_func_momentum(z0m / obu)
    )
    # Stable
    ustar3 = _kv * um / (jnp.log(zldis / z0m) + 5.0 * zeta - 5.0 * z0m / obu)
    # Very stable
    ustar4 = _kv * um / (
        jnp.log(obu / z0m) + 5.0 - 5.0 * z0m / obu
        + (5.0 * jnp.log(zeta) + zeta - 1.0)
    )

    ustar = jnp.where(zeta < -zetam, ustar1,
            jnp.where(zeta < 0.0,    ustar2,
            jnp.where(zeta <= 1.0,   ustar3, ustar4)))
    return ustar


def _temperature_humidity_relation(zldis: jax.Array, obu: jax.Array,
                                   z0h: jax.Array) -> jax.Array:
    """Compute θ* / (θ_atm - θ_sfc) (4 regimes, CLM5 / DifferBESS)."""
    zetat = 0.465
    zeta  = zldis / obu

    ch1 = _kv / (
        jnp.log(-zetat * obu / z0h)
        - _stability_func_heat(-zetat)
        + _stability_func_heat(z0h / obu)
        + 0.8 * (1.0 / jnp.cbrt(zetat) - 1.0 / jnp.cbrt(-zeta))
    )
    ch2 = _kv / (
        jnp.log(zldis / z0h)
        - _stability_func_heat(zeta)
        + _stability_func_heat(z0h / obu)
    )
    ch3 = _kv / (jnp.log(zldis / z0h) + 5.0 * zeta - 5.0 * z0h / obu)
    ch4 = _kv / (
        jnp.log(obu / z0h) + 5.0 - 5.0 * z0h / obu
        + (5.0 * jnp.log(zeta) + zeta - 1.0)
    )

    ch = jnp.where(zeta < -zetat, ch1,
         jnp.where(zeta < 0.0,    ch2,
         jnp.where(zeta <= 1.0,   ch3, ch4)))
    return ch


def _monin_obukhov_init(ur: jax.Array, Tv_atm: jax.Array,
                        dthv: jax.Array, zldis: jax.Array,
                        z0m: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Initialise MOST via bulk Richardson number (Zeng et al. 1998)."""
    wc  = 0.5
    um  = jnp.where(dthv >= 0.0, jnp.maximum(ur, 0.1), jnp.sqrt(ur**2 + wc**2))
    rib = _g * zldis * dthv / (Tv_atm * um**2)

    zeta = jnp.where(
        rib >= 0.0,
        rib * jnp.log(zldis / z0m) / (1.0 - 5.0 * jnp.minimum(rib, 0.19)),
        rib * jnp.log(zldis / z0m),
    )
    zeta = jnp.where(
        rib >= 0.0,
        jnp.clip(zeta, 0.01, _ZETA_MAX_STABLE),
        jnp.clip(zeta, -100.0, -0.01),
    )
    obu = zldis / zeta
    return um, obu


def _stability_step(carry: jax.Array, _xs: None,
                    forcing: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Single Monin-Obukhov fixed-point iteration.

    carry : [z0h, obu, um]
    forcing: [Ta, Tv_atm, q_atm, q_atm, zldis, z0m, dq, dth, ur]
    output : [ustar, tstar, qstar, thvstar, ch, zeta]
    """
    z0h, obu, um = carry[0], carry[1], carry[2]
    Ta, Tv_atm, q_atm, _q2, zldis, z0m, dq, dth, ur = (
        forcing[0], forcing[1], forcing[2], forcing[3],
        forcing[4], forcing[5], forcing[6], forcing[7], forcing[8],
    )

    ustar = _friction_velocity(zldis, z0m, obu, um)
    ch    = _temperature_humidity_relation(zldis, obu, z0h)
    tstar = ch * dth
    qstar = ch * dq
    thvstar = tstar * (1.0 + 0.61 * q_atm) + 0.61 * Ta * qstar

    zeta  = zldis * _kv * _g * thvstar / (ustar**2 * Tv_atm)

    zeta_stable = jnp.clip(zeta, 0.01, _ZETA_MAX_STABLE)
    um_stable   = jnp.maximum(ur, 0.1)
    zeta_unstable = jnp.clip(zeta, -100.0, -0.01)
    wc_unstable = jnp.cbrt(jnp.maximum(
        -_g * ustar * thvstar * _CONV_BDY_HEIGHT / Tv_atm, 0.0))
    um_unstable = jnp.sqrt(ur**2 + wc_unstable**2)

    is_stable = zeta >= 0.0
    zeta_new = jnp.where(is_stable, zeta_stable, zeta_unstable)
    um_new   = jnp.where(is_stable, um_stable,   um_unstable)
    obu_new  = zldis / zeta_new

    new_carry = jnp.array([z0h, obu_new, um_new])
    output    = jnp.array([ustar, tstar, qstar, thvstar, ch, zeta_new])
    return new_carry, output


def monin_obukhov_stability(
    ur: jax.Array,
    Ta: jax.Array,
    Tv_atm: jax.Array,
    Tc: jax.Array,
    q_atm: jax.Array,
    q_c: jax.Array,
    zldis: jax.Array,
    z0m: jax.Array,
    n_iters: int = 5,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Above-canopy MOST iteration via jax.lax.scan.

    Parameters
    ----------
    ur     : wind speed at reference height [m/s]
    Ta     : air temperature [K]
    Tv_atm : virtual potential temperature of air [K]
    Tc     : canopy air temperature [K]  (first-guess or converged)
    q_atm  : air specific humidity [kg/kg]
    q_c    : canopy air specific humidity [kg/kg]
    zldis  : reference height minus displacement height [m]
    z0m    : roughness length for momentum [m]
    n_iters: number of fixed-point iterations

    Returns
    -------
    ustar    : friction velocity [m/s]
    rah      : aerodynamic resistance to heat [s/m]
    raw      : aerodynamic resistance to water vapour [s/m]
    uav      : mean wind speed within canopy (used for Rb) [m/s]
    zeta     : stability parameter [-]
    """
    z0h  = z0m / jnp.exp(2.0)   # heat roughness length
    dth  = Ta - Tc
    dq   = q_atm - q_c
    dthv = (Ta - Tc) * (1.0 + 0.61 * q_atm) + 0.61 * Ta * (q_atm - q_c)

    um, obu = _monin_obukhov_init(ur, Tv_atm, dthv, zldis, z0m)

    init    = jnp.array([z0h, obu, um])
    forcing = jnp.array([Ta, Tv_atm, q_atm, q_atm, zldis, z0m, dq, dth, ur])
    step_fn = partial(_stability_step, forcing=forcing)

    final_carry, outputs = jax.lax.scan(step_fn, init, xs=None, length=n_iters)

    ustar, _tstar, _qstar, _thvstar, ch, zeta = (
        outputs[-1, 0], outputs[-1, 1], outputs[-1, 2],
        outputs[-1, 3], outputs[-1, 4], outputs[-1, 5],
    )

    rah = 1.0 / jnp.maximum(ch * ustar, 1e-9)
    raw = rah  # same for sensible heat and water vapour (neutral Prandtl)

    um_final = final_carry[2]
    ram      = jnp.maximum(um_final / ustar**2, 1e-9)
    uav      = jnp.sqrt(um_final / ram)

    return ustar, rah, raw, uav, zeta


# ---------------------------------------------------------------------------
# Leaf boundary-layer resistance
# ---------------------------------------------------------------------------

@jax.jit
def compute_boundary_layer_resistance(
    uav: jax.Array,
    LAI: jax.Array,
    fSun: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Leaf boundary-layer resistance for sunlit and shaded leaves.

    Based on forced convection scaling (DifferBESS / CLM5):
      rb = 1 / (cv * sqrt(uav / d_leaf))
      Rb_Sun = rb / max(LAI * fSun,   1e-6)
      Rb_Sh  = rb / max(LAI * (1-fSun), 1e-6)

    Parameters
    ----------
    uav  : mean wind speed within/at canopy [m/s]
    LAI  : leaf area index [m2/m2]
    fSun : sunlit fraction [-]

    Returns
    -------
    Rb_Sun, Rb_Sh : boundary-layer resistance [s/m]
    """
    d_leaf = 0.04    # characteristic leaf dimension [m]
    cv     = 0.01    # convective transfer coefficient
    rb     = 1.0 / (cv * jnp.sqrt(jnp.maximum(uav / d_leaf, 1e-9)))

    LAI_Sun = jnp.maximum(LAI * fSun,         1e-6)
    LAI_Sh  = jnp.maximum(LAI * (1.0 - fSun), 1e-6)

    Rb_Sun = rb / LAI_Sun
    Rb_Sh  = rb / LAI_Sh
    return Rb_Sun, Rb_Sh


# ---------------------------------------------------------------------------
# Below-canopy aerodynamic resistance
# ---------------------------------------------------------------------------

@jax.jit
def compute_below_canopy_resistance(
    uav: jax.Array,
    CI: jax.Array,
    LAI: jax.Array,
    z0mg: float = _z0mg,
) -> tuple[jax.Array, jax.Array]:
    """Below-canopy aerodynamic resistance using clumping-weighted Cs approach.

    From DifferBESS / CLM5:
      w     = exp(-0.5 * CI * LAI)
      Cs    = Csbare * w + Csdense * (1 - w)
      rah = raw = 1 / (Cs * uav)

    Parameters
    ----------
    uav  : mean wind speed [m/s]
    CI   : clumping index [-]
    LAI  : leaf area index [m2/m2]
    z0mg : bare-soil roughness length [m]

    Returns
    -------
    rah_soil, raw_soil : below-canopy aerodynamic resistance [s/m]
    """
    nu       = 1.5e-5   # kinematic viscosity of air [m2/s]
    Csdense  = 0.004    # dense-canopy drag coefficient
    Csbare   = _kv / 0.13 * (z0mg * jnp.maximum(uav, 1e-3) / nu) ** (-0.45)

    w   = jnp.exp(-0.5 * CI * LAI)
    Cs  = Csbare * w + Csdense * (1.0 - w)
    res = 1.0 / jnp.maximum(Cs * uav, 1e-9)
    return res, res
