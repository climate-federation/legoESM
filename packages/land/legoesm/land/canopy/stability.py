"""Monin-Obukhov similarity theory and canopy aerodynamics.

Provides:
  - Aerodynamic roughness / displacement from canopy height and LAI.
  - Above-canopy MOST stability iteration (CLM5 unstable regimes; Beljaars &
    Holtslag 1991 stable side).
  - Below-canopy resistance (clumping-weighted Cs approach).
  - Leaf boundary-layer resistance from wind speed.

All functions are pure JAX, JIT-compatible, and differentiable.
Sources:
  DifferBESS/process/stability.py (Ryu et al. / CLM5 stability functions)
  DifferBESS/process/CarbonWaterFluxes.py (aerodynamic block)

Faithfulness
------------
The above-canopy MOST core is an exact port of the CLM5 ``FrictionVelocityMod``
4-regime Monin-Obukhov similarity functions (Oleson et al. 2013 CLM5 Tech Note;
Zeng et al. 1998).  ``tests/land/unit/test_canopy_stability_faithful.py`` pins the
per-regime resistance FORMS to round-off (rel 1e-9) against an independent scalar
reimplementation of those functions, at MATCHED ``zeta`` / ``z0`` / ``obu``:

  * momentum ``ustar`` (:func:`_friction_velocity`) and heat/scalar ``ch``
    (:func:`_temperature_humidity_relation`) in the two UNSTABLE regimes —
    very-unstable ``zeta < -zetam`` (mom) / ``< -zetat`` (heat) and unstable;
  * the Paulson (1970) unstable ``psi_m``/``psi_h``
    (:func:`_stability_func_momentum`/:func:`_stability_func_heat`), the
    free-convection matches (momentum ``1.14 * ((-zeta)^1/3 - zetam^1/3)``, heat
    ``0.8 * (zetat^-1/3 - (-zeta)^-1/3)`` — CLM5's INVERSE cube-root);
  * the neutral log-law limit ``ustar -> kappa u / ln(z/z0)`` and continuity of
    the forms across the free-convection transitions (the matches are C0).

DEPARTURE from CLM5 on the STABLE side (``zeta >= 0``, user 2026-09-25): CLM5's
linear ``psi = -5 zeta`` (``0 <= zeta <= 1``) and very-stable log branch are
replaced by Beljaars & Holtslag (1991), the shared
``legoesm.core.bulk_flux.psi_m/psi_h(..., "beljaars_holtslag1991")``.  Under
the CLM5 forms a dense canopy at night could have NO root for the canopy-air
energy balance (sensible heat fell with a growing air-surface temperature
difference), so the canopy solve stalled; BH weakens that decline.  Heat flux is
not strictly monotone for tall rough canopies at low wind (a ~1 W m-2 dip just
below the ``zeta <= 0.5`` cap remains).  See
``docs/land/canopy_stable_stability_bh91.md``.

The pin is on the FORMS at fixed ``zeta``, NOT the whole solve, because the
iteration DRIVER is a departure (see below), so a converged ``zeta`` is
model-specific.

DEPARTURES from the gSAM/CESM-LSM4 sibling MOST ``transfer_coef.f90`` (a related
Businger-Dyer scheme on disk — cross-checked in the shared regimes, NOT the same
scheme; each departure is a test canary):
  * very-unstable HEAT uses CLM5's INVERSE cube-root ``zetat^-1/3 - (-zeta)^-1/3``
    whereas the LSM4 sibling uses a growing ``(-zeta)^1/3 - zetat^1/3`` — the
    heat free-convection correction genuinely differs between the two models;
  * ``kB^-1 = 0`` (``z0h = z0m``, CLM5 vegetation; DifferBESS aa6e8b9) vs the
    LSM4 ``kB^-1 approx 2`` (``z0h = 0.135 z0``, i.e. ``ln(z0/z0h) = 2.0025``,
    the LSM4 rounding of the exact ``kB^-1 = 2`` -> ``z0h = z0 e^-2``);
  * no high-wind roughness reduction ``z0 (1 + U/10)^-0.6`` and no LSM4
    post-solve flux limiters (50%-slowdown cap; the LSM4 sibling also applies an
    UNCONDITIONAL ``ustar -> sqrt(ustar^2 + 0.05^2)`` floor, absent here);
  * the Obukhov solve is a FIXED ``n_iters`` (default 5) buoyancy-flux fixed
    point (``zeta = zldis kappa g thetav*/(ustar^2 Tv)``, Zeng 1998 bulk-Ri init),
    NOT the LSM4 bulk-Richardson ``zeta = r fm^2/fh`` iterated to tolerance.

NUMERICS / AD guards (no-ops in their own regime): ``_MOST_ARG_FLOOR`` floors
the log/cbrt args of the DISCARDED where-branches so ``0*NaN`` cannot poison the
reverse-mode gradient.  The scalar oracle does NOT reproduce this floor — it
evaluates only the in-regime branch (if/elif) and so never touches the discarded
args; the floor is instead exercised by the AD test (eager ``where`` evaluates
every regime).  ``zeta`` is clamped to ``[1e-6, 0.5]`` (stable; the 0.5 cap is a
smooth min, see ``_ZETA_CAP_SMOOTHING_WIDTH`` — CLM5 clips hard) / ``[-100,
-1e-6]`` (unstable) each iterate (CLM5: 0.01, see ``_ZETA_NEUTRAL_FLOOR``); wind floors (0.1, 1e-3 m/s) and resistance
floors (1e-9) guard calm/degenerate columns.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

from legoesm import constants
from legoesm.core.bulk_flux import psi_h, psi_m
from legoesm.thermo import saturation_vapor_pressure_aerk

# Module-local numerics / stability parameters (not physical constants —
# those come from ``legoesm.constants``).
_ZETA_MAX_STABLE = 0.5
# Smooth-min width on the stable cap: a hard clip at 0.5 is a derivative kink
# in zeta (and ustar, rah, Rb) that stalls the canopy Newton solve for tall
# rough canopies in stable low-wind air.  zeta approaches 0.5 asymptotically;
# at zeta = 0.5 it reads 0.5 - w*ln2.
_ZETA_CAP_SMOOTHING_WIDTH = 0.05
# Near-neutral |zeta| floor (keeps obu = zldis/zeta finite).  CLM5 uses 0.01,
# which makes zeta, ustar and the resistances JUMP by that much at neutral
# stability; the canopy Newton solve then stalls whenever its root sits at
# neutral (sunset).  At 1e-6 the jump is below the solver tolerance and the
# stability forms are continuous through zeta = 0 (user 2026-09-25).
_ZETA_NEUTRAL_FLOOR = 1.0e-6
_CONV_BDY_HEIGHT = 1000.0  # convective boundary layer height [m]
_Z0MG_BARE = 0.01          # bare-soil momentum roughness length [m]

# --- CLM5 Monin-Obukhov similarity-theory (MOST) constants ---
_MOST_GAMMA_UNSTABLE = 16.0   # Businger-Dyer (1 - 16 ζ) unstable-branch factor
_MOST_BETA_STABLE    = 5.0    # stable-branch linear slope
_ZETAM = 1.574                # momentum stability-regime transition
_ZETAT = 0.465                # heat stability-regime transition
# Stable-side (zeta >= 0) similarity functions: Beljaars & Holtslag (1991), the
# shared core implementation (user 2026-09-25; departure from CLM5's linear
# -5 zeta + very-stable log forms, see docs/land/canopy_stable_stability_bh91.md).
_STABLE_SCHEME = "beljaars_holtslag1991"
_MOST_MOM_CONV_COEF  = 1.14   # very-unstable momentum convective correction
_MOST_HEAT_CONV_COEF = 0.8    # very-unstable heat convective correction
# Positivity floor applied to log/cbrt arguments in the OUT-OF-REGIME MOST
# branches only (a no-op inside each branch's own regime).  Every regime's
# ustar/ch is evaluated unconditionally and combined with jnp.where, whose VJP
# runs both sides — an out-of-domain sqrt/log/cbrt would return NaN and
# 0*NaN = NaN poisons the reverse-mode gradient of every surface turbulent
# flux.  Flooring keeps the discarded branch finite without touching the
# selected value.  (<=1e-6 grad-safety floor.)
_MOST_ARG_FLOOR = 1e-12
# Per-leaf-class area floor in the boundary-layer resistance rb / LAI_class.
# At or below it a leaf class is bare ground for the canopy closure, which
# pins that column's leaf state instead of solving it (see canopy/solver.py).
LEAF_AREA_FLOOR = 1e-6        # [m2/m2]
_VIRT_T_COEF = 0.61           # virtual-temperature coefficient (≈ 1/ε − 1, rounded)
_RIB_MAX = 0.19               # bulk Richardson-number cap (Zeng et al. 1998 init)

# --- Below-canopy resistance Cs (CLM5 / DifferBESS) ---
_NU_AIR       = 1.5e-5        # kinematic viscosity of air [m2 s-1]
_CS_DENSE     = 0.004         # dense-canopy turbulent transfer coefficient [-]
_CS_BARE_COEF = 0.13          # bare-soil Cs prefactor (κ / 0.13)
_CS_BARE_EXP  = 0.45          # bare-soil Cs Reynolds-number exponent


# ---------------------------------------------------------------------------
# Saturation specific humidity helper
# ---------------------------------------------------------------------------

@jax.jit
def sat_specific_humidity(T: jax.Array, p: jax.Array) -> jax.Array:
    """Saturation specific humidity from temperature and pressure.

    Uses ``legoesm.thermo.saturation_vapor_pressure`` for the Tetens
    formula (CLAUDE.md: never inline Tetens).  Returns specific humidity
    (not mixing ratio), matching the canopy air / leaf boundary-layer
    variable convention.

    Parameters
    ----------
    T : temperature [K]
    p : pressure [Pa]

    Returns
    -------
    q_sat [kg kg-1]
    """
    e_s = saturation_vapor_pressure_aerk(T)
    return constants.epsilon * e_s / (p - (1.0 - constants.epsilon) * e_s)


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
        egvf * jnp.log(jnp.maximum(hc * rz0m, _Z0MG_BARE))
        + (1.0 - egvf) * jnp.log(_Z0MG_BARE)
    )
    return z0m, displa


# ---------------------------------------------------------------------------
# MOST stability functions
# ---------------------------------------------------------------------------

def _stability_func_momentum(zeta: jax.Array) -> jax.Array:
    """Ψ_m(ζ) — momentum stability function (unstable branch).

    Valid only for ζ ≤ 0.  Callers in the stable regimes never use it, but the
    where-combined ustar evaluates it unconditionally, so clamp ζ ≤ 0 here to
    keep the sqrt argument ≥ 1 (grad-safe); a no-op for every in-regime call.
    """
    zeta = jnp.minimum(zeta, 0.0)
    chik2 = jnp.sqrt(1.0 - _MOST_GAMMA_UNSTABLE * zeta)
    chik  = jnp.sqrt(chik2)
    return (2.0 * jnp.log((1.0 + chik) * 0.5)
            + jnp.log((1.0 + chik2) * 0.5)
            - 2.0 * jnp.arctan(chik)
            + jnp.pi * 0.5)


def _stability_func_heat(zeta: jax.Array) -> jax.Array:
    """Ψ_h(ζ) — heat stability function (unstable branch).

    Valid only for ζ ≤ 0 (see ``_stability_func_momentum``); clamp for
    grad-safety, a no-op for every in-regime call.
    """
    zeta = jnp.minimum(zeta, 0.0)
    chik2 = jnp.sqrt(1.0 - _MOST_GAMMA_UNSTABLE * zeta)
    return 2.0 * jnp.log((1.0 + chik2) * 0.5)


def _friction_velocity(zldis: jax.Array, z0m: jax.Array,
                       obu: jax.Array, um: jax.Array) -> jax.Array:
    """Compute ustar using 4-regime stability functions (CLM5 / DifferBESS)."""
    zetam = _ZETAM  # momentum regime transition
    zeta = zldis / obu

    # Very unstable (valid: obu < 0, zeta < -zetam).  Floor the log/cbrt args so
    # the discarded (obu > 0) branch stays finite; both floors are no-ops here.
    ustar1 = constants.kappa_vk * um / (
        jnp.log(jnp.maximum(-zetam * obu / z0m, _MOST_ARG_FLOOR))
        - _stability_func_momentum(-zetam)
        + _stability_func_momentum(z0m / obu)
        + _MOST_MOM_CONV_COEF * (jnp.cbrt(jnp.maximum(-zeta, zetam)) - jnp.cbrt(zetam))
    )
    # Unstable
    ustar2 = constants.kappa_vk * um / (
        jnp.log(zldis / z0m)
        - _stability_func_momentum(zeta)
        + _stability_func_momentum(z0m / obu)
    )
    # Stable (zeta >= 0): Beljaars & Holtslag (1991), all stable zeta.  psi_m
    # is finite for the negative arguments of the discarded (obu < 0) branch.
    ustar3 = constants.kappa_vk * um / (
        jnp.log(zldis / z0m)
        - psi_m(zeta, _STABLE_SCHEME) + psi_m(z0m / obu, _STABLE_SCHEME))

    ustar = jnp.where(zeta < -zetam, ustar1,
            jnp.where(zeta < 0.0,    ustar2, ustar3))
    return ustar


def _temperature_humidity_relation(zldis: jax.Array, obu: jax.Array,
                                   z0h: jax.Array) -> jax.Array:
    """Compute θ* / (θ_atm - θ_sfc) (4 regimes, CLM5 / DifferBESS)."""
    zetat = _ZETAT
    zeta  = zldis / obu

    # Very unstable (valid: obu < 0, zeta < -zetat).  Floor log/cbrt args; no-ops here.
    ch1 = constants.kappa_vk / (
        jnp.log(jnp.maximum(-zetat * obu / z0h, _MOST_ARG_FLOOR))
        - _stability_func_heat(-zetat)
        + _stability_func_heat(z0h / obu)
        + _MOST_HEAT_CONV_COEF * (1.0 / jnp.cbrt(zetat)
                                  - 1.0 / jnp.cbrt(jnp.maximum(-zeta, zetat)))
    )
    ch2 = constants.kappa_vk / (
        jnp.log(zldis / z0h)
        - _stability_func_heat(zeta)
        + _stability_func_heat(z0h / obu)
    )
    # Stable (zeta >= 0): Beljaars & Holtslag (1991), all stable zeta.  The core
    # psi_h floors its stable argument at 1e-10, so the (1 + 2z/3)^1.5 term stays
    # finite for the negative arguments of the discarded (obu < 0) branch.
    ch3 = constants.kappa_vk / (
        jnp.log(zldis / z0h)
        - psi_h(zeta, _STABLE_SCHEME) + psi_h(z0h / obu, _STABLE_SCHEME))

    ch = jnp.where(zeta < -zetat, ch1,
         jnp.where(zeta < 0.0,    ch2, ch3))
    return ch


def _cap_stable_zeta(zeta: jax.Array) -> jax.Array:
    """Smooth cap at ``_ZETA_MAX_STABLE`` (no kink), then the neutral floor.

    Floor LAST: the softplus leaks ~w*exp(-0.5/w) below the cap, which would
    push a floored 1e-6 negative (into the unstable forms).
    """
    w = _ZETA_CAP_SMOOTHING_WIDTH
    z = zeta - w * jax.nn.softplus((zeta - _ZETA_MAX_STABLE) / w)
    return jnp.maximum(z, _ZETA_NEUTRAL_FLOOR)


def _monin_obukhov_init(ur: jax.Array, Tv_atm: jax.Array,
                        dthv: jax.Array, zldis: jax.Array,
                        z0m: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Initialise MOST via bulk Richardson number (Zeng et al. 1998)."""
    wc  = 0.5
    um  = jnp.where(dthv >= 0.0, jnp.maximum(ur, 0.1), jnp.sqrt(ur**2 + wc**2))  # coeff-ok: 0.1 m/s wind floor
    rib = constants.g * zldis * dthv / (Tv_atm * um**2)

    zeta = jnp.where(
        rib >= 0.0,
        rib * jnp.log(zldis / z0m) / (1.0 - _MOST_BETA_STABLE * jnp.minimum(rib, _RIB_MAX)),
        rib * jnp.log(zldis / z0m),
    )
    zeta = jnp.where(
        rib >= 0.0,
        _cap_stable_zeta(zeta),
        jnp.clip(zeta, -100.0, -_ZETA_NEUTRAL_FLOOR),  # coeff-ok: very-unstable bound on ζ
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
    thvstar = tstar * (1.0 + _VIRT_T_COEF * q_atm) + _VIRT_T_COEF * Ta * qstar

    zeta  = zldis * constants.kappa_vk * constants.g * thvstar / (ustar**2 * Tv_atm)

    zeta_stable = _cap_stable_zeta(zeta)
    um_stable   = jnp.maximum(ur, 0.1)                     # coeff-ok: 0.1 m/s wind floor
    zeta_unstable = jnp.clip(zeta, -100.0, -_ZETA_NEUTRAL_FLOOR)  # coeff-ok: very-unstable bound on ζ
    # Floor the cbrt argument to a POSITIVE value, not 0: cbrt'(0)=inf and the
    # maximum's subgradient is 0 below the clamp, so cbrt(maximum(x, 0)) gives
    # 0*inf = NaN in the reverse-mode gradient whenever x<=0 (the stable regime,
    # where this unstable-branch wc is discarded).  cbrt(1e-12)~1e-4 m/s ~ 0.
    wc_unstable = jnp.cbrt(jnp.maximum(
        -constants.g * ustar * thvstar * _CONV_BDY_HEIGHT / Tv_atm, _MOST_ARG_FLOOR))
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
    # Heat roughness length.  CLM5 vegetation uses kB^-1 = 0 (z0h = z0m); the
    # earlier BESS default kB^-1 = 2 (z0h = z0m / e^2) over-suppressed heat
    # exchange and warm-biased canopy/skin temperature (DifferBESS aa6e8b9).
    z0h  = z0m   # kB^-1 = 0
    dth  = Ta - Tc
    dq   = q_atm - q_c
    dthv = (Ta - Tc) * (1.0 + _VIRT_T_COEF * q_atm) + _VIRT_T_COEF * Ta * (q_atm - q_c)

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
    cv: jax.Array,
    d_leaf: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Leaf boundary-layer resistance for sunlit and shaded leaves.

    Forced-convection scaling (Campbell & Norman 1998 / CLM5 / DifferBESS):
      rb = 1 / (cv * sqrt(uav / d_leaf))
      Rb_Sun = rb / max(LAI * fSun,   1e-6)
      Rb_Sh  = rb / max(LAI * (1-fSun), 1e-6)

    Parameters
    ----------
    uav    : mean wind speed within/at canopy [m/s]
    LAI    : leaf area index [m2/m2]
    fSun   : sunlit fraction [-]
    cv     : forced-convection transfer coefficient [m^-0.5 s^0.5]
             (CanopyConfig.cv, default 0.0135; was the hard-coded BESS 0.01)
    d_leaf : characteristic leaf width [m] (per-column PFT_LEAF_WIDTH;
             default 0.025; was the hard-coded BESS 0.04)

    Returns
    -------
    Rb_Sun, Rb_Sh : boundary-layer resistance [s/m]
    """
    rb     = 1.0 / (cv * jnp.sqrt(jnp.maximum(uav / d_leaf, 1e-9)))

    LAI_Sun = jnp.maximum(LAI * fSun,         LEAF_AREA_FLOOR)
    LAI_Sh  = jnp.maximum(LAI * (1.0 - fSun), LEAF_AREA_FLOOR)

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
    z0mg: float = _Z0MG_BARE,
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
    nu       = _NU_AIR    # kinematic viscosity of air [m2/s]
    Csdense  = _CS_DENSE  # dense-canopy turbulent transfer coefficient
    Csbare   = constants.kappa_vk / _CS_BARE_COEF * (z0mg * jnp.maximum(uav, 1e-3) / nu) ** (-_CS_BARE_EXP)  # coeff-ok: 1e-3 m/s wind floor

    w   = jnp.exp(-0.5 * CI * LAI)
    Cs  = Csbare * w + Csdense * (1.0 - w)
    res = 1.0 / jnp.maximum(Cs * uav, 1e-9)
    return res, res
