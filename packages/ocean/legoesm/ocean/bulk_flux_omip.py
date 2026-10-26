"""OMIP-2 / CORE-II open-ocean bulk flux formulas (NCAR / Large & Yeager).

References
----------
Large, W. G., & Yeager, S. G. (2004). "Diurnal to decadal global forcing
for ocean and sea-ice models", NCAR/TN-460+STR (the NCAR algorithm).

Large, W. G., & Yeager, S. G. (2009). "The global climatology of an
interannually varying air-sea flux data set", Climate Dynamics 33,
341-364 (Eq. 6 neutral drag, including the U^6 high-wind correction).

NEMO 5.0.1 ``sbcblk_algo_ncar.F90`` / ``sbc_phy.F90`` (Brodeau's aerobulk
port) — the EXACT reference implementation this module mirrors, because the
OMIP-faithful comparison target (NEMO ORCA1, ``ln_NCAR=.true.``) computes its
surface fluxes with that code.

What
----
Converts the CORE-II forcing channels (u10, v10, T_air, q_air, and optionally
slp) plus the model SST into the open-ocean surface fluxes:

* ``tau_x``, ``tau_y``  -- wind stress [Pa], ATMOSPHERIC convention
  (``-rho Cd |U| u``, opposing the wind; the dynamics core applies the
  ``-tau`` ocean reaction, see ``ocean_pe_latlon_cgrid``)
* ``shflx``             -- sensible heat flux [W/m^2] (positive into ocean)
* ``lhflx``             -- latent heat flux [W/m^2]   (positive into ocean)
* ``evap``              -- evaporation [kg/m^2/s]     (positive UP, >= 0 when
  evaporating; exactly ``-lhflx / L_vap(SST)``)

Algorithms (``algo=``)
----------------------
``"ncar"`` (default, NEMO-faithful): the full stability-iterated NCAR
algorithm at ``zt = zu = 10 m`` (the CORE-II measurement heights, so the
height-shift branch of NEMO's ``turb_ncar`` is exactly zero and omitted):

0. NEMO temperature preprocessing (blk_oce_1 / sbc_phy, ``ln_tair_pot=F``):
   10-m pressure ``p10 = pres_temp(q, slp, 10, T_abs)`` (3-iteration
   barometric with moist molar mass + Goff saturation), potential air
   temperature ``theta_air = theta_exner(T_abs, p10) = T (1e5/p10)^gamma_dry``
   and potential SST ``theta_sst = theta_exner(SST, slp)``.  The saturation
   humidity stays at the ABSOLUTE SST: ``ssq = 0.98 q_sat_goff(SST, slp)``.
1. ``Ub = max(|U10|, 0.5)``; neutral coefficients ``CdN`` (LY09 Eq. 6 with
   NEMO's cyclone plateau + ``1e-4`` floor), ``CeN = 34.6e-3 sqrt(CdN)``,
   ``ChN = (32.7 unstable | 18 stable) e-3 sqrt(CdN)`` from the
   ``theta_air/theta_sst`` virtual-temperature difference.
2. 5 fixed-point iterations (NEMO ``nb_iter0``): friction scales
   ``u* = sqrt(Cd) Ub``, ``theta* = Ch/sqrt(Cd) dtheta``,
   ``q* = Ce/sqrt(Cd) dq``; Obukhov ``1/L`` (virtual-temperature form,
   ``|1/L| <= 200``); ``zeta = 10/L`` clipped to ``|zeta| <= 10``;
   neutral-wind update ``UN10 = Ub (1 + sqrt(Cd)/kappa psi_m)`` (>= 0.25);
   coefficient updates ``Cd = CdN/(1 - sqrt(CdN)/kappa psi_m)^2`` and
   ``Cx = CxN r / (1 + CxN (-psi_h)/(kappa sqrt(CdN)))`` with
   ``r = sqrt(Cd/CdN)``, all floored at ``1e-4``.
3. Fluxes (NEMO BULK_FORMULA): moist-air density ``rho(T_abs, q, p10)``
   (floored at 0.8; the assembly uses ``Ub max(rho, 1.0)``), moist
   ``cp_air(q)``, sensible on ``(theta_air - theta_sst)``, latent with
   ``L_vap(theta_sst)`` — exactly the (potential/absolute) mix NEMO uses.

The NCAR path retains NEMO's literal Paulson/Dyer statement order, including
its arithmetic stable/unstable blend and the ``|zeta| <= 10`` clip.  The
general MOST solver keeps its independently configurable stability functions.

``"ly09_2coeff"`` (legacy): the previous two-constant approximation
(``Ch = Ce in {1.46e-3 unstable, 1.18e-3 stable}``, constant air density,
caller-supplied ``q_sfc``).  Kept ONLY as a pinned regression scheme: it
biases the turbulent fluxes by 10-30 % vs the NCAR reference and was the
production scheme for all OMIP runs before 2026-06-10.

All operations are pure JAX (float64), fixed iteration count (statically
unrolled — JIT/vmap/grad safe, no data-dependent control flow).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core import transcendentals as tx
from legoesm.core.source_rounding import (
    nemo_fortran_max,
    nemo_fortran_min,
    nemo_fortran_sign as _sign,
    nemo_source_barrier as B,
    nemo_source_round as R,
)
from legoesm.ocean.eos import nemo_eos_fzp_eos80
from legoesm.ocean.forcing.nemo_fld_read import nemo_fld_time_interp
from legoesm.ocean.forcing.nemo_surface import (
    NemoSurfaceMeans,
    NemoTStressLbcPlan,
    nemo_orca2_s2,
    nemo_sbc_ssm_step,
    nemo_s2_to_ocean_interfaces,
    nemo_t_stress_lbc_neighbors,
)
from legoesm.thermo import saturation_vapor_pressure_goff


_U10_FLOOR_M_S: float = 0.5    # NEMO turb_ncar: Ubzu = MAX(0.5, U)
_CH_UNSTABLE: float = 1.46e-3  # legacy ly09_2coeff only
_CH_STABLE: float = 1.18e-3    # legacy ly09_2coeff only

_ZU_M: float = 10.0            # CORE-II wind/temperature/humidity height [m]
_NB_ITER_NCAR: int = 5         # NEMO sbc_phy nb_iter0
_CX_MIN: float = 0.1e-3        # NEMO sbc_phy Cx_min (coefficient floor)
_UN10_FLOOR_M_S: float = 0.25  # NEMO turb_ncar neutral-wind floor
_ZETA_ABS_MAX: float = 10.0    # NEMO |z/L| cap
_ONE_ON_L_ABS_MAX: float = 200.0  # NEMO One_on_L |1/L| cap [1/m]
_RHO_AIR_FLOOR: float = 0.8    # NEMO rho_air floor [kg/m^3]
_RHO_FLUX_FLOOR: float = 1.0   # NEMO BULK_FORMULA zUrho = Ub*MAX(rho, 1.0)
_QSAT_SALT_FACTOR: float = 0.98  # NEMO sbc_phy rdct_qsat_salt
# NEMO-parity derived constants (sbc_phy): rctv0 = R_vap/R_dry - 1,
# reps0 = R_dry/R_vap, gamma_dry = R_gas/(M_dry cp_dry) — all derived from
# the NEMO base constants so the parity chain stays exact.
_RCTV0: float = constants.R_v_nemo / constants.R_d - 1.0
_REPS0: float = constants.R_d / constants.R_v_nemo
_GAMMA_DRY: float = constants.R_gas_molar / (
    constants.M_dry_air * constants.c_p_dry_air_nemo)
_P_EXNER_REF: float = 1.0e5    # NEMO sbc_phy rpref [Pa]
_N_ITER_PRES: int = 3          # NEMO pres_temp barometric iterations

_VALID_OMIP_BULK_ALGOS = ("ncar", "ly09_2coeff")

# --- NEMO sbc_ice_if constants (sbcice_if.f90:154, :192) ---
_IIF_ZTRP_W_M2_K: float = -40.0   # restoring term for temperature [W/m2/K]
_IIF_ZQI_OFFSET_W_M2: float = -3.0  # zqi = -3 + SIGN(1, ff_f): -2 N, -4 S


def _fmax(first, second):
    """Record-object MAXSD ordering: the second source operand wins a tie."""
    return nemo_fortran_max(first, second)


def _fmin(first, second):
    """Record-object MINSD ordering: the second source operand wins a tie."""
    return nemo_fortran_min(first, second)


def _q_sat_nemo(T_K, p_Pa):
    e_s = B(saturation_vapor_pressure_goff(B(T_K), nemo_source_exact=True))
    numerator = R(_REPS0 * e_s)
    pressure_correction = R(R(1.0 - _REPS0) * e_s)
    denominator = R(B(p_Pa) - pressure_correction)
    return R(numerator / B(denominator))


def exner_potential_temperature(T_K, p_Pa, *, nemo_source_exact: bool = False):
    """Potential temperature via the Exner function (NEMO ``theta_exner``).

    ``theta = T (rpref / p)^gamma_dry`` with ``rpref = 1e5 Pa`` and the
    dry-air Poisson constant ``gamma_dry = R_gas/(M_dry cp_dry)``.  Used for
    BOTH the 10-m air temperature (with the 10-m pressure) and the potential
    sea-surface temperature (with the sea-level pressure), exactly like
    NEMO blk_oce_1.
    """
    T = jnp.asarray(T_K, dtype=jnp.float64)
    p = jnp.asarray(p_Pa, dtype=jnp.float64)
    if nemo_source_exact:
        pressure_ratio = R(_P_EXNER_REF / p)
        return R(T * tx.pow(pressure_ratio, _GAMMA_DRY))
    return T * (_P_EXNER_REF / p) ** _GAMMA_DRY


def pressure_at_height(
    q_air, slp_Pa, z_m, T_abs_K, *, nemo_source_exact: bool = False
):
    """Air pressure at height ``z_m`` (NEMO ``pres_temp``, absolute-T branch).

    3 fixed-point iterations of the barometric law with the MOIST molar mass
    ``xm = (1 - q/q_sat) M_dry + (q/q_sat) M_water`` (q_sat from the Goff
    curve at the CURRENT pressure iterate):

        p <- slp * exp( -g xm z / (R_gas T) )

    Static iteration count — jit/grad-safe.
    """
    q = jnp.asarray(q_air, dtype=jnp.float64)
    slp = jnp.asarray(slp_Pa, dtype=jnp.float64)
    if not nemo_source_exact:
        T = jnp.maximum(jnp.asarray(T_abs_K, dtype=jnp.float64), 180.0)
        p = slp
        for _ in range(_N_ITER_PRES):
            e_s = saturation_vapor_pressure_goff(T)
            q_sat = _REPS0 * e_s / (p - (1.0 - _REPS0) * e_s)
            ratio = jnp.clip(q / q_sat, 0.0, 2.0)
            xm = ((1.0 - ratio) * constants.M_dry_air
                  + ratio * constants.M_water)
            p = slp * jnp.exp(
                -constants.g_nemo * xm * z_m / (constants.R_gas_molar * T))
        return p

    q = B(q)
    slp = B(slp)
    T = B(jnp.asarray(T_abs_K, dtype=jnp.float64))
    p = slp
    for _ in range(_N_ITER_PRES):
        q_sat = B(_q_sat_nemo(B(T), B(p)))
        ratio = R(q / q_sat)
        dry_fraction = R(1.0 - ratio)
        dry_mass = R(dry_fraction * constants.M_dry_air)
        water_mass = R(ratio * constants.M_water)
        xm = R(dry_mass + water_mass)
        numerator = R(R(R(-constants.g_nemo * xm) * z_m))
        denominator = B(R(constants.R_gas_molar * T))
        exponent = R(numerator / denominator)
        p = R(slp * B(tx.exp(exponent)))
    return p


def _pressure_from_potential(q_air, slp_Pa, z_m, theta_K):
    """Literal potential-temperature arm of NEMO ``pres_temp``."""
    q = B(jnp.asarray(q_air, dtype=jnp.float64))
    slp = B(jnp.asarray(slp_Pa, dtype=jnp.float64))
    theta = B(jnp.asarray(theta_K, dtype=jnp.float64))
    p = slp
    T = jnp.zeros_like(theta)
    for _ in range(_N_ITER_PRES):
        pressure_ratio = R(p / B(_P_EXNER_REF))
        power = B(tx.pow(pressure_ratio, _GAMMA_DRY))
        T = R(theta * power)
        q_sat = B(_q_sat_nemo(B(T), B(p)))
        ratio = R(q / q_sat)
        dry_fraction = R(1.0 - ratio)
        dry_mass = R(dry_fraction * constants.M_dry_air)
        water_mass = R(ratio * constants.M_water)
        xm = R(dry_mass + water_mass)
        numerator = R(R(R(-constants.g_nemo * xm) * z_m))
        denominator = B(R(constants.R_gas_molar * T))
        exponent = R(numerator / denominator)
        p = R(slp * B(tx.exp(exponent)))
    return p, T


def potential_air_temperature_10m(
    T_air_K, q_air, slp_Pa=None, *, nemo_source_exact: bool = False
):
    """NEMO blk_oce_1 air-temperature preprocessing at zt = 10 m.

    Returns ``(theta_air, p10)``: the potential air temperature
    ``theta_exner(T_abs, p10)`` and the 10-m pressure from
    :func:`pressure_at_height`.  ``slp_Pa=None`` -> standard atmosphere.
    """
    p_slp = constants.p_atm_std if slp_Pa is None else slp_Pa
    p10 = pressure_at_height(
        q_air, p_slp, _ZU_M, T_air_K, nemo_source_exact=nemo_source_exact
    )
    return exner_potential_temperature(
        T_air_K, p10, nemo_source_exact=nemo_source_exact
    ), p10


def seawater_q_sat(T_sfc_K, slp_Pa=None, *, nemo_source_exact: bool = False):
    """Salt-reduced saturation specific humidity at the sea surface [kg/kg].

    NEMO: ``ssq = rdct_qsat_salt * q_sat(SST, slp)`` with the Goff (1957)
    vapour-pressure curve and ``q = eps e/(p - (1-eps) e)``.  The 0.98 factor
    is the vapour-pressure lowering over saline water (Large & Yeager 2004,
    NEMO sbc_phy ``rdct_qsat_salt``) — omitting it overestimates the sea-air
    humidity difference and hence evaporation by ~10 % in the tropics.
    ``slp_Pa=None`` falls back to the standard atmosphere.
    """
    p = constants.p_atm_std if slp_Pa is None else slp_Pa
    T = jnp.asarray(T_sfc_K, dtype=jnp.float64)
    p = jnp.asarray(p, dtype=jnp.float64)
    if nemo_source_exact:
        q = B(_q_sat_nemo(B(T), B(p)))
        return R(_QSAT_SALT_FACTOR * q)
    e_s = saturation_vapor_pressure_goff(T)
    q = _REPS0 * e_s / (p - (1.0 - _REPS0) * e_s)
    return _QSAT_SALT_FACTOR * q


def rho_air_moist(T_air_K, q_air, slp_Pa=None, *, nemo_source_exact: bool = False):
    """Moist-air density from the ideal-gas law [kg/m^3] (NEMO ``rho_air``).

    ``rho = slp / (R_d T (1 + 0.608 q))``, floored at 0.8 like NEMO.
    ``slp_Pa=None`` -> standard atmosphere (constant-pressure fallback for
    forcing sets without an slp channel).
    """
    p = constants.p_atm_std if slp_Pa is None else slp_Pa
    if nemo_source_exact:
        T = jnp.asarray(T_air_K, dtype=jnp.float64)
        q = jnp.asarray(q_air, dtype=jnp.float64)
        humidity_factor = R(1.0 + R(_RCTV0 * q))
        denominator = R(R(constants.R_d * T) * humidity_factor)
        rho = R(jnp.asarray(p, dtype=jnp.float64) / denominator)
        return R(_fmax(rho, _RHO_AIR_FLOOR))
    T = jnp.maximum(jnp.asarray(T_air_K, dtype=jnp.float64), 180.0)
    q = jnp.maximum(jnp.asarray(q_air, dtype=jnp.float64), 1.0e-6)
    rho = jnp.asarray(p, dtype=jnp.float64) / (
        constants.R_d * T * (1.0 + _RCTV0 * q)
    )
    return jnp.maximum(rho, _RHO_AIR_FLOOR)


def latent_heat_vaporization_sst(
    T_sfc_K, *, nemo_source_exact: bool = False
):
    """SST-dependent latent heat, with a selectable NEMO-literal arm."""
    T = jnp.asarray(T_sfc_K, dtype=jnp.float64)
    if nemo_source_exact:
        delta = R(T - constants.T_freeze)
        slope = R(0.00237 * delta)
        return R(R(2.501 - slope) * 1.0e6)
    from legoesm.thermo import latent_heat_vaporization as _l
    return _l(T)


def moist_air_cp(q_air, *, nemo_source_exact: bool = False):
    """Moist-air specific heat [J/(kg K)] (NEMO ``cp_air``).

    NEMO-parity float64 pin around the shared
    :func:`legoesm.thermo.moist_air_cp` (#762).
    """
    q = jnp.asarray(q_air, dtype=jnp.float64)
    if nemo_source_exact:
        return R(constants.c_p_dry_air_nemo + R(1860.0 * q))
    from legoesm.thermo import moist_air_cp as _cp_moist
    return _cp_moist(q)


def _virt_temp(T_K, q):
    """(Absolute or potential) virtual temperature (NEMO ``virt_temp``)."""
    return T_K * (1.0 + _RCTV0 * q)


def _one_on_obukhov(theta_K, q, u_star, theta_star, q_star):
    """Parent/public inverse Obukhov implementation.

    Keep this separate from ``_one_on_obukhov_nemo``.  The two expressions
    are physically equivalent, but the literal NEMO arm deliberately inserts
    source-statement barriers and implements Fortran ``SIGN``/``MIN``.  They
    are not bit-identical, while this arm is the compatibility boundary for
    every pre-existing caller.
    """
    zqa = 1.0 + _RCTV0 * q
    inv_L = (
        constants.g_nemo * constants.kappa_von_karman
        * (theta_star * zqa + _RCTV0 * theta_K * q_star)
        / jnp.maximum(u_star * u_star * theta_K * zqa, 1.0e-9)
    )
    return jnp.clip(inv_L, -_ONE_ON_L_ABS_MAX, _ONE_ON_L_ABS_MAX)


def _one_on_obukhov_nemo(theta_K, q, u_star, theta_star, q_star):
    """Inverse Obukhov length 1/L [1/m] (NEMO ``One_on_L``), |1/L| <= 200.

    Virtual-temperature flux form (option (a) in NEMO):
    ``<w'thv'> ~ theta*(1 + 0.608 q) + 0.608 theta q*``.
    """
    theta_K = B(theta_K)
    q = B(q)
    u_star = B(u_star)
    theta_star = B(theta_star)
    q_star = B(q_star)
    zqa = R(1.0 + R(_RCTV0 * q))
    theta_flux = R(theta_star * B(zqa))
    humidity_scale = R(_RCTV0 * theta_K)
    humidity_flux = R(humidity_scale * q_star)
    virtual_flux = R(theta_flux + humidity_flux)
    gravity_kappa = R(constants.g_nemo * constants.kappa_von_karman)
    numerator = R(gravity_kappa * virtual_flux)
    ustar_squared = R(u_star * u_star)
    denominator = R(ustar_squared * theta_K)
    denominator = R(denominator * zqa)
    denominator = R(_fmax(denominator, 1.0e-9))
    inv_L = R(numerator / B(denominator))
    return R(_sign(_fmin(jnp.abs(inv_L), _ONE_ON_L_ABS_MAX), inv_L))


def _ch_n10(sqrt_cdn, stable_01):
    """Parent/public neutral sensible-heat coefficient.

    The adjacent NEMO form retains statement barriers and Fortran maximum
    semantics; sharing it would change legacy words.  Only ``_virt_temp`` is
    common because its two arms are measured bit-identical.
    """
    return jnp.maximum(
        1.0e-3 * sqrt_cdn * (18.0 * stable_01 + 32.7 * (1.0 - stable_01)),
        _CX_MIN,
    )


def _ch_n10_nemo(sqrt_cdn, stable_01):
    """Neutral 10-m sensible-heat coefficient (LY08 Eq. 9/12)."""
    value = R(
        1.0e-3 * sqrt_cdn * (18.0 * stable_01 + 32.7 * (1.0 - stable_01)),
    )
    return R(_fmax(value, _CX_MIN))


def _ce_n10(sqrt_cdn):
    """Parent/public neutral latent-heat coefficient."""
    return jnp.maximum(1.0e-3 * 34.6 * sqrt_cdn, _CX_MIN)


def _ce_n10_nemo(sqrt_cdn):
    """Neutral 10-m latent-heat coefficient (LY08 Eq. 6b)."""
    scaled_root = R(34.6 * sqrt_cdn)
    value = R(1.0e-3 * scaled_root)
    return R(_fmax(value, _CX_MIN))


def _un10_from_cd(zu, bulk_wind, cd, psi):
    """Literal ``z0_from_Cd`` then ``UN10_from_CD`` exp/log chain."""
    sqrt_cd = R(jnp.sqrt(cd))
    k_over_root = R(B(constants.kappa_von_karman) / B(sqrt_cd))
    exponent = R(-R(k_over_root + psi))
    z0 = R(zu * tx.exp(exponent))
    friction_wind = R(sqrt_cd * bulk_wind)
    velocity_scale = R(friction_wind / B(constants.kappa_von_karman))
    height_ratio = R(10.0 / z0)
    return R(velocity_scale * tx.log(height_ratio))


def _ncar_transfer_coefficients_exact(theta_air_K, q_air, theta_sst_K, q_sfc,
                                      wind_speed, *, nb_iter: int):
    """Stability-iterated NCAR transfer coefficients at zt = zu = 10 m.

    Exact port of NEMO ``turb_ncar`` for the CORE-II case (all forcing at
    10 m, so ``l_zt_equal_zu`` and every ``LOG(zu/10)`` term vanishes).
    Both temperatures are POTENTIAL (NEMO passes ``theta_air_zt`` and the
    Exner potential SST ``zsspt``); ``q_sfc`` is the salt-reduced saturation
    humidity at the ABSOLUTE SST.  ``nb_iter`` is a static Python int
    (fixed unrolled loop: differentiable, no traced control flow).

    Returns ``(Cd, Ch, Ce, Ub)`` with ``Ub = max(wind_speed, 0.5)``.
    """
    from legoesm.core.bulk_flux import (
        large_yeager_neutral_cd,
        psi_h,
        psi_m,
    )

    k = constants.kappa_von_karman
    theta = jnp.asarray(theta_air_K, dtype=jnp.float64)
    q = jnp.asarray(q_air, dtype=jnp.float64)
    sst = jnp.asarray(theta_sst_K, dtype=jnp.float64)
    ssq = jnp.asarray(q_sfc, dtype=jnp.float64)

    Ub = R(_fmax(_U10_FLOOR_M_S,
                  jnp.asarray(wind_speed, dtype=jnp.float64)))

    # First-guess stability from the air-sea virtual-temperature difference.
    dTv = R(_virt_temp(theta, q) - _virt_temp(sst, ssq))
    stable = R(0.5 + _sign(0.5, dTv))

    CdN = large_yeager_neutral_cd(
        Ub, nemo_parity=True, nemo_source_exact=True
    )
    sqrt_CdN = R(jnp.sqrt(CdN))
    Cd = R(CdN)
    Ch = _ch_n10_nemo(sqrt_CdN, stable)
    Ce = _ce_n10_nemo(sqrt_CdN)
    sqrt_Cd = R(sqrt_CdN)

    theta_zu = R(_fmax(theta, 180.0))
    q_zu = R(_fmax(q, 1.0e-6))

    for _ in range(int(nb_iter)):
        d_theta = R(theta_zu - sst)
        d_q = R(q_zu - ssq)
        u_star = R(sqrt_Cd * Ub)
        theta_star = R(Ch / sqrt_Cd * d_theta)
        q_star = R(Ce / sqrt_Cd * d_q)

        inv_L = _one_on_obukhov_nemo(
            theta_zu, q_zu, u_star, theta_star, q_star
        )
        zeta = R(_ZU_M * inv_L)
        zeta = R(_sign(_fmin(jnp.abs(zeta), _ZETA_ABS_MAX), zeta))
        psim = psi_m(zeta, nemo_source_exact=True)

        UN10 = R(_fmax(
            _UN10_FLOOR_M_S, _un10_from_cd(_ZU_M, Ub, Cd, psim)
        ))

        CdN = large_yeager_neutral_cd(
            UN10, nemo_parity=True, nemo_source_exact=True
        )
        sqrt_CdN = R(jnp.sqrt(CdN))

        log_height = tx.log(R(_ZU_M / 10.0))
        psi_difference = R(log_height - psim)
        neutral_over_k = R(sqrt_CdN / B(k))
        stability_term = R(neutral_over_k * psi_difference)
        denom = R(1.0 + stability_term)
        denom_squared = R(denom * denom)
        Cd = R(_fmax(CdN / B(denom_squared), _CX_MIN))
        sqrt_Cd = R(jnp.sqrt(Cd))

        height_log = log_height
        heat_psi = psi_h(zeta, nemo_source_exact=True)
        heat_difference = R(height_log - heat_psi)
        heat_over_k = R(heat_difference / B(k))
        zh = R(heat_over_k / B(sqrt_CdN))
        ratio = R(sqrt_Cd / B(sqrt_CdN))
        stable = R(0.5 + _sign(0.5, zeta))
        ChN = _ch_n10_nemo(sqrt_CdN, stable)
        CeN = _ce_n10_nemo(sqrt_CdN)
        ch_numerator = R(ChN * ratio)
        ch_denominator = R(1.0 + R(ChN * zh))
        ce_numerator = R(CeN * ratio)
        ce_denominator = R(1.0 + R(CeN * zh))
        Ch = R(_fmax(ch_numerator / B(ch_denominator), _CX_MIN))
        Ce = R(_fmax(ce_numerator / B(ce_denominator), _CX_MIN))

    return Cd, Ch, Ce, Ub, theta_zu, q_zu


def ncar_transfer_coefficients(theta_air_K, q_air, theta_sst_K, q_sfc,
                               wind_speed, *, nb_iter: int = _NB_ITER_NCAR):
    """Legacy/public coefficient interface, bit-identical to the parent."""
    from legoesm.core.bulk_flux import large_yeager_neutral_cd, psi_m, psi_h

    k = constants.kappa_von_karman
    theta = jnp.asarray(theta_air_K, dtype=jnp.float64)
    q = jnp.asarray(q_air, dtype=jnp.float64)
    sst = jnp.asarray(theta_sst_K, dtype=jnp.float64)
    ssq = jnp.asarray(q_sfc, dtype=jnp.float64)
    Ub = jnp.maximum(jnp.asarray(wind_speed, dtype=jnp.float64), _U10_FLOOR_M_S)
    dTv = _virt_temp(theta, q) - _virt_temp(sst, ssq)
    stable = jnp.where(dTv > 0.0, 1.0, 0.0)
    CdN = large_yeager_neutral_cd(Ub, nemo_parity=True)
    sqrt_CdN = jnp.sqrt(CdN)
    Cd = CdN
    Ch = _ch_n10(sqrt_CdN, stable)
    Ce = _ce_n10(sqrt_CdN)
    sqrt_Cd = sqrt_CdN
    d_theta = theta - sst
    d_q = q - ssq
    for _ in range(int(nb_iter)):
        u_star = sqrt_Cd * Ub
        theta_star = Ch / sqrt_Cd * d_theta
        q_star = Ce / sqrt_Cd * d_q
        inv_L = _one_on_obukhov(
            theta, q, u_star, theta_star, q_star
        )
        zeta = jnp.clip(_ZU_M * inv_L, -_ZETA_ABS_MAX, _ZETA_ABS_MAX)
        psim = psi_m(zeta)
        UN10 = jnp.maximum(
            _UN10_FLOOR_M_S, Ub * (1.0 + sqrt_Cd / k * psim)
        )
        CdN = large_yeager_neutral_cd(UN10, nemo_parity=True)
        sqrt_CdN = jnp.sqrt(CdN)
        denom = 1.0 - sqrt_CdN / k * psim
        Cd = jnp.maximum(CdN / (denom * denom), _CX_MIN)
        sqrt_Cd = jnp.sqrt(Cd)
        zh = -psi_h(zeta) / k / sqrt_CdN
        ratio = sqrt_Cd / sqrt_CdN
        stable = jnp.where(zeta > 0.0, 1.0, 0.0)
        ChN = _ch_n10(sqrt_CdN, stable)
        CeN = _ce_n10(sqrt_CdN)
        Ch = jnp.maximum(ChN * ratio / (1.0 + ChN * zh), _CX_MIN)
        Ce = jnp.maximum(CeN * ratio / (1.0 + CeN * zh), _CX_MIN)
    return Cd, Ch, Ce, Ub


def large_yeager_cd(u10_speed):
    """L&Y 2009 open-ocean wind-speed-dependent neutral drag coefficient.

    Delegates to the canonical substrate kernel
    :func:`legoesm.core.bulk_flux.large_yeager_neutral_cd` (redundancy audit) so
    the ocean uses the FULL LY09 Eq. 6 — including the ``-3.14807e-10·U⁶``
    high-wind correction the OMIP-2 protocol (Griffies 2016) requires — and the
    same ``[0.5e-3, 3.0e-3]`` clip as the MOST flux solver.  Legacy
    ``ly09_2coeff`` helper; the NCAR path uses ``nemo_parity=True`` internally.
    Returns ``C_d`` (dimensionless).
    """
    from legoesm.core.bulk_flux import large_yeager_neutral_cd

    return large_yeager_neutral_cd(jnp.asarray(u10_speed, dtype=jnp.float64))


def large_yeager_ch(T_air_K, T_sfc_K):
    """Legacy two-regime constant transfer coefficient (``ly09_2coeff``).

    Unstable (ocean warmer) -> 1.46e-3, stable -> 1.18e-3.  A single-constant
    approximation of LY08; superseded by :func:`ncar_transfer_coefficients`
    (kept for the pinned legacy scheme only).
    """
    return jnp.where(T_sfc_K > T_air_K, _CH_UNSTABLE, _CH_STABLE)


class _NemoTurbulentState(NamedTuple):
    theta_air: object
    theta_sst: object
    ssq: object
    tsk_c: object
    cd: object
    ch: object
    ce: object
    rhoa: object
    sensible: object
    latent: object
    evap: object
    utau: object
    vtau: object
    taum: object
    wndm: object


def _nemo_ncar_turbulent(
    u_wind, v_wind, T_air_K, q_air, T_sfc_K, slp_Pa, mask, evap_factor,
    *, nb_iter: int, q_sfc_override=None, rho_override=None,
    latent_heat_override=None,
):
    """The executing round-5 ``blk_oce_1`` NCAR path, statement ordered."""
    u = jnp.asarray(u_wind, dtype=jnp.float64)
    v = jnp.asarray(v_wind, dtype=jnp.float64)
    wet = jnp.asarray(mask, dtype=jnp.float64)
    active = wet != 0.0
    # Masked forcing files use zero fills.  Give every differentiated source
    # expression a finite inactive operand; wet-cell primal values are unchanged.
    T_air = jnp.where(
        active, jnp.asarray(T_air_K, dtype=jnp.float64), 180.0
    )
    q = jnp.where(active, jnp.asarray(q_air, dtype=jnp.float64), 1.0e-6)
    sst = jnp.where(
        active, jnp.asarray(T_sfc_K, dtype=jnp.float64), constants.T_freeze
    )
    slp = jnp.where(
        active, jnp.asarray(slp_Pa, dtype=jnp.float64), constants.p_atm_std
    )

    tsk = R(sst)
    theta_sst = exner_potential_temperature(
        tsk, slp, nemo_source_exact=True
    )
    u2 = R(u * u)
    v2 = R(v * v)
    wind_squared = R(u2 + v2)
    safe_wind_squared = jnp.where(wind_squared > 0.0, wind_squared, 1.0)
    wndm_nonzero = R(jnp.sqrt(safe_wind_squared))
    wndm = jnp.where(wind_squared > 0.0, wndm_nonzero, 0.0)
    ssq = (seawater_q_sat(tsk, slp, nemo_source_exact=True)
           if q_sfc_override is None
           else jnp.asarray(q_sfc_override, dtype=jnp.float64))
    theta_air, _ = potential_air_temperature_10m(
        T_air, q, slp, nemo_source_exact=True
    )
    cd, ch, ce, Ub, theta_zu, q_zu = _ncar_transfer_coefficients_exact(
        theta_air, q, theta_sst, ssq, wndm, nb_iter=nb_iter
    )
    pressure, tabs = _pressure_from_potential(
        q_zu, slp, _ZU_M, theta_zu
    )
    rhoa = (rho_air_moist(
        tabs, q_zu, pressure, nemo_source_exact=True
    ) if rho_override is None
            else jnp.asarray(rho_override, dtype=jnp.float64))
    Urho = R(Ub * _fmax(rhoa, _RHO_FLUX_FLOOR))
    taum = R(Urho * cd * wndm)
    zevap = R(Urho * ce * (q_zu - ssq))
    sensible = R(
        R(R(Urho * ch) * R(theta_zu - theta_sst))
        * moist_air_cp(q_zu, nemo_source_exact=True)
    )
    latent_heat = (latent_heat_vaporization_sst(
        theta_sst, nemo_source_exact=True
    )
                   if latent_heat_override is None
                   else jnp.asarray(latent_heat_override, dtype=jnp.float64))
    latent = R(latent_heat * zevap)
    evap = R(-evap_factor * zevap)

    sensible = R(sensible * wet)
    latent = R(latent * wet)
    taum = R(taum * wet)
    evap = R(evap * wet)
    safe_wndm = jnp.where(wndm > 0.0, wndm, 1.0)
    ratio_nonzero = R(taum / safe_wndm)
    ratio = jnp.where(wndm > 0.0, ratio_nonzero, 0.0)
    utau = R(ratio * u)
    vtau = R(ratio * v)
    tsk_c = R((tsk - constants.T_freeze) * wet)
    return _NemoTurbulentState(
        theta_air, theta_sst, ssq, tsk_c, cd, ch, ce, rhoa,
        sensible, latent, evap, utau, vtau, taum, wndm,
    )


class NemoBlkOceResult(NamedTuple):
    theta_air: object
    q_air: object
    precip: object
    sst: object
    ssu: object
    ssv: object
    tsk: object
    ssq: object
    Cd: object
    Ch: object
    Ce: object
    rhoa: object
    sensible: object
    latent: object
    evap: object
    qsr: object
    qlwn: object
    qns: object
    emp: object
    utau: object
    vtau: object
    taum: object
    wndm: object
    ts_surface: object = None
    fr_i: object = None


class NemoSbcIceIfInputs(NamedTuple):
    """``nn_ice = 1`` inputs.  ``sn_ice`` record slots and ``smask0`` are
    ``A2D(0)``; the ocean fields are ``A2D(nn_hls)``; record times are
    integer seconds (fldread.F90:244)."""
    fdta_before: object
    fdta_after: object
    isecsbc: object
    nrec_sec_before: object
    nrec_sec_after: object
    sss_m: object
    sst_m: object
    ts_surface: object
    ff_f: object
    tmask_surface: object
    smask0: object


class NemoS0Inputs(NamedTuple):
    """State and instantaneous operands for the ``sbc_ssm`` production arm."""
    prior: NemoSurfaceMeans
    kt: object
    nit000: object
    forcing_stride: object
    restart_means_present: object
    use_conservative_temperature: object
    remove_inverse_barometer: object
    u_before: object
    v_before: object
    temperature_now: object
    salinity_now: object
    ssh_now: object
    ssh_inverse_barometer: object
    e3t_reference: object
    r3t_now: object
    surface_mask: object
    shortwave_fraction_top: object


class NemoS2Inputs(NamedTuple):
    """Post-bulk controls/state for the standalone ORCA2 ``sbcmod`` tail."""
    stress_lbc: NemoTStressLbcPlan
    post_ice_sfx: object
    runoff_now: object
    prior_runoff_mass: object
    prior_runoff_temperature_content: object
    prior_runoff_salinity_content: object
    prior_restoring_volume_flux: object
    prior_restoring_heat_flux: object
    prior_restoring_coefficient: object
    sss_target: object
    runoff_mouth_mask: object
    surface_mask: object
    tmask_east: object
    tmask_north: object
    umask: object
    vmask: object
    freshwater_budget_correction: object
    runoff_factor: object
    reference_density: object
    restoring_mm_per_day: object
    restoring_bound_mm_per_day: object
    seconds_per_day: object
    salinity_floor: object
    heat_capacity: object


class NemoStandaloneSurfaceResult(NamedTuple):
    means: NemoSurfaceMeans
    bulk: NemoBlkOceResult
    s2: object
    surface_forcing: object
    freshwater_forcing: object
    bulk_before_ice: NemoBlkOceResult
    ice_if: object


class NemoStandaloneAtmosphereFields(NamedTuple):
    """One mapped CORE record for the standalone card's S0-F boundary."""

    u_wind: object
    v_wind: object
    T_air_K: object
    q_air: object
    slp_Pa: object
    qsr_down: object
    qlw_down: object
    precip: object
    snow: object


class NemoStandaloneFieldPair(NamedTuple):
    """One ``fld_read`` two-record input and its own integer-second clock."""

    before: object
    after: object
    isecsbc: object
    nrec_sec_before: object
    nrec_sec_after: object


class NemoStandaloneS0FInputs(NamedTuple):
    """Two physical records and NEMO's integer-second interpolation clock."""

    before: NemoStandaloneAtmosphereFields
    after: NemoStandaloneAtmosphereFields
    isecsbc: object
    nrec_sec_before: object
    nrec_sec_after: object
    runoff: NemoStandaloneFieldPair
    sss_target: NemoStandaloneFieldPair


class NemoStandaloneK0Config(NamedTuple):
    """Explicit Decision-175 selectors for the standalone ORCA2 card.

    These fields deliberately have no defaults.  The card must carry the K0
    choices in its resolved configuration rather than inheriting library
    defaults.
    """

    nn_ice: int
    ln_drgice_imp: bool
    ln_mxl0: bool
    nn_mxlice: int
    ln_icebergs: bool


class NemoStandaloneK0OceanCard(NamedTuple):
    """The certified rung-10 ocean plus explicit standalone K0 selectors."""

    ocean: object
    k0: NemoStandaloneK0Config
    albedo: object
    precip_factor: object
    evap_factor: object
    nb_iter: int


class NemoStandaloneK0StepResult(NamedTuple):
    """One composed surface calculation followed by one production ocean step."""

    surface: NemoStandaloneSurfaceResult
    ocean: object
    next_s0: NemoS0Inputs
    next_s2: NemoS2Inputs


def validate_nemo_standalone_k0(config: NemoStandaloneK0Config) -> None:
    """Refuse every surface-card configuration other than Decision-175 K0."""

    expected = NemoStandaloneK0Config(1, False, True, 0, False)
    if config != expected:
        raise ValueError(
            "standalone ORCA2 requires explicit Decision-175 K0 selectors "
            f"{expected!r}; got {config!r}"
        )


def _validate_k0_ocean_card(card: NemoStandaloneK0OceanCard) -> None:
    """Bind Decision 175 to the real rung-10 selectors that consume it."""

    validate_nemo_standalone_k0(card.k0)
    ocean = card.ocean
    config = ocean.recipe.model_config
    tke = config.physics.vertical_mixing.tke
    resolved = {
        "ln_drgice_imp": bool(config.implicit_ice_ocean_top_drag),
        "ln_mxl0": bool(
            tke.tke_mxl_choice == 3
            and tke.nemo_mxl0_surface_tmask
            and tke.nemo_mxl0_rmxl_min_overwrite
        ),
        "nn_mxlice": int(tke.nemo_mxlice),
        "ln_icebergs": bool(ocean.icebergs_enabled),
    }
    expected = {
        "ln_drgice_imp": card.k0.ln_drgice_imp,
        "ln_mxl0": card.k0.ln_mxl0,
        "nn_mxlice": card.k0.nn_mxlice,
        "ln_icebergs": card.k0.ln_icebergs,
    }
    if resolved != expected:
        raise ValueError(
            "standalone K0 ocean selectors do not match the rung-10 card: "
            f"resolved={resolved!r}, expected={expected!r}"
        )


def build_nemo_orca2_k0_ocean_card(deck_root):
    """Wrap, without copying, the certified rung-10 ORCA2 ocean card."""

    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card

    result = NemoStandaloneK0OceanCard(
        ocean=build_orca2_zps_card(deck_root),
        k0=NemoStandaloneK0Config(1, False, True, 0, False),
        albedo=0.066,
        precip_factor=1.0,
        evap_factor=1.0,
        nb_iter=5,
    )
    _validate_k0_ocean_card(result)
    return result


def nemo_orca2_k0_s0f(inputs: NemoStandaloneS0FInputs):
    """Execute the card's own CORE, runoff and restoring S0-F interpolation."""

    values = (
        nemo_fld_time_interp(before, after, inputs.isecsbc,
                             inputs.nrec_sec_before, inputs.nrec_sec_after)
        for before, after in zip(inputs.before, inputs.after, strict=True)
    )
    atmosphere = NemoStandaloneAtmosphereFields(*values)

    def interpolate(pair):
        return nemo_fld_time_interp(
            pair.before, pair.after, pair.isecsbc,
            pair.nrec_sec_before, pair.nrec_sec_after)

    return atmosphere, interpolate(inputs.runoff), interpolate(inputs.sss_target)


def nemo_orca2_k0_surface_card(
    u_wind, v_wind, T_air_K, q_air, slp_Pa,
    qsr_down, qlw_down, precip, snow, mask, albedo, precip_factor,
    evap_factor, nb_iter, *, k0_config, ice_if, s0, s2,
    prior_surface=None, chl=None, geothermal_heat_flux_mwm2=None,
):
    """Run the production S0 -> S1 -> S3 -> S2 path for standalone K0.

    The routine accepts physical fields, including CORE fields and the
    Decision-171 ice-cover interpolation operands, but no NEMO exchange
    record.  It is intentionally a thin binding around
    :func:`nemo_blk_oce_ncar`; the source-ordered arithmetic remains in that
    shared production path.
    """

    validate_nemo_standalone_k0(k0_config)
    return nemo_blk_oce_ncar(
        u_wind, v_wind, T_air_K, q_air, slp_Pa,
        None, None, None,
        qsr_down, qlw_down, precip, snow, mask, albedo, precip_factor,
        evap_factor, nb_iter,
        nn_ice=k0_config.nn_ice, ice_if=ice_if, s0=s0, s2=s2,
        prior_surface=prior_surface, chl=chl,
        geothermal_heat_flux_mwm2=geothermal_heat_flux_mwm2,
    )


def nemo_orca2_k0_ocean_step(
    card, state, *, s0f, ice_if, s0, s2, chl,
    geothermal_heat_flux_mwm2, prior_surface=None, model=None, sponge=None,
    nemo_kt=None,
):
    """Run S0-F -> S0-M -> S1 -> S3 -> S2, then the rung-10 ocean step.

    ``card.ocean`` is the existing certified rung-10 recipe.  This function
    contains no ocean arithmetic: it calls :class:`LatLonCGridOceanModel`'s
    production ``step`` with forcing made by the shared literal surface path.
    The atmospheric input is the card's own pair of physical file records;
    no NEMO exchange record is accepted by this API.
    """

    _validate_k0_ocean_card(card)
    # sbc(kstp,Nbb,Nbb) reads the live rung-10 state.  The caller owns only
    # the explicit controls/carry; it cannot shadow S0-M or S3 with stale
    # state arrays.
    if (isinstance(nemo_kt, bool) or not isinstance(nemo_kt, (int, np.integer))
            or nemo_kt < 1):
        raise ValueError("standalone ORCA2 requires nemo_kt=<positive step>")
    if int(s0.forcing_stride) != 2:
        raise ValueError("standalone ORCA2 K0 requires nn_fsbc=2")
    temperature = jnp.asarray(state.T.data, dtype=jnp.float64)
    salinity = jnp.asarray(state.S.data, dtype=jnp.float64)
    u_level = state.u if state.u_before is None else state.u_before
    v_level = state.v if state.v_before is None else state.v_before
    u_before = jnp.asarray(u_level.data, dtype=jnp.float64)[:, 1:, 0]
    v_before = jnp.asarray(v_level.data, dtype=jnp.float64)[1:, :, 0]
    from legoesm.ocean.eos import nemo_r3t_stretch

    stretch = nemo_r3t_stretch(
        card.ocean.recipe.z_coord, state.eta.data, state.H_bathy.data,
        evaluation="nemo_reciprocal",
    )
    e3t_reference = jnp.asarray(
        card.ocean.recipe.z_coord.nemo_e3t_0[..., 0], dtype=jnp.float64)
    from legoesm.ocean.physics.shortwave_penetration import (
        top_layer_absorbed_fraction,
    )
    shortwave_fraction_top = top_layer_absorbed_fraction(
        e3t_reference * stretch, chl_surface=chl,
    )
    s0 = s0._replace(
        kt=int(nemo_kt),
        u_before=u_before,
        v_before=v_before,
        temperature_now=temperature[..., 0],
        salinity_now=salinity[..., 0],
        ssh_now=jnp.asarray(state.eta.data, dtype=jnp.float64),
        e3t_reference=e3t_reference,
        r3t_now=stretch - jnp.asarray(1.0, dtype=jnp.float64),
        surface_mask=jnp.asarray(state.land_mask.data, dtype=jnp.float64),
        shortwave_fraction_top=shortwave_fraction_top,
    )
    ice_if = ice_if._replace(ts_surface=temperature[..., 0])
    fields, runoff_now, sss_target = nemo_orca2_k0_s0f(s0f)
    s2 = s2._replace(runoff_now=runoff_now, sss_target=sss_target)
    surface = nemo_orca2_k0_surface_card(
        *fields[:5], *fields[5:],
        s2.surface_mask, card.albedo, card.precip_factor,
        card.evap_factor, card.nb_iter,
        k0_config=card.k0, ice_if=ice_if, s0=s0, s2=s2,
        prior_surface=prior_surface, chl=chl,
        geothermal_heat_flux_mwm2=geothermal_heat_flux_mwm2,
    )
    # sbc_ice_if writes ts(:,:,1,jp_tem,Kmm) before the ocean step on forcing
    # updates.  K0 calls sbc with Kbb=Kmm, so this is the live T slot.
    if (int(nemo_kt) - 1) % int(s0.forcing_stride) == 0:
        clamped = _crop_to_surface_interior(
            surface.ice_if.ts_surface, temperature.shape[:2])
        temperature = temperature.at[..., 0].set(clamped)
        state = state._replace(T=state.T.replace(data=temperature))
    if model is None:
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
            LatLonCGridOceanModel,
        )

        recipe = card.ocean.recipe
        model = LatLonCGridOceanModel(
            recipe.grid, recipe.z_coord, recipe.model_config,
            iwm_forcing=recipe.iwm_forcing,
        )
    kwargs = {
        "freshwater": surface.freshwater_forcing,
        "surface_forcing": surface.surface_forcing,
    }
    if sponge is not None:
        kwargs["sponge"] = sponge
    kwargs["nemo_kt"] = int(nemo_kt)
    stepped = model.step(state, card.ocean.dt_s, **kwargs)
    next_s0 = s0._replace(prior=surface.means)
    next_s2 = s2._replace(
        prior_runoff_mass=surface.s2.runoff.mass_flux,
        prior_runoff_temperature_content=surface.s2.runoff.temperature_content,
        prior_runoff_salinity_content=surface.s2.runoff.salinity_content,
        prior_restoring_volume_flux=surface.s2.erp,
        prior_restoring_heat_flux=surface.s2.qrp,
        prior_restoring_coefficient=surface.s2.restoring_coefficient,
    )
    return NemoStandaloneK0StepResult(
        surface=surface, ocean=stepped, next_s0=next_s0, next_s2=next_s2)


def _crop_to_surface_interior(value, interior_shape):
    """Crop symmetric NEMO halos from a full T array to ``A2D(0)``."""
    value = jnp.asarray(value, dtype=jnp.float64)
    if value.shape == interior_shape:
        return value
    if value.ndim != 2 or len(interior_shape) != 2:
        raise ValueError("standalone surface arrays must be two-dimensional")
    delta = tuple(full - inner for full, inner in zip(value.shape, interior_shape))
    if any(amount <= 0 or amount % 2 for amount in delta):
        raise ValueError("S0 means must be symmetric NEMO halos around A2D(0)")
    halo = tuple(amount // 2 for amount in delta)
    return value[
        halo[0]:value.shape[0] - halo[0],
        halo[1]:value.shape[1] - halo[1],
    ]


class BlkOce2Result(NamedTuple):
    qlwn: object
    qns: object
    emp: object


def blk_oce_2(
    theta_air, tsk_c, qlw_down, precip, snow, evap, sensible, latent,
    *, mask=1.0, precip_factor=1.0, nemo_source_exact: bool = False,
    array_namespace=jnp,
):
    """Assemble NEMO ``blk_oce_2`` once for literal and legacy callers.

    ``nemo_source_exact=True`` retains the record build's statement boundaries.
    The default branch retains the established OMIP host/scan arithmetic; the
    caller supplies NumPy or JAX as ``array_namespace`` to preserve its backend.
    """
    if nemo_source_exact:
        wet = jnp.asarray(mask, dtype=jnp.float64)
        rain = jnp.asarray(precip, dtype=jnp.float64)
        solid = jnp.asarray(snow, dtype=jnp.float64)
        tsk_c_arr = jnp.asarray(tsk_c, dtype=jnp.float64)
        cp_rain = R(
            R(jnp.asarray(theta_air, dtype=jnp.float64) - constants.T_freeze)
            * R(constants.c_p_seawater * wet)
        )
        cp_snow = R(
            R(_fmin(theta_air, constants.T_freeze) - constants.T_freeze)
            * R(constants.c_p_ice_nemo * wet)
        )
        cp_sst = R(tsk_c_arr * R(constants.c_p_seawater * wet))
        tsk_K = R(tsk_c_arr + constants.T_freeze)
        t2 = R(tsk_K * tsk_K)
        outgoing_lw = R(R(constants.sigma_sb_nemo * t2) * t2)
        net_lw = R(jnp.asarray(qlw_down, dtype=jnp.float64) - outgoing_lw)
        qlwn = R(constants.emissivity_seawater_lw * net_lw)
        emp = R(
            R(jnp.asarray(evap, dtype=jnp.float64) - R(rain * precip_factor))
            * wet
        )
        qns = R(
            R(R(R(qlwn + sensible) + latent)
              - R(R(solid * precip_factor) * constants.L_fus_nemo))
            - R(evap * cp_sst)
        )
        qns = R(
            R(qns + R(R(R(rain - solid) * precip_factor) * cp_rain))
            + R(R(solid * precip_factor) * cp_snow)
        )
        return BlkOce2Result(qlwn, R(qns * wet), emp)

    xp = array_namespace
    theta_air_c = xp.asarray(theta_air) - float(constants.T_freeze)
    tsk_c_arr = xp.asarray(tsk_c)
    precip_arr = xp.asarray(precip)
    snow_arr = xp.asarray(snow)
    wet = xp.asarray(mask)
    tsk_K = tsk_c_arr + float(constants.T_freeze)
    qlwn = float(constants.emissivity_seawater_lw) * (
        xp.asarray(qlw_down) - float(constants.sigma_sb) * tsk_K ** 4
    )
    liquid = xp.maximum(precip_arr - snow_arr, 0.0)
    # This is an externally observed compatibility boundary.  Preserve the
    # parent host/scan statement grouping exactly: the precipitation/evap
    # terms are reduced first, followed by sh + lh + lw + precip.  Flattening
    # the sum changes q_net by a few ulps even though it is algebraically
    # equivalent.
    q_precip_evap = (
        -snow_arr * float(precip_factor) * float(constants.L_fus_nemo)
        - xp.asarray(evap) * float(constants.c_p_seawater) * tsk_c_arr
        + liquid * float(precip_factor) * float(constants.c_p_seawater) * theta_air_c
        + snow_arr * float(precip_factor) * float(constants.c_p_ice_nemo)
        * xp.minimum(theta_air_c, 0.0)
    )
    qns = (
        xp.asarray(sensible) + xp.asarray(latent) + qlwn + q_precip_evap
    )
    emp = (xp.asarray(evap) - precip_arr * float(precip_factor)) * wet
    return BlkOce2Result(qlwn, qns * wet, emp)


def nemo_blk_oce_ncar(
    u_wind, v_wind, T_air_K, q_air, slp_Pa, sst_c, ssu, ssv,
    qsr_down, qlw_down, precip, snow, mask, albedo, precip_factor,
    evap_factor, nb_iter, *, nn_ice, ice_if, s0, s2,
    prior_surface=None, chl=None, geothermal_heat_flux_mwm2=None,
):
    """Literal S0 -> NCAR S1 -> optional S3 -> optional S2 surface path.

    Every physical input and resolved deck scalar is required: this boundary
    has no standard-atmosphere, mask, precipitation, or coefficient defaults.
    The returned stresses use NEMO's ocean (downward) convention.

    ``nn_ice`` follows sbcmod.F90's ``SELECT CASE( nn_ice )`` after the bulk
    call: ``0`` returns the bulk fields (``ice_if`` must be None); ``1`` runs
    ``fld_read``'s time blend of ``sn_ice`` and ``sbc_ice_if`` on ``qsr``/``qns``
    (``ice_if`` a :class:`NemoSbcIceIfInputs`; the bulk arrays are ``A2D(0)``)
    and fills ``ts_surface``/``fr_i``.  Any other value raises. ``s0=None,
    s2=None`` selects the established S1/S3 path.  Supplying both selects the
    standalone S0 -> S1 -> S3 -> S2 arm; in that arm the explicit mean
    arguments ``sst_c``, ``ssu`` and ``ssv`` must be ``None`` because S0
    produces them.  Half-selected or shadowed inputs refuse.
    """
    if nn_ice not in (0, 1):
        raise ValueError(f"nemo_blk_oce_ncar: unsupported nn_ice={nn_ice!r}")
    if (nn_ice == 1) != (ice_if is not None):
        raise ValueError("nemo_blk_oce_ncar: ice_if is required iff nn_ice == 1")
    if (s0 is None) != (s2 is None):
        raise ValueError("nemo_blk_oce_ncar requires S0 and S2 together")
    if s2 is not None and nn_ice != 1:
        raise ValueError("standalone S0/S2 path requires composed nn_ice=1 S3")
    if s0 is None:
        if sst_c is None or ssu is None or ssv is None:
            raise ValueError("S1-only path requires explicit sst_c, ssu and ssv")
        means = None
    else:
        if sst_c is not None or ssu is not None or ssv is not None:
            raise ValueError("standalone S0 arm produces sst_c, ssu and ssv")
        means = nemo_sbc_ssm_step(
            s0.prior,
            kt=s0.kt,
            nit000=s0.nit000,
            forcing_stride=s0.forcing_stride,
            restart_means_present=s0.restart_means_present,
            use_conservative_temperature=s0.use_conservative_temperature,
            remove_inverse_barometer=s0.remove_inverse_barometer,
            u_before=s0.u_before,
            v_before=s0.v_before,
            temperature_now=s0.temperature_now,
            salinity_now=s0.salinity_now,
            ssh_now=s0.ssh_now,
            ssh_inverse_barometer=s0.ssh_inverse_barometer,
            e3t_reference=s0.e3t_reference,
            r3t_now=s0.r3t_now,
            surface_mask=s0.surface_mask,
            shortwave_fraction_top=s0.shortwave_fraction_top,
        )
        interior_shape = jnp.shape(mask)
        sst_c = _crop_to_surface_interior(means.sst, interior_shape)
        ssu = _crop_to_surface_interior(means.ssu, interior_shape)
        ssv = _crop_to_surface_interior(means.ssv, interior_shape)
        update = (int(s0.kt) - 1) % int(s0.forcing_stride) == 0
        if not update:
            if prior_surface is None:
                raise ValueError(
                    "nn_fsbc=2 preservation step requires prior_surface carry")
            return prior_surface._replace(means=means)
    wet = jnp.asarray(mask, dtype=jnp.float64)
    sst_c_arr = jnp.asarray(sst_c, dtype=jnp.float64)
    state = _nemo_ncar_turbulent(
        u_wind, v_wind, T_air_K, q_air,
        R(sst_c_arr + constants.T_freeze), slp_Pa, wet, evap_factor,
        nb_iter=int(nb_iter),
    )
    rain = jnp.asarray(precip, dtype=jnp.float64)
    solid = jnp.asarray(snow, dtype=jnp.float64)
    surface = blk_oce_2(
        state.theta_air, state.tsk_c, qlw_down, rain, solid,
        state.evap, state.sensible, state.latent,
        mask=wet, precip_factor=precip_factor, nemo_source_exact=True,
    )
    qsr = R((1.0 - albedo) * jnp.asarray(qsr_down, dtype=jnp.float64) * wet)
    qns = surface.qns
    bulk_before_ice = NemoBlkOceResult(
        state.theta_air, jnp.asarray(q_air, dtype=jnp.float64), rain,
        sst_c_arr, jnp.asarray(ssu, dtype=jnp.float64),
        jnp.asarray(ssv, dtype=jnp.float64), state.tsk_c, state.ssq,
        state.cd, state.ch, state.ce, state.rhoa,
        state.sensible, state.latent, state.evap, qsr,
        surface.qlwn, qns, surface.emp,
        state.utau, state.vtau, state.taum, state.wndm, None, None,
    )
    ts_surface = fr_i = None
    iif = None
    if nn_ice == 1:
        if means is not None:
            ice_if = ice_if._replace(sss_m=means.sss, sst_m=means.sst)
        halo = {(a - b) for a, b in zip(jnp.shape(ice_if.ts_surface),
                                         jnp.shape(qns))}
        if len(halo) != 1 or min(halo) < 0 or min(halo) % 2:
            raise ValueError("ice_if fields must be A2D(nn_hls) around A2D(0)")
        cover = nemo_fld_time_interp(
            ice_if.fdta_before, ice_if.fdta_after, ice_if.isecsbc,
            ice_if.nrec_sec_before, ice_if.nrec_sec_after,
        )
        iif = nemo_sbc_ice_if(
            cover, ice_if.sss_m, ice_if.sst_m, ice_if.ts_surface, ice_if.ff_f,
            ice_if.tmask_surface, ice_if.smask0, qsr, qns,
            nn_hls=min(halo) // 2,
        )
        qsr, qns, ts_surface, fr_i = iif.qsr, iif.qns, iif.ts_surface, iif.fr_i
    bulk = NemoBlkOceResult(
        state.theta_air, jnp.asarray(q_air, dtype=jnp.float64), rain,
        sst_c_arr, jnp.asarray(ssu, dtype=jnp.float64),
        jnp.asarray(ssv, dtype=jnp.float64), state.tsk_c, state.ssq,
        state.cd, state.ch, state.ce, state.rhoa,
        state.sensible, state.latent, state.evap, qsr,
        surface.qlwn, qns, surface.emp,
        state.utau, state.vtau, state.taum, state.wndm, ts_surface, fr_i,
    )
    if s2 is None:
        return bulk
    sss_mean = _crop_to_surface_interior(means.sss, jnp.shape(mask))
    utau, vtau, utau_east, vtau_north = nemo_t_stress_lbc_neighbors(
        bulk.utau, bulk.vtau, s2.stress_lbc
    )
    tail = nemo_orca2_s2(
        kt=s0.kt,
        forcing_stride=s0.forcing_stride,
        post_ice_utau=utau,
        post_ice_vtau=vtau,
        post_ice_utau_east=utau_east,
        post_ice_vtau_north=vtau_north,
        post_ice_taum=bulk.taum,
        post_ice_qsr=bulk.qsr,
        post_ice_qns=bulk.qns,
        post_ice_emp=bulk.emp,
        post_ice_sfx=s2.post_ice_sfx,
        runoff_now=s2.runoff_now,
        prior_runoff_mass=s2.prior_runoff_mass,
        prior_runoff_temperature_content=s2.prior_runoff_temperature_content,
        prior_runoff_salinity_content=s2.prior_runoff_salinity_content,
        prior_restoring_volume_flux=s2.prior_restoring_volume_flux,
        prior_restoring_heat_flux=s2.prior_restoring_heat_flux,
        prior_restoring_coefficient=s2.prior_restoring_coefficient,
        sst_mean=sst_c,
        sss_mean=sss_mean,
        sss_target=s2.sss_target,
        runoff_mouth_mask=s2.runoff_mouth_mask,
        surface_mask=s2.surface_mask,
        tmask_east=s2.tmask_east,
        tmask_north=s2.tmask_north,
        umask=s2.umask,
        vmask=s2.vmask,
        freshwater_budget_correction=s2.freshwater_budget_correction,
        runoff_factor=s2.runoff_factor,
        reference_density=s2.reference_density,
        restoring_mm_per_day=s2.restoring_mm_per_day,
        restoring_bound_mm_per_day=s2.restoring_bound_mm_per_day,
        seconds_per_day=s2.seconds_per_day,
        salinity_floor=s2.salinity_floor,
        heat_capacity=s2.heat_capacity,
    )
    # The dynamics ice mask consumes the same ``sf_ice(1)%fnow`` field used
    # by K0's ice-if heat-flux update, never ``fr_i`` (which is a freezing
    # temperature / binary freeze flag at different points in that routine).
    ice_concentration = jnp.asarray(iif.ice_cover, dtype=jnp.float64)
    forcing, freshwater = nemo_s2_to_ocean_interfaces(
        tail, chl=chl, ice_concentration=ice_concentration,
        geothermal_heat_flux_mwm2=geothermal_heat_flux_mwm2)
    return NemoStandaloneSurfaceResult(
        means, bulk, tail, forcing, freshwater, bulk_before_ice, iif)


class NemoSbcIceIfResult(NamedTuple):
    """Post-``sbc_ice_if`` state.  ``ts_surface``/``fr_i`` are full
    ``A2D(nn_hls)`` arrays; ``qsr``/``qns`` are interior ``A2D(0)`` arrays."""
    ts_surface: jnp.ndarray
    fr_i: jnp.ndarray
    qsr: jnp.ndarray
    qns: jnp.ndarray
    freezing_temperature: jnp.ndarray
    ice_cover: jnp.ndarray


def nemo_sbc_ice_if(ice_cover, sss_m, sst_m, ts_surface, ff_f,
                    tmask_surface, smask0, qsr, qns, *, nn_hls):
    """Literal record-build ``sbc_ice_if`` update block (nn_ice = 1).

    Transcribes sbcice_if.f90:154-196, the body of ``IF( MOD(kt-1, nn_fsbc)
    == 0 )``; on every other step the routine changes nothing, so the caller
    applies that gate.  ``ice_cover`` is ``sf_ice(1)%fnow`` after
    ``fld_read`` (owned by S0-F).  ``ts_surface`` is ``ts(:,:,1,jp_tem,Nbb)``:
    the RK3 record build calls ``sbc(kstp, Nbb, Nbb)``, so Kbb = Kmm and the
    flux loop reads the clamped temperature.  The source leaves ``emp``,
    ``sfx`` and the stresses untouched and never reads ``zsice``.

    Ties follow the object code: MAXSD keeps ``zt_fzp`` (second operand);
    MINSD in ``MIN(0., x)`` keeps the literal ``0.`` (first operand).
    """
    h = int(nn_hls)
    f64 = jnp.float64
    sss_m = jnp.asarray(sss_m, dtype=f64)
    ny, nx = sss_m.shape
    inner = ((slice(None), slice(None)) if h == 0 else
             (slice(h, ny - h), slice(h, nx - h)))
    # sbcice_if.f90:158-159: eos_fzp on A2D(nn_hls), then masked.
    fzp = R(nemo_eos_fzp_eos80(sss_m) * jnp.asarray(tmask_surface, dtype=f64))
    # :164-167: clamp on the full A2D(nn_hls) halo.
    ts_new = _fmax(jnp.asarray(ts_surface, dtype=f64), fzp)
    zt = fzp[inner]
    zfr = jnp.asarray(ice_cover, dtype=f64)
    ts_b = ts_new[inner]
    mask0 = jnp.asarray(smask0, dtype=f64)
    # :173-175
    fr_in = jnp.where(jnp.asarray(sst_m, dtype=f64)[inner] <= zt, 1.0, 0.0)
    open_frac = R(1.0 - zfr)
    # :177
    qsr_new = R(open_frac * jnp.asarray(qsr, dtype=f64))
    # :182-185
    zqri = R(_IIF_ZTRP_W_M2_K * R(ts_b - R(zt - 1.0)))
    zqrj = R(_IIF_ZTRP_W_M2_K * _fmin(R(ts_b - zt), 0.0))
    zqrp = R(R(zfr * R(R(R(1.0 - fr_in) * zqri) + R(fr_in * zqrj))) * mask0)
    # :192-195
    zqi = R(_IIF_ZQI_OFFSET_W_M2 + _sign(1.0, jnp.asarray(ff_f, dtype=f64)[inner]))
    qns_new = R(R(R(R(open_frac * jnp.asarray(qns, dtype=f64))
                    + R(R(zfr * fr_in) * zqi)) * mask0) + zqrp)
    return NemoSbcIceIfResult(
        ts_new, fzp.at[inner].set(fr_in), qsr_new, qns_new, fzp, zfr)


def air_sea_fluxes(u10, v10, T_air_K, q_air, T_sfc_K, q_sfc=None,
                   rho_air=None, *, slp_Pa=None, algo: str = "ncar",
                   nb_iter: int = _NB_ITER_NCAR, L_latent=None,
                   u_oce=None, v_oce=None, vfac: float = 0.0):
    """Open-ocean bulk fluxes ``(tau_x, tau_y, shflx, lhflx, evap)``.

    Sign conventions: ``shflx``/``lhflx`` > 0 add heat to the ocean;
    ``tau_x``/``tau_y`` are the ATMOSPHERIC-convention stress (``-rho Cd U u``,
    opposing the wind — the dynamics core applies the ``-tau`` ocean reaction);
    ``evap`` is positive UP (water leaving the ocean) and satisfies
    ``evap = -lhflx / L_vap`` exactly.

    Parameters
    ----------
    q_sfc : array or None
        Sea-surface saturation specific humidity.  ``None`` (NCAR default)
        computes the NEMO-faithful ``0.98 q_sat_goff(SST, slp)`` internally;
        the legacy ``ly09_2coeff`` algo REQUIRES an explicit value (it
        predates the salt factor and its callers pin the old behaviour).
    rho_air : array/float or None
        ``None`` -> NCAR computes moist-air density from ``slp_Pa``/T/q;
        the legacy algo falls back to ``constants.rho_air``.  An explicit
        value overrides both (used by tests).
    slp_Pa : array or None
        Sea-level pressure [Pa] for the density + saturation-humidity
        computations.  ``None`` -> standard atmosphere (forcing sets without
        an slp channel).
    algo : str
        ``"ncar"`` (NEMO-faithful, default) or ``"ly09_2coeff"`` (legacy).
    L_latent : float or None
        Optional diagnostic override.  Production NCAR callers leave it None
        and use the SST-dependent literal NEMO ``L_vap``.
    u_oce, v_oce : array or None
        Ocean surface-current components in the SAME frame as ``u10``/``v10``
        (the OMIP applicator supplies both as geographic east/north).  NEMO
        ``ln_crt_dwn`` current feedback: with ``vfac > 0`` these are subtracted
        (component-wise, scaled by ``vfac``) from the wind BEFORE the wind speed
        + stress bulk, so the stress uses the RELATIVE vector.  ``None`` (the
        default) keeps the absolute wind.
    vfac : float
        NEMO ``rn_vfac`` current-feedback fraction in [0, 1] (static Python
        float, resolved at trace time).  ``0.0`` (default) is BYTE-IDENTICAL to
        the absolute-wind behaviour: the current is never touched (no float op),
        so an existing caller is unchanged.  ``1.0`` = full feedback.
    """
    if algo not in _VALID_OMIP_BULK_ALGOS:
        raise ValueError(
            f"Unknown OMIP bulk algo {algo!r}; expected one of "
            f"{_VALID_OMIP_BULK_ALGOS}."
        )
    u_arr = jnp.asarray(u10, dtype=jnp.float64)
    v_arr = jnp.asarray(v10, dtype=jnp.float64)
    T_air = jnp.asarray(T_air_K, dtype=jnp.float64)
    q_a = jnp.asarray(q_air, dtype=jnp.float64)
    sst = jnp.asarray(T_sfc_K, dtype=jnp.float64)
    # --- Relative wind / current feedback (NEMO ln_crt_dwn, rn_vfac) --------
    # Frame convention: u10/v10 are the wind components in THEIR frame (positive
    # along that frame's axes); u_oce/v_oce are the ocean surface-current
    # components in the SAME frame (the OMIP applicator supplies both as
    # geographic east/north).  dU = u10 - vfac*u_oce, dV = v10 - vfac*v_oce, and
    # the wind speed + BOTH stress branches below use the RELATIVE vector
    # (dU, dV) -- NOT |dU|*wind_direction.  Newton's 3rd law is untouched: only
    # the wind VECTOR feeding the bulk changes; the ocean still feels -tau
    # downstream (the applicator's -tau reaction is unchanged).
    # STATIC Python guard (vfac is a config float, not traced): with vfac == 0.0
    # (default) OR no current supplied, u_rel/v_rel ARE u_arr/v_arr (same array,
    # zero float ops), so every path below is BYTE-IDENTICAL to the absolute-wind
    # behaviour -- even when u_oce carries NaN/inf on masked cells.  NOT a
    # jnp.where (which would trace both branches and touch the default path).
    if vfac != 0.0 and u_oce is not None and v_oce is not None:
        u_rel = u_arr - vfac * jnp.asarray(u_oce, dtype=jnp.float64)
        v_rel = v_arr - vfac * jnp.asarray(v_oce, dtype=jnp.float64)
    else:
        u_rel = u_arr
        v_rel = v_arr
    wind_speed = jnp.sqrt(u_rel ** 2 + v_rel ** 2 + 1e-12)

    if algo == "ly09_2coeff":
        if q_sfc is None:
            raise ValueError(
                "air_sea_fluxes(algo='ly09_2coeff') requires an explicit "
                "q_sfc (the legacy scheme predates the internal 0.98 "
                "Goff saturation)."
            )
        rho = constants.rho_air if rho_air is None else rho_air
        Cd = large_yeager_cd(wind_speed)
        Ch = large_yeager_ch(T_air, sst)
        L = constants.L_v if L_latent is None else L_latent
        tau_x = -rho * Cd * wind_speed * u_rel
        tau_y = -rho * Cd * wind_speed * v_rel
        shflx = rho * constants.c_pd * Ch * wind_speed * (T_air - sst)
        lhflx = rho * L * Ch * wind_speed * (q_a - q_sfc)
        evap = -lhflx / L
        return tau_x, tau_y, shflx, lhflx, evap

    slp_for_sst = constants.p_atm_std if slp_Pa is None else jnp.asarray(
        slp_Pa, dtype=jnp.float64)
    ssq = seawater_q_sat(sst, slp_Pa) if q_sfc is None else jnp.asarray(
        q_sfc, dtype=jnp.float64)
    theta_air, p10 = potential_air_temperature_10m(T_air, q_a, slp_Pa)
    theta_sst = exner_potential_temperature(sst, slp_for_sst)
    Cd, Ch, Ce, Ub = ncar_transfer_coefficients(
        theta_air, q_a, theta_sst, ssq, wind_speed, nb_iter=nb_iter,
    )
    rho = rho_air_moist(T_air, q_a, p10) if rho_air is None else jnp.asarray(
        rho_air, dtype=jnp.float64)
    Urho = Ub * jnp.maximum(rho, _RHO_FLUX_FLOOR)
    tau_x = -Urho * Cd * u_rel
    tau_y = -Urho * Cd * v_rel
    L_vap = latent_heat_vaporization_sst(theta_sst)
    z_evap = Urho * Ce * (q_a - ssq)
    shflx = Urho * Ch * (theta_air - theta_sst) * moist_air_cp(q_a)
    lhflx = L_vap * z_evap
    evap = -z_evap
    return tau_x, tau_y, shflx, lhflx, evap


__all__ = [
    "air_sea_fluxes",
    "blk_oce_2",
    "BlkOce2Result",
    "exner_potential_temperature",
    "large_yeager_cd",
    "large_yeager_ch",
    "latent_heat_vaporization_sst",
    "moist_air_cp",
    "nemo_blk_oce_ncar",
    "NemoBlkOceResult",
    "nemo_sbc_ice_if",
    "NemoSbcIceIfResult",
    "NemoSbcIceIfInputs",
    "NemoS0Inputs",
    "NemoS2Inputs",
    "NemoStandaloneAtmosphereFields",
    "NemoStandaloneFieldPair",
    "NemoStandaloneS0FInputs",
    "NemoStandaloneK0Config",
    "NemoStandaloneK0OceanCard",
    "NemoStandaloneK0StepResult",
    "NemoStandaloneSurfaceResult",
    "build_nemo_orca2_k0_ocean_card",
    "nemo_orca2_k0_ocean_step",
    "nemo_orca2_k0_s0f",
    "nemo_orca2_k0_surface_card",
    "NemoTStressLbcPlan",
    "ncar_transfer_coefficients",
    "potential_air_temperature_10m",
    "pressure_at_height",
    "rho_air_moist",
    "seawater_q_sat",
]
