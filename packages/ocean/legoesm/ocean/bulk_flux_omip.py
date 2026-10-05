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

``psi_m`` / ``psi_h`` are the canonical Paulson/Dyer forms shared with the
MOST solver (``legoesm.core.bulk_flux.psi_m/psi_h`` — identical to NEMO's
``psi_m_ncar``/``psi_h_ncar`` including the ``|zeta| <= 10`` clip).

``"ly09_2coeff"`` (legacy): the previous two-constant approximation
(``Ch = Ce in {1.46e-3 unstable, 1.18e-3 stable}``, constant air density,
caller-supplied ``q_sfc``).  Kept ONLY as a pinned regression scheme: it
biases the turbulent fluxes by 10-30 % vs the NCAR reference and was the
production scheme for all OMIP runs before 2026-06-10.

All operations are pure JAX (float64), fixed iteration count (statically
unrolled — JIT/vmap/grad safe, no data-dependent control flow).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
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


def exner_potential_temperature(T_K, p_Pa):
    """Potential temperature via the Exner function (NEMO ``theta_exner``).

    ``theta = T (rpref / p)^gamma_dry`` with ``rpref = 1e5 Pa`` and the
    dry-air Poisson constant ``gamma_dry = R_gas/(M_dry cp_dry)``.  Used for
    BOTH the 10-m air temperature (with the 10-m pressure) and the potential
    sea-surface temperature (with the sea-level pressure), exactly like
    NEMO blk_oce_1.
    """
    T = jnp.asarray(T_K, dtype=jnp.float64)
    p = jnp.asarray(p_Pa, dtype=jnp.float64)
    return T * (_P_EXNER_REF / p) ** _GAMMA_DRY


def pressure_at_height(q_air, slp_Pa, z_m, T_abs_K):
    """Air pressure at height ``z_m`` (NEMO ``pres_temp``, absolute-T branch).

    3 fixed-point iterations of the barometric law with the MOIST molar mass
    ``xm = (1 - q/q_sat) M_dry + (q/q_sat) M_water`` (q_sat from the Goff
    curve at the CURRENT pressure iterate):

        p <- slp * exp( -g xm z / (R_gas T) )

    Static iteration count — jit/grad-safe.
    """
    q = jnp.asarray(q_air, dtype=jnp.float64)
    slp = jnp.asarray(slp_Pa, dtype=jnp.float64)
    T = jnp.maximum(jnp.asarray(T_abs_K, dtype=jnp.float64), 180.0)
    p = slp
    for _ in range(_N_ITER_PRES):
        e_s = saturation_vapor_pressure_goff(T)
        q_sat = _REPS0 * e_s / (p - (1.0 - _REPS0) * e_s)
        # NEMO uses w = q/q_sat UNCLIPPED.  The moist molar mass goes
        # NEGATIVE (-> NaN via the barometric exponential) only for
        # w > M_dry/(M_dry - M_water) ~ 2.65, far beyond any physical
        # supersaturation — that regime is reached ONLY by unmasked garbage
        # cells (land points at the 180 K floor, w ~ 1e5).  Clip at 2.0:
        # bit-identical to NEMO for every physical input INCLUDING
        # supersaturated air (codex round-2 MED), NaN-proof on garbage.
        w = jnp.clip(q / q_sat, 0.0, 2.0)
        xm = (1.0 - w) * constants.M_dry_air + w * constants.M_water
        p = slp * jnp.exp(
            -constants.g_nemo * xm * z_m / (constants.R_gas_molar * T))
    return p


def potential_air_temperature_10m(T_air_K, q_air, slp_Pa=None):
    """NEMO blk_oce_1 air-temperature preprocessing at zt = 10 m.

    Returns ``(theta_air, p10)``: the potential air temperature
    ``theta_exner(T_abs, p10)`` and the 10-m pressure from
    :func:`pressure_at_height`.  ``slp_Pa=None`` -> standard atmosphere.
    """
    p_slp = constants.p_atm_std if slp_Pa is None else slp_Pa
    p10 = pressure_at_height(q_air, p_slp, _ZU_M, T_air_K)
    return exner_potential_temperature(T_air_K, p10), p10


def seawater_q_sat(T_sfc_K, slp_Pa=None):
    """Salt-reduced saturation specific humidity at the sea surface [kg/kg].

    NEMO: ``ssq = rdct_qsat_salt * q_sat(SST, slp)`` with the Goff (1957)
    vapour-pressure curve and ``q = eps e/(p - (1-eps) e)``.  The 0.98 factor
    is the vapour-pressure lowering over saline water (Large & Yeager 2004,
    NEMO sbc_phy ``rdct_qsat_salt``) — omitting it overestimates the sea-air
    humidity difference and hence evaporation by ~10 % in the tropics.
    ``slp_Pa=None`` falls back to the standard atmosphere.
    """
    p = constants.p_atm_std if slp_Pa is None else slp_Pa
    e_s = saturation_vapor_pressure_goff(jnp.asarray(T_sfc_K, dtype=jnp.float64))
    p = jnp.asarray(p, dtype=jnp.float64)
    q = _REPS0 * e_s / (p - (1.0 - _REPS0) * e_s)
    return _QSAT_SALT_FACTOR * q


def rho_air_moist(T_air_K, q_air, slp_Pa=None):
    """Moist-air density from the ideal-gas law [kg/m^3] (NEMO ``rho_air``).

    ``rho = slp / (R_d T (1 + 0.608 q))``, floored at 0.8 like NEMO.
    ``slp_Pa=None`` -> standard atmosphere (constant-pressure fallback for
    forcing sets without an slp channel).
    """
    p = constants.p_atm_std if slp_Pa is None else slp_Pa
    T = jnp.maximum(jnp.asarray(T_air_K, dtype=jnp.float64), 180.0)
    q = jnp.maximum(jnp.asarray(q_air, dtype=jnp.float64), 1.0e-6)
    rho = jnp.asarray(p, dtype=jnp.float64) / (
        constants.R_d * T * (1.0 + _RCTV0 * q)
    )
    return jnp.maximum(rho, _RHO_AIR_FLOOR)


def latent_heat_vaporization_sst(T_sfc_K):
    """SST-dependent latent heat of vaporization [J/kg] at NEMO-parity float64.

    Float64 pin around the shared Kirchhoff :func:`legoesm.thermo.latent_heat_vaporization`
    (one formula, two dtype contracts, #762).  NEMO's own ``L_vap`` uses an
    empirical slope of 2370 J/kg/K against this form's 2372; the ~2.5e-5
    relative difference at 30 degC is below every parity tolerance here.
    """
    from legoesm.thermo import latent_heat_vaporization as _l
    return _l(jnp.asarray(T_sfc_K, dtype=jnp.float64))


def moist_air_cp(q_air):
    """Moist-air specific heat [J/(kg K)] (NEMO ``cp_air``).

    NEMO-parity float64 pin around the shared
    :func:`legoesm.thermo.moist_air_cp` (#762).
    """
    from legoesm.thermo import moist_air_cp as _cp_moist
    return _cp_moist(jnp.asarray(q_air, dtype=jnp.float64))


def _virt_temp(T_K, q):
    """(Absolute or potential) virtual temperature (NEMO ``virt_temp``)."""
    return T_K * (1.0 + _RCTV0 * q)


def _one_on_obukhov(theta_K, q, u_star, theta_star, q_star):
    """Inverse Obukhov length 1/L [1/m] (NEMO ``One_on_L``), |1/L| <= 200.

    Virtual-temperature flux form (option (a) in NEMO):
    ``<w'thv'> ~ theta*(1 + 0.608 q) + 0.608 theta q*``.
    """
    zqa = 1.0 + _RCTV0 * q
    inv_L = (
        constants.g_nemo * constants.kappa_von_karman
        * (theta_star * zqa + _RCTV0 * theta_K * q_star)
        / jnp.maximum(u_star * u_star * theta_K * zqa, 1.0e-9)
    )
    return jnp.clip(inv_L, -_ONE_ON_L_ABS_MAX, _ONE_ON_L_ABS_MAX)


def _ch_n10(sqrt_cdn, stable_01):
    """Neutral 10-m sensible-heat coefficient (LY08 Eq. 9/12)."""
    return jnp.maximum(
        1.0e-3 * sqrt_cdn * (18.0 * stable_01 + 32.7 * (1.0 - stable_01)),
        _CX_MIN,
    )


def _ce_n10(sqrt_cdn):
    """Neutral 10-m latent-heat coefficient (LY08 Eq. 6b)."""
    return jnp.maximum(1.0e-3 * 34.6 * sqrt_cdn, _CX_MIN)


def ncar_transfer_coefficients(theta_air_K, q_air, theta_sst_K, q_sfc,
                               wind_speed, *, nb_iter: int = _NB_ITER_NCAR):
    """Stability-iterated NCAR transfer coefficients at zt = zu = 10 m.

    Exact port of NEMO ``turb_ncar`` for the CORE-II case (all forcing at
    10 m, so ``l_zt_equal_zu`` and every ``LOG(zu/10)`` term vanishes).
    Both temperatures are POTENTIAL (NEMO passes ``theta_air_zt`` and the
    Exner potential SST ``zsspt``); ``q_sfc`` is the salt-reduced saturation
    humidity at the ABSOLUTE SST.  ``nb_iter`` is a static Python int
    (fixed unrolled loop: differentiable, no traced control flow).

    Returns ``(Cd, Ch, Ce, Ub)`` with ``Ub = max(wind_speed, 0.5)``.
    """
    from legoesm.core.bulk_flux import large_yeager_neutral_cd, psi_m, psi_h

    k = constants.kappa_von_karman
    theta = jnp.asarray(theta_air_K, dtype=jnp.float64)
    q = jnp.asarray(q_air, dtype=jnp.float64)
    sst = jnp.asarray(theta_sst_K, dtype=jnp.float64)
    ssq = jnp.asarray(q_sfc, dtype=jnp.float64)

    Ub = jnp.maximum(jnp.asarray(wind_speed, dtype=jnp.float64), _U10_FLOOR_M_S)

    # First-guess stability from the air-sea virtual-temperature difference.
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

        inv_L = _one_on_obukhov(theta, q, u_star, theta_star, q_star)
        zeta = jnp.clip(_ZU_M * inv_L, -_ZETA_ABS_MAX, _ZETA_ABS_MAX)
        psim = psi_m(zeta)

        # Neutral wind at 10 m: UN10 = u*/k ln(10/z0) = Ub (1 + sqrt(Cd)/k psi_m)
        # (exact at zu = 10), floored like NEMO.
        UN10 = jnp.maximum(_UN10_FLOOR_M_S, Ub * (1.0 + sqrt_Cd / k * psim))

        CdN = large_yeager_neutral_cd(UN10, nemo_parity=True)
        sqrt_CdN = jnp.sqrt(CdN)

        # L&Y 2004 Eq. (10a) at LOG(zu/10) = 0.
        denom = 1.0 - sqrt_CdN / k * psim
        Cd = jnp.maximum(CdN / (denom * denom), _CX_MIN)
        sqrt_Cd = jnp.sqrt(Cd)

        # L&Y 2004 Eq. (10b/10c) at LOG(zu/10) = 0.
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
        Override the latent heat (legacy algo only; the NCAR path uses the
        SST-dependent NEMO ``L_vap``).
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

    # --- NCAR (NEMO-faithful) ---------------------------------------------
    # NEMO blk_oce_1 preprocessing: ssq at the ABSOLUTE SST + slp; potential
    # air temperature via the 10-m barometric pressure + Exner; potential
    # SST via Exner at slp.  Stability, sensible flux and L_vap then use the
    # POTENTIAL pair (theta_air, theta_sst) exactly like NEMO.
    slp_for_sst = constants.p_atm_std if slp_Pa is None else jnp.asarray(
        slp_Pa, dtype=jnp.float64)
    ssq = seawater_q_sat(sst, slp_Pa) if q_sfc is None else jnp.asarray(
        q_sfc, dtype=jnp.float64)
    theta_air, p10 = potential_air_temperature_10m(T_air, q_a, slp_Pa)
    theta_sst = exner_potential_temperature(sst, slp_for_sst)

    Cd, Ch, Ce, Ub = ncar_transfer_coefficients(
        theta_air, q_a, theta_sst, ssq, wind_speed, nb_iter=nb_iter,
    )

    # NEMO: rhoa = rho_air(ztabs, q, zpre) with the 10-m pressure.
    rho = rho_air_moist(T_air, q_a, p10) if rho_air is None else jnp.asarray(
        rho_air, dtype=jnp.float64)
    # NEMO BULK_FORMULA: zUrho = Ub * MAX(rho, 1.0)
    Urho = Ub * jnp.maximum(rho, _RHO_FLUX_FLOOR)

    # Stress components: |tau| = Urho*Cd*wind, direction along the (relative)
    # wind vector; atmospheric convention (leading minus) as documented.  With
    # the current feedback on (vfac>0), u_rel/v_rel are dU/dV and Ub (hence Urho)
    # is |dU| -- so tau = -rho*Cd*|dU|*dU, the NEMO relative-stress form.
    tau_x = -Urho * Cd * u_rel
    tau_y = -Urho * Cd * v_rel

    L_vap = latent_heat_vaporization_sst(theta_sst)   # NEMO L_vap(pTs=zsspt)
    z_evap = Urho * Ce * (q_a - ssq)          # NEMO zevap (<0 evaporating)
    shflx = Urho * Ch * (theta_air - theta_sst) * moist_air_cp(q_a)
    lhflx = L_vap * z_evap
    evap = -z_evap                            # rn_efac = 1
    return tau_x, tau_y, shflx, lhflx, evap


__all__ = [
    "air_sea_fluxes",
    "exner_potential_temperature",
    "large_yeager_cd",
    "large_yeager_ch",
    "latent_heat_vaporization_sst",
    "moist_air_cp",
    "ncar_transfer_coefficients",
    "potential_air_temperature_10m",
    "pressure_at_height",
    "rho_air_moist",
    "seawater_q_sat",
]
