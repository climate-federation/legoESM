"""Selectable SI3/BL99 layered thermodynamics for the existing ice model.

This promotes the previously parked ice-only Thomas solve into the production
package and implements the constrained NEMO 5.0.2 identity used by the
``C1D_OMIP_L3`` card: three snow plus three ice layers, P07 conductivity, and
the iterative surface-flux boundary condition.  Source identity:
``icethd_zdf_bl99.F90:34-590``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.core import transcendentals
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ice.constants_config import IceConstantsConfig
from legoesm.ice.snow import snow_ice_flooding
from legoesm.timestepping.tridiagonal import thomas_solve

from legoesm import constants as canonical_constants


class SI3SurfaceForcing(NamedTuple):
    """Boundary values seen at ``ice_thd`` entry (``ice1D.F90:393-421``)."""

    qns_ice: jax.Array
    qsr_ice: jax.Array
    dqns_ice: jax.Array
    qtr_ice_top: jax.Array
    t_bottom: jax.Array
    sss: jax.Array
    evaporation: jax.Array
    snow_precipitation: jax.Array
    qprec_ice: jax.Array
    qcn_ice_bottom: jax.Array
    qsb_ice_bottom: jax.Array
    fhld: jax.Array
    qlead: jax.Array
    frq_m: jax.Array | float = 1.0
    sst_m: jax.Array | None = None  # sst_1d [degC], read only by ice_thd_da

class SI3ZDFResult(NamedTuple):
    T_surface: jax.Array
    T_ice: jax.Array
    T_snow: jax.Array
    e_ice: jax.Array
    e_snow: jax.Array
    qns_ice: jax.Array
    qtr_ice_bottom: jax.Array
    qcn_ice_top: jax.Array
    qcn_ice_bottom: jax.Array
    iterations: jax.Array


class SI3ColumnArrays(NamedTuple):
    """Array carrier at one registered `ice_thd` boundary."""

    concentration: jax.Array
    h_ice: jax.Array
    h_snow: jax.Array
    T_surface: jax.Array
    e_ice: jax.Array
    e_snow: jax.Array
    S_bulk: jax.Array
    S_layers: jax.Array
    age_volume: jax.Array


class SI3StepTrace(NamedTuple):
    """Rule-1d boundary trace in `icethd.F90:112-225` order."""

    entry: SI3ColumnArrays
    post_zdf: SI3ColumnArrays
    post_dh: SI3ColumnArrays
    post_temp1: SI3ColumnArrays
    post_sal: SI3ColumnArrays
    post_temp2: SI3ColumnArrays
    post_do: SI3ColumnArrays
    exit: SI3ColumnArrays


_EPS10 = 1.0e-10
_EPS20 = 1.0e-20
_H_MIN = 1.0e-3
_T_CONVERGENCE = 1.0e-4
_T_SURFACE_EPS = 1.0e-5
_MAX_ITERATIONS = 200

# Fixed values in the sole supported ORCA1/SI3 identity.  Keeping them in this
# module-level provenance block prevents them from masquerading as tunables or
# combining into an unsupported selector mixture.  Sources:
# namelist_ice_cfg:81-124; icevar.F90:537-598,1598-1599;
# icethd_zdf_bl99.F90:83,151-223,261-275; icethd_do.F90:172-269.
_P07_BETA_SALINITY = 0.09
_P07_BETA_TEMPERATURE = 0.011
_P07_CONDUCTIVITY_MIN = 0.1
_BRINE_HEAT_CAPACITY_FACTOR = 18009.0
_SNOW_EXTINCTION = 10.0
_ICE_EXTINCTION = 1.0  # C1D card rn_kappa_i (namelist_ice_ref:170)
_SNOW_SSL_DEPTH = 0.03
_ICE_SSL_DEPTH = 0.10
_SNOW_FRACTION_DEPTH = 0.02
_SNOW_BLOW_FRACTION = 0.66
_MINIMUM_ICE_SALINITY = 0.1
_NEW_ICE_SALINITY_FRACTION = 0.75
_DRAINAGE_TARGET = 5.0
_DRAINAGE_TIMESCALE = 1730000.0
_FLUSHING_TARGET = 2.0
_FLUSHING_TIMESCALE = 864000.0
_OPTION2_LOWER_TRANSITION = 3.5
_OPTION2_UPPER_TRANSITION = 4.5
_OPTION2_LAYER_SHAPE = (1.0 / 3.0, 1.0, 5.0 / 3.0)
_NEW_ICE_THICKNESS = 0.05  # C1D card rn_hinew, c1d_omip_l3_namelist_ice_cfg:113
_MAXIMUM_ICE_CONCENTRATION = 0.99999

# --- salinity-4 drainage/flushing (NEMO 5.0.2 icethd_sal.f90:240-249) ---
_EPS06 = 1.0e-6
_SAL4_CP_BRINE = 4.0e6
_SAL4_BETA = 7.5e-4
_SAL4_CONDUCTIVITY_BRINE = 0.523
_SAL4_VISCOSITY_BRINE = 1.8e-6
_SAL4_PERMEABILITY = 3.0e-8
_SAL4_BRINE_DENSITY_SLOPE = 0.8e-3
_SAL4_BRINE_VOLUME_SENTINEL = 1.0e20
_SAL4_VC19_LINEAR = 18.7
_SAL4_VC19_QUADRATIC = 0.519
_SAL4_VC19_CUBIC = 0.00535


# Compatibility constants for the public functions exposed by the parked
# prototype.  The selected SI3 path below always receives IceConstantsConfig;
# these helpers preserve callers of the promoted prototype without changing
# their canonical-legoESM convention (negative enthalpy for cold ice).
_LEGACY_T_FLOOR_C = 1.0e-2
_LEGACY_DISC_FLOOR = 1.0


def _legacy_neg_temp_c(T_K):
    T_c = T_K - canonical_constants.T_freeze
    return -jnp.maximum(-T_c, _LEGACY_T_FLOOR_C)


def freezing_temperature(S):
    """Legacy BL99 melting temperature [degC], retained after promotion."""

    return -canonical_constants.mu_ice_freeze * S


def ice_enthalpy(T_K, S):
    """Legacy canonical BL99 enthalpy [J m-3], negative for cold ice."""

    Tm = freezing_temperature(S)
    Tc = _legacy_neg_temp_c(T_K)
    return -canonical_constants.rho_ice * (
        canonical_constants.c_pi * (Tm - Tc)
        + canonical_constants.L_f * (1.0 - Tm / Tc)  # latent-ok: Bitz-Lipscomb mushy-ice enthalpy at the melting point (oracle thermodynamics)
        - canonical_constants.c_pw * Tm
    )


def option2_salinity_profile(
    S_bulk,
    sss,
    *,
    _nemo_source_round: bool = True,
    _nemo_salprof_factoring: bool = True,
):
    """SI3 option-2 diagnostic linear profile (`icevar.F90:488-506,560-578`)."""

    sr = nemo_source_round if _nemo_source_round else (lambda x: x)
    S_bulk = jnp.asarray(S_bulk)
    alpha = jnp.where(
        S_bulk >= _OPTION2_UPPER_TRANSITION,
        0.0,
        jnp.where(
            S_bulk <= _OPTION2_LOWER_TRANSITION,
            1.0,
            _OPTION2_UPPER_TRANSITION - S_bulk,
        ),
    )
    alpha = jnp.where(
        S_bulk >= 0.5 * _NEW_ICE_SALINITY_FRACTION * sss,
        0.0,
        alpha,
    )
    alpha = jnp.where(
        S_bulk
        <= 3.0
        * jnp.minimum(
            _MINIMUM_ICE_SALINITY,
            _NEW_ICE_SALINITY_FRACTION * sss,
        ),
        0.0,
        alpha,
    )
    if _nemo_salprof_factoring:
        jk_minus_half = jnp.asarray((0.5, 1.5, 2.5), dtype=S_bulk.dtype)
        r1_nlay_i = jnp.asarray(1.0 / 3.0, dtype=S_bulk.dtype)
        sloped = sr(
            sr(
                sr(sr(alpha[..., None] * S_bulk[..., None]) * 2.0)
                * jk_minus_half
            )
            * r1_nlay_i
        )
        constant = sr((1.0 - alpha[..., None]) * S_bulk[..., None])
        return sloped + constant
    z = jnp.asarray(_OPTION2_LAYER_SHAPE, dtype=S_bulk.dtype)
    return S_bulk[..., None] * (sr(alpha[..., None] * z) + (1.0 - alpha[..., None]))


def ice_enthalpy_from_temperature(
    T_K,
    S,
    constants: IceConstantsConfig,
    *,
    _nemo_order: bool = True,
    _nemo_source_round: bool = True,
):
    """NEMO positive energy-of-melting `icevar.F90:938-946` [J m-3]."""

    sr = nemo_source_round if _nemo_source_round else (lambda x: x)
    T_m = sr(-constants.liquidus_slope * S)
    if not _nemo_order:
        T_c = jnp.minimum(T_K - constants.T0, -_EPS10)
        T_c = jnp.minimum(T_c, T_m)
        return constants.rho_ice * (
            constants.c_ice * (T_m - T_c)
            + constants.latent_fusion * jnp.maximum(0.0, 1.0 - T_m / T_c)
            - constants.c_ocean * T_m
        )
    # Preserve the assignment and rounding order in `icevar.F90:938-946`
    # (compiled `icevar.f90:953-961`): first clip the Kelvin temperature, then
    # form Celsius, and finally evaluate the three parenthesised energy terms
    # without FMA contraction across the sum/difference.
    # icevar.f90:955 MIN(t_i, ztmelts+rt0) is `minsd t_i, ztmelts+rt0`;
    # :958 MIN(t-rt0, -epsi10) and MAX(0, .) keep the constant on ties/NaN.
    T_clipped = _minsd(sr(T_m + constants.T0), T_K)
    T_c = sr(T_clipped - constants.T0)
    sensible = sr(constants.c_ice * sr(T_m - T_c))
    liquid = _maxsd(sr(1.0 - sr(T_m / _minsd(T_c, -_EPS10))), 0.0)
    latent = sr(constants.latent_fusion * liquid)
    ocean = sr(constants.c_ocean * T_m)
    return sr(constants.rho_ice * sr(sr(sensible + latent) - ocean))


def ice_temperature_from_enthalpy(
    e,
    S,
    constants: IceConstantsConfig | None = None,
    *,
    _nemo_order: bool = True,
    _nemo_source_round: bool = True,
):
    """Invert enthalpy [K] in the selected NEMO or legacy convention.

    Supplying ``constants`` selects NEMO's positive energy-of-melting
    convention (`icethd.F90:221-247`, `icevar.F90:390-396`).  Omitting it
    preserves the public API of the promoted prototype, whose enthalpy is
    negative for cold ice.
    """

    if constants is None:
        Tm = freezing_temperature(S)
        q_spec = e / canonical_constants.rho_ice
        b = ((canonical_constants.c_pw - canonical_constants.c_pi) * Tm
             - canonical_constants.L_f - q_spec)  # latent-ok: Bitz-Lipscomb mushy-ice enthalpy at the melting point (oracle thermodynamics)
        c = canonical_constants.L_f * Tm  # latent-ok: Bitz-Lipscomb mushy-ice enthalpy at the melting point (oracle thermodynamics)
        disc = jnp.maximum(
            b * b - 4.0 * canonical_constants.c_pi * c,
            _LEGACY_DISC_FLOOR,
        )
        T_c = (-b - jnp.sqrt(disc)) / (2.0 * canonical_constants.c_pi)
        return T_c + canonical_constants.T_freeze

    sr = nemo_source_round if _nemo_source_round else (lambda x: x)
    T_m = sr(-constants.liquidus_slope * S)
    if not _nemo_order:
        b = (
            (constants.c_ocean - constants.c_ice) * T_m
            + e / constants.rho_ice
            - constants.latent_fusion
        )
        disc = jnp.maximum(
            b * b - 4.0 * constants.c_ice * constants.latent_fusion * T_m,
            0.0,
        )
        return constants.T0 - (b + jnp.sqrt(disc)) / (2.0 * constants.c_ice)
    # Written order of `icethd.F90:233-243` and `icevar.F90:390-396`
    # (compiled `icevar.f90:404-410`).  NEMO multiplies by its precomputed
    # reciprocals, applies `* 0.5 * r1_rcpi`, and adds `rt0` to the rounded
    # Celsius root without FMA contraction across `zbbb` or `zccc`.
    inverse_rho_ice = 1.0 / constants.rho_ice
    inverse_c_ice = 1.0 / constants.c_ice
    b = (
        sr(
            sr((constants.c_ocean - constants.c_ice) * T_m)
            + sr(e * inverse_rho_ice)
        )
        - constants.latent_fusion
    )
    # icethd.f90:285 MAX(., 0) is `maxsd 0, .` (+0 on a tie or NaN), then `sqrtsd`.
    disc = _maxsd(
        sr(b * b) - sr((4.0 * constants.c_ice * constants.latent_fusion) * T_m),
        0.0,
    )
    return sr(
        constants.T0 - sr(sr(sr(b + sr(jnp.sqrt(disc))) * 0.5) * inverse_c_ice)
    )


def ice_thermal_conductivity(T_K, S):
    """Legacy Untersteiner conductivity [W m-1 K-1]."""

    T_c = _legacy_neg_temp_c(T_K)
    conductivity = (
        canonical_constants.k_ice_default
        + canonical_constants.beta_ice_cond * S / T_c
    )
    return jnp.maximum(conductivity, _P07_CONDUCTIVITY_MIN)


def ice_specific_heat(T_K, S):
    """Legacy BL99 effective specific heat [J kg-1 K-1]."""

    T_c = _legacy_neg_temp_c(T_K)
    return (
        canonical_constants.c_pi
        + canonical_constants.L_f * canonical_constants.mu_ice_freeze * S  # latent-ok: Bitz-Lipscomb mushy-ice enthalpy at the melting point (oracle thermodynamics)
        / (T_c * T_c)
    )


def bitz_lipscomb_conduction_step(q_layers, h, S, F_top, T_bottom_K, dt):
    """Compatibility entry point for the former parked ice-only solver."""

    n = q_layers.shape[-1]
    S_b = jnp.broadcast_to(S[..., None] if S.ndim == h.ndim else S,
                           q_layers.shape)
    dz = jnp.maximum(h, _H_MIN) / n
    dz_e = dz[..., None]
    T_old = ice_temperature_from_enthalpy(q_layers, S_b)
    c_i = ice_specific_heat(T_old, S_b)
    k_c = ice_thermal_conductivity(T_old, S_b)
    k_iface = 2.0 * k_c[..., :-1] * k_c[..., 1:] / (
        k_c[..., :-1] + k_c[..., 1:]
    )
    heat_cap = canonical_constants.rho_ice * c_i * dz_e / dt
    g_int = k_iface / dz_e
    g_bot = 2.0 * k_c[..., -1] / dz
    zero = jnp.zeros_like(heat_cap[..., :1])
    a = jnp.concatenate([zero, -g_int], axis=-1)
    c = jnp.concatenate([-g_int, zero], axis=-1)
    g_up = jnp.concatenate([zero, g_int], axis=-1)
    g_dn = jnp.concatenate([g_int, zero], axis=-1)
    b = heat_cap + g_up + g_dn
    b = b.at[..., -1].add(g_bot)
    d = heat_cap * T_old
    d = d.at[..., 0].add(F_top)
    d = d.at[..., -1].add(g_bot * T_bottom_K)
    T_new = thomas_solve(a, b, c, d)
    q_new = q_layers + canonical_constants.rho_ice * c_i * (T_new - T_old)
    F_bottom = g_bot * (T_new[..., -1] - T_bottom_K)
    return q_new, T_new, F_bottom


def snow_enthalpy_from_temperature(
    T_K,
    constants: IceConstantsConfig,
    *,
    _nemo_source_round: bool = True,
):
    """Snow energy of melting, `icevar.F90:949-952` [J m-3]."""

    sr = nemo_source_round if _nemo_source_round else (lambda x: x)
    return sr(
        constants.rho_snow
        * sr(sr(constants.c_ice * sr(constants.T0 - T_K)) + constants.latent_fusion)
    )


def snow_temperature_from_enthalpy(
    e,
    constants: IceConstantsConfig,
    *,
    _nemo_order: bool = True,
    _nemo_glo2eqv_snow_temp: bool = True,
    _nemo_source_round: bool = True,
):
    """Invert snow enthalpy with NEMO's global-to-equivalent bounds.

    `icevar.F90:412-413` (compiled `icevar.f90:426-427`) reconstructs this
    diagnostic immediately before each `ice_thd` call as
    ``rt0 + MAX(-100, MIN(r1_rcpi * (-r1_rhos * e_s + rLfus), 0))``.
    Setting ``_nemo_glo2eqv_snow_temp=False`` retains the `icethd_dh.F90:498-503`
    expanded form for before/after ablation.
    """
    inverse_rho_snow = 1.0 / constants.rho_snow
    inverse_c_ice = 1.0 / constants.c_ice
    if _nemo_order and _nemo_glo2eqv_snow_temp:
        sr = nemo_source_round if _nemo_source_round else (lambda x: x)
        return sr(
            constants.T0
            + jnp.maximum(
                -100.0,
                jnp.minimum(
                    sr(
                        inverse_c_ice
                        * sr(sr(-inverse_rho_snow * e) + constants.latent_fusion)
                    ),
                    0.0,
                ),
            )
        )
    if _nemo_order:
        unbounded = constants.T0 + (
            -e * inverse_rho_snow * inverse_c_ice
            + constants.latent_fusion * inverse_c_ice
        )
    else:
        unbounded = constants.T0 + (
            -e / constants.rho_snow + constants.latent_fusion
        ) / constants.c_ice
    return jnp.clip(unbounded, constants.T0 - 100.0, constants.T0)


def p07_conductivity(
    T_K,
    S,
    constants: IceConstantsConfig,
    *,
    _nemo_source_round: bool = True,
):
    """Pringle et al. selection, ``icethd_zdf_bl99.f90:273-285`` floored at :293.

    ``rcnd_i + 0.09 * S / MIN(-epsi10, T - rt0) - 0.011 * (T - rt0)``: only the
    divisor is capped; the last term uses the uncapped Celsius temperature.
    """

    sr = nemo_source_round if _nemo_source_round else (lambda x: x)
    T_raw = sr(T_K - constants.T0)
    T_c = _minsd(T_raw, -_EPS10)  # emitted `minsd -epsi10, T - rt0`
    return _maxsd(
        sr(constants.k_ice + sr(_P07_BETA_SALINITY * S / T_c))
        - sr(_P07_BETA_TEMPERATURE * T_raw),
        _P07_CONDUCTIVITY_MIN,
    )  # :293 MAX(zkimin, ztcond_i_cp) is emitted `maxsd zkimin, ztcond_i_cp`


def _maxsd(dst, src):
    """gfortran ``MAX`` as emitted: ``maxsd src, dst`` keeps ``dst`` only when strictly larger."""
    return jnp.where(dst > src, dst, src)


def _minsd(dst, src):
    """gfortran ``MIN`` as emitted: ``minsd src, dst`` keeps ``dst`` only when strictly smaller."""
    return jnp.where(dst < src, dst, src)


def _div_const(x, c):
    """``x / c`` as a true divide: XLA CPU otherwise rewrites a constant divisor as ``x * (1/c)``."""
    return x / nemo_source_round(jnp.asarray(c, dtype=jnp.result_type(x)))


def _sum0(terms):
    """Fortran ``SUM`` over the last axis: ``+0`` then each term in order."""
    acc = nemo_source_round(jnp.zeros(terms.shape[:-1], dtype=terms.dtype))
    for k in range(terms.shape[-1]):
        acc = nemo_source_round(acc + terms[..., k])
    return acc


def _radiation(
    h_i,
    h_s,
    qtr_top,
    T_surface,
    *,
    rn_kappa_i,
    rn_kappa_s,
    _nemo_zdf_layer_reciprocals: bool = True,
    _nemo_source_round: bool = True,
    _nlay=(3, 3),
):
    """SSL transmission/absorption, ``icethd_zdf_bl99.f90:215-240`` (nn_qtrice=0)."""

    del T_surface  # nn_qtrice=0: zraext_s = rn_kappa_s (`icethd_zdf_bl99.f90:161-162`)
    sr = nemo_source_round if _nemo_source_round else (lambda x: x)
    nlay_i, nlay_s = _nlay
    dtype = h_i.dtype
    js = jnp.arange(1, nlay_s + 1, dtype=dtype)
    ji = jnp.arange(1, nlay_i + 1, dtype=dtype)
    if _nemo_zdf_layer_reciprocals:
        # `icethd_zdf_bl99.f90:172-186`; MAX(zh_min, h) is `maxsd zh_min, h`.
        dzs = jnp.where(
            h_s > 0.0, _maxsd(h_s, _H_MIN) * (1.0 / nlay_s), 0.0
        )[..., None]
        dzi = jnp.where(
            h_i > 0.0, _maxsd(h_i, _H_MIN) * (1.0 / nlay_i), 0.0
        )[..., None]
    else:
        dzs = jnp.maximum(h_s, _H_MIN)[..., None] / nlay_s
        dzi = jnp.maximum(h_i, _H_MIN)[..., None] / nlay_i
    snow_fraction = h_s / (
        h_s + _SNOW_FRACTION_DEPTH
    )  # nn_snwfra=2, icevar.f90:1613
    # :219 `zradtr_s(0) * EXP( - zraext_s * MAX( 0, zh_s * REAL(jk) - zhs_ssl ) )`
    tr_s = qtr_top[..., None] * transcendentals.exp(
        -rn_kappa_s * _maxsd(sr(dzs * js) - _SNOW_SSL_DEPTH, 0.0)
    )
    tr_s0 = jnp.concatenate([qtr_top[..., None], tr_s], axis=-1)
    ab_s = tr_s0[..., :-1] - tr_s0[..., 1:]
    through_snow = tr_s[..., -1]
    top_i = sr(snow_fraction * through_snow) + sr((1.0 - snow_fraction) * qtr_top)
    tr_i = sr(
        snow_fraction[..., None] * through_snow[..., None]
        * transcendentals.exp(-rn_kappa_i * _maxsd(sr(dzi * ji) - _H_MIN, 0.0))
    ) + sr(
        (1.0 - snow_fraction)[..., None] * qtr_top[..., None]
        * transcendentals.exp(
            -rn_kappa_i * _maxsd(sr(dzi * ji) - _ICE_SSL_DEPTH, 0.0)
        )
    )
    tr_i0 = jnp.concatenate([top_i[..., None], tr_i], axis=-1)
    ab_i = tr_i0[..., :-1] - tr_i0[..., 1:]
    return ab_s, ab_i, tr_i[..., -1]


class _SI3ZDFCore(NamedTuple):
    t_su: jax.Array
    t_i: jax.Array
    t_s: jax.Array
    qns_ice: jax.Array
    qtr_ice_bot: jax.Array
    k_i: jax.Array
    k_s: jax.Array
    kappa_comb: jax.Array
    tcond_i: jax.Array
    isnow: jax.Array
    iterations: jax.Array


def _si3_zdf_bl99_core(
    *,
    h_i,
    h_s,
    t_su,
    t_i_old,
    t_s_old,
    sz_i,
    t_bo,
    qns_ice,
    qsr_ice,
    dqns_ice,
    qtr_ice_top,
    dt: float,
    constants: IceConstantsConfig,
    rn_kappa_i: float,
    rn_kappa_s: float,
    _maximum_iterations: int = _MAX_ITERATIONS,
    _nemo_branch_ranges: bool = True,
    _nemo_zdf_layer_reciprocals: bool = True,
    _nemo_source_round: bool = True,
) -> _SI3ZDFCore:
    """`ice_thd_zdf_BL99(np_cnd_OFF)` iteration, ``icethd_zdf_bl99.f90:170-603``.

    Layer counts are the trailing extents of ``t_i_old`` (nlay_i) and ``t_s_old``
    (nlay_s).  The static 200-iterate ``fori_loop`` freezes each column after
    NEMO's ``1e-4 K`` test, which keeps reverse-mode AD available.
    """

    sr = nemo_source_round if _nemo_source_round else (lambda x: x)
    nlay_i, nlay_s = t_i_old.shape[-1], t_s_old.shape[-1]
    if nlay_i < 2 or nlay_s < 1:
        raise ValueError(
            f"SI3 BL99 needs nlay_i >= 2 and nlay_s >= 1 (the nlay_i == 1 arms are not transcribed); got {nlay_i}, {nlay_s}"
        )
    dtype = t_i_old.dtype
    T0 = constants.T0
    k_snow = constants.k_snow
    T_surface = _minsd(t_su, T0 - _T_SURFACE_EPS)  # :202
    ab_s, ab_i, qtr_bottom = _radiation(
        h_i, h_s, qtr_ice_top, T_surface,
        rn_kappa_i=rn_kappa_i, rn_kappa_s=rn_kappa_s,
        _nemo_zdf_layer_reciprocals=_nemo_zdf_layer_reciprocals,
        _nemo_source_round=_nemo_source_round,
        _nlay=(nlay_i, nlay_s),
    )

    dz_i = jnp.maximum(h_i, _H_MIN) / nlay_i
    dz_s = jnp.maximum(h_s, _H_MIN) / nlay_s
    snow_present = h_s > 0.0
    ice_present = h_i > 0.0
    zh_i_pos = _maxsd(h_i, _H_MIN) * (1.0 / nlay_i)  # :173
    zh_s_pos = _maxsd(h_s, _H_MIN) * (1.0 / nlay_s)  # :181
    zh_i = jnp.where(ice_present, zh_i_pos, 0.0)
    z1_h_i = jnp.where(ice_present, 1.0 / zh_i_pos, 0.0)
    zh_s = jnp.where(snow_present, zh_s_pos, 0.0)
    z1_h_s = jnp.where(snow_present, 1.0 / zh_s_pos, 0.0)
    isnow = jnp.where(snow_present, 1.0, 0.0)
    isnow_comb = jnp.where(h_s < _H_MIN, _div_const(h_s, _H_MIN), 1.0)  # :190-194
    inverse_rho_ice = 1.0 / constants.rho_ice
    inverse_rho_snow = 1.0 / constants.rho_snow
    inverse_c_ice = 1.0 / constants.c_ice

    def body(_iteration, carry):
        (Tsu, Tsu_previous, Ti, Ts, qns, k_i_frozen, k_s_frozen, comb_frozen,
         tcond_frozen, converged, iterations) = carry
        active = ~converged

        # P07 conductivity at the nlay_i+1 interfaces (:273-285), floor :293.
        Ti_iface = 0.5 * (Ti[..., :-1] + Ti[..., 1:])
        S_iface = 0.5 * (sz_i[..., :-1] + sz_i[..., 1:])
        k0 = p07_conductivity(Ti[..., 0], sz_i[..., 0], constants,
                              _nemo_source_round=_nemo_source_round)
        km = p07_conductivity(Ti_iface, S_iface, constants,
                              _nemo_source_round=_nemo_source_round)
        kb = p07_conductivity(t_bo, sz_i[..., -1], constants,
                              _nemo_source_round=_nemo_source_round)
        tcond_i = jnp.concatenate([k0[..., None], km, kb[..., None]], axis=-1)
        if _nemo_zdf_layer_reciprocals:
            # :326-341, zghe = 1 (ln_virtual_itd = F).
            # Every product materialised: XLA otherwise re-expands zkappa/zeta
            # inside the matrix-entry fusions and regroups their constants.
            k_i = sr(tcond_i * z1_h_i[..., None])
            k_s_bulk = jnp.broadcast_to(
                sr(k_snow * z1_h_s)[..., None], Ti.shape[:-1] + (nlay_s + 1,)
            )
            k_si = (
                sr(sr(isnow * k_snow) * k0)
                / sr(0.5 * (sr(k0 * zh_s) + sr(k_snow * zh_i)))
            )
        else:
            k_i = tcond_i / dz_i[..., None]
            k_s_bulk = jnp.full(Ti.shape[:-1] + (nlay_s + 1,), k_snow,
                                dtype=dtype) / dz_s[..., None]
            k_si = k_snow * (k_i[..., 0] * dz_i) / (
                0.5 * ((k_i[..., 0] * dz_i) * dz_s + k_snow * dz_i)
            )
        k_s = k_s_bulk.at[..., -1].set(k_si)
        # :339 before the :341 snow-ice overwrite of zkappa_i(0).
        kappa_comb = sr(isnow_comb * k_s[..., 0]) + sr((1.0 - isnow_comb) * k_i[..., 0])
        k_i = k_i.at[..., 0].set(jnp.where(snow_present, k_si, k_i[..., 0]))

        # :353 `zcpi = rcpi + zgamma * sz_i / MAX( (t_i - rt0) * (ztiold - rt0), epsi10 )`
        ci = constants.c_ice + sr(
            _BRINE_HEAT_CAPACITY_FACTOR
            * sz_i
            / _maxsd((Ti - T0) * (t_i_old - T0), _EPS10)
        )
        if _nemo_zdf_layer_reciprocals:
            eta_i = sr(sr(dt * inverse_rho_ice * z1_h_i[..., None]) / ci)  # :354
            eta_s = jnp.broadcast_to(
                sr(dt * inverse_rho_snow * inverse_c_ice * z1_h_s)[..., None],  # :362
                Ts.shape,
            )
        else:
            eta_i = dt / (constants.rho_ice * dz_i[..., None] * ci)
            eta_s = dt / (constants.rho_snow * dz_s[..., None] * constants.c_ice)

        qns_new = qns + sr(dqns_ice * (Tsu - Tsu_previous))  # :387
        fnet = qsr_ice - qtr_ice_top + qns_new  # :389

        zero = jnp.zeros_like(Tsu)
        one = jnp.ones_like(Tsu)
        rows = []  # (a, b, c, d) for NEMO jm = 1..nrow, snow-covered arm
        rows.append((zero, dqns_ice - sr(2.0 * k_s[..., 0]),
                     sr(2.0 * k_s[..., 0]), sr(dqns_ice * Tsu) - fnet))  # :449-452
        rows.append((sr(-eta_s[..., 0] * k_s[..., 0]) * 2.0,
                     1.0 + sr(eta_s[..., 0] * (k_s[..., 1] + sr(2.0 * k_s[..., 0]))),
                     sr(-eta_s[..., 0] * k_s[..., 1]),
                     t_s_old[..., 0] + sr(eta_s[..., 0] * ab_s[..., 0])))  # :455-458
        for k in range(1, nlay_s):  # :429-435, jk = k + 1
            rows.append((sr(-eta_s[..., k] * k_s[..., k]),
                         1.0 + sr(eta_s[..., k] * (k_s[..., k] + k_s[..., k + 1])),
                         sr(-eta_s[..., k] * k_s[..., k + 1]),
                         t_s_old[..., k] + sr(eta_s[..., k] * ab_s[..., k])))
        for k in range(nlay_i - 1):  # :409-415, jk = k + 1
            rows.append((sr(-eta_i[..., k] * k_i[..., k]),
                         1.0 + sr(eta_i[..., k] * (k_i[..., k] + k_i[..., k + 1])),
                         sr(-eta_i[..., k] * k_i[..., k + 1]),
                         t_i_old[..., k] + sr(eta_i[..., k] * ab_i[..., k])))
        rows.append((sr(-eta_i[..., -1] * k_i[..., -2]),
                     1.0 + sr(eta_i[..., -1] * (k_i[..., -2] + sr(2.0 * k_i[..., -1]))),
                     zero,
                     t_i_old[..., -1]
                     + sr(eta_i[..., -1] * (ab_i[..., -1] + sr(2.0 * k_i[..., -1] * t_bo)))))  # :419-423
        # Snow-free arm (:475-490): identity above jm_min, surface at jm = nlay_s+1.
        nosnow = [(zero, one, zero, zero) for _ in range(nlay_s)]
        nosnow.append((zero, dqns_ice - sr(2.0 * k_i[..., 0]),
                       sr(2.0 * k_i[..., 0]), sr(dqns_ice * Tsu) - fnet))
        nosnow.append((sr(-eta_i[..., 0] * k_i[..., 0]) * 2.0,
                       1.0 + sr(eta_i[..., 0] * (k_i[..., 1] + sr(2.0 * k_i[..., 0]))),
                       sr(-eta_i[..., 0] * k_i[..., 1]),
                       t_i_old[..., 0] + sr(eta_i[..., 0] * ab_i[..., 0])))
        nosnow.extend(rows[nlay_s + 2:])
        a, b, c, d = (jnp.stack([r[n] for r in rows], axis=-1) for n in range(4))
        na, nb, nc, nd = (jnp.stack([r[n] for r in nosnow], axis=-1) for n in range(4))
        branch_snow_present = (
            snow_present if _nemo_branch_ranges else jnp.ones_like(snow_present)
        )
        sel = branch_snow_present[..., None]
        a, b, c, d = (jnp.where(sel, x, y) for x, y in ((a, na), (b, nb), (c, nc), (d, nd)))

        melting_surface = jnp.where(_nemo_branch_ranges, Tsu >= T0, False)
        snow_melting = branch_snow_present & melting_surface
        no_snow_melting = (~branch_snow_present) & melting_surface
        b = b.at[..., 0].set(jnp.where(snow_melting, 1.0, b[..., 0]))
        c = c.at[..., 0].set(jnp.where(snow_melting, 0.0, c[..., 0]))
        d = d.at[..., 0].set(jnp.where(snow_melting, Tsu, d[..., 0]))
        a = a.at[..., 1].set(jnp.where(snow_melting, 0.0, a[..., 1]))
        d = d.at[..., 1].set(jnp.where(
            snow_melting,
            t_s_old[..., 0]
            + sr(eta_s[..., 0] * (ab_s[..., 0] + sr(2.0 * k_s[..., 0] * Tsu))),
            d[..., 1],
        ))  # :466-469
        s = nlay_s
        b = b.at[..., s].set(jnp.where(no_snow_melting, 1.0, b[..., s]))
        c = c.at[..., s].set(jnp.where(no_snow_melting, 0.0, c[..., s]))
        d = d.at[..., s].set(jnp.where(no_snow_melting, Tsu, d[..., s]))
        a = a.at[..., s + 1].set(jnp.where(no_snow_melting, 0.0, a[..., s + 1]))
        d = d.at[..., s + 1].set(jnp.where(
            no_snow_melting,
            t_i_old[..., 0]
            + sr(eta_i[..., 0] * (ab_i[..., 0] + sr(2.0 * k_i[..., 0] * Tsu))),
            d[..., s + 1],
        ))  # :509-512

        thomas_order = (
            "nemo_unnormalised" if _nemo_source_round else "nemo_unnormalised_unrounded"
        )
        solution = thomas_solve(a, b, c, d, thomas_order)  # :536-567
        solved_surface = jnp.where(
            branch_snow_present, solution[..., 0], solution[..., s]
        )
        # :577 MAX( MIN( t_su, rt0 ), rt0 - 100 ); :582 likewise for snow.
        Tsu_candidate = _maxsd(
            _minsd(jnp.where(melting_surface, Tsu, solved_surface), T0), T0 - 100.0
        )
        Ts_candidate = jnp.where(
            branch_snow_present[..., None],
            _maxsd(_minsd(solution[..., 1:s + 1], T0), T0 - 100.0),
            Ts,
        )
        # :588-589 `ztmelts = -rTmlt*sz + rt0`; MIN(t_i, ztmelts) is `minsd t_i, ztmelts`.
        melt = T0 - sr(constants.liquidus_slope * sz_i)
        Ti_candidate = _maxsd(_minsd(melt, solution[..., s + 1:]), T0 - 100.0)
        delta = jnp.maximum(
            jnp.abs(Tsu_candidate - Tsu),
            jnp.maximum(
                jnp.max(jnp.abs(Ts_candidate - Ts), axis=-1),
                jnp.max(jnp.abs(Ti_candidate - Ti), axis=-1),
            ),
        )
        just_converged = active & (delta < _T_CONVERGENCE)
        Tsu = jnp.where(active, Tsu_candidate, Tsu)
        Ti = jnp.where(active[..., None], Ti_candidate, Ti)
        Ts = jnp.where(active[..., None], Ts_candidate, Ts)
        qns = jnp.where(active, qns_new, qns)
        k_i_frozen = jnp.where(active[..., None], k_i, k_i_frozen)
        k_s_frozen = jnp.where(active[..., None], k_s, k_s_frozen)
        comb_frozen = jnp.where(active, kappa_comb, comb_frozen)
        tcond_frozen = jnp.where(active[..., None], _maxsd(tcond_i, _P07_CONDUCTIVITY_MIN),
                                 tcond_frozen)
        iterations = jnp.where(just_converged, _iteration + 1, iterations)
        return (Tsu, jnp.where(active, carry[0], Tsu_previous), Ti, Ts, qns,
                k_i_frozen, k_s_frozen, comb_frozen, tcond_frozen,
                converged | just_converged, iterations)

    init = (
        T_surface,
        jnp.asarray(t_su, dtype=dtype),
        t_i_old,
        t_s_old,
        jnp.asarray(qns_ice, dtype=dtype),
        jnp.zeros(t_i_old.shape[:-1] + (nlay_i + 1,), dtype=dtype),
        jnp.zeros(t_i_old.shape[:-1] + (nlay_s + 1,), dtype=dtype),
        jnp.zeros_like(T_surface),
        jnp.zeros(t_i_old.shape[:-1] + (nlay_i + 1,), dtype=dtype),
        jnp.zeros_like(h_i, dtype=bool),
        jnp.full_like(h_i, _maximum_iterations, dtype=jnp.int32),
    )
    (Tsu, _, Ti, Ts, qns, k_i_frozen, k_s_frozen, comb_frozen, tcond_frozen,
     _, iterations) = jax.lax.fori_loop(0, _maximum_iterations, body, init)
    return _SI3ZDFCore(Tsu, Ti, Ts, qns, qtr_bottom, k_i_frozen, k_s_frozen,
                       comb_frozen, tcond_frozen, isnow, iterations)


def _zdf_interface_fluxes(core: _SI3ZDFCore, t_bo, sr):
    """`icethd_zdf_bl99.f90:782-784`, from the iteration-frozen kappas (:319-343)."""

    zg1 = 2.0
    zg1s = 2.0
    isnow = core.isnow
    qcn_bottom = -core.k_i[..., -1] * zg1 * (t_bo - core.t_i[..., -1])
    qcn_top = sr(
        -isnow * core.k_s[..., 0] * zg1s * (core.t_s[..., 0] - core.t_su)
    ) - sr(
        (1.0 - isnow) * core.k_i[..., 0] * zg1 * (core.t_i[..., 0] - core.t_su)
    )
    return qcn_bottom, qcn_top


class SI3ZDFOutputs(NamedTuple):
    """`ice_thd_zdf_BL99(np_cnd_OFF)` exit, ``icethd_zdf_bl99.f90:778-879``."""

    t_su: jax.Array
    t_i: jax.Array
    t_s: jax.Array
    e_i: jax.Array
    e_s: jax.Array
    qns_ice: jax.Array
    qcn_ice_top: jax.Array
    qcn_ice_bot: jax.Array
    qtr_ice_bot: jax.Array
    hfx_err_dif: jax.Array
    hfx_dif: jax.Array
    cnd_ice: jax.Array
    t1_ice: jax.Array
    t_si: jax.Array
    # Picard passes this cell executed; NEMO's per-cell `tice_cvgstp_1d` (:596).
    n_picard: jax.Array


def _required(name, value):
    if value is None:
        raise ValueError(f"SI3 thermodynamics requires {name!r}; no default")
    return jnp.asarray(value) if getattr(value, "dtype", None) == jnp.float32 else jnp.asarray(value, dtype=jnp.float64)


def si3_ice_thd_zdf_bl99(
    *,
    h_i, h_s, t_su, t_i, t_s, e_i, e_s, sz_i, a_i, t_bo,
    qns_ice, qsr_ice, dqns_ice, qtr_ice_top, hfx_err_dif, hfx_dif,
    rDt_ice, rn_cnd_s, rn_kappa_i, rn_kappa_s, nn_qtrice, nn_snwfra,
    ln_cndi_P07, ln_cndflx, ln_virtual_itd, ln_zdf_chkcvg,
    constants: IceConstantsConfig, _maximum_iterations: int = _MAX_ITERATIONS,
) -> SI3ZDFOutputs:
    """Exact-input SI3 heat diffusion for one gathered category (any nlay_i >= 2, nlay_s).

    Arrays are NEMO's 1-D gathers at ``ice_thd_zdf`` entry, layer axis last.
    Only the arm the production deck executes is transcribed; every other
    selector is refused.
    """

    for name, value, wanted in (
        ("nn_qtrice", nn_qtrice, 0), ("nn_snwfra", nn_snwfra, 2),
        ("ln_cndi_P07", ln_cndi_P07, True), ("ln_cndflx", ln_cndflx, False),
        ("ln_virtual_itd", ln_virtual_itd, False),
        ("ln_zdf_chkcvg", ln_zdf_chkcvg, False),
    ):
        if value is None or type(value) is not type(wanted) or value != wanted:
            raise ValueError(f"SI3 BL99 transcribes only {name} = {wanted!r}; got {value!r}")
    for name, value in (("rDt_ice", rDt_ice), ("rn_cnd_s", rn_cnd_s),
                        ("rn_kappa_i", rn_kappa_i), ("rn_kappa_s", rn_kappa_s)):
        if value is None:
            raise ValueError(f"SI3 BL99 requires {name!r}; no default")
    sr = nemo_source_round
    rDt_ice = float(rDt_ice)
    r1_Dt_ice = 1.0 / rDt_ice
    cst = constants._replace(k_snow=float(rn_cnd_s))  # icethd_zdf.f90: rcnd_s = rn_cnd_s
    h_i, h_s, t_su, t_i, t_s, e_i, e_s, sz_i, a_i, t_bo = (
        _required(n, v) for n, v in (
            ("h_i", h_i), ("h_s", h_s), ("t_su", t_su), ("t_i", t_i), ("t_s", t_s),
            ("e_i", e_i), ("e_s", e_s), ("sz_i", sz_i), ("a_i", a_i), ("t_bo", t_bo)))
    qns_ice, qsr_ice, dqns_ice, qtr_ice_top, hfx_err_dif, hfx_dif = (
        _required(n, v) for n, v in (
            ("qns_ice", qns_ice), ("qsr_ice", qsr_ice), ("dqns_ice", dqns_ice),
            ("qtr_ice_top", qtr_ice_top), ("hfx_err_dif", hfx_err_dif),
            ("hfx_dif", hfx_dif)))
    nlay_i, nlay_s = t_i.shape[-1], t_s.shape[-1]
    r1_nlay_i, r1_nlay_s = 1.0 / nlay_i, 1.0 / nlay_s
    # :148-150 diag error on heat, PART 1.
    zq_ini = sr(sr(sr(_sum0(e_i) * h_i) * r1_nlay_i)
                + sr(sr(_sum0(e_s) * h_s) * r1_nlay_s))
    core = _si3_zdf_bl99_core(
        h_i=h_i, h_s=h_s, t_su=t_su, t_i_old=t_i, t_s_old=t_s, sz_i=sz_i,
        t_bo=t_bo, qns_ice=qns_ice, qsr_ice=qsr_ice, dqns_ice=dqns_ice,
        qtr_ice_top=qtr_ice_top, dt=rDt_ice, constants=cst,
        rn_kappa_i=float(rn_kappa_i), rn_kappa_s=float(rn_kappa_s), _maximum_iterations=_maximum_iterations,
    )
    qcn_bot, qcn_top = _zdf_interface_fluxes(core, t_bo, sr)
    hfx_err_dif = hfx_err_dif - sr((core.qns_ice - qns_ice) * a_i)  # :786
    # ice_var_enthalpy (`icevar.f90:952-967`), called at :813.
    ti_new = _minsd(cst.T0 - sr(cst.liquidus_slope * sz_i), core.t_i)  # :955
    e_i_new = ice_enthalpy_from_temperature(ti_new, sz_i, cst)
    e_s_new = snow_enthalpy_from_temperature(core.t_s, cst)
    zdq = -zq_ini + sr(sr(sr(_sum0(e_i_new) * h_i) * r1_nlay_i)
                       + sr(sr(_sum0(e_s_new) * h_s) * r1_nlay_s))  # :817-818
    zdq_rate = sr(zdq * r1_Dt_ice)
    tr_bot = core.qtr_ice_bot
    head = jnp.where(
        core.t_su < cst.T0,
        sr(core.qns_ice + qsr_ice),
        sr(qcn_top + qtr_ice_top),
    )  # :822-828
    zhfx_err = sr(sr(sr(sr(head - tr_bot) - qcn_bot) + zdq_rate) * a_i)
    hfx_err_dif = hfx_err_dif + zhfx_err  # :838
    hfx_dif = hfx_dif - sr(zdq_rate * a_i)  # :841
    cnd_ice = jnp.where(
        h_i >= _ICE_SSL_DEPTH,
        2.0 * core.kappa_comb,
        _div_const(2.0 * core.tcond_i[..., 0], _ICE_SSL_DEPTH),
    )  # :853-858
    isnow = core.isnow
    t1_ice = sr(isnow * core.t_s[..., 0]) + sr((1.0 - isnow) * ti_new[..., 0])  # :859
    lhs = sr(sr(cst.k_snow * h_i) * r1_nlay_i)
    rhs = sr(sr(core.tcond_i[..., 1] * h_s) * r1_nlay_s)
    t_si = jnp.where(
        h_s >= _SNOW_SSL_DEPTH,
        (sr(lhs * core.t_s[..., -1]) + sr(rhs * ti_new[..., 0])) / (lhs + rhs),
        core.t_su,
    )  # :871-878
    return SI3ZDFOutputs(core.t_su, ti_new, core.t_s, e_i_new, e_s_new,
                         core.qns_ice, qcn_top, qcn_bot, tr_bot, hfx_err_dif,
                         hfx_dif, cnd_ice, t1_ice, t_si, core.iterations)


class SI3DHFluxes(NamedTuple):
    """1-D flux accumulators ``ice_thd_dh`` updates (``ice1D``), in source order."""

    hfx_res: jax.Array
    wfx_snw_sum: jax.Array
    hfx_spr: jax.Array
    wfx_spr: jax.Array
    hfx_sub: jax.Array
    wfx_snw_sub: jax.Array
    hfx_snw: jax.Array
    wfx_res: jax.Array
    sfx_res: jax.Array
    hfx_thd: jax.Array
    hfx_sum: jax.Array
    wfx_sum: jax.Array
    sfx_sum: jax.Array
    wfx_ice_sub: jax.Array
    sfx_sub: jax.Array
    wfx_err_sub: jax.Array
    hfx_bog: jax.Array
    wfx_bog: jax.Array
    sfx_bog: jax.Array
    hfx_bom: jax.Array
    wfx_bom: jax.Array
    sfx_bom: jax.Array
    sfx_sni: jax.Array
    wfx_sni: jax.Array
    wfx_snw_sni: jax.Array


class SI3DHIncrements(NamedTuple):
    """``dh_*`` diagnostics, zeroed by ``ice_thd`` before each category (``icethd.f90:192-194``)."""

    dh_s_itm: jax.Array
    dh_s_sum: jax.Array
    dh_i_itm: jax.Array
    dh_i_sum: jax.Array
    dh_i_sub: jax.Array
    dh_i_bog: jax.Array
    dh_i_bom: jax.Array
    dh_snowice: jax.Array


class SI3DHOutputs(NamedTuple):
    h_i: jax.Array
    h_s: jax.Array
    t_su: jax.Array
    a_i: jax.Array
    e_i: jax.Array
    e_s: jax.Array
    t_s: jax.Array
    sz_i: jax.Array
    s_i: jax.Array
    qml_ice: jax.Array
    dh: SI3DHIncrements
    fluxes: SI3DHFluxes


def _cumulative_remap(h_old, content_old, nlay, sr):
    """``ice_var_vremap`` (`icevar.f90:1010-1046`) and ``snw_ent`` (`icethd_dh.f90:584-619`).

    ``h_old``/``content_old`` are lists of the old segments' thickness and
    content (``q*h``); the result is the new per-layer intensive value.
    """

    # Materialized seed: XLA folds 0 + x to x, which would keep a -0 segment's sign.
    zero = nemo_source_round(jnp.zeros_like(h_old[0]))
    h_cum0, e_cum0 = [zero], [zero]
    for h, e in zip(h_old, content_old):
        h_cum0.append(h_cum0[-1] + h)
        e_cum0.append(e_cum0[-1] + e)
    zhnew = h_cum0[-1] * (1.0 / nlay)  # SUM(ph_old) * r1_nlay, same +0-seeded sequence
    h_cum1 = [zero]
    for _ in range(nlay):
        h_cum1.append(h_cum1[-1] + zhnew)
    e_cum1 = [zero] * (nlay + 1)
    for old in range(1, len(h_old) + 1):
        for new in range(1, nlay):
            selected = (h_cum1[new] <= h_cum0[old]) & (h_cum1[new] > h_cum0[old - 1])
            denominator = h_cum0[old] - h_cum0[old - 1]
            interpolated = (
                sr(e_cum0[old - 1] * (h_cum0[old] - h_cum1[new]))
                + sr(e_cum0[old] * (h_cum1[new] - h_cum0[old - 1]))
            ) / jnp.where(selected, denominator, 1.0)
            e_cum1[new] = jnp.where(selected, interpolated, e_cum1[new])
    e_cum1[nlay] = e_cum0[-1]
    # MAX(0, .) is `maxsd 0, x`; MAX(zhnew, epsi20) is `maxsd epsi20, zhnew`.
    divisor = _maxsd(zhnew, _EPS20)
    return jnp.stack([
        _maxsd(e_cum1[k] - e_cum1[k - 1], 0.0) / divisor for k in range(1, nlay + 1)
    ], axis=-1)


def _si3_ice_thd_dh_core(
    *,
    h_i, h_s, t_su, a_i, at_i, t_i, t_s, e_i, e_s, sz_i, s_i,
    t_bo, sss, sst, qns_ice, qsr_ice, qtr_ice_top, qcn_ice_top, qcn_ice_bot,
    qtr_ice_bot, qsb_ice_bot, fhld, frq_m, sprecip, qprec_ice, evap_ice,
    fluxes: SI3DHFluxes,
    rDt_ice: float, rn_snwblow: float, rn_sinew: float, nn_icesal: int,
    constants: IceConstantsConfig,
) -> SI3DHOutputs:
    """``ice_thd_dh`` with ``ln_cndflx = F``, ``icethd_dh.f90:102-541``.

    Every statement is applied to every gathered cell; NEMO's ``IF`` blocks
    become ``where`` on the same predicate so a skipped update keeps its bits.
    MIN/MAX operand roles follow the record build's emitted ``minsd``/``maxsd``.
    """

    if nn_icesal not in (2, 4):
        raise ValueError(f"SI3 ice_thd_dh transcribes nn_icesal 2 and 4 only; got {nn_icesal!r}")
    sr = nemo_source_round
    nlay_i, nlay_s = e_i.shape[-1], e_s.shape[-1]
    r1_nlay_i, r1_nlay_s = 1.0 / nlay_i, 1.0 / nlay_s
    rhoi, rhos, rho0 = constants.rho_ice, constants.rho_snow, constants.rho_ocean
    rcp, rcpi, rLfus = constants.c_ocean, constants.c_ice, constants.latent_fusion
    rTmlt, rt0 = constants.liquidus_slope, constants.T0
    r1_rhoi, r1_rhos = 1.0 / rhoi, 1.0 / rhos
    dt = float(rDt_ice)
    r1_dt = 1.0 / dt
    f = fluxes._asdict()
    zsw = 1.0  # zswitch_sal, nn_icesal = 2 or 4 (:102-105)

    def acc(name, cond, term):
        """``f(name) = f(name) + term`` inside ``IF (cond)``; term already rounded."""
        new = f[name] + term
        f[name] = new if cond is None else jnp.where(cond, new, f[name])

    def rate(*factors):
        """Left-to-right product ``x1 * x2 * ... * a_i * r1_Dt_ice`` without contraction."""
        out = factors[0]
        for x in factors[1:] + (a_i, r1_dt):
            out = sr(out * x)
        return out

    zero = sr(jnp.zeros_like(h_i))  # materialized: NEMO's 0._wp + (-0) is +0, XLA's folded 0 + x is not
    dh = {k: zero for k in SI3DHIncrements._fields}
    zsnw = sr(1.0 - sr(transcendentals.pow(1.0 - at_i, rn_snwblow)))  # :108, icevar.f90:1642
    z1_rho = 1.0 / (rhos + rho0 - rhoi)  # :111

    # Available heat (:120-130).
    qml = jnp.where(
        t_su >= rt0, qns_ice + qsr_ice - qtr_ice_top - qcn_ice_top, 0.0
    )
    zq_top = _maxsd(sr(qml * dt), 0.0)
    zf_tt = sr(sr(qcn_ice_bot + qsb_ice_bot) + fhld) + sr(qtr_ice_bot * frq_m)
    zq_bot = _maxsd(sr(zf_tt * dt), 0.0)
    zs_i = [sz_i[..., k] for k in range(nlay_i)] if nn_icesal == 4 else [s_i] * nlay_i
    # Layer carriers 0..nlay_i+1 (:138-147) and 0..nlay_s (:150-155).
    dz0 = sr(h_i * r1_nlay_i)
    zh_i = [zero] + [dz0] * nlay_i + [zero]
    zh_i_old = list(zh_i)
    ze_i_old = [zero] + [sr(dz0 * e_i[..., k]) for k in range(nlay_i)] + [zero]
    zs_i_old = [zero] + [sr(dz0 * zs_i[k]) for k in range(nlay_i)] + [zero]
    zh_s = [zero] + [sr(h_s * r1_nlay_s)] * nlay_s
    ze_s = [zero] + [e_s[..., k] for k in range(nlay_s)]

    # Snow internal melting (:164-174).
    for jk in range(1, nlay_s + 1):
        warm = t_s[..., jk - 1] > rt0
        acc("hfx_res", warm, -rate(ze_s[jk], zh_s[jk]))
        acc("wfx_snw_sum", warm, rate(rhos, zh_s[jk]))
        dh["dh_s_itm"] = jnp.where(warm, dh["dh_s_itm"] - zh_s[jk], dh["dh_s_itm"])
        h_s = jnp.where(warm, _maxsd(h_s - zh_s[jk], 0.0), h_s)
        zh_s[jk] = jnp.where(warm, 0.0, zh_s[jk])
        ze_s[jk] = jnp.where(warm, 0.0, ze_s[jk])

    # Snow precipitation (:178-187).
    falling = sprecip > 0.0
    zh_s[0] = jnp.where(falling, sr(sr(sr(zsnw * sprecip) * dt) * r1_rhos) / at_i, 0.0)
    ze_s[0] = jnp.where(falling, _maxsd(-qprec_ice, 0.0), 0.0)
    acc("hfx_spr", falling, rate(ze_s[0], zh_s[0]))
    acc("wfx_spr", falling, -rate(rhos, zh_s[0]))
    h_s = jnp.where(falling, h_s + zh_s[0], h_s)

    # Sublimation / deposition (:194-212).  MAX(-A, -h_s) is emitted -MIN(A, h_s).
    zdeltah = -_minsd(sr(sr(evap_ice * r1_rhos) * dt), h_s)
    zevap_rema = sr(evap_ice * dt) + sr(zdeltah * rhos)
    deposit = (zdeltah > 0.0) & (ze_s[0] == 0.0)
    ze_s[0] = jnp.where(deposit, sr(rhos * sr(rLfus - sr(rcpi * (t_su - rt0)))), ze_s[0])
    for jk in range(nlay_s + 1):
        zdum = _maxsd(-zh_s[jk], zdeltah)  # :200
        acc("hfx_sub", None, rate(ze_s[jk], zdum))
        acc("wfx_snw_sub", None, -rate(rhos, zdum))
        h_s = _maxsd(h_s + zdum, 0.0)
        zh_s[jk] = _maxsd(zh_s[jk] + zdum, 0.0)
        zdeltah = _minsd(zdeltah - zdum, 0.0)

    # Snow melting (:218-235).
    for jk in range(nlay_s + 1):
        melt = (zh_s[jk] > 0.0) & (zq_top > 0.0)
        zdum = _maxsd(-zq_top / _maxsd(_EPS20, ze_s[jk]), -zh_s[jk])  # :221-222
        acc("hfx_snw", melt, -rate(ze_s[jk], zdum))
        acc("wfx_snw_sum", melt, -rate(rhos, zdum))
        dh["dh_s_sum"] = jnp.where(melt, dh["dh_s_sum"] + zdum, dh["dh_s_sum"])
        zq_top = jnp.where(melt, _maxsd(zq_top + sr(zdum * ze_s[jk]), 0.0), zq_top)
        h_s = jnp.where(melt, _maxsd(h_s + zdum, 0.0), h_s)
        zh_s[jk] = jnp.where(melt, _maxsd(zh_s[jk] + zdum, 0.0), zh_s[jk])

    def melt_layer(jk, heat, internal_melt, active, kind):
        """One surface (:243-296) or basal (:378-433) layer step; ``kind`` 'sum'/'bom'."""

        nonlocal h_i
        ztmelts = sr(-rTmlt * sz_i[..., jk - 1])
        zEi = sr(-e_i[..., jk - 1] * r1_rhoi)
        zEw = sr(rcp * ztmelts)
        zdE = zEi - zEw
        zfmdt_gross = -heat / zdE
        zdum_melt = _minsd(_maxsd(sr(-zfmdt_gross * r1_rhoi), -zh_i[jk]), 0.0)
        zdum_itm = _minsd(-zh_i[jk], 0.0)
        zdum = jnp.where(internal_melt, zdum_itm, zdum_melt)
        surface = active & ~internal_melt
        internal = active & internal_melt
        heat = jnp.where(surface, _maxsd(heat - sr(sr(zdum * rhoi) * zdE), 0.0), heat)
        dh["dh_i_itm"] = jnp.where(internal, dh["dh_i_itm"] + zdum, dh["dh_i_itm"])
        key = "dh_i_sum" if kind == "sum" else "dh_i_bom"
        dh[key] = jnp.where(surface, dh[key] + zdum, dh[key])
        zfmdt_itm = sr(-zdum * rhoi)
        acc("hfx_res", internal, rate(zEi, zfmdt_itm))
        acc("wfx_res", internal, -rate(rhoi, zdum))
        acc("sfx_res", internal, -rate(rhoi, zdum, zs_i[jk - 1]))
        zfmdt = sr(-rhoi * zdum)
        acc("hfx_thd", surface, rate(zEw, zfmdt))
        acc("hfx_" + kind, surface, -rate(zdE, zfmdt))
        acc("wfx_" + kind, surface, -rate(rhoi, zdum))
        acc("sfx_" + kind, surface, -rate(rhoi, zdum, zs_i[jk - 1]))
        zh_i[jk] = jnp.where(active, _maxsd(zh_i[jk] + zdum, 0.0), zh_i[jk])
        h_i = jnp.where(active, _maxsd(h_i + zdum, 0.0), h_i)
        zs_i_old[jk] = jnp.where(active, zs_i_old[jk] + sr(zdum * zs_i[jk - 1]), zs_i_old[jk])
        ze_i_old[jk] = jnp.where(active, ze_i_old[jk] + sr(zdum * e_i[..., jk - 1]), ze_i_old[jk])
        zh_i_old[jk] = jnp.where(active, zh_i_old[jk] + zdum, zh_i_old[jk])
        return heat

    # Surface ice melting and sublimation (:243-325).
    icount = []
    always = jnp.ones_like(h_i, dtype=bool)
    for jk in range(1, nlay_i + 1):
        tmelt = sr(-rTmlt * sz_i[..., jk - 1])
        internal_melt = t_i[..., jk - 1] >= tmelt + rt0
        zq_top = melt_layer(jk, zq_top, internal_melt, always, "sum")
        # MAX(-zh_i, -zevap_rema*r1_rhoi) is emitted -MIN(zevap_rema*r1_rhoi, zh_i).
        zdum = _maxsd(-sr(zevap_rema * r1_rhoi), -zh_i[jk])
        acc("hfx_sub", None, rate(e_i[..., jk - 1], zdum))
        acc("wfx_ice_sub", None, -rate(rhoi, zdum))
        acc("sfx_sub", None, -rate(rhoi, zdum, zs_i[jk - 1]))
        zevap_rema = zevap_rema + sr(zdum * rhoi)
        zh_i[jk] = _maxsd(zh_i[jk] + zdum, 0.0)
        h_i = _maxsd(h_i + zdum, 0.0)
        dh["dh_i_sub"] = dh["dh_i_sub"] + zdum
        zs_i_old[jk] = zs_i_old[jk] + sr(zdum * zs_i[jk - 1])
        ze_i_old[jk] = ze_i_old[jk] + sr(zdum * e_i[..., jk - 1])
        zh_i_old[jk] = zh_i_old[jk] + zdum
        icount.append(jnp.where(zh_i[jk] > 0.0, 0, 1))
    acc("wfx_err_sub", None, -rate(zevap_rema))  # :328

    # Basal growth (:338-374).
    growing = zf_tt < 0.0
    zs_i_new = jnp.where(
        growing, sr(sr(zsw * rn_sinew) * sss) + sr((1.0 - zsw) * zs_i[0]), 0.0
    )
    ztmelts = sr(-rTmlt * zs_i_new)
    zt_i_new = sr(zsw * t_bo) + sr((1.0 - zsw) * t_i[..., -1])
    zEi = (
        sr(rcpi * sr(zt_i_new - sr(ztmelts + rt0)))
        - sr(rLfus * _maxsd(sr(1.0 - sr(ztmelts / _minsd(zt_i_new - rt0, -_EPS10))), 0.0))
    ) + sr(rcp * ztmelts)
    zEw = sr(rcp * sr(t_bo - rt0))
    zdE = zEi - zEw
    dh_bog = jnp.where(growing, sr(dt * _maxsd(sr(zf_tt / sr(zdE * rhoi)), 0.0)), 0.0)
    dh["dh_i_bog"] = dh_bog
    zfmdt = sr(-rhoi * dh_bog)
    acc("hfx_thd", growing, rate(zEw, zfmdt))
    acc("hfx_bog", growing, -rate(zdE, zfmdt))
    acc("wfx_bog", growing, -rate(rhoi, dh_bog))
    acc("sfx_bog", growing, -rate(rhoi, dh_bog, zs_i_new))
    top = nlay_i + 1
    zh_i[top] = jnp.where(growing, zh_i[top] + dh_bog, zh_i[top])
    h_i = jnp.where(growing, h_i + dh_bog, h_i)
    zs_i_old[top] = jnp.where(growing, zs_i_old[top] + sr(dh_bog * zs_i_new), zs_i_old[top])
    ze_i_old[top] = jnp.where(growing, ze_i_old[top] + sr(dh_bog * sr(-zEi * rhoi)), ze_i_old[top])
    zh_i_old[top] = jnp.where(growing, zh_i_old[top] + dh_bog, zh_i_old[top])

    # Basal melt (:378-434).
    for jk in range(nlay_i, 0, -1):
        active = (zf_tt > 0.0) & (jk > icount[jk - 1])
        tmelt = sr(-rTmlt * sz_i[..., jk - 1])
        internal_melt = t_i[..., jk - 1] >= tmelt + rt0
        zq_bot = melt_layer(jk, zq_bot, internal_melt, active, "bom")

    # Remove snow if the ice melted entirely (:438-449).
    gone = h_i == 0.0
    for jk in range(nlay_s + 1):
        acc("hfx_res", gone, -rate(ze_s[jk], zh_s[jk]))
        acc("wfx_res", gone, rate(rhos, zh_s[jk]))
        ze_s[jk] = jnp.where(gone, 0.0, ze_s[jk])
        zh_s[jk] = jnp.where(gone, 0.0, zh_s[jk])
    h_s = jnp.where(gone, 0.0, h_s)

    # Snow-ice formation (:456-495).
    dh_sni = _maxsd(sr(sr(sr(rhos * h_s) + sr((rhoi - rho0) * h_i)) * z1_rho), 0.0)
    dh["dh_snowice"] = dh_sni
    h_i = h_i + dh_sni
    h_s = h_s - dh_sni
    zfmdt = sr((rhos - rhoi) * dh_sni)
    zEw = sr(rcp * sst)
    acc("hfx_thd", None, rate(zEw, zfmdt))
    acc("sfx_sni", None, rate(sss, zfmdt))
    acc("wfx_sni", None, -rate(dh_sni, rhoi))
    acc("wfx_snw_sni", None, rate(dh_sni, rhos))
    zh_i[0] = zh_i[0] + dh_sni
    zdeltah = dh_sni
    zs_i_old[0] = zs_i_old[0] - sr(sr(zfmdt * sss) * r1_rhoi)
    zh_i_old[0] = zh_i_old[0] + dh_sni
    ze_i_old[0] = ze_i_old[0] + sr(zfmdt * zEw)
    for jk in range(nlay_s, -1, -1):
        zdum = _minsd(zh_s[jk], zdeltah)  # :490 emitted `minsd zdeltah, zh_s`
        zh_s[jk] = _maxsd(zh_s[jk] - zdum, 0.0)
        ze_i_old[0] = ze_i_old[0] + sr(zdum * ze_s[jk])
        zdeltah = _maxsd(zdeltah - zdum, 0.0)

    # Remaps (:506-529).
    e_s_new = _cumulative_remap(zh_s, [sr(e * h) for e, h in zip(ze_s, zh_s)], nlay_s, sr)
    t_s_new = jnp.where(
        (h_s > 0.0)[..., None],
        rt0 + (sr(sr(-e_s_new * r1_rhos) * (1.0 / rcpi)) + sr(rLfus * (1.0 / rcpi))),
        rt0,
    )
    e_i_new = _cumulative_remap(zh_i_old, ze_i_old, nlay_i, sr)
    if nn_icesal == 4:
        sz_i_new = _cumulative_remap(zh_i_old, zs_i_old, nlay_i, sr)
        s_i_new = s_i
    else:
        sz_i_new = sz_i
        zs_sni = sr(sr(sss * (rhoi - rhos)) * r1_rhoi)
        hden = _maxsd(h_i, _EPS10)  # `maxsd epsi10, h_i`
        zds = sr(sr(zs_sni - s_i) * dh_sni) / hden
        zds = zds + sr(sr(zs_i_new - s_i) * dh_bog) / hden
        s_i_new = s_i + zds

    # :537-541
    gone = h_i == 0.0
    a_i = jnp.where(gone, 0.0, a_i)
    h_s = jnp.where(gone, 0.0, h_s)
    t_su = jnp.where(gone, rt0, t_su)
    return SI3DHOutputs(
        h_i, h_s, t_su, a_i, e_i_new, e_s_new, t_s_new, sz_i_new, s_i_new, qml,
        SI3DHIncrements(**dh), SI3DHFluxes(**f),
    )


def si3_ice_thd_temp(*, h_i, e_i, sz_i, constants: IceConstantsConfig):
    """``ice_thd_temp``, ``icethd.f90:279-291``: t_i from enthalpy where h_i > 0, else rt0."""

    h_i, e_i, sz_i = (_required(n, v) for n, v in (("h_i", h_i), ("e_i", e_i), ("sz_i", sz_i)))
    t = ice_temperature_from_enthalpy(e_i, sz_i, constants)
    return jnp.where((h_i > 0.0)[..., None], t, constants.T0)


_DH_INPUTS = (
    "h_i", "h_s", "t_su", "a_i", "at_i", "t_i", "t_s", "e_i", "e_s", "sz_i", "s_i",
    "t_bo", "sss", "sst", "qns_ice", "qsr_ice", "qtr_ice_top", "qcn_ice_top",
    "qcn_ice_bot", "qtr_ice_bot", "qsb_ice_bot", "fhld", "frq_m", "sprecip",
    "qprec_ice", "evap_ice",
)


def si3_ice_thd_dh(*, fluxes: SI3DHFluxes, rDt_ice, rn_snwblow, rn_sinew,
                   nn_icesal, ln_cndflx, ln_cndemulate,
                   constants: IceConstantsConfig, **arrays) -> SI3DHOutputs:
    """Exact-input ``ice_thd_dh`` for one gathered category; every input required."""

    if ln_cndflx is not False or ln_cndemulate is not False:
        raise ValueError("SI3 ice_thd_dh transcribes ln_cndflx = ln_cndemulate = .false. only")
    if type(nn_icesal) is not int:
        raise ValueError(f"nn_icesal must be an int; got {nn_icesal!r}")
    for name, value in (("rDt_ice", rDt_ice), ("rn_snwblow", rn_snwblow),
                        ("rn_sinew", rn_sinew)):
        if value is None:
            raise ValueError(f"SI3 ice_thd_dh requires {name!r}; no default")
    unknown = set(arrays) - set(_DH_INPUTS)
    if unknown:
        raise ValueError(f"unknown ice_thd_dh inputs {sorted(unknown)}")
    if fluxes is None:
        raise ValueError("SI3 ice_thd_dh requires the flux accumulators; no default")
    x = {n: _required(n, arrays.get(n)) for n in _DH_INPUTS}
    fluxes = SI3DHFluxes(*(_required(n, v) for n, v in zip(SI3DHFluxes._fields, fluxes)))
    return _si3_ice_thd_dh_core(
        **x, fluxes=fluxes, rDt_ice=float(rDt_ice), rn_snwblow=float(rn_snwblow),
        rn_sinew=float(rn_sinew), nn_icesal=nn_icesal, constants=constants,
    )


class SI3CategoryStages(NamedTuple):
    """O9 projection of one category: ZDF, DH and first TEMP exits (``icethd.f90:192-201``)."""

    zdf: SI3ZDFOutputs
    dh: SI3DHOutputs
    t_i: jax.Array


class SI3SalinityOutputs(NamedTuple):
    """Production salinity-4 exit and bounded-subcycle diagnostics."""

    sz_i: jax.Array
    sfx_bri: jax.Array
    sfx_res: jax.Array
    ndrainage: jax.Array
    nflushing: jax.Array
    cfl_drain: jax.Array
    cfl_flush: jax.Array
    smin_drain: jax.Array
    smin_flush: jax.Array


class SI3O10CategoryStages(NamedTuple):
    """O9 exits followed by O10 salinity and second-temperature exits."""

    o9: SI3CategoryStages
    sal: SI3SalinityOutputs
    t_i: jax.Array


def si3_ice_thd_category_stages(*, entry: dict, fluxes: SI3DHFluxes, namelist: dict,
                                rDt_ice, constants: IceConstantsConfig) -> SI3CategoryStages:
    """``ice_thd`` category body from the 1-D gather to the first ``ice_thd_temp``.

    ``entry`` holds NEMO's 1-D arrays after ``ice_thd_1d2d(jl, 1)``; ``namelist``
    the resolved scalars.  Stops before ``ice_thd_sal`` (O10).
    """

    nml = dict(namelist)
    if nml.get("ln_icedH") is not True:
        raise ValueError("O9 stage view transcribes ln_icedH = .true. only")
    zdf = si3_ice_thd_zdf_bl99(
        h_i=entry["h_i"], h_s=entry["h_s"], t_su=entry["t_su"], t_i=entry["t_i"],
        t_s=entry["t_s"], e_i=entry["e_i"], e_s=entry["e_s"], sz_i=entry["sz_i"],
        a_i=entry["a_i"], t_bo=entry["t_bo"], qns_ice=entry["qns_ice"],
        qsr_ice=entry["qsr_ice"], dqns_ice=entry["dqns_ice"],
        qtr_ice_top=entry["qtr_ice_top"], hfx_err_dif=entry["hfx_err_dif"],
        hfx_dif=entry["hfx_dif"], rDt_ice=rDt_ice,
        **{k: nml.get(k) for k in ("rn_cnd_s", "rn_kappa_i", "rn_kappa_s", "nn_qtrice",
                                   "nn_snwfra", "ln_cndi_P07", "ln_cndflx",
                                   "ln_virtual_itd", "ln_zdf_chkcvg")},
        constants=constants,
    )
    dh = si3_ice_thd_dh(
        h_i=entry["h_i"], h_s=entry["h_s"], t_su=zdf.t_su, a_i=entry["a_i"],
        at_i=entry["at_i"], t_i=zdf.t_i, t_s=zdf.t_s, e_i=zdf.e_i, e_s=zdf.e_s,
        sz_i=entry["sz_i"], s_i=entry["s_i"], t_bo=entry["t_bo"], sss=entry["sss"],
        sst=entry["sst"], qns_ice=zdf.qns_ice, qsr_ice=entry["qsr_ice"],
        qtr_ice_top=entry["qtr_ice_top"], qcn_ice_top=zdf.qcn_ice_top,
        qcn_ice_bot=zdf.qcn_ice_bot, qtr_ice_bot=zdf.qtr_ice_bot,
        qsb_ice_bot=entry["qsb_ice_bot"], fhld=entry["fhld"], frq_m=entry["frq_m"],
        sprecip=entry["sprecip"], qprec_ice=entry["qprec_ice"],
        evap_ice=entry["evap_ice"], fluxes=fluxes, rDt_ice=rDt_ice,
        **{k: nml.get(k) for k in ("rn_snwblow", "rn_sinew", "nn_icesal", "ln_cndflx",
                                   "ln_cndemulate")},
        constants=constants,
    )
    t_i = si3_ice_thd_temp(h_i=dh.h_i, e_i=dh.e_i, sz_i=dh.sz_i, constants=constants)
    return SI3CategoryStages(zdf, dh, t_i)


def _sal4_vc19_brine(t_i, T0):
    """VC19 brine salinity in the source order of ``icethd_sal.f90:342-350``."""

    zt1 = nemo_source_round(t_i - T0)
    zt2 = nemo_source_round(zt1 * zt1)
    zt3 = nemo_source_round(zt2 * zt1)
    linear = nemo_source_round(-_SAL4_VC19_LINEAR * zt1)
    quadratic = nemo_source_round(_SAL4_VC19_QUADRATIC * zt2)
    cubic = nemo_source_round(_SAL4_VC19_CUBIC * zt3)
    return nemo_source_round(nemo_source_round(linear - quadratic) - cubic)


def _sal4_effective_permeability(zv_br, *, initial):
    """Production harmonic mean; only the pre-loop form has the early exit."""

    nlay = zv_br.shape[-1]
    out = []
    one = jnp.ones_like(zv_br[..., 0])
    zero = jnp.zeros_like(one)
    for jk in range(nlay):
        ztmp = zero
        active = jnp.ones_like(one, dtype=bool)
        for jk2 in range(jk, nlay):
            low = zv_br[..., jk2] < _EPS06
            stop = low if initial else zv_br[..., jk2] == 0.0  # absorb +0 without 1/0
            go = active & ~stop
            safe_v = jnp.where(go, zv_br[..., jk2], one)
            zperm = nemo_source_round(_SAL4_PERMEABILITY * safe_v)
            zperm = nemo_source_round(zperm * safe_v)
            zperm = nemo_source_round(zperm * safe_v)
            reciprocal = nemo_source_round(one / zperm)
            candidate = nemo_source_round(ztmp + reciprocal)
            ztmp = jnp.where(go, candidate, jnp.where(active & stop, zero, ztmp))
            active = go
        count = jnp.asarray(float(nlay - jk), dtype=zv_br.dtype)
        safe_sum = jnp.where(ztmp > 0.0, ztmp, one)
        value = nemo_source_round(count / safe_sum)
        out.append(jnp.where(ztmp > 0.0, value, zero))
    return jnp.stack(out, axis=-1)


def _sal4_rjw_velocity(sz_i, zs_br, zmsk, sss, h_i, z_mid, z_edge, z1_h_i,
                      *, gravity, rn_alpha_rjw, rn_rc_rjw, initial):
    """RJW14 drainage velocity (`icethd_sal.f90:364-422,455-507`)."""

    nlay = sz_i.shape[-1]
    one = jnp.ones_like(h_i)
    zero = jnp.zeros_like(h_i)
    divisor = (
        _maxsd(zs_br[..., 1:-1], _EPS10)
        if initial
        else zs_br[..., 1:-1]
    )
    volume_numerator = nemo_source_round(zmsk * sz_i)
    zv_br = nemo_source_round(volume_numerator / jnp.where(volume_numerator == 0.0, 1.0, divisor))
    zperm_eff = _sal4_effective_permeability(zv_br, initial=initial)

    z1_cp_br = nemo_source_round(_div_const(one, _SAL4_CP_BRINE))
    z1_cnd_br = nemo_source_round(_div_const(one, _SAL4_CONDUCTIVITY_BRINE))
    z1_visc = nemo_source_round(_div_const(one, _SAL4_VISCOSITY_BRINE))
    transport_scale = nemo_source_round(z1_cnd_br * z1_visc)
    rayleigh = []
    for jk in range(nlay):
        positive = _maxsd(nemo_source_round(zs_br[..., jk + 1] - sss), zero)
        value = nemo_source_round(_SAL4_CP_BRINE * gravity)
        value = nemo_source_round(value * _SAL4_BETA)
        value = nemo_source_round(value * positive)
        value = nemo_source_round(value * zperm_eff[..., jk])
        value = nemo_source_round(value * h_i)
        value = nemo_source_round(value * nemo_source_round(1.0 - z_mid[jk]))
        value = nemo_source_round(value * transport_scale)
        rayleigh.append(value)
    rayleigh = jnp.stack(rayleigh, axis=-1)

    jc = jnp.full(h_i.shape, nlay + 1, dtype=jnp.int32)
    zc = zero
    zrae = zero
    for jk in range(nlay - 1, -1, -1):
        excess = nemo_source_round(rayleigh[..., jk] - rn_rc_rjw)
        hit = rayleigh[..., jk] >= rn_rc_rjw
        zrae = jnp.where(hit, _maxsd(zrae, excess), zrae)
        zc = jnp.where(hit, z_edge[jk], zc)
        jc = jnp.where(hit, jk + 1, jc)

    one_minus_zc = nemo_source_round(one - zc)
    z1_c2 = nemo_source_round(one / nemo_source_round(one_minus_zc * one_minus_zc))
    conductivity_over_capacity = nemo_source_round(_SAL4_CONDUCTIVITY_BRINE * z1_cp_br)
    velocity = []
    for jk in range(nlay):
        value = nemo_source_round(-rn_alpha_rjw * zrae)
        value = nemo_source_round(value * conductivity_over_capacity)
        value = nemo_source_round(value * nemo_source_round(z_mid[jk] - zc))
        value = nemo_source_round(value * z1_h_i)
        value = nemo_source_round(value * z1_c2)
        velocity.append(jnp.where((jk + 1) >= jc, value, zero))
    return jnp.stack(velocity, axis=-1)


def _sal4_flux_increment(rho_i, a_i, h_i, r1_nlay_i, zds, r1_Dt_ice):
    value = nemo_source_round(rho_i * a_i)
    value = nemo_source_round(value * h_i)
    value = nemo_source_round(value * r1_nlay_i)
    value = nemo_source_round(value * zds)
    return nemo_source_round(value * r1_Dt_ice)


def _si3_ice_thd_sal4_core(*, h_i, t_su, t_i, sz_i, t_bo, sss, a_i,
                           dh_i_sum, dh_s_sum, sfx_bri, sfx_res, rDt_ice,
                           r1_Dt_ice, r1_nlay_i, rho_i, rho_s, rho_w, gravity,
                           rn_simin, rn_sinew, rn_sal_himin, nn_drainage,
                           nn_flushing, rn_flushrate, rn_vbrc, rn_alpha_rjw,
                           rn_rc_rjw, constants: IceConstantsConfig):
    """Executing salinity-4 statements of ``ice_thd_sal.f90:311-683``."""

    nlay = sz_i.shape[-1]
    dtype = sz_i.dtype
    one = jnp.ones_like(h_i)
    zero = jnp.zeros_like(h_i)
    rDt_ice = jnp.asarray(rDt_ice, dtype=dtype)
    r1_Dt_ice = jnp.asarray(r1_Dt_ice, dtype=dtype)
    r1_nlay_i = jnp.asarray(r1_nlay_i, dtype=dtype)

    indices = jnp.arange(1, nlay + 1, dtype=dtype)
    z_mid = nemo_source_round(nemo_source_round(indices - 0.5) * r1_nlay_i)
    z_edge = nemo_source_round(jnp.arange(0, nlay + 1, dtype=dtype) * r1_nlay_i)
    zs_inner = _sal4_vc19_brine(t_i, constants.T0)
    zs_br = jnp.concatenate(
        [zero[..., None], zs_inner, sss[..., None]], axis=-1
    )
    zmsk = jnp.where(
        nemo_source_round(t_i - constants.T0) < -_EPS06,
        jnp.ones_like(t_i),
        jnp.zeros_like(t_i),
    )
    zhmelt = nemo_source_round(dh_s_sum + dh_i_sum)

    layer_thickness = nemo_source_round(h_i * r1_nlay_i)
    z1_h_drain = nemo_source_round(one / _maxsd(layer_thickness, _EPS10))
    zs_brmax = zero
    for jk in range(nlay):
        zs_brmax = _maxsd(zs_brmax, zs_inner[..., jk])
    drain_cell = (
        (h_i >= rn_sal_himin)
        & (zhmelt >= 0.0)
        & (zs_brmax > sss)
        & (t_su <= t_bo)
    )
    w0 = _sal4_rjw_velocity(
        sz_i, zs_br, zmsk, sss, h_i, z_mid, z_edge, z1_h_drain,
        gravity=gravity, rn_alpha_rjw=rn_alpha_rjw, rn_rc_rjw=rn_rc_rjw,
        initial=True,
    )
    cfl_test = zero
    for jk in range(nlay):
        cfl = nemo_source_round(nemo_source_round(w0[..., jk] * rDt_ice) * z1_h_drain)
        cfl_test = _maxsd(cfl_test, jnp.abs(cfl))
    requested = jnp.ceil(nemo_source_round(cfl_test * 2.0)).astype(jnp.int32)
    ndrain = jnp.where(
        drain_cell, jnp.minimum(jnp.asarray(nn_drainage, jnp.int32), requested), 0
    )
    zdt_drain = nemo_source_round(
        rDt_ice / jnp.maximum(jnp.asarray(1, jnp.int32), ndrain).astype(dtype)
    )

    def drainage_body(iteration, carry):
        salinity, salt_flux, cfl_max, sal_min = carry
        velocity = jax.lax.cond(
            iteration > 0,
            lambda _: _sal4_rjw_velocity(
                salinity, zs_br, zmsk, sss, h_i, z_mid, z_edge, z1_h_drain,
                gravity=gravity, rn_alpha_rjw=rn_alpha_rjw,
                rn_rc_rjw=rn_rc_rjw, initial=False,
            ),
            lambda _: w0,
            operand=None,
        )
        active = drain_cell & (iteration < ndrain)
        for jk in range(nlay):
            cfl = nemo_source_round(
                nemo_source_round(velocity[..., jk] * zdt_drain) * z1_h_drain
            )
            brine_gradient = nemo_source_round(zs_br[..., jk + 2] - zs_br[..., jk + 1])
            zds_raw = nemo_source_round(-cfl * brine_gradient)
            raw_salinity = nemo_source_round(salinity[..., jk] + zds_raw)
            cfl_max_next = _maxsd(cfl_max, jnp.abs(cfl))
            sal_min_next = _minsd(raw_salinity, sal_min)
            upper_product = nemo_source_round(rn_sinew * sss)
            lower_bound = _minsd(upper_product, rn_simin)
            floor_delta = nemo_source_round(-salinity[..., jk] + lower_bound)
            zds = _maxsd(zds_raw, floor_delta)
            sal_next = nemo_source_round(salinity[..., jk] + zds)
            flux_next = nemo_source_round(
                salt_flux
                - _sal4_flux_increment(
                    rho_i, a_i, h_i, r1_nlay_i, zds, r1_Dt_ice
                )
            )
            salinity = salinity.at[..., jk].set(
                jnp.where(active, sal_next, salinity[..., jk])
            )
            salt_flux = jnp.where(active, flux_next, salt_flux)
            cfl_max = jnp.where(active, cfl_max_next, cfl_max)
            sal_min = jnp.where(active, sal_min_next, sal_min)
        return salinity, salt_flux, cfl_max, sal_min

    sz_i, sfx_bri, cfl_drain, smin_drain = jax.lax.fori_loop(
        0,
        nn_drainage,
        jax.checkpoint(drainage_body),
        (sz_i, sfx_bri, zero, zero),
    )

    # Flushing uses upward meltwater flow and the same positive-ocean salt flux.
    z1_h_flush = nemo_source_round(one / _maxsd(layer_thickness, _EPS10))

    def flushing_velocity(jk):
        ice_mass = nemo_source_round(dh_i_sum * rho_i)
        snow_mass = nemo_source_round(dh_s_sum * rho_s)
        melt_mass = nemo_source_round(ice_mass + snow_mass)
        numerator = nemo_source_round(-rn_flushrate * melt_mass)
        density_term = nemo_source_round(_SAL4_BRINE_DENSITY_SLOPE * zs_inner[..., jk])
        density = nemo_source_round(rho_w * nemo_source_round(1.0 + density_term))
        return nemo_source_round(nemo_source_round(numerator / density) * r1_Dt_ice)

    flush_cell = (h_i >= rn_sal_himin) & (zhmelt < 0.0)
    flush_velocity = jnp.stack([flushing_velocity(jk) for jk in range(nlay)], axis=-1)
    cfl_test = zero
    for jk in range(nlay):
        cfl = nemo_source_round(
            nemo_source_round(flush_velocity[..., jk] * rDt_ice) * z1_h_flush
        )
        cfl_test = _maxsd(cfl_test, jnp.abs(cfl))
    requested = jnp.ceil(nemo_source_round(cfl_test * 2.0)).astype(jnp.int32)
    nflush = jnp.where(
        flush_cell, jnp.minimum(jnp.asarray(nn_flushing, jnp.int32), requested), 0
    )
    zdt_flush = nemo_source_round(
        rDt_ice / jnp.maximum(jnp.asarray(1, jnp.int32), nflush).astype(dtype)
    )

    def flushing_body(iteration, carry):
        salinity, salt_flux, cfl_max, sal_min = carry
        zv_br = nemo_source_round(zmsk * salinity)
        safe_brine = jnp.where(zs_inner != 0.0, zs_inner, jnp.ones_like(zs_inner))
        zv_br = nemo_source_round(zv_br / safe_brine)
        zv_brmin = jnp.full_like(h_i, _SAL4_BRINE_VOLUME_SENTINEL)
        for jk in range(nlay):
            zv_brmin = _minsd(zv_brmin, zv_br[..., jk])
        active = flush_cell & (iteration < nflush) & (zv_brmin >= rn_vbrc)
        for jk in range(nlay):
            cfl = nemo_source_round(
                nemo_source_round(flush_velocity[..., jk] * zdt_flush) * z1_h_flush
            )
            cfl_max_next = _maxsd(cfl_max, jnp.abs(cfl))
            brine_gradient = nemo_source_round(zs_br[..., jk + 1] - zs_br[..., jk])
            zds_raw = nemo_source_round(-cfl * brine_gradient)
            raw_salinity = nemo_source_round(salinity[..., jk] + zds_raw)
            sal_min_next = _minsd(sal_min, raw_salinity)
            upper_product = nemo_source_round(rn_sinew * sss)
            lower_bound = _minsd(upper_product, rn_simin)
            floor_delta = nemo_source_round(-salinity[..., jk] + lower_bound)
            # -O3 line-644 roles: inner dst=0 only at layer 9; outer dst=floor at 9/10.
            nonpositive = _minsd(zero, zds_raw) if jk == 8 else _minsd(zds_raw, zero)
            zds = _maxsd(floor_delta, nonpositive) if jk in (8, 9) else _maxsd(nonpositive, floor_delta)
            sal_next = nemo_source_round(salinity[..., jk] + zds)
            flux_next = nemo_source_round(
                salt_flux
                - _sal4_flux_increment(
                    rho_i, a_i, h_i, r1_nlay_i, zds, r1_Dt_ice
                )
            )
            salinity = salinity.at[..., jk].set(
                jnp.where(active, sal_next, salinity[..., jk])
            )
            salt_flux = jnp.where(active, flux_next, salt_flux)
            cfl_max = jnp.where(active, cfl_max_next, cfl_max)
            sal_min = jnp.where(active, sal_min_next, sal_min)
        return salinity, salt_flux, cfl_max, sal_min

    sz_i, sfx_bri, cfl_flush, smin_flush = jax.lax.fori_loop(
        0,
        nn_flushing,
        jax.checkpoint(flushing_body),
        (sz_i, sfx_bri, zero, zero),
    )

    for jk in range(nlay):
        zsal = sz_i[..., jk]
        lower = _maxsd(zsal, rn_simin)
        upper = nemo_source_round(rn_sinew * sss)
        bounded = _minsd(lower, upper)
        delta = nemo_source_round(bounded - zsal)
        flux_next = nemo_source_round(
            sfx_res
            - _sal4_flux_increment(
                rho_i, a_i, h_i, r1_nlay_i, delta, r1_Dt_ice
            )
        )
        sz_i = sz_i.at[..., jk].set(bounded)
        sfx_res = flux_next

    return SI3SalinityOutputs(
        sz_i, sfx_bri, sfx_res, ndrain, nflush,
        cfl_drain, cfl_flush, smin_drain, smin_flush,
    )


_SAL2_NAMELIST = (
    "nn_icesal", "rn_simin", "rn_sinew", "rn_sal_gd", "rn_time_gd",
    "rn_sal_fl", "rn_time_fl",
)

_SAL4_NAMELIST = (
    "nn_icesal", "ln_drainage", "ln_flushing", "rn_simin", "rn_sinew",
    "rn_sal_himin", "nn_liquidus", "nn_drainage", "nn_flushing",
    "rn_flushrate", "rn_vbrc", "nn_sal_scheme", "rn_alpha_RJW",
    "rn_Rc_RJW", "ln_sal_chk",
)


def si3_ice_thd_sal(*, h_i, t_su, t_i, sz_i, t_bo, sss, a_i, dh_i_sum,
                    dh_s_sum, sfx_bri, sfx_res, namelist, rDt_ice,
                    r1_Dt_ice, r1_nlay_i, rho_i, rho_s, rho_w, gravity,
                    constants: IceConstantsConfig,
                    s_i=None) -> SI3SalinityOutputs:
    """Exact-input production salinity stage (nn_icesal = 2 or 4); every physical input is required."""

    arrays = {
        "h_i": h_i, "t_su": t_su, "t_i": t_i, "sz_i": sz_i,
        "t_bo": t_bo, "sss": sss, "a_i": a_i, "dh_i_sum": dh_i_sum,
        "dh_s_sum": dh_s_sum, "sfx_bri": sfx_bri, "sfx_res": sfx_res,
    }
    arrays = {name: _required(name, value) for name, value in arrays.items()}
    scalars = {
        "rDt_ice": rDt_ice, "r1_Dt_ice": r1_Dt_ice,
        "r1_nlay_i": r1_nlay_i, "rho_i": rho_i, "rho_s": rho_s,
        "rho_w": rho_w, "gravity": gravity,
    }
    for name, value in scalars.items():
        if value is None:
            raise ValueError(f"SI3 ice_thd_sal requires {name!r}; no default")
    if constants is None:
        raise ValueError("SI3 ice_thd_sal requires constants; no default")
    if namelist is None:
        raise ValueError("SI3 ice_thd_sal requires the resolved namelist; no default")
    nml = dict(namelist)
    if nml.get("nn_icesal") == 2 and s_i is not None:
        missing = [name for name in _SAL2_NAMELIST if nml.get(name) is None]
        if missing:
            raise ValueError(f"SI3 ice_thd_sal missing namelist values {missing}")
        s_i_arr = _required("s_i", s_i)
        dt = float(rDt_ice)
        drain = jnp.where(
            arrays["t_su"] >= constants.T0,
            -jnp.maximum(s_i_arr - float(nml["rn_sal_fl"]), 0.0)
            * dt
            / float(nml["rn_time_fl"]),
            jnp.where(
                arrays["t_su"] <= arrays["t_bo"],
                -jnp.maximum(s_i_arr - float(nml["rn_sal_gd"]), 0.0)
                * dt
                / float(nml["rn_time_gd"]),
                0.0,
            ),
        )
        s_i_new = jnp.clip(
            s_i_arr + drain,
            float(nml["rn_simin"]),
            float(nml["rn_sinew"]) * arrays["sss"],
        )
        sz_i_new = option2_salinity_profile(s_i_new, arrays["sss"])
        zero_f = jnp.zeros_like(arrays["h_i"])
        zero_i = jnp.zeros_like(arrays["h_i"], dtype=jnp.int32)
        return SI3SalinityOutputs(
            sz_i_new, arrays["sfx_bri"], arrays["sfx_res"], zero_i, zero_i,
            zero_f, zero_f, s_i_new, s_i_new,
        )
    missing = [name for name in _SAL4_NAMELIST if nml.get(name) is None]
    if missing:
        raise ValueError(f"SI3 ice_thd_sal missing namelist values {missing}")
    expected = {
        "nn_icesal": 4, "ln_drainage": True, "ln_flushing": True,
        "nn_liquidus": 2, "nn_drainage": 50, "nn_flushing": 5,
        "nn_sal_scheme": 1, "ln_sal_chk": False,
    }
    wrong = {name: (nml[name], value) for name, value in expected.items() if nml[name] != value}
    if wrong:
        raise ValueError(f"SI3 ice_thd_sal transcribes only the production selectors; got {wrong}")
    if arrays["sz_i"].shape[-1] != 10:
        raise ValueError("SI3 O10 requires the production 10 ice layers")
    return _si3_ice_thd_sal4_core(
        **arrays, **scalars, rn_simin=nml["rn_simin"],
        rn_sinew=nml["rn_sinew"], rn_sal_himin=nml["rn_sal_himin"],
        nn_drainage=nml["nn_drainage"], nn_flushing=nml["nn_flushing"],
        rn_flushrate=nml["rn_flushrate"], rn_vbrc=nml["rn_vbrc"],
        rn_alpha_rjw=nml["rn_alpha_RJW"], rn_rc_rjw=nml["rn_Rc_RJW"],
        constants=constants,
    )


def si3_ice_thd_o10_category_stages(*, entry: dict, fluxes: SI3DHFluxes,
                                     sfx_bri, namelist: dict, rDt_ice,
                                     r1_Dt_ice, r1_nlay_i, rho_i, rho_s,
                                     rho_w, gravity,
                                     constants: IceConstantsConfig) -> SI3O10CategoryStages:
    """Selectable O9 + O10 category path through the second temperature call."""

    o9 = si3_ice_thd_category_stages(
        entry=entry, fluxes=fluxes, namelist=namelist,
        rDt_ice=rDt_ice, constants=constants,
    )
    sal = si3_ice_thd_sal(
        h_i=o9.dh.h_i, t_su=o9.dh.t_su, t_i=o9.t_i, sz_i=o9.dh.sz_i,
        t_bo=entry["t_bo"], sss=entry["sss"], a_i=o9.dh.a_i,
        dh_i_sum=o9.dh.dh.dh_i_sum, dh_s_sum=o9.dh.dh.dh_s_sum,
        sfx_bri=sfx_bri, sfx_res=o9.dh.fluxes.sfx_res, namelist=namelist,
        rDt_ice=rDt_ice, r1_Dt_ice=r1_Dt_ice, r1_nlay_i=r1_nlay_i,
        rho_i=rho_i, rho_s=rho_s, rho_w=rho_w, gravity=gravity,
        constants=constants, s_i=getattr(o9.dh, "s_i", None),
    )
    t_i = si3_ice_thd_temp(
        h_i=o9.dh.h_i, e_i=o9.dh.e_i, sz_i=sal.sz_i, constants=constants,
    )
    return SI3O10CategoryStages(o9, sal, t_i)


class SI3DOOutputs(NamedTuple):
    """Updated fields from ``ice_thd_do`` (O14 kind 12 -> kind 13)."""

    a_i: jax.Array
    v_i: jax.Array
    sv_i: jax.Array
    e_i: jax.Array
    szv_i: jax.Array
    at_i: jax.Array
    sfx_opw: jax.Array
    wfx_opw: jax.Array
    hfx_thd: jax.Array
    hfx_opw: jax.Array


def si3_ice_thd_do(
    *,
    a_i,
    v_i,
    sv_i,
    e_i,
    szv_i,
    qlead,
    t_bo,
    sfx_opw,
    wfx_opw,
    ht_i_new,
    fraz_frac,
    hfx_thd,
    hfx_opw,
    rn_amax_2d,
    sss_m,
    hi_max,
    nn_icesal: int,
    rn_sinew: float,
    r1_Dt_ice: float,
    r1_nlay_i: float,
    constants: IceConstantsConfig,
) -> SI3DOOutputs:
    """NEMO SI3 open-water growth, ``icethd_do.f90:177-384``.

    Array axes are ``(..., category)`` for category state and
    ``(..., ice_layer, category)`` for layer state.  Every source operation is
    materialized in the order emitted by the admitted record build.  The four
    MIN/MAX sites use that object's SSE operand roles; ``ice_thd_do`` has no
    executing SIGN intrinsic on this deck.
    """

    if nn_icesal not in (2, 4):
        raise ValueError(f"SI3 ice_thd_do transcribes nn_icesal 2 and 4 only; got {nn_icesal!r}")
    sr = nemo_source_round
    a_i = jnp.asarray(a_i)
    dtype = a_i.dtype
    v_i = jnp.asarray(v_i, dtype=dtype)
    sv_i = jnp.asarray(sv_i, dtype=dtype)
    e_i = jnp.asarray(e_i, dtype=dtype)
    szv_i = jnp.asarray(szv_i, dtype=dtype)
    qlead = jnp.asarray(qlead, dtype=dtype)
    t_bo = jnp.asarray(t_bo, dtype=dtype)
    sfx_opw = jnp.asarray(sfx_opw, dtype=dtype)
    wfx_opw = jnp.asarray(wfx_opw, dtype=dtype)
    ht_i_new = jnp.asarray(ht_i_new, dtype=dtype)
    fraz_frac = jnp.asarray(fraz_frac, dtype=dtype)
    hfx_thd = jnp.asarray(hfx_thd, dtype=dtype)
    hfx_opw = jnp.asarray(hfx_opw, dtype=dtype)
    rn_amax_2d = jnp.asarray(rn_amax_2d, dtype=dtype)
    sss_m = jnp.asarray(sss_m, dtype=dtype)
    hi_max = jnp.asarray(hi_max, dtype=dtype)
    nlay_i, jpl = e_i.shape[-2:]
    if a_i.shape[-1] != jpl or v_i.shape != a_i.shape or sv_i.shape != a_i.shape:
        raise ValueError("ice_thd_do category state shapes disagree")
    if szv_i.shape != e_i.shape or nlay_i < 1:
        raise ValueError("ice_thd_do layer state shapes disagree")
    if hi_max.shape != (jpl + 1,):
        raise ValueError(f"ice_thd_do needs hi_max shape {(jpl + 1,)}, got {hi_max.shape}")

    zero = sr(jnp.zeros_like(qlead))
    one = jnp.asarray(1.0, dtype=dtype)
    eps10 = jnp.asarray(_EPS10, dtype=dtype)
    eps20 = jnp.asarray(_EPS20, dtype=dtype)
    rhoi = jnp.asarray(constants.rho_ice, dtype=dtype)
    r1_rhoi = sr(one / rhoi)
    r1_dt = jnp.asarray(r1_Dt_ice, dtype=dtype)
    r1_nlay = jnp.asarray(r1_nlay_i, dtype=dtype)
    cooling = _ieee_lt_zero(qlead)

    # :177 SUM starts from +0 and visits categories in Fortran order.
    at_i_before = _sum0(a_i)

    # :208-219 converts extensive layer contents to intensive values.
    positive_volume = v_i > zero[..., None]
    safe_volume = jnp.where(positive_volume, v_i, one)
    e_intensive = sr(sr(e_i / safe_volume[..., None, :]) * nlay_i)
    sz_intensive = sr(sr(szv_i / safe_volume[..., None, :]) * nlay_i)
    e_intensive = jnp.where(positive_volume[..., None, :], e_intensive, zero[..., None, None])
    sz_intensive = jnp.where(
        positive_volume[..., None, :], sz_intensive, zero[..., None, None]
    )

    # :225-226, nn_icesal = 2 or 4 in the admitted identities.
    zs_newice = sr(jnp.asarray(rn_sinew, dtype=dtype) * sss_m)

    # :242-273.  Object audit: MIN keeps -epsi10 only when strictly smaller;
    # latent MAX keeps its +0 source on a signed-zero tie.
    ztmelts = sr(-jnp.asarray(constants.liquidus_slope, dtype=dtype) * zs_newice)
    t_bo_c = sr(t_bo - jnp.asarray(constants.T0, dtype=dtype))
    capped_t_bo = _minsd(-eps10, t_bo_c)
    latent_ratio = sr(ztmelts / capped_t_bo)
    latent_factor = _maxsd(sr(one - latent_ratio), zero)
    sensible_ice = sr(jnp.asarray(constants.c_ice, dtype=dtype) * sr(ztmelts - t_bo_c))
    latent_ice = sr(jnp.asarray(constants.latent_fusion, dtype=dtype) * latent_factor)
    seawater_reference = sr(jnp.asarray(constants.c_ocean, dtype=dtype) * ztmelts)
    ze_newice = sr(
        rhoi * sr(sr(sensible_ice + latent_ice) - seawater_reference)
    )
    z_ei = -sr(ze_newice * r1_rhoi)
    z_ew = sr(jnp.asarray(constants.c_ocean, dtype=dtype) * t_bo_c)
    z_de = sr(z_ei - z_ew)
    safe_de = jnp.where(cooling, z_de, one)
    # The -O3 record object cancels the two source negations before the multiply.
    qlead_over_de = sr(qlead / safe_de)
    zfmdt = -qlead_over_de
    zv_newice_initial = sr(qlead_over_de * r1_rhoi)

    hfx_thd_new = sr(hfx_thd + sr(sr(zfmdt * z_ew) * r1_dt))
    hfx_opw_new = sr(hfx_opw - sr(sr(zfmdt * z_de) * r1_dt))
    new_ice_mass = sr(rhoi * zv_newice_initial)
    wfx_opw_new = sr(wfx_opw - sr(new_ice_mass * r1_dt))
    sfx_opw_new = sr(sfx_opw - sr(sr(new_ice_mass * zs_newice) * r1_dt))

    # :275-283, frazil collection at the existing ice bottom.
    has_ice = at_i_before > zero
    zv_frazb = jnp.where(has_ice, sr(fraz_frac * zv_newice_initial), zero)
    zv_newice = jnp.where(
        has_ice,
        sr(sr(one - fraz_frac) * zv_newice_initial),
        zv_newice_initial,
    )
    safe_h_new = jnp.where(cooling, ht_i_new, one)
    za_newice = sr(zv_newice / safe_h_new)

    # :288-299.  The record object emits MAX(candidate,+0): +0 survives ties.
    available_area = _maxsd(sr(rn_amax_2d - at_i_before), zero)
    excessive = za_newice > available_area
    zda_res = jnp.where(excessive, sr(za_newice - available_area), zero)
    zdv_res = jnp.where(excessive, sr(zda_res * ht_i_new), zero)
    za_newice = jnp.where(excessive, _maxsd(sr(za_newice - zda_res), zero), za_newice)
    zv_newice = jnp.where(excessive, _maxsd(sr(zv_newice - zdv_res), zero), zv_newice)

    # :301-320, one category selected by the HFN interval (lower, upper].
    selected = (ht_i_new[..., None] > hi_max[:-1]) & (
        ht_i_new[..., None] <= hi_max[1:]
    )
    a_lateral = jnp.where(selected, sr(a_i + za_newice[..., None]), a_i)
    v_lateral = jnp.where(selected, sr(v_i + zv_newice[..., None]), v_i)
    at_i_lateral = _sum0(a_lateral)
    mix_divisor = _maxsd(eps20, v_lateral)
    e_numerator = sr(
        sr(ze_newice[..., None, None] * zv_newice[..., None, None])
        + sr(e_intensive * v_i[..., None, :])
    )
    sz_numerator = sr(
        sr(zs_newice[..., None, None] * zv_newice[..., None, None])
        + sr(sz_intensive * v_i[..., None, :])
    )
    e_mixed = sr(e_numerator / mix_divisor[..., None, :])
    sz_mixed = sr(sz_numerator / mix_divisor[..., None, :])
    # :311 tests the saved pre-growth category area (``za_b``), not volume.
    old_selected_ice = a_i > zero[..., None]
    e_selected = jnp.where(old_selected_ice[..., None, :], e_mixed, ze_newice[..., None, None])
    sz_selected = jnp.where(
        old_selected_ice[..., None, :], sz_mixed, zs_newice[..., None, None]
    )
    e_intensive = jnp.where(selected[..., None, :], e_selected, e_intensive)
    sz_intensive = jnp.where(selected[..., None, :], sz_selected, sz_intensive)

    # :322-356, distribute residual/frazil volume then remap every category.
    enough_area = at_i_lateral >= eps20
    residual_and_frazil = sr(zdv_res + zv_frazb)
    a_categories = []
    v_categories = []
    sv_categories = []
    e_categories = []
    szv_categories = []
    for jl in range(jpl):
        area = jnp.where(enough_area, a_lateral[..., jl], zero)
        numerator = sr(residual_and_frazil * a_lateral[..., jl])
        denominator = _maxsd(at_i_lateral, eps20)
        zv_newfra = jnp.where(enough_area, sr(numerator / denominator), zero)
        v_before = v_lateral[..., jl]
        v_after = sr(v_before + zv_newfra)

        layer_h = sr(v_before * r1_nlay)
        h_old = [layer_h for _ in range(nlay_i)] + [zv_newfra]
        e_old = [
            sr(sr(e_intensive[..., jk, jl] * v_before) * r1_nlay)
            for jk in range(nlay_i)
        ] + [sr(ze_newice * zv_newfra)]
        sz_old = [
            sr(sr(sz_intensive[..., jk, jl] * v_before) * r1_nlay)
            for jk in range(nlay_i)
        ] + [sr(zs_newice * zv_newfra)]
        e_remapped = _cumulative_remap(h_old, e_old, nlay_i, sr)
        sz_remapped = (
            _cumulative_remap(h_old, sz_old, nlay_i, sr)
            if nn_icesal == 4
            else sz_intensive[..., :, jl]
        )

        volume_change = sr(v_after - v_i[..., jl])
        sv_after = sr(sv_i[..., jl] + sr(zs_newice * volume_change))
        e_extensive = sr(sr(e_remapped * v_after[..., None]) * r1_nlay)
        szv_extensive = sr(sr(sz_remapped * v_after[..., None]) * r1_nlay)
        a_categories.append(area)
        v_categories.append(v_after)
        sv_categories.append(sv_after)
        e_categories.append(e_extensive)
        szv_categories.append(szv_extensive)

    a_updated = jnp.stack(a_categories, axis=-1)
    v_updated = jnp.stack(v_categories, axis=-1)
    sv_updated = jnp.stack(sv_categories, axis=-1)
    e_updated = jnp.stack(e_categories, axis=-1)
    szv_updated = jnp.stack(szv_categories, axis=-1)
    category_mask = cooling[..., None]
    layer_mask = cooling[..., None, None]
    a_out = jnp.where(category_mask, a_updated, a_i)
    v_out = jnp.where(category_mask, v_updated, v_i)
    sv_out = jnp.where(category_mask, sv_updated, sv_i)
    e_out = jnp.where(layer_mask, e_updated, e_i)
    szv_out = jnp.where(layer_mask, szv_updated, szv_i)
    # :177 executes over every interior cell.  The gathered ``at_i_1d`` is
    # changed below, but :366-375 does not scatter it back to the 2-D field.
    at_i_out = at_i_before
    return SI3DOOutputs(
        a_i=a_out,
        v_i=v_out,
        sv_i=sv_out,
        e_i=e_out,
        szv_i=szv_out,
        at_i=at_i_out,
        sfx_opw=jnp.where(cooling, sfx_opw_new, sfx_opw),
        wfx_opw=jnp.where(cooling, wfx_opw_new, wfx_opw),
        hfx_thd=jnp.where(cooling, hfx_thd_new, hfx_thd),
        hfx_opw=jnp.where(cooling, hfx_opw_new, hfx_opw),
    )


def si3_column_step_arrays(state: SI3ColumnArrays,
                           forcing: SI3SurfaceForcing,
                           dt: float,
                           constants: IceConstantsConfig, *,
                           si3=None,
                           namelist: dict | None = None,
                           at_i=None,
                           t_i_entry=None,
                           t_s_entry=None,
                           hfx_err_dif=None,
                           hfx_dif=None,
                           fluxes: SI3DHFluxes | None = None,
                           sfx_bri=None,
                           sfx_lam=None,
                           hfx_thd=None,
                           wfx_lam=None,
                           _return_stage_outputs: bool = False, _zdf_dh_override=None) -> SI3StepTrace:
    """Execute the selected `ice_thd` chain and retain every oracle boundary."""

    e_ice = jnp.asarray(state.e_ice)
    dtype = e_ice.dtype
    h_ice = jnp.asarray(state.h_ice, dtype=dtype)
    h_snow = jnp.asarray(state.h_snow, dtype=dtype)
    S_layers = jnp.asarray(state.S_layers, dtype=dtype)
    e_snow = jnp.asarray(state.e_snow, dtype=dtype)
    if t_i_entry is None:
        T_i_old = ice_temperature_from_enthalpy(e_ice, S_layers, constants)
    else:
        T_i_old = jnp.asarray(t_i_entry, dtype=dtype)
    if t_s_entry is None:
        T_s_old = snow_temperature_from_enthalpy(e_snow, constants)
        # NEMO's equivalent-state conversion resets absent snow to T0 before ZDF (`icevar.F90:404-416`).
        T_s_old = jnp.where((h_snow > 0.0)[..., None], T_s_old, constants.T0)
    else:
        T_s_old = jnp.asarray(t_s_entry, dtype=dtype)
    zero = jnp.zeros_like(h_ice)
    if si3 is None or getattr(si3, "salinity_scheme", None) is None: raise ValueError("si3_column_step_arrays requires an explicit SI3ThermoConfig (si3); no default")
    nn_icesal, fluxes_in = int(si3.salinity_scheme), SI3DHFluxes(*([zero] * len(SI3DHFluxes._fields))) if fluxes is None else fluxes
    nml = {
        "ln_icedH": True,
        "rn_cnd_s": float(constants.k_snow),
        "rn_kappa_i": _ICE_EXTINCTION,
        "rn_kappa_s": _SNOW_EXTINCTION,
        "nn_qtrice": 0,
        "nn_snwfra": 2,
        "ln_cndi_P07": True,
        "ln_cndflx": False,
        "ln_cndemulate": False,
        "ln_virtual_itd": False,
        "ln_zdf_chkcvg": False,
        "rn_snwblow": _SNOW_BLOW_FRACTION,
        "rn_sinew": _NEW_ICE_SALINITY_FRACTION,
        "rn_hinew": _NEW_ICE_THICKNESS,
        "ln_frazil": False,
        "nn_icesal": nn_icesal,
        "rn_simin": _MINIMUM_ICE_SALINITY,
        "rn_sal_gd": _DRAINAGE_TARGET,
        "rn_time_gd": _DRAINAGE_TIMESCALE,
        "rn_sal_fl": _FLUSHING_TARGET,
        "rn_time_fl": _FLUSHING_TIMESCALE,
    }
    if namelist is not None:
        nml.update(namelist)
    a_i_arr = jnp.asarray(state.concentration, dtype=dtype)
    at_i_arr = a_i_arr if at_i is None else jnp.asarray(at_i, dtype=dtype)
    t_bo_arr = jnp.asarray(forcing.t_bottom, dtype=dtype)
    if nml["nn_icesal"] == 4 and forcing.sst_m is not None:
        sst_dh = jnp.asarray(forcing.sst_m, dtype=dtype)
    else:
        # C1D has no SST in its forcing record; snow-ice water enthalpy uses t_bo - rt0.
        sst_dh = t_bo_arr - constants.T0
    entry = {
        "h_i": h_ice,
        "h_s": h_snow,
        "t_su": jnp.asarray(state.T_surface, dtype=dtype),
        "t_i": T_i_old,
        "t_s": T_s_old,
        "e_i": e_ice,
        "e_s": e_snow,
        "sz_i": S_layers,
        "s_i": jnp.asarray(state.S_bulk, dtype=dtype),
        "a_i": a_i_arr,
        "at_i": at_i_arr,
        "t_bo": t_bo_arr,
        "sss": jnp.asarray(forcing.sss, dtype=dtype),
        "sst": sst_dh,
        "qns_ice": jnp.asarray(forcing.qns_ice, dtype=dtype),
        "qsr_ice": jnp.asarray(forcing.qsr_ice, dtype=dtype),
        "dqns_ice": jnp.asarray(forcing.dqns_ice, dtype=dtype),
        "qtr_ice_top": jnp.asarray(forcing.qtr_ice_top, dtype=dtype),
        "qsb_ice_bot": jnp.asarray(forcing.qsb_ice_bottom, dtype=dtype),
        "fhld": jnp.asarray(forcing.fhld, dtype=dtype),
        "frq_m": jnp.asarray(forcing.frq_m, dtype=dtype),
        "sprecip": jnp.asarray(forcing.snow_precipitation, dtype=dtype),
        "qprec_ice": jnp.asarray(forcing.qprec_ice, dtype=dtype),
        "evap_ice": jnp.asarray(forcing.evaporation, dtype=dtype),
        "hfx_err_dif": zero if hfx_err_dif is None else jnp.asarray(hfx_err_dif, dtype=dtype),
        "hfx_dif": zero if hfx_dif is None else jnp.asarray(hfx_dif, dtype=dtype),
    }
    rDt_ice = float(dt)
    o10 = None if _zdf_dh_override else si3_ice_thd_o10_category_stages(
        entry=entry,
        fluxes=fluxes_in,
        sfx_bri=zero if sfx_bri is None else jnp.asarray(sfx_bri, dtype=dtype),
        namelist=nml,
        rDt_ice=rDt_ice,
        r1_Dt_ice=1.0 / rDt_ice,
        r1_nlay_i=1.0 / S_layers.shape[-1],
        rho_i=constants.rho_ice,
        rho_s=constants.rho_snow,
        rho_w=canonical_constants.rho_water,
        gravity=canonical_constants.g_nemo,
        constants=constants,
    )
    post_zdf = _zdf_dh_override[0] if _zdf_dh_override else state._replace(
        T_surface=o10.o9.zdf.t_su,
        e_ice=o10.o9.zdf.e_i,
        e_snow=o10.o9.zdf.e_s,
    )
    post_dh = _zdf_dh_override[1] if _zdf_dh_override else SI3ColumnArrays(
        o10.o9.dh.a_i,
        o10.o9.dh.h_i,
        o10.o9.dh.h_s,
        o10.o9.dh.t_su,
        o10.o9.dh.e_i,
        o10.o9.dh.e_s,
        o10.o9.dh.s_i,
        o10.o9.dh.sz_i,
        state.age_volume,
    )
    # `ice_thd_temp` is diagnostic because enthalpy is prognostic (:221-247).
    post_temp1 = post_dh
    S_bulk = _zdf_dh_override[2] if _zdf_dh_override else (o10.sal.smin_drain if nml["nn_icesal"] == 2 else post_dh.S_bulk)
    post_sal = post_temp1._replace(S_bulk=S_bulk, S_layers=_zdf_dh_override[3] if _zdf_dh_override else o10.sal.sz_i)
    post_temp2 = post_sal
    if _return_stage_outputs:
        post_da, da_out = _si3_column_thd_da(
            state, post_temp2, forcing, si3, dt, constants,
            at_i=at_i_arr, sfx_lam=sfx_lam,
            hfx_thd=o10.o9.dh.fluxes.hfx_thd if hfx_thd is None else hfx_thd,
            wfx_lam=wfx_lam, _return_da_outputs=True,
        )
    else:
        post_da = _si3_column_thd_da(state, post_temp2, forcing, si3, dt, constants)
        da_out = None
    # `ice_thd_frazil` supplies ht_i_new (O7_REF ppsrc `icethd.f90:174`).
    frazil = si3_ice_thd_frazil(
        forcing.qlead,
        rn_hinew=float(nml["rn_hinew"]),
        ln_frazil=bool(nml["ln_frazil"]),
    )
    a_do = post_da.concentration[..., None]
    v_do = (post_da.h_ice * post_da.concentration)[..., None]
    sv_do = (post_da.S_bulk * v_do[..., 0])[..., None]
    e_do = post_da.e_ice[..., :, None] * v_do[..., None, :] / post_da.e_ice.shape[-1]
    szv_do = post_da.S_layers[..., :, None] * v_do[..., None, :] / post_da.S_layers.shape[-1]
    do_out = si3_ice_thd_do(
        a_i=a_do,
        v_i=v_do,
        sv_i=sv_do,
        e_i=e_do,
        szv_i=szv_do,
        qlead=forcing.qlead,
        t_bo=forcing.t_bottom,
        sfx_opw=zero,
        wfx_opw=zero,
        ht_i_new=frazil.ht_i_new,
        fraz_frac=frazil.fraz_frac,
        hfx_thd=zero,
        hfx_opw=zero,
        rn_amax_2d=jnp.full_like(zero, _MAXIMUM_ICE_CONCENTRATION),
        sss_m=forcing.sss,
        hi_max=jnp.asarray([0.0, float("inf")], dtype=dtype),
        nn_icesal=nn_icesal,
        rn_sinew=float(nml["rn_sinew"]),
        r1_Dt_ice=1.0 / rDt_ice,
        r1_nlay_i=1.0 / post_da.e_ice.shape[-1],
        constants=constants,
    )
    a_total = do_out.a_i[..., 0]
    v_total = do_out.v_i[..., 0]
    h_total = jnp.where(a_total > _EPS10, v_total / a_total, 0.0)
    # C1D retains its certified prognostic set; O14's layer/salt/flux outputs
    # are carried by the full five-category production stage above.
    post_do = post_da._replace(concentration=a_total, h_ice=h_total)
    exit_state = post_do._replace(age_volume=post_do.age_volume + a_total * dt)
    trace = SI3StepTrace(state, post_zdf, post_dh, post_temp1, post_sal,
                         post_temp2, post_do, exit_state)
    if _return_stage_outputs:
        return trace, o10, da_out, frazil
    return trace


class SI3Frazil(NamedTuple):
    """``ice_thd_frazil`` outputs: new-ice collection thickness and frazil fraction."""

    ht_i_new: jax.Array
    fraz_frac: jax.Array


_SIGN_BIT_SHIFT = 63
_NAN_SHIFTED_FLOOR = 0xFFE0000000000000  # (exponent all ones) << 1: above it is NaN


def _ieee_lt_zero(value):
    """``value < 0._wp`` as gfortran compiles it (``comisd``/``ja``), on the bit pattern.

    True for every negative non-NaN, including a negative subnormal that XLA CPU
    ``jit`` would flush to ``-0`` before a float compare; ``-0`` and NaN are false.
    """
    bits = jax.lax.bitcast_convert_type(jnp.asarray(value, dtype=jnp.float64), jnp.uint64)
    shifted = bits << jnp.uint64(1)
    return ((bits >> jnp.uint64(_SIGN_BIT_SHIFT)) == jnp.uint64(1)) & (
        shifted != jnp.uint64(0)) & (shifted <= jnp.uint64(_NAN_SHIFTED_FLOOR))


def si3_ice_thd_frazil(qlead, *, rn_hinew, ln_frazil) -> SI3Frazil:
    """``ice_thd_frazil`` with collection off (``icethd_do.f90:393-483``, ``ORCA2_SAS_ICE_O7_REF``).

    Executes ``fraz_frac(:,:) = 0._wp`` (``:419``) and
    ``WHERE( qlead(:,:) < 0._wp ) ht_i_new = rn_hinew ELSEWHERE ht_i_new = 0._wp`` (``:422-426``).
    Shape-agnostic: the routine has no category axis on this arm (``hi_max(jpl)`` is
    read only by the collection arm, ``:428-482``), so C1D and jpl=5 ORCA2 share it.
    The collection arm (``ln_frazil = .true.``) is refused, not transcribed.
    """
    if qlead is None or rn_hinew is None:
        raise ValueError("ice_thd_frazil needs qlead and rn_hinew; no defaults")
    if not isinstance(ln_frazil, bool):
        raise ValueError(f"ln_frazil must be the namelist logical, got {ln_frazil!r}")
    if ln_frazil:
        raise ValueError("ice_thd_frazil collection arm (ln_frazil=T) is not transcribed")
    qlead = jnp.asarray(qlead)
    zero = jnp.zeros_like(qlead)
    ht_i_new = jnp.where(_ieee_lt_zero(qlead), jnp.asarray(rn_hinew, qlead.dtype), zero)
    return SI3Frazil(ht_i_new=ht_i_new, fraz_frac=zero)


# Public name of the bit-pattern ``< 0`` above, shared with the other SI3 statement
# modules (the definition stays where the O8 receipt cites it).
ieee_lt_zero = _ieee_lt_zero


__all__ = (
    "freezing_temperature",
    "ice_enthalpy",
    "ice_thermal_conductivity",
    "ice_specific_heat",
    "bitz_lipscomb_conduction_step",
    "SI3SurfaceForcing",
    "SI3ZDFResult",
    "SI3ColumnArrays",
    "SI3StepTrace",
    "option2_salinity_profile",
    "ice_enthalpy_from_temperature",
    "ice_temperature_from_enthalpy",
    "snow_enthalpy_from_temperature",
    "snow_temperature_from_enthalpy",
    "p07_conductivity",
    "si3_column_step_arrays",
    "SI3Frazil",
    "si3_ice_thd_frazil",
    "si3_ice_thd_do",
    "SI3DOOutputs",
    "ieee_lt_zero",
    "SI3SalinityOutputs",
    "SI3O10CategoryStages",
    "si3_ice_thd_sal",
    "si3_ice_thd_o10_category_stages",
)


def _si3_column_thd_da(entry, post_temp2, forcing, si3, dt, constants, *,
                       at_i=None, sfx_lam=None, hfx_thd=None, wfx_lam=None,
                       _return_da_outputs: bool = False):
    """``IF( ln_icedA ) CALL ice_thd_da`` on the C1D column (O7_REF ppsrc ``icethd.f90:210``).

    ``si3`` is ``SeaIceConfig.si3`` (required; ``lateral_melt=False`` is
    ``ln_icedA = .false.`` and returns ``post_temp2`` itself).  NEMO's inputs:
    ``at_i_1d`` is the gather-time total (one category: entry ``a_i``), and only
    cells gathered at entry (``a_i > epsi10``, ``icethd.f90:181``) change.
    Flux accumulators start at +0 unless passed explicitly.
    """
    if si3 is None:
        raise ValueError("_si3_column_thd_da requires an explicit SI3ThermoConfig (si3); no default")
    if not si3.lateral_melt:
        if si3.namthd_da is not None:
            raise ValueError("si3.namthd_da is set but lateral_melt is False (ln_icedA)")
        if _return_da_outputs:
            return post_temp2, None
        return post_temp2
    if si3.namthd_da is None:
        raise ValueError("lateral_melt=True (ln_icedA) needs si3.namthd_da; no defaults")
    if forcing.sst_m is None:
        raise ValueError("lateral_melt=True (ln_icedA) needs SI3SurfaceForcing.sst_m")
    from legoesm.ice.si3_thd_da import si3_ice_thd_da  # deferred: it imports this module

    zero = jnp.zeros_like(post_temp2.concentration)
    da = si3_ice_thd_da(
        at_i=entry.concentration if at_i is None else at_i,
        a_i=post_temp2.concentration,
        h_i=post_temp2.h_ice, h_s=post_temp2.h_snow,
        e_i=post_temp2.e_ice, e_s=post_temp2.e_snow,
        s_i=post_temp2.S_bulk, sz_i=post_temp2.S_layers,
        sst=forcing.sst_m, t_bo=forcing.t_bottom,
        sfx_lam=zero if sfx_lam is None else sfx_lam,
        hfx_thd=zero if hfx_thd is None else hfx_thd,
        wfx_lam=zero if wfx_lam is None else wfx_lam,
        rn_beta=si3.namthd_da.rn_beta, rn_dmin=si3.namthd_da.rn_dmin,
        rDt_ice=dt, nn_icesal=si3.salinity_scheme, constants=constants,
    )
    gathered = entry.concentration > _EPS10
    post_da = post_temp2._replace(
        concentration=jnp.where(gathered, da.a_i, post_temp2.concentration),
        h_ice=jnp.where(gathered, da.h_i, post_temp2.h_ice),
        h_snow=jnp.where(gathered, da.h_s, post_temp2.h_snow),
    )
    if _return_da_outputs:
        return post_da, da
    return post_da
