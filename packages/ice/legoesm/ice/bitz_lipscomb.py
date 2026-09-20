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
_NEW_ICE_THICKNESS = 0.05
_MAXIMUM_ICE_CONCENTRATION = 0.99999


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
        + canonical_constants.L_f * (1.0 - Tm / Tc)
        - canonical_constants.c_pw * Tm
    )


def option2_salinity_profile(S_bulk, sss):
    """SI3 option-2 diagnostic linear profile (`icevar.F90:537-598`)."""

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
    z = jnp.asarray(_OPTION2_LAYER_SHAPE, dtype=S_bulk.dtype)
    return S_bulk[..., None] * (alpha[..., None] * z + (1.0 - alpha[..., None]))


def ice_enthalpy_from_temperature(T_K, S, constants: IceConstantsConfig, *,
                                  _nemo_order: bool = True):
    """NEMO positive energy-of-melting `icevar.F90:938-946` [J m-3]."""

    T_m = -constants.liquidus_slope * S
    if not _nemo_order:
        T_c = jnp.minimum(T_K - constants.T0, -_EPS10)
        T_c = jnp.minimum(T_c, T_m)
        return constants.rho_ice * (
            constants.c_ice * (T_m - T_c)
            + constants.latent_fusion * jnp.maximum(0.0, 1.0 - T_m / T_c)
            - constants.c_ocean * T_m
        )
    # Preserve the assignment order in `icevar.F90:938-946`: first clip the
    # Kelvin temperature, then form Celsius, and finally evaluate the three
    # parenthesised energy terms.  This is the selected SI3 identity's
    # operation order, not a user-selectable numerical variant.
    T_clipped = jnp.minimum(T_K, T_m + constants.T0)
    T_c = T_clipped - constants.T0
    sensible = constants.c_ice * (T_m - T_c)
    liquid = jnp.maximum(
        0.0,
        1.0 - T_m / jnp.minimum(T_c, -_EPS10),
    )
    latent = constants.latent_fusion * liquid
    ocean = constants.c_ocean * T_m
    return constants.rho_ice * ((sensible + latent) - ocean)


def ice_temperature_from_enthalpy(e, S, constants: IceConstantsConfig | None = None,
                                  *, _nemo_order: bool = True):
    """Invert enthalpy [K] in the selected NEMO or legacy convention.

    Supplying ``constants`` selects NEMO's positive energy-of-melting
    convention (`icethd.F90:221-247`).  Omitting it preserves the public API
    of the promoted prototype, whose enthalpy is negative for cold ice.
    """

    if constants is None:
        Tm = freezing_temperature(S)
        q_spec = e / canonical_constants.rho_ice
        b = ((canonical_constants.c_pw - canonical_constants.c_pi) * Tm
             - canonical_constants.L_f - q_spec)
        c = canonical_constants.L_f * Tm
        disc = jnp.maximum(
            b * b - 4.0 * canonical_constants.c_pi * c,
            _LEGACY_DISC_FLOOR,
        )
        T_c = (-b - jnp.sqrt(disc)) / (2.0 * canonical_constants.c_pi)
        return T_c + canonical_constants.T_freeze

    T_m = -constants.liquidus_slope * S
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
    # Written order of `icethd.F90:233-243`.  NEMO multiplies by its
    # precomputed reciprocals and applies `* 0.5 * r1_rcpi`; the algebraically
    # equivalent combined division loses observable ULPs in the oracle gate.
    inverse_rho_ice = 1.0 / constants.rho_ice
    inverse_c_ice = 1.0 / constants.c_ice
    b = (
        (constants.c_ocean - constants.c_ice) * T_m
        + e * inverse_rho_ice
        - constants.latent_fusion
    )
    disc = jnp.maximum(
        b * b - 4.0 * constants.c_ice * constants.latent_fusion * T_m,
        0.0,
    )
    return constants.T0 - (b + jnp.sqrt(disc)) * 0.5 * inverse_c_ice


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
        + canonical_constants.L_f * canonical_constants.mu_ice_freeze * S
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


def snow_enthalpy_from_temperature(T_K, constants: IceConstantsConfig):
    """Snow energy of melting, `icevar.F90:949-952` [J m-3]."""

    return constants.rho_snow * (
        constants.c_ice * (constants.T0 - T_K) + constants.latent_fusion
    )


def snow_temperature_from_enthalpy(e, constants: IceConstantsConfig, *,
                                   _nemo_order: bool = True):
    """Invert snow enthalpy with NEMO's global-to-equivalent bounds.

    `icevar.F90:404-416` reconstructs this diagnostic immediately before each
    `ice_thd` call and bounds it to ``[rt0-100, rt0]``.  The same algebra also
    appears in `icethd_dh.F90:498-503` after snow remapping.
    """
    inverse_rho_snow = 1.0 / constants.rho_snow
    inverse_c_ice = 1.0 / constants.c_ice
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


def p07_conductivity(T_K, S, constants: IceConstantsConfig):
    """Pringle et al. selection, `icethd_zdf_bl99.F90:261-275`."""

    T_c = jnp.minimum(T_K - constants.T0, -_EPS10)
    return jnp.maximum(
        _P07_CONDUCTIVITY_MIN,
        constants.k_ice
        + _P07_BETA_SALINITY * S / T_c
        - _P07_BETA_TEMPERATURE * T_c,
    )


def _radiation(h_i, h_s, qtr_top, T_surface):
    """SSL transmission/absorption, `icethd_zdf_bl99.F90:143-230`."""

    del T_surface  # nn_qtrice=0 in the resolved deck
    dtype = h_i.dtype
    js = jnp.arange(1, 4, dtype=dtype)
    ji = jnp.arange(1, 4, dtype=dtype)
    dzs = jnp.maximum(h_s, _H_MIN)[..., None] / 3.0
    dzi = jnp.maximum(h_i, _H_MIN)[..., None] / 3.0
    snow_fraction = h_s / (
        h_s + _SNOW_FRACTION_DEPTH
    )  # nn_snwfra=2, icevar.F90:1598-1599
    tr_s = qtr_top[..., None] * jnp.exp(
        -_SNOW_EXTINCTION * jnp.maximum(0.0, dzs * js - _SNOW_SSL_DEPTH)
    )
    tr_s0 = jnp.concatenate([qtr_top[..., None], tr_s], axis=-1)
    ab_s = tr_s0[..., :-1] - tr_s0[..., 1:]
    through_snow = tr_s[..., -1]
    top_i = snow_fraction * through_snow + (1.0 - snow_fraction) * qtr_top
    tr_i = (
        snow_fraction[..., None] * through_snow[..., None]
        * jnp.exp(-jnp.maximum(0.0, dzi * ji - _H_MIN))
        + (1.0 - snow_fraction)[..., None] * qtr_top[..., None]
        * jnp.exp(-jnp.maximum(0.0, dzi * ji - _ICE_SSL_DEPTH))
    )
    tr_i0 = jnp.concatenate([top_i[..., None], tr_i], axis=-1)
    ab_i = tr_i0[..., :-1] - tr_i0[..., 1:]
    return ab_s, ab_i, tr_i[..., -1]


def _si3_zdf_bl99_step(
    e_ice,
    e_snow,
    S_layers,
    h_ice,
    h_snow,
    T_surface,
    forcing: SI3SurfaceForcing,
    dt: float,
    constants: IceConstantsConfig,
    *,
    _maximum_iterations: int = _MAX_ITERATIONS,
    _nemo_branch_ranges: bool = True,
    _nemo_snow_temperature_bounds: bool = True,
    _nemo_eos_order: bool = True,
) -> SI3ZDFResult:
    """Iterative 3+3-layer BL99/P07 solve for the resolved ORCA1 arm.

    The fixed 200-iteration `fori_loop` freezes each column after the same
    ``1e-4 K`` convergence test as NEMO.  A static loop keeps reverse-mode AD
    available, unlike a data-dependent ``while_loop``.  The private iteration
    bound exists only to expose a single Picard iterate to the oracle gate.
    """

    e_ice = jnp.asarray(e_ice)
    dtype = e_ice.dtype
    h_ice = jnp.asarray(h_ice, dtype=dtype)
    h_snow = jnp.asarray(h_snow, dtype=dtype)
    T_surface_input = jnp.asarray(T_surface, dtype=dtype)
    S_layers = jnp.asarray(S_layers, dtype=dtype)
    T_i_old = ice_temperature_from_enthalpy(
        e_ice, S_layers, constants, _nemo_order=_nemo_eos_order
    )
    e_snow = jnp.asarray(e_snow, dtype=dtype)
    T_s_old = snow_temperature_from_enthalpy(
        e_snow, constants, _nemo_order=_nemo_eos_order
    )
    if not _nemo_snow_temperature_bounds:
        # Gate-only one-variable arm reproducing the pre-Phase-4 operand.
        inverse_rho_snow = 1.0 / constants.rho_snow
        inverse_c_ice = 1.0 / constants.c_ice
        T_s_old = constants.T0 + (
            -e_snow * inverse_rho_snow * inverse_c_ice
            + constants.latent_fusion * inverse_c_ice
        )
    # NEMO's equivalent-state conversion resets absent snow to T0 before ZDF
    # (`icevar.F90:404-416`).  The stored absent-layer enthalpy is therefore
    # not a temperature carrier when h_s == 0.
    T_s_old = jnp.where((h_snow > 0.0)[..., None], T_s_old, constants.T0)
    T_surface = jnp.minimum(T_surface_input, constants.T0 - _T_SURFACE_EPS)
    qns_initial = jnp.asarray(forcing.qns_ice, dtype=dtype)
    dqns = jnp.asarray(forcing.dqns_ice, dtype=dtype)
    qsr = jnp.asarray(forcing.qsr_ice, dtype=dtype)
    qtr_top = jnp.asarray(forcing.qtr_ice_top, dtype=dtype)
    T_bottom = jnp.asarray(forcing.t_bottom, dtype=dtype)
    ab_s, ab_i, qtr_bottom = _radiation(h_ice, h_snow, qtr_top, T_surface)

    dz_i = jnp.maximum(h_ice, _H_MIN) / 3.0
    dz_s = jnp.maximum(h_snow, _H_MIN) / 3.0
    snow_present = h_snow > 0.0

    def body(_iteration, carry):
        Tsu, Tsu_previous, Ti, Ts, qns, converged, iterations = carry
        active = ~converged

        # P07 conductivity at the four ice interfaces (top, 2 interior, base).
        Ti_iface = 0.5 * (Ti[..., :-1] + Ti[..., 1:])
        S_iface = 0.5 * (S_layers[..., :-1] + S_layers[..., 1:])
        k0 = p07_conductivity(Ti[..., 0], S_layers[..., 0], constants)
        km = p07_conductivity(Ti_iface, S_iface, constants)
        kb = p07_conductivity(T_bottom, S_layers[..., -1], constants)
        k_i = jnp.concatenate([k0[..., None], km, kb[..., None]], axis=-1) / dz_i[..., None]
        k_s_bulk = jnp.full_like(k_i, constants.k_snow) / dz_s[..., None]
        k_si = constants.k_snow * (k_i[..., 0] * dz_i) / (
            0.5 * ((k_i[..., 0] * dz_i) * dz_s + constants.k_snow * dz_i)
        )
        k_s = k_s_bulk.at[..., -1].set(k_si)
        k_i = k_i.at[..., 0].set(jnp.where(snow_present, k_si, k_i[..., 0]))

        ci = constants.c_ice + _BRINE_HEAT_CAPACITY_FACTOR * S_layers / jnp.maximum(
            (Ti - constants.T0) * (T_i_old - constants.T0), _EPS10
        )
        eta_i = dt / (constants.rho_ice * dz_i[..., None] * ci)
        eta_s = dt / (constants.rho_snow * dz_s[..., None] * constants.c_ice)

        qns_new = qns + dqns * (Tsu - Tsu_previous)
        fnet = qsr - qtr_top + qns_new

        shape = Ti.shape[:-1] + (7,)
        a = jnp.zeros(shape, dtype=dtype)
        b = jnp.zeros(shape, dtype=dtype)
        c = jnp.zeros(shape, dtype=dtype)
        d = jnp.zeros(shape, dtype=dtype)

        # Surface plus snow rows, NEMO jm=1..4 (:438-448,:419-425).
        b = b.at[..., 0].set(dqns - 2.0 * k_s[..., 0])
        c = c.at[..., 0].set(2.0 * k_s[..., 0])
        d = d.at[..., 0].set(dqns * Tsu - fnet)
        a = a.at[..., 1].set(-eta_s[..., 0] * 2.0 * k_s[..., 0])
        b = b.at[..., 1].set(1.0 + eta_s[..., 0] * (k_s[..., 1] + 2.0 * k_s[..., 0]))
        c = c.at[..., 1].set(-eta_s[..., 0] * k_s[..., 1])
        d = d.at[..., 1].set(T_s_old[..., 0] + eta_s[..., 0] * ab_s[..., 0])
        for k in (1, 2):
            row = k + 1
            a = a.at[..., row].set(-eta_s[..., k] * k_s[..., k])
            b = b.at[..., row].set(1.0 + eta_s[..., k] * (k_s[..., k] + k_s[..., k + 1]))
            c = c.at[..., row].set(-eta_s[..., k] * k_s[..., k + 1])
            d = d.at[..., row].set(T_s_old[..., k] + eta_s[..., k] * ab_s[..., k])

        # Ice rows, NEMO jm=5..7 (:399-413).
        for k in (0, 1):
            row = k + 4
            a = a.at[..., row].set(-eta_i[..., k] * k_i[..., k])
            b = b.at[..., row].set(1.0 + eta_i[..., k] * (k_i[..., k] + k_i[..., k + 1]))
            c = c.at[..., row].set(-eta_i[..., k] * k_i[..., k + 1])
            d = d.at[..., row].set(T_i_old[..., k] + eta_i[..., k] * ab_i[..., k])
        a = a.at[..., 6].set(-eta_i[..., 2] * k_i[..., 2])
        b = b.at[..., 6].set(1.0 + eta_i[..., 2] * (k_i[..., 2] + 2.0 * k_i[..., 3]))
        d = d.at[..., 6].set(
            T_i_old[..., 2]
            + eta_i[..., 2] * (ab_i[..., 2] + 2.0 * k_i[..., 3] * T_bottom)
        )

        # BL99 changes the solved row range at two physical boundaries
        # (`icethd_zdf_bl99.F90:433-513`): snow-free columns put the surface
        # equation in row 4, and a melting surface is fixed at T0 and removed
        # from the solve.  Keeping a seven-row carrier is convenient for JAX,
        # but unused rows must be identities so no phantom snow layer couples
        # into the ice solution.
        no_snow_a = jnp.zeros_like(a)
        no_snow_b = jnp.ones_like(b)
        no_snow_c = jnp.zeros_like(c)
        no_snow_d = jnp.zeros_like(d)
        no_snow_b = no_snow_b.at[..., 3].set(dqns - 2.0 * k_i[..., 0])
        no_snow_c = no_snow_c.at[..., 3].set(2.0 * k_i[..., 0])
        no_snow_d = no_snow_d.at[..., 3].set(dqns * Tsu - fnet)
        no_snow_a = no_snow_a.at[..., 4].set(-eta_i[..., 0] * 2.0 * k_i[..., 0])
        no_snow_b = no_snow_b.at[..., 4].set(
            1.0 + eta_i[..., 0] * (k_i[..., 1] + 2.0 * k_i[..., 0])
        )
        no_snow_c = no_snow_c.at[..., 4].set(-eta_i[..., 0] * k_i[..., 1])
        no_snow_d = no_snow_d.at[..., 4].set(
            T_i_old[..., 0] + eta_i[..., 0] * ab_i[..., 0]
        )
        for row in (5, 6):
            no_snow_a = no_snow_a.at[..., row].set(a[..., row])
            no_snow_b = no_snow_b.at[..., row].set(b[..., row])
            no_snow_c = no_snow_c.at[..., row].set(c[..., row])
            no_snow_d = no_snow_d.at[..., row].set(d[..., row])
        branch_snow_present = (
            snow_present if _nemo_branch_ranges else jnp.ones_like(snow_present)
        )
        a = jnp.where(branch_snow_present[..., None], a, no_snow_a)
        b = jnp.where(branch_snow_present[..., None], b, no_snow_b)
        c = jnp.where(branch_snow_present[..., None], c, no_snow_c)
        d = jnp.where(branch_snow_present[..., None], d, no_snow_d)

        melting_surface = jnp.where(
            _nemo_branch_ranges, Tsu >= constants.T0, False
        )
        snow_melting = branch_snow_present & melting_surface
        no_snow_melting = (~branch_snow_present) & melting_surface
        b = b.at[..., 0].set(jnp.where(snow_melting, 1.0, b[..., 0]))
        c = c.at[..., 0].set(jnp.where(snow_melting, 0.0, c[..., 0]))
        d = d.at[..., 0].set(jnp.where(snow_melting, Tsu, d[..., 0]))
        a = a.at[..., 1].set(jnp.where(snow_melting, 0.0, a[..., 1]))
        d = d.at[..., 1].set(jnp.where(
            snow_melting,
            T_s_old[..., 0] + eta_s[..., 0]
            * (ab_s[..., 0] + 2.0 * k_s[..., 0] * Tsu),
            d[..., 1],
        ))
        b = b.at[..., 3].set(jnp.where(no_snow_melting, 1.0, b[..., 3]))
        c = c.at[..., 3].set(jnp.where(no_snow_melting, 0.0, c[..., 3]))
        d = d.at[..., 3].set(jnp.where(no_snow_melting, Tsu, d[..., 3]))
        a = a.at[..., 4].set(jnp.where(no_snow_melting, 0.0, a[..., 4]))
        d = d.at[..., 4].set(jnp.where(
            no_snow_melting,
            T_i_old[..., 0] + eta_i[..., 0]
            * (ab_i[..., 0] + 2.0 * k_i[..., 0] * Tsu),
            d[..., 4],
        ))

        solution = thomas_solve(a, b, c, d, "nemo_unnormalised")
        solved_surface = jnp.where(
            branch_snow_present, solution[..., 0], solution[..., 3]
        )
        Tsu_candidate = jnp.clip(
            jnp.where(melting_surface, Tsu, solved_surface),
            constants.T0 - 100.0,
            constants.T0,
        )
        Ts_candidate = jnp.where(
            branch_snow_present[..., None],
            jnp.clip(solution[..., 1:4], constants.T0 - 100.0, constants.T0),
            Ts,
        )
        melt = constants.T0 - constants.liquidus_slope * S_layers
        Ti_candidate = jnp.maximum(
            jnp.minimum(solution[..., 4:7], melt), constants.T0 - 100.0
        )
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
        iterations = jnp.where(just_converged, _iteration + 1, iterations)
        return (
            Tsu,
            jnp.where(active, carry[0], Tsu_previous),
            Ti,
            Ts,
            qns,
            converged | just_converged,
            iterations,
        )

    init = (
        T_surface,
        T_surface_input,
        T_i_old,
        T_s_old,
        qns_initial,
        jnp.zeros_like(h_ice, dtype=bool),
        jnp.full_like(h_ice, _maximum_iterations, dtype=jnp.int32),
    )
    Tsu, _, Ti, Ts, qns, _, iterations = jax.lax.fori_loop(
        0, _maximum_iterations, body, init
    )
    e_i = ice_enthalpy_from_temperature(
        Ti, S_layers, constants, _nemo_order=_nemo_eos_order
    )
    e_s = snow_enthalpy_from_temperature(Ts, constants)

    # Interface conductive diagnostics, `icethd_zdf_bl99.F90:743-761`.
    k_top_i = p07_conductivity(Ti[..., 0], S_layers[..., 0], constants) / dz_i
    k_base_i = p07_conductivity(T_bottom, S_layers[..., -1], constants) / dz_i
    k_top = jnp.where(
        snow_present,
        constants.k_snow / dz_s,
        k_top_i,
    )
    T_first = jnp.where(snow_present, Ts[..., 0], Ti[..., 0])
    qcn_top = 2.0 * k_top * (Tsu - T_first)
    qcn_bottom = 2.0 * k_base_i * (Ti[..., -1] - T_bottom)
    return SI3ZDFResult(
        Tsu, Ti, Ts, e_i, e_s, qns, qtr_bottom, qcn_top, qcn_bottom, iterations
    )


def _piecewise_remap(thickness, enthalpy, new_total, n_layers: int = 3):
    """Conservative constant-cell remap used by `ice_var_vremap`/`snw_ent`."""

    old_lo = jnp.cumsum(thickness, axis=-1) - thickness
    old_hi = jnp.cumsum(thickness, axis=-1)
    dz_new = new_total / n_layers
    idx = jnp.arange(n_layers, dtype=new_total.dtype)
    new_lo = dz_new[..., None] * idx
    new_hi = dz_new[..., None] * (idx + 1.0)
    overlap = jnp.maximum(
        0.0,
        jnp.minimum(new_hi[..., :, None], old_hi[..., None, :])
        - jnp.maximum(new_lo[..., :, None], old_lo[..., None, :]),
    )
    content = jnp.sum(overlap * enthalpy[..., None, :], axis=-1)
    return jnp.where(dz_new[..., None] > _EPS10, content / dz_new[..., None], 0.0)


def _nemo_snow_enthalpy_remap(thickness, enthalpy):
    """NEMO ``snw_ent`` cumulative remap (`icethd_dh.F90:535-613`)."""

    zero = jnp.zeros_like(thickness[..., 0])
    old_h_cumulative = [zero]
    old_e_cumulative = [zero]
    total = zero
    for layer in range(4):
        total = total + thickness[..., layer]
        old_h_cumulative.append(total)
        old_e_cumulative.append(
            old_e_cumulative[-1]
            + enthalpy[..., layer] * thickness[..., layer]
        )
    old_h_cumulative = jnp.stack(old_h_cumulative, axis=-1)
    old_e_cumulative = jnp.stack(old_e_cumulative, axis=-1)

    new_layer = total * (1.0 / 3.0)
    new_h_cumulative = [zero]
    for _ in range(3):
        new_h_cumulative.append(new_h_cumulative[-1] + new_layer)
    new_h_cumulative = jnp.stack(new_h_cumulative, axis=-1)
    new_e_cumulative = jnp.zeros_like(new_h_cumulative)
    for old in range(1, 5):
        for new in range(1, 3):
            selected = (
                (new_h_cumulative[..., new] <= old_h_cumulative[..., old])
                & (new_h_cumulative[..., new] > old_h_cumulative[..., old - 1])
            )
            denominator = old_h_cumulative[..., old] - old_h_cumulative[..., old - 1]
            safe_denominator = jnp.where(selected, denominator, 1.0)
            interpolated = (
                old_e_cumulative[..., old - 1]
                * (old_h_cumulative[..., old] - new_h_cumulative[..., new])
                + old_e_cumulative[..., old]
                * (new_h_cumulative[..., new] - old_h_cumulative[..., old - 1])
            ) / safe_denominator
            new_e_cumulative = new_e_cumulative.at[..., new].set(jnp.where(
                selected, interpolated, new_e_cumulative[..., new]
            ))
    new_e_cumulative = new_e_cumulative.at[..., 3].set(
        old_e_cumulative[..., 4]
    )
    return jnp.stack([
        jnp.maximum(
            0.0,
            new_e_cumulative[..., layer] - new_e_cumulative[..., layer - 1],
        ) / jnp.maximum(new_layer, _EPS20)
        for layer in range(1, 4)
    ], axis=-1)


def _dh_step(state: SI3ColumnArrays, zdf: SI3ZDFResult,
             forcing: SI3SurfaceForcing, dt: float,
             constants: IceConstantsConfig, *,
             _snow_deposition: bool = True,
             _surface_melt: bool = True,
             _basal_melt: bool = True,
             _nemo_basal_layer_loop: bool = True,
             _snow_ice_salinity: bool = True,
             _nemo_snow_sublimation_order: bool = True,
             _nemo_snow_remap: bool = True,
             ) -> SI3ColumnArrays:
    """Resolved no-lateral-melt thickness sequence (`icethd_dh.F90:91-533`)."""

    a = state.concentration
    h_i0, h_s0 = state.h_ice, state.h_snow
    # Snowfall partition and snow-first sublimation (:97-202).  Four segments
    # are ordered new precipitation, then the three old snow layers.
    snow_partition = 1.0 - jnp.power(
        jnp.maximum(1.0 - a, 0.0), _SNOW_BLOW_FRACTION
    )
    inverse_snow_density = 1.0 / constants.rho_snow
    if _nemo_snow_sublimation_order:
        h_precip = (
            snow_partition * forcing.snow_precipitation * dt
            * inverse_snow_density / jnp.maximum(a, _EPS10)
        )
        old_snow_layer = h_s0 * (1.0 / 3.0)
    else:
        h_precip = snow_partition * forcing.snow_precipitation * dt / (
            jnp.maximum(a, _EPS10) * constants.rho_snow
        )
        old_snow_layer = h_s0 / 3.0
    hseg_s = jnp.concatenate(
        [h_precip[..., None],
         jnp.broadcast_to(old_snow_layer[..., None], state.e_snow.shape)],
        axis=-1,
    )
    h_s_work = h_s0 + h_precip
    # `ze_s(0)` is initialized to zero and is assigned from qprec only inside
    # NEMO's positive-snowfall arm (`icethd_dh.F90:166-177`).  In particular,
    # a nonzero qprec diagnostic with zero snowfall must not suppress the
    # negative-evaporation deposition-temperature formula below.
    precip_enthalpy = jnp.where(
        forcing.snow_precipitation > 0.0,
        jnp.maximum(0.0, -forcing.qprec_ice),
        0.0,
    )
    deposited_enthalpy = constants.rho_snow * (
        constants.latent_fusion
        - constants.c_ice * (zdf.T_surface - constants.T0)
    )
    eseg_s = jnp.concatenate(
        [precip_enthalpy[..., None], zdf.e_snow], axis=-1
    )
    if _nemo_snow_sublimation_order:
        evaporation = jnp.where(
            _snow_deposition | (forcing.evaporation >= 0.0),
            forcing.evaporation,
            0.0,
        )
        snow_delta = jnp.maximum(
            -evaporation * inverse_snow_density * dt,
            -h_s_work,
        )
        remaining_mass = evaporation * dt + snow_delta * constants.rho_snow
        eseg_s = eseg_s.at[..., 0].set(jnp.where(
            (snow_delta > 0.0) & (eseg_s[..., 0] == 0.0),
            deposited_enthalpy,
            eseg_s[..., 0],
        ))
        kept = []
        for k in range(4):
            delta = jnp.maximum(-hseg_s[..., k], snow_delta)
            h_s_work = jnp.maximum(0.0, h_s_work + delta)
            kept.append(jnp.maximum(0.0, hseg_s[..., k] + delta))
            snow_delta = jnp.minimum(snow_delta - delta, 0.0)
        hseg_s = jnp.stack(kept, axis=-1)
    else:
        deposition = jnp.where(
            _snow_deposition,
            jnp.maximum(-forcing.evaporation * dt / constants.rho_snow, 0.0),
            0.0,
        )
        eseg_s = eseg_s.at[..., 0].set(jnp.where(
            (deposition > 0.0) & (eseg_s[..., 0] == 0.0),
            deposited_enthalpy,
            eseg_s[..., 0],
        ))
        hseg_s = hseg_s.at[..., 0].add(deposition)
        remaining_mass = jnp.maximum(forcing.evaporation * dt, 0.0)
        kept = []
        for k in range(4):
            remove = jnp.minimum(
                remaining_mass / constants.rho_snow, hseg_s[..., k]
            )
            kept.append(hseg_s[..., k] - remove)
            remaining_mass = jnp.maximum(
                remaining_mass - remove * constants.rho_snow, 0.0
            )
        hseg_s = jnp.stack(kept, axis=-1)
        h_s_work = jnp.sum(hseg_s, axis=-1)

    # Surface melt consumes the four snow segments before ice
    # (`icethd_dh.F90:204-280`).  Unlike the shared bulk snow helper, this path
    # must retain each NEMO enthalpy segment for the subsequent conservative
    # remap, so it is expressed on the existing segment carrier.
    qml = jnp.where(
        zdf.T_surface >= constants.T0,
        zdf.qns_ice + forcing.qsr_ice - forcing.qtr_ice_top - zdf.qcn_ice_top,
        0.0,
    )
    surface_energy = jnp.where(_surface_melt, jnp.maximum(qml * dt, 0.0), 0.0)
    melted_snow = []
    for k in range(4):
        active = (hseg_s[..., k] > 0.0) & (surface_energy > 0.0)
        delta = jnp.where(
            active,
            jnp.maximum(
                -surface_energy / jnp.maximum(eseg_s[..., k], 1.0e-20),
                -hseg_s[..., k],
            ),
            0.0,
        )
        melted_snow.append(hseg_s[..., k] + delta)
        h_s_work = jnp.maximum(0.0, h_s_work + delta)
        surface_energy = jnp.maximum(
            surface_energy + delta * eseg_s[..., k], 0.0
        )
    hseg_s = jnp.stack(melted_snow, axis=-1)

    old_dz = h_i0 / 3.0
    old_h = jnp.broadcast_to(old_dz[..., None], state.e_ice.shape)
    melt_temperature = constants.T0 - constants.liquidus_slope * state.S_layers
    for k in range(3):
        internal_melt = _surface_melt & (zdf.T_ice[..., k] >= melt_temperature[..., k])
        ice_specific = -zdf.e_ice[..., k] / constants.rho_ice
        water_specific = constants.c_ocean * (
            -constants.liquidus_slope * state.S_layers[..., k]
        )
        energy_difference = ice_specific - water_specific
        surface_remove = jnp.where(
            internal_melt,
            old_h[..., k],
            jnp.where(
                _surface_melt,
                jnp.minimum(
                    jnp.maximum(
                        surface_energy
                        / jnp.maximum(-energy_difference * constants.rho_ice, 1.0e-20),
                        0.0,
                    ),
                    old_h[..., k],
                ),
                0.0,
            ),
        )
        old_h = old_h.at[..., k].add(-surface_remove)
        surface_energy = jnp.where(
            internal_melt,
            surface_energy,
            jnp.maximum(
                surface_energy
                - surface_remove * (-energy_difference * constants.rho_ice),
                0.0,
            ),
        )
        sublimation = jnp.minimum(
            remaining_mass / constants.rho_ice, old_h[..., k]
        )
        old_h = old_h.at[..., k].add(-sublimation)
        remaining_mass = jnp.maximum(
            remaining_mass - sublimation * constants.rho_ice, 0.0
        )

    # Bottom thermodynamic growth/melt imbalance (:321-424).  The accepted
    # column has qtr=0 at kt=1; qtr is retained for the general selected arm.
    zf = zdf.qcn_ice_bottom + forcing.qsb_ice_bottom + forcing.fhld + zdf.qtr_ice_bottom
    S_new = _NEW_ICE_SALINITY_FRACTION * forcing.sss
    Tm = -constants.liquidus_slope * S_new
    Tnew_c = forcing.t_bottom - constants.T0
    e_i_specific = (
        constants.c_ice * (Tnew_c - Tm)
        - constants.latent_fusion * jnp.maximum(0.0, 1.0 - Tm / jnp.minimum(Tnew_c, -_EPS10))
        + constants.c_ocean * Tm
    )
    e_water = constants.c_ocean * (forcing.t_bottom - constants.T0)
    dE = e_i_specific - e_water
    dh_growth = dt * jnp.maximum(0.0, zf / (dE * constants.rho_ice))
    # Warm-ocean basal melt, limited to the available column; surface-melt and
    # internal-melt arms remain zero until their actual flux becomes positive.
    basal_energy = dt * jnp.maximum(zf, 0.0)
    if _nemo_basal_layer_loop:
        for k in (2, 1, 0):
            internal_melt = (zf > 0.0) & (
                zdf.T_ice[..., k] >= melt_temperature[..., k]
            )
            ice_specific = -zdf.e_ice[..., k] / constants.rho_ice
            water_specific = constants.c_ocean * (
                -constants.liquidus_slope * state.S_layers[..., k]
            )
            energy_difference = ice_specific - water_specific
            basal_remove = jnp.where(
                _basal_melt,
                jnp.where(
                    internal_melt,
                    old_h[..., k],
                    jnp.minimum(
                        jnp.maximum(
                            basal_energy
                            / jnp.maximum(
                                -energy_difference * constants.rho_ice, 1.0e-20
                            ),
                            0.0,
                        ),
                        old_h[..., k],
                    ),
                ),
                0.0,
            )
            old_h = old_h.at[..., k].add(-basal_remove)
            basal_energy = jnp.where(
                internal_melt,
                basal_energy,
                jnp.maximum(
                    basal_energy
                    - basal_remove * (-energy_difference * constants.rho_ice),
                    0.0,
                ),
            )
        h_i_pre_flood = jnp.sum(old_h, axis=-1) + dh_growth
    else:
        bottom_specific = -zdf.e_ice[..., -1] / constants.rho_ice
        bottom_water = constants.c_ocean * (
            -constants.liquidus_slope * state.S_layers[..., -1]
        )
        bottom_difference = bottom_specific - bottom_water
        legacy_remove = jnp.where(
            _basal_melt,
            jnp.minimum(
                jnp.maximum(
                    -basal_energy / (bottom_difference * constants.rho_ice),
                    0.0,
                ),
                jnp.sum(old_h, axis=-1),
            ),
            0.0,
        )
        h_i_pre_flood = jnp.sum(old_h, axis=-1) + dh_growth - legacy_remove
        old_h = old_h.at[..., -1].set(
            jnp.maximum(old_h[..., -1] - legacy_remove, 0.0)
        )

    # Remap old layer energy plus bottom new ice.  The first zero-thickness
    # segment is the top snow-ice slot in NEMO's `zh_i_old(0:nlay_i+1)`.
    hseg_i = jnp.concatenate(
        [jnp.zeros_like(h_i0)[..., None], old_h,
         dh_growth[..., None]], axis=-1
    )
    e_new_ice = -e_i_specific * constants.rho_ice
    eseg_i = jnp.concatenate(
        [jnp.zeros_like(h_i0)[..., None], zdf.e_ice,
         e_new_ice[..., None]], axis=-1
    )

    # Snow-ice flooding (:441-519).  Convert basal snow into ice at equal
    # thickness and include its seawater+snow energy as the top remap segment.
    h_i, h_s_final, flood = snow_ice_flooding(
        h_i_pre_flood,
        h_s_work,
        constants.rho_ice,
        constants.rho_snow,
        constants.rho_ocean,
    )
    hseg_i = hseg_i.at[..., 0].set(flood)
    flood_remaining = flood
    flood_content = (
        (constants.rho_snow - constants.rho_ice)
        * flood
        * constants.c_ocean
        * (forcing.t_bottom - constants.T0)
    )
    flooded_snow = list(jnp.moveaxis(hseg_s, -1, 0))
    for k in (3, 2, 1, 0):
        amount = jnp.minimum(flood_remaining, flooded_snow[k])
        flooded_snow[k] = jnp.maximum(flooded_snow[k] - amount, 0.0)
        flood_content = flood_content + amount * eseg_s[..., k]
        flood_remaining = jnp.maximum(flood_remaining - amount, 0.0)
    hseg_s = jnp.stack(flooded_snow, axis=-1)
    eseg_i = eseg_i.at[..., 0].set(jnp.where(
        flood > 0.0, flood_content / jnp.maximum(flood, 1.0e-20), 0.0
    ))
    e_i = _piecewise_remap(hseg_i, eseg_i, h_i)
    e_s = (
        _nemo_snow_enthalpy_remap(hseg_s, eseg_s)
        if _nemo_snow_remap
        else _piecewise_remap(hseg_s, eseg_s, h_s_final)
    )

    snow_ice_salinity = (
        forcing.sss
        * (constants.rho_ice - constants.rho_snow)
        / constants.rho_ice
    )
    snow_ice_salt_change = jnp.where(
        _snow_ice_salinity,
        (snow_ice_salinity - state.S_bulk) * flood,
        0.0,
    )
    S_bulk = state.S_bulk + (
        snow_ice_salt_change + (S_new - state.S_bulk) * dh_growth
    ) / jnp.maximum(h_i, _EPS10)
    return SI3ColumnArrays(a, h_i, h_s_final, zdf.T_surface, e_i, e_s,
                           S_bulk, state.S_layers, state.age_volume)


def si3_column_step_arrays(state: SI3ColumnArrays,
                           forcing: SI3SurfaceForcing,
                           dt: float,
                           constants: IceConstantsConfig, *,
                           _snow_deposition: bool = True,
                           _surface_melt: bool = True,
                           _zdf_branch_ranges: bool = True,
                           _basal_melt: bool = True,
                           _nemo_snow_temperature_bounds: bool = True,
                           _nemo_basal_layer_loop: bool = True,
                           _snow_ice_salinity: bool = True,
                           _nemo_eos_order: bool = True,
                           _nemo_snow_sublimation_order: bool = True,
                           _nemo_snow_remap: bool = True,
                           ) -> SI3StepTrace:
    """Execute the selected `ice_thd` chain and retain every oracle boundary."""

    zdf = _si3_zdf_bl99_step(
        state.e_ice, state.e_snow, state.S_layers, state.h_ice, state.h_snow,
        state.T_surface, forcing, dt, constants,
        _nemo_branch_ranges=_zdf_branch_ranges,
        _nemo_snow_temperature_bounds=_nemo_snow_temperature_bounds,
        _nemo_eos_order=_nemo_eos_order,
    )
    post_zdf = state._replace(T_surface=zdf.T_surface, e_ice=zdf.e_ice,
                              e_snow=zdf.e_snow)
    post_dh = _dh_step(
        post_zdf, zdf, forcing, dt, constants,
        _snow_deposition=_snow_deposition,
        _surface_melt=_surface_melt,
        _basal_melt=_basal_melt,
        _nemo_basal_layer_loop=_nemo_basal_layer_loop,
        _snow_ice_salinity=_snow_ice_salinity,
        _nemo_snow_sublimation_order=_nemo_snow_sublimation_order,
        _nemo_snow_remap=_nemo_snow_remap,
    )
    # `ice_thd_temp` is diagnostic because enthalpy is prognostic (:221-247).
    post_temp1 = post_dh
    drain = jnp.where(
        post_dh.T_surface >= constants.T0,
        -jnp.maximum(post_dh.S_bulk - _FLUSHING_TARGET, 0.0)
        * dt
        / _FLUSHING_TIMESCALE,
        jnp.where(
            post_dh.T_surface <= forcing.t_bottom,
            -jnp.maximum(post_dh.S_bulk - _DRAINAGE_TARGET, 0.0)
            * dt
            / _DRAINAGE_TIMESCALE,
            0.0,
        ),
    )
    S_bulk = jnp.clip(
        post_dh.S_bulk + drain,
        _MINIMUM_ICE_SALINITY,
        _NEW_ICE_SALINITY_FRACTION * forcing.sss,
    )
    S_layers = option2_salinity_profile(S_bulk, forcing.sss)
    post_sal = post_temp1._replace(S_bulk=S_bulk, S_layers=S_layers)
    post_temp2 = post_sal

    # Open-water growth (`icethd_do.F90:171-307`), active only for qlead<0.
    Tm = -constants.liquidus_slope * (
        _NEW_ICE_SALINITY_FRACTION * forcing.sss
    )
    Tbo_c = forcing.t_bottom - constants.T0
    e_new = constants.rho_ice * (
        constants.c_ice * (Tm - Tbo_c)
        + constants.latent_fusion * jnp.maximum(0.0, 1.0 - Tm / jnp.minimum(Tbo_c, -_EPS10))
        - constants.c_ocean * Tm
    )
    zEi = -e_new / constants.rho_ice
    zEw = constants.c_ocean * Tbo_c
    vnew = jnp.where(
        forcing.qlead < 0.0,
        -(-forcing.qlead / (zEi - zEw)) / constants.rho_ice,
        0.0,
    )
    anew = jnp.minimum(
        vnew / _NEW_ICE_THICKNESS,
        jnp.maximum(
            0.0,
            _MAXIMUM_ICE_CONCENTRATION - post_temp2.concentration,
        ),
    )
    residual = jnp.maximum(vnew - anew * _NEW_ICE_THICKNESS, 0.0)
    v_old = post_temp2.h_ice * post_temp2.concentration
    v_total = v_old + anew * _NEW_ICE_THICKNESS + residual
    a_total = post_temp2.concentration + anew
    h_total = jnp.where(a_total > _EPS10, v_total / a_total, 0.0)
    post_do = post_temp2._replace(concentration=a_total, h_ice=h_total)
    exit_state = post_do._replace(age_volume=post_do.age_volume + a_total * dt)
    return SI3StepTrace(state, post_zdf, post_dh, post_temp1, post_sal,
                        post_temp2, post_do, exit_state)


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
)
