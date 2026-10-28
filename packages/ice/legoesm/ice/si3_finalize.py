"""NEMO-literal SI3 post-thermodynamic correction and ocean exchange.

This is the single production implementation of ladder rungs O15 and O16:
``ice_cor(kt,2)``, ``ice_update_flx``, and the recorded implicit-drag arm of
``ice_update_tau``.  Source-order barriers follow the admitted NEMO 5.0.2
``ORCA2_SAS_ICE`` record build:

* ``BLD/ppsrc/nemo/icecor.f90:55-141``
* ``BLD/ppsrc/nemo/iceitd.f90:432-700``
* ``BLD/ppsrc/nemo/iceupdate.f90:84-423``

The production deck selects five categories, salinity option 4, level ponds,
thermodynamics on, conduction-flux mode off, ``nn_fsbc=1``, and explicit
ice-ocean drag.  Unsupported arms fail closed.  Literal switches remain
piecewise in the forward pass; safe denominators are used only on untaken
branches so JAX gradients remain finite.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.core.source_rounding import nemo_source_round as sr
from legoesm.ice.si3_orca2_state import (
    SI3ORCA2StateConfig,
    SI3State,
    fortran_sum_categories,
    si3_update_snwice_mass,
    si3_var_agg,
    si3_var_glo2eqv,
)
from legoesm.ice.si3_surface import SI3SurfaceConfig, si3_ice_alb
from legoesm.ice.transport import si3_var_zapsmall

from legoesm import constants as canonical_constants

Field = jax.Array

_ZERO = 0.0
_ONE = 1.0
_HALF = 0.5
_QUARTER = 0.25
_EPSI10 = 1.0e-10
_EPSI20 = 1.0e-20


class SI3Fluxes(NamedTuple):
    """All stateful SI3 thermodynamic flux channels recorded at O15/O16."""

    wfx_snw: Field
    wfx_snw_sni: Field
    wfx_snw_sum: Field
    wfx_pnd: Field
    wfx_spr: Field
    wfx_sub: Field
    wfx_snw_sub: Field
    wfx_ice_sub: Field
    wfx_ice: Field
    wfx_sni: Field
    wfx_opw: Field
    wfx_bog: Field
    wfx_dyn: Field
    wfx_bom: Field
    wfx_sum: Field
    wfx_lam: Field
    wfx_res: Field
    wfx_err_sub: Field
    sfx_bog: Field
    sfx_bom: Field
    sfx_lam: Field
    sfx_sum: Field
    sfx_sni: Field
    sfx_opw: Field
    sfx_bri: Field
    sfx_dyn: Field
    sfx_res: Field
    sfx_sub: Field
    hfx_bog: Field
    hfx_bom: Field
    hfx_sum: Field
    hfx_opw: Field
    hfx_dif: Field
    hfx_snw: Field
    hfx_err_dif: Field
    hfx_sub: Field
    hfx_spr: Field
    hfx_thd: Field
    hfx_dyn: Field
    hfx_res: Field
    qlead: Field
    qsb_ice_bot: Field
    fhld: Field


class SI3CorrectionResult(NamedTuple):
    state: SI3State
    fluxes: SI3Fluxes


class SI3UpdateFluxForcing(NamedTuple):
    """Resolved O16 atmosphere/ocean fields; every member is explicit."""

    qsr_oce: Field
    qns_oce: Field
    qemp_oce: Field
    qemp_ice: Field
    emp_oce: Field
    emp_ice: Field
    qns_tot: Field
    qsr_tot: Field
    frq_m: Field
    a_i_b: Field
    qsr_ice: Field
    qns_ice: Field
    qtr_ice_bot: Field
    qevap_ice: Field
    qcn_ice: Field
    qcn_ice_top: Field
    qcn_ice_bot: Field
    qml_ice: Field
    qtr_ice_top: Field
    alb_ice: Field
    at_i_b: Field
    wfx_snw_dyn: Field
    cloud_fra: Field
    qsr: Field
    sfx: Field
    tmask: Field
    smask0: Field
    snwice_mass: Field
    snwice_mass_b: Field


class SI3UpdateFluxOutputs(NamedTuple):
    qsr: Field
    qns: Field
    emp: Field
    sfx: Field
    fwfice: Field
    wfx_ice: Field
    wfx_snw: Field
    wfx_pnd: Field
    fr_i: Field
    tn_ice: Field
    alb_ice: Field
    snwice_mass: Field
    snwice_mass_b: Field
    snwice_fmass: Field


class SI3O16Result(NamedTuple):
    state: SI3State
    fluxes: SI3Fluxes
    outputs: SI3UpdateFluxOutputs


class SI3StressInputs(NamedTuple):
    uu_kbb: Field
    vv_kbb: Field
    umask_k1: Field
    vmask_k1: Field
    u_ice: Field
    v_ice: Field
    u_oce: Field
    v_oce: Field
    drag_io: Field
    at_i: Field
    utau: Field
    vtau: Field
    tmod_io: Field
    taum: Field
    utau_oce: Field
    vtau_oce: Field


class SI3StressOutputs(NamedTuple):
    utau: Field
    vtau: Field
    taum: Field
    tmod_io: Field
    utau_oce: Field
    vtau_oce: Field


class SI3FinalizeForcing(NamedTuple):
    """Resolved late-stage forcing carried through ``step_sea_ice``.

    This is the production boundary for the O15 ``ice_cor`` and O16
    ``ice_update_flx``/``ice_update_tau`` continuation.  It deliberately
    carries the already resolved NEMO fields instead of inventing a lossy
    ``AtmToSurface`` projection for the oracle lane.
    """

    fluxes: SI3Fluxes
    update_state: SI3State
    update: SI3UpdateFluxForcing
    stress: SI3StressInputs
    sss_m: Field
    sst_m: Field
    rn_amax_2d: Field
    rn_himin: float
    surface_config: SI3SurfaceConfig
    ln_drgice_imp: bool
    plant_skip_rebin: bool = False
    plant_strip_transfer_barriers: bool = False
    plant_omit_hfx_res: bool = False


class SI3FinalizeResponse(NamedTuple):
    """Observable O15/O16 stage boundaries returned by production stepping."""

    correction: SI3CorrectionResult
    final: SI3O16Result
    stress: SI3StressOutputs


def _safe_ratio(num: Field, den: Field, active: Field) -> Field:
    return jnp.where(active, sr(num / jnp.where(active, den, _ONE)), _ZERO)


def _category_sum(terms: Field) -> Field:
    return fortran_sum_categories(terms, axis=-1)


def _validate_finalize_state(state: SI3State, config: SI3ORCA2StateConfig) -> None:
    """Reject state/config disagreement before late-stage arithmetic runs."""
    if not isinstance(config, SI3ORCA2StateConfig):
        raise TypeError("SI3 finalization requires an explicit SI3ORCA2StateConfig")
    if config.jpl < 1 or config.nlay_i < 1 or config.nlay_s < 1:
        raise ValueError("SI3 finalization requires positive jpl/nlay_i/nlay_s")
    if config.nn_hls < 0 or config.rDt_ice <= 0.0:
        raise ValueError("SI3 finalization requires nn_hls >= 0 and rDt_ice > 0")
    jpi, jpj = state.a_i.shape[:2]
    expected3 = (jpi, jpj, config.jpl)
    expected4_i = (jpi, jpj, config.nlay_i, config.jpl)
    expected4_s = (jpi, jpj, config.nlay_s, config.jpl)
    checks = {
        "a_i": (state.a_i.shape, expected3),
        "v_i": (state.v_i.shape, expected3),
        "v_s": (state.v_s.shape, expected3),
        "e_i": (state.e_i.shape, expected4_i),
        "szv_i": (state.szv_i.shape, expected4_i),
        "e_s": (state.e_s.shape, expected4_s),
        "hi_max": (state.bounds.hi_max.shape, (config.jpl + 1,)),
        "hi_mean": (state.bounds.hi_mean.shape, (config.jpl,)),
    }
    wrong = [
        f"{name}={actual}, expected {expected}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if wrong:
        raise ValueError("SI3 finalization state/config shape mismatch: " + "; ".join(wrong))
    if config.nn_hls and (jpi <= 2 * config.nn_hls or jpj <= 2 * config.nn_hls):
        raise ValueError("SI3 finalization halo width leaves no owned interior")


def _roundoff_after_shift(
    active: Field,
    *,
    a_i: Field,
    v_i: Field,
    v_s: Field,
    sv_i: Field,
    oa_i: Field,
    a_ip: Field,
    v_ip: Field,
    v_il: Field,
    e_s: Field,
    e_i: Field,
    szv_i: Field,
    config: SI3ORCA2StateConfig,
) -> tuple[Field, ...]:
    """``ice_var_roundoff`` on the cells selected by one rebin boundary."""
    a3 = active[..., None]
    a4 = active[..., None, None]

    def nonnegative(x: Field, mask: Field) -> Field:
        return jnp.where(mask & (x < _ZERO), _ZERO, x)

    a_i = nonnegative(a_i, a3)
    v_i = nonnegative(v_i, a3)
    v_s = nonnegative(v_s, a3)
    oa_i = nonnegative(oa_i, a3)
    e_i = nonnegative(e_i, a4)
    e_s = nonnegative(e_s, a4)
    if config.ln_pnd_LEV or config.ln_pnd_TOPO:
        a_ip = nonnegative(a_ip, a3)
        v_ip = nonnegative(v_ip, a3)
        if config.ln_pnd_lids:
            v_il = jnp.where(
                a3 & (v_il < _ZERO) & (v_il > -_EPSI10), _ZERO, v_il
            )
    if config.nn_icesal == 4:
        szv_i = nonnegative(szv_i, a4)
    else:
        sv_i = nonnegative(sv_i, a3)
    return a_i, v_i, v_s, sv_i, oa_i, a_ip, v_ip, v_il, e_s, e_i, szv_i


def _shift_boundary(
    donor: int,
    receiver: int,
    active: Field,
    pdaice: Field,
    pdvice: Field,
    *,
    a_i: Field,
    v_i: Field,
    v_s: Field,
    sv_i: Field,
    oa_i: Field,
    a_ip: Field,
    v_ip: Field,
    v_il: Field,
    t_su: Field,
    h_i: Field,
    e_s: Field,
    e_i: Field,
    szv_i: Field,
    rn_amax_2d: Field,
    config: SI3ORCA2StateConfig,
    _materialize_transfer_statements: bool = True,
) -> tuple[Field, ...]:
    """One ``itd_shiftice`` call, including its roundoff and cap sections."""
    statement_result = sr if _materialize_transfer_statements else (lambda x: x)
    v_d = v_i[..., donor]
    a_d = a_i[..., donor]
    zworkv = _safe_ratio(pdvice, v_d, active & (v_d >= _EPSI10))
    zworka = _safe_ratio(pdaice, a_d, active & (a_d >= _EPSI10))
    za_tsfn = sr(a_i * t_su)

    def transfer3(x: Field, fraction: Field) -> Field:
        ztrans = sr(x[..., donor] * fraction)
        donor_new = statement_result(x[..., donor] - ztrans)
        receiver_new = statement_result(x[..., receiver] + ztrans)
        x = x.at[..., donor].set(jnp.where(active, donor_new, x[..., donor]))
        return x.at[..., receiver].set(
            jnp.where(active, receiver_new, x[..., receiver])
        )

    def transfer4(x: Field, fraction: Field) -> Field:
        ztrans = sr(x[..., donor] * fraction[..., None])
        donor_new = statement_result(x[..., donor] - ztrans)
        receiver_new = statement_result(x[..., receiver] + ztrans)
        x = x.at[..., donor].set(
            jnp.where(active[..., None], donor_new, x[..., donor])
        )
        return x.at[..., receiver].set(
            jnp.where(active[..., None], receiver_new, x[..., receiver])
        )

    a_donor_new = statement_result(a_i[..., donor] - pdaice)
    a_i = a_i.at[..., donor].set(
        jnp.where(active, a_donor_new, a_i[..., donor])
    )
    a_receiver_new = statement_result(a_i[..., receiver] + pdaice)
    a_i = a_i.at[..., receiver].set(
        jnp.where(active, a_receiver_new, a_i[..., receiver])
    )
    v_donor_new = statement_result(v_i[..., donor] - pdvice)
    v_i = v_i.at[..., donor].set(
        jnp.where(active, v_donor_new, v_i[..., donor])
    )
    v_receiver_new = statement_result(v_i[..., receiver] + pdvice)
    v_i = v_i.at[..., receiver].set(
        jnp.where(active, v_receiver_new, v_i[..., receiver])
    )
    v_s = transfer3(v_s, zworkv)
    oa_i = transfer3(oa_i, zworka)
    za_tsfn = transfer3(za_tsfn, zworka)
    if config.ln_pnd_LEV or config.ln_pnd_TOPO:
        a_ip = transfer3(a_ip, zworka)
        v_ip = transfer3(v_ip, zworkv)
        if config.ln_pnd_lids:
            v_il = transfer3(v_il, zworkv)
    e_s = transfer4(e_s, zworkv)
    e_i = transfer4(e_i, zworkv)
    if config.nn_icesal == 4:
        szv_i = transfer4(szv_i, zworkv)
    else:
        sv_i = transfer3(sv_i, zworkv)

    (
        a_i,
        v_i,
        v_s,
        sv_i,
        oa_i,
        a_ip,
        v_ip,
        v_il,
        e_s,
        e_i,
        szv_i,
    ) = _roundoff_after_shift(
        active,
        a_i=a_i,
        v_i=v_i,
        v_s=v_s,
        sv_i=sv_i,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        config=config,
    )

    at_i = _category_sum(a_i)
    over = active & (at_i > rn_amax_2d)
    for jl in range(a_i.shape[-1]):
        scaled = sr(sr(a_i[..., jl] * rn_amax_2d) / jnp.where(over, at_i, _ONE))
        a_i = a_i.at[..., jl].set(jnp.where(over, scaled, a_i[..., jl]))

    present = a_i >= _EPSI20
    h_i_updated = jnp.where(
        present, sr(v_i / jnp.where(present, a_i, _ONE)), _ZERO
    )
    t_su_updated = jnp.where(
        present,
        sr(za_tsfn / jnp.where(present, a_i, _ONE)),
        jnp.asarray(config.ice_constants.T0, dtype=a_i.dtype),
    )
    # itd_shiftice gathers only nptidx-selected cells; inactive cells are not
    # written by iceitd.f90:575-595.
    h_i = jnp.where(active[..., None], h_i_updated, h_i)
    t_su = jnp.where(active[..., None], t_su_updated, t_su)
    return a_i, v_i, v_s, sv_i, oa_i, a_ip, v_ip, v_il, t_su, e_s, e_i, szv_i, h_i


def _ice_itd_reb(
    state: SI3State,
    rn_amax_2d: Field,
    config: SI3ORCA2StateConfig,
    *,
    _materialize_transfer_statements: bool = True,
) -> SI3State:
    """Production ``ice_itd_reb``/``itd_shiftice`` for five categories."""
    if config.jpl <= 1:
        return state
    a_i, v_i, v_s, sv_i = state.a_i, state.v_i, state.v_s, state.sv_i
    oa_i, a_ip, v_ip, v_il = state.oa_i, state.a_ip, state.v_ip, state.v_il
    t_su, e_s, e_i, szv_i = state.t_su, state.e_s, state.e_i, state.szv_i
    h_i = state.equivalent.h_i
    for jl in range(config.jpl - 1):
        active = (a_i[..., jl] > _ZERO) & (
            v_i[..., jl] > sr(a_i[..., jl] * state.bounds.hi_max[jl + 1])
        )
        pdaice = sr(_HALF * a_i[..., jl])
        retained = sr(sr((_ONE - _HALF) * a_i[..., jl]) * state.bounds.hi_mean[jl])
        pdvice = sr(v_i[..., jl] - retained)
        (
            a_i, v_i, v_s, sv_i, oa_i, a_ip, v_ip, v_il, t_su,
            e_s, e_i, szv_i, h_i,
        ) = _shift_boundary(
            jl,
            jl + 1,
            active,
            pdaice,
            pdvice,
            a_i=a_i,
            v_i=v_i,
            v_s=v_s,
            sv_i=sv_i,
            oa_i=oa_i,
            a_ip=a_ip,
            v_ip=v_ip,
            v_il=v_il,
            t_su=t_su,
            h_i=h_i,
            e_s=e_s,
            e_i=e_i,
            szv_i=szv_i,
            rn_amax_2d=rn_amax_2d,
            config=config,
            _materialize_transfer_statements=_materialize_transfer_statements,
        )

    for jl in range(config.jpl - 2, -1, -1):
        donor = jl + 1
        active = (a_i[..., donor] > _ZERO) & (
            v_i[..., donor] <= sr(a_i[..., donor] * state.bounds.hi_max[jl + 1])
        )
        pdaice = a_i[..., donor]
        pdvice = v_i[..., donor]
        (
            a_i, v_i, v_s, sv_i, oa_i, a_ip, v_ip, v_il, t_su,
            e_s, e_i, szv_i, h_i,
        ) = _shift_boundary(
            donor,
            jl,
            active,
            pdaice,
            pdvice,
            a_i=a_i,
            v_i=v_i,
            v_s=v_s,
            sv_i=sv_i,
            oa_i=oa_i,
            a_ip=a_ip,
            v_ip=v_ip,
            v_il=v_il,
            t_su=t_su,
            h_i=h_i,
            e_s=e_s,
            e_i=e_i,
            szv_i=szv_i,
            rn_amax_2d=rn_amax_2d,
            config=config,
            _materialize_transfer_statements=_materialize_transfer_statements,
        )

    eqv = state.equivalent._replace(h_i=h_i, a_i=a_i, v_s=v_s)
    return state._replace(
        a_i=a_i,
        v_i=v_i,
        v_s=v_s,
        sv_i=sv_i,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        t_su=t_su,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        equivalent=eqv,
    )


def si3_ice_cor(
    state: SI3State,
    fluxes: SI3Fluxes,
    *,
    sss_m: Field,
    sst_m: Field,
    rn_amax_2d: Field,
    rn_himin: float,
    config: SI3ORCA2StateConfig,
    _plant_skip_rebin: bool = False,
    _plant_strip_transfer_barriers: bool = False,
) -> SI3CorrectionResult:
    """Execute production ``ice_cor(kt,2)`` in NEMO statement order."""
    _validate_finalize_state(state, config)
    if rn_himin <= 0.0:
        raise ValueError("O15 requires rn_himin > 0")
    if config.nn_icesal != 4:
        raise ValueError("O15 owns the production-deck nn_icesal=4 arm only")
    if not (config.ln_pnd_LEV and config.ln_pnd_lids) or config.ln_pnd_TOPO:
        raise ValueError("O15 requires the resolved level-pond/lid arm")

    jpi, jpj, _ = state.a_i.shape
    interior = jnp.zeros((jpi, jpj), dtype=state.a_i.dtype)
    h = int(config.nn_hls)
    sl = (slice(h, -h), slice(h, -h)) if h else (slice(None), slice(None))
    interior = interior.at[sl].set(_ONE)
    interior3 = interior[..., None].astype(bool)
    statement_result = sr if not _plant_strip_transfer_barriers else (lambda x: x)
    # icecor.f90:81-87 are separate stored statements.  Materializing the
    # divisor also prevents XLA replacing `/ rn_himin` by a rounded reciprocal
    # multiply before the result enters the category-transfer sum.
    rn_himin_value = statement_result(jnp.asarray(rn_himin, dtype=state.a_i.dtype))
    present = state.a_i >= _EPSI20
    h_i = jnp.where(present, sr(state.v_i / jnp.where(present, state.a_i, _ONE)), _ZERO)
    a_ip = statement_result(
        jnp.where(
            interior3 & (h_i < rn_himin),
            sr(sr(state.a_ip * h_i) / rn_himin_value),
            state.a_ip,
        )
    )
    a_i = statement_result(
        jnp.where(
            interior3 & (h_i < rn_himin),
            sr(sr(state.a_i * h_i) / rn_himin_value),
            state.a_i,
        )
    )
    at_i = _category_sum(a_i)
    over = interior.astype(bool) & (at_i > rn_amax_2d)
    for jl in range(config.jpl):
        scaled = sr(sr(a_i[..., jl] * rn_amax_2d) / jnp.where(over, at_i, _ONE))
        a_i = a_i.at[..., jl].set(jnp.where(over, scaled, a_i[..., jl]))
    state = state._replace(
        a_i=a_i,
        a_ip=a_ip,
        equivalent=state.equivalent._replace(h_i=h_i, a_i=a_i),
        aggregated=state.aggregated._replace(at_i=at_i),
    )

    if config.jpl > 1 and not _plant_skip_rebin:
        state = _ice_itd_reb(
            state,
            rn_amax_2d,
            config,
            _materialize_transfer_statements=not _plant_strip_transfer_barriers,
        )

    dtype = state.a_i.dtype
    r1_nlay_i = sr(jnp.asarray(_ONE / float(config.nlay_i), dtype=dtype))
    r1_dt = sr(jnp.asarray(_ONE / config.rDt_ice, dtype=dtype))
    rhoi = jnp.asarray(config.ice_constants.rho_ice, dtype=dtype)
    sss = jnp.asarray(sss_m, dtype=dtype)
    szv_i = state.szv_i
    sfx_res = fluxes.sfx_res
    for jl in range(config.jpl):
        for jk in range(config.nlay_i):
            old = szv_i[..., jk, jl]
            lower = sr(sr(config.rn_simin * state.v_i[..., jl]) * r1_nlay_i)
            upper = sr(
                sr(sr(sr(config.rn_sinew * sss) * state.v_i[..., jl]) * r1_nlay_i)
            )
            # icecor.o: MAX(current, lower) keeps current on a tie.  At 0x1dbe,
            # ``minsd %xmm6,%xmm9`` has bounded_low in src and upper in dst,
            # so bounded_low is retained on an ordered equality (including
            # the reachable +0/-0 tie).
            bounded_low = jnp.where(lower > old, lower, old)
            bounded = jnp.where(upper < bounded_low, upper, bounded_low)
            bounded = jnp.where(interior.astype(bool), bounded, old)
            szv_i = szv_i.at[..., jk, jl].set(bounded)
            delta = sr(bounded - old)
            correction = sr(sr(delta * rhoi) * r1_dt)
            sfx_res = jnp.where(
                interior.astype(bool), sr(sfx_res - correction), sfx_res
            )
    state = state._replace(szv_i=szv_i)

    eqv = state.equivalent
    agg = state.aggregated
    (
        v_i,
        v_s,
        sv_i,
        oa_i,
        a_i,
        a_ip,
        v_ip,
        v_il,
        t_su,
        h_i,
        h_s,
        h_ip,
        h_il,
        e_s,
        e_i,
        szv_i,
        sz_i,
        t_i,
        t_s,
        at_i,
        vt_i,
        ato_i,
        wfx_res,
        hfx_res,
        sfx_res,
    ) = si3_var_zapsmall(
        v_i=state.v_i,
        v_s=state.v_s,
        sv_i=state.sv_i,
        oa_i=state.oa_i,
        a_i=state.a_i,
        a_ip=state.a_ip,
        v_ip=state.v_ip,
        v_il=state.v_il,
        t_su=state.t_su,
        h_i=eqv.h_i,
        h_s=eqv.h_s,
        h_ip=eqv.h_ip,
        h_il=eqv.h_il,
        e_s=state.e_s,
        e_i=state.e_i,
        szv_i=state.szv_i,
        sz_i=eqv.sz_i,
        t_i=eqv.t_i,
        t_s=eqv.t_s,
        at_i=agg.at_i,
        vt_i=agg.vt_i,
        ato_i=agg.ato_i,
        wfx_res=fluxes.wfx_res,
        hfx_res=fluxes.hfx_res,
        sfx_res=sfx_res,
        rDt_ice=config.rDt_ice,
        sss_m=sss_m,
        sst_m=sst_m,
        nn_hls=config.nn_hls,
        nn_icesal=config.nn_icesal,
        rn_simin=config.rn_simin,
        rn_sinew=config.rn_sinew,
        rt0=config.ice_constants.T0,
        rhoi=config.ice_constants.rho_ice,
        rhos=config.ice_constants.rho_snow,
        rhow=canonical_constants.rho_water,
    )
    eqv = eqv._replace(
        h_i=h_i,
        h_s=h_s,
        h_ip=h_ip,
        h_il=h_il,
        sz_i=sz_i,
        t_i=t_i,
        t_s=t_s,
        a_i=a_i,
        v_s=v_s,
    )
    agg = agg._replace(at_i=at_i, vt_i=vt_i, ato_i=ato_i)
    state = state._replace(
        v_i=v_i,
        v_s=v_s,
        sv_i=sv_i,
        oa_i=oa_i,
        a_i=a_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        t_su=t_su,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        equivalent=eqv,
        aggregated=agg,
    )
    return SI3CorrectionResult(
        state=state,
        fluxes=fluxes._replace(wfx_res=wfx_res, hfx_res=hfx_res, sfx_res=sfx_res),
    )


def si3_o15_step(
    state: SI3State,
    fluxes: SI3Fluxes,
    *,
    sss_m: Field,
    sst_m: Field,
    rn_amax_2d: Field,
    rn_himin: float,
    config: SI3ORCA2StateConfig,
    _plant_skip_rebin: bool = False,
    _plant_strip_transfer_barriers: bool = False,
) -> SI3CorrectionResult:
    """Production O15 call site; removing ``si3_ice_cor`` changes this result."""
    return si3_ice_cor(
        state,
        fluxes,
        sss_m=sss_m,
        sst_m=sst_m,
        rn_amax_2d=rn_amax_2d,
        rn_himin=rn_himin,
        config=config,
        _plant_skip_rebin=_plant_skip_rebin,
        _plant_strip_transfer_barriers=_plant_strip_transfer_barriers,
    )


def _sum_flux_terms(*terms: Field) -> Field:
    acc = sr(jnp.zeros_like(terms[0]))
    for term in terms:
        acc = sr(acc + term)
    return acc


def si3_ice_update_flx(
    state: SI3State,
    fluxes: SI3Fluxes,
    forcing: SI3UpdateFluxForcing,
    *,
    config: SI3ORCA2StateConfig,
    surface_config: SI3SurfaceConfig,
    _plant_omit_hfx_res: bool = False,
) -> tuple[SI3Fluxes, SI3UpdateFluxOutputs]:
    """Execute the active production-deck ``ice_update_flx`` arm."""
    if not config.ln_icethd:
        raise ValueError("O16 requires resolved ln_icethd=True")
    if surface_config.ln_cndflx:
        raise ValueError("O16 owns the resolved ln_cndflx=False arm")

    agg = state.aggregated
    eqv = state.equivalent
    h = int(config.nn_hls)
    sl = (slice(h, -h), slice(h, -h)) if h else (slice(None), slice(None))
    at_i_interior = agg.at_i[sl]
    qt_atm_oi = sr(forcing.qns_tot + forcing.qsr_tot)
    transmitted = _category_sum(sr(forcing.a_i_b * sr(forcing.qsr_ice - forcing.qtr_ice_bot)))
    zqsr = sr(forcing.qsr_tot - transmitted)

    qt_oce_ai = sr(qt_atm_oi - fluxes.hfx_sum)
    qt_oce_ai = sr(qt_oce_ai - fluxes.hfx_bom)
    qt_oce_ai = sr(qt_oce_ai - fluxes.hfx_bog)
    qt_oce_ai = sr(qt_oce_ai - fluxes.hfx_dif)
    qt_oce_ai = sr(qt_oce_ai - fluxes.hfx_opw)
    qt_oce_ai = sr(qt_oce_ai - fluxes.hfx_snw)
    qt_oce_ai = sr(qt_oce_ai + fluxes.hfx_thd)
    qt_oce_ai = sr(qt_oce_ai + fluxes.hfx_dyn)
    if not _plant_omit_hfx_res:
        qt_oce_ai = sr(qt_oce_ai + fluxes.hfx_res)
    qt_oce_ai = sr(qt_oce_ai + fluxes.hfx_sub)
    qevap = _category_sum(sr(forcing.qevap_ice * forcing.a_i_b))
    qt_oce_ai = sr(qt_oce_ai - qevap)
    qt_oce_ai = sr(qt_oce_ai + fluxes.hfx_spr)

    one_minus_at_b = sr(_ONE - forcing.at_i_b)
    one_minus_frq = sr(_ONE - forcing.frq_m)
    open_qsr = sr(sr(one_minus_at_b * forcing.qsr_oce) * one_minus_frq)
    ice_qsr = _category_sum(sr(forcing.a_i_b * forcing.qtr_ice_bot))
    ice_qsr = sr(ice_qsr * one_minus_frq)
    warm_qsr = sr(open_qsr + ice_qsr)
    qsr = jnp.where((fluxes.fhld > _ZERO) & (at_i_interior > _ZERO), warm_qsr, zqsr)
    qns = sr(qt_oce_ai - qsr)

    wfx_sub = sr(fluxes.wfx_snw_sub + fluxes.wfx_ice_sub)
    wfx_ice = _sum_flux_terms(
        fluxes.wfx_bog,
        fluxes.wfx_bom,
        fluxes.wfx_sum,
        fluxes.wfx_sni,
        fluxes.wfx_opw,
        fluxes.wfx_dyn,
        fluxes.wfx_res,
        fluxes.wfx_lam,
    )
    wfx_snw = _sum_flux_terms(
        fluxes.wfx_snw_sni, forcing.wfx_snw_dyn, fluxes.wfx_snw_sum
    )
    fwfice = _sum_flux_terms(wfx_ice, wfx_snw, fluxes.wfx_pnd, fluxes.wfx_err_sub)
    emp = sr(forcing.emp_oce - wfx_ice)
    emp = sr(emp - wfx_snw)
    emp = sr(emp - fluxes.wfx_pnd)
    emp = sr(emp - fluxes.wfx_err_sub)
    sfx = _sum_flux_terms(
        fluxes.sfx_bog,
        fluxes.sfx_bom,
        fluxes.sfx_sum,
        fluxes.sfx_sni,
        fluxes.sfx_opw,
        fluxes.sfx_res,
        fluxes.sfx_dyn,
        fluxes.sfx_bri,
        fluxes.sfx_sub,
        fluxes.sfx_lam,
    )

    snwice_mass, snwice_mass_b, snwice_fmass = si3_update_snwice_mass(
        vt_i=agg.vt_i,
        vt_s=agg.vt_s,
        vt_ip=agg.vt_ip,
        vt_il=agg.vt_il,
        snwice_mass_prev=forcing.snwice_mass,
        tmask=forcing.tmask,
        config=config,
        is_init=False,
    )
    fr_i = agg.at_i

    tn_ice = state.t_su[sl]
    alb_ice = si3_ice_alb(
        t_su=state.t_su[sl],
        h_i=eqv.h_i[sl],
        h_s=eqv.h_s[sl],
        a_ip_eff=eqv.a_ip_eff[sl],
        h_ip=eqv.h_ip[sl],
        h_il=eqv.h_il[sl],
        cloud_fra=forcing.cloud_fra,
        smask0=forcing.smask0,
        config=surface_config,
    )
    fluxes_out = fluxes._replace(
        wfx_sub=wfx_sub,
        wfx_ice=wfx_ice,
        wfx_snw=wfx_snw,
    )
    outputs = SI3UpdateFluxOutputs(
        qsr=qsr,
        qns=qns,
        emp=emp,
        sfx=sfx,
        fwfice=fwfice,
        wfx_ice=wfx_ice,
        wfx_snw=wfx_snw,
        wfx_pnd=fluxes.wfx_pnd,
        fr_i=fr_i,
        tn_ice=tn_ice,
        alb_ice=alb_ice,
        snwice_mass=snwice_mass,
        snwice_mass_b=snwice_mass_b,
        snwice_fmass=snwice_fmass,
    )
    return fluxes_out, outputs


def si3_o16_step(
    state: SI3State,
    fluxes: SI3Fluxes,
    forcing: SI3UpdateFluxForcing,
    *,
    sss_m: Field,
    config: SI3ORCA2StateConfig,
    surface_config: SI3SurfaceConfig,
    _plant_omit_hfx_res: bool = False,
) -> SI3O16Result:
    """Production O16 call site: aggregate kind 18, then ``ice_update_flx``."""
    _validate_finalize_state(state, config)
    if surface_config.nn_hls != config.nn_hls:
        raise ValueError("O16 surface and state configurations must use the same nn_hls")
    eqv = si3_var_glo2eqv(
        v_i=state.v_i,
        v_s=state.v_s,
        sv_i=state.sv_i,
        a_i=state.a_i,
        oa_i=state.oa_i,
        a_ip=state.a_ip,
        v_ip=state.v_ip,
        v_il=state.v_il,
        e_s=state.e_s,
        e_i=state.e_i,
        szv_i=state.szv_i,
        kn=2,
        hi_max_top=state.bounds.hi_max[-1],
        sss_m=sss_m,
        config=config,
    )
    agg = si3_var_agg(
        v_i=state.v_i,
        v_s=eqv.v_s,
        sv_i=state.sv_i,
        a_i=eqv.a_i,
        oa_i=state.oa_i,
        a_ip=state.a_ip,
        v_ip=state.v_ip,
        v_il=state.v_il,
        t_su=state.t_su,
        t_si=state.t_si,
        e_s=state.e_s,
        e_i=state.e_i,
        szv_i=state.szv_i,
        a_ip_eff=eqv.a_ip_eff,
        t_i=eqv.t_i,
        t_s=eqv.t_s,
        kn=2,
        config=config,
    )
    state = state._replace(a_i=eqv.a_i, v_s=eqv.v_s, equivalent=eqv, aggregated=agg)
    fluxes_out, outputs = si3_ice_update_flx(
        state,
        fluxes,
        forcing,
        config=config,
        surface_config=surface_config,
        _plant_omit_hfx_res=_plant_omit_hfx_res,
    )
    return SI3O16Result(state=state, fluxes=fluxes_out, outputs=outputs)


def si3_ice_update_tau(
    inputs: SI3StressInputs,
    *,
    config: SI3ORCA2StateConfig,
    ln_drgice_imp: bool,
) -> SI3StressOutputs:
    """Execute the production-deck ``ice_update_tau`` resolved drag arm."""
    if config.nn_fsbc != 1:
        raise ValueError("O16 master record requires resolved nn_fsbc=1")
    h = int(config.nn_hls)
    if h < 1:
        raise ValueError("ice_update_tau requires at least one halo cell")
    full_shape = inputs.u_ice.shape
    full_fields = (
        "uu_kbb", "vv_kbb", "umask_k1", "vmask_k1", "v_ice", "u_oce",
        "v_oce", "drag_io", "at_i", "utau", "vtau", "tmod_io",
    )
    wrong = [name for name in full_fields if getattr(inputs, name).shape != full_shape]
    interior_shape = (full_shape[0] - 2 * h, full_shape[1] - 2 * h)
    wrong.extend(
        name for name in ("taum", "utau_oce", "vtau_oce")
        if getattr(inputs, name).shape != interior_shape
    )
    if wrong:
        raise ValueError("ice_update_tau forcing/config shape mismatch: " + ", ".join(wrong))
    dtype = inputs.u_ice.dtype
    rho0 = jnp.asarray(config.ice_constants.rho_ocean, dtype=dtype)

    zu = sr(sr(inputs.u_ice[1:-1, 1:-1] + inputs.u_ice[:-2, 1:-1]) -
            sr(inputs.u_oce[1:-1, 1:-1] + inputs.u_oce[:-2, 1:-1]))
    zv = sr(sr(inputs.v_ice[1:-1, 1:-1] + inputs.v_ice[1:-1, :-2]) -
            sr(inputs.v_oce[1:-1, 1:-1] + inputs.v_oce[1:-1, :-2]))
    zmod = sr(_QUARTER * sr(sr(zu * zu) + sr(zv * zv)))
    tmod_inner = sr(sr(rho0 * inputs.drag_io[1:-1, 1:-1]) * jnp.sqrt(zmod))
    tmod_io = inputs.tmod_io.at[1:-1, 1:-1].set(tmod_inner)

    sl = (slice(h, -h), slice(h, -h))
    left = (slice(h - 1, -h - 1), slice(h, -h))
    below = (slice(h, -h), slice(h - 1, -h - 1))
    zu_i = sr(sr(inputs.u_ice[sl] + inputs.u_ice[left]) -
              sr(inputs.u_oce[sl] + inputs.u_oce[left]))
    zv_i = sr(sr(inputs.v_ice[sl] + inputs.v_ice[below]) -
              sr(inputs.v_oce[sl] + inputs.v_oce[below]))
    zmod_i = sr(_QUARTER * sr(sr(zu_i * zu_i) + sr(zv_i * zv_i)))
    at_i = inputs.at_i[sl]
    drag_i = inputs.drag_io[sl]
    taum_ice = sr(sr(sr(at_i * rho0) * drag_i) * zmod_i)
    taum = sr(sr(sr(_ONE - at_i) * inputs.taum) + taum_ice)

    utau_oce = inputs.utau[sl]
    vtau_oce = inputs.vtau[sl]
    # The admitted deck resolves ln_drgice_imp=F, hence zflagi=1.  Keeping the
    # resolved logical explicit also preserves the already supported implicit arm.
    zflagi = _ZERO if ln_drgice_imp else _ONE
    pu = sr(inputs.uu_kbb[sl] + inputs.uu_kbb[left])
    pv = sr(inputs.vv_kbb[sl] + inputs.vv_kbb[below])
    urel = sr(sr(inputs.u_ice[sl] + inputs.u_ice[left]) - sr(zflagi * pu))
    vrel = sr(sr(inputs.v_ice[sl] + inputs.v_ice[below]) - sr(zflagi * pv))
    zutau_ice = sr(sr(_HALF * tmod_io[sl]) * urel)
    zvtau_ice = sr(sr(_HALF * tmod_io[sl]) * vrel)
    utau_i = sr(sr(sr(_ONE - at_i) * utau_oce) + sr(at_i * zutau_ice))
    vtau_i = sr(sr(sr(_ONE - at_i) * vtau_oce) + sr(at_i * zvtau_ice))
    utau = inputs.utau.at[sl].set(utau_i)
    vtau = inputs.vtau.at[sl].set(vtau_i)
    return SI3StressOutputs(
        utau=utau,
        vtau=vtau,
        taum=taum,
        tmod_io=tmod_io,
        utau_oce=utau_oce,
        vtau_oce=vtau_oce,
    )


__all__ = (
    "SI3CorrectionResult",
    "SI3FinalizeForcing",
    "SI3FinalizeResponse",
    "SI3Fluxes",
    "SI3O16Result",
    "SI3StressInputs",
    "SI3StressOutputs",
    "SI3UpdateFluxForcing",
    "SI3UpdateFluxOutputs",
    "si3_ice_cor",
    "si3_ice_update_flx",
    "si3_ice_update_tau",
    "si3_o15_step",
    "si3_o16_step",
)
