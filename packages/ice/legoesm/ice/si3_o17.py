"""Typed O17 composition of the admitted ORCA2 SI3 production operators.

This module adds no physical kernel.  It is the line-ordered ``ice_stp`` /
``ice_dyn`` / ``ice_thd`` call graph whose leaves are the O4--O16 kernels.
The deliberately explicit forcing records are also the acquisition boundary:
there is no fallback from a missing NEMO field to a guessed value.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from legoesm.ice.bitz_lipscomb import (
    SI3DHFluxes,
    SI3DOOutputs,
    SI3Frazil,
    si3_ice_thd_do,
    si3_ice_thd_frazil,
)
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG, IceConstantsConfig
from legoesm.ice.dynamics import (
    SI3CGridAEVPConfig,
    SI3CGridAEVPForcing,
    SI3CGridAEVPState,
    SI3CGridMetrics,
    SI3ORCA2FoldProgram,
    SI3ORCA2Strength,
    SI3ORCA2StrengthConfig,
    si3_cgrid_deformation,
    si3_orca2_aevp_solver,
    si3_orca2_ice_strength,
)
from legoesm.ice.itd import (
    SI3IceCorConfig,
    SI3IceCorState,
    si3_ice_cor_dyn,
    si3_ice_itd_rem,
)
from legoesm.ice.ridging import SI3RdgrftConfig, SI3RdgrftResult, SI3RdgrftState, si3_ice_dyn_rdgrft
from legoesm.ice.si3_finalize import (
    SI3FinalizeResponse,
    SI3Fluxes,
    SI3StressInputs,
    SI3UpdateFluxForcing,
    si3_ice_update_tau,
    si3_o15_step,
    si3_o16_step,
)
from legoesm.ice.si3_orca2_state import (
    SI3AdvectionDiagnostics,
    SI3ORCA2AdvectionGrid,
    SI3ORCA2StateConfig,
    SI3State,
    si3_apply_state_halos_orca2,
    si3_dyn_orca2,
    si3_var_agg,
    si3_var_glo2eqv,
)
from legoesm.ice.si3_surface import (
    SI3SurfaceConfig,
    SI3SurfaceOutputs,
    orca2_flx_other_neighbours,
    si3_blk_ice_1,
    si3_blk_ice_2,
    si3_ice_alb,
    si3_ice_flx_other,
    si3_ice_sbc_tau_lbc,
)
from legoesm.ice.si3_thd_pnd import SI3O12Stages, si3_ice_thd_o12_stages

from legoesm import constants


class SI3O17Card(NamedTuple):
    """Every non-array selector used by the frozen O17 production deck."""

    state: SI3ORCA2StateConfig
    aevp: SI3CGridAEVPConfig
    strength: SI3ORCA2StrengthConfig
    fold: SI3ORCA2FoldProgram
    redistribution: SI3RdgrftConfig
    correction: SI3IceCorConfig
    surface: SI3SurfaceConfig
    thermo_namelist: dict
    constants: IceConstantsConfig
    rn_himin: float
    rn_hinew: float
    ln_frazil: bool
    ln_drgice_imp: bool
    differentiable: bool
    core2_precipitation_closure: bool


class SI3O17StepState(NamedTuple):
    ice: SI3State
    aevp: SI3CGridAEVPState


class SI3O17SurfaceForcing(NamedTuple):
    """Raw O7 meteorology/ocean fields, before any O7 product is formed."""

    wndi: jnp.ndarray
    wndj: jnp.ndarray
    theta_air_zt: jnp.ndarray
    q_air_zt: jnp.ndarray
    slp: jnp.ndarray
    rhoa: jnp.ndarray
    cloud_fra: jnp.ndarray
    smask0: jnp.ndarray
    qlw: jnp.ndarray
    precip: jnp.ndarray
    snow: jnp.ndarray
    qsr: jnp.ndarray
    emp: jnp.ndarray
    at_i_b: jnp.ndarray
    a_i_b: jnp.ndarray
    qns_oce: jnp.ndarray
    qsr_oce: jnp.ndarray
    ssu_m: jnp.ndarray
    ssv_m: jnp.ndarray
    utau: jnp.ndarray
    vtau: jnp.ndarray
    frq_m: jnp.ndarray
    e3t_m: jnp.ndarray
    t_bo: jnp.ndarray


class SI3O17LiveForcing(NamedTuple):
    """Complete fields at ``ice_stp`` entry; all children are JAX pytrees."""

    kt: int
    sss_m: jnp.ndarray
    sst_m: jnp.ndarray
    rn_amax_2d: jnp.ndarray
    advection_grid: SI3ORCA2AdvectionGrid
    aevp_forcing: SI3CGridAEVPForcing
    aevp_metrics: SI3CGridMetrics
    ssh_m: jnp.ndarray
    u_ice_b: jnp.ndarray
    v_ice_b: jnp.ndarray
    surface: SI3O17SurfaceForcing
    stress: SI3StressInputs


class SI3O17Trace(NamedTuple):
    """Named call-order boundaries, in the order executed by NEMO."""

    surface_stress: SI3SurfaceOutputs
    strength: SI3ORCA2Strength
    aevp: SI3CGridAEVPState
    deformation: tuple[jnp.ndarray, jnp.ndarray]
    advection: SI3AdvectionDiagnostics
    redistribution: SI3RdgrftResult
    postdyn_correction: SI3IceCorState
    frazil: SI3Frazil
    thermodynamics: SI3O12Stages
    remap: dict
    open_water: SI3DOOutputs
    finalize: SI3FinalizeResponse


def build_orca2_sas_ice_o17_card(*, surface: SI3SurfaceConfig, thermo_namelist: dict) -> SI3O17Card:
    """Write every frozen ORCA2_SAS_ICE selector into one O17 card."""

    if not isinstance(surface, SI3SurfaceConfig):
        raise TypeError("O17 requires an explicit resolved SI3SurfaceConfig")
    if not isinstance(thermo_namelist, dict):
        raise TypeError("O17 requires the explicit resolved thermodynamic namelist")
    cfg = SI3ORCA2StateConfig(
        jpl=5,
        nlay_i=10,
        nlay_s=5,
        nn_icesal=4,
        rn_icesal=4.0,
        rn_sal_gd=5.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_sal_fl=2.0,
        rn_simax=35.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_simin=0.1,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_sinew=0.75,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_himean=2.0,
        rn_himax=99.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        opt_hips=0,
        ln_pnd=True,
        ln_pnd_LEV=True,
        ln_pnd_TOPO=False,
        ln_pnd_lids=True,
        ln_pnd_alb=True,
        rn_apnd_min=0.15,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_apnd_max=0.80,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_pnd_hl_min=0.005,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_pnd_hl_max=0.015,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        nn_snwfra=2,
        ln_icedyn=True,
        ln_icethd=True,
        ln_dynadv_pra=True,
        ln_mppdelay=True,
        nn_fsbc=1,
        rDt_ice=10800.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        nn_hls=2,
        ln_cpl=False,
        ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
    )
    aevp = SI3CGridAEVPConfig(
        scheme="si3_aevp",
        staggering="si3_c_grid",
        dt_s=10800.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        n_subcycles=100,
        eccentricity=2.0,
        creep_limit_s_inv=2.0e-9,
        strength_parameter_pa=2.0e4,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        strength_decay=20.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rho_snow=330.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rho_ice=917.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rho_water=1000.0,
        rho_ocean=constants.rho_ocean_nemo,
        gravity=constants.g_nemo,
        rn_ishlat=2.0,
        halo_width=2,
        category_count=5,
        landfast=False,
        landfast_depth_fraction=0.125,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        landfast_basal_friction_n_m3=15.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        landfast_relaxation_s_inv=1.0e-5,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        landfast_tensile_fraction=0.05,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        convergence_check=0,
    )
    redistribution = SI3RdgrftConfig(
        rn_csrdg=0.5,
        ln_partf_exp=True,
        rn_astar=0.03,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        ln_distf_lin=True,
        ln_distf_exp=False,
        rn_murdg=3.0,
        ln_ridging=True,
        rn_hstar=25.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_porordg=0.3,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_fsnwrdg=0.5,
        rn_fpndrdg=1.0,
        ln_rafting=True,
        rn_hraft=0.75,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_craft=5.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_fsnwrft=0.5,
        rn_fpndrft=1.0,
        ln_rhg_evp_or_vp=True,
        ln_pnd=True,
        ln_pnd_lids=True,
        nn_icesal=4,
        rdt_ice=10800.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rhoi=NEMO_SI3_CONSTANTS_CONFIG.rho_ice,
        rhos=NEMO_SI3_CONSTANTS_CONFIG.rho_snow,
        rhow=constants.rho_water,
        rcp=NEMO_SI3_CONSTANTS_CONFIG.c_ocean,
    )
    correction = SI3IceCorConfig(
        rn_himin=0.1,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_simin=0.1,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_sinew=0.75,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        nn_icesal=4,
        ln_pnd=True,
        ln_pnd_lids=True,
        rdt_ice=10800.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rhoi=NEMO_SI3_CONSTANTS_CONFIG.rho_ice,
        rhos=NEMO_SI3_CONSTANTS_CONFIG.rho_snow,
        rhow=constants.rho_water,
        rt0=NEMO_SI3_CONSTANTS_CONFIG.T0,
    )
    return SI3O17Card(
        state=cfg,
        aevp=aevp,
        strength=SI3ORCA2StrengthConfig(
            ln_str_h79=True,
            rn_pstar=2.0e4,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
            rn_crhg=20.0,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
            ln_str_smooth=True,
        ),
        fold=SI3ORCA2FoldProgram(nn_hls=2, nfd_full_when_absent=True),
        redistribution=redistribution,
        correction=correction,
        surface=surface,
        thermo_namelist=dict(thermo_namelist),
        constants=NEMO_SI3_CONSTANTS_CONFIG,
        rn_himin=0.1,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        rn_hinew=0.1,  # coeff-ok: frozen ORCA2_SAS_ICE deck selector
        ln_frazil=False,
        ln_drgice_imp=False,
        differentiable=True,
        core2_precipitation_closure=False,
    )


def _state_from_rdgrft(state: SI3State, rdg: SI3RdgrftState) -> SI3State:
    return state._replace(
        **{
            name: getattr(rdg, name)
            for name in ("a_i", "v_i", "v_s", "oa_i", "a_ip", "v_ip", "v_il", "e_s", "e_i", "szv_i")
        }
    )


def _state_from_cor(state: SI3State, cor: SI3IceCorState) -> SI3State:
    return state._replace(
        a_i=cor.a_i,
        v_i=cor.v_i,
        v_s=cor.v_s,
        oa_i=cor.oa_i,
        a_ip=cor.a_ip,
        v_ip=cor.v_ip,
        v_il=cor.v_il,
        e_s=cor.e_s,
        e_i=cor.e_i,
        szv_i=cor.szv_i,
        t_su=cor.t_su,
        equivalent=state.equivalent._replace(
            h_i=cor.h_i,
            h_s=cor.h_s,
            h_ip=cor.h_ip,
            h_il=cor.h_il,
            sz_i=cor.sz_i,
            t_i=cor.t_i,
            t_s=cor.t_s,
            a_i=cor.a_i,
            v_s=cor.v_s,
        ),
        aggregated=state.aggregated._replace(at_i=cor.at_i, vt_i=cor.vt_i, ato_i=cor.ato_i),
    )


def _rdgrft_state(state: SI3State, fluxes: SI3Fluxes) -> SI3RdgrftState:
    return SI3RdgrftState(
        a_i=state.a_i,
        v_i=state.v_i,
        v_s=state.v_s,
        oa_i=state.oa_i,
        a_ip=state.a_ip,
        v_ip=state.v_ip,
        v_il=state.v_il,
        e_s=state.e_s,
        e_i=state.e_i,
        szv_i=state.szv_i,
        at_i=state.aggregated.at_i,
        ato_i=state.aggregated.ato_i,
        wfx_dyn=fluxes.wfx_dyn,
        sfx_dyn=fluxes.sfx_dyn,
        hfx_dyn=fluxes.hfx_dyn,
        wfx_snw_dyn=jnp.zeros_like(fluxes.wfx_dyn),
        wfx_pnd=fluxes.wfx_pnd,
    )


def _correction_state(state: SI3State, fluxes: SI3Fluxes) -> SI3IceCorState:
    eq = state.equivalent
    return SI3IceCorState(
        a_i=state.a_i,
        v_i=state.v_i,
        v_s=state.v_s,
        oa_i=state.oa_i,
        a_ip=state.a_ip,
        v_ip=state.v_ip,
        v_il=state.v_il,
        e_s=state.e_s,
        e_i=state.e_i,
        szv_i=state.szv_i,
        at_i=state.aggregated.at_i,
        ato_i=state.aggregated.ato_i,
        sz_i=eq.sz_i,
        t_su=state.t_su,
        t_i=eq.t_i,
        t_s=eq.t_s,
        h_i=eq.h_i,
        h_s=eq.h_s,
        h_ip=eq.h_ip,
        h_il=eq.h_il,
        vt_i=state.aggregated.vt_i,
        wfx_res=fluxes.wfx_res,
        sfx_res=fluxes.sfx_res,
        hfx_res=fluxes.hfx_res,
    )


def _thermo_entries(
    state: SI3State,
    forcing: SI3O17LiveForcing,
    surface: SI3SurfaceOutputs,
    fluxes: SI3Fluxes,
    nn_hls: int,
) -> tuple[dict, ...]:
    eq = state.equivalent
    h = int(nn_hls)
    entries = []
    for jl in range(state.a_i.shape[-1]):
        e = dict(
            h_i=eq.h_i[..., jl],
            h_s=eq.h_s[..., jl],
            t_su=state.t_su[..., jl],
            t_i=eq.t_i[..., :, jl],
            t_s=eq.t_s[..., :, jl],
            e_i=state.e_i[..., :, jl],
            e_s=state.e_s[..., :, jl],
            sz_i=eq.sz_i[..., :, jl],
            s_i=eq.s_i[..., jl],
            a_i=state.a_i[..., jl],
            o_i=eq.o_i[..., jl],
            at_i=state.aggregated.at_i,
            sss=forcing.sss_m,
            sst=forcing.sst_m,
            t_bo=_scatter(jnp.zeros_like(forcing.sst_m), forcing.surface.t_bo, h),
            qns_ice=_scatter(
                jnp.zeros_like(state.a_i[..., jl]), surface.blk_ice_2.qns_ice[..., jl], h
            ),
            qsr_ice=_scatter(
                jnp.zeros_like(state.a_i[..., jl]), surface.blk_ice_2.qsr_ice[..., jl], h
            ),
            dqns_ice=_scatter(
                jnp.zeros_like(state.a_i[..., jl]), surface.blk_ice_2.dqns_ice[..., jl], h
            ),
            qtr_ice_top=_scatter(
                jnp.zeros_like(state.a_i[..., jl]), surface.blk_ice_2.qtr_ice_top[..., jl], h
            ),
            qsb_ice_bot=_scatter(jnp.zeros_like(forcing.sst_m), surface.flx_other.qsb_ice_bot, h),
            fhld=_scatter(jnp.zeros_like(forcing.sst_m), surface.flx_other.fhld, h),
            frq_m=_scatter(jnp.zeros_like(forcing.sst_m), forcing.surface.frq_m, h),
            sprecip=_scatter(jnp.zeros_like(forcing.sst_m), surface.blk_ice_2.sprecip, h),
            qprec_ice=_scatter(jnp.zeros_like(forcing.sst_m), surface.blk_ice_2.qprec_ice, h),
            evap_ice=_scatter(
                jnp.zeros_like(state.a_i[..., jl]), surface.blk_ice_2.evap_ice[..., jl], h
            ),
            hfx_err_dif=fluxes.hfx_err_dif,
            hfx_dif=fluxes.hfx_dif,
        )
        entries.append(e)
    return tuple(entries)


def _pack_categories(stages: SI3O12Stages, field: str) -> jnp.ndarray:
    return jnp.stack([getattr(c, field) for c in stages.categories], axis=-1)


def _scatter(full: jnp.ndarray, interior: jnp.ndarray, h: int) -> jnp.ndarray:
    if not h:
        return interior
    return full.at[h:-h, h:-h, ...].set(interior)


def si3_o17_step(
    entry: SI3O17StepState,
    forcing: SI3O17LiveForcing,
    card: SI3O17Card,
    *,
    _plant_swap_rheology_advection: bool = False,
    _plant_drop_redistribution: bool = False,
) -> tuple[SI3O17StepState, SI3O17Trace]:
    """One exact production call graph in NEMO ``ice_stp`` order."""

    if card.core2_precipitation_closure is not False:
        raise ValueError("O17 freezes core2_precipitation_closure=False")
    if not (card.state.ln_icedyn and card.state.ln_icethd and card.state.ln_dynadv_pra):
        raise ValueError(
            "O17 requires explicit live dynamics, Prather advection, and thermodynamics"
        )
    state = entry.ice
    sf = forcing.surface
    h = int(card.state.nn_hls)
    owned = (slice(h, -h), slice(h, -h)) if h else (slice(None), slice(None))

    b1 = si3_blk_ice_1(
        wndi=sf.wndi,
        wndj=sf.wndj,
        theta_air_zt=sf.theta_air_zt,
        q_air_zt=sf.q_air_zt,
        slp=sf.slp,
        tm_su=state.aggregated.tm_su[owned],
        rhoa=sf.rhoa,
        config=card.surface,
    )
    utau_ice, vtau_ice = si3_ice_sbc_tau_lbc(b1.utau_ice, b1.vtau_ice, card.fold)

    strength = si3_orca2_ice_strength(
        a_i=state.a_i,
        v_i=state.v_i,
        tmask=forcing.aevp_forcing.tmask_t,
        config=card.strength,
        fold=card.fold,
    )
    aevp_forcing = forcing.aevp_forcing._replace(
        concentration_t=strength.at_i,
        ice_volume_t=state.aggregated.vt_i,
        snow_volume_t=state.aggregated.vt_s,
        pond_volume_t=state.aggregated.vt_ip,
        lid_volume_t=state.aggregated.vt_il,
        air_stress_u_t=utau_ice,
        air_stress_v_t=vtau_ice,
    )
    aevp_entry = entry.aevp._replace(u_ice_u=state.u_ice, v_ice_v=state.v_ice)
    aevp = si3_orca2_aevp_solver(
        aevp_entry,
        aevp_forcing,
        forcing.aevp_metrics,
        card.aevp,
        strength_t=strength.strength,
        u_ice_b=forcing.u_ice_b,
        v_ice_b=forcing.v_ice_b,
        ssh_m=forcing.ssh_m,
        snwice_mass=state.snwice_mass,
        snwice_mass_b=state.snwice_mass_b,
        ln_ice_embd=False,
        fold=card.fold,
        differentiable=card.differentiable,
    )
    state = state._replace(u_ice=aevp.u_ice_u, v_ice=aevp.v_ice_v)
    zeros = jnp.zeros_like(state.u_ice)
    # icestp.diag_set0 assigns +0 to every SI3 flux accumulator at step
    # entry.  They are not live forcing and cannot be injected mid-step.
    step_fluxes = SI3Fluxes(*(jnp.zeros_like(forcing.sss_m) for _ in SI3Fluxes._fields))
    if _plant_swap_rheology_advection:
        adv_state, adv = si3_dyn_orca2(
            entry.ice,
            forcing.advection_grid,
            kt=forcing.kt,
            sss_m=forcing.sss_m,
            sst_m=forcing.sst_m,
            rn_uice=None,
            rn_vice=None,
            config=card.state,
            wfx_res=zeros,
            hfx_res=zeros,
            sfx_res=zeros,
            diag_adv_mass=zeros,
            diag_adv_salt=zeros,
            diag_adv_heat=zeros,
        )
        state = adv_state._replace(u_ice=aevp.u_ice_u, v_ice=aevp.v_ice_v)
    else:
        state, adv = si3_dyn_orca2(
            state,
            forcing.advection_grid,
            kt=forcing.kt,
            sss_m=forcing.sss_m,
            sst_m=forcing.sst_m,
            rn_uice=None,
            rn_vice=None,
            config=card.state,
            wfx_res=zeros,
            hfx_res=zeros,
            sfx_res=zeros,
            diag_adv_mass=zeros,
            diag_adv_salt=zeros,
            diag_adv_heat=zeros,
        )
    divu, delta = si3_cgrid_deformation(
        aevp, aevp_forcing, forcing.aevp_metrics, card.aevp, fold=card.fold
    )
    rdg_entry = _rdgrft_state(state, step_fluxes)
    if _plant_drop_redistribution:
        rdg = SI3RdgrftResult(
            rdg_entry,
            jnp.zeros_like(zeros, bool),
            jnp.asarray(0),
            jnp.zeros_like(zeros, jnp.int32),
            jnp.zeros_like(zeros, bool),
            jnp.zeros_like(zeros, bool),
        )
    else:
        rdg = si3_ice_dyn_rdgrft(
            rdg_entry,
            divu_i=divu,
            delta_i=delta,
            sst_m=forcing.sst_m,
            sss_m=forcing.sss_m,
            hi_max=state.bounds.hi_max,
            nn_hls=card.state.nn_hls,
            config=card.redistribution,
        )
    state = _state_from_rdgrft(state, rdg.state)
    fluxes = step_fluxes._replace(
        wfx_dyn=rdg.state.wfx_dyn,
        sfx_dyn=rdg.state.sfx_dyn,
        hfx_dyn=rdg.state.hfx_dyn,
        wfx_pnd=rdg.state.wfx_pnd,
        wfx_res=adv.wfx_res,
        hfx_res=adv.hfx_res,
        sfx_res=adv.sfx_res,
    )
    cor_entry = _correction_state(state, fluxes)
    cor = si3_ice_cor_dyn(
        cor_entry,
        rn_amax_2d=forcing.rn_amax_2d,
        sss_m=forcing.sss_m,
        sst_m=forcing.sst_m,
        hi_max=state.bounds.hi_max,
        hi_mean=state.bounds.hi_mean,
        nn_hls=card.state.nn_hls,
        config=card.correction,
    )
    state = _state_from_cor(state, cor)
    # icedyn.f90's final full LBC follows redistribution and ice_cor(kt,1).
    # si3_dyn_orca2 also exchanges its advection exit for its standalone O4
    # record; repeating the admitted exchange here restores the production
    # placement after the O6 interior mutations.
    state = si3_apply_state_halos_orca2(state, config=card.state)
    fluxes = fluxes._replace(wfx_res=cor.wfx_res, hfx_res=cor.hfx_res, sfx_res=cor.sfx_res)
    # icestp.f90 immediately converts the post-dynamics global arrays and
    # aggregates them before the O7 albedo/flux calls.  These are the already
    # admitted O2/O4 state operators, not acquired mid-step forcing.
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
        sss_m=forcing.sss_m,
        config=card.state,
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
        kn=1,
        config=card.state,
    )
    state = state._replace(a_i=eqv.a_i, v_s=eqv.v_s, equivalent=eqv, aggregated=agg)

    alb = si3_ice_alb(
        t_su=state.t_su[owned],
        h_i=state.equivalent.h_i[owned],
        h_s=state.equivalent.h_s[owned],
        a_ip_eff=state.equivalent.a_ip_eff[owned],
        h_ip=state.equivalent.h_ip[owned],
        h_il=state.equivalent.h_il[owned],
        cloud_fra=sf.cloud_fra,
        smask0=sf.smask0,
        config=card.surface,
    )
    b2 = si3_blk_ice_2(
        t_su=state.t_su[owned],
        h_s=state.equivalent.h_s[owned],
        h_i=state.equivalent.h_i[owned],
        alb_ice=alb,
        theta_air_zt=sf.theta_air_zt,
        q_air_zt=sf.q_air_zt,
        slp=sf.slp,
        qlw=sf.qlw,
        precip=sf.precip,
        snow=sf.snow,
        qsr=sf.qsr,
        rhoa=sf.rhoa,
        wndm_ice=b1.wndm_ice,
        Ch_ice=b1.Ch_ice,
        Ce_ice=b1.Ce_ice,
        theta_zu_i=b1.theta_zu_i,
        q_zu_i=b1.q_zu_i,
        smask0=sf.smask0,
        sst_m=forcing.sst_m[owned],
        emp=sf.emp,
        at_i_b=sf.at_i_b,
        a_i_b=sf.a_i_b,
        qns_oce=sf.qns_oce,
        qsr_oce=sf.qsr_oce,
        cloud_fra=sf.cloud_fra,
        config=card.surface,
    )
    other = si3_ice_flx_other(
        **orca2_flx_other_neighbours(
            u_ice=state.u_ice,
            v_ice=state.v_ice,
            ssu_m=sf.ssu_m,
            ssv_m=sf.ssv_m,
            nn_hls=card.state.nn_hls,
        ),
        utau=sf.utau,
        vtau=sf.vtau,
        smask0=sf.smask0,
        at_i=state.aggregated.at_i[owned],
        at_i_b=sf.at_i_b,
        vt_i=state.aggregated.vt_i[owned],
        qsr_oce=sf.qsr_oce,
        qns_oce=sf.qns_oce,
        qemp_oce=b2.qemp_oce,
        frq_m=sf.frq_m,
        e3t_m=sf.e3t_m,
        sst_m=forcing.sst_m[owned],
        t_bo=sf.t_bo,
        rn_amax_2d=forcing.rn_amax_2d[owned],
        config=card.surface,
    )
    surface = SI3SurfaceOutputs(b1, utau_ice, vtau_ice, alb, b2, other)

    frazil = si3_ice_thd_frazil(other.qlead, rn_hinew=card.rn_hinew, ln_frazil=card.ln_frazil)
    dh_fluxes = SI3DHFluxes(**{name: getattr(fluxes, name) for name in SI3DHFluxes._fields})
    stages = si3_ice_thd_o12_stages(
        category_entries=_thermo_entries(state, forcing, surface, fluxes, card.state.nn_hls),
        category_fluxes=dh_fluxes,
        sfx_bri=fluxes.sfx_bri,
        sfx_lam=fluxes.sfx_lam,
        hfx_thd=fluxes.hfx_thd,
        wfx_lam=fluxes.wfx_lam,
        a_ip=state.a_ip,
        v_ip=state.v_ip,
        h_ip=state.equivalent.h_ip,
        v_il=state.v_il,
        h_il=state.equivalent.h_il,
        a_ip_frac=state.equivalent.a_ip_frac,
        wfx_pnd=fluxes.wfx_pnd,
        namelist=card.thermo_namelist,
        rDt_ice=card.state.rDt_ice,
        r1_Dt_ice=1.0 / card.state.rDt_ice,
        r1_nlay_i=1.0 / card.state.nlay_i,
        r1_nlay_s=1.0 / card.state.nlay_s,
        rho_i=card.constants.rho_ice,
        rho_s=card.constants.rho_snow,
        rho_w=constants.rho_water,
        gravity=constants.g_nemo,
        nn_hls=card.state.nn_hls,
        constants=card.constants,
    )
    carried = stages.categories[-1].o10.o9.dh.fluxes
    carried_updates = {
        name: getattr(carried, name) for name in carried._fields if name in SI3Fluxes._fields
    }
    last_da = stages.categories[-1].da
    if last_da is not None:
        carried_updates.update(
            sfx_lam=last_da.sfx_lam,
            hfx_thd=last_da.hfx_thd,
            wfx_lam=last_da.wfx_lam,
        )
    fluxes = fluxes._replace(
        **carried_updates,
        wfx_pnd=stages.pnd.wfx_pnd,
        qlead=_scatter(fluxes.qlead, other.qlead, h),
        qsb_ice_bot=_scatter(fluxes.qsb_ice_bot, other.qsb_ice_bot, h),
        fhld=_scatter(fluxes.fhld, other.fhld, h),
    )
    thermo_state = state._replace(
        a_i=_pack_categories(stages, "a_i"),
        v_i=_pack_categories(stages, "v_i"),
        v_s=_pack_categories(stages, "v_s"),
        sv_i=_pack_categories(stages, "sv_i"),
        oa_i=_pack_categories(stages, "oa_i"),
        t_su=_pack_categories(stages, "t_su"),
        e_i=_pack_categories(stages, "e_i"),
        e_s=_pack_categories(stages, "e_s"),
        szv_i=_pack_categories(stages, "szv_i"),
        a_ip=stages.pnd.a_ip,
        v_ip=stages.pnd.v_ip,
        v_il=stages.pnd.v_il,
        equivalent=state.equivalent._replace(
            h_i=_pack_categories(stages, "h_i"),
            h_s=_pack_categories(stages, "h_s"),
            h_ip=stages.pnd.h_ip,
            h_il=stages.pnd.h_il,
            a_ip_frac=stages.pnd.a_ip_frac,
            s_i=_pack_categories(stages, "s_i"),
            sz_i=_pack_categories(stages, "sz_i"),
            t_i=_pack_categories(stages, "t_i"),
            t_s=_pack_categories(stages, "t_s"),
            a_i=_pack_categories(stages, "a_i"),
            v_s=_pack_categories(stages, "v_s"),
        ),
        aggregated=state.aggregated._replace(
            at_i=stages.pnd.at_i,
        ),
    )
    remap = si3_ice_itd_rem(
        h_i_b=state.equivalent.h_i[owned],
        a_i_b=state.a_i[owned],
        h_i=_pack_categories(stages, "h_i")[owned],
        a_i=thermo_state.a_i[owned],
        v_i=thermo_state.v_i[owned],
        v_s=thermo_state.v_s[owned],
        sv_i=thermo_state.sv_i[owned],
        oa_i=thermo_state.oa_i[owned],
        a_ip=thermo_state.a_ip[owned],
        v_ip=thermo_state.v_ip[owned],
        v_il=thermo_state.v_il[owned],
        t_su=thermo_state.t_su[owned],
        e_s=thermo_state.e_s[owned],
        e_i=thermo_state.e_i[owned],
        szv_i=thermo_state.szv_i[owned],
        hi_max=state.bounds.hi_max,
        rn_amax_2d=forcing.rn_amax_2d[owned],
        nn_icesal=card.state.nn_icesal,
        ln_pnd=card.state.ln_pnd,
        ln_pnd_lids=card.state.ln_pnd_lids,
        epsi10=1.0e-10,
        epsi20=1.0e-20,
        rt0=card.constants.T0,
    )
    state = thermo_state._replace(
        **{
            k: _scatter(getattr(thermo_state, k), remap[k], h)
            for k in (
                "a_i",
                "v_i",
                "v_s",
                "sv_i",
                "oa_i",
                "a_ip",
                "v_ip",
                "v_il",
                "t_su",
                "e_s",
                "e_i",
                "szv_i",
            )
        },
        equivalent=thermo_state.equivalent._replace(
            h_i=_scatter(thermo_state.equivalent.h_i, remap["h_i"], h),
            a_i=_scatter(thermo_state.equivalent.a_i, remap["a_i"], h),
        ),
    )
    open_water = si3_ice_thd_do(
        a_i=state.a_i[owned],
        v_i=state.v_i[owned],
        sv_i=state.sv_i[owned],
        e_i=state.e_i[owned],
        szv_i=state.szv_i[owned],
        qlead=other.qlead,
        t_bo=sf.t_bo,
        sfx_opw=fluxes.sfx_opw[owned],
        wfx_opw=fluxes.wfx_opw[owned],
        ht_i_new=frazil.ht_i_new,
        fraz_frac=frazil.fraz_frac,
        hfx_thd=fluxes.hfx_thd[owned],
        hfx_opw=fluxes.hfx_opw[owned],
        rn_amax_2d=forcing.rn_amax_2d[owned],
        sss_m=forcing.sss_m[owned],
        hi_max=state.bounds.hi_max,
        nn_icesal=card.state.nn_icesal,
        rn_sinew=card.state.rn_sinew,
        r1_Dt_ice=1.0 / card.state.rDt_ice,
        r1_nlay_i=1.0 / card.state.nlay_i,
        constants=card.constants,
    )
    state = state._replace(
        a_i=_scatter(state.a_i, open_water.a_i, h),
        v_i=_scatter(state.v_i, open_water.v_i, h),
        sv_i=_scatter(state.sv_i, open_water.sv_i, h),
        e_i=_scatter(state.e_i, open_water.e_i, h),
        szv_i=_scatter(state.szv_i, open_water.szv_i, h),
    )
    fluxes = fluxes._replace(
        wfx_pnd=stages.pnd.wfx_pnd,
        sfx_opw=_scatter(fluxes.sfx_opw, open_water.sfx_opw, h),
        wfx_opw=_scatter(fluxes.wfx_opw, open_water.wfx_opw, h),
        hfx_thd=_scatter(fluxes.hfx_thd, open_water.hfx_thd, h),
        hfx_opw=_scatter(fluxes.hfx_opw, open_water.hfx_opw, h),
    )
    correction = si3_o15_step(
        state,
        fluxes,
        sss_m=forcing.sss_m,
        sst_m=forcing.sst_m,
        rn_amax_2d=forcing.rn_amax_2d,
        rn_himin=card.rn_himin,
        config=card.state,
    )
    owned = (slice(h, -h), slice(h, -h)) if h else (slice(None), slice(None))
    owned_fluxes = SI3Fluxes(*(x[owned] for x in correction.fluxes))
    qtr_bot = jnp.stack([c.o10.o9.zdf.qtr_ice_bot for c in stages.categories], axis=-1)[owned]
    qcn_top = jnp.stack([c.o10.o9.zdf.qcn_ice_top for c in stages.categories], axis=-1)[owned]
    qcn_bot = jnp.stack([c.o10.o9.zdf.qcn_ice_bot for c in stages.categories], axis=-1)[owned]
    qml_ice = jnp.stack([c.o10.o9.dh.qml_ice for c in stages.categories], axis=-1)[owned]
    zero2 = jnp.zeros_like(sf.smask0)
    zero3 = jnp.zeros_like(sf.a_i_b)
    update = SI3UpdateFluxForcing(
        **{
            name: (
                zero3
                if name
                in {
                    "a_i_b",
                    "qsr_ice",
                    "qns_ice",
                    "qtr_ice_bot",
                    "qevap_ice",
                    "qcn_ice",
                    "qcn_ice_top",
                    "qcn_ice_bot",
                    "qml_ice",
                    "qtr_ice_top",
                    "alb_ice",
                }
                else zero2
            )
            for name in SI3UpdateFluxForcing._fields
        }
    )._replace(
        qsr_oce=sf.qsr_oce,
        qns_oce=sf.qns_oce,
        qemp_oce=b2.qemp_oce,
        qemp_ice=b2.qemp_ice,
        qns_ice=b2.qns_ice,
        emp_oce=b2.emp_oce,
        emp_ice=b2.emp_ice,
        qns_tot=b2.qns_tot,
        qsr_tot=b2.qsr_tot,
        frq_m=sf.frq_m,
        a_i_b=sf.a_i_b,
        at_i_b=sf.at_i_b,
        qsr_ice=b2.qsr_ice,
        qtr_ice_top=b2.qtr_ice_top,
        qtr_ice_bot=qtr_bot,
        qevap_ice=b2.qevap_ice,
        qcn_ice=jnp.zeros_like(qtr_bot),
        qcn_ice_top=qcn_top,
        qcn_ice_bot=qcn_bot,
        qml_ice=qml_ice,
        alb_ice=alb,
        cloud_fra=sf.cloud_fra,
        smask0=sf.smask0,
        qsr=zero2,
        sfx=zero2,
        tmask=forcing.aevp_forcing.tmask_t,
        snwice_mass=state.snwice_mass,
        snwice_mass_b=state.snwice_mass_b,
        wfx_snw_dyn=rdg.state.wfx_snw_dyn[owned],
    )
    final = si3_o16_step(
        correction.state,
        owned_fluxes,
        update,
        sss_m=forcing.sss_m,
        config=card.state,
        surface_config=card.surface,
    )
    stress_inputs = forcing.stress._replace(
        u_ice=final.state.u_ice,
        v_ice=final.state.v_ice,
        at_i=final.state.aggregated.at_i,
    )
    stress = si3_ice_update_tau(stress_inputs, config=card.state, ln_drgice_imp=card.ln_drgice_imp)
    finalize = SI3FinalizeResponse(correction, final, stress)
    return SI3O17StepState(final.state, aevp), SI3O17Trace(
        surface,
        strength,
        aevp,
        (divu, delta),
        adv,
        rdg,
        cor,
        frazil,
        stages,
        remap,
        open_water,
        finalize,
    )


__all__ = (
    "build_orca2_sas_ice_o17_card",
    "SI3O17Card",
    "SI3O17LiveForcing",
    "SI3O17StepState",
    "SI3O17SurfaceForcing",
    "SI3O17Trace",
    "si3_o17_step",
)
