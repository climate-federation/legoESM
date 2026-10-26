"""Literal NEMO ORCA2 standalone surface means and post-ice surface terms.

S0-M follows ``sbcssm.f90:83-176``.  S2 begins from the explicit post-S3
boundary and follows ``sbcmod.f90:517-570`` through runoff, SSS restoring,
freshwater-budget correction, and native-face stress construction.  The S3
``sbcice_if`` operation is deliberately outside this module.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.source_rounding import nemo_source_round as _sr
from legoesm.ocean.eos import nemo_potential_temperature_from_conservative
from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    NemoRunoffSurface,
    nemo_orca2_surface_runoff,
)
from legoesm.ocean.forcing.sss_restoring import nemo_orca2_sss_restoring
from legoesm.ocean.state import OceanSurfaceForcing


class NemoSurfaceMeans(NamedTuple):
    ssu: jnp.ndarray
    ssv: jnp.ndarray
    sst: jnp.ndarray
    sss: jnp.ndarray
    ssh: jnp.ndarray
    e3t: jnp.ndarray
    frq: jnp.ndarray


class NemoS2Result(NamedTuple):
    runoff: NemoRunoffSurface
    erp: jnp.ndarray
    qrp: jnp.ndarray
    qns: jnp.ndarray
    qsr: jnp.ndarray
    emp: jnp.ndarray
    sfx: jnp.ndarray
    taum: jnp.ndarray
    utau: jnp.ndarray
    vtau: jnp.ndarray
    utau_u: jnp.ndarray
    vtau_v: jnp.ndarray
    penetrative_qsr: jnp.ndarray


def _surface_temperature(temperature, salinity, use_conservative_temperature):
    temperature = _sr(jnp.asarray(temperature, dtype=jnp.float64))
    salinity = _sr(jnp.asarray(salinity, dtype=jnp.float64))
    if use_conservative_temperature:
        return _sr(nemo_potential_temperature_from_conservative(temperature, salinity))
    return temperature


def nemo_sbc_ssm_step(
    prior,
    *,
    kt,
    nit000,
    forcing_stride,
    restart_means_present,
    use_conservative_temperature,
    remove_inverse_barometer,
    u_before,
    v_before,
    temperature_now,
    salinity_now,
    ssh_now,
    ssh_inverse_barometer,
    e3t_reference,
    r3t_now,
    surface_mask,
    shortwave_fraction_top,
):
    """One every-step ``sbc_ssm`` update with explicit prior mean state."""
    if not isinstance(forcing_stride, jax.core.Tracer) and forcing_stride < 2:
        raise ValueError("this stateful S0-M arm requires forcing_stride >= 2")
    u = _sr(jnp.asarray(u_before, dtype=jnp.float64))
    v = _sr(jnp.asarray(v_before, dtype=jnp.float64))
    salinity = _sr(jnp.asarray(salinity_now, dtype=jnp.float64))
    temperature = _surface_temperature(
        temperature_now, salinity, use_conservative_temperature
    )
    ssh = _sr(jnp.asarray(ssh_now, dtype=jnp.float64))
    ssh_ib = _sr(jnp.asarray(ssh_inverse_barometer, dtype=jnp.float64))
    if remove_inverse_barometer:
        ssh_instant = _sr(ssh - ssh_ib)
    else:
        ssh_instant = ssh
    e3t_0 = _sr(jnp.asarray(e3t_reference, dtype=jnp.float64))
    r3t = _sr(jnp.asarray(r3t_now, dtype=jnp.float64))
    tmask = _sr(jnp.asarray(surface_mask, dtype=jnp.float64))
    one_plus = _sr(_sr(1.0) + _sr(r3t * tmask))
    e3t_instant = _sr(e3t_0 * one_plus)
    frq = _sr(jnp.asarray(shortwave_fraction_top, dtype=jnp.float64))

    first_from_rest = jnp.logical_and(
        jnp.asarray(kt) == jnp.asarray(nit000),
        jnp.logical_not(jnp.asarray(restart_means_present)),
    )
    reset = jnp.mod(jnp.asarray(kt) - 2, jnp.asarray(forcing_stride)) == 0
    zcoef_initial = _sr(jnp.asarray(forcing_stride - 1, dtype=jnp.float64))
    zero = _sr(jnp.zeros_like(temperature))
    instant = NemoSurfaceMeans(u, v, temperature, salinity, ssh_instant, e3t_instant, frq)
    base = NemoSurfaceMeans(
        *(
            _sr(
                jnp.where(
                    first_from_rest,
                    _sr(zcoef_initial * value),
                    jnp.where(
                        reset,
                        zero,
                        _sr(jnp.asarray(previous, dtype=jnp.float64)),
                    ),
                )
            )
            for value, previous in zip(instant, prior, strict=True)
        )
    )

    accumulated = NemoSurfaceMeans(
        _sr(base.ssu + u),
        _sr(base.ssv + v),
        _sr(base.sst + temperature),
        _sr(base.sss + salinity),
        _sr(base.ssh + ssh_instant),
        _sr(base.e3t + e3t_instant),
        _sr(base.frq + frq),
    )
    divide = jnp.mod(jnp.asarray(kt) - 1, jnp.asarray(forcing_stride)) == 0
    zcoef = _sr(
        _sr(1.0) / _sr(jnp.asarray(forcing_stride, dtype=jnp.float64))
    )
    return NemoSurfaceMeans(
        *(_sr(jnp.where(divide, _sr(value * zcoef), value)) for value in accumulated)
    )


def nemo_t_stress_to_uv(
    utau,
    vtau,
    utau_east,
    vtau_north,
    tmask,
    tmask_east,
    tmask_north,
    umask,
    vmask,
):
    """``sbcmod`` T-stress to native U/V rotation with explicit halo peers."""
    utau = _sr(jnp.asarray(utau, dtype=jnp.float64))
    vtau = _sr(jnp.asarray(vtau, dtype=jnp.float64))
    utau_east = _sr(jnp.asarray(utau_east, dtype=jnp.float64))
    vtau_north = _sr(jnp.asarray(vtau_north, dtype=jnp.float64))
    tmask = _sr(jnp.asarray(tmask, dtype=jnp.float64))
    tmask_east = _sr(jnp.asarray(tmask_east, dtype=jnp.float64))
    tmask_north = _sr(jnp.asarray(tmask_north, dtype=jnp.float64))
    umask = _sr(jnp.asarray(umask, dtype=jnp.float64))
    vmask = _sr(jnp.asarray(vmask, dtype=jnp.float64))
    u_sum = _sr(utau + utau_east)
    u_mean = _sr(_sr(0.5) * u_sum)
    u_unmask = _sr(_sr(2.0) - umask)
    u_wet = _sr(jnp.maximum(tmask, tmask_east))
    utau_u = _sr(_sr(u_mean * u_unmask) * u_wet)
    v_sum = _sr(vtau + vtau_north)
    v_mean = _sr(_sr(0.5) * v_sum)
    v_unmask = _sr(_sr(2.0) - vmask)
    v_wet = _sr(jnp.maximum(tmask, tmask_north))
    vtau_v = _sr(_sr(v_mean * v_unmask) * v_wet)
    return utau_u, vtau_v


def nemo_orca2_s2(
    *,
    kt,
    forcing_stride,
    post_ice_utau,
    post_ice_vtau,
    post_ice_utau_east,
    post_ice_vtau_north,
    post_ice_taum,
    post_ice_qsr,
    post_ice_qns,
    post_ice_emp,
    post_ice_sfx,
    runoff_now,
    prior_runoff_mass,
    prior_runoff_temperature_content,
    prior_runoff_salinity_content,
    sst_mean,
    sss_mean,
    sss_target,
    runoff_mouth_mask,
    surface_mask,
    tmask_east,
    tmask_north,
    umask,
    vmask,
    freshwater_budget_correction,
    runoff_factor,
    reference_density,
    restoring_mm_per_day,
    restoring_bound_mm_per_day,
    seconds_per_day,
    salinity_floor,
    heat_capacity,
):
    """Execute the deck's S2 operations from an explicit post-S3 boundary."""
    qns = _sr(jnp.asarray(post_ice_qns, dtype=jnp.float64))
    qsr = _sr(jnp.asarray(post_ice_qsr, dtype=jnp.float64))
    emp = _sr(jnp.asarray(post_ice_emp, dtype=jnp.float64))
    sfx = _sr(jnp.asarray(post_ice_sfx, dtype=jnp.float64))
    taum = _sr(jnp.asarray(post_ice_taum, dtype=jnp.float64))
    utau = _sr(jnp.asarray(post_ice_utau, dtype=jnp.float64))
    vtau = _sr(jnp.asarray(post_ice_vtau, dtype=jnp.float64))
    smask = _sr(jnp.asarray(surface_mask, dtype=jnp.float64))
    sst = _sr(jnp.asarray(sst_mean, dtype=jnp.float64))

    updated_runoff = nemo_orca2_surface_runoff(
        runoff_now,
        sst,
        smask,
        runoff_factor=runoff_factor,
        reference_density=reference_density,
    )
    update = jnp.mod(jnp.asarray(kt) - 1, jnp.asarray(forcing_stride)) == 0
    prior_runoff_mass = _sr(jnp.asarray(prior_runoff_mass, dtype=jnp.float64))
    prior_runoff_temperature_content = _sr(
        jnp.asarray(prior_runoff_temperature_content, dtype=jnp.float64)
    )
    prior_runoff_salinity_content = _sr(
        jnp.asarray(prior_runoff_salinity_content, dtype=jnp.float64)
    )
    runoff = NemoRunoffSurface(
        _sr(jnp.where(update, updated_runoff.mass_flux, prior_runoff_mass)),
        _sr(jnp.where(
            update,
            updated_runoff.temperature_content,
            prior_runoff_temperature_content,
        )),
        _sr(jnp.where(
            update,
            updated_runoff.salinity_content,
            prior_runoff_salinity_content,
        )),
    )
    erp, qrp = nemo_orca2_sss_restoring(
        sss_mean,
        sss_target,
        sst,
        runoff_mouth_mask,
        smask,
        restoring_mm_per_day=restoring_mm_per_day,
        bound_mm_per_day=restoring_bound_mm_per_day,
        seconds_per_day=seconds_per_day,
        salinity_floor=salinity_floor,
        heat_capacity=heat_capacity,
    )
    erp = _sr(jnp.where(update, erp, _sr(jnp.zeros_like(erp))))
    qrp = _sr(jnp.where(update, qrp, _sr(jnp.zeros_like(qrp))))
    emp = _sr(emp + erp)
    qns = _sr(qns + qrp)
    correction = _sr(jnp.asarray(freshwater_budget_correction, dtype=jnp.float64))
    correction_masked = _sr(jnp.where(update, _sr(correction * smask), 0.0))
    emp = _sr(emp + correction_masked)
    correction_heat = _sr(jnp.where(
        update,
        _sr(_sr(_sr(correction * _sr(heat_capacity)) * sst) * smask),
        0.0,
    ))
    qns = _sr(qns - correction_heat)
    utau_u, vtau_v = nemo_t_stress_to_uv(
        utau,
        vtau,
        post_ice_utau_east,
        post_ice_vtau_north,
        smask,
        tmask_east,
        tmask_north,
        umask,
        vmask,
    )
    penetrative_qsr = _sr(qsr)
    return NemoS2Result(
        runoff,
        erp,
        qrp,
        qns,
        qsr,
        emp,
        sfx,
        taum,
        utau,
        vtau,
        utau_u,
        vtau_v,
        penetrative_qsr,
    )


def nemo_s2_to_ocean_interfaces(result):
    """Bind an S2 result to the existing ocean forcing/runoff interfaces.

    K0 ``nn_ice=1`` contributes no separate freshwater or salt-mass channel;
    S3's heat effects are already present in the explicit post-ice boundary.
    The deck's disabled runoff-depth options select the existing top-cell arm.
    """
    q_net = _sr(result.qns + result.qsr)
    zero = _sr(jnp.zeros_like(result.emp))
    forcing = OceanSurfaceForcing(
        sw_down=result.penetrative_qsr,
        q_net=q_net,
        tau_x=result.utau,
        tau_y=result.vtau,
        freshwater=_sr(-result.emp),
        q_solar=result.penetrative_qsr,
        taum=result.taum,
        tau_i_native=result.utau_u,
        tau_j_native=result.vtau_v,
        runoff_tracer_content=(
            result.runoff.temperature_content,
            result.runoff.salinity_content,
        ),
        q_nonsolar=result.qns,
        salt_flux_sfx=result.sfx,
    )
    freshwater = FreshwaterForcing(
        precip=zero,
        evap=result.emp,
        runoff=result.runoff.mass_flux,
        ice_fw=zero,
        restoring=zero,
    )
    return forcing, freshwater


__all__ = (
    "NemoS2Result",
    "NemoSurfaceMeans",
    "nemo_orca2_s2",
    "nemo_sbc_ssm_step",
    "nemo_s2_to_ocean_interfaces",
    "nemo_t_stress_to_uv",
)
