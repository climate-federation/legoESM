"""SCM microphysics substepping regression tests."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax import lax

from legoesm import constants
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    WING_P_SFC,
    wing2018_pressure_profile,
    wing2018_qv_profile,
    wing2018_temperature_profile,
)
from legoesm.atmosphere.physics import (
    ConvectionConfig,
    GravityWaveDragConfig,
    MicrophysicsConfig,
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.physics.microphysics.integration import get_microphysics_fn
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
from legoesm.thermo import saturation_mixing_ratio

from scripts.run import run_scm_rce_campaign as campaign


_DT_S = campaign.DEFAULT_DT_S
_SUBSTEPS = campaign.DEFAULT_SCM_MICROPHYSICS_SUBSTEPS
_NLEV = 30
_DAYS = 0.5
_TINY_NEGATIVE = -1.0e-12


def _synthetic_campaign_ref() -> campaign.ReferenceProfiles:
    z_m = np.linspace(32450.0, 550.0, _NLEV)
    z_half = np.empty(_NLEV + 1)
    z_half[0] = z_m[0] + (z_m[0] - z_m[1]) / 2.0
    z_half[1:-1] = (z_m[:-1] + z_m[1:]) / 2.0
    z_half[-1] = 0.0
    sigma_half = np.asarray(
        wing2018_pressure_profile(jnp.asarray(z_half, dtype=jnp.float64))
        / WING_P_SFC,
        dtype=float,
    )
    sigma_full = (sigma_half[:-1] + sigma_half[1:]) / 2.0
    mass_weights = np.diff(sigma_half)
    mass_weights = mass_weights / np.sum(mass_weights)
    zeros = np.zeros(_NLEV)
    return campaign.ReferenceProfiles(
        z_m=z_m,
        sigma_half=sigma_half,
        sigma_full=sigma_full,
        mass_weights=mass_weights,
        T_ref=zeros,
        qv_ref=zeros,
        qcond_ref=zeros,
        files_used=[],
        sfc_cross_check={},
    )


def _campaign_style_min_qcond(scheme: str, microphysics_substeps: int) -> float:
    ref = _synthetic_campaign_ref()
    cfg = campaign.make_physics_config(microphysics=scheme)
    T0, qv0 = campaign.wing_initial_profiles(ref)
    forcing = SCMForcing(
        prescribe="T_s",
        T_s=lambda _t: campaign.FIXED_SST_K,
    )
    scm = SingleColumnModel.create(
        physics_config=cfg,
        nlev=len(ref.z_m),
        dt=_DT_S,
        T_profile=T0,
        q_v_profile=qv0,
        p_s=WING_P_SFC,
        latitude_deg=0.0,
        time_integrator="forward_euler",
        forcing=forcing,
        dtype=jnp.float64,
        microphysics_substeps=microphysics_substeps,
    )
    scm.sigma_coord = campaign.make_sigma_coordinate_from_reference(ref)
    campaign._preseed_column_tracers(scm, cfg.microphysics.scheme)

    sst_col = jnp.asarray([campaign.FIXED_SST_K], dtype=jnp.float64)
    forcing_dict = {"T_sfc": sst_col}
    step_fn = scm._step_fn
    physics_fn = scm.physics_fn
    grid = scm.grid
    sigma_coord = scm.sigma_coord
    dt_arr = jnp.asarray(_DT_S, dtype=jnp.float64)
    nsteps = int(round(_DAYS * campaign.SECONDS_PER_DAY / _DT_S))

    def fixed_sst_tendency(state, phys_state, _t):
        phys_state = phys_state._replace(surface_T_sfc_override=sst_col)
        tend, phys_out = physics_fn(
            state,
            grid,
            sigma_coord,
            phys_state=phys_state,
            forcing=forcing_dict,
        )
        if phys_out is None:
            phys_out = phys_state
        else:
            phys_out = phys_out._replace(surface_T_sfc_override=sst_col)
        return tend, phys_out

    def body(carry, k):
        state, phys_state = carry
        t = k.astype(jnp.float64) * dt_arr
        new_state, new_phys = step_fn(
            state, phys_state, fixed_sst_tendency, _DT_S, t,
        )
        qcond = campaign._qcond_from_tracers(
            new_state.tracers,
            cfg.microphysics.scheme,
        )
        return (new_state, new_phys), jnp.min(qcond)

    @jax.jit
    def driver(state0, phys0):
        (_state_f, _phys_f), qcond_min = lax.scan(
            body, (state0, phys0), jnp.arange(nsteps),
        )
        return jnp.min(qcond_min)

    return float(driver(scm.state, scm.phys_state))


def test_scm_substeps_prevent_campaign_negative_condensate() -> None:
    # morrison now keeps condensate non-negative even at a single forward-Euler
    # microphysics step (dt=600 s): its scheme-level positivity floor clamps the
    # stiff ice/sedimentation removal at exactly zero (min qcond == 0.0, not a
    # negative overshoot).  The SCM substep coupling is then redundant-but-
    # harmless defense.  (This mirrors the thompson #477 fix below; the earlier
    # premise that single-step morrison OVERSHOOTS negative is stale — it was
    # already clamped at base, so the old ``< -1e-6`` assertion was a
    # permanent red.)  Assert BOTH paths stay non-negative.
    morrison_single = _campaign_style_min_qcond("morrison", microphysics_substeps=1)
    morrison_subbed = _campaign_style_min_qcond("morrison", microphysics_substeps=_SUBSTEPS)
    assert morrison_single >= _TINY_NEGATIVE
    assert morrison_subbed >= _TINY_NEGATIVE

    # thompson gained a scheme-level sublimation/sedimentation flux-limiter fix
    # (#477) that keeps condensate non-negative even at a single dt=600 s step;
    # the SCM substep is then redundant-but-harmless defense. Assert both paths
    # stay non-negative (documents the #477 + substep alignment).
    thompson_single = _campaign_style_min_qcond("thompson", microphysics_substeps=1)
    thompson_subbed = _campaign_style_min_qcond("thompson", microphysics_substeps=_SUBSTEPS)
    assert thompson_single >= _TINY_NEGATIVE
    assert thompson_subbed >= _TINY_NEGATIVE


def _substep_water_residual(scheme: str) -> tuple[float, float]:
    scheme_name, micro_fn, scheme_cfg = get_microphysics_fn(
        MicrophysicsConfig(scheme=scheme),
    )
    assert scheme_name == scheme
    assert micro_fn is not None

    ncol = 1
    nlev = 10
    T = jnp.full((ncol, nlev), 280.0, dtype=jnp.float64)
    p_half = jnp.linspace(1.0e4, 1.0e5, nlev + 1, dtype=jnp.float64)[None, :]
    p_full = (p_half[:, :-1] + p_half[:, 1:]) / 2.0
    dp = p_half[:, 1:] - p_half[:, :-1]
    rho = p_full / (constants.R_d * T)
    dz = dp / (rho * constants.g)
    q_v = saturation_mixing_ratio(T, p_full) * 1.001
    hydro = HydrometeorState(
        q_c=jnp.full_like(T, 1.0e-4),
        q_r=jnp.full_like(T, 5.0e-3),
        q_i=jnp.full_like(T, 1.0e-4),
        q_s=jnp.full_like(T, 1.0e-3),
        q_g=jnp.zeros_like(T),
        N_c=jnp.zeros_like(T),
        N_r=jnp.zeros_like(T),
        N_i=jnp.full_like(T, 1.0e4),
    )
    sub_dt = _DT_S / _SUBSTEPS

    def water_path(q_v_cur, hydro_cur):
        total_q = (
            q_v_cur + hydro_cur.q_c + hydro_cur.q_r
            + hydro_cur.q_i + hydro_cur.q_s + hydro_cur.q_g
        )
        return jnp.sum(total_q * dp / constants.g, axis=-1)

    water0 = water_path(q_v, hydro)

    def body(carry, _i):
        T_cur, qv_cur, hydro_cur, precip_acc = carry
        out = micro_fn(
            T_cur,
            qv_cur,
            hydro_cur,
            p_full,
            p_half,
            rho,
            dz,
            sub_dt,
            scheme_cfg,
        )
        hydro_next = hydro_cur._replace(
            q_c=hydro_cur.q_c + sub_dt * out.dq_c_dt,
            q_r=hydro_cur.q_r + sub_dt * out.dq_r_dt,
            q_i=hydro_cur.q_i + sub_dt * out.dq_i_dt,
            q_s=hydro_cur.q_s + sub_dt * out.dq_s_dt,
            q_g=hydro_cur.q_g + sub_dt * out.dq_g_dt,
            N_c=hydro_cur.N_c + sub_dt * out.dN_c_dt,
            N_r=hydro_cur.N_r + sub_dt * out.dN_r_dt,
            N_i=hydro_cur.N_i + sub_dt * out.dN_i_dt,
        )
        carry_next = (
            T_cur + sub_dt * out.dT_dt,
            qv_cur + sub_dt * out.dq_v_dt,
            hydro_next,
            precip_acc + sub_dt * out.precipitation,
        )
        return carry_next, None

    (T_f, qv_f, hydro_f, precip_acc), _ = lax.scan(
        body,
        (T, q_v, hydro, jnp.zeros((ncol,), dtype=jnp.float64)),
        jnp.arange(_SUBSTEPS),
    )
    del T_f
    residual = water_path(qv_f, hydro_f) + precip_acc - water0
    scale = jnp.maximum(jnp.abs(precip_acc) + jnp.abs(water0), 1.0e-12)
    min_cond = jnp.min(
        hydro_f.q_c + hydro_f.q_r + hydro_f.q_i + hydro_f.q_s + hydro_f.q_g
    )
    return float(jnp.max(jnp.abs(residual / scale))), float(min_cond)


@pytest.mark.parametrize("scheme", ("morrison", "thompson"))
def test_scm_substep_sequence_conserves_water_plus_precipitation(
    scheme: str,
) -> None:
    relative_residual, min_condensate = _substep_water_residual(scheme)

    assert relative_residual < 1.0e-10
    assert min_condensate >= _TINY_NEGATIVE


def test_scm_microphysics_substeps_preserve_reverse_mode_ad() -> None:
    z_m = jnp.asarray(_synthetic_campaign_ref().z_m, dtype=jnp.float64)
    T0 = wing2018_temperature_profile(z_m)
    qv0 = wing2018_qv_profile(z_m)
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="morrison"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )

    def loss(qv_scale):
        scm = SingleColumnModel.create(
            physics_config=cfg,
            nlev=_NLEV,
            dt=_DT_S,
            T_profile=T0,
            q_v_profile=qv0 * qv_scale,
            p_s=WING_P_SFC,
            time_integrator="forward_euler",
            dtype=jnp.float64,
            microphysics_substeps=3,
        )
        state, _phys = scm._step_fn(
            scm.state,
            scm.phys_state,
            scm._tend_fn,
            scm.dt,
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        qcond = campaign._qcond_from_tracers(
            state.tracers,
            cfg.microphysics.scheme,
        )
        return jnp.sum(state.T.data) + jnp.sum(qcond)

    grad = jax.grad(loss)(jnp.asarray(1.0, dtype=jnp.float64))

    assert jnp.isfinite(grad)
