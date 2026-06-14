"""Analytical AMIP teacher-data generation for the joint ML parameterization."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics._shared import compute_heights_from_sigma
from legoesm.atmosphere.physics.convection.config import MassFluxConfig
from legoesm.atmosphere.physics.convection.mass_flux import diagnose_mass_flux_closure
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig,
    SundqvistConfig,
)
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.physics.microphysics.sundqvist import (
    diagnose_sundqvist_process_rates,
)
from legoesm.atmosphere.physics.turbulence.config import LouisConfig
from legoesm.atmosphere.physics.turbulence.louis import louis_turbulence
from legoesm.forcing.surface_utils import blend_surface_temperature
from legoesm.thermo import saturation_specific_humidity


DEFAULT_SAMPLE_DAYS: tuple[float, ...] = (
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 14.0, 21.0, 28.0,
)


class PhysicsColumnBatch(NamedTuple):
    """Column state used to train the joint ML parameterization."""

    T: jax.Array
    u: jax.Array
    v: jax.Array
    q_v: jax.Array
    q_c: jax.Array
    q_r: jax.Array
    p_full: jax.Array
    p_half: jax.Array
    z_full: jax.Array
    z_half: jax.Array
    p_s: jax.Array
    T_sfc: jax.Array
    q_sfc: jax.Array
    lat: jax.Array
    cape: jax.Array
    M_c: jax.Array
    dt: jax.Array
    day: jax.Array


class PhysicsTeacherDataset(NamedTuple):
    """Teacher dataset pairing column state with ``Km/Kh`` and ``M_eq`` labels."""

    columns: PhysicsColumnBatch
    Km: jax.Array
    Kh: jax.Array
    M_eq: jax.Array
    dq_v_dt_micro: jax.Array
    dq_c_dt_micro: jax.Array
    dq_r_dt_micro: jax.Array
    precip_micro: jax.Array
    rain_survival_fraction: jax.Array
    sample_days: tuple[float, ...]
    microphysics_scheme: str = "none"


def _copy_array(arr: jax.Array) -> jax.Array:
    """Detach a JAX array from driver-owned buffers."""
    return jnp.asarray(np.asarray(jax.device_get(arr)).copy())


def capture_physics_teacher_snapshot(
    driver,
    day_tag: float,
    dt: float,
    *,
    mass_flux_config: MassFluxConfig | None = None,
    louis_config: LouisConfig | None = None,
) -> PhysicsTeacherDataset:
    """Capture one joint teacher snapshot from a running ModelDriver."""
    ad = driver.physics.adapter
    T = driver.state.T.data
    u = driver.state.u.data
    v = driver.state.v.data
    p_s = driver.state.p_s.data
    q_v = driver.q_v
    q_c = driver.q_c
    q_r = driver.q_r
    nlev = driver.config.grid.nlev
    p_full = p_s[..., None] * driver.sigma.sigma_full
    p_half = p_s[..., None] * driver.sigma.sigma_half
    T_col = ad.flatten_3d(T)
    u_col = ad.flatten_3d(u)
    v_col = ad.flatten_3d(v)
    q_v_col = ad.flatten_3d(q_v)
    q_c_col = ad.flatten_3d(jnp.zeros_like(q_v) if q_c is None else q_c)
    q_r_col = ad.flatten_3d(jnp.zeros_like(q_v) if q_r is None else q_r)
    p_full_col = ad.flatten_3d(p_full)
    p_half_col = p_half.reshape(ad.ncol, nlev + 1)
    z_full_col, z_half_col = compute_heights_from_sigma(T_col, p_half_col)
    p_s_col = ad.flatten_2d(p_s)

    lat_field = getattr(driver, "_physics_lat", None)
    if lat_field is None:
        lat_field = driver._grid_lat
    lat_col = ad.flatten_2d(lat_field)

    sst, sic = driver.get_sst_sic(day_tag)
    sst = jnp.asarray(sst)
    sic = jnp.asarray(sic)
    T_sfc = blend_surface_temperature(sst, sic, driver.config.T_ice)
    q_sfc = saturation_specific_humidity(T_sfc, p_s)
    T_sfc_col = ad.flatten_2d(T_sfc)
    q_sfc_col = ad.flatten_2d(q_sfc)
    rho_col = p_full_col / (constants.R_d * T_col)

    conv_prog = driver._carry_aux.get("conv_prog")
    if conv_prog is None:
        conv_prog = jnp.zeros((ad.ncol,), dtype=T.dtype)

    mass_flux_config = mass_flux_config or MassFluxConfig()
    louis_config = louis_config or LouisConfig()
    closure = diagnose_mass_flux_closure(
        T=T_col,
        q_v=q_v_col,
        p_full=p_full_col,
        p_half=p_half_col,
        M_c=conv_prog,
        dt=dt,
        config=mass_flux_config,
    )
    turb_out = louis_turbulence(
        u=u_col,
        v=v_col,
        T=T_col,
        q_v=q_v_col,
        p_full=p_full_col,
        p_half=p_half_col,
        z_full=z_full_col,
        z_half=z_half_col,
        T_sfc=T_sfc_col,
        q_sfc=q_sfc_col,
        rho=rho_col,
        dt=dt,
        config=louis_config,
    )
    zeros_3d = jnp.zeros_like(T_col)
    zeros_2d = jnp.zeros((ad.ncol,), dtype=T.dtype)
    micro_scheme = getattr(driver.config, "microphysics", "none")
    dq_v_dt_micro = zeros_3d
    dq_c_dt_micro = zeros_3d
    dq_r_dt_micro = zeros_3d
    precip_micro = zeros_2d
    rain_survival_fraction = jnp.ones((ad.ncol,), dtype=T.dtype)
    if micro_scheme in ("kessler", "sundqvist"):
        micro_config = getattr(driver.physics, "micro_config", None)
        hydrometeors = HydrometeorState(
            q_c=q_c_col,
            q_r=q_r_col,
            q_i=zeros_3d,
            q_s=zeros_3d,
            q_g=zeros_3d,
            N_c=zeros_3d,
            N_r=zeros_3d,
            N_i=zeros_3d,
        )
        dz_col = (p_half_col[:, 1:] - p_half_col[:, :-1]) / (rho_col * constants.g)
        if micro_scheme == "kessler":
            if micro_config is None or not hasattr(micro_config, "autoconversion_rate"):
                micro_config = KesslerConfig()
            micro_out = kessler_microphysics(
                T=T_col,
                q_v=q_v_col,
                hydrometeors=hydrometeors,
                p_full=p_full_col,
                p_half=p_half_col,
                rho=rho_col,
                dz=dz_col,
                dt=dt,
                config=micro_config,
            )
            dq_v_dt_micro = _copy_array(micro_out.dq_v_dt)
            dq_c_dt_micro = _copy_array(micro_out.dq_c_dt)
            dq_r_dt_micro = _copy_array(micro_out.dq_r_dt)
            precip_micro = _copy_array(micro_out.precipitation)
        else:
            if micro_config is None or not hasattr(micro_config, "rh_crit"):
                micro_config = SundqvistConfig()
            rates = diagnose_sundqvist_process_rates(
                T=T_col,
                q_v=q_v_col,
                hydrometeors=hydrometeors,
                p_full=p_full_col,
                p_half=p_half_col,
                rho=rho_col,
                dz=dz_col,
                dt=dt,
                config=micro_config,
            )
            generated_rain_flux = jnp.sum(rates.autoconversion * rho_col * dz_col, axis=1)
            rain_survival_fraction = _copy_array(
                jnp.where(
                    generated_rain_flux > 1.0e-12,
                    jnp.clip(rates.precipitation / generated_rain_flux, 0.0, 1.0),
                    1.0,
                )
            )
            # Diagnostic-rain semantics — must match
            # ``microphysics/sundqvist.py``: q_r is drained to the
            # surface every step, surface precipitation includes the
            # drained mass.  The teacher snapshot is what the ML model
            # learns to reproduce, so the snapshot must reflect the
            # corrected leaf, not the pre-fix ``dq_r_dt = autoconv - evap``
            # formulation that double-counted rain.
            dt_safe_train = jnp.maximum(dt, 1e-10)
            q_r_in_col = jnp.clip(q_r_col, 0.0, None)
            dp_col = p_half_col[:, 1:] - p_half_col[:, :-1]
            q_r_drain_flux_col = jnp.sum(q_r_in_col * dp_col, axis=1) / (
                constants.g * dt_safe_train
            )
            dq_v_dt_micro = _copy_array(-rates.condensation + rates.evaporation)
            dq_c_dt_micro = _copy_array(rates.condensation - rates.autoconversion)
            dq_r_dt_micro = _copy_array(-q_r_in_col / dt_safe_train)
            precip_micro = _copy_array(rates.precipitation + q_r_drain_flux_col)
    elif micro_scheme != "none":
        raise ValueError(
            "capture_physics_teacher_snapshot currently supports "
            "microphysics='none', 'kessler', or 'sundqvist'",
        )

    columns = PhysicsColumnBatch(
        T=_copy_array(T_col),
        u=_copy_array(u_col),
        v=_copy_array(v_col),
        q_v=_copy_array(q_v_col),
        q_c=_copy_array(q_c_col),
        q_r=_copy_array(q_r_col),
        p_full=_copy_array(p_full_col),
        p_half=_copy_array(p_half_col),
        z_full=_copy_array(z_full_col),
        z_half=_copy_array(z_half_col),
        p_s=_copy_array(p_s_col),
        T_sfc=_copy_array(T_sfc_col),
        q_sfc=_copy_array(q_sfc_col),
        lat=_copy_array(lat_col),
        cape=_copy_array(closure.cape),
        M_c=_copy_array(conv_prog),
        dt=_copy_array(jnp.full((ad.ncol,), dt, dtype=T.dtype)),
        day=_copy_array(jnp.full((ad.ncol,), day_tag, dtype=T.dtype)),
    )
    return PhysicsTeacherDataset(
        columns=columns,
        Km=_copy_array(turb_out.Km),
        Kh=_copy_array(turb_out.Kh),
        M_eq=_copy_array(closure.M_eq),
        dq_v_dt_micro=dq_v_dt_micro,
        dq_c_dt_micro=dq_c_dt_micro,
        dq_r_dt_micro=dq_r_dt_micro,
        precip_micro=precip_micro,
        rain_survival_fraction=rain_survival_fraction,
        sample_days=(float(day_tag),),
        microphysics_scheme=micro_scheme,
    )


def concatenate_physics_teacher_datasets(
    datasets: Sequence[PhysicsTeacherDataset],
) -> PhysicsTeacherDataset:
    """Concatenate a list of teacher datasets over the sample axis."""
    columns = PhysicsColumnBatch(
        **{
            field: jnp.concatenate(
                [getattr(dataset.columns, field) for dataset in datasets],
                axis=0,
            )
            for field in PhysicsColumnBatch._fields
        },
    )
    sample_days: list[float] = []
    microphysics_scheme = datasets[0].microphysics_scheme
    for dataset in datasets:
        if dataset.microphysics_scheme != microphysics_scheme:
            raise ValueError("Cannot concatenate teacher datasets with mixed microphysics schemes")
        sample_days.extend(dataset.sample_days)
    return PhysicsTeacherDataset(
        columns=columns,
        Km=jnp.concatenate([dataset.Km for dataset in datasets], axis=0),
        Kh=jnp.concatenate([dataset.Kh for dataset in datasets], axis=0),
        M_eq=jnp.concatenate([dataset.M_eq for dataset in datasets], axis=0),
        dq_v_dt_micro=jnp.concatenate(
            [dataset.dq_v_dt_micro for dataset in datasets],
            axis=0,
        ),
        dq_c_dt_micro=jnp.concatenate(
            [dataset.dq_c_dt_micro for dataset in datasets],
            axis=0,
        ),
        dq_r_dt_micro=jnp.concatenate(
            [dataset.dq_r_dt_micro for dataset in datasets],
            axis=0,
        ),
        precip_micro=jnp.concatenate(
            [dataset.precip_micro for dataset in datasets],
            axis=0,
        ),
        rain_survival_fraction=jnp.concatenate(
            [dataset.rain_survival_fraction for dataset in datasets],
            axis=0,
        ),
        sample_days=tuple(sample_days),
        microphysics_scheme=microphysics_scheme,
    )


def generate_amip_physics_teacher_dataset(
    experiment_config,
    *,
    sample_days: Sequence[float] | None = None,
) -> PhysicsTeacherDataset:
    """Generate a joint teacher dataset from an analytical AMIP baseline run."""
    from legoesm.driver.model_driver import ModelDriver

    if experiment_config.dataset != "analytical":
        raise ValueError("generate_amip_physics_teacher_dataset is analytical-only")
    if experiment_config.convection != "mass_flux":
        raise ValueError(
            "generate_amip_physics_teacher_dataset expects convection='mass_flux'",
        )
    if experiment_config.turbulence != "louis":
        raise ValueError(
            "generate_amip_physics_teacher_dataset expects turbulence='louis'",
        )
    if experiment_config.microphysics not in ("none", "kessler", "sundqvist"):
        raise ValueError(
            "generate_amip_physics_teacher_dataset currently supports "
            "microphysics='none', 'kessler', or 'sundqvist'",
        )

    dt = float(experiment_config.dycore.dt)
    driver = ModelDriver(experiment_config)
    driver.setup()
    requested_days = tuple(
        DEFAULT_SAMPLE_DAYS if sample_days is None
        else tuple(float(day) for day in sample_days)
    )
    missing_days = list(requested_days)
    captured: dict[float, PhysicsTeacherDataset] = {}

    def _capture_matching_days(current_day: float) -> None:
        tol = max(dt / 86400.0, 1e-8)
        ready = [day for day in missing_days if current_day + tol >= day]
        for day in ready:
            captured[day] = capture_physics_teacher_snapshot(
                driver=driver,
                day_tag=day,
                dt=dt,
            )
            missing_days.remove(day)

    _capture_matching_days(experiment_config.start_day)

    def _segment_callback(driver_obj, day, dt_segment):
        del driver_obj, dt_segment
        _capture_matching_days(float(day))

    status = driver.run(segment_callback=_segment_callback)
    if status != "COMPLETED":
        raise RuntimeError(f"AMIP teacher run failed: {status}")
    if missing_days:
        raise RuntimeError(f"Did not capture requested AMIP sample days: {missing_days}")

    ordered = [captured[day] for day in requested_days]
    dataset = concatenate_physics_teacher_datasets(ordered)
    expected = len(requested_days) * int(driver.physics.adapter.ncol)
    if dataset.columns.T.shape[0] != expected:
        raise RuntimeError(
            f"Expected {expected} sampled columns, got {dataset.columns.T.shape[0]}",
        )
    return dataset._replace(sample_days=requested_days)
