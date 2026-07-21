#!/usr/bin/env python
"""Gradient-train SCM RCE physics parameters against plane-CRM profiles.

Default run:

    JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 .venv/bin/python \
        scripts/run/train_scm_rce_params.py

Use ``--quick`` for the CPU/GPU smoke path exercised by the unit test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax

from legoesm.atmosphere.physics import PhysicsConfig
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
from legoesm.ml.training import TrainingConfig, create_optimizer
from legoesm.training.param_collector import (
    apply_param_overrides,
    build_registry,
    build_trainable_params,
)
from legoesm.training.scm_rce_metrics import score_profiles_jax
from legoesm.training.trainable_params import (
    ParamConstraint,
    TrainablePhysicsParams,
    range_to_sigmoid_array,
)

from scripts.run import run_scm_rce_campaign as campaign


DEFAULT_OUTDIR = Path("results/scm_rce_training")
CAMPAIGN_SCHEME_FIELDS = (
    "radiation",
    "turbulence",
    "microphysics",
    "convection",
    "gravity_wave_drag",
)
BECHTOLD_SCHEME_KEY = "atm.conv.BechtoldConfig"
BECHTOLD_STOCHASTIC_PARAMS = {
    "atm.conv.BechtoldConfig.stochastic_amplitude",
    "atm.conv.BechtoldConfig.stochastic_decorrelation",
}
DEFAULT_STEPS = 8
DEFAULT_LR = 1.0e-1
DEFAULT_SPINUP_DAYS = 45.0
DEFAULT_TRAIN_DAYS = 5.0
DEFAULT_TARGET_LOSS = 0.976
QUICK_STEPS = 3
QUICK_TRAIN_DAYS = 0.03
GRAD_NONZERO_TOL = 1.0e-14


def _require_x64() -> None:
    jax.config.update("jax_enable_x64", True)


def _jsonable(obj: Any) -> Any:
    return campaign._to_jsonable(obj)


def _read_tuned_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Campaign tuned parameter JSON not found: {path}")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list of tuned parameter records")
    return rows


def _read_recommended_defaults(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Campaign recommended defaults JSON not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _optional_config_for_path(path: str):
    if path == "radiation.cloud_config":
        from legoesm.atmosphere.physics.clouds.config import CloudConfig

        return CloudConfig()
    if path == "turbulence.clubb":
        from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig

        return CLUBBConfig()
    raise ValueError(
        f"recommended_physics_config field {path!r} is a nested object, but "
        "the corresponding default config is None and no factory is registered"
    )


def _merge_config_dict(template: Any, values: dict[str, Any], path: str = "") -> Any:
    fields = getattr(template, "_fields", None)
    if fields is None:
        raise ValueError(
            f"recommended_physics_config field {path or '<root>'!r} is not a "
            f"NamedTuple config ({type(template).__name__})"
        )
    updates: dict[str, Any] = {}
    for key, value in values.items():
        if key not in fields:
            raise ValueError(
                f"Unknown recommended_physics_config field "
                f"{path + '.' if path else ''}{key!r}; known={list(fields)}"
            )
        current = getattr(template, key)
        child_path = f"{path}.{key}" if path else key
        if isinstance(value, dict):
            if current is None:
                current = _optional_config_for_path(child_path)
            updates[key] = _merge_config_dict(current, value, child_path)
        else:
            updates[key] = value
    return template._replace(**updates)


def _physics_config_from_recommended_dict(data: dict[str, Any]) -> PhysicsConfig:
    return _merge_config_dict(PhysicsConfig(), data)


def _recommended_scheme_winners(
    recommended: dict[str, Any],
    recommended_cfg: PhysicsConfig | None,
) -> dict[str, str]:
    winners_raw = recommended.get("winners", {})
    if winners_raw is None:
        winners_raw = {}
    if not isinstance(winners_raw, dict):
        raise ValueError("recommended_defaults.json field 'winners' must be an object")

    scheme_sources: list[dict[str, Any]] = []
    for key in ("tuned_best", "best_per_category", "baseline"):
        value = recommended.get(key)
        if isinstance(value, dict) and isinstance(value.get("config"), dict):
            scheme_sources.append(value["config"])

    winners: dict[str, str] = {}
    for field in CAMPAIGN_SCHEME_FIELDS:
        value = winners_raw.get(field)
        if value is None and recommended_cfg is not None:
            value = getattr(recommended_cfg, field).scheme
        if value is None:
            for source in scheme_sources:
                if source.get(field) is not None:
                    value = source[field]
                    break
        if value is None:
            raise ValueError(
                f"recommended defaults do not specify a winner for {field!r}"
            )
        winners[field] = str(value)
    return winners


def _apply_bechtold_policy_to_winners(
    winners: dict[str, str],
    policy: str,
) -> tuple[dict[str, str], str]:
    out = dict(winners)
    convection = out["convection"]
    if policy == "mass_flux":
        out["convection"] = "mass_flux"
        return (
            out,
            "explicit CLI override substituted `mass_flux` for the campaign "
            f"recommended convection winner `{convection}`.",
        )
    if convection != "bechtold":
        return (
            out,
            f"not applicable; campaign recommended convection winner is "
            f"`{convection}`, so the trainer follows that winner.",
        )
    if policy == "train_deterministic":
        return (
            out,
            "trained deterministic Bechtold tier-1/2 leaves and explicitly "
            "excluded stochastic leaves because `enable_stochastic=False` makes "
            "them structurally zero-gradient.",
        )
    if policy == "force":
        return (
            out,
            "forced Bechtold into the gradient preflight; any zero-gradient "
            "leaves fail the nonzero-gradient gate.",
        )
    if policy == "freeze":
        return (
            out,
            "frozen at the campaign-tuned values by explicit CLI policy.",
        )
    return (
        out,
        "frozen at the campaign-tuned values. Bechtold tier-2 stochastic "
        "parameters are structurally dead with `enable_stochastic=False`, so "
        "`auto` avoids handing MUON zero-gradient leaves.",
    )


def _build_recommended_base_config(
    recommended: dict[str, Any],
    *,
    bechtold_policy: str,
) -> tuple[PhysicsConfig, dict[str, str], str]:
    recommended_cfg = None
    recommended_cfg_raw = recommended.get("recommended_physics_config")
    if recommended_cfg_raw is not None:
        if not isinstance(recommended_cfg_raw, dict):
            raise ValueError(
                "recommended_defaults.json field 'recommended_physics_config' "
                "must be an object when present"
            )
        recommended_cfg = _physics_config_from_recommended_dict(recommended_cfg_raw)

    winners = _recommended_scheme_winners(recommended, recommended_cfg)
    winners, bechtold_note = _apply_bechtold_policy_to_winners(
        winners,
        bechtold_policy,
    )
    rad_update_steps = (
        recommended_cfg.radiation.update_interval_steps
        if recommended_cfg is not None
        else campaign.DEFAULT_RRTMGP_UPDATE_INTERVAL_STEPS
    )
    cfg = campaign.make_physics_config(
        radiation=winners["radiation"],
        radiation_update_interval_steps=rad_update_steps,
        turbulence=winners["turbulence"],
        microphysics=winners["microphysics"],
        gravity_wave_drag=winners["gravity_wave_drag"],
        convection=winners["convection"],
        base=recommended_cfg,
    )
    return cfg, winners, bechtold_note


def _active_subconfig_by_scheme_key(
    cfg: PhysicsConfig,
) -> dict[str, tuple[str, Any, Any]]:
    out: dict[str, tuple[str, Any, Any]] = {}
    for component_name in (
        "radiation",
        "turbulence",
        "microphysics",
        "convection",
        "gravity_wave_drag",
    ):
        component = getattr(cfg, component_name)
        scheme = component.scheme
        subcfg = getattr(component, scheme, None)
        if subcfg is None and component_name == "turbulence" and scheme == "clubb":
            from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig

            subcfg = CLUBBConfig()
        scheme_key = campaign._scheme_key_for_subconfig(subcfg)
        if scheme_key is not None:
            out[scheme_key] = (component_name, component, subcfg)
    return out


def _replace_active_subconfig(
    cfg: PhysicsConfig,
    component_name: str,
    component: Any,
    subcfg: Any,
) -> PhysicsConfig:
    scheme = component.scheme
    return cfg._replace(
        **{component_name: component._replace(**{scheme: subcfg})}
    )


def _apply_static_record_values(
    cfg: PhysicsConfig,
    records: list[dict[str, Any]],
) -> PhysicsConfig:
    by_scheme = _active_subconfig_by_scheme_key(cfg)
    grouped: dict[str, dict[str, float]] = {}
    for row in records:
        scheme_key = str(row["scheme_key"])
        if scheme_key not in by_scheme:
            continue
        grouped.setdefault(scheme_key, {})[str(row["parameter"])] = float(row["tuned"])

    out = cfg
    for scheme_key, field_values in grouped.items():
        component_name, component, subcfg = _active_subconfig_by_scheme_key(out)[scheme_key]
        tuned_tunable = apply_param_overrides(campaign._tunable_subconfig(subcfg), field_values)
        tuned_subcfg = campaign._rewrap_tunable_subconfig(subcfg, tuned_tunable)
        out = _replace_active_subconfig(out, component_name, component, tuned_subcfg)
    return out


def _apply_trainable_params(
    base_cfg: PhysicsConfig,
    params: TrainablePhysicsParams,
) -> PhysicsConfig:
    """Apply traced trainable leaves to active subconfigs inside the loss."""
    out = base_cfg
    overrides = params.to_overrides()
    for scheme_key, field_values in overrides.items():
        by_scheme = _active_subconfig_by_scheme_key(out)
        if scheme_key not in by_scheme:
            raise ValueError(
                f"Trainable parameter scheme {scheme_key!r} is not active in "
                f"the SCM config; active={sorted(by_scheme)}"
            )
        component_name, component, subcfg = by_scheme[scheme_key]
        # Descend into CLUBB's nested .params and re-wrap (traced-safe: isinstance
        # on the static config type, NamedTuple._replace is a pytree op).
        new_tunable = apply_param_overrides(campaign._tunable_subconfig(subcfg), field_values)
        new_subcfg = campaign._rewrap_tunable_subconfig(subcfg, new_tunable)
        out = _replace_active_subconfig(out, component_name, component, new_subcfg)
    return out


def _raw_from_physical(value: float, constraint: ParamConstraint) -> jax.Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    if constraint.transform == "sigmoid":
        return range_to_sigmoid_array(
            arr, constraint.min_val, constraint.max_val,
        ).astype(jnp.float64)
    if constraint.transform == "softplus":
        y = jnp.maximum(arr, jnp.asarray(campaign.PROFILE_FLOOR, dtype=jnp.float64))
        return (y + jnp.log1p(-jnp.exp(-y))).astype(jnp.float64)
    return arr


def _initial_params_from_campaign(
    *,
    active_scheme_keys: set[str],
    records: list[dict[str, Any]],
    exclude: set[str],
) -> TrainablePhysicsParams:
    params = build_trainable_params(
        active_scheme_keys=active_scheme_keys,
        tier="extended",
        exclude=tuple(sorted(exclude)),
        dtype=jnp.float64,
    )
    tuned_by_name = {
        f"{row['scheme_key']}.{row['parameter']}": float(row["tuned"])
        for row in records
    }
    defaults = params.as_dict()
    raw_values = {}
    for constraint in params.constraints:
        physical = tuned_by_name.get(
            constraint.name, float(defaults[constraint.name])
        )
        raw_values[constraint.name] = _raw_from_physical(physical, constraint)
    return TrainablePhysicsParams(raw_values=raw_values, constraints=params.constraints)


def _active_tuned_scheme_keys(
    cfg: PhysicsConfig,
    records: list[dict[str, Any]],
) -> set[str]:
    tuned_scheme_keys = {str(row["scheme_key"]) for row in records}
    active_scheme_keys = set(_active_subconfig_by_scheme_key(cfg))
    return active_scheme_keys & tuned_scheme_keys


def _filter_params(
    params: TrainablePhysicsParams,
    keep_names: set[str],
) -> TrainablePhysicsParams:
    constraints = [c for c in params.constraints if c.name in keep_names]
    raw_values = {c.name: params.raw_values[c.name] for c in constraints}
    return TrainablePhysicsParams(raw_values=raw_values, constraints=constraints)


def _finite_and_nonzero_param_grads(
    grads: TrainablePhysicsParams,
    *,
    tol: float,
) -> dict[str, dict[str, float | bool]]:
    stats: dict[str, dict[str, float | bool]] = {}
    for constraint in grads.constraints:
        grad = grads.raw_values[constraint.name]
        abs_max = float(jnp.max(jnp.abs(grad)))
        finite = bool(jnp.all(jnp.isfinite(grad)))
        stats[constraint.name] = {
            "abs_max": abs_max,
            "finite": finite,
            "nonzero": bool(finite and abs_max > tol),
        }
    return stats


def _assert_strict_bounds(params: TrainablePhysicsParams) -> None:
    physical = params.as_dict()
    for constraint in params.constraints:
        value = physical[constraint.name]
        lo = jnp.asarray(constraint.min_val, dtype=value.dtype)
        hi = jnp.asarray(constraint.max_val, dtype=value.dtype)
        if not bool(jnp.all(jnp.isfinite(value))):
            raise RuntimeError(f"{constraint.name} is non-finite after transform")
        if not bool(jnp.all((value > lo) & (value < hi))):
            raise RuntimeError(
                f"{constraint.name}={np.asarray(value)} escaped strict bounds "
                f"({constraint.min_val}, {constraint.max_val})"
            )


def _qcond_from_tracers(tracers, microphysics_scheme: str) -> jax.Array:
    return campaign._qcond_from_tracers(tracers, microphysics_scheme)


def _use_cached_radiation(cfg: PhysicsConfig) -> bool:
    return cfg.radiation.scheme != "none" and int(cfg.radiation.update_interval_steps) > 1


def _create_scm(cfg: PhysicsConfig, ref: campaign.ReferenceProfiles, dt: float) -> SingleColumnModel:
    T0, qv0 = campaign.wing_initial_profiles(ref)
    scm_cfg = (
        campaign._without_radiation_config(cfg)
        if _use_cached_radiation(cfg)
        else cfg
    )
    use_split_convection = (
        campaign._effective_scm_convection_substeps(
            cfg.convection.scheme,
            campaign.DEFAULT_SCM_CONVECTION_SUBSTEPS,
        )
        > 1
        and cfg.convection.scheme != "none"
    )
    wind_profile = jnp.full(
        (len(ref.z_m),), campaign.DEFAULT_SCM_RCE_SURFACE_WIND_M_S,
        dtype=jnp.float64,
    )
    zero_wind_profile = jnp.zeros((len(ref.z_m),), dtype=jnp.float64)
    forcing = SCMForcing(
        prescribe="T_s",
        T_s=lambda _t: campaign.FIXED_SST_K,
        f_c=campaign.DEFAULT_SCM_RCE_CORIOLIS_S_INV,
        u_geo=lambda _t: wind_profile,
        v_geo=lambda _t: zero_wind_profile,
    )
    scm = SingleColumnModel.create(
        physics_config=(
            campaign._without_convection_config(scm_cfg)
            if use_split_convection
            else scm_cfg
        ),
        nlev=len(ref.z_m),
        dt=dt,
        T_profile=T0,
        q_v_profile=qv0,
        u=campaign.DEFAULT_SCM_RCE_SURFACE_WIND_M_S,
        v=0.0,
        p_s=campaign.WING_P_SFC,
        latitude_deg=0.0,
        time_integrator="forward_euler",
        forcing=forcing,
        dtype=jnp.float64,
        microphysics_substeps=campaign._effective_scm_microphysics_substeps(
            cfg.microphysics.scheme,
            campaign.DEFAULT_SCM_MICROPHYSICS_SUBSTEPS,
        ),
    )
    scm.sigma_coord = campaign.make_sigma_coordinate_from_reference(ref)
    campaign._preseed_column_tracers(scm, cfg.microphysics.scheme)
    return scm


def _initial_step_carry(scm: SingleColumnModel, cfg: PhysicsConfig):
    if _use_cached_radiation(cfg):
        return (
            scm.state,
            scm.phys_state,
            _zero_radiation_tendency_like_state(scm.state),
        )
    return scm.state, scm.phys_state


def _zero_radiation_tendency_like_state(state):
    tend = campaign._zero_tendency_like_state(state)
    return tend._replace(
        du_dt=tend.du_dt.replace(name="du_dt_rad"),
        dT_dt=tend.dT_dt.replace(name="dT_dt_rad"),
        dp_s_dt=tend.dp_s_dt.replace(name="dp_s_dt_rad"),
        dphis_dt=tend.dphis_dt.replace(name="dphis_dt_rad"),
        dv_dt=(
            None if tend.dv_dt is None
            else tend.dv_dt.replace(name="dv_dt_rad")
        ),
    )


def _make_step_once(
    scm: SingleColumnModel,
    cfg: PhysicsConfig,
    ref: campaign.ReferenceProfiles,
    dt: float,
):
    sst_col = jnp.asarray([campaign.FIXED_SST_K], dtype=jnp.float64)
    forcing_dict = {"T_sfc": sst_col}
    step_fn = scm._step_fn
    physics_fn = scm.physics_fn
    grid = scm.grid
    sigma_coord = scm.sigma_coord
    dt_arr = jnp.asarray(dt, dtype=jnp.float64)
    microphysics_scheme = cfg.microphysics.scheme
    use_cached_radiation = _use_cached_radiation(cfg)
    radiation_interval = max(1, int(cfg.radiation.update_interval_steps))
    rad_physics_fn = None
    if use_cached_radiation:
        rad_physics_fn = campaign.make_physics(
            campaign._radiation_only_config(cfg),
            model_type="hydrostatic",
            dt=dt,
        )
    effective_convection_substeps = campaign._effective_scm_convection_substeps(
        cfg.convection.scheme,
        campaign.DEFAULT_SCM_CONVECTION_SUBSTEPS,
    )
    use_split_convection = (
        cfg.convection.scheme != "none" and effective_convection_substeps > 1
    )
    conv_physics_fn = None
    if use_split_convection:
        conv_physics_fn = campaign.make_physics(
            campaign._convection_only_config(cfg),
            model_type="hydrostatic",
            dt=dt / effective_convection_substeps,
        )
    z_profile = jnp.asarray(ref.z_m, dtype=jnp.float64)
    z_above_lowest = jnp.maximum(z_profile - z_profile[-1], 0.0)
    bl_mask = z_above_lowest <= campaign.DEFAULT_SCM_RCE_BL_TOP_M

    def apply_surface_sst_anchor(state):
        T_bl = jnp.asarray(campaign.FIXED_SST_K, dtype=state.T.data.dtype) - (
            campaign.DEFAULT_SCM_RCE_BL_LAPSE_K_M
            * z_above_lowest.astype(state.T.data.dtype)
        )
        T_bl = jnp.maximum(
            T_bl,
            jnp.asarray(campaign.DEFAULT_SCM_RCE_BL_MIN_T_K, dtype=state.T.data.dtype),
        )
        mask = bl_mask.reshape(1, 1, 1, -1)
        T_data = jnp.where(mask, T_bl.reshape(1, 1, 1, -1), state.T.data)
        return state._replace(T=state.T.replace(data=T_data))

    def call_physics_with_fixed_sst(fn, state, phys_state):
        phys_state = phys_state._replace(surface_T_sfc_override=sst_col)
        tend, phys_out = fn(
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

    def fixed_sst_tendency(state, phys_state, _t):
        return call_physics_with_fixed_sst(physics_fn, state, phys_state)

    def apply_convection_substeps(state, phys_state):
        if not use_split_convection:
            return state, phys_state
        sub_dt = dt / effective_convection_substeps

        def substep(carry, _i):
            sub_state, sub_phys = carry
            conv_tend, conv_phys = conv_physics_fn(
                sub_state,
                grid,
                sigma_coord,
                phys_state=sub_phys,
            )
            if conv_phys is None:
                conv_phys = sub_phys
            return (
                campaign.apply_tendencies(sub_state, conv_tend, sub_dt),
                conv_phys,
            ), None

        (new_state, new_phys), _ = lax.scan(
            substep,
            (state, phys_state),
            jnp.arange(effective_convection_substeps, dtype=jnp.int32),
        )
        return new_state, new_phys

    def step_once(carry, k):
        state, phys_state = carry
        t = k.astype(jnp.float64) * dt_arr
        new_state, new_phys = step_fn(
            state, phys_state, fixed_sst_tendency, dt, t,
        )
        new_state, new_phys = apply_convection_substeps(new_state, new_phys)
        new_state = apply_surface_sst_anchor(new_state)
        qv = new_state.tracers["q_v"].data[0, 0, 0]
        qcond = _qcond_from_tracers(
            new_state.tracers, microphysics_scheme,
        )[0, 0, 0]
        out = (new_state.T.data[0, 0, 0], qv, qcond)
        return (new_state, new_phys), out

    def cached_radiation_step_once(carry, k):
        state, phys_state, rad_cache = carry
        t = k.astype(jnp.float64) * dt_arr

        def refresh(_unused):
            rad_tend, _rad_phys = call_physics_with_fixed_sst(
                rad_physics_fn,
                state,
                phys_state,
            )
            del _rad_phys
            return rad_tend

        rad_tend = lax.cond(
            jnp.mod(k, radiation_interval) == 0,
            refresh,
            lambda _unused: rad_cache,
            operand=None,
        )

        def cached_fixed_sst_tendency(step_state, step_phys_state, _t):
            nonrad_tend, phys_out = fixed_sst_tendency(
                step_state,
                step_phys_state,
                _t,
            )
            return campaign.add_tendencies(nonrad_tend, rad_tend), phys_out

        new_state, new_phys = step_fn(
            state, phys_state, cached_fixed_sst_tendency, dt, t,
        )
        new_state, new_phys = apply_convection_substeps(new_state, new_phys)
        new_state = apply_surface_sst_anchor(new_state)
        qv = new_state.tracers["q_v"].data[0, 0, 0]
        qcond = _qcond_from_tracers(
            new_state.tracers, microphysics_scheme,
        )[0, 0, 0]
        out = (new_state.T.data[0, 0, 0], qv, qcond)
        return (new_state, new_phys, rad_tend), out

    return cached_radiation_step_once if use_cached_radiation else step_once


def _compute_static_spinup_carry(
    cfg: PhysicsConfig,
    ref: campaign.ReferenceProfiles,
    *,
    spinup_days: float,
    dt: float,
    chunk_steps: int,
):
    spinup_steps = max(0, int(round(spinup_days * campaign.SECONDS_PER_DAY / dt)))
    scm = _create_scm(cfg, ref, dt)
    carry0 = _initial_step_carry(scm, cfg)
    if spinup_steps == 0:
        return carry0
    chunk_steps = max(1, int(chunk_steps))
    nchunks = int(math.ceil(spinup_steps / chunk_steps))
    step_once = _make_step_once(scm, cfg, ref, dt)

    def masked_step(carry, k):
        return lax.cond(
            k < spinup_steps,
            lambda c: step_once(c, k)[0],
            lambda c: c,
            carry,
        ), None

    def chunk_body(carry, chunk_index):
        start = chunk_index * chunk_steps
        ks = start + jnp.arange(chunk_steps)
        return lax.scan(masked_step, carry, ks)

    @jax.jit
    def driver(initial_carry):
        carry, _ = lax.scan(
            chunk_body,
            initial_carry,
            jnp.arange(nchunks),
        )
        return carry

    return driver(carry0)


def _rollout_profiles(
    params: TrainablePhysicsParams,
    *,
    base_cfg: PhysicsConfig,
    ref: campaign.ReferenceProfiles,
    initial_carry: tuple[Any, Any] | None,
    spinup_days: float,
    days: float,
    dt: float,
    analysis_days: float,
    chunk_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    spinup_steps = max(0, int(round(spinup_days * campaign.SECONDS_PER_DAY / dt)))
    nsteps = max(1, int(round(days * campaign.SECONDS_PER_DAY / dt)))
    analysis_steps = max(1, int(round(analysis_days * campaign.SECONDS_PER_DAY / dt)))
    analysis_steps = min(analysis_steps, nsteps)
    chunk_steps = max(1, int(chunk_steps))
    nchunks = int(math.ceil(nsteps / chunk_steps))
    total_steps = nchunks * chunk_steps

    cfg = _apply_trainable_params(base_cfg, params)
    scm = _create_scm(cfg, ref, dt)
    initial_step_carry = _initial_step_carry(scm, cfg)
    nlev = len(ref.z_m)
    step_once = jax.checkpoint(_make_step_once(scm, cfg, ref, dt))

    def masked_step(carry, k):
        zeros = (
            jnp.zeros((nlev,), dtype=jnp.float64),
            jnp.zeros((nlev,), dtype=jnp.float64),
            jnp.zeros((nlev,), dtype=jnp.float64),
        )
        return lax.cond(
            k < spinup_steps + nsteps,
            lambda c: step_once(c, k),
            lambda c: (c, zeros),
            carry,
        )

    def spinup_chunk_body(carry, chunk_index):
        start = chunk_index * chunk_steps
        ks = start + jnp.arange(chunk_steps)

        def spinup_masked_step(c, k):
            return lax.cond(
                k < spinup_steps,
                lambda cc: step_once(cc, k)[0],
                lambda cc: cc,
                c,
            ), None

        return lax.scan(spinup_masked_step, carry, ks)

    if initial_carry is not None:
        carry0 = jax.tree_util.tree_map(lax.stop_gradient, initial_carry)
    elif spinup_steps:
        spinup_chunks = int(math.ceil(spinup_steps / chunk_steps))
        spinup_chunk_body = jax.checkpoint(spinup_chunk_body)
        carry0, _ = lax.scan(
            spinup_chunk_body,
            initial_step_carry,
            jnp.arange(spinup_chunks),
        )
        carry0 = jax.tree_util.tree_map(lax.stop_gradient, carry0)
    else:
        carry0 = initial_step_carry

    def chunk_body(carry, chunk_index):
        start = chunk_index * chunk_steps
        ks = spinup_steps + start + jnp.arange(chunk_steps)
        return lax.scan(masked_step, carry, ks)

    chunk_body = jax.checkpoint(chunk_body)
    _final_carry, history_chunks = lax.scan(
        chunk_body,
        carry0,
        jnp.arange(nchunks),
    )
    del _final_carry
    T_hist, qv_hist, qcond_hist = (
        x.reshape((total_steps, nlev))[:nsteps] for x in history_chunks
    )
    T_profile = jnp.mean(T_hist[-analysis_steps:], axis=0)
    qv_profile = jnp.mean(qv_hist[-analysis_steps:], axis=0)
    qcond_profile = jnp.maximum(jnp.mean(qcond_hist[-analysis_steps:], axis=0), 0.0)
    return T_profile, qv_profile, qcond_profile


def _make_loss_fn(
    *,
    base_cfg: PhysicsConfig,
    ref: campaign.ReferenceProfiles,
    initial_carry: tuple[Any, Any] | None,
    spinup_days: float,
    days: float,
    dt: float,
    analysis_days: float,
    chunk_steps: int,
):
    def loss_fn(params: TrainablePhysicsParams) -> jax.Array:
        T_profile, qv_profile, qcond_profile = _rollout_profiles(
            params,
            base_cfg=base_cfg,
            ref=ref,
            initial_carry=initial_carry,
            spinup_days=spinup_days,
            days=days,
            dt=dt,
            analysis_days=analysis_days,
            chunk_steps=chunk_steps,
        )
        *_components, combined = score_profiles_jax(
            ref,
            T_profile,
            qv_profile,
            qcond_profile,
            profile_floor=campaign.PROFILE_FLOOR,
        )
        return combined

    return loss_fn


def _materialize_static_config(
    base_cfg: PhysicsConfig,
    params: TrainablePhysicsParams,
) -> PhysicsConfig:
    cfg = _apply_trainable_params(base_cfg, params)

    def to_float_if_scalar(x):
        if isinstance(x, jax.Array):
            arr = np.asarray(x)
            if arr.ndim == 0:
                return float(arr)
        return x

    return jax.tree_util.tree_map(to_float_if_scalar, cfg)


def _run_campaign_eval(
    cfg: PhysicsConfig,
    ref: campaign.ReferenceProfiles,
    *,
    label: str,
    days: float,
    dt: float,
    analysis_days: float,
) -> campaign.RunDiagnostics:
    return campaign.run_scm_rce(
        cfg,
        ref,
        label=label,
        days=days,
        dt=dt,
        analysis_days=analysis_days,
        require_equilibrium=False,
        require_realism=False,
        equil_T_tol_K=campaign.EQUIL_T_TOL_K,
        equil_qv_tol=campaign.EQUIL_QV_TOL,
        equil_qcond_tol=campaign.EQUIL_QCOND_TOL,
    )


def _plot_loss_curve(path: Path, losses: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(np.arange(len(losses)), losses, marker="o", color="#1f4e79")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_profiles_before_after(
    path: Path,
    ref: campaign.ReferenceProfiles,
    before: campaign.RunDiagnostics,
    after: campaign.RunDiagnostics,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_km = ref.z_m / campaign.M_PER_KM
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 5.2), sharey=False)
    panels = [
        ("T", ref.T_ref, before.T_profile, after.T_profile, "K"),
        (
            "qv",
            ref.qv_ref * campaign.MSE_KJ_TO_J,
            np.asarray(before.qv_profile) * campaign.MSE_KJ_TO_J,
            np.asarray(after.qv_profile) * campaign.MSE_KJ_TO_J,
            "g/kg",
        ),
        (
            "condensate",
            ref.qcond_ref * campaign.MSE_KJ_TO_J,
            np.asarray(before.qcond_profile) * campaign.MSE_KJ_TO_J,
            np.asarray(after.qcond_profile) * campaign.MSE_KJ_TO_J,
            "g/kg",
        ),
    ]
    for ax, (name, ref_prof, before_prof, after_prof, units) in zip(axes[:3], panels):
        ax.plot(ref_prof, z_km, color="#1f4e79", lw=2.0, label="CRM")
        ax.plot(before_prof, z_km, color="#d95f02", lw=1.8, label="campaign tuned")
        ax.plot(after_prof, z_km, color="#2a9d8f", lw=1.8, label="gradient trained")
        ax.set_xlabel(f"{name} [{units}]")
        ax.set_ylim(0.0, float(np.nanmax(z_km)))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("z [km]")
    axes[0].legend(loc="best")
    axes[3].bar(
        [0, 1, 2],
        [ref.precip_ref_mm_day, before.precip_mm_day, after.precip_mm_day],
        color=["#1f4e79", "#d95f02", "#2a9d8f"],
        width=0.65,
    )
    axes[3].set_xticks([0, 1, 2])
    axes[3].set_xticklabels(["CRM", "before", "after"])
    axes[3].set_ylabel("surface precip [mm/day]")
    axes[3].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _parameter_table(
    *,
    report_params_all: TrainablePhysicsParams,
    trained_params: TrainablePhysicsParams,
    records: list[dict[str, Any]],
    frozen_reasons: dict[str, str],
) -> list[dict[str, Any]]:
    registry = {m.qualified_name: m for m in build_registry()}
    campaign_tuned = {
        f"{row['scheme_key']}.{row['parameter']}": float(row["tuned"])
        for row in records
    }
    initial_values = report_params_all.as_dict()
    trained_values = trained_params.as_dict()
    rows = []
    for constraint in report_params_all.constraints:
        meta = registry[constraint.name]
        trained = constraint.name in trained_values
        value = (
            float(trained_values[constraint.name])
            if trained
            else float(initial_values[constraint.name])
        )
        rows.append(
            {
                "name": constraint.name,
                "scheme_key": constraint.scheme_key,
                "field": constraint.field,
                "units": meta.units,
                "default": meta.default,
                "campaign_tuned": campaign_tuned.get(constraint.name),
                "gradient_trained": value,
                "lower": float(constraint.min_val),
                "upper": float(constraint.max_val),
                "trained": trained,
                "frozen_reason": None if trained else frozen_reasons.get(
                    constraint.name, "zero/non-finite preflight gradient"
                ),
            }
        )
    return rows


def _write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    ref: campaign.ReferenceProfiles,
    loss_history: list[float],
    before_eval: campaign.RunDiagnostics,
    after_eval: campaign.RunDiagnostics,
    grad_stats: dict[str, dict[str, float | bool]],
    parameter_rows: list[dict[str, Any]],
    bechtold_note: str,
) -> None:
    trained_count = sum(1 for row in parameter_rows if row["trained"])
    lines = [
        "# SCM RCE Gradient Training Summary",
        "",
        f"Reference files: last {len(ref.files_used)} CRM volume snapshots.",
        f"Training window: {args.train_days:g} days; analysis window: "
        f"{args.analysis_days:g} days; spinup: {args.spinup_days:g} days; "
        f"dt: {args.dt:g} s.",
        "The rollout uses `lax.scan` chunks wrapped with `jax.checkpoint`; "
        "the spinup carry is stop-gradiented and no JIT donation is used.",
        "",
        "Optimizer: `legoesm.ml.training.create_optimizer` with "
        f"`optimizer={args.optimizer}`, peak lr `{args.lr:g}`, "
        f"warmup `{args.warmup_steps}`, cosine decay over `{args.steps}` steps, "
        f"global clip `{args.grad_clip_norm:g}`.",
        "",
        f"Bechtold AD treatment: {bechtold_note}",
        "",
        f"Initial training loss: {loss_history[0]:.8g}",
        f"Final training loss: {loss_history[-1]:.8g}",
        f"Campaign-tuned eval score: {before_eval.score:.8g} ({before_eval.status})",
        f"Gradient-trained eval score: {after_eval.score:.8g} ({after_eval.status})",
        f"Target loss threshold: {args.target_loss:.8g}",
    ]
    if after_eval.score < args.target_loss:
        lines.append("Verification: final eval score is below the requested threshold.")
    else:
        lines.append(
            "Verification: final eval score did not beat the requested threshold. "
            "The run optimized only active winning-scheme parameters with finite "
            "nonzero gradients over the differentiable window; this can limit "
            "improvement relative to the derivative-free campaign optimum."
        )
    if args.spinup_days > 0.0:
        spinup_mode = (
            "fixed campaign-tuned warm start"
            if args.fixed_spinup
            else "current-parameter spinup inside each loss"
        )
        lines += [
            "",
            f"Truncated-gradient note: spinup mode is `{spinup_mode}`.  The "
            "trainer stops gradients through the spinup carry and "
            "differentiates the final equilibrium window.  This keeps the "
            "optimized forward target aligned with the campaign final-window "
            "profile score while bounding reverse-mode memory; final "
            "verification is reported with the campaign evaluation path.",
        ]
    lines += [
        "",
        f"Trained parameters: {trained_count}",
        "",
        "| parameter | trained | initial grad abs max | final value | bounds |",
        "|---|---:|---:|---:|---|",
    ]
    grad_by_name = grad_stats
    for row in parameter_rows:
        stat = grad_by_name.get(row["name"], {})
        grad = stat.get("abs_max")
        grad_s = "" if grad is None else f"{float(grad):.3e}"
        lines.append(
            f"| `{row['name']}` | {row['trained']} | {grad_s} | "
            f"{row['gradient_trained']:.8g} | "
            f"[{row['lower']:.8g}, {row['upper']:.8g}] |"
        )
    path.write_text("\n".join(lines) + "\n")


def _choose_active_scheme_keys(
    args: argparse.Namespace,
    cfg: PhysicsConfig,
    records: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    active = _active_tuned_scheme_keys(cfg, records)
    exclude: set[str] = set()
    convection_key = None
    by_scheme = _active_subconfig_by_scheme_key(cfg)
    for scheme_key, (component_name, _component, _subcfg) in by_scheme.items():
        if component_name == "convection":
            convection_key = scheme_key
            break
    if convection_key == BECHTOLD_SCHEME_KEY and args.bechtold_policy == "train_deterministic":
        exclude |= BECHTOLD_STOCHASTIC_PARAMS
    elif convection_key == BECHTOLD_SCHEME_KEY and args.bechtold_policy in {
        "auto",
        "freeze",
    }:
        active.discard(BECHTOLD_SCHEME_KEY)
    return active, exclude


def train(args: argparse.Namespace) -> dict[str, Any]:
    _require_x64()
    args.outdir.mkdir(parents=True, exist_ok=True)

    records = _read_tuned_records(args.tuned_parameters)
    recommended = _read_recommended_defaults(args.recommended)
    ref = campaign.build_reference_profiles(args.reference_dir, args.last_reference_files)
    base_cfg, winners, bechtold_note = _build_recommended_base_config(
        recommended,
        bechtold_policy=args.bechtold_policy,
    )
    campaign_tuned_cfg = _apply_static_record_values(base_cfg, records)
    initial_carry = None
    if args.fixed_spinup and args.spinup_days > 0.0:
        print(
            f"[spinup] computing fixed warm-start carry for "
            f"{args.spinup_days:g} days"
        )
        initial_carry = _compute_static_spinup_carry(
            campaign_tuned_cfg,
            ref,
            spinup_days=args.spinup_days,
            dt=args.dt,
            chunk_steps=args.chunk_steps,
        )
    active_scheme_keys, exclude_names = _choose_active_scheme_keys(
        args,
        campaign_tuned_cfg,
        records,
    )
    report_scheme_keys = _active_tuned_scheme_keys(campaign_tuned_cfg, records)
    if not active_scheme_keys:
        raise RuntimeError(
            "No active campaign-winning tuned scheme has trainable parameters; "
            f"winners={winners}, tuned_scheme_keys="
            f"{sorted({str(row['scheme_key']) for row in records})}"
        )
    initial_params_all = _initial_params_from_campaign(
        active_scheme_keys=active_scheme_keys,
        records=records,
        exclude=exclude_names,
    )
    report_params_all = _initial_params_from_campaign(
        active_scheme_keys=report_scheme_keys,
        records=records,
        exclude=set(),
    )

    loss_fn_all = _make_loss_fn(
        base_cfg=campaign_tuned_cfg,
        ref=ref,
        initial_carry=initial_carry,
        days=args.train_days,
        spinup_days=args.spinup_days,
        dt=args.dt,
        analysis_days=args.analysis_days,
        chunk_steps=args.chunk_steps,
    )
    print(
        f"[preflight] {len(initial_params_all.constraints)} candidate params; "
        f"spinup_days={args.spinup_days:g}, train_days={args.train_days:g}, "
        f"chunk_steps={args.chunk_steps}"
    )
    preflight_loss, preflight_grads = eqx.filter_value_and_grad(loss_fn_all)(
        initial_params_all
    )
    preflight_loss_val = float(preflight_loss)
    if not np.isfinite(preflight_loss_val):
        raise RuntimeError(f"Preflight loss is non-finite: {preflight_loss_val}")
    preflight_stats = _finite_and_nonzero_param_grads(
        preflight_grads, tol=args.grad_nonzero_tol,
    )
    frozen_reasons = {
        name: (
            "non-finite gradient"
            if not bool(stat["finite"])
            else f"|grad| <= {args.grad_nonzero_tol:g} in preflight"
        )
        for name, stat in preflight_stats.items()
        if not bool(stat["nonzero"])
    }
    for constraint in report_params_all.constraints:
        if constraint.scheme_key == BECHTOLD_SCHEME_KEY:
            frozen_reasons.setdefault(constraint.name, bechtold_note)
    if args.bechtold_policy == "force" and frozen_reasons:
        raise RuntimeError(
            "Forced Bechtold/all-parameter training found zero or non-finite "
            f"gradients: {frozen_reasons}"
        )
    keep_names = {
        name for name, stat in preflight_stats.items() if bool(stat["nonzero"])
    }
    if not keep_names:
        raise RuntimeError("No candidate parameter has a finite nonzero gradient")
    params = _filter_params(initial_params_all, keep_names)
    _assert_strict_bounds(params)

    loss_fn = _make_loss_fn(
        base_cfg=campaign_tuned_cfg,
        ref=ref,
        initial_carry=initial_carry,
        days=args.train_days,
        spinup_days=args.spinup_days,
        dt=args.dt,
        analysis_days=args.analysis_days,
        chunk_steps=args.chunk_steps,
    )
    optimizer = create_optimizer(
        TrainingConfig(
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            total_steps=max(args.steps, 1),
            grad_clip_norm=args.grad_clip_norm,
            optimizer=args.optimizer,
        )
    )
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))
    loss_history = [float(loss_fn(params))]
    print(f"[train] step=0 loss={loss_history[-1]:.8g} trained={len(params.constraints)}")

    final_grad_stats: dict[str, dict[str, float | bool]] = {}
    for step in range(1, args.steps + 1):
        t0 = time.time()
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        loss_val = float(loss)
        grad_stats = _finite_and_nonzero_param_grads(
            grads, tol=args.grad_nonzero_tol,
        )
        bad = {
            name: stat for name, stat in grad_stats.items()
            if not bool(stat["finite"]) or not bool(stat["nonzero"])
        }
        if bad:
            raise RuntimeError(f"Gradient gate failed at step {step}: {bad}")
        grad_norm = float(optax.global_norm(eqx.filter(grads, eqx.is_array)))
        updates, opt_state_candidate = optimizer.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            eqx.filter(params, eqx.is_array),
        )
        accepted = False
        best_params = params
        best_loss = loss_val
        for scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001):
            candidate = eqx.apply_updates(params, _scale_updates(updates, scale))
            _assert_strict_bounds(candidate)
            candidate_loss = float(loss_fn(candidate))
            if np.isfinite(candidate_loss) and candidate_loss < best_loss:
                best_params = candidate
                best_loss = candidate_loss
                accepted = True
                break
        if not accepted:
            print(
                f"[train] early_stop step={step} no reducing MUON update; "
                f"loss={loss_val:.8g}"
            )
            break
        params = best_params
        opt_state = opt_state_candidate
        loss_history.append(best_loss)
        final_grad_stats = grad_stats
        print(
            f"[train] step={step} loss={best_loss:.8g} "
            f"grad_norm={grad_norm:.3e} time={time.time() - t0:.1f}s"
        )

    _assert_strict_bounds(params)
    if not final_grad_stats:
        _loss, final_grads = eqx.filter_value_and_grad(loss_fn)(params)
        final_grad_stats = _finite_and_nonzero_param_grads(
            final_grads, tol=args.grad_nonzero_tol,
        )

    before_eval = _run_campaign_eval(
        campaign_tuned_cfg,
        ref,
        label="campaign-tuned",
        days=args.days,
        dt=args.dt,
        analysis_days=args.analysis_days,
    )
    trained_cfg = _materialize_static_config(campaign_tuned_cfg, params)
    after_eval = _run_campaign_eval(
        trained_cfg,
        ref,
        label="gradient-trained",
        days=args.days,
        dt=args.dt,
        analysis_days=args.analysis_days,
    )

    parameter_rows = _parameter_table(
        report_params_all=report_params_all,
        trained_params=params,
        records=records,
        frozen_reasons=frozen_reasons,
    )
    all_grad_stats = {**preflight_stats, **final_grad_stats}
    output = {
        "optimizer": {
            "name": args.optimizer,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "total_steps": args.steps,
            "grad_clip_norm": args.grad_clip_norm,
            "schedule": "linear warmup then cosine decay to zero",
        },
        "rollout": {
            "spinup_days": args.spinup_days,
            "train_days": args.train_days,
            "analysis_days": args.analysis_days,
            "fixed_spinup": args.fixed_spinup,
            "chunk_steps": args.chunk_steps,
        },
        "campaign_recommendation": {
            "recommended": str(args.recommended),
            "winners": winners,
            "active_tuned_scheme_keys": sorted(report_scheme_keys),
            "trained_scheme_keys": sorted(active_scheme_keys),
        },
        "bechtold_ad_treatment": bechtold_note,
        "reference": _jsonable(ref),
        "loss_history": loss_history,
        "loss": {
            "initial_train_loss": loss_history[0],
            "final_train_loss": loss_history[-1],
            "campaign_tuned_eval_loss": before_eval.score,
            "gradient_trained_eval_loss": after_eval.score,
            "target_loss": args.target_loss,
        },
        "gradient_check": {
            "tol": args.grad_nonzero_tol,
            "preflight_stats": preflight_stats,
            "final_stats": final_grad_stats,
            "stats": all_grad_stats,
            "all_trained_finite_nonzero": all(
                bool(all_grad_stats[row["name"]]["finite"])
                and bool(all_grad_stats[row["name"]]["nonzero"])
                for row in parameter_rows
                if row["trained"]
            ),
        },
        "parameters": parameter_rows,
        "campaign_tuned_config": _jsonable(campaign_tuned_cfg),
        "gradient_trained_config": _jsonable(trained_cfg),
        "before_eval": asdict(before_eval),
        "after_eval": asdict(after_eval),
    }
    (args.outdir / "trained_parameters.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    _plot_loss_curve(args.outdir / "loss_curve.png", loss_history)
    _plot_profiles_before_after(
        args.outdir / "profiles_vs_crm_before_after.png",
        ref,
        before_eval,
        after_eval,
    )
    _write_summary(
        args.outdir / "summary.md",
        args=args,
        ref=ref,
        loss_history=loss_history,
        before_eval=before_eval,
        after_eval=after_eval,
        grad_stats=preflight_stats,
        parameter_rows=parameter_rows,
        bechtold_note=bechtold_note,
    )
    print(f"[done] wrote {args.outdir}")
    print(
        f"[result] train_loss {loss_history[0]:.8g} -> {loss_history[-1]:.8g}; "
        f"eval_score {before_eval.score:.8g} -> {after_eval.score:.8g}"
    )
    return output


def _scale_updates(updates, scale: float):
    return jax.tree_util.tree_map(lambda x: x * scale, updates)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=campaign.DEFAULT_REFERENCE_DIR)
    parser.add_argument("--campaign-dir", type=Path, default=campaign.DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--recommended",
        "--recommended-defaults",
        dest="recommended",
        type=Path,
        default=None,
        help=(
            "Campaign recommended_defaults.json. Defaults to "
            "--campaign-dir/recommended_defaults.json."
        ),
    )
    parser.add_argument("--tuned-parameters", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--days", type=float, default=campaign.DEFAULT_DAYS)
    parser.add_argument("--spinup-days", type=float, default=DEFAULT_SPINUP_DAYS)
    parser.add_argument("--train-days", type=float, default=DEFAULT_TRAIN_DAYS)
    parser.add_argument("--analysis-days", type=float, default=campaign.DEFAULT_ANALYSIS_DAYS)
    parser.add_argument("--dt", type=float, default=campaign.DEFAULT_DT_S)
    parser.add_argument(
        "--last-reference-files",
        type=int,
        default=campaign.DEFAULT_LAST_REFERENCE_FILES,
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--optimizer",
        choices=("muon", "muon_partitioned", "adam", "adamw"),
        default="muon",
    )
    parser.add_argument("--chunk-steps", type=int, default=144)
    parser.add_argument(
        "--fixed-spinup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use one fixed campaign-tuned spinup carry before the "
            "differentiable final-window rollout."
        ),
    )
    parser.add_argument(
        "--bechtold-policy",
        choices=("auto", "freeze", "train_deterministic", "force", "mass_flux"),
        default="auto",
    )
    parser.add_argument("--target-loss", type=float, default=DEFAULT_TARGET_LOSS)
    parser.add_argument("--grad-nonzero-tol", type=float, default=GRAD_NONZERO_TOL)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.recommended is None:
        args.recommended = args.campaign_dir / "recommended_defaults.json"
    if args.tuned_parameters is None:
        args.tuned_parameters = args.campaign_dir / "tuned_parameters.json"
    if args.quick:
        args.days = min(args.days, QUICK_TRAIN_DAYS)
        args.spinup_days = 0.0
        args.train_days = min(args.train_days, QUICK_TRAIN_DAYS)
        args.analysis_days = min(args.analysis_days, args.train_days)
        args.steps = min(args.steps, QUICK_STEPS)
        args.chunk_steps = min(args.chunk_steps, 2)
        args.last_reference_files = min(args.last_reference_files, 2)
    if args.analysis_days > args.train_days:
        raise SystemExit("--analysis-days must be <= --train-days")
    if args.spinup_days < 0.0:
        raise SystemExit("--spinup-days must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
