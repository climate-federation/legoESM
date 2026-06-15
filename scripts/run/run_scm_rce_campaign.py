#!/usr/bin/env python
"""SCM RCE physics-scheme campaign against the plane-CRM RCEMIP ocean RCE.

The campaign builds equilibrium reference profiles from
``results/rcemip1_n128_ocean/snapshots3d/vol_*.npz`` and compares
single-column-model RCE equilibria against those profiles.  It is intentionally
script-local: this is a reproducible campaign driver, not a production model
API.

Run on CPU, for example:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \
        scripts/run/run_scm_rce_campaign.py

Use ``--quick`` for a short smoke run used by the unit test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import traceback
from dataclasses import asdict, dataclass, field
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
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
from legoesm.atmosphere.physics.microphysics.integration import min_tracer_slots
from legoesm.atmosphere.scm import SingleColumnModel
from legoesm.atmosphere.scm_forcing import SCMForcing
from legoesm.core.field import Field
from legoesm.grids.vertical import SigmaCoordinate

try:
    from legoesm.training.param_collector import (
        apply_param_overrides,
        build_registry,
        build_trainable_params,
    )
    from legoesm.training.trainable_params import (
        TrainablePhysicsParams,
        range_to_sigmoid_array,
    )
except ModuleNotFoundError:  # pragma: no cover - compatibility with older trees
    from legoesm.driver.param_collector import (  # type: ignore
        apply_param_overrides,
        build_registry,
        build_trainable_params,
    )
    from legoesm.training.trainable_params import (  # type: ignore
        TrainablePhysicsParams,
        range_to_sigmoid_array,
    )


SECONDS_PER_DAY = 86_400.0
MSE_KJ_TO_J = 1_000.0
M_PER_KM = 1_000.0
DEFAULT_RESULTS_DIR = Path("results/scm_rce_campaign")
DEFAULT_REFERENCE_DIR = Path("results/rcemip1_n128_ocean")
FIXED_SST_K = 300.0
DEFAULT_DT_S = 600.0
DEFAULT_DAYS = 50.0
DEFAULT_LAST_REFERENCE_FILES = 5
DEFAULT_ANALYSIS_DAYS = 5.0
QUICK_DAYS = 0.03
QUICK_TUNE_EVALS = 2
PROFILE_FLOOR = 1.0e-12
QV_NEGATIVE_TOL = -1.0e-8
COND_NEGATIVE_TOL = -1.0e-8
T_MIN_VALID_K = 150.0
T_MAX_VALID_K = 330.0
EQUIL_T_TOL_K = 1.0
EQUIL_QV_TOL = 5.0e-4
EQUIL_QCOND_TOL = 5.0e-5
DIMS_3D = ("face", "x", "y", "level")


BASELINE_SCHEMES = {
    "radiation": "gray",
    "convection": "mass_flux",
    "turbulence": "louis",
    "microphysics": "kessler",
    "gravity_wave_drag": "none",
}

SCHEME_SWEEPS = {
    "turbulence": (
        "louis", "smagorinsky", "tke", "mynn25", "clubb_lite", "clubb",
        "holtslag_boville", "ysu", "edmf",
    ),
    "microphysics": (
        "kessler", "sundqvist", "seifert_beheng", "morrison", "thompson",
        "p3",
    ),
    "gravity_wave_drag": (
        "none", "rayleigh", "lindzen", "mcfarlane", "hines",
        "e3sm_cam", "prognostic_spectral",
    ),
    "convection": (
        "sbm", "dca", "kuo", "mass_flux", "edmf", "zhang_mcfarlane",
        "kain_fritsch", "emanuel", "tiedtke", "bechtold",
    ),
}

QUICK_SCHEME_SWEEPS = {
    "turbulence": ("louis", "smagorinsky"),
    "microphysics": ("kessler", "sundqvist"),
    "gravity_wave_drag": ("none", "rayleigh"),
    "convection": ("mass_flux", "kuo"),
}

CATEGORY_CONFIG_FIELD = {
    "turbulence": "turbulence",
    "microphysics": "microphysics",
    "gravity_wave_drag": "gravity_wave_drag",
    "convection": "convection",
}


@dataclass
class ReferenceProfiles:
    z_m: np.ndarray
    sigma_half: np.ndarray
    sigma_full: np.ndarray
    mass_weights: np.ndarray
    T_ref: np.ndarray
    qv_ref: np.ndarray
    qcond_ref: np.ndarray
    files_used: list[str]
    sfc_cross_check: dict[str, float]


@dataclass
class RunDiagnostics:
    label: str
    config: dict[str, Any]
    status: str
    reason: str
    T_rmse: float
    qv_rmse: float
    cloud_rmse: float
    score: float
    drift_T_rmse_K: float
    drift_qv_rmse: float
    drift_qcond_rmse: float
    T_profile: list[float] = field(default_factory=list)
    qv_profile: list[float] = field(default_factory=list)
    qcond_profile: list[float] = field(default_factory=list)


@dataclass
class TuneRecord:
    category: str
    scheme: str
    scheme_key: str
    parameter: str
    default: float
    tuned: float
    lower: float
    upper: float
    units: str
    score_default: float
    score_tuned: float


def _require_cpu() -> None:
    platforms = os.environ.get("JAX_PLATFORMS", "")
    if platforms and platforms.lower() != "cpu":
        raise SystemExit(
            "This campaign must run CPU-only. Set JAX_PLATFORMS=cpu "
            f"(currently {platforms!r})."
        )
    jax.config.update("jax_enable_x64", True)


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if hasattr(obj, "_asdict"):
        return {k: _to_jsonable(v) for k, v in obj._asdict().items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.ndarray, jax.Array)):
        arr = np.asarray(obj)
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _weighted_std(profile: np.ndarray, weights: np.ndarray) -> float:
    mean = float(np.sum(weights * profile))
    var = float(np.sum(weights * (profile - mean) ** 2))
    return math.sqrt(max(var, 0.0))


def _weighted_rmse(diff: np.ndarray, weights: np.ndarray) -> float:
    return math.sqrt(float(np.sum(weights * diff ** 2)))


def _score_profiles(
    ref: ReferenceProfiles,
    T_profile: np.ndarray,
    qv_profile: np.ndarray,
    qcond_profile: np.ndarray,
) -> tuple[float, float, float, float]:
    weights = ref.mass_weights
    T_std = max(_weighted_std(ref.T_ref, weights), PROFILE_FLOOR)
    qv_std = max(_weighted_std(ref.qv_ref, weights), PROFILE_FLOOR)
    qcond_std = max(_weighted_std(ref.qcond_ref, weights), PROFILE_FLOOR)
    T_rmse = _weighted_rmse((T_profile - ref.T_ref) / T_std, weights)
    qv_rmse = _weighted_rmse((qv_profile - ref.qv_ref) / qv_std, weights)
    cloud_rmse = _weighted_rmse((qcond_profile - ref.qcond_ref) / qcond_std, weights)
    combined = math.sqrt((T_rmse**2 + qv_rmse**2 + cloud_rmse**2) / 3.0)
    return T_rmse, qv_rmse, cloud_rmse, combined


def _reference_files(reference_dir: Path, last_n: int) -> list[Path]:
    files = sorted((reference_dir / "snapshots3d").glob("vol_*.npz"))
    if len(files) < last_n:
        raise FileNotFoundError(
            f"Need at least {last_n} vol_*.npz files under "
            f"{reference_dir / 'snapshots3d'}, found {len(files)}."
        )
    return files[-last_n:]


def _surface_cross_check(
    reference_dir: Path,
    z_m: np.ndarray,
    T_ref: np.ndarray,
    qv_ref: np.ndarray,
    qcond_ref: np.ndarray,
) -> dict[str, float]:
    files = sorted((reference_dir / "snapshots").glob("sfc_*.npz"))
    if not files:
        return {}
    with np.load(files[-1]) as ds:
        if not {"heights", "T_levels", "qv_levels", "cond_levels"} <= set(ds.files):
            return {}
        h = np.asarray(ds["heights"], dtype=float)
        idx = [int(np.argmin(np.abs(z_m - zz))) for zz in h]
        T_levels = np.asarray(ds["T_levels"], dtype=float).mean(axis=(1, 2))
        qv_levels = np.asarray(ds["qv_levels"], dtype=float).mean(axis=(1, 2))
        cond_levels = np.asarray(ds["cond_levels"], dtype=float).mean(axis=(1, 2))
    return {
        "file": str(files[-1]),
        "max_abs_T_K": float(np.max(np.abs(T_ref[idx] - T_levels))),
        "max_abs_qv": float(np.max(np.abs(qv_ref[idx] - qv_levels))),
        "max_abs_qcond": float(np.max(np.abs(qcond_ref[idx] - cond_levels))),
    }


def build_reference_profiles(reference_dir: Path, last_n: int) -> ReferenceProfiles:
    files = _reference_files(reference_dir, last_n)
    T_acc = []
    qv_acc = []
    qcond_acc = []
    z_m = None
    for path in files:
        with np.load(path) as ds:
            z_cur = np.asarray(ds["z"], dtype=float)
            if z_m is None:
                z_m = z_cur
            elif not np.allclose(z_m, z_cur):
                raise ValueError(f"Reference z grid changed in {path}.")
            T = np.asarray(ds["T"], dtype=float)
            mse_J_kg = np.asarray(ds["mse"], dtype=float) * MSE_KJ_TO_J
            z_4d = z_cur.reshape(1, 1, -1)
            qv = (mse_J_kg - constants.c_pd * T - constants.g * z_4d) / constants.L_v
            T_acc.append(T.mean(axis=(0, 1)))
            qv_acc.append(qv.mean(axis=(0, 1)))
            qcond_acc.append(np.asarray(ds["cond"], dtype=float).mean(axis=(0, 1)))

    assert z_m is not None
    z_m = np.asarray(z_m, dtype=float)
    z_half = np.empty(z_m.size + 1, dtype=float)
    z_half[0] = z_m[0] + 0.5 * (z_m[0] - z_m[1])
    z_half[1:-1] = 0.5 * (z_m[:-1] + z_m[1:])
    z_half[-1] = 0.0
    sigma_half = np.asarray(
        wing2018_pressure_profile(jnp.asarray(z_half, dtype=jnp.float64))
        / WING_P_SFC,
        dtype=float,
    )
    if not np.all(np.diff(sigma_half) > 0.0):
        raise ValueError("Derived CRM-matching sigma half-levels are not monotone.")
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    dsigma = np.diff(sigma_half)
    mass_weights = dsigma / np.sum(dsigma)

    T_ref = np.mean(np.stack(T_acc), axis=0)
    qv_ref = np.mean(np.stack(qv_acc), axis=0)
    qcond_ref = np.maximum(np.mean(np.stack(qcond_acc), axis=0), 0.0)
    sfc_cross_check = _surface_cross_check(
        reference_dir, z_m, T_ref, qv_ref, qcond_ref,
    )
    return ReferenceProfiles(
        z_m=z_m,
        sigma_half=sigma_half,
        sigma_full=sigma_full,
        mass_weights=mass_weights,
        T_ref=T_ref,
        qv_ref=np.maximum(qv_ref, 0.0),
        qcond_ref=qcond_ref,
        files_used=[str(p) for p in files],
        sfc_cross_check=sfc_cross_check,
    )


def make_sigma_coordinate_from_reference(ref: ReferenceProfiles) -> SigmaCoordinate:
    sigma_half = jnp.asarray(ref.sigma_half, dtype=jnp.float64)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    dsigma = sigma_half[1:] - sigma_half[:-1]
    tiny = jnp.asarray(np.finfo(np.float32).tiny, dtype=jnp.float64)
    sigma_half_safe = jnp.clip(sigma_half, tiny, None)
    ln_ratio = jnp.log(sigma_half_safe[1:] / sigma_half_safe[:-1])
    alpha = 1.0 - (sigma_half_safe[:-1] / dsigma) * ln_ratio
    sigma_range = 1.0 - sigma_half[0]
    fractional_sigma = (sigma_half[1:] - sigma_half[0]) / sigma_range
    dsigma_full = jnp.diff(sigma_full)
    return SigmaCoordinate(
        n_levels=len(ref.z_m),
        sigma_full=sigma_full,
        sigma_half=sigma_half,
        dsigma=dsigma,
        ln_ratio=ln_ratio,
        alpha=alpha,
        fractional_sigma=fractional_sigma,
        dsigma_full=dsigma_full,
    )


def wing_initial_profiles(ref: ReferenceProfiles) -> tuple[jax.Array, jax.Array]:
    z = jnp.asarray(ref.z_m, dtype=jnp.float64)
    T = wing2018_temperature_profile(z)
    qv = wing2018_qv_profile(z)
    return T, qv


def make_physics_config(
    *,
    turbulence: str = BASELINE_SCHEMES["turbulence"],
    microphysics: str = BASELINE_SCHEMES["microphysics"],
    gravity_wave_drag: str = BASELINE_SCHEMES["gravity_wave_drag"],
    convection: str = BASELINE_SCHEMES["convection"],
    base: PhysicsConfig | None = None,
) -> PhysicsConfig:
    cfg = base if base is not None else PhysicsConfig()
    return cfg._replace(
        radiation=RadiationConfig(scheme="gray", gray=cfg.radiation.gray._replace(
            perpetual_equinox=True,
        )),
        convection=cfg.convection._replace(scheme=convection),
        turbulence=cfg.turbulence._replace(scheme=turbulence),
        microphysics=cfg.microphysics._replace(scheme=microphysics),
        gravity_wave_drag=cfg.gravity_wave_drag._replace(scheme=gravity_wave_drag),
    )


def _config_scheme_dict(cfg: PhysicsConfig) -> dict[str, str]:
    return {
        "radiation": cfg.radiation.scheme,
        "convection": cfg.convection.scheme,
        "turbulence": cfg.turbulence.scheme,
        "microphysics": cfg.microphysics.scheme,
        "gravity_wave_drag": cfg.gravity_wave_drag.scheme,
    }


def _config_cache_key(cfg: PhysicsConfig, days: float, dt: float) -> str:
    payload = {
        "days": days,
        "dt": dt,
        "config": _to_jsonable(cfg),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _preseed_column_tracers(scm: SingleColumnModel, microphysics_scheme: str) -> None:
    tracers = dict(scm.state.tracers or {})
    template = tracers["q_v"].data
    for name in ("q_c", "q_r", "q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
        if name not in tracers:
            tracers[name] = Field(
                data=jnp.zeros_like(template),
                name=name,
                dims=DIMS_3D,
                units="kg/kg" if not name.startswith("N_") else "1",
            )
    slots = min_tracer_slots(microphysics_scheme)
    if slots > len(("q_v", "q_c", "q_r", "q_i", "q_s", "q_g", "N_c", "N_r", "N_i")):
        raise ValueError(
            f"microphysics={microphysics_scheme!r} requires {slots} tracer slots, "
            "more than the SCM campaign standard allocation."
        )
    scm.state = scm.state._replace(tracers=tracers)


def _qcond_from_tracers(tracers: dict[str, Field], microphysics_scheme: str) -> jax.Array:
    qcond = jnp.zeros_like(tracers["q_v"].data)
    names = ("q_c", "q_r", "q_i", "q_s")
    if microphysics_scheme != "p3":
        names = names + ("q_g",)
    for name in names:
        if name in tracers:
            qcond = qcond + tracers[name].data
    return qcond


def run_scm_rce(
    cfg: PhysicsConfig,
    ref: ReferenceProfiles,
    *,
    label: str,
    days: float,
    dt: float,
    analysis_days: float,
    require_equilibrium: bool,
    equil_T_tol_K: float,
    equil_qv_tol: float,
    equil_qcond_tol: float,
) -> RunDiagnostics:
    nsteps = max(1, int(round(days * SECONDS_PER_DAY / dt)))
    T0, qv0 = wing_initial_profiles(ref)
    forcing = SCMForcing(prescribe="T_s", T_s=lambda _t: FIXED_SST_K)
    scm = SingleColumnModel.create(
        physics_config=cfg,
        nlev=len(ref.z_m),
        dt=dt,
        T_profile=T0,
        q_v_profile=qv0,
        p_s=WING_P_SFC,
        latitude_deg=0.0,
        time_integrator="forward_euler",
        forcing=forcing,
        dtype=jnp.float64,
    )
    scm.sigma_coord = make_sigma_coordinate_from_reference(ref)
    _preseed_column_tracers(scm, cfg.microphysics.scheme)

    sst_col = jnp.asarray([FIXED_SST_K], dtype=jnp.float64)
    forcing_dict = {"T_sfc": sst_col}
    step_fn = scm._step_fn
    physics_fn = scm.physics_fn
    grid = scm.grid
    sigma_coord = scm.sigma_coord
    dt_arr = jnp.asarray(dt, dtype=jnp.float64)
    microphysics_scheme = cfg.microphysics.scheme

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
            state, phys_state, fixed_sst_tendency, dt, t,
        )
        qv = new_state.tracers["q_v"].data[0, 0, 0]
        qcond = _qcond_from_tracers(
            new_state.tracers, microphysics_scheme,
        )[0, 0, 0]
        out = (new_state.T.data[0, 0, 0], qv, qcond)
        return (new_state, new_phys), out

    @jax.jit
    def driver(state0, phys0):
        (state_f, phys_f), history = lax.scan(
            body, (state0, phys0), jnp.arange(nsteps),
        )
        return state_f, phys_f, history

    try:
        final_state, _final_phys, history = driver(scm.state, scm.phys_state)
        del final_state
        T_hist, qv_hist, qcond_hist = (np.asarray(x, dtype=float) for x in history)
        last_steps = max(1, int(round(analysis_days * SECONDS_PER_DAY / dt)))
        last_steps = min(last_steps, nsteps)
        T_profile = T_hist[-last_steps:].mean(axis=0)
        qv_profile = qv_hist[-last_steps:].mean(axis=0)
        qcond_profile = np.maximum(qcond_hist[-last_steps:].mean(axis=0), 0.0)

        prev_end = nsteps - last_steps
        prev_start = max(0, prev_end - last_steps)
        if prev_end > prev_start:
            T_prev = T_hist[prev_start:prev_end].mean(axis=0)
            qv_prev = qv_hist[prev_start:prev_end].mean(axis=0)
            qcond_prev = np.maximum(qcond_hist[prev_start:prev_end].mean(axis=0), 0.0)
        else:
            T_prev = T_profile
            qv_prev = qv_profile
            qcond_prev = qcond_profile

        drift_T = _weighted_rmse(T_profile - T_prev, ref.mass_weights)
        drift_qv = _weighted_rmse(qv_profile - qv_prev, ref.mass_weights)
        drift_qcond = _weighted_rmse(qcond_profile - qcond_prev, ref.mass_weights)
        reasons = []
        status = "ok"
        finite = (
            np.all(np.isfinite(T_profile))
            and np.all(np.isfinite(qv_profile))
            and np.all(np.isfinite(qcond_profile))
        )
        if not finite:
            reasons.append("non-finite profile")
        if float(np.min(T_profile)) < T_MIN_VALID_K or float(np.max(T_profile)) > T_MAX_VALID_K:
            reasons.append(
                f"T outside [{T_MIN_VALID_K}, {T_MAX_VALID_K}] K "
                f"(min={float(np.min(T_profile)):.3g}, max={float(np.max(T_profile)):.3g})"
            )
        if float(np.min(qv_hist[-last_steps:])) < QV_NEGATIVE_TOL:
            reasons.append(f"negative q_v (min={float(np.min(qv_hist[-last_steps:])):.3g})")
        if float(np.min(qcond_hist[-last_steps:])) < COND_NEGATIVE_TOL:
            reasons.append(
                f"negative condensate (min={float(np.min(qcond_hist[-last_steps:])):.3g})"
            )
        if require_equilibrium and (
            drift_T > equil_T_tol_K
            or drift_qv > equil_qv_tol
            or drift_qcond > equil_qcond_tol
        ):
            reasons.append(
                "not equilibrated "
                f"(drift_T={drift_T:.3g} K, drift_qv={drift_qv:.3g}, "
                f"drift_qcond={drift_qcond:.3g})"
            )
        if reasons:
            status = "failed"
            score = float("inf")
            T_rmse = qv_rmse = cloud_rmse = float("inf")
        else:
            T_rmse, qv_rmse, cloud_rmse, score = _score_profiles(
                ref, T_profile, qv_profile, qcond_profile,
            )
        return RunDiagnostics(
            label=label,
            config=_config_scheme_dict(cfg),
            status=status,
            reason="; ".join(reasons),
            T_rmse=float(T_rmse),
            qv_rmse=float(qv_rmse),
            cloud_rmse=float(cloud_rmse),
            score=float(score),
            drift_T_rmse_K=float(drift_T),
            drift_qv_rmse=float(drift_qv),
            drift_qcond_rmse=float(drift_qcond),
            T_profile=T_profile.tolist(),
            qv_profile=qv_profile.tolist(),
            qcond_profile=qcond_profile.tolist(),
        )
    except Exception as exc:  # noqa: BLE001 - campaign records per-scheme failures
        return RunDiagnostics(
            label=label,
            config=_config_scheme_dict(cfg),
            status="crashed",
            reason=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
            T_rmse=float("inf"),
            qv_rmse=float("inf"),
            cloud_rmse=float("inf"),
            score=float("inf"),
            drift_T_rmse_K=float("inf"),
            drift_qv_rmse=float("inf"),
            drift_qcond_rmse=float("inf"),
        )


def run_cached(
    cache: dict[str, RunDiagnostics],
    cfg: PhysicsConfig,
    ref: ReferenceProfiles,
    *,
    label: str,
    days: float,
    dt: float,
    analysis_days: float,
    require_equilibrium: bool,
    equil_T_tol_K: float,
    equil_qv_tol: float,
    equil_qcond_tol: float,
) -> RunDiagnostics:
    key = _config_cache_key(cfg, days, dt)
    if key not in cache:
        cache[key] = run_scm_rce(
            cfg,
            ref,
            label=label,
            days=days,
            dt=dt,
            analysis_days=analysis_days,
            require_equilibrium=require_equilibrium,
            equil_T_tol_K=equil_T_tol_K,
            equil_qv_tol=equil_qv_tol,
            equil_qcond_tol=equil_qcond_tol,
        )
    cached = cache[key]
    return RunDiagnostics(
        **{**asdict(cached), "label": label, "config": _config_scheme_dict(cfg)}
    )


def _sort_category_results(category: str, results: list[RunDiagnostics]) -> list[RunDiagnostics]:
    status_rank = {"ok": 0, "failed": 1, "crashed": 2}
    scheme_order = {s: i for i, s in enumerate(SCHEME_SWEEPS[category])}
    baseline_scheme = BASELINE_SCHEMES[category]
    finite_scores = [r.score for r in results if math.isfinite(r.score)]
    best_score = min(finite_scores) if finite_scores else float("inf")

    def score_bin(r: RunDiagnostics) -> float:
        if not math.isfinite(r.score):
            return float("inf")
        if abs(r.score - best_score) <= 1.0e-12:
            return best_score
        return r.score

    return sorted(
        results,
        key=lambda r: (
            status_rank.get(r.status, 3),
            score_bin(r),
            0 if r.config[CATEGORY_CONFIG_FIELD[category]] == baseline_scheme else 1,
            scheme_order.get(r.config[CATEGORY_CONFIG_FIELD[category]], 10_000),
        ),
    )


def _write_ranking_csv(path: Path, category: str, results: list[RunDiagnostics]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category", "scheme", "T_RMSE", "qv_RMSE", "cloud_RMSE",
                "combined_score", "status", "reason", "drift_T_rmse_K",
                "drift_qv_rmse", "drift_qcond_rmse",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "category": category,
                    "scheme": r.config[CATEGORY_CONFIG_FIELD[category]],
                    "T_RMSE": r.T_rmse,
                    "qv_RMSE": r.qv_rmse,
                    "cloud_RMSE": r.cloud_rmse,
                    "combined_score": r.score,
                    "status": r.status,
                    "reason": r.reason,
                    "drift_T_rmse_K": r.drift_T_rmse_K,
                    "drift_qv_rmse": r.drift_qv_rmse,
                    "drift_qcond_rmse": r.drift_qcond_rmse,
                }
            )


def _plot_ranking(path: Path, category: str, results: list[RunDiagnostics]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] skipped ranking plot {path}: {exc}")
        return
    schemes = [r.config[CATEGORY_CONFIG_FIELD[category]] for r in results]
    scores = [r.score if math.isfinite(r.score) else np.nan for r in results]
    colors = ["#2a9d8f" if r.status == "ok" else "#b0b0b0" for r in results]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(results)), 4.2))
    ax.bar(np.arange(len(results)), scores, color=colors)
    ax.set_xticks(np.arange(len(results)))
    ax.set_xticklabels(schemes, rotation=40, ha="right")
    ax.set_ylabel("combined normalized RMSE")
    ax.set_title(f"SCM RCE {category} ranking")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_profiles(path: Path, ref: ReferenceProfiles, run: RunDiagnostics, title: str) -> None:
    if not run.T_profile:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] skipped profile plot {path}: {exc}")
        return
    z_km = ref.z_m / M_PER_KM
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.2), sharey=True)
    panels = [
        ("T", ref.T_ref, np.asarray(run.T_profile), "K"),
        ("qv", ref.qv_ref * MSE_KJ_TO_J, np.asarray(run.qv_profile) * MSE_KJ_TO_J, "g/kg"),
        (
            "condensate",
            ref.qcond_ref * MSE_KJ_TO_J,
            np.asarray(run.qcond_profile) * MSE_KJ_TO_J,
            "g/kg",
        ),
    ]
    for ax, (name, ref_prof, scm_prof, units) in zip(axes, panels):
        ax.plot(ref_prof, z_km, color="#1f4e79", lw=2.0, label="CRM")
        ax.plot(scm_prof, z_km, color="#d95f02", lw=2.0, label="SCM")
        ax.set_xlabel(f"{name} [{units}]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("z [km]")
    axes[0].invert_yaxis()
    axes[0].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _active_subconfig(top_cfg: PhysicsConfig, category: str):
    component = getattr(top_cfg, CATEGORY_CONFIG_FIELD[category])
    scheme = component.scheme
    subcfg = getattr(component, scheme, None)
    if subcfg is None and category == "turbulence" and scheme == "clubb":
        from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig

        subcfg = CLUBBConfig()
    return component, scheme, subcfg


def _scheme_key_for_subconfig(subcfg) -> str | None:
    if subcfg is None:
        return None
    cls_name = type(subcfg).__name__
    for meta in build_registry():
        if meta.config_class == cls_name:
            return meta.scheme_key
    return None


def _set_active_subconfig(top_cfg: PhysicsConfig, category: str, subcfg) -> PhysicsConfig:
    component = getattr(top_cfg, CATEGORY_CONFIG_FIELD[category])
    scheme = component.scheme
    component_new = component._replace(**{scheme: subcfg})
    return top_cfg._replace(**{CATEGORY_CONFIG_FIELD[category]: component_new})


def _raw_from_physical(value: float, constraint) -> jax.Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    if constraint.transform == "sigmoid":
        return range_to_sigmoid_array(arr, constraint.min_val, constraint.max_val).astype(jnp.float64)
    if constraint.transform == "softplus":
        y = jnp.maximum(arr, jnp.asarray(PROFILE_FLOOR, dtype=jnp.float64))
        return (y + jnp.log1p(-jnp.exp(-y))).astype(jnp.float64)
    return arr


def _candidate_values(defaults: dict[str, float], constraints, n_eval: int, seed: int):
    yield defaults
    if n_eval <= 1:
        return
    rng = np.random.default_rng(seed)
    names = list(defaults)
    budget = n_eval - 1
    for frac in (0.25, 0.75):
        for name, c in zip(names, constraints):
            if budget <= 0:
                return
            lo = float(c.min_val)
            hi = float(c.max_val)
            cand = dict(defaults)
            cand[name] = lo + frac * (hi - lo)
            budget -= 1
            yield cand
    while budget > 0:
        cand = {}
        for name, c in zip(names, constraints):
            lo = float(c.min_val)
            hi = float(c.max_val)
            cand[name] = float(rng.uniform(lo, hi))
        budget -= 1
        yield cand


def tune_category_winner(
    category: str,
    base_cfg: PhysicsConfig,
    ref: ReferenceProfiles,
    cache: dict[str, RunDiagnostics],
    *,
    days: float,
    dt: float,
    analysis_days: float,
    require_equilibrium: bool,
    tune_evals: int,
    seed: int,
    equil_T_tol_K: float,
    equil_qv_tol: float,
    equil_qcond_tol: float,
) -> tuple[PhysicsConfig, list[TuneRecord], RunDiagnostics]:
    _component, scheme, subcfg = _active_subconfig(base_cfg, category)
    scheme_key = _scheme_key_for_subconfig(subcfg)
    default_run = run_cached(
        cache,
        base_cfg,
        ref,
        label=f"tune-default:{category}:{scheme}",
        days=days,
        dt=dt,
        analysis_days=analysis_days,
        require_equilibrium=require_equilibrium,
        equil_T_tol_K=equil_T_tol_K,
        equil_qv_tol=equil_qv_tol,
        equil_qcond_tol=equil_qcond_tol,
    )
    if scheme_key is None or subcfg is None:
        return base_cfg, [], default_run
    params = build_trainable_params(
        active_scheme_keys={scheme_key},
        tier="extended",
        dtype=jnp.float64,
    )
    if not params.constraints:
        return base_cfg, [], default_run
    constraints = params.constraints
    defaults = {c.name: float(params.as_dict()[c.name]) for c in constraints}
    best_cfg = base_cfg
    best_run = default_run
    best_values = defaults
    for i, values in enumerate(_candidate_values(defaults, constraints, tune_evals, seed)):
        raw_values = {
            c.name: _raw_from_physical(values[c.name], c) for c in constraints
        }
        trial_params = TrainablePhysicsParams(
            raw_values=raw_values,
            constraints=constraints,
        )
        overrides = trial_params.to_overrides()
        field_values = overrides.get(scheme_key, {})
        for c in constraints:
            value = float(field_values[c.field])
            lo = float(c.min_val)
            hi = float(c.max_val)
            if not (lo <= value <= hi):
                raise AssertionError(
                    f"{c.name}={value} escaped bounds [{lo}, {hi}]"
                )
        tuned_subcfg = apply_param_overrides(subcfg, field_values)
        trial_cfg = _set_active_subconfig(base_cfg, category, tuned_subcfg)
        trial_run = run_cached(
            cache,
            trial_cfg,
            ref,
            label=f"tune:{category}:{scheme}:eval{i:03d}",
            days=days,
            dt=dt,
            analysis_days=analysis_days,
            require_equilibrium=require_equilibrium,
            equil_T_tol_K=equil_T_tol_K,
            equil_qv_tol=equil_qv_tol,
            equil_qcond_tol=equil_qcond_tol,
        )
        if trial_run.status == "ok" and trial_run.score < best_run.score:
            best_cfg = trial_cfg
            best_run = trial_run
            best_values = {
                c.name: float(field_values[c.field]) for c in constraints
            }
    records = []
    meta_by_name = {m.qualified_name: m for m in build_registry()}
    for c in constraints:
        meta = meta_by_name[c.name]
        tuned = float(best_values[c.name])
        lo = float(c.min_val)
        hi = float(c.max_val)
        if not (lo <= tuned <= hi):
            raise AssertionError(f"tuned value escaped bounds: {c.name}={tuned}")
        records.append(
            TuneRecord(
                category=category,
                scheme=scheme,
                scheme_key=scheme_key,
                parameter=c.field,
                default=float(defaults[c.name]),
                tuned=tuned,
                lower=lo,
                upper=hi,
                units=meta.units,
                score_default=float(default_run.score),
                score_tuned=float(best_run.score),
            )
        )
    return best_cfg, records, best_run


def _parse_categories(raw: str, include_convection: bool) -> list[str]:
    if raw == "required":
        cats = ["turbulence", "microphysics", "gravity_wave_drag"]
    elif raw == "all":
        cats = ["turbulence", "microphysics", "gravity_wave_drag", "convection"]
    else:
        cats = [x.strip() for x in raw.split(",") if x.strip()]
    if include_convection and "convection" not in cats:
        cats.append("convection")
    unknown = [c for c in cats if c not in SCHEME_SWEEPS]
    if unknown:
        raise SystemExit(f"Unknown categories {unknown}; known={sorted(SCHEME_SWEEPS)}")
    return cats


def _scheme_sweeps_for(args, categories: list[str]) -> dict[str, tuple[str, ...]]:
    source = QUICK_SCHEME_SWEEPS if args.quick else SCHEME_SWEEPS
    sweeps = {cat: source[cat] for cat in categories}
    if args.only:
        selected = tuple(x.strip() for x in args.only.split(",") if x.strip())
        sweeps = {
            cat: tuple(s for s in schemes if s in selected)
            for cat, schemes in sweeps.items()
        }
    empty = [cat for cat, schemes in sweeps.items() if not schemes]
    if empty:
        raise SystemExit(f"No schemes selected for categories: {empty}")
    return sweeps


def _write_summary(
    path: Path,
    ref: ReferenceProfiles,
    rankings: dict[str, list[RunDiagnostics]],
    baseline: RunDiagnostics,
    best: RunDiagnostics,
    tuned_best: RunDiagnostics | None,
    tuned_records: list[TuneRecord],
    recommended_cfg: PhysicsConfig,
    args,
) -> None:
    lines = [
        "# SCM RCE Campaign Summary",
        "",
        "Reference profiles: horizontal/time mean of the last "
        f"{len(ref.files_used)} CRM 3-D daily volumes:",
    ]
    lines += [f"- `{p}`" for p in ref.files_used]
    lines += [
        "",
        "The CRM `mse` diagnostic is stored in kJ/kg; qv was derived as "
        "`(1000*mse - c_pd*T - g*z)/L_v` using `legoesm.constants`.",
        "The full campaign uses `SCMForcing(prescribe='T_s')` at construction "
        "and a JIT-compatible fixed-SST scan that feeds the same 300 K boundary "
        "through traced radiation forcing plus `PhysicsState.surface_T_sfc_override` "
        "for turbulence.",
        f"SCM fixed SST: {FIXED_SST_K:.1f} K; dt: {args.dt:.1f} s; "
        f"days: {args.days:.3g}; analysis window: {args.analysis_days:.3g} d.",
        "",
        "Recommended defaults live in the atmosphere physics `*Config` "
        "NamedTuple defaults and the combined `PhysicsConfig`; this campaign "
        "writes a recommended override JSON, it does not mutate production "
        "defaults.",
        "",
        "## Rankings",
    ]
    for category, rows in rankings.items():
        heading = "convection (bonus)" if category == "convection" else category
        lines.append(f"### {heading}")
        lines.append("| rank | scheme | score | T | qv | cloud | status | reason |")
        lines.append("|---:|---|---:|---:|---:|---:|---|---|")
        for i, r in enumerate(rows, start=1):
            scheme = r.config[CATEGORY_CONFIG_FIELD[category]]
            score = "inf" if not math.isfinite(r.score) else f"{r.score:.6g}"
            T = "inf" if not math.isfinite(r.T_rmse) else f"{r.T_rmse:.6g}"
            qv = "inf" if not math.isfinite(r.qv_rmse) else f"{r.qv_rmse:.6g}"
            cloud = "inf" if not math.isfinite(r.cloud_rmse) else f"{r.cloud_rmse:.6g}"
            lines.append(
                f"| {i} | {scheme} | {score} | {T} | {qv} | {cloud} | "
                f"{r.status} | {r.reason.replace('|', '/')} |"
            )
        lines.append("")
    lines += [
        "## Best Config",
        "",
        f"Baseline score: {baseline.score:.6g} ({baseline.status}).",
        f"Per-category best score: {best.score:.6g} ({best.status}).",
    ]
    if tuned_best is not None:
        lines.append(f"Tuned-best score: {tuned_best.score:.6g} ({tuned_best.status}).")
    lines += [
        "",
        "Recommended scheme selection:",
        "",
        "```json",
        json.dumps(_config_scheme_dict(recommended_cfg), indent=2, sort_keys=True),
        "```",
        "",
        "Full nested config is written to `recommended_defaults.json`.",
    ]
    if tuned_records:
        lines += ["", "## Tuned Parameters", ""]
        lines.append("| category | scheme | parameter | default | tuned | bounds | units |")
        lines.append("|---|---|---|---:|---:|---|---|")
        for r in tuned_records:
            lines.append(
                f"| {r.category} | {r.scheme} | {r.parameter} | {r.default:.6g} | "
                f"{r.tuned:.6g} | [{r.lower:.6g}, {r.upper:.6g}] | {r.units} |"
            )
    if ref.sfc_cross_check:
        lines += [
            "",
            "## Surface Snapshot Cross-Check",
            "",
            json.dumps(ref.sfc_cross_check, indent=2, sort_keys=True),
        ]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    _require_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--days", type=float, default=DEFAULT_DAYS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--analysis-days", type=float, default=DEFAULT_ANALYSIS_DAYS)
    parser.add_argument("--last-reference-files", type=int, default=DEFAULT_LAST_REFERENCE_FILES)
    parser.add_argument(
        "--categories",
        default="all",
        help="'required', 'all', or comma-separated categories.",
    )
    parser.add_argument("--include-convection", action="store_true")
    parser.add_argument("--only", default=None, help="comma-separated global scheme subset")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tune-evals", type=int, default=32)
    parser.add_argument("--tune-seed", type=int, default=20260615)
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--equil-T-tol-K", type=float, default=EQUIL_T_TOL_K)
    parser.add_argument("--equil-qv-tol", type=float, default=EQUIL_QV_TOL)
    parser.add_argument("--equil-qcond-tol", type=float, default=EQUIL_QCOND_TOL)
    args = parser.parse_args(argv)

    if args.quick:
        args.days = min(args.days, QUICK_DAYS)
        args.analysis_days = min(args.analysis_days, args.days)
        args.tune_evals = min(args.tune_evals, QUICK_TUNE_EVALS)
    require_equilibrium = not args.quick
    args.outdir.mkdir(parents=True, exist_ok=True)

    ref = build_reference_profiles(args.reference_dir, args.last_reference_files)
    categories = _parse_categories(args.categories, args.include_convection)
    sweeps = _scheme_sweeps_for(args, categories)
    run_cache: dict[str, RunDiagnostics] = {}

    baseline_cfg = make_physics_config()
    baseline = run_cached(
        run_cache,
        baseline_cfg,
        ref,
        label="baseline",
        days=args.days,
        dt=args.dt,
        analysis_days=args.analysis_days,
        require_equilibrium=require_equilibrium,
        equil_T_tol_K=args.equil_T_tol_K,
        equil_qv_tol=args.equil_qv_tol,
        equil_qcond_tol=args.equil_qcond_tol,
    )
    print(f"[baseline] {baseline.status} score={baseline.score:.6g} {baseline.reason}")

    rankings: dict[str, list[RunDiagnostics]] = {}
    winners: dict[str, str] = {}
    for category, schemes in sweeps.items():
        rows = []
        for scheme in schemes:
            scheme_kwargs = dict(BASELINE_SCHEMES)
            scheme_kwargs[category] = scheme
            cfg = make_physics_config(
                turbulence=scheme_kwargs["turbulence"],
                microphysics=scheme_kwargs["microphysics"],
                gravity_wave_drag=scheme_kwargs["gravity_wave_drag"],
                convection=scheme_kwargs["convection"],
            )
            print(f"[run] {category}={scheme}")
            rows.append(
                run_cached(
                    run_cache,
                    cfg,
                    ref,
                    label=f"{category}:{scheme}",
                    days=args.days,
                    dt=args.dt,
                    analysis_days=args.analysis_days,
                    require_equilibrium=require_equilibrium,
                    equil_T_tol_K=args.equil_T_tol_K,
                    equil_qv_tol=args.equil_qv_tol,
                    equil_qcond_tol=args.equil_qcond_tol,
                )
            )
            print(
                f"       -> {rows[-1].status} score={rows[-1].score:.6g} "
                f"{rows[-1].reason.splitlines()[0] if rows[-1].reason else ''}"
            )
        rows = _sort_category_results(category, rows)
        rankings[category] = rows
        ok_rows = [r for r in rows if r.status == "ok"]
        winner = ok_rows[0] if ok_rows else rows[0]
        winners[category] = winner.config[CATEGORY_CONFIG_FIELD[category]]
        _write_ranking_csv(args.outdir / f"ranking_{category}.csv", category, rows)
        _plot_ranking(args.outdir / f"ranking_{category}.png", category, rows)

    best_kwargs = dict(BASELINE_SCHEMES)
    best_kwargs.update(winners)
    best_cfg = make_physics_config(
        turbulence=best_kwargs["turbulence"],
        microphysics=best_kwargs["microphysics"],
        gravity_wave_drag=best_kwargs["gravity_wave_drag"],
        convection=best_kwargs["convection"],
    )
    best = run_cached(
        run_cache,
        best_cfg,
        ref,
        label="best-per-category",
        days=args.days,
        dt=args.dt,
        analysis_days=args.analysis_days,
        require_equilibrium=require_equilibrium,
        equil_T_tol_K=args.equil_T_tol_K,
        equil_qv_tol=args.equil_qv_tol,
        equil_qcond_tol=args.equil_qcond_tol,
    )
    print(f"[best] {best.status} score={best.score:.6g} {best.reason}")

    tuned_records: list[TuneRecord] = []
    tuned_cfg = best_cfg
    tuned_best = None
    if not args.skip_tuning:
        for category in categories:
            print(f"[tune] {category}={winners[category]}")
            tuned_cfg, records, tuned_run = tune_category_winner(
                category,
                tuned_cfg,
                ref,
                run_cache,
                days=args.days,
                dt=args.dt,
                analysis_days=args.analysis_days,
                require_equilibrium=require_equilibrium,
                tune_evals=args.tune_evals,
                seed=args.tune_seed + len(tuned_records),
                equil_T_tol_K=args.equil_T_tol_K,
                equil_qv_tol=args.equil_qv_tol,
                equil_qcond_tol=args.equil_qcond_tol,
            )
            tuned_records.extend(records)
            tuned_best = tuned_run
            print(
                f"       -> {len(records)} params, score={tuned_run.score:.6g} "
                f"{tuned_run.status}"
            )

    recommended = {
        "baseline": asdict(baseline),
        "best_per_category": asdict(best),
        "tuned_best": asdict(tuned_best) if tuned_best is not None else None,
        "winners": winners,
        "recommended_physics_config": _to_jsonable(tuned_cfg),
        "reference": _to_jsonable(ref),
    }
    (args.outdir / "recommended_defaults.json").write_text(
        json.dumps(recommended, indent=2, sort_keys=True) + "\n"
    )
    (args.outdir / "tuned_parameters.json").write_text(
        json.dumps([asdict(r) for r in tuned_records], indent=2, sort_keys=True) + "\n"
    )
    _plot_profiles(args.outdir / "profiles_best_vs_crm.png", ref, best, "SCM best vs CRM")
    if tuned_best is not None:
        _plot_profiles(
            args.outdir / "profiles_tuned_best_vs_crm.png",
            ref,
            tuned_best,
            "SCM tuned best vs CRM",
        )
    _write_summary(
        args.outdir / "summary.md",
        ref,
        rankings,
        baseline,
        best,
        tuned_best,
        tuned_records,
        tuned_cfg,
        args,
    )
    print(f"[done] wrote {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
