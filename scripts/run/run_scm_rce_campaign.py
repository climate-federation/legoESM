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
from collections.abc import Sequence
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
    make_physics,
)
from legoesm.atmosphere.physics._shared import (
    compute_layer_dz as _compute_layer_dz,
    compute_rho as _compute_rho,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.microphysics.integration import (
    get_microphysics_fn,
    min_tracer_slots,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig,
    OzoneProfileConfig,
    RRTMGPConfig,
)
from legoesm.atmosphere.physics.radiation.integration import sam_ocean_albedo
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel, apply_tendencies
from legoesm.atmosphere.forcing.scm.scm_forcing import (
    SCMForcing,
    add_tendencies,
    compute_forcing_tendencies,
)
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticTendencies
from legoesm.grids.vertical import SigmaCoordinate
from legoesm.training.scm_rce_metrics import (
    COLD_POINT_MAX_K,
    COLD_POINT_MIN_K,
    DEFAULT_SUBCLOUD_TOP_M,
    DEFAULT_THERMO_MIN_P_PA,
    DEFAULT_THERMO_RH_BLEND_WIDTH_K,
    DEFAULT_THERMO_RH_SCALE,
    DEFAULT_THERMO_T_SCALE_K,
    MADIAB_MAX_TOL_K,
    MADIAB_MEAN_TOL_K,
    PRECIP_NORMALIZATION_MM_DAY,
    PRECIP_SCORE_WEIGHT,
    TROP_MAX_Z_KM,
    TROP_MIN_Z_KM,
    moist_adiabat_diagnostics_jax,
    realism_reasons_from_diagnostics,
    relative_humidity_profile,
    score_profiles_precip_jax,
    score_thermo_jax,
    tropospheric_mass_weights,
    subcloud_bulk_state,
    subcloud_objective_jax,
    subcloud_mass_weights,
    weighted_rmse as weighted_rmse_jax,
    weighted_std as weighted_std_jax,
)

# Each fallback below narrows on ``exc.name``: only the module that actually
# moved may trigger it.  Any OTHER ModuleNotFoundError (a broken install, a
# renamed transitive dependency) must surface as itself rather than being
# rerouted to an obsolete path and reported as the wrong error.
try:
    from legoesm.core.param_overrides import apply_param_overrides
except ModuleNotFoundError as exc:  # pragma: no cover - older trees
    if exc.name != "legoesm.core.param_overrides":
        raise
    try:  # the helper used to live next to the collector
        from legoesm.training.param_collector import (  # type: ignore
            apply_param_overrides,
        )
    except ModuleNotFoundError as exc2:
        if exc2.name != "legoesm.training.param_collector":
            raise
        from legoesm.driver.param_collector import (  # type: ignore
            apply_param_overrides,
        )

try:
    from legoesm.training.param_collector import (
        build_registry,
        build_trainable_params,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - older trees
    if exc.name != "legoesm.training.param_collector":
        raise
    from legoesm.driver.param_collector import (  # type: ignore
        build_registry,
        build_trainable_params,
    )

from legoesm.training.trainable_params import (
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
DEFAULT_SCM_MICROPHYSICS_SUBSTEPS = 30
SCM_MICROPHYSICS_SUBSTEP_SCHEMES = (
    "kessler",
    "sundqvist",
    "seifert_beheng",
    "morrison",
    "thompson",
    "p3",
)
DEFAULT_DAYS = 100.0
DEFAULT_LAST_REFERENCE_FILES = 5
DEFAULT_ANALYSIS_DAYS = 5.0
DEFAULT_RRTMGP_UPDATE_INTERVAL_STEPS = 72
DEFAULT_SCM_RCE_SURFACE_WIND_M_S = 5.0
DEFAULT_SCM_RCE_CORIOLIS_S_INV = 2.5e-5
# Depth over which the boundary layer is anchored to an SST-rooted lapse-rate
# profile.  0.0 READS as "anchor disabled" and was surely meant that way, but
# the mask below is `z_above_lowest <= BL_TOP_M` and z_above_lowest is exactly
# 0 at the lowest level, so 0.0 anchors EXACTLY that level: it is reset to
# 300.000 K every step, T_a == T_sfc identically, and the sensible heat flux
# is therefore IDENTICALLY ZERO by construction (measured: SHF = 0.000 W/m^2
# in every configuration, jobs 9361582/9361587/9361599).  The near-surface air
# then cannot respond to radiation, convection or turbulence, and q_sat there
# is pinned, which constrains RH and hence evaporation.
#
# The default is UNCHANGED so the 2026-08-10 arms remain reproducible; use
# `--bl-anchor-top-m -1` (any negative value) to disable the anchor entirely,
# which is what a zero depth was meant to express.
DEFAULT_SCM_RCE_BL_TOP_M = 0.0
DEFAULT_SCM_RCE_BL_LAPSE_K_M = 6.5e-3
DEFAULT_SCM_RCE_BL_MIN_T_K = 285.0
DEFAULT_SCM_CONVECTION_SUBSTEPS = 10
CRM_CLEAR_SKY_COND_THRESHOLD = 1.0e-6
SCM_RCE_LARGE_SCALE_FORCING_CHOICES = ("none", "crm_clear_sky_subsidence")
DEFAULT_SCM_RCE_LARGE_SCALE_FORCING = "none"
# EVERY convection scheme sub-steps identically. This used to be an opt-in
# tuple naming 6 of the 10 schemes; the other 4 (sbm, kuo, mass_flux, edmf)
# silently fell through to a 600 s convective step while these 6 took 60 s.
#
# That made the intercomparison UNCONTROLLED: a score gap between, say, sbm and
# bechtold conflated the scheme with a 10x difference in convective timestep,
# and no kernel arm controlled for it. Ranking schemes under different
# timesteps measures the timestep as much as the physics.
#
# The gate is now removed rather than extended to all 10 names, deliberately:
# a tuple that must list every scheme is a drift hazard — the next scheme added
# would silently inherit the 600 s step and reintroduce exactly this bug. With
# no gate, uniformity is structural.
#
# Retained as an explicitly empty marker so the two operator-facing messages
# that used to enumerate it keep working and now state the uniform behaviour.
SCM_CONVECTION_SUBSTEP_SCHEMES = ()  # (unused: all schemes sub-step uniformly)
QUICK_DAYS = 0.03
QUICK_TUNE_EVALS = 2
RCEMIP_S0_W_M2 = 551.58
RCEMIP_COS_ZENITH = 0.7425
CRM_PRECIP_RAW_TO_MM_DAY_THRESHOLD = 1.0e-3
PRECIP_MAX_SANE_MM_DAY = 25.0
SCM_PROGNOSTIC_SPECTRAL_GWD_THERMAL_TENDENCY = False
PROFILE_FLOOR = 1.0e-12
QV_NEGATIVE_TOL = -1.0e-8
COND_NEGATIVE_TOL = -1.0e-7
T_MIN_VALID_K = 150.0
T_MAX_VALID_K = 330.0
EQUIL_T_TOL_K = 1.0
EQUIL_QV_TOL = 5.0e-4
EQUIL_QCOND_TOL = 5.0e-5
DIMS_3D = ("face", "x", "y", "level")


BASELINE_SCHEMES = {
    "radiation": "rrtmgp",
    "convection": "dca",
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
    precip_ref_mm_day: float = float("nan")
    #: Surface evaporation of the REFERENCE [mm/day].  Preferably the
    #: archive's own measured latent heat flux; NaN when no such file is
    #: present, in which case the sub-cloud objective falls back to the
    #: equilibrium identity E = P and the provenance string says so.
    evap_ref_mm_day: float = float("nan")
    evap_ref_note: str = ""
    precip_files_used: list[str] = field(default_factory=list)
    precip_unit_note: str = ""
    crm_clear_sky_subsidence_m_s: np.ndarray | None = None
    crm_cloud_fraction: np.ndarray | None = None


@dataclass
class RunDiagnostics:
    label: str
    config: dict[str, Any]
    status: str
    reason: str
    T_rmse: float
    qv_rmse: float
    cloud_rmse: float
    precip_rmse: float
    score: float
    drift_T_rmse_K: float
    drift_qv_rmse: float
    drift_qcond_rmse: float
    precip_mm_day: float = float("nan")
    # Surface evaporation [mm/day], co-sampled with precip_mm_day over the same
    # analysis window from the same applied tendency, so d(CWV+CWC)/dt = E - P
    # can be checked without comparing quantities sampled differently.
    evap_mm_day: float = float("nan")
    precip_ref_mm_day: float = float("nan")
    moist_adiabat_mean_abs_K: float = float("nan")
    moist_adiabat_max_abs_K: float = float("nan")
    moist_adiabat_bias_K: float = float("nan")
    cold_point_T_K: float = float("nan")
    cold_point_z_km: float = float("nan")
    realism_status: str = "not_checked"
    T_profile: list[float] = field(default_factory=list)
    qv_profile: list[float] = field(default_factory=list)
    qcond_profile: list[float] = field(default_factory=list)
    #: The two halves ``qcond_profile`` sums to, so an excess can be attributed
    #: to detrained cloud or to un-sedimented precipitation rather than only
    #: reported in total.  Comparable term by term against the reference's own
    #: ``qcloud`` (clw+cli) and ``cond - qcloud`` (plw+pli).
    qcloud_profile: list[float] = field(default_factory=list)
    qprecip_profile: list[float] = field(default_factory=list)
    #: Sub-cloud-layer profile errors, normalized by the reference's own
    #: sub-cloud spread.  Reported for every run and optionally MINIMISED
    #: (``--objective subcloud``), because the full-column score is condensate-
    #: dominated and moves by less than the tuner's noise when this layer
    #: changes.
    subcloud_T_rmse: float = float("nan")
    subcloud_qv_rmse: float = float("nan")
    subcloud_evap_term: float = float("nan")
    subcloud_score: float = float("nan")
    subcloud_top_m: float = float("nan")
    subcloud_n_levels: int = 0
    #: Lowest-level bulk-flux state: what the surface fluxes actually see.
    #: ``sfc_driver_kg_kg`` is ``r_sat(SST) - r_air``, the humidity difference
    #: bulk evaporation is proportional to; ``sfc_delta_T_K`` is ``SST - T_air``.
    sfc_relative_humidity: float = float("nan")
    sfc_driver_kg_kg: float = float("nan")
    sfc_delta_T_K: float = float("nan")
    #: TEMPERATURE-AND-HUMIDITY objective (``--objective thermo``).  Full-column
    #: temperature plus TROPOSPHERIC RELATIVE humidity, each divided by a fixed
    #: physical tolerance (1 K, 5 % RH) rather than by the reference's own
    #: spread.  Reported for every run whether or not it is the one minimised,
    #: so an arm tuned on one objective can still be read on the other.
    thermo_T_term: float = float("nan")
    thermo_rh_term: float = float("nan")
    thermo_score: float = float("nan")
    #: Physical-unit companions to the terms above: mean absolute RH error over
    #: the same masked levels, and the number of levels the mask kept.  A score
    #: is not interpretable in a caption; these are.
    trop_rh_rmse: float = float("nan")
    trop_qv_rmse_g_kg: float = float("nan")
    trop_n_levels: int = 0
    #: The SAME thermo score evaluated on a HELD-OUT analysis window ending one
    #: window earlier.  A parameter set that matches the CRM only during the
    #: window it was tuned on — because the column is still drifting, or because
    #: the scheme is intermittent and one favourable phase happened to land
    #: there — shows up as a gap between these two numbers and nowhere else.
    heldout_thermo_score: float = float("nan")
    #: PHYSICAL units on the held-out window (K, g/kg) — deliberately NOT the
    #: same quantity as ``T_rmse``/``qv_rmse`` above, which are normalized by
    #: the reference's own spread.  The unit suffix is the whole point.
    heldout_T_rmse_K: float = float("nan")
    heldout_qv_rmse_g_kg: float = float("nan")


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


#: What the derivative-free tuner minimises.  ``combined`` is the historical
#: full-column score, which §8.2d measured to be 87-100 % condensate; the
#: ``subcloud`` objective exists because that score cannot see the sub-cloud
#: layer at all; ``thermo`` exists because it cannot see TEMPERATURE or
#: HUMIDITY either (T contributes 0.01-2.3 % of it and q_v 0.008-3.4 %).  A
#: typo must select nothing (dispatch-hardening), so the resolver raises rather
#: than defaulting.
TUNE_OBJECTIVES = ("combined", "subcloud", "thermo")

#: The scalar each objective minimises, as a field of ``RunDiagnostics``.  One
#: mapping, so the tuner, the CSV sort key and the figure label cannot pick
#: three different numbers — a real failure mode: the shipped writer sorted a
#: sub-cloud-tuned table by the combined score.
OBJECTIVE_FIELD = {
    "combined": "score",
    "subcloud": "subcloud_score",
    "thermo": "thermo_score",
}


def objective_value(run: "RunDiagnostics", objective: str) -> float:
    """The scalar the tuner minimises for ``run``, by name.

    A failed run scores ``+inf`` on every objective, so a scheme cannot win by
    crashing; a NaN is mapped to ``+inf`` for the same reason (a NaN incumbent
    makes every ``trial < best`` comparison False, which silently discards the
    entire search — the bug this repo already hit once).
    """
    if objective not in TUNE_OBJECTIVES:
        raise ValueError(
            f"objective_value: unknown objective {objective!r}; "
            f"expected one of {TUNE_OBJECTIVES}")
    value = float(getattr(run, OBJECTIVE_FIELD[objective]))
    return value if math.isfinite(value) else float("inf")


def _weighted_std(profile: np.ndarray, weights: np.ndarray) -> float:
    return float(weighted_std_jax(jnp.asarray(profile), jnp.asarray(weights)))


def _weighted_rmse(diff: np.ndarray, weights: np.ndarray) -> float:
    return float(weighted_rmse_jax(jnp.asarray(diff), jnp.asarray(weights)))


def _subcloud_diagnostics(
    ref: ReferenceProfiles,
    T_profile: np.ndarray,
    qv_profile: np.ndarray,
    *,
    subcloud_top_m: float,
    evap_mm_day: float,
) -> dict[str, float]:
    """Sub-cloud profile scores plus the lowest level's bulk-flux state.

    The bulk state is read at index ``-1`` because the SCM runs on the CRM
    reference's own grid, so that index is the lowest full level on BOTH sides
    and the numbers are directly comparable to the reference's.  All of it is
    computed from the shared helpers in ``scm_rce_metrics``; nothing here
    re-derives a saturation curve or a weighting.
    """
    weights = subcloud_mass_weights(
        jnp.asarray(ref.z_m), jnp.asarray(ref.mass_weights),
        top_m=subcloud_top_m,
    )
    T_sc, qv_sc, evap_sc, combined_sc = subcloud_objective_jax(
        ref,
        jnp.asarray(T_profile),
        jnp.asarray(qv_profile),
        jnp.asarray(evap_mm_day, dtype=jnp.float64),
        subcloud_weights=weights,
    )
    rh, driver, delta_T = subcloud_bulk_state(
        T_air_K=jnp.asarray(T_profile[-1], dtype=jnp.float64),
        r_air=jnp.asarray(qv_profile[-1], dtype=jnp.float64),
        p_air_Pa=jnp.asarray(ref.sigma_full[-1] * WING_P_SFC, dtype=jnp.float64),
        sst_K=jnp.asarray(FIXED_SST_K, dtype=jnp.float64),
        p_sfc_Pa=jnp.asarray(WING_P_SFC, dtype=jnp.float64),
    )
    return {
        "subcloud_T_rmse": float(T_sc),
        "subcloud_qv_rmse": float(qv_sc),
        "subcloud_evap_term": float(evap_sc),
        "subcloud_score": float(combined_sc),
        "subcloud_top_m": float(subcloud_top_m),
        "subcloud_n_levels": int(np.sum(np.asarray(ref.z_m) <= subcloud_top_m)),
        "sfc_relative_humidity": float(rh),
        "sfc_driver_kg_kg": float(driver),
        "sfc_delta_T_K": float(delta_T),
    }


def reference_pressure_profile(ref: ReferenceProfiles) -> np.ndarray:
    """Full-level pressure [Pa] of the shared column.

    ONE pressure profile is used for the SCM and for the CRM: the SCM runs on
    the CRM reference's own vertical grid at the RCEMIP surface pressure, so a
    relative humidity computed on two different pressure profiles would differ
    by a nominal number rather than by anything physical.  The same expression
    already backs the sub-cloud bulk state.
    """
    return np.asarray(ref.sigma_full, dtype=float) * WING_P_SFC


def _thermo_diagnostics(
    ref: ReferenceProfiles,
    T_profile: np.ndarray,
    qv_profile: np.ndarray,
    *,
    min_p_Pa: float = DEFAULT_THERMO_MIN_P_PA,
    T_scale_K: float = DEFAULT_THERMO_T_SCALE_K,
    rh_scale: float = DEFAULT_THERMO_RH_SCALE,
    blend_width_K: float = DEFAULT_THERMO_RH_BLEND_WIDTH_K,
) -> dict[str, float]:
    """Temperature + tropospheric-relative-humidity score and its companions.

    Everything is computed by the shared helpers in ``scm_rce_metrics``; the
    only arithmetic here is the physical-unit RMSE pair reported alongside the
    score, and that reuses the same mass-weighted RMSE the score does.
    """
    p_full = reference_pressure_profile(ref)
    weights = tropospheric_mass_weights(
        jnp.asarray(p_full), jnp.asarray(ref.mass_weights), min_p_Pa=min_p_Pa,
    )
    T_term, rh_term, combined = score_thermo_jax(
        ref,
        jnp.asarray(T_profile),
        jnp.asarray(qv_profile),
        jnp.asarray(p_full),
        trop_weights=weights,
        T_scale_K=T_scale_K,
        rh_scale=rh_scale,
        blend_width_K=blend_width_K,
    )
    rh = relative_humidity_profile(
        jnp.asarray(T_profile), jnp.asarray(qv_profile), jnp.asarray(p_full),
        blend_width_K=blend_width_K)
    rh_ref = relative_humidity_profile(
        jnp.asarray(ref.T_ref), jnp.asarray(ref.qv_ref), jnp.asarray(p_full),
        blend_width_K=blend_width_K)
    return {
        "thermo_T_term": float(T_term),
        "thermo_rh_term": float(rh_term),
        "thermo_score": float(combined),
        "trop_rh_rmse": _weighted_rmse(
            np.asarray(rh) - np.asarray(rh_ref), np.asarray(weights)),
        "trop_qv_rmse_g_kg": _weighted_rmse(
            (np.asarray(qv_profile) - np.asarray(ref.qv_ref)) * 1000.0,
            np.asarray(weights)),
        "trop_n_levels": int(np.sum(p_full >= min_p_Pa)),
    }


def _score_profiles(
    ref: ReferenceProfiles,
    T_profile: np.ndarray,
    qv_profile: np.ndarray,
    qcond_profile: np.ndarray,
    precip_mm_day: float,
) -> tuple[float, float, float, float, float]:
    T_rmse, qv_rmse, cloud_rmse, precip_rmse, combined = score_profiles_precip_jax(
        ref,
        jnp.asarray(T_profile),
        jnp.asarray(qv_profile),
        jnp.asarray(qcond_profile),
        jnp.asarray(precip_mm_day, dtype=jnp.float64),
        profile_floor=PROFILE_FLOOR,
        precip_weight=PRECIP_SCORE_WEIGHT,
        precip_normalization_mm_day=PRECIP_NORMALIZATION_MM_DAY,
    )
    return (
        float(T_rmse),
        float(qv_rmse),
        float(cloud_rmse),
        float(precip_rmse),
        float(combined),
    )


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


def _reference_surface_precip(
    reference_dir: Path,
    analysis_days: float,
) -> tuple[float, list[str], str]:
    files = sorted((reference_dir / "snapshots").glob("sfc_*.npz"))
    if not files:
        return float("nan"), [], "no surface snapshots found"

    days: list[float] = []
    for path in files:
        with np.load(path) as ds:
            days.append(float(np.asarray(ds["day"])) if "day" in ds.files else float("nan"))
    finite_days = [d for d in days if math.isfinite(d)]
    if finite_days:
        end_day = max(finite_days)
        start_day = end_day - max(float(analysis_days), 0.0)
        selected = [
            path for path, day in zip(files, days)
            if math.isfinite(day) and day > start_day
        ]
    else:
        selected = files[-max(1, min(len(files), DEFAULT_LAST_REFERENCE_FILES)):]
    if not selected:
        selected = [files[-1]]

    means = []
    for path in selected:
        with np.load(path) as ds:
            if "precip" not in ds.files:
                continue
            means.append(float(np.nanmean(np.asarray(ds["precip"], dtype=float))))
    if not means:
        return float("nan"), [str(p) for p in selected], "surface precip field absent"
    raw_mean = float(np.mean(means))
    if abs(raw_mean) < CRM_PRECIP_RAW_TO_MM_DAY_THRESHOLD:
        precip = raw_mean * SECONDS_PER_DAY
        note = "CRM precip interpreted as kg m^-2 s^-1 and converted to mm/day"
    else:
        precip = raw_mean
        note = "CRM precip interpreted as already in mm/day"
    return precip, [str(p) for p in selected], note


def _reference_crm_clear_sky_subsidence(
    files: list[Path],
    *,
    condensate_threshold: float = CRM_CLEAR_SKY_COND_THRESHOLD,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Diagnose CRM-resolved environmental descent from reference volumes.

    RCEMIP/RCE does not prescribe large-scale subsidence; the plane CRM
    resolves convective updrafts and their compensating clear-sky descent.
    For an SCM comparability experiment we can approximate that missing
    resolved circulation by the *area-weighted* clear-sky vertical velocity,
    ``<w 1_clear>``, using the same RCEMIP cloud mask threshold as the cloud
    fraction diagnostic.  Positive-upward values are clipped to zero because
    this optional forcing represents subsidence only, not gravity-wave ascent.
    The model top and surface full levels are closed explicitly.

    This is intentionally not the conditional mean ``<w | clear>``; applying
    that to a single mean column would overstate the per-domain descent by
    the clear-sky area fraction.
    """
    w_profiles: list[np.ndarray] = []
    cf_profiles: list[np.ndarray] = []
    for path in files:
        with np.load(path) as ds:
            if "w" not in ds.files or "cond" not in ds.files:
                continue
            w = np.asarray(ds["w"], dtype=float)
            cond = np.maximum(np.asarray(ds["cond"], dtype=float), 0.0)
        clear = cond < condensate_threshold
        w_profiles.append(np.where(clear, w, 0.0).mean(axis=(0, 1)))
        cf_profiles.append((~clear).mean(axis=(0, 1)))
    if not w_profiles:
        return None, None
    w_sub = np.minimum(np.mean(np.stack(w_profiles), axis=0), 0.0)
    if w_sub.size:
        w_sub = w_sub.copy()
        w_sub[0] = 0.0
        w_sub[-1] = 0.0
    cloud_fraction = np.mean(np.stack(cf_profiles), axis=0)
    return w_sub, cloud_fraction


def reference_evap_from_hfls(reference_dir: Path) -> tuple[float, str] | None:
    """Reference evaporation [mm/day] from the archive's own ``hfls_avg``.

    ``None`` when the file is absent — the caller then falls back to the
    equilibrium identity E = P and SAYS SO.  A missing measurement must never
    be silently replaced by a derived one.

    Averaged over the last quarter of the record, matching the campaign's
    "settled window" convention rather than averaging spin-up in.
    """
    path = (Path(reference_dir) / "aux_0D"
            / "SAM_CRM_RCE_small300_0D_hfls_avg.nc")
    if not path.exists():
        return None
    try:
        import xarray as xr
    except ImportError:  # pragma: no cover - environment-dependent
        return None
    with xr.open_dataset(path) as ds:
        name = "hfls_avg" if "hfls_avg" in ds else next(iter(ds.data_vars))
        series = np.asarray(ds[name].values, dtype=float).reshape(-1)
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return None
    tail = finite[-max(1, finite.size // 4):]
    hfls = float(np.mean(tail))
    return (hfls / constants.L_v * SECONDS_PER_DAY,
            f"measured hfls_avg={hfls:.2f} W/m^2 "
            f"(last {tail.size}/{finite.size} samples)")


def build_reference_profiles(
    reference_dir: Path,
    last_n: int,
    precip_analysis_days: float = DEFAULT_ANALYSIS_DAYS,
) -> ReferenceProfiles:
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
    precip_ref, precip_files, precip_note = _reference_surface_precip(
        reference_dir, precip_analysis_days,
    )
    crm_subsidence, crm_cloud_fraction = _reference_crm_clear_sky_subsidence(
        files,
    )
    measured_evap = reference_evap_from_hfls(reference_dir)
    if measured_evap is None:
        evap_ref, evap_note = float(precip_ref), (
            "EQUILIBRIUM IDENTITY E=P (no measured hfls_avg on disk)")
    else:
        evap_ref, evap_note = measured_evap
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
        precip_ref_mm_day=precip_ref,
        evap_ref_mm_day=evap_ref,
        evap_ref_note=evap_note,
        precip_files_used=precip_files,
        precip_unit_note=precip_note,
        crm_clear_sky_subsidence_m_s=crm_subsidence,
        crm_cloud_fraction=crm_cloud_fraction,
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
    radiation: str = BASELINE_SCHEMES["radiation"],
    radiation_update_interval_steps: int = 1,
    turbulence: str = BASELINE_SCHEMES["turbulence"],
    microphysics: str = BASELINE_SCHEMES["microphysics"],
    gravity_wave_drag: str = BASELINE_SCHEMES["gravity_wave_drag"],
    convection: str = BASELINE_SCHEMES["convection"],
    base: PhysicsConfig | None = None,
    prognostic_spectral_gwd_thermal_tendency: bool = (
        SCM_PROGNOSTIC_SPECTRAL_GWD_THERMAL_TENDENCY
    ),
    hard_saturation_adjustment: bool = False,
    hard_sat_adjust_threshold: float | None = None,
    hard_sat_max_heating_K: float | None = None,
) -> PhysicsConfig:
    cfg = base if base is not None else PhysicsConfig()
    radiation_update_interval_steps = max(1, int(radiation_update_interval_steps))
    if radiation == "gray":
        radiation_cfg = RadiationConfig(
            scheme="gray",
            update_interval_steps=radiation_update_interval_steps,
            gray=GrayRadiationConfig(
                **{
                    **cfg.radiation.gray._asdict(),
                    "S_0": RCEMIP_S0_W_M2,
                    "perpetual_equinox": True,
                }
            ),
            diurnal_cycle=False,
            rce_fixed_cos_zenith=RCEMIP_COS_ZENITH,
        )
    elif radiation == "rrtmgp":
        radiation_cfg = RadiationConfig(
            scheme="rrtmgp",
            update_interval_steps=radiation_update_interval_steps,
            diurnal_cycle=False,
            rrtmgp=RRTMGPConfig(
                S_0=RCEMIP_S0_W_M2,
                co2_ppmv=355.0,
                ch4_ppbv=1700.0,
                n2o_ppbv=320.0,
                sfc_albedo_direct=float(
                    sam_ocean_albedo(RCEMIP_COS_ZENITH, FIXED_SST_K)
                ),
                sfc_albedo=0.07,
                include_clouds=True,
            ),
            ozone=OzoneProfileConfig(source="mls"),
            cloud_scheme="resolved",
            cloud_config=CloudConfig(scheme="resolved", r_eff_liq=14.0e-6),
            rce_fixed_cos_zenith=RCEMIP_COS_ZENITH,
        )
    else:
        raise ValueError(
            f"Unknown campaign radiation scheme: {radiation!r}; "
            "choose from 'gray' or 'rrtmgp'."
        )
    # IN-SCHEME liquid super-saturation guard (the IFS/SAM "no liquid
    # super-saturation" half; the ice half is the Koop/Kärcher homogeneous-
    # freezing allowance, which lives inside morrison/thompson/p3 and needs no
    # switch here).  Threaded through the SHARED
    # ``apply_microphysics_experiment_flags`` so a scheme that cannot carry the
    # flag raises instead of silently ignoring it — never a private copy of
    # that dispatch.
    micro_cfg = cfg.microphysics._replace(scheme=microphysics)
    if (hard_saturation_adjustment
            or hard_sat_adjust_threshold is not None
            or hard_sat_max_heating_K is not None):
        if not hard_saturation_adjustment:
            # A float override without the boolean gate is SILENTLY INERT (the
            # schemes branch on a static ``if config.hard_saturation_
            # adjustment``), so refuse it rather than let a caller believe the
            # threshold took effect.  Mirrors ExperimentConfig.validate_strict.
            raise ValueError(
                "hard_sat_adjust_threshold / hard_sat_max_heating_K require "
                "hard_saturation_adjustment=True (the override would be "
                "silently inert without it)."
            )
        from legoesm.atmosphere.physics.microphysics.config import (
            apply_microphysics_experiment_flags,
        )
        micro_cfg = micro_cfg._replace(**{microphysics: (
            apply_microphysics_experiment_flags(
                getattr(micro_cfg, microphysics), microphysics,
                hard_saturation_adjustment=hard_saturation_adjustment,
                hard_sat_adjust_threshold=hard_sat_adjust_threshold,
                hard_sat_max_heating_K=hard_sat_max_heating_K,
            ))})
    gwd_cfg = cfg.gravity_wave_drag._replace(scheme=gravity_wave_drag)
    if gravity_wave_drag == "prognostic_spectral":
        gwd_cfg = gwd_cfg._replace(
            prognostic_spectral=gwd_cfg.prognostic_spectral._replace(
                thermal_tendency=prognostic_spectral_gwd_thermal_tendency,
            )
        )
    return cfg._replace(
        radiation=radiation_cfg,
        convection=cfg.convection._replace(scheme=convection),
        turbulence=cfg.turbulence._replace(scheme=turbulence),
        microphysics=micro_cfg,
        gravity_wave_drag=gwd_cfg,
    )


def _config_scheme_dict(cfg: PhysicsConfig) -> dict[str, str]:
    return {
        "radiation": cfg.radiation.scheme,
        "convection": cfg.convection.scheme,
        "turbulence": cfg.turbulence.scheme,
        "microphysics": cfg.microphysics.scheme,
        "gravity_wave_drag": cfg.gravity_wave_drag.scheme,
    }


def _effective_scm_microphysics_substeps(
    microphysics_scheme: str,
    requested_substeps: int,
) -> int:
    if microphysics_scheme in SCM_MICROPHYSICS_SUBSTEP_SCHEMES:
        return requested_substeps
    return 1


def _effective_scm_convection_substeps(
    convection_scheme: str,
    requested_substeps: int,
) -> int:
    """Convective sub-steps per outer step — IDENTICAL for every scheme.

    Returns ``requested_substeps`` unconditionally. ``convection_scheme`` is
    kept in the signature so every call site still reads as scheme-aware and
    so a future scheme-specific need has an obvious home, but it MUST NOT be
    used to vary the count: that is the uncontrolled-comparison bug this
    replaced (6 of 10 schemes at 60 s, 4 at 600 s).

    ``scheme == "none"`` is unaffected — callers already gate on
    ``cfg.convection.scheme != "none"`` before sub-stepping.
    """
    return requested_substeps


def _without_convection_config(cfg: PhysicsConfig) -> PhysicsConfig:
    return cfg._replace(convection=ConvectionConfig(scheme="none"))


def _convection_only_config(cfg: PhysicsConfig) -> PhysicsConfig:
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=cfg.convection,
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def _config_cache_key(
    cfg: PhysicsConfig,
    days: float,
    dt: float,
    scm_microphysics_substeps: int,
    scm_convection_substeps: int,
    surface_wind_m_s: float,
    coriolis_s_inv: float,
    large_scale_forcing: str,
    bl_anchor_top_m: float = DEFAULT_SCM_RCE_BL_TOP_M,
    subcloud_top_m: float = DEFAULT_SUBCLOUD_TOP_M,
) -> str:
    effective_microphysics_substeps = _effective_scm_microphysics_substeps(
        cfg.microphysics.scheme,
        scm_microphysics_substeps,
    )
    effective_convection_substeps = _effective_scm_convection_substeps(
        cfg.convection.scheme,
        scm_convection_substeps,
    )
    payload = {
        "days": days,
        "dt": dt,
        "scm_microphysics_substeps": effective_microphysics_substeps,
        "scm_convection_substeps": effective_convection_substeps,
        "surface_wind_m_s": surface_wind_m_s,
        "coriolis_s_inv": coriolis_s_inv,
        "large_scale_forcing": large_scale_forcing,
        "bl_anchor_top_m": bl_anchor_top_m,
        "subcloud_top_m": subcloud_top_m,
        "config": _to_jsonable(cfg),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _make_scm_rce_forcing(
    ref: ReferenceProfiles,
    *,
    surface_wind_m_s: float,
    coriolis_s_inv: float,
    large_scale_forcing: str,
) -> SCMForcing:
    if large_scale_forcing not in SCM_RCE_LARGE_SCALE_FORCING_CHOICES:
        raise ValueError(
            f"Unknown SCM RCE large-scale forcing {large_scale_forcing!r}; "
            f"choose from {SCM_RCE_LARGE_SCALE_FORCING_CHOICES}."
        )
    wind_profile = jnp.full((len(ref.z_m),), surface_wind_m_s, dtype=jnp.float64)
    zero_wind_profile = jnp.zeros((len(ref.z_m),), dtype=jnp.float64)
    subsidence_w = None
    if large_scale_forcing == "crm_clear_sky_subsidence":
        if ref.crm_clear_sky_subsidence_m_s is None:
            raise ValueError(
                "large_scale_forcing='crm_clear_sky_subsidence' requires "
                "reference volumes containing `w` and `cond`."
            )
        w_sub = jnp.asarray(ref.crm_clear_sky_subsidence_m_s, dtype=jnp.float64)
        if w_sub.shape != (len(ref.z_m),):
            raise ValueError(
                "CRM clear-sky subsidence profile shape "
                f"{tuple(w_sub.shape)} does not match nlev={len(ref.z_m)}."
            )
        subsidence_w = lambda _t, w=w_sub: w
    return SCMForcing(
        prescribe="T_s",
        T_s=lambda _t: FIXED_SST_K,
        f_c=coriolis_s_inv,
        u_geo=lambda _t: wind_profile,
        v_geo=lambda _t: zero_wind_profile,
        subsidence_w=subsidence_w,
    )


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


#: The condensate species, in the ONE order both the total and the split sum
#: them in.  Keeping a single ordered list is what lets the split be added
#: without perturbing the total by a rounding ulp, so a re-run of an existing
#: arm still reproduces its scores bit-for-bit.
_CONDENSATE_TRACERS = ("q_c", "q_r", "q_i", "q_s", "q_g")
#: The non-precipitating half.  Mirrors the CRM reference's own split, whose
#: ``cond`` is ``clw_avg + cli_avg + plw_avg + pli_avg`` and whose ``qcloud``
#: is ``clw_avg + cli_avg``; the archive's precipitating ice covers snow AND
#: graupel, so both land in the precipitating half here.
_CLOUD_TRACERS = frozenset({"q_c", "q_i"})


def _condensate_species(microphysics_scheme: str) -> tuple[str, ...]:
    """Condensate tracer names carried by ``microphysics_scheme``, in order.

    p3 carries no separate graupel category — its rimed ice lives in the ice
    variable — so ``q_g`` is excluded there.  This is the reference exclusion
    the summed form has always applied, stated once instead of twice.
    """
    if microphysics_scheme == "p3":
        return tuple(n for n in _CONDENSATE_TRACERS if n != "q_g")
    return _CONDENSATE_TRACERS


def _qcond_from_tracers(tracers: dict[str, Field], microphysics_scheme: str) -> jax.Array:
    """Total condensate [kg/kg]."""
    qcond = jnp.zeros_like(tracers["q_v"].data)
    for name in _condensate_species(microphysics_scheme):
        if name in tracers:
            qcond = qcond + tracers[name].data
    return qcond


def _condensate_parts_from_tracers(
    tracers: dict[str, Field], microphysics_scheme: str,
) -> tuple[jax.Array, jax.Array]:
    """Split the column condensate into ``(cloud, precipitating)`` [kg/kg].

    Scored against the CRM the total alone cannot say whether an excess is
    cloud the convection scheme detrains or rain the microphysics fails to
    sediment — opposite defects with opposite fixes.  The reference stores both
    halves, so the SCM must too.

    The sum of the two parts equals ``_qcond_from_tracers`` up to floating-point
    association only; the TOTAL used for scoring is still taken from that
    function, so this diagnostic cannot move a score.
    """
    zero = jnp.zeros_like(tracers["q_v"].data)
    cloud = zero
    precip = zero
    for name in _condensate_species(microphysics_scheme):
        if name not in tracers:
            continue
        if name in _CLOUD_TRACERS:
            cloud = cloud + tracers[name].data
        else:
            precip = precip + tracers[name].data
    return cloud, precip


def _column_tracer(
    state,
    name: str,
    ncol: int,
    nlev: int,
    dtype,
) -> jax.Array:
    if state.tracers is not None and name in state.tracers:
        raw = state.tracers[name]
        data = raw.data if hasattr(raw, "data") else raw
        return jnp.maximum(data.reshape(ncol, nlev), 0.0)
    return jnp.zeros((ncol, nlev), dtype=dtype)


def _make_microphysics_precip_diagnostic(cfg: PhysicsConfig, dt: float):
    scheme_name, micro_fn, scheme_config = get_microphysics_fn(cfg.microphysics)
    if micro_fn is None:
        def zero_precip(state, grid, sigma_coord):
            del grid, sigma_coord
            return jnp.zeros((), dtype=state.T.data.dtype)

        return zero_precip

    def precip_mm_day(state, grid, sigma_coord):
        del grid
        T = state.T.data
        p_s = state.p_s.data
        nlev = sigma_coord.n_levels
        shape_2d = p_s.shape
        ncol = 1
        for size in shape_2d:
            ncol *= int(size)
        dtype = T.dtype
        T_col = T.reshape(ncol, nlev)
        p_full_col = sigma_coord.pressure_at_full(p_s).reshape(ncol, nlev)
        p_half_col = sigma_coord.pressure_at_half(p_s).reshape(ncol, nlev + 1)
        q_v = _column_tracer(state, "q_v", ncol, nlev, dtype)
        rho = _compute_rho(T_col, p_full_col, q_v)
        dz = _compute_layer_dz(T_col, p_half_col, q_v)
        hydrometeors = HydrometeorState(
            q_c=_column_tracer(state, "q_c", ncol, nlev, dtype),
            q_r=_column_tracer(state, "q_r", ncol, nlev, dtype),
            q_i=_column_tracer(state, "q_i", ncol, nlev, dtype),
            q_s=_column_tracer(state, "q_s", ncol, nlev, dtype),
            q_g=_column_tracer(state, "q_g", ncol, nlev, dtype),
            N_c=_column_tracer(state, "N_c", ncol, nlev, dtype),
            N_r=_column_tracer(state, "N_r", ncol, nlev, dtype),
            N_i=_column_tracer(state, "N_i", ncol, nlev, dtype),
        )
        micro_out = micro_fn(
            T_col,
            q_v,
            hydrometeors,
            p_full_col,
            p_half_col,
            rho,
            dz,
            dt,
            scheme_config,
        )
        return jnp.maximum(micro_out.precipitation[0], 0.0) * SECONDS_PER_DAY

    precip_mm_day.scheme_name = scheme_name
    return precip_mm_day


def _make_convective_precip_diagnostic(cfg: PhysicsConfig, dt: float):
    if cfg.convection.scheme == "none":
        def zero_precip(state, phys_state, grid, sigma_coord):
            del phys_state, grid, sigma_coord
            return jnp.zeros((), dtype=state.T.data.dtype)

        return zero_precip
    conv_physics_fn = make_physics(
        PhysicsConfig(
            radiation=RadiationConfig(scheme="none"),
            convection=cfg.convection,
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(scheme="none"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        ),
        model_type="hydrostatic",
        dt=dt,
    )

    def precip_mm_day(state, phys_state, grid, sigma_coord):
        tend, _phys_out = conv_physics_fn(
            state, grid, sigma_coord, phys_state=phys_state,
        )
        return _convective_precip_rate_from_tendency(state, tend, sigma_coord)

    return precip_mm_day


def _convective_precip_rate_from_tendency(
    state,
    tend: HydrostaticTendencies,
    sigma_coord: SigmaCoordinate,
) -> jax.Array:
    if tend.tracer_tendencies is None or "q_c" not in tend.tracer_tendencies:
        return jnp.zeros((), dtype=state.T.data.dtype)
    dq_c_dt = jnp.maximum(tend.tracer_tendencies["q_c"].data[0, 0, 0], 0.0)
    column_mass = sigma_coord.dsigma * state.p_s.data[0, 0, 0] / constants.g
    return jnp.sum(dq_c_dt * column_mass) * SECONDS_PER_DAY


def _zero_tendency_like_state(state) -> HydrostaticTendencies:
    dv_dt = None
    if state.v is not None:
        dv_dt = Field(
            data=jnp.zeros_like(state.v.data),
            name="dv_dt_zero",
            dims=state.v.dims,
            units="m/s^2",
        )
    return HydrostaticTendencies(
        du_dt=Field(
            data=jnp.zeros_like(state.u.data),
            name="du_dt_zero",
            dims=state.u.dims,
            units="m/s^2",
        ),
        dT_dt=Field(
            data=jnp.zeros_like(state.T.data),
            name="dT_dt_zero",
            dims=state.T.dims,
            units="K/s",
        ),
        dp_s_dt=Field(
            data=jnp.zeros_like(state.p_s.data),
            name="dp_s_dt_zero",
            dims=state.p_s.dims,
            units="Pa/s",
        ),
        dphis_dt=Field(
            data=jnp.zeros_like(state.p_s.data),
            name="dphis_dt_zero",
            dims=state.p_s.dims,
            units="m^2/s^3",
        ),
        dv_dt=dv_dt,
        tracer_tendencies=None,
    )


def _zero_tendency_like(tend: HydrostaticTendencies) -> HydrostaticTendencies:
    tracer_tendencies = None
    if tend.tracer_tendencies is not None:
        tracer_tendencies = {
            name: field.replace(data=jnp.zeros_like(field.data))
            for name, field in tend.tracer_tendencies.items()
        }
    return tend._replace(
        du_dt=tend.du_dt.replace(data=jnp.zeros_like(tend.du_dt.data)),
        dT_dt=tend.dT_dt.replace(data=jnp.zeros_like(tend.dT_dt.data)),
        dp_s_dt=tend.dp_s_dt.replace(data=jnp.zeros_like(tend.dp_s_dt.data)),
        dphis_dt=tend.dphis_dt.replace(data=jnp.zeros_like(tend.dphis_dt.data)),
        dv_dt=(
            None if tend.dv_dt is None
            else tend.dv_dt.replace(data=jnp.zeros_like(tend.dv_dt.data))
        ),
        tracer_tendencies=tracer_tendencies,
    )


def _radiation_only_config(cfg: PhysicsConfig) -> PhysicsConfig:
    return PhysicsConfig(
        radiation=cfg.radiation,
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def _without_radiation_config(cfg: PhysicsConfig) -> PhysicsConfig:
    return cfg._replace(radiation=cfg.radiation._replace(scheme="none"))


def _realism_diagnostics(ref: ReferenceProfiles, T_profile, qv_profile) -> dict[str, float]:
    p_full = jnp.asarray(ref.sigma_full * WING_P_SFC, dtype=jnp.float64)
    diag_jax = moist_adiabat_diagnostics_jax(
        T_profile=jnp.asarray(T_profile, dtype=jnp.float64),
        qv_profile=jnp.asarray(qv_profile, dtype=jnp.float64),
        p_full=p_full,
        sigma_full=jnp.asarray(ref.sigma_full, dtype=jnp.float64),
        z_m=jnp.asarray(ref.z_m, dtype=jnp.float64),
    )
    return {
        key: float(np.asarray(value)) if np.asarray(value).ndim == 0 else value
        for key, value in diag_jax.items()
        if key != "moist_adiabat"
    }


def run_scm_rce(
    cfg: PhysicsConfig,
    ref: ReferenceProfiles,
    *,
    label: str,
    days: float,
    dt: float,
    analysis_days: float,
    require_equilibrium: bool,
    require_realism: bool,
    equil_T_tol_K: float,
    equil_qv_tol: float,
    equil_qcond_tol: float,
    scm_microphysics_substeps: int = DEFAULT_SCM_MICROPHYSICS_SUBSTEPS,
    scm_convection_substeps: int = DEFAULT_SCM_CONVECTION_SUBSTEPS,
    surface_wind_m_s: float = DEFAULT_SCM_RCE_SURFACE_WIND_M_S,
    coriolis_s_inv: float = DEFAULT_SCM_RCE_CORIOLIS_S_INV,
    large_scale_forcing: str = DEFAULT_SCM_RCE_LARGE_SCALE_FORCING,
    bl_anchor_top_m: float = DEFAULT_SCM_RCE_BL_TOP_M,
    subcloud_top_m: float = DEFAULT_SUBCLOUD_TOP_M,
) -> RunDiagnostics:
    nsteps = max(1, int(round(days * SECONDS_PER_DAY / dt)))
    T0, qv0 = wing_initial_profiles(ref)
    forcing = _make_scm_rce_forcing(
        ref,
        surface_wind_m_s=surface_wind_m_s,
        coriolis_s_inv=coriolis_s_inv,
        large_scale_forcing=large_scale_forcing,
    )
    radiation_interval = max(1, int(cfg.radiation.update_interval_steps))
    use_cached_radiation = (
        cfg.radiation.scheme != "none" and radiation_interval > 1
    )
    scm_cfg = _without_radiation_config(cfg) if use_cached_radiation else cfg
    effective_microphysics_substeps = _effective_scm_microphysics_substeps(
        cfg.microphysics.scheme,
        scm_microphysics_substeps,
    )
    effective_convection_substeps = _effective_scm_convection_substeps(
        cfg.convection.scheme,
        scm_convection_substeps,
    )
    use_split_convection = (
        cfg.convection.scheme != "none" and effective_convection_substeps > 1
    )
    scm = SingleColumnModel.create(
        physics_config=(
            _without_convection_config(scm_cfg)
            if use_split_convection
            else scm_cfg
        ),
        nlev=len(ref.z_m),
        dt=dt,
        T_profile=T0,
        q_v_profile=qv0,
        u=surface_wind_m_s,
        v=0.0,
        p_s=WING_P_SFC,
        latitude_deg=0.0,
        time_integrator="forward_euler",
        forcing=forcing,
        dtype=jnp.float64,
        microphysics_substeps=effective_microphysics_substeps,
    )
    scm.sigma_coord = make_sigma_coordinate_from_reference(ref)
    _preseed_column_tracers(scm, cfg.microphysics.scheme)
    rad_physics_fn = None
    if use_cached_radiation:
        rad_physics_fn = make_physics(
            _radiation_only_config(cfg),
            model_type="hydrostatic",
            dt=dt,
        )
    conv_physics_fn = None
    if use_split_convection:
        conv_physics_fn = make_physics(
            _convection_only_config(cfg),
            model_type="hydrostatic",
            dt=dt / effective_convection_substeps,
        )

    sst_col = jnp.asarray([FIXED_SST_K], dtype=jnp.float64)
    forcing_dict = {"T_sfc": sst_col}
    z_profile = jnp.asarray(ref.z_m, dtype=jnp.float64)
    z_above_lowest = jnp.maximum(z_profile - z_profile[-1], 0.0)
    # A NEGATIVE depth disables the anchor outright: no level satisfies
    # `z >= 0 <= negative`.  `<=` is kept (not changed to `<`) so a zero depth
    # keeps its historical meaning and the published arms stay reproducible.
    bl_mask = z_above_lowest <= bl_anchor_top_m
    physics_fn = scm.physics_fn
    grid = scm.grid
    sigma_coord = scm.sigma_coord
    dt_arr = jnp.asarray(dt, dtype=jnp.float64)
    microphysics_scheme = cfg.microphysics.scheme
    # _make_microphysics_precip_diagnostic is NOT built here any more: it was
    # the wrong-dt second evaluation whose value the score used to report, and
    # leaving a loaded closure around for someone to reach for again is how the
    # defect would come back (codex review, 2026-08-11).
    convective_precip_diagnostic = _make_convective_precip_diagnostic(cfg, dt)

    def apply_surface_sst_anchor(state):
        T_bl = jnp.asarray(FIXED_SST_K, dtype=state.T.data.dtype) - (
            DEFAULT_SCM_RCE_BL_LAPSE_K_M
            * z_above_lowest.astype(state.T.data.dtype)
        )
        T_bl = jnp.maximum(
            T_bl, jnp.asarray(DEFAULT_SCM_RCE_BL_MIN_T_K, dtype=state.T.data.dtype),
        )
        mask = bl_mask.reshape(1, 1, 1, -1)
        T_data = jnp.where(mask, T_bl.reshape(1, 1, 1, -1), state.T.data)
        return state._replace(T=state.T.replace(data=T_data))

    def _call_physics_with_fixed_sst(fn, state, phys_state):
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

    def fixed_sst_tendency(state, phys_state, t):
        tend, phys_out = _call_physics_with_fixed_sst(
            physics_fn, state, phys_state,
        )
        forcing_tend = compute_forcing_tendencies(
            state, sigma_coord, forcing, t,
        )
        summed = add_tendencies(tend, forcing_tend)
        # RESTORE the surface-flux diagnostics the sum drops. add_tendencies
        # rebuilds HydrostaticTendencies from six fields, so lhflx_sfc/
        # shflx_sfc (and precip) are None on its result -- which is exactly how
        # the evaporation readout first measured E = 0.0000 on a column whose
        # own bulk formula gives 1.559 mm/day. The forcing tendency carries no
        # surface fluxes, so taking the physics values is the whole answer.
        # precip is deliberately NOT restored: it is read from the microphysics
        # tendency at the point of application, and putting it here as well
        # would invite exactly the double count that was just removed.
        return summed._replace(
            lhflx_sfc=tend.lhflx_sfc, shflx_sfc=tend.shflx_sfc), phys_out

    def apply_convection_substeps(state, phys_state):
        if not use_split_convection:
            return state, phys_state, jnp.zeros((), dtype=state.T.data.dtype)
        sub_dt = dt / effective_convection_substeps
        sub_weight = jnp.asarray(1.0 / effective_convection_substeps, dtype=state.T.data.dtype)

        def substep(carry, _i):
            sub_state, sub_phys, precip_accum = carry
            conv_tend, conv_phys = conv_physics_fn(
                sub_state,
                grid,
                sigma_coord,
                phys_state=sub_phys,
            )
            if conv_phys is None:
                conv_phys = sub_phys
            precip_rate = _convective_precip_rate_from_tendency(
                sub_state, conv_tend, sigma_coord,
            )
            new_sub_state = apply_tendencies(sub_state, conv_tend, sub_dt)
            return (
                new_sub_state,
                conv_phys,
                precip_accum + sub_weight * precip_rate,
            ), None

        (new_state, new_phys, precip_rate), _ = lax.scan(
            substep,
            (state, phys_state, jnp.zeros((), dtype=state.T.data.dtype)),
            jnp.arange(effective_convection_substeps, dtype=jnp.int32),
        )
        return new_state, new_phys, precip_rate

    def apply_split_microphysics_step(state, tend):
        if effective_microphysics_substeps <= 1:
            return (
                apply_tendencies(state, tend, dt),
                applied_precip_mm_day(tend, state),
            )
        sub_dt = dt / effective_microphysics_substeps
        microphysics_fn = scm._microphysics_fn
        sub_weight = jnp.asarray(
            1.0 / effective_microphysics_substeps,
            dtype=state.T.data.dtype,
        )

        def substep(sub_state, _i):
            micro_tend = microphysics_fn(sub_state, grid, sigma_coord)
            new_sub_state = apply_tendencies(
                sub_state, add_tendencies(tend, micro_tend), sub_dt)
            # Read precip from MICRO_TEND, not from the sum: add_tendencies
            # rebuilds a HydrostaticTendencies from six fields only
            # (du/dT/dp_s/dphis/dv/tracers) and DROPS every diagnostic field,
            # precip included. Taking it from the sum would have silently
            # reported None -> 0.0, i.e. reproduced the bug this fixes.
            precip_rate = applied_precip_mm_day(micro_tend, sub_state)
            return new_sub_state, sub_weight * precip_rate

        new_state, precip_rates = lax.scan(
            substep,
            state,
            jnp.arange(effective_microphysics_substeps, dtype=jnp.int32),
        )
        return new_state, jnp.sum(precip_rates)

    def body(carry, k):
        state, phys_state = carry
        t = k.astype(jnp.float64) * dt_arr
        base_tend, new_phys = fixed_sst_tendency(state, phys_state, t)
        new_state, micro_precip = apply_split_microphysics_step(
            state, base_tend,
        )
        new_state, new_phys, convective_precip = apply_convection_substeps(
            new_state, new_phys,
        )
        new_state = apply_surface_sst_anchor(new_state)
        qv = new_state.tracers["q_v"].data[0, 0, 0]
        qcond = _qcond_from_tracers(
            new_state.tracers, microphysics_scheme,
        )[0, 0, 0]
        # Split diagnostic, recorded alongside the total.  The SCORE keeps
        # using the total computed above, so this cannot perturb it.
        qcloud_col, qprecip_col = _condensate_parts_from_tracers(
            new_state.tracers, microphysics_scheme,
        )
        qcloud = qcloud_col[0, 0, 0]
        qprecip = qprecip_col[0, 0, 0]
        # WHETHER convection is sub-stepped is irrelevant to whether its
        # condensate is double-counted; what matters is whether MICROPHYSICS
        # will sediment it.  The rule used to branch on sub-stepping, so the
        # non-sub-stepped path added a convective diagnostic on top of
        # condensate that microphysics would later rain out and count again
        # (codex review, 2026-08-11).  The default ten substeps hid it.
        if microphysics_scheme == "none":
            convective_precip_for_score = (
                convective_precip if use_split_convection
                else convective_precip_diagnostic(
                    new_state, new_phys, grid, sigma_coord)
            )
        else:
            convective_precip_for_score = jnp.zeros(
                (), dtype=new_state.T.data.dtype)
        precip = micro_precip + convective_precip_for_score
        # E recorded ALONGSIDE P, from the same applied tendency and with the
        # same per-step weight, so the water budget can be closed with both
        # terms sampled identically (see applied_evap_mm_day).
        evap = applied_evap_mm_day(base_tend, new_state)
        out = (new_state.T.data[0, 0, 0], qv, qcond, qcloud, qprecip,
               precip, evap)
        return (new_state, new_phys), out

    def cached_radiation_body(carry, k):
        state, phys_state, rad_cache = carry
        t = k.astype(jnp.float64) * dt_arr
        nonrad_tend, new_phys = fixed_sst_tendency(state, phys_state, t)

        def refresh(_unused):
            rad_tend, _rad_phys = _call_physics_with_fixed_sst(
                rad_physics_fn, state, phys_state,
            )
            return rad_tend

        rad_tend = lax.cond(
            jnp.mod(k, radiation_interval) == 0,
            refresh,
            lambda _unused: rad_cache,
            operand=None,
        )
        new_state, micro_precip = apply_split_microphysics_step(
            state, add_tendencies(nonrad_tend, rad_tend),
        )
        new_state, new_phys, convective_precip = apply_convection_substeps(
            new_state, new_phys,
        )
        new_state = apply_surface_sst_anchor(new_state)
        qv = new_state.tracers["q_v"].data[0, 0, 0]
        qcond = _qcond_from_tracers(
            new_state.tracers, microphysics_scheme,
        )[0, 0, 0]
        # Split diagnostic, recorded alongside the total.  The SCORE keeps
        # using the total computed above, so this cannot perturb it.
        qcloud_col, qprecip_col = _condensate_parts_from_tracers(
            new_state.tracers, microphysics_scheme,
        )
        qcloud = qcloud_col[0, 0, 0]
        qprecip = qprecip_col[0, 0, 0]
        # WHETHER convection is sub-stepped is irrelevant to whether its
        # condensate is double-counted; what matters is whether MICROPHYSICS
        # will sediment it.  The rule used to branch on sub-stepping, so the
        # non-sub-stepped path added a convective diagnostic on top of
        # condensate that microphysics would later rain out and count again
        # (codex review, 2026-08-11).  The default ten substeps hid it.
        if microphysics_scheme == "none":
            convective_precip_for_score = (
                convective_precip if use_split_convection
                else convective_precip_diagnostic(
                    new_state, new_phys, grid, sigma_coord)
            )
        else:
            convective_precip_for_score = jnp.zeros(
                (), dtype=new_state.T.data.dtype)
        precip = micro_precip + convective_precip_for_score
        evap = applied_evap_mm_day(nonrad_tend, new_state)
        out = (new_state.T.data[0, 0, 0], qv, qcond, qcloud, qprecip,
               precip, evap)
        return (new_state, new_phys, rad_tend), out

    if use_cached_radiation:
        rad_sample, _ = _call_physics_with_fixed_sst(
            rad_physics_fn, scm.state, scm.phys_state,
        )
        rad_tend0 = _zero_tendency_like(rad_sample)

        @jax.jit
        def driver(state0, phys0):
            (state_f, phys_f, _rad_f), history = lax.scan(
                cached_radiation_body,
                (state0, phys0, rad_tend0),
                jnp.arange(nsteps),
            )
            return state_f, phys_f, history
    else:
        @jax.jit
        def driver(state0, phys0):
            (state_f, phys_f), history = lax.scan(
                body, (state0, phys0), jnp.arange(nsteps),
            )
            return state_f, phys_f, history

    try:
        final_state, _final_phys, history = driver(scm.state, scm.phys_state)
        del final_state
        (T_hist, qv_hist, qcond_hist, qcloud_hist, qprecip_hist,
         precip_hist, evap_hist) = (
            np.asarray(x, dtype=float) for x in history
        )
        last_steps = max(1, int(round(analysis_days * SECONDS_PER_DAY / dt)))
        last_steps = min(last_steps, nsteps)
        T_profile = T_hist[-last_steps:].mean(axis=0)
        qv_profile = qv_hist[-last_steps:].mean(axis=0)
        qcond_profile = np.maximum(qcond_hist[-last_steps:].mean(axis=0), 0.0)
        # Averaged over the SAME window as the total they sum to, so the three
        # profiles can be compared term by term against the reference's own
        # cloud/precipitating split.
        qcloud_profile = np.maximum(qcloud_hist[-last_steps:].mean(axis=0), 0.0)
        qprecip_profile = np.maximum(qprecip_hist[-last_steps:].mean(axis=0), 0.0)
        precip_mm_day = float(np.maximum(np.mean(precip_hist[-last_steps:]), 0.0))
        evap_mm_day = float(np.mean(evap_hist[-last_steps:]))

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
        realism = _realism_diagnostics(ref, T_profile, qv_profile)
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
        if not math.isfinite(precip_mm_day) or precip_mm_day < 0.0:
            reasons.append(f"invalid precipitation ({precip_mm_day:.3g} mm/day)")
        if (
            (require_equilibrium or require_realism)
            and math.isfinite(precip_mm_day)
            and precip_mm_day > PRECIP_MAX_SANE_MM_DAY
        ):
            reasons.append(
                f"precipitation too large ({precip_mm_day:.3g} mm/day)"
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
        realism_reasons = []
        if require_realism and finite:
            realism_reasons = realism_reasons_from_diagnostics(
                realism,
                mean_tol_K=MADIAB_MEAN_TOL_K,
                max_tol_K=MADIAB_MAX_TOL_K,
                cold_point_min_K=COLD_POINT_MIN_K,
                cold_point_max_K=COLD_POINT_MAX_K,
                trop_min_z_km=TROP_MIN_Z_KM,
                trop_max_z_km=TROP_MAX_Z_KM,
            )
            reasons.extend(realism_reasons)
        if finite:
            T_rmse, qv_rmse, cloud_rmse, precip_rmse, score = _score_profiles(
                ref, T_profile, qv_profile, qcond_profile, precip_mm_day,
            )
        else:
            score = float("inf")
            T_rmse = qv_rmse = cloud_rmse = precip_rmse = float("inf")
        if reasons:
            status = "failed"
        return RunDiagnostics(
            label=label,
            config=_config_scheme_dict(cfg),
            status=status,
            reason="; ".join(reasons),
            T_rmse=float(T_rmse),
            qv_rmse=float(qv_rmse),
            cloud_rmse=float(cloud_rmse),
            precip_rmse=float(precip_rmse),
            score=float(score),
            drift_T_rmse_K=float(drift_T),
            drift_qv_rmse=float(drift_qv),
            drift_qcond_rmse=float(drift_qcond),
            precip_mm_day=precip_mm_day,
            evap_mm_day=evap_mm_day,
            precip_ref_mm_day=float(ref.precip_ref_mm_day),
            moist_adiabat_mean_abs_K=float(realism.get("mean_abs_K", float("nan"))),
            moist_adiabat_max_abs_K=float(realism.get("max_abs_K", float("nan"))),
            moist_adiabat_bias_K=float(realism.get("bias_K", float("nan"))),
            cold_point_T_K=float(realism.get("cold_point_T_K", float("nan"))),
            cold_point_z_km=float(realism.get("cold_point_z_km", float("nan"))),
            realism_status=(
                "not_checked" if not require_realism
                else ("failed" if realism_reasons else "ok")
            ),
            T_profile=T_profile.tolist(),
            qv_profile=qv_profile.tolist(),
            qcond_profile=qcond_profile.tolist(),
            qcloud_profile=qcloud_profile.tolist(),
            qprecip_profile=qprecip_profile.tolist(),
            **(
                _subcloud_diagnostics(
                    ref, T_profile, qv_profile,
                    subcloud_top_m=subcloud_top_m, evap_mm_day=evap_mm_day,
                )
                # A non-finite column would make the saturation call return
                # garbage rather than raise; leave the sub-cloud fields at NaN
                # so the objective reads +inf and the run cannot win.
                if finite else {}
            ),
            **(_thermo_diagnostics(ref, T_profile, qv_profile) if finite else {}),
            **(
                # The HELD-OUT window: the analysis window immediately before the
                # scored one, already averaged above for the drift diagnostic.
                # Free, and it is the only thing that separates "the column sits
                # where the CRM does" from "one five-day slice of a drifting or
                # intermittent column happened to land there".
                {
                    "heldout_thermo_score": _thermo_diagnostics(
                        ref, T_prev, qv_prev)["thermo_score"],
                    "heldout_T_rmse_K": _weighted_rmse(
                        T_prev - ref.T_ref, ref.mass_weights),
                    "heldout_qv_rmse_g_kg": _weighted_rmse(
                        (qv_prev - ref.qv_ref) * 1000.0, ref.mass_weights),
                }
                if finite and prev_end > prev_start else {}
            ),
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
            precip_rmse=float("inf"),
            score=float("inf"),
            drift_T_rmse_K=float("inf"),
            drift_qv_rmse=float("inf"),
            drift_qcond_rmse=float("inf"),
            precip_ref_mm_day=float(ref.precip_ref_mm_day),
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
    require_realism: bool,
    equil_T_tol_K: float,
    equil_qv_tol: float,
    equil_qcond_tol: float,
    scm_microphysics_substeps: int = DEFAULT_SCM_MICROPHYSICS_SUBSTEPS,
    scm_convection_substeps: int = DEFAULT_SCM_CONVECTION_SUBSTEPS,
    surface_wind_m_s: float = DEFAULT_SCM_RCE_SURFACE_WIND_M_S,
    coriolis_s_inv: float = DEFAULT_SCM_RCE_CORIOLIS_S_INV,
    large_scale_forcing: str = DEFAULT_SCM_RCE_LARGE_SCALE_FORCING,
    bl_anchor_top_m: float = DEFAULT_SCM_RCE_BL_TOP_M,
    subcloud_top_m: float = DEFAULT_SUBCLOUD_TOP_M,
) -> RunDiagnostics:
    key = _config_cache_key(
        cfg,
        days,
        dt,
        scm_microphysics_substeps,
        scm_convection_substeps,
        surface_wind_m_s,
        coriolis_s_inv,
        large_scale_forcing,
        bl_anchor_top_m,
        # Post-processing only, but it CHANGES a recorded number, so a cache
        # entry computed at one layer top must never be served for another.
        subcloud_top_m,
    )
    if key not in cache:
        # COMPILED-EXECUTABLE HYGIENE.  Every entry here is a DIFFERENT static
        # PhysicsConfig, so ``run_scm_rce`` builds a fresh jitted closure and
        # XLA emits a fresh executable — nothing is ever reused between evals.
        # Retaining them is therefore pure cost, and it is not a small one:
        # the RRTMGP k-distribution tables are baked into each executable as
        # literals, so a 48-eval tune accumulated enough that XLA:CPU's LLVM
        # JIT could no longer mmap a section and aborted the process
        # ("LLVM ERROR: Unable to allocate section memory", jobs 9331806/7,
        # rc=134, 8 of 10 schemes lost).  Dropping the caches before each new
        # compile bounds the footprint at roughly one executable.
        jax.clear_caches()
        cache[key] = run_scm_rce(
            cfg,
            ref,
            label=label,
            days=days,
            dt=dt,
            scm_microphysics_substeps=scm_microphysics_substeps,
            scm_convection_substeps=scm_convection_substeps,
            analysis_days=analysis_days,
            require_equilibrium=require_equilibrium,
            require_realism=require_realism,
            equil_T_tol_K=equil_T_tol_K,
            equil_qv_tol=equil_qv_tol,
            equil_qcond_tol=equil_qcond_tol,
            surface_wind_m_s=surface_wind_m_s,
            coriolis_s_inv=coriolis_s_inv,
            large_scale_forcing=large_scale_forcing,
            bl_anchor_top_m=bl_anchor_top_m,
            subcloud_top_m=subcloud_top_m,
        )
    cached = cache[key]
    return RunDiagnostics(
        **{**asdict(cached), "label": label, "config": _config_scheme_dict(cfg)}
    )


def applied_precip_mm_day(applied_tend, like) -> jax.Array:
    """Surface precipitation [mm/day] from the tendency that was APPLIED.

    ``HydrostaticTendencies.precip`` is the microphysics' own surface
    sedimentation flux [kg/m^2/s, +into surface], summed over rain, cloud ice,
    snow and graupel; 1 kg/m^2 == 1 mm of liquid water, so the conversion is a
    single factor.

    This replaces a SECOND, diagnostic-only invocation of the microphysics,
    which was wrong twice over: it evaluated a different call than the one
    whose tendencies advanced the column, and its closure was built with the
    OUTER dt (600 s) while the applied operator runs at dt/substeps (20 s).
    Measured consequence: the campaign reported 1e-18..3e-5 mm/day for every
    scheme while its columns were losing 1.2-1.8 mm/day of water (jobs
    9361582/9361587/9361599).  The global model reads the applied value
    (``physics_pipeline.py:1473``), which is why it never showed this.

    CALL THIS ON THE MICROPHYSICS TENDENCY, never on a summed one:
    ``add_tendencies`` rebuilds ``HydrostaticTendencies`` from six fields
    (du/dT/dp_s/dphis/dv/tracers) and DROPS every diagnostic field, ``precip``
    included, so a summed tendency reports ``None`` -> 0.0 and silently
    reproduces the defect.  Gated by
    ``tests/unit/test_scm_rce_applied_precip.py``.
    """
    if applied_tend.precip is None:
        return jnp.zeros((), dtype=like.T.data.dtype)
    flat = jnp.reshape(applied_tend.precip.data, (-1,))
    if flat.shape[0] != 1:
        # Taking [0] of a multi-column field would silently score column 0 and
        # call it "the column".  This driver is single-column by construction;
        # a multi-column state means the caller is not what this readout
        # assumes, so refuse instead of reporting one column's rain.
        raise ValueError(
            f"applied_precip_mm_day expects a single-column state, got "
            f"{flat.shape[0]} columns; this readout scores one column.")
    return flat[0] * SECONDS_PER_DAY


def applied_evap_mm_day(applied_tend, like) -> jax.Array:
    """Surface evaporation [mm/day] from the tendency that was APPLIED.

    ``HydrostaticTendencies.lhflx_sfc`` is the turbulence scheme's own surface
    latent-heat flux [W/m^2]; ``E = LHF / L_v`` in kg/m^2/s, and 1 kg/m^2 ==
    1 mm of liquid water.  ``None`` when turbulence is off or the scheme
    computes no surface fluxes.

    Recorded per step ALONGSIDE the precipitation so the column's water budget
    ``d(CWV+CWC)/dt = E - P`` can be closed with both terms sampled the SAME
    way over the SAME window.  Both reviewers of the precipitation fix asked
    for exactly this: without it, a residual can always be blamed on comparing
    a snapshot E against a window-mean P, and never tested.  It is also what
    would catch water lost to a downstream positivity clip, which the readout
    tests cannot see.
    """
    if applied_tend.lhflx_sfc is None:
        return jnp.zeros((), dtype=like.T.data.dtype)
    flat = jnp.reshape(applied_tend.lhflx_sfc.data, (-1,))
    if flat.shape[0] != 1:
        raise ValueError(
            f"applied_evap_mm_day expects a single-column state, got "
            f"{flat.shape[0]} columns.")
    return flat[0] / constants.L_v * SECONDS_PER_DAY


def physical_profile_rmse(
    ref: ReferenceProfiles,
    run: RunDiagnostics,
) -> dict[str, float]:
    """Mass-weighted profile RMSE in PHYSICAL units (K, kg/kg) vs the CRM.

    The campaign's own ``score`` normalises each component by the reference's
    mass-weighted standard deviation, which makes the four terms commensurable
    for the optimiser but is not a quantity a reader can interpret.  Reports
    and figures need K and kg/kg, so both tuning drivers call THIS helper —
    same weights, same reference arrays, one implementation.

    A crashed run (empty ``T_profile``) yields NaNs rather than a zero RMSE,
    so a failure cannot masquerade as a perfect fit in a table.
    """
    if not run.T_profile:
        return {
            "T_rmse_K": float("nan"),
            "qv_rmse_kg_kg": float("nan"),
            "qcond_rmse_kg_kg": float("nan"),
        }
    w = jnp.asarray(ref.mass_weights)
    return {
        "T_rmse_K": float(weighted_rmse_jax(
            jnp.asarray(run.T_profile) - jnp.asarray(ref.T_ref), w)),
        "qv_rmse_kg_kg": float(weighted_rmse_jax(
            jnp.asarray(run.qv_profile) - jnp.asarray(ref.qv_ref), w)),
        "qcond_rmse_kg_kg": float(weighted_rmse_jax(
            jnp.asarray(run.qcond_profile) - jnp.asarray(ref.qcond_ref), w)),
    }


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
                "category", "scheme", "radiation", "T_RMSE", "qv_RMSE",
                "cloud_RMSE", "precip_RMSE", "precip_mm_day",
                "crm_precip_mm_day", "combined_score", "status",
                "realism_status", "reason", "drift_T_rmse_K",
                "drift_qv_rmse", "drift_qcond_rmse",
                "moist_adiabat_mean_abs_K", "moist_adiabat_max_abs_K",
                "cold_point_T_K", "cold_point_z_km",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "category": category,
                    "scheme": r.config[CATEGORY_CONFIG_FIELD[category]],
                    "radiation": r.config["radiation"],
                    "T_RMSE": r.T_rmse,
                    "qv_RMSE": r.qv_rmse,
                    "cloud_RMSE": r.cloud_rmse,
                    "precip_RMSE": r.precip_rmse,
                    "precip_mm_day": r.precip_mm_day,
                    "crm_precip_mm_day": r.precip_ref_mm_day,
                    "combined_score": r.score,
                    "status": r.status,
                    "realism_status": r.realism_status,
                    "reason": r.reason,
                    "drift_T_rmse_K": r.drift_T_rmse_K,
                    "drift_qv_rmse": r.drift_qv_rmse,
                    "drift_qcond_rmse": r.drift_qcond_rmse,
                    "moist_adiabat_mean_abs_K": r.moist_adiabat_mean_abs_K,
                    "moist_adiabat_max_abs_K": r.moist_adiabat_max_abs_K,
                    "cold_point_T_K": r.cold_point_T_K,
                    "cold_point_z_km": r.cold_point_z_km,
                }
            )


def _write_profile_csv(path: Path, ref: ReferenceProfiles, results: list[RunDiagnostics]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scheme", "z_km", "sigma", "T_K", "qv_kg_kg",
                "qcond_kg_kg", "T_crm_K", "qv_crm_kg_kg", "qcond_crm_kg_kg",
            ],
        )
        writer.writeheader()
        for r in results:
            if not r.T_profile or not r.qv_profile or not r.qcond_profile:
                continue
            scheme = r.config["convection"]
            for k, z_m in enumerate(ref.z_m):
                writer.writerow(
                    {
                        "scheme": scheme,
                        "z_km": float(z_m) / 1000.0,
                        "sigma": float(ref.sigma_full[k]),
                        "T_K": r.T_profile[k],
                        "qv_kg_kg": r.qv_profile[k],
                        "qcond_kg_kg": r.qcond_profile[k],
                        "T_crm_K": float(ref.T_ref[k]),
                        "qv_crm_kg_kg": float(ref.qv_ref[k]),
                        "qcond_crm_kg_kg": float(ref.qcond_ref[k]),
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
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 5.2), sharey=False)
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
    for ax, (name, ref_prof, scm_prof, units) in zip(axes[:3], panels):
        ax.plot(ref_prof, z_km, color="#1f4e79", lw=2.0, label="CRM")
        ax.plot(scm_prof, z_km, color="#d95f02", lw=2.0, label="SCM")
        ax.set_xlabel(f"{name} [{units}]")
        ax.set_ylim(0.0, float(np.nanmax(z_km)))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("z [km]")
    axes[0].legend(loc="best")
    precip_ax = axes[3]
    precip_ax.bar(
        [0, 1],
        [ref.precip_ref_mm_day, run.precip_mm_day],
        color=["#1f4e79", "#d95f02"],
        width=0.6,
    )
    precip_ax.set_xticks([0, 1])
    precip_ax.set_xticklabels(["CRM", "SCM"])
    precip_ax.set_ylabel("surface precip [mm/day]")
    precip_ax.grid(axis="y", alpha=0.25)
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


def _tunable_subconfig(subcfg):
    """The object whose DIRECT NamedTuple fields carry the spec'd tunable params.

    Flat for every scheme except full CLUBB, which nests its closure coefficients
    in ``.params`` (a ``CLUBBParams``); the ``__param_spec__``/scheme_key live on
    that nested tuple, so it — not the ``CLUBBConfig`` wrapper — is the override
    target. ``None`` passes through."""
    if subcfg is None:
        return None
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    if isinstance(subcfg, CLUBBConfig):
        return subcfg.params
    return subcfg


def _rewrap_tunable_subconfig(subcfg, tuned):
    """Inverse of :func:`_tunable_subconfig`: fold a tuned tunable-object back
    into the scheme sub-config that ``_set_active_subconfig`` writes."""
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    if isinstance(subcfg, CLUBBConfig):
        return subcfg._replace(params=tuned)
    return tuned


def _scheme_key_for_subconfig(subcfg) -> str | None:
    if subcfg is None:
        return None
    # Descend into CLUBB's nested .params so the spec'd CLUBBParams is found.
    cls_name = type(_tunable_subconfig(subcfg)).__name__
    for meta in build_registry():
        if meta.config_class == cls_name:
            return meta.scheme_key
    return None


def _set_active_subconfig(top_cfg: PhysicsConfig, category: str, subcfg) -> PhysicsConfig:
    component = getattr(top_cfg, CATEGORY_CONFIG_FIELD[category])
    scheme = component.scheme
    component_new = component._replace(**{scheme: subcfg})
    return top_cfg._replace(**{CATEGORY_CONFIG_FIELD[category]: component_new})


# --------------------------------------------------------------------------- #
# Matched-kernel override (convection-scheme intercomparison)
# --------------------------------------------------------------------------- #
SUBSIDENCE_SOLVE_MODES = ("as_shipped", "implicit_flux", "advective")


def apply_subsidence_solve_override(
    cfg: PhysicsConfig,
    mode: str,
    *,
    category: str = "convection",
) -> tuple[PhysicsConfig, str]:
    """Force the mass-flux family onto ONE vertical transport kernel.

    Rankings across convection schemes are confounded when the schemes do not
    share a transport kernel: Bechtold/EDMF/Kain-Fritsch ship the conservative
    ``"implicit_flux"`` solve while Tiedtke/Emanuel/Zhang-McFarlane/mass_flux
    ship the leaky ``"advective"`` one, so a score difference partly measures
    the KERNEL, not the scheme.  ``mode="implicit_flux"`` puts every scheme
    that HAS the knob on the conservative solve, isolating scheme physics;
    ``mode="as_shipped"`` leaves every default untouched (what users get).

    Returns ``(cfg, status)``.  ``status`` is a short human-readable string
    recorded next to every number so a reader can never mistake which arm a
    table came from:

    * ``"as_shipped"``                    -- nothing changed;
    * ``"forced:<scheme>=<solve>"``       -- the knob was set;
    * ``"not_applicable:<scheme>"``       -- this scheme has NO
      ``subsidence_solve`` field, so it is OUTSIDE the matched-kernel family.
      Reported explicitly rather than silently skipped, because a reader must
      not assume such a scheme was kernel-matched.  Three distinct reasons:

      - ``sbm``, ``dca``, ``kuo`` -- adjustment / Kuo-type closures with no
        compensating-subsidence mass-flux kernel at all;
      - ``emanuel`` -- its SHIPPED path (``use_genuine_mixing=True``) is a
        buoyancy-sorting mixing matrix that never calls the shared kernel;
        only the legacy surrogate branch does, and that branch's
        ``sort_multiplier`` rescaling would leave an unpaired vapor debit
        under a vapor-debiting solve.  So Emanuel cannot be kernel-matched.

    Raises on an unknown ``mode`` (dispatch-hardening: a typo must not
    silently select the as-shipped arm and be reported as the matched one).
    """
    if mode not in SUBSIDENCE_SOLVE_MODES:
        raise ValueError(
            f"apply_subsidence_solve_override: unknown mode {mode!r}; "
            f"expected one of {SUBSIDENCE_SOLVE_MODES}"
        )
    _component, scheme, subcfg = _active_subconfig(cfg, category)
    if mode == "as_shipped":
        return cfg, "as_shipped"
    if subcfg is None or not hasattr(subcfg, "subsidence_solve"):
        return cfg, f"not_applicable:{scheme}"
    new_subcfg = subcfg._replace(subsidence_solve=mode)
    return _set_active_subconfig(cfg, category, new_subcfg), f"forced:{scheme}={mode}"


def _raw_from_physical(value: float, constraint) -> jax.Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    if constraint.transform == "sigmoid":
        return range_to_sigmoid_array(arr, constraint.min_val, constraint.max_val).astype(jnp.float64)
    if constraint.transform == "softplus":
        y = jnp.maximum(arr, jnp.asarray(PROFILE_FLOOR, dtype=jnp.float64))
        return (y + jnp.log1p(-jnp.exp(-y))).astype(jnp.float64)
    return arr


# A parameter whose bounds span this many decades is sampled LOG-uniformly.
# Linear sampling of, say, [1e-6, 1e-2] puts ~90 % of the draws in the top
# decade and never visits the bottom three, so the tuned value is biased high
# by construction — for autoconversion / entrainment / rate coefficients that
# is most of the plausible range.  Two decades is the threshold at which the
# distortion (a factor ~100 in sampling density across the range) stops being
# a detail.
_LOG_SAMPLING_DECADES = 2.0


def _sample_scale(constraint) -> str:
    """``"log"`` for a strictly-positive range spanning >= 2 decades."""
    lo = float(constraint.min_val)
    hi = float(constraint.max_val)
    if lo > 0.0 and hi > lo and (math.log10(hi) - math.log10(lo)) >= _LOG_SAMPLING_DECADES:
        return "log"
    return "linear"


def _interp(constraint, frac: float) -> float:
    """Value at fraction ``frac`` of the range, in the parameter's own scale.

    Clamped to the bounds: ``lo * (hi/lo)**frac`` is not exactly ``hi`` at
    ``frac == 1`` in floating point, and the tuner asserts every candidate is
    inside ``[lo, hi]`` — a one-ULP overshoot would abort a whole scheme's arm
    hours in.
    """
    lo = float(constraint.min_val)
    hi = float(constraint.max_val)
    if _sample_scale(constraint) == "log":
        value = lo * (hi / lo) ** frac
    else:
        value = lo + frac * (hi - lo)
    return float(min(max(value, lo), hi))


def _frac_from_value(constraint, value: float) -> float:
    """Inverse of :func:`_interp`: where ``value`` sits in the range, in the
    parameter's own sampling scale.

    Used by the local refinement so a step of "0.1 of the range" means the same
    thing for a linearly-sampled fraction and for a log-sampled rate
    coefficient (where it is a multiplicative step).  Clamped to [0, 1] so a
    default sitting fractionally outside its own declared bounds — which the
    registry permits, the bounds being a search range rather than a hard
    validity limit — cannot produce a negative or >1 fraction and silently skip
    every refinement of that parameter.
    """
    lo = float(constraint.min_val)
    hi = float(constraint.max_val)
    if not (hi > lo):
        return 0.0
    if _sample_scale(constraint) == "log":
        value = min(max(float(value), lo), hi)
        frac = math.log(value / lo) / math.log(hi / lo)
    else:
        frac = (float(value) - lo) / (hi - lo)
    return float(min(max(frac, 0.0), 1.0))


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
            cand = dict(defaults)
            cand[name] = _interp(c, frac)
            budget -= 1
            yield cand
    while budget > 0:
        cand = {}
        for name, c in zip(names, constraints):
            # ONE rng.uniform(0, 1) draw per parameter regardless of scale, so
            # the random STREAM is identical for linear and log parameters and
            # a longer budget stays a strict superset of a shorter one.
            cand[name] = _interp(c, float(rng.uniform(0.0, 1.0)))
        budget -= 1
        yield cand


#: Selectable tuning parameter sets.  The first three are the registry's own
#: tiers; ``physical`` is the DERIVATIVE-FREE calibration set defined below.
PARAM_SETS = ("core", "extended", "aggressive", "physical")

#: Marker the registry uses on a tier-0 parameter that is physically real but
#: whose AD gradient vanishes (``see _CAPE_TRIGGER_AD_NOTE``, #1417).  Matched
#: on the spec's own reference string so a newly re-classified parameter is
#: picked up automatically instead of needing a name added here.
_AD_UNREACHABLE_MARKER = "AD-unreachable"

#: Parameters excluded from the ``physical`` set BY NAME, each with the reason.
#: Only for cases the ``category`` field cannot express.
PHYSICAL_SET_NAME_EXCLUSIONS = {
    "atm.conv.KainFritschConfig.dtlcl_dx_scale":
        "grid-length scaling (Kain 2004): a single column has no grid length, "
        "so tuning it absorbs a resolution dependence into a column fit",
}


def resolve_param_selection(
    scheme_key: str, param_set: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """``(tier, include_tier0, exclude)`` for ``build_trainable_params``.

    ``core``/``extended``/``aggressive`` pass straight through to the registry
    tiers.  ``physical`` is the set a DERIVATIVE-FREE calibration should use
    when it is asked to tune "all the parameters of the scheme", and it differs
    from ``aggressive`` in both directions — which is the whole reason it
    exists, because ``aggressive`` is neither a superset nor a subset of "the
    physics":

    * it DROPS every ``category == "numerics"`` parameter (measured: four of
      them are tier 3, e.g. a sigmoid layer-edge width and a mass-flux
      normalisation scale), because a value chosen to fit a column is then a
      statement about the discretisation;
    * it ADDS the tier-0 parameters whose exclusion reason is that their AD
      GRADIENT VANISHES — the CAPE trigger threshold of eight of the ten
      schemes.  Those are textbook closure parameters; a gradient-free search
      can move them, and leaving them out would mean the campaign never touched
      the trigger of most of the schemes it claims to have tuned.
    """
    if param_set not in PARAM_SETS:
        raise ValueError(
            f"resolve_param_selection: unknown param_set {param_set!r}; "
            f"expected one of {PARAM_SETS}")
    if param_set != "physical":
        return param_set, (), ()
    metas = [m for m in build_registry() if m.scheme_key == scheme_key]
    include_tier0 = tuple(sorted(
        m.qualified_name for m in metas
        if m.tunable_tier == 0 and _AD_UNREACHABLE_MARKER in m.reference
    ))
    exclude = tuple(sorted(
        m.qualified_name for m in metas
        if m.tunable_tier != 0 and (
            m.category == "numerics"
            or m.qualified_name in PHYSICAL_SET_NAME_EXCLUSIONS
        )
    ))
    return "aggressive", include_tier0, exclude


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
    require_realism: bool,
    tune_evals: int,
    seed: int,
    equil_T_tol_K: float,
    equil_qv_tol: float,
    equil_qcond_tol: float,
    scm_microphysics_substeps: int = DEFAULT_SCM_MICROPHYSICS_SUBSTEPS,
    scm_convection_substeps: int = DEFAULT_SCM_CONVECTION_SUBSTEPS,
    surface_wind_m_s: float = DEFAULT_SCM_RCE_SURFACE_WIND_M_S,
    coriolis_s_inv: float = DEFAULT_SCM_RCE_CORIOLIS_S_INV,
    large_scale_forcing: str = DEFAULT_SCM_RCE_LARGE_SCALE_FORCING,
    bl_anchor_top_m: float = DEFAULT_SCM_RCE_BL_TOP_M,
    subcloud_top_m: float = DEFAULT_SUBCLOUD_TOP_M,
    objective: str = "combined",
    param_set: str = "extended",
    refine_frac: float = 0.0,
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
        require_realism=require_realism,
        equil_T_tol_K=equil_T_tol_K,
        equil_qv_tol=equil_qv_tol,
        equil_qcond_tol=equil_qcond_tol,
        scm_microphysics_substeps=scm_microphysics_substeps,
        scm_convection_substeps=scm_convection_substeps,
        surface_wind_m_s=surface_wind_m_s,
        coriolis_s_inv=coriolis_s_inv,
        large_scale_forcing=large_scale_forcing,
        bl_anchor_top_m=bl_anchor_top_m,
        subcloud_top_m=subcloud_top_m,
    )
    if scheme_key is None or subcfg is None:
        return base_cfg, [], default_run
    tier, include_tier0, exclude = resolve_param_selection(scheme_key, param_set)
    params = build_trainable_params(
        active_scheme_keys={scheme_key},
        tier=tier,
        include_tier0=include_tier0,
        exclude=exclude,
        dtype=jnp.float64,
    )
    if not params.constraints:
        return base_cfg, [], default_run
    constraints = params.constraints
    defaults = {c.name: float(params.as_dict()[c.name]) for c in constraints}
    best_cfg = base_cfg
    best_run = default_run
    best_values = defaults
    # A non-finite incumbent score makes EVERY ``trial < best`` comparison
    # False, so the tuner would report the defaults as "tuned" while silently
    # discarding every trial.  ``objective_value`` maps NaN to +inf for exactly
    # that reason.
    best_score = objective_value(default_run, objective)
    # The random phase gets the budget minus whatever the refinement stage is
    # given.  Computed BEFORE the loop so the two phases cannot overspend
    # between them, and floored at 1 so `refine_frac=1.0` still evaluates the
    # defaults.
    n_refine = max(0, int(round(float(refine_frac) * tune_evals)))
    n_random = max(1, tune_evals - n_refine)
    candidates = list(_candidate_values(defaults, constraints, n_random, seed))

    def _try(values: dict[str, float], label: str) -> None:
        """Evaluate one candidate and keep it if it beats the incumbent."""
        nonlocal best_score, best_cfg, best_run, best_values
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
        # Apply to the tunable object (CLUBB's nested .params, else subcfg) and
        # re-wrap so _set_active_subconfig receives the full scheme sub-config.
        tuned_tunable = apply_param_overrides(_tunable_subconfig(subcfg), field_values)
        tuned_subcfg = _rewrap_tunable_subconfig(subcfg, tuned_tunable)
        trial_cfg = _set_active_subconfig(base_cfg, category, tuned_subcfg)
        trial_run = run_cached(
            cache,
            trial_cfg,
            ref,
            label=label,
            days=days,
            dt=dt,
            analysis_days=analysis_days,
            require_equilibrium=require_equilibrium,
            require_realism=require_realism,
            equil_T_tol_K=equil_T_tol_K,
            equil_qv_tol=equil_qv_tol,
            equil_qcond_tol=equil_qcond_tol,
            scm_microphysics_substeps=scm_microphysics_substeps,
            scm_convection_substeps=scm_convection_substeps,
            surface_wind_m_s=surface_wind_m_s,
            coriolis_s_inv=coriolis_s_inv,
            large_scale_forcing=large_scale_forcing,
            bl_anchor_top_m=bl_anchor_top_m,
            subcloud_top_m=subcloud_top_m,
        )
        trial_score = objective_value(trial_run, objective)
        if trial_run.status == "ok" and trial_score < best_score:
            best_score = trial_score
            best_cfg = trial_cfg
            best_run = trial_run
            best_values = {
                c.name: float(field_values[c.field]) for c in constraints
            }

    for i, values in enumerate(candidates):
        _try(values, f"tune:{category}:{scheme}:eval{i:03d}")

    # LOCAL REFINEMENT.  A uniform random search over 19-25 parameters resolves
    # nothing: 200 draws in 19 dimensions sit ~0.76 of the range apart on every
    # axis, so the incumbent is a lucky corner rather than a local optimum.  A
    # coordinate sweep AROUND the incumbent — each parameter moved alone by a
    # shrinking fraction of its own range, both directions, incumbent updated
    # greedily — costs the same per evaluation and is the cheapest thing that
    # turns "the best of N lottery tickets" into "a point no single-parameter
    # move improves".  It does NOT establish convergence, and does not remove
    # the cross-scheme dimensionality confound; that is what the second seed is
    # for.  Off by default (refine_frac=0.0) so every existing campaign keeps
    # its exact search.
    if n_refine > 0:
        step_fracs = (0.25, 0.10, 0.04)
        spent = 0
        for step in step_fracs:
            if spent >= n_refine:
                break
            for c in constraints:
                if spent >= n_refine:
                    break
                for direction in (+1.0, -1.0):
                    if spent >= n_refine:
                        break
                    incumbent = dict(best_values)
                    lo = float(c.min_val)
                    hi = float(c.max_val)
                    current = float(incumbent[c.name])
                    frac = (
                        # Move in the parameter's OWN sampling scale, so a
                        # log-sampled rate coefficient takes a multiplicative
                        # step rather than one dominated by its top decade.
                        _frac_from_value(c, current) + direction * step
                    )
                    if not (0.0 <= frac <= 1.0):
                        continue
                    proposal = _interp(c, frac)
                    if proposal == current or not (lo <= proposal <= hi):
                        continue
                    incumbent[c.name] = proposal
                    _try(
                        incumbent,
                        f"refine:{category}:{scheme}:s{step:g}:"
                        f"{c.field}:{'+' if direction > 0 else '-'}",
                    )
                    spent += 1

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
                # The objective actually minimised, not the historical combined
                # score: a record whose "score_tuned" comes from a different
                # metric than the search used reads as a failed search whenever
                # the two disagree.  ``tune_focused_params`` already does this.
                score_default=objective_value(default_run, objective),
                score_tuned=objective_value(best_run, objective),
            )
        )
    return best_cfg, records, best_run


def tune_focused_params(
    base_cfg: PhysicsConfig,
    ref: ReferenceProfiles,
    cache: dict[str, RunDiagnostics],
    *,
    categories: Sequence[str],
    include: Sequence[str],
    days: float,
    dt: float,
    analysis_days: float,
    require_equilibrium: bool,
    require_realism: bool,
    tune_evals: int,
    seed: int,
    equil_T_tol_K: float,
    equil_qv_tol: float,
    equil_qcond_tol: float,
    tier: str = "extended",
    objective: str = "subcloud",
    scm_microphysics_substeps: int = DEFAULT_SCM_MICROPHYSICS_SUBSTEPS,
    scm_convection_substeps: int = DEFAULT_SCM_CONVECTION_SUBSTEPS,
    surface_wind_m_s: float = DEFAULT_SCM_RCE_SURFACE_WIND_M_S,
    coriolis_s_inv: float = DEFAULT_SCM_RCE_CORIOLIS_S_INV,
    large_scale_forcing: str = DEFAULT_SCM_RCE_LARGE_SCALE_FORCING,
    bl_anchor_top_m: float = DEFAULT_SCM_RCE_BL_TOP_M,
    subcloud_top_m: float = DEFAULT_SUBCLOUD_TOP_M,
) -> tuple[PhysicsConfig, list[TuneRecord], RunDiagnostics, RunDiagnostics]:
    """Tune ONE NAMED parameter set that spans SEVERAL scheme categories.

    ``tune_category_winner`` searches every extended-tier parameter of a single
    category.  The sub-cloud layer is not owned by one category: it is
    ventilated by the boundary-layer scheme's mixing, cooled and moistened by
    the microphysics' rain re-evaporation, and dried by the convection scheme's
    downdrafts.  Those knobs interact, so tuning them jointly is a different
    experiment from tuning each category's whole parameter set in turn, and it
    is the one the sub-cloud objective needs.

    ``include`` is an explicit list of registry-qualified ``scheme_key.field``
    names — the experiment's hypothesis written down, not a tier sweep. Every
    name must resolve to a parameter of one of the ACTIVE schemes; a name that
    silently selects nothing would shrink the search while the log still
    claimed N parameters, so it raises.

    Returns ``(best_cfg, records, default_run, best_run)``.  ``default_run`` is
    returned explicitly rather than left implicit: when no candidate wins,
    ``best_run IS default_run``, and a caller that cannot tell the two apart
    reads "tuned == default" as a code-path signature, which it is not.
    """
    if objective not in TUNE_OBJECTIVES:
        raise ValueError(
            f"tune_focused_params: unknown objective {objective!r}; "
            f"expected one of {TUNE_OBJECTIVES}")
    resolved: dict[str, tuple[str, Any]] = {}
    for category in categories:
        if category not in CATEGORY_CONFIG_FIELD:
            raise ValueError(
                f"tune_focused_params: unknown category {category!r}; "
                f"expected one of {sorted(CATEGORY_CONFIG_FIELD)}")
        _component, scheme, subcfg = _active_subconfig(base_cfg, category)
        scheme_key = _scheme_key_for_subconfig(subcfg)
        if scheme_key is None or subcfg is None:
            # e.g. convection="none": no tunable sub-config exists.  Skipping is
            # correct, but it must be visible, never silent.
            print(f"  [focused-tune] {category}={scheme}: no tunable "
                  f"sub-config, skipped")
            continue
        resolved[scheme_key] = (category, subcfg)
    if not resolved:
        raise ValueError(
            "tune_focused_params: none of the requested categories "
            f"{list(categories)} has a tunable sub-config in this config.")

    # RESOLVE THE PARAMETER SET FIRST.  A bad ``include`` must fail before the
    # 100-day default column is paid for, not after it.
    params = build_trainable_params(
        active_scheme_keys=set(resolved),
        tier=tier,
        include=tuple(include),
        dtype=jnp.float64,
    )
    constraints = [c for c in params.constraints if c.name in set(include)]
    missing = sorted(set(include) - {c.name for c in constraints})
    if missing:
        raise ValueError(
            f"tune_focused_params: {missing} matched no parameter of the "
            f"active schemes {sorted(resolved)}. A selector that matches "
            "nothing is a hard error, never a quietly smaller search.")
    if not constraints:
        raise ValueError("tune_focused_params: empty parameter set.")
    print(f"  [focused-tune] {len(constraints)} parameters across "
          f"{len(resolved)} schemes, objective={objective}, "
          f"{tune_evals} evaluations")

    default_run = run_cached(
        cache, base_cfg, ref,
        label="focused-tune-default",
        days=days, dt=dt, analysis_days=analysis_days,
        require_equilibrium=require_equilibrium,
        require_realism=require_realism,
        equil_T_tol_K=equil_T_tol_K,
        equil_qv_tol=equil_qv_tol,
        equil_qcond_tol=equil_qcond_tol,
        scm_microphysics_substeps=scm_microphysics_substeps,
        scm_convection_substeps=scm_convection_substeps,
        surface_wind_m_s=surface_wind_m_s,
        coriolis_s_inv=coriolis_s_inv,
        large_scale_forcing=large_scale_forcing,
        bl_anchor_top_m=bl_anchor_top_m,
        subcloud_top_m=subcloud_top_m,
    )

    defaults = {c.name: float(params.as_dict()[c.name]) for c in constraints}
    best_cfg = base_cfg
    best_run = default_run
    best_values = dict(defaults)
    best_score = objective_value(default_run, objective)

    for i, values in enumerate(
            _candidate_values(defaults, constraints, tune_evals, seed)):
        trial_params = TrainablePhysicsParams(
            raw_values={c.name: _raw_from_physical(values[c.name], c)
                        for c in constraints},
            constraints=constraints,
        )
        overrides = trial_params.to_overrides()
        trial_cfg = base_cfg
        # The value that was APPLIED, not the one that was requested: the
        # sampler's physical draw goes through raw <-> sigmoid, so the two can
        # differ at round-off and it is the applied one that produced the score.
        applied: dict[str, float] = {}
        for scheme_key, (category, subcfg) in resolved.items():
            field_values = overrides.get(scheme_key, {})
            if not field_values:
                continue
            for c in constraints:
                if c.name.rsplit(".", 1)[0] != scheme_key:
                    continue
                value = float(field_values[c.field])
                lo, hi = float(c.min_val), float(c.max_val)
                if not (lo <= value <= hi):
                    raise AssertionError(
                        f"{c.name}={value} escaped bounds [{lo}, {hi}]")
                applied[c.name] = value
            tuned_tunable = apply_param_overrides(
                _tunable_subconfig(subcfg), field_values)
            trial_cfg = _set_active_subconfig(
                trial_cfg, category,
                _rewrap_tunable_subconfig(subcfg, tuned_tunable))
        if len(applied) != len(constraints):
            raise AssertionError(
                f"focused tune applied {len(applied)} of {len(constraints)} "
                "parameters; an override was dropped between the sampler and "
                "the config.")
        trial_run = run_cached(
            cache, trial_cfg, ref,
            label=f"focused-tune:eval{i:03d}",
            days=days, dt=dt, analysis_days=analysis_days,
            require_equilibrium=require_equilibrium,
            require_realism=require_realism,
            equil_T_tol_K=equil_T_tol_K,
            equil_qv_tol=equil_qv_tol,
            equil_qcond_tol=equil_qcond_tol,
            scm_microphysics_substeps=scm_microphysics_substeps,
            scm_convection_substeps=scm_convection_substeps,
            surface_wind_m_s=surface_wind_m_s,
            coriolis_s_inv=coriolis_s_inv,
            large_scale_forcing=large_scale_forcing,
            bl_anchor_top_m=bl_anchor_top_m,
            subcloud_top_m=subcloud_top_m,
        )
        trial_score = objective_value(trial_run, objective)
        if trial_run.status == "ok" and trial_score < best_score:
            best_score = trial_score
            best_cfg = trial_cfg
            best_run = trial_run
            best_values = dict(applied)

    meta_by_name = {m.qualified_name: m for m in build_registry()}
    records = []
    for c in constraints:
        meta = meta_by_name[c.name]
        tuned = float(best_values[c.name])
        lo, hi = float(c.min_val), float(c.max_val)
        if not (lo <= tuned <= hi):
            raise AssertionError(f"tuned value escaped bounds: {c.name}={tuned}")
        scheme_key = c.name.rsplit(".", 1)[0]
        category, subcfg = resolved[scheme_key]
        records.append(
            TuneRecord(
                category=category,
                scheme=_active_subconfig(base_cfg, category)[1],
                scheme_key=scheme_key,
                parameter=c.field,
                default=float(defaults[c.name]),
                tuned=tuned,
                lower=lo,
                upper=hi,
                units=meta.units,
                # The objective actually minimised, so a reader cannot mistake
                # a sub-cloud tune for a combined-score one.
                score_default=objective_value(default_run, objective),
                score_tuned=objective_value(best_run, objective),
            )
        )
    return best_cfg, records, default_run, best_run


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
        f"SCM fixed SST: {FIXED_SST_K:.1f} K; radiation: `{args.radiation}`; "
        f"radiation refresh: every {args.radiation_update_interval_steps} "
        f"steps; dt: {args.dt:.1f} s; background surface wind: "
        f"{args.surface_wind_m_s:.2f} m/s with f={args.coriolis_s_inv:.2e} s^-1; "
        f"large-scale forcing: `{args.large_scale_forcing}`; "
        "lowest-level air temperature is anchored to the fixed SST "
        "after each SCM step while water vapor remains prognostic; "
        "SCM microphysics substeps for "
        f"{', '.join(SCM_MICROPHYSICS_SUBSTEP_SCHEMES)}: "
        f"{args.scm_microphysics_substeps}; "
        "SCM convection substeps (ALL schemes, uniform): "
        f"{args.scm_convection_substeps} "
        f"({args.dt / max(args.scm_convection_substeps, 1):.0f} s convective "
        f"step at --dt {args.dt:.0f}); "
        f"days: {args.days:.3g}; analysis window: {args.analysis_days:.3g} d.",
        f"CRM equilibrium surface precipitation: {ref.precip_ref_mm_day:.3f} mm/day "
        f"({ref.precip_unit_note}; {len(ref.precip_files_used)} surface files).",
        "Combined score is the normalized T/qv/cloud profile RMSE plus a "
        f"surface-precipitation term normalized by {PRECIP_NORMALIZATION_MM_DAY:g} "
        f"mm/day with weight {PRECIP_SCORE_WEIGHT:g}.",
        "Realism gate: finite bounded profiles, q_v and condensate non-negative, "
        "small equilibrium drift, free-tropospheric temperature within "
        f"{MADIAB_MEAN_TOL_K:g} K mean and {MADIAB_MAX_TOL_K:g} K max of the "
        "surface-anchored moist pseudo-adiabat, and a cold point in "
        f"[{COLD_POINT_MIN_K:g}, {COLD_POINT_MAX_K:g}] K at "
        f"[{TROP_MIN_Z_KM:g}, {TROP_MAX_Z_KM:g}] km.",
        "",
        "Large-scale forcing note: the plane CRM RCE itself imposes no external "
        "subsidence. When `large_scale_forcing='crm_clear_sky_subsidence'`, the "
        "SCM campaign approximates the CRM-resolved convective circulation by "
        "applying the area-weighted clear-sky descent `<w 1_clear>` diagnosed "
        f"from the CRM volumes with condensate threshold "
        f"{CRM_CLEAR_SKY_COND_THRESHOLD:g} kg/kg. The profile is positive-upward, "
        "clipped to subsidence only, and closed at the model top and surface. "
        "This is an SCM comparability forcing, not a change to the CRM/plane "
        "path; use `--large-scale-forcing none` for a free-running SCM.",
        "",
        "Recommended defaults live in the atmosphere physics `*Config` "
        "NamedTuple defaults and the combined `PhysicsConfig`; this campaign "
        "writes a recommended override JSON, it does not mutate production "
        "defaults.",
        "",
        "## Rankings",
    ]
    if args.radiation == "rrtmgp":
        lines += [
            "",
            "> Caveat: the bundled CRM reference under "
            "`results/rcemip1_n128_ocean` was generated with gray radiation. "
            "These RRTMGP-SCM results are the physical SCM default and are "
            "realism-gated, but the stratospheric profile comparison is not "
            "strictly apples-to-apples until a matching RRTMGP CRM truth run "
            "is generated.",
            "",
        ]
    for category, rows in rankings.items():
        heading = "convection (bonus)" if category == "convection" else category
        lines.append(f"### {heading}")
        lines.append(
            "| rank | scheme | score | T | qv | cloud | precip | SCM P | CRM P | "
            "realism | status | reason |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
        for i, r in enumerate(rows, start=1):
            scheme = r.config[CATEGORY_CONFIG_FIELD[category]]
            score = "inf" if not math.isfinite(r.score) else f"{r.score:.6g}"
            T = "inf" if not math.isfinite(r.T_rmse) else f"{r.T_rmse:.6g}"
            qv = "inf" if not math.isfinite(r.qv_rmse) else f"{r.qv_rmse:.6g}"
            cloud = "inf" if not math.isfinite(r.cloud_rmse) else f"{r.cloud_rmse:.6g}"
            precip = "inf" if not math.isfinite(r.precip_rmse) else f"{r.precip_rmse:.6g}"
            lines.append(
                f"| {i} | {scheme} | {score} | {T} | {qv} | {cloud} | "
                f"{precip} | {r.precip_mm_day:.3g} | {r.precip_ref_mm_day:.3g} | "
                f"{r.realism_status} | {r.status} | {r.reason.replace('|', '/')} |"
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
    parser.add_argument(
        "--scm-microphysics-substeps",
        type=int,
        default=DEFAULT_SCM_MICROPHYSICS_SUBSTEPS,
        help=(
            "Fixed SCM-only microphysics substeps per outer step. Default "
            f"{DEFAULT_SCM_MICROPHYSICS_SUBSTEPS} gives a 20 s "
            "microphysics step at --dt 600 for active campaign microphysics "
            "schemes and lets the campaign record actual pre-update "
            "surface-precipitation rates."
        ),
    )
    parser.add_argument(
        "--scm-convection-substeps",
        type=int,
        default=DEFAULT_SCM_CONVECTION_SUBSTEPS,
        help=(
            "Fixed SCM-only convection substeps per outer step, applied "
            "UNIFORMLY to every convection scheme. Default "
            f"{DEFAULT_SCM_CONVECTION_SUBSTEPS} gives a 60 s convective "
            "adjustment step at --dt 600 without changing the plane/CRM path. "
            "Uniformity is required for the intercomparison to be controlled: "
            "ranking schemes run at different convective timesteps measures "
            "the timestep as much as the physics."
        ),
    )
    parser.add_argument("--analysis-days", type=float, default=DEFAULT_ANALYSIS_DAYS)
    parser.add_argument(
        "--surface-wind-m-s",
        type=float,
        default=DEFAULT_SCM_RCE_SURFACE_WIND_M_S,
        help=(
            "SCM RCE background near-surface wind used to drive ocean "
            "bulk evaporation. Default 5 m/s is a tropical trade-wind scale."
        ),
    )
    parser.add_argument(
        "--coriolis-s-inv",
        type=float,
        default=DEFAULT_SCM_RCE_CORIOLIS_S_INV,
        help=(
            "Weak geostrophic-relaxation Coriolis parameter used with "
            "--surface-wind-m-s so surface drag does not spin the column "
            "down to a zero-evaporation state."
        ),
    )
    parser.add_argument(
        "--large-scale-forcing",
        choices=SCM_RCE_LARGE_SCALE_FORCING_CHOICES,
        default=DEFAULT_SCM_RCE_LARGE_SCALE_FORCING,
        help=(
            "Optional SCM-only large-scale forcing. "
            "'crm_clear_sky_subsidence' diagnoses the area-weighted clear-sky "
            "descent from the same CRM reference volumes and applies it through "
            "SCMForcing.subsidence_w; 'none' leaves the free-running SCM "
            "without large-scale vertical advection. The low-level SCM default "
            "remains no subsidence."
        ),
    )
    parser.add_argument(
        "--radiation",
        choices=["rrtmgp", "gray"],
        default=BASELINE_SCHEMES["radiation"],
        help=(
            "SCM RCE radiation. Default rrtmgp uses the RCEMIP fixed-sun, "
            "MLS-ozone setup; gray is kept for the existing gray-CRM comparison "
            "and fast smoke runs."
        ),
    )
    parser.add_argument(
        "--radiation-update-interval-steps",
        type=int,
        default=None,
        help=(
            "SCM-only radiation refresh cadence. Default is "
            f"{DEFAULT_RRTMGP_UPDATE_INTERVAL_STEPS} steps for RRTMGP "
            "and 1 for gray. Cached radiation tendency is applied between "
            "refreshes; plane/CRM coupling is unchanged."
        ),
    )
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
    parser.add_argument(
        "--prognostic-spectral-gwd-thermal-tendency",
        action="store_true",
        default=SCM_PROGNOSTIC_SPECTRAL_GWD_THERMAL_TENDENCY,
        help=(
            "Allow the unvalidated prognostic-spectral GWD KE-to-thermal "
            "tendency in SCM RCE. Default false keeps the SCM comparison "
            "momentum-only for this opt-in GWD scheme."
        ),
    )
    args = parser.parse_args(argv)
    if args.scm_microphysics_substeps < 1:
        raise SystemExit("--scm-microphysics-substeps must be a positive integer")
    if args.scm_convection_substeps < 1:
        raise SystemExit("--scm-convection-substeps must be a positive integer")
    if args.surface_wind_m_s < 0.0:
        raise SystemExit("--surface-wind-m-s must be non-negative")
    if args.quick:
        args.days = min(args.days, QUICK_DAYS)
        args.analysis_days = min(args.analysis_days, args.days)
        args.tune_evals = min(args.tune_evals, QUICK_TUNE_EVALS)
        args.radiation = "gray"
    if args.radiation_update_interval_steps is None:
        args.radiation_update_interval_steps = (
            DEFAULT_RRTMGP_UPDATE_INTERVAL_STEPS
            if args.radiation == "rrtmgp"
            else 1
        )
    if args.radiation_update_interval_steps < 1:
        raise SystemExit("--radiation-update-interval-steps must be positive")
    require_equilibrium = not args.quick
    require_realism = not args.quick
    args.outdir.mkdir(parents=True, exist_ok=True)

    ref = build_reference_profiles(
        args.reference_dir,
        args.last_reference_files,
        precip_analysis_days=args.analysis_days,
    )
    categories = _parse_categories(args.categories, args.include_convection)
    sweeps = _scheme_sweeps_for(args, categories)
    run_cache: dict[str, RunDiagnostics] = {}

    baseline_cfg = make_physics_config(
        radiation=args.radiation,
        radiation_update_interval_steps=args.radiation_update_interval_steps,
        prognostic_spectral_gwd_thermal_tendency=(
            args.prognostic_spectral_gwd_thermal_tendency
        ),
    )
    baseline = run_cached(
        run_cache,
        baseline_cfg,
        ref,
        label="baseline",
        days=args.days,
        dt=args.dt,
        analysis_days=args.analysis_days,
        require_equilibrium=require_equilibrium,
        require_realism=require_realism,
        equil_T_tol_K=args.equil_T_tol_K,
        equil_qv_tol=args.equil_qv_tol,
        equil_qcond_tol=args.equil_qcond_tol,
        scm_microphysics_substeps=args.scm_microphysics_substeps,
        scm_convection_substeps=args.scm_convection_substeps,
        surface_wind_m_s=args.surface_wind_m_s,
        coriolis_s_inv=args.coriolis_s_inv,
        large_scale_forcing=args.large_scale_forcing,
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
                radiation=args.radiation,
                radiation_update_interval_steps=args.radiation_update_interval_steps,
                turbulence=scheme_kwargs["turbulence"],
                microphysics=scheme_kwargs["microphysics"],
                gravity_wave_drag=scheme_kwargs["gravity_wave_drag"],
                convection=scheme_kwargs["convection"],
                prognostic_spectral_gwd_thermal_tendency=(
                    args.prognostic_spectral_gwd_thermal_tendency
                ),
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
                    require_realism=require_realism,
                    equil_T_tol_K=args.equil_T_tol_K,
                    equil_qv_tol=args.equil_qv_tol,
                    equil_qcond_tol=args.equil_qcond_tol,
                    scm_microphysics_substeps=args.scm_microphysics_substeps,
                    scm_convection_substeps=args.scm_convection_substeps,
                    surface_wind_m_s=args.surface_wind_m_s,
                    coriolis_s_inv=args.coriolis_s_inv,
                    large_scale_forcing=args.large_scale_forcing,
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
        _write_profile_csv(args.outdir / f"profiles_{category}.csv", ref, rows)
        _plot_ranking(args.outdir / f"ranking_{category}.png", category, rows)

    best_kwargs = dict(BASELINE_SCHEMES)
    best_kwargs.update(winners)
    best_cfg = make_physics_config(
        radiation=args.radiation,
        radiation_update_interval_steps=args.radiation_update_interval_steps,
        turbulence=best_kwargs["turbulence"],
        microphysics=best_kwargs["microphysics"],
        gravity_wave_drag=best_kwargs["gravity_wave_drag"],
        convection=best_kwargs["convection"],
        prognostic_spectral_gwd_thermal_tendency=(
            args.prognostic_spectral_gwd_thermal_tendency
        ),
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
        require_realism=require_realism,
        equil_T_tol_K=args.equil_T_tol_K,
        equil_qv_tol=args.equil_qv_tol,
        equil_qcond_tol=args.equil_qcond_tol,
        scm_microphysics_substeps=args.scm_microphysics_substeps,
        scm_convection_substeps=args.scm_convection_substeps,
        surface_wind_m_s=args.surface_wind_m_s,
        coriolis_s_inv=args.coriolis_s_inv,
        large_scale_forcing=args.large_scale_forcing,
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
                require_realism=require_realism,
                tune_evals=args.tune_evals,
                seed=args.tune_seed + len(tuned_records),
                equil_T_tol_K=args.equil_T_tol_K,
                equil_qv_tol=args.equil_qv_tol,
                equil_qcond_tol=args.equil_qcond_tol,
                scm_microphysics_substeps=args.scm_microphysics_substeps,
                scm_convection_substeps=args.scm_convection_substeps,
                surface_wind_m_s=args.surface_wind_m_s,
                coriolis_s_inv=args.coriolis_s_inv,
                large_scale_forcing=args.large_scale_forcing,
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
