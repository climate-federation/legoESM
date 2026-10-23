#!/usr/bin/env python3
"""Run and score the preregistered GYRE rounding-order closing check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
RUN_DIR = REPO / "scripts/run"
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from nemo_testcase_l2_gyre_hires_divergence import (  # noqa: E402
    divergence_curve,
    fit_growth_laws,
    normalized_rms_and_correlation,
    wet_span_bins,
)
from nemo_testcase_l2_gyre_hires_round17_vorticity import (  # noqa: E402
    COMMON_RESTART,
    COMMON_RESTART_SHA256,
    common_card,
    common_state,
    sha256,
)
from nemo_testcase_l2_gyre_phase3_gate import (  # noqa: E402
    expected_masks,
    require,
)
from run_gyre_hires import (  # noqa: E402
    FIELD_NAMES,
    _mean_accumulators,
    _state_arrays,
    build_scanned_day_mean_stepper,
    state_bitwise_identity,
)

PREREGISTRATION = REPO / (
    "docs/ocean/fidelity/testcases/"
    "nemo_testcases_l2_gyre_hires_round24_preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "3a9cc855a1f90fb7ee8a7f13a3844d07f431f1dc7a13eeece67f84a2139ab95b"
)
PREREGISTRATION_COMMIT = "3117150a54f2e4abf29b18f7d29d5b8557ca56f8"
ROUND25_PREREGISTRATION = REPO / (
    "docs/ocean/fidelity/testcases/"
    "nemo_testcases_l2_gyre_hires_round25_preregistration.md"
)
ROUND25_PREREGISTRATION_SHA256 = (
    "8d555a6a9c1bdf2bedd75775fe85b9ce75eaef6424560b46c569fba42d7f1624"
)
ROUND25_PREREGISTRATION_COMMIT = (
    "17957ee9252e9fb6eeb1223c66d86cf2de28ee28"
)
ROUND11_ARRAYS = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/"
    "round11/divergence_regional_v3/gyre_hires_divergence_arrays.npz"
)
ROUND11_ARRAYS_SHA256 = (
    "d96f9cf31ec96d598d05418ce1835dbedc64b2e49c1f4be171c44913e1eb48b6"
)
CHECKPOINT_DAYS = (30, 60, 90, 120, 180)
TOTAL_DAYS = 180
FIRST_STEP = 4321
STEPS_PER_DAY = 24
REGISTERED_SEED = {
    "u": {"max_abs": 1.852884572118782e-21, "unequal_active_words": 3789},
    "v": {"max_abs": 3.3881317890172014e-21, "unequal_active_words": 3261},
}
FIELDS = {
    "ssh": ("ssh_m_daily_mean", "T"),
    "sst": ("surface_temperature_c_daily_mean", "T"),
    "surface_u": ("surface_u_m_s_daily_mean", "u"),
    "surface_v": ("surface_v_m_s_daily_mean", "v"),
    "depth_mean_u": ("barotropic_u_m_s_daily_mean", "u"),
    "depth_mean_v": ("barotropic_v_m_s_daily_mean", "v"),
}


def _write_json(path: Path, payload: dict) -> None:
    def numpy_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable")

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
            default=numpy_scalar,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_registration(producer_commit: str) -> None:
    require(sha256(PREREGISTRATION) == PREREGISTRATION_SHA256,
            "round-24 preregistration moved")
    require(len(producer_commit) == 40, "producer commit is not a full SHA")
    ancestry = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
         PREREGISTRATION_COMMIT, producer_commit],
        check=False, capture_output=True,
    )
    require(ancestry.returncode == 0,
            "producer does not descend from the round-24 preregistration")


def validate_round25_registration(producer_commit: str) -> None:
    require(sha256(ROUND25_PREREGISTRATION)
            == ROUND25_PREREGISTRATION_SHA256,
            "round-25 preregistration moved")
    require(len(producer_commit) == 40, "producer commit is not a full SHA")
    ancestry = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor",
         ROUND25_PREREGISTRATION_COMMIT, producer_commit],
        check=False, capture_output=True,
    )
    require(ancestry.returncode == 0,
            "producer does not descend from the round-25 preregistration")


def registered_verdict(rows: dict[str, dict[str, float]]) -> str:
    """Apply only the terminal rules committed before acquisition."""
    require(set(rows) == set(FIELDS), "closing-check field inventory differs")
    if all(row["self"] >= row["legoesm_nemo"] for row in rows.values()):
        return "CONFIRM_FULL"
    if all(row["self"] <= row["nemo_twin"] for row in rows.values()):
        return "REFUTE_ROUNDING_EXPLANATION"
    return "PARTIAL_OR_INCONCLUSIVE"


def seed_statistics(
    delta_u: np.ndarray, delta_v: np.ndarray, masks: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    """Measure the exact full-face seed on the registered active support."""
    native = {
        "u": np.asarray(delta_u)[:, 1:, :],
        "v": np.asarray(delta_v)[1:, :, :],
    }
    result = {}
    for face in ("u", "v"):
        active = np.asarray(masks[face], dtype=bool)
        values = native[face][active]
        result[face] = {
            "max_abs": float(np.max(np.abs(values))),
            "unequal_active_words": int(np.count_nonzero(values != 0.0)),
        }
    return result


def _formed_smagorinsky_pair(card, state):
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_vertex_mask,
        nemo_smagorinsky_bilaplacian_coefficients_cgrid,
    )

    cell3 = (
        np.asarray(card.recipe.z_coord.is_active, dtype=np.float64)
        * np.asarray(state.land_mask.data)[..., None]
    )
    vertex = jax.vmap(
        lambda level: compute_vertex_mask(level, grid=card.recipe.grid),
        in_axes=-1, out_axes=-1,
    )(jnp.asarray(cell3))
    return nemo_smagorinsky_bilaplacian_coefficients_cgrid(
        state.u.data, state.v.data, card.recipe.grid,
        card.recipe.model_config.lateral_viscosity.C_smag, card.dt_s,
        mask=state.land_mask.data, cell_mask=jnp.asarray(cell3),
        u_mask=state.u_mask.data, v_mask=state.v_mask.data,
        vertex_mask=vertex,
    )


def make_seed(output: Path) -> dict:
    """Materialize the measured CPU/libm composition residue once."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_forcings
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", "seed construction requires CPU")
    stamp = worktree_stamp()
    require(stamp["clean"], "seed construction requires a clean commit")
    validate_registration(stamp["commit"])
    require(sha256(COMMON_RESTART) == COMMON_RESTART_SHA256,
            "common restart moved")
    require(not output.exists(), "seed output already exists")
    output.mkdir(parents=True)
    card = common_card()
    state = common_state(card)
    freshwater, surface = gyre_surface_forcings(card, state, FIRST_STEP)

    def trace(override=None):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_live_stage_operands=True,
                nemo_stage_rhs_accumulation_order_arm=(
                    "stage1-source-order", None),
                smagorinsky_coefficient_override=override,
            ),
        )
        return jax.device_get(model.step(
            state, card.dt_s, freshwater=freshwater,
            surface_forcing=surface))

    live = trace()
    pair = tuple(jax.device_get(value) for value in _formed_smagorinsky_pair(
        card, state))
    materialized = trace(pair)
    live_ops = live.operator_operands[0]
    materialized_ops = materialized.operator_operands[0]
    delta_u = np.asarray(materialized_ops["after_ldf_u"].data) - np.asarray(
        live_ops["after_ldf_u"].data)
    delta_v = np.asarray(materialized_ops["after_ldf_v"].data) - np.asarray(
        live_ops["after_ldf_v"].data)
    stats = seed_statistics(delta_u, delta_v, expected_masks(card))
    require(stats == REGISTERED_SEED,
            f"formed seed differs from round 22: {stats!r}")
    seed_path = output / "stage1_ldf_composition_seed.npz"
    with seed_path.open("wb") as stream:
        np.savez_compressed(stream, delta_u=delta_u, delta_v=delta_v)
    manifest = {
        "schema": "gyre-r4-smag-round24-seed-v1",
        "producer": stamp,
        "policy": "fp64-libm-certified-cpu",
        "common_restart_sha256": COMMON_RESTART_SHA256,
        "step": FIRST_STEP,
        "definition": "materialized post-LDF accumulator minus live composed",
        "statistics": stats,
        "seed_file": seed_path.name,
        "seed_sha256": sha256(seed_path),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "instrument_sha256": sha256(Path(__file__)),
    }
    _write_json(output / "seed_manifest.json", manifest)
    return manifest


def _load_seed(root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest = json.loads((root / "seed_manifest.json").read_text())
    path = root / manifest["seed_file"]
    require(sha256(path) == manifest["seed_sha256"], "seed artifact moved")
    require(manifest["statistics"] == REGISTERED_SEED,
            "seed manifest statistics differ")
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["delta_u"]), np.asarray(data["delta_v"]), manifest


def _build_arm_steppers(production, intervention, card):
    return (
        build_scanned_day_mean_stepper(
            production, card, STEPS_PER_DAY - 1),
        build_scanned_day_mean_stepper(
            intervention, card, STEPS_PER_DAY - 1),
        build_scanned_day_mean_stepper(production, card, STEPS_PER_DAY),
        build_scanned_day_mean_stepper(intervention, card, STEPS_PER_DAY),
    )


def run_experiment(
    seed_root: Path, output: Path, physical_gpu: int, *, recurring: bool = False,
) -> dict:
    """Run the registered one-time or every-step residue experiment."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import gyre_surface_forcings
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    require(type(physical_gpu) is int and physical_gpu >= 0,
            "physical GPU must be explicit")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu),
            "CUDA_VISIBLE_DEVICES differs from --physical-gpu")
    set_policy(PrecisionPolicy.fp64(transcendentals="native"))
    require(jax.default_backend() == "gpu", "closing run requires GPU")
    stamp = worktree_stamp()
    require(stamp["clean"], "closing run requires a clean commit")
    registration = (
        validate_round25_registration if recurring else validate_registration)
    registration(stamp["commit"])
    require(not output.exists(), "run output already exists")
    delta_u, delta_v, seed_manifest = _load_seed(seed_root)
    card = common_card()
    require(seed_statistics(delta_u, delta_v, expected_masks(card))
            == REGISTERED_SEED, "loaded seed differs from registered pattern")
    state = common_state(card)
    freshwater, surface = gyre_surface_forcings(card, state, FIRST_STEP)
    production = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    identity = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            stage1_momentum_rhs_transform=lambda u, v: (u, v)),
    )
    seeded = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            stage1_momentum_rhs_transform=lambda u, v: (
                u + jnp.asarray(delta_u), v + jnp.asarray(delta_v))),
    )
    ordinary_state = production.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    control_state = identity.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    perturbed_state = seeded.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    jax.block_until_ready(perturbed_state.eta.data)
    passivity = state_bitwise_identity(ordinary_state, control_state)
    movement = state_bitwise_identity(control_state, perturbed_state)
    require(passivity["status"] == "PASS",
            "identity transform moved the ordinary step")
    require(movement["unequal_words"] > 0,
            "registered residue seed moved no state word")

    preregistration_sha256 = (
        ROUND25_PREREGISTRATION_SHA256 if recurring
        else PREREGISTRATION_SHA256)
    wrapper = REPO / "scripts/run" / (
        "run_gyre_hires_round25_recurring_rounding_gpu.sh" if recurring
        else "run_gyre_hires_round24_rounding_gpu.sh")
    output.mkdir(parents=True)
    manifest = {
        "schema": (
            "gyre-r4-smag-round25-recurring-rounding-run-v1" if recurring
            else "gyre-r4-smag-round24-rounding-run-v1"),
        "status": "pending",
        "producer": stamp,
        "backend": jax.default_backend(),
        "physical_gpu_index": physical_gpu,
        "policy": "fp64-native-gpu-non-certified",
        "common_restart_sha256": COMMON_RESTART_SHA256,
        "relative_days": TOTAL_DAYS,
        "absolute_steps": [FIRST_STEP, FIRST_STEP + TOTAL_DAYS * STEPS_PER_DAY - 1],
        "seed_manifest_sha256": sha256(seed_root / "seed_manifest.json"),
        "seed_sha256": seed_manifest["seed_sha256"],
        "identity_transform": passivity,
        "seeded_step_movement": movement,
        "seed_application": {
            "cadence": "every_step" if recurring else "first_step_only",
            "count": TOTAL_DAYS * STEPS_PER_DAY if recurring else 1,
            "first_step": FIRST_STEP,
            "last_step": (
                FIRST_STEP + TOTAL_DAYS * STEPS_PER_DAY - 1
                if recurring else FIRST_STEP),
        },
        "fields": list(FIELD_NAMES),
        "preregistration_sha256": preregistration_sha256,
        "instrument_sha256": sha256(Path(__file__)),
        "wrapper_sha256": sha256(wrapper),
        "daily_sha256": {},
    }
    _write_json(output / "manifest.pending.json", manifest)
    intervention = seeded if recurring else production
    control_tail, perturbed_tail, control_day, perturbed_day = (
        _build_arm_steppers(production, intervention, card))
    control_means0 = _mean_accumulators(control_state, card)
    perturbed_means0 = _mean_accumulators(perturbed_state, card)
    hashes = {}
    for day in range(1, TOTAL_DAYS + 1):
        if day == 1:
            control_state, control_means = control_tail(
                control_state, FIRST_STEP + 1,
                initial_means=control_means0, initial_count=1)
            perturbed_state, perturbed_means = perturbed_tail(
                perturbed_state, FIRST_STEP + 1,
                initial_means=perturbed_means0, initial_count=1)
        else:
            first = FIRST_STEP + (day - 1) * STEPS_PER_DAY
            control_state, control_means = control_day(control_state, first)
            perturbed_state, perturbed_means = perturbed_day(
                perturbed_state, first)
        jax.block_until_ready(perturbed_state.eta.data)
        arrays = {"day": np.int64(day)}
        for arm, arm_state, means in (
            ("control", control_state, control_means),
            ("perturbed", perturbed_state, perturbed_means),
        ):
            for name, snapshot, mean in zip(
                    FIELD_NAMES, _state_arrays(arm_state), means, strict=True):
                arrays[f"{arm}.{name}"] = np.asarray(snapshot, dtype=np.float64)
                arrays[f"{arm}.{name}_daily_mean"] = np.asarray(
                    mean, dtype=np.float64)
        name = f"day{day:04d}.npz"
        temporary = output / f".{name}.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(output / name)
        hashes[name] = sha256(output / name)
        manifest["daily_sha256"] = hashes
        manifest["completed_days"] = day
        _write_json(output / "manifest.pending.json", manifest)
    manifest["status"] = "complete"
    _write_json(output / "manifest.pending.json", manifest)
    (output / "manifest.pending.json").replace(output / "manifest.json")
    return manifest


def _load_run(root: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    schema = manifest.get("schema")
    require(schema in {
        "gyre-r4-smag-round24-rounding-run-v1",
        "gyre-r4-smag-round25-recurring-rounding-run-v1",
    }, f"unregistered rounding-run schema: {schema!r}")
    require(manifest["status"] == "complete"
            and manifest["completed_days"] == TOTAL_DAYS,
            "rounding run is incomplete")
    if schema == "gyre-r4-smag-round25-recurring-rounding-run-v1":
        validate_round25_registration(manifest["producer"]["commit"])
        require(manifest["preregistration_sha256"]
                == ROUND25_PREREGISTRATION_SHA256,
                "round-25 run names the wrong preregistration")
        require(manifest["seed_application"] == {
            "cadence": "every_step",
            "count": TOTAL_DAYS * STEPS_PER_DAY,
            "first_step": FIRST_STEP,
            "last_step": FIRST_STEP + TOTAL_DAYS * STEPS_PER_DAY - 1,
        }, "round-25 run does not apply the residue at every step")
    else:
        validate_registration(manifest["producer"]["commit"])
    series = {
        arm: {field: [] for field in FIELDS}
        for arm in ("control", "perturbed")
    }
    for day in range(1, TOTAL_DAYS + 1):
        name = f"day{day:04d}.npz"
        path = root / name
        require(sha256(path) == manifest["daily_sha256"].get(name),
                f"{name}: digest differs")
        with np.load(path, allow_pickle=False) as data:
            require(int(data["day"]) == day, f"{name}: day differs")
            for arm in series:
                for field, (stored, _mask) in FIELDS.items():
                    values = np.asarray(data[f"{arm}.{stored}"], dtype=np.float64)
                    require(np.all(np.isfinite(values)),
                            f"{name}: non-finite {arm}.{field}")
                    series[arm][field].append(values)
    return {
        arm: {field: np.stack(values) for field, values in fields.items()}
        for arm, fields in series.items()
    }, manifest


def _regional_self_sse(
    control: np.ndarray, perturbed: np.ndarray, mask: np.ndarray,
) -> list[dict]:
    bins = wet_span_bins(mask)
    names = ("western_boundary", "western_interior", "central_interior",
             "eastern_interior", "eastern_boundary")
    error = (perturbed[29:180] - control[29:180]) ** 2
    rows = []
    total = float(np.sum(error[:, mask], dtype=np.float64))
    require(total > 0.0, "rounding experiment has zero regional error")
    for index, name in enumerate(names):
        active = bins == index
        value = float(np.sum(error[:, active], dtype=np.float64))
        rows.append({"region": name, "sse": value, "share": value / total})
    return rows


def score(run_root: Path, output: Path, round11_arrays: Path) -> dict:
    """Score self-divergence against the registered NEMO envelope."""
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(not output.exists(), "score output already exists")
    require(sha256(round11_arrays) == ROUND11_ARRAYS_SHA256,
            "round-11 comparison arrays moved")
    series, manifest = _load_run(run_root)
    recurring = (
        manifest["schema"]
        == "gyre-r4-smag-round25-recurring-rounding-run-v1")
    report_stem = (
        "round25_recurring_rounding" if recurring else "round24_rounding")
    card = common_card()
    masks0 = expected_masks(card)
    masks = {"T": masks0["T"][..., 0],
             "u": masks0["u"][..., 0], "v": masks0["v"][..., 0]}
    curves = {
        field: divergence_curve(
            series["control"][field], series["perturbed"][field],
            masks[mask_name])
        for field, (_stored, mask_name) in FIELDS.items()
    }
    controls = {}
    for field, (_stored, mask_name) in FIELDS.items():
        reference = series["control"][field][0]
        exact = normalized_rms_and_correlation(reference, reference,
                                               masks[mask_name])
        planted = np.array(reference, copy=True)
        index = tuple(np.argwhere(masks[mask_name])[0])
        planted[index] = np.nextafter(planted[index], np.inf)
        one_ulp = normalized_rms_and_correlation(
            reference, planted, masks[mask_name])
        require(exact["normalized_rms_difference"] == 0.0
                and one_ulp["normalized_rms_difference"] > 0.0,
                f"{field}: exact/one-ULP controls did not separate")
        controls[field] = {"exact": exact, "one_ulp": one_ulp,
                           "one_ulp_index": list(index)}
    with np.load(round11_arrays, allow_pickle=False) as comparison:
        day180 = {
            field: {
                "self": float(curves[field]["normalized_rms_difference"][179]),
                "legoesm_nemo": float(comparison[
                    f"legoesm_candidate.{field}.normalized_rms_difference"][179]),
                "nemo_twin": float(comparison[
                    f"nemo_scalar.{field}.normalized_rms_difference"][179]),
            }
            for field in FIELDS
        }
        checkpoints = []
        for day in CHECKPOINT_DAYS:
            checkpoints.append({
                "day": day,
                **{
                    f"{field}.{name}": value
                    for field in FIELDS
                    for name, value in (
                        ("self", float(curves[field][
                            "normalized_rms_difference"][day - 1])),
                        ("legoesm_nemo", float(comparison[
                            f"legoesm_candidate.{field}.normalized_rms_difference"
                        ][day - 1])),
                        ("nemo_twin", float(comparison[
                            f"nemo_scalar.{field}.normalized_rms_difference"
                        ][day - 1])),
                    )
                },
            })
    for row in day180.values():
        row["self_over_nemo_twin"] = (
            row["self"] / row["nemo_twin"] if row["nemo_twin"] else None)
        row["legoesm_nemo_over_nemo_twin"] = (
            row["legoesm_nemo"] / row["nemo_twin"]
            if row["nemo_twin"] else None)
    growth = {}
    regional = {}
    days = np.arange(30.0, 181.0)
    for field, (_stored, mask_name) in FIELDS.items():
        absolute = curves[field]["absolute_rms_difference"][29:180]
        positive = absolute > 0.0
        require(np.count_nonzero(positive) >= 3,
                f"{field}: too few positive values for growth fit")
        growth[field] = fit_growth_laws(days[positive], absolute[positive])
        regional[field] = _regional_self_sse(
            series["control"][field], series["perturbed"][field],
            masks[mask_name])
    output.mkdir(parents=True)
    arrays_path = output / f"{report_stem}_curves.npz"
    np.savez_compressed(arrays_path, **{
        f"self.{field}.{metric}": values
        for field, metrics in curves.items()
        for metric, values in metrics.items()
    })
    report = {
        "schema": (
            "gyre-r4-smag-round25-recurring-rounding-score-v1" if recurring
            else "gyre-r4-smag-round24-rounding-score-v1"),
        "instrument": worktree_stamp(),
        "run_manifest_sha256": sha256(run_root / "manifest.json"),
        "round11_arrays": {"path": str(round11_arrays),
                           "sha256": ROUND11_ARRAYS_SHA256},
        "definition": (
            "RMS(perturbed-control) / RMS(control spatial anomaly); daily means"
        ),
        "checkpoints": checkpoints,
        "day180": day180,
        "verdict": registered_verdict(day180),
        "growth_days_30_180": growth,
        "regional_days_30_180": regional,
        "controls": controls,
        "run": manifest,
        "artifacts": {arrays_path.name: sha256(arrays_path)},
    }
    _write_json(output / f"{report_stem}_report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seed_parser = sub.add_parser("make-seed")
    seed_parser.add_argument("--output", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--seed-root", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--physical-gpu", type=int, required=True)
    recurring_parser = sub.add_parser("run-recurring")
    recurring_parser.add_argument("--seed-root", type=Path, required=True)
    recurring_parser.add_argument("--output", type=Path, required=True)
    recurring_parser.add_argument("--physical-gpu", type=int, required=True)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--run-root", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--round11-arrays", type=Path,
                              default=ROUND11_ARRAYS)
    args = parser.parse_args(argv)
    try:
        if args.command == "make-seed":
            result = make_seed(args.output)
        elif args.command in ("run", "run-recurring"):
            result = run_experiment(
                args.seed_root, args.output, args.physical_gpu,
                recurring=args.command == "run-recurring")
        else:
            result = score(args.run_root, args.output, args.round11_arrays)
    except Exception as exc:  # noqa: BLE001 - fail-closed instrument boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"command": args.command,
                      "status": result.get("status", result.get("verdict", "PASS"))},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
