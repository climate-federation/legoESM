#!/usr/bin/env python3
"""Approach-window runner and matched scorer for OVERFLOW-zps instability.

This is the committed instrument for the Lane-1 Rule-9 finding.  NEMO files are
the exact Nbb step-entry frame; legoESM file ``completed_XXXXXXXX.npz`` is the
prognostic state after that many completed steps, hence it matches NEMO
``kt=completed+1``.  All comparisons reuse the certified phase-3 halo stripping
and wet-face registry.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_overflow_zps_card
from legoesm.ocean.vertical import compute_layer_thickness


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ORACLE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/stability/overflow_nemo_kt2601_2878"
)
DEFAULT_CANDIDATE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/stability/overflow_legoesm_baseline"
)
CERTIFIED_ORACLE_KT1 = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/overflow_zps/"
    "oracle_step_entry_kt00000001.bin"
)
CAPTURE_START = 2600
CAPTURE_END = 2877
ROUND_PAD_ULPS = 64.0
GROSS_RELATIVE_EXCURSION = 1.0e-6


class ProbeError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ProbeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_phase3():
    path = REPO_ROOT / (
        "scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_phase3_trajectory_gate.py"
    )
    spec = importlib.util.spec_from_file_location("nemo_l1_phase3", path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PHASE3 = _load_phase3()
read_entry = _PHASE3.read_entry
expected_masks = _PHASE3.expected_masks
lego_fields = _PHASE3.lego_fields


def _state_arrays(state) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in lego_fields(state).items()}


def _budget(state, card) -> dict[str, float]:
    h = np.asarray(compute_layer_thickness(
        state.eta.data,
        state.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    ))
    active = np.asarray(card.recipe.z_coord.is_active, dtype=bool)
    wet = np.asarray(state.land_mask.data) > 0.5
    use = active & wet[..., None]
    area = np.asarray(card.recipe.grid.area_T)[..., None]
    volume = area * h * use
    return {
        "volume_m3": float(np.sum(volume, dtype=np.float64)),
        "heat_content_proxy_m3_C": float(
            np.sum(volume * np.asarray(state.T.data), dtype=np.float64)),
        "salt_content_proxy_m3_psu": float(
            np.sum(volume * np.asarray(state.S.data), dtype=np.float64)),
    }


def run_legoesm(output: Path, arm: str, end_step: int, capture_start: int) -> dict:
    require(
        arm in {
            "baseline",
            "no_tracer_vertical_transport",
            "no_primary_transport_average",
        },
        f"bad arm {arm}",
    )
    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(0 <= capture_start <= end_step, "invalid capture window")
    card = build_overflow_zps_card()
    hooks = _NEMOWSRK3TestHooks(
        disable_tracer_vertical_transport=(arm == "no_tracer_vertical_transport"),
        primary_transport_average=(arm != "no_primary_transport_average"),
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=hooks,
    )
    state = card.recipe.initial_state
    output.mkdir(parents=True, exist_ok=True)
    dtype_receipt = {
        "state": {name: str(value.dtype) for name, value in _state_arrays(state).items()},
        "geometry": {
            name: str(np.asarray(getattr(card.recipe.z_coord, name)).dtype)
            for name in ("t_depth_ref", "dz_ref", "z_full_ref", "z_half_ref", "h_partial")
        },
    }
    require(set(dtype_receipt["state"].values()) == {"float64"}, str(dtype_receipt))
    require(set(dtype_receipt["geometry"].values()) == {"float64"}, str(dtype_receipt))
    initial_budget = _budget(state, card)
    records = []
    first_nonfinite = None
    started = time.perf_counter()
    previous = _state_arrays(state)
    for completed in range(end_step + 1):
        fields = _state_arrays(state)
        finite = {name: bool(np.all(np.isfinite(value))) for name, value in fields.items()}
        if completed >= capture_start:
            arrays = {name: np.array(value, copy=True) for name, value in fields.items()}
            snap = output / f"completed_{completed:08d}.npz"
            np.savez_compressed(snap, **arrays)
            row = {
                "completed_step": completed,
                "physical_time_s": completed * card.dt_s,
                "finite": finite,
                "snapshot": snap.name,
                "snapshot_sha256": sha256(snap),
                "budgets": _budget(state, card) if all(finite.values()) else None,
                "fields": {},
            }
            for name, values in fields.items():
                mask = expected_masks(card)[name]
                use = np.asarray(mask, dtype=bool)
                if not use.any():
                    row["fields"][name] = {"status": "UNMEASURED_NO_ACTIVE_FACE"}
                    continue
                selected = values[use]
                delta = values - previous[name]
                delta_selected = np.abs(delta[use])
                if np.all(np.isfinite(selected)):
                    index_flat = int(np.argmax(delta_selected))
                    index = tuple(int(v) for v in np.argwhere(use)[index_flat])
                    row["fields"][name] = {
                        "min": float(np.min(selected)),
                        "max": float(np.max(selected)),
                        "max_abs": float(np.max(np.abs(selected))),
                        "max_abs_one_step_increment": float(np.max(delta_selected)),
                        "increment_argmax_index": list(index),
                    }
                else:
                    row["fields"][name] = {"status": "NONFINITE"}
            records.append(row)
        if not all(finite.values()):
            first_nonfinite = completed
            break
        if completed == end_step:
            break
        previous = {name: np.array(value, copy=True) for name, value in fields.items()}
        state = model.step(state, dt=card.dt_s)

    artifact = {
        "format": "nemo-testcase-l1-overflow-stability-run-v1",
        "git_sha": git_sha(),
        "case": "OVERFLOW-zps",
        "arm": arm,
        "reference": (
            "NEMO 5.0.2 executed configuration"
            if arm == "baseline"
            else "experimental harness ablation; no reference model"
        ),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "precision_policy": repr(get_policy()),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "dtype_receipt": dtype_receipt,
        "capture_completed_steps": (
            [capture_start, min(end_step, records[-1]["completed_step"])]
            if records
            else None
        ),
        "requested_end_step": end_step,
        "first_nonfinite_completed_step": first_nonfinite,
        "initial_budgets": initial_budget,
        "wall_time_s": time.perf_counter() - started,
        "records": records,
    }
    path = output / "run.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: artifact[key] for key in artifact if key != "records"}, indent=2))
    return artifact


def paired_step_scale(output: Path, completed_before: int) -> dict:
    """Measure the private tracer-vertical arm from the identical input state."""
    require(completed_before >= 0, "negative completed-before step")
    set_policy(PrecisionPolicy.fp64())
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = build_overflow_zps_card()
    baseline = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config
    )
    arm = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            disable_tracer_vertical_transport=True
        ),
    )
    state = card.recipe.initial_state
    for _ in range(completed_before):
        state = baseline.step(state, dt=card.dt_s)
    before = _state_arrays(state)
    baseline_after = _state_arrays(baseline.step(state, dt=card.dt_s))
    arm_after = _state_arrays(arm.step(state, dt=card.dt_s))
    masks = expected_masks(card)
    rows = []
    for name in ("T", "S", "u", "v", "ssh"):
        use = np.asarray(masks[name], dtype=bool)
        if not use.any():
            rows.append({
                "field": name,
                "status": "UNMEASURED_NO_ACTIVE_FACE",
            })
            continue
        effect = np.abs(arm_after[name] - baseline_after[name])
        increment = np.abs(baseline_after[name] - before[name])
        masked_increment = np.where(use, increment, -np.inf)
        increment_index = tuple(
            int(value)
            for value in np.unravel_index(
                np.argmax(masked_increment), masked_increment.shape
            )
        )
        effect_max = float(np.max(effect[use]))
        increment_max = float(increment[increment_index])
        effect_at_increment = float(effect[increment_index])
        row = {
            "field": name,
            "same_input_state": True,
            "baseline_completed_before": completed_before,
            "compared_completed_after": completed_before + 1,
            "baseline_max_abs_one_step_increment": increment_max,
            "increment_argmax_index": list(increment_index),
            "arm_effect_at_increment_argmax": effect_at_increment,
            "arm_effect_linf": effect_max,
            "effect_at_increment_argmax_over_increment": (
                effect_at_increment / increment_max if increment_max > 0.0 else None
            ),
            "effect_linf_over_increment_linf": (
                effect_max / increment_max if increment_max > 0.0 else None
            ),
        }
        if name != "ssh":
            row["t_depth_m"] = float(
                np.asarray(card.recipe.z_coord.t_depth_ref)[increment_index[-1]]
            )
        row["x_km"] = float(
            increment_index[1] - (0.0 if name == "u" else 0.5)
        )
        rows.append(row)
    report = {
        "format": "nemo-testcase-l1-overflow-stability-paired-scale-v1",
        "git_sha": git_sha(),
        "case": "OVERFLOW-zps",
        "backend": jax.default_backend(),
        "precision_policy": repr(get_policy()),
        "arm": "disable_tracer_vertical_transport",
        "arm_reference": "experimental harness ablation; no reference model",
        "same_input_state": True,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return report


def summarize_run(root: Path) -> dict:
    """Turn raw per-step records into citable budget and growth diagnostics."""
    run_path = root / "run.json"
    require(run_path.is_file(), f"missing {run_path}")
    run = json.loads(run_path.read_text())
    records = run["records"]
    budget_rows = [row for row in records if row.get("budgets") is not None]
    budgets = {}
    for key, initial in run["initial_budgets"].items():
        values = [row["budgets"][key] for row in budget_rows]
        drift = [value - initial for value in values]
        budgets[key] = {
            "initial": initial,
            "last": values[-1] if values else None,
            "max_abs_drift": max((abs(value) for value in drift), default=None),
            "max_abs_relative_drift": (
                max((abs(value / initial) for value in drift), default=None)
                if initial != 0.0
                else None
            ),
        }
    fields = {}
    for name in ("T", "S", "u", "v", "ssh"):
        raw = []
        previous = None
        for row in records:
            value = row["fields"].get(name, {}).get("max_abs_one_step_increment")
            if value is None:
                continue
            ratio = value / previous if previous not in {None, 0.0} else None
            raw.append({
                "completed_step": row["completed_step"],
                "max_abs_one_step_increment": value,
                "successive_ratio": ratio,
                "argmax_index": row["fields"][name]["increment_argmax_index"],
            })
            previous = value
        fields[name] = {
            "raw": raw,
            "last_twelve": raw[-12:],
        }
    report = {
        "format": "nemo-testcase-l1-overflow-stability-run-summary-v1",
        "git_sha": git_sha(),
        "case": run["case"],
        "arm": run["arm"],
        "reference": run["reference"],
        "first_nonfinite_completed_step": run["first_nonfinite_completed_step"],
        "budget_closure": budgets,
        "increment_growth": fields,
    }
    output = root / "run_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "format": report["format"],
        "arm": report["arm"],
        "first_nonfinite_completed_step": report["first_nonfinite_completed_step"],
        "budget_closure": report["budget_closure"],
        "last_twelve": {
            name: value["last_twelve"] for name, value in fields.items()
        },
    }, indent=2))
    return report


def compare_arms(baseline_root: Path, arm_root: Path, output: Path) -> dict:
    """Apply the frozen Arm-A movement rule without post-hoc relabeling."""
    baseline = json.loads((baseline_root / "run.json").read_text())
    arm = json.loads((arm_root / "run.json").read_text())
    base_fail = baseline["first_nonfinite_completed_step"]
    arm_fail = arm["first_nonfinite_completed_step"]
    require(base_fail is not None and arm_fail is not None, "both arms must fail")
    movement = arm_fail - base_fail
    supports = arm_fail >= base_fail + 100 or arm_fail > 3200
    refutes_primary = 2870 <= arm_fail <= 2884 and abs(movement) < 0.1 * base_fail
    verdict = "CONFIRMED" if supports else "REFUTED_PRIMARY" if refutes_primary else "PLAUSIBLE"
    report = {
        "format": "nemo-testcase-l1-overflow-stability-arm-comparison-v1",
        "git_sha": git_sha(),
        "case": "OVERFLOW-zps",
        "arm": "disable_tracer_vertical_transport",
        "arm_reference": "experimental harness ablation; no reference model",
        "baseline_first_nonfinite_completed_step": base_fail,
        "arm_first_nonfinite_completed_step": arm_fail,
        "failure_step_movement": movement,
        "frozen_support_threshold": "moves >=100 steps later or beyond 3200",
        "frozen_refute_threshold": "fails in 2870..2884 with <10% movement and increment movement",
        "verdict": verdict,
        "interpretation": (
            f"The ablation destabilizes {abs(movement)} steps earlier. It shows the current "
            "explicit vertical tracer transport is necessary to this trajectory, "
            "but neither confirms nor refutes ownership by NEMO's complete "
            "adaptive-implicit RK3 package."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return report


def _load_candidate(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing {path}")
    with np.load(path) as data:
        required = {"T", "S", "u", "v", "ssh"}
        require(set(data.files) == required, f"{path}: inventory {data.files}")
        return {name: np.asarray(data[name]) for name in required}


def _argmax_row(name, oracle, candidate, mask, card) -> dict:
    use = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == use.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate dtype {candidate.dtype}")
    require(np.all(np.isfinite(oracle[use])), f"{name}: oracle nonfinite")
    require(np.all(np.isfinite(candidate[use])), f"{name}: candidate nonfinite")
    error = np.abs(candidate - oracle)
    masked = np.where(use, error, -np.inf)
    index = tuple(int(value) for value in np.unravel_index(np.argmax(masked), masked.shape))
    ovalues = oracle[use]
    cvalues = candidate[use]
    scale = max(float(np.max(np.abs(ovalues))), 1.0)
    pad = ROUND_PAD_ULPS * np.finfo(np.float64).eps * scale
    low_margin = float(np.min(cvalues) - np.min(ovalues))
    high_margin = float(np.max(cvalues) - np.max(ovalues))
    outside_padded = low_margin < -pad or high_margin > pad
    gross = max(-low_margin, high_margin, 0.0) / scale > GROSS_RELATIVE_EXCURSION
    result = {
        "field": name,
        "frame": (
            "T-centre instantaneous Nbb" if name in {"T", "S"}
            else f"instantaneous prognostic Nbb {name.upper()}-face"
            if name in {"u", "v"}
            else "T-centre instantaneous Nbb SSH"
        ),
        "reduction": "elementwise L-infinity on the certified common wet mask",
        "normalized_linf": float(error[index] / scale),
        "absolute_linf": float(error[index]),
        "argmax_index": list(index),
        "oracle_at_argmax": float(oracle[index]),
        "candidate_at_argmax": float(candidate[index]),
        "oracle_min": float(np.min(ovalues)),
        "oracle_max": float(np.max(ovalues)),
        "candidate_min": float(np.min(cvalues)),
        "candidate_max": float(np.max(cvalues)),
        "low_envelope_margin": low_margin,
        "high_envelope_margin": high_margin,
        "roundoff_pad": pad,
        "outside_roundoff_padded_oracle_range": bool(outside_padded),
        "gross_relative_excursion": bool(gross),
        "n": int(use.sum()),
    }
    if name != "ssh":
        level = index[-1]
        result["t_depth_m"] = float(np.asarray(card.recipe.z_coord.t_depth_ref)[level])
    if name in {"T", "S", "ssh"}:
        result["x_km"] = float(index[1] - 0.5)
    else:
        result["x_km"] = float(index[1] - (0.0 if name == "u" else 0.5))
    return result


def _certified_kt1_control(card, masks) -> dict:
    """Prove this probe still reproduces the certified initial comparator frame."""
    oracle = read_entry(CERTIFIED_ORACLE_KT1, "OVERFLOW-zps")
    candidate = _state_arrays(card.recipe.initial_state)
    rows = []
    for name in ("T", "S", "u", "ssh"):
        reference = np.asarray(oracle[name])
        if name != "ssh":
            reference = reference[..., :card.recipe.z_coord.n_levels]
        use = np.asarray(masks[name], dtype=bool)
        equal = bool(np.array_equal(reference[use], candidate[name][use]))
        rows.append({
            "field": name,
            "exact": equal,
            "n": int(use.sum()),
            "reference_sha256": sha256(CERTIFIED_ORACLE_KT1),
        })
        require(equal, f"certified kt=1 exact control failed for {name}")
    return {
        "status": "VERIFIED",
        "oracle": str(CERTIFIED_ORACLE_KT1),
        "rows": rows,
    }


def score(
    oracle_root: Path,
    candidate_root: Path,
    *,
    start_completed: int = CAPTURE_START,
    end_completed: int = CAPTURE_END,
    plant: str | None = None,
) -> dict:
    require(0 <= start_completed <= end_completed, "invalid score window")
    card = build_overflow_zps_card()
    masks = expected_masks(card)
    kt1_control = _certified_kt1_control(card, masks)
    rows = []
    first_outside = None
    first_gross = None
    first_nonfinite = None
    previous_errors = None
    for completed in range(start_completed, end_completed + 1):
        kt = completed + 1
        oracle_path = oracle_root / f"oracle_step_entry_kt{kt:08d}.bin"
        candidate_path = candidate_root / f"completed_{completed:08d}.npz"
        oracle = read_entry(oracle_path, "OVERFLOW-zps")
        require(oracle["step"] == kt, f"{oracle_path}: step mismatch")
        candidate = _load_candidate(candidate_path)
        if plant == "shift_step" and completed == start_completed:
            require(False, "planted step-number mismatch")
        if plant in {"hot", "nan"} and completed == start_completed:
            candidate["T"] = candidate["T"].copy()
            first = tuple(np.argwhere(masks["T"])[0])
            candidate["T"][first] = 50.0 if plant == "hot" else np.nan
        field_rows = []
        for name in ("T", "S", "u", "v", "ssh"):
            reference = np.asarray(oracle[name])
            if name != "ssh":
                reference = reference[..., :card.recipe.z_coord.n_levels]
            if name == "v" and not np.asarray(masks[name]).any():
                field_rows.append({
                    "field": name,
                    "status": "UNMEASURED_NO_ACTIVE_FACE",
                    "reason": "three-row closed tank has no active meridional face",
                })
                continue
            use = np.asarray(masks[name], dtype=bool)
            if not np.all(np.isfinite(candidate[name][use])):
                field_rows.append({
                    "field": name,
                    "status": "NONFINITE",
                    "frame": (
                        "T-centre instantaneous Nbb" if name in {"T", "S"}
                        else f"instantaneous prognostic Nbb {name.upper()}-face"
                        if name in {"u", "v"}
                        else "T-centre instantaneous Nbb SSH"
                    ),
                    "reduction": "finiteness on the certified common wet mask",
                    "n_nonfinite": int(np.count_nonzero(~np.isfinite(candidate[name][use]))),
                })
                if first_nonfinite is None:
                    first_nonfinite = {
                        "kt": kt,
                        "completed_step": completed,
                        "field": name,
                    }
                continue
            field_rows.append(
                _argmax_row(name, reference, candidate[name], masks[name], card)
            )
        gross = [row["field"] for row in field_rows if row.get("gross_relative_excursion")]
        if gross and first_gross is None:
            first_gross = {"kt": kt, "completed_step": completed, "fields": gross}
        outside = [
            row["field"] for row in field_rows
            if row.get("outside_roundoff_padded_oracle_range")
        ]
        if outside and first_outside is None:
            first_outside = {"kt": kt, "completed_step": completed, "fields": outside}
        current_errors = {
            row["field"]: row["normalized_linf"]
            for row in field_rows if "normalized_linf" in row
        }
        growth = None
        if previous_errors is not None:
            growth = {
                name: (current_errors[name] / previous_errors[name]
                       if previous_errors[name] > 0.0 else None)
                for name in current_errors
            }
        rows.append({
            "kt": kt,
            "completed_step": completed,
            "fields": field_rows,
            "successive_error_ratios": growth,
        })
        previous_errors = current_errors
    report = {
        "format": "nemo-testcase-l1-overflow-stability-score-v1",
        "git_sha": git_sha(),
        "case": "OVERFLOW-zps",
        "oracle_frame": "NEMO instantaneous Nbb step-entry state",
        "candidate_frame": "legoESM instantaneous prognostic state after kt-1 completed steps",
        "matched_kt": [start_completed + 1, end_completed + 1],
        "roundoff_pad_ulps": ROUND_PAD_ULPS,
        "gross_relative_excursion": GROSS_RELATIVE_EXCURSION,
        "certified_kt1_exact_control": kt1_control,
        "first_outside_roundoff_padded_oracle_range": first_outside,
        "first_gross_excursion": first_gross,
        "first_nonfinite": first_nonfinite,
        "status": "DEBT" if first_gross is not None or first_nonfinite is not None else "MEASURED",
        "rows": rows,
    }
    output = candidate_root / "matched_score.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "rows"}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run-legoesm")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_CANDIDATE)
    run_parser.add_argument(
        "--arm",
        choices=(
            "baseline",
            "no_tracer_vertical_transport",
            "no_primary_transport_average",
        ),
        default="baseline",
    )
    run_parser.add_argument("--end-step", type=int, default=CAPTURE_END)
    run_parser.add_argument("--capture-start", type=int, default=CAPTURE_START)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE)
    score_parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE)
    score_parser.add_argument("--start-completed", type=int, default=CAPTURE_START)
    score_parser.add_argument("--end-completed", type=int, default=CAPTURE_END)
    score_parser.add_argument("--plant", choices=("hot", "nan", "shift_step"))
    scale_parser = sub.add_parser("paired-step-scale")
    scale_parser.add_argument("--completed-before", type=int, default=2875)
    scale_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CANDIDATE / "paired_step_scale.json",
    )
    summary_parser = sub.add_parser("summarize-run")
    summary_parser.add_argument("--root", type=Path, default=DEFAULT_CANDIDATE)
    compare_parser = sub.add_parser("compare-arms")
    compare_parser.add_argument("--baseline-root", type=Path, default=DEFAULT_CANDIDATE)
    compare_parser.add_argument("--arm-root", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run-legoesm":
        run_legoesm(args.output, args.arm, args.end_step, args.capture_start)
    elif args.command == "score":
        report = score(
            args.oracle_root,
            args.candidate_root,
            start_completed=args.start_completed,
            end_completed=args.end_completed,
            plant=args.plant,
        )
        return 1 if report["status"] == "DEBT" else 0
    elif args.command == "paired-step-scale":
        paired_step_scale(args.output, args.completed_before)
    elif args.command == "summarize-run":
        summarize_run(args.root)
    else:
        compare_arms(args.baseline_root, args.arm_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
