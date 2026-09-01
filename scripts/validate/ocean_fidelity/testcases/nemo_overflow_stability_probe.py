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


def run_legoesm(output: Path, arm: str, end_step: int) -> dict:
    require(arm in {"baseline", "no_tracer_vertical_transport"}, f"bad arm {arm}")
    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(end_step >= CAPTURE_START, "end step precedes capture window")
    card = build_overflow_zps_card()
    hooks = _NEMOWSRK3TestHooks(
        disable_tracer_vertical_transport=(arm == "no_tracer_vertical_transport")
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
        if completed >= CAPTURE_START:
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
        "capture_completed_steps": [CAPTURE_START, min(end_step, records[-1]["completed_step"])],
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


def score(oracle_root: Path, candidate_root: Path, *, plant: str | None = None) -> dict:
    card = build_overflow_zps_card()
    masks = expected_masks(card)
    rows = []
    first_outside = None
    previous_errors = None
    for completed in range(CAPTURE_START, CAPTURE_END + 1):
        kt = completed + 1
        oracle_path = oracle_root / f"oracle_step_entry_kt{kt:08d}.bin"
        candidate_path = candidate_root / f"completed_{completed:08d}.npz"
        oracle = read_entry(oracle_path, "OVERFLOW-zps")
        require(oracle["step"] == kt, f"{oracle_path}: step mismatch")
        candidate = _load_candidate(candidate_path)
        if plant == "shift_step" and completed == CAPTURE_START:
            require(False, "planted step-number mismatch")
        if plant in {"hot", "nan"} and completed == CAPTURE_START:
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
            field_rows.append(_argmax_row(
                name, reference, candidate[name], masks[name], card
            ))
        gross = [row["field"] for row in field_rows if row.get("gross_relative_excursion")]
        require(not gross, f"kt={kt}: gross excursions {gross}")
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
        "matched_kt": [CAPTURE_START + 1, CAPTURE_END + 1],
        "roundoff_pad_ulps": ROUND_PAD_ULPS,
        "gross_relative_excursion": GROSS_RELATIVE_EXCURSION,
        "first_outside_roundoff_padded_oracle_range": first_outside,
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
        "--arm", choices=("baseline", "no_tracer_vertical_transport"), default="baseline"
    )
    run_parser.add_argument("--end-step", type=int, default=CAPTURE_END)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE)
    score_parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE)
    score_parser.add_argument("--plant", choices=("hot", "nan", "shift_step"))
    args = parser.parse_args()
    if args.command == "run-legoesm":
        run_legoesm(args.output, args.arm, args.end_step)
    else:
        score(args.oracle_root, args.candidate_root, plant=args.plant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
