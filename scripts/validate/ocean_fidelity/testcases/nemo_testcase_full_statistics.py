#!/usr/bin/env python3
"""Run and score the preregistered full-duration NEMO testcase extension.

The run mode writes only legoESM milestone states.  The score mode consumes
those states plus the existing NEMO FCT2 and preregistered NEMO FCT4 artifacts.
All oracle entry loading, halo stripping, and C-grid staggering reuse the
certified phase-3 gate implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.rpe import pack_sorted_rpe
from legoesm.ocean.vertical import compute_layer_thickness
from legoesm.ocean.fidelity.provenance import (
    allow_dirty_stamps,
    scoped_allow_dirty,
    worktree_stamp,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l1")
FULL_ROOT = ARTIFACT_ROOT / "full_statistical"
PREREG_SHA = "27e569b20932e44a0fc1d1f4812c7fe8b4fc79b2"
FP32_DISCRIMINATOR_PREREG_SHA = "b3a813c8a2e3"
FP32_DISCRIMINATOR_PATH = (
    ARTIFACT_ROOT / "round3_aimp/fp32_temperature_trace_3060.json"
)
FP32_DISCRIMINATOR_SHA256 = (
    "a093cf1127da5a4572010d8f49fa7fda752b99b9d847ba1a47e0c58a79907dfa"
)
CASES = {
    "LOCK_EXCHANGE-zco": {
        "slug": "lock_exchange_zco",
        "n_steps": 61200,
        "dt_s": 1.0,
        "mid_kt": 30600,
        "baseline": ARTIFACT_ROOT / "lock_exchange_zco",
        "alternative": FULL_ROOT / "nemo_fct4/lock_exchange_zco",
        "restart": "LOCK_EXCHANGE_OMIP_L1_ZCO_00061200_restart.nc",
        "entry_hashes": {
            1: "c9f23d441865c566e3edf0301aca4f6e440259ade1722b725a0aa0fd4c2839e1",
            30600: "76bb241ddcc1f1a7148eceb2a0807e7bd6afb376b4f1139d100ee8234709a342",
        },
        "restart_hash": "15a7883e5e27df2fec36716c29375ec947892a842ee901e5526e10525db5c7a6",
        "namelist_hash": "ae34648ecdf44893e8543f0516511d239ce59161fa5b2de9924f0169d8185dd4",
        "binary_hash": "d297e236afd0097fc64533f4182fada9d58cc0458c359af06e08106fb796a259",
        "phase3_gate": ARTIFACT_ROOT / "phase3/lock_trajectory_gate_kt60.json",
        "phase3_gate_hash": "bcde1f59e92cd132c5f1622af75156a7901c7dd06be198564ca04fa66f0709ac",
        "temperature_range": (5.0, 30.0),
    },
    "OVERFLOW-zps": {
        "slug": "overflow_zps",
        "n_steps": 6120,
        "dt_s": 10.0,
        "mid_kt": 3060,
        "baseline": ARTIFACT_ROOT / "overflow_zps",
        "alternative": FULL_ROOT / "nemo_fct4/overflow_zps",
        "restart": "OVERFLOW_OMIP_L1_ZPS_00006120_restart.nc",
        "entry_hashes": {
            1: "cf0183e563aba8b8848bc5dea470c9e50aab2d987ff5da86241e8f82494d5c66",
            3060: "ae3e27c43bb649a40db50de24357421d55a6378dc6658c3a5b9e5f64a8382d4c",
        },
        "restart_hash": "dab392f2f058b44e8c10c600a41c9be73ba37656e2af478193a3f6f27bd67160",
        "namelist_hash": "ec1eac4a45fb8c07a0facce5e4eefb6510d8e3f1e364f5c5597e60ae83ccc53e",
        "binary_hash": "eb4acf9651b887a3da8834281112d472692caa0bbadcb0d69779e91dee92e6cb",
        "phase3_gate": ARTIFACT_ROOT / "face_thickness/overflow_trajectory_gate_kt60.json",
        "phase3_gate_hash": "0b46df0aa025c10ae3c7c7b371e2fe9c91cd4812bcaddbe4d4a9b50fc5d65582",
        "temperature_range": (10.0, 20.0),
    },
}
REGISTERED_METRICS = {
    "LOCK_EXCHANGE-zco": {
        "front_position_km",
        "front_speed_anchor_ratio",
        "rpe_relative",
        "temperature_variance_fraction",
        "temperature_linf",
        "instantaneous_u_linf",
    },
    "OVERFLOW-zps": {
        "plume_descent_m",
        "plume_front_km",
        "final_temperature_histogram_tv",
        "final_water_mass_census",
        "temperature_linf",
        "instantaneous_u_linf",
    },
}


class StatisticalError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise StatisticalError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(*, allow_dirty: bool = False) -> str:
    """Exact legoESM producer revision (fails closed on tracked dirt)."""
    from legoesm.ocean.fidelity.provenance import git_sha as _stamp

    try:
        return _stamp(allow_dirty=allow_dirty)
    except RuntimeError as error:
        raise StatisticalError(f"cannot stamp legoESM git SHA: {error}") from error


def _load_module(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TRAJECTORY_GATE = _load_module(
    "nemo_l1_phase3_gate",
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_phase3_trajectory_gate.py",
)
_EOS_GATE = _load_module(
    "nemo_l1_eos_gate",
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_phase3_eos_gate.py",
)
read_entry = _TRAJECTORY_GATE.read_entry
expected_masks = _TRAJECTORY_GATE.expected_masks
nemo_literal_density = _EOS_GATE.nemo_literal_density
parse_teos10_density_coefficients = _EOS_GATE.parse_teos10_density_coefficients


def sample_completed_steps(case: str) -> tuple[int, int, int]:
    spec = CASES[case]
    return (0, int(spec["mid_kt"]) - 1, int(spec["n_steps"]))


def _state_arrays(state) -> dict[str, np.ndarray]:
    return {
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        "u": np.asarray(state.u.data),
        "v": np.asarray(state.v.data),
        "ssh": np.asarray(state.eta.data),
    }


def run_legoesm(
    case: str,
    precision: str,
    output_dir: Path,
    stamped_sha: str,
    *,
    check_finite_every_step: bool = False,
) -> dict:
    policy = PrecisionPolicy.fp64() if precision == "fp64" else PrecisionPolicy.fp32()
    set_policy(policy)
    require(get_policy() == policy, f"failed to set {precision} policy")
    require(
        bool(jax.config.jax_enable_x64) == (precision == "fp64"),
        f"{precision} requires jax_enable_x64={precision == 'fp64'}",
    )
    card = build_nemo_testcase_card(case)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    completed_samples = sample_completed_steps(case)
    captured: dict[int, dict[str, np.ndarray]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry_dtypes = {
        name: str(np.asarray(getattr(card.recipe.z_coord, name)).dtype)
        for name in ("t_depth_ref", "dz_ref", "z_full_ref", "z_half_ref", "h_partial")
    }
    initial_dtypes = {name: str(values.dtype) for name, values in _state_arrays(state).items()}
    expected_dtype = "float64" if precision == "fp64" else "float32"
    require(
        all(value == expected_dtype for value in geometry_dtypes.values()),
        f"{case} {precision}: geometry dtypes {geometry_dtypes}",
    )
    require(
        all(value == expected_dtype for value in initial_dtypes.values()),
        f"{case} {precision}: state dtypes {initial_dtypes}",
    )
    print(
        json.dumps(
            {
                "event": "dtype_receipt",
                "case": case,
                "precision": precision,
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "state": initial_dtypes,
                "geometry": geometry_dtypes,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    started = time.perf_counter()
    next_progress = 10
    for completed in range(int(CASES[case]["n_steps"]) + 1):
        if completed in completed_samples:
            captured[completed] = {
                name: np.array(values, copy=True) for name, values in _state_arrays(state).items()
            }
        if completed == int(CASES[case]["n_steps"]):
            break
        state = model.step(state, dt=card.dt_s)
        if check_finite_every_step:
            finite = {
                name: bool(np.all(np.isfinite(values)))
                for name, values in _state_arrays(state).items()
            }
            if not all(finite.values()):
                failure = {
                    "worktree": worktree_stamp(),
                    "format": "nemo-testcase-l1-full-failure-v1",
                    "preregistration_commit": PREREG_SHA,
                    "git_sha": stamped_sha,
                    "case": case,
                    "precision": precision,
                    "backend": jax.default_backend(),
                    "devices": [str(device) for device in jax.devices()],
                    "first_nonfinite_completed_step": completed + 1,
                    "physical_time_s": (completed + 1) * float(CASES[case]["dt_s"]),
                    "fields_finite": finite,
                    "wall_time_s": time.perf_counter() - started,
                    "state_dtypes": initial_dtypes,
                    "geometry_dtypes": geometry_dtypes,
                    "diagnostic_check_every_step": True,
                }
                failure_path = output_dir / "failure.json"
                failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
                print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
                raise StatisticalError(
                    f"{case} {precision}: first nonfinite completed step {completed + 1}"
                )
        percentage = int(100 * (completed + 1) / int(CASES[case]["n_steps"]))
        if percentage >= next_progress:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "case": case,
                        "precision": precision,
                        "completed_steps": completed + 1,
                        "percent": percentage,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            next_progress += 10
    wall_s = time.perf_counter() - started
    require(set(captured) == set(completed_samples), "milestone capture incomplete")

    arrays = {}
    for completed, fields in captured.items():
        for name, values in fields.items():
            require(np.all(np.isfinite(values)), f"nonfinite {case} {completed} {name}")
            arrays[f"step_{completed}_{name}"] = values
    state_path = output_dir / "states.npz"
    np.savez_compressed(state_path, **arrays)
    metadata = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l1-full-state-v1",
        "preregistration_commit": PREREG_SHA,
        "git_sha": stamped_sha,
        "case": case,
        "precision": precision,
        "precision_policy": repr(policy),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "completed_steps": list(completed_samples),
        "physical_times_s": [
            float(value) * float(CASES[case]["dt_s"]) for value in completed_samples
        ],
        "wall_time_s": wall_s,
        "state_dtypes": initial_dtypes,
        "geometry_dtypes": geometry_dtypes,
        "states_sha256": sha256(state_path),
        "diagnostic_check_every_step": check_finite_every_step,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return metadata


def classify_fp32_temperature_trace(
    excess: np.ndarray,
    *,
    all_finite: bool,
    field_scale: float = 20.0,
    eps: float = float(np.finfo(np.float32).eps),
) -> dict:
    """Apply the committed gradual-accumulation versus jump corridors."""
    values = np.asarray(excess, dtype=np.float64)
    require(values.ndim == 1 and values.size >= 2, "fp32 trace needs at least two steps")
    require(np.all(values >= 0.0), "fp32 trace excess must be nonnegative")
    n_steps = values.size - 1
    peak = float(np.max(values))
    increments = np.maximum(np.diff(values), 0.0)
    jump_index = int(np.argmax(increments)) + 1
    jump = float(increments[jump_index - 1])
    jump_fraction = jump / peak if peak > 0.0 else 0.0
    ulp_per_step = peak / (field_scale * eps * n_steps)
    if not all_finite:
        classification = "NONFINITE"
    elif peak <= 0.0:
        classification = "UNMEASURED"
    elif jump_fraction >= 0.5:
        classification = "LIMITER_EVENT"
    elif jump_fraction <= 0.05 and ulp_per_step <= 1.0:
        classification = "PRECISION_ACCUMULATION"
    else:
        classification = "UNMEASURED"
    return {
        "classification": classification,
        "n_steps": n_steps,
        "peak_excess_K": peak,
        "largest_positive_jump_K": jump,
        "largest_positive_jump_completed_step": jump_index,
        "largest_jump_fraction_of_peak": jump_fraction,
        "peak_ulp_per_step": ulp_per_step,
        "corridors": {
            "accumulation_max_jump_fraction": 0.05,
            "accumulation_max_ulp_per_step": 1.0,
            "limiter_event_min_jump_fraction": 0.5,
        },
    }


def run_fp32_temperature_trace(output: Path, stamped_sha: str) -> dict:
    """Run the preregistered 3,060-step OVERFLOW fp32 range discriminator."""
    case = "OVERFLOW-zps"
    policy = PrecisionPolicy.fp32()
    set_policy(policy)
    require(get_policy() == policy, "failed to set fp32 policy")
    require(not bool(jax.config.jax_enable_x64), "fp32 trace requires jax_enable_x64=False")
    card = build_nemo_testcase_card(case)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    wet = expected_masks(card)["T"]
    n_steps = int(CASES[case]["mid_kt"])
    low, high = CASES[case]["temperature_range"]
    trace = []
    all_finite = True
    started = time.perf_counter()
    for completed in range(n_steps + 1):
        arrays = _state_arrays(state)
        finite = {name: bool(np.all(np.isfinite(value))) for name, value in arrays.items()}
        all_finite = all_finite and all(finite.values())
        temperature = arrays["T"][wet]
        raw_min = float(np.min(temperature))
        raw_max = float(np.max(temperature))
        excess = max(low - raw_min, raw_max - high, 0.0)
        trace.append(
            {
                "completed_step": completed,
                "raw_min_C": raw_min,
                "raw_max_C": raw_max,
                "excess_K": excess,
                "fields_finite": finite,
            }
        )
        if completed < n_steps:
            state = model.step(state, dt=card.dt_s)
    classification = classify_fp32_temperature_trace(
        np.asarray([row["excess_K"] for row in trace]), all_finite=all_finite
    )
    gross_threshold = 1.0e-6 * max(abs(low), abs(high), 1.0)
    crossings = [row["completed_step"] for row in trace if row["excess_K"] > gross_threshold]
    report = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l1-fp32-temperature-trace-v1",
        "preregistration_commit": FP32_DISCRIMINATOR_PREREG_SHA,
        "git_sha": stamped_sha,
        "case": case,
        "precision": "fp32",
        "precision_policy": repr(policy),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "temperature_dtype": str(np.asarray(state.T.data).dtype),
        "geometry_dtypes": {
            name: str(np.asarray(getattr(card.recipe.z_coord, name)).dtype)
            for name in ("t_depth_ref", "dz_ref", "z_full_ref", "z_half_ref", "h_partial")
        },
        "all_finite": all_finite,
        "gross_guard_relative": 1.0e-6,
        "first_gross_excursion_completed_step": crossings[0] if crossings else None,
        "classification": classification,
        "reviewed_prior_scaling_context": {
            "ratio_of_ratios": 0.477,
            "estimated_ulp_per_step": 0.19,
            "status": "PLAUSIBLE-strong before this discriminating arm",
        },
        "wall_time_s": time.perf_counter() - started,
        "trace": trace,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_namelist_values(path: Path) -> dict[str, str]:
    section = None
    result = {}
    for raw in path.read_text().splitlines():
        line = raw.split("!", 1)[0].strip()
        if line.startswith("&"):
            section = line[1:].split()[0].lower()
        elif line == "/":
            section = None
        elif section and "=" in line:
            lhs, rhs = line.split("=", 1)
            lhs = lhs.strip().lower()
            if re.fullmatch(r"[a-z][a-z0-9_%]*(?:\([^)]*\))?", lhs):
                result[f"{section}.{lhs}"] = rhs.strip().rstrip(",").strip()
    return result


def verify_alternative_namelist(case: str) -> dict:
    spec = CASES[case]
    baseline = Path(spec["baseline"]) / "namelist_cfg"
    alternative = Path(spec["alternative"]) / "namelist_cfg"
    base_values = parse_namelist_values(baseline)
    alt_values = parse_namelist_values(alternative)
    require(set(base_values) == set(alt_values), f"{case}: namelist inventory changed")
    changed = {
        key: {"baseline": base_values[key], "alternative": alt_values[key]}
        for key in base_values
        if base_values[key] != alt_values[key]
    }
    require(
        set(changed) == {"namrun.cn_exp", "namtra_adv.nn_fct_h", "namtra_adv.nn_fct_v"},
        f"{case}: alternative semantic diff is {changed}",
    )
    require(
        changed["namtra_adv.nn_fct_h"] == {"baseline": "2", "alternative": "4"}
        and changed["namtra_adv.nn_fct_v"] == {"baseline": "2", "alternative": "4"},
        f"{case}: FCT diff is not 2->4",
    )
    return {
        "baseline_sha256": sha256(baseline),
        "alternative_sha256": sha256(alternative),
        "changed_assignments": changed,
    }


def _read_restart(path: Path, case: str) -> dict[str, np.ndarray]:
    card = build_nemo_testcase_card(case)
    nlev = card.recipe.z_coord.n_levels
    with netCDF4.Dataset(path) as dataset:

        def yz(name: str) -> np.ndarray:
            return np.asarray(dataset.variables[name][0], dtype=np.float64).transpose(1, 2, 0)[
                ..., :nlev
            ]

        result = {
            "T": yz("tn"),
            "S": yz("sn"),
            "u": yz("un"),
            "v": yz("vn"),
            "ssh": np.asarray(dataset.variables["sshn"][0], dtype=np.float64),
        }
        kt = int(round(float(np.asarray(dataset.variables["kt"][:]))))
    require(kt == int(CASES[case]["n_steps"]), f"{path}: restart kt={kt}")
    return result


def load_nemo_states(case: str, root: Path, *, baseline: bool) -> dict[int, dict]:
    spec = CASES[case]
    if baseline:
        require(sha256(root / "namelist_cfg") == spec["namelist_hash"], "baseline namelist hash")
        require(sha256(root / "nemo.exe") == spec["binary_hash"], "baseline binary hash")
    states = {}
    for kt, expected_hash in spec["entry_hashes"].items():
        path = root / f"oracle_step_entry_kt{kt:08d}.bin"
        require(path.is_file(), f"missing {path}")
        if baseline:
            require(sha256(path) == expected_hash, f"{path}: hash changed")
        entry = read_entry(path, case)
        nlev = build_nemo_testcase_card(case).recipe.z_coord.n_levels
        states[(kt - 1) * int(spec["dt_s"])] = {
            name: np.asarray(entry[name])[..., :nlev] if name != "ssh" else np.asarray(entry[name])
            for name in ("T", "S", "u", "v", "ssh")
        }
    if baseline:
        restart = root / str(spec["restart"])
        require(sha256(restart) == spec["restart_hash"], f"{restart}: hash changed")
    else:
        restarts = sorted(root.glob("*restart.nc"))
        require(len(restarts) == 1, f"{root}: expected one alternative restart")
        restart = restarts[0]
    states[int(spec["n_steps"] * spec["dt_s"])] = _read_restart(restart, case)
    require(len(states) == 3, f"{case}: expected three NEMO states")
    return dict(sorted(states.items()))


def load_legoesm_states(case: str, precision: str, root: Path) -> tuple[dict, dict]:
    spec = CASES[case]
    run_root = root / spec["slug"] / precision
    metadata = json.loads((run_root / "metadata.json").read_text())
    state_path = run_root / "states.npz"
    require(metadata["states_sha256"] == sha256(state_path), f"{run_root}: state hash")
    require(metadata["case"] == case and metadata["precision"] == precision, "metadata mismatch")
    expected_dtype = "float64" if precision == "fp64" else "float32"
    require(
        set(metadata["state_dtypes"].values()) == {expected_dtype}
        and set(metadata["geometry_dtypes"].values()) == {expected_dtype},
        f"{run_root}: dtype receipt mismatch",
    )
    states = {}
    with np.load(state_path) as archive:
        for completed in sample_completed_steps(case):
            states[int(completed * spec["dt_s"])] = {
                name: np.asarray(archive[f"step_{completed}_{name}"])
                for name in ("T", "S", "u", "v", "ssh")
            }
    return dict(sorted(states.items())), metadata


def load_legoesm_failure(case: str, precision: str, root: Path) -> dict | None:
    path = root / CASES[case]["slug"] / precision / "failure.json"
    if not path.is_file():
        return None
    failure = json.loads(path.read_text())
    require(failure["format"] == "nemo-testcase-l1-full-failure-v1", "failure format")
    require(failure["case"] == case and failure["precision"] == precision, "failure identity")
    require(failure["preregistration_commit"] == PREREG_SHA, "failure preregistration")
    require(failure["diagnostic_check_every_step"] is True, "failure diagnostic cadence")
    require(
        0 < int(failure["first_nonfinite_completed_step"]) < int(CASES[case]["n_steps"]),
        "failure step outside incomplete-run interval",
    )
    require(not all(failure["fields_finite"].values()), "failure artifact is vacuous")
    return {**failure, "artifact": str(path), "artifact_sha256": sha256(path)}


def score_incomplete_case(case: str, lego_root: Path, failures: dict[str, dict]) -> dict:
    """Classify every registered metric OUTSIDE when the candidate is non-finite."""
    require(set(failures) == {"fp64", "fp32"}, f"{case}: incomplete failure inventory")
    rows = [
        {
            "name": name,
            "verdict": "OUTSIDE",
            "candidate_distance": None,
            "precision_floor": None,
            "scheme_spread": None,
            "units": "not computed",
            "predicate": "missing/non-finite registered full-duration input",
            "reduction": "not run; fail-closed before metric reduction",
            "reason": (
                f"fp64 first non-finite completed step "
                f"{failures['fp64']['first_nonfinite_completed_step']} of "
                f"{CASES[case]['n_steps']}"
            ),
        }
        for name in sorted(REGISTERED_METRICS[case])
    ]
    require({row["name"] for row in rows} == REGISTERED_METRICS[case], "failure metric coverage")
    return {
        "case": case,
        "preregistration_commit": PREREG_SHA,
        "status": "OUTSIDE",
        "verdict_counts": {
            "INDISTINGUISHABLE-AT-FLOOR": 0,
            "WITHIN-SCHEME-SPREAD": 0,
            "OUTSIDE": len(rows),
        },
        "alternative_namelist_diff": verify_alternative_namelist(case),
        "state_frame": "full-duration comparison unavailable: candidate became non-finite",
        "mask_and_reduction": "not applied to missing full-duration candidate states",
        "legoesm_runs": {precision: failures[precision] for precision in ("fp64", "fp32")},
        "metrics": {},
        "deterministic_bridge": {
            "status": "TRUNCATED_BEFORE_REGISTERED_MIDPOINT",
            "phase3_gate": str(CASES[case]["phase3_gate"]),
            "phase3_gate_sha256": sha256(Path(CASES[case]["phase3_gate"])),
        },
        "rows": rows,
        "controls": {
            "failure_inventory": sorted(failures),
            "all_registered_metrics_forced_outside": True,
        },
        "lego_root": str(lego_root),
    }


def mapped_fields(fields: dict, source: str) -> dict[str, np.ndarray]:
    if source.startswith("L"):
        return {
            **fields,
            "u": np.asarray(fields["u"])[:, 1:, :],
            "v": np.asarray(fields["v"])[1:, :, :],
        }
    return fields


def _geometry(case: str, ssh: np.ndarray) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    card = build_nemo_testcase_card(case)
    masks = expected_masks(card)
    h = np.asarray(
        compute_layer_thickness(
            jnp.asarray(ssh, dtype=jnp.float64),
            jnp.asarray(card.recipe.initial_state.H_bathy.data, dtype=jnp.float64),
            card.recipe.z_coord,
        ),
        dtype=np.float64,
    )
    area = np.asarray(card.recipe.grid.area, dtype=np.float64)
    volume = area[..., None] * h * masks["T"]
    centres = np.cumsum(h, axis=-1) - 0.5 * h
    return card, masks["T"], volume, centres


def _section_row(mask: np.ndarray) -> int:
    counts = np.count_nonzero(mask, axis=(1, 2))
    row = int(np.argmax(counts))
    require(np.count_nonzero(counts == counts[row]) == 1, f"ambiguous section rows {counts}")
    return row


def _x_centres_km(card, row: int) -> np.ndarray:
    dx = np.asarray(card.recipe.grid.dx_T, dtype=np.float64)[row]
    edges = np.concatenate(([0.0], np.cumsum(dx))) / 1000.0
    return 0.5 * (edges[:-1] + edges[1:])


def _bottom_values(temperature: np.ndarray, mask: np.ndarray, row: int) -> np.ndarray:
    values = np.full(mask.shape[1], np.nan, dtype=np.float64)
    for ix in range(mask.shape[1]):
        active = np.flatnonzero(mask[row, ix])
        if active.size:
            values[ix] = temperature[row, ix, active[-1]]
    return values


def ascending_crossings(x: np.ndarray, values: np.ndarray, threshold: float) -> np.ndarray:
    crossings = []
    delta = np.asarray(values, dtype=np.float64) - threshold
    for index in range(delta.size - 1):
        left, right = delta[index : index + 2]
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        if left <= 0.0 < right:
            weight = -left / (right - left) if right != left else 0.5
            crossings.append(x[index] + weight * (x[index + 1] - x[index]))
    return np.asarray(crossings, dtype=np.float64)


def rightmost_ascending_crossing(
    x: np.ndarray, values: np.ndarray, threshold: float, *, label: str,
) -> tuple[float, int]:
    """Apply the preregistered connected-front reduction."""
    crossings = ascending_crossings(x, values, threshold)
    require(crossings.size >= 1, f"{label}: missing ascending front")
    return float(crossings[-1]), int(crossings.size)


def precision_floor_guard(
    case: str,
    source: str,
    dtype: np.dtype,
    *,
    discriminator_path: Path = FP32_DISCRIMINATOR_PATH,
    expected_sha256: str = FP32_DISCRIMINATOR_SHA256,
) -> tuple[float, dict]:
    """Return the preregistered guard only for the validated OVERFLOW L32 arm."""
    if case != "OVERFLOW-zps" or source != "L32":
        return 1.0e-6, {"basis": "fixed gross-excursion guard", "relative_guard": 1.0e-6}
    require(np.dtype(dtype) == np.dtype(np.float32), "OVERFLOW L32 floor arm must be fp32")
    require(sha256(discriminator_path) == expected_sha256, "fp32 discriminator hash mismatch")
    discriminator = json.loads(discriminator_path.read_text())
    classification = discriminator["classification"]
    require(
        discriminator["preregistration_commit"] == FP32_DISCRIMINATOR_PREREG_SHA,
        "fp32 discriminator preregistration mismatch",
    )
    require(
        discriminator["case"] == case
        and discriminator["precision"] == "fp32"
        and discriminator["all_finite"] is True
        and classification["classification"] == "PRECISION_ACCUMULATION"
        and classification["n_steps"] == int(CASES[case]["mid_kt"]),
        "fp32 discriminator did not confirm precision accumulation",
    )
    relative_guard = max(
        1.0e-6, int(CASES[case]["n_steps"]) * float(np.finfo(dtype).eps)
    )
    return relative_guard, {
        "basis": "preregistered linear precision-accumulation floor-arm guard",
        "formula": "max(1e-6, N_steps * eps(dtype))",
        "relative_guard": relative_guard,
        "discriminator": str(discriminator_path),
        "discriminator_sha256": expected_sha256,
        "discriminator_preregistration_commit": FP32_DISCRIMINATOR_PREREG_SHA,
        "classification": classification,
    }


def _snap_temperature(
    values: np.ndarray,
    case: str,
    dtype: np.dtype,
    *,
    gross_guard_relative: float = 1.0e-6,
    guard_evidence: dict | None = None,
) -> tuple[np.ndarray, dict]:
    low, high = CASES[case]["temperature_range"]
    eps = np.finfo(dtype).eps
    scale = max(abs(low), abs(high), 1.0)
    floor = float(math.sqrt(int(CASES[case]["n_steps"])) * eps * scale)
    raw_min = float(np.min(values))
    raw_max = float(np.max(values))
    excess = max(low - raw_min, raw_max - high, 0.0)
    excess_relative = excess / scale
    require(
        excess_relative <= gross_guard_relative,
        f"{case}: gross T excursion raw=[{raw_min:.17g},{raw_max:.17g}] "
        f"excess={excess:.17g} relative={excess_relative:.17g}",
    )
    return np.clip(values, low, high), {
        "raw_min_C": raw_min,
        "raw_max_C": raw_max,
        "excess_K": excess,
        "excess_relative": excess_relative,
        "endpoint_floor_K": floor,
        "roundoff_status": "AT-BAR" if excess <= floor else "UNMEASURED",
        "gross_guard_relative": gross_guard_relative,
        "gross_guard_evidence": guard_evidence
        or {"basis": "fixed gross-excursion guard", "relative_guard": gross_guard_relative},
    }


def _surface_density(temperature, salinity, coefficients) -> np.ndarray:
    return nemo_literal_density(
        temperature,
        salinity,
        np.zeros_like(np.asarray(temperature)),
        coefficients,
    )


def _rpe(fields: dict, case: str, coefficients) -> float:
    card, mask, volume, _ = _geometry(case, fields["ssh"])
    density = _surface_density(fields["T"], fields["S"], coefficients)
    wet = mask & (volume > 0.0)
    total_area = float(np.sum(np.asarray(card.recipe.grid.area)[mask[..., 0]]))
    return pack_sorted_rpe(
        density[wet], volume[wet], total_area, g_val=float(card.recipe.model_config.g)
    )


def arm_metrics(case: str, states: dict[int, dict], source: str, coefficients) -> dict:
    card = build_nemo_testcase_card(case)
    masks = expected_masks(card)
    times = np.asarray(sorted(states), dtype=np.float64)
    mapped = {time_s: mapped_fields(states[int(time_s)], source) for time_s in times}
    for time_s, fields in mapped.items():
        for name in ("T", "S", "u", "v", "ssh"):
            mask = masks[name]
            require(fields[name].shape == mask.shape, f"{case} {source} {time_s} {name} shape")
            require(
                np.all(np.isfinite(fields[name][mask])),
                f"{case} {source} {time_s} {name} nonfinite",
            )

    result = {"times_s": times.tolist(), "source": source}
    row = _section_row(masks["T"])
    x_km = _x_centres_km(card, row)
    if case == "OVERFLOW-zps":
        descent = []
        fronts = []
        crossing_counts = []
        for time_s in times:
            fields = mapped[int(time_s)]
            _, active, _, centres = _geometry(case, fields["ssh"])
            cold = active & (fields["T"] <= 15.0)
            require(bool(np.any(cold)), f"{source}: no cold plume at t={time_s}")
            descent.append(float(np.max(centres[cold])))
            front, count = rightmost_ascending_crossing(
                x_km, _bottom_values(fields["T"], active, row), 15.0,
                label=f"{source} overflow t={time_s}")
            # Frozen preregistration, metric 2: the plume front is the
            # RIGHTMOST ascending crossing connected to the initial cold
            # reservoir.  Interior mixed lenses may add crossings behind it;
            # rejecting those made the scorer contradict its own registered
            # reducer as soon as the first complete candidate reached scoring.
            fronts.append(front)
            crossing_counts.append(count)
        final = mapped[int(times[-1])]
        _, active, volume, _ = _geometry(case, final["ssh"])
        bathy = np.asarray(card.recipe.initial_state.H_bathy.data)
        slope = (bathy > 500.0) & (bathy < 2000.0)
        select = active & slope[..., None]
        temperature_dtype = np.asarray(final["T"]).dtype
        gross_guard, guard_evidence = precision_floor_guard(case, source, temperature_dtype)
        final_values, range_receipt = _snap_temperature(
            np.asarray(final["T"])[select],
            case,
            temperature_dtype,
            gross_guard_relative=gross_guard,
            guard_evidence=guard_evidence,
        )
        weights = volume[select]
        bins = np.linspace(10.0, 20.0, 41)
        histogram = np.histogram(final_values, bins=bins, weights=weights)[0]
        histogram = histogram / np.sum(histogram)
        census = np.asarray(
            [
                np.sum(weights[(final_values >= 10.0) & (final_values < 12.0)]),
                np.sum(weights[(final_values >= 12.0) & (final_values < 18.0)]),
                np.sum(weights[(final_values >= 18.0) & (final_values <= 20.0)]),
            ],
            dtype=np.float64,
        )
        census = census / np.sum(census)
        result.update(
            plume_descent_m=descent,
            plume_front_km=fronts,
            plume_front_crossing_count=crossing_counts,
            final_temperature_histogram=histogram.tolist(),
            final_temperature_bins_C=bins.tolist(),
            final_water_mass_census=census.tolist(),
            final_water_mass_labels=["cold_10_12", "mixed_12_18", "ambient_18_20"],
            final_temperature_range_receipt=range_receipt,
        )
    else:
        fronts = []
        crossing_counts = []
        rpe = []
        variance = []
        for time_s in times:
            fields = mapped[int(time_s)]
            _, active, volume, _ = _geometry(case, fields["ssh"])
            front, count = rightmost_ascending_crossing(
                x_km, _bottom_values(fields["T"], active, row), 17.5,
                label=f"{source} lock t={time_s}")
            fronts.append(front)
            crossing_counts.append(count)
            rpe.append(_rpe(fields, case, coefficients))
            wet = active & (volume > 0.0)
            values = np.asarray(fields["T"], dtype=np.float64)[wet]
            weights = volume[wet]
            mean = float(np.sum(weights * values) / np.sum(weights))
            variance.append(float(np.sum(weights * (values - mean) ** 2) / np.sum(weights)))
        elapsed = times - times[0]
        displacement_m = (np.asarray(fronts) - fronts[0]) * 1000.0
        speed = float(np.sum(elapsed[1:] * displacement_m[1:]) / np.sum(elapsed[1:] ** 2))
        density_cold = float(_surface_density(np.asarray(5.0), np.asarray(35.0), coefficients))
        density_warm = float(_surface_density(np.asarray(30.0), np.asarray(35.0), coefficients))
        rho_ref = 0.5 * (density_cold + density_warm)
        g_prime = float(card.recipe.model_config.g) * (density_cold - density_warm) / rho_ref
        anchor = 0.5 * math.sqrt(g_prime * 20.0)
        result.update(
            front_position_km=fronts,
            front_crossing_count=crossing_counts,
            front_speed_m_s=speed,
            benjamin_anchor_m_s=anchor,
            front_speed_anchor_ratio=speed / anchor,
            g_prime_m_s2=g_prime,
            surface_density_cold_kg_m3=density_cold,
            surface_density_warm_kg_m3=density_warm,
            rpe_J=rpe,
            rpe_relative=((np.asarray(rpe) - rpe[0]) / abs(rpe[0])).tolist(),
            temperature_variance_K2=variance,
            temperature_variance_fraction=(np.asarray(variance) / variance[0]).tolist(),
        )
    return result


def verdict(candidate: float, floor: float, spread: float) -> str:
    require(
        all(np.isfinite(value) and value >= 0.0 for value in (candidate, floor, spread)),
        "invalid distances",
    )
    if candidate <= floor:
        return "INDISTINGUISHABLE-AT-FLOOR"
    if candidate <= spread:
        return "WITHIN-SCHEME-SPREAD"
    return "OUTSIDE"


def _curve_distance(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    require(left.shape == right.shape, "curve distance shape mismatch")
    return float(np.max(np.abs(left - right)))


def _histogram_tv(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    require(left.shape == right.shape, "histogram shape mismatch")
    return float(0.5 * np.sum(np.abs(left - right)))


def _field_linf(case: str, arms: dict[str, dict[int, dict]], field: str) -> dict[str, float]:
    card = build_nemo_testcase_card(case)
    mask = expected_masks(card)[field]
    times = sorted(arms["N2"])[1:]

    def distance(left_name: str, right_name: str) -> float:
        values = []
        for time_s in times:
            left = mapped_fields(arms[left_name][time_s], left_name)[field]
            right = mapped_fields(arms[right_name][time_s], right_name)[field]
            scale = max(
                float(np.max(np.abs(mapped_fields(arms["N2"][time_s], "N2")[field][mask]))), 1.0
            )
            values.append(float(np.max(np.abs(left[mask] - right[mask]))) / scale)
        return max(values)

    return {
        "candidate_distance": distance("L64", "N2"),
        "precision_floor": distance("L32", "L64"),
        "scheme_spread": distance("N4", "N2"),
    }


def deterministic_bridge(case: str, arms: dict[str, dict[int, dict]]) -> dict:
    """Join the certified before-entry kt1..60 series to full-run samples."""
    path = Path(CASES[case]["phase3_gate"])
    require(sha256(path) == CASES[case]["phase3_gate_hash"], f"{case}: phase3 gate hash")
    phase3 = json.loads(path.read_text())
    require(phase3["case"] == case and len(phase3["steps"]) == 60, "phase3 ladder mismatch")
    result = {
        "phase3_gate": str(path),
        "phase3_gate_sha256": sha256(path),
        "join": (
            "phase3 kt is Nbb/before at completed_step=kt-1; midpoint is the same "
            "before-entry frame; final point changes frame to tn/un restart after N_steps"
        ),
        "series": {},
    }
    card = build_nemo_testcase_card(case)
    masks = expected_masks(card)
    for field in ("T", "u"):
        completed_steps = []
        values = []
        for step in phase3["steps"]:
            rows = [row for row in step["rows"] if row["name"].rsplit(".", 1)[-1] == field]
            require(len(rows) == 1, f"{case} kt{step['kt']} {field}: row inventory")
            completed_steps.append(int(step["kt"]) - 1)
            values.append(float(rows[0]["normalized_max_abs"]))
        mask = masks[field]
        for time_s in sorted(arms["N2"]):
            completed = int(round(time_s / float(CASES[case]["dt_s"])))
            if completed <= completed_steps[-1]:
                continue
            oracle = mapped_fields(arms["N2"][time_s], "N2")[field]
            candidate = mapped_fields(arms["L64"][time_s], "L64")[field]
            scale = max(float(np.max(np.abs(oracle[mask]))), 1.0)
            completed_steps.append(completed)
            values.append(float(np.max(np.abs(candidate[mask] - oracle[mask]))) / scale)
        require(
            completed_steps[-2:] == [int(CASES[case]["mid_kt"]) - 1, int(CASES[case]["n_steps"])],
            f"{case} {field}: full-duration bridge samples",
        )
        result["series"][field] = {
            "completed_steps": completed_steps,
            "normalized_max_abs": values,
            "staggering": "T centre" if field == "T" else "instantaneous U face",
        }
    return result


def _row(
    name: str, candidate: float, floor: float, spread: float, units: str, reduction: str
) -> dict:
    return {
        "name": name,
        "verdict": verdict(candidate, floor, spread),
        "candidate_distance": candidate,
        "precision_floor": floor,
        "scheme_spread": spread,
        "units": units,
        "predicate": (
            "D<=F" if candidate <= floor else "F<D<=S" if candidate <= spread else "D>F and D>S"
        ),
        "reduction": reduction,
    }


def score_case(
    case: str,
    lego_root: Path,
    *,
    plant_state: bool = False,
    plant_census: bool = False,
    plant_unregistered: bool = False,
) -> dict:
    spec = CASES[case]
    failures = {
        precision: failure
        for precision in ("fp64", "fp32")
        if (failure := load_legoesm_failure(case, precision, lego_root)) is not None
    }
    if failures:
        require(
            not (plant_state or plant_census or plant_unregistered),
            "planted metric controls require complete states",
        )
        return score_incomplete_case(case, lego_root, failures)
    alternative_diff = verify_alternative_namelist(case)
    roots = {"N2": Path(spec["baseline"]), "N4": Path(spec["alternative"])}
    arms = {
        "N2": load_nemo_states(case, roots["N2"], baseline=True),
        "N4": load_nemo_states(case, roots["N4"], baseline=False),
    }
    arms["L64"], l64_metadata = load_legoesm_states(case, "fp64", lego_root)
    arms["L32"], l32_metadata = load_legoesm_states(case, "fp32", lego_root)
    require(
        set(map(tuple, (arm.keys() for arm in arms.values()))) == {tuple(arms["N2"].keys())},
        "arm time mismatch",
    )

    if plant_state:
        final_time = max(arms["L64"])
        mask = expected_masks(build_nemo_testcase_card(case))["T"]
        location = tuple(np.argwhere(mask)[0])
        for arm in ("L64", "L32"):
            arms[arm][final_time]["T"] = np.array(arms[arm][final_time]["T"], copy=True)
            arms[arm][final_time]["T"][location] += 10.0
    if plant_census and case == "OVERFLOW-zps":
        final_time = max(arms["L64"])
        card = build_nemo_testcase_card(case)
        mask = expected_masks(card)["T"]
        bathy = np.asarray(card.recipe.initial_state.H_bathy.data)
        select = mask & ((bathy > 500.0) & (bathy < 2000.0))[..., None]
        # Plant candidate and precision arm together.  Planting L64 alone also
        # enlarges |L32-L64| and can make the floor cancel the planted signal.
        for arm in ("L64", "L32"):
            state = arms[arm][final_time]
            state["T"] = np.where(select, 10.0, state["T"])

    coefficients = parse_teos10_density_coefficients()
    metrics = {}
    invalid_arms = {}
    for arm, states in arms.items():
        try:
            metrics[arm] = arm_metrics(case, states, arm, coefficients)
        except StatisticalError as error:
            invalid_arms[arm] = str(error)
    if invalid_arms:
        require(
            not (plant_state or plant_census or plant_unregistered),
            "planted metric controls require valid metric arms",
        )
        rows = [
            {
                "name": name,
                "verdict": "OUTSIDE",
                "candidate_distance": None,
                "precision_floor": None,
                "scheme_spread": None,
                "predicate": "FAILED_REGISTERED_ARM_CONTROL",
                "reason": "; ".join(
                    f"{arm}: {reason}" for arm, reason in sorted(invalid_arms.items())
                ),
            }
            for name in sorted(REGISTERED_METRICS[case])
        ]
        return {
            "case": case,
            "git_sha": git_sha(),
            "preregistration_commit": PREREG_SHA,
            "status": "OUTSIDE",
            "verdict_counts": {
                "INDISTINGUISHABLE-AT-FLOOR": 0,
                "WITHIN-SCHEME-SPREAD": 0,
                "OUTSIDE": len(rows),
            },
            "alternative_namelist_diff": alternative_diff,
            "invalid_metric_arms": invalid_arms,
            "legoesm_runs": {"fp64": l64_metadata, "fp32": l32_metadata},
            "metrics": metrics,
            "rows": rows,
            "state_frame": (
                "initial/midpoint NEMO Nbb entry vs legoESM pre-step prognostic; "
                "final NEMO tn/un/sshn restart vs legoESM after full completed duration"
            ),
            "mask_and_reduction": (
                "phase3 gate halo strip and common wet T/U intersection; volume "
                "metrics use certified live partial-cell thickness; float64 accumulation"
            ),
        }
    bridge = deterministic_bridge(case, arms)
    rows = []

    def curve(name: str, units: str):
        candidate = _curve_distance(metrics["L64"][name], metrics["N2"][name])
        floor = _curve_distance(metrics["L32"][name], metrics["L64"][name])
        spread = _curve_distance(metrics["N4"][name], metrics["N2"][name])
        rows.append(
            _row(
                name, candidate, floor, spread, units, "maximum absolute registered-time separation"
            )
        )

    if case == "OVERFLOW-zps":
        curve("plume_descent_m", "m")
        curve("plume_front_km", "km")
        for name, key, reducer, units in (
            (
                "final_temperature_histogram_tv",
                "final_temperature_histogram",
                _histogram_tv,
                "probability",
            ),
            (
                "final_water_mass_census",
                "final_water_mass_census",
                _curve_distance,
                "volume fraction",
            ),
        ):
            candidate = reducer(metrics["L64"][key], metrics["N2"][key])
            floor = reducer(metrics["L32"][key], metrics["L64"][key])
            spread = reducer(metrics["N4"][key], metrics["N2"][key])
            rows.append(_row(name, candidate, floor, spread, units, reducer.__name__))
    else:
        curve("front_position_km", "km")
        curve("rpe_relative", "fraction")
        curve("temperature_variance_fraction", "fraction")
        name = "front_speed_anchor_ratio"
        candidate = abs(metrics["L64"][name] - metrics["N2"][name])
        floor = abs(metrics["L32"][name] - metrics["L64"][name])
        spread = abs(metrics["N4"][name] - metrics["N2"][name])
        rows.append(_row(name, candidate, floor, spread, "ratio", "absolute scalar separation"))

    for name, field in (
        ("temperature_linf", "T"),
        ("instantaneous_u_linf", "u"),
    ):
        distances = _field_linf(case, arms, field)
        rows.append(
            _row(
                name,
                distances["candidate_distance"],
                distances["precision_floor"],
                distances["scheme_spread"],
                "normalized L-inf",
                "maximum over midpoint-entry and final-restart wet field",
            )
        )

    names = {row["name"] for row in rows}
    if plant_unregistered:
        names.add("planted_unregistered_metric")
    require(
        names == REGISTERED_METRICS[case],
        f"{case}: metric coverage mismatch {names ^ REGISTERED_METRICS[case]}",
    )
    if plant_state:
        row = next(row for row in rows if row["name"] == "temperature_linf")
        require(
            row["verdict"] == "OUTSIDE" and row["candidate_distance"] >= 0.45,
            "planted state did not produce its registered gross distance",
        )
    if plant_census and case == "OVERFLOW-zps":
        row = next(row for row in rows if row["name"] == "final_water_mass_census")
        require(
            row["verdict"] == "OUTSIDE" and row["candidate_distance"] >= 0.9,
            "planted census did not produce its registered gross distance",
        )

    return {
        "case": case,
        "git_sha": git_sha(),
        "preregistration_commit": PREREG_SHA,
        "status": "OUTSIDE"
        if any(row["verdict"] == "OUTSIDE" for row in rows)
        else "CLASSIFIED_NO_OUTSIDE",
        "verdict_counts": {
            value: sum(row["verdict"] == value for row in rows)
            for value in ("INDISTINGUISHABLE-AT-FLOOR", "WITHIN-SCHEME-SPREAD", "OUTSIDE")
        },
        "alternative_namelist_diff": alternative_diff,
        "state_frame": (
            "initial/midpoint NEMO Nbb entry vs legoESM pre-step prognostic; "
            "final NEMO tn/un/sshn restart vs legoESM after full completed duration"
        ),
        "mask_and_reduction": (
            "phase3 gate halo strip and common wet T/U intersection; volume metrics "
            "use certified live partial-cell thickness; float64 accumulation"
        ),
        "legoesm_runs": {"fp64": l64_metadata, "fp32": l32_metadata},
        "metrics": metrics,
        "deterministic_bridge": bridge,
        "rows": sorted(rows, key=lambda row: row["name"]),
        "controls": {
            "plant_state": plant_state,
            "plant_census": plant_census,
            "plant_unregistered": plant_unregistered,
        },
    }


@scoped_allow_dirty
def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-legoesm")
    run_parser.add_argument("--case", choices=tuple(CASES), required=True)
    run_parser.add_argument("--precision", choices=("fp64", "fp32"), required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--allow-dirty", action="store_true",
                            help="stamp '<sha>-dirty' instead of refusing a dirty tree")
    run_parser.add_argument("--check-finite-every-step", action="store_true")
    trace_parser = subparsers.add_parser("run-fp32-temperature-trace")
    trace_parser.add_argument("--output", type=Path, required=True)
    trace_parser.add_argument("--allow-dirty", action="store_true",
                              help="stamp '<sha>-dirty' instead of refusing a dirty tree")
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--case", choices=tuple(CASES), required=True)
    score_parser.add_argument("--lego-root", type=Path, default=FULL_ROOT / "legoesm")
    score_parser.add_argument("--output", type=Path)
    score_parser.add_argument("--plant-state", action="store_true")
    score_parser.add_argument("--plant-census", action="store_true")
    score_parser.add_argument("--plant-unregistered", action="store_true")
    args = parser.parse_args()

    # This gate owns an --allow-dirty flag; bridge it to the shared stamp or
    # wiring the stamp in would kill the flag and raise after the model run.
    allow_dirty_stamps(args.allow_dirty)
    if args.command == "run-legoesm":
        report = run_legoesm(
            args.case,
            args.precision,
            args.output_dir,
            git_sha(allow_dirty=args.allow_dirty),
            check_finite_every_step=args.check_finite_every_step,
        )
    elif args.command == "run-fp32-temperature-trace":
        report = run_fp32_temperature_trace(
            args.output, git_sha(allow_dirty=args.allow_dirty))
    else:
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
        require(bool(jax.config.jax_enable_x64), "scoring requires JAX x64")
        report = score_case(
            args.case,
            args.lego_root,
            plant_state=args.plant_state,
            plant_census=args.plant_census,
            plant_unregistered=args.plant_unregistered,
        )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StatisticalError as error:
        print(f"OUTSIDE: {error}")
        raise SystemExit(1)
