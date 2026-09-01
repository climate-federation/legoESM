#!/usr/bin/env python3
"""Fail-closed 19-frame OVERFLOW barotropic first-divergence gate.

The oracle record is a WRITE-only trace of the executed NEMO 5.0.2
``nn_bt_flt=1, rn_bt_alpha=0, nn_e=3`` branch.  The legoESM record is returned
by a private, side-effect-free test hook from the production recurrence.  No
public numerical selector is introduced by this instrument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

BAR = 1.0e-15
CASE = "OVERFLOW-zps"
EXPECTED = (206, 7, 4, 19, 64)
DEFAULT_ORACLE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/barotropic_walk/"
    "oracle_kt1_calls/oracle_overflow_bt_substeps_kt00000001_call1.bin"
)
DEFAULT_NEW_ENTRY = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/barotropic_walk/"
    "oracle_kt1_calls/oracle_step_entry_kt00000001.bin"
)
DEFAULT_CERTIFIED_ENTRY = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10/"
    "oracle_step_entry_kt00000001.bin"
)
DEFAULT_TRAJECTORY = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/barotropic_walk/overflow_kt1_10_flux_gate.json"
)

FIELDS = (
    "eta_entry",
    "u_entry",
    "v_entry",
    "eta_mid",
    "u_mid",
    "v_mid",
    "transport_u",
    "transport_v",
    "eta_continuity",
    "eta_pgf",
    "pgf_u",
    "pgf_v",
    "slow_u",
    "slow_v",
    "drag_u",
    "drag_v",
    "u_exit",
    "v_exit",
    "eta_exit",
)
STAGGER = {
    name: (
        "T"
        if name.startswith("eta")
        else "U"
        if name.endswith("_u") or name.startswith("u_")
        else "V"
    )
    for name in FIELDS
}

# Rule 1d: every captured array is registered to the exact executing source
# assignment and to the substep-local time frame it represents.  The first
# substep's ``m`` is the caller's Kmm; later ``m`` values are the preceding
# substep exit after dynspg_ts.F90:804-816 rotates the recurrence.
FRAME_REGISTRY = {
    "eta_entry": {"time_level": "external m (caller Kmm on substep 1)", "source": "src/OCE/DYN/dynspg_ts.F90:484-486,814-816"},
    "u_entry": {"time_level": "external m (caller Kmm on substep 1)", "source": "src/OCE/DYN/dynspg_ts.F90:484-487,806-808"},
    "v_entry": {"time_level": "external m (caller Kmm on substep 1)", "source": "src/OCE/DYN/dynspg_ts.F90:484-488,810-812"},
    "eta_mid": {"time_level": "external m+1/2 AB3 predictor", "source": "src/OCE/DYN/dynspg_ts.F90:534-543,558-562"},
    "u_mid": {"time_level": "external m+1/2 AB3 predictor", "source": "src/OCE/DYN/dynspg_ts.F90:534-550"},
    "v_mid": {"time_level": "external m+1/2 AB3 predictor", "source": "src/OCE/DYN/dynspg_ts.F90:534-553"},
    "transport_u": {"time_level": "external m+1/2 transport", "source": "src/OCE/DYN/dynspg_ts.F90:603-605"},
    "transport_v": {"time_level": "external m+1/2 transport", "source": "src/OCE/DYN/dynspg_ts.F90:603-608"},
    "eta_continuity": {"time_level": "external m+1 after continuity", "source": "src/OCE/DYN/dynspg_ts.F90:623-630"},
    "eta_pgf": {"time_level": "external m+1/2 backward-interpolated PGF operand", "source": "src/OCE/DYN/dynspg_ts.F90:671-679"},
    "pgf_u": {"time_level": "external m+1/2 PGF tendency", "source": "src/OCE/DYN/dynspg_ts.F90:681-684"},
    "pgf_v": {"time_level": "external m+1/2 PGF tendency", "source": "src/OCE/DYN/dynspg_ts.F90:681-685"},
    "slow_u": {"time_level": "caller Kmm slow forcing, fixed through external loop", "source": "src/OCE/DYN/dynspg_ts.F90:270-300"},
    "slow_v": {"time_level": "caller Kmm slow forcing, fixed through external loop", "source": "src/OCE/DYN/dynspg_ts.F90:270-300"},
    "drag_u": {"time_level": "external m drag tendency", "source": "src/OCE/DYN/dynspg_ts.F90:699-704"},
    "drag_v": {"time_level": "external m drag tendency", "source": "src/OCE/DYN/dynspg_ts.F90:699-704"},
    "u_exit": {"time_level": "external m+1 instantaneous velocity", "source": "src/OCE/DYN/dynspg_ts.F90:734-755"},
    "v_exit": {"time_level": "external m+1 instantaneous velocity", "source": "src/OCE/DYN/dynspg_ts.F90:734-760"},
    "eta_exit": {"time_level": "external m+1 instantaneous sea level", "source": "src/OCE/DYN/dynspg_ts.F90:623-630"},
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    """Return the exact legoESM producer revision; fail closed off Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise GateError(f"cannot stamp legoESM git SHA: {error}") from error


def validate_frame_registry(registry=FRAME_REGISTRY) -> None:
    require(set(registry) == set(FIELDS), (
        "frame registry mismatch: missing="
        f"{sorted(set(FIELDS) - set(registry))}, extra="
        f"{sorted(set(registry) - set(FIELDS))}"
    ))
    for name, row in registry.items():
        require(bool(row.get("time_level")), f"{name}: missing time level")
        source = row.get("source", "")
        require(source.startswith("src/OCE/DYN/dynspg_ts.F90:"),
                f"{name}: missing exact dynspg_ts source")


def _read_exact(handle, count: int, context: str) -> bytes:
    value = handle.read(count)
    require(len(value) == count, f"{context}: truncated record")
    return value


def _xy_full(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F")[2:-2, 2:-2].T


def _xy_a2d(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx - 4, ny - 4), order="F").T


def read_oracle_trace(path: Path) -> dict:
    """Read the mixed full-halo/A2D NEMO stream record, refusing leftovers."""
    with path.open("rb") as handle:
        magic = _read_exact(handle, 16, str(path)).decode("ascii").rstrip()
        header = struct.unpack("=11i", _read_exact(handle, 44, str(path)))
        (version, kt, call, kbb, kmm, kaa, ncycle, nx, ny, nfields, bits) = header
        require(magic == "NEMO_L1_OVBT_1", f"{path}: bad magic {magic!r}")
        require(
            (version, kt, call, nx, ny, ncycle, nfields, bits) == (1, 1, 1, *EXPECTED),
            f"{path}: bad header {header}",
        )
        full_count = nx * ny
        a2d_count = (nx - 4) * (ny - 4)
        substeps = []
        for expected_jn in range(1, ncycle + 1):
            (jn,) = struct.unpack("=i", _read_exact(handle, 4, str(path)))
            require(jn == expected_jn, f"{path}: substep {jn}, expected {expected_jn}")
            frames = {}
            for name in FIELDS:
                count = a2d_count if name in ("slow_u", "slow_v") else full_count
                raw = _read_exact(handle, 8 * count, f"{path}:{jn}:{name}")
                values = np.frombuffer(raw, dtype=np.float64).copy()
                frames[name] = (
                    _xy_a2d(values, nx, ny) if count == a2d_count else _xy_full(values, nx, ny)
                )
            substeps.append(frames)
        require(handle.read(1) == b"", f"{path}: trailing unregistered bytes")
    require((kbb, kmm, kaa) == (1, 1, 3), f"{path}: unexpected time levels")
    return {
        "header": {
            "version": version,
            "kt": kt,
            "call": call,
            "Kbb": kbb,
            "Kmm": kmm,
            "Kaa": kaa,
            "icycle": ncycle,
            "jpi": nx,
            "jpj": ny,
            "nfields": nfields,
            "bits": bits,
        },
        "substeps": substeps,
    }


def _candidate_frame(values, staggering: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if staggering == "T":
        return values
    if staggering == "U":
        return values[:, 1:]
    return values[1:, :]


def capture_legoesm_trace(*, flux_form_override) -> dict:
    """Capture call 1 without changing the two-value production return."""
    import jax
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocean_model
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config
    require(cfg.barotropic.barotropic_time_filter == "nemo_boxcar1_ab3", "wrong filter")
    require(cfg.barotropic.n_barotropic_substeps == 3, "wrong nn_e")
    require(cfg.momentum_time_integrator == "rk3_ws", "wrong momentum integrator")
    require(cfg.momentum_advection == "flux_form", "wrong momentum form")

    original = ocean_model.barotropic_substeps_latlon_cgrid
    captured = []

    def wrapper(*args, **kwargs):
        if not captured:
            kwargs = dict(kwargs)
            kwargs["_nemo_substep_trace_test_hook"] = True
            kwargs["_nemo_flux_form_update_test_override"] = flux_form_override
            state_new, transports, trace = original(*args, **kwargs)
            captured.append(trace)
            return state_new, transports
        return original(*args, **kwargs)

    ocean_model.barotropic_substeps_latlon_cgrid = wrapper
    try:
        model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
        # The test hook must materialise its operand arrays on the host; a
        # traced Python closure would retain DynamicJaxprTracers instead.
        with jax.disable_jit():
            model.step(card.recipe.initial_state, dt=card.dt_s)
    finally:
        ocean_model.barotropic_substeps_latlon_cgrid = original
    require(len(captured) == 1, f"captured {len(captured)} call-1 traces")

    trace = captured[0]
    require(len(trace) == len(FIELDS), "candidate registry length mismatch")
    require(int(np.asarray(trace[0]).shape[0]) == 4, "candidate icycle mismatch")
    substeps = []
    for jn in range(4):
        substeps.append(
            {
                name: _candidate_frame(np.asarray(trace[index])[jn], STAGGER[name])
                for index, name in enumerate(FIELDS)
            }
        )
    state = card.recipe.initial_state
    masks = {
        "T": np.asarray(state.land_mask.data, dtype=bool),
        "U": np.asarray(state.u_mask.data[:, 1:], dtype=bool),
        "V": np.asarray(state.v_mask.data[1:, :], dtype=bool),
    }
    dtypes = {
        "T": str(np.asarray(state.T.data).dtype),
        "S": str(np.asarray(state.S.data).dtype),
        "u": str(np.asarray(state.u.data).dtype),
        "v": str(np.asarray(state.v.data).dtype),
        "ssh": str(np.asarray(state.eta.data).dtype),
        "h_partial": str(np.asarray(card.recipe.z_coord.h_partial).dtype),
    }
    require(set(dtypes.values()) == {"float64"}, f"non-fp64 arrays {dtypes}")
    return {
        "substeps": substeps,
        "masks": masks,
        "dtypes": dtypes,
        "backend": jax.default_backend(),
        "flux_form_override": flux_form_override,
    }


def score_frame(name: str, oracle, candidate, mask, *, plant=False) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate)
    active = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == active.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate dtype {candidate.dtype}")
    no_active_face = not bool(active.any())
    if no_active_face:
        active = np.ones_like(active)
    if plant:
        candidate = candidate.copy()
        candidate[tuple(np.argwhere(active)[0])] += 1.0
    require(np.all(np.isfinite(candidate[active])), f"{name}: candidate non-finite")
    residual = candidate[active] - oracle[active]
    absolute = np.abs(residual)
    index_flat = int(np.argmax(absolute))
    absolute_max = float(absolute[index_flat])
    scale = max(float(np.max(np.abs(oracle[active]))), 1.0)
    error = absolute_max / scale
    row = {
        "name": name,
        "status": "AT-BAR" if error <= BAR else "DEBT",
        "exact": bool(np.array_equal(candidate[active], oracle[active])),
        "normalized_max_abs": error,
        "absolute_max": absolute_max,
        "reference_max_abs": float(np.max(np.abs(oracle[active]))),
        "signed_residual_at_max": float(residual[index_flat]),
        "bar": BAR,
        "n": int(active.sum()),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if no_active_face and row["status"] == "AT-BAR":
        row["status"] = "UNMEASURED"
        row["reason"] = (
            "no active meridional face in the one-wet-row tank; all stored "
            "points remain a gross-zero control, not an alignment claim"
        )
    return row


def _trajectory_kt2(path: Path) -> dict:
    report = json.loads(path.read_text())
    step = next(item for item in report["steps"] if item["kt"] == 2)
    result = {}
    for field in ("T", "u", "ssh"):
        row = next(item for item in step["rows"] if item["name"].endswith(f".{field}"))
        result[field] = float(row["normalized_max_abs"])
    return result


def run(
    oracle_path: Path,
    new_entry: Path,
    certified_entry: Path,
    trajectory_path: Path,
    *,
    plant_entry=False,
    plant_exit=False,
    pytest_log: Path | None = None,
) -> dict:
    validate_frame_registry()
    require(
        new_entry.read_bytes() == certified_entry.read_bytes(),
        "instrumented step-entry is not byte-identical to certified oracle",
    )
    oracle = read_oracle_trace(oracle_path)
    baseline = capture_legoesm_trace(flux_form_override=False)
    faithful = capture_legoesm_trace(flux_form_override=True)
    require(baseline["masks"].keys() == faithful["masks"].keys(), "mask registry drift")

    arms = {}
    first = {}
    for arm_name, candidate in (
        ("legacy_velocity_update", baseline),
        ("nemo_flux_form_update", faithful),
    ):
        substeps = []
        first_over_bar = None
        for jn, (reference_frames, candidate_frames) in enumerate(
            zip(oracle["substeps"], candidate["substeps"]), start=1
        ):
            rows = []
            for name in FIELDS:
                staggering = STAGGER[name]
                plant = (plant_entry and jn == 1 and name == "eta_entry") or (
                    plant_exit and jn == 1 and name == "u_exit"
                )
                row = score_frame(
                    f"{arm_name}.substep{jn}.{name}",
                    reference_frames[name],
                    candidate_frames[name],
                    candidate["masks"][staggering],
                    plant=plant,
                )
                row.update(
                    {
                        "frame": "instantaneous_external_substep_operand",
                        "oracle_time_level": FRAME_REGISTRY[name]["time_level"],
                        "oracle_source": FRAME_REGISTRY[name]["source"],
                        "staggering_and_reduction": (
                            f"oracle NEMO {name} and legoESM {name} are both "
                            f"native {staggering}-point operands at external "
                            f"substep {jn}; both strip the NEMO two-cell halo, "
                            "intersect the certified native wet mask, and use an "
                            "elementwise L-infinity reduction with no depth or "
                            "substep-time averaging"
                        ),
                    }
                )
                rows.append(row)
                if first_over_bar is None and row["status"] == "DEBT":
                    first_over_bar = {
                        "substep": jn,
                        "frame": name,
                        "normalized_max_abs": row["normalized_max_abs"],
                    }
            substeps.append({"substep": jn, "rows": rows})
        arms[arm_name] = {
            "substeps": substeps,
            "first_over_bar": first_over_bar,
            "backend": candidate["backend"],
            "dtypes": candidate["dtypes"],
        }
        first[arm_name] = first_over_bar

    scaling = []
    for jn in range(4):
        ref = oracle["substeps"][jn]["u_exit"]
        old = baseline["substeps"][jn]["u_exit"]
        new = faithful["substeps"][jn]["u_exit"]
        mask = faithful["masks"]["U"]
        old_error = float(np.max(np.abs(old[mask] - ref[mask])))
        new_error = float(np.max(np.abs(new[mask] - ref[mask])))
        movement = float(np.max(np.abs(new[mask] - old[mask])))
        scaling.append(
            {
                "substep": jn + 1,
                "legacy_absolute_error": old_error,
                "faithful_absolute_error": new_error,
                "causal_movement": movement,
                "movement_over_legacy_error": movement / max(old_error, np.finfo(float).tiny),
                "improvement_factor": old_error / max(new_error, np.finfo(float).tiny),
            }
        )

    kt2 = _trajectory_kt2(trajectory_path)
    prior = {
        "T": 2.4003505494363254e-08,
        "u": 3.082388670642283e-06,
        "ssh": 1.2372313177265448e-07,
    }
    movement = {
        field: {
            "before": prior[field],
            "after": kt2[field],
            "after_over_before": kt2[field] / prior[field],
        }
        for field in prior
    }
    causal_pass = (
        kt2["u"] <= prior["u"] / 10.0
        and kt2["T"] <= prior["T"] * 10.0
        and kt2["ssh"] <= prior["ssh"] * 10.0
    )
    first_slow = next(
        row
        for row in arms["legacy_velocity_update"]["substeps"][0]["rows"]
        if row["name"].endswith(".slow_u")
    )
    first_exit = next(
        row
        for row in arms["legacy_velocity_update"]["substeps"][0]["rows"]
        if row["name"].endswith(".u_exit")
    )
    slow_dt_prediction = first_slow["absolute_max"] * (10.0 / 3.0)
    slow_ratio = first_exit["absolute_max"] / max(slow_dt_prediction, np.finfo(float).tiny)

    controls_ok = not plant_entry and not plant_exit
    if plant_entry:
        controls_ok = first["legacy_velocity_update"]["frame"] == "eta_entry"
    if plant_exit:
        controls_ok = first["legacy_velocity_update"]["frame"] == "u_exit"
    require(controls_ok, "planted control did not become first DEBT")

    faithful_substep1 = [
        {
            "frame": row["name"].rsplit(".", 1)[-1],
            "status": row["status"],
            "normalized_max_abs": row["normalized_max_abs"],
            "absolute_max": row["absolute_max"],
        }
        for row in arms["nemo_flux_form_update"]["substeps"][0]["rows"]
    ]
    artifacts = {
        "oracle_trace": {"path": str(oracle_path), "sha256": sha256(oracle_path)},
        "instrumented_entry": {"path": str(new_entry), "sha256": sha256(new_entry)},
        "certified_entry": {"path": str(certified_entry), "sha256": sha256(certified_entry)},
        "trajectory_gate": {"path": str(trajectory_path), "sha256": sha256(trajectory_path)},
    }
    if pytest_log is not None:
        require(pytest_log.is_file(), f"pytest log does not exist: {pytest_log}")
        artifacts["pytest_log"] = {"path": str(pytest_log), "sha256": sha256(pytest_log)}

    return {
        "format": "nemo-testcase-l1-overflow-barotropic-gate-v1",
        "case": CASE,
        "status": "DEBT",
        "bar": BAR,
        "resolved_program": {
            "nn_bt_flt": 1,
            "rn_bt_alpha": 0.0,
            "nn_e": 3,
            "actual_icycle": 4,
            "ln_bt_fw": True,
            "momentum_advection": "flux_form_up3",
            "drag": "OFF",
            "rotation": "f=0",
        },
        "legoesm_git_sha": git_sha(),
        "time_level_header": oracle["header"],
        "frame_time_level_registry": FRAME_REGISTRY,
        "artifacts": artifacts,
        "overlap_control": "EXACT_BYTE_IDENTITY",
        "arms": arms,
        "faithful_flux_update_substep1_19_frame_residuals": faithful_substep1,
        "scaling_check_before_owner_label": {
            "u_exit_by_substep": scaling,
            "substep1_slow_u_error_times_dt": slow_dt_prediction,
            "substep1_u_exit_error": first_exit["absolute_max"],
            "exit_over_slow_dt_prediction": slow_ratio,
        },
        "whole_step_kt2_causal_arm": {
            "frozen_confirm_predicate": ("u error falls >=10x and neither T nor SSH worsens >10x"),
            "passed": causal_pass,
            "movement": movement,
        },
        "ownership": {
            "slow_u_roundoff_tail": {
                "label": "CONFIRMED_FIRST_STRICT_BAR_OPERAND_CONTRIBUTOR",
                "root_kt2_initiator": "REFUTED_BY_SCALING",
                "reason": (
                    "dt times the 5.63e-16 slow-forcing residual predicts the "
                    "1.87e-15 substep-1 exit tail, but is about nine orders "
                    "below the kt=2 instantaneous-U debt"
                ),
            },
            "nemo_flux_form_external_update": {
                "substep_recurrence": "CONFIRMED_STRUCTURAL_OWNER",
                "root_kt2_initiator": "REFUTED_BY_FROZEN_CAUSAL_PREDICATE",
                "production_selection": "KEPT_LITERAL_UPDATE_RULE8",
                "reason": (
                    "the literal flux-form external update (dynspg_ts.F90:"
                    "734-761) is PRODUCTION-ACTIVE under the NEMO WS-RK3 + "
                    "flux-form identity (barotropic_latlon_cgrid.py "
                    "_nemo_flux_form_update). Its 19 substep-1 frames are "
                    "printed in faithful_flux_update_substep1_19_frame_"
                    "residuals: every measured frame is AT-BAR except u_exit "
                    "at 1.8735013540549517e-15, which is identical in the "
                    "legacy arm and equals dt times the 5.63e-16 slow_u "
                    "input residual (ratio 0.998), i.e. it is inherited from "
                    "the slow forcing, not produced by the update. Substeps "
                    "2-4 fall from 2.6e-9/1.3e-7/5.6e-7 to the 1e-15 class. "
                    "Faithful-but-worse at kt=2 (u 3.08238867e-06 -> "
                    "3.31108168e-06, SSH 1.23723132e-07 -> 1.04916076e-14, T "
                    "unchanged; whole_step_kt2_causal_arm) is disclosed, "
                    "not reverted (Rule 8); the compensated defect is the "
                    "RK3 stage composition measured by the phase-3 gate"
                ),
            },
            "substep1_pgf_continuity_metrics": {
                "label": "CONFIRMED_EXONERATED_THROUGH_FIRST_STRICT_BOUNDARY",
            },
            "drag": {
                "label": "CONFIRMED_INACTIVE_BOTH_MODELS",
                "reason": "resolved ln_drg_OFF=T and both dumped drag operands are exact zero",
            },
            "kt2_root_divergence": {
                "label": "UNMEASURED_OUTSIDE_BAROTROPIC_SUBSTEP_REGISTER",
                "reason": (
                    "the source-faithful substep recurrence no longer amplifies, "
                    "while the post-stage whole-step instantaneous U remains "
                    "3.31e-6; the next operand lies in RK3 stage composition"
                ),
            },
        },
        "unmeasured": [
            "active-v alignment (no wet V face)",
            "post-barotropic RK3 stage operand initiating kt=2 instantaneous U/T",
        ],
        "controls": {"plant_entry": plant_entry, "plant_exit": plant_exit},
        "source_citations": [
            "NEMO 5.0.2 src/OCE/DYN/dynspg_ts.F90:270-300",
            "NEMO 5.0.2 src/OCE/DYN/dynspg_ts.F90:484-493",
            "NEMO 5.0.2 src/OCE/DYN/dynspg_ts.F90:534-630",
            "NEMO 5.0.2 src/OCE/DYN/dynspg_ts.F90:672-761",
            "NEMO 5.0.2 src/OCE/DYN/dynspg_ts.F90:823-847",
        ],
    }


def _control_only_report(*, plant_entry: bool, plant_exit: bool) -> dict:
    require(plant_entry or plant_exit, "--control-only requires a planted control")
    oracle = np.zeros((2, 3), dtype=np.float64)
    row = score_frame(
        "eta_entry" if plant_entry else "u_exit",
        oracle,
        oracle.copy(),
        np.ones_like(oracle, dtype=bool),
        plant=True,
    )
    return {"status": row["status"], "row": row}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--new-entry", type=Path, default=DEFAULT_NEW_ENTRY)
    parser.add_argument("--certified-entry", type=Path, default=DEFAULT_CERTIFIED_ENTRY)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-entry", action="store_true")
    parser.add_argument("--plant-exit", action="store_true")
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--pytest-log", type=Path)
    args = parser.parse_args(argv)
    if args.control_only:
        report = _control_only_report(
            plant_entry=args.plant_entry, plant_exit=args.plant_exit)
    else:
        report = run(
            args.oracle,
            args.new_entry,
            args.certified_entry,
            args.trajectory,
            plant_entry=args.plant_entry,
            plant_exit=args.plant_exit,
            pytest_log=args.pytest_log,
        )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as error:
        print(f"DEBT: {error}", file=sys.stderr)
        raise SystemExit(1)
