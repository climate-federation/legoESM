#!/usr/bin/env python3
"""Locate rung-0's live-thickness refusal and score fixed growth blocks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round71_independent_month_gate as prior_month,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round130_rung0_month_gate as month,
)


STEPS = 240
CHECKPOINTS = tuple(range(10, 100, 10)) + (95,)
EXPECTED_ERROR = "raw-mesh e3w_int must contain only finite values > 0"
EXPECTED_PRODUCTION_BOUNDARY = 96
EXPECTED_COLUMN = (86, 159)
PLANTS = (
    "none",
    "initial-entry",
    "boundary-step",
    "boundary-cell",
    "growth-provenance",
)


class GateError(RuntimeError):
    """The month boundary or record coverage violates the frozen protocol."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_restart(root: Path, step: int, card) -> tuple[dict[str, np.ndarray], dict]:
    """Read one ordinary two-rank NEMO restart at an explicit step."""

    paths = [root / f"ORCA2_{step:08d}_restart_{rank:04d}.nc" for rank in (0, 1)]
    present = [path.is_file() for path in paths]
    if not any(present):
        raise FileNotFoundError(paths[0])
    require(all(present), f"step {step}: restart rank coverage is partial")
    slabs: list[dict[str, np.ndarray]] = []
    coordinates: list[tuple[np.ndarray, np.ndarray]] = []
    files = []
    for path in paths:
        with Dataset(path) as dataset:
            require(float(np.asarray(dataset["kt"][:])) == float(step),
                    f"{path.name}: kt is not {step}")
            slab = {
                "T": prior_month._restart_xyz(dataset["tn"], "tn"),
                "S": prior_month._restart_xyz(dataset["sn"], "sn"),
                "u": prior_month._restart_xyz(dataset["un"], "un"),
                "v": prior_month._restart_xyz(dataset["vn"], "vn"),
                "ssh": prior_month._restart_xy(dataset["sshn"], "sshn"),
            }
            coordinates.append((
                np.asarray(dataset["nav_lat"][:], dtype=np.float32),
                np.asarray(dataset["nav_lon"][:], dtype=np.float32),
            ))
        require(all(np.isfinite(values).all() for values in slab.values()),
                f"{path.name}: non-finite restart payload")
        slabs.append(slab)
        files.append({"path": str(path), "sha256": sha256(path)})
    fields = {
        name: np.concatenate([slabs[0][name], slabs[1][name]], axis=1)
        for name in month.FIELDS
    }
    lat = np.concatenate([coordinates[0][0], coordinates[1][0]], axis=1)
    lon = np.concatenate([coordinates[0][1], coordinates[1][1]], axis=1)
    expected_lat = np.asarray(
        np.rad2deg(np.asarray(card.recipe.grid.lat_T)), dtype=np.float32)
    expected_lon = np.asarray(
        np.rad2deg(np.asarray(card.recipe.grid.lon_T)), dtype=np.float32)
    require(np.array_equal(lat.view(np.uint32), expected_lat.view(np.uint32)),
            f"step {step}: restart latitude orientation moved")
    require(np.array_equal(lon.view(np.uint32), expected_lon.view(np.uint32)),
            f"step {step}: restart longitude orientation moved")
    return fields, {"step": step, "files": files, "status": "BIT_EXACT_ORIENTATION"}


def _invalid_live_thickness(card, state, eta, label: str) -> dict[str, object]:
    """Replay one recorded eta boundary outside the executable and locate debt."""

    import jax

    from legoesm.ocean.eos import nemo_bn2_live_geometry, nemo_r3t_stretch

    _, _, e3w = nemo_bn2_live_geometry(
        card.recipe.z_coord, eta, state.H_bathy.data,
        r3t_evaluation="nemo_reciprocal")
    stretch = nemo_r3t_stretch(
        card.recipe.z_coord, eta, state.H_bathy.data,
        evaluation="nemo_reciprocal")
    e3w_np, stretch_np = (np.asarray(value) for value in jax.device_get((e3w, stretch)))
    invalid = ~(np.isfinite(e3w_np) & (e3w_np > 0.0))
    locations = np.argwhere(invalid)
    first = None
    if locations.size:
        index = tuple(int(value) for value in locations[0])
        raw = np.asarray(card.recipe.z_coord.nemo_e3w_0)[..., 1:]
        first = {
            "index_jik": list(index),
            "value": float(e3w_np[index]),
            "raw_e3w_0": float(raw[index]),
            "stretch": float(stretch_np[index[:2]]),
            "eta": float(np.asarray(eta)[index[:2]]),
            "H_bathy": float(np.asarray(state.H_bathy.data)[index[:2]]),
        }
    return {
        "boundary": label,
        "source": "offline replay of ordinary completed state",
        "shape": list(e3w_np.shape),
        "invalid_count": int(locations.shape[0]),
        "first_invalid": first,
        "minimum": float(np.nanmin(e3w_np)),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(e3w_np))),
        "nonpositive_finite_count": int(np.count_nonzero(np.isfinite(e3w_np) & (e3w_np <= 0.0))),
    }


def _stage_live_thickness_replay(
    card, state, hooks, freshwater, surface,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Bypass only the known guard, recover eta, then replay its three stages."""

    import jax

    import legoesm.ocean.eos as eos
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    original_error_if = eos.eqx.error_if

    def selective_error_if(value, predicate, message, **kwargs):
        if message == EXPECTED_ERROR:
            return value
        return original_error_if(value, predicate, message, **kwargs)

    # A momentum-stage exposure changes only returned u/v, making this a fresh
    # compiled replay while leaving the completed barotropic eta untouched.
    replay_hooks = hooks._replace(expose_momentum_stage=1)
    eos.eqx.error_if = selective_error_if
    try:
        replay = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=replay_hooks)
        replay_state = jax.device_get(jax.block_until_ready(replay.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface)))
    finally:
        eos.eqx.error_if = original_error_if
    eta_entry = np.asarray(state.eta.data)
    eta_after = np.asarray(replay_state.eta.data)
    require(bool(np.isfinite(eta_after).all()),
            "guard-bypassed replay did not return a finite eta")
    etas = (
        ("step_entry", eta_entry),
        ("stage1_one_third", eta_entry + (eta_after - eta_entry) / 3.0),
        ("stage2_one_half", 0.5 * (eta_entry + eta_after)),
        ("barotropic_after", eta_after),
    )
    rows = [_invalid_live_thickness(card, state, eta, label)
            for label, eta in etas]
    first = next((copy.deepcopy(row) for row in rows
                  if int(row["invalid_count"]) > 0), None)
    require(first is not None,
            "guard-bypassed stage replay has no invalid live thickness")
    return rows, first


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    """Validate protocol and retain changed scientific predictions."""

    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "initial-entry":
        report["initial_entry"]["T"]["bit_exact"] = False
    elif plant == "boundary-step":
        report["runtime_refusal"]["step"] = 0
    elif plant == "boundary-cell":
        report["live_thickness"]["first_invalid"]["index_jik"] = [999, 999, 999]
    elif plant == "growth-provenance":
        report["growth_table"][0]["claim_label"] = "given NEMO's entry"

    require(report.get("claim_label") == "independent",
            "month claim is not independent")
    require(report.get("initial_mode") == "card_own_state",
            "month did not use the card's own initial state")
    initial = report.get("initial_entry", {})
    require(tuple(initial) == month.FIELDS and all(
        row["bit_exact"] and int(row["unequal"]) == 0
        for row in initial.values()), "independent entry is not bit-exact")
    refusal = report.get("runtime_refusal")
    require(refusal is not None and EXPECTED_ERROR in str(refusal.get("message")),
            "terminal is not the registered live-thickness refusal")
    require(1 <= int(refusal["step"]) <= STEPS,
            "runtime-refusal step is outside the month")
    require(int(report.get("steps_completed", -1)) == int(refusal["step"]) - 1,
            "completed-step count disagrees with refusal")
    live = report.get("live_thickness", {})
    require(int(live.get("invalid_count", 0)) > 0,
            "offline geometry has no invalid live thickness")
    first = live.get("first_invalid")
    require(first is not None and len(first.get("index_jik", [])) == 3,
            "first invalid live-thickness index is absent")
    shape = live.get("shape", [])
    require(len(shape) == 3 and all(
        0 <= int(index) < int(size)
        for index, size in zip(first["index_jik"], shape)),
        "first invalid live-thickness index is outside the field")
    production = not bool(report.get("private_halo_unit"))
    expected_checkpoints = (
        CHECKPOINTS if production else
        tuple(step for step in CHECKPOINTS if step < int(refusal["step"]))
    )
    table = report.get("growth_table", [])
    require([int(row["step"]) for row in table] == list(expected_checkpoints),
            "growth table checkpoint registry moved")
    require(all(row.get("claim_label") == "independent" for row in table),
            "growth table mixed claim populations")
    missing = [int(value) for value in report.get("missing_oracle_steps", [])]
    require(all(value in expected_checkpoints for value in missing),
            "missing-oracle registry contains an unrequested step")
    for row in table:
        require(tuple(row["candidate_max_abs"]) == month.FIELDS,
                f"step {row['step']}: candidate field registry moved")
        if int(row["step"]) in missing:
            require(row.get("error_rows") is None,
                    f"step {row['step']}: missing oracle has a score")
        else:
            require(tuple(row.get("error_rows", {})) == month.FIELDS,
                    f"step {row['step']}: oracle score registry moved")

    boundary_step = int(refusal["step"])
    column = tuple(int(value) for value in first["index_jik"][:2])
    complete_scores = production and not missing
    late_growth = False
    if complete_scores and len(table) >= 2:
        penultimate, last = table[-2:]
        late_growth = any(
            float(last["error_rows"][name]["max_abs"])
            > float(penultimate["error_rows"][name]["max_abs"])
            for name in ("T", "S", "u", "v")
        )
    report["prediction_ledger"] = {
        "R186-P1": ("CONFIRMED" if production and boundary_step == 96 else
                    "NOT_APPLICABLE_PRIVATE" if not production else "REFUTED"),
        "R186-P2": "CONFIRMED" if column == EXPECTED_COLUMN else "REFUTED",
        "R186-P3": "CONFIRMED" if complete_scores and late_growth else "REFUTED",
        "R186-P5": ("CONFIRMED" if not production and boundary_step <= 96 else
                    "NOT_APPLICABLE_PRODUCTION" if production else "REFUTED"),
        "R186-P6": "CONFIRMED",
    }
    report["status"] = (
        "STOPPED_FOR_RECORD" if missing else "PASS_R186_MONTH_BOUNDARY")
    return report


def measure(
    deck_root: Path,
    frame_root: Path,
    oracle_root: Path,
    expect_commit: str,
    *,
    private_halo_unit: bool = False,
) -> dict[str, object]:
    """Run the ordinary month and replay geometry only after completed steps."""

    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-186 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "month requires production JIT on CPU")
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    oracle_entry = rung0.assemble_frame(frame_root, 1, 0)
    candidate_entry = rung0.candidate_fields(state)
    initial = {
        name: {
            "bit_exact": bool(np.array_equal(candidate_entry[name], oracle_entry[name])),
            "unequal": int(np.count_nonzero(candidate_entry[name] != oracle_entry[name])),
        }
        for name in month.FIELDS
    }
    freshwater, surface = ladder._zero_forcing(tuple(np.asarray(state.eta.data).shape))
    hooks = _NEMOWSRK3TestHooks()
    if private_halo_unit:
        hooks = _NEMOWSRK3TestHooks(
            barotropic_external_mode_association=True,
            barotropic_reference_face_depth_override=(
                rung0.ladder.build_reference_depth_override(card)),
            barotropic_unmasked_v_transport=True,
            barotropic_materialize_v_transport=True,
        )
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    started = time.time()
    completed = 0
    refusal = None
    rows: list[dict[str, object]] = []
    missing: list[int] = []
    for step in range(1, STEPS + 1):
        try:
            pending = model.step(
                state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
            state = jax.device_get(jax.block_until_ready(pending))
        except ValueError as error:
            refusal = {"step": step, "message": str(error)}
            break
        completed = step
        found = month.first_nonfinite(rung0.candidate_fields(state))
        require(found is None, f"step {step}: state became non-finite: {found}")
        if step in CHECKPOINTS:
            candidate = rung0.candidate_fields(state)
            error_rows = None
            record = None
            try:
                oracle, record = _read_restart(oracle_root, step, card)
            except FileNotFoundError:
                missing.append(step)
            else:
                error_rows = {
                    name: month.score_field(candidate[name], oracle[name])
                    for name in month.FIELDS
                }
            rows.append({
                "step": step,
                "claim_label": "independent",
                "candidate_max_abs": {
                    name: float(np.max(np.abs(candidate[name])))
                    for name in month.FIELDS
                },
                "error_rows": error_rows,
                "oracle_restart": record,
            })
            print(f"MONTH_PROGRESS step={step}/{STEPS} "
                  f"wall_s={time.time() - started:.1f}", flush=True)
    require(refusal is not None, "month completed without the frozen refusal")
    stage_live, first_live = _stage_live_thickness_replay(
        card, state, hooks, freshwater, surface)
    return classify({
        "format": "nemo-testcase-l4-orca2-round186-month-growth-v1",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "private_halo_unit": private_halo_unit,
        "unmeasured_features": list(card.unmeasured_features),
        "initial_entry": initial,
        "steps_completed": completed,
        "runtime_refusal": refusal,
        "stage_live_thickness": stage_live,
        "live_thickness": first_live,
        "growth_table": rows,
        "missing_oracle_steps": missing,
        "oracle_root": str(oracle_root),
        "wall_seconds": time.time() - started,
        "worktree": stamp,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--private-halo-unit", action="store_true")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.oracle_root,
                         args.expect_commit)), "measurement arguments are incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.oracle_root,
                args.expect_commit, private_halo_unit=args.private_halo_unit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
