#!/usr/bin/env python3
"""Locate the first ORCA2 rung-0 growth boundary in steps 1 through 10."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as rung0_ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round143_growth_walk as shared,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round144_initial_growth_acquisition import (
    check_record,
)


STEPS = tuple(range(1, 11))
ADMISSION_SHA256 = "5f9d66af9adfca2492b9e669cda7c3d6d9c038d174037890aba520fd7fb887b0"
PLANTS = ("none", "admission", "passivity", "source-order")


class GateError(RuntimeError):
    """The admitted record or source-ordered walk violated its contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def validate_admission(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing admission JSON: {path}")
    require(shared.sha256(path) == ADMISSION_SHA256,
            "round-144 initial admission digest changed")
    report = json.loads(path.read_text())
    require(report.get("status") == "PASS_R144_INITIAL_GROWTH_RECORD",
            "initial-growth record admission is not PASS")
    require(report.get("claim_label") == "independent",
            "initial-growth record is not independent")
    require(tuple(report.get("steps", ())) == STEPS,
            "initial-growth step registry changed")
    require(report.get("rank_coverage") == "exactly-once per step",
            "initial-growth rank coverage changed")
    require(len(report.get("records", ())) == 2 * len(STEPS),
            "initial-growth rank/step census changed")
    require(len(report.get("calibration_restart_comparisons", ())) == 20,
            "initial-growth calibration census changed")
    return {
        "path": str(path), "sha256": shared.sha256(path),
        "status": report["status"], "rank_step_records": len(report["records"]),
        "calibration_restart_comparisons": len(
            report["calibration_restart_comparisons"]),
    }


def _payload(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata = check_record.read_record(path)
    raw = path.read_bytes()
    header = check_record.HEADER.unpack_from(raw, 16)
    offset = 16 + check_record.HEADER.size
    arrays: dict[str, np.ndarray] = {}
    for _ in range(header[-1]):
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        _rank, ndim, n1, n2, n3 = check_record.GROUP.unpack_from(raw, offset)
        offset += check_record.GROUP.size
        count = n1 * n2 * n3
        shape = (n1, n2) if ndim == 2 else (n1, n2, n3)
        arrays[name] = np.frombuffer(
            raw, dtype="=f8", count=count, offset=offset).copy().reshape(
                shape, order="F")
        offset += 8 * count
    require(offset == len(raw), f"{path.name}: extraction did not consume record")
    require(tuple(arrays) == check_record.FIELDS,
            f"{path.name}: extracted field registry changed")
    return metadata, arrays


def assemble_record(root: Path, step: int) -> tuple[dict[str, np.ndarray], list[dict]]:
    assembled = {
        name: np.empty((148, 180) if name in check_record.FIELDS_2D
                       else (148, 180, 31), dtype=np.float64)
        for name in check_record.FIELDS
    }
    coverage = np.zeros((148, 180), dtype=np.int8)
    rows = []
    for rank in (0, 1):
        path = root / f"oracle_r144_growth_rank{rank:04d}_kt{step:08d}.bin"
        metadata, payload = _payload(path)
        require(metadata["kt"] == step and metadata["rank"] == rank,
                f"{path.name}: filename/header identity mismatch")
        nimpp, njmpp = metadata["origin"]
        ntsi, ntsj, ntei, ntej = metadata["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: owned slab is outside the global domain")
        coverage[j0:j1, i0:i1] += 1
        for name, values in payload.items():
            block = values.transpose(1, 0) if values.ndim == 2 else values.transpose(1, 0, 2)
            assembled[name][j0:j1, i0:i1, ...] = block
        rows.append(metadata)
    require(bool(np.all(coverage == 1)), f"kt={step}: coverage is not exactly once")
    return assembled, rows


def first_over_floor(steps: dict[str, object]) -> dict[str, object] | None:
    for step in STEPS:
        for name in shared.ROW_ORDER:
            row = steps[str(step)]["rows"][name]
            if row["over_floor"]:
                return {"step": step, "row": name, **row}
    return None


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "admission":
        report["admission"]["status"] = "FAIL"
    elif plant == "passivity":
        report["steps"]["1"]["passivity"]["T"] = False
    elif plant == "source-order":
        report["row_order"][0], report["row_order"][1] = (
            report["row_order"][1], report["row_order"][0])

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "initial-growth walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("target_ji") == list(shared.TARGET), "target column changed")
    require(float(report.get("floor")) == shared.FLOOR, "comparison floor changed")
    require(tuple(report.get("row_order", ())) == shared.ROW_ORDER,
            "source row order changed")
    require(report.get("admission", {}).get("status") ==
            "PASS_R144_INITIAL_GROWTH_RECORD", "initial-growth record is not admitted")
    require(tuple(int(step) for step in report.get("steps", {})) == STEPS,
            "measured step registry changed")
    for step in STEPS:
        block = report["steps"][str(step)]
        require(tuple(block["rows"]) == shared.ROW_ORDER,
                f"kt={step}: row registry changed")
        require(all(block["passivity"].values()),
                f"kt={step}: passive trace changed returned state")
        for name in shared.ROW_ORDER:
            row = block["rows"][name]
            require(row["count"] > 0, f"kt={step} {name}: empty score")
            require(row["first_nonfinite_k"] is not None
                    or (row["max_abs"] is not None
                        and math.isfinite(float(row["max_abs"]))),
                    f"kt={step} {name}: invalid score")
    observed = first_over_floor(report["steps"])
    require(observed == report.get("first_over_floor"),
            "first-over-floor selection is not source ordered")
    any_nonfinite = any(
        report["steps"][str(step)]["rows"][name]["first_nonfinite_k"] is not None
        for step in STEPS for name in shared.ROW_ORDER)
    entry_exact = all(
        report["steps"]["1"]["rows"][name]["bit_exact"]
        for name in ("ssh_entry", "r3t_entry"))
    report["prediction_ledger"] = {
        "R145-P1": {"status": "CONFIRMED"},
        "R145-P2": {"status": "CONFIRMED" if entry_exact else "REFUTED"},
        "R145-P3": {
            "status": ("CONFIRMED" if observed and observed["step"] == 1
                       and observed["row"] == "ssh_after" else "REFUTED"),
            "observed": observed,
        },
        "R145-P4": {"status": "REFUTED" if any_nonfinite else "CONFIRMED"},
        "R145-P5": {"status": "UNMEASURED"},
        "R145-P6": {"status": "CONFIRMED", "observed": "measurement-only"},
    }
    return {**report, "status": "PASS_ROUND145_INITIAL_GROWTH_WALK"}


def measure(deck_root: Path, record_root: Path, admission: Path,
            expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-145 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-145 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "initial-growth walk requires production JIT on CPU")

    card = shared.rung0.build_rung0_card(deck_root)
    shared.rung0.validate_rung0_card(card)
    require(card.unmeasured_features == ("linear_implicit_bottom_drag",),
            "rung-0 unmeasured-feature registry changed")
    freshwater, surface = rung0_ladder._zero_forcing(
        tuple(np.asarray(card.recipe.initial_state.eta.data).shape))
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    traced = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_live_stage_operands=True))

    state = card.recipe.initial_state
    measured: dict[str, object] = {}
    record_census = []
    started = time.time()
    for step in STEPS:
        trace = jax.device_get(traced.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        next_state = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        passivity = shared.state_bit_rows(trace.state_after, next_state)
        require(all(passivity.values()), f"kt={step}: passive live trace moved state")
        record, census = assemble_record(record_root, step)
        measured[str(step)] = {
            "rows": shared.target_rows(record, state, trace, card),
            "passivity": passivity,
        }
        record_census.extend(census)
        state = next_state
        print(f"ROUND145_INITIAL_GROWTH_PROGRESS step={step}/10 "
              f"wall_s={time.time() - started:.1f}", flush=True)

    return {
        "format": "nemo-testcase-l4-orca2-round145-initial-growth-v1",
        "claim_label": "independent", "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "target_ji": list(shared.TARGET), "floor": shared.FLOOR,
        "row_order": list(shared.ROW_ORDER), "admission": validate_admission(admission),
        "record_census": record_census, "steps": measured,
        "first_over_floor": first_over_floor(measured),
        "unmeasured_features": list(card.unmeasured_features),
        "worktree": stamp, "wall_seconds": time.time() - started,
        "compiled_citations": {
            "external_before_stage1":
                "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/stprk3.f90:202-221",
            "slow_forcing":
                "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/stp2d.f90:206-234",
            "split_explicit_entry":
                "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:288-380",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(not any((args.deck_root, args.record_root, args.admission,
                             args.expect_commit)),
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(all((args.deck_root, args.record_root, args.admission,
                         args.expect_commit)),
                    "runtime mode requires deck, record, admission, and commit")
            raw = measure(args.deck_root, args.record_root, args.admission,
                          args.expect_commit)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, shared.GateError, shared.rung0.GateError, OSError,
            KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND145_INITIAL_GROWTH_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
