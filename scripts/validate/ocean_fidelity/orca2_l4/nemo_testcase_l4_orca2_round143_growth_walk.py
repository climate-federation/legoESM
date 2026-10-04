#!/usr/bin/env python3
"""Compare rung-0 NEMO/legoESM growth boundaries at steps 30 through 36."""

from __future__ import annotations

import argparse
import hashlib
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
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)


STEPS = tuple(range(30, 37))
TARGET = (87, 159)
FLOOR = 2.0e-10
ADMISSION_SHA256 = "0e455a46fe3a9a506eb9ad301ca44a9b6d9181d7712d1f881c22a6462b0fd18c"
ROW_ORDER = (
    "ssh_entry", "r3t_entry",
    "uub_entry_w", "uub_entry_e", "vvb_entry_s", "vvb_entry_n",
    "ssh_after", "r3t_after",
    "uub_after_w", "uub_after_e", "vvb_after_s", "vvb_after_n",
    "un_adv_w", "un_adv_e", "vn_adv_s", "vn_adv_n",
    "r3t_stage1",
    "zFu_stage1_w", "zFu_stage1_e",
    "zFv_stage1_s", "zFv_stage1_n", "zFw_stage1",
)
PLANTS = ("none", "admission", "passivity", "source-order")


class GateError(RuntimeError):
    """The growth record or source-ordered comparison violated its contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_admission(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing admission JSON: {path}")
    require(sha256(path) == ADMISSION_SHA256, "round-141 admission digest changed")
    report = json.loads(path.read_text())
    require(report.get("status") == "PASS_R141_GROWTH_RECORD",
            "growth record admission is not PASS")
    require(report.get("claim_label") == "independent",
            "growth record is not independent")
    require(tuple(report.get("steps", ())) == STEPS,
            "growth record step registry changed")
    require(report.get("rank_coverage") == "exactly-once per step",
            "growth record rank coverage changed")
    require(len(report.get("records", ())) == 2 * len(STEPS),
            "growth record rank/step census changed")
    require(len(report.get("calibration_restart_comparisons", ())) == 20,
            "growth record calibration census changed")
    return {
        "path": str(path), "sha256": sha256(path),
        "status": report["status"], "rank_step_records": len(report["records"]),
        "calibration_restart_comparisons": len(
            report["calibration_restart_comparisons"]),
    }


def _payload(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Extract arrays only after the committed admission parser accepts them."""

    from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round141_growth_acquisition import (
        check_record,
    )

    metadata = check_record.read_record(path)
    raw = path.read_bytes()
    header = check_record.HEADER.unpack_from(raw, 16)
    nfields = header[-1]
    offset = 16 + check_record.HEADER.size
    arrays: dict[str, np.ndarray] = {}
    for _ in range(nfields):
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
    from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round141_growth_acquisition import (
        check_record,
    )

    assembled = {
        name: np.empty((148, 180) if name in check_record.FIELDS_2D
                       else (148, 180, 31), dtype=np.float64)
        for name in check_record.FIELDS
    }
    coverage = np.zeros((148, 180), dtype=np.int8)
    rows = []
    for rank in (0, 1):
        path = root / f"oracle_r141_growth_rank{rank:04d}_kt{step:08d}.bin"
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


def _bits_equal(left, right) -> bool:
    a = np.ascontiguousarray(np.asarray(left))
    b = np.ascontiguousarray(np.asarray(right))
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def state_bit_rows(left, right) -> dict[str, bool]:
    fields_left = rung0.candidate_fields(left)
    fields_right = rung0.candidate_fields(right)
    rows = {name: _bits_equal(fields_left[name], fields_right[name])
            for name in rung0.FIELDS}
    rows["uu_b"] = _bits_equal(left.uu_b.data, right.uu_b.data)
    rows["vv_b"] = _bits_equal(left.vv_b.data, right.vv_b.data)
    return rows


def score_values(lego, nemo) -> dict[str, object]:
    left = np.asarray(lego, dtype=np.float64).reshape(-1)
    right = np.asarray(nemo, dtype=np.float64).reshape(-1)
    require(left.shape == right.shape and left.size > 0, "row shape mismatch")
    finite = np.isfinite(left) & np.isfinite(right)
    first_nonfinite = next((int(i) for i in range(left.size) if not finite[i]), None)
    if bool(finite.any()):
        delta = np.abs(left[finite] - right[finite])
        finite_indices = np.flatnonzero(finite)
        max_position = int(np.argmax(delta))
        argmax = int(finite_indices[max_position])
        max_abs = float(delta[max_position])
    else:
        argmax, max_abs = None, None
    exact = _bits_equal(left, right)
    return {
        "count": int(left.size), "bit_exact": exact,
        "max_abs": max_abs, "argmax_k": argmax,
        "lego_at_argmax": None if argmax is None else float(left[argmax]),
        "nemo_at_argmax": None if argmax is None else float(right[argmax]),
        "first_nonfinite_k": first_nonfinite,
        "over_floor": (first_nonfinite is not None
                       or (max_abs is not None and max_abs > FLOOR)),
    }


def target_rows(record: dict[str, np.ndarray], state, trace, card) -> dict[str, dict]:
    import jax.numpy as jnp
    from legoesm.ocean.eos import nemo_r3t_rk3_stage1_stretch, nemo_r3t_stretch

    j, i = TARGET
    entry_eta = np.asarray(state.eta.data)
    after_eta = np.asarray(trace.barotropic_targets[4])
    depth = np.asarray(state.H_bathy.data)
    entry_r3t = np.asarray(nemo_r3t_stretch(
        card.recipe.z_coord, jnp.asarray(entry_eta), jnp.asarray(depth),
        evaluation="nemo_reciprocal")) - 1.0
    after_r3t = np.asarray(nemo_r3t_stretch(
        card.recipe.z_coord, jnp.asarray(after_eta), jnp.asarray(depth),
        evaluation="nemo_reciprocal")) - 1.0
    stage1_r3t = np.asarray(nemo_r3t_rk3_stage1_stretch(
        card.recipe.z_coord, jnp.asarray(entry_eta), jnp.asarray(after_eta),
        jnp.asarray(depth))) - 1.0

    entry_u = np.asarray(state.uu_b.data)[:, 1:]
    entry_v = np.asarray(state.vv_b.data)[1:, :]
    after_u = np.asarray(trace.barotropic_targets[0])[:, 1:]
    after_v = np.asarray(trace.barotropic_targets[1])[1:, :]
    un_adv = np.asarray(trace.barotropic_targets[2])[:, 1:]
    vn_adv = np.asarray(trace.barotropic_targets[3])[1:, :]
    geometry = trace.stage_geometry[0]
    zfu = np.asarray(geometry[7])[:, 1:, :]
    zfv = np.asarray(geometry[8])[1:, :, :]
    zfw = np.asarray(geometry[2]) * np.asarray(card.recipe.grid.area)[..., None]

    pairs = {
        "ssh_entry": (entry_eta[j, i], record["ssh_entry"][j, i]),
        "r3t_entry": (entry_r3t[j, i], record["r3t_entry"][j, i]),
        "uub_entry_w": (entry_u[j, i - 1], record["uub_entry"][j, i - 1]),
        "uub_entry_e": (entry_u[j, i], record["uub_entry"][j, i]),
        "vvb_entry_s": (entry_v[j - 1, i], record["vvb_entry"][j - 1, i]),
        "vvb_entry_n": (entry_v[j, i], record["vvb_entry"][j, i]),
        "ssh_after": (after_eta[j, i], record["ssh_after"][j, i]),
        "r3t_after": (after_r3t[j, i], record["r3t_after"][j, i]),
        "uub_after_w": (after_u[j, i - 1], record["uub_after"][j, i - 1]),
        "uub_after_e": (after_u[j, i], record["uub_after"][j, i]),
        "vvb_after_s": (after_v[j - 1, i], record["vvb_after"][j - 1, i]),
        "vvb_after_n": (after_v[j, i], record["vvb_after"][j, i]),
        "un_adv_w": (un_adv[j, i - 1], record["un_adv"][j, i - 1]),
        "un_adv_e": (un_adv[j, i], record["un_adv"][j, i]),
        "vn_adv_s": (vn_adv[j - 1, i], record["vn_adv"][j - 1, i]),
        "vn_adv_n": (vn_adv[j, i], record["vn_adv"][j, i]),
        "r3t_stage1": (stage1_r3t[j, i], record["r3t_stage1"][j, i]),
        "zFu_stage1_w": (zfu[j, i - 1], record["zFu_stage1"][j, i - 1]),
        "zFu_stage1_e": (zfu[j, i], record["zFu_stage1"][j, i]),
        "zFv_stage1_s": (zfv[j - 1, i], record["zFv_stage1"][j - 1, i]),
        "zFv_stage1_n": (zfv[j, i], record["zFv_stage1"][j, i]),
        "zFw_stage1": (zfw[j, i], record["zFw_stage1"][j, i]),
    }
    require(tuple(pairs) == ROW_ORDER, "target row source order changed")
    return {name: score_values(*pairs[name]) for name in ROW_ORDER}


def first_over_floor(steps: dict[str, object]) -> dict[str, object] | None:
    for step in STEPS:
        for name in ROW_ORDER:
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
        report["steps"]["30"]["passivity"]["T"] = False
    elif plant == "source-order":
        report["row_order"][0], report["row_order"][1] = (
            report["row_order"][1], report["row_order"][0])

    require(report.get("claim_label") == "independent"
            and report.get("initial_mode") == "card_own_state",
            "growth walk is not independent")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy changed")
    require(report.get("target_ji") == list(TARGET), "target column changed")
    require(float(report.get("floor")) == FLOOR, "comparison floor changed")
    require(tuple(report.get("row_order", ())) == ROW_ORDER,
            "source row order changed")
    require(report.get("admission", {}).get("status") == "PASS_R141_GROWTH_RECORD",
            "growth record is not admitted")
    require(tuple(int(step) for step in report.get("steps", {})) == STEPS,
            "measured step registry changed")
    for step in STEPS:
        block = report["steps"][str(step)]
        require(tuple(block["rows"]) == ROW_ORDER,
                f"kt={step}: row registry changed")
        require(all(block["passivity"].values()),
                f"kt={step}: passive trace changed the returned state")
        for name in ROW_ORDER:
            row = block["rows"][name]
            require(row["count"] > 0, f"kt={step} {name}: empty score")
            require(row["first_nonfinite_k"] is not None
                    or (row["max_abs"] is not None
                        and math.isfinite(float(row["max_abs"]))),
                    f"kt={step} {name}: invalid score")
    observed = first_over_floor(report["steps"])
    require(observed == report.get("first_over_floor"),
            "first-over-floor selection is not source ordered")

    step36_nonfinite = any(
        report["steps"]["36"]["rows"][name]["first_nonfinite_k"] is not None
        for name in ROW_ORDER)
    earlier_nonfinite = any(
        report["steps"][str(step)]["rows"][name]["first_nonfinite_k"] is not None
        for step in STEPS[:-1] for name in ROW_ORDER)
    prediction = observed or {}
    thickness_first = str(prediction.get("row", "")).startswith(("ssh", "r3t"))
    report["prediction_ledger"] = {
        "R143-P1": {"status": "CONFIRMED"},
        "R143-P2": {
            "status": "CONFIRMED" if step36_nonfinite and not earlier_nonfinite else "REFUTED",
            "step36_nonfinite": step36_nonfinite,
            "earlier_nonfinite": earlier_nonfinite,
        },
        "R143-P3": {
            "status": ("CONFIRMED" if prediction.get("step") == 30
                       and prediction.get("row") == "ssh_entry" else "REFUTED"),
            "observed": observed,
        },
        "R143-P4": {
            "status": "CONFIRMED" if thickness_first else "REFUTED",
            "observed": prediction.get("row"),
        },
        "R143-P5": {"status": "CONFIRMED", "observed": "measurement-only"},
    }
    return {**report, "status": "PASS_ROUND143_GROWTH_WALK"}


def measure(deck_root: Path, record_root: Path, admission: Path,
            expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-143 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-143 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "growth walk requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
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
    for step in range(1, 37):
        if step < STEPS[0]:
            state = jax.device_get(ordinary.step(
                state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        else:
            trace = jax.device_get(traced.step(
                state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
            next_state = jax.device_get(ordinary.step(
                state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
            passivity = state_bit_rows(trace.state_after, next_state)
            record, census = assemble_record(record_root, step)
            measured[str(step)] = {
                "rows": target_rows(record, state, trace, card),
                "passivity": passivity,
            }
            record_census.extend(census)
            state = next_state
        if step % 5 == 0 or step >= STEPS[0]:
            print(f"ROUND143_GROWTH_PROGRESS step={step}/36 "
                  f"wall_s={time.time() - started:.1f}", flush=True)

    report = {
        "format": "nemo-testcase-l4-orca2-round143-growth-walk-v1",
        "claim_label": "independent", "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "target_ji": list(TARGET), "floor": FLOOR,
        "row_order": list(ROW_ORDER), "admission": validate_admission(admission),
        "record_census": record_census, "steps": measured,
        "first_over_floor": first_over_floor(measured),
        "unmeasured_features": list(card.unmeasured_features),
        "worktree": stamp, "wall_seconds": time.time() - started,
        "compiled_citations": {
            "external_before_stage1":
                "ORCA2_OMIP_L4_R141GROWTH/BLD/ppsrc/nemo/stprk3.f90:210-222",
            "stage1_transport":
                "ORCA2_OMIP_L4_R141GROWTH/BLD/ppsrc/nemo/traadv.f90:259-273",
        },
    }
    return report


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
    except (GateError, rung0.GateError, OSError, KeyError, TypeError,
            ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_ROUND143_GROWTH_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
