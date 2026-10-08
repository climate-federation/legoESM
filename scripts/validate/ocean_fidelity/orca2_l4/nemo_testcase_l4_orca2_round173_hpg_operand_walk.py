#!/usr/bin/env python3
"""Walk the independent ORCA2 kt=8 HPG operands and statements."""

from __future__ import annotations

import argparse
import copy
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round42_stage1_hpg_walk_gate as r42,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round93_rhs_walk as r93,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round166_external_substep_gate as r166,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round172_passive_rhs_replay as r172,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round172_hpg8_acquisition import (
    check_record,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)


INPUT_ORDER = ("rhd", "e3w", "gdept_z0", "r1_e1u", "r1_e2v")
STATEMENT_ORDER = ("zhpi_u", "zhpi_v", "zuap_u", "zuap_v", "sum_u", "sum_v")
WALK_ORDER = INPUT_ORDER + STATEMENT_ORDER
ABS_EXPLOSIVE = np.float64(1.0e20)
REL_EXPLOSIVE = np.float64(1.0e12)
PLANTS = (
    "none", "rank-placement", "self-replay", "first-boundary",
    "explosive-classification",
)


class GateError(RuntimeError):
    """A record, instrument, or frozen selector prerequisite moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _read_arrays(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read payloads using the record's own field headers."""

    metadata = check_record.read_record(path)
    raw = path.read_bytes()
    offset = 80
    arrays: dict[str, np.ndarray] = {}
    for expected in check_record.NAMES:
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        ndim, n1, n2, n3 = struct.unpack_from("=4i", raw, offset)
        offset += 16
        require(name == expected, f"{path.name}: field registry moved")
        require(ndim in (2, 3), f"{path.name}: invalid rank for {name}")
        count = n1 * n2 * n3
        end = offset + count * 8
        require(end <= len(raw), f"{path.name}: truncated {name}")
        local = np.frombuffer(raw, dtype="=f8", count=count, offset=offset)
        local = local.reshape((n1, n2, n3), order="F")
        ntsi, ntsj, ntei, ntej = metadata["owned"]
        owned = local[ntsi - 1:ntei, ntsj - 1:ntej, :].transpose(1, 0, 2)
        arrays[name] = np.array(owned[..., 0] if ndim == 2 else owned, copy=True)
        offset = end
    require(offset == len(raw), f"{path.name}: trailing payload")
    return metadata, arrays


def assemble_record(root: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Assemble the two owned 90-column slabs into global arrays."""

    arrays = {
        name: np.empty(
            (148, 180, 31) if name in check_record.THREE_D else (148, 180),
            dtype=np.float64,
        )
        for name in check_record.NAMES
    }
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    for rank in (0, 1):
        path = root / f"oracle_r172_hpg8_rank{rank:04d}_kt00000008.bin"
        metadata, values = _read_arrays(path)
        require(metadata["rank"] == rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = metadata["origin"]
        ntsi, ntsj, ntei, ntej = metadata["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: global placement moved")
        coverage[j0:j1, i0:i1] += 1
        for name, owned in values.items():
            expected = (148, 90, 31) if name in check_record.THREE_D else (148, 90)
            require(owned.shape == expected, f"{path.name}: {name} shape moved")
            arrays[name][j0:j1, i0:i1, ...] = owned
        records.append({
            "rank": rank, "sha256": metadata["sha256"],
            "origin": metadata["origin"], "owned": metadata["owned"],
        })
    require(bool(np.all(coverage == 1)), "rank slabs are not exactly-once")
    return arrays, {"rank_coverage": "exactly-once", "records": records}


def _explosive(row: dict[str, object]) -> bool:
    return bool(
        np.float64(row["absolute_max"]) >= ABS_EXPLOSIVE
        and np.float64(row["candidate_max_abs"])
        >= REL_EXPLOSIVE * max(np.float64(row["reference_max_abs"]), np.float64(1.0))
    )


def _first_nonbit(rows: dict[str, dict[str, object]]) -> dict[str, object] | None:
    for name in WALK_ORDER:
        if not rows[name]["bit_exact"]:
            return {"boundary": name, **rows[name]}
    return None


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    """Apply the frozen selectors and known-answer controls."""

    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "rank-placement":
        report["admission"]["rank_coverage"] = "overlap"
    elif plant == "self-replay":
        report["recorded_self_replay"]["sum_u"]["bit_exact"] = False
    elif plant == "first-boundary":
        first = report["first_nonbit"]
        require(first is not None, "first-boundary plant has no selector")
        report["rows"][first["boundary"]]["bit_exact"] = True
    elif plant == "explosive-classification":
        first = report["first_nonbit"]
        require(first is not None, "explosive plant has no selector")
        report["rows"][first["boundary"]]["explosive"] = not report["rows"][
            first["boundary"]]["explosive"]

    require(report["admission"]["rank_coverage"] == "exactly-once"
            and len(report["admission"]["records"]) == 2,
            "rank-complete HPG record admission moved")
    require(all(row["bit_exact"] for row in report["recorded_self_replay"].values()),
            "recorded operands do not reproduce the recorded HPG statements")
    require(report["candidate_literal_matches_live_hpg"]["u"]["bit_exact"]
            and report["candidate_literal_matches_live_hpg"]["v"]["bit_exact"],
            "candidate operand graph does not reproduce the live HPG boundary: "
            f"{report['candidate_literal_matches_live_hpg']}")
    require(report["one_ulp_control"]["differing_cells"] == 1
            and not report["one_ulp_control"]["bit_exact"],
            "one-ULP known-answer control did not fire")
    for name in WALK_ORDER:
        require(report["rows"][name]["explosive"] == _explosive(report["rows"][name]),
                f"{name} explosive classification moved")
    first = _first_nonbit(report["rows"])
    require(first == report["first_nonbit"], "first non-bit selector moved")

    first_name = None if first is None else first["boundary"]
    arm = report["recorded_rhd_arm"]
    arm_confirmed = bool(
        arm["u"]["candidate_max_abs"] < ABS_EXPLOSIVE
        and arm["v"]["candidate_max_abs"] < ABS_EXPLOSIVE
        and arm["u_improvement_factor"] >= REL_EXPLOSIVE
        and arm["v_improvement_factor"] >= REL_EXPLOSIVE
    )
    report["prediction_dispositions"] = {
        "R173-P1": "CONFIRMED",
        "R173-P2": "CONFIRMED",
        "R173-P3": "CONFIRMED" if first_name == "rhd" and first["explosive"] else "REFUTED",
        "R173-P4": "CONFIRMED" if arm_confirmed else "REFUTED",
        "R173-P5": "CONFIRMED",
    }
    report["status"] = "PASS_ROUND173_HPG_OPERAND_WALK"
    return report


def _native(value: np.ndarray, face: str) -> np.ndarray:
    return np.asarray(value)[:, 1:, :] if face == "u" else np.asarray(value)[1:, :, :]


def _factor(before: float, after: float) -> float:
    if after == 0.0:
        return float("inf") if before > 0.0 else 1.0
    return float(before / after)


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            baseline_root: Path, expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-173 measurement requires its clean committed instrument")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-173 walk requires production JIT on CPU")

    admission = check_record.run(record_root, baseline_root)
    oracle, census = assemble_record(record_root)
    card, state, freshwater, surface = r166._setup(deck_root, frame_root)
    hooks = r166._hooks(card)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    offline = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    offline.prime_step_caches(state)
    for kt in range(1, 8):
        state = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        print(f"PROGRESS round173 complete kt={kt}", file=sys.stderr, flush=True)

    candidate_inputs = r42._candidate_inputs(offline, state)
    candidate_literal = r42._literal_from_inputs(
        candidate_inputs, offline.config.g, grid=offline.grid)
    component_call = jax.jit(lambda source_state, forcing: r172._stage1_tendency(
        offline, source_state, forcing, card.dt_s, components=True))
    _total, _diagnostics, live_parts = jax.device_get(component_call(state, surface))

    masks = phase3_gate.expected_masks(card)
    tmask = np.asarray(masks["T"], dtype=bool)
    umask = np.asarray(masks["u"], dtype=bool)
    vmask = np.asarray(masks["v"], dtype=bool)
    face_masks = {"u": umask, "v": vmask}
    metric_masks = {"u": np.any(umask, axis=-1), "v": np.any(vmask, axis=-1)}

    recorded_inputs = {name: oracle[name][..., :30] if oracle[name].ndim == 3
                       else oracle[name] for name in INPUT_ORDER}
    recorded_replay = r42._literal_from_inputs(recorded_inputs, offline.config.g)
    self_replay = {}
    for name in STATEMENT_ORDER:
        face = name[-1]
        self_replay[name] = r93.score(
            _native(recorded_replay[name], face), oracle[name][..., :30],
            face_masks[face])

    live_identity = {}
    for face in ("u", "v"):
        live_identity[face] = r93.score(
            _native(candidate_literal[f"sum_{face}"], face),
            _native(np.asarray(live_parts[f"hpg_{face}"].data), face),
            face_masks[face])

    rows: dict[str, dict[str, object]] = {}
    for name in ("rhd", "e3w", "gdept_z0"):
        rows[name] = r93.score(
            candidate_inputs[name], recorded_inputs[name], tmask)
    rows["r1_e1u"] = r93.score(
        candidate_inputs["r1_e1u"], recorded_inputs["r1_e1u"], metric_masks["u"])
    rows["r1_e2v"] = r93.score(
        candidate_inputs["r1_e2v"], recorded_inputs["r1_e2v"], metric_masks["v"])
    for name in STATEMENT_ORDER:
        face = name[-1]
        rows[name] = r93.score(
            _native(candidate_literal[name], face), oracle[name][..., :30],
            face_masks[face])
    for row in rows.values():
        row["explosive"] = _explosive(row)

    rhd_inputs = dict(candidate_inputs)
    rhd_inputs["rhd"] = recorded_inputs["rhd"]
    rhd_literal = r42._literal_from_inputs(
        rhd_inputs, offline.config.g, grid=offline.grid)
    rhd_arm = {}
    for face in ("u", "v"):
        row = r93.score(
            _native(rhd_literal[f"sum_{face}"], face), oracle[f"sum_{face}"][..., :30],
            face_masks[face])
        baseline_error = rows[f"sum_{face}"]["absolute_max"]
        rhd_arm[face] = row
        rhd_arm[f"{face}_improvement_factor"] = _factor(
            baseline_error, row["absolute_max"])

    synthetic = np.zeros((2, 2, 2), dtype=np.float64)
    planted = synthetic.copy()
    planted[0, 0, 0] = np.nextafter(0.0, np.float64(np.inf))
    control = r93.score(planted, synthetic, np.ones_like(synthetic, dtype=bool))
    raw = {
        "format": "nemo-testcase-l4-orca2-round173-hpg-operand-walk-v1",
        "claim_label": "independent",
        "execution": {
            "backend": jax.default_backend(), "dtype": "float64",
            "transcendentals": get_policy().transcendentals,
        },
        "worktree": stamp,
        "admission": admission,
        "assembled_record": census,
        "completed_kt": 7,
        "kt": 8,
        "walk_order": list(WALK_ORDER),
        "recorded_self_replay": self_replay,
        "candidate_literal_matches_live_hpg": live_identity,
        "rows": rows,
        "first_nonbit": _first_nonbit(rows),
        "recorded_rhd_arm": rhd_arm,
        "thresholds": {"absolute": float(ABS_EXPLOSIVE),
                       "relative": float(REL_EXPLOSIVE)},
        "one_ulp_control": control,
    }
    return classify(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.baseline_root, args.expect_commit)),
                    "measurement requires all record roots and commit")
            result = measure(args.deck_root, args.frame_root, args.record_root,
                             args.baseline_root, args.expect_commit)
        else:
            require(args.report_in is not None, "classification requires --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError,
            GateError, check_record.Refusal, r166.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_ROUND173_HPG_OPERAND_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
