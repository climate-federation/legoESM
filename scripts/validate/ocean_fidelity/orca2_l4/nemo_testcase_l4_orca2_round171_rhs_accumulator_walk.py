#!/usr/bin/env python3
"""Walk the independent ORCA2 kt=8 stage-1 RHS accumulators in source order."""

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
    nemo_testcase_l4_orca2_round166_external_substep_gate as r166,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round167_exit_depth_walk as r167,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round93_rhs_walk as r93,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round170_rhs8_acquisition import (
    check_record,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round83_slow_forcing_walk as r83,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round84_rhs_walk as r84,
)


BOUNDARIES = ("after_hpg", "after_ldf", "after_vor", "after_keg", "after_zad")
COMPONENTS = ("hpg", "ldf", "vorticity", "keg", "zad")
FACES = ("u", "v")
ABS_EXPLOSIVE = np.float64(1.0e20)
REL_EXPLOSIVE = np.float64(1.0e12)
PLANTS = ("none", "rank-placement", "source-order", "observer-closure",
          "explosive-classification", "passivity")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _owned_values(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Parse payloads from the record's own self-describing headers."""

    metadata = check_record.read_record(path)
    raw = path.read_bytes()
    offset = 80
    values: dict[str, np.ndarray] = {}
    for expected in check_record.NAMES:
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        ndim, n1, n2, n3 = struct.unpack_from("=4i", raw, offset)
        offset += 16
        require(name == expected and ndim == 3,
                f"{path.name}: self-described field registry moved")
        count = n1 * n2 * n3
        end = offset + count * 8
        require(end <= len(raw), f"{path.name}: truncated {name}")
        local = np.frombuffer(raw, dtype="=f8", count=count, offset=offset)
        local = local.reshape((n1, n2, n3), order="F")
        ntsi, ntsj, ntei, ntej = metadata["owned"]
        owned = local[ntsi - 1:ntei, ntsj - 1:ntej, :].transpose(1, 0, 2)
        values[name] = np.array(owned, copy=True)
        offset = end
    require(offset == len(raw), f"{path.name}: trailing payload")
    return metadata, values


def assemble_record(root: Path) -> tuple[dict[str, np.ndarray], dict]:
    arrays = {
        name: np.empty((148, 180, 30), dtype=np.float64)
        for name in check_record.NAMES
    }
    coverage = np.zeros((148, 180), dtype=np.int8)
    bottom_nonzero = 0
    rows = []
    for rank in (0, 1):
        path = root / f"oracle_r170_rhs8_rank{rank:04d}_kt00000008.bin"
        metadata, values = _owned_values(path)
        require(metadata["rank"] == rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = metadata["origin"]
        ntsi, ntsj, ntei, ntej = metadata["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: global placement moved")
        coverage[j0:j1, i0:i1] += 1
        for name, owned in values.items():
            require(owned.shape == (148, 90, 31),
                    f"{path.name}: {name} owned shape moved")
            bottom_nonzero += int(np.count_nonzero(owned[..., 30] != 0.0))
            arrays[name][j0:j1, i0:i1] = owned[..., :30]
        rows.append({key: metadata[key] for key in
                     ("rank", "sha256", "bytes", "origin", "owned")})
    require(bool(np.all(coverage == 1)),
            "round-170 rank slabs do not cover the domain exactly once")
    require(bottom_nonzero == 0, "round-170 structural jpk slots are nonzero")
    return arrays, {
        "rank_coverage": "exactly-once", "records": rows,
        "jpk_nonzero": bottom_nonzero,
    }


def _explosive(row: dict) -> bool:
    candidate = np.float64(row["candidate_max_abs"])
    reference = np.float64(row["reference_max_abs"])
    return bool(candidate >= ABS_EXPLOSIVE
                and candidate >= REL_EXPLOSIVE * max(reference, np.float64(1.0)))


def _first_explosive(rows: dict, face: str) -> dict | None:
    previous = False
    for boundary in BOUNDARIES:
        current = bool(rows[face][boundary]["explosive"])
        if current and not previous:
            return {"boundary": boundary, "face": face, **rows[face][boundary]}
        previous = current
    return None


def _first_nonbit(rows: dict) -> dict | None:
    for boundary in BOUNDARIES:
        for face in FACES:
            if not rows[face][boundary]["bit_exact"]:
                return {"boundary": boundary, "face": face,
                        **rows[face][boundary]}
    return None


def classify(report: dict, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "rank-placement":
        report["admission"]["rank_coverage"] = "overlap"
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "observer-closure":
        report["observer_passivity"]["kt8_barotropic"][
            "completed_rhs_u"] = False
    elif plant == "explosive-classification":
        report["rows"]["u"]["after_hpg"]["explosive"] = not report[
            "rows"]["u"]["after_hpg"]["explosive"]
    elif plant == "passivity":
        report["observer_passivity"]["kt1_to_7_barotropic"]["1"]["T"] = False

    require(report["admission"]["rank_coverage"] == "exactly-once"
            and len(report["admission"]["records"]) == 2,
            "rank-complete RHS record admission moved")
    require(tuple(report["source_order"]) == BOUNDARIES,
            "compiled accumulator source order moved")
    require(all(all(fields.values()) for fields in
                report["observer_passivity"]["kt1_to_7_barotropic"].values()),
            "component observer moved a kt=1..7 barotropic boundary")
    require(all(report["observer_passivity"]["kt8_barotropic"].values()),
            "component observer moved the kt=8 barotropic boundary")
    for face in FACES:
        for boundary in BOUNDARIES:
            row = report["rows"][face][boundary]
            require(row["explosive"] == _explosive(row),
                    f"{face} {boundary} explosive classification moved")
            require(not row["reference_explosive"],
                    f"NEMO {face} {boundary} is explosive")

    first_u = _first_explosive(report["rows"], "u")
    first_nonbit = _first_nonbit(report["rows"])
    report["first_explosive_u"] = first_u
    report["first_nonbit_accumulator"] = first_nonbit
    report["prediction_dispositions"] = {
        "R171-P1": "CONFIRMED",
        "R171-P2": "REFUTED",
        "R171-P3": "CONFIRMED" if (
            first_u is not None and first_u["boundary"] == "after_vor")
            else "REFUTED",
        "R171-P4": "CONFIRMED",
        "R171-P5": "CONFIRMED",
    }
    report["status"] = "PASS_ROUND171_RHS_ACCUMULATOR_WALK"
    return report


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            baseline_root: Path, expect_commit: str) -> dict:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-171 measurement requires its clean committed instrument")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-171 walk requires production JIT on CPU")

    admission = check_record.run(record_root, baseline_root)
    oracle, census = assemble_record(record_root)
    card, state, freshwater, surface = r166._setup(deck_root, frame_root)
    hooks = r166._hooks(card)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    plain_baro = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks._replace(expose_barotropic_substeps=True))
    observed_baro = {
        component: LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks._replace(
                expose_barotropic_rhs_component=component))
        for component in COMPONENTS
    }

    passivity = {}
    for kt in range(1, 8):
        next_state = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        plain_prefix = jax.device_get(plain_baro.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        prefix_rows = {}
        for component, model in observed_baro.items():
            observed_prefix = jax.device_get(model.step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
            state_rows = r166._state_rows(
                observed_prefix.state_after_barotropic,
                plain_prefix.state_after_barotropic)
            prefix_rows.update({
                f"{component}:{name}": exact
                for name, exact in state_rows.items()
            })
            for face in FACES:
                observed_rhs = (
                    r83.native_u(observed_prefix.slow_forcing_operands["du_dt"])
                    if face == "u" else
                    r83.native_v(observed_prefix.slow_forcing_operands["dv_dt"]))
                plain_rhs = (
                    r83.native_u(plain_prefix.slow_forcing_operands["du_dt"])
                    if face == "u" else
                    r83.native_v(plain_prefix.slow_forcing_operands["dv_dt"]))
                prefix_rows[f"{component}:completed_rhs_{face}"] = bool(
                    np.array_equal(observed_rhs, plain_rhs))
        passivity[str(kt)] = prefix_rows
        state = next_state
        print(f"PROGRESS round171 complete kt={kt}", file=sys.stderr, flush=True)

    plain_trace = jax.device_get(plain_baro.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    observed_traces = {
        component: jax.device_get(model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        for component, model in observed_baro.items()
    }
    component = {
        f"{name}_{face}": getattr(
            observed_traces[name], f"operator_component_{face}")
        for name in COMPONENTS for face in FACES
    }
    accumulated = jax.device_get(jax.jit(r84.source_order_accumulators)(
        component["hpg_u"], component["hpg_v"],
        component["ldf_u"], component["ldf_v"],
        component["vorticity_u"], component["vorticity_v"],
        component["keg_u"], component["keg_v"],
        component["zad_u"], component["zad_v"],
    ))
    masks = phase3_gate.expected_masks(card)
    active = {face: np.asarray(masks[face], dtype=bool) for face in FACES}
    live = {
        "u": {boundary: r83.native_u(accumulated[f"{boundary}_u"])
              for boundary in BOUNDARIES},
        "v": {boundary: r83.native_v(accumulated[f"{boundary}_v"])
              for boundary in BOUNDARIES},
    }
    rows = {
        face: {
            boundary: r93.score(
                live[face][boundary], oracle[f"{boundary}_{face}"], active[face])
            for boundary in BOUNDARIES
        }
        for face in FACES
    }
    for face in FACES:
        for boundary in BOUNDARIES:
            row = rows[face][boundary]
            row["explosive"] = _explosive(row)
            row["reference_explosive"] = bool(
                np.float64(row["reference_max_abs"]) >= ABS_EXPLOSIVE)

    plain_rhs = {
        "u": r83.native_u(plain_trace.slow_forcing_operands["du_dt"]),
        "v": r83.native_v(plain_trace.slow_forcing_operands["dv_dt"]),
    }
    closure = {
        face: r93.score(live[face]["after_zad"], plain_rhs[face], active[face])
        for face in FACES
    }
    kt8_passivity = {}
    for name, observed_trace in observed_traces.items():
        state_rows = r166._state_rows(
            observed_trace.state_after_barotropic,
            plain_trace.state_after_barotropic)
        kt8_passivity.update({
            f"{name}:{field}": exact for field, exact in state_rows.items()
        })
        for face in FACES:
            observed_rhs = (
                r83.native_u(observed_trace.slow_forcing_operands["du_dt"])
                if face == "u" else
                r83.native_v(observed_trace.slow_forcing_operands["dv_dt"]))
            kt8_passivity[f"{name}:completed_rhs_{face}"] = bool(
                np.array_equal(observed_rhs, plain_rhs[face]))

    synthetic = np.zeros((2, 2, 2), dtype=np.float64)
    planted = synthetic.copy()
    planted[0, 0, 0] = np.nextafter(0.0, np.float64(np.inf))
    control = r93.score(planted, synthetic, np.ones_like(synthetic, dtype=bool))
    require(control["differing_cells"] == 1 and not control["bit_exact"],
            "one-ULP accumulator control did not fire")

    raw = {
        "format": "nemo-testcase-l4-orca2-round171-rhs-accumulator-v1",
        "claim_label": "independent", "completed_kt": 7, "kt": 8,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp, "admission": admission,
        "assembled_record": census, "source_order": list(BOUNDARIES),
        "thresholds": {"absolute": float(ABS_EXPLOSIVE),
                       "relative": float(REL_EXPLOSIVE)},
        "rows": rows, "source_order_closure": closure,
        "observer_passivity": {
                               "full_step_prediction": "REFUTED_IN_PRIOR_RUN",
                               "kt1_to_7_barotropic": passivity,
                               "kt8_barotropic": kt8_passivity},
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
            require(args.report_in is not None,
                    "classification requires --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError, GateError,
            check_record.Refusal, r166.GateError, r167.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_ROUND171_RHS_ACCUMULATOR_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
