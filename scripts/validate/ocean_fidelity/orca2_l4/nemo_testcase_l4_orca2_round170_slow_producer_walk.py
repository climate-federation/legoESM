#!/usr/bin/env python3
"""Walk ORCA2 rung-0 kt=8 slow forcing from its recorded 3-D operands."""

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
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round169_slow8_acquisition import (
    check_record,
)


SOURCE_ORDER = (
    "e3_u", "rhs_u", "mask_u", "r1_h0_u", "depth_u",
    "e3_v", "rhs_v", "mask_v", "r1_h0_v", "depth_v",
    "drag_u", "drag_v", "wind_u", "wind_v",
)
PLANTS = (
    "none", "rank-placement", "literal-replay", "rhs-arm",
    "source-order", "zero-wind",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bits(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64).view(np.uint64)


def _score(candidate, oracle, active) -> dict:
    return r167._score(candidate, oracle, active)


def _with_jpk_zero(value) -> np.ndarray:
    """Append NEMO's structural, non-contributing jpk record slot."""

    value = np.asarray(value, dtype=np.float64)
    require(value.ndim == 3 and value.shape[-1] == 30,
            "candidate physical-level count moved")
    return np.concatenate([value, np.zeros_like(value[..., :1])], axis=-1)


def _payload_values(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Parse values from the record's own field headers after admission."""

    metadata = check_record.read_record(path)
    raw = path.read_bytes()
    offset = 84
    values: dict[str, np.ndarray] = {}
    for _ in range(len(check_record.NAMES)):
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        ndim, n1, n2, n3 = struct.unpack_from("=4i", raw, offset)
        offset += 16
        shape = (n1,) if ndim == 1 else ((n1, n2) if ndim == 2 else (n1, n2, n3))
        count = int(np.prod(shape))
        value = np.frombuffer(raw, dtype="=f8", count=count, offset=offset)
        offset += count * 8
        if ndim == 1:
            owned = np.array(value, copy=True)
        else:
            local = value.reshape(shape, order="F")
            owned = np.array(check_record._owned_slice(local, metadata), copy=True)
            owned = owned.transpose(1, 0, 2) if ndim == 3 else owned.T
        values[name] = owned
    require(offset == len(raw), f"{path.name}: value parser did not consume record")
    require(tuple(values) == check_record.NAMES,
            f"{path.name}: value parser field registry moved")
    return metadata, values


def assemble_record(root: Path) -> tuple[dict[str, np.ndarray], dict]:
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, np.ndarray] = {}
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    for rank in (0, 1):
        path = root / f"oracle_r169_slow8_rank{rank:04d}_kt00000008.bin"
        metadata, values = _payload_values(path)
        require(metadata["rank"] == rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = metadata["origin"]
        ntsi, ntsj, ntei, ntej = metadata["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: global placement moved")
        coverage[j0:j1, i0:i1] += 1
        for name, owned in values.items():
            if name in check_record.SCALARS:
                if name in scalars:
                    require(np.array_equal(scalars[name], owned),
                            f"{path.name}: rank-invariant {name} differs")
                else:
                    scalars[name] = owned
                continue
            shape = (148, 180, owned.shape[-1]) if owned.ndim == 3 else (148, 180)
            arrays.setdefault(name, np.empty(shape, dtype=np.float64))[
                j0:j1, i0:i1, ...
            ] = owned
        records.append({key: metadata[key] for key in (
            "rank", "sha256", "bytes", "origin", "owned")})
    require(bool(np.all(coverage == 1)),
            "round-169 rank slabs do not cover the global domain exactly once")
    return {**arrays, **scalars}, {
        "rank_coverage": "exactly-once", "records": records,
        "two_or_three_dimensional_fields": len(arrays),
        "rank_invariant_scalars": len(scalars),
    }


def _source_sum(e3, rhs, mask, reciprocal) -> np.ndarray:
    """Replay stp2d.f90:219-220 without reassociating its scalar SUM."""

    product = (np.asarray(e3) * np.asarray(rhs)) * np.asarray(mask)
    require(product.ndim == 3 and product.shape[-1] == 31,
            "slow-forcing vertical field shape moved")
    total = np.array(product[..., 0], copy=True)
    for level in range(1, product.shape[-1] - 1):
        total = total + product[..., level]
    return total * np.asarray(reciprocal)


def _candidate_reference_operands(card, state):
    """Instantiate the exact reference-mesh operands used by production."""

    import jax.numpy as jnp

    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import compute_face_masks_3d
    from legoesm.ocean.vertical import (
        OceanPartialCellCoordinate,
        compute_layer_thickness,
        nemo_qco_card_mesh_operands,
    )

    z_coord = card.recipe.z_coord
    config = card.recipe.model_config
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, z_coord,
        min_water_column_m=config.min_water_column_m,
    ).astype(state.u.data.dtype)
    require(isinstance(z_coord, OceanPartialCellCoordinate),
            "rung-0 card no longer uses partial cells")
    umask, vmask = compute_face_masks_3d(z_coord.is_active, card.recipe.grid)
    umask = umask.astype(state.u.data.dtype)
    vmask = vmask.astype(state.v.data.dtype)
    ops = nemo_qco_card_mesh_operands(
        h_ref, umask, vmask, card.recipe.grid, state.u.data.dtype)
    one = jnp.asarray(1.0, dtype=state.u.data.dtype)
    wet_u = (ops.hu_0 > 0.0).astype(state.u.data.dtype)
    wet_v = (ops.hv_0 > 0.0).astype(state.v.data.dtype)
    return {
        # ``nemo_qco_card_mesh_operands`` is already native-face shaped;
        # unlike model velocity arrays it has no compact halo to remove.
        "e3_u": _with_jpk_zero(ops.e3u_0),
        "e3_v": _with_jpk_zero(ops.e3v_0),
        "mask_u": _with_jpk_zero(ops.umask3),
        "mask_v": _with_jpk_zero(ops.vmask3),
        "r1_h0_u": np.asarray(wet_u / (ops.hu_0 + one - wet_u)),
        "r1_h0_v": np.asarray(wet_v / (ops.hv_0 + one - wet_v)),
    }


def _trace_at_kt8(card, state, freshwater, surface, hooks):
    import jax

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks._replace(expose_barotropic_substeps=True),
    )
    return jax.device_get(model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))


def classify(report: dict, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "rank-placement":
        report["admission"]["rank_coverage"] = "overlap"
    elif plant == "literal-replay":
        report["recorded_literal_replay"]["u"]["differing_cells"] += 1
        report["recorded_literal_replay"]["u"]["bit_exact"] = False
    elif plant == "rhs-arm":
        report["rhs_override_depth"]["u"]["coverage"] = 0
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "zero-wind":
        report["recorded_post_drag_wind_final_identity"]["u"] = False

    admission = report["admission"]
    require(admission["rank_coverage"] == "exactly-once"
            and len(admission["records"]) == 2,
            "rank-complete slow record admission moved")
    require(len(admission["terminal_restart_comparisons"]) == 20,
            "terminal restart comparison census moved")
    require(tuple(report["source_order"]) == SOURCE_ORDER,
            "compiled slow-forcing source order moved")
    require(all(row["bit_exact"] for row in
                report["recorded_literal_replay"].values()),
            "recorded operands no longer replay NEMO's depth boundary")
    require(all(row["coverage"] > 0 for row in
                report["rhs_override_depth"].values()),
            "one-variable recorded-RHS arm has empty coverage")
    require(all(report["recorded_post_drag_wind_final_identity"].values()),
            "zero-wind/load recorded boundary identity moved")

    first = report["first_nonbit_operand"]
    p3 = (
        first is not None and first["name"] in ("rhs_u", "rhs_v")
        and all(row["bit_exact"] for row in report["rhs_override_depth"].values())
    )
    report["prediction_dispositions"] = {
        "R170-P1": "CONFIRMED",
        "R170-P2": "CONFIRMED",
        "R170-P3": "CONFIRMED" if p3 else "REFUTED",
        "R170-P4": "CONFIRMED" if (
            report["resolved_switches"]["candidate_zero_stress"]
            and all(report["recorded_post_drag_wind_final_identity"].values())
            and not any(report["resolved_switches"][name] for name in (
                "nemo_ln_apr_dyn", "nemo_ln_ice_embd", "nemo_ln_bern_srfc"))
        ) else "REFUTED",
        "R170-P5": "CONFIRMED",
    }
    report["status"] = "PASS_ROUND170_SLOW_PRODUCER_WALK"
    return report


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            baseline_root: Path, expect_commit: str) -> dict:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-170 measurement requires its clean committed instrument")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-170 walk requires production JIT on CPU")

    admission = check_record.run(record_root, baseline_root)
    oracle, census = assemble_record(record_root)
    card, state, freshwater, surface = r166._setup(deck_root, frame_root)
    hooks = r166._hooks(card)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    for kt in range(1, 8):
        state = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        print(f"PROGRESS round170 complete kt={kt}", file=sys.stderr, flush=True)

    trace = _trace_at_kt8(card, state, freshwater, surface, hooks)
    operands = trace.slow_forcing_operands
    candidate = _candidate_reference_operands(card, state)
    candidate.update({
        "rhs_u": _with_jpk_zero(r97._native_u(operands["du_dt"])),
        "rhs_v": _with_jpk_zero(r97._native_v(operands["dv_dt"])),
        "depth_u": r97._native_u(operands["depth_u"]),
        "depth_v": r97._native_v(operands["depth_v"]),
        "drag_u": r97._native_u(operands["post_drag_u"]),
        "drag_v": r97._native_v(operands["post_drag_v"]),
        "wind_u": r97._native_u(operands["post_wind_u"]),
        "wind_v": r97._native_v(operands["post_wind_v"]),
    })

    reference = {
        "e3_u": oracle["e3u"], "rhs_u": oracle["rhs_u"],
        "mask_u": oracle["umask"], "r1_h0_u": oracle["r1_hu0"],
        "depth_u": oracle["depth_u"], "drag_u": oracle["drag_u"],
        "wind_u": oracle["wind_u"], "final_u": oracle["final_u"],
        "e3_v": oracle["e3v"], "rhs_v": oracle["rhs_v"],
        "mask_v": oracle["vmask"], "r1_h0_v": oracle["r1_hv0"],
        "depth_v": oracle["depth_v"], "drag_v": oracle["drag_v"],
        "wind_v": oracle["wind_v"], "final_v": oracle["final_v"],
    }
    active3 = {face: oracle[f"{face}mask"] != 0.0 for face in ("u", "v")}
    active2 = {face: active3[face][..., 0] for face in ("u", "v")}
    rows = []
    for name in SOURCE_ORDER:
        face = name[-1]
        value = candidate[name]
        mask = (np.ones_like(reference[name], dtype=bool)
                if name.startswith("mask_") else
                (active3[face] if reference[name].ndim == 3 else active2[face]))
        require(np.shape(value) == np.shape(reference[name]) == np.shape(mask),
                f"{name} score shape mismatch: candidate={np.shape(value)} "
                f"oracle={np.shape(reference[name])} mask={np.shape(mask)}")
        rows.append({"name": name, **_score(value, reference[name], mask)})

    recorded_replay = {}
    for face in ("u", "v"):
        replay = _source_sum(
            oracle[f"e3{face}"], oracle[f"rhs_{face}"], oracle[f"{face}mask"],
            oracle[f"r1_h{face}0"])
        recorded_replay[face] = _score(replay, oracle[f"depth_{face}"], active2[face])

    rhs_override = (
        oracle["rhs_u"][..., :-1],
        oracle["rhs_v"][..., :-1],
    )
    arm_trace = _trace_at_kt8(
        card, state, freshwater, surface,
        hooks._replace(slow_forcing_rhs_override=rhs_override))
    arm_ops = arm_trace.slow_forcing_operands
    rhs_arm = {
        "u": {**_score(r97._native_u(arm_ops["depth_u"]),
                       oracle["depth_u"], active2["u"]),
              "coverage": int(np.count_nonzero(active2["u"]))},
        "v": {**_score(r97._native_v(arm_ops["depth_v"]),
                       oracle["depth_v"], active2["v"]),
              "coverage": int(np.count_nonzero(active2["v"]))},
    }

    first = next((row for row in rows if not row["bit_exact"]), None)
    output = (record_root / "ocean.output").read_text(errors="replace")
    namelist = (record_root / "namelist_cfg").read_text(errors="replace")
    zero_identity = {
        face: bool(np.array_equal(_bits(oracle[f"drag_{face}"]),
                                  _bits(oracle[f"wind_{face}"]))
                   and np.array_equal(_bits(oracle[f"wind_{face}"]),
                                      _bits(oracle[f"final_{face}"])))
        for face in ("u", "v")
    }
    resolved = {
        "candidate_surface_stress_implicit": bool(
            card.recipe.model_config.surface_stress_implicit),
        "candidate_zero_stress": bool(
            np.count_nonzero(np.asarray(operands["wind_tau_u"])) == 0
            and np.count_nonzero(np.asarray(operands["wind_tau_v"])) == 0),
        "nemo_ln_apr_dyn": "ln_apr_dyn    =  T" in output,
        "nemo_ln_ice_embd": "ln_ice_embd   =  T" in output,
        "nemo_ln_bern_srfc": not bool(
            "ln_bern_srfc= .false." in namelist
            or "ln_bern_srfc = .false." in namelist),
        "candidate_barotropic_drag_substep": bool(
            card.recipe.model_config.barotropic_drag_substep),
        "card_unmeasured_features": list(card.unmeasured_features),
    }
    raw = {
        "format": "nemo-testcase-l4-orca2-round170-slow-producer-v1",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "admission": admission,
        "assembled_record": census,
        "completed_kt": 7,
        "kt": 8,
        "source_order": list(SOURCE_ORDER),
        "rows": rows,
        "first_nonbit_operand": first,
        "recorded_literal_replay": recorded_replay,
        "rhs_override_depth": rhs_arm,
        "recorded_post_drag_wind_final_identity": zero_identity,
        "resolved_switches": resolved,
        "worktree": stamp,
    }
    return classify(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
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
                    "measurement requires deck/frame/record/baseline roots "
                    "and commit")
            result = measure(
                args.deck_root, args.frame_root, args.record_root,
                args.baseline_root, args.expect_commit)
        else:
            require(args.report_in is not None, "classification requires --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, UnicodeDecodeError, ValueError, KeyError, GateError,
            check_record.Refusal, r166.GateError, r167.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_ROUND170_SLOW_PRODUCER_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
