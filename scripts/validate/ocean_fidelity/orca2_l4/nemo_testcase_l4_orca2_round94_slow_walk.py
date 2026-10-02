#!/usr/bin/env python3
"""Walk the independent ORCA2 rung-0 slow and external boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    nemo_literal_depth_mean,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round93_rhs_walk as rhs_walk,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round93_slow_acquisition import (
    check_record,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round16_slow_forcing as round16,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round83_slow_forcing_walk as round83,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round84_rhs_walk as rhs_common,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)


SOURCE_ORDER = (
    "depth_u", "depth_v", "drag_u", "drag_v", "wind_u", "wind_v",
    "final_u", "final_v", "ssh_rhs", "ssh_after", "ub_after", "vb_after",
)
PLANTS = ("none", "layout", "record-bit", "trace-bit")


class GateError(RuntimeError):
    """The record or measurement no longer supports the registered claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def assemble_slow(root: Path, *, plant: str = "none") -> tuple[dict[str, np.ndarray], dict]:
    """Assemble all self-described owned fields over the global ORCA2 domain."""

    arrays = {
        name: np.empty((148, 180), dtype=np.float64)
        for name in check_record.NAMES
    }
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    for expected_rank in (0, 1):
        path = root / f"oracle_r93_slow_rank{expected_rank:04d}_kt00000001.bin"
        record = check_record.read_record(path, include_owned_values=True)
        require(record["rank"] == expected_rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = record["origin"]
        ntsi, ntsj, ntei, ntej = record["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        if plant == "layout" and expected_rank == 1:
            i0 -= 1
        i1 = i0 + ntei - ntsi + 1
        j1 = j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: global placement moved")
        coverage[j0:j1, i0:i1] += 1
        require(set(record["owned_values"]) == set(check_record.NAMES),
                f"{path.name}: owned field registry moved")
        for name, values in record["owned_values"].items():
            require(values.shape == (148, 90),
                    f"{path.name}: {name} owned shape moved")
            arrays[name][j0:j1, i0:i1] = values
        records.append({
            key: record[key]
            for key in ("rank", "sha256", "bytes", "field_shapes")
        })
    require(bool(np.all(coverage == 1)),
            "rank-owned slow slabs do not cover the domain exactly once")
    return arrays, {"coverage": "exactly-once", "records": records}


def first_nonbit(rows: dict[str, dict]) -> dict | None:
    """Select the first active non-bit row in compiled source order."""

    for name in SOURCE_ORDER:
        if not rows[name]["bit_exact"]:
            return {"boundary": name, **rows[name]}
    return None


def operand_walk(
    root: Path, operands: dict, oracle: dict[str, np.ndarray], masks: dict,
) -> dict:
    """Walk the rank-0 operands and arithmetic of NEMO's vertical average."""

    legacy = round16.read_slow_forcing(
        root / "oracle_slow_forcing_kt00000001.bin", dims=(94, 152, 31))
    result = {}
    for face in ("u", "v"):
        native = round83.native_u if face == "u" else round83.native_v
        candidate = {
            "e3": native(np.asarray(operands[f"h_{face}"]))[:, :90],
            "rhs": native(np.asarray(operands[f"d{face}_dt"]))[:, :90],
            "mask": np.asarray(masks[face], dtype=np.float64)[:, :90],
            "reciprocal": (
                np.float64(1.0)
                / native(np.asarray(operands[f"H_{face}"]))[:, :90]
            ),
            "depth": native(np.asarray(operands[f"depth_{face}"]))[:, :90],
        }
        reference = {
            "e3": np.asarray(legacy[f"e3{face}"]),
            "rhs": np.asarray(legacy[f"krhs_{face}"]),
            "mask": np.asarray(legacy[f"{face}mask"]),
            "reciprocal": np.asarray(legacy[f"r1_h{face}0"]),
            "depth": np.asarray(legacy[f"depth_{face}"]),
        }
        active3 = reference["mask"] != 0.0
        active2 = active3[..., 0]
        cross_record = rhs_walk.score(
            reference["depth"], oracle[f"depth_{face}"][:, :90], active2)
        require(cross_record["bit_exact"],
                f"rank-0 legacy and self-described depth_{face} records differ")
        rows = {
            "e3": rhs_walk.score(candidate["e3"], reference["e3"], active3),
            "rhs": rhs_walk.score(candidate["rhs"], reference["rhs"], active3),
            "mask": rhs_walk.score(candidate["mask"], reference["mask"], active3),
            "reciprocal": rhs_walk.score(
                candidate["reciprocal"], reference["reciprocal"], active2),
            "production_depth": rhs_walk.score(
                candidate["depth"], reference["depth"], active2),
        }
        source_candidate = round83._round140_source_sum(
            candidate["e3"], candidate["rhs"], candidate["mask"],
            candidate["reciprocal"])
        source_oracle = round83._round140_source_sum(
            reference["e3"], reference["rhs"], reference["mask"],
            reference["reciprocal"])
        recorded_reciprocal_arm = round83._round140_source_sum(
            candidate["e3"], candidate["rhs"], candidate["mask"],
            reference["reciprocal"])
        recorded_product_arm = round83._round140_source_sum(
            reference["e3"], reference["rhs"], reference["mask"],
            candidate["reciprocal"])
        arithmetic = {
            "oracle_literal_replay": rhs_walk.score(
                source_oracle, reference["depth"], active2),
            "candidate_literal_replay": rhs_walk.score(
                source_candidate, reference["depth"], active2),
            "recorded_reciprocal_only": rhs_walk.score(
                recorded_reciprocal_arm, reference["depth"], active2),
            "recorded_product_only": rhs_walk.score(
                recorded_product_arm, reference["depth"], active2),
        }
        require(arithmetic["oracle_literal_replay"]["bit_exact"],
                f"rank-0 NEMO depth_{face} record does not replay its source")
        result[face] = {
            "cross_record_depth": cross_record,
            "rows": rows,
            "arithmetic_arms": arithmetic,
        }
    return result


def measure(
    deck_root: Path, record_root: Path, expect_commit: str, *, plant: str,
) -> dict:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_seed_depth_mean,
    )

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-94 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-94 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-94 walk requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    cfg = card.recipe.model_config
    entry = rung0.assemble_frame(record_root, 1, 0)
    state = rung0.bridge_entry(card, entry)
    entry_rows = rung0.ladder.compare_fields(rung0.candidate_fields(state), entry)
    require(entry_rows["first_non_bit_field"] is None,
            "independent stage-0 entry is no longer bit exact")
    freshwater, surface = rhs_walk._forcing(entry["ssh"].shape)
    oracle, record_census = assemble_slow(record_root, plant=plant)

    ordinary_model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    observed_model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    ordinary = jax.device_get(ordinary_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    observed, parts = rhs_walk._capture_stage1_parts(
        observed_model, state, card.dt_s, freshwater, surface)
    observed_fields = rung0.candidate_fields(observed)
    if plant == "trace-bit":
        observed_fields["T"] = np.array(observed_fields["T"], copy=True)
        observed_fields["T"][0, 0, 0] = np.nextafter(
            observed_fields["T"][0, 0, 0], np.float64(np.inf))
    passivity = rung0.ladder.compare_fields(
        observed_fields, rung0.candidate_fields(ordinary))
    require(passivity["first_non_bit_field"] is None,
            "slow-boundary observer changes the production trajectory")

    accumulated = jax.device_get(jax.jit(rhs_common.source_order_accumulators)(
        parts["hpg_u"], parts["hpg_v"], parts["ldf_u"], parts["ldf_v"],
        parts["vorticity_u"], parts["vorticity_v"],
        parts["keg_u"], parts["keg_v"], parts["zad_u"], parts["zad_v"],
    ))
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_barotropic_substeps=True),
    )
    trace = jax.device_get(trace_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    operands = trace.slow_forcing_operands
    masks = phase3_gate.expected_masks(card)
    active = {
        "u": masks["u"][..., 0],
        "v": masks["v"][..., 0],
        "ssh": masks["ssh"],
    }
    rhs_calibration = {
        "u": rhs_walk.score(
            round83.native_u(np.asarray(operands["du_dt"])),
            round83.native_u(np.asarray(accumulated["after_zad_u"])),
            masks["u"],
        ),
        "v": rhs_walk.score(
            round83.native_v(np.asarray(operands["dv_dt"])),
            round83.native_v(np.asarray(accumulated["after_zad_v"])),
            masks["v"],
        ),
    }
    require(all(row["bit_exact"] for row in rhs_calibration.values()),
            "slow trace and admitted exact RHS observer disagree")

    literal_u, literal_v = jax.device_get(jax.jit(
        lambda du, dv, hu, hv, Hu, Hv, mu, mv: (
            _nemo_literal_seed_depth_mean(
                du, hu, mu, np.float64(1.0) / Hu),
            _nemo_literal_seed_depth_mean(
                dv, hv, mv, np.float64(1.0) / Hv),
        )
    )(
        operands["du_dt"], operands["dv_dt"],
        operands["h_u"], operands["h_v"],
        operands["H_u"], operands["H_v"],
        state.u_mask.data, state.v_mask.data,
    ))
    literal_arm = {
        "u": rhs_walk.score(
            round83.native_u(np.asarray(literal_u)),
            oracle["depth_u"], active["u"]),
        "v": rhs_walk.score(
            round83.native_v(np.asarray(literal_v)),
            oracle["depth_v"], active["v"]),
    }
    barrier_u, barrier_v = jax.device_get(jax.jit(
        lambda du, dv, hu, hv, Hu, Hv, m3u, m3v, m2u, m2v: (
            nemo_literal_depth_mean(
                du, hu, m2u, np.float64(1.0) / Hu, level_mask=m3u),
            nemo_literal_depth_mean(
                dv, hv, m2v, np.float64(1.0) / Hv, level_mask=m3v),
        )
    )(
        operands["du_dt"], operands["dv_dt"],
        operands["h_u"], operands["h_v"],
        operands["H_u"], operands["H_v"],
        (operands["h_u"] > 0.0).astype(operands["h_u"].dtype),
        (operands["h_v"] > 0.0).astype(operands["h_v"].dtype),
        state.u_mask.data, state.v_mask.data,
    ))
    barrier_literal_arm = {
        "u": rhs_walk.score(
            round83.native_u(np.asarray(barrier_u)),
            oracle["depth_u"], active["u"]),
        "v": rhs_walk.score(
            round83.native_v(np.asarray(barrier_v)),
            oracle["depth_v"], active["v"]),
    }

    state_after = trace.state_after_barotropic
    live = {
        "depth_u": round83.native_u(np.asarray(operands["depth_u"])),
        "depth_v": round83.native_v(np.asarray(operands["depth_v"])),
        "drag_u": round83.native_u(np.asarray(operands["post_drag_u"])),
        "drag_v": round83.native_v(np.asarray(operands["post_drag_v"])),
        "wind_u": round83.native_u(np.asarray(operands["post_wind_u"])),
        "wind_v": round83.native_v(np.asarray(operands["post_wind_v"])),
        "final_u": round83.native_u(np.asarray(operands["pre_external_u"])),
        "final_v": round83.native_v(np.asarray(operands["pre_external_v"])),
        "ssh_rhs": np.asarray(trace.slow_forcing[0]),
        "ssh_after": np.asarray(state_after.eta.data),
        "ub_after": round83.native_u(np.asarray(state_after.uu_b.data)),
        "vb_after": round83.native_v(np.asarray(state_after.vv_b.data)),
    }
    rows = {}
    for name in SOURCE_ORDER:
        face = "u" if name.endswith("_u") or name == "ub_after" else (
            "v" if name.endswith("_v") or name == "vb_after" else "ssh")
        rows[name] = rhs_walk.score(live[name], oracle[name], active[face])

    depth_operands = operand_walk(record_root, operands, oracle, masks)

    if plant == "record-bit":
        synthetic = np.zeros((2, 2), dtype=np.float64)
        planted = synthetic.copy()
        planted[0, 0] = np.nextafter(0.0, np.float64(np.inf))
        control = rhs_walk.score(
            planted, synthetic, np.ones_like(synthetic, dtype=bool))
        require(control["differing_cells"] == 1 and not control["bit_exact"],
                "one-ULP slow-boundary control stayed green")
        raise GateError("record-bit plant fired")

    first = first_nonbit(rows)
    return {
        "status": "MEASURED_R94_SLOW_WALK",
        "claim_label": "independent",
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record": record_census,
        "stage0_entry": entry_rows,
        "observer_passivity": passivity,
        "rhs_trace_calibration": rhs_calibration,
        "source_literal_jit_arm": literal_arm,
        "barrier_literal_jit_arm": barrier_literal_arm,
        "source_order": list(SOURCE_ORDER),
        "rows": rows,
        "depth_operand_walk_rank0": depth_operands,
        "first_non_bit_statement": first,
        "predictions": {
            "R94_P1_record_sound": True,
            "R94_P2_depth_bit_exact": bool(
                rows["depth_u"]["bit_exact"] and rows["depth_v"]["bit_exact"]),
            "R94_P3_drag_first": bool(
                first is not None and first["boundary"] in ("drag_u", "drag_v")),
            "R94_P4_observer_passive": True,
        },
        "compiled_sources": {
            "depth": "stp2d.f90:206-219",
            "drag": "stp2d.f90:231-236; dynspg_ts.f90:1404-1493",
            "wind": "stp2d.f90:238-250",
            "ssh_rhs": "stp2d.f90:290-312",
            "split_explicit_output": "stp2d.f90:317-326",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = measure(
            args.deck_root, args.record_root, args.expect_commit, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, check_record.Refusal, OSError, ValueError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS MEASURED_R94_SLOW_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
