#!/usr/bin/env python3
"""Split OMT-4's fold-local substep-1 slow-V forcing in NEMO order."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(TESTCASES) not in sys.path:
    sys.path.insert(0, str(TESTCASES))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_o1_acquisition_gate as o1_gate,
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
    nemo_testcase_l4_orca2_round229_vector_unit_bisect as r229,
    nemo_testcase_l4_orca2_round235_omt4_substep_table as r235,
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)
from nemo_testcase_l2_gyre_round16_slow_forcing import (  # noqa: E402
    read_slow_forcing,
)

FLOOR = np.float64(2.0e-10)
SOURCE_ORDER = (
    "e3v_3d", "completed_v_rhs", "vmask_3d", "r1_hv0",
    "depth_mean_v", "post_drag_v", "post_wind_incoming_v",
    "coriolis_v", "ssvmask", "final_slow_v",
)
MEASURE_PLANTS = ("none", "record-bit", "passivity")
CLASSIFY_PLANTS = ("none", "source-order", "label-coverage")


class GateError(RuntimeError):
    """The admitted record or the frozen operand attribution moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _native_v(value) -> np.ndarray:
    return r97._native_v(np.asarray(value, dtype=np.float64))[:, :90, ...]


def _row(candidate, oracle, support) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    require(candidate.shape == oracle.shape, f"operand shape mismatch {candidate.shape} != {oracle.shape}")
    if support.ndim < candidate.ndim:
        support = np.broadcast_to(
            support[(...,) + (None,) * (candidate.ndim - support.ndim)],
            candidate.shape,
        )
    require(support.shape == candidate.shape, "owner support shape moved")
    delta = candidate - oracle
    selected = delta[support]
    unequal = support & (candidate != oracle)
    absolute = np.where(support, np.abs(delta), -np.inf)
    flat = int(np.argmax(absolute)) if np.any(support) else 0
    maximum = float(np.max(np.abs(selected), initial=0.0))
    return {
        "bit_exact": bool(np.array_equal(candidate[support], oracle[support])),
        "at_floor": bool(np.isfinite(maximum) and maximum <= FLOOR),
        "unequal": int(np.count_nonzero(unequal)),
        "scored": int(np.count_nonzero(support)),
        "max_abs": maximum,
        "argmax": [int(i) for i in np.unravel_index(flat, candidate.shape)],
    }


def _source_final(incoming, coriolis, mask) -> np.ndarray:
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round

    b = nemo_source_round
    return np.asarray(b(jnp.asarray(incoming) - b(
        jnp.asarray(coriolis) * jnp.asarray(mask))), dtype=np.float64)


def _source_depth(e3v, rhs, mask, reciprocal) -> np.ndarray:
    """Literal scalar-order replay of stp2d's vertical SUM and multiply."""

    product = (np.asarray(e3v, np.float64) * np.asarray(rhs, np.float64))
    product = product * np.asarray(mask, np.float64)
    total = np.array(product[..., 0], copy=True)
    # The reader has already removed NEMO's structural jpk slot, leaving the
    # full 1:jpkm1 physical range. Include every remaining level; an older
    # helper's second truncation stayed green only because its oracle bottom
    # product was zero and fails on the candidate's partial bottom cells.
    for level in range(1, product.shape[-1]):
        total = total + product[..., level]
    return total * np.asarray(reciprocal, np.float64)


def _admit(twin_a: Path, twin_b: Path) -> list[dict[str, object]]:
    masks = o1_gate._defined_masks(twin_a)
    rows = []
    for family, name in (
        ("slow", "oracle_slow_forcing_kt00000001.bin"),
        ("substeps", "oracle_bt_substeps_kt00000001.bin"),
    ):
        left, right = twin_a / name, twin_b / name
        require(left.is_file() and right.is_file(), f"missing inherited {name}")
        exact = o1_gate._compare_hygiene_record(left, right, masks)
        require(exact["status"] == "EXACT_DEFINED_BYTES", f"{name}: twin payload differs")
        rows.append({
            "family": family, "name": name, "status": exact["status"],
            "defined_f64": exact["defined_f64"],
        })
    return rows


def _run_trace(card, state, freshwater, surface):
    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
        _NEMOWSRK3TestHooks,
    )

    # Round 235's temporary one-field raw-mask hook was removed when that
    # measurement restored production. Rebuild the same slow-forcing arm from
    # the surviving passive producer trace; the atomic hook owns the later
    # vector-mask part of the indivisible unit.
    seed_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True))
    seed = jax.device_get(seed_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(seed, _NEMOWSLiveOperandTrace),
            "slow-forcing seed trace return type moved")
    producer = seed.slow_forcing_producer
    raw_vmask = r229._raw_vmask(card)
    b = nemo_source_round
    slow_override = (
        jnp.asarray(producer["final_u"]),
        b(jnp.asarray(producer["incoming_v"]) - b(
            jnp.asarray(producer["coriolis_v"]) * raw_vmask)),
    )
    common = dict(
        barotropic_slow_forcing_override=slow_override,
        barotropic_atomic_fold_unit=True,
    )
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**common),
    )
    observed = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True, **common),
    )
    plain = jax.device_get(ordinary.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    traced = jax.device_get(observed.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            "live operand trace return type moved")
    equality = passive._ordinary_state_equal(traced.state_after, plain)
    return traced, equality


def measure(deck_root: Path, frame_root: Path, twin_a: Path, twin_b: Path,
            label: str, expect_commit: str, *, plant: str = "none") -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_qco_card_mesh_operands,
    )

    require(plant in MEASURE_PLANTS, f"unknown plant {plant!r}")
    require(label in ("independent", "given_nemo_entry"), f"bad label {label!r}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-236 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-236 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-236 measurement requires production JIT on CPU")

    admission = _admit(twin_a, twin_b)
    slow = read_slow_forcing(
        twin_a / "oracle_slow_forcing_kt00000001.bin", dims=(94, 152, 31))
    substeps = phase1._parse_bt(
        twin_a / "oracle_bt_substeps_kt00000001.bin", "substeps")
    # The campaign reader owns the field schema; _parse_bt above is only the
    # self-describing header admission.
    from scripts.validate.ocean_fidelity.testcases import (
        nemo_testcase_l2_gyre_phase3_gate as phase3,
    )
    substeps = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    traced, passivity = _run_trace(card, state, freshwater, surface)
    if plant == "passivity":
        passivity["eta"] = False
    require(all(passivity.values()), "live operand trace moved completed state")
    producer = traced.slow_forcing_producer

    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "OMT-4 card has no raw NEMO operand bundle")
    raw_e3v = np.asarray(raw.e3v_0, dtype=np.float64)[:, :90, :]
    raw_vmask3 = np.asarray(raw.vmask, dtype=np.float64)[:, :90, :]
    raw_hv = np.asarray(raw.hv_0, dtype=np.float64)[:, :90]
    wet = (raw_hv > 0.0).astype(np.float64)
    raw_r1_hv = wet / (raw_hv + 1.0 - wet)
    raw_ssvmask = np.max(raw_vmask3, axis=-1)

    # Recreate the exact static branch at ocean_model_latlon_cgrid.py:6267-6337.
    # The raw mesh is selected only where NEMO's fold keeps a V face wet but
    # the reconstructed compact mesh marks it dry; every other face uses the
    # shared reconstructed operands. Comparing raw geometry everywhere was a
    # round-236 instrument refusal because it did not replay the candidate's
    # own completed depth mean.
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    ).astype(jnp.float64)
    um3, vm3 = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    ops = nemo_qco_card_mesh_operands(
        h_ref, um3.astype(jnp.float64), vm3.astype(jnp.float64),
        card.recipe.grid, jnp.float64)
    ops_e3v = np.asarray(ops.e3v_0, np.float64)[:, :90, :]
    ops_vmask = np.asarray(ops.vmask3, np.float64)[:, :90, :]
    ops_hv = np.asarray(ops.hv_0, np.float64)[:, :90]
    ops_wet = (ops_hv > 0.0).astype(np.float64)
    ops_r1 = ops_wet / (ops_hv + 1.0 - ops_wet)
    fold_unit = (wet > 0.0) & (ops_wet == 0.0)
    effective_e3v = np.where(fold_unit[..., None], raw_e3v, ops_e3v)
    effective_vmask = np.where(
        fold_unit[..., None], raw_vmask3, ops_vmask)
    effective_r1 = np.where(fold_unit, raw_r1_hv, ops_r1)

    candidate = {
        "e3v_3d": effective_e3v,
        "completed_v_rhs": _native_v(producer["rhs_v"]),
        "vmask_3d": effective_vmask,
        "r1_hv0": effective_r1,
        "depth_mean_v": _native_v(producer["depth_v"]),
        "post_drag_v": _native_v(producer["post_drag_v"]),
        "post_wind_incoming_v": _native_v(producer["incoming_v"]),
        "coriolis_v": _native_v(producer["coriolis_v"]),
        "ssvmask": raw_ssvmask,
        "final_slow_v": _native_v(producer["final_v"]),
    }
    oracle = {
        "e3v_3d": slow["e3v"],
        "completed_v_rhs": slow["krhs_v"],
        "vmask_3d": slow["vmask"],
        "r1_hv0": slow["r1_hv0"],
        "depth_mean_v": slow["depth_v"],
        "post_drag_v": slow["post_drag_v"],
        "post_wind_incoming_v": slow["post_wind_v"],
        "coriolis_v": np.asarray(substeps["cor_v"][0], dtype=np.float64),
        "ssvmask": raw_ssvmask,
        "final_slow_v": np.asarray(substeps["slow_v"][0], dtype=np.float64),
    }
    if plant == "record-bit":
        oracle["completed_v_rhs"] = np.array(oracle["completed_v_rhs"], copy=True)
        oracle["completed_v_rhs"][-1, 0, 0] = np.nextafter(
            oracle["completed_v_rhs"][-1, 0, 0], np.inf)

    owner = candidate["final_slow_v"] != oracle["final_slow_v"]
    require(int(np.count_nonzero(owner)) == 35,
            f"final slow-V owner support moved: {np.count_nonzero(owner)} != 35")
    require(int(np.count_nonzero(owner[:-3])) == 0,
            "final slow-V owner support escaped the fold band")
    rows = {name: _row(candidate[name], oracle[name], owner)
            for name in SOURCE_ORDER}
    first = next((name for name in SOURCE_ORDER if not rows[name]["bit_exact"]), None)

    depth_replay = {
        "candidate_statement": _row(
            _source_depth(
                candidate["e3v_3d"], candidate["completed_v_rhs"],
                candidate["vmask_3d"], candidate["r1_hv0"]),
            candidate["depth_mean_v"], owner),
        "oracle_statement": _row(
            _source_depth(
                oracle["e3v_3d"], oracle["completed_v_rhs"],
                oracle["vmask_3d"], oracle["r1_hv0"]),
            oracle["depth_mean_v"], owner),
    }
    require(depth_replay["candidate_statement"]["at_floor"],
            "candidate operands do not replay its completed depth mean within the floor")
    require(depth_replay["oracle_statement"]["bit_exact"],
            "oracle operands do not replay its completed depth mean")
    depth_accumulated = {
        "e3v": candidate["e3v_3d"],
        "rhs": candidate["completed_v_rhs"],
        "mask": candidate["vmask_3d"],
        "reciprocal": candidate["r1_hv0"],
    }
    for name, key in (("e3v", "e3v_3d"),
                      ("rhs", "completed_v_rhs"),
                      ("mask", "vmask_3d"),
                      ("reciprocal", "r1_hv0")):
        depth_accumulated[name] = oracle[key]
        depth_replay[f"through_{name}"] = _row(
            _source_depth(
                depth_accumulated["e3v"], depth_accumulated["rhs"],
                depth_accumulated["mask"], depth_accumulated["reciprocal"]),
            oracle["depth_mean_v"], owner)

    # Certify the source statement on each side, then replace its three
    # operands in NEMO order without feeding any replay back into the model.
    candidate_replay = _source_final(
        candidate["post_wind_incoming_v"], candidate["coriolis_v"],
        candidate["ssvmask"])
    oracle_replay = _source_final(
        oracle["post_wind_incoming_v"], oracle["coriolis_v"], oracle["ssvmask"])
    replay = {
        "candidate_statement": _row(
            candidate_replay, candidate["final_slow_v"], owner),
        "oracle_statement": _row(
            oracle_replay, oracle["final_slow_v"], owner),
    }
    accumulated = {
        "incoming": candidate["post_wind_incoming_v"],
        "coriolis": candidate["coriolis_v"],
        "mask": candidate["ssvmask"],
    }
    for name, key in (("incoming", "post_wind_incoming_v"),
                      ("coriolis", "coriolis_v"), ("mask", "ssvmask")):
        accumulated[name] = oracle[key]
        replay[f"through_{name}"] = _row(
            _source_final(accumulated["incoming"], accumulated["coriolis"],
                          accumulated["mask"]),
            oracle["final_slow_v"], owner)
    require(plant == "none", f"{plant} plant stayed green")
    return {
        "format": "nemo-testcase-l4-orca2-round236-slow-v-split-v1",
        "status": "PASS_R236_SLOW_V_SPLIT",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "floor": float(FLOOR),
        "record_admission": admission,
        "passivity": passivity,
        "source_order": list(SOURCE_ORDER),
        "owner_support": r228._difference(
            candidate["final_slow_v"], oracle["final_slow_v"]),
        "rows": rows,
        "first_unequal": first,
        "depth_replay": depth_replay,
        "final_replay": replay,
        "compiled_source": {
            "depth_drag_wind": "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stp2d.f90:194-230",
            "copy_and_coriolis": "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:289-328",
        },
    }


def classify(reports: list[dict], *, plant: str = "none") -> dict:
    require(plant in CLASSIFY_PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == {"independent", "given_nemo_entry"},
            "claim-label coverage moved")
    for label, report in by_label.items():
        require(report["status"] == "PASS_R236_SLOW_V_SPLIT",
                f"{label}: measurement did not pass")
        require(all(report["passivity"].values()), f"{label}: passivity moved")
        require(tuple(report["source_order"]) == SOURCE_ORDER,
                f"{label}: source order moved")
        require(report["owner_support"]["unequal"] == 35,
                f"{label}: owner support moved")
    if plant == "source-order":
        by_label["independent"]["source_order"][0:2] = reversed(
            by_label["independent"]["source_order"][0:2])
    require(all(tuple(report["source_order"]) == SOURCE_ORDER
                for report in by_label.values()), "source-order plant fired")
    first = {label: report["first_unequal"] for label, report in by_label.items()}
    require(len(set(first.values())) == 1, "labels disagree on first operand")
    rows = {
        label: {
            "first_unequal": report["first_unequal"],
            "first_row": report["rows"][report["first_unequal"]],
            "owner_support": report["owner_support"],
            "depth_replay": report["depth_replay"],
            "final_replay": report["final_replay"],
        }
        for label, report in by_label.items()
    }
    first_name = next(iter(first.values()))
    prediction = (
        "CONFIRMED_CORIOLIS_REMOVAL_UNIT"
        if first_name in {"coriolis_v", "ssvmask", "final_slow_v"}
        else f"REFUTED_EARLIER_{first_name.upper()}"
    )
    incoming_only_closes = all(
        report["final_replay"]["through_incoming"]["bit_exact"]
        for report in by_label.values())
    require(plant == "none", f"{plant} plant stayed green")
    return {
        "format": "nemo-testcase-l4-orca2-round236-classification-v1",
        "status": "HELD_R236_SLOW_V_OWNER_NAMED",
        "source_order": list(SOURCE_ORDER),
        "rows": rows,
        "predictions": {
            "R236-P1": "CONFIRMED_ADMITTED_PASSIVE",
            "R236-P2": prediction,
            "R236-P3": (
                "REFUTED_INCOMING_ONLY_CLOSES_FINAL"
                if incoming_only_closes else "CONFIRMED_CANCELLING_PAIR"),
            "R236-P4": "CONFIRMED_LABEL_AGREEMENT",
            "R236-P5": "CONFIRMED_MEASUREMENT_ONLY",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"))
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=MEASURE_PLANTS, default="none")
    parser.add_argument("--classify", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--classify-plant", choices=CLASSIFY_PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify:
            result = classify(
                [json.loads(path.read_text(encoding="utf-8"))
                 for path in args.classify], plant=args.classify_plant)
        else:
            require(all(value is not None for value in (
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit)), "measurement arguments incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit, plant=args.plant)
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        planted = (args.classify_plant != "none" if args.classify
                   else args.plant != "none")
        print(f"STATUS {'PLANT-FIRED' if planted else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
