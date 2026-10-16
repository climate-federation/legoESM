#!/usr/bin/env python3
"""Split the ORCA2 completed-stage top-V association at matched points."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round281_stage1_source_split as r281,
)

FLOOR = np.float64(2.0e-10)
SOURCE_ROW = 146
TARGET_ROW = 147
PLANTS = ("none", "one-bit", "wrong-sign", "wrong-row", "empty-support")


class GateError(RuntimeError):
    """An admitted record, matched point, or known-answer control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(left, right, support=None) -> dict[str, object]:
    return r281._exact_score(
        np.asarray(left, np.float64), np.asarray(right, np.float64), support)


def measure(deck_root: Path, record_root: Path, baseline_root: Path,
            expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
        _rk3_fold_exchange_top_v,
        compute_face_masks_3d,
    )
    from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-289 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round 289 requires production JIT on CPU")

    admission = r281.admission_gate.run(record_root, baseline_root)
    oracle = r281._assemble_values(record_root)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    require(not card.recipe.model_config.rk3_fold_exchange_top_v_row,
            "Decision-125 top-V selector unexpectedly enabled")
    state = card.recipe.initial_state
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, False))
    traced = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, False, live=True))
    ordinary_after = jax.device_get(ordinary.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    observed = jax.device_get(traced.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(observed, _NEMOWSLiveOperandTrace),
            "live operand trace type moved")
    passivity = passive._ordinary_state_equal(observed.state_after, ordinary_after)
    require(all(passivity.values()), "live trace moved the completed state")

    # The current card keeps the association off.  Its stage output therefore
    # is the matched pre-lbc_lnk value; the association touches only its final
    # redundant V row, leaving SOURCE_ROW available in the same array.
    stage_v = np.asarray(observed.stage_outputs[0][1], np.float64)
    require(stage_v.shape == (149, 180, 30),
            f"redundant stage-V shape moved: {stage_v.shape}")
    candidate_native = stage_v[1:]
    oracle_cor = np.asarray(oracle["cor_v"], np.float64)[..., :-1]
    oracle_final = np.asarray(oracle["final_v"], np.float64)[..., :-1]
    require(candidate_native.shape == oracle_cor.shape == oracle_final.shape,
            "matched native V shapes moved")

    compact_u, compact_v = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    resolved = nemo_qco_resolved_mesh_operands(
        card.recipe.z_coord, card.recipe.grid, compact_u, compact_v,
        np.float64, card.recipe.z_coord.is_active.shape[-1])
    resolved_v = np.asarray(compact_v, np.float64).copy()
    resolved_v[1:] = np.asarray(resolved.vmask3, np.float64)
    mask_delta = np.asarray(compact_v) != resolved_v
    changed_rows = np.flatnonzero(np.any(mask_delta, axis=(1, 2)))
    require(np.array_equal(changed_rows, np.asarray([148]))
            and int(np.count_nonzero(mask_delta)) == 1319,
            "resolved-mask delta no longer lives only on the target top row")

    perm = np.asarray(card.recipe.grid.fold.perm_v)
    sign = np.float64(card.recipe.grid.fold.vector_sign_v)
    require(sign == -1.0 and perm.shape == (180,),
            "ORCA2 T-pivot V permutation/sign moved")

    oracle_replay = oracle_cor.copy()
    oracle_replay[TARGET_ROW] = sign * oracle_cor[SOURCE_ROW, perm]
    associated = np.asarray(jax.device_get(_rk3_fold_exchange_top_v(
        jnp.asarray(stage_v), card.recipe.grid, jnp.asarray(resolved_v))))[1:]
    no_mask = candidate_native.copy()
    no_mask[TARGET_ROW] = sign * candidate_native[SOURCE_ROW, perm]
    source_substitution = candidate_native.copy()
    source_substitution[TARGET_ROW] = (
        sign * oracle_cor[SOURCE_ROW, perm])
    plus_sign = candidate_native.copy()
    plus_sign[TARGET_ROW] = candidate_native[SOURCE_ROW, perm]
    wrong_row = candidate_native.copy()
    wrong_row[TARGET_ROW] = sign * candidate_native[SOURCE_ROW - 1, perm]
    right_half = candidate_native.copy()
    right_half[TARGET_ROW, 90:] = (
        sign * candidate_native[SOURCE_ROW, perm][90:])

    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round289-topv-split-v1",
        "status": "PASS_R289_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1",
        "worktree": stamp,
        "admission": admission,
        "execution": "production-jit-cpu-fp64-libm; matched offline association replay",
        "program_points": {
            "source": "stage-1 V after barotropic correction, before lbc_lnk",
            "target": "stage-1 V after completed-stage lbc_lnk",
        },
        "compiled_sources": {
            "update": "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480",
            "association": "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:812-820",
            "fold_rule": "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/lbcnfd.f90:684-717",
        },
        "passivity": passivity,
        "mask_delta": {"unequal": int(np.count_nonzero(mask_delta)),
                       "rows": changed_rows.tolist()},
        "rows": {
            "nemo_association_replay": _score(oracle_replay, oracle_final),
            "candidate_source_vs_nemo": _score(
                candidate_native[SOURCE_ROW], oracle_cor[SOURCE_ROW]),
            "candidate_pre_target_vs_nemo": _score(
                candidate_native[TARGET_ROW], oracle_final[TARGET_ROW]),
            "candidate_current_copy_vs_nemo": _score(
                associated[TARGET_ROW], oracle_final[TARGET_ROW]),
            "candidate_source_substitution_vs_nemo": _score(
                source_substitution[TARGET_ROW], oracle_final[TARGET_ROW]),
            "candidate_no_mask_vs_current_copy": _score(
                no_mask[TARGET_ROW], associated[TARGET_ROW]),
            "candidate_plus_sign_vs_nemo": _score(
                plus_sign[TARGET_ROW], oracle_final[TARGET_ROW]),
            "candidate_wrong_row_vs_nemo": _score(
                wrong_row[TARGET_ROW], oracle_final[TARGET_ROW]),
            "candidate_right_half_only_vs_nemo": _score(
                right_half[TARGET_ROW], oracle_final[TARGET_ROW]),
            "candidate_full_stage_v_vs_nemo": _score(
                associated, oracle_final),
        },
        "one_bit_control": _score(one, np.nextafter(one, np.inf)),
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    rows = report["rows"]
    if plant == "one-bit":
        report["one_bit_control"]["unequal"] = 0
    elif plant == "wrong-sign":
        rows["candidate_plus_sign_vs_nemo"]["max_abs"] = 0.0
    elif plant == "wrong-row":
        rows["candidate_wrong_row_vs_nemo"]["max_abs"] = 0.0
    elif plant == "empty-support":
        report["mask_delta"]["unequal"] = 0

    require(report["status"] == "PASS_R289_MEASUREMENT",
            "measurement status moved")
    require(report["claim_label"] == "independent_omt4_kt1_stage1",
            "claim label moved")
    require(report["admission"]["status"] == "PASS_R280_OMT4_STAGE1_ADMISSION",
            "round-280 record admission moved")
    require(all(report["passivity"].values()), "live trace is not passive")
    require(report["mask_delta"] == {"unequal": 1319, "rows": [148]},
            "resolved-mask target-only control moved")
    require(rows["nemo_association_replay"]["unequal"] == 0,
            "compiled NEMO association replay is not bit-exact")
    require(report["one_bit_control"]["unequal"] == 1,
            "one-bit control did not fire")
    require(rows["candidate_plus_sign_vs_nemo"]["max_abs"] > FLOOR,
            "wrong-sign control stayed at the floor")
    require(rows["candidate_wrong_row_vs_nemo"]["max_abs"] > FLOOR,
            "wrong-row control stayed at the floor")

    source = rows["candidate_source_vs_nemo"]
    source_valued = source["max_abs"] > FLOOR
    source_closes = rows["candidate_source_substitution_vs_nemo"]["unequal"] == 0
    mask_null = rows["candidate_no_mask_vs_current_copy"]["max_abs"] == 0.0
    full = rows["candidate_current_copy_vs_nemo"]
    half = rows["candidate_right_half_only_vs_nemo"]
    support_closes = half["max_abs"] < full["max_abs"]
    report["first_nonbit_statement"] = {
        "boundary": ("copied source row" if source_valued else
                     "pivot-row support" if support_closes else
                     "unresolved after source/support/sign split"),
        "nemo_source": report["compiled_sources"]["association"],
    }
    report["prediction_ledger"] = {
        "R289-P1": "CONFIRMED",
        "R289-P2": "CONFIRMED",
        "R289-P3": ("REFUTED_SOURCE_ROW_OWNS" if source_valued and source_closes
                     else "CONFIRMED_SUPPORT_OWNS" if support_closes
                     else "REFUTED_UNRESOLVED"),
        "R289-P4": "CONFIRMED" if mask_null else "REFUTED_MASK_MOVES",
        "R289-P5": "CONFIRMED",
    }
    report["status"] = (
        "HELD_R289_SOURCE_ROW" if source_valued and source_closes else
        "HELD_R289_SUPPORT" if support_closes else "HELD_R289_UNRESOLVED")
    require(plant == "none", f"{plant} plant stayed green")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.record_root, args.baseline_root,
                         args.expect_commit)), "measurement arguments incomplete")
            result = measure(args.deck_root, args.record_root, args.baseline_root,
                             args.expect_commit)
        else:
            require(args.classify is not None, "classification input missing")
            result = classify(json.loads(args.classify.read_text(encoding="utf-8")),
                              plant=args.plant)
    except (GateError, r281.GateError, OSError, KeyError, TypeError, ValueError,
            IndexError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
