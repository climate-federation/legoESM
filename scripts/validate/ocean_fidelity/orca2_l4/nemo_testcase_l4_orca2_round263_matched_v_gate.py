#!/usr/bin/env python3
"""Reconcile the scalar-ratio fold and the stage-1 V comparison point."""

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

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round260_stage2_pivot_gate as r260,
)

LABELS = r249.LABELS
PLANTS = (
    "none", "label-coverage", "scalar-sign", "scalar-row", "ratio-closure",
    "passivity", "raw-boundary", "matched-boundary", "causal-closure",
    "one-bit",
)


class GateError(RuntimeError):
    """A frozen round-263 discriminator moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _source_ratio_table(deck_root: Path, frame_root: Path) -> dict[str, object]:
    """Build r3va before/after the compiled scalar V fold association."""

    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.vertical import compute_layer_thickness

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    stage1 = rung0.assemble_frame(frame_root, 1, 1)
    z_coord, grid = card.recipe.z_coord, card.recipe.grid
    raw = z_coord.nemo_een_barotropic
    require(raw is not None and grid.fold is not None,
            "ORCA2 raw face geometry or fold descriptor is missing")
    dtype = jnp.float64
    h_ref = compute_layer_thickness(
        jnp.zeros_like(jnp.asarray(stage1["ssh"], dtype=dtype)),
        jnp.asarray(card.recipe.initial_state.H_bathy.data, dtype=dtype),
        z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    )
    require(h_ref.shape[-1] == 30, "ORCA2 vertical dimension moved")
    area_t = jnp.asarray(z_coord.nemo_e1e2t, dtype=dtype)
    area_v = jnp.asarray(raw.e1v * raw.e2v, dtype=dtype)
    vmask = jnp.asarray(raw.vmask, dtype=dtype)
    surface_vmask = jnp.max(vmask, axis=-1)
    hv = jnp.asarray(raw.hv_0, dtype=dtype)
    r1_hv0 = surface_vmask / (hv + 1.0 - surface_vmask)
    r1_area_v = jnp.where(area_v > 0.0, 1.0 / area_v, 0.0)
    eta = jnp.asarray(stage1["ssh"], dtype=dtype)
    north_eta = jnp.concatenate([eta[1:], jnp.zeros_like(eta[:1])], axis=0)
    north_area = jnp.concatenate(
        [area_t[1:], jnp.zeros_like(area_t[:1])], axis=0)
    unassociated = nemo_source_round(
        nemo_source_round(
            nemo_source_round(
                jnp.float64(0.5) * nemo_source_round(
                    nemo_source_round(area_t * eta)
                    + nemo_source_round(north_area * north_eta)))
            * r1_hv0)
        * r1_area_v)
    perm = jnp.asarray(grid.fold.perm_v)
    plus = unassociated.at[-1].set(unassociated[-2, perm])
    minus = unassociated.at[-1].set(-unassociated[-2, perm])
    unassociated = np.asarray(unassociated, np.float64)
    plus = np.asarray(plus, np.float64)
    minus = np.asarray(minus, np.float64)

    def stats(values: np.ndarray) -> dict[str, object]:
        row = values[-1]
        return {
            "min": float(np.min(row)),
            "max": float(np.max(row)),
            "nonzero": int(np.count_nonzero(row)),
            "negative": int(np.count_nonzero(row < 0.0)),
            "positive": int(np.count_nonzero(row > 0.0)),
        }

    return {
        "unassociated_pivot_row": stats(unassociated),
        "scalar_plus_pivot_row": stats(plus),
        "vector_minus_pivot_row": stats(minus),
        "plus_vs_unassociated": r249._score(plus[-1], unassociated[-1]),
        "minus_vs_plus": r249._score(minus[-1], plus[-1]),
        "association": {"point_type": "V", "sign": 1.0,
                        "destination": "pivot row", "source": "south row via perm_v"},
    }


def _compiled_contract(compiled_root: Path) -> dict[str, object]:
    stage_path = compiled_root / "stprk3_stg.f90"
    fold_path = compiled_root / "lbcnfd.f90"
    stage = stage_path.read_text(encoding="utf-8")
    fold = fold_path.read_text(encoding="utf-8")
    scalar_call = "r3va, 'V', 1._wp"
    vector_call = "vv(:,:,:       ,Kaa), 'V', -1._wp"
    require(scalar_call in stage, "compiled scalar r3va V/+1 call moved")
    require(vector_call in stage, "compiled completed-V V/-1 call moved")
    require("CASE ( 'V' )" in fold and "ptab(jf)%pt4d(ii1,ij1,jk,jl) = psgn(jf)" in fold,
            "compiled V fold rule moved")
    return {
        "scalar_ratio": {"point_type": "V", "sign": 1.0,
                         "citation": "stprk3_stg.f90:163-180"},
        "completed_velocity": {"point_type": "V", "sign": -1.0,
                               "citation": "stprk3_stg.f90:779-802"},
        "fold_rule": "lbcnfd.f90:684-721",
    }


def measure(deck_root: Path, frame_root: Path, compiled_root: Path,
            prior_ratio: Path, label: str, expect_commit: str) -> dict[str, object]:
    require(label in LABELS, f"bad label {label!r}")
    prior = json.loads(prior_ratio.read_text(encoding="utf-8"))
    require(prior["label"] == label and prior["status"] == "PASS_R261_MEASUREMENT",
            "round-262 labelled source replay moved")
    matched = r260.measure(deck_root, frame_root, label, expect_commit)
    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round263-matched-v-v1",
        "status": "PASS_R263_MEASUREMENT",
        "label": label,
        "ratio_table": _source_ratio_table(deck_root, frame_root),
        "ratio_oracle_closure": {
            "e3v": prior["operand_rows"]["e3v"],
            "r1_hv": prior["correction_input_rows"]["r1_hv"],
            "zvb": prior["operand_rows"]["zvb"],
            "zFv": prior["zFv"],
            "candidate_replay": prior["replay"]["candidate"],
            "oracle_replay": prior["replay"]["oracle"],
            "records": prior["records"],
        },
        "matched_stage": matched,
        "compiled_contract": _compiled_contract(compiled_root),
        "one_bit_control": r249._score(one, np.nextafter(one, np.inf)),
    }


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {row["label"]: row for row in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    independent = by_label["independent"]
    if plant == "scalar-sign":
        independent["compiled_contract"]["scalar_ratio"]["sign"] = -1.0
    elif plant == "scalar-row":
        independent["ratio_table"]["minus_vs_plus"]["unequal"] = 0
    elif plant == "ratio-closure":
        independent["ratio_oracle_closure"]["e3v"]["unequal"] = 1
    elif plant == "passivity":
        independent["matched_stage"]["passivity"]["T"] = False
    elif plant == "raw-boundary":
        independent["matched_stage"]["baseline_stage1"]["v"]["fold_band"]["max_abs"] = 0.0
    elif plant == "matched-boundary":
        independent["matched_stage"]["associated_stage1"]["v"]["fold_band"]["max_abs"] = 1.0
    elif plant == "causal-closure":
        independent["matched_stage"]["arm_stage2"]["S"]["fold_band"]["max_abs"] = 1.0
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    summary = {}
    records = by_label[LABELS[0]]["ratio_oracle_closure"]["records"]
    for label in LABELS:
        row = by_label[label]
        require(row["status"] == "PASS_R263_MEASUREMENT",
                f"{label}: measurement status moved")
        contract = row["compiled_contract"]
        require(contract["scalar_ratio"] == {
            "point_type": "V", "sign": 1.0,
            "citation": "stprk3_stg.f90:163-180"},
            f"{label}: scalar V-ratio convention moved")
        require(contract["completed_velocity"]["sign"] == -1.0,
                f"{label}: completed V convention moved")
        ratio = row["ratio_table"]
        require(ratio["association"]["sign"] == 1.0,
                f"{label}: source ratio replay sign moved")
        require(ratio["minus_vs_plus"]["unequal"] > 0,
                f"{label}: wrong-sign scalar plant is vacuous")
        closure = row["ratio_oracle_closure"]
        require(closure["records"] == records, f"{label}: record identity moved")
        for name in ("e3v", "r1_hv", "zvb"):
            require(closure[name]["unequal"] == 0,
                    f"{label}: scalar ratio downstream {name} is non-bit")
        require(closure["zFv"]["unequal"] == 181
                and closure["zFv"]["max_abs"] == 0.0,
                f"{label}: signed-zero zFv census moved")
        require(closure["candidate_replay"]["unequal"] == 0
                and closure["oracle_replay"]["unequal"] == 0,
                f"{label}: source replay moved")

        matched = row["matched_stage"]
        require(matched["status"] == "PASS_R260_MEASUREMENT"
                and all(matched["passivity"].values()),
                f"{label}: matched-point observer moved the state")
        raw_v = matched["baseline_stage1"]["v"]["fold_band"]
        after_v = matched["associated_stage1"]["v"]["fold_band"]
        require(raw_v["max_abs"] == 0.027227136128767482,
                f"{label}: raw pre-association V fingerprint moved")
        require(after_v["max_abs"] <= float(r249.FLOOR),
                f"{label}: matched post-association V is above the floor")
        require(matched["baseline_stage2"]["S"]["fold_band"]["max_abs"]
                == 6.817836451535307,
                f"{label}: next stage-2 salinity boundary moved")
        for field in ("T", "S"):
            require(matched["arm_stage2"][field]["fold_band"]["max_abs"]
                    <= float(r249.FLOOR),
                    f"{label}: causal post-stage association did not close {field}")
        require(row["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        summary[label] = {
            "scalar_ratio_pivot_rows": ratio,
            "raw_pre_association_v": raw_v,
            "matched_post_association_v": after_v,
            "next_stage2_salinity": matched["baseline_stage2"]["S"]["fold_band"],
            "causal_stage2_salinity": matched["arm_stage2"]["S"]["fold_band"],
        }
    return {
        "format": "nemo-testcase-l4-orca2-round263-classification-v1",
        "status": "PASS_R263_PROGRAM_POINT_RECONCILIATION",
        "labels": summary,
        "predictions": {
            "R263-P1": "CONFIRMED_SCALAR_V_PLUS_ONE",
            "R263-P2": "CONFIRMED_PROGRAM_POINT_MISMATCH_RETRACT_RAW_BOUNDARY",
            "R263-P3": "CONFIRMED_POST_STAGE_ASSOCIATION_IS_NEXT_CAUSAL_BOUNDARY",
            "R263-P4": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--compiled-root", type=Path)
    parser.add_argument("--prior-ratio", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(args.plant == "none", "measurement does not accept plants")
            require(all((args.deck_root, args.frame_root, args.compiled_root,
                         args.prior_ratio, args.label, args.expect_commit)),
                    "measurement arguments are incomplete")
            result = measure(args.deck_root, args.frame_root, args.compiled_root,
                             args.prior_ratio, args.label, args.expect_commit)
        else:
            require(args.classify is not None, "classification JSONs are required")
            result = classify([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.classify], plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r249.GateError, r260.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
