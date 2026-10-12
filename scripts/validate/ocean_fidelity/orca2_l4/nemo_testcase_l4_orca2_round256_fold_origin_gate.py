#!/usr/bin/env python3
"""Restart the OMT-4 private-unit growth walk from kt=1."""

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
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
)

LABELS = r249.LABELS
FIELDS = r249.FIELDS
PLANTS = (
    "none", "label-coverage", "production-route", "passivity",
    "boundary-order", "nemo-fold", "growth-owner", "source-owner",
    "one-bit",
)


class GateError(RuntimeError):
    """The round-256 growth or source-order claim moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _one_bit_control() -> dict[str, object]:
    one = np.asarray([1.0], np.float64)
    return r249._score(one, np.nextafter(one, np.inf))


def _associated_stage_values(card, raw_values):
    """Apply the compiled T-pivot U/V/T/S association to one stage."""

    import jax
    import jax.numpy as jnp
    from legoesm.grids.operators_latlon_cgrid import pad_ns_vector_v

    raw = tuple(np.asarray(value, np.float64) for value in raw_values[:4])
    native = {
        "u": r249._native("u", raw[0]),
        "v": r249._native("v", raw[1]),
        "T": raw[2],
        "S": raw[3],
    }
    fold = card.recipe.grid.fold
    half = int(np.asarray(fold.perm_T).size) // 2
    associated = {name: np.array(value, copy=True)
                  for name, value in native.items()}
    perm_t = np.asarray(fold.perm_T)
    perm_u = np.asarray(fold.perm_u)
    associated["u"][-1, half:] = (
        fold.vector_sign_u * native["u"][-1, perm_u[half:]])
    for name in ("T", "S"):
        associated[name][-1, half:] = native[name][-1, perm_t[half:]]
    v_full = np.asarray(jax.device_get(pad_ns_vector_v(
        jnp.asarray(raw[1][1:-1]), card.recipe.grid)), np.float64)
    associated["v"] = r249._native("v", v_full)
    u_full = np.array(raw[0], copy=True)
    u_full[:, 1:] = associated["u"]
    return (u_full, v_full, associated["T"], associated["S"])


def _stage_association_measure(deck_root: Path, record_root: Path, label: str,
                               expect_commit: str) -> dict[str, object]:
    """Replay NEMO's post-stage U/V/T/S lbc_lnk on passive stage 1."""
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSLiveOperandTrace,
    )

    r249._policy_stamp(expect_commit)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(record_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, True))
    traced = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r249._hooks(card, True, live=True))
    arm = r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True)
    with arm:
        ordinary_after = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        observed = jax.device_get(traced.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(observed, _NEMOWSLiveOperandTrace),
            "stage association observer return type moved")
    passivity = passive._ordinary_state_equal(observed.state_after, ordinary_after)
    require(all(passivity.values()), "stage association observer moved state")

    raw_values = observed.stage_outputs[0]
    raw = {
        "u": np.asarray(raw_values[0], np.float64),
        "v": np.asarray(raw_values[1], np.float64),
        "T": np.asarray(raw_values[2], np.float64),
        "S": np.asarray(raw_values[3], np.float64),
    }
    raw_native = {name: r249._native(name, values) for name, values in raw.items()}
    associated_values = _associated_stage_values(card, raw_values)
    associated_native = {
        name: r249._native(name, values)
        for name, values in zip(("u", "v", "T", "S"), associated_values)
    }
    oracle = rung0.assemble_frame(record_root, 1, 1)
    rows = {}
    for name in ("u", "v", "T", "S"):
        before = raw_native[name]
        after = associated_native[name]
        rows[name] = {
            "raw_vs_nemo": r249._score(before, oracle[name]),
            "associated_vs_nemo": r249._score(after, oracle[name]),
            "raw_fold_band_vs_nemo": r249._score(
                before[-3:], oracle[name][-3:]),
            "associated_fold_band_vs_nemo": r249._score(
                after[-3:], oracle[name][-3:]),
            "association_delta": r249._score(after, before),
            "raw_fold_identity": r228.fold_residual(
                before, r249.POINT_TYPES[name], card.recipe.grid.fold,
                r249.SIGNS[name]),
            "associated_fold_identity": r228.fold_residual(
                after, r249.POINT_TYPES[name], card.recipe.grid.fold,
                r249.SIGNS[name]),
            "nemo_fold_identity": r228.fold_residual(
                oracle[name], r249.POINT_TYPES[name], card.recipe.grid.fold,
                r249.SIGNS[name]),
        }
    return {
        "format": "nemo-testcase-l4-orca2-round256-stage-association-v1",
        "status": "PASS_R256_STAGE_ASSOCIATION_REPLAY",
        "label": label,
        "operand_route": "production-d114-resolved",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "passivity": passivity,
        "stage": 1,
        "source_order": ["raw_stage_update", "post_stage_lbc_lnk"],
        "rows": rows,
        "compiled_source": {
            "external_association": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:721-739",
            "history_rotation": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:781-793",
            "post_stage_association": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802",
        },
    }


def measure(deck_root: Path, record_root: Path, label: str,
            expect_commit: str) -> dict[str, object]:
    stage = r249.measure_stages(
        deck_root, record_root, label, True, "resolved-candidate",
        expect_commit, max_kt=7)
    association = _stage_association_measure(
        deck_root, record_root, label, expect_commit)
    return {
        "format": "nemo-testcase-l4-orca2-round256-measurement-v1",
        "status": "PASS_R256_MEASUREMENT",
        "label": label,
        "stage_growth": stage,
        "stage_association": association,
        "one_bit_control": _one_bit_control(),
    }


def _first(rows: list[dict[str, object]], key: str) -> dict[str, object] | None:
    return next((row for row in rows if bool(row[key])), None)


def classify(reports: list[dict[str, object]], *,
             associations: list[dict[str, object]] | None = None,
             plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    if associations is not None:
        by_association = {report["label"]: report for report in associations}
        require(set(by_association) == set(LABELS),
                "both corrected association labels are required")
        for label in LABELS:
            by_label[label]["stage_association"] = by_association[label]
    independent = by_label["independent"]
    if plant == "production-route":
        independent["stage_growth"]["operand_route"] = "retired-route"
    elif plant == "passivity":
        independent["stage_growth"]["passivity"][0]["fields"]["T"] = False
    elif plant == "boundary-order":
        independent["stage_growth"]["boundaries"][1:3] = reversed(
            independent["stage_growth"]["boundaries"][1:3])
    elif plant == "nemo-fold":
        independent["stage_growth"]["boundaries"][1]["fields"]["v"][
            "nemo_fold_identity"]["max_abs"] = 1.0
    elif plant == "growth-owner":
        for boundary in independent["stage_growth"]["boundaries"]:
            for field in FIELDS:
                boundary["fields"][field]["fold_band"]["max_abs"] = 0.0
    elif plant == "source-owner":
        independent["stage_association"]["rows"]["v"][
            "associated_fold_identity"]["unequal"] = 1
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    summaries = {}
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R256_MEASUREMENT",
                f"{label}: measurement status moved")
        stage = report["stage_growth"]
        require(stage["status"] == "PASS_R249_STAGE_TABLE"
                and stage["unit"] is True
                and stage["operand_route"] == "resolved-candidate",
                f"{label}: private-unit route moved")
        require(len(stage["passivity"]) == 7
                and all(all(row["fields"].values())
                        for row in stage["passivity"]),
                f"{label}: stage observer is not passive through kt=7")
        expected_order = [(kt, stg) for kt in range(1, 8) for stg in range(4)]
        actual_order = [(row["kt"], row["stage"])
                        for row in stage["boundaries"]]
        require(actual_order == expected_order,
                f"{label}: kt/stage boundary order moved")
        for boundary in stage["boundaries"]:
            for field in FIELDS:
                nemo_fold = boundary["fields"][field]["nemo_fold_identity"]
                require(float(nemo_fold["max_abs"]) == 0.0,
                        f"{label}: NEMO {field} fold identity moved in value")

        fold_growth = r249._growth_rows(stage["boundaries"], "fold_band")
        interior_growth = r249._growth_rows(stage["boundaries"], "interior")
        first_fold = _first(fold_growth, "over_10x")
        first_interior = _first(interior_growth, "over_10x")
        require(first_fold is not None or first_interior is not None,
                f"{label}: no greater-than-10x boundary was measured")

        kt1 = stage["boundaries"][1]
        assoc = report["stage_association"]
        require(assoc["status"] == "PASS_R256_STAGE_ASSOCIATION_REPLAY"
                and assoc["operand_route"] == "production-d114-resolved"
                and all(assoc["passivity"].values()),
                f"{label}: stage association replay moved")
        require(assoc["source_order"]
                == ["raw_stage_update", "post_stage_lbc_lnk"],
                f"{label}: source order moved")
        owner_rows = {name: assoc["rows"][name] for name in ("T", "S", "v")}
        for name, owner_row in owner_rows.items():
            require(owner_row["nemo_fold_identity"]["unequal"] == 0,
                    f"{label}: NEMO {name} fold identity is not exact")
            require(float(owner_row["raw_fold_identity"]["max_abs"]) > 0.0,
                    f"{label}: raw stage {name} did not break fold identity")
            require(owner_row["associated_fold_identity"]["unequal"] == 0,
                    f"{label}: post-stage association did not close {name}")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        summaries[label] = {
            "kt1_stage1": {
                field: {
                    "fold_band": kt1["fields"][field]["fold_band"],
                    "fold_identity": kt1["fields"][field]["fold_identity"],
                }
                for field in FIELDS
            },
            "growth": {"fold_band": fold_growth, "interior": interior_growth},
            "first_growth": {
                "fold_band": first_fold, "interior": first_interior},
            "source_owner": {
                "statement": "post_stage_lbc_lnk_T_S_v",
                "rows": {
                    name: {
                        "raw_fold_identity": row["raw_fold_identity"],
                        "associated_fold_identity": row[
                            "associated_fold_identity"],
                        "association_delta": row["association_delta"],
                        "associated_fold_band_vs_nemo": row[
                            "associated_fold_band_vs_nemo"],
                    }
                    for name, row in owner_rows.items()
                },
            },
        }
    require(summaries["independent"]["first_growth"]
            == summaries["given_nemo_entry"]["first_growth"],
            "claim labels disagree on the first growth boundary")
    return {
        "format": "nemo-testcase-l4-orca2-round256-classification-v1",
        "status": "HELD_R256_POST_STAGE_T_S_V_ASSOCIATION_NAMED",
        "floor": float(r249.FLOOR),
        "growth_threshold": float(r249.GROWTH),
        "labels": summaries,
        "predictions": {
            "R256-P1": "CONFIRMED",
            "R256-P2": "CONFIRMED",
            "R256-P3": "CONFIRMED_KT1_STAGE1_V_IDENTITY_BREAK",
            "R256-P4": "CONFIRMED_POST_STAGE_LBC_LNK_T_S_V",
            "R256-P5": "CONFIRMED_HELD",
            "R256-P6": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--association-only", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--associations", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.association_only:
            require(all((args.deck_root, args.record_root, args.label,
                         args.expect_commit)),
                    "association measurement arguments are incomplete")
            require(not args.measure and args.classify is None,
                    "association-only mode is exclusive")
            result = _stage_association_measure(
                args.deck_root, args.record_root, args.label,
                args.expect_commit)
        elif args.measure:
            require(all((args.deck_root, args.record_root, args.label,
                         args.expect_commit)),
                    "measurement arguments are incomplete")
            require(args.classify is None, "measure and classify are exclusive")
            result = measure(args.deck_root, args.record_root, args.label,
                             args.expect_commit)
        else:
            require(args.classify is not None,
                    "classification requires both labelled reports")
            reports = [json.loads(path.read_text(encoding="utf-8"))
                       for path in args.classify]
            associations = (
                [json.loads(path.read_text(encoding="utf-8"))
                 for path in args.associations]
                if args.associations else None)
            result = classify(
                reports, associations=associations, plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, KeyError, TypeError, ValueError, GateError,
            r228.GateError, r249.GateError, rung0.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
