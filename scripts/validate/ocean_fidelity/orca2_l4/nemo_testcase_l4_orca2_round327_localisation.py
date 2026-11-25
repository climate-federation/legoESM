#!/usr/bin/env python3
"""Round 327: localise rung-10's first residual without changing the card."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
)

FIELD_ORDER = ("T", "S", "u", "v", "ssh")
CHECKPOINT_ORDER = ("entry", "stage1", "stage2", "stage3")
PLANTS = ("none", "row-order", "owner-replay", "locality-partition", "ranking")
EXPECTED_FIRST = {
    "kt": 1,
    "checkpoint": "entry",
    "field": "ssh",
    "unequal": 16433,
    "count": 26640,
    "max_abs": 0.015479333813968585,
    "rms": 0.012157480547162937,
}
EXPECTED_KT10 = {
    "ssh": (0.018533089817820425, 3.734978452696325e-05),
    "T": (0.011429686050284715, 7.703006966558457e-05),
    "u": (0.005295100146905263, 0.003046847350500703),
    "v": (0.0031473714762187688, 0.003149083070336161),
    "S": (0.0018793894167785652, 1.2237551899829668e-05),
}


class GateError(RuntimeError):
    """The localisation no longer supports its registered claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_rows(ladder: dict[str, object]) -> list[dict[str, object]]:
    rows = list(ladder["rows"])
    return sorted(
        rows,
        key=lambda row: (
            int(row["kt"]),
            CHECKPOINT_ORDER.index(str(row["checkpoint"])),
            FIELD_ORDER.index(str(row["field"])),
        ),
    )


def _first_numerical_departure(ladder: dict[str, object]) -> dict[str, object] | None:
    """The exact numerical bar ignores signed-zero-only bit differences."""
    return next(
        (row for row in _ordered_rows(ladder) if float(row["max_abs"]) > 0.0),
        None,
    )


def _row(ladder: dict[str, object], kt: int, checkpoint: str,
         field: str) -> dict[str, object]:
    matches = [
        row for row in ladder["rows"]
        if int(row["kt"]) == kt
        and row["checkpoint"] == checkpoint
        and row["field"] == field
    ]
    require(len(matches) == 1, f"missing or duplicate row {kt}/{checkpoint}/{field}")
    return matches[0]


def _candidate_fields(card) -> dict[str, np.ndarray]:
    state = card.recipe.initial_state
    return {
        "T": np.asarray(state.T.data, dtype=np.float64)[..., :30],
        "S": np.asarray(state.S.data, dtype=np.float64)[..., :30],
        "u": np.asarray(state.u.data, dtype=np.float64)[:, 1:, :30],
        "v": np.asarray(state.v.data, dtype=np.float64)[1:, :, :30],
        "ssh": np.asarray(state.eta.data, dtype=np.float64),
    }


def _coastal_mask(wet: np.ndarray) -> np.ndarray:
    """Four-neighbour coast on periodic-i/tripole-j geometry."""
    nj, ni = wet.shape
    east = np.roll(wet, -1, axis=1)
    west = np.roll(wet, 1, axis=1)
    south = np.empty_like(wet)
    south[0] = wet[0]
    south[1:] = wet[:-1]
    north = np.empty_like(wet)
    north[:-1] = wet[1:]
    fold_i = (-np.arange(ni)) % ni
    north[-1] = wet[-1, fold_i]
    return wet & ~(east & west & south & north)


def _cell(j: int, i: int, *, lat: np.ndarray, lon: np.ndarray,
          fr_i: np.ndarray, coast: np.ndarray, nj: int, ni: int) -> dict[str, object]:
    return {
        "index_ji_zero_based": [int(j), int(i)],
        "latitude_deg": float(lat[j, i]),
        "longitude_deg": float(lon[j, i]),
        "vertical_index": None,
        "depth": "surface_2d_ssh",
        "ice_fraction": float(fr_i[j, i]),
        "ice_covered": bool(fr_i[j, i] > 0.0),
        "coastal": bool(coast[j, i]),
        "fold_row": bool(j == nj - 1),
        "zonal_seam": bool(i in (0, ni - 1)),
    }


def _partition(mask: np.ndarray, selector: np.ndarray) -> dict[str, int]:
    return {
        "true": int(np.count_nonzero(mask & selector)),
        "false": int(np.count_nonzero(mask & ~selector)),
    }


def _score(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    return rung10.score(np.asarray(candidate), np.asarray(reference))


def measure(deck_root: Path, frames_root: Path, independent_json: Path,
            given_json: Path, expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_orca2_zps_card,
        orca2_l3_eiv_gm_redi,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-327 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-327 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu", "localisation must run on CPU")

    independent = json.loads(independent_json.read_text(encoding="utf-8"))
    given = json.loads(given_json.read_text(encoding="utf-8"))
    for label, payload in (("independent", independent), ("given_nemo_entry", given)):
        require(payload["label"] == label, f"{label}: ladder label moved")
        require(payload["rung"] == "rung10" and payload["arm"] == "mxlice",
                f"{label}: not the current rung-10 mxlice arm")
        require(payload["status"] == "PASS_R276_SOURCE_LADDER",
                f"{label}: ladder did not complete")

    base = build_orca2_zps_card(deck_root)
    card = rung10.comparison_card(base)
    config = card.recipe.model_config
    require(config.implicit_ice_ocean_top_drag,
            "current comparison card lost the top-drag replay")
    require(config.physics.vertical_mixing.tke.nemo_mxlice == 2,
            "current comparison card lost nn_mxlice=2")
    require(config.gm_redi == orca2_l3_eiv_gm_redi()
            and config.rk3_stage3_l9_eiv_nemo_order,
            "current comparison card lost L3/L9")

    first_independent = _first_numerical_departure(independent["ladder"])
    first_given = _first_numerical_departure(given["ladder"])
    require(first_independent is not None and first_given is not None,
            "a ladder has no numerical departure")
    same_given_rows = {
        field: _row(given["ladder"], 1, "entry", field)
        for field in FIELD_ORDER
    }

    oracle = rung10.assemble_state_fields(frames_root, 1, stage=None)
    candidate = _candidate_fields(card)
    entry_rows = {field: _score(oracle[field], candidate[field])
                  for field in FIELD_ORDER}
    mismatch = candidate["ssh"].view(np.uint64) != oracle["ssh"].view(np.uint64)
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    require(mismatch.shape == wet.shape, "SSH/wet-mask shape moved")
    delta = candidate["ssh"] - oracle["ssh"]
    signed_bits = np.ascontiguousarray(delta[mismatch]).view(np.uint64)
    unique_signed = np.unique(signed_bits)

    surface = rung10.assemble_surface_fields(frames_root, 1)
    fr_i = np.asarray(surface["fr_i"], dtype=np.float64)
    coast = _coastal_mask(wet)
    nj, ni = mismatch.shape
    lat = np.asarray(card.recipe.grid.native_lat_T_deg, dtype=np.float64)
    lon = np.rad2deg(np.asarray(card.recipe.grid.lon_T, dtype=np.float64))
    first_j, first_i = np.argwhere(mismatch)[0]
    max_j, max_i = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)

    replay = {field: np.array(value, copy=True) for field, value in candidate.items()}
    replay["ssh"] = np.array(oracle["ssh"], copy=True)
    replay_rows = {field: _score(oracle[field], replay[field]) for field in FIELD_ORDER}

    ranking = []
    for field in FIELD_ORDER:
        left = _row(independent["ladder"], 10, "entry", field)
        right = _row(given["ladder"], 10, "entry", field)
        ranking.append({
            "field": field,
            "ranking_max_abs": max(float(left["max_abs"]), float(right["max_abs"])),
            "independent": {key: left[key] for key in ("max_abs", "rms", "unequal", "count")},
            "given_nemo_entry": {key: right[key] for key in ("max_abs", "rms", "unequal", "count")},
            "statement_owner": (
                "initial_ice_mass_ssh_adjustment" if field == "ssh"
                else "UNMEASURED_COMPOUNDED_FAMILY"
            ),
            "owner_claim": "CONFIRMED" if field == "ssh" else "PLAUSIBLE",
        })
    ranking.sort(key=lambda row: (-float(row["ranking_max_abs"]),
                                  FIELD_ORDER.index(str(row["field"]))))

    report = {
        "format": "nemo-testcase-l4-orca2-round327-localisation-v1",
        "execution": "offline-passive-entry-replay-cpu-fp64-libm",
        "current_comparison_card": {
            "implicit_ice_ocean_top_drag": True,
            "l3_eiv_l9_order": True,
            "nemo_mxlice": 2,
        },
        "first_numerical_departure": {
            "independent": {key: first_independent[key] for key in EXPECTED_FIRST},
            "given_nemo_entry": {
                "kt": int(first_given["kt"]),
                "checkpoint": first_given["checkpoint"],
                "field": first_given["field"],
                "max_abs": first_given["max_abs"],
            },
            "given_at_independent_row": same_given_rows,
        },
        "entry_localisation": {
            "first_cell": _cell(first_j, first_i, lat=lat, lon=lon, fr_i=fr_i,
                                coast=coast, nj=nj, ni=ni),
            "maximum_cell": _cell(max_j, max_i, lat=lat, lon=lon, fr_i=fr_i,
                                  coast=coast, nj=nj, ni=ni),
            "unequal": int(np.count_nonzero(mismatch)),
            "wet_cells": int(np.count_nonzero(wet)),
            "all_and_only_wet": bool(np.array_equal(mismatch, wet)),
            "unique_signed_offsets": int(unique_signed.size),
            "candidate_minus_nemo_m": float(delta[mismatch][0]),
            "ice_covered_partition": _partition(mismatch, fr_i > 0.0),
            "coastal_partition": _partition(mismatch, coast),
            "fold_partition": _partition(
                mismatch, np.broadcast_to(np.arange(nj)[:, None] == nj - 1, mismatch.shape)),
            "seam_partition": _partition(
                mismatch, np.broadcast_to(np.isin(np.arange(ni), (0, ni - 1)), mismatch.shape)),
            "coastal_definition": "wet T cell with a dry four-neighbour; periodic i, tripole north",
        },
        "entry_rows": entry_rows,
        "owner_output_replay": {
            "substitution": "replace_only_card_ssh_with_passive_completed_nemo_entry_ssh",
            "rows": replay_rows,
            "all_fields_bit_identical": all(row["bit_identical"] for row in replay_rows.values()),
        },
        "inheritance": "INHERITED_ENTRY",
        "owner": {
            "claim": "CONFIRMED",
            "nemo_statement": "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:442-459",
            "legoesm_statement": "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1762-1796",
            "certified_rung": None,
            "rung_note": "G5 consumes the completed ice/ocean state; no certified rung constructs this initial SI3 identity",
        },
        "kt10_family_ranking": ranking,
        "inputs": {
            "frames_root": str(frames_root),
            "independent_json": {"path": str(independent_json), "sha256": sha256(independent_json)},
            "given_json": {"path": str(given_json), "sha256": sha256(given_json)},
        },
        "worktree": stamp,
        "in_executable_observers": 0,
    }
    return classify(report)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "row-order":
        report["first_numerical_departure"]["independent"]["checkpoint"] = "stage1"
    elif plant == "owner-replay":
        report["owner_output_replay"]["rows"]["ssh"]["unequal"] = 1
        report["owner_output_replay"]["all_fields_bit_identical"] = False
    elif plant == "locality-partition":
        report["entry_localisation"]["ice_covered_partition"]["true"] -= 1
    elif plant == "ranking":
        report["kt10_family_ranking"][0]["field"] = "T"

    require(report.get("execution") == "offline-passive-entry-replay-cpu-fp64-libm",
            "execution policy moved")
    require(report.get("in_executable_observers") == 0,
            "an in-executable observer entered the replay")
    first = report["first_numerical_departure"]["independent"]
    require(first == EXPECTED_FIRST, "independent first departure moved")
    given_same = report["first_numerical_departure"]["given_at_independent_row"]
    require(all(float(given_same[field]["max_abs"]) == 0.0 for field in FIELD_ORDER),
            "given-entry label is not exact at the independent first row")
    loc = report["entry_localisation"]
    require(loc["unequal"] == EXPECTED_FIRST["unequal"]
            and loc["wet_cells"] == EXPECTED_FIRST["unequal"]
            and loc["all_and_only_wet"], "entry mismatch is not exactly the wet domain")
    require(loc["unique_signed_offsets"] == 1, "entry SSH offset is not uniform")
    for name in ("ice_covered_partition", "coastal_partition", "fold_partition", "seam_partition"):
        require(sum(loc[name].values()) == loc["unequal"], f"{name} does not partition the mismatch")
    replay = report["owner_output_replay"]
    require(replay["all_fields_bit_identical"]
            and all(row["unequal"] == 0 for row in replay["rows"].values()),
            "one-field owner-output replay did not close the entry")
    require(report["inheritance"] == "INHERITED_ENTRY", "inheritance verdict moved")
    require(report["owner"]["claim"] == "CONFIRMED"
            and report["owner"]["certified_rung"] is None,
            "initial SSH ownership/rung verdict moved")
    ranking = report["kt10_family_ranking"]
    require([row["field"] for row in ranking] == list(EXPECTED_KT10),
            "kt10 field-family ranking moved")
    for row in ranking:
        expected = EXPECTED_KT10[row["field"]]
        observed = (row["independent"]["max_abs"], row["given_nemo_entry"]["max_abs"])
        require(observed == expected, f"{row['field']}: kt10 maxima moved")
    report["predictions"] = {
        "R327-P1": "CONFIRMED",
        "R327-P2": "CONFIRMED",
        "R327-P3": "CONFIRMED",
        "R327-P4": "CONFIRMED",
        "R327-P5": "CONFIRMED" if plant == "none" else "PLANT",
    }
    report["status"] = "PASS_R327_LOCALISATION"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--independent-json", type=Path)
    parser.add_argument("--given-json", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(not any((args.deck_root, args.frames_root, args.independent_json,
                             args.given_json, args.expect_commit)),
                    "classification mode cannot take measurement arguments")
            raw = json.loads(args.classify_json.read_text(encoding="utf-8"))
            result = classify(raw, plant=args.plant)
        else:
            require(args.plant == "none", "plants require --classify-json")
            require(all((args.deck_root, args.frames_root, args.independent_json,
                         args.given_json, args.expect_commit)),
                    "measurement arguments incomplete")
            result = measure(args.deck_root, args.frames_root, args.independent_json,
                             args.given_json, args.expect_commit)
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
