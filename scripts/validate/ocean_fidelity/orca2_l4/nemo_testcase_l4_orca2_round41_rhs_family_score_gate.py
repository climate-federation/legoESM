#!/usr/bin/env python3
"""Round 41: score ORCA2's stage-1 momentum families in compiled order."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _package in ("packages/core", "packages/ocean"):
    if str(REPO_ROOT / _package) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round14_barotropic_owner_gate as round14,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round38_ranked_slow_forcing_gate as round38,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round40_rhs_family_gate as round40,
)

FAMILIES = ("hpg", "ldf", "vor", "keg", "zad")
PART_KEYS = {
    "hpg": "after_hpg",
    "ldf": "after_ldf",
    "vor": "after_vor",
    "keg": "after_keg",
    "zad": "after_adv",
}
RAW_KEYS = ("hpg", "ldf", "vorticity", "keg", "zad")
RESTARTS = round40.RESTARTS
PLANTS = (
    "header", "truncation", "swapped-rank", "final-ulp",
    "parent-byte", "restart-byte",
)
_PP = "ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM/BLD/ppsrc/nemo"
CITATIONS = {
    "stage1_momentum_order": f"{_PP}/stp2d.f90:155-187",
    "hpg_sco": f"{_PP}/dynhpg.f90:340-451",
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data(value) -> np.ndarray:
    return np.asarray(value.data if hasattr(value, "data") else value)


def first_non_bit(rows: dict[str, dict[str, dict]]) -> dict | None:
    """Return the first family boundary, with U ordered before V."""
    for family in FAMILIES:
        for face in ("u", "v"):
            row = rows[face][family]
            if not row["bit_exact"]:
                return {"family": family, "face": face, **row}
    return None


def residual_identity(
    candidate: dict[str, np.ndarray], oracle: dict[str, np.ndarray]
) -> dict[str, dict]:
    """Compare every later residual bitwise with the post-HPG residual."""
    baseline = candidate["hpg"] - oracle["hpg"]
    return {
        family: round40._identity(
            candidate[family] - oracle[family], baseline
        )
        for family in FAMILIES[1:]
    }


def admit(root: Path, baseline_root: Path) -> tuple[list[dict], list[dict], dict]:
    """Recheck the immutable acquisition products without producer-tree state."""
    admission_path = root / "round40_rhs_family_admission.json"
    admission = json.loads(admission_path.read_text())
    require(admission.get("status") == "PASS_ACQUISITION",
            "round-40 acquisition is not admitted")
    require(admission.get("label") == "given NEMO's entry",
            "round-40 acquisition label changed")
    records = []
    parents = []
    closure = {}
    for rank in range(2):
        name = round40.RECORD_TEMPLATE.format(rank=rank)
        path = root / name
        require(_sha256(path) == admission["record_sha256"][str(rank)],
                f"rank-{rank} family digest moved")
        stamp = (root / f"{name}.stamp").read_text().split()
        require(len(stamp) == 3 and stamp[0] == _sha256(path)
                and stamp[2] == name, f"rank-{rank} stamp changed")
        record = round40.read_record_bytes(path.read_bytes(), rank)
        parent_name = round40.PARENT_TEMPLATE.format(rank=rank)
        require((root / parent_name).read_bytes()
                == (baseline_root / parent_name).read_bytes(),
                f"rank-{rank} inherited stream moved")
        parent = round38.read_ranked(root / parent_name, rank)
        records.append(record)
        parents.append(parent)
        fields = record["fields"]
        closure[str(rank)] = {}
        for face in ("u", "v"):
            final = fields[f"after_zad_{face}"][2:-2, 2:-2, : round40.NZ - 1]
            row = round40._identity(final, parent[f"krhs_{face}"])
            require(row["bit_exact"], f"rank-{rank} post-ZAD {face} moved")
            closure[str(rank)][face] = row
    for name in RESTARTS:
        require((root / name).read_bytes() == (baseline_root / name).read_bytes(),
                f"restart moved: {name}")
    for plant in PLANTS:
        planted = json.loads((root / f"round40_{plant}_plant.json").read_text())
        log = (root / f"round40_{plant}_plant.log").read_text()
        require(planted.get("status") == "PLANT-FIRED"
                and "STATUS PLANT-FIRED" in log,
                f"round-40 {plant} plant did not fire")
    return records, parents, {
        "status": "ADMITTED_ROUND40_RHS_FAMILIES",
        "producer_commit": admission["worktree"]["commit"],
        "record_sha256": admission["record_sha256"],
        "final_closure": closure,
        "plants_fired": list(PLANTS),
    }


def _candidate_rank(parts: dict, family: str, face: str, rank: int) -> np.ndarray:
    value = _data(parts[f"{PART_KEYS[family]}_{face}"])
    return round38._native_rank(value, face, rank)


def _oracle_rank(record: dict, family: str, face: str) -> np.ndarray:
    fields = record["fields"]
    return fields[f"after_{family}_{face}"][2:-2, 2:-2, : round40.NZ - 1]


def run(
    deck_root: Path,
    boundary_root: Path,
    family_root: Path,
    baseline_root: Path,
    support_json: Path,
    json_out: Path | None,
    *,
    plant: str = "none",
) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    records, parents, admission = admit(family_root, baseline_root)
    _, card = ladder.card_fields(deck_root)
    entry = ladder.assemble_state_fields(boundary_root, 1, stage=None)
    state = round14._seeded_state(card, entry)
    surface_fields = ladder.assemble_surface_fields(boundary_root, 1)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)

    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True),
    )
    ordered = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True,
            nemo_stage_rhs_accumulation_order_arm=True),
    )
    ordinary_trace = jax.device_get(ordinary.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    ordered_trace = jax.device_get(ordered.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    ordinary_parts = ordinary_trace.operator_operands[0]
    ordered_parts = ordered_trace.operator_operands[0]

    raw_identity = {face: {} for face in ("u", "v")}
    for face in ("u", "v"):
        for name in RAW_KEYS:
            raw_identity[face][name] = round40._identity(
                _data(ordinary_parts[f"{name}_{face}"]),
                _data(ordered_parts[f"{name}_{face}"]),
            )
            require(raw_identity[face][name]["bit_exact"],
                    f"source-order arm moved raw {name} {face}")

    rows = {str(rank): {face: {} for face in ("u", "v")} for rank in range(2)}
    candidates = {}
    oracles = {}
    for rank in range(2):
        candidates[rank] = {face: {} for face in ("u", "v")}
        oracles[rank] = {face: {} for face in ("u", "v")}
        for face in ("u", "v"):
            mask = np.asarray(parents[rank][f"{face}mask"]) > 0.0
            for family in FAMILIES:
                candidate = _candidate_rank(ordered_parts, family, face, rank)
                oracle = _oracle_rank(records[rank], family, face)
                candidates[rank][face][family] = candidate
                oracles[rank][face][family] = oracle
                rows[str(rank)][face][family] = round14.compare(
                    candidate, oracle, mask)

    support_document = json.loads(support_json.read_text())
    mismatch = support_document["walk"]["round19_reproduction"][
        "substep1_exit_mismatch"]
    coordinates = np.asarray(mismatch["coordinates"], dtype=np.int64)
    require(mismatch["count"] == 64 and coordinates.shape == (64, 2),
            "round-20 disputed-row support changed")
    support_rows = coordinates[:, 0]
    require(np.unique(support_rows).size == 64,
            "round-20 disputed rows are not unique")
    support_candidate = {
        family: candidates[1]["u"][family][support_rows, -1, :]
        for family in FAMILIES
    }
    support_oracle = {
        family: oracles[1]["u"][family][support_rows, -1, :]
        for family in FAMILIES
    }
    support_mask = np.asarray(parents[1]["umask"])[support_rows, -1, :] > 0.0
    support_rows_scored = {
        family: round14.compare(
            support_candidate[family], support_oracle[family], support_mask)
        for family in FAMILIES
    }
    support_first = next(({
        "family": family, "face": "u", **support_rows_scored[family]
    } for family in FAMILIES if not support_rows_scored[family]["bit_exact"]), None)
    support_residual_identity = residual_identity(
        support_candidate, support_oracle)

    plant_fires = None
    if plant == "first-boundary":
        planted_oracle = dict(support_oracle)
        planted_oracle["hpg"] = np.array(support_candidate["hpg"], copy=True)
        planted_rows = {
            family: round14.compare(
                support_candidate[family], planted_oracle[family], support_mask)
            for family in FAMILIES
        }
        planted_first = next((family for family in FAMILIES
                              if not planted_rows[family]["bit_exact"]), None)
        plant_fires = planted_first != "hpg"
        require(plant_fires, "first-boundary plant did not move the selector")
    elif plant == "residual-equality":
        planted_oracle = {name: np.array(value, copy=True)
                          for name, value in support_oracle.items()}
        plant_fires = False
        for raw_location in np.argwhere(support_mask):
            location = tuple(raw_location)
            original = planted_oracle["ldf"][location]
            for direction in (np.inf, -np.inf):
                planted_oracle["ldf"][location] = np.nextafter(
                    original, direction)
                planted = residual_identity(
                    support_candidate, planted_oracle)
                if not planted["ldf"]["bit_exact"]:
                    plant_fires = True
                    break
                planted_oracle["ldf"][location] = original
            if plant_fires:
                break
        require(plant_fires, "residual-equality plant did not fire")

    result = {
        "gate": "nemo_testcase_l4_orca2_round41_rhs_family_score_gate",
        "status": "PLANT_FIRED" if plant != "none" else "MEASURED",
        "label": "given NEMO's entry",
        "provenance": worktree_stamp(),
        "citations": CITATIONS,
        "admission": admission,
        "raw_terms_unchanged_by_order_arm": raw_identity,
        "full_rank_walks": rows,
        "first_non_bit_full_walk": {
            str(rank): first_non_bit({
                face: rows[str(rank)][face] for face in ("u", "v")})
            for rank in range(2)
        },
        "rank1_disputed_u_rows": {
            "row_count": 64,
            "rows": support_rows.tolist(),
            "walk": support_rows_scored,
            "first_non_bit": support_first,
            "later_residual_equals_hpg": support_residual_identity,
        },
        "plant": plant,
        "plant_fires": plant_fires,
        "single_statement_eligible": False,
    }
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--boundary-root", type=Path, required=True)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--support-json", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--plant", choices=("none", "first-boundary", "residual-equality"),
        default="none")
    args = parser.parse_args()
    try:
        result = run(
            args.deck_root, args.boundary_root, args.family_root,
            args.baseline_root, args.support_json, args.json_out,
            plant=args.plant)
    except (GateError, round38.GateError, round40.GateError,
            OSError, ValueError, KeyError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "admission": result["admission"],
        "first_non_bit_full_walk": result["first_non_bit_full_walk"],
        "rank1_disputed_u_rows": result["rank1_disputed_u_rows"],
        "plant": result["plant"],
        "plant_fires": result["plant_fires"],
    }, indent=2, sort_keys=True))
    return 1 if args.plant != "none" else 0


if __name__ == "__main__":
    raise SystemExit(main())
