#!/usr/bin/env python3
"""Fail-closed gate for round 28's consumer-local raw-F trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round27_consumer_gate as r27,
)


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _read(path: Path) -> dict:
    require(path.is_file(), f"missing artifact: {path}")
    return json.loads(path.read_text())


def _row(document: dict, kt: int, checkpoint: str, field: str) -> dict:
    for item in document["candidate_trajectory"]["checkpoints"]:
        if item["kt"] == kt and item["checkpoint"] == checkpoint:
            return item["rows"][field]
    raise GateError(f"missing score kt={kt}:{checkpoint}:{field}")


def _compare(parent: dict, arm: dict, name: str) -> dict:
    fields = ("T", "S", "u", "v", "ssh")
    before = parent["candidate_trajectory"]["checkpoints"]
    after = arm["candidate_trajectory"]["checkpoints"]
    require(len(after) <= len(before), f"{name}: too many checkpoints")
    moved = []
    exact_left = []
    for bcp, acp in zip(before, after, strict=False):
        require((bcp["kt"], bcp["checkpoint"]) ==
                (acp["kt"], acp["checkpoint"]),
                f"{name}: checkpoint ordering differs")
        for field in fields:
            brow = bcp["rows"][field]
            arow = acp["rows"][field]
            if brow == arow:
                continue
            key = f"kt={acp['kt']}:{acp['checkpoint']}:{field}"
            moved.append(key)
            if brow["bit_identical"] and not arow["bit_identical"]:
                exact_left.append(key)
    return {
        "checkpoint_count": len(after),
        "moved_row_count": len(moved),
        "first_moved_row": moved[0] if moved else None,
        "formerly_exact_rows_left": exact_left,
        "first_non_bit_statement_unchanged": (
            parent["candidate_trajectory"]["first_non_bit_statement"] ==
            arm["candidate_trajectory"]["first_non_bit_statement"]),
    }


def capture_direct_ldf(deck_root: Path, record_root: Path) -> dict:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean import vertical
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy, "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "round 28 is CPU-only")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")

    parent = r27._ldf_inputs(deck_root, record_root, 2)[0]["base"]
    original = vertical.nemo_qco_live_vorticity_e3f_cgrid

    def raw_f(*args, **kwargs):
        kwargs["use_bridge_raw"] = True
        return original(*args, **kwargs)

    vertical.nemo_qco_live_vorticity_e3f_cgrid = raw_f
    try:
        carried = r27._ldf_inputs(deck_root, record_root, 2)[0]["base"]
    finally:
        vertical.nemo_qco_live_vorticity_e3f_cgrid = original

    return {
        "status": "CAPTURED",
        "claim_label": "given NEMO's entry",
        "consumer": "dynldf_lev F-curl thickness",
        "kt": 2,
        "base_vs_carried_raw_f": {
            "u": r27.array_score(parent[0], carried[0]),
            "v": r27.array_score(parent[1], carried[1]),
        },
        "worktree": worktree_stamp(),
    }


def evaluate(
    parent: dict,
    een: dict,
    ldf: dict,
    shared: dict,
    refusal_text: str,
    direct: dict,
    vorticity_parent: dict,
    vorticity_ldf: dict,
    *,
    plant: str | None = None,
) -> dict:
    require(parent["status"] == een["status"] == ldf["status"] ==
            shared["status"] == "LADDER_MEASURED", "a ladder is not measured")
    require(len(parent["candidate_trajectory"]["checkpoints"]) == 40,
            "parent ladder is incomplete")
    require(len(een["candidate_trajectory"]["checkpoints"]) == 12,
            "EEN pre-refusal ladder must contain kt=1..3")
    require(len(ldf["candidate_trajectory"]["checkpoints"]) == 40,
            "LDF-only ladder is incomplete")
    require(parent["card"] == een["card"] == ldf["card"],
            "resolved cards differ")
    for name, document in (("parent", parent), ("EEN", een), ("LDF", ldf)):
        require(document["worktree"]["clean"], f"{name} worktree is dirty")

    refusal = "raw-mesh e3w_int must contain only finite values > 0"
    require(refusal in refusal_text, "EEN-only kt=4 refusal is absent")
    require("REFUSE: the production step raised an unregistered refusal" in
            refusal_text, "EEN-only run did not fail closed")

    een_result = _compare(parent, een, "EEN-only")
    ldf_result = _compare(parent, ldf, "LDF-only")
    if plant == "exact_row":
        ldf_result["formerly_exact_rows_left"] = ["PLANTED"]
    require(not een_result["formerly_exact_rows_left"],
            "EEN-only moved a formerly exact row")
    require(not ldf_result["formerly_exact_rows_left"],
            "LDF-only moved a formerly exact row")
    require(een_result["first_non_bit_statement_unchanged"],
            "EEN-only changed the first NEMO mismatch")
    require(ldf_result["first_non_bit_statement_unchanged"],
            "LDF-only changed the first NEMO mismatch")

    shared_u = _row(shared, 1, "stage2", "u")
    shared_v = _row(shared, 1, "stage2", "v")
    require(_row(een, 1, "stage2", "u") == shared_u and
            _row(een, 1, "stage2", "v") == shared_v,
            "EEN-only does not reproduce the shared arm's first movement")
    require(een_result["first_moved_row"] == "kt=1:stage2:u",
            "EEN-only first movement is not kt=1 stage-2 U")
    require(_row(ldf, 1, "stage2", "u") ==
            _row(parent, 1, "stage2", "u") and
            _row(ldf, 1, "stage2", "v") ==
            _row(parent, 1, "stage2", "v"),
            "LDF-only changes the kt=1 zero-LDF checkpoint")

    labels = {
        "een_only": "dynvor EEN potential-vorticity thickness",
        "ldf_only": "dynldf_lev F-curl thickness",
    }
    if plant == "consumer_labels":
        labels = {"een_only": labels["ldf_only"],
                  "ldf_only": labels["een_only"]}
    require(labels["een_only"].startswith("dynvor"),
            "EEN arm consumer label is wrong")
    require(labels["ldf_only"].startswith("dynldf"),
            "LDF arm consumer label is wrong")

    direct_rows = direct["base_vs_carried_raw_f"]
    require(direct["consumer"] == labels["ldf_only"],
            "direct result is not the LDF consumer")
    require(not direct_rows["u"]["bit_identical"] and
            not direct_rows["v"]["bit_identical"],
            "carried raw F does not change the direct non-rest LDF result")
    require(vorticity_parent["stage2_vorticity"]["digest"] ==
            vorticity_ldf["stage2_vorticity"]["digest"],
            "LDF-only arm changed the EEN vorticity digest")

    predictions = {
        "R28-P1": True,
        "R28-P2": True,
        "R28-P3": True,
        "R28-P4": True,
        "R28-P5": plant is None,
    }
    return {
        "status": "HELD",
        "claim_label": "independent with Decision-52 SSH",
        "consumer_labels": labels,
        "een_only": een_result,
        "ldf_only": ldf_result,
        "kt10_stage3": {
            field: {
                "parent": _row(parent, 10, "stage3", field),
                "ldf_only": _row(ldf, 10, "stage3", field),
            }
            for field in ("u", "v")
        },
        "direct_ldf": direct_rows,
        "predictions": {key: "CONFIRMED" if value else "REFUTED"
                        for key, value in predictions.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture-direct-ldf")
    capture.add_argument("--deck-root", type=Path, required=True)
    capture.add_argument("--record-root", type=Path, required=True)
    capture.add_argument("--json-out", type=Path, required=True)
    outcome = sub.add_parser("outcome")
    for name in ("parent", "een", "ldf", "shared", "direct",
                 "vorticity-parent", "vorticity-ldf"):
        outcome.add_argument(f"--{name}", type=Path, required=True)
    outcome.add_argument("--een-refusal", type=Path, required=True)
    outcome.add_argument("--json-out", type=Path)
    outcome.add_argument("--plant", choices=("exact_row", "consumer_labels"))
    args = parser.parse_args()
    try:
        if args.command == "capture-direct-ldf":
            result = capture_direct_ldf(args.deck_root, args.record_root)
        else:
            result = evaluate(
                _read(args.parent), _read(args.een), _read(args.ldf),
                _read(args.shared), args.een_refusal.read_text(),
                _read(args.direct), _read(args.vorticity_parent),
                _read(args.vorticity_ldf), plant=args.plant)
    except (GateError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    return 2 if result["status"] == "HELD" else 0


if __name__ == "__main__":
    raise SystemExit(main())
