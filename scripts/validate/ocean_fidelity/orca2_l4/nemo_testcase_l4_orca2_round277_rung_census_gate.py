#!/usr/bin/env python3
"""Round 277: ascending-sum census in ladder order and the G3 stored-r3u arm.

Reuses round 276's arm selection, ladder and early/late classifier.  Adds
the shipped rung-10 card under BOTH labels (``given_nemo_entry`` with the
Decision-52 entry SSH, ``independent`` without it) and any arm on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round276_omt_source_census_gate as r276,
)

GateError = r276.GateError
require = r276.require

STAGE_R3U = "zdf_momentum_stage_r3u_nemo_literal"
# dynzdf.f90:192-198 reads r3u(Kmm) only inside the face-thickness divisor
# statement, so the stored ratio has no host without it.
r276.MODEL.add(STAGE_R3U)
r276.ARMS["g3_divisors_stage_r3u"] = (
    "zdf_momentum_nemo_face_thickness", STAGE_R3U)
RUNGS = ("rung0", "omt1", "omt2", "omt3", "omt4", "rung10")


def _rung10_ladder(deck_root: Path, twin_a: Path, arm: str, label: str):
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    )

    _, base = rung10.card_fields(deck_root)
    cfg = base.recipe.model_config
    on = sorted(name for name in r276.MODEL if getattr(cfg, name, False))
    on += sorted(name for name in r276.BAROTROPIC
                 if getattr(cfg.barotropic, name, False))
    if base.recipe.z_coord.nemo_r3t_stored_product:
        on.append("nemo_r3t_stored_product")
    require(not on, f"production rung 10 already selects {on}")
    card = r276.arm_card(base, arm)
    trajectory = rung10.candidate_trajectory(
        deck_root, twin_a, card, max_step=10,
        bridge_ssh=(label == "given_nemo_entry"))
    rows, first = [], None
    for checkpoint in trajectory["checkpoints"]:
        if first is None and checkpoint["first_non_bit_field"] is not None:
            first = {"kt": checkpoint["kt"],
                     "checkpoint": checkpoint["checkpoint"],
                     "field": checkpoint["first_non_bit_field"]}
        for field, score in checkpoint["rows"].items():
            rows.append({"kt": checkpoint["kt"],
                         "checkpoint": checkpoint["checkpoint"],
                         "field": field, **score})
    require(len(rows) == 200 and first is not None,
            "shipped-card arm ladder is incomplete")
    return {"rows": rows, "row_count": len(rows),
            "first_non_bit_checkpoint": first}


def measure(rung: str, deck_root: Path, twin_a: Path, arm: str,
            label: str, expect_commit: str) -> dict:
    require(rung in RUNGS, f"unsupported rung {rung}")
    if rung != "rung10":
        require(label == "independent",
                "rung 0 and OMT ladders carry the independent label only")
        result = r276.measure(rung, deck_root, twin_a, arm, expect_commit,
                              "none")
    else:
        require(label in ("given_nemo_entry", "independent"),
                f"unknown label {label}")
        import jax
        from legoesm.core.precision import (
            PrecisionPolicy, get_policy, set_policy)
        from legoesm.ocean.fidelity.provenance import worktree_stamp

        stamp = worktree_stamp()
        require(stamp["clean"]
                and stamp["commit"].lower() == expect_commit.lower(),
                "round-277 measurement requires its clean committed tree")
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
        require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
                and bool(jax.config.jax_enable_x64), "fp64/libm is not active")
        require(jax.default_backend() == "cpu"
                and not jax.config.jax_disable_jit,
                "round-277 measurement requires production JIT on CPU")
        result = {
            "rung": rung, "label": label, "arm": arm,
            "selected_statements": list(r276.ARMS[arm]),
            "worktree": stamp,
            "execution": "production-jit-cpu-fp64-x64-libm",
            "ladder": _rung10_ladder(deck_root, twin_a, arm, label),
        }
    result["format"] = "nemo-testcase-l4-orca2-round277-source-ladder-v1"
    # The classifier is round 276's; it admits only its own ladder status.
    result["status"] = "PASS_R276_SOURCE_LADDER"
    result["round"] = 277
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--rung", choices=RUNGS)
    parser.add_argument("--label", default="independent",
                        choices=("independent", "given_nemo_entry"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--arm", choices=tuple(r276.ARMS))
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--plant", choices=(
        "none", "exact-loss", "r3u-without-host"), default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.plant == "r3u-without-host":
            # The stored ratio alone must refuse, never run as a no-op arm.
            from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
                LatLonCGridOceanModel)
            from scripts.validate.ocean_fidelity.orca2_l4 import (
                nemo_testcase_l4_orca2_round269_g1_decision96_gate as prior,
            )
            card = r276.arm_card(
                prior._selected_card("rung0", args.deck_root), "control")
            cfg = card.recipe.model_config._replace(**{STAGE_R3U: True})
            from scripts.validate.ocean_fidelity.orca2_l4 import (
                nemo_testcase_l4_orca2_round103_rung0_ladder_gate as ladder,
            )
            freshwater, surface = ladder._zero_forcing((148, 180))
            model = LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord, cfg)
            try:
                model.step(card.recipe.initial_state, card.dt_s,
                           freshwater=freshwater, surface_forcing=surface)
            except ValueError as error:
                require(STAGE_R3U + " requires" not in str(error),
                        f"stored r3u refused without its host: {error}")
                raise
            result = {"status": "R3U_WITHOUT_HOST_RAN"}
        elif args.measure:
            require(all((args.rung, args.deck_root, args.twin_a, args.arm,
                         args.expect_commit)),
                    "measurement arguments are incomplete")
            result = measure(args.rung, args.deck_root, args.twin_a,
                             args.arm, args.label, args.expect_commit)
        else:
            require(args.classify is not None,
                    "classification inputs are required")
            result = r276.classify(*args.classify, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, OSError, KeyError, TypeError, ValueError,
            StopIteration) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
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
