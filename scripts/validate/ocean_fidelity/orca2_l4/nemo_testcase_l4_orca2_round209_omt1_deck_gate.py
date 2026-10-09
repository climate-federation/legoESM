#!/usr/bin/env python3
"""Render and validate Decision-109 ORCA2 mini-ladder rung OMT-1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round199_omt0_record_gate as omt0,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


SOURCE_SHA256 = "9279c638a9fe50b551fcca8dc62dc1c19c0fc2c55bdcae7f3501a8afeb5ca8ac"
CHANGED = {
    "namdyn_adv.ln_dynadv_off": ".false.",
    "namdyn_adv.ln_dynadv_vec": ".true.",
}
RETAINED_OFF = {
    "namdyn_ldf.ln_dynldf_off": ".true.",
    "namtra_adv.ln_traadv_off": ".true.",
    "namtra_ldf.ln_traldf_off": ".true.",
    "namdrg.ln_drg_off": ".true.",
}
PLANTS = ("none", "extra-delta", "wrong-selector")


class GateError(RuntimeError):
    """The OMT-1 deck is not the admitted one-module edge from OMT-0."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normal(value: str) -> str:
    return omt0._normal(value)


def render_omt1(source: str) -> str:
    """Change only OMT-0's mutually-exclusive momentum-advection selectors."""
    text = source
    for qualified, value in CHANGED.items():
        text = omt0.rung0_deck._replace_in_group(text, qualified, value)
    return text


def validate_deck(source: Path, candidate: Path, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    require(sha256(source) == SOURCE_SHA256, "admitted OMT-0 deck changed")
    expected = render_omt1(source.read_text())
    actual = candidate.read_text()
    if plant == "extra-delta":
        actual = actual.replace("nn_fsbc     = 2", "nn_fsbc     = 3", 1)
    elif plant == "wrong-selector":
        actual, count = re.subn(
            r"(?mi)^(\s*ln_dynadv_vec\s*=\s*)\.true\.",
            r"\g<1>.false.", actual, count=1,
        )
        require(count == 1, "selector plant target disappeared")
    require(actual == expected, "candidate is not the exact OMT-1 rendering")

    before = namelist_values(source)
    after = namelist_values(candidate)
    changed = sorted(
        key for key in before
        if key in after and _normal(before[key]) != _normal(after[key])
    )
    require(changed == sorted(CHANGED), f"OMT-1 assignment delta moved: {changed}")
    require(set(after) == set(before), "OMT-1 added or removed an assignment")
    for key, wanted in {**CHANGED, **RETAINED_OFF}.items():
        require(key in after and _normal(after[key]) == wanted,
                f"{key}: resolved value moved")
    return {
        "format": "nemo-testcase-l4-orca2-round209-omt1-deck-v1",
        "status": "PASS_R209_OMT1_DECK",
        "source_sha256": SOURCE_SHA256,
        "candidate_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "changed_assignments": changed,
        "retained_off_assignments": RETAINED_OFF,
    }


def render_run_deck(text: str, *, itend: int, stock: int,
                    restart_steps: tuple[int, ...]) -> str:
    return omt0.render_run_deck(
        text, itend=itend, stock=stock, restart_steps=restart_steps,
    )


def validate_run_deck(canonical: Path, root: Path, *, itend: int, stock: int,
                      restart_steps: tuple[int, ...]) -> dict:
    row = omt0.validate_run_deck(
        canonical, root, itend=itend, stock=stock,
        restart_steps=restart_steps,
    )
    ocean = (root / "ocean.output").read_text()
    required = {
        "ln_dynadv_OFF": r"linear dynamics : no momentum advection\s+ln_dynadv_OFF\s+=\s+F\b",
        "ln_dynadv_vec": r"Vector form: 2nd order centered scheme\s+ln_dynadv_vec\s+=\s+T\b",
        "vector_program": r"==>>>\s+vector form : keg \+ zad \+ vor is used",
        "ln_dynldf_OFF": r"no explicit diffusion\s+ln_dynldf_OFF\s+=\s+T\b",
        "ln_traadv_OFF": r"No advection on T & S\s+ln_traadv_OFF\s+=\s+T\b",
        "ln_traldf_OFF": r"no explicit diffusion\s+ln_traldf_OFF\s+=\s+T\b",
        "ln_drg_OFF": r"free-slip\s+: Cd = 0\s+ln_drg_OFF\s+=\s+T\b",
    }
    for label, pattern in required.items():
        require(re.search(pattern, ocean), f"resolved OMT-1 selector moved: {label}")
    return {**row, "resolved_omt1": sorted(required)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--render-deck", type=Path)
    parser.add_argument("--render-run-deck", type=Path)
    parser.add_argument("--itend", type=int)
    parser.add_argument("--stock", type=int)
    parser.add_argument("--restart-steps")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.render_deck:
            require(args.source is not None and args.plant == "none",
                    "deck rendering requires source and no plant")
            require(sha256(args.source) == SOURCE_SHA256,
                    "admitted OMT-0 deck changed")
            args.render_deck.write_text(render_omt1(args.source.read_text()))
            print(f"STATUS RENDERED_R209_OMT1_DECK {args.render_deck}")
            return 0
        if args.render_run_deck:
            require(args.candidate is not None and args.itend and args.stock
                    and args.restart_steps and args.plant == "none",
                    "run-deck rendering requires canonical deck and controls")
            steps = tuple(int(item) for item in args.restart_steps.split(","))
            args.render_run_deck.write_text(render_run_deck(
                args.candidate.read_text(), itend=args.itend,
                stock=args.stock, restart_steps=steps,
            ))
            print(f"STATUS RENDERED_R209_OMT1_RUN_DECK {args.render_run_deck}")
            return 0
        require(args.source is not None and args.candidate is not None,
                "deck validation requires source and candidate")
        result = validate_deck(args.source, args.candidate, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, omt0.GateError, OSError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
