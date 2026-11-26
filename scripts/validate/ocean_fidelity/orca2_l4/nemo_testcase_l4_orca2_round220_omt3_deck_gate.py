#!/usr/bin/env python3
"""Render and validate Decision-109 OMT-3 lateral momentum diffusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round199_omt0_record_gate as protocol,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round218_omt2_deck_gate as omt2,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


SOURCE_SHA256 = "bc5fc68188e4d79479acce38085e497f9f2543a36d6544f1dc1cea95bf51d169"
CHANGED = {"namdyn_ldf.ln_dynldf_lap": ".true."}
REMOVED = {"namdyn_ldf.ln_dynldf_off": ".true."}
RETAINED = {
    "namdyn_adv.ln_dynadv_off": ".false.",
    "namdyn_adv.ln_dynadv_vec": ".true.",
    "namdrg.ln_lin": ".true.",
    "namtra_adv.ln_traadv_off": ".true.",
    "namtra_ldf.ln_traldf_off": ".true.",
}
PLANTS = ("none", "extra-delta", "wrong-selector")


class GateError(RuntimeError):
    """The candidate is not the exact OMT-2 to OMT-3 module edge."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normal(value: str) -> str:
    return protocol._normal(value)


def render_omt3(source: str) -> str:
    """Restore exactly the rung-0 lateral momentum-diffusion selector."""
    text = omt2.omt1.omt0.rung0_deck._replace_in_group(
        source, "namdyn_ldf.ln_dynldf_lap", ".true.",
    )
    lines = text.splitlines(keepends=True)
    in_group = False
    removed = 0
    rendered: list[str] = []
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("&"):
            in_group = stripped.split()[0] == "&namdyn_ldf"
        if in_group and re.match(r"(?i)^\s*ln_dynldf_off\s*=", line):
            removed += 1
            continue
        rendered.append(line)
        if in_group and stripped.startswith("/"):
            in_group = False
    require(removed == 1, "ln_dynldf_off override was not removed exactly once")
    return "".join(rendered)


def validate_deck(source: Path, candidate: Path, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    require(sha256(source) == SOURCE_SHA256, "admitted OMT-2 deck changed")
    expected = render_omt3(source.read_text())
    actual = candidate.read_text()
    if plant == "extra-delta":
        actual = actual.replace("nn_fsbc     = 2", "nn_fsbc     = 3", 1)
    elif plant == "wrong-selector":
        actual, count = re.subn(
            r"(?mi)^(\s*ln_dynldf_lap\s*=\s*)\.true\.",
            r"\g<1>.false.", actual, count=1,
        )
        require(count == 1, "lateral-diffusion plant target disappeared")
    require(actual == expected, "candidate is not the exact OMT-3 rendering")

    before = namelist_values(source)
    after = namelist_values(candidate)
    changed = sorted(
        key for key in before if key in after
        and _normal(before[key]) != _normal(after[key])
    )
    removed = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    require(changed == sorted(CHANGED), f"OMT-3 changed assignments moved: {changed}")
    require(removed == sorted(REMOVED), f"OMT-3 removed assignments moved: {removed}")
    require(not added, f"OMT-3 added assignments: {added}")
    for key, wanted in {**CHANGED, **RETAINED}.items():
        require(key in after and _normal(after[key]) == wanted,
                f"{key}: resolved deck value moved")
    return {
        "format": "nemo-testcase-l4-orca2-round220-omt3-deck-v1",
        "status": "PASS_R220_OMT3_DECK",
        "source_sha256": SOURCE_SHA256,
        "candidate_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "changed_assignments": changed,
        "removed_assignments": removed,
        "retained_assignments": RETAINED,
        "resolved_from_namelist_ref": {
            "namdyn_ldf.ln_dynldf_off": ".false.",
            "namdyn_ldf.ln_dynldf_lev": ".true.",
            "namdyn_ldf.nn_ahm_ijk_t": "-30",
        },
    }


def render_run_deck(text: str, *, itend: int, stock: int,
                    restart_steps: tuple[int, ...]) -> str:
    return protocol.render_run_deck(
        text, itend=itend, stock=stock, restart_steps=restart_steps,
    )


def validate_run_deck(canonical: Path, root: Path, *, itend: int, stock: int,
                      restart_steps: tuple[int, ...]) -> dict:
    row = protocol.validate_run_deck(
        canonical, root, itend=itend, stock=stock,
        restart_steps=restart_steps,
    )
    ocean = (root / "ocean.output").read_text()
    required = {
        "ln_dynadv_OFF": r"linear dynamics : no momentum advection\s+ln_dynadv_OFF\s+=\s+F\b",
        "ln_dynadv_vec": r"Vector form: 2nd order centered scheme\s+ln_dynadv_vec\s+=\s+T\b",
        "vector_program": r"==>>>\s+vector form : keg \+ zad \+ vor is used",
        "ln_dynldf_OFF": r"no explicit diffusion\s+ln_dynldf_OFF\s+=\s+F\b",
        "ln_dynldf_lap": r"laplacian operator\s+ln_dynldf_lap\s+=\s+T\b",
        "ln_dynldf_lev": r"iso-level\s+ln_dynldf_lev\s+=\s+T\b",
        "ldf_program": r"==>>>\s+iso-level laplacian operator",
        "nn_ahm_ijk_t": r"type of time-space variation\s+nn_ahm_ijk_t\s+=\s+-30\b",
        "ln_traadv_OFF": r"No advection on T & S\s+ln_traadv_OFF\s+=\s+T\b",
        "ln_traldf_OFF": r"no explicit diffusion\s+ln_traldf_OFF\s+=\s+T\b",
        "ln_drg_OFF": r"free-slip\s+: Cd = 0\s+ln_drg_OFF\s+=\s+F\b",
        "ln_lin": r"linear\s+drag\s+: Cd = Cd0\s+ln_lin\s+=\s+T\b",
        "ln_drgimp": r"implicit friction\s+ln_drgimp\s+=\s+T\b",
    }
    for label, pattern in required.items():
        require(re.search(pattern, ocean), f"resolved OMT-3 selector moved: {label}")
    return {**row, "resolved_omt3": sorted(required)}


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
                    "admitted OMT-2 deck changed")
            args.render_deck.write_text(render_omt3(args.source.read_text()))
            print(f"STATUS RENDERED_R220_OMT3_DECK {args.render_deck}")
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
            print(f"STATUS RENDERED_R220_OMT3_RUN_DECK {args.render_run_deck}")
            return 0
        require(args.source is not None and args.candidate is not None,
                "deck validation requires source and candidate")
        result = validate_deck(args.source, args.candidate, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, omt2.GateError, protocol.GateError,
            OSError, ValueError) as error:
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
