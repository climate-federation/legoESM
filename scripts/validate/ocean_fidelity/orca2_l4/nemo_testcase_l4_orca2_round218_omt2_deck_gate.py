#!/usr/bin/env python3
"""Render and validate Decision-109 OMT-2 linear bottom drag."""

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
    nemo_testcase_l4_orca2_round209_omt1_deck_gate as omt1,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


SOURCE_SHA256 = "d0cccebd76c9c4631c91e51a1af3a9b552d97f307c7c523020160c883fb5e73f"
CHANGED = {"namdrg.ln_lin": ".true."}
REMOVED = {"namdrg.ln_drg_off": ".true."}
RETAINED_OFF = {
    "namdyn_ldf.ln_dynldf_off": ".true.",
    "namtra_adv.ln_traadv_off": ".true.",
    "namtra_ldf.ln_traldf_off": ".true.",
}
PLANTS = ("none", "extra-delta", "wrong-selector")


class GateError(RuntimeError):
    """The candidate is not the exact OMT-1 to OMT-2 module edge."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normal(value: str) -> str:
    return protocol._normal(value)


def render_omt2(source: str) -> str:
    """Restore exactly the rung-0 namdrg assignment block."""
    text = omt1.omt0.rung0_deck._replace_in_group(
        source, "namdrg.ln_lin", ".true.",
    )
    lines = text.splitlines(keepends=True)
    in_group = False
    removed = 0
    rendered: list[str] = []
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith("&"):
            in_group = stripped.split()[0] == "&namdrg"
        if in_group and re.match(r"(?i)^\s*ln_drg_off\s*=", line):
            removed += 1
            continue
        rendered.append(line)
        if in_group and stripped.startswith("/"):
            in_group = False
    require(removed == 1, "ln_drg_off override was not removed exactly once")
    return "".join(rendered)


def validate_deck(source: Path, candidate: Path, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    require(sha256(source) == SOURCE_SHA256, "admitted OMT-1 deck changed")
    expected = render_omt2(source.read_text())
    actual = candidate.read_text()
    if plant == "extra-delta":
        actual = actual.replace("nn_fsbc     = 2", "nn_fsbc     = 3", 1)
    elif plant == "wrong-selector":
        actual, count = re.subn(
            r"(?mi)^(\s*ln_lin\s*=\s*)\.true\.",
            r"\g<1>.false.", actual, count=1,
        )
        require(count == 1, "linear-drag plant target disappeared")
    require(actual == expected, "candidate is not the exact OMT-2 rendering")

    before = namelist_values(source)
    after = namelist_values(candidate)
    changed = sorted(
        key for key in before if key in after
        and _normal(before[key]) != _normal(after[key])
    )
    removed = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    require(changed == sorted(CHANGED), f"OMT-2 changed assignments moved: {changed}")
    require(removed == sorted(REMOVED), f"OMT-2 removed assignments moved: {removed}")
    require(not added, f"OMT-2 added assignments: {added}")
    for key, wanted in CHANGED.items():
        require(_normal(after[key]) == wanted, f"{key}: resolved deck value moved")
    for key, wanted in RETAINED_OFF.items():
        require(key in after and _normal(after[key]) == wanted,
                f"{key}: retained OMT-1 selector moved")
    return {
        "format": "nemo-testcase-l4-orca2-round218-omt2-deck-v1",
        "status": "PASS_R218_OMT2_DECK",
        "source_sha256": SOURCE_SHA256,
        "candidate_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "changed_assignments": changed,
        "removed_assignments": removed,
        "retained_off_assignments": RETAINED_OFF,
        "resolved_from_namelist_ref": {
            "namdrg.ln_drg_off": ".false.",
            "namdrg.ln_drgimp": ".true.",
            "namdrg_bot.rn_cd0": "1.e-3",
            "namdrg_bot.rn_uc0": "0.4",
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
        "ln_dynldf_OFF": r"no explicit diffusion\s+ln_dynldf_OFF\s+=\s+T\b",
        "ln_traadv_OFF": r"No advection on T & S\s+ln_traadv_OFF\s+=\s+T\b",
        "ln_traldf_OFF": r"no explicit diffusion\s+ln_traldf_OFF\s+=\s+T\b",
        "ln_drg_OFF": r"free-slip\s+: Cd = 0\s+ln_drg_OFF\s+=\s+F\b",
        "ln_lin": r"linear\s+drag\s+: Cd = Cd0\s+ln_lin\s+=\s+T\b",
        "ln_drgimp": r"implicit friction\s+ln_drgimp\s+=\s+T\b",
        "rn_Cd0": r"drag coefficient\s+rn_Cd0\s+=\s+1\.0000000000000000E-003",
        "rn_Uc0": r"characteristic velocity \(linear case\)\s+rn_Uc0\s+=\s+0\.40000000000000002",
    }
    for label, pattern in required.items():
        require(re.search(pattern, ocean), f"resolved OMT-2 selector moved: {label}")
    return {**row, "resolved_omt2": sorted(required)}


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
                    "admitted OMT-1 deck changed")
            args.render_deck.write_text(render_omt2(args.source.read_text()))
            print(f"STATUS RENDERED_R218_OMT2_DECK {args.render_deck}")
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
            print(f"STATUS RENDERED_R218_OMT2_RUN_DECK {args.render_run_deck}")
            return 0
        require(args.source is not None and args.candidate is not None,
                "deck validation requires source and candidate")
        result = validate_deck(args.source, args.candidate, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, omt1.GateError, protocol.GateError, OSError, ValueError) as error:
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
