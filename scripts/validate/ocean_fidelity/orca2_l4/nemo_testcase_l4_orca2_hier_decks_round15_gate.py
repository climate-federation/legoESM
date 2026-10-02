#!/usr/bin/env python3
"""Preflight and admit Decision 83's replacement rung-4 oracle record."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hier_decks_round8_gate as legacy,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hier_decks_round14_gate as upper,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (  # noqa: E402
    namelist_values,
)

HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round15_acquisition"
MANIFEST = ACQUISITION / "rung4_havtb0_manifest.json"
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung4")
CURRENT_RECORD = ROOT / "record"
SUPERSEDED_RECORD = ROOT / "record_havtb1_superseded"
CURRENT_DECK = ROOT / "deck"
SUPERSEDED_DECK = ROOT / "deck_havtb1_superseded"
CURRENT_ADMISSION = ROOT / "rung4_admission.json"
SUPERSEDED_ADMISSION = ROOT / "rung4_admission_havtb1_superseded.json"
UPPER_ADMISSION = upper.CURRENT_ADMISSION
COMPILED = legacy.COMPILED

OLD_LINE = "   nn_havtb    =    1         !  horizontal shape for avtb (=1) or not (=0)"
NEW_LINE = "   nn_havtb    =    0         !  horizontal shape for avtb (=1) or not (=0)"
OLD_ADMISSION_SHA = "439bee8c4404cf86f6659fa5ee08a2e077c1ff92c87425ad321b0a590ac67bc7"
OLD_INVENTORY_SHA = "91f85604de305994b358d0197340e6de91c335972fb524b43a4807daad8f96fb"
OLD_CFG_SHA = legacy.RUNG4_CFG_SHA
OLD_EXEC_SHA = legacy.EXECUTION_CFG_SHA
NEW_CFG_SHA = "f764a5781a8babedd9350872bebc6fe1df41b5a1d78de7c5eadddb0ecf29bdfa"
NEW_EXEC_SHA = "dc810d645e14f87cc494db4442346004c3180f36210e4e6816d972206b6c1c34"
UPPER_ADMISSION_SHA = "0c6543e08bef727b5a01096406f67fd7a41f33ca5f5cedbd36a0ebaf0e97e98a"

PRECHECK_PLANTS = (
    "deck-extra",
    "missing-havtb",
    "retained-coefficient",
    "source-pin",
    "superseded-pin",
    "upper-pin",
)
RECORD_PLANTS = legacy.RECORD_PLANTS + ("resolved-havtb",)
PLANTS = ("none", *PRECHECK_PLANTS, *RECORD_PLANTS)


class GateError(RuntimeError):
    """A frozen Decision-83 rung-4 predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _old_paths() -> tuple[Path, Path, Path]:
    record = SUPERSEDED_RECORD if SUPERSEDED_RECORD.exists() else CURRENT_RECORD
    deck = SUPERSEDED_DECK if SUPERSEDED_DECK.exists() else CURRENT_DECK
    admission = SUPERSEDED_ADMISSION if SUPERSEDED_ADMISSION.exists() else CURRENT_ADMISSION
    return record, deck, admission


def _replace_havtb(payload: bytes, *, plant: str = "none") -> bytes:
    text = payload.decode()
    require(text.count(OLD_LINE) == 1, "superseded nn_havtb line changed")
    if plant != "missing-havtb":
        text = text.replace(OLD_LINE, NEW_LINE)
    if plant == "deck-extra":
        target = "   rn_rfact    =   1.e0    !  multiplicative factor for runoff"
        require(text.count(target) == 1, "deck-extra target changed")
        text = text.replace(target, target.replace("1.e0", "2.e0"))
    if plant == "retained-coefficient":
        target = "   rn_avm0     =   1.2e-4     !  vertical eddy viscosity   [m2/s]"
        require(text.count(target) == 1, "retained-coefficient target changed")
        text = text.replace(target, target.replace("1.2e-4", "2.2e-4"))
    return text.encode()


def rung4_cfg_bytes(*, plant: str = "none") -> bytes:
    record, _, _ = _old_paths()
    return _replace_havtb((record / "namelist_cfg.deck").read_bytes(), plant=plant)


def execution_cfg_bytes() -> bytes:
    record, _, _ = _old_paths()
    return _replace_havtb((record / "namelist_cfg").read_bytes())


def _assignment_map(payload: bytes, root: Path) -> dict[str, str]:
    target = root / "namelist_cfg"
    target.write_bytes(payload)
    return namelist_values(target)


def preflight(*, plant: str = "none") -> dict[str, object]:
    require(plant in ("none", *PRECHECK_PLANTS), f"invalid preflight plant: {plant}")
    legacy.preflight()
    record, deck, admission_path = _old_paths()
    expected_old_admission = OLD_ADMISSION_SHA
    if plant == "superseded-pin":
        expected_old_admission = "0" * 64
    require(sha256(admission_path) == expected_old_admission, "superseded admission changed")
    require(sha256(record / "SHA256SUMS") == OLD_INVENTORY_SHA, "superseded inventory changed")
    require(sha256(record / "namelist_cfg.deck") == OLD_CFG_SHA, "superseded exact deck changed")
    require(sha256(record / "namelist_cfg") == OLD_EXEC_SHA, "superseded execution deck changed")
    require(sha256(deck / "namelist_cfg") == OLD_CFG_SHA, "superseded staged deck changed")

    expected_upper = UPPER_ADMISSION_SHA if plant != "upper-pin" else "0" * 64
    require(sha256(UPPER_ADMISSION) == expected_upper, "replacement rung-5 admission changed")
    upper_admission = json.loads(UPPER_ADMISSION.read_text())
    require(upper_admission.get("status") == "PASS_RUNG5_HAVTB0_RECORD", "rung 5 is not admitted")

    manifest = json.loads(MANIFEST.read_text())
    require(manifest.get("rung") == 4, "manifest rung changed")
    require(
        manifest.get("run")
        == {
            "mpi_ranks": 2,
            "first_step": 1,
            "last_step": 240,
            "from_rest": True,
            "initial_state_output": 1,
        },
        "run protocol changed",
    )
    expected_binary = legacy.BINARY_SHA if plant != "source-pin" else "0" * 64
    require(manifest["build"]["binary_sha256"] == expected_binary, "manifest build pin changed")
    require(sha256(record / "nemo") == legacy.BINARY_SHA, "recorder binary changed")

    cfg = rung4_cfg_bytes(plant=plant)
    require(hashlib.sha256(cfg).hexdigest() == NEW_CFG_SHA, "replacement exact deck changed")
    require(
        hashlib.sha256(execution_cfg_bytes()).hexdigest() == NEW_EXEC_SHA,
        "replacement execution deck changed",
    )
    require(
        manifest["namelists"]["namelist_cfg_sha256"] == NEW_CFG_SHA, "manifest deck pin changed"
    )
    require(
        manifest["namelists"]["execution_namelist_cfg_sha256"] == NEW_EXEC_SHA,
        "manifest execution pin changed",
    )

    with tempfile.TemporaryDirectory(prefix="orca2-rung4-havtb0-") as temporary:
        root = Path(temporary)
        (root / "old").mkdir()
        (root / "new").mkdir()
        (root / "upper").mkdir()
        old_values = _assignment_map((record / "namelist_cfg.deck").read_bytes(), root / "old")
        new_values = _assignment_map(cfg, root / "new")
        upper_values = _assignment_map(upper.rung5_cfg_bytes(), root / "upper")
    replacement_delta = {
        name: [
            legacy.rung5_deck.rung6.rung10._token(old_values[name]),
            legacy.rung5_deck.rung6.rung10._token(new_values[name]),
        ]
        for name in sorted(set(old_values) | set(new_values))
        if old_values.get(name) != new_values.get(name)
    }
    require(
        replacement_delta == {"namzdf.nn_havtb": ["1", "0"]},
        f"replacement delta changed: {replacement_delta}",
    )
    boundary_delta = {
        name: [
            legacy.rung5_deck.rung6.rung10._token(upper_values[name]),
            legacy.rung5_deck.rung6.rung10._token(new_values[name]),
        ]
        for name in sorted(set(upper_values) | set(new_values))
        if upper_values.get(name) != new_values.get(name)
    }
    require(
        boundary_delta
        == {
            "namsbc.ln_traqsr": [".true.", ".false."],
            "namtra_qsr.ln_qsr_rgb": [".true.", ".false."],
            "namtra_qsr.nn_chldta": ["1", "0"],
        },
        f"rung-5/rung-4 boundary is not shortwave-only: {boundary_delta}",
    )
    old_lines = (record / "namelist_cfg.deck").read_text().splitlines()
    new_lines = cfg.decode().splitlines()
    line_delta = [
        [index + 1, before, after]
        for index, (before, after) in enumerate(zip(old_lines, new_lines, strict=True))
        if before != after
    ]
    require(line_delta == [[403, OLD_LINE, NEW_LINE]], f"physical line delta changed: {line_delta}")
    source = (COMPILED / "zdfphy.f90").read_text()
    require(
        "avtb_2d(:,:) = 1._wp" in source and "IF( nn_havtb == 1 ) THEN" in source,
        "compiled background-shape branch changed",
    )
    return {
        "status": "PREFLIGHT_PASS_RUNG4_HAVTB0",
        "claim_label": "independent",
        "rung": 4,
        "decision": 83,
        "replacement_delta": replacement_delta,
        "line_delta": line_delta,
        "rung5_boundary_delta": boundary_delta,
        "superseded_record": str(record),
        "binary_sha256": legacy.BINARY_SHA,
    }


def stage_deck(root: Path) -> None:
    record, _, _ = _old_paths()
    expected = {
        "namelist_cfg": rung4_cfg_bytes(),
        "namelist_ice_cfg": (record / "namelist_ice_cfg").read_bytes(),
        "manifest.json": MANIFEST.read_bytes(),
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in expected.items():
        target = root / name
        if target.exists():
            require(
                target.is_file() and target.read_bytes() == payload,
                f"existing deck differs: {target}",
            )
        else:
            target.write_bytes(payload)
    rows = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n" for name, payload in expected.items()
    )
    ledger = root / "SHA256SUMS"
    if ledger.exists():
        require(ledger.read_text() == rows, "deck SHA256SUMS changed")
    else:
        ledger.write_text(rows)


def validate_resolved(root: Path, *, plant: str = "none") -> dict[str, object]:
    inherited_plant = plant if plant in set(upper.PLANTS) else "none"
    inherited = upper.validate_resolved(root, plant=inherited_plant)
    ocean = (root / "ocean.output").read_text(errors="strict")
    checks = {
        "ln_traqsr_false": re.search(
            r"Light penetration in temperature Eq\.\s+ln_traqsr\s*=\s*F\b", ocean
        )
        is not None,
        "shortwave_initializer_absent": "tra_qsr_init : penetration" not in ocean,
        "rgb_print_absent": "RGB (Red-Green-Blue) light penetration" not in ocean,
    }
    if plant == "shortwave-consequence":
        checks["ln_traqsr_false"] = False
    if plant == "resolved-havtb":
        inherited["nn_havtb_uniform"] = False
    require(
        all(checks.values()) and inherited.get("nn_havtb_uniform") is True,
        f"resolved rung-4 checks failed: {checks}, upper={inherited}",
    )
    return {"status": "PASS_RUNG4_HAVTB0_RESOLVED", "upper_rung": inherited, **checks}


def validate_record(root: Path, *, expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant: {plant}")
    preflight(plant=plant if plant in PRECHECK_PLANTS else "none")
    old_record, _, _ = _old_paths()
    require(
        (root / "producer_commit.txt").read_text().strip() == expect_commit,
        "producer commit differs",
    )
    require(sha256(root / "nemo") == legacy.BINARY_SHA, "record binary differs")
    require((root / "namelist_cfg").read_bytes() == execution_cfg_bytes(), "execution deck differs")
    require((root / "namelist_cfg.deck").read_bytes() == rung4_cfg_bytes(), "exact deck differs")
    require(
        json.loads((root / "hierarchy_manifest.json").read_text())
        == json.loads(MANIFEST.read_text()),
        "record manifest differs",
    )
    require(
        (root / "recorder_repair_manifest.json").read_bytes()
        == (old_record / "recorder_repair_manifest.json").read_bytes(),
        "repair manifest differs",
    )
    frame_plant = (
        plant
        if plant
        in {
            "field-name",
            "truncated",
            "frame-nonfinite",
            "absent-as-zero",
            "owner-on",
            "missing-frame",
        }
        else "none"
    )
    inherited_plant = (
        plant if plant in {"terminal-nonfinite", "terminal-step", "sha-inventory"} else "none"
    )
    frames = legacy.rung5_record.validate_frames(root, plant=frame_plant)
    terminal = legacy.rung5_deck.rung6.rung7.rung8.rung9.validate_terminal(
        root, plant=inherited_plant
    )
    month = legacy.rung5_deck.rung6.rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    inventory = legacy.rung5_deck.rung6.rung10.validate_sha_inventory(root, plant=inherited_plant)
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung4-havtb0-record-v1",
        "status": "PASS_RUNG4_HAVTB0_RECORD",
        "claim_label": "independent",
        "rung": 4,
        "decision": 83,
        "producer_commit": expect_commit,
        "deck_delta_from_superseded": {"namzdf.nn_havtb": [1, 0]},
        "frames": frames,
        "terminal": terminal,
        "month_products": month,
        "resolved": resolved,
        "sha_inventory": inventory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stage-deck", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight_only:
            report = preflight(plant=args.plant)
            if args.stage_deck:
                stage_deck(args.stage_deck)
        else:
            require(
                args.record is not None and args.expect_commit,
                "record and expected commit are required",
            )
            report = validate_record(
                args.record, expect_commit=args.expect_commit, plant=args.plant
            )
        report["worktree"] = worktree_stamp()
    except (RuntimeError, OSError, KeyError, TypeError, UnicodeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
