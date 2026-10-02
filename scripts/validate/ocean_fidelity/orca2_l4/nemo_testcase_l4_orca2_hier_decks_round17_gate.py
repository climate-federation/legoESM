#!/usr/bin/env python3
"""Preflight and admit Decision 83's replacement rung-2 oracle record."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hier_decks_round10_gate as legacy,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hier_decks_round16_gate as upper,
)

HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round17_acquisition"
MANIFEST = ACQUISITION / "rung2_havtb0_manifest.json"
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung2")
CURRENT_RECORD = ROOT / "record"
SUPERSEDED_RECORD = ROOT / "record_havtb1_superseded"
CURRENT_DECK = ROOT / "deck"
SUPERSEDED_DECK = ROOT / "deck_havtb1_superseded"
CURRENT_ADMISSION = ROOT / "rung2_admission.json"
SUPERSEDED_ADMISSION = ROOT / "rung2_admission_havtb1_superseded.json"
UPPER_ADMISSION = upper.CURRENT_ADMISSION

OLD_LINE = upper.OLD_LINE
NEW_LINE = upper.NEW_LINE
OLD_ADMISSION_SHA = "b1553d791db7df6ade5d9f3da06790e789a21b0be2ab4758b88e27f21509003e"
OLD_INVENTORY_SHA = "72b65f0fcb930fb928ad97f3cbc490bd18f66fc6640970e59e0ee61d8c21607b"
OLD_CFG_SHA = legacy.RUNG2_CFG_SHA
OLD_EXEC_SHA = legacy.EXECUTION_CFG_SHA
NEW_CFG_SHA = "b4b8cb87249bf41ef41ea13637273ef097f6632ae4879dca883feebdd9011e38"
NEW_EXEC_SHA = "d691aec39430a952bc4cff73e3af2487fe10a4cf728111fe3ec8fc66609c0416"
UPPER_ADMISSION_SHA = "591bce28aea5626ea009c674271061ae5f0d4c7959df2ef879bd9b36134fb4b0"

PRECHECK_PLANTS = (
    "deck-extra",
    "missing-havtb",
    "retained-coefficient",
    "source-pin",
    "superseded-pin",
    "upper-pin",
)
RECORD_PLANTS = tuple(dict.fromkeys((*legacy.RECORD_PLANTS, "resolved-havtb")))
PLANTS = ("none", *PRECHECK_PLANTS, *RECORD_PLANTS)


class GateError(RuntimeError):
    """A frozen Decision-83 rung-2 predicate failed."""


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


def rung2_cfg_bytes(*, plant: str = "none") -> bytes:
    record, _, _ = _old_paths()
    return _replace_havtb((record / "namelist_cfg.deck").read_bytes(), plant=plant)


def execution_cfg_bytes() -> bytes:
    record, _, _ = _old_paths()
    return _replace_havtb((record / "namelist_cfg").read_bytes())


def preflight(*, plant: str = "none") -> dict[str, object]:
    require(plant in ("none", *PRECHECK_PLANTS), f"invalid preflight plant: {plant}")
    record, deck, admission_path = _old_paths()
    expected_old = OLD_ADMISSION_SHA if plant != "superseded-pin" else "0" * 64
    require(sha256(admission_path) == expected_old, "superseded admission changed")
    require(sha256(record / "SHA256SUMS") == OLD_INVENTORY_SHA, "superseded inventory changed")
    require(sha256(record / "namelist_cfg.deck") == OLD_CFG_SHA, "superseded exact deck changed")
    require(sha256(record / "namelist_cfg") == OLD_EXEC_SHA, "superseded execution deck changed")
    require(sha256(deck / "namelist_cfg") == OLD_CFG_SHA, "superseded staged deck changed")

    expected_upper = UPPER_ADMISSION_SHA if plant != "upper-pin" else "0" * 64
    require(sha256(UPPER_ADMISSION) == expected_upper, "replacement rung-3 admission changed")
    upper_admission = json.loads(UPPER_ADMISSION.read_text())
    require(
        upper_admission.get("status") == "PASS_RUNG3_HAVTB0_RECORD",
        "replacement rung 3 is not admitted",
    )

    manifest = json.loads(MANIFEST.read_text())
    require(manifest.get("rung") == 2, "manifest rung changed")
    require(
        manifest.get("upper_rung_admission_sha256") == UPPER_ADMISSION_SHA,
        "manifest upper admission changed",
    )
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
    for name, expected in legacy.SOURCE_SHA.items():
        actual = sha256(legacy.COMPILED / f"{name}.f90")
        if plant == "source-pin" and name == "ldftra":
            actual = "0" * 64
        require(actual == expected, f"compiled source changed: {name}")
        require(
            manifest["build"].get(f"compiled_{name}_sha256") == expected,
            f"manifest source pin changed: {name}",
        )
    require(sha256(record / "nemo") == legacy.rung3.rung4.BINARY_SHA, "recorder binary changed")

    cfg = rung2_cfg_bytes(plant=plant)
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

    with tempfile.TemporaryDirectory(prefix="orca2-rung2-havtb0-") as temporary:
        root = Path(temporary)
        for name in ("old", "new", "upper"):
            (root / name).mkdir()
        old = legacy._assignment_map((record / "namelist_cfg.deck").read_bytes(), root / "old")
        new = legacy._assignment_map(cfg, root / "new")
        upper_values = legacy._assignment_map(upper.rung3_cfg_bytes(), root / "upper")
    replacement_delta = {
        name: [legacy._token(old[name]), legacy._token(new[name])]
        for name in sorted(set(old) | set(new))
        if old.get(name) != new.get(name)
    }
    require(
        replacement_delta == {"namzdf.nn_havtb": ["1", "0"]},
        f"replacement delta changed: {replacement_delta}",
    )
    boundary_delta = {
        name: [
            legacy._token(upper_values[name]) if name in upper_values else "ABSENT",
            legacy._token(new[name]) if name in new else "ABSENT",
        ]
        for name in sorted(set(upper_values) | set(new))
        if upper_values.get(name) != new.get(name)
    }
    old_admission = json.loads(admission_path.read_text())
    require(
        boundary_delta == old_admission.get("deck_delta_from_rung3"),
        f"rung-3/rung-2 boundary changed: {boundary_delta}",
    )
    old_lines = (record / "namelist_cfg.deck").read_text().splitlines()
    new_lines = cfg.decode().splitlines()
    line_delta = [
        [index + 1, before, after]
        for index, (before, after) in enumerate(zip(old_lines, new_lines, strict=True))
        if before != after
    ]
    require(line_delta == [[419, OLD_LINE, NEW_LINE]], f"physical line delta changed: {line_delta}")
    source = (upper.COMPILED / "zdfphy.f90").read_text()
    require(
        "avtb_2d(:,:) = 1._wp" in source and "IF( nn_havtb == 1 ) THEN" in source,
        "compiled background-shape branch changed",
    )
    return {
        "status": "PREFLIGHT_PASS_RUNG2_HAVTB0",
        "claim_label": "independent",
        "rung": 2,
        "decision": 83,
        "replacement_delta": replacement_delta,
        "line_delta": line_delta,
        "rung3_boundary_delta": boundary_delta,
        "superseded_record": str(record),
        "binary_sha256": legacy.rung3.rung4.BINARY_SHA,
    }


def stage_deck(root: Path) -> None:
    record, _, _ = _old_paths()
    expected = {
        "namelist_cfg": rung2_cfg_bytes(),
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
    own_plant = plant if plant in set(legacy.RECORD_PLANTS) else "none"
    upper_plant = plant if plant in set(upper.RECORD_PLANTS) else "none"
    own = legacy.validate_gm_mle_resolved(root, plant=own_plant)
    inherited = upper.validate_resolved(root, plant=upper_plant)
    return {"status": "PASS_RUNG2_HAVTB0_RESOLVED", "upper_rung": inherited, "gm_mle": own}


def validate_record(root: Path, *, expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant: {plant}")
    report = preflight(plant=plant if plant in PRECHECK_PLANTS else "none")
    old_record, _, _ = _old_paths()
    require(
        (root / "producer_commit.txt").read_text().strip() == expect_commit,
        "producer commit differs",
    )
    require(sha256(root / "nemo") == legacy.rung3.rung4.BINARY_SHA, "record binary differs")
    require((root / "namelist_cfg").read_bytes() == execution_cfg_bytes(), "execution deck differs")
    require((root / "namelist_cfg.deck").read_bytes() == rung2_cfg_bytes(), "exact deck differs")
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
    for name, expected in legacy.SOURCE_SHA.items():
        require(
            sha256(root / f"compiled_{name}.f90") == expected,
            f"record compiled source changed: {name}",
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
    inherited_plant = plant if plant in set(legacy.rung3.rung4.rung5_deck.PLANTS) else "none"
    frames = legacy.rung3.rung4.rung5_record.validate_frames(root, plant=frame_plant)
    terminal = legacy.rung3.rung4.rung5_deck.rung6.rung7.rung8.rung9.validate_terminal(
        root, plant=inherited_plant
    )
    month = legacy.rung3.rung4.rung5_deck.rung6.rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    zero_flux = legacy.rung3._validate_zero_flux(root, plant=plant)
    inventory = legacy.rung3.rung4.rung5_deck.rung6.rung10.validate_sha_inventory(
        root, plant=inherited_plant
    )
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung2-havtb0-record-v1",
        "status": "PASS_RUNG2_HAVTB0_RECORD",
        "claim_label": "independent",
        "rung": 2,
        "decision": 83,
        "producer_commit": expect_commit,
        "deck_delta_from_superseded": {"namzdf.nn_havtb": [1, 0]},
        "deck_delta_from_rung3": report["rung3_boundary_delta"],
        "frames": frames,
        "zero_flux": zero_flux,
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
