#!/usr/bin/env python3
"""Preflight and admit Decision 83's replacement rung-1 oracle record."""

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
    nemo_testcase_l4_orca2_hier_decks_round11_gate as legacy,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hier_decks_round17_gate as upper,
)

HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round18_acquisition"
MANIFEST = ACQUISITION / "rung1_havtb0_manifest.json"
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung1")
CURRENT_RECORD = ROOT / "record"
SUPERSEDED_RECORD = ROOT / "record_havtb1_superseded"
CURRENT_DECK = ROOT / "deck"
SUPERSEDED_DECK = ROOT / "deck_havtb1_superseded"
CURRENT_ADMISSION = ROOT / "rung1_admission.json"
SUPERSEDED_ADMISSION = ROOT / "rung1_admission_havtb1_superseded.json"
UPPER_ADMISSION = upper.CURRENT_ADMISSION
MAIN_RUNG0_CFG = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round96/"
    "acquisition/orca2_rung0_spgts_ranked_10step_np2/namelist_cfg"
)

OLD_LINE = upper.OLD_LINE
NEW_LINE = upper.NEW_LINE
OLD_ADMISSION_SHA = "4a89741a11b8379cd7e5e27768a314ac2936b06c1b19ec93faf895df9e9a28fa"
OLD_INVENTORY_SHA = "6aa478b735436077d2bdb514bba2964b6f2e0536ac34d69493e9647a9cf87df5"
OLD_CFG_SHA = legacy.RUNG1_CFG_SHA
OLD_EXEC_SHA = legacy.EXECUTION_CFG_SHA
NEW_CFG_SHA = "6849907f5237083bddacc32623d85228fa4c3608f44d2bf6564cf4b2f507d961"
NEW_EXEC_SHA = "f74f089cc65195225ff560fb1f69c726de08d6030152232f6d22ec0eac62342b"
UPPER_ADMISSION_SHA = "90ba18953dac0082eceeb1e713d38e4a6f28f3cbbed0e4cf2f356e0e415dd04f"
MAIN_RUNG0_CFG_SHA = "5192355842d9233d8356ab87b4ff8b65eac539e66dc135f77a07e26451d360e8"
ZDFPHY_SHA = "903e0d1220a2f9b38cdd55b6eece5dbbfed992f12f497155cca8633702e7451e"

PRECHECK_PLANTS = (
    "deck-extra",
    "missing-havtb",
    "retained-coefficient",
    "source-pin",
    "superseded-pin",
    "upper-pin",
    "main-rung0-pin",
)
RECORD_PLANTS = tuple(
    dict.fromkeys((*legacy.RECORD_PLANTS, *upper.RECORD_PLANTS, "resolved-havtb"))
)
PLANTS = ("none", *PRECHECK_PLANTS, *RECORD_PLANTS)


class GateError(RuntimeError):
    """A frozen Decision-83 rung-1 predicate failed."""


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


def rung1_cfg_bytes(*, plant: str = "none") -> bytes:
    record, _, _ = _old_paths()
    return _replace_havtb((record / "namelist_cfg.deck").read_bytes(), plant=plant)


def execution_cfg_bytes() -> bytes:
    record, _, _ = _old_paths()
    return _replace_havtb((record / "namelist_cfg").read_bytes())


def _main_rung0_diff(lower: dict[str, str]) -> dict[str, list[str | None]]:
    main = legacy.namelist_values(MAIN_RUNG0_CFG)
    return legacy._main_rung0_diff(lower, main)


def _validate_main_boundary(diff: dict[str, list[str | None]]) -> None:
    formerly_semantic = {
        "namagrif.ln_spc_dyn",
        "namtra_qsr.nn_chldta",
        "namsbc_ssr.ln_sssr_bnd",
        "namzdf.ln_zdfgls",
        "namzdf.ln_zdfmfc",
        "namzdf.ln_zdfnpc",
        "namzdf.ln_zdfosm",
        "namzdf.ln_zdfric",
        "namzdf.ln_zdfswm",
    }
    # Decision-83 harmonization must remove every formerly semantic row above.
    require(
        not (formerly_semantic & set(diff)),
        f"main rung-0 inert/background harmonization is incomplete: {diff}",
    )
    expected_remaining = {
        "namrun.ln_rst_list",
        "namrun.nn_itend",
        "namrun.nn_stock",
        "namrun.nn_stocklist",
        "namsbc_flx.sn_emp",
        "namsbc_flx.sn_qsr",
        "namsbc_flx.sn_qtot",
        "namsbc_flx.sn_utau",
        "namsbc_flx.sn_vtau",
        "namtra_dmp.ln_tradmp",
        "namtsd.ln_tsd_dmp",
    }
    # The four restart rows and five zero-file names are protocol-only.
    allowed = expected_remaining
    require(set(diff) == allowed, f"rung-1/rung-0 boundary changed: {diff}")
    require(diff["namtra_dmp.ln_tradmp"] == [".true.", ".false."], "tracer damping changed")
    require(diff["namtsd.ln_tsd_dmp"] == [".true.", ".false."], "T/S damping changed")
    for name in ("sn_emp", "sn_qsr", "sn_qtot", "sn_utau", "sn_vtau"):
        left, right = diff[f"namsbc_flx.{name}"]
        require(
            left is not None
            and right is not None
            and left.replace("'rung3_zero_flux'", "'rung0_zero_flux'") == right,
            f"zero-file operand changed beyond its filename: {name}",
        )


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
    require(sha256(UPPER_ADMISSION) == expected_upper, "replacement rung-2 admission changed")
    upper_admission = json.loads(UPPER_ADMISSION.read_text())
    require(
        upper_admission.get("status") == "PASS_RUNG2_HAVTB0_RECORD",
        "replacement rung 2 is not admitted",
    )

    main_sha = sha256(MAIN_RUNG0_CFG)
    if plant == "main-rung0-pin":
        main_sha = "0" * 64
    require(main_sha == MAIN_RUNG0_CFG_SHA, "main-lane rung-0 deck changed")

    manifest = json.loads(MANIFEST.read_text())
    require(manifest.get("rung") == 1, "manifest rung changed")
    require(
        manifest.get("upper_rung_admission_sha256") == UPPER_ADMISSION_SHA,
        "manifest upper admission changed",
    )
    require(
        manifest.get("main_rung0_deck_sha256") == MAIN_RUNG0_CFG_SHA,
        "manifest main-rung-0 pin changed",
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
        if plant == "source-pin" and name == "trabbl":
            actual = "0" * 64
        require(actual == expected, f"compiled source changed: {name}")
        require(
            manifest["build"].get(f"compiled_{name}_sha256") == expected,
            f"manifest source pin changed: {name}",
        )
    require(
        sha256(record / "nemo") == legacy.rung2.rung3.rung4.BINARY_SHA, "recorder binary changed"
    )

    cfg = rung1_cfg_bytes(plant=plant)
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
    zdfphy = upper.upper.COMPILED / "zdfphy.f90"
    require(sha256(zdfphy) == ZDFPHY_SHA, "compiled background-shape source changed")
    require(
        manifest["build"].get("compiled_zdfphy_sha256") == ZDFPHY_SHA,
        "manifest background-shape source pin changed",
    )

    with tempfile.TemporaryDirectory(prefix="orca2-rung1-havtb0-") as temporary:
        root = Path(temporary)
        for name in ("old", "new", "upper"):
            (root / name).mkdir()
        old = legacy._assignment_map((record / "namelist_cfg.deck").read_bytes(), root / "old")
        new = legacy._assignment_map(cfg, root / "new")
        upper_values = legacy._assignment_map(upper.rung2_cfg_bytes(), root / "upper")
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
    expected_boundary = {
        name: [old, new_value] for name, (old, new_value) in legacy.CHANGES.items()
    }
    require(
        boundary_delta == expected_boundary, f"rung-2/rung-1 boundary changed: {boundary_delta}"
    )
    old_lines = (record / "namelist_cfg.deck").read_text().splitlines()
    new_lines = cfg.decode().splitlines()
    line_delta = [
        [index + 1, before, after]
        for index, (before, after) in enumerate(zip(old_lines, new_lines, strict=True))
        if before != after
    ]
    require(line_delta == [[419, OLD_LINE, NEW_LINE]], f"physical line delta changed: {line_delta}")
    main_diff = _main_rung0_diff(new)
    _validate_main_boundary(main_diff)
    source = zdfphy.read_text()
    require(
        "avtb_2d(:,:) = 1._wp" in source and "IF( nn_havtb == 1 ) THEN" in source,
        "compiled background-shape branch changed",
    )
    return {
        "status": "PREFLIGHT_PASS_RUNG1_HAVTB0",
        "claim_label": "independent",
        "rung": 1,
        "decision": 83,
        "replacement_delta": replacement_delta,
        "line_delta": line_delta,
        "rung2_boundary_delta": boundary_delta,
        "main_rung0_semantic_differences": main_diff,
        "superseded_record": str(record),
        "binary_sha256": legacy.rung2.rung3.rung4.BINARY_SHA,
    }


def stage_deck(root: Path) -> None:
    record, _, _ = _old_paths()
    expected = {
        "namelist_cfg": rung1_cfg_bytes(),
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
    own_plant = plant if plant in set(legacy.OWN_RECORD_PLANTS) else "none"
    upper_plant = plant if plant in set(upper.RECORD_PLANTS) else "none"
    own = legacy.validate_bbl_bbc_resolved(root, plant=own_plant)
    inherited = upper.validate_resolved(root, plant=upper_plant)
    return {"status": "PASS_RUNG1_HAVTB0_RESOLVED", "upper_rung": inherited, "bbl_bbc": own}


def validate_record(root: Path, *, expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant: {plant}")
    report = preflight(plant=plant if plant in PRECHECK_PLANTS else "none")
    old_record, _, _ = _old_paths()
    require(
        (root / "producer_commit.txt").read_text().strip() == expect_commit,
        "producer commit differs",
    )
    require(sha256(root / "nemo") == legacy.rung2.rung3.rung4.BINARY_SHA, "record binary differs")
    require((root / "namelist_cfg").read_bytes() == execution_cfg_bytes(), "execution deck differs")
    require((root / "namelist_cfg.deck").read_bytes() == rung1_cfg_bytes(), "exact deck differs")
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
    inherited_plant = plant if plant in set(legacy.rung2.rung3.rung4.rung5_deck.PLANTS) else "none"
    frames = legacy.rung2.rung3.rung4.rung5_record.validate_frames(root, plant=frame_plant)
    terminal = legacy.rung2.rung3.rung4.rung5_deck.rung6.rung7.rung8.rung9.validate_terminal(
        root, plant=inherited_plant
    )
    month = legacy.rung2.rung3.rung4.rung5_deck.rung6.rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    zero_flux = legacy.rung2.rung3._validate_zero_flux(root, plant=plant)
    inventory = legacy.rung2.rung3.rung4.rung5_deck.rung6.rung10.validate_sha_inventory(
        root, plant=inherited_plant
    )
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung1-havtb0-record-v1",
        "status": "PASS_RUNG1_HAVTB0_RECORD",
        "claim_label": "independent",
        "rung": 1,
        "decision": 83,
        "producer_commit": expect_commit,
        "deck_delta_from_superseded": {"namzdf.nn_havtb": [1, 0]},
        "deck_delta_from_rung2": report["rung2_boundary_delta"],
        "main_rung0_semantic_differences": report["main_rung0_semantic_differences"],
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
