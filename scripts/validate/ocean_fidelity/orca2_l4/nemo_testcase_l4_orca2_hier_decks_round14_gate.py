#!/usr/bin/env python3
"""Preflight and admit Decision 83's replacement rung-5 oracle record."""

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
    nemo_testcase_l4_orca2_hier_decks_round7_gate as legacy,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hier_decks_round13_gate as rung6,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (  # noqa: E402
    namelist_values,
)

HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round14_acquisition"
MANIFEST = ACQUISITION / "rung5_havtb0_manifest.json"
RUNG5_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung5")
CURRENT_RECORD = RUNG5_ROOT / "record"
OLD_RECORD = RUNG5_ROOT / "record_absent_v2"
SUPERSEDED_RECORD = RUNG5_ROOT / "record_havtb1_superseded"
OLD_DECK = RUNG5_ROOT / "deck"
SUPERSEDED_DECK = RUNG5_ROOT / "deck_havtb1_superseded"
OLD_ADMISSION = RUNG5_ROOT / "rung5_round8_admission.json"
SUPERSEDED_ADMISSION = RUNG5_ROOT / "rung5_admission_havtb1_superseded.json"
CURRENT_ADMISSION = RUNG5_ROOT / "rung5_admission.json"
COMPILED = rung6.COMPILED

OLD_CFG_SHA = "f537e3d29a6e2472f89cc9a8d23ec70d18756e3ad112088e7256804698bc1d59"
OLD_EXEC_SHA = "53b45d1c98af7554e254e25e5b07acbab56cb09ed22896cf090415b371f293b8"
NEW_CFG_SHA = "9bb44379e04861da2c22d76e07afc2ac2a8a1ce82b3dcc262583b841d2e71ead"
NEW_EXEC_SHA = "aa1c49f2c46181674b68ae3eb3aabd71b5590d973d785335cfdb1236869e7e70"
OLD_ADMISSION_SHA = "9f5de467d92d2b2b7514f8907604bf008e61606cbf69ab16a61e94e0b8289b70"
OLD_INVENTORY_SHA = "879b8181feff012017ef217ac3c3b93056c2d54135a7d42a5566044f19aa8c7a"
OLD_MANIFEST_SHA = "09b500303d2c35b7ee221c8ba55284c336ae28df2aa37824e331ed1498f8f8e3"
REPAIR_MANIFEST_SHA = "089fa5ee0035c8189f1bd9892228deb10764b74a091c068871c843e7984473e1"
UPPER_ADMISSION_SHA = "0cc13243c3fb100af19e048e4890b07138081d4d690477a44b828e8580cd34f2"
BINARY_SHA = "cee66aec4a9a9c2b85e755af0250c9b27cccef6ed7333b454a742af0692bc9ec"
WRITER_SHA = "81304be81526980fb9f80549fdfd02c20df293b7a1a0725e94ef9b7e25c88c72"
STPRK3_SHA = "b9da36788bb5a3d8ccff03cae7c1123d1b3510c39a0cd86263fcc293da526422"

PRECHECK_PLANTS = (
    "deck-extra",
    "missing-havtb",
    "retained-coefficient",
    "source-pin",
    "superseded-pin",
    "upper-pin",
)
RECORD_PLANTS = (
    "field-name",
    "truncated",
    "frame-nonfinite",
    "absent-as-zero",
    "owner-on",
    "missing-frame",
    "terminal-nonfinite",
    "terminal-step",
    "ice-sentinel-read",
    "tke-sentinel-read",
    "resolved-consequence",
    "resolved-havtb",
    "runoff-group-unread",
    "active-runoff-print",
    "sha-inventory",
)
PLANTS = ("none", *PRECHECK_PLANTS, *RECORD_PLANTS)


class GateError(RuntimeError):
    """A frozen Decision-83 rung-5 predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _old_paths() -> tuple[Path, Path, Path]:
    record = SUPERSEDED_RECORD if SUPERSEDED_RECORD.exists() else OLD_RECORD
    deck = SUPERSEDED_DECK if SUPERSEDED_DECK.exists() else OLD_DECK
    admission = SUPERSEDED_ADMISSION if SUPERSEDED_ADMISSION.exists() else OLD_ADMISSION
    return record, deck, admission


def _replace_havtb(payload: bytes, *, plant: str = "none") -> bytes:
    text = payload.decode()
    require(text.count(rung6.OLD_LINE) == 1, "superseded nn_havtb line changed")
    if plant != "missing-havtb":
        text = text.replace(rung6.OLD_LINE, rung6.NEW_LINE)
    if plant == "deck-extra":
        target = "   rn_rfact    =   1.e0    !  multiplicative factor for runoff"
        require(text.count(target) == 1, "deck-extra target changed")
        text = text.replace(target, target.replace("1.e0", "2.e0"))
    if plant == "retained-coefficient":
        target = (
            "   rn_avt0     =   1.2e-5     !  vertical eddy diffusivity [m2/s]"
            "       (background Kz if ln_zdfcst=F)"
        )
        require(text.count(target) == 1, "retained-coefficient target changed")
        text = text.replace(target, target.replace("1.2e-5", "2.2e-5"))
    return text.encode()


def rung5_cfg_bytes(*, plant: str = "none") -> bytes:
    return _replace_havtb(legacy.rung5.rung5_cfg_bytes(), plant=plant)


def execution_cfg_bytes() -> bytes:
    return _replace_havtb(legacy.rung5.execution_cfg_bytes())


def _assignment_map(payload: bytes, root: Path) -> dict[str, str]:
    target = root / "namelist_cfg"
    target.write_bytes(payload)
    return namelist_values(target)


def preflight(*, plant: str = "none") -> dict[str, object]:
    require(plant in ("none", *PRECHECK_PLANTS), f"invalid preflight plant: {plant}")
    rung6.preflight()
    record, deck, admission_path = _old_paths()
    require(record.is_dir() and deck.is_dir(), "superseded rung-5 evidence is absent")
    require(admission_path.is_file(), "superseded rung-5 admission is absent")

    admission_sha = sha256(admission_path)
    inventory_sha = sha256(record / "SHA256SUMS")
    if plant == "superseded-pin":
        inventory_sha = "0" * 64
    require(admission_sha == OLD_ADMISSION_SHA, "superseded admission changed")
    require(inventory_sha == OLD_INVENTORY_SHA, "superseded inventory changed")
    admission = json.loads(admission_path.read_text())
    require(admission.get("status") == "PASS_RUNG5_ABSENT_RECORD", "old record was not admitted")
    require(admission.get("frames", {}).get("frames") == 480, "old frame count changed")
    require(sha256(record / "namelist_cfg.deck") == OLD_CFG_SHA, "old exact deck changed")
    require(sha256(record / "namelist_cfg") == OLD_EXEC_SHA, "old execution deck changed")
    require(sha256(record / "hierarchy_manifest.json") == OLD_MANIFEST_SHA, "old manifest changed")
    require(
        sha256(record / "recorder_repair_manifest.json") == REPAIR_MANIFEST_SHA,
        "recorder repair manifest changed",
    )
    require(sha256(deck / "namelist_cfg") == OLD_CFG_SHA, "old deck directory changed")

    upper_sha = sha256(rung6.CURRENT_ADMISSION)
    if plant == "upper-pin":
        upper_sha = "0" * 64
    require(upper_sha == UPPER_ADMISSION_SHA, "replacement rung-6 admission changed")
    upper_admission = json.loads(rung6.CURRENT_ADMISSION.read_text())
    require(
        upper_admission.get("status") == "PASS_RUNG6_HAVTB0_RECORD",
        "replacement rung-6 is not admitted",
    )

    manifest = json.loads(MANIFEST.read_text())
    require(manifest.get("rung") == 5 and manifest.get("decision") == 83, "manifest scope changed")
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
    require(
        manifest["superseded"]["admission_sha256"] == OLD_ADMISSION_SHA,
        "manifest old admission pin changed",
    )
    require(
        manifest["superseded"]["record_inventory_sha256"] == OLD_INVENTORY_SHA,
        "manifest old inventory pin changed",
    )
    require(manifest["build"]["binary_sha256"] == BINARY_SHA, "manifest binary pin changed")
    require(
        manifest["build"]["compiled_writer_sha256"] == WRITER_SHA, "manifest writer pin changed"
    )
    require(manifest["build"]["compiled_stprk3_sha256"] == STPRK3_SHA, "manifest step pin changed")
    require(
        manifest["build"]["recorder_repair_manifest_sha256"] == REPAIR_MANIFEST_SHA,
        "manifest repair pin changed",
    )

    cfg = rung5_cfg_bytes(plant=plant)
    require(hashlib.sha256(cfg).hexdigest() == NEW_CFG_SHA, "replacement deck changed")
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

    zdfphy_sha = sha256(COMPILED / "zdfphy.f90")
    if plant == "source-pin":
        zdfphy_sha = "0" * 64
    require(zdfphy_sha == rung6.ZDFPHY_SHA, "compiled zdfphy changed")
    require(sha256(record / "nemo") == BINARY_SHA, "repaired recorder binary changed")
    require(
        sha256(record / "compiled_l4_r69_surface.f90") == WRITER_SHA, "compiled recorder changed"
    )
    require(sha256(record / "compiled_stprk3.f90") == STPRK3_SHA, "compiled step program changed")

    with tempfile.TemporaryDirectory(prefix="orca2-rung5-havtb0-") as temporary:
        root = Path(temporary)
        for name in ("old", "new", "upper"):
            (root / name).mkdir()
        old = _assignment_map(legacy.rung5.rung5_cfg_bytes(), root / "old")
        new = _assignment_map(cfg, root / "new")
        upper = _assignment_map(rung6.rung6_cfg_bytes(), root / "upper")
    replacement_delta = {
        name: [
            legacy.rung5.rung6.rung10._token(old[name]),
            legacy.rung5.rung6.rung10._token(new[name]),
        ]
        for name in sorted(set(old) | set(new))
        if old.get(name) != new.get(name)
    }
    boundary_delta = {
        name: [
            legacy.rung5.rung6.rung10._token(upper[name]),
            legacy.rung5.rung6.rung10._token(new[name]),
        ]
        for name in sorted(set(upper) | set(new))
        if upper.get(name) != new.get(name)
    }
    require(
        replacement_delta == {"namzdf.nn_havtb": ["1", "0"]},
        f"replacement delta changed: {replacement_delta}",
    )
    require(
        boundary_delta == {"namsbc.ln_rnf": [".true.", ".false."]},
        f"rung boundary changed: {boundary_delta}",
    )
    old_lines = legacy.rung5.rung5_cfg_bytes().decode().splitlines()
    new_lines = cfg.decode().splitlines()
    line_delta = [
        [index + 1, before, after]
        for index, (before, after) in enumerate(zip(old_lines, new_lines))
        if before != after
    ]
    require(
        len(old_lines) == len(new_lines) and line_delta == [[403, rung6.OLD_LINE, rung6.NEW_LINE]],
        f"physical-line delta changed: {line_delta}",
    )
    source = (COMPILED / "zdfphy.f90").read_text()
    for needle in ("avtb_2d(:,:) = 1._wp", "IF( nn_havtb == 1 ) THEN"):
        require(needle in source, f"compiled background-shape branch changed: {needle}")
    return {
        "status": "PREFLIGHT_PASS_RUNG5_HAVTB0",
        "claim_label": "independent",
        "rung": 5,
        "decision": 83,
        "superseded_record": str(record),
        "replacement_delta": replacement_delta,
        "line_delta": line_delta,
        "rung6_boundary_delta": boundary_delta,
        "compiled_consequences": {
            "runoff": False,
            "nn_havtb": 0,
            "avtb_2d": "uniform_1",
        },
        "binary_sha256": BINARY_SHA,
        "cpp_keys": list(rung6.legacy.rung10.CPP_KEYS),
    }


def stage_deck(root: Path) -> None:
    record, _, _ = _old_paths()
    expected = {
        "namelist_cfg": rung5_cfg_bytes(),
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
    inherited_plant = (
        plant if plant in ("ice-sentinel-read", "tke-sentinel-read", "resolved-havtb") else "none"
    )
    inherited = rung6.validate_resolved(root, plant=inherited_plant)
    ocean = (root / "ocean.output").read_text(errors="strict")
    echoed = (root / "output.namelist.dyn").read_text(errors="strict")
    checks = {
        "ln_rnf_false": re.search(r"runoff / runoff mouths\s+ln_rnf\s*=\s*F\b", ocean) is not None,
        "runoff_active_print_absent": "sbc_rnf_init : runoff" not in ocean,
        "runoff_group_echoed": re.search(r"&NAMSBC_RNF\b", echoed, re.IGNORECASE) is not None,
        "runoff_false_values_echoed": all(
            re.search(pattern, echoed, re.IGNORECASE) is not None
            for pattern in (
                r"LN_RNF_MOUTH\s*=\s*F",
                r"LN_RNF_TEM\s*=\s*F",
                r"LN_RNF_SAL\s*=\s*F",
                r"LN_RNF_ICB\s*=\s*F",
            )
        ),
    }
    if plant == "resolved-consequence":
        checks["ln_rnf_false"] = False
    if plant == "runoff-group-unread":
        checks["runoff_group_echoed"] = False
    if plant == "active-runoff-print":
        checks["runoff_active_print_absent"] = False
    require(all(checks.values()), f"resolved rung-5 checks failed: {checks}")
    return {"status": "PASS_RUNG5_HAVTB0_RESOLVED", "upper_rung": inherited, **checks}


def validate_record(root: Path, *, expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant: {plant}")
    preflight(plant=plant if plant in PRECHECK_PLANTS else "none")
    record, _, _ = _old_paths()
    require(
        (root / "producer_commit.txt").read_text().strip() == expect_commit,
        "producer commit differs",
    )
    require(sha256(root / "nemo") == BINARY_SHA, "record binary differs")
    require(sha256(root / "compiled_l4_r69_surface.f90") == WRITER_SHA, "record writer differs")
    require(sha256(root / "compiled_stprk3.f90") == STPRK3_SHA, "record step program differs")
    require((root / "namelist_cfg").read_bytes() == execution_cfg_bytes(), "execution deck differs")
    require((root / "namelist_cfg.deck").read_bytes() == rung5_cfg_bytes(), "exact deck differs")
    require(
        json.loads((root / "hierarchy_manifest.json").read_text())
        == json.loads(MANIFEST.read_text()),
        "record manifest differs",
    )
    require(
        (root / "recorder_repair_manifest.json").read_bytes()
        == (record / "recorder_repair_manifest.json").read_bytes(),
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
    frames = legacy.validate_frames(root, plant=frame_plant)
    terminal = legacy.rung5.rung6.rung7.rung8.rung9.validate_terminal(root, plant=inherited_plant)
    month = legacy.rung5.rung6.rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    inventory = legacy.rung5.rung6.rung10.validate_sha_inventory(root, plant=inherited_plant)
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung5-havtb0-record-v1",
        "status": "PASS_RUNG5_HAVTB0_RECORD",
        "claim_label": "independent",
        "rung": 5,
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
    except (
        RuntimeError,
        GateError,
        rung6.GateError,
        legacy.GateError,
        legacy.rung5.GateError,
        legacy.rung5.rung6.GateError,
        OSError,
        KeyError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as error:
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
