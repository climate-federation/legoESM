#!/usr/bin/env python3
"""Preflight and admit Decision 83's replacement rung-6 oracle record."""

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
    nemo_testcase_l4_orca2_hier_decks_round5_gate as legacy,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (  # noqa: E402
    namelist_values,
)

HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round13_acquisition"
MANIFEST = ACQUISITION / "rung6_havtb0_manifest.json"
RUNG6_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung6"
)
CURRENT_RECORD = RUNG6_ROOT / "record"
SUPERSEDED_RECORD = RUNG6_ROOT / "record_havtb1_superseded"
CURRENT_ADMISSION = RUNG6_ROOT / "rung6_admission.json"
SUPERSEDED_ADMISSION = RUNG6_ROOT / "rung6_admission_havtb1_superseded.json"
COMPILED = legacy.COMPILED

OLD_LINE = (
    "   nn_havtb    =    1         !  horizontal shape for avtb (=1) or not (=0)"
)
NEW_LINE = OLD_LINE.replace("=    1", "=    0")
OLD_CFG_SHA = legacy.RUNG6_CFG_SHA
OLD_EXECUTION_CFG_SHA = legacy.EXECUTION_CFG_SHA
NEW_CFG_SHA = "351ef6310b3f98e82dd4907bb370cb50288750db2502295c5c6c72eeef0feef8"
NEW_EXECUTION_CFG_SHA = "b4967ae2138af43c112d6826bf17d509a82e196da169b6881135bb5b9553089b"
OLD_ADMISSION_SHA = "45d76acb89ff7377563ae54734c9644441aae07b8a3bed82fae4a3ed7b8e1d46"
OLD_RECORD_INVENTORY_SHA = "f2e3f89de5859d51c0135c805c92ebd7b5157d6e31537095df159a2f3928f488"
OLD_MANIFEST_SHA = "64f9f4b609aa9e7d603d300038600aecae8f314e886eda1b3ef9e77d388ec7d6"
ZDFPHY_SHA = legacy.SOURCE_SHA["zdfphy"]

PRECHECK_PLANTS = (
    "deck-extra",
    "missing-havtb",
    "retained-coefficient",
    "source-pin",
    "superseded-pin",
)
RECORD_PLANTS = (
    "field-name",
    "truncated",
    "missing-frame",
    "frame-nonfinite",
    "terminal-nonfinite",
    "terminal-step",
    "ice-sentinel-read",
    "tke-sentinel-read",
    "resolved-consequence",
    "resolved-havtb",
    "sha-inventory",
)
PLANTS = ("none", *PRECHECK_PLANTS, *RECORD_PLANTS)


class GateError(RuntimeError):
    """A frozen Decision-83 rung-6 predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        target = (
            "   rn_avt0     =   1.2e-5     !  vertical eddy diffusivity [m2/s]"
            "       (background Kz if ln_zdfcst=F)"
        )
        require(text.count(target) == 1, "retained-coefficient target changed")
        text = text.replace(target, target.replace("1.2e-5", "2.2e-5"))
    return text.encode()


def rung6_cfg_bytes(*, plant: str = "none") -> bytes:
    return _replace_havtb(legacy.rung6_cfg_bytes(), plant=plant)


def execution_cfg_bytes() -> bytes:
    return _replace_havtb(legacy.execution_cfg_bytes())


def _superseded_paths() -> tuple[Path, Path]:
    record = SUPERSEDED_RECORD if SUPERSEDED_RECORD.exists() else CURRENT_RECORD
    admission = (
        SUPERSEDED_ADMISSION if SUPERSEDED_ADMISSION.exists() else CURRENT_ADMISSION
    )
    return record, admission


def _assignment_map(payload: bytes, root: Path) -> dict[str, str]:
    target = root / "namelist_cfg"
    target.write_bytes(payload)
    return namelist_values(target)


def preflight(*, plant: str = "none") -> dict[str, object]:
    require(plant in ("none", *PRECHECK_PLANTS), f"invalid preflight plant: {plant}")
    legacy.preflight()
    record, admission_path = _superseded_paths()
    require(record.is_dir(), "superseded rung-6 record is absent")
    require(admission_path.is_file(), "superseded rung-6 admission is absent")

    admission_sha = sha256(admission_path)
    inventory_sha = sha256(record / "SHA256SUMS")
    if plant == "superseded-pin":
        inventory_sha = "0" * 64
    require(admission_sha == OLD_ADMISSION_SHA, "superseded admission changed")
    require(inventory_sha == OLD_RECORD_INVENTORY_SHA, "superseded inventory changed")
    admission = json.loads(admission_path.read_text())
    require(admission.get("status") == "PASS_RUNG6_RECORD", "superseded record was not admitted")
    require(admission.get("frames", {}).get("frames") == 480, "superseded frame count changed")
    require(sha256(record / "namelist_cfg.deck") == OLD_CFG_SHA, "superseded deck changed")
    require(
        sha256(record / "namelist_cfg") == OLD_EXECUTION_CFG_SHA,
        "superseded execution deck changed",
    )
    require(
        sha256(record / "hierarchy_manifest.json") == OLD_MANIFEST_SHA,
        "superseded manifest changed",
    )

    manifest = json.loads(MANIFEST.read_text())
    require(manifest.get("rung") == 6, "manifest rung changed")
    require(manifest.get("decision") == 83, "manifest decision changed")
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
        manifest["superseded"].get("admission_sha256") == OLD_ADMISSION_SHA,
        "manifest superseded admission pin changed",
    )
    require(
        manifest["superseded"].get("record_inventory_sha256")
        == OLD_RECORD_INVENTORY_SHA,
        "manifest superseded inventory pin changed",
    )
    require(
        manifest["namelists"].get("namelist_cfg_sha256") == NEW_CFG_SHA,
        "manifest deck pin changed",
    )
    require(
        manifest["namelists"].get("execution_namelist_cfg_sha256")
        == NEW_EXECUTION_CFG_SHA,
        "manifest execution pin changed",
    )
    require(
        manifest["build"].get("binary_sha256") == legacy.rung10.EXPECTED["binary"],
        "manifest binary pin changed",
    )

    cfg = rung6_cfg_bytes(plant=plant)
    require(hashlib.sha256(cfg).hexdigest() == NEW_CFG_SHA, "replacement deck changed")
    require(
        hashlib.sha256(execution_cfg_bytes()).hexdigest() == NEW_EXECUTION_CFG_SHA,
        "replacement execution deck changed",
    )
    zdfphy_sha = sha256(COMPILED / "zdfphy.f90")
    if plant == "source-pin":
        zdfphy_sha = "0" * 64
    require(zdfphy_sha == ZDFPHY_SHA, "compiled zdfphy changed")

    with tempfile.TemporaryDirectory(prefix="orca2-rung6-havtb0-") as temporary:
        root = Path(temporary)
        (root / "old").mkdir()
        (root / "new").mkdir()
        (root / "upper").mkdir()
        old = _assignment_map(legacy.rung6_cfg_bytes(), root / "old")
        new = _assignment_map(cfg, root / "new")
        upper = _assignment_map(legacy.rung7.rung7_cfg_bytes(), root / "upper")
    replacement_delta = {
        name: [legacy.rung10._token(old[name]), legacy.rung10._token(new[name])]
        for name in sorted(set(old) | set(new))
        if old.get(name) != new.get(name)
    }
    require(
        replacement_delta == {"namzdf.nn_havtb": ["1", "0"]},
        f"replacement is not nn_havtb-only: {replacement_delta}",
    )
    boundary_delta = {
        name: [
            "<reference:.false.>"
            if name == "namzdf.ln_zdfcst" and name not in upper
            else legacy.rung10._token(upper[name]),
            legacy.rung10._token(new[name]),
        ]
        for name in sorted(set(upper) | set(new))
        if upper.get(name) != new.get(name)
    }
    require(
        boundary_delta
        == {
            "namzdf.ln_zdfcst": ["<reference:.false.>", ".true."],
            "namzdf.ln_zdftke": [".true.", ".false."],
            "namzdf.nn_havtb": ["1", "0"],
        },
        f"Decision-83 rung boundary changed: {boundary_delta}",
    )
    old_lines = legacy.rung6_cfg_bytes().decode().splitlines()
    new_lines = cfg.decode().splitlines()
    line_delta = [
        [index + 1, pair[0], pair[1]]
        for index, pair in enumerate(zip(old_lines, new_lines))
        if pair[0] != pair[1]
    ]
    require(len(old_lines) == len(new_lines) and line_delta == [[403, OLD_LINE, NEW_LINE]], f"physical-line delta changed: {line_delta}")

    source = (COMPILED / "zdfphy.f90").read_text()
    for needle in (
        "avtb_2d(:,:) = 1._wp",
        "IF( nn_havtb == 1 ) THEN",
        "avt_k(:,:,jk)    = avtb_2d(:,:) * avtb(jk)",
    ):
        require(needle in source, f"compiled background-shape branch changed: {needle}")
    return {
        "status": "PREFLIGHT_PASS_RUNG6_HAVTB0",
        "claim_label": "independent",
        "rung": 6,
        "decision": 83,
        "superseded_record": str(record),
        "replacement_delta": replacement_delta,
        "line_delta": line_delta,
        "rung7_boundary_delta": boundary_delta,
        "compiled_consequences": {
            "closure": "np_CST",
            "rn_avm0": 1.2e-4,
            "rn_avt0": 1.2e-5,
            "nn_avb": 0,
            "nn_havtb": 0,
            "avtb_2d": "uniform_1",
            "tke_initializer_called": False,
        },
        "binary_sha256": legacy.rung10.EXPECTED["binary"],
        "cpp_keys": list(legacy.rung10.CPP_KEYS),
    }


def stage_deck(root: Path) -> None:
    expected = {
        "namelist_cfg": rung6_cfg_bytes(),
        "namelist_ice_cfg": legacy.rung10.RUNG_ICE_CFG.read_bytes(),
        "manifest.json": MANIFEST.read_bytes(),
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in expected.items():
        target = root / name
        if target.exists():
            require(target.is_file() and target.read_bytes() == payload, f"existing deck differs: {target}")
        else:
            target.write_bytes(payload)
    rows = "".join(f"{hashlib.sha256(payload).hexdigest()}  {name}\n" for name, payload in expected.items())
    ledger = root / "SHA256SUMS"
    if ledger.exists():
        require(ledger.read_text() == rows, "deck SHA256SUMS changed")
    else:
        ledger.write_text(rows)


def validate_resolved(root: Path, *, plant: str = "none") -> dict[str, object]:
    inherited_plant = plant if plant == "ice-sentinel-read" else "none"
    inherited = legacy.rung7.rung8.rung9.validate_resolved(
        root, plant=inherited_plant, tke_active=False
    )
    ocean = (root / "ocean.output").read_text(errors="strict")
    checks = {
        "ln_zdfcst_true": re.search(r"constant vertical mixing coefficient\s+ln_zdfcst\s*=\s*T\b", ocean) is not None,
        "ln_zdftke_false": re.search(r"Turbulent Kinetic Energy closure \(TKE\)\s+ln_zdftke\s*=\s*F\b", ocean) is not None,
        "rn_avm0_retained": re.search(r"rn_avm0\s*=\s*1\.2000000000000000E-004", ocean) is not None,
        "rn_avt0_retained": re.search(r"rn_avt0\s*=\s*1\.2000000000000000E-005", ocean) is not None,
        "nn_avb_retained": re.search(r"nn_avb\s*=\s*0\b", ocean) is not None,
        "nn_havtb_uniform": re.search(r"nn_havtb\s*=\s*0\b", ocean) is not None,
        "tke_initializer_absent": "zdf_tke_init :" not in ocean,
        "tke_sentinel_unread": "THIS_GROUP_MUST_NOT_BE_READ" not in ocean,
    }
    if plant == "tke-sentinel-read":
        checks["tke_sentinel_unread"] = False
    if plant == "resolved-consequence":
        checks["ln_zdfcst_true"] = False
    if plant == "resolved-havtb":
        checks["nn_havtb_uniform"] = False
    require(all(checks.values()), f"resolved replacement checks failed: {checks}")
    return {"status": "PASS_RUNG6_HAVTB0_RESOLVED", "no_ice": inherited, **checks}


def validate_record(
    root: Path, *, expect_commit: str, plant: str = "none"
) -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant: {plant}")
    preflight(plant=plant if plant in PRECHECK_PLANTS else "none")
    require((root / "producer_commit.txt").read_text().strip() == expect_commit, "producer commit differs")
    require(sha256(root / "nemo") == legacy.rung10.EXPECTED["binary"], "record binary differs")
    require((root / "namelist_cfg").read_bytes() == execution_cfg_bytes(), "execution deck differs")
    require((root / "namelist_cfg.deck").read_bytes() == rung6_cfg_bytes(), "exact deck differs")
    require(json.loads((root / "hierarchy_manifest.json").read_text()) == json.loads(MANIFEST.read_text()), "record manifest differs")
    frames = legacy.rung7.rung8.rung9.validate_frames(root, plant=plant)
    terminal = legacy.rung7.rung8.rung9.validate_terminal(root, plant=plant)
    month = legacy.rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    inventory = legacy.rung10.validate_sha_inventory(root, plant=plant)
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung6-havtb0-record-v1",
        "status": "PASS_RUNG6_HAVTB0_RECORD",
        "claim_label": "independent",
        "rung": 6,
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
            require(args.record is not None and args.expect_commit, "record and expected commit are required")
            report = validate_record(args.record, expect_commit=args.expect_commit, plant=args.plant)
        report["worktree"] = worktree_stamp()
    except (
        RuntimeError,
        GateError,
        legacy.GateError,
        legacy.rung7.GateError,
        legacy.rung7.rung8.GateError,
        legacy.rung7.rung8.rung9.GateError,
        legacy.rung10.GateError,
        legacy.rung10.surface.GateError,
        KeyError,
        OSError,
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
