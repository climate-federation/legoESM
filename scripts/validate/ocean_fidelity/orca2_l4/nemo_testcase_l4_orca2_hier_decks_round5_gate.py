#!/usr/bin/env python3
"""Preflight and admit the ORCA2 hierarchy rung-6 oracle record."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_hier_decks_round1_gate as rung10,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_hier_decks_round4_gate as rung7,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round5_acquisition"
MANIFEST = ACQUISITION / "rung6_manifest.json"
RUNG7_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung7/record"
)
RUNG7_ADMISSION = RUNG7_RECORD.parent / "rung7_admission.json"
COMPILED = rung10.COMPILED

RUNG6_CFG_SHA = "d2828565a16bd79bd7f89a39297b59d3f9b953583294f2f1769f6a66c8fca331"
EXECUTION_CFG_SHA = "adb03ad62d304b9c8141c38eb699e639d7ef0b559ddd95ef1e62572ed163233f"
SOURCE_SHA = {
    "asmbkg": "e138ec54ef30dbf4108734e0411abaa5629e260dc3be01bb0fabca7b537dc1da",
    "dia25h": "b3d530c9b7a16c74d3deb2e442c54626c877f7731f9183dcdb375d65325ee61c",
    "zdf_oce": "875d466f3ae19d354dfdd4846d4224a957842f2d959c9b7b6d3c71655b4bc98b",
    "zdfphy": "903e0d1220a2f9b38cdd55b6eece5dbbfed992f12f497155cca8633702e7451e",
    "zdftke": "0f290190a33e4713bba218047451b35e25327bd589a8a74536a2db506efe81c2",
}
MANAGER_ANCHOR = (
    "&namzdf        !   vertical physics manager                             (default: NO selection)\n"
    "!-----------------------------------------------------------------------\n"
)
CONSTANT_SELECTOR = "   ln_zdfcst   = .true.       !  constant mixing"
TKE_SELECTOR = (
    "   ln_zdftke   = .true.       !  Turbulent Kinetic Energy closure       (T =>   fill namzdf_tke)",
    "   ln_zdftke   = .false.      !  Turbulent Kinetic Energy closure       (T =>   fill namzdf_tke)",
)
SENTINEL = (
    "      nn_mxl      =   3       !  mixing length: = 0 bounded by the distance to surface and bottom",
    "      nn_mxl      = THIS_GROUP_MUST_NOT_BE_READ",
)
RETAINED_PARAMETERS = {
    "namzdf.rn_avm0": "1.2e-4",
    "namzdf.rn_avt0": "1.2e-5",
    "namzdf.nn_avb": "0",
    "namzdf.nn_havtb": "1",
}
PLANTS = (
    "none",
    "deck-extra",
    "missing-constant-selector",
    "missing-tke-selector",
    "retained-coefficient",
    "build-pin",
    "field-name",
    "truncated",
    "missing-frame",
    "frame-nonfinite",
    "terminal-nonfinite",
    "terminal-step",
    "ice-sentinel-read",
    "tke-sentinel-read",
    "resolved-consequence",
    "sha-inventory",
)


class GateError(RuntimeError):
    """A frozen rung-6 hierarchy predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def rung6_cfg_bytes(*, plant: str = "none") -> bytes:
    result = rung7.rung7_cfg_bytes().decode()
    old_tke, new_tke = TKE_SELECTOR
    require(result.count(MANAGER_ANCHOR) == 1, "upper-rung manager anchor changed")
    require(result.count(old_tke) == 1, "upper-rung TKE selector changed")
    if plant != "missing-constant-selector":
        result = result.replace(MANAGER_ANCHOR, MANAGER_ANCHOR + CONSTANT_SELECTOR + "\n")
    if plant != "missing-tke-selector":
        result = result.replace(old_tke, new_tke)
    if plant == "deck-extra":
        target = "   rn_rfact    =   1.e0    !  multiplicative factor for runoff"
        require(result.count(target) == 1, "deck-extra plant target changed")
        result = result.replace(target, target.replace("1.e0", "2.e0"))
    if plant == "retained-coefficient":
        target = "   rn_avm0     =   1.2e-4     !  vertical eddy viscosity   [m2/s]       (background Kz if ln_zdfcst=F)"
        require(result.count(target) == 1, "retained-coefficient plant target changed")
        result = result.replace(target, target.replace("1.2e-4", "2.2e-4"))
    return result.encode()


def execution_cfg_bytes() -> bytes:
    result = rung6_cfg_bytes().decode()
    old, new = SENTINEL
    require(result.count(old) == 1, "TKE sentinel target changed")
    return result.replace(old, new).encode()


def _assignment_map(payload: bytes, root: Path) -> dict[str, str]:
    target = root / "namelist_cfg"
    target.write_bytes(payload)
    return namelist_values(target)


def _symbol_files(symbol: str) -> list[str]:
    return sorted(
        path.name
        for path in COMPILED.glob("*.f90")
        if symbol in path.read_text(errors="ignore")
    )


def preflight(*, plant: str = "none") -> dict[str, object]:
    require(
        plant in (
            "none",
            "deck-extra",
            "missing-constant-selector",
            "missing-tke-selector",
            "retained-coefficient",
            "build-pin",
        ),
        f"invalid preflight plant: {plant}",
    )
    rung7.preflight()
    admission = json.loads(RUNG7_ADMISSION.read_text())
    require(admission.get("status") == "PASS_RUNG7_RECORD", "rung-7 admission is not PASS")
    require(admission.get("frames", {}).get("frames") == 480, "rung-7 frame evidence changed")

    manifest = json.loads(MANIFEST.read_text())
    require(manifest["format"] == "nemo-testcase-l4-orca2-hierarchy-deck-v1", "manifest format changed")
    require(manifest["rung"] == 6, "manifest rung changed")
    require(manifest["run"] == {
        "mpi_ranks": 2,
        "first_step": 1,
        "last_step": 240,
        "from_rest": True,
        "initial_state_output": 1,
    }, "run protocol changed")
    require(tuple(manifest["build"]["cpp_keys"]) == rung10.CPP_KEYS, "CPP keys changed")
    expected_build = {
        "binary_sha256": rung10.EXPECTED["binary"],
        "cpp_keys_sha256": rung10.EXPECTED["cpp"],
        "compiled_stprk3_sha256": rung10.EXPECTED["stprk3"],
        "compiled_writer_sha256": rung10.EXPECTED["writer"],
        **{f"compiled_{name}_sha256": digest for name, digest in SOURCE_SHA.items()},
    }
    if plant == "build-pin":
        expected_build["compiled_zdfphy_sha256"] = "0" * 64
    for name, expected in expected_build.items():
        require(manifest["build"].get(name) == expected, f"manifest build pin changed: {name}")
    for name, expected in SOURCE_SHA.items():
        require(rung10.sha256(COMPILED / f"{name}.f90") == expected, f"compiled source changed: {name}")

    cfg = rung6_cfg_bytes(plant=plant)
    require(rung7.rung8.rung9.sha256_bytes(cfg) == RUNG6_CFG_SHA, "rung-6 ocean namelist digest/delta changed")
    require(rung7.rung8.rung9.sha256_bytes(execution_cfg_bytes()) == EXECUTION_CFG_SHA, "execution sentinel changed")
    require(manifest["namelists"]["namelist_cfg_sha256"] == RUNG6_CFG_SHA, "manifest deck pin changed")
    require(manifest["namelists"]["execution_namelist_cfg_sha256"] == EXECUTION_CFG_SHA, "manifest execution pin changed")
    require(manifest["namelists"]["namelist_ice_cfg_sha256"] == rung7.rung8.rung9.RUNG10_ICE_CFG_SHA, "retained ice artifact changed")

    with tempfile.TemporaryDirectory(prefix="orca2-rung6-deck-") as temporary:
        root = Path(temporary)
        (root / "upper").mkdir()
        (root / "lower").mkdir()
        upper = _assignment_map(rung7.rung7_cfg_bytes(), root / "upper")
        lower = _assignment_map(cfg, root / "lower")
    require("namzdf.ln_zdfcst" not in upper, "upper rung unexpectedly overrides reference ln_zdfcst")
    require(
        re.search(r"^\s*ln_zdfcst\s*=\s*\.false\.", (RUNG7_RECORD / "namelist_ref").read_text(), re.MULTILINE) is not None,
        "upper-rung reference ln_zdfcst default changed",
    )
    changed = {
        name: [
            "<reference:.false.>" if name == "namzdf.ln_zdfcst" and name not in upper else rung10._token(upper[name]),
            rung10._token(lower[name]),
        ]
        for name in sorted(set(upper) | set(lower))
        if upper.get(name) != lower.get(name)
    }
    require(changed == {
        "namzdf.ln_zdfcst": ["<reference:.false.>", ".true."],
        "namzdf.ln_zdftke": [".true.", ".false."],
    }, f"rung delta is not the frozen module: {changed}")
    for name, expected in RETAINED_PARAMETERS.items():
        require(rung10._token(lower[name]) == expected, f"retained coefficient changed: {name}")

    upper_lines = rung7.rung7_cfg_bytes().decode().splitlines()
    lower_lines = cfg.decode().splitlines()
    require(lower_lines[:389] == upper_lines[:389], "lines before closure delta changed")
    require(lower_lines[389] == CONSTANT_SELECTOR, "constant selector line changed")
    require(lower_lines[390] == TKE_SELECTOR[1], "TKE selector line changed")
    require(lower_lines[391:] == upper_lines[390:], "lines after closure delta changed")
    line_delta = [
        [390, "<absent; namelist_ref is .false.>", CONSTANT_SELECTOR],
        [391, TKE_SELECTOR[0], TKE_SELECTOR[1]],
    ]

    source = {name: (COMPILED / f"{name}.f90").read_text() for name in SOURCE_SHA}
    needles = {
        "asmbkg": ("IF( ln_zdftke ) THEN", "IF( ln_zdftke )   CALL iom_rstput"),
        "dia25h": ("IF( ln_zdftke ) THEN", 'CALL iom_put("tke25h", zw3d)'),
        "zdf_oce": ("LOGICAL , PUBLIC ::   ln_zdfcst", "LOGICAL , PUBLIC ::   ln_zdftke"),
        "zdfphy": (
            "IF( ln_zdfcst ) THEN   ;   ioptio = ioptio + 1   ;    nzdf_phy = np_CST",
            "IF( ln_zdftke ) THEN   ;   ioptio = ioptio + 1   ;    nzdf_phy = np_TKE   ;   CALL zdf_tke_init",
            "IF( ioptio /= 1 )    CALL ctl_stop",
            "IF( ln_zdfcst .OR. ln_zdfosm .OR. ln_zdfric ) THEN   ;   l_zdfsh2 = .FALSE.",
            "CASE( np_TKE )   ;   CALL zdf_tke",
            "IF( ln_zdftke )   CALL tke_rst",
        ),
        "zdftke": ("SUBROUTINE zdf_tke_init", "READ  ( numnam_cfg, namzdf_tke"),
    }
    for name, required in needles.items():
        require(all(needle in source[name] for needle in required), f"compiled rung-6 branch changed: {name}")
    selector_files = {
        "ln_zdfcst": _symbol_files("ln_zdfcst"),
        "ln_zdftke": _symbol_files("ln_zdftke"),
    }
    require(selector_files == {
        "ln_zdfcst": ["zdf_oce.f90", "zdfphy.f90"],
        "ln_zdftke": ["asmbkg.f90", "dia25h.f90", "zdf_oce.f90", "zdfphy.f90"],
    }, f"compiled closure selector inventory changed: {selector_files}")

    return {
        "status": "PREFLIGHT_PASS_RUNG6",
        "rung": 6,
        "claim_label": "independent",
        "upper_rung_admission": "PASS_RUNG7_RECORD",
        "assignment_delta": changed,
        "line_delta": line_delta,
        "retained_parameters": RETAINED_PARAMETERS,
        "compiled_consequences": {
            "closure": "np_CST",
            "tke_initializer_called": False,
            "tke_step_called": False,
            "shear_production_computed": False,
            "tke_restart_written": False,
            "rn_avm0": 1.2e-4,
            "rn_avt0": 1.2e-5,
            "nn_avb": 0,
            "nn_havtb": 1,
        },
        "compiled_selector_files": selector_files,
        "cpp_keys": list(rung10.CPP_KEYS),
        "binary_sha256": rung10.EXPECTED["binary"],
        "ln_spc_dyn": "RETAINED_INERT_WITHOUT_key_agrif",
    }


def stage_deck(root: Path) -> None:
    expected = {
        "namelist_cfg": rung6_cfg_bytes(),
        "namelist_ice_cfg": rung10.RUNG_ICE_CFG.read_bytes(),
        "manifest.json": MANIFEST.read_bytes(),
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in expected.items():
        target = root / name
        if target.exists():
            require(target.is_file() and target.read_bytes() == payload, f"existing deck artifact differs: {target}")
        else:
            target.write_bytes(payload)
    rows = "".join(f"{rung7.rung8.rung9.sha256_bytes(payload)}  {name}\n" for name, payload in expected.items())
    ledger = root / "SHA256SUMS"
    if ledger.exists():
        require(ledger.read_text() == rows, "existing deck SHA256SUMS differs")
    else:
        ledger.write_text(rows)


def validate_resolved(root: Path, *, plant: str = "none") -> dict[str, object]:
    inherited_plant = plant if plant == "ice-sentinel-read" else "none"
    inherited = rung7.rung8.rung9.validate_resolved(
        root, plant=inherited_plant, tke_active=False
    )
    ocean = (root / "ocean.output").read_text(errors="strict")
    checks = {
        "ln_zdfcst_true": re.search(r"constant vertical mixing coefficient\s+ln_zdfcst\s*=\s*T\b", ocean) is not None,
        "ln_zdftke_false": re.search(r"Turbulent Kinetic Energy closure \(TKE\)\s+ln_zdftke\s*=\s*F\b", ocean) is not None,
        "rn_avm0_retained": re.search(r"rn_avm0\s*=\s*1\.2000000000000000E-004", ocean) is not None,
        "rn_avt0_retained": re.search(r"rn_avt0\s*=\s*1\.2000000000000000E-005", ocean) is not None,
        "nn_avb_retained": re.search(r"nn_avb\s*=\s*0\b", ocean) is not None,
        "nn_havtb_retained": re.search(r"nn_havtb\s*=\s*1\b", ocean) is not None,
        "tke_initializer_absent": "zdf_tke_init :" not in ocean,
        "tke_namelist_absent": "minimum value of tke" not in ocean,
        "tke_restart_absent": "---- tke_rst ----" not in ocean,
        "tke_sentinel_unread": "THIS_GROUP_MUST_NOT_BE_READ" not in ocean,
    }
    if plant == "tke-sentinel-read":
        checks["tke_sentinel_unread"] = False
    if plant == "resolved-consequence":
        checks["ln_zdfcst_true"] = False
    require(all(checks.values()), f"resolved rung-6 checks failed: {checks}")
    return {"status": "PASS_RUNG6_RESOLVED", "no_ice": inherited, **checks}


def validate_record(root: Path, *, expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant: {plant}")
    if plant in (
        "deck-extra",
        "missing-constant-selector",
        "missing-tke-selector",
        "retained-coefficient",
        "build-pin",
    ):
        preflight(plant=plant)
    else:
        preflight()
    require((root / "producer_commit.txt").read_text().strip() == expect_commit, "producer commit differs")
    require(rung10.sha256(root / "nemo") == rung10.EXPECTED["binary"], "record binary differs")
    require((root / "namelist_cfg").read_bytes() == execution_cfg_bytes(), "record execution namelist differs")
    require((root / "namelist_cfg.deck").read_bytes() == rung6_cfg_bytes(), "record exact deck namelist differs")
    require((root / "namelist_ice_cfg").read_bytes() == rung7.rung8.rung9.SENTINEL.read_bytes(), "record ice sentinel differs")
    require(json.loads((root / "hierarchy_manifest.json").read_text()) == json.loads(MANIFEST.read_text()), "record manifest differs")
    frames = rung7.rung8.rung9.validate_frames(root, plant=plant)
    terminal = rung7.rung8.rung9.validate_terminal(root, plant=plant)
    month = rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    sha_inventory = rung10.validate_sha_inventory(root, plant=plant)
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung6-record-v1",
        "status": "PASS_RUNG6_RECORD",
        "claim_label": "independent",
        "rung": 6,
        "producer_commit": expect_commit,
        "deck_delta_from_rung7": {
            "namzdf.ln_zdfcst": [False, True],
            "namzdf.ln_zdftke": [True, False],
        },
        "frames": frames,
        "terminal": terminal,
        "month_products": month,
        "resolved": resolved,
        "sha_inventory": sha_inventory,
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
        GateError,
        rung7.GateError,
        rung7.rung8.GateError,
        rung7.rung8.rung9.GateError,
        rung10.GateError,
        rung10.surface.GateError,
        KeyError,
        OSError,
        TypeError,
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
