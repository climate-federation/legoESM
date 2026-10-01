#!/usr/bin/env python3
"""Preflight and admit the ORCA2 hierarchy rung-8 oracle record."""

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
    nemo_testcase_l4_orca2_hier_decks_round2_gate as rung9,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round3_acquisition"
MANIFEST = ACQUISITION / "rung8_manifest.json"
SENTINEL = rung9.SENTINEL
RUNG9_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung9/record"
)
RUNG9_ADMISSION = RUNG9_RECORD.parent / "rung9_admission.json"
COMPILED = rung10.COMPILED

RUNG8_CFG_SHA = "6d9679c9ce9cae41c36a6d0a3dd4c996921302fba677b36975452c9a8b99656d"
SOURCE_SHA = {
    "zdfphy": "903e0d1220a2f9b38cdd55b6eece5dbbfed992f12f497155cca8633702e7451e",
    "sbcrnf": "1604b3054264eab1dd4207b6f882d591ffa48ec7e01ca334d801d4f69d2e0ebe",
    "zdfiwm": "13ba29ab568a4592ab1a712be93c30ba0322e41c93cd7ac39a6977c00ac76073",
    "trazdf": "029056166047354c397bc2b5748b48c4bd2b227edad0e6134176257c55f0410e",
}
SELECTORS = {
    "namsbc_rnf.ln_rnf_mouth": (
        "   ln_rnf_mouth = .true.    !  specific treatment at rivers mouths",
        "   ln_rnf_mouth = .false.   !  specific treatment at rivers mouths",
        162,
    ),
    "namzdf.ln_zdfddm": (
        "   ln_zdfddm   = .true.    ! double diffusive mixing",
        "   ln_zdfddm   = .false.   ! double diffusive mixing",
        394,
    ),
    "namzdf_iwm.ln_tsdiff": (
        "   ln_tsdiff   = .true.    !  account for differential T/S mixing (T) or not (F)",
        "   ln_tsdiff   = .false.   !  account for differential T/S mixing (T) or not (F)",
        425,
    ),
}
RETAINED_PARAMETERS = {
    "namsbc_rnf.rn_hrnf": "15.e0",
    "namsbc_rnf.rn_avt_rnf": "1.e-3",
    "namzdf.rn_avts": "1.e-4",
    "namzdf.rn_hsbfr": "1.6",
}
PLANTS = (
    "none",
    "deck-extra",
    "missing-selector",
    "retained-parameter",
    "build-pin",
    "field-name",
    "truncated",
    "missing-frame",
    "frame-nonfinite",
    "terminal-nonfinite",
    "terminal-step",
    "ice-sentinel-read",
    "resolved-consequence",
    "sha-inventory",
)


class GateError(RuntimeError):
    """A frozen rung-8 hierarchy predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def rung8_cfg_bytes(*, plant: str = "none") -> bytes:
    result = rung9.rung9_cfg_bytes().decode()
    for index, (old, new, _line) in enumerate(SELECTORS.values()):
        require(result.count(old) == 1, f"upper-rung selector line changed: {old}")
        if plant == "missing-selector" and index == 0:
            continue
        result = result.replace(old, new)
    if plant == "deck-extra":
        old = "   rn_rfact    =   1.e0    !  multiplicative factor for runoff"
        require(result.count(old) == 1, "deck-extra plant target changed")
        result = result.replace(old, old.replace("1.e0", "2.e0"))
    if plant == "retained-parameter":
        old = "      rn_avts  =    1.e-4     !  maximum avs (vertical mixing on salinity)"
        require(result.count(old) == 1, "retained-parameter plant target changed")
        result = result.replace(old, old.replace("1.e-4", "2.e-4"))
    return result.encode()


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
        plant in ("none", "deck-extra", "missing-selector", "retained-parameter", "build-pin"),
        f"invalid preflight plant: {plant}",
    )
    rung9.preflight()
    admission = json.loads(RUNG9_ADMISSION.read_text())
    require(admission.get("status") == "PASS_RUNG9_RECORD", "rung-9 admission is not PASS")
    require(admission.get("frames", {}).get("frames") == 480, "rung-9 frame evidence changed")

    manifest = json.loads(MANIFEST.read_text())
    require(manifest["format"] == "nemo-testcase-l4-orca2-hierarchy-deck-v1", "manifest format changed")
    require(manifest["rung"] == 8, "manifest rung changed")
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

    cfg = rung8_cfg_bytes(plant=plant)
    require(rung9.sha256_bytes(cfg) == RUNG8_CFG_SHA, "rung-8 ocean namelist digest/delta changed")
    require(manifest["namelists"]["namelist_cfg_sha256"] == RUNG8_CFG_SHA, "manifest deck pin changed")
    require(
        manifest["namelists"]["namelist_ice_cfg_sha256"] == rung9.RUNG10_ICE_CFG_SHA,
        "manifest retained ice artifact changed",
    )

    with tempfile.TemporaryDirectory(prefix="orca2-rung8-deck-") as temporary:
        root = Path(temporary)
        (root / "upper").mkdir()
        (root / "lower").mkdir()
        upper = _assignment_map(rung9.rung9_cfg_bytes(), root / "upper")
        lower = _assignment_map(cfg, root / "lower")
    changed = {
        name: [rung10._token(upper[name]), rung10._token(lower[name])]
        for name in sorted(set(upper) | set(lower))
        if upper.get(name) != lower.get(name)
    }
    expected_delta = {name: [".true.", ".false."] for name in SELECTORS}
    require(changed == expected_delta, f"rung delta is not the frozen module: {changed}")
    for name, expected in RETAINED_PARAMETERS.items():
        require(rung10._token(lower[name]) == expected, f"retained parameter changed: {name}")

    upper_lines = rung9.rung9_cfg_bytes().decode().splitlines()
    lower_lines = cfg.decode().splitlines()
    line_delta = [
        [index + 1, before, after]
        for index, (before, after) in enumerate(zip(upper_lines, lower_lines, strict=True))
        if before != after
    ]
    require(
        [row[0] for row in line_delta] == [value[2] for value in SELECTORS.values()],
        f"unexpected line delta: {line_delta}",
    )

    source = {name: (COMPILED / f"{name}.f90").read_text() for name in SOURCE_SHA}
    needles = {
        "zdfphy": (
            "IF( ln_rnf_mouth ) THEN",
            "IF( ln_zdfddm ) THEN",
            "avs(ji,jj,jk) = avt(ji,jj,jk)",
            "IF( ln_zdfiwm )   CALL zdf_iwm",
        ),
        "sbcrnf": (
            "IF( ln_rnf_mouth ) THEN",
            "rnfmsk(ji,jj) = 0._wp",
            "rnfmsk_z(:)   = 0._wp",
            "nkrnf = 0",
        ),
        "zdfiwm": (
            "IF( ln_tsdiff ) THEN",
            "zav_ratio(ji,jj) = 1._wp",
            "p_avs(ji,jj,jk) = p_avs(ji,jj,jk) + zav_wave(ji,jj) * zav_ratio(ji,jj)",
            "avmb(:) = rnu",
            "avtb(:) = 1.e-10_wp",
        ),
        "trazdf": (
            "jn == jp_sal .AND. ln_zdfddm",
            "use avt  for temperature",
        ),
    }
    for name, required in needles.items():
        require(all(needle in source[name] for needle in required), f"compiled rung-8 branch changed: {name}")
    symbol_files = {
        symbol: _symbol_files(symbol)
        for symbol in ("ln_rnf_mouth", "rn_hrnf", "rn_avt_rnf", "ln_tsdiff")
    }
    require(symbol_files == {
        "ln_rnf_mouth": ["sbcrnf.f90", "zdfphy.f90"],
        "rn_hrnf": ["sbcrnf.f90"],
        "rn_avt_rnf": ["sbcrnf.f90", "zdfphy.f90"],
        "ln_tsdiff": ["zdfiwm.f90"],
    }, f"compiled consumer inventory changed: {symbol_files}")

    return {
        "status": "PREFLIGHT_PASS_RUNG8",
        "rung": 8,
        "claim_label": "independent",
        "upper_rung_admission": "PASS_RUNG9_RECORD",
        "assignment_delta": changed,
        "line_delta": line_delta,
        "retained_parameters": RETAINED_PARAMETERS,
        "compiled_consequences": {
            "river_mouth_mask": 0,
            "river_mouth_levels": 0,
            "double_diffusion_called": False,
            "avs_equals_avt_before_waves": True,
            "internal_wave_salt_heat_ratio": 1,
            "internal_wave_background_reset": True,
        },
        "compiled_symbol_files": symbol_files,
        "cpp_keys": list(rung10.CPP_KEYS),
        "binary_sha256": rung10.EXPECTED["binary"],
        "ln_spc_dyn": "RETAINED_INERT_WITHOUT_key_agrif",
    }


def stage_deck(root: Path) -> None:
    expected = {
        "namelist_cfg": rung8_cfg_bytes(),
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
    rows = "".join(f"{rung9.sha256_bytes(payload)}  {name}\n" for name, payload in expected.items())
    ledger = root / "SHA256SUMS"
    if ledger.exists():
        require(ledger.read_text() == rows, "existing deck SHA256SUMS differs")
    else:
        ledger.write_text(rows)


def validate_resolved(root: Path, *, plant: str = "none") -> dict[str, object]:
    inherited_plant = plant if plant == "ice-sentinel-read" else "none"
    inherited = rung9.validate_resolved(root, plant=inherited_plant)
    ocean = (root / "ocean.output").read_text(errors="strict")
    checks = {
        "ln_rnf_mouth_false": re.search(r"specific river mouths treatment\s+ln_rnf_mouth\s*=\s*F\b", ocean) is not None,
        "river_mouth_arm_off": "No specific treatment at river mouths" in ocean,
        "ln_zdfddm_false": re.search(r"double diffusive mixing\s+ln_zdfddm\s*=\s*F\b", ocean) is not None,
        "avs_equals_avt": "No  double diffusive mixing: avs = avt" in ocean,
        "ln_zdfiwm_true": re.search(r"internal wave .*ln_zdfiwm\s*=\s*T\b", ocean) is not None,
        "ln_tsdiff_false": re.search(r"Differential internal wave-driven mixing .* =\s*F\b", ocean) is not None,
        "iwm_background_reset": "Force the background value applied to avm & avt in TKE" in ocean,
    }
    if plant == "resolved-consequence":
        checks["avs_equals_avt"] = False
    require(all(checks.values()), f"resolved rung-8 checks failed: {checks}")
    return {"status": "PASS_RUNG8_RESOLVED", "no_ice": inherited, **checks}


def validate_record(root: Path, *, expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant: {plant}")
    if plant in ("deck-extra", "missing-selector", "retained-parameter", "build-pin"):
        preflight(plant=plant)
    else:
        preflight()
    require((root / "producer_commit.txt").read_text().strip() == expect_commit, "producer commit differs")
    require(rung10.sha256(root / "nemo") == rung10.EXPECTED["binary"], "record binary differs")
    require((root / "namelist_cfg").read_bytes() == rung8_cfg_bytes(), "record ocean namelist differs")
    require((root / "namelist_ice_cfg").read_bytes() == SENTINEL.read_bytes(), "record ice sentinel differs")
    require(json.loads((root / "hierarchy_manifest.json").read_text()) == json.loads(MANIFEST.read_text()), "record manifest differs")
    frames = rung9.validate_frames(root, plant=plant)
    terminal = rung9.validate_terminal(root, plant=plant)
    month = rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    sha_inventory = rung10.validate_sha_inventory(root, plant=plant)
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung8-record-v1",
        "status": "PASS_RUNG8_RECORD",
        "claim_label": "independent",
        "rung": 8,
        "producer_commit": expect_commit,
        "deck_delta_from_rung9": {name: [True, False] for name in SELECTORS},
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
        rung9.GateError,
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
