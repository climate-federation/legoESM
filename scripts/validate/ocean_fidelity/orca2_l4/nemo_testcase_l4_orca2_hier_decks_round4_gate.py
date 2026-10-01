#!/usr/bin/env python3
"""Preflight and admit the ORCA2 hierarchy rung-7 oracle record."""

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
    nemo_testcase_l4_orca2_hier_decks_round3_gate as rung8,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round4_acquisition"
MANIFEST = ACQUISITION / "rung7_manifest.json"
RUNG8_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung8/record"
)
RUNG8_ADMISSION = RUNG8_RECORD.parent / "rung8_admission.json"
COMPILED = rung10.COMPILED

RUNG7_CFG_SHA = "2a36188c237b8d1dcfd52f1fb1a009fd0f45634eacc36b42ff28340f64edd27d"
EXECUTION_CFG_SHA = "4804d56f905bfedf124ed3472cfd0589dfceb6c88aeaedc77db9451715f365f8"
SOURCE_SHA = {
    "zdfphy": "903e0d1220a2f9b38cdd55b6eece5dbbfed992f12f497155cca8633702e7451e",
    "zdfiwm": "13ba29ab568a4592ab1a712be93c30ba0322e41c93cd7ac39a6977c00ac76073",
    "zdftke": "0f290190a33e4713bba218047451b35e25327bd589a8a74536a2db506efe81c2",
}
SELECTOR = (
    "   ln_zdfiwm   = .true.       ! internal wave-induced mixing            (T =>   fill namzdf_iwm)",
    "   ln_zdfiwm   = .false.      ! internal wave-induced mixing            (T =>   fill namzdf_iwm)",
    397,
)
SENTINEL = (
    "   ln_mevar    = .false.    !  variable (T) or constant (F) mixing efficiency",
    "   ln_mevar    = THIS_GROUP_MUST_NOT_BE_READ",
    424,
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
    "missing-selector",
    "retained-background",
    "build-pin",
    "field-name",
    "truncated",
    "missing-frame",
    "frame-nonfinite",
    "terminal-nonfinite",
    "terminal-step",
    "ice-sentinel-read",
    "iwm-sentinel-read",
    "resolved-consequence",
    "sha-inventory",
)


class GateError(RuntimeError):
    """A frozen rung-7 hierarchy predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def rung7_cfg_bytes(*, plant: str = "none") -> bytes:
    result = rung8.rung8_cfg_bytes().decode()
    old, new, _line = SELECTOR
    require(result.count(old) == 1, "upper-rung internal-wave selector changed")
    if plant != "missing-selector":
        result = result.replace(old, new)
    if plant == "deck-extra":
        target = "   rn_rfact    =   1.e0    !  multiplicative factor for runoff"
        require(result.count(target) == 1, "deck-extra plant target changed")
        result = result.replace(target, target.replace("1.e0", "2.e0"))
    if plant == "retained-background":
        target = "   rn_avt0     =   1.2e-5     !  vertical eddy diffusivity [m2/s]       (background Kz if ln_zdfcst=F)"
        require(result.count(target) == 1, "retained-background plant target changed")
        result = result.replace(target, target.replace("1.2e-5", "2.2e-5"))
    return result.encode()


def execution_cfg_bytes() -> bytes:
    result = rung7_cfg_bytes().decode()
    old, new, _line = SENTINEL
    require(result.count(old) == 1, "internal-wave sentinel target changed")
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
        plant in ("none", "deck-extra", "missing-selector", "retained-background", "build-pin"),
        f"invalid preflight plant: {plant}",
    )
    rung8.preflight()
    admission = json.loads(RUNG8_ADMISSION.read_text())
    require(admission.get("status") == "PASS_RUNG8_RECORD", "rung-8 admission is not PASS")
    require(admission.get("frames", {}).get("frames") == 480, "rung-8 frame evidence changed")

    manifest = json.loads(MANIFEST.read_text())
    require(manifest["format"] == "nemo-testcase-l4-orca2-hierarchy-deck-v1", "manifest format changed")
    require(manifest["rung"] == 7, "manifest rung changed")
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
        expected_build["compiled_zdftke_sha256"] = "0" * 64
    for name, expected in expected_build.items():
        require(manifest["build"].get(name) == expected, f"manifest build pin changed: {name}")
    for name, expected in SOURCE_SHA.items():
        require(rung10.sha256(COMPILED / f"{name}.f90") == expected, f"compiled source changed: {name}")

    cfg = rung7_cfg_bytes(plant=plant)
    require(rung8.rung9.sha256_bytes(cfg) == RUNG7_CFG_SHA, "rung-7 ocean namelist digest/delta changed")
    require(rung8.rung9.sha256_bytes(execution_cfg_bytes()) == EXECUTION_CFG_SHA, "execution sentinel changed")
    require(manifest["namelists"]["namelist_cfg_sha256"] == RUNG7_CFG_SHA, "manifest deck pin changed")
    require(manifest["namelists"]["execution_namelist_cfg_sha256"] == EXECUTION_CFG_SHA, "manifest execution pin changed")
    require(manifest["namelists"]["namelist_ice_cfg_sha256"] == rung8.rung9.RUNG10_ICE_CFG_SHA, "retained ice artifact changed")

    with tempfile.TemporaryDirectory(prefix="orca2-rung7-deck-") as temporary:
        root = Path(temporary)
        (root / "upper").mkdir()
        (root / "lower").mkdir()
        upper = _assignment_map(rung8.rung8_cfg_bytes(), root / "upper")
        lower = _assignment_map(cfg, root / "lower")
    changed = {
        name: [rung10._token(upper[name]), rung10._token(lower[name])]
        for name in sorted(set(upper) | set(lower))
        if upper.get(name) != lower.get(name)
    }
    require(changed == {"namzdf.ln_zdfiwm": [".true.", ".false."]}, f"rung delta is not the frozen module: {changed}")
    for name, expected in RETAINED_PARAMETERS.items():
        require(rung10._token(lower[name]) == expected, f"retained background changed: {name}")

    upper_lines = rung8.rung8_cfg_bytes().decode().splitlines()
    lower_lines = cfg.decode().splitlines()
    line_delta = [
        [index + 1, before, after]
        for index, (before, after) in enumerate(zip(upper_lines, lower_lines, strict=True))
        if before != after
    ]
    require([row[0] for row in line_delta] == [SELECTOR[2]], f"unexpected line delta: {line_delta}")

    source = {name: (COMPILED / f"{name}.f90").read_text() for name in SOURCE_SHA}
    needles = {
        "zdfphy": (
            "IF( ln_zdfiwm )   CALL zdf_iwm_init",
            "IF( ln_zdfiwm )   CALL zdf_iwm",
            "avmb(:) = rn_avm0",
            "avtb(:) = rn_avt0",
        ),
        "zdfiwm": (
            "avmb(:) = rnu",
            "avtb(:) = 1.e-10_wp",
            "avtb_2d(:,:) = 1._wp",
        ),
        "zdftke": (
            "IF( ln_zdfiwm ) THEN",
            "rn_emin  = 1.e-10_wp",
            "rmxl_min = 1.e-03_wp",
            "rmxl_min = 1.e-6_wp / ( rn_ediff * SQRT( rn_emin ) )",
        ),
    }
    for name, required in needles.items():
        require(all(needle in source[name] for needle in required), f"compiled rung-7 branch changed: {name}")
    symbol_files = _symbol_files("ln_zdfiwm")
    require(symbol_files == ["zdf_oce.f90", "zdfphy.f90", "zdftke.f90"], f"compiled selector inventory changed: {symbol_files}")

    return {
        "status": "PREFLIGHT_PASS_RUNG7",
        "rung": 7,
        "claim_label": "independent",
        "upper_rung_admission": "PASS_RUNG8_RECORD",
        "assignment_delta": changed,
        "line_delta": line_delta,
        "retained_parameters": RETAINED_PARAMETERS,
        "compiled_consequences": {
            "internal_wave_initializer_called": False,
            "internal_wave_step_called": False,
            "rn_avm0": 1.2e-4,
            "rn_avt0": 1.2e-5,
            "rn_emin": 1.0e-6,
            "rmxl_min": 1.0e-2,
            "nn_havtb": 1,
        },
        "compiled_selector_files": symbol_files,
        "cpp_keys": list(rung10.CPP_KEYS),
        "binary_sha256": rung10.EXPECTED["binary"],
        "ln_spc_dyn": "RETAINED_INERT_WITHOUT_key_agrif",
    }


def stage_deck(root: Path) -> None:
    expected = {
        "namelist_cfg": rung7_cfg_bytes(),
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
    rows = "".join(f"{rung8.rung9.sha256_bytes(payload)}  {name}\n" for name, payload in expected.items())
    ledger = root / "SHA256SUMS"
    if ledger.exists():
        require(ledger.read_text() == rows, "existing deck SHA256SUMS differs")
    else:
        ledger.write_text(rows)


def validate_resolved(root: Path, *, plant: str = "none") -> dict[str, object]:
    inherited_plant = plant if plant == "ice-sentinel-read" else "none"
    inherited = rung8.rung9.validate_resolved(root, plant=inherited_plant)
    ocean = (root / "ocean.output").read_text(errors="strict")
    checks = {
        "ln_zdfiwm_false": re.search(r"internal wave .*ln_zdfiwm\s*=\s*F\b", ocean) is not None,
        "iwm_initializer_absent": "zdf_iwm_init : internal wave-driven mixing" not in ocean,
        "iwm_reset_absent": "Force the background value applied to avm & avt in TKE" not in ocean,
        "rn_avm0_retained": re.search(r"rn_avm0\s*=\s*1\.2000000000000000E-004", ocean) is not None,
        "rn_avt0_retained": re.search(r"rn_avt0\s*=\s*1\.2000000000000000E-005", ocean) is not None,
        "rn_emin_retained": re.search(r"minimum value of tke\s+rn_emin\s*=\s*9\.9999999999999995E-007", ocean) is not None,
        "rmxl_min_standard": re.search(r"minimum mixing length with your parameters rmxl_min\s*=\s*9\.9999999999999985E-003", ocean) is not None,
        "iwm_sentinel_unread": "THIS_GROUP_MUST_NOT_BE_READ" not in ocean,
    }
    if plant == "iwm-sentinel-read":
        checks["iwm_sentinel_unread"] = False
    if plant == "resolved-consequence":
        checks["rmxl_min_standard"] = False
    require(all(checks.values()), f"resolved rung-7 checks failed: {checks}")
    return {"status": "PASS_RUNG7_RESOLVED", "no_ice": inherited, **checks}


def validate_record(root: Path, *, expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant: {plant}")
    if plant in ("deck-extra", "missing-selector", "retained-background", "build-pin"):
        preflight(plant=plant)
    else:
        preflight()
    require((root / "producer_commit.txt").read_text().strip() == expect_commit, "producer commit differs")
    require(rung10.sha256(root / "nemo") == rung10.EXPECTED["binary"], "record binary differs")
    require((root / "namelist_cfg").read_bytes() == execution_cfg_bytes(), "record execution namelist differs")
    require((root / "namelist_cfg.deck").read_bytes() == rung7_cfg_bytes(), "record exact deck namelist differs")
    require((root / "namelist_ice_cfg").read_bytes() == rung8.SENTINEL.read_bytes(), "record ice sentinel differs")
    require(json.loads((root / "hierarchy_manifest.json").read_text()) == json.loads(MANIFEST.read_text()), "record manifest differs")
    frames = rung8.rung9.validate_frames(root, plant=plant)
    terminal = rung8.rung9.validate_terminal(root, plant=plant)
    month = rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    sha_inventory = rung10.validate_sha_inventory(root, plant=plant)
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung7-record-v1",
        "status": "PASS_RUNG7_RECORD",
        "claim_label": "independent",
        "rung": 7,
        "producer_commit": expect_commit,
        "deck_delta_from_rung8": {"namzdf.ln_zdfiwm": [True, False]},
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
        rung8.GateError,
        rung8.rung9.GateError,
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
