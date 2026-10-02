#!/usr/bin/env python3
"""Preflight and admit the ORCA2 hierarchy rung-4 oracle record."""

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
    nemo_testcase_l4_orca2_hier_decks_round6_gate as rung5_deck,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hier_decks_round7_gate as rung5_record,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (  # noqa: E402
    namelist_values,
)

HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round8_acquisition"
MANIFEST = ACQUISITION / "rung4_manifest.json"
RUNG5_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung5/record_absent_v2"
)
RUNG5_ADMISSION = RUNG5_RECORD.parent / "rung5_round8_admission.json"
COMPILED = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo"
)
NAMELIST_REF = COMPILED.parents[2] / "EXP00/namelist_ref"

RUNG4_CFG_SHA = "96796ce98339c744f2d488cc8bbc962023ca769877de82a1ee7ee7e579b0a14c"
EXECUTION_CFG_SHA = "0e84726416b62e1a4e15b94b4e529af4816a7053a58ef15996d53f45794bbf6b"
BINARY_SHA = "cee66aec4a9a9c2b85e755af0250c9b27cccef6ed7333b454a742af0692bc9ec"
RUNG5_ADMISSION_SHA = "9f5de467d92d2b2b7514f8907604bf008e61606cbf69ab16a61e94e0b8289b70"
STPRK3_SHA = "b9da36788bb5a3d8ccff03cae7c1123d1b3510c39a0cd86263fcc293da526422"
WRITER_SHA = "81304be81526980fb9f80549fdfd02c20df293b7a1a0725e94ef9b7e25c88c72"
SOURCE_SHA = {
    "traqsr": "e35de18655671e28f806c42663330efc552ab838d417438beec4142e6bfd4493",
    "stprk3_stg": "ec48b5d3cfc5637b0f3b4af49f3275735250078e0819b6ea48db31cae2848044",
    "trasbc": "906d5479e5e3a1dc9dd74a50a58ba395a0d007b31a1c51261ae0dde6acd66441",
    "sbcssm": "f0c78b59ea35a4987c0a12e782355e21ca03828b733345807a918804431f2fe0",
    "sbcmod": "f2ce40cd468c056e3b39ccaa74cdf7b76fd3d477ab6aaa08b0bf6d18e807c52b",
}
REFERENCE_SHA = "b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd"
SELECTORS = {
    "namsbc.ln_traqsr": (
        "   ln_traqsr   = .true.    !  Light penetration in the ocean"
        "            (T => fill namtra_qsr)",
        "   ln_traqsr   = .false.   !  Light penetration in the ocean"
        "            (T => fill namtra_qsr)",
    ),
    "namtra_qsr.ln_qsr_rgb": (
        "   ln_qsr_rgb  = .true.       !  RGB light penetration (Red-Green-Blue)",
        "   ln_qsr_rgb  = .false.      !  RGB light penetration (Red-Green-Blue)",
    ),
    "namtra_qsr.nn_chldta": (
        "   nn_chldta   =      1       !  RGB : Chl data (=1) or cst value (=0)",
        "   nn_chldta   =      0       !  RGB : Chl data (=1) or cst value (=0)",
    ),
}
PRECHECK_PLANTS = (
    "deck-extra",
    "missing-penetration-selector",
    "two-band-fallback",
    "build-pin",
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
    "runoff-group-unread",
    "active-runoff-print",
    "shortwave-consequence",
    "sha-inventory",
)
PLANTS = ("none", *PRECHECK_PLANTS, *RECORD_PLANTS)


class GateError(RuntimeError):
    """A frozen rung-4 hierarchy predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rung4_cfg_bytes(*, plant: str = "none") -> bytes:
    result = rung5_deck.rung5_cfg_bytes().decode()
    for name, (old, new) in SELECTORS.items():
        require(result.count(old) == 1, f"upper-rung selector changed: {name}")
        if not (plant == "missing-penetration-selector" and name == "namsbc.ln_traqsr"):
            result = result.replace(old, new)
    if plant == "two-band-fallback":
        anchor = "   ln_qsr_rgb  = .false.      !  RGB light penetration (Red-Green-Blue)"
        result = result.replace(
            anchor, anchor + "\n   ln_qsr_2bd  = .true.       !  2BD light penetration (two bands)"
        )
    if plant == "deck-extra":
        target = (
            "   ln_ssr      = .true.    !  Sea Surface Restoring on T and/or S"
            "       (T => fill namsbc_ssr)"
        )
        require(result.count(target) == 1, "deck-extra plant target changed")
        result = result.replace(target, target.replace(".true.", ".false."))
    return result.encode()


def execution_cfg_bytes() -> bytes:
    result = rung5_deck.execution_cfg_bytes().decode()
    for name, (old, new) in SELECTORS.items():
        require(result.count(old) == 1, f"execution selector changed: {name}")
        result = result.replace(old, new)
    return result.encode()


def _assignment_map(payload: bytes, root: Path) -> dict[str, str]:
    target = root / "namelist_cfg"
    target.write_bytes(payload)
    return namelist_values(target)


def _symbol_files(symbol: str) -> list[str]:
    return sorted(
        path.name for path in COMPILED.glob("*.f90") if symbol in path.read_text(errors="ignore")
    )


def preflight(*, plant: str = "none") -> dict[str, object]:
    require(plant in ("none", *PRECHECK_PLANTS), f"invalid preflight plant: {plant}")
    rung5_record.preflight()
    require(sha256(RUNG5_ADMISSION) == RUNG5_ADMISSION_SHA, "rung-5 admission digest changed")
    admission = json.loads(RUNG5_ADMISSION.read_text())
    require(admission.get("status") == "PASS_RUNG5_ABSENT_RECORD", "rung 5 is not admitted")
    require(admission.get("frames", {}).get("frames") == 480, "rung-5 frame evidence changed")

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
    expected_build = {
        "binary_sha256": BINARY_SHA,
        "compiled_stprk3_sha256": STPRK3_SHA,
        "compiled_writer_sha256": WRITER_SHA,
        "namelist_ref_sha256": REFERENCE_SHA,
        **{f"compiled_{name}_sha256": digest for name, digest in SOURCE_SHA.items()},
    }
    if plant == "build-pin":
        expected_build["compiled_traqsr_sha256"] = "0" * 64
    for name, expected in expected_build.items():
        require(manifest["build"].get(name) == expected, f"manifest build pin changed: {name}")
    require(sha256(RUNG5_RECORD / "nemo") == BINARY_SHA, "repaired recorder binary changed")
    require(sha256(NAMELIST_REF) == REFERENCE_SHA, "namelist_ref changed")
    for name, expected in SOURCE_SHA.items():
        require(sha256(COMPILED / f"{name}.f90") == expected, f"compiled source changed: {name}")

    cfg = rung4_cfg_bytes(plant=plant)
    require(hashlib.sha256(cfg).hexdigest() == RUNG4_CFG_SHA, "rung-4 exact deck changed")
    require(
        hashlib.sha256(execution_cfg_bytes()).hexdigest() == EXECUTION_CFG_SHA,
        "rung-4 execution deck changed",
    )
    require(manifest["namelists"]["namelist_cfg_sha256"] == RUNG4_CFG_SHA, "deck pin changed")
    require(
        manifest["namelists"]["execution_namelist_cfg_sha256"] == EXECUTION_CFG_SHA,
        "execution deck pin changed",
    )
    require(
        manifest["namelists"]["namelist_ice_cfg_sha256"]
        == sha256(RUNG5_RECORD / "namelist_ice_cfg"),
        "retained ice namelist changed",
    )

    with tempfile.TemporaryDirectory(prefix="orca2-rung4-deck-") as temporary:
        root = Path(temporary)
        (root / "upper").mkdir()
        (root / "lower").mkdir()
        upper = _assignment_map(rung5_deck.rung5_cfg_bytes(), root / "upper")
        lower = _assignment_map(cfg, root / "lower")
    changed = {
        name: [
            rung5_deck.rung6.rung10._token(upper[name]),
            rung5_deck.rung6.rung10._token(lower[name]),
        ]
        for name in sorted(set(upper) | set(lower))
        if upper.get(name) != lower.get(name)
    }
    require(
        changed
        == {
            "namsbc.ln_traqsr": [".true.", ".false."],
            "namtra_qsr.ln_qsr_rgb": [".true.", ".false."],
            "namtra_qsr.nn_chldta": ["1", "0"],
        },
        f"rung delta is not shortwave-only: {changed}",
    )
    upper_lines = rung5_deck.rung5_cfg_bytes().decode().splitlines()
    lower_lines = cfg.decode().splitlines()
    different = [
        i + 1 for i, pair in enumerate(zip(upper_lines, lower_lines)) if pair[0] != pair[1]
    ]
    require(
        len(upper_lines) == len(lower_lines) and different == [90, 137, 139],
        f"line delta changed: {different}",
    )

    reference = NAMELIST_REF.read_text()
    for pattern in (
        r"^\s*ln_traqsr\s*=\s*\.false\.",
        r"^\s*ln_qsr_rgb\s*=\s*\.false\.",
        r"^\s*ln_qsr_2bd\s*=\s*\.false\.",
        r"^\s*ln_qsr_5bd\s*=\s*\.false\.",
        r"^\s*ln_qsr_bio\s*=\s*\.false\.",
        r"^\s*nn_chldta\s*=\s*0\b",
    ):
        require(
            re.search(pattern, reference, re.MULTILINE) is not None,
            f"reference default changed: {pattern}",
        )
    source = {name: (COMPILED / f"{name}.f90").read_text() for name in SOURCE_SHA}
    needles = {
        "traqsr": (
            "IF( ioptio /= 1 )   CALL ctl_stop",
            "IF( ln_qsr_2bd                      )   nqsr = np_2BD",
        ),
        "stprk3_stg": ("IF( ln_traqsr  )   CALL tra_qsr",),
        "trasbc": (
            "IF( .NOT.ln_traqsr ) THEN",
            "qns(ji,jj) = qns(ji,jj) + qsr(ji,jj)",
            "qsr(ji,jj) = 0._wp",
        ),
        "sbcssm": ("IF( .NOT. ln_traqsr )   fraqsr_1lev(:,:) = 1._wp",),
        "sbcmod": ("Light penetration in temperature Eq.",),
    }
    for name, required in needles.items():
        require(
            all(needle in source[name] for needle in required),
            f"compiled shortwave branch changed: {name}",
        )
    selector_files = {
        "ln_traqsr": _symbol_files("ln_traqsr"),
        "ln_qsr_rgb": _symbol_files("ln_qsr_rgb"),
        "nn_chldta": _symbol_files("nn_chldta"),
    }
    require(
        selector_files
        == {
            "ln_traqsr": [
                "diahsb.f90",
                "nemogcm.f90",
                "sbcmod.f90",
                "sbcssm.f90",
                "stprk3_stg.f90",
                "traqsr.f90",
                "trasbc.f90",
                "zdfosm.f90",
            ],
            "ln_qsr_rgb": ["traqsr.f90"],
            "nn_chldta": ["traqsr.f90"],
        },
        f"compiled shortwave selector inventory changed: {selector_files}",
    )
    return {
        "status": "PREFLIGHT_PASS_RUNG4",
        "rung": 4,
        "claim_label": "independent",
        "upper_rung_admission": "PASS_RUNG5_ABSENT_RECORD",
        "assignment_delta": changed,
        "line_delta": [
            [90, SELECTORS["namsbc.ln_traqsr"][0], SELECTORS["namsbc.ln_traqsr"][1]],
            [137, SELECTORS["namtra_qsr.ln_qsr_rgb"][0], SELECTORS["namtra_qsr.ln_qsr_rgb"][1]],
            [139, SELECTORS["namtra_qsr.nn_chldta"][0], SELECTORS["namtra_qsr.nn_chldta"][1]],
        ],
        "compiled_consequences": {
            "shortwave_penetration": False,
            "two_band_fallback": False,
            "surface_qsr_moved_to_qns": True,
            "fraqsr_1lev": 1.0,
        },
        "compiled_selector_files": selector_files,
        "cpp_keys": list(rung5_deck.rung6.rung10.CPP_KEYS),
        "binary_sha256": BINARY_SHA,
        "ln_spc_dyn": "RETAINED_INERT_WITHOUT_key_agrif",
    }


def stage_deck(root: Path) -> None:
    expected = {
        "namelist_cfg": rung4_cfg_bytes(),
        "namelist_ice_cfg": (RUNG5_RECORD / "namelist_ice_cfg").read_bytes(),
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
    inherited_plant = plant if plant in set(rung5_deck.PLANTS) else "none"
    inherited = rung5_deck.validate_resolved(root, plant=inherited_plant)
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
    require(all(checks.values()), f"resolved rung-4 checks failed: {checks}")
    return {"status": "PASS_RUNG4_RESOLVED", "upper_rung": inherited, **checks}


def validate_record(root: Path, *, expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in ("none", *RECORD_PLANTS), f"invalid record plant: {plant}")
    preflight()
    require(
        (root / "producer_commit.txt").read_text().strip() == expect_commit,
        "producer commit differs",
    )
    require(sha256(root / "nemo") == BINARY_SHA, "record binary differs")
    require(
        (root / "namelist_cfg").read_bytes() == execution_cfg_bytes(), "execution namelist differs"
    )
    require((root / "namelist_cfg.deck").read_bytes() == rung4_cfg_bytes(), "exact deck differs")
    require(
        (root / "namelist_ice_cfg").read_bytes()
        == (RUNG5_RECORD / "namelist_ice_cfg").read_bytes(),
        "ice namelist differs",
    )
    require(
        json.loads((root / "hierarchy_manifest.json").read_text())
        == json.loads(MANIFEST.read_text()),
        "record manifest differs",
    )
    require(
        (root / "recorder_repair_manifest.json").read_bytes()
        == (RUNG5_RECORD / "recorder_repair_manifest.json").read_bytes(),
        "recorder repair manifest differs",
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
    inherited_plant = plant if plant in set(rung5_deck.PLANTS) else "none"
    frames = rung5_record.validate_frames(root, plant=frame_plant)
    terminal = rung5_deck.rung6.rung7.rung8.rung9.validate_terminal(root, plant=inherited_plant)
    month = rung5_deck.rung6.rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    inventory = rung5_deck.rung6.rung10.validate_sha_inventory(root, plant=inherited_plant)
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung4-record-v2",
        "status": "PASS_RUNG4_RECORD",
        "claim_label": "independent",
        "rung": 4,
        "producer_commit": expect_commit,
        "deck_delta_from_rung5": {
            "namsbc.ln_traqsr": [True, False],
            "namtra_qsr.ln_qsr_rgb": [True, False],
            "namtra_qsr.nn_chldta": [1, 0],
        },
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
        GateError,
        rung5_record.GateError,
        rung5_deck.GateError,
        rung5_deck.rung6.GateError,
        rung5_deck.rung6.rung7.GateError,
        rung5_deck.rung6.rung7.rung8.GateError,
        rung5_deck.rung6.rung7.rung8.rung9.GateError,
        rung5_deck.rung6.rung10.GateError,
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
