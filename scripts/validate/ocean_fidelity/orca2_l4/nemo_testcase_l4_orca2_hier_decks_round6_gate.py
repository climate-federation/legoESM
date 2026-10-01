#!/usr/bin/env python3
"""Preflight and admit the ORCA2 hierarchy rung-5 oracle record."""

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

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hier_decks_round5_gate as rung6,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (  # noqa: E402
    namelist_values,
)

HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round6_acquisition"
MANIFEST = ACQUISITION / "rung5_manifest.json"
RUNG6_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung6/record")
RUNG6_ADMISSION = RUNG6_RECORD.parent / "rung6_admission.json"
COMPILED = rung6.COMPILED

RUNG5_CFG_SHA = "f537e3d29a6e2472f89cc9a8d23ec70d18756e3ad112088e7256804698bc1d59"
EXECUTION_CFG_SHA = "53b45d1c98af7554e254e25e5b07acbab56cb09ed22896cf090415b371f293b8"
RUNOFF_SELECTOR = (
    "   ln_rnf      = .true.    !  runoffs                         "
    "          (T => fill namsbc_rnf)",
    "   ln_rnf      = .false.   !  runoffs                         "
    "          (T => fill namsbc_rnf)",
)
SOURCE_SHA = {
    "sbcmod": "f2ce40cd468c056e3b39ccaa74cdf7b76fd3d477ab6aaa08b0bf6d18e807c52b",
    "sbcrnf": "1604b3054264eab1dd4207b6f882d591ffa48ec7e01ca334d801d4f69d2e0ebe",
    "trasbc": "906d5479e5e3a1dc9dd74a50a58ba395a0d007b31a1c51261ae0dde6acd66441",
    "divhor": "45f670c175119eb1e88df17922e129c77e9d0121cbc6e100a1edefed4044f382",
    "stp2d": "4dcbbb0217d2ef3bcfb643a0a8f649e29249dbaa87f04a7d7f3d8e31a8d5c6bc",
    "sbcssr": "134ac06878e79ff57156cfc73b5d504f60adf569c761278cf1939d2c00eca820",
    "sbcfwb": "c1e200df05ef9092abf9e4b4329fa48f720711f1e755a87bf25ba1a8fce8258e",
    "sshwzv": "9d0225953304ecbd75bdfb4dd8d5556f34c33b4dcae0ddd591ab2634449829da",
    "traadv": "7b82662ce69dc6cdf1fe027fc82a99d7165aae0a0e89e71db3a829adfb3bacfa",
}
RUNOFF_SYMBOL_FILES = [
    "bdyvol.f90",
    "diahsb.f90",
    "diawri.f90",
    "divhor.f90",
    "oce_trc.f90",
    "p4zbc.f90",
    "sbc_oce.f90",
    "sbcclo.f90",
    "sbccpl.f90",
    "sbcfwb.f90",
    "sbcmod.f90",
    "sbcrnf.f90",
    "sbcssr.f90",
    "sshwzv.f90",
    "stp2d.f90",
    "traadv.f90",
    "trasbc.f90",
    "trc.f90",
    "trd_oce.f90",
    "trdpen.f90",
    "trdtra.f90",
    "wet_dry.f90",
    "zdfosm.f90",
    "zdfphy.f90",
]
RETAINED = {
    "namsbc.ln_blk": ".true.",
    "namsbc.ln_ssr": ".true.",
    "namsbc.nn_fwb": "2",
    "namsbc.ln_traqsr": ".true.",
    "namsbc_rnf.ln_rnf_mouth": ".false.",
    "namsbc_rnf.rn_hrnf": "15.e0",
    "namsbc_rnf.rn_avt_rnf": "1.e-3",
    "namsbc_rnf.rn_rfact": "1.e0",
    "namzdf.ln_zdfcst": ".true.",
    "namzdf.ln_zdftke": ".false.",
}
PLANTS = (
    "none",
    "deck-extra",
    "missing-selector",
    "retained-selector",
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
    "runoff-group-unread",
    "active-runoff-print",
    "sha-inventory",
)


class GateError(RuntimeError):
    """A frozen rung-5 hierarchy predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def rung5_cfg_bytes(*, plant: str = "none") -> bytes:
    result = rung6.rung6_cfg_bytes().decode()
    old, new = RUNOFF_SELECTOR
    require(result.count(old) == 1, "upper-rung runoff selector changed")
    if plant != "missing-selector":
        result = result.replace(old, new)
    if plant == "deck-extra":
        target = "   rn_rfact    =   1.e0    !  multiplicative factor for runoff"
        require(result.count(target) == 1, "deck-extra plant target changed")
        result = result.replace(target, target.replace("1.e0", "2.e0"))
    if plant == "retained-selector":
        target = (
            "   ln_ssr      = .true.    !  Sea Surface Restoring on T and/or S"
            "       (T => fill namsbc_ssr)"
        )
        require(result.count(target) == 1, "retained-selector plant target changed")
        result = result.replace(target, target.replace(".true.", ".false."))
    return result.encode()


def execution_cfg_bytes() -> bytes:
    result = rung6.execution_cfg_bytes().decode()
    old, new = RUNOFF_SELECTOR
    require(result.count(old) == 1, "upper-rung execution runoff selector changed")
    return result.replace(old, new).encode()


def _assignment_map(payload: bytes, root: Path) -> dict[str, str]:
    target = root / "namelist_cfg"
    target.write_bytes(payload)
    return namelist_values(target)


def _symbol_files(symbol: str) -> list[str]:
    return sorted(
        path.name for path in COMPILED.glob("*.f90") if symbol in path.read_text(errors="ignore")
    )


def preflight(*, plant: str = "none") -> dict[str, object]:
    require(
        plant in ("none", "deck-extra", "missing-selector", "retained-selector", "build-pin"),
        f"invalid preflight plant: {plant}",
    )
    rung6.preflight()
    admission = json.loads(RUNG6_ADMISSION.read_text())
    require(admission.get("status") == "PASS_RUNG6_RECORD", "rung-6 admission is not PASS")
    require(admission.get("frames", {}).get("frames") == 480, "rung-6 frame evidence changed")

    manifest = json.loads(MANIFEST.read_text())
    require(
        manifest["format"] == "nemo-testcase-l4-orca2-hierarchy-deck-v1", "manifest format changed"
    )
    require(manifest["rung"] == 5, "manifest rung changed")
    require(
        manifest["run"]
        == {
            "mpi_ranks": 2,
            "first_step": 1,
            "last_step": 240,
            "from_rest": True,
            "initial_state_output": 1,
        },
        "run protocol changed",
    )
    require(tuple(manifest["build"]["cpp_keys"]) == rung6.rung10.CPP_KEYS, "CPP keys changed")
    expected_build = {
        "binary_sha256": rung6.rung10.EXPECTED["binary"],
        "cpp_keys_sha256": rung6.rung10.EXPECTED["cpp"],
        "compiled_stprk3_sha256": rung6.rung10.EXPECTED["stprk3"],
        "compiled_writer_sha256": rung6.rung10.EXPECTED["writer"],
        **{f"compiled_{name}_sha256": digest for name, digest in SOURCE_SHA.items()},
    }
    if plant == "build-pin":
        expected_build["compiled_sbcmod_sha256"] = "0" * 64
    for name, expected in expected_build.items():
        require(manifest["build"].get(name) == expected, f"manifest build pin changed: {name}")
    for name, expected in SOURCE_SHA.items():
        require(
            rung6.rung10.sha256(COMPILED / f"{name}.f90") == expected,
            f"compiled source changed: {name}",
        )

    cfg = rung5_cfg_bytes(plant=plant)
    require(
        rung6.rung7.rung8.rung9.sha256_bytes(cfg) == RUNG5_CFG_SHA,
        "rung-5 ocean namelist digest/delta changed",
    )
    require(
        rung6.rung7.rung8.rung9.sha256_bytes(execution_cfg_bytes()) == EXECUTION_CFG_SHA,
        "execution namelist changed",
    )
    require(
        manifest["namelists"]["namelist_cfg_sha256"] == RUNG5_CFG_SHA, "manifest deck pin changed"
    )
    require(
        manifest["namelists"]["execution_namelist_cfg_sha256"] == EXECUTION_CFG_SHA,
        "manifest execution pin changed",
    )
    require(
        manifest["namelists"]["namelist_ice_cfg_sha256"]
        == rung6.rung7.rung8.rung9.RUNG10_ICE_CFG_SHA,
        "ice artifact changed",
    )

    with tempfile.TemporaryDirectory(prefix="orca2-rung5-deck-") as temporary:
        root = Path(temporary)
        (root / "upper").mkdir()
        (root / "lower").mkdir()
        upper = _assignment_map(rung6.rung6_cfg_bytes(), root / "upper")
        lower = _assignment_map(cfg, root / "lower")
    changed = {
        name: [rung6.rung10._token(upper[name]), rung6.rung10._token(lower[name])]
        for name in sorted(set(upper) | set(lower))
        if upper.get(name) != lower.get(name)
    }
    require(
        changed == {"namsbc.ln_rnf": [".true.", ".false."]},
        f"rung delta is not runoff only: {changed}",
    )
    for name, expected in RETAINED.items():
        require(
            rung6.rung10._token(lower[name]) == expected, f"retained assignment changed: {name}"
        )

    upper_lines = rung6.rung6_cfg_bytes().decode().splitlines()
    lower_lines = cfg.decode().splitlines()
    require(len(lower_lines) == len(upper_lines), "physical line count changed")
    different = [
        index + 1 for index, pair in enumerate(zip(upper_lines, lower_lines)) if pair[0] != pair[1]
    ]
    require(different == [92], f"physical-line delta changed: {different}")

    source = {name: (COMPILED / f"{name}.f90").read_text() for name in SOURCE_SHA}
    needles = {
        "sbcmod": (
            "ln_rnf   , nn_fwb",
            "CALL sbc_rnf_init( Kmm )",
            "IF( ln_rnf         )   CALL sbc_rnf( kt )",
        ),
        "sbcrnf": (
            "READ(numnam_cfg",
            "IF( .NOT. ln_rnf ) THEN",
            "ln_rnf_mouth  = .FALSE.",
            "RETURN",
        ),
        "trasbc": ("IF( ln_rnf ) THEN", "input of heat and salt due to river runoff"),
        "divhor": ("IF( ln_rnf )   CALL sbc_rnf_div",),
        "stp2d": ("IF( ln_rnf )   sshe_rhs(:,:) = sshe_rhs(:,:) - rnf(:,:)",),
        "sbcssr": ("IF( ln_rnf ) coefice",),
        "sbcfwb": ("IF( ln_rnf )               zemp",),
        "sshwzv": ("IF( ln_rnf )  zwght(:,:) = zwght(:,:) - rnf_b(:,:)        + rnf(:,:)",),
        "traadv": ("IF( ln_mus_ups .AND. .NOT. ln_rnf ) THEN",),
    }
    for name, required in needles.items():
        require(
            all(needle in source[name] for needle in required),
            f"compiled runoff branch changed: {name}",
        )
    selector_files = _symbol_files("ln_rnf")
    require(
        selector_files == RUNOFF_SYMBOL_FILES,
        f"compiled runoff symbol inventory changed: {selector_files}",
    )

    return {
        "status": "PREFLIGHT_PASS_RUNG5",
        "rung": 5,
        "claim_label": "independent",
        "upper_rung_admission": "PASS_RUNG6_RECORD",
        "assignment_delta": changed,
        "line_delta": [[92, RUNOFF_SELECTOR[0], RUNOFF_SELECTOR[1]]],
        "retained_assignments": RETAINED,
        "compiled_consequences": {
            "runoff": False,
            "runoff_namelist_read": True,
            "runoff_initializer_active_body": False,
            "runoff_surface_call": False,
            "runoff_tracer_source": False,
            "runoff_continuity_source": False,
            "runoff_external_mode_source": False,
            "ln_rnf_mouth": False,
        },
        "compiled_symbol_files": selector_files,
        "cpp_keys": list(rung6.rung10.CPP_KEYS),
        "binary_sha256": rung6.rung10.EXPECTED["binary"],
        "ln_spc_dyn": "RETAINED_INERT_WITHOUT_key_agrif",
    }


def stage_deck(root: Path) -> None:
    expected = {
        "namelist_cfg": rung5_cfg_bytes(),
        "namelist_ice_cfg": rung6.rung10.RUNG_ICE_CFG.read_bytes(),
        "manifest.json": MANIFEST.read_bytes(),
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in expected.items():
        target = root / name
        if target.exists():
            require(
                target.is_file() and target.read_bytes() == payload,
                f"existing deck artifact differs: {target}",
            )
        else:
            target.write_bytes(payload)
    rows = "".join(
        f"{rung6.rung7.rung8.rung9.sha256_bytes(payload)}  {name}\n"
        for name, payload in expected.items()
    )
    ledger = root / "SHA256SUMS"
    if ledger.exists():
        require(ledger.read_text() == rows, "existing deck SHA256SUMS differs")
    else:
        ledger.write_text(rows)


def validate_resolved(root: Path, *, plant: str = "none") -> dict[str, object]:
    inherited_plant = plant if plant in ("ice-sentinel-read", "tke-sentinel-read") else "none"
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
    return {"status": "PASS_RUNG5_RESOLVED", "upper_rung": inherited, **checks}


def validate_record(root: Path, *, expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant: {plant}")
    if plant in ("deck-extra", "missing-selector", "retained-selector", "build-pin"):
        preflight(plant=plant)
    else:
        preflight()
    require(
        (root / "producer_commit.txt").read_text().strip() == expect_commit,
        "producer commit differs",
    )
    require(
        rung6.rung10.sha256(root / "nemo") == rung6.rung10.EXPECTED["binary"],
        "record binary differs",
    )
    require(
        (root / "namelist_cfg").read_bytes() == execution_cfg_bytes(),
        "record execution namelist differs",
    )
    require(
        (root / "namelist_cfg.deck").read_bytes() == rung5_cfg_bytes(),
        "record exact deck namelist differs",
    )
    require(
        (root / "namelist_ice_cfg").read_bytes() == rung6.rung7.rung8.rung9.SENTINEL.read_bytes(),
        "record ice sentinel differs",
    )
    require(
        json.loads((root / "hierarchy_manifest.json").read_text())
        == json.loads(MANIFEST.read_text()),
        "record manifest differs",
    )
    frames = rung6.rung7.rung8.rung9.validate_frames(root, plant=plant)
    terminal = rung6.rung7.rung8.rung9.validate_terminal(root, plant=plant)
    month = rung6.rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    sha_inventory = rung6.rung10.validate_sha_inventory(root, plant=plant)
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung5-record-v1",
        "status": "PASS_RUNG5_RECORD",
        "claim_label": "independent",
        "rung": 5,
        "producer_commit": expect_commit,
        "deck_delta_from_rung6": {"namsbc.ln_rnf": [True, False]},
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
        rung6.GateError,
        rung6.rung7.GateError,
        rung6.rung7.rung8.GateError,
        rung6.rung7.rung8.rung9.GateError,
        rung6.rung10.GateError,
        rung6.rung10.surface.GateError,
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
