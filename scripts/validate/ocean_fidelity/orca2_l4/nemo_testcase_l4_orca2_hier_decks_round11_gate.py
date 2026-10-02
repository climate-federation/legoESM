#!/usr/bin/env python3
"""Preflight and admit the ORCA2 hierarchy rung-1 oracle record."""

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
    nemo_testcase_l4_orca2_hier_decks_round10_gate as rung2,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (  # noqa: E402
    namelist_values,
)

HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round11_acquisition"
MANIFEST = ACQUISITION / "rung1_manifest.json"
RUNG2_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung2"
)
RUNG2_RECORD = RUNG2_ROOT / "record"
RUNG2_ADMISSION = RUNG2_ROOT / "rung2_admission.json"
MAIN_RUNG0_CFG = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round90/"
    "acquisition/orca2_rung0_entry_stage_runoff_guarded_10step_np2/namelist_cfg"
)
COMPILED = rung2.COMPILED

RUNG2_ADMISSION_SHA = "b1553d791db7df6ade5d9f3da06790e789a21b0be2ab4758b88e27f21509003e"
MAIN_RUNG0_CFG_SHA = "d25c69958aeb7d4dffeeab6b08c89f6b314dd7c6d130ed94acfee7cb90643c2c"
RUNG1_CFG_SHA = "44a15a9d7c8e1b77bebf8704d6455edc83ba7ebea93f358d8bbe093d5e1f0670"
EXECUTION_CFG_SHA = "f078d39c37ba5c94603c9bc5b12d28ee64cba03aa465debb670afccfa35a3002"
SOURCE_SHA = {
    "nemogcm": "3e0e832dab3e0e045ebc4ea3521399400b8efd4bcb4b755f9e157fdd0b469700",
    "trabbl": "0d899c0aaf58a33e49e5480629150c527b1f787f5e4d35514d3ab49576d38a09",
    "trabbc": "65e0d2f111f675d08e1224f4336bd65b0f0911c151ea2ae8b306b66937581ac6",
    "stprk3_stg": "ec48b5d3cfc5637b0f3b4af49f3275735250078e0819b6ea48db31cae2848044",
}
CHANGES = {
    "nambbc.ln_trabbc": (".true.", ".false."),
    "nambbl.ln_trabbl": (".true.", ".false."),
}
PRECHECK_PLANTS = (
    "deck-extra",
    "source-pin",
    "upper-admission",
    "main-rung0-pin",
)
OWN_RECORD_PLANTS = ("bbl-bbc-consequence",)
RECORD_PLANTS = (*rung2.RECORD_PLANTS, *OWN_RECORD_PLANTS)
PLANTS = ("none", *PRECHECK_PLANTS, *RECORD_PLANTS)


class GateError(RuntimeError):
    """A frozen rung-1 hierarchy predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_once(payload: bytes, old: bytes, new: bytes) -> bytes:
    require(payload.count(old) == 1, f"expected one selector line: {old.decode()}")
    return payload.replace(old, new)


def _render(payload: bytes, *, plant: str = "none") -> bytes:
    result = _replace_once(
        payload,
        b"ln_trabbc   = .true.",
        b"ln_trabbc   = .false.",
    )
    result = _replace_once(
        result,
        b"ln_trabbl   = .true.",
        b"ln_trabbl   = .false.",
    )
    if plant == "deck-extra":
        result = _replace_once(
            result,
            b"ln_tradmp   =  .true.",
            b"ln_tradmp   =  .false.",
        )
    return result


def rung1_cfg_bytes(*, plant: str = "none") -> bytes:
    return _render(rung2.rung2_cfg_bytes(), plant=plant)


def execution_cfg_bytes() -> bytes:
    return _render(rung2.execution_cfg_bytes())


def _assignment_map(payload: bytes, root: Path) -> dict[str, str]:
    target = root / "namelist_cfg"
    target.write_bytes(payload)
    return namelist_values(target)


def _token(value: str) -> str:
    return rung2._token(value)


def _semantic_token(value: str | None) -> str | None:
    """Remove namelist comments without treating exclamation marks in strings."""
    if value is None:
        return None
    quote: str | None = None
    result: list[str] = []
    for character in value:
        if character in ("'", '"'):
            quote = None if quote == character else character if quote is None else quote
        if character == "!" and quote is None:
            break
        if not character.isspace():
            result.append(character.lower())
    return "".join(result)


def _main_rung0_diff(rung1: dict[str, str], rung0: dict[str, str]) -> dict[str, list[str | None]]:
    return {
        name: [_semantic_token(rung1.get(name)), _semantic_token(rung0.get(name))]
        for name in sorted(set(rung1) | set(rung0))
        if _semantic_token(rung1.get(name)) != _semantic_token(rung0.get(name))
    }


def preflight(*, plant: str = "none") -> dict[str, object]:
    require(plant in ("none", *PRECHECK_PLANTS), f"invalid preflight plant: {plant}")
    rung2.preflight()
    admission_sha = sha256(RUNG2_ADMISSION)
    if plant == "upper-admission":
        admission_sha = "0" * 64
    require(admission_sha == RUNG2_ADMISSION_SHA, "rung-2 admission changed")
    admission = json.loads(RUNG2_ADMISSION.read_text())
    require(admission.get("status") == "PASS_RUNG2_RECORD", "rung 2 is not admitted")
    require(admission.get("frames", {}).get("frames") == 480, "rung-2 frames changed")

    manifest = json.loads(MANIFEST.read_text())
    require(manifest.get("rung") == 1, "manifest rung changed")
    require(
        manifest.get("upper_rung_admission_sha256") == RUNG2_ADMISSION_SHA,
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
    for name, expected in SOURCE_SHA.items():
        actual = sha256(COMPILED / f"{name}.f90")
        if plant == "source-pin" and name == "trabbl":
            actual = "0" * 64
        require(actual == expected, f"compiled source changed: {name}")
        require(
            manifest["build"].get(f"compiled_{name}_sha256") == expected,
            f"manifest source pin changed: {name}",
        )

    cfg = rung1_cfg_bytes(plant=plant)
    require(hashlib.sha256(cfg).hexdigest() == RUNG1_CFG_SHA, "rung-1 exact deck changed")
    require(
        hashlib.sha256(execution_cfg_bytes()).hexdigest() == EXECUTION_CFG_SHA,
        "rung-1 execution deck changed",
    )
    require(
        manifest["namelists"]["namelist_cfg_sha256"] == RUNG1_CFG_SHA,
        "manifest deck pin changed",
    )
    require(
        manifest["namelists"]["execution_namelist_cfg_sha256"] == EXECUTION_CFG_SHA,
        "manifest execution-deck pin changed",
    )

    main_rung0_sha = sha256(MAIN_RUNG0_CFG)
    if plant == "main-rung0-pin":
        main_rung0_sha = "0" * 64
    require(main_rung0_sha == MAIN_RUNG0_CFG_SHA, "main-lane rung-0 deck changed")

    with tempfile.TemporaryDirectory(prefix="orca2-rung1-deck-") as temporary:
        root = Path(temporary)
        (root / "upper").mkdir()
        (root / "lower").mkdir()
        upper = _assignment_map(rung2.rung2_cfg_bytes(), root / "upper")
        lower = _assignment_map(cfg, root / "lower")
        main_rung0 = namelist_values(MAIN_RUNG0_CFG)
    delta = {
        name: [_token(upper[name]), _token(lower[name])]
        for name in sorted(set(upper) | set(lower))
        if upper.get(name) != lower.get(name)
    }
    expected_delta = {name: [old, new] for name, (old, new) in CHANGES.items()}
    require(delta == expected_delta, f"rung delta is not BBL/geothermal-only: {delta}")
    rung0_diff = _main_rung0_diff(lower, main_rung0)
    require(
        rung0_diff.get("namtra_dmp.ln_tradmp") == [".true.", ".false."],
        "main-lane rung-0 tracer-damping boundary changed",
    )
    require(
        rung0_diff.get("namtsd.ln_tsd_dmp") == [".true.", ".false."],
        "main-lane rung-0 T/S-damping boundary changed",
    )

    source = {name: (COMPILED / f"{name}.f90").read_text() for name in SOURCE_SHA}
    needles = {
        "nemogcm": (
            "CALL tra_bbc_init      ! bottom heat flux",
            "CALL tra_bbl_init      ! advective (and/or diffusive) bottom boundary layer scheme",
        ),
        "trabbl": (
            "IF( .NOT.ln_trabbl )   RETURN",
            "IF( tra_bbl_alloc() /= 0 )",
        ),
        "trabbc": (
            "IF( ln_trabbc ) THEN",
            "ALLOCATE( qgh_trd0(jpi,jpj) )",
            "==>>>   no geothermal heat flux",
        ),
        "stprk3_stg": (
            "IF( kstg == 3 .AND. ln_trabbl )   CALL bbl",
            "IF( ln_trabbc  )   CALL tra_bbc",
            "IF( ln_trabbl  )   CALL tra_bbl",
        ),
    }
    for name, required in needles.items():
        require(
            all(needle in source[name] for needle in required),
            f"compiled BBL/geothermal branch changed: {name}",
        )
    return {
        "status": "PREFLIGHT_PASS_RUNG1",
        "rung": 1,
        "claim_label": "independent",
        "upper_rung_admission": "PASS_RUNG2_RECORD",
        "assignment_delta": delta,
        "compiled_consequences": {
            "bottom_boundary_layer": False,
            "geothermal_heating": False,
            "stage3_bbl_coefficient_call": False,
            "stage3_bbl_tracer_call": False,
            "stage3_geothermal_tracer_call": False,
        },
        "main_rung0_comparison": {
            "source": str(MAIN_RUNG0_CFG),
            "sha256": MAIN_RUNG0_CFG_SHA,
            "semantic_differences": rung0_diff,
        },
        "cpp_keys": list(rung2.rung3.rung4.rung5_deck.rung6.rung10.CPP_KEYS),
        "binary_sha256": rung2.rung3.rung4.BINARY_SHA,
        "ln_spc_dyn": "RETAINED_INERT_WITHOUT_key_agrif",
    }


def stage_deck(root: Path) -> None:
    expected = {
        "namelist_cfg": rung1_cfg_bytes(),
        "namelist_ice_cfg": (RUNG2_RECORD / "namelist_ice_cfg").read_bytes(),
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
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in expected.items()
    )
    ledger = root / "SHA256SUMS"
    if ledger.exists():
        require(ledger.read_text() == rows, "deck SHA256SUMS changed")
    else:
        ledger.write_text(rows)


def validate_resolved(root: Path, *, plant: str = "none") -> dict[str, object]:
    inherited_plant = plant if plant in set(rung2.RECORD_PLANTS) else "none"
    inherited = rung2.validate_resolved(root, plant=inherited_plant)
    ocean = (root / "ocean.output").read_text(errors="strict")
    checks = {
        "ln_trabbc_false": re.search(
            r"geothermal heating at ocean bottom\s+ln_trabbc\s*=\s*F\b", ocean
        )
        is not None,
        "geothermal_not_used": "==>>>   no geothermal heat flux" in ocean,
        "geothermal_file_unread": "read heatflow" not in ocean,
        "ln_trabbl_false": re.search(
            r"bottom boundary layer flag\s+ln_trabbl\s*=\s*F\b", ocean
        )
        is not None,
        "bbl_active_prints_absent": "diffusive bbl (=1)" not in ocean
        and "advective bbl (=1/2)" not in ocean,
    }
    if plant == "bbl-bbc-consequence":
        checks["ln_trabbl_false"] = False
    require(all(checks.values()), f"resolved rung-1 checks failed: {checks}")
    return {"status": "PASS_RUNG1_RESOLVED", "upper_rung": inherited, **checks}


def validate_record(
    root: Path, *, expect_commit: str, plant: str = "none"
) -> dict[str, object]:
    require(plant in ("none", *RECORD_PLANTS), f"invalid record plant: {plant}")
    report = preflight()
    require(
        (root / "producer_commit.txt").read_text().strip() == expect_commit,
        "producer commit differs",
    )
    require(sha256(root / "nemo") == rung2.rung3.rung4.BINARY_SHA, "record binary differs")
    require(
        (root / "namelist_cfg").read_bytes() == execution_cfg_bytes(),
        "execution namelist differs",
    )
    require(
        (root / "namelist_cfg.deck").read_bytes() == rung1_cfg_bytes(),
        "exact deck differs",
    )
    require(
        json.loads((root / "hierarchy_manifest.json").read_text())
        == json.loads(MANIFEST.read_text()),
        "record manifest differs",
    )
    require(
        (root / "recorder_repair_manifest.json").read_bytes()
        == (RUNG2_RECORD / "recorder_repair_manifest.json").read_bytes(),
        "recorder repair manifest differs",
    )
    for name, expected in SOURCE_SHA.items():
        require(
            sha256(root / f"compiled_{name}.f90") == expected,
            f"record compiled source changed: {name}",
        )

    frame_plant = plant if plant in {
        "field-name",
        "truncated",
        "frame-nonfinite",
        "absent-as-zero",
        "owner-on",
        "missing-frame",
    } else "none"
    inherited_plant = plant if plant in set(rung2.rung3.rung4.rung5_deck.PLANTS) else "none"
    frames = rung2.rung3.rung4.rung5_record.validate_frames(root, plant=frame_plant)
    terminal = rung2.rung3.rung4.rung5_deck.rung6.rung7.rung8.rung9.validate_terminal(
        root, plant=inherited_plant
    )
    month = rung2.rung3.rung4.rung5_deck.rung6.rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    zero_flux = rung2.rung3._validate_zero_flux(root, plant=plant)
    inventory = rung2.rung3.rung4.rung5_deck.rung6.rung10.validate_sha_inventory(
        root, plant=inherited_plant
    )
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung1-record-v2",
        "status": "PASS_RUNG1_RECORD",
        "claim_label": "independent",
        "rung": 1,
        "producer_commit": expect_commit,
        "deck_delta_from_rung2": report["assignment_delta"],
        "main_rung0_comparison": report["main_rung0_comparison"],
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
    except (RuntimeError, KeyError, OSError, TypeError, UnicodeError, ValueError) as error:
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
