#!/usr/bin/env python3
"""Preflight and admit the ORCA2 hierarchy rung-2 oracle record."""

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
    nemo_testcase_l4_orca2_hier_decks_round9_gate as rung3,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (  # noqa: E402
    namelist_values,
)

HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round10_acquisition"
MANIFEST = ACQUISITION / "rung2_manifest.json"
RUNG3_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung3"
)
RUNG3_RECORD = RUNG3_ROOT / "record"
RUNG3_ADMISSION = RUNG3_ROOT / "rung3_admission.json"
COMPILED = rung3.COMPILED

RUNG3_ADMISSION_SHA = "5e4ce3e944b8561e46f472b0109006c435f5dfa36552bce1668c8e1eef39238e"
RUNG2_CFG_SHA = "fa8d0a34d6f3bce8cce4c98ae56fefa9d2a0f5e51e4ba582aa695362154d75b7"
EXECUTION_CFG_SHA = "cf42f174857eef17f3654518357e28f4c0f73f8e25e7c0a4b07e77813bfbf39a"
SOURCE_SHA = {
    "ldftra": "016119ca2225a0041fa5b5dc3f73b8d2b37dceb92eec6d99ed922e3659a33029",
    "traadv": "7b82662ce69dc6cdf1fe027fc82a99d7165aae0a0e89e71db3a829adfb3bacfa",
    "tramle": "c1f3d44e7757958f178f59ef4e0f98321a06dadbb48dfdaea609e61e4bd1febf",
}
CHANGES = {
    "namtra_eiv.ln_ldfeiv": (".true.", ".false."),
    "namtra_mle.ln_mle": (".true.", ".false."),
}
PRECHECK_PLANTS = ("deck-extra", "source-pin", "upper-admission")
OWN_RECORD_PLANTS = ("gm-mle-consequence",)
RECORD_PLANTS = (*rung3.RECORD_PLANTS, *OWN_RECORD_PLANTS)
PLANTS = ("none", *PRECHECK_PLANTS, *RECORD_PLANTS)


class GateError(RuntimeError):
    """A frozen rung-2 hierarchy predicate failed."""


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
        b"ln_mle      = .true.",
        b"ln_mle      = .false.",
    )
    result = _replace_once(
        result,
        b"ln_ldfeiv   = .true.",
        b"ln_ldfeiv   = .false.",
    )
    if plant == "deck-extra":
        result = _replace_once(result, b"ln_tradmp   =  .true.", b"ln_tradmp   =  .false.")
    return result


def rung2_cfg_bytes(*, plant: str = "none") -> bytes:
    return _render(rung3.rung3_cfg_bytes(), plant=plant)


def execution_cfg_bytes() -> bytes:
    return _render(rung3.execution_cfg_bytes())


def _assignment_map(payload: bytes, root: Path) -> dict[str, str]:
    target = root / "namelist_cfg"
    target.write_bytes(payload)
    return namelist_values(target)


def _token(value: str) -> str:
    return rung3._token(value)


def preflight(*, plant: str = "none") -> dict[str, object]:
    require(plant in ("none", *PRECHECK_PLANTS), f"invalid preflight plant: {plant}")
    rung3.preflight()
    admission_sha = sha256(RUNG3_ADMISSION)
    if plant == "upper-admission":
        admission_sha = "0" * 64
    require(admission_sha == RUNG3_ADMISSION_SHA, "rung-3 admission changed")
    admission = json.loads(RUNG3_ADMISSION.read_text())
    require(admission.get("status") == "PASS_RUNG3_RECORD", "rung 3 is not admitted")
    require(admission.get("frames", {}).get("frames") == 480, "rung-3 frames changed")

    manifest = json.loads(MANIFEST.read_text())
    require(manifest.get("rung") == 2, "manifest rung changed")
    require(
        manifest.get("upper_rung_admission_sha256") == RUNG3_ADMISSION_SHA,
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
        if plant == "source-pin" and name == "ldftra":
            actual = "0" * 64
        require(actual == expected, f"compiled source changed: {name}")
        require(
            manifest["build"].get(f"compiled_{name}_sha256") == expected,
            f"manifest source pin changed: {name}",
        )

    cfg = rung2_cfg_bytes(plant=plant)
    require(hashlib.sha256(cfg).hexdigest() == RUNG2_CFG_SHA, "rung-2 exact deck changed")
    require(
        hashlib.sha256(execution_cfg_bytes()).hexdigest() == EXECUTION_CFG_SHA,
        "rung-2 execution deck changed",
    )
    require(
        manifest["namelists"]["namelist_cfg_sha256"] == RUNG2_CFG_SHA,
        "manifest deck pin changed",
    )
    require(
        manifest["namelists"]["execution_namelist_cfg_sha256"] == EXECUTION_CFG_SHA,
        "manifest execution-deck pin changed",
    )

    with tempfile.TemporaryDirectory(prefix="orca2-rung2-deck-") as temporary:
        root = Path(temporary)
        (root / "upper").mkdir()
        (root / "lower").mkdir()
        upper = _assignment_map(rung3.rung3_cfg_bytes(), root / "upper")
        lower = _assignment_map(cfg, root / "lower")
    delta = {
        name: [_token(upper[name]), _token(lower[name])]
        for name in sorted(set(upper) | set(lower))
        if upper.get(name) != lower.get(name)
    }
    expected_delta = {name: [old, new] for name, (old, new) in CHANGES.items()}
    require(delta == expected_delta, f"rung delta is not GM/MLE-only: {delta}")

    source = {name: (COMPILED / f"{name}.f90").read_text() for name in SOURCE_SHA}
    needles = {
        "ldftra": (
            "IF( .NOT.ln_ldfeiv ) THEN",
            "eddy induced velocity param is NOT used",
            "ALLOCATE( aeiu(jpi,jpj,jpk), aeiv(jpi,jpj,jpk), STAT=ierr )",
        ),
        "traadv": (
            "IF( ln_ldfeiv .AND. .NOT. ln_traldf_triad ) THEN",
            "IF( ln_mle    )   THEN",
            "CALL tra_mle_trp( kt, pFu, pFv, pFw, Kmm )",
        ),
        "tramle": (
            "IF( ln_mle ) THEN",
            "Mixed Layer Eddy parametrisation NOT used",
            "ALLOCATE( rfu(jpi,jpj) , rfv(jpi,jpj) , STAT= ierr )",
        ),
    }
    for name, required in needles.items():
        require(
            all(needle in source[name] for needle in required),
            f"compiled GM/MLE branch changed: {name}",
        )
    return {
        "status": "PREFLIGHT_PASS_RUNG2",
        "rung": 2,
        "claim_label": "independent",
        "upper_rung_admission": "PASS_RUNG3_RECORD",
        "assignment_delta": delta,
        "compiled_consequences": {
            "gm_eddy_induced_velocity": False,
            "mixed_layer_eddies": False,
            "stage3_eiv_transport_called": False,
            "stage3_mle_transport_called": False,
        },
        "cpp_keys": list(rung3.rung4.rung5_deck.rung6.rung10.CPP_KEYS),
        "binary_sha256": rung3.rung4.BINARY_SHA,
        "ln_spc_dyn": "RETAINED_INERT_WITHOUT_key_agrif",
    }


def stage_deck(root: Path) -> None:
    expected = {
        "namelist_cfg": rung2_cfg_bytes(),
        "namelist_ice_cfg": (RUNG3_RECORD / "namelist_ice_cfg").read_bytes(),
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
    inherited_plant = plant if plant in set(rung3.RECORD_PLANTS) else "none"
    inherited = rung3.validate_resolved(root, plant=inherited_plant)
    ocean = (root / "ocean.output").read_text(errors="strict")
    checks = {
        "ln_ldfeiv_false": re.search(
            r"Eddy Induced Velocity \(eiv\) param\.\s+ln_ldfeiv\s*=\s*F\b", ocean
        )
        is not None,
        "eiv_not_used": "eddy induced velocity param is NOT used" in ocean,
        "eiv_active_print_absent": "use eddy induced velocity parametrization" not in ocean,
        "ln_mle_false": re.search(
            r"use mixed layer eddy .*ln_mle\s*=\s*F\b", ocean
        )
        is not None,
        "mle_not_used": "Mixed Layer Eddy parametrisation NOT used" in ocean,
        "mle_active_print_absent": "Mixed Layer Eddy induced transport added" not in ocean,
    }
    if plant == "gm-mle-consequence":
        checks["ln_mle_false"] = False
    require(all(checks.values()), f"resolved rung-2 checks failed: {checks}")
    return {"status": "PASS_RUNG2_RESOLVED", "upper_rung": inherited, **checks}


def validate_record(
    root: Path, *, expect_commit: str, plant: str = "none"
) -> dict[str, object]:
    require(plant in ("none", *RECORD_PLANTS), f"invalid record plant: {plant}")
    report = preflight()
    require(
        (root / "producer_commit.txt").read_text().strip() == expect_commit,
        "producer commit differs",
    )
    require(sha256(root / "nemo") == rung3.rung4.BINARY_SHA, "record binary differs")
    require(
        (root / "namelist_cfg").read_bytes() == execution_cfg_bytes(),
        "execution namelist differs",
    )
    require(
        (root / "namelist_cfg.deck").read_bytes() == rung2_cfg_bytes(),
        "exact deck differs",
    )
    require(
        json.loads((root / "hierarchy_manifest.json").read_text())
        == json.loads(MANIFEST.read_text()),
        "record manifest differs",
    )
    require(
        (root / "recorder_repair_manifest.json").read_bytes()
        == (RUNG3_RECORD / "recorder_repair_manifest.json").read_bytes(),
        "recorder repair manifest differs",
    )
    for name, expected in SOURCE_SHA.items():
        require(sha256(root / f"compiled_{name}.f90") == expected,
                f"record compiled source changed: {name}")

    frame_plant = plant if plant in {
        "field-name",
        "truncated",
        "frame-nonfinite",
        "absent-as-zero",
        "owner-on",
        "missing-frame",
    } else "none"
    inherited_plant = plant if plant in set(rung3.rung4.rung5_deck.PLANTS) else "none"
    frames = rung3.rung4.rung5_record.validate_frames(root, plant=frame_plant)
    terminal = rung3.rung4.rung5_deck.rung6.rung7.rung8.rung9.validate_terminal(
        root, plant=inherited_plant
    )
    month = rung3.rung4.rung5_deck.rung6.rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    zero_flux = rung3._validate_zero_flux(root, plant=plant)
    inventory = rung3.rung4.rung5_deck.rung6.rung10.validate_sha_inventory(
        root, plant=inherited_plant
    )
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung2-record-v2",
        "status": "PASS_RUNG2_RECORD",
        "claim_label": "independent",
        "rung": 2,
        "producer_commit": expect_commit,
        "deck_delta_from_rung3": report["assignment_delta"],
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
