#!/usr/bin/env python3
"""Preflight and admit the ORCA2 hierarchy rung-3 oracle record."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_hier_decks_round8_gate as rung4,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (  # noqa: E402
    namelist_values,
)

HERE = Path(__file__).resolve().parent
ACQUISITION = HERE / "nemo_testcase_l4_orca2_hier_decks_round9_acquisition"
MANIFEST = ACQUISITION / "rung3_manifest.json"
RUNG4_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung4"
)
RUNG4_RECORD = RUNG4_ROOT / "record"
RUNG4_ADMISSION = RUNG4_ROOT / "rung4_admission.json"
COMPILED = rung4.COMPILED
ZERO_FILE = "rung3_zero_flux.nc"

RUNG4_ADMISSION_SHA = "439bee8c4404cf86f6659fa5ee08a2e077c1ff92c87425ad321b0a590ac67bc7"
RUNG3_CFG_SHA = "6dbb38d77721e49dec63e452aca2c6e58cd159028a23eb5643016c7b40f6d6ab"
EXECUTION_CFG_SHA = "ef1894f7a55f4ac753523287daf9bd1d2c9910390e5e6169dbb9f75f320ca391"
SOURCE_SHA = {
    "sbcmod": "f2ce40cd468c056e3b39ccaa74cdf7b76fd3d477ab6aaa08b0bf6d18e807c52b",
    "sbcflx": "3e027b53d115bb03376fa0f2ac24fee6ba71f3ca9aa4053c1ce5cd24d8c03ee1",
    "usrdef_sbc": "46aa4264ec30256548597b42d91594c41cd32e049e3df9051e6e574cbb29d9f2",
    "sbcssr": "134ac06878e79ff57156cfc73b5d504f60adf569c761278cf1939d2c00eca820",
    "sbcfwb": "c1e200df05ef9092abf9e4b4329fa48f720711f1e755a87bf25ba1a8fce8258e",
}

CHANGES = {
    "namsbc.ln_blk": (".true.", ".false."),
    "namsbc.ln_ssr": (".true.", ".false."),
    "namsbc.nn_fwb": ("2", "0"),
    "namsbc_ssr.ln_sssr_bnd": (".true.", ".false."),
}
ADDED = {
    "namsbc.ln_usr": ".false.",
    "namsbc.ln_flx": ".true.",
    "namsbc.ln_abl": ".false.",
    "namsbc.ln_cpl": ".false.",
    "namsbc.ln_mixcpl": ".false.",
    "namsbc.ln_dm2dc": ".false.",
}
PRECHECK_PLANTS = (
    "deck-extra",
    "user-arm",
    "nonzero-file-selector",
    "source-pin",
)
OWN_RECORD_PLANTS = ("zero-flux-nonzero", "zero-flux-missing", "surface-consequence")
RECORD_PLANTS = (*rung4.RECORD_PLANTS, *OWN_RECORD_PLANTS)
PLANTS = ("none", *PRECHECK_PLANTS, *RECORD_PLANTS)


class GateError(RuntimeError):
    """A frozen rung-3 hierarchy predicate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_spans(text: str) -> dict[str, tuple[int, int]]:
    spans = {}
    pattern = re.compile(
        r"(?ms)^[ \t]*&([A-Za-z_]\w*)\b.*?^[ \t]*/[ \t]*(?:![^\n]*)?$"
    )
    for match in pattern.finditer(text):
        group = match.group(1).lower()
        require(group not in spans, f"duplicate namelist group {group}")
        spans[group] = match.span()
    return spans


def _replace(text: str, qualified: str, value: str) -> str:
    group, key = qualified.split(".", 1)
    start, end = _group_spans(text)[group]
    body = text[start:end]
    pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*=\s*)([^!\n]*)(.*)$", re.I)
    body, count = pattern.subn(rf"\g<1>{value:<12}\g<3>", body)
    require(count == 1, f"{qualified}: expected one assignment, found {count}")
    return text[:start] + body + text[end:]


def _add(text: str, qualified: str, value: str) -> str:
    group, key = qualified.split(".", 1)
    start, end = _group_spans(text)[group]
    body = text[start:end]
    require(re.search(rf"(?mi)^\s*{re.escape(key)}\s*=", body) is None,
            f"{qualified}: assignment already exists")
    slash = body.rfind("/")
    require(slash >= 0, f"{group}: missing terminator")
    body = body[:slash] + f"   {key:<14} = {value}\n" + body[slash:]
    return text[:start] + body + text[end:]


def _render(payload: bytes, *, plant: str = "none") -> bytes:
    text = payload.decode()
    for qualified, (_, value) in CHANGES.items():
        text = _replace(text, qualified, value)
    for qualified, value in ADDED.items():
        text = _add(text, qualified, value)
    spans = _group_spans(text)
    insert = text.find("\n", spans["namsbc"][1]) + 1
    require(insert > spans["namsbc"][0], "cannot place namsbc_flx")
    flux_name = (
        "not_zero_flux"
        if plant == "nonzero-file-selector"
        else ZERO_FILE.removesuffix(".nc")
    )
    flux = f"""!-----------------------------------------------------------------------
&namsbc_flx    ! hierarchy rung-3 exact-zero flux formulation (ln_flx=T)
!-----------------------------------------------------------------------
   cn_dir      = './'
   sn_utau     = '{flux_name}', -12., 'utau', .false., .true., 'yearly', '', '', ''
   sn_vtau     = '{flux_name}', -12., 'vtau', .false., .true., 'yearly', '', '', ''
   sn_qtot     = '{flux_name}', -12., 'qtot', .false., .true., 'yearly', '', '', ''
   sn_qsr      = '{flux_name}', -12., 'qsr',  .false., .true., 'yearly', '', '', ''
   sn_emp      = '{flux_name}', -12., 'emp',  .false., .true., 'yearly', '', '', ''
/
"""
    text = text[:insert] + flux + text[insert:]
    if plant == "deck-extra":
        text = _replace(text, "namsbc.nn_fsbc", "3")
    if plant == "user-arm":
        text = _replace(text, "namsbc.ln_usr", ".true.")
        text = _replace(text, "namsbc.ln_flx", ".false.")
    return text.encode()


def rung3_cfg_bytes(*, plant: str = "none") -> bytes:
    return _render(rung4.rung4_cfg_bytes(), plant=plant)


def execution_cfg_bytes() -> bytes:
    return _render(rung4.execution_cfg_bytes())


def _assignment_map(payload: bytes, root: Path) -> dict[str, str]:
    target = root / "namelist_cfg"
    target.write_bytes(payload)
    return namelist_values(target)


def _token(value: str) -> str:
    return rung4.rung5_deck.rung6.rung10._token(value)


def preflight(*, plant: str = "none") -> dict[str, object]:
    require(plant in ("none", *PRECHECK_PLANTS), f"invalid preflight plant: {plant}")
    rung4.preflight()
    require(sha256(RUNG4_ADMISSION) == RUNG4_ADMISSION_SHA, "rung-4 admission changed")
    admission = json.loads(RUNG4_ADMISSION.read_text())
    require(admission.get("status") == "PASS_RUNG4_RECORD", "rung 4 is not admitted")
    require(admission.get("frames", {}).get("frames") == 480, "rung-4 frames changed")

    manifest = json.loads(MANIFEST.read_text())
    require(manifest.get("rung") == 3, "manifest rung changed")
    require(manifest.get("upper_rung_admission_sha256") == RUNG4_ADMISSION_SHA,
            "manifest upper admission changed")
    require(manifest.get("run") == {
        "mpi_ranks": 2, "first_step": 1, "last_step": 240,
        "from_rest": True, "initial_state_output": 1,
    }, "run protocol changed")
    for name, expected in SOURCE_SHA.items():
        actual = sha256(COMPILED / f"{name}.f90")
        if plant == "source-pin" and name == "sbcmod":
            actual = "0" * 64
        require(actual == expected, f"compiled source changed: {name}")
        require(manifest["build"].get(f"compiled_{name}_sha256") == expected,
                f"manifest source pin changed: {name}")
    require(sha256(RUNG4_RECORD / "nemo") == rung4.BINARY_SHA, "binary changed")

    cfg = rung3_cfg_bytes(plant=plant)
    require(hashlib.sha256(cfg).hexdigest() == RUNG3_CFG_SHA, "rung-3 exact deck changed")
    require(hashlib.sha256(execution_cfg_bytes()).hexdigest() == EXECUTION_CFG_SHA,
            "rung-3 execution deck changed")
    require(manifest["namelists"]["namelist_cfg_sha256"] == RUNG3_CFG_SHA,
            "manifest deck pin changed")
    require(manifest["namelists"]["execution_namelist_cfg_sha256"] == EXECUTION_CFG_SHA,
            "manifest execution-deck pin changed")

    with tempfile.TemporaryDirectory(prefix="orca2-rung3-deck-") as temporary:
        root = Path(temporary)
        (root / "upper").mkdir()
        (root / "lower").mkdir()
        upper = _assignment_map(rung4.rung4_cfg_bytes(), root / "upper")
        lower = _assignment_map(cfg, root / "lower")
    delta = {
        name: [_token(upper[name]) if name in upper else "ABSENT",
               _token(lower[name]) if name in lower else "ABSENT"]
        for name in sorted(set(upper) | set(lower)) if upper.get(name) != lower.get(name)
    }
    expected_delta = {
        **{name: [old, new] for name, (old, new) in CHANGES.items()},
        **{name: ["ABSENT", value] for name, value in ADDED.items()},
        "namsbc_flx.cn_dir": ["ABSENT", "'./'"],
        **{
            f"namsbc_flx.sn_{name}": [
                "ABSENT",
                f"'{ZERO_FILE.removesuffix('.nc')}', -12., '{name}', "
                f"{'.false., .true.' if name not in ('qsr', 'emp') else ' .false., .true.'}, "
                "'yearly', '', '', ''",
            ]
            for name in ("utau", "vtau", "qtot", "qsr", "emp")
        },
    }
    require(delta == expected_delta, f"rung delta is not surface-module-only: {delta}")

    source = {name: (COMPILED / f"{name}.f90").read_text() for name in SOURCE_SHA}
    needles = {
        "sbcmod": (
            "IF( icpt /= 1 )    CALL ctl_stop( 'sbc_init : choose ONE and only ONE sbc option' )",
            "CASE( jp_usr     )   ;   CALL usrdef_sbc_oce",
            "CASE( jp_flx     )   ;   CALL sbc_flx",
            "IF( ln_ssr         )   CALL sbc_ssr",
            "IF( nn_fwb    /= 0 )   CALL sbc_fwb",
        ),
        "sbcflx": (
            "qns (ji,jj) = ( sf(jp_qtot)%fnow",
            "emp (ji,jj) =   sf(jp_emp )%fnow",
        ),
        "usrdef_sbc": (
            "qsr (ji,jj) =  230 * COS",
            "utau(ji,jj) = - ztaun * SIN",
        ),
        "sbcssr": ("IF( ln_sssr_bnd )",),
        "sbcfwb": ("SUBROUTINE sbc_fwb",),
    }
    for name, required in needles.items():
        require(all(needle in source[name] for needle in required),
                f"compiled surface branch changed: {name}")
    return {
        "status": "PREFLIGHT_PASS_RUNG3",
        "rung": 3,
        "claim_label": "independent",
        "upper_rung_admission": "PASS_RUNG4_RECORD",
        "assignment_delta": delta,
        "compiled_consequences": {
            "surface_formulation": "flx",
            "usr_gyre_forcing_selected": False,
            "bulk_forcing_selected": False,
            "sea_surface_restoring": False,
            "freshwater_budget": 0,
            "zero_flux_fields": ["utau", "vtau", "qtot", "qsr", "emp"],
        },
        "zero_file_schema_source": "main-lane round 82",
        "cpp_keys": list(rung4.rung5_deck.rung6.rung10.CPP_KEYS),
        "binary_sha256": rung4.BINARY_SHA,
        "ln_spc_dyn": "RETAINED_INERT_WITHOUT_key_agrif",
    }


def stage_deck(root: Path) -> None:
    expected = {
        "namelist_cfg": rung3_cfg_bytes(),
        "namelist_ice_cfg": (RUNG4_RECORD / "namelist_ice_cfg").read_bytes(),
        "manifest.json": MANIFEST.read_bytes(),
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in expected.items():
        target = root / name
        if target.exists():
            require(target.is_file() and target.read_bytes() == payload,
                    f"existing deck differs: {target}")
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


def _validate_zero_flux(root: Path, *, plant: str) -> dict[str, object]:
    path = root / ZERO_FILE
    if plant == "zero-flux-missing":
        path = root / "missing_zero_flux.nc"
    require(path.is_file(), f"missing exact-zero flux file: {path}")
    fields = {}
    with Dataset(path) as dataset:
        require(len(dataset.dimensions["x"]) == 180 and len(dataset.dimensions["y"]) == 148,
                "zero-flux grid is not ORCA2 180x148")
        for name in ("utau", "vtau", "qtot", "qsr", "emp"):
            require(name in dataset.variables, f"zero-flux variable missing: {name}")
            variable = dataset[name]
            variable.set_auto_maskandscale(False)
            values = np.asarray(variable[:]).copy()
            if plant == "zero-flux-nonzero" and name == "utau":
                values.reshape(-1)[0] = np.nextafter(0.0, 1.0)
            require(variable.dtype == np.dtype("float64"), f"{name} is not fp64")
            require(np.array_equal(values, np.zeros_like(values)), f"{name} is not exactly zero")
            fields[name] = {"shape": list(values.shape), "dtype": str(variable.dtype)}
    entries = {}
    for line in (root / "input_files.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name] = digest
    require(entries.get(ZERO_FILE) == sha256(root / ZERO_FILE),
            "zero-flux file is absent or stale in input manifest")
    return {"sha256": sha256(root / ZERO_FILE), "fields": fields}


def validate_resolved(root: Path, *, plant: str = "none") -> dict[str, object]:
    inherited_plant = plant if plant in set(rung4.RECORD_PLANTS) else "none"
    inherited = rung4.validate_resolved(root, plant=inherited_plant)
    ocean = (root / "ocean.output").read_text(errors="strict")
    checks = {
        "ln_usr_false": re.search(
            r"user defined formulation\s+ln_usr\s+=\s+F\b", ocean
        ) is not None,
        "ln_flx_true": re.search(r"flux\s+formulation\s+ln_flx\s+=\s+T\b", ocean) is not None,
        "ln_blk_false": re.search(r"bulk\s+formulation\s+ln_blk\s+=\s+F\b", ocean) is not None,
        "ln_ssr_false": re.search(r"Sea Surface Restoring.*ln_ssr\s+=\s+F\b", ocean) is not None,
        "nn_fwb_zero": re.search(r"FreshWater Budget control.*nn_fwb\s+=\s+0\b", ocean) is not None,
        "flux_dispatch": "==>>>   flux formulation" in ocean,
        "bulk_initializer_absent": "sbc_blk_init :" not in ocean,
        "restoring_initializer_absent": "Namelist namsbc_ssr" not in ocean,
        "freshwater_budget_absent": "sbc_fwb :" not in ocean,
        "gyre_usr_forcing_absent": "usrdef_sbc_oce :" not in ocean,
    }
    if plant == "surface-consequence":
        checks["ln_flx_true"] = False
    require(all(checks.values()), f"resolved rung-3 checks failed: {checks}")
    return {"status": "PASS_RUNG3_RESOLVED", "upper_rung": inherited, **checks}


def validate_record(root: Path, *, expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in ("none", *RECORD_PLANTS), f"invalid record plant: {plant}")
    preflight()
    require((root / "producer_commit.txt").read_text().strip() == expect_commit,
            "producer commit differs")
    require(sha256(root / "nemo") == rung4.BINARY_SHA, "record binary differs")
    require((root / "namelist_cfg").read_bytes() == execution_cfg_bytes(),
            "execution namelist differs")
    require((root / "namelist_cfg.deck").read_bytes() == rung3_cfg_bytes(),
            "exact deck differs")
    require(json.loads((root / "hierarchy_manifest.json").read_text())
            == json.loads(MANIFEST.read_text()), "record manifest differs")
    require((root / "recorder_repair_manifest.json").read_bytes()
            == (RUNG4_RECORD / "recorder_repair_manifest.json").read_bytes(),
            "recorder repair manifest differs")
    frame_plant = plant if plant in {
        "field-name", "truncated", "frame-nonfinite", "absent-as-zero",
        "owner-on", "missing-frame",
    } else "none"
    inherited_plant = plant if plant in set(rung4.rung5_deck.PLANTS) else "none"
    frames = rung4.rung5_record.validate_frames(root, plant=frame_plant)
    terminal = rung4.rung5_deck.rung6.rung7.rung8.rung9.validate_terminal(
        root, plant=inherited_plant
    )
    month = rung4.rung5_deck.rung6.rung10.validate_month_products(root)
    resolved = validate_resolved(root, plant=plant)
    zero_flux = _validate_zero_flux(root, plant=plant)
    inventory = rung4.rung5_deck.rung6.rung10.validate_sha_inventory(
        root, plant=inherited_plant
    )
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung3-record-v2",
        "status": "PASS_RUNG3_RECORD",
        "claim_label": "independent",
        "rung": 3,
        "producer_commit": expect_commit,
        "deck_delta_from_rung4": preflight()["assignment_delta"],
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
            require(args.record is not None and args.expect_commit,
                    "record and expected commit are required")
            report = validate_record(args.record, expect_commit=args.expect_commit,
                                     plant=args.plant)
        report["worktree"] = worktree_stamp()
    except (
        RuntimeError,
        GateError,
        rung4.GateError,
        rung4.rung5_record.GateError,
        rung4.rung5_deck.GateError,
        rung4.rung5_deck.rung6.GateError,
        rung4.rung5_deck.rung6.rung7.GateError,
        rung4.rung5_deck.rung6.rung7.rung8.GateError,
        rung4.rung5_deck.rung6.rung7.rung8.rung9.GateError,
        rung4.rung5_deck.rung6.rung10.GateError,
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
