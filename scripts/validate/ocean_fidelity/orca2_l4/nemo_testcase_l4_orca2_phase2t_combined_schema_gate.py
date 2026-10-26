#!/usr/bin/env python3
"""Walk all 100 records in the combined Phase-2s ORCA2 oracle root.

The frozen Phase-1 schemas own the original 90 streams.  Later acquisitions
own the ten named extensions below.  This composition is deliberately explicit:
raw twin identity is not a substitute for parsing every stream to exact EOF.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_een_discriminator_gate as een,
    nemo_testcase_l4_orca2_o1_acquisition_gate as o1,
    nemo_testcase_l4_orca2_phase2o_bbl_gate as bbl,
    nemo_testcase_l4_orca2_phase2b_exchange_gate as surface,
    nemo_testcase_l4_orca2_phase2s_zdf_acquisition_gate as zdf,
    nemo_testcase_l4_orca2_wzv_gate as wzv,
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def validate(root: Path, mesh: Path, control: Path, plants: bool) -> dict:
    expected_base = phase1.expected_inventory()
    extensions = {
        bbl.NAME,
        *een.NEW_STREAMS,
        surface.RECORD,
        o1.RECORD,
        wzv.RECORD,
        *zdf.ZDF_NAMES,
    }
    actual = {path.name for path in root.glob("oracle_*.bin")}
    require(len(expected_base) == 90 and len(extensions) == 10,
            "internal 90+10 schema partition changed")
    require(actual == expected_base | extensions,
            f"100-stream inventory mismatch: missing={sorted((expected_base | extensions)-actual)} "
            f"extra={sorted(actual-(expected_base | extensions))}")

    with tempfile.TemporaryDirectory(prefix="orca2-p2t-base90-") as td:
        base = Path(td)
        for name in expected_base:
            os.symlink(root / name, base / name)
        base_rows = phase1.validate_records(base)

        base_plants = {}
        if plants:
            manifest = base / "manifest.sha256"
            manifest.write_text("".join(
                f"{phase1.sha256(root / name)}  {name}\n"
                for name in sorted(expected_base)
            ))
            base_plants = phase1.planted_controls(
                root, manifest, control, root)

    extension_rows = {
        surface.RECORD: surface.validate_surface(
            root / surface.RECORD, icebergs_off=True),
        o1.RECORD: o1.read_o1(
            root / o1.RECORD, require_canonical_undefined=True),
        wzv.RECORD: wzv.read_record(root / wzv.RECORD)["header"],
        bbl.NAME: bbl.read_bbl(root / bbl.NAME, mesh),
        zdf.ZDF_NAMES[0]: zdf.read_zdf_v2(
            root / zdf.ZDF_NAMES[0], mesh, require_shear=False),
        zdf.ZDF_NAMES[1]: zdf.read_zdf_v2(
            root / zdf.ZDF_NAMES[1], mesh, require_shear=True),
        "een": een.validate_new(root),
    }
    extension_plants = {}
    if plants:
        extension_plants = {
            "surface": surface.planted_controls(
                root / surface.RECORD, icebergs_off=True),
            "o1": o1.planted_controls(
                root / o1.RECORD, require_canonical_undefined=True),
        }
    return {
        "status": "PASS",
        "inventory": {"base": 90, "extensions": 10, "total": 100},
        "base_exact_eof_streams": len(base_rows["records"]),
        "base": base_rows,
        "extensions": extension_rows,
        "plants": {"base": base_plants, **extension_plants},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--plants", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.root, args.mesh, args.control, args.plants)
    except (GateError, phase1.GateError, surface.GateError, o1.GateError,
            wzv.GateError, bbl.GateError, zdf.GateError, een.GateError,
            OSError, UnicodeError, ValueError) as error:
        print(f"FAIL: {error}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
