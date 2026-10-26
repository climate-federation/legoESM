#!/usr/bin/env python3
"""Fail-closed gate for round-87 rung-0 surface-field absence records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class GateError(RuntimeError):
    """The round-87 writer or record is not admissible."""


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "nemo_testcase_l4_orca2_round87_rung0_frames"
PATCH = ARTIFACTS / "stprk3_round87_surface_absent.patch"
R84_ARTIFACTS = HERE / "nemo_testcase_l4_orca2_round84_rung0_frames"
R84_PATCH = R84_ARTIFACTS / "stprk3_round84.patch"
R84_MODULE = R84_ARTIFACTS / "l4_r84_frames.F90"
SOURCE = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "ORCA2_OMIP_L4/MY_SRC/stprk3.F90"
)
MAGIC = "NEMO_L4_SBCIN_2"
HEADER = struct.Struct("=12i")
FIELD_HEADER = struct.Struct("=4i")
FIELDS = (
    "utau", "vtau", "utauU", "vtauV", "utau_b", "vtau_b",
    "utau_icb", "vtau_icb", "taum", "wndm", "qsr", "qns", "qns_b",
    "qsr_tot", "qns_tot", "emp", "emp_b", "sfx", "sfx_b", "emp_tot",
    "fwfice", "rnf", "rnf_b", "fwficb", "fr_i", "snwice_mass",
    "snwice_mass_b", "snwice_fmass", "rCdU_ice", "berg_calving",
    "berg_calv_hflx", "berg_float_melt", "berg_stored_heat", "rnf_tsc",
    "rnf_tsc_b",
)
RUNOFF_FIELDS = frozenset(("rnf", "rnf_b", "rnf_tsc", "rnf_tsc_b"))
ICEBERG_FIELDS = frozenset((
    "utau_icb", "vtau_icb", "berg_calving", "berg_calv_hflx",
    "berg_float_melt", "berg_stored_heat",
))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _take(raw: bytes, offset: int, size: int, what: str) -> tuple[bytes, int]:
    end = offset + size
    require(end <= len(raw), f"truncated {what}")
    return raw[offset:end], end


def _header(raw: bytes) -> tuple[tuple[int, ...], int]:
    chunk, offset = _take(raw, 0, 16, "magic")
    require(chunk.decode("ascii").rstrip() == MAGIC, "bad surface-record magic")
    chunk, offset = _take(raw, offset, HEADER.size, "surface header")
    return HEADER.unpack(chunk), offset


def _mutate(raw: bytes, plant: str | None) -> bytes:
    if plant is None or plant == "owner-on":
        return raw
    planted = bytearray(raw)
    if plant == "header":
        planted[0] = ord("X")
        return bytes(planted)
    if plant == "truncation":
        planted.pop()
        return bytes(planted)
    header, offset = _header(raw)
    nx, ny = header[3], header[4]
    for _ in FIELDS:
        _, offset = _take(raw, offset, 16, "field name")
        dims_offset = offset
        chunk, offset = _take(raw, offset, FIELD_HEADER.size, "field header")
        ndim, dx, dy, dz = FIELD_HEADER.unpack(chunk)
        count = math.prod((dx, dy, dz)) if ndim else 0
        if plant == "field-name":
            planted[dims_offset - 16] = ord("X")
            return bytes(planted)
        if plant == "nonfinite" and ndim:
            struct.pack_into("=d", planted, offset, float("nan"))
            return bytes(planted)
        if plant == "absent-as-zero" and ndim == 0:
            replacement = FIELD_HEADER.pack(2, nx, ny, 1) + bytes(nx * ny * 8)
            return raw[:dims_offset] + replacement + raw[offset:]
        offset += count * 8
    raise GateError(f"plant {plant} found no eligible field")


def _one_assignment(text: str, name: str) -> str:
    pattern = re.compile(
        rf"^\s*{re.escape(name)}\s*=\s*([^!,\s/]+)", re.IGNORECASE | re.MULTILINE
    )
    values = pattern.findall(text)
    require(len(values) == 1, f"expected one {name} assignment, found {values}")
    return values[0].lower()


def _namelist_switches(path: Path) -> dict[str, bool | int]:
    text = path.read_text()
    bools: dict[str, bool | int] = {}
    for name in ("ln_rnf", "ln_icebergs"):
        token = _one_assignment(text, name)
        require(token in (".true.", ".false."), f"{name}: invalid boolean {token}")
        bools[name] = token == ".true."
    token = _one_assignment(text, "nn_ice")
    require(re.fullmatch(r"[+-]?\d+", token) is not None,
            f"nn_ice: invalid integer {token}")
    bools["nn_ice"] = int(token)
    return bools


def _output_switches(path: Path) -> dict[str, bool | int]:
    text = path.read_text()
    runoff = re.findall(r"runoff / runoff mouths\s+ln_rnf\s+=\s+([TF])", text)
    ice = re.findall(r"ice management in the sbc.*nn_ice\s+=\s+(\d+)", text)
    require(len(runoff) == 1, f"ocean.output runoff resolution is {runoff}")
    require(len(ice) == 1, f"ocean.output ice resolution is {ice}")
    no_icebergs = text.count("==>>>   No icebergs used")
    require(no_icebergs == 1, "ocean.output does not resolve icebergs off exactly once")
    return {"ln_rnf": runoff[0] == "T", "ln_icebergs": False, "nn_ice": int(ice[0])}


def read_surface(path: Path, namelist: Path, ocean_output: Path,
                 plant: str | None = None) -> dict:
    switches = _namelist_switches(namelist)
    output_switches = _output_switches(ocean_output)
    require(switches == output_switches,
            f"namelist/output switch mismatch {switches} != {output_switches}")
    if plant == "owner-on":
        switches = dict(switches)
        switches["ln_rnf"] = True
    raw = _mutate(path.read_bytes(), plant)
    header, offset = _header(raw)
    (version, kt, level, nx, ny, ntr, halo, bits, nfields,
     header_rnf, header_icebergs, header_ice) = header
    require(version == 2 and kt == 1 and level in (1, 2, 3),
            f"invalid version/step/level {header[:3]}")
    require(nx > 0 and ny > 0 and ntr == 2 and halo == 2 and bits == 64,
            f"invalid surface header {header}")
    require(nfields == len(FIELDS), f"field count {nfields} != {len(FIELDS)}")
    require((header_rnf, header_icebergs, header_ice) == (
        int(bool(switches["ln_rnf"])), int(bool(switches["ln_icebergs"])),
        switches["nn_ice"],
    ), "record switch header disagrees with resolved namelist")
    require(switches["nn_ice"] == 0, "round-87 gate is rung-0/no-ice only")

    fields: dict[str, dict] = {}
    for expected in FIELDS:
        chunk, offset = _take(raw, offset, 16, f"{expected} name")
        name = chunk.decode("ascii").rstrip()
        require(name == expected, f"expected {expected!r}, got {name!r}")
        chunk, offset = _take(raw, offset, FIELD_HEADER.size, f"{name} header")
        ndim, dx, dy, dz = FIELD_HEADER.unpack(chunk)
        owner_on = (bool(switches["ln_rnf"]) if name in RUNOFF_FIELDS else
                    bool(switches["ln_icebergs"]) if name in ICEBERG_FIELDS else True)
        if not owner_on:
            require((ndim, dx, dy, dz) == (0, 0, 0, 0),
                    f"{name}: owner is off but field is not ABSENT")
            fields[name] = {"status": "ABSENT"}
            continue
        require(ndim in (2, 3), f"{name}: owner is on but field is ABSENT")
        require(dx > 0 and dy > 0 and dz > 0, f"{name}: invalid dimensions")
        require(ndim == 3 or dz == 1, f"{name}: 2-D record has dz={dz}")
        count = math.prod((dx, dy, dz))
        chunk, offset = _take(raw, offset, count * 8, f"{name} payload")
        values = np.frombuffer(chunk, dtype="=f8")
        require(bool(np.isfinite(values).all()), f"{name}: non-finite payload")
        fields[name] = {
            "status": "PRESENT", "shape": [dx, dy, dz], "count": count,
            "min": float(values.min()), "max": float(values.max()),
        }
    require(offset == len(raw), "trailing bytes after physical EOF")
    absent = [name for name, value in fields.items() if value["status"] == "ABSENT"]
    require(len(absent) == 10, f"absent census is {len(absent)}, expected 10")
    return {
        "status": "PASS_SURFACE_ABSENCE",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "switches": switches,
        "present_count": len(fields) - len(absent),
        "absent_count": len(absent),
        "absent": absent,
        "fields": fields,
    }


def preflight() -> dict:
    for path in (PATCH, R84_PATCH, R84_MODULE, SOURCE):
        require(path.is_file(), f"missing committed acquisition artifact {path}")
    for patch in (PATCH, R84_PATCH):
        removed = [line for line in patch.read_text().splitlines()
                   if line.startswith("-") and not line.startswith("---")]
        require(not removed, f"{patch.name} is not additions-only")
    with tempfile.TemporaryDirectory(prefix="orca2-r87-preflight-") as tmp:
        target = Path(tmp) / "stprk3.F90"
        shutil.copyfile(SOURCE, target)
        for patch in (PATCH, R84_PATCH):
            result = subprocess.run(
                ["patch", "-s", "--fuzz=0", str(target), str(patch)],
                text=True, capture_output=True, check=False,
            )
            require(result.returncode == 0,
                    f"{patch.name} does not apply: {result.stderr or result.stdout}")
        patched = target.read_text()
    require(patched.count("CALL r84_dump_frame") == 4,
            "combined source does not carry four frame calls")
    require(patched.count("CALL l4_r87_put_absent") == 10,
            "surface writer does not carry ten explicit ABSENT calls")
    require(patched.count("CALL l4_r87_dump_surface_input") == 1,
            "surface replacement does not have one call")
    return {
        "status": "PREFLIGHT_PASS",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "repair_patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
        "frame_patch_sha256": hashlib.sha256(R84_PATCH.read_bytes()).hexdigest(),
        "frame_module_sha256": hashlib.sha256(R84_MODULE.read_bytes()).hexdigest(),
        "removed_source_lines": 0,
        "surface_fields": len(FIELDS),
        "explicit_absent_calls": 10,
        "frame_calls": 4,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--namelist", type=Path)
    parser.add_argument("--ocean-output", type=Path)
    parser.add_argument("--plant", choices=(
        "header", "field-name", "truncation", "nonfinite",
        "absent-as-zero", "owner-on",
    ))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight:
            require(args.record is None and args.namelist is None,
                    "preflight does not accept record arguments")
            report = preflight()
        else:
            require(args.record and args.namelist and args.ocean_output,
                    "--record, --namelist and --ocean-output are required")
            report = read_surface(
                args.record, args.namelist, args.ocean_output, args.plant
            )
            require(args.plant is None, f"{args.plant} plant stayed green")
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"STATUS {report['status']}")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (GateError, OSError, UnicodeError, struct.error, ValueError) as error:
        if args.plant:
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
