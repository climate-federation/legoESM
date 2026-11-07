#!/usr/bin/env python3
"""Admit the ORCA1-ice/ORCA2 twin oracle as a distinct root family.

The original Lane-4 formats predate self-describing extent tables.  Their
category and layer counts are therefore taken from each record header and the
payload count is derived from the WRITE list.  New ORCA1-ice formats decode
from their headers alone via the Phase-2x acquisition gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_een_discriminator_gate as een,
    nemo_testcase_l4_orca2_o1_acquisition_gate as o1,
    nemo_testcase_l4_orca2_phase2b_exchange_gate as surface,
    nemo_testcase_l4_orca2_phase2o_bbl_gate as bbl,
    nemo_testcase_l4_orca2_phase2s_zdf_acquisition_gate as zdf,
    nemo_testcase_l4_orca2_phase2u_tke_acquisition_gate as tke,
    nemo_testcase_l4_orca2_phase2x_orca1ice_acquisition_gate as orca1ice,
    nemo_testcase_l4_orca2_wzv_gate as wzv,
)

NX, NY = phase1.NX, phase1.NY
N2, NI = phase1.N2, phase1.NI
ODD_KT = phase1.ODD_KT
VARIABLE_LEGACY = {
    "oracle_si3_exchange_frames.bin",
    "oracle_si3_thd_frames.bin",
    "oracle_si3_zdf_inputs.bin",
    "oracle_si3_reassoc_operands.bin",
    *{
        f"oracle_si3_prather_kt{kt:08d}_s{stage}.bin"
        for kt in ODD_KT for stage in (0, 1)
    },
}
TAILORED = {
    *{f"oracle_orca1ice_dyn_kt{kt:08d}.bin" for kt in (1, 3, 5)},
    *{
        f"oracle_orca1ice_{family}_kt{kt:08d}_f{frame}.bin"
        for family in ("thd", "prather") for kt in (1, 3, 5)
        for frame in (0, 1)
    },
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact(handle, size: int, label: str) -> bytes:
    value = handle.read(size)
    require(len(value) == size, f"{label}: truncated {len(value)}/{size}")
    return value


def _magic(handle, wanted: str, label: str) -> None:
    got = _exact(handle, 16, label).decode("ascii").rstrip()
    require(got == wanted, f"{label}: magic {got!r} != {wanted!r}")


def _finite(handle, count: int, label: str) -> None:
    left = count
    while left:
        take = min(left, 1_000_000)
        values = np.frombuffer(_exact(handle, 8 * take, label), dtype=np.float64)
        require(bool(np.isfinite(values).all()), f"{label}: non-finite payload")
        left -= take


def _eof(handle, label: str) -> None:
    require(handle.read(1) == b"", f"{label}: trailing payload")


def _parse_exchange(path: Path) -> dict[str, object]:
    rows = []
    with path.open("rb", buffering=0) as handle:
        for kt in range(1, 11):
            _magic(handle, "NEMO_L3XCHG_001", path.name)
            header = struct.unpack("=6i", _exact(handle, 24, path.name))
            version, got_kt, nx, ny, jpl, bits = header
            require((version, got_kt, nx, ny, bits) == (1, kt, NX, NY, 64),
                    f"{path.name}: bad header {header}")
            # icestp.F90:l3xchg_dump WRITE list: thirteen category fields and
            # thirteen A2D(1), nine full-domain, one A2D(0) 2-D fields.
            payload = (13 * jpl + 13) * NI + 9 * N2 + (NX - 2) * (NY - 2)
            _finite(handle, payload, path.name)
            rows.append({"header": list(header), "payload_f64": payload})
        _eof(handle, path.name)
    return {"frames": len(rows), "rows": rows, "exact_eof": True}


def _parse_thd(path: Path) -> dict[str, object]:
    rows = []
    with path.open("rb", buffering=0) as handle:
        for kt in ODD_KT:
            expected = [(0, 0)] + [
                (stage, 1) for _category in range(1) for stage in range(1, 6)
            ] + [(6, 0), (7, 0)]
            for stage, kind in expected:
                _magic(handle, "NEMO_L3THD_001", path.name)
                header = struct.unpack("=11i", _exact(handle, 44, path.name))
                version, got_kt, got_stage, got_kind, nx, ny, jpl, nli, nls, npti, bits = header
                require((version, got_kt, got_stage, got_kind, nx, ny, bits)
                        == (1, kt, stage, kind, NX, NY, 64),
                        f"{path.name}: bad header {header}")
                require((jpl, nli, nls) == (1, 3, 3),
                        f"{path.name}: unexpected ORCA1 ice allocation {jpl}/{nli}/{nls}")
                payload = ((4 + 2 * nli + nls) * npti if kind else
                           (9 + 2 * nli + nls) * jpl * nx * ny)
                if kind:
                    require(npti > 0, f"{path.name}: non-positive compressed npti")
                _finite(handle, payload, path.name)
                rows.append({"header": list(header), "payload_f64": payload})
        _eof(handle, path.name)
    return {"frames": len(rows), "rows": rows, "exact_eof": True}


def _parse_zdf(path: Path) -> dict[str, object]:
    rows = []
    with path.open("rb", buffering=0) as handle:
        for kt in ODD_KT:
            _magic(handle, "NEMO_L3ZIN_002", path.name)
            header = struct.unpack("=6i", _exact(handle, 24, path.name))
            version, got_kt, category, npti, bits, count = header
            require((version, got_kt, category, bits) == (2, kt, 1, 64),
                    f"{path.name}: bad header {header}")
            require(count == (13 + 3) * npti and npti > 0,
                    f"{path.name}: payload/header mismatch {header}")
            _finite(handle, count, path.name)
            rows.append({"header": list(header), "payload_f64": count})
        _eof(handle, path.name)
    return {"frames": len(rows), "rows": rows, "exact_eof": True}


def _parse_reassoc(path: Path) -> dict[str, object]:
    with path.open("rb", buffering=0) as handle:
        _magic(handle, "NEMO_L3REA_001", path.name)
        header = struct.unpack("=8i", _exact(handle, 32, path.name))
        version, kt, category, npti, nli, nls, bits, count = header
        require((version, kt, category, nli, nls, bits) == (1, 3, 1, 3, 3, 64),
                f"{path.name}: bad header {header}")
        require(count == (3 + 3 * nli + 2 * nls) * npti and npti > 0,
                f"{path.name}: payload/header mismatch {header}")
        _finite(handle, count, path.name)
        _eof(handle, path.name)
    return {"frames": 1, "header": list(header), "payload_f64": count,
            "exact_eof": True}


def _parse_prather(path: Path, kt: int, stage: int) -> dict[str, object]:
    with path.open("rb", buffering=0) as handle:
        _magic(handle, "NEMO_L4_PRA_001", path.name)
        header = struct.unpack("=11i", _exact(handle, 44, path.name))
        version, got_kt, got_stage, nx, ny, jpl, nli, nls, nn_icesal, ponds, bits = header
        require((version, got_kt, got_stage, nx, ny, bits) ==
                (1, kt, stage, NX, NY, 64), f"{path.name}: bad header {header}")
        require((jpl, nli, nls, nn_icesal, ponds) == (1, 3, 3, 2, 0),
                f"{path.name}: unexpected selector/allocation header {header}")
        fields_per_category = 25 + 5 * nls + 5 * nli
        if nn_icesal == 4:
            fields_per_category += 5 * nli
        if ponds:
            fields_per_category += 15
        payload = fields_per_category * jpl * nx * ny
        _finite(handle, payload, path.name)
        _eof(handle, path.name)
    return {"header": list(header), "payload_f64": payload, "exact_eof": True}


def _parse_variable_legacy(root: Path) -> dict[str, object]:
    result = {
        "oracle_si3_exchange_frames.bin": _parse_exchange(root / "oracle_si3_exchange_frames.bin"),
        "oracle_si3_thd_frames.bin": _parse_thd(root / "oracle_si3_thd_frames.bin"),
        "oracle_si3_zdf_inputs.bin": _parse_zdf(root / "oracle_si3_zdf_inputs.bin"),
        "oracle_si3_reassoc_operands.bin": _parse_reassoc(root / "oracle_si3_reassoc_operands.bin"),
    }
    for kt in ODD_KT:
        for stage in (0, 1):
            name = f"oracle_si3_prather_kt{kt:08d}_s{stage}.bin"
            result[name] = _parse_prather(root / name, kt, stage)
    return result


def _ordinary_identity(run_a: Path, run_b: Path) -> dict[str, object]:
    exact_files, icebergs = phase1._identity_exact_files(run_a, run_b)
    exact = []
    for name in exact_files:
        require((run_a / name).read_bytes() == (run_b / name).read_bytes(),
                f"ordinary exact-byte mismatch: {name}")
        exact.append({"file": name, "sha256": sha256(run_a / name)})
    histories = []
    for name in phase1.HISTORY:
        phase1._netcdf_equal_except_timestamp(run_a / name, run_b / name)
        histories.append({"file": name, "status": "EXACT_RAW_DATA_VARIABLES_EXCEPT_TIMESTAMP",
                          "a_sha256": sha256(run_a / name),
                          "b_sha256": sha256(run_b / name)})
    require(phase1._normalized_ocean(run_a / "ocean.output") ==
            phase1._normalized_ocean(run_b / "ocean.output"),
            "normalized ocean.output differs")
    from netCDF4 import Dataset
    restart_variables = []
    for name in sorted(n for n in exact_files if "_restart" in n):
        with Dataset(run_a / name) as left, Dataset(run_b / name) as right:
            require(list(left.variables) == list(right.variables),
                    f"restart variable inventory differs: {name}")
            restart_variables.append({"file": name, "variables": list(left.variables)})
    return {"status": "PASS", "icebergs_enabled": icebergs,
            "exact_byte_files": exact, "restart_shards_exact": len(restart_variables),
            "restart_variables": restart_variables,
            "history_payloads_exact": len(histories), "histories": histories,
            "ocean_output": "EXACT_TEXT_EXCEPT_DUMP_NOTICES"}


def _extensions(root: Path, mesh: Path) -> dict[str, object]:
    return {
        surface.RECORD: surface.validate_surface(root / surface.RECORD, icebergs_off=True),
        o1.RECORD: o1.read_o1(root / o1.RECORD, require_canonical_undefined=True),
        wzv.RECORD: wzv.read_record(root / wzv.RECORD)["header"],
        bbl.NAME: bbl.read_bbl(root / bbl.NAME, mesh),
        "een": een.validate_new(root),
        zdf.ZDF_NAMES[0]: zdf.read_zdf_v2(root / zdf.ZDF_NAMES[0], mesh, require_shear=False),
        zdf.ZDF_NAMES[1]: zdf.read_zdf_v2(root / zdf.ZDF_NAMES[1], mesh, require_shear=True),
        tke.RECORD: tke.read_tke(root / tke.RECORD, mesh),
    }


def _plants(root: Path, run_b: Path, baseline: Path, mesh: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    parsers = {
        "legacy_exchange": lambda p: _parse_exchange(p),
        "legacy_thd": lambda p: _parse_thd(p),
        "legacy_zdf": lambda p: _parse_zdf(p),
        "legacy_reassoc": lambda p: _parse_reassoc(p),
        "legacy_prather": lambda p: _parse_prather(p, 1, 0),
    }
    samples = {
        "legacy_exchange": "oracle_si3_exchange_frames.bin",
        "legacy_thd": "oracle_si3_thd_frames.bin",
        "legacy_zdf": "oracle_si3_zdf_inputs.bin",
        "legacy_reassoc": "oracle_si3_reassoc_operands.bin",
        "legacy_prather": "oracle_si3_prather_kt00000001_s0.bin",
    }
    with tempfile.TemporaryDirectory(prefix="orca2-p2y-plants-") as td:
        tmp = Path(td)
        for label, parser in parsers.items():
            raw = (root / samples[label]).read_bytes()
            for kind, value in (("magic", b"X" + raw[1:]),
                                ("truncated", raw[:-1]), ("trailing", raw + b"X")):
                path = tmp / f"{label}-{kind}.bin"
                path.write_bytes(value)
                try:
                    parser(path)
                except (GateError, ValueError, UnicodeError):
                    result[f"{label}_{kind}"] = "PASS_NONZERO"
                else:
                    raise GateError(f"plant did not bind: {label}_{kind}")
    tailored = orca1ice._self_test()["binding_plants"]
    require(len(tailored) == 9 and all(tailored.values()), "tailored schema plants failed")
    result.update({f"tailored_{name}": "PASS_NONZERO" for name in tailored})

    # The twin assertion itself must be binding.
    target = "oracle_orca1ice_dyn_kt00000001.bin"
    altered = bytearray((run_b / target).read_bytes())
    altered[-1] ^= 1
    require(bytes(altered) != (root / target).read_bytes(), "twin one-byte plant inert")
    result["twin_record_identity_byte"] = "PASS_NONZERO"

    # Reuse the frozen V2 gate to prove every inherited parser family and its
    # ordinary-output identity plant still bind on the current code revision.
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_phase2t_combined_schema_gate as combined,
    )
    with tempfile.TemporaryDirectory(prefix="orca2-p2y-v2-100-") as td:
        inherited_root = Path(td)
        for path in baseline.iterdir():
            if path.is_file() and path.name != tke.RECORD:
                os.symlink(path, inherited_root / path.name)
        inherited = combined.validate(
            inherited_root, mesh, baseline, True)["plants"]
    require(inherited, "inherited plants absent")
    result["inherited_100_stream_gate"] = "PASS_WITH_BINDING_PLANTS"
    return result


def validate(run_a: Path, run_b: Path, baseline: Path, plants: bool) -> dict[str, object]:
    base = phase1.expected_inventory()
    extensions = {
        surface.RECORD, o1.RECORD, wzv.RECORD, bbl.NAME, *een.NEW_STREAMS,
        *zdf.ZDF_NAMES, tke.RECORD,
    }
    wanted = base | extensions | TAILORED
    actual_a = {p.name for p in run_a.glob("oracle_*.bin")}
    actual_b = {p.name for p in run_b.glob("oracle_*.bin")}
    require(len(base) == 90 and len(extensions) == 11 and len(TAILORED) == 15,
            "internal 90+11+15 inventory partition changed")
    require(actual_a == actual_b == wanted,
            f"116-stream inventory mismatch: missing={sorted(wanted-actual_a)} "
            f"extra={sorted(actual_a-wanted)}")
    twin_rows = []
    for name in sorted(wanted):
        require((run_a / name).read_bytes() == (run_b / name).read_bytes(),
                f"twin record mismatch: {name}")
        twin_rows.append({"file": name, "sha256": sha256(run_a / name),
                          "bytes": (run_a / name).stat().st_size})

    # Validate the 76 allocation-invariant Phase-1 streams with the frozen
    # parser.  Its fourteen category/layer-dependent inputs come from V2 only
    # in this temporary view; the real ORCA1-ice forms are parsed immediately
    # below with header-derived allocations.
    with tempfile.TemporaryDirectory(prefix="orca2-p2y-base90-") as td:
        view = Path(td)
        for name in base:
            source = baseline if name in VARIABLE_LEGACY else run_a
            os.symlink(source / name, view / name)
        invariant_rows = phase1.validate_records(view)
    variable_rows = _parse_variable_legacy(run_a)
    mesh = run_a / "mesh_mask_0000.nc"
    extension_rows = _extensions(run_a, mesh)
    tailored_rows = [orca1ice._parse(run_a / name) for name in sorted(TAILORED)]
    return {
        "status": "PASS", "root_label": "VARIANT_ORACLE_ORCA1ICE",
        "inventory": {"phase1": 90, "extensions": 11, "orca1ice": 15,
                      "total": len(wanted)},
        "twin_raw_exact": len(twin_rows), "twin_total": len(twin_rows),
        "records": twin_rows,
        "phase1_allocation_invariant_streams": len(base - VARIABLE_LEGACY),
        "phase1_frozen_composite": invariant_rows,
        "phase1_variable_legacy": variable_rows,
        "extensions": extension_rows, "orca1ice_streams": tailored_rows,
        "ordinary_output_identity": _ordinary_identity(run_a, run_b),
        "plants": _plants(run_a, run_b, baseline, mesh) if plants else {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plants", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.run_a, args.run_b, args.baseline, args.plants)
    except (GateError, phase1.GateError, OSError, UnicodeError, ValueError) as error:
        print(f"FAIL: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json:
        args.json.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
