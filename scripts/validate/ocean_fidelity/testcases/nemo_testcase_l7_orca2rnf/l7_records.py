#!/usr/bin/env python3
"""Read and police the L7 river-mouth before/after record."""

from __future__ import annotations

import argparse
import json
import re
import struct
import tempfile
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_L7_RNFKZ_1 "
TAGS = (b"CLOSURE_AVT     ", b"RIVER_AVT       ", b"EVD_DDM_AVT     ",
        b"IWM_AVT         ")
RUN_PAIRS = (("smoke_2step", "smoke_2step"), ("twin_a", "twin_a"),
             ("twin_b", "twin_b"), ("twin_c", "twin_c"),
             ("ref_boundary", "ref_100step"))
EXPECTED_RECORDS = {"smoke_2step": 45, "twin_a": 108,
                    "twin_b": 108, "twin_c": 108}
EXPECTED_RESTARTS = {"smoke_2step": 2, "twin_a": 2, "twin_b": 6,
                     "twin_c": 5, "ref_boundary": 10}


def read(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as stream:
        magic = stream.read(16)
        header = stream.read(28)
        if len(header) != 28:
            raise ValueError("short L7 header")
        version, kt, nx, ny, nz, bits, nfields = struct.unpack("=7i", header)
        if magic != MAGIC or (version, kt, bits, nfields) != (1, 1, 64, 4):
            raise ValueError("wrong L7 record identity")
        if min(nx, ny, nz) <= 0:
            raise ValueError("nonpositive L7 shape")
        out: dict[str, np.ndarray] = {}
        count = nx * ny * nz
        for expected in TAGS:
            tag = stream.read(16)
            raw = stream.read(count * 8)
            if tag != expected or len(raw) != count * 8:
                raise ValueError("wrong tag or short L7 field")
            values = np.frombuffer(raw, dtype=np.float64).reshape(
                (nx, ny, nz), order="F").copy()
            if not np.isfinite(values).all():
                raise ValueError("nonfinite L7 observer value")
            out[tag.decode().strip()] = values
        if stream.read(1):
            raise ValueError("trailing bytes in L7 record")
    return out


def write_synthetic(path: Path, *, acting: bool) -> None:
    shape = (4, 3, 5)
    before = np.arange(np.prod(shape), dtype=np.float64).reshape(shape, order="F")
    after = before.copy()
    if acting:
        after[1, 1, 1] += 2.0e-3
    fields = (before, after, after + 1.0e-4, after + 2.0e-4)
    with path.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("=7i", 1, 1, *shape, 64, 4))
        for tag, field in zip(TAGS, fields, strict=True):
            stream.write(tag)
            stream.write(np.asarray(field, order="F").tobytes(order="F"))


def check_acting(path: Path) -> None:
    record = read(path)
    if np.count_nonzero(record["RIVER_AVT"] - record["CLOSURE_AVT"]) == 0:
        raise ValueError("river-mouth statement did not act on any defined cell")


def selftest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.bin"
        inert = Path(tmp) / "inert.bin"
        write_synthetic(good, acting=True)
        write_synthetic(inert, acting=False)
        check_acting(good)
        try:
            check_acting(inert)
        except ValueError:
            pass
        else:
            raise AssertionError("inert acting-control was accepted")
        raw = good.read_bytes()
        short = Path(tmp) / "short.bin"
        short.write_bytes(raw[:-8])
        try:
            read(short)
        except ValueError:
            pass
        else:
            raise AssertionError("short record was accepted")


def _same_file(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            xa, xb = a.read(8 * 1024 * 1024), b.read(8 * 1024 * 1024)
            if xa != xb:
                return False
            if not xa:
                return True


def _require_same_file(left: Path, right: Path, *, plant: bool = False) -> None:
    if plant or not _same_file(left, right):
        raise ValueError(f"files differ: {left} != {right}")


def _value_equal(left, right) -> bool:
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left, right))


def _compare_netcdf_payload(left: Path, right: Path) -> int:
    """Compare every dimension, variable word, and attribute but run timestamp."""
    import netCDF4

    with netCDF4.Dataset(left) as a, netCDF4.Dataset(right) as b:
        if set(a.dimensions) != set(b.dimensions):
            raise ValueError(f"netCDF dimensions differ: {left.name}")
        for name in a.dimensions:
            da, db = a.dimensions[name], b.dimensions[name]
            if len(da) != len(db) or da.isunlimited() != db.isunlimited():
                raise ValueError(f"netCDF dimension differs: {left.name}:{name}")
        if set(a.variables) != set(b.variables):
            raise ValueError(f"netCDF variables differ: {left.name}")
        for attr in set(a.ncattrs()) | set(b.ncattrs()):
            if attr == "TimeStamp":
                continue
            if attr not in a.ncattrs() or attr not in b.ncattrs() or not _value_equal(
                    a.getncattr(attr), b.getncattr(attr)):
                raise ValueError(f"netCDF global attribute differs: {left.name}:{attr}")
        for name in a.variables:
            va, vb = a.variables[name], b.variables[name]
            if va.dtype != vb.dtype or va.dimensions != vb.dimensions:
                raise ValueError(f"netCDF variable schema differs: {left.name}:{name}")
            for attr in set(va.ncattrs()) | set(vb.ncattrs()):
                if attr not in va.ncattrs() or attr not in vb.ncattrs() or not _value_equal(
                        va.getncattr(attr), vb.getncattr(attr)):
                    raise ValueError(
                        f"netCDF variable attribute differs: {left.name}:{name}:{attr}")
            va.set_auto_maskandscale(False)
            vb.set_auto_maskandscale(False)
            if not _value_equal(np.asarray(va[:]), np.asarray(vb[:])):
                raise ValueError(f"netCDF variable words differ: {left.name}:{name}")
        return len(a.variables)


def _audit_failed_observer(path: Path) -> dict:
    """Reproduce the acquired short-field and nonconforming-assignment failure."""
    with path.open("rb") as stream:
        if stream.read(16) != MAGIC:
            raise ValueError("wrong failed-observer magic")
        version, kt, nx, ny, nz, bits, nfields = struct.unpack(
            "=7i", stream.read(28))
        if (version, kt, bits, nfields) != (1, 1, 64, 4):
            raise ValueError("wrong failed-observer header")
        count = nx * ny * nz
        fields = []
        for expected in TAGS[:3]:
            if stream.read(16) != expected:
                raise ValueError("wrong failed-observer tag")
            fields.append(np.frombuffer(stream.read(count * 8), dtype=np.float64))
        if stream.read(16) != TAGS[3]:
            raise ValueError("wrong failed-observer final tag")
        final = np.frombuffer(stream.read(), dtype=np.float64)
    interior_count = (nx - 4) * (ny - 4) * nz
    difference = (fields[1] - fields[0])[:interior_count]
    report = {
        "header_shape": [nx, ny, nz],
        "header_values": count,
        "fourth_field_values": int(final.size),
        "actual_bytes": path.stat().st_size,
        "expected_bytes": 16 + 28 + 4 * (16 + count * 8),
        "river_minus_closure_unequal_first_interior_count": int(
            np.count_nonzero(difference)),
        "river_minus_closure_min_first_interior_count": float(difference.min()),
    }
    if report != {
            "header_shape": [184, 152, 31],
            "header_values": 867008,
            "fourth_field_values": 825840,
            "actual_bytes": 27415020,
            "expected_bytes": 27744364,
            "river_minus_closure_unequal_first_interior_count": 31143,
            "river_minus_closure_min_first_interior_count": -930748.7023730353}:
        raise ValueError(f"failed-observer reproduction changed: {report}")
    return report


def _river_selector(path: Path) -> bool:
    match = re.search(r"^\s*ln_rnf_mouth\s*=\s*\.(true|false)\.",
                      path.read_text(), flags=re.IGNORECASE | re.MULTILINE)
    if match is None:
        raise ValueError(f"ln_rnf_mouth is absent: {path}")
    return match.group(1).lower() == "true"


def compare_v5_evidence(l7_root: Path, v5_root: Path, *, plant: bool = False) -> dict:
    """Fail-closed identity census of every comparable L7/V5 NEMO output."""
    l7_root, v5_root = Path(l7_root), Path(v5_root)
    if not l7_root.is_dir() or not v5_root.is_dir():
        raise ValueError("L7/V5 evidence root missing")
    report = {"status": "PASS_L7_V5_NEMO_IDENTITY", "records": {},
              "restarts": {}, "history": {}, "other_netcdf": {}}
    planted = False
    for l7_name, v5_name in RUN_PAIRS:
        l7_dir, v5_dir = l7_root / l7_name, v5_root / v5_name
        if not l7_dir.is_dir() or not v5_dir.is_dir():
            raise ValueError(f"evidence run missing: {l7_name}/{v5_name}")

        l7_records = {p.name for p in l7_dir.glob("oracle_*.bin")}
        v5_records = {p.name for p in v5_dir.glob("oracle_*.bin")}
        if l7_name in EXPECTED_RECORDS:
            l7_only = {"oracle_l7_rnf_kt00000001.bin"}
            v5_only = {f"oracle_v1_hm_i_kt{kt:08d}.bin" for kt in range(
                1, 3 if l7_name == "smoke_2step" else 11)}
            if l7_records - l7_only != v5_records - v5_only:
                raise ValueError(f"standard record inventory differs: {l7_name}")
            common = sorted(l7_records & v5_records)
            if len(common) != EXPECTED_RECORDS[l7_name]:
                raise ValueError(f"standard record count differs: {l7_name}={len(common)}")
            for name in common:
                fire = plant and not planted
                _require_same_file(l7_dir / name, v5_dir / name, plant=fire)
                planted |= fire
            report["records"][l7_name] = len(common)

        l7_restarts = {p.name for p in l7_dir.glob("ORCA2G1_*restart*.nc")}
        v5_restarts = {p.name for p in v5_dir.glob("ORCA2G1_*restart*.nc")}
        if l7_restarts != v5_restarts or len(l7_restarts) != EXPECTED_RESTARTS[l7_name]:
            raise ValueError(f"restart inventory differs: {l7_name}")
        counts = {"ocean": 0, "ice": 0}
        for name in sorted(l7_restarts):
            _require_same_file(l7_dir / name, v5_dir / name)
            variables = _compare_netcdf_payload(l7_dir / name, v5_dir / name)
            if "_ice.nc" in name:
                kind, want = "ice", 208 if l7_name == "smoke_2step" else 210
            else:
                kind, want = "ocean", 42
            if variables != want:
                raise ValueError(f"unexpected restart variable count: {name}={variables}")
            counts[kind] += 1
        report["restarts"][l7_name] = counts

        histories = {p.name for p in l7_dir.glob("ORCA2G1_*_grid_?_0000.nc")}
        if histories != {p.name for p in v5_dir.glob("ORCA2G1_*_grid_?_0000.nc")} \
                or len(histories) != 4:
            raise ValueError(f"history inventory differs: {l7_name}")
        report["history"][l7_name] = {
            name: _compare_netcdf_payload(l7_dir / name, v5_dir / name)
            for name in sorted(histories)
        }
        for name in ("mesh_mask.nc",):
            _require_same_file(l7_dir / name, v5_dir / name)
        aborts = {p.name for p in l7_dir.glob("output.abort*.nc")}
        if aborts != {p.name for p in v5_dir.glob("output.abort*.nc")}:
            raise ValueError(f"abort-output inventory differs: {l7_name}")
        for name in aborts:
            _require_same_file(l7_dir / name, v5_dir / name)
        report["other_netcdf"][l7_name] = {
            "mesh_mask": 1, "abort": len(aborts)}

    broken = []
    for name in ("smoke_2step", "twin_a", "twin_b", "twin_c"):
        path = l7_root / name / "oracle_l7_rnf_kt00000001.bin"
        if path.stat().st_size != 27_415_020:
            raise ValueError(f"unexpected failed-observer size: {path}")
        try:
            read(path)
        except ValueError as exc:
            if str(exc) != "wrong tag or short L7 field":
                raise
        else:
            raise ValueError(f"failed observer unexpectedly parsed: {path}")
        broken.append(str(path.relative_to(l7_root)))
    failed_observer = _audit_failed_observer(
        l7_root / "smoke_2step" / "oracle_l7_rnf_kt00000001.bin")
    selectors = {
        "l7_ref_boundary": _river_selector(l7_root / "ref_boundary" / "namelist_cfg"),
        "l7_twin_b": _river_selector(l7_root / "twin_b" / "namelist_cfg"),
        "v5_ref_100step": _river_selector(v5_root / "ref_100step" / "namelist_cfg"),
        "v5_twin_b": _river_selector(v5_root / "twin_b" / "namelist_cfg"),
    }
    if selectors != {"l7_ref_boundary": False, "l7_twin_b": True,
                     "v5_ref_100step": True, "v5_twin_b": True}:
        raise ValueError(f"unexpected archived river selectors: {selectors}")
    for kt in range(2, 11, 2):
        restart = f"ORCA2G1_{kt:08d}_restart.nc"
        _require_same_file(l7_root / "ref_boundary" / restart,
                           l7_root / "twin_b" / restart)
        _require_same_file(l7_root / "ref_boundary" / restart,
                           v5_root / "ref_100step" / restart)
    report["selector_identity"] = {
        **selectors,
        "l7_off_vs_l7_on_raw_identical_ocean_restarts": 5,
        "l7_off_vs_v5_on_raw_identical_ocean_restarts": 5,
        "note": ("the archived V5 reference and twin decks both select river-mouth "
                 "mixing; they are not an OFF/ON deck pair"),
    }
    report["noncomparable"] = {
        "l7_observer": broken,
        "l7_observer_reason": "fourth field is 825840 values, not header's 867008",
        "l7_observer_reproduction": failed_observer,
        "v5_hm_i": 32,
        "v5_hm_i_reason": "V5-only historical observer absent from L7 source card",
        "runtime_metadata": ["nemo", "namelist_cfg", "ocean.output",
                             "output.namelist.dyn", "run.user.log",
                             "timing.output", "timing_step.nc"],
        "runtime_metadata_reason": "producer, selected deck, log, or wall-clock metadata",
        "history_container_reason": (
            "payloads match; raw files carry different TimeStamp attributes"),
    }
    report["totals"] = {
        "standard_records": sum(report["records"].values()),
        "twin_restart_files": sum(sum(report["restarts"][name].values())
                                  for name in ("twin_a", "twin_b", "twin_c")),
        "all_restart_files": sum(sum(value.values())
                                 for value in report["restarts"].values()),
        "history_files": sum(len(value) for value in report["history"].values()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--check", type=Path)
    parser.add_argument("--check-acting", type=Path)
    parser.add_argument("--compare-defined", nargs=2, type=Path)
    parser.add_argument("--compare-v5-evidence", nargs=2, type=Path)
    parser.add_argument("--plant-evidence-record", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.check:
        read(args.check)
    elif args.check_acting:
        check_acting(args.check_acting)
    elif args.compare_defined:
        left, right = map(read, args.compare_defined)
        if any(not np.array_equal(left[key], right[key]) for key in left):
            raise SystemExit("defined L7 twins differ")
    elif args.compare_v5_evidence:
        print(json.dumps(compare_v5_evidence(*args.compare_v5_evidence,
                                             plant=args.plant_evidence_record),
                         indent=2, sort_keys=True))
    else:
        parser.error("choose one action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
