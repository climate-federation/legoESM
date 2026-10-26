#!/usr/bin/env python3
"""Admit raw round-339 ``fr_i`` records and build ``ice_cover_clim.nc``."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round339_common as common,
)

MAGIC = "NEMO_R339_ICE1"
HEADER_NAMES = (
    "version", "kt", "rank", "nimpp", "njmpp", "jpi", "jpj",
    "ntsi", "ntei", "ntsj", "ntej", "real_bits", "nn_fsbc",
)
NX, NY = 180, 148
N_SAMPLES = 1460
EXPECTED_KT = tuple(range(1, 2920, 2))
RANKS = (0, 1)
PLANTS = ("header", "truncation", "checksum", "rank", "nonfinite", "one-ulp")

GateError = common.GateError
require = common.require


def record_name(kt: int, rank: int) -> str:
    return f"oracle_r339_fr_i_kt{kt:08d}_r{rank:04d}.bin"


def xor_checksum(values: np.ndarray) -> int:
    words = np.asarray(values, dtype="=f8", order="F").ravel(order="F").view("=u8")
    return int(np.bitwise_xor.reduce(words, initial=np.uint64(0)))


def read_record(path: Path, *, plant: str | None = None) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    need = 16 + 4 * len(HEADER_NAMES) + 8
    require(len(raw) >= need, f"truncated record {path.name}")
    magic = raw[:16].decode("ascii").rstrip()
    header = dict(zip(HEADER_NAMES, struct.unpack_from(f"={len(HEADER_NAMES)}i", raw, 16), strict=True))
    if plant == "header":
        header["real_bits"] = 32
    require(magic == MAGIC, f"record magic {magic!r}")
    require(header["version"] == 1, f"record version {header['version']}")
    require(header["real_bits"] == 64, "record is not binary64")
    require(header["nn_fsbc"] == 2, "record nn_fsbc is not 2")
    nwords = header["jpi"] * header["jpj"]
    expected_size = need + nwords * 8
    require(len(raw) == expected_size, f"record size {len(raw)} != {expected_size}")
    checksum, = struct.unpack_from("=q", raw, 16 + 4 * len(HEADER_NAMES))
    values = np.frombuffer(raw, dtype="=f8", count=nwords, offset=need).reshape(
        (header["jpi"], header["jpj"]), order="F").copy()
    if plant == "checksum":
        values[0, 0] = np.nextafter(values[0, 0], np.inf)
    if plant == "rank":
        header["rank"] = 1 - header["rank"]
    if plant == "nonfinite":
        values[0, 0] = np.nan
    require(xor_checksum(values) == checksum % (1 << 64), "payload checksum mismatch")
    require(bool(np.all(np.isfinite(values))), "record contains non-finite fr_i")
    require(bool(np.all((values >= 0.0) & (values <= 1.0))), "fr_i is outside [0,1]")
    require(1 <= header["ntsi"] <= header["ntei"] <= header["jpi"], "invalid i owned range")
    require(1 <= header["ntsj"] <= header["ntej"] <= header["jpj"], "invalid j owned range")
    return {"header": header, "values": values, "sha256": common.sha256(path)}


def owned_global(record: dict) -> tuple[slice, slice, np.ndarray]:
    h = record["header"]
    # nimpp/njmpp are one-based full-grid origins; native forcing excludes
    # the two global halo rows/columns, while Python slices are zero-based.
    i0 = h["nimpp"] + h["ntsi"] - 4
    j0 = h["njmpp"] + h["ntsj"] - 4
    i1 = i0 + h["ntei"] - h["ntsi"] + 1
    j1 = j0 + h["ntej"] - h["ntsj"] + 1
    values = record["values"][h["ntsi"] - 1:h["ntei"], h["ntsj"] - 1:h["ntej"]].T
    require((j1 - j0, i1 - i0) == values.shape, "owned/global shape mismatch")
    require(0 <= i0 < i1 <= NX and 0 <= j0 < j1 <= NY, "owned range leaves native grid")
    return slice(j0, j1), slice(i0, i1), values


def build_mean(root: Path, *, plant: str | None = None) -> tuple[np.ndarray, dict]:
    total = np.zeros((NY, NX), dtype=np.float64)
    record_hashes: dict[str, str] = {}
    for sample_index, kt in enumerate(EXPECTED_KT):
        frame = np.empty((NY, NX), dtype=np.float64)
        coverage = np.zeros((NY, NX), dtype=np.uint8)
        for rank in RANKS:
            parser_plant = plant if sample_index == 0 and rank == 0 and plant in PLANTS[:-1] else None
            path = root / record_name(kt, rank)
            require(path.is_file(), f"missing record {path.name}")
            record = read_record(path, plant=parser_plant)
            h = record["header"]
            require((h["kt"], h["rank"]) == (kt, rank),
                    f"record tag {(h['kt'], h['rank'])} != {(kt, rank)}")
            ys, xs, values = owned_global(record)
            require(not np.any(coverage[ys, xs]), f"rank {rank} overlaps another owned region")
            frame[ys, xs] = values
            coverage[ys, xs] += 1
            record_hashes[path.name] = record["sha256"]
        require(bool(np.all(coverage == 1)), f"kt {kt}: owned grid has gaps or overlaps")
        total = total + frame
    mean = total / np.float64(N_SAMPLES)
    if plant == "one-ulp":
        mean[0, 0] = np.nextafter(mean[0, 0], np.inf)
    return mean, record_hashes


def write_climatology(path: Path, values: np.ndarray, provenance: dict) -> None:
    import netCDF4

    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("time_counter", None)
        dataset.createDimension("y", NY)
        dataset.createDimension("x", NX)
        variable = dataset.createVariable("ice_cover", "f8", ("time_counter", "y", "x"),
                                          fill_value=False)
        variable[0, :, :] = values
        dataset.setncattr("producer_commit", provenance["producer_commit"])
        dataset.setncattr("sample_count", N_SAMPLES)
        dataset.setncattr("sample_kt_first", EXPECTED_KT[0])
        dataset.setncattr("sample_kt_last", EXPECTED_KT[-1])
        dataset.setncattr("summation_order", f"binary64 increasing kt; one divide by {N_SAMPLES}")


def verify_climatology(path: Path, expected: np.ndarray) -> dict:
    import netCDF4

    with netCDF4.Dataset(path) as dataset:
        require(set(dataset.dimensions) == {"time_counter", "y", "x"}, "climatology dimensions moved")
        require(dataset.dimensions["time_counter"].isunlimited(), "time_counter is not unlimited")
        require(tuple(len(dataset.dimensions[n]) for n in ("time_counter", "y", "x")) == (1, NY, NX),
                "climatology dimension lengths moved")
        require(set(dataset.variables) == {"ice_cover"}, "climatology variable inventory moved")
        variable = dataset.variables["ice_cover"]
        require(variable.dimensions == ("time_counter", "y", "x"), "ice_cover dimension order moved")
        require(variable.dtype == np.dtype("float64"), "ice_cover is not binary64")
        require("_FillValue" not in variable.ncattrs(), "ice_cover has a fill value")
        require("scale_factor" not in variable.ncattrs() and "add_offset" not in variable.ncattrs(),
                "ice_cover has packing attributes")
        observed = np.asarray(variable[0, :, :])
    require(np.array_equal(observed.view(np.uint64), expected.view(np.uint64)),
            "ice_cover payload differs from kt-order mean")
    return {"sha256": common.sha256(path), "minimum": float(expected.min()),
            "maximum": float(expected.max())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--producer-commit")
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--same-program", type=Path)
    parser.add_argument("--reference-bld", type=Path)
    parser.add_argument("--observer-patch", type=Path)
    parser.add_argument("--new-cfg")
    parser.add_argument("--reference-cfg")
    args = parser.parse_args()
    try:
        if args.same_program:
            require(all((args.reference_bld, args.observer_patch, args.new_cfg, args.reference_cfg)),
                    "same-program arguments incomplete")
            result = common.same_program(
                args.same_program, args.reference_bld, args.observer_patch,
                observer_unit="sbcmod", writer_unit="round339_ice_cover_writer",
                new_cfg=args.new_cfg, reference_cfg=args.reference_cfg)
        else:
            require(all((args.record_dir, args.output, args.producer_commit)),
                    "admission arguments incomplete")
            values, records = build_mean(args.record_dir, plant=args.plant)
            require(args.plant is None, f"{args.plant} plant stayed green")
            if not args.output.exists():
                write_climatology(args.output, values, {"producer_commit": args.producer_commit})
            result = {
                "status": "PASS_R339_ICE_COVER_PRODUCTION",
                "producer_commit": args.producer_commit,
                "sample_count": N_SAMPLES,
                "records": records,
                "climatology": verify_climatology(args.output, values),
            }
    except (GateError, OSError, ValueError, UnicodeError, struct.error) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant else 'REFUSE'}: {error}")
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
