#!/usr/bin/env python3
"""Make the hash-bound, exact-record initialization file for SI3 rung 3.6.

This is not a forcing generator.  It copies the first float32 C1D ERA5
surface record into the two vertical slots required by NEMO's one-wet-layer
``jpk=2`` ingestion schema.  Repeating the source bits in the terminal masked
slot makes that inactive storage deterministic.  Horizontal interpolation of
the real REG05 chlorophyll is deliberately *not* implemented here: the rung
uses NEMO's shipped ``tools/WEIGHTS`` workflow for that operation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


SOURCE_TO_TARGET = {
    "sst": "votemper",
    "sss": "vosaline",
    "ssu": "u_current",
    "ssv": "v_current",
}


def make_initial_file(source: Path, output: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with Dataset(source, "r") as src, Dataset(output, "w", format="NETCDF4_CLASSIC") as dst:
        # NEMO iom identifies the record axis by its unlimited status.  A
        # fixed-size time_counter is misclassified as a fourth spatial axis.
        dst.createDimension("time_counter", None)
        dst.createDimension("deptht", 2)
        dst.createDimension("latitude", 1)
        dst.createDimension("longitude", 1)

        time = dst.createVariable("time_counter", "f4", ("time_counter",))
        time.units = src.variables["time_counter"].units
        time.calendar = src.variables["time_counter"].calendar
        time[:] = src.variables["time_counter"][0]

        lon = dst.createVariable("longitude", "f4", ("longitude",))
        lat = dst.createVariable("latitude", "f4", ("latitude",))
        depth = dst.createVariable("deptht", "f8", ("deptht",))
        lon.units = "degrees_east"
        lat.units = "degrees_north"
        depth.units = "m"
        lon[:] = np.asarray(src.variables["nav_lon"][0, 0], dtype=np.float32)
        lat[:] = np.asarray(src.variables["nav_lat"][0, 0], dtype=np.float32)
        depth[:] = np.asarray([0.5, 1.0], dtype=np.float64)

        for source_name, target_name in SOURCE_TO_TARGET.items():
            source_value = np.asarray(src.variables[source_name][0, 0, 0], dtype=np.float32)
            target = dst.createVariable(
                target_name,
                "f4",
                ("time_counter", "deptht", "latitude", "longitude"),
            )
            target[0, :, 0, 0] = np.asarray([source_value, source_value], dtype=np.float32)
            values[source_name] = float(source_value)

        dst.source_file = str(source)
        dst.source_record = np.int32(0)
        dst.construction = "exact float32 bit copy; terminal masked level repeats level 1"

    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing derived input: {args.output}")
    values = make_initial_file(args.source, args.output)
    print(f"created={args.output}")
    for name in ("sst", "sss", "ssu", "ssv"):
        value = np.float32(values[name])
        print(f"{name}={value!r} bits=0x{value.view(np.uint32):08x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
