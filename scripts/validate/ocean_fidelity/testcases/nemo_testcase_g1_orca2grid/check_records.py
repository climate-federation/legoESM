#!/usr/bin/env python3
"""Admit the ORCA2G1 step records written by g1_step_record.F90.

Everything checked is read from the streams' own headers: no array size is
predicted.  ``--plant`` corrupts the in-memory copy of one record and the
check must then exit nonzero.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"NEMO_G1_STEP_01 "
HEADER_NAMES = ("version", "step", "Nbb", "Nnn", "Naa", "Nrhs", "jpi", "jpj",
                "jpk", "Nis0", "Njs0", "Nie0", "Nje0", "word_bits")
STAGE_FRAMES = ("e", "b", "1", "2", "3", "f")
PLANTS = ("none", "truncation", "nonfinite", "missing-frame", "twin-ulp",
          "missing-step")


class Refuse(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def read_stream(blob: bytes, label: str):
    require(blob[:16] == MAGIC, f"{label}: bad magic {blob[:16]!r}")
    header = dict(zip(HEADER_NAMES, struct.unpack("<14i", blob[16:72])))
    require(header["word_bits"] == 64, f"{label}: word size is not 64 bits")
    offset, fields = 72, {}
    while offset < len(blob):
        require(offset + 32 <= len(blob), f"{label}: truncated record header")
        name = blob[offset:offset + 16].decode("ascii").strip()
        rank, n1, n2, n3 = struct.unpack("<4i", blob[offset + 16:offset + 32])
        count = n1 * n2 * n3
        offset += 32
        require(offset + 8 * count <= len(blob),
                f"{label}: truncated payload of {name}")
        require(name not in fields, f"{label}: duplicate field {name}")
        data = np.frombuffer(blob, "<f8", count, offset).reshape(
            (n2, n1) if rank == 2 else (n3, n2, n1))
        require(data.shape[-1] == header["jpi"] and data.shape[-2] == header["jpj"],
                f"{label}: {name} is not a (jpj, jpi) field of its own header")
        fields[name] = data
        offset += 8 * count
    return header, fields


def step_path(root: Path, step: int) -> Path:
    return root / f"oracle_g1step_kt{step:08d}.bin"


def check(evidence: Path, twin: Path | None, steps: int, stage_steps: int,
          plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    first_header, names_by_step, worst = None, {}, 0.0
    for step in range(1, steps + 1):
        path = step_path(evidence, step)
        if plant == "missing-step" and step == steps:
            path = evidence / "absent.bin"
        require(path.is_file(), f"missing record {path.name}")
        blob = path.read_bytes()
        if plant == "truncation" and step == 1:
            blob = blob[:-8]
        header, fields = read_stream(blob, path.name)
        if plant == "missing-frame" and step == 1:
            fields = {k: v for k, v in fields.items() if not k.startswith("b_")}
        if plant == "nonfinite" and step == 1:
            fields = dict(fields)
            name = next(iter(fields))
            bad = fields[name].copy()
            bad.flat[0] = np.nan
            fields[name] = bad
        require(header["step"] == step, f"{path.name}: header step {header['step']}")
        geometry = {k: header[k] for k in HEADER_NAMES[6:]}
        if first_header is None:
            first_header = geometry
        require(geometry == first_header, f"{path.name}: geometry header moved")
        frames = {name.split("_", 1)[0] for name in fields}
        wanted = set(STAGE_FRAMES) if step <= stage_steps else {"f"}
        require(frames == wanted,
                f"{path.name}: frames {sorted(frames)} != {sorted(wanted)}")
        for name, data in fields.items():
            require(bool(np.all(np.isfinite(data))), f"{path.name}: {name} non-finite")
        names_by_step[step] = sorted(fields)
        kind = 1 if step <= stage_steps else stage_steps + 1
        require(names_by_step[step] == names_by_step[kind],
                f"{path.name}: field list moved")
        worst = max(worst, float(np.max(np.abs(fields["f_ssh_bb"]))))
        if twin is not None and step_path(twin, step).is_file():
            other = step_path(twin, step).read_bytes()
            if plant == "twin-ulp" and step == 1:
                other = other[:-1] + bytes([other[-1] ^ 1])
            require(other == blob, f"{path.name}: twin run differs")
    require(worst > 0.0, "ssh is identically zero: the bump was not applied")
    return {"status": "PASS_G1_RECORD", "steps": steps,
            "stage_steps": stage_steps, "geometry": first_header,
            "max_abs_ssh": worst, "fields_stage": names_by_step[1],
            "fields_final_only": names_by_step.get(stage_steps + 1, [])}


def restarts_identical(reference: Path, candidate: Path, steps: list[int]) -> dict:
    import netCDF4
    compared = {}
    for step in steps:
        name = f"ORCA2G1_{step:08d}_restart.nc"
        require((reference / name).is_file() and (candidate / name).is_file(),
                f"restart {name} missing from the reference or the candidate")
        with netCDF4.Dataset(reference / name) as a, netCDF4.Dataset(candidate / name) as b:
            require(set(a.variables) == set(b.variables), f"{name}: variable set differs")
            for var in a.variables:
                left = np.asarray(a.variables[var][:])
                right = np.asarray(b.variables[var][:])
                require(left.tobytes() == right.tobytes(),
                        f"{name}: {var} differs between reference and record build")
            compared[name] = len(a.variables)
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--twin", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--stage-steps", type=int, default=10)
    parser.add_argument("--restart-steps", default="")
    parser.add_argument("--plant", default="none", choices=PLANTS)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        result = check(args.evidence, args.twin, args.steps, args.stage_steps,
                       args.plant)
        if args.reference is not None:
            result["passive_restarts"] = restarts_identical(
                args.reference, args.evidence,
                [int(s) for s in args.restart_steps.split(",") if s])
    except Refuse as exc:
        print(f"REFUSE: {exc}" + ("  STATUS PLANT-FIRED" if args.plant != "none" else ""),
              file=sys.stderr)
        return 1
    if args.plant != "none":
        print(f"REFUSE: plant {args.plant} stayed green", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(result["status"], args.evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
