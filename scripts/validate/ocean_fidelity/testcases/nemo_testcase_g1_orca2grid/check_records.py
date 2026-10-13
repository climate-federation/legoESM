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
SUB_PLANTS = ("missing-substep",)       # needs --substep-steps
# the least each frame must hold for the round-4 walk (more is recorded)
SUB_ENTRY = ("sshn_e", "un_e", "vn_e", "zu_frc", "zv_frc", "ffu_nw", "ffu_ne",
             "ffu_sw", "ffu_se", "ffv_nw", "ffv_ne", "ffv_sw", "ffv_se")
SUB_STEP = ("ssha_e", "ua_new", "va_new", "zhU", "zhV", "hup2_e", "hvp2_e",
            "cor_u", "cor_v")
SUB_EXIT = ("ssh_aa", "uu_b_aa", "vv_b_aa", "un_adv", "vn_adv")
SUB_MAGIC = b"NEMO_G1_SPGTS1  "
SUB_HEADER = ("version", "step", "Nbb", "Nmm", "Naa", "Nrhs", "jpi", "jpj", "jpk",
              "icycle", "Nis0", "Njs0", "Nie0", "Nje0", "word_bits")


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


def substep_path(root: Path, step: int) -> Path:
    return root / f"oracle_g1spgts_kt{step:08d}.bin"


def read_substeps(blob: bytes, label: str):
    """One solver substep stream (g1_spgts_record.F90): header and every
    named array, sized by its own record header (arrays differ in extent)."""
    require(blob[:16] == SUB_MAGIC, f"{label}: bad magic {blob[:16]!r}")
    header = dict(zip(SUB_HEADER, struct.unpack("<15i", blob[16:76])))
    require(header["word_bits"] == 64, f"{label}: word size is not 64 bits")
    offset, fields = 76, {}
    while offset < len(blob):
        require(offset + 32 <= len(blob), f"{label}: truncated record header")
        name = blob[offset:offset + 16].decode("ascii").strip()
        rank, n1, n2, n3 = struct.unpack("<4i", blob[offset + 16:offset + 32])
        count = n1 * n2 * n3
        offset += 32
        require(rank in (1, 2) and count > 0, f"{label}: {name} has rank {rank}")
        require(offset + 8 * count <= len(blob),
                f"{label}: truncated payload of {name}")
        require(name not in fields, f"{label}: duplicate field {name}")
        data = np.frombuffer(blob, "<f8", count, offset)
        fields[name] = data.reshape((n2, n1)) if rank == 2 else data
        offset += 8 * count
    return header, fields


def check_substeps(evidence: Path, twin: Path | None, steps: list[int],
                   plant: str = "none") -> dict:
    """The substep streams exist at exactly the listed steps, are finite,
    hold the entry frame, one frame per substep of their own header's count
    and the exit frame, and equal the twin run's byte for byte."""
    out = {}
    wanted = list(steps) + ([max(steps) + 1] if plant == "missing-substep" else [])
    for step in wanted:
        path = substep_path(evidence, step)
        require(path.is_file(), f"missing substep record {path.name}")
        blob = path.read_bytes()
        header, fields = read_substeps(blob, path.name)
        require(header["step"] == step, f"{path.name}: header step {header['step']}")
        require(header["icycle"] >= 1, f"{path.name}: no substep ({header['icycle']})")
        for frame, need in (("i000", SUB_ENTRY), ("j001", SUB_STEP), ("o000", SUB_EXIT)):
            lacking = [n for n in need if f"{frame}_{n}" not in fields]
            require(not lacking, f"{path.name}: frame {frame} lacks {lacking}")
        for n in ("i000_sshn_e", "j001_ssha_e", "o000_ssh_aa"):
            require(fields[n].shape == (header["jpj"], header["jpi"]),
                    f"{path.name}: {n} is not a (jpj, jpi) field of its own header")
        frames = {name.split("_", 1)[0] for name in fields}
        want = {"i000", "o000"} | {f"j{n:03d}" for n in range(1, header["icycle"] + 1)}
        require(frames == want, f"{path.name}: frames {sorted(frames)[:4]}.. != "
                                f"entry, {header['icycle']} substeps, exit")
        first = sorted(n.split("_", 1)[1] for n in fields if n.startswith("j001_"))
        for jn in range(2, header["icycle"] + 1):
            names = sorted(n.split("_", 1)[1] for n in fields
                           if n.startswith(f"j{jn:03d}_"))
            require(names == first, f"{path.name}: substep {jn} field list moved")
        for name, data in fields.items():
            require(bool(np.all(np.isfinite(data))), f"{path.name}: {name} non-finite")
        if twin is not None and step_path(twin, step).is_file():
            # the twin ran this step: its substep stream must exist and agree
            require(substep_path(twin, step).is_file(),
                    f"{path.name}: the twin run has no substep record")
            require(substep_path(twin, step).read_bytes() == blob,
                    f"{path.name}: twin run differs")
        out[step] = {"icycle": header["icycle"], "n_fields": len(fields),
                     "substep_fields": first}
    stray = sorted(p.name for p in evidence.glob("oracle_g1spgts_kt*.bin")
                   if p not in {substep_path(evidence, s) for s in steps})
    require(not stray, f"substep records at unlisted steps: {stray[:3]}")
    return out


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
    parser.add_argument("--substep-steps", default="",
                        help="steps that must hold a solver substep stream")
    parser.add_argument("--plant", default="none", choices=PLANTS + SUB_PLANTS)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        result = check(args.evidence, args.twin, args.steps, args.stage_steps,
                       "none" if args.plant == "missing-substep" else args.plant)
        if args.substep_steps:
            result["substeps"] = check_substeps(
                args.evidence, args.twin,
                [int(s) for s in args.substep_steps.split(",") if s], args.plant)
        require(args.plant != "missing-substep" or bool(args.substep_steps),
                "the missing-substep plant needs --substep-steps")
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
