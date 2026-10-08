#!/usr/bin/env python3
"""Admit (or refuse) a TSUNAMI lane-round-1 acquisition.

Every record is parsed from its OWN header; no size is predicted.  Refuses on:
a missing or extra record, a wrong magic/version, a filename/header step
disagreement, a payload whose length disagrees with its declared extents, a
missing required group, any non-finite value or |x| >= 1e10 sentinel, and any
bit difference between the reference and the instrumented run's own NEMO
output fields (the passivity criterion; TSUNAMI's step writes no restart).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

STEP_MAGIC = "NEMO_L1_TSUSTP1"
SPGTS_MAGIC = "NEMO_L1_SPGTS1"
STEP_NINT = 14     # tsunami_r1_step_record.F90 tsu_open
SPGTS_NINT = 15    # vortex_r12_spgts_terms.F90 spgts_r12_open
# header position of Nis0 (then Njs0, Nie0, Nje0): halos are not checked for
# finiteness, NEMO does not define every work array there
STEP_BOUNDS_AT = 9
SPGTS_BOUNDS_AT = 10
FULL_STEPS = 10    # kt = 1..10 carry every frame and the substep record
SENTINEL = 1.0e10
ENTRY = [f"e_{v}_{k}" for v in ("ssh", "uu_b", "vv_b", "r3t", "r3u", "r3v")
         for k in (1, 2, 3)] + ["e_r3f", "e_un_adv", "e_vn_adv"]
AFTER = ["a_ssh_aa", "a_uu_b_aa", "a_vv_b_aa"]
FULL = ENTRY + ["r_r3t_aa", "r_r3u_aa", "r_r3v_aa"] + AFTER + [
    "a_un_adv", "a_vn_adv", "a_uu_rhs_k1", "a_vv_rhs_k1",
    "a_uu_nn_k1", "a_vv_nn_k1"]
OUTPUT_FIELDS = ("sossheig", "souubaro", "somebaro")


class Refusal(Exception):
    pass


def parse(path: Path, magic: str, nint: int,
          bounds_at: int) -> tuple[list[int], dict]:
    """Return (header ints, {name: array}); raise Refusal on any mismatch."""
    raw = path.read_bytes()
    if raw[:16].decode("ascii", "replace").rstrip() != magic:
        raise Refusal(f"{path.name}: magic {raw[:16]!r} is not {magic}")
    off = 16 + 4 * nint
    if len(raw) < off:
        raise Refusal(f"{path.name}: truncated header")
    head = list(struct.unpack(f"<{nint}i", raw[16:off]))
    if head[0] != 1 or head[-1] != 64:
        raise Refusal(f"{path.name}: version {head[0]} / word {head[-1]}")
    is0, js0, ie0, je0 = head[bounds_at:bounds_at + 4]
    groups: dict[str, np.ndarray] = {}
    while off < len(raw):
        if off + 32 > len(raw):
            raise Refusal(f"{path.name}: truncated group header")
        name = raw[off:off + 16].decode("ascii", "replace").rstrip()
        rank, n1, n2, n3 = struct.unpack("<4i", raw[off + 16:off + 32])
        off += 32
        count = n1 * n2 * n3
        if rank not in (1, 2) or count <= 0 or off + 8 * count > len(raw):
            raise Refusal(f"{path.name}: group {name!r} extents {rank},{n1},{n2},{n3}")
        data = np.frombuffer(raw, "<f8", count, off)
        off += 8 * count
        if name in groups:
            raise Refusal(f"{path.name}: duplicate group {name!r}")
        arr = data.reshape((n2, n1) if rank == 2 else (n1,))
        owned = arr[js0 - 1:je0, is0 - 1:ie0] if rank == 2 else arr
        if (owned.size == 0 or not np.all(np.isfinite(owned))
                or np.any(np.abs(owned) >= SENTINEL)):
            raise Refusal(f"{path.name}: group {name!r} non-finite or sentinel")
        groups[name] = arr
    return head, groups


def step_of(path: Path) -> int:
    return int(path.stem.rsplit("kt", 1)[1])


def check(evidence: Path, reference: Path, steps: int) -> dict:
    out: dict = {"records": {}}
    stepfiles = sorted(evidence.glob("oracle_tsustep_kt*.bin"))
    if [step_of(p) for p in stepfiles] != list(range(1, steps + 1)):
        raise Refusal(f"step records are {[step_of(p) for p in stepfiles]}")
    shape = None
    for p in stepfiles:
        head, groups = parse(p, STEP_MAGIC, STEP_NINT, STEP_BOUNDS_AT)
        if head[1] != step_of(p):
            raise Refusal(f"{p.name}: header step {head[1]}")
        need = FULL if head[1] <= FULL_STEPS else AFTER
        missing = [g for g in need if g not in groups]
        if missing:
            raise Refusal(f"{p.name}: missing groups {missing}")
        shapes = {a.shape for a in groups.values()}
        if len(shapes) != 1 or (shape is not None and shapes != {shape}):
            raise Refusal(f"{p.name}: inconsistent shapes {shapes}")
        shape = shapes.pop()
        out["records"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    spgts = sorted(evidence.glob("oracle_spgts_kt*.bin"))
    if [step_of(p) for p in spgts] != list(range(1, FULL_STEPS + 1)):
        raise Refusal(f"substep records are {[step_of(p) for p in spgts]}")
    for p in spgts:
        head, groups = parse(p, SPGTS_MAGIC, SPGTS_NINT, SPGTS_BOUNDS_AT)
        if head[1] != step_of(p) or "o000_ssh_aa" not in groups:
            raise Refusal(f"{p.name}: header step {head[1]} or no exit frame")
        out["records"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    if not list(evidence.glob("mesh_mask*.nc")):
        raise Refusal("no mesh_mask*.nc in the evidence directory")
    out["output_identity"] = compare_outputs(evidence, reference)
    out["shape"] = list(shape)
    return out


def compare_outputs(run: Path, ref: Path) -> dict:
    """Bitwise identity of NEMO's own ssh/barotropic output, both runs."""
    import netCDF4

    seen = {}
    for field in OUTPUT_FIELDS:
        a = _find_field(run, field, netCDF4)
        b = _find_field(ref, field, netCDF4)
        if a.shape != b.shape or a.tobytes() != b.tobytes():
            raise Refusal(f"{field}: instrumented run differs from reference")
        seen[field] = list(a.shape)
    return seen


def _find_field(directory: Path, field: str, netCDF4) -> np.ndarray:
    hits = []
    for nc in sorted(directory.glob("*grid_[TUV]*.nc")):
        with netCDF4.Dataset(nc) as ds:
            if field in ds.variables:
                ds.set_auto_mask(False)
                hits.append(np.asarray(ds.variables[field][:], dtype="<f8"))
    if len(hits) != 1:
        raise Refusal(f"{field}: found in {len(hits)} files under {directory}")
    return hits[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        result = check(args.evidence, args.reference, args.steps)
    except Refusal as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    args.json.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"ADMITTED {len(result['records'])} records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
