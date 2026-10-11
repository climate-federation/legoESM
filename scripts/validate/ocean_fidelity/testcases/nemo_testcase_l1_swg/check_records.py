#!/usr/bin/env python3
"""Admit SWG stage-entry records and prove the instrumented run passive."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = "NEMO_L1_SWGSTG1"
NINT = 15
BOUNDS_AT = 10
STEPS = 10
STAGES = (1, 2, 3)
SENTINEL = 1.0e10
GROUPS = {
    "ssh_k", "u_k", "v_k", "r3t_k", "r3u_k", "r3v_k", "r3f",
    "u_before", "v_before", "utau_before", "vtau_before",
    "utau_now", "vtau_now",
}
GRID_KINDS = ("T", "U", "V", "W")


class Refusal(Exception):
    pass


def parse(path: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    raw = path.read_bytes()
    if raw[:16].decode("ascii", "replace").rstrip() != MAGIC:
        raise Refusal(f"{path.name}: wrong magic")
    off = 16 + 4 * NINT
    if len(raw) < off:
        raise Refusal(f"{path.name}: truncated header")
    head = list(struct.unpack(f"<{NINT}i", raw[16:off]))
    if head[0] != 1 or head[-1] != 64:
        raise Refusal(f"{path.name}: version/word {head[0]}/{head[-1]}")
    is0, js0, ie0, je0 = head[BOUNDS_AT:BOUNDS_AT + 4]
    groups = {}
    while off < len(raw):
        if off + 32 > len(raw):
            raise Refusal(f"{path.name}: truncated group header")
        name = raw[off:off + 16].decode("ascii", "replace").rstrip()
        rank, n1, n2, n3 = struct.unpack("<4i", raw[off + 16:off + 32])
        off += 32
        count = n1 * n2 * n3
        if rank != 2 or n3 != 1 or count <= 0 or off + 8 * count > len(raw):
            raise Refusal(f"{path.name}: bad group {name!r} extents")
        data = np.frombuffer(raw, "<f8", count, off).reshape(n2, n1)
        off += 8 * count
        if name in groups:
            raise Refusal(f"{path.name}: duplicate group {name!r}")
        owned = data[js0 - 1:je0, is0 - 1:ie0]
        if (owned.size == 0 or not np.all(np.isfinite(owned))
                or np.any(np.abs(owned) >= SENTINEL)):
            raise Refusal(f"{path.name}: {name!r} non-finite or sentinel")
        groups[name] = data
    if set(groups) != GROUPS:
        raise Refusal(f"{path.name}: missing {sorted(GROUPS-set(groups))} "
                      f"extra {sorted(set(groups)-GROUPS)}")
    return head, groups


def compare_outputs(run: Path, reference: Path) -> dict:
    """Require every numeric field in NEMO's four grid outputs to match bits."""
    import netCDF4
    result = {}
    for kind in GRID_KINDS:
        a_files = sorted(run.glob(f"*grid_{kind}*.nc"))
        b_files = sorted(reference.glob(f"*grid_{kind}*.nc"))
        if len(a_files) != 1 or len(b_files) != 1:
            raise Refusal(f"grid_{kind}: expected one run and reference file")
        with netCDF4.Dataset(a_files[0]) as a, netCDF4.Dataset(b_files[0]) as b:
            a.set_auto_mask(False); b.set_auto_mask(False)
            if set(a.variables) != set(b.variables):
                raise Refusal(f"grid_{kind}: variable sets differ")
            seen = []
            for name in sorted(a.variables):
                av, bv = np.asarray(a[name][:]), np.asarray(b[name][:])
                if av.shape != bv.shape or av.dtype != bv.dtype or av.tobytes() != bv.tobytes():
                    raise Refusal(f"grid_{kind}.{name}: instrument changed NEMO output")
                seen.append(name)
            result[kind] = seen
    return result


def check(evidence: Path, reference: Path) -> dict:
    stamp = evidence.parent / "manifest/legoesm_commit"
    if not stamp.is_file():
        raise Refusal(f"missing commit stamp {stamp}")
    commit = stamp.read_text().strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise Refusal(f"invalid commit stamp {commit!r}")
    files = sorted(evidence.glob("oracle_swgstage_kt*_s*.bin"))
    expected = [(kt, stage) for kt in range(1, STEPS + 1) for stage in STAGES]
    found = []
    hashes = {}
    shape = None
    for path in files:
        head, groups = parse(path)
        kt, stage = head[1], head[2]
        found.append((kt, stage))
        if path.name != f"oracle_swgstage_kt{kt:08d}_s{stage}.bin":
            raise Refusal(f"{path.name}: filename/header mismatch")
        shapes = {a.shape for a in groups.values()}
        if len(shapes) != 1 or (shape is not None and shapes != {shape}):
            raise Refusal(f"{path.name}: inconsistent shapes {shapes}")
        shape = shapes.pop()
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if found != expected:
        raise Refusal(f"stage sequence is {found}")
    if not list(evidence.glob("mesh_mask*.nc")):
        raise Refusal("missing mesh_mask")
    return {
        "legoesm_git_sha": commit, "records": hashes, "shape": list(shape),
        "output_identity": compare_outputs(evidence, reference),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--json", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        out = check(a.evidence, a.reference)
    except (Refusal, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    a.json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"ADMITTED {len(out['records'])} STAGE-ENTRY RECORDS; PASSIVE OUTPUTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
