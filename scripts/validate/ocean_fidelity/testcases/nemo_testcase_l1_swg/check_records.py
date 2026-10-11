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
        if (owned.size == 0 or not np.all(np.isfinite(data))
                or np.any(np.abs(data) >= SENTINEL)):
            raise Refusal(f"{path.name}: {name!r} non-finite or sentinel")
        groups[name] = data
    if set(groups) != GROUPS:
        raise Refusal(f"{path.name}: missing {sorted(GROUPS-set(groups))} "
                      f"extra {sorted(set(groups)-GROUPS)}")
    return head, groups


def _compare_dataset(a_path: Path, b_path: Path, label: str) -> list[str]:
    """Require every stored variable in two netCDF datasets to match bits."""
    import netCDF4
    with netCDF4.Dataset(a_path) as a, netCDF4.Dataset(b_path) as b:
        a.set_auto_mask(False); b.set_auto_mask(False)
        if set(a.variables) != set(b.variables):
            raise Refusal(f"{label}: variable sets differ")
        seen = []
        for name in sorted(a.variables):
            av, bv = np.asarray(a[name][:]), np.asarray(b[name][:])
            if (av.shape != bv.shape or av.dtype != bv.dtype
                    or av.tobytes() != bv.tobytes()):
                raise Refusal(f"{label}.{name}: instrument changed NEMO output")
            seen.append(name)
        return seen


def compare_outputs(run: Path, reference: Path) -> dict:
    """Require every numeric field in NEMO's four grid outputs to match bits."""
    result = {}
    for kind in GRID_KINDS:
        a_files = sorted(run.glob(f"*grid_{kind}*.nc"))
        b_files = sorted(reference.glob(f"*grid_{kind}*.nc"))
        if len(a_files) != 1 or len(b_files) != 1:
            raise Refusal(f"grid_{kind}: expected one run and reference file")
        result[kind] = _compare_dataset(
            a_files[0], b_files[0], f"grid_{kind}")
    return result


def compare_restarts(run: Path, reference: Path, steps: tuple[int, ...]) -> dict:
    """Require each requested daily restart to exist and match field bits."""
    result = {}
    for kt in steps:
        name = f"SWG_{kt:08d}_restart.nc"
        a_path, b_path = run / name, reference / name
        if not a_path.is_file() or not b_path.is_file():
            raise Refusal(f"restart kt={kt}: expected run and reference files")
        result[str(kt)] = _compare_dataset(a_path, b_path, f"restart.{kt}")
    return result


def check(evidence: Path, reference: Path, *,
          record_kts: tuple[int, ...] = tuple(range(1, STEPS + 1)),
          stages: tuple[int, ...] = STAGES,
          restart_kts: tuple[int, ...] = ()) -> dict:
    stamp = evidence.parent / "manifest/legoesm_commit"
    if not stamp.is_file():
        raise Refusal(f"missing commit stamp {stamp}")
    commit = stamp.read_text().strip()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise Refusal(f"invalid commit stamp {commit!r}")
    files = sorted(evidence.glob("oracle_swgstage_kt*_s*.bin"))
    expected = [(kt, stage) for kt in record_kts for stage in stages]
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
    out = {
        "legoesm_git_sha": commit, "records": hashes, "shape": list(shape),
        "output_identity": compare_outputs(evidence, reference),
    }
    if restart_kts:
        out["restart_identity"] = compare_restarts(
            evidence, reference, restart_kts)
    return out


def _int_csv(value: str) -> tuple[int, ...]:
    try:
        out = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not out or any(item < 1 for item in out) or len(set(out)) != len(out):
        raise argparse.ArgumentTypeError("integers must be positive and unique")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--record-kts", type=_int_csv,
                   default=tuple(range(1, STEPS + 1)))
    p.add_argument("--stages", type=_int_csv, default=STAGES)
    p.add_argument("--restart-kts", type=_int_csv, default=())
    a = p.parse_args(argv)
    try:
        out = check(a.evidence, a.reference, record_kts=a.record_kts,
                    stages=a.stages, restart_kts=a.restart_kts)
    except (Refusal, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    a.json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"ADMITTED {len(out['records'])} STAGE-ENTRY RECORDS; PASSIVE OUTPUTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
