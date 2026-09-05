#!/usr/bin/env python3
"""Schema, canonical-slot, twin, and inherited-record gate for Phase 2p ZDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np

NX, NY, NZ = 94, 152, 31
HALO = 2
NAME = "oracle_zdf_sh2_operands_kt00000001.bin"
MAGIC = "NEMO_L4_ZSH2_1"
HEADER_INTS = 13
HEADER_BYTES = 16 + 4 * HEADER_INTS
FIELDS_3D = (
    "sh2", "avm_k_pre", "avt_k_pre", "en_pre", "rn2", "rn2b",
    "u_Kbb", "u_Kmm", "v_Kbb", "v_Kmm",
    "e3uw_Kbb", "e3uw_Kmm", "e3vw_Kbb", "e3vw_Kmm",
    "umask", "vmask", "wumask", "wvmask", "gdepw_Kmm",
    "e3t_Kmm", "e3w_Kmm",
)
FIELDS_2D = ("taum", "fr_i", "rCdU_bot", "mbkt_real")
GRID_3D = (
    "W", "W", "W", "W", "W", "W", "U", "U", "V", "V",
    "X", "X", "Y", "Y", "U", "V", "X", "Y", "W", "T", "W",
)


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


def _rank0_masks(domain: Path) -> dict[str, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(domain) as dataset:
        bottom = np.asarray(dataset["bottom_level"][:], dtype=np.int32)
    level = np.arange(NZ)[None, None, :]
    t_global = level < bottom[..., None]
    u_global = t_global * np.roll(t_global, -1, axis=1)
    v_global = np.zeros_like(t_global)
    v_global[:-1] = t_global[:-1] * t_global[1:]
    w_global = np.zeros_like(t_global)
    w_global[..., 0] = t_global[..., 0]
    w_global[..., 1:] = t_global[..., 1:] * t_global[..., :-1]
    wu_global = np.zeros_like(t_global)
    wv_global = np.zeros_like(t_global)
    wu_global[..., 0] = u_global[..., 0]
    wv_global[..., 0] = v_global[..., 0]
    wu_global[..., 1:] = u_global[..., 1:] * u_global[..., :-1]
    wv_global[..., 1:] = v_global[..., 1:] * v_global[..., :-1]

    masks: dict[str, np.ndarray] = {}
    for grid, source in {
        "T": t_global, "U": u_global, "V": v_global,
        "W": w_global, "X": wu_global, "Y": wv_global,
    }.items():
        local = np.zeros((NX, NY, NZ), dtype=bool)
        local[HALO:-HALO, HALO:-HALO] = source[:, : NX - 2 * HALO].transpose(1, 0, 2)
        masks[grid] = local
    return masks


def read_zdf(path: Path, domain: Path) -> dict[str, object]:
    n2, n3 = NX * NY, NX * NY * NZ
    payload = len(FIELDS_3D) * n3 + len(FIELDS_2D) * n2
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack(f"={HEADER_INTS}i", handle.read(4 * HEADER_INTS))
        values = np.fromfile(handle, dtype=np.float64)
    expected = (1, 1, 1, 2, 3, NX, NY, NZ, 64,
                len(FIELDS_3D), len(FIELDS_2D), payload, 1)
    require(magic == MAGIC, f"bad magic {magic!r}")
    require(header == expected, f"bad header {header!r}")
    require(values.size == payload, f"derived payload {values.size} != {payload}")
    require(path.stat().st_size == HEADER_BYTES + 8 * payload,
            "derived EOF size mismatch")
    require(np.isfinite(values).all(), "non-finite payload")

    masks = _rank0_masks(domain)
    cursor = 0
    arrays: dict[str, np.ndarray] = {}
    canonical: dict[str, str] = {}
    for name, grid in zip(FIELDS_3D, GRID_3D, strict=True):
        field = values[cursor:cursor + n3].reshape((NX, NY, NZ), order="F")
        cursor += n3
        require(np.all(field[~masks[grid]] == 0.0),
                f"{name}: nonzero unowned/halo/land slot")
        arrays[name] = field
        canonical[name] = f"ZERO_OUTSIDE_{grid}"
    t2 = masks["T"][..., 0]
    for name in FIELDS_2D:
        field = values[cursor:cursor + n2].reshape((NX, NY), order="F")
        cursor += n2
        require(np.all(field[~t2] == 0.0), f"{name}: nonzero inactive T slot")
        arrays[name] = field
        canonical[name] = "ZERO_OUTSIDE_T_SURFACE"
    require(cursor == values.size, "parser did not consume payload")
    return {
        "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
        "magic": magic, "header": list(header), "derived_payload_f64": payload,
        "fields_3d": list(FIELDS_3D), "fields_2d": list(FIELDS_2D),
        "canonical_slots": canonical,
    }


def validate(a: Path, b: Path, inherited: Path, domain: Path) -> dict[str, object]:
    ia = {p.name for p in a.glob("oracle_*.bin")}
    ib = {p.name for p in b.glob("oracle_*.bin")}
    old = {p.name for p in inherited.glob("oracle_*.bin")}
    require(ia == ib, f"twin inventories differ: {sorted(ia ^ ib)}")
    require(ia == old | {NAME}, "candidate inventory is not inherited V2 plus ZDF")
    twin_rows = []
    for name in sorted(ia):
        require((a / name).read_bytes() == (b / name).read_bytes(),
                f"twin record differs: {name}")
        twin_rows.append({"file": name, "sha256": sha256(a / name),
                          "status": "EXACT_BYTES"})
    inherited_rows = []
    for name in sorted(old):
        require((a / name).read_bytes() == (inherited / name).read_bytes(),
                f"inherited record differs: {name}")
        inherited_rows.append({"file": name, "sha256": sha256(a / name),
                               "status": "EXACT_BYTES"})
    return {
        "status": "PASS", "schema": read_zdf(a / NAME, domain),
        "twin_raw_exact": len(twin_rows), "twin_total": len(twin_rows),
        "inherited_raw_exact": len(inherited_rows),
        "inherited_total": len(inherited_rows), "records": twin_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--inherited", type=Path, required=True)
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_a = args.run_a
    temp: tempfile.TemporaryDirectory[str] | None = None
    if args.plant:
        temp = tempfile.TemporaryDirectory(prefix="orca2-l4-zdf-plant-")
        planted = Path(temp.name)
        for source in args.run_a.glob("oracle_*.bin"):
            (planted / source.name).symlink_to(source)
        target = planted / NAME
        target.unlink()
        shutil.copyfile(args.run_a / NAME, target)
        with target.open("r+b") as handle:
            handle.seek(HEADER_BYTES)
            handle.write(struct.pack("=d", 1.0))  # first halo payload slot
        run_a = planted
    try:
        result = validate(run_a, args.run_b, args.inherited, args.domain)
    except GateError as error:
        print(f"FAIL: {error}")
        return 1
    finally:
        if temp is not None:
            temp.cleanup()
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
