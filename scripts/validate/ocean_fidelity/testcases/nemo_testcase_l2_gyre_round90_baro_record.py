#!/usr/bin/env python3
"""Admit the Round-90 correction-site NEMO record and its final-add control."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

MAGIC = "NEMO_L2_R90BARO1"
DIMS = (36, 26, 31)
HEADER_NAMES = (
    "version", "kt", "stage", "Kaa", "jpi", "jpj", "jpk", "jpkm1",
    "ntsi", "ntei", "ntsj", "ntej", "bits",
)
REQUIRED = {
    "baro_raw_u", "baro_raw_v", "baro_target_u", "baro_target_v",
    "baro_zub", "baro_zvb", "baro_e3u_0", "baro_e3v_0",
    "baro_r1_hu_0", "baro_r1_hv_0", "baro_umask", "baro_vmask",
    "baro_final_u", "baro_final_v",
}
OWNED_2D = {"baro_zub", "baro_zvb"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _xy(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F").T


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F").transpose(1, 0, 2)


def read_record(path: Path) -> dict[str, object]:
    arrays: dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        raw_magic = handle.read(16)
        require(len(raw_magic) == 16, f"{path}: short magic")
        magic = raw_magic.decode("ascii").rstrip()
        raw_header = handle.read(4 * len(HEADER_NAMES))
        require(len(raw_header) == 4 * len(HEADER_NAMES), f"{path}: short header")
        header = dict(zip(
            HEADER_NAMES, struct.unpack(f"={len(HEADER_NAMES)}i", raw_header),
            strict=True))
        require(magic == MAGIC, f"{path}: wrong magic {magic!r}")
        require((header["version"], header["kt"], header["stage"],
                 header["jpi"], header["jpj"], header["jpk"],
                 header["jpkm1"], header["bits"]) ==
                (1, 2, 1, *DIMS, 30, 64), f"{path}: wrong header {header}")
        require((header["ntsi"], header["ntei"], header["ntsj"],
                 header["ntej"]) == (3, 34, 3, 24),
                f"{path}: wrong owned bounds {header}")
        while True:
            raw_name = handle.read(16)
            if not raw_name:
                break
            require(len(raw_name) == 16, f"{path}: truncated field name")
            name = raw_name.decode("ascii").rstrip()
            require(name not in arrays, f"{path}: duplicate field {name}")
            require(name in REQUIRED, f"{path}: unexpected field {name}")
            raw_shape = handle.read(16)
            require(len(raw_shape) == 16, f"{path}: short shape for {name}")
            rank, n1, n2, n3 = struct.unpack("=4i", raw_shape)
            require(rank in (2, 3), f"{path}: invalid rank for {name}")
            owned = (header["ntei"] - header["ntsi"] + 1,
                     header["ntej"] - header["ntsj"] + 1)
            expected = ((*owned, 1) if name in OWNED_2D else
                        (*DIMS[:2], 1) if rank == 2 else DIMS)
            expected_rank = 2 if name in OWNED_2D or name in {
                "baro_target_u", "baro_target_v", "baro_r1_hu_0",
                "baro_r1_hv_0",
            } else 3
            require(rank == expected_rank and (n1, n2, n3) == expected,
                    f"{path}: wrong shape for {name}: {(rank, n1, n2, n3)}")
            count = n1 * n2 * (1 if rank == 2 else n3)
            raw = handle.read(8 * count)
            require(len(raw) == 8 * count, f"{path}: short payload for {name}")
            values = np.frombuffer(raw, dtype=np.float64)
            require(np.isfinite(values).all(), f"{path}: non-finite {name}")
            arrays[name] = (_xy(values, n1, n2) if rank == 2 else
                            _xyz(values, n1, n2, n3))
    require(set(arrays) == REQUIRED,
            f"{path}: field contract differs: missing={sorted(REQUIRED-set(arrays))}, "
            f"extra={sorted(set(arrays)-REQUIRED)}")
    return {"header": header, "arrays": arrays}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(args) -> dict[str, object]:
    record = read_record(args.record)
    arrays = record["arrays"]
    stamp_words = args.stamp.read_text().split()
    require(len(stamp_words) == 3, "record stamp must contain digest, commit, name")
    require(stamp_words[1].lower() == args.expect_commit.lower(),
            "record producer commit differs")
    require(stamp_words[2] == args.record.name, "record stamp names another file")
    require(stamp_words[0] == _sha256(args.record), "record digest differs")
    faces = {}
    for face in ("u", "v"):
        raw = arrays[f"baro_raw_{face}"][2:-2, 2:-2]
        correction = arrays["baro_zub" if face == "u" else "baro_zvb"]
        mask = arrays[f"baro_{face}mask"][2:-2, 2:-2]
        final = arrays[f"baro_final_{face}"][2:-2, 2:-2].copy()
        active = mask > 0.5
        replay = raw + correction[..., None] * mask
        if args.plant == "final-ulp" and face == "u":
            candidates = np.argwhere(active & (final != 0.0))
            require(candidates.size > 0, "final plant found no nonzero wet cell")
            at = tuple(candidates[0])
            final[at] = np.nextafter(final[at], np.inf)
        unequal = replay[active].view(np.uint64) != final[active].view(np.uint64)
        faces[face] = {
            "n": int(active.sum()),
            "n_unequal": int(np.count_nonzero(unequal)),
            "max_abs": float(np.max(np.abs(replay[active] - final[active]))),
            "bit_exact": bool(not np.any(unequal)),
            "target_nonzero_wet": int(np.count_nonzero(
                arrays[f"baro_target_{face}"][2:-2, 2:-2][active[..., 0]])),
        }
    exact = all(row["bit_exact"] for row in faces.values())
    if args.plant:
        require(not exact, "final-add plant stayed green")
        status = "PLANT_FIRED"
    else:
        require(exact, "recorded final add does not replay bit-exactly")
        require(all(row["target_nonzero_wet"] > 0 for row in faces.values()),
                "correction-site target is still empty")
        status = "READY"
    return {
        "format": "nemo-testcase-l2-gyre-round90-baro-record-v1",
        "worktree": worktree_stamp(),
        "status": status,
        "record_sha256": _sha256(args.record),
        "producer_commit": stamp_words[1],
        "header": record["header"],
        "faces": faces,
        "plant": args.plant,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--stamp", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", choices=("final-ulp",))
    args = parser.parse_args()
    report = measure(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("ROUND90_BARO_RECORD", report["status"])
    return 1 if args.plant else 0


if __name__ == "__main__":
    raise SystemExit(main())
