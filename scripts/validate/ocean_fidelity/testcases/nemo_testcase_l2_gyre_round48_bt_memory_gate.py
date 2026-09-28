#!/usr/bin/env python3
"""Fail-closed gate for the round-48 GYRE kt=1 -> kt=2 memory record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import require


ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round48/oracle_bt_memory"
)
MAGIC = "NEMO_L2_R48BTM1"
DIMS = (36, 26)
HEADER_FIELDS = (
    "version", "kt", "phase", "Kbb", "Kmm", "Kaa", "jpi", "jpj",
    "ntsi", "ntei", "ntsj", "ntej", "bits",
)
HISTORIES = ("ubb_e", "ub_e", "vbb_e", "vb_e", "sshbb_e", "sshb_e")
REQUIRED = set(HISTORIES) | {
    "un_e", "vn_e", "sshn_e", "un_adv", "vn_adv",
    "ubar_Kmm", "vbar_Kmm", "ssh_Kmm",
    "ubar_Kaa", "vbar_Kaa", "ssh_Kaa",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _xy(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float64).reshape(DIMS, order="F").T.copy()


def read_record(path: Path, *, plant: str | None = None) -> dict:
    """Read every named payload through physical EOF; reject extras/duplicates."""
    with path.open("rb") as stream:
        raw_magic = stream.read(16)
        require(len(raw_magic) == 16, f"{path}: short magic")
        magic = raw_magic.decode("ascii").rstrip()
        raw_header = stream.read(13 * 4)
        require(len(raw_header) == 13 * 4, f"{path}: short header")
        values = list(struct.unpack("=13i", raw_header))
        if plant == "header":
            values[2] = 9
        header = dict(zip(HEADER_FIELDS, values, strict=True))
        require(magic == MAGIC, f"{path}: bad magic {magic!r}")
        require(
            header["version"] == 1 and header["bits"] == 64,
            f"{path}: wrong version/dtype {header}",
        )
        require((header["jpi"], header["jpj"]) == DIMS, f"{path}: wrong grid")
        require(
            (header["ntsi"], header["ntei"], header["ntsj"], header["ntej"])
            == (3, 34, 3, 24),
            f"{path}: wrong owned bounds {header}",
        )
        require(
            (header["kt"], header["phase"]) in ((1, 1), (2, 2)),
            f"{path}: wrong kt/phase {header}",
        )
        require(header["Kbb"] == header["Kmm"], f"{path}: RK3 Kbb != Kmm")
        arrays = {}
        while True:
            raw_name = stream.read(16)
            if not raw_name:
                break
            require(len(raw_name) == 16, f"{path}: truncated field name")
            name = raw_name.decode("ascii").rstrip()
            require(name not in arrays, f"{path}: duplicate field {name!r}")
            raw_shape = stream.read(16)
            require(len(raw_shape) == 16, f"{path}: short shape for {name}")
            rank, n1, n2, n3 = struct.unpack("=4i", raw_shape)
            require((rank, n1, n2, n3) == (2, *DIMS, 1), f"{path}: bad {name} shape")
            raw = stream.read(8 * n1 * n2)
            if plant == "truncation" and name == "ubb_e":
                raw = raw[:-8]
            require(len(raw) == 8 * n1 * n2, f"{path}: short payload for {name}")
            value = _xy(raw)
            require(np.isfinite(value).all(), f"{path}: non-finite {name}")
            arrays[name] = value
    require(arrays.keys() == REQUIRED, f"{path}: field contract moved: {sorted(arrays)}")
    return {"header": header, "arrays": arrays}


def _bit_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.shape == right.shape
        and np.array_equal(
            np.ascontiguousarray(left).view(np.uint64),
            np.ascontiguousarray(right).view(np.uint64),
        )
    )


def run(root: Path, *, expect_commit: str, plant: str | None) -> dict:
    stamp = worktree_stamp()
    require(stamp.get("clean") is True,
            "barotropic-memory gate requires a clean producer worktree")
    expected = "0" * 40 if plant == "stamp" else expect_commit.lower()
    require(
        len(expected) == 40 and stamp["commit"].lower() == expected,
        f"commit stamp mismatch: {stamp['commit']} != {expected}",
    )
    end_path = root / "oracle_bt_memory_kt00000001_end.bin"
    start_path = root / "oracle_bt_memory_kt00000002_start.bin"
    end = read_record(
        end_path, plant=plant if plant in {"header", "truncation"} else None
    )
    start = read_record(start_path)
    ea, sa = end["arrays"], start["arrays"]
    if plant == "boundary":
        sa["ub_e"][3, 3] = np.nextafter(sa["ub_e"][3, 3], np.inf)
    if plant == "seed":
        sa["un_e"][3, 3] += 1.0
    if plant == "reset":
        sa["un_adv"][3, 3] = 1.0

    boundary = {name: _bit_equal(ea[name], sa[name]) for name in HISTORIES}
    end_commit = {
        "un_e=ubar_Kaa": _bit_equal(ea["un_e"], ea["ubar_Kaa"]),
        "vn_e=vbar_Kaa": _bit_equal(ea["vn_e"], ea["vbar_Kaa"]),
        "sshn_e=ssh_Kaa": _bit_equal(ea["sshn_e"], ea["ssh_Kaa"]),
    }
    start_seed = {
        "un_e=ubar_Kmm": _bit_equal(sa["un_e"], sa["ubar_Kmm"]),
        "vn_e=vbar_Kmm": _bit_equal(sa["vn_e"], sa["vbar_Kmm"]),
        "sshn_e=ssh_Kmm": _bit_equal(sa["sshn_e"], sa["ssh_Kmm"]),
    }
    reset = {
        "un_adv_zero": bool(np.count_nonzero(sa["un_adv"]) == 0),
        "vn_adv_zero": bool(np.count_nonzero(sa["vn_adv"]) == 0),
        "ubar_Kaa_zero": bool(np.count_nonzero(sa["ubar_Kaa"]) == 0),
        "vbar_Kaa_zero": bool(np.count_nonzero(sa["vbar_Kaa"]) == 0),
        "ssh_Kaa_zero": bool(np.count_nonzero(sa["ssh_Kaa"]) == 0),
    }
    require(all(boundary.values()), f"cross-step histories moved: {boundary}")
    require(all(end_commit.values()), f"kt1 final commit mismatch: {end_commit}")
    require(all(start_seed.values()), f"kt2 current seed mismatch: {start_seed}")
    require(all(reset.values()), f"kt2 accumulator reset mismatch: {reset}")
    return {
        "format": "nemo-testcase-l2-gyre-round48-bt-memory-v1",
        "worktree": stamp,
        "headers": {"kt1_end": end["header"], "kt2_start": start["header"]},
        "record_sha256": {
            end_path.name: _sha256(end_path), start_path.name: _sha256(start_path)
        },
        "history_boundary_bit_identical": boundary,
        "kt1_final_commit_bit_identical": end_commit,
        "kt2_current_seed_bit_identical": start_seed,
        "kt2_reset_exact": reset,
        "plant": plant,
        "status": "PASS",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument(
        "--plant", choices=("header", "truncation", "boundary", "seed", "reset", "stamp")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run(args.root, expect_commit=args.expect_commit, plant=args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print("STATUS", report["status"])
    return 1 if args.plant else 0


if __name__ == "__main__":
    raise SystemExit(main())
