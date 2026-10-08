#!/usr/bin/env python3
"""Admit the rank-complete, self-describing ORCA2 round-95 SPG record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = "NEMO_L4_R95SPG1"
ENTRY = (
    "ssh_frc", "zu_frc", "zv_frc", "un_e", "vn_e", "ub_e", "vb_e",
    "ubb_e", "vbb_e", "sshn_e", "sshb_e", "sshbb_e", "hu_e", "hv_e",
    "hur_e", "hvr_e", "zCdU_u", "zCdU_v", "wgtbtp1", "wgtbtp2",
    "entry_sc",
)
SUBSTEP = (
    "ua_ext", "va_ext", "sshp2_mid", "htp2_e", "hup2_e", "hvp2_e",
    "ext_coef", "zhU", "zhV", "ssha_e", "un_adv", "vn_adv", "sshu_a",
    "sshv_a", "sshp2_bck", "zu_spg", "zv_spg", "bck_coef", "cor_u",
    "cor_v", "trd_u", "trd_v", "ua_new", "va_new", "hu_e", "hv_e",
    "hur_e", "hvr_e", "uub_sum", "vvb_sum", "ssh_sum", "sum_coef",
)
EXIT = ("un_adv", "vn_adv", "uu_b_aa", "vv_b_aa", "ssh_aa")
PLANTS = (
    "none", "header", "field-name", "field-dims", "truncation",
    "missing-frame", "swapped-rank", "restart-byte",
)


class Refusal(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def required_names(icycle: int) -> set[str]:
    names = {f"i000_{name}" for name in ENTRY}
    for substep in range(1, icycle + 1):
        names.update(f"j{substep:03d}_{name}" for name in SUBSTEP)
    names.update(f"o000_{name}" for name in EXIT)
    return names


def read_record(path: Path, plant: str = "none", *, expected_kt: int = 1,
                expected_magic: str = MAGIC) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-1]
    require(len(raw) >= 88, f"{path.name}: record too short")
    magic = raw[:16].decode("ascii", "replace").rstrip(" \x00")
    if plant == "header":
        magic = "X" + magic[1:]
    require(magic == expected_magic, f"{path.name}: bad magic {magic!r}")
    header = list(struct.unpack_from("=18i", raw, 16))
    (version, kt, kbb, kmm, kaa, krhs, rank, nx, ny, nz, icycle,
     nimpp, njmpp, ntsi, ntsj, ntei, ntej, bits) = header
    if plant == "swapped-rank":
        rank = 1 - rank
    require(version == 1, f"{path.name}: unsupported version {version}")
    require(kt == expected_kt, f"{path.name}: unexpected timestep {kt}")
    require(bits == 64, f"{path.name}: precision is not fp64")
    require(nx > 0 and ny > 0 and nz > 0 and icycle > 0,
            f"{path.name}: non-positive header extent")
    require(1 <= ntsi <= ntei <= nx and 1 <= ntsj <= ntej <= ny,
            f"{path.name}: owned bounds outside local domain")
    offset = 88
    groups: dict[str, dict] = {}
    group_index = 0
    while offset < len(raw):
        require(offset + 32 <= len(raw),
                f"{path.name}: truncated group header at byte {offset}")
        name = raw[offset:offset + 16].decode("ascii", "replace").rstrip(" \x00")
        offset += 16
        ndim, n1, n2, n3 = struct.unpack_from("=4i", raw, offset)
        offset += 16
        if plant == "field-name" and group_index == 0:
            name = "i000_missing"
        if plant == "field-dims" and group_index == 0:
            n1 += 1
        require(name and name not in groups,
                f"{path.name}: empty or duplicate group {group_index} {name!r}")
        require(ndim in (1, 2), f"{path.name}: {name!r} has rank {ndim}")
        require(n1 > 0 and n2 > 0 and n3 > 0,
                f"{path.name}: {name!r} has non-positive extent")
        require((ndim == 1 and n2 == n3 == 1) or (ndim == 2 and n3 == 1),
                f"{path.name}: {name!r} has inconsistent dimensions")
        count = n1 * (n2 if ndim == 2 else 1)
        end = offset + 8 * count
        require(end <= len(raw),
                f"{path.name}: {name!r} declares {count} doubles but is truncated")
        values = np.frombuffer(raw, dtype="=f8", count=count, offset=offset)
        require(bool(np.isfinite(values).all()), f"{path.name}: nonfinite {name!r}")
        groups[name] = {"rank": ndim, "shape": [n1] if ndim == 1 else [n1, n2]}
        offset = end
        group_index += 1
    require(offset == len(raw), f"{path.name}: trailing bytes")
    if plant == "missing-frame":
        for name in [key for key in groups if key.startswith(f"j{icycle:03d}_")]:
            groups.pop(name)
    expected = required_names(icycle)
    missing = sorted(expected - groups.keys())
    unexpected = sorted(groups.keys() - expected)
    require(not missing,
            f"{path.name}: missing group(s) {missing[:6]} ({len(missing)} total)")
    require(not unexpected,
            f"{path.name}: unexpected group(s) {unexpected[:6]} ({len(unexpected)} total)")
    return {
        "rank": rank, "kt": kt, "levels": [kbb, kmm, kaa, krhs],
        "shape": [nx, ny, nz], "icycle": icycle,
        "origin": [nimpp, njmpp], "owned": [ntsi, ntsj, ntei, ntej],
        "frames": icycle + 2, "groups": len(groups), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def run(root: Path, baseline: Path, plant: str, *, expected_kt: int = 1,
        prefix: str = "oracle_r95_spg", expected_magic: str = MAGIC) -> dict:
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    for expected_rank in (0, 1):
        path = root / (
            f"{prefix}_rank{expected_rank:04d}_kt{expected_kt:08d}.bin")
        applied = plant if expected_rank == 0 and plant not in ("restart-byte", "none") else "none"
        record = read_record(
            path, applied, expected_kt=expected_kt,
            expected_magic=expected_magic)
        require(record["rank"] == expected_rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = record["origin"]
        ntsi, ntsj, ntei, ntej = record["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: slab outside global domain")
        coverage[j0:j1, i0:i1] += 1
        records.append(record)
    require(bool(np.all(coverage == 1)), "rank slabs do not cover domain exactly once")
    require(records[0]["icycle"] == records[1]["icycle"], "rank icycle values differ")
    restarts = sorted(baseline.glob("ORCA2_000000??_restart_????.nc"))
    require(len(restarts) == 20, "baseline lacks 20 ocean restarts")
    restart_rows = []
    for index, source in enumerate(restarts):
        target = root / source.name
        require(target.is_file(), f"missing target restart {source.name}")
        left, right = source.read_bytes(), target.read_bytes()
        if plant == "restart-byte" and index == 0:
            right = right[:-1] + bytes([right[-1] ^ 1])
        require(left == right, f"write-only acquisition changed {source.name}")
        restart_rows.append({"name": source.name,
                             "sha256": hashlib.sha256(right).hexdigest()})
    return {
        "status": "PASS_R95_SPGTS_ADMISSION",
        "rank_coverage": "exactly-once", "records": records,
        "terminal_restart_comparisons": restart_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-kt", type=int, default=1)
    parser.add_argument("--prefix", default="oracle_r95_spg")
    parser.add_argument("--magic", default=MAGIC)
    args = parser.parse_args()
    try:
        result = run(
            args.root, args.baseline, args.plant,
            expected_kt=args.expected_kt, prefix=args.prefix,
            expected_magic=args.magic)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, UnicodeDecodeError, ValueError, Refusal) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R95_SPGTS_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
