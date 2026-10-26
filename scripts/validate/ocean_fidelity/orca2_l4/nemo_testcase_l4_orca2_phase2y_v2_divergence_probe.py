#!/usr/bin/env python3
"""P2Y-1 cross-root divergence probe: VARIANT_ORACLE_V2 vs VARIANT_ORACLE_ORCA1ICE.

Byte-compares the 101 oracle_*.bin streams common to both pinned roots (the
90 Phase-1 streams plus the 11 orca2_l4 extensions; the ORCA1-ice root's
additional 15 TAILORED SI3 streams have no V2 counterpart and are excluded).
Reports, per stream: byte-identical or not, and if not, the first differing
byte offset.  For the chronologically-first record (`oracle_step_entry_
kt00000001.bin`, the write-only Nbb dump at the top of `stp_RK3` --
`nemo502_MY_SRC/stprk3.F90.patch` hunk documented in the Phase-2y receipt),
also decodes the header (jpi,jpj,jpk,jpts) and localizes the difference to
its field (ts/uu/vv/ssh) and, if it is ssh, prints the max/mean |delta| so the
size of the divergence can be sanity-checked (Diagnosis Discipline: a probe
number is not reported without a plausibility check).

This is a read-only comparison; it does not judge PASS/FAIL (the two roots
are EXPECTED to diverge -- P2Y-1 asks only for the first such record and
why).  Exit 0 always; the JSON result is the citable artifact.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_een_discriminator_gate as een,
    nemo_testcase_l4_orca2_o1_acquisition_gate as o1,
    nemo_testcase_l4_orca2_phase2b_exchange_gate as surface,
    nemo_testcase_l4_orca2_phase2o_bbl_gate as bbl,
    nemo_testcase_l4_orca2_phase2s_zdf_acquisition_gate as zdf,
    nemo_testcase_l4_orca2_phase2u_tke_acquisition_gate as tke,
    nemo_testcase_l4_orca2_wzv_gate as wzv,
)

STEP_ENTRY_1 = "oracle_step_entry_kt00000001.bin"
# stprk3.F90.patch: WRITE(itraj) cl_magic,1,kstp,Nbb,jpi,jpj,jpk,jpts,STORAGE_SIZE
# (16 + 8*4 = 48-byte header), then l4_canon_4d(ts(...,Nbb)),
# l4_canon_3d(uu(...,Nbb)), l4_canon_3d(vv(...,Nbb)), l4_canon_2d(ssh(...,Nbb)).
STEP_ENTRY_HEADER_BYTES = 48


def common_streams() -> set[str]:
    base = phase1.expected_inventory()
    extensions = {
        surface.RECORD, o1.RECORD, wzv.RECORD, bbl.NAME, *een.NEW_STREAMS,
        *zdf.ZDF_NAMES, tke.RECORD,
    }
    assert len(base) == 90 and len(extensions) == 11, (
        f"inventory partition changed: base={len(base)} extensions={len(extensions)}"
    )
    return base | extensions


def first_diff_offset(a: bytes, b: bytes) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def _decode_step_entry_field(offset: int, jpi: int, jpj: int, jpk: int, jpts: int) -> str:
    ts_n, uu_n, vv_n = jpi * jpj * jpk * jpts, jpi * jpj * jpk, jpi * jpj * jpk
    bounds = [
        ("ts", STEP_ENTRY_HEADER_BYTES, STEP_ENTRY_HEADER_BYTES + 8 * ts_n),
    ]
    bounds.append(("uu", bounds[-1][2], bounds[-1][2] + 8 * uu_n))
    bounds.append(("vv", bounds[-1][2], bounds[-1][2] + 8 * vv_n))
    bounds.append(("ssh", bounds[-1][2], bounds[-1][2] + 8 * jpi * jpj))
    for name, lo, hi in bounds:
        if lo <= offset < hi:
            return name
    return "trailing/unknown"


def _ssh_sanity(path_v2: Path, path_o1ice: Path) -> dict[str, object]:
    def load(path: Path):
        raw = path.read_bytes()
        magic = raw[:16]
        version, kstp, nbb, jpi, jpj, jpk, jpts, bits = struct.unpack(
            "<8i", raw[16:48]
        )
        ts_n, uu_n, vv_n = jpi * jpj * jpk * jpts, jpi * jpj * jpk, jpi * jpj * jpk
        ssh_off = 48 + 8 * (ts_n + uu_n + vv_n)
        ssh = np.frombuffer(raw, dtype="<f8", count=jpi * jpj, offset=ssh_off)
        ts = np.frombuffer(raw, dtype="<f8", count=ts_n, offset=48)
        uu = np.frombuffer(raw, dtype="<f8", count=uu_n, offset=48 + 8 * ts_n)
        vv = np.frombuffer(raw, dtype="<f8", count=vv_n, offset=48 + 8 * (ts_n + uu_n))
        return dict(magic=magic, version=version, kstp=kstp, nbb=nbb, jpi=jpi,
                    jpj=jpj, jpk=jpk, jpts=jpts, bits=bits, ts=ts, uu=uu, vv=vv,
                    ssh=ssh, total_bytes=len(raw))

    a, b = load(path_v2), load(path_o1ice)
    for key in ("jpi", "jpj", "jpk", "jpts", "bits", "kstp"):
        if a[key] != b[key]:
            return {"comparable": False, "reason": f"{key} differs: {a[key]} vs {b[key]}"}
    ts_equal = bool(np.array_equal(a["ts"], b["ts"]))
    uu_equal = bool(np.array_equal(a["uu"], b["uu"]))
    vv_equal = bool(np.array_equal(a["vv"], b["vv"]))
    d = a["ssh"] - b["ssh"]
    n_diff = int(np.count_nonzero(d))
    return {
        "comparable": True,
        "jpi": a["jpi"], "jpj": a["jpj"], "jpk": a["jpk"], "jpts": a["jpts"],
        "ts_bit_identical": ts_equal, "uu_bit_identical": uu_equal,
        "vv_bit_identical": vv_equal,
        "ssh_cells_total": int(d.size), "ssh_cells_differing": n_diff,
        "ssh_max_abs_diff_m": float(np.max(np.abs(d))) if n_diff else 0.0,
        "ssh_mean_abs_diff_over_differing_m": (
            float(np.mean(np.abs(d[d != 0]))) if n_diff else 0.0
        ),
    }


def validate(v2_root: Path, orca1ice_root: Path) -> dict[str, object]:
    common = common_streams()
    v2_files = {p.name for p in v2_root.glob("oracle_*.bin")}
    o1ice_files = {p.name for p in orca1ice_root.glob("oracle_*.bin")}
    missing_v2 = sorted(common - v2_files)
    missing_o1ice = sorted(common - o1ice_files)
    rows = []
    for name in sorted(common & v2_files & o1ice_files):
        a = (v2_root / name).read_bytes()
        b = (orca1ice_root / name).read_bytes()
        offset = None if a == b else first_diff_offset(a, b)
        rows.append({
            "file": name, "bytes_v2": len(a), "bytes_orca1ice": len(b),
            "identical": a == b, "first_diff_offset": offset,
        })
    identical = [r["file"] for r in rows if r["identical"]]
    differing = [r for r in rows if not r["identical"]]

    step_entry = next(
        (r for r in rows if r["file"] == STEP_ENTRY_1 and not r["identical"]), None
    )
    first_record_field = None
    ssh_sanity = None
    if step_entry is not None:
        with (v2_root / STEP_ENTRY_1).open("rb") as fh:
            fh.seek(20)
            _kstp, _nbb, jpi, jpj, jpk, jpts = struct.unpack("<6i", fh.read(24))
        first_record_field = _decode_step_entry_field(
            step_entry["first_diff_offset"], jpi, jpj, jpk, jpts
        )
        ssh_sanity = _ssh_sanity(v2_root / STEP_ENTRY_1, orca1ice_root / STEP_ENTRY_1)

    return {
        "v2_root": str(v2_root), "orca1ice_root": str(orca1ice_root),
        "common_stream_count": len(common),
        "missing_in_v2": missing_v2, "missing_in_orca1ice": missing_o1ice,
        "identical_count": len(identical), "identical_streams": identical,
        "differing_count": len(differing),
        "first_chronological_record": STEP_ENTRY_1,
        "first_chronological_record_identical": step_entry is None,
        "first_chronological_record_first_differing_field": first_record_field,
        "first_chronological_record_ssh_sanity": ssh_sanity,
        "all_differing_streams": rows if len(differing) <= len(rows) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--orca1ice-root", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = validate(args.v2_root, args.orca1ice_root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json:
        args.json.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
