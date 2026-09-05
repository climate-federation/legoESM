#!/usr/bin/env python3
"""Fail-closed schema gate for the ORCA1-ice/ORCA2 acquisition records.

All formats are decoded from their on-disk headers alone.  The allocation
table below is an independent validity check, never side knowledge needed to
find the end of a field or record.
"""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path

import numpy as np


THD_MAGIC = b"NEMO_L4_O1ITHD1"
DYN_MAGIC = b"NEMO_L4_O1IDYN1"
PRA_MAGIC = b"NEMO_L4_O1IPRA1"


def _fail(message: str) -> None:
    raise ValueError(message)


def _parse(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    if len(raw) < 16:
        _fail("truncated magic")
    magic = raw[:16].rstrip(b" ")
    if magic == THD_MAGIC:
        nh, ndim, expected_nf, nscalar = 11, 4, 16, 0
        names = [
            "a_i", "v_i", "v_s", "sv_i", "t_su", "e_i", "e_s",
            "szv_i", "qns_ice", "qsr_ice", "dqns_ice", "evap_ice",
            "qprec_ice", "qml_ice", "qcn_ice", "qtr_ice_top",
        ]
    elif magic == DYN_MAGIC:
        nh, ndim, expected_nf, nscalar = 12, 3, 28, 5
        names = [
            "u_ice", "v_ice", "utau_ice", "vtau_ice", "strength",
            "at_i", "vt_i", "vt_s", "tau_icebfr", "icb_tmask",
            "fast_tmask", "zaU", "zaV", "taux_ai", "tauy_ai",
            "dragx", "dragy", "taux_base", "tauy_base", "mU", "mV",
            "ht_Kmm", "hu_Kmm", "hv_Kmm", "stress1", "stress2",
            "stress12", "a_i",
        ]
    elif magic == PRA_MAGIC:
        nh, ndim, expected_nf, nscalar = 11, 4, 35, 0
        names = [
            "sxice", "syice", "sxxice", "syyice", "sxyice",
            "sxsn", "sysn", "sxxsn", "syysn", "sxysn",
            "sxa", "sya", "sxxa", "syya", "sxya",
            "sxage", "syage", "sxxage", "syyage", "sxyage",
            "sxc0", "syc0", "sxxc0", "syyc0", "sxyc0",
            "sxe", "sye", "sxxe", "syye", "sxye",
            "sxsal", "sysal", "sxxsal", "syysal", "sxysal",
        ]
    else:
        _fail(f"bad magic {magic!r}")

    offset = 16
    need = offset + 4 * nh
    if len(raw) < need:
        _fail("truncated base header")
    header = np.frombuffer(raw, dtype="<i4", count=nh, offset=offset).copy()
    offset = need
    if int(header[0]) != 1:
        _fail(f"bad version {header[0]}")
    if magic in (THD_MAGIC, PRA_MAGIC):
        version, kt, frame, jpi, jpj, jpl, nlay_i, nlay_s, wpbits, nf, payload = map(int, header)
    else:
        version, kt, kmm, jpi, jpj, jpl, nlay_i, nlay_s, wpbits, nf, ns, payload = map(int, header)
        if ns != nscalar:
            _fail(f"scalar count {ns} != {nscalar}")
    if wpbits != 64 or nf != expected_nf:
        _fail(f"header wpbits/nfield mismatch: {wpbits}/{nf}")
    if min(jpi, jpj, jpl, nlay_i, nlay_s) <= 0:
        _fail("non-positive allocation extent")

    extent_count = ndim * nf
    need = offset + 4 * extent_count
    if len(raw) < need:
        _fail("truncated extent table")
    ext = np.frombuffer(raw, dtype="<i4", count=extent_count, offset=offset).copy()
    ext = ext.reshape((ndim, nf), order="F")
    offset = need
    if np.any(ext <= 0):
        _fail("non-positive field extent")

    if magic == THD_MAGIC:
        expected = np.empty((4, 16), dtype=np.int32)
        expected[:, :5] = np.array([jpi, jpj, jpl, 1])[:, None]
        expected[:, 5] = [jpi, jpj, nlay_i, jpl]
        expected[:, 6] = [jpi, jpj, nlay_s, jpl]
        expected[:, 7] = [jpi, jpj, nlay_i, jpl]
        expected[:, 8:12] = np.array([jpi, jpj, jpl, 1])[:, None]
        expected[:, 12] = [jpi, jpj, 1, 1]
        expected[:, 13:16] = np.array([jpi, jpj, jpl, 1])[:, None]
    elif magic == DYN_MAGIC:
        expected = np.tile(np.array([jpi, jpj, 1], dtype=np.int32)[:, None], (1, 28))
        expected[:, 27] = [jpi, jpj, jpl]
    else:
        expected = np.empty((4, 35), dtype=np.int32)
        expected[:, :20] = np.array([jpi, jpj, jpl, 1])[:, None]
        expected[:, 20:25] = np.array([jpi, jpj, nlay_s, jpl])[:, None]
        expected[:, 25:30] = np.array([jpi, jpj, nlay_i, jpl])[:, None]
        expected[:, 30:35] = np.array([jpi, jpj, jpl, 1])[:, None]
    if not np.array_equal(ext, expected):
        _fail("extent table disagrees with allocation contract")

    derived_payload = nscalar + int(sum(np.prod(ext[:, jf], dtype=np.int64) for jf in range(nf)))
    if payload != derived_payload:
        _fail(f"payload count {payload} != derived {derived_payload}")
    expected_eof = offset + 8 * payload
    if len(raw) != expected_eof:
        _fail(f"exact-EOF failure: size {len(raw)} != {expected_eof}")
    values = np.frombuffer(raw, dtype="<f8", count=payload, offset=offset)
    if not np.all(np.isfinite(values)):
        _fail("non-finite payload")

    return {
        "path": str(path), "magic": magic.decode(), "kt": kt,
        "frame": frame if magic in (THD_MAGIC, PRA_MAGIC) else None,
        "Kmm": kmm if magic == DYN_MAGIC else None,
        "jpi": jpi, "jpj": jpj, "jpl": jpl, "nlay_i": nlay_i,
        "nlay_s": nlay_s, "field_count": nf, "field_names": names,
        "payload_values": payload, "bytes": len(raw), "exact_eof": True,
    }


def _synthetic(magic: bytes) -> bytes:
    jpi, jpj, jpl, ni, ns = 3, 4, 1, 3, 3
    if magic == THD_MAGIC:
        ext = np.empty((4, 16), dtype="<i4")
        ext[:, :5] = np.array([jpi, jpj, jpl, 1])[:, None]
        ext[:, 5] = [jpi, jpj, ni, jpl]
        ext[:, 6] = [jpi, jpj, ns, jpl]
        ext[:, 7] = [jpi, jpj, ni, jpl]
        ext[:, 8:12] = np.array([jpi, jpj, jpl, 1])[:, None]
        ext[:, 12] = [jpi, jpj, 1, 1]
        ext[:, 13:] = np.array([jpi, jpj, jpl, 1])[:, None]
        payload = int(sum(np.prod(ext[:, k]) for k in range(16)))
        head = [1, 1, 0, jpi, jpj, jpl, ni, ns, 64, 16, payload]
    elif magic == DYN_MAGIC:
        ext = np.tile(np.array([jpi, jpj, 1], dtype="<i4")[:, None], (1, 28))
        ext[:, 27] = [jpi, jpj, jpl]
        payload = 5 + int(sum(np.prod(ext[:, k]) for k in range(28)))
        head = [1, 1, 3, jpi, jpj, jpl, ni, ns, 64, 28, 5, payload]
    else:
        ext = np.empty((4, 35), dtype="<i4")
        ext[:, :20] = np.array([jpi, jpj, jpl, 1])[:, None]
        ext[:, 20:25] = np.array([jpi, jpj, ns, jpl])[:, None]
        ext[:, 25:30] = np.array([jpi, jpj, ni, jpl])[:, None]
        ext[:, 30:35] = np.array([jpi, jpj, jpl, 1])[:, None]
        payload = int(sum(np.prod(ext[:, k]) for k in range(35)))
        head = [1, 1, 0, jpi, jpj, jpl, ni, ns, 64, 35, payload]
    return magic.ljust(16, b" ") + struct.pack("<" + "i" * len(head), *head) + ext.tobytes(order="F") + bytes(8 * payload)


def _self_test() -> dict[str, object]:
    plants: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for label, magic in (("thd", THD_MAGIC), ("dyn", DYN_MAGIC), ("pra", PRA_MAGIC)):
            valid = _synthetic(magic)
            path = root / f"{label}.bin"
            path.write_bytes(valid)
            _parse(path)
            variants = {
                "magic": b"X" + valid[1:],
                "truncated": valid[:-1],
                "trailing": valid + b"X",
            }
            for kind, data in variants.items():
                path.write_bytes(data)
                try:
                    _parse(path)
                except ValueError:
                    plants[f"{label}_{kind}"] = True
                else:
                    _fail(f"plant did not bind: {label}_{kind}")
    return {"valid_synthetic_formats": 3, "binding_plants": plants}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result: dict[str, object] = {}
    if args.self_test:
        result["self_test"] = _self_test()
    if args.run_dir:
        paths = sorted(args.run_dir.glob("oracle_orca1ice_*.bin"))
        if not paths:
            _fail("no ORCA1-ice records")
        result["records"] = [_parse(path) for path in paths]
    if not result:
        parser.error("select --self-test and/or --run-dir")
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
