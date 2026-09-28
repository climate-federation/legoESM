#!/usr/bin/env python3
"""Admit the Round-156 developed-state RK3 stage-2 momentum record.

The record is written by ``l2_r156_stage2.F90`` at NEMO's step 1081, stage 2,
from the acquisition card in
``nemo_testcase_l2_gyre_round156_developed_stage2/``.  Admission follows
operator note AS: an inherited stream from ANOTHER build is information only,
never a refusal; what decides passivity is (1) the NEMO restarts being
byte-identical to the un-instrumented Round-132 daily reference and (2) the
in-run calibration, in which a derived stream is rebuilt from the record's own
operands.

The calibration used here is NEMO's own corrected-velocity statement,
``GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90`` at the
``uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)`` loop.  A
fused multiply-add would normally make that a bad thing to gate; it is safe
ONLY because ``umask`` is exactly 0 or 1, so the product is exact and the
rebuild has one rounding either way.  The thickness-weighted assignment is
REPORTED, never gated: its Fortran association may be contracted by the
compiler and the campaign proves that statement under production JIT, not
here.

NEMO's correction loop runs ``1..jpkm1``; the rebuild compares every level,
which is equivalent only because ``umask`` is zero on the bottom level, and
the mask is checked to be binary below.

The record is also tied to the state the walk scores: NEMO binds stage 2's
``uu(Kaa)`` to stage 3's ``uu(Kmm)``, so this record's final velocity must be
byte-identical to the Round-154 transport record's ``uu_Kmm``.  That check is
gated whenever the transport record is given, and it proves passivity AT step
1081, which the restart comparison (steps 180..1080) cannot.

Plants (each prints ``STATUS PLANT-FIRED: <name>`` and exits nonzero):
``stamp``, ``truncation``, ``restart-byte``, ``operand-ulp``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

RECORD = "oracle_developed_stage2_kt00001081.bin"
MAGIC = b"NEMO_L2_R156ST2 "
GROUPS = (
    "rhs_entry", "uu_vv_Kbb", "uu_vv_Kmm", "r3u_r3v_Kbb", "r3u_r3v_Kmm",
    "r3u_r3v_Kaa", "r3t_Kmm_r3f", "ssh_Kmm_ssh_Kaa", "umask_vmask",
    "rDt_r1_Dt", "flags_vec_linssh", "rhd_ww", "after_hpg", "after_vor",
    "after_adv",
    "uu_vv_Kaa_raw", "zub_zvb", "uu_b_vv_b_Kaa", "uu_vv_Kaa_final",
)
RESTARTS = tuple(f"GYRE_OMIP_L2_P3_{step:08d}_restart.nc"
                 for step in (180, 360, 540, 720, 900, 1080))
PLANTS = ("stamp", "truncation", "restart-byte", "operand-ulp",
          "stage3-alignment")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_record(path: Path, *, truncate: int = 0) -> dict:
    """Parse the self-describing stage-2 record into named array pairs.

    The writer opens with ``ACCESS='STREAM'``, so the bytes carry no Fortran
    record markers: a 16-character magic, sixteen 4-byte integers, then one
    group per pair as a 16-character name, four 4-byte integers
    ``(rank, n1, n2, n3)`` and two column-major ``real(wp)`` payloads.
    """
    raw = path.read_bytes()
    if truncate:
        raw = raw[:-truncate]
    require(len(raw) > 80, f"{path}: record is {len(raw)} bytes")
    require(raw[:16] == MAGIC, f"{path}: wrong magic {raw[:16]!r}")
    header = np.frombuffer(raw, dtype="<i4", count=16, offset=16)
    meta = {
        "version": int(header[0]), "kt": int(header[1]),
        "kstg": int(header[2]), "Kbb": int(header[3]),
        "Kmm": int(header[4]), "Krhs": int(header[5]), "Kaa": int(header[6]),
        "jpi": int(header[7]), "jpj": int(header[8]), "jpk": int(header[9]),
        "storage_size": int(header[10]), "group_count": int(header[11]),
        "ntsi": int(header[12]), "ntei": int(header[13]),
        "ntsj": int(header[14]), "ntej": int(header[15]),
    }
    offset = 16 + 64
    fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    while offset < len(raw):
        require(offset + 32 <= len(raw), f"{path}: truncated group header")
        name = raw[offset:offset + 16].decode("ascii").strip()
        dims = np.frombuffer(raw, dtype="<i4", count=4, offset=offset + 16)
        offset += 32
        rank, n1, n2, n3 = (int(value) for value in dims)
        shape = {0: (), 2: (n1, n2), 3: (n1, n2, n3)}.get(rank)
        require(shape is not None, f"{path}/{name}: unsupported rank {rank}")
        count = int(np.prod(shape)) if shape else 1
        require(name not in fields, f"{path}/{name}: duplicate group")
        pair = []
        for _ in range(2):
            require(offset + count * 8 <= len(raw),
                    f"{path}/{name}: truncated payload")
            pair.append(np.frombuffer(
                raw, dtype="<f8", count=count,
                offset=offset).reshape(shape, order="F").copy())
            offset += count * 8
        fields[name] = (pair[0], pair[1])
    require(offset == len(raw), f"{path}: {len(raw) - offset} trailing bytes")
    return {"meta": meta, "fields": fields, "sha256": sha256(path),
            "bytes": len(raw)}


def compare_stage3_alignment(stage2_fields: dict,
                             transport_fields: dict) -> dict:
    """Stage 2's ``uu(Kaa)`` IS stage 3's ``uu(Kmm)``; prove it byte for byte.

    The restart comparison can only speak for steps 180..1080, all of them
    BEFORE the only step at which this writer executes.  This ties the new
    record to the state the walk already scores, and it is the check that
    would see a writer that perturbs NEMO at step 1081 alone.
    """
    rows = {}
    for tag, mine, theirs in (
            ("u", stage2_fields["uu_vv_Kaa_final"][0],
             transport_fields["uu_Kmm"]),
            ("v", stage2_fields["uu_vv_Kaa_final"][1],
             transport_fields["vv_Kmm"])):
        mine = np.ascontiguousarray(mine, dtype=np.float64)
        theirs = np.ascontiguousarray(theirs, dtype=np.float64)
        require(mine.shape == theirs.shape,
                f"stage-3 alignment {tag}: {mine.shape} vs {theirs.shape}")
        unequal = int(np.count_nonzero(
            mine.view(np.uint64) != theirs.view(np.uint64)))
        rows[tag] = {
            "cells_scored": int(mine.size), "cells_unequal": unequal,
            "max_abs": float(np.max(np.abs(mine - theirs), initial=0.0)),
        }
    require(all(row["cells_unequal"] == 0 for row in rows.values()),
            "stage-2 after-velocity is not byte-identical to the Round-154 "
            "stage-3 Kmm velocity: the writer perturbs step 1081 or the two "
            "records are not the same run")
    return rows


def _load_transport_gate(root: Path):
    """Load the committed Round-154 reader; never re-derive its layout."""
    path = (Path(__file__).resolve().parent
            / "nemo_testcase_l2_gyre_round154_developed_transport_gate.py")
    require(path.is_file(), f"missing Round-154 gate {path}")
    spec = importlib.util.spec_from_file_location("_r154_gate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit(root: Path, baseline: Path, expect_commit: str,
          plant: str | None = None, transport_record: Path | None = None
          ) -> dict:
    root = Path(root)
    baseline = Path(baseline)
    record_path = root / RECORD
    require(record_path.is_file(), f"missing {record_path}")
    record = read_record(
        record_path, truncate=8 if plant == "truncation" else 0)
    meta = record["meta"]
    require(meta["kt"] == 1081 and meta["kstg"] == 2,
            f"record is kt={meta['kt']} kstg={meta['kstg']}, not 1081/2")
    require(meta["storage_size"] == 64, "record is not 64-bit")
    require(meta["group_count"] == len(GROUPS),
            f"header declares {meta['group_count']} groups, "
            f"expected {len(GROUPS)}")
    require(tuple(record["fields"]) == GROUPS,
            "record groups differ from the registry: "
            f"{tuple(record['fields'])}")
    nonfinite = {
        name: int(np.count_nonzero(~np.isfinite(values)))
        for name, pair in record["fields"].items()
        for index, values in enumerate(pair)
        if not np.all(np.isfinite(values))
    }
    require(not nonfinite, f"non-finite payloads: {nonfinite}")

    stamp_path = root / f"{RECORD}.stamp"
    require(stamp_path.is_file(), f"missing {stamp_path}")
    stamp = stamp_path.read_text(encoding="utf-8").split()
    require(len(stamp) == 3, f"{stamp_path}: malformed stamp")
    recorded_sha = "0" * 64 if plant == "stamp" else stamp[0]
    require(recorded_sha == record["sha256"],
            f"{stamp_path}: stamp SHA does not match the record")
    require(stamp[1] == expect_commit,
            f"{stamp_path}: producer commit is not {expect_commit}")

    # Note AS, condition (1): the NEMO restarts of an instrumented build must
    # be byte-identical to the un-instrumented Round-132 daily reference.
    restarts = {}
    for name in RESTARTS:
        here, there = root / name, baseline / name
        require(here.is_file(), f"missing restart {here}")
        require(there.is_file(), f"missing baseline restart {there}")
        mine = sha256(here)
        if plant == "restart-byte" and name == RESTARTS[-1]:
            mine = "0" * 64
        restarts[name] = {"record": mine, "baseline": sha256(there),
                          "identical": mine == sha256(there)}
    differing = [name for name, row in restarts.items()
                 if not row["identical"]]
    require(not differing,
            "instrumented build moved NEMO's own restarts: "
            + ", ".join(differing))

    # Note AS, condition (2): rebuild a derived stream from the record's own
    # operands.  One multiply and one add per cell, so this is exact.
    umask, vmask = record["fields"]["umask_vmask"]
    raw_u, raw_v = record["fields"]["uu_vv_Kaa_raw"]
    zub, zvb = record["fields"]["zub_zvb"]
    final_u, final_v = record["fields"]["uu_vv_Kaa_final"]
    i0, i1 = meta["ntsi"] - 1, meta["ntei"]
    j0, j1 = meta["ntsj"] - 1, meta["ntej"]
    require(zub.shape == (i1 - i0, j1 - j0),
            f"zub is {zub.shape}, not the interior {(i1 - i0, j1 - j0)}")
    calibration = {}
    for tag, raw, correction, mask, final in (
            ("u", raw_u, zub, umask, final_u),
            ("v", raw_v, zvb, vmask, final_v)):
        rebuilt = raw.copy()
        window = (slice(i0, i1), slice(j0, j1))
        rebuilt[window] = (
            raw[window] + correction[..., None] * mask[window])
        if plant == "operand-ulp":
            rebuilt[i0, j0, 0] = np.nextafter(rebuilt[i0, j0, 0], np.inf)
        unequal = int(np.count_nonzero(
            rebuilt[window].view(np.uint64) != final[window].view(np.uint64)))
        calibration[tag] = {
            "cells_scored": int(rebuilt[window].size),
            "cells_unequal": unequal,
            "max_abs": float(np.max(np.abs(rebuilt[window] - final[window]),
                                    initial=0.0)),
        }
    require(all(row["cells_unequal"] == 0 for row in calibration.values()),
            "in-run calibration failed: the recorded corrected velocity is "
            "not the recorded raw velocity plus the recorded correction")

    alignment = None
    if transport_record is not None:
        gate154 = _load_transport_gate(Path(transport_record))
        transport = gate154.read_record(
            Path(transport_record) / gate154.RECORD)
        fields = dict(record["fields"])
        if plant == "stage3-alignment":
            moved = np.array(fields["uu_vv_Kaa_final"][0], copy=True)
            moved[0, 0, 0] = np.nextafter(moved[0, 0, 0], np.inf)
            fields["uu_vv_Kaa_final"] = (
                moved, fields["uu_vv_Kaa_final"][1])
        alignment = compare_stage3_alignment(fields, transport["fields"])

    # REPORTED, never gated: the thickness-weighted assignment, whose Fortran
    # association the compiler may contract.
    assignment = {}
    one = np.float64(1.0)
    for tag, bb, rhs, r3bb, r3mm, r3aa, mask, raw in (
            ("u", record["fields"]["uu_vv_Kbb"][0],
             record["fields"]["after_adv"][0],
             record["fields"]["r3u_r3v_Kbb"][0],
             record["fields"]["r3u_r3v_Kmm"][0],
             record["fields"]["r3u_r3v_Kaa"][0], umask, raw_u),
            ("v", record["fields"]["uu_vv_Kbb"][1],
             record["fields"]["after_adv"][1],
             record["fields"]["r3u_r3v_Kbb"][1],
             record["fields"]["r3u_r3v_Kmm"][1],
             record["fields"]["r3u_r3v_Kaa"][1], vmask, raw_v)):
        rdt = record["fields"]["rDt_r1_Dt"][0]
        rebuilt = ((one + r3bb[..., None]) * bb
                   + (rdt * (one + r3mm[..., None])) * rhs)
        rebuilt = rebuilt / (one + r3aa[..., None]) * mask
        window = (slice(i0, i1), slice(j0, j1), slice(0, meta["jpk"] - 1))
        assignment[tag] = {
            "cells_scored": int(rebuilt[window].size),
            "cells_unequal": int(np.count_nonzero(
                rebuilt[window].view(np.uint64)
                != raw[window].view(np.uint64))),
            "max_abs": float(np.max(np.abs(rebuilt[window] - raw[window]),
                                    initial=0.0)),
            "gated": False,
        }

    return {
        "format": "nemo-testcase-l2-gyre-round156-developed-stage2-v1",
        "status": "PASS", "record": {
            "path": str(record_path), "sha256": record["sha256"],
            "bytes": record["bytes"], "groups": list(GROUPS),
            "meta": meta,
        },
        "restarts": restarts,
        "in_run_calibration": calibration,
        "stage3_alignment": alignment,
        "assignment_rebuild_reported": assignment,
        "inherited_stream_check": "WAIVED by operator note AS",
        "expect_commit": expect_commit,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--transport-record", type=Path,
                        help="admitted Round-154 transport run directory; "
                             "its stage-3 Kmm velocity must equal this "
                             "record's stage-2 after-velocity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(args.root, args.baseline, args.expect_commit,
                       plant=args.plant,
                       transport_record=args.transport_record)
    except GateError as error:
        if args.plant:
            print(f"STATUS PLANT-FIRED: {args.plant}: {error}")
            return 1
        print(f"REFUSE Round-156 stage-2 record: {error}", file=sys.stderr)
        return 2
    if args.plant:
        print(f"REFUSE: {args.plant} plant stayed green", file=sys.stderr)
        return 3
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"STATUS PASS: round-156 stage-2 record admitted "
          f"({report['record']['bytes']} bytes, {len(GROUPS)} groups)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
