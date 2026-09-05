#!/usr/bin/env python3
"""Admit the Round-21 GYRE oracle extension by compared consumed record fields.

NEMO stream dumps contain whole work arrays, including halos and, for the
pre-``tra_adv_trp`` transport record, a ``zFw`` slot not yet defined in the
executed vector-invariant branch.  Raw-file inequality in those bytes is not a
state change.  This gate nevertheless fails on every bit change visible to the
existing record parsers, on a common BTORD operand, or in the final restart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


NX, NY, NZ = 36, 26, 31
N2, N3, NC = NX * NY, NX * NY * NZ, (NX - 4) * (NY - 4)
WRITER_OVERRIDE: str | None = None
BASE = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round19_oracle_v2_external")
CAND = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round21_oracle_v2_stage_ww")
TWIN = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round19_oracle_v2_external_coeff")
RESTART = "GYRE_OMIP_L2_P3_00000010_restart.nc"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _owned3(a: np.ndarray) -> np.ndarray:
    return a.reshape((NX, NY, NZ), order="F")[2:-2, 2:-2, :]


def _owned2(a: np.ndarray) -> np.ndarray:
    return a.reshape((NX, NY), order="F")[2:-2, 2:-2]


def _bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(a.view(np.uint64), b.view(np.uint64))


def _layout(kind: str):
    if kind == "transport":
        return 48, [(x, N3, "3", x != "zFw") for x in ("zFu", "zFv", "zFw")]
    if kind == "trtrp":
        return 60, [(x, N3, "3", True) for x in ("zFu", "zFv", "zFw")]
    if kind == "wzv":
        return 48, [(x, N3, "3", True) for x in ("ww_pre_aimp", "ww_post_aimp", "pFw")]
    if kind == "trpop":
        fields = (("e2u", N2, "2"), ("e3u", N3, "3"), ("uu", N3, "3"),
                  ("zub", N2, "2"), ("umask", N3, "3"), ("zFu", N3, "3"),
                  ("e1v", N2, "2"), ("e3v", N3, "3"), ("vv", N3, "3"),
                  ("zvb", N2, "2"), ("vmask", N3, "3"), ("zFv", N3, "3"),
                  ("un_adv", N2, "2"), ("r1_hu", N2, "2"), ("uu_b", N2, "2"),
                  ("vn_adv", N2, "2"), ("r1_hv", N2, "2"), ("vv_b", N2, "2"))
        return 48, [(n, s, k, True) for n, s, k in fields]
    if kind == "rktracer":
        names = ("zero_T", "zero_S", "zFu", "zFv", "zFw", "after_advection_T",
                 "after_advection_S", "after_sbc_T", "after_sbc_S", "Kbb_T",
                 "Kbb_S", "Kmm_T", "Kmm_S", "Kaa_T", "Kaa_S")
        return 60, ([(n, N3, "3", True) for n in names]
                    + [(n, N2, "2", True) for n in ("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")])
    if kind == "slow":
        fields = ([(n, N3, "3", True) for n in
                   ("e3u", "krhs_u", "umask", "e3v", "krhs_v", "vmask")]
                  + [(n, NC, "c", True) for n in ("depth_u", "depth_v")]
                  + [(n, N2, "2", True) for n in ("r1_hu0", "r1_hv0")]
                  + [(n, NC, "c", True) for n in ("post_drag_u", "post_drag_v")]
                  + [(n, N2, "2", True) for n in ("cd_u", "cd_v")]
                  + [("r1_rho0", 1, "s", True)]
                  + [(n, N2, "2", True) for n in ("utau", "vtau", "r1_hu", "r1_hv")]
                  + [(n, NC, "c", True) for n in ("post_wind_u", "post_wind_v")])
        return 76, fields
    raise ValueError(kind)


KINDS = {
    "oracle_rkstage1_transport_operands_kt00000001.bin": "trpop",
    "oracle_rkstage3_wzv_kt00000001.bin": "wzv",
    "oracle_rktracer_operands_kt00000001_s1.bin": "rktracer",
    "oracle_rktracer_operands_kt00000001_s2.bin": "rktracer",
    "oracle_slow_forcing_kt00000001.bin": "slow",
    "oracle_tracer_transport_kt00000001_s3.bin": "trtrp",
    **{f"oracle_transport_kt00000001_s{s}.bin": "transport" for s in (1, 2, 3)},
}

SOURCES = {
    "trpop": ["MY_SRC/stprk3_stg.F90:297-302", "nemo_testcase_l2_gyre_round13_tracer.py:91-154"],
    "wzv": ["MY_SRC/traadv.F90:225-238", "nemo_testcase_l2_gyre_stage3_completion_gate.py:110-147"],
    "rktracer": ["MY_SRC/stprk3_stg.F90:657-670", "nemo_testcase_l2_gyre_round13_tracer.py:38-87"],
    "slow": ["MY_SRC/stp2d.F90:187-231", "nemo_testcase_l2_gyre_round16_slow_forcing.py:49-116"],
    "trtrp": ["MY_SRC/stprk3_stg.F90:622-626", "nemo_testcase_l2_gyre_phase3_gate.py:189-234"],
    "transport": ["MY_SRC/stprk3_stg.F90:343-348", "nemo_testcase_l2_gyre_phase3_gate.py:189-234"],
}


def _compare_layout(a_path: Path, b_path: Path, kind: str, plant: list[bool]) -> dict:
    a_bytes, b_bytes = a_path.read_bytes(), b_path.read_bytes()
    if len(a_bytes) != len(b_bytes):
        return {"consumed_equal": False, "error": "size mismatch"}
    header, fields = _layout(kind)
    if a_bytes[:header] != b_bytes[:header]:
        return {"consumed_equal": False, "error": "header mismatch"}
    offset, changed, consumed_equal = header, [], True
    for name, count, projection, consumed in fields:
        aa = np.frombuffer(a_bytes, np.float64, count, offset)
        bb = np.frombuffer(b_bytes, np.float64, count, offset)
        byte_a = np.frombuffer(a_bytes, np.uint8, count * 8, offset)
        byte_b = np.frombuffer(b_bytes, np.uint8, count * 8, offset)
        element_diff = aa.view(np.uint64) != bb.view(np.uint64)
        if np.any(element_diff):
            ids = np.flatnonzero(element_diff)
            if projection == "3":
                ii, jj, kk = ids % NX, (ids // NX) % NY, ids // N2
                in_projection = (ii >= 2) & (ii < NX - 2) & (jj >= 2) & (jj < NY - 2)
                first = [int(ii[0]), int(jj[0]), int(kk[0])]
            elif projection == "2":
                ii, jj = ids % NX, ids // NX
                in_projection = (ii >= 2) & (ii < NX - 2) & (jj >= 2) & (jj < NY - 2)
                first = [int(ii[0]), int(jj[0])]
            else:
                in_projection = np.ones(ids.size, dtype=bool)
                first = [int(ids[0])]
            changed.append({
                "field": name,
                "changed_elements": int(ids.size),
                "changed_bytes": int(np.count_nonzero(byte_a != byte_b)),
                "changed_in_parser_projection": int(np.count_nonzero(in_projection)),
                "first_index_0based": first,
                "slot_consumed": consumed,
            })
        pa = _owned3(aa) if projection == "3" else _owned2(aa) if projection == "2" else aa
        pb = _owned3(bb) if projection == "3" else _owned2(bb) if projection == "2" else bb
        if consumed and plant and not plant[0]:
            pb = pb.copy()
            flat = pb.reshape(-1)
            flat[0:1].view(np.uint64)[:] ^= np.uint64(1)
            plant[0] = True
        if consumed and not _bits_equal(np.ascontiguousarray(pa), np.ascontiguousarray(pb)):
            consumed_equal = False
        offset += count * 8
    assert offset == len(a_bytes), (a_path, offset, len(a_bytes))
    raw = np.frombuffer(a_bytes, np.uint8) != np.frombuffer(b_bytes, np.uint8)
    return {
        "consumed_equal": consumed_equal,
        "raw_differing_bytes": int(np.count_nonzero(raw)),
        "first_differing_byte_1based": int(np.flatnonzero(raw)[0] + 1) if np.any(raw) else None,
        "changed_fields": changed,
        "writer": WRITER_OVERRIDE or SOURCES[kind][0],
        "parser": SOURCES[kind][1],
    }


BT_NAMES = ("u_entry", "v_entry", "u_history_b", "v_history_b", "u_history_bb",
            "v_history_bb", "eta_entry", "eta_history_b", "eta_history_bb", "u_mid",
            "v_mid", "eta_mid", "face_depth_u_mid", "face_depth_v_mid",
            "metric_transport_u", "metric_transport_v", "metric_e2u", "metric_e1v",
            "r1_area", "continuity_du", "continuity_dv", "continuity_divergence",
            "continuity_forcing", "eta_exit", "face_ssh_u_exit", "face_ssh_v_exit",
            "eta_pgf", "r1_dx_u", "r1_dy_v", "pgf_u", "pgf_v", "cor_u", "cor_v",
            "trd_u", "trd_v", "slow_u", "slow_v", "u_exit", "v_exit",
            "face_depth_u_exit", "face_depth_v_exit", "r1_face_depth_u_exit",
            "r1_face_depth_v_exit")
BT_EXTRA = ("ffu_nw", "ffu_ne", "ffu_sw", "ffu_se", "ffv_sw", "ffv_se", "ffv_nw", "ffv_ne")


def _read_bt(path: Path) -> tuple[str, tuple, float, list]:
    with path.open("rb") as f:
        magic = f.read(16).decode("ascii").rstrip()
        header = struct.unpack("=6i", f.read(24))
        dt = float(np.fromfile(f, np.float64, 1)[0])
        names = BT_NAMES + (BT_EXTRA if magic.endswith("_2") else ())
        rows = []
        for _ in range(2):
            jn = struct.unpack("=i", f.read(4))[0]
            weights = np.fromfile(f, np.float64, 7)
            values = {}
            for name in names:
                values[name] = np.fromfile(f, np.float64, NC if name in {"slow_u", "slow_v", *BT_EXTRA} else N2)
            rows.append((jn, weights, values))
        assert f.read(1) == b""
    return magic, header, dt, rows


def _compare_bt(a: Path, b: Path) -> dict:
    aa, bb = _read_bt(a), _read_bt(b)
    equal = aa[2] == bb[2]
    common_diffs = []
    for ar, br in zip(aa[3], bb[3]):
        equal &= ar[0] == br[0] and _bits_equal(ar[1], br[1])
        for name in BT_NAMES:
            if not _bits_equal(ar[2][name], br[2][name]):
                common_diffs.append([ar[0], name])
                equal = False
    raw_a, raw_b = a.read_bytes(), b.read_bytes()
    limit = min(len(raw_a), len(raw_b))
    raw = np.frombuffer(raw_a[:limit], np.uint8) != np.frombuffer(raw_b[:limit], np.uint8)
    return {
        "consumed_equal": bool(equal), "common_field_differences": common_diffs,
        "schema": f"{aa[0]}->{bb[0]}", "appended_fields": list(BT_EXTRA),
        "raw_differing_bytes": int(np.count_nonzero(raw)),
        "appended_bytes": max(0, len(raw_b) - len(raw_a)),
        "first_differing_byte_1based": int(np.flatnonzero(raw)[0] + 1) if np.any(raw) else limit + 1,
        "writer": "MY_SRC/dynspg_ts.F90:598-599,934-942",
        "parser": "nemo_testcase_l2_gyre_round14_advmean.py:123-179",
    }


def main() -> int:
    global NX, NY, NZ, N2, N3, NC, WRITER_OVERRIDE
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASE)
    parser.add_argument("--candidate", type=Path, default=CAND)
    parser.add_argument("--twin", type=Path, default=TWIN)
    parser.add_argument("--plant-consumed", action="store_true")
    # The consumed-field admission RULE is card-independent; only the record
    # geometry and inventory are not.  Round 26 reuses it for the LOCK
    # external-mode acquisition instead of writing a second copy.
    parser.add_argument("--dims", type=int, nargs=3, metavar=("NX", "NY", "NZ"),
                        default=(NX, NY, NZ),
                        help="oracle array dims INCLUDING the 2-cell halo")
    parser.add_argument("--restart", default=RESTART,
                        help="restart file that must stay byte-identical")
    parser.add_argument("--allowed-new", nargs="*", default=None,
                        help="record names the candidate may add")
    parser.add_argument("--writer",
                        help="instrument file:line for this card's records")
    args = parser.parse_args()
    NX, NY, NZ = args.dims
    N2, N3, NC = NX * NY, NX * NY * NZ, (NX - 4) * (NY - 4)
    WRITER_OVERRIDE = args.writer
    plant = [not args.plant_consumed]
    baseline_names = sorted(p.name for p in args.baseline.glob("oracle_*.bin"))
    candidate_names = sorted(p.name for p in args.candidate.glob("oracle_*.bin"))
    allowed_new = (
        set(args.allowed_new) if args.allowed_new is not None
        else {f"oracle_rkstage_ww_kt00000001_s{s}.bin" for s in (1, 2, 3)})
    violations, rows, exact = [], [], 0
    if set(candidate_names) - set(baseline_names) != allowed_new:
        violations.append("unexpected candidate oracle inventory")
    if set(baseline_names) - set(candidate_names):
        violations.append("missing inherited oracle record")
    for name in baseline_names:
        a, b = args.baseline / name, args.candidate / name
        if a.read_bytes() == b.read_bytes():
            exact += 1
            continue
        if name == "oracle_bt_ordered_operands_kt00000001.bin":
            result = _compare_bt(a, b)
        elif name in KINDS:
            result = _compare_layout(a, b, KINDS[name], plant)
        else:
            result = {"consumed_equal": False, "error": "changed record has no registered schema"}
        if args.twin.is_dir() and (args.twin / name).is_file():
            t = args.twin / name
            twin = (_compare_bt(t, b) if name == "oracle_bt_ordered_operands_kt00000001.bin"
                    else _compare_layout(t, b, KINDS[name], [True]) if name in KINDS else {})
            result["round19_twin_raw_equal"] = t.read_bytes() == b.read_bytes()
            result["round19_twin_consumed_equal"] = twin.get("consumed_equal")
        rows.append({"record": name, **result})
        if not result.get("consumed_equal", False):
            violations.append(name)
    restart = args.restart
    restart_equal = _sha(args.baseline / restart) == _sha(args.candidate / restart)
    if not restart_equal:
        violations.append(restart)
    report = {
        "baseline": str(args.baseline), "candidate": str(args.candidate),
        "baseline_oracle_records": len(baseline_names), "byte_identical_records": exact,
        "classified_changed_records": rows, "restart_byte_identical": restart_equal,
        "plant_consumed": args.plant_consumed,
        # plant[] is a one-shot SENTINEL that starts True when no plant was
        # requested, so publishing it as "plant_applied" said `true` on runs
        # that planted nothing.  Report the fact instead.
        "plant_applied": bool(args.plant_consumed and plant[0]),
        "dims_with_halo": [NX, NY, NZ], "restart": restart,
        "verdict": "PASS" if not violations else "FAIL", "violations": violations,
        "artifacts": {"baseline_restart_sha256": _sha(args.baseline / restart),
                      "candidate_restart_sha256": _sha(args.candidate / restart)},
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"ROUND21_WRITE_ONLY_ADMISSION {report['verdict']}: exact={exact}/{len(baseline_names)} changed={len(rows)} restart_equal={restart_equal} plant={args.plant_consumed}")
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
