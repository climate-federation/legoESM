"""Lane B rung L1 (G5 + ORCA2's bottom boundary layer): does the record show BBL acting?

NEMO runs ``bbl``/``tra_bbl`` only in the stage-3 tracer update (stprk3_stg.f90:560, :739), so the
state after kt 1 (the kt-2 step-entry record) must keep G5's u, v and sea level bit for bit and may
change T/S only in columns that touch a BBL face.  A face carries a coefficient only when both T
points are wet at the surface (``ssumask`` in ``ahu_bbl_0``, trabbl.f90:600-601) and their bottom
levels differ (``mgrhu`` /= 0, trabbl.f90:583-591; on a flat face the gate is closed, measured on
ORCA2's own ``ahu_bbl`` record by the diffusive-BBL gate).  The two north-fold rows are admitted
wholesale (their V-face neighbours are folded images).

``--check-bbl-entry DIR`` exits 0 when DIR's kt-2 entry record passes, 1 otherwise.
``--check-bbl-stages DIR`` does the same on DIR's kt-1 stage records: stages 1-2 equal G5's in
every field, stage 3 differs only in T/S inside the footprint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4
import numpy as np

TESTCASES = Path(__file__).resolve().parents[1]
if str(TESTCASES) not in sys.path:
    sys.path.insert(0, str(TESTCASES))

from nemo_testcase_l1_vortex_kt2_walk import read_stage  # noqa: E402
from nemo_testcase_phase3_trajectory_gate import read_entry  # noqa: E402

G5_ENTRY_KT2 = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round31/"
    "oracle_g5_r31/twin_a/oracle_step_entry_kt00000002.bin")
DOMAIN = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round21/"
              "oracle_g4_r21/p6_a/domain_cfg.nc")


def bbl_footprint(bottom_level: np.ndarray) -> np.ndarray:
    """Wet (y, x) columns touching a wet U or V face whose two bottom levels differ."""
    mb = np.asarray(bottom_level, dtype=np.int64)
    wet = mb > 0
    u_face = wet & np.roll(wet, -1, axis=1) & (mb != np.roll(mb, -1, axis=1))  # cyclic E-W
    v_face = wet[:-1] & wet[1:] & (mb[:-1] != mb[1:])
    touched = u_face | np.roll(u_face, 1, axis=1)
    touched[:-1] |= v_face
    touched[1:] |= v_face
    touched[-2:] = True                                    # north-fold rows
    return touched & wet


def check_bbl_entry(l1_entry: Path, g5_entry: Path, bottom_level: np.ndarray) -> dict:
    l1, g5 = read_entry(l1_entry, "ORCA2L1-bbl"), read_entry(g5_entry, "ORCA2G5-ice")
    if (l1["step"], g5["step"]) != (2, 2):
        raise ValueError(f"expected two kt-2 entry records, got steps {l1['step']}, {g5['step']}")
    fp = bbl_footprint(bottom_level)
    if fp.shape != l1["T"].shape[:2]:
        raise ValueError(f"domain {fp.shape} does not match the record {l1['T'].shape[:2]}")
    out = {"footprint_columns": int(fp.sum()), "wet_columns": int((bottom_level > 0).sum())}
    for name in ("u", "v", "ssh"):
        out[f"{name}_unequal"] = int(np.sum(np.asarray(l1[name]) != np.asarray(g5[name])))
    moved = np.zeros(fp.shape, dtype=bool)
    for name in ("T", "S"):
        ne = np.asarray(l1[name]) != np.asarray(g5[name])
        out[f"{name}_unequal"] = int(ne.sum())
        moved |= ne.any(axis=-1)
    out["moved_columns"] = int(moved.sum())
    out["moved_outside_footprint"] = int((moved & ~fp).sum())
    out["pass"] = bool(out["u_unequal"] == out["v_unequal"] == out["ssh_unequal"] == 0
                       and out["moved_columns"] > 0 and out["moved_outside_footprint"] == 0)
    return out


def check_bbl_stages(l1_dir: Path, g5_dir: Path, bottom_level: np.ndarray) -> dict:
    fp = bbl_footprint(bottom_level)
    out = {}
    for st in (1, 2, 3):
        name = f"oracle_stage_kt00000001_s{st}.bin"
        a = read_stage(l1_dir / name, expect_step=1, expect_stage=st)
        b = read_stage(g5_dir / name, expect_step=1, expect_stage=st)
        moved = np.zeros(fp.shape, dtype=bool)
        for f in ("T", "S", "u", "v", "ssh"):
            ne = np.asarray(a[f]) != np.asarray(b[f])
            out[f"s{st}_{f}_unequal"] = int(ne.sum())
            if f in ("T", "S"):
                moved |= ne.any(axis=-1)
        out[f"s{st}_moved_outside_footprint"] = int((moved & ~fp).sum())
    out["pass"] = bool(
        not any(v for k, v in out.items() if k.startswith(("s1_", "s2_")))
        and out["s3_u_unequal"] == out["s3_v_unequal"] == out["s3_ssh_unequal"] == 0
        and out["s3_T_unequal"] + out["s3_S_unequal"] > 0
        and out["s3_moved_outside_footprint"] == 0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-bbl-entry", type=Path, metavar="DIR")
    mode.add_argument("--check-bbl-stages", type=Path, metavar="DIR")
    ap.add_argument("--g5-entry", type=Path, default=G5_ENTRY_KT2)
    ap.add_argument("--domain", type=Path, default=DOMAIN)
    a = ap.parse_args(argv)
    with netCDF4.Dataset(a.domain) as ds:
        mb = np.squeeze(np.asarray(ds.variables["bottom_level"][:]))
    if a.check_bbl_stages:
        res = check_bbl_stages(a.check_bbl_stages, a.g5_entry.parent, mb)
        print("L1_BBL_STAGES " + " ".join(f"{k}={v}" for k, v in res.items()))
        return 0 if res["pass"] else 1
    res = check_bbl_entry(a.check_bbl_entry / "oracle_step_entry_kt00000002.bin", a.g5_entry, mb)
    print("L1_BBL_ENTRY " + " ".join(f"{k}={v}" for k, v in res.items()))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
