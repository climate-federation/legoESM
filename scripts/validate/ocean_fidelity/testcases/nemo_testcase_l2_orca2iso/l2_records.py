#!/usr/bin/env python3
"""Fail-closed proof that ORCA2 L2's isoneutral tracer flux acts."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import netCDF4
import numpy as np

TESTCASES = Path(__file__).resolve().parents[1]
if str(TESTCASES) not in sys.path:
    sys.path.insert(0, str(TESTCASES))

from nemo_testcase_l1_vortex_kt2_walk import read_bt_frame  # noqa: E402
from nemo_testcase_phase3_trajectory_gate import read_entry  # noqa: E402

G5_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
    "round31/oracle_g5_r31/twin_a")
DOMAIN = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
    "round21/oracle_g4_r21/p6_a/domain_cfg.nc")
STAGE_NAME = "oracle_rktracer_stage3_kt00000001.bin"
ENTRY_NAME = "oracle_step_entry_kt00000002.bin"
BT_NAME = "oracle_bt_frames_kt00000001.bin"
HALO = 2


def _xyz(values, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F")[
        HALO:-HALO, HALO:-HALO].transpose(1, 0, 2)


def read_tracer_stage3(path: Path) -> dict[str, np.ndarray | int]:
    """Read the record's own extents and all source-ordered arrays to EOF."""

    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype="=f8")
    version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits = header
    if magic != "NEMO_L2_RKTR3_1" or (version, kt, stage, bits) != (1, 1, 3, 64):
        raise ValueError(f"{path}: unsupported header {magic!r} {header!r}")
    count = nx * ny * nz
    if values.size != 16 * count + 3 * nx * ny or not np.isfinite(values).all():
        raise ValueError(f"{path}: truncated, extended, or non-finite payload")
    names = (
        "zero_T", "zero_S", "after_advection_T", "after_advection_S",
        "after_sbc_T", "after_sbc_S", "before_ldf_T", "before_ldf_S",
        "after_ldf_T", "after_ldf_S", "Kbb_T", "Kbb_S", "Kmm_T",
        "Kmm_S", "Kaa_T", "Kaa_S",
    )
    out = {name: _xyz(values[i * count:(i + 1) * count], nx, ny, nz)
           for i, name in enumerate(names)}
    out.update({"Kbb": kbb, "Kmm": kmm, "Krhs": krhs, "Kaa": kaa})
    return out


def check_l2(
    l2_dir: Path,
    g5_dir: Path,
    domain: Path,
    check_barotropic: bool = False,
) -> dict[str, int | bool]:
    l2_stage = read_tracer_stage3(l2_dir / STAGE_NAME)
    g5_stage = read_tracer_stage3(g5_dir / STAGE_NAME)
    l2_entry = read_entry(l2_dir / ENTRY_NAME, "ORCA2L2-iso")
    g5_entry = read_entry(g5_dir / ENTRY_NAME, "ORCA2G5-ice")
    with netCDF4.Dataset(domain) as ds:
        bottom = np.squeeze(np.asarray(ds.variables["bottom_level"][:])).astype(int)
    nz = np.asarray(l2_stage["after_ldf_T"]).shape[-1]
    wet = np.arange(nz)[None, None, :] < bottom[..., None]
    out: dict[str, int | bool] = {}
    moved = 0
    dry_moved = 0
    completed = 0
    for tracer in ("T", "S"):
        delta = np.asarray(l2_stage[f"after_ldf_{tracer}"]) - np.asarray(
            l2_stage[f"before_ldf_{tracer}"])
        moved += int(np.count_nonzero(delta[wet]))
        dry_moved += int(np.count_nonzero(delta[~wet]))
        completed += int(np.count_nonzero(
            np.asarray(l2_stage[f"Kaa_{tracer}"]) !=
            np.asarray(g5_stage[f"Kaa_{tracer}"])))
    out["ldf_wet_nonzero"] = moved
    out["ldf_dry_nonzero"] = dry_moved
    out["completed_tracer_unequal"] = completed
    for name in ("T", "S", "u", "v", "ssh"):
        out[f"entry_{name}_unequal"] = int(np.count_nonzero(
            np.asarray(l2_entry[name]) != np.asarray(g5_entry[name])))
    out["barotropic_checked"] = check_barotropic
    if check_barotropic:
        l2_bt = read_bt_frame(l2_dir / BT_NAME, expect_step=1)
        g5_bt = read_bt_frame(g5_dir / BT_NAME, expect_step=1)
        out["barotropic_unequal"] = int(l2_bt["Kaa"] != g5_bt["Kaa"]) + sum(
            int(np.count_nonzero(
                np.asarray(l2_bt[name]) != np.asarray(g5_bt[name])))
            for name in ("uu_b", "vv_b", "un_adv", "vn_adv")
        )
    else:
        out["barotropic_unequal"] = 0
    out["pass"] = bool(
        moved > 0 and dry_moved == 0 and completed > 0
        and out["entry_T_unequal"] + out["entry_S_unequal"] > 0
        and out["entry_u_unequal"] == out["entry_v_unequal"]
        == out["entry_ssh_unequal"] == out["barotropic_unequal"] == 0)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path, required=True, metavar="L2_DIR")
    parser.add_argument("--g5", type=Path, default=G5_ROOT)
    parser.add_argument("--domain", type=Path, default=DOMAIN)
    parser.add_argument("--require-barotropic", action="store_true")
    args = parser.parse_args(argv)
    result = check_l2(
        args.check, args.g5, args.domain,
        check_barotropic=args.require_barotropic)
    if args.require_barotropic:
        result["pass"] = bool(result["pass"] and result["barotropic_checked"])
    print("L2_ISO_ACTING " + " ".join(f"{k}={v}" for k, v in result.items()))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
