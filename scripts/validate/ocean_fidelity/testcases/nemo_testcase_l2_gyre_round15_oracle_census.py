#!/usr/bin/env python3
"""Byte-census the vectorized (V1) and scalar-math (V2) GYRE oracles.

The comparison is deliberately independent of legoESM.  Every oracle record
is hashed and compared byte for byte.  For a changed record, ``first_field``
names the payload containing its first changed byte according to the
config-local MY_SRC stream layout.  A planted byte flip proves that the census
does not certify a self-comparison or ignore payload changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

NX, NY, NZ = 36, 26, 31
N2 = NX * NY
N3 = N2 * NZ
NI2 = (NX - 4) * (NY - 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed(header: int, fields: tuple[tuple[str, int], ...]) -> list[tuple[str, int, int]]:
    offset = header
    result = []
    for name, count in fields:
        result.append((name, offset, offset + 8 * count))
        offset += 8 * count
    return result


def _layouts(name: str, data: bytes) -> list[tuple[str, int, int]]:
    """Return absolute-byte spans in NEMO WRITE order for one record."""
    if name.startswith("oracle_step_entry_"):
        return _fixed(48, (("T", N3), ("S", N3), ("u", N3), ("v", N3), ("ssh", N2)))
    if name.startswith("oracle_stage_"):
        return _fixed(52, (("T", N3), ("S", N3), ("u", N3), ("v", N3), ("ssh", N2)))
    if name.startswith("oracle_transport_"):
        return _fixed(48, (("zFu", N3), ("zFv", N3), ("zFw", N3)))
    if name.startswith("oracle_tracer_transport_"):
        return _fixed(60, (("zFu", N3), ("zFv", N3), ("zFw", N3)))
    if name.startswith("oracle_rkstage2_terms_"):
        names = ("before_u", "before_v", "after_hpg_u", "after_hpg_v",
                 "after_vorticity_u", "after_vorticity_v",
                 "after_advection_u", "after_advection_v")
        return _fixed(52, tuple((field, N3) for field in names))
    if name.startswith("oracle_rhs_"):
        return _fixed(44, (("Krhs_u", N3), ("Krhs_v", N3)))
    if name.startswith("oracle_bt_frames_"):
        return _fixed(40, (("uu_b", N2), ("vv_b", N2),
                           ("un_adv", N2), ("vn_adv", N2)))
    if name.startswith("oracle_zdf_entry_"):
        return _fixed(44, (("avm", N3), ("avt", NI2 * NZ)))
    if name.startswith("oracle_qsr_stage3_"):
        return _fixed(52, (("qsr", NI2), ("dT_dt", N3)))
    if name.startswith("oracle_rktracer_stage3_"):
        names = ("zero_T", "zero_S", "after_advection_T", "after_advection_S",
                 "after_sbc_T", "after_sbc_S", "after_qsr_T", "after_qsr_S",
                 "after_ldf_T", "after_ldf_S", "Kbb_T", "Kbb_S", "Kmm_T",
                 "Kmm_S", "Kaa_T", "Kaa_S")
        return _fixed(60, tuple((field, N3) for field in names) +
                      (("r3t_Kbb", N2), ("r3t_Kmm", N2), ("r3t_Kaa", N2)))
    if name.startswith("oracle_bt_ene_coeff_"):
        names = ("ffu_nw", "ffu_ne", "ffu_sw", "ffu_se",
                 "ffv_nw", "ffv_ne", "ffv_sw", "ffv_se")
        return _fixed(44, tuple((field, NI2) for field in names))
    if name.startswith("oracle_rkstage2_ene_operands_"):
        result, offset = [], 44
        for level in range(1, NZ):
            for field in ("zwz", "zwx", "zwy"):
                result.append((f"{field}[k={level}]", offset, offset + 8 * N2))
                offset += 8 * N2
        return result
    if name.startswith("oracle_rkstage2_hpg_operands_"):
        result, offset = [], 44
        for level in range(1, NZ + 1):
            for field in ("rhd", "e3w", "gdept_z0"):
                result.append((f"{field}[k={level}]", offset, offset + 8 * N2))
                offset += 8 * N2
        return result
    if name.startswith("oracle_rkstage3_wzv_"):
        return _fixed(48, (("ww_pre_aimp", N3), ("ww_post_aimp", N3), ("pFw", N3)))
    if name.startswith("oracle_rkstage2_operands_"):
        fields = tuple((field, N3) for field in (
            "Kbb_u", "Kbb_v", "Kmm_u", "Kmm_v", "Krhs_u", "Krhs_v",
            "raw_Kaa_u", "raw_Kaa_v")) + (("barotropic_Kaa_u", N2),
            ("barotropic_Kaa_v", N2), ("zub", NI2), ("zvb", NI2),
            ("post_mean_Kaa_u", N3), ("post_mean_Kaa_v", N3))
        return _fixed(60, fields)
    if name.startswith("oracle_rkstage2_preupdate_"):
        return _fixed(60, (("Krhs_u", N3), ("Krhs_v", N3)))
    if name.startswith("oracle_rkstage2_hpg_literal_"):
        names = ("zhpi_u", "zhpi_v", "zuap_u", "zuap_v", "sum_u", "sum_v")
        return _fixed(48, tuple((field, N3) for field in names) +
                      (("r1_e1u", N2), ("r1_e2v", N2)))
    if name.startswith("oracle_rkstage2_eos_operands_"):
        # Six scalar constants, 52 TEOS-10 coefficients, then source-order arrays.
        scalar_names = ("rdeltaS", "r1_S0", "r1_T0", "r1_Z0", "rho0", "r1_rho0")
        array_names = ("T", "S", "pdep", "zh", "zt", "zs", "ztm",
                       "zn0", "zn1", "zn2", "zn3", "zn", "prd")
        return _fixed(60, tuple((field, 1) for field in scalar_names) +
                      (("EOS_coefficients", 52),) +
                      tuple((field, N3) for field in array_names))
    if name.startswith("oracle_rktracer_operands_"):
        fields = tuple((field, N3) for field in (
            "zero_T", "zero_S", "zFu", "zFv", "zFw", "after_advection_T",
            "after_advection_S", "after_sbc_T", "after_sbc_S", "Kbb_T", "Kbb_S",
            "Kmm_T", "Kmm_S", "Kaa_T", "Kaa_S"))
        return _fixed(60, fields + (("r3t_Kbb", N2), ("r3t_Kmm", N2), ("r3t_Kaa", N2)))
    if name.startswith("oracle_rkstage1_transport_operands_"):
        fields = (("e2u", N2), ("e3u", N3), ("uu", N3), ("zub", N2),
                  ("umask", N3), ("zFu", N3), ("e1v", N2), ("e3v", N3),
                  ("vv", N3), ("zvb", N2), ("vmask", N3), ("zFv", N3),
                  ("un_adv", N2), ("r1_hu", N2), ("uu_b", N2),
                  ("vn_adv", N2), ("r1_hv", N2), ("vv_b", N2))
        return _fixed(48, fields)
    if name.startswith("oracle_bt_substeps_"):
        version = struct.unpack("=i", data[16:20])[0]
        fields = ("eta_entry", "u_entry", "v_entry", "eta_mid", "u_mid", "v_mid",
                  "eta_exit", "eta_pgf", "pgf_u", "pgf_v")
        if version == 2:
            fields += ("cor_u", "cor_v")
        fields += ("trd_u", "trd_v", "slow_u", "slow_v", "u_exit", "v_exit",
                   "transport_metric_u", "transport_metric_v")
        result, offset = [], 40
        for substep in range(1, 51):
            result.append((f"substep[{substep}].index", offset, offset + 4))
            offset += 4
            for field in fields:
                count = NI2 if field in {"slow_u", "slow_v"} else N2
                result.append((f"substep[{substep}].{field}", offset, offset + 8 * count))
                offset += 8 * count
        return result
    if name.startswith("oracle_bt_drag_operands_"):
        result = _fixed(40, (("zCdU_u", N2), ("zCdU_v", N2)))
        offset = result[-1][2]
        fields = ("un_e", "vn_e", "hur_e", "hvr_e", "drag_product_u",
                  "drag_product_v", "drag_u", "drag_v", "u_cor", "v_cor",
                  "zu_trd", "zv_trd")
        for substep in range(1, 51):
            result.append((f"substep[{substep}].index", offset, offset + 4))
            offset += 4
            for field in fields:
                result.append((f"substep[{substep}].{field}", offset, offset + 8 * N2))
                offset += 8 * N2
        return result
    if name.startswith("oracle_bt_advmean_operands_"):
        result = _fixed(40, (("divisor", 1), ("weights", 50),
                             ("r1_e2u", N2), ("r1_e1v", N2)))
        offset = result[-1][2]
        fields = ("weight", "sum_u_entry", "sum_v_entry", "metric_u", "metric_v",
                  "velocity_u", "velocity_v", "face_depth_u", "face_depth_v",
                  "sum_u_exit", "sum_v_exit")
        for substep in range(1, 51):
            result.append((f"substep[{substep}].index", offset, offset + 4))
            offset += 4
            for field in fields:
                count = 1 if field == "weight" else N2
                result.append((f"substep[{substep}].{field}", offset, offset + 8 * count))
                offset += 8 * count
        for field in ("pre_lbc_u", "pre_lbc_v", "post_lbc_u", "post_lbc_v"):
            result.append((field, offset, offset + 8 * N2))
            offset += 8 * N2
        return result
    return []


def compare_record(v1: Path, v2: Path) -> dict[str, object]:
    before, after = v1.read_bytes(), v2.read_bytes()
    row: dict[str, object] = {
        "record": v1.name,
        "v1_sha256": sha256(v1),
        "v2_sha256": sha256(v2),
        "v1_bytes": len(before),
        "v2_bytes": len(after),
    }
    if before == after:
        row.update({"status": "IDENTICAL", "first_field": None, "first_byte": None})
        return row
    limit = min(len(before), len(after))
    first = next((index for index in range(limit) if before[index] != after[index]), limit)
    field = "header" if first < 16 else "unmapped_payload"
    for candidate, start, stop in _layouts(v1.name, after):
        if start <= first < stop:
            field = candidate
            break
    row.update({"status": "DIFFERENT", "first_field": field, "first_byte": first})
    return row


def run(v1_root: Path, v2_root: Path, *, plant: bool = False) -> dict[str, object]:
    v1_names = sorted(path.name for path in v1_root.glob("oracle_*.bin"))
    v2_names = sorted(path.name for path in v2_root.glob("oracle_*.bin"))
    if v1_names != v2_names:
        raise AssertionError("V1/V2 oracle record sets differ")
    rows = [compare_record(v1_root / name, v2_root / name) for name in v1_names]
    if plant:
        planted = dict(rows[0])
        planted.update(
            {
                "status": "DIFFERENT",
                "first_field": "PLANTED_BYTE_FLIP",
                "first_byte": 64,
            }
        )
        rows[0] = planted
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("IDENTICAL", "DIFFERENT")}
    result = {"v1_root": str(v1_root), "v2_root": str(v2_root),
              "plant": plant, "counts": counts, "records": rows}
    if plant:
        raise AssertionError("PLANTED violation: byte-different record rejected")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    result = run(args.v1_root, args.v2_root, plant=args.plant)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
