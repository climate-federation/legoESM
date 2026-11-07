#!/usr/bin/env python3
"""Fail-closed Phase-1 gate for the NEMO 5.0.2 ORCA2 Lane-4 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np

NX, NY, NZ, NTR = 94, 152, 31, 2
NLAY_I, NLAY_S = 10, 5
N2 = NX * NY
N3 = N2 * NZ
NI = (NX - 4) * (NY - 4)
NI3 = NI * NZ
ODD_KT = (1, 3, 5, 7, 9)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_inventory() -> set[str]:
    names = {
        "oracle_bt_advmean_operands_kt00000001.bin",
        "oracle_bt_drag_operands_kt00000001.bin",
        "oracle_bt_ene_coeff_kt00000001.bin",
        "oracle_bt_ordered_operands_kt00000001.bin",
        "oracle_bt_substeps_kt00000001.bin",
        "oracle_qsr_stage3_kt00000001.bin",
        "oracle_rgb_chl_kt00000001.bin",
        "oracle_rhs_kt00000001.bin",
        "oracle_rkstage1_transport_operands_kt00000001.bin",
        "oracle_rkstage2_eos_operands_kt00000001.bin",
        "oracle_rkstage2_hpg_literal_kt00000001.bin",
        "oracle_rkstage2_hpg_operands_kt00000001.bin",
        "oracle_rkstage2_operands_kt00000001.bin",
        "oracle_rkstage2_preupdate_kt00000001.bin",
        "oracle_rkstage2_terms_kt00000001.bin",
        "oracle_rkstage3_wzv_kt00000001.bin",
        "oracle_rktracer_operands_kt00000001_s1.bin",
        "oracle_rktracer_operands_kt00000001_s2.bin",
        "oracle_rktracer_stage3_kt00000001.bin",
        "oracle_si3_bulk_operands.bin",
        "oracle_si3_exchange_frames.bin",
        "oracle_si3_reassoc_operands.bin",
        "oracle_si3_thd_frames.bin",
        "oracle_si3_zdf_inputs.bin",
        "oracle_slow_forcing_kt00000001.bin",
        "oracle_tracer_transport_kt00000001_s3.bin",
        "oracle_zdf_entry_kt00000001.bin",
    }
    names |= {f"oracle_step_entry_kt{kt:08d}.bin" for kt in range(1, 11)}
    names |= {f"oracle_bt_frames_kt{kt:08d}.bin" for kt in range(1, 11)}
    names |= {
        f"oracle_stage_kt{kt:08d}_s{stage}.bin"
        for kt in range(1, 11) for stage in range(1, 4)
    }
    names |= {f"oracle_transport_kt00000001_s{stage}.bin" for stage in range(1, 4)}
    names |= {
        f"oracle_si3_prather_kt{kt:08d}_s{stage}.bin"
        for kt in ODD_KT for stage in (0, 1)
    }
    require(len(names) == 90, f"internal inventory has {len(names)} files")
    return names


def _header(path: Path, magic: str, fmt: str) -> tuple[tuple[int, ...], int]:
    size = struct.calcsize(fmt)
    with path.open("rb", buffering=0) as handle:
        raw_magic = handle.read(16)
        require(len(raw_magic) == 16, f"{path.name}: truncated magic")
        require(raw_magic.decode("ascii").rstrip() == magic, f"{path.name}: bad magic")
        raw = handle.read(size)
        require(len(raw) == size, f"{path.name}: truncated header")
    return struct.unpack(fmt, raw), 16 + size


def _finite_payload(path: Path, offset: int, count: int) -> None:
    expected = offset + 8 * count
    require(path.stat().st_size == expected,
            f"{path.name}: size {path.stat().st_size} != schema {expected}")
    values = np.memmap(path, dtype=np.float64, mode="r", offset=offset, shape=(count,))
    block = 1_000_000
    for start in range(0, count, block):
        require(bool(np.isfinite(values[start:start + block]).all()),
                f"{path.name}: non-finite payload at/after value {start}")


def _fixed(path: Path, magic: str, fmt: str, wanted: tuple[int, ...], count: int) -> dict:
    header, offset = _header(path, magic, fmt)
    require(header == wanted, f"{path.name}: header {header} != {wanted}")
    _finite_payload(path, offset, count)
    return {"magic": magic, "header": list(header), "payload_f64": count,
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def _read_exact(handle, size: int, label: str) -> bytes:
    data = handle.read(size)
    require(len(data) == size, f"{label}: truncated ({len(data)}/{size})")
    return data


def _read_finite(handle, count: int, label: str) -> None:
    left = count
    while left:
        take = min(left, 1_000_000)
        raw = _read_exact(handle, take * 8, label)
        require(bool(np.isfinite(np.frombuffer(raw, dtype=np.float64)).all()),
                f"{label}: non-finite payload")
        left -= take


def _append_magic(handle, wanted: str, label: str) -> None:
    raw = _read_exact(handle, 16, label)
    require(raw.decode("ascii").rstrip() == wanted, f"{label}: bad magic")


def _parse_bulk(path: Path) -> dict:
    expected = [(kt, stage, n) for kt in ODD_KT for stage, n in ((0, 17), (1, 50), (2, 39))]
    rows = []
    with path.open("rb", buffering=0) as handle:
        for kt, stage, nvalue in expected:
            _append_magic(handle, "NEMO_L3BULK_001", path.name)
            header = struct.unpack("=5i", _read_exact(handle, 20, path.name))
            require(header == (1, kt, stage, nvalue, 64), f"{path.name}: bad header {header}")
            _read_finite(handle, nvalue, path.name)
            rows.append(list(header))
        require(handle.read(1) == b"", f"{path.name}: trailing/interleaved payload")
    return {"frames": len(rows), "headers": rows}


def _parse_exchange(path: Path) -> dict:
    rows = []
    payload = 78 * NI + 9 * N2 + (NX - 2) * (NY - 2)
    with path.open("rb", buffering=0) as handle:
        for kt in range(1, 11):
            _append_magic(handle, "NEMO_L3XCHG_001", path.name)
            header = struct.unpack("=6i", _read_exact(handle, 24, path.name))
            require(header == (1, kt, NX, NY, 5, 64), f"{path.name}: bad header {header}")
            _read_finite(handle, payload, path.name)
            rows.append(list(header))
        require(handle.read(1) == b"", f"{path.name}: trailing/interleaved payload")
    return {"frames": len(rows), "headers": rows, "payload_f64_per_frame": payload}


def _parse_thd(path: Path) -> dict:
    rows = []
    expected_stages = []
    for kt in ODD_KT:
        expected_stages.extend([(kt, 0, 0)])
        expected_stages.extend((kt, stage, 1) for _category in range(5) for stage in range(1, 6))
        expected_stages.extend(((kt, 6, 0), (kt, 7, 0)))
    with path.open("rb", buffering=0) as handle:
        for kt, stage, payload_kind in expected_stages:
            _append_magic(handle, "NEMO_L3THD_001", path.name)
            h = struct.unpack("=11i", _read_exact(handle, 44, path.name))
            version, got_kt, got_stage, got_kind, nx, ny, jpl, nli, nls, npti, bits = h
            require((version, got_kt, got_stage, got_kind, nx, ny, jpl, nli, nls, bits)
                    == (1, kt, stage, payload_kind, NX, NY, 5, 10, 5, 64),
                    f"{path.name}: bad header {h}")
            if payload_kind:
                require(npti > 0, f"{path.name}: empty compressed category")
                count = (4 + 2 * nli + nls) * npti
            else:
                count = (9 + 2 * nli + nls) * jpl * nx * ny
            _read_finite(handle, count, path.name)
            rows.append(list(h))
        require(handle.read(1) == b"", f"{path.name}: trailing/interleaved payload")
    return {"frames": len(rows), "headers": rows}


def _parse_zdf(path: Path) -> dict:
    rows = []
    with path.open("rb", buffering=0) as handle:
        for kt in ODD_KT:
            for category in range(1, 6):
                _append_magic(handle, "NEMO_L3ZIN_002", path.name)
                h = struct.unpack("=6i", _read_exact(handle, 24, path.name))
                version, got_kt, got_category, npti, bits, count = h
                require((version, got_kt, got_category, bits) == (2, kt, category, 64),
                        f"{path.name}: bad header {h}")
                expected_count = (13 + NLAY_S) * npti
                require(npti > 0 and count == expected_count,
                        f"{path.name}: bad payload size header {h}")
                _read_finite(handle, count, path.name)
                rows.append(list(h))
        require(handle.read(1) == b"", f"{path.name}: trailing/interleaved payload")
    return {"frames": len(rows), "headers": rows}


def _parse_reassoc(path: Path) -> dict:
    rows = []
    with path.open("rb", buffering=0) as handle:
        for category in range(1, 6):
            _append_magic(handle, "NEMO_L3REA_001", path.name)
            h = struct.unpack("=8i", _read_exact(handle, 32, path.name))
            version, kt, got_category, npti, nli, nls, bits, count = h
            require((version, kt, got_category, nli, nls, bits)
                    == (1, 3, category, 10, 5, 64), f"{path.name}: bad header {h}")
            expected_count = (3 + 3 * nli + 2 * nls) * npti
            require(npti > 0 and count == expected_count,
                    f"{path.name}: bad payload size header {h}")
            _read_finite(handle, count, path.name)
            rows.append(list(h))
        require(handle.read(1) == b"", f"{path.name}: trailing/interleaved payload")
    return {"frames": len(rows), "headers": rows}


def _parse_bt(path: Path, family: str) -> dict:
    magic = {"substeps": "NEMO_L2_BTSUB_2", "drag": "NEMO_L2_BTDRG_1",
             "advmean": "NEMO_L2_BTADV_2", "ordered": "NEMO_L2_BTORD_2"}[family]
    with path.open("rb", buffering=0) as handle:
        _append_magic(handle, magic, path.name)
        h = struct.unpack("=6i", _read_exact(handle, 24, path.name))
        version, kt, ncycle, nx, ny, bits = h
        require((kt, nx, ny, bits) == (1, NX, NY, 64), f"{path.name}: bad header {h}")
        if family == "substeps":
            require((version, ncycle) == (2, 65), f"{path.name}: bad header {h}")
            for jn in range(1, 66):
                require(struct.unpack("=i", _read_exact(handle, 4, path.name))[0] == jn,
                        f"{path.name}: bad substep sequence")
                _read_finite(handle, 18 * N2 + 2 * NI, path.name)
        elif family == "drag":
            require((version, ncycle) == (1, 65), f"{path.name}: bad header {h}")
            _read_finite(handle, 2 * N2, path.name)
            for jn in range(1, 66):
                require(struct.unpack("=i", _read_exact(handle, 4, path.name))[0] == jn,
                        f"{path.name}: bad substep sequence")
                _read_finite(handle, 12 * N2, path.name)
        elif family == "advmean":
            require((version, ncycle) == (2, 65), f"{path.name}: bad header {h}")
            _read_finite(handle, 1 + 65 + 2 * N2, path.name)
            for jn in range(1, 66):
                require(struct.unpack("=i", _read_exact(handle, 4, path.name))[0] == jn,
                        f"{path.name}: bad substep sequence")
                _read_finite(handle, 1 + 10 * N2, path.name)
            _read_finite(handle, 4 * N2, path.name)
        else:
            require((version, ncycle) == (2, 2), f"{path.name}: bad header {h}")
            _read_finite(handle, 1, path.name)  # rDt_e
            for jn in range(1, 3):
                require(struct.unpack("=i", _read_exact(handle, 4, path.name))[0] == jn,
                        f"{path.name}: bad substep sequence")
                _read_finite(handle, 7 + 41 * N2 + 10 * NI, path.name)
        require(handle.read(1) == b"", f"{path.name}: trailing payload")
    return {"frames": ncycle, "header": list(h), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def _manifest(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        result[Path(name.strip().lstrip("*")).name] = digest
    return result


def validate_records(root: Path, record_manifest: Path | None = None) -> dict:
    expected = expected_inventory()
    actual = {p.name for p in root.glob("oracle_*.bin") if p.is_file()}
    require(actual == expected,
            f"inventory mismatch missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    if record_manifest:
        hashes = _manifest(record_manifest)
        require(set(hashes) == expected, "record manifest inventory mismatch")
        for name in sorted(expected):
            require(sha256(root / name) == hashes[name], f"{name}: SHA-256 mismatch")

    rows: dict[str, object] = {}
    for kt in range(1, 11):
        level = 1 if kt % 2 == 1 else 3
        rows[f"oracle_step_entry_kt{kt:08d}.bin"] = _fixed(
            root / f"oracle_step_entry_kt{kt:08d}.bin", "NEMO_L1_ENTRY_1", "=8i",
            (1, kt, level, NX, NY, NZ, NTR, 64), 4 * N3 + N2)
        bt_level = 3 if kt % 2 == 1 else 1
        rows[f"oracle_bt_frames_kt{kt:08d}.bin"] = _fixed(
            root / f"oracle_bt_frames_kt{kt:08d}.bin", "NEMO_L1_BTFRM_1", "=6i",
            (1, kt, bt_level, NX, NY, 64), 4 * N2)
        for stage, stage_level in ((1, 3 if kt % 2 else 1), (2, 2), (3, 3 if kt % 2 else 1)):
            name = f"oracle_stage_kt{kt:08d}_s{stage}.bin"
            rows[name] = _fixed(root / name, "NEMO_L1_STAGE_1", "=9i",
                                (1, kt, stage, stage_level, NX, NY, NZ, NTR, 64),
                                4 * N3 + N2)
    for stage, level in ((1, 1), (2, 3), (3, 2)):
        name = f"oracle_transport_kt00000001_s{stage}.bin"
        rows[name] = _fixed(root / name, "NEMO_L1_TRANSP_1", "=8i",
                            (1, 1, stage, level, NX, NY, NZ, 64), 3 * N3)
    fixed = {
        "oracle_rhs_kt00000001.bin": ("NEMO_L1_RHS___1", "=7i", (1, 1, 3, NX, NY, NZ, 64), 2*N3),
        "oracle_zdf_entry_kt00000001.bin": ("NEMO_L4_ZDF___2", "=7i", (2, 1, 1, NX, NY, NZ, 64), N3+3*NI3),
        "oracle_rkstage1_transport_operands_kt00000001.bin": ("NEMO_L2_TRPOP_2", "=8i", (2,1,1,1,NX,NY,NZ,64), 8*N3+10*N2),
        "oracle_rkstage2_terms_kt00000001.bin": ("NEMO_L2_RKTRM_1", "=9i", (1,1,2,3,2,NX,NY,NZ,64), 8*N3),
        "oracle_rkstage2_hpg_operands_kt00000001.bin": ("NEMO_L2_HPGOP_1", "=7i", (1,1,3,NX,NY,NZ,64), 3*N3),
        "oracle_rkstage2_hpg_literal_kt00000001.bin": ("NEMO_L2_HPGLT_1", "=8i", (1,1,3,2,NX,NY,NZ,64), 6*N3+2*N2),
        "oracle_rkstage2_preupdate_kt00000001.bin": ("NEMO_L2_RKPRE_1", "=11i", (1,1,2,1,3,2,2,NX,NY,NZ,64), 2*N3),
        "oracle_rkstage2_operands_kt00000001.bin": ("NEMO_L2_RKSTG_1", "=11i", (1,1,2,1,3,2,2,NX,NY,NZ,64), 10*N3+2*N2+2*NI),
        "oracle_rkstage2_eos_operands_kt00000001.bin": ("NEMO_L2_EOSOP_1", "=11i", (1,1,2,3,3,2,NX,NY,NZ,64,0), 6+52+13*N3),
        "oracle_tracer_transport_kt00000001_s3.bin": ("NEMO_L2_TRTRP_1", "=11i", (1,1,3,1,2,3,3,NX,NY,NZ,64), 3*N3),
        "oracle_rktracer_stage3_kt00000001.bin": ("NEMO_L2_RKTR3_1", "=11i", (1,1,3,1,2,3,3,NX,NY,NZ,64), 16*N3+3*N2),
        "oracle_qsr_stage3_kt00000001.bin": ("NEMO_L2_QSR___1", "=9i", (1,1,3,2,3,NX,NY,NZ,64), NI+N3),
        "oracle_rkstage3_wzv_kt00000001.bin": ("NEMO_L2_WZVOP_1", "=8i", (1,1,3,2,NX,NY,NZ,64), 3*N3),
        "oracle_rgb_chl_kt00000001.bin": ("NEMO_L4_CHL_001", "=8i", (1,1,2,NX,NY,NZ,1,64), 2*N2+NI+NZ),
        "oracle_bt_ene_coeff_kt00000001.bin": ("NEMO_L2_ENECO_1", "=7i", (1,1,1,3,NX,NY,64), 8*NI),
    }
    for name, schema in fixed.items():
        rows[name] = _fixed(root / name, *schema)
    for stage, levels in ((1, (1,1,3,3)), (2, (1,3,2,2))):
        name = f"oracle_rktracer_operands_kt00000001_s{stage}.bin"
        rows[name] = _fixed(root / name, "NEMO_L2_RKTRA_1", "=11i",
                            (1,1,stage,*levels,NX,NY,NZ,64), 15*N3+3*N2)
    for kt in ODD_KT:
        for stage in (0, 1):
            name = f"oracle_si3_prather_kt{kt:08d}_s{stage}.bin"
            rows[name] = _fixed(root/name, "NEMO_L4_PRA_001", "=11i",
                                (1,kt,stage,NX,NY,5,10,5,4,1,64), 825*N2)
    rows["oracle_slow_forcing_kt00000001.bin"] = _fixed(
        root/"oracle_slow_forcing_kt00000001.bin", "NEMO_L2_SLOW_2", "=15i",
        (2,1,1,3,NX,NY,NZ,64,N3,N3,N3,N3,N3,N3,NI), 6*N3+6*NI+8*N2+1)
    for family, name in (("substeps","oracle_bt_substeps_kt00000001.bin"),
                         ("drag","oracle_bt_drag_operands_kt00000001.bin"),
                         ("advmean","oracle_bt_advmean_operands_kt00000001.bin"),
                         ("ordered","oracle_bt_ordered_operands_kt00000001.bin")):
        rows[name] = _parse_bt(root/name, family)
    append = {
        "bulk": _parse_bulk(root/"oracle_si3_bulk_operands.bin"),
        "exchange": _parse_exchange(root/"oracle_si3_exchange_frames.bin"),
        "thermodynamics": _parse_thd(root/"oracle_si3_thd_frames.bin"),
        "zdf_inputs": _parse_zdf(root/"oracle_si3_zdf_inputs.bin"),
        "reassociation": _parse_reassoc(root/"oracle_si3_reassoc_operands.bin"),
    }
    for name in ("oracle_si3_bulk_operands.bin", "oracle_si3_exchange_frames.bin",
                 "oracle_si3_thd_frames.bin", "oracle_si3_zdf_inputs.bin",
                 "oracle_si3_reassoc_operands.bin"):
        rows[name] = {"bytes": (root/name).stat().st_size, "sha256": sha256(root/name)}
    return {"status": "PASS", "inventory_files": len(actual), "records": rows,
            "append_streams": append}


GENERATED_ALWAYS = [
    "layout.dat", "layout.nc", "mesh_mask_0000.nc", "mesh_mask_0001.nc",
    "output.init_0000.nc", "output.init_0001.nc", "output.init_ice_0000.nc",
    "output.init_ice_0001.nc", "output.namelist.dyn", "output.namelist.ice",
    "time.step",
]
OCEAN_RESTARTS = [
    f"ORCA2_00000010_restart_{rank:04d}.nc" for rank in (0, 1)
]
ICE_RESTARTS = [
    f"ORCA2_00000010_restart_ice_{rank:04d}.nc" for rank in (0, 1)
]
ICEBERG_RESTARTS = [
    f"ORCA2_00000010_restart_icb_{rank:04d}.nc" for rank in (0, 1)
]
ICEBERG_TRAJECTORIES = [
    f"trajectory_icebergs_00010101-00010102_{rank:04d}.nc"
    for rank in (0, 1)
]
HISTORY = [f"ORCA2_30h_00010101_00010102_grid_{grid}_{rank:04d}.nc"
           for grid in "TUVW" for rank in (0,1)]
TIMING_EXCLUDED = ["communication_report.txt", "timing.output", "timing_gnuplot.sh",
                   "timing_step.nc", "timing_ts_allmpi_step.nc",
                   "timing_tsum_allmpi_t1_t10.nc", "run.user.stdout.log",
                   "run.user.time.log", "run.launcher.log"]

# Rule-1 fail-closed coverage inventory. These are the variables provided by
# each accepted kt=10 restart shard, including coordinates/control metadata.
RESTART_VARIABLES = {
    "ocean": """nav_lon nav_lat nav_lev time_counter kt ndastp adatrj ntime nn_fsbc
        ssu_m ssv_m sst_m sss_m ssh_m e3t_m frq_m a_fwb a_fwb_b emp_corr en
        avt_k avm_k dissl sshbb_e ubb_e vbb_e sshb_e ub_e vb_e fraqsr_1lev rdt
        DELAY__r8_cflice DELAY_re8_fwb2 DELAY_im8_fwb2 sshn un vn tn sn uu_n
        vv_n ssha""".split(),
    "ice": """nav_lon nav_lat numcat time_counter stress1_i stress2_i stress12_i
        sxice syice sxxice syyice sxyice sxsn sysn sxxsn syysn sxysn sxa sya
        sxxa syya sxya sxage syage sxxage syyage sxyage
        sxc0_l01 syc0_l01 sxxc0_l01 syyc0_l01 sxyc0_l01
        sxc0_l02 syc0_l02 sxxc0_l02 syyc0_l02 sxyc0_l02
        sxc0_l03 syc0_l03 sxxc0_l03 syyc0_l03 sxyc0_l03
        sxc0_l04 syc0_l04 sxxc0_l04 syyc0_l04 sxyc0_l04
        sxc0_l05 syc0_l05 sxxc0_l05 syyc0_l05 sxyc0_l05
        sxe_l01 sye_l01 sxxe_l01 syye_l01 sxye_l01
        sxe_l02 sye_l02 sxxe_l02 syye_l02 sxye_l02
        sxe_l03 sye_l03 sxxe_l03 syye_l03 sxye_l03
        sxe_l04 sye_l04 sxxe_l04 syye_l04 sxye_l04
        sxe_l05 sye_l05 sxxe_l05 syye_l05 sxye_l05
        sxe_l06 sye_l06 sxxe_l06 syye_l06 sxye_l06
        sxe_l07 sye_l07 sxxe_l07 syye_l07 sxye_l07
        sxe_l08 sye_l08 sxxe_l08 syye_l08 sxye_l08
        sxe_l09 sye_l09 sxxe_l09 syye_l09 sxye_l09
        sxe_l10 sye_l10 sxxe_l10 syye_l10 sxye_l10
        sxsi_l01 sysi_l01 sxxsi_l01 syysi_l01 sxysi_l01
        sxsi_l02 sysi_l02 sxxsi_l02 syysi_l02 sxysi_l02
        sxsi_l03 sysi_l03 sxxsi_l03 syysi_l03 sxysi_l03
        sxsi_l04 sysi_l04 sxxsi_l04 syysi_l04 sxysi_l04
        sxsi_l05 sysi_l05 sxxsi_l05 syysi_l05 sxysi_l05
        sxsi_l06 sysi_l06 sxxsi_l06 syysi_l06 sxysi_l06
        sxsi_l07 sysi_l07 sxxsi_l07 syysi_l07 sxysi_l07
        sxsi_l08 sysi_l08 sxxsi_l08 syysi_l08 sxysi_l08
        sxsi_l09 sysi_l09 sxxsi_l09 syysi_l09 sxysi_l09
        sxsi_l10 sysi_l10 sxxsi_l10 syysi_l10 sxysi_l10
        sxap syap sxxap syyap sxyap sxvp syvp sxxvp syyvp sxyvp sxvl syvl
        sxxvl syyvl sxyvl snwice_mass snwice_mass_b nn_fsbc kt_ice
        DELAY__r8_cflice DELAY_re8_fwb2 DELAY_im8_fwb2 v_i v_s a_i t_su u_ice
        v_ice oa_i a_ip v_ip v_il e_s_l01 e_s_l02 e_s_l03 e_s_l04 e_s_l05
        e_i_l01 e_i_l02 e_i_l03 e_i_l04 e_i_l05 e_i_l06 e_i_l07 e_i_l08
        e_i_l09 e_i_l10 sv_i szv_i_l01 szv_i_l02 szv_i_l03 szv_i_l04
        szv_i_l05 szv_i_l06 szv_i_l07 szv_i_l08 szv_i_l09 szv_i_l10""".split(),
    "iceberg": """kount calving calving_hflx stored_ice stored_heat lon lat xi yj
        uvel vvel mass thickness width length number year day mass_scaling
        mass_of_bits heat_density""".split(),
}


def _resolved_icebergs(root: Path) -> bool:
    values = []
    for line in (root / "namelist_cfg").read_text().splitlines():
        code = line.split("!", 1)[0]
        match = re.match(
            r"\s*ln_icebergs\s*=\s*(\.true\.|\.false\.)",
            code,
            flags=re.IGNORECASE,
        )
        if match:
            values.append(match.group(1).lower() == ".true.")
    require(len(values) == 1, f"{root}: resolved ln_icebergs values {values}")
    return values[0]


def _identity_exact_files(control: Path, instrumented: Path) -> tuple[list[str], bool]:
    control_icebergs = _resolved_icebergs(control)
    instrumented_icebergs = _resolved_icebergs(instrumented)
    require(
        control_icebergs == instrumented_icebergs,
        "identity pair resolves different ln_icebergs values",
    )

    def optional(root: Path) -> set[str]:
        restarts = {p.name for p in root.glob("ORCA2_00000010_restart*.nc")}
        trajectories = {p.name for p in root.glob("trajectory_icebergs_*.nc")}
        return restarts | trajectories

    expected = set(OCEAN_RESTARTS + ICE_RESTARTS)
    if control_icebergs:
        expected |= set(ICEBERG_RESTARTS + ICEBERG_TRAJECTORIES)
    control_optional = optional(control)
    instrumented_optional = optional(instrumented)
    require(
        control_optional == instrumented_optional,
        "identity pair restart/trajectory inventories differ",
    )
    require(
        control_optional == expected,
        "restart/trajectory inventory differs from resolved ln_icebergs",
    )
    return GENERATED_ALWAYS + sorted(expected), control_icebergs


def _restart_inventory(root: Path, icebergs_enabled: bool) -> dict:
    from netCDF4 import Dataset
    patterns = {
        "ocean": "ORCA2_00000010_restart_{rank:04d}.nc",
        "ice": "ORCA2_00000010_restart_ice_{rank:04d}.nc",
        "iceberg": "ORCA2_00000010_restart_icb_{rank:04d}.nc",
    }
    result = {}
    active_families = ("ocean", "ice", "iceberg") if icebergs_enabled else (
        "ocean", "ice"
    )
    for family in active_families:
        pattern = patterns[family]
        wanted = RESTART_VARIABLES[family]
        shards = []
        for rank in (0, 1):
            path = root / pattern.format(rank=rank)
            # Rank 1 owns no iceberg particles in this endpoint, so NEMO
            # writes only the five gridded/counter variables there.
            rank_wanted = wanted[:5] if family == "iceberg" and rank == 1 else wanted
            with Dataset(path) as dataset:
                found = list(dataset.variables)
            require(found == rank_wanted,
                    f"{path.name}: restart variable inventory differs "
                    f"missing={sorted(set(rank_wanted)-set(found))} "
                    f"extra={sorted(set(found)-set(rank_wanted))}")
            shards.append({"file": path.name, "variables": len(found)})
        result[family] = {"count": len(wanted), "names": wanted, "shards": shards}
    if not icebergs_enabled:
        result["iceberg"] = {
            "count": 0,
            "names": [],
            "shards": [],
            "status": "INACTIVE_BY_RESOLVED_NAMELIST",
        }
    return result


def _netcdf_equal_except_timestamp(a: Path, b: Path) -> None:
    from netCDF4 import Dataset
    with Dataset(a) as left, Dataset(b) as right:
        require(left.dimensions.keys() == right.dimensions.keys(), f"{a.name}: dimensions differ")
        for name in left.dimensions:
            require(len(left.dimensions[name]) == len(right.dimensions[name]), f"{a.name}: dim {name}")
        require(left.variables.keys() == right.variables.keys(), f"{a.name}: variables differ")
        attrs = (set(left.ncattrs()) | set(right.ncattrs())) - {"TimeStamp"}
        for attr in attrs:
            require(np.array_equal(getattr(left, attr, None), getattr(right, attr, None)),
                    f"{a.name}: global attribute {attr} differs")
        for name in left.variables:
            lv, rv = left[name], right[name]
            require(lv.dimensions == rv.dimensions and lv.dtype == rv.dtype,
                    f"{a.name}: variable {name} metadata differs")
            require(set(lv.ncattrs()) == set(rv.ncattrs()), f"{a.name}: {name} attrs differ")
            for attr in lv.ncattrs():
                require(np.array_equal(getattr(lv, attr), getattr(rv, attr)),
                        f"{a.name}: {name}:{attr} differs")
            # Compare the raw stored values, including fill values.  Automatic
            # masking would otherwise make differing bytes at masked elements
            # compare equal.
            lv.set_auto_maskandscale(False)
            rv.set_auto_maskandscale(False)
            left_raw = np.ascontiguousarray(lv[:])
            right_raw = np.ascontiguousarray(rv[:])
            require(left_raw.tobytes(order="C") == right_raw.tobytes(order="C"),
                    f"{a.name}: {name} raw payload differs")


def _normalized_ocean(path: Path) -> list[str]:
    pattern = re.compile(r"(?:LANE[124]|L2_).*(?:DUMP|dump)")
    return [line for line in path.read_text(errors="strict").splitlines() if not pattern.search(line)]


def validate_identity(control: Path, instrumented: Path) -> dict:
    rows = []
    restart_shards_exact = 0
    history_payloads_exact = 0
    exact_files, icebergs_enabled = _identity_exact_files(control, instrumented)
    for name in exact_files:
        require((control/name).read_bytes() == (instrumented/name).read_bytes(),
                f"identity mismatch: {name}")
        rows.append({"file": name, "status": "EXACT_BYTES", "sha256": sha256(control/name)})
        if "_restart" in name:
            restart_shards_exact += 1
    for name in HISTORY:
        _netcdf_equal_except_timestamp(control/name, instrumented/name)
        rows.append({"file": name, "status": "EXACT_NETCDF_EXCEPT_GLOBAL_TIMESTAMP",
                     "control_sha256": sha256(control/name),
                     "instrumented_sha256": sha256(instrumented/name)})
        history_payloads_exact += 1
    require(_normalized_ocean(control/"ocean.output") == _normalized_ocean(instrumented/"ocean.output"),
            "ocean.output differs after removing WRITE-only dump notices")
    rows.append({"file": "ocean.output", "status": "EXACT_TEXT_EXCEPT_DUMP_NOTICES",
                 "control_sha256": sha256(control/"ocean.output"),
                 "instrumented_sha256": sha256(instrumented/"ocean.output")})
    rows.extend({"file": name, "status": "EXCLUDED_TIMING_OR_LAUNCH_PROVENANCE"}
                for name in TIMING_EXCLUDED)
    return {"status": "PASS", "rows": rows,
            "restart_shards_exact": restart_shards_exact,
            "history_payloads_exact": history_payloads_exact,
            "icebergs_enabled": icebergs_enabled,
            "restart_variable_inventory": _restart_inventory(
                instrumented, icebergs_enabled
            )}


def planted_controls(root: Path, manifest: Path, control: Path, instrumented: Path) -> dict:
    results = {}
    inventory = expected_inventory()

    def overlay() -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="orca2-l4-plant-"))
        for name in inventory:
            os.symlink(root/name, tmp/name)
        return tmp

    def expect(name: str, mutate) -> None:
        tmp = overlay()
        try:
            mutate(tmp)
            try:
                validate_records(tmp, manifest)
            except GateError:
                results[name] = "PASS_NONZERO"
            else:
                raise GateError(f"planted control {name} did not fail")
        finally:
            shutil.rmtree(tmp)

    target = "oracle_bt_ene_coeff_kt00000001.bin"
    def materialize(tmp: Path, name: str) -> Path:
        (tmp/name).unlink()
        shutil.copyfile(root/name, tmp/name)
        return tmp/name
    expect("malformed_magic", lambda tmp: _patch(materialize(tmp,target), 0, b"X"))
    expect("wrong_time_level", lambda tmp: _patch(materialize(tmp,"oracle_bt_frames_kt00000001.bin"), 24, struct.pack("=i", 99)))
    expect("truncated_payload", lambda tmp: materialize(tmp,target).write_bytes((root/target).read_bytes()[:-8]))
    expect("trailing_byte", lambda tmp: _append(materialize(tmp,target), b"X"))
    expect("missing_file", lambda tmp: (tmp/target).unlink())
    expect("extra_file", lambda tmp: (tmp/"oracle_unregistered.bin").write_bytes(b"x"))
    expect("nan_payload", lambda tmp: _patch(materialize(tmp,target), 44, struct.pack("=d", float("nan"))))
    pra = "oracle_si3_prather_kt00000001_s0.bin"
    expect("si3_one_ulp", lambda tmp: _one_ulp(materialize(tmp,pra), 60))

    with tempfile.TemporaryDirectory(prefix="orca2-l4-identity-") as td:
        altered_root = Path(td)
        exact_files, _ = _identity_exact_files(control, instrumented)
        for name in exact_files + HISTORY + ["ocean.output", "namelist_cfg"]:
            os.symlink(instrumented/name, altered_root/name)
        target = "ORCA2_00000010_restart_0000.nc"
        (altered_root/target).unlink()
        shutil.copyfile(instrumented/target, altered_root/target)
        _patch(altered_root/target, 128, b"X")
        try:
            validate_identity(control, altered_root)
        except GateError:
            results["restart_identity_byte"] = "PASS_NONZERO"
        else:
            raise GateError("validate_identity accepted planted restart byte")

    with tempfile.TemporaryDirectory(prefix="orca2-l4-identity-inventory-") as td:
        altered_root = Path(td)
        for name in exact_files + HISTORY + ["ocean.output", "namelist_cfg"]:
            os.symlink(instrumented/name, altered_root/name)
        planted_extra = altered_root / ICEBERG_RESTARTS[0]
        if planted_extra.is_symlink():
            planted_extra.unlink()
        planted_extra.write_bytes(b"planted")
        try:
            validate_identity(control, altered_root)
        except GateError:
            results["restart_inventory_asymmetry"] = "PASS_NONZERO"
        else:
            raise GateError("validate_identity accepted asymmetric restart inventory")
    return results


def _patch(path: Path, offset: int, data: bytes) -> None:
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(data)


def _append(path: Path, data: bytes) -> None:
    with path.open("ab") as handle:
        handle.write(data)


def _one_ulp(path: Path, offset: int) -> None:
    with path.open("r+b") as handle:
        handle.seek(offset)
        raw = handle.read(8)
        value = struct.unpack("=d", raw)[0]
        handle.seek(offset)
        handle.write(struct.pack("=d", np.nextafter(value, np.inf)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--record-manifest", type=Path)
    parser.add_argument("--plant-controls", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = {"records": validate_records(args.root, args.record_manifest)}
        if args.control:
            result["identity"] = validate_identity(args.control, args.root)
        if args.plant_controls:
            require(args.record_manifest is not None and args.control is not None,
                    "planted controls require --record-manifest and --control")
            result["planted_controls"] = planted_controls(
                args.root, args.record_manifest, args.control, args.root)
        result["status"] = "PASS"
    except (GateError, OSError, UnicodeError, struct.error, ValueError) as exc:
        result = {"status": "FAIL", "error": str(exc)}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
