#!/usr/bin/env python3
"""Fail-closed Phase-1 gate for the NEMO C1D_OMIP_L3 SI3 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import netCDF4
import numpy as np


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


STAGES = {
    0: ("ENTRY", "now", "icethd.F90:86-115"),
    1: ("POST_ZDF", "now", "icethd.F90:148"),
    2: ("POST_DH", "now", "icethd.F90:150"),
    3: ("POST_TEMP1", "now", "icethd.F90:152"),
    4: ("POST_SAL", "now", "icethd.F90:154"),
    5: ("POST_TEMP2", "now", "icethd.F90:156"),
    6: ("POST_DO", "now", "icethd.F90:181"),
    7: ("EXIT", "now", "icethd.F90:183-216"),
}

REQUIRED_RESTART = {
    "nn_fsbc", "kt_ice", "v_i", "v_s", "a_i", "t_su", "u_ice", "v_ice",
    "oa_i", "a_ip", "v_ip", "v_il", "sv_i", "snwice_mass", "snwice_mass_b",
    *(f"e_s_l{i:02d}" for i in range(1, 4)),
    *(f"e_i_l{i:02d}" for i in range(1, 4)),
    *(f"szv_i_l{i:02d}" for i in range(1, 4)),
}
META_RESTART = {"nav_lon", "nav_lat", "time_counter", "x", "y"}
WAIVED_RESTART = {
    "cnd_ice": "Jules coupling inactive",
    "t1_ice": "Jules coupling inactive",
    "stress1_i": "ln_c1d skips ice dynamics",
    "stress2_i": "ln_c1d skips ice dynamics",
    "stress12_i": "ln_c1d skips ice dynamics",
}


def parse_logical(token: str) -> bool:
    value = token.strip().strip(",").replace(".", "").upper()
    if value in {"T", "TRUE"}:
        return True
    if value in {"F", "FALSE"}:
        return False
    raise GateError(f"invalid Fortran logical {token!r}")


def namelist_values(path: Path) -> dict[str, str]:
    section = ""
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("!", 1)[0].strip()
        if line.startswith("&"):
            section = line[1:].split()[0].lower()
        elif line == "/":
            section = ""
        elif section and "=" in line:
            key, value = line.split("=", 1)
            key = key.strip().lower()
            if re.fullmatch(r"[a-z][a-z0-9_]*", key):
                values[f"{section}.{key}"] = value.strip().rstrip(",")
    return values


def check_resolved(root: Path) -> dict[str, str]:
    ocean = namelist_values(root / "output.namelist.dyn")
    ice = namelist_values(root / "output.namelist.ice")
    expected_bool = {
        "namsbc_blk.ln_ncar": True, "namsbc_blk.ln_ecmwf": False,
        "namsbc_blk.ln_cx_ice_cst": True, "nameos.ln_teos10": True,
        "nameos.ln_eos80": False, "namdom.ln_c1d": True,
        "nampar.ln_icethd": True, "namitd.ln_cat_hfn": True,
        "namthd.ln_icedh": True, "namthd.ln_iceda": False,
        "namthd.ln_icedo": True, "namthd.ln_leadhfx": True,
        "namthd_zdf.ln_zdf_bl99": True,
        "namthd_zdf.ln_cndi_p07": True, "namthd_zdf.ln_cndi_u64": False,
        "namthd_sal.ln_flushing": True, "namthd_sal.ln_drainage": True,
        "namthd_pnd.ln_pnd": False,
    }
    merged = ocean | ice
    for key, wanted in expected_bool.items():
        require(key in merged, f"resolved selector missing: {key}")
        require(parse_logical(merged[key]) is wanted, f"resolved selector differs: {key}")
    expected_num = {
        "namrun.nn_it000": 1, "namrun.nn_itend": 8760, "namrun.nn_stock": 8760,
        "namdom.rn_dt": 3600, "namsbc.nn_fsbc": 1, "namsbc.nn_ice": 2,
        "namsbc_blk.rn_cd_ia": 1e-3, "namsbc_blk.rn_ce_ia": 1e-3,
        "namsbc_blk.rn_ch_ia": 1e-3, "nampar.jpl": 1, "nampar.nlay_i": 3,
        "nampar.nlay_s": 3, "namitd.rn_himin": .05, "namitd.rn_himax": 99,
        "namthd_zdf.rn_cnd_s": .5, "namthd_do.rn_hinew": .05,
        "namthd_sal.nn_icesal": 2, "namthd_sal.rn_sinew": .75,
    }
    for key, wanted in expected_num.items():
        require(key in merged, f"resolved value missing: {key}")
        got = float(merged[key].replace("D", "E").replace("d", "e"))
        require(got == wanted, f"resolved value differs: {key}={got}")
    return merged


def read_thd_frames(path: Path, plant: bool = False, nsteps: int = 8760) -> tuple[list[float], dict]:
    thickness: list[float] = []
    count = 0
    with path.open("rb") as handle:
        for kt in range(1, nsteps + 1):
            for wanted_stage in range(8):
                magic = handle.read(16)
                require(magic == b"NEMO_L3THD_001  ", f"thermo magic at frame {count}")
                raw = handle.read(44)
                require(len(raw) == 44, "truncated thermo header")
                version, step, stage, payload, nx, ny, nc, ni, ns, npti, bits = struct.unpack("=11i", raw)
                require((version, step, stage, nc, ni, ns, bits) == (1, kt, wanted_stage, 1, 3, 3, 64),
                        f"thermo frame registry violation at step {kt} stage {wanted_stage}")
                require(stage in STAGES and STAGES[stage][1] == "now", "unregistered time level")
                n2 = nx * ny
                if payload == 0:
                    nval = n2 * nc * 9 + n2 * nc * (ni + ns + ni)
                elif payload == 1:
                    nval = npti * 4 + npti * (ni + ns + ni)
                else:
                    raise GateError(f"unknown thermo payload {payload}")
                values = np.fromfile(handle, dtype=np.float64, count=nval)
                require(values.size == nval, "truncated thermo payload")
                require(np.all(np.isfinite(values)), f"non-finite thermo payload step {kt} stage {stage}")
                if stage == 7:
                    cube = n2 * nc
                    area = values[:cube].reshape((nx, ny, nc), order="F")
                    volume = values[cube:2*cube].reshape((nx, ny, nc), order="F")
                    i, j = nx // 2, ny // 2
                    require(area[i, j].sum() > 0, f"no ice at EXIT step {kt}")
                    thickness.append(float(volume[i, j].sum() / area[i, j].sum()))
                count += 1
        require(handle.read(1) == b"", "trailing thermo bytes")
    require(count == nsteps * 8, "wrong thermo frame count")
    series = np.asarray(thickness)
    if plant:
        series[min(100, series.size - 1)] = 100.0
    require(np.all((series >= .05) & (series <= 99.0)), "HFN thickness bound violation")
    changes = series[24:] - series[:-24]
    require(np.any(changes > 0), "documented seasonal growth absent")
    require(np.any(changes < 0), "documented seasonal melt absent")
    return thickness, {"frames": count, "min_thickness_m": float(series.min()),
                       "max_thickness_m": float(series.max()),
                       "max_24h_growth_m": float(changes.max()),
                       "max_24h_melt_m": float(changes.min())}


def read_exchange_frames(path: Path, plant: bool = False, nsteps: int = 8760) -> dict:
    count = 0
    fr_min, fr_max = np.inf, -np.inf
    with path.open("rb") as handle:
        for kt in range(1, nsteps + 1):
            require(handle.read(16) == b"NEMO_L3XCHG_001 ", f"exchange magic step {kt}")
            raw = handle.read(24)
            require(len(raw) == 24, "truncated exchange header")
            version, step, nx, ny, nc, bits = struct.unpack("=6i", raw)
            require((version, step, nc, bits) == (1, kt, 1, 64), f"exchange registry step {kt}")
            n2 = nx * ny
            nval = 13 * n2 * nc + 23 * n2
            values = np.fromfile(handle, dtype=np.float64, count=nval)
            require(values.size == nval, "truncated exchange payload")
            require(np.all(np.isfinite(values)), f"non-finite exchange payload step {kt}")
            fr = values[-n2:].copy()
            if plant and kt == 1:
                fr[0] += 2.0
            require(np.all((fr >= 0) & (fr <= 1)), f"exchange fr_i bound step {kt}")
            fr_min, fr_max = min(fr_min, float(fr.min())), max(fr_max, float(fr.max()))
            count += 1
        require(handle.read(1) == b"", "trailing exchange bytes")
    require(count == nsteps, "wrong exchange frame count")
    return {"frames": count, "fr_i_min": fr_min, "fr_i_max": fr_max}


def check_restart(path: Path, plant: bool = False) -> dict:
    with netCDF4.Dataset(path) as ds:
        actual = set(ds.variables)
        if plant:
            actual.add("PLANTED_UNACCOUNTED_RESTART_ARRAY")
        missing = REQUIRED_RESTART - actual
        unknown = actual - REQUIRED_RESTART - META_RESTART
        require(not missing, f"restart required missing={sorted(missing)}")
        require(not unknown, f"restart coverage unaccounted={sorted(unknown)}")
        for name in REQUIRED_RESTART:
            data = np.asarray(ds.variables[name][:])
            require(data.dtype == np.float64, f"restart {name} is not fp64")
            require(np.all(np.isfinite(data)), f"restart {name} non-finite")
    return {"verified": sorted(REQUIRED_RESTART), "waived_inactive": WAIVED_RESTART,
            "waived_metadata": sorted(actual & META_RESTART)}


def run(root: Path, forcing: Path, archive: Path, plant_restart: bool = False,
        plant_exchange: bool = False, plant_thickness: bool = False) -> dict:
    require(sha256(archive) == "54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382",
            "forcing archive SHA256")
    require(sha256(forcing) == "e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe",
            "forcing file SHA256")
    resolved = check_resolved(root)
    restarts = sorted(root.glob("*restart_ice*.nc"))
    require(len(restarts) == 1, f"expected one final ice restart, got {len(restarts)}")
    result = {
        "status": "VERIFIED",
        "forcing_sha256": sha256(forcing),
        "archive_sha256": sha256(archive),
        "resolved_sha256": {"ocean": sha256(root / "output.namelist.dyn"),
                            "ice": sha256(root / "output.namelist.ice")},
        "thermodynamics": read_thd_frames(root / "oracle_si3_thd_frames.bin", plant_thickness)[1],
        "exchange": read_exchange_frames(root / "oracle_si3_exchange_frames.bin", plant_exchange),
        "restart": check_restart(restarts[0], plant_restart),
        "resolved_count": len(resolved),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--forcing", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--plant-restart", action="store_true")
    parser.add_argument("--plant-exchange", action="store_true")
    parser.add_argument("--plant-thickness", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = run(args.run_root, args.forcing, args.archive, args.plant_restart,
                 args.plant_exchange, args.plant_thickness)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
