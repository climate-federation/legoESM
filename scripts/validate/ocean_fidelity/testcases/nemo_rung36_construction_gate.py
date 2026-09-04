#!/usr/bin/env python3
"""Fail-closed construction gate for the rung-3.6 coupled C1D oracle.

The first registered boundary is deliberately evaluated before any legoESM
ocean operator.  NEMO writes PRE_SSM after SI3 initialization, so this frame
contains both the ingested ERA5 T/S/U/V values and the SSH/e3 response to the
initial snow+ice mass.  A non-positive wet-layer thickness stops the ordered
walk; downstream frames are then UNMEASURED rather than silently scored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import struct
import subprocess
from pathlib import Path

import numpy as np

_EXCHANGE_GATE_PATH = Path(__file__).with_name("nemo_si3_exchange_drift_gate.py")
_EXCHANGE_SPEC = importlib.util.spec_from_file_location(
    "nemo_si3_exchange_drift_gate_for_rung36", _EXCHANGE_GATE_PATH
)
assert _EXCHANGE_SPEC and _EXCHANGE_SPEC.loader
exchange_gate = importlib.util.module_from_spec(_EXCHANGE_SPEC)
_EXCHANGE_SPEC.loader.exec_module(exchange_gate)

MAGIC = b"NEMO_L3SSM__001 "
HEADER = struct.Struct("=7i")
SOURCE = {
    "u_Kbb": 0.0,
    "v_Kbb": 0.0,
    "temperature_Kmm": -1.690032958984375,
    "salinity_Kmm": 34.0,
}
FIELDS = (
    "u_Kbb",
    "v_Kbb",
    "temperature_Kmm",
    "salinity_Kmm",
    "ssh_Kmm",
    "e3t_Kmm",
    "fraqsr_1lev",
    "ssu_carry",
    "ssv_carry",
    "sst_carry",
    "sss_carry",
    "ssh_carry",
    "e3t_carry",
    "frq_carry",
    "carry_defined",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_first_pre_ssm(path: Path) -> tuple[dict[str, int], dict[str, float]]:
    with path.open("rb") as stream:
        magic = stream.read(16)
        if magic != MAGIC:
            raise ValueError(f"bad PRE_SSM magic: {magic!r}")
        version, kt, kbb, kmm, stage, count, bits = HEADER.unpack(stream.read(HEADER.size))
        if (version, kt, stage, count, bits) != (1, 1, 0, 15, 64):
            raise ValueError("first frame is not registered kt=1 PRE_SSM fp64")
        values = np.fromfile(stream, dtype="=f8", count=count)
    return (
        {"version": version, "kt": kt, "Kbb": kbb, "Kmm": kmm, "stage": stage, "bits": bits},
        dict(zip(FIELDS, (float(value) for value in values), strict=True)),
    )


def _row(name: str, actual: float, expected: float, bar: float = 1.0e-15) -> dict[str, object]:
    absolute = abs(actual - expected)
    normalized = absolute / max(abs(expected), 1.0)
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "absolute_error": absolute,
        "normalized_error": normalized,
        "bit_identical": np.float64(actual).tobytes() == np.float64(expected).tobytes(),
        "bar": bar,
        "status": "AT_BAR" if normalized <= bar else "DEBT",
    }


def evaluate(run_root: Path, *, plant: str | None = None) -> dict[str, object]:
    stream = run_root / "oracle_rung36_ssm_frames.bin"
    registry, values = read_first_pre_ssm(stream)
    before_plant = dict(values)
    if plant == "temperature":
        values["temperature_Kmm"] = values["temperature_Kmm"] + 1.0e-8
    elif plant == "salinity":
        values["salinity_Kmm"] = values["salinity_Kmm"] + 1.0e-8
    elif plant == "geometry":
        values["e3t_Kmm"] = values["e3t_Kmm"] - 0.125
    elif plant is not None:
        raise ValueError(f"unknown plant {plant!r}")

    output = (run_root / "ocean.output").read_text(encoding="utf-8", errors="replace")
    namelist_path = run_root / "namelist_cfg"
    namelist = namelist_path.read_text(encoding="utf-8")
    rho_match = re.search(r"volumic mass of reference\s+rho0\s+=\s+([0-9.E+-]+)", output)
    r1_rho_match = re.search(r"1\. / rho0\s+r1_rho0\s+=\s+([0-9.E+-]+)", output)
    ssh_match = re.search(r"rn_ssh_init\s*=\s*([0-9.E+-]+)", namelist)
    bathy_match = re.search(r"rn_bathy\s*=\s*([0-9.E+-]+)", namelist)
    if not (rho_match and r1_rho_match and ssh_match and bathy_match):
        raise ValueError("missing resolved rho0/r1_rho0, rn_ssh_init, or rn_bathy")
    rho0 = float(rho_match.group(1))
    r1_rho0 = float(r1_rho_match.group(1))
    ssh_initial = float(ssh_match.group(1))
    bathy = float(bathy_match.group(1))
    exchange_stream = run_root / "oracle_si3_exchange_frames.bin"
    ice_mass = exchange_gate.read_active_value(exchange_stream, 1, "snwice_mass_b")

    rows = [_row(name, values[name], expected) for name, expected in SOURCE.items()]
    rows.append(
        _row(
            "ssh_from_initial_ice_mass",
            values["ssh_Kmm"],
            ssh_initial - ice_mass * r1_rho0,
        )
    )
    # The active key_qco expansion is domzgr_substitute.h90:126. For this
    # single full-step wet level E3t_0 is the configured bathymetry.
    rows.append(_row("e3t_source_expansion", values["e3t_Kmm"], bathy + values["ssh_Kmm"]))
    rows.append(
        {
            "name": "positive_wet_layer_thickness",
            "actual": values["e3t_Kmm"],
            "expected": "> 0",
            "absolute_error": None,
            "normalized_error": None,
            "bit_identical": False,
            "bar": 0.0,
            "status": "AT_BAR" if values["e3t_Kmm"] > 0.0 else "DEBT",
        }
    )

    abort = re.search(
        r"kt\s+(\d+) \|ssh\| max\s+([0-9.E+-]+).*?"
        r"kt\s+\1 \|U\|\s+max\s+([0-9.E+-]+).*?"
        r"kt\s+\1 \|V\|\s+max\s+([0-9.E+-]+)",
        output,
        re.S,
    )
    exe = run_root / "nemo.exe"
    nm = subprocess.run(
        ["nm", "-D", str(exe)], check=True, text=True, stdout=subprocess.PIPE
    ).stdout
    failures = [row["name"] for row in rows if row["status"] != "AT_BAR"]
    target = {
        "temperature": "temperature_Kmm",
        "salinity": "salinity_Kmm",
        "geometry": "e3t_Kmm",
    }.get(plant)
    plant_binding = None
    if target is not None:
        plant_binding = {
            "target": target,
            "before_bits": np.float64(before_plant[target]).view(np.uint64).item(),
            "after_bits": np.float64(values[target]).view(np.uint64).item(),
            "changed": bool(before_plant[target] != values[target]),
        }
    return {
        "verdict": "CONSTRUCTION_DEBT" if failures else "AT_BAR",
        "ordered_stop": "INITIAL_STATE.positive_wet_layer_thickness" if failures else None,
        "owner": "NEMO SI3 initial snow+ice mass sea-level adjustment on a one-metre column",
        "source_identity": {
            "initial_mass": "iceistate.F90:400-401",
            "levitating_global_adjustment": "iceistate.F90:408-426",
            "qco_thickness_refresh": "iceistate.F90:431-433",
            "e3t_expansion": "domzgr_substitute.h90:126",
            "one_metre_depth": "C1D_OMIP_L3_COUPLED_SM/EXP00/namelist_cfg:19",
        },
        "registry": registry,
        "rows": rows,
        "initial_mass_operands": {
            "rn_ssh_init_m": ssh_initial,
            "snwice_mass_b_kg_m2": ice_mass,
            "rho0_kg_m3": rho0,
            "r1_rho0_m3_kg": r1_rho0,
            "displacement_m": ice_mass * r1_rho0,
            "rn_bathy_m": bathy,
        },
        "run_abort": (
            {
                "step": int(abort.group(1)),
                "max_abs_ssh_m": float(abort.group(2)),
                "max_abs_u_m_s": float(abort.group(3)),
                "max_abs_v_m_s": float(abort.group(4)),
            }
            if abort
            else None
        ),
        "provenance": {
            "ssm_stream_sha256": sha256(stream),
            "exchange_stream_sha256": sha256(exchange_stream),
            "namelist_cfg_sha256": sha256(namelist_path),
            "initial_input_sha256": sha256(run_root / "C1D_OMIP_L3_COUPLED_init_v2.nc"),
            "nemo_exe_sha256": sha256(exe),
            "dynamic_ZGV_symbols": nm.count("_ZGV"),
        },
        "coverage": {
            "INITIAL_STATE_subset": (
                "VERIFIED T/S/U/V ingestion, namelist SSH, first exchange "
                "snwice_mass_b, rho0, and load-adjusted SSH/e3; exact complete "
                "ice state and pre-adjustment runtime slots UNMEASURED"
            ),
            "PRE_SSM": "VERIFIED at kt=1",
            "POST_SSM_and_later": "UNMEASURED: ordered walk stopped at INITIAL_STATE",
        },
        "plant": plant,
        "plant_binding": plant_binding,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--plant", choices=("temperature", "salinity", "geometry"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.run_root, plant=args.plant)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.plant is not None and not report["plant_binding"]["changed"]:
        return 2
    return 0 if report["verdict"] == "AT_BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
