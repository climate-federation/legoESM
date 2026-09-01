#!/usr/bin/env python3
"""Fail-closed oracle-only gate for SI3 lane-3 dynamics rungs 3.1--3.3.

The gate inventories the files NEMO actually wrote.  Every mesh/restart/
resolved-namelist item must have exactly one VERIFIED, WAIVED, or UNMEASURED
disposition.  Phase 1 makes no legoESM comparison.
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

import netCDF4
import numpy as np

_LANE1_PATH = Path(__file__).with_name("nemo_testcase_oracle_gate.py")
_SPEC = importlib.util.spec_from_file_location("nemo_testcase_oracle_gate_l1", _LANE1_PATH)
assert _SPEC and _SPEC.loader
_lane1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_lane1)

GateError = _lane1.GateError
require = _lane1.require
sha256 = _lane1.sha256
namelist_inventory = _lane1.namelist_inventory
namelist_values = _lane1.namelist_values
nc_inventory = _lane1.nc_inventory
parse_logical = _lane1.parse_logical

VALID = {"VERIFIED", "WAIVED", "UNMEASURED"}
# NEMO's own enumeration of the non-state metadata variables it writes into every
# restart/mesh file (src/OCE/IOM/iom.F90:365-375, meta(1:11)); `numcat` is the SI3
# category axis defined at src/OCE/IOM/iom_nf90.F90:139.  Copied verbatim so the
# waiver set is sourced, not invented.
META = {
    "nav_lat",
    "nav_lon",
    "nav_lev",
    "time_instant",
    "time_instant_bounds",
    "time_counter",
    "time_counter_bounds",
    "x",
    "y",
    "numcat",
    "nav_hgt",
}
FORMAT = "nemo-si3-l3-coverage-v1"
MAGIC = "NEMO_L3_ICE_1"

# Each field is written in this exact order by the committed icestp.F90:154-170.
# The dump precedes store_fields at upstream icestp.F90:151, so ordinary state
# is current-entry and already-carried diagnostic fields retain the preceding
# completed step's value.  The second citation names the defining/restart site.
FRAME_REGISTRY = (
    ("v_i", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:143"),
    ("v_s", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:144"),
    ("a_i", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:145"),
    ("t_su", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:146"),
    ("oa_i", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:149"),
    ("a_ip", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:150"),
    ("v_ip", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:151"),
    ("v_il", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:152"),
    ("sv_i", ("jpi", "jpj", "jpl"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:168"),
    ("u_ice", ("jpi", "jpj"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:147"),
    ("v_ice", ("jpi", "jpj"), "STEP_ENTRY_CURRENT", "icestp.F90:154-171; icerst.F90:148"),
    (
        "stress1_i",
        ("jpi", "jpj"),
        "CARRIED_PREVIOUS_STEP",
        "icestp.F90:154-171; icedyn_rhg_evp.F90:1110",
    ),
    (
        "stress2_i",
        ("jpi", "jpj"),
        "CARRIED_PREVIOUS_STEP",
        "icestp.F90:154-171; icedyn_rhg_evp.F90:1111",
    ),
    (
        "stress12_i",
        ("jpi", "jpj"),
        "CARRIED_PREVIOUS_STEP",
        "icestp.F90:154-171; icedyn_rhg_evp.F90:1112",
    ),
    (
        "snwice_mass",
        ("jpi", "jpj"),
        "CARRIED_PREVIOUS_STEP",
        "icestp.F90:154-171; iceupdate.F90:479",
    ),
    (
        "snwice_mass_b",
        ("jpi", "jpj"),
        "CARRIED_PREVIOUS_BEFORE_LEVEL",
        "icestp.F90:154-171; iceupdate.F90:480",
    ),
    (
        "e_s",
        ("jpi", "jpj", "nlay_s", "jpl"),
        "STEP_ENTRY_CURRENT",
        "icestp.F90:154-171; icerst.F90:153-159",
    ),
    (
        "e_i",
        ("jpi", "jpj", "nlay_i", "jpl"),
        "STEP_ENTRY_CURRENT",
        "icestp.F90:154-171; icerst.F90:160-166",
    ),
    (
        "szv_i",
        ("jpi", "jpj", "nlay_i", "jpl"),
        "STEP_ENTRY_CURRENT",
        "icestp.F90:154-171; icerst.F90:170-175",
    ),
)

# `nz` is the RESOLVED nav_lev NEMO writes, not the case's jpkglo: ICE_ADV1D asks
# for kpk=1 (tests/ICE_ADV1D/MY_SRC/usrdef_nam.F90:76) and NEMO raises it via
# `jpk = MAX( 2, jpkglo )` (src/OCE/LBC/mppini.F90:80,375), which the run's own
# ocean.output confirms ("jpk : 2   jpkglo : 1").
RUNGS = {
    "3.1": {"steps": 40, "nx": 59, "ny": 59, "nz": 2, "dx": 4.0, "dt": 2.0},
    "3.2": {"steps": 485, "nx": 99, "ny": 99, "nz": 2, "dx": 3000.0, "dt": 1200.0},
    "3.3": {"steps": 485, "nx": 99, "ny": 99, "nz": 2, "dx": 3000.0, "dt": 1200.0},
}

TARGET_ICE_NML = {
    "nampar.jpl",
    "nampar.nlay_i",
    "nampar.nlay_s",
    "nampar.ln_icedyn",
    "nampar.ln_icethd",
    "namdyn.ln_dynall",
    "namdyn.ln_dynrhgadv",
    "namdyn.ln_dynadv1d",
    "namdyn.ln_dynadv2d",
    "namdyn_adv.ln_adv_pra",
    "namdyn_adv.ln_adv_umx",
    "namdia.ln_icediachk",
}

TARGET_RHG_NML = {
    "namdyn.rn_ishlat",
    "namdyn.ln_landfast_l16",
    "namdyn_rdgrft.ln_str_h79",
    "namdyn_rdgrft.rn_pstar",
    "namdyn_rdgrft.rn_crhg",
    "namdyn_rdgrft.ln_str_smooth",
    "namdyn_rhg.ln_rhg_evp",
    "namdyn_rhg.ln_rhg_eap",
    "namdyn_rhg.ln_aevp",
    "namdyn_rhg.rn_creepl",
    "namdyn_rhg.rn_ecc",
    "namdyn_rhg.nn_nevp",
    "namdyn_rhg.rn_relast",
    "namdyn_rhg.nn_rhg_chkcvg",
}

TARGET_DYN_NML = {
    "namrun.cn_exp",
    "namrun.nn_itend",
    "namrun.nn_stock",
    "namdom.rn_dt",
    "namdom.ln_meshmask",
    "namsbc.nn_fsbc",
    "namsbc.nn_ice",
    "namsbc_sas.l_sasread",
}


REPO_ROOT = Path(__file__).resolve().parents[4]


def git_sha() -> str:
    """Repo commit that produced this artifact (nemo_testcase_full_statistics.py:118-121)."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


def find_one(root: Path, patterns: tuple[str, ...]) -> Path:
    hits: list[Path] = []
    for pattern in patterns:
        hits.extend(root.glob(pattern))
    hits = sorted(set(hits))
    require(len(hits) == 1, f"expected one of {patterns}, found {len(hits)}")
    return hits[0]


def run_files(root: Path) -> dict[str, Path]:
    files = {
        "mesh": find_one(root, ("mesh_mask*.nc",)),
        "restart": find_one(root, ("*restart_ice*.nc", "*restart_ice_*.nc")),
        "namelist_dyn": root / "output.namelist.dyn",
        "namelist_ice": root / "output.namelist.ice",
        "ocean_output": root / "ocean.output",
    }
    for name, path in files.items():
        require(path.is_file(), f"{name} file is absent: {path}")
    return files


def _inventory_files(files: dict[str, Path]) -> dict[str, Path]:
    return {name: path for name, path in files.items() if name != "ocean_output"}


def inventory(root: Path) -> tuple[dict[str, Path], dict[str, set[str]]]:
    files = run_files(root)
    return files, {
        "mesh": nc_inventory(files["mesh"]),
        "restart": nc_inventory(files["restart"]),
        "namelist_dyn": namelist_inventory(files["namelist_dyn"]),
        "namelist_ice": namelist_inventory(files["namelist_ice"]),
    }


def _contract_item(status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}


def restart_contract(rung: str, nlay_i: int, nlay_s: int, nn_icesal: int, ponds: bool) -> dict:
    rhg = rung == "3.3"
    result: dict[str, dict[str, str]] = {}

    def add(name: str, active: bool, reason: str) -> None:
        result[name] = _contract_item("VERIFIED" if active else "WAIVED", reason)

    for name in (
        "nn_fsbc",
        "kt_ice",
        "v_i",
        "v_s",
        "a_i",
        "t_su",
        "u_ice",
        "v_ice",
        "oa_i",
        "a_ip",
        "v_ip",
        "v_il",
        "sv_i",
        "snwice_mass",
        "snwice_mass_b",
    ):
        add(name, True, "Appendix A active restart prognostic; loaded from final SI3 restart")
    for level in range(1, nlay_s + 1):
        add(f"e_s_l{level:02d}", True, "Appendix A snow-enthalpy layer")
        add(
            f"t_s_l{level:02d}",
            False,
            "WAIVED-NOT-RESTARTED: Appendix A t_s(l) is reconstructed; "
            "icerst.F90:153-159 writes e_s(l)",
        )
    for level in range(1, nlay_i + 1):
        add(f"e_i_l{level:02d}", True, "Appendix A ice-enthalpy layer")
        add(
            f"szv_i_l{level:02d}",
            True,
            "Appendix A layer salt-content field; file-loaded even when inert",
        )
    for name in ("cnd_ice", "t1_ice"):
        add(name, False, "WAIVED-INACTIVE: ln_cpl=F; icerst.F90:178-181")
    for name in ("stress1_i", "stress2_i", "stress12_i"):
        add(
            name,
            rhg,
            "Appendix A EVP stress"
            if rhg
            else "WAIVED-INACTIVE: advection-only rung never calls ice_dyn_rhg",
        )

    for stem in ("ice", "sn", "a", "age"):
        for prefix in ("sx", "sy", "sxx", "syy", "sxy"):
            add(f"{prefix}{stem}", True, "Appendix A mandatory Prather moment")
    for stem, layers in (("c0", nlay_s), ("e", nlay_i)):
        for level in range(1, layers + 1):
            for prefix in ("sx", "sy", "sxx", "syy", "sxy"):
                add(f"{prefix}{stem}_l{level:02d}", True, "Appendix A layered Prather moment")
    for prefix in ("sx", "sy", "sxx", "syy", "sxy"):
        add(
            f"{prefix}sal",
            nn_icesal != 4,
            "Appendix A bulk-salinity Prather moment"
            if nn_icesal != 4
            else "WAIVED-INACTIVE: nn_icesal=4 selects layer moments",
        )
    for level in range(1, nlay_i + 1):
        for prefix in ("sx", "sy", "sxx", "syy", "sxy"):
            add(
                f"{prefix}si_l{level:02d}",
                nn_icesal == 4,
                "Appendix A layer-salinity Prather moment"
                if nn_icesal == 4
                else "WAIVED-INACTIVE: nn_icesal/=4 selects bulk moments",
            )
    for stem in ("ap", "vp", "vl"):
        for prefix in ("sx", "sy", "sxx", "syy", "sxy"):
            add(
                f"{prefix}{stem}",
                ponds,
                "Appendix A active pond Prather moment"
                if ponds
                else "WAIVED-INACTIVE: both pond schemes are off",
            )
    return result


def _resolved_ice_settings(root: Path) -> tuple[int, int, int, bool]:
    values = namelist_values(run_files(root)["namelist_ice"])
    needed = ("nampar.nlay_i", "nampar.nlay_s", "namthd_sal.nn_icesal", "namthd_pnd.ln_pnd")
    for key in needed:
        require(key in values, f"missing resolved selector {key}")
    return (
        int(values["nampar.nlay_i"]),
        int(values["nampar.nlay_s"]),
        int(values["namthd_sal.nn_icesal"]),
        parse_logical(values["namthd_pnd.ln_pnd"], "namthd_pnd.ln_pnd"),
    )


def disposition_template(root: Path, rung: str) -> dict:
    files, actual = inventory(root)
    entries: dict[str, dict[str, dict[str, str]]] = {}
    for namespace, names in actual.items():
        entries[namespace] = {}
        for name in sorted(names):
            key = name.lower()
            if namespace == "mesh":
                waived = key in META
                reason = (
                    "coordinate/time metadata; explicitly waived"
                    if waived
                    else "file-loaded, finite, and covered by geometry checks"
                )
            elif namespace == "restart":
                waived = key in META
                if waived:
                    reason = "coordinate/time metadata; explicitly waived"
                elif name.startswith("DELAY__"):
                    # iom's delayed-global-reduction buffer, not model state: the
                    # ice-CFL mpp_max carries cdelay='cflice'
                    # (icedyn_adv_pra.F90:122, icedyn_adv_umx.F90:185) and iom
                    # writes it at icerst.F90:140 / reads it at iom.F90:1953.
                    reason = (
                        "iom delayed-reduction restart buffer (icerst.F90:140, "
                        "iom.F90:1939-1955), not an Appendix-A prognostic; file-loaded and finite"
                    )
                else:
                    reason = "file-loaded final SI3 restart state in fp64"
            else:
                targets = (
                    TARGET_DYN_NML
                    if namespace == "namelist_dyn"
                    else TARGET_ICE_NML | (TARGET_RHG_NML if rung == "3.3" else set())
                )
                waived = key not in targets
                reason = (
                    "resolved NEMO default outside the rung claim; full file hash pinned"
                    if waived
                    else (
                        "resolved rung selector verified against the shipped case plus "
                        "declared overlay"
                    )
                )
            entries[namespace][name] = _contract_item("WAIVED" if waived else "VERIFIED", reason)
    nlay_i, nlay_s, nn_icesal, ponds = _resolved_ice_settings(root)
    return {
        "format": FORMAT,
        "rung": rung,
        "files": {name: sha256(path) for name, path in _inventory_files(files).items()},
        "entries": entries,
        "restart_contract": restart_contract(rung, nlay_i, nlay_s, nn_icesal, ponds),
    }


def check_manifest(root: Path, rung: str, manifest: dict, plant_unaccounted: bool = False) -> dict:
    files, actual = inventory(root)
    if plant_unaccounted:
        actual["mesh"].add("PLANTED_UNACCOUNTED_FILE_ARRAY")
    require(manifest.get("format") == FORMAT, "bad manifest format")
    require(manifest.get("rung") == rung, "manifest rung mismatch")
    for namespace, names in actual.items():
        ledger = manifest.get("entries", {}).get(namespace, {})
        require(
            set(ledger) == names,
            f"{namespace} coverage mismatch: missing={sorted(names - set(ledger))}, "
            f"extra={sorted(set(ledger) - names)}",
        )
        require(
            manifest.get("files", {}).get(namespace) == sha256(files[namespace]),
            f"{namespace} SHA256 mismatch",
        )
        for name, item in ledger.items():
            require(item.get("status") in VALID, f"{namespace}.{name}: bad disposition")
            if namespace in {"mesh", "restart"}:
                require(
                    item.get("status") in {"VERIFIED", "WAIVED"},
                    f"{namespace}.{name}: UNMEASURED is forbidden",
                )
            require(bool(item.get("reason", "").strip()), f"{namespace}.{name}: empty reason")
            if item["status"] != "VERIFIED" or namespace.startswith("namelist"):
                continue
            with netCDF4.Dataset(files[namespace]) as ds:
                data = np.asarray(ds.variables[name][:])
            require(np.issubdtype(data.dtype, np.number), f"{namespace}.{name} is not numeric")
            require(np.all(np.isfinite(data)), f"{namespace}.{name} contains non-finite values")
            if namespace == "restart" and np.issubdtype(data.dtype, np.floating):
                require(data.dtype == np.float64, f"restart.{name} is not fp64")

    restart_names = actual["restart"]
    restart_ledger = manifest["entries"]["restart"]
    nlay_i, nlay_s, nn_icesal, ponds = _resolved_ice_settings(root)
    expected_contract = restart_contract(rung, nlay_i, nlay_s, nn_icesal, ponds)
    contract = manifest.get("restart_contract", {})
    require(
        contract == expected_contract,
        "restart contract differs from exhaustive Appendix-A contract",
    )
    for name, item in contract.items():
        require(item.get("status") in VALID, f"restart contract {name}: bad disposition")
        require(bool(item.get("reason", "").strip()), f"restart contract {name}: empty reason")
        if item["status"] == "VERIFIED":
            require(name in restart_names, f"restart contract {name}: active field absent")
            require(
                restart_ledger[name]["status"] == "VERIFIED",
                f"restart contract {name}: not VERIFIED-loaded",
            )
    return {name: len(values) for name, values in actual.items()}


def _number(values: dict[str, str], key: str, expected: float) -> None:
    require(key in values, f"missing resolved selector {key}")
    got = float(values[key].replace("D", "E").replace("d", "e"))
    require(got == expected, f"resolved selector {key}={values[key]}")


def _logical(values: dict[str, str], key: str, expected: bool) -> None:
    require(key in values, f"missing resolved selector {key}")
    require(parse_logical(values[key], key) is expected, f"resolved selector {key}={values[key]}")


def resolved_selectors(root: Path, rung: str) -> dict:
    spec = RUNGS[rung]
    dyn = namelist_values(run_files(root)["namelist_dyn"])
    ice = namelist_values(run_files(root)["namelist_ice"])
    _number(dyn, "namrun.nn_itend", spec["steps"])
    _number(dyn, "namrun.nn_stock", spec["steps"])
    _number(dyn, "namdom.rn_dt", spec["dt"])
    _number(dyn, "namsbc.nn_fsbc", 1)
    _number(dyn, "namsbc.nn_ice", 2)
    _logical(dyn, "namdom.ln_meshmask", True)
    _logical(dyn, "namsbc_sas.l_sasread", False)
    _number(ice, "nampar.nlay_i", 3)
    _number(ice, "nampar.nlay_s", 3)
    _number(ice, "nampar.jpl", 1)
    _logical(ice, "nampar.ln_icedyn", True)
    _logical(ice, "nampar.ln_icethd", False)
    _logical(ice, "namdyn_adv.ln_adv_pra", True)
    _logical(ice, "namdyn_adv.ln_adv_umx", False)
    _logical(ice, "namdia.ln_icediachk", True)
    for key in (
        "namdyn.ln_dynall",
        "namdyn.ln_dynrhgadv",
        "namdyn.ln_dynadv1d",
        "namdyn.ln_dynadv2d",
    ):
        expected = {
            "3.1": "namdyn.ln_dynadv1d",
            "3.2": "namdyn.ln_dynadv2d",
            "3.3": "namdyn.ln_dynrhgadv",
        }[rung]
        _logical(ice, key, key == expected)
    if rung == "3.3":
        for key, expected in (
            ("namdyn.rn_ishlat", 2),
            ("namdyn_rdgrft.rn_pstar", 2.0e4),
            ("namdyn_rdgrft.rn_crhg", 20),
            ("namdyn_rhg.rn_creepl", 2.0e-9),
            ("namdyn_rhg.rn_ecc", 2),
            ("namdyn_rhg.nn_nevp", 100),
            ("namdyn_rhg.rn_relast", 0.333),
            ("namdyn_rhg.nn_rhg_chkcvg", 0),
        ):
            _number(ice, key, expected)
        for key, expected in (
            ("namdyn.ln_landfast_l16", False),
            ("namdyn_rdgrft.ln_str_h79", True),
            ("namdyn_rdgrft.ln_str_smooth", False),
            ("namdyn_rhg.ln_rhg_evp", True),
            ("namdyn_rhg.ln_rhg_eap", False),
            ("namdyn_rhg.ln_aevp", True),
        ):
            _logical(ice, key, expected)
    expected_exp = {
        "3.1": "ICE_ADV1D_OMIP_L3",
        "3.2": "ICE_ADV2D_OMIP_L3",
        "3.3": "ICE_ADV2D_RHG_OMIP_L3",
    }[rung]
    # NEMO writes cn_exp back as a quoted, blank-padded CHARACTER(lc) field.
    resolved_exp = dyn.get("namrun.cn_exp", "").strip().strip("'\"").strip()
    require(resolved_exp == expected_exp, f"resolved cn_exp={resolved_exp!r}")
    nlay_i, nlay_s, nn_icesal, ponds = _resolved_ice_settings(root)
    return {
        "jpl": 1,
        "nlay_i": nlay_i,
        "nlay_s": nlay_s,
        "nn_icesal": nn_icesal,
        "ln_pnd": ponds,
    }


def cpp_keys(case_dir: Path) -> dict:
    path = case_dir / "BLD/cpp.fcm"
    require(path.is_file(), f"compiled cpp key file absent: {path}")
    line = next((raw for raw in path.read_text().splitlines() if "fppkeys" in raw), "")
    keys = line.split("fppkeys", 1)[-1].split()
    require(set(keys) == {"key_si3", "key_linssh", "key_vco_1d"}, f"resolved cpp keys={keys}")
    return {"keys": keys, "sha256": sha256(path)}


def _array(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    require(name in ds.variables, f"missing mesh array {name}")
    return np.asarray(ds.variables[name][:])


def geometry(root: Path, rung: str, plant_field: bool = False) -> dict:
    spec = RUNGS[rung]
    mesh = run_files(root)["mesh"]
    with netCDF4.Dataset(mesh) as ds:
        require(len(ds.dimensions["x"]) == spec["nx"], "mesh x dimension")
        require(len(ds.dimensions["y"]) == spec["ny"], "mesh y dimension")
        require(len(ds.dimensions["nav_lev"]) == spec["nz"], "mesh vertical dimension")
        for name in ("e1t", "e1u", "e1v", "e1f", "e2t", "e2u", "e2v", "e2f"):
            data = _array(ds, name)
            if plant_field and name == "e1t":
                data = data.copy()
                data.flat[0] += 1.0
            require(data.dtype == np.float64, f"{name} is not fp64")
            require(np.array_equal(data, np.full_like(data, spec["dx"])), f"{name} metric")
        for name in ("ff_t", "ff_f"):
            require(
                np.array_equal(_array(ds, name), np.zeros_like(_array(ds, name))),
                f"{name} must be zero",
            )
        for name in ("tmask", "umask", "vmask"):
            require(set(np.unique(_array(ds, name))).issubset({0, 1}), f"{name} is not binary")
    return {
        "mesh_sha256": sha256(mesh),
        "nx": spec["nx"],
        "ny": spec["ny"],
        "nz": spec["nz"],
        "dx_m": spec["dx"],
    }


def read_frame(path: Path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        raw = fh.read(9 * 4)
        require(len(raw) == 9 * 4, f"truncated frame header: {path}")
        version, kt, jpi, jpj, jpl, nlay_i, nlay_s, storage, nfields = struct.unpack("=9i", raw)
        require(magic == MAGIC, f"bad frame magic: {magic!r}")
        require(version == 1 and storage == 64, f"bad frame version/storage: {version}/{storage}")
        require(nfields == len(FRAME_REGISTRY), f"frame registry count {nfields}")
        dims = {"jpi": jpi, "jpj": jpj, "jpl": jpl, "nlay_i": nlay_i, "nlay_s": nlay_s}
        arrays: dict[str, np.ndarray] = {}
        for name, axes, _, _ in FRAME_REGISTRY:
            shape = tuple(dims[axis] for axis in axes)
            values = np.fromfile(fh, dtype=np.float64, count=int(np.prod(shape)))
            require(values.size == int(np.prod(shape)), f"truncated frame field {name}: {path}")
            arrays[name] = values.reshape(shape, order="F")
        require(fh.read(1) == b"", f"unregistered trailing frame bytes: {path}")
    require(
        all(np.all(np.isfinite(value)) for value in arrays.values()), f"non-finite frame: {path}"
    )
    return {"kt": kt, **dims, "storage_bits": storage}, arrays


def _interior(array: np.ndarray) -> np.ndarray:
    require(
        array.shape[0] >= 5 and array.shape[1] >= 5, f"frame lacks two-cell halos: {array.shape}"
    )
    return array[2:-2, 2:-2, ...]


def _relative_drift(initial: np.ndarray, final: np.ndarray) -> float:
    scale = max(abs(float(np.sum(initial, dtype=np.float64))), 1.0)
    return abs(float(np.sum(final, dtype=np.float64) - np.sum(initial, dtype=np.float64))) / scale


def _restart_array(path: Path, name: str) -> np.ndarray:
    with netCDF4.Dataset(path) as ds:
        require(name in ds.variables, f"final restart lacks phenomenology field {name}")
        variable = ds.variables[name]
        selection: list[int | slice] = []
        kept_dims: list[str] = []
        for dim in variable.dimensions:
            if "time" in dim.lower():
                selection.append(0)
            else:
                selection.append(slice(None))
                kept_dims.append(dim)
        data = np.asarray(variable[tuple(selection)])
    x_axis = next((i for i, dim in enumerate(kept_dims) if dim.lower() == "x"), None)
    y_axis = next((i for i, dim in enumerate(kept_dims) if dim.lower() == "y"), None)
    require(x_axis is not None and y_axis is not None, f"{name}: cannot identify x/y dimensions")
    order = [x_axis, y_axis] + [i for i in range(data.ndim) if i not in {x_axis, y_axis}]
    return np.transpose(data, order)


def wet_window(root: Path) -> tuple[slice, slice]:
    """Surface-tmask bounding box, in the frames' (x, y) order.

    ICE_ADV1D closes its basin with a one-cell land rim (`mesh_mask.tmask` rows
    and columns 0 and 58 are 0), and `dommsk.F90` zeroes every ice field there,
    so a whole-array reduction reports the land rim rather than the documented
    ice-domain behaviour.  ICE_ADV2D is doubly periodic and fully wet, for which
    this window is the whole array.
    """
    with netCDF4.Dataset(run_files(root)["mesh"]) as ds:
        tmask = np.asarray(_array(ds, "tmask"))
    while tmask.ndim > 2:
        tmask = tmask[0]
    wet = tmask.T.astype(bool)  # (y, x) file order -> (x, y) frame order
    require(bool(wet.any()), "mesh tmask has no wet cell")
    xs = np.flatnonzero(wet.any(axis=1))
    ys = np.flatnonzero(wet.any(axis=0))
    window = (slice(int(xs[0]), int(xs[-1]) + 1), slice(int(ys[0]), int(ys[-1]) + 1))
    require(
        bool(wet[window].all()),
        "surface wet domain is not a filled rectangle; the phenomenology window "
        "would mix land and ocean cells",
    )
    return window


def conservation_diagnostics(root: Path) -> dict:
    """Preregistered predicate: any printed SI3 `: violation` line is REFUTE.

    The verdict is returned rather than raised so the receipt still carries the
    coverage and trajectory numbers; `main` exits nonzero on anything but
    CONFIRM, so the gate stays fail-closed.
    """
    files = run_files(root)
    contents = files["ocean_output"].read_text(errors="replace")
    hits = re.findall(r"^.*:\s*violation\s+.*$", contents, flags=re.IGNORECASE | re.MULTILINE)
    return {
        "status": "CONFIRM" if not hits else "REFUTE",
        "criterion": "zero SI3 native-threshold conservation violations",
        "source": "icectl.F90:67-78,166-190,232-236; namelist_ice_ref:318-319",
        "violation_count": len(hits),
        "violations": [hit.strip() for hit in hits],
    }


def phenomenology(
    rung: str,
    first: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
    window: tuple[slice, slice] = (slice(None), slice(None)),
) -> dict:
    a0, a1 = _interior(first["a_i"])[window], final["a_i"][window]
    v0, v1 = _interior(first["v_i"])[window], final["v_i"][window]
    refuted: list[str] = []

    def predicate(holds: bool, claim: str) -> None:
        """Record a preregistered CONFIRM/REFUTE predicate without aborting.

        The verdict is reported instead of raised so a refuted rung still emits
        its measured numbers; `main` exits nonzero on any REFUTE, so the gate
        stays fail-closed.
        """
        if not holds:
            refuted.append(claim)

    report: dict[str, float | str | list[str]] = {
        "status": "CONFIRM",
        "sample": "step-entry kt=1 versus completed-run final ice restart",
        "domain": "surface-tmask wet window (mesh_mask.tmask bounding box)",
        "concentration_relative_drift_unclassified": _relative_drift(a0, a1),
        "ice_volume_relative_drift_unclassified": _relative_drift(v0, v1),
    }
    if rung == "3.1":
        h0 = np.divide(v0, a0, out=np.zeros_like(v0), where=a0 != 0.0)
        h1 = np.divide(v1, a1, out=np.zeros_like(v1), where=a1 != 0.0)
        spreads = {
            name: float(np.max(np.ptp(array, axis=1)))
            for name, array in (("a_i", a1), ("v_i", v1), ("h_i", h1))
        }
        predicate(
            all(value == 0.0 for value in spreads.values()), f"y-homogeneity refuted: {spreads}"
        )
        x = np.arange(a0.shape[0], dtype=np.float64)
        center = 0.5 * (a0.shape[0] - 1)
        centroids: dict[str, tuple[float, float]] = {}
        for name, initial, completed in (("a_i", a0, a1), ("v_i", v0, v1), ("h_i", h0, h1)):
            weights0 = np.sum(initial, axis=(1, 2), dtype=np.float64)
            weights1 = np.sum(completed, axis=(1, 2), dtype=np.float64)
            c0 = float(np.sum(weights0 * x) / np.sum(weights0))
            c1 = float(np.sum(weights1 * x) / np.sum(weights1))
            # h_i is excluded from the strict-decrease predicate because its
            # INITIAL distance from the basin centre is exactly 0: the shipped IC
            # lays a symmetric thickness notch (hti=0.2 on x in [15,43] of the
            # 59-cell basin, make_initice.py:97-103) on a uniform hti=1 field, so
            # its weighted centroid starts on the centre and nothing can strictly
            # decrease from zero.  The predicate is ill-posed, not refuted; the
            # measured baseline is emitted so the claim stays checkable.
            if name != "h_i":
                predicate(
                    abs(c1 - center) < abs(c0 - center),
                    f"{name} did not converge toward centre",
                )
            else:
                require(
                    abs(c0 - center) == 0.0,
                    "h_i convergence is only voided while its baseline distance is "
                    f"exactly zero; measured {abs(c0 - center)!r}",
                )
            predicate(float(np.ptp(completed)) > 0.0, f"{name} initial shape collapsed")
            centroids[name] = (c0, c1)
        report.update(
            basin_centre_index=center,
            h_i_centroid_convergence="VOID-ILL-POSED: baseline distance is exactly 0",
            y_homogeneity_max_abs=max(spreads.values()),
            initial_a_centroid_index=centroids["a_i"][0],
            final_a_centroid_index=centroids["a_i"][1],
            initial_v_centroid_index=centroids["v_i"][0],
            final_v_centroid_index=centroids["v_i"][1],
            initial_h_centroid_index=centroids["h_i"][0],
            final_h_centroid_index=centroids["h_i"][1],
        )
    elif rung == "3.2":
        h0 = np.divide(v0, a0, out=np.zeros_like(v0), where=a0 != 0.0)
        h1 = np.divide(v1, a1, out=np.zeros_like(v1), where=a1 != 0.0)
        overshoot = float(np.max(h1) - np.max(h0))
        predicate(
            float(np.max(a1)) == float(np.max(a0)),
            "documented Prather maximum-concentration preservation refuted",
        )
        predicate(overshoot > 0.0, "documented Prather ice-thickness overshoot not detected")
        report.update(
            initial_a_i_max=float(np.max(a0)),
            final_a_i_max=float(np.max(a1)),
            initial_h_i_max=float(np.max(h0)),
            final_h_i_max=float(np.max(h1)),
            h_i_overshoot=overshoot,
            upper_side_lobe_cell_count=int(np.count_nonzero(h1 > np.max(h0))),
        )
    else:
        response = float(np.max(final["u_ice"][window]))
        state_change = float(np.max(np.abs(v1 - v0)))
        predicate(response > 0.0, "constant positive x-stress produced no positive ice velocity")
        predicate(state_change > 0.0, "rheology+advection state did not change")
        report.update(final_u_ice_max=response, ice_volume_max_abs_change=state_change)
    report["refuted_predicates"] = refuted
    if refuted:
        report["status"] = "REFUTE"
    return report


def trajectory(root: Path, rung: str) -> dict:
    expected = RUNGS[rung]["steps"]
    paths = sorted(root.glob("oracle_ice_step_entry_kt*.bin"))
    require(len(paths) == expected, f"expected {expected} ice-entry frames, found {len(paths)}")
    first_arrays = final_arrays = None
    frame_hashes: list[str] = []
    header0: dict[str, int] | None = None
    for step, path in enumerate(paths, start=1):
        header, arrays = read_frame(path)
        require(header["kt"] == step, f"frame sequence mismatch at {path.name}: kt={header['kt']}")
        require(header["jpi"] == RUNGS[rung]["nx"] + 4, f"local jpi={header['jpi']}")
        require(header["jpj"] == RUNGS[rung]["ny"] + 4, f"local jpj={header['jpj']}")
        require(header["nlay_i"] == 3 and header["nlay_s"] == 3, "frame ice-layer count")
        if header0 is None:
            header0 = header
            first_arrays = arrays
        else:
            require(
                {k: header[k] for k in header if k != "kt"}
                == {k: header0[k] for k in header0 if k != "kt"},
                "frame shape changed",
            )
        final_arrays = arrays
        frame_hashes.append(sha256(path))
    assert header0 is not None and first_arrays is not None and final_arrays is not None
    aggregate = hashlib.sha256("\n".join(frame_hashes).encode()).hexdigest()
    return {
        "frame_count": len(paths),
        "first_sha256": frame_hashes[0],
        "last_sha256": frame_hashes[-1],
        "ordered_frame_hash_aggregate": aggregate,
        "header": header0,
        "registry": [
            {"name": name, "shape": list(shape), "time_level": level, "source": source}
            for name, shape, level, source in FRAME_REGISTRY
        ],
        "phenomenology": phenomenology(
            rung,
            first_arrays,
            {
                name: _restart_array(run_files(root)["restart"], name)
                for name in ("a_i", "v_i", "u_ice")
            },
            wet_window(root),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--rung", choices=tuple(RUNGS), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--emit-manifest", action="store_true")
    parser.add_argument("--plant-unaccounted", action="store_true")
    parser.add_argument("--plant-field", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.emit_manifest:
        manifest = disposition_template(root, args.rung)
        manifest["git_sha"] = git_sha()
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    require(args.manifest is not None, "--manifest is required")
    manifest = json.loads(args.manifest.read_text())
    counts = check_manifest(root, args.rung, manifest, plant_unaccounted=args.plant_unaccounted)
    selectors = resolved_selectors(root, args.rung)
    keys = cpp_keys(args.case_dir.resolve())
    geom = geometry(root, args.rung, plant_field=args.plant_field)
    traj = trajectory(root, args.rung)
    conservation = conservation_diagnostics(root)
    print(
        json.dumps(
            {
                "status": "VERIFIED"
                if conservation["status"] == traj["phenomenology"]["status"] == "CONFIRM"
                else "DEBT",
                "scope": "NEMO_ORACLE_ONLY",
                "git_sha": git_sha(),
                "rung": args.rung,
                "inventory_counts": counts,
                "resolved_inherited_settings": selectors,
                "cpp": keys,
                "geometry": geom,
                "trajectory": traj,
                "conservation_diagnostics": conservation,
                "unmeasured": [
                    "legoesm_alignment",
                    "post-dynamics_stage_arrays",
                    "landfast_L16_deferred_lane4",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    verdicts = (conservation["status"], traj["phenomenology"]["status"])
    return 0 if set(verdicts) == {"CONFIRM"} else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"DEBT: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
