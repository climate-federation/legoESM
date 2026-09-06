#!/usr/bin/env python3
"""Fail-closed geometry, coverage, and trajectory gate for NEMO testcases.

Phase 1 deliberately does not compare against legoESM.  The manifest is an
exhaustive ledger: each discovered mesh/restart/namelist item must occur once
with a VERIFIED, WAIVED, or loud UNMEASURED disposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import netCDF4
import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

VALID = {"VERIFIED", "WAIVED", "UNMEASURED"}
GROSS_TRACER_EXCESS_RELATIVE = 1.0e-6
BASE_UNMEASURED = [
    "teos10_density",
    "rab",
    "bn2",
    "adaptive_vertical_advection_partition",
    "bbl_transport",
    "bbl_downslope_mask_geometry",
    "global_tracer_inventory_closure",
]
META = {"nav_lon", "nav_lat", "nav_lev", "time_counter"}
TARGET_NML = {
    "namrun.cn_exp",
    "namrun.nn_itend",
    "namrun.nn_stock",
    "namdom.rn_dt",
    "namdom.ln_meshmask",
    "nameos.ln_teos10",
    "nameos.ln_eos80",
    "nameos.ln_seos",
    "namtra_adv.ln_traadv_fct",
    "namtra_adv.nn_fct_h",
    "namtra_adv.nn_fct_v",
    "namtra_adv.nn_fct_imp",
    "namzdf.ln_zad_aimp",
    "nambbl.ln_trabbl",
    "nambbl.nn_bbl_ldf",
    "nambbl.nn_bbl_adv",
    "nambbl.rn_ahtbbl",
    "nambbl.rn_gambbl",
    "namusr_def.nn_coord",
}
GYRE_TARGET_NML = {
    "namusr_def.nn_gyre",
    "namdyn_hpg.ln_hpg_zco",
    "namdyn_hpg.ln_hpg_sco",
    "namdyn_adv.ln_dynadv_vec",
    "namdyn_adv.ln_dynadv_up3",
    "namdyn_vor.ln_dynvor_ene",
    "namdyn_spg.nn_bt_flt",
    "namdyn_spg.rn_bt_alpha",
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def parse_logical(value: str, key: str = "logical") -> bool:
    """Parse a Fortran logical without silently coercing malformed input."""
    token = value.strip().replace(".", "").upper()
    if token in {"T", "TRUE"}:
        return True
    if token in {"F", "FALSE"}:
        return False
    raise GateError(f"resolved selector {key} has invalid logical {value!r}")


def tracer_bar(excess_relative: float, floor_relative: float) -> str:
    """Classify a measured tracer range against its registered fp64 floor."""
    return "AT-BAR" if excess_relative <= floor_relative else "UNMEASURED"


def measured_trajectory_status(records: list[dict]) -> str:
    """Summarize measured rows independently of the prose-only gap ledger."""
    unresolved = any(
        row[name] == "UNMEASURED" for row in records for name in ("temperature_bar", "salinity_bar")
    )
    return "UNMEASURED" if unresolved else "VERIFIED"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_one(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(pattern))
    require(len(hits) == 1, f"expected one {pattern}, found {len(hits)}")
    return hits[0]


def namelist_inventory(path: Path) -> set[str]:
    section = None
    found: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("&"):
            section = line[1:].split()[0].lower()
            continue
        if line == "/":
            section = None
            continue
        if section and "=" in line:
            lhs = line.split("=", 1)[0].strip().lower()
            if re.fullmatch(r"[a-z][a-z0-9_%]*(?:\([^)]*\))?", lhs):
                found.add(f"{section}.{lhs}")
    return found


def namelist_values(path: Path) -> dict[str, str]:
    section = None
    found: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("&"):
            section = line[1:].split()[0].lower()
        elif line == "/":
            section = None
        elif section and "=" in line:
            lhs, rhs = line.split("=", 1)
            lhs = lhs.strip().lower()
            if re.fullmatch(r"[a-z][a-z0-9_%]*(?:\([^)]*\))?", lhs):
                found[f"{section}.{lhs}"] = rhs.strip().rstrip(",").strip()
    return found


def nc_inventory(path: Path) -> set[str]:
    with netCDF4.Dataset(path) as ds:
        return set(ds.variables)


def expected_inventory(root: Path) -> tuple[Path, dict[str, set[str]]]:
    mesh = find_one(root, "mesh_mask*.nc")
    restarts = sorted(root.glob("*restart*.nc"))
    require(restarts, "final restart is absent")
    restart = restarts[-1]
    resolved = root / "output.namelist.dyn"
    require(resolved.is_file(), "resolved output.namelist.dyn is absent")
    return restart, {
        "mesh": nc_inventory(mesh),
        "restart": nc_inventory(restart),
        "namelist": namelist_inventory(resolved),
    }


def disposition_template(root: Path, case: str | None = None) -> dict:
    restart, inventory = expected_inventory(root)
    target_nml = TARGET_NML | (GYRE_TARGET_NML if case == "gyre" else set())
    entries: dict[str, dict[str, dict[str, str]]] = {}
    for namespace, names in inventory.items():
        entries[namespace] = {}
        for name in sorted(names):
            key = name.lower()
            if namespace == "mesh":
                status = "WAIVED" if name in META else "VERIFIED"
                reason = (
                    "coordinate/time alias; scientific double-precision fields are "
                    "verified independently"
                    if status == "WAIVED"
                    else "finite, typed, dimensioned, and covered by analytic geometry checks"
                )
            elif namespace == "restart":
                status = (
                    "WAIVED"
                    if key in {"nav_lon", "nav_lat", "nav_lev", "time_counter"}
                    else "VERIFIED"
                )
                reason = (
                    "coordinate/time metadata"
                    if status == "WAIVED"
                    else "finite restart state with registered dimensions and time level"
                )
            else:
                status = "VERIFIED" if key in target_nml else "WAIVED"
                reason = (
                    "phase-1 pinned science selector"
                    if status == "VERIFIED"
                    else "resolved NEMO default outside the phase-1 claim; full file hash pinned"
                )
            entries[namespace][name] = {"status": status, "reason": reason}
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l1-coverage-v1",
        "files": {
            "mesh": sha256(find_one(root, "mesh_mask*.nc")),
            "restart": sha256(restart),
            "namelist": sha256(root / "output.namelist.dyn"),
        },
        "entries": entries,
    }


def check_manifest(
    root: Path, manifest: dict, plant_unaccounted: bool = False
) -> dict[str, set[str]]:
    restart, actual = expected_inventory(root)
    if plant_unaccounted:
        # The real threat is a newly written file-side array absent from the
        # reviewed ledger.  Plant in that direction so the control reports it
        # as missing, not as a stale extra manifest entry.
        actual["mesh"].add("PLANTED_UNACCOUNTED_FILE_ARRAY")
    require(manifest.get("format") == "nemo-testcase-l1-coverage-v1", "bad manifest format")
    files = {
        "mesh": find_one(root, "mesh_mask*.nc"),
        "restart": restart,
        "namelist": root / "output.namelist.dyn",
    }
    for namespace, names in actual.items():
        ledger = manifest.get("entries", {}).get(namespace, {})
        require(
            set(ledger) == names,
            f"{namespace} coverage mismatch: "
            f"missing={sorted(names - set(ledger))}, "
            f"extra={sorted(set(ledger) - names)}",
        )
        for name, item in ledger.items():
            require(item.get("status") in VALID, f"{namespace}.{name}: bad disposition")
            require(bool(item.get("reason", "").strip()), f"{namespace}.{name}: empty reason")
        require(
            manifest.get("files", {}).get(namespace) == sha256(files[namespace]),
            f"{namespace} SHA256 mismatch",
        )
    for namespace in ("mesh", "restart"):
        with netCDF4.Dataset(files[namespace]) as ds:
            for name, item in manifest["entries"][namespace].items():
                if item["status"] != "VERIFIED":
                    continue
                data = np.asarray(ds.variables[name][:])
                require(np.issubdtype(data.dtype, np.number), f"{namespace}.{name} is not numeric")
                require(np.all(np.isfinite(data)), f"{namespace}.{name} contains non-finite values")
                if namespace == "restart":
                    require(data.dtype == np.float64, f"restart.{name} is not fp64")
    return actual


def resolved_selectors(root: Path, case: str, coord: str) -> None:
    values = namelist_values(root / "output.namelist.dyn")

    def logical(key: str, expected: bool) -> None:
        require(key in values, f"missing resolved selector {key}")
        got = parse_logical(values[key], key)
        require(got is expected, f"resolved selector {key}={values[key]}")

    def number(key: str, expected: float) -> None:
        require(key in values, f"missing resolved selector {key}")
        require(
            float(values[key].replace("D", "E")) == expected,
            f"resolved selector {key}={values[key]}",
        )

    logical("nameos.ln_teos10", True)
    logical("nameos.ln_eos80", False)
    logical("nameos.ln_seos", False)
    logical("namtra_adv.ln_traadv_fct", True)
    number("namtra_adv.nn_fct_h", 2)
    number("namtra_adv.nn_fct_v", 2)
    number("namtra_adv.nn_fct_imp", 1)
    logical("namzdf.ln_zad_aimp", case != "gyre")
    logical("namdom.ln_meshmask", True)
    if case == "overflow":
        number("namrun.nn_itend", 6120)
        number("namrun.nn_stock", 6120)
        number("namdom.rn_dt", 10)
        number("namusr_def.nn_coord", 1 if coord == "zps" else 2)
        logical("nambbl.ln_trabbl", True)
        number("nambbl.nn_bbl_ldf", 0)
        number("nambbl.nn_bbl_adv", 2)
        number("nambbl.rn_ahtbbl", 1000)
        number("nambbl.rn_gambbl", 20)
    elif case == "lock_exchange":
        number("namrun.nn_itend", 61200)
        number("namrun.nn_stock", 61200)
        number("namdom.rn_dt", 1)
        logical("nambbl.ln_trabbl", False)
    else:
        number("namrun.nn_itend", 4320)
        number("namrun.nn_stock", 4320)
        number("namdom.rn_dt", 14400)
        number("namusr_def.nn_gyre", 1)
        logical("namdyn_hpg.ln_hpg_zco", False)
        logical("namdyn_hpg.ln_hpg_sco", True)
        logical("namdyn_adv.ln_dynadv_vec", True)
        logical("namdyn_adv.ln_dynadv_up3", False)
        logical("namdyn_vor.ln_dynvor_ene", True)
        number("namdyn_spg.nn_bt_flt", 3)
        number("namdyn_spg.rn_bt_alpha", 0.07)


def build_run_provenance(root: Path) -> dict[str, str | int | float]:
    """Fail closed on the registered build keys and normal run completion."""
    provenance = root / "provenance"
    cpp = provenance / "cpp_GYRE_OMIP_L2.fcm"
    cpp_history = provenance / "cpp.history"
    instrument = provenance / "stprk3.F90"
    executable = root / "nemo.exe"
    stdout = root / "ocean.output"
    time_step = root / "time.step"
    for path in (cpp, cpp_history, instrument, executable, stdout, time_step):
        require(path.is_file(), f"missing build/run provenance {path.name}")
    require(
        Path(cpp_history.read_text().strip()).name == cpp.name,
        "cpp.history does not resolve the pinned cpp configuration",
    )
    match = re.search(r"^\s*bld::tool::fppkeys\s+(.+?)\s*$", cpp.read_text(), re.MULTILINE)
    require(match is not None, "missing compiled cpp key line")
    keys = match.group(1).split()
    require(keys == ["key_qco", "key_vco_1d3d", "key_RK3"], f"compiled cpp keys={keys}")
    require(
        sha256(instrument) == "fc34801ae6855e0c559472be4fd9d1b16858befbc5241c07dd9996a99b0affa6",
        "lane-1 step-entry instrument SHA256 mismatch",
    )
    require(int(time_step.read_text()) == 4320, "time.step is not 4320")
    log = stdout.read_text(errors="replace")
    bad = re.search(
        r"ctl_stop|\b(?:nan|inf|infinity)\b|floating[- ]point exception|E R R O R",
        log,
        re.IGNORECASE,
    )
    require(bad is None, f"ocean.output contains failure token {bad.group(0)!r}" if bad else "")
    require("LANE1_STEP_ENTRY_DUMP         4320" in log, "final step-entry marker absent")
    require("GYRE_OMIP_L2_00004320_restart.nc ok" in log, "final restart close absent")
    restart = find_one(root, "*restart*.nc")
    with netCDF4.Dataset(restart) as ds:
        require(float(np.asarray(ds.variables["kt"][:])) == 4320.0, "restart kt")
        require(float(np.asarray(ds.variables["ndastp"][:])) == 21230.0, "restart date")
        require(float(np.asarray(ds.variables["adatrj"][:])) == 720.0, "restart elapsed days")
        require(float(np.asarray(ds.variables["time_counter"][:])[0]) == 4320.0, "restart time")
        require(float(np.asarray(ds.variables["rdt"][:])) == 14400.0, "restart time step")
        scalar = {"kt", "ndastp", "adatrj", "ntime", "rdt"}
        three_d = {"en", "avt_k", "avm_k", "dissl", "un", "vn", "tn", "sn"}
        two_d = {
            "sshbb_e", "ubb_e", "vbb_e", "sshb_e", "ub_e", "vb_e", "fraqsr_1lev",
            "sshn", "uu_n", "vv_n", "ssha",
        }
        for name in scalar:
            require(ds.variables[name].dimensions == (), f"restart {name} dimensions")
        for name in three_d:
            require(ds.variables[name].shape == (1, 31, 22, 32), f"restart {name} dimensions")
        for name in two_d:
            require(ds.variables[name].shape == (1, 22, 32), f"restart {name} dimensions")
    return {
        "compiled_keys": "key_qco key_vco_1d3d key_RK3",
        "executable_sha256": sha256(executable),
        "stdout_sha256": sha256(stdout),
        "completed_step": 4320,
        "elapsed_model_days": 720.0,
        "restart_sha256": sha256(restart),
    }


def array(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    require(name in ds.variables, f"missing mesh array {name}")
    return np.asarray(ds.variables[name][:])


def geometry(
    root: Path, case: str, coord: str, plant: bool = False
) -> dict[str, float | int | str]:
    mesh = find_one(root, "mesh_mask*.nc")
    with netCDF4.Dataset(mesh) as ds:
        if case == "gyre":
            return gyre_geometry(ds, mesh, plant)
        nx, nz, dx = (202, 101, 1000.0) if case == "overflow" else (130, 21, 500.0)
        require(
            len(ds.dimensions["x"]) == nx and len(ds.dimensions["y"]) == 3, "horizontal dimensions"
        )
        require(len(ds.dimensions["nav_lev"]) == nz, "vertical dimension")
        for name, var in ds.variables.items():
            data = np.asarray(var[:])
            if np.issubdtype(data.dtype, np.number):
                require(np.all(np.isfinite(data)), f"mesh {name} contains non-finite values")
        for name in ("e1t", "e1u", "e1v", "e1f", "e2t", "e2u", "e2v", "e2f"):
            a = array(ds, name)
            if plant and name == "e1t":
                a = a.copy()
                a[0, 1, 1] += 1.0
            require(
                a.dtype == np.float64 and np.array_equal(a, np.full_like(a, dx)), f"{name} metric"
            )
        require(np.array_equal(array(ds, "ff_t"), np.zeros((1, 3, nx))), "ff_t must be zero")
        require(np.array_equal(array(ds, "ff_f"), np.zeros((1, 3, nx))), "ff_f must be zero")
        for name in ("tmask", "umask", "vmask"):
            require(set(np.unique(array(ds, name))).issubset({0, 1}), f"{name} is not binary")
        x = array(ds, "glamt")[0, 1]
        require(np.array_equal(np.diff(x), np.full(nx - 1, dx / 1000.0)), "glamt spacing")
        if case == "overflow":
            e3t = array(ds, "e3t_0")[0]
            tmask = array(ds, "tmask")[0]
            wet_depth = np.sum(e3t * tmask, axis=0)
            analytic = 500.0 + 750.0 * (1.0 + np.tanh((x - 40.0) / 7.0))
            if coord == "sco":
                require(
                    np.allclose(wet_depth[1, 1:-1], analytic[1:-1], rtol=0, atol=8e-12),
                    "sco bathymetry",
                )
                require(np.all(array(ds, "mbathy")[0, 1, 1:-1] == 100), "sco bottom index")
                ratio = e3t[:, 1, 1:-1] / analytic[None, 1:-1]
                require(
                    np.allclose(ratio, 0.01, rtol=0, atol=2e-16), "sco terrain-following thickness"
                )
            else:
                mb = array(ds, "mbathy")[0, 1].astype(int)
                bottom = np.array([e3t[k - 1, 1, i] for i, k in enumerate(mb)])
                require(
                    np.all(bottom[1:-1] > 0.0) and np.all(bottom[1:-1] <= 20.0),
                    "zps bottom thickness",
                )
                reconstructed = (mb - 1) * 20.0 + bottom
                require(
                    np.array_equal(wet_depth[1, 1:-1], reconstructed[1:-1]), "zps reconstruction"
                )
                for face in ("e3u_0", "e3v_0", "e3f_0"):
                    require(
                        np.array_equal(array(ds, face), array(ds, "e3t_0")),
                        f"zps face-min specialization {face}",
                    )
        else:
            require(np.all(array(ds, "mbathy")[0, 1, 1:-1] == 20), "LOCK flat bottom")
            require(np.array_equal(array(ds, "e3t_1d"), np.ones((1, nz))), "LOCK unit layers")
    result = {"mesh_sha256": sha256(mesh), "nx": nx, "ny": 3, "nz": nz, "dx_m": dx}
    if case == "overflow" and coord == "zps":
        result["face_min_specialization"] = "VERIFIED"
        result["ten_percent_minimum_arm"] = "WAIVED_DEAD_ARM_key_vco_3d"
    return result


def gyre_geometry(
    ds: netCDF4.Dataset, mesh: Path, plant: bool
) -> dict[str, float | int | str]:
    """Certify the analytic 32 x 22 x 31 GYRE beta-plane mesh."""
    nx, ny, nz, dx = 32, 22, 31, 106000.0
    require(
        (len(ds.dimensions["x"]), len(ds.dimensions["y"]), len(ds.dimensions["nav_lev"]))
        == (nx, ny, nz),
        "GYRE dimensions",
    )
    dims_2d = ("time_counter", "y", "x")
    dims_3d = ("time_counter", "nav_lev", "y", "x")
    dims_1d = ("time_counter", "nav_lev")
    expected_mesh: dict[str, tuple[np.dtype, tuple[str, ...]]] = {}
    for name in ("tmask", "umask", "vmask"):
        expected_mesh[name] = (np.dtype("int8"), dims_3d)
    expected_mesh["fmask"] = (np.dtype("float32"), dims_3d)
    for name in ("tmaskutil", "umaskutil", "vmaskutil"):
        expected_mesh[name] = (np.dtype("int8"), dims_2d)
    for name in (
        "glamt", "glamu", "glamv", "glamf", "gphit", "gphiu", "gphiv", "gphif",
        "e1t", "e1u", "e1v", "e1f", "e2t", "e2u", "e2v", "e2f", "ff_f", "ff_t",
    ):
        expected_mesh[name] = (np.dtype("float64"), dims_2d)
    for name in ("mbathy", "misf"):
        expected_mesh[name] = (np.dtype("int32"), dims_2d)
    for name in ("e3t_1d", "e3w_1d", "gdept_1d", "gdepw_1d"):
        expected_mesh[name] = (np.dtype("float64"), dims_1d)
    for name in ("e3t_0", "e3u_0", "e3v_0", "e3f_0"):
        expected_mesh[name] = (np.dtype("float64"), dims_3d)
    require(set(ds.variables) - META == set(expected_mesh), "GYRE mesh field inventory")
    for name, (dtype, dimensions) in expected_mesh.items():
        var = ds.variables[name]
        require(np.dtype(var.dtype) == dtype, f"GYRE {name} dtype")
        require(var.dimensions == dimensions, f"GYRE {name} dimensions")
    for name, var in ds.variables.items():
        data = np.asarray(var[:])
        if np.issubdtype(data.dtype, np.number):
            require(np.all(np.isfinite(data)), f"mesh {name} contains non-finite values")
    for name in ("e1t", "e1u", "e1v", "e1f", "e2t", "e2u", "e2v", "e2f"):
        a = array(ds, name)
        if plant and name == "e1t":
            a = a.copy()
            a[0, 1, 1] += 1.0
        require(a.dtype == np.float64, f"{name} is not fp64")
        require(np.array_equal(a, np.full_like(a, dx)), f"{name} metric")
    for name in (
        "tmask",
        "umask",
        "vmask",
        "fmask",
        "tmaskutil",
        "umaskutil",
        "vmaskutil",
    ):
        require(set(np.unique(array(ds, name))).issubset({0, 1}), f"{name} is not binary")
    tmask = array(ds, "tmask")[0]
    mbathy = array(ds, "mbathy")[0]
    # usrdef_zgr assigns the analytic flat bottom everywhere; domain boundary
    # masks, not mbathy, impose the free-slip closed walls.
    require(np.all(mbathy == 30), "GYRE flat bottom index")
    for name in ("tmask", "umask", "vmask", "fmask"):
        mask = array(ds, name)[0]
        require(
            np.all(np.sum(mask[:, [0, -1], :], axis=0) == 0)
            and np.all(np.sum(mask[:, :, [0, -1]], axis=0) == 0),
            f"GYRE {name} walls",
        )
    expected_u = np.zeros_like(tmask)
    expected_v = np.zeros_like(tmask)
    expected_f = np.zeros_like(tmask)
    expected_u[:, :, :-1] = tmask[:, :, :-1] * tmask[:, :, 1:]
    expected_v[:, :-1, :] = tmask[:, :-1, :] * tmask[:, 1:, :]
    expected_f[:, :-1, :-1] = (
        tmask[:, :-1, :-1]
        * tmask[:, :-1, 1:]
        * tmask[:, 1:, :-1]
        * tmask[:, 1:, 1:]
    )
    for name, expected in (("umask", expected_u), ("vmask", expected_v), ("fmask", expected_f)):
        require(np.array_equal(array(ds, name)[0], expected), f"GYRE {name} C-grid topology")
    require(np.all(np.sum(tmask[:, 1:-1, 1:-1], axis=0) == 30), "GYRE wet-column levels")
    for mask, utility in (("tmask", "tmaskutil"), ("umask", "umaskutil"), ("vmask", "vmaskutil")):
        require(
            np.array_equal(array(ds, mask)[0, 0], array(ds, utility)[0]),
            f"GYRE {utility} surface identity",
        )
    # NEMO encodes a no-cavity column with the first wet W level, index 1.
    require(np.array_equal(array(ds, "misf"), np.ones((1, ny, nx))), "GYRE no-ISF mask")
    e3t = array(ds, "e3t_0")[0]
    e3t_1d = array(ds, "e3t_1d")[0]
    e3w_1d = array(ds, "e3w_1d")[0]
    gdept_1d = array(ds, "gdept_1d")[0]
    gdepw_1d = array(ds, "gdepw_1d")[0]
    require(np.all(e3t_1d > 0.0) and np.all(e3w_1d > 0.0), "GYRE positive layers")
    require(np.all(np.diff(gdept_1d) > 0.0), "GYRE T depths")
    require(np.all(np.diff(gdepw_1d) > 0.0), "GYRE W depths")
    # Re-evaluate usrdef_zgr.F90:131-166, including depth_to_e3 and
    # e3_to_depth.  Python and Fortran libm differ by at most a few ulps.
    k = np.arange(1, nz + 1, dtype=np.float64)
    zsur, za0, za1, zkth, zacr = (
        -2033.194295283385,
        155.8325369664153,
        146.3615918601890,
        17.28520372419791,
        5.0,
    )
    raw_w = zsur + za0 * k + za1 * zacr * np.log(np.cosh((k - zkth) / zacr))
    raw_t = zsur + za0 * (k + 0.5) + za1 * zacr * np.log(
        np.cosh((k + 0.5 - zkth) / zacr)
    )
    expected_e3w = np.empty(nz)
    expected_e3t = np.empty(nz)
    expected_e3w[0] = 2.0 * (raw_t[0] - raw_w[0])
    expected_e3w[1:] = np.diff(raw_t)
    expected_e3t[:-1] = np.diff(raw_w)
    expected_e3t[-1] = 2.0 * (raw_t[-1] - raw_w[-1])
    expected_w = np.empty(nz)
    expected_t = np.empty(nz)
    expected_w[0] = 0.0
    expected_t[0] = 0.5 * expected_e3w[0]
    expected_w[1:] = np.cumsum(expected_e3t[:-1])
    expected_t[1:] = expected_t[0] + np.cumsum(expected_e3w[1:])
    mi96_libm_pointwise = 0.0
    for name, got, expected in (
        ("e3t_1d", e3t_1d, expected_e3t),
        ("e3w_1d", e3w_1d, expected_e3w),
        ("gdept_1d", gdept_1d, expected_t),
        ("gdepw_1d", gdepw_1d, expected_w),
    ):
        relative = float(np.max(np.abs(got - expected) / np.maximum(np.abs(expected), 1.0)))
        mi96_libm_pointwise = max(mi96_libm_pointwise, relative)
        require(relative <= 512 * np.finfo(np.float64).eps, f"GYRE MI96 libm diagnostic {name}")
    reconstructed_w = np.empty(nz)
    reconstructed_t = np.empty(nz)
    reconstructed_w[0] = 0.0
    reconstructed_t[0] = 0.5 * e3w_1d[0]
    reconstructed_w[1:] = np.cumsum(e3t_1d[:-1])
    reconstructed_t[1:] = reconstructed_t[0] + np.cumsum(e3w_1d[1:])
    for name, got, expected in (
        ("gdepw_1d", gdepw_1d, reconstructed_w),
        ("gdept_1d", gdept_1d, reconstructed_t),
    ):
        relative = np.max(np.abs(got - expected) / np.maximum(np.abs(expected), 1.0))
        require(relative <= 1e-15, f"GYRE pointwise depth-thickness identity {name}")
    require(
        np.array_equal(e3t, np.broadcast_to(e3t_1d[:, None, None], e3t.shape)),
        "GYRE zco thickness",
    )
    for face in ("e3u_0", "e3v_0", "e3f_0"):
        require(np.array_equal(array(ds, face)[0], e3t), f"GYRE zco {face}")
    for name in ("glamt", "glamu", "glamv", "glamf", "gphit", "gphiu", "gphiv", "gphif"):
        a = array(ds, name)[0]
        scale = max(float(np.max(np.abs(a))), 1.0)
        require(np.max(np.abs(np.diff(a, n=2, axis=0))) <= 1e-15 * scale, f"{name} y-affinity")
        require(np.max(np.abs(np.diff(a, n=2, axis=1))) <= 1e-15 * scale, f"{name} x-affinity")
    lon = array(ds, "glamt")[0]
    lat = array(ds, "gphit")[0]
    dlon_x = float(np.median(np.diff(lon, axis=1)))
    dlon_y = float(np.median(np.diff(lon, axis=0)))
    dlat_x = float(np.median(np.diff(lat, axis=1)))
    dlat_y = float(np.median(np.diff(lat, axis=0)))
    require(dlon_x > 0 and dlon_y < 0 and dlat_x > 0 and dlat_y > 0, "GYRE 45-degree orientation")
    require(
        np.isclose(abs(dlon_x), abs(dlon_y), rtol=1e-15, atol=0.0),
        "GYRE longitude rotation",
    )
    require(
        np.isclose(abs(dlat_x), abs(dlat_y), rtol=1e-15, atol=0.0),
        "GYRE latitude rotation",
    )
    for name in ("ff_t", "ff_f"):
        f = array(ds, name)[0]
        require(float(np.ptp(f)) > 0.0 and np.all(f > 0.0), f"{name} beta plane")
        # The analytic source makes f affine in latitude over this all-northern
        # domain.  Boundary rows/columns are subsequently halo-filled, so the
        # source identity is tested over physical interior cells.
        stagger_lat = array(ds, "gphit" if name == "ff_t" else "gphif")[0]
        physical = (slice(1, -1), slice(1, -1))
        coeff = np.polyfit(stagger_lat[physical].ravel(), f[physical].ravel(), 1)
        residual = np.max(
            np.abs(f[physical] - np.polyval(coeff, stagger_lat[physical]))
        ) / np.max(np.abs(f[physical]))
        require(residual <= 1e-15, f"{name} beta-plane affinity")
    return {
        "mesh_sha256": sha256(mesh),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx_m": dx,
        "wet_levels": 30,
        "beta_plane": "VERIFIED",
        "rotation_45deg": "VERIFIED",
        "mi96_libm_diagnostic_pointwise_relative": mi96_libm_pointwise,
    }


def trajectory(root: Path, case: str, steps: list[int], dims: tuple[int, int, int]) -> dict:
    nx, ny, nz = dims
    count = nx * ny * nz
    records = []
    with netCDF4.Dataset(find_one(root, "mesh_mask*.nc")) as mesh:
        x_global = np.asarray(mesh.variables["glamt"][0, 1, :])
        tmask_global = np.asarray(mesh.variables["tmask"][0])
        if "e3t_0" in mesh.variables:
            e3_global = np.asarray(mesh.variables["e3t_0"][0])
        else:
            e3_global = np.broadcast_to(
                np.asarray(mesh.variables["e3t_1d"][0])[:, None, None], tmask_global.shape
            )
    for step in steps:
        path = root / f"oracle_step_entry_kt{step:08d}.bin"
        require(path.is_file(), f"missing trajectory step {step}")
        with path.open("rb") as fh:
            magic = fh.read(16).decode("ascii").rstrip()
            header = struct.unpack("=8i", fh.read(32))
            version, got_step, nbb, jpi, jpj, jpk, jpts, storage = header
            require(magic == "NEMO_L1_ENTRY_1", "trajectory magic")
            require(
                (version, got_step, jpi, jpj, jpk, jpts, storage) == (1, step, nx, ny, nz, 2, 64),
                "trajectory header",
            )
            values = np.fromfile(fh, dtype=np.float64)
        require(values.size == (2 * count + count + count + nx * ny), "trajectory length")
        require(np.all(np.isfinite(values)), f"trajectory {step} non-finite")
        temp = values[:count]
        sal = values[count : 2 * count]
        u = values[2 * count : 3 * count]
        v = values[3 * count : 4 * count]
        ssh = values[4 * count :]
        tlo, thi = (10.0, 20.0) if case == "overflow" else (5.0, 30.0)
        eps = np.finfo(np.float64).eps
        wet_t = temp[temp != 0.0]
        wet_s = sal[sal != 0.0]
        require(bool(wet_t.size and wet_s.size), "empty wet tracer field")
        # Relative closed-range excess, scaled separately for each tracer.
        # The fp64 roundoff floor is sqrt(N_steps) * eps * field_scale.
        # Roundoff-scale excess is classified without prejudging its origin;
        # a separate gross guard ensures a real limiter failure cannot pass.
        temp_scale = max(abs(tlo), abs(thi), 1.0)
        sal_scale = 35.0
        temp_excess = max(tlo - float(wet_t.min()), float(wet_t.max()) - thi, 0.0)
        sal_excess = float(np.max(np.abs(wet_s - 35.0)))
        floor_relative = np.sqrt(float(step)) * eps
        temp_relative = temp_excess / temp_scale
        sal_relative = sal_excess / sal_scale
        require(
            temp_relative <= GROSS_TRACER_EXCESS_RELATIVE,
            f"gross temperature range excursion at step {step}: {temp_relative:.17g} relative",
        )
        require(
            sal_relative <= GROSS_TRACER_EXCESS_RELATIVE,
            f"gross salinity range excursion at step {step}: {sal_relative:.17g} relative",
        )
        temp_status = tracer_bar(temp_relative, floor_relative)
        sal_status = tracer_bar(sal_relative, floor_relative)
        temp_global = temp.reshape((nx, ny, nz), order="F")[2:-2, 2:-2, :].transpose(2, 1, 0)
        cold_weight = np.maximum(thi - temp_global, 0.0) * e3_global * tmask_global
        cold_center = float(np.sum(cold_weight * x_global[None, None, :]) / np.sum(cold_weight))
        records.append(
            {
                "step": step,
                "Nbb": nbb,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "temperature_min": float(wet_t.min()),
                "temperature_max": float(wet_t.max()),
                "salinity_min": float(wet_s.min()),
                "salinity_max": float(wet_s.max()),
                "velocity_max_abs": float(max(np.max(np.abs(u)), np.max(np.abs(v)))),
                "ssh_max_abs": float(np.max(np.abs(ssh))),
                "cold_center_x_km": cold_center,
                "bar_definition": "excess/field_scale <= sqrt(N_steps)*fp64_eps",
                "roundoff_floor_relative": floor_relative,
                "temperature_excess_relative": temp_relative,
                "temperature_excess_eps_relative": temp_relative / eps,
                "salinity_excess_relative": sal_relative,
                "salinity_excess_eps_relative": sal_relative / eps,
                "temperature_bar": temp_status,
                "salinity_bar": sal_status,
            }
        )
    return {"time_level": "Nbb/before", "storage_bits": 64, "records": records}


def gyre_trajectory(root: Path, steps: list[int]) -> dict:
    """Check exact GYRE IC structure and gross seasonal spin-up phenomenology."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    nx, ny, nz = 36, 26, 31  # serial local domain, including two-cell halos
    count = nx * ny * nz
    with netCDF4.Dataset(find_one(root, "mesh_mask*.nc")) as mesh:
        wet = np.asarray(mesh.variables["tmask"][0], dtype=bool)
        latitude = np.asarray(mesh.variables["gphit"][0])
    records = []
    dump_level: str | None = None
    for step in steps:
        path = root / f"oracle_step_entry_kt{step:08d}.bin"
        require(path.is_file(), f"missing trajectory step {step}")
        level = time_level_for_dump(path.name)
        require(level == "before", f"trajectory {step} is not registered before")
        dump_level = level
        with path.open("rb") as fh:
            magic = fh.read(16).decode("ascii").rstrip()
            version, got_step, nbb, jpi, jpj, jpk, jpts, storage = struct.unpack("=8i", fh.read(32))
            require(magic == "NEMO_L1_ENTRY_1", "trajectory magic")
            require(
                (version, got_step, jpi, jpj, jpk, jpts, storage)
                == (1, step, nx, ny, nz, 2, 64),
                "trajectory header",
            )
            values = np.fromfile(fh, dtype=np.float64)
        require(values.size == 4 * count + nx * ny, "trajectory length")
        require(np.all(np.isfinite(values)), f"trajectory {step} non-finite")
        fields = []
        for offset in range(4):
            local = values[offset * count : (offset + 1) * count].reshape((nx, ny, nz), order="F")
            fields.append(local[2:-2, 2:-2, :].transpose(2, 1, 0))
        temp, sal, u, v = fields
        ssh = values[4 * count :].reshape((nx, ny), order="F")[2:-2, 2:-2].T
        wet_t, wet_s = temp[wet], sal[wet]
        wet_u, wet_v = u[wet], v[wet]
        wet_ssh = ssh[wet[0]]
        require(wet_t.size > 0 and wet_s.size > 0, "empty wet tracer field")
        if step == 1:
            for name, field in (("temperature", temp), ("salinity", sal)):
                for level in range(nz):
                    level_wet = wet[level]
                    if np.any(level_wet):
                        require(
                            np.ptp(field[level][level_wet]) == 0.0,
                            f"GYRE initial {name} is not horizontally uniform",
                        )
            require(np.max(np.abs(wet_u)) == 0.0, "GYRE initial u is not zero")
            require(np.max(np.abs(wet_v)) == 0.0, "GYRE initial v is not zero")
            require(np.max(np.abs(wet_ssh)) == 0.0, "GYRE initial ssh is not zero")
        south = wet[0] & (latitude <= 30.0)
        north = wet[0] & (latitude >= 40.0)
        records.append(
            {
                "step": step,
                "Nbb": nbb,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "temperature_min": float(wet_t.min()),
                "temperature_max": float(wet_t.max()),
                "salinity_min": float(wet_s.min()),
                "salinity_max": float(wet_s.max()),
                "velocity_max_abs": float(max(np.max(np.abs(wet_u)), np.max(np.abs(wet_v)))),
                "ssh_min": float(wet_ssh.min()),
                "ssh_max": float(wet_ssh.max()),
                "ssh_south_mean": float(np.mean(ssh[south])),
                "ssh_north_mean": float(np.mean(ssh[north])),
            }
        )
    final = records[-1]
    require(final["velocity_max_abs"] > 0.0, "GYRE did not spin up velocity")
    require(final["ssh_min"] < 0.0 < final["ssh_max"], "GYRE final SSH lacks gyre relief")
    require(
        final["ssh_south_mean"] > 0.0 > final["ssh_north_mean"],
        "GYRE final SSH lacks north-south double-gyre structure",
    )
    return {
        "time_level": f"Nbb/{dump_level}",
        "storage_bits": 64,
        "initial_condition_status": "VERIFIED",
        "seasonal_gyre_spinup_status": "VERIFIED",
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--case", choices=("overflow", "lock_exchange", "gyre"), required=True)
    ap.add_argument("--coord", choices=("zps", "sco", "zco"), required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--emit-manifest", action="store_true")
    ap.add_argument("--plant-unaccounted", action="store_true")
    ap.add_argument("--plant-geometry", action="store_true")
    ap.add_argument("--geometry-coverage-only", action="store_true")
    args = ap.parse_args()
    root = args.run_dir.resolve()
    require(args.case != "gyre" or args.coord == "zco", "GYRE coordinate label must be zco")
    if args.emit_manifest:
        print(json.dumps(disposition_template(root, args.case), indent=2, sort_keys=True))
        return 0
    require(args.manifest is not None, "--manifest is required")
    manifest = json.loads(args.manifest.read_text())
    inventory = check_manifest(root, manifest, plant_unaccounted=args.plant_unaccounted)
    resolved_selectors(root, args.case, args.coord)
    provenance = build_run_provenance(root) if args.case == "gyre" else None
    g = geometry(root, args.case, args.coord, plant=args.plant_geometry)
    if args.geometry_coverage_only:
        print(
            json.dumps(
                {
                    "status": "VERIFIED",
                    "case": args.case,
                    "coord": args.coord,
                    "inventory_counts": {k: len(v) for k, v in inventory.items()},
                    "geometry": g,
                    **({"build_run": provenance} if provenance is not None else {}),
                    "unmeasured": BASE_UNMEASURED,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.case == "gyre":
        t = gyre_trajectory(root, [1, 2160, 4320])
        report = {
            "status": "VERIFIED",
            "case": args.case,
            "coord": args.coord,
            "inventory_counts": {k: len(v) for k, v in inventory.items()},
            "geometry": g,
            "build_run": provenance,
            "trajectory": t,
            "unmeasured": list(BASE_UNMEASURED),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    steps = [1, 3060, 6120] if args.case == "overflow" else [1, 30600, 61200]
    # Serial local arrays retain the two-cell NEMO halo on every horizontal
    # side; mesh_mask is written on the 202x3 / 130x3 global domain.
    dims = (206, 7, 101) if args.case == "overflow" else (134, 7, 21)
    t = trajectory(root, args.case, steps, dims)
    unmeasured = list(BASE_UNMEASURED)
    if any(r["temperature_bar"] == "UNMEASURED" for r in t["records"]):
        unmeasured.append("temperature_range_roundoff_origin")
    if any(r["salinity_bar"] == "UNMEASURED" for r in t["records"]):
        unmeasured.append("salinity_range_roundoff_origin")
    report = {
        # Coverage gaps are carried separately in `unmeasured`; status reports
        # only whether a measured trajectory row cleared its registered bar.
        "status": measured_trajectory_status(t["records"]),
        "case": args.case,
        "coord": args.coord,
        "inventory_counts": {k: len(v) for k, v in inventory.items()},
        "geometry": g,
        "trajectory": t,
        "unmeasured": unmeasured,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"DEBT: {exc}", file=sys.stderr)
        raise SystemExit(1)
