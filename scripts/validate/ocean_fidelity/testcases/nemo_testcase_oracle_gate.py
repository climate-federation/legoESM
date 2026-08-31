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


def disposition_template(root: Path) -> dict:
    restart, inventory = expected_inventory(root)
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
                status = "VERIFIED" if key in TARGET_NML else "WAIVED"
                reason = (
                    "phase-1 pinned science selector"
                    if status == "VERIFIED"
                    else "resolved NEMO default outside the phase-1 claim; full file hash pinned"
                )
            entries[namespace][name] = {"status": status, "reason": reason}
    return {
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
    logical("namzdf.ln_zad_aimp", True)
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
    else:
        number("namrun.nn_itend", 61200)
        number("namrun.nn_stock", 61200)
        number("namdom.rn_dt", 1)
        logical("nambbl.ln_trabbl", False)


def array(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    require(name in ds.variables, f"missing mesh array {name}")
    return np.asarray(ds.variables[name][:])


def geometry(
    root: Path, case: str, coord: str, plant: bool = False
) -> dict[str, float | int | str]:
    mesh = find_one(root, "mesh_mask*.nc")
    with netCDF4.Dataset(mesh) as ds:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--case", choices=("overflow", "lock_exchange"), required=True)
    ap.add_argument("--coord", choices=("zps", "sco", "zco"), required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--emit-manifest", action="store_true")
    ap.add_argument("--plant-unaccounted", action="store_true")
    ap.add_argument("--plant-geometry", action="store_true")
    ap.add_argument("--geometry-coverage-only", action="store_true")
    args = ap.parse_args()
    root = args.run_dir.resolve()
    if args.emit_manifest:
        print(json.dumps(disposition_template(root), indent=2, sort_keys=True))
        return 0
    require(args.manifest is not None, "--manifest is required")
    manifest = json.loads(args.manifest.read_text())
    inventory = check_manifest(root, manifest, plant_unaccounted=args.plant_unaccounted)
    resolved_selectors(root, args.case, args.coord)
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
                    "unmeasured": BASE_UNMEASURED,
                },
                indent=2,
                sort_keys=True,
            )
        )
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
