#!/usr/bin/env python3
"""Coverage-first legoESM geometry/IC/kt=1 gate for certified NEMO cases."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import netCDF4
import numpy as np

POINTWISE_BAR = 1.0e-15
ACCUMULATING_BAR = 1.0e-12
VALID = {"VERIFIED", "WAIVED", "UNMEASURED"}
UNMEASURED = [
    "kt_gt_1_trajectory",
    "rk3_stage_tendencies",
    "teos10_density",
    "rab",
    "bn2",
    "fct_tendency",
    "adaptive_vertical_advection_partition",
    "active_up3_vertical_momentum_parity",
    "oracle_barotropic_time_filters",
    "tracer_rk3_parity",
    "bbl_transport",
]
ROOTS = {
    "LOCK_EXCHANGE-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/lock_exchange_zco"
    ),
    "OVERFLOW-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/overflow_zps"
    ),
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _llz(a) -> np.ndarray:
    """NEMO ``(z,y,x)`` to legoESM ``(y,x,z)``."""
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


def _score(
    rows: list[dict],
    dtypes: dict[str, dict[str, str]],
    name: str,
    oracle,
    lego,
    *,
    mask=None,
    exact: bool = False,
    arithmetic_class: str = "POINTWISE",
    plant: bool = False,
) -> None:
    oracle = np.asarray(oracle)
    lego = np.asarray(lego)
    if plant:
        lego = lego.copy()
        lego.flat[0] += 1.0
    require(oracle.shape == lego.shape, f"{name}: shape {oracle.shape} != {lego.shape}")
    dtypes[name] = {"oracle": str(oracle.dtype), "legoesm": str(lego.dtype)}
    if np.issubdtype(lego.dtype, np.floating):
        require(lego.dtype == np.float64, f"{name}: legoESM dtype {lego.dtype} != float64")
    require(np.all(np.isfinite(oracle)), f"{name}: oracle nonfinite")
    require(np.all(np.isfinite(lego)), f"{name}: legoESM nonfinite")
    use = np.ones(oracle.shape, dtype=bool) if mask is None else np.asarray(mask, bool)
    require(use.shape == oracle.shape and bool(use.any()), f"{name}: empty/invalid mask")
    if exact:
        error = 0.0 if np.array_equal(oracle[use], lego[use]) else float("inf")
        bar = 0.0
    else:
        scale = max(float(np.max(np.abs(oracle[use]))), 1.0)
        error = float(np.max(np.abs(lego[use] - oracle[use]))) / scale
        bar = POINTWISE_BAR if arithmetic_class == "POINTWISE" else ACCUMULATING_BAR
    status = "AT-BAR" if error <= bar else "DEBT"
    rows.append(
        {
            "name": name,
            "status": status,
            "arithmetic_class": arithmetic_class,
            "normalized_max_abs": error,
            "bar": bar,
            "n": int(use.sum()),
        }
    )
    require(status == "AT-BAR", f"{name}: {error:.17g} > {bar:.1e}")


def _load_registry(path: Path, mesh: netCDF4.Dataset, plant: bool) -> dict:
    registry = json.loads(path.read_text())
    require(registry.get("format") == "nemo-testcase-l1-phase2-registry-v1", "bad registry format")
    actual = set(mesh.variables)
    if plant:
        actual.add("PLANTED_UNACCOUNTED_FILE_ARRAY")
    ledger = registry.get("mesh", {})
    require(
        set(ledger) == actual,
        "mesh coverage mismatch: "
        f"missing={sorted(actual - set(ledger))}, extra={sorted(set(ledger) - actual)}",
    )
    for name, item in ledger.items():
        require(item.get("status") in VALID, f"mesh.{name}: bad disposition")
        require(bool(item.get("reason", "").strip()), f"mesh.{name}: empty reason")
    return registry


def _card(case: str):
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64())
    card = build_nemo_testcase_card(case)
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    return card


def _expected_masks(active: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    umask = active & np.roll(active, -1, axis=1)
    umask[:, -1, :] = False
    vmask = active & np.roll(active, -1, axis=0)
    vmask[-1, :, :] = False
    fmask = umask & np.roll(umask, -1, axis=0)
    fmask[-1, :, :] = False
    return umask, vmask, fmask


def geometry_gate(
    case: str,
    root: Path,
    registry_path: Path,
    *,
    plant_unaccounted: bool = False,
    plant_geometry: bool = False,
) -> tuple[object, list[dict], dict, dict]:
    card = _card(case)
    recipe = card.recipe
    grid, zc = recipe.grid, recipe.z_coord
    mesh_path = root / "mesh_mask.nc"
    require(mesh_path.is_file(), f"missing {mesh_path}")
    rows: list[dict] = []
    dtypes: dict[str, dict[str, str]] = {}
    with netCDF4.Dataset(mesh_path) as ds:
        registry = _load_registry(registry_path, ds, plant_unaccounted)
        for name, item in registry["mesh"].items():
            if item["status"] == "VERIFIED":
                require(np.issubdtype(np.asarray(ds[name][:]).dtype, np.number), f"mesh.{name}: nonnumeric")

        ny, nx, nlev = 3, grid.n_lon, zc.n_levels
        wet2 = np.asarray(recipe.initial_state.land_mask.data) > 0.5
        active = np.asarray(zc.is_active) & wet2[..., None]
        active_file = np.concatenate([active, np.zeros((ny, nx, 1), bool)], axis=-1)
        umask, vmask, fmask = _expected_masks(active_file)

        dx = float(np.asarray(grid.dx_T)[0, 0])
        x_t = np.asarray(grid.lon_T) * grid.radius / 1000.0
        y_t = np.asarray(grid.lat_T) * grid.radius / 1000.0
        x_u = np.broadcast_to(np.arange(nx, dtype=np.float64) * dx / 1000.0, (ny, nx))
        y_v = np.broadcast_to((np.arange(ny, dtype=np.float64) * dx / 1000.0)[:, None], (ny, nx))

        common = {
            "glamt": x_t,
            "glamu": x_u,
            "glamv": x_t,
            "glamf": x_u,
            "gphit": y_t,
            "gphiu": y_t,
            "gphiv": y_v,
            "gphif": y_v,
            "e1t": np.asarray(grid.dx_T),
            "e1u": np.asarray(grid.dx_u)[:, 1:],
            "e1v": np.asarray(grid.dx_v)[1:],
            "e1f": np.full((ny, nx), dx),
            "e2t": np.asarray(grid.dy_T),
            "e2u": np.asarray(grid.dy_u)[:, 1:],
            "e2v": np.asarray(grid.dy_v)[1:],
            "e2f": np.full((ny, nx), dx),
            "ff_t": np.asarray(grid.f_T),
            "ff_f": np.asarray(grid.f_v)[1:],
        }
        for name, lego in common.items():
            _score(
                rows,
                dtypes,
                f"geometry.{name}",
                np.asarray(ds[name][0]),
                lego,
                plant=plant_geometry and name == "e1t",
            )

        exact = {
            "tmask": active_file,
            "umask": umask,
            "vmask": vmask,
            "fmask": fmask,
            "tmaskutil": active_file[..., 0],
            "umaskutil": umask[..., 0],
            "vmaskutil": vmask[..., 0],
        }
        for name, lego in exact.items():
            _score(rows, dtypes, f"geometry.{name}", _llz(ds[name][0]) if name in {"tmask", "umask", "vmask", "fmask"} else np.asarray(ds[name][0]), lego, exact=True)

        mbathy = active.sum(axis=-1).astype(np.int32)
        _score(rows, dtypes, "geometry.mbathy", np.asarray(ds["mbathy"][0]), mbathy, mask=wet2, exact=True)

        dz = np.asarray(zc.dz_ref)
        tdepth = np.abs(np.asarray(zc.z_full_ref))
        wdepth = np.abs(np.asarray(zc.z_half_ref))
        vertical_1d = {
            "e3t_1d": np.concatenate([dz, dz[-1:]]),
            "e3w_1d": np.concatenate([dz, dz[-1:]]),
            "gdept_1d": np.concatenate([tdepth, tdepth[-1:] + dz[-1]]),
            "gdepw_1d": wdepth,
        }
        for name, lego in vertical_1d.items():
            _score(rows, dtypes, f"geometry.{name}", np.asarray(ds[name][0]), lego)

        if case == "OVERFLOW-zps":
            from legoesm.ocean.dynamics.latlon_cgrid_operators import (
                min_cell_to_uface,
                min_cell_to_vface,
            )

            hp = np.asarray(zc.h_partial)
            file_e3t = _llz(ds["e3t_0"][0])
            _score(rows, dtypes, "geometry.e3t_0_wet", file_e3t[..., :nlev], hp, mask=active)
            hu = np.asarray(min_cell_to_uface(zc.h_partial))[:, 1:, :]
            _score(rows, dtypes, "geometry.e3u_0_min", _llz(ds["e3u_0"][0])[..., :nlev], hu, mask=umask[..., :nlev])
            # The three-row tank has no active V/F transport face.  Their mesh
            # arrays are still source-verified as the OVERFLOW specialization
            # e3v=e3f=e3t, while the registry states the dynamical waiver.
            _score(rows, dtypes, "geometry.e3v_0_specialization", _llz(ds["e3v_0"][0]), file_e3t)
            _score(rows, dtypes, "geometry.e3f_0_specialization", _llz(ds["e3f_0"][0]), file_e3t)
            e3w = np.broadcast_to(np.concatenate([dz, dz[-1:]]), file_e3t.shape)
            for name in ("e3w_0", "e3uw_0", "e3vw_0"):
                _score(rows, dtypes, f"geometry.{name}", _llz(ds[name][0]), e3w)
            depth_t = np.broadcast_to(np.concatenate([tdepth, tdepth[-1:] + dz[-1]]), file_e3t.shape)
            depth_w = np.broadcast_to(wdepth, file_e3t.shape)
            _score(rows, dtypes, "geometry.gdept_0", _llz(ds["gdept_0"][0]), depth_t)
            _score(rows, dtypes, "geometry.gdepw_0", _llz(ds["gdepw_0"][0]), depth_w)
            require(not np.allclose(hu[active[:, :, :nlev]], 0.5 * (hp + np.roll(hp, -1, axis=1))[active]), "min-rule control is vacuous")

    return card, rows, dtypes, registry


def read_step_entry(path: Path, case: str) -> dict[str, np.ndarray | int | str]:
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    require(time_level_for_dump(path.name) == "before", "kt=1 dump is not registered before")
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        version, step, nbb, nx, ny, nz, ntr, bits = struct.unpack("=8i", fh.read(32))
        data = np.fromfile(fh, dtype=np.float64)
    require(magic == "NEMO_L1_ENTRY_1", "step-entry magic")
    expected = (206, 7, 101) if case == "OVERFLOW-zps" else (134, 7, 21)
    require((version, step, nx, ny, nz, ntr, bits) == (1, 1, *expected, 2, 64), "step-entry header")
    count = nx * ny * nz
    require(data.size == 4 * count + nx * ny, "step-entry payload length")

    def xyz(values):
        return values.reshape((nx, ny, nz), order="F")[2:-2, 2:-2, :].transpose(1, 0, 2)

    return {
        "time_level": "before",
        "Nbb": nbb,
        "T": xyz(data[:count]),
        "S": xyz(data[count : 2 * count]),
        "u": xyz(data[2 * count : 3 * count]),
        "v": xyz(data[3 * count : 4 * count]),
        "ssh": data[4 * count :].reshape((nx, ny), order="F")[2:-2, 2:-2].T,
    }


def ic_step1_gate(card, root: Path, *, plant_ic: bool = False) -> tuple[list[dict], dict]:
    state = card.recipe.initial_state
    oracle = read_step_entry(root / "oracle_step_entry_kt00000001.bin", card.case)
    nlev = card.recipe.z_coord.n_levels
    rows: list[dict] = []
    dtypes: dict[str, dict[str, str]] = {}
    fields = {
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        "u": np.asarray(state.u.data)[:, 1:, :],
        "v": np.asarray(state.v.data)[1:, :, :],
        "ssh": np.asarray(state.eta.data),
    }
    for name, lego in fields.items():
        ref = np.asarray(oracle[name])
        if name != "ssh":
            ref = ref[..., :nlev]
        _score(
            rows,
            dtypes,
            f"step1.before.{name}",
            ref,
            lego,
            exact=True,
            plant=plant_ic and name == "T",
        )
        if name != "T":
            # Exact equality is retained as a control, but these fields carry
            # no alignment signal at kt=1: S is spatially uniform over wet
            # cells and u/v/SSH are at-rest zeros.  Only the T front can expose
            # a staggering/index displacement in this dump.
            rows[-1]["alignment_status"] = rows[-1]["status"]
            rows[-1]["status"] = "UNMEASURED"
            rows[-1]["reason"] = (
                "exact kt=1 control is non-informative for staggering; "
                "only the nonuniform T front measures alignment"
            )
    return rows, dtypes


def run(args: argparse.Namespace) -> dict:
    root = args.run_dir or ROOTS[args.case]
    card, geometry_rows, geometry_dtypes, registry = geometry_gate(
        args.case,
        root,
        args.registry,
        plant_unaccounted=args.plant_unaccounted,
        plant_geometry=args.plant_geometry,
    )
    step_rows, step_dtypes = ic_step1_gate(card, root, plant_ic=args.plant_ic)
    all_rows = geometry_rows + step_rows
    return {
        "status": "VERIFIED" if all(r["status"] != "DEBT" for r in all_rows) else "DEBT",
        "case": args.case,
        "precision_policy": "fp64",
        "bars": {"POINTWISE": POINTWISE_BAR, "ACCUMULATING": ACCUMULATING_BAR},
        "coverage": {
            "provided": len(registry["mesh"]),
            "verified": sum(v["status"] == "VERIFIED" for v in registry["mesh"].values()),
            "waived": sum(v["status"] == "WAIVED" for v in registry["mesh"].values()),
            "unmeasured": sum(v["status"] == "UNMEASURED" for v in registry["mesh"].values()),
        },
        "geometry": geometry_rows,
        "initial_condition": {
            "identity": (
                "T alignment is measured by exact equality to registered "
                "kt=1 Nbb/before; S/u/v/SSH exact controls are alignment "
                "UNMEASURED because uniform or zero"
            ),
            "rows": step_rows,
        },
        "dtypes": {**geometry_dtypes, **step_dtypes},
        "unmeasured": UNMEASURED,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(ROOTS), required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-unaccounted", action="store_true")
    parser.add_argument("--plant-geometry", action="store_true")
    parser.add_argument("--plant-ic", action="store_true")
    args = parser.parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"DEBT: {exc}", file=sys.stderr)
        raise SystemExit(1)
