#!/usr/bin/env python3
"""Discriminate the SWG_DG certified-CPU last-bit stage debt."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

DG_STAGE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/"
    "round1/oracle_swg_dg_r1/instrumented")
SHIPPED_STAGE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/"
    "round1/oracle_swg_r1/run")
RANKCHECK_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/"
    "maddison_deck/rankcheck_kt10")

RESTART_STATE_FIELDS = (
    "utau_b", "vtau_b", "qns_b", "emp_b", "sfx_b", "sshn", "un",
    "vn", "tn", "sn", "uu_n", "vv_n",
)
RESTART_SCALARS = ("kt", "ndastp", "adatrj", "ntime", "rdt")


TRACE_SOURCE = {
    "zonal_difference": "divhor.f90:135-136",
    "meridional_difference": "divhor.f90:137-138",
    "numerator": "divhor.f90:135-139",
    "area_product": "divhor.f90:139",
    "transport_divergence": "divhor.f90:139",
    "hdiv": "divhor.f90:139",
    "e3_hdiv_product": "sshwzv.f90:139",
    "e3_hdiv": "sshwzv.f90:139",
}
PROBE_NAMES = ("baseline", *TRACE_SOURCE)


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    require(spec is not None and spec.loader is not None,
            f"cannot load sibling module {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def bit_summary(reference, candidate) -> dict:
    """Return exact IEEE comparison metadata without treating signed zero equal."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    require(reference.shape == candidate.shape, "comparison shapes differ")
    require(reference.dtype == candidate.dtype and reference.dtype.kind == "f",
            "comparison requires matching floating dtypes")
    word = np.dtype(f"u{reference.dtype.itemsize}")
    different = (
        np.ascontiguousarray(reference).view(word)
        != np.ascontiguousarray(candidate).view(word)
    )
    at = np.argwhere(different)
    delta = np.abs(candidate - reference)
    return {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "bit_identical": not bool(at.size),
        "bit_unequal": int(at.shape[0]),
        "first_unequal_cell": at[0].tolist() if at.size else None,
        "max_abs": float(np.max(delta)) if delta.size else 0.0,
        "reference_sha256": array_sha256(reference),
        "candidate_sha256": array_sha256(candidate),
    }


def _divergence_probe(probe_at: str):
    """Return the production divergence with one diagnostic identity boundary."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round

    require(probe_at in PROBE_NAMES, f"unknown statement probe {probe_at}")

    def mark(name, value):
        return nemo_source_round(value) if probe_at == name else value

    def divergence(
        flux_u,
        flux_u_west,
        flux_v,
        flux_v_south,
        r1_area_t,
        live_e3t,
        tmask3,
        *,
        runoff_mass_flux=None,
        r1_rho0=None,
    ):
        require(runoff_mass_flux is None and r1_rho0 is None,
                "round-6 probe supports the executed no-runoff arm only")
        zonal = jax.lax.optimization_barrier(flux_u - flux_u_west)
        zonal = mark("zonal_difference", zonal)
        meridional = jax.lax.optimization_barrier(flux_v - flux_v_south)
        meridional = mark("meridional_difference", meridional)
        numerator = jax.lax.optimization_barrier(zonal + meridional)
        numerator = mark("numerator", numerator)
        area_product = jax.lax.optimization_barrier(numerator * r1_area_t)
        area_product = mark("area_product", area_product)
        transport_divergence = area_product * tmask3
        transport_divergence = mark(
            "transport_divergence", transport_divergence)
        safe_e3t = jnp.where(tmask3 > 0.5, live_e3t, 1.0)
        hdiv = jax.lax.optimization_barrier(
            transport_divergence / safe_e3t)
        hdiv = mark("hdiv", hdiv)
        e3_hdiv_product = jax.lax.optimization_barrier(live_e3t * hdiv)
        e3_hdiv_product = mark("e3_hdiv_product", e3_hdiv_product)
        e3_hdiv = e3_hdiv_product * tmask3
        return mark("e3_hdiv", e3_hdiv)

    return divergence


def build_statement_bisect(entry, card):
    """Replay the full production stage with one optional statement barrier."""
    import jax
    import legoesm.ocean.dynamics.swe_rk3 as swe
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    original = swe.nemo_transport_wzv_divergence_level
    outputs = {}
    try:
        for probe_at in PROBE_NAMES:
            swe.nemo_transport_wzv_divergence_level = _divergence_probe(probe_at)
            model = LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord,
                card.recipe.model_config)
            result = model.replay_swe_rk3_stage(
                entry, card.dt_s, 1, entry).ssh_k
            jax.block_until_ready(result)
            outputs[probe_at] = result
    finally:
        swe.nemo_transport_wzv_divergence_level = original
    return outputs


def _stage_item(fields):
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.swe_rk3 import SWERK3StageEntry

    return SWERK3StageEntry(**{
        name: jnp.asarray(fields[name], dtype=jnp.float64)
        for name in SWERK3StageEntry._fields
    })


def run_replay(stage_root: Path, deck: str, npz_path: Path, *, plant=False) -> dict:
    """Replay admitted kt=2 stage 1 and save a backend-comparable trace."""
    import jax
    import jaxlib
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card
    from legoesm.ocean.fidelity.provenance import git_sha

    ladder = _load_sibling(
        "nemo_testcase_l1_swg_ladder_round6", "nemo_testcase_l1_swg_ladder.py")
    backend = jax.default_backend()
    require(backend in ("cpu", "gpu"), f"unsupported replay backend {backend!r}")
    set_policy(PrecisionPolicy.fp64(
        transcendentals="libm" if backend == "cpu" else "native"))
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    namelist = ladder._deck_namelist(deck)  # noqa: SLF001 - one shared deck registry
    entries = ladder.read_stage_entries(stage_root)
    current = {name: value.copy() for name, value in entries[2, 1].items()}
    target = entries[2, 2]
    card = build_swg_card(namelist)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    masks = ladder._card_masks(card)  # noqa: SLF001 - exact ladder masks

    entry = _stage_item(current)
    output = model.replay_swe_rk3_stage(entry, card.dt_s, 1, entry)
    jax.block_until_ready(output.ssh_k)
    probes = build_statement_bisect(entry, card)
    output_np = {name: np.asarray(value)
                 for name, value in output._asdict().items()}
    probe_np = {name: np.asarray(value) for name, value in probes.items()}
    trace_control = bit_summary(output_np["ssh_k"], probe_np["baseline"])
    require(trace_control["bit_identical"],
            "statement-bisect baseline SSH differs from production stage")

    plant_result = None
    if plant:
        wet_nonzero = np.argwhere(masks["U"] & (current["u_k"] != 0.0))
        require(wet_nonzero.size > 0, "one-ULP plant has no nonzero wet U")
        at = tuple(wet_nonzero[np.argmin(
            np.abs(current["u_k"][tuple(wet_nonzero.T)]))])
        moved = {name: value.copy() for name, value in current.items()}
        moved["u_k"][at] = np.nextafter(
            moved["u_k"][at], np.float64(np.inf))
        moved_entry = _stage_item(moved)
        moved_output = model.replay_swe_rk3_stage(
            moved_entry, card.dt_s, 1, moved_entry)
        plant_result = bit_summary(
            output_np["ssh_k"], np.asarray(moved_output.ssh_k))
        require(not plant_result["bit_identical"],
                "one-ULP admitted U plant did not change production SSH")
        plant_result.update({"field": "u_k", "cell": list(at)})

    require(not npz_path.exists(), f"refusing to overwrite {npz_path}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    saved = {f"probe__{name}": value for name, value in probe_np.items()}
    saved.update({f"output__{name}": value for name, value in output_np.items()})
    saved.update({f"input__{name}": value for name, value in current.items()})
    np.savez(npz_path, **saved)

    field_rows = {}
    for name, value in output_np.items():
        regions = ladder._localized_comparison(  # noqa: SLF001
            target[name], value, ladder._field_mask(name, masks))  # noqa: SLF001
        field_rows[name] = {
            "bit_unequal": sum(row["bit_unequal"] for row in regions.values()),
            "max_abs": max(row["max_abs"] for row in regions.values()),
            "regions": regions,
        }
    policy = get_policy()
    return {
        "label": "SWG-DG-ROUND6-BACKEND-REPLAY",
        "git_sha": git_sha(),
        "deck": deck,
        "backend": backend,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "precision_policy": {
            "compute": str(policy.compute),
            "transcendentals": policy.transcendentals,
        },
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "jaxlib_git_hash": getattr(
            __import__("jaxlib.version", fromlist=["_git_hash"]),
            "_git_hash", None),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "stage_root": str(stage_root.resolve()),
        "stage_record_hashes": ladder._pinned_stage_hashes(stage_root),  # noqa: SLF001
        "namelist": namelist._asdict(),
        "input_sha256": {
            name: array_sha256(value) for name, value in current.items()},
        "field_rows": field_rows,
        "statement_probe_rows": {
            name: bit_summary(target["ssh_k"], value)
            for name, value in probe_np.items()
        },
        "probe_baseline_matches_production": trace_control,
        "probe_source": TRACE_SOURCE,
        "replay_npz": str(npz_path.resolve()),
        "replay_npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "plant": plant_result,
    }


def compare_replays(left_path: Path, right_path: Path) -> dict:
    """Find the first source-ordered difference between two saved replays."""
    require(left_path.is_file() and right_path.is_file(),
            "both replay NPZ files are required")
    with np.load(left_path) as left, np.load(right_path) as right:
        require(set(left.files) == set(right.files), "replay payload keys differ")
        input_rows = {
            key.removeprefix("input__"): bit_summary(left[key], right[key])
            for key in left.files if key.startswith("input__")
        }
        require(all(row["bit_identical"] for row in input_rows.values()),
                "replay inputs differ")
        probe_rows = []
        for name in PROBE_NAMES:
            row = bit_summary(left[f"probe__{name}"], right[f"probe__{name}"])
            row.update({
                "statement": name,
                "source": TRACE_SOURCE.get(name, "production baseline"),
            })
            probe_rows.append(row)
        output_rows = {
            key.removeprefix("output__"): bit_summary(left[key], right[key])
            for key in left.files if key.startswith("output__")
        }
    return {
        "label": "SWG-DG-ROUND6-REPLAY-COMPARISON",
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
        "inputs_bit_identical": True,
        "first_probe_difference": next(
            (row for row in probe_rows if not row["bit_identical"]), None),
        "probe_rows": probe_rows,
        "output_rows": output_rows,
    }


def assemble_coordinate_tiles_exact(tiles, expected_lon, expected_lat):
    """Assemble restart tiles by coordinate and reject unequal overlaps."""
    expected_lon = np.asarray(expected_lon)
    expected_lat = np.asarray(expected_lat)
    require(expected_lon.shape == expected_lat.shape,
            "expected coordinate shapes differ")
    keys = {
        (float(lon), float(lat)): at
        for at, lon, lat in zip(
            np.ndindex(expected_lon.shape), expected_lon.flat,
            expected_lat.flat, strict=True)
    }
    require(len(keys) == expected_lon.size, "expected coordinates are not unique")
    tiles = list(tiles)
    require(bool(tiles), "no coordinate tiles supplied")
    dtype = np.asarray(tiles[0][2]).dtype
    assembled = np.empty(expected_lon.shape, dtype=dtype)
    filled = np.zeros(expected_lon.shape, dtype=bool)
    for lon, lat, values in tiles:
        lon, lat, values = np.asarray(lon), np.asarray(lat), np.asarray(values)
        require(lon.shape == lat.shape == values.shape,
                "rank tile coordinate/value shapes differ")
        require(values.dtype == dtype, "rank tile dtypes differ")
        for key, value in zip(
                zip(lon.flat, lat.flat, strict=True), values.flat, strict=True):
            at = keys.get((float(key[0]), float(key[1])))
            if at is None:
                continue
            if filled[at]:
                require(bit_summary(
                    np.asarray([assembled[at]]), np.asarray([value]))["bit_identical"],
                        f"conflicting rank overlap at coordinate {key}")
            else:
                assembled[at] = value
                filled[at] = True
    require(np.all(filled),
            f"coordinate assembly missed {np.count_nonzero(~filled)} points")
    return assembled


def compare_rank_restarts(root: Path) -> dict:
    """Compare one-rank and 32-rank step-10 NEMO restarts word for word."""
    import netCDF4

    one_path = root / "np1/SWG_00000010_restart.nc"
    many_paths = sorted((root / "np32").glob("SWG_00000010_restart_*.nc"))
    require(one_path.is_file(), f"missing one-rank restart {one_path}")
    require(len(many_paths) == 32, "rank check requires exactly 32 restart tiles")
    with netCDF4.Dataset(one_path) as one:
        one.set_auto_mask(False)
        expected_variables = set(RESTART_STATE_FIELDS + RESTART_SCALARS + (
            "nav_lon", "nav_lat", "nav_lev", "time_counter"))
        require(set(one.variables) == expected_variables,
                "one-rank restart variable coverage changed")
        expected_lon = np.asarray(one["nav_lon"][:])
        expected_lat = np.asarray(one["nav_lat"][:])
        rows = {}
        datasets = [netCDF4.Dataset(path) for path in many_paths]
        try:
            for ds in datasets:
                ds.set_auto_mask(False)
                require(set(ds.variables) == expected_variables,
                        "32-rank restart variable coverage changed")
            rows["nav_lon"] = bit_summary(expected_lon, assemble_coordinate_tiles_exact(
                ((ds["nav_lon"][:], ds["nav_lat"][:], ds["nav_lon"][:])
                 for ds in datasets), expected_lon, expected_lat))
            rows["nav_lat"] = bit_summary(expected_lat, assemble_coordinate_tiles_exact(
                ((ds["nav_lon"][:], ds["nav_lat"][:], ds["nav_lat"][:])
                 for ds in datasets), expected_lon, expected_lat))
            for name in RESTART_STATE_FIELDS:
                reference = np.asarray(one[name][:])
                require(reference.shape[0] == 1, f"{name}: missing time record")
                reference = reference[0]
                if reference.ndim == 2:
                    candidate = assemble_coordinate_tiles_exact(
                        ((ds["nav_lon"][:], ds["nav_lat"][:], ds[name][0])
                         for ds in datasets), expected_lon, expected_lat)
                else:
                    require(reference.ndim == 3, f"{name}: unsupported rank")
                    levels = [assemble_coordinate_tiles_exact(
                        ((ds["nav_lon"][:], ds["nav_lat"][:], ds[name][0, level])
                         for ds in datasets), expected_lon, expected_lat)
                        for level in range(reference.shape[0])]
                    candidate = np.stack(levels)
                rows[name] = bit_summary(reference, candidate)
            for name in RESTART_SCALARS + ("nav_lev", "time_counter"):
                reference = np.asarray(one[name][:])
                for ds in datasets:
                    require(bit_summary(reference, np.asarray(ds[name][:]))["bit_identical"],
                            f"rank scalar/coordinate differs: {name}")
                rows[name] = bit_summary(reference, np.asarray(datasets[0][name][:]))
        finally:
            for ds in datasets:
                ds.close()
    return {
        "label": "SWG-DG-ROUND6-NEMO-RANK-CHECK",
        "root": str(root.resolve()),
        "one_rank_sha256": hashlib.sha256(one_path.read_bytes()).hexdigest(),
        "rank_files": len(many_paths),
        "rank_file_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in many_paths},
        "rows": rows,
        "all_bit_identical": all(row["bit_identical"] for row in rows.values()),
        "compared_variables": sorted(rows),
    }


def _write_json(path: Path, payload: dict) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--deck", choices=("shipped", "maddison"),
                        default="maddison")
    replay.add_argument("--stage-root", type=Path)
    replay.add_argument("--npz", type=Path, required=True)
    replay.add_argument("--json", type=Path, required=True)
    replay.add_argument("--plant", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("--json", type=Path, required=True)
    rank = sub.add_parser("rank-check")
    rank.add_argument("--root", type=Path, default=RANKCHECK_ROOT)
    rank.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "replay":
            root = args.stage_root or (
                DG_STAGE_ROOT if args.deck == "maddison" else SHIPPED_STAGE_ROOT)
            payload = run_replay(root, args.deck, args.npz, plant=args.plant)
        elif args.command == "compare":
            payload = compare_replays(args.left, args.right)
        else:
            payload = compare_rank_restarts(args.root)
        _write_json(args.json, payload)
    except Exception as exc:  # noqa: BLE001 - fail-closed measurement CLI
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(payload["label"])
    if args.command == "replay":
        ssh = payload["field_rows"]["ssh_k"]
        print(f"backend={payload['backend']} ssh_bit_unequal={ssh['bit_unequal']} "
              f"ssh_max_abs={ssh['max_abs']:.17e}")
    elif args.command == "compare":
        first = payload["first_probe_difference"]
        print("first_probe_difference="
              + ("NONE" if first is None else first["statement"]))
    else:
        print(f"all_bit_identical={payload['all_bit_identical']} "
              f"variables={len(payload['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
