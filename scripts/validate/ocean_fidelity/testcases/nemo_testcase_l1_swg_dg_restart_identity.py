#!/usr/bin/env python3
"""Compare legoESM SWG_DG daily endpoints with tiled NEMO restarts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nemo_testcase_checkout import bootstrap_checkout  # noqa: E402

CHECKOUT_IMPORT = bootstrap_checkout(__file__, enforce_x64=__name__ == "__main__")
from nemo_testcase_l1_swg_dg_climate import (  # noqa: E402
    assemble_coordinate_tiles,
    require_legoesm_deck,
    require_nemo_deck,
)
from nemo_testcase_l1_swg_dg_round6 import bit_summary  # noqa: E402
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

FIELD_MAPPING = {
    "ssh_m": {"nemo": "sshn", "grid": "T", "nemo_index": "[0]"},
    "u_m_s": {"nemo": "un", "grid": "U", "nemo_index": "[0, 0]"},
    "v_m_s": {"nemo": "vn", "grid": "V", "nemo_index": "[0, 0]"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restart_paths(root: Path, step: int) -> list[Path]:
    tiled = sorted(root.glob(f"SWG_{step:08d}_restart_[0-9][0-9][0-9][0-9].nc"))
    single = root / f"SWG_{step:08d}_restart.nc"
    require(not (tiled and single.exists()), "mixed tiled and single restart files")
    paths = tiled or ([single] if single.is_file() else [])
    require(bool(paths), f"no NEMO restart files for step {step} under {root}")
    return paths


def _mesh_paths(root: Path) -> list[Path]:
    tiled = sorted(root.glob("mesh_mask_[0-9][0-9][0-9][0-9].nc"))
    single = root / "mesh_mask.nc"
    require(not (tiled and single.exists()), "mixed tiled and single mesh files")
    paths = tiled or ([single] if single.is_file() else [])
    require(bool(paths), f"no NEMO mesh-mask files under {root}")
    return paths


def _card_geometry() -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_MADDISON_NAMELIST,
        build_swg_card,
        swg_horizontal_coordinates,
    )

    source = swg_horizontal_coordinates(SWG_MADDISON_NAMELIST)
    card = build_swg_card(SWG_MADDISON_NAMELIST)
    state = card.recipe.initial_state
    coordinates = {
        kind: (np.asarray(source["glamt"]), np.asarray(source["gphit"]))
        for kind in "TUV"
    }
    masks = {
        "T": np.asarray(card.recipe.land_mask, dtype=bool),
        "U": np.asarray(state.u_mask.data[:, 1:], dtype=bool),
        "V": np.asarray(state.v_mask.data[1:, :], dtype=bool),
    }
    return coordinates, masks


def load_nemo_restart(
        root: Path, step: int,
        coordinates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict]:
    """Assemble the completed-step SWE fields from one or more rank files."""
    import netCDF4

    paths = _restart_paths(root, step)
    fields: dict[str, np.ndarray] = {}
    excluded: dict[str, dict] = {}
    with ExitStack() as stack:
        datasets = [stack.enter_context(netCDF4.Dataset(path)) for path in paths]
        for ds in datasets:
            ds.set_auto_mask(False)
            for name in ("kt", "rdt", "sshn", "un", "vn", "uu_n", "vv_n"):
                require(name in ds.variables, f"restart tile is missing {name}")
            require(float(np.asarray(ds["kt"][:])) == float(step),
                    f"restart kt differs from requested step {step}")
            require(float(np.asarray(ds["rdt"][:])) == 50.0,
                    "restart rdt differs from the 50 s deck")
            for name in ("sshn", "un", "vn", "uu_n", "vv_n"):
                require(np.dtype(ds[name].dtype) == np.dtype("float64"),
                        f"{name} restart storage is not float64")
        for lego_name, mapping in FIELD_MAPPING.items():
            nemo_name = mapping["nemo"]
            tiles = []
            for ds in datasets:
                values = np.asarray(ds[nemo_name][:])
                if nemo_name == "sshn":
                    require(values.ndim == 3 and values.shape[0] == 1,
                            "sshn restart layout differs")
                    values = values[0]
                else:
                    require(values.ndim == 4 and values.shape[:2] == (1, 2),
                            f"{nemo_name} restart layout differs")
                    require(np.all(values[0, 1] == 0.0),
                            f"{nemo_name} dummy level is not zero")
                    values = values[0, 0]
                tiles.append((np.asarray(ds["nav_lon"][:]),
                              np.asarray(ds["nav_lat"][:]), values))
            fields[lego_name] = assemble_coordinate_tiles(
                tiles, *coordinates[mapping["grid"]], word_exact=True)
        for name in ("uu_n", "vv_n"):
            tiles = [
                (np.asarray(ds["nav_lon"][:]), np.asarray(ds["nav_lat"][:]),
                 np.asarray(ds[name][0]))
                for ds in datasets
            ]
            values = assemble_coordinate_tiles(
                tiles, *coordinates["T"], word_exact=True)
            excluded[name] = {
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(values).tobytes()).hexdigest(),
                "nonzero_words": int(np.count_nonzero(values)),
                "reason": (
                    "NEMO SWE restart barotropic workspace; not the completed "
                    "layer velocity. The compared velocities are un/vn level 0."),
            }
    provenance = {
        "root": str(root.resolve()),
        "step": step,
        "rank_files": len(paths),
        "restart_files_sha256": {path.name: sha256(path) for path in paths},
        "excluded_restart_fields": excluded,
    }
    return fields, provenance


def load_nemo_mesh_masks(
        root: Path,
        coordinates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict]:
    """Assemble and content-pin the wet masks used by the NEMO run."""
    import netCDF4

    paths = _mesh_paths(root)
    fields = {"T": "tmask", "U": "umask", "V": "vmask"}
    masks = {}
    with ExitStack() as stack:
        datasets = [stack.enter_context(netCDF4.Dataset(path)) for path in paths]
        for ds in datasets:
            ds.set_auto_mask(False)
            for name in (*fields.values(), "nav_lon", "nav_lat"):
                require(name in ds.variables, f"mesh tile is missing {name}")
        for kind, name in fields.items():
            tiles = []
            for ds in datasets:
                values = np.asarray(ds[name][:])
                require(values.ndim == 4 and values.shape[:2] == (1, 2),
                        f"{name} mesh layout differs")
                require(np.all(values[0, 1] == 0),
                        f"{name} dummy level is not dry")
                require(np.all((values[0, 0] == 0) | (values[0, 0] == 1)),
                        f"{name} contains a value other than zero or one")
                tiles.append((np.asarray(ds["nav_lon"][:]),
                              np.asarray(ds["nav_lat"][:]), values[0, 0]))
            masks[kind] = assemble_coordinate_tiles(
                tiles, *coordinates[kind], word_exact=True).astype(bool)
    provenance = {
        "rank_files": len(paths),
        "mesh_files_sha256": {path.name: sha256(path) for path in paths},
        "mask_sha256": {
            kind: hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
            for kind, mask in masks.items()
        },
    }
    return masks, provenance


def load_legoesm_endpoint(path: Path, *, day: int, step: int) -> tuple[dict, dict]:
    """Load one content-pinned end-of-day instantaneous legoESM state."""
    require(path.is_file(), f"missing legoESM endpoint {path}")
    manifest_path = path.parent / "manifest.json"
    require(manifest_path.is_file(), f"missing legoESM manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("status") == "complete", "legoESM manifest is incomplete")
    require(manifest.get("mode") == "gpu-native", "endpoint is not gpu-native")
    require(manifest.get("integration") == "one-jitted-scan-per-day",
            "endpoint did not use the one-JIT-per-day composition")
    require(manifest.get("steps_per_day") == 1_728,
            "legoESM daily cadence differs")
    require_legoesm_deck(manifest.get("namelist", {}))
    require(manifest.get("daily_sha256", {}).get(path.name) == sha256(path),
            "legoESM endpoint digest differs from its manifest")
    with np.load(path, allow_pickle=False) as saved:
        require(int(saved["day"]) == day, "embedded legoESM day differs")
        require(int(saved["step"]) == step, "embedded legoESM step differs")
        require(float(saved["dt_s"]) == 50.0, "embedded legoESM dt differs")
        fields = {}
        for name in FIELD_MAPPING:
            value = np.asarray(saved[name])
            require(value.dtype == np.dtype("float64"),
                    f"{name} endpoint storage is not float64")
            require(np.all(np.isfinite(value)),
                    f"{name} endpoint has non-finite values")
            fields[name] = value
    return fields, {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "legoesm_git_sha": manifest.get("legoesm_git_sha"),
        "mode": manifest.get("mode"),
        "integration": manifest.get("integration"),
        "backend": manifest.get("backend"),
        "transcendentals": manifest.get("transcendentals"),
    }


def compare_fields(
        nemo_fields: dict[str, np.ndarray], lego_fields: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Compare every declared wet word and retain exact/ULP classifications."""
    require(set(nemo_fields) == set(lego_fields) == set(FIELD_MAPPING),
            "endpoint field coverage differs")
    rows = {}
    for lego_name, mapping in FIELD_MAPPING.items():
        reference = np.asarray(nemo_fields[lego_name])
        candidate = np.asarray(lego_fields[lego_name])
        mask = np.asarray(masks[mapping["grid"]], dtype=bool)
        require(reference.shape == candidate.shape == mask.shape,
                f"{lego_name}: shape or mask differs")
        row = bit_summary(reference[mask], candidate[mask])
        row.update({
            "nemo_field": mapping["nemo"],
            "nemo_index": mapping["nemo_index"],
            "legoesm_field": lego_name,
            "grid": mapping["grid"],
            "wet_words": int(np.count_nonzero(mask)),
            "mask_sha256": hashlib.sha256(
                np.ascontiguousarray(mask).tobytes()).hexdigest(),
        })
        rows[lego_name] = row
    return rows


def build_report(
        *, day: int, step: int, rows: dict[str, dict],
        nemo_provenance: dict, lego_provenance: dict,
) -> dict:
    unequal = sum(row["bit_unequal"] for row in rows.values())
    max_abs = max(row["max_abs"] for row in rows.values())
    all_exact = all(row["bit_identical"] for row in rows.values())
    return {
        "schema": "swg-dg-yearly-restart-identity-v1",
        "label": "NEMO-versus-legoESM completed-step restart identity",
        "status": "PASS" if all_exact else "DEBT",
        "day": day,
        "step": step,
        "capture_instant": {
            "NEMO": (
                "restart kt=N after RK3 stage 3 completes step N; sshn and "
                "un/vn level 0 are the completed SWE layer state"),
            "legoESM": (
                "dayNNNN.npz end-of-day snapshot after exactly N production "
                "steps in the one-jitted-scan-per-day composition"),
            "excluded": (
                "NEMO uu_n/vv_n are barotropic workspace arrays and are not "
                "used as evidence for layer-velocity identity"),
        },
        "field_mapping": FIELD_MAPPING,
        "rows": rows,
        "summary": {
            "fields": len(rows),
            "wet_words": sum(row["wet_words"] for row in rows.values()),
            "bit_unequal": unequal,
            "max_abs": max_abs,
            "all_bit_identical": all_exact,
            "prognostic_signed_zero_words": sum(
                row["signed_zero_only"] for row in rows.values()),
        },
        "provenance": {"NEMO": nemo_provenance, "legoESM": lego_provenance},
    }


def run(nemo_root: Path, lego_npz: Path, *, day: int, step: int) -> dict:
    deck_provenance = require_nemo_deck(nemo_root)
    coordinates, masks = _card_geometry()
    nemo_masks, mesh_provenance = load_nemo_mesh_masks(nemo_root, coordinates)
    for kind in "TUV":
        require(np.array_equal(nemo_masks[kind], masks[kind]),
                f"NEMO and legoESM {kind}-grid wet masks differ")
    nemo_fields, nemo_provenance = load_nemo_restart(
        nemo_root, step, coordinates)
    nemo_provenance["deck"] = deck_provenance
    nemo_provenance["mesh"] = mesh_provenance
    lego_fields, lego_provenance = load_legoesm_endpoint(
        lego_npz, day=day, step=step)
    rows = compare_fields(nemo_fields, lego_fields, masks)
    return build_report(
        day=day, step=step, rows=rows,
        nemo_provenance=nemo_provenance, lego_provenance=lego_provenance)


def _write_json(path: Path, payload: dict) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--legoesm-npz", type=Path, required=True)
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(args.day > 0 and args.step == args.day * 1_728,
                "step must equal day*1728 for the 50 s deck")
        payload = run(
            args.nemo_root, args.legoesm_npz, day=args.day, step=args.step)
        _write_json(args.json, payload)
    except Exception as exc:  # noqa: BLE001 - fail-closed measurement boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    summary = payload["summary"]
    print(
        f"SWG_DG_RESTART_IDENTITY {payload['status']} day={args.day} "
        f"step={args.step} fields={summary['fields']} "
        f"wet_words={summary['wet_words']} unequal={summary['bit_unequal']} "
        f"max_abs={summary['max_abs']:.17e}")
    return 0 if summary["all_bit_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
