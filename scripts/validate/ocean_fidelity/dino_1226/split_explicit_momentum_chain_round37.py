#!/usr/bin/env python3
"""Score the scoped U/V D03--D06 momentum-RHS term ladder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import netCDF4
import numpy as np

import split_explicit_momentum_chain_round35 as r35
from zu_frc_term_walk import _load_full_3d


ROUND36_SHA = "f680200f2a3733558d1de7be7f97f5a80e575c003cbebf554fc1aa80e4428270"
ORDERED_EXACT = (
    "vertical_advection",
    "vorticity_coriolis",
    "lateral_friction",
    "ke_gradient_plus_hpg",
    "d06_total",
)
SOURCE_LINES = {
    "surface_stress_outside_d03_d06": [
        "dynzdf.F90:353-363",
        "ocean_pe_latlon_cgrid.py:3646-3711",
    ],
    "vertical_advection": [
        "stpmlf.F90:309-314",
        "dynadv.F90:97-103",
    ],
    "vorticity_coriolis": [
        "stpmlf.F90:315-318",
        "dynvor.F90:143-179",
    ],
    "lateral_friction": [
        "stpmlf.F90:319-322",
        "dynldf.F90:79-115",
    ],
    "ke_gradient_plus_hpg": [
        "dynadv.F90:89-96",
        "dynhpg.F90:117-133",
        "stpmlf.F90:309-328",
    ],
    "d06_total": [
        "stpmlf.F90:269-270",
        "stpmlf.F90:309-328",
    ],
}
KT = 5761
STREAMS = tuple(
    f"stp_dump_0{stage}_{name}_kt{KT:08d}_{component}.bin"
    for stage, name in ((3, "dynadv"), (4, "dynvor"),
                        (5, "dynldf"), (6, "dynhpg"))
    for component in ("du", "dv")
) + tuple(
    f"{name}_dump_d{component}.bin"
    for name in ("keg", "zad", "vor", "ldf", "hpg")
    for component in ("u", "v")
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked(root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True).strip()


def _load_capture(directory: Path, metadata: dict, component: str,
                  name: str) -> np.ndarray:
    filename = f"{component}_{name}.bin"
    info = metadata["files"][filename]
    path = directory / filename
    if _sha(path) != info["sha256"] or path.stat().st_size != info["size_bytes"]:
        raise SystemExit(f"capture file admission failed: {filename}")
    return np.fromfile(path, dtype="<f8").reshape(info["shape"])


def _score(candidate, oracle, mask):
    return r35._metric(candidate, oracle, mask,
                       "dyn_zdf (momentum implicit vertical solve)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--round36", type=Path, required=True)
    parser.add_argument("--bracket", type=Path, required=True)
    parser.add_argument("--bracket-sha", required=True)
    parser.add_argument("--manifest-artifact", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if _tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    if _sha(args.round36) != ROUND36_SHA:
        raise SystemExit("official round-36 receipt changed")
    round36 = json.loads(args.round36.read_text())
    if round36.get("disposition") != "WIND_PLACEMENT_REFUTED":
        raise SystemExit("round 36 does not open this ladder")
    if _sha(args.bracket) != args.bracket_sha:
        raise SystemExit("round-37 bracket SHA slot disagrees")
    bracket = json.loads(args.bracket.read_text())
    if (bracket.get("disposition") != "CAPTURE_BRACKET_EXACT"
            or not bracket.get("shared_exact")):
        raise SystemExit("duplicate capture bracket is not exact")

    capture = args.capture.resolve()
    metadata_path = capture / "capture.json"
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("schema")
            != "dino-split-explicit-momentum-chain-round37-capture-v2"
            or metadata.get("session_id") != session):
        raise SystemExit("capture metadata/session mismatch")
    if str(capture) not in (bracket["capture_a"], bracket["capture_b"]):
        raise SystemExit("capture was not admitted by the bracket")
    expected_shapes = {"u": [199, 53, 36], "v": [200, 52, 36]}
    if metadata.get("native_shapes") != expected_shapes:
        raise SystemExit("capture native full-stagger shapes changed")
    if any(not item["all_finite"] for item in metadata["closure"].values()):
        raise SystemExit("capture contains nonfinite diagnostic values")
    if any(item["max_abs_diagnostic_sum_minus_total"] > 1e-15
           for item in metadata["closure"].values()):
        raise SystemExit("public diagnostic closure is outside pointwise bar")

    run = args.run_stepdump.resolve()
    manifest_artifact = json.loads(args.manifest_artifact.read_text())
    manifest = manifest_artifact["bracket"]["off_stream_manifest"]
    for name in STREAMS:
        path = run / name
        expected = manifest.get(name)
        if not path.is_file() or expected is None:
            raise SystemExit(f"retained manifest does not admit {name}")
        if _sha(path) != expected["sha256"]:
            raise SystemExit(f"retained stream changed: {name}")

    jpi, jpj, jpk, hls = 56, 203, 36, 2
    load3 = lambda name: _load_full_3d(
        str(run / name), jpi, jpj, jpk - 1, hls)
    with netCDF4.Dataset(run / "mesh_mask.nc") as dataset:
        umask = np.moveaxis(np.asarray(dataset["umask"][0]), 0, -1)[..., :35] > 0.5
        vmask = np.moveaxis(np.asarray(dataset["vmask"][0]), 0, -1)[..., :35] > 0.5
    if int(umask.sum()) != 336_338 or int(vmask.sum()) != 340_271:
        raise SystemExit(
            f"wet populations changed: U={umask.sum()} V={vmask.sum()}")

    nemo = {}
    dedicated = {}
    for component, suffix in (("u", "du"), ("v", "dv")):
        d03 = load3(f"stp_dump_03_dynadv_kt{KT:08d}_{suffix}.bin")
        d04 = load3(f"stp_dump_04_dynvor_kt{KT:08d}_{suffix}.bin")
        d05 = load3(f"stp_dump_05_dynldf_kt{KT:08d}_{suffix}.bin")
        d06 = load3(f"stp_dump_06_dynhpg_kt{KT:08d}_{suffix}.bin")
        keg = load3(f"keg_dump_d{component}.bin")
        zad = load3(f"zad_dump_d{component}.bin")
        vor = load3(f"vor_dump_d{component}.bin")
        ldf = load3(f"ldf_dump_d{component}.bin")
        hpg = load3(f"hpg_dump_d{component}.bin")
        nemo[component] = {
            "vertical_advection": zad,
            "vorticity_coriolis": d04 - d03,
            "lateral_friction": d05 - d04,
            "ke_gradient_plus_hpg": keg + hpg,
            "d06_total": d06,
            "dynadv_partial": d03,
            "dynhpg_partial": d06 - d05,
        }
        dedicated[component] = {
            "d03": (d03, keg + zad),
            "vorticity": (d04 - d03, vor),
            "lateral": (d05 - d04, ldf),
            "hpg": (d06 - d05, hpg),
            "d06": (d06, keg + zad + vor + ldf + hpg),
        }

    def crop(component: str, value: np.ndarray) -> np.ndarray:
        if component == "u":
            return np.asarray(value)[:, 1:, :35]
        return np.asarray(value)[1:, :, :35]

    twin = {component: {
        "vertical_advection": crop(component, _load_capture(
            capture, metadata, component, "vertical_advection")),
        "vorticity_coriolis": crop(component, _load_capture(
            capture, metadata, component, "vorticity")),
        "lateral_friction": crop(component, _load_capture(
            capture, metadata, component, "lateral_friction")),
        "ke_gradient_plus_hpg": crop(component, _load_capture(
            capture, metadata, component, "ke_gradient_plus_hpg")),
        "d06_total": crop(component, _load_capture(
            capture, metadata, component, "mapped_d06")),
        "dynadv_partial": crop(component, _load_capture(
            capture, metadata, component, "vertical_advection")),
        "dynhpg_partial": crop(component, _load_capture(
            capture, metadata, component, "ke_gradient_plus_hpg")),
    } for component in ("u", "v")}

    rows = {}
    first_failure = None
    for name in ORDERED_EXACT:
        rows[name] = {}
        for component, mask in (("u", umask), ("v", vmask)):
            rows[name][component] = _score(
                twin[component][name], nemo[component][name], mask)
        if first_failure is None and any(
                item["gate_status"] != "AT BAR" for item in rows[name].values()):
            first_failure = name
    partial_rows = {
        name: {
            component: _score(twin[component][name], nemo[component][name], mask)
            for component, mask in (("u", umask), ("v", vmask))
        }
        for name in ("dynadv_partial", "dynhpg_partial")
    }
    cumulative_controls = {
        name: {
            component: _score(pair[0], pair[1], mask)
            for component, mask, pair in (
                ("u", umask, dedicated["u"][name]),
                ("v", vmask, dedicated["v"][name]),
            )
        }
        for name in ("d03", "vorticity", "lateral", "hpg", "d06")
    }
    cumulative_at_bar = all(
        item["gate_status"] == "AT BAR"
        for row in cumulative_controls.values() for item in row.values())

    closure_controls = {}
    coverage_rows = {}
    for component, mask in (("u", umask), ("v", vmask)):
        diagnostic_total = crop(component, _load_capture(
            capture, metadata, component, "diagnostic_total"))
        diagnostic_sum = crop(component, _load_capture(
            capture, metadata, component, "diagnostic_sum"))
        closure_controls[f"public_closure_{component}"] = (
            _score(diagnostic_sum, diagnostic_total, mask)["gate_status"]
            == "AT BAR")
        mapped_sum = sum((twin[component][name] for name in
                          ORDERED_EXACT[:-1]), np.zeros_like(diagnostic_total))
        closure_controls[f"mapped_sum_{component}"] = (
            _score(mapped_sum, twin[component]["d06_total"], mask)["gate_status"]
            == "AT BAR")
        stress = crop(component, _load_capture(
            capture, metadata, component,
            "surface_stress_outside_d03_d06"))
        coverage_rows[component] = {
            "max_abs_tendency": float(np.max(np.abs(stress[mask]))),
            "nonzero_count": int(np.count_nonzero(stress[mask])),
            "nemo_application": "dynzdf.F90:353-363 (after D06/dyn_spg)",
            "d03_d06_owner": False,
            "round36_disposition": round36["disposition"],
        }
    closure_controls["surface_stress_u_present"] = (
        coverage_rows["u"]["nonzero_count"] > 0)
    closure_controls["surface_stress_v_structural_zero"] = (
        coverage_rows["v"]["nonzero_count"] == 0)

    plant_controls = {}
    for component, mask in (("u", umask), ("v", vmask)):
        oracle = nemo[component]["d06_total"]
        point = tuple(np.argwhere(mask)[0])
        plant = np.array(oracle, copy=True)
        plant[point] += 2e-12 * np.sqrt(np.mean(oracle[mask] ** 2))
        plant_controls[f"two_bar_{component}"] = (
            _score(plant, oracle, mask)["gate_status"] != "AT BAR")
        plant_controls[f"roll_{component}"] = (
            _score(np.roll(oracle, 1, axis=1), oracle, mask)["gate_status"]
            != "AT BAR")
        plant_controls[f"sign_{component}"] = (
            _score(-oracle, oracle, mask)["gate_status"] != "AT BAR")
        nan_plant = np.array(oracle, copy=True)
        nan_plant[point] = np.nan
        plant_controls[f"wet_nan_{component}"] = not np.isfinite(
            nan_plant[mask]).all()
    controls = {
        "cumulative_increment_identities_at_bar": cumulative_at_bar,
        **closure_controls,
        **plant_controls,
    }
    controls = {name: bool(value) for name, value in controls.items()}
    valid = all(controls.values())
    if not valid:
        disposition = "INVALID"
    elif first_failure is not None:
        disposition = f"ROW4_RHS_LOCALIZED_TO_{first_failure.upper()}"
    else:
        disposition = "ROW4_RHS_LOCALIZED_TO_POST_D06_COMPOSITION"

    sources = {
        "stpmlf": args.nemo_root / "cfgs/DINO/MY_SRC/stpmlf.F90",
        "dynadv": args.nemo_root / "cfgs/DINO/MY_SRC/dynadv.F90",
        "dynvor": args.nemo_root / "cfgs/DINO/MY_SRC/dynvor.F90",
        "dynldf": args.nemo_root / "cfgs/DINO/MY_SRC/dynldf.F90",
        "dynhpg": args.nemo_root / "cfgs/DINO/MY_SRC/dynhpg.F90",
        "dynzdf": args.nemo_root / "cfgs/DINO/MY_SRC/dynzdf.F90",
    }
    paths = {
        "round36": args.round36,
        "bracket": args.bracket,
        "manifest_artifact": args.manifest_artifact,
        "capture_metadata": metadata_path,
        "mesh_mask": run / "mesh_mask.nc",
        "scorer": Path(__file__).resolve(),
        "preregistration": root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round37.md",
        "diagnostics_model": root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py",
        "diagnostics_kernel": root / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py",
        **sources,
    }
    for name in STREAMS:
        paths[f"stream:{name}"] = run / name
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round37-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "ordered_exact_rows": list(ORDERED_EXACT),
        "nemo_source_lines_by_row": SOURCE_LINES,
        "coverage_row_surface_stress_outside_d03_d06": coverage_rows,
        "first_failing_exact_term": first_failure,
        "rows": rows,
        "partial_rows_nonowning": partial_rows,
        "nemo_cumulative_increment_controls": cumulative_controls,
        "controls": controls,
        "bindings": {name: _sha(path.resolve()) for name, path in paths.items()},
        "disposition": disposition,
        "ordered_next": 4,
    }
    if _tracked(root):
        raise SystemExit("tracked worktree changed during score")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition} first_failing_exact_term={first_failure}")
    for name, row in rows.items():
        print(name, {component: item["gate_status"]
                     for component, item in row.items()})
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
