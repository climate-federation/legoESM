#!/usr/bin/env python3
"""Capture the matched day-180 U/V public momentum-term diagnostics."""
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
import numpy as np

import kamm_twin_90d as twin
from legoesm.core.precision import PrecisionPolicy, set_policy


ROUND36_SHA = "f680200f2a3733558d1de7be7f97f5a80e575c003cbebf554fc1aa80e4428270"
COMPONENTS = (
    "KE_PGF", "vortcor", "Dterm", "vertadv", "Ah_lap", "Bh_bilap",
    "Cs_smag", "Cl_leith", "botdrag", "Av_vert", "phys",
    "surface_stress", "sponge",
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


def _data(diag, name: str, component: str, like: np.ndarray) -> np.ndarray:
    field = getattr(diag, f"{name}_{component}", None)
    if field is None:
        return np.zeros_like(like)
    return np.asarray(field.data, dtype=np.float64)


def _write(path: Path, value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value, dtype="<f8")
    path.write_bytes(array.tobytes(order="C"))
    return {
        "dtype": "<f8",
        "shape": list(array.shape),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--restart-file", default="DINO_00005760_restart.nc")
    parser.add_argument("--round36", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    set_policy(PrecisionPolicy.fp64())

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
        raise SystemExit("round 36 does not open the no-wind RHS term peel")
    output = args.output_dir.resolve()
    if not output.is_dir() or any(output.iterdir()):
        raise SystemExit("output directory must already exist and be empty")

    bridge, cfg, model_cfg, model, forcing, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()),
        str(args.run_stepdump.resolve()), bridge_before=True,
        restart_file=args.restart_file,
    )
    if any(getattr(state, name) is None for name in
           ("T_before", "S_before", "u_before", "v_before")):
        raise SystemExit("matched MLF BEFORE state is incomplete")
    ldf_state = (state.T_before.data, state.S_before.data,
                 state.u_before.data, state.v_before.data)
    with jax.disable_jit():
        tendencies, diag = model.tendencies_with_diagnostics(
            state, surface_forcing=sf, dt=twin.DT, ldf_state=ldf_state)

    files: dict[str, dict[str, object]] = {}
    closure = {}
    for component in ("u", "v"):
        total = _data(diag, "total", component,
                      np.asarray(getattr(tendencies, f"d{component}_dt").data,
                                 dtype=np.float64))
        ke_hpg = _data(diag, "KE_PGF", component, total)
        vertical = _data(diag, "vertadv", component, total)
        vorticity = _data(diag, "vortcor", component, total)
        surface_stress = _data(diag, "surface_stress", component, total)
        lateral = sum((_data(diag, name, component, total)
                       for name in ("Ah_lap", "Bh_bilap", "Cs_smag", "Cl_leith")),
                      np.zeros_like(total))
        mapped = ke_hpg + vertical + vorticity + lateral
        diagnostic_sum = sum((_data(diag, name, component, total)
                              for name in COMPONENTS), np.zeros_like(total))
        without_surface_stress = diagnostic_sum - surface_stress
        closure[component] = {
            "max_abs_diagnostic_sum_minus_total": float(
                np.max(np.abs(diagnostic_sum - total))),
            "max_abs_without_surface_stress_minus_total": float(
                np.max(np.abs(without_surface_stress - total))),
            "max_abs_surface_stress": float(np.max(np.abs(surface_stress))),
            "surface_stress_nonzero_count": int(np.count_nonzero(surface_stress)),
            "all_finite": bool(all(np.isfinite(value).all() for value in
                                    (total, ke_hpg, vertical, vorticity,
                                     lateral, surface_stress, mapped,
                                     diagnostic_sum))),
        }
        arrays = {
            "vertical_advection": vertical,
            "vorticity": vorticity,
            "lateral_friction": lateral,
            "ke_gradient_plus_hpg": ke_hpg,
            "mapped_d06": mapped,
            "surface_stress_outside_d03_d06": surface_stress,
            "diagnostic_total": total,
            "diagnostic_sum": diagnostic_sum,
        }
        for name, value in arrays.items():
            filename = f"{component}_{name}.bin"
            files[filename] = _write(output / filename, value)

    expected_shapes = {
        "u": [199, 53, 36],
        "v": [200, 52, 36],
    }
    for component, shape in expected_shapes.items():
        if files[f"{component}_mapped_d06.bin"]["shape"] != shape:
            raise SystemExit(
                f"{component} native stagger changed: "
                f"{files[f'{component}_mapped_d06.bin']['shape']} != {shape}")
    if not all(item["all_finite"] for item in closure.values()):
        raise SystemExit("nonfinite public momentum diagnostic")
    if any(item["max_abs_diagnostic_sum_minus_total"] > 1e-15
           for item in closure.values()):
        raise SystemExit(f"public diagnostic closure failed: {closure}")
    if (closure["u"]["surface_stress_nonzero_count"] == 0
            or closure["v"]["surface_stress_nonzero_count"] != 0):
        raise SystemExit(
            f"DINO zonal-only surface-stress coverage changed: {closure}")

    bindings = {
        "round36": _sha(args.round36.resolve()),
        "entry_restart": _sha(
            args.run_traj.resolve() / args.restart_file),
        "mesh_mask": _sha(args.run_stepdump.resolve() / "mesh_mask.nc"),
        "capture": _sha(Path(__file__).resolve()),
        "preregistration": _sha(
            root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round37.md"),
        "model": _sha(
            root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
        "tendency_kernel": _sha(
            root / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py"),
    }
    metadata = {
        "schema": "dino-split-explicit-momentum-chain-round37-capture-v2",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "recipe": "nemo_dino_kamm_mlf",
        "restart_file": args.restart_file,
        "time_levels": {
            "advection_vorticity_hpg": "Nnn/NOW",
            "lateral_friction": "Nbb/BEFORE via ldf_state",
        },
        "native_shapes": expected_shapes,
        "closure": closure,
        "files": files,
        "bindings": bindings,
        "selectors": {
            "outer_integrator": str(cfg.outer_integrator),
            "vorticity_scheme": str(getattr(model_cfg, "vorticity_scheme", None)),
            "een_metric_weighting": str(getattr(
                model_cfg, "een_metric_weighting", None)),
            "lateral_viscosity_evaluation": str(
                getattr(model_cfg, "lateral_viscosity_evaluation", None)),
        },
    }
    metadata_path = output / "capture.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if _tracked(root):
        raise SystemExit("tracked worktree changed during capture")
    print(f"capture={output} metadata_sha256={_sha(metadata_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
