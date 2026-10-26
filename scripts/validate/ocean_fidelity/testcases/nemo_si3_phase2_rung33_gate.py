#!/usr/bin/env python3
"""Fail-closed partial gate for SI3 lane-3 rung 3.3.

This boundary covers geometry, cold entry, and the first completed full
dynamics-plus-Prather step.  The 485-frame sweep and restart carry remain
UNMEASURED and therefore force an overall UNMEASURED exit even if every
implemented row were at the immutable pointwise bar.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import cast

import jax
import netCDF4
import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

POINTWISE_BAR = 1.0e-15
PLANTED_STRESS_DIVERGENCE_WEIGHT = 0.5001
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d_rhg/final")
HERE = Path(__file__).resolve().parent
PHASE1 = HERE / "nemo_si3_oracle_gate.py"
RUNG32 = HERE / "nemo_si3_phase2_adv2d_gate.py"
MANIFEST = HERE / "manifests" / "ice_adv2d_rhg_l3.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle_gate = _load("nemo_si3_oracle_gate_rung33", PHASE1)
rung32_gate = _load("nemo_si3_phase2_adv2d_gate_rung33", RUNG32)
GateError = rung32_gate.GateError
require = rung32_gate.require
_score = rung32_gate._score


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def oracle_surface_temperature_c(root: Path) -> np.ndarray:
    with netCDF4.Dataset(root / "output.init_ice.nc") as dataset:
        value = np.asarray(dataset["sst"][0]).T
    require(value.shape == (99, 99), f"ICE_ADV2D_RHG sst shape {value.shape}")
    return cast(np.ndarray, value)


def _physical(value: np.ndarray, halo: int) -> np.ndarray:
    return cast(np.ndarray, np.asarray(value)[halo:-halo, halo:-halo, ...])


def _candidate_field(card, state, name: str) -> np.ndarray:
    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import ICE_ADV2D_TRACERS

    if name in ICE_ADV2D_TRACERS:
        value = state.contents[..., ICE_ADV2D_TRACERS.index(name)]
    elif name == "sv_i":
        value = state.bulk_salt_diagnostic
    elif name == "t_surface":
        value = state.t_surface
    else:
        value = {
            "u_ice": state.dynamics.u_ice_u,
            "v_ice": state.dynamics.v_ice_v,
            "stress1_i": state.dynamics.stress1_t,
            "stress2_i": state.dynamics.stress2_t,
            "stress12_i": state.dynamics.stress12_f,
        }[name]
    return _physical(np.asarray(value), card.base.halo_width)


def _oracle_field(frame: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name == "t_surface":
        return cast(np.ndarray, oracle_gate._interior(frame["t_su"])[..., 0])
    if name == "sv_i":
        return cast(np.ndarray, oracle_gate._interior(frame["sv_i"])[..., 0])
    if name.startswith(("e_s_l", "e_i_l")):
        family = name[:3]
        layer = int(name.rsplit("_l", 1)[1]) - 1
        return cast(np.ndarray, oracle_gate._interior(frame[family])[..., layer, 0])
    if name.startswith("szv_i_l"):
        layer = int(name.rsplit("_l", 1)[1]) - 1
        return cast(np.ndarray, oracle_gate._interior(frame["szv_i"])[..., layer, 0])
    value = oracle_gate._interior(frame[name])
    return cast(np.ndarray, value[..., 0] if value.ndim == 3 else value)


def _row_names() -> tuple[str, ...]:
    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import ICE_ADV2D_TRACERS

    fields = (
        *ICE_ADV2D_TRACERS,
        "sv_i",
        "t_surface",
        "u_ice",
        "v_ice",
        "stress1_i",
        "stress2_i",
        "stress12_i",
    )
    return tuple(
        f"{boundary}.{name}"
        for boundary in ("entry.kt00000001", "trajectory.post_step_00000001")
        for name in fields
    )


def _validate_registry(rows: list[dict]) -> None:
    geometry = {
        f"geometry.{name}"
        for name in (
            "glamt", "glamu", "glamv", "glamf",
            "gphit", "gphiu", "gphiv", "gphif",
            "e1t", "e1u", "e1v", "e1f",
            "e2t", "e2u", "e2v", "e2f",
            "ff_t", "ff_f", "tmask", "umask", "vmask", "fmask",
        )
    }
    expected = geometry | set(_row_names())
    actual_names = [str(row["name"]) for row in rows]
    require(len(actual_names) == len(set(actual_names)), "duplicate rung-3.3 row")
    actual = set(actual_names)
    require(
        actual == expected,
        "rung-3.3 tripwire registry mismatch: "
        f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
    )


def _status_from_rows(rows: list[dict], unmeasured: dict | None = None) -> str:
    """Derive the only allowed overall verdict from measurements/coverage."""

    if any(row["status"] == "DEBT" for row in rows):
        return "DEBT"
    if unmeasured:
        return "UNMEASURED"
    return "AT-BAR"


def _binding_stress_control(card, oracle_frame) -> dict[str, object]:
    """Perturb the production solver and require a scored trajectory row to fail."""

    from legoesm.ice.fidelity.nemo_adv2d_rhg_testcase_recipe import (
        step_ice_adv2d_rhg_card,
    )

    planted = step_ice_adv2d_rhg_card(
        card,
        completed_steps=0,
        stress_divergence_outer_weight=PLANTED_STRESS_DIVERGENCE_WEIGHT,
    )
    rows: list[dict] = []
    dtypes: dict[str, dict[str, str]] = {}
    row = _score(
        rows,
        dtypes,
        "plant.trajectory.post_step_00000001.u_ice",
        _oracle_field(oracle_frame, "u_ice"),
        _candidate_field(card, planted, "u_ice"),
    )
    require(row["status"] == "DEBT", "internal stress-divergence plant did not score red")
    return {"status": "RED_AS_REQUIRED", "scored_row": row}


def run_gate(root: Path = ROOT, *, plant_geometry: bool = False) -> tuple[dict, int]:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ice.fidelity.nemo_adv2d_rhg_testcase_recipe import (
        build_ice_adv2d_rhg_card,
        step_ice_adv2d_rhg_card,
    )

    set_policy(PrecisionPolicy.fp64())
    _, first = oracle_gate.read_frame(root / "oracle_ice_step_entry_kt00000001.bin")
    _, second = oracle_gate.read_frame(root / "oracle_ice_step_entry_kt00000002.bin")
    card = build_ice_adv2d_rhg_card(oracle_surface_temperature_c(root), first)
    manifest = json.loads(MANIFEST.read_text())
    coverage = oracle_gate.check_manifest(root, "3.3", manifest)
    rows: list[dict] = []
    dtypes: dict[str, str] = {}
    rung32_gate.geometry_gate(root, card.base, rows, dtypes, plant=plant_geometry)
    completed = jax.jit(
        lambda state: step_ice_adv2d_rhg_card(card, state, completed_steps=0)
    )(card.initial_state)
    jax.block_until_ready(completed)
    from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import ICE_ADV2D_TRACERS

    fields = (
        *ICE_ADV2D_TRACERS,
        "sv_i", "t_surface", "u_ice", "v_ice",
        "stress1_i", "stress2_i", "stress12_i",
    )
    for boundary, frame, state in (
        ("entry.kt00000001", first, card.initial_state),
        ("trajectory.post_step_00000001", second, completed),
    ):
        for name in fields:
            _score(
                rows,
                dtypes,
                f"{boundary}.{name}",
                _oracle_field(frame, name),
                _candidate_field(card, state, name),
            )
    _validate_registry(rows)
    numeric_status = _status_from_rows(rows)
    debt = [row for row in rows if row["status"] == "DEBT"]
    unmeasured = {
        "trajectory_frames_3_to_485": "UNMEASURED: partial gate delegates to full gate",
        "restart_carry": "UNMEASURED: partial gate delegates to full gate",
        "first_divergence_owner": "UNMEASURED: partial gate delegates to replay gate",
    }
    overall_status = _status_from_rows(rows, unmeasured)
    report = {
        "format": "nemo-si3-phase2-rung33-partial-v1",
        "worktree": worktree_stamp(),
        "status": overall_status,
        "numeric_status": numeric_status,
        "bar": POINTWISE_BAR,
        "execution_path": "JIT (CPU, fp64, scalar-libm)",
        "root": str(root),
        "oracle_hashes": {
            "mesh_mask.nc": _sha256(root / "mesh_mask.nc"),
            "frame_kt1": _sha256(root / "oracle_ice_step_entry_kt00000001.bin"),
            "frame_kt2": _sha256(root / "oracle_ice_step_entry_kt00000002.bin"),
        },
        "coverage": coverage,
        "dtypes": dtypes,
        "rows": rows,
        "debt": debt,
        "binding_control": _binding_stress_control(card, second),
        "unmeasured": unmeasured,
    }
    return report, 0 if overall_status == "AT-BAR" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--plant-geometry", action="store_true")
    args = parser.parse_args()
    report, code = run_gate(args.root, plant_geometry=args.plant_geometry)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.artifact:
        args.artifact.write_text(payload + "\n")
    print(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
