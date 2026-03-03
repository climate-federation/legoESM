#!/usr/bin/env python
"""Numerical cubed-sphere edge continuity diagnostics.

Runs selected atmospheric dycore cases and quantifies face-edge jumps.

For each field at requested snapshot times, it computes:
1. Absolute edge jump across connected faces.
2. Jump normalized by local one-cell normal gradients on both sides.

The normalized metric is useful to distinguish true discontinuities from
expected gradients over non-collocated edge-adjacent cell centers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.halo import CONNECTIVITY, EAST, NORTH, SOUTH, WEST


def _edge_strip(arr: np.ndarray, face: int, edge: int) -> np.ndarray:
    if edge == WEST:
        return arr[face, 0, :]
    if edge == EAST:
        return arr[face, -1, :]
    if edge == SOUTH:
        return arr[face, :, 0]
    if edge == NORTH:
        return arr[face, :, -1]
    raise ValueError(f"Unknown edge: {edge}")


def _interior_strip(arr: np.ndarray, face: int, edge: int) -> np.ndarray:
    if edge == WEST:
        return arr[face, 1, :]
    if edge == EAST:
        return arr[face, -2, :]
    if edge == SOUTH:
        return arr[face, :, 1]
    if edge == NORTH:
        return arr[face, :, -2]
    raise ValueError(f"Unknown edge: {edge}")


def _continuity_stats(arr: np.ndarray) -> dict[str, float | str]:
    """Compute edge continuity statistics for a scalar face field."""
    arr = np.asarray(arr, dtype=np.float64)
    eps = 1.0e-12

    jumps = []
    rels = []
    worst = {"pair": "", "jump_max": -1.0, "rel_max": -1.0}

    for face in range(6):
        for edge, (nbr_face, nbr_edge, reversed_idx) in CONNECTIVITY[face].items():
            a0 = _edge_strip(arr, face, edge)
            a1 = _interior_strip(arr, face, edge)
            b0 = _edge_strip(arr, nbr_face, nbr_edge)
            b1 = _interior_strip(arr, nbr_face, nbr_edge)

            if reversed_idx:
                b0 = b0[::-1]
                b1 = b1[::-1]

            jump = np.abs(a0 - b0)
            scale = 0.5 * (np.abs(a0 - a1) + np.abs(b0 - b1)) + eps
            rel = jump / scale

            jumps.append(jump)
            rels.append(rel)

            jump_max = float(np.max(jump))
            rel_max = float(np.max(rel))
            if jump_max > worst["jump_max"]:
                worst["jump_max"] = jump_max
                worst["rel_max"] = rel_max
                worst["pair"] = f"{face}:{edge}->{nbr_face}:{nbr_edge},rev={reversed_idx}"

    jump_all = np.concatenate(jumps)
    rel_all = np.concatenate(rels)

    return {
        "jump_mean": float(np.mean(jump_all)),
        "jump_p95": float(np.percentile(jump_all, 95.0)),
        "jump_p99": float(np.percentile(jump_all, 99.0)),
        "jump_max": float(np.max(jump_all)),
        "rel_mean": float(np.mean(rel_all)),
        "rel_p95": float(np.percentile(rel_all, 95.0)),
        "rel_p99": float(np.percentile(rel_all, 99.0)),
        "rel_max": float(np.max(rel_all)),
        "frac_rel_gt3": float(np.mean(rel_all > 3.0)),
        "frac_rel_gt5": float(np.mean(rel_all > 5.0)),
        "worst_pair": str(worst["pair"]),
    }


def _run_sw2(resolution: int, solver: str) -> tuple[dict[int, dict[str, np.ndarray]], float]:
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water import ShallowWaterConfig, ShallowWaterModel
    from legoesm.atmosphere.dynamics.williamson import williamson_test2

    dt = 450.0
    n_steps = int(5 * 86400 / dt)
    targets = {0, n_steps // 2, n_steps}

    grid = create_cubed_sphere(resolution)
    config = ShallowWaterConfig(
        hyperdiff_coeff=5.0e16 * (48.0 / resolution) ** 4,
        time_integrator=solver,
    )
    model = ShallowWaterModel(grid, config)
    state = williamson_test2(grid)

    captured: dict[int, dict[str, np.ndarray]] = {}

    def _capture(step: int):
        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data)
        captured[step] = {
            "wind_speed": np.sqrt(u * u + v * v),
            "height": np.asarray(state.h.data),
        }

    _capture(0)
    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1
        if step in targets:
            _capture(step)

    jax.block_until_ready(state.h.data)
    return captured, dt


def _run_nh_tc2a(resolution: int, n_levels: int, solver: str) -> tuple[dict[int, dict[str, np.ndarray]], float]:
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.dcmip2025 import dcmip25_tc2_init
    from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerConfig, CompressibleEulerModel

    dt = 0.2
    n_steps = int(0.05 * 3600.0 / dt)  # 3 minutes
    targets = {0, n_steps // 2, n_steps}

    base_grid = create_cubed_sphere(resolution)
    state, height_coord, terrain_metric, small_grid = dcmip25_tc2_init(
        base_grid,
        n_levels=n_levels,
        subcase="a",
    )
    model = CompressibleEulerModel(
        small_grid,
        height_coord,
        terrain_metric,
        CompressibleEulerConfig(
            n_acoustic_substeps=6,
            sponge_width=15000.0,
            sponge_coeff=1.0 / (0.1 * 86400.0),
            small_earth_factor=20.0,
            outer_integrator=solver,
            edge_blend_uv=0.22 if resolution >= 24 else 0.0,
            edge_blend_w=0.22 if resolution >= 24 else 0.0,
            edge_blend_theta=0.12 if resolution >= 24 else 0.0,
            edge_blend_rho=0.26 if resolution >= 24 else 0.0,
            edge_blend_width=4 if resolution >= 24 else 1,
        ),
    )

    captured: dict[int, dict[str, np.ndarray]] = {}

    def _capture(step: int):
        u = np.asarray(state.u.data)[..., -1]
        v = np.asarray(state.v.data)[..., -1]
        w = np.asarray(state.w.data)
        captured[step] = {
            "wind_low": np.sqrt(u * u + v * v),
            "rho_prime": np.asarray(state.rho_prime.data)[..., -1],
            "w_mid": w[..., w.shape[-1] // 2],
        }

    _capture(0)
    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1
        if step in targets:
            _capture(step)

    jax.block_until_ready(state.u.data)
    return captured, dt


def _run_nh_tc1(resolution: int, n_levels: int, solver: str) -> tuple[dict[int, dict[str, np.ndarray]], float]:
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.dcmip2025 import dcmip25_tc1_init
    from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerConfig, CompressibleEulerModel

    dt = 1.0
    n_steps = int(1.0 * 3600.0 / dt)  # 1 hour
    targets = {0, n_steps // 2, n_steps}

    grid = create_cubed_sphere(resolution)
    state, height_coord, terrain_metric = dcmip25_tc1_init(
        grid,
        n_levels=n_levels,
    )
    model = CompressibleEulerModel(
        grid,
        height_coord,
        terrain_metric,
        CompressibleEulerConfig(
            n_acoustic_substeps=8,
            sponge_width=10000.0,
            sponge_coeff=0.06,
            outer_integrator=solver,
            edge_blend_uv=0.22 if resolution >= 24 else 0.0,
            edge_blend_w=0.14 if resolution >= 24 else 0.0,
            edge_blend_theta=0.12 if resolution >= 24 else 0.0,
            edge_blend_rho=0.18 if resolution >= 24 else 0.0,
            edge_blend_width=3 if resolution >= 24 else 1,
        ),
    )

    captured: dict[int, dict[str, np.ndarray]] = {}

    def _capture(step: int):
        u = np.asarray(state.u.data)[..., -1]
        v = np.asarray(state.v.data)[..., -1]
        w = np.asarray(state.w.data)
        captured[step] = {
            "wind_low": np.sqrt(u * u + v * v),
            "rho_prime": np.asarray(state.rho_prime.data)[..., -1],
            "w_mid": w[..., w.shape[-1] // 2],
        }

    _capture(0)
    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1
        if step in targets:
            _capture(step)

    jax.block_until_ready(state.u.data)
    return captured, dt


def _compute_case(case: str, snapshots: dict[int, dict[str, np.ndarray]]) -> dict:
    out = {"case": case, "steps": {}}
    for step in sorted(snapshots):
        fields = snapshots[step]
        out["steps"][str(step)] = {}
        for name, arr in fields.items():
            out["steps"][str(step)][name] = _continuity_stats(arr)
    return out


def _write_report(path: Path, payload: dict) -> None:
    with open(path / "edge_continuity_summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    lines = []
    lines.append("# Edge Continuity Diagnostics")
    lines.append("")
    lines.append(f"Generated: {payload['generated']}")
    lines.append("")
    lines.append(f"- Resolution: C{payload['resolution']}")
    lines.append(f"- Solver: {payload['solver']}")
    lines.append("")

    for case in payload["cases"]:
        lines.append(f"## {case['case']}")
        lines.append("")
        lines.append("| Step | Field | jump_p99 | jump_max | rel_p99 | rel_max | frac_rel_gt3 | frac_rel_gt5 |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for step, fields in case["steps"].items():
            for field_name, stats in fields.items():
                lines.append(
                    f"| {step} | {field_name} | "
                    f"{stats['jump_p99']:.6e} | {stats['jump_max']:.6e} | "
                    f"{stats['rel_p99']:.3f} | {stats['rel_max']:.3f} | "
                    f"{stats['frac_rel_gt3']:.3f} | {stats['frac_rel_gt5']:.3f} |"
                )
        lines.append("")

    with open(path / "edge_continuity_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Cubed-sphere edge continuity diagnostics")
    parser.add_argument("--resolution", type=int, default=36)
    parser.add_argument("--solver", type=str, default="ssp45")
    parser.add_argument("--nh-levels", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/edge_continuity_pass2"))
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["sw2", "nh_tc1", "nh_tc2a"],
        choices=["sw2", "nh_tc1", "nh_tc2a"],
        help="Cases to run",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    case_payloads = []

    if "sw2" in args.cases:
        print("Running SW Williamson 2 continuity diagnostics...")
        snaps, _ = _run_sw2(args.resolution, args.solver)
        case_payloads.append(_compute_case("SW Williamson 2", snaps))

    if "nh_tc2a" in args.cases:
        print("Running NH TC2a continuity diagnostics...")
        snaps, _ = _run_nh_tc2a(args.resolution, args.nh_levels, args.solver)
        case_payloads.append(_compute_case("NH TC2a", snaps))

    if "nh_tc1" in args.cases:
        print("Running NH TC1 continuity diagnostics...")
        snaps, _ = _run_nh_tc1(args.resolution, args.nh_levels, args.solver)
        case_payloads.append(_compute_case("NH TC1", snaps))

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resolution": args.resolution,
        "solver": args.solver,
        "wall_time_s": time.time() - t0,
        "cases": case_payloads,
    }
    _write_report(args.output, payload)
    print(f"Wrote diagnostics to {args.output}")


if __name__ == "__main__":
    main()
