#!/usr/bin/env python3
"""Short-window (10 walk steps, completed_step 9..18) timing and stage diagnosis for DECISION 177."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
from legoesm.core import transcendentals as tr
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ice.dynamics import si3_cgrid_aevp_solver, si3_cgrid_deformation
from legoesm.ice.fidelity import nemo_rheo_testcase_recipe as recipe
from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import (
    apply_si3_prather_source_corrections,
)
from legoesm.ice.transport import (
    advect_si3_prather_2d,
    si3_prather_pack_intensives,
    si3_prather_unpack_intensives,
)


def _instrument_libm():
    stats = {"calls": 0, "elements": 0, "wall_s": 0.0, "by_name": {}}
    orig = tr._scalar_libm

    def wrapped(name: str, values: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        out = orig(name, values)
        dt = time.perf_counter() - t0
        n = int(np.asarray(values).size)
        stats["calls"] += 1
        stats["elements"] += n
        stats["wall_s"] += dt
        entry = stats["by_name"].setdefault(name, {"calls": 0, "elements": 0, "wall_s": 0.0})
        entry["calls"] += 1
        entry["elements"] += n
        entry["wall_s"] += dt
        return out

    tr._scalar_libm = wrapped
    return stats


def _reset_libm(stats):
    stats["calls"] = 0
    stats["elements"] = 0
    stats["wall_s"] = 0.0
    stats["by_name"] = {}


def _snapshot_libm(stats):
    return {
        "calls": stats["calls"],
        "elements": stats["elements"],
        "wall_s": round(stats["wall_s"], 4),
        "by_name": {
            k: {
                "calls": v["calls"],
                "elements": v["elements"],
                "wall_s": round(v["wall_s"], 4),
            }
            for k, v in stats["by_name"].items()
        },
    }


def _trajectory_rows_sha256(step_records) -> str:
    """Hash only scored rows, excluding labels and non-deterministic timings."""
    rows = [row for record in step_records for row in record["rows"]]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _tree_leaf_sha256(tree) -> dict[str, str]:
    """Bit fingerprints for every carried-state leaf, with dtype and shape pinned."""

    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(leaf)
        digest = hashlib.sha256()
        digest.update(array.dtype.str.encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes(order="C"))
        out[jax.tree_util.keystr(path)] = digest.hexdigest()
    return out


def _stage_breakdown(card, state, ice_step_index: int, transport_step_index: int, libm_stats):
    """Time each stage of one compiled step (after warmup) on the given state."""

    @jax.jit
    def stage_rhg(st, idx):
        forcing = recipe._forcing_for_state(card, st, idx)
        dynamics = si3_cgrid_aevp_solver(st.dynamics, forcing, card.metrics, card.dynamics_config)
        div, delta = si3_cgrid_deformation(dynamics, forcing, card.metrics, card.dynamics_config)
        return dynamics, div, delta

    @jax.jit
    def stage_adv(st, dynamics):
        cell_area = jnp.full_like(dynamics.u_ice_u, recipe._ICE_RHEO_CELL_AREA_M2, dtype=jnp.float64)
        wet = card.forcing_template.tmask_t.astype(bool)
        entry_contents = si3_prather_pack_intensives(st.contents, cell_area)
        contents, moments, _ = advect_si3_prather_2d(
            entry_contents,
            st.moments,
            dynamics.u_ice_u,
            dynamics.v_ice_v,
            cell_area,
            wet,
            card.dt_s,
            dx=recipe._ICE_RHEO_SPACING_M,
            dy=recipe._ICE_RHEO_SPACING_M,
            ice_step_index=transport_step_index,
            nn_fsbc=card.nn_fsbc,
            halo_width=card.halo_width,
            ice_volume_index=recipe.ICE_RHEO_TRACERS.index("v_i"),
            concentration_index=recipe.ICE_RHEO_TRACERS.index("a_i"),
            subcycles=card.subcycles,
        )
        contents = si3_prather_unpack_intensives(contents, cell_area, wet)
        contents = apply_si3_prather_source_corrections(
            st.contents,
            contents,
            tracer_names=recipe.ICE_RHEO_TRACERS,
            nlay_i=card.nlay_i,
            nlay_s=card.nlay_s,
            cell_area_m2=recipe._ICE_RHEO_CELL_AREA_M2,
            halo_width=card.halo_width,
            entry_intensive_contents=st.contents,
            contents_are_intensive=True,
        )
        entry_area = recipe._intensive(st.contents, "a_i")
        transported_area = recipe._intensive(contents, "a_i")
        u_transport = dynamics.u_ice_u * card.metrics.e2u
        v_transport = dynamics.v_ice_v * card.metrics.e1v
        flux_divergence = (
            (u_transport - jnp.roll(u_transport, 1, axis=0))
            + (v_transport - jnp.roll(v_transport, 1, axis=1))
        ) / card.metrics.area_t
        open_water = jnp.maximum(
            recipe._ICE_RHEO_ZERO,
            st.open_water_area - (transported_area - entry_area) - flux_divergence * card.dt_s,
        )
        return contents, moments, open_water

    has_o6_ridge = hasattr(recipe, "ridge_ice_rheo")

    @jax.jit
    def stage_rdgrft(contents, open_water, div, delta):
        if has_o6_ridge:
            res = recipe.ridge_ice_rheo(card, contents, open_water, div, delta)
            return (
                recipe._set_category_fields(contents, res.state),
                res.n_passes,
                jnp.sum(res.selected.astype(jnp.int32)),
                jnp.max(res.n_shifts),
            )
        redistributed, losses = recipe.apply_si3_jpl1_ridging(
            recipe._ridging_state(contents, open_water),
            div,
            delta,
            card.dt_s,
            config=card.ridging_config,
        )
        return (
            recipe._contents_after_ridging(contents, redistributed),
            jnp.max(losses.iterations),
            jnp.sum((losses.iterations > 0).astype(jnp.int32)),
            jnp.max(losses.iterations),
        )

    @jax.jit
    def stage_prep_statements(contents, open_water):
        """Aligned first-pass participation statements for old jpl=1 and O6."""

        area = recipe._intensive(contents, "a_i")
        cfg = card.ridging_config
        if has_o6_ridge:
            sr = nemo_source_round
            total_area = sr(open_water + area)
            has_total = total_area > 1.0e-10
            safe_total = jnp.where(has_total, total_area, 1.0)
            inverse_total = jnp.where(has_total, sr(1.0 / safe_total), 0.0)
            g_open = sr(open_water * inverse_total)
            inverse_astar = 1.0 / cfg.rn_astar
            exp_constant = tr.exp(jnp.asarray(-inverse_astar, jnp.float64))
            exponential_scale = sr(1.0 / sr(1.0 - exp_constant))
            open_argument = -sr(g_open * inverse_astar)
            exp_open = tr.exp(open_argument)
            g_zero = sr(sr(exp_open) * exponential_scale)
            g_one = sr(sr(exp_constant) * exponential_scale)
            participation_open = sr(exponential_scale - g_zero)
            participation_ice = sr(g_zero - g_one)
        else:
            total_area = open_water + area
            safe_total = jnp.where(total_area > 1.0e-10, total_area, 1.0)
            inverse_total = jnp.where(total_area > 1.0e-10, 1.0 / safe_total, 0.0)
            g_open = open_water * inverse_total
            inverse_astar = 1.0 / cfg.astar
            exp_constant = jnp.exp(-inverse_astar)
            exponential_scale = 1.0 / (1.0 - exp_constant)
            open_argument = -g_open * inverse_astar
            exp_open = jnp.exp(open_argument)
            g_zero = exp_open * exponential_scale
            g_one = exp_constant * exponential_scale
            participation_open = exponential_scale - g_zero
            participation_ice = g_zero - g_one
        return (
            total_area,
            inverse_total,
            g_open,
            jnp.asarray(inverse_astar, jnp.float64),
            exp_constant,
            exponential_scale,
            open_argument,
            exp_open,
            g_zero,
            g_one,
            participation_open,
            participation_ice,
        )

    @jax.jit
    def stage_cor(contents, t_surface):
        return recipe._ice_cor(card, contents, t_surface)

    idx_arr = jnp.asarray(ice_step_index, dtype=jnp.int32)
    # Warmup compile + 1st run, then timed 2nd run
    dynamics, div, delta = jax.block_until_ready(stage_rhg(state, idx_arr))
    contents, moments, open_water = jax.block_until_ready(stage_adv(state, dynamics))
    prep_statements = jax.block_until_ready(stage_prep_statements(contents, open_water))
    contents_r, _, _, _ = jax.block_until_ready(stage_rdgrft(contents, open_water, div, delta))
    _ = jax.block_until_ready(stage_cor(contents_r, state.t_surface))

    out = {}
    _reset_libm(libm_stats)
    t0 = time.perf_counter()
    dynamics, div, delta = jax.block_until_ready(stage_rhg(state, idx_arr))
    out["rhg"] = {"wall_s": round(time.perf_counter() - t0, 4), "libm": _snapshot_libm(libm_stats)}
    out["rhg"]["output_leaf_sha256"] = _tree_leaf_sha256((dynamics, div, delta))

    _reset_libm(libm_stats)
    t0 = time.perf_counter()
    contents, moments, open_water = jax.block_until_ready(stage_adv(state, dynamics))
    out["adv"] = {"wall_s": round(time.perf_counter() - t0, 4), "libm": _snapshot_libm(libm_stats)}
    out["adv"]["output_leaf_sha256"] = _tree_leaf_sha256((contents, moments, open_water))
    prep_names = (
        "01_total_area",
        "02_inverse_total",
        "03_g_open",
        "04_inverse_astar",
        "05_exp_neg_inverse_astar",
        "06_exponential_scale",
        "07_neg_g_open_inverse_astar",
        "08_exp_neg_g_open_inverse_astar",
        "09_g_zero",
        "10_g_one",
        "11_participation_open",
        "12_participation_ice",
    )
    out["rdgrft_prep_statement_sha256"] = {
        name: _tree_leaf_sha256(value)[""]
        for name, value in zip(prep_names, prep_statements, strict=True)
    }

    _reset_libm(libm_stats)
    t0 = time.perf_counter()
    contents_r, n_passes, n_sel, max_shifts = jax.block_until_ready(
        stage_rdgrft(contents, open_water, div, delta)
    )
    out["rdgrft"] = {
        "wall_s": round(time.perf_counter() - t0, 4),
        "n_passes_active": int(n_passes),
        "selected_cells": int(n_sel),
        "max_cell_shifts": int(max_shifts),
        "libm": _snapshot_libm(libm_stats),
        "output_leaf_sha256": _tree_leaf_sha256(contents_r),
    }

    _reset_libm(libm_stats)
    t0 = time.perf_counter()
    corrected, corrected_surface = jax.block_until_ready(stage_cor(contents_r, state.t_surface))
    out["cor"] = {"wall_s": round(time.perf_counter() - t0, 4), "libm": _snapshot_libm(libm_stats)}
    out["cor"]["output_leaf_sha256"] = _tree_leaf_sha256(
        (corrected, corrected_surface)
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-walk-steps", type=int, default=10)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--trace-step", type=int)
    ap.add_argument("--skip-step11-breakdown", action="store_true")
    args = ap.parse_args()

    import importlib.util

    tgate_path = (
        args.tree
        / "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_rung34_trajectory_gate.py"
    )
    spec = importlib.util.spec_from_file_location("tgate", tgate_path)
    assert spec and spec.loader
    tgate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tgate)
    gate = tgate.gate
    root = tgate.ROOT
    restart_root = tgate._ACTIVE_RESTART_ROOT

    libm_stats = _instrument_libm()

    _, entry = gate.oracle_gate.read_frame(
        root / f"oracle_ice_step_entry_kt{tgate._ACTIVE_COMPLETED_STEP:08d}.bin"
    )
    with netCDF4.Dataset(root / "mesh_mask.nc") as dataset:
        mesh = {name: np.asarray(dataset[name][:]) for name in gate.MESH_FIELDS}
    with netCDF4.Dataset(root / "output.init_ice.nc") as dataset:
        sst_m = np.asarray(dataset["sst"][0])
    if not hasattr(recipe, "ridge_ice_rheo"):
        from legoesm import constants as C

        sst_in = sst_m + C.T_freeze
    else:
        sst_in = sst_m
    card = recipe.build_ice_rheo_card(entry, mesh, sst_in)
    entry_restart = restart_root / (
        f"ICE_RHEO_OMIP_L3_{tgate._ACTIVE_ENTRY_RESTART_STEP:08d}_restart_ice.nc"
    )
    state0 = card.initial_state._replace(moments=tgate._restart_moments(entry_restart, card))

    print(f"[{args.label}] running setup step 8 (eager)...", file=sys.stderr, flush=True)
    _reset_libm(libm_stats)
    t0 = time.perf_counter()
    state = jax.block_until_ready(
        recipe.step_ice_rheo_card(card, state0, completed_steps=tgate._ACTIVE_ENTRY_RESTART_STEP)
    )
    step8_state_leaf_sha256 = _tree_leaf_sha256(state)
    step8_eager_s = round(time.perf_counter() - t0, 4)
    step8_libm = _snapshot_libm(libm_stats)
    print(f"[{args.label}] step 8 eager done in {step8_eager_s}s", file=sys.stderr, flush=True)

    odd_step = jax.jit(
        lambda current, ice_index: recipe._step_ice_rheo_card_impl(
            card, current, ice_step_index=ice_index, transport_step_index=1
        )
    )
    even_step = jax.jit(
        lambda current, ice_index: recipe._step_ice_rheo_card_impl(
            card, current, ice_step_index=ice_index, transport_step_index=2
        )
    )

    print(f"[{args.label}] compiling odd and even steps...", file=sys.stderr, flush=True)
    compile_times = {}
    compiled_steps = {}
    for parity, stepper, step_index in (
        ("odd", odd_step, tgate._FULL_WALK_FIRST_STEP),
        ("even", even_step, tgate._FULL_WALK_FIRST_STEP + 1),
    ):
        t0 = time.perf_counter()
        compiled_steps[parity] = stepper.lower(
            state, jnp.asarray(step_index, dtype=jnp.int32)
        ).compile()
        compile_times[parity] = round(time.perf_counter() - t0, 4)
        print(
            f"[{args.label}] {parity} compile done in {compile_times[parity]}s",
            file=sys.stderr,
            flush=True,
        )

    step_records = []
    first_step = tgate._FULL_WALK_FIRST_STEP
    last_step = first_step + args.n_walk_steps - 1
    step11_state_in = None
    trace_state_in = None
    if args.trace_step is not None and not first_step < args.trace_step <= last_step + 1:
        raise ValueError("trace step must follow a state produced in this window")

    for completed_step in range(first_step, last_step + 1):
        parity = "odd" if completed_step % 2 else "even"
        stepper = compiled_steps[parity]
        if completed_step == 11:
            step11_state_in = state
        _reset_libm(libm_stats)
        t0 = time.perf_counter()
        compiled_state = jax.block_until_ready(
            stepper(state, jnp.asarray(completed_step, dtype=jnp.int32))
        )
        step_wall_s = round(time.perf_counter() - t0, 4)
        step_libm = _snapshot_libm(libm_stats)

        state = compiled_state
        if args.trace_step is not None and completed_step == args.trace_step - 1:
            trace_state_in = state
        frame_number = completed_step + 1
        _, oracle = gate.oracle_gate.read_frame(
            root / f"oracle_ice_step_entry_kt{frame_number:08d}.bin"
        )
        rows = tgate._ordinary_rows_from_frame(card, state, oracle, completed_step)
        rec = {
            "completed_step": completed_step,
            "compiled_step_wall_s": step_wall_s,
            "odd_cache_size": odd_step._cache_size(),
            "even_cache_size": even_step._cache_size(),
            "libm": step_libm,
            "rows": rows,
            "state_leaf_sha256": _tree_leaf_sha256(state),
        }
        step_records.append(rec)
        print(
            f"[{args.label}] step {completed_step}: compiled={step_wall_s}s "
            f"libm_calls={step_libm['calls']} "
            f"libm_s={step_libm['wall_s']}s cache=({odd_step._cache_size()},{even_step._cache_size()})",
            file=sys.stderr,
            flush=True,
        )

    stage_breakdown_step11 = None
    if not args.skip_step11_breakdown:
        print(f"[{args.label}] measuring stage breakdown at step 11...", file=sys.stderr, flush=True)
        stage_breakdown_step11 = _stage_breakdown(card, step11_state_in, 11, 1, libm_stats)
        print(
            f"[{args.label}] stage breakdown step 11: {json.dumps(stage_breakdown_step11)}",
            file=sys.stderr,
            flush=True,
        )

    trace_breakdown = None
    if args.trace_step is not None:
        if trace_state_in is None:
            raise ValueError("trace input state was not captured")
        transport_index = 1 if args.trace_step % 2 else 2
        print(
            f"[{args.label}] tracing isolated stages at step {args.trace_step}...",
            file=sys.stderr,
            flush=True,
        )
        trace_breakdown = _stage_breakdown(
            card, trace_state_in, args.trace_step, transport_index, libm_stats
        )

    steady = [r["compiled_step_wall_s"] for r in step_records if r["completed_step"] >= 11]
    step9_total = step_records[0]["compiled_step_wall_s"]
    step10_total = step_records[1]["compiled_step_wall_s"]
    report = {
        "label": args.label,
        "tree": str(args.tree),
        "n_walk_steps": args.n_walk_steps,
        "walk_steps_range": [first_step, last_step],
        "step8_eager_setup_wall_s": step8_eager_s,
        "step8_eager_setup_libm": step8_libm,
        "step8_state_leaf_sha256": step8_state_leaf_sha256,
        "odd_compile_wall_s": compile_times["odd"],
        "even_compile_wall_s": compile_times["even"],
        "step9_first_odd_execute_wall_s": step9_total,
        "step10_first_even_execute_wall_s": step10_total,
        "steady_state_steps_11_to_end_mean_s": round(float(np.mean(steady)), 4),
        "steady_state_steps_11_to_end_min_s": round(float(np.min(steady)), 4),
        "steady_state_steps_11_to_end_max_s": round(float(np.max(steady)), 4),
        "stage_breakdown_step11": stage_breakdown_step11,
        "trajectory_named_rows": sum(len(record["rows"]) for record in step_records),
        "trajectory_at_bar_rows": sum(
            row["status"] == "AT-BAR" for record in step_records for row in record["rows"]
        ),
        "trajectory_rows_sha256": _trajectory_rows_sha256(step_records),
        "steps": step_records,
    }
    if trace_breakdown is not None:
        report["trace_step"] = args.trace_step
        report["trace_stage_breakdown"] = trace_breakdown
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[{args.label}] wrote {args.output}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
