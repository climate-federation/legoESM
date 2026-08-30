#!/usr/bin/env python3
"""CPU-only, fail-closed reproducer for the DINO Kamm FE stability stop.

This probe deliberately mirrors the forcing and stepping loop in
``kamm_twin_90d.py`` while materialising every outer step.  Run it with JIT
disabled; otherwise it refuses to start.  NaN debugging is stamped rather
than required because it catches a masked dry-face intermediate at step 1,
before the trajectory-level checked error seen by the production arm.  The
JSON receipt is written even when the model raises, so the first failing step
and the unwrapped exception remain citable.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocean_model_module
from legoesm.ocean.experiments.dino import (
    DINOConfig,
    DINO_RECIPES,
    apply_dino_lat_lon_surface_forcing,
)

import kamm_twin_90d as twin


def parse_recipe_overrides(raw_values: list[str]) -> dict[str, object]:
    """Parse repeatable ``FIELD=JSON`` overrides for controlled CPU arms."""
    legal = set(DINOConfig.__dataclass_fields__)
    parsed: dict[str, object] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(
                f"invalid --recipe-override {raw!r}; expected FIELD=JSON")
        field, encoded = raw.split("=", 1)
        if field not in legal:
            raise ValueError(
                f"unknown DINOConfig override {field!r}; refusing an inert arm")
        if field in parsed:
            raise ValueError(f"duplicate DINOConfig override {field!r}")
        try:
            parsed[field] = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON for DINOConfig override {field!r}: {encoded!r}"
            ) from exc
    return parsed


def _finite_max(value) -> tuple[bool, float]:
    array = np.asarray(value, dtype=np.float64)
    finite = bool(np.isfinite(array).all())
    return finite, float(np.max(np.abs(array))) if finite else float("nan")


def _operand_max(value) -> dict[str, object]:
    """Return a citable maximum and location without changing model numerics."""
    array = np.asarray(value, dtype=np.float64)
    finite = bool(np.isfinite(array).all())
    if not finite:
        return {"finite": False, "max_abs": float("nan"), "index": None}
    flat_index = int(np.argmax(np.abs(array)))
    return {
        "finite": True,
        "max_abs": float(np.abs(array).reshape(-1)[flat_index]),
        "index": [int(i) for i in np.unravel_index(flat_index, array.shape)],
    }


def column_mean_deposit(before_u, before_v, after_u, after_v) -> dict[str, object]:
    """Measure the vertical solve's change to barotropic column means."""
    return {
        "u": _operand_max(np.asarray(after_u) - np.asarray(before_u)),
        "v": _operand_max(np.asarray(after_v) - np.asarray(before_v)),
    }


def column_mean_deposit_plant() -> dict[str, object]:
    """Non-vacuity control: a planted target mismatch must be detected."""
    before_u = np.zeros((2, 3), dtype=np.float64)
    before_v = np.zeros((3, 2), dtype=np.float64)
    after_u = before_u.copy()
    after_v = before_v.copy()
    after_u[1, 2] = np.nextafter(0.0, 1.0)
    metrics = column_mean_deposit(before_u, before_v, after_u, after_v)
    fired = metrics["u"]["max_abs"] > 0.0 and metrics["u"]["index"] == [1, 2]
    return {"fired": bool(fired), "metrics": metrics}


def _install_operand_trace(model, trace_rows, trace_context, start_step, end_step):
    """Spy on existing production seams; return a restoration callback."""
    original_baro = ocean_model_module.barotropic_substeps_latlon_cgrid
    model_type = type(model)
    original_vmix = model_type._apply_implicit_vertical_mixing

    def active_row():
        step = trace_context["step"]
        if start_step <= step <= end_step:
            return trace_rows.setdefault(str(step), {"step": step})
        return None

    def baro_spy(state, *positional, **keywords):
        row = active_row()
        if row is not None:
            row["barotropic_input"] = {
                name: _operand_max(getattr(state, name).data)
                for name in ("eta", "u", "v")
            }
            row["frozen_slow_forcing"] = {
                name: _operand_max(keywords.get(f"F_slow_{name}"))
                for name in ("eta", "u", "v")
            }
        answer = original_baro(state, *positional, **keywords)
        if row is not None:
            after = answer[0]
            row["barotropic_output"] = {
                name: _operand_max(getattr(after, name).data)
                for name in ("eta", "u", "v")
            }
        return answer

    def vmix_spy(self, state, *positional, **keywords):
        row = active_row()
        z_coord = keywords.get("z_coord")
        config = keywords.get("config")
        grid = keywords.get("grid")
        if row is not None:
            before_u, before_v = self._fixed_depth_means(
                state, z_coord=z_coord, config=config, grid=grid)
        answer = original_vmix(self, state, *positional, **keywords)
        after_state = answer[0] if isinstance(answer, tuple) else answer
        if row is not None:
            after_u, after_v = self._fixed_depth_means(
                after_state, z_coord=z_coord, config=config, grid=grid)
            row["vertical_solve_column_mean_before"] = {
                "u": _operand_max(before_u), "v": _operand_max(before_v)}
            row["vertical_solve_column_mean_after"] = {
                "u": _operand_max(after_u), "v": _operand_max(after_v)}
            row["vertical_solve_column_mean_deposit"] = column_mean_deposit(
                before_u, before_v, after_u, after_v)
        return answer

    ocean_model_module.barotropic_substeps_latlon_cgrid = baro_spy
    model_type._apply_implicit_vertical_mixing = vmix_spy

    def restore():
        ocean_model_module.barotropic_substeps_latlon_cgrid = original_baro
        model_type._apply_implicit_vertical_mixing = original_vmix

    return restore


def _write(path: str, receipt: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    if not bool(jax.config.jax_disable_jit):
        raise SystemExit("REFUSING: set JAX_DISABLE_JIT=1")
    if jax.default_backend() != "cpu":
        raise SystemExit(
            f"REFUSING: CPU backend required, got {jax.default_backend()!r}")
    if args.recipe != "nemo_dino_kamm":
        raise SystemExit("REFUSING: this reproducer is frozen to nemo_dino_kamm")

    set_policy(PrecisionPolicy.fp64())
    producer_commit, producer_dirty = twin._git_provenance()
    if producer_dirty:
        raise SystemExit("REFUSING: producer checkout has dirty tracked files")
    ladder = twin.resolve_ladder_mode(False)
    br, cfg, mc, model, forcing, sf, state = twin._build_twin_state(
        args.recipe,
        args.run_traj,
        args.run_stepdump,
        bridge_tke=True,
        bridge_before=True,
        bridge_before_stress_tpoint=True,
        e3t_mode=ladder,
    )
    restart = f"{args.run_stepdump}/{twin.RESTART_FILE}"
    t0_seconds = twin.seasonal_t0_seconds(restart)
    placement = cfg.surface_tendency_placement
    receipt = {
        "backend": jax.default_backend(),
        "barotropic_diffusion_alpha": float(
            mc.barotropic.barotropic_diffusion_alpha),
        "bridge_before": True,
        "bridge_before_stress_tpoint": True,
        "bridge_tke": True,
        "exception_message": None,
        "exception_type": None,
        "first_failing_step": None,
        "jax_debug_nans": bool(jax.config.jax_debug_nans),
        "jax_disable_jit": True,
        "ladder": ladder,
        "max_steps": args.max_steps,
        "outer_integrator": mc.outer_integrator,
        "producer_commit": producer_commit,
        "recipe": args.recipe,
        "recipe_overrides": args.recipe_overrides,
        "status": "RUNNING",
        "steps": [],
        "surface_tendency_placement": placement,
        "trace_end_step": args.trace_end_step,
        "trace_rows": {},
        "trace_start_step": args.trace_start_step,
        "traceback": None,
    }
    plant = column_mean_deposit_plant()
    if not plant["fired"]:
        raise AssertionError("column-mean deposit planted control did not fire")
    receipt["column_mean_deposit_plant"] = plant
    trace_context = {"step": 0}
    restore_trace = _install_operand_trace(
        model, receipt["trace_rows"], trace_context,
        args.trace_start_step, args.trace_end_step)

    print(
        "FE_REPRO_CONFIG "
        f"recipe={args.recipe} integrator={mc.outer_integrator} "
        f"alpha={mc.barotropic.barotropic_diffusion_alpha} "
        f"ladder={ladder} placement={placement} "
        f"overrides={json.dumps(args.recipe_overrides, sort_keys=True)}",
        flush=True,
    )
    for index in range(args.max_steps):
        step = index + 1
        trace_context["step"] = step
        print(f"STEP_START={step}", flush=True)
        try:
            if placement == "leapfrog_rhs":
                state, external_rate = apply_dino_lat_lon_surface_forcing(
                    state,
                    forcing,
                    br.z_coord,
                    cfg,
                    twin.DT,
                    t_seconds=t0_seconds + step * twin.DT,
                    return_rate=True,
                )
                state = model.step(
                    state,
                    twin.DT,
                    surface_forcing=sf,
                    external_tracer_rate=external_rate,
                )
            else:
                state = apply_dino_lat_lon_surface_forcing(
                    state,
                    forcing,
                    br.z_coord,
                    cfg,
                    twin.DT,
                    t_seconds=t0_seconds + step * twin.DT,
                )
                state = model.step(state, twin.DT, surface_forcing=sf)

            row = {"step": step}
            for field in ("eta", "u", "v", "T", "S"):
                finite, maximum = _finite_max(getattr(state, field).data)
                row[f"{field}_finite"] = finite
                row[f"{field}_max_abs"] = maximum
            receipt["steps"].append(row)
            print(
                "STEP_OK="
                f"{step} eta_max_abs={row['eta_max_abs']:.17g} "
                f"u_max_abs={row['u_max_abs']:.17g} "
                f"v_max_abs={row['v_max_abs']:.17g}",
                flush=True,
            )
        except Exception as exc:  # The exception and its owner are the result.
            receipt.update(
                exception_message=str(exc),
                exception_type=type(exc).__name__,
                first_failing_step=step,
                status="BLOCKED-NEEDS-FE-STABILIZATION",
                traceback=traceback.format_exc(),
            )
            _write(args.output, receipt)
            print(
                f"FIRST_FAILING_STEP={step} "
                f"EXCEPTION={type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            print(f"RECEIPT={args.output}", flush=True)
            restore_trace()
            return 2

    receipt["status"] = "NO_FAILURE_IN_REQUESTED_WINDOW"
    _write(args.output, receipt)
    restore_trace()
    print(f"NO_FAILURE_THROUGH_STEP={args.max_steps}", flush=True)
    print(f"RECEIPT={args.output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", default="nemo_dino_kamm")
    parser.add_argument("--run-traj", default=twin.RUN_TRAJ)
    parser.add_argument("--run-stepdump", default=twin.RUN_STEPDUMP)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--trace-start-step", type=int, default=25)
    parser.add_argument("--trace-end-step", type=int, default=35)
    parser.add_argument(
        "--recipe-override",
        action="append",
        default=[],
        metavar="FIELD=JSON",
        help="repeatable controlled DINOConfig recipe override",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be >= 1")
    if not (1 <= args.trace_start_step <= args.trace_end_step <= args.max_steps):
        parser.error(
            "trace window must satisfy 1 <= start <= end <= max-steps")
    try:
        args.recipe_overrides = parse_recipe_overrides(args.recipe_override)
    except ValueError as exc:
        parser.error(str(exc))
    original = DINO_RECIPES.get(args.recipe)
    if original is None:
        parser.error(f"unknown DINO recipe {args.recipe!r}")
    DINO_RECIPES[args.recipe] = {**original, **args.recipe_overrides}
    try:
        return run(args)
    finally:
        DINO_RECIPES[args.recipe] = original


if __name__ == "__main__":
    raise SystemExit(main())
