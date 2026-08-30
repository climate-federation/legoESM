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
        "traceback": None,
    }

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
            return 2

    receipt["status"] = "NO_FAILURE_IN_REQUESTED_WINDOW"
    _write(args.output, receipt)
    print(f"NO_FAILURE_THROUGH_STEP={args.max_steps}", flush=True)
    print(f"RECEIPT={args.output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", default="nemo_dino_kamm")
    parser.add_argument("--run-traj", default=twin.RUN_TRAJ)
    parser.add_argument("--run-stepdump", default=twin.RUN_STEPDUMP)
    parser.add_argument("--max-steps", type=int, default=64)
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
