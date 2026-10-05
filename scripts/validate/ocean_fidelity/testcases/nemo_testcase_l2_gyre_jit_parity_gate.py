#!/usr/bin/env python3
"""Fail-closed JIT/eager parity ladder for the GYRE WS-RK3 program.

One invocation traces one private WRITE-only boundary.  This process split is
intentional: compiling every hook into one executable was the host-memory
failure that originally led the campaign harness to set ``JAX_DISABLE_JIT``.
The public ``step`` entry remains production-jitted in both measurements; the
outer context is changed only to prove that diagnostics cannot select another
arithmetic path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from nemo_testcase_l2_gyre_phase3_gate import CASE, _surface_forcings
from legoesm.ocean.fidelity.provenance import worktree_stamp

MODES = (
    "eos_prd", "hpg", "vorticity", "advection", "stage2_kaa",
    "tra_adv", "tra_zdf", "dyn_zdf", "stage_transport",
    "external_drag", "final_update",
)


def _ordered_bits(values: np.ndarray) -> np.ndarray:
    """Map finite float64 bit patterns to monotonically ordered uint64."""
    bits = np.asarray(values, dtype=np.float64).view(np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    return np.where(bits & sign, ~bits, bits | sign)


def _score(name: str, eager: np.ndarray, jitted: np.ndarray) -> dict:
    eager = np.asarray(eager, dtype=np.float64)
    jitted = np.asarray(jitted, dtype=np.float64)
    if eager.shape != jitted.shape:
        raise AssertionError(f"{name}: shape mismatch {eager.shape} != {jitted.shape}")
    if not np.all(np.isfinite(eager)) or not np.all(np.isfinite(jitted)):
        raise AssertionError(f"{name}: non-finite value")
    a = _ordered_bits(eager)
    b = _ordered_bits(jitted)
    ulp = np.maximum(a, b) - np.minimum(a, b)
    return {
        "name": name,
        "shape": list(eager.shape),
        "max_abs_delta": float(np.max(np.abs(eager - jitted), initial=0.0)),
        "max_ulp_delta": int(np.max(ulp, initial=np.uint64(0))),
        "bitwise_equal": bool(np.array_equal(eager, jitted)),
    }


def _state_arrays(state, names=("T", "S", "u", "v", "eta")) -> dict:
    return {name: np.asarray(getattr(state, name).data) for name in names}


def _run(mode: str, eager: bool):
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        compute_frozen_geom_density,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    freshwater, surface = _surface_forcings(card, card.recipe.initial_state, 1)

    if mode == "eos_prd":
        def evaluate():
            jacobian, thickness, prd, pressure_anomaly = compute_frozen_geom_density(
                card.recipe.initial_state, card.recipe.grid,
                card.recipe.z_coord, cfg)
            return jacobian, thickness, prd, pressure_anomaly

        if eager:
            with jax.disable_jit():
                values = evaluate()
        else:
            values = jax.jit(evaluate)()
        return {
            name: np.asarray(value)
            for name, value in zip(
                ("jacobian", "thickness", "prd", "pressure_anomaly"),
                values, strict=True)
        }

    hooks = _NEMOWSRK3TestHooks()
    if mode in ("hpg", "vorticity", "advection"):
        hooks = hooks._replace(expose_momentum_operator=mode)
    elif mode == "stage2_kaa":
        hooks = hooks._replace(expose_momentum_stage=2)
    elif mode == "tra_adv":
        hooks = hooks._replace(expose_stage3_advection_content=True)
    elif mode == "tra_zdf":
        # T/S are the completed implicit tracer solve; u/v are retained only
        # for a stable common result type and are not reported for this row.
        pass
    elif mode == "dyn_zdf":
        pass
    elif mode == "stage_transport":
        hooks = hooks._replace(expose_tracer_transport_stage=3)
    elif mode == "external_drag":
        hooks = hooks._replace(expose_barotropic_substeps=True)
    elif mode == "final_update":
        pass
    else:
        raise ValueError(mode)

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=hooks)
    # Prime from concrete state before either context, so cache construction is
    # not itself part of the comparison.
    model.prime_step_caches(card.recipe.initial_state)
    with jax.disable_jit(eager):
        result = model.step(
            card.recipe.initial_state, dt=card.dt_s,
            freshwater=freshwater, surface_forcing=surface)
    if mode == "external_drag":
        leaves = jax.tree_util.tree_leaves(result)
        return {
            f"trace_leaf_{index:03d}": np.asarray(value)
            for index, value in enumerate(leaves)
            if np.issubdtype(np.asarray(value).dtype, np.number)
        }
    if mode in ("hpg", "vorticity", "advection", "stage2_kaa", "dyn_zdf"):
        return _state_arrays(result, ("u", "v"))
    if mode in ("tra_adv", "tra_zdf"):
        return _state_arrays(result, ("T", "S"))
    if mode == "stage_transport":
        return {
            "zFu": np.asarray(result.u.data),
            "zFv": np.asarray(result.v.data),
            "zFw": np.asarray(result.T.data),
        }
    return _state_arrays(result)


def run(mode: str, output: Path, plant: bool = False) -> dict:
    eager = _run(mode, True)
    jitted = _run(mode, False)
    if eager.keys() != jitted.keys():
        raise AssertionError("eager/JIT output registry mismatch")
    rows = [_score(name, eager[name], jitted[name]) for name in eager]
    if plant:
        planted = np.array(jitted[rows[0]["name"]], copy=True)
        planted.flat[0] += 1.0
        rows[0] = _score(rows[0]["name"], eager[rows[0]["name"]], planted)
    passed = all(row["max_ulp_delta"] <= 2 for row in rows)
    report = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-jit-parity-v1",
        "case": CASE,
        "mode": mode,
        "policy": "fp64/x64/cpu; production step forced at public boundary",
        "acceptance": "bitwise equality preferred; <=2 ulp permitted",
        "plant": plant,
        "rows": rows,
        "verdict": "PASS" if passed else "FAIL",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    report = run(args.mode, args.output, args.plant)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
