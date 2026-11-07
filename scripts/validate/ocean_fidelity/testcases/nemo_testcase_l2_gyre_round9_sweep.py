#!/usr/bin/env python3
"""Minimal production-JIT GYRE kt=1..10 sweep for baseline comparisons."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from nemo_testcase_l2_gyre_phase3_gate import (
    CASE,
    ROOT,
    _surface_forcings,
    expected_masks,
    lego_fields,
    read_entry,
    require,
    score,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp


def run(*, support_gate: Path | None = None) -> dict:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit),
            "trajectory sweep requires production JIT")
    card = build_nemo_testcase_card(CASE)
    surface_forcings = _surface_forcings
    if support_gate is not None:
        spec = importlib.util.spec_from_file_location(
            "gyre_sweep_revision_support", support_gate)
        require(spec is not None and spec.loader is not None,
                f"cannot load support gate {support_gate}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        surface_forcings = module._surface_forcings
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    state = card.recipe.initial_state
    steps = []
    for kt in range(1, 11):
        oracle = read_entry(ROOT / f"oracle_step_entry_kt{kt:08d}.bin")
        fields = lego_fields(state)
        rows = []
        for field in ("T", "S", "u", "v", "ssh"):
            reference = (oracle[field] if field == "ssh"
                         else oracle[field][..., :nlev])
            row = score(
                f"{CASE}.kt{kt}.{field}", reference, fields[field], masks[field])
            if kt == 1 and field in ("u", "v", "ssh"):
                row["status"] = "UNINFORMATIVE"
                row["reason"] = "at-rest structural zero"
            rows.append(row)
        steps.append({"kt": kt, "rows": rows})
        if kt < 10:
            freshwater, surface = surface_forcings(card, state, kt)
            state = model.step(
                state, dt=card.dt_s, freshwater=freshwater,
                surface_forcing=surface)
    jax.tree_util.tree_map(
        lambda leaf: np.asarray(leaf) if isinstance(leaf, jax.Array) else leaf,
        state)
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round9-jit-sweep-v1",
        "case": CASE,
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "steps": steps,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-gate", type=Path)
    args = parser.parse_args(argv)
    report = run(support_gate=args.support_gate)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
