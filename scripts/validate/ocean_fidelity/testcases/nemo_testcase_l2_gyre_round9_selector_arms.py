#!/usr/bin/env python3
"""Production-JIT one-variable audit of GYRE round-8 card choices."""

from __future__ import annotations

import argparse
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


def run(*, plant: bool = False, only: str | None = None) -> dict:
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
            "selector audit requires production JIT")

    card = build_nemo_testcase_card(CASE)
    faithful = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    tke = faithful.physics.vertical_mixing.tke
    evd = faithful.physics.convection.enhanced_diffusion

    arm_configs = {
        "tke_n2_time_level": faithful._replace(
            physics=faithful.physics._replace(
                vertical_mixing=faithful.physics.vertical_mixing._replace(
                    tke=tke._replace(tke_n2_time_level="step_entry")))),
        "evd_n2_time_level": faithful._replace(
            physics=faithful.physics._replace(
                convection=faithful.physics.convection._replace(
                    enhanced_diffusion=evd._replace(
                        evd_n2_time_level="solver_state")))),
        "een_e3f_scheme": faithful._replace(een_e3f_scheme="min"),
        "een_metric_weighting": faithful._replace(een_metric_weighting="off"),
        "een_q_boundary": faithful._replace(een_q_boundary="neumann_fill"),
        "shortwave_penetration_scheme": faithful._replace(
            physics=faithful.physics._replace(
                shortwave_penetration=(
                    faithful.physics.shortwave_penetration._replace(
                        scheme="jerlov_2band")))),
    }
    if only is not None:
        require(only in arm_configs, f"unknown selector arm {only!r}")
        arm_configs = {only: arm_configs[only]}
    if plant:
        arm_configs["tke_n2_time_level"] = arm_configs[
            "tke_n2_time_level"]._replace(een_metric_weighting="off")

    def selections(cfg):
        return {
            "tke_n2_time_level": (
                cfg.physics.vertical_mixing.tke.tke_n2_time_level),
            "evd_n2_time_level": (
                cfg.physics.convection.enhanced_diffusion.evd_n2_time_level),
            "een_e3f_scheme": cfg.een_e3f_scheme,
            "een_metric_weighting": cfg.een_metric_weighting,
            "een_q_boundary": cfg.een_q_boundary,
            "shortwave_penetration_scheme": (
                cfg.physics.shortwave_penetration.scheme),
        }

    selected = selections(faithful)
    manifest_rows = []
    for name, arm_cfg in arm_configs.items():
        changed = [
            key for key, value in selections(arm_cfg).items()
            if value != selected[key]
        ]
        manifest_rows.append({
            "name": name,
            "changed_choices": changed,
            "status": "VERIFIED" if changed == [name] else "DEBT",
        })
    require(all(row["status"] == "VERIFIED" for row in manifest_rows),
            f"one-variable manifest violation: {manifest_rows}")

    def trajectory(cfg):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg)
        state = card.recipe.initial_state
        snapshots = {}
        for kt in range(1, 11):
            if kt in (2, 10):
                snapshots[kt] = jax.tree_util.tree_map(
                    lambda leaf: np.asarray(leaf)
                    if isinstance(leaf, (jax.Array, np.ndarray)) else leaf,
                    state,
                )
            if kt < 10:
                freshwater, surface = _surface_forcings(card, state, kt)
                state = model.step(
                    state, dt=card.dt_s, freshwater=freshwater,
                    surface_forcing=surface)
        # Force the final dispatch before the next static model exists.
        jax.tree_util.tree_map(
            lambda leaf: np.asarray(leaf)
            if isinstance(leaf, jax.Array) else leaf,
            state,
        )
        jax.clear_caches()
        return snapshots

    faithful_states = trajectory(faithful)
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    reports = {}
    for name, arm_cfg in arm_configs.items():
        arm_states = trajectory(arm_cfg)
        by_step = {}
        for kt in (2, 10):
            oracle = read_entry(ROOT / f"oracle_step_entry_kt{kt:08d}.bin")
            faithful_fields = lego_fields(faithful_states[kt])
            arm_fields = lego_fields(arm_states[kt])
            rows = {}
            for field in ("T", "S", "u", "v", "ssh"):
                reference = (oracle[field] if field == "ssh"
                             else oracle[field][..., :nlev])
                row = score(
                    f"{CASE}.kt{kt}.selector_arm.{name}.{field}",
                    reference, arm_fields[field], masks[field])
                active = masks[field]
                scale = max(float(np.max(np.abs(reference[active]))), 1.0)
                row["movement_from_faithful"] = float(np.max(np.abs(
                    arm_fields[field][active]
                    - faithful_fields[field][active]))) / scale
                rows[field] = row
            by_step[str(kt)] = rows
        reports[name] = {
            "faithful_value": selected[name],
            "ablated_value": selections(arm_cfg)[name],
            "steps": by_step,
        }

    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round9-selector-arms-v1",
        "case": CASE,
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "manifest_rows": manifest_rows,
        "sources": {
            "tke_n2_time_level": (
                "EXP00/namelist_cfg:204,215-217; stprk3.F90:154-165"),
            "evd_n2_time_level": (
                "EXP00/namelist_cfg:205-207; stprk3.F90:154-165"),
            "een_e3f_scheme": (
                "EXPREF/namelist_ref:1072-1073; dynvor.F90:918-950"),
            "een_metric_weighting": (
                "EXP00/namelist_cfg:163-165; dynvor.F90:518-531"),
            "een_q_boundary": (
                "EXPREF/namelist_ref:1069; dynvor.F90:450-490"),
            "shortwave_penetration_scheme": (
                "EXP00/namelist_cfg:73,76-79; traqsr.F90:665-712,1274-1276"),
        },
        "arms": reports,
        "status": "MEASURED",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument(
        "--only", choices=(
            "tke_n2_time_level", "evd_n2_time_level", "een_e3f_scheme",
            "een_metric_weighting", "een_q_boundary",
            "shortwave_penetration_scheme"))
    args = parser.parse_args(argv)
    try:
        report = run(plant=args.plant, only=args.only)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 2
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
