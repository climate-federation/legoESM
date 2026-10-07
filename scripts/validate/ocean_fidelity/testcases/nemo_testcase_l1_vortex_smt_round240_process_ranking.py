#!/usr/bin/env python3
"""Rank SMT-4 day-100 process-family intervention leverage.

The six arms are frozen in the round-240 preregistration.  They all retain the
same admitted SMT-4 NEMO trajectory and alter one registered legoESM process
family.  Consequently this tool ranks intervention leverage; it does not call
an ablation response source-exact ownership.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nemo_testcase_l1_vortex_round210_100day_comparison import (  # noqa: E402
    CARDS, TABLE_DAYS, expected_masks, load_lego, load_nemo, run_lego_card,
    sanity_check_kt1_10, score_day,
)
from nemo_testcase_l2_gyre_card_reconciliation_gate import flatten  # noqa: E402
from nemo_testcase_phase3_trajectory_gate import GateError, require  # noqa: E402

CASE = "VORTEX_SMT4_VEC-zps"
TAG = "smt4"
ARM_NAMES = (
    "baseline",
    "hpg_source_order",
    "momentum_ldf_off",
    "tracer_ldf_off",
    "bottom_drag_off",
    "vertical_mixing_evd_off",
    "barotropic_replacement_off",
)
EXPECTED_CONFIG_DIFFS = {
    "baseline": (),
    "hpg_source_order": (),
    "momentum_ldf_off": ("lateral_viscosity.A_h",),
    "tracer_ldf_off": ("gm_redi",),
    "bottom_drag_off": ("bottom_drag.bottom_drag_cd0",),
    "vertical_mixing_evd_off": (
        "A_v", "K_v", "physics.convection.enhanced_diffusion.K_conv"),
    "barotropic_replacement_off": (),
}
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round240")


def _replace_config(card, arm: str):
    cfg = card.recipe.model_config
    if arm == "momentum_ldf_off":
        cfg = cfg._replace(lateral_viscosity=cfg.lateral_viscosity._replace(A_h=0.0))
    elif arm == "tracer_ldf_off":
        cfg = cfg._replace(gm_redi=None)
    elif arm == "bottom_drag_off":
        cfg = cfg._replace(bottom_drag=cfg.bottom_drag._replace(bottom_drag_cd0=0.0))
    elif arm == "vertical_mixing_evd_off":
        convection = cfg.physics.convection
        enhanced = convection.enhanced_diffusion._replace(K_conv=0.0)
        cfg = cfg._replace(
            A_v=0.0,
            K_v=0.0,
            physics=cfg.physics._replace(convection=convection._replace(
                enhanced_diffusion=enhanced)),
        )
    return card._replace(recipe=card.recipe._replace(model_config=cfg))


def build_arm(card, arm: str):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    require(arm in ARM_NAMES, f"unregistered process arm {arm!r}")
    selected = _replace_config(card, arm)
    hooks = _NEMOWSRK3TestHooks(
        nemo_stage_rhs_accumulation_order_arm=(arm == "hpg_source_order"),
        stage_barotropic_correction=(arm != "barotropic_replacement_off"),
    )
    before_config = card.recipe.model_config
    after_config = selected.recipe.model_config
    before = flatten(before_config)
    after = flatten(after_config)
    if arm == "tracer_ldf_off":
        # ``gm_redi=None`` replaces a nested config subtree.  Register that as
        # the one intended family switch, then prove every other leaf is
        # unchanged after making the same replacement on the control object.
        require(before_config.gm_redi is not None and after_config.gm_redi is None,
                "tracer_ldf_off did not disable the resolved GM/Redi family")
        control = flatten(before_config._replace(gm_redi=None))
        require(control == after,
                "tracer_ldf_off changed a field outside the GM/Redi subtree")
        keys = ["gm_redi"]
        before = {**before, "gm_redi": before_config.gm_redi}
    else:
        keys = sorted(key for key in set(before) | set(after)
                      if key not in before or key not in after
                      or before[key] != after[key])
    require(tuple(keys) == EXPECTED_CONFIG_DIFFS[arm],
            f"{arm}: resolved config diff {keys} != registered "
            f"{list(EXPECTED_CONFIG_DIFFS[arm])}")
    rows = [{"field": key, "baseline": repr(before.get(key)),
             "arm": repr(after.get(key))} for key in keys]
    hook_rows = []
    if arm == "hpg_source_order":
        hook_rows.append({"field": "hook.nemo_stage_rhs_accumulation_order_arm",
                          "baseline": "False", "arm": "True"})
    if arm == "barotropic_replacement_off":
        hook_rows.append({"field": "hook.stage_barotropic_correction",
                          "baseline": "True", "arm": "False"})
    return selected, hooks, rows + hook_rows


def rank_rows(rows: dict[str, dict]) -> list[dict]:
    require(set(rows) == set(ARM_NAMES),
            f"family registry differs: missing={sorted(set(ARM_NAMES)-set(rows))}, "
            f"extra={sorted(set(rows)-set(ARM_NAMES))}")
    baseline = float(rows["baseline"]["100"]["T_rms"])
    ranked = []
    for name in ARM_NAMES[1:]:
        value = float(rows[name]["100"]["T_rms"])
        ranked.append({
            "arm": name,
            "day100_T_rms": value,
            "removed_T_rms": baseline - value,
            "fraction_removed": (baseline - value) / baseline,
            "finite": bool(np.isfinite(value)),
        })
    return sorted(ranked, key=lambda row: row["removed_T_rms"], reverse=True)


def _plant(reference: Path, kind: str) -> dict:
    report = json.loads(reference.read_text())
    if kind == "registry":
        report["arms"].pop("bottom_drag_off")
        rank_rows(report["arms"])
    elif kind == "effect":
        report["arms"]["hpg_source_order"]["100"]["T_rms"] += 1.0e-8
        ranked = rank_rows(report["arms"])
        hpg = next(row for row in ranked if row["arm"] == "hpg_source_order")
        require(abs(hpg["removed_T_rms"]) < 2.0e-10,
                "HPG floor control moved beyond 2e-10 K")
    else:  # pragma: no cover - argparse prevents this
        raise GateError(f"unknown plant {kind}")
    raise GateError(f"{kind} plant did not fire")


def run(root: Path, *, allow_dirty: bool = False) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "ranking must run on CPU")
    sanity = sanity_check_kt1_10(TAG, CASE, CARDS[TAG][1])
    require(sanity["status"] == "REPRODUCED",
            f"SMT-4 certified ladder calibration failed: {sanity['mismatches']}")

    base_card = build_nemo_testcase_card(CASE)
    masks = expected_masks(base_card)
    nlev = int(base_card.recipe.z_coord.n_levels)
    rows, diffs = {}, {}
    for arm in ARM_NAMES:
        print(f"ARM {arm}", flush=True)
        card, hooks, diff = build_arm(base_card, arm)
        diffs[arm] = diff
        arm_dir = root / f"lego_{arm}"
        run_lego_card(card, arm_dir, model_hooks=hooks,
                      snapshot_days=TABLE_DAYS)
        arm_rows = {}
        for day in TABLE_DAYS:
            arm_rows[str(day)] = score_day(
                load_lego(arm_dir, day), load_nemo(CARDS[TAG][1], day, nlev), masks)
        rows[arm] = arm_rows
        print(f"  day100 T_rms={arm_rows['100']['T_rms']:.16e}", flush=True)

    baseline = rows["baseline"]["100"]["T_rms"]
    require(baseline == 2.552708052055443e-04,
            f"baseline day100 T RMS {baseline!r} does not reproduce certified value")
    ranking = rank_rows(rows)
    hpg = next(row for row in ranking if row["arm"] == "hpg_source_order")
    require(abs(hpg["removed_T_rms"]) < 2.0e-10,
            f"HPG arm moved {hpg['removed_T_rms']} K beyond the floor")
    require(all(row["finite"] for row in ranking), "one or more arms is non-finite")
    return {
        "format": "nemo-testcase-l1-vortex-smt-round240-process-ranking-v1",
        "case": CASE,
        "legoesm_git_sha": sha,
        "precision_policy": "fp64/libm",
        "jax_backend": jax.default_backend(),
        "oracle_root": str(CARDS[TAG][1]),
        "table_days": list(TABLE_DAYS),
        "certified_ladder_sanity": sanity,
        "config_diffs": diffs,
        "arms": rows,
        "ranking": ranking,
        "winner": ranking[0]["arm"] if ranking[0]["removed_T_rms"] > 0 else None,
        "prediction_tracer_ldf_wins_and_removes_half": bool(
            ranking[0]["arm"] == "tracer_ldf_off"
            and ranking[0]["fraction_removed"] >= 0.5),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--plant", choices=("registry", "effect"))
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.plant:
            require(args.reference is not None, "--plant requires --reference")
            report = _plant(args.reference, args.plant)
        else:
            report = run(args.root, allow_dirty=args.allow_dirty)
    except GateError as error:
        if args.plant:
            print(f"STATUS PLANT-FIRED: {error}", file=sys.stderr)
        else:
            print(f"REFUSE: {error}", file=sys.stderr)
        return 1
    output = args.output or args.root / "process_family_ranking.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"STATUS PASS: wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
