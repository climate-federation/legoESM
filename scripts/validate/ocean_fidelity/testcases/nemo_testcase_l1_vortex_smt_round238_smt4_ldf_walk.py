#!/usr/bin/env python3
"""Locate SMT-4's first stage-3 momentum-LDF boundary under production JIT.

The admitted round-237 record writes NEMO's cumulative stage-3 momentum RHS
after HPG, vorticity, vector advection, and lateral diffusion.  This gate
drives legoESM's production-jitted step from NEMO's recorded stage-3 entry and
publishes those same boundaries through existing write-only seams.  The first
non-bit boundary therefore prevents a downstream operator from being blamed
for inherited input.  Exactness is literal cell equality, not the trajectory
bar.

The record is self-described and parsed by its admission checker (note BD).
The plant changes one scored wet cell and deliberately exits nonzero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nemo_testcase_l1_vortex_kt2_walk import (  # noqa: E402
    _seed_from_record, _u_full, _v_full, read_bt_frame, read_stage,
)
from nemo_testcase_l1_vortex_round200_flux_stage1 import (  # noqa: E402
    read_flux_stage_terms,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    GateError, expected_masks, lego_fields, read_entry, require, score,
)

CASE = "VORTEX_SMT4_VEC-zps"
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round237/"
    "oracle_vortex_smt4/kt1_10"
)
BOUNDARIES = ("hpg", "vor", "adv", "pre_ldf", "post_ldf")


def run(root: Path, *, plant: str | None = None,
        allow_dirty: bool = False, source_order: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    require(plant is None or plant in BOUNDARIES,
            f"unknown plant {plant!r}; expected one of {BOUNDARIES}")
    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "this walk must run on CPU")

    card = build_nemo_testcase_card(CASE)
    nlev = int(card.recipe.z_coord.n_levels)
    masks = expected_masks(card)
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry1 = read_entry(root / "oracle_step_entry_kt00000001.bin", CASE,
                        expect_interior=interior)
    entry2 = read_entry(root / "oracle_step_entry_kt00000002.bin", CASE,
                        expect_interior=interior)
    stages = {
        stage: read_stage(
            root / f"oracle_stage_kt00000001_s{stage}.bin",
            expect_step=1, expect_stage=stage)
        for stage in (1, 2, 3)
    }
    groups = read_flux_stage_terms(root, 3)
    frame = read_bt_frame(root / "oracle_bt_frames_kt00000001.bin",
                          expect_step=1)
    seed = _seed_from_record(card.recipe.initial_state, entry1, nlev)
    external = (
        jnp.asarray(entry2["ssh"]),
        jnp.asarray(_u_full(frame["uu_b"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vv_b"][..., None])[..., 0]),
        jnp.asarray(_u_full(frame["un_adv"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vn_adv"][..., None])[..., 0]),
    )
    stage2 = stages[2]
    stage3_entry = (
        3,
        jnp.asarray(_u_full(stage2["u"][..., :nlev])),
        jnp.asarray(_v_full(stage2["v"][..., :nlev])),
        jnp.asarray(stage2["T"][..., :nlev]),
        jnp.asarray(stage2["S"][..., :nlev]),
        jnp.asarray(stage2["ssh"]),
    )

    def production(boundary: str):
        hook_args = dict(
            stage_barotropic_output_override=external,
            stage_entry_override=stage3_entry,
            nemo_stage_rhs_accumulation_order_arm=source_order,
        )
        if boundary in ("pre_ldf", "post_ldf"):
            hook_args["expose_stage3_momentum_rhs"] = boundary
        else:
            hook_args.update(
                expose_momentum_operator={
                    "hpg": "after_hpg",
                    "vor": "after_vor",
                    "adv": "after_zad",
                }[boundary],
                expose_momentum_operator_stage=3,
            )
        hooks = _NEMOWSRK3TestHooks(**hook_args)
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        return lego_fields(model.step(seed, dt=card.dt_s))

    references = {
        "hpg": (groups["hpg_u"], groups["hpg_v"]),
        "vor": (groups["vor_u"], groups["vor_v"]),
        "adv": (groups["adv_u"], groups["adv_v"]),
        "pre_ldf": (groups["adv_u"], groups["adv_v"]),
        "post_ldf": (groups["ldf_u"], groups["ldf_v"]),
    }
    rows = []
    for boundary in BOUNDARIES:
        fields = production(boundary)
        for face, index in (("u", 0), ("v", 1)):
            candidate = np.asarray(fields[face])[..., :nlev]
            reference = np.asarray(references[boundary][index])[..., :nlev]
            active = np.asarray(masks[face], dtype=bool)
            planted = plant == boundary and face == "u"
            unplanted = candidate.copy()
            if planted:
                candidate = candidate.copy()
                where = tuple(np.argwhere(active)[0])
                candidate[where] = np.nextafter(
                    candidate[where], np.float64(np.inf))
            row = score(
                f"{CASE}.stage3.{boundary}.{face}",
                reference, candidate, active)
            delta = candidate - reference
            row.update(
                cells_unequal=int(np.count_nonzero(
                    (candidate != reference)[active])),
                max_abs=float(np.max(np.abs(delta[active]))),
                bit_exact=bool(np.array_equal(
                    candidate[active], reference[active])),
                execution_regime="production_step_jit",
                nemo_boundary={
                    "hpg": "stprk3_stg cumulative Krhs after dyn_hpg",
                    "vor": "stprk3_stg cumulative Krhs after dyn_vor",
                    "adv": "stprk3_stg cumulative Krhs after dyn_adv",
                    "pre_ldf": (
                        "stprk3_stg cumulative Krhs immediately before dyn_ldf"),
                    "post_ldf": (
                        "stprk3_stg cumulative Krhs immediately after dyn_ldf"),
                }[boundary],
                planted=planted,
                plant_movement_cells=int(np.count_nonzero(
                    candidate != unplanted)),
                plant_max_abs_movement=float(np.max(
                    np.abs(candidate - unplanted))),
            )
            rows.append(row)

    owner = None
    for boundary in BOUNDARIES:
        selected = [row for row in rows
                    if f".stage3.{boundary}." in row["name"]]
        if any(not row["bit_exact"] for row in selected):
            owner = boundary
            break
    report = {
        "format": "nemo-testcase-l1-vortex-smt-round238-ldf-walk-v1",
        "case": CASE,
        "oracle_root": str(root),
        "legoesm_git_sha": sha,
        "precision_policy": "fp64/libm",
        "jax_backend": jax.default_backend(),
        "execution_regime": "production_step_jit",
        "source_order_arm": source_order,
        "plant": plant,
        "rows": rows,
        "first_non_bit_boundary": owner,
        "status": "PLANT-FIRED" if plant else ("DEBT" if owner else "BIT"),
    }
    if plant:
        planted = [row for row in rows if row["planted"]]
        require(len(planted) == 1
                and planted[0]["plant_movement_cells"] == 1
                and planted[0]["plant_max_abs_movement"] > 0.0,
                "the one-ULP stage-boundary plant did not move exactly one "
                "scored cell")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=BOUNDARIES)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--source-order", action="store_true",
                        help="one-variable NEMO accumulator-order arm")
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, plant=args.plant,
                     allow_dirty=args.allow_dirty,
                     source_order=args.source_order)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    for row in report["rows"]:
        print(f"{row['name']:<52} unequal={row['cells_unequal']:<7d} "
              f"max={row['max_abs']:.17e} exact={row['bit_exact']}")
    print("first_non_bit_boundary:", report["first_non_bit_boundary"])
    print("STATUS", report["status"])
    return 1 if args.plant else 0


if __name__ == "__main__":
    sys.exit(main())
