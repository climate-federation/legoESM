#!/usr/bin/env python3
"""Locate SMT-4's first stage-3 momentum-LDF boundary under production JIT.

The admitted round-237 record writes NEMO's cumulative stage-3 momentum RHS
immediately before and after ``CALL dyn_ldf``.  This gate drives legoESM's
production-jitted step from NEMO's recorded stage-3 entry and publishes the
same two boundaries through the existing write-only stage-3 RHS seam.  Thus a
non-bit pre-LDF row is inherited; an exact pre-LDF row followed by a non-bit
post-LDF row is owned by ``dyn_ldf``.  Exactness is literal cell equality, not
the trajectory bar.

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
BOUNDARIES = ("pre_ldf", "post_ldf")


def run(root: Path, *, plant: str | None = None,
        allow_dirty: bool = False) -> dict:
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
        hooks = _NEMOWSRK3TestHooks(
            stage_barotropic_output_override=external,
            stage_entry_override=stage3_entry,
            expose_stage3_momentum_rhs=boundary,
        )
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        return lego_fields(model.step(seed, dt=card.dt_s))

    references = {
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
                nemo_boundary=(
                    "stprk3_stg cumulative Krhs immediately before dyn_ldf"
                    if boundary == "pre_ldf" else
                    "stprk3_stg cumulative Krhs immediately after dyn_ldf"),
                planted=planted,
            )
            rows.append(row)

    pre = [row for row in rows if ".pre_ldf." in row["name"]]
    post = [row for row in rows if ".post_ldf." in row["name"]]
    if any(not row["bit_exact"] for row in pre):
        owner = "inherited_before_dyn_ldf"
    elif any(not row["bit_exact"] for row in post):
        owner = "dyn_ldf"
    else:
        owner = None
    report = {
        "format": "nemo-testcase-l1-vortex-smt-round238-ldf-walk-v1",
        "case": CASE,
        "oracle_root": str(root),
        "legoesm_git_sha": sha,
        "precision_policy": "fp64/libm",
        "jax_backend": jax.default_backend(),
        "execution_regime": "production_step_jit",
        "plant": plant,
        "rows": rows,
        "first_non_bit_boundary": owner,
        "status": "PLANT-FIRED" if plant else ("DEBT" if owner else "BIT"),
    }
    if plant:
        planted = [row for row in rows if row["planted"]]
        require(len(planted) == 1 and planted[0]["cells_unequal"] > 0,
                "the one-ULP stage-boundary plant did not fire")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=BOUNDARIES)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, plant=args.plant,
                     allow_dirty=args.allow_dirty)
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
