#!/usr/bin/env python3
"""Production-JIT VORTEX stage-2/3 dyn_zad operand discriminator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nemo_testcase_l1_vortex_kt2_walk import (  # noqa: E402
    _seed_from_record, _u_full, _v_full, read_stage,
)
from nemo_testcase_l1_vortex_round193_stage_terms import (  # noqa: E402
    CASE, DEFAULT_ROOT, _external, read_stage_terms,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    GateError, expected_masks, lego_fields, read_entry, require,
)

EXPECTED_BASELINE = {
    2: {"u": 1.8741000428408085e-09, "v": 1.8522221618642312e-09},
    3: {"u": 1.8905432975169711e-09, "v": 1.8685471232664545e-09},
}
OPERAND_ORDER = ("velocity_u", "velocity_v", "w", "h_u", "h_v")


def _row(name, reference, candidate, mask) -> dict:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    active = np.broadcast_to(np.asarray(mask, dtype=bool), reference.shape)
    require(reference.shape == candidate.shape,
            f"{name}: shape mismatch {reference.shape} != {candidate.shape}")
    delta = candidate - reference
    unequal = int(np.count_nonzero((candidate != reference)[active]))
    ref_active = reference[active]
    candidate_active = candidate[active]
    return {
        "name": name,
        "cells_compared": int(np.count_nonzero(active)),
        "cells_unequal": unequal,
        "max_abs": float(np.max(np.abs(delta[active]))) if np.any(active) else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(delta[active])))) if np.any(active) else 0.0,
        "reference_max_abs": float(np.max(np.abs(ref_active))) if np.any(active) else 0.0,
        "candidate_max_abs": float(np.max(np.abs(candidate_active))) if np.any(active) else 0.0,
        "bit_exact": unequal == 0,
    }


def _first_active(mask) -> tuple[int, ...]:
    where = np.argwhere(np.asarray(mask, dtype=bool))
    require(where.size > 0, "plant mask has no active cell")
    return tuple(int(v) for v in where[len(where) // 2])


def _next_up(value):
    value = np.asarray(value).copy()
    return np.nextafter(value, np.inf)


def run(root: Path, expect_commit: str, *, plant: str | None = None) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.vertical import compute_layer_thickness

    valid_plants = ("velocity-ulp", "w-ulp", "thickness-ulp", "stamp")
    require(plant is None or plant in valid_plants,
            f"unknown plant {plant!r}; expected one of {valid_plants}")
    stamp = worktree_stamp()
    expected = "0" * 40 if plant == "stamp" else expect_commit.lower()
    require(stamp["clean"], "producer worktree is dirty")
    require(stamp["commit"].lower() == expected,
            f"producer commit mismatch: {stamp['commit']} != {expected}")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    require(jax.default_backend() == "cpu", "the ZAD walk must run on CPU")

    card = build_nemo_testcase_card(CASE)
    nlev = int(card.recipe.z_coord.n_levels)
    masks = expected_masks(card)
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry1 = read_entry(root / "oracle_step_entry_kt00000001.bin", CASE,
                        expect_interior=interior)
    entry2 = read_entry(root / "oracle_step_entry_kt00000002.bin", CASE,
                        expect_interior=interior)
    stages = {s: read_stage(root / f"oracle_stage_kt00000001_s{s}.bin",
                            expect_step=1, expect_stage=s) for s in (1, 2, 3)}
    seed = _seed_from_record(card.recipe.initial_state, entry1, nlev)
    external = _external(root, entry2)
    h_ref = compute_layer_thickness(
        jnp.zeros_like(seed.eta.data), seed.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    u_live_mask = seed.u_mask.data[..., None]
    v_live_mask = seed.v_mask.data[..., None]

    def stage_override(stage: int, perturb_velocity: bool = False):
        source = stages[stage - 1]
        u = np.asarray(_u_full(source["u"][..., :nlev])).copy()
        if perturb_velocity:
            owned = u[:, 1:, :]
            idx = _first_active(masks["u"])
            owned[idx] = np.nextafter(owned[idx], np.inf)
        return (stage, jnp.asarray(u),
                jnp.asarray(_v_full(source["v"][..., :nlev])),
                jnp.asarray(source["T"][..., :nlev]),
                jnp.asarray(source["S"][..., :nlev]),
                jnp.asarray(source["ssh"]))

    def execute(stage: int, *, override=None, observe=False,
                perturb_velocity=False):
        captured: dict[str, np.ndarray] = {}

        def observer(values):
            captured.update({name: np.asarray(value).copy()
                             for name, value in values.items()})

        kwargs = {
            "stage_barotropic_output_override": external,
            "stage_entry_override": stage_override(stage, perturb_velocity),
            "expose_momentum_operator": "after_zad",
            "expose_momentum_operator_stage": stage,
        }
        if stage == 2:
            kwargs["stage2_zad_operand_override"] = override
            kwargs["stage2_zad_operand_observer"] = observer if observe else None
        else:
            kwargs["stage3_zad_operand_override"] = override
            kwargs["stage3_zad_operand_observer"] = observer if observe else None
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**kwargs))
        fields = lego_fields(model.step(seed, dt=card.dt_s))
        jax.block_until_ready(fields["u"])
        if observe:
            require(set(captured) == {"u", "v", "w", "h_u", "h_v"},
                    f"stage {stage}: operand callback incomplete: {set(captured)}")
        return fields, captured

    stage_reports = []
    for stage in (2, 3):
        groups = read_stage_terms(root, stage)
        reference = {
            "velocity_u": groups["kmm_u"][..., :nlev],
            "velocity_v": groups["kmm_v"][..., :nlev],
            "w": groups["ww"],
        }
        baseline, captured = execute(stage, observe=True)
        candidate_operands = {
            "velocity_u": captured["u"][:, 1:, :nlev],
            "velocity_v": captured["v"][1:, :, :nlev],
            "w": captured["w"],
            "h_u": captured["h_u"][:, 1:, :nlev],
            "h_v": captured["h_v"][1:, :, :nlev],
        }
        # The record carries Kmm SSH, from which the compiled domqco/domzgr
        # statements uniquely reconstruct the live face thickness operands.
        h_u_ref, h_v_ref = _nemo_ws_qco_stage_faces(
            jnp.asarray(groups["ssh_kmm"]), h_ref, u_live_mask, v_live_mask,
            card.recipe.grid)[:2]
        reference["h_u"] = np.asarray(h_u_ref)[:, 1:, :nlev]
        reference["h_v"] = np.asarray(h_v_ref)[1:, :, :nlev]
        operand_masks = {
            "velocity_u": masks["u"], "velocity_v": masks["v"],
            "w": np.ones(reference["w"].shape, dtype=bool),
            "h_u": masks["u"], "h_v": masks["v"],
        }
        operand_rows = [
            _row(f"{CASE}.stage{stage}.zad.{name}", reference[name],
                 candidate_operands[name], operand_masks[name])
            for name in OPERAND_ORDER
        ]

        ref_w = reference["w"].copy()
        ref_hu_full = np.asarray(h_u_ref).copy()
        ref_hv_full = np.asarray(h_v_ref).copy()
        if plant == "w-ulp" and stage == 2:
            idx = _first_active(np.ones(ref_w.shape, dtype=bool))
            ref_w[idx] = np.nextafter(ref_w[idx], np.inf)
        if plant == "thickness-ulp" and stage == 2:
            owned_hu = ref_hu_full[:, 1:, :nlev]
            owned_hu[masks["u"]] = np.nextafter(
                owned_hu[masks["u"]], np.inf)

        arms = {}
        substitutions = {
            "baseline": None,
            "velocity": None,
            "w": (jnp.asarray(ref_w), None, None),
            "thickness": (None, jnp.asarray(ref_hu_full), jnp.asarray(ref_hv_full)),
            "all": (jnp.asarray(ref_w), jnp.asarray(ref_hu_full),
                    jnp.asarray(ref_hv_full)),
        }
        for arm, replacement in substitutions.items():
            fields = baseline if arm == "baseline" else execute(
                stage, override=replacement,
                perturb_velocity=(plant == "velocity-ulp" and stage == 2))[0]
            arm_rows = {}
            for face in ("u", "v"):
                row = _row(
                    f"{CASE}.stage{stage}.after_zad.{arm}.{face}",
                    groups[f"zad_{face}"][..., :nlev], fields[face][..., :nlev],
                    masks[face])
                expected_max = EXPECTED_BASELINE[stage][face]
                if arm == "baseline":
                    require(abs(row["max_abs"] - expected_max)
                            <= 0.01 * expected_max,
                            f"stage {stage} {face}: baseline calibration moved")
                arm_rows[face] = row
            arms[arm] = arm_rows
        for arm in ("velocity", "w", "thickness", "all"):
            for face in ("u", "v"):
                base = arms["baseline"][face]["max_abs"]
                after = arms[arm][face]["max_abs"]
                arms[arm][face]["fraction_removed"] = (base - after) / base

        first = next((row["name"].rsplit(".", 1)[-1] for row in operand_rows
                      if not row["bit_exact"]), None)
        h_ref_np = np.asarray(h_ref)
        column = np.sum(h_ref_np, axis=-1)
        cumulative = np.flip(np.cumsum(np.flip(h_ref_np, axis=-1), axis=-1), axis=-1)
        stretch = -(entry2["ssh"] - entry1["ssh"])[..., None] / card.dt_s
        stretch = stretch * cumulative / np.where(
            column[..., None] > 0.0, column[..., None], 1.0)
        stretch = np.concatenate([stretch, np.zeros_like(stretch[..., :1])], axis=-1)
        w_delta = candidate_operands["w"] - reference["w"]
        vertical = {
            "surface_k0_unequal": int(np.count_nonzero(
                candidate_operands["w"][..., 0] != reference["w"][..., 0])),
            "interior_unequal": int(np.count_nonzero(
                candidate_operands["w"][..., 1:nlev]
                != reference["w"][..., 1:nlev])),
            "bottom_kjpkm1_unequal": int(np.count_nonzero(
                candidate_operands["w"][..., nlev]
                != reference["w"][..., nlev])),
            "max_abs_by_interface": [float(value) for value in np.max(
                np.abs(w_delta), axis=(0, 1))],
            "max_abs_delta_minus_stretch": float(np.max(np.abs(w_delta - stretch))),
            "max_abs_delta_plus_stretch": float(np.max(np.abs(w_delta + stretch))),
        }
        stage_reports.append({"stage": stage, "operand_rows": operand_rows,
                              "first_non_bit_operand": first,
                              "vertical_index_convention": vertical,
                              "arms": arms})

    if plant in ("velocity-ulp", "w-ulp", "thickness-ulp"):
        arm_name = {"velocity-ulp": "velocity", "w-ulp": "w",
                    "thickness-ulp": "thickness"}[plant]
        changed = any(
            stage_reports[0]["arms"][arm_name][face]["max_abs"]
            != stage_reports[0]["arms"]["baseline"][face]["max_abs"]
            for face in ("u", "v"))
        require(changed, f"{plant} did not change the production ZAD boundary")

    return {
        "format": "nemo-testcase-l1-vortex-round194-zad-operands-v1",
        "case": CASE, "oracle_root": str(root), "legoesm_git_sha": stamp["commit"],
        "precision_policy": "fp64/libm", "execution_regime": "production_step_jit",
        "plant": plant, "stages": stage_reports,
        "status": "PLANT-FIRED" if plant else "MEASURED",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant")
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, args.expect_commit, plant=args.plant)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    for stage in report["stages"]:
        print(f"stage {stage['stage']} first_non_bit={stage['first_non_bit_operand']}")
        for row in stage["operand_rows"]:
            print(f"  {row['name']:<44} unequal={row['cells_unequal']:<6d} "
                  f"max={row['max_abs']:.17e}")
        for arm, rows in stage["arms"].items():
            print(f"  {arm:<10} U={rows['u']['max_abs']:.17e} "
                  f"V={rows['v']['max_abs']:.17e}")
    print("STATUS", report["status"])
    return 1 if args.plant else 0


if __name__ == "__main__":
    sys.exit(main())
