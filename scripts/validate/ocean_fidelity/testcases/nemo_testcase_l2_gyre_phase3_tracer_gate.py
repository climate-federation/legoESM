#!/usr/bin/env python3
"""Fail-closed GYRE WS-RK3 tracer-stage operand gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nemo_testcase_l2_gyre_phase3_gate import (
    BAR,
    CASE,
    _surface_forcings,
    expected_masks,
    lego_fields,
    read_tracer_stage_operands,
    require,
    score,
    sha256,
)


EXPECTED_IDENTITY = {
    "oracle_step_entry_kt00000001.bin": "9def5f4986853f497a5e0507bea185fe1ec4348715e2aeca0b14507df8e24fa0",
    "oracle_stage_kt00000001_s1.bin": "35e6892b799aeaf8d06d4affcd71b5ba0c71dc41bc0e8970c033459c46cd1402",
    "oracle_stage_kt00000001_s2.bin": "55e780b8d56eb249e5387123b20aeae8f735325200714c02a816ef719d6b3351",
    "oracle_stage_kt00000001_s3.bin": "3703a9f2e369f8f05cc439564729e36801227b54eb8afb7df2552a5120a45f2e",
}


def run(root: Path, *, plant_tracer_sbc: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")

    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True
    )
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    freshwater0, surface0 = _surface_forcings(card, card.recipe.initial_state, 1)

    operand_paths = {
        stage: root / f"oracle_rktracer_operands_kt00000001_s{stage}.bin"
        for stage in (1, 2)
    }
    for path in operand_paths.values():
        require(path.is_file(), f"missing tracer operand artifact {path}")
    operands = {
        stage: read_tracer_stage_operands(path, stage)
        for stage, path in operand_paths.items()
    }
    transport = operands[1]

    identity_rows = []
    for name, expected in EXPECTED_IDENTITY.items():
        observed = sha256(root / name)
        identity_rows.append({
            "name": f"instrumentation_bit_identity.{name}",
            "status": "VERIFIED" if observed == expected else "DEBT",
            "expected_sha256": expected,
            "observed_sha256": observed,
        })
    require(
        all(row["status"] == "VERIFIED" for row in identity_rows),
        "WRITE-only tracer instrumentation changed a pre-existing artifact",
    )

    def step_with(hooks):
        result = LatLonCGridOceanModel(
            card.recipe.grid,
            card.recipe.z_coord,
            cfg,
            _nemo_ws_test_hooks=hooks,
        ).step(
            card.recipe.initial_state,
            dt=card.dt_s,
            freshwater=freshwater0,
            surface_forcing=surface0,
        )
        return result

    stage1 = operands[1]
    source_rows = []
    for tracer in ("T", "S"):
        source_rows.append(score(
            f"{CASE}.kt1.tracer_stage1.zero_rhs.{tracer}",
            np.zeros_like(stage1[f"zero_{tracer}"][..., :nlev]),
            stage1[f"zero_{tracer}"][..., :nlev],
            masks[tracer],
        ))

    input_state = step_with(_NEMOWSRK3TestHooks(expose_tracer_stage_input=True))
    input_fields = lego_fields(input_state)
    first_over_bar = None
    for tracer in ("T", "S"):
        row = score(
            f"{CASE}.kt1.tracer_stage1.kmm_input.{tracer}",
            stage1[f"kmm_{tracer}"][..., :nlev],
            input_fields[tracer],
            masks[tracer],
        )
        source_rows.append(row)
        if row["status"] == "DEBT" and first_over_bar is None:
            first_over_bar = {
                "name": row["name"],
                "absolute_max": row["absolute_max"],
                "reference_max_abs": row["reference_max_abs"],
            }
    del input_state
    jax.clear_caches()

    faithful_stage_state = step_with(
        _NEMOWSRK3TestHooks(expose_tracer_stage=1))
    faithful_stage_fields = lego_fields(faithful_stage_state)
    faithful_stage_rows = []
    for tracer in ("T", "S"):
        row = score(
            f"{CASE}.kt1.tracer_stage1.kaa.{tracer}",
            stage1[f"kaa_{tracer}"][..., :nlev],
            faithful_stage_fields[tracer],
            masks[tracer],
        )
        faithful_stage_rows.append(row)
        if row["status"] == "DEBT" and first_over_bar is None:
            first_over_bar = {
                "name": row["name"],
                "absolute_max": row["absolute_max"],
                "reference_max_abs": row["reference_max_abs"],
            }
    del faithful_stage_state
    jax.clear_caches()

    transport_state = step_with(
        _NEMOWSRK3TestHooks(expose_tracer_transport_stage=1)
    )
    candidate_transport = {
        "zFu": (
            np.asarray(transport_state.u.data)[:, 1:, :]
            * np.asarray(card.recipe.grid.dy_u)[:, 1:, None]
        ),
        "zFv": (
            np.asarray(transport_state.v.data)[1:, :, :]
            * np.asarray(card.recipe.grid.dx_v)[1:, :, None]
        ),
        "zFw": (
            np.asarray(transport_state.T.data)
            * np.asarray(card.recipe.grid.area_T)[..., None]
        ),
    }
    transport_masks = {
        "zFu": masks["u"],
        "zFv": masks["v"],
        "zFw": np.broadcast_to(
            masks["ssh"][..., None], candidate_transport["zFw"].shape
        ),
    }
    transport_rows = [
        score(
            f"{CASE}.kt1.tracer_stage1.transport.{name}",
            transport[name][..., :candidate_transport[name].shape[-1]],
            candidate_transport[name],
            transport_masks[name],
        )
        for name in ("zFu", "zFv", "zFw")
    ]
    del transport_state
    jax.clear_caches()

    oracle_base = (
        jnp.asarray(stage1["kmm_T"][..., :nlev]),
        jnp.asarray(stage1["kmm_S"][..., :nlev]),
    )
    zero_rate = jnp.zeros_like(oracle_base[0])
    zero_sources = ((zero_rate, zero_rate),) * 3
    base_state = step_with(_NEMOWSRK3TestHooks(
        expose_tracer_stage=1,
        tracer_stage_base_override=oracle_base,
        tracer_stage_source_override=zero_sources,
    ))
    base_fields = lego_fields(base_state)
    r3_kbb = stage1["r3t_kbb"][..., None]
    r3_kmm = stage1["r3t_kmm"][..., None]
    r3_kaa = stage1["r3t_kaa"][..., None]
    advection_rows = []
    base_full_rows = []
    oracle_adv_outputs = {}
    base_movements = []
    faithful_residuals = []
    for tracer in ("T", "S"):
        oracle_adv_kaa = (
            (1.0 + r3_kbb) * stage1[f"kbb_{tracer}"][..., :nlev]
            + (card.dt_s / 3.0)
            * (1.0 + r3_kmm)
            * stage1[f"after_adv_{tracer}"][..., :nlev]
        ) / (1.0 + r3_kaa)
        oracle_adv_outputs[tracer] = oracle_adv_kaa
        advection_rows.append(score(
            f"{CASE}.kt1.tracer_stage1.after_advection.{tracer}",
            oracle_adv_kaa,
            base_fields[tracer],
            masks[tracer],
        ))
        reference = stage1[f"kaa_{tracer}"][..., :nlev]
        base_full_rows.append(score(
            f"{CASE}.kt1.tracer_stage1.oracle_kmm_base.{tracer}",
            reference,
            base_fields[tracer],
            masks[tracer],
        ))
        faithful = input_fields[tracer]
        active = masks[tracer]
        faithful_residuals.append(float(np.max(np.abs(faithful[active] - reference[active]))))
        base_movements.append(float(np.max(np.abs(base_fields[tracer][active] - faithful[active]))))
    faithful_residual = max(faithful_residuals)
    base_movement = max(base_movements)
    base_scaling = {
        "faithful_stage1_residual": faithful_residual,
        "causal_movement": base_movement,
        "movement_over_faithful_residual": base_movement / faithful_residual,
        "arm_residual": max(row["absolute_max"] for row in base_full_rows),
        "scaling_check_before_owner_label": True,
        "owner_label": (
            "CONFIRMED_CAUSAL_OWNER_OF_STAGE1_TRACER_BASE"
            if all(row["status"] == "AT-BAR" for row in base_full_rows)
            else "CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER"
        ),
    }

    oracle_w_stage1 = (
        jnp.asarray(stage1["zFw"])
        / jnp.asarray(card.recipe.grid.area_T)[..., None]
    )
    w_state = step_with(_NEMOWSRK3TestHooks(
        expose_tracer_stage=1,
        tracer_stage_base_override=oracle_base,
        tracer_stage_vertical_transport_override=(oracle_w_stage1, None, None),
        tracer_stage_source_override=zero_sources,
    ))
    w_fields = lego_fields(w_state)
    w_rows = []
    w_movements = []
    base_adv_residuals = []
    for tracer in ("T", "S"):
        active = masks[tracer]
        reference = oracle_adv_outputs[tracer]
        w_rows.append(score(
            f"{CASE}.kt1.tracer_stage1.oracle_zFw.{tracer}",
            reference,
            w_fields[tracer],
            active,
        ))
        base_adv_residuals.append(float(np.max(np.abs(
            base_fields[tracer][active] - reference[active]
        ))))
        w_movements.append(float(np.max(np.abs(
            w_fields[tracer][active] - base_fields[tracer][active]
        ))))
    base_adv_residual = max(base_adv_residuals)
    w_movement = max(w_movements)
    w_scaling = {
        "base_advection_residual": base_adv_residual,
        "causal_movement": w_movement,
        "movement_over_base_advection_residual": (
            w_movement / base_adv_residual if base_adv_residual else None
        ),
        "arm_residual": max(row["absolute_max"] for row in w_rows),
        "scaling_check_before_owner_label": True,
        "owner_label": (
            "NEAR_NULL_NO_DISCRIMINATING_POWER"
            if w_movement < 0.1 * base_adv_residual
            else (
                "CONFIRMED_CAUSAL_OWNER_OF_STAGE1_VERTICAL_TRANSPORT"
                if all(row["status"] == "AT-BAR" for row in w_rows)
                else "CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER"
            )
        ),
    }

    centered_state = step_with(_NEMOWSRK3TestHooks(
        expose_tracer_stage=1,
        tracer_stage_base_override=oracle_base,
        tracer_stage_vertical_transport_override=(oracle_w_stage1, None, None),
        tracer_stage_advection_schedule_override=("nemo_cen2", "nemo_cen2", "fct2"),
        tracer_stage_source_override=zero_sources,
    ))
    centered_fields = lego_fields(centered_state)
    centered_rows = []
    centered_movements = []
    pre_centered_residuals = []
    for tracer in ("T", "S"):
        active = masks[tracer]
        reference = oracle_adv_outputs[tracer]
        centered_rows.append(score(
            f"{CASE}.kt1.tracer_stage1.cen2_schedule.{tracer}",
            reference,
            centered_fields[tracer],
            active,
        ))
        pre_centered_residuals.append(float(np.max(np.abs(
            w_fields[tracer][active] - reference[active]
        ))))
        centered_movements.append(float(np.max(np.abs(
            centered_fields[tracer][active] - w_fields[tracer][active]
        ))))
    pre_centered_residual = max(pre_centered_residuals)
    centered_movement = max(centered_movements)
    centered_scaling = {
        "base_plus_zFw_residual": pre_centered_residual,
        "causal_movement": centered_movement,
        "movement_over_pre_centered_residual": (
            centered_movement / pre_centered_residual
            if pre_centered_residual else None
        ),
        "arm_residual": max(row["absolute_max"] for row in centered_rows),
        "scaling_check_before_owner_label": True,
        "owner_label": (
            "NEAR_NULL_NO_DISCRIMINATING_POWER"
            if centered_movement < 0.1 * pre_centered_residual
            else (
                "CONFIRMED_CAUSAL_OWNER_OF_STAGE1_ADVECTION_IDENTITY"
                if all(row["status"] == "AT-BAR" for row in centered_rows)
                else "CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER"
            )
        ),
    }

    oracle_zfu_stage1 = (
        jnp.asarray(stage1["zFu"][..., :nlev])
        / jnp.asarray(card.recipe.grid.dy_u)[:, 1:, None]
    )
    oracle_zfv_stage1 = (
        jnp.asarray(stage1["zFv"][..., :nlev])
        / jnp.asarray(card.recipe.grid.dx_v)[1:, :, None]
    )
    oracle_transport_stage1 = (
        jnp.concatenate(
            (oracle_zfu_stage1[:, -1:, :], oracle_zfu_stage1), axis=1),
        jnp.concatenate(
            (jnp.zeros_like(oracle_zfv_stage1[:1]), oracle_zfv_stage1), axis=0),
        oracle_w_stage1,
    )
    bundle_state = step_with(_NEMOWSRK3TestHooks(
        expose_tracer_stage=1,
        tracer_stage_base_override=oracle_base,
        tracer_stage_transport_override=(oracle_transport_stage1, None, None),
        tracer_stage_advection_schedule_override=("nemo_cen2", "nemo_cen2", "fct2"),
        tracer_stage_source_override=zero_sources,
    ))
    bundle_fields = lego_fields(bundle_state)
    bundle_rows = []
    bundle_movements = []
    pre_bundle_residuals = []
    for tracer in ("T", "S"):
        active = masks[tracer]
        reference = oracle_adv_outputs[tracer]
        bundle_rows.append(score(
            f"{CASE}.kt1.tracer_stage1.oracle_transport_bundle.{tracer}",
            reference,
            bundle_fields[tracer],
            active,
        ))
        pre_bundle_residuals.append(float(np.max(np.abs(
            centered_fields[tracer][active] - reference[active]
        ))))
        bundle_movements.append(float(np.max(np.abs(
            bundle_fields[tracer][active] - centered_fields[tracer][active]
        ))))
    pre_bundle_residual = max(pre_bundle_residuals)
    bundle_movement = max(bundle_movements)
    bundle_scaling = {
        "candidate_face_transport_residual": pre_bundle_residual,
        "causal_movement": bundle_movement,
        "movement_over_pre_bundle_residual": (
            bundle_movement / pre_bundle_residual
            if pre_bundle_residual else None
        ),
        "arm_residual": max(row["absolute_max"] for row in bundle_rows),
        "scaling_check_before_owner_label": True,
        "owner_label": (
            "NEAR_NULL_NO_DISCRIMINATING_POWER"
            if (pre_bundle_residual == 0.0
                or bundle_movement < 0.1 * pre_bundle_residual)
            else (
                "CONFIRMED_CAUSAL_OWNER_OF_STAGE1_TRANSPORT_ROUNDING"
                if all(row["status"] == "AT-BAR" for row in bundle_rows)
                else "CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER"
            )
        ),
    }

    oracle_rhs_stage1 = (
        jnp.asarray(stage1["after_adv_T"][..., :nlev]),
        jnp.asarray(stage1["after_adv_S"][..., :nlev]),
    )
    rhs_state = step_with(_NEMOWSRK3TestHooks(
        expose_tracer_stage=1,
        tracer_stage_base_override=oracle_base,
        tracer_stage_rhs_override=(oracle_rhs_stage1, None, None),
        tracer_stage_source_override=zero_sources,
    ))
    rhs_fields = lego_fields(rhs_state)
    rhs_rows = []
    rhs_movements = []
    pre_rhs_residuals = []
    for tracer in ("T", "S"):
        active = masks[tracer]
        reference = oracle_adv_outputs[tracer]
        rhs_rows.append(score(
            f"{CASE}.kt1.tracer_stage1.oracle_after_adv_rhs.{tracer}",
            reference,
            rhs_fields[tracer],
            active,
        ))
        pre_rhs_residuals.append(float(np.max(np.abs(
            bundle_fields[tracer][active] - reference[active]
        ))))
        rhs_movements.append(float(np.max(np.abs(
            rhs_fields[tracer][active] - bundle_fields[tracer][active]
        ))))
    pre_rhs_residual = max(pre_rhs_residuals)
    rhs_movement = max(rhs_movements)
    rhs_scaling = {
        "transport_bundle_residual": pre_rhs_residual,
        "causal_movement": rhs_movement,
        "movement_over_pre_rhs_residual": (
            rhs_movement / pre_rhs_residual if pre_rhs_residual else None
        ),
        "arm_residual": max(row["absolute_max"] for row in rhs_rows),
        "scaling_check_before_owner_label": True,
        "owner_label": (
            "NEAR_NULL_NO_DISCRIMINATING_POWER"
            if rhs_movement < 0.1 * pre_rhs_residual
            else (
                "CONFIRMED_CAUSAL_OWNER_OF_STAGE1_DIVERGENCE"
                if all(row["status"] == "AT-BAR" for row in rhs_rows)
                else "CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER"
            )
        ),
    }

    stage_sources = []
    for stage in (1, 2):
        item = operands[stage]
        stage_sources.append((
            jnp.asarray(item["after_sbc_T"][..., :nlev] - item["after_adv_T"][..., :nlev]),
            jnp.asarray(item["after_sbc_S"][..., :nlev] - item["after_adv_S"][..., :nlev]),
        ))
    stage_sources.append((zero_rate, zero_rate))
    sbc_state = step_with(_NEMOWSRK3TestHooks(
        expose_tracer_stage=1,
        tracer_stage_base_override=oracle_base,
        tracer_stage_vertical_transport_override=(oracle_w_stage1, None, None),
        tracer_stage_transport_override=(oracle_transport_stage1, None, None),
        tracer_stage_rhs_override=(oracle_rhs_stage1, None, None),
        tracer_stage_source_override=tuple(stage_sources),
        tracer_stage_advection_schedule_override=("nemo_cen2", "nemo_cen2", "fct2"),
    ))
    sbc_fields = lego_fields(sbc_state)
    sbc_rows = []
    sbc_movements = []
    pre_sbc_residuals = []
    for tracer in ("T", "S"):
        reference = stage1[f"kaa_{tracer}"][..., :nlev]
        active = masks[tracer]
        sbc_rows.append(score(
            f"{CASE}.kt1.tracer_stage1.oracle_sbc_rate.{tracer}",
            reference,
            sbc_fields[tracer],
            active,
        ))
        pre_sbc_residuals.append(float(np.max(np.abs(centered_fields[tracer][active] - reference[active]))))
        sbc_movements.append(float(np.max(np.abs(sbc_fields[tracer][active] - centered_fields[tracer][active]))))
    pre_sbc_residual = max(pre_sbc_residuals)
    sbc_movement = max(sbc_movements)
    sbc_scaling = {
        "base_plus_zFw_residual": max(
            float(np.max(np.abs(
                centered_fields[tracer][masks[tracer]]
                - stage1[f"kaa_{tracer}"][..., :nlev][masks[tracer]]
            )))
            for tracer in ("T", "S")
        ),
        "incremental_sbc_movement": sbc_movement,
        "movement_over_pre_sbc_residual": (
            sbc_movement / pre_sbc_residual if pre_sbc_residual else None
        ),
        "arm_residual": max(row["absolute_max"] for row in sbc_rows),
        "scaling_check_before_owner_label": True,
        "owner_label": (
            "CONFIRMED_CAUSAL_OWNER_OF_STAGE1_SBC_GAP"
            if all(row["status"] == "AT-BAR" for row in sbc_rows)
            else (
                "NEAR_NULL_NO_DISCRIMINATING_POWER"
                if sbc_movement < 0.1 * pre_sbc_residual
                else "CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER"
            )
        ),
    }

    planted = score(
        f"{CASE}.kt1.tracer_stage1.control.post_sbc.T",
        stage1["after_sbc_T"][..., :nlev],
        stage1["after_sbc_T"][..., :nlev],
        masks["T"],
        plant=plant_tracer_sbc,
    )
    planted_control = {"name": "control.planted_tracer_post_sbc", "status": "NOT_REQUESTED"}
    if plant_tracer_sbc:
        require(
            planted["status"] == "DEBT" and planted["absolute_max"] >= 1.0,
            "planted tracer post-SBC violation did not fire",
        )
        planted_control = {
            "name": "control.planted_tracer_post_sbc",
            "status": "VERIFIED",
            "observed_absolute_max": planted["absolute_max"],
            "expected_minimum": 1.0,
        }

    return {
        "format": "nemo-testcase-l2-gyre-phase3-tracer-v1",
        "case": CASE,
        "status": "AT-BAR" if first_over_bar is None else "DEBT",
        "bar": BAR,
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "oracle_root": str(root),
        "source_order": ["zero_rhs", "kmm_input", "transport", "fct_advection", "sbc", "qco_combine"],
        "first_over_bar": first_over_bar,
        "source_rows": source_rows,
        "faithful_stage_rows": faithful_stage_rows,
        "transport_rows": transport_rows,
        "advection_only_integrated_rows": advection_rows,
        "oracle_kmm_base_arm_rows": base_full_rows,
        "base_causal_scaling": base_scaling,
        "oracle_vertical_transport_arm_rows": w_rows,
        "vertical_transport_causal_scaling": w_scaling,
        "centered_stage_schedule_arm_rows": centered_rows,
        "centered_stage_schedule_causal_scaling": centered_scaling,
        "oracle_transport_bundle_arm_rows": bundle_rows,
        "transport_bundle_causal_scaling": bundle_scaling,
        "oracle_after_adv_rhs_arm_rows": rhs_rows,
        "after_adv_rhs_causal_scaling": rhs_scaling,
        "oracle_sbc_rate_arm_rows": sbc_rows,
        "sbc_causal_scaling": sbc_scaling,
        "planted_control": planted_control,
        "instrumentation_bit_identity": identity_rows,
        "artifacts": {
            path.name: sha256(path)
            for path in operand_paths.values()
        },
        "next_boundary": (
            "FCT_ADVECTION_OPERANDS"
            if any(row["status"] == "DEBT" for row in advection_rows)
            else "SBC_RATE"
        ),
        "trajectory_status": "UNMEASURED_STOPPED_AT_FIRST_DIVERGENCE",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-tracer-sbc", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, plant_tracer_sbc=args.plant_tracer_sbc)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
