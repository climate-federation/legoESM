#!/usr/bin/env python3
"""Walk the independent rung-0 literal Coriolis residual after e3f_0vor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round93_rhs_walk as rhs_walk,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)


PLANTS = ("none", "coefficient-bit", "application-bit")
COEFFICIENTS = (
    "ffu_nw", "ffu_ne", "ffu_sw", "ffu_se",
    "ffv_sw", "ffv_se", "ffv_nw", "ffv_ne",
)


class GateError(RuntimeError):
    """The admitted record cannot distinguish the registered statement."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def strict_application(u_bar, v_bar, coefficients):
    """Apply compiled dyn_cor_2D association with materialized NumPy ops."""

    ua = np.asarray(u_bar)[:, 1:]
    va = np.asarray(v_bar)[1:, :]
    east_v = np.roll(va, -1, axis=1)
    south_v = np.concatenate([np.zeros_like(va[:1]), va[:-1]], axis=0)
    southeast_v = np.roll(south_v, -1, axis=1)
    north_u = np.concatenate([ua[1:], np.zeros_like(ua[:1])], axis=0)
    west_u = np.roll(ua, 1, axis=1)
    northwest_u = np.roll(north_u, 1, axis=1)

    def product(name, value):
        return np.asarray(coefficients[name]) * value

    u_nw = product("ffu_nw", va)
    u_ne = product("ffu_ne", east_v)
    u_sw = product("ffu_sw", south_v)
    u_se = product("ffu_se", southeast_v)
    v_sw = product("ffv_sw", west_u)
    v_se = product("ffv_se", ua)
    v_nw = product("ffv_nw", northwest_u)
    v_ne = product("ffv_ne", north_u)
    u_north = u_nw + u_ne
    u_south = u_sw + u_se
    v_south = v_sw + v_se
    v_north = v_nw + v_ne
    return u_north + u_south, -(v_south + v_north), {
        "u_nw": u_nw, "u_ne": u_ne, "u_sw": u_sw, "u_se": u_se,
        "v_sw": v_sw, "v_se": v_se, "v_nw": v_nw, "v_ne": v_ne,
        "u_north_pair": u_north, "u_south_pair": u_south,
        "v_south_pair": v_south, "v_north_pair": v_north,
    }


def score_application(trace, coefficients, oracle, active, *, plant="none"):
    rows = []
    product_signs = {}
    for index in range(2):
        prefix = f"j{index + 1:03d}"
        u, v, products = strict_application(
            trace["u_mid"][index], trace["v_mid"][index], coefficients)
        if plant == "application-bit" and index == 0:
            u = np.array(u, copy=True)
            u[1, 49] = np.nextafter(u[1, 49], np.float64(np.inf))
        u_row = rhs_walk.score(u, oracle[f"{prefix}_cor_u"], active["u"])
        v_row = rhs_walk.score(v, oracle[f"{prefix}_cor_v"], active["v"])
        rows.extend(({"substep": index + 1, "face": "u", **u_row},
                     {"substep": index + 1, "face": "v", **v_row}))
        product_signs[prefix] = {
            name: {
                "positive_zero": int(np.count_nonzero(
                    np.asarray(value).view(np.uint64) == np.uint64(0))),
                "negative_zero": int(np.count_nonzero(
                    np.asarray(value).view(np.uint64)
                    == np.uint64(0x8000000000000000))),
            }
            for name, value in products.items()
        }
    return {"rows": rows, "product_zero_signs": product_signs}


def coefficient_movement(base, candidate):
    rows = {}
    for name in COEFFICIENTS:
        a = np.asarray(base[name])
        b = np.asarray(candidate[name])
        moved = a != b
        rows[name] = {
            "differing_cells": int(np.count_nonzero(moved)),
            "non_fold_differing_cells": int(np.count_nonzero(moved[:-1])),
            "fold_differing_cells": int(np.count_nonzero(moved[-1])),
            "maximum_absolute": float(np.max(np.abs(a - b))),
        }
    return rows


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            expect_commit: str, *, plant: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_een_coefficients,
        barotropic_substeps_latlon_cgrid,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_dynvor_e3f_0vor,
        nemo_qco_live_vorticity_e3f_cgrid,
    )

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-98 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-98 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-98 walk requires production JIT on CPU")

    oracle, census = r97.assemble_record(spg_root)
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = rung0.bridge_entry(card, rung0.assemble_frame(frame_root, 1, 0))
    freshwater, surface = rhs_walk._forcing(state.eta.data.shape)
    masks = phase3_gate.expected_masks(card)
    active = {
        "u": np.asarray(masks["u"][..., 0], dtype=bool),
        "v": np.asarray(masks["v"][..., 0], dtype=bool),
    }

    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_barotropic_substeps=True),
    )
    passive = jax.device_get(trace_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    slow = (r97._to_model_u(oracle["i000_zu_frc"]),
            r97._to_model_v(oracle["i000_zv_frc"]))
    raw_history = (
        r97._to_model_u(oracle["i000_ub_e"]),
        r97._to_model_u(oracle["i000_ubb_e"]),
        r97._to_model_v(oracle["i000_vb_e"]),
        r97._to_model_v(oracle["i000_vbb_e"]),
        oracle["i000_sshb_e"], oracle["i000_sshbb_e"],
    )

    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "literal EEN path has no carried operands")
    e3t_0 = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    )
    source_divisor = nemo_dynvor_e3f_0vor(
        e3t_0, card.recipe.z_coord.is_active, grid=card.recipe.grid,
        dtype=jnp.float64, substitute_e3f=raw.e3f_0)
    source_z = card.recipe.z_coord._replace(
        nemo_een_barotropic=raw._replace(e3f_0=source_divisor))
    source_model = LatLonCGridOceanModel(
        card.recipe.grid, source_z, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_slow_forcing_override=(slow[0], slow[1]),
            barotropic_raw_history_override=raw_history,
        ),
    )
    source_observed = jax.device_get(source_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    source_trace = source_observed.substeps
    source_score = r97._coriolis_arm(source_trace, oracle, active)

    # The literal builder freezes at Kmm. On rung 0 this is the bridged state
    # eta; prove that by equality to all eight production-traced coefficients.
    coefficient_eta = source_trace["coefficient_eta"][0]
    source_coeff = jax.device_get(_nemo_literal_een_coefficients(
        coefficient_eta, source_z, jnp.float64, scheme="een"))
    coefficient_seed = {
        name: rhs_walk.score(np.asarray(source_coeff[name]),
                             np.asarray(source_trace[name][0]),
                             np.ones_like(np.asarray(source_coeff[name]), dtype=bool))
        for name in COEFFICIENTS
    }
    require(all(row["bit_exact"] for row in coefficient_seed.values()),
            "external literal coefficient seed is not the production Kmm seed: "
            + json.dumps(coefficient_seed, sort_keys=True))

    source_application = score_application(
        source_trace, source_coeff, oracle, active,
        plant="application-bit" if plant == "application-bit" else "none")
    if plant == "application-bit":
        require(not source_application["rows"][0]["bit_exact"],
                "application-bit plant stayed green")
        raise GateError("application-bit plant fired")

    live_source = nemo_qco_live_vorticity_e3f_cgrid(
        coefficient_eta, source_z, jnp.float64, grid=card.recipe.grid,
        e3t_0=card.recipe.z_coord.nemo_e3t_0,
        tmask=card.recipe.z_coord.is_active,
        reference_e3f=source_divisor,
    )[1:, 1:, :]
    require(live_source.shape == raw.e3f_0.shape,
            "fold-aware live source divisor does not have native A2D shape")

    # Recreate only the current literal builder's live-F divisor so the arm
    # can replace its northern fold row and leave every non-fold value fixed.
    b = jax.lax.optimization_barrier
    one = jnp.asarray(1.0, dtype=jnp.float64)
    half = jnp.asarray(0.5, dtype=jnp.float64)
    quarter = jnp.asarray(0.25, dtype=jnp.float64)
    eta = jnp.asarray(coefficient_eta, dtype=jnp.float64)
    area_eta = b(b(jnp.asarray(raw.e1t) * jnp.asarray(raw.e2t)) * eta)
    east = jnp.roll(area_eta, -1, axis=1)
    north = jnp.roll(area_eta, -1, axis=0)
    northeast = jnp.roll(north, -1, axis=1)
    wet_f = (jnp.asarray(raw.hf_0) > 0.0).astype(jnp.float64)
    r1_hf0 = b(wet_f / b(jnp.asarray(raw.hf_0) + one - wet_f))
    quad = b(b(area_eta + east) + b(north + northeast))
    r3f = b(b(quarter * quad) * r1_hf0
            / b(jnp.asarray(raw.e1f) * jnp.asarray(raw.e2f)))
    current_live = b(source_divisor * b(
        one + r3f[..., None] * jnp.asarray(raw.fmask)))
    fold_only = current_live.at[-1].set(live_source[-1])
    divisor_movement = {
        "full_source_vs_current": rhs_walk.score(
            np.asarray(live_source), np.asarray(current_live),
            np.ones_like(np.asarray(current_live), dtype=bool)),
        "fold_only_vs_current": rhs_walk.score(
            np.asarray(fold_only), np.asarray(current_live),
            np.ones_like(np.asarray(current_live), dtype=bool)),
        "full_source_nonfold_unequal": int(np.count_nonzero(
            np.asarray(live_source[:-1]) != np.asarray(current_live[:-1]))),
    }

    fold_coeff = jax.device_get(_nemo_literal_een_coefficients(
        coefficient_eta, source_z, jnp.float64, scheme="een",
        _e3f_test_override=fold_only))
    if plant == "coefficient-bit":
        fold_coeff = dict(fold_coeff)
        planted = np.array(fold_coeff["ffu_nw"], copy=True)
        planted[147, 134] = np.nextafter(planted[147, 134], np.float64(np.inf))
        fold_coeff["ffu_nw"] = jnp.asarray(planted)

    pre = {
        "coefficient_evaluation": "nemo_literal",
        "literal_coefficients": fold_coeff,
        "scheme": "een",
    }
    substep_dt = float(oracle["i000_entry_sc"][0])
    substep_count = int(oracle["i000_entry_sc"][2])

    def candidate_run(seed, f_eta, f_u, f_v):
        return barotropic_substeps_latlon_cgrid(
            seed, substep_dt, substep_count,
            card.recipe.grid, source_z, card.recipe.model_config,
            F_slow_eta=f_eta, F_slow_u=f_u, F_slow_v=f_v,
            add_barotropic_coriolis=True,
            u_now=seed.u.data, v_now=seed.v.data,
            een_pre_override=pre,
            _nemo_raw_history_test_override=raw_history,
            _nemo_substep_trace_test_hook=True,
        )

    _, _, fold_trace = jax.device_get(jax.jit(candidate_run)(
        state, passive.slow_forcing[0], slow[0], slow[1]))
    fold_score = r97._coriolis_arm(fold_trace, oracle, active)
    if plant == "coefficient-bit":
        require(fold_score != source_score, "coefficient-bit plant stayed green")
        raise GateError("coefficient-bit plant fired")

    movement = coefficient_movement(source_coeff, fold_coeff)
    return {
        "status": "MEASURED_R98_CORIOLIS_RESIDUAL",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": census,
        "resolved_config": {
            "coefficient_evaluation": (
                card.recipe.model_config.barotropic
                .barotropic_een_coefficient_evaluation),
            "e3f_scheme": card.recipe.model_config.een_e3f_scheme,
            "barotropic_coriolis": (
                card.recipe.model_config.barotropic.barotropic_coriolis),
            "fold_active": bool(card.recipe.grid.fold.is_active),
        },
        "coefficient_seed_identity": coefficient_seed,
        "source_divisor_score": source_score,
        "source_associated_application": source_application,
        "divisor_movement": divisor_movement,
        "fold_only_coefficient_movement": movement,
        "fold_only_score": fold_score,
        "worktree": stamp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--spg-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.deck_root, args.frame_root, args.spg_root,
                         args.expect_commit, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, GateError, rhs_walk.GateError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS MEASURED_R98_CORIOLIS_RESIDUAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
