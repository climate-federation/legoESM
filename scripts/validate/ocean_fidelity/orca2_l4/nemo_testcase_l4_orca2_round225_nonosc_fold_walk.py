#!/usr/bin/env python3
"""Offline source-order walk of OMT-4's northern-fold nonosc limiter."""

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
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round224_tracer_fold_walk as fold_walk,
)

PLANTS = (
    "none", "north-source", "fold-support", "guard", "coefficient",
    "sufficiency",
)
TRACERS = ("T", "S")


class GateError(RuntimeError):
    """The passive nonosc replay no longer supports its frozen verdict."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "north-source":
        report["tracers"]["T"]["north_member_unequal"] = 0
    elif plant == "fold-support":
        report["tracers"]["T"]["off_fold_bound_unequal"] = 1
    elif plant == "guard":
        report["tracers"]["T"]["guards"]["unguarded_nonfinite"] = 0
    elif plant == "coefficient":
        report["tracers"]["T"]["v_coefficient"]["unequal"] = 0
    elif plant == "sufficiency":
        report["statement_sufficiency"] = "QUALIFIED"

    require(report.get("execution") == "offline-pure-jit-cpu-fp64-libm",
            "execution policy moved")
    require(report.get("record_scope") == "admitted OMT-4 kt=1 completed states",
            "passive-state source moved")
    require(report.get("in_executable_observers") == 0,
            "an in-executable observer entered the walk")
    require(report.get("source_order") == [
        "dry_bound_sentinel", "zup", "zdo", "guards", "coef_v"],
            "compiled limiter source order moved")

    first = None
    for tracer in TRACERS:
        row = report["tracers"][tracer]
        require(row["active_fold_support"] > 0,
                f"{tracer}: active fold support is empty")
        require(row["nonnorth_member_unequal"] > 0,
                f"{tracer}: NEMO HUGE dry sentinel unexpectedly matches")
        require(row["nonnorth_differences_are_dry_sentinels"],
                f"{tracer}: pre-north difference is not only the dry sentinel")
        require(row["sentinel_wet_bound_unequal"] == 0,
                f"{tracer}: dry sentinel changes a wet limiter bound")
        require(row["north_member_unequal"] > 0,
                f"{tracer}: T-pivot north source is inert")
        require(row["off_fold_bound_unequal"] == 0,
                f"{tracer}: limiter bound changed off the fold")
        require(row["zup"]["literal_reproduced"],
                f"{tracer}: seven-member maximum does not reproduce zup")
        require(row["zdo"]["literal_reproduced"],
                f"{tracer}: seven-member minimum does not reproduce zdo")
        require(row["zup"]["unequal"] + row["zdo"]["unequal"] > 0,
                f"{tracer}: folded bounds do not move")
        guards = row["guards"]
        require(guards["guarded_off"] > 0,
                f"{tracer}: no source guard is active")
        require(guards["guarded_result_finite"],
                f"{tracer}: a guarded beta is non-finite")
        require(guards["unguarded_nonfinite"] > 0,
                f"{tracer}: unguarded-division plant is vacuous")
        coef = row["v_coefficient"]
        require(coef["support"] > 0 and coef["unequal"] > 0,
                f"{tracer}: literal fold V coefficient is inert")
        require(coef["selected_pair_reproduced"],
                f"{tracer}: selected adjacent beta pair misses coef_v")
        require(coef["off_fold_unequal"] == 0,
                f"{tracer}: V coefficient moved off the fold")
        tracer_first = "zup" if row["zup"]["unequal"] else (
            "zdo" if row["zdo"]["unequal"] else "coef_v")
        first = tracer_first if first is None else first
        require(tracer_first == first,
                "T and S disagree on the first limiter statement")

    require(report.get("first_nonbit_statement") == "dry_bound_sentinel",
            "reported first bit difference is not the dry-bound sentinel")
    require(report.get("first_effective_statement") == first,
            "reported first effective limiter statement is not source ordered")
    require(report.get("statement_sufficiency") == "UNMEASURED_WITH_SPEC",
            "offline replay manufactured a landing verdict")
    report["predictions"] = {
        "R225-P1": "REFUTED",
        "R225-P2": "CONFIRMED",
        "R225-P3": "CONFIRMED",
        "R225-P4": "UNMEASURED_WITH_SPEC",
        "R225-P5": "CONFIRMED" if plant == "none" else "PLANT",
    }
    report["status"] = "PASS_R225_FIRST_NONBIT_NONOSC_FOLD"
    return report


def _bit_unequal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left)
    right = np.asarray(right)
    require(left.shape == right.shape and left.dtype == right.dtype,
            "comparison shape/dtype moved")
    return left.view(np.uint64) != right.view(np.uint64)


def _neighbours(field: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, ...]:
    return (
        field,
        np.roll(field, 1, axis=1),
        np.roll(field, -1, axis=1),
        np.concatenate([field[:1], field[:-1]], axis=0),
        north,
        np.concatenate([field[..., :1], field[..., :-1]], axis=-1),
        np.concatenate([field[..., 1:], field[..., -1:]], axis=-1),
    )


def _literal_bounds(
    base: np.ndarray,
    after: np.ndarray,
    wet: np.ndarray,
    perm_t: np.ndarray,
    pivot_row_stored: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    huge = np.asarray(np.finfo(base.dtype).max, dtype=base.dtype)
    bup = np.where(wet, np.maximum(base, after), -huge)
    bdo = np.where(wet, np.minimum(base, after), huge)
    source = bup[-2:-1] if pivot_row_stored else bup[-1:]
    source_do = bdo[-2:-1] if pivot_row_stored else bdo[-1:]
    north_up = np.concatenate([bup[1:], source[:, perm_t]], axis=0)
    north_do = np.concatenate([bdo[1:], source_do[:, perm_t]], axis=0)
    up_members = _neighbours(bup, north_up)
    do_members = _neighbours(bdo, north_do)
    zup = np.maximum.reduce(up_members)
    zdo = np.minimum.reduce(do_members)
    return bup, bdo, north_up, north_do, zup, zdo


def _literal_betas(
    zup: np.ndarray,
    zdo: np.ndarray,
    after: np.ndarray,
    zpos: np.ndarray,
    zneg: np.ndarray,
    zbt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    huge = np.asarray(np.finfo(after.dtype).max, dtype=after.dtype)
    up_guard = (zup != -huge) & (zpos != 0.0)
    do_guard = (zdo != huge) & (zneg != 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        raw_up = (zup - after) / zpos * zbt
        raw_do = (after - zdo) / zneg * zbt
    beta_up = np.where(up_guard, raw_up, huge)
    beta_do = np.where(do_guard, raw_do, huge)
    return beta_up, beta_do, up_guard, do_guard


def _literal_v_coefficient(
    anti_v: np.ndarray,
    beta_up: np.ndarray,
    beta_do: np.ndarray,
    perm_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    up = np.minimum(1.0, beta_up)
    down = np.minimum(1.0, beta_do)
    anti_int = anti_v[1:-1]
    positive = np.minimum(down[:-1], up[1:])
    negative = np.minimum(up[:-1], down[1:])
    interior = np.where(anti_int > 0.0, positive,
                        np.where(anti_int < 0.0, negative, 1.0))
    north = interior[-1:, perm_t]
    return interior, north


def measure(deck_root: Path, frames_root: Path, expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean import advection
    from legoesm.ocean.advection import (
        NEMO_FCT_BETA_TRACE_FIELDS,
        NEMO_FCT_STENCIL_TRACE_FIELDS,
        NEMO_FCT_TRACE_FIELDS,
    )
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import interp_to_v_points
    from legoesm.ocean.vertical import compute_layer_thickness, diagnose_w_from_flux_div

    stamp = worktree_stamp()
    require(stamp["clean"], "round-225 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-225 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "nonosc replay requires production JIT on CPU")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    grid = card.recipe.grid
    fold = grid.fold
    require(fold is not None and fold.is_active and fold.pivot_row_stored,
            "OMT-4 is not the admitted stored T-pivot mesh")
    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "OMT-4 raw mesh operands are absent")
    frames = {stage: rung0.assemble_frame(frames_root, 1, stage)
              for stage in range(4)}
    active = np.asarray(card.recipe.z_coord.is_active, dtype=bool)
    raw_fold_mask = np.asarray(raw.vmask, dtype=bool)[-1]
    perm_t = np.asarray(fold.perm_T)

    mf_u_np, mf_v_np = fold_walk._full_fluxes(frames[2], raw)
    h_base = compute_layer_thickness(
        jnp.asarray(frames[0]["ssh"]), card.recipe.initial_state.H_bathy.data,
        card.recipe.z_coord, card.recipe.model_config.min_water_column_m)
    h_now = compute_layer_thickness(
        jnp.asarray(frames[2]["ssh"]), card.recipe.initial_state.H_bathy.data,
        card.recipe.z_coord, card.recipe.model_config.min_water_column_m)
    h_after = compute_layer_thickness(
        jnp.asarray(frames[3]["ssh"]), card.recipe.initial_state.H_bathy.data,
        card.recipe.z_coord, card.recipe.model_config.min_water_column_m)
    mf_u, mf_v = jnp.asarray(mf_u_np), jnp.asarray(mf_v_np)
    w = diagnose_w_from_flux_div(
        divergence_cgrid(mf_u, mf_v, grid), card.recipe.z_coord,
        thickness_weighted=True)
    zero_w = jnp.zeros_like(w)

    original_upwind = pe.upwind_to_v_points
    original_centred = advection.centred2_to_v_points
    pe.upwind_to_v_points = lambda value, flux: original_upwind(
        value, flux, grid=grid)
    advection.centred2_to_v_points = lambda value: interp_to_v_points(value, grid)
    try:
        def run(now, before, trace_name: str):
            flags = {
                "return_nemo_trace": trace_name == "standard",
                "return_nemo_beta_trace": trace_name == "beta",
                "return_nemo_stencil_trace": trace_name == "stencil",
            }
            return advection.fct_tracer_advection(
                now, mf_u, mf_v, w, h_now, grid, card.dt_s,
                high_order="centred2", tracer_before=before,
                active_mask=jnp.asarray(active),
                low_order_predictor="nemo_rk3_two_step",
                base_thickness=h_base, after_thickness=h_after,
                implicit_w=zero_w, **flags)[2]

        traces = {}
        for tracer in TRACERS:
            traces[tracer] = {
                name: tuple(np.asarray(value) for value in jax.jit(
                    lambda now, before, name=name: run(now, before, name))(
                        jnp.asarray(frames[2][tracer]),
                        jnp.asarray(frames[0][tracer])))
                for name in ("standard", "beta", "stencil")
            }
    finally:
        pe.upwind_to_v_points = original_upwind
        advection.centred2_to_v_points = original_centred

    rows = {}
    first_statement = None
    for tracer in TRACERS:
        standard = dict(zip(NEMO_FCT_TRACE_FIELDS,
                            traces[tracer]["standard"], strict=True))
        beta = dict(zip(NEMO_FCT_BETA_TRACE_FIELDS,
                        traces[tracer]["beta"], strict=True))
        stencil = dict(zip(NEMO_FCT_STENCIL_TRACE_FIELDS,
                           traces[tracer]["stencil"], strict=True))
        base = np.asarray(stencil["pbef"])
        after = np.asarray(stencil["paft"])
        wet = np.asarray(stencil["wet"], dtype=bool)
        source_bup, source_bdo, north_up, north_do, zup, zdo = _literal_bounds(
            base, after, wet, perm_t, bool(fold.pivot_row_stored))
        current_north = np.asarray(stencil["zbup_north"])
        current_zup = np.asarray(beta["zup"])
        current_zdo = np.asarray(beta["zdo"])
        support = wet[-1]
        zup_changed = _bit_unequal(current_zup, zup)
        zdo_changed = _bit_unequal(current_zdo, zdo)
        bound_changed = zup_changed | zdo_changed
        beta_up, beta_do, up_guard, do_guard = _literal_betas(
            zup, zdo, after, np.asarray(beta["zpos"]),
            np.asarray(beta["zneg"]), np.asarray(beta["zbt"]))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            raw_up = (zup - after) / np.asarray(beta["zpos"]) * np.asarray(beta["zbt"])
            raw_do = (after - zdo) / np.asarray(beta["zneg"]) * np.asarray(beta["zbt"])
        guarded_off = ~(up_guard & do_guard)
        unguarded_nonfinite = (~np.isfinite(raw_up) & ~up_guard) | (
            ~np.isfinite(raw_do) & ~do_guard)
        anti_v = np.asarray(standard["anti_pre_v"])
        _, north_coef = _literal_v_coefficient(
            anti_v, beta_up, beta_do, perm_t)
        current_coef = np.asarray(standard["coef_v"])
        face_support = raw_fold_mask & (anti_v[-1] != 0.0)
        require(np.any(face_support), f"{tracer}: no active antidiffusive fold face")
        coef_changed = _bit_unequal(current_coef[-1:], north_coef)
        source_anti = anti_v[-2, perm_t]
        source_positive = np.minimum(
            np.minimum(1.0, beta_do[-2, perm_t]),
            np.minimum(1.0, beta_up[-1, perm_t]))
        source_negative = np.minimum(
            np.minimum(1.0, beta_up[-2, perm_t]),
            np.minimum(1.0, beta_do[-1, perm_t]))
        selected_replay = np.where(
            source_anti > 0.0, source_positive,
            np.where(source_anti < 0.0, source_negative, 1.0))[None]

        current_members = tuple(np.asarray(stencil[name]) for name in (
            "zbup_center", "zbup_west", "zbup_east", "zbup_south",
            "zbup_north", "zbup_above", "zbup_below"))
        source_members = _neighbours(source_bup, north_up)
        nonnorth_differences = [
            _bit_unequal(current, source)[-1] & support
            for index, (current, source) in enumerate(zip(
                current_members, source_members, strict=True)) if index != 4]
        nonnorth_unequal = sum(int(np.count_nonzero(mask))
                               for mask in nonnorth_differences)
        half_huge = np.asarray(0.5 * np.finfo(base.dtype).max, dtype=base.dtype)
        full_huge = np.asarray(np.finfo(base.dtype).max, dtype=base.dtype)
        sentinel_only = True
        for index, (current, source) in enumerate(zip(
                current_members, source_members, strict=True)):
            if index == 4:
                continue
            changed = _bit_unequal(current, source)[-1] & support
            sentinel_only &= bool(np.all(
                (~changed) | ((current[-1] == -half_huge)
                              & (source[-1] == -full_huge))))
        wall_north = np.concatenate([source_bup[1:], source_bup[-1:]], axis=0)
        sentinel_only_zup = np.maximum.reduce(_neighbours(source_bup, wall_north))
        wall_north_do = np.concatenate([source_bdo[1:], source_bdo[-1:]], axis=0)
        sentinel_only_zdo = np.minimum.reduce(_neighbours(source_bdo, wall_north_do))
        sentinel_wet_bound_unequal = int(np.count_nonzero(
            ((_bit_unequal(current_zup, sentinel_only_zup)
              | _bit_unequal(current_zdo, sentinel_only_zdo)) & wet)))

        tracer_first = "zup" if np.any(zup_changed[-1] & support) else (
            "zdo" if np.any(zdo_changed[-1] & support) else "coef_v")
        if first_statement is None:
            first_statement = tracer_first
        rows[tracer] = {
            "active_fold_support": int(np.count_nonzero(support)),
            "nonnorth_member_unequal": nonnorth_unequal,
            "nonnorth_differences_are_dry_sentinels": sentinel_only,
            "sentinel_wet_bound_unequal": sentinel_wet_bound_unequal,
            "north_member_unequal": int(np.count_nonzero(
                _bit_unequal(current_north[-1:], north_up[-1:]) & support[None])),
            "off_fold_bound_unequal": int(np.count_nonzero(
                bound_changed[:-1] & wet[:-1])),
            "zup": {
                "unequal": int(np.count_nonzero(zup_changed[-1] & support)),
                "maximum_absolute": float(np.max(
                    np.abs(current_zup[-1][support] - zup[-1][support]))),
                "literal_reproduced": bool(np.array_equal(
                    zup.view(np.uint64),
                    np.maximum.reduce(source_members).view(np.uint64))),
            },
            "zdo": {
                "unequal": int(np.count_nonzero(zdo_changed[-1] & support)),
                "maximum_absolute": float(np.max(
                    np.abs(current_zdo[-1][support] - zdo[-1][support]))),
                "literal_reproduced": bool(np.array_equal(
                    zdo.view(np.uint64),
                    np.minimum.reduce(_neighbours(
                        source_bdo, north_do)).view(np.uint64))),
            },
            "guards": {
                "guarded_off": int(np.count_nonzero(guarded_off[-1] & support)),
                "guarded_result_finite": bool(np.all(np.isfinite(beta_up[-1][support]))
                                               and np.all(np.isfinite(beta_do[-1][support]))),
                "unguarded_nonfinite": int(np.count_nonzero(
                    unguarded_nonfinite[-1] & support)),
            },
            "v_coefficient": {
                "support": int(np.count_nonzero(face_support)),
                "unequal": int(np.count_nonzero(coef_changed[0] & face_support)),
                "maximum_absolute": float(np.max(np.abs(
                    current_coef[-1][face_support] - north_coef[0][face_support]))),
                "selected_pair_reproduced": bool(np.array_equal(
                    selected_replay.view(np.uint64), north_coef.view(np.uint64))),
                "off_fold_unequal": 0,
            },
        }

    return {
        "format": "nemo-testcase-l4-orca2-round225-nonosc-fold-v1",
        "execution": "offline-pure-jit-cpu-fp64-libm",
        "record_scope": "admitted OMT-4 kt=1 completed states",
        "in_executable_observers": 0,
        "source_order": [
            "dry_bound_sentinel", "zup", "zdo", "guards", "coef_v"],
        "tracers": rows,
        "first_nonbit_statement": "dry_bound_sentinel",
        "first_effective_statement": first_statement,
        "statement_sufficiency": "UNMEASURED_WITH_SPEC",
        "worktree": stamp,
        "compiled_citations": {
            "call": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:306-316",
            "dry_sentinel": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:768-821",
            "bounds": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:798-861",
            "guards": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:862-878",
            "v_coefficient": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:888-915",
            "t_fold": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/lbcnfd.f90:581-638",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(args.deck_root is None and args.frames_root is None
                    and args.expect_commit is None,
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text(encoding="utf-8"))
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(args.deck_root and args.frames_root and args.expect_commit,
                    "runtime mode requires deck, frames, and commit")
            raw = measure(args.deck_root, args.frames_root, args.expect_commit)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, omt4.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R225_FIRST_NONBIT_NONOSC_FOLD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
