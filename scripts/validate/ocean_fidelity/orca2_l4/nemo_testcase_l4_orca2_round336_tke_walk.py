#!/usr/bin/env python3
"""Replay the admitted kt=2 ORCA2 TKE record through production arithmetic."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[4]
for package in (REPO, REPO / "packages/core", REPO / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.tke import (  # noqa: E402
    _mxl0_surface_anchor,
    _tke_raw_mixing_length,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    nemo_testcase_l4_orca2_round336_tke_gate as admission,
)


class WalkError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise WalkError(message)


def _bits(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, np.float64)).view(np.uint64)


def _score(actual, expected, wet) -> dict[str, object]:
    actual = np.asarray(actual, np.float64)
    expected = np.asarray(expected, np.float64)
    wet = np.asarray(wet, bool)
    require(actual.shape == expected.shape == wet.shape,
            f"score shape mismatch {actual.shape}/{expected.shape}/{wet.shape}")
    unequal = (_bits(actual) != _bits(expected)) & wet
    delta = np.abs(actual - expected)
    return {
        "wet_words": int(np.count_nonzero(wet)),
        "unequal_wet_words": int(np.count_nonzero(unequal)),
        "max_abs": float(np.max(delta[wet], initial=0.0)),
        "max_abs_unequal": float(np.max(delta[unequal], initial=0.0)),
        "bit_identical": not bool(np.any(unequal)),
    }


def _rank_arrays(record: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    f, h = record["fields"], record["header"]
    i = slice(h["ntsi"] - 1, h["ntei"])
    j = slice(h["ntsj"] - 1, h["ntej"])
    trans = lambda a: (a.transpose(1, 0, *range(2, a.ndim))
                       if a.ndim >= 2 else a)
    local = {name: trans(value) for name, value in f.items()}
    full = {name: trans(value[i, j]) for name, value in f.items()
            if name in admission.FULL_3D | admission.FULL_2D}
    return local, full


def _length_boundaries(local, full, cfg, en_value=None):
    en = jnp.asarray(local["after_tke_en"][..., 1:-1]
                     if en_value is None else en_value)
    rn2 = jnp.asarray(full["entry_rn2"][..., 1:-1])
    anchor = _mxl0_surface_anchor(
        cfg, jnp.asarray(local["entry_taum"]), 1026.0, 9.80665,
        jnp.asarray(full["entry_tmask"][..., 0]),
        jnp.asarray(full["entry_fr_i"]), jnp.asarray(local["entry_hm_i"]))
    raw = _tke_raw_mixing_length(en, rn2, cfg)
    stack = jnp.concatenate((anchor[..., None], raw), axis=-1)
    e3 = jnp.asarray(local["entry_e3t_kmm"])
    lup = [stack[..., 0]]
    for k in range(1, 30):
        lup.append(jnp.minimum(lup[-1] + e3[..., k - 1], stack[..., k]))
    lup = jnp.stack(lup, axis=-1)
    ldown = [None] * 30
    ldown[-1] = jnp.minimum(jnp.asarray(cfg.mxl_min) + e3[..., -1], stack[..., -1])
    for k in range(28, 0, -1):
        ldown[k] = jnp.minimum(ldown[k + 1] + e3[..., k + 1], stack[..., k])
    ldown[0] = stack[..., 0]
    ldown = jnp.stack(ldown, axis=-1)
    final = jnp.minimum(lup, ldown)
    eps = jnp.sqrt(lup * ldown)
    return tuple(np.asarray(x) for x in (stack, lup, ldown, final, eps))


def _assemble(records, name, full=False):
    pieces = []
    for record in records:
        value, h = record["fields"][name], record["header"]
        if full:
            value = value[h["ntsi"] - 1:h["ntei"],
                          h["ntsj"] - 1:h["ntej"]]
        pieces.append(value)
    return np.concatenate(pieces, axis=0).transpose(
        1, 0, *range(2, pieces[0].ndim))


def _solver_boundaries(local):
    diagonal = np.array(local["matrix_diag"][..., :-1], copy=True)
    lower = local["matrix_lower"][..., :-1]
    upper = local["matrix_upper"][..., :-1]
    rhs = local["matrix_rhs_en"][..., :-1]
    for k in range(1, 30):
        diagonal[..., k] = (diagonal[..., k]
                            - lower[..., k] * upper[..., k - 1]
                            / diagonal[..., k - 1])
    work = np.array(lower, copy=True)
    for k in range(1, 30):
        work[..., k] = (rhs[..., k] - lower[..., k]
                        / diagonal[..., k - 1] * work[..., k - 1])
    bottom = np.array(rhs, copy=True)
    bottom[..., 29] = work[..., 29] / diagonal[..., 29]
    back = np.array(bottom, copy=True)
    for k in range(28, 0, -1):
        back[..., k] = ((work[..., k] - upper[..., k] * back[..., k + 1])
                        / diagonal[..., k])
    return diagonal, work, bottom, back


def measure(record_root: Path, deck_root: Path,
            plant: str | None = None) -> dict[str, object]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", "walk is CPU-only")
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)

    records = [admission.read_record(record_root / admission.record_name(rank))
               for rank in (0, 1)]
    admission_rows = {str(rank): admission.verify_record(record)
                      for rank, record in enumerate(records)}
    _, shipped = ladder.card_fields(deck_root)
    card = ladder.comparison_card(shipped)
    cfg = card.recipe.model_config.physics.vertical_mixing.tke
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        iwm_forcing=card.recipe.iwm_forcing,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True))
    state = card.recipe.initial_state
    result = None
    for kt in (1, 2):
        fields = ladder.assemble_surface_fields(deck_root, kt)
        freshwater, surface = ladder._surface_forcings(
            card, deck_root, fields, kt)
        result = jax.device_get(model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface,
            **ladder.damping_step_kwargs(card, deck_root, kt)))
        state = result.state_after
    trace = result.tke_statement_trace
    require(trace is not None, "production card returned no TKE trace")

    local = {name: _assemble(records, name) for name in admission.FIELDS
             if name not in admission.ONE_D}
    full = {name: _assemble(records, name, full=True)
            for name in admission.FULL_3D | admission.FULL_2D}
    wet30 = full["tke_wmask"][..., :-1] != 0.0
    wet29 = full["tke_wmask"][..., 1:-1] != 0.0
    diag, work, bottom, back = _solver_boundaries(local)
    stack, lup, ldown, final, eps = _length_boundaries(
        local, full, cfg, np.asarray(state.tke.data))
    raw_k = np.asarray(jax.jit(
        lambda e, length: cfg.c_k * length * jnp.sqrt(e))(
            jnp.asarray(state.tke.data), jnp.asarray(final[..., 1:])))
    pre = np.maximum(cfg.kappaH_min, raw_k)

    ordered = (
        "after_bc", "after_langmuir", "matrix_rhs", "matrix_diag",
        "matrix_upper", "matrix_lower", "solve_diag", "solve_rhs",
        "solve_bottom", "solve_back", "after_solve", "after_tke",
        "pre_sweep_mxl", "after_lup", "after_ldown", "final_mxl",
        "final_mxld", "zav", "pre_pdlr_avt", "final_avm", "final_avt",
    )
    actual = {
        "after_bc": (trace.en_after_boundaries, local["after_bc_en"][..., :-1], wet30),
        "after_langmuir": (trace.en_after_langmuir, local["post_langmuir_en"][..., :-1], wet30),
        "matrix_rhs": (trace.rhs_pre_sweep, local["matrix_rhs_en"][..., :-1], wet30),
        "matrix_diag": (trace.matrix_diag, local["matrix_diag"][..., 1:-1], wet29),
        "matrix_upper": (trace.matrix_upper, local["matrix_upper"][..., 1:-1], wet29),
        "matrix_lower": (trace.matrix_lower, local["matrix_lower"][..., 1:-1], wet29),
        "solve_diag": (diag, local["solve_diag"][..., :-1], wet30),
        "solve_rhs": (work, local["solve_rhs"][..., :-1], wet30),
        "solve_bottom": (bottom, local["solve_bottom"][..., :-1], wet30),
        "solve_back": (back, local["solve_back"][..., :-1], wet30),
        "after_solve": (trace.en_post_sweep, local["after_solve_en"][..., :-1], wet30),
        "after_tke": (state.tke.data, local["after_tke_en"][..., 1:-1], wet29),
        "pre_sweep_mxl": (stack, local["pre_sweep_mxl"][..., :-1], wet30),
        "after_lup": (lup, local["after_lup_mxld"][..., :-1], wet30),
        "after_ldown": (ldown, local["after_ldown_mxl"][..., :-1], wet30),
        "final_mxl": (final, local["final_mxl"][..., :-1], wet30),
        "final_mxld": (eps, local["final_mxld"][..., :-1], wet30),
        "zav": (raw_k, cfg.c_k * local["final_mxl"][..., 1:-1]
                * np.sqrt(local["after_tke_en"][..., 1:-1]), wet29),
        "pre_pdlr_avt": (pre, local["pre_pdlr_avt"][..., 1:-1], wet29),
        "final_avm": (state.tke_avm.data, full["final_avm"][..., 1:-1], wet29),
        "final_avt": (state.tke_avt.data, local["final_avt"][..., 1:-1], wet29),
    }
    rows = {}
    for name in ordered:
        candidate, expected, wet = actual[name]
        if plant == name:
            expected = np.array(expected, copy=True)
            expected[tuple(np.argwhere(wet)[0])] = np.nextafter(
                expected[tuple(np.argwhere(wet)[0])], np.inf)
        rows[name] = _score(candidate, expected, wet)
    first = next((name for name in ordered if not rows[name]["bit_identical"]), None)
    if plant is not None:
        require(first == plant, f"plant {plant} did not become first ({first})")
        raise WalkError(f"{plant} plant fired")
    return {
        "format": "nemo-testcase-l4-orca2-round336-tke-walk-v1",
        "status": "PASS_R336_TKE_WALK",
        "execution": {"backend": jax.default_backend(), "dtype": "float64",
                      "jit": "two actual production-card steps"},
        "record_admission": admission_rows,
        "ordered_rows": rows,
        "first_non_bit_boundary": first,
        "boundary_kind": {
            "solve_diag": "source-order replay of recorded live inputs",
            "solve_rhs": "source-order replay of recorded live inputs",
            "solve_bottom": "source-order replay of recorded live inputs",
            "solve_back": "source-order replay of recorded live inputs",
            "after_solve": "actual production-card JIT output",
        },
        "compiled_solver_walk": _compiled_solver_walk(
            local, full, trace, cfg, wet30, wet29, card, records),
    }


def _instrumented_production_tke_solve_impl(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    rhs: jnp.ndarray,
    surface_en: jnp.ndarray,
    w_active: jnp.ndarray,
    floor: float,
    source_rounded: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Debug-only observer of ``_nemo_literal_tke_solve`` (tke.py:1111-1224).

    Executes the exact compiled arithmetic of the production solver and
    returns ``(solve_diag, solve_rhs, solve_bottom, solve_back, after_solve,
    e_new)`` without modifying any production expression.
    """
    from legoesm.core.source_rounding import nemo_source_round
    rp = nemo_source_round if source_rounded else (lambda value: value)
    jpkm1 = a.shape[-1] - 1
    diag_seed = 1.0 / jnp.asarray(surface_en, dtype=b.dtype)
    work_seed = jnp.ones_like(diag_seed)

    diag_inputs = tuple(jnp.moveaxis(x, -1, 0) for x in (
        a[..., 1:jpkm1 + 1], b[..., 1:jpkm1 + 1], c[..., :jpkm1]))

    def diagonal_step(previous, operands):
        lower, diagonal, previous_upper = operands
        current = diagonal - lower * previous_upper / previous
        return current, current

    _, diagonal_rows = jax.lax.scan(diagonal_step, diag_seed, diag_inputs)
    diagonal = jnp.concatenate(
        [diag_seed[..., None], jnp.moveaxis(diagonal_rows, 0, -1)], axis=-1)

    rhs_inputs = tuple(jnp.moveaxis(x, -1, 0) for x in (
        rhs[..., 1:jpkm1 + 1], a[..., 1:jpkm1 + 1], diagonal[..., :jpkm1]))

    def rhs_step(previous, operands):
        source, lower, previous_diagonal = operands
        current = source - rp(lower / previous_diagonal * previous)
        return current, current

    _, work_rows = jax.lax.scan(rhs_step, work_seed, rhs_inputs)
    work = jnp.concatenate(
        [work_seed[..., None], jnp.moveaxis(work_rows, 0, -1)], axis=-1)

    terminal = work[..., jpkm1] / diagonal[..., jpkm1]
    solve_bottom = rhs.at[..., jpkm1].set(terminal)

    reverse_inputs = tuple(
        jnp.moveaxis(x[..., 1:jpkm1][..., ::-1], -1, 0)
        for x in (work, c, diagonal))

    def reverse_step(next_value, operands):
        source, upper, current_diagonal = operands
        current = (source - rp(upper * next_value)) / current_diagonal
        return current, current

    _, reverse_rows = jax.lax.scan(reverse_step, terminal, reverse_inputs)
    solved_prefix = jnp.concatenate(
        [jnp.moveaxis(reverse_rows, 0, -1)[..., ::-1],
         terminal[..., None]], axis=-1)
    solve_back = jnp.concatenate([rhs[..., :1], solved_prefix], axis=-1)
    e_new = (jnp.maximum(solved_prefix,
                         jnp.asarray(floor, dtype=solved_prefix.dtype))
             * jnp.asarray(w_active, dtype=solved_prefix.dtype))
    after_solve = jnp.concatenate([rhs[..., :1], e_new], axis=-1)
    return diagonal, work, solve_bottom, solve_back, after_solve, e_new


_instrumented_production_tke_solve = jax.jit(
    _instrumented_production_tke_solve_impl,
    static_argnames=("source_rounded",),
)


def _score_with_ranks(actual, expected, mask) -> dict[str, object]:
    row = _score(actual, expected, mask)
    actual_a = np.asarray(actual, np.float64)
    expected_a = np.asarray(expected, np.float64)
    mask_a = np.asarray(mask, bool)
    unequal = (_bits(actual_a) != _bits(expected_a)) & mask_a
    if unequal.ndim >= 2 and unequal.shape[1] == 180:
        row["rank0_unequal_wet_words"] = int(np.count_nonzero(unequal[:, :90]))
        row["rank1_unequal_wet_words"] = int(np.count_nonzero(unequal[:, 90:]))
    return row


def _evaluate_compiled_solver(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    rhs: jnp.ndarray,
    surface_en: jnp.ndarray,
    w_active: jnp.ndarray,
    floor: float,
    source_rounded: bool,
    local: dict[str, np.ndarray],
    wet30: np.ndarray,
    bottom_bc_mask30: np.ndarray,
    active_solve30: np.ndarray,
    all30: np.ndarray,
    plant_stage: str | None = None,
) -> dict[str, object]:
    from legoesm.ocean.physics.vertical_mixing.tke import _nemo_literal_tke_solve
    prod_fn = jax.jit(_nemo_literal_tke_solve, static_argnames=("source_rounded",))
    prod_e_new = np.asarray(jax.device_get(prod_fn(
        a, b, c, rhs, surface_en, w_active, floor,
        source_rounded=source_rounded)))
    stages = tuple(np.asarray(x) for x in jax.device_get(
        _instrumented_production_tke_solve(
            a, b, c, rhs, surface_en, w_active, floor,
            source_rounded=source_rounded)))
    s_diag, s_rhs, s_bot, s_back, s_after, e_new = stages
    require(np.array_equal(_bits(e_new), _bits(prod_e_new)),
            "instrumented solver output differs from production _nemo_literal_tke_solve")
    stage_pairs = []
    for name, cand, exp in (
        ("matrix_rhs", np.asarray(rhs), local["matrix_rhs_en"][..., :-1]),
        ("solve_diag", s_diag, local["solve_diag"][..., :-1]),
        ("solve_rhs", s_rhs, local["solve_rhs"][..., :-1]),
        ("solve_bottom", s_bot, local["solve_bottom"][..., :-1]),
        ("solve_back", s_back, local["solve_back"][..., :-1]),
        ("after_solve", s_after, local["after_solve_en"][..., :-1]),
    ):
        if plant_stage == name:
            exp = np.array(exp, copy=True)
            idx = tuple(np.argwhere(wet30)[0])
            exp[idx] = np.nextafter(exp[idx], np.inf)
        stage_pairs.append((name, cand, exp))
    solver_order = ("solve_diag", "solve_rhs", "solve_bottom", "solve_back", "after_solve")
    wet_rows = {name: _score_with_ranks(cand, exp, wet30) for name, cand, exp in stage_pairs}
    bot_rows = {name: _score_with_ranks(cand, exp, bottom_bc_mask30) for name, cand, exp in stage_pairs}
    active_rows = {name: _score_with_ranks(cand, exp, active_solve30) for name, cand, exp in stage_pairs}
    all_rows = {name: _score_with_ranks(cand, exp, all30) for name, cand, exp in stage_pairs}
    first_wet = next((n for n in solver_order if not wet_rows[n]["bit_identical"]), None)
    first_active = next((n for n, _, _ in stage_pairs if not active_rows[n]["bit_identical"]), None)
    if plant_stage is not None:
        expected_first = first_wet if plant_stage in solver_order else first_active
        require(expected_first == plant_stage,
                f"compiled solver plant {plant_stage} did not become first ({expected_first})")
        raise WalkError(f"compiled solver {plant_stage} plant fired")
    return {
        "bit_identical_to_uninstrumented_production": True,
        "wet30_rows": wet_rows,
        "bottom_bc30_rows": bot_rows,
        "active_solve30_rows": active_rows,
        "all30_rows": all_rows,
        "first_non_bit_wet30_solver_stage": first_wet,
        "first_non_bit_active_solve30_stage": first_active,
        "_arrays": {
            "solve_diag": s_diag, "solve_rhs": s_rhs, "solve_bottom": s_bot,
            "solve_back": s_back, "after_solve": s_after, "e_new": e_new,
        },
    }


def _compiled_solver_walk(
    local: dict[str, np.ndarray],
    full: dict[str, np.ndarray],
    trace: object,
    cfg: object,
    wet30: np.ndarray,
    wet29: np.ndarray,
    card: object,
    records: list[dict],
) -> dict[str, object]:
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.physics.vertical_mixing.tke import (
        nemo_bottom_tke_dirichlet, nemo_etau_injection,
        nemo_tke_effective_ice_fraction)

    mbkt = full["entry_mbkt"].astype(np.int64)
    ssmask = full["entry_ssmask"] != 0.0
    valid_bot = (mbkt >= 1) & (mbkt <= 29) & ssmask
    bottom_bc_mask30 = np.zeros_like(wet30, dtype=bool)
    jj, ii = np.meshgrid(np.arange(wet30.shape[0]), np.arange(wet30.shape[1]), indexing="ij")
    bottom_bc_mask30[jj[valid_bot], ii[valid_bot], mbkt[valid_bot]] = True
    active_solve30 = wet30 | bottom_bc_mask30
    all30 = np.ones_like(wet30, dtype=bool)

    w_active = jnp.asarray(full["tke_wmask"][..., 1:-1], dtype=jnp.float64)
    floor = float(cfg.tke_background)
    source_rounded = bool(getattr(cfg, "nemo_tke_solver_source_rounded", True))

    # 1. Recorded kt=2 matrix/RHS fed into compiled production solver.
    z0 = jnp.zeros_like(jnp.asarray(local["matrix_rhs_en"][..., :1]))
    o0 = jnp.ones_like(z0)
    a_rec = jnp.concatenate([z0, jnp.asarray(local["matrix_lower"][..., 1:-1])], axis=-1)
    b_rec = jnp.concatenate([o0, jnp.asarray(local["matrix_diag"][..., 1:-1])], axis=-1)
    c_rec = jnp.concatenate(
        [z0, jnp.asarray(local["matrix_upper"][..., 1:-1]).at[..., -1].set(0.0)],
        axis=-1)
    rhs_rec = jnp.asarray(local["matrix_rhs_en"][..., :-1])
    sfc_rec = jnp.asarray(local["matrix_rhs_en"][..., 0])
    recorded_eval = _evaluate_compiled_solver(
        a_rec, b_rec, c_rec, rhs_rec, sfc_rec, w_active, floor,
        source_rounded, local, wet30, bottom_bc_mask30, active_solve30, all30)
    recorded_eval.pop("_arrays")

    # 2. Production card live kt=2 matrix/RHS fed into compiled production solver.
    a_live = jnp.concatenate([z0, jnp.asarray(trace.matrix_lower)], axis=-1)
    b_live = jnp.concatenate([o0, jnp.asarray(trace.matrix_diag)], axis=-1)
    c_live = jnp.concatenate(
        [z0, jnp.asarray(trace.matrix_upper).at[..., -1].set(0.0)], axis=-1)
    rhs_live = jnp.asarray(trace.rhs_pre_sweep)
    sfc_live = jnp.asarray(trace.surface_dirichlet)
    live_eval = _evaluate_compiled_solver(
        a_live, b_live, c_live, rhs_live, sfc_live, w_active, floor,
        source_rounded, local, wet30, bottom_bc_mask30, active_solve30, all30)
    live_arrays = live_eval.pop("_arrays")
    require(
        np.array_equal(_bits(live_arrays["after_solve"]),
                       _bits(np.asarray(trace.en_post_sweep))),
        "compiled solver on live card matrix/RHS differs from trace.en_post_sweep")

    # 3. Non-landed literal fix proposal: north-fold top-V face mask in _tke_bottom_dirichlet.
    tmask = jnp.asarray(full["entry_tmask"][..., :-1], dtype=jnp.float64)
    uu_east = jnp.asarray(full["entry_uu_kbb"][..., :-1], dtype=jnp.float64)
    u_f = jnp.concatenate([uu_east[:, -1:, :], uu_east], axis=1)
    vv_north = jnp.asarray(full["entry_vv_kbb"][..., :-1], dtype=jnp.float64)
    v_f = jnp.concatenate([jnp.zeros_like(vv_north[:1]), vv_north], axis=0)
    um3, vm3_unfixed = compute_face_masks_3d(tmask)
    um3 = um3.astype(jnp.float64)
    vm3_unfixed = vm3_unfixed.astype(jnp.float64)
    vm3_fixed = vm3_unfixed.at[-1].set(
        jnp.asarray(full["entry_vmask"][-1, :, :-1], dtype=jnp.float64))
    bl_idx = jnp.maximum(jnp.asarray(mbkt - 1), 0)[..., None]
    _at_bottom = lambda arr: jnp.take_along_axis(arr, bl_idx, axis=-1)[..., 0]
    ssmask_f = _at_bottom(tmask)
    r_t = jnp.where(ssmask_f > 0.0, 0.0004, 0.0)

    def _eval_bottom_bc(vm_use):
        m_uw, m_ue = _at_bottom(um3[:, :-1, :]), _at_bottom(um3[:, 1:, :])
        m_vs, m_vn = _at_bottom(vm_use[:-1, :, :]), _at_bottom(vm_use[1:, :, :])
        u_sum = (2.0 - m_uw * m_ue) * (
            _at_bottom(u_f[:, :-1, :]) * m_uw + _at_bottom(u_f[:, 1:, :]) * m_ue)
        v_sum = (2.0 - m_vs * m_vn) * (
            _at_bottom(v_f[:-1, :, :]) * m_vs + _at_bottom(v_f[1:, :, :]) * m_vn)
        return np.asarray(ssmask_f * nemo_bottom_tke_dirichlet(r_t, u_sum, v_sum, cfg))

    bot_unfixed = _eval_bottom_bc(vm3_unfixed)
    bot_fixed = _eval_bottom_bc(vm3_fixed)
    nemo_bot_2d = np.zeros_like(bot_fixed)
    nemo_bot_2d[valid_bot] = local["matrix_rhs_en"][jj[valid_bot], ii[valid_bot], mbkt[valid_bot]]

    rhs_fixed_np = np.array(np.asarray(trace.rhs_pre_sweep), copy=True)
    rhs_fixed_np[jj[valid_bot], ii[valid_bot], mbkt[valid_bot]] = bot_fixed[valid_bot]
    fixed_eval = _evaluate_compiled_solver(
        a_live, b_live, c_live, jnp.asarray(rhs_fixed_np), sfc_live,
        w_active, floor, source_rounded, local, wet30, bottom_bc_mask30,
        active_solve30, all30)
    fixed_arrays = fixed_eval.pop("_arrays")

    # Propagate fixed e_new through etau, mixing length, and avm/avt closure.
    ice_eff = nemo_tke_effective_ice_fraction(
        jnp.asarray(full["entry_fr_i"]), int(cfg.eice))
    e_after_tke_fixed = np.asarray(jax.device_get(jax.jit(
        lambda e_in: nemo_etau_injection(
            e_in,
            jnp.asarray(local["entry_taum"]),
            jnp.asarray(local["entry_gdepw_kmm"][..., 1:-1]),
            cfg, rho_0=1026.0,
            lat_deg=jnp.asarray(card.recipe.grid.native_lat_T_deg),
            ice_frac=ice_eff,
            wmask=w_active,
            surface_tmask=jnp.asarray(full["entry_tmask"][..., 0]),
        ))(jnp.asarray(fixed_arrays["e_new"]))))
    stack_f, lup_f, ldown_f, final_f, eps_f = _length_boundaries(
        local, full, cfg, e_after_tke_fixed)
    raw_k_f = np.asarray(jax.jit(
        lambda e, length: cfg.c_k * length * jnp.sqrt(e))(
            jnp.asarray(e_after_tke_fixed), jnp.asarray(final_f[..., 1:])))
    pre_f = np.maximum(cfg.kappaH_min, raw_k_f)
    avtb_1d = np.asarray(records[0]["fields"]["entry_avtb"][1:-1], dtype=np.float64)
    avt_f = np.maximum(
        local["tke_pdlr"][..., 1:-1] * pre_f,
        local["entry_avtb_2d"][..., None] * avtb_1d[None, None, :],
    ) * full["tke_wmask"][..., 1:-1]
    fixed_downstream = {
        "after_tke": _score_with_ranks(e_after_tke_fixed, local["after_tke_en"][..., 1:-1], wet29),
        "pre_sweep_mxl": _score_with_ranks(stack_f, local["pre_sweep_mxl"][..., :-1], wet30),
        "after_lup": _score_with_ranks(lup_f, local["after_lup_mxld"][..., :-1], wet30),
        "after_ldown": _score_with_ranks(ldown_f, local["after_ldown_mxl"][..., :-1], wet30),
        "final_mxl": _score_with_ranks(final_f, local["final_mxl"][..., :-1], wet30),
        "final_mxld": _score_with_ranks(eps_f, local["final_mxld"][..., :-1], wet30),
        "zav": _score_with_ranks(raw_k_f, cfg.c_k * local["final_mxl"][..., 1:-1]
                                 * np.sqrt(local["after_tke_en"][..., 1:-1]), wet29),
        "pre_pdlr_avt": _score_with_ranks(pre_f, local["pre_pdlr_avt"][..., 1:-1], wet29),
        "final_avt": _score_with_ranks(avt_f, local["final_avt"][..., 1:-1], wet29),
    }

    return {
        "recorded_matrix_rhs_inputs": recorded_eval,
        "production_card_live_inputs": live_eval,
        "bottom_dirichlet_audit": {
            "in_solve_seafloor_columns": int(np.count_nonzero(valid_bot)),
            "unfixed_production_vm3": _score_with_ranks(bot_unfixed, nemo_bot_2d, valid_bot),
            "proposed_north_fold_vmask_fix": _score_with_ranks(bot_fixed, nemo_bot_2d, valid_bot),
        },
        "proposed_fix_solver_and_downstream": {
            "solver": fixed_eval,
            "downstream": fixed_downstream,
        },
    }



def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record-dir", required=True, type=Path)
    p.add_argument("--deck-root", required=True, type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--plant")
    args = p.parse_args()
    try:
        result = measure(args.record_dir, args.deck_root, args.plant)
    except (WalkError, admission.GateError, OSError, ValueError) as error:
        print(f"REFUSE: {error}")
        return 2
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
