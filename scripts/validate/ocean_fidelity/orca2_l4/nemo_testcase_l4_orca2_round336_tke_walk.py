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
