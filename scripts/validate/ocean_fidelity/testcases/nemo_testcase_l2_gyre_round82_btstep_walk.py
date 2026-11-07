#!/usr/bin/env python3
"""Compare the admitted kt=2 external step with the production-JIT trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46  # noqa: E402
import nemo_testcase_l2_gyre_round78_uamid_walk as round78  # noqa: E402
import nemo_testcase_l2_gyre_round81_btstep_gate as round81  # noqa: E402
import nemo_testcase_l2_gyre_year_owners as year_owners  # noqa: E402
from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy,
    get_policy,
    set_policy,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from legoesm.ocean.dynamics import (  # noqa: E402
    ocean_model_latlon_cgrid as model_module,
)

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
RECORD_COMMIT = "295a42edc9d9f45707a5349097e7a0183f57463c"
STAGE_ROOT = ROOT / "round46/oracle_kt2_stage"
ENTRY_ROOT = ROOT / "round75/oracle_advmean_kt2"
STAGE_CLOSURE_ROOT = ROOT / "round94/oracle_stage_closure"
PREREG_FINAL_SSH_MAX = np.float64(7.072560112143626e-7)
DEVELOPED_RECORD_COMMIT = "4be746b6f8022672458a8349624cbb2f65b497f2"
DEVELOPED_ENTRY_STEP = 1080
DEVELOPED_PROCESS_STEP = 1081
DEVELOPED_RESTART_SHA256 = (
    "6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976")
DEVELOPED_PROCESS_SHA256 = (
    "526d1fc73faeda990c661f2363a5bb328168bae17d4aea315daf05c35b4cd7b0")
DEVELOPED_RECORD_SHA256 = (
    "6bc0f990ccba183a48ad09549b72b549603694e917389eb4f8b9c03c88ceaf09")
DEVELOPED_QCO_SHA256 = (
    "626d21e229f7ced8f606f6385e04224fb2e086cef92d81d95dff7e2ef3e90878")

TRACE_KEYS = {
    "u_b": "u_history_b",
    "v_b": "v_history_b",
    "eta_b": "eta_history_b",
    "u_bb": "u_history_bb",
    "v_bb": "v_history_bb",
    "eta_bb": "eta_history_bb",
    "depth_u_mid": "transport_face_depth_u",
    "depth_v_mid": "transport_face_depth_v",
    "transport_u": "transport_metric_u",
    "transport_v": "transport_metric_v",
    "ssh_forcing": "continuity_forcing",
    "continuity_div": "continuity_divergence",
    "swap_u": "u_exit",
    "swap_v": "v_exit",
    "swap_eta": "eta_continuity",
}

SOURCE_ORDER = (
    "mid_coefficient_1", "mid_coefficient_2", "mid_coefficient_3",
    "u_entry", "v_entry", "eta_entry",
    "u_b", "v_b", "eta_b", "u_bb", "v_bb", "eta_bb",
    "u_mid", "v_mid", "eta_mid",
    "depth_u_mid", "depth_v_mid", "transport_u", "transport_v",
    "ssh_forcing", "continuity_div", "eta_continuity",
    "back_coefficient_0", "back_coefficient_1",
    "back_coefficient_2", "back_coefficient_3", "eta_pgf",
    "pgf_u", "pgf_v", "cor_u", "cor_v",
    "drag_coefficient_u", "drag_coefficient_v",
    "inverse_depth_u", "inverse_depth_v", "trd_u", "trd_v",
    "slow_u", "slow_v", "u_exit", "v_exit",
    "swap_u", "swap_v", "swap_eta",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def first_non_bit(rows: list[dict]) -> dict | None:
    """Return the first row in external-substep then compiled-source order."""
    for row in rows:
        for boundary in SOURCE_ORDER:
            result = row[boundary]
            if not result["bit_exact"]:
                return {"substep": row["substep"], "boundary": boundary, **result}
    return None


def _stagger(name: str) -> str:
    if name.startswith(("u_", "depth_u_")) or name.endswith("_u"):
        return "u"
    if name.startswith(("v_", "depth_v_")) or name.endswith("_v"):
        return "v"
    return "t"


def _native(trace, record_name: str) -> np.ndarray:
    key = TRACE_KEYS.get(record_name, record_name)
    return gate._trace_native(trace[key], key)


def _execute_step(model, state, dt, freshwater, surface, execution_mode: str,
                  *, eta_after_override=None):
    """Run the complete production closure with or without production JIT."""
    require(
        execution_mode in ("production-jit", "production-eager"),
        f"unknown execution mode {execution_mode!r}",
    )
    if execution_mode == "production-jit":
        return model.step(
            state, dt, freshwater=freshwater, surface_forcing=surface,
            _nemo_stage1_zad_eta_after_override=eta_after_override)
    state = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value)
        if isinstance(value, (np.ndarray, np.generic)) else value,
        state,
    )
    freshwater = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value)
        if isinstance(value, (np.ndarray, np.generic)) else value,
        freshwater,
    )
    surface = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value)
        if isinstance(value, (np.ndarray, np.generic)) else value,
        surface,
    )
    state = model._seed_tke_preclosure_carry(state)
    model.prime_step_caches(state)
    with jax.disable_jit():
        return model_module.LatLonCGridOceanModel._step_jitted.__wrapped__(
            model, state, dt, freshwater, surface,
            _nemo_stage1_zad_eta_after_override=eta_after_override)


def _pytree_identity(left, right) -> dict:
    """Bit census for every array leaf of two otherwise identical pytrees."""
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    require(left_tree == right_tree, "pytree structures differ")
    require(len(left_leaves) == len(right_leaves), "pytree leaf counts differ")
    unequal = 0
    cells = 0
    maximum = 0.0
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        require(left_array.shape == right_array.shape,
                "pytree leaf extents differ")
        require(left_array.dtype == right_array.dtype,
                "pytree leaf dtypes differ")
        left_bytes = np.ascontiguousarray(left_array).view(np.uint8).reshape(
            left_array.size, left_array.dtype.itemsize)
        right_bytes = np.ascontiguousarray(right_array).view(np.uint8).reshape(
            right_array.size, right_array.dtype.itemsize)
        unequal += int(np.count_nonzero(np.any(left_bytes != right_bytes, axis=1)))
        cells += int(left_array.size)
        if np.issubdtype(left_array.dtype, np.inexact) and left_array.size:
            maximum = max(
                maximum,
                float(np.max(np.abs(
                    left_array.astype(np.float64)
                    - right_array.astype(np.float64)), initial=0.0)),
            )
    return {
        "bit_exact": unequal == 0,
        "differing_cells": unequal,
        "cells": cells,
        "leaf_count": len(left_leaves),
        "absolute_max": maximum,
    }


def _context(args):
    """Build the inherited kt=2 entry, then use the shared capture path."""
    base, context = round78.round72._capture_seeded_context(args)
    card, seeded, freshwater, surface, _ = context
    return _capture_external_context(
        args, card, seeded, freshwater, surface, base=base)


def _capture_external_context(args, card, seeded, freshwater, surface, *,
                              base=None, eta_after_override=None,
                              traced_hooks=None, plain_hooks=None,
                              require_state_identity=True):
    """Capture all external substeps in one complete production closure."""
    captures = []
    real_barotropic = model_module.barotropic_substeps_latlon_cgrid

    def sink(eta, uu_b, vv_b, hu_avg, hv_avg, tree, *leaves):
        captures.append({
            "eta": np.asarray(eta),
            "uu_b": np.asarray(uu_b),
            "vv_b": np.asarray(vv_b),
            "Hu_avg": np.asarray(hu_avg),
            "Hv_avg": np.asarray(hv_avg),
            "substeps": jax.tree_util.tree_unflatten(
                tree, [np.asarray(value) for value in leaves]),
        })

    def capture_barotropic(*values, **kwargs):
        require(not kwargs.get("_nemo_substep_trace_test_hook", False),
                "full-step capture received a second substep-trace request")
        result = real_barotropic(
            *values, **dict(kwargs, _nemo_substep_trace_test_hook=True))
        state_after, (hu_avg, hv_avg), substeps = result
        leaves, tree = jax.tree_util.tree_flatten(substeps)
        jax.debug.callback(
            lambda eta, uu_b, vv_b, hu, hv, *got: sink(
                eta, uu_b, vv_b, hu, hv, tree, *got),
            state_after.eta.data, state_after.uu_b.data,
            state_after.vv_b.data, hu_avg, hv_avg, *leaves,
            ordered=True,
        )
        return state_after, (hu_avg, hv_avg)

    traced_model = model_module.LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=traced_hooks,
    )
    traced_model.prime_step_caches(seeded)
    model_module.barotropic_substeps_latlon_cgrid = capture_barotropic
    jax.clear_caches()
    try:
        traced_state = jax.device_get(_execute_step(
            traced_model, seeded, card.dt_s, freshwater, surface,
            args.execution_mode, eta_after_override=eta_after_override))
        jax.effects_barrier()
    finally:
        model_module.barotropic_substeps_latlon_cgrid = real_barotropic
        jax.clear_caches()
    require(captures, "full-step barotropic callback did not fire")
    for duplicate in captures[1:]:
        require(_pytree_identity(duplicate, captures[0])["bit_exact"],
                "full-step barotropic callbacks differ")
    plain_model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=plain_hooks)
    plain_model.prime_step_caches(seeded)
    plain_state = jax.device_get(_execute_step(
        plain_model, seeded, card.dt_s, freshwater, surface,
        args.execution_mode, eta_after_override=eta_after_override))
    trace_identity = _pytree_identity(traced_state, plain_state)
    if require_state_identity:
        require(trace_identity["bit_exact"],
                "full-step substep capture moved the returned production state")
    captured = captures[0]
    trace = SimpleNamespace(
        state_after=traced_state,
        plain_state=plain_state,
        trace_state_identity=trace_identity,
        substeps=captured["substeps"],
        state_after_barotropic=SimpleNamespace(
            eta=SimpleNamespace(data=captured["eta"]),
            uu_b=SimpleNamespace(data=captured["uu_b"]),
            vv_b=SimpleNamespace(data=captured["vv_b"]),
        ),
        transport_average=(captured["Hu_avg"], captured["Hv_avg"]),
    )
    return base, card, seeded, freshwater, surface, trace


def _isolated_jit_replay(fields: dict) -> dict[str, np.ndarray]:
    """JIT the record-local written expressions; this is not production."""
    names = round81.ARRAY_FIELDS

    @jax.jit
    def replay(mid, back, dt_s, *values):
        arrays = dict(zip(names, values, strict=True))
        u_mid = ((mid[:, 0, None, None] * arrays["u_entry"]
                  + mid[:, 1, None, None] * arrays["u_b"])
                 + mid[:, 2, None, None] * arrays["u_bb"])
        v_mid = ((mid[:, 0, None, None] * arrays["v_entry"]
                  + mid[:, 1, None, None] * arrays["v_b"])
                 + mid[:, 2, None, None] * arrays["v_bb"])
        eta_mid = ((mid[:, 0, None, None] * arrays["eta_entry"]
                    + mid[:, 1, None, None] * arrays["eta_b"])
                   + mid[:, 2, None, None] * arrays["eta_bb"])
        eta_continuity = (
            arrays["eta_entry"]
            - dt_s * (arrays["ssh_forcing"] + arrays["continuity_div"])
        ) * jnp.asarray(fields["t_mask"])[None, ...]
        eta_pgf = (
            back[:, 0, None, None] * eta_continuity
            + back[:, 1, None, None] * arrays["eta_entry"]
            + back[:, 2, None, None] * arrays["eta_b"]
            + back[:, 3, None, None] * arrays["eta_bb"]
        )
        trd_u = arrays["cor_u"] + (
            arrays["drag_coefficient_u"] * arrays["u_entry"]
            * arrays["inverse_depth_u"])
        trd_v = arrays["cor_v"] + (
            arrays["drag_coefficient_v"] * arrays["v_entry"]
            * arrays["inverse_depth_v"])
        u_exit = (
            arrays["u_entry"]
            + dt_s * (arrays["pgf_u"] + trd_u + arrays["slow_u"])
        ) * jnp.asarray(fields["u_mask"])[None, ...]
        v_exit = (
            arrays["v_entry"]
            + dt_s * (arrays["pgf_v"] + trd_v + arrays["slow_v"])
        ) * jnp.asarray(fields["v_mask"])[None, ...]
        return {
            "u_mid": u_mid, "v_mid": v_mid, "eta_mid": eta_mid,
            "eta_continuity": eta_continuity, "eta_pgf": eta_pgf,
            "trd_u": trd_u, "trd_v": trd_v,
            "u_exit": u_exit, "v_exit": v_exit,
            "swap_u": u_exit, "swap_v": v_exit,
            "swap_eta": eta_continuity,
        }

    values = tuple(jnp.asarray(fields[name]) for name in names)
    result = replay(
        jnp.asarray(fields["mid_coefficients"]),
        jnp.asarray(fields["back_coefficients"]),
        jnp.asarray(fields["dt_s"]),
        *values,
    )
    return {name: np.asarray(value) for name, value in jax.device_get(result).items()}


def _isolated_rows(fields: dict, active: dict[str, np.ndarray]) -> list[dict]:
    replay = _isolated_jit_replay(fields)
    rows = []
    for name, candidate in replay.items():
        mask = np.broadcast_to(active[_stagger(name)], candidate.shape)
        row = round78.comparison(candidate, fields[name], mask)
        row["boundary"] = name
        rows.append(row)
    return rows


def _stage_impact(args, card, seeded, freshwater, surface, external,
                  oracle_eta: np.ndarray, masks: dict) -> dict:
    """Score the one-variable N+1-SSH handoff through kt2 stage 3."""
    require(args.execution_mode == "production-jit",
            "stage impact is certified only under production JIT")
    def run(override):
        hooks = model_module._NEMOWSRK3TestHooks(
            expose_live_stage_operands=True,
            stage_barotropic_output_override=override,
        )
        model = model_module.LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord,
            card.recipe.model_config, _nemo_ws_test_hooks=hooks)
        model.prime_step_caches(seeded)
        return jax.device_get(_execute_step(
            model, seeded, card.dt_s, freshwater, surface,
            "production-jit"))

    ordinary = run(None)
    live_tuple = (
        jnp.asarray(oracle_eta), ordinary.barotropic_targets[0],
        ordinary.barotropic_targets[1], ordinary.barotropic_targets[2],
        ordinary.barotropic_targets[3],
    )
    directed = run(live_tuple)
    stage = round46.read_stage(
        args.stage_root / "oracle_momstage_kt00000002_s3.bin",
        expected_kt=2, expected_stage=3)
    next_entry = gate.read_entry(
        args.entry_root / "oracle_step_entry_kt00000003.bin")
    header = stage["header"]
    transport = gate.read_transport(
        args.stage_closure_root / "oracle_tracer_transport_kt00000002_s3.bin",
        3, expected_kt=2,
        expected_slots=tuple(
            header[name] for name in ("Kbb", "Kmm", "Kaa", "Krhs")),
    )
    area = np.asarray(card.recipe.grid.area_T)

    def values(trace):
        geometry = trace.stage_geometry[2]
        output = trace.stage_outputs[2]
        return {
            "zFu": np.asarray(geometry[7])[:, 1:, :],
            "zFv": np.asarray(geometry[8])[1:, :, :],
            "zFw": np.asarray(geometry[2]) * area[..., None],
            "T": np.asarray(output[2]),
            "S": np.asarray(output[3]),
        }

    refs = {
        "zFu": np.asarray(transport["zFu"])[..., :30],
        "zFv": np.asarray(transport["zFv"])[..., :30],
        "zFw": np.asarray(transport["zFw"])[..., :31],
        "T": np.asarray(next_entry["T"])[..., :30],
        "S": np.asarray(next_entry["S"])[..., :30],
    }
    active = {
        "zFu": masks["u"], "zFv": masks["v"],
        "zFw": round46._owned3(stage["arrays"]["wmask"], 31) > 0.5,
        "T": masks["T"], "S": masks["S"],
    }
    trace_handoff = {
        "ssh": np.asarray(external.state_after_barotropic.eta.data),
        "uu_b": np.asarray(external.state_after_barotropic.uu_b.data)[:, 1:],
        "vv_b": np.asarray(external.state_after_barotropic.vv_b.data)[1:, :],
        "Hu_avg": np.asarray(external.transport_average[0])[:, 1:],
        "Hv_avg": np.asarray(external.transport_average[1])[1:, :],
    }
    ordinary_handoff = {
        "ssh": np.asarray(ordinary.barotropic_targets[4]),
        "uu_b": np.asarray(ordinary.barotropic_targets[0])[:, 1:],
        "vv_b": np.asarray(ordinary.barotropic_targets[1])[1:, :],
        "Hu_avg": np.asarray(ordinary.barotropic_targets[2])[:, 1:],
        "Hv_avg": np.asarray(ordinary.barotropic_targets[3])[1:, :],
    }
    directed_handoff = {
        "ssh": np.asarray(directed.barotropic_targets[4]),
        "uu_b": np.asarray(directed.barotropic_targets[0])[:, 1:],
        "vv_b": np.asarray(directed.barotropic_targets[1])[1:, :],
        "Hu_avg": np.asarray(directed.barotropic_targets[2])[:, 1:],
        "Hv_avg": np.asarray(directed.barotropic_targets[3])[1:, :],
    }
    handoff_masks = {
        "ssh": masks["ssh"],
        "uu_b": masks["u"][..., 0],
        "vv_b": masks["v"][..., 0],
        "Hu_avg": masks["u"][..., 0],
        "Hv_avg": masks["v"][..., 0],
    }
    trace_handoff_identity = {
        name: round78.comparison(
            ordinary_handoff[name], trace_handoff[name], handoff_masks[name])
        for name in trace_handoff
    }
    directed_identity = {
        name: round78.comparison(
            directed_handoff[name],
            oracle_eta if name == "ssh" else ordinary_handoff[name],
            handoff_masks[name])
        for name in trace_handoff
    }
    require(all(row["bit_exact"] for row in directed_identity.values()),
            "record-directed arm changed more than the registered SSH field")
    plain_vs_stage_trace = _pytree_identity(
        ordinary.state_after, external.plain_state)
    require(plain_vs_stage_trace["bit_exact"],
            "live-stage trace moved the returned production state")
    ordinary_values = values(ordinary)
    directed_values = values(directed)
    rows = {}
    moved = {}
    for name in ("zFu", "zFv", "zFw", "T", "S"):
        require(
            ordinary_values[name].shape == directed_values[name].shape
            == refs[name].shape == active[name].shape,
            f"stage-impact {name} extents disagree",
        )
        rows[name] = {
            "ordinary": round78.comparison(
                ordinary_values[name], refs[name], active[name]),
            "record_directed_ssh": round78.comparison(
                directed_values[name], refs[name], active[name]),
        }
        moved[name] = round78.comparison(
            directed_values[name], ordinary_values[name], active[name])
    return {
        "intervention": (
            "replace only weighted external N+1 SSH; retain live uu_b, vv_b, "
            "Hu_avg, and Hv_avg"),
        "full_step_substep_trace_vs_stage_trace_handoff": (
            trace_handoff_identity),
        "full_step_substep_trace_state_vs_plain": (
            external.trace_state_identity),
        "stage_trace_state_vs_plain": plain_vs_stage_trace,
        "directed_handoff_identity": directed_identity,
        "rows": rows,
        "directed_vs_ordinary": moved,
    }


def _handoff_plant_control(args, oracle_eta: np.ndarray,
                           active_t: np.ndarray) -> dict:
    """One-ULP final-SSH control through the shared production QCO helper."""
    stage = round46.read_stage(
        args.stage_root / "oracle_momstage_kt00000002_s3.bin",
        expected_kt=2, expected_stage=3)
    arrays = stage["arrays"]

    def owned3(name):
        return round46._owned3(arrays[name], 31)

    e3u_0 = owned3("e3u_0")
    e3v_0 = owned3("e3v_0")
    umask = owned3("umask")
    vmask = owned3("vmask")
    hu_0 = np.sum(e3u_0 * umask, axis=-1, dtype=np.float64)
    hv_0 = np.sum(e3v_0 * vmask, axis=-1, dtype=np.float64)
    area_t = round46._owned2(arrays["e1e2t"])
    area_u = round46._owned2(arrays["e1e2u"])
    area_v = round46._owned2(arrays["e1e2v"])
    require(oracle_eta.shape == active_t.shape == area_t.shape,
            "handoff plant T-cell extents disagree")
    location = tuple(int(value) for value in np.argwhere(
        active_t & np.isfinite(oracle_eta))[0])
    planted = np.array(oracle_eta, copy=True)
    before_bits = int(planted[location].view(np.uint64))
    planted[location] = np.nextafter(
        planted[location], np.float64(np.inf))
    after_bits = int(planted[location].view(np.uint64))

    @jax.jit
    def ratio(eta):
        result = model_module.nemo_qco_live_face_geometry_from_operands(
            eta, e3u_0, e3v_0, umask, vmask, hu_0, hv_0,
            area_t, area_u, area_v)
        return result.r3u

    ordinary_ratio = np.asarray(jax.device_get(ratio(jnp.asarray(oracle_eta))))
    planted_ratio = np.asarray(jax.device_get(ratio(jnp.asarray(planted))))
    active_u = np.any(umask > 0.5, axis=-1)
    ssh_row = round78.comparison(planted, oracle_eta, active_t)
    ratio_row = round78.comparison(planted_ratio, ordinary_ratio, active_u)
    return {
        "location": list(location),
        "before_bits": before_bits,
        "after_bits": after_bits,
        "ssh_delta": ssh_row,
        "derived_r3u_delta": ratio_row,
        "fires": bool(
            ssh_row["differing_cells"] == 1
            and ratio_row["differing_cells"] > 0),
    }


def _face_replay_from_live_cells(trace) -> tuple[np.ndarray, np.ndarray]:
    """Replay compiled dyn_drg_init's two face averages from its live input."""
    cell = np.asarray(trace["drag_coefficient_t"], dtype=np.float64)
    if cell.ndim == 3:
        cell = cell[0]
    require(cell.ndim == 2, "live cell drag coefficient is not two-dimensional")
    u_inner = np.float64(0.5) * (np.roll(cell, 1, axis=1) + cell)
    u_full = np.concatenate([u_inner, u_inner[:, :1]], axis=1)
    v_inner = np.float64(0.5) * (cell[:-1, :] + cell[1:, :])
    v_full = np.pad(v_inner, ((1, 1), (0, 0)), mode="edge")
    return u_full[:, 1:], v_full[1:, :]


def _admit(args) -> dict:
    admission = json.loads(args.admission.read_text())
    require(admission["verdict"] == "PASS", "Round-81 twin admission failed")
    require(
        (
            admission["byte_identical_records"],
            len(admission["classified_changed_records"]),
            admission["admitted_difference_count"],
        ) == (46, 24, 264),
        "Round-81 inherited-record census changed",
    )
    producer = (args.record_root / "producer_commit.txt").read_text().strip()
    require(producer == RECORD_COMMIT, "Round-81 producer commit changed")
    record = args.record_root / round81.RECORD
    stamp = record.with_name(record.name + ".stamp")
    require(
        stamp.read_text().split() == [round81.sha256(record), producer, record.name],
        "Round-81 record stamp mismatch",
    )
    require(record.stat().st_size == round81.EXPECTED_SIZE, "Round-81 record size changed")
    fields = round81.read_record(record)
    require(
        all(
            result["bit_exact"]
            for row in round81.validate_fields(fields)
            for name, result in row.items()
            if name != "substep"
        ),
        "Round-81 arithmetic replay changed",
    )
    uamid = round81.compare_uamid(fields, args.uamid_root / round81.UAMID_RECORD)
    require(all(row["bit_exact"] for row in uamid.values()), "Round-77 U identity changed")
    return fields


def _developed_comparison(actual, expected, mask) -> dict:
    """Bit and norm census with signed-zero sensitivity and a first index."""
    actual = np.array(actual, dtype=np.float64, copy=True, order="C")
    expected = np.array(expected, dtype=np.float64, copy=True, order="C")
    mask = np.asarray(mask, dtype=bool)
    require(actual.shape == expected.shape == mask.shape,
            "developed comparison extents differ: "
            f"actual={actual.shape}, expected={expected.shape}, "
            f"mask={mask.shape}")
    unequal = (actual.view(np.uint64) != expected.view(np.uint64)) & mask
    delta = actual[mask] - expected[mask]
    first = ([int(value) for value in np.argwhere(unequal)[0]]
             if np.any(unequal) else None)
    return {
        "bit_exact": not bool(np.any(unequal)),
        "cells_scored": int(np.count_nonzero(mask)),
        "differing_cells": int(np.count_nonzero(unequal)),
        "absolute_max": float(np.max(np.abs(delta), initial=0.0)),
        "rms": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "first_unequal_index": first,
    }


def _developed_unequal_mask(actual, expected, mask) -> np.ndarray:
    actual = np.array(actual, dtype=np.float64, copy=True, order="C")
    expected = np.array(expected, dtype=np.float64, copy=True, order="C")
    mask = np.asarray(mask, dtype=bool)
    require(actual.shape == expected.shape == mask.shape,
            "developed mismatch-mask extents differ")
    return (actual.view(np.uint64) != expected.view(np.uint64)) & mask


def _validate_developed_registry(boundaries=SOURCE_ORDER) -> None:
    require(tuple(boundaries) == SOURCE_ORDER,
            "developed external boundary registry changed")
    require(len(SOURCE_ORDER) == len(set(SOURCE_ORDER)),
            "developed external boundary registry contains duplicates")
    require(set(TRACE_KEYS) <= set(round81.ARRAY_FIELDS),
            "developed trace mapping names an unrecorded field")


def _verify_manifest(root: Path) -> dict:
    manifest = root / "round137_outputs.sha256"
    require(manifest.is_file(), "Round-137 closed SHA manifest is missing")
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split()
        path = root / name
        require(path.is_file(), f"Round-137 manifest member is missing: {name}")
        require(round81.sha256(path) == digest,
                f"Round-137 manifest member changed: {name}")
        entries[name] = digest
    return entries


def _admit_developed(args) -> tuple[dict, dict, dict]:
    """Re-admit the developed external/QCO pair and passive twin controls."""
    root = args.record_root
    producer = (root / "producer_commit.txt").read_text().strip()
    require(producer == DEVELOPED_RECORD_COMMIT,
            "Round-137 producer commit changed")
    manifest = _verify_manifest(root)
    record_path = root / round81.DEVELOPED_RECORD
    qco_path = root / round81.DEVELOPED_QCO_RECORD
    require(record_path.stat().st_size == round81.EXPECTED_DEVELOPED_SIZE,
            "developed external record size changed")
    require(qco_path.stat().st_size == round81.EXPECTED_QCO_SIZE,
            "developed QCO record size changed")
    require(round81.sha256(record_path) == DEVELOPED_RECORD_SHA256,
            "developed external record digest changed")
    require(round81.sha256(qco_path) == DEVELOPED_QCO_SHA256,
            "developed QCO record digest changed")
    for path, digest in ((record_path, DEVELOPED_RECORD_SHA256),
                         (qco_path, DEVELOPED_QCO_SHA256)):
        stamp = path.with_name(path.name + ".stamp")
        require(stamp.read_text().split() == [digest, producer, path.name],
                f"developed record stamp changed: {stamp.name}")

    fields = round81.read_record(
        record_path, expected_kt=DEVELOPED_PROCESS_STEP,
        has_final_pssh=True)
    qco = round81.read_qco_record(
        qco_path, expected_kt=DEVELOPED_PROCESS_STEP)
    replay = round81.validate_fields(fields)
    boundary = round81.validate_developed_boundary(fields, qco)
    require(all(value["bit_exact"] for value in boundary.values()),
            "Round-137 developed boundary admission changed")

    restart = root / "GYRE_OMIP_L2_P3_00001080_restart.nc"
    process = root / "oracle_process_budget_kt00001081.bin"
    require(round81.sha256(restart) == DEVELOPED_RESTART_SHA256,
            "Round-137 passive restart identity changed")
    require(round81.sha256(process) == DEVELOPED_PROCESS_SHA256,
            "Round-137 passive process identity changed")
    ocean_lines = [line.strip() for line in (
        root / "ocean.output").read_text().splitlines() if line.strip()]
    require(ocean_lines and ocean_lines[-1] == "AAAAAAAA",
            "Round-137 NEMO ocean log lacks its terminal completion marker")
    plants = {}
    plant_files = {
        "final-pssh-ulp": "final-pssh-ulp",
        "header": "header",
        "passive-admission": "passive_admission",
        "qco-ulp": "qco-ulp",
        "replay-ulp": "replay-ulp",
        "stamp": "stamp",
        "swap-ulp": "swap-ulp",
        "truncation": "truncation",
    }
    for name, file_stem in plant_files.items():
        path = root / f"round137_{file_stem}_plant.log"
        require(path.is_file(), f"Round-137 {name} plant log is missing")
        marker = f"STATUS PLANT-FIRED: {name}"
        require(path.read_text().splitlines()[-1] == marker,
                f"Round-137 {name} plant marker changed")
        plants[name] = marker
    require(manifest[record_path.name] == DEVELOPED_RECORD_SHA256,
            "closed manifest external digest changed")
    require(manifest[qco_path.name] == DEVELOPED_QCO_SHA256,
            "closed manifest QCO digest changed")
    return fields, qco, {
        "producer_commit": producer,
        "record_sha256": DEVELOPED_RECORD_SHA256,
        "qco_record_sha256": DEVELOPED_QCO_SHA256,
        "restart_sha256": DEVELOPED_RESTART_SHA256,
        "process_sha256": DEVELOPED_PROCESS_SHA256,
        "record_local_replay_rows": replay,
        "developed_boundary": boundary,
        "plants": plants,
        "closed_manifest_members": len(manifest),
    }


def _developed_inputs(args):
    """Map the admitted step-1080 restart onto the certified GYRE card."""
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        _u_east_to_face, _v_north_to_face)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    year = year_owners._year()
    audit_json = json.loads(args.daily_audit.read_text())
    audit_commit = audit_json["tool"]["commit"]
    audit, daily_hashes = year._daily_record_contract(
        args.daily_audit, args.daily_root, audit_commit)
    restart_path = year._daily_restart_path(
        args.daily_root, DEVELOPED_ENTRY_STEP)
    payload = year._load_daily_reset_payload(restart_path, 30)
    expected_digest = daily_hashes.get(restart_path.name)
    require(expected_digest is not None,
            "step-1080 restart is absent from the daily SHA manifest")
    year._require_payload_digest(payload, expected_digest)
    require(payload["sha256"] == DEVELOPED_RESTART_SHA256,
            "daily and Round-137 step-1080 restarts differ")

    card = build_nemo_testcase_card(year_owners.CASE)
    state, _ = year_owners._gyre_checkpoint_state(card, restart_path)
    history = (
        jnp.asarray(year._face_history(payload["ub_e"], "u")),
        jnp.asarray(year._face_history(payload["ubb_e"], "u")),
        jnp.asarray(year._face_history(payload["vb_e"], "v")),
        jnp.asarray(year._face_history(payload["vbb_e"], "v")),
        jnp.asarray(payload["sshb_e"]),
        jnp.asarray(payload["sshbb_e"]),
    )
    state = state._replace(bt_hist=history)
    land = np.asarray(state.land_mask.data) > 0.5
    checks = {
        "T": year_owners._bit_mismatch_count(
            state.T.data, payload["tn"], gate.expected_masks(card)["T"]),
        "S": year_owners._bit_mismatch_count(
            state.S.data, payload["sn"], gate.expected_masks(card)["T"]),
        "u": year_owners._bit_mismatch_count(
            state.u.data, _u_east_to_face(payload["un"])),
        "v": year_owners._bit_mismatch_count(
            state.v.data, _v_north_to_face(payload["vn"])),
        "eta": year_owners._bit_mismatch_count(
            state.eta.data, payload["sshn"]),
    }
    for index, (actual, expected) in enumerate(zip(
            state.bt_hist, history, strict=True)):
        checks[f"bt_hist_{index}"] = year_owners._bit_mismatch_count(
            actual, expected)
    expected_tke = np.where(land[..., None], payload["en"][..., 1:], 0.0)
    expected_avm = np.where(land[..., None], payload["avm_k"][..., 1:], 0.0)
    expected_avt = np.where(land[..., None], payload["avt_k"][..., 1:], 0.0)
    expected_dissl = np.where(
        land[..., None], payload["dissl"][..., 1:], 0.0)
    expected_surface = np.where(land, payload["avm_k"][..., 0], 0.0)
    for name, expected in (
            ("tke", expected_tke), ("tke_avm", expected_avm),
            ("tke_avt", expected_avt), ("tke_dissl", expected_dissl),
            ("tke_avm_surface", expected_surface)):
        checks[name] = year_owners._bit_mismatch_count(
            getattr(state, name).data, expected)
    require(all(value == 0 for value in checks.values()),
            "developed restart bridge changed a mapped field: "
            + ", ".join(f"{name}={value}" for name, value in checks.items()
                        if value))
    freshwater, surface = gate._surface_forcings(
        card, state, DEVELOPED_PROCESS_STEP)
    return card, state, freshwater, surface, payload, {
        "daily_audit": str(args.daily_audit),
        "daily_audit_commit": audit_commit,
        "daily_record_count": audit["inventory"]["observed_count"],
        "restart": str(restart_path),
        "restart_sha256": payload["sha256"],
        "source_mapping_cells_unequal": checks,
    }


def _developed_rows(fields, trace, active) -> tuple[list[dict], dict]:
    live_arrays = {
        name: np.asarray(_native(trace, name), dtype=np.float64)
        for name in round81.ARRAY_FIELDS
    }
    live_mid = np.stack(
        [np.asarray(trace[f"mid_weight_{index}"], dtype=np.float64)
         for index in (1, 2, 3)], axis=-1)
    live_back = np.stack(
        [np.asarray(trace[f"back_weight_{index}"], dtype=np.float64)
         for index in (0, 1, 2, 3)], axis=-1)
    rows = []
    scalar = np.ones((), dtype=bool)
    for substep in range(round81.N_CYCLE):
        row = {"substep": substep + 1}
        for index in range(3):
            row[f"mid_coefficient_{index + 1}"] = _developed_comparison(
                live_mid[substep, index],
                fields["mid_coefficients"][substep, index], scalar)
        for name in round81.ARRAY_FIELDS[:19]:
            row[name] = _developed_comparison(
                live_arrays[name][substep], fields[name][substep],
                active[_stagger(name)])
        for index in range(4):
            row[f"back_coefficient_{index}"] = _developed_comparison(
                live_back[substep, index],
                fields["back_coefficients"][substep, index], scalar)
        for name in round81.ARRAY_FIELDS[19:]:
            row[name] = _developed_comparison(
                live_arrays[name][substep], fields[name][substep],
                active[_stagger(name)])
        require(set(row) == {"substep", *SOURCE_ORDER},
                f"substep {substep + 1} omitted a registered boundary")
        rows.append(row)
    return rows, live_arrays


def _process_qco_arm(card, state, freshwater, surface, eta_override,
                     transport_average, zad_eta, active, qco) -> dict:
    """Direct one external endpoint through the production-JIT QCO path."""
    hooks = model_module._NEMOWSRK3TestHooks(
        external_mode_result_override=(
            jnp.asarray(eta_override),
            jnp.asarray(transport_average[0]),
            jnp.asarray(transport_average[1])),
        tracer_process_trace=(),
    )
    trace_model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    trace_model.prime_step_caches(state)
    trace = jax.device_get(trace_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=jnp.asarray(zad_eta)))
    plain_hooks = hooks._replace(tracer_process_trace=None)
    plain_model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=plain_hooks)
    plain_model.prime_step_caches(state)
    plain = jax.device_get(plain_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=jnp.asarray(zad_eta)))
    observer = _pytree_identity(trace.state_after, plain)
    require(observer["bit_exact"],
            "production QCO observer changed the directed returned state")
    q_actual = np.asarray(trace.q_Kaa, dtype=np.float64)
    owned_r1 = qco["r1_ht_0"][
        round81.NTSJ - 1:round81.NTEJ,
        round81.NTSI - 1:round81.NTEI]
    expected = np.float64(1.0) + np.asarray(eta_override) * owned_r1
    return {
        "observer_state_identity": observer,
        "consumed_q_Kaa_vs_one_plus_input_times_recorded_r1_ht_0": (
            _developed_comparison(q_actual, expected, active)),
        "q_Kaa": q_actual,
    }


def _standard_process_qco(card, state, freshwater, surface, zad_eta) -> dict:
    hooks = model_module._NEMOWSRK3TestHooks(tracer_process_trace=())
    model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    model.prime_step_caches(state)
    return jax.device_get(model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface,
        _nemo_stage1_zad_eta_after_override=jnp.asarray(zad_eta)))


def measure_developed(args) -> dict:
    """Measure the Round-137 day-180 external solve in compiled order."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64 libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-138 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-138 measurement commit mismatch")
    if args.plant == "missing-boundary":
        fired = False
        try:
            _validate_developed_registry(SOURCE_ORDER[:-1])
        except RuntimeError:
            fired = True
        require(fired, "missing-boundary plant did not trip the registry")
        return {"format": "gyre-developed-external-walk-v1",
                "status": "PLANT-FIRED", "plant": args.plant,
                "plant_fires": True, "worktree": stamp}
    _validate_developed_registry()
    fields, qco, admission = _admit_developed(args)
    card, state, freshwater, surface, payload, entry = _developed_inputs(args)
    masks = gate.expected_masks(card)
    active = {
        "t": masks["ssh"],
        "u": masks["u"][..., 0],
        "v": masks["v"][..., 0],
    }
    for name, expected in (("t_mask", active["t"]),
                           ("u_mask", active["u"]),
                           ("v_mask", active["v"])):
        require(np.array_equal(fields[name] != 0.0, expected),
                f"{name} differs from the developed live mask")

    if args.plant == "entry-ssh-ulp":
        location = tuple(int(value) for value in np.argwhere(active["t"])[0])
        planted_eta = np.array(state.eta.data, copy=True)
        planted_eta[location] = np.nextafter(
            planted_eta[location], np.float64(np.inf))
        planted_state = state._replace(
            eta=state.eta.replace(data=jnp.asarray(planted_eta)))
        _, _, _, _, _, planted = _capture_external_context(
            args, card, planted_state, freshwater, surface,
            eta_after_override=jnp.asarray(payload["ssha"]))
        entry_row = _developed_comparison(
            _native(planted.substeps, "eta_entry")[0],
            fields["eta_entry"][0], active["t"])
        require(entry_row["differing_cells"] > 0,
                "entry-SSH ULP plant moved no registered entry row")
        return {"format": "gyre-developed-external-walk-v1",
                "status": "PLANT-FIRED", "plant": args.plant,
                "plant_fires": True, "location": list(location),
                "entry_row": entry_row, "worktree": stamp}

    (_, _, _, _, _, captured) = _capture_external_context(
        args, card, state, freshwater, surface,
        eta_after_override=jnp.asarray(payload["ssha"]))
    rows, live_arrays = _developed_rows(fields, captured.substeps, active)
    first = first_non_bit(rows)
    # The callback's barotropic state retains its solver halo, whereas the
    # registered substep trace and NEMO stream both use the owned native box.
    # The final swap is the exact pssh field returned by that solve.
    live_final = np.asarray(live_arrays["swap_eta"][-1], dtype=np.float64)
    oracle_final = np.array(fields["final_pssh"], copy=True)
    if args.plant == "final-pssh-ulp":
        location = tuple(int(value) for value in np.argwhere(active["t"])[0])
        oracle_final[location] = np.nextafter(
            oracle_final[location], np.float64(np.inf))
    final_row = _developed_comparison(live_final, oracle_final, active["t"])
    if args.plant == "final-pssh-ulp":
        baseline = _developed_comparison(
            live_final, fields["final_pssh"], active["t"])
        require(final_row != baseline,
                "final-pssh ULP plant moved no registered final row")
        return {"format": "gyre-developed-external-walk-v1",
                "status": "PLANT-FIRED", "plant": args.plant,
                "plant_fires": True, "final_row": final_row,
                "baseline_final_row": baseline, "worktree": stamp}

    final_mask = _developed_unequal_mask(
        live_final, fields["final_pssh"], active["t"])
    first_appearance = np.zeros(active["t"].shape, dtype=np.int16)
    ssh_first_non_bit = None
    for substep in range(round81.N_CYCLE):
        mismatch = _developed_unequal_mask(
            live_arrays["eta_continuity"][substep],
            fields["eta_continuity"][substep], active["t"])
        if ssh_first_non_bit is None and np.any(mismatch):
            ssh_first_non_bit = substep + 1
        first_appearance[(first_appearance == 0) & mismatch] = substep + 1
    appearances = [
        {"j": int(j), "i": int(i),
         "first_unequal_substep": int(first_appearance[j, i])}
        for j, i in np.argwhere(final_mask)
    ]

    qco_proof = None
    qco_set_equal = None
    if args.execution_mode == "production-jit":
        standard = _standard_process_qco(
            card, state, freshwater, surface, payload["ssha"])
        standard_identity = _pytree_identity(
            standard.state_after, captured.plain_state)
        require(standard_identity["bit_exact"],
                "process-QCO observer changed the ordinary production state")
        owned_r3ta = qco["r3ta"][
            round81.NTSJ - 1:round81.NTEJ,
            round81.NTSI - 1:round81.NTEI]
        nemo_q = np.float64(1.0) + owned_r3ta
        standard_q = np.asarray(standard.q_Kaa, dtype=np.float64)
        standard_mask = _developed_unequal_mask(
            standard_q, nemo_q, active["t"])
        qco_set_equal = bool(np.array_equal(final_mask, standard_mask))
        model_arm = _process_qco_arm(
            card, state, freshwater, surface, live_final,
            captured.transport_average, payload["ssha"], active["t"], qco)
        nemo_arm = _process_qco_arm(
            card, state, freshwater, surface, fields["final_pssh"],
            captured.transport_average, payload["ssha"], active["t"], qco)
        nemo_vs_record = _developed_comparison(
            nemo_arm.pop("q_Kaa"), nemo_q, active["t"])
        model_arm.pop("q_Kaa")
        qco_proof = {
            "standard_observer_state_identity": standard_identity,
            "standard_q_Kaa_vs_recorded_NEMO_one_plus_r3ta": (
                _developed_comparison(standard_q, nemo_q, active["t"])),
            "standard_q_Kaa_unequal_set_equals_final_pssh_set": qco_set_equal,
            "model_pssh_directed_arm": model_arm,
            "NEMO_pssh_directed_arm": {
                **nemo_arm,
                "consumed_q_Kaa_vs_recorded_NEMO_one_plus_r3ta": (
                    nemo_vs_record),
            },
        }

    predictions = {
        "final_pssh_599_columns": final_row["differing_cells"] == 599,
        "final_pssh_set_equals_round136_q_Kaa_set": qco_set_equal is True,
        "first_boundary_is_substep1_slow_momentum": bool(
            first is not None and first["substep"] == 1
            and first["boundary"] in ("slow_u", "slow_v")),
        "substep1_ssh_is_bit": rows[0]["eta_continuity"]["bit_exact"],
        "ssh_first_diff_no_earlier_than_substep2": bool(
            ssh_first_non_bit is not None and ssh_first_non_bit >= 2),
        "all_final_columns_appear_inside_substeps": bool(
            appearances and all(
                row["first_unequal_substep"] > 0 for row in appearances)),
    }
    if args.execution_mode == "production-eager":
        predictions["final_pssh_set_equals_round136_q_Kaa_set"] = None
    return {
        "format": "gyre-developed-external-walk-v1",
        "status": "MEASURED",
        "worktree": stamp,
        "execution_regime": args.execution_mode + "-cpu-fp64-x64-libm",
        "entry": entry,
        "admission": admission,
        "observer_state_identity": captured.trace_state_identity,
        "first_non_bit_statement": first,
        "first_ssh_non_bit_substep": ssh_first_non_bit,
        "weighted_final_pssh_vs_NEMO": final_row,
        "final_pssh_unequal_columns": [
            [int(j), int(i)] for j, i in np.argwhere(final_mask)],
        "final_pssh_column_first_appearance": appearances,
        "qco_exact_input_proof": qco_proof,
        "rows": rows,
        "predictions": predictions,
        "all_applicable_frozen_predictions_confirmed": all(
            value for value in predictions.values() if value is not None),
        "plant": args.plant,
        "plant_fires": False,
        "scope": {
            "production_physics_changed": False,
            "DINO": "NO-PRODUCTION-CHANGE",
            "LOCK_EXCHANGE": "NO-PRODUCTION-CHANGE",
            "OVERFLOW": "NO-PRODUCTION-CHANGE",
            "ORCA2": "UNMEASURED-WITH-SPEC; diagnostic-only round",
        },
    }


def measure(args) -> dict:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
        "precision policy is not fp64 libm",
    )
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-116 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-116 measurement commit mismatch")

    fields = _admit(args)
    (base, card, seeded, freshwater, surface,
     captured) = _context(args)
    require(base["status"] == "MEASURED", "inherited kt=2 seeded context changed")
    trace = captured.substeps
    masks = gate.expected_masks(card)
    active = {
        "t": masks["ssh"],
        "u": masks["u"][..., 0],
        "v": masks["v"][..., 0],
    }
    for name, expected in (
        ("t_mask", active["t"]),
        ("u_mask", active["u"]),
        ("v_mask", active["v"]),
    ):
        require(
            np.array_equal(fields[name] != 0.0, expected),
            f"{name} differs from the live native mask",
        )

    live_arrays = {
        name: np.asarray(_native(trace, name), dtype=np.float64)
        for name in round81.ARRAY_FIELDS
    }
    oracle_arrays = {
        name: np.array(fields[name], copy=True) for name in round81.ARRAY_FIELDS
    }
    live_mid = np.stack(
        [np.asarray(trace[f"mid_weight_{index}"], dtype=np.float64)
         for index in (1, 2, 3)], axis=-1)
    live_back = np.stack(
        [np.asarray(trace[f"back_weight_{index}"], dtype=np.float64)
         for index in (0, 1, 2, 3)], axis=-1)
    oracle_mid = np.array(fields["mid_coefficients"], copy=True)
    oracle_back = np.array(fields["back_coefficients"], copy=True)
    replay_drag_u, replay_drag_v = _face_replay_from_live_cells(trace)
    for name in round81.ARRAY_FIELDS:
        require(
            live_arrays[name].shape == oracle_arrays[name].shape,
            f"{name} live/oracle shape mismatch: "
            f"{live_arrays[name].shape} != {oracle_arrays[name].shape}",
        )
    require(
        live_mid.shape == oracle_mid.shape,
        f"mid coefficient shape mismatch: {live_mid.shape} != {oracle_mid.shape}",
    )
    require(
        live_back.shape == oracle_back.shape,
        f"back coefficient shape mismatch: {live_back.shape} != {oracle_back.shape}",
    )

    next_entry = gate.read_entry(
        args.entry_root / "oracle_step_entry_kt00000003.bin")
    oracle_final_eta = np.asarray(next_entry["ssh"], dtype=np.float64)
    live_final_eta = np.asarray(
        captured.state_after_barotropic.eta.data, dtype=np.float64)
    require(oracle_final_eta.shape == live_final_eta.shape == active["t"].shape,
            "weighted final SSH/next-entry extents disagree")
    final_ssh = round78.comparison(
        live_final_eta, oracle_final_eta, active["t"])

    plant_detail = None
    handoff_plant = None
    ordinary_slow_u = round78.comparison(
        live_arrays["slow_u"][0], oracle_arrays["slow_u"][0], active["u"])
    if args.plant == "history-ulp":
        location = (0, *tuple(np.argwhere(active["u"])[0]))
        oracle_arrays["u_b"][location] = np.nextafter(
            oracle_arrays["u_b"][location], np.float64(np.inf))
        plant_detail = {
            "field": "u_b", "location": [int(index) for index in location]}
    elif args.plant == "slow-u-ulp":
        delta = np.where(
            active["u"],
            np.abs(live_arrays["slow_u"][0] - oracle_arrays["slow_u"][0]),
            -np.inf,
        )
        location = (0, *np.unravel_index(np.argmax(delta), delta.shape))
        direction = (
            np.float64(np.inf)
            if oracle_arrays["slow_u"][location] >= live_arrays["slow_u"][location]
            else np.float64(-np.inf)
        )
        oracle_arrays["slow_u"][location] = np.nextafter(
            oracle_arrays["slow_u"][location], direction)
        plant_detail = {
            "field": "slow_u", "location": [int(index) for index in location]}
    elif args.plant == "null-slow-u":
        ordinary = round78.comparison(
            live_arrays["slow_u"][0], oracle_arrays["slow_u"][0], active["u"])
        require(not ordinary["bit_exact"], "null slow-U plant target is already exact")
        oracle_arrays["slow_u"][0] = live_arrays["slow_u"][0]
        plant_detail = {"field": "slow_u", "substep": 1}
    elif args.plant == "handoff-ssh-ulp":
        require(args.execution_mode == "production-jit",
                "handoff plant requires the production JIT")
        handoff_plant = _handoff_plant_control(
            args, oracle_final_eta, active["t"])
        plant_detail = {
            "field": "weighted_final_ssh",
            "location": handoff_plant["location"],
            "before_bits": handoff_plant["before_bits"],
            "after_bits": handoff_plant["after_bits"],
        }

    scalar_mask = np.ones((), dtype=bool)
    rows = []
    for substep in range(round81.N_CYCLE):
        row = {
            "substep": substep + 1,
            **{
                f"mid_coefficient_{index + 1}": round78.comparison(
                    live_mid[substep, index], oracle_mid[substep, index], scalar_mask)
                for index in range(3)
            },
            **{
                name: round78.comparison(
                    live_arrays[name][substep], oracle_arrays[name][substep],
                    active[_stagger(name)],
                )
                for name in round81.ARRAY_FIELDS[:19]
            },
            **{
                f"back_coefficient_{index}": round78.comparison(
                    live_back[substep, index], oracle_back[substep, index], scalar_mask)
                for index in range(4)
            },
            **{
                name: round78.comparison(
                    live_arrays[name][substep], oracle_arrays[name][substep],
                    active[_stagger(name)],
                )
                for name in round81.ARRAY_FIELDS[19:]
            },
        }
        rows.append(row)

    first = first_non_bit(rows)
    final_alignment_confirmed = bool(
        final_ssh["differing_cells"] == 600
        and abs(final_ssh["absolute_max"] - PREREG_FINAL_SSH_MAX)
        <= np.float64(1.0e-12)
    )
    if args.execution_mode == "production-jit":
        prediction_confirmed = bool(
            args.plant == "none"
            and first is not None
            and first["substep"] == 1
            and first["boundary"] in {"slow_u", "slow_v"}
            and final_alignment_confirmed
        )
    else:
        # P1 preregisters only the final alignment for the eager control; its
        # first boundary is explicitly allowed to expose a different rounding.
        prediction_confirmed = bool(
            args.plant == "none" and final_alignment_confirmed)
    plant_fires = False
    if args.plant == "history-ulp":
        plant_fires = bool(first and first["substep"] == 1 and first["boundary"] == "u_b")
    elif args.plant == "slow-u-ulp":
        plant_fires = rows[0]["slow_u"] != ordinary_slow_u
    elif args.plant == "null-slow-u":
        plant_fires = rows[0]["slow_u"]["bit_exact"]
    elif args.plant == "handoff-ssh-ulp":
        plant_fires = bool(handoff_plant and handoff_plant["fires"])

    isolated_rows = _isolated_rows(fields, active)
    stage_impact = None
    if args.execution_mode == "production-jit" and args.plant == "none":
        stage_impact = _stage_impact(
            args, card, seeded, freshwater, surface, captured,
            oracle_final_eta, masks)

    return {
        "format": "nemo-testcase-l2-gyre-round116-btstep-walk-v2",
        "status": "CONFIRMED" if prediction_confirmed else "REFUTED",
        "worktree": stamp,
        "execution_regime": (
            args.execution_mode + "-cpu-fp64-x64-libm"),
        "record_commit": RECORD_COMMIT,
        "record_sha256": round81.sha256(args.record_root / round81.RECORD),
        "record_size": round81.EXPECTED_SIZE,
        "admission_counts": {"exact": 46, "total": 70, "changed": 24, "admitted": 264},
        "dtype": {"record": str(fields["u_entry"].dtype),
                  "live": str(live_arrays["u_entry"].dtype)},
        "first_non_bit_statement": first,
        "final_alignment_confirmed": final_alignment_confirmed,
        "weighted_final_ssh_vs_next_entry": final_ssh,
        "drag_face_statement_on_live_cell_input": {
            "u": round78.comparison(
                replay_drag_u, oracle_arrays["drag_coefficient_u"][0], active["u"]),
            "v": round78.comparison(
                replay_drag_v, oracle_arrays["drag_coefficient_v"][0], active["v"]),
        },
        "isolated_jit_replay": {
            "label": "isolated-closure JIT; not production",
            "rows": isolated_rows,
        },
        "stage3_record_directed_ssh_impact": stage_impact,
        "rows": rows,
        "plant": args.plant,
        "plant_detail": plant_detail,
        "handoff_plant_control": handoff_plant,
        "plant_fires": plant_fires,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--developed", action="store_true",
        help="score the admitted Round-137 step-1081 developed record")
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument(
        "--record-root", type=Path,
        default=ROOT / "round81/oracle_btstep_kt2")
    parser.add_argument(
        "--uamid-root", type=Path,
        default=ROOT / "round77/oracle_uamid_kt2")
    parser.add_argument(
        "--admission", type=Path,
        default=ROOT / "round81/oracle_btstep_kt2/round81_admission.json")
    parser.add_argument("--stage-root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--entry-root", type=Path, default=ENTRY_ROOT)
    parser.add_argument(
        "--stage-closure-root", type=Path, default=STAGE_CLOSURE_ROOT)
    parser.add_argument(
        "--execution-mode",
        choices=("production-jit", "production-eager"),
        default="production-jit")
    parser.add_argument(
        "--plant", choices=(
            "none", "history-ulp", "slow-u-ulp", "null-slow-u",
            "handoff-ssh-ulp", "missing-boundary", "entry-ssh-ulp",
            "final-pssh-ulp"),
        default="none")
    parser.add_argument(
        "--daily-root", type=Path,
        default=ROOT / "round132/oracle_daily_restarts")
    parser.add_argument(
        "--daily-audit", type=Path,
        default=ROOT / "round136/daily_record_audit.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = measure_developed(args) if args.developed else measure(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (RuntimeError, AssertionError) as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    if args.developed:
        if args.plant != "none":
            print(
                f"ROUND138 {args.plant.upper()} PLANT STATUS "
                f"{'PLANT-FIRED' if report['plant_fires'] else 'PLANT-INERT'}"
            )
            return 1
        print(
            "ROUND138 DEVELOPED BTSTEP MEASURED: "
            f"first={report['first_non_bit_statement']} "
            f"final={report['weighted_final_pssh_vs_NEMO']}"
        )
        return 0
    if args.plant != "none":
        print(
            f"ROUND116 {args.plant.upper()} PLANT STATUS "
            f"{'PLANT-FIRED' if report['plant_fires'] else 'PLANT-INERT'}"
        )
        return 1
    print(
        f"ROUND116 BTSTEP {report['status']}: "
        f"first={report['first_non_bit_statement']}"
    )
    return 0 if report["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
