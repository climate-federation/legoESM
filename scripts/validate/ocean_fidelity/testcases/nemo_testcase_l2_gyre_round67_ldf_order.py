#!/usr/bin/env python3
"""Round-67 ordered Krhs substitutions and pre-edit LDF-routing prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate
import nemo_testcase_l2_gyre_round14_advmean as round14
import nemo_testcase_l2_gyre_round54_tracer_decomposition as round54
import nemo_testcase_l2_gyre_round66_content_operands as round66
import nemo_testcase_l2_gyre_round111_fct_writer_gate as round111
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ocean import advection as advection_module
from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
TRACERS = ("T", "S")
ROUND112_ROWS = (
    "first_u", "first_v", "first_w", "first_div", "midpoint",
    "average_u", "average_v", "average_w", "final_div", "rhs_after",
)
ROUND113_INPUTS = (
    "base", "transport_u", "transport_v", "transport_w",
    "h_kbb", "h_kmm", "tmask", "wmask", "r1_area", "p2dt",
)
ROUND114_U_INPUTS = (
    "e2u", "e3u_kmm", "one_plus_r3u", "u_kmm", "un_adv",
    "r1_hu_kmm", "uu_b_kmm", "umask", "zub", "corrected_u", "zfu",
)
ROUND114_DIRECT_ORDER = (
    "e2u", "one_plus_r3u", "e3u_kmm", "u_kmm", "un_adv",
    "uu_b_kmm", "zfu",
)
ROUND115_TRACE_FIELDS = (
    "west_product", "east_product", "parenthesized_sum", "half_sum",
    "depth_product", "r3u", "one_plus_r3u",
)
ROUND115_OBSERVATION_FIELDS = (
    "full_step_ssh", "half_step_ssh", "live_area_t", "live_r1_hu0",
    "live_r1_area_u",
    "ordinary_one_plus_r3u", "live_full_r3u", "live_half_r3u",
    *ROUND115_TRACE_FIELDS,
    "oracle_full_r3u", "oracle_interpolated_half_r3u",
    "oracle_interpolated_half_one_plus_r3u",
)


class _Round113EagerBoundaryComplete(RuntimeError):
    """Internal stop after both eager production FCT calls were observed."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def _array_sha256(value) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _copy_operands(values: dict) -> dict:
    return {
        name: (np.array(value, dtype=np.float64, copy=True)
               if np.ndim(value) else np.float64(value))
        for name, value in values.items()
    }


def pytree_exact_census(got, want) -> dict:
    """Count exact differences over every array leaf of two state pytrees."""
    got_leaves, got_tree = jax.tree_util.tree_flatten(got)
    want_leaves, want_tree = jax.tree_util.tree_flatten(want)
    require(got_tree == want_tree, "state pytree structures differ")
    require(len(got_leaves) == len(want_leaves), "state leaf counts differ")
    cells = 0
    unequal = 0
    max_abs = 0.0
    for got_leaf, want_leaf in zip(got_leaves, want_leaves, strict=True):
        got_array = np.asarray(got_leaf)
        want_array = np.asarray(want_leaf)
        require(got_array.shape == want_array.shape,
                "corresponding state leaf shapes differ")
        cells += got_array.size
        unequal += int(np.count_nonzero(got_array != want_array))
        if got_array.size and np.issubdtype(got_array.dtype, np.number):
            max_abs = max(
                max_abs,
                float(np.max(np.abs(
                    got_array.astype(np.float64)
                    - want_array.astype(np.float64)))))
    return {
        "leaves": len(got_leaves),
        "cells": cells,
        "cells_unequal": unequal,
        "max_abs": max_abs,
    }


def worsening_census(
    baseline: np.ndarray,
    candidate: np.ndarray,
    oracle: np.ndarray,
    wet: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Count candidate cells worsening by more than two row-scale fp64 ULPs."""
    scale = np.float64(max(float(np.max(np.abs(oracle[wet]))), 1.0))
    ulp = np.spacing(scale)
    degradation = np.abs(candidate - oracle) - np.abs(baseline - oracle)
    worsened = np.asarray(wet, dtype=bool) & (degradation > 2.0 * ulp)
    return {
        "row_scale_ulp": float(ulp),
        "cells_worsened": int(np.count_nonzero(worsened)),
        "max_worsening_ulps": float(
            max(0.0, float(np.max(degradation[wet]))) / ulp),
    }, worsened


def cumulative_substitutions(
    *, live: dict, oracle: dict, oracle_content: np.ndarray,
    boundaries: dict[str, np.ndarray], source: np.ndarray,
    qsr: np.ndarray, ldf: np.ndarray, wet: np.ndarray,
    routed_krhs: np.ndarray | None = None,
) -> dict:
    """Replace cumulative Krhs boundaries in NEMO's compiled order."""
    live_adv = live["Krhs"] - source
    live_sbc = source - qsr
    components = {
        "advection": (live_adv, boundaries["after_adv"]),
        "sbc": (live_sbc, boundaries["after_sbc"] - boundaries["after_adv"]),
        "qsr": (qsr, boundaries["after_qsr"] - boundaries["after_sbc"]),
        "ldf": (ldf, boundaries["after_ldf"] - boundaries["after_qsr"]),
    }
    arms = {
        "routed_live": (
            live["Krhs"] + ldf if routed_krhs is None else routed_krhs),
        "nemo_post_sbc": boundaries["after_sbc"] + qsr + ldf,
        "nemo_post_qsr": boundaries["after_qsr"] + ldf,
        "nemo_post_ldf": boundaries["after_ldf"],
    }
    component_rows = {
        name: round54.field_stats(got, want, wet)
        for name, (got, want) in components.items()
    }
    arm_rows = {}
    arm_content = {}
    for name, krhs in arms.items():
        operands = dict(live)
        operands["Krhs"] = krhs
        content = round66.content_statement(operands)
        arm_content[name] = content
        arm_rows[name] = {
            "krhs": round54.field_stats(krhs, oracle["Krhs"], wet),
            "content": round54.field_stats(content, oracle_content, wet),
        }
    return {
        "component_rows": component_rows,
        "arm_rows": arm_rows,
        "arm_content": arm_content,
    }


def _one_ulp_plant(content: np.ndarray, wet: np.ndarray) -> dict:
    planted = np.array(content, dtype=np.float64, copy=True)
    first_wet = tuple(np.argwhere(wet)[0])
    planted[first_wet] = np.nextafter(
        planted[first_wet], np.float64(np.inf))
    row = round54.field_stats(planted, content, wet)
    require(row["cells_unequal"] == 1 and row["max_abs"] > 0.0,
            "one-ULP cumulative-content plant did not fire")
    return row


def production_fct_content(
    advection_content: np.ndarray,
    source: np.ndarray,
    ldf: np.ndarray,
    operands: dict,
) -> np.ndarray:
    """Use the production FCT content and NEMO's Kmm source association."""
    return (
        advection_content
        + operands["p2dt"] * operands["e3t_Kmm"] * (source + ldf)
    )


def _round112_owned_record(path: Path, *, plant: bool = False) -> dict:
    """Map the passive R111 record onto legoESM's owned C-grid domain."""
    record = round111.read_record(path)
    fields = record["fields"]

    def value(name):
        return np.asarray(fields[name]["values"], dtype=np.float64)

    def owned3(name, islice, jslice):
        return np.ascontiguousarray(value(name)[islice, jslice].transpose(1, 0, 2))

    def owned2(name, islice, jslice):
        return np.ascontiguousarray(value(name)[islice, jslice, 0].T)

    bundle = {
        "sha256": record["sha256"],
        "p2dt": np.float64(value("p2dt").item()),
        "p_u": owned3("transport_u", slice(1, 34), slice(2, 24)),
        "p_v": owned3("transport_v", slice(2, 34), slice(1, 24)),
        "p_w": np.pad(
            owned3("transport_w", slice(1, 33), slice(1, 23)),
            ((0, 0), (0, 0), (0, 1))),
        "e3t": owned3("e3t_3d", slice(1, 33), slice(1, 23)),
        "r3t_kbb": owned2("r3t_Kbb", slice(1, 33), slice(1, 23)),
        "r3t_kmm": owned2("r3t_Kmm", slice(1, 33), slice(1, 23)),
        "tmask": owned3("tmask", slice(1, 33), slice(1, 23)),
        "wmask": owned3("wmask", slice(1, 33), slice(1, 23)),
        "r1_area": owned2("r1_e1e2t", slice(1, 33), slice(1, 23)),
        "base_T": owned3("base_T", slice(2, 34), slice(2, 24)),
        "base_S": owned3("base_S", slice(2, 34), slice(2, 24)),
    }
    expected = {}
    mappings = {
        "first_u": (slice(1, 34), slice(2, 24)),
        "first_v": (slice(2, 34), slice(1, 24)),
        "first_w": (slice(1, 33), slice(1, 23)),
        "first_div": (slice(1, 33), slice(1, 23)),
        "midpoint": (slice(1, 33), slice(1, 23)),
        "average_u": (slice(None), slice(1, 23)),
        "average_v": (slice(1, 33), slice(None)),
        "average_w": (slice(1, 33), slice(1, 23)),
        "final_div": (slice(None), slice(None)),
        "rhs_after": (slice(None), slice(None)),
    }
    for tracer in TRACERS:
        for name, (islice, jslice) in mappings.items():
            expected[(tracer, name)] = np.ascontiguousarray(
                value(f"{name}_{tracer}")[islice, jslice].transpose(1, 0, 2))
    bundle["expected"] = expected
    bundle["h_kbb"] = np.ascontiguousarray(
        bundle["e3t"]
        * (np.float64(1.0)
           + bundle["r3t_kbb"][..., None] * bundle["tmask"]))
    bundle["h_kmm"] = np.ascontiguousarray(
        bundle["e3t"]
        * (np.float64(1.0)
           + bundle["r3t_kmm"][..., None] * bundle["tmask"]))

    planted = np.array(bundle["p_u"], copy=True)
    base = bundle["base_T"]
    left = np.roll(base, 1, axis=1)
    left = np.concatenate([left, left[:, :1]], axis=1)
    right = np.concatenate([base, base[:, :1]], axis=1)
    plant_row = None
    for index in np.argwhere(np.isfinite(planted) & (planted != 0.0)):
        at = tuple(int(item) for item in index)
        old = planted[at]
        new = np.nextafter(old, np.float64(np.inf))
        q = left[at] if old > 0.0 else right[at]
        if new * q != old * q:
            plant_row = {
                "field": "p_u", "index": list(at),
                "before_bits": int(np.asarray(old, dtype="=f8").view("=u8")),
                "after_bits": int(np.asarray(new, dtype="=f8").view("=u8")),
            }
            if plant:
                planted[at] = new
            break
    require(plant_row is not None,
            "no finite nonzero p_u word propagates a one-ULP plant")
    if plant:
        bundle["p_u"] = planted
    bundle["plant_row"] = plant_row
    return bundle


def _round112_face_neighbours(field: jax.Array) -> tuple:
    west = jnp.roll(field, 1, axis=1)
    return (
        jnp.concatenate([west, west[:, :1]], axis=1),
        jnp.concatenate([field, field[:, :1]], axis=1),
    )


def _round112_vface_neighbours(field: jax.Array) -> tuple:
    padded = jnp.pad(field, ((1, 1), (0, 0), (0, 0)))
    return padded[:-1], padded[1:]


def _round112_literal_rows(base: jax.Array, values: dict) -> tuple:
    """Literal compiled fct_up1_2stp statements, with source stores."""
    sr = nemo_source_round
    p_u, p_v, p_w = values["p_u"], values["p_v"], values["p_w"]
    e3t = values["e3t"]
    r3t_kbb, r3t_kmm = values["r3t_kbb"], values["r3t_kmm"]
    tmask, wmask = values["tmask"], values["wmask"]
    r1_area, p2dt = values["r1_area"], values["p2dt"]
    zero = jnp.asarray(0.0, dtype=base.dtype)
    one = jnp.asarray(1.0, dtype=base.dtype)
    half = jnp.asarray(0.5, dtype=base.dtype)

    west, east = _round112_face_neighbours(base)
    first_u = sr(
        sr(jnp.maximum(p_u, zero) * west)
        + sr(jnp.minimum(p_u, zero) * east))
    south, north = _round112_vface_neighbours(base)
    first_v = sr(
        sr(jnp.maximum(p_v, zero) * south)
        + sr(jnp.minimum(p_v, zero) * north))
    first_w_int = sr(
        sr(jnp.maximum(p_w[..., 1:30], zero) * base[..., 1:])
        + sr(sr(jnp.minimum(p_w[..., 1:30], zero) * base[..., :-1])
             * wmask[..., 1:30]))
    first_w = jnp.pad(first_w_int, ((0, 0), (0, 0), (1, 1)))

    du = sr(first_u[:, 1:] - first_u[:, :-1])
    dv = sr(first_v[1:] - first_v[:-1])
    dw = sr(first_w[..., :-1] - first_w[..., 1:])
    first_div = sr(-sr(sr(du + dv) + dw) * r1_area[..., None])
    h_kbb = sr(e3t * sr(one + sr(r3t_kbb[..., None] * tmask)))
    h_kmm = sr(e3t * sr(one + sr(r3t_kmm[..., None] * tmask)))
    midpoint = sr(
        sr(sr(sr(h_kbb * base) + sr(sr(half * p2dt) * first_div))
           / h_kmm) * tmask)

    west, east = _round112_face_neighbours(midpoint)
    next_u = sr(
        sr(jnp.maximum(p_u, zero) * west)
        + sr(jnp.minimum(p_u, zero) * east))
    south, north = _round112_vface_neighbours(midpoint)
    next_v = sr(
        sr(jnp.maximum(p_v, zero) * south)
        + sr(jnp.minimum(p_v, zero) * north))
    average_u = sr(half * sr(first_u + next_u))
    average_v = sr(half * sr(first_v + next_v))
    next_w = sr(
        sr(jnp.maximum(p_w[..., 1:30], zero) * midpoint[..., 1:])
        + sr(sr(jnp.minimum(p_w[..., 1:30], zero) * midpoint[..., :-1])
             * wmask[..., 1:30]))
    average_w = jnp.pad(
        sr(half * sr(first_w[..., 1:30] + next_w)),
        ((0, 0), (0, 0), (1, 1)))

    du = sr(average_u[:, 1:] - average_u[:, :-1])
    dv = sr(average_v[1:] - average_v[:-1])
    dw = sr(average_w[..., :-1] - average_w[..., 1:])
    final_div = sr(-sr(sr(du + dv) + dw) * r1_area[..., None])
    rhs_after = sr(sr(sr(final_div / h_kmm) * tmask)
                   + jnp.zeros_like(final_div))
    # XLA may delete the source's addition to the positive-zero Krhs entry,
    # preserving a negative zero from ``negative_divergence * 0`` instead.
    # NEMO's stored dry-cell result is +0; make that signed-zero consequence
    # of the compiled mask explicit without changing any wet value.
    rhs_after = jnp.where(tmask > 0.5, rhs_after, jnp.zeros_like(rhs_after))
    return (first_u, first_v, first_w, first_div, midpoint,
            average_u, average_v, average_w, final_div, rhs_after)


def _round112_current_rows(base: jax.Array, values: dict, grid) -> tuple:
    """Current metric-free FCT representation at the same source boundaries."""
    from legoesm.grids.operators_latlon_cgrid import (
        divergence_cgrid, upwind_cell_to_uface, upwind_cell_to_vface,
    )

    p_u, p_v, p_w = values["p_u"], values["p_v"], values["p_w"]
    tmask = values["tmask"]
    p2dt = values["p2dt"]
    dy_u = jnp.asarray(grid.dy_u)[..., None]
    dx_v = jnp.asarray(grid.dx_v)[..., None]
    area = jnp.asarray(grid.area_T)[..., None]
    mass_u, mass_v, w_full = p_u / dy_u, p_v / dx_v, p_w / area

    up_u = upwind_cell_to_uface(base, mass_u)
    up_v = upwind_cell_to_vface(base, mass_v, grid)
    flux_u = mass_u * up_u
    flux_v = mass_v * up_v
    first_u, first_v = flux_u * dy_u, flux_v * dx_v
    w_int = w_full[..., 1:30]
    first_w_int = w_int * jnp.where(
        w_int > 0.0, base[..., 1:], base[..., :-1])
    metric_first_w = jnp.pad(
        first_w_int, ((0, 0), (0, 0), (1, 1))) * area
    first_w = jnp.pad(first_w_int, ((0, 0), (0, 0), (1, 1)))
    positive_div = (
        divergence_cgrid(flux_u, flux_v, grid)
        + first_w[..., :-1] - first_w[..., 1:])
    first_div = -positive_div

    # These are exact injected NEMO thickness operands; only the FCT source
    # program, not their upstream construction, is under test here.
    h_kbb, h_kmm = values["h_kbb"], values["h_kmm"]
    midpoint = (
        h_kbb * base - (jnp.asarray(0.5, base.dtype) * p2dt) * positive_div
    ) / jnp.maximum(h_kmm, jnp.asarray(1.0e-30, base.dtype))
    midpoint = jnp.where(tmask > 0.5, midpoint, base)

    next_u = mass_u * upwind_cell_to_uface(midpoint, mass_u)
    next_v = mass_v * upwind_cell_to_vface(midpoint, mass_v, grid)
    average_u_raw = 0.5 * (flux_u + next_u)
    average_v_raw = 0.5 * (flux_v + next_v)
    average_u, average_v = average_u_raw * dy_u, average_v_raw * dx_v
    next_w = w_int * jnp.where(
        w_int > 0.0, midpoint[..., 1:], midpoint[..., :-1])
    average_w_int = 0.5 * (first_w_int + next_w)
    average_w_raw = jnp.pad(
        average_w_int, ((0, 0), (0, 0), (1, 1)))
    average_w = average_w_raw * area
    positive_final = (
        divergence_cgrid(average_u_raw, average_v_raw, grid)
        + average_w_raw[..., :-1] - average_w_raw[..., 1:])
    final_div = -positive_final
    rhs_after = (
        final_div / jnp.maximum(h_kmm, jnp.asarray(1.0e-10, base.dtype))
        * tmask)
    return (first_u, first_v, metric_first_w, first_div, midpoint,
            average_u, average_v, average_w, final_div, rhs_after)


def _round112_all_rows(values: dict, grid) -> tuple:
    rows = []
    for source in (_round112_current_rows, _round112_literal_rows):
        for tracer in TRACERS:
            base = values[f"base_{tracer}"]
            rows.extend(
                source(base, values, grid)
                if source is _round112_current_rows
                else source(base, values))
    return tuple(rows)


def _round112_observation_rows(observation: tuple, bundle: dict) -> dict:
    require(len(observation) == 4 * len(ROUND112_ROWS),
            "round-112 callback schema changed")
    rows = {}
    cursor = 0
    for source in ("current", "source_literal"):
        rows[source] = {}
        for tracer in TRACERS:
            rows[source][tracer] = {}
            for name in ROUND112_ROWS:
                got = np.asarray(observation[cursor], dtype=np.float64)
                want = bundle["expected"][(tracer, name)]
                require(got.shape == want.shape,
                        f"round-112 {source} {tracer} {name} shape changed")
                rows[source][tracer][name] = round54.field_stats(
                    got, want, np.ones(want.shape, dtype=bool))
                cursor += 1
    return rows


def _round112_collapse(observations: list[tuple], label: str) -> tuple:
    require(observations, f"round-112 {label} callback did not execute")
    first = observations[0]
    for duplicate in observations[1:]:
        require(len(duplicate) == len(first),
                f"round-112 {label} callback schema changed")
        require(all(np.array_equal(left, right)
                    for left, right in zip(first, duplicate, strict=True)),
                f"round-112 {label} observed distinct duplicate executions")
    return first


def _round113_input_tuple(
    values: tuple, kwargs: dict, *, tracer: str,
) -> tuple:
    """Materialize the actual production FCT inputs in NEMO record units."""
    grid = values[5]
    base = kwargs.get("tracer_before")
    require(base is not None, "round-113 FCT call has no Kbb tracer base")
    active = kwargs.get("active_mask")
    require(active is not None, "round-113 FCT call has no active mask")
    h_kbb = kwargs.get("base_thickness")
    require(h_kbb is not None, "round-113 FCT call has no Kbb thickness")
    return (
        base,
        values[1] * jnp.asarray(grid.dy_u)[..., None],
        values[2] * jnp.asarray(grid.dx_v)[..., None],
        values[3] * jnp.asarray(grid.area_T)[..., None],
        h_kbb,
        values[4],
        active,
        active,
        jnp.asarray(1.0, values[0].dtype) / jnp.asarray(grid.area_T),
        jnp.asarray(values[6], values[0].dtype),
    )


def _round113_patch_inputs(
    values: tuple, kwargs: dict, bundle: dict, tracer: str,
    family: str | None,
) -> tuple[tuple, dict]:
    """Replace one recorded input family at the real production FCT call."""
    if family is None:
        return values, kwargs
    patched = list(values)
    patched_kwargs = dict(kwargs)
    grid = values[5]
    if family == "base":
        patched_kwargs["tracer_before"] = jnp.asarray(bundle[f"base_{tracer}"])
    elif family in ("transport", "transport_u", "transport_v", "transport_w"):
        if family in ("transport", "transport_u"):
            patched[1] = jnp.asarray(bundle["p_u"]) / jnp.asarray(
                grid.dy_u)[..., None]
        if family in ("transport", "transport_v"):
            patched[2] = jnp.asarray(bundle["p_v"]) / jnp.asarray(
                grid.dx_v)[..., None]
        if family in ("transport", "transport_w"):
            patched[3] = jnp.asarray(bundle["p_w"]) / jnp.asarray(
                grid.area_T)[..., None]
    elif family == "thickness":
        patched[4] = jnp.asarray(bundle["h_kmm"])
        patched_kwargs["base_thickness"] = jnp.asarray(bundle["h_kbb"])
    elif family == "mask":
        patched_kwargs["active_mask"] = jnp.asarray(bundle["tmask"])
    elif family == "p2dt":
        patched[6] = jnp.asarray(bundle["p2dt"], values[0].dtype)
    else:
        raise RuntimeError(f"round-113 family {family!r} is not substitutable")
    return tuple(patched), patched_kwargs


def _round113_collapse_calls(
    calls: list[tuple], label: str, *, allow_prior_distinct: bool = False,
) -> tuple:
    """Collapse duplicate JAX callback passes, preserving the T/S pair."""
    require(len(calls) >= 2 and len(calls) % 2 == 0,
            f"round-113 {label} did not emit complete T/S pairs")
    pair = tuple(calls[-2:])
    for start in range(0, len(calls) - 2, 2):
        prior = calls[start:start + 2]
        exact = all(
            len(left) == len(right)
            and all(np.array_equal(a, b)
                    for a, b in zip(left, right, strict=True))
            for left, right in zip(prior, pair, strict=True))
        require(exact or allow_prior_distinct,
                f"round-113 {label} observed distinct duplicate executions")
    return pair


def _round113_score_inputs(observation: tuple, bundle: dict) -> dict:
    require(len(observation) == 2, "round-113 observation is not a T/S pair")
    rows = {}
    common_reference = {
        "transport_u": bundle["p_u"],
        "transport_v": bundle["p_v"],
        "transport_w": bundle["p_w"],
        "h_kbb": bundle["h_kbb"],
        "h_kmm": bundle["h_kmm"],
        "tmask": bundle["tmask"],
        "wmask": bundle["wmask"],
        "r1_area": bundle["r1_area"],
        "p2dt": np.asarray(bundle["p2dt"]),
    }
    tmask = np.asarray(bundle["tmask"] > 0.5)
    u_core = tmask * np.roll(tmask, 1, axis=1)
    u_mask = np.concatenate([u_core, u_core[:, :1]], axis=1)
    v_mask = np.pad(
        tmask[:-1] * tmask[1:], ((1, 1), (0, 0), (0, 0)))
    w_mask = np.zeros(np.asarray(bundle["p_w"]).shape, dtype=bool)
    w_mask[..., 1:30] = np.asarray(bundle["wmask"][..., 1:30] > 0.5)
    masks = {
        "base": tmask,
        "transport_u": u_mask,
        "transport_v": v_mask,
        "transport_w": w_mask,
        "h_kbb": tmask,
        "h_kmm": tmask,
        "tmask": np.ones(tmask.shape, dtype=bool),
        "wmask": np.ones(tmask.shape, dtype=bool),
        "r1_area": tmask[..., 0],
        "p2dt": np.ones((), dtype=bool),
    }
    for tracer, values in zip(TRACERS, observation, strict=True):
        require(len(values) == len(ROUND113_INPUTS) + 1,
                "round-113 callback schema changed")
        rows[tracer] = {}
        for name, got in zip(ROUND113_INPUTS, values[:-1], strict=True):
            want = (bundle[f"base_{tracer}"] if name == "base"
                    else common_reference[name])
            got = np.asarray(got, dtype=np.float64)
            want = np.asarray(want, dtype=np.float64)
            require(got.shape == want.shape,
                    f"round-113 {tracer} {name} shape differs")
            row = round54.field_stats(got, want, masks[name])
            row["full_domain"] = round54.field_stats(
                got, want, np.ones(want.shape, dtype=bool))
            rows[tracer][name] = row
        upstream = np.asarray(values[-1], dtype=np.float64)
        want_upstream = bundle["expected"][(tracer, "rhs_after")]
        rows[tracer]["adv_up1"] = round54.field_stats(
            upstream, want_upstream, tmask)
        rows[tracer]["adv_up1"]["full_domain"] = round54.field_stats(
            upstream, want_upstream,
            np.ones(want_upstream.shape, dtype=bool))
    for index, name in enumerate(ROUND113_INPUTS[1:], start=1):
        require(np.array_equal(observation[0][index], observation[1][index]),
                f"round-113 common input {name} differs between T and S calls")
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round114_owned_u_bundle(
    *, record_root: Path, stage_baseline_root: Path, admission_path: Path,
    expected_commit: str, p_u: np.ndarray, plant: bool = False,
) -> dict:
    """Admit and map direct kt2/stage-3 U operands onto owned faces."""
    admission = json.loads(admission_path.read_text())
    require(admission.get("verdict") == "PASS",
            "round-77 inherited-record admission is not PASS")
    require((admission.get("byte_identical_records"),
             len(admission.get("classified_changed_records", ())),
             admission.get("admitted_difference_count")) == (45, 24, 281),
            "round-77 inherited-record admission census changed")
    require(_file_sha256(admission_path)
            == "cddd662d6380d7aa053bb4cfc20fbcb481a9da43009200dcbaa359e8c4e2f11d",
            "round-77 admission digest changed")
    producer = (record_root / "producer_commit.txt").read_text().strip()
    require(producer.lower() == expected_commit.lower(),
            "round-77 producer commit mismatch")
    require(admission["worktree"]["commit"].lower() == producer.lower(),
            "round-77 admission/worktree commit mismatch")

    stage_path = record_root / "oracle_momstage_kt00000002_s3.bin"
    baseline_path = stage_baseline_root / "oracle_momstage_kt00000002_s3.bin"
    advmean_path = record_root / "oracle_bt_advmean_operands_kt00000002.bin"
    require(_file_sha256(stage_path)
            == "11da7f370ec495181adc80c7fd9daf3300b245ea8db646085122a8f46b2007e8",
            "round-77 kt2 stage-3 record digest changed")
    require(_file_sha256(advmean_path)
            == "412f917a8652002beef2db22af2ef1a916171173b10fc0fb14b62fd6077d95e2",
            "round-77 kt2 barotropic-mean record digest changed")
    require((stage_baseline_root / "producer_commit.txt").read_text().strip()
            == "715c9865008e7e419004ad1e5e643e8b71257c00",
            "round-46 stage-record producer changed")

    record = round66.round46.read_stage(
        stage_path, expected_kt=2, expected_stage=3)
    baseline = round66.round46.read_stage(
        baseline_path, expected_kt=2, expected_stage=3)
    require(record["header"] == baseline["header"],
            "round-46/77 kt2 stage-3 headers differ")
    advmean = round14.read_advmean(
        advmean_path, expected_kt=2, include_halo=True)

    def owned_u3(array):
        return np.ascontiguousarray(
            np.asarray(array, dtype=np.float64)[2:24, 1:34, :30])

    def owned_u2(array):
        return np.ascontiguousarray(
            np.asarray(array, dtype=np.float64)[2:24, 1:34])

    direct_fields = (
        "e2u", "e3u_Kmm", "r3u_Kmm", "u_Kmm", "umask", "uu_b_Kmm",
    )
    inherited_exact = {}
    for name in direct_fields:
        mapper = owned_u3 if np.ndim(record["arrays"][name]) == 3 else owned_u2
        candidate = mapper(record["arrays"][name])
        prior = mapper(baseline["arrays"][name])
        inherited_exact[name] = bool(np.array_equal(candidate, prior))
        require(inherited_exact[name],
                f"round-77 inherited owned stage field {name} moved")

    arrays = record["arrays"]
    e2u = owned_u2(arrays["e2u"])
    e3u = owned_u3(arrays["e3u_Kmm"])
    r3u = owned_u2(arrays["r3u_Kmm"])
    umask = owned_u3(arrays["umask"])
    one_plus_r3u = np.ascontiguousarray(
        np.float64(1.0) + r3u)
    bundle = {
        "producer_commit": producer,
        "stage_sha256": _file_sha256(stage_path),
        "stage_baseline_sha256": _file_sha256(baseline_path),
        "advmean_sha256": _file_sha256(advmean_path),
        "admission_sha256": _file_sha256(admission_path),
        "inherited_owned_exact": inherited_exact,
        "e2u": e2u,
        "e3u_kmm": e3u,
        "one_plus_r3u": one_plus_r3u,
        "u_kmm": owned_u3(arrays["u_Kmm"]),
        "un_adv": np.ascontiguousarray(
            np.asarray(advmean["post_lbc_u"], dtype=np.float64)[2:24, 1:34]),
        "uu_b_kmm": owned_u2(arrays["uu_b_Kmm"]),
        "umask": umask,
        "zfu": np.ascontiguousarray(np.asarray(p_u, dtype=np.float64)),
    }
    expected_shapes = {
        "e2u": (22, 33), "e3u_kmm": (22, 33, 30),
        "one_plus_r3u": (22, 33), "u_kmm": (22, 33, 30),
        "un_adv": (22, 33), "uu_b_kmm": (22, 33),
        "umask": (22, 33, 30), "zfu": (22, 33, 30),
    }
    require(all(bundle[name].shape == shape
                for name, shape in expected_shapes.items()),
            "round-114 mapped U operand shape changed")

    planted = np.array(bundle["e3u_kmm"], copy=True)
    plant_row = None
    for raw_index in np.argwhere(
            (bundle["umask"] > 0.5) & np.isfinite(planted)
            & (planted != 0.0) & (bundle["zfu"] != 0.0)):
        index = tuple(int(item) for item in raw_index)
        old = planted[index]
        new = np.nextafter(old, np.float64(np.inf))
        ji = index[:2]
        old_metric_h = np.float64(bundle["e2u"][ji] * old)
        new_metric_h = np.float64(bundle["e2u"][ji] * new)
        if new_metric_h == old_metric_h or old_metric_h == 0.0:
            continue
        velocity = np.float64(bundle["zfu"][index] / old_metric_h)
        if np.float64(new_metric_h * velocity) == np.float64(
                old_metric_h * velocity):
            continue
        plant_row = {
            "field": "e3u_kmm", "index": list(index),
            "before_bits": int(np.asarray(old, dtype="=f8").view("=u8")),
            "after_bits": int(np.asarray(new, dtype="=f8").view("=u8")),
        }
        if plant:
            planted[index] = new
        break
    require(plant_row is not None,
            "no recorded e3u word propagates a one-ULP metric product")
    if plant:
        bundle["e3u_kmm"] = planted
    bundle["plant_row"] = plant_row
    return bundle


def _round114_u_statement(
    e2u, e3u_kmm, u_kmm, un_adv, r1_hu_kmm, uu_b_kmm, umask,
) -> tuple:
    """Literal compiled zub/corrected-U/zFu source associations."""
    zub = nemo_source_round(
        nemo_source_round(un_adv * r1_hu_kmm) - uu_b_kmm)
    corrected = nemo_source_round(
        u_kmm + nemo_source_round(zub[..., None] * umask))
    zfu = model_module._nemo_metric_stage_transport(e2u, e3u_kmm, corrected)
    return zub, corrected, zfu


def _round114_score_u(observation: tuple, bundle: dict) -> dict:
    require(len(observation) == len(ROUND114_U_INPUTS),
            "round-114 U callback schema changed")
    got = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in zip(ROUND114_U_INPUTS, observation, strict=True)
    }
    references = {
        name: bundle.get(name) for name in ROUND114_U_INPUTS
    }
    wet3 = np.asarray(bundle["umask"] > 0.5)
    wet2 = np.any(wet3, axis=-1)
    masks = {
        "e2u": wet2, "e3u_kmm": wet3, "one_plus_r3u": wet2,
        "u_kmm": wet3, "un_adv": wet2, "r1_hu_kmm": wet2,
        "uu_b_kmm": wet2, "umask": np.ones(wet3.shape, dtype=bool),
        "zub": wet2, "corrected_u": wet3, "zfu": wet3,
    }
    rows = {}
    for name in ROUND114_U_INPUTS:
        reference = references[name]
        if reference is None:
            rows[name] = {
                "classification": "UNMEASURED_WITH_SPEC",
                "reason": (
                    "the admitted records do not directly store this "
                    "statement boundary; no downstream inversion is used"),
                "shape": list(got[name].shape),
            }
            continue
        reference = np.asarray(reference, dtype=np.float64)
        require(got[name].shape == reference.shape,
                f"round-114 U {name} shape differs")
        row = round54.field_stats(got[name], reference, masks[name])
        row["full_domain"] = round54.field_stats(
            got[name], reference, np.ones(reference.shape, dtype=bool))
        row["classification"] = (
            "BIT" if row["cells_unequal"] == 0 else "DEBT")
        rows[name] = row
    first = next(
        (name for name in ROUND114_DIRECT_ORDER
         if rows[name]["classification"] != "BIT"), None)
    return {"rows": rows, "first_direct_nonbit": first}


def _round114_collapse_u(
    calls: list[tuple], label: str, *, allow_prior_distinct: bool = False,
) -> tuple:
    require(calls, f"round-114 {label} U callback did not execute")
    selected = calls[-1]
    for prior in calls[:-1]:
        exact = len(prior) == len(selected) and all(
            np.array_equal(left, right)
            for left, right in zip(prior, selected, strict=True))
        require(exact or allow_prior_distinct,
                f"round-114 {label} observed distinct duplicate U executions")
    return selected


def _round115_u_redundant(value):
    """Map native east-face U values to the model's redundant west-face row."""
    return jnp.concatenate([value[:, -1:], value], axis=1)


def _round115_u_trace(eta, area_t, r1_hu0, r1_area_u) -> tuple:
    """Trace every written boundary of compiled ``dom_qco_r3c_RK3`` U."""
    sr = nemo_source_round
    dtype = jnp.asarray(eta).dtype
    half = jnp.asarray(0.5, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    west = sr(jnp.asarray(area_t, dtype=dtype) * jnp.asarray(eta, dtype=dtype))
    east = jnp.roll(west, -1, axis=1)
    parenthesized = sr(west + east)
    half_sum = sr(half * parenthesized)
    depth_product = sr(half_sum * jnp.asarray(r1_hu0, dtype=dtype))
    r3u = sr(depth_product * jnp.asarray(r1_area_u, dtype=dtype))
    one_plus = sr(one + r3u)
    return west, east, parenthesized, half_sum, depth_product, r3u, one_plus


def _round115_scalar_u_trace(
    eta: np.ndarray,
    area_t: np.ndarray,
    r1_hu0: np.ndarray,
    r1_area_u: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Independent scalar replay of the compiled U expression."""
    eta = np.asarray(eta, dtype=np.float64)
    area_t = np.asarray(area_t, dtype=np.float64)
    r1_hu0 = np.asarray(r1_hu0, dtype=np.float64)
    r1_area_u = np.asarray(r1_area_u, dtype=np.float64)
    require(eta.shape == area_t.shape == r1_hu0.shape == r1_area_u.shape,
            "round-115 scalar U operands must have one native-face shape")
    outputs = tuple(np.empty_like(eta) for _ in ROUND115_TRACE_FIELDS)
    for jj in range(eta.shape[0]):
        for ji in range(eta.shape[1]):
            east_i = (ji + 1) % eta.shape[1]
            west = np.float64(area_t[jj, ji] * eta[jj, ji])
            east = np.float64(area_t[jj, east_i] * eta[jj, east_i])
            parenthesized = np.float64(west + east)
            half_sum = np.float64(np.float64(0.5) * parenthesized)
            depth_product = np.float64(half_sum * r1_hu0[jj, ji])
            r3u = np.float64(depth_product * r1_area_u[jj, ji])
            one_plus = np.float64(np.float64(1.0) + r3u)
            for output, result in zip(
                outputs,
                (west, east, parenthesized, half_sum, depth_product, r3u,
                 one_plus),
                strict=True,
            ):
                output[jj, ji] = result
    return outputs


def _round115_half_ratio(r3u_before, r3u_full):
    """Compiled HYB stage-2 ``r1_2 * (r3u(Kbb) + r3ua)`` association."""
    sr = nemo_source_round
    dtype = jnp.asarray(r3u_before).dtype
    return sr(jnp.asarray(0.5, dtype=dtype) * sr(r3u_before + r3u_full))


def _round115_source_depth(
    e3_face_0: np.ndarray, face_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay compiled ``hu_0`` accumulation and masked reciprocal."""
    e3_face_0 = np.asarray(e3_face_0, dtype=np.float64)
    face_mask = np.asarray(face_mask, dtype=np.float64)
    require(e3_face_0.shape == face_mask.shape,
            "round-115 reference thickness/mask shapes differ")
    depth = np.zeros(e3_face_0.shape[:2], dtype=np.float64)
    for level in range(e3_face_0.shape[-1]):
        product = np.float64(e3_face_0[..., level] * face_mask[..., level])
        depth = np.float64(depth + product)
    surface_mask = face_mask[..., 0]
    denominator = np.float64(
        np.float64(depth + np.float64(1.0)) - surface_mask)
    reciprocal = np.float64(surface_mask / denominator)
    return depth, reciprocal


def _round115_owned_geometry_bundle(
    stage_root: Path, *, plant: bool = False,
) -> dict:
    """Admit the kt2/stage-3 full/half SSH and QCO operand bundle."""
    producer = (stage_root / "producer_commit.txt").read_text().strip()
    require(producer == "715c9865008e7e419004ad1e5e643e8b71257c00",
            "round-46 stage-record producer changed")
    stage_path = stage_root / "oracle_momstage_kt00000002_s3.bin"
    require(_file_sha256(stage_path)
            == "a26121247da49d421da6ce8489627dd0b8302d10bdcd4fd1becf7752eb88524f",
            "round-46 kt2/stage-3 record digest changed")
    record = round66.round46.read_stage(
        stage_path, expected_kt=2, expected_stage=3)
    arrays = record["arrays"]

    def cell2(name):
        return np.ascontiguousarray(
            np.asarray(arrays[name], dtype=np.float64)[2:24, 2:34])

    def native_u2(name):
        return np.ascontiguousarray(
            np.asarray(arrays[name], dtype=np.float64)[2:24, 2:34])

    def native_u3(name):
        return np.ascontiguousarray(
            np.asarray(arrays[name], dtype=np.float64)[2:24, 2:34, :30])

    def native_v2(name):
        return np.ascontiguousarray(
            np.asarray(arrays[name], dtype=np.float64)[2:24, 2:34])

    def native_v3(name):
        return np.ascontiguousarray(
            np.asarray(arrays[name], dtype=np.float64)[2:24, 2:34, :30])

    def redundant_u2(name):
        return np.ascontiguousarray(
            np.asarray(arrays[name], dtype=np.float64)[2:24, 1:34])

    e3u_0 = native_u3("e3u_0")
    e3v_0 = native_v3("e3v_0")
    umask = native_u3("umask")
    vmask = native_v3("vmask")
    hu_0, r1_hu0 = _round115_source_depth(e3u_0, umask)
    hv_0, r1_hv0 = _round115_source_depth(e3v_0, vmask)
    ssh_n1 = cell2("ssh_Kaa")
    ssh_half = cell2("ssh_Kmm")
    area_t = cell2("e1e2t")
    r1_area_u = native_u2("r1_e1e2u")
    r1_area_v = native_v2("r1_e1e2v")
    area_u = native_u2("e1e2u")
    area_v = native_v2("e1e2v")
    r3u_before = native_u2("r3u_Kbb")
    r3u_full = native_u2("r3u_Kaa")
    r3u_half = native_u2("r3u_Kmm")
    wet_u_native = np.any(umask > 0.5, axis=-1)

    plant_row = None
    planted = np.array(ssh_n1, copy=True)
    for raw_index in np.argwhere(
            np.isfinite(planted) & (planted != 0.0)):
        jj, ji = (int(item) for item in raw_index)
        for face_i in (ji, (ji - 1) % planted.shape[1]):
            if not wet_u_native[jj, face_i]:
                continue
            old = planted[jj, ji]
            new = np.nextafter(old, np.float64(np.inf))
            old_trace = _round115_scalar_u_trace(
                planted, area_t, r1_hu0, r1_area_u)
            candidate = np.array(planted, copy=True)
            candidate[jj, ji] = new
            new_trace = _round115_scalar_u_trace(
                candidate, area_t, r1_hu0, r1_area_u)
            old_half = np.float64(np.float64(0.5) * np.float64(
                r3u_before[jj, face_i] + old_trace[5][jj, face_i]))
            new_half = np.float64(np.float64(0.5) * np.float64(
                r3u_before[jj, face_i] + new_trace[5][jj, face_i]))
            product_changed = bool(
                old_trace[0][jj, face_i] != new_trace[0][jj, face_i]
                or old_trace[1][jj, face_i] != new_trace[1][jj, face_i])
            if (product_changed
                    and old_trace[5][jj, face_i] != new_trace[5][jj, face_i]
                    and old_half != new_half):
                plant_row = {
                    "field": "ssh_n1", "index": [jj, ji],
                    "affected_native_u_index": [jj, face_i],
                    "before_bits": int(
                        np.asarray(old, dtype="=f8").view("=u8")),
                    "after_bits": int(
                        np.asarray(new, dtype="=f8").view("=u8")),
                }
                if plant:
                    planted[jj, ji] = new
                break
        if plant_row is not None:
            break
    require(plant_row is not None,
            "no finite nonzero SSH word propagates through full and half r3u")
    if plant:
        ssh_n1 = planted

    scalar_trace = _round115_scalar_u_trace(
        ssh_n1, area_t, r1_hu0, r1_area_u)
    source_half = np.empty_like(r3u_before)
    for jj in range(source_half.shape[0]):
        for ji in range(source_half.shape[1]):
            summed = np.float64(r3u_before[jj, ji] + scalar_trace[5][jj, ji])
            source_half[jj, ji] = np.float64(np.float64(0.5) * summed)

    expected_header = {
        "version": 1, "kt": 2, "stage": 3,
        "Kbb": 3, "Kmm": 2, "Krhs": 1, "Kaa": 1,
        "jpi": 36, "jpj": 26, "jpk": 31, "jpkm1": 30,
        "ntsi": 3, "ntei": 34, "ntsj": 3, "ntej": 24, "bits": 64,
    }
    require(record["header"] == expected_header,
            "round-46 kt2/stage-3 header changed")
    wet_t = cell2("tmask")[..., 0] > 0.5
    wet_u_redundant = np.any(
        np.asarray(arrays["umask"], dtype=np.float64)[
            2:24, 1:34, :30] > 0.5,
        axis=-1)
    expected_shapes = {
        "cell": (22, 32), "native_u": (22, 32),
        "redundant_u": (22, 33), "native_u_3d": (22, 32, 30),
    }
    require(ssh_n1.shape == ssh_half.shape == area_t.shape
            == wet_t.shape == expected_shapes["cell"],
            "round-115 admitted T-cell operand shape changed")
    require(r1_area_u.shape == r1_hu0.shape == r3u_before.shape
            == r3u_full.shape == r3u_half.shape
            == wet_u_native.shape == expected_shapes["native_u"],
            "round-115 admitted native-U operand shape changed")
    require(e3u_0.shape == umask.shape == expected_shapes["native_u_3d"],
            "round-115 admitted native-U 3-D operand shape changed")
    require(wet_u_redundant.shape == expected_shapes["redundant_u"],
            "round-115 admitted redundant-U mask shape changed")
    return {
        "producer_commit": producer,
        "stage_sha256": _file_sha256(stage_path),
        "header": record["header"],
        "ssh_n1": ssh_n1,
        "ssh_half": ssh_half,
        "ssh_before": cell2("ssh_Kbb"),
        "area_t": area_t,
        "area_u": area_u,
        "area_v": area_v,
        "r1_area_u": r1_area_u,
        "r1_area_v": r1_area_v,
        "e3u_0": e3u_0,
        "e3v_0": e3v_0,
        "umask": umask,
        "vmask": vmask,
        "hu_0": hu_0,
        "hv_0": hv_0,
        "r1_hu0": r1_hu0,
        "r1_hv0": r1_hv0,
        "r3u_before": r3u_before,
        "r3u_full": r3u_full,
        "r3u_half": r3u_half,
        "one_plus_r3u_half": redundant_u2("r3u_Kmm") + np.float64(1.0),
        "wet_t": wet_t,
        "wet_u_native": wet_u_native,
        "wet_u_redundant": wet_u_redundant,
        "scalar_trace": scalar_trace,
        "source_half": source_half,
        "plant_row": plant_row,
    }


def _round115_score_geometry(observation: tuple, bundle: dict) -> dict:
    require(len(observation) == len(ROUND115_OBSERVATION_FIELDS),
            "round-115 geometry callback schema changed")
    got = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in zip(
            ROUND115_OBSERVATION_FIELDS, observation, strict=True)
    }
    wet_t = np.asarray(bundle["wet_t"], dtype=bool)
    wet_u = np.asarray(bundle["wet_u_native"], dtype=bool)
    wet_ur = np.asarray(bundle["wet_u_redundant"], dtype=bool)

    def row(candidate, reference, mask):
        scored = round54.field_stats(candidate, reference, mask)
        scored["classification"] = (
            "BIT" if scored["cells_unequal"] == 0 else "DEBT")
        return scored

    input_rows = {
        "e1e2t": row(got["live_area_t"], bundle["area_t"], wet_t),
        "full_step_ssh": row(
            got["full_step_ssh"], bundle["ssh_n1"], wet_t),
        "r1_hu_0": row(got["live_r1_hu0"], bundle["r1_hu0"], wet_u),
        "r1_e1e2u": row(
            got["live_r1_area_u"], bundle["r1_area_u"], wet_u),
        "half_step_ssh": row(got["half_step_ssh"], bundle["ssh_half"], wet_t),
    }
    trace_rows = {
        name: row(got[name], reference, wet_u)
        for name, reference in zip(
            ROUND115_TRACE_FIELDS, bundle["scalar_trace"], strict=True)
    }
    direct_rows = {
        "live_full_r3u": row(
            got["live_full_r3u"], bundle["r3u_full"], wet_u),
        "live_half_r3u": row(
            got["live_half_r3u"], bundle["r3u_half"], wet_u),
        "oracle_full_r3u": row(
            got["oracle_full_r3u"], bundle["r3u_full"], wet_u),
        "oracle_interpolated_half_r3u": row(
            got["oracle_interpolated_half_r3u"], bundle["r3u_half"], wet_u),
        "oracle_interpolated_half_one_plus_r3u": row(
            got["oracle_interpolated_half_one_plus_r3u"],
            np.float64(1.0) + bundle["r3u_half"], wet_u),
        "ordinary_one_plus_r3u": row(
            got["ordinary_one_plus_r3u"],
            bundle["one_plus_r3u_half"], wet_ur),
    }
    first_nonbit_input = next(
        (name for name in (
            "e1e2t", "full_step_ssh", "r1_hu_0", "r1_e1e2u",
            "half_step_ssh")
         if input_rows[name]["classification"] != "BIT"),
        None,
    )
    return {
        "inputs": input_rows,
        "source_trace_given_nemo_inputs": trace_rows,
        "direct_outputs": direct_rows,
        "first_nonbit_input": first_nonbit_input,
    }


def _round115_score_isolated(
    trace: tuple, full_r3u: np.ndarray, half_r3u: np.ndarray, bundle: dict,
) -> dict:
    """Score the separately labelled isolated-JIT source replay."""
    require(len(trace) == len(ROUND115_TRACE_FIELDS),
            "round-115 isolated trace schema changed")
    wet_u = np.asarray(bundle["wet_u_native"], dtype=bool)

    def row(candidate, reference):
        scored = round54.field_stats(candidate, reference, wet_u)
        scored["classification"] = (
            "BIT" if scored["cells_unequal"] == 0 else "DEBT")
        return scored

    return {
        "source_trace_given_nemo_inputs": {
            name: row(candidate, reference)
            for name, candidate, reference in zip(
                ROUND115_TRACE_FIELDS, trace, bundle["scalar_trace"],
                strict=True)
        },
        "direct_outputs": {
            "oracle_full_r3u": row(full_r3u, bundle["r3u_full"]),
            "oracle_interpolated_half_r3u": row(
                half_r3u, bundle["r3u_half"]),
        },
    }


def _round115_collapse(
    calls: list[tuple], label: str, *, allow_prior_distinct: bool = False,
) -> tuple:
    require(calls, f"round-115 {label} callback did not execute")
    selected = calls[-1]
    for prior in calls[:-1]:
        exact = len(prior) == len(selected) and all(
            np.array_equal(left, right)
            for left, right in zip(prior, selected, strict=True))
        require(exact or allow_prior_distinct,
                f"round-115 {label} observed distinct duplicate executions")
    return selected


def _replace_kmm_resume(resume, override):
    """Replace only selected stage-2 tracers at the final WS helper boundary."""
    require(resume is not None and resume[0] == 2,
            "final WS helper did not receive the stage-2 Kmm tracer")
    if override is None:
        return resume
    target_t, target_s = override
    return (
        2,
        resume[1] if target_t is None else target_t,
        resume[2] if target_s is None else target_s,
    )


def measure(args) -> dict:
    require(sum((args.round69_native, args.round70_pair, args.round71_kmm,
                 args.round111_fct_split, args.round112_fct_walk,
                 args.round113_live_inputs, args.round114_u_operands,
                 args.round115_u_geometry)) <= 1,
            "round-69/70/71/111/112/113/114/115 modes are exclusive")
    require(not (args.plant_pair_null_fct and args.plant_pair_content_ulp),
            "round-70 plants are mutually exclusive")
    require(args.round70_pair or not (
        args.plant_pair_null_fct or args.plant_pair_content_ulp),
        "round-70 plants require --round70-pair")
    require(not (args.plant_kmm_null and args.plant_kmm_content_ulp),
            "round-71 plants are mutually exclusive")
    require(args.round71_kmm or not (
        args.plant_kmm_null or args.plant_kmm_content_ulp),
        "round-71 plants require --round71-kmm")
    require(args.round111_fct_split or not args.plant_fct_split_ulp,
            "the FCT split plant requires --round111-fct-split")
    require(args.round112_fct_walk or not args.plant_fct_walk_ulp,
            "the round-112 plant requires --round112-fct-walk")
    require(args.round113_live_inputs or not args.plant_live_input_ulp,
            "the round-113 plant requires --round113-live-inputs")
    require(args.round114_u_operands or not args.plant_u_factor_ulp,
            "the round-114 plant requires --round114-u-operands")
    require(args.round115_u_geometry or not args.plant_u_geometry_ulp,
            "the round-115 plant requires --round115-u-geometry")
    require(not args.round114_u_operands or args.expect_u_record_commit,
            "round 114 requires --expect-u-record-commit")
    round114_config = {}
    if args.round114_u_operands or args.round115_u_geometry:
        config = round66.build_nemo_testcase_card(
            "GYRE-zco").recipe.model_config
        round114_config = {
            "card": "GYRE-zco",
            "tracer_time_integrator": config.tracer_time_integrator,
            "tracer_advection": config.tracer_advection,
            "outer_integrator": config.outer_integrator,
            "barotropic_solver": config.barotropic.barotropic_solver,
            "nemo_stage_mean_imposition": bool(
                config.barotropic.nemo_stage_mean_imposition),
            "adaptive_implicit_vertadv": bool(
                config.adaptive_implicit_vertadv),
            "gm_redi_enabled": config.gm_redi is not None,
        }
        require(round114_config == {
            "card": "GYRE-zco",
            "tracer_time_integrator": "rk3_ws",
            "tracer_advection": "fct2",
            "outer_integrator": "forward_euler",
            "barotropic_solver": "explicit_substep",
            "nemo_stage_mean_imposition": True,
            "adaptive_implicit_vertadv": False,
            "gm_redi_enabled": True,
        }, f"round-114/115 resolved configuration changed: {round114_config}")
        print(("ROUND115" if args.round115_u_geometry else "ROUND114")
              + " RESOLVED CONFIG "
              + json.dumps(round114_config, sort_keys=True), flush=True)
    reciprocal_calls: list[dict] = []
    pair_sources: list[tuple[np.ndarray, ...]] = []
    pair_kmm: list[tuple[np.ndarray, np.ndarray]] = []
    qsr_calls: list[np.ndarray] = []
    ldf_calls: list[tuple[np.ndarray, np.ndarray]] = []
    step_calls: list[tuple] = []
    step_results: list[object] = []
    fct_split_calls: list[tuple[np.ndarray, ...]] = []
    fct_walk_calls: list[tuple[np.ndarray, ...]] = []
    live_input_calls: list[tuple[np.ndarray, ...]] = []
    u_operand_calls: list[tuple[np.ndarray, ...]] = []
    u_geometry_calls: list[tuple[np.ndarray, ...]] = []

    round112_bundle = None
    round112_values = None
    if (args.round112_fct_walk or args.round113_live_inputs
            or args.round114_u_operands):
        round112_bundle = _round112_owned_record(
            args.fct_walk_record,
            plant=(args.plant_fct_walk_ulp or args.plant_live_input_ulp))
        round112_values = {
            name: jnp.asarray(value)
            for name, value in round112_bundle.items()
            if name not in ("sha256", "expected", "plant_row")
        }

    round114_bundle = None
    round114_values = None
    if args.round114_u_operands:
        require(round112_bundle is not None,
                "round-114 FCT transport record was not loaded")
        round114_bundle = _round114_owned_u_bundle(
            record_root=args.u_record_root,
            stage_baseline_root=args.stage_baseline_root,
            admission_path=args.u_admission,
            expected_commit=args.expect_u_record_commit,
            p_u=round112_bundle["p_u"],
            plant=args.plant_u_factor_ulp)
        round114_values = {
            name: jnp.asarray(value)
            for name, value in round114_bundle.items()
            if isinstance(value, np.ndarray)
        }

    round115_bundle = None
    round115_values = None
    if args.round115_u_geometry:
        round115_bundle = _round115_owned_geometry_bundle(
            args.stage_baseline_root, plant=args.plant_u_geometry_ulp)
        round115_values = {
            name: jnp.asarray(value)
            for name, value in round115_bundle.items()
            if isinstance(value, np.ndarray)
        }

    real_reciprocal = round66.reciprocal_substitutions
    real_pair = model_module._nemo_ws_rk3_tracer_pair_step
    real_qsr = model_module._nemo_qsr_stage3_rate
    real_ldf = model_module.gm_redi_tracer_tendency_latlon
    real_step = model_module.LatLonCGridOceanModel.step
    real_fct = advection_module.fct_tracer_advection
    real_stage_transport = model_module._nemo_ws_stage_transport
    real_stage_faces = model_module._nemo_ws_qco_stage_faces
    real_face_geometry = model_module.nemo_qco_live_face_geometry_cgrid
    real_native_geometry = (
        model_module.nemo_qco_live_face_geometry_from_operands)
    active_kmm_override: tuple[object | None, object | None] | None = None
    capture_fct_split = False
    capture_fct_walk = False
    capture_live_inputs = False
    fct_walk_trace_index = 0
    live_input_trace_index = 0
    active_live_input_family: str | None = None
    active_u_factor: str | None = None
    round113_eager_direct = False
    round114_stage3_active = False
    round114_stage3_faces = None
    round115_stage3_active = False
    round115_full_step_ssh = None

    def reciprocal_capture(live, oracle, oracle_content, wet):
        reciprocal_calls.append({
            "live": _copy_operands(live),
            "oracle": _copy_operands(oracle),
            "oracle_content": np.array(oracle_content, dtype=np.float64, copy=True),
            "wet": np.array(wet, dtype=bool, copy=True),
        })
        return real_reciprocal(live, oracle, oracle_content, wet)

    def pair_sink(
        source_t, source_s, tracer_t, tracer_s, content_t, content_s,
        advection_content_t, advection_content_s,
    ):
        pair_sources.append(tuple(
            np.asarray(value, dtype=np.float64)
            for value in (
                source_t, source_s, tracer_t, tracer_s, content_t, content_s,
                advection_content_t, advection_content_s)))

    def pair_capture(*values, **kwargs):
        nonlocal capture_fct_split, capture_fct_walk, capture_live_inputs
        nonlocal fct_walk_trace_index, live_input_trace_index
        call_kwargs = kwargs
        if kwargs.get("return_final_content", False):
            resume = _replace_kmm_resume(
                kwargs.get("resume"), active_kmm_override)
            resume_t, resume_s = resume[1], resume[2]
            if resume is not kwargs.get("resume"):
                call_kwargs = dict(kwargs)
                call_kwargs["resume"] = resume
            jax.debug.callback(
                lambda t, s: pair_kmm.append((
                    np.asarray(t, dtype=np.float64),
                    np.asarray(s, dtype=np.float64))),
                resume_t, resume_s, ordered=True)
        prior_capture = capture_fct_split
        capture_fct_split = bool(
            args.round111_fct_split
            and kwargs.get("return_final_content", False))
        capture_fct_walk = bool(
            args.round112_fct_walk
            and kwargs.get("return_final_content", False))
        capture_live_inputs = bool(
            (args.round113_live_inputs or args.round114_u_operands)
            and kwargs.get("return_final_content", False))
        fct_walk_trace_index = 0
        live_input_trace_index = 0
        try:
            result = real_pair(*values, **call_kwargs)
        finally:
            capture_fct_split = prior_capture
            capture_fct_walk = False
            capture_live_inputs = False
        if kwargs.get("return_final_content", False):
            source_t, source_s = kwargs["stage_source_rates"][2]
            jax.debug.callback(
                pair_sink, source_t, source_s, values[0], values[1],
                result[2], result[3], result[4], result[5],
                ordered=True)
        return result

    def fct_split_sink(*values):
        fct_split_calls.append(tuple(
            np.asarray(value, dtype=np.float64) for value in values))

    def fct_walk_sink(*values):
        fct_walk_calls.append(tuple(
            np.asarray(value, dtype=np.float64) for value in values))

    def live_input_sink(*values):
        live_input_calls.append(tuple(
            np.asarray(value, dtype=np.float64) for value in values))

    def u_operand_sink(*values):
        u_operand_calls.append(tuple(
            np.asarray(value, dtype=np.float64) for value in values))

    def u_geometry_sink(*values):
        u_geometry_calls.append(tuple(
            np.asarray(value, dtype=np.float64) for value in values))

    def face_geometry_capture(*values, **kwargs):
        result = real_face_geometry(*values, **kwargs)
        if not round115_stage3_active:
            return result
        require(round115_values is not None,
                "round-115 QCO oracle values were not loaded")
        require(round115_full_step_ssh is not None,
                "round-115 stage-3 full-step SSH was not captured")
        require(len(values) == 10,
                "round-115 live QCO operand schema changed")
        (half_step_ssh, e3u_0, e3v_0, umask, vmask, hu_0, hv_0,
         area_t, area_u, area_v) = values
        dtype = jnp.asarray(half_step_ssh).dtype
        one = jnp.asarray(1.0, dtype=dtype)
        sr = nemo_source_round
        live_wet_u = (jnp.asarray(hu_0, dtype=dtype) > 0.0).astype(dtype)
        live_r1_hu0 = sr(
            live_wet_u / (jnp.asarray(hu_0, dtype=dtype) + one - live_wet_u))
        live_r1_area_u = sr(one / jnp.asarray(area_u, dtype=dtype))
        live_half = real_native_geometry(*values)
        live_full = real_native_geometry(
            round115_full_step_ssh, e3u_0, e3v_0, umask, vmask,
            hu_0, hv_0, area_t, area_u, area_v)
        oracle_full = real_native_geometry(
            round115_values["ssh_n1"], round115_values["e3u_0"],
            round115_values["e3v_0"], round115_values["umask"],
            round115_values["vmask"], round115_values["hu_0"],
            round115_values["hv_0"], round115_values["area_t"],
            round115_values["area_u"], round115_values["area_v"])
        oracle_trace = _round115_u_trace(
            round115_values["ssh_n1"], round115_values["area_t"],
            round115_values["r1_hu0"], round115_values["r1_area_u"])
        interpolated_half = _round115_half_ratio(
            round115_values["r3u_before"], oracle_full.r3u)
        observation = (
            round115_full_step_ssh, half_step_ssh, area_t, live_r1_hu0,
            live_r1_area_u, result[2], live_full.r3u, live_half.r3u,
            *oracle_trace, oracle_full.r3u, interpolated_half,
            sr(one + interpolated_half),
        )
        jax.debug.callback(
            u_geometry_sink, *observation, ordered=True)
        return result

    def stage_faces_capture(*values, **kwargs):
        nonlocal round114_stage3_faces
        result = real_stage_faces(*values, **kwargs)
        if round114_stage3_active:
            require(round114_values is not None,
                    "round-114 U oracle values were not loaded")
            result = list(result)
            if active_u_factor == "e3u_kmm":
                result[0] = round114_values["e3u_kmm"]
            round114_stage3_faces = tuple(result)
            return tuple(result)
        return result

    def stage_transport_capture(*values, **kwargs):
        nonlocal round114_stage3_active, round114_stage3_faces
        nonlocal round115_stage3_active, round115_full_step_ssh
        if args.round115_u_geometry and values[2] == 2:
            require(kwargs.get("eta_after") is not None,
                    "round-115 stage-3 call lacks full-step SSH")
            round115_full_step_ssh = kwargs["eta_after"]
            round115_stage3_active = True
            try:
                return real_stage_transport(*values, **kwargs)
            finally:
                round115_stage3_active = False
                round115_full_step_ssh = None
        if not args.round114_u_operands or values[2] != 2:
            return real_stage_transport(*values, **kwargs)
        require(round114_values is not None,
                "round-114 U oracle values were not loaded")
        call_values = list(values)
        call_kwargs = dict(kwargs)
        stage_velocity = call_values[0]
        if active_u_factor == "u_kmm":
            call_values[0] = (round114_values["u_kmm"], stage_velocity[1])
        if active_u_factor == "un_adv":
            call_kwargs["Hu_avg"] = round114_values["un_adv"]
        if active_u_factor == "uu_b_kmm":
            barotropic = call_kwargs.get("barotropic_velocity")
            require(barotropic is not None,
                    "round-114 stage-3 call lacks carried barotropic U")
            call_kwargs["barotropic_velocity"] = (
                round114_values["uu_b_kmm"], barotropic[1])
        round114_stage3_faces = None
        round114_stage3_active = True
        try:
            result = real_stage_transport(*call_values, **call_kwargs)
        finally:
            round114_stage3_active = False
        require(round114_stage3_faces is not None,
                "round-114 stage-3 face helper did not execute")
        faces = round114_stage3_faces
        u_kmm = call_values[0][0]
        un_adv = call_kwargs["Hu_avg"]
        r1_hu = faces[4]
        barotropic = call_kwargs.get("barotropic_velocity")
        require(barotropic is not None,
                "round-114 stage-3 call used the unrecorded reduced mean")
        uu_b = barotropic[0]
        umask = call_kwargs["u_mask_3d"]
        zub, corrected, replay_zfu = _round114_u_statement(
            jnp.asarray(call_kwargs["grid"].dy_u), faces[0], u_kmm,
            un_adv, r1_hu, uu_b, umask)
        # The replay is inside the complete production trace and must be the
        # exact helper output before it can label the intermediate rows.
        replay_equal = jnp.all(replay_zfu == result[7])
        callback_values = (
            jnp.asarray(call_kwargs["grid"].dy_u), faces[0], faces[2],
            u_kmm, un_adv, r1_hu, uu_b, umask, zub, corrected, result[7],
        )
        if round113_eager_direct:
            require(bool(np.asarray(jax.device_get(replay_equal))),
                    "round-114 eager U replay differs from production")
            u_operand_sink(*jax.device_get(callback_values))
        else:
            def checked_sink(equal, *operands):
                require(bool(np.asarray(equal)),
                        "round-114 production-JIT U replay differs from helper")
                u_operand_sink(*operands)
            jax.debug.callback(
                checked_sink, replay_equal, *callback_values, ordered=True)
        return result

    def fct_capture(*values, **kwargs):
        nonlocal fct_walk_trace_index, live_input_trace_index
        if capture_fct_walk:
            require(round112_values is not None,
                    "round-112 FCT values were not loaded")
            if fct_walk_trace_index == 0:
                walk_rows = _round112_all_rows(
                    round112_values, values[5])
                jax.debug.callback(
                    fct_walk_sink, *walk_rows, ordered=True)
            fct_walk_trace_index += 1
        if capture_live_inputs:
            require(round112_bundle is not None,
                    "round-113/114 FCT values were not loaded")
            require(live_input_trace_index < 2,
                    "round-113/114 observed more than the T/S FCT calls")
            tracer = TRACERS[live_input_trace_index]
            call_values, call_kwargs = _round113_patch_inputs(
                values, kwargs, round112_bundle, tracer,
                active_live_input_family)
            result = real_fct(
                *call_values, return_nemo_split=True, **call_kwargs)
            div_h, div_w, split = result
            low_h, low_w, _, _ = split
            h_kmm = call_values[4]
            upstream_rhs = nemo_source_round(
                -nemo_source_round(low_h + low_w)
                / jnp.maximum(h_kmm, jnp.asarray(1.0e-10, h_kmm.dtype)))
            input_values = _round113_input_tuple(
                call_values, call_kwargs, tracer=tracer)
            if round113_eager_direct:
                live_input_sink(*jax.device_get((*input_values, upstream_rhs)))
            else:
                jax.debug.callback(
                    live_input_sink, *input_values, upstream_rhs, ordered=True)
            live_input_trace_index += 1
            if round113_eager_direct and live_input_trace_index == 2:
                raise _Round113EagerBoundaryComplete
            return div_h, div_w
        if not capture_fct_split:
            return real_fct(*values, **kwargs)
        result = real_fct(*values, return_nemo_split=True, **kwargs)
        div_h, div_w, split = result
        low_h, low_w, anti_h, anti_w = split
        h_kmm = values[4]
        h_safe = jnp.maximum(h_kmm, jnp.asarray(1.0e-10, h_kmm.dtype))
        low_div = nemo_source_round(low_h + low_w)
        anti_div = nemo_source_round(anti_h + anti_w)
        upstream_rhs = nemo_source_round(-low_div / h_safe)
        anti_rhs = nemo_source_round(-anti_div / h_safe)
        if args.plant_fct_split_ulp:
            active = kwargs.get("active_mask")
            require(active is not None,
                    "the production FCT split plant requires an active mask")
            at = jnp.argmax(jnp.ravel(active > 0.5))
            flat = jnp.ravel(upstream_rhs)
            upstream_rhs = flat.at[at].set(jnp.nextafter(
                flat[at], jnp.asarray(jnp.inf, flat.dtype))).reshape(
                    upstream_rhs.shape)
        split_rhs = nemo_source_round(upstream_rhs + anti_rhs)
        combined_rhs = nemo_source_round(
            -nemo_source_round(div_h + div_w) / h_safe)
        base = kwargs.get("tracer_before")
        if base is None:
            base = values[0]
        h_kbb = kwargs.get("base_thickness")
        if h_kbb is None:
            h_kbb = h_kmm
        dt = jnp.asarray(values[6], dtype=h_kmm.dtype)
        split_content = nemo_source_round(
            nemo_source_round(h_kbb * base)
            + nemo_source_round(nemo_source_round(dt * h_kmm) * split_rhs))
        jax.debug.callback(
            fct_split_sink, upstream_rhs, anti_rhs, split_rhs, combined_rhs,
            split_content, low_div, anti_div, ordered=True)
        return div_h, div_w

    def qsr_sink(value):
        qsr_calls.append(np.asarray(value, dtype=np.float64))

    def qsr_capture(*values, **kwargs):
        result = real_qsr(*values, **kwargs)
        jax.debug.callback(qsr_sink, values[2], ordered=True)
        return result

    def ldf_sink(value_t, value_s):
        ldf_calls.append((np.asarray(value_t, dtype=np.float64),
                          np.asarray(value_s, dtype=np.float64)))

    def ldf_capture(*values, **kwargs):
        result = real_ldf(*values, **kwargs)
        require(len(result) == 2,
                "GYRE GM/Redi unexpectedly returned bolus transports")
        jax.debug.callback(ldf_sink, result[0], result[1], ordered=True)
        return result

    def step_capture(self, state, dt, freshwater=None, surface_forcing=None,
                     *values, **kwargs):
        step_calls.append((self, state, dt, freshwater, surface_forcing))
        result = real_step(self, state, dt, freshwater, surface_forcing,
                           *values, **kwargs)
        step_results.append(result)
        return result

    round66.reciprocal_substitutions = reciprocal_capture
    model_module._nemo_ws_rk3_tracer_pair_step = pair_capture
    advection_module.fct_tracer_advection = fct_capture
    model_module._nemo_qsr_stage3_rate = qsr_capture
    model_module.gm_redi_tracer_tendency_latlon = ldf_capture
    model_module.LatLonCGridOceanModel.step = step_capture
    model_module._nemo_ws_qco_stage_faces = stage_faces_capture
    model_module._nemo_ws_stage_transport = stage_transport_capture
    model_module.nemo_qco_live_face_geometry_cgrid = face_geometry_capture
    try:
        base = round66.measure(args)
    finally:
        round66.reciprocal_substitutions = real_reciprocal
        model_module._nemo_ws_rk3_tracer_pair_step = real_pair
        advection_module.fct_tracer_advection = real_fct
        model_module._nemo_qsr_stage3_rate = real_qsr
        model_module.gm_redi_tracer_tendency_latlon = real_ldf
        model_module.LatLonCGridOceanModel.step = real_step
        model_module._nemo_ws_qco_stage_faces = real_stage_faces
        model_module._nemo_ws_stage_transport = real_stage_transport
        model_module.nemo_qco_live_face_geometry_cgrid = real_face_geometry

    require(len(reciprocal_calls) == 2,
            f"expected T/S reciprocal calls, got {len(reciprocal_calls)}")
    require(pair_sources and qsr_calls and ldf_calls and len(step_calls) >= 2,
            "one or more production captures did not fire")
    # round66's final production call is the kt=2 oracle-seeded call. Multiple
    # callback observations can occur under JAX, but the last observation must
    # carry that call's exact Kbb tracer.
    (source_t, source_s, pair_t, pair_s, content_t, content_s,
     advection_content_t, advection_content_s) = pair_sources[-1]
    _, seeded, dt, freshwater, surface = step_calls[-1]
    require(np.array_equal(pair_t, np.asarray(seeded.T.data, dtype=np.float64))
            and np.array_equal(pair_s, np.asarray(seeded.S.data, dtype=np.float64)),
            "last source capture is not the oracle-seeded kt=2 call")

    record = round54._read_r63_stream(args.krhs_record, kind="krhs")
    arrays = record["arrays"]
    qsr_t = qsr_calls[-1]
    qsr = {"T": qsr_t, "S": np.zeros_like(qsr_t)}
    ldf_t, ldf_s = ldf_calls[-1]
    ldf = {"T": ldf_t, "S": ldf_s}
    sources = {"T": source_t, "S": source_s}
    captured_content = {"T": content_t, "S": content_s}
    captured_advection_content = {
        "T": advection_content_t,
        "S": advection_content_s,
    }

    results = {}
    routed_content = {}
    implementation_content = {}
    implementation_oracle_content = {}
    production_prediction_content = {}
    production_prediction_rows = {}
    production_baseline_rebuild = {}
    production_vs_round67_routed = {}
    oracle_advection_content = {}
    for name, capture in zip(TRACERS, reciprocal_calls, strict=True):
        boundaries = {
            boundary: round66.transposed(arrays[f"{boundary}_{name}"])
            for boundary in ("after_adv", "after_sbc", "after_qsr", "after_ldf")
        }
        oracle_adv_operands = dict(capture["oracle"])
        oracle_adv_operands["Krhs"] = boundaries["after_adv"]
        oracle_advection_content[name] = round66.content_statement(
            oracle_adv_operands)
        live = capture["live"]
        source = sources[name]
        routed_krhs = None
        if args.expect_model_order == "after":
            # The production content already contains LDF.  Remove it only
            # from the diagnostic decomposition; score the captured Krhs
            # directly as the routed arm so subtraction/addition rounding
            # cannot masquerade as a production discrepancy.
            routed_krhs = live["Krhs"]
            live = dict(live)
            live["Krhs"] = live["Krhs"] - ldf[name]
            source = source - ldf[name]
        result = cumulative_substitutions(
            live=live, oracle=capture["oracle"],
            oracle_content=capture["oracle_content"], boundaries=boundaries,
            source=source, qsr=qsr[name], ldf=ldf[name],
            wet=capture["wet"], routed_krhs=routed_krhs)
        routed_content[name] = result.pop("arm_content")["routed_live"]
        implementation_content[name] = round54.field_stats(
            captured_content[name], routed_content[name], capture["wet"])
        implementation_oracle_content[name] = round54.field_stats(
            captured_content[name], capture["oracle_content"], capture["wet"])
        current_ldf = (
            np.zeros_like(ldf[name])
            if args.expect_production_route == "after" else ldf[name]
        )
        production_baseline = production_fct_content(
            captured_advection_content[name], sources[name],
            np.zeros_like(ldf[name]), capture["live"])
        production_prediction = production_fct_content(
            captured_advection_content[name], sources[name],
            current_ldf, capture["live"])
        production_prediction_content[name] = production_prediction
        production_baseline_rebuild[name] = round54.field_stats(
            production_baseline, captured_content[name], capture["wet"])
        production_prediction_rows[name] = {
            "content_vs_oracle": round54.field_stats(
                production_prediction, capture["oracle_content"], capture["wet"]),
            "implementation_vs_prediction": round54.field_stats(
                captured_content[name], production_prediction, capture["wet"]),
        }
        production_vs_round67_routed[name] = round54.field_stats(
            production_prediction, routed_content[name], capture["wet"])
        results[name] = result

    round111_matrix = {}
    if args.round111_fct_split:
        require(len(fct_split_calls) >= 2,
                "production FCT split callbacks did not execute for T/S")
        selected_splits = fct_split_calls[-2:]
        split_rows = {}
        split_observations = {}
        for name, capture, values in zip(
            TRACERS, reciprocal_calls, selected_splits, strict=True
        ):
            (upstream_rhs, anti_rhs, split_rhs, combined_rhs,
             split_content, low_div, anti_div) = values
            oracle_upstream = round66.transposed(arrays[f"adv_up1_{name}"])
            oracle_final = round66.transposed(arrays[f"after_adv_{name}"])
            wet = capture["wet"]
            first_wet = tuple(np.argwhere(wet)[0])
            split_rows[name] = {
                "upstream_rhs_vs_adv_up1": round54.field_stats(
                    upstream_rhs, oracle_upstream, wet),
                "split_rhs_vs_after_adv": round54.field_stats(
                    split_rhs, oracle_final, wet),
                "combined_rhs_vs_after_adv": round54.field_stats(
                    combined_rhs, oracle_final, wet),
                "split_vs_combined": round54.field_stats(
                    split_rhs, combined_rhs, wet),
                "split_advection_content_vs_oracle": round54.field_stats(
                    split_content, oracle_advection_content[name], wet),
                "anti_rhs_vs_posthoc_boundary_difference": round54.field_stats(
                    anti_rhs, oracle_final - oracle_upstream, wet),
                "low_content_divergence": round54.field_stats(
                    low_div, np.zeros_like(low_div), wet),
                "anti_content_divergence": round54.field_stats(
                    anti_div, np.zeros_like(anti_div), wet),
            }
            split_observations[name] = {
                "first_wet_index": [int(index) for index in first_wet],
                "upstream_rhs_bits": int(np.asarray(
                    upstream_rhs[first_wet], dtype="=f8").view("=u8")),
                "split_rhs_bits": int(np.asarray(
                    split_rhs[first_wet], dtype="=f8").view("=u8")),
            }
        local_frozen = {
            "T": {"content": np.float64(5.743498263655056e-5),
                  "kt3": np.float64(8.600420500215478e-7)},
            "S": {"content": np.float64(7.387909136014059e-6),
                  "kt3": np.float64(6.979443156751586e-8)},
        }
        split_criteria = {
            "callbacks_observed": len(fct_split_calls) >= 2,
            "upstream_predicted_nonbit": {
                name: split_rows[name]["upstream_rhs_vs_adv_up1"][
                    "cells_unequal"] > 0
                for name in TRACERS
            },
            "final_predicted_nonbit": {
                name: split_rows[name]["combined_rhs_vs_after_adv"][
                    "cells_unequal"] > 0
                for name in TRACERS
            },
            "frozen_content_max": {
                name: bool(
                    implementation_oracle_content[name]["max_abs"]
                    == local_frozen[name]["content"])
                for name in TRACERS
            },
        }
        candidate_eligible = bool(all(
            split_rows[name][boundary]["cells_unequal"] == 0
            for name in TRACERS
            for boundary in (
                "upstream_rhs_vs_adv_up1", "split_rhs_vs_after_adv")
        ))
        plant_fired = False
        if args.plant_fct_split_ulp:
            frozen = json.loads(args.fct_split_report.read_text())[
                "round111_fct_split"]
            plant_fired = bool(any(
                split_observations[name]["upstream_rhs_bits"]
                != frozen["observations"][name]["upstream_rhs_bits"]
                for name in TRACERS
            ))
            require(plant_fired,
                    "one-ULP production FCT split plant was not observed")
        round111_matrix = {
            "status": (
                "PLANT-FIRED" if plant_fired else
                "CONFIRMED" if all(
                    all(value.values()) if isinstance(value, dict) else value
                    for value in split_criteria.values())
                else "REFUTED"),
            "plant_ulp": bool(args.plant_fct_split_ulp),
            "candidate_eligible": candidate_eligible,
            "rows": split_rows,
            "observations": split_observations,
            "criteria": split_criteria,
            "caveat": (
                "The anti-boundary subtraction is post-hoc and is not an "
                "input proof; eligibility uses only directly recorded rows."),
        }

    content_criterion = {name: True for name in TRACERS}
    if args.expect_model_order == "after":
        prediction = json.loads(args.prediction_report.read_text())
        for name in TRACERS:
            predicted = prediction["cumulative_substitutions"][name][
                "arm_rows"]["routed_live"]["content"]
            actual = implementation_oracle_content[name]
            association = implementation_content[name]
            content_criterion[name] = bool(
                actual["max_abs"]
                <= predicted["max_abs"] + association["max_abs"])

    post_ldf = results["T"]["arm_rows"]["nemo_post_ldf"]["content"]
    oracle_exact = base["substitution_rows"]["T"]["oracle_baseline"]
    require(oracle_exact["cells_unequal"] == 0,
            "all-oracle five-operand statement did not rebuild exact content")
    expected_post_ldf = (
        base["substitution_rows"]["T"]["oracle_into_live"]["Krhs"])
    require(post_ldf == expected_post_ldf,
            "post-LDF single substitution did not reproduce round-66 Krhs row")
    plant = _one_ulp_plant(
        round66.transposed(arrays["content_T"]), reciprocal_calls[0]["wet"])

    model, seeded, dt, freshwater, surface = step_calls[-1]
    if (args.round113_live_inputs or args.round114_u_operands
            or args.round115_u_geometry):
        require(step_results,
                "round-113/114/115 seeded production result is absent")
        trace = jax.device_get(step_results[-1])
        require(hasattr(trace, "state_after"),
                "round-113/114/115 seeded result is not the live operand trace")
        baseline_state = trace.state_after
    else:
        baseline_state = jax.device_get(type(model)(
            model.grid, model.z_coord, model.config).step(
                seeded, dt, freshwater=freshwater, surface_forcing=surface))

    round112_matrix = {}
    if args.round112_fct_walk:
        require(round112_bundle is not None and round112_values is not None,
                "round-112 record was not prepared")
        jax.effects_barrier()
        production_jit = _round112_collapse(
            fct_walk_calls, "production-step JIT")

        mode_observations = {
            "production_step_jit": production_jit,
        }
        if not args.plant_fct_walk_ulp:
            # The production step() shim deliberately re-enables JIT. Drive
            # the identical full step body directly under disable_jit for the
            # separate eager label required by the campaign; this is not an
            # isolated FCT closure and still executes every preceding
            # production operator. The plant qualifies only in production
            # JIT, so it deliberately skips these two non-qualifying replays.
            fct_walk_calls.clear()
            model_module._nemo_ws_rk3_tracer_pair_step = pair_capture
            advection_module.fct_tracer_advection = fct_capture
            try:
                with jax.disable_jit():
                    eager_state = model._step_impl(
                        seeded, dt, freshwater=freshwater,
                        surface_forcing=surface)
                    jax.device_get(eager_state)
                jax.effects_barrier()
            finally:
                model_module._nemo_ws_rk3_tracer_pair_step = real_pair
                advection_module.fct_tracer_advection = real_fct
            production_eager = _round112_collapse(
                fct_walk_calls, "production eager")
            isolated = jax.device_get(jax.jit(
                lambda: _round112_all_rows(round112_values, model.grid))())
            mode_observations.update({
                "production_eager": production_eager,
                "isolated_closure_jit": tuple(isolated),
            })
        mode_rows = {
            name: _round112_observation_rows(observation, round112_bundle)
            for name, observation in mode_observations.items()
        }

        source_boundaries = (
            ("horizontal_first_faces", ("first_u", "first_v")),
            ("vertical_first_face", ("first_w",)),
            ("first_divergence", ("first_div",)),
            ("midpoint", ("midpoint",)),
            ("averaged_faces", ("average_u", "average_v", "average_w")),
            ("final_divergence", ("final_div",)),
            ("divided_rhs_write", ("rhs_after",)),
        )
        first_nonbit = None
        jit_current = mode_rows["production_step_jit"]["current"]
        for boundary, members in source_boundaries:
            if any(jit_current[tracer][member]["cells_unequal"] > 0
                   for tracer in TRACERS for member in members):
                first_nonbit = boundary
                break

        literal_bit = {
            mode: {
                tracer: all(rows["source_literal"][tracer][name][
                    "cells_unequal"] == 0 for name in ROUND112_ROWS)
                for tracer in TRACERS
            }
            for mode, rows in mode_rows.items()
        }
        criteria = {
            "first_nonbit_is_horizontal_faces": (
                first_nonbit == "horizontal_first_faces"),
            "both_tracers_horizontal_faces_nonbit": {
                tracer: bool(
                    jit_current[tracer]["first_u"]["cells_unequal"] > 0
                    or jit_current[tracer]["first_v"]["cells_unequal"] > 0)
                for tracer in TRACERS
            },
            "later_boundaries_nonbit": {
                tracer: bool(
                    jit_current[tracer]["midpoint"]["cells_unequal"] > 0
                    and jit_current[tracer]["rhs_after"]["cells_unequal"] > 0)
                for tracer in TRACERS
            },
            "source_literal_all_bit": literal_bit,
        }

        plant_at = tuple(round112_bundle["plant_row"]["index"])

        def observation_bits(observation, source_offset):
            value = np.asarray(observation[source_offset])[plant_at]
            return int(np.asarray(value, dtype="=f8").view("=u8"))

        observations = {
            mode: {
                "current_first_u_T_bits": observation_bits(value, 0),
                "source_literal_first_u_T_bits": observation_bits(
                    value, 2 * len(ROUND112_ROWS)),
            }
            for mode, value in mode_observations.items()
        }

        def all_true(value) -> bool:
            if isinstance(value, dict):
                return all(all_true(item) for item in value.values())
            return bool(value)

        plant_fired = False
        plant_propagation = {}
        if args.plant_fct_walk_ulp:
            frozen = json.loads(args.fct_walk_report.read_text())[
                "round112_fct_walk"]
            plant_propagation = {
                key: bool(
                    observations["production_step_jit"][key]
                    != frozen["observations"]["production_step_jit"][key])
                for key in (
                    "current_first_u_T_bits",
                    "source_literal_first_u_T_bits")
            }
            # The exact-input/source-literal row is the calibrated boundary.
            # The current divide/re-multiply representation may erase a
            # one-ULP transport perturbation; requiring it to propagate would
            # make the control conditional on the defect being measured.
            plant_fired = plant_propagation[
                "source_literal_first_u_T_bits"]
            require(plant_fired,
                    "round-112 production-JIT one-ULP input plant was inert")

        round112_matrix = {
            "status": (
                "PLANT-FIRED" if plant_fired else
                "CONFIRMED" if all_true(criteria) else "REFUTED"),
            "plant_ulp": bool(args.plant_fct_walk_ulp),
            "record_sha256": round112_bundle["sha256"],
            "plant_target": round112_bundle["plant_row"],
            "first_nonbit_boundary": first_nonbit,
            "first_nonbit_compiled_statement": (
                "traadv_fct.f90:508-510 horizontal first-upwind faces"
                if first_nonbit == "horizontal_first_faces" else first_nonbit),
            "modes": mode_rows,
            "observations": observations,
            "plant_propagation": plant_propagation,
            "criteria": criteria,
        }

    round113_matrix = {}
    if args.round113_live_inputs:
        require(round112_bundle is not None,
                "round-113 record was not prepared")
        jax.effects_barrier()
        jit_observation = _round113_collapse_calls(
            live_input_calls, "production-step JIT", allow_prior_distinct=True)
        jit_rows = _round113_score_inputs(jit_observation, round112_bundle)

        families = (
            ("base", ("base",)),
            ("transport", ("transport_u", "transport_v", "transport_w")),
            ("thickness", ("h_kbb", "h_kmm")),
            ("mask", ("tmask", "wmask")),
            ("metric", ("r1_area",)),
            ("p2dt", ("p2dt",)),
        )
        first_nonbit_family = None
        for family, names in families:
            if any(jit_rows[tracer][name]["cells_unequal"] > 0
                   for tracer in TRACERS for name in names):
                first_nonbit_family = family
                break

        eager_rows = {}
        if not args.plant_live_input_ulp:
            live_input_calls.clear()
            active_live_input_family = None
            round113_eager_direct = True
            model_module._nemo_ws_rk3_tracer_pair_step = pair_capture
            advection_module.fct_tracer_advection = fct_capture
            try:
                try:
                    with jax.disable_jit():
                        model._step_impl(
                            seeded, dt, freshwater=freshwater,
                            surface_forcing=surface)
                except _Round113EagerBoundaryComplete:
                    pass
            finally:
                round113_eager_direct = False
                model_module._nemo_ws_rk3_tracer_pair_step = real_pair
                advection_module.fct_tracer_advection = real_fct
            eager_observation = _round113_collapse_calls(
                live_input_calls, "production eager")
            eager_rows = _round113_score_inputs(
                eager_observation, round112_bundle)

        substitutable = {"base", "transport", "thickness", "mask", "p2dt"}
        require(first_nonbit_family in substitutable,
                "first non-bit live family is absent or not substitutable: "
                f"{first_nonbit_family!r}")
        live_input_calls.clear()
        active_live_input_family = first_nonbit_family
        model_module._nemo_ws_rk3_tracer_pair_step = pair_capture
        advection_module.fct_tracer_advection = fct_capture
        try:
            arm_state = jax.device_get(type(model)(
                model.grid, model.z_coord, model.config).step(
                    seeded, dt, freshwater=freshwater,
                    surface_forcing=surface))
            jax.effects_barrier()
        finally:
            active_live_input_family = None
            model_module._nemo_ws_rk3_tracer_pair_step = real_pair
            advection_module.fct_tracer_advection = real_fct
        arm_observation = _round113_collapse_calls(
            live_input_calls, f"{first_nonbit_family} production-JIT arm")
        arm_rows = _round113_score_inputs(arm_observation, round112_bundle)

        entry3 = gate.read_entry(
            args.entry_root / "oracle_step_entry_kt00000003.bin")
        kt3_rows = {"baseline": {}, first_nonbit_family: {}}
        for tracer in TRACERS:
            wet = reciprocal_calls[TRACERS.index(tracer)]["wet"]
            baseline_field = np.asarray(
                gate.lego_fields(baseline_state)[tracer], dtype=np.float64)
            arm_field = np.asarray(
                gate.lego_fields(arm_state)[tracer], dtype=np.float64)
            target = entry3[tracer][..., :baseline_field.shape[-1]]
            kt3_rows["baseline"][tracer] = round54.field_stats(
                baseline_field, target, wet)
            kt3_rows[first_nonbit_family][tracer] = round54.field_stats(
                arm_field, target, wet)

        observations = {
            "production_step_jit": {
                "transport_u_sha256": _array_sha256(jit_observation[0][1]),
                "adv_up1_T_sha256": _array_sha256(jit_observation[0][-1]),
            },
            "substitution_arm": {
                "transport_u_sha256": _array_sha256(arm_observation[0][1]),
                "adv_up1_T_sha256": _array_sha256(arm_observation[0][-1]),
            },
        }
        plant_fired = False
        plant_propagation = {}
        if args.plant_live_input_ulp:
            frozen = json.loads(args.live_input_report.read_text())[
                "round113_live_inputs"]
            plant_propagation = {
                key: bool(observations["substitution_arm"][key]
                          != frozen["observations"]["substitution_arm"][key])
                for key in ("transport_u_sha256", "adv_up1_T_sha256")
            }
            plant_fired = all(plant_propagation.values())
            require(plant_fired,
                    "round-113 production-JIT one-ULP input plant was inert")

        baseline_adv = jit_rows["T"]["adv_up1"]["max_abs"]
        arm_adv = arm_rows["T"]["adv_up1"]["max_abs"]
        criteria = {
            "predicted_base_bit": {
                tracer: jit_rows[tracer]["base"]["cells_unequal"] == 0
                for tracer in TRACERS
            },
            "predicted_first_nonbit_transport": (
                first_nonbit_family == "transport"),
            "transport_improves_adv_up1_T_twofold": bool(
                first_nonbit_family == "transport"
                and arm_adv * 2.0 <= baseline_adv),
            "both_adv_up1_rows_bit": all(
                arm_rows[tracer]["adv_up1"]["cells_unequal"] == 0
                for tracer in TRACERS),
            "kt3_T_improves_ten_percent": bool(
                kt3_rows[first_nonbit_family]["T"]["max_abs"]
                <= 0.9 * kt3_rows["baseline"]["T"]["max_abs"]),
        }
        round113_matrix = {
            "status": "PLANT-FIRED" if plant_fired else "MEASURED",
            "plant_ulp": bool(args.plant_live_input_ulp),
            "record_sha256": round112_bundle["sha256"],
            "plant_target": round112_bundle["plant_row"],
            "first_nonbit_family": first_nonbit_family,
            "modes": {
                "production_step_jit": jit_rows,
                "production_eager": eager_rows,
            },
            "substitution_arm": {
                "family": first_nonbit_family,
                "rows": arm_rows,
                "kt3": kt3_rows,
                "state_vs_baseline": pytree_exact_census(
                    gate.lego_fields(arm_state),
                    gate.lego_fields(baseline_state)),
            },
            "observations": observations,
            "plant_propagation": plant_propagation,
            "criteria": criteria,
        }

    round114_matrix = {}
    if args.round114_u_operands:
        require(round112_bundle is not None and round114_bundle is not None,
                "round-114 records were not prepared")
        jax.effects_barrier()
        jit_live_observation = _round113_collapse_calls(
            live_input_calls, "round-114 production-step JIT",
            allow_prior_distinct=True)
        jit_live_rows = _round113_score_inputs(
            jit_live_observation, round112_bundle)
        jit_u_observation = _round114_collapse_u(
            u_operand_calls, "production-step JIT",
            allow_prior_distinct=True)
        jit_u = _round114_score_u(jit_u_observation, round114_bundle)

        entry3 = gate.read_entry(
            args.entry_root / "oracle_step_entry_kt00000003.bin")

        def kt3_rows(state):
            result = {}
            fields = gate.lego_fields(state)
            for tracer in TRACERS:
                wet = reciprocal_calls[TRACERS.index(tracer)]["wet"]
                field = np.asarray(fields[tracer], dtype=np.float64)
                target = entry3[tracer][..., :field.shape[-1]]
                result[tracer] = round54.field_stats(field, target, wet)
            return result

        baseline_kt3 = kt3_rows(baseline_state)

        eager_u = {}
        eager_live_rows = {}
        if not args.plant_u_factor_ulp:
            live_input_calls.clear()
            u_operand_calls.clear()
            active_live_input_family = None
            active_u_factor = None
            round113_eager_direct = True
            model_module._nemo_ws_rk3_tracer_pair_step = pair_capture
            advection_module.fct_tracer_advection = fct_capture
            model_module._nemo_ws_qco_stage_faces = stage_faces_capture
            model_module._nemo_ws_stage_transport = stage_transport_capture
            try:
                try:
                    with jax.disable_jit():
                        model._step_impl(
                            seeded, dt, freshwater=freshwater,
                            surface_forcing=surface)
                except _Round113EagerBoundaryComplete:
                    pass
            finally:
                round113_eager_direct = False
                model_module._nemo_ws_rk3_tracer_pair_step = real_pair
                advection_module.fct_tracer_advection = real_fct
                model_module._nemo_ws_qco_stage_faces = real_stage_faces
                model_module._nemo_ws_stage_transport = real_stage_transport
            eager_u_observation = _round114_collapse_u(
                u_operand_calls, "production eager",
                allow_prior_distinct=True)
            eager_u = _round114_score_u(
                eager_u_observation, round114_bundle)
            eager_live = _round113_collapse_calls(
                live_input_calls, "round-114 production eager",
                allow_prior_distinct=True)
            eager_live_rows = _round113_score_inputs(
                eager_live, round112_bundle)

        def run_round114_arm(
            label: str, *, live_family: str | None = None,
            u_factor: str | None = None,
        ):
            nonlocal active_live_input_family, active_u_factor
            live_input_calls.clear()
            u_operand_calls.clear()
            active_live_input_family = live_family
            active_u_factor = u_factor
            model_module._nemo_ws_rk3_tracer_pair_step = pair_capture
            advection_module.fct_tracer_advection = fct_capture
            model_module._nemo_ws_qco_stage_faces = stage_faces_capture
            model_module._nemo_ws_stage_transport = stage_transport_capture
            try:
                state = jax.device_get(type(model)(
                    model.grid, model.z_coord, model.config).step(
                        seeded, dt, freshwater=freshwater,
                        surface_forcing=surface))
                jax.effects_barrier()
            finally:
                active_live_input_family = None
                active_u_factor = None
                model_module._nemo_ws_rk3_tracer_pair_step = real_pair
                advection_module.fct_tracer_advection = real_fct
                model_module._nemo_ws_qco_stage_faces = real_stage_faces
                model_module._nemo_ws_stage_transport = real_stage_transport
            live_observation = _round113_collapse_calls(
                live_input_calls, label, allow_prior_distinct=True)
            u_observation = _round114_collapse_u(
                u_operand_calls, label, allow_prior_distinct=True)
            return state, live_observation, u_observation

        split_arms = {}
        split_observations = {}
        if not args.plant_u_factor_ulp:
            for family in ("transport_u", "transport_v", "transport_w",
                           "transport"):
                state, live_observation, u_observation = run_round114_arm(
                    f"{family} production-JIT arm", live_family=family)
                live_rows = _round113_score_inputs(
                    live_observation, round112_bundle)
                split_arms[family] = {
                    "inputs": live_rows,
                    "kt3": kt3_rows(state),
                    "state_vs_baseline": pytree_exact_census(
                        gate.lego_fields(state),
                        gate.lego_fields(baseline_state)),
                    "producer_u": _round114_score_u(
                        u_observation, round114_bundle),
                }
                split_observations[family] = {
                    "transport_sha256": _array_sha256(
                        live_observation[0][{
                            "transport_u": 1, "transport_v": 2,
                            "transport_w": 3, "transport": 1,
                        }[family]]),
                    "adv_up1_T_sha256": _array_sha256(
                        live_observation[0][-1]),
                }

        first_direct = jit_u["first_direct_nonbit"]
        factor_family = (
            "e3u_kmm" if first_direct == "one_plus_r3u" else first_direct)
        substitutable_u = {"e3u_kmm", "u_kmm", "un_adv", "uu_b_kmm"}
        factor_state = None
        factor_live_observation = None
        factor_u_observation = None
        if factor_family in substitutable_u:
            factor_state, factor_live_observation, factor_u_observation = (
                run_round114_arm(
                    f"{factor_family} production-JIT arm",
                    u_factor=factor_family))
        require(not args.plant_u_factor_ulp or (
                    first_direct == "one_plus_r3u"
                    and factor_family == "e3u_kmm"),
                "round-114 ULP plant is calibrated only for the first "
                "r3u/e3u face-thickness family")
        require(factor_state is not None,
                "first direct non-bit U factor is not substitutable: "
                f"{first_direct!r}")
        factor_live_rows = _round113_score_inputs(
            factor_live_observation, round112_bundle)
        factor_u = _round114_score_u(
            factor_u_observation, round114_bundle)
        factor_kt3 = kt3_rows(factor_state)
        factor_observations = {
            "e3u_kmm_sha256": _array_sha256(factor_u_observation[1]),
            "zfu_sha256": _array_sha256(factor_u_observation[-1]),
            "adv_up1_T_sha256": _array_sha256(
                factor_live_observation[0][-1]),
        }

        isolated_result = jax.device_get(jax.jit(_round114_u_statement)(
            *[jnp.asarray(jit_u_observation[index]) for index in
              (0, 1, 3, 4, 5, 6, 7)]))
        isolated_observation = list(jit_u_observation)
        isolated_observation[8:] = isolated_result
        isolated_observation = tuple(isolated_observation)
        isolated_u = _round114_score_u(
            isolated_observation, round114_bundle)
        isolated_vs_production = {
            name: round54.field_stats(
                isolated_observation[index], jit_u_observation[index],
                (np.any(round114_bundle["umask"] > 0.5, axis=-1)
                 if name == "zub" else round114_bundle["umask"] > 0.5))
            for name, index in (("zub", 8), ("corrected_u", 9), ("zfu", 10))
        }

        triplet_reproduced = False
        single_family_criteria = {}
        if not args.plant_u_factor_ulp:
            triplet = split_arms["transport"]
            triplet_reproduced = bool(
                triplet["inputs"]["T"]["adv_up1"]["max_abs"]
                == np.float64(1.632720658269341e-16)
                and triplet["kt3"]["T"]["max_abs"]
                == np.float64(5.763797261693071e-8))
            baseline_adv_hash = _array_sha256(jit_live_observation[0][-1])
            for family in ("transport_u", "transport_v", "transport_w"):
                single_family_criteria[family] = {
                    "changes_adv_up1_T": (
                        split_observations[family]["adv_up1_T_sha256"]
                        != baseline_adv_hash),
                    "both_adv_rows_bit": all(
                        split_arms[family]["inputs"][tracer]["adv_up1"][
                            "cells_unequal"] == 0 for tracer in TRACERS),
                }

        factor_reduction = (
            factor_kt3["T"]["max_abs"]
            / baseline_kt3["T"]["max_abs"])
        prediction_results = {
            "e2u_predicted_bit": (
                jit_u["rows"]["e2u"]["classification"] == "BIT"),
            "predicted_first_r3u_e3u_family": (
                first_direct in ("one_plus_r3u", "e3u_kmm")),
            "factor_kt3_reduction_under_ten_percent": factor_reduction > 0.9,
            "single_family": single_family_criteria,
            "triplet_reproduces_round113": triplet_reproduced,
        }

        plant_propagation = {}
        plant_fired = False
        if args.plant_u_factor_ulp:
            frozen = json.loads(args.u_operand_report.read_text())[
                "round114_u_operands"]
            plant_propagation = {
                name: factor_observations[name]
                != frozen["factor_arm"]["observations"][name]
                for name in (
                    "e3u_kmm_sha256", "zfu_sha256", "adv_up1_T_sha256")
            }
            plant_fired = all(plant_propagation.values())
            require(plant_fired,
                    "round-114 production-JIT e3u one-ULP plant was inert")

        round114_matrix = {
            "status": "PLANT-FIRED" if plant_fired else "MEASURED",
            "plant_ulp": bool(args.plant_u_factor_ulp),
            "records": {
                name: round114_bundle[name] for name in (
                    "producer_commit", "stage_sha256",
                    "stage_baseline_sha256", "advmean_sha256",
                    "admission_sha256", "inherited_owned_exact")
            },
            "resolved_config": round114_config,
            "plant_target": round114_bundle["plant_row"],
            "first_direct_nonbit": first_direct,
            "modes": {
                "production_step_jit": jit_u,
                "production_eager": eager_u,
                "isolated_closure_jit": isolated_u,
                "isolated_vs_production": isolated_vs_production,
                "fct_consumer_production_jit": jit_live_rows,
                "fct_consumer_production_eager": eager_live_rows,
            },
            "transport_split": split_arms,
            "transport_split_observations": split_observations,
            "baseline_kt3": baseline_kt3,
            "factor_arm": {
                "first_nonbit_factor": first_direct,
                "substituted_family": factor_family,
                "u_rows": factor_u,
                "fct_inputs": factor_live_rows,
                "kt3": factor_kt3,
                "kt3_T_ratio_to_baseline": factor_reduction,
                "state_vs_baseline": pytree_exact_census(
                    gate.lego_fields(factor_state),
                    gate.lego_fields(baseline_state)),
                "observations": factor_observations,
            },
            "prediction_results": prediction_results,
            "plant_propagation": plant_propagation,
            "candidate_eligible": False,
            "candidate_disposition": (
                "diagnostic substitution only; this boundary does not "
                "distinguish a non-bit half-step ssh/r3 input from the "
                "face-geometry statement, and no source correction was "
                "constructed or proven"),
        }

    round115_matrix = {}
    if args.round115_u_geometry:
        require(round115_bundle is not None and round115_values is not None,
                "round-115 record was not prepared")
        jax.effects_barrier()
        jit_observation = _round115_collapse(
            u_geometry_calls, "production-step JIT",
            allow_prior_distinct=True)
        jit_geometry = _round115_score_geometry(
            jit_observation, round115_bundle)

        eager_geometry = {}
        if not args.plant_u_geometry_ulp:
            u_geometry_calls.clear()
            model_module._nemo_ws_stage_transport = stage_transport_capture
            model_module.nemo_qco_live_face_geometry_cgrid = (
                face_geometry_capture)
            try:
                with jax.disable_jit():
                    eager_state = model._step_impl(
                        seeded, dt, freshwater=freshwater,
                        surface_forcing=surface)
                    jax.device_get(eager_state)
                jax.effects_barrier()
            finally:
                model_module._nemo_ws_stage_transport = real_stage_transport
                model_module.nemo_qco_live_face_geometry_cgrid = (
                    real_face_geometry)
            eager_observation = _round115_collapse(
                u_geometry_calls, "production eager",
                allow_prior_distinct=True)
            eager_geometry = _round115_score_geometry(
                eager_observation, round115_bundle)

        isolated_trace = jax.device_get(jax.jit(_round115_u_trace)(
            round115_values["ssh_n1"], round115_values["area_t"],
            round115_values["r1_hu0"], round115_values["r1_area_u"]))
        isolated_full = jax.device_get(jax.jit(real_native_geometry)(
            round115_values["ssh_n1"], round115_values["e3u_0"],
            round115_values["e3v_0"], round115_values["umask"],
            round115_values["vmask"], round115_values["hu_0"],
            round115_values["hv_0"], round115_values["area_t"],
            round115_values["area_u"], round115_values["area_v"]))
        isolated_half = jax.device_get(jax.jit(_round115_half_ratio)(
            round115_values["r3u_before"], isolated_full.r3u))
        isolated_geometry = _round115_score_isolated(
            tuple(isolated_trace), np.asarray(isolated_full.r3u),
            np.asarray(isolated_half), round115_bundle)

        inputs = jit_geometry["inputs"]
        direct = jit_geometry["direct_outputs"]
        source_trace = jit_geometry["source_trace_given_nemo_inputs"]
        ordinary = direct["ordinary_one_plus_r3u"]
        ordinary_reproduced = bool(
            ordinary["cells_unequal"] == 580
            and ordinary["max_abs"]
            == np.float64(7.552691805301492e-11))
        static_inputs_bit = bool(all(
            inputs[name]["cells_unequal"] == 0
            for name in ("e1e2t", "r1_hu_0", "r1_e1e2u")))
        source_trace_bit = bool(all(
            source_trace[name]["cells_unequal"] == 0
            for name in ROUND115_TRACE_FIELDS))
        oracle_full_bit = bool(
            direct["oracle_full_r3u"]["cells_unequal"] == 0)
        oracle_half_bit = bool(
            direct["oracle_interpolated_half_r3u"]["cells_unequal"] == 0)
        ssh_nonbit = bool(
            inputs["full_step_ssh"]["cells_unequal"] > 0
            or inputs["half_step_ssh"]["cells_unequal"] > 0)
        ssh_max = max(
            inputs["full_step_ssh"]["max_abs"],
            inputs["half_step_ssh"]["max_abs"])
        if not static_inputs_bit:
            owner = "static_qco_operand"
        elif not oracle_full_bit:
            owner = "qco_u_geometry_statement"
        elif not oracle_half_bit:
            owner = "hybrid_half_ratio_interpolation"
        elif ssh_nonbit:
            owner = "external_mode_ssh_chain"
        else:
            owner = "no_nonbit_boundary"

        observations = {
            "west_product_sha256": _array_sha256(
                jit_observation[ROUND115_OBSERVATION_FIELDS.index(
                    "west_product")]),
            "east_product_sha256": _array_sha256(
                jit_observation[ROUND115_OBSERVATION_FIELDS.index(
                    "east_product")]),
            "oracle_full_r3u_sha256": _array_sha256(
                jit_observation[ROUND115_OBSERVATION_FIELDS.index(
                    "oracle_full_r3u")]),
            "oracle_interpolated_half_r3u_sha256": _array_sha256(
                jit_observation[ROUND115_OBSERVATION_FIELDS.index(
                    "oracle_interpolated_half_r3u")]),
        }
        plant_propagation = {}
        plant_fired = False
        if args.plant_u_geometry_ulp:
            frozen = json.loads(args.u_geometry_report.read_text())[
                "round115_u_geometry"]
            prior = frozen["observations"]
            product_changed = bool(
                observations["west_product_sha256"]
                != prior["west_product_sha256"]
                or observations["east_product_sha256"]
                != prior["east_product_sha256"])
            plant_propagation = {
                "written_product_changed": product_changed,
                "oracle_full_r3u_changed": (
                    observations["oracle_full_r3u_sha256"]
                    != prior["oracle_full_r3u_sha256"]),
                "oracle_interpolated_half_r3u_changed": (
                    observations["oracle_interpolated_half_r3u_sha256"]
                    != prior["oracle_interpolated_half_r3u_sha256"]),
            }
            plant_fired = all(plant_propagation.values())
            require(plant_fired,
                    "round-115 production-step SSH one-ULP plant was inert")

        prediction_results = {
            "ordinary_endpoint_reproduced": ordinary_reproduced,
            "static_inputs_bit": static_inputs_bit,
            "model_ssh_nonbit": ssh_nonbit,
            "model_ssh_max_in_frozen_range": bool(
                np.float64(1.0e-8) <= ssh_max <= np.float64(1.0e-5)),
            "source_trace_given_nemo_inputs_bit": source_trace_bit,
            "oracle_full_r3u_bit": oracle_full_bit,
            "oracle_interpolated_half_r3u_bit": oracle_half_bit,
        }
        round115_matrix = {
            "status": "PLANT-FIRED" if plant_fired else "MEASURED",
            "plant_ulp": bool(args.plant_u_geometry_ulp),
            "records": {
                "producer_commit": round115_bundle["producer_commit"],
                "stage_sha256": round115_bundle["stage_sha256"],
                "header": round115_bundle["header"],
            },
            "resolved_config": round114_config,
            "plant_target": round115_bundle["plant_row"],
            "first_nonbit_input": jit_geometry["first_nonbit_input"],
            "owner": owner,
            "modes": {
                "production_step_jit": jit_geometry,
                "production_eager": eager_geometry,
                "isolated_closure_jit": isolated_geometry,
            },
            "observations": observations,
            "prediction_results": prediction_results,
            "plant_propagation": plant_propagation,
            "candidate_eligible": owner in (
                "qco_u_geometry_statement",
                "hybrid_half_ratio_interpolation"),
            "candidate_disposition": (
                "diagnostic upstream SSH owner; no production statement is "
                "changed" if owner == "external_mode_ssh_chain" else
                "same-stage statement requires a frozen landing addendum "
                "before any candidate measurement"),
        }

    pair_matrix = {}
    kmm_matrix = {}
    if args.round70_pair or args.round71_kmm:
        original_fct_target = {
            name: np.array(value, dtype=np.float64, copy=True)
            for name, value in oracle_advection_content.items()
        }
        fct_target = {
            name: np.array(value, dtype=np.float64, copy=True)
            for name, value in original_fct_target.items()
        }
        if args.plant_pair_content_ulp:
            first_wet = tuple(np.argwhere(reciprocal_calls[0]["wet"])[0])
            fct_target["T"][first_wet] = np.nextafter(
                fct_target["T"][first_wet], np.float64(np.inf))

        def run_pair_arm(
            *, route_ldf: bool, use_fct: bool,
            kmm_override: tuple[object | None, object | None] | None = None,
        ):
            nonlocal active_kmm_override
            pair_start = len(pair_sources)
            kmm_start = len(pair_kmm)
            ldf_start = len(ldf_calls)
            override = None
            if use_fct:
                override = (jnp.asarray(fct_target["T"]),
                            jnp.asarray(fct_target["S"]))
            model_module._nemo_ws_rk3_tracer_pair_step = pair_capture
            model_module.gm_redi_tracer_tendency_latlon = ldf_capture
            active_kmm_override = kmm_override
            try:
                arm_state = jax.device_get(type(model)(
                    model.grid, model.z_coord, model.config,
                    _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
                        route_gm_redi_stage3_source=route_ldf,
                        stage3_advection_content_override=override),
                ).step(seeded, dt, freshwater=freshwater,
                       surface_forcing=surface))
            finally:
                active_kmm_override = None
                model_module._nemo_ws_rk3_tracer_pair_step = real_pair
                model_module.gm_redi_tracer_tendency_latlon = real_ldf
            pair_run = pair_sources[pair_start:]
            kmm_run = pair_kmm[kmm_start:]
            ldf_run = ldf_calls[ldf_start:]
            require(pair_run and kmm_run and ldf_run,
                    "FCT arm did not execute production captures")
            require(all(all(np.array_equal(value, reference)
                            for value, reference in zip(
                                duplicate, pair_run[0], strict=True))
                        for duplicate in pair_run[1:]),
                    "round-70 arm observed distinct content evaluations")
            require(all(all(np.array_equal(value, reference)
                            for value, reference in zip(
                                duplicate, ldf_run[0], strict=True))
                        for duplicate in ldf_run[1:]),
                    "round-70 arm observed distinct LDF evaluations")
            require(all(all(np.array_equal(value, reference)
                            for value, reference in zip(
                                duplicate, kmm_run[0], strict=True))
                        for duplicate in kmm_run[1:]),
                    "FCT arm observed distinct Kmm evaluations")
            return arm_state, pair_run[0], ldf_run[0], kmm_run[0]

        untouched = run_pair_arm(route_ldf=False, use_fct=False)
        ldf_only = run_pair_arm(route_ldf=True, use_fct=False)
        if args.plant_pair_null_fct:
            fct_target = {
                "T": np.array(untouched[1][6], copy=True),
                "S": np.array(untouched[1][7], copy=True),
            }
        fct_only = run_pair_arm(route_ldf=False, use_fct=True)
        paired = run_pair_arm(route_ldf=True, use_fct=True)
        arms = {
            "untouched": untouched,
            "ldf_only": ldf_only,
            "fct_only": fct_only,
            "paired": paired,
        }
        entry3_pair = gate.read_entry(
            args.entry_root / "oracle_step_entry_kt00000003.bin")
        arm_rows = {}
        criteria = {
            "ordinary_state_exact": pytree_exact_census(
                untouched[0], baseline_state)["cells_unequal"] == 0,
            "ordinary_content_exact": {},
            "same_step_ldf_exact": {},
            "fct_target_exact": {},
            "active_content_moves": {},
            "active_state_moves": {},
            "finite": {},
            "paired_content_ceiling": {},
            "paired_improves_ldf_eightfold": {},
            "paired_kt3_improves_both": {},
            "ldf_worsening_removed_99pct": {},
        }
        for name, capture in zip(TRACERS, reciprocal_calls, strict=True):
            index = TRACERS.index(name)
            wet = capture["wet"]
            oracle_content = capture["oracle_content"]
            oracle_entry = entry3_pair[name][..., :np.asarray(
                gate.lego_fields(baseline_state)[name]).shape[-1]]
            untouched_content = untouched[1][4 + index]
            untouched_adv = untouched[1][6 + index]
            untouched_field = np.asarray(
                gate.lego_fields(untouched[0])[name], dtype=np.float64)
            arm_rows[name] = {}
            for arm_name, (arm_state, arm_pair, arm_ldf, _) in arms.items():
                field = np.asarray(
                    gate.lego_fields(arm_state)[name], dtype=np.float64)
                content = arm_pair[4 + index]
                arm_rows[name][arm_name] = {
                    "content_vs_oracle": round54.field_stats(
                        content, oracle_content, wet),
                    "content_vs_untouched": round54.field_stats(
                        content, untouched_content, wet),
                    "kt3_vs_oracle": round54.field_stats(
                        field, oracle_entry, wet),
                    "kt3_vs_untouched": round54.field_stats(
                        field, untouched_field, wet),
                    "source_vs_expected": round54.field_stats(
                        arm_pair[index],
                        (untouched[1][index] + untouched[2][index]
                         * wet.astype(np.float64)
                         if arm_name in ("ldf_only", "paired")
                         else untouched[1][index]), wet),
                    "advection_content_vs_target": round54.field_stats(
                        arm_pair[6 + index],
                        (fct_target[name] if arm_name in ("fct_only", "paired")
                         else untouched_adv), wet),
                }
                criteria["finite"][f"{name}_{arm_name}"] = bool(
                    np.all(np.isfinite(field))
                    and np.all(np.isfinite(content))
                    and np.all(np.isfinite(arm_ldf[index])))
            criteria["ordinary_content_exact"][name] = bool(
                round54.field_stats(
                    untouched_content, captured_content[name], wet)[
                        "cells_unequal"] == 0)
            criteria["same_step_ldf_exact"][name] = bool(all(
                np.array_equal(arm[2][index], untouched[2][index])
                for arm in arms.values()))
            frozen_target_row = round54.field_stats(
                fct_only[1][6 + index], original_fct_target[name], wet)
            criteria["fct_target_exact"][name] = bool(
                frozen_target_row["cells_unequal"] == 0)
            for arm_name in ("ldf_only", "fct_only", "paired"):
                criteria["active_content_moves"][f"{name}_{arm_name}"] = bool(
                    arm_rows[name][arm_name]["content_vs_untouched"][
                        "cells_unequal"] > 0)
                criteria["active_state_moves"][f"{name}_{arm_name}"] = bool(
                    arm_rows[name][arm_name]["kt3_vs_untouched"][
                        "cells_unequal"] > 0)
            paired_max = arm_rows[name]["paired"]["content_vs_oracle"][
                "max_abs"]
            ldf_max = arm_rows[name]["ldf_only"]["content_vs_oracle"][
                "max_abs"]
            ceiling = 6.0e-6 if name == "T" else 1.0e-6
            criteria["paired_content_ceiling"][name] = bool(
                paired_max <= ceiling)
            criteria["paired_improves_ldf_eightfold"][name] = bool(
                paired_max * 8.0 <= ldf_max)
            paired_kt3 = arm_rows[name]["paired"]["kt3_vs_oracle"]["max_abs"]
            criteria["paired_kt3_improves_both"][name] = bool(
                paired_kt3
                < arm_rows[name]["untouched"]["kt3_vs_oracle"]["max_abs"]
                and paired_kt3
                < arm_rows[name]["ldf_only"]["kt3_vs_oracle"]["max_abs"])
            ldf_worsening, ldf_mask = worsening_census(
                untouched_field,
                np.asarray(gate.lego_fields(ldf_only[0])[name]),
                oracle_entry, wet)
            paired_worsening, paired_mask = worsening_census(
                untouched_field,
                np.asarray(gate.lego_fields(paired[0])[name]),
                oracle_entry, wet)
            retained = int(np.count_nonzero(ldf_mask & paired_mask))
            denominator = ldf_worsening["cells_worsened"]
            removed_fraction = (
                1.0 if denominator == 0 else 1.0 - retained / denominator)
            criteria["ldf_worsening_removed_99pct"][name] = bool(
                removed_fraction >= 0.99)
            arm_rows[name]["worsening"] = {
                "ldf_only": ldf_worsening,
                "paired": paired_worsening,
                "ldf_cells_still_worsened_by_pair": retained,
                "ldf_worsening_removed_fraction": removed_fraction,
                "frozen_target": frozen_target_row,
            }

        def all_true(value) -> bool:
            if isinstance(value, dict):
                return all(all_true(item) for item in value.values())
            return bool(value)

        pair_matrix = {
            "status": (
                "CONFIRMED" if all_true(criteria) else "REFUTED"),
            "plant_null_fct": bool(args.plant_pair_null_fct),
            "plant_content_ulp": bool(args.plant_pair_content_ulp),
            "rows": arm_rows,
            "criteria": criteria,
        }

        if args.round71_kmm:
            stage3 = round66.round46.read_stage(
                args.stage_root / "oracle_momstage_kt00000002_s3.bin")
            require(stage3["header"] == {
                "version": 1, "kt": 2, "stage": 3,
                "Kbb": 3, "Kmm": 2, "Krhs": 1, "Kaa": 1,
                "jpi": 36, "jpj": 26, "jpk": 31, "jpkm1": 30,
                "ntsi": 3, "ntei": 34, "ntsj": 3, "ntej": 24,
                "bits": 64,
            }, "round-46 kt2 stage-3 header changed")
            original_kmm_target = {
                name: np.asarray(round66.round46._owned3(
                    stage3["arrays"][f"{name}_Kmm"]), dtype=np.float64)
                for name in TRACERS
            }
            kmm_target = {
                name: np.array(value, dtype=np.float64, copy=True)
                for name, value in original_kmm_target.items()
            }
            if args.plant_kmm_null:
                kmm_target = {
                    "T": np.array(ldf_only[3][0], copy=True),
                    "S": np.array(ldf_only[3][1], copy=True),
                }
            if args.plant_kmm_content_ulp:
                first_wet = tuple(np.argwhere(reciprocal_calls[0]["wet"])[0])
                kmm_target["T"][first_wet] = np.nextafter(
                    kmm_target["T"][first_wet], np.float64(np.inf))

            kmm_t = run_pair_arm(
                route_ldf=True, use_fct=False,
                kmm_override=(jnp.asarray(kmm_target["T"]), None))
            kmm_s = run_pair_arm(
                route_ldf=True, use_fct=False,
                kmm_override=(None, jnp.asarray(kmm_target["S"])))
            kmm_ts = run_pair_arm(
                route_ldf=True, use_fct=False,
                kmm_override=(jnp.asarray(kmm_target["T"]),
                              jnp.asarray(kmm_target["S"])))
            kmm_arms = {"kmm_T": kmm_t, "kmm_S": kmm_s, "kmm_TS": kmm_ts}
            kmm_rows = {name: {} for name in TRACERS}
            kmm_criteria = {
                "round70_rows_exact": False,
                "round70_criteria_exact": False,
                "source_operand_non_bit": {},
                "injected_target_exact": {},
                "cross_tracer_isolation": {},
                "joint_equals_single": {},
                "same_step_ldf_exact": {},
                "active_content_moves": {},
                "active_state_moves": {},
                "finite": {},
                "kt3_improves_ldf_twofold": {},
                "kt3_remains_worse_than_output_pair": {},
                "retained_set_reduced_half": {},
                "retained_debt_remains": {},
            }
            frozen_round70 = json.loads(args.round70_report.read_text())[
                "round70_pair_matrix"]
            kmm_criteria["round70_rows_exact"] = bool(
                pair_matrix["rows"] == frozen_round70["rows"])
            kmm_criteria["round70_criteria_exact"] = bool(
                pair_matrix["criteria"] == frozen_round70["criteria"])

            source_split = {}
            for name, capture in zip(TRACERS, reciprocal_calls, strict=True):
                index = TRACERS.index(name)
                wet = capture["wet"]
                oracle_entry = entry3_pair[name][..., :np.asarray(
                    gate.lego_fields(baseline_state)[name]).shape[-1]]
                untouched_field = np.asarray(
                    gate.lego_fields(untouched[0])[name], dtype=np.float64)
                ldf_field = np.asarray(
                    gate.lego_fields(ldf_only[0])[name], dtype=np.float64)
                paired_field = np.asarray(
                    gate.lego_fields(paired[0])[name], dtype=np.float64)
                original_target = original_kmm_target[name]
                live_kmm = ldf_only[3][index]
                source_row = round54.field_stats(
                    live_kmm, original_target, wet)
                expected_count = 17994 if name == "T" else 16769
                kmm_criteria["source_operand_non_bit"][name] = bool(
                    source_row["cells_unequal"] == expected_count
                    and source_row["max_abs"] > 0.0)

                for arm_name, arm in kmm_arms.items():
                    field = np.asarray(
                        gate.lego_fields(arm[0])[name], dtype=np.float64)
                    content = arm[1][4 + index]
                    kmm_rows[name][arm_name] = {
                        "content_vs_oracle": round54.field_stats(
                            content, capture["oracle_content"], wet),
                        "content_vs_ldf_only": round54.field_stats(
                            content, ldf_only[1][4 + index], wet),
                        "kt3_vs_oracle": round54.field_stats(
                            field, oracle_entry, wet),
                        "kt3_vs_ldf_only": round54.field_stats(
                            field, ldf_field, wet),
                        "injected_kmm_vs_original_target": round54.field_stats(
                            arm[3][index], original_target, wet),
                    }
                    kmm_criteria["finite"][f"{name}_{arm_name}"] = bool(
                        np.all(np.isfinite(field))
                        and np.all(np.isfinite(content))
                        and np.all(np.isfinite(arm[3][index])))
                    kmm_criteria["same_step_ldf_exact"][
                        f"{name}_{arm_name}"] = bool(np.array_equal(
                            arm[2][index], ldf_only[2][index]))

                target_arm = kmm_t if name == "T" else kmm_s
                other_arm = kmm_s if name == "T" else kmm_t
                kmm_criteria["injected_target_exact"][name] = bool(
                    kmm_rows[name]["kmm_TS"][
                        "injected_kmm_vs_original_target"][
                            "cells_unequal"] == 0)
                kmm_criteria["cross_tracer_isolation"][name] = bool(
                    round54.field_stats(
                        np.asarray(gate.lego_fields(other_arm[0])[name]),
                        ldf_field, wet)["cells_unequal"] == 0
                    and round54.field_stats(
                        other_arm[1][4 + index],
                        ldf_only[1][4 + index], wet)["cells_unequal"] == 0)
                kmm_criteria["joint_equals_single"][name] = bool(
                    round54.field_stats(
                        np.asarray(gate.lego_fields(kmm_ts[0])[name]),
                        np.asarray(gate.lego_fields(target_arm[0])[name]),
                        wet)["cells_unequal"] == 0
                    and round54.field_stats(
                        kmm_ts[1][4 + index], target_arm[1][4 + index], wet)[
                            "cells_unequal"] == 0)
                kmm_criteria["active_content_moves"][name] = bool(
                    kmm_rows[name]["kmm_TS"]["content_vs_ldf_only"][
                        "cells_unequal"] > 0)
                kmm_criteria["active_state_moves"][name] = bool(
                    kmm_rows[name]["kmm_TS"]["kt3_vs_ldf_only"][
                        "cells_unequal"] > 0)

                ldf_worsening, ldf_mask = worsening_census(
                    untouched_field, ldf_field, oracle_entry, wet)
                paired_worsening, paired_mask = worsening_census(
                    untouched_field, paired_field, oracle_entry, wet)
                kmm_field = np.asarray(
                    gate.lego_fields(kmm_ts[0])[name], dtype=np.float64)
                kmm_worsening, kmm_mask = worsening_census(
                    untouched_field, kmm_field, oracle_entry, wet)
                retained_mask = ldf_mask & paired_mask
                new_pair_mask = (~ldf_mask) & paired_mask & wet
                retained_count = int(np.count_nonzero(retained_mask))
                new_pair_count = int(np.count_nonzero(new_pair_mask))
                expected_retained = 1375 if name == "T" else 1891
                expected_new = 1800 if name == "T" else 6374
                require(retained_count == expected_retained,
                        f"round-70 retained {name} set changed")
                require(new_pair_count == expected_new,
                        f"round-70 newly-worsened {name} set changed")
                retained_kmm = int(np.count_nonzero(retained_mask & kmm_mask))
                new_pair_kmm = int(np.count_nonzero(new_pair_mask & kmm_mask))
                retained_removed_fraction = 1.0 - retained_kmm / retained_count
                source_split[name] = {
                    "live_kmm_vs_oracle": source_row,
                    "ldf_only_worsening": ldf_worsening,
                    "output_pair_worsening": paired_worsening,
                    "kmm_pair_worsening": kmm_worsening,
                    "round70_retained_cells": retained_count,
                    "round70_newly_worsened_cells": new_pair_count,
                    "retained_cells_still_worsened_by_kmm": retained_kmm,
                    "retained_cells_removed_fraction": retained_removed_fraction,
                    "round70_new_cells_worsened_by_kmm": new_pair_kmm,
                }
                kmm_max = kmm_rows[name]["kmm_TS"][
                    "kt3_vs_oracle"]["max_abs"]
                ldf_max = arm_rows[name]["ldf_only"][
                    "kt3_vs_oracle"]["max_abs"]
                output_pair_max = arm_rows[name]["paired"][
                    "kt3_vs_oracle"]["max_abs"]
                kmm_criteria["kt3_improves_ldf_twofold"][name] = bool(
                    2.0 * kmm_max <= ldf_max)
                kmm_criteria["kt3_remains_worse_than_output_pair"][name] = bool(
                    kmm_max > output_pair_max)
                kmm_criteria["retained_set_reduced_half"][name] = bool(
                    retained_removed_fraction >= 0.5)
                kmm_criteria["retained_debt_remains"][name] = bool(
                    retained_kmm > 0)

            kmm_matrix = {
                "status": (
                    "CONFIRMED" if all_true(kmm_criteria) else "REFUTED"),
                "plant_null": bool(args.plant_kmm_null),
                "plant_content_ulp": bool(args.plant_kmm_content_ulp),
                "dtype": {
                    "live_kmm": str(np.asarray(ldf_only[3][0]).dtype),
                    "injected_kmm": str(np.asarray(kmm_ts[3][0]).dtype),
                    "record_kmm": str(original_kmm_target["T"].dtype),
                },
                "rows": kmm_rows,
                "source_split": source_split,
                "criteria": kmm_criteria,
            }

    if args.round69_native:
        require(args.expect_model_order == "after",
                "the landed round-69 gate requires production LDF routing")
        require(not args.plant_native_null and not args.plant_native_content_ulp,
                "round-69 plants belong to the committed pre-edit instrument")
    native_state = baseline_state
    native_pair = (
        source_t, source_s, pair_t, pair_s, content_t, content_s,
        advection_content_t, advection_content_s)
    native_ldf = (ldf_t, ldf_s)
    false_pair = native_pair
    false_pair_for_control = list(false_pair)
    route_enabled = args.expect_model_order == "after"
    false_state_exact = pytree_exact_census(baseline_state, baseline_state)
    native_state_move = pytree_exact_census(native_state, baseline_state)
    false_content_exact = {
        "T": round54.field_stats(
            false_pair_for_control[4], captured_content["T"],
            reciprocal_calls[0]["wet"]),
        "S": round54.field_stats(
            false_pair_for_control[5], captured_content["S"],
            reciprocal_calls[1]["wet"]),
    }

    override_state = type(model)(
        model.grid, model.z_coord, model.config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            pre_implicit_tracer_content_override=(
                jnp.asarray(routed_content["T"]),
                jnp.asarray(routed_content["S"]),
            )),
    ).step(seeded, dt, freshwater=freshwater, surface_forcing=surface)
    baseline_fields = gate.lego_fields(baseline_state)
    override_fields = gate.lego_fields(jax.device_get(override_state))
    production_override_state = type(model)(
        model.grid, model.z_coord, model.config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            pre_implicit_tracer_content_override=(
                jnp.asarray(production_prediction_content["T"]),
                jnp.asarray(production_prediction_content["S"]),
            )),
    ).step(seeded, dt, freshwater=freshwater, surface_forcing=surface)
    production_override_fields = gate.lego_fields(
        jax.device_get(production_override_state))
    native_fields = gate.lego_fields(native_state)
    entry3 = gate.read_entry(args.entry_root / "oracle_step_entry_kt00000003.bin")
    kt3 = {}
    production_kt3 = {}
    native_source_rows = {}
    for name, capture in zip(TRACERS, reciprocal_calls, strict=True):
        tracer_index = TRACERS.index(name)
        native_source = native_pair[tracer_index]
        false_source = false_pair[tracer_index]
        native_content = native_pair[4 + tracer_index]
        expected_source = native_source
        oracle_entry = entry3[name][..., :baseline_fields[name].shape[-1]]
        kt3[name] = {
            "baseline": round54.field_stats(
                baseline_fields[name], oracle_entry, capture["wet"]),
            "routed_content_override": round54.field_stats(
                override_fields[name], oracle_entry, capture["wet"]),
            "override_vs_baseline": round54.field_stats(
                override_fields[name], baseline_fields[name], capture["wet"]),
        }
        production_kt3[name] = {
            "baseline": round54.field_stats(
                baseline_fields[name], oracle_entry, capture["wet"]),
            "production_content_override": round54.field_stats(
                production_override_fields[name], oracle_entry, capture["wet"]),
            "override_vs_baseline": round54.field_stats(
                production_override_fields[name], baseline_fields[name],
                capture["wet"]),
        }
        native_source_rows[name] = {
            "source_injection": round54.field_stats(
                native_source, expected_source, capture["wet"]),
            "content_vs_oracle": round54.field_stats(
                native_content, capture["oracle_content"], capture["wet"]),
            "content_vs_false": round54.field_stats(
                native_content, false_pair[4 + tracer_index], capture["wet"]),
            "content_vs_host_reconstruction": round54.field_stats(
                native_content, production_prediction_content[name],
                capture["wet"]),
            "kt3_vs_oracle": round54.field_stats(
                native_fields[name], oracle_entry, capture["wet"]),
            "kt3_vs_false": round54.field_stats(
                native_fields[name], baseline_fields[name], capture["wet"]),
            "finite": bool(
                np.all(np.isfinite(native_source))
                and np.all(np.isfinite(native_content))
                and np.all(np.isfinite(np.asarray(native_fields[name])))),
        }

    if args.round111_fct_split:
        round111_matrix["local_rows"] = {
            name: {
                "content_vs_oracle": implementation_oracle_content[name],
                "kt3_vs_oracle": kt3[name]["baseline"],
            }
            for name in TRACERS
        }
        round111_matrix["criteria"]["frozen_kt3_max"] = {
            name: bool(
                kt3[name]["baseline"]["max_abs"]
                == local_frozen[name]["kt3"])
            for name in TRACERS
        }
        if not args.plant_fct_split_ulp:
            round111_matrix["status"] = (
                "CONFIRMED" if all(
                    all(value.values()) if isinstance(value, dict) else value
                    for value in round111_matrix["criteria"].values())
                else "REFUTED")

    prediction_match = {name: True for name in TRACERS}
    override_match = {name: True for name in TRACERS}
    if args.expect_model_order == "after":
        for name in TRACERS:
            prediction_match[name] = bool(
                kt3[name]["baseline"]
                == prediction["kt3_prediction"][name]["routed_content_override"])
            override_match[name] = bool(
                kt3[name]["override_vs_baseline"]["cells_unequal"] == 0)

    status = (
        "CONFIRMED"
        if (all(content_criterion.values())
            and all(prediction_match.values())
            and all(override_match.values()))
        else "REFUTED")

    production_criteria = {
        "baseline_rebuild_exact": {
            name: production_baseline_rebuild[name]["cells_unequal"] == 0
            for name in TRACERS
        },
        "implementation_exact": {
            name: production_prediction_rows[name][
                "implementation_vs_prediction"]["cells_unequal"] == 0
            for name in TRACERS
        },
        "frozen_prediction_exact": {name: True for name in TRACERS},
        "frozen_kt3_exact": {name: True for name in TRACERS},
        "override_equals_implementation": {name: True for name in TRACERS},
    }
    if args.expect_production_route == "after":
        production_report = json.loads(
            args.production_prediction_report.read_text())
        frozen_rows = production_report["production_fct_prediction"]
        for name in TRACERS:
            production_criteria["frozen_prediction_exact"][name] = bool(
                production_prediction_rows[name]["content_vs_oracle"]
                == frozen_rows["content_rows"][name]["content_vs_oracle"])
            production_criteria["frozen_kt3_exact"][name] = bool(
                production_kt3[name]["baseline"]
                == frozen_rows["kt3"][name]["production_content_override"])
            production_criteria["override_equals_implementation"][name] = bool(
                production_kt3[name]["override_vs_baseline"][
                    "cells_unequal"] == 0)
        status = (
            "CONFIRMED"
            if all(all(rows.values()) for rows in production_criteria.values())
            else "REFUTED")
    else:
        round67_report = json.loads(args.prediction_report.read_text())
        prediction_t = production_prediction_rows["T"]["content_vs_oracle"]
        baseline_t = base["substitution_rows"]["T"]["live_baseline"]
        improves_20x = bool(
            prediction_t["max_abs"] * 20.0 <= baseline_t["max_abs"])
        within_round67_floor = {}
        for name in TRACERS:
            frozen = round67_report["cumulative_substitutions"][name][
                "arm_rows"]["routed_live"]["content"]["max_abs"]
            association = production_vs_round67_routed[name]["max_abs"]
            within_round67_floor[name] = bool(
                production_prediction_rows[name]["content_vs_oracle"][
                    "max_abs"] <= frozen + association)
        production_criteria["improves_t_20x"] = {"T": improves_20x}
        production_criteria["within_round67_floor"] = within_round67_floor
        status = (
            "CONFIRMED"
            if (all(production_criteria["baseline_rebuild_exact"].values())
                and improves_20x
                and all(within_round67_floor.values()))
            else "REFUTED")

    native_criteria = {}
    if args.round69_native:
        round68_report = json.loads(args.production_prediction_report.read_text())
        frozen_host = round68_report["production_fct_prediction"]
        native_prediction = json.loads(args.native_prediction_report.read_text())
        frozen_native = native_prediction["native_source_arm"]
        native_improves_t_20x = bool(
            native_source_rows["T"]["content_vs_oracle"]["max_abs"] * 20.0
            <= round68_report["implementation_content_vs_oracle"]["T"][
                "max_abs"])
        native_floors = {
            "T": np.float64(5.954039670541533e-5),
            "S": np.float64(7.651457963220310e-6),
        }
        within_native_floor = {
            name: bool(
                native_source_rows[name]["content_vs_oracle"]["max_abs"]
                <= native_floors[name]
                + native_source_rows[name][
                    "content_vs_host_reconstruction"]["max_abs"])
            for name in TRACERS
        }
        native_criteria = {
            "frozen_content_metrics_exact": {
                name: bool(
                    native_source_rows[name]["content_vs_oracle"]
                    == frozen_native["rows"][name]["content_vs_oracle"])
                for name in TRACERS
            },
            "frozen_kt3_metrics_exact": {
                name: bool(
                    native_source_rows[name]["kt3_vs_oracle"]
                    == frozen_native["rows"][name]["kt3_vs_oracle"])
                for name in TRACERS
            },
            "private_arm_removed": bool(
                "route_gm_redi_stage3_source"
                not in model_module._NEMOWSRK3TestHooks._fields),
            "finite": {
                name: native_source_rows[name]["finite"]
                for name in TRACERS
            },
            "improves_t_20x": native_improves_t_20x,
            "within_native_floor": within_native_floor,
            "round68_host_retraction_retained": bool(
                frozen_host["baseline_rebuild"]["T"]["cells_unequal"] == 3
                and frozen_host["baseline_rebuild"]["S"][
                    "cells_unequal"] == 0),
            "native_association_census_reproduced": {
                name: bool(
                    production_baseline_rebuild[name]["cells_unequal"]
                    == frozen_native["rows"][name][
                        "content_vs_host_reconstruction"]["cells_unequal"]
                    and production_baseline_rebuild[name]["max_abs"]
                    == frozen_native["rows"][name][
                        "content_vs_host_reconstruction"]["max_abs"])
                for name in TRACERS
            },
            "pre_edit_plants": {
                "null_exit": 1,
                "content_ulp_exit": 1,
            },
        }

        def all_true(value) -> bool:
            if isinstance(value, dict):
                return all(all_true(item) for item in value.values())
            return bool(value)

        status = (
            "CONFIRMED"
            if all(all_true(value) for value in native_criteria.values())
            else "REFUTED")
    if args.round70_pair:
        status = pair_matrix["status"]
    if args.round71_kmm:
        status = kmm_matrix["status"]
    if args.round111_fct_split:
        status = round111_matrix["status"]
    if args.round112_fct_walk:
        status = round112_matrix["status"]
    if args.round113_live_inputs:
        status = round113_matrix["status"]
    if args.round114_u_operands:
        status = round114_matrix["status"]
    if args.round115_u_geometry:
        status = round115_matrix["status"]

    return {
        "format": (
            "nemo-testcase-l2-gyre-round115-u-geometry-v1"
            if args.round115_u_geometry else
            "nemo-testcase-l2-gyre-round114-u-operands-v1"
            if args.round114_u_operands else
            "nemo-testcase-l2-gyre-round113-live-inputs-v1"
            if args.round113_live_inputs else
            "nemo-testcase-l2-gyre-round112-fct-walk-v1"
            if args.round112_fct_walk else
            "nemo-testcase-l2-gyre-round111-fct-split-v1"
            if args.round111_fct_split else
            "nemo-testcase-l2-gyre-round71-fct-kmm-v1"
            if args.round71_kmm else
            "nemo-testcase-l2-gyre-round70-fct-ldf-pair-v1"
            if args.round70_pair else
            "nemo-testcase-l2-gyre-round69-native-source-v1"
            if args.round69_native else
            "nemo-testcase-l2-gyre-round68-production-fct-v1"),
        "status": status,
        "worktree": base["worktree"],
        "record_producer": base["record_producer"],
        "admission_counts": base["admission_counts"],
        "dtype": base["dtype"],
        "cumulative_substitutions": results,
        "implementation_content_vs_routed": implementation_content,
        "implementation_content_vs_oracle": implementation_oracle_content,
        "kt3_prediction": kt3,
        "production_fct_prediction": {
            "content_rows": production_prediction_rows,
            "baseline_rebuild": production_baseline_rebuild,
            "vs_round67_recomputed_routed": production_vs_round67_routed,
            "kt3": production_kt3,
            "criteria": production_criteria,
        },
        "native_source_arm": {
            "enabled": bool(args.round69_native),
            "route_enabled": bool(route_enabled),
            "default_state_exact": false_state_exact,
            "default_content_exact": false_content_exact,
            "state_move": native_state_move,
            "rows": native_source_rows,
            "criteria": native_criteria,
        },
        "round70_pair_matrix": pair_matrix,
        "round71_kmm_matrix": kmm_matrix,
        "round111_fct_split": round111_matrix,
        "round112_fct_walk": round112_matrix,
        "round113_live_inputs": round113_matrix,
        "round114_u_operands": round114_matrix,
        "round115_u_geometry": round115_matrix,
        "criteria": {
            "content_within_floor_plus_association": content_criterion,
            "kt3_exact_pre_edit_prediction": prediction_match,
            "content_override_equals_implementation": override_match,
        },
        "controls": {"one_ulp_content": plant,
                     "all_oracle_exact": oracle_exact,
                     "post_ldf_round66_reproduction": post_ldf},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument(
        "--expect-model-order", choices=("before", "after"), required=True)
    parser.add_argument(
        "--expect-production-route", choices=("before", "after"),
        default="before")
    parser.add_argument("--round69-native", action="store_true")
    parser.add_argument("--plant-native-null", action="store_true")
    parser.add_argument("--plant-native-content-ulp", action="store_true")
    parser.add_argument("--round70-pair", action="store_true")
    parser.add_argument("--plant-pair-null-fct", action="store_true")
    parser.add_argument("--plant-pair-content-ulp", action="store_true")
    parser.add_argument("--round71-kmm", action="store_true")
    parser.add_argument("--plant-kmm-null", action="store_true")
    parser.add_argument("--plant-kmm-content-ulp", action="store_true")
    parser.add_argument("--round111-fct-split", action="store_true")
    parser.add_argument("--plant-fct-split-ulp", action="store_true")
    parser.add_argument(
        "--fct-split-report", type=Path,
        default=ROOT / "round111/fct_split_jit.json")
    parser.add_argument("--round112-fct-walk", action="store_true")
    parser.add_argument("--plant-fct-walk-ulp", action="store_true")
    parser.add_argument(
        "--fct-walk-record", type=Path,
        default=(ROOT / "round111/oracle_fct_writers/"
                 "oracle_fct_writers_kt00000002_s3.bin"))
    parser.add_argument(
        "--fct-walk-report", type=Path,
        default=ROOT / "round112/fct_walk_before.json")
    parser.add_argument("--round113-live-inputs", action="store_true")
    parser.add_argument("--plant-live-input-ulp", action="store_true")
    parser.add_argument(
        "--live-input-report", type=Path,
        default=ROOT / "round113/live_inputs.json")
    parser.add_argument("--round114-u-operands", action="store_true")
    parser.add_argument("--plant-u-factor-ulp", action="store_true")
    parser.add_argument("--expect-u-record-commit")
    parser.add_argument(
        "--u-record-root", type=Path,
        default=ROOT / "round77/oracle_uamid_kt2")
    parser.add_argument(
        "--u-admission", type=Path,
        default=ROOT / "round77/oracle_uamid_kt2/round77_admission.json")
    parser.add_argument(
        "--stage-baseline-root", type=Path,
        default=ROOT / "round46/oracle_kt2_stage")
    parser.add_argument(
        "--u-operand-report", type=Path,
        default=ROOT / "round114/u_transport_walk.json")
    parser.add_argument("--round115-u-geometry", action="store_true")
    parser.add_argument("--plant-u-geometry-ulp", action="store_true")
    parser.add_argument(
        "--u-geometry-report", type=Path,
        default=ROOT / "round115/u_geometry_walk.json")
    parser.add_argument(
        "--prediction-report", type=Path,
        default=ROOT / "round67/round67_ldf_order_before.json")
    parser.add_argument(
        "--production-prediction-report", type=Path,
        default=ROOT / "round68/round68_production_fct_before.json")
    parser.add_argument(
        "--native-prediction-report", type=Path,
        default=ROOT / "round69/round69_native_source_before.json")
    parser.add_argument(
        "--round70-report", type=Path,
        default=ROOT / "round70/round70_pair_before.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry-root", type=Path,
                        default=ROOT / "year_owners/nemo_seed0")
    parser.add_argument("--stage-root", type=Path,
                        default=ROOT / "round46/oracle_kt2_stage")
    parser.add_argument("--krhs-record", type=Path, default=(
        ROOT / "round64/oracle_krhs_split/oracle_krhs_split_kt00000002.bin"))
    parser.add_argument("--krhs-stamp", type=Path, default=(
        ROOT / "round64/oracle_krhs_split/oracle_krhs_split_kt00000002.bin.stamp"))
    parser.add_argument("--producer-commit", type=Path,
                        default=ROOT / "round64/oracle_krhs_split/producer_commit.txt")
    parser.add_argument("--admission", type=Path, default=(
        ROOT / "round64/oracle_krhs_split/round64_admission.json"))
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except Exception as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    if args.round115_u_geometry:
        matrix = report["round115_u_geometry"]
        inputs = matrix["modes"]["production_step_jit"]["inputs"]
        outputs = matrix["modes"]["production_step_jit"]["direct_outputs"]
        prefix = (
            "STATUS PLANT-FIRED" if report["status"] == "PLANT-FIRED"
            else f"ROUND115 U GEOMETRY {report['status']}")
        print(
            f"{prefix}: first={matrix['first_nonbit_input']} "
            f"owner={matrix['owner']} "
            f"ssh_full={inputs['full_step_ssh']['cells_unequal']}/"
            f"{inputs['full_step_ssh']['max_abs']:.12e} "
            f"oracle_r3u={outputs['oracle_full_r3u']['cells_unequal']}/"
            f"{outputs['oracle_full_r3u']['max_abs']:.12e} "
            f"half={outputs['oracle_interpolated_half_r3u']['cells_unequal']}/"
            f"{outputs['oracle_interpolated_half_r3u']['max_abs']:.12e}")
    elif args.round114_u_operands:
        matrix = report["round114_u_operands"]
        row = matrix["modes"]["production_step_jit"]["rows"]
        arm = matrix["factor_arm"]
        prefix = (
            "STATUS PLANT-FIRED" if report["status"] == "PLANT-FIRED"
            else f"ROUND114 U OPERANDS {report['status']}")
        print(
            f"{prefix}: first={matrix['first_direct_nonbit']} "
            f"e2u={row['e2u']['cells_unequal']}/"
            f"{row['e2u']['max_abs']:.12e} "
            f"e3u={row['e3u_kmm']['cells_unequal']}/"
            f"{row['e3u_kmm']['max_abs']:.12e} "
            f"zfu={row['zfu']['cells_unequal']}/"
            f"{row['zfu']['max_abs']:.12e} "
            f"factor_kt3_T={arm['kt3']['T']['max_abs']:.12e}")
    elif args.round113_live_inputs:
        matrix = report["round113_live_inputs"]
        row = matrix["modes"]["production_step_jit"]["T"]
        arm = matrix["substitution_arm"]["rows"]["T"]["adv_up1"]
        prefix = (
            "STATUS PLANT-FIRED" if report["status"] == "PLANT-FIRED"
            else f"ROUND113 LIVE INPUTS {report['status']}")
        print(
            f"{prefix}: first={matrix['first_nonbit_family']} "
            f"base={row['base']['cells_unequal']}/"
            f"{row['base']['max_abs']:.12e} "
            f"u={row['transport_u']['cells_unequal']}/"
            f"{row['transport_u']['max_abs']:.12e} "
            f"adv_arm={arm['cells_unequal']}/{arm['max_abs']:.12e}")
    elif args.round112_fct_walk:
        matrix = report["round112_fct_walk"]
        row = matrix["modes"]["production_step_jit"]["current"]["T"]
        prefix = (
            "STATUS PLANT-FIRED" if report["status"] == "PLANT-FIRED"
            else f"ROUND112 FCT WALK {report['status']}")
        print(
            f"{prefix}: first={matrix['first_nonbit_boundary']} "
            f"u={row['first_u']['cells_unequal']}/"
            f"{row['first_u']['max_abs']:.12e} "
            f"rhs={row['rhs_after']['cells_unequal']}/"
            f"{row['rhs_after']['max_abs']:.12e}")
    elif args.round111_fct_split:
        row = report["round111_fct_split"]["rows"]["T"]
        prefix = (
            "STATUS PLANT-FIRED" if report["status"] == "PLANT-FIRED"
            else f"ROUND111 FCT SPLIT {report['status']}")
        print(
            f"{prefix}: "
            f"upstream={row['upstream_rhs_vs_adv_up1']['max_abs']:.12e} "
            f"final={row['split_rhs_vs_after_adv']['max_abs']:.12e}")
    elif args.round71_kmm:
        row = report["round71_kmm_matrix"]["rows"]["T"]
        ldf_max = report["round70_pair_matrix"]["rows"]["T"][
            "ldf_only"]["kt3_vs_oracle"]["max_abs"]
        print(
            f"ROUND71 FCT KMM {report['status']}: "
            f"ldf_T={ldf_max:.12e} "
            f"kmm_T={row['kmm_TS']['kt3_vs_oracle']['max_abs']:.12e}")
    elif args.round70_pair:
        row = report["round70_pair_matrix"]["rows"]["T"]
        print(
            f"ROUND70 FCT LDF PAIR {report['status']}: "
            f"ldf_T={row['ldf_only']['content_vs_oracle']['max_abs']:.12e} "
            f"pair_T={row['paired']['content_vs_oracle']['max_abs']:.12e}")
    elif args.round69_native:
        row = report["native_source_arm"]["rows"]["T"]
        print(
            f"ROUND69 NATIVE SOURCE {report['status']}: "
            f"content_T={row['content_vs_oracle']['max_abs']:.12e} "
            f"kt3_T={row['kt3_vs_oracle']['max_abs']:.12e}")
    else:
        row = report["production_fct_prediction"]["content_rows"]["T"]
        kt3 = report["production_fct_prediction"]["kt3"]["T"][
            "production_content_override"]
        print(
            f"ROUND68 PRODUCTION FCT {report['status']}: "
            f"content_T={row['content_vs_oracle']['max_abs']:.12e} "
            f"kt3_T={kt3['max_abs']:.12e}")
    if args.round114_u_operands or args.round115_u_geometry:
        return 1 if report["status"] == "PLANT-FIRED" else 0
    return 0 if report["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
