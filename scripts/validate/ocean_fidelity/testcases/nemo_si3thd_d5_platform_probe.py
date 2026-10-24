#!/usr/bin/env python3
"""SI3 ladder rung D5 platform probe, reviewer reproduction, and 4-step C1D year rescore.

Evaluates on the active JAX backend (CPU or GPU, under ``JAX_ENABLE_X64=1`` and
``PrecisionPolicy.fp64()``):

1. Stage A -- Compiled smallest-subnormal (``0x0000000000000001`` = ``5e-324``)
   identity and arithmetic bit-pattern probes (``x``, ``x * 1.0``, ``x + 0.0``,
   and ``(2^-1022) * (2^-52)``).
2. Stage B -- Compiled ``h_s > 0.0`` predicate and ``isnow`` selector on the
   recorded C1D ``h_s_1d`` operand across all subnormal-snow steps
   (``kt = 5842..5867``).
3. Stage C -- Exact compiled ZDF (``_si3_zdf_bl99_step`` and full-step
   ``si3_column_step_arrays``) outputs on the recorded operands at ``kt = 5841``,
   ``kt = 5842``, ``kt = 5860`` (the Phase 6 ``0.02510099530281747`` row), and
   ``kt = 5867``, plus all 26 subnormal-snow steps before and after the
   selectable ``_nemo_zdf_layer_reciprocals`` statement fix.
4. Step 1 Reviewer Reproduction -- Committed reproduction of the Round 2
   operator-review numbers:
   - Recorded ZDF-entry snow temperature reproduced in ``1,811`` of ``7,385``
     snow steps as shipped in Round 2, ``2,469`` of ``7,385`` after Step 2
     harness ``e_s / v_s * 3``, and ``7,385`` of ``7,385`` (`0` ULP) after Step 3
     ``icevar.F90:412-413``.
   - With NEMO-exact ZDF inputs across all ``8,734`` normal steps (``26,202``
     ``POST_ZDF`` ``t_su``/``e_i``/``e_s`` rows), all ``26,202`` rows are at
     ``0`` ULP (``1,625`` over-bar -> ``0``) on GPU and CPU-without-fusion, and
     also on CPU production JIT once Step 4 ``nemo_source_round`` is active.
5. Stage D -- Full 8,760-step C1D year exact-entry operator sweep rescore
   (``621,960`` stage-field rows) across the ordered Round 3 steps:
   - Round 2 baseline (``_nemo_entry_conversion=False, _nemo_glo2eqv_snow_temp=False, _nemo_source_round=False``)
   - Step 2 harness entry conversion (``icevar.F90:368,374,390,412,488-506``)
   - Step 3 model solver-entry snow temperature (``icevar.F90:412-413``)
   - Step 4 CPU fusion prevention via ``nemo_source_round`` (DECISION 158)
   plus isolated single-step ablations and the remaining owner ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice.bitz_lipscomb import (
    SI3SurfaceForcing,
    _si3_zdf_bl99_step,
    ice_temperature_from_enthalpy,
    si3_column_step_arrays,
    snow_temperature_from_enthalpy,
)
from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_card
from legoesm.ocean.fidelity.provenance import worktree_stamp

TESTCASES_DIR = Path(__file__).resolve().parent
if str(TESTCASES_DIR) not in sys.path:
    sys.path.insert(0, str(TESTCASES_DIR))

import nemo_si3thd_phase2_gate as phase2  # noqa: E402
import nemo_si3thd_phase2b_year_gate as year_gate  # noqa: E402

DEFAULT_ORACLE_DIR = year_gate.REPLAY_ROOT
SMALLEST_SUBNORMAL_BITS = np.uint64(1)
SMALLEST_SUBNORMAL_F64 = SMALLEST_SUBNORMAL_BITS.view(np.float64)
POST_ZDF_SOLVED_FIELDS = ("t_su", "e_i", "e_s")


def _u64_hex(val: float | np.ndarray) -> str | list[str]:
    arr = np.asarray(val, dtype=np.float64)
    u64 = arr.view(np.uint64)
    if u64.ndim == 0:
        return f"0x{int(u64):016x}"
    return [f"0x{int(x):016x}" for x in u64.ravel()]


def _ulp_diff(a: np.ndarray, b: np.ndarray) -> int:
    ua = np.asarray(a, dtype=np.float64).view(np.int64)
    ub = np.asarray(b, dtype=np.float64).view(np.int64)

    def _to_lex(u: np.ndarray) -> np.ndarray:
        return np.where(u < 0, np.int64(-9223372036854775808) - u, u)

    la = _to_lex(ua)
    lb = _to_lex(ub)
    return int(np.max(np.abs(la - lb)))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@jax.jit
def _jit_identity(x: jax.Array) -> jax.Array:
    return x


@jax.jit
def _jit_mul_one(x: jax.Array, one: jax.Array) -> jax.Array:
    return x * one


@jax.jit
def _jit_add_zero(x: jax.Array, zero: jax.Array) -> jax.Array:
    return x + zero


@jax.jit
def _jit_produce_subnormal(a: jax.Array, b: jax.Array) -> jax.Array:
    return a * b


@jax.jit
def _jit_snow_pred(h_s: jax.Array) -> jax.Array:
    return h_s > jnp.asarray(0.0, dtype=jnp.float64)


@jax.jit
def _jit_isnow(h_s: jax.Array) -> jax.Array:
    return jnp.where(
        h_s > jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
    )


def run_stage_a_subnormal_probe(*, plant_zero_input: bool = False) -> dict:
    """Stage A: compiled smallest-subnormal identity and arithmetic probe."""
    x_np = (
        np.float64(0.0)
        if plant_zero_input
        else np.array(SMALLEST_SUBNORMAL_F64, dtype=np.float64)
    )
    a_np = np.array(np.ldexp(1.0, -1022), dtype=np.float64)
    b_np = np.array(np.ldexp(1.0, -52), dtype=np.float64)
    if plant_zero_input:
        b_np = np.array(0.0, dtype=np.float64)

    x_jax = jnp.asarray(x_np, dtype=jnp.float64)
    one_jax = jnp.asarray(np.float64(1.0), dtype=jnp.float64)
    zero_jax = jnp.asarray(np.float64(0.0), dtype=jnp.float64)
    a_jax = jnp.asarray(a_np, dtype=jnp.float64)
    b_jax = jnp.asarray(b_np, dtype=jnp.float64)

    out_id = np.asarray(_jit_identity(x_jax), dtype=np.float64)
    out_mul = np.asarray(_jit_mul_one(x_jax, one_jax), dtype=np.float64)
    out_add = np.asarray(_jit_add_zero(x_jax, zero_jax), dtype=np.float64)
    out_prod = np.asarray(_jit_produce_subnormal(a_jax, b_jax), dtype=np.float64)

    id_bits = _u64_hex(out_id)
    mul_bits = _u64_hex(out_mul)
    add_bits = _u64_hex(out_add)
    prod_bits = _u64_hex(out_prod)

    expected_bits = "0x0000000000000001"
    arithmetic_preserves = (
        mul_bits == expected_bits
        and add_bits == expected_bits
        and prod_bits == expected_bits
    )
    # Note: `x * 1.0` and `x + 0.0` have a subnormal result (`5e-324`), so they
    # flush to `0.0` under output flush-to-zero (FTZ) alone even when input
    # denormals-are-zero (DAZ) is off.  Only the Stage B `h_s > 0.0` predicate
    # (whose boolean/1.0 output is normal) isolates input flushing (DAZ), while
    # `(2^-1022) * (2^-52)` from normal inputs isolates output flushing (FTZ).
    ftz_or_daz_flushes_mul_add = (
        mul_bits == "0x0000000000000000" and add_bits == "0x0000000000000000"
    )
    ftz_flushes_output = (
        not plant_zero_input and prod_bits == "0x0000000000000000"
    )

    return {
        "input_x_float": float(x_np),
        "input_x_bits": _u64_hex(x_np),
        "input_a_bits": _u64_hex(a_np),
        "input_b_bits": _u64_hex(b_np),
        "jit_identity_out_float": float(out_id),
        "jit_identity_out_bits": id_bits,
        "jit_mul_one_out_float": float(out_mul),
        "jit_mul_one_out_bits": mul_bits,
        "jit_add_zero_out_float": float(out_add),
        "jit_add_zero_out_bits": add_bits,
        "jit_produce_subnormal_out_float": float(out_prod),
        "jit_produce_subnormal_out_bits": prod_bits,
        "identity_preserves_bits": id_bits == expected_bits,
        "arithmetic_preserves_subnormal": arithmetic_preserves,
        "ftz_or_daz_flushes_mul_add": ftz_or_daz_flushes_mul_add,
        "ftz_flushes_subnormal_output": ftz_flushes_output,
    }


def classify_ftz_daz(
    stage_a: dict,
    stage_b: dict,
    *,
    plant_invalid_daz_from_mul_add: bool = False,
) -> dict:
    """Classify FTZ (output flushing) and DAZ (input flushing) from Stages A & B.

    - ``ftz_flushes_subnormal_output`` comes from Stage A ``(2^-1022) * (2^-52)``
      (normal inputs producing a subnormal output).
    - ``daz_flushes_subnormal_input`` comes ONLY from Stage B's ``h_s > 0.0``
      predicate on positive subnormal inputs (boolean/normal output, which FTZ
      alone cannot flush).  ``x * 1.0`` and ``x + 0.0`` have subnormal outputs
      and flush under FTZ alone, so they cannot classify DAZ.
    """
    ftz_flushes_output = bool(stage_a["ftz_flushes_subnormal_output"])
    if plant_invalid_daz_from_mul_add:
        daz_flushes_input = bool(
            stage_a["jit_mul_one_out_bits"] == "0x0000000000000000"
            and stage_a["jit_add_zero_out_bits"] == "0x0000000000000000"
        )
    else:
        daz_flushes_input = bool(
            stage_b.get("inputs_are_positive_subnormal", False)
            and stage_b["predicate_all_false"]
        )

    if stage_a["arithmetic_preserves_subnormal"] and stage_b["predicate_all_true"]:
        ftz_daz_mode = "IEEE754_GRADUAL_UNDERFLOW"
    elif ftz_flushes_output and daz_flushes_input:
        ftz_daz_mode = "FTZ_DAZ_FLUSH_TO_ZERO"
    elif ftz_flushes_output and not daz_flushes_input:
        ftz_daz_mode = "FTZ_ONLY_FLUSH_TO_ZERO"
    elif daz_flushes_input and not ftz_flushes_output:
        ftz_daz_mode = "DAZ_ONLY_FLUSH_TO_ZERO"
    else:
        ftz_daz_mode = "PARTIAL_OR_PLANTED_SUBNORMAL_FLUSH"

    return {
        "daz_flushes_subnormal_input": daz_flushes_input,
        "ftz_flushes_subnormal_output": ftz_flushes_output,
        "ftz_daz_mode": ftz_daz_mode,
    }


def _evaluate_subnormal_window_arm(
    subnormal_records: list[tuple[int, list[dict], dict, object]],
    card,
    *,
    zdf_layer_reciprocals: bool,
    flush_recorded_operand: bool = False,
) -> dict:
    """Evaluate one statement arm across the subnormal-snow window."""
    tiny_normal = np.finfo(np.float64).tiny
    over_bar_zdf_rows = []
    nonzero_ulp_zdf_rows = []
    over_bar_all_stage_rows = []
    detailed_steps = {}

    if not subnormal_records:
        return {
            "zdf_layer_reciprocals": zdf_layer_reciprocals,
            "over_bar_post_zdf_count": 0,
            "over_bar_post_zdf_rows": [],
            "nonzero_ulp_post_zdf_count": 0,
            "nonzero_ulp_post_zdf_rows": [],
            "over_bar_all_stage_count_in_subnormal_window": 0,
            "over_bar_all_stage_rows_in_subnormal_window": [],
            "detailed_post_zdf_steps": {},
        }

    states_in = [
        (
            entry._replace(h_snow=np.zeros_like(entry.h_snow))
            if flush_recorded_operand
            else entry
        )
        for _, _, _, entry in subnormal_records
    ]
    batch_entry = jax.tree.map(
        lambda *vals: jnp.asarray(np.concatenate(vals, axis=0), dtype=jnp.float64),
        *states_in,
    )
    batch_forcing = SI3SurfaceForcing(
        *(
            jnp.asarray(
                np.concatenate(
                    [
                        np.asarray(exact_input[name], dtype=np.float64)
                        for _, _, exact_input, _ in subnormal_records
                    ],
                    axis=0,
                ),
                dtype=jnp.float64,
            )
            for name in phase2.ZDF_INPUT_NAMES
        )
    )
    traces = jax.device_get(
        si3_column_step_arrays(
            batch_entry,
            batch_forcing,
            card.dt_seconds,
            card.config.ice_constants,
            _nemo_zdf_layer_reciprocals=zdf_layer_reciprocals,
        )
    )
    stage_maps = year_gate._stage_maps(traces)

    for offset, (kt, frames, _exact_input, entry) in enumerate(subnormal_records):
        hs_val = float(np.asarray(entry.h_snow, dtype=np.float64).ravel()[0])
        is_subnormal_zdf_entry = 0.0 < hs_val < tiny_normal
        in_subnormal_window = 5842 <= kt <= 5867

        for stage_idx, (oracle_frame, cand_map) in enumerate(
            zip(frames, stage_maps, strict=True)
        ):
            stage_name = phase2.STAGES[stage_idx][1]
            oracle = (
                oracle_frame
                if 1 <= stage_idx <= 5
                else phase2._center_global(oracle_frame)
            )
            names = (
                phase2.SELECTED_NAMES
                if 1 <= stage_idx <= 5
                else phase2.GLOBAL_NAMES
            )
            for fname in names:
                wanted = np.asarray(oracle[fname], dtype=np.float64).ravel()
                got = np.asarray(
                    cand_map[fname][offset : offset + 1], dtype=np.float64
                ).ravel()
                abs_err, norm_err = year_gate._normalised_error(wanted, got)
                scale = max(1.0, float(np.max(np.abs(wanted))))
                max_ulp = _ulp_diff(got, wanted)
                row_info = {
                    "kt": kt,
                    "stage": stage_name,
                    "field": fname,
                    "norm_err": norm_err,
                    "max_abs": abs_err,
                    "scale": scale,
                    "max_ulp": max_ulp,
                }
                if in_subnormal_window and norm_err > phase2.BAR:
                    over_bar_all_stage_rows.append(row_info)
                if stage_idx == 1 and is_subnormal_zdf_entry:
                    if norm_err > phase2.BAR:
                        over_bar_zdf_rows.append(row_info)
                    if max_ulp > 0:
                        nonzero_ulp_zdf_rows.append(row_info)

        if kt in (5841, 5842, 5860, 5867):
            post_zdf_oracle = frames[1]
            post_zdf_cand = stage_maps[1]
            step_detail = {}
            for fname in phase2.SELECTED_NAMES:
                act_arr = np.asarray(
                    post_zdf_cand[fname][offset : offset + 1], dtype=np.float64
                ).ravel()
                ref_arr = np.asarray(post_zdf_oracle[fname], dtype=np.float64).ravel()
                abs_err, norm_err = year_gate._normalised_error(ref_arr, act_arr)
                step_detail[fname] = {
                    "actual_float": [float(x) for x in act_arr],
                    "oracle_float": [float(x) for x in ref_arr],
                    "actual_bits": _u64_hex(act_arr),
                    "oracle_bits": _u64_hex(ref_arr),
                    "norm_err": norm_err,
                    "max_abs": abs_err,
                    "scale": max(1.0, float(np.max(np.abs(ref_arr)))),
                    "max_ulp": _ulp_diff(act_arr, ref_arr),
                }
            detailed_steps[f"kt{kt:04d}"] = step_detail

    over_bar_zdf_rows.sort(key=lambda r: r["norm_err"], reverse=True)
    nonzero_ulp_zdf_rows.sort(key=lambda r: (r["max_ulp"], r["norm_err"]), reverse=True)
    over_bar_all_stage_rows.sort(key=lambda r: r["norm_err"], reverse=True)

    return {
        "zdf_layer_reciprocals": zdf_layer_reciprocals,
        "over_bar_post_zdf_count": len(over_bar_zdf_rows),
        "over_bar_post_zdf_rows": over_bar_zdf_rows,
        "nonzero_ulp_post_zdf_count": len(nonzero_ulp_zdf_rows),
        "nonzero_ulp_post_zdf_rows": nonzero_ulp_zdf_rows,
        "over_bar_all_stage_count_in_subnormal_window": len(
            over_bar_all_stage_rows
        ),
        "over_bar_all_stage_rows_in_subnormal_window": over_bar_all_stage_rows,
        "detailed_post_zdf_steps": detailed_steps,
    }


def run_step1_reviewer_reproduction(
    oracle_dir: Path = DEFAULT_ORACLE_DIR,
    *,
    max_steps: int | None = None,
    plant_corrupt_snow_temp: bool = False,
) -> dict:
    """Step 1: Committed reproduction of the Round 2 operator-review findings.

    Measures across the C1D year (8,760 steps; 7,385 snow-present steps with
    ``v_s > 1e-20``; 8,734 normal steps outside ``kt = 5842..5867`` = 26,202
    ``POST_ZDF`` ``t_su``/``e_i``/``e_s`` rows):
    1. Recorded ZDF-entry snow temperature (``exact["t_s"]`` in
       ``oracle_si3_zdf_inputs.bin``) vs:
       - Round 2 as-shipped: ``(e_s * 3.0) / v_s`` + ``icethd_dh.F90:498-503``
         (``1,811`` of ``7,385`` snow steps at 0 ULP).
       - Step 2 harness only: ``(e_s / v_s) * 3.0`` + ``icethd_dh.F90:498-503``
         (``2,469`` of ``7,385`` snow steps at 0 ULP).
       - Step 2 + Step 3 model statement: ``(e_s / v_s) * 3.0`` +
         ``icevar.F90:412-413`` (``7,385`` of ``7,385`` snow steps at 0 ULP).
    2. Recorded ZDF-entry ice temperature (``exact["t_i"]``) and layer salinity
       (``frames[1]["sz_i"]``) vs Step 2 harness + ``ice_temperature_from_enthalpy``
       (``8,760`` of ``8,760`` steps at 0 ULP).
    3. ``POST_ZDF`` (``t_su``, ``e_i``, ``e_s`` across the ``8,734`` normal
       steps = ``26,202`` rows) under:
       - Round 2 as-shipped (``_nemo_entry_conversion=False, _nemo_glo2eqv_snow_temp=False, _nemo_source_round=False``)
       - NEMO-exact ZDF inputs without source rounding (``_nemo_entry_conversion=True, _nemo_glo2eqv_snow_temp=True, _nemo_source_round=False``)
       - NEMO-exact ZDF inputs with Step 4 ``nemo_source_round`` (``_nemo_entry_conversion=True, _nemo_glo2eqv_snow_temp=True, _nemo_source_round=True``)
    """
    card = build_c1d_omip_l3_card(oracle_root=oracle_dir)
    c = card.config.ice_constants
    nsteps = card.nsteps if max_steps is None else min(int(max_steps), card.nsteps)
    thd_path = oracle_dir / "oracle_si3_thd_frames.bin"
    zin_path = oracle_dir / "oracle_si3_zdf_inputs.bin"

    entries_r2 = []
    entries_s2 = []
    forcings = []
    oracle_post_zdf = []
    exact_t_s_list = []
    snow_active_flags = []
    normal_step_flags = []

    with thd_path.open("rb") as thd, zin_path.open("rb") as zin:
        for kt in range(1, nsteps + 1):
            frames = phase2._read_thd_step(thd, kt)
            exact = phase2._read_zdf_input_step(zin, kt)
            snow_active_flags.append(
                float(np.asarray(frames[1]["h_s"]).ravel()[0]) > 0.0
            )
            normal_step_flags.append(not (5842 <= kt <= 5867))
            entries_r2.append(
                year_gate._exact_1d_entry(frames, _nemo_entry_conversion=False)
            )
            entries_s2.append(
                year_gate._exact_1d_entry(frames, _nemo_entry_conversion=True)
            )
            forcings.append(
                tuple(
                    np.asarray(exact[name], dtype=np.float64)
                    for name in phase2.ZDF_INPUT_NAMES
                )
            )
            oracle_post_zdf.append(frames[1])
            exact_t_s_list.append(
                np.asarray(exact["t_s"], dtype=np.float64).reshape(1, 3)
            )

    batch_r2 = jax.tree.map(
        lambda *vals: jnp.asarray(np.concatenate(vals, axis=0), dtype=jnp.float64),
        *entries_r2,
    )
    batch_s2 = jax.tree.map(
        lambda *vals: jnp.asarray(np.concatenate(vals, axis=0), dtype=jnp.float64),
        *entries_s2,
    )

    ts_r2 = np.asarray(
        jax.device_get(
            snow_temperature_from_enthalpy(
                batch_r2.e_snow,
                c,
                _nemo_order=True,
                _nemo_glo2eqv_snow_temp=False,
                _nemo_source_round=False,
            )
        ),
        dtype=np.float64,
    )
    ts_s2_only = np.asarray(
        jax.device_get(
            snow_temperature_from_enthalpy(
                batch_s2.e_snow,
                c,
                _nemo_order=True,
                _nemo_glo2eqv_snow_temp=False,
                _nemo_source_round=False,
            )
        ),
        dtype=np.float64,
    )
    ts_s2_s3 = np.asarray(
        jax.device_get(
            snow_temperature_from_enthalpy(
                batch_s2.e_snow,
                c,
                _nemo_order=True,
                _nemo_glo2eqv_snow_temp=(not plant_corrupt_snow_temp),
                _nemo_source_round=True,
            )
        ),
        dtype=np.float64,
    )

    snow_step_count = 0
    ts_r2_exact_steps = 0
    ts_r2_max_ulp = 0
    ts_s2_exact_steps = 0
    ts_s2_max_ulp = 0
    ts_s2_s3_exact_steps = 0
    ts_s2_s3_max_ulp = 0

    szi_r2_exact_steps = 0
    szi_s2_exact_steps = 0

    for idx in range(nsteps):
        ref_ts = exact_t_s_list[idx].ravel()
        ref_szi = np.asarray(oracle_post_zdf[idx]["sz_i"], dtype=np.float64).ravel()

        if _ulp_diff(np.asarray(entries_r2[idx].S_layers).ravel(), ref_szi) == 0:
            szi_r2_exact_steps += 1
        if _ulp_diff(np.asarray(entries_s2[idx].S_layers).ravel(), ref_szi) == 0:
            szi_s2_exact_steps += 1

        if snow_active_flags[idx]:
            snow_step_count += 1
            u_r2 = _ulp_diff(ts_r2[idx].ravel(), ref_ts)
            u_s2 = _ulp_diff(ts_s2_only[idx].ravel(), ref_ts)
            u_s2_s3 = _ulp_diff(ts_s2_s3[idx].ravel(), ref_ts)
            ts_r2_max_ulp = max(ts_r2_max_ulp, u_r2)
            ts_s2_max_ulp = max(ts_s2_max_ulp, u_s2)
            ts_s2_s3_max_ulp = max(ts_s2_s3_max_ulp, u_s2_s3)
            if u_r2 == 0:
                ts_r2_exact_steps += 1
            if u_s2 == 0:
                ts_s2_exact_steps += 1
            if u_s2_s3 == 0:
                ts_s2_s3_exact_steps += 1

    def _eval_post_zdf_arm(
        entries_list,
        *,
        zdf_layer_reciprocals: bool = True,
        glo2eqv_snow_temp: bool,
        source_round: bool,
    ) -> dict:
        normal_steps = 0
        total_rows = 0
        zero_ulp_rows = 0
        nonzero_ulp_rows = 0
        over_bar_rows = 0
        max_ulp = 0
        max_norm_err = 0.0
        by_field = {
            f: {"total": 0, "zero_ulp": 0, "nonzero_ulp": 0, "over_bar": 0, "max_ulp": 0}
            for f in POST_ZDF_SOLVED_FIELDS
        }

        chunk = 1000
        for start in range(0, nsteps, chunk):
            stop = min(nsteps, start + chunk)
            b_entry = jax.tree.map(
                lambda *vals: jnp.asarray(
                    np.concatenate(vals, axis=0), dtype=jnp.float64
                ),
                *entries_list[start:stop],
            )
            b_forcing = SI3SurfaceForcing(
                *(
                    jnp.asarray(
                        np.concatenate(
                            [f[i] for f in forcings[start:stop]], axis=0
                        ),
                        dtype=jnp.float64,
                    )
                    for i in range(len(phase2.ZDF_INPUT_NAMES))
                )
            )
            zdf_out = jax.device_get(
                _si3_zdf_bl99_step(
                    b_entry.e_ice,
                    b_entry.e_snow,
                    b_entry.S_layers,
                    b_entry.h_ice,
                    b_entry.h_snow,
                    b_entry.T_surface,
                    b_forcing,
                    card.dt_seconds,
                    c,
                    _nemo_zdf_layer_reciprocals=zdf_layer_reciprocals,
                    _nemo_glo2eqv_snow_temp=glo2eqv_snow_temp,
                    _nemo_source_round=source_round,
                )
            )
            cand_fields = {
                "t_su": np.asarray(zdf_out.T_surface, dtype=np.float64),
                "e_i": np.asarray(zdf_out.e_ice, dtype=np.float64),
                "e_s": np.asarray(zdf_out.e_snow, dtype=np.float64),
            }
            for offset, idx in enumerate(range(start, stop)):
                if not normal_step_flags[idx]:
                    continue
                normal_steps += 1
                oracle_f1 = oracle_post_zdf[idx]
                for fname in POST_ZDF_SOLVED_FIELDS:
                    total_rows += 1
                    by_field[fname]["total"] += 1
                    wanted = np.asarray(oracle_f1[fname], dtype=np.float64).ravel()
                    got = cand_fields[fname][offset : offset + 1].ravel()
                    _, norm_err = year_gate._normalised_error(wanted, got)
                    u = _ulp_diff(got, wanted)
                    max_ulp = max(max_ulp, u)
                    max_norm_err = max(max_norm_err, norm_err)
                    by_field[fname]["max_ulp"] = max(by_field[fname]["max_ulp"], u)
                    if u == 0:
                        zero_ulp_rows += 1
                        by_field[fname]["zero_ulp"] += 1
                    else:
                        nonzero_ulp_rows += 1
                        by_field[fname]["nonzero_ulp"] += 1
                    if norm_err > phase2.BAR:
                        over_bar_rows += 1
                        by_field[fname]["over_bar"] += 1

        return {
            "normal_steps_evaluated": normal_steps,
            "total_post_zdf_rows": total_rows,
            "zero_ulp_rows": zero_ulp_rows,
            "nonzero_ulp_rows": nonzero_ulp_rows,
            "over_bar_rows": over_bar_rows,
            "max_ulp": max_ulp,
            "max_norm_err": max_norm_err,
            "by_field": by_field,
        }

    post_zdf_r2 = _eval_post_zdf_arm(
        entries_r2,
        zdf_layer_reciprocals=True,
        glo2eqv_snow_temp=False,
        source_round=False,
    )
    post_zdf_exact_inputs_pre_r2_reciprocals = _eval_post_zdf_arm(
        entries_s2,
        zdf_layer_reciprocals=False,
        glo2eqv_snow_temp=(not plant_corrupt_snow_temp),
        source_round=False,
    )
    post_zdf_exact_inputs_unrounded = _eval_post_zdf_arm(
        entries_s2,
        zdf_layer_reciprocals=True,
        glo2eqv_snow_temp=(not plant_corrupt_snow_temp),
        source_round=False,
    )
    post_zdf_exact_inputs_source_rounded = _eval_post_zdf_arm(
        entries_s2,
        zdf_layer_reciprocals=True,
        glo2eqv_snow_temp=(not plant_corrupt_snow_temp),
        source_round=True,
    )

    return {
        "steps_evaluated": nsteps,
        "snow_present_step_count": snow_step_count,
        "zdf_entry_snow_temperature": {
            "r2_as_shipped_exact_steps": ts_r2_exact_steps,
            "r2_as_shipped_nonzero_ulp_steps": snow_step_count - ts_r2_exact_steps,
            "r2_as_shipped_max_ulp": ts_r2_max_ulp,
            "step2_harness_only_exact_steps": ts_s2_exact_steps,
            "step2_harness_only_nonzero_ulp_steps": snow_step_count - ts_s2_exact_steps,
            "step2_harness_only_max_ulp": ts_s2_max_ulp,
            "step2_plus_step3_exact_steps": ts_s2_s3_exact_steps,
            "step2_plus_step3_nonzero_ulp_steps": snow_step_count - ts_s2_s3_exact_steps,
            "step2_plus_step3_max_ulp": ts_s2_s3_max_ulp,
        },
        "zdf_entry_salinity_profile": {
            "szi_r2_exact_steps": szi_r2_exact_steps,
            "szi_r2_nonzero_ulp_steps": nsteps - szi_r2_exact_steps,
            "szi_step2_exact_steps": szi_s2_exact_steps,
            "szi_step2_nonzero_ulp_steps": nsteps - szi_s2_exact_steps,
        },
        "normal_steps_post_zdf_26202_rows": {
            "r2_as_shipped": post_zdf_r2,
            "nemo_exact_inputs_pre_r2_reciprocals": post_zdf_exact_inputs_pre_r2_reciprocals,
            "nemo_exact_inputs_unrounded": post_zdf_exact_inputs_unrounded,
            "nemo_exact_inputs_source_rounded": post_zdf_exact_inputs_source_rounded,
        },
    }


def _init_arm_accumulator() -> dict:
    return {
        "total_field_rows": 0,
        "over_bar_rows": 0,
        "zero_ulp_rows": 0,
        "nonzero_ulp_rows": 0,
        "over_bar_by_stage": {s[1]: 0 for s in phase2.STAGES},
        "zero_ulp_by_stage": {s[1]: 0 for s in phase2.STAGES},
        "nonzero_ulp_by_stage": {s[1]: 0 for s in phase2.STAGES},
        "normal_post_zdf_3field": {
            "total_rows": 0,
            "zero_ulp_rows": 0,
            "nonzero_ulp_rows": 0,
            "over_bar_rows": 0,
        },
        "normal_post_zdf_7field": {
            "total_rows": 0,
            "zero_ulp_rows": 0,
            "nonzero_ulp_rows": 0,
            "over_bar_rows": 0,
        },
        "subnormal_window_kt5842_5867": {
            "total_rows": 0,
            "over_bar_rows": 0,
            "zero_ulp_rows": 0,
            "nonzero_ulp_rows": 0,
            "post_zdf_3field_over_bar_rows": 0,
            "post_zdf_3field_zero_ulp_rows": 0,
        },
        "max_normalised_error_row": None,
    }


def _update_arm_accumulator(
    acc: dict,
    *,
    kt: int,
    stage_idx: int,
    stage_name: str,
    fname: str,
    norm_err: float,
    abs_err: float,
    scale: float,
    ulp: int,
    in_subnormal_window: bool,
) -> None:
    acc["total_field_rows"] += 1
    is_over = norm_err > phase2.BAR
    if is_over:
        acc["over_bar_rows"] += 1
        acc["over_bar_by_stage"][stage_name] += 1
    if ulp == 0:
        acc["zero_ulp_rows"] += 1
        acc["zero_ulp_by_stage"][stage_name] += 1
    else:
        acc["nonzero_ulp_rows"] += 1
        acc["nonzero_ulp_by_stage"][stage_name] += 1

    if in_subnormal_window:
        sw = acc["subnormal_window_kt5842_5867"]
        sw["total_rows"] += 1
        if is_over:
            sw["over_bar_rows"] += 1
        if ulp == 0:
            sw["zero_ulp_rows"] += 1
        else:
            sw["nonzero_ulp_rows"] += 1
        if stage_idx == 1 and fname in POST_ZDF_SOLVED_FIELDS:
            if is_over:
                sw["post_zdf_3field_over_bar_rows"] += 1
            if ulp == 0:
                sw["post_zdf_3field_zero_ulp_rows"] += 1
    elif stage_idx == 1:
        p7 = acc["normal_post_zdf_7field"]
        p7["total_rows"] += 1
        if is_over:
            p7["over_bar_rows"] += 1
        if ulp == 0:
            p7["zero_ulp_rows"] += 1
        else:
            p7["nonzero_ulp_rows"] += 1
        if fname in POST_ZDF_SOLVED_FIELDS:
            p3 = acc["normal_post_zdf_3field"]
            p3["total_rows"] += 1
            if is_over:
                p3["over_bar_rows"] += 1
            if ulp == 0:
                p3["zero_ulp_rows"] += 1
            else:
                p3["nonzero_ulp_rows"] += 1

    cur_max = acc["max_normalised_error_row"]
    if cur_max is None or norm_err > cur_max["norm_err"]:
        acc["max_normalised_error_row"] = {
            "kt": kt,
            "stage": stage_name,
            "field": fname,
            "norm_err": norm_err,
            "max_abs": abs_err,
            "scale": scale,
            "max_ulp": ulp,
        }


def run_year_rescore(
    oracle_dir: Path = DEFAULT_ORACLE_DIR,
    *,
    max_steps: int | None = None,
) -> dict:
    """Stage D: full 8,760-step C1D exact-entry operator sweep across Round 2 & Round 3 arms."""
    card = build_c1d_omip_l3_card(oracle_root=oracle_dir)
    nsteps = card.nsteps if max_steps is None else min(int(max_steps), card.nsteps)
    thd_path = oracle_dir / "oracle_si3_thd_frames.bin"
    zin_path = oracle_dir / "oracle_si3_zdf_inputs.bin"

    entries_r2 = []
    entries_s2 = []
    forcings = []
    oracle_frames = []
    with thd_path.open("rb") as thd, zin_path.open("rb") as zin:
        for kt in range(1, nsteps + 1):
            frames = phase2._read_thd_step(thd, kt)
            exact = phase2._read_zdf_input_step(zin, kt)
            entries_r2.append(
                year_gate._exact_1d_entry(frames, _nemo_entry_conversion=False)
            )
            entries_s2.append(
                year_gate._exact_1d_entry(frames, _nemo_entry_conversion=True)
            )
            forcings.append(
                tuple(
                    np.asarray(exact[name], dtype=np.float64)
                    for name in phase2.ZDF_INPUT_NAMES
                )
            )
            oracle_frames.append(frames)

    arm_specs = {
        "r2_pre_reciprocals": (
            entries_r2,
            {
                "_nemo_zdf_layer_reciprocals": False,
                "_nemo_glo2eqv_snow_temp": False,
                "_nemo_source_round": False,
            },
        ),
        "r2_baseline": (
            entries_r2,
            {
                "_nemo_zdf_layer_reciprocals": True,
                "_nemo_glo2eqv_snow_temp": False,
                "_nemo_source_round": False,
            },
        ),
        "step2_harness_only": (
            entries_s2,
            {
                "_nemo_zdf_layer_reciprocals": True,
                "_nemo_glo2eqv_snow_temp": False,
                "_nemo_source_round": False,
            },
        ),
        "step3_model_only": (
            entries_r2,
            {
                "_nemo_zdf_layer_reciprocals": True,
                "_nemo_glo2eqv_snow_temp": True,
                "_nemo_source_round": False,
            },
        ),
        "step4_source_round_only": (
            entries_r2,
            {
                "_nemo_zdf_layer_reciprocals": True,
                "_nemo_glo2eqv_snow_temp": False,
                "_nemo_source_round": True,
            },
        ),
        "step2_plus_step3": (
            entries_s2,
            {
                "_nemo_zdf_layer_reciprocals": True,
                "_nemo_glo2eqv_snow_temp": True,
                "_nemo_source_round": False,
            },
        ),
        "step2_plus_step3_plus_step4": (
            entries_s2,
            {
                "_nemo_zdf_layer_reciprocals": True,
                "_nemo_glo2eqv_snow_temp": True,
                "_nemo_source_round": True,
            },
        ),
    }

    arm_funcs = {}
    for arm_name, (_, kwargs) in arm_specs.items():
        kw = dict(kwargs)

        def _make_runner(k_dict):
            def _runner(b_entry, b_forcing):
                return si3_column_step_arrays(
                    b_entry,
                    b_forcing,
                    card.dt_seconds,
                    card.config.ice_constants,
                    **k_dict,
                )

            return _runner

        arm_funcs[arm_name] = _make_runner(kw)

    arm_accs = {name: _init_arm_accumulator() for name in arm_specs}

    # Retained top-level before (`r2_baseline`) vs after (`step2_plus_step3_plus_step4`)
    # accounting fields for compatibility and direct diff summary:
    total_field_rows = 0
    over_bar_before = 0
    over_bar_after = 0
    over_bar_before_by_stage = {s[1]: 0 for s in phase2.STAGES}
    over_bar_after_by_stage = {s[1]: 0 for s in phase2.STAGES}
    subnormal_window_over_bar_before = 0
    subnormal_window_over_bar_after = 0

    max_before_row = None
    max_after_row = None
    kt5860_post_zdf_es_before = None
    kt5860_post_zdf_es_after = None

    touched_field_row_count = 0
    untouched_field_row_count = 0
    touched_steps: set[int] = set()
    touched_by_stage = {
        s[1]: {
            "changed_rows": 0,
            "closed_to_at_bar": 0,
            "opened_to_debt": 0,
            "moved_to_zero_ulp_vs_oracle": 0,
            "moved_from_zero_ulp_vs_oracle": 0,
        }
        for s in phase2.STAGES
    }
    touched_by_stage_field: dict[str, dict[str, int]] = {}
    closed_rows = []
    opened_rows = []
    subnormal_window_touched_rows = []

    # Remaining owner ranking accumulator on final arm (`step2_plus_step3_plus_step4`)
    remaining_owners: dict[str, dict] = {}

    chunk = 1000
    for start in range(0, nsteps, chunk):
        stop = min(nsteps, start + chunk)
        batch_entry_r2 = jax.tree.map(
            lambda *vals: jnp.asarray(
                np.concatenate(vals, axis=0), dtype=jnp.float64
            ),
            *entries_r2[start:stop],
        )
        batch_entry_s2 = jax.tree.map(
            lambda *vals: jnp.asarray(
                np.concatenate(vals, axis=0), dtype=jnp.float64
            ),
            *entries_s2[start:stop],
        )
        batch_forcing = SI3SurfaceForcing(
            *(
                jnp.asarray(
                    np.concatenate([f[i] for f in forcings[start:stop]], axis=0),
                    dtype=jnp.float64,
                )
                for i in range(len(phase2.ZDF_INPUT_NAMES))
            )
        )

        arm_maps = {}
        for arm_name, (entries_ref, _) in arm_specs.items():
            b_ent = batch_entry_s2 if entries_ref is entries_s2 else batch_entry_r2
            traces = jax.device_get(arm_funcs[arm_name](b_ent, batch_forcing))
            arm_maps[arm_name] = year_gate._stage_maps(traces)

        maps_before = arm_maps["r2_baseline"]
        maps_after = arm_maps["step2_plus_step3_plus_step4"]

        for offset, kt in enumerate(range(start + 1, stop + 1)):
            frames = oracle_frames[kt - 1]
            in_subnormal_window = 5842 <= kt <= 5867

            for stage_idx in range(len(phase2.STAGES)):
                stage_name = phase2.STAGES[stage_idx][1]
                o_frame = frames[stage_idx]
                oracle = (
                    o_frame
                    if 1 <= stage_idx <= 5
                    else phase2._center_global(o_frame)
                )
                names = (
                    phase2.SELECTED_NAMES
                    if 1 <= stage_idx <= 5
                    else phase2.GLOBAL_NAMES
                )
                for fname in names:
                    wanted = np.asarray(oracle[fname], dtype=np.float64).ravel()
                    scale = max(1.0, float(np.max(np.abs(wanted))))

                    for arm_name, s_maps in arm_maps.items():
                        got_arm = np.asarray(
                            s_maps[stage_idx][fname][offset : offset + 1],
                            dtype=np.float64,
                        ).ravel()
                        abs_arm, norm_arm = year_gate._normalised_error(
                            wanted, got_arm
                        )
                        ulp_arm = _ulp_diff(got_arm, wanted)
                        _update_arm_accumulator(
                            arm_accs[arm_name],
                            kt=kt,
                            stage_idx=stage_idx,
                            stage_name=stage_name,
                            fname=fname,
                            norm_err=norm_arm,
                            abs_err=abs_arm,
                            scale=scale,
                            ulp=ulp_arm,
                            in_subnormal_window=in_subnormal_window,
                        )

                    total_field_rows += 1
                    got_bef = np.asarray(
                        maps_before[stage_idx][fname][offset : offset + 1],
                        dtype=np.float64,
                    ).ravel()
                    got_aft = np.asarray(
                        maps_after[stage_idx][fname][offset : offset + 1],
                        dtype=np.float64,
                    ).ravel()

                    abs_bef, norm_bef = year_gate._normalised_error(wanted, got_bef)
                    abs_aft, norm_aft = year_gate._normalised_error(wanted, got_aft)
                    ulp_bef = _ulp_diff(got_bef, wanted)
                    ulp_aft = _ulp_diff(got_aft, wanted)
                    ulp_change = _ulp_diff(got_aft, got_bef)

                    is_over_bef = norm_bef > phase2.BAR
                    is_over_aft = norm_aft > phase2.BAR

                    if is_over_bef:
                        over_bar_before += 1
                        over_bar_before_by_stage[stage_name] += 1
                        if in_subnormal_window:
                            subnormal_window_over_bar_before += 1
                    if is_over_aft:
                        over_bar_after += 1
                        over_bar_after_by_stage[stage_name] += 1
                        if in_subnormal_window:
                            subnormal_window_over_bar_after += 1

                    # Populate remaining owner ranking on the final arm (`got_aft`)
                    if is_over_aft or ulp_aft > 0:
                        owner_key = (
                            "D5_PLATFORM_FLOOR_KT5842_5867_SUBNORMAL_SNOW"
                            if in_subnormal_window
                            else f"{stage_name}.{fname}"
                        )
                        if owner_key not in remaining_owners:
                            remaining_owners[owner_key] = {
                                "owner": owner_key,
                                "over_bar_rows": 0,
                                "nonzero_ulp_rows": 0,
                                "max_ulp": 0,
                                "max_norm_err": 0.0,
                                "worst_row": None,
                            }
                        orec = remaining_owners[owner_key]
                        if is_over_aft:
                            orec["over_bar_rows"] += 1
                        if ulp_aft > 0:
                            orec["nonzero_ulp_rows"] += 1
                        orec["max_ulp"] = max(orec["max_ulp"], ulp_aft)
                        if (
                            orec["worst_row"] is None
                            or norm_aft > orec["max_norm_err"]
                        ):
                            orec["max_norm_err"] = norm_aft
                            orec["worst_row"] = {
                                "kt": kt,
                                "stage": stage_name,
                                "field": fname,
                                "norm_err": norm_aft,
                                "max_abs": abs_aft,
                                "scale": scale,
                                "max_ulp": ulp_aft,
                            }

                    if (
                        max_before_row is None
                        or norm_bef > max_before_row["norm_err"]
                    ):
                        max_before_row = {
                            "kt": kt,
                            "stage": stage_name,
                            "field": fname,
                            "norm_err": norm_bef,
                            "max_abs": abs_bef,
                            "scale": scale,
                            "max_ulp": ulp_bef,
                        }
                    if (
                        max_after_row is None
                        or norm_aft > max_after_row["norm_err"]
                    ):
                        max_after_row = {
                            "kt": kt,
                            "stage": stage_name,
                            "field": fname,
                            "norm_err": norm_aft,
                            "max_abs": abs_aft,
                            "scale": scale,
                            "max_ulp": ulp_aft,
                        }

                    if kt == 5860 and stage_name == "POST_ZDF" and fname == "e_s":
                        kt5860_post_zdf_es_before = {
                            "kt": kt,
                            "stage": stage_name,
                            "field": fname,
                            "actual_float": [float(x) for x in got_bef],
                            "oracle_float": [float(x) for x in wanted],
                            "actual_bits": _u64_hex(got_bef),
                            "oracle_bits": _u64_hex(wanted),
                            "norm_err": norm_bef,
                            "max_abs": abs_bef,
                            "scale": scale,
                            "max_ulp": ulp_bef,
                        }
                        kt5860_post_zdf_es_after = {
                            "kt": kt,
                            "stage": stage_name,
                            "field": fname,
                            "actual_float": [float(x) for x in got_aft],
                            "oracle_float": [float(x) for x in wanted],
                            "actual_bits": _u64_hex(got_aft),
                            "oracle_bits": _u64_hex(wanted),
                            "norm_err": norm_aft,
                            "max_abs": abs_aft,
                            "scale": scale,
                            "max_ulp": ulp_aft,
                        }

                    if ulp_change > 0:
                        touched_field_row_count += 1
                        touched_steps.add(kt)
                        st_rec = touched_by_stage[stage_name]
                        st_rec["changed_rows"] += 1
                        sf_key = f"{stage_name}.{fname}"
                        if sf_key not in touched_by_stage_field:
                            touched_by_stage_field[sf_key] = {
                                "changed_rows": 0,
                                "closed_to_at_bar": 0,
                                "opened_to_debt": 0,
                                "moved_to_zero_ulp_vs_oracle": 0,
                                "moved_from_zero_ulp_vs_oracle": 0,
                                "max_ulp_before": 0,
                                "max_ulp_after": 0,
                            }
                        sf_rec = touched_by_stage_field[sf_key]
                        sf_rec["changed_rows"] += 1
                        sf_rec["max_ulp_before"] = max(
                            sf_rec["max_ulp_before"], ulp_bef
                        )
                        sf_rec["max_ulp_after"] = max(
                            sf_rec["max_ulp_after"], ulp_aft
                        )
                        if ulp_bef > 0 and ulp_aft == 0:
                            st_rec["moved_to_zero_ulp_vs_oracle"] += 1
                            sf_rec["moved_to_zero_ulp_vs_oracle"] += 1
                        elif ulp_bef == 0 and ulp_aft > 0:
                            st_rec["moved_from_zero_ulp_vs_oracle"] += 1
                            sf_rec["moved_from_zero_ulp_vs_oracle"] += 1

                        row_summary = {
                            "kt": kt,
                            "stage": stage_name,
                            "field": fname,
                            "norm_err_before": norm_bef,
                            "norm_err_after": norm_aft,
                            "ulp_vs_oracle_before": ulp_bef,
                            "ulp_vs_oracle_after": ulp_aft,
                            "ulp_change_after_vs_before": ulp_change,
                        }
                        if is_over_bef and not is_over_aft:
                            st_rec["closed_to_at_bar"] += 1
                            sf_rec["closed_to_at_bar"] += 1
                            closed_rows.append(row_summary)
                        elif not is_over_bef and is_over_aft:
                            st_rec["opened_to_debt"] += 1
                            sf_rec["opened_to_debt"] += 1
                            opened_rows.append(row_summary)

                        if in_subnormal_window:
                            subnormal_window_touched_rows.append(row_summary)
                    else:
                        untouched_field_row_count += 1

    phase2.require(
        touched_field_row_count + untouched_field_row_count == total_field_rows,
        "year rescore row accounting mismatch",
    )

    ranked_owners = sorted(
        remaining_owners.values(),
        key=lambda r: (r["over_bar_rows"], r["nonzero_ulp_rows"], r["max_norm_err"]),
        reverse=True,
    )

    return {
        "steps_evaluated": nsteps,
        "total_field_rows": total_field_rows,
        "over_bar_before": over_bar_before,
        "over_bar_after": over_bar_after,
        "net_over_bar_delta": over_bar_after - over_bar_before,
        "over_bar_before_by_stage": over_bar_before_by_stage,
        "over_bar_after_by_stage": over_bar_after_by_stage,
        "subnormal_window_kt5842_5867": {
            "over_bar_before": subnormal_window_over_bar_before,
            "over_bar_after": subnormal_window_over_bar_after,
            "touched_row_count": len(subnormal_window_touched_rows),
            "touched_rows": subnormal_window_touched_rows,
        },
        "max_normalised_error_before": max_before_row,
        "max_normalised_error_after": max_after_row,
        "kt5860_post_zdf_es_before": kt5860_post_zdf_es_before,
        "kt5860_post_zdf_es_after": kt5860_post_zdf_es_after,
        "touched_field_row_count": touched_field_row_count,
        "untouched_field_row_count": untouched_field_row_count,
        "touched_step_count": len(touched_steps),
        "touched_by_stage": touched_by_stage,
        "touched_by_stage_field": touched_by_stage_field,
        "closed_to_at_bar_count": len(closed_rows),
        "closed_to_at_bar_rows": closed_rows,
        "opened_to_debt_count": len(opened_rows),
        "opened_to_debt_rows": opened_rows,
        "arms": arm_accs,
        "cumulative_progression_order": [
            "r2_pre_reciprocals",
            "r2_baseline",
            "step2_harness_only",
            "step2_plus_step3",
            "step2_plus_step3_plus_step4",
        ],
        "isolated_single_step_arms_from_r2_baseline": [
            "r2_baseline",
            "step2_harness_only",
            "step3_model_only",
            "step4_source_round_only",
        ],
        "remaining_owner_ranking_after_step4": ranked_owners,
    }


def run_probe(
    oracle_dir: Path = DEFAULT_ORACLE_DIR,
    *,
    plant: str | None = None,
    include_year_rescore: bool = True,
    include_step1_reproduction: bool = True,
    year_max_steps: int | None = None,
    allow_dirty: bool = False,
) -> dict:
    """Run all preregistered D5 probe stages on the active JAX backend."""
    set_policy(PrecisionPolicy.fp64())
    phase2.require(get_policy() == PrecisionPolicy.fp64(), "precision policy")

    dev = jax.devices()[0]
    platform_info = {
        "backend": jax.default_backend(),
        "device_kind": getattr(dev, "device_kind", str(dev)),
        "device_str": str(dev),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }

    stage_a = run_stage_a_subnormal_probe(
        plant_zero_input=(plant == "zero_subnormal_input")
    )

    card = build_c1d_omip_l3_card(oracle_root=oracle_dir)
    thd_path = oracle_dir / "oracle_si3_thd_frames.bin"
    zin_path = oracle_dir / "oracle_si3_zdf_inputs.bin"

    tiny_normal = np.finfo(np.float64).tiny
    subnormal_steps: list[int] = []
    stage_b_steps = {}
    pred_true_count = 0
    subnormal_records = []

    with thd_path.open("rb") as thd, zin_path.open("rb") as zin:
        for kt in range(1, card.nsteps + 1):
            frames = phase2._read_thd_step(thd, kt)
            exact_input = phase2._read_zdf_input_step(zin, kt)
            entry = year_gate._exact_1d_entry(frames)
            hs_1d = np.asarray(entry.h_snow, dtype=np.float64)
            hs_val = float(hs_1d.ravel()[0])

            is_subnormal_zdf_entry = 0.0 < hs_val < tiny_normal
            has_any_subnormal_hs = is_subnormal_zdf_entry or any(
                0.0 < float(np.asarray(f["h_s"]).ravel()[0]) < tiny_normal
                for f in frames[1:6]
            )
            if not has_any_subnormal_hs and kt != 5841:
                continue

            subnormal_records.append((kt, frames, exact_input, entry))
            if is_subnormal_zdf_entry:
                subnormal_steps.append(kt)
                hs_probe = (
                    np.zeros_like(hs_1d)
                    if plant == "flush_recorded_operand"
                    else hs_1d
                )
                pred_val = bool(
                    np.asarray(_jit_snow_pred(jnp.asarray(hs_probe))).ravel()[0]
                )
                isnow_val = np.asarray(
                    _jit_isnow(jnp.asarray(hs_probe)), dtype=np.float64
                )
                if pred_val:
                    pred_true_count += 1
                stage_b_steps[f"kt{kt:04d}"] = {
                    "kt": kt,
                    "entry_h_s_float": float(hs_probe.ravel()[0]),
                    "entry_h_s_bits": _u64_hex(hs_probe.ravel()[0]),
                    "entry_v_s_float": float(
                        phase2._center_global(frames[0])["v_s"]
                    ),
                    "entry_v_s_bits": _u64_hex(
                        phase2._center_global(frames[0])["v_s"]
                    ),
                    "jit_snow_pred": pred_val,
                    "jit_isnow_float": float(isnow_val.ravel()[0]),
                    "jit_isnow_bits": _u64_hex(isnow_val.ravel()[0]),
                }

    inputs_are_positive_subnormal = bool(
        len(subnormal_steps) > 0
        and all(
            0.0 < s["entry_h_s_float"] < tiny_normal
            for s in stage_b_steps.values()
        )
    )
    stage_b = {
        "subnormal_zdf_entry_step_count": len(subnormal_steps),
        "first_subnormal_kt": subnormal_steps[0] if subnormal_steps else None,
        "last_subnormal_kt": subnormal_steps[-1] if subnormal_steps else None,
        "inputs_are_positive_subnormal": inputs_are_positive_subnormal,
        "predicate_true_count": pred_true_count,
        "predicate_all_true": (
            pred_true_count == len(subnormal_steps) and len(subnormal_steps) > 0
        ),
        "predicate_all_false": (
            pred_true_count == 0 and len(subnormal_steps) > 0
        ),
        "daz_flushes_subnormal_input": bool(
            inputs_are_positive_subnormal
            and pred_true_count == 0
            and len(subnormal_steps) > 0
        ),
        "sampled_steps": stage_b_steps,
    }
    ftz_daz_classification = classify_ftz_daz(stage_a, stage_b)
    stage_a["daz_flushes_subnormal_input"] = ftz_daz_classification[
        "daz_flushes_subnormal_input"
    ]
    stage_a["ftz_daz_mode"] = ftz_daz_classification["ftz_daz_mode"]

    before_arm = _evaluate_subnormal_window_arm(
        subnormal_records,
        card,
        zdf_layer_reciprocals=False,
        flush_recorded_operand=(plant == "flush_recorded_operand"),
    )
    after_arm = _evaluate_subnormal_window_arm(
        subnormal_records,
        card,
        zdf_layer_reciprocals=(plant != "revert_statement_fix"),
        flush_recorded_operand=(plant == "flush_recorded_operand"),
    )

    stage_c = {
        "evaluated_subnormal_zdf_step_count": len(subnormal_steps),
        "before_statement_fix": before_arm,
        "after_statement_fix": after_arm,
        "over_bar_post_zdf_count": after_arm["over_bar_post_zdf_count"],
        "over_bar_post_zdf_rows": after_arm["over_bar_post_zdf_rows"],
        "nonzero_ulp_post_zdf_count": after_arm["nonzero_ulp_post_zdf_count"],
        "nonzero_ulp_post_zdf_rows": after_arm["nonzero_ulp_post_zdf_rows"],
        "over_bar_all_stage_count_in_subnormal_window": after_arm[
            "over_bar_all_stage_count_in_subnormal_window"
        ],
        "over_bar_all_stage_rows_in_subnormal_window": after_arm[
            "over_bar_all_stage_rows_in_subnormal_window"
        ],
        "detailed_post_zdf_steps": after_arm["detailed_post_zdf_steps"],
    }

    step1_reproduction = (
        run_step1_reviewer_reproduction(
            oracle_dir,
            max_steps=year_max_steps,
            plant_corrupt_snow_temp=(plant == "revert_snow_temp_statement"),
        )
        if include_step1_reproduction
        else None
    )

    stage_d = (
        run_year_rescore(oracle_dir, max_steps=year_max_steps)
        if include_year_rescore
        else None
    )

    if stage_a["arithmetic_preserves_subnormal"] and stage_b["predicate_all_true"]:
        platform_disposition = "SUBNORMAL_PRESERVED"
    else:
        platform_disposition = "D5_PLATFORM_FLOOR"

    return {
        "format": "nemo-si3thd-d5-platform-probe-v2",
        "worktree": worktree_stamp(allow_dirty=allow_dirty),
        "oracle_dir": str(oracle_dir),
        "oracle_sha256": {
            "oracle_si3_thd_frames.bin": _sha256(thd_path),
            "oracle_si3_zdf_inputs.bin": _sha256(zin_path),
        },
        "platform": platform_info,
        "plant": plant,
        "stage_a_subnormal_probe": stage_a,
        "stage_b_predicate_probe": stage_b,
        "stage_c_zdf_probe": stage_c,
        "step1_reviewer_reproduction": step1_reproduction,
        "stage_d_year_rescore": stage_d,
        "platform_disposition": platform_disposition,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-year-rescore",
        action="store_true",
        help="skip the 8,760-step Stage D year rescore",
    )
    parser.add_argument(
        "--skip-step1-reproduction",
        action="store_true",
        help="skip the Step 1 reviewer reproduction sweep",
    )
    parser.add_argument(
        "--expect-disposition",
        choices=("D5_PLATFORM_FLOOR", "SUBNORMAL_PRESERVED"),
        help="if set, exit 1 unless platform_disposition matches",
    )
    parser.add_argument(
        "--expect-zero-over-bar",
        action="store_true",
        help="if set, exit 1 unless stage_c over_bar_all_stage_count_in_subnormal_window == 0",
    )
    parser.add_argument(
        "--expect-zero-post-zdf-over-bar",
        action="store_true",
        help="if set, exit 1 unless stage_c over_bar_post_zdf_count == 0",
    )
    parser.add_argument(
        "--expect-bit-exact-zdf",
        action="store_true",
        help="if set, exit 1 unless stage_c nonzero_ulp_post_zdf_count == 0",
    )
    parser.add_argument(
        "--expect-step1-exact-snow-temp",
        action="store_true",
        help="if set, exit 1 unless Step 1 step2_plus_step3_nonzero_ulp_steps == 0",
    )
    parser.add_argument(
        "--plant",
        choices=(
            "zero_subnormal_input",
            "flush_recorded_operand",
            "revert_statement_fix",
            "revert_snow_temp_statement",
        ),
        help="planted violation that must cause the expected check to fail",
    )
    args = parser.parse_args(argv)

    report = run_probe(
        args.oracle_dir,
        plant=args.plant,
        include_year_rescore=not args.skip_year_rescore,
        include_step1_reproduction=not args.skip_step1_reproduction,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")

    failed = False
    if (
        args.expect_disposition
        and report["platform_disposition"] != args.expect_disposition
    ):
        failed = True
    if (
        args.expect_zero_over_bar
        and report["stage_c_zdf_probe"][
            "over_bar_all_stage_count_in_subnormal_window"
        ]
        != 0
    ):
        failed = True
    if (
        args.expect_zero_post_zdf_over_bar
        and report["stage_c_zdf_probe"]["over_bar_post_zdf_count"] != 0
    ):
        failed = True
    if (
        args.expect_bit_exact_zdf
        and report["stage_c_zdf_probe"]["nonzero_ulp_post_zdf_count"] != 0
    ):
        failed = True
    if args.expect_step1_exact_snow_temp and (
        report["step1_reviewer_reproduction"] is None
        or report["step1_reviewer_reproduction"][
            "zdf_entry_snow_temperature"
        ]["step2_plus_step3_nonzero_ulp_steps"]
        != 0
    ):
        failed = True

    if args.plant:
        return 1 if failed else 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
