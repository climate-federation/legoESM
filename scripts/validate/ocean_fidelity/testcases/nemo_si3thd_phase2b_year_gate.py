#!/usr/bin/env python3
"""Arithmetic replay and full-year SI3 column gate for C1D_OMIP_L3.

The operator sweep restarts from each NEMO ENTRY frame.  The trajectory sweep
starts once and consumes all 8,760 NEMO-written ZDF input frames.  They answer
different questions and are deliberately reported separately.
"""

from __future__ import annotations

import argparse
import json
import struct
from datetime import date, timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nemo_si3thd_phase2_gate as phase2


BAR = phase2.BAR
REPLAY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l3/"
    "c1d_omip_l3_sasice_phase2b_replay"
)
EXPANDED_ZDF_SHA256 = "aad46579fb2d2cc19299d7a25992802525603bb9858adf1e892ff5d85bf40442"
REASSOC_SHA256 = "4b832b0c274d6aab032f16958224ebfb6fea603af22e1a1f1d474ff71b1e4589"
REPORT_STEPS = (1, 10, 100, 1000, 8760)
CONTINUOUS_FIELDS = ("t_su", "e_i", "e_s", "h_i", "h_s", "a_i", "sv_i")


def _ulp_distance(left: float, right: float) -> int:
    """ULP distance for finite, same-sign binary64 operands."""

    x, y = np.float64(left), np.float64(right)
    phase2.require(np.isfinite(x) and np.isfinite(y), "ULP nonfinite")
    phase2.require(np.signbit(x) == np.signbit(y), "ULP sign mismatch")
    return abs(int(x.view(np.int64)) - int(y.view(np.int64)))


def _read_operand_steps(path: Path) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    with path.open("rb") as stream:
        while magic := stream.read(16):
            phase2.require(magic == b"NEMO_L3ZDF_001  ", "ZDF-operand magic")
            raw = stream.read(36)
            phase2.require(len(raw) == 36, "truncated ZDF-operand header")
            header = struct.unpack("=9i", raw)
            version, step, frame, iteration, npti, ni, ns, bits, nval = header
            phase2.require(
                (version, npti, ni, ns, bits) == (1, 1, 3, 3, 64),
                f"ZDF-operand header {header}",
            )
            phase2.require(frame in phase2.ZDF_OPERAND_REGISTRY,
                           f"unregistered ZDF frame {frame}")
            name, expected, source = phase2.ZDF_OPERAND_REGISTRY[frame]
            phase2.require(nval == expected, f"ZDF-operand count {header}")
            values = np.fromfile(stream, np.float64, nval)
            phase2.require(values.size == nval and np.all(np.isfinite(values)),
                           "bad ZDF-operand payload")
            grouped.setdefault(step, []).append(
                {"frame": frame, "name": name, "iteration": iteration,
                 "source": source, "values": values}
            )
    phase2.require(set(grouped) == {1, 3}, f"operand steps {sorted(grouped)}")
    for step, frames in grouped.items():
        phase2.require(len(frames) >= 7 and (len(frames) - 1) % 6 == 0,
                       f"step {step}: incomplete iteration registry")
        niter = (len(frames) - 1) // 6
        expected = [(0, 0)] + [
            (frame, iteration)
            for iteration in range(1, niter + 1)
            for frame in range(1, 7)
        ]
        phase2.require(
            [(f["frame"], f["iteration"]) for f in frames] == expected,
            f"step {step}: operand order",
        )
        convergence = [np.asarray(f["values"]) for f in frames if f["frame"] == 6]
        phase2.require(
            all(v[8] == 0.0 for v in convergence[:-1]) and convergence[-1][8] == 1.0,
            f"step {step}: convergence flags",
        )
    return grouped


def _read_reassociation_operands(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as stream:
        phase2.require(stream.read(16) == b"NEMO_L3REA_001  ", "reassoc magic")
        raw = stream.read(32)
        phase2.require(len(raw) == 32, "truncated reassoc header")
        header = struct.unpack("=8i", raw)
        phase2.require(header == (1, 3, 1, 1, 3, 3, 64, 18),
                       f"reassoc header {header}")
        values = np.fromfile(stream, np.float64, 18)
        phase2.require(values.size == 18 and np.all(np.isfinite(values)),
                       "bad reassoc payload")
        phase2.require(stream.read(1) == b"", "trailing reassoc bytes")
    return {
        "t_su": values[0:1], "t_i": values[1:4], "t_s": values[4:7],
        "s_i": values[7:10], "e_i": values[10:13], "e_s": values[13:16],
        "qns_ice": values[16:17], "dqns_ice": values[17:18],
    }


def _frame(frames, frame: int, iteration: int) -> np.ndarray:
    return np.asarray(next(
        item["values"] for item in frames
        if item["frame"] == frame and item["iteration"] == iteration
    ))


def _nemo_thomas(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalar replay of `icethd_zdf_bl99.F90:516-558`, in written order."""

    a, b, c, d = [np.array(matrix[k * 7:(k + 1) * 7], copy=True)
                  for k in range(4)]
    for k in range(1, 7):
        b[k] = b[k] - (a[k] * c[k - 1]) / b[k - 1]
        d[k] = d[k] - (a[k] * d[k - 1]) / b[k - 1]
    solution = np.empty_like(b)
    solution[-1] = d[-1] / b[-1]
    for k in range(5, -1, -1):
        solution[k] = (d[k] - c[k] * solution[k + 1]) / b[k]
    return b, d, solution


def _normalised_thomas(matrix: np.ndarray) -> np.ndarray:
    """Scalar replay of shared `tridiagonal.py:103-159` association."""

    a, b, c, d = [np.array(matrix[k * 7:(k + 1) * 7], copy=True)
                  for k in range(4)]
    cstar, dstar = np.empty_like(c), np.empty_like(d)
    cstar[0], dstar[0] = c[0] / b[0], d[0] / b[0]
    for k in range(1, 7):
        denominator = b[k] - a[k] * cstar[k - 1]
        cstar[k] = c[k] / denominator
        dstar[k] = (d[k] - a[k] * dstar[k - 1]) / denominator
    solution = np.empty_like(b)
    solution[-1] = dstar[-1]
    for k in range(5, -1, -1):
        solution[k] = dstar[k] - cstar[k] * solution[k + 1]
    return solution


def _ice_enthalpy_nemo_order(temperature, salinity, constants) -> np.ndarray:
    """Replay `icevar.F90:938-946` with scalar binary64 operations."""

    result = []
    for temperature_k, salinity_psu in zip(temperature, salinity, strict=True):
        melting_c = np.float64(-np.float64(constants.liquidus_slope)
                               * np.float64(salinity_psu))
        temperature_c = np.float64(np.float64(temperature_k)
                                   - np.float64(constants.T0))
        sensible = np.float64(np.float64(constants.c_ice)
                              * np.float64(melting_c - temperature_c))
        liquid = np.float64(max(
            0.0,
            np.float64(1.0 - np.float64(
                melting_c / np.float64(min(temperature_c, -1.0e-10))
            )),
        ))
        latent = np.float64(np.float64(constants.latent_fusion) * liquid)
        ocean = np.float64(np.float64(constants.c_ocean) * melting_c)
        result.append(np.float64(
            np.float64(constants.rho_ice)
            * np.float64(np.float64(sensible + latent) - ocean)
        ))
    return np.asarray(result, dtype=np.float64)


def _arithmetic_replay(root: Path, constants, *, plant=False) -> dict[str, object]:
    from legoesm.ice.bitz_lipscomb import _si3_zdf_bl99_step

    zdf_path = root / "oracle_si3_zdf_operands.bin"
    reassoc_path = root / "oracle_si3_reassoc_operands.bin"
    phase2.require(phase2.sha256(zdf_path) == EXPANDED_ZDF_SHA256,
                   "expanded ZDF operand SHA256")
    phase2.require(phase2.sha256(reassoc_path) == REASSOC_SHA256,
                   "reassociation operand SHA256")
    grouped = _read_operand_steps(zdf_path)
    post = _read_reassociation_operands(reassoc_path)

    replay_rows = []
    for step in (1, 3):
        frames = grouped[step]
        for iteration in (1, 2):
            matrix = _frame(frames, 3, iteration)
            forward = _frame(frames, 4, iteration)
            solution = _frame(frames, 5, iteration)
            diagonal, rhs, solved = _nemo_thomas(matrix)
            if plant and step == 1 and iteration == 1:
                solution = solution.copy()
                solution[0] = np.nextafter(solution[0], np.inf)
                solution[0] = np.nextafter(solution[0], np.inf)
                solution[0] = np.nextafter(solution[0], np.inf)
            replay_rows.append({
                "step": step,
                "iteration": iteration,
                "nemo_forward_max_ulp": max(
                    _ulp_distance(x, y)
                    for x, y in zip(np.concatenate((diagonal, rhs)), forward,
                                    strict=True)
                ),
                "nemo_solution_max_ulp": max(
                    _ulp_distance(x, y)
                    for x, y in zip(solved, solution[:7], strict=True)
                ),
                "normalised_solution_max_ulp": max(
                    _ulp_distance(x, y)
                    for x, y in zip(_normalised_thomas(matrix), solution[:7], strict=True)
                ),
            })

    init1 = _frame(grouped[1], 0, 0)
    solution1 = _frame(grouped[1], 5, 1)
    solution2 = _frame(grouped[1], 5, 2)
    qns_nemo = np.float64(
        init1[13] + init1[14] * np.float64(solution1[0] - solution1[7])
    )
    normalised_tsu1 = _normalised_thomas(_frame(grouped[1], 3, 1))[0]
    qns_normalised = np.float64(
        init1[13] + init1[14] * np.float64(normalised_tsu1 - solution1[7])
    )

    production = {}
    with (
        (root / "oracle_si3_thd_frames.bin").open("rb") as thd,
        (root / "oracle_si3_zdf_inputs.bin").open("rb") as zin,
    ):
        for step in range(1, 4):
            thermo = phase2._read_thd_step(thd, step)
            exact = phase2._read_zdf_input_step(zin, step)
            if step not in (1, 3):
                continue
            entry = jax.tree.map(
                lambda value: jnp.asarray(value, dtype=jnp.float64),
                phase2._entry_arrays(thermo[0]),
            )
            zdf = _si3_zdf_bl99_step(
                entry.e_ice, entry.e_snow, entry.S_layers, entry.h_ice,
                entry.h_snow, entry.T_surface, _forcing(exact, jnp.float64),
                3600.0, constants,
            )
            production[step] = jax.device_get(zdf)

    enthalpy_nemo = _ice_enthalpy_nemo_order(post["t_i"], post["s_i"], constants)
    qns_nemo_ulp = _ulp_distance(qns_nemo, solution2[8])
    enthalpy_nemo_ulp = max(
        _ulp_distance(x, y) for x, y in zip(enthalpy_nemo, post["e_i"], strict=True)
    )
    max_nemo_replay_ulp = max(
        [qns_nemo_ulp, enthalpy_nemo_ulp]
        + [max(row["nemo_forward_max_ulp"], row["nemo_solution_max_ulp"])
           for row in replay_rows]
    )
    phase2.require(max_nemo_replay_ulp <= 2,
                   f"NEMO operation-order replay exceeds 2 ULP: {max_nemo_replay_ulp}")
    return {
        "classification": "FLOAT RE-ASSOCIATION",
        "source": {
            "thomas": "icethd_zdf_bl99.F90:516-558",
            "qns_update": "icethd_zdf_bl99.F90:367-379",
            "enthalpy": "icevar.F90:938-946",
            "legoesm_normalised_thomas": "tridiagonal.py:103-159",
        },
        "nemo_written_order_max_ulp": max_nemo_replay_ulp,
        "thomas_rows": replay_rows,
        "iteration2_qns": {
            "oracle": float(solution2[8]),
            "nemo_order_replay": float(qns_nemo),
            "nemo_order_ulp": qns_nemo_ulp,
            "normalised_order_replay": float(qns_normalised),
            "normalised_minus_oracle": float(qns_normalised - solution2[8]),
            "normalised_ulp": _ulp_distance(qns_normalised, solution2[8]),
            "executing_legoesm": float(np.asarray(production[1].qns_ice)[0]),
            "executing_legoesm_minus_oracle": float(
                np.asarray(production[1].qns_ice)[0] - solution2[8]
            ),
            "executing_legoesm_ulp": _ulp_distance(
                np.asarray(production[1].qns_ice)[0], solution2[8]
            ),
        },
        "kt3_enthalpy": {
            "oracle": post["e_i"].tolist(),
            "nemo_order_replay": enthalpy_nemo.tolist(),
            "nemo_order_max_ulp": enthalpy_nemo_ulp,
            "first_upstream_normalised_solver_max_ulp": replay_rows[2][
                "normalised_solution_max_ulp"
            ],
            "executing_legoesm": np.asarray(production[3].e_ice)[0].tolist(),
            "executing_legoesm_minus_oracle": (
                np.asarray(production[3].e_ice)[0] - post["e_i"]
            ).tolist(),
            "executing_legoesm_max_ulp": max(
                _ulp_distance(x, y)
                for x, y in zip(
                    np.asarray(production[3].e_ice)[0], post["e_i"], strict=True
                )
            ),
        },
    }


def _forcing(values, dtype):
    from legoesm.ice.bitz_lipscomb import SI3SurfaceForcing

    return SI3SurfaceForcing(*(
        jnp.asarray(values[name], dtype=dtype) for name in phase2.ZDF_INPUT_NAMES
    ))


def _metric_fields(global_state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    a = np.asarray(global_state["a_i"]).reshape(-1)
    vi = np.asarray(global_state["v_i"]).reshape(-1)
    vs = np.asarray(global_state["v_s"]).reshape(-1)
    return {
        "t_su": np.asarray(global_state["t_su"]).reshape(-1),
        "e_i": np.asarray(global_state["e_i"]).reshape(-1),
        "e_s": np.asarray(global_state["e_s"]).reshape(-1),
        "h_i": (vi / a).reshape(-1),
        "h_s": (vs / a).reshape(-1),
        "a_i": a,
        "sv_i": np.asarray(global_state["sv_i"]).reshape(-1),
    }


def _normalised_error(oracle, candidate) -> tuple[float, float]:
    oracle, candidate = np.asarray(oracle), np.asarray(candidate)
    phase2.require(oracle.shape == candidate.shape, "metric shape")
    phase2.require(np.all(np.isfinite(oracle)) and np.all(np.isfinite(candidate)),
                   "metric nonfinite")
    absolute = float(np.max(np.abs(candidate - oracle)))
    scale = max(1.0, float(np.max(np.abs(oracle))))
    return absolute, absolute / scale


def _stage_maps(trace) -> tuple[dict[str, np.ndarray], ...]:
    states = (trace.entry, trace.post_zdf, trace.post_dh, trace.post_temp1,
              trace.post_sal, trace.post_temp2, trace.post_do, trace.exit)
    return tuple(
        phase2._selected_from_arrays(state)
        if 1 <= index <= 5 else phase2._global_from_arrays(state)
        for index, state in enumerate(states)
    )


def _operator_sweep(thd_path: Path, zin_path: Path, card, step_function) -> dict[str, object]:
    """Run oracle-entry steps in eager batches, preserving the reviewed gate order."""

    from legoesm.ice.bitz_lipscomb import SI3SurfaceForcing

    summary: dict[str, object] = {
        "frames_compared": 0, "field_rows_compared": 0,
        "over_bar_field_rows": 0, "maximum_normalised_error": 0.0,
        "first_over_bar_frame": None,
        "execution": "eager batched; same arithmetic path as phase-2 boundary gate",
    }
    chunk_size = 1000
    with thd_path.open("rb") as thd, zin_path.open("rb") as zin:
        for first_step in range(1, card.nsteps + 1, chunk_size):
            last_step = min(card.nsteps, first_step + chunk_size - 1)
            chunk_frames, entries, exact_inputs = [], [], []
            for kt in range(first_step, last_step + 1):
                frames = phase2._read_thd_step(thd, kt)
                exact = phase2._read_zdf_input_step(zin, kt)
                chunk_frames.append(frames)
                entries.append(phase2._entry_arrays(frames[0]))
                exact_inputs.append(exact)
            batch_entry = jax.tree.map(
                lambda *values: jnp.asarray(np.concatenate(values, axis=0),
                                            dtype=jnp.float64),
                *entries,
            )
            batch_forcing = SI3SurfaceForcing(*(
                jnp.asarray(
                    np.concatenate([values[name] for values in exact_inputs]),
                    dtype=jnp.float64,
                )
                for name in phase2.ZDF_INPUT_NAMES
            ))
            batch_maps = _stage_maps(jax.device_get(
                step_function(batch_entry, batch_forcing)
            ))
            for offset, frames in enumerate(chunk_frames):
                kt = first_step + offset
                summary["frames_compared"] += len(frames)
                for stage_index, (oracle_frame, candidates) in enumerate(
                    zip(frames, batch_maps, strict=True)
                ):
                    oracle = (oracle_frame if 1 <= stage_index <= 5
                              else phase2._center_global(oracle_frame))
                    names = (phase2.SELECTED_NAMES if 1 <= stage_index <= 5
                             else phase2.GLOBAL_NAMES)
                    for name in names:
                        wanted = np.asarray(oracle[name])
                        got = np.asarray(candidates[name][offset:offset + 1])
                        if got.ndim == 2 and wanted.ndim == 1:
                            wanted = wanted[None, :]
                        absolute, normalised = _normalised_error(
                            np.atleast_1d(wanted), np.atleast_1d(got)
                        )
                        summary["field_rows_compared"] += 1
                        summary["maximum_normalised_error"] = max(
                            summary["maximum_normalised_error"], normalised
                        )
                        if normalised > BAR:
                            summary["over_bar_field_rows"] += 1
                            if summary["first_over_bar_frame"] is None:
                                summary["first_over_bar_frame"] = {
                                    "step": kt,
                                    "sub_call": phase2.STAGES[stage_index][1],
                                    "variable": name,
                                    "absolute_max": absolute,
                                    "normalised_max_abs": normalised,
                                    "bar": BAR,
                                }
        phase2.require(thd.read(1) == b"", "trailing thermodynamics frame")
        phase2.require(zin.read(1) == b"", "trailing ZDF input frame")
    return summary


def _phenology(series: np.ndarray) -> dict[str, object]:
    phase2.require(series.shape == (8760,) and np.all(np.isfinite(series)),
                   "phenology series")
    start = date(2018, 1, 1)
    daily = series.reshape(365, 24)[:, -1]
    maximum_day = int(np.argmax(daily))

    def sequence_after(anchor: int, positive: bool):
        for day_index in range(anchor + 1, len(daily) - 6):
            change = daily[day_index:day_index + 7] - daily[day_index - 1:day_index + 6]
            if bool(np.all(change > 0.0) if positive else np.all(change < 0.0)):
                return day_index
        return None

    melt = sequence_after(maximum_day, positive=False)
    minimum_search_start = maximum_day + 1
    minimum_day = minimum_search_start + int(np.argmin(daily[minimum_search_start:]))
    growth = sequence_after(minimum_day, positive=True)
    positive = np.flatnonzero(series > 0.0)
    phase2.require(positive.size > 0, "no positive-concentration thickness")
    minimum_hour = int(positive[np.argmin(series[positive])])
    maximum_hour = int(positive[np.argmax(series[positive])])

    def day_string(index):
        return None if index is None else str(start + timedelta(days=index))

    return {
        "minimum_m": float(series[minimum_hour]),
        "minimum_utc": f"{start + timedelta(hours=minimum_hour):%Y-%m-%dT%H}:00:00Z",
        "maximum_m": float(series[maximum_hour]),
        "maximum_utc": f"{start + timedelta(hours=maximum_hour):%Y-%m-%dT%H}:00:00Z",
        "melt_onset": day_string(melt),
        "growth_onset": day_string(growth),
        "melt_onset_day": melt,
        "growth_onset_day": growth,
    }


def _phenomenology_rows(nemo, fp64, fp32) -> list[dict[str, object]]:
    rows = []
    for key, units in (
        ("minimum_m", "m"), ("maximum_m", "m"),
        ("melt_onset_day", "day"), ("growth_onset_day", "day"),
    ):
        values = [model[key] for model in (nemo, fp64, fp32)]
        if any(value is None for value in values):
            rows.append({"quantity": key, "units": units, "status": "UNMEASURED",
                         "nemo": values[0], "legoesm_fp64": values[1],
                         "legoesm_fp32": values[2]})
            continue
        distance = abs(float(values[1]) - float(values[0]))
        floor = abs(float(values[2]) - float(values[1]))
        rows.append({
            "quantity": key, "units": units, "nemo": values[0],
            "legoesm_fp64": values[1], "legoesm_fp32": values[2],
            "legoesm_nemo_distance": distance,
            "legoesm_fp32_fp64_floor": floor,
            "status": "AT-FLOOR" if distance <= floor else "ABOVE-FLOOR",
        })
    return rows


def run(*, root: Path = REPLAY_ROOT, plant_arithmetic=False,
        plant_truncate=False) -> dict[str, object]:
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ice.bitz_lipscomb import si3_column_step_arrays
    from legoesm.ice.c1d_omip_l3 import (
        FORCING_SHA256, THERMO_STREAM_SHA256, ZDF_INPUT_STREAM_SHA256,
        build_c1d_omip_l3_card,
    )

    set_policy(PrecisionPolicy.fp64())
    phase2.require(get_policy() == PrecisionPolicy.fp64(), "precision policy")
    phase2.require(jax.default_backend() == "cpu", "CPU backend")
    card = build_c1d_omip_l3_card(oracle_root=root)
    thd_path = root / "oracle_si3_thd_frames.bin"
    zin_path = root / "oracle_si3_zdf_inputs.bin"
    phase2.require(phase2.sha256(card.forcing_path) == FORCING_SHA256, "forcing SHA256")
    phase2.require(phase2.sha256(thd_path) == THERMO_STREAM_SHA256, "thermo SHA256")
    phase2.require(phase2.sha256(zin_path) == ZDF_INPUT_STREAM_SHA256, "ZDF input SHA256")
    arithmetic = _arithmetic_replay(root, card.config.ice_constants,
                                    plant=plant_arithmetic)

    def eager_step(state, forcing):
        return si3_column_step_arrays(
            state, forcing, card.dt_seconds, card.config.ice_constants
        )

    step = jax.jit(lambda state, forcing: si3_column_step_arrays(
        state, forcing, card.dt_seconds, card.config.ice_constants
    ))
    operator = _operator_sweep(thd_path, zin_path, card, eager_step)
    continuous_per_step = []
    sample_table: dict[str, object] = {}
    h_nemo, h_fp64, h_fp32 = [], [], []
    state64 = state32 = None

    with thd_path.open("rb") as thd, zin_path.open("rb") as zin:
        for kt in range(1, card.nsteps + 1):
            if plant_truncate and kt == card.nsteps:
                raise phase2.GateError("planted truncated thermodynamics stream")
            frames = phase2._read_thd_step(thd, kt)
            exact_input = phase2._read_zdf_input_step(zin, kt)
            entry = phase2._entry_arrays(frames[0])
            forcing64 = _forcing(exact_input, jnp.float64)

            if kt == 1:
                state64 = jax.tree.map(
                    lambda value: jnp.asarray(value, dtype=jnp.float64), entry
                )
                state32 = jax.tree.map(
                    lambda value: jnp.asarray(value, dtype=jnp.float32), entry
                )
            trace64 = step(state64, forcing64)
            trace32 = step(state32, _forcing(exact_input, jnp.float32))
            state64, state32 = trace64.exit, trace32.exit
            exit64 = phase2._global_from_arrays(jax.device_get(state64))
            exit32 = phase2._global_from_arrays(jax.device_get(state32))
            oracle_exit = phase2._center_global(frames[7])
            oracle_metrics = _metric_fields(oracle_exit)
            fp64_metrics = _metric_fields(exit64)
            fp32_metrics = _metric_fields(exit32)
            row = {"step": kt, "fields": {}}
            for name in CONTINUOUS_FIELDS:
                absolute, normalised = _normalised_error(
                    oracle_metrics[name], fp64_metrics[name]
                )
                row["fields"][name] = {
                    "absolute_max": absolute, "normalised_linf": normalised,
                }
            continuous_per_step.append(row)
            if kt in REPORT_STEPS:
                sample_table[str(kt)] = row["fields"]
            h_nemo.append(float(oracle_metrics["h_i"][0]))
            h_fp64.append(float(fp64_metrics["h_i"][0]))
            h_fp32.append(float(fp32_metrics["h_i"][0]))
        phase2.require(thd.read(1) == b"", "trailing thermodynamics frame")
        phase2.require(zin.read(1) == b"", "trailing ZDF input frame")

    phase2.require(operator["frames_compared"] == 70080,
                   "thermodynamics frame count")
    phase2.require(len(continuous_per_step) == 8760, "trajectory step count")
    phase2.require(operator["first_over_bar_frame"] is not None,
                   "expected scientific debt disappeared")
    nemo_phen = _phenology(np.asarray(h_nemo))
    fp64_phen = _phenology(np.asarray(h_fp64))
    fp32_phen = _phenology(np.asarray(h_fp32))
    return {
        "status": "DEBT",
        "backend": jax.default_backend(),
        "precision_policy": str(get_policy()),
        "compute_dtypes": {"oracle": "float64", "legoesm_fp64": str(state64.e_ice.dtype),
                           "legoesm_floor_run": str(state32.e_ice.dtype)},
        "card": {"name": card.name, "dt_seconds": card.dt_seconds,
                 "nsteps": card.nsteps, "scope": "ORCA1-resolved identity only"},
        "hashes": {
            "forcing": phase2.sha256(card.forcing_path),
            "thermodynamics": phase2.sha256(thd_path),
            "zdf_inputs": phase2.sha256(zin_path),
            "zdf_operands": phase2.sha256(root / "oracle_si3_zdf_operands.bin"),
            "reassociation_operands": phase2.sha256(
                root / "oracle_si3_reassoc_operands.bin"
            ),
        },
        "arithmetic_replay": arithmetic,
        "oracle_entry_operator_sweep": operator,
        "continuous_trajectory": {
            "metric": "max(abs(legoesm-NEMO))/max(1,max(abs(NEMO)))",
            "sample_steps": sample_table,
            "per_step": continuous_per_step,
        },
        "phenomenology": {
            "definition": (
                "hourly positive-concentration extrema; daily EXIT onset is the first "
                "day beginning seven consecutive signed thickness changes"
            ),
            "nemo": nemo_phen, "legoesm_fp64": fp64_phen,
            "legoesm_fp32": fp32_phen,
            "classification_rows": _phenomenology_rows(nemo_phen, fp64_phen, fp32_phen),
            "floor_scope": "legoesm fp32-vs-fp64, not a NEMO scheme spread",
        },
        "coupled_rung_debt": (
            "NEMO sbcblk bulk-flux computation is NOT CERTIFIED: this isolated "
            "column consumes NEMO-written qns_ice/dqns_ice."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPLAY_ROOT)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--plant-arithmetic", action="store_true")
    parser.add_argument("--plant-truncate", action="store_true")
    args = parser.parse_args()
    result = run(root=args.root, plant_arithmetic=args.plant_arithmetic,
                 plant_truncate=args.plant_truncate)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(rendered + "\n")
    print(rendered)
    if result["status"] != "AT-BAR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
