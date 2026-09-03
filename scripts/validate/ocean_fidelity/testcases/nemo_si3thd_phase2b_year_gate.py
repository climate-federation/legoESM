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
from datetime import date, datetime, timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nemo_si3thd_phase2_gate as phase2


BAR = phase2.BAR
REPLAY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l3/"
    "c1d_omip_l3_sasice_phase4_operands"
)
EXPANDED_ZDF_SHA256 = "aad46579fb2d2cc19299d7a25992802525603bb9858adf1e892ff5d85bf40442"
REASSOC_SHA256 = "4b832b0c274d6aab032f16958224ebfb6fea603af22e1a1f1d474ff71b1e4589"
ROUND4_ZDF_INPUT_SHA256 = "cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd"
REPORT_STEPS = (1, 10, 100, 1000, 3000, 5000, 8760)
CONTINUOUS_FIELDS = ("t_su", "e_i", "e_s", "h_i", "h_s", "a_i", "sv_i")
BASELINE_JSON = (
    Path(__file__).resolve().parents[4]
    / "docs/ocean/fidelity/testcases/nemo_testcases_l3thd_phase2b_year_gate.json"
)
BASELINE_JSON_SHA256 = "6c21d14f3c85d0fa99be7e31770a4a4dcad548d98f857ea78455bd6344622160"
BASELINE_OPERATOR_JSON = (
    Path(__file__).resolve().parents[4]
    / "docs/ocean/fidelity/testcases/nemo_testcases_l3thd_phase3_baseline_operator.json"
)
BASELINE_OPERATOR_JSON_SHA256 = (
    "d642f532ed92910b49f29919dd5e4ae3fe84381ecd1536b7886a2c1f4b36f95f"
)


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


def _branch_label(stage_index: int, frames, exact_input, T0: float) -> str:
    """Name the active NEMO branch associated with a boundary row."""

    if stage_index == 0:
        return "ENTRY"
    post_zdf = frames[1]
    if stage_index == 1:
        if float(post_zdf["t_su"][0]) >= T0:
            return "ZDF_FIXED_MELTING_SURFACE"
        if float(post_zdf["h_s"][0]) > 0.0:
            return "ZDF_SNOW_PRESENT"
        return "ZDF_SNOW_FREE_ROW_RANGE"
    if stage_index == 2:
        if float(exact_input["evaporation"][0]) < 0.0:
            return "DH_NEGATIVE_EVAPORATION_SNOW_DEPOSITION"
        if float(post_zdf["t_su"][0]) >= T0:
            return "DH_SNOW_FIRST_SURFACE_MELT"
        return "DH_SNOW_SUBLIMATION_BASAL_THERMO_REMAP"
    if stage_index == 3:
        return "TEMP_FROM_ENTHALPY_AFTER_DH"
    if stage_index == 4:
        return "SAL_OPTION2_DRAINAGE_FLUSHING"
    if stage_index == 5:
        return "TEMP_FROM_ENTHALPY_AFTER_SAL"
    if stage_index == 6:
        return "DO_OPEN_WATER_GROWTH" if float(exact_input["qlead"][0]) < 0.0 else "DO_INACTIVE"
    return "ICE_COR_ZAPSMALL_AND_LBC"


def _branch_detail(stage_index: int, frames, candidates, offset: int,
                   T0: float, exact_input=None) -> dict[str, object]:
    """Expose both operands of the ZDF branch comparison."""

    if stage_index != 1:
        return {}
    oracle_tsu = float(frames[1]["t_su"][0])
    lego_tsu = float(np.asarray(candidates["t_su"])[offset])
    entry = phase2._center_global(frames[0])
    snow_volume = float(entry["v_s"])
    snow_energy = np.asarray(entry["e_s"])
    detail = {
        "nemo_surface_melting_condition": oracle_tsu >= T0,
        "legoesm_surface_melting_condition": lego_tsu >= T0,
        "surface_melting_condition_differs": ((oracle_tsu >= T0)
                                                != (lego_tsu >= T0)),
        "nemo_snow_present_condition": float(frames[1]["h_s"][0]) > 0.0,
        "entry_snow_volume_m": snow_volume,
        "entry_snow_enthalpy_all_zero": bool(np.all(snow_energy == 0.0)),
        "entry_temperature_operand_status": "REGISTERED_POST_GLO2EQV_PRE_ZDF",
    }
    if exact_input is not None and "t_s" in exact_input:
        detail["nemo_t_s_operand_K"] = np.asarray(exact_input["t_s"]).tolist()
    return detail


def _operator_sweep(thd_path: Path, zin_path: Path, card, step_function, *,
                    retain_over_bar_rows: bool = False,
                    snapshot_steps: tuple[int, ...] = ()) -> dict[str, object]:
    """Run oracle-entry steps in eager batches, preserving the reviewed gate order."""

    from legoesm.ice.bitz_lipscomb import SI3SurfaceForcing

    summary: dict[str, object] = {
        "frames_compared": 0, "field_rows_compared": 0,
        "over_bar_field_rows": 0, "maximum_normalised_error": 0.0,
        "first_over_bar_frame": None,
        "largest_outlier": None,
        "per_step": [],
        "execution": "eager batched; same arithmetic path as phase-2 boundary gate",
    }
    if retain_over_bar_rows:
        summary["over_bar_by_step"] = {}
        summary["first_above_1e-12"] = None
        summary["first_above_1e-3"] = None
    if snapshot_steps:
        summary["full_state_snapshots"] = {}
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
                exact_input = exact_inputs[offset]
                step_record = {
                    "step": kt,
                    "maximum_normalised_error": 0.0,
                    "over_bar_field_rows": 0,
                    "owner": None,
                }
                summary["frames_compared"] += len(frames)
                if kt in snapshot_steps:
                    summary["full_state_snapshots"][str(kt)] = {
                        "exact_zdf_input": {
                            name: np.asarray(exact_input[name]).tolist()
                            for name in (*phase2.ZDF_INPUT_NAMES,
                                         *phase2.ZDF_STATE_OPERAND_NAMES)
                        },
                        "frames": [],
                    }
                for stage_index, (oracle_frame, candidates) in enumerate(
                    zip(frames, batch_maps, strict=True)
                ):
                    oracle = (oracle_frame if 1 <= stage_index <= 5
                              else phase2._center_global(oracle_frame))
                    names = (phase2.SELECTED_NAMES if 1 <= stage_index <= 5
                             else phase2.GLOBAL_NAMES)
                    if kt in snapshot_steps:
                        summary["full_state_snapshots"][str(kt)]["frames"].append({
                            "stage": phase2.STAGES[stage_index][1],
                            "time_level": phase2.STAGES[stage_index][2],
                            "source": phase2.STAGES[stage_index][3],
                            "branch": _branch_label(
                                stage_index, frames, exact_input,
                                card.config.ice_constants.T0,
                            ),
                            "oracle": {
                                name: np.asarray(oracle[name]).reshape(-1).tolist()
                                for name in names
                            },
                            "legoesm": {
                                name: np.asarray(candidates[name][offset:offset + 1])
                                .reshape(-1).tolist()
                                for name in names
                            },
                        })
                    for name in names:
                        wanted = np.asarray(oracle[name])
                        got = np.asarray(candidates[name][offset:offset + 1])
                        if got.ndim == 2 and wanted.ndim == 1:
                            wanted = wanted[None, :]
                        absolute, normalised = _normalised_error(
                            np.atleast_1d(wanted), np.atleast_1d(got)
                        )
                        scale = max(1.0, float(np.max(np.abs(wanted))))
                        outlier = {
                            "step": kt,
                            "sub_call": phase2.STAGES[stage_index][1],
                            "variable": name,
                            "absolute_numerator": absolute,
                            "normalisation_denominator": scale,
                            "normalised_quotient": normalised,
                            "branch": _branch_label(
                                stage_index, frames, exact_input,
                                card.config.ice_constants.T0,
                            ),
                        }
                        if normalised >= step_record["maximum_normalised_error"]:
                            step_record["maximum_normalised_error"] = normalised
                            step_record["owner"] = outlier
                        largest = summary["largest_outlier"]
                        if (largest is None
                                or normalised > largest["normalised_quotient"]):
                            summary["largest_outlier"] = outlier
                        summary["field_rows_compared"] += 1
                        summary["maximum_normalised_error"] = max(
                            summary["maximum_normalised_error"], normalised
                        )
                        if normalised > BAR:
                            summary["over_bar_field_rows"] += 1
                            step_record["over_bar_field_rows"] += 1
                            record = {
                                "sub_call": phase2.STAGES[stage_index][1],
                                "variable": name,
                                "absolute_max": absolute,
                                "normalised_max_abs": normalised,
                                "branch": _branch_label(
                                    stage_index, frames, exact_input,
                                    card.config.ice_constants.T0,
                                ),
                                **_branch_detail(
                                    stage_index, frames, candidates, offset,
                                    card.config.ice_constants.T0,
                                    exact_input,
                                ),
                            }
                            if summary["first_over_bar_frame"] is None:
                                summary["first_over_bar_frame"] = {
                                    "step": kt, **record, "bar": BAR,
                                }
                            if retain_over_bar_rows:
                                summary["over_bar_by_step"].setdefault(str(kt), []).append(record)
                                for threshold, key in (
                                    (1.0e-12, "first_above_1e-12"),
                                    (1.0e-3, "first_above_1e-3"),
                                ):
                                    if normalised > threshold and summary[key] is None:
                                        summary[key] = {"step": kt, **record,
                                                        "reporting_threshold": threshold}
                summary["per_step"].append(step_record)
        phase2.require(thd.read(1) == b"", "trailing thermodynamics frame")
        phase2.require(zin.read(1) == b"", "trailing ZDF input frame")
    return summary


def _validate_operator_trajectory(operator, *, plant=False) -> None:
    """Fail closed on the complete exact-entry trajectory and max attribution."""

    rows = list(operator["per_step"])
    if plant:
        rows.pop()
    phase2.require(len(rows) == 8760, "oracle-entry per-step count")
    phase2.require([row["step"] for row in rows] == list(range(1, 8761)),
                   "oracle-entry per-step sequence")
    recomputed = max(rows, key=lambda row: row["maximum_normalised_error"])
    largest = operator["largest_outlier"]
    phase2.require(
        largest is not None
        and largest["step"] == recomputed["step"]
        and largest["normalised_quotient"]
        == recomputed["maximum_normalised_error"]
        and largest["absolute_numerator"]
        / largest["normalisation_denominator"]
        == largest["normalised_quotient"],
        "oracle-entry largest-outlier attribution",
    )


def _validate_step74_branch_snapshot(operator, *, plant=False) -> None:
    """Fail closed if the retained event row is missing or mislabelled."""

    label = operator["full_state_snapshots"]["74"]["frames"][2]["branch"]
    if plant:
        label = "PLANTED_WRONG_BRANCH"
    phase2.require(label == "DH_NEGATIVE_EVAPORATION_SNOW_DEPOSITION",
                   "step-74 branch census plant")


def _read_exact_steps(root: Path, wanted: tuple[int, ...]):
    """Read the registered oracle frames/inputs for a small arm set."""

    selected = {}
    with (
        (root / "oracle_si3_thd_frames.bin").open("rb") as thd,
        (root / "oracle_si3_zdf_inputs.bin").open("rb") as zin,
    ):
        for step in range(1, max(wanted) + 1):
            frames = phase2._read_thd_step(thd, step)
            exact = phase2._read_zdf_input_step(zin, step)
            if step in wanted:
                selected[step] = (frames, exact)
    phase2.require(set(selected) == set(wanted), "owner-arm exact steps")
    return selected


def _arm_row(oracle, disabled, enabled, name: str) -> dict[str, object]:
    """Score a one-variable disabled/enabled owner arm."""

    old_abs, old_norm = _normalised_error(oracle, disabled)
    new_abs, new_norm = _normalised_error(oracle, enabled)
    denominator = max(new_abs, np.finfo(np.float64).eps)
    return {
        "variable": name,
        "disabled_absolute_max": old_abs,
        "disabled_normalised_max_abs": old_norm,
        "enabled_absolute_max": new_abs,
        "enabled_normalised_max_abs": new_norm,
        "improvement_factor": old_abs / denominator,
    }


def _owner_arms(root: Path, card, *, plant_deposition=False,
                plant_surface=False, plant_snow_temperature=False) -> dict[str, object]:
    """Evaluate preregistered one-variable DH arms on exact NEMO entries."""

    from legoesm.ice.bitz_lipscomb import si3_column_step_arrays

    selected = _read_exact_steps(root, (73, 74, 3836, 4238, 4239, 5285))

    def trace(step, **kwargs):
        frames, exact = selected[step]
        state = jax.tree.map(
            lambda value: jnp.asarray(value, dtype=jnp.float64),
            phase2._entry_arrays(frames[0]),
        )
        return jax.device_get(si3_column_step_arrays(
            state, _forcing(exact, jnp.float64), card.dt_seconds,
            card.config.ice_constants, **kwargs,
        ))

    frames74, exact74 = selected[74]
    deposition_on = trace(74)
    deposition_off = trace(74, _snow_deposition=False)
    if plant_deposition:
        deposition_on = deposition_off
    dep_on = phase2._selected_from_arrays(deposition_on.post_dh)
    dep_off = phase2._selected_from_arrays(deposition_off.post_dh)
    deposition_rows = [
        _arm_row(frames74[2][name], dep_off[name], dep_on[name], name)
        for name in ("h_s", "e_s")
    ]
    phase2.require(
        min(row["improvement_factor"] for row in deposition_rows) >= 100.0,
        "negative-evaporation deposition arm failed 100-fold discriminator",
    )

    surface_rows = []
    for step, fields in ((3836, ("h_s", "e_s")), (4238, ("h_i", "e_i"))):
        frames, _ = selected[step]
        surface_on = trace(step)
        surface_off = trace(step, _surface_melt=False)
        if plant_surface:
            surface_on = surface_off
        on = phase2._selected_from_arrays(surface_on.post_dh)
        off = phase2._selected_from_arrays(surface_off.post_dh)
        rows = [_arm_row(frames[2][name], off[name], on[name], name)
                for name in fields]
        surface_rows.append({"step": step, "rows": rows})
    phase2.require(
        min(row["improvement_factor"] for group in surface_rows
            for row in group["rows"]) >= 100.0,
        "surface-melt arm failed 100-fold discriminator",
    )

    snow_temperature_steps = []
    for step in (4239, 5285):
        frames, exact = selected[step]
        enabled = trace(step)
        disabled = trace(step, _nemo_snow_temperature_bounds=False)
        if plant_snow_temperature:
            enabled = disabled
        oracle = frames[1]
        on = phase2._selected_from_arrays(enabled.post_zdf)
        off = phase2._selected_from_arrays(disabled.post_zdf)
        rows = [_arm_row(oracle[name], off[name], on[name], name)
                for name in ("t_su", "e_i", "e_s")]
        entry = phase2._entry_arrays(frames[0])
        c = card.config.ice_constants
        unbounded = c.T0 + (
            -np.asarray(entry.e_snow) / c.rho_snow / c.c_ice
            + c.latent_fusion / c.c_ice
        )
        dumped = np.asarray(exact["t_s"])
        snow_temperature_steps.append({
            "step": step,
            "registered_time_level": phase2.ZDF_STATE_OPERAND_REGISTRY["t_s"],
            "nemo_t_s_operand_K": dumped.tolist(),
            "source_replay_K": np.clip(unbounded, c.T0 - 100.0, c.T0).tolist(),
            "pre_fix_unbounded_operand_K": unbounded.tolist(),
            "source_replay_bit_exact": bool(np.array_equal(
                dumped, np.clip(unbounded, c.T0 - 100.0, c.T0)
            )),
            "nemo_cold_surface_condition": bool(oracle["t_su"][0] < c.T0),
            "pre_fix_cold_surface_condition": bool(off["t_su"][0] < c.T0),
            "post_fix_cold_surface_condition": bool(on["t_su"][0] < c.T0),
            "rows": rows,
        })
    phase2.require(
        all(row["source_replay_bit_exact"] for row in snow_temperature_steps),
        "registered snow temperature disagrees with NEMO source replay",
    )
    phase2.require(
        min(
            row["improvement_factor"]
            for step in snow_temperature_steps for row in step["rows"]
            if row["variable"] == "e_i"
        ) >= 100.0,
        "snow-temperature bounds arm failed 100-fold discriminator",
    )

    _, exact73 = selected[73]
    return {
        "step74_negative_evaporation_deposition": {
            "source": "icethd_dh.F90:166-202,494-507,535-613",
            "step73_evaporation": float(exact73["evaporation"][0]),
            "step74_evaporation": float(exact74["evaporation"][0]),
            "step74_snow_precipitation": float(
                exact74["snow_precipitation"][0]
            ),
            "first_disagreeing_boundary": "POST_DH",
            "rows": deposition_rows,
            "verdict": "CONFIRMED",
        },
        "surface_melt": {
            "source": "icethd_dh.F90:107-120,204-315",
            "first_snow_removal_step": 3836,
            "first_ice_removal_step": 4238,
            "snow_then_ice_order": True,
            "rows": surface_rows,
            "verdict": "CONFIRMED",
        },
        "snow_temperature_bounds": {
            "source": (
                "icestp.F90:182-206; icevar.F90:404-416; "
                "icethd.F90:343-355,418-435; "
                "icethd_zdf_bl99.F90:159-198,433-513"
            ),
            "nemo_predicate": "v_s > epsi20 (epsi20=1e-20), then [rt0-100,rt0]",
            "legoesm_pre_fix_predicate": "h_s > 0, unbounded enthalpy inverse",
            "steps": snow_temperature_steps,
            "verdict": "CONFIRMED",
        },
    }


def _melt_scaling(root: Path, card) -> dict[str, object]:
    """Independent exact-entry candidate scales over all 8,760 ice steps."""

    from legoesm.ice.bitz_lipscomb import SI3SurfaceForcing, si3_column_step_arrays

    totals = {
        "surface_ice_m": 0.0,
        "surface_snow_m": 0.0,
        "basal_ice_m": 0.0,
        "shortwave_signed_ice_m": 0.0,
        "shortwave_signed_snow_m": 0.0,
    }
    first_active = {}
    chunk_size = 1000
    with (
        (root / "oracle_si3_thd_frames.bin").open("rb") as thd,
        (root / "oracle_si3_zdf_inputs.bin").open("rb") as zin,
    ):
        for first_step in range(1, card.nsteps + 1, chunk_size):
            last_step = min(card.nsteps, first_step + chunk_size - 1)
            entries, exact_inputs = [], []
            for step in range(first_step, last_step + 1):
                frames = phase2._read_thd_step(thd, step)
                exact = phase2._read_zdf_input_step(zin, step)
                entries.append(phase2._entry_arrays(frames[0]))
                exact_inputs.append(exact)
            state = jax.tree.map(
                lambda *values: jnp.asarray(np.concatenate(values, axis=0),
                                            dtype=jnp.float64),
                *entries,
            )
            forcing = SI3SurfaceForcing(*(
                jnp.asarray(np.concatenate([row[name] for row in exact_inputs]),
                            dtype=jnp.float64)
                for name in phase2.ZDF_INPUT_NAMES
            ))
            full = si3_column_step_arrays(
                state, forcing, card.dt_seconds, card.config.ice_constants
            )
            no_surface = si3_column_step_arrays(
                state, forcing, card.dt_seconds, card.config.ice_constants,
                _surface_melt=False,
            )
            no_basal = si3_column_step_arrays(
                state, forcing, card.dt_seconds, card.config.ice_constants,
                _basal_melt=False,
            )
            no_shortwave = si3_column_step_arrays(
                state,
                forcing._replace(qtr_ice_top=jnp.zeros_like(forcing.qtr_ice_top)),
                card.dt_seconds,
                card.config.ice_constants,
            )
            values = {
                "surface_ice_m": np.asarray(
                    no_surface.post_dh.h_ice - full.post_dh.h_ice
                ),
                "surface_snow_m": np.asarray(
                    no_surface.post_dh.h_snow - full.post_dh.h_snow
                ),
                "basal_ice_m": np.asarray(
                    no_basal.post_dh.h_ice - full.post_dh.h_ice
                ),
                "shortwave_signed_ice_m": np.asarray(
                    full.post_dh.h_ice - no_shortwave.post_dh.h_ice
                ),
                "shortwave_signed_snow_m": np.asarray(
                    full.post_dh.h_snow - no_shortwave.post_dh.h_snow
                ),
            }
            for name, value in values.items():
                totals[name] += float(np.sum(value))
                nonzero = np.flatnonzero(value != 0.0)
                if name not in first_active and nonzero.size:
                    offset = int(nonzero[0])
                    first_active[name] = {
                        "step": first_step + offset,
                        "one_step_thickness_m": float(value[offset]),
                    }
        phase2.require(thd.read(1) == b"", "melt scaling thermo cursor")
        phase2.require(zin.read(1) == b"", "melt scaling input cursor")

    observed_gap = 1.2273381761932358
    snow_ice_equivalent = (
        totals["surface_snow_m"] * card.config.ice_constants.rho_snow
        / card.config.ice_constants.rho_ice
    )
    return {
        "metric": "sum of independent exact-entry POST_DH arm thickness differences",
        "observed_phase2b_minimum_gap_m": observed_gap,
        "rows": [
            {
                "candidate": "qml surface ice melt",
                "source": "icethd_dh.F90:107-120,231-315",
                "scale_m": totals["surface_ice_m"],
                "ratio_to_observed_gap": totals["surface_ice_m"] / observed_gap,
                "first_active": first_active["surface_ice_m"],
                "baseline_presence": "MISSING",
            },
            {
                "candidate": "snow-melt-first shield (ice-equivalent)",
                "source": "icethd_dh.F90:204-225",
                "scale_m": snow_ice_equivalent,
                "raw_snow_depth_m": totals["surface_snow_m"],
                "ratio_to_observed_gap": snow_ice_equivalent / observed_gap,
                "first_active": first_active["surface_snow_m"],
                "baseline_presence": "MISSING WITH qml PATH",
            },
            {
                "candidate": "bottom melt sensitivity",
                "source": "icethd_dh.F90:321-424",
                "scale_m": totals["basal_ice_m"],
                "ratio_to_observed_gap": totals["basal_ice_m"] / observed_gap,
                "first_active": first_active["basal_ice_m"],
                "baseline_presence": "PRESENT; NOT THE FIRST MISSING BRANCH",
            },
            {
                "candidate": "qtr shortwave transmission sensitivity",
                "source": "icethd_zdf_bl99.F90:143-230; icethd_dh.F90:107-120",
                "scale_m": totals["shortwave_signed_ice_m"],
                "snow_scale_m": totals["shortwave_signed_snow_m"],
                "ratio_to_observed_gap": (
                    totals["shortwave_signed_ice_m"] / observed_gap
                ),
                "first_active": first_active["shortwave_signed_ice_m"],
                "baseline_presence": "PRESENT FROM THE PINNED NEMO INPUT",
            },
        ],
    }


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
    for key, units in (("minimum_m", "m"), ("maximum_m", "m")):
        values = [model[key] for model in (nemo, fp64, fp32)]
        distance = abs(float(values[1]) - float(values[0]))
        floor = abs(float(values[2]) - float(values[1]))
        rows.append({
            "quantity": key, "units": units, "nemo": values[0],
            "legoesm_fp64": values[1], "legoesm_fp32": values[2],
            "legoesm_nemo_distance": distance,
            "legoesm_fp32_fp64_floor": floor,
            "status": "AT-FLOOR" if distance <= floor else "ABOVE-FLOOR",
        })

    for key, units, divisor in (
        ("minimum_utc", "hour", 3600.0),
        ("maximum_utc", "hour", 3600.0),
        ("melt_onset", "day", 86400.0),
        ("growth_onset", "day", 86400.0),
    ):
        values = [model[key] for model in (nemo, fp64, fp32)]
        if any(value is None for value in values):
            rows.append({"quantity": key, "units": units, "status": "UNMEASURED",
                         "nemo": values[0], "legoesm_fp64": values[1],
                         "legoesm_fp32": values[2]})
            continue
        parsed = [datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                  for value in values]
        distance = abs((parsed[1] - parsed[0]).total_seconds()) / divisor
        floor = abs((parsed[2] - parsed[1]).total_seconds()) / divisor
        rows.append({
            "quantity": key, "units": units, "nemo": values[0],
            "legoesm_fp64": values[1], "legoesm_fp32": values[2],
            "legoesm_nemo_distance": distance,
            "legoesm_fp32_fp64_floor": floor,
            "status": "AT-FLOOR" if distance <= floor else "ABOVE-FLOOR",
        })
    return rows


def run(*, root: Path = REPLAY_ROOT, plant_arithmetic=False,
        plant_truncate=False, plant_branch_census=False,
        plant_deposition=False, plant_surface=False,
        plant_snow_temperature=False,
        plant_operator_trajectory=False) -> dict[str, object]:
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ice.bitz_lipscomb import si3_column_step_arrays
    from legoesm.ice.c1d_omip_l3 import (
        FORCING_SHA256, THERMO_STREAM_SHA256,
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
    phase2.require(phase2.sha256(zin_path) == ROUND4_ZDF_INPUT_SHA256,
                   "Round-4 ZDF input SHA256")
    arithmetic = _arithmetic_replay(root, card.config.ice_constants,
                                    plant=plant_arithmetic)
    owner_arms = _owner_arms(
        root, card, plant_deposition=plant_deposition,
        plant_surface=plant_surface,
        plant_snow_temperature=plant_snow_temperature,
    )

    def eager_step(state, forcing):
        return si3_column_step_arrays(
            state, forcing, card.dt_seconds, card.config.ice_constants
        )

    step = jax.jit(lambda state, forcing: si3_column_step_arrays(
        state, forcing, card.dt_seconds, card.config.ice_constants
    ))
    no_surface_step = jax.jit(lambda state, forcing: si3_column_step_arrays(
        state, forcing, card.dt_seconds, card.config.ice_constants,
        _surface_melt=False,
    ))
    no_basal_step = jax.jit(lambda state, forcing: si3_column_step_arrays(
        state, forcing, card.dt_seconds, card.config.ice_constants,
        _basal_melt=False,
    ))
    operator = _operator_sweep(
        thd_path, zin_path, card, eager_step, retain_over_bar_rows=True,
        snapshot_steps=(73, 74, 75, 76, 4239, 5285),
    )
    _validate_step74_branch_snapshot(operator, plant=plant_branch_census)
    _validate_operator_trajectory(operator, plant=plant_operator_trajectory)
    continuous_per_step = []
    sample_table: dict[str, object] = {}
    h_nemo, h_fp64, h_fp32 = [], [], []
    h_no_surface, h_no_basal = [], []
    state64 = state32 = state_no_surface = state_no_basal = None

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
                state_no_surface = state64
                state_no_basal = state64
            trace64 = step(state64, forcing64)
            trace32 = step(state32, _forcing(exact_input, jnp.float32))
            trace_no_surface = no_surface_step(state_no_surface, forcing64)
            trace_no_basal = no_basal_step(state_no_basal, forcing64)
            state64, state32 = trace64.exit, trace32.exit
            state_no_surface = trace_no_surface.exit
            state_no_basal = trace_no_basal.exit
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
            h_no_surface.append(float(state_no_surface.h_ice[0]))
            h_no_basal.append(float(state_no_basal.h_ice[0]))
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
    no_surface_phen = _phenology(np.asarray(h_no_surface))
    no_basal_phen = _phenology(np.asarray(h_no_basal))
    phase2.require(phase2.sha256(BASELINE_JSON) == BASELINE_JSON_SHA256,
                   "Phase-2b baseline JSON SHA256")
    phase2.require(
        phase2.sha256(BASELINE_OPERATOR_JSON) == BASELINE_OPERATOR_JSON_SHA256,
        "Phase-2b retained operator JSON SHA256",
    )
    baseline = json.loads(BASELINE_JSON.read_text())
    baseline_operator = json.loads(BASELINE_OPERATOR_JSON.read_text())
    baseline_first = baseline_operator["first_over_bar_frame"]
    phase2.require(
        operator["first_over_bar_frame"]["step"] > baseline_first["step"],
        "first over-bar frame did not move later",
    )
    old_phen = baseline["phenomenology"]["legoesm_fp64"]
    phase2.require(
        abs(fp64_phen["minimum_m"] - nemo_phen["minimum_m"])
        < abs(old_phen["minimum_m"] - nemo_phen["minimum_m"]),
        "minimum thickness did not move toward NEMO",
    )
    phase2.require(
        abs(fp64_phen["growth_onset_day"] - nemo_phen["growth_onset_day"])
        < abs(old_phen["growth_onset_day"] - nemo_phen["growth_onset_day"]),
        "growth onset did not move toward NEMO",
    )
    before_after = {}
    for report_step in REPORT_STEPS:
        key = str(report_step)
        before_after[key] = {
            field: {
                "before_normalised_linf": baseline["continuous_trajectory"]
                ["sample_steps"][key][field]["normalised_linf"],
                "after_normalised_linf": sample_table[key][field]["normalised_linf"],
            }
            for field in CONTINUOUS_FIELDS
        }
    return {
        "status": "DEBT",
        "backend": jax.default_backend(),
        "precision_policy": str(get_policy()),
        "compute_dtypes": {"oracle": "float64", "legoesm_fp64": str(state64.e_ice.dtype),
                           "legoesm_floor_run": str(state32.e_ice.dtype)},
        "card": {"name": card.name, "dt_seconds": card.dt_seconds,
                 "nsteps": card.nsteps, "scope": "ORCA1-resolved identity only"},
        "frame_registry": {
            "thermodynamics": {
                str(stage): {"name": name, "time_level": time_level,
                             "source": source}
                for stage, name, time_level, source in phase2.STAGES
            },
            "zdf_state_operands": phase2.ZDF_STATE_OPERAND_REGISTRY,
        },
        "hashes": {
            "forcing": phase2.sha256(card.forcing_path),
            "thermodynamics": phase2.sha256(thd_path),
            "zdf_inputs": phase2.sha256(zin_path),
            "zdf_operands": phase2.sha256(root / "oracle_si3_zdf_operands.bin"),
            "reassociation_operands": phase2.sha256(
                root / "oracle_si3_reassoc_operands.bin"
            ),
            "phase2b_baseline_json": phase2.sha256(BASELINE_JSON),
            "phase2b_retained_operator_json": phase2.sha256(
                BASELINE_OPERATOR_JSON
            ),
        },
        "arithmetic_replay": arithmetic,
        "owner_arms": owner_arms,
        "melt_candidate_scaling": _melt_scaling(root, card),
        "oracle_entry_operator_sweep": operator,
        "oracle_entry_operator_sweep_before": {
            "artifact": str(BASELINE_OPERATOR_JSON.relative_to(
                BASELINE_OPERATOR_JSON.parents[3]
            )),
            "artifact_sha256": phase2.sha256(BASELINE_OPERATOR_JSON),
            **{
                key: value for key, value in baseline_operator.items()
                if key not in ("over_bar_by_step", "full_state_snapshots")
            },
        },
        "continuous_trajectory": {
            "metric": "max(abs(legoesm-NEMO))/max(1,max(abs(NEMO)))",
            "sample_steps": sample_table,
            "per_step": continuous_per_step,
        },
        "continuous_growth_table_before_after": before_after,
        "phenomenology": {
            "definition": (
                "hourly positive-concentration extrema; daily EXIT onset is the first "
                "day beginning seven consecutive signed thickness changes"
            ),
            "nemo": nemo_phen, "legoesm_fp64": fp64_phen,
            "legoesm_fp32": fp32_phen,
            "private_arm_no_surface_melt": no_surface_phen,
            "private_arm_no_basal_melt": no_basal_phen,
            "before_legoesm_fp64": old_phen,
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
    parser.add_argument("--plant-branch-census", action="store_true")
    parser.add_argument("--plant-deposition", action="store_true")
    parser.add_argument("--plant-surface", action="store_true")
    parser.add_argument("--plant-snow-temperature", action="store_true")
    parser.add_argument("--plant-operator-trajectory", action="store_true")
    args = parser.parse_args()
    result = run(root=args.root, plant_arithmetic=args.plant_arithmetic,
                 plant_truncate=args.plant_truncate,
                 plant_branch_census=args.plant_branch_census,
                 plant_deposition=args.plant_deposition,
                 plant_surface=args.plant_surface,
                 plant_snow_temperature=args.plant_snow_temperature,
                 plant_operator_trajectory=args.plant_operator_trajectory)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(rendered + "\n")
    print(rendered)
    if result["status"] != "AT-BAR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
