#!/usr/bin/env python3
"""Arithmetic replay and full-year SI3 column gate for C1D_OMIP_L3.

The operator sweep restarts from each NEMO ENTRY frame.  The trajectory sweep
starts once and consumes all 8,760 NEMO-written ZDF input frames.  They answer
different questions and are deliberately reported separately.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nemo_si3thd_phase2_gate as phase2


BAR = phase2.BAR
REPLAY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l3/"
    "c1d_omip_l3_sasice_scalarmath_v2_a"
)
EXPANDED_ZDF_SHA256 = "aad46579fb2d2cc19299d7a25992802525603bb9858adf1e892ff5d85bf40442"
REASSOC_SHA256 = "4b832b0c274d6aab032f16958224ebfb6fea603af22e1a1f1d474ff71b1e4589"
ROUND4_ZDF_INPUT_SHA256 = "cd1b15c821f19442a840e99c067c640e5146b754fc137a2c81e88856d6ea7efd"
DH_OPERAND_SHA256 = "9efbcb9113c2748f9085c519092848596daf0f573ba8a4625d02ee3c08bc8b14"
DH_REMAP_SHA256 = "8fbeb7df70b3c66b4e7acdd7ab0df3ed8fc01bfaa6150dbc47e3b444638b40a5"
REPORT_STEPS = (1, 10, 100, 1000, 3000, 5000, 8760)
CONTINUOUS_FIELDS = ("t_su", "e_i", "e_s", "h_i", "h_s", "a_i", "sv_i")

# The seven l3thd year-gate JSONs (8-26 MB each) are runtime evidence, not
# source -- they live on disk here, not in git (docs/ocean/fidelity/testcases/
# keeps only a .gitignore rule naming them). LEGOESM_NEMO_EVIDENCE_ROOT lets a
# caller point at a different copy (e.g. an empty dir, to exercise the
# absent-file path); the default is this host's evidence store.
EVIDENCE_ROOT_ENV = "LEGOESM_NEMO_EVIDENCE_ROOT"
DEFAULT_EVIDENCE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/evidence"
)


def evidence_path(basename: str) -> Path:
    """Resolve a large SI3 l3thd year-gate JSON in the evidence store."""
    root = Path(os.environ.get(EVIDENCE_ROOT_ENV, str(DEFAULT_EVIDENCE_ROOT)))
    return root / "si3thd" / basename


BASELINE_JSON = evidence_path("nemo_testcases_l3thd_phase2b_year_gate.json")
BASELINE_JSON_SHA256 = "6c21d14f3c85d0fa99be7e31770a4a4dcad548d98f857ea78455bd6344622160"
BASELINE_OPERATOR_JSON = evidence_path(
    "nemo_testcases_l3thd_phase3_baseline_operator.json"
)
BASELINE_OPERATOR_JSON_SHA256 = (
    "d642f532ed92910b49f29919dd5e4ae3fe84381ecd1536b7886a2c1f4b36f95f"
)
PHASE3_JSON = evidence_path("nemo_testcases_l3thd_phase3_year_gate.json")
PHASE3_JSON_SHA256 = "77c78a816b254484afe06632d77378336ddac1153ef9190d17c555f41c45a6e1"
PHASE5_JSON = evidence_path("nemo_testcases_l3thd_phase5_year_gate.json")
PHASE5_JSON_SHA256 = "d5cd3dc274687b56370075ea51b9df82cc01bf912727e25f2078b6c26fe95cc7"

DH_OPERAND_REGISTRY = {
    0: ("INITIALIZED", "after segment initialization, before snowfall; "
        "zdeltah/zevap are explicit unavailable zero sentinels",
        "icethd_dh.F90:127-145"),
    1: ("POST_PRECIP", "after snowfall, before sublimation/deposition; "
        "zdeltah/zevap are explicit unavailable zero sentinels",
        "icethd_dh.F90:166-177"),
    2: ("POST_SUBLIMATION", "after sequential snow sublimation/deposition",
        "icethd_dh.F90:179-202"),
    3: ("POST_SNOW_MELT", "after snow-first surface melt",
        "icethd_dh.F90:204-225"),
    4: ("POST_ICE_SURFACE", "after ice surface melt and sublimation",
        "icethd_dh.F90:231-315"),
    5: ("POST_BASAL", "after basal growth/melt",
        "icethd_dh.F90:321-424"),
    6: ("POST_NO_ICE", "after complete-ice-loss snow removal",
        "icethd_dh.F90:426-439"),
    7: ("POST_FLOOD", "after basal-up snow-ice conversion",
        "icethd_dh.F90:441-485"),
    8: ("SNW_ENT", "inside snw_ent after cumulative remap",
        "icethd_dh.F90:535-613"),
    9: ("POST_SNW_ENT", "after snw_ent, before snow-temperature inverse",
        "icethd_dh.F90:494-507"),
}


def _ulp_distance(left: float, right: float) -> int:
    """ULP distance for finite, same-sign binary64 operands."""

    x, y = np.float64(left), np.float64(right)
    phase2.require(np.isfinite(x) and np.isfinite(y), "ULP nonfinite")
    phase2.require(np.signbit(x) == np.signbit(y), "ULP sign mismatch")
    return abs(int(x.view(np.int64)) - int(y.view(np.int64)))


def _read_dh_operands(path: Path) -> dict[int, list[dict[str, object]]]:
    """Read the fixed C1D DH branch operands and fail closed on their order."""

    grouped: dict[int, list[dict[str, object]]] = {}
    with path.open("rb") as stream:
        while magic := stream.read(16):
            phase2.require(magic == b"NEMO_L3DHO_001  ", "DH-operand magic")
            raw = stream.read(24)
            phase2.require(len(raw) == 24, "truncated DH-operand header")
            version, step, stage, point, bits, nval = struct.unpack("=6i", raw)
            phase2.require(
                (version, point, bits, nval) == (1, 1, 64, 28)
                and stage in DH_OPERAND_REGISTRY,
                f"DH-operand header {(version, step, stage, point, bits, nval)}",
            )
            values = np.fromfile(stream, np.float64, nval)
            phase2.require(values.size == nval and np.all(np.isfinite(values)),
                           "bad DH-operand payload")
            grouped.setdefault(step, []).append({
                "stage": stage,
                "name": DH_OPERAND_REGISTRY[stage][0],
                "time_level": DH_OPERAND_REGISTRY[stage][1],
                "source": DH_OPERAND_REGISTRY[stage][2],
                "h_i": values[0], "h_s": values[1],
                "zdeltah": values[2], "zevap_rema": values[3],
                "zq_top": values[4], "dh_snowice": values[5],
                "zh_s": values[6:10], "ze_s": values[10:14],
                "zh_i": values[14:19], "zh_i_old": values[19:24],
                "sum_zh_s": values[24], "sum_zh_s_ze_s": values[25],
                "sprecip": values[26], "evaporation": values[27],
            })
    phase2.require(set(grouped) == {4242, 5734},
                   f"DH-operand steps {sorted(grouped)}")
    for step, frames in grouped.items():
        phase2.require([frame["stage"] for frame in frames]
                       == [0, 1, 2, 3, 4, 5, 6, 7, 9],
                       f"step {step}: DH-operand order")
    return grouped


def _read_dh_remap_operands(path: Path) -> dict[int, dict[str, object]]:
    """Read the two registered `snw_ent` cumulative-remap records."""

    result = {}
    record_order = []
    with path.open("rb") as stream:
        while magic := stream.read(16):
            phase2.require(magic == b"NEMO_L3DHR_001  ", "DH-remap magic")
            raw = stream.read(20)
            phase2.require(len(raw) == 20, "truncated DH-remap header")
            version, step, stage, bits, nval = struct.unpack("=5i", raw)
            phase2.require((version, stage, bits, nval) == (1, 8, 64, 30),
                           f"DH-remap header {(version, step, stage, bits, nval)}")
            values = np.fromfile(stream, np.float64, nval)
            phase2.require(values.size == nval and np.all(np.isfinite(values)),
                           "bad DH-remap payload")
            phase2.require(step not in result,
                           f"duplicate DH-remap step {step}")
            record_order.append(step)
            result[step] = {
                "stage": stage, "name": DH_OPERAND_REGISTRY[stage][0],
                "time_level": DH_OPERAND_REGISTRY[stage][1],
                "source": DH_OPERAND_REGISTRY[stage][2],
                "zh_s": values[0:4], "ze_s": values[4:8],
                "zhnew": values[8], "zh_cum0": values[9:14],
                "zeh_cum0": values[14:19], "zh_cum1": values[19:23],
                "zeh_cum1": values[23:27], "e_s": values[27:30],
            }
        phase2.require(stream.read(1) == b"", "trailing DH-remap bytes")
    phase2.require(record_order == [4242, 5734],
                   f"DH-remap record order {record_order}")
    return result


def _snow_enthalpy_remap_replay(thickness, enthalpy) -> dict[str, np.ndarray | float]:
    """Scalar binary64 replay of NEMO `snw_ent` in written order."""

    thickness = np.asarray(thickness, dtype=np.float64)
    enthalpy = np.asarray(enthalpy, dtype=np.float64)
    zh_cum0 = np.zeros(5, dtype=np.float64)
    zeh_cum0 = np.zeros(5, dtype=np.float64)
    for index in range(1, 5):
        zeh_cum0[index] = zeh_cum0[index - 1] + (
            enthalpy[index - 1] * thickness[index - 1]
        )
        zh_cum0[index] = zh_cum0[index - 1] + thickness[index - 1]
    zhnew = np.float64(np.sum(thickness) * np.float64(1.0 / 3.0))
    zh_cum1 = np.zeros(4, dtype=np.float64)
    for index in range(1, 4):
        zh_cum1[index] = zh_cum1[index - 1] + zhnew
    zeh_cum1 = np.zeros(4, dtype=np.float64)
    for old in range(1, 5):
        for new in range(1, 3):
            if zh_cum1[new] <= zh_cum0[old] and zh_cum1[new] > zh_cum0[old - 1]:
                zeh_cum1[new] = (
                    zeh_cum0[old - 1] * (zh_cum0[old] - zh_cum1[new])
                    + zeh_cum0[old] * (zh_cum1[new] - zh_cum0[old - 1])
                ) / (zh_cum0[old] - zh_cum0[old - 1])
    zeh_cum1[3] = zeh_cum0[4]
    remapped = np.empty(3, dtype=np.float64)
    for index in range(1, 4):
        remapped[index - 1] = max(
            np.float64(0.0), zeh_cum1[index] - zeh_cum1[index - 1]
        ) / max(zhnew, np.float64(1.0e-20))
    return {
        "zh_cum0": zh_cum0, "zeh_cum0": zeh_cum0,
        "zhnew": zhnew, "zh_cum1": zh_cum1,
        "zeh_cum1": zeh_cum1, "e_s": remapped,
    }


def _snow_sublimation_replay(frame, rho_snow: float, dt: float) -> dict[str, object]:
    """Scalar replay of `icethd_dh.F90:179-202` in written order."""

    rho = np.float64(rho_snow)
    timestep = np.float64(dt)
    inverse_rho = np.float64(1.0) / rho
    evaporation = np.float64(frame["evaporation"])
    h_s = np.float64(frame["h_s"])
    zh_s = np.array(frame["zh_s"], dtype=np.float64, copy=True)
    zdeltah = max(
        np.float64(-evaporation * inverse_rho * timestep),
        np.float64(-h_s),
    )
    zevap_rema = np.float64(evaporation * timestep + zdeltah * rho)
    for layer in range(4):
        delta = max(np.float64(-zh_s[layer]), zdeltah)
        h_s = max(np.float64(0.0), np.float64(h_s + delta))
        zh_s[layer] = max(
            np.float64(0.0), np.float64(zh_s[layer] + delta)
        )
        zdeltah = min(np.float64(zdeltah - delta), np.float64(0.0))
    return {
        "h_s": h_s, "zh_s": zh_s, "zdeltah": zdeltah,
        "zevap_rema": zevap_rema,
    }


def _dh_owner_evidence(root: Path, card, *, plant=False,
                       plant_bridge=False) -> dict[str, object]:
    """Own the preregistered kt5734 DH interaction and retain its controls."""

    from legoesm.ice.bitz_lipscomb import (
        _piecewise_remap,
        si3_column_step_arrays,
    )

    operand_path = root / "oracle_si3_dh_operands.bin"
    remap_path = root / "oracle_si3_dh_remap_operands.bin"
    phase2.require(phase2.sha256(operand_path) == DH_OPERAND_SHA256,
                   "DH-operand SHA256")
    phase2.require(phase2.sha256(remap_path) == DH_REMAP_SHA256,
                   "DH-remap SHA256")
    operands = _read_dh_operands(operand_path)
    remaps = _read_dh_remap_operands(remap_path)

    replay_rows = {}
    for step in (4242, 5734):
        post_precip = next(row for row in operands[step] if row["stage"] == 1)
        post_sublimation = next(
            row for row in operands[step] if row["stage"] == 2
        )
        replay = _snow_sublimation_replay(
            post_precip, card.config.ice_constants.rho_snow, card.dt_seconds
        )
        sublimation_ulps = {
            "h_s": _ulp_distance(replay["h_s"], post_sublimation["h_s"]),
            "zdeltah": _ulp_distance(
                replay["zdeltah"], post_sublimation["zdeltah"]
            ),
            "zevap_rema": _ulp_distance(
                replay["zevap_rema"], post_sublimation["zevap_rema"]
            ),
            "zh_s": [
                _ulp_distance(left, right)
                for left, right in zip(
                    replay["zh_s"], post_sublimation["zh_s"], strict=True
                )
            ],
        }
        phase2.require(
            max(sublimation_ulps.values(), key=lambda value: (
                max(value) if isinstance(value, list) else value
            )) is not None,
            "sublimation ULP registry",
        )
        phase2.require(
            max(
                max(value) if isinstance(value, list) else value
                for value in sublimation_ulps.values()
            ) <= 2,
            f"step {step}: NEMO sublimation replay exceeds 2 ULP",
        )
        remap_replay = _snow_enthalpy_remap_replay(
            remaps[step]["zh_s"], remaps[step]["ze_s"]
        )
        remap_ulps = [
            _ulp_distance(left, right)
            for left, right in zip(
                remap_replay["e_s"], remaps[step]["e_s"], strict=True
            )
        ]
        phase2.require(max(remap_ulps) <= 2,
                       f"step {step}: snw_ent replay exceeds 2 ULP")
        replay_rows[str(step)] = {
            "sublimation_max_ulp": max(
                max(value) if isinstance(value, list) else value
                for value in sublimation_ulps.values()
            ),
            "sublimation_ulps": sublimation_ulps,
            "post_sublimation_nemo": {
                name: (value.tolist() if isinstance(value, np.ndarray)
                       else float(value))
                for name, value in replay.items()
            },
            "remap_max_ulp": max(remap_ulps),
            "remap_ulps": remap_ulps,
            "remapped_e_s": np.asarray(remap_replay["e_s"]).tolist(),
        }

    selected = _read_exact_steps(
        root, (4241, 4242, 4243, 5733, 5734, 5735)
    )
    for step in (4242, 5734):
        initialized = next(
            row for row in operands[step] if row["stage"] == 0
        )
        _validate_exact_entry_bridge(
            selected[step][0], initialized,
            plant=plant_bridge and step == 5734,
        )

    def trace(step, **kwargs):
        frames, exact = selected[step]
        state = jax.tree.map(
            lambda value: jnp.asarray(value, dtype=jnp.float64),
            _exact_1d_entry(frames),
        )
        return jax.device_get(si3_column_step_arrays(
            state, _forcing(exact, jnp.float64), card.dt_seconds,
            card.config.ice_constants, **kwargs,
        ))

    enabled = trace(5734)
    legacy = trace(
        5734, _nemo_snow_sublimation_order=False,
        _nemo_snow_remap=False,
    )
    sublimation_only = trace(5734, _nemo_snow_remap=False)
    remap_only = trace(5734, _nemo_snow_sublimation_order=False)
    oracle = selected[5734][0][2]
    enabled_row = phase2._selected_from_arrays(enabled.post_dh)
    legacy_row = phase2._selected_from_arrays(legacy.post_dh)
    sublimation_only_row = phase2._selected_from_arrays(
        sublimation_only.post_dh
    )
    remap_only_row = phase2._selected_from_arrays(remap_only.post_dh)
    if plant:
        enabled_row = legacy_row
    row = _arm_row(
        oracle["e_s"], legacy_row["e_s"], enabled_row["e_s"], "e_s"
    )
    phase2.require(row["improvement_factor"] >= 100.0,
                   "DH sublimation/remap arm failed 100-fold discriminator")
    phase2.require(
        np.array_equal(sublimation_only_row["e_s"], legacy_row["e_s"])
        and np.array_equal(remap_only_row["e_s"], legacy_row["e_s"]),
        "DH single-operation arms must be final-output inert",
    )
    phase2.require(
        all(np.array_equal(left[name], right[name])
            for left, right in ((
                phase2._selected_from_arrays(enabled.post_zdf),
                phase2._selected_from_arrays(legacy.post_zdf),
            ),)
            for name in phase2.SELECTED_NAMES),
        "DH arm changed an upstream boundary",
    )
    enabled_4242 = phase2._selected_from_arrays(trace(4242).post_dh)
    legacy_4242 = phase2._selected_from_arrays(trace(
        4242, _nemo_snow_sublimation_order=False,
        _nemo_snow_remap=False,
    ).post_dh)
    phase2.require(np.array_equal(enabled_4242["h_i"], legacy_4242["h_i"]),
                   "kt4242 isolation from snow-remap arm")

    scaling = []
    residual = np.asarray(remaps[5734]["zh_s"])
    enthalpy = np.asarray(remaps[5734]["ze_s"])
    for scale in (1.0, 0.5, 0.25):
        scaled = residual * np.float64(scale)
        nemo = np.asarray(_snow_enthalpy_remap_replay(scaled, enthalpy)["e_s"])
        legacy_scaled = np.asarray(_piecewise_remap(
            jnp.asarray(scaled), jnp.asarray(enthalpy), jnp.asarray(0.0)
        ))
        scaling.append({
            "residual_scale": scale,
            "residual_thickness_m": float(np.sum(scaled)),
            "nemo_e_s_J_m3": nemo.tolist(),
            "legacy_e_s_J_m3": legacy_scaled.tolist(),
            "absolute_error_J_m3": float(np.max(np.abs(nemo - legacy_scaled))),
        })

    return {
        "verdict": "CONFIRMED two-operation interaction",
        "first_different_branch": {
            "step": 5734,
            "stage": "POST_SUBLIMATION",
            "nemo_condition_and_update": (
                "zdeltah=max(-evap/rhos*dt,-h_s); segment delta=max(-zh_s,zdeltah)"
            ),
            "legoesm_legacy_condition_and_update": (
                "remove=min(remaining_mass/rho_snow,segment thickness)"
            ),
            "nemo_residual_segment_m": float(remaps[5734]["zh_s"][-1]),
            "legoesm_legacy_residual_segment_m": 0.0,
            "source": "icethd_dh.F90:179-202",
        },
        "replays": replay_rows,
        "private_arms": {
            "combined": row,
            "sublimation_only_final_e_s": sublimation_only_row["e_s"].tolist(),
            "remap_only_final_e_s": remap_only_row["e_s"].tolist(),
            "legacy_final_e_s": legacy_row["e_s"].tolist(),
            "kt4242_h_i_bit_identical": True,
        },
        "scaling_before_owner": scaling,
        "operand_registry": {
            str(stage): {"name": row[0], "time_level": row[1], "source": row[2]}
            for stage, row in DH_OPERAND_REGISTRY.items()
        },
        "boundary_snapshot_steps": [4241, 4242, 4243, 5733, 5734, 5735],
        "nemo_switch": "NONE; source operations are unconditional",
    }


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
                _exact_1d_entry(thermo),
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


def _exact_1d_entry(frames):
    """Bridge NEMO's unchanged registered 1-D thickness operands."""

    entry = phase2._entry_arrays(frames[0])
    return entry._replace(
        h_ice=np.asarray(frames[1]["h_i"], dtype=np.float64),
        h_snow=np.asarray(frames[1]["h_s"], dtype=np.float64),
    )


def _validate_exact_entry_bridge(frames, dh_initialized, *, plant=False) -> None:
    """Plantable kt5734 check that quotient reconstruction is not exact."""

    bridged = _exact_1d_entry(frames)
    if plant:
        bridged = phase2._entry_arrays(frames[0])
    phase2.require(
        np.array_equal(np.asarray(bridged.h_snow),
                       np.atleast_1d(dh_initialized["h_s"]))
        and np.array_equal(np.asarray(bridged.h_ice),
                           np.atleast_1d(dh_initialized["h_i"])),
        "exact-entry 1-D thickness bridge",
    )


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
                entries.append(_exact_1d_entry(frames))
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
                                "normalisation_denominator": scale,
                                "normalised_max_abs": normalised,
                                "oracle_exactly_zero": bool(np.all(wanted == 0.0)),
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


SCOPE_ARM_KWARGS = {
    "zdf_no_snow_melting_row_ranges": {"_zdf_branch_ranges": False},
    "basal_layer_loop": {"_nemo_basal_layer_loop": False},
    "snow_ice_salinity": {"_snow_ice_salinity": False},
    "eos_operation_order": {"_nemo_eos_order": False},
}


def _array_max_ulp(left: np.ndarray, right: np.ndarray) -> int | None:
    """Return maximum same-sign finite binary64 ULP distance, else ``None``."""

    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if (not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
            or np.any(np.signbit(left) != np.signbit(right))):
        return None
    return int(np.max(np.abs(
        left.view(np.int64).astype(object) - right.view(np.int64).astype(object)
    )))


def _scope_ablation_sweep(thd_path: Path, zin_path: Path, card) -> dict[str, object]:
    """Measure each formerly bundled operation as a private one-variable arm."""

    from legoesm.ice.bitz_lipscomb import SI3SurfaceForcing, si3_column_step_arrays

    summaries = {
        name: {
            "private_hook": next(iter(kwargs)),
            "field_rows_compared": 0,
            "changed_field_rows": 0,
            "unchanged_field_rows": 0,
            "disabled_over_bar_field_rows": 0,
            "enabled_over_bar_field_rows": 0,
            "disabled_maximum_normalised_error": 0.0,
            "first_changed_row": None,
            "largest_disabled_outlier": None,
            "changed_by_sub_call": Counter(),
            "changed_by_variable": Counter(),
            "changed_by_step": Counter(),
            "maximum_enabled_disabled_ulp": 0,
            "non_ulp_comparable_rows": 0,
            "zdf_surface_branch_splits": 0,
            "enabled_closer_rows": 0,
            "disabled_closer_rows": 0,
            "equal_oracle_error_rows": 0,
            "largest_enabled_improvement": None,
        }
        for name, kwargs in SCOPE_ARM_KWARGS.items()
    }
    chunk_size = 1000
    with thd_path.open("rb") as thd, zin_path.open("rb") as zin:
        for first_step in range(1, card.nsteps + 1, chunk_size):
            last_step = min(card.nsteps, first_step + chunk_size - 1)
            chunk_frames, entries, exact_inputs = [], [], []
            for kt in range(first_step, last_step + 1):
                chunk_frames.append(phase2._read_thd_step(thd, kt))
                exact_inputs.append(phase2._read_zdf_input_step(zin, kt))
                entries.append(_exact_1d_entry(chunk_frames[-1]))
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
            enabled_maps = _stage_maps(jax.device_get(si3_column_step_arrays(
                batch_entry, batch_forcing, card.dt_seconds,
                card.config.ice_constants,
            )))
            disabled_maps = {
                arm_name: _stage_maps(jax.device_get(si3_column_step_arrays(
                    batch_entry, batch_forcing, card.dt_seconds,
                    card.config.ice_constants, **kwargs,
                )))
                for arm_name, kwargs in SCOPE_ARM_KWARGS.items()
            }
            for offset, frames in enumerate(chunk_frames):
                kt = first_step + offset
                exact_input = exact_inputs[offset]
                for stage_index, (oracle_frame, enabled) in enumerate(
                    zip(frames, enabled_maps, strict=True)
                ):
                    oracle = (oracle_frame if 1 <= stage_index <= 5
                              else phase2._center_global(oracle_frame))
                    names = (phase2.SELECTED_NAMES if 1 <= stage_index <= 5
                             else phase2.GLOBAL_NAMES)
                    for field_name in names:
                        wanted = np.asarray(oracle[field_name])
                        enabled_value = np.asarray(
                            enabled[field_name][offset:offset + 1]
                        )
                        if enabled_value.ndim == 2 and wanted.ndim == 1:
                            wanted = wanted[None, :]
                        if wanted.shape != enabled_value.shape:
                            phase2.require(
                                wanted.size == enabled_value.size,
                                "scope oracle/enabled element count",
                            )
                            wanted = wanted.reshape(enabled_value.shape)
                        for arm_name, maps in disabled_maps.items():
                            result = summaries[arm_name]
                            disabled_value = np.asarray(
                                maps[stage_index][field_name][offset:offset + 1]
                            )
                            phase2.require(
                                wanted.shape == disabled_value.shape,
                                f"scope metric shape {arm_name} kt{kt} "
                                f"{phase2.STAGES[stage_index][1]}.{field_name}: "
                                f"{wanted.shape} != {disabled_value.shape}",
                            )
                            result["field_rows_compared"] += 1
                            absolute, normalised = _normalised_error(
                                wanted, disabled_value
                            )
                            enabled_abs, enabled_norm = _normalised_error(
                                wanted, enabled_value
                            )
                            if normalised > BAR:
                                result["disabled_over_bar_field_rows"] += 1
                            if enabled_norm > BAR:
                                result["enabled_over_bar_field_rows"] += 1
                            if enabled_norm < normalised:
                                result["enabled_closer_rows"] += 1
                            elif normalised < enabled_norm:
                                result["disabled_closer_rows"] += 1
                            else:
                                result["equal_oracle_error_rows"] += 1
                            result["disabled_maximum_normalised_error"] = max(
                                result["disabled_maximum_normalised_error"],
                                normalised,
                            )
                            disabled_outlier = {
                                "step": kt,
                                "sub_call": phase2.STAGES[stage_index][1],
                                "variable": field_name,
                                "absolute_numerator": absolute,
                                "normalisation_denominator": max(
                                    1.0, float(np.max(np.abs(wanted)))
                                ),
                                "normalised_quotient": normalised,
                                "branch": _branch_label(
                                    stage_index, frames, exact_input,
                                    card.config.ice_constants.T0,
                                ),
                            }
                            largest = result["largest_disabled_outlier"]
                            if (largest is None or normalised
                                    > largest["normalised_quotient"]):
                                result["largest_disabled_outlier"] = disabled_outlier
                            changed = not np.array_equal(
                                enabled_value, disabled_value
                            )
                            if changed:
                                result["changed_field_rows"] += 1
                                stage_name = phase2.STAGES[stage_index][1]
                                result["changed_by_sub_call"][stage_name] += 1
                                result["changed_by_variable"][field_name] += 1
                                result["changed_by_step"][str(kt)] += 1
                                delta = float(np.max(np.abs(
                                    disabled_value - enabled_value
                                )))
                                improvement = normalised - enabled_norm
                                best = result["largest_enabled_improvement"]
                                if best is None or improvement > best["normalised_gain"]:
                                    result["largest_enabled_improvement"] = {
                                        **disabled_outlier,
                                        "enabled_normalised_max_abs": enabled_norm,
                                        "normalised_gain": improvement,
                                    }
                                if result["first_changed_row"] is None:
                                    result["first_changed_row"] = {
                                        **disabled_outlier,
                                        "enabled_absolute_max": enabled_abs,
                                        "enabled_normalised_max_abs": enabled_norm,
                                        "enabled_disabled_absolute_max": delta,
                                        "disabled_enabled_improvement_factor": (
                                            absolute / max(
                                                enabled_abs,
                                                np.finfo(np.float64).eps,
                                            )
                                        ),
                                    }
                                ulp = _array_max_ulp(enabled_value, disabled_value)
                                if ulp is None:
                                    result["non_ulp_comparable_rows"] += 1
                                else:
                                    result["maximum_enabled_disabled_ulp"] = max(
                                        result["maximum_enabled_disabled_ulp"], ulp
                                    )
                            else:
                                result["unchanged_field_rows"] += 1
                    if stage_index == 1:
                        enabled_melt = bool(
                            np.asarray(enabled["t_su"])[offset] >=
                            card.config.ice_constants.T0
                        )
                        for arm_name, maps in disabled_maps.items():
                            disabled_melt = bool(
                                np.asarray(maps[stage_index]["t_su"])[offset] >=
                                card.config.ice_constants.T0
                            )
                            summaries[arm_name]["zdf_surface_branch_splits"] += int(
                                enabled_melt != disabled_melt
                            )
        phase2.require(thd.read(1) == b"", "scope sweep thermo cursor")
        phase2.require(zin.read(1) == b"", "scope sweep input cursor")

    for result in summaries.values():
        for key in ("changed_by_sub_call", "changed_by_variable", "changed_by_step"):
            result[key] = dict(sorted(result[key].items(), key=lambda item: item[0]))
    return summaries


def _validate_scope_ablation_accounting(scope, *, plant=False) -> None:
    """Plantable completeness check for all four preregistered scope arms."""

    expected_names = set(SCOPE_ARM_KWARGS)
    phase2.require(set(scope) == expected_names, "scope-arm register")
    expected_rows = next(iter(scope.values()))["field_rows_compared"]
    phase2.require(expected_rows > 0, "scope-arm empty sweep")
    for index, result in enumerate(scope.values()):
        changed = result["changed_field_rows"] - int(plant and index == 0)
        phase2.require(
            result["field_rows_compared"] == expected_rows
            and changed + result["unchanged_field_rows"] == expected_rows,
            "scope-arm row accounting",
        )


def _operator_debt_histogram(operator) -> dict[str, object]:
    """Partition exact-entry debt, including the denominator-one zero class."""

    rows = [
        {"step": int(step), **row}
        for step, step_rows in operator["over_bar_by_step"].items()
        for row in step_rows
    ]
    zero_rows, genuine = [], []
    for row in rows:
        target = (zero_rows if row["normalisation_denominator"] == 1.0
                  and row["oracle_exactly_zero"] else genuine)
        target.append(row)
    largest_genuine = max(genuine, key=lambda row: row["normalised_max_abs"])
    by_step = Counter(str(row["step"]) for row in rows)
    frequency = Counter(str(count) for count in by_step.values())
    return {
        "all_over_bar_rows": len(rows),
        "by_sub_call": dict(sorted(Counter(
            row["sub_call"] for row in rows
        ).items())),
        "by_variable": dict(sorted(Counter(
            row["variable"] for row in rows
        ).items())),
        "by_step": dict(sorted(by_step.items(), key=lambda item: int(item[0]))),
        "rows_per_step_frequency": dict(sorted(
            frequency.items(), key=lambda item: int(item[0])
        )),
        "steps_with_over_bar_rows": len(by_step),
        "denominator_one_oracle_exactly_zero_rows": len(zero_rows),
        "genuine_relative_rows": len(genuine),
        "largest_genuine_relative_row": largest_genuine,
    }


def _continuous_jump_attribution(continuous, operator) -> dict[str, object]:
    """Relate the conspicuous continuous jumps to same-step exact-entry debt."""

    # These rows are post-hoc diagnostics selected from the committed complete
    # trajectory; they are not promoted to preregistered owners.
    selected = ((4239, "e_i"), (4239, "h_i"), (4943, "h_s"),
                (5495, "t_su"), (5850, "t_su"))
    rows = []
    for step, variable in selected:
        previous = continuous[step - 2]["fields"][variable]["normalised_linf"]
        current = continuous[step - 1]["fields"][variable]["normalised_linf"]
        exact_step = operator["per_step"][step - 1]
        same_variable = [
            row for row in operator["over_bar_by_step"].get(str(step), [])
            if row["variable"] == variable
        ]
        rows.append({
            "step": step,
            "variable": variable,
            "continuous_previous_normalised_linf": previous,
            "continuous_current_normalised_linf": current,
            "exact_entry_step_maximum_normalised_error": (
                exact_step["maximum_normalised_error"]
            ),
            "exact_entry_step_owner": exact_step["owner"],
            "same_variable_over_bar_rows": same_variable,
            "exact_entry_step_is_2e-15_class": (
                exact_step["maximum_normalised_error"] <= 2.0e-15
            ),
        })
    initial_noise_steps = [4239, 5495]
    noise_result = all(
        operator["per_step"][step - 1]["maximum_normalised_error"] <= 2.0e-15
        for step in initial_noise_steps
    )
    material = {
        "first_above_1e-12": operator["first_above_1e-12"],
        "first_above_1e-3": operator["first_above_1e-3"],
    }
    return {
        "selection_status": "POST-HOC from the complete retained trajectory",
        "rows": rows,
        "initial_threshold_amplification_from_2e-15_class_steps": noise_result,
        "terminal_classification": (
            "MIXED DEBT: the kt4239 and kt5495 continuous jumps amplify "
            "2e-15-class exact-entry differences, but kt4242 remains the first "
            "injection above 1e-12 and subnormal-snow ZDF rows exceed 1e-3 "
            "from kt5842; the whole remaining trajectory cannot be classified "
            "as summation-order noise"
        ),
        "material_exact_entry_debt": material,
        "next_discriminator": (
            "bit-exact whole-step NEMO summation order is required to remove "
            "the threshold-amplified component; kt4242 needs a separate DH "
            "ice-thickness operand replay, and kt5842 needs a ZDF discriminator "
            "for NEMO-positive subnormal snow versus JAX/XLA arithmetic"
        ),
    }


def _validate_operator_trajectory(operator, *, plant=False,
                                  plant_outlier=False) -> None:
    """Fail closed on the complete exact-entry trajectory and max attribution."""

    rows = list(operator["per_step"])
    if plant:
        rows.pop()
    phase2.require(len(rows) == 8760, "oracle-entry per-step count")
    phase2.require([row["step"] for row in rows] == list(range(1, 8761)),
                   "oracle-entry per-step sequence")
    recomputed = max(rows, key=lambda row: row["maximum_normalised_error"])
    largest = dict(operator["largest_outlier"])
    if plant_outlier:
        largest["normalisation_denominator"] *= 2.0
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


def _historical_largest_outlier(root: Path, baseline_operator) -> dict[str, object]:
    """Attribute the retained pre-Phase-3 1.1e8 exact-entry row."""

    candidates = [
        (row["normalised_max_abs"], int(step), row)
        for step, rows in baseline_operator["over_bar_by_step"].items()
        for row in rows
    ]
    normalised, step, row = max(candidates, key=lambda item: item[0])
    frames, _ = _read_exact_steps(root, (step,))[step]
    stage_index = next(
        index for index, (_, name, _, _) in enumerate(phase2.STAGES)
        if name == row["sub_call"]
    )
    oracle = np.asarray(frames[stage_index][row["variable"]])
    denominator = max(1.0, float(np.max(np.abs(oracle))))
    phase2.require(row["absolute_max"] / denominator == normalised,
                   "historical largest-outlier denominator")
    return {
        "artifact": str(BASELINE_OPERATOR_JSON.relative_to(
            BASELINE_OPERATOR_JSON.parents[3]
        )),
        "artifact_sha256": phase2.sha256(BASELINE_OPERATOR_JSON),
        "step": step,
        "sub_call": row["sub_call"],
        "variable": row["variable"],
        "absolute_numerator": row["absolute_max"],
        "normalisation_denominator": denominator,
        "normalised_quotient": normalised,
        "oracle_value": oracle.reshape(-1).tolist(),
        "owner": (
            "pre-Phase-3 missing DH snow-first surface melt left snow "
            "enthalpy where NEMO had removed all snow"
        ),
        "source": "icethd_dh.F90:107-120,204-315",
    }


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
            _exact_1d_entry(frames),
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
                entries.append(_exact_1d_entry(frames))
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
        plant_operator_trajectory=False,
        plant_outlier_attribution=False,
        plant_scope_accounting=False,
        plant_dh_owner=False,
        plant_entry_bridge=False) -> dict[str, object]:
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ice.bitz_lipscomb import si3_column_step_arrays
    from legoesm.ice.c1d_omip_l3 import (
        FORCING_SHA256, ORACLE_VERSION, THERMO_STREAM_SHA256,
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
    dh_owner = _dh_owner_evidence(
        root, card, plant=plant_dh_owner, plant_bridge=plant_entry_bridge
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
    scope_steps = {
        arm_name: jax.jit(
            lambda state, forcing, arm_kwargs=arm_kwargs:
            si3_column_step_arrays(
                state, forcing, card.dt_seconds, card.config.ice_constants,
                **arm_kwargs,
            )
        )
        for arm_name, arm_kwargs in SCOPE_ARM_KWARGS.items()
    }
    operator = _operator_sweep(
        thd_path, zin_path, card, eager_step, retain_over_bar_rows=True,
        snapshot_steps=(
            73, 74, 75, 76, 4239, 4241, 4242, 4243, 5285,
            5733, 5734, 5735,
        ),
    )
    _validate_step74_branch_snapshot(operator, plant=plant_branch_census)
    _validate_operator_trajectory(
        operator, plant=plant_operator_trajectory,
        plant_outlier=plant_outlier_attribution,
    )
    scope_ablations = _scope_ablation_sweep(thd_path, zin_path, card)
    _validate_scope_ablation_accounting(
        scope_ablations, plant=plant_scope_accounting
    )
    continuous_per_step = []
    sample_table: dict[str, object] = {}
    h_nemo, h_fp64, h_fp32 = [], [], []
    h_no_surface, h_no_basal = [], []
    h_scope = {arm_name: [] for arm_name in SCOPE_ARM_KWARGS}
    scope_continuous = {
        arm_name: {
            "changed_field_steps": 0,
            "unchanged_field_steps": 0,
            "first_changed": None,
            "maximum_enabled_disabled_absolute": 0.0,
            "changed_by_variable": Counter(),
            "sample_steps": {},
        }
        for arm_name in SCOPE_ARM_KWARGS
    }
    state64 = state32 = state_no_surface = state_no_basal = None
    state_scope = {}

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
                state_scope = {arm_name: state64 for arm_name in SCOPE_ARM_KWARGS}
            trace64 = step(state64, forcing64)
            trace32 = step(state32, _forcing(exact_input, jnp.float32))
            trace_no_surface = no_surface_step(state_no_surface, forcing64)
            trace_no_basal = no_basal_step(state_no_basal, forcing64)
            trace_scope = {
                arm_name: arm_step(state_scope[arm_name], forcing64)
                for arm_name, arm_step in scope_steps.items()
            }
            state64, state32 = trace64.exit, trace32.exit
            state_no_surface = trace_no_surface.exit
            state_no_basal = trace_no_basal.exit
            state_scope = {
                arm_name: arm_trace.exit
                for arm_name, arm_trace in trace_scope.items()
            }
            exit64 = phase2._global_from_arrays(jax.device_get(state64))
            exit32 = phase2._global_from_arrays(jax.device_get(state32))
            exit_scope = {
                arm_name: phase2._global_from_arrays(jax.device_get(arm_state))
                for arm_name, arm_state in state_scope.items()
            }
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
            for arm_name, arm_exit in exit_scope.items():
                arm_metrics = _metric_fields(arm_exit)
                h_scope[arm_name].append(float(arm_metrics["h_i"][0]))
                arm_summary = scope_continuous[arm_name]
                sample = {}
                for field_name in CONTINUOUS_FIELDS:
                    delta = float(np.max(np.abs(
                        arm_metrics[field_name] - fp64_metrics[field_name]
                    )))
                    if delta == 0.0:
                        arm_summary["unchanged_field_steps"] += 1
                    else:
                        arm_summary["changed_field_steps"] += 1
                        arm_summary["changed_by_variable"][field_name] += 1
                        arm_summary["maximum_enabled_disabled_absolute"] = max(
                            arm_summary["maximum_enabled_disabled_absolute"], delta
                        )
                        if arm_summary["first_changed"] is None:
                            arm_summary["first_changed"] = {
                                "step": kt, "variable": field_name,
                                "enabled_disabled_absolute": delta,
                            }
                    if kt in REPORT_STEPS:
                        absolute, normalised = _normalised_error(
                            oracle_metrics[field_name], arm_metrics[field_name]
                        )
                        sample[field_name] = {
                            "absolute_max": absolute,
                            "normalised_linf": normalised,
                            "enabled_disabled_absolute": delta,
                        }
                if kt in REPORT_STEPS:
                    arm_summary["sample_steps"][str(kt)] = sample
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
    for arm_name, arm_summary in scope_continuous.items():
        arm_summary["changed_by_variable"] = dict(sorted(
            arm_summary["changed_by_variable"].items()
        ))
        arm_summary["phenomenology"] = _phenology(
            np.asarray(h_scope[arm_name])
        )
    phase2.require(phase2.sha256(BASELINE_JSON) == BASELINE_JSON_SHA256,
                   "Phase-2b baseline JSON SHA256")
    phase2.require(
        phase2.sha256(BASELINE_OPERATOR_JSON) == BASELINE_OPERATOR_JSON_SHA256,
        "Phase-2b retained operator JSON SHA256",
    )
    phase2.require(phase2.sha256(PHASE3_JSON) == PHASE3_JSON_SHA256,
                   "Phase-3 year JSON SHA256")
    phase2.require(phase2.sha256(PHASE5_JSON) == PHASE5_JSON_SHA256,
                   "Phase-5 year JSON SHA256")
    baseline = json.loads(BASELINE_JSON.read_text())
    baseline_operator = json.loads(BASELINE_OPERATOR_JSON.read_text())
    phase3 = json.loads(PHASE3_JSON.read_text())
    phase5 = json.loads(PHASE5_JSON.read_text())
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
        baseline_fields = phase5["continuous_trajectory"]["sample_steps"][key]
        before_after[key] = {
            field: {
                "before_normalised_linf": baseline_fields[field]["normalised_linf"],
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
                 "nsteps": card.nsteps, "scope": "ORCA1-resolved identity only",
                 "oracle_version": ORACLE_VERSION,
                 "oracle_root": str(card.oracle_root)},
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
            "phase3_year_json": phase2.sha256(PHASE3_JSON),
            "phase5_year_json": phase2.sha256(PHASE5_JSON),
            "dh_operands": phase2.sha256(root / "oracle_si3_dh_operands.bin"),
            "dh_remap_operands": phase2.sha256(
                root / "oracle_si3_dh_remap_operands.bin"
            ),
        },
        "arithmetic_replay": arithmetic,
        "owner_arms": owner_arms,
        "round6_dh_owner": dh_owner,
        "round4_bundled_change_ablations": {
            "exact_entry": scope_ablations,
            "continuous": scope_continuous,
            "preregister": (
                "docs/ocean/fidelity/testcases/"
                "nemo_testcases_l3thd_phase5_scope_preregister.md"
            ),
        },
        "melt_candidate_scaling": _melt_scaling(root, card),
        "oracle_entry_operator_sweep": operator,
        "oracle_entry_over_bar_histogram": _operator_debt_histogram(operator),
        "oracle_entry_operator_sweep_before": {
            "artifact": str(PHASE3_JSON.relative_to(
                PHASE3_JSON.parents[3]
            )),
            "artifact_sha256": phase2.sha256(PHASE3_JSON),
            **{
                key: value
                for key, value in phase3["oracle_entry_operator_sweep"].items()
                if key not in ("over_bar_by_step", "full_state_snapshots")
            },
        },
        "historical_1p1e8_outlier": _historical_largest_outlier(
            root, baseline_operator
        ),
        "continuous_trajectory": {
            "metric": "max(abs(legoesm-NEMO))/max(1,max(abs(NEMO)))",
            "sample_steps": sample_table,
            "per_step": continuous_per_step,
        },
        "continuous_jump_attribution": _continuous_jump_attribution(
            continuous_per_step, operator
        ),
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
            "before_legoesm_fp64": phase5["phenomenology"]["legoesm_fp64"],
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
    parser.add_argument("--plant-outlier-attribution", action="store_true")
    parser.add_argument("--plant-scope-accounting", action="store_true")
    parser.add_argument("--plant-dh-owner", action="store_true")
    parser.add_argument("--plant-entry-bridge", action="store_true")
    args = parser.parse_args()
    for evidence_json in (BASELINE_JSON, BASELINE_OPERATOR_JSON, PHASE3_JSON,
                          PHASE5_JSON):
        if not evidence_json.exists():
            print(f"REFUSED: evidence file not present: {evidence_json}")
            raise SystemExit(1)
    result = run(root=args.root, plant_arithmetic=args.plant_arithmetic,
                 plant_truncate=args.plant_truncate,
                 plant_branch_census=args.plant_branch_census,
                 plant_deposition=args.plant_deposition,
                 plant_surface=args.plant_surface,
                 plant_snow_temperature=args.plant_snow_temperature,
                 plant_operator_trajectory=args.plant_operator_trajectory,
                 plant_outlier_attribution=args.plant_outlier_attribution,
                 plant_scope_accounting=args.plant_scope_accounting,
                 plant_dh_owner=args.plant_dh_owner,
                 plant_entry_bridge=args.plant_entry_bridge)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(rendered + "\n")
    print(rendered)
    if result["status"] != "AT-BAR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
