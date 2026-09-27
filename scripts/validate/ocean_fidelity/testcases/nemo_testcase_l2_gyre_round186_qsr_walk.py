#!/usr/bin/env python3
"""Admit the Round-186 developed ``qsr_2BD`` operand/replay record."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


MAGIC = b"NEMO_L2_R186QSR1"
# At step 1080 the persistent three-slot rotation enters stage 3 with Kmm=2
# and Krhs=1 (stprk3.f90's two in-stage swaps plus its end-of-step swap).
HEADER = (1, 1080, 3, 2, 1, 36, 26, 31, 64)
HEADER_INTS = len(HEADER)
JPI, JPJ, JPK = 36, 26, 31
# qsr is allocated on NEMO's no-halo domain (Nis0:Nie0,Njs0:Nje0), while
# r3t and every 3-D field in the same WRITE retain the full local domain.
# With nn_hls=2 that no-halo extent is (36 - 4) by (26 - 4).
NN_HLS = 2
QSR_NI, QSR_NJ = JPI - 2 * NN_HLS, JPJ - 2 * NN_HLS
VALUE_COUNT = JPK + QSR_NI * QSR_NJ + JPI * JPJ + 5 * JPI * JPJ * JPK
RECORD_BYTES = 16 + 4 * HEADER_INTS + 8 * VALUE_COUNT
_HERE = Path(__file__).resolve().parent


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_record(path: Path) -> dict:
    raw = path.read_bytes()
    require(len(raw) == RECORD_BYTES,
            f"{path}: {len(raw)} bytes, expected {RECORD_BYTES}")
    require(raw[:16] == MAGIC, f"{path}: bad magic {raw[:16]!r}")
    header = struct.unpack(f"={HEADER_INTS}i", raw[16:16 + 4 * HEADER_INTS])
    require(header == HEADER, f"{path}: header is {header}, expected {HEADER}")
    values = np.frombuffer(raw, dtype="=f8", offset=16 + 4 * HEADER_INTS)
    require(values.size == VALUE_COUNT, f"{path}: bad value count")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    offset = 0

    def take(count: int, shape: tuple[int, ...], name: str) -> np.ndarray:
        nonlocal offset
        result = values[offset:offset + count].reshape(shape, order="F").copy()
        offset += count
        require(result.size == count, f"{path}: truncated {name}")
        return result

    n2 = JPI * JPJ
    n3 = n2 * JPK
    result = {
        "gdepw_1d": take(JPK, (JPK,), "gdepw_1d"),
        "qsr": take(QSR_NI * QSR_NJ, (QSR_NI, QSR_NJ), "qsr"),
        "r3t_Kmm": take(n2, (JPI, JPJ), "r3t_Kmm"),
        "e3t_3d": take(n3, (JPI, JPJ, JPK), "e3t_3d"),
        "tmask": take(n3, (JPI, JPJ, JPK), "tmask"),
        "wmask": take(n3, (JPI, JPJ, JPK), "wmask"),
        "actual_increment": take(n3, (JPI, JPJ, JPK), "actual_increment"),
        "replay_increment": take(n3, (JPI, JPJ, JPK), "replay_increment"),
    }
    require(offset == VALUE_COUNT, f"{path}: unread values")
    result.update(
        path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
        header=list(header), bytes=len(raw),
        qsr_bounds_1based=[1 + NN_HLS, JPI - NN_HLS,
                           1 + NN_HLS, JPJ - NN_HLS],
    )
    return result


def admit(root: Path, expect_commit: str, plant: str | None = None) -> dict:
    require(plant in (None, "actual-increment-ulp"), f"unknown plant {plant}")
    record_path = root / "oracle_qsr_walk_kt00001080.bin"
    record = read_record(record_path)
    producer = (root / "producer_commit.txt").read_text().strip()
    require(producer == expect_commit,
            f"producer commit {producer}, expected {expect_commit}")
    stamp_words = (root / "qsr_record.stamp").read_text().split()
    require(stamp_words == [record["sha256"], producer, record_path.name],
            "qsr record stamp mismatch")

    actual = record["actual_increment"].copy()
    replay = record["replay_increment"]
    if plant == "actual-increment-ulp":
        actual.flat[0] = np.nextafter(actual.flat[0], np.inf)
    unequal = int(np.count_nonzero(actual.view(np.uint64) != replay.view(np.uint64)))
    max_abs = float(np.max(np.abs(actual - replay)))
    if plant:
        require(unequal > 0, "actual-increment ULP plant did not move replay row")
        raise GateError(f"STATUS PLANT-FIRED: {plant}; unequal={unequal}")
    require(unequal == 0,
            f"post-call qsr replay is not BIT: {unequal} cells, max {max_abs}")
    return {
        "format": "gyre-round186-qsr-walk-admission-v1",
        "status": "PASS",
        "producer_commit": producer,
        "record": {"path": record["path"], "sha256": record["sha256"],
                   "bytes": record["bytes"], "header": record["header"],
                   "qsr_bounds_1based": record["qsr_bounds_1based"]},
        "calibration": {"cells_unequal": unequal, "max_abs_K_s-1": max_abs},
        "worktree": {"commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()},
    }


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / filename)
    require(spec is not None and spec.loader is not None,
            f"cannot load sibling tool {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _score(reference: np.ndarray, candidate: np.ndarray,
           mask: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    require(reference.shape == candidate.shape == mask.shape,
            "shortwave walk score shape mismatch")
    selected_reference = reference[mask]
    selected_candidate = candidate[mask]
    unequal = int(np.count_nonzero(
        selected_reference.view(np.uint64)
        != selected_candidate.view(np.uint64)))
    difference = selected_candidate - selected_reference
    return {
        "cells_scored": int(mask.sum()),
        "cells_unequal": unequal,
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "rms": float(np.sqrt(np.mean(difference * difference))
                     if difference.size else 0.0),
        "classification": "BIT" if unequal == 0 else "NON-BIT",
    }


def _removed_fraction(before: float, after: float) -> float:
    require(before > 0.0, "cannot rank removal from a zero baseline")
    return float((before - after) / before)


def _associate_qsr(bsbc, qmm, qaa, rate):
    """Apply the stage-3 QSR source to an already-associated accumulator."""
    content_before = bsbc * qaa[..., None]
    after = content_before + 14400.0 * qmm[..., None] * rate
    return after / qaa[..., None] - bsbc


def _compiled_direct_rate(record: dict, *, return_rows: bool = False):
    """Replay qsr_2BD through its compiled scalar statement order.

    This deliberately excludes the final ``(Krhs + rate) - Krhs`` update;
    the independently admitted process record supplies that operand below.
    """
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
    from legoesm.ocean.physics.shortwave_penetration import JERLOV_TYPES

    interior = (slice(NN_HLS, -NN_HLS), slice(NN_HLS, -NN_HLS))
    qsr = record["qsr"]
    r3t = record["r3t_Kmm"][interior]
    e3t = record["e3t_3d"][interior]
    tmask = record["tmask"][interior]
    wmask = record["wmask"][interior]
    gdepw = record["gdepw_1d"]
    params = JERLOV_TYPES["I"]
    r1_rho0_rcp = 1.0 / (
        float(NEMO_CONSTANTS_CONFIG.rho_0)
        * float(NEMO_CONSTANTS_CONFIG.c_sw))
    r1_si0 = 1.0 / params.zeta1
    r1_si1 = 1.0 / params.zeta2
    zz0 = params.R * r1_rho0_rcp
    zz1 = (1.0 - params.R) * r1_rho0_rcp
    result = np.zeros((QSR_NI, QSR_NJ, JPK - 1), dtype=np.float64)
    zatt = np.empty((QSR_NI, QSR_NJ), dtype=np.float64)
    surface_depth = np.empty_like(zatt)
    surface_arguments = np.empty(zatt.shape + (2,), dtype=np.float64)
    surface_exponentials = np.empty_like(surface_arguments)
    live_depth = np.empty(zatt.shape + (17,), dtype=np.float64)
    live_arguments = np.empty(zatt.shape + (17, 2), dtype=np.float64)
    live_exponentials = np.empty_like(live_arguments)
    live_attenuation = np.empty_like(live_depth)
    live_ze3t = np.empty_like(live_depth)
    absorbed = np.empty_like(live_depth)
    numerator = np.empty_like(live_depth)
    for i in range(QSR_NI):
        for j in range(QSR_NJ):
            stretch = 1.0 + r3t[i, j]
            surface_depth[i, j] = gdepw[0] * stretch
            surface_arguments[i, j, 0] = -surface_depth[i, j] * r1_si0
            surface_arguments[i, j, 1] = -surface_depth[i, j] * r1_si1
            surface_exponentials[i, j, 0] = math.exp(
                surface_arguments[i, j, 0])
            surface_exponentials[i, j, 1] = math.exp(
                surface_arguments[i, j, 1])
            zatt[i, j] = (
                zz0 * surface_exponentials[i, j, 0]
                + zz1 * surface_exponentials[i, j, 1])
    # qsr_ext_lev on this admitted GYRE card resolves nk0=2 and nkV=17.
    for k in range(17):
        for i in range(QSR_NI):
            for j in range(QSR_NJ):
                stretch = 1.0 + r3t[i, j]
                ze3t = e3t[i, j, k] * (
                    1.0 + r3t[i, j] * tmask[i, j, k])
                live_ze3t[i, j, k] = ze3t
                live_depth[i, j, k] = gdepw[k + 1] * stretch
                live_arguments[i, j, k, 0] = (
                    -live_depth[i, j, k] * r1_si0)
                live_arguments[i, j, k, 1] = (
                    -live_depth[i, j, k] * r1_si1)
                live_exponentials[i, j, k, 0] = math.exp(
                    live_arguments[i, j, k, 0])
                live_exponentials[i, j, k, 1] = math.exp(
                    live_arguments[i, j, k, 1])
                if k < 2:
                    next_attenuation = (
                        zz0 * live_exponentials[i, j, k, 0]
                        + zz1 * live_exponentials[i, j, k, 1]
                    ) * wmask[i, j, k + 1]
                else:
                    next_attenuation = (
                        zz1 * live_exponentials[i, j, k, 1]
                        * wmask[i, j, k + 1])
                live_attenuation[i, j, k] = next_attenuation
                absorbed[i, j, k] = zatt[i, j] - next_attenuation
                numerator[i, j, k] = qsr[i, j] * absorbed[i, j, k]
                result[i, j, k] = numerator[i, j, k] / ze3t
                zatt[i, j] = next_attenuation
    if not return_rows:
        return result
    return result, {
        "coefficients": np.asarray([zz0, zz1]),
        "surface_depth": surface_depth,
        "surface_arguments": surface_arguments,
        "surface_exponentials": surface_exponentials,
        "surface_attenuation": (
            zz0 * surface_exponentials[..., 0]
            + zz1 * surface_exponentials[..., 1]),
        "live_depth": live_depth,
        "live_arguments": live_arguments,
        "live_exponentials": live_exponentials,
        "live_ze3t": live_ze3t,
        "live_attenuation": live_attenuation,
        "absorbed_flux_fraction": absorbed,
        "absorbed_flux_numerator": numerator,
        "direct_rate": result[..., :17],
    }


def walk(root: Path, process_record: Path, lego_trace: Path,
         plant: str | None = None) -> dict:
    """Walk developed qsr_2BD using the admitted record and existing trace."""
    require(plant in (None, "r3t-ulp"), f"unknown walk plant {plant}")
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "shortwave walk is not fp64/libm")
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.physics.shortwave_penetration import (
        ShortwavePenetrationConfig, _nemo_qsr_2bd_tendency)

    record = read_record(root / "oracle_qsr_walk_kt00001080.bin")
    year = _load_sibling(
        "_round188_year_owners", "nemo_testcase_l2_gyre_year_owners.py")
    process = year.read_process_record(process_record)
    require(process["kstp"] == 1080, "process operand is not step 1080")
    before = np.asarray(
        process["rhs_after_surface_boundary"], dtype=np.float64).transpose(
            1, 0, 2)[..., :JPK - 1]
    interior = (slice(NN_HLS, -NN_HLS), slice(NN_HLS, -NN_HLS))
    wet = record["tmask"][interior][..., :JPK - 1] > 0.5
    direct, compiled_rows = _compiled_direct_rate(record, return_rows=True)
    replay = record["replay_increment"][interior][..., :JPK - 1]
    compiled_associated = (before + direct) - before
    require(_score(replay, compiled_associated, wet)["cells_unequal"] == 0,
            "Round-185 Krhs-before does not reproduce Round-186 replay")

    qsr = np.asarray(record["qsr"], dtype=np.float64)
    stretch = 1.0 + np.asarray(
        record["r3t_Kmm"][interior], dtype=np.float64)
    if plant == "r3t-ulp":
        planted = stretch.copy()
        i, j = (int(value) for value in np.argwhere(np.any(wet, axis=-1))[0])
        planted[i, j] = np.nextafter(planted[i, j], np.inf)
        stretch = planted
    dz_ref = np.asarray(record["e3t_3d"][2, 2, :JPK - 1])
    config = ShortwavePenetrationConfig(
        scheme="nemo_qsr_2bd", water_type="I", nemo_time_step_s=14400.0)

    def literal_rate(qsr_value, stretch_value):
        return _nemo_qsr_2bd_tendency(
            qsr_value, jnp.asarray(-record["gdepw_1d"]),
            jnp.asarray(dz_ref), stretch_value, config,
            NEMO_CONSTANTS_CONFIG.rho_0, NEMO_CONSTANTS_CONFIG.c_sw)

    def associated(qsr_value, stretch_value, before_value):
        rate = literal_rate(qsr_value, stretch_value)
        return (before_value + rate) - before_value

    def literal_rows(qsr_value, stretch_value):
        from legoesm.core.source_rounding import nemo_source_round as sr
        from legoesm.core.transcendentals import exp as precision_exp
        from legoesm.ocean.physics.shortwave_penetration import JERLOV_TYPES
        dtype = qsr_value.dtype
        one = jnp.asarray(1.0, dtype=dtype)
        params = JERLOV_TYPES["I"]
        rho0_csw = sr(
            jnp.asarray(NEMO_CONSTANTS_CONFIG.rho_0, dtype=dtype)
            * jnp.asarray(NEMO_CONSTANTS_CONFIG.c_sw, dtype=dtype))
        reciprocal = sr(one / rho0_csw)
        r1_si0 = sr(one / jnp.asarray(params.zeta1, dtype=dtype))
        r1_si1 = sr(one / jnp.asarray(params.zeta2, dtype=dtype))
        rn_abs = jnp.asarray(params.R, dtype=dtype)
        zz0 = sr(rn_abs * reciprocal)
        zz1 = sr(sr(one - rn_abs) * reciprocal)
        depth = sr(
            jnp.asarray(record["gdepw_1d"])
            * stretch_value[..., jnp.newaxis])
        arg0 = sr((-depth) * r1_si0)
        arg1 = sr((-depth) * r1_si1)
        exp0 = precision_exp(arg0)
        exp1 = precision_exp(arg1)
        both = sr(sr(zz0 * exp0) + sr(zz1 * exp1))
        visible = sr(zz1 * exp1)
        wmask = jnp.asarray(record["wmask"][interior])
        next_attenuation = jnp.concatenate((
            both[..., 1:3] * wmask[..., 1:3],
            visible[..., 3:18] * wmask[..., 3:18]), axis=-1)
        previous_attenuation = jnp.concatenate((
            both[..., :1], next_attenuation[..., :-1]), axis=-1)
        ze3t = sr(jnp.asarray(dz_ref) * stretch_value[..., jnp.newaxis])
        absorbed_value = sr(previous_attenuation - next_attenuation)
        numerator_value = sr(qsr_value[..., jnp.newaxis] * absorbed_value)
        rate_value = sr(numerator_value / ze3t[..., :17])
        return {
            "coefficients": jnp.stack((zz0, zz1)),
            "surface_depth": depth[..., 0],
            "surface_arguments": jnp.stack((arg0[..., 0], arg1[..., 0]), -1),
            "surface_exponentials": jnp.stack((exp0[..., 0], exp1[..., 0]), -1),
            "surface_attenuation": both[..., 0],
            "live_depth": depth[..., 1:18],
            "live_arguments": jnp.stack((arg0[..., 1:18], arg1[..., 1:18]), -1),
            "live_exponentials": jnp.stack((exp0[..., 1:18], exp1[..., 1:18]), -1),
            "live_ze3t": ze3t[..., :17],
            "live_attenuation": next_attenuation,
            "absorbed_flux_fraction": absorbed_value,
            "absorbed_flux_numerator": numerator_value,
            "direct_rate": rate_value,
        }

    eager_rate = np.asarray(literal_rate(jnp.asarray(qsr), jnp.asarray(stretch)))
    jit_rate = np.asarray(jax.jit(literal_rate)(
        jnp.asarray(qsr), jnp.asarray(stretch)))
    eager_associated = np.asarray(associated(
        jnp.asarray(qsr), jnp.asarray(stretch), jnp.asarray(before)))
    jit_associated = np.asarray(jax.jit(associated)(
        jnp.asarray(qsr), jnp.asarray(stretch), jnp.asarray(before)))
    eager_rows = {
        name: np.asarray(value) for name, value in literal_rows(
            jnp.asarray(qsr), jnp.asarray(stretch)).items()}
    jit_rows = {
        name: np.asarray(value) for name, value in jax.jit(literal_rows)(
            jnp.asarray(qsr), jnp.asarray(stretch)).items()}
    if plant:
        moved = _score(direct, jit_rate, wet)["cells_unequal"]
        require(moved > 0, "r3t ULP plant did not move the JIT qsr row")
        raise GateError(f"STATUS PLANT-FIRED: {plant}; unequal={moved}")

    card = build_nemo_testcase_card("GYRE-zco")
    gate = _load_sibling(
        "nemo_testcase_l2_gyre_phase3_gate",
        "nemo_testcase_l2_gyre_phase3_gate.py")
    _, surface = gate._surface_forcings(
        card, card.recipe.initial_state, 1080)
    model_qsr = np.asarray(surface.sw_down, dtype=np.float64).T
    model_qmm = np.asarray(
        np.load(lego_trace / "q_Kmm.npy", mmap_mode="r")[1079],
        dtype=np.float64).T
    nemo_qmm = stretch
    model_frame = {
        name: np.asarray(
            np.load(lego_trace / f"{name}.npy", mmap_mode="r")[1079])
        for name in year.LEGO_PROCESS_FIELDS
    }
    model_process_row = year.lego_process_temperature_rows(
        model_frame)["shortwave"].transpose(1, 0, 2)
    nemo_process_row = year.process_temperature_rows(
        process)["shortwave"].transpose(1, 0, 2)
    qmm_mask = np.any(wet, axis=-1)
    static_depth = -np.asarray(card.recipe.z_coord.z_half_ref)
    static_thickness = np.broadcast_to(
        np.asarray(card.recipe.z_coord.dz_ref),
        record["e3t_3d"][interior][..., :JPK - 1].shape)
    input_rows = {
        "qsr_surface_flux": _score(record["qsr"], model_qsr,
                                   qmm_mask),
        "gdepw_1d": _score(record["gdepw_1d"], static_depth,
                            np.ones(record["gdepw_1d"].shape, dtype=bool)),
        "e3t_reference": _score(
            record["e3t_3d"][interior][..., :JPK - 1], static_thickness,
            wet),
        "r3t_Kmm": _score(nemo_qmm, model_qmm, qmm_mask),
    }
    first_inherited = next(
        name for name, row in input_rows.items() if row["cells_unequal"])
    row_scores = {}
    for name, reference in compiled_rows.items():
        if reference.ndim == 1:
            row_mask = np.ones(reference.shape, dtype=bool)
        elif reference.shape[:2] == wet.shape[:2]:
            if reference.ndim >= 3 and reference.shape[2] == 17:
                base_mask = wet[..., :17]
                if reference.ndim == 4:
                    base_mask = np.broadcast_to(
                        base_mask[..., None], reference.shape).copy()
                    if name in ("live_arguments", "live_exponentials"):
                        # The IR arm is evaluated only through nk0=2; levels
                        # 3..nkV execute the visible expression alone.
                        base_mask[..., 2:, 0] = False
                row_mask = np.broadcast_to(base_mask, reference.shape)
            elif reference.ndim == 2:
                row_mask = qmm_mask
            else:
                row_mask = np.broadcast_to(
                    qmm_mask[..., None], reference.shape)
        else:  # pragma: no cover - every registered row is covered above
            raise GateError(f"unregistered statement-row shape {name}")
        row_scores[name] = {
            "isolated_eager": _score(reference, eager_rows[name], row_mask),
            "isolated_jit": _score(reference, jit_rows[name], row_mask),
        }
    return {
        "format": "gyre-round188-developed-qsr-walk-v1",
        "status": "HELD",
        "precision": "fp64/libm",
        "oracle_step": 1080,
        "input_rows": input_rows,
        "first_inherited_operand": first_inherited,
        "statement_rows": {
            "compiled_order": row_scores,
            "isolated_eager_direct_rate": _score(direct, eager_rate, wet),
            "isolated_jit_direct_rate": _score(direct, jit_rate, wet),
            "isolated_eager_associated_update": _score(
                replay, eager_associated, wet),
            "isolated_jit_associated_update": _score(
                replay, jit_associated, wet),
            "production_jit_own_chain_shortwave": _score(
                nemo_process_row, model_process_row, wet),
            "production_eager_own_chain_shortwave": {
                "classification": "UNMEASURED"},
            "production_jit_given_nemo_stage_entry": {
                "classification": "UNMEASURED"},
            "production_eager_given_nemo_stage_entry": {
                "classification": "UNMEASURED"},
        },
        "verdict": (
            "qsr_2BD is BIT given NEMO operands in isolated eager/JIT; "
            "the first own-chain input difference is r3t(Kmm), inherited "
            "from the stage free surface; full-step NEMO-entry ownership "
            "is unmeasured"),
    }


def _trace_frame(year, trace) -> dict[str, np.ndarray]:
    return year._trace_frame(trace)


def _producer_walk(record: dict, process: dict, reference_frame: dict,
                   candidate_trace, card, wet2: np.ndarray,
                   restart_path: Path) -> dict:
    """Walk the source-ordered producer only after QSR closes."""
    from netCDF4 import Dataset
    import jax.numpy as jnp
    from legoesm.ocean.eos import nemo_r3t_stretch

    oracle = {
        name: 1.0 + np.asarray(process[name], dtype=np.float64)
        for name in ("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")
    }
    model = {
        name: np.asarray(reference_frame[name], dtype=np.float64)
        for name in ("q_Kbb", "q_Kmm", "q_Kaa")
    }
    oracle_blend = 1.0 + 0.5 * (
        (oracle["r3t_Kbb"] - 1.0) + (oracle["r3t_Kaa"] - 1.0))
    calibration = _score(oracle["r3t_Kmm"], oracle_blend, wet2)
    require(calibration["cells_unequal"] == 0,
            "compiled half-step ratio does not reproduce recorded r3t(Kmm)")
    model_blend = 1.0 + 0.5 * (
        (model["q_Kbb"] - 1.0) + (model["q_Kaa"] - 1.0))

    with Dataset(restart_path, "r") as handle:
        sshn = np.asarray(handle.variables["sshn"][0], dtype=np.float64)
    require(sshn.shape == wet2.shape,
            f"restart ssh shape {sshn.shape} != {wet2.shape}")
    ratio_from_nemo_ssh = np.asarray(nemo_r3t_stretch(
        card.recipe.z_coord, jnp.asarray(sshn),
        card.recipe.initial_state.H_bathy.data,
        evaluation="nemo_reciprocal"))
    model_eta_after = np.asarray(candidate_trace.state_after.eta.data)
    rows = {
        "step_entry_stretch": _score(
            oracle["r3t_Kbb"], model["q_Kbb"], wet2),
        "after_ssh": _score(sshn, model_eta_after, wet2),
        "after_ratio_statement_given_nemo_ssh": _score(
            oracle["r3t_Kaa"], ratio_from_nemo_ssh, wet2),
        "after_stretch": _score(
            oracle["r3t_Kaa"], model["q_Kaa"], wet2),
        "half_step_blend_from_model_operands": _score(
            oracle["r3t_Kmm"], model_blend, wet2),
        "production_half_step_stretch": _score(
            oracle["r3t_Kmm"], model["q_Kmm"], wet2),
    }
    first = next((name for name, row in rows.items()
                  if row["cells_unequal"]), None)
    return {
        "compiled_blend_calibration": calibration,
        "rows": rows,
        "first_non_bit": first,
        "compiled_source_order": [
            "step_entry_stretch", "after_ssh",
            "after_ratio_statement_given_nemo_ssh", "after_stretch",
            "half_step_blend_from_model_operands",
            "production_half_step_stretch"],
    }


def association_ranking(root: Path, process_record: Path,
                        reference_trace: Path) -> dict:
    """Rank the remaining QSR inputs after the production ratio arm fails."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "shortwave association ranking is not fp64/libm")
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
    from legoesm.ocean.physics.shortwave_penetration import (
        ShortwavePenetrationConfig, _nemo_qsr_2bd_tendency)

    year = _load_sibling(
        "_round189_association_year", "nemo_testcase_l2_gyre_year_owners.py")
    process = year.read_process_record(process_record)
    record = read_record(root / "oracle_qsr_walk_kt00001080.bin")
    frame = {
        name: np.asarray(
            np.load(reference_trace / f"{name}.npy", mmap_mode="r")[1079])
        for name in year.LEGO_PROCESS_FIELDS
    }
    gate = _load_sibling(
        "_round189_association_gate", "nemo_testcase_l2_gyre_phase3_gate.py")
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    card = build_nemo_testcase_card("GYRE-zco")
    wet = gate.expected_masks(card)["T"]
    wet2 = np.any(wet, axis=-1)
    qsr = record["qsr"].T
    dz_ref = np.asarray(record["e3t_3d"][2, 2, :JPK - 1])
    config = ShortwavePenetrationConfig(
        scheme="nemo_qsr_2bd", water_type="I", nemo_time_step_s=14400.0)

    @jax.jit
    def direct_rate(qsr_value, stretch_value):
        return _nemo_qsr_2bd_tendency(
            qsr_value, jnp.asarray(-record["gdepw_1d"]),
            jnp.asarray(dz_ref), stretch_value, config,
            NEMO_CONSTANTS_CONFIG.rho_0, NEMO_CONSTANTS_CONFIG.c_sw)

    model_direct = np.asarray(direct_rate(
        jnp.asarray(qsr), jnp.asarray(frame["q_Kmm"])))
    nemo_direct = _compiled_direct_rate(record).transpose(1, 0, 2)
    nemo_rows = year.process_temperature_rows(process)
    model_rows = year.lego_process_temperature_rows(frame)
    nemo_shortwave = np.asarray(nemo_rows["shortwave"])
    model_shortwave = np.asarray(model_rows["shortwave"])

    qbb = 1.0 + np.asarray(process["r3t_Kbb"])
    qmm = 1.0 + np.asarray(process["r3t_Kmm"])
    qaa = 1.0 + np.asarray(process["r3t_Kaa"])
    tbb = np.asarray(process["Tbb"])[..., :JPK - 1]
    base = qbb[..., None] * tbb
    rhs_sbc = np.asarray(
        process["rhs_after_surface_boundary"])[..., :JPK - 1]
    rhs_qsr = np.asarray(process["rhs_after_shortwave"])[..., :JPK - 1]
    nemo_bsbc = (base + 14400.0 * qmm[..., None] * rhs_sbc) / qaa[..., None]
    nemo_bqsr = (base + 14400.0 * qmm[..., None] * rhs_qsr) / qaa[..., None]
    require(_score(
        nemo_shortwave, nemo_bqsr - nemo_bsbc, wet)["cells_unequal"] == 0,
        "NEMO cumulative QSR boundaries do not reproduce the process row")

    rebuilt_model = _associate_qsr(
        frame["Bsbc"], frame["q_Kmm"], frame["q_Kaa"], model_direct)
    rebuild_vs_actual = _score(model_shortwave, rebuilt_model, wet)
    actual_vs_nemo = _score(nemo_shortwave, model_shortwave, wet)
    arms = {
        "rebuilt_model_operands": _score(
            nemo_shortwave, rebuilt_model, wet),
        "nemo_qmm_only": _score(
            nemo_shortwave, _associate_qsr(
                frame["Bsbc"], qmm, frame["q_Kaa"], model_direct), wet),
        "nemo_qaa_only": _score(
            nemo_shortwave, _associate_qsr(
                frame["Bsbc"], frame["q_Kmm"], qaa, model_direct), wet),
        "nemo_direct_rate_only": _score(
            nemo_shortwave, _associate_qsr(
                frame["Bsbc"], frame["q_Kmm"], frame["q_Kaa"],
                nemo_direct), wet),
        "nemo_preceding_accumulator_only": _score(
            nemo_shortwave, _associate_qsr(
                nemo_bsbc, frame["q_Kmm"], frame["q_Kaa"],
                model_direct), wet),
        "all_nemo_inputs": _score(
            nemo_shortwave, _associate_qsr(
                nemo_bsbc, qmm, qaa, nemo_direct), wet),
    }
    for row in arms.values():
        row["max_abs_removed_fraction_vs_actual"] = _removed_fraction(
            actual_vs_nemo["max_abs"], row["max_abs"])
        row["rms_removed_fraction_vs_actual"] = _removed_fraction(
            actual_vs_nemo["rms"], row["rms"])
    first_surviving = (
        "stage3_qsr_source_association"
        if rebuild_vs_actual["cells_unequal"] else None)
    return {
        "format": "gyre-round189-qsr-association-ranking-v1",
        "status": "HELD", "step": 1080, "precision": "fp64/libm",
        "input_rows": {
            "qsr_surface_flux": _score(record["qsr"].T, qsr, wet2),
            "step_entry_stretch": _score(qbb, frame["q_Kbb"], wet2),
            "live_stretch": _score(qmm, frame["q_Kmm"], wet2),
            "after_stretch": _score(qaa, frame["q_Kaa"], wet2),
            "direct_rate": _score(nemo_direct, model_direct, wet),
            "preceding_accumulator": _score(
                nemo_bsbc, frame["Bsbc"], wet),
        },
        "actual_production_row": actual_vs_nemo,
        "direct_rebuild_vs_actual_production": rebuild_vs_actual,
        "isolated_association_arms": arms,
        "first_surviving_boundary": first_surviving,
        "verdict": (
            "the production QSR process row is not the direct qsr_2BD rate "
            "associated with its recorded qmm/qaa and preceding accumulator"
            if first_surviving else
            "the direct QSR association reproduces the production row"),
    }


def association_split(process_record: Path, reference_trace: Path,
                      expect_commit: str, plant: bool = False) -> dict:
    """Expose and close the production-JIT stage-3 QSR association."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "production association split is not fp64/libm")
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"association split requires clean tree: {stamp['dirty_paths']}")
    require(stamp["commit"] == expect_commit,
            "association split commit differs from --expect-commit")
    year = _load_sibling(
        "_round190_association_year", "nemo_testcase_l2_gyre_year_owners.py")
    trace_manifest = json.loads((reference_trace / "manifest.json").read_text())
    trace_admission = year.validate_lego_process_trace(
        reference_trace, trace_manifest["producer_commit"])
    require(trace_admission["layout"]["steps"] == [1, 1080],
            "reference process trace does not cover steps 1..1080")
    process = year.read_process_record(process_record)
    require(process["kstp"] == 1080, "process operand is not step 1080")
    card = build_nemo_testcase_card("GYRE-zco")
    gate = _load_sibling(
        "_round190_phase3_gate", "nemo_testcase_l2_gyre_phase3_gate.py")
    wet = gate.expected_masks(card)["T"]

    state = card.recipe.initial_state
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    started = time.time()
    for completed in range(1079):
        kt = completed + 1
        freshwater, surface = gate._surface_forcings(card, state, kt)
        state = ordinary_model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface)
        if kt % 180 == 0:
            print(f"  round190 production prefix step {kt:4d}  "
                  f"{time.time() - started:7.1f} s", flush=True)

    freshwater, surface = gate._surface_forcings(card, state, 1080)
    hooks = _NEMOWSRK3TestHooks(tracer_process_trace=())
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    trace = jax.device_get(trace_model.step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface))
    frame = _trace_frame(year, trace)
    association = trace.qsr_association
    arrays = {
        "tendency_kbb": np.asarray(association.tendency_kbb),
        "qsr_kbb": np.asarray(association.qsr_kbb),
        "qsr_kmm": np.asarray(association.qsr_kmm),
        "thickness_kbb": np.asarray(association.thickness_kbb),
        "thickness_kmm": np.asarray(association.thickness_kmm),
        "process_qsr_kbb": np.asarray(association.process_qsr_kbb),
        "process_surface_rate": np.asarray(association.process_surface_rate),
        "process_qsr_rate": np.asarray(association.process_qsr_rate),
    }
    require(all(value.shape == wet.shape for value in arrays.values()),
            "association trace array shape differs from the wet-cell mask")

    tendency = arrays["tendency_kbb"]
    qsr_kbb = arrays["qsr_kbb"]
    qsr_kmm = arrays["qsr_kmm"]
    h_kbb = arrays["thickness_kbb"]
    h_kmm = arrays["thickness_kmm"]
    process_qsr_kbb = arrays["process_qsr_kbb"]
    surface_rebuilt = ((tendency - process_qsr_kbb) * h_kbb
                       / np.maximum(h_kmm, 1.0e-10))
    stage3_rebuilt = ((tendency - qsr_kbb) * h_kbb
                      / np.maximum(h_kmm, 1.0e-10) + qsr_kmm)
    qsr_rebuilt = stage3_rebuilt - surface_rebuilt

    model_shortwave = np.asarray(frame["Bqsr"] - frame["Bsbc"])
    direct_rebuild = _associate_qsr(
        frame["Bsbc"], frame["q_Kmm"], frame["q_Kaa"], qsr_kmm)
    reproduced_split = _score(model_shortwave, direct_rebuild, wet)
    require(reproduced_split["cells_unequal"] == 9679,
            "Round-189 direct-rebuild split moved to "
            f"{reproduced_split['cells_unequal']} cells")
    require(reproduced_split["max_abs"] == 2.467770444880557e-06,
            "Round-189 direct-rebuild maximum moved to "
            f"{reproduced_split['max_abs']}")

    identity_rows = {
        "process_qsr_kbb_vs_qsr_kbb": _score(
            qsr_kbb, process_qsr_kbb, wet),
        "surface_rate_bridge": _score(
            arrays["process_surface_rate"], surface_rebuilt, wet),
        "qsr_rate_bridge": _score(
            arrays["process_qsr_rate"], qsr_rebuilt, wet),
        "qsr_rate_vs_qsr_kmm": _score(
            qsr_kmm, arrays["process_qsr_rate"], wet),
    }
    cumulative_rebuild = frame["Bsbc"] + _associate_qsr(
        frame["Bsbc"], frame["q_Kmm"], frame["q_Kaa"],
        arrays["process_qsr_rate"])
    cumulative_row = _score(frame["Bqsr"], cumulative_rebuild, wet)

    if plant:
        plant_index = tuple(int(value) for value in np.argwhere(wet)[0])
        plant_delta = float(np.ldexp(1.0, -40))
        plant_model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                tracer_process_trace=(*plant_index, plant_delta)))
        planted = jax.device_get(plant_model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface))
        planted_frame = _trace_frame(year, planted)
        plant_row = _score(frame["Bqsr"], planted_frame["Bqsr"], wet)
        require(plant_row["cells_unequal"] > 0,
                "production association plant moved no Bqsr cell")
        raise GateError(
            "STATUS PLANT-FIRED: production-qsr-association; "
            f"unequal={plant_row['cells_unequal']}")

    bridge_bit = all(
        identity_rows[name]["cells_unequal"] == 0
        for name in ("process_qsr_kbb_vs_qsr_kbb",
                     "surface_rate_bridge", "qsr_rate_bridge"))
    return {
        "format": "gyre-round190-production-qsr-association-v1",
        "status": "HELD", "precision": "fp64/libm", "step": 1080,
        "worktree": stamp, "reference_trace": trace_admission,
        "round189_split_reproduced": reproduced_split,
        "production_jit_identity_rows": identity_rows,
        "returned_cumulative_boundary_rebuild": cumulative_row,
        "first_non_bit_statement": None if bridge_bit else next(
            name for name, row in identity_rows.items()
            if row["cells_unequal"]),
        "verdict": (
            "bridge source algebra is BIT; the non-bit process row is an "
            "observer cumulative-boundary classification artifact"
            if bridge_bit else
            "the first non-bit bridge identity owns the association walk"),
    }


def production_substitution(root: Path, process_record: Path,
                            reference_trace: Path, expect_commit: str,
                            plant: str | None = None) -> dict:
    """Substitute NEMO's r3t(Kmm) at the production-JIT QSR boundary."""
    require(plant in (None, "production-r3t-ulp", "production-r3t-effect"),
            f"unknown production substitution plant {plant}")
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "production shortwave substitution is not fp64/libm")
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"],
            f"production substitution requires clean tree: "
            f"{stamp['dirty_paths']}")
    require(stamp["commit"] == expect_commit,
            "production substitution commit differs from --expect-commit")
    year = _load_sibling(
        "_round189_year_owners", "nemo_testcase_l2_gyre_year_owners.py")
    trace_manifest = json.loads((reference_trace / "manifest.json").read_text())
    trace_admission = year.validate_lego_process_trace(
        reference_trace, trace_manifest["producer_commit"])
    require(trace_admission["layout"]["steps"] == [1, 1080],
            "reference process trace does not cover steps 1..1080")
    process = year.read_process_record(process_record)
    require(process["kstp"] == 1080, "process operand is not step 1080")
    record = read_record(root / "oracle_qsr_walk_kt00001080.bin")
    card = build_nemo_testcase_card("GYRE-zco")
    gate = _load_sibling(
        "_round189_phase3_gate", "nemo_testcase_l2_gyre_phase3_gate.py")
    wet = gate.expected_masks(card)["T"]
    wet2 = np.any(wet, axis=-1)
    reference_frame = {
        name: np.asarray(
            np.load(reference_trace / f"{name}.npy", mmap_mode="r")[1079])
        for name in year.LEGO_PROCESS_FIELDS
    }
    nemo_rows = year.process_temperature_rows(process)
    nemo_shortwave = np.asarray(nemo_rows["shortwave"])
    reference_rows = year.lego_process_temperature_rows(reference_frame)
    baseline = _score(nemo_shortwave, reference_rows["shortwave"], wet)
    require(baseline["cells_unequal"] == 9666,
            f"Round-188 baseline moved to {baseline['cells_unequal']} cells")
    require(baseline["max_abs"] == 2.4678031493863273e-06,
            f"Round-188 baseline max moved to {baseline['max_abs']}")

    state = card.recipe.initial_state
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    started = time.time()
    for completed in range(1079):
        kt = completed + 1
        freshwater, surface = gate._surface_forcings(card, state, kt)
        state = ordinary_model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface)
        if kt % 180 == 0:
            print(f"  round189 production prefix step {kt:4d}  "
                  f"{time.time() - started:7.1f} s", flush=True)

    nemo_stretch = (1.0 + np.asarray(
        record["r3t_Kmm"][NN_HLS:-NN_HLS, NN_HLS:-NN_HLS],
        dtype=np.float64)).T
    override = np.array(nemo_stretch, copy=True)
    if plant:
        j, i = (int(value) for value in np.argwhere(wet2)[0])
        if plant == "production-r3t-ulp":
            override[j, i] = np.nextafter(override[j, i], np.inf)
        else:
            override[j, i] += 2.0 ** -20
    hooks = _NEMOWSRK3TestHooks(
        tracer_process_trace=(), stage3_qsr_stretch_override=jnp.asarray(
            override))
    candidate_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    freshwater, surface = gate._surface_forcings(card, state, 1080)
    candidate = candidate_model.step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface)
    candidate = jax.device_get(candidate)
    candidate_frame = _trace_frame(year, candidate)

    pre_fields = ("Tbb", "q_Kbb", "q_Kmm", "q_Kaa", "B0", "Badv", "Bsbc")
    pre_rows = {
        name: _score(reference_frame[name], candidate_frame[name],
                     wet if candidate_frame[name].ndim == 3 else wet2)
        for name in pre_fields
    }
    require(all(row["cells_unequal"] == 0 for row in pre_rows.values()),
            "ratio substitution moved a registered pre-shortwave boundary")
    candidate_rows = year.lego_process_temperature_rows(candidate_frame)
    candidate_score = _score(
        nemo_shortwave, candidate_rows["shortwave"], wet)
    if plant:
        exact_model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                tracer_process_trace=(),
                stage3_qsr_stretch_override=jnp.asarray(nemo_stretch)))
        exact = jax.device_get(exact_model.step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface))
        exact_frame = _trace_frame(year, exact)
        planted = _score(exact_frame["Bqsr"], candidate_frame["Bqsr"], wet)
        require(planted["cells_unequal"] > 0,
                f"{plant} moved no QSR boundary cell")
        raise GateError(
            f"STATUS PLANT-FIRED: {plant}; "
            f"unequal={planted['cells_unequal']}")

    comparison = {
        "baseline": baseline,
        "nemo_r3t_Kmm_only": candidate_score,
        "max_abs_removed_fraction": _removed_fraction(
            baseline["max_abs"], candidate_score["max_abs"]),
        "rms_removed_fraction": _removed_fraction(
            baseline["rms"], candidate_score["rms"]),
    }
    producer = None
    if candidate_score["cells_unequal"] == 0:
        producer = _producer_walk(
            record, process, reference_frame, candidate, card, wet2,
            root / "GYRE_OMIP_L2_P3_00001080_restart.nc")
    nemo_cumulative = year.process_temperature_rows(process)
    nemo_bsbc = np.asarray(nemo_cumulative["geometry"])
    nemo_bsbc = nemo_bsbc + np.asarray(nemo_cumulative["advection"])
    nemo_bsbc = nemo_bsbc + np.asarray(nemo_cumulative["surface_boundary"])
    remaining = {
        "qsr_surface_flux": _score(
            record["qsr"].T, np.asarray(surface.sw_down), wet2),
        "stage3_step_entry_stretch": _score(
            1.0 + np.asarray(process["r3t_Kbb"]),
            reference_frame["q_Kbb"], wet2),
        "stage3_live_stretch": _score(
            nemo_stretch, reference_frame["q_Kmm"], wet2),
        "preceding_accumulator": _score(
            nemo_bsbc,
            reference_frame["Bsbc"], wet),
    }
    return {
        "format": "gyre-round189-production-qsr-r3t-substitution-v1",
        "status": "HELD", "precision": "fp64/libm", "step": 1080,
        "worktree": stamp, "reference_trace": trace_admission,
        "registered_pre_shortwave_rows": pre_rows,
        "comparison": comparison,
        "prediction": {
            "at_least_90pct_max_removed": (
                comparison["max_abs_removed_fraction"] >= 0.90),
            "confirmed": comparison["max_abs_removed_fraction"] >= 0.90,
        },
        "remaining_input_and_association_rows": remaining,
        "producer_walk": producer,
        "verdict": (
            "r3t(Kmm) closes the production-JIT QSR row; producer walked"
            if producer is not None else
            "r3t(Kmm) alone does not close the production-JIT QSR row; "
            "upstream producer walk withheld"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=("actual-increment-ulp",))
    parser.add_argument("--walk", action="store_true")
    parser.add_argument("--production-substitution", action="store_true")
    parser.add_argument("--association-ranking", action="store_true")
    parser.add_argument("--association-split", action="store_true")
    parser.add_argument("--association-plant", action="store_true")
    parser.add_argument("--process-record", type=Path)
    parser.add_argument("--lego-trace", type=Path)
    parser.add_argument("--walk-plant", choices=("r3t-ulp",))
    parser.add_argument("--production-plant",
                        choices=("production-r3t-ulp",
                                 "production-r3t-effect"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        if args.association_split:
            require(args.process_record is not None and args.lego_trace is not None,
                    "--association-split requires --process-record and "
                    "--lego-trace")
            report = association_split(
                args.process_record, args.lego_trace, args.expect_commit,
                args.association_plant)
        elif args.association_ranking:
            require(args.process_record is not None and args.lego_trace is not None,
                    "--association-ranking requires --process-record and "
                    "--lego-trace")
            report = association_ranking(
                args.root, args.process_record, args.lego_trace)
        elif args.production_substitution:
            require(args.process_record is not None and args.lego_trace is not None,
                    "--production-substitution requires --process-record "
                    "and --lego-trace")
            report = production_substitution(
                args.root, args.process_record, args.lego_trace,
                args.expect_commit, args.production_plant)
        elif args.walk:
            require(args.process_record is not None and args.lego_trace is not None,
                    "--walk requires --process-record and --lego-trace")
            report = walk(
                args.root, args.process_record, args.lego_trace,
                args.walk_plant)
        else:
            report = admit(args.root, args.expect_commit, args.plant)
    except GateError as exc:
        print(str(exc))
        return 1
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print("STATUS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
