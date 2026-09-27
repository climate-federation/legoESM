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
    return {
        "cells_scored": int(mask.sum()),
        "cells_unequal": unequal,
        "max_abs": float(np.max(np.abs(
            selected_reference - selected_candidate), initial=0.0)),
        "classification": "BIT" if unequal == 0 else "NON-BIT",
    }


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=("actual-increment-ulp",))
    parser.add_argument("--walk", action="store_true")
    parser.add_argument("--process-record", type=Path)
    parser.add_argument("--lego-trace", type=Path)
    parser.add_argument("--walk-plant", choices=("r3t-ulp",))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        if args.walk:
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
