#!/usr/bin/env python3
"""Fail-closed reader and production-path walk for the kt=2 TKE record."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

from legoesm import constants

MAGIC = b"NEMO_L2_R56TKE2 "
FIELDS = (
    "rn_Dt", "rn_ediff", "rn_ediss", "rn_ebb", "rn_emin", "rn_emin0",
    "rmxl_min", "rn_mxl0", "rn_bshear", "rn_lc", "nn_pdl", "nn_mxl",
    "ln_mxl0", "nn_etau", "nn_htau", "nn_eice", "ln_lc", "taum_entry",
    "tmask", "wmask", "avm_floor", "avt_floor", "en_entry", "avm_entry",
    "avt_entry", "dissl_entry", "rn2", "rn2b", "sh2", "e3t_Kmm",
    "e3w_Kmm", "matrix_diag",
    "matrix_upper", "matrix_lower", "rhs_pre_sweep", "en_post_sweep",
    "mxl_momentum", "mxl_dissipation", "pdlr", "avm_closure",
    "avt_closure", "dissl_output", "avm_pre_evd", "avt_pre_evd",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _unequal_bits(actual: np.ndarray, expected: np.ndarray,
                  mask: np.ndarray) -> int:
    """Count unequal fp64 bit patterns on the explicitly consumed cells."""
    actual_bits = np.asarray(actual, dtype="=f8").view("=u8")
    expected_bits = np.asarray(expected, dtype="=f8").view("=u8")
    return int(np.count_nonzero(actual_bits[mask] != expected_bits[mask]))


def _calibrate_closure(arrays: dict, head: dict) -> dict[str, int]:
    """Rebuild NEMO's recorded Prandtl factor and closure outputs exactly."""
    jpkm1 = head["jpkm1"]
    interior = np.zeros_like(arrays["wmask"], dtype=bool)
    interior[:, :, 1:jpkm1] = arrays["wmask"][:, :, 1:jpkm1] != 0.0
    wet = arrays["wmask"][:, :, :jpkm1] != 0.0

    # Compiled R56TKE zdftke.f90:751,394-413.  Preserve NEMO's branch and
    # association; in particular, do not turn the exact-zero guard into an
    # epsilon or evaluate the inactive division arm.
    rn2b = arrays["rn2b"][:, :, 1:jpkm1]
    avm_entry = arrays["avm_entry"][:, :, 1:jpkm1]
    sh2 = arrays["sh2"][:, :, 1:jpkm1]
    zri = np.zeros_like(rn2b)
    stable = rn2b > np.float64(0.0)
    zdiv = sh2 + np.float64(arrays["rn_bshear"])
    zero_div = stable & (zdiv == np.float64(0.0))
    nonzero_div = stable & ~zero_div
    numerator = rn2b * avm_entry
    zri[zero_div] = numerator[zero_div] / np.float64(arrays["rn_bshear"])
    zri[nonzero_div] = numerator[nonzero_div] / zdiv[nonzero_div]
    ri_cri = (np.float64(2.0) /
              (np.float64(2.0) + np.float64(arrays["rn_ediss"]) /
               np.float64(arrays["rn_ediff"])))
    pdlr_expected = np.maximum(
        np.float64(0.1), ri_cri / np.maximum(ri_cri, zri))
    pdlr_actual = arrays["pdlr"][:, :, 1:jpkm1]
    pdlr_unequal = _unequal_bits(
        pdlr_actual, pdlr_expected, interior[:, :, 1:jpkm1])

    # Compiled R56TKE zdftke.f90:690-701.  This is the calibration arm: use
    # only recorded NEMO operands and reproduce its exact statement order.
    en = arrays["en_post_sweep"][:, :, :jpkm1]
    zsqen = np.sqrt(en)
    zav = np.float64(arrays["rn_ediff"]) * arrays[
        "mxl_momentum"][:, :, :jpkm1] * zsqen
    avm_expected = np.maximum(
        zav, arrays["avm_floor"][:, :, :jpkm1]) * arrays[
            "wmask"][:, :, :jpkm1]
    avt_expected = np.maximum(
        zav, arrays["avt_floor"][:, :, :jpkm1]) * arrays[
            "wmask"][:, :, :jpkm1]
    avt_expected[:, :, 1:jpkm1] = np.maximum(
        arrays["pdlr"][:, :, 1:jpkm1] * avt_expected[:, :, 1:jpkm1],
        arrays["avt_floor"][:, :, 1:jpkm1]) * arrays[
            "wmask"][:, :, 1:jpkm1]
    dissl_expected = zsqen / arrays["mxl_dissipation"][:, :, :jpkm1]

    counts = {
        "wet_closure_cells": int(np.count_nonzero(wet)),
        "wet_prandtl_cells": int(np.count_nonzero(interior)),
        "prandtl_unequal": pdlr_unequal,
        "avm_unequal": _unequal_bits(
            arrays["avm_closure"][:, :, :jpkm1], avm_expected, wet),
        "avt_unequal": _unequal_bits(
            arrays["avt_closure"][:, :, :jpkm1], avt_expected, wet),
        "dissl_unequal": _unequal_bits(
            arrays["dissl_output"][:, :, :jpkm1], dissl_expected, wet),
    }
    require(all(counts[f"{name}_unequal"] == 0
                for name in ("prandtl", "avm", "avt", "dissl")),
            f"closure calibration over {counts['wet_closure_cells']} wet "
            f"closure cells/{counts['wet_prandtl_cells']} wet Prandtl cells "
            "has unequal cells: "
            + ", ".join(f"{name}={counts[f'{name}_unequal']}"
                        for name in ("prandtl", "avm", "avt", "dissl")))
    return counts


def _calibrate_en_and_mixing(arrays: dict, head: dict) -> dict[str, int]:
    """Rebuild NEMO's TKE sweep and nn_mxl=3 lengths in source order."""
    jpkm1 = head["jpkm1"]
    wet_solve = arrays["wmask"][:, :, 1:jpkm1] != 0.0
    wet_mixing = arrays["wmask"][:, :, :jpkm1] != 0.0

    # Compiled R58TKE zdftke.f90:466-483.  NEMO first eliminates the
    # diagonal, then overwrites zd_lw with the RHS recurrence, seeds jpkm1,
    # back-substitutes, and finally applies MAX(en,rn_emin)*wmask.
    diag = arrays["matrix_diag"][:, :, :jpkm1].copy()
    upper = arrays["matrix_upper"][:, :, :jpkm1]
    lower = arrays["matrix_lower"][:, :, :jpkm1]
    rhs = arrays["rhs_pre_sweep"][:, :, :jpkm1]
    work = np.zeros_like(rhs)
    work[:, :, 0] = lower[:, :, 0]
    for k in range(1, jpkm1):
        diag[:, :, k] = (diag[:, :, k]
                         - lower[:, :, k] * upper[:, :, k - 1]
                         / diag[:, :, k - 1])
    for k in range(1, jpkm1):
        work[:, :, k] = (rhs[:, :, k]
                         - lower[:, :, k] / diag[:, :, k - 1]
                         * work[:, :, k - 1])
    solved = np.zeros_like(rhs)
    solved[:, :, jpkm1 - 1] = (
        work[:, :, jpkm1 - 1] / diag[:, :, jpkm1 - 1])
    for k in range(jpkm1 - 2, 0, -1):
        solved[:, :, k] = ((work[:, :, k]
                            - upper[:, :, k] * solved[:, :, k + 1])
                           / diag[:, :, k])
    solved[:, :, 1:jpkm1] = np.maximum(
        solved[:, :, 1:jpkm1], np.float64(arrays["rn_emin"])) * arrays[
            "wmask"][:, :, 1:jpkm1]

    # Compiled R58TKE zdftke.f90:589,601-603,612-619,628-629,634,
    # 669-682.  The GYRE record resolves ln_mxl0=T and nn_mxl=3.
    floor = np.float64(arrays["rmxl_min"])
    mxlm = np.full_like(arrays["en_post_sweep"], floor)
    mxld = np.full_like(mxlm, floor)
    zraug = (np.float64(constants.kappa_von_karman) * np.float64(2.0e5)
             / (np.float64(constants.rho_ocean_nemo)
                * np.float64(constants.g_nemo)))
    mxlm[:, :, 0] = np.maximum(
        np.float64(arrays["rn_mxl0"]),
        zraug * arrays["taum_entry"] * arrays["tmask"][:, :, 0])
    rsmall = np.float64(0.5) * np.finfo(np.float64).eps
    zrn2 = np.maximum(arrays["rn2"][:, :, 1:jpkm1], rsmall)
    mxlm[:, :, 1:jpkm1] = np.maximum(
        floor, np.sqrt(
            np.float64(2.0) * arrays["en_post_sweep"][:, :, 1:jpkm1]
            / zrn2))
    mxld[:, :, 0] = mxlm[:, :, 0]
    for k in range(1, jpkm1):
        mxld[:, :, k] = np.minimum(
            mxld[:, :, k - 1] + arrays["e3t_Kmm"][:, :, k - 1],
            mxlm[:, :, k])
    for k in range(jpkm1 - 1, 0, -1):
        mxlm[:, :, k] = np.minimum(
            mxlm[:, :, k + 1] + arrays["e3t_Kmm"][:, :, k + 1],
            mxlm[:, :, k])
    momentum = np.minimum(mxld[:, :, :jpkm1], mxlm[:, :, :jpkm1])
    dissipation = np.sqrt(
        mxld[:, :, :jpkm1] * mxlm[:, :, :jpkm1])

    counts = {
        "wet_solve_cells": int(np.count_nonzero(wet_solve)),
        "wet_mixing_cells": int(np.count_nonzero(wet_mixing)),
        "en_unequal": _unequal_bits(
            arrays["en_post_sweep"][:, :, 1:jpkm1],
            solved[:, :, 1:jpkm1], wet_solve),
        "mxl_momentum_unequal": _unequal_bits(
            arrays["mxl_momentum"][:, :, :jpkm1], momentum, wet_mixing),
        "mxl_dissipation_unequal": _unequal_bits(
            arrays["mxl_dissipation"][:, :, :jpkm1],
            dissipation, wet_mixing),
    }
    require(all(counts[name] == 0 for name in (
        "en_unequal", "mxl_momentum_unequal", "mxl_dissipation_unequal")),
        f"source-order calibration over {counts['wet_solve_cells']} wet solve "
        f"cells/{counts['wet_mixing_cells']} wet mixing cells has unequal "
        "cells: " + ", ".join(f"{name}={counts[name]}" for name in (
            "en_unequal", "mxl_momentum_unequal",
            "mxl_dissipation_unequal")))
    return counts


def read_record(path: Path, *, plant: str | None = None) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    offset = 0

    def take(count: int) -> bytes:
        nonlocal offset
        require(offset + count <= len(raw), "record is truncated")
        value = raw[offset:offset + count]
        offset += count
        return value

    magic = take(16)
    if plant == "header":
        magic = b"X" + magic[1:]
    require(magic == MAGIC, f"wrong magic {magic!r}")
    header = struct.unpack("=13i", take(13 * 4))
    keys = ("version", "kt", "Kbb", "Kmm", "jpi", "jpj", "jpk", "jpkm1",
            "ntsi", "ntei", "ntsj", "ntej", "real_bits")
    head = dict(zip(keys, header, strict=True))
    require(head["version"] == 2 and head["kt"] == 2,
            f"unexpected version/kt {head['version']}/{head['kt']}")
    require((head["jpi"], head["jpj"], head["jpk"], head["jpkm1"])
            == (32, 22, 31, 30), f"unexpected GYRE shape {head}")
    require(head["real_bits"] == 64, "record is not fp64")

    arrays = {}
    for expected in FIELDS:
        label = take(16).decode("ascii").rstrip()
        require(label == expected, f"field order mismatch: {label!r} != {expected!r}")
        ndim, n1, n2, n3 = struct.unpack("=4i", take(16))
        require(ndim in (0, 2, 3), f"invalid rank for {label}: {ndim}")
        shape = (() if ndim == 0 else
                 (n1, n2) if ndim == 2 else (n1, n2, n3))
        count = 1 if ndim == 0 else int(np.prod(shape))
        value = np.frombuffer(take(count * 8), dtype="=f8").copy()
        if shape:
            value = value.reshape(shape, order="F")
        else:
            value = value.reshape(()).item()
        arrays[label] = value
    require(offset == len(raw), f"record has {len(raw) - offset} trailing bytes")
    expected_2d = (head["jpi"], head["jpj"])
    expected_3d = (*expected_2d, head["jpk"])
    if plant == "shape":
        # Corrupt an array against the unchanged producer stamp.  Mutating
        # jpi instead would be rejected earlier by the fixed-GYRE census and
        # would never exercise this stamped-extent-vs-array-shape guard.
        arrays["avt_pre_evd"] = arrays["avt_pre_evd"][:-1, :, :]
    for name, value in arrays.items():
        shape = np.shape(value)
        if shape:
            require(shape == (expected_2d if name == "taum_entry" else expected_3d),
                    f"{name} shape {shape} disagrees with stamped extents")

    if plant == "nan":
        arrays["avt_pre_evd"] = np.array(arrays["avt_pre_evd"], copy=True)
        arrays["avt_pre_evd"].flat[0] = np.nan
    elif plant == "config":
        arrays["rn_mxl0"] = 0.04
    elif plant == "copy":
        arrays["avt_pre_evd"] = np.array(arrays["avt_pre_evd"], copy=True)
        arrays["avt_pre_evd"][head["ntsi"] - 1, head["ntsj"] - 1, 1] += 1.0
    elif plant == "sweep":
        wet = np.argwhere(arrays["wmask"][:, :, 1:head["jpkm1"]] != 0.0)
        require(wet.size != 0, "sweep plant found no wet solved interface")
        i, j, k0 = wet[0]
        arrays["rhs_pre_sweep"] = np.array(
            arrays["rhs_pre_sweep"], copy=True)
        arrays["rhs_pre_sweep"][i, j, k0 + 1] += np.float64(1.0)
    elif plant == "prandtl":
        wet = np.argwhere(arrays["wmask"][:, :, 1:head["jpkm1"]] != 0.0)
        require(wet.size != 0, "Prandtl plant found no wet recorded interface")
        i, j, k0 = wet[0]
        arrays["pdlr"] = np.array(arrays["pdlr"], copy=True)
        arrays["pdlr"][i, j, k0 + 1] = np.nextafter(
            arrays["pdlr"][i, j, k0 + 1], np.float64(np.inf))
    for name, value in arrays.items():
        require(np.all(np.isfinite(value)), f"{name} contains NaN/Inf")
    require(arrays["rn_Dt"] == 14400.0, f"wrong rn_Dt {arrays['rn_Dt']}")
    expected = {
        "rn_ediff": 0.1, "rn_ediss": 0.7, "rn_ebb": 67.83,
        "rn_emin": 1.0e-6, "rn_emin0": 1.0e-4, "rn_bshear": 1.0e-20,
        "rn_lc": 0.15, "nn_pdl": 1.0, "nn_mxl": 3.0,
        "ln_mxl0": 1.0, "nn_etau": 0.0, "nn_htau": 1.0,
        "nn_eice": 0.0, "ln_lc": 1.0,
    }
    for name, wanted in expected.items():
        require(np.isclose(arrays[name], wanted, rtol=0.0, atol=1.0e-15),
                f"wrong resolved {name}: {arrays[name]} != {wanted}")
    require(abs(arrays["rmxl_min"] - 0.01) <= 2.0e-18,
            f"wrong derived rmxl_min {arrays['rmxl_min']}")
    require(arrays["rn_mxl0"] == arrays["rmxl_min"],
            "ln_mxl0 did not overwrite rn_mxl0 with rmxl_min")
    for name in ("en_entry", "avm_entry", "avt_entry", "dissl_entry",
                 "en_post_sweep", "mxl_momentum", "mxl_dissipation",
                 "avm_closure", "avt_closure", "dissl_output",
                 "avm_pre_evd", "avt_pre_evd"):
        require(np.min(arrays[name]) >= 0.0, f"{name} has a negative value")
    # zdfphy copies closure Kz on the consumed interior before EVD.  This is
    # the acquisition's reason for instrumenting zdfphy as well as zdftke.
    ii = slice(head["ntsi"] - 1, head["ntei"])
    jj = slice(head["ntsj"] - 1, head["ntej"])
    kk = slice(1, head["jpkm1"])
    for name in ("avm", "avt"):
        require(np.array_equal(arrays[f"{name}_pre_evd"][ii, jj, kk],
                               arrays[f"{name}_closure"][ii, jj, kk]),
                f"zdfphy pre-EVD {name} is not a bit copy of closure {name}")
    calibration = _calibrate_en_and_mixing(arrays, head)
    calibration.update(_calibrate_closure(arrays, head))
    return {"header": head, "arrays": arrays, "calibration": calibration}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--producer-commit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--walk", action="store_true",
        help="run the preregistered source-ordered legoESM closure walk")
    parser.add_argument("--plant", choices=(
        "header", "truncation", "nan", "config", "copy", "shape",
        "sweep", "prandtl", "stamp", "walk"))
    args = parser.parse_args(argv)
    producer = None
    try:
        expected = args.expect_commit.lower()
        producer = args.producer_commit.read_text().strip().lower()
        if args.plant == "stamp":
            producer = "0" * 40
        require(len(expected) == 40 and producer == expected,
                f"producer stamp mismatch: {producer} != {expected}")
        rec = read_record(args.record, plant=args.plant)
        if args.plant == "walk":
            require(args.walk, "walk plant requires --walk")
            wet = np.argwhere(
                rec["arrays"]["wmask"][:, :, 1:rec["header"]["jpkm1"]]
                != 0.0)
            require(wet.size != 0, "walk plant found no consumed wet interface")
            i, j, k0 = wet[0]
            planted = np.array(rec["arrays"]["avt_closure"], copy=True)
            planted[i, j, k0 + 1] = np.nextafter(
                planted[i, j, k0 + 1], np.float64(np.inf))
            rec["arrays"]["avt_closure"] = planted
        walk = (_model_substitution_walk(rec["arrays"], rec["header"])
                if args.walk else None)
        summary = {}
        for name, value in rec["arrays"].items():
            arr = np.asarray(value)
            summary[name] = {
                "shape": list(arr.shape),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        report = {
            "format": "gyre-round56-tke-operands-v2",
            "worktree": worktree_stamp(),
            "producer_commit": producer,
            "record": str(args.record),
            "header": rec["header"],
            "calibration": rec["calibration"],
            "model_substitution_walk": walk,
            "fields": summary,
            "plant": args.plant,
            "status": "PASS",
        }
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.write_text(payload)
        print(payload, end="")
        return 0
    except (GateError, OSError, UnicodeDecodeError, struct.error, ValueError) as exc:
        if args.output is not None and args.plant is None:
            failure = {
                "error": str(exc),
                "gate_worktree": worktree_stamp(),
                "producer_commit": producer,
                "record": str(args.record),
                "status": "FAIL",
            }
            args.output.write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(f"STATUS FAIL: {exc}")
        return 1


def _model_substitution_walk(arrays: dict, head: dict) -> dict:
    """Run the frozen source-ordered closure walk through production code.

    NEMO records arrays in (x,y,k); legoESM owns (y,x,k).  Rows after the
    matrix substitution deliberately call the shared production
    ``compute_mixing_lengths``/``compute_K_from_tke`` functions: this is not a
    detached transcription of the candidate statement.
    """
    import jax
    import jax.numpy as jnp

    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_gyre_zco_card
    from legoesm.ocean.physics.vertical_mixing.tke import (
        TKEEntryN2Bundle,
        _mxl0_surface_anchor,
        compute_K_from_tke,
        compute_mixing_lengths,
        tke_vertical_mixing,
    )

    require(jax.config.x64_enabled, "model substitution walk requires JAX fp64")
    jpkm1 = head["jpkm1"]

    def yx(name: str) -> np.ndarray:
        value = np.asarray(arrays[name], dtype=np.float64)
        return np.transpose(value, (1, 0, 2)) if value.ndim == 3 else value.T

    card = build_gyre_zco_card()
    require(card.case == "GYRE-zco", f"wrong instantiated card {card.case!r}")
    cfg = card.recipe.model_config.physics.vertical_mixing.tke
    require(cfg.tke_mxl_choice == 3, "GYRE card no longer selects nn_mxl=3")
    require(cfg.tke_mxl_raw_evaluation in ("factored", "nemo_literal"),
            f"unknown instantiated raw evaluation {cfg.tke_mxl_raw_evaluation!r}")
    const = card.recipe.model_config.constants

    # The consumed zdfphy copy is jk=2..jpkm1: 29 W rows x 600 wet columns.
    wet = yx("wmask")[..., 1:jpkm1] != 0.0
    oracle_avt = yx("avt_pre_evd")[..., 1:jpkm1]
    require(int(np.count_nonzero(wet)) == 17400,
            f"unexpected zdfphy wet-copy census {np.count_nonzero(wet)}")

    e_post = jnp.asarray(yx("en_post_sweep")[..., 1:jpkm1])
    rn2 = jnp.asarray(yx("rn2")[..., 1:jpkm1])
    rn2b = jnp.asarray(yx("rn2b")[..., 1:jpkm1])
    sh2 = jnp.asarray(yx("sh2")[..., 1:jpkm1])
    e3w = jnp.asarray(yx("e3w_Kmm")[..., 1:jpkm1])
    e3t_full = jnp.asarray(yx("e3t_Kmm"))
    taum = jnp.asarray(yx("taum_entry"))
    tmask_surface = jnp.asarray(yx("tmask")[..., 0])
    wmask = jnp.asarray(yx("wmask")[..., 1:jpkm1])
    avm_entry = jnp.asarray(yx("avm_entry")[..., 1:jpkm1])
    avt_entry = jnp.asarray(yx("avt_entry")[..., 1:jpkm1])
    dissl_entry = jnp.asarray(yx("dissl_entry")[..., 1:jpkm1])
    avm_surface = jnp.asarray(yx("avm_entry")[..., 0])

    anchor = _mxl0_surface_anchor(
        cfg, taum, float(const.rho_0), float(const.g), tmask_surface)

    def close_from_en(en_value, l_value=None):
        if l_value is None:
            l_value, _ = compute_mixing_lengths(
                en_value, rn2, e3w, cfg, signed_n2=True,
                dz_cell=e3t_full, l_surface_anchor=anchor)
        return compute_K_from_tke(
            en_value, l_value, cfg, N2=rn2, shear_sq=jnp.zeros_like(sh2),
            N2_prandtl=rn2b, p_sh2_override=lambda _: sh2,
            prandtl_K_M=avm_entry)[1]

    # Rule 10: invoke the public production orchestrator with the printed card
    # configuration.  The record supplies every carried closure operand; only
    # the source routines themselves remain legoESM code.
    z = card.recipe.z_coord
    jacobian = e3t_full[..., :jpkm1] / jnp.asarray(z.dz_ref)
    # z-star has one horizontal Jacobian. Fail closed if the record does not.
    require(np.allclose(
        np.asarray(jacobian),
        np.asarray(jacobian[..., :1]) + np.zeros(np.shape(jacobian)),
        rtol=2.0 * np.finfo(np.float64).eps, atol=0.0),
        "recorded e3t is not a single z-star column Jacobian")
    jac2 = jacobian[..., 0]
    gdepw = (-jnp.asarray(z.z_half_ref[1:jpkm1])
             * jac2[..., None])
    bundle = TKEEntryN2Bundle(
        rn2=rn2, rn2b=rn2b, gdepw_Kmm=gdepw, e3w_Kmm=e3w,
        e3t_Kmm=e3t_full[..., :jpkm1])
    shape30 = e_post.shape[:-1] + (jpkm1,)
    dummy = jnp.zeros(shape30, dtype=e_post.dtype)
    full = jax.jit(lambda: tke_vertical_mixing(
        dummy, dummy, dummy, dummy, dummy, e3w,
        jnp.asarray(yx("en_entry")[..., 1:jpkm1]), None, None,
        float(arrays["rn_Dt"]), cfg, rho_0=float(const.rho_0),
        g=float(const.g), n_iterations=1, taum_surface=taum,
        dz_ref=jnp.asarray(z.dz_ref), jacobian=jac2,
        z_interface=jnp.asarray(z.z_half_ref[1:jpkm1]),
        dz_surface=-jnp.asarray(z.z_full_ref[0]) * jac2,
        surface_tmask=tmask_surface, w_active=wmask,
        preclosure_K_M=avm_entry, preclosure_K_H=avt_entry,
        preclosure_K_M_surface=avm_surface,
        preclosure_dissl=dissl_entry, precomputed_p_sh2=sh2,
        precomputed_n2_bundle=bundle))().K_H

    # Cumulative rows. The early model row includes NEMO's recorded carried
    # production/buoyancy/dissipation operands. The next row substitutes the
    # recorded matrix/RHS/sweep/floor result, then source-order downstream
    # operands are substituted one family at a time.
    model_after_solve = close_from_en(e_post)
    model_after_surface = close_from_en(e_post)
    nemo_l = jnp.asarray(yx("mxl_momentum")[..., 1:jpkm1])
    model_after_mxl = close_from_en(e_post, nemo_l)
    zsqen = np.sqrt(np.asarray(e_post))
    zav = np.float64(arrays["rn_ediff"]) * np.asarray(nemo_l) * zsqen
    avt_base = np.maximum(zav, yx("avt_floor")[..., 1:jpkm1]) * np.asarray(wmask)
    after_prandtl = np.maximum(
        yx("pdlr")[..., 1:jpkm1] * avt_base,
        yx("avt_floor")[..., 1:jpkm1]) * np.asarray(wmask)

    candidates = (
        ("carried_production_buoyancy_dissipation", np.asarray(full)),
        ("matrix_rhs_sweep_en_floor", np.asarray(model_after_solve)),
        ("surface_rn_ebb_taum_and_taum_tmask", np.asarray(model_after_surface)),
        ("raw_mixing_length_derived_floor_and_bounds", np.asarray(model_after_mxl)),
        ("prandtl_factor", after_prandtl),
        ("avt_derivation", yx("avt_closure")[..., 1:jpkm1]),
        ("zdfphy_copy_before_evd", yx("avt_pre_evd")[..., 1:jpkm1]),
    )
    rows = []
    for name, actual in candidates:
        unequal = _unequal_bits(actual, oracle_avt, wet)
        rows.append({
            "name": name,
            "unequal": unequal,
            "wet_cells": int(np.count_nonzero(wet)),
            "max_abs": float(np.max(np.abs(actual[wet] - oracle_avt[wet]))),
            "exact": unequal == 0,
        })
    exact_rows = [row["name"] for row in rows if row["exact"]]
    require(exact_rows, "no cumulative substitution row made avt bit-exact")
    first_exact = exact_rows[0]
    expected_first = ("prandtl_factor"
                      if cfg.tke_mxl_raw_evaluation == "factored"
                      else "matrix_rhs_sweep_en_floor")
    require(first_exact == expected_first,
            "frozen first-exact prediction refuted: got " + first_exact)
    first_index = next(i for i, row in enumerate(rows) if row["exact"])
    require(all(row["exact"] for row in rows[first_index:]),
            "a downstream substitution lost exactness after the first exact row")
    return {
        "card": card.case,
        "instantiated_tke_config": {
            "tke_mxl_choice": cfg.tke_mxl_choice,
            "tke_mxl_raw_evaluation": cfg.tke_mxl_raw_evaluation,
            "mxl_min_effective": float(np.float64(arrays["rmxl_min"])),
            "kappa_convention": cfg.kappa_convention,
            "prandtl_mode": cfg.prandtl_mode,
            "tke_matrix_evaluation": cfg.tke_matrix_evaluation,
            "tke_solver_evaluation": cfg.tke_solver_evaluation,
        },
        "wet_copy_cells": int(np.count_nonzero(wet)),
        "rows": rows,
        "first_exact": first_exact,
        "status": "PASS",
    }


if __name__ == "__main__":
    raise SystemExit(main())
