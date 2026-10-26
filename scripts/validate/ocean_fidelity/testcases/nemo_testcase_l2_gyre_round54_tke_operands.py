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


def read_record(
    path: Path,
    *,
    plant: str | None = None,
    expected_kt: int = 2,
    expected_slots: tuple[int, int] = (3, 3),
) -> dict:
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
    if plant == "slot":
        head["Kbb"] = 1
    expected_clock = (2, expected_kt, *expected_slots)
    require(
        (head["version"], head["kt"], head["Kbb"], head["Kmm"])
        == expected_clock,
        "TKE record slot mismatch: expected "
        f"version=2/kt={expected_kt}/Kbb={expected_slots[0]}/"
        f"Kmm={expected_slots[1]}, "
        f"got version={head['version']}/kt={head['kt']}/"
        f"Kbb={head['Kbb']}/Kmm={head['Kmm']}",
    )
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


def _operand_score(actual: np.ndarray, expected: np.ndarray,
                   mask: np.ndarray) -> dict:
    """Bit census for one carried operand on its consumed slots."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    mask = np.asarray(mask, dtype=bool)
    require(actual.shape == expected.shape == mask.shape,
            f"operand shapes disagree: {actual.shape}, {expected.shape}, "
            f"{mask.shape}")
    finite = mask & np.isfinite(actual) & np.isfinite(expected)
    require(np.array_equal(finite, mask),
            "operand has NaN/Inf on a consumed slot")
    unequal = _unequal_bits(actual, expected, mask)
    return {
        "compared_cells": int(np.count_nonzero(mask)),
        "unequal": unequal,
        "max_abs": (float(np.max(np.abs(actual[mask] - expected[mask])))
                    if np.any(mask) else 0.0),
        "exact": unequal == 0,
    }


def _positive_ulp_distance(actual: np.ndarray, expected: np.ndarray,
                           mask: np.ndarray) -> int:
    """Maximum binary64 encoding distance for finite non-negative K_H."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(actual.shape == expected.shape == mask.shape,
            "K_H ULP-distance shapes disagree")
    require(np.all(np.isfinite(actual[mask]))
            and np.all(np.isfinite(expected[mask]))
            and np.all(actual[mask] >= 0.0)
            and np.all(expected[mask] >= 0.0),
            "K_H ULP distance requires finite non-negative values")
    actual_bits = actual[mask].view(np.uint64)
    expected_bits = expected[mask].view(np.uint64)
    distance = np.maximum(actual_bits, expected_bits) - np.minimum(
        actual_bits, expected_bits)
    return int(np.max(distance)) if distance.size else 0


def _nemo_raw_mixing_length(
    energy: np.ndarray, n2: np.ndarray, floor: np.float64,
) -> np.ndarray:
    """Evaluate compiled ``zdftke.f90:627-630`` in binary64 order."""
    energy = np.asarray(energy, dtype=np.float64)
    n2 = np.asarray(n2, dtype=np.float64)
    rsmall = np.float64(0.5) * np.finfo(np.float64).eps
    zrn2 = np.maximum(n2, rsmall)
    return np.maximum(
        np.sqrt((np.float64(2.0) * energy) / zrn2),
        np.float64(floor),
    )


def _plant_entry_ulp_at_changed_output(
    entry: np.ndarray, baseline: np.ndarray, bumped: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Raise one entry value by one ULP where the measured output responds."""
    entry = np.asarray(entry, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    bumped = np.asarray(bumped, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(entry.shape == baseline.shape == bumped.shape == mask.shape,
            "K_H entry/output plant shapes disagree")
    mobile = mask & (baseline.view(np.uint64) != bumped.view(np.uint64))
    require(np.any(mobile),
            "no consumed K_H output responds when every TKE entry cell is "
            "raised by one input ULP")
    # Prefer the largest responsive carry.  This maximizes the absolute size
    # of a one-ULP input violation while remaining exactly one binary64 ULP.
    selected = int(np.argmax(np.where(mobile, entry, -np.inf)))
    index = tuple(int(value) for value in np.unravel_index(
        selected, entry.shape))
    planted = entry.copy()
    planted[index] = np.nextafter(entry[index], np.float64(np.inf))
    return planted, index


ISOLATED_EAGER_LABEL = "isolated-closure eager"
ISOLATED_JIT_LABEL = "isolated-closure JIT"
PRODUCTION_STEP_LABEL = "recorded-entry production step (_step_jitted)"
PRODUCTION_INJECTION_LABEL = (
    "recorded-entry production step with NEMO en_post_sweep injection "
    "(_step_jitted)"
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--producer-commit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stage-root", type=Path,
        help=("round-46/93+ stage-twin root carrying the NEMO-recorded "
              "production entry, velocities, and face metrics"))
    parser.add_argument(
        "--walk", action="store_true",
        help="run the preregistered source-ordered legoESM closure walk")
    parser.add_argument("--plant", choices=(
        "header", "truncation", "nan", "config", "copy", "shape",
        "sweep", "prandtl", "stamp", "walk", "operand", "kh-entry-ulp",
        "slot"))
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
        if args.plant == "operand":
            require(args.walk, "operand plant requires --walk")
        if args.plant == "kh-entry-ulp":
            require(args.walk, "K_H entry-ULP plant requires --walk")
        if args.plant == "slot":
            require(args.walk, "slot plant requires --walk")
        walk = (_model_substitution_walk(
                    rec["arrays"], rec["header"],
                    plant_operand=args.plant == "operand",
                    plant_kh_entry=args.plant == "kh-entry-ulp",
                    plant_slot=args.plant == "slot",
                    stage_root=args.stage_root)
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
            "format": "gyre-round56-tke-operands-v5",
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
    except (GateError, RuntimeError, OSError, UnicodeDecodeError,
            struct.error, ValueError) as exc:
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


def _model_substitution_walk(
    arrays: dict, head: dict, *, plant_operand: bool = False,
    plant_kh_entry: bool = False,
    plant_slot: bool = False,
    stage_root: Path | None = None,
) -> dict:
    """Run the frozen source-ordered closure walk through production code.

    NEMO records arrays in (x,y,k); legoESM owns (y,x,k).  Rows after the
    matrix substitution deliberately call the shared production
    ``compute_mixing_lengths``/``compute_K_from_tke`` functions: this is not a
    detached transcription of the candidate statement.
    """
    import jax
    import jax.numpy as jnp
    import importlib.util

    from legoesm.core.precision import (
        PrecisionPolicy,
        get_policy,
        set_policy,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_gyre_zco_card
    from legoesm.ocean.physics.vertical_mixing.tke import (
        TKEEntryN2Bundle,
        _mxl0_surface_anchor,
        _safe_stress_modulus,
        _tke_raw_mixing_length,
        compute_K_from_tke,
        compute_mixing_lengths,
        tke_vertical_mixing,
    )
    import legoesm.ocean.physics.vertical_mixing.tke as tke_module
    from legoesm.ocean.vertical import compute_ocean_jacobian

    require(jax.config.x64_enabled, "model substitution walk requires JAX fp64")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "model substitution walk did not resolve fp64/libm")
    jpkm1 = head["jpkm1"]
    stage_gate_module = None

    def stage_twin_gate():
        """Load the existing round-93+ recorded-entry stage-twin gate."""
        nonlocal stage_gate_module
        if stage_gate_module is None:
            reader_path = Path(__file__).with_name(
                "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py")
            reader_spec = importlib.util.spec_from_file_location(
                "round54_round46_stage_twin", reader_path)
            require(reader_spec is not None and reader_spec.loader is not None,
                    f"cannot load stamped stage twin {reader_path}")
            stage_gate_module = importlib.util.module_from_spec(reader_spec)
            reader_spec.loader.exec_module(stage_gate_module)
        return stage_gate_module

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

    # Instantiate the same production model and forcing helper as the
    # certified kt=1..10 ladder.  One ordinary cold-start step produces the
    # exact legoESM state handed to kt=2; no record field is injected here.
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    forcing_path = Path(__file__).with_name(
        "nemo_testcase_l2_gyre_phase3_gate.py")
    spec = importlib.util.spec_from_file_location(
        "round61_gyre_phase3_forcing", forcing_path)
    require(spec is not None and spec.loader is not None,
            f"cannot load certified forcing helper {forcing_path}")
    forcing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(forcing_module)
    state1 = card.recipe.initial_state
    freshwater1, surface1 = forcing_module._surface_forcings(card, state1, 1)
    state2 = model.step(
        state1, dt=card.dt_s, freshwater=freshwater1,
        surface_forcing=surface1)
    require(all(getattr(state2, name, None) is not None for name in (
        "tke", "tke_avm", "tke_avt", "tke_avm_surface", "tke_dissl")),
        "kt=1 exit did not populate the complete TKE carry")
    _, surface2 = forcing_module._surface_forcings(card, state2, 2)
    model_bundle = model._tke_step_entry_n2_bundle(state2)
    model_sh2 = model._tke_step_entry_p_sh2(state2)
    require(model_bundle is not None and model_sh2 is not None,
            "kt=2 production operands were not materialized")
    model_taum = (jnp.maximum(jnp.asarray(surface2.taum), 0.0)
                  if surface2.taum is not None else
                  _safe_stress_modulus(surface2.tau_x, surface2.tau_y))

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

    model_wmask = (jnp.asarray(card.recipe.z_coord.is_active[..., 1:])
                   & (state2.land_mask.data[..., None] > 0.5))
    model_tmask = (jnp.asarray(card.recipe.z_coord.is_active)
                   & (state2.land_mask.data[..., None] > 0.5))
    model_avm_full = jnp.concatenate(
        [state2.tke_avm_surface.data[..., None], state2.tke_avm.data], axis=-1)
    oracle_operands = {
        "sh2": sh2,
        "rn2": rn2,
        "rn2b": rn2b,
        "entry_en": jnp.asarray(yx("en_entry")[..., 1:jpkm1]),
        "entry_avm": jnp.asarray(yx("avm_entry")[..., :jpkm1]),
        "entry_avt": avt_entry,
        "entry_dissl": dissl_entry,
        "taum": taum,
        "e3t_Kmm": jnp.asarray(yx("e3t_Kmm")[..., :jpkm1]),
        "e3w_Kmm": e3w,
        "tmask": jnp.asarray(yx("tmask")[..., :jpkm1]),
        "wmask": wmask,
    }
    model_operands = {
        "sh2": jnp.asarray(model_sh2),
        "rn2": jnp.asarray(model_bundle.rn2),
        "rn2b": jnp.asarray(model_bundle.rn2b),
        "entry_en": jnp.asarray(state2.tke.data),
        "entry_avm": model_avm_full,
        "entry_avt": jnp.asarray(state2.tke_avt.data),
        "entry_dissl": jnp.asarray(state2.tke_dissl.data),
        "taum": jnp.asarray(model_taum),
        "e3t_Kmm": jnp.asarray(model_bundle.e3t_Kmm),
        "e3w_Kmm": jnp.asarray(model_bundle.e3w_Kmm),
        "tmask": model_tmask.astype(jnp.float64),
        "wmask": model_wmask.astype(jnp.float64),
    }
    if plant_operand:
        planted_e3t = np.asarray(model_operands["e3t_Kmm"]).copy()
        first_wet = tuple(np.argwhere(
            np.asarray(oracle_operands["tmask"]) != 0.0)[0])
        planted_e3t[first_wet] = np.nextafter(
            planted_e3t[first_wet], np.float64(np.inf))
        model_operands["e3t_Kmm"] = jnp.asarray(planted_e3t)
    # Numerical fields are scored on NEMO's consumed wet slots.  The masks
    # themselves are scored over every physical cropped slot because a dry
    # discrepancy changes coast averaging/control flow.
    operand_masks = {}
    for name in oracle_operands:
        if name in ("tmask", "wmask"):
            mask = np.ones(np.shape(oracle_operands[name]), dtype=bool)
        elif name == "e3t_Kmm":
            mask = np.asarray(oracle_operands["tmask"]) != 0.0
        elif name == "entry_avm":
            mask = np.asarray(yx("wmask")[..., :jpkm1]) != 0.0
        elif name == "taum":
            mask = np.asarray(oracle_operands["tmask"])[..., 0] != 0.0
        else:
            mask = np.asarray(oracle_operands["wmask"]) != 0.0
        operand_masks[name] = mask
    operand_rows = {
        name: _operand_score(
            np.asarray(model_operands[name]),
            np.asarray(oracle_operands[name]), operand_masks[name])
        for name in oracle_operands
    }
    operand_rows["masks"] = {
        "compared_cells": (
            operand_rows["tmask"]["compared_cells"]
            + operand_rows["wmask"]["compared_cells"]),
        "unequal": (operand_rows["tmask"]["unequal"]
                    + operand_rows["wmask"]["unequal"]),
        "max_abs": max(operand_rows["tmask"]["max_abs"],
                       operand_rows["wmask"]["max_abs"]),
        "exact": (operand_rows["tmask"]["exact"]
                  and operand_rows["wmask"]["exact"]),
    }
    require(not plant_operand,
            "planted kt=2 e3t_Kmm operand violation detected: "
            f"{operand_rows['e3t_Kmm']['unequal']} unequal")

    sh2_velocity_attribution = None
    if stage_root is not None:
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            compute_face_masks_3d,
        )
        from legoesm.ocean.physics.vertical_mixing._shared import (
            avm_weighted_shear_production,
        )

        reader_module = stage_twin_gate()
        stage_path = stage_root / "oracle_momstage_kt00000002_s1.bin"
        stage = reader_module.read_stage(
            stage_path, expected_kt=2, expected_stage=1)["arrays"]

        def owned3(name):
            return np.asarray(stage[name])[2:-2, 2:-2, :jpkm1]

        def uface(name, *, metric=False):
            value = owned3(name)
            if metric:
                value = np.asarray(stage[name])[
                    2:-2, 2:-2, 1:jpkm1]
            return jnp.asarray(np.concatenate([value[:, -1:, :], value], axis=1))

        def vface(name, *, metric=False):
            value = owned3(name)
            if metric:
                value = np.asarray(stage[name])[
                    2:-2, 2:-2, 1:jpkm1]
            pad = np.ones_like(value[:1]) if metric else np.zeros_like(value[:1])
            return jnp.asarray(np.concatenate([pad, value], axis=0))

        u_mask, v_mask = compute_face_masks_3d(card.recipe.z_coord.is_active)
        # Compiled zdfsh2 expands the qco macro as
        # e3w_1d(jk)*(1+r3u/r3v(time-level)); it does not consume e3u/e3v.
        e3w_ref = np.asarray(stage["e3w_0"])[
            2:-2, 2:-2, 1:jpkm1]
        ref_u = np.concatenate([e3w_ref[:, -1:, :], e3w_ref], axis=1)
        ref_v = np.concatenate([e3w_ref[:1, :, :], e3w_ref], axis=0)

        def ur3(name):
            value = np.asarray(stage[name])[2:-2, 2:-2]
            return np.concatenate([value[:, -1:], value], axis=1)

        def vr3(name):
            value = np.asarray(stage[name])[2:-2, 2:-2]
            return np.concatenate([np.zeros_like(value[:1]), value], axis=0)

        face_metrics = tuple(jnp.asarray(value) for value in (
            ref_u * (1.0 + ur3("r3u_Kmm"))[..., None],
            ref_u * (1.0 + ur3("r3u_Kbb"))[..., None],
            ref_v * (1.0 + vr3("r3v_Kmm"))[..., None],
            ref_v * (1.0 + vr3("r3v_Kbb"))[..., None],
        ))
        dummy_dz = jnp.ones_like(sh2)

        def face_sh2(un, vn, ub, vb):
            return avm_weighted_shear_production(
                un, vn, ub, vb, dummy_dz, u_mask, v_mask, avm_entry,
                face_metrics=face_metrics)

        nemo_velocity_sh2 = face_sh2(
            uface("u_Kmm"), vface("v_Kmm"),
            uface("u_Kbb"), vface("v_Kbb"))
        model_now_velocity_sh2 = face_sh2(
            state2.u.data, state2.v.data,
            uface("u_Kbb"), vface("v_Kbb"))
        model_same_level_sh2 = face_sh2(
            state2.u.data, state2.v.data,
            state2.u.data, state2.v.data)
        sh2_velocity_attribution = {
            "stage_record": str(stage_path),
            "nemo_velocities": _operand_score(
                np.asarray(nemo_velocity_sh2), np.asarray(sh2),
                operand_masks["sh2"]),
            "model_Kmm_with_nemo_Kbb": _operand_score(
                np.asarray(model_now_velocity_sh2), np.asarray(sh2),
                operand_masks["sh2"]),
            "model_Kmm_used_for_both_levels": _operand_score(
                np.asarray(model_same_level_sh2), np.asarray(sh2),
                operand_masks["sh2"]),
            "model_selected_statement": operand_rows["sh2"],
            "model_has_Kbb_face_velocity_carry": (
                state2.u_before is not None and state2.v_before is not None),
        }

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

    # These two rows intentionally remain isolated-closure diagnostics.  The
    # separate production row below drives the recorded-entry stage twin all
    # the way through ``LatLonCGridOceanModel.step`` / ``_step_jitted``.
    jitted_close_from_en = jax.jit(close_from_en)
    oracle_mxl = np.asarray(yx("mxl_momentum")[..., 1:jpkm1])
    oracle_mxld = np.asarray(yx("mxl_dissipation")[..., 1:jpkm1])
    oracle_avm = np.asarray(yx("avm_closure")[..., 1:jpkm1])
    oracle_dissl = np.asarray(yx("dissl_output")[..., 1:jpkm1])
    oracle_floor_m = np.asarray(yx("avm_floor")[..., 1:jpkm1])
    oracle_floor_h = np.asarray(yx("avt_floor")[..., 1:jpkm1])
    source_raw_mxl = _nemo_raw_mixing_length(
        np.asarray(e_post), np.asarray(rn2),
        np.float64(arrays["rmxl_min"]))
    source_sqrt_en = np.sqrt(np.asarray(e_post))
    source_zav = ((np.float64(arrays["rn_ediff"]) * oracle_mxl)
                  * source_sqrt_en)
    source_raw_avt = (np.maximum(source_zav, oracle_floor_h)
                      * np.asarray(wmask))
    active_zav = wet & (oracle_avm > oracle_floor_m)

    def kh_statement_outputs(energy_value):
        """Materialize the executing ``tke_avn`` statement boundaries."""
        energy_value = jnp.asarray(energy_value)
        raw_mxl = _tke_raw_mixing_length(energy_value, rn2, cfg)
        bounded_mxl, bounded_mxld = compute_mixing_lengths(
            energy_value, rn2, e3w, cfg, signed_n2=True,
            dz_cell=e3t_full, l_surface_anchor=anchor)
        sqrt_en = jnp.sqrt(energy_value)
        raw_zav = ((jnp.asarray(cfg.c_k, dtype=energy_value.dtype)
                    * bounded_mxl)
                   * jnp.where(
                       energy_value > 0.0,
                       jnp.sqrt(jnp.where(
                           energy_value > 0.0, energy_value, 1.0)),
                       0.0))
        raw_avt = (jnp.maximum(raw_zav, jnp.asarray(
            cfg.kappaH_min, dtype=raw_zav.dtype)) * wmask)
        k_m, k_h = compute_K_from_tke(
            energy_value, bounded_mxl, cfg, N2=rn2,
            shear_sq=jnp.zeros_like(sh2), N2_prandtl=rn2b,
            p_sh2_override=lambda _: sh2, prandtl_K_M=avm_entry)
        dissl = sqrt_en / bounded_mxld
        isolated_k_h = close_from_en(
            energy_value, jnp.asarray(oracle_mxl))
        return {
            "raw_buoyancy_length": raw_mxl,
            "bounded_mixing_length": bounded_mxl,
            "sqrt_en": sqrt_en,
            "raw_zav": raw_zav,
            "avm_floor_and_mask": k_m,
            "avt_floor_and_mask_before_prandtl": raw_avt,
            "dissipation_length_output": dissl,
            "avt_inverse_prandtl_update": k_h,
            "isolated_downstream": isolated_k_h,
        }

    jitted_kh_statement_outputs = jax.jit(kh_statement_outputs)

    def score_kh_execution(outputs, whole_closure_k_h):
        """Score one execution mode without changing its arithmetic graph."""
        rows = [
            {
                "name": "surface_mixing_length",
                "citation": "zdftke.f90:589,612-620",
                **_operand_score(
                    np.asarray(anchor),
                    np.asarray(yx("mxl_momentum")[..., 0]),
                    np.asarray(yx("wmask")[..., 0]) != 0.0),
            },
            {
                "name": "raw_buoyancy_length",
                "citation": "zdftke.f90:627-630",
                **_operand_score(
                    np.asarray(outputs["raw_buoyancy_length"]),
                    source_raw_mxl, wet),
            },
            {
                "name": "bounded_mixing_length",
                "citation": "zdftke.f90:634-683",
                **_operand_score(
                    np.asarray(outputs["bounded_mixing_length"]),
                    oracle_mxl, wet),
            },
            {
                "name": "sqrt_en",
                "citation": "zdftke.f90:691",
                **_operand_score(
                    np.asarray(outputs["sqrt_en"]), source_sqrt_en, wet),
            },
            {
                "name": "raw_zav",
                "citation": "zdftke.f90:692",
                **_operand_score(
                    np.asarray(outputs["raw_zav"]),
                    oracle_avm, active_zav),
            },
            {
                "name": "avm_floor_and_mask",
                "citation": "zdftke.f90:693",
                **_operand_score(
                    np.asarray(outputs["avm_floor_and_mask"]),
                    oracle_avm, wet),
            },
            {
                "name": "avt_floor_and_mask_before_prandtl",
                "citation": "zdftke.f90:694",
                **_operand_score(
                    np.asarray(outputs[
                        "avt_floor_and_mask_before_prandtl"]),
                    source_raw_avt, wet),
            },
            {
                "name": "dissipation_length_output",
                "citation": "zdftke.f90:695",
                **_operand_score(
                    np.asarray(outputs["dissipation_length_output"]),
                    oracle_dissl, wet),
            },
            {
                "name": "avt_inverse_prandtl_update",
                "citation": "zdftke.f90:699-702",
                **_operand_score(
                    np.asarray(outputs["avt_inverse_prandtl_update"]),
                    oracle_avt, wet),
            },
        ]
        isolated = {
            "name": "avt_inverse_prandtl_update_with_recorded_prior_rows",
            "citation": "zdftke.f90:699-702",
            **_operand_score(
                np.asarray(outputs["isolated_downstream"]),
                oracle_avt, wet),
        }
        source_order = {
            "name": "source_order_closure_k_h",
            "citation": "zdftke.f90:627-702",
            **_operand_score(
                np.asarray(whole_closure_k_h), oracle_avt, wet),
        }
        first = next((row for row in rows if not row["exact"]), None)
        return {
            "rows": rows,
            "first_non_bit": (None if first is None else {
                "name": first["name"],
                "citation": first["citation"],
                "unequal": first["unequal"],
                "max_abs": first["max_abs"],
            }),
            "isolated_downstream": isolated,
            "whole_closure_row": source_order,
            "all_statement_rows_exact": all(row["exact"] for row in rows),
        }

    eager_statement_outputs = kh_statement_outputs(e_post)
    jit_statement_outputs = jitted_kh_statement_outputs(e_post)
    eager_whole_closure = close_from_en(e_post)
    jit_whole_closure = jitted_close_from_en(e_post)

    production_records = None

    def production_step_k_h(entry_ulp=None, post_sweep=None):
        """Extract K_H from the existing recorded-entry production-step twin."""
        nonlocal production_records
        gate = stage_twin_gate()
        root = Path(stage_root) if stage_root is not None else gate.ROOT
        if production_records is None:
            production_records = {
                (kt, stage): gate.read_stage(
                    root / f"oracle_momstage_kt{kt:08d}_s{stage}.bin",
                    expected_kt=kt, expected_stage=stage,
                    plant=("slot" if plant_slot and (kt, stage) == (2, 1)
                           else None))
                for kt, stage in gate.STAGES
            }
        return gate._stage_twin(
            production_records, root, gate.ADVMEAN_ROOT, gate.MEMORY_ROOT,
            gate.BTSTEP_ROOT, gate.STAGE_CLOSURE_ROOT, None,
            production_tke_only=True,
            production_tke_entry_ulp=entry_ulp,
            production_tke_taum=jnp.asarray(yx("taum_entry")),
            production_tke_post_sweep=post_sweep,
        )

    production_result = production_step_k_h()
    production_candidate = np.asarray(production_result["candidate_k_h"])
    entry_references = {
        "tke": np.asarray(yx("en_entry")[..., 1:jpkm1]),
        "tke_avm": np.asarray(yx("avm_entry")[..., 1:jpkm1]),
        "tke_avt": np.asarray(yx("avt_entry")[..., 1:jpkm1]),
        "tke_dissl": np.asarray(yx("dissl_entry")[..., 1:jpkm1]),
        "tke_avm_surface": np.asarray(yx("avm_entry")[..., 0]),
    }
    entry_masks = {
        name: (np.asarray(yx("wmask")[..., 0]) != 0.0
               if name == "tke_avm_surface" else wet)
        for name in entry_references
    }
    production_entry_identity = {
        name: _operand_score(
            np.asarray(production_result["entry_carry"][name]), reference,
            entry_masks[name])
        for name, reference in entry_references.items()
    }
    require(all(row["exact"] for row in production_entry_identity.values()),
            "round-93+ stage twin does not carry the round-59 NEMO TKE entry "
            f"bit for bit: {production_entry_identity}")
    production_forcing_identity = {
        "taum": _operand_score(
            np.asarray(production_result["forcing_entry"]["taum"]),
            np.asarray(yx("taum_entry")),
            np.asarray(yx("tmask")[..., 0]) != 0.0),
    }
    require(all(row["exact"] for row in production_forcing_identity.values()),
            "recorded-entry production step does not carry kt=2 forcing "
            f"bit for bit: {production_forcing_identity}")
    derived = production_result["derived_entry_operands"]
    production_derived_entry_identity = {
        "rn2": _operand_score(
            np.asarray(derived["rn2"]), np.asarray(yx("rn2")[..., 1:jpkm1]),
            wet),
        "rn2b": _operand_score(
            np.asarray(derived["rn2b"]),
            np.asarray(yx("rn2b")[..., 1:jpkm1]), wet),
        "sh2": _operand_score(
            np.asarray(derived["sh2"]), np.asarray(yx("sh2")[..., 1:jpkm1]),
            wet),
        "e3t_Kmm": _operand_score(
            np.asarray(derived["e3t_Kmm"]),
            np.asarray(yx("e3t_Kmm")[..., :jpkm1]),
            np.asarray(yx("tmask")[..., :jpkm1]) != 0.0),
        "e3w_Kmm": _operand_score(
            np.asarray(derived["e3w_Kmm"]),
            np.asarray(yx("e3w_Kmm")[..., 1:jpkm1]), wet),
    }
    require(all(row["exact"]
                for row in production_derived_entry_identity.values()),
            "derived kt=2 production entry operands are not bit-exact: "
            f"{production_derived_entry_identity}")
    production_kh_upstream = {
        "en_post_sweep": _operand_score(
            np.asarray(production_result["candidate_tke_post_sweep"]),
            np.asarray(e_post), wet),
    }
    full_entry_audit = production_result["entry_state_identity"]
    require(full_entry_audit["all_exact"],
            "kt=2 full recorded-entry audit is not exact")
    full_entry_rows = [
        row for group in full_entry_audit["groups"].values()
        for row in group.values()
    ]
    upstream_unequal = (
        sum(row["unequal"] for row in full_entry_rows)
        + sum(row["unequal"] for row in production_entry_identity.values())
        + sum(row["unequal"] for row in production_forcing_identity.values())
        + sum(row["unequal"]
              for row in production_derived_entry_identity.values())
        + sum(row["unequal"] for row in production_kh_upstream.values()))
    production_kh_score = _operand_score(production_candidate, oracle_avt, wet)
    oracle_scale_ulp = float(np.max(np.abs(np.spacing(oracle_avt[wet]))))
    ulp_scale_bound = 8.0 * oracle_scale_ulp
    injected_result = production_step_k_h(
        post_sweep=np.asarray(e_post, dtype=np.float64))
    require(injected_result["post_sweep_injection"] is not None,
            "production NEMO en_post_sweep intervention was not installed")
    require(injected_result["entry_slots"] == {"Nbb": 3, "Kbb": 3, "Kmm": 3},
            "production NEMO en_post_sweep intervention moved time slots")
    require(injected_result["entry_state_identity"]["all_exact"],
            "production NEMO en_post_sweep intervention moved the entry bridge")
    require(
        injected_result["entry_record"] == production_result["entry_record"]
        and injected_result["entry_state_identity"]["entry_sha256"]
        == production_result["entry_state_identity"]["entry_sha256"],
        "production NEMO en_post_sweep intervention changed the entry record",
    )
    injected_forcing_score = _operand_score(
        np.asarray(injected_result["forcing_entry"]["taum"]),
        np.asarray(yx("taum_entry")),
        np.asarray(yx("tmask")[..., 0]) != 0.0,
    )
    require(injected_forcing_score["exact"],
            "production NEMO en_post_sweep intervention changed kt=2 forcing")
    injected_energy_score = _operand_score(
        np.asarray(injected_result["candidate_tke_post_sweep"]),
        np.asarray(e_post), wet)
    require(injected_energy_score["exact"],
            "NEMO en_post_sweep intervention did not reach the returned "
            f"production TKE slot: {injected_energy_score}")
    injected_candidate = np.asarray(injected_result["candidate_k_h"])
    injected_kh_score = _operand_score(injected_candidate, oracle_avt, wet)
    intervention_effect = _operand_score(
        injected_candidate, production_candidate, wet)
    injection_is_ulp_scale = injected_kh_score["max_abs"] <= ulp_scale_bound
    attribution_status = (
        "MEASURED" if injection_is_ulp_scale else "UNMEASURED")
    remaining_candidate_inputs = ([] if attribution_status == "MEASURED" else [
        "downstream rn2/rn2b consumption (entry snapshots audit exact)",
        "downstream e3t_Kmm/e3w_Kmm and tmask/wmask consumption "
        "(entry snapshots audit exact)",
        "taum surface anchor and surface-boundary inputs "
        "(recorded forcing audits exact)",
        "downstream sh2/p_sh2 and carried tke_avm used by inverse Prandtl "
        "(entry snapshots audit exact)",
        "production mixing-length and inverse-Prandtl source-order arithmetic",
    ])
    causal_attribution = {
        "status": attribution_status,
        "claim": (
            "the production en_post_sweep difference causes the non-ULP "
            "K_H residual" if attribution_status == "MEASURED" else
            "cause unmeasured"),
        "preregistered_decider": (
            "MEASURED iff the same-bridge NEMO en_post_sweep injection makes "
            "the production K_H residual binary64-ULP scale"),
        "ulp_scale_bound": ulp_scale_bound,
        "remaining_candidate_inputs": remaining_candidate_inputs,
    }
    literal_candidate = cfg.tke_mxl_raw_evaluation == "nemo_literal"
    upstream_nonexact = [
        row["name"] for row in full_entry_rows if not row["exact"]
    ] + [
        f"round59.{name}" for name, row in production_entry_identity.items()
        if not row["exact"]
    ] + [
        f"round59.forcing.{name}"
        for name, row in production_forcing_identity.items()
        if not row["exact"]
    ] + [
        f"round59.derived_entry.{name}"
        for name, row in production_derived_entry_identity.items()
        if not row["exact"]
    ] + [
        f"round59.{name}" for name, row in production_kh_upstream.items()
        if not row["exact"]
    ]
    # A correctly aligned binary64 K_H comparison can move at source/JIT
    # rounding scale.  A physical-size residual with no recorded upstream
    # difference is an instrument failure (the fix-round-3 0.059 row).
    if literal_candidate and upstream_unequal == 0:
        require(production_kh_score["max_abs"] <= ulp_scale_bound,
                "physical-range sanity failed: exact kt=2 upstream entry but "
                f"K_H max_abs={production_kh_score['max_abs']} exceeds the "
                f"8-ULP-scale bound {ulp_scale_bound}")
    physical_range_sanity = {
        "expectation": (
            "with an exact recorded kt=2 entry, K_H mismatch must be "
            "binary64-ULP scale; otherwise name the differing upstream input"),
        "oracle_wet_min": float(np.min(oracle_avt[wet])),
        "oracle_wet_max": float(np.max(oracle_avt[wet])),
        "candidate_wet_min": float(np.min(production_candidate[wet])),
        "candidate_wet_max": float(np.max(production_candidate[wet])),
        "max_abs": production_kh_score["max_abs"],
        "max_cell_ulp_distance": _positive_ulp_distance(
            production_candidate, oracle_avt, wet),
        "max_reference_scale_ulp": oracle_scale_ulp,
        "ulp_scale_bound": ulp_scale_bound,
        "upstream_unequal": upstream_unequal,
        "upstream_nonexact": upstream_nonexact,
        "literal_candidate_expected_ulp_scale": literal_candidate,
        "classification": (
            "ULP_SCALE" if production_kh_score["max_abs"] <= ulp_scale_bound
            else "MEASURED_EN_POST_SWEEP_CAUSE"
            if attribution_status == "MEASURED"
            else "CAUSE_UNMEASURED"),
    }
    production_row = {
        "name": PRODUCTION_STEP_LABEL,
        **production_kh_score,
        "target": "NEMO avt_pre_evd",
        "kt": production_result["kt"],
        "stage": production_result["stage"],
        "entry": production_result["entry"],
        "entry_slots": production_result["entry_slots"],
        "entry_record": production_result["entry_record"],
        "forcing_kt": production_result["forcing_kt"],
        "stage_barotropic_handoff_kt": production_result[
            "stage_barotropic_handoff_kt"],
        "output_carry": production_result["output_carry"],
        "extraction_citation": production_result["extraction_citation"],
        "step_citation": production_result["step_citation"],
        "carry_declaration_citation": production_result[
            "carry_declaration_citation"],
        "stage_twin_format": production_result["format"],
        "recorded_entry_identity": production_entry_identity,
        "recorded_forcing_identity": production_forcing_identity,
        "derived_entry_identity": production_derived_entry_identity,
        "k_h_upstream_identity": production_kh_upstream,
        "full_entry_state_identity": full_entry_audit,
        "physical_range_sanity": physical_range_sanity,
        "causal_attribution": causal_attribution,
        "bridge_citation": production_result["bridge_citation"],
        "existing_stage_twin_citation": production_result[
            "existing_stage_twin_citation"],
    }
    production_injection_row = {
        "name": PRODUCTION_INJECTION_LABEL,
        **injected_kh_score,
        "target": "NEMO avt_pre_evd",
        "kt": injected_result["kt"],
        "stage": injected_result["stage"],
        "entry_slots": injected_result["entry_slots"],
        "same_bridge_identity": {
            "bridge_citation": injected_result["bridge_citation"],
            "entry_record": injected_result["entry_record"],
            "entry_sha256": injected_result["entry_state_identity"][
                "entry_sha256"],
            "entry_state_all_exact": injected_result[
                "entry_state_identity"]["all_exact"],
            "forcing_kt": injected_result["forcing_kt"],
            "forcing_identity": injected_forcing_score,
            "stage_barotropic_handoff_kt": injected_result[
                "stage_barotropic_handoff_kt"],
        },
        "injected_input": "NEMO en_post_sweep",
        "injected_input_identity": injected_energy_score,
        "intervention_vs_baseline": intervention_effect,
        "max_cell_ulp_distance": _positive_ulp_distance(
            injected_candidate, oracle_avt, wet),
        "ulp_scale_bound": ulp_scale_bound,
        "classification": (
            "ULP_SCALE" if injection_is_ulp_scale else "NON_ULP_RESIDUAL"),
        "injection": injected_result["post_sweep_injection"],
        "step_citation": injected_result["step_citation"],
        "output_carry": injected_result["output_carry"],
    }
    isolated_eager = score_kh_execution(
        eager_statement_outputs, eager_whole_closure)
    isolated_eager["execution"] = ISOLATED_EAGER_LABEL
    isolated_jit = score_kh_execution(
        jit_statement_outputs, jit_whole_closure)
    isolated_jit["execution"] = ISOLATED_JIT_LABEL
    kh_walk = {
        "isolated_entry": (
            "NEMO en_post_sweep/rn2/e3t/taum and carried coefficients"),
        "production_step_call_site": (
            "packages/ocean/legoesm/ocean/dynamics/"
            "ocean_model_latlon_cgrid.py:11012-11025"),
        ISOLATED_EAGER_LABEL: isolated_eager,
        ISOLATED_JIT_LABEL: isolated_jit,
        PRODUCTION_STEP_LABEL: production_row,
        PRODUCTION_INJECTION_LABEL: production_injection_row,
        "causal_attribution": causal_attribution,
    }
    if plant_kh_entry:
        baseline_k_h = {
            ISOLATED_EAGER_LABEL: np.asarray(eager_whole_closure),
            ISOLATED_JIT_LABEL: np.asarray(jit_whole_closure),
        }
        bumped_entry = np.nextafter(
            np.asarray(e_post), np.float64(np.inf))
        bumped_k_h = {
            ISOLATED_EAGER_LABEL: np.asarray(
                close_from_en(jnp.asarray(bumped_entry))),
            ISOLATED_JIT_LABEL: np.asarray(
                jitted_close_from_en(jnp.asarray(bumped_entry))),
        }
        common_response = np.asarray(wet, dtype=bool).copy()
        for mode in (ISOLATED_EAGER_LABEL, ISOLATED_JIT_LABEL):
            common_response &= (
                baseline_k_h[mode].view(np.uint64)
                != bumped_k_h[mode].view(np.uint64))
        planted_energy, planted_index = _plant_entry_ulp_at_changed_output(
            np.asarray(e_post), baseline_k_h[ISOLATED_EAGER_LABEL],
            bumped_k_h[ISOLATED_EAGER_LABEL], common_response)
        planted_k_h = {
            ISOLATED_EAGER_LABEL: np.asarray(
                close_from_en(jnp.asarray(planted_energy))),
            ISOLATED_JIT_LABEL: np.asarray(
                jitted_close_from_en(jnp.asarray(planted_energy))),
        }
        response_rows = {
            mode: _operand_score(
                planted_k_h[mode], baseline_k_h[mode], wet)
            for mode in (ISOLATED_EAGER_LABEL, ISOLATED_JIT_LABEL)
        }
        production_plant_field = "tke_avm"
        production_plant_entry = np.asarray(
            production_result["entry_carry"][production_plant_field])
        require(production_plant_entry.shape == wet.shape,
                "production K_M entry plant shape does not match wet mask")
        production_plant_index = tuple(int(value) for value in np.unravel_index(
            int(np.argmax(np.where(wet, production_plant_entry, -np.inf))),
            production_plant_entry.shape))
        planted_production = production_step_k_h(
            (production_plant_field, production_plant_index))
        response_rows[PRODUCTION_STEP_LABEL] = _operand_score(
            np.asarray(planted_production["candidate_k_h"]),
            production_candidate, wet)
        require(all(not row["exact"] and row["unequal"] >= 1
                    for row in response_rows.values()),
                "one-ULP TKE entry plant did not change all three K_H rows: "
                f"{response_rows}")
        raise GateError(
            "planted one-ULP closure-entry violations detected: isolated "
            f"TKE at {planted_index} changed "
            f"{response_rows[ISOLATED_EAGER_LABEL]['unequal']} "
            f"{ISOLATED_EAGER_LABEL}, "
            f"{response_rows[ISOLATED_JIT_LABEL]['unequal']} "
            f"{ISOLATED_JIT_LABEL}; production {production_plant_field} at "
            f"{production_plant_index} changed "
            f"{response_rows[PRODUCTION_STEP_LABEL]['unequal']} "
            f"{PRODUCTION_STEP_LABEL} cell(s)")

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

    model_jacobian = compute_ocean_jacobian(
        state2.eta.data, state2.H_bathy.data, card.recipe.z_coord)
    model_gdepw = jnp.asarray(model_bundle.gdepw_Kmm)
    model_e3w_surface = jnp.asarray(model_bundle.e3w_surface_Kmm)

    def closure_output_with(ops):
        """Call the shared closure with one explicit kt=2 operand bundle."""
        avm_all = ops["entry_avm"]
        op_bundle = TKEEntryN2Bundle(
            rn2=ops["rn2"], rn2b=ops["rn2b"],
            gdepw_Kmm=model_gdepw, e3w_Kmm=ops["e3w_Kmm"],
            e3t_Kmm=ops["e3t_Kmm"], e3w_surface_Kmm=model_e3w_surface)
        return tke_vertical_mixing(
            state2.u.data[:, :-1, :], state2.v.data[:-1, :, :],
            state2.T.data, state2.S.data,
            jnp.zeros_like(state2.T.data), ops["e3w_Kmm"],
            ops["entry_en"], None, None,
            float(arrays["rn_Dt"]), cfg, rho_0=float(const.rho_0),
            g=float(const.g), n_iterations=1,
            taum_surface=ops["taum"], dz_ref=jnp.asarray(z.dz_ref),
            jacobian=model_jacobian,
            z_interface=jnp.asarray(z.z_half_ref[1:jpkm1]),
            dz_surface=-jnp.asarray(z.z_full_ref[0]) * model_jacobian,
            surface_tmask=ops["tmask"][..., 0],
            w_active=ops["wmask"], preclosure_K_M=avm_all[..., 1:],
            preclosure_K_H=ops["entry_avt"],
            preclosure_K_M_surface=avm_all[..., 0],
            preclosure_dissl=ops["entry_dissl"],
            precomputed_p_sh2=ops["sh2"],
            precomputed_n2_bundle=op_bundle)

    def closure_with(ops):
        return closure_output_with(ops).K_H

    # Capture the operands at the production literal solver itself. The
    # record carries these arrays immediately before NEMO's recurrences.
    solver_inputs: dict[str, list[np.ndarray]] = {}
    rhs_inputs: dict[str, list[np.ndarray]] = {}
    rhs_meta: dict[str, bool] = {}
    real_literal_solve = tke_module._nemo_literal_tke_solve
    real_backward_solve = tke_module._solve_tke_backward_euler

    def solver_sink(name):
        return lambda value: solver_inputs.setdefault(name, []).append(
            np.asarray(value, dtype=np.float64))

    def capture_literal_solve(a, b, c, rhs, *args, **kwargs):
        for name, value in (("lower", a), ("diagonal", b),
                            ("upper", c), ("rhs", rhs)):
            jax.debug.callback(solver_sink(name), value)
        return real_literal_solve(a, b, c, rhs, *args, **kwargs)

    def capture_backward_solve(*args, **kwargs):
        for name in ("e_old", "K_H_old", "P_s", "N2",
                     "dissl_old", "w_active"):
            require(kwargs.get(name) is not None,
                    f"production TKE RHS capture lacks {name}")
            jax.debug.callback(solver_sink("rhs_" + name), kwargs[name])
        rhs_meta["literal_external_rhs_is_none"] = (
            kwargs.get("literal_external_rhs") is None)
        if kwargs.get("literal_external_rhs") is not None:
            jax.debug.callback(solver_sink("rhs_literal_external_rhs"),
                               kwargs["literal_external_rhs"])
        rhs_meta["external_source_is_none"] = (
            kwargs.get("external_source") is None)
        if kwargs.get("external_source") is not None:
            jax.debug.callback(solver_sink("rhs_external_source"),
                               kwargs["external_source"])
        return real_backward_solve(*args, **kwargs)

    tke_module._nemo_literal_tke_solve = capture_literal_solve
    tke_module._solve_tke_backward_euler = capture_backward_solve
    try:
        captured_output = jax.jit(
            lambda: closure_output_with(oracle_operands).tke_new)()
        captured_output.block_until_ready()
    finally:
        tke_module._nemo_literal_tke_solve = real_literal_solve
        tke_module._solve_tke_backward_euler = real_backward_solve
    solver_names = {"lower", "diagonal", "upper", "rhs"}
    require(solver_names.issubset(solver_inputs),
            f"production TKE solve capture incomplete: {sorted(solver_inputs)}")
    require(all(len(values) == 1 for values in solver_inputs.values()),
            "production TKE solver did not execute exactly once")
    solver_mask = np.asarray(yx("wmask")[..., :jpkm1]) != 0.0
    # Row zero is a virtual identity in production; the literal solver makes
    # NEMO's surface seeds itself. The deepest upper coefficient is unread.
    solver_mask[..., 0] = False
    solver_oracle = {
        "lower": yx("matrix_lower")[..., :jpkm1],
        "diagonal": yx("matrix_diag")[..., :jpkm1],
        "upper": yx("matrix_upper")[..., :jpkm1],
        "rhs": yx("rhs_pre_sweep")[..., :jpkm1],
    }
    solver_masks = {name: solver_mask.copy() for name in solver_names}
    solver_masks["upper"][..., -1] = False
    production_solver_inputs = {
        name: _operand_score(values[0], solver_oracle[name], solver_masks[name])
        for name, values in solver_inputs.items() if name in solver_names
    }
    for name in ("e_old", "K_H_old", "P_s", "N2",
                 "dissl_old", "w_active"):
        key = "rhs_" + name
        require(key in solver_inputs and len(solver_inputs[key]) == 1,
                f"production TKE RHS capture incomplete for {name}")
        rhs_inputs[name] = solver_inputs.pop(key)
    if rhs_meta.get("literal_external_rhs_is_none"):
        base_rhs = rhs_inputs["e_old"][0]
        if not rhs_meta.get("external_source_is_none"):
            key = "rhs_external_source"
            require(key in solver_inputs and len(solver_inputs[key]) == 1,
                    "production TKE external source capture is incomplete")
            base_rhs = (base_rhs + np.float64(arrays["rn_Dt"])
                        * solver_inputs[key][0])
    else:
        key = "rhs_literal_external_rhs"
        require(key in solver_inputs and len(solver_inputs[key]) == 1,
                "production TKE external RHS capture is incomplete")
        base_rhs = solver_inputs[key][0]
    source_rhs = (
        base_rhs + np.float64(arrays["rn_Dt"]) * (
            rhs_inputs["P_s"][0]
            - rhs_inputs["K_H_old"][0] * rhs_inputs["N2"][0]
            + ((np.float64(0.5) * np.float64(arrays["rn_ediss"]))
               * rhs_inputs["dissl_old"][0] * base_rhs)
        ) * rhs_inputs["w_active"][0]
    )
    association_mask = solver_mask[..., 1:].copy()
    association_mask[..., -1] = False
    production_solver_inputs["rhs_nemo_written_association"] = _operand_score(
        source_rhs, solver_oracle["rhs"][..., 1:], association_mask)

    # First-use order in compiled zdftke: surface BC, Langmuir, inverse
    # Prandtl, matrix construction, RHS.  "masks" is reported as the requested
    # joint field while tmask/wmask remain independently scored above.
    substitution_order = (
        "taum", "rn2b", "e3w_Kmm", "sh2", "entry_avm",
        "e3t_Kmm", "entry_dissl", "entry_en", "entry_avt", "rn2",
        "masks",
    )

    def replaced(base, name):
        out = dict(base)
        if name == "masks":
            out["tmask"] = oracle_operands["tmask"]
            out["wmask"] = oracle_operands["wmask"]
        else:
            out[name] = oracle_operands[name]
        return out

    baseline = np.asarray(jax.jit(lambda: closure_with(model_operands))())

    def closure_score(name, value):
        value = np.asarray(value)
        unequal = _unequal_bits(value, oracle_avt, wet)
        return {
            "name": name,
            "unequal": unequal,
            "wet_cells": int(np.count_nonzero(wet)),
            "max_abs": float(np.max(np.abs(value[wet] - oracle_avt[wet]))),
            "exact": unequal == 0,
        }

    single_rows = [closure_score("all_model_operands", baseline)]
    for name in substitution_order:
        value = jax.jit(lambda op=replaced(model_operands, name):
                        closure_with(op))()
        single_rows.append(closure_score(name, value))
    cumulative_rows = []
    cumulative_ops = dict(model_operands)
    for name in substitution_order:
        cumulative_ops = replaced(cumulative_ops, name)
        value = jax.jit(lambda op=dict(cumulative_ops): closure_with(op))()
        cumulative_rows.append(closure_score(name, value))

    oracle_en = np.asarray(e_post)

    def production_score(name, value):
        value = np.asarray(value)
        unequal = _unequal_bits(value, oracle_en, wet)
        return {
            "name": name,
            "unequal": unequal,
            "wet_cells": int(np.count_nonzero(wet)),
            "max_abs": float(np.max(np.abs(value[wet] - oracle_en[wet]))),
            "exact": unequal == 0,
        }

    production_single_rows = [production_score(
        "all_model_operands",
        jax.jit(lambda: closure_output_with(model_operands).tke_new)())]
    for name in substitution_order:
        value = jax.jit(
            lambda op=replaced(model_operands, name):
            closure_output_with(op).tke_new)()
        production_single_rows.append(production_score(name, value))
    production_oracle = production_score(
        "all_nemo_operands",
        jax.jit(lambda: closure_output_with(oracle_operands).tke_new)())
    if sh2_velocity_attribution is not None:
        velocity_only_ops = dict(oracle_operands)
        velocity_only_ops["sh2"] = model_now_velocity_sh2
        velocity_production = production_score(
            "model_Kmm_with_nemo_Kbb_sh2_only",
            jax.jit(lambda: closure_output_with(
                velocity_only_ops).tke_new)())
        sh2_velocity_attribution["production_row_velocity_only"] = (
            velocity_production)

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
    cumulative_exact = [row["name"] for row in cumulative_rows if row["exact"]]
    single_exact = [row["name"] for row in single_rows[1:] if row["exact"]]
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
        "kt2_carried_operands": operand_rows,
        "sh2_velocity_attribution": sh2_velocity_attribution,
        "single_operand_substitutions": single_rows,
        "cumulative_operand_substitutions": cumulative_rows,
        "production_single_operand_substitutions": production_single_rows,
        "production_all_nemo_operands": production_oracle,
        "production_solver_inputs_all_nemo_operands": production_solver_inputs,
        "kh_statement_walk": kh_walk,
        "single_exact": single_exact,
        "cumulative_first_exact": (
            cumulative_exact[0] if cumulative_exact else None),
        "rows": rows,
        "first_exact": first_exact,
        "status": "PASS",
    }


if __name__ == "__main__":
    raise SystemExit(main())
