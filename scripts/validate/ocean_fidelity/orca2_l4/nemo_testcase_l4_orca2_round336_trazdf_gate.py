#!/usr/bin/env python3
"""Admit the passive round-336 kt=2 stage-3 ``tra_zdf`` record."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round331_tke_gate as r331,
    nemo_testcase_l4_orca2_round333_tracer_gate as tracer_gate,
    nemo_testcase_l4_orca2_round333_traldf_gate as ldf_gate,
)

MAGIC = "NEMO_L4_R336TZ"
HEADER_NAMES = (
    "version", "kt", "Kbb", "Kmm", "Krhs", "Kaa", "rank", "nx", "ny",
    "nz", "ntracer", "real_bits", "n_fields", "ln_zdfddm", "ln_traldf_msc",
    "ntsi", "ntei", "ntsj", "ntej", "zdfphy_phase_mask",
)
THREE_D_FIELDS = (
    "avt", "avs", "akz", "e3t_3d", "e3w_Kmm", "e3t_Kbb", "e3t_Kmm", "e3t_Kaa", "tmask",
    "avt_k_zdf", "avt_after_tke", "avs_after_tke", "avt_after_runoff",
    "avs_after_runoff", "avt_after_evd", "avs_after_evd", "avt_after_ddm",
    "avs_after_ddm", "avt_after_iwm", "avs_after_iwm",
)
INTERIOR_THREE_D_FIELDS = (
    "avt", "avs", "avt_k_zdf", "avt_after_tke", "avs_after_tke",
    "avt_after_runoff", "avs_after_runoff", "avt_after_evd", "avs_after_evd",
    "avt_after_ddm", "avs_after_ddm", "avt_after_iwm", "avs_after_iwm",
)
ZDFPHY_PHASE_FIELDS = INTERIOR_THREE_D_FIELDS[2:]
TWO_D_FIELDS = ("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")
ONE_D_FIELDS = ("e3w_1d",)
FOUR_D_FIELDS = (
    "effective_K", "Kbb_tracer", "Krhs_tracer", "content_rhs", "lower",
    "diagonal", "upper", "eliminated", "forward", "backsub", "actual_Kaa",
    "solve_seen",
)
FIELDS = (
    "avt", "avs", "akz", "e3t_3d", "e3w_1d", "r3t_Kbb", "r3t_Kmm", "r3t_Kaa",
    "effective_K", "e3w_Kmm", "e3t_Kbb", "e3t_Kmm",
    "e3t_Kaa", "Kbb_tracer", "Krhs_tracer", "content_rhs", "lower",
    "diagonal", "upper", "eliminated", "forward", "backsub", "actual_Kaa",
    "solve_seen", "tmask", "avt_k_zdf", "avt_after_tke", "avs_after_tke",
    "avt_after_runoff", "avs_after_runoff", "avt_after_evd", "avs_after_evd",
    "avt_after_ddm", "avs_after_ddm", "avt_after_iwm", "avs_after_iwm",
)
PLANTS = (
    "header", "field-order", "truncation", "checksum", "stamp", "selector",
    "extent", "nonfinite", "signed-zero", "wet", "replay", "association",
    "actual", "time-level", "filled-range", "zdfphy",
    *(f"zdfphy-{name.replace('_', '-')}" for name in ZDFPHY_PHASE_FIELDS),
)
OBSERVER_UNITS = ldf_gate.OBSERVER_UNITS + ("trazdf",)
WRITER_UNITS = ldf_gate.WRITER_UNITS + ("round336_trazdf_writer",)

GateError = tracer_gate.GateError
require = tracer_gate.require


def record_name(rank: int) -> str:
    return f"oracle_r336_trazdf_kt00000002_r{rank:04d}.bin"


def _shape(name: str, header: dict[str, int]) -> tuple[int, ...]:
    base = (header["nx"], header["ny"], header["nz"])
    if name in INTERIOR_THREE_D_FIELDS:
        return (header["ntei"] - header["ntsi"] + 1,
                header["ntej"] - header["ntsj"] + 1, header["nz"])
    if name in THREE_D_FIELDS:
        return base
    if name in FOUR_D_FIELDS:
        return (*base, header["ntracer"])
    if name in TWO_D_FIELDS:
        return base[:2]
    if name in ONE_D_FIELDS:
        return base[2:]
    raise GateError(f"unknown round-336 field {name}")


def read_record(path: Path, *, plant: str | None = None,
                expected_shape: tuple[int, int, int] = (94, 152, 31)) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = tracer_gate._take(stream, 16, "magic").decode("ascii").rstrip()
    header = dict(zip(HEADER_NAMES, struct.unpack(
        f"={len(HEADER_NAMES)}i",
        tracer_gate._take(stream, 4 * len(HEADER_NAMES), "header")), strict=True))
    if plant == "header":
        header["kt"] = 1
    elif plant == "time-level":
        header["Kmm"] = header["Kbb"]
    elif plant == "selector":
        header["ln_zdfddm"] = 0
    require(magic == MAGIC, f"bad magic {magic!r}")
    require((header["version"], header["kt"], header["real_bits"]) == (2, 2, 64),
            "wrong version/step/precision")
    require(tuple(header[name] for name in ("Kbb", "Kmm", "Krhs", "Kaa")) ==
            tracer_gate.rk3_levels_for_step_stage(2, 3), "RK time levels moved")
    require(header["rank"] in (0, 1), "rank tag moved")
    require((header["nx"], header["ny"], header["nz"]) == expected_shape,
            "record lacks exact jpi,jpj,jpk extents")
    require((header["ntsi"], header["ntei"], header["ntsj"], header["ntej"]) ==
            (3, expected_shape[0] - 2, 3, expected_shape[1] - 2),
            "recorded NEMO filled-range bounds moved")
    require(header["ntracer"] == 2 and header["n_fields"] == len(FIELDS),
            "tracer or field census moved")
    require((header["ln_zdfddm"], header["ln_traldf_msc"]) == (1, 1),
            "recorded solver selectors moved")
    require(header["zdfphy_phase_mask"] == 31,
            "kt=2 zdf_phy contribution phase census moved")

    fields: dict[str, np.ndarray] = {}
    checksums: dict[str, int] = {}
    order: list[str] = []
    for index in range(header["n_fields"]):
        name = tracer_gate._take(stream, 24, f"field {index} name").decode("ascii").rstrip()
        metadata = list(struct.unpack("=9i", tracer_gate._take(
            stream, 36, f"{name} metadata")))
        if plant == "extent" and index == 0:
            metadata[1] -= 1
        rank, *dims, i0, j0, k0, n0 = metadata
        shape = _shape(name, header)
        padded = (*shape, *(1 for _ in range(4 - len(shape))))
        require((rank, *dims, i0, j0, k0, n0) ==
                (len(shape), *padded, 1, 1, 1, 1),
                f"{name}: wrong rank, extent, or origin")
        checksum, = struct.unpack("=Q", tracer_gate._take(stream, 8, f"{name} checksum"))
        payload = tracer_gate._take(stream, int(np.prod(shape)) * 8, f"{name} payload")
        fields[name] = np.frombuffer(payload, dtype="=f8").copy().reshape(shape, order="F")
        checksums[name] = checksum
        order.append(name)
    require(stream.read(1) == b"", "trailing bytes")
    if plant == "field-order":
        order[0], order[1] = order[1], order[0]
    require(tuple(order) == FIELDS, "field order differs from source order")
    return {"header": header, "fields": fields, "checksums": checksums,
            "sha256": hashlib.sha256(raw).hexdigest()}


def _bit_equal(actual: np.ndarray, expected: np.ndarray,
               support: np.ndarray, message: str) -> None:
    support = np.broadcast_to(support, actual.shape)
    require(actual.shape == expected.shape, f"{message}: shape moved")
    require(bool(np.array_equal(
        np.ascontiguousarray(actual[support]).view(np.uint64),
        np.ascontiguousarray(expected[support]).view(np.uint64))), message)


def verify_record(record: dict, *, plant: str | None = None) -> dict[str, int]:
    fields, header = record["fields"], record["header"]
    seen = fields["solve_seen"] == 1.0
    wet = np.broadcast_to(fields["tmask"][..., None] != 0.0, seen.shape)
    owned = np.zeros_like(seen)
    owned[header["ntsi"] - 1:header["ntei"],
          header["ntsj"] - 1:header["ntej"], :header["nz"] - 1, :] = True
    wet_index = tuple(np.argwhere(wet & owned)[0])
    interior_wet_index = (wet_index[0] - (header["ntsi"] - 1),
                          wet_index[1] - (header["ntsj"] - 1), wet_index[2])
    if plant is not None and plant.startswith("zdfphy-"):
        planted_name = plant.removeprefix("zdfphy-").replace("-", "_")
        require(planted_name in ZDFPHY_PHASE_FIELDS,
                f"unknown zdf_phy phase plant {planted_name}")
        fields[planted_name][interior_wet_index] = np.nextafter(
            fields[planted_name][interior_wet_index], np.inf)
    r331._verify_checksums(record)
    if plant == "nonfinite":
        fields["forward"][wet_index] = np.nan
    require(all(bool(np.isfinite(value).all()) for value in fields.values()),
            "non-finite payload")
    require(bool(np.all(seen[wet & owned])), "wet solve coverage is incomplete")
    if plant == "wet":
        index = tuple(np.argwhere(wet & owned)[0])
        seen[index] = False
    require(bool(np.all(seen[wet & owned])), "wet solve coverage is incomplete")
    if plant == "signed-zero":
        for value in fields.values():
            value[value == 0.0] = 0.0
    signed_zeros = sum(int(np.count_nonzero((value == 0.0) & np.signbit(value)))
                       for value in fields.values())
    require(signed_zeros > 0, "signed zeros were normalized")

    # These supports are the literal ranges filled by trazdf.f90:184-216 and
    # zdfphy.f90:368-392. Every unfilled record cell must retain the -0 sentinel.
    i0, i1 = header["ntsi"] - 1, header["ntei"]
    j0, j1 = header["ntsj"] - 1, header["ntej"]
    metric3 = np.zeros(fields["tmask"].shape, dtype=bool)
    metric3[i0:i1, j0:j1, :header["nz"] - 1] = True
    e3w_support = np.zeros_like(metric3)
    e3w_support[i0:i1, j0:j1, :] = True
    process_support = np.zeros(fields["avt"].shape, dtype=bool)
    process_support[:, :, 1:header["nz"] - 1] = True

    if plant == "filled-range":
        fields["e3t_Kaa"][0, 0, 0] = 1.0
    if plant == "zdfphy":
        fields["avt_after_iwm"][interior_wet_index] = np.nextafter(
            fields["avt_after_iwm"][interior_wet_index], np.inf)

    def filled(name: str, expected: np.ndarray, support: np.ndarray) -> None:
        _bit_equal(fields[name], expected, support, f"{name} filled-range replay moved")
        outside = ~support
        require(bool(np.all((fields[name][outside] == 0.0)
                            & np.signbit(fields[name][outside]))),
                f"{name} overwrote NEMO-unfilled cells")

    expected_e3w = fields["e3w_1d"][None, None, :] * (
        np.float64(1.0) + fields["r3t_Kmm"][..., None])
    filled("e3w_Kmm", expected_e3w, e3w_support)
    for level in ("Kbb", "Kmm", "Kaa"):
        expected_metric = fields["e3t_3d"] * (
            np.float64(1.0) + fields[f"r3t_{level}"][..., None] * fields["tmask"])
        filled(f"e3t_{level}", expected_metric, metric3)

    for name in ("avt_k_zdf", "avt_after_tke", "avs_after_tke",
                 "avt_after_runoff", "avs_after_runoff", "avt_after_evd",
                 "avs_after_evd", "avt_after_ddm", "avs_after_ddm",
                 "avt_after_iwm", "avs_after_iwm"):
        outside = ~process_support
        require(bool(np.all((fields[name][outside] == 0.0)
                            & np.signbit(fields[name][outside]))),
                f"{name} overwrote zdf_phy-unfilled cells")
    _bit_equal(fields["avt_after_tke"], fields["avt_k_zdf"], process_support,
               "TKE avt_k copy boundary moved")
    _bit_equal(fields["avs_after_runoff"], fields["avs_after_tke"], process_support,
               "runoff boundary changed avs")
    _bit_equal(fields["avs_after_evd"], fields["avs_after_runoff"], process_support,
               "EVD boundary changed avs")
    _bit_equal(fields["avt_after_iwm"], fields["avt"], process_support,
               "IWM avt boundary differs from final tra_zdf input")
    _bit_equal(fields["avs_after_iwm"], fields["avs"], process_support,
               "IWM avs boundary differs from final tra_zdf input")

    if plant == "replay":
        fields["diagonal"][wet_index] = np.nextafter(
            fields["diagonal"][wet_index], np.inf)
    if plant == "actual":
        index = tuple(np.argwhere(wet & owned)[0])
        fields["actual_Kaa"][index] = np.nextafter(fields["actual_Kaa"][index], np.inf)

    active3 = np.any(seen, axis=-1)
    effective = np.full_like(fields["effective_K"], -0.0)
    effective[i0:i1, j0:j1, 0, :] = 0.0
    effective[i0:i1, j0:j1, 1:, 0] = (
        fields["avt"][:, :, 1:] + fields["akz"][i0:i1, j0:j1, 1:])
    effective[i0:i1, j0:j1, 1:, 1] = (
        fields["avs"][:, :, 1:] + fields["akz"][i0:i1, j0:j1, 1:])
    effective_support = np.zeros_like(seen)
    effective_support[i0:i1, j0:j1, :, :] = True
    _bit_equal(fields["effective_K"], effective, effective_support,
               "effective avt/avs plus akz replay moved")
    require(bool(np.all((fields["effective_K"][~effective_support] == 0.0)
                        & np.signbit(fields["effective_K"][~effective_support]))),
            "effective_K overwrote NEMO-unfilled cells")

    # Replay the exact source statements in scalar loop order. Boundary/halo
    # slots outside solve_seen are intentionally not claimed.
    expected = {name: np.array(fields[name], copy=True) for name in (
        "lower", "diagonal", "upper", "eliminated", "content_rhs", "forward", "backsub")}
    for name in expected:
        expected[name][seen] = -0.0
    dt = np.float64(10800.0)
    nx, ny, nz, nt = seen.shape
    association_discriminators = 0
    association_plant_set = False
    for j in range(2, ny - 2):
        for n in range(nt):
            for k in range(nz - 1):
                for i in range(2, nx - 2):
                    lo = (-dt * fields["effective_K"][i, j, k, n]
                          / (fields["e3w_1d"][k]
                             * (np.float64(1.0) + fields["r3t_Kmm"][i, j])))
                    up = (-dt * fields["effective_K"][i, j, k + 1, n]
                          / (fields["e3w_1d"][k + 1]
                             * (np.float64(1.0) + fields["r3t_Kmm"][i, j])))
                    expected["lower"][i, j, k, n] = lo
                    expected["upper"][i, j, k, n] = up
                    expected["diagonal"][i, j, k, n] = (
                        fields["e3t_3d"][i, j, k]
                        * (np.float64(1.0) + fields["r3t_Kaa"][i, j]
                           * fields["tmask"][i, j, k]) - (lo + up))
            for i in range(2, nx - 2):
                expected["eliminated"][i, j, 0, n] = expected["diagonal"][i, j, 0, n]
            for k in range(1, nz - 1):
                for i in range(2, nx - 2):
                    expected["eliminated"][i, j, k, n] = (
                        expected["diagonal"][i, j, k, n]
                        - expected["lower"][i, j, k, n]
                        * expected["upper"][i, j, k - 1, n]
                        / expected["eliminated"][i, j, k - 1, n])
            for k in range(nz - 1):
                for i in range(2, nx - 2):
                    stretch = (np.float64(1.0) + fields["r3t_Kmm"][i, j]
                               * fields["tmask"][i, j, k])
                    kbb_content = (fields["e3t_3d"][i, j, k]
                                   * (np.float64(1.0) + fields["r3t_Kbb"][i, j]
                                      * fields["tmask"][i, j, k])
                                   * fields["Kbb_tracer"][i, j, k, n])
                    # trazdf.f90:285/:289 evaluates p2dt * (e3t * stretch)
                    # before multiplying Krhs.  The superficially equivalent
                    # ((p2dt * e3t) * stretch) association differs on real data.
                    tendency = (dt * (fields["e3t_3d"][i, j, k] * stretch)
                                * fields["Krhs_tracer"][i, j, k, n])
                    wrong_tendency = ((dt * fields["e3t_3d"][i, j, k]) * stretch
                                      * fields["Krhs_tracer"][i, j, k, n])
                    rhs = kbb_content + tendency
                    wrong_rhs = kbb_content + wrong_tendency
                    if (seen[i, j, k, n]
                            and np.float64(rhs).view(np.uint64)
                            != np.float64(wrong_rhs).view(np.uint64)):
                        association_discriminators += 1
                        if plant == "association" and not association_plant_set:
                            fields["content_rhs"][i, j, k, n] = wrong_rhs
                            association_plant_set = True
                    expected["content_rhs"][i, j, k, n] = rhs
                    expected["forward"][i, j, k, n] = rhs if k == 0 else (
                        rhs - expected["lower"][i, j, k, n]
                        / expected["eliminated"][i, j, k - 1, n]
                        * expected["forward"][i, j, k - 1, n])
            k = nz - 2
            for i in range(2, nx - 2):
                expected["backsub"][i, j, k, n] = (
                    expected["forward"][i, j, k, n]
                    / expected["eliminated"][i, j, k, n]
                    * fields["tmask"][i, j, k])
            for k in range(nz - 3, -1, -1):
                for i in range(2, nx - 2):
                    expected["backsub"][i, j, k, n] = (
                        (expected["forward"][i, j, k, n]
                         - expected["upper"][i, j, k, n]
                         * expected["backsub"][i, j, k + 1, n])
                        / expected["eliminated"][i, j, k, n]
                        * fields["tmask"][i, j, k])
    for name in expected:
        _bit_equal(fields[name], expected[name], seen, f"{name} source replay moved")
    require(association_discriminators > 0,
            "record has no wet cell that discriminates NEMO RHS association")
    require(plant != "association" or association_plant_set,
            "association plant found no admitted discriminator")
    _bit_equal(fields["actual_Kaa"], fields["backsub"], seen,
               "outside-loop replay does not reproduce live Kaa")
    return {"wet_seen": int(np.count_nonzero(wet & owned)),
            "signed_zeros": signed_zeros,
            "rhs_association_discriminators": association_discriminators,
            "active_cells": int(np.count_nonzero(active3)),
            "zdfphy_process_cells": int(np.count_nonzero(process_support))}


def _bits_equal(actual: np.ndarray, expected: np.ndarray, message: str) -> None:
    require(actual.shape == expected.shape, f"{message}: shape moved")
    require(bool(np.array_equal(actual.view(np.uint64), expected.view(np.uint64))), message)


def run_gate(root: Path, *, expect_commit: str, plant: str | None = None) -> dict:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    produced = (root / "producer_commit.txt").read_text().strip()
    if plant == "stamp":
        produced = "0" * 40
    require(produced == expect_commit, "producer commit moved")
    inherited = ldf_gate.run_gate(root, expect_commit=expect_commit)
    require(inherited["status"] == "PASS_R333_TRALDF_INTERNAL_ACQUISITION",
            "inherited round-333 admission moved")
    reports = {}
    records = {}
    for rank in (0, 1):
        parser_plant = plant if rank == 0 and plant in (
            "header", "field-order", "truncation", "selector", "extent",
            "time-level") else None
        record = read_record(root / record_name(rank), plant=parser_plant)
        require(record["header"]["rank"] == rank, "rank tag moved")
        if rank == 0 and plant == "checksum":
            header = record["header"]
            full_wet = record["fields"]["tmask"] != 0.0
            full_wet[:header["ntsi"] - 1, :, :] = False
            full_wet[header["ntei"]:, :, :] = False
            full_wet[:, :header["ntsj"] - 1, :] = False
            full_wet[:, header["ntej"]:, :] = False
            global_index = tuple(np.argwhere(full_wet)[0])
            index = (global_index[0] - (header["ntsi"] - 1),
                     global_index[1] - (header["ntsj"] - 1), global_index[2])
            record["fields"]["avt"][index] = np.nextafter(
                record["fields"]["avt"][index], np.inf)
        reports[str(rank)] = verify_record(
            record, plant=plant if rank == 0 and (
                plant in ("nonfinite", "signed-zero", "wet", "replay", "association",
                          "actual", "filled-range", "zdfphy") or
                (plant is not None and plant.startswith("zdfphy-"))) else None)
        records[rank] = record

        tracer = tracer_gate.read_stage3(root / tracer_gate.record_name(rank))["fields"]
        ldf = ldf_gate.read_record(root / ldf_gate.record_name(rank))["fields"]
        fields, header = record["fields"], record["header"]
        _bits_equal(fields["akz"], ldf["akz"], "akz differs from admitted LDF input")
        _bits_equal(fields["e3t_3d"], ldf["e3t_3d"],
                    "e3t_3d differs from admitted geometry")
        _bits_equal(fields["e3w_1d"], ldf["e3w_1d"],
                    "e3w_1d differs from admitted geometry")
        for level in ("Kbb", "Kmm", "Kaa"):
            _bits_equal(fields[f"r3t_{level}"], tracer[f"r3t_{level}"],
                        f"r3t_{level} differs from admitted tracer record")
        for index, name in enumerate(("T", "S")):
            _bits_equal(fields["Kbb_tracer"][..., index], tracer[f"Kbb_{name}"],
                        f"Kbb {name} differs from admitted tracer record")
            _bits_equal(fields["Krhs_tracer"][..., index], tracer[f"after_damping_{name}"],
                        f"Krhs {name} differs from admitted tracer record")
            _bits_equal(fields["actual_Kaa"][..., index], tracer[f"Kaa_{name}"],
                        f"Kaa {name} differs from admitted tracer record")
        support = np.zeros(fields["tmask"].shape, dtype=bool)
        support[header["ntsi"] - 1:header["ntei"],
                header["ntsj"] - 1:header["ntej"], :header["nz"] - 1] = True
        for level, out_name in (("Kbb", "e3t_Kbb"), ("Kmm", "e3t_Kmm"),
                                ("Kaa", "e3t_Kaa")):
            expected = ldf["e3t_3d"] * (
                np.float64(1.0) + tracer[f"r3t_{level}"][..., None] * ldf["tmask"])
            _bit_equal(fields[out_name], expected, support,
                       f"{out_name} differs from admitted geometry")
        e3w_support = np.zeros_like(support)
        e3w_support[header["ntsi"] - 1:header["ntei"],
                    header["ntsj"] - 1:header["ntej"], :] = True
        expected_e3w = ldf["e3w_1d"][None, None, :] * (
            np.float64(1.0) + tracer["r3t_Kmm"][..., None])
        _bit_equal(fields["e3w_Kmm"], expected_e3w, e3w_support,
                   "e3w_Kmm differs from admitted geometry")

    return {
        "format": "nemo-testcase-l4-orca2-round336-trazdf-v1",
        "status": "PASS_R336_TRAZDF_ACQUISITION",
        "producer_commit": produced,
        "inherited_status": inherited["status"],
        "recorded_fields": list(FIELDS),
        "records": {record_name(rank): records[rank]["sha256"] for rank in records},
        "rank_census": reports,
        "observer_location": "single call after tra_zdf_imp, outside its loops",
    }


def _inserted_lines(reference: list[str], observed: list[str]) -> list[str]:
    inserted = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(
            None, reference, observed, autojunk=False).get_opcodes():
        if tag == "insert":
            inserted.extend(observed[j1:j2])
    return inserted


def same_program(new_bld: Path, ref_bld: Path, patches: dict[str, Path], *,
                 new_cfg: str, ref_cfg: str) -> dict:
    """Extend the inherited same-program proof by exactly one observer unit."""
    require(tuple(patches) == OBSERVER_UNITS,
            f"wrong observer unit order: {tuple(patches)!r}")
    for unit in ("traldf_iso", "trazdf"):
        reference = (ref_bld / f"ppsrc/nemo/{unit}.f90").read_text().splitlines()
        observed = (new_bld / f"ppsrc/nemo/{unit}.f90").read_text().splitlines()
        expected = [line[1:] for line in patches[unit].read_text().splitlines()
                    if line.startswith("+") and not line.startswith("+++")]
        actual = _inserted_lines(reference, observed)
        mismatch = next((index for index, pair in enumerate(zip(expected, actual))
                         if pair[0] != pair[1]), min(len(expected), len(actual)))
        require(expected == actual,
                f"{unit} patch ledger mismatch expected={len(expected)} "
                f"observed={len(actual)} first={mismatch}")
    return ldf_gate.dyn_gate.same_program_observers(
        new_bld, ref_bld, patches, WRITER_UNITS,
        new_cfg=new_cfg, ref_cfg=ref_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--same-program", type=Path)
    parser.add_argument("--reference-bld", type=Path)
    parser.add_argument("--observer-patch", action="append", default=[])
    parser.add_argument("--new-cfg")
    parser.add_argument("--reference-cfg")
    args = parser.parse_args()
    try:
        if args.same_program:
            entries = [item.split("=", 1) for item in args.observer_patch]
            result = same_program(
                args.same_program, args.reference_bld,
                {unit: Path(path) for unit, path in entries},
                new_cfg=args.new_cfg, ref_cfg=args.reference_cfg)
        else:
            require(args.record_dir is not None and args.expect_commit is not None,
                    "admission arguments incomplete")
            result = run_gate(args.record_dir, expect_commit=args.expect_commit,
                              plant=args.plant)
            require(args.plant is None, f"{args.plant} plant stayed green")
    except (GateError, OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
