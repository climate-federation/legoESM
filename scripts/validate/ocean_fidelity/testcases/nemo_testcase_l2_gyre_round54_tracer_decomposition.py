#!/usr/bin/env python3
"""Decompose GYRE kt=2 tracer creation at every WS-RK3 stage.

The model starts from NEMO's kt=2 T/S/u/v/ssh and barotropic pair.  A second
arm also replaces every TKE quantity carried by the round-46 stage record.
Stage-1 and stage-2 references are the following stages' Kmm fields; the
kt=3 ENTRY record is the stage-3 reference.  See the round-54 preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

ENTRY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners/nemo_seed0"
)
STAGE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/oracle_kt2_stage"
)
ZDF_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round38_oracle_trazdf_kt2/oracle_trazdf_matrix_kt00000002.bin"
)
EXPECTED_T_RMS_K = 4.1543946279e-4
EXPECTED_T_UNEQUAL = 17_999
ANCHOR_ATOL_K = 5.0e-15

R63_KRHS_FIELDS = (
    "p2dt", "krhs_zero_T", "krhs_zero_S", "adv_up1_T", "adv_up1_S",
    "after_adv_T", "after_adv_S", "after_sbc_T", "after_sbc_S",
    "after_qsr_T", "after_qsr_S", "after_ldf_T", "after_ldf_S",
    "T_Kbb", "S_Kbb", "e3t_Kbb", "e3t_Kmm", "r3t_Kbb", "r3t_Kmm",
    "e3t_3d", "tmask", "content_T", "content_S",
)
R63_TKE_FIELDS = (
    "rn_Dt", "zfact3", "en_rhs_entry", "shear", "avt", "rn2", "dissl",
    "strat_product", "diss_product", "wmask", "en_rhs_post",
)


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _read_r63_stream(path: Path, *, kind: str,
                     plant: str | None = None) -> dict:
    """Read one round-63 stream; this extends the existing tracer parser."""
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    offset = 0

    def take(count: int) -> bytes:
        nonlocal offset
        require(offset + count <= len(raw), f"{kind} record is truncated")
        out = raw[offset:offset + count]
        offset += count
        return out

    if kind == "krhs":
        magic = b"NEMO_L2_R63KRS1 "
        keys = ("version", "kt", "kstg", "Kbb", "Kmm", "Krhs",
                "nx", "ny", "nlev", "ntsi", "ntsj", "real_bits",
                "field_count")
        fields = R63_KRHS_FIELDS
    else:
        require(kind == "tke", f"unknown round-63 stream kind {kind}")
        magic = b"NEMO_L2_R63TKR1 "
        keys = ("version", "kt", "nx", "ny", "nlev", "ntsi", "ntsj",
                "real_bits", "field_count")
        fields = R63_TKE_FIELDS
    require(take(16) == magic, f"wrong {kind} magic")
    header = dict(zip(
        keys, struct.unpack(f"={len(keys)}i", take(4 * len(keys))),
        strict=True))
    require(header["version"] == 1 and header["kt"] == 2,
            f"wrong {kind} version/kt")
    require((header["nx"], header["ny"], header["nlev"]) == (32, 22, 30),
            f"wrong {kind} inner-domain shape")
    require(header["real_bits"] == 64, f"{kind} record is not fp64")
    require(header["field_count"] == len(fields),
            f"{kind} field-count stamp disagrees with schema")
    arrays = {}
    for expected in fields:
        label = take(16).decode("ascii").rstrip()
        require(label == expected,
                f"{kind} field order mismatch: {label!r} != {expected!r}")
        ndim, n1, n2, n3 = struct.unpack("=4i", take(16))
        require(ndim in (0, 2, 3), f"invalid {kind} rank {ndim}")
        shape = (() if ndim == 0 else
                 (n1, n2) if ndim == 2 else (n1, n2, n3))
        require(shape in ((), (32, 22), (32, 22, 30)),
                f"wrong {kind} field shape {label}={shape}")
        count = 1 if not shape else int(np.prod(shape))
        value = np.frombuffer(take(8 * count), dtype="=f8").copy()
        arrays[label] = (value.reshape(shape, order="F")
                         if shape else value[0])
    require(offset == len(raw), f"{kind} record has trailing bytes")
    require(all(np.isfinite(value).all() for value in arrays.values()),
            f"{kind} record contains NaN/Inf")
    if plant == "ulp":
        target = "content_T" if kind == "krhs" else "en_rhs_post"
        mask_name = "tmask" if kind == "krhs" else "wmask"
        wet = np.argwhere(arrays[mask_name] != 0.0)
        require(wet.size != 0, f"{kind} one-ULP plant found no wet cell")
        at = tuple(wet[0])
        arrays[target][at] = np.nextafter(
            arrays[target][at], np.float64(np.inf))
    return {"header": header, "arrays": arrays}


def _r63_unequal(left, right) -> int:
    a = np.ascontiguousarray(left, dtype="=f8").view("=u8")
    b = np.ascontiguousarray(right, dtype="=f8").view("=u8")
    require(a.shape == b.shape, "round-63 calibration shape mismatch")
    return int(np.count_nonzero(a != b))


def _calibrate_r63_krhs(arrays: dict) -> dict:
    """Rebuild compiled trazdf.f90:545-548 in its exact association."""
    e3b = arrays["e3t_3d"] * (
        np.float64(1.0) + arrays["r3t_Kbb"][..., None] * arrays["tmask"])
    e3m = arrays["e3t_3d"] * (
        np.float64(1.0) + arrays["r3t_Kmm"][..., None] * arrays["tmask"])
    rows = {
        "e3t_Kbb_unequal": _r63_unequal(e3b, arrays["e3t_Kbb"]),
        "e3t_Kmm_unequal": _r63_unequal(e3m, arrays["e3t_Kmm"]),
    }
    for tracer in ("T", "S"):
        # Preserve NEMO's written left product + p2dt * e3 * Krhs.
        rebuilt = (arrays["e3t_Kbb"] * arrays[f"{tracer}_Kbb"]
                   + np.float64(arrays["p2dt"]) * arrays["e3t_Kmm"]
                   * arrays[f"after_ldf_{tracer}"])
        rows[f"content_{tracer}_unequal"] = _r63_unequal(
            rebuilt, arrays[f"content_{tracer}"])
    require(all(value == 0 for value in rows.values()),
            f"round-63 content calibration is not exact: {rows}")
    return rows


def _calibrate_r63_tke(arrays: dict) -> dict:
    """Rebuild compiled zdftke.f90:416-424 without reassociation."""
    strat = arrays["avt"] * arrays["rn2"]
    diss = (np.float64(arrays["zfact3"]) * arrays["dissl"]
            * arrays["en_rhs_entry"])
    post = (arrays["en_rhs_entry"] + np.float64(arrays["rn_Dt"])
            * ((arrays["shear"] - arrays["strat_product"])
               + arrays["diss_product"]) * arrays["wmask"])
    rows = {
        "strat_product_unequal": _r63_unequal(
            strat, arrays["strat_product"]),
        "diss_product_unequal": _r63_unequal(
            diss, arrays["diss_product"]),
        "rhs_statement_unequal": _r63_unequal(post, arrays["en_rhs_post"]),
    }
    require(all(value == 0 for value in rows.values()),
            f"round-63 TKE calibration is not exact: {rows}")
    return rows


def _verify_r63_stamp(record: Path, stamp: Path, producer: str) -> dict:
    words = stamp.read_text().split()
    require(len(words) == 3, f"bad per-file stamp {stamp}")
    digest, stamped_commit, stamped_name = words
    actual = hashlib.sha256(record.read_bytes()).hexdigest()
    require(digest == actual, f"sha256 mismatch for {record.name}")
    require(stamped_commit == producer, f"commit mismatch in {stamp.name}")
    require(stamped_name == record.name, f"filename mismatch in {stamp.name}")
    return {"sha256": digest, "producer_commit": stamped_commit}


def _r63_resolved_config(path: Path, expect_itend: int = 2) -> dict:
    text = path.read_text(errors="replace")
    patterns = {
        "run_horizon": (
            rf"number of the last time step\s+nn_itend\s*=\s*{expect_itend}\b"
        ),
        "no_assimilation": r"Assimilation cycle\s+nn_no\s*=\s*0\b",
        "no_tiling": r"ln_tile\s*=\s*F\b",
        "qsr": r"ln_traqsr\s*=\s*T\b",
        "no_bdy": r"ln_bdy\s*=\s*F\b",
        "no_isf": r"ln_isfcav\s*=\s*F\b",
        "fct": r"ln_traadv_fct\s*=\s*T\b",
        "fct_h2": r"nn_fct_h\s*=\s*2\b",
        "fct_v2": r"nn_fct_v\s*=\s*2\b",
        "fct_imp1": r"nn_fct_imp\s*=\s*1\b",
        "no_adaptive": r"ln_zad_Aimp\s*=\s*F\b",
        "no_msc": r"ln_traldf_msc\s*=\s*F\b",
        "no_bbc": r"ln_trabbc\s*=\s*F\b",
        "no_bbl": r"ln_trabbl\s*=\s*F\b",
        "no_damping": r"ln_tradmp\s*=\s*F\b",
        "no_mfc": r"ln_zdfmfc\s*=\s*F\b",
        "no_osm": r"ln_zdfosm\s*=\s*F\b",
        "no_npc": r"ln_zdfnpc\s*=\s*F\b",
    }
    rows = {}
    lines = text.splitlines()
    for name, pattern in patterns.items():
        matches = [(number, line.strip())
                   for number, line in enumerate(lines, 1)
                   if re.search(pattern, line)]
        require(len(matches) == 1,
                f"resolved ocean.output check {name} matched {len(matches)} lines")
        rows[name] = {"line": matches[0][0], "text": matches[0][1]}
    return rows


def r63_calibrate(*, krhs_record: Path, tke_record: Path,
                  krhs_stamp: Path, tke_stamp: Path,
                  expect_commit: str, producer_commit: Path,
                  resolved_output: Path,
                  expect_itend: int = 2,
                  plant: str | None = None) -> dict:
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    producer = producer_commit.read_text().strip().lower()
    expected = expect_commit.lower()
    if plant == "stamp":
        producer = "0" * 40
    require(len(expected) == 40 and producer == expected,
            f"producer stamp mismatch: {producer} != {expected}")
    stamps = {
        "krhs": _verify_r63_stamp(krhs_record, krhs_stamp, producer),
        "tke": _verify_r63_stamp(tke_record, tke_stamp, producer),
    }
    krhs_plant = ("truncation" if plant == "krhs-truncation"
                  else "ulp" if plant == "krhs-ulp" else None)
    tke_plant = ("truncation" if plant == "tke-truncation"
                 else "ulp" if plant == "tke-ulp" else None)
    krhs = _read_r63_stream(krhs_record, kind="krhs", plant=krhs_plant)
    tke = _read_r63_stream(tke_record, kind="tke", plant=tke_plant)
    return {
        "format": "gyre-round63-krhs-split-v1",
        "worktree": worktree_stamp(),
        "producer_commit": producer,
        "records": {"krhs": str(krhs_record), "tke": str(tke_record)},
        "stamps": stamps,
        "headers": {"krhs": krhs["header"], "tke": tke["header"]},
        "calibration": {
            "krhs": _calibrate_r63_krhs(krhs["arrays"]),
            "tke": _calibrate_r63_tke(tke["arrays"]),
        },
        "resolved_configuration": _r63_resolved_config(
            resolved_output, expect_itend),
        "expected_last_step": expect_itend,
        "plant": plant,
        "status": "PASS",
    }


def _provisional_krhs_boundary_rows(live_content: dict, arrays: dict,
                                    wet: dict) -> dict:
    """Compare the first complete stage-3 term without re-associating NEMO.

    The record is Fortran ``(i,j,k)`` while legoESM is ``(j,i,k)``.  Content
    is the primary comparison because it is the value each implementation
    actually materializes; the concentration-Krhs row is a diagnostic
    inversion of that same boundary and is labelled as such in the report.
    """
    transpose = lambda value: np.ascontiguousarray(  # noqa: E731
        np.asarray(value, dtype=np.float64).transpose(1, 0, 2))
    p2dt = np.float64(arrays["p2dt"])
    rows = {}
    for tracer in ("T", "S"):
        mask = np.asarray(wet[tracer], dtype=bool)
        zero = transpose(arrays[f"krhs_zero_{tracer}"])
        nemo_krhs = transpose(arrays[f"after_adv_{tracer}"])
        base = transpose(arrays["e3t_Kbb"] * arrays[f"{tracer}_Kbb"])
        coefficient = p2dt * transpose(arrays["e3t_Kmm"])
        nemo_content = base + coefficient * nemo_krhs
        model_content = np.asarray(live_content[tracer], dtype=np.float64)
        model_krhs = np.zeros_like(model_content)
        np.divide(model_content - base, coefficient, out=model_krhs,
                  where=mask & (coefficient != 0.0))
        rows[tracer] = {
            "zero": field_stats(np.zeros_like(zero), zero, mask),
            "after_complete_fct_advection_content": field_stats(
                model_content, nemo_content, mask),
            "after_complete_fct_advection_derived_krhs": field_stats(
                model_krhs, nemo_krhs, mask),
        }
    require(all(row["zero"]["cells_unequal"] == 0 for row in rows.values()),
            "the recorded zero boundary is not exact")
    return rows


def provisional_krhs_walk(*, entry_root: Path, stage_root: Path,
                          krhs_record: Path, krhs_stamp: Path,
                          producer_commit: Path, admission: Path,
                          expect_commit: str,
                          expect_record_commit: str) -> dict:
    """Read the rejected R63 record only as a preregistration preview."""
    import jax
    import nemo_testcase_l2_gyre_phase3_gate as gate
    import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46
    import nemo_testcase_overflow_barotropic_gate as baro
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["commit"].lower() == expect_commit.lower(),
            f"commit stamp mismatch: {stamp['commit']} != {expect_commit}")
    rejected = json.loads(admission.read_text())
    require(rejected.get("verdict") == "FAIL",
            "this mode is restricted to a rejected, PROVISIONAL record")
    require(Path(rejected.get("candidate", "")).resolve()
            == krhs_record.parent.resolve(),
            "admission candidate does not own the Krhs record")
    require(all(row.get("consumed_equal") is True
                for row in rejected.get("classified_changed_records", [])),
            "rejected run has a changed consumed field; preview refused")
    require(rejected.get("violations") and all(
        item.startswith("missing inherited records:")
        for item in rejected["violations"]),
        "rejected run failed for more than the truncated horizon")

    producer = producer_commit.read_text().strip().lower()
    require(producer == expect_record_commit.lower(),
            f"record commit mismatch: {producer} != {expect_record_commit}")
    record_stamp = _verify_r63_stamp(krhs_record, krhs_stamp, producer)
    record = _read_r63_stream(krhs_record, kind="krhs")
    calibration = _calibrate_r63_krhs(record["arrays"])

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    previous_tke = round46.read_stage(
        stage_root / "oracle_momstage_kt00000001_s1.bin")
    entry2 = gate.read_entry(entry_root / "oracle_step_entry_kt00000002.bin")
    frames = gate.read_bt(entry_root / "oracle_bt_frames_kt00000001.bin", 1)
    card = build_nemo_testcase_card("GYRE-zco")
    masks = gate.expected_masks(card)
    base_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    initial = card.recipe.initial_state
    freshwater1, surface1 = gate._surface_forcings(card, initial, 1)
    free_entry = base_model.step(initial, dt=card.dt_s,
                                 freshwater=freshwater1,
                                 surface_forcing=surface1)
    seeded = baro.state_from_oracle_entry(free_entry, entry2, masks)
    seeded = _bridge_barotropic(seeded, frames, masks)
    seeded = _bridge_tke(seeded, previous_tke["arrays"])
    seeded_fields = gate.lego_fields(seeded)
    input_identity = {}
    for tracer in ("T", "S"):
        record_kbb = np.ascontiguousarray(
            record["arrays"][f"{tracer}_Kbb"].transpose(1, 0, 2))
        input_identity[tracer] = field_stats(
            seeded_fields[tracer], record_kbb, masks[tracer])
        require(input_identity[tracer]["cells_unequal"] == 0,
                f"model {tracer} input is not the record's Kbb field")
    freshwater2, surface2 = gate._surface_forcings(card, seeded, 2)
    exposed = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_stage3_advection_content=True),
    ).step(seeded, dt=card.dt_s, freshwater=freshwater2,
           surface_forcing=surface2)
    fields = gate.lego_fields(exposed)
    live_content = {name: np.asarray(fields[name], dtype=np.float64)
                    for name in ("T", "S")}
    rows = _provisional_krhs_boundary_rows(
        live_content, record["arrays"], {name: masks[name]
                                         for name in ("T", "S")})
    first = {
        tracer: ("complete_fct_advection" if
                 row["after_complete_fct_advection_content"]["cells_unequal"]
                 else "not_reached")
        for tracer, row in rows.items()
    }
    zc = card.recipe.z_coord
    dtypes = {name: str(np.asarray(getattr(zc, name)).dtype)
              for name in ("t_depth_ref", "dz_ref", "z_full_ref",
                           "z_half_ref", "h_partial")}
    return {
        "format": "gyre-round64-provisional-krhs-walk-v1",
        "status": "NOT ADMISSIBLE -- PROVISIONAL READ ONLY",
        "reason": rejected["violations"],
        "worktree": stamp,
        "record_stamp": record_stamp,
        "record_commit": producer,
        "record": str(krhs_record),
        "admission": str(admission),
        "precision": {"jax_x64": bool(jax.config.jax_enable_x64),
                      "geometry_dtypes": dtypes,
                      "content_dtype": str(live_content["T"].dtype)},
        "record_calibration": calibration,
        "model_input_vs_record_Kbb": input_identity,
        "first_differing_boundary": first,
        "rows": rows,
    }


def _owned2(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[2:-2, 2:-2]


def _owned3(value, nlev: int = 30) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[2:-2, 2:-2, :nlev]


def field_stats(candidate, oracle, mask, weights=None) -> dict:
    """Exact and norm statistics on one explicit, non-empty wet mask."""
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(oracle, dtype=np.float64)
    wet = np.asarray(mask, dtype=bool)
    require(left.shape == right.shape == wet.shape, "field/mask shape mismatch")
    require(bool(wet.any()), "empty metric mask")
    require(np.isfinite(left[wet]).all() and np.isfinite(right[wet]).all(),
            "non-finite metric input")
    delta = left - right
    active_delta = delta[wet]
    unequal = int(np.count_nonzero(
        left[wet].view(np.uint64) != right[wet].view(np.uint64)))
    if weights is None:
        active_weights = np.ones(active_delta.shape, dtype=np.float64)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        require(weight_array.shape == wet.shape, "weight/mask shape mismatch")
        active_weights = weight_array[wet]
        require(np.isfinite(active_weights).all() and np.all(active_weights > 0.0),
                "metric weights must be finite and positive")
    return {
        "rms": float(np.sqrt(np.mean(active_delta * active_delta))),
        "max_abs": float(np.max(np.abs(active_delta))),
        "cells_unequal": unequal,
        "cells": int(wet.sum()),
        "weighted_signed_mean": float(
            np.sum(active_weights * active_delta) / np.sum(active_weights)),
        "sum_squared_error": float(np.sum(active_delta * active_delta)),
    }


def _substitute_zdf_operands(live: dict, oracle: dict, names=(), *,
                             content_rows: str | None = None) -> dict:
    """Return one explicit ZDF-boundary substitution without mutating inputs."""
    valid = {"K", "dz", "e3w", "wet", "content_T", "content_S"}
    requested = set(names)
    require(requested <= valid,
            f"unknown ZDF operand substitutions: {sorted(requested - valid)}")
    require(content_rows in (None, "surface", "interior", "bottom"),
            f"unknown content row selection: {content_rows}")
    result = {name: np.array(value, copy=True) for name, value in live.items()}
    for name in requested:
        require(name in oracle and result[name].shape == np.asarray(oracle[name]).shape,
                f"ZDF operand {name} is missing or shape-incompatible")
        result[name] = np.array(oracle[name], copy=True)
    if content_rows is not None:
        nlev = result["content_T"].shape[-1]
        levels = np.zeros(nlev, dtype=bool)
        if content_rows == "surface":
            levels[0] = True
        elif content_rows == "bottom":
            levels[-1] = True
        else:
            levels[1:-1] = True
        for name in ("content_T", "content_S"):
            result[name][..., levels] = np.asarray(oracle[name])[..., levels]
    return result


def _partitions(wet3: np.ndarray) -> tuple[dict, dict, dict]:
    """The preregistered level, vertical-role, and fixed-index region cuts."""
    wet = np.asarray(wet3, dtype=bool)
    require(wet.ndim == 3 and wet.shape[-1] > 2, "invalid tracer mask")
    wet2 = np.any(wet, axis=-1)
    js, is_ = np.nonzero(wet2)
    require(js.size > 0, "no wet columns")
    j_mid = (int(js.min()) + int(js.max()) + 1) / 2.0
    i_span = int(is_.max()) - int(is_.min()) + 1
    i_one = int(is_.min()) + i_span / 3.0
    i_two = int(is_.min()) + 2.0 * i_span / 3.0
    jj = np.arange(wet.shape[0])[:, None]
    ii = np.arange(wet.shape[1])[None, :]
    halves = {"south_half": wet2 & (jj < j_mid),
              "north_half": wet2 & (jj >= j_mid)}
    thirds = {"west_third": wet2 & (ii < i_one),
              "middle_third": wet2 & (ii >= i_one) & (ii < i_two),
              "east_third": wet2 & (ii >= i_two)}
    regions = {**halves, **thirds}
    for h_name, h_mask in halves.items():
        for t_name, t_mask in thirds.items():
            regions[f"{h_name}.{t_name}"] = h_mask & t_mask

    levels = {f"k{k:02d}": wet & (np.arange(wet.shape[-1])[None, None, :] == k)
              for k in range(wet.shape[-1])}
    bottom_index = np.max(
        np.where(wet, np.arange(wet.shape[-1])[None, None, :], -1), axis=-1)
    kk = np.arange(wet.shape[-1])[None, None, :]
    surface = wet & (kk == 0)
    bottom = wet & (kk == bottom_index[..., None])
    vertical = {"surface": surface, "interior": wet & ~surface & ~bottom,
                "bottom": bottom}
    require(sum(int(x.sum()) for x in vertical.values()) == int(wet.sum()),
            "vertical partitions overlap or omit wet cells")
    return levels, vertical, regions


def _tke_rows(state, stage_arrays: dict) -> dict:
    """Compare the independently advanced kt=2 TKE carry to recorded NEMO."""
    mapping = {
        "en": (state.tke, stage_arrays["tke_en"][..., 1:30]),
        "avm_k": (state.tke_avm, _owned3(stage_arrays["tke_avm_k"], 31)[..., 1:30]),
        "avt_k": (state.tke_avt, stage_arrays["tke_avt_k"][..., 1:30]),
        "dissl": (state.tke_dissl, stage_arrays["tke_dissl"][..., 1:30]),
    }
    wet_w = _owned3(stage_arrays["wmask"], 31)[..., 1:30] > 0.5
    rows = {}
    for name, (field, oracle) in mapping.items():
        require(field is not None, f"kt=2 state has no {name}")
        rows[name] = field_stats(field.data, oracle, wet_w)
    return rows


def _bridge_tke(state, arrays: dict):
    import jax.numpy as jnp
    from legoesm.core.field import Field

    def field(values, name, dims=("lat", "lon", "level")):
        return Field(data=jnp.asarray(values), name=name, dims=dims)

    avm = _owned3(arrays["tke_avm_k"], 31)
    return state._replace(
        tke=field(arrays["tke_en"][..., 1:30], "tke"),
        tke_avm=field(avm[..., 1:30], "tke_avm"),
        tke_avt=field(arrays["tke_avt_k"][..., 1:30], "tke_avt"),
        tke_dissl=field(arrays["tke_dissl"][..., 1:30], "tke_dissl"),
        tke_avm_surface=field(avm[..., 0], "tke_avm_surface", ("lat", "lon")),
    )


def _bridge_barotropic(state, frames: dict, masks: dict):
    import jax.numpy as jnp

    require(state.uu_b is not None and state.vv_b is not None,
            "GYRE card does not carry its barotropic pair")
    u = np.array(state.uu_b.data, dtype=np.float64, copy=True)
    v = np.array(state.vv_b.data, dtype=np.float64, copy=True)
    u[:, 1:] = np.where(masks["u"][..., 0], frames["uu_b"], u[:, 1:])
    v[1:, :] = np.where(masks["v"][..., 0], frames["vv_b"], v[1:, :])
    return state._replace(uu_b=state.uu_b.replace(data=jnp.asarray(u)),
                          vv_b=state.vv_b.replace(data=jnp.asarray(v)))


def _stage_reference(records: dict, entry3: dict, stage: int, field: str):
    if stage == 1:
        return _owned3(records[2]["arrays"][f"{field}_Kmm"])
    if stage == 2:
        return _owned3(records[3]["arrays"][f"{field}_Kmm"])
    if stage == 3:
        return np.asarray(entry3[field], dtype=np.float64)[..., :30]
    raise GateError(f"invalid stage {stage}")


def _stage_weights(records: dict, stage: int) -> np.ndarray:
    arrays = records[stage]["arrays"]
    thickness = (arrays["e3t_Kmm"] if stage < 3 else arrays["e3t_Kaa"])
    return _owned3(thickness) * _owned2(arrays["e1e2t"])[..., None]


def _peak_rows(candidate, oracle, wet_mask, count: int = 2) -> list[dict]:
    """Largest wet-cell absolute residuals, retaining their raw operands."""
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(oracle, dtype=np.float64)
    wet = np.asarray(wet_mask, dtype=bool)
    require(left.shape == right.shape == wet.shape, "peak field/mask shape mismatch")
    require(0 < count <= int(wet.sum()), "invalid wet peak count")
    difference = left - right
    ranked = np.where(wet, np.abs(difference), -np.inf).ravel()
    selected = np.argpartition(ranked, -count)[-count:]
    rows = []
    for index in selected[np.argsort(ranked[selected])[::-1]]:
        j, i, k = np.unravel_index(int(index), difference.shape)
        row = {"j": int(j), "i": int(i), "k": int(k),
               "delta": float(difference[j, i, k]),
               "candidate": float(left[j, i, k]),
               "oracle": float(right[j, i, k])}
        require(wet[j, i, k], "peak selector returned a dry cell")
        require(row["candidate"] - row["oracle"] == row["delta"],
                "peak spot-check does not reproduce raw-array subtraction")
        rows.append(row)
    return rows


def _run_arm(card, start, masks, records, entry3, *, tke_exact: bool) -> dict:
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from nemo_testcase_l2_gyre_phase3_gate import _surface_forcings, lego_fields

    freshwater, surface = _surface_forcings(card, start, 2)
    rows = {}
    outputs = {}
    for stage in (1, 2, 3):
        hooks = _NEMOWSRK3TestHooks(expose_tracer_stage=stage if stage < 3 else 0)
        out = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks,
        ).step(start, dt=card.dt_s, freshwater=freshwater,
               surface_forcing=surface)
        fields = lego_fields(out)
        outputs[stage] = {name: np.asarray(fields[name], dtype=np.float64)
                          for name in ("T", "S")}
        rows[f"stage{stage}"] = {
            name: field_stats(outputs[stage][name],
                              _stage_reference(records, entry3, stage, name),
                              masks[name], _stage_weights(records, stage))
            for name in ("T", "S")
        }
    rows["tke_exact_input"] = tke_exact
    return {"rows": rows, "outputs": outputs}


def measure(*, entry_root: Path, stage_root: Path, expect_commit: str,
            plant: str | None = None) -> dict:
    import jax
    import nemo_testcase_l2_gyre_phase3_gate as gate
    import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46
    import nemo_testcase_overflow_barotropic_gate as baro
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    expected = "0" * 40 if plant == "stamp" else expect_commit.lower()
    require(len(expected) == 40 and stamp["commit"].lower() == expected,
            f"commit stamp mismatch: {stamp['commit']} != {expected}")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")

    records = {}
    for stage in (1, 2, 3):
        read_plant = "truncation" if plant == "truncation" and stage == 1 else None
        records[stage] = round46.read_stage(
            stage_root / f"oracle_momstage_kt00000002_s{stage}.bin",
            plant=read_plant)
    if plant == "stage-swap":
        records[1], records[2] = records[2], records[1]
    for stage in (1, 2, 3):
        require(records[stage]["header"]["stage"] == stage,
                f"stage mapping is not self-consistent at stage {stage}")
    previous_tke = round46.read_stage(
        stage_root / "oracle_momstage_kt00000001_s1.bin")

    entry2 = gate.read_entry(entry_root / "oracle_step_entry_kt00000002.bin")
    entry3 = gate.read_entry(entry_root / "oracle_step_entry_kt00000003.bin")
    frames = gate.read_bt(entry_root / "oracle_bt_frames_kt00000001.bin", 1)
    card = build_nemo_testcase_card("GYRE-zco")
    masks = gate.expected_masks(card)
    base_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    initial = card.recipe.initial_state
    freshwater1, surface1 = gate._surface_forcings(card, initial, 1)
    free_entry = base_model.step(initial, dt=card.dt_s, freshwater=freshwater1,
                                 surface_forcing=surface1)
    tke_input = _tke_rows(free_entry, records[1]["arrays"])
    seeded = baro.state_from_oracle_entry(free_entry, entry2, masks)
    seeded = _bridge_barotropic(seeded, frames, masks)
    if plant == "input-noop":
        seeded = free_entry
    input_rows = {}
    seeded_fields = gate.lego_fields(seeded)
    for name in ("T", "S", "u", "v", "ssh"):
        ref = np.asarray(entry2[name], dtype=np.float64)
        if ref.ndim == 3:
            ref = ref[..., :np.asarray(seeded_fields[name]).shape[-1]]
        input_rows[name] = field_stats(seeded_fields[name], ref, masks[name])
        require(input_rows[name]["cells_unequal"] == 0,
                f"kt=2 input {name} is not bit-exact")

    carried = _run_arm(card, seeded, masks, records, entry3, tke_exact=False)
    # zdf_phy runs before the kt=2 stage records are opened.  The coefficient
    # memory entering kt=2 is therefore the kt=1 post-zdf_phy record, not the
    # kt=2 record (which is the value this step must reproduce).
    exact_start = _bridge_tke(seeded, previous_tke["arrays"])
    exact = _run_arm(card, exact_start, masks, records, entry3, tke_exact=True)

    primary = carried["rows"]["stage3"]
    if plant == "tracer-bit":
        primary = dict(primary)
        primary["T"] = dict(primary["T"])
        primary["T"]["rms"] += 1.0e-8
    require(abs(primary["T"]["rms"] - EXPECTED_T_RMS_K) <= ANCHOR_ATOL_K,
            f"reconciled kt3 T RMS moved: {primary['T']['rms']}")
    require(primary["T"]["cells_unequal"] == EXPECTED_T_UNEQUAL,
            f"reconciled kt3 T unequal count moved: {primary['T']['cells_unequal']}")

    final_t = carried["outputs"][3]["T"]
    oracle_t = _stage_reference(records, entry3, 3, "T")
    diff = final_t - oracle_t
    wet = masks["T"]
    total = float(np.sum(diff[wet] ** 2))
    require(total > 0.0, "self-comparison cannot decompose an exact array")
    levels, vertical, regions = _partitions(wet)

    def partition_rows(parts, *, expand_2d=False):
        result = {}
        for name, part in parts.items():
            mask = wet & (part[..., None] if expand_2d else part)
            row = field_stats(final_t, oracle_t, mask)
            row["share_of_sum_dT2"] = row["sum_squared_error"] / total
            result[name] = row
        return result

    peaks = _peak_rows(final_t, oracle_t, wet)

    report = {
        "format": "gyre-round54-step2-tracer-decomposition-v3",
        "retracts": {
            "formats": ["gyre-round54-step2-tracer-decomposition-v1",
                        "gyre-round54-step2-tracer-decomposition-v2"],
            "reason": (
                "v1 peak selection included dry cells.  v2 fixed that, but "
                "its auxiliary all-recorded-TKE arm seeded kt=2 from the "
                "kt=2 post-zdf_phy record and advanced it twice.  v3 seeds "
                "that arm from kt=1; the ordinary independent-model arm and "
                "all decomposition aggregates were unaffected"
            ),
        },
        "worktree": stamp,
        "entry_root": str(entry_root),
        "stage_root": str(stage_root),
        "plant": plant,
        "input": input_rows,
        "independently_advanced_tke_vs_nemo": tke_input,
        "arms": {"recorded_prognostics_with_legoesm_tke": carried["rows"],
                 "recorded_prognostics_and_kt1_tke_memory": exact["rows"]},
        "temperature_final": {
            "levels": partition_rows(levels),
            "vertical_role": partition_rows(vertical),
            "regions": partition_rows(regions, expand_2d=True),
            "peaks": peaks,
        },
    }
    return report


def self_check(plant: str | None = None) -> dict:
    oracle = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    candidate = oracle.copy()
    candidate[0, 0, 1] += 2.0
    mask = np.ones_like(oracle, dtype=bool)
    if plant == "self-compare":
        candidate = oracle
    row = field_stats(candidate, oracle, mask)
    require(row["cells_unequal"] == 1 and row["max_abs"] == 2.0,
            "synthetic metric did not recover its planted answer")
    levels, vertical, regions = _partitions(mask)
    require(len(levels) == 3 and len(vertical) == 3 and len(regions) == 11,
            "synthetic partitions have the wrong cardinality")
    return {"status": "PASS", "metric": row}


def zdf_score(*, entry_root: Path, stage_root: Path, record: Path,
              expect_commit: str, plant: str | None = None) -> dict:
    """Score kt=2 ZDF's recorded solve and the live model content operand."""
    import jax
    import jax.numpy as jnp
    import nemo_testcase_l2_gyre_phase3_gate as gate
    import nemo_testcase_l2_gyre_round35_trazdf_matrix as round35
    import nemo_testcase_l2_gyre_round38_matrix_operands as round38
    import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46
    import nemo_testcase_overflow_barotropic_gate as baro
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    expected = "0" * 40 if plant == "stamp" else expect_commit.lower()
    require(len(expected) == 40 and stamp["commit"].lower() == expected,
            f"commit stamp mismatch: {stamp['commit']} != {expected}")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")

    rec = round35.read_trazdf_matrix(record, expect_kt=2)
    header = rec["header"]
    nlev = header["jpkm1"]
    wet_raw = np.asarray(round35._box(rec, "tmask", nlev), dtype=np.float64) > 0.5
    direct = {"assembly": {}, "sweep": {}, "rhs_content": {}}
    assembled = round35.lego_assembly(rec)
    for name in ("zwi", "zwd", "zws"):
        direct["assembly"][name] = field_stats(
            assembled[name], round35._box(rec, name), wet_raw)
    swept = round35.lego_sweep(rec)
    content = round35.lego_rhs_content(rec)
    for name in ("T", "S"):
        direct["sweep"][name] = field_stats(
            swept[name], round35._box(rec, f"sol_{name}_pre_clamp", nlev), wet_raw)
        direct["rhs_content"][name] = field_stats(
            content[f"faithful_{name}"], round35._box(rec, f"rhs_{name}", nlev),
            wet_raw)

    transpose = lambda value: np.ascontiguousarray(  # noqa: E731
        np.asarray(value, dtype=np.float64).transpose(1, 0, 2))
    nemo_rhs = {name: transpose(round35._box(rec, f"rhs_{name}", nlev))
                for name in ("T", "S")}
    nemo_solution = {
        "T": transpose(round35._box(rec, "sol_T_pre_clamp", nlev)),
        "S": transpose(round35._box(rec, "sol_S_post_clamp", nlev)),
    }

    records = {stage: round46.read_stage(
        stage_root / f"oracle_momstage_kt00000002_s{stage}.bin")
        for stage in (1, 2, 3)}
    previous_tke = round46.read_stage(
        stage_root / "oracle_momstage_kt00000001_s1.bin")
    entry2 = gate.read_entry(entry_root / "oracle_step_entry_kt00000002.bin")
    entry3 = gate.read_entry(entry_root / "oracle_step_entry_kt00000003.bin")
    frames = gate.read_bt(entry_root / "oracle_bt_frames_kt00000001.bin", 1)
    card = build_nemo_testcase_card("GYRE-zco")
    masks = gate.expected_masks(card)
    plain_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    initial = card.recipe.initial_state
    freshwater1, surface1 = gate._surface_forcings(card, initial, 1)
    free_entry = plain_model.step(initial, dt=card.dt_s, freshwater=freshwater1,
                                  surface_forcing=surface1)
    seeded = baro.state_from_oracle_entry(free_entry, entry2, masks)
    seeded = _bridge_barotropic(seeded, frames, masks)
    seeded = _bridge_tke(seeded, previous_tke["arrays"])
    freshwater2, surface2 = gate._surface_forcings(card, seeded, 2)

    def run_hooks(hooks):
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks,
        ).step(seeded, dt=card.dt_s, freshwater=freshwater2,
               surface_forcing=surface2)

    exposed = run_hooks(_NEMOWSRK3TestHooks(expose_pre_implicit_content=True))
    exposed_fields = gate.lego_fields(exposed)
    model_content = {
        name: field_stats(exposed_fields[name], nemo_rhs[name], masks[name])
        for name in ("T", "S")
    }
    if plant == "content":
        model_content["T"] = dict(model_content["T"])
        model_content["T"]["max_abs"] += 1.0

    import legoesm.ocean.physics.vertical_mixing as vmix

    caught: dict[str, list[np.ndarray]] = {}
    real_dispatch = vmix.implicit_vertical_diffusion_ocean_tracer_pair_dispatch

    def sink(name):
        return lambda value: caught.setdefault(name, []).append(
            np.asarray(value, dtype=np.float64))

    def capture_dispatch(f1, f2, c1, c2, k, dz_after, e3w_now, dt, wet, **kw):
        for name, value in (("K", k), ("dz", dz_after), ("e3w", e3w_now),
                            ("content_T", c1), ("content_S", c2),
                            ("wet", jnp.asarray(wet, dtype=k.dtype))):
            jax.debug.callback(sink(name), value)
        return real_dispatch(f1, f2, c1, c2, k, dz_after, e3w_now, dt, wet, **kw)

    vmix.implicit_vertical_diffusion_ocean_tracer_pair_dispatch = capture_dispatch
    try:
        run_hooks(_NEMOWSRK3TestHooks())
    finally:
        vmix.implicit_vertical_diffusion_ocean_tracer_pair_dispatch = real_dispatch
    require(set(caught) == {"K", "dz", "e3w", "content_T", "content_S", "wet"},
            f"live ZDF capture missed an operand: {sorted(caught)}")
    require(all(len(values) == 1 for values in caught.values()),
            "live ZDF capture did not fire exactly once")
    live = {name: values[0] for name, values in caught.items()}
    oracle_operands = round38._oracle(rec)
    wet_cell = oracle_operands["wet"] > 0.5
    wet_face = wet_cell[..., 1:] & wet_cell[..., :-1]
    live_rows = {
        "K": field_stats(live["K"], oracle_operands["K"], wet_face),
        # The carried field is zdfphy's pre-EVD ``avt_k``; the record's
        # ``avt`` is post-zdf_phy and therefore contains rn_evd=100 where the
        # trigger fires.  Keep that stage transition explicit in the name.
        "carried_avt_k_vs_post_zdf_phy_avt": field_stats(
            seeded.tke_avt.data, oracle_operands["avt"], wet_face),
        "dz_after": field_stats(live["dz"], oracle_operands["dz"], wet_cell),
        "e3w_now": field_stats(live["e3w"], oracle_operands["e3w"], wet_face),
        "wet": field_stats(live["wet"], oracle_operands["wet"], wet_cell),
        "content_T": field_stats(live["content_T"], nemo_rhs["T"], wet_cell),
        "content_S": field_stats(live["content_S"], nemo_rhs["S"], wet_cell),
    }
    from legoesm.ocean.physics.vertical_mixing import nemo_tracer_tridiagonal

    live_matrix = nemo_tracer_tridiagonal(
        jnp.asarray(live["K"]), jnp.asarray(live["dz"]),
        jnp.asarray(live["e3w"]), float(rec["arrays"]["rDt"]),
        jnp.asarray(live["wet"]) > 0.5, dtype=jnp.float64)
    live_matrix_rows = {
        name: field_stats(np.asarray(value), oracle_operands[name], wet_cell)
        for name, value in zip(("zwi", "zwd", "zws"), live_matrix, strict=True)
    }

    # rn_evd is exactly 100 on this resolved card (namelist_cfg:210).  These
    # frozen source-defined partitions distinguish a trigger disagreement
    # from the stable TKE-coefficient debt without inventing an error contour.
    oracle_evd = wet_face & (oracle_operands["avt"] == 100.0)
    oracle_stable = wet_face & ~oracle_evd
    live_k_partitions = {
        "oracle_evd_100": field_stats(
            live["K"], oracle_operands["K"], oracle_evd),
        "oracle_stable": field_stats(
            live["K"], oracle_operands["K"], oracle_stable),
    }

    def run_dispatch_arm(operands: dict, *, direct_solution: bool = False):
        """Re-enter the shared production dispatcher at the captured seam.

        The ordinary model step above supplies and stamps every live operand.
        Re-entering only this already-reached shared boundary avoids compiling
        the complete model once per arm (which exhausted LLVM memory) while
        retaining the production solver rather than a detached transcription.
        """
        if direct_solution:
            return {name: np.asarray(nemo_solution[name])
                    for name in ("T", "S")}

        def dispatch():
            return real_dispatch(
                jnp.zeros_like(jnp.asarray(operands["content_T"])),
                jnp.zeros_like(jnp.asarray(operands["content_S"])),
                jnp.asarray(operands["content_T"], dtype=jnp.float64),
                jnp.asarray(operands["content_S"], dtype=jnp.float64),
                jnp.asarray(operands["K"], dtype=jnp.float64),
                jnp.asarray(operands["dz"], dtype=jnp.float64),
                jnp.asarray(operands["e3w"], dtype=jnp.float64),
                float(rec["arrays"]["rDt"]),
                jnp.asarray(operands["wet"]) > 0.5,
                evaluation="nemo_literal", implicit_w=None)

        out_t, out_s = jax.jit(dispatch)()
        return {"T": np.asarray(out_t), "S": np.asarray(out_s)}

    baseline_state = run_hooks(_NEMOWSRK3TestHooks())
    baseline = gate.lego_fields(baseline_state)
    live_for_arms = {
        "K": live["K"], "dz": live["dz"], "e3w": live["e3w"],
        "wet": live["wet"], "content_T": live["content_T"],
        "content_S": live["content_S"],
    }
    oracle_for_arms = {
        "K": oracle_operands["K"], "dz": oracle_operands["dz"],
        "e3w": oracle_operands["e3w"], "wet": oracle_operands["wet"],
        "content_T": nemo_rhs["T"], "content_S": nemo_rhs["S"],
    }
    plant_detection = None
    if plant == "operand-ulp":
        planted = np.array(oracle_for_arms["content_T"], copy=True)
        first_wet = tuple(np.argwhere(wet_cell)[0])
        planted[first_wet] = np.nextafter(planted[first_wet], np.float64(np.inf))
        plant_detection = field_stats(
            planted, oracle_for_arms["content_T"], wet_cell)
        require(plant_detection["cells_unequal"] == 1
                and plant_detection["max_abs"] > 0.0,
                "one-ulp ZDF operand plant was not detected")
        oracle_for_arms["content_T"] = planted

    oracle_evd_k = _substitute_zdf_operands(live_for_arms, oracle_for_arms)
    oracle_evd_k["K"][oracle_evd] = oracle_for_arms["K"][oracle_evd]
    oracle_stable_k = _substitute_zdf_operands(live_for_arms, oracle_for_arms)
    oracle_stable_k["K"][oracle_stable] = oracle_for_arms["K"][oracle_stable]
    zero_k = _substitute_zdf_operands(live_for_arms, oracle_for_arms)
    zero_k["K"] = np.zeros_like(zero_k["K"])
    arm_operands = {
        "nemo_content_substitution": _substitute_zdf_operands(
            live_for_arms, oracle_for_arms, ("content_T", "content_S")),
        "nemo_content_surface_row_substitution": _substitute_zdf_operands(
            live_for_arms, oracle_for_arms, content_rows="surface"),
        "nemo_content_interior_rows_substitution": _substitute_zdf_operands(
            live_for_arms, oracle_for_arms, content_rows="interior"),
        "nemo_content_bottom_row_substitution": _substitute_zdf_operands(
            live_for_arms, oracle_for_arms, content_rows="bottom"),
        "nemo_effective_K_substitution": _substitute_zdf_operands(
            live_for_arms, oracle_for_arms, ("K",)),
        "nemo_evd_K_interfaces_substitution": oracle_evd_k,
        "nemo_stable_K_interfaces_substitution": oracle_stable_k,
        "nemo_e3t_Kaa_substitution": _substitute_zdf_operands(
            live_for_arms, oracle_for_arms, ("dz",)),
        "nemo_e3w_Kmm_substitution": _substitute_zdf_operands(
            live_for_arms, oracle_for_arms, ("e3w",)),
        "nemo_wet_substitution": _substitute_zdf_operands(
            live_for_arms, oracle_for_arms, ("wet",)),
        "nemo_content_plus_K": _substitute_zdf_operands(
            live_for_arms, oracle_for_arms,
            ("content_T", "content_S", "K")),
        "nemo_content_K_plus_e3t_Kaa": _substitute_zdf_operands(
            live_for_arms, oracle_for_arms,
            ("content_T", "content_S", "K", "dz")),
        "nemo_content_K_e3t_plus_e3w": _substitute_zdf_operands(
            live_for_arms, oracle_for_arms,
            ("content_T", "content_S", "K", "dz", "e3w")),
        "zero_effective_K_ablation": zero_k,
        "all_nemo_zdf_operands_substitution": _substitute_zdf_operands(
            live_for_arms, oracle_for_arms, tuple(oracle_for_arms)),
    }
    final_fields = {name: run_dispatch_arm(operands)
                    for name, operands in arm_operands.items()}
    # If the all-operand arm remains above bar, this boundary discriminator
    # says whether the owner is inside the solve or in the caller's mask/
    # return statements immediately after it.
    final_fields["nemo_recorded_solution_return"] = run_dispatch_arm(
        live_for_arms, direct_solution=True)
    final_rows = {}
    for arm, fields in (("baseline_all_recorded", baseline),
                        *final_fields.items()):
        final_rows[arm] = {
            name: field_stats(fields[name], entry3[name][..., :nlev], masks[name])
            for name in ("T", "S")
        }
        for name in ("T", "S"):
            scale = float(np.max(np.abs(entry3[name][..., :nlev][masks[name]])))
            final_rows[arm][name]["normalized_max_abs"] = (
                final_rows[arm][name]["max_abs"] / scale)
            final_rows[arm][name]["bar"] = 1.0e-15
            final_rows[arm][name]["status"] = (
                "AT-BAR" if final_rows[arm][name]["normalized_max_abs"] <= 1.0e-15
                else "DEBT")
    record_to_entry = {
        name: field_stats(nemo_solution[name], entry3[name][..., :nlev], masks[name])
        for name in ("T", "S")
    }
    require(all(row["cells_unequal"] == 0 for row in record_to_entry.values()),
            "round-38 solution is not the year-owner kt=3 entry")
    if plant == "content":
        require(model_content["T"]["max_abs"] > 0.5,
                "content plant did not move its row")

    return {
        "format": "gyre-round54-kt2-zdf-score-v3",
        "retracts": {
            "formats": ["gyre-round54-kt2-zdf-score-v1",
                        "gyre-round54-kt2-zdf-score-v2"],
            "reason": (
                "those artifacts seeded kt=2 from its post-zdf_phy TKE "
                "record; v3 uses kt=1 post-zdf_phy memory, the value that "
                "actually enters the kt=2 closure"
            ),
        },
        "worktree": stamp,
        "record": str(record),
        "direct_given_nemo_inputs": direct,
        "record_solution_to_kt3_entry": record_to_entry,
        "model_path_content_rhs": model_content,
        "model_path_live_operands": live_rows,
        "model_path_live_K_source_partitions": live_k_partitions,
        "model_path_tke_output": _tke_rows(
            baseline_state, records[1]["arrays"]),
        "model_path_live_matrix": live_matrix_rows,
        "final_rows": final_rows,
        "operator_plant_detection": plant_detection,
        "plant": plant,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("measure", "self-check", "zdf-score", "krhs-calibrate",
                 "krhs-preview"),
        default="measure")
    parser.add_argument("--entry-root", type=Path, default=ENTRY_ROOT)
    parser.add_argument("--stage-root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--zdf-record", type=Path, default=ZDF_RECORD)
    parser.add_argument("--krhs-record", type=Path)
    parser.add_argument("--tke-rhs-record", type=Path)
    parser.add_argument("--krhs-stamp", type=Path)
    parser.add_argument("--tke-rhs-stamp", type=Path)
    parser.add_argument("--producer-commit", type=Path)
    parser.add_argument("--resolved-output", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--expect-record-commit")
    parser.add_argument("--expect-itend", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=(
        "stamp", "truncation", "stage-swap", "tracer-bit", "input-noop",
        "self-compare", "content", "operand-ulp", "krhs-ulp", "tke-ulp",
        "krhs-truncation", "tke-truncation"))
    args = parser.parse_args(argv)
    try:
        if args.mode == "self-check":
            report = self_check(args.plant)
        elif args.mode == "measure":
            require(args.expect_commit is not None,
                    "--expect-commit is required for a measurement")
            require(args.plant != "self-compare",
                    "self-compare is a self-check-only plant")
            report = measure(entry_root=args.entry_root,
                             stage_root=args.stage_root,
                             expect_commit=args.expect_commit,
                             plant=args.plant)
        elif args.mode == "zdf-score":
            require(args.expect_commit is not None,
                    "--expect-commit is required for a measurement")
            require(args.plant in (None, "stamp", "content", "operand-ulp"),
                    "zdf-score accepts only stamp, content, or operand-ulp plants")
            report = zdf_score(
                entry_root=args.entry_root, stage_root=args.stage_root,
                record=args.zdf_record, expect_commit=args.expect_commit,
                plant=args.plant)
        elif args.mode == "krhs-calibrate":
            require(args.expect_commit is not None,
                    "--expect-commit is required for round-63 calibration")
            require(all(value is not None for value in (
                args.krhs_record, args.tke_rhs_record, args.krhs_stamp,
                args.tke_rhs_stamp, args.producer_commit,
                args.resolved_output)),
                "round-63 calibration requires both records, both stamps, "
                "and --producer-commit")
            require(args.plant in (
                None, "stamp", "krhs-ulp", "tke-ulp",
                "krhs-truncation", "tke-truncation"),
                "invalid round-63 calibration plant")
            report = r63_calibrate(
                krhs_record=args.krhs_record,
                tke_record=args.tke_rhs_record,
                krhs_stamp=args.krhs_stamp,
                tke_stamp=args.tke_rhs_stamp,
                expect_commit=args.expect_commit,
                producer_commit=args.producer_commit,
                resolved_output=args.resolved_output,
                expect_itend=args.expect_itend,
                plant=args.plant)
        else:
            require(args.expect_commit is not None,
                    "--expect-commit is required for a provisional preview")
            require(args.expect_record_commit is not None,
                    "--expect-record-commit is required for a provisional preview")
            require(all(value is not None for value in (
                args.krhs_record, args.krhs_stamp, args.producer_commit,
                args.admission)),
                "krhs-preview requires record, stamp, producer, and admission")
            require(args.plant is None,
                    "krhs-preview has no post-hoc plant selector")
            report = provisional_krhs_walk(
                entry_root=args.entry_root, stage_root=args.stage_root,
                krhs_record=args.krhs_record, krhs_stamp=args.krhs_stamp,
                producer_commit=args.producer_commit,
                admission=args.admission, expect_commit=args.expect_commit,
                expect_record_commit=args.expect_record_commit)
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        if args.plant:
            print(f"STATUS PLANT-FIRED: {args.plant}")
            return 1
        print(f"STATUS {report.get('status', 'PASS')}")
        return 0
    except (GateError, RuntimeError, AssertionError, OSError, ValueError) as error:
        if (args.plant == "self-compare"
                and str(error)
                == "synthetic metric did not recover its planted answer"):
            print("STATUS PLANT-FIRED: self-compare")
            return 1
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
