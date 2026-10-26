#!/usr/bin/env python3
"""Round-50 literal ``dynldf_lev`` replay and first-intermediate gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import _surface_forcings, require
from nemo_testcase_l2_gyre_round46_kt2_stage_gate import (
    ROOT,
    _owned2,
    _owned3,
    read_stage,
    sha256,
)


SOURCE = (
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/"
    "dynldf_lev.f90:121-140"
)

DEVELOPED_MAGIC = "NEMO_L2_R148LDF"
DEVELOPED_RECORD = "oracle_developed_ldf_kt00001081.bin"
DEVELOPED_NX, DEVELOPED_NY, DEVELOPED_NZ = 36, 26, 31
DEVELOPED_COUNT = DEVELOPED_NX * DEVELOPED_NY * DEVELOPED_NZ
DEVELOPED_SURFACE_COUNT = DEVELOPED_NX * DEVELOPED_NY
DEVELOPED_FIELDS = (
    ("u_kbb", "3d"), ("v_kbb", "3d"),
    ("tmask", "3d"), ("umask", "3d"),
    ("vmask", "3d"), ("fmask", "3d"),
    ("ahmt", "3d"), ("ahmf", "3d"),
    ("e3t_kbb", "3d"), ("e3u_kbb", "3d"),
    ("e3v_kbb", "3d"), ("e3f_live", "3d"),
    ("e3u_kmm", "3d"), ("e3v_kmm", "3d"),
    ("pre_u", "3d"), ("pre_v", "3d"),
    ("e2u", "2d"), ("e1v", "2d"),
    ("e2v", "2d"), ("e1u", "2d"),
    ("r1_e1e2t", "2d"), ("r1_e1e2f", "2d"),
    ("r1_e1u", "2d"), ("r1_e2v", "2d"),
    ("r1_e2u", "2d"), ("r1_e1v", "2d"),
    ("zcur", "3d"), ("zdiv", "3d"),
    ("post_u", "3d"), ("post_v", "3d"),
)
DEVELOPED_EXPECTED_SIZE = (
    16 + 9 * 4 + len(DEVELOPED_FIELDS) * 4
    + (20 * DEVELOPED_COUNT + 10 * DEVELOPED_SURFACE_COUNT) * 8
)


def read_developed_ldf_bytes(payload: bytes) -> dict:
    """Decode the Round-148 direct LDF stream with a closed field census."""
    require(len(payload) == DEVELOPED_EXPECTED_SIZE,
            f"developed LDF record is {len(payload)} bytes, expected "
            f"{DEVELOPED_EXPECTED_SIZE}")
    magic = payload[:16].decode("ascii").rstrip()
    header = struct.unpack_from("=9i", payload, 16)
    expected_header = (1, 1081, 1, 1, 3, DEVELOPED_NX,
                       DEVELOPED_NY, DEVELOPED_NZ, 64)
    require(magic == DEVELOPED_MAGIC and header == expected_header,
            f"bad developed LDF header: {(magic, header)!r}")
    offset = 16 + 9 * 4
    sizes = struct.unpack_from(f"={len(DEVELOPED_FIELDS)}i", payload, offset)
    expected_sizes = tuple(
        DEVELOPED_COUNT if kind == "3d" else DEVELOPED_SURFACE_COUNT
        for _, kind in DEVELOPED_FIELDS)
    require(sizes == expected_sizes,
            f"bad developed LDF field sizes: {sizes!r}")
    offset += len(DEVELOPED_FIELDS) * 4
    fields, offsets = {}, {}
    for (name, kind), count in zip(DEVELOPED_FIELDS, sizes):
        offsets[name] = offset
        end = offset + count * 8
        require(end <= len(payload), f"truncated developed LDF field {name}")
        values = np.frombuffer(payload[offset:end], dtype=np.float64).copy()
        require(values.size == count and np.all(np.isfinite(values)),
                f"non-finite or incomplete developed LDF field {name}")
        if kind == "3d":
            fields[name] = values.reshape(
                (DEVELOPED_NX, DEVELOPED_NY, DEVELOPED_NZ),
                order="F").transpose(1, 0, 2)
        else:
            fields[name] = values.reshape(
                (DEVELOPED_NX, DEVELOPED_NY), order="F").T
        offset = end
    require(offset == len(payload), "trailing developed LDF payload")
    require(tuple(fields) == tuple(name for name, _ in DEVELOPED_FIELDS),
            "developed LDF field registry changed")
    return {
        "header": {"magic": magic, "version": header[0], "kt": header[1],
                   "Kbb": header[2], "Kmm": header[3], "Krhs": header[4],
                   "nx": header[5], "ny": header[6], "nz": header[7],
                   "bits": header[8]},
        "fields": fields,
        "offsets": offsets,
    }


def _bit_identity(got: np.ndarray, reference: np.ndarray) -> dict:
    require(got.shape == reference.shape,
            f"identity shape mismatch: {got.shape} versus {reference.shape}")
    unequal = got.view(np.uint64) != reference.view(np.uint64)
    return {
        "bit_exact": not bool(np.any(unequal)),
        "differing_cells": int(np.count_nonzero(unequal)),
        "max_abs": float(np.max(np.abs(got - reference))) if got.size else 0.0,
    }


def admit_developed_ldf_record(
    record_path: Path,
    family_path: Path,
    stamp_path: Path,
    *,
    expect_commit: str,
    plant: str | None = None,
    parent_path: Path | None = None,
    parent_baseline_path: Path | None = None,
    family_baseline_path: Path | None = None,
    inherited_root: Path | None = None,
    inherited_baseline_root: Path | None = None,
) -> dict:
    """Admit the passive direct record against its family boundaries."""
    from nemo_testcase_l2_gyre_round146_rhs_family_gate import (
        _exact_inherited,
        _parent_comparison,
        read_record_bytes,
        read_round140_bytes,
    )

    original = record_path.read_bytes()
    original_record = read_developed_ldf_bytes(original)
    payload = bytearray(original)
    if plant == "header":
        payload[0] ^= 1
    elif plant == "truncation":
        del payload[-8:]
    elif plant == "missing-field":
        struct.pack_into("=i", payload, 16 + 9 * 4
                         + (len(DEVELOPED_FIELDS) - 1) * 4, 0)
    elif plant in ("zcur-ulp", "post-ulp"):
        name = "zcur" if plant == "zcur-ulp" else "post_u"
        offset = original_record["offsets"][name]
        value = struct.unpack_from("=d", payload, offset)[0]
        struct.pack_into("=d", payload, offset, np.nextafter(value, np.inf))
    elif plant not in (None, "parent-wet-ulp", "parent-dry-ulp",
                       "restart-byte"):
        raise ValueError(f"unknown developed LDF plant {plant!r}")

    record = read_developed_ldf_bytes(bytes(payload))
    stamp_rows = stamp_path.read_text().split()
    require(len(stamp_rows) == 3, "bad developed LDF stamp")
    expected_sha, stamped_commit, stamped_name = stamp_rows
    actual_sha = hashlib.sha256(bytes(payload)).hexdigest()
    require((actual_sha, stamped_commit, stamped_name)
            == (expected_sha, expect_commit, record_path.name),
            "developed LDF stamp mismatch")

    family_payload = family_path.read_bytes()
    family = read_record_bytes(family_payload)["fields"]
    fields = record["fields"]
    masks = {"u": fields["umask"] != 0.0,
             "v": fields["vmask"] != 0.0}
    boundaries = {}
    for face in ("u", "v"):
        mask = masks[face]
        boundaries[f"pre_{face}"] = _bit_identity(
            fields[f"pre_{face}"][mask], family[f"after_hpg_{face}"][mask])
        boundaries[f"post_{face}"] = _bit_identity(
            fields[f"post_{face}"][mask], family[f"after_ldf_{face}"][mask])
    require(all(row["bit_exact"] for row in boundaries.values()),
            f"developed LDF boundaries do not close: {boundaries}")

    zcur_owned = np.zeros_like(fields["zcur"], dtype=bool)
    zdiv_owned = np.zeros_like(fields["zdiv"], dtype=bool)
    zcur_owned[0:25, 0:35, :30] = True
    zdiv_owned[1:26, 1:36, :30] = True
    require(np.all(np.isfinite(fields["zcur"][zcur_owned]))
            and np.all(np.isfinite(fields["zdiv"][zdiv_owned])),
            "non-finite owned developed LDF intermediate")
    report = {
        "format": "nemo-testcase-l2-gyre-round148-developed-ldf-admission-v1",
        "record_sha256": actual_sha,
        "record_size": len(payload),
        "header": record["header"],
        "field_registry": list(fields),
        "boundaries": boundaries,
        "owned_intermediate_cells": {
            "zcur": int(np.count_nonzero(zcur_owned)),
            "zdiv": int(np.count_nonzero(zdiv_owned)),
        },
        "status": "PASS",
    }
    parent_args = (parent_path, parent_baseline_path, family_baseline_path,
                   inherited_root, inherited_baseline_root)
    require(all(value is not None for value in parent_args)
            or all(value is None for value in parent_args),
            "developed run admission paths must be supplied together")
    if parent_path is not None:
        candidate_parent = read_round140_bytes(parent_path.read_bytes())
        baseline_parent = read_round140_bytes(
            parent_baseline_path.read_bytes())
        if plant in ("parent-wet-ulp", "parent-dry-ulp"):
            mask = candidate_parent["fields"]["umask"] != 0.0
            target = mask if plant == "parent-wet-ulp" else ~mask
            locations = np.argwhere(target)
            require(locations.size > 0, "parent plant has no target cell")
            index = tuple(int(value) for value in locations[0])
            value = candidate_parent["fields"]["rhs_u"][index]
            candidate_parent["fields"]["rhs_u"][index] = np.nextafter(
                value, np.inf)
        parent = _parent_comparison(candidate_parent, baseline_parent)
        report["parent_vs_round146"] = parent
        require(parent["passive"],
                "Round-148 parent moved on a model-owned cell")
        if plant == "parent-dry-ulp":
            require(not parent["rows"]["rhs_u"]["excluded"]["bit_exact"]
                    and parent["rows"]["rhs_u"]["owned"]["bit_exact"],
                    "excluded parent plant was not classified")
            raise RuntimeError(
                "excluded-cell change classified without admitting an "
                "owned-cell change")
        family_baseline_payload = family_baseline_path.read_bytes()
        report["family_vs_round146"] = {
            "bit_exact": family_payload == family_baseline_payload,
            "candidate_sha256": hashlib.sha256(family_payload).hexdigest(),
            "baseline_sha256": hashlib.sha256(
                family_baseline_payload).hexdigest(),
        }
        require(report["family_vs_round146"]["bit_exact"],
                "Round-146 family record moved")
        report["exact_inherited"] = _exact_inherited(
            inherited_root, inherited_baseline_root,
            plant_restart=plant == "restart-byte")
    return report


def _u_face(value: np.ndarray) -> np.ndarray:
    value = _owned3(value)
    return np.concatenate((value[:, -1:, :], value), axis=1)


def _v_face(value: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
    value = _owned3(value)
    return np.concatenate((np.full_like(value[:1], fill), value), axis=0)


def _op(name: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """One materialized fp64 source operation."""
    if name == "mul":
        out = np.multiply(left, right)
    elif name == "add":
        out = np.add(left, right)
    elif name == "sub":
        out = np.subtract(left, right)
    elif name == "div":
        out = np.divide(left, right)
    else:  # pragma: no cover - internal dispatch is closed above
        raise ValueError(name)
    return np.asarray(out, dtype=np.float64)


def literal_ldf_numpy(
    u: np.ndarray,
    v: np.ndarray,
    *,
    e3t_kbb: np.ndarray,
    e3u_kbb: np.ndarray,
    e3v_kbb: np.ndarray,
    e3f_live: np.ndarray,
    e3u_kmm: np.ndarray,
    e3v_kmm: np.ndarray,
    e2u: np.ndarray,
    e1v: np.ndarray,
    e2v: np.ndarray,
    e1u: np.ndarray,
    r1_e1e2t: np.ndarray,
    r1_e1e2f: np.ndarray,
    r1_e1u: np.ndarray,
    r1_e2v: np.ndarray,
    r1_e2u: np.ndarray,
    r1_e1v: np.ndarray,
    ahmt: np.ndarray,
    ahmf: np.ndarray,
    tmask: np.ndarray,
    umask: np.ndarray,
    vmask: np.ndarray,
    fmask: np.ndarray,
    plant: bool = False,
) -> dict[str, np.ndarray]:
    """Replay compiled lines 121--140 with one NumPy ufunc per operation."""
    zu = _op("mul", e2u[..., None], e3u_kbb)
    zu = _op("mul", zu, u)
    zv = _op("mul", e1v[..., None], e3v_kbb)
    zv = _op("mul", zv, v)
    div_i = _op("sub", zu[:, 1:], zu[:, :-1])
    div_j = _op("sub", zv[1:], zv[:-1])
    div_bracket = _op("add", div_i, div_j)
    zdiv_scale = _op("mul", ahmt, r1_e1e2t[..., None])
    zdiv_scale = _op("div", zdiv_scale, np.where(e3t_kbb > 0, e3t_kbb, 1.0))
    zdiv = _op("mul", zdiv_scale, div_bracket)

    cur_v = _op("mul", e2v[..., None], v)
    cur_u = _op("mul", e1u[..., None], u)
    cur_v_pad = np.pad(cur_v, ((0, 0), (1, 1), (0, 0)), mode="wrap")
    cur_di = _op("sub", cur_v_pad[:, 1:], cur_v_pad[:, :-1])
    cur_u_pad = np.pad(cur_u, ((1, 1), (0, 0), (0, 0)))
    cur_dj = _op("sub", cur_u_pad[:-1], cur_u_pad[1:])
    cur_bracket = _op("add", cur_di, cur_dj)
    zcur_scale = _op("mul", ahmf, e3f_live)
    zcur_scale = _op("mul", zcur_scale, r1_e1e2f[..., None])
    zcur = _op("mul", zcur_scale, cur_bracket)
    if plant:
        wet = fmask > 0
        require(bool(wet.any()), "LDF plant has no wet F point")
        target = tuple(np.argwhere(wet)[0])
        zcur[target] = np.nextafter(zcur[target], np.inf)

    zdiv_i = _op(
        "sub",
        np.pad(zdiv, ((0, 0), (1, 1), (0, 0)), mode="wrap")[:, 1:],
        np.pad(zdiv, ((0, 0), (1, 1), (0, 0)), mode="wrap")[:, :-1],
    )
    grad_div_u = _op("mul", zdiv_i, r1_e1u[..., None])
    zdiv_j = _op(
        "sub",
        np.pad(zdiv, ((1, 1), (0, 0), (0, 0)))[1:],
        np.pad(zdiv, ((1, 1), (0, 0), (0, 0)))[:-1],
    )
    grad_div_v = _op("mul", zdiv_j, r1_e2v[..., None])
    grad_div_v[[0, -1], ...] = 0.0

    zcur_j = _op("sub", zcur[1:], zcur[:-1])
    curl_u = _op("mul", -zcur_j, r1_e2u[..., None])
    curl_u = _op("div", curl_u, np.where(e3u_kmm > 0, e3u_kmm, 1.0))
    zcur_i = _op("sub", zcur[:, 1:], zcur[:, :-1])
    curl_v = _op("mul", zcur_i, r1_e1v[..., None])
    curl_v = _op("div", curl_v, np.where(e3v_kmm > 0, e3v_kmm, 1.0))
    curl_v[[0, -1], ...] = 0.0
    visc_u = _op("mul", _op("add", curl_u, grad_div_u), umask)
    visc_v = _op("mul", _op("add", curl_v, grad_div_v), vmask)
    return {
        "div_u_product": zu,
        "div_v_product": zv,
        "div_bracket": div_bracket,
        "zdiv_scale": zdiv_scale,
        "zdiv": zdiv,
        "curl_u_product": cur_u,
        "curl_v_product": cur_v,
        "curl_bracket": cur_bracket,
        "zcur_scale": zcur_scale,
        "zcur": zcur,
        "grad_div_u": grad_div_u,
        "grad_div_v": grad_div_v,
        "curl_u": curl_u,
        "curl_v": curl_v,
        "visc_u": visc_u,
        "visc_v": visc_v,
    }


def _ordered_bits(value: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(value, dtype=np.float64).view(np.int64)
    return np.where(bits < 0, np.iinfo(np.int64).min - bits, bits)


def _score(name: str, got: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> dict:
    unequal = (got != ref) & mask
    delta = got[mask] - ref[mask]
    ulp = _ordered_bits(got[unequal]) - _ordered_bits(ref[unequal])
    return {
        "name": name,
        "n": int(mask.sum()),
        "n_unequal": int(unequal.sum()),
        "max_abs": float(np.max(np.abs(delta))),
        "signed_ulp_histogram": dict(sorted(Counter(map(int, ulp)).items())),
    }


def _spatial(name: str, got: np.ndarray, ref: np.ndarray, wet: np.ndarray) -> dict:
    unequal = (got != ref) & wet
    west = np.roll(wet, 1, axis=1)
    east = np.roll(wet, -1, axis=1)
    south = np.concatenate((np.zeros_like(wet[:1]), wet[:-1]), axis=0)
    north = np.concatenate((wet[1:], np.zeros_like(wet[:1])), axis=0)
    ring = wet & ~(west & east & south & north)
    corner = wet & ((~west | ~east) & (~south | ~north))
    by_j = np.count_nonzero(unequal, axis=(1, 2))
    return {
        "name": name,
        "first_wet_ring": int(np.count_nonzero(unequal & ring)),
        "wet_corners": int(np.count_nonzero(unequal & corner)),
        "interior": int(np.count_nonzero(unequal & ~ring)),
        "native_j_rows": {
            str(j + 3): int(count) for j, count in enumerate(by_j) if count
        },
    }


def _developed_identity(name: str, got, reference, active) -> dict:
    """Bitwise row for one directly aligned developed-state operand."""
    got = np.asarray(got, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require(got.shape == reference.shape == active.shape,
            f"developed row {name} shape mismatch: "
            f"{got.shape}, {reference.shape}, {active.shape}")
    unequal = (got.view(np.uint64) != reference.view(np.uint64)) & active
    delta = got[active] - reference[active]
    return {
        "name": name,
        "cells": int(np.count_nonzero(active)),
        "cells_unequal": int(np.count_nonzero(unequal)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "bit_exact": not bool(np.any(unequal)),
    }


def run_developed(
    developed_root: Path,
    *,
    expect_commit: str,
    daily_root: Path,
    daily_audit: Path,
    plant: bool,
) -> dict:
    """Walk day-180 LDF operands from the production-jitted step."""
    import argparse

    import jax
    import jax.numpy as jnp

    import nemo_testcase_l2_gyre_round82_btstep_walk as round82
    import nemo_testcase_l2_gyre_round83_slow_forcing_walk as round83
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
        nemo_lateral_viscosity_coefficients,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_qco_live_vorticity_e3f_cgrid,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "developed LDF walk requires fp64/libm")
    require(bool(jax.config.jax_enable_x64), "developed LDF walk requires x64")
    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            f"developed LDF worktree stamp changed: {stamp}")

    admission_path = developed_root / "round148_developed_ldf_admission.json"
    admission = json.loads(admission_path.read_text())
    record_path = developed_root / DEVELOPED_RECORD
    require(admission.get("status") == "PASS"
            and admission.get("record_sha256") == sha256(record_path),
            "Round-148 direct LDF record is not admitted")
    record = read_developed_ldf_bytes(record_path.read_bytes())["fields"]

    args = argparse.Namespace(daily_root=daily_root, daily_audit=daily_audit)
    card, state, freshwater, surface, payload, entry = round82._developed_inputs(args)
    eta_after = jnp.asarray(payload["ssha"])

    def trace(one_state):
        return round83._round117_live_trace(
            card, one_state, freshwater, surface, "production-jit",
            eta_after_override=eta_after)

    exposed = trace(state)
    parts = exposed.operator_operands[0]
    require(parts is not None, "production step omitted stage-1 operands")
    grid = card.recipe.grid
    h_k = np.asarray(parts["operand_h_k"], dtype=np.float64)
    u = np.asarray(parts["operand_ldf_velocity_u"], dtype=np.float64)
    v = np.asarray(parts["operand_ldf_velocity_v"], dtype=np.float64)
    u_live_mask, v_live_mask = compute_face_masks_3d(
        card.recipe.z_coord.is_active, grid)
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    live_u, live_v = _nemo_ws_qco_stage_faces(
        state.eta.data, h_ref, u_live_mask, v_live_mask, grid)[:2]
    model = {
        "u": u,
        "v": v,
        "e3t": h_k,
        "e3u": np.asarray(live_u),
        "e3v": np.asarray(live_v),
        "e3f": np.asarray(nemo_qco_live_vorticity_e3f_cgrid(
            state.eta.data, card.recipe.z_coord, state.eta.data.dtype,
            grid=grid)),
        "e2u": np.asarray(grid.dy_u),
        "e1v": np.asarray(grid.dx_v),
        "e2v": np.asarray(grid.dy_v),
        "e1u": np.asarray(grid.dx_u),
        "r1_e1e2t": np.asarray(1.0 / grid.area_T),
        "r1_e1e2f": np.asarray(1.0 / grid.area_q),
        "r1_e1u": np.asarray(1.0 / grid.dx_u),
        "r1_e2v": np.asarray(1.0 / grid.dy_v),
        "r1_e2u": np.asarray(1.0 / grid.dy_u),
        "r1_e1v": np.asarray(1.0 / grid.dx_v),
    }

    def t3(name):
        return record[name][2:-2, 2:-2, :30]

    def u3(name):
        return record[name][2:-2, 1:-2, :30]

    def v3(name):
        return record[name][1:-2, 2:-2, :30]

    def f3(name):
        return record[name][1:-2, 1:-2, :30]

    def t2(name):
        return record[name][2:-2, 2:-2]

    def u2(name):
        return record[name][2:-2, 1:-2]

    def v2(name):
        return record[name][1:-2, 2:-2]

    def f2(name):
        return record[name][1:-2, 1:-2]

    nemo = {
        "u": u3("u_kbb"), "v": v3("v_kbb"),
        "tmask": t3("tmask"), "umask": u3("umask"),
        "vmask": v3("vmask"), "fmask": f3("fmask"),
        "ahmt": t3("ahmt"), "ahmf": f3("ahmf"),
        "e3t": t3("e3t_kbb"), "e3u": u3("e3u_kbb"),
        "e3v": v3("e3v_kbb"), "e3f": f3("e3f_live"),
        "e3u_kmm": u3("e3u_kmm"), "e3v_kmm": v3("e3v_kmm"),
        "pre_u": u3("pre_u"), "pre_v": v3("pre_v"),
        "post_u": u3("post_u"), "post_v": v3("post_v"),
        "e2u": u2("e2u"), "e1v": v2("e1v"),
        "e2v": v2("e2v"), "e1u": u2("e1u"),
        "r1_e1e2t": t2("r1_e1e2t"),
        "r1_e1e2f": f2("r1_e1e2f"),
        "r1_e1u": u2("r1_e1u"), "r1_e2v": v2("r1_e2v"),
        "r1_e2u": u2("r1_e2u"), "r1_e1v": v2("r1_e1v"),
        "zcur": f3("zcur"), "zdiv": t3("zdiv"),
    }
    for name in ("u", "v", "e3t", "e3u", "e3v", "e3f"):
        require(model[name].shape == nemo[name].shape,
                f"developed {name} mapping changed: "
                f"{model[name].shape} != {nemo[name].shape}")

    half_uv = (card.recipe.model_config.lateral_viscosity.A_h
               / (grid.radius * grid.dlon))
    ahmt_1d, ahmf_1d = nemo_lateral_viscosity_coefficients(grid, half_uv)
    model["ahmt"] = (np.asarray(ahmt_1d)[:, None, None]
                     * (nemo["tmask"] != 0.0))
    model["ahmf"] = (np.asarray(ahmf_1d)[:, None, None]
                     * (nemo["fmask"] != 0.0))

    masks = {
        "t": nemo["tmask"] != 0.0,
        "u": nemo["umask"] != 0.0,
        "v": nemo["vmask"] != 0.0,
        "f": nemo["fmask"] != 0.0,
    }
    rows = []

    def add(name, got, ref, active):
        rows.append(_developed_identity(name, got, ref, active))

    # Compiled line 159, then its velocity bracket (160--161).
    add("zcur.ahmf", model["ahmf"], nemo["ahmf"], masks["f"])
    add("zcur.e3f_live", model["e3f"], nemo["e3f"], masks["f"])
    add("zcur.r1_e1e2f", model["r1_e1e2f"], nemo["r1_e1e2f"],
        masks["f"][..., 0])
    add("zcur.e2v", model["e2v"], nemo["e2v"],
        masks["v"][..., 0])
    add("zcur.v", model["v"], nemo["v"], masks["v"])
    add("zcur.e1u", model["e1u"], nemo["e1u"],
        masks["u"][..., 0])
    add("zcur.u", model["u"], nemo["u"], masks["u"])
    # Compiled lines 163--165.
    add("zdiv.ahmt", model["ahmt"], nemo["ahmt"], masks["t"])
    add("zdiv.r1_e1e2t", model["r1_e1e2t"], nemo["r1_e1e2t"],
        masks["t"][..., 0])
    add("zdiv.e3t_kbb", model["e3t"], nemo["e3t"], masks["t"])
    add("zdiv.e2u", model["e2u"], nemo["e2u"],
        masks["u"][..., 0])
    add("zdiv.e3u_kbb", model["e3u"], nemo["e3u"], masks["u"])
    add("zdiv.e1v", model["e1v"], nemo["e1v"],
        masks["v"][..., 0])
    add("zdiv.e3v_kbb", model["e3v"], nemo["e3v"], masks["v"])

    literal = literal_ldf_numpy(
        nemo["u"], nemo["v"], e3t_kbb=nemo["e3t"],
        e3u_kbb=nemo["e3u"], e3v_kbb=nemo["e3v"],
        e3f_live=nemo["e3f"], e3u_kmm=nemo["e3u_kmm"],
        e3v_kmm=nemo["e3v_kmm"], e2u=nemo["e2u"],
        e1v=nemo["e1v"], e2v=nemo["e2v"], e1u=nemo["e1u"],
        r1_e1e2t=nemo["r1_e1e2t"], r1_e1e2f=nemo["r1_e1e2f"],
        r1_e1u=nemo["r1_e1u"], r1_e2v=nemo["r1_e2v"],
        r1_e2u=nemo["r1_e2u"], r1_e1v=nemo["r1_e1v"],
        ahmt=nemo["ahmt"], ahmf=nemo["ahmf"],
        tmask=nemo["tmask"], umask=nemo["umask"],
        vmask=nemo["vmask"], fmask=nemo["fmask"])
    add("nemo_literal.zcur", literal["zcur"], nemo["zcur"], masks["f"])
    add("nemo_literal.zdiv", literal["zdiv"], nemo["zdiv"], masks["t"])
    add("nemo_literal.post_u", _op("add", nemo["pre_u"], literal["visc_u"]),
        nemo["post_u"], masks["u"])
    add("nemo_literal.post_v", _op("add", nemo["pre_v"], literal["visc_v"]),
        nemo["post_v"], masks["v"])

    production_u = np.asarray(parts["ldf_u"].data, dtype=np.float64)
    production_v = np.asarray(parts["ldf_v"].data, dtype=np.float64)
    add("production_step.ldf_term_u", production_u, literal["visc_u"],
        masks["u"])
    add("production_step.ldf_term_v", production_v, literal["visc_v"],
        masks["v"])
    add("production_step.after_ldf_u", parts["after_ldf_u"].data,
        nemo["post_u"], masks["u"])
    add("production_step.after_ldf_v", parts["after_ldf_v"].data,
        nemo["post_v"], masks["v"])

    first = next((row for row in rows if not row["bit_exact"]), None)
    require(first is not None, "developed LDF walk unexpectedly has no debt")
    plant_row = None
    if plant:
        eta = np.asarray(state.eta.data, dtype=np.float64).copy()
        active = h_k[..., 0] > 0.0
        location = tuple(int(value) for value in np.argwhere(active)[0])
        eta[location] = eta[location] + np.float64(65536.0) * np.spacing(
            eta[location] if eta[location] != 0.0 else np.float64(1.0))
        planted_state = state._replace(
            eta=state.eta.replace(data=jnp.asarray(eta)))
        planted_parts = trace(planted_state).operator_operands[0]
        planted_u = np.asarray(planted_parts["ldf_u"].data, dtype=np.float64)
        changed = planted_u.view(np.uint64) != production_u.view(np.uint64)
        plant_row = {
            "eta_location": list(location),
            "ldf_u_cells_moved": int(np.count_nonzero(changed)),
            "ldf_u_max_abs": float(np.max(np.abs(planted_u - production_u))),
        }
        require(plant_row["ldf_u_cells_moved"] > 0,
                "production entry-thickness plant did not move LDF")

    return {
        "format": "nemo-testcase-l2-gyre-round149-developed-ldf-walk-v1",
        "status": "PLANT-FIRED" if plant else "MEASURED",
        "worktree": stamp,
        "execution_regime": "production-jit-cpu-fp64-x64-libm",
        "entry": entry,
        "record_sha256": sha256(record_path),
        "admission_sha256": sha256(admission_path),
        "rows": rows,
        "first_nonbit": first,
        "plant": plant_row,
    }


def _scale_variants(jnp, a, b, c):
    """Closed association ladder for the compiled ``a*b/c`` statement."""
    return {
        "ab_div_c": (a * b) / c,
        "a_mul_bdivc": a * (b / c),
        "b_mul_adivc": b * (a / c),
        "ab_mul_recipc": (a * b) * (1.0 / c),
        "a_div_cdivb": a / (c / b),
        "b_div_cdiva": b / (c / a),
    }


def run(root: Path, *, expect_commit: str, plant: bool) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.grids.operators_latlon_cgrid import compute_vertex_mask
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_ldf_lap_viscosity_e3_cgrid,
        nemo_lateral_viscosity_coefficients,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.vertical import nemo_qco_live_vorticity_e3f_cgrid

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "round50 requires x64")
    stamp = worktree_stamp()
    expected = ("0" * 40) if plant == "stamp" else expect_commit.lower()
    require(stamp["clean"] and stamp["commit"].lower() == expected,
            f"commit stamp mismatch: {stamp} != {expected}")

    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    ocean = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    grid = card.recipe.grid
    half_uv = cfg.lateral_viscosity.A_h / (grid.radius * grid.dlon)
    ahmt_1d, ahmf_1d = nemo_lateral_viscosity_coefficients(grid, half_uv)
    rows, spatial, intermediate_rows, association_rows, hashes = [], [], [], [], {}
    for stage in (1, 3):
        path = root / f"oracle_momstage_kt00000002_s{stage}.bin"
        record = read_stage(path)["arrays"]
        hashes[path.name] = sha256(path)
        u, v = _u_face(record["u_Kbb"]), _v_face(record["v_Kbb"])
        ju, jv = _u_face(record["u_Kmm"]), _v_face(record["v_Kmm"])
        umask, vmask = _u_face(record["umask"]), _v_face(record["vmask"])
        tmask = _owned3(record["tmask"])
        fmask = np.asarray(jax.vmap(
            lambda m: compute_vertex_mask(m, grid=grid), in_axes=-1, out_axes=-1
        )(jnp.asarray(tmask)))
        e3f = np.asarray(nemo_qco_live_vorticity_e3f_cgrid(
            jnp.asarray(_owned2(record["ssh_Kmm"])), card.recipe.z_coord,
            jnp.float64, nn_e3f_typ=0))
        operands = {
            "u": u,
            "v": v,
            "e3t_kbb": _owned3(record["e3t_Kbb"]),
            "e3u_kbb": _u_face(record["e3u_Kbb"]),
            "e3v_kbb": _v_face(record["e3v_Kbb"], fill=1.0),
            "e3f_live": e3f,
            "e3u_kmm": _u_face(record["e3u_Kmm"]),
            "e3v_kmm": _v_face(record["e3v_Kmm"], fill=1.0),
            "e2u": np.asarray(grid.dy_u),
            "e1v": np.asarray(grid.dx_v),
            "e2v": np.asarray(grid.dy_v),
            "e1u": np.asarray(grid.dx_u),
            "r1_e1e2t": np.asarray(1.0 / grid.area_T),
            "r1_e1e2f": np.asarray(1.0 / grid.area_q),
            "r1_e1u": np.asarray(1.0 / grid.dx_u),
            "r1_e2v": np.asarray(1.0 / grid.dy_v),
            "r1_e2u": np.asarray(1.0 / grid.dy_u),
            "r1_e1v": np.asarray(1.0 / grid.dx_v),
            "ahmt": np.asarray(ahmt_1d)[:, None, None] * tmask,
            "ahmf": np.asarray(ahmf_1d)[:, None, None] * fmask,
            "tmask": tmask,
            "umask": umask,
            "vmask": vmask,
            "fmask": fmask,
            "plant": plant,
        }
        literal = literal_ldf_numpy(**operands)

        zdiv_variants = jax.jit(lambda a0, b0, c0: _scale_variants(
            jnp, a0, b0, c0))(
                jnp.asarray(operands["ahmt"]),
                jnp.asarray(operands["r1_e1e2t"])[..., None],
                jnp.where(jnp.asarray(operands["e3t_kbb"]) > 0,
                          jnp.asarray(operands["e3t_kbb"]), 1.0))
        for name, value in zdiv_variants.items():
            association_rows.append(_score(
                f"kt2.s{stage}.zdiv_scale.{name}", np.asarray(value),
                literal["zdiv_scale"], tmask > 0.5))
        curl_u_variants = jax.jit(lambda a0, b0, c0: _scale_variants(
            jnp, a0, b0, c0))(
                jnp.asarray(-np.subtract(literal["zcur"][1:], literal["zcur"][:-1])),
                jnp.asarray(operands["r1_e2u"])[..., None],
                jnp.where(jnp.asarray(operands["e3u_kmm"]) > 0,
                          jnp.asarray(operands["e3u_kmm"]), 1.0))
        for name, value in curl_u_variants.items():
            association_rows.append(_score(
                f"kt2.s{stage}.curl_u.{name}", np.asarray(value),
                literal["curl_u"], umask > 0.5))

        thickness_values = tuple(jnp.asarray(operands[name]) for name in (
            "e3t_kbb", "e3u_kbb", "e3v_kbb", "e3f_live",
            "e3u_kmm", "e3v_kmm"))
        reciprocal_values = tuple(jnp.asarray(operands[name]) for name in (
            "r1_e1e2t", "r1_e1e2f", "r1_e1u", "r1_e2v",
            "r1_e2u", "r1_e1v"))
        direct_u, direct_v, model_intermediates = jax.jit(
            lambda uu, vv, thickness_args, reciprocal_args:
            nemo_ldf_lap_viscosity_e3_cgrid(
                uu, vv, grid, ahmt_1d, ahmf_1d, jnp.asarray(operands["e3t_kbb"]),
                mask=jnp.asarray(tmask), u_mask=jnp.asarray(umask),
                v_mask=jnp.asarray(vmask), vertex_mask=jnp.asarray(fmask),
                thickness_operands=thickness_args,
                metric_reciprocal_operands=reciprocal_args,
                return_intermediates=True,
            )
        )(jnp.asarray(u), jnp.asarray(v), thickness_values, reciprocal_values)
        model_intermediates = {
            name: np.asarray(value) for name, value in model_intermediates.items()
        }

        state = card.recipe.initial_state._replace(
            u=card.recipe.initial_state.u.replace(data=jnp.asarray(ju)),
            v=card.recipe.initial_state.v.replace(data=jnp.asarray(jv)),
            T=card.recipe.initial_state.T.replace(data=jnp.asarray(_owned3(record["T_Kmm"]))),
            S=card.recipe.initial_state.S.replace(data=jnp.asarray(_owned3(record["S_Kmm"]))),
            eta=card.recipe.initial_state.eta.replace(data=jnp.asarray(_owned2(record["ssh_Kmm"]))),
        )
        _, surface = _surface_forcings(card, state, 2)
        _, diagnostics, _ = jax.jit(lambda s, thickness_args, reciprocal_args:
            ocean.tendencies(
            s, surface, dt=card.dt_s, momentum_only=True,
            skip_lateral_viscosity=False,
            ldf_state=(s.T.data, s.S.data, jnp.asarray(u), jnp.asarray(v)),
            momentum_flux_face_thickness=(jnp.asarray(operands["e3u_kmm"]),
                                              jnp.asarray(operands["e3v_kmm"])),
            ldf_thickness_operands=thickness_args,
            ldf_metric_reciprocal_operands=reciprocal_args,
            zad_continuity_dt=np.float64(1.0 / record["r1_Dt"]),
            nemo_operator_association=True,
            return_nemo_operator_components=True,
        ))(state, thickness_values, reciprocal_values)
        model_u = sum(np.asarray(getattr(diagnostics, f"{name}_u").data)
                      for name in ("Ah_lap", "Bh_bilap", "Cs_smag", "Cl_leith"))
        model_v = sum(np.asarray(getattr(diagnostics, f"{name}_v").data)
                      for name in ("Ah_lap", "Bh_bilap", "Cs_smag", "Cl_leith"))
        for face, literal_term, model_term, wet in (
            ("u", literal["visc_u"][:, 1:], model_u[:, 1:], _owned3(record["umask"]) > 0.5),
            ("v", literal["visc_v"][1:], model_v[1:], _owned3(record["vmask"]) > 0.5),
        ):
            prior_name = "after_hpg" if stage == 1 else "after_adv"
            prior = _owned3(record[f"{prior_name}_{face}"])
            ref = _owned3(record[f"after_ldf_{face}"])
            post_literal = _op("add", prior, literal_term)
            post_model = _op("add", prior, model_term)
            rows.append(_score(f"kt2.s{stage}.numpy_literal.post_ldf.{face}", post_literal, ref, wet))
            rows.append(_score(f"kt2.s{stage}.model_path.post_ldf.{face}", post_model, ref, wet))
            rows.append(_score(f"kt2.s{stage}.numpy_vs_model.term.{face}", literal_term, model_term, wet))
            spatial.append(_spatial(f"kt2.s{stage}.{face}", post_model, ref, wet))

        # Keep the isolated helper as a diagnostic, but admission is owned by
        # the production ``ocean.tendencies`` route above (Rule 10).  In
        # particular, XLA may fuse the caller's mask/slope multiplications into
        # the helper and legitimately give the isolated expression a different
        # lowering; record that distinction instead of substituting the helper
        # for the production path.
        rows.append(_score(
            f"kt2.s{stage}.direct_vs_model.term.u", np.asarray(direct_u),
            model_u, umask > 0.5))
        rows.append(_score(
            f"kt2.s{stage}.direct_vs_model.term.v", np.asarray(direct_v),
            model_v, vmask > 0.5))

        intermediate_masks = {
            "div_u_product": umask > 0.5,
            "div_v_product": vmask > 0.5,
            "div_bracket": tmask > 0.5,
            "zdiv_scale": tmask > 0.5,
            "zdiv": tmask > 0.5,
            "curl_bracket": fmask > 0.5,
            "zcur_scale": fmask > 0.5,
            "zcur": fmask > 0.5,
            "grad_div_u": umask > 0.5,
            "grad_div_v": vmask > 0.5,
            "curl_u": umask > 0.5,
            "curl_v": vmask > 0.5,
            "visc_u": umask > 0.5,
            "visc_v": vmask > 0.5,
        }
        for name, model_value in model_intermediates.items():
            mask = intermediate_masks[name]
            intermediate_rows.append(_score(
                f"kt2.s{stage}.{name}", literal[name], model_value, mask))

    calibration = [r for r in rows if ".numpy_literal.post_ldf." in r["name"]]
    model_rows = [r for r in rows if ".model_path.post_ldf." in r["name"]]
    require(all(r["n_unequal"] == 0 for r in calibration),
            f"literal replay did not calibrate: {calibration}")
    exact = all(r["n_unequal"] == 0 for r in model_rows)
    first_by_stage = {}
    source_order = tuple(intermediate_masks)
    for stage in (1, 3):
        stage_rows = {r["name"].split(".")[-1]: r for r in intermediate_rows
                      if f".s{stage}." in r["name"]}
        first_by_stage[str(stage)] = next(
            (name for name in source_order if stage_rows[name]["n_unequal"]), None)
    if plant:
        require(any(r["n_unequal"] for r in calibration),
                "nextafter LDF plant did not move a scored row")
    return {
        "format": "nemo-testcase-l2-gyre-round50-ldf-association-v1",
        "worktree": stamp,
        "compiled_source": SOURCE,
        "resolved": {
            "dtype": str(np.asarray(grid.area_T).dtype),
            "nn_ahm_ijk_t": 0,
            "rn_Uv": 2.0,
            "rn_Lv": 100000.0,
            "ahmt_unique_wet": sorted(map(float, np.unique(
                np.asarray(ahmt_1d)[:, None, None] * (_owned3(read_stage(
                    root / "oracle_momstage_kt00000002_s1.bin")["arrays"]["tmask"]) > 0)))),
        },
        "record_sha256": hashes,
        "rows": rows,
        "spatial": spatial,
        "intermediate_rows": intermediate_rows,
        "association_rows": association_rows,
        "first_differing_intermediate": first_by_stage,
        "plant": plant,
        "status": "PLANT_FIRED" if plant else ("CONFIRMED" if exact else "DEBT"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--admit-developed", action="store_true")
    parser.add_argument("--walk-developed", action="store_true")
    parser.add_argument("--developed-record", type=Path)
    parser.add_argument("--developed-family-record", type=Path)
    parser.add_argument("--developed-family-baseline", type=Path)
    parser.add_argument("--developed-parent-record", type=Path)
    parser.add_argument("--developed-parent-baseline", type=Path)
    parser.add_argument("--developed-root", type=Path)
    parser.add_argument("--developed-baseline-root", type=Path)
    parser.add_argument("--developed-stamp", type=Path)
    parser.add_argument("--daily-root", type=Path)
    parser.add_argument("--daily-audit", type=Path)
    parser.add_argument(
        "--developed-plant",
        choices=("header", "truncation", "missing-field", "zcur-ulp",
                 "post-ulp", "parent-wet-ulp", "parent-dry-ulp",
                 "restart-byte"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.walk_developed:
        require(not args.admit_developed,
                "developed walk and admission are mutually exclusive")
        require(args.developed_root is not None
                and args.daily_root is not None
                and args.daily_audit is not None,
                "developed walk requires developed/daily roots and audit")
        report = run_developed(
            args.developed_root, expect_commit=args.expect_commit,
            daily_root=args.daily_root, daily_audit=args.daily_audit,
            plant=args.plant)
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        return 1 if args.plant else 0
    if args.admit_developed:
        require(not args.plant, "use --developed-plant in developed mode")
        require(args.developed_record is not None
                and args.developed_family_record is not None
                and args.developed_stamp is not None,
                "developed admission requires record, family record, and stamp")
        try:
            report = admit_developed_ldf_record(
                args.developed_record,
                args.developed_family_record,
                args.developed_stamp,
                expect_commit=args.expect_commit,
                plant=args.developed_plant,
                parent_path=args.developed_parent_record,
                parent_baseline_path=args.developed_parent_baseline,
                family_baseline_path=args.developed_family_baseline,
                inherited_root=args.developed_root,
                inherited_baseline_root=args.developed_baseline_root,
            )
        except Exception as error:
            if args.developed_plant:
                print(f"ROUND148 DEVELOPED LDF {args.developed_plant.upper()} "
                      f"STATUS PLANT-FIRED: {error}")
                return 1
            raise
        if args.developed_plant:
            print(f"REFUSE: developed LDF {args.developed_plant} plant stayed green")
            return 2
        text = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        return 0
    report = run(args.root, expect_commit=args.expect_commit, plant=args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 1 if args.plant or report["status"] != "CONFIRMED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
